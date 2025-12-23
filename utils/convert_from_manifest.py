#!/usr/bin/env python3
import argparse
import json
import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

try:
    import orjson  # type: ignore
except Exception:
    orjson = None


def convert_to_jsonl(
    data_dir,
    output_file,
    max_svg_len=18000,
    workers=None,
    num_shards=1,
    shard_id=0,
    manifest_file="manifest.jsonl",
):
    data_dir = Path(data_dir)
    manifest_path = data_dir / manifest_file

    if not manifest_path.exists():
        raise ValueError(f"Manifest file not found: {manifest_path}")

    if workers is None or workers <= 0:
        # Reasonable default: 2x CPU cores but capped
        workers = min(32, (os.cpu_count() or 8) * 2)

    if num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= shard_id < num_shards):
        raise ValueError("--shard-id must be in [0, num_shards)")

    print(f"Reading manifest: {manifest_path}")
    print(f"  Data dir  : {data_dir}")
    print(f"  Workers   : {workers}")
    print(f"  Sharding  : shard {shard_id}/{num_shards}")

    # Read all entries from manifest
    all_entries = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    all_entries.append(entry)
                except json.JSONDecodeError:
                    continue

    print(f"\nManifest Statistics:")
    print(f"  Total entries: {len(all_entries)}")

    # Shard the entries (length filtering happens during processing based on actual SVG file)
    if num_shards > 1:
        selected_entries = [
            entry
            for entry in all_entries
            if (zlib.crc32(entry["id"].encode("utf-8")) % num_shards) == shard_id
        ]
    else:
        selected_entries = all_entries

    print(f"\nShard contains {len(selected_entries)} entries to convert.")

    # Choose JSON dumper
    if orjson is not None:
        def dumps(obj):
            return orjson.dumps(obj).decode("utf-8")
        print("Using orjson for serialization.")
    else:
        def dumps(obj):
            return json.dumps(obj, ensure_ascii=False)
        print("Using stdlib json for serialization (install orjson for more speed).")

    # Worker function run in threads
    def process_entry(entry):
        try:
            # Get description directly from manifest
            description = entry.get("description", "").strip()
            if not description:
                return None

            # Get SVG path from manifest
            paths = entry.get("paths", {})
            svg_path = paths.get("svg")

            if not svg_path:
                return None

            # Construct full path and read SVG file
            svg_file = data_dir / svg_path

            if not svg_file.exists():
                return None

            svg_content = svg_file.read_text(encoding="utf-8", errors="ignore").strip()

            if not svg_content:
                return None

            # Filter by actual optimized SVG length
            if len(svg_content) > max_svg_len:
                return None

            # Build output example (only fields needed for training)
            example = {
                "id": entry.get("id"),
                "description": description,
            }

            return dumps(example)

        except Exception:
            return None  # skip on any error

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    skipped = 0

    print(f"\nWriting JSONL to: {output_path}")
    t0 = time.perf_counter()

    # Parallel I/O via threads, streaming writes (no big list in memory)
    with output_path.open("w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for line in tqdm(
                ex.map(process_entry, selected_entries),
                total=len(selected_entries),
                desc="Converting",
            ):
                if line is None:
                    skipped += 1
                    continue
                out_f.write(line + "\n")
                n_written += 1

    elapsed = time.perf_counter() - t0

    print(f"\n✅ Done. Wrote {n_written} examples to {output_path}")
    print(f"   Skipped {skipped} examples (errors / empty / too long)")
    print(f"   Elapsed time: {elapsed:.1f} s")
    if elapsed > 0:
        print(f"   Approx throughput: {n_written / elapsed:.1f} examples/s")
    print(f"   Final file size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    return n_written


def main():
    default_input = "/mnt/haonan-central3a-disk/haonan/data/data0_train_fill"
    parser = argparse.ArgumentParser(
        description="Convert SVG training data to JSONL format using manifest.jsonl (parallel, sharded)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=default_input,
        help="Input directory containing manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSONL file. If using sharding, you probably want something like "
            "training_data_manifest.shard{ID}.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifest.jsonl",
        help="Name of manifest file (default: manifest.jsonl)",
    )
    parser.add_argument(
        "--max-svg-len",
        type=int,
        default=18000,
        help="Max allowed SVG length in characters",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel I/O workers (threads). "
             "Default: 2x CPU cores, capped at 32.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="How many shards to split the dataset into.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Which shard index this process should handle (0-based).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if args.output is None:
        # If sharded, encode shard index in filename
        if args.num_shards > 1:
            output_file = (
                input_dir
                / f"training_data_manifest.shard{args.shard_id:03d}.jsonl"
            )
        else:
            output_file = input_dir / "training_data_manifest.jsonl"
    else:
        output_file = Path(args.output)

    convert_to_jsonl(
        input_dir,
        output_file,
        max_svg_len=args.max_svg_len,
        workers=args.workers,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        manifest_file=args.manifest,
    )


if __name__ == "__main__":
    main()
