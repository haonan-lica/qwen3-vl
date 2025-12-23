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
            # Get ID and description from manifest
            entry_id = entry.get("id")
            if not entry_id:
                return None

            description = entry.get("description", "").strip()
            if not description:
                return None

            # Construct SVG path from ID
            # Try to get from manifest first (if it has paths field), otherwise construct it
            paths = entry.get("paths", {})
            svg_path = paths.get("svg") if paths else None

            if not svg_path:
                # Construct path directly: svg_optimized/{id}.svg
                svg_file = data_dir / "svg_optimized" / f"{entry_id}.svg"
            else:
                svg_file = data_dir / svg_path

            if not svg_file.exists():
                return None

            svg_content = svg_file.read_text(encoding="utf-8", errors="ignore").strip()

            if not svg_content:
                return None

            # Get SVG length for training metadata
            svg_length = len(svg_content)

            # Filter by actual optimized SVG length
            if svg_length > max_svg_len:
                return None

            # Construct image path directly using ID (no scanning needed!)
            img_file = data_dir / "rendered" / f"{entry_id}.png"

            # Build output example with image, messages, and length for training
            example = {
                "id": entry_id,
                "description": description,
                "svg": svg_content,
                "target_svg_length": svg_length,
                "image": str(img_file),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Generate SVG code based on the following description along with the image file \n "
                                    f"Description: {description} \n "
                                ),
                            },
                            {
                                "type": "image",
                                "path": str(img_file),
                            },
                        ],
                    },
                    {"role": "assistant", "content": svg_content},
                ],
            }
            return dumps(example)

        except Exception:
            return None  # skip on any error

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    skipped = 0
    total_svg_length = 0
    min_svg_length = float('inf')
    max_svg_length = 0

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

                # Track SVG length statistics
                try:
                    entry_data = json.loads(line)
                    svg_len = entry_data.get("target_svg_length", 0)
                    if svg_len > 0:
                        total_svg_length += svg_len
                        min_svg_length = min(min_svg_length, svg_len)
                        max_svg_length = max(max_svg_length, svg_len)
                except:
                    pass

    elapsed = time.perf_counter() - t0

    print(f"\n✅ Done. Wrote {n_written} examples to {output_path}")
    print(f"   Skipped {skipped} examples (errors / empty / too long)")
    print(f"   Elapsed time: {elapsed:.1f} s")
    if elapsed > 0:
        print(f"   Approx throughput: {n_written / elapsed:.1f} examples/s")
    print(f"   Final file size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    # SVG length statistics
    if n_written > 0:
        avg_svg_length = total_svg_length / n_written
        print(f"\nSVG Length Statistics:")
        print(f"   Average: {avg_svg_length:.0f} characters")
        print(f"   Min: {min_svg_length} characters")
        print(f"   Max: {max_svg_length} characters")
        print(f"   Total SVG data: {total_svg_length / (1024 * 1024):.2f} MB")

    return n_written


def main():
    default_input = "/mnt/haonan-us-1b/data/data0_train"
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
            "training_data_optimized_image.shard{ID}.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="training_data_manifest.jsonl",
        help="Name of manifest file (default: manifest.jsonl)",
    )
    parser.add_argument(
        "--max-svg-len",
        type=int,
        default=8192,
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
                / f"training_data_optimized_image.shard{args.shard_id:03d}.jsonl"
            )
        else:
            output_file = input_dir / "training_data_optimized_image.jsonl"
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
