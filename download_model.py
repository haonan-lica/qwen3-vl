"""Pre-download the Qwen model to cache before training."""

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"

print(f"Downloading {MODEL_NAME}...")
print("This may take 10-15 minutes for a 30B model...")

# Download tokenizer
print("\n1. Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("✓ Tokenizer downloaded")

# Download model
print("\n2. Downloading model weights...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
print("✓ Model downloaded successfully!")

# print(f"\nModel cached at: ~/.cache/huggingface/hub/")
print("You can now run train.py")
