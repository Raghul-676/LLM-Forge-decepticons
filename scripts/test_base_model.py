import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B-Base"

print("=" * 70)
print("Loading:", MODEL_NAME)
print("GPU:", torch.cuda.get_device_name(0))
print("=" * 70)

# -------------------------------------------------
# 1. 4-bit quantization configuration
# -------------------------------------------------

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# -------------------------------------------------
# 2. Load tokenizer
# -------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded.")

# -------------------------------------------------
# 3. Load model
# -------------------------------------------------

print("\nLoading model in 4-bit mode...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.eval()

print("Model loaded successfully.")

# -------------------------------------------------
# 4. GPU memory information
# -------------------------------------------------

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"\nGPU memory allocated: {allocated:.2f} GB")
    print(f"GPU memory reserved : {reserved:.2f} GB")

# -------------------------------------------------
# 5. Generation function
# -------------------------------------------------

def generate(prompt, max_new_tokens=100):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Put input tensors on the model's first device
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    return generated


# -------------------------------------------------
# 6. Test prompts
# -------------------------------------------------

prompts = [

    # General language
    """
The capital of France is
""",

    # Simple reasoning/language
    """
Question: Why do people need laws?
Answer:
""",

    # General legal question
    """
Question: What is a constitution?
Answer:
""",

    # Indian law related
    """
Question: What is Article 21 of the Constitution of India?
Answer:
""",

    # Legal scenario
    """
Question:
A person borrowed money and issued a cheque.
The cheque was returned by the bank because of insufficient funds.
What legal issues may arise?

Answer:
""",

]

# -------------------------------------------------
# 7. Generate and save outputs
# -------------------------------------------------

os.makedirs("outputs", exist_ok=True)

output_file = "outputs/base_model_results.txt"

with open(output_file, "w", encoding="utf-8") as file:

    for i, prompt in enumerate(prompts, start=1):

        print("\n" + "=" * 70)
        print(f"TEST {i}")
        print("=" * 70)

        print("\nPROMPT:")
        print(prompt.strip())

        result = generate(prompt)

        print("\nMODEL OUTPUT:")
        print(result)

        file.write("=" * 70 + "\n")
        file.write(f"TEST {i}\n")
        file.write("=" * 70 + "\n\n")

        file.write("PROMPT:\n")
        file.write(prompt.strip())
        file.write("\n\n")

        file.write("MODEL OUTPUT:\n")
        file.write(result)
        file.write("\n\n")


print("\n" + "=" * 70)
print("Finished.")
print("Results saved to:")
print(output_file)
print("=" * 70)