import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16
).to(device)

# Function to generate a response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the bot's response after "Bot:" to avoid repeating the prompt
    if "Bot:" in response:
        response = response.split("Bot:")[1].strip()
    return response

# Chat loop
print("Hello! I'm DeepSeek-R1. Type 'exit' or press Ctrl+C to quit.")

while True:
    try:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        prompt = f"User: {user_input}\nBot:"
        response = generate_response(prompt)
        print(f"\nBot: {response}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break

