import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run inference with DeepSeek-V3")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    args = parser.parse_args()

    # model_name = "deepseek-ai/DeepSeek-V3"
    # model_name = "deepseek_model_local"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {args.prompt}\nResponse: {response}")

if __name__ == "__main__":
    main()

