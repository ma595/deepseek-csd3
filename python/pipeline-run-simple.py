# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

messages = [
    {"role": "user", "content": "How can you help me?"},
    {"role": "user", "content": "What is the capital of France"},
]

device = "cuda" if torch.cuda.is_available() else "cpu"
# pipe = pipeline("text-generation", model="deepseek_model_local", device=device)
pipe = pipeline(
    "text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device=device
)

response = pipe(
    messages,
    max_new_tokens=20000,
    temperature=0.7,
)
# print(pipe(messages))


print(response[0])
