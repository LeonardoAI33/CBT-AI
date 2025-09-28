from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/Phi-3.5-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

pipe = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user",   "content": "Quem foi Andy Yen?"}
]

response = pipe(messages, max_new_tokens=512, do_sample=True, top_p=0.7, temperature=0.1)
