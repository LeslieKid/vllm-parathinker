import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer and model from Hugging Face
model_id = "Leslie04/ParaThinker-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
)