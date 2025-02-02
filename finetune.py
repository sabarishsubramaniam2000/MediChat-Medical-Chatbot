import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
from huggingface_hub import login
import os

# Authenticate with Hugging Face
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Load LLaMA-2 Model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Load & Tokenize Dataset
dataset = load_dataset("json", data_files={"train": "custom_data.json", "test": "custom_data_test.json"})

def tokenize_function(examples):
    return tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Apply LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./finetuned_llama",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()

# Save Fine-Tuned Model
model.save_pretrained("./finetuned_llama")
tokenizer.save_pretrained("./finetuned_llama")
print("Fine-tuning complete! Model saved at ./finetuned_llama")
