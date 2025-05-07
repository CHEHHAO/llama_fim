import yaml, json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments
from llama_factory import LLaMAFactory, LoRAConfig

cfg = yaml.safe_load(open("config.yaml"))
base_model = "deepseek-ai/deepseek-coder-1.3b-base"

print("Loading base model…")
model = AutoModelForCausalLM.from_pretrained(base_model)

print("Applying LoRA…")
lora_cfg = LoRAConfig(
    r=cfg["lora"]["r"],
    alpha=cfg["lora"]["alpha"],
    dropout=cfg["lora"]["dropout"],
    target_modules=cfg["lora"]["modules"],
)
trainer = LLaMAFactory(model=model, peft_config=lora_cfg, tokenizer=cfg["tokenizer"])

print("Loading FIM dataset…")
train_ds = load_dataset("json", data_files="data/fim_dataset.json", split="train")

print("Tokenizing…")

def encode(batch):
    inputs  = [f"<|fim_begin|> {p} <|fim_hole|> {s} <|fim_end|>" for p, s in zip(batch["prefix"], batch["suffix"])]
    outputs = batch["target"]
    return trainer.tokenize(inputs, outputs)

train_ds = train_ds.map(encode, batched=True, remove_columns=train_ds.column_names)

args = TrainingArguments(**cfg["train_args"])
trainer.train(train_ds, args)
trainer.save_adapter("adapter/")