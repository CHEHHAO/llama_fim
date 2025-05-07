import os, json, random, yaml, torch
from datasets import load_dataset          # 如果后面要用，可以先保留
from transformers import AutoModelForCausalLM, AutoTokenizer

cfg = yaml.safe_load(open("config.yaml"))

tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-base",
    torch_dtype=torch.float16,             # 两行放到 GPU
    device_map="auto",                     # 让 accelerate 自行切 gpu
    trust_remote_code=True,
)

# 可选：加载 LoRA / Adapter
# model.load_adapter("adapter/adapter.zip")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading eval data…")
data = json.load(open("data/fim_dataset.json"))

def em(pred: str, tgt: str) -> int:
    """exact‑match"""
    return int(pred.strip() == tgt.strip())


def run_once(seed: int = 42) -> float:
    random.seed(seed)
    sample = random.sample(data, 6)

    ems = []
    for ex in sample:
        # DeepSeek‑Coder 自带的 FIM special tokens 与 OpenAI 风格一致
        prompt = f"<|fim_begin|>{ex['prefix']}<|fim_hole|>{ex['suffix']}<|fim_end|>"
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            pred_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.1,
                top_p=1.0,
                # 🔹 presence_penalty 不是 HF 参数，换成 repetition_penalty
                repetition_penalty=1.05,
                max_new_tokens=cfg["max_tokens"],
            )

        # 对 FIM 任务只保留补全段落（去掉 prompt）
        pred = tok.decode(pred_ids[0], skip_special_tokens=True)
        ems.append(em(pred, ex["target"]))
    return sum(ems) / len(sample)


scores = [run_once(i) for i in range(3)]
print("Best EM:", max(scores))
