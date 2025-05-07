import random
import yaml
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# 1. 加载配置
cfg = yaml.safe_load(open("config.yaml"))

# 2. 准备模型 + Tokenizer
tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")

# add new special tokens
extra = {"additional_special_tokens": ["<|fim_begin|>", "<|fim_hole|>", "<|fim_end|>"]}
# added = tok.add_special_tokens(extra)
tok.add_special_tokens(
    {"additional_special_tokens": extra["additional_special_tokens"],
     "bos_token": "<|fim_begin|>", "eos_token": "<|fim_end|>"}
)

model.resize_token_embeddings(len(tok))

# model.load_adapter("adapter/adapter.zip")
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# 3. 加载 FIM 数据集
data = load_dataset("json", data_files="data/fim_valid_dataset.json", split="train")

# 4. 加载 ExactMatch 指标
metric = evaluate.load("exact_match")

def first_n_lines(text, n=6):
    """提取前 n 行，去掉结尾空白。"""
    return [l.rstrip() for l in text.splitlines()[:n]]

def sample_score(pred, ref):
    """计算单个样本的 6 行 EM 平均值。"""
    pred_lines = first_n_lines(pred, 6)
    ref_lines  = first_n_lines(ref, 6)
    # 如果预测行数不足 6，补空行保证长度一致
    while len(pred_lines) < 6: pred_lines.append("")
    while len(ref_lines)  < 6: ref_lines.append("")
    em = metric.compute(predictions=pred_lines, references=ref_lines)["exact_match"]
    return em   # 已经是 0~1 之间的平均分

scores = []
for ex in data:                       # 遍历全部评测样本
    prompt = f"<|fim_begin|>{ex['prefix']}<|fim_hole|>{ex['suffix']}<|fim_end|>"
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    out_ids = model.generate(
        **inputs,
        do_sample=True, temperature=0.1, top_p=1.0,
        max_new_tokens=cfg["max_tokens"],
    )
    pred_completion = tok.decode(out_ids[0], skip_special_tokens=True)
    print(f"pred:" + pred_completion)
    ref_completion  = ex["target"]
    scores.append(sample_score(pred_completion, ref_completion))

final_score = sum(scores) / len(scores)
print(f"Overall EM‑6 score: {final_score:.4f}")
