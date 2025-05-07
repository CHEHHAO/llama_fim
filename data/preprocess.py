import random, json, yaml
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载配置（包含 mask_ratio 与输出路径）
cfg = yaml.safe_load(open("../config.yaml"))

print("Loading cleaned Java dataset…")
raw_ds = load_dataset("ammarnasr/the-stack-java-clean", split="test")
print("Dataset size:", len(raw_ds))

print("Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])

max_len = cfg.get("max_tokens", 256) * 4  # 保证片段不过长

out = []
for ex in raw_ds:
    code = ex["content"]
    # 略过超长文件以提升效率
    if len(code) > max_len:
        continue
    toks = tokenizer.tokenize(code)
    if len(toks) < 20:
        continue
    n = len(toks)
    hole_len = max(1, int(n * cfg.get("mask_ratio", 0.3)))
    start = random.randint(1, n - hole_len - 1)
    prefix = tokenizer.convert_tokens_to_string(toks[:start])
    hole   = tokenizer.convert_tokens_to_string(toks[start:start+hole_len])
    suffix = tokenizer.convert_tokens_to_string(toks[start+hole_len:])
    out.append({"prefix": prefix, "suffix": suffix, "target": hole})

print("Generated", len(out), "FIM examples")
json.dump(out, open("fim_test_dataset.json", "w"), ensure_ascii=False, indent=2)