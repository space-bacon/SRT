#!/usr/bin/env python3
"""Measure bare backbone CE on validation data (no adapter)."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from srt.data.dataset import SRTAdapterDataset, make_collate_fn

print("Loading backbone...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B", torch_dtype=torch.bfloat16
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading val data (1000 samples)...")
ds = SRTAdapterDataset("data/all_val.jsonl", tokenizer, max_seq_len=512, max_samples=1000)
loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=make_collate_fn(tokenizer.pad_token_id))

total_ce = 0.0
count = 0
with torch.no_grad():
    for batch in loader:
        ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        mask = batch["attention_mask"].cuda()
        out = model(input_ids=ids, attention_mask=mask)
        shift_logits = out.logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_ce += ce.item()
        count += 1
        if count % 10 == 0:
            print(f"  batch {count}: running CE = {total_ce / count:.4f}")

print(f"\nBare Qwen 2.5-7B CE on val (1000 samples): {total_ce / count:.4f}")
