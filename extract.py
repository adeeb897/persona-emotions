"""Stage 5: replay each generated message through the probed model and hook the
residual stream.

Every message is replayed inside ITS OWN persona's chat template -- the system
prompt a message was generated under is the system prompt it is hooked under.
Activations are mean-pooled over the assistant response tokens only: the system
prompt and user turn are excluded, so a persona's system prompt cannot leak
into the pooled vector directly (it still conditions the response, which is the
effect we want to measure).

Forward hooks on the decoder layers, no TransformerLens.  Output is one .npy of
shape (n_rows, n_layers, hidden) plus a metadata parquet in the same row order.
No per-token activations are kept.

    python extract.py --limit 8                 # smoke test
    PP_PRESET=0.5b python extract.py --limit 8  # plumbing test, ~1GB model
    python extract.py                           # the real run
"""

import argparse
import json
import sys
import time

import config  # sets HF_HOME -- must precede transformers
import gen_data
import personas


def load_records(limit: int | None) -> list[dict]:
    if not config.GENERATIONS_JSONL.exists():
        print(f"[extract] ERROR: {config.GENERATIONS_JSONL} not found -- run gen_data.py first")
        sys.exit(1)
    records = [
        json.loads(line)
        for line in config.GENERATIONS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        records = records[:limit]
    return records


def tokenize_record(tok, record: dict, suffix_len: int) -> tuple[list[int], int, int]:
    """Return (input_ids, response_start, response_end).

    response_start is the first token of the assistant's message; response_end
    is one past its last token, excluding the trailing <|im_end|>\\n that the
    template appends.  Verified exact on the Qwen2.5 template.
    """
    messages = personas.build_messages(
        record["persona_id"], gen_data.user_prompt_for(record)
    )
    prefix = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    full = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": record["text"]}], tokenize=True
    )
    return full, len(prefix), len(full) - suffix_len


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only N records")
    ap.add_argument("--batch-size", type=int, default=config.EXTRACT_BATCH_SIZE)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = load_records(args.limit)
    print(f"[extract] {len(records)} records from {config.GENERATIONS_JSONL.name}")
    if args.limit is not None and config.ACTIVATIONS_NPY.exists():
        import numpy as _np
        existing = _np.load(config.ACTIVATIONS_NPY, mmap_mode="r").shape[0]
        if existing > len(records):
            print(f"[extract] WARNING: --limit {args.limit} will overwrite "
                  f"{config.ACTIVATIONS_NPY} which currently holds {existing} rows. "
                  f"Use PP_OUT=outputs/smoke to keep them separate.")
    print(f"[extract] model={config.PROBED_MODEL} dtype={config.DTYPE} layers={config.LAYERS}")

    tok = AutoTokenizer.from_pretrained(config.PROBED_MODEL)

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        config.PROBED_MODEL,
        dtype=getattr(torch, config.DTYPE),   # `torch_dtype` is deprecated in 4.57
        device_map="auto",          # accelerate places the shards
        low_cpu_mem_usage=True,     # stream shards in, never materialise a full copy
    )
    model.eval()
    load_s = time.time() - t0
    print(f"[extract] loaded in {load_s:.1f}s, "
          f"peak GPU {torch.cuda.max_memory_allocated() / 1024**3:.1f}GB")

    n_layers, hidden = config.check_model_dims(model.config)
    print(f"[extract] model reports {n_layers} layers, hidden {hidden}")

    # Tokens the template appends after the assistant content (<|im_end|>\n).
    # Measured, not assumed, so a template change cannot silently shift the pool.
    probe_msgs = [{"role": "user", "content": "x"}]
    suffix_len = len(
        tok.apply_chat_template(probe_msgs + [{"role": "assistant", "content": ""}],
                                tokenize=True)
    ) - len(tok.apply_chat_template(probe_msgs, add_generation_prompt=True, tokenize=True))
    print(f"[extract] assistant suffix is {suffix_len} token(s): "
          f"{tok.decode(tok.apply_chat_template(probe_msgs + [{'role': 'assistant', 'content': ''}], tokenize=True)[-suffix_len:])!r}")

    # --- hooks ------------------------------------------------------------
    captured: dict[int, "torch.Tensor"] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            # Decoder layers return a tuple; [0] is the residual stream.
            captured[layer_idx] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [
        model.model.layers[i].register_forward_hook(make_hook(i))
        for i in config.LAYERS
    ]

    # --- prepare ----------------------------------------------------------
    prepared, skipped = [], []
    for r in records:
        ids, start, end = tokenize_record(tok, r, suffix_len)
        if end <= start:
            skipped.append((r, "empty response slice"))
        elif len(ids) > config.MAX_SEQ_LEN:
            skipped.append((r, f"{len(ids)} tokens > MAX_SEQ_LEN"))
        else:
            prepared.append((r, ids, start, end))

    if skipped:
        print(f"[extract] WARNING: skipping {len(skipped)} record(s):")
        for r, why in skipped[:5]:
            print(f"    {r['persona_id']}/{r['emotion']}/{r['scenario_id']}: {why}")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    acts = np.zeros((len(prepared), len(config.LAYERS), hidden), dtype=np.float32)
    meta_rows = []

    # --- forward ----------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        for b0 in tqdm(range(0, len(prepared), args.batch_size), desc="extracting", unit="batch"):
            batch = prepared[b0 : b0 + args.batch_size]
            width = max(len(ids) for _, ids, _, _ in batch)

            # Right padding: causal attention plus an attention mask means the
            # pad tail cannot affect any real token's activations, and the
            # per-sample response slices stay valid as computed.
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), width), dtype=torch.long)
            for j, (_, ids, _, _) in enumerate(batch):
                input_ids[j, : len(ids)] = torch.tensor(ids)
                attn[j, : len(ids)] = 1

            device = next(model.parameters()).device
            model(input_ids=input_ids.to(device), attention_mask=attn.to(device))

            for li, layer in enumerate(config.LAYERS):
                hs = captured[layer].float()
                for j, (r, _, start, end) in enumerate(batch):
                    acts[b0 + j, li] = hs[j, start:end].mean(dim=0).cpu().numpy()

            for j, (r, ids, start, end) in enumerate(batch):
                meta_rows.append({
                    "row": b0 + j,
                    "persona_id": r["persona_id"],
                    "emotion": r["emotion"],
                    "scenario_id": r["scenario_id"],
                    "topic": r["topic"],
                    "n_tokens_total": len(ids),
                    "n_tokens_response": end - start,
                    "generator_model": r.get("generator_model"),
                })
            captured.clear()

    for h in handles:
        h.remove()

    fwd_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[extract] forward passes: {fwd_s:.1f}s "
          f"({len(prepared) / max(fwd_s, 1e-6):.1f} rows/s), peak GPU {peak:.1f}GB")

    # --- save -------------------------------------------------------------
    np.save(config.ACTIVATIONS_NPY, acts)
    meta = pd.DataFrame(meta_rows)
    meta.to_parquet(config.ACTIVATIONS_META, index=False)
    config.ACTIVATIONS_INFO.write_text(json.dumps({
        "model": config.PROBED_MODEL,
        "preset": config.PRESET,
        "dtype": config.DTYPE,
        "layers": config.LAYERS,          # axis 1 of the .npy is in THIS order
        "hidden": hidden,
        "n_model_layers": n_layers,
        "shape": list(acts.shape),
        "n_skipped": len(skipped),
        "pooling": "mean over assistant response tokens only",
        "load_seconds": round(load_s, 1),
        "forward_seconds": round(fwd_s, 1),
        "peak_gpu_gb": round(peak, 1),
    }, indent=2))

    print(f"[extract] acts {acts.shape} -> {config.ACTIVATIONS_NPY}")
    print(f"[extract] meta {meta.shape} -> {config.ACTIVATIONS_META}")
    print(f"[extract] info -> {config.ACTIVATIONS_INFO}")
    print("\n[extract] rows per persona:")
    for pid, n in meta["persona_id"].value_counts().items():
        print(f"    {pid:<18} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
