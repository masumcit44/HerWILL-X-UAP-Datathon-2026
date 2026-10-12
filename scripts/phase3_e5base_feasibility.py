"""Phase 3(d): E5-base frozen-CPU feasibility probe. NO training, inference only."""
import os
import time
import torch
import numpy as np
import pandas as pd
import psutil

MODEL = "intfloat/multilingual-e5-base"
MAX_LEN = 128
THREADS = 8
BS_PROBE = 64
N_PROBE = 512

torch.set_num_threads(THREADS)
os.environ.setdefault("HF_HUB_OFFLINE", "0")


def main():
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to("cpu").eval()
    load_s = time.time() - t0
    dim = mdl.config.hidden_size
    print(f"model            : {MODEL}")
    print(f"hidden size      : {dim}")
    print(f"load seconds     : {load_s:.1f}")
    print(f"torch threads    : {torch.get_num_threads()}")
    print(f"RAM available GB : {psutil.virtual_memory().available/1024**3:.2f}")

    train = pd.read_csv("train.csv", encoding="utf-8")
    train["text"] = train["text"].astype("string").fillna("").astype(str)
    texts = ["query: " + t for t in train["text"].head(N_PROBE).tolist()]
    lens = [len(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]) for t in texts[:200]]
    print(f"token len mean/max (first 200): {np.mean(lens):.1f} / {max(lens)}")

    t1 = time.time()
    with torch.no_grad():
        b = tok(texts, padding=True, truncation=True, max_length=MAX_LEN,
                return_tensors="pt")
        h = mdl(**b).last_hidden_state.float()
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp_min(1)
        e = F.normalize(e, dim=-1).numpy()
    dt = time.time() - t1
    rate = N_PROBE / dt
    total = (47817 + 11955) / rate
    print(f"\nprobe            : {N_PROBE} texts in {dt:.1f}s -> {rate:.2f} texts/s")
    print(f"finite           : {bool(np.isfinite(e).all())}  shape {e.shape}")
    print(f"PROJECTED total  : {total/60:.1f} min for 59,772 texts (train+test)")
    est_mem = psutil.virtual_memory().available / 1024**3
    print(f"RAM still free   : {est_mem:.2f} GB")
    out = e.astype(np.float32)
    os.makedirs("cache/v3", exist_ok=True)
    np.save("cache/v3/_e5base_probe.npy", out)
    print("probe embeddings saved -> cache/v3/_e5base_probe.npy")
    print("VERDICT:", "FEASIBLE on CPU" if total / 60 < 90 else "INFEASIBLE (too slow)")


if __name__ == "__main__":
    main()
