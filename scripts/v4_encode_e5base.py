"""Resumable FROZEN E5-Base CPU encoder (inference only, NO fine-tuning).

Saves chunks immediately so a power loss never loses more than one chunk.
Re-running is safe: already-written chunks are skipped.
"""
import os
import time
import argparse

import numpy as np
import torch

MODEL = "intfloat/multilingual-e5-base"
MAX_LEN = 128
CHUNK = 2000
OUT_TRAIN = "cache/v4/emb_e5base_train.npy"
OUT_TEST = "cache/v4/emb_e5base_test.npy"
DIR = "cache/v4"
THREADS = 8

torch.set_num_threads(THREADS)


def assemble(base, nchunks, n, out):
    parts = [np.load("{}.chunk{:04d}.npy".format(base, i)) for i in range(nchunks)]
    full = np.concatenate(parts, axis=0).astype(np.float32)
    assert full.shape[0] == n, (full.shape, n)
    assert np.isfinite(full).all()
    tmp = out + ".tmp.npy"
    np.save(tmp, full)
    os.replace(tmp, out)
    print("[{}] assembled {} -> {}".format(base, full.shape, out), flush=True)
    for i in range(nchunks):
        try:
            os.remove("{}.chunk{:04d}.npy".format(base, i))
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["train", "test"], required=True)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(DIR, exist_ok=True)

    import pandas as pd
    from sentence_transformers import SentenceTransformer

    csv = "train.csv" if args.which == "train" else "test.csv"
    df = pd.read_csv(csv, encoding="utf-8")
    df["text"] = df["text"].astype("string").fillna("").astype(str)
    texts = df["text"].tolist()
    n = len(texts)
    out = OUT_TRAIN if args.which == "train" else OUT_TEST
    base = os.path.splitext(out)[0]

    nchunks = (n + CHUNK - 1) // CHUNK
    done = set()
    for i in range(nchunks):
        cp = "{}.chunk{:04d}.npy".format(base, i)
        if os.path.exists(cp):
            try:
                a = np.load(cp)
                if a.shape[0] == min(CHUNK, n - i * CHUNK):
                    done.add(i)
            except Exception:
                pass
    print("[{}] rows={} chunks={} already_done={}".format(
        args.which, n, nchunks, len(done)), flush=True)
    if len(done) == nchunks:
        assemble(base, nchunks, n, out)
        return

    mdl = SentenceTransformer(MODEL, device="cpu")
    dim = mdl.get_sentence_embedding_dimension()
    print("[{}] model loaded dim={}".format(args.which, dim), flush=True)

    t0 = time.time()
    for i in range(nchunks):
        cp = "{}.chunk{:04d}.npy".format(base, i)
        if i in done:
            continue
        lo, hi = i * CHUNK, min((i + 1) * CHUNK, n)
        arr = mdl.encode(texts[lo:hi], batch_size=args.bs,
                         show_progress_bar=False, normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
        tmp = cp + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, cp)
        print("[{}] chunk {}/{} rows {}/{} elapsed={:.1f}min".format(
            args.which, i + 1, nchunks, hi, n, (time.time() - t0) / 60), flush=True)
    assemble(base, nchunks, n, out)
    print("[{}] DONE in {:.1f} min".format(args.which, (time.time() - t0) / 60),
          flush=True)


if __name__ == "__main__":
    main()
