"""Check the LIVE D:\\Desktop\\UAP.zip + verify raw CSV integrity vs the competition zip."""
import os
import re
import zipfile
import hashlib

KEY = re.compile(r"xlmr|xlm-roberta|idhead|e5|emb_|fasttext|all3|test_probs"
                 r"base_sources|results_summary|CANDIDATE|uap_results", re.I)
ARRAYISH = {".npy", ".npz", ".pt", ".bin", ".safetensors", ".pth", ".joblib", ".pkl"}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    live = r"D:\Desktop\UAP.zip"
    print("=" * 78)
    print("LIVE D:\\Desktop\\UAP.zip")
    print("=" * 78)
    print("exists:", os.path.exists(live),
          f"({os.path.getsize(live):,} bytes)" if os.path.exists(live) else "")
    if os.path.exists(live):
        with zipfile.ZipFile(live) as zf:
            names = zf.namelist()
            print("entries:", len(names))
            gpuish = [n for n in names
                      if (KEY.search(os.path.basename(n))
                          and os.path.splitext(n)[1].lower() in ARRAYISH)]
            print("GPU array entries:", len(gpuish))
            for n in gpuish[:40]:
                print(f"   {n[-90:]:<90} {zf.getinfo(n).file_size:>13,}")
            newer = [n for n in names if n.endswith("uap_gpu.ipynb")]
            print("contains uap_gpu.ipynb:", bool(newer))

    print()
    print("=" * 78)
    print("RAW CSV INTEGRITY: competition zip vs working copies")
    print("=" * 78)
    comp = os.path.join("recovered", "HerWILL-Safe-Social-Media-Datathon-2026.zip")
    with zipfile.ZipFile(comp) as zf:
        for n in zf.namelist():
            if not n.endswith(".csv"):
                continue
            data = zf.read(n)
            zh = sha256_bytes(data)
            local = n if os.path.exists(n) else None
            print(f"  {n}")
            print(f"     zip sha256      = {zh}")
            if local:
                lb = open(local, "rb").read()
                lh = sha256_bytes(lb)
                print(f"     local sha256    = {lh}")
                print(f"     IDENTICAL       = {zh == lh}")
            else:
                print(f"     local           = MISSING at repo root ({n})")


if __name__ == "__main__":
    main()
