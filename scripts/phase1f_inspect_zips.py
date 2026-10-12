"""List zip contents (no extraction) and hunt for GPU probability artifacts."""
import os
import re
import zipfile

KEY = re.compile(r"xlmr|xlm-roberta|idhead|e5|emb_|fasttext|all3|oof|test_probs"
                 r"fold_ids|fold_assign|base_sources|results_summary|CANDIDATE|uap_results",
                 re.I)
ARRAYISH = {".npy", ".npz", ".pt", ".bin", ".safetensors", ".pth", ".joblib",
            ".pkl", ".json", ".csv"}


def main():
    for z in sorted(os.listdir("recovered")):
        if not z.lower().endswith(".zip"):
            continue
        p = os.path.join("recovered", z)
        print("=" * 78)
        print(f"ZIP: {z}  ({os.path.getsize(p):,} bytes)")
        print("=" * 78)
        try:
            with zipfile.ZipFile(p) as zf:
                names = zf.namelist()
                print(f"  entries: {len(names)}")
                interesting = [n for n in names
                               if KEY.search(os.path.basename(n))
                               and os.path.splitext(n)[1].lower() in ARRAYISH]
                print(f"  GPU/array-looking entries: {len(interesting)}")
                for n in interesting[:60]:
                    info = zf.getinfo(n)
                    print(f"     {n[-95:]:<95} {info.file_size:>13,}")
                if not interesting:
                    top = sorted({n.split("/")[0] for n in names})[:25]
                    print("  top-level entries:", top)
        except zipfile.BadZipFile as e:
            print(f"  NOT A VALID ZIP: {e}")
        print()


if __name__ == "__main__":
    main()
