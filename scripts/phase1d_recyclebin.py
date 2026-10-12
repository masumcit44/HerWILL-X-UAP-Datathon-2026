"""Read Recycle Bin $I metadata to recover ORIGINAL filenames (read-only)."""
import os
import struct
import datetime

BIN_DIRS = [r"C:\$Recycle.Bin", r"D:\$Recycle.Bin",
            r"E:\$Recycle.Bin", r"F:\$Recycle.Bin"]


def parse_I(path):
    """$I file: 8-byte header, 8-byte file size, 8-byte FILETIME, then UTF-16 name."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 28:
        return None
    ver = struct.unpack("<I", data[0:4])[0]
    size = struct.unpack("<Q", data[8:16])[0]
    ft = struct.unpack("<Q", data[16:24])[0]
    if ver == 2:
        nchars = struct.unpack("<I", data[24:28])[0]
        name = data[28:28 + nchars * 2].decode("utf-16-le", "replace")
    else:
        name = data[24:].decode("utf-16-le", "replace").split("\x00")[0]
    try:
        when = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=ft // 10)
    except Exception:
        when = "?"
    return {"orig_name": name, "orig_size": size, "deleted": str(when)}


def main():
    rows = []
    for b in BIN_DIRS:
        if not os.path.isdir(b):
            continue
        for r, dd, ff in os.walk(b):
            dd[:] = [x for x in dd if x != "System"]
            for fn in ff:
                if not fn.startswith("$I"):
                    continue
                p = os.path.join(r, fn)
                try:
                    meta = parse_I(p)
                except Exception as e:
                    meta = {"orig_name": f"<err {e}>", "orig_size": 0, "deleted": "?"}
                if not meta:
                    continue
                rid = fn[2:]
                r_content = os.path.join(r, "$R" + rid)
                csize = os.path.getsize(r_content) if os.path.exists(r_content) else 0
                clean = (meta["orig_name"] or "").rstrip().replace("\x00", "")
                rows.append((clean, meta["orig_size"], csize,
                             meta["deleted"], r_content))
    print(f"recycle-bin items parsed: {len(rows)}")
    print("\n--- items whose original name is a zip / npz / npy / pt / safetensors ---")
    key = (".zip", ".npz", ".npy", ".pt", ".pth", ".safetensors", ".bin", ".pkl")
    n = 0
    for name, osz, csz, when, rc in sorted(rows, key=lambda x: x[3], reverse=True):
        if name.lower().endswith(key):
            n += 1
            print(f"  {name:<45} orig={osz:>12,}  recovered={csz:>12,}  {when}")
            print(f"      content: {rc}")
    print("  matching count:", n)
    print("\n--- all recovered content that is >= 1 MB ---")
    for name, osz, csz, when, rc in sorted(rows, key=lambda x: x[2], reverse=True):
        if csz >= 1_000_000:
            print(f"  {name:<45} orig={osz:>12,}  recovered={csz:>12,}  {when}")


if __name__ == "__main__":
    main()
