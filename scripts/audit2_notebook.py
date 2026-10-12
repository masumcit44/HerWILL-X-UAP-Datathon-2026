"""READ-ONLY: audit the EXECUTED notebook cell-by-cell. Modifies nothing."""
import os
import json
import hashlib
import nbformat

NB = r"_fresh_exec_test/submission_reproducible_file_2.ipynb"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    nb = nbformat.read(NB, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    md_cells = [c for c in nb.cells if c.cell_type == "markdown"]

    print("notebook path        :", NB)
    print("notebook sha256      :", sha(NB))
    print("total cells          :", len(nb.cells))
    print("  markdown / code    :", len(md_cells), "/", len(code_cells))

    counts = [c.get("execution_count") for c in code_cells]
    executed = [i for i, c in enumerate(counts, 1) if c is not None]
    print("code cells w/ exec_count:", len(executed), "/", len(code_cells))
    print("execution_count seq  :", counts)
    print("all code executed    :", all(c is not None for c in counts))
    print("counts strictly incr :",
          all(counts[i] < counts[i + 1] for i in range(len(counts) - 1)
              if counts[i] is not None and counts[i + 1] is not None))

    errs = []
    for i, c in enumerate(nb.cells, 1):
        for o in c.get("outputs", []):
            if o.output_type == "error":
                errs.append((i, o.get("ename"), o.get("evalue")))
    print("cells with error out :", len(errs), errs)

    print()
    print("=" * 78)
    print("EXECUTED CODE-CELL OUTPUT")
    print("=" * 78)
    for i, c in enumerate(nb.cells, 1):
        if c.cell_type != "code":
            continue
        txt = ""
        for o in c.get("outputs", []):
            if o.output_type == "stream":
                txt += o.get("text", "")
            elif o.output_type == "execute_result":
                txt += str(o.get("data", {}).get("text/plain", ""))
            elif o.output_type == "error":
                txt += "ERROR " + o.get("ename", "") + ": " + o.get("evalue", "")
        if txt.strip():
            print("-" * 70)
            print("CELL {} (exec_count={})".format(i, c.get("execution_count")))
            print("-" * 70)
            print(txt.rstrip())


if __name__ == "__main__":
    main()
