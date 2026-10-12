"""Assemble submission_reproducible_file_3.ipynb from scripts/nb3_cells/.

Each cell body lives in its own plain-text file, so no nested quoting is
needed anywhere. The first line of each .md cell file is a '#' comment naming
the cell and is dropped from the notebook.
"""
import glob
import json
import os

CELL_DIR = "scripts/nb3_cells"
OUT = "uap_submission/reproducible/submission_reproducible_file_3.ipynb"

cells = []
for path in sorted(glob.glob(os.path.join(CELL_DIR, "*"))):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    stem = os.path.splitext(os.path.basename(path))[0]
    # drop the leading '# Notebook cell ..' comment and any trailing blanks
    if lines and lines[0].startswith("# Notebook cell"):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    src = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
    if path.endswith(".md"):
        cells.append({"cell_type": "markdown", "id": stem, "metadata": {},
                      "source": src})
    else:
        cells.append({"cell_type": "code", "id": stem, "execution_count": None,
                      "metadata": {}, "outputs": [], "source": src})
    print("  {:<28} {} ({:>4} lines)".format(os.path.basename(path),
                                           cells[-1]["cell_type"], len(src)))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("wrote {} with {} cells".format(OUT, len(cells)))
