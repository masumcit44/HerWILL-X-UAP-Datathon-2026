"""Assemble scripts/cells/cells.txt into the submission notebook."""
import nbformat as nbf

SRC = "scripts/cells/cells.txt"
OUT = "uap_submission/reproducible/submission_reproducible_file_2.ipynb"

with open(SRC, encoding="utf-8") as f:
    raw = f.read()

cells = []
kind = None
buf = []
for line in raw.splitlines():
    if line == "===MD===":
        if kind:
            cells.append((kind, "\n".join(buf).strip("\n")))
        kind, buf = "md", []
    elif line == "===CODE===":
        if kind:
            cells.append((kind, "\n".join(buf).strip("\n")))
        kind, buf = "code", []
    elif line == "===END===":
        if kind:
            cells.append((kind, "\n".join(buf).strip("\n")))
        kind, buf = None, []
    elif kind:
        buf.append(line)
if kind:
    cells.append((kind, "\n".join(buf).strip("\n")))

nb = nbf.v4.new_notebook()
for kind, src in cells:
    nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md"
                    else nbf.v4.new_code_cell(src))
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
}
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(nb.cells), "cells")
print("  markdown:", sum(1 for k, _ in cells if k == "md"),
      " code:", sum(1 for k, _ in cells if k == "code"))
