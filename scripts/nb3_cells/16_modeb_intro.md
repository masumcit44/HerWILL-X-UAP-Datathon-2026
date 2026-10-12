# Notebook cell 16 - markdown: Mode B
## 14. Mode B - optional GPU retraining of the two transformer sources

Disabled by default (`RUN_GPU_RETRAIN = False`). Setting it to `True` retrains
`xlmr_idhead` and `e5_base` on a T4 (~70 min) and replaces the bundled
probability arrays with the freshly trained ones. **fp16 GPU training is not
bit-reproducible**, so the resulting CSV may differ slightly from the packaged
one. Mode A remains the reference path.
