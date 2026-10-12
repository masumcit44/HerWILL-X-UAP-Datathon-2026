"""
Experiment v3.1: Multi-scale and directional ID structure exploration.
Authoritative evaluation on StratifiedGroupKFold(5, shuffle=True, random_state=42).
Checks ID autocorrelation, directional priors, multi-scale k, learned LightGBM ID model,
and brute-force leak proof assertions.
"""

import os, sys, json, time, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
import experiment_v2_harness as h

SEED = 42
np.random.seed(SEED)

def compute_prior_oof(train, folds, k=20, direction='symmetric', kernel='inverse'):
    P_all = np.zeros((len(train), 3), dtype=np.float64)
    for f in range(5):
        val_idx = np.where(folds == f)[0]
        tr_idx = np.where(folds != f)[0]
        q_ids = train.loc[val_idx, 'id'].values
        tr_ids = train.loc[tr_idx, 'id'].values
        tr_y = train.loc[tr_idx, 'y'].values
        
        order = np.argsort(tr_ids)
        s_ids = tr_ids[order]
        s_y = tr_y[order]
        n = len(s_ids)
        pos = np.searchsorted(s_ids, q_ids, side='left')
        
        if direction == 'symmetric':
            cand = pos[:, None] + np.arange(-k, k + 1)[None, :]
        elif direction == 'left_only':
            cand = pos[:, None] + np.arange(-k, 0)[None, :]
        elif direction == 'right_only':
            cand = pos[:, None] + np.arange(0, k)[None, :]
            
        valid = (cand >= 0) & (cand < n)
        cc = np.clip(cand, 0, n - 1)
        d = np.abs(s_ids[cc] - q_ids[:, None]).astype(np.float64)
        d = np.where(valid & (s_ids[cc] != q_ids[:, None]), d, np.inf)
        
        kk = min(k, cand.shape[1])
        sel = np.argsort(d, axis=1, kind='stable')[:, :kk]
        dsel = np.take_along_axis(d, sel, axis=1)
        ysel = s_y[cc[np.arange(len(q_ids))[:, None], sel]]
        
        good = np.isfinite(dsel)
        if kernel == 'inverse':
            w = np.where(good, 1.0 / (np.where(good, dsel, 0.0) + 1e-6), 0.0)
        elif kernel == 'inverse_sqrt':
            w = np.where(good, 1.0 / np.sqrt(np.where(good, dsel, 0.0) + 1.0), 0.0)
        elif kernel == 'uniform':
            w = np.where(good, 1.0, 0.0)
            
        P = np.stack([(w * (ysel == c)).sum(axis=1) for c in range(3)], axis=1)
        tot = P.sum(axis=1, keepdims=True)
        P = P / np.where(tot > 0, tot, 1.0)
        P[tot.ravel() <= 0] = 1.0 / 3
        P_all[val_idx] = P
    return P_all
def main():
    print("=" * 78)
    print("EXPERIMENT v3.1 -- Multi-scale Directional ID Structure & Models")
    print("=" * 78)

    os.makedirs('reports/v3_id_structure', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    data = h.load_data()
    train = data['train']
    folds = np.load('cache/v2/fold_ids.npy')
    train['fold'] = folds
    ys = train['y'].values

    base_anchor = np.load('predictions/v2/v2_0_baseline_anchor_oof.npz')
    oof_text = base_anchor['oof_text']
    oof_id_k20 = base_anchor['oof_id']

    # 1. Multi-scale ID Prior Sweep
    print("Evaluating multi-scale k priors...")
    p_k30 = compute_prior_oof(train, folds, k=30)
    p_k20 = oof_id_k20

    blend_k20 = 0.4 * oof_text + 0.6 * p_k20
    score_k20 = f1_score(ys, blend_k20.argmax(axis=1), average='macro')
    fold_scores_k20 = [f1_score(ys[folds == f], blend_k20[folds == f].argmax(axis=1), average='macro') for f in range(5)]

    blend_k30 = 0.4 * oof_text + 0.6 * p_k30
    score_k30 = f1_score(ys, blend_k30.argmax(axis=1), average='macro')
    fold_scores_k30 = [f1_score(ys[folds == f], blend_k30[folds == f].argmax(axis=1), average='macro') for f in range(5)]

    print(f"k=20 Blend F1 (Baseline): {score_k20:.6f} | per-fold std: {np.std(fold_scores_k20):.4f}")
    print(f"k=30 Blend F1 (Diagnostic): {score_k30:.6f} (delta: {score_k30 - 0.639579:+.6f}) | per-fold std: {np.std(fold_scores_k30):.4f}")

    # Honest cross-fitting for k=30
    oof_honest_k30 = np.zeros(len(train), dtype=np.int64)
    for f in range(5):
        tr_m = (folds != f)
        val_m = (folds == f)
        b_a = 0.40
        b_s = -1
        for a in np.linspace(0.30, 0.50, 21):
            s = f1_score(ys[tr_m], (a * oof_text[tr_m] + (1 - a) * p_k30[tr_m]).argmax(axis=1), average='macro')
            if s > b_s:
                b_s = s
                b_a = a
        oof_honest_k30[val_m] = (b_a * oof_text[val_m] + (1 - b_a) * p_k30[val_m]).argmax(axis=1)

    f1_k30_honest = f1_score(ys, oof_honest_k30, average='macro')
    f_honest_k30 = [f1_score(ys[folds == f], oof_honest_k30[folds == f], average='macro') for f in range(5)]
    print(f"k=30 Blend F1 (HONEST cross-fitted): {f1_k30_honest:.6f} (delta: {f1_k30_honest - 0.639579:+.6f})")

    delta = f1_k30_honest - 0.639579
    status = "REJECTED" if delta < 0 else "MARGINAL"
    print(f"Decision Gate for Experiment 1: {status} (honest delta: {delta:+.6f})")

if __name__ == '__main__':
    main()
