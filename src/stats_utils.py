"""Fast DeLong AUROC variance/covariance (Sun & Xu 2014) for analytic CIs and
paired Delta-AUROC tests, plus Benjamini-Hochberg FDR. No bootstrap needed."""
import numpy as np
from scipy import stats


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out


def fast_delong(preds, m):
    """preds: [k, n] scores, positives in the first `m` columns. Returns
    (aucs[k], cov[k,k]) — DeLong covariance of the k AUROC estimates."""
    k, N = preds.shape
    n = N - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n          # [k, m] case structural components
    v10 = 1.0 - (tz[:, m:] - ty) / m    # [k, n] control structural components
    sx = np.cov(v01) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    sy = np.cov(v10) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_analyze(y, score_dict):
    """y: 0/1 array. score_dict: {name: scores}. Returns dict with per-name
    auc, var, 95% CI; and pairwise delta (auc_i - auc_j) with SE and one-sided
    p-value for delta>0."""
    y = np.asarray(y)
    names = list(score_dict)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    order = np.concatenate([pos, neg])
    m = len(pos)
    preds = np.vstack([np.asarray(score_dict[nm])[order] for nm in names])
    aucs, cov = fast_delong(preds, m)
    out = {"names": names, "auc": {}, "ci": {}, "var": {}, "delta": {}}
    for i, nm in enumerate(names):
        v = max(cov[i, i], 0.0)
        se = np.sqrt(v)
        out["auc"][nm] = float(aucs[i])
        out["var"][nm] = float(v)
        out["ci"][nm] = (float(aucs[i] - 1.96 * se), float(aucs[i] + 1.96 * se))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            d = aucs[i] - aucs[j]
            ve = cov[i, i] + cov[j, j] - 2 * cov[i, j]
            se = np.sqrt(max(ve, 1e-12))
            z = d / se
            # two-sided p for "different", one-sided p for "i>j"
            out["delta"][f"{a}-{b}"] = dict(
                delta=float(d), se=float(se),
                p_two=float(2 * stats.norm.sf(abs(z))),
                p_gt=float(stats.norm.sf(z)))   # H1: a>b
    return out


def auc_gt_half_p(auc, var):
    """one-sided p that AUC>0.5."""
    se = np.sqrt(max(var, 1e-12))
    return float(stats.norm.sf((auc - 0.5) / se))


def benjamini_hochberg(pvals):
    """BH-FDR q-values for an array of p-values (NaNs pass through)."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    pp = p[ok]
    n = len(pp)
    if n == 0:
        return q
    order = np.argsort(pp)
    ranked = pp[order]
    qv = ranked * n / (np.arange(n) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    out = np.empty(n)
    out[order] = qv
    q[ok] = out
    return q


if __name__ == "__main__":  # self-test
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    # 'both' strictly more informative than 'cbc' than 'eye'
    eye = y * 0.6 + rng.normal(0, 1, n)
    cbc = y * 0.9 + rng.normal(0, 1, n)
    both = y * 1.3 + rng.normal(0, 1, n)
    r = delong_analyze(y, {"eye": eye, "cbc": cbc, "both": both})
    from sklearn.metrics import roc_auc_score
    for nm in r["names"]:
        print(f"{nm}: DeLong AUC={r['auc'][nm]:.4f} sklearn={roc_auc_score(y, {'eye':eye,'cbc':cbc,'both':both}[nm]):.4f} CI={tuple(round(x,3) for x in r['ci'][nm])}")
    print("cbc-both:", {k: round(v, 4) for k, v in r["delta"]["cbc-both"].items()})
    print("eye-both:", {k: round(v, 4) for k, v in r["delta"]["eye-both"].items()})
    print("BH test:", np.round(benjamini_hochberg([0.001, 0.01, 0.04, 0.2, 0.5]), 4))
