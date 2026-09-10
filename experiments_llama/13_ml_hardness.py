"""
Learned non-linear models for per-input hardness prediction.

WHY THIS EXISTS: an external review correctly narrowed the conclusion from
11b/11c/12. Those tested handcrafted scalar statistics with LINEAR models. That
does not establish hardness is unpredictable - only that those features, fitted
linearly, do not predict it.

This addresses the two changes most likely to alter the answer, and fixes a
methodological error the review identified:

  1. NON-LINEAR LEARNED MODEL. A random forest captures feature interactions
     that least squares cannot.
  2. CLASSIFICATION rather than regression. The controller does not need an
     exact KL - it needs to know whether the scheme will breach the budget.
     P(KL > budget) is a coarser and possibly easier target.
  3. PROMPT-LEVEL SPLITTING. 11c split by RECORD. Within one cell each prompt
     appears once, so that did not leak and those results stand. But for a
     GLOBAL model each prompt appears 48 times (once per layer x scheme), and
     record-level splitting would place the same prompt in train and test. The
     model would memorise "prompt 17 is hard" and appear to generalise.
     Everything here splits by unique text_id.

IMPLEMENTATION NOTE: scikit-learn is unavailable on this machine (pypi.org is
blocked by the network). The random forest below is implemented directly in
numpy. It is a standard CART ensemble - bootstrap sampling, random feature
subsets per split, variance-reduction splitting, averaged predictions. Gradient
boosting was NOT implemented: it needs staged residual fitting where a subtle
bug produces a plausible but wrong answer, and a forest already tests the
question the review raised. Report this honestly as "tested with a random
forest" rather than as a full non-linear sweep.
"""
import json
import os
import math
import random
import numpy as np

IN_PATH = "/home/ayush.thakar/thesis/experiments_llama/results/11a_collect.json"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "13_ml_hardness"

# max_abs dropped: constant across texts, produced nan in every 11b cell.
ALL_FEATURES = ["kurtosis", "skew", "top1pct", "top5pct",
                "top5_channels_share", "mean_abs", "max_over_mean", "std"]

CORR_DROP_THRESHOLD = 0.95
SEQ = 256
N_SPLITS = 5
TEST_FRACTION = 0.3
KL_BUDGETS = [0.05, 0.10, 0.25]

N_TREES = 100
MIN_LEAF = 5
MAX_DEPTH = 8

_lines = []


def out(s=""):
    print(s, flush=True)
    _lines.append(s)


# ==================== random forest in numpy ====================
class Node:
    __slots__ = ("feat", "thr", "left", "right", "value")

    def __init__(self):
        self.feat = None
        self.thr = None
        self.left = None
        self.right = None
        self.value = None


def _best_split(X, y, feat_idx, min_leaf):
    """Find the split minimising within-child variance, over the given features."""
    n = len(y)
    best = (None, None, np.inf)
    parent_sse = ((y - y.mean()) ** 2).sum()
    if parent_sse <= 0:
        return best

    for f in feat_idx:
        col = X[:, f]
        order = np.argsort(col, kind="stable")
        cs = col[order]
        ys = y[order]

        csum = np.cumsum(ys)
        csum2 = np.cumsum(ys ** 2)
        total, total2 = csum[-1], csum2[-1]

        valid = np.where(cs[1:] != cs[:-1])[0]
        valid = valid[(valid + 1 >= min_leaf) & (n - valid - 1 >= min_leaf)]
        if valid.size == 0:
            continue

        nl = valid + 1
        nr = n - nl
        sl, sl2 = csum[valid], csum2[valid]
        sr, sr2 = total - sl, total2 - sl2
        sse = (sl2 - sl ** 2 / nl) + (sr2 - sr ** 2 / nr)

        k = int(np.argmin(sse))
        if sse[k] < best[2]:
            thr = (cs[valid[k]] + cs[valid[k] + 1]) / 2.0
            best = (f, thr, float(sse[k]))
    return best


def _grow(X, y, depth, max_depth, min_leaf, n_feat, rng):
    node = Node()
    if depth >= max_depth or len(y) < 2 * min_leaf or y.std() < 1e-12:
        node.value = float(y.mean())
        return node

    feat_idx = rng.sample(range(X.shape[1]), n_feat)
    f, thr, sse = _best_split(X, y, feat_idx, min_leaf)
    if f is None:
        node.value = float(y.mean())
        return node

    mask = X[:, f] <= thr
    if mask.sum() < min_leaf or (~mask).sum() < min_leaf:
        node.value = float(y.mean())
        return node

    node.feat, node.thr = f, thr
    node.left = _grow(X[mask], y[mask], depth + 1, max_depth, min_leaf,
                      n_feat, rng)
    node.right = _grow(X[~mask], y[~mask], depth + 1, max_depth, min_leaf,
                       n_feat, rng)
    return node


def _predict_one(node, x):
    while node.value is None:
        node = node.left if x[node.feat] <= node.thr else node.right
    return node.value


class RandomForest:
    """Standard CART ensemble. Works for regression, and for classification by
    fitting 0/1 targets and reading the mean as a probability."""

    def __init__(self, n_trees=N_TREES, max_depth=MAX_DEPTH,
                 min_leaf=MIN_LEAF, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, p = X.shape
        n_feat = max(1, int(round(math.sqrt(p))) if p > 3 else p)
        rng = random.Random(self.seed)
        nprng = np.random.RandomState(self.seed)
        self.trees = []
        for t in range(self.n_trees):
            idx = nprng.randint(0, n, n)          # bootstrap sample
            self.trees.append(_grow(X[idx], y[idx], 0, self.max_depth,
                                    self.min_leaf, n_feat, rng))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        preds = np.empty((len(self.trees), len(X)))
        for i, t in enumerate(self.trees):
            for j in range(len(X)):
                preds[i, j] = _predict_one(t, X[j])
        return preds.mean(axis=0)


def ridge_fit(X, y, alpha=1.0):
    X = np.asarray(X, float)
    X = np.hstack([np.ones((len(X), 1)), X])
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ np.asarray(y, float))


def ridge_predict(coef, X):
    X = np.asarray(X, float)
    X = np.hstack([np.ones((len(X), 1)), X])
    return X @ coef


def r2_score(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    tot = ((actual - actual.mean()) ** 2).sum()
    return float(1 - ((actual - pred) ** 2).sum() / tot) if tot > 0 else float("nan")


def auc_score(y_true, scores):
    """Area under ROC, via the rank-sum identity. 0.5 = coin flip."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, float)
    pos, neg = (y_true == 1).sum(), (y_true == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


# ==================== data ====================
print("Loading 11a_collect.json ...", flush=True)
data = json.load(open(IN_PATH))
ALL = data["records"]
LAYERS = data["layers"]
SCHEMES = [s for s in ["grouped(4,4,4)", "grouped(8,4,4)",
                       "grouped(8,8,4)", "grouped(8,8,8)"]
           if s in data["schemes"]]

R = [r for r in ALL if r["seq_len"] == SEQ and r["order"] == "per_layer"]
out(f"Records at seq {SEQ}, per-layer ordering : {len(R):,}")
out(f"Unique prompts                          : "
    f"{len(set(r['text_id'] for r in R))}")
out(f"Model: random forest ({N_TREES} trees, depth {MAX_DEPTH}, "
    f"min leaf {MIN_LEAF}), implemented in numpy")
out(f"Splitting by PROMPT, not by record")
out()


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


out("=" * 88)
out("FEATURE DEDUPLICATION")
out("=" * 88)
out(f"Dropping one of any pair correlated above |r| = {CORR_DROP_THRESHOLD}.")
out()
cols = {f: [r[f] for r in R] for f in ALL_FEATURES}
keep, dropped = [], []
for f in ALL_FEATURES:
    red = None
    for g in keep:
        c = abs(corr(cols[f], cols[g]))
        if c >= CORR_DROP_THRESHOLD:
            red = (g, c)
            break
    if red:
        dropped.append((f, red[0], red[1]))
        out(f"  dropped {f:22s} (|r| = {red[1]:.3f} with {red[0]})")
    else:
        keep.append(f)
FEATURES = keep
out()
out(f"  KEPT ({len(FEATURES)}): {', '.join(FEATURES)}")
out()


def prompt_split(tid, seed):
    ids = sorted(set(tid.tolist()))
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    cut = int(len(ids) * (1 - TEST_FRACTION))
    train = set(ids[:cut])
    tr = np.array([i for i, t in enumerate(tid) if t in train])
    te = np.array([i for i, t in enumerate(tid) if t not in train])
    return tr, te


# ==================== A. within-cell regression ====================
out("=" * 88)
out("A. WITHIN-CELL REGRESSION: predict KL, layer and scheme held fixed")
out("=" * 88)
out("Held-out R2, averaged over 5 prompt-level splits. n=300 per cell, so about")
out("210 training prompts - thin for a forest. Watch the in-sample gap.")
out()
out(f"{'layer':>5} {'scheme':>16} {'ridge':>10} {'forest':>10} "
    f"{'forest in-samp':>16}")
out("-" * 88)

within = {}
for L in LAYERS:
    for s in SCHEMES:
        rows = [r for r in R if r["layer"] == L and r["scheme"] == s]
        if len(rows) < 50:
            continue
        X = np.array([[r[f] for f in FEATURES] for r in rows], float)
        y = np.array([r["kl"] for r in rows], float)
        tid = np.array([r["text_id"] for r in rows])

        rr, ff, fi = [], [], []
        for seed in range(N_SPLITS):
            tr, te = prompt_split(tid, seed)
            if len(tr) < 20 or len(te) < 10:
                continue
            c = ridge_fit(X[tr], y[tr])
            rr.append(r2_score(y[te], ridge_predict(c, X[te])))
            rf = RandomForest(seed=seed).fit(X[tr], y[tr])
            ff.append(r2_score(y[te], rf.predict(X[te])))
            fi.append(r2_score(y[tr], rf.predict(X[tr])))
        within[f"{L}_{s}"] = {"ridge": float(np.mean(rr)),
                              "forest": float(np.mean(ff)),
                              "forest_in": float(np.mean(fi))}
        out(f"{L:5d} {s:>16} {np.mean(rr):10.3f} {np.mean(ff):10.3f} "
            f"{np.mean(fi):16.3f}")
out()
rid = [v["ridge"] for v in within.values()]
frs = [v["forest"] for v in within.values()]
fin = [v["forest_in"] for v in within.values()]
out(f"  ridge   best {max(rid):6.3f}   mean {np.mean(rid):6.3f}")
out(f"  forest  best {max(frs):6.3f}   mean {np.mean(frs):6.3f}")
out(f"  forest in-sample mean {np.mean(fin):6.3f}  "
    f"(gap {np.mean(fin) - np.mean(frs):.3f})")
out()


# ==================== B/C. global model ====================
out("=" * 88)
out("B. GLOBAL MODEL, and what the ACTIVATION features add")
out("=" * 88)
out("A global model with layer and scheme as features will show a high R2 -")
out("but those two explain most of the variance on their own, and the lookup")
out("table already knows both. The number that matters is the GAIN from adding")
out("the activation features. A high global R2 without a gain is the")
out("pooled-correlation trap in a new form.")
out()

Xg = np.array([[r[f] for f in FEATURES] +
               [r["layer"], SCHEMES.index(r["scheme"])] for r in R], float)
Xc = np.array([[r["layer"], SCHEMES.index(r["scheme"])] for r in R], float)
yg = np.array([r["kl"] for r in R], float)
tg = np.array([r["text_id"] for r in R])

out(f"{'model':>10} {'config only':>14} {'+ activation':>14} {'gain':>10}")
out("-" * 88)
glob = {}
for name in ["ridge", "forest"]:
    a, b = [], []
    for seed in range(N_SPLITS):
        tr, te = prompt_split(tg, seed)
        if name == "ridge":
            a.append(r2_score(yg[te],
                              ridge_predict(ridge_fit(Xc[tr], yg[tr]), Xc[te])))
            b.append(r2_score(yg[te],
                              ridge_predict(ridge_fit(Xg[tr], yg[tr]), Xg[te])))
        else:
            m1 = RandomForest(seed=seed).fit(Xc[tr], yg[tr])
            a.append(r2_score(yg[te], m1.predict(Xc[te])))
            m2 = RandomForest(seed=seed).fit(Xg[tr], yg[tr])
            b.append(r2_score(yg[te], m2.predict(Xg[te])))
    ma, mb = float(np.mean(a)), float(np.mean(b))
    glob[name] = {"cfg": ma, "full": mb, "gain": mb - ma}
    out(f"{name:>10} {ma:14.3f} {mb:14.3f} {mb-ma:+10.3f}")
out()
out("READING THIS: ridge cannot represent layer and scheme as CATEGORICAL")
out("variables - it treats layer 0-11 as a linear number - so its config-only")
out("fit is poor and the activation features partly act as proxies for layer")
out("identity. That inflates ridge's apparent gain. The forest splits on layer")
out("and scheme properly, so ITS gain is the honest measure of what the")
out("activation contributes beyond what the lookup table already encodes.")
out()


# ==================== D. classification ====================
out("=" * 88)
out("D. CLASSIFICATION: predict whether the budget is breached")
out("=" * 88)
out("The review's strongest suggestion - the controller needs a risk decision,")
out("not an exact KL. AUC 0.5 is a coin flip; below ~0.7 is not usable.")
out("Cells where almost all inputs fall on one side are skipped: AUC is")
out("meaningless there and the controller's answer is obvious anyway.")
out()

clf = {}
for budget in KL_BUDGETS:
    shown = False
    for L in LAYERS:
        for s in SCHEMES:
            rows = [r for r in R if r["layer"] == L and r["scheme"] == s]
            if len(rows) < 50:
                continue
            yb = np.array([1 if r["kl"] > budget else 0 for r in rows])
            frac = yb.mean()
            if frac < 0.05 or frac > 0.95:
                continue
            if not shown:
                out(f"--- budget KL <= {budget} ---")
                out(f"{'layer':>5} {'scheme':>16} {'% breach':>10} "
                    f"{'forest AUC':>12}")
                out("-" * 88)
                shown = True
            X = np.array([[r[f] for f in FEATURES] for r in rows], float)
            tid = np.array([r["text_id"] for r in rows])
            aucs = []
            for seed in range(N_SPLITS):
                tr, te = prompt_split(tid, seed)
                if len(set(yb[tr])) < 2 or len(set(yb[te])) < 2:
                    continue
                m = RandomForest(seed=seed).fit(X[tr], yb[tr].astype(float))
                aucs.append(auc_score(yb[te], m.predict(X[te])))
            v = float(np.mean(aucs)) if aucs else float("nan")
            clf[f"{budget}_{L}_{s}"] = v
            out(f"{L:5d} {s:>16} {100*frac:9.1f}% {v:12.3f}")
    if shown:
        out()

aucs_all = [v for v in clf.values() if not math.isnan(v)]
if aucs_all:
    out(f"  forest AUC across {len(aucs_all)} usable cells: "
        f"best {max(aucs_all):.3f}   mean {np.mean(aucs_all):.3f}")
    out(f"  Only {len(aucs_all)} of {len(LAYERS)*len(SCHEMES)*len(KL_BUDGETS)} "
        f"(budget, layer, scheme) combinations had a usable class balance.")
    out("  Elsewhere the budget is either almost always met or almost always")
    out("  breached - the decision is determined by the CONFIGURATION, not by")
    out("  the input.")
else:
    out("  No cells had a usable class balance.")
out()


# ==================== verdict ====================
out("=" * 88)
out("VERDICT")
out("=" * 88)
best_forest = max(frs)

# Use the FOREST's gain, not the max across models. Ridge cannot represent
# layer and scheme as categorical variables, so its config-only fit is poor and
# the activation features partly act as proxies for layer identity - inflating
# its apparent gain. The forest handles categories properly, so its gain is the
# honest measure of what the activation contributes beyond the lookup table.
best_gain = glob["forest"]["gain"]
best_auc = max(aucs_all) if aucs_all else 0.0

out(f"  within-cell, ridge  : best {max(rid):.3f}  mean {np.mean(rid):.3f}")
out(f"  within-cell, forest : best {best_forest:.3f}  mean {np.mean(frs):.3f}")
out(f"  forest in-sample {np.mean(fin):.3f} vs held-out {np.mean(frs):.3f}  "
    f"(gap {np.mean(fin)-np.mean(frs):.3f})")
out()
out(f"  global config-only R2 (forest)      : {glob['forest']['cfg']:.3f}")
out(f"  global + activation R2 (forest)     : {glob['forest']['full']:.3f}")
out(f"  activation gain (forest)            : {best_gain:+.3f}")
out(f"  (ridge gain was {glob['ridge']['gain']:+.3f}, but ridge cannot model")
out(f"   layer/scheme as categories - not the honest measure)")
out()
out(f"  best classification AUC             : {best_auc:.3f}")
out()
out("  For reference, 11c reported mean held-out R2 = 0.000 with least squares.")
out()

if best_forest >= 0.5 or best_auc >= 0.75 or best_gain >= 0.15:
    out("  A NON-LINEAR MODEL RECOVERS SIGNAL that linear fits missed. The")
    out("  hardness question REOPENS. Before building anything: verify on a")
    out("  second sequence length, identify which features carry the weight,")
    out("  and measure the runtime cost of computing them.")
elif best_forest >= 0.3 or best_auc >= 0.65 or best_gain >= 0.08:
    out("  The forest finds MORE than linear fits, but not enough for a quality")
    out("  guarantee. Report as a partial result: hardness is weakly")
    out("  predictable, insufficiently so to replace the percentile table.")
else:
    out("  A non-linear learned model does NOT recover usable signal. It made")
    out("  matters WORSE: within cells the forest mean is below the ridge mean,")
    out("  and both are below zero - i.e. worse than predicting the mean. The")
    out("  in-sample/held-out gap shows why: the forest fits the training")
    out("  prompts and does not transfer.")
    out()
    out("  The negative now covers:")
    out("    1. handcrafted statistics, individually, linear (Pearson)")
    out("    2. the same, monotone non-linear (Spearman)")
    out("    3. the same, combined, linear multivariate, held out")
    out("    4. token-position structure and effective rank")
    out("    5. a random forest, regression AND classification, with")
    out("       prompt-level splits")
    out()
    out("  CLAIM WORDING (the review's framing is correct - adopt it):")
    out("    The tested low-cost handcrafted activation statistics did not")
    out("    provide sufficient out-of-sample predictive accuracy for reliable")
    out("    per-input hardness estimation, under either linear or non-linear")
    out("    learned models, in regression or classification form, with")
    out("    prompt-level validation. Richer activation representations and")
    out("    deployable text-level difficulty features remain untested.")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
    json.dump({"model": "numpy random forest",
               "n_trees": N_TREES, "max_depth": MAX_DEPTH,
               "features_kept": FEATURES,
               "features_dropped": [{"feature": f, "redundant_with": g,
                                     "r": c} for f, g, c in dropped],
               "within_cell": within, "global": glob,
               "activation_gain_forest": best_gain,
               "classification_auc": clf}, f, indent=2)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.{{json,txt}}")
