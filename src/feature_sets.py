"""
feature_sets.py — runtime node-feature selection (NO re-preprocessing needed)
=============================================================================
The preprocessed .pt files always carry the full 18 features. The trainer
slices columns at load time, so switching feature sets is a --features flag:

    full18   : everything (your v3 behavior)
    no_spef16: drops the two post-route SPEF features  -> solves flaw F1
    paper10  : exactly the paper's input scope: port flag, direction flag,
               4 die-boundary distances, 4 Liberty pin caps (EL/RF)

Your 18-feature layout (from model_configurable.py):
    0-1  x, y (norm)          9   is_endpoint
    2    is_PI                10  fanout_norm
    3    is_IN                11  net_cap_norm   (SPEF, post-route!)
    4    is_OUT               12  logic_depth_norm
    5-8  dist L/R/B/T         13  net_res_norm   (SPEF, post-route!)
                              14-17 pin caps ff-rise/ff-fall/ss-rise/ss-fall
"""
import torch

FEATURE_SETS = {
    "full18":    list(range(18)),
    "no_spef16": [i for i in range(18) if i not in (11, 13)],
    # paper-equivalent 10: is_PI (~paper's is-port), is_IN (~direction),
    # 4 boundary distances, 4 pin caps. (Add 4 for an 11-feature variant
    # keeping both direction flags.)
    "paper10":   [2, 3, 5, 6, 7, 8, 14, 15, 16, 17],
}


def select_features(g, set_name):
    """Slice g['x'] to the chosen set. Returns (g, n_features)."""
    idx = FEATURE_SETS[set_name]
    g = dict(g)                       # shallow copy; tensors shared
    g["x"] = g["x"][:, idx].contiguous()
    return g, len(idx)
