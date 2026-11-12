"""
tree_v2.py -- placement-only routing-tree estimate per net (rectilinear spanning tree rooted at
the driver) and the Elmore-delay ingredients computed on it.

For unit wire resistance r and capacitance c the Elmore delay to a sink is
        r*c * EW  +  r * EP
    EW = sum over path edges  len_e * (len_e/2 + wire length below e)        [um^2]
    EP = sum over path edges  len_e * (pin cap below e)                       [um*fF]
and the cap-weighted mean path resistance of the whole net (how strongly the wire shields its
load from the driver) is  r * (c*MW + MP) / Ctotal  with
    MW = sum over edges  len_e * (path length to parent + len_e/2)           [um^2]
    MP = sum over sinks  pin cap * path length                               [um*fF]
"""
import numpy as np


def net_trees(n_pins, xy, net_src, net_dst, sink_cap_ff):
    """net_src/net_dst: [E] edge list; sink_cap_ff: [E] pin cap of each sink.
    Returns per-edge (path_len, EW, EP, depth) and per-driver (tree_len, MW, MP)."""
    E = len(net_src)
    e_lp = np.zeros(E, np.float32); e_ew = np.zeros(E, np.float32)
    e_ep = np.zeros(E, np.float32); e_dep = np.zeros(E, np.float32)
    n_lt = np.zeros(n_pins, np.float32); n_mw = np.zeros(n_pins, np.float32); n_mp = np.zeros(n_pins, np.float32)
    order = np.argsort(net_src, kind="stable")
    src_s = net_src[order]
    starts = np.flatnonzero(np.r_[True, src_s[1:] != src_s[:-1]])
    ends = np.r_[starts[1:], E]
    cnt = ends - starts
    # fanout-1 nets: closed form
    one = starts[cnt == 1]
    eo = order[one]
    ln = np.abs(xy[net_dst[eo]] - xy[net_src[eo]]).sum(1)
    e_lp[eo] = ln; e_ew[eo] = ln * ln / 2; e_ep[eo] = ln * sink_cap_ff[eo]; e_dep[eo] = 1
    d = net_src[eo]
    n_lt[d] = ln; n_mw[d] = ln * ln / 2; n_mp[d] = ln * sink_cap_ff[eo]
    for a, b in zip(starts[cnt > 1], ends[cnt > 1]):
        eidx = order[a:b]
        drv = net_src[eidx[0]]
        pts = np.vstack([xy[drv][None], xy[net_dst[eidx]]]).astype(np.float64)
        caps = np.r_[0.0, sink_cap_ff[eidx]]
        n = len(pts)
        dist = np.abs(pts[:, None, 0] - pts[None, :, 0]) + np.abs(pts[:, None, 1] - pts[None, :, 1])
        in_tree = np.zeros(n, bool); in_tree[0] = True
        best = dist[0].copy(); par = np.zeros(n, np.int64)
        seq = np.zeros(n - 1, np.int64)
        best[0] = np.inf
        for t in range(n - 1):
            j = int(np.argmin(best))
            seq[t] = j
            in_tree[j] = True
            dj = dist[j]
            upd = (dj < best) & ~in_tree
            par[upd] = j
            best[upd] = dj[upd]
            best[j] = np.inf
        ln = dist[np.arange(n), par]; ln[0] = 0.0
        lsub = np.zeros(n); csub = caps.copy()
        for j in seq[::-1]:
            lsub[par[j]] += lsub[j] + ln[j]
            csub[par[j]] += csub[j]
        lp = np.zeros(n); ew = np.zeros(n); ep = np.zeros(n); dep = np.zeros(n)
        mw = 0.0
        for j in seq:
            q = par[j]
            lp[j] = lp[q] + ln[j]
            ew[j] = ew[q] + ln[j] * (ln[j] / 2 + lsub[j])
            ep[j] = ep[q] + ln[j] * csub[j]
            dep[j] = dep[q] + 1
            mw += ln[j] * (lp[q] + ln[j] / 2)
        e_lp[eidx] = lp[1:]; e_ew[eidx] = ew[1:]; e_ep[eidx] = ep[1:]; e_dep[eidx] = dep[1:]
        n_lt[drv] = ln.sum(); n_mw[drv] = mw; n_mp[drv] = float((caps * lp).sum())
    return e_lp, e_ew, e_ep, e_dep, n_lt, n_mw, n_mp
