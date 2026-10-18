"""
model_configurable.py
======================
TimingGNN model architecture: 12 node features, 4 net-edge features,
1-table Kronecker CellLUTModule (the proven, non-collapsing approach).

Matches the graph tensor structure produced by the uploaded
parse_real_design.py + interactive_train_real.py:real_graph_to_tensors().

NODE FEATURES (12) -- must match real_graph_to_tensors() exactly:
    0:  x / canvas_scale
    1:  y / canvas_scale
    2:  is_PI
    3:  is_IN
    4:  is_OUT
    5:  dist_left   / die_w
    6:  dist_right  / die_w
    7:  dist_bottom / die_h
    8:  dist_top    / die_h
    9:  is_endpoint
    10: fanout_norm
    11: input_delay_norm (real SDC set_input_delay / 10.0, 0.0 if none)

NET EDGE FEATURES (4): [dx, dy, manhattan, fanout_norm]

This file has NOT been run yet. Numbers below are NOT claimed --
run train.py on the correct paper split and report what you actually get.
"""
import torch
import torch.nn as nn

NODE_IN_DIM    = 12
NET_EDGE_FEATS = 4


class ModelConfig:
    def __init__(self, hidden_dim=32, net_layers=3, aggregation="sum_max",
                 lut_k=7, cell_correction=True):
        self.hidden_dim      = hidden_dim
        self.net_layers      = net_layers
        self.aggregation     = aggregation
        self.lut_k           = lut_k
        self.cell_correction = cell_correction
        self.weight_decay        = 1e-4
        self.early_stop_patience = 25

    def describe(self):
        return (f"hid={self.hidden_dim} layers={self.net_layers} "
                f"agg={self.aggregation} lut_k={self.lut_k} "
                f"corr={self.cell_correction}")


def sinusoidal_level_encoding(levels, dim, device):
    """
    Multi-frequency sinusoidal encoding of topological level (like NeRF/
    Transformer positional encodings). Combats signal decay across many
    propagation levels -- PreRoutGNN (AAAI 2024) uses this exact mechanism
    to fix the same symptom this project measured on xtea (AT error
    concentrating at specific topological-level bands; xtea has the
    deepest DAG of all 7 test designs at 97 levels). See ADR-001.

    levels: (N,) long tensor of per-pin topological level (0-indexed)
    dim: output encoding width (must be even)
    Returns: (N, dim) float tensor
    """
    assert dim % 2 == 0, "level encoding dim must be even (sin/cos pairs)"
    levels = levels.float().unsqueeze(-1)  # (N, 1)
    i = torch.arange(dim // 2, device=device, dtype=torch.float32)
    freqs = torch.exp(-i * (torch.log(torch.tensor(10000.0)) / (dim // 2)))  # (dim/2,)
    angles = levels * freqs.unsqueeze(0)  # (N, dim/2)
    enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (N, dim)
    return enc


class NetEmbedding(nn.Module):
    """
    Stage 1: embeds each pin using its own features plus aggregated
    messages from driver->sink net edges (topological, level-by-level
    handled in TimingGNN.forward via net_by_level).
    """
    def __init__(self, cfg: ModelConfig, in_dim=NODE_IN_DIM, edge_dim=NET_EDGE_FEATS,
                 level_enc_dim=16):
        super().__init__()
        self.cfg = cfg
        self.level_enc_dim = level_enc_dim
        h = cfg.hidden_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, h),
        )
        # Projects the sinusoidal level encoding into hidden_dim space so
        # it can be added residually to h0 -- keeps in_dim (raw node
        # features) untouched, so nothing elsewhere that assumes 12 raw
        # features (e.g. checkpoint in_dim, other assertions) needs to
        # change. This is purely an additive signal on top of h0.
        self.level_proj = nn.Sequential(
            nn.Linear(level_enc_dim, h),
            nn.LeakyReLU(0.2),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, h),
            nn.LeakyReLU(0.2),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, h),
        )
        self.update_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(h * 2, h), nn.LeakyReLU(0.2), nn.Linear(h, h))
            for _ in range(cfg.net_layers)
        ])

        # Virtual node: one extra learned node, implicitly connected to
        # every real pin, giving every pin a path of length <=2 to any
        # other pin's information regardless of the graph's actual DAG
        # depth. Targets designs like xtea (deepest DAG of all 7 test
        # designs at 97 levels, but sparsest per-level density -- net_layers
        # alone gives a much smaller effective receptive field there than
        # for wide, shallow designs like aes192/jpeg_encoder). Standard
        # cheap alternative to deepening message passing (which causes
        # oversmoothing on small graphs -- confirmed on this project:
        # net_layers 3->5 helped xtea +0.01 but cost spm -0.03). This adds
        # a residual broadcast signal on top of existing local message
        # passing -- does not replace or disturb it.
        self.vn_init = nn.Parameter(torch.zeros(1, h))
        self.vn_update_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, h))
            for _ in range(cfg.net_layers)
        ])

        # net delay head: predicts raw ns delay per net edge. Its VALUE is
        # used both as an auxiliary MSE-loss target AND is summed directly
        # into AT propagation for every net edge (see TimingGNN.forward,
        # `cand_at = AT[d] + nd_for_prop[ne]`) -- gradient is blocked into
        # this head from that path via .detach(), but the predicted value
        # itself still numerically affects every net-edge AT computation.
        # (An earlier version of this comment incorrectly said net delay
        # was NOT propagated into AT -- that was stale and did not match
        # the forward() code below it; corrected here, 2026-08-19.)
        #
        # Input is [driver_embedding, sink_embedding, edge_geometry_embedding].
        # Previously this head only saw the driver pin (h[net_src]) plus
        # edge geometry -- on a multi-fanout net, every sink shared the
        # identical driver-side signal and could only be distinguished by
        # edge_geo's dx/dy/manhattan, with no access to the sink's own
        # learned features (its load-side context, fanout position, etc.
        # as captured by the node MLP). Adding h[net_dst] lets the head
        # condition on the receiving pin directly, not just geometry to it.
        # NOTE: edge_mlp projects edge_dim -> h, so edge_h has width h,
        # not edge_dim. Input here must be h + h + h, not h + edge_dim.
        self.net_delay_head = nn.Sequential(
            nn.Linear(h + h + h, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, 1),
        )

    def forward(self, x, net_src, net_dst, edge_geo, net_by_level, n_nodes, pin_level):
        h0 = self.node_mlp(x)
        level_enc = sinusoidal_level_encoding(pin_level, self.level_enc_dim, x.device)
        h0 = h0 + self.level_proj(level_enc)
        h = h0.clone()
        edge_h = self.edge_mlp(edge_geo) if edge_geo.numel() > 0 else edge_geo

        vn = self.vn_init  # (1, h) -- broadcasts against (N, h) via torch broadcasting

        for i, layer in enumerate(self.update_mlps):
            # broadcast virtual node's current state into every node BEFORE
            # this layer's local net-edge aggregation, so local messages
            # this round already carry a global-graph summary from the
            # previous round.
            h = h + vn
            agg = torch.zeros_like(h)
            if net_src.numel() > 0:
                msg_in = torch.cat([h[net_src], edge_h], dim=-1)
                msg = self.msg_mlp(msg_in)
                agg.index_add_(0, net_dst, msg)
            h = layer(torch.cat([h, agg], dim=-1)) + h  # residual
            # update virtual node from a pooled summary of this round's
            # updated node embeddings, residually (so it can't destabilize
            # by overwriting -- only add to -- its running state).
            node_pool = h.mean(dim=0, keepdim=True)
            vn = vn + self.vn_update_mlps[i](node_pool)

        net_delay_pred = torch.zeros(net_src.shape[0], device=x.device)
        if net_src.numel() > 0:
            # driver + sink + edge-geometry embeddings -- see net_delay_head
            # comment above for why sink (h[net_dst]) was added.
            nd_in = torch.cat([h[net_src], h[net_dst], edge_h], dim=-1)
            net_delay_pred = self.net_delay_head(nd_in).squeeze(-1)
            net_delay_pred = torch.nn.functional.softplus(net_delay_pred) * 0.1  # keep small, ns scale

        return h, net_delay_pred


class CellLUTModule(nn.Module):
    """
    PROVEN approach: simple coef_slew/coef_load Kronecker product against
    the 7x7 liberty LUT table. Do NOT replace with an 8-table learned
    table_weight MLP -- that caused training collapse (train R2 stuck
    at 0.74) in a prior run and must not be reintroduced without a fix.
    """
    def __init__(self, cfg: ModelConfig, in_dim, k=7):
        super().__init__()
        self.k = k
        h = cfg.hidden_dim
        self.coef_mlp = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, 2 * k),  # k coeffs for slew axis, k coeffs for load axis
        )
        self.correction_scale = 0.2
        self.correction_mlp = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, 1),
        )

    def forward(self, feat, lut_delay, lut_slew):
        """
        feat: (E, in_dim) per-edge conditioning features
        lut_delay, lut_slew: (E, k, k)
        Returns: delay_pred (E,), slew_pred (E,)
        """
        E = feat.shape[0]
        if E == 0:
            z = torch.zeros(0, device=feat.device)
            return z, z

        coefs = self.coef_mlp(feat)
        coef_slew = torch.softmax(coefs[:, :self.k], dim=-1)   # (E,k)
        coef_load = torch.softmax(coefs[:, self.k:], dim=-1)   # (E,k)

        # Kronecker-style weighted sum over the LUT grid
        weight = torch.einsum('ei,ej->eij', coef_slew, coef_load)  # (E,k,k)
        delay_base = (weight * lut_delay).sum(dim=(1, 2))
        slew_base  = (weight * lut_slew).sum(dim=(1, 2))

        correction = self.correction_mlp(feat).squeeze(-1) * self.correction_scale
        delay_pred = delay_base + correction
        slew_pred  = slew_base

        return delay_pred, slew_pred


class TimingGNN(nn.Module):
    def __init__(self, cfg: ModelConfig, n_gates, in_dim=None, edge_dim=None):
        """
        in_dim / edge_dim: if None, MUST be passed explicitly by the caller
        based on the actual data's tensor width (g["x"].shape[1] and
        g["edge_geo"].shape[1]). Do NOT rely on the NODE_IN_DIM /
        NET_EDGE_FEATS module constants for this -- they reflect only the
        interactive_train_real.py sample pipeline (12 / 4). A real local
        pipeline (preprocess.py) may produce a different feature count
        (e.g. 11), and a hardcoded mismatch crashes the first Linear layer.
        This was a real bug, caught by an actual local run, not theoretical.
        """
        super().__init__()
        if in_dim is None or edge_dim is None:
            raise ValueError(
                "TimingGNN requires explicit in_dim and edge_dim matching "
                "your actual data (g['x'].shape[1], g['edge_geo'].shape[1]). "
                "Do not rely on module-level defaults -- they only match "
                "the sample pipeline, not necessarily your real preprocess.py."
            )
        self.cfg = cfg
        h = cfg.hidden_dim

        self.net_embed = NetEmbedding(cfg, in_dim=in_dim, edge_dim=edge_dim)

        # cell LUT conditioning features: [h_in, gate_onehot, slew_in]
        # +1 for the propagated input slew fed as explicit LUT query coordinate.
        # The liberty LUT is a 2D table indexed by (input_slew, output_load).
        # Previously coef_mlp had to infer the slew query from h[cell_src] alone
        # (no actual slew value visible at LUT call time). Now the actual physical
        # slew at each gate input -- propagated level-by-level -- is passed in
        # directly. Cell delay computation is moved inside the propagation loop
        # so SLEW[cell_src] reflects the true running state, not a static guess.
        cell_in_dim = h + n_gates + 1
        self.cell_lut = CellLUTModule(cfg, in_dim=cell_in_dim, k=cfg.lut_k)

        # slew propagation degradation MLP (replaces hardcoded 1.05x / 0.9x multiplier)
        self.slew_degrade_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Learned per-node initial AT bias, applied directly to the raw
        # node feature vector (not the embedded h -- this needs to run
        # before/independent of the net-embedding stage).
        #
        # WHY THIS EXISTS: AT propagation previously started every node
        # at a hardcoded 0.0 (torch.zeros(n_nodes)) and only updates a
        # node's AT via index_reduce on the SINK side of a net/cell edge.
        # Primary-input pins are, by construction in parse_real_design.py
        # (signal_driver vs signal_sinks), ALWAYS a driver and NEVER a
        # sink -- confirmed directly: 0 of 9 PI pins in uni_counter4b
        # ever appear as an edge destination. That means a PI's predicted
        # AT was structurally frozen at exactly 0 forever, no matter how
        # much or how well the model trained, even though real STA
        # arrival time at a PI equals its SDC set_input_delay (uniformly
        # 2.0ns for uni_counter4b, 5.206ns for xtea -- both nonzero).
        #
        # This head lets every node predict a learned starting AT informed
        # by its own features -- most directly by input_delay_norm
        # (feature index 11, added alongside this fix), which is the
        # real SDC value made an explicit model input rather than
        # something the model would otherwise have no way to access.
        # softplus keeps the output non-negative (arrival times can't be
        # negative). For internal/gate nodes, propagation via index_reduce
        # amax naturally dominates this small init once real accumulated
        # delay exceeds it, so this does not disturb previously-correct
        # behavior for non-PI nodes.
        self.at_init_mlp = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
        )

    def forward(self, g, groundtruth=False):
        """
        groundtruth: if True (use during training), AT/SLEW propagation
        uses TRUE net_delay/cell_delay values (g["nd_true"]/g["cd_true"])
        instead of the model's own predictions, when available. This is
        teacher forcing -- matches the paper's actual training approach
        (an established project learning I had NOT implemented, which
        was the real cause of per-design instability: early-training
        noisy delay predictions compound across DAG levels via max-
        propagation, hitting small designs like spm hardest since they
        have fewer pins to average the noise out over).

        At eval/inference, ALWAYS use groundtruth=False (the default) --
        true labels aren't available for genuine inference anyway, and
        this is what the reported test R2 numbers must reflect.
        """
        x           = g["x"]
        net_src     = g["net_src"]
        net_dst     = g["net_dst"]
        cell_src    = g["cell_src"]
        cell_dst    = g["cell_dst"]
        edge_geo    = g["edge_geo"]
        gate_onehot = g["gate_onehot"]
        lut_delay   = g["lut_delay"]
        lut_slew    = g["lut_slew"]
        net_by_level  = g["net_by_level"]
        cell_by_level = g["cell_by_level"]
        pin_level = g["pin_level"]
        n_nodes = g["n_nodes"]
        device = x.device

        h, net_delay_pred = self.net_embed(x, net_src, net_dst, edge_geo, net_by_level, n_nodes, pin_level)

        # Cell delay / slew are computed level-by-level inside the propagation
        # loop below, using the currently propagated SLEW as the LUT slew-axis
        # query. Per-level (ce, cd_lvl) pairs are collected in lists and
        # assembled into cell_delay_pred via a single scatter after the loop.
        # This avoids the O(L × n_cell) chained-scatter memory pattern that
        # caused CUDA OOM on large designs (each out-of-place scatter in the
        # loop kept a new n_cell tensor alive in autograd until backward).
        n_cell = cell_src.numel()
        cell_delay_pred = torch.zeros(n_cell, device=device)
        _ce_lvl_parts: list = []
        _cd_lvl_parts: list = []

        # ---- Stage 2: level-ordered AT / SLEW propagation (teacher forcing off
        # at inference -- uses predicted delays, not ground truth) ----
        # AT now starts from a learned per-node value instead of hardcoded
        # zero -- see at_init_mlp comment above for why this is necessary,
        # not optional, for primary-input pins to ever predict correctly.
        at_init_raw = self.at_init_mlp(x).squeeze(-1)
        AT = torch.nn.functional.softplus(at_init_raw)
        SLEW = torch.full((n_nodes,), 0.15, device=device)

        # NOTE: in-place index_reduce_ across multiple levels breaks autograd
        # (RuntimeError: modified by an inplace operation) because each
        # level's write bumps the tensor's version counter out from under
        # the saved-for-backward buffers of earlier levels. Non-inplace
        # index_reduce + torch.maximum avoids this -- caught by smoke test.
        levels = g["levels"]
        for lvl in levels:
            ne = net_by_level[lvl]
            if ne.numel() > 0:
                d = net_src[ne]
                s = net_dst[ne]
                # .detach() when self-predicting: AT still uses the model's
                # own predicted delay VALUE (so AT quality genuinely
                # reflects delay-head quality), but this stops AT's loss
                # gradient from flowing back into net_delay_pred through
                # index_reduce's sparse amax backward -- that path was
                # fighting net_delay_pred's own direct MSE supervision
                # (confirmed root cause of net_delay R2 collapsing to
                # -4093 on spm once scheduled sampling made groundtruth=False
                # the common case; see 20260814_191126_train20_test7 log).
                nd_for_prop = net_delay_pred.detach()
                if groundtruth and "nd_true" in g and g["nd_true"].numel() == net_delay_pred.numel():
                    nd_for_prop = g["nd_true"]
                cand_at = AT[d] + nd_for_prop[ne]
                at_update = AT.index_reduce(0, s, cand_at, "amax", include_self=True)
                AT = torch.maximum(AT, at_update)

                wire_len = (edge_geo[ne][:, 2] if edge_geo.numel() > 0
                            else torch.zeros(ne.shape[0], device=device))
                degrade = self.slew_degrade_mlp(wire_len.unsqueeze(-1)).squeeze(-1)
                cand_slew = SLEW[d] * (1.0 + 0.2 * degrade)
                slew_update = SLEW.index_reduce(0, s, cand_slew, "amax", include_self=True)
                SLEW = torch.maximum(SLEW, slew_update)

            ce = cell_by_level[lvl]
            if ce.numel() > 0:
                i = cell_src[ce]
                o = cell_dst[ce]
                # Compute cell delay HERE using the currently propagated SLEW as
                # the LUT input-slew coordinate. At level L, SLEW[i] has been
                # updated by every net and cell arc at levels < L, so it is the
                # actual physical slew arriving at pin i -- exactly what the
                # liberty 2D LUT indexes on. This is the key architectural fix:
                # the old pre-loop version had no slew context (SLEW was 0.15
                # everywhere) and coef_mlp had to guess the table row blindly.
                # .detach() prevents gradient from back-propagating through the
                # entire SLEW propagation chain (same rationale as nd/cd_for_prop
                # -- stops AT's sparse-amax backward from corrupting LUT params).
                slew_in = SLEW[i].detach().unsqueeze(-1)             # (|ce|, 1)
                cell_feat_lvl = torch.cat([h[i], gate_onehot[ce], slew_in], dim=-1)
                cd_lvl, cs_lvl = self.cell_lut(cell_feat_lvl, lut_delay[ce], lut_slew[ce])
                # Collect for later single-scatter assembly (avoids OOM).
                _ce_lvl_parts.append(ce)
                _cd_lvl_parts.append(cd_lvl)

                cd_for_prop = cd_lvl.detach()
                if groundtruth and "cd_true" in g and g["cd_true"].numel() == n_cell:
                    cd_for_prop = g["cd_true"][ce]
                cand_at = AT[i] + cd_for_prop.clamp(min=0.001)
                at_update = AT.index_reduce(0, o, cand_at, "amax", include_self=True)
                AT = torch.maximum(AT, at_update)

                cand_slew = cs_lvl.clamp(min=0.005)
                slew_update = SLEW.index_reduce(0, o, cand_slew, "amax", include_self=True)
                SLEW = torch.maximum(SLEW, slew_update)

        # Single scatter: O(n_cell) instead of O(L × n_cell) chain.
        # Backward through cat routes gradients correctly to each cd_lvl.
        if _ce_lvl_parts:
            all_ce = torch.cat(_ce_lvl_parts)
            all_cd = torch.cat(_cd_lvl_parts)
            cell_delay_pred = cell_delay_pred.scatter(0, all_ce, all_cd)

        return {
            "AT": AT,
            "SLEW": SLEW,
            "net_delay": net_delay_pred,
            "cell_delay": cell_delay_pred,
        }