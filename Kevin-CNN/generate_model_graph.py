"""
generate_model_graph.py
───────────────────────
Introspects FullADCTModel layer-by-layer (via torch hooks when torch is
available, or analytic shape simulation otherwise), builds a directed
NetworkX graph, and renders a publication-quality PNG with matplotlib.

Usage:
    python generate_model_graph.py            # standalone
    # OR call from adct_model_v3.py:
    from generate_model_graph import generate_model_graph
    generate_model_graph(out_dir="training_logs")

Output:  training_logs/FullADCTModel_graph.png
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
#  Hyperparameters — single source of truth (mirrors adct_model_v3.py)
# ─────────────────────────────────────────────────────────────────────────────
HPARAMS = {
    "depth":       3,
    "filters":     64,
    "kernel_size": 3,
    "padding":     1,
    "dropout":     0.4,
    "fc_hidden":   128,
    "eeg_flat":    21504,   # 64 × 7 × 48
    "bio_flat":    704,     # 64 × 11
    "fusion":      23616,   # 21504 + 704×3
    "lr":          0.0005,
    "batch":       32,
    "epochs":      30,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Shape helpers
# ─────────────────────────────────────────────────────────────────────────────
def _co(dim, k=3, p=1, s=1):
    return math.floor((dim + 2 * p - k) / s + 1)

def _po(dim, k=2):
    return dim // k


# ─────────────────────────────────────────────────────────────────────────────
#  Trace shapes
# ─────────────────────────────────────────────────────────────────────────────
def _trace_torch(hparams):
    """Hook-based live tracing — used when torch is importable."""
    import torch
    import torch.nn as nn

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            f, k, p = hparams["filters"], hparams["kernel_size"], hparams["padding"]
            d = hparams["depth"]
            eeg_layers = []
            for i in range(d):
                eeg_layers += [nn.Conv2d(1 if i == 0 else f, f, k, padding=p), nn.ReLU(), nn.MaxPool2d(2)]
            eeg_layers.append(nn.Flatten())
            self.eeg_branch = nn.Sequential(*eeg_layers)

            def bio():
                layers = []
                for i in range(d):
                    layers += [nn.Conv1d(3 if i == 0 else f, f, k, padding=p), nn.ReLU(), nn.MaxPool1d(2)]
                layers.append(nn.Flatten())
                return nn.Sequential(*layers)

            self.act_branch = bio()
            self.pup_branch = bio()
            self.spc_branch = bio()
            self.fc = nn.Sequential(
                nn.Linear(hparams["fusion"], hparams["fc_hidden"]), nn.ReLU(),
                nn.Dropout(hparams["dropout"]), nn.Linear(hparams["fc_hidden"], 1),
            )

        def forward(self, eeg, act, pup, spc):
            b1 = self.eeg_branch(eeg.unsqueeze(1))
            b2, b3, b4 = self.act_branch(act), self.pup_branch(pup), self.spc_branch(spc)
            return self.fc(torch.cat((b1, b2, b3, b4), dim=1))

    model = _Model()
    model.eval()
    traces = {}

    def _run_branch(branch_mod, x_dummy):
        rows = []
        shp = tuple(x_dummy.shape[1:])
        rows.append(("Input", "×".join(str(s) for s in shp)))
        x = x_dummy
        for layer in branch_mod:
            with torch.no_grad():
                x = layer(x)
            shp = tuple(x.shape[1:])
            tag = type(layer).__name__
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                tag = f"{tag}\n(k={layer.kernel_size[0]}, p={layer.padding[0]}, f={layer.out_channels})\n+ReLU"
            elif isinstance(layer, (nn.MaxPool2d, nn.MaxPool1d)):
                ks = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
                tag = f"{tag}\n(k={ks})"
            rows.append((tag, "×".join(str(s) for s in shp) if shp else "flat"))
        return rows

    traces["EEG Branch\n(Conv2d)"]      = _run_branch(model.eeg_branch, torch.zeros(1, 1, 60, 384))
    traces["Action Branch\n(Conv1d)"]   = _run_branch(model.act_branch,  torch.zeros(1, 3, 90))
    traces["Pupil Branch\n(Conv1d)"]    = _run_branch(model.pup_branch,  torch.zeros(1, 3, 90))
    traces["Speech Branch\n(Conv1d)"]   = _run_branch(model.spc_branch,  torch.zeros(1, 3, 90))

    # Fusion
    x = torch.cat([torch.zeros(1, hparams["eeg_flat"])] + [torch.zeros(1, hparams["bio_flat"])] * 3, dim=1)
    fuse = [("Concat\n(×4 branches)", str(x.shape[1]))]
    for layer in model.fc:
        with torch.no_grad():
            x = layer(x)
        shp = tuple(x.shape[1:])
        tag = type(layer).__name__
        if isinstance(layer, nn.Linear):
            tag = f"Linear\n({layer.in_features}→{layer.out_features})"
        elif isinstance(layer, nn.Dropout):
            tag = f"Dropout\np={layer.p}"
        fuse.append((tag, "×".join(str(s) for s in shp) if shp else "1"))
    fuse.append(("Output\n(score)", "1"))
    traces["Fusion Head"] = fuse
    return traces


def _trace_analytic(hparams):
    """Pure-math shape simulation — no torch needed."""
    f = hparams["filters"]
    d = hparams["depth"]

    # EEG
    eeg, h, w = [("Input", "1×60×384")], 60, 384
    for i in range(d):
        h, w = _co(h), _co(w)
        eeg.append((f"Conv2d\n(k=3, p=1, f={f})\n+ReLU", f"{f}×{h}×{w}"))
        h, w = _po(h), _po(w)
        eeg.append((f"MaxPool2d\n(k=2)", f"{f}×{h}×{w}"))
    eeg.append(("Flatten", f"{f*h*w:,}"))

    # 1-D branches
    bio_traces = {}
    for bname in ["Action Branch\n(Conv1d)", "Pupil Branch\n(Conv1d)", "Speech Branch\n(Conv1d)"]:
        rows, l = [("Input", "3×90")], 90
        for i in range(d):
            l = _co(l)
            rows.append((f"Conv1d\n(k=3, p=1, f={f})\n+ReLU", f"{f}×{l}"))
            l = _po(l)
            rows.append((f"MaxPool1d\n(k=2)", f"{f}×{l}"))
        rows.append(("Flatten", f"{f*l:,}"))
        bio_traces[bname] = rows

    # Fusion
    fuse = [
        ("Concat\n(×4 branches)",               f"{hparams['fusion']:,}"),
        (f"Linear\n({hparams['fusion']:,}→{hparams['fc_hidden']})\n+ReLU",
                                                 f"{hparams['fc_hidden']}"),
        (f"Dropout\np={hparams['dropout']}",     f"{hparams['fc_hidden']}"),
        ("Linear\n(128→1)",                      "1"),
        ("Output\n(score)",                      "1"),
    ]

    return {
        "EEG Branch\n(Conv2d)":      eeg,
        **bio_traces,
        "Fusion Head":               fuse,
    }


def trace_model(hparams):
    try:
        import torch
        traces = _trace_torch(hparams)
        print("  (live torch forward-pass tracing)")
    except Exception:
        traces = _trace_analytic(hparams)
        print("  (analytic shape simulation)")
    return traces


# ─────────────────────────────────────────────────────────────────────────────
#  Build directed graph
# ─────────────────────────────────────────────────────────────────────────────
BRANCH_ORDER = [
    "EEG Branch\n(Conv2d)",
    "Action Branch\n(Conv1d)",
    "Pupil Branch\n(Conv1d)",
    "Speech Branch\n(Conv1d)",
    "Fusion Head",
]

BRANCH_COL = {
    "EEG Branch\n(Conv2d)":    0,
    "Action Branch\n(Conv1d)": 1,
    "Pupil Branch\n(Conv1d)":  2,
    "Speech Branch\n(Conv1d)": 3,
    "Fusion Head":             1.5,
}

def build_graph(traces):
    G, meta = nx.DiGraph(), {}
    tail = {}

    for branch in BRANCH_ORDER:
        nodes = traces[branch]
        prev  = None
        for i, (lbl, shp) in enumerate(nodes):
            nid  = f"{branch}__{i}"
            kind = _classify(lbl)
            G.add_node(nid)
            meta[nid] = {
                "label":  lbl,
                "shp":    shp,
                "branch": branch,
                "kind":   kind,
                "col":    BRANCH_COL[branch],
                "row":    i,
            }
            if prev:
                G.add_edge(prev, nid)
            prev = nid
        tail[branch] = prev

    # cross-branch edges → concat node
    concat_nid = "Fusion Head__0"
    for b in BRANCH_ORDER[:4]:
        G.add_edge(tail[b], concat_nid)

    return G, meta


def _classify(lbl):
    l = lbl.lower()
    if "input"   in l: return "input"
    if "conv2d"  in l: return "conv2d"
    if "conv1d"  in l: return "conv1d"
    if "pool2d"  in l or "pool1d" in l or "maxpool" in l: return "pool"
    if "flatten" in l: return "flatten"
    if "concat"  in l: return "concat"
    if "linear"  in l: return "fc"
    if "dropout" in l: return "dropout"
    if "output"  in l: return "output"
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────
COL_SPACING = 3.6
ROW_SPACING = 1.6

def compute_layout(traces, meta):
    max_branch_rows = max(
        len(traces[b]) for b in BRANCH_ORDER[:4]
    )
    pos = {}
    for nid, m in meta.items():
        col = BRANCH_COL[m["branch"]]
        row = m["row"]
        if m["branch"] == "Fusion Head":
            row = max_branch_rows + 1 + row
        pos[nid] = (col * COL_SPACING, -row * ROW_SPACING)
    return pos


# ─────────────────────────────────────────────────────────────────────────────
#  Colours
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "input":   ("#1e3a5f", "#4a90d9"),
    "conv2d":  ("#1a4731", "#2ecc71"),
    "conv1d":  ("#3b1c5a", "#9b59b6"),
    "pool":    ("#2d2d2d", "#888888"),
    "flatten": ("#3d2b00", "#f39c12"),
    "concat":  ("#1a1a2e", "#e74c3c"),
    "fc":      ("#2c1654", "#c39bd3"),
    "dropout": ("#1c1c1c", "#666666"),
    "output":  ("#1b2631", "#f1c40f"),
    "default": ("#222222", "#aaaaaa"),
}

BRANCH_HDR_COLOR = {
    "EEG Branch\n(Conv2d)":    ("#1e4d6b", "#4a90d9"),
    "Action Branch\n(Conv1d)": ("#3b1c5a", "#9b59b6"),
    "Pupil Branch\n(Conv1d)":  ("#3b1c5a", "#9b59b6"),
    "Speech Branch\n(Conv1d)": ("#3b1c5a", "#9b59b6"),
    "Fusion Head":             ("#1a1a2e", "#e74c3c"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Render
# ─────────────────────────────────────────────────────────────────────────────
BOX_W = 2.8
BOX_H = 1.2
RPAD  = 0.16

def render(G, meta, pos, traces, out_path, hparams):
    # Dynamic figure sizing
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    x_range = max(all_x) - min(all_x) + BOX_W + 2.0
    y_range = max(all_y) - min(all_y) + BOX_H + 5.0
    fig_w = max(18.0, x_range + 1.5)
    fig_h = max(20.0, y_range + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.set_xlim(min(all_x) - BOX_W, max(all_x) + BOX_W)
    ax.set_ylim(min(all_y) - BOX_H - 2.5, max(all_y) + BOX_H + 3.5)
    ax.axis("off")

    # ── Edges ─────────────────────────────────────────────────────────────────
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        lateral = abs(meta[u]["col"] - meta[v]["col"]) > 0.1
        color  = "#e74c3c" if lateral else "#4a6fa5"
        rad    = 0.22     if lateral else 0.0
        ax.annotate(
            "", xy=(x1, y1 + BOX_H / 2), xytext=(x0, y0 - BOX_H / 2),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.6,
                connectionstyle=f"arc3,rad={rad}", mutation_scale=14,
            ), zorder=2,
        )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for nid, m in meta.items():
        x, y   = pos[nid]
        bg, bd = PALETTE.get(m["kind"], PALETTE["default"])

        box = mpatches.FancyBboxPatch(
            (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle=f"round,pad={RPAD}",
            linewidth=1.8, edgecolor=bd, facecolor=bg, zorder=3,
        )
        ax.add_patch(box)

        # Layer name
        ax.text(x, y + 0.18, m["label"],
                ha="center", va="center",
                fontsize=6.8, color="#e8e8e8",
                fontfamily="monospace", fontweight="bold", zorder=4,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="#000")])

        # Shape badge
        ax.text(x, y - 0.34, f"▸ {m['shp']}",
                ha="center", va="center",
                fontsize=6.3, color=bd,
                fontfamily="monospace", zorder=4)

    # ── Branch headers ─────────────────────────────────────────────────────────
    for branch in BRANCH_ORDER:
        branch_nodes = [nid for nid, m in meta.items() if m["branch"] == branch]
        top_y = max(pos[n][1] for n in branch_nodes) + BOX_H / 2 + 0.45
        cx    = BRANCH_COL[branch] * COL_SPACING
        bg, bd = BRANCH_HDR_COLOR[branch]
        short  = branch.replace("\n", " ")
        ax.text(cx, top_y, short,
                ha="center", va="bottom",
                fontsize=9.5, color="#ffffff",
                fontfamily="monospace", fontweight="bold",
                bbox=dict(facecolor=bg, edgecolor=bd,
                          boxstyle="round,pad=0.35", linewidth=1.4),
                zorder=5)

    # ── Title ─────────────────────────────────────────────────────────────────
    title_y = max(all_y) + BOX_H / 2 + 2.2
    ax.text((min(all_x) + max(all_x)) / 2, title_y,
            "FullADCTModel — Architecture Graph",
            ha="center", va="bottom",
            fontsize=15, color="#ffffff",
            fontfamily="monospace", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground="#000")])

    sub = (f"depth={hparams['depth']}  filters={hparams['filters']}  "
           f"kernel={hparams['kernel_size']}  padding={hparams['padding']}  "
           f"dropout={hparams['dropout']}  fc_hidden={hparams['fc_hidden']}  "
           f"lr={hparams['lr']}  batch={hparams['batch']}  epochs={hparams['epochs']}")
    ax.text((min(all_x) + max(all_x)) / 2, title_y - 0.75, sub,
            ha="center", va="bottom",
            fontsize=7.5, color="#888888", fontfamily="monospace")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        ("Input",       "input"),
        ("Conv2d",      "conv2d"),
        ("Conv1d",      "conv1d"),
        ("MaxPool",     "pool"),
        ("Flatten",     "flatten"),
        ("Concat",      "concat"),
        ("Linear (FC)", "fc"),
        ("Dropout",     "dropout"),
        ("Output",      "output"),
    ]
    lx0  = min(all_x) - BOX_W / 2
    ly0  = min(all_y) - BOX_H / 2 - 1.8
    lw, lh = 2.2, 0.5
    ax.text(lx0, ly0 + 0.6, "Layer types:",
            fontsize=8.5, color="#aaaaaa",
            fontfamily="monospace", va="center")
    for i, (name, kind) in enumerate(legend_items):
        bg, bd = PALETTE[kind]
        col = i % 5
        row = i // 5
        lx  = lx0 + col * (lw + 0.3)
        ly  = ly0 - row * (lh + 0.2)
        p   = mpatches.FancyBboxPatch(
            (lx, ly - lh / 2), lw, lh,
            boxstyle="round,pad=0.06",
            linewidth=1.2, edgecolor=bd, facecolor=bg, zorder=5,
        )
        ax.add_patch(p)
        ax.text(lx + lw / 2, ly, name,
                ha="center", va="center",
                fontsize=7, color="#e8e8e8",
                fontfamily="monospace", zorder=6)

    fig.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor="#0d0d0d", edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────
def generate_model_graph(out_dir="training_logs", hparams=None):
    """
    Generate and save the FullADCTModel architecture graph.
    Call this from adct_model_v3.py:

        from generate_model_graph import generate_model_graph
        generate_model_graph(out_dir="training_logs", hparams=HPARAMS)
    """
    if hparams is None:
        hparams = HPARAMS
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "FullADCTModel_graph.png")

    print("\nGenerating model architecture graph...")
    traces       = trace_model(hparams)
    G, meta      = build_graph(traces)
    pos          = compute_layout(traces, meta)
    render(G, meta, pos, traces, out_path, hparams)
    return out_path


if __name__ == "__main__":
    path = generate_model_graph()
    print(f"Done → {path}")