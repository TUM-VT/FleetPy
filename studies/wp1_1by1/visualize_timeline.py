#!/usr/bin/env python3
"""
Temporal replay animation for FleetPy AGIMO WP1 simulation results.

Reads 2-0_op-stats.csv and 1_user-stats.csv, reconstructs vehicle positions
and request states frame by frame, and renders an animated map with a
scrolling event log and time bar.

Usage (from repo root):
    python studies/wp1_1by1/visualize_timeline.py [result_dir] [--step 30] [--speed 150] [--save]
"""
import json
import logging
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

LOG = logging.getLogger(__name__)

STUDY_DIR = Path(__file__).parent
REPO_ROOT  = STUDY_DIR.parent.parent

# ── run configuration ─────────────────────────────────────────────────────────
RESULT_DIR    = 'studies/wp1_1by1/results/grid_l1_w1_hubs1_cell100_10pkm2h_0.5dir_all_normal_seed0_dtd_r0.25/'   # Path to a result dir containing 00_config.json + op/user stats CSVs.
                     # Example: 'studies/wp1_1by1/results/grid_l1_w1_hubs1_cell100_10rph_0.5dir_all_normal_seed0_dtd_n5'
                     # Leave '' to auto-pick the first subdirectory of studies/wp1_1by1/results/.
FRAME_STEP    = 30     # seconds per animation frame
INTERVAL_MS   = 300    # milliseconds between frames during playback
SAVE_GIF      = False  # True = write timeline.gif instead of interactive window
RETURN_TO_HUB = False  # True = snap idle vehicles to nearest hub between trips

# ── colour palette ───────────────────────────────────────────────────────────
BG      = "#12122a"
PANEL   = "#1a1a38"

VEH_COLORS = {
    "idle":          "#888888",
    "route_empty":   "#aec7e8",
    "route_loaded":  "#1f77b4",
    "boarding":      "#2ca02c",
}
RQ_WAIT  = "#d62728"
RQ_BOARD = "#ffd700"

# TODO read through
# ── data helpers ─────────────────────────────────────────────────────────────

def _node(pos_str: str) -> int:
    """'116;-1;-1' → 116"""
    return int(str(pos_str).split(";")[0])


def _traj(traj_str) -> list:
    """'116:0;115:12.0;…' → [(116, 0.0), (115, 12.0), …]"""
    if not isinstance(traj_str, str) or not traj_str.strip():
        return []
    out = []
    for part in traj_str.split(";"):
        n, t = part.split(":")
        out.append((int(n), float(t)))
    return out


def find_result_dir(base: Path) -> Path:
    if not base.exists():
        raise FileNotFoundError(f"Results dir not found: {base}")
    dirs = sorted(d for d in base.iterdir() if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No subdirectories in {base}")
    return dirs[0]


def load_nodes(nw_name: str):
    nodes_f = REPO_ROOT / "data" / "networks" / nw_name / "base" / "nodes.csv"
    arr = np.atleast_2d(np.loadtxt(nodes_f, delimiter=",", skiprows=1, usecols=(0, 2, 3)))
    coords = {int(r[0]): (float(r[1]), float(r[2])) for r in arr}

    hubs_f = REPO_ROOT / "data" / "networks" / nw_name / "base" / "hubs.csv"
    hubs = pd.read_csv(hubs_f)["node_index"].tolist() if hubs_f.exists() else []
    return coords, hubs


def load_data(result_dir: Path):
    with open(result_dir / "00_config.json") as f:
        cfg = json.load(f)
    sp       = cfg.get("scenario_parameters", cfg)
    nw_name  = sp.get("network_name") or sp.get("nw_name", "")
    end_time = float(sp.get("end_time", 3600))

    ops = pd.read_csv(result_dir / "2-0_op-stats.csv")
    ops["start_time"] = pd.to_numeric(ops["start_time"])
    ops["end_time"]   = pd.to_numeric(ops["end_time"])
    ops["occupancy"]  = (pd.to_numeric(ops["occupancy"], errors="coerce")
                           .fillna(0).astype(int))
    ops["start_node"] = ops["start_pos"].apply(_node)
    ops["end_node"]   = ops["end_pos"].apply(_node)
    ops["traj"]       = ops["trajectory"].apply(_traj)

    req = pd.read_csv(result_dir / "1_user-stats.csv")
    req["start_node"]   = req["start"].apply(_node)
    req["end_node"]     = req["end"].apply(_node)
    req["pickup_time"]  = pd.to_numeric(req["pickup_time"],  errors="coerce")
    req["dropoff_time"] = pd.to_numeric(req["dropoff_time"], errors="coerce")
    req["vehicle_id"]   = pd.to_numeric(req["vehicle_id"],   errors="coerce")
    # boarding node: pickup_location when the vehicle meets them at a different node
    req["pu_node"] = req.apply(
        lambda r: _node(r["pickup_location"])
        if pd.notna(r.get("pickup_location")) and str(r.get("pickup_location")).strip() not in ("", "nan")
        else r["start_node"], axis=1)
    req["do_node"] = req.apply(
        lambda r: _node(r["dropoff_location"])
        if pd.notna(r.get("dropoff_location")) and str(r.get("dropoff_location")).strip() not in ("", "nan")
        else r["end_node"], axis=1)

    return ops, req, nw_name, end_time


# ── vehicle state reconstruction ─────────────────────────────────────────────

def build_segments(ops: pd.DataFrame) -> dict:
    """dict: vehicle_id → list of segment dicts, sorted by start_time."""
    segs: dict = {}
    for _, row in ops.iterrows():
        vid = int(row["vehicle_id"])
        segs.setdefault(vid, []).append({
            "start": row["start_time"],
            "end":   row["end_time"],
            "status": row["status"],
            "occ":   row["occupancy"],
            "snode": int(row["start_node"]),
            "enode": int(row["end_node"]),
            "traj":  row["traj"],
        })
    for vid in segs:
        segs[vid].sort(key=lambda s: s["start"])
    return segs


def nearest_hub(node, hubs, coords):
    """Return the hub node closest (Euclidean) to *node*, or *node* if no hubs."""
    if not hubs or node not in coords:
        return node
    x, y = coords[node]
    return min(
        (h for h in hubs if h in coords),
        key=lambda h: (coords[h][0] - x) ** 2 + (coords[h][1] - y) ** 2,
        default=node,
    )


def veh_state_at(segs: list, t: float,
                 hubs=(), coords=None, return_to_hub: bool = False):
    """Return (node_idx, color_key) for one vehicle at simulation time t.

    When *return_to_hub* is True, vehicles snap to the nearest hub whenever
    they are in an idle (between-assignment) period instead of waiting at the
    last drop-off node.
    """
    def _idle_node(node):
        if return_to_hub and hubs and coords:
            return nearest_hub(node, hubs, coords)
        return node

    last_node = segs[0]["snode"] if segs else None
    for seg in segs:
        if t < seg["start"]:
            return _idle_node(last_node), "idle"
        if seg["start"] <= t < seg["end"]:
            if seg["status"] == "boarding":
                return seg["snode"], "boarding"
            traj = seg["traj"]
            if not traj:
                ck = "route_loaded" if seg["occ"] > 0 else "route_empty"
                return seg["snode"], ck
            cur = traj[0][0]
            for node, node_t in traj:
                if node_t <= t:
                    cur = node
                else:
                    break
            ck = "route_loaded" if seg["occ"] > 0 else "route_empty"
            return cur, ck
        last_node = seg["enode"]
    return _idle_node(last_node), "idle"


# ── event log ────────────────────────────────────────────────────────────────

def build_events(req: pd.DataFrame) -> list:
    """Sorted list of (time, message) for all request lifecycle events."""
    events = []
    for _, rq in req.iterrows():
        rid  = int(rq["request_id"])
        grp  = rq.get("user_group", "?")
        vid  = int(rq["vehicle_id"]) if pd.notna(rq.get("vehicle_id")) else "?"
        wait = float(rq.get("t_outside_wait_s") or 0) + float(rq.get("t_home_wait_s") or 0)
        dis  = float(rq.get("disutility_eur") or 0)

        walk_tag = ""
        if rq["pu_node"] != rq["start_node"]:
            walk_tag = f" [walk→n{rq['pu_node']}]"
        events.append((float(rq["rq_time"]),
                       f"rq_{rid:02d} ({grp}) submitted{walk_tag}"))
        if pd.notna(rq["pickup_time"]):
            events.append((float(rq["pickup_time"]),
                           f"V{vid} picks up rq_{rid:02d}  waited {wait:.0f}s"))
        if pd.notna(rq["dropoff_time"]):
            do_tag = f" [walk←n{rq['do_node']}]" if rq["do_node"] != rq["end_node"] else ""
            events.append((float(rq["dropoff_time"]),
                           f"V{vid} drops  rq_{rid:02d}  {dis:.2f}€{do_tag}"))

    events.sort(key=lambda e: e[0])
    return events


# ── animation ────────────────────────────────────────────────────────────────

def make_animation(ops, req, coords, hubs, end_time,
                   frame_step=30, interval_ms=150, return_to_hub: bool = False):
    segs    = build_segments(ops)
    vids    = sorted(segs.keys())
    events  = build_events(req)

    # Log all events upfront (the chronological "simulation log")
    for t, msg in events:
        LOG.info("t=%6.0fs | %s", t, msg)
    n_served = int(req["dropoff_time"].notna().sum())
    avg_wait = (req.get("t_outside_wait_s", pd.Series(dtype=float)).fillna(0)
                + req.get("t_home_wait_s",    pd.Series(dtype=float)).fillna(0)).mean()
    avg_dis  = req["disutility_eur"].fillna(0).mean()
    LOG.info("=== %d served | avg wait %.1fs | avg disutility %.2f€ ===",
             n_served, avg_wait, avg_dis)

    # Static node arrays
    all_nodes = sorted(coords)
    nxy = np.array([coords[n] for n in all_nodes])
    hub_xy = np.array([coords[h] for h in hubs if h in coords]) if hubs else np.empty((0, 2))
    xs, ys = nxy[:, 0], nxy[:, 1]
    margin = max((xs.max() - xs.min()), (ys.max() - ys.min())) * 0.08 + 30

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[3, 1],
        hspace=0.06, wspace=0.04,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_log = fig.add_subplot(gs[0, 1])

    for ax in (ax_map, ax_log):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor("#334466")
        ax.tick_params(colors="#99aacc")

    # ── map: static elements ──────────────────────────────────────────────────
    ax_map.scatter(nxy[:, 0], nxy[:, 1], c="#2a2a55", s=18, zorder=1)
    if hub_xy.shape[0]:
        ax_map.scatter(hub_xy[:, 0], hub_xy[:, 1],
                       c="#ff9500", s=220, marker="*", zorder=3,
                       edgecolors="white", linewidths=0.5)
    ax_map.set_aspect("equal")
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.set_xlim(xs.min() - margin, xs.max() + margin)
    ax_map.set_ylim(ys.min() - margin, ys.max() + margin)
    ax_map.set_title("AGIMO WP1 — Temporal Replay",
                     color="white", pad=6, fontsize=11)

    # vehicle scatter + labels
    veh_sc = ax_map.scatter(np.zeros(len(vids)), np.zeros(len(vids)),
                            c=[VEH_COLORS["idle"]] * len(vids), s=160,
                            zorder=5, edgecolors="white", linewidths=0.7)
    veh_lbl = [ax_map.text(0, 0, f"V{v}", color="white", fontsize=7.5,
                           ha="center", va="bottom", zorder=6,
                           fontweight="bold")
               for v in vids]

    # request scatter: origin dots (hollow), pickup/dropoff node dots (filled), on-board
    rq_origin_sc = ax_map.scatter([], [], facecolors="none", edgecolors=RQ_WAIT,
                                  s=40, marker="o", zorder=4, linewidths=1.0, alpha=0.7)
    rq_wait_sc   = ax_map.scatter([], [], c=RQ_WAIT,  s=75,
                                  marker="o", zorder=4, alpha=0.9)
    rq_board_sc  = ax_map.scatter([], [], c=RQ_BOARD, s=75,
                                  marker="D", zorder=4, alpha=0.9)
    rq_dest_sc   = ax_map.scatter([], [], facecolors="none", edgecolors="#66ddaa",
                                  s=40, marker="o", zorder=4, linewidths=1.0, alpha=0.7)
    # walk lines: origin → pickup node, and dropoff node → destination
    from matplotlib.collections import LineCollection
    walk_lines = LineCollection([], colors="#ff6666", linewidths=0.8,
                                linestyles="dashed", alpha=0.6, zorder=3)
    ax_map.add_collection(walk_lines)
    walk_do_lines = LineCollection([], colors="#66ddaa", linewidths=0.8,
                                   linestyles="dashed", alpha=0.6, zorder=3)
    ax_map.add_collection(walk_do_lines)

    # legend
    _m = lambda fc, mk, lbl: Line2D(
        [0], [0], marker=mk, color="none",
        markerfacecolor=fc, markersize=9, label=lbl)
    legend_handles = [
        _m(VEH_COLORS["idle"],          "o", "Veh idle"),
        _m(VEH_COLORS["route_empty"],   "o", "Veh routing (empty)"),
        _m(VEH_COLORS["route_loaded"],  "o", "Veh routing (pax)"),
        _m(VEH_COLORS["boarding"],      "o", "Veh boarding"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor="#ff9500", markersize=12, label="Hub"),
        _m(RQ_WAIT,  "o", "Pickup stop (waiting)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=RQ_WAIT, markersize=7, label="Request origin"),
        Line2D([0], [0], color="#ff6666", linewidth=1.2,
               linestyle="dashed", label="Walk to stop"),
        Line2D([0], [0], color="#66ddaa", linewidth=1.2,
               linestyle="dashed", label="Walk from stop"),
        _m(RQ_BOARD, "D", "Request on board"),
    ]
    ax_map.legend(handles=legend_handles, loc="lower left",
                  facecolor="#12122a", edgecolor="#334466",
                  labelcolor="white", fontsize=7.5)

    # ── event log panel ───────────────────────────────────────────────────────
    MAX_LOG_LINES = 14
    log_scroll = {"offset": 0}   # 0 = pinned to bottom (most recent)

    ax_log.set_xlim(0, 1); ax_log.set_ylim(0, 1)
    ax_log.set_xticks([]); ax_log.set_yticks([])
    ax_log.set_title("Event Log  (scroll to browse)", color="white", pad=6, fontsize=9)

    log_top_ind = ax_log.text(
        0.98, 0.99, "", transform=ax_log.transAxes,
        color="#ffaa44", fontsize=7, va="top", ha="right", fontfamily="monospace",
    )
    log_txt = ax_log.text(
        0.04, 0.93, "", transform=ax_log.transAxes,
        color="#99bbff", fontsize=7.0, va="top", ha="left", fontfamily="monospace",
    )
    log_bot_ind = ax_log.text(
        0.98, 0.01, "", transform=ax_log.transAxes,
        color="#ffaa44", fontsize=7, va="bottom", ha="right", fontfamily="monospace",
    )

    def _update_log(t):
        past = [f"t={et:5.0f}s  {msg}" for et, msg in events if et <= t]
        n = len(past)
        off = log_scroll["offset"]
        # clamp offset so we never scroll past all events
        off = max(0, min(off, max(0, n - MAX_LOG_LINES)))
        log_scroll["offset"] = off

        end_idx   = n - off if off > 0 else n
        start_idx = max(0, end_idx - MAX_LOG_LINES)
        log_txt.set_text("\n".join(past[start_idx:end_idx]))

        # "▲ N older" above, "▼ N newer" below
        log_top_ind.set_text(f"▲ {start_idx} older" if start_idx > 0 else "")
        log_bot_ind.set_text(f"▼ {off} newer"       if off > 0       else "")

    # ── update function (shared by interactive and save paths) ───────────────
    label_offset = (ys.max() - ys.min()) * 0.04
    frame_times  = np.arange(0, end_time + frame_step, frame_step)

    def update(t):
        vxy   = []
        vclrs = []
        vpos  = {}
        for i, vid in enumerate(vids):
            node, ck = veh_state_at(segs[vid], t, hubs=hubs, coords=coords,
                                         return_to_hub=return_to_hub)
            x, y = coords.get(node, (xs.min(), ys.min())) if node is not None else (xs.min(), ys.min())
            vxy.append([x, y])
            vclrs.append(VEH_COLORS[ck])
            vpos[vid] = (x, y)
            veh_lbl[i].set_position((x, y + label_offset))

        veh_sc.set_offsets(np.array(vxy))
        veh_sc.set_facecolor(vclrs)

        wait_xy    = []   # pickup stop node
        origin_xy  = []   # actual origin node (when different from pickup stop)
        walk_segs  = []   # [(origin_xy, pickup_xy), ...] for dashed walk lines
        board_xy   = []
        dest_xy    = []   # actual destination node (when different from dropoff stop)
        walk_do_segs = [] # [(dropoff_xy, dest_xy), ...] for dashed walk lines
        for _, rq in req.iterrows():
            if t < float(rq["rq_time"]):
                continue
            pt = rq["pickup_time"]
            dt = rq["dropoff_time"]
            if pd.isna(pt) or t < float(pt):
                pu_n  = int(rq["pu_node"])
                ori_n = int(rq["start_node"])
                if pu_n in coords:
                    wait_xy.append(coords[pu_n])
                    if pu_n != ori_n and ori_n in coords:
                        origin_xy.append(coords[ori_n])
                        walk_segs.append([coords[ori_n], coords[pu_n]])
            elif pd.isna(dt) or t < float(dt):
                vid = int(rq["vehicle_id"]) if pd.notna(rq["vehicle_id"]) else None
                if vid in vpos:
                    board_xy.append(vpos[vid])
            elif t < float(dt) + 300:   # show walk-from-dropoff for 5 min
                do_n  = int(rq["do_node"])
                end_n = int(rq["end_node"])
                if do_n != end_n and do_n in coords and end_n in coords:
                    dest_xy.append(coords[end_n])
                    walk_do_segs.append([coords[do_n], coords[end_n]])

        rq_wait_sc.set_offsets(  np.array(wait_xy)   if wait_xy   else np.empty((0, 2)))
        rq_origin_sc.set_offsets(np.array(origin_xy) if origin_xy else np.empty((0, 2)))
        rq_board_sc.set_offsets( np.array(board_xy)  if board_xy  else np.empty((0, 2)))
        rq_dest_sc.set_offsets(  np.array(dest_xy)   if dest_xy   else np.empty((0, 2)))
        walk_lines.set_segments(walk_segs)
        walk_do_lines.set_segments(walk_do_segs)

        _update_log(t)

        return [veh_sc, rq_wait_sc, rq_origin_sc, rq_board_sc, rq_dest_sc,
                walk_lines, walk_do_lines,
                log_txt, log_top_ind, log_bot_ind] + veh_lbl

    # ── interactive controls ──────────────────────────────────────────────────
    # Leave room at the bottom for slider + buttons.
    fig.subplots_adjust(bottom=0.17, top=0.95, left=0.02, right=0.98)

    # Slider  (horizontal bar across most of the bottom)
    ax_slider = fig.add_axes([0.08, 0.09, 0.84, 0.03])
    ax_slider.set_facecolor(PANEL)
    slider = Slider(ax_slider, "", 0.0, end_time,
                    valinit=0.0, valstep=float(frame_step),
                    color="#4466cc", initcolor="none")
    slider.label.set_color("white")
    slider.valtext.set_color("#99bbff")
    slider.valtext.set_fontsize(8)
    # Draw a dim track behind the slider
    ax_slider.set_facecolor("#2a2a55")

    # Buttons  (below the slider, centred)
    btn_w, btn_h = 0.07, 0.04
    btn_y = 0.03
    centres = [0.35, 0.44, 0.53, 0.62]   # ◀◀  ◀  ▶/⏸  ▶▶
    ax_back2 = fig.add_axes([centres[0] - btn_w/2, btn_y, btn_w, btn_h])
    ax_back1 = fig.add_axes([centres[1] - btn_w/2, btn_y, btn_w, btn_h])
    ax_play  = fig.add_axes([centres[2] - btn_w/2, btn_y, btn_w, btn_h])
    ax_fwd1  = fig.add_axes([centres[3] - btn_w/2, btn_y, btn_w, btn_h])

    _bc = "#2a2a55"   # button face colour
    _hc = "#334466"   # hover colour
    btn_back2 = Button(ax_back2, "|◀", color=_bc, hovercolor=_hc)
    btn_back1 = Button(ax_back1, "◀",  color=_bc, hovercolor=_hc)
    btn_play  = Button(ax_play,  "▶",  color=_bc, hovercolor=_hc)
    btn_fwd1  = Button(ax_fwd1,  "▶▶", color=_bc, hovercolor=_hc)
    for btn in (btn_back2, btn_back1, btn_play, btn_fwd1):
        btn.label.set_color("white")
        btn.label.set_fontsize(11)

    # ── playback state ────────────────────────────────────────────────────────
    state = {"playing": False}

    def _set_t(t):
        """Jump to time t, update slider + canvas."""
        t = float(np.clip(t, 0, end_time))
        slider.set_val(t)           # triggers on_slider_change → update(t)

    def on_slider_change(val):
        update(float(val))
        fig.canvas.draw_idle()

    def on_play_pause(_):
        state["playing"] = not state["playing"]
        btn_play.label.set_text("⏸" if state["playing"] else "▶")
        fig.canvas.draw_idle()

    def on_back2(_):
        state["playing"] = False
        btn_play.label.set_text("▶")
        _set_t(0.0)

    def on_back1(_):
        state["playing"] = False
        btn_play.label.set_text("▶")
        _set_t(slider.val - frame_step)

    def on_fwd1(_):
        state["playing"] = False
        btn_play.label.set_text("▶")
        _set_t(slider.val + frame_step)

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_pause)
    btn_back2.on_clicked(on_back2)
    btn_back1.on_clicked(on_back1)
    btn_fwd1.on_clicked(on_fwd1)

    # Canvas timer — fires every interval_ms to advance one step when playing
    timer = fig.canvas.new_timer(interval=interval_ms)

    def _tick():
        if not state["playing"]:
            return
        next_t = slider.val + frame_step
        if next_t > end_time:
            state["playing"] = False
            btn_play.label.set_text("▶")
            fig.canvas.draw_idle()
        else:
            _set_t(next_t)

    timer.add_callback(_tick)
    timer.start()

    # Scroll wheel over the log panel browses event history
    def on_scroll(event):
        if event.inaxes is not ax_log:
            return
        # scroll up (step > 0) → reveal older events (increase offset)
        # scroll down (step < 0) → return to newer events (decrease offset)
        log_scroll["offset"] = max(0, log_scroll["offset"] + int(event.step))
        _update_log(slider.val)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Draw initial frame
    update(0.0)

    return fig, frame_times, update, timer


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(message)s",
        stream=sys.stdout,
    )

    result_dir = (Path(RESULT_DIR)
                  if RESULT_DIR
                  else find_result_dir(STUDY_DIR / "results"))
    LOG.info("Loading from: %s", result_dir)

    ops, req, nw_name, end_time = load_data(result_dir)
    coords, hubs = load_nodes(nw_name)
    LOG.info("Network: %s  (%d nodes, %d hubs)", nw_name, len(coords), len(hubs))

    fig, frame_times, update, timer = make_animation(
        ops, req, coords, hubs, end_time,
        frame_step=FRAME_STEP, interval_ms=INTERVAL_MS,
    )

    if SAVE_GIF:
        timer.stop()
        out = result_dir / "timeline.gif"
        LOG.info("Saving animation → %s", out)
        ani = FuncAnimation(fig, update, frames=frame_times,
                            interval=INTERVAL_MS, blit=False, repeat=False)
        ani.save(str(out), writer="pillow", fps=max(1, 1000 // INTERVAL_MS))
        LOG.info("Saved.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
