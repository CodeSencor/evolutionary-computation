#!/usr/bin/env python3
"""Validate Lab7 results (large neighbourhood search) and render report.pdf (same layout as the Lab5 report).

Result files (written by LNS.cs), named <instance lower>S_LNS_<RN|SP|MP>_<LS|noLS>.csv, e.g. tspaS_LNS_MP_LS.csv:
for every run one line with its running time in µs, then ceil(n/2) lines "x;y;cost".
<instance lower>_LNS_iterations.csv: "Method,Iterations", one line per run (number of iterations of the main loop).
For comparison the results of the previous assignments are read from ../Lab2, ../Lab5 and ../Lab6.

Checks per run: correct number of nodes, every node exists in the instance, no repeats, and for the methods that
end with local search that no improving move is left in the neighbourhood (inter-route node exchange + two-edges
exchange), recomputed independently here.

Requires: numpy, reportlab, matplotlib (only its bundled DejaVu fonts are used).

Usage: python lab7_report.py [--results DIR] [--out FILE]
"""
import argparse
import csv
import html
from pathlib import Path

import matplotlib
import numpy as np
from reportlab.graphics.shapes import Circle, Drawing, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
                                XPreformatted)

HERE = Path(__file__).resolve().parent
INSTANCES = ["TSPA", "TSPB"]

# ---- Text you fill in (CONCLUSIONS: reportlab markup, e.g. <b>, &lt;) --------------------------------
AUTHORS = "Paweł Charkiewicz 160288, Piotr Franc 160306"
SOURCE_URL = "https://github.com/CodeSencor/evolutionary-computation"
PSEUDO_INTRO = (
    "Notation: n = number of nodes, N = ⌈n/2⌉ = number of nodes in a solution, s = the current solution "
    "(a cycle s[0], …, s[N − 1], positions modulo N), d(a, b) = Euclidean distance rounded to the nearest integer, "
    "Δ = change of the objective (Δ < 0 = improvement), R = round(0.3 · N) = number of removed nodes. "
    "All LNS versions start from a random solution improved by local search and stop after the average running "
    "time of MSLS from assignment 6. The three destroy operators are used separately; each is run with and without "
    "local search after repair (6 methods, 20 runs each).")
PSEUDOCODE = {  # title -> pseudocode (plain text, shown in <pre>)
    "LNS: large neighbourhood search": """\
Input: time_limit = average running time of one MSLS run (Lab6), destroy ∈ {RN, SP, MP}, use_ls ∈ {yes, no}
x = LocalSearch(RandomStart())                 (local search is always applied to the initial solution)
repeat until elapsed time ≥ time_limit
    y = Destroy(x)                             (partial cycle with N − R nodes)
    y = Repair(y)                              (full solution with N nodes again)
    if use_ls: y = LocalSearch(y)
    iterations = iterations + 1
    if f(y) < f(x): x = y                      (minimisation: accept only improvements)
return x""",
    "Destroy RN: removal of randomly selected nodes": """\
1. Shuffle the positions 0 … N − 1 of x.
2. Remove the nodes at the first R positions.
3. The remaining N − R nodes keep their order in the cycle → partial cycle.""",
    "Destroy SP / MP: removal of one (SP) or 3-5 (MP) random subpaths": """\
k = 1 (SP) or a random integer from {3, 4, 5} (MP)
1. Split R into k random positive parts r_1 … r_k            (removed subpaths)
   and N − R into k random positive parts g_1 … g_k          (kept parts between them)
   (random split = k − 1 distinct random cut points of 1 … total − 1)
2. Choose a random start position p in x.
3. for i = 1 … k:  remove r_i consecutive nodes starting at p,  then p = p + r_i + g_i  (modulo N)
4. The remaining N − R nodes keep their order in the cycle → partial cycle.
The removed and kept parts alternate, so there are exactly k disjoint subpaths with R nodes in total.""",
    "Repair: greedy cycle + weighted 2-regret (best greedy heuristic, Lab2)": """\
Input: partial cycle y (N − R nodes); every not selected node (also the removed ones) can be inserted
while |y| < N:
    for every not selected node k:
        for every edge (a, b) of y, including the closing edge:
            insertion cost = d(a, k) + d(k, b) − d(a, b) + cost(k)
        best1(k), best2(k) = smallest and second smallest insertion cost, pos(k) = edge of best1(k)
        score(k) = regret(k) − best1(k),    regret(k) = best2(k) − best1(k)      (weights 1 : 1)
    insert the node with the greatest score (ties: smaller best1) into the edge pos(k)
The repair works directly on the cycle, so no cycle has to be broken.""",
    "Local search: steepest with the list of improving moves (LM, Lab5)": """\
Neighbourhood: node exchange (inter-route) + two-edges exchange (intra-route).
LM = all improving moves sorted by Δ; take the best applicable move, apply it, evaluate only the moves that touch
the changed edges / nodes; stop when no move in LM is applicable (local optimum).""",
    "RANDOM: random starting solution": """\
1. Shuffle all node indices (one generator with seed 42 for the whole experiment).
2. The first N of them, in that order, form the cycle.""",
}
CONCLUSIONS = (
    "All six LNS versions are clearly better than the greedy heuristics, a single local search and MSLS on both "
    "instances, using the same running time as MSLS. The destroy operator matters more than the local search: "
    "removing 3-5 random subpaths (MP) is the best operator on both instances, random nodes (RN) and a single "
    "subpath (SP) are clearly worse. Removing several medium-sized chunks of the cycle lets the greedy-regret "
    "repair rebuild whole regions of the cycle in a different way, while the scattered single nodes of RN are "
    "mostly put back where they were and one subpath (SP) changes only one region at a time.<br/><br/>"
    "Local search after repair improves the average of every operator except RN on TSPB, at the cost of about "
    "20% fewer iterations (about 1 450 - 1 550 instead of 1 800 - 1 900 per run). The repaired solution is already "
    "close to a local optimum, so the local search is cheap; one iteration is dominated by the greedy-regret "
    "repair, which evaluates every insertion of every not selected node for each of the 30 inserted nodes.<br/><br/>"
    "On TSPA LNS with MP and local search matches ILS on average (71709.85 vs 71726.05) and is much more stable "
    "(max 71767 vs 72722), but ILS found the better best solution (71500 vs 71682). On TSPB ILS remains better "
    "(48408.85 vs 48806.25 on average). Note that the LNS times were measured on a different machine than the "
    "Lab6 results, so the number of iterations is only approximately comparable.")
# --------------------------------------------------------------------------------------------------

METHODS = [  # (file relative to --results, label, runs, local optimum expected, key in the iterations CSV)
    ("../Lab2/{inst}_greedy_cycle_weighted.csv", "Greedy cycle + weighted 2-regret (Lab2)", 200, False, None),
    ("../Lab2/{inst}_NNAP_weighted.csv", "Nearest neighbour (any position) + weighted 2-regret (Lab2)", 200, False,
     None),
    ("../Lab5/{lower}S_ES_R_LM.csv", "Steepest LS with LM, random start (Lab5)", 200, True, None),
    ("../Lab6/{lower}S_MSLS.csv", "MSLS, 400 local searches per run (Lab6)", 20, True, None),
    ("../Lab6/{lower}S_ILS.csv", "ILS (Lab6, perturbation sizes 8 × 11, then 20)", 20, True, None),
    ("{lower}S_LNS_RN_LS.csv", "LNS, random nodes, with LS", 20, True, "LNS_RN_LS"),
    ("{lower}S_LNS_RN_noLS.csv", "LNS, random nodes, without LS", 20, False, "LNS_RN_noLS"),
    ("{lower}S_LNS_SP_LS.csv", "LNS, single subpath, with LS", 20, True, "LNS_SP_LS"),
    ("{lower}S_LNS_SP_noLS.csv", "LNS, single subpath, without LS", 20, False, "LNS_SP_noLS"),
    ("{lower}S_LNS_MP_LS.csv", "LNS, 3-5 subpaths, with LS", 20, True, "LNS_MP_LS"),
    ("{lower}S_LNS_MP_noLS.csv", "LNS, 3-5 subpaths, without LS", 20, False, "LNS_MP_noLS"),
]


def read_rows(path):
    """Node lines -> (x, y, cost) ints; a single-value line is a run time (decimal comma accepted)."""
    rows = [line.split(";") for line in path.read_text().split() if line.strip()]
    return [(float(r[0].replace(",", ".")),) if len(r) == 1 else tuple(int(v) for v in r) for r in rows]


def load_instance(path):
    nodes = np.array(read_rows(path), dtype=np.int64)
    d = np.hypot(*(nodes[:, None, :2] - nodes[None, :, :2]).transpose(2, 0, 1))
    return nodes, np.round(d).astype(np.int64), nodes[:, 2]


def split_runs(rows, size):
    """Returns ([node rows per run], [time per run] or None)."""
    if not any(len(r) == 1 for r in rows):
        return [rows[i:i + size] for i in range(0, len(rows), size)], None
    runs, times = [], []
    for r in rows:
        if len(r) == 1:
            runs.append([])
            times.append(float(r[0]))
        elif runs:
            runs[-1].append(r)
    return runs, times


def load_runs(path, nodes, expected_runs):
    """Returns (list of index arrays, or None for an invalid run; times or None; list of error messages)."""
    lookup = {tuple(r): i for i, r in enumerate(nodes.tolist())}
    size = (len(nodes) + 1) // 2
    chunks, times = split_runs(read_rows(path), size)
    errors, runs = [], []
    for run_no, chunk in enumerate(chunks):
        idx = [lookup.get(row[:3]) for row in chunk]
        if len(chunk) != size:
            errors.append(f"run #{run_no}: {len(chunk)} nodes, expected {size}")
        elif None in idx:
            errors.append(f"run #{run_no}: node {chunk[idx.index(None)]} is not in the instance")
        elif len(set(idx)) != size:
            errors.append(f"run #{run_no}: repeated nodes")
        else:
            runs.append(np.array(idx))
            continue
        runs.append(None)
    if len(runs) != expected_runs:
        errors.append(f"{len(runs)} runs, expected {expected_runs}")
    return runs, times, errors


def load_iterations(path):
    """Method -> list of main loop iterations per run."""
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["Method"], []).append(int(r["Iterations"]))
    return out


def objective(t, D, c):
    return int(D[t, np.roll(t, -1)].sum() + c[t].sum())


def best_delta(t, D, c):
    """Smallest objective change over inter-route exchange + two-edges exchange (>= 0 means local optimum)."""
    prv, nxt = np.roll(t, 1), np.roll(t, -1)
    unused = np.setdiff1d(np.arange(len(c)), t)
    inter = (D[prv[:, None], unused] + D[unused[None, :], nxt[:, None]] + c[unused][None, :]
             - (D[prv, t] + D[t, nxt] + c[t])[:, None])
    # two-edges exchange: remove (t_i,t_i+1),(t_j,t_j+1), add (t_i,t_j),(t_i+1,t_j+1); adjacent pairs give 0
    edge = D[t, nxt]
    intra_d = (D[t[:, None], t] + D[nxt[:, None], nxt] - edge[:, None] - edge[None, :])[
        np.triu_indices(len(t), 1)]  # i < j only: the diagonal would be a bogus negative
    return int(min(inter.min(), intra_d.min()))


def self_test():
    rng = np.random.default_rng(0)
    nodes = np.column_stack([rng.integers(0, 100, (12, 2)), rng.integers(0, 50, 12)])
    D = np.round(np.hypot(*(nodes[:, None, :2] - nodes[None, :, :2]).transpose(2, 0, 1))).astype(np.int64)
    c, t = nodes[:, 2], rng.permutation(12)[:6]
    base = objective(t, D, c)
    inter = [objective(np.where(t == a, u, t), D, c) - base for a in t for u in set(range(12)) - set(t)]
    opt2 = [objective(np.r_[t[:i + 1], t[i + 1:j + 1][::-1], t[j + 1:]], D, c) - base
            for i in range(6) for j in range(i + 1, 6)]
    assert best_delta(t, D, c) == min(inter + opt2)
    runs, times = split_runs([(5,), (1, 2, 3), (4, 5, 6), (7,), (1, 2, 3), (4, 5, 6)], 2)
    assert times == [5, 7] and len(runs) == 2 and runs[1] == [(1, 2, 3), (4, 5, 6)]


def stats(values):
    return (float(np.mean(values)), min(values), max(values)) if values else None


def evaluate(path, nodes, D, c, expected_runs, check_local_opt, iterations):
    res = {"file": path.name, "exists": path.exists(), "check_local_opt": check_local_opt, "iters": stats(iterations)}
    if not path.exists():
        res["errors"] = [f"missing file {path}"]
        return res
    runs, times, errors = load_runs(path, nodes, expected_runs)
    objs = [objective(t, D, c) if t is not None else None for t in runs]
    valid = [o for o in objs if o is not None]
    res.update(runs=runs, objs=objs, errors=errors, n_runs=len(runs), n_valid=len(valid),
               obj=stats(valid), time=stats([t / 1000 for t in times] if times else times))  # µs in the CSV -> ms
    if valid:
        res["best"] = min((o, i) for i, o in enumerate(objs) if o is not None)
    deltas = [best_delta(t, D, c) for t in runs if t is not None]
    res["local_opt"] = sum(d >= 0 for d in deltas)
    res["worst_delta"] = min(deltas) if deltas else None
    return res


# ---- PDF ------------------------------------------------------------------------------------------
# DejaVu (shipped with matplotlib) has ł, Δ, ⌈ ⌉, →, ≥, which the built-in PDF fonts lack.
_FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for _name, _file in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                     ("DejaVuMono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(_name, str(_FONTS / _file)))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")

MARGIN = 2 * cm
FRAME_W = A4[0] - 2 * MARGIN
OK, BAD, INFO = "#1a7f37", "#b42318", "#666666"
_ss = getSampleStyleSheet()


def _style(name, parent, **kw):
    kw.setdefault("fontName", "DejaVu")
    return ParagraphStyle(name, parent=_ss[parent], **kw)


H1 = _style("h1", "Heading1", fontName="DejaVu-Bold")
H2 = _style("h2", "Heading2", fontName="DejaVu-Bold")
H3 = _style("h3", "Heading3", fontName="DejaVu-Bold", fontSize=11)
H4 = _style("h4", "Heading4", fontName="DejaVu-Bold", fontSize=9)
BODY = _style("body", "BodyText", fontSize=9.5, leading=13)
CELL = _style("cell", "BodyText", fontSize=8.5, leading=10.5, alignment=TA_RIGHT)
CELL_L = _style("cell_l", "BodyText", fontSize=8.5, leading=10.5)
CELL_C = _style("cell_c", "BodyText", fontSize=8.5, leading=10.5, alignment=TA_CENTER)
SMALL = _style("small", "BodyText", fontSize=8, leading=10, spaceBefore=0, spaceAfter=1)
MONO = _style("mono", "Code", fontName="DejaVuMono", fontSize=7.5, leading=9.5, leftIndent=0,
              backColor=colors.HexColor("#f6f6f6"), borderPadding=4, spaceBefore=4, spaceAfter=8)
esc = html.escape


def _table(data, first_col_w):
    rest = (FRAME_W - first_col_w) / (len(data[0]) - 1)
    t = Table(data, colWidths=[first_col_w] + [rest] * (len(data[0]) - 1), repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def results_table(rows, key, digits, best_of=min):
    """rows: [(label, {instance: res})]; per instance columns Min / Avg / Max; the best value in each column in bold."""
    stats_order = (("Min", 1), ("Avg", 0), ("Max", 2))  # (header, index in the (av, lo, hi) tuple)
    cols = [(i, k) for i in INSTANCES for _, k in stats_order]
    best = {c: best_of((r[c[0]][key][c[1]] for _, r in rows if r[c[0]].get(key)), default=None) for c in cols}
    data = [[Paragraph("<b>Method</b>", CELL_L)] + [x for i in INSTANCES for x in
                                                   (Paragraph(f"<b>{i}</b>", CELL_C), "", "")],
            [""] + [Paragraph(f"<b>{h}</b>", CELL) for _ in INSTANCES for h, _ in stats_order]]
    for label, r in rows:
        line = [Paragraph(esc(label), CELL_L)]
        for i, k in cols:
            v = r[i][key][k] if r[i].get(key) else None
            txt = "n/a" if v is None else f"{v:.{0 if isinstance(v, (int, np.integer)) else digits}f}"
            line.append(Paragraph(f"<b>{txt}</b>" if v is not None and v == best[(i, k)] else txt, CELL))
        data.append(line)
    t = _table(data, 130)
    t.setStyle(TableStyle([("SPAN", (0, 0), (0, 1)), ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f2f2f2"))] +
                          [("SPAN", (1 + 3 * n, 0), (3 + 3 * n, 0)) for n in range(len(INSTANCES))]))
    return t


def _colored(txt, ok, color=None):
    return Paragraph(f'<font color="{color or (OK if ok else BAD)}"><b>{esc(txt)}</b></font>', CELL)


def validation_table(rows):
    data = [[Paragraph("<b>Method</b>", CELL_L)] + [
        Paragraph(f"<b>{i} {what}</b>", CELL) for i in INSTANCES for what in ("valid", "local optima")]]
    for label, r in rows:
        line = [Paragraph(esc(label), CELL_L)]
        for inst in INSTANCES:
            res = r[inst]
            if not res["exists"]:
                line += [_colored("missing", False), Paragraph("", CELL)]
                continue
            ok = res["n_valid"] == res["n_runs"] and not res["errors"]
            line.append(_colored(f'{res["n_valid"]} / {res["n_runs"]}', ok))
            if res["n_valid"]:
                lo_ok = res["local_opt"] == res["n_valid"]
                extra = "" if lo_ok else f' (best Δ {res["worst_delta"]})'
                # methods without a final local search are not expected to end in a local optimum: grey, not red
                line.append(_colored(f'{res["local_opt"]} / {res["n_valid"]}{extra}', lo_ok,
                                     None if res["check_local_opt"] else INFO))
            else:
                line.append(Paragraph("–", CELL))
        data.append(line)
    return _table(data, 170)


def plot_drawing(t, nodes, W=440):
    x, y, cost = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    pad, legend_h = 10, 18
    H = min(W * 0.75, (W - 2 * pad) * max(np.ptp(y), 1) / max(np.ptp(x), 1) + 2 * pad)  # follow the data's aspect
    k = W / 640  # marker sizes as in the 640-wide HTML plots
    s = min((W - 2 * pad) / max(np.ptp(x), 1), (H - 2 * pad) / max(np.ptp(y), 1))
    px = lambda i: (pad + (x[i] - x.min()) * s, legend_h + pad + (y[i] - y.min()) * s)
    f = (cost - cost.min()) / max(np.ptp(cost), 1)
    shade = lambda v: colors.Color((255 - 120 * v) / 255, (220 - 220 * v) / 255, (60 - 60 * v) / 255)
    d = Drawing(W, H + legend_h)
    d.add(Rect(0, legend_h, W, H, fillColor=colors.white, strokeColor=colors.HexColor("#dddddd"), strokeWidth=0.5))
    d.add(Polygon([v for i in t for v in px(i)], fillColor=None,
                  strokeColor=colors.HexColor("#3b6ea5"), strokeWidth=1.4 * k))
    used = set(t.tolist())
    for i in range(len(nodes)):
        cx, cy = px(i)
        r = (3 + 6 * f[i]) * k
        if i in used:
            d.add(Circle(cx, cy, r, fillColor=shade(f[i]), strokeColor=colors.HexColor("#222222"), strokeWidth=0.6 * k))
        else:
            d.add(Circle(cx, cy, r, fillColor=None, strokeColor=colors.HexColor("#aaaaaa"), strokeWidth=0.8 * k))
    cx, cy = px(t[0])
    d.add(Circle(cx, cy, (6 + 6 * f[t[0]]) * k, fillColor=None, strokeColor=colors.HexColor("#222222"),
                 strokeWidth=1.2 * k))
    for j in range(30):  # gradient bar of the legend
        d.add(Rect(j * 3, 4, 3, 8, fillColor=shade(j / 29), strokeColor=None))
    d.add(String(96, 5, f"node cost {cost.min()} → {cost.max()} (colour and size) · hollow grey = not selected "
                        "· ring = first node", fontName="DejaVu", fontSize=7, fillColor=colors.HexColor("#555555")))
    return d


def instance_section(inst, nodes, rows):
    out = [PageBreak(), Paragraph(f"Instance {inst}", H2)]
    for label, r in rows:
        res = r[inst]
        if "best" not in res:
            out += [Paragraph(esc(label), H3), Paragraph(f'<font color="{BAD}"><b>No valid runs.</b></font>', BODY)]
            continue
        obj, run = res["best"]
        t = res["runs"][run]
        out.append(KeepTogether([
            Paragraph(esc(label), H3),
            Paragraph(f"Best of {res['n_valid']} valid runs: run #{run} with objective <b>{obj}</b>.", BODY),
            plot_drawing(t, nodes),
            Paragraph("Best solution (node indices, from 0)", H4),
            Paragraph(", ".join(map(str, t)), _style(f"nodes{inst}{run}{label}", "BodyText", fontName="DejaVuMono",
                                                       fontSize=7, leading=9)),
        ]))
    return out


def _page_number(canvas, doc):
    canvas.setFont("DejaVu", 8)
    canvas.drawRightString(A4[0] - MARGIN, 1.2 * cm, str(doc.page))


def build_pdf(out, rows, data, problems):
    longest = max(len(line) for v in PSEUDOCODE.values() for line in v.splitlines())
    mono = ParagraphStyle("mono_fit", parent=MONO, fontSize=min(8.0, FRAME_W / (0.602 * longest)))
    mono.leading = mono.fontSize * 1.3
    lns_rows = [(label, r) for (_, label, _, _, it_key), (_, r) in zip(METHODS, rows) if it_key]
    story = [Paragraph("Large neighbourhood search", H1), Paragraph(esc(AUTHORS), H2),
             Paragraph("Objective = cycle length (Euclidean, rounded to nearest integer) + sum of selected node "
                       "costs. Lower is better.", BODY),
             Paragraph("Pseudocode", H2), Paragraph(esc(PSEUDO_INTRO), BODY)]
    for k, v in PSEUDOCODE.items():
        story += [KeepTogether([Paragraph(esc(k), H3), XPreformatted(esc(v), mono)])]
    story += [PageBreak(), Paragraph("Computational experiment", H2),
              Paragraph(f"Instances: {', '.join(INSTANCES)}. Six LNS versions (three destroy operators, each with and "
                        "without local search after repair), each run 20 times with the time limit equal to the "
                        "average running time of MSLS from assignment 6. For comparison: the best greedy heuristics "
                        "(Lab2, 200 runs, one per starting node), the best steepest local search (Lab5, 200 runs), "
                        "MSLS and ILS (Lab6, 20 runs). For each instance: minimum, average and maximum over the valid "
                        "runs; in each column the best value is in bold.", BODY),
              Paragraph("Objective function", H3), results_table(rows, "obj", 2),
              Paragraph("Running time [ms]", H3),
              Paragraph("Time of one run (n/a: the greedy result files store no times).", BODY),
              results_table(rows, "time", 2),
              Paragraph("Number of iterations of the main loop", H3),
              Paragraph("Number of destroy-repair(-local search) iterations of one LNS run (the local search of the "
                        "initial solution is not counted); the highest value in each column is in bold.", BODY),
              results_table(lns_rows, "iters", 1, best_of=max),
              KeepTogether([
                  Paragraph("Validation", H3),
                  Paragraph("Valid = the run has ⌈n/2⌉ distinct nodes of the instance. Local optima = no improving "
                            "move remains in the neighbourhood of the local search (checked independently by the "
                            "report script; \"best Δ\" is the most improving move left). Grey = the method does not "
                            "end with local search, so a local optimum is not expected.", BODY),
                  validation_table(rows)]), Spacer(1, 6)]
    story += ([Paragraph(f'<font color="{BAD}">• {esc(p)}</font>', SMALL) for p in problems] if problems else
              [Paragraph(f'<font color="{OK}"><b>All runs valid and all runs of the methods ending with local search '
                         'are local optima.</b></font>', BODY)])
    for inst in INSTANCES:
        story += instance_section(inst, data[inst][0], lns_rows)
    story += [Spacer(1, 12), Paragraph(f'Source code: <a href="{SOURCE_URL}" color="blue">{SOURCE_URL}</a>', BODY),
              Paragraph("Conclusions", H2), Paragraph(CONCLUSIONS, BODY)]
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                      bottomMargin=MARGIN, title="TSP large neighbourhood search", author=AUTHORS
                      ).build(story, onFirstPage=_page_number, onLaterPages=_page_number)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE,
                    help="folder with the Lab7 result CSVs and TSPA.csv/TSPB.csv (default: this folder)")
    ap.add_argument("--out", type=Path, default=HERE / "report.pdf")
    a = ap.parse_args()
    self_test()

    rows = [(label, {}) for _, label, _, _, _ in METHODS]
    data = {}
    for inst in INSTANCES:
        inst_file = a.results / f"{inst}.csv"
        if not inst_file.exists():
            raise SystemExit(f"{inst_file} not found")
        nodes, D, c = data[inst] = load_instance(inst_file)
        iterations = load_iterations(a.results / f"{inst.lower()}_LNS_iterations.csv")
        for (file, label, runs, local_opt, it_key), (_, r) in zip(METHODS, rows):
            print(f"checking {inst} {label} ...", flush=True)
            path = a.results / file.format(inst=inst, lower=inst.lower())
            r[inst] = evaluate(path, nodes, D, c, runs, local_opt, iterations.get(it_key))

    problems = [f"{inst} / {label}: {e}" for label, r in rows for inst in INSTANCES
                for e in r[inst]["errors"][:3] + ([f"… {len(r[inst]['errors']) - 3} more errors"]
                                                   if len(r[inst]["errors"]) > 3 else [])]
    problems += [f"{inst} / {label}: {r[inst]['n_valid'] - r[inst]['local_opt']} runs are not local optima "
                 f"(best remaining Δ = {r[inst]['worst_delta']})"
                 for label, r in rows for inst in INSTANCES
                 if r[inst]["check_local_opt"] and r[inst].get("n_valid")
                 and r[inst]["local_opt"] != r[inst]["n_valid"]]

    build_pdf(a.out, rows, data, problems)
    print(f"\nwrote {a.out}")
    print("\n".join(problems) if problems else "all checks passed")


if __name__ == "__main__":
    main()
