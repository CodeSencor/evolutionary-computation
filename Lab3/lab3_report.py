#!/usr/bin/env python3
"""Validate Lab3 local-search results and render report.pdf (same content as the Lab1/Lab2 reports).

Result files (DataUtility.ExportPaths with times), named <instance lower><key>.csv, e.g. tspaS_ES_GR.csv:
for every run one line with its running time in ms, then ceil(n/2) lines "x;y;cost".
Files without time lines (runs simply concatenated) are still read; the time table then shows "n/a".

Checks per run: correct number of nodes, every node exists in the instance, no repeats, and that no
improving move is left in the neighbourhood (inter-route node exchange + the method's intra-route move),
recomputed independently here.

Requires: numpy, reportlab, matplotlib (only its bundled DejaVu fonts are used).

Usage: python lab3_report.py [--results DIR] [--out FILE]
"""
import argparse
import html
from pathlib import Path

import matplotlib
import numpy as np
from reportlab.graphics.shapes import Circle, Drawing, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
                                XPreformatted)

HERE = Path(__file__).resolve().parent
INSTANCES = ["TSPA", "TSPB"]

# ---- Text you fill in (RANDOMIZATION / CONCLUSIONS: reportlab markup, e.g. <b>, &lt;) --------------------------------
AUTHORS = "Paweł Charkiewicz 160288, Piotr Franc 160306"
SOURCE_URL = "https://github.com/CodeSencor/evolutionary-computation"
PSEUDO_INTRO = (
    "All methods read the instance into masterArray, which stores every node (x, y, cost). M = number of "
    "runs = 200, N = solution size = ⌈n/2⌉, d(a, b) = Euclidean distance rounded to the nearest integer, "
    "s = the current solution (a cycle s[0] → s[1] → … → s[N − 1] → s[0]), prev(i) / next(i) = the nodes "
    "before / after position i on the cycle, Δ = change of the objective (negative = improvement). "
    "Every method is one option from each of three groups: the starting solution (InitialSolutionsType), "
    "the intra-route move (NeighborhoodType) and the type of local search (SearchMode). The inter-route "
    "move is always the exchange of a selected and a not selected node. All Δ are computed from the "
    "changed edges only; no neighbour solution is built to evaluate a move.")
PSEUDOCODE = {  # title -> pseudocode (plain text, shown in <pre>)
    "Local search (common loop)": """\
0. Compute an n × n distance matrix d between every pair of nodes.
1. Build the move lists once:
       exchanges:  every pair (position i in s, index j in the list of not selected nodes)
       node swaps: every pair of positions (i, j), i < j
       edge swaps: every pair of positions (i, j), j ≥ i + 2, except (0, N − 1)
                   (the two edges would share node s[0])
2. Create M starting solutions (RANDOM or GREEDY_REGRET), timing the construction of each one.
3. For every starting solution s:
4.     Start a second timer for the search.
5.     Repeat:
6.         unused = nodes not in s.
7.         Shuffle the exchange list and the intra-route move list.
8.         Run one step of the search (GREEDY or STEEP); it applies at most one move.
9.     Until the step finds no move with Δ < 0.
10.    Stop the timer and save s and its running time = construction time + search time.""",
    "Inter-route move: exchange of a selected and a not selected node": """\
1. Take position i (node v = s[i], neighbours a = prev(i), b = next(i)) and a not selected node u.
2. Δ = d(a, u) + d(u, b) + cost(u) − d(a, v) − d(v, b) − cost(v)
3. Apply: put u at position i, mark v as not selected and u as selected.""",
    "NODE_SWAP: two-nodes exchange (intra-route)": """\
1. Take positions i < j, nodes x = s[i] and y = s[j].
2. If y comes right after x (… a, x, y, b …):
       Δ = d(a, y) + d(x, b) − d(a, x) − d(y, b)
3. Else if x comes right after y (i = 0, j = N − 1, the cycle wraps: … a, y, x, b …):
       Δ = d(a, x) + d(y, b) − d(a, y) − d(x, b)
4. Otherwise (… a, x, b … c, y, e …):
       Δ = d(a, y) + d(y, b) + d(c, x) + d(x, e) − d(a, x) − d(x, b) − d(c, y) − d(y, e)
5. Apply: swap s[i] and s[j].""",
    "EDGE_SWAP: two-edges exchange (intra-route)": """\
1. Take positions i < j (from the edge swap list), edges (s[i], s[i+1]) and (s[j], s[j+1]).
2. Δ = d(s[i], s[j]) + d(s[i+1], s[j+1]) − d(s[i], s[i+1]) − d(s[j], s[j+1])
3. Apply: reverse the segment s[i+1] … s[j]; the cycle now uses edges (s[i], s[j]) and (s[i+1], s[j+1]).""",
    "STEEP: steepest local search (one step)": """\
1. For every exchange in the list: compute Δ, remember the smallest one (bestInter).
2. For every intra-route move in the list: compute Δ, remember the smallest one (bestIntra).
3. If bestInter ≥ 0 and bestIntra ≥ 0: s is a local optimum, stop.
4. If bestInter < bestIntra: apply the best exchange.
5. Otherwise apply the best intra-route move.""",
    "GREEDY: greedy local search (one step)": """\
1. Both move lists were just shuffled. Set pointers e = 0 (exchanges) and t = 0 (intra-route moves).
2. While some move has not been evaluated:
3.     If both lists still have moves: toss a fair coin to pick the list.
       Otherwise: pick the list that still has moves.
4.     Take the next move from the picked list and compute its Δ.
5.     If Δ < 0: apply the move and end the step (improvement found).
6. If every move was evaluated without improvement: s is a local optimum, stop.""",
    "RANDOM: random starting solution": """\
1. Start with an empty cycle.
2. Until it holds N nodes: draw a random node index that is not yet used and append it.""",
    "GREEDY_REGRET: greedy cycle + weighted sum starting solution": """\
1. Run m (m = 0 … 199) starts from node m alone.
2. While the cycle holds fewer than N nodes:
3.     For every unused node k, compute the insertion cost for every edge (a, b) of the cycle,
       including the closing edge:
           d(a, k) + d(k, b) − d(a, b) + cost(k)
       Take the smallest value best1(k), the second smallest best2(k), and the edge pos(k)
       that gives best1(k). regret(k) = best2(k) − best1(k) (0 when there is only one edge).
4.     score(k) = w1 · regret(k) − w2 · best1(k),    w1 = 1, w2 = 1.5
5.     Choose the node k with the greatest score (ties: smaller best1), insert it into the edge pos(k)
       and mark it as used.""",
}
RANDOMIZATION = (
    "Before every step both move lists (the exchanges and the intra-route moves) are shuffled. The greedy "
    "search then walks through them with two pointers: while both lists still have moves, a fair coin "
    "decides whether the next evaluated move is an exchange or an intra-route move; when one list is "
    "exhausted the other one is continued. The first move with Δ &lt; 0 is applied, so only the moves up to "
    "that one are evaluated. Neither kind of move is always tried first.")
CONCLUSIONS = "Steepest search beats the greedy search in quality at the cost of performance. Our longest runs were all in steepest mode, which resulted in more than four times the expected cost per run. Greedy + 2-regret provides a very strong baseline solution which does not leave much room for improvement even in steepest mode, resulting in shorter runs. Quite unexpectedly the best performance overall was achieved by a random sourced solution from TSPB dataset. It can be hypothesized that stochasticity in this scenario is beneficial, as it provides explorativeness to environment where exploitativeness is limited."
# --------------------------------------------------------------------------------------------------

LS_METHODS = [  # (file key, label, intra-route move)
    (f"{m}_{nb}_{st}", f"{ml} LS, {nl}, {sl}", nb)
    for m, ml in (("S", "Steepest"), ("G", "Greedy"))
    for nb, nl in (("NS", "two-nodes exchange"), ("ES", "two-edges exchange"))
    for st, sl in (("R", "random start"), ("GR", "greedy start"))
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


def load_runs(path, nodes):
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
    if len(runs) != len(nodes):
        errors.append(f"{len(runs)} runs, expected {len(nodes)}")
    return runs, times, errors


def objective(t, D, c):
    return int(D[t, np.roll(t, -1)].sum() + c[t].sum())


def best_delta(t, D, c, intra):
    """Smallest objective change over inter-route exchange + intra-route moves (>= 0 means local optimum)."""
    prv, nxt = np.roll(t, 1), np.roll(t, -1)
    unused = np.setdiff1d(np.arange(len(c)), t)
    inter = (D[prv[:, None], unused] + D[unused[None, :], nxt[:, None]] + c[unused][None, :]
             - (D[prv, t] + D[t, nxt] + c[t])[:, None])
    if intra == "ES":  # remove (t_i,t_i+1),(t_j,t_j+1), add (t_i,t_j),(t_i+1,t_j+1); adjacent pairs give 0
        edge = D[t, nxt]
        intra_d = (D[t[:, None], t] + D[nxt[:, None], nxt] - edge[:, None] - edge[None, :])[
            np.triu_indices(len(t), 1)]  # i < j only: the diagonal would be a bogus negative
    else:  # two-nodes exchange: recompute every swapped tour (handles adjacent nodes without special cases)
        i, j = np.triu_indices(len(t), 1)
        tours = np.tile(t, (len(i), 1))
        rows = np.arange(len(i))
        tours[rows, i], tours[rows, j] = t[j], t[i]
        intra_d = D[tours, np.roll(tours, -1, axis=1)].sum(1) - D[t, nxt].sum()
    return int(min(inter.min(), intra_d.min()))


def self_test():
    rng = np.random.default_rng(0)
    nodes = np.column_stack([rng.integers(0, 100, (12, 2)), rng.integers(0, 50, 12)])
    D = np.round(np.hypot(*(nodes[:, None, :2] - nodes[None, :, :2]).transpose(2, 0, 1))).astype(np.int64)
    c, t = nodes[:, 2], rng.permutation(12)[:6]
    base = objective(t, D, c)
    inter = [objective(np.where(t == a, u, t), D, c) - base for a in t for u in set(range(12)) - set(t)]
    swaps = [objective(np.where(t == a, b, np.where(t == b, a, t)), D, c) - base for a in t for b in t if a != b]
    opt2 = [objective(np.r_[t[:i + 1], t[i + 1:j + 1][::-1], t[j + 1:]], D, c) - base
            for i in range(6) for j in range(i + 1, 6)]
    assert best_delta(t, D, c, "NS") == min(inter + swaps)
    assert best_delta(t, D, c, "ES") == min(inter + opt2)
    runs, times = split_runs([(5,), (1, 2, 3), (4, 5, 6), (7,), (1, 2, 3), (4, 5, 6)], 2)
    assert times == [5, 7] and len(runs) == 2 and runs[1] == [(1, 2, 3), (4, 5, 6)]


def stats(values):
    return (float(np.mean(values)), min(values), max(values)) if values else None


def evaluate(path, nodes, D, c, intra):
    res = {"file": path.name, "exists": path.exists()}
    if not path.exists():
        res["errors"] = [f"missing file {path}"]
        return res
    runs, times, errors = load_runs(path, nodes)
    objs = [objective(t, D, c) if t is not None else None for t in runs]
    valid = [o for o in objs if o is not None]
    res.update(runs=runs, objs=objs, errors=errors, n_runs=len(runs), n_valid=len(valid),
               obj=stats(valid), time=stats(times))
    if valid:
        res["best"] = min((o, i) for i, o in enumerate(objs) if o is not None)
    deltas = [best_delta(t, D, c, intra) for t in runs if t is not None]
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
OK, BAD = "#1a7f37", "#b42318"
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
SMALL = _style("small", "BodyText", fontSize=8, leading=10, spaceBefore=0, spaceAfter=1)
MONO = _style("mono", "Code", fontName="DejaVuMono", fontSize=7.5, leading=9.5, leftIndent=0,
              backColor=colors.HexColor("#f6f6f6"), borderPadding=4, spaceBefore=4, spaceAfter=8)
esc = html.escape


def fmt_cell(s, digits, bold):
    if s is None:
        return Paragraph("n/a", CELL)
    av, lo, hi = s
    mm = 0 if float(lo).is_integer() and float(hi).is_integer() else digits
    txt = f"{av:.{digits}f} ({lo:.{mm}f} – {hi:.{mm}f})"
    return Paragraph(f"<b>{txt}</b>" if bold else txt, CELL)


def _table(data, first_col_w):
    rest = (FRAME_W - first_col_w) / (len(data[0]) - 1)
    t = Table(data, colWidths=[first_col_w] + [rest] * (len(data[0]) - 1), repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def results_table(rows, key, digits):
    """rows: [(label, {instance: res})]; cell = av (min – max); best average per instance in bold."""
    best = {}
    for inst in INSTANCES:
        avs = [r[inst][key][0] for _, r in rows if r[inst].get(key)]
        best[inst] = min(avs) if avs else None
    data = [[Paragraph("<b>Method</b>", CELL_L)] + [Paragraph(f"<b>{i}</b>", CELL) for i in INSTANCES]]
    for label, r in rows:
        data.append([Paragraph(esc(label), CELL_L)] + [
            fmt_cell(r[i].get(key), digits, r[i].get(key) is not None and r[i][key][0] == best[i])
            for i in INSTANCES])
    return _table(data, 150)


def _colored(txt, ok):
    return Paragraph(f'<font color="{OK if ok else BAD}"><b>{esc(txt)}</b></font>', CELL)


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
                line.append(_colored(f'{res["local_opt"]} / {res["n_valid"]}{extra}', lo_ok))
            else:
                line.append(Paragraph("–", CELL))
        data.append(line)
    return _table(data, 150)


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
            Paragraph(", ".join(map(str, t)), _style(f"nodes{inst}{run}", "BodyText", fontName="DejaVuMono",
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
    story = [Paragraph("Local search report", H1), Paragraph(esc(AUTHORS), H2),
             Paragraph("Objective = cycle length (Euclidean, rounded to nearest integer) + sum of selected node "
                       "costs. Lower is better.", BODY),
             Paragraph("Pseudocode", H2), Paragraph(esc(PSEUDO_INTRO), BODY)]
    for k, v in PSEUDOCODE.items():
        story += [KeepTogether([Paragraph(esc(k), H3), XPreformatted(esc(v), mono)])]
    story += [Paragraph("Randomisation in the greedy version", H2), Paragraph(RANDOMIZATION, BODY),
              PageBreak(), Paragraph("Computational experiment", H2),
              Paragraph(f"Instances: {', '.join(INSTANCES)}. Eight methods (steepest / greedy × two-nodes / "
                        "two-edges exchange × random / greedy start), each run 200 times. Each cell: average "
                        "(min – max) over the valid runs; the best average on each instance is in bold.", BODY),
              Paragraph("Objective function", H3), results_table(rows, "obj", 2),
              Paragraph("Running time [ms]", H3),
              Paragraph("Time of one run: building its starting solution + the local search.", BODY),
              results_table(rows, "time", 2),
              KeepTogether([
                  Paragraph("Validation", H3),
                  Paragraph("Valid = the run has ⌈n/2⌉ distinct nodes of the instance. Local optima = no improving "
                            "move remains in the method's neighbourhood (checked independently by the report script; "
                            "\"best Δ\" is the most improving move left).", BODY),
                  validation_table(rows)]), Spacer(1, 6)]
    story += ([Paragraph(f'<font color="{BAD}">• {esc(p)}</font>', SMALL) for p in problems] if problems else
              [Paragraph(f'<font color="{OK}"><b>All runs valid and all runs are local optima.</b></font>', BODY)])
    for inst in INSTANCES:
        story += instance_section(inst, data[inst][0], rows)
    story += [Spacer(1, 12), Paragraph(f'Source code: <a href="{SOURCE_URL}" color="blue">{SOURCE_URL}</a>', BODY),
              Paragraph("Conclusions", H2), Paragraph(CONCLUSIONS, BODY)]
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                      bottomMargin=MARGIN, title="TSP local search report", author=AUTHORS
                      ).build(story, onFirstPage=_page_number, onLaterPages=_page_number)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "Local_search/bin/Debug/net10.0",
                    help="folder with the Lab3 result CSVs and TSPA.csv/TSPB.csv")
    ap.add_argument("--out", type=Path, default=HERE / "report.pdf")
    a = ap.parse_args()
    self_test()

    rows = [(label, {}) for _, label, _ in LS_METHODS]
    data = {}
    for inst in INSTANCES:
        inst_file = a.results / f"{inst}.csv"
        if not inst_file.exists():
            raise SystemExit(f"{inst_file} not found")
        nodes, D, c = data[inst] = load_instance(inst_file)
        for (key, label, intra), (_, r) in zip(LS_METHODS, rows):
            print(f"checking {inst} {label} ...", flush=True)
            r[inst] = evaluate(a.results / f"{inst.lower()}{key}.csv", nodes, D, c, intra)

    problems = [f"{inst} / {label}: {e}" for label, r in rows for inst in INSTANCES
                for e in r[inst]["errors"][:3] + ([f"… {len(r[inst]['errors']) - 3} more errors"]
                                                   if len(r[inst]["errors"]) > 3 else [])]
    problems += [f"{inst} / {label}: {r[inst]['n_valid'] - r[inst]['local_opt']} runs are not local optima "
                 f"(best remaining Δ = {r[inst]['worst_delta']})"
                 for label, r in rows for inst in INSTANCES
                 if r[inst].get("n_valid") and r[inst]["local_opt"] != r[inst]["n_valid"]]

    build_pdf(a.out, rows, data, problems)
    print(f"\nwrote {a.out}")
    print("\n".join(problems) if problems else "all checks passed")


if __name__ == "__main__":
    main()
