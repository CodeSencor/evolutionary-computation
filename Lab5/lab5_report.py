#!/usr/bin/env python3
"""Validate Lab5 results (list of improving moves) and render report.pdf (same layout as the Lab3 report).

Result files (written by Move_evaluations.cs), named <instance lower>S_ES_R_<base|LM>.csv, e.g. tspaS_ES_R_LM.csv:
for every run one line with its running time in µs, then ceil(n/2) lines "x;y;cost".

Checks per run: correct number of nodes, every node exists in the instance, no repeats, and that no
improving move is left in the neighbourhood (inter-route node exchange + two-edges exchange),
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

# ---- Text you fill in (CONCLUSIONS: reportlab markup, e.g. <b>, &lt;) --------------------------------
AUTHORS = "Paweł Charkiewicz 160288, Piotr Franc 160306"
SOURCE_URL = "https://github.com/CodeSencor/evolutionary-computation"
PSEUDO_INTRO = (
    "All methods read the instance into masterArray, which stores every node (x, y, cost). M = number of "
    "runs = 200, N = solution size = ⌈n/2⌉, d(a, b) = Euclidean distance rounded to the nearest integer, "
    "s = the current solution (a cycle s[0] → s[1] → … → s[N − 1] → s[0]), prev(v) / next(v) = the nodes "
    "before / after node v on the cycle, Δ = change of the objective (negative = improvement). "
    "Both methods use the neighbourhood that was the best in assignment 3: the exchange of a selected and a not "
    "selected node (inter-route) + the two-edges exchange (intra-route), steepest search, random starting "
    "solutions. Both are run on the same 200 starting solutions. All Δ are computed from the changed edges only.")
PSEUDOCODE = {  # title -> pseudocode (plain text, shown in <pre>)
    "Moves": """\
Node exchange (a, v, b, u): removes edges (a, v), (v, b), puts the not selected node u in place of v.
    Δ = d(a, u) + d(u, b) + cost(u) − d(a, v) − d(v, b) − cost(v)
    Applicable: u is not selected and v lies between a and b (the direction does not matter).
Two-edges exchange (a, a', b, b'): removes edges a → a' and b → b', adds edges (a, b) and (a', b').
    Δ = d(a, b) + d(a', b') − d(a, a') − d(b, b')
    Same relative direction: a → a' and b → b' in s, or a' → a and b' → b (both reversed).
    Apply: reverse the path a' … b (both reversed: the path a … b'); the shorter side of the
    cycle is reversed, which gives the same cycle.
    Relatively inverted move of the edges a → a', b → b' of s: (a', a, b, b'), adds (a', b), (a, b').
    It is not applicable now, but it is after a later move reverses exactly one of the two edges.
    One move has 4 notations: (a, a', b, b'), (a', a, b', b), (b, b', a, a'), (b', b, a', a);
    LM keeps the one starting with the smallest node, so it holds no duplicates.""",
    "Baseline: steepest local search": """\
1. s = random starting solution.
2. Repeat:
3.     Evaluate every node exchange (selected v, not selected u) and every two-edges exchange
       of two non-adjacent edges of s.
4.     If the best Δ ≥ 0: s is a local optimum, stop.
5.     Apply the best move.""",
    "LM: steepest local search with the list of improving moves": """\
1. s = random starting solution, LM = empty list of moves ordered by Δ (best first).
2. Evaluate all moves of s as in the baseline, plus for every pair of edges the relatively
   inverted two-edges exchange; add every move with Δ < 0 to LM.
3. Repeat:
4.     Browse LM from the best move:
         a removed edge is not in s any more (or u is selected)  -> remove the move from LM
         removed edges in s, different relative direction        -> keep it, browse further
         removed edges in s, same relative direction (or both
         reversed)                                               -> this is the move m, stop
5.     If m was found: remove m from LM, apply it, add the new improving moves (next block).
6.     Else if LM was just filled in step 2 or 7: s is a local optimum, stop.
7.     Else evaluate all moves of s as in step 2 (a safety net: finds nothing new in practice).""",
    "LM: new moves after applying a move": """\
1. changed = the nodes whose neighbours changed:
       node exchange (a, v, b, u): a, u, b     two-edges exchange (a, a', b, b'): a, a', b, b'
2. For every changed node w and every not selected node u: evaluate the node exchange of w, u.
3. After a node exchange v is not selected any more: evaluate putting v in place of every
   selected node.
4. Every new edge is w → next(w) for a changed node w. For every such edge and every other edge
   of s sharing no node with it: evaluate the two-edges exchange and its inverted version.
5. Add every evaluated move with Δ < 0 to LM. All other moves keep the Δ evaluated earlier:
   Δ depends only on the removed and added edges, which did not change.""",
    "RANDOM: random starting solution": """\
1. Shuffle all node indices (one generator with seed 42 for all 200 starts).
2. The first N of them, in that order, form the cycle.""",
}
CONCLUSIONS = (
    "The list of improving moves does not change what the search does, only how fast it finds the next move: LM "
    "always holds every improving move of the current solution, so the best applicable move in LM is the move the "
    "baseline would pick. The quality of both methods is therefore the same (the small differences in the averages "
    "come only from ties between moves with equal Δ), and every run of both methods ends in a local optimum. "
    "LM is about 2.5 times as fast. After a move only the moves touching the few changed edges and nodes are "
    "evaluated, instead of the whole neighbourhood of about 15 000 moves, but this gain is partly eaten by the "
    "initial evaluation (which includes the relatively inverted moves) and by keeping the sorted list: from a "
    "random start a large part of the evaluated moves is improving and has to be inserted into LM. The fallback "
    "full evaluation (step 7) never found a new move, which confirms that the incremental update misses nothing.")
# --------------------------------------------------------------------------------------------------

LS_METHODS = [  # (file key, label, intra-route move)
    ("S_ES_R_base", "Steepest LS, baseline (all moves evaluated every iteration)", "ES"),
    ("S_ES_R_LM", "Steepest LS, list of improving moves (LM)", "ES"),
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
    story = [Paragraph("Local search with the list of improving moves", H1), Paragraph(esc(AUTHORS), H2),
             Paragraph("Objective = cycle length (Euclidean, rounded to nearest integer) + sum of selected node "
                       "costs. Lower is better.", BODY),
             Paragraph("Pseudocode", H2), Paragraph(esc(PSEUDO_INTRO), BODY)]
    for k, v in PSEUDOCODE.items():
        story += [KeepTogether([Paragraph(esc(k), H3), XPreformatted(esc(v), mono)])]
    times = {i: [r[i]["time"][0] if r[i].get("time") else None for _, r in rows] for i in INSTANCES}
    speedup = ", ".join(f"{i} {t[0] / t[1]:.2f}×" for i, t in times.items() if None not in t)
    story += [PageBreak(), Paragraph("Computational experiment", H2),
              Paragraph(f"Instances: {', '.join(INSTANCES)}. Two methods (baseline steepest LS and steepest LS "
                        "with the list of improving moves), each run 200 times from the same random starting "
                        "solutions. Each cell: average (min – max) over the valid runs; the best average on each "
                        "instance is in bold.", BODY),
              Paragraph("Objective function", H3), results_table(rows, "obj", 2),
              Paragraph("Running time [µs]", H3),
              Paragraph("Time of one run: the local search from the random starting solution (for LM including "
                        "the initial evaluation of all moves).", BODY),
              results_table(rows, "time", 2),
              Paragraph(f"Average speed-up of LM over the baseline (average baseline time / average LM time): "
                        f"{speedup or 'n/a'}.", BODY),
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
                      bottomMargin=MARGIN, title="TSP local search with the list of improving moves", author=AUTHORS
                      ).build(story, onFirstPage=_page_number, onLaterPages=_page_number)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE,
                    help="folder with the Lab5 result CSVs and TSPA.csv/TSPB.csv (default: this folder)")
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
