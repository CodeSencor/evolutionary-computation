#!/usr/bin/env python3
"""Validate Lab4 candidate-move local-search results and render report.pdf (same layout as the Lab3 report).

Result files (written by CandidateLocalSearch.cs), named <instance lower>S_<ES|NS>_R_<base|C<K>>.csv, e.g.
tspaS_ES_R_base.csv (steepest, two-edges exchange, random start, all moves = baseline) and
tspaS_ES_R_C10.csv (the same, but only candidate moves, K = 10 nearest nodes).
For every run one line with its running time in ms (decimal fraction), then ceil(n/2) lines "x;y;cost".
Which methods appear in the report is discovered from the files found in the results folder.

Checks per run: correct number of nodes, every node exists in the instance, no repeats, and that no improving
move is left in the method's own neighbourhood, recomputed independently here:
  baseline  -> inter-route node exchange + intra-route move over ALL moves,
  candidate -> only the moves that introduce a candidate edge (the candidate lists are rebuilt here).
For candidate methods it is also reported how many runs are local optima of the FULL neighbourhood.

Requires: numpy, reportlab, matplotlib (only its bundled DejaVu fonts are used).

Usage: python lab4_report.py [--results DIR] [--out FILE]
"""
import argparse
import html
import re
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
RESULT_DIRS = ["CandidateSearch", "Candidate_search"]  # default places of the project folder (first that exists)
INSTANCE_DIRS = [HERE / "CandidateSearch", HERE / "Candidate_search", HERE, HERE.parent / "Lab2"]

# ---- Text you fill in (CONCLUSIONS: reportlab markup, e.g. <b>, &lt;) ---------------------------------------------
AUTHORS = "Paweł Charkiewicz 160288, Piotr Franc 160306"
SOURCE_URL = "https://github.com/CodeSencor/evolutionary-computation"
PSEUDO_INTRO = (
    "All methods read the instance into masterArray, which stores every node (x, y, cost). M = number of "
    "runs = 200, N = solution size = ⌈n/2⌉, d(a, b) = Euclidean distance rounded to the nearest integer, "
    "s = the current solution (a cycle s[0] → s[1] → … → s[N − 1] → s[0]), prev(i) / next(i) = the nodes "
    "before / after position i on the cycle, Δ = change of the objective (negative = improvement). "
    "Every method is the steepest local search from a random starting solution with one intra-route move "
    "(NODE_SWAP or EDGE_SWAP) and the inter-route move (exchange of a selected and a not selected node). The "
    "baseline evaluates every move of the neighbourhood; the candidate version evaluates only the moves "
    "that introduce at least one candidate edge into the solution. All Δ are computed from the changed "
    "edges only; no neighbour solution is built to evaluate a move.")
PSEUDOCODE = {  # title -> pseudocode (plain text, shown in <pre>)
    "Local search (common loop)": """\
0. Compute an n × n distance matrix d between every pair of nodes.
1. CANDIDATES only: build the candidate lists once per instance (not included in the running times).
2. Create M random starting solutions (the same seed for every method, so every method starts from
   exactly the same M solutions).
3. For every starting solution s:
4.     Start the timer (it also covers building the starting solution).
5.     Repeat:
6.         pos[v] = position of node v in s, or −1 when v is not selected.
7.         Evaluate the moves (BASELINE: all of them, CANDIDATES: only the candidate moves) and
           remember the one with the smallest Δ (steepest step).
8.     Until the smallest Δ found is ≥ 0, otherwise apply that move and go on.
9.     Stop the timer and save s and its running time = construction time + search time.""",
    "Candidate lists": """\
For every node v:
1. For every other node u compute the key  d(v, u) + cost(u).
2. cand(v) = the K nodes u with the smallest key (ties: the smaller node index first).""",
    "Candidate moves: only moves that add a candidate edge (a, c)": """\
For every position i in s (a = s[i]) and for every c in cand(a):
1. Let j = pos[c].
2. If c is selected (j ≥ 0): two intra-route moves add the edge (a, c)
       EDGE_SWAP: the edge swaps (i, j) and (i − 1, j − 1)
                  (the first adds (s[i], s[j]) and (s[i+1], s[j+1]), the second (s[i−1], s[j−1]) and
                  (s[i], s[j]))
       NODE_SWAP: swap s[j] with s[i + 1] and swap s[j] with s[i − 1] (c lands next to a)
   If c is not selected (j = −1): two inter-route moves add the edge (a, c)
       exchange the node at position i + 1 for c and exchange the node at position i − 1 for c.
3. Positions are taken modulo N. EDGE_SWAP pairs are ordered (p < q) and skipped when q − p < 2 or
   (p, q) = (0, N − 1); NODE_SWAP pairs with p = q are skipped.
Only these moves are generated and evaluated; all other moves are never visited. Exchanging a selected node
with its own nearest neighbour is not a candidate move, because it need not add any candidate edge.""",
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
1. Take positions i < j, edges (s[i], s[i+1]) and (s[j], s[j+1]).
2. Δ = d(s[i], s[j]) + d(s[i+1], s[j+1]) − d(s[i], s[i+1]) − d(s[j], s[j+1])
3. Apply: reverse the segment s[i+1] … s[j]; the cycle now uses edges (s[i], s[j]) and (s[i+1], s[j+1]).""",
    "STEEP: steepest local search (one step)": """\
1. Over the evaluated moves (all of them for the baseline, only the candidate moves otherwise):
   remember the move with the smallest Δ; the first one found wins a tie.
2. If that Δ ≥ 0: s is a local optimum of the (candidate) neighbourhood, stop.
3. Otherwise apply the move (an exchange also updates the set of selected nodes).""",
    "RANDOM: random starting solution": """\
1. Start with an empty cycle.
2. Until it holds N nodes: draw a random node index that is not yet used and append it.""",
}
RANDOMIZATION = (
    "The steepest search itself is deterministic: it always evaluates the moves in the same order and applies "
    "the best one (the first one found wins a tie). The only randomness is in the 200 random starting "
    "solutions. Every method restarts the random generator with the same seed (42), so all methods, the "
    "baseline and every K, start from exactly the same 200 solutions and the differences between them come "
    "from the neighbourhood alone.")
CONCLUSIONS = "The main task was to improve the speed of steepest local search by using candidate moves." \
"As we can see in instance B it even improves the performance. We ran the code with 3 different seeds," \
"and for each of them in instance B the performance improved with the use of candidate moves. We think it means that" \
"maybe the few steepest first steps may seem the best but lead into a local optimum which is impossible" \
"to escape later. Regarding the types of exchanges, the random start lead to two nodes " \
"exchange lead to notoriously poor results."  # your own text; the comparison generated from the tables below is always printed
# --------------------------------------------------------------------------------------------------

KEY_RE = re.compile(r"^(tsp[ab])S_(ES|NS)_R_(base|C(\d+))$", re.IGNORECASE)
MOVE_NAMES = {"ES": "two-edges exchange", "NS": "two-nodes exchange"}


def method_label(move, k):
    base = f"Steepest LS, {MOVE_NAMES[move]}, random start"
    return f"{base}, baseline (all moves)" if k == 0 else f"{base}, candidate moves K = {k}"


def discover_methods(results):
    """[(file key, label, move 'ES'|'NS', K (0 = baseline))] sorted by move, baseline first, then K ascending."""
    found = set()
    for p in results.glob("*.csv"):
        m = KEY_RE.match(p.stem)
        if m:
            found.add((m.group(2).upper(), 0 if m.group(3).lower() == "base" else int(m.group(4))))
    return [(f"S_{mv}_R_{'base' if k == 0 else f'C{k}'}", method_label(mv, k), mv, k) for mv, k in sorted(found)]


def read_rows(path):
    """Node lines -> (x, y, cost) ints; a single-value line is a run time (decimal comma accepted)."""
    rows = [line.split(";") for line in path.read_text().split() if line.strip()]
    return [(float(r[0].replace(",", ".")),) if len(r) == 1 else tuple(int(v) for v in r) for r in rows]


def distances(nodes):
    d = np.hypot(*(nodes[:, None, :2] - nodes[None, :, :2]).transpose(2, 0, 1))
    return np.round(d).astype(np.int64)


def load_instance(path):
    nodes = np.array(read_rows(path), dtype=np.int64)
    return nodes, distances(nodes), nodes[:, 2]


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


def build_candidates(D, c, k):
    """cand[v] = the k nodes u != v with the smallest d(v, u) + cost(u); ties -> smaller index (stable sort)."""
    key = D + c[None, :]
    np.fill_diagonal(key, key.max() + 1)
    return np.argsort(key, axis=1, kind="stable")[:, :k]


def _exchange_deltas(t, D, c, pos, new):
    """Delta of replacing the node at each position pos[i] by node new[i]."""
    prv, nxt = np.roll(t, 1), np.roll(t, -1)
    old = t[pos]
    return (D[prv[pos], new] + D[new, nxt[pos]] + c[new] - (D[prv[pos], old] + D[old, nxt[pos]] + c[old]))


def _edge_deltas(t, D, lo, hi):
    """Delta of the two-edges exchange of positions lo < hi (reverse t[lo+1..hi])."""
    nxt = np.roll(t, -1)
    return D[t[lo], t[hi]] + D[nxt[lo], nxt[hi]] - D[t[lo], nxt[lo]] - D[t[hi], nxt[hi]]


def _node_swap_deltas(t, D, lo, hi):
    """Delta of swapping the nodes at positions lo < hi: recompute every swapped tour (adjacent nodes included)."""
    tours = np.tile(t, (len(lo), 1))
    rows = np.arange(len(lo))
    tours[rows, lo], tours[rows, hi] = t[hi], t[lo]
    return D[tours, np.roll(tours, -1, axis=1)].sum(1) - D[t, np.roll(t, -1)].sum()


def best_delta(t, D, c, intra):
    """Smallest objective change over ALL inter-route exchanges + intra-route moves (>= 0 means local optimum)."""
    n = len(t)
    unused = np.setdiff1d(np.arange(len(c)), t)
    pos, new = np.repeat(np.arange(n), len(unused)), np.tile(unused, n)
    best = _exchange_deltas(t, D, c, pos, new).min() if len(unused) else 0
    lo, hi = np.triu_indices(n, 1)  # i < j only
    if intra == "ES":  # adjacent pairs give 0, so they never beat a real improvement
        best = min(best, _edge_deltas(t, D, lo, hi).min())
    else:
        best = min(best, _node_swap_deltas(t, D, lo, hi).min())
    return int(best)


def candidate_best_delta(t, D, c, intra, cand):
    """Smallest Δ over the candidate moves only (>= 0 means a local optimum of the candidate neighbourhood).

    Rebuilds the move set exactly as described in the report: for every position i and every c in cand(t[i]),
    the moves that put the edge (t[i], c) into the solution."""
    n = len(t)
    where = np.full(len(c), -1)
    where[t] = np.arange(n)
    swaps, exch = set(), set()
    for i in range(n):
        for u in cand[t[i]]:
            j = where[u]
            if j >= 0:
                pairs = [(i, j), (i - 1, j - 1)] if intra == "ES" else [(j, i + 1), (j, i - 1)]
                for x, y in pairs:
                    lo, hi = sorted((x % n, y % n))
                    if (intra == "ES" and (hi - lo < 2 or (lo == 0 and hi == n - 1))) or (intra == "NS" and lo == hi):
                        continue
                    swaps.add((lo, hi))
            else:
                exch.update({((i + 1) % n, int(u)), ((i - 1) % n, int(u))})
    best = 0
    if exch:
        pos, new = map(np.array, zip(*exch))
        best = min(best, _exchange_deltas(t, D, c, pos, new).min())
    if swaps:
        lo, hi = map(np.array, zip(*swaps))
        best = min(best, (_edge_deltas if intra == "ES" else _node_swap_deltas)(t, D, lo, hi).min())
    return int(best)


def self_test():
    rng = np.random.default_rng(0)
    nodes = np.column_stack([rng.integers(0, 100, (12, 2)), rng.integers(0, 50, 12)])
    D = distances(nodes)
    c, t = nodes[:, 2], rng.permutation(12)[:6]
    base = objective(t, D, c)
    inter = [objective(np.where(t == a, u, t), D, c) - base for a in t for u in set(range(12)) - set(t)]
    swaps = [objective(np.where(t == a, b, np.where(t == b, a, t)), D, c) - base for a in t for b in t if a != b]
    opt2 = [objective(np.r_[t[:i + 1], t[i + 1:j + 1][::-1], t[j + 1:]], D, c) - base
            for i in range(6) for j in range(i + 1, 6)]
    assert best_delta(t, D, c, "NS") == min(inter + swaps)
    assert best_delta(t, D, c, "ES") == min(inter + opt2)
    # candidate moves: with every other node as a candidate they cover the whole neighbourhood ...
    allc = build_candidates(D, c, 11)
    assert all(candidate_best_delta(t, D, c, m, allc) == best_delta(t, D, c, m) for m in ("NS", "ES"))
    # ... and with a few candidates they are a subset of it (never a better move than the full neighbourhood)
    for k in (1, 2, 3):
        cand = build_candidates(D, c, k)
        assert all(candidate_best_delta(t, D, c, m, cand) >= best_delta(t, D, c, m) for m in ("NS", "ES"))
    # the candidate lists: sorted by d + cost, never contain the node itself
    cand = build_candidates(D, c, 4)
    assert all(v not in cand[v] for v in range(12))
    assert all(list(D[v, cand[v]] + c[cand[v]]) == sorted(D[v, cand[v]] + c[cand[v]]) for v in range(12))
    runs, times = split_runs([(5.5,), (1, 2, 3), (4, 5, 6), (7.25,), (1, 2, 3), (4, 5, 6)], 2)
    assert times == [5.5, 7.25] and len(runs) == 2 and runs[1] == [(1, 2, 3), (4, 5, 6)]


def stats(values):
    return (float(np.mean(values)), min(values), max(values)) if values else None


def evaluate(path, nodes, D, c, move, k):
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
    good = [t for t in runs if t is not None]
    full = [best_delta(t, D, c, move) for t in good]
    if k:  # candidate method: local optimum of its own (candidate) neighbourhood + of the full one
        cand = build_candidates(D, c, k)
        own = [candidate_best_delta(t, D, c, move, cand) for t in good]
    else:
        own = full
    res["local_opt"] = sum(d >= 0 for d in own)
    res["worst_delta"] = min(own) if own else None
    res["full_opt"] = sum(d >= 0 for d in full)
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


def fmt_cell(s, digits, bold, int_range):
    if s is None:
        return Paragraph("n/a", CELL)
    av, lo, hi = s
    mm = 0 if int_range else digits
    txt = f"{av:.{digits}f} ({lo:.{mm}f} – {hi:.{mm}f})"
    return Paragraph(f"<b>{txt}</b>" if bold else txt, CELL)


def _table(data, first_col_w):
    rest = (FRAME_W - first_col_w) / (len(data[0]) - 1)
    t = Table(data, colWidths=[first_col_w] + [rest] * (len(data[0]) - 1), repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def results_table(rows, key, digits, int_range):
    """rows: [(label, {instance: res}, move, K)]; cell = av (min – max); best average per instance in bold."""
    best = {}
    for inst in INSTANCES:
        avs = [r[inst][key][0] for _, r, _, _ in rows if r[inst].get(key)]
        best[inst] = min(avs) if avs else None
    data = [[Paragraph("<b>Method</b>", CELL_L)] + [Paragraph(f"<b>{i}</b>", CELL) for i in INSTANCES]]
    for label, r, _, _ in rows:
        data.append([Paragraph(esc(label), CELL_L)] + [
            fmt_cell(r[i].get(key), digits, r[i].get(key) is not None and r[i][key][0] == best[i], int_range)
            for i in INSTANCES])
    return _table(data, 190)


def baseline_of(rows, move):
    return next((r for _, r, mv, k in rows if mv == move and k == 0), None)


def comparison(rows):
    """{(label, instance): (speed-up, objective change in %)} of every candidate method vs its own baseline."""
    out = {}
    for label, r, move, k in rows:
        base = baseline_of(rows, move)
        for inst in INSTANCES:
            if k and base and r[inst].get("time") and base[inst].get("time") and r[inst].get("obj") and base[inst].get("obj"):
                tb, ob = base[inst]["time"][0], base[inst]["obj"][0]
                t, o = r[inst]["time"][0], r[inst]["obj"][0]
                out[label, inst] = (tb / t if t > 0 else float("inf"), 100 * (o - ob) / ob)
    return out


def comparison_table(rows, comp):
    data = [[Paragraph("<b>Method</b>", CELL_L)] + [
        Paragraph(f"<b>{i} {what}</b>", CELL) for i in INSTANCES for what in ("speed-up", "objective")]]
    for label, r, move, k in rows:
        if not k:
            continue
        line = [Paragraph(esc(label), CELL_L)]
        for inst in INSTANCES:
            v = comp.get((label, inst))
            line += [Paragraph(f"{v[0]:.2f}×" if v else "n/a", CELL), Paragraph(f"{v[1]:+.2f} %" if v else "n/a", CELL)]
        data.append(line)
    return _table(data, 190) if len(data) > 1 else None


def summary_text(rows, comp):
    parts = []
    for inst in INSTANCES:
        have = [(label, r[inst]) for label, r, _, _ in rows if r[inst].get("obj") and r[inst].get("time")]
        if not have:
            continue
        best_q = min(have, key=lambda x: x[1]["obj"][0])
        fastest = min(have, key=lambda x: x[1]["time"][0])
        s = (f"<b>{inst}:</b> the best average objective is {best_q[1]['obj'][0]:.2f} ({esc(best_q[0])}); the "
             f"fastest method is “{esc(fastest[0])}” with {fastest[1]['time'][0]:.3f} ms per run on average")
        v = comp.get((fastest[0], inst))
        if v:
            s += f", {v[0]:.2f}× faster than its baseline with an average objective {v[1]:+.2f} % different"
        parts.append(s + ".")
    return " ".join(parts)


def _colored(txt, ok):
    return Paragraph(f'<font color="{OK if ok else BAD}"><b>{esc(txt)}</b></font>', CELL)


def validation_table(rows):
    data = [[Paragraph("<b>Method</b>", CELL_L)] + [
        Paragraph(f"<b>{i} {what}</b>", CELL) for i in INSTANCES for what in ("valid", "local optima", "full nbhd.")]]
    for label, r, move, k in rows:
        line = [Paragraph(esc(label), CELL_L)]
        for inst in INSTANCES:
            res = r[inst]
            if not res["exists"]:
                line += [_colored("missing", False), Paragraph("", CELL), Paragraph("", CELL)]
                continue
            ok = res["n_valid"] == res["n_runs"] and not res["errors"]
            line.append(_colored(f'{res["n_valid"]} / {res["n_runs"]}', ok))
            if res["n_valid"]:
                lo_ok = res["local_opt"] == res["n_valid"]
                extra = "" if lo_ok else f' (best Δ {res["worst_delta"]})'
                line.append(_colored(f'{res["local_opt"]} / {res["n_valid"]}{extra}', lo_ok))
                # informational for candidate methods (they only promise a local optimum of the candidate nbhd.)
                line.append(Paragraph(f'{res["full_opt"]} / {res["n_valid"]}', CELL))
            else:
                line += [Paragraph("–", CELL), Paragraph("–", CELL)]
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
    for label, r, _, _ in rows:
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
            Paragraph(", ".join(map(str, t)), _style(f"nodes{inst}{len(out)}", "BodyText", fontName="DejaVuMono",
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
    ks = sorted({k for _, _, _, k in rows if k})
    moves = sorted({mv for _, _, mv, _ in rows})
    comp = comparison(rows)
    n_runs = max((r[i].get("n_runs", 0) for _, r, _, _ in rows for i in INSTANCES), default=0)
    story = [Paragraph("Local search with candidate moves report", H1), Paragraph(esc(AUTHORS), H2),
             Paragraph("Objective = cycle length (Euclidean, rounded to nearest integer) + sum of selected node "
                       "costs. Lower is better.", BODY),
             Paragraph("Pseudocode", H2), Paragraph(esc(PSEUDO_INTRO), BODY)]
    for k, v in PSEUDOCODE.items():
        story += [KeepTogether([Paragraph(esc(k), H3), XPreformatted(esc(v), mono)])]
    story += [Paragraph("Randomisation", H2), Paragraph(RANDOMIZATION, BODY),
              PageBreak(), Paragraph("Computational experiment", H2),
              Paragraph(f"Instances: {', '.join(INSTANCES)}. Steepest local search from random starting solutions, "
                        f"intra-route move: {' and '.join(MOVE_NAMES[m] for m in moves)}; the baseline uses all "
                        f"moves, the candidate versions use K = {', '.join(map(str, ks)) or '–'} nearest nodes "
                        f"(by edge length + node cost) per node. Each method was run {n_runs} times. Each cell: average "
                        "(min – max) over the valid runs; the best average on each instance is in bold.", BODY),
              Paragraph("Objective function", H3), results_table(rows, "obj", 2, True),
              Paragraph("Running time [ms]", H3),
              Paragraph("Time of one run: building its starting solution + the local search. The candidate lists "
                        "are built once per instance and are not included.", BODY),
              results_table(rows, "time", 3, False)]
    ct = comparison_table(rows, comp)
    if ct is not None:
        story += [Paragraph("Candidate moves versus the baseline", H3),
                  Paragraph("Speed-up = average baseline time / average time of the method; objective = relative "
                            "change of the average objective against the baseline (positive = worse).", BODY), ct]
    story += [KeepTogether([
                  Paragraph("Validation", H3),
                  Paragraph("Valid = the run has ⌈n/2⌉ distinct nodes of the instance. Local optima = no improving "
                            "move remains in the method's own neighbourhood (all moves for the baseline, only the "
                            "candidate moves for K = …; checked independently by the report script; \"best Δ\" is "
                            "the most improving move left). Full nbhd. = how many runs are also local optima of "
                            "the full neighbourhood (informational).", BODY),
                  validation_table(rows)]), Spacer(1, 6)]
    story += ([Paragraph(f'<font color="{BAD}">• {esc(p)}</font>', SMALL) for p in problems] if problems else
              [Paragraph(f'<font color="{OK}"><b>All runs valid and all runs are local optima of their own '
                         'neighbourhood.</b></font>', BODY)])
    for inst in INSTANCES:
        story += instance_section(inst, data[inst][0], rows)
    story += [Spacer(1, 12), Paragraph(f'Source code: <a href="{SOURCE_URL}" color="blue">{SOURCE_URL}</a>', BODY),
              Paragraph("Conclusions", H2), Paragraph(summary_text(rows, comp), BODY)]
    if CONCLUSIONS:
        story.append(Paragraph(CONCLUSIONS, BODY))
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                      bottomMargin=MARGIN, title="TSP candidate moves report", author=AUTHORS
                      ).build(story, onFirstPage=_page_number, onLaterPages=_page_number)


def find_instance(inst, results):
    for folder in [results] + INSTANCE_DIRS:
        f = folder / f"{inst}.csv"
        if f.exists():
            return f
    raise SystemExit(f"{inst}.csv not found (looked in {results} and {', '.join(map(str, INSTANCE_DIRS))})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None,
                    help="folder with the result CSVs (default: the CandidateSearch project folder next to this "
                         "script; TSPA.csv/TSPB.csv are looked up there too)")
    ap.add_argument("--out", type=Path, default=HERE / "report.pdf")
    a = ap.parse_args()
    results = a.results or next((HERE / d for d in RESULT_DIRS if (HERE / d).is_dir()), HERE)
    self_test()

    methods = discover_methods(results)
    if not methods:
        raise SystemExit(f"no result files like tspaS_ES_R_base.csv / tspaS_ES_R_C10.csv found in {results}")
    print(f"results folder: {results}; methods: {', '.join(label for _, label, _, _ in methods)}")
    rows = [(label, {}, move, k) for _, label, move, k in methods]
    data = {}
    for inst in INSTANCES:
        nodes, D, c = data[inst] = load_instance(find_instance(inst, results))
        for (key, label, move, k), (_, r, _, _) in zip(methods, rows):
            print(f"checking {inst} {label} ...", flush=True)
            r[inst] = evaluate(results / f"{inst.lower()}{key}.csv", nodes, D, c, move, k)

    problems = [f"{inst} / {label}: {e}" for label, r, _, _ in rows for inst in INSTANCES
                for e in r[inst]["errors"][:3] + ([f"… {len(r[inst]['errors']) - 3} more errors"]
                                                   if len(r[inst]["errors"]) > 3 else [])]
    problems += [f"{inst} / {label}: {r[inst]['n_valid'] - r[inst]['local_opt']} runs are not local optima "
                 f"(best remaining Δ = {r[inst]['worst_delta']})"
                 for label, r, _, _ in rows for inst in INSTANCES
                 if r[inst].get("n_valid") and r[inst]["local_opt"] != r[inst]["n_valid"]]

    build_pdf(a.out, rows, data, problems)
    print(f"\nwrote {a.out}")
    print("\n".join(problems) if problems else "all checks passed")


if __name__ == "__main__":
    main()