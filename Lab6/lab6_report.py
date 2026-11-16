#!/usr/bin/env python3
"""Validate Lab6 (MSLS & ILS) results and render report.pdf.

Reads:
- Instance files: TSPA.csv, TSPB.csv
- MSLS paths: tspaS_MSLS.csv, tspbS_MSLS.csv (times in µs, followed by ceil(n/2) lines "x;y;cost")
- ILS summary tables: tspa_experiment_summary.csv, tspb_experiment_summary.csv
- ILS best paths: tspaS_ILS.csv, tspbS_ILS.csv

Usage: python lab6_report.py [--results DIR] [--out FILE]
"""
import argparse
import csv
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

AUTHORS = "Paweł Charkiewicz 160288, Piotr Franc 160306"
SOURCE_URL = "https://github.com/CodeSencor/evolutionary-computation"

PSEUDO_INTRO = (
    "n = number of nodes of the instance, N = ⌈n/2⌉ = size of a solution, M = 20 = number of runs, "
    "d(a, b) = Euclidean distance rounded to the nearest integer, "
    "s = the current solution (a cycle s[0], …, s[N − 1]; positions are taken modulo N), "
    "pos[v] = position of node v in s or −1 if v is not selected, Δ = change of the objective "
    "(negative = improvement). Multiple Start Local Search (MSLS) runs steepest local search (using the list "
    "of improving moves, LM) from 400 random starting solutions per run. Iterated Local Search (ILS) starts "
    "from a local optimum, iteratively perturbs the solution, applies local search, and accepts the new "
    "solution if it improves the objective (minimization, '<'), repeating until the average MSLS running time "
    "is reached.")

PSEUDOCODE = {
    "MSLS: Multiple Start Local Search": """\
Input: instance, MslsStarts = 400
1. best_obj ← ∞
2. for start = 1, …, MslsStarts:
3.     s ← LocalSearch(RandomStart())
4.     if objective(s) < best_obj:
5.         best_obj ← objective(s)
6.         best_s ← s
7. return best_s""",
    "ILS: Iterated Local Search": """\
Input: instance, time_limit (from MSLS average time), perturbation_distribution
1. s ← LocalSearch(RandomStart())
2. current_s ← s, current_obj ← objective(s)
3. pt ← 0
4. while elapsed_time < time_limit:
5.     perturb_count ← distribution[pt]
6.     pt ← min(pt + 1, length(distribution) - 1)
7.     s' ← Perturb(current_s, perturb_count)  # In-place random node substitution (Strategy 0)
8.     s_next ← LocalSearch(s')
9.     if objective(s_next) < current_obj:
10.        current_s ← s_next, current_obj ← objective(s_next)
11.        pt ← 0  # reset perturbation sequence index on improvement
12. return current_s""",
    "Perturbation Strategy (In-place substitution of nodes)": """\
- Select `count` distinct positions randomly within the current cycle.
- Replace their nodes with randomly chosen unselected nodes in-place, preserving the overall cycle structure.""",
    "Local Search: Steepest Search with Move List (LM)": """\
1. LM ← empty sorted set of improving moves (ordered by Δ ascending)
2. EvaluateAll: find all improving inter-route (node exchange) and intra-route (2-edges exchange) moves (Δ < 0), add to LM
3. Repeat:
4.     Find best valid move m in LM using incremental checks and updates (EvaluateNew).
5.     If m found: apply m, update LM.
6.     Else: local optimum reached, stop.""",
}

CONCLUSIONS = (
    "Iterated Local Search (ILS) outperforms Multiple Start Local Search (MSLS) by exploiting "
    "the clustering property of the fitness landscape, anchoring its search trajectory to the best-known "
    "local optimum rather than repeatedly restarting from unguided random positions.<br/><br/>"
    "We also implemented an alternative destroy-and-repair perturbation strategy (<b>randomSeparate</b>), "
    "which removes X nodes from the cycle and inserts X unselected nodes into random positions. "
    "However, in-place random node substitution decisively outperformed it by preserving "
    "the broader geometric skeleton of the cycle and enabling higher local search throughput.<br/><br/>"
    "Based on the experiment results, structured perturbation distributions featuring final steps of size 20 "
    "achieved the best "
    "objective values. They provide an optimal balance "
    "between intensive local exploitation and sufficient diversification to escape local traps within the time limit."
)


def read_rows(path):
    rows = [line.split(";") for line in path.read_text().split() if line.strip()]
    return [(float(r[0].replace(",", ".")),) if len(r) == 1 else tuple(int(v) for v in r) for r in rows]


def load_instance(path):
    nodes = np.array(read_rows(path), dtype=np.int64)
    d = np.hypot(*(nodes[:, None, :2] - nodes[None, :, :2]).transpose(2, 0, 1))
    return nodes, np.round(d).astype(np.int64), nodes[:, 2]


def split_runs(rows, size):
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
    return runs, times, errors


def objective(t, D, c):
    return int(D[t, np.roll(t, -1)].sum() + c[t].sum())


def best_delta(t, D, c):
    prv, nxt = np.roll(t, 1), np.roll(t, -1)
    unused = np.setdiff1d(np.arange(len(c)), t)
    inter = (D[prv[:, None], unused] + D[unused[None, :], nxt[:, None]] + c[unused][None, :]
             - (D[prv, t] + D[t, nxt] + c[t])[:, None])
    edge = D[t, nxt]
    intra_d = (D[t[:, None], t] + D[nxt[:, None], nxt] - edge[:, None] - edge[None, :])[
        np.triu_indices(len(t), 1)]
    return int(min(inter.min(), intra_d.min()))


def evaluate_paths(path, nodes, D, c):
    res = {"file": path.name, "exists": path.exists()}
    if not path.exists():
        res["errors"] = [f"missing file {path}"]
        return res
    runs, times, errors = load_runs(path, nodes)
    objs = [objective(t, D, c) if t is not None else None for t in runs]
    valid = [o for o in objs if o is not None]
    times_ms = [t / 1000.0 for t in times] if times else []
    res.update(
        runs=runs,
        objs=objs,
        errors=errors,
        n_runs=len(runs),
        n_valid=len(valid),
        avg_obj=float(np.mean(valid)) if valid else None,
        min_obj=min(valid) if valid else None,
        max_obj=max(valid) if valid else None,
        avg_time_ms=float(np.mean(times_ms)) if times_ms else None,
    )
    if valid:
        res["best"] = min((o, i) for i, o in enumerate(objs) if o is not None)
    deltas = [best_delta(t, D, c) for t in runs if t is not None]
    res["local_opt"] = sum(d >= 0 for d in deltas)
    res["worst_delta"] = min(deltas) if deltas else None
    return res


def load_ils_summary(path):
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dist_formatted = r["Distribution"].replace(";", ", ")
            rows.append({
                "strategy": r["Strategy"],
                "distribution": dist_formatted,
                "avg_obj": float(r["AvgObjective"]),
                "min_obj": int(r["MinObjective"]),
                "max_obj": int(r["MaxObjective"]),
                "avg_time_ms": float(r["AvgTimeMs"]),
                "avg_ls": float(r["AvgLocalSearches"]),
                "n_runs": 20,
            })
    return rows


# ---- PDF Setup ------------------------------------------------------------------------------------
_FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for _name, _file in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                     ("DejaVuMono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(_name, str(_FONTS / _file)))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")

MARGIN = 1.6 * cm
FRAME_W = A4[0] - 2 * MARGIN
OK, BAD = "#1a7f37", "#b42318"
_ss = getSampleStyleSheet()


def _style(name, parent, **kw):
    kw.setdefault("fontName", "DejaVu")
    return ParagraphStyle(name, parent=_ss[parent], **kw)


H1 = _style("h1", "Heading1", fontName="DejaVu-Bold", fontSize=16)
H2 = _style("h2", "Heading2", fontName="DejaVu-Bold", fontSize=13, spaceBefore=10)
H3 = _style("h3", "Heading3", fontName="DejaVu-Bold", fontSize=10.5, spaceBefore=8)
H4 = _style("h4", "Heading4", fontName="DejaVu-Bold", fontSize=9)
BODY = _style("body", "BodyText", fontSize=9.5, leading=13)
CELL = _style("cell", "BodyText", fontSize=8.0, leading=10.0, alignment=TA_RIGHT)
CELL_L = _style("cell_l", "BodyText", fontSize=8.0, leading=10.0)
SMALL = _style("small", "BodyText", fontSize=8, leading=10, spaceBefore=0, spaceAfter=1)
MONO = _style("mono", "Code", fontName="DejaVuMono", fontSize=7.5, leading=9.5, leftIndent=0,
              backColor=colors.HexColor("#f6f6f6"), borderPadding=4, spaceBefore=4, spaceAfter=8)
esc = html.escape


def instance_results_table(msls_res, ils_rows):
    header = [
        Paragraph("<b>Method / Perturbation</b>", CELL_L),
        Paragraph("<b>Avg (Min – Max)</b>", CELL),
        Paragraph("<b>Time [ms]</b>", CELL),
        Paragraph("<b>Runs / LS iters</b>", CELL),
    ]
    data = [header]

    all_avgs = [r["avg_obj"] for r in ils_rows]
    if msls_res.get("avg_obj") is not None:
        all_avgs.append(msls_res["avg_obj"])
    best_avg = min(all_avgs) if all_avgs else None

    if msls_res.get("avg_obj") is not None:
        is_best = (msls_res["avg_obj"] == best_avg)
        obj_str = f"{msls_res['avg_obj']:.2f} ({msls_res['min_obj']} – {msls_res['max_obj']})"
        if is_best:
            obj_str = f"<b>{obj_str}</b>"
        data.append([
            Paragraph("<b>MSLS</b> (400 random starts)", CELL_L),
            Paragraph(obj_str, CELL),
            Paragraph(f"{msls_res['avg_time_ms']:.2f}", CELL),
            Paragraph(f"{msls_res['n_valid']} runs (400 LS/run)", CELL),
        ])
    else:
        data.append([
            Paragraph("<b>MSLS</b> (missing CSV)", CELL_L),
            Paragraph("n/a", CELL),
            Paragraph("n/a", CELL),
            Paragraph("0", CELL),
        ])

    for r in ils_rows:
        is_best = (r["avg_obj"] == best_avg)
        label = f"<b>ILS ({esc(r['strategy'])})</b><br/><font size='7' color='#444444'>dist: [{esc(r['distribution'])}]</font>"
        obj_str = f"{r['avg_obj']:.2f} ({r['min_obj']} – {r['max_obj']})"
        if is_best:
            obj_str = f"<b>{obj_str}</b>"
        data.append([
            Paragraph(label, CELL_L),
            Paragraph(obj_str, CELL),
            Paragraph(f"{r['avg_time_ms']:.2f}", CELL),
            Paragraph(f"{r['n_runs']} runs ({r['avg_ls']:.1f} LS)", CELL),
        ])

    col_widths = [FRAME_W * 0.44, FRAME_W * 0.26, FRAME_W * 0.13, FRAME_W * 0.17]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaeaea")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f7f9fc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def plot_drawing(t, nodes, W=440):
    x, y, cost = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    pad, legend_h = 10, 18
    H = min(W * 0.72, (W - 2 * pad) * max(np.ptp(y), 1) / max(np.ptp(x), 1) + 2 * pad)
    k = W / 640
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
    for j in range(30):
        d.add(Rect(j * 3, 4, 3, 8, fillColor=shade(j / 29), strokeColor=None))
    d.add(String(96, 5, f"node cost {cost.min()} → {cost.max()} · hollow grey = not selected · ring = first node",
                 fontName="DejaVu", fontSize=7, fillColor=colors.HexColor("#555555")))
    return d


def instance_plots_section(inst, nodes, msls_res, ils_res):
    out = [PageBreak(), Paragraph(f"Best Solutions – Instance {inst}", H2)]
    for title, res in [("Multiple Start Local Search (MSLS)", msls_res),
                       ("Iterated Local Search (ILS – best run)", ils_res)]:
        if "best" not in res:
            out += [Paragraph(esc(title), H3), Paragraph(f'<font color="{BAD}"><b>No valid paths found in {esc(res["file"])}.</b></font>', BODY)]
            continue
        obj, run = res["best"]
        t = res["runs"][run]
        out.append(KeepTogether([
            Paragraph(esc(title), H3),
            Paragraph(f"Best of {res['n_valid']} valid runs: run #{run} with objective <b>{obj}</b>.", BODY),
            plot_drawing(t, nodes),
            Paragraph("Best solution (node indices, from 0)", H4),
            Paragraph(", ".join(map(str, t)), _style(f"nodes{inst}{title[:4]}", "BodyText", fontName="DejaVuMono",
                                                       fontSize=7, leading=9)),
        ]))
    return out


def _page_number(canvas, doc):
    canvas.setFont("DejaVu", 8)
    canvas.drawRightString(A4[0] - MARGIN, 1.0 * cm, str(doc.page))


def build_pdf(out, exp_data, problems):
    longest = max(len(line) for v in PSEUDOCODE.values() for line in v.splitlines())
    mono = ParagraphStyle("mono_fit", parent=MONO, fontSize=min(8.0, FRAME_W / (0.602 * longest)))
    mono.leading = mono.fontSize * 1.3

    story = [
        Paragraph("Multiple Start Local Search (MSLS) & Iterated Local Search (ILS)", H1),
        Paragraph(esc(AUTHORS), H2),
        Paragraph("Objective = cycle length (Euclidean, rounded to nearest integer) + sum of selected node "
                  "costs. Lower is better.", BODY),
        Paragraph("Pseudocode", H2),
        Paragraph(esc(PSEUDO_INTRO), BODY),
    ]
    for k, v in PSEUDOCODE.items():
        story += [KeepTogether([Paragraph(esc(k), H3), XPreformatted(esc(v), mono)])]

    story += [
        PageBreak(),
        Paragraph("Computational Experiment", H2),
        Paragraph("Each instance was evaluated across 20 runs. MSLS performs 400 steepest local searches per run "
                  "to establish the baseline quality and average running time limit. ILS uses that time budget "
                  "to test perturbation distributions using in-place random node substitution. "
                  "Best average objective per instance is highlighted in bold.", BODY),
    ]

    for inst in INSTANCES:
        d = exp_data[inst]
        story += [
            Paragraph(f"Instance {inst}", H3),
            instance_results_table(d["msls"], d["ils_summary"]),
            Spacer(1, 8),
        ]

    story.append(Paragraph("Path Validation", H3))
    if problems:
        story += [Paragraph(f'<font color="{BAD}">• {esc(p)}</font>', SMALL) for p in problems]
    else:
        story.append(Paragraph(f'<font color="{OK}"><b>All exported MSLS and ILS paths are valid (⌈n/2⌉ distinct nodes) and local optima.</b></font>', BODY))

    for inst in INSTANCES:
        d = exp_data[inst]
        story += instance_plots_section(inst, d["nodes"], d["msls"], d["ils_paths"])

    story += [
        Spacer(1, 12),
        Paragraph(f'Source code: <a href="{SOURCE_URL}" color="blue">{SOURCE_URL}</a>', BODY),
        Paragraph("Conclusions", H2),
        Paragraph(CONCLUSIONS, BODY),
    ]

    SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
        bottomMargin=MARGIN, title="MSLS and ILS Report", author=AUTHORS
    ).build(story, onFirstPage=_page_number, onLaterPages=_page_number)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE, help="folder with CSVs (default: script folder)")
    ap.add_argument("--out", type=Path, default=HERE / "report.pdf", help="output PDF path")
    args = ap.parse_args()

    exp_data = {}
    problems = []

    for inst in INSTANCES:
        inst_file = args.results / f"{inst}.csv"
        if not inst_file.exists():
            raise SystemExit(f"{inst_file} not found")
        nodes, D, c = load_instance(inst_file)
        prefix = inst.lower()

        msls_res = evaluate_paths(args.results / f"{prefix}S_MSLS.csv", nodes, D, c)
        ils_paths_res = evaluate_paths(args.results / f"{prefix}S_ILS.csv", nodes, D, c)
        ils_summary = load_ils_summary(args.results / f"{prefix}_experiment_summary.csv")

        if not ils_summary:
            problems.append(f"{inst}: missing or empty {prefix}_experiment_summary.csv")

        for label, res in [("MSLS", msls_res), ("ILS", ils_paths_res)]:
            if not res["exists"]:
                problems.append(f"{inst} / {label}: missing file {res['file']}")
                continue
            for e in res["errors"][:3]:
                problems.append(f"{inst} / {label}: {e}")
            if res.get("n_valid") and res["local_opt"] != res["n_valid"]:
                problems.append(f"{inst} / {label}: {res['n_valid'] - res['local_opt']} runs are not local optima")

        exp_data[inst] = {
            "nodes": nodes,
            "msls": msls_res,
            "ils_paths": ils_paths_res,
            "ils_summary": ils_summary,
        }

    build_pdf(args.out, exp_data, problems)
    print(f"Wrote {args.out}")
    if problems:
        print("\n".join(problems))
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()