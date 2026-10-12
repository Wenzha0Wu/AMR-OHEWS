"""AMR-OHEWS framework figure (paper Figure 1), emitted as draw.io + rendered preview.

One layout spec, two backends:
  * figures/fig0_framework.drawio         editable in draw.io / diagrams.net
  * figures/fig0_framework.{pdf,svg,png}  matplotlib render of the same spec

Coordinates are draw.io pixels (origin top-left). The matplotlib backend uses a
figure 1 pt per pixel wide, so font sizes match between the two.

Run:  .venv/bin/python scripts/framework_figure.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amrews.config import PALETTE  # noqa: E402

W, H = 1000, 466
INK, MUTED, LINE = "#1a202c", "#4a5568", "#a0aec0"
PANEL, CORE = "#f7fafc", "#ebf4ff"
ORANGE, ORANGE_DARK, ORANGE_LIGHT, EVAL_BG = "#dd6b20", "#9c4221", "#fbd38d", "#fffaf0"
HUMAN, ANIMAL, FOOD, ENV = PALETTE.human, PALETTE.animal, PALETTE.food, PALETTE.environment
ALERT, GREY = PALETTE.alert, PALETTE.baseline
NONE = "none"

T, B, S = 12, 11, 10          # title / body / small font sizes (px)


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    text: str
    size: float = B
    color: str = MUTED
    bold: bool = False
    italic: bool = False


@dataclass
class Box:
    id: str
    x: float
    y: float
    w: float
    h: float
    fill: str = "#ffffff"
    stroke: str = LINE
    sw: float = 1.0
    r: float = 5
    dashed: bool = False
    lines: list[Line] = field(default_factory=list)
    align: str = "left"          # left | center | right
    valign: str = "top"          # top | middle
    pad: float = 6
    shape: str = "rect"          # rect | triangle


@dataclass
class Edge:
    id: str
    pts: list[tuple[float, float]]
    color: str = INK
    sw: float = 1.6
    dashed: bool = False
    arrow: bool = True
    src: str | None = None
    tgt: str | None = None


BOXES: dict[str, Box] = {}
EDGES: list[Edge] = []


def box(id, x, y, w, h, *lines, **kw) -> Box:
    b = Box(id, x, y, w, h, lines=[l if isinstance(l, Line) else Line(l) for l in lines], **kw)
    BOXES[id] = b
    return b


def label(id, x, y, w, h, *lines, **kw) -> Box:
    kw.setdefault("align", "center")
    kw.setdefault("valign", "middle")
    return box(id, x, y, w, h, *lines, fill=NONE, stroke=NONE, pad=0, **kw)


def edge(id, *pts, **kw) -> Edge:
    e = Edge(id, list(pts), **kw)
    EDGES.append(e)
    return e


def header(id, x, w, num, title, obj=None):
    box(id, x, 0, w, 24, Line(f"{num}  {title}", T, "#ffffff", bold=True),
        fill=INK, stroke=INK, r=4, valign="middle", pad=8)
    if obj:
        tag(f"{id}_tag", x + w - 28, 5, obj)


def tag(id, x, y, text):
    box(id, x, y, 23, 14, Line(text, 9, "#ffffff", bold=True), fill=ORANGE,
        stroke=ORANGE, r=7, align="center", valign="middle", pad=0)


# column geometry -------------------------------------------------------------
C1 = (0, 192)
C2 = (222, 152)
C3 = (396, 226)
C4 = (644, 180)
C5 = (854, 146)
TOP, BOT = 30, 330               # panel band
EV_Y = 352                       # evaluation band top

# 1 One Health data -----------------------------------------------------------
x, w = C1
header("h1", x, w, "1", "One Health data", "O1")
box("p1", x, TOP, w, BOT - TOP, fill=PANEL)
sectors = [("Human", "clinical isolates", HUMAN),
           ("Animal / veterinary", "veterinary isolates", ANIMAL),
           ("Food chain", "retail meat & food samples", FOOD),
           ("Environment", "wastewater & effluent", ENV)]
for i, (name, sub, col) in enumerate(sectors):
    y = 38 + i * 43
    box(f"sec{i}", x + 8, y, w - 16, 37, Line(name, 12, col, bold=True), Line(sub),
        stroke=col, sw=1.3, pad=13, valign="middle")
    box(f"sec{i}_strip", x + 8, y, 6, 37, fill=col, stroke=col, r=2)
label("srcnote", x + 4, 210, w - 8, 14, Line("both sources cover all four sectors",
                                             9.5, MUTED, italic=True))
box("sim", x + 6, 227, 87, 95, Line("Simulator", 12, INK, bold=True),
    Line("ground truth", S, ORANGE, bold=True), Line("emergence"), Line("events,"),
    Line("delays, gaps,"), Line("spatial spread"), stroke=INK, sw=1.1, pad=5)
box("real", x + 99, 227, 87, 95, Line("Real data", 12, INK, bold=True),
    Line("observed", S, ORANGE, bold=True), Line("NCBI Pathogen", 10.5), Line("Detection", 10.5),
    Line("EFSA/ECDC", 10.5), Line("WHO GLASS", 10.5), stroke=INK, sw=1.1, pad=5)

# AMR-OHEWS core (stages 2-4) -------------------------------------------------
box("core", 214, TOP, 618, BOT - TOP, fill=CORE, stroke=HUMAN, sw=1.1, r=6, dashed=True)
label("corelabel", 222, 312, 152, 14,
      Line("AMR-OHEWS core", S, HUMAN, bold=True, italic=True), align="left")

# 2 Harmonise -----------------------------------------------------------------
x, w = C2
header("h2", x, w, "2", "Harmonise data")
steps = [("Align", ["region × week ×", "resistance determinant"]),
         ("Reporting delays", ["only data available", "at prediction time"]),
         ("Features", ["sector trends, antibiotic", "use, sampling intensity"]),
         ("Region graph", ["inter-region links", "for spatial spread"])]
for i, (t, body) in enumerate(steps):
    y = 38 + i * 68
    box(f"st{i}", x, y, w, 58, Line("      " + t, 12, INK, bold=True), *map(Line, body))
    box(f"st{i}_b", x + 6, y + 5, 15, 15, Line("abcd"[i], 9.5, "#ffffff", bold=True),
        fill=INK, stroke=INK, r=3, align="center", valign="middle", pad=0)
    if i:
        edge(f"e_st{i}", (x + w / 2, y - 10), (x + w / 2, y), color=MUTED, sw=1.2,
             src=f"st{i-1}", tgt=f"st{i}")

# 3 Models --------------------------------------------------------------------
x, w = C3
header("h3", x, w, "3", "Weekly risk models", "O1")
box("base", x, 38, w, 54, Line("Clinical-only baseline", 12, GREY, bold=True),
    Line("outbreak detection on human data"), Line("(Farrington; Salmon et al.)"),
    stroke=GREY, sw=1.2)
box("xsec", x, 100, w, 146, Line("Cross-sector AI models", 12, HUMAN, bold=True),
    stroke=HUMAN, sw=1.4)
models = [("Gradient", "boosting", "tabular"), ("Temporal", "neural net", "time series"),
          ("Graph", "neural net", "spatial")]
mw = (w - 16 - 12) / 3
for i, (a, b_, sub) in enumerate(models):
    mx = x + 8 + i * (mw + 6)
    box(f"m{i}", mx, 124, mw, 56, Line(a, B, INK, bold=True), Line(b_, B, INK, bold=True),
        Line(sub, S), fill=CORE, stroke=HUMAN, sw=0.9, r=4, align="center",
        valign="middle", pad=2)
    edge(f"e_m{i}", (mx + mw / 2, 180), (x + w / 2 + (i - 1) * 40, 196), color=HUMAN,
         sw=1.0, src=f"m{i}", tgt="ens")
box("ens", x + 22, 196, w - 44, 40, Line("Ensemble + calibration", 12, "#ffffff", bold=True),
    Line("weighted combination", S, "#e2e8f0"), fill=HUMAN, stroke=HUMAN, r=5,
    align="center", valign="middle", pad=2)
box("out", x, 256, w, 50, Line("weekly, per region × determinant", S, "#e2e8f0"),
    Line("P(established in humans", 11.5, "#ffffff", bold=True),
    Line("within forecast horizon)", 11.5, "#ffffff", bold=True),
    fill=INK, stroke=INK, r=5, align="center", valign="middle", pad=2)
edge("e_out", (x + w / 2, 236), (x + w / 2, 256), color=INK, sw=1.4, src="ens", tgt="out")

# 4 Alerts --------------------------------------------------------------------
x, w = C4
header("h4", x, w, "4", "Interpretable alerts")
box("thr", x, 38, w, 66, Line("Alert threshold", 12, ALERT, bold=True),
    Line("fixed false-alarm budget"), stroke=ALERT, sw=1.2)
series = [(x + 10 + i * 8, y) for i, y in enumerate(
    [97, 95, 98, 96, 97, 94, 96, 95, 93, 90, 86, 83, 84, 81, 82, 80, 81, 79, 80, 79, 80])]
edge("thr_series", *series, color=HUMAN, sw=1.3, arrow=False)
edge("thr_line", (x + 8, 89), (x + w - 8, 89), color=ALERT, sw=1.0, dashed=True, arrow=False)
box("thr_mark", x + 88, 79, 9, 7, fill=ALERT, stroke=ALERT, shape="triangle")

box("card", x, 112, w, 194, stroke=ALERT, sw=1.5, r=5)
box("card_h", x, 112, w, 32, Line("ALERT", 11, "#ffffff", bold=True),
    Line("region R · determinant D · week t", 9.5, "#fed7d7"), fill=ALERT, stroke=ALERT,
    r=5, valign="middle", pad=7)
label("risk_l", x + 7, 148, 90, 16, Line("Estimated risk"), align="left")
label("risk_v", x + w - 77, 148, 70, 16, Line("0.78", 13, INK, bold=True), align="right")
bx, bw = x + 7, w - 14
box("bar_bg", bx, 167, bw, 7, fill="#edf2f7", stroke=NONE, r=3)
box("bar", bx, 167, bw * 0.78, 7, fill=ALERT, stroke=NONE, r=3)
edge("ci", (bx + bw * 0.64, 179), (bx + bw * 0.88, 179), color=INK, sw=1.2, arrow=False)
for k, f in enumerate((0.64, 0.88)):
    edge(f"ci{k}", (bx + bw * f, 176), (bx + bw * f, 182), color=INK, sw=1.2, arrow=False)
label("ci_l", x + 7, 184, w - 14, 14, Line("uncertainty 0.64–0.88", S), align="left")
label("shap_l", x + 7, 203, w - 14, 14, Line("Main drivers (SHAP)", B, INK, bold=True),
      align="left")
tag("shap_t", x + w - 30, 203, "O3")
drivers = [("animal ↑", 0.9, ANIMAL), ("wastewater ↑", 0.65, ENV),
           ("neighbour ↑", 0.42, GREY)]
for i, (lab, v, col) in enumerate(drivers):
    y = 221 + i * 16
    label(f"drv{i}", x + 7, y, 82, 13, Line(lab, S, INK), align="left")
    box(f"drvb{i}", x + 90, y + 2, (w - 98) * v, 9, fill=col, stroke=NONE, r=2)
label("dens", x + 7, 280, w - 14, 24, Line("sampling intensity & testing", S, MUTED, italic=True),
      Line("denominators shown", S, MUTED, italic=True), align="left")

# 5 Decision support ----------------------------------------------------------
x, w = C5
header("h5", x, w, "5", "Decision support")
box("p5", x, TOP, w, BOT - TOP, fill=PANEL)
box("dash", x + 8, 38, w - 16, 60, Line("Dashboard", 12, INK, bold=True),
    Line("Streamlit · Plotly"), stroke=INK, sw=1.1, align="center", pad=5)
for i, c in enumerate([PALETTE.grid, ALERT, PALETTE.grid, "#feb2b2", PALETTE.grid]):
    box(f"map{i}", x + 26 + i * 19, 80, 15, 10, fill=c, stroke=NONE, r=2)
label("users", x + 8, 106, w - 16, 76, Line("Users", 12, INK, bold=True),
      Line("• PH researchers"), Line("• clinicians"), Line("• PH agencies"),
      Line("• policymakers"), align="left", valign="top")
label("acts", x + 8, 186, w - 16, 76, Line("Human-led actions", 12, INK, bold=True),
      Line("• targeted sampling"), Line("• stewardship review"),
      Line("• supply-chain checks"), Line("• effluent controls"), align="left", valign="top")
box("note", x + 8, 272, w - 16, 50, Line("screening &", S, ALERT),
    Line("prioritisation tool,", S, ALERT), Line("not automatic action", S, ALERT),
    stroke=ALERT, sw=1.0, align="center", valign="middle", pad=2)

# 6 Evaluation ----------------------------------------------------------------
box("ev", 0, EV_Y, W, H - EV_Y - 2, fill=EVAL_BG, stroke=ORANGE, sw=1.3, r=5)
box("ev_h", 0, EV_Y, W, 22, Line("6  Evaluation framework", T, "#ffffff", bold=True),
    fill=ORANGE, stroke=ORANGE, r=5, valign="middle", pad=8)
label("ev_leg", 330, EV_Y + 4, 662, 14,
      Line("O1 integrated system   ·   O2 earlier than clinical-only?   ·   "
           "O3 sector contribution & interpretability", S, "#ffffff", bold=True),
      align="right")
cards = [
    (C1, "Two tracks", ["O2"], ["Simulation: lead time vs", "known emergence events",
                                "Real data: test assumptions,", "replicate core comparison"]),
    (C2, "Leakage-free", [], ["chronological train/test", "as-of-time features",
                              "same pipeline, all models"]),
    (C3, "Fair comparison", ["O2", "O3"],
     ["clinical-only vs cross-sector", "same false-alarm budget",
      "clinical-only variants · sector ablation"]),
    (C4, "Metrics", ["O2"], ["sensitivity", "realised false-alarm rate",
                             "lead time (weeks)"]),
    (C5, "Outcome", [], ["earlier warning?", "data gaps for", "deployment"]),
]
for i, ((cx, cw), title, objs, body) in enumerate(cards):
    if i == 0:
        cx, cw = cx + 6, cw - 6
    if i == 4:
        cw -= 6
    box(f"card{i}", cx, EV_Y + 28, cw, H - EV_Y - 38, Line(title, 12, ORANGE_DARK, bold=True),
        *map(Line, body), stroke=ORANGE_LIGHT, pad=5)
    for j, o in enumerate(objs):
        tag(f"card{i}_t{j}", cx + cw - 28 - (len(objs) - 1 - j) * 26, EV_Y + 33, o)

# main flow arrows ------------------------------------------------------------
edge("f1", (192, 67), (222, 67), sw=2.2, src="p1", tgt="st0")
edge("f2", (374, 65), (396, 65), color=GREY, sw=1.3, dashed=True, src="st0", tgt="base")
edge("f3", (374, 206), (396, 206), sw=2.2, src="st2", tgt="xsec")
edge("f4", (622, 281), (644, 281), sw=2.2, src="out", tgt="card")
edge("f5", (824, 209), (854, 209), sw=2.2, src="card", tgt="p5")
for eid, src, xx in [("v1", "sim", 50), ("v2", "real", 141)]:
    edge(eid, (xx, 322), (xx, EV_Y + 28), color=ORANGE, sw=1.5, src=src, tgt="card0")
edge("v3", (509, 306), (509, EV_Y + 28), color=ORANGE, sw=1.5, src="out", tgt="card2")
edge("v4", (734, 306), (734, EV_Y + 28), color=ORANGE, sw=1.5, src="card", tgt="card3")


# --------------------------------------------------------------------------- #
# draw.io backend
# --------------------------------------------------------------------------- #
def html_label(b: Box) -> str:
    rows = []
    for l in b.lines:
        st = f"font-size:{l.size}px;color:{l.color};line-height:1.2;"
        st += "font-weight:bold;" if l.bold else ""
        st += "font-style:italic;" if l.italic else ""
        rows.append(f'<div style="{st}">{escape(l.text).replace("  ", "&nbsp; ")}</div>')
    return "".join(rows)


def drawio_style(b: Box) -> str:
    if b.shape == "triangle":
        s = ["triangle", "direction=south"]
    elif b.fill == NONE and b.stroke == NONE:
        s = ["text"]
    else:
        s = ["rounded=1", "absoluteArcSize=1", f"arcSize={2 * b.r}"]
    s += ["whiteSpace=wrap", "html=1", f"fillColor={b.fill}", f"strokeColor={b.stroke}",
          f"strokeWidth={b.sw}", "fontFamily=Helvetica", f"fontSize={B}",
          f"align={b.align}", f"verticalAlign={b.valign}", "spacing=0",
          f"spacingLeft={b.pad}", f"spacingRight={b.pad}", f"spacingTop={b.pad}",
          f"spacingBottom={b.pad}"]
    if b.dashed:
        s += ["dashed=1", "dashPattern=6 3"]
    return ";".join(s) + ";"


def frac(b: Box, p):
    return round((p[0] - b.x) / b.w, 4), round((p[1] - b.y) / b.h, 4)


def to_drawio() -> str:
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for b in BOXES.values():
        cells.append(
            f'<mxCell id="{b.id}" value="{escape(html_label(b))}" style="{drawio_style(b)}" '
            f'vertex="1" parent="1"><mxGeometry x="{b.x:g}" y="{b.y:g}" width="{b.w:g}" '
            f'height="{b.h:g}" as="geometry"/></mxCell>')
    for e in EDGES:
        st = ["html=1", "edgeStyle=none", "rounded=0", f"strokeColor={e.color}",
              f"strokeWidth={e.sw}",
              "endArrow=block;endFill=1;endSize=5" if e.arrow else "endArrow=none"]
        if e.dashed:
            st.append("dashed=1;dashPattern=4 3")
        attrs, geo = "", ""
        if e.src and e.tgt:
            ex, ey = frac(BOXES[e.src], e.pts[0])
            nx, ny = frac(BOXES[e.tgt], e.pts[-1])
            st += [f"exitX={ex}", f"exitY={ey}", "exitDx=0", "exitDy=0", "exitPerimeter=0",
                   f"entryX={nx}", f"entryY={ny}", "entryDx=0", "entryDy=0",
                   "entryPerimeter=0"]
            attrs = f' source="{e.src}" target="{e.tgt}"'
        else:
            (sx, sy), (tx, ty) = e.pts[0], e.pts[-1]
            geo = (f'<mxPoint x="{sx:g}" y="{sy:g}" as="sourcePoint"/>'
                   f'<mxPoint x="{tx:g}" y="{ty:g}" as="targetPoint"/>')
        mids = e.pts[1:-1]
        if mids:
            geo += ('<Array as="points">' +
                    "".join(f'<mxPoint x="{px:g}" y="{py:g}"/>' for px, py in mids) +
                    "</Array>")
        cells.append(f'<mxCell id="{e.id}" style="{";".join(st)};" edge="1" parent="1"{attrs}>'
                     f'<mxGeometry relative="1" as="geometry">{geo}</mxGeometry></mxCell>')
    body = "\n        ".join(cells)
    return f"""<mxfile host="app.diagrams.net">
  <diagram id="amr-ohews-framework" name="AMR-OHEWS framework">
    <mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="2" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W + 20}" pageHeight="{H + 20}" background="#ffffff" math="0" shadow="0">
      <root>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


# --------------------------------------------------------------------------- #
# matplotlib backend
# --------------------------------------------------------------------------- #
def render(out: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    texts = []

    for z, b in enumerate(BOXES.values()):
        fc = "none" if b.fill == NONE else b.fill
        ec = "none" if b.stroke == NONE else b.stroke
        if b.shape == "triangle":
            ax.add_patch(Polygon([(b.x, b.y), (b.x + b.w, b.y), (b.x + b.w / 2, b.y + b.h)],
                                 fc=fc, ec=ec, lw=b.sw * 0.75, zorder=z))
        elif fc != "none" or ec != "none":
            ax.add_patch(FancyBboxPatch(
                (b.x, b.y), b.w, b.h, boxstyle=f"round,pad=0,rounding_size={min(b.r, b.h / 2)}",
                fc=fc, ec=ec, lw=b.sw * 0.75, ls=(0, (6, 3)) if b.dashed else "-", zorder=z))
        if not b.lines:
            continue
        total = sum(l.size * 1.2 for l in b.lines)
        y = b.y + b.pad if b.valign == "top" else b.y + (b.h - total) / 2
        tx = {"left": b.x + b.pad, "center": b.x + b.w / 2, "right": b.x + b.w - b.pad}[b.align]
        for l in b.lines:
            t = ax.text(tx, y + l.size * 0.6, l.text, fontsize=l.size, color=l.color,
                        fontweight="bold" if l.bold else "normal",
                        fontstyle="italic" if l.italic else "normal",
                        ha=b.align, va="center", zorder=500)
            y += l.size * 1.2
            texts.append((b, t))
        if y > b.y + b.h + 1:
            print(f"  [v-overflow] {b.id}: text ends {y - b.y - b.h:.1f}px below box")

    for e in EDGES:
        ls = (0, (4, 3)) if e.dashed else "-"
        if e.arrow:
            ax.add_patch(FancyArrowPatch(e.pts[0], e.pts[-1], arrowstyle="-|>", color=e.color,
                                         lw=e.sw * 0.75, ls=ls, mutation_scale=4 + 3 * e.sw,
                                         shrinkA=0, shrinkB=0, zorder=400))
        else:
            xs, ys = zip(*e.pts)
            ax.plot(xs, ys, color=e.color, lw=e.sw * 0.75, ls=ls, zorder=400,
                    solid_joinstyle="round")

    # Flag any line wider than its box (DejaVu is wider than draw.io's Helvetica,
    # so a clean preview means the .drawio file fits too).
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for b, t in texts:
        bb = t.get_window_extent()
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        lo, hi = (b.x, b.x + b.w) if b.stroke == NONE else (b.x + 2, b.x + b.w - 2)
        if x0 < lo - 0.5 or x1 > hi + 0.5:
            print(f"  [h-overflow] {b.id}: '{t.get_text()}' spans {x0:.0f}-{x1:.0f}, "
                  f"box {lo:.0f}-{hi:.0f}")

    for ext in ("pdf", "svg", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    out = ROOT / "figures" / "fig0_framework"
    out.with_suffix(".drawio").write_text(to_drawio(), encoding="utf-8")
    render(out)
    print(f"wrote {out}.drawio and .pdf/.svg/.png")
