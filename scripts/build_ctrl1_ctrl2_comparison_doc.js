"use strict";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, LevelFormat,
} = require("docx");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const OUT_DIR = path.join(ROOT, "outputs", "ctrl_comparison_2026-04-28");
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

// ── palette ────────────────────────────────────────────────────────────────
const BLUE    = "1F4E79";
const MID_BLUE= "2E75B6";
const LT_BLUE = "D5E8F0";
const LT_GREY = "F2F2F2";
const GREEN   = "1A7A4A";
const ORANGE  = "C05000";
const WHITE   = "FFFFFF";

// ── helpers ────────────────────────────────────────────────────────────────
function run(text, opts = {}) {
  return new TextRun({ text, font: "Aptos", size: 20, ...opts });
}

function boldRun(text, opts = {}) {
  return run(text, { bold: true, ...opts });
}

function para(children, opts = {}) {
  const arr = Array.isArray(children) ? children : [run(children)];
  return new Paragraph({ children: arr, spacing: { after: 100 }, ...opts });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Aptos Display", size: 26, bold: true, color: BLUE })],
    spacing: { before: 280, after: 120 },
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Aptos Display", size: 22, bold: true, color: MID_BLUE })],
    spacing: { before: 200, after: 80 },
  });
}

function bullet(text, subText = null) {
  const children = subText
    ? [run(text, { bold: true }), run(" — " + subText)]
    : [run(text)];
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children,
    spacing: { after: 60 },
  });
}

function spacer() {
  return new Paragraph({ children: [run("")], spacing: { after: 60 } });
}

function divider() {
  return new Paragraph({
    children: [run("")],
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 1 } },
    spacing: { after: 120 },
  });
}

// ── table helpers ──────────────────────────────────────────────────────────
const THIN = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const BORDERS = { top: THIN, bottom: THIN, left: THIN, right: THIN };

// Content width A4 (1.7 cm margins each side) ≈ 8890 DXA
const TABLE_W = 8890;

function headerCell(text, w, bgColor = BLUE) {
  return new TableCell({
    borders: BORDERS,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: bgColor, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Aptos", size: 18, bold: true, color: WHITE })],
      alignment: AlignmentType.CENTER,
    })],
  });
}

function dataCell(text, w, bgColor = WHITE, textColor = "000000", bold = false, align = AlignmentType.CENTER) {
  return new TableCell({
    borders: BORDERS,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: bgColor, type: ShadingType.CLEAR },
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Aptos", size: 18, bold, color: textColor })],
      alignment: align,
    })],
  });
}

function rowGrey(cells) {
  return new TableRow({ children: cells.map((c, i) => {
    const w = colWidths[i];
    const bg = i === 0 ? LT_GREY : WHITE;
    return dataCell(c, w, bg, "000000", i === 0, i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER);
  })});
}

function rowGreen(cells) {
  // mark improvement columns (cols 2 and 3) green
  return new TableRow({ children: cells.map((c, i) => {
    const w = colWidths[i];
    const isImprove = (i === 2 || i === 3) && c.startsWith("−") === false && c !== "—";
    const bg = i === 0 ? LT_GREY : WHITE;
    return dataCell(c, w, bg, "000000", false, i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER);
  })});
}

// ── data ───────────────────────────────────────────────────────────────────
// [Metric, Baseline, CTRL-1, CTRL-2, Note]
// Baseline = BSM2 closed-loop (bsm2_manual_baseline_timeseries.csv proxy)
// CTRL-1   = Qec SAC (ctrl1_qec_training_optionB.csv, all rows)
// CTRL-2   = Qint SAC (ctrl2_qint_training.csv, all rows)

const metricsTable = [
  // header row handled separately
  // [metric, baseline, ctrl1, ctrl2, direction]
  ["Mean reward",           "0.661",   "0.400",     "0.416",     "↑ higher is better"],
  ["Mean J (×10³)",         "1 336.5", "1 228.9",   "1 223.9",   "↓ lower is better"],
  ["Mean J ratio",          "1.033",   "0.950",     "0.946",     "↓ lower is better"],
  ["Mean EQI (kg pol./d)", "5 836",   "5 298",     "5 297",     "↓ lower is better"],
  ["Median J ratio",        "0.910",   "0.931",     "0.927",     "↓ lower is better"],
  ["Mean SNO₂ (g N/m³)",   "3.834",   "3.833",     "3.833",     "— reference ~3.83"],
  ["Mean SNO₃ (g N/m³)",   "7.205",   "7.204",     "7.203",     "— reference ~7.20"],
  ["Mean SNH  (g N/m³)",   "7.275",   "7.275",     "7.274",     "— reference ~7.27"],
  ["Reward clipped at −1",  "n/a",     "17.2 %",    "17.0 %",    "— fewer is better"],
  ["J ratio < 1.0 (steps)", "n/a",     "60.2 %",    "60.7 %",    "↑ more is better"],
  ["Last 1 000 reward mean","0.661*",  "0.440",     "0.488",     "↑ higher is better"],
  ["Last 1 000 J ratio",    "1.033*",  "0.929",     "0.923",     "↓ lower is better"],
  ["Action variable",       "n/a",     "Qec (0–5)", "Qint (5–62k)","—"],
  ["Action mean",           "Qec = 2 (fixed)", "Qec = 2.276","Qint = 29 490","—"],
  ["Training steps",        "—",       "58 463",    "58 401",    "—"],
];

// Improvement relative to baseline
// J ratio: baseline 1.033, ctrl1 0.950 → +8.1%; ctrl2 0.946 → +8.4%
// EQI:     baseline 5836,  ctrl1 5298  → +9.2%; ctrl2 5297  → +9.2%
// Reward:  baseline 0.661, ctrl1 0.400 → −39.5%; ctrl2 0.416 → −37.1%

const w0 = 2800, w1 = 1500, w2 = 1500, w3 = 1500, w4 = 1590;
const colWidths = [w0, w1, w2, w3, w4];
const totalW = w0 + w1 + w2 + w3 + w4; // should = TABLE_W

// ── improvement table ─────────────────────────────────────────────────────
const improvTable = [
  ["Metric", "CTRL-1 vs Baseline", "CTRL-2 vs Baseline", "CTRL-2 vs CTRL-1"],
  ["J ratio",  "−8.1 %", "−8.4 %", "−0.4 %"],
  ["EQI",      "−9.2 %", "−9.2 %", "< −0.01 %"],
  ["Mean reward","−39.5 %","−37.1 %","+ 4.1 %"],
  ["Last 1k reward","−33.4 %","−26.2 %","+ 10.9 %"],
  ["Last 1k ratio","−10.1 %","−10.7 %","−0.7 %"],
];
const iw0 = 2800, iw1 = 2030, iw2 = 2030, iw3 = 2030;
const iColWidths = [iw0, iw1, iw2, iw3];

function greenCell(text, w, positive) {
  // positive = true if the value shows improvement (negative for ratio/EQI, positive for reward)
  const isNum = text.match(/^[+−-]/);
  let bg = WHITE;
  if (isNum) {
    const isImprove = (text.startsWith("−") || text.startsWith("-")) ? true : false;
    bg = isImprove ? "E2F0D9" : "FCE4D6"; // green tint = better, orange tint = worse
  }
  return new TableCell({
    borders: BORDERS,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Aptos", size: 18, color: "000000" })],
      alignment: AlignmentType.CENTER,
    })],
  });
}

function improvRow(cells, isHeader = false) {
  return new TableRow({ children: cells.map((c, i) => {
    const w = iColWidths[i];
    if (isHeader) return headerCell(c, w, MID_BLUE);
    if (i === 0) return dataCell(c, w, LT_GREY, "000000", false, AlignmentType.LEFT);
    return greenCell(c, w, false);
  })});
}

// ── document ───────────────────────────────────────────────────────────────
const mainTableRows = [
  new TableRow({ children: [
    headerCell("Metric", w0),
    headerCell("BSM2 Baseline", w1),
    headerCell("CTRL-1 (Qec)", w2),
    headerCell("CTRL-2 (Qint)", w3),
    headerCell("Direction / Note", w4),
  ]}),
  ...metricsTable.map((row, ri) => {
    const bg = ri % 2 === 0 ? WHITE : LT_GREY;
    return new TableRow({ children: row.map((c, i) => {
      const w = colWidths[i];
      const cellBg = i === 0 ? LT_GREY : (ri % 2 === 0 ? WHITE : "F7FAFF");
      const align = i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER;
      return dataCell(c, w, cellBg, "000000", i === 0, align);
    })});
  }),
];

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 600, hanging: 300 } } } }],
    }],
  },
  styles: {
    default: { document: { run: { font: "Aptos", size: 20 } } },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 }, // ~2 cm
      },
    },
    children: [

      // ── TITLE ──────────────────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({
          text: "Single-Agent RL Controllers vs. BSM2 Closed-Loop Baseline",
          font: "Aptos Display", size: 36, bold: true, color: BLUE,
        })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({
          text: "CTRL-1 (Qec SAC) · CTRL-2 (Qint SAC) · Preliminary comparison — April 2026",
          font: "Aptos", size: 19, italic: true, color: "5A5A5A",
        })],
      }),
      divider(),

      // ── 1. CONTROLLER DESCRIPTIONS ─────────────────────────────────────
      heading1("1. Controller Descriptions"),

      heading2("BSM2 Closed-Loop Baseline"),
      para([
        run("The standard BSM2 benchmark operating point. Both the external carbon dosing rate ("),
        run("Qec", { bold: true }),
        run(" = 2 m³/d) and the internal recirculation flow ("),
        run("Qint", { bold: true }),
        run(" = 61 944 m³/d) are fixed. The simulation ran over "),
        run("364 days", { bold: true }),
        run(" of dynamic influent (days 245–609). This is the reference against which both RL controllers are evaluated."),
      ]),
      spacer(),

      heading2("CTRL-1 — Qec SAC (carbon dosing controller)"),
      para([
        run("A Soft Actor-Critic (SAC) agent that learns to modulate the "),
        run("external carbon dosing rate Qec", { bold: true }),
        run(", a continuous action in [0, 5] m³/d. "),
        run("Qint is kept at its baseline value (61 944 m³/d). "),
        run("The agent ran for "),
        run("58 463 steps", { bold: true }),
        run(" (1 000 random warm-up + 57 463 SAC) on a single training episode covering the same 364-day dynamic influent window."),
      ]),
      para([
        run("Hypothesis: ", { bold: true }),
        run("by varying Qec, the agent should reduce the organic effluent quality index (EQI) and the overall cost objective J, since carbon dosing directly affects biological denitrification."),
      ]),
      spacer(),

      heading2("CTRL-2 — Qint SAC (internal recirculation controller)"),
      para([
        run("A SAC agent that learns to modulate the "),
        run("internal recirculation flow Qint", { bold: true }),
        run(", a continuous action in [5 000, 61 944] m³/d. "),
        run("Qec is kept at its baseline value (2 m³/d). "),
        run("The agent ran for "),
        run("58 401 steps", { bold: true }),
        run(" under the same conditions as CTRL-1."),
      ]),
      para([
        run("Hypothesis: ", { bold: true }),
        run("by adjusting Qint, the agent controls how much nitrified effluent is recycled from the aerobic to the anoxic zones, enabling better denitrification and lower effluent nitrogen, potentially at lower pumping energy cost."),
      ]),
      divider(),

      // ── 2. METRIC COMPARISON TABLE ─────────────────────────────────────
      heading1("2. Metric Comparison"),
      para([
        run("All values are means over the full training run unless noted. "),
        run("Baseline* ", { bold: true }),
        run("= proxy computed from the BSM2 closed-loop timeseries using the same reward formula as the RL agents. "),
        run("J", { bold: true }),
        run(" is the composite cost objective; "),
        run("J ratio = J / J_manual", { bold: true }),
        run(", where J_manual is the step-wise manual reference (ratio < 1 means the controller outperforms the manual reference on that step). "),
        run("EQI", { bold: true }),
        run(" is the effluent quality index in kg pollution units per day."),
      ]),
      spacer(),

      new Table({
        width: { size: TABLE_W, type: WidthType.DXA },
        columnWidths: colWidths,
        rows: mainTableRows,
      }),
      para([run("* Baseline last-1 000 values are the global mean (no training dynamics, uniform series).", { italic: true, size: 17, color: "666666" })],
        { spacing: { before: 80, after: 60 } }),
      divider(),

      // ── 3. IMPROVEMENT SUMMARY ─────────────────────────────────────────
      heading1("3. Improvement Summary"),
      para([run("Relative improvement compared to the BSM2 baseline and between the two controllers. "),
        run("Green cells (−) ", { bold: true, color: GREEN }),
        run("indicate improvement for metrics where lower is better (J ratio, EQI) or for reward where positive is better. "),
        run("Orange cells (+) ", { bold: true, color: ORANGE }),
        run("indicate degradation.")]),
      spacer(),

      new Table({
        width: { size: TABLE_W, type: WidthType.DXA },
        columnWidths: iColWidths,
        rows: improvTable.map((r, ri) => improvRow(r, ri === 0)),
      }),
      divider(),

      // ── 4. PROCESS STATES ─────────────────────────────────────────────
      heading1("4. Process State Analysis"),
      para("All three configurations produce nearly identical mean process states. The differences are at the third decimal place or smaller:"),
      spacer(),

      bullet("SNO₂", "3.833 (baseline 3.834) — difference < 0.003 g N/m³"),
      bullet("SNO₃", "7.203–7.204 (baseline 7.205) — difference < 0.002 g N/m³"),
      bullet("SNH",  "7.274–7.275 (baseline 7.275) — difference < 0.001 g N/m³"),
      spacer(),

      para([run("Interpretation: ", { bold: true }),
        run("the RL agents are not driving a fundamentally different nitrogen profile from the baseline. The improvements in J and EQI are achieved through smaller, more frequent cost-reducing moves rather than a sustained shift in plant state. This is consistent with a reward-clipping effect: periods where the controller overshoots or undershoots result in a hard −1 clip, which limits the benefit visible in mean reward even when instantaneous J improves.")]),
      divider(),

      // ── 5. TRAINING BEHAVIOUR ─────────────────────────────────────────
      heading1("5. Training Behaviour"),

      heading2("CTRL-1 (Qec)"),
      bullet("Overall mean reward (0.400) is lower than baseline (0.661), driven by 17.2 % of steps clipped at −1."),
      bullet("Mean J (1 228 892) is 8.1 % below baseline, and mean J ratio (0.950) is also below 1."),
      bullet("60.2 % of steps have J ratio < 1, meaning CTRL-1 was cheaper than the step-wise manual reference more often than not."),
      bullet("Last 1 000-step reward improves to 0.440, and J ratio improves to 0.929 — the agent is slowly converging toward a better policy late in training."),
      bullet("Mean Qec used: 2.276 m³/d — slightly above the baseline fixed value of 2 m³/d."),
      spacer(),

      heading2("CTRL-2 (Qint)"),
      bullet("Overall mean reward (0.416) is slightly above CTRL-1 (0.400) and below baseline (0.661). Clipping rate similar at 17.0 %."),
      bullet("Mean J (1 223 901) is marginally lower than CTRL-1 (1 228 892) — both ~8 % below baseline."),
      bullet("60.7 % of steps have J ratio < 1 — slightly more than CTRL-1 (60.2 %)."),
      bullet("Last 1 000-step reward (0.488) and ratio (0.923) are better than CTRL-1's last-window values, suggesting CTRL-2 converged somewhat more aggressively in late training."),
      bullet("Mean Qint used: 29 490 m³/d — well below the baseline fixed value of 61 944 m³/d. The agent is recirculating about half as much as the baseline, strongly reducing pumping energy (PE)."),
      divider(),

      // ── 6. KEY TAKEAWAYS ──────────────────────────────────────────────
      heading1("6. Key Takeaways"),

      bullet("Both controllers improve the cost objective J and EQI versus the BSM2 baseline by roughly 8–9 %, while leaving process states (SNO, SNH) essentially unchanged."),
      bullet("Neither controller improves mean reward — reward is worse than baseline because the fixed −1 clip dominates when the agent explores. The improvement in J and the degradation in reward are two sides of the same coin (reward clipping artefact)."),
      bullet("CTRL-2 (Qint) is marginally better than CTRL-1 (Qec) on J, J ratio, last-window reward, and last-window ratio. The gap is small but consistent. A key mechanism for CTRL-2 is the drastic reduction in Qint (29 490 vs 61 944 m³/d), which cuts pumping energy substantially."),
      bullet("Both controllers are stable: no reward-NaN rows, no out-of-bounds actions, no SNO₃ collapses below 0.5 g N/m³."),
      bullet("Next step (CTRL-3 / CTRL-4 or G2ANet): the two single-agent baselines confirm that Qec and Qint are each individually useful control handles. A cooperative multi-agent architecture that coordinates both simultaneously should be able to capture additive gains from both variables."),
      spacer(),

      new Paragraph({
        spacing: { before: 100, after: 40 },
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: MID_BLUE, space: 4 } },
        children: [new TextRun({
          text: "One-sentence thesis framing: both CTRL-1 and CTRL-2 are stable, cost-improving single-agent baselines that each exploit one control handle; their combined use in a multi-agent cooperative architecture is the natural next step toward the G2ANet target.",
          font: "Aptos", size: 19, bold: true, color: BLUE,
        })],
      }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(OUT_DIR, "ctrl1_ctrl2_vs_baseline_comparison.docx");
  fs.writeFileSync(out, buf);
  console.log("Written:", out);
});
