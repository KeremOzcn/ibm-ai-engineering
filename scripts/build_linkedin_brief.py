"""Render a five-page portfolio PDF from verified public Kaggle evaluation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 720, 900
NAVY = colors.HexColor("#12283F")
INK = colors.HexColor("#17334A")
TEAL = colors.HexColor("#138477")
MINT = colors.HexColor("#96E2CC")
PAPER = colors.HexColor("#F2F6F5")
MUTED = colors.HexColor("#5C707F")
ORANGE = colors.HexColor("#D36D42")


def para(c, text, x, top, width=612, size=18, color=INK, bold=False):
    """Lay out a wrapped paragraph and return its lower edge."""
    style = ParagraphStyle(
        "body",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=size * 1.34,
        textColor=color,
    )
    p = Paragraph(text, style)
    _, height = p.wrap(width, HEIGHT)
    p.drawOn(c, x, top - height)
    return top - height


def frame(c, page, category, title):
    c.setFillColor(PAPER)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(54, 831, 34, 5, fill=1, stroke=0)
    para(c, category.upper(), 54, 810, size=13, color=TEAL, bold=True)
    para(c, title, 54, 769, size=39, bold=True)
    c.setStrokeColor(colors.HexColor("#D3DDDC"))
    c.line(54, 62, 666, 62)
    para(c, "KEREM OZCAN  /  IBM AI ENGINEERING PORTFOLIO", 54, 47, size=10, color=MUTED)
    para(c, f"{page:02d} / 05", 603, 47, width=63, size=10, color=MUTED)


def card(c, x, top, width, height, fill=colors.white):
    c.setFillColor(fill)
    c.roundRect(x, top - height, width, height, 12, fill=1, stroke=0)


def load(slug):
    return json.loads((ROOT / f"reports/kaggle/{slug}/metrics.json").read_text())


def build():
    """Use only collected successful public Kaggle execution evidence."""
    release = json.loads((ROOT / "reports/kaggle/release.json").read_text())
    if (
        len(release) != 3
        or {row["project"] for row in release}
        != {
            "satellite",
            "instruction_tuning",
            "aircraft",
        }
        or not all(row["status"] == "COMPLETE" and not row["is_private"] for row in release)
    ):
        raise ValueError("The public Kaggle executions must pass before the PDF is finalized.")
    for row in release:
        metrics = ROOT / f"reports/kaggle/{row['project']}/metrics.json"
        if hashlib.sha256(metrics.read_bytes()).hexdigest() != row["files"]["metrics.json"]:
            raise ValueError("The Kaggle results changed after verification.")
    sat, lora, aircraft = [load(s) for s in ["satellite", "instruction_tuning", "aircraft"]]
    output = ROOT / "output/pdf/ibm-ai-engineering-portfolio.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(output), pagesize=(WIDTH, HEIGHT))
    c.setTitle("IBM AI Engineering | Three reproducible experiments")
    c.setAuthor("Kerem Ozcan")

    c.setFillColor(NAVY)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    para(c, "IBM AI ENGINEERING  /  PORTFOLIO STUDY", 54, 825, size=13, color=MINT, bold=True)
    para(
        c,
        "From course labs<br/>to measured experiments.",
        54,
        758,
        size=48,
        color=colors.white,
        bold=True,
    )
    para(
        c,
        "Three case studies in vision, language-model adaptation and caption reliability.",
        54,
        571,
        size=23,
        color=colors.HexColor("#CEDCDf"),
    )
    cards = [
        (
            "01",
            "SATELLITE VISION",
            "CNN vs spatial<br/>CNN-Transformer",
            "A source-scene holdout,<br/>fresh training and<br/>checkpoint verification.",
        ),
        (
            "02",
            "LANGUAGE MODELS",
            "Base vs LoRA",
            "Actual adapter updates,<br/>response-only labels and<br/>a fresh base reload.",
        ),
        (
            "03",
            "CAPTION RELIABILITY",
            "Crack or dent?",
            "VGG16 classification<br/>beside separate<br/>BLIP descriptions.",
        ),
    ]
    for x, (number, eyebrow, title, body) in zip([54, 262, 470], cards, strict=False):
        card(c, x, 450, 196, 255, colors.HexColor("#203E56"))
        para(c, number, x + 18, 430, width=160, size=29, color=MINT, bold=True)
        para(c, eyebrow, x + 18, 383, width=160, size=10, color=MINT, bold=True)
        para(c, title, x + 18, 355, width=160, size=19, color=colors.white, bold=True)
        para(c, body, x + 18, 288, width=160, size=13, color=colors.HexColor("#D6E0E6"))
    para(c, "Kerem Ozcan", 54, 142, size=23, color=colors.white, bold=True)
    para(
        c,
        "Measured on Kaggle on 25 November 2026. Prepared with AI assistance.<br/>Source, data and model attribution accompany each notebook.",
        54,
        106,
        size=12,
        color=colors.HexColor("#B7CAD6"),
    )
    c.showPage()

    frame(c, 2, "01 / Satellite classification", "A different scene.<br/>A harder test.")
    para(
        c,
        "A random image split can hide source similarity. This experiment reserves complete filename-defined source scenes.",
        54,
        631,
        size=19,
    )
    card(c, 54, 548, 612, 91)
    for x, value, label in [
        (74, "3,383", "TRAIN"),
        (280, "93", "VALIDATION"),
        (486, "2,524", "TEST"),
    ]:
        para(c, value, x, 531, width=160, size=31, bold=True)
        para(c, label, x, 486, width=160, size=11, color=MUTED)
    para(c, "Held-out test macro F1", 54, 420, size=17, bold=True)
    for y, name, key, fill in [
        (363, "CNN", "cnn", TEAL),
        (291, "CNN-Transformer", "spatial_hybrid", NAVY),
    ]:
        score = sat["models"][key]["test"]["macro_f1"]
        para(c, name, 54, y + 22, width=170, size=15)
        c.setFillColor(colors.HexColor("#DDE7E4"))
        c.roundRect(240, y, 350, 24, 4, fill=1, stroke=0)
        c.setFillColor(fill)
        c.roundRect(240, y, 350 * score, 24, 4, fill=1, stroke=0)
        para(c, f"{score:.3f}", 607, y + 23, width=60, size=18, bold=True)
    para(
        c,
        "One setting, not a general architecture ranking.",
        54,
        233,
        size=23,
        color=TEAL,
        bold=True,
    )
    para(
        c,
        "Three epochs, one seed. Shared stem initialization and image order; different parameter counts. Sixteen image patches plus CLS preserve spatial attention. Geographic overlap and near-duplicates remain unverified.",
        54,
        184,
        size=15,
    )
    c.showPage()

    frame(c, 3, "02 / Instruction tuning", "Train it.<br/>Reload it. Compare it.")
    ratio = lora["trainable_parameters"] / lora["total_parameters"]
    para(
        c,
        f"OPT-350M with LoRA on Q/V projections. {ratio:.2%} of parameters are trainable.",
        54,
        631,
        size=20,
    )
    card(c, 54, 550, 612, 95)
    para(c, f"{lora['optimizer_steps']} optimizer steps", 75, 531, size=29, bold=True)
    para(c, "512 train  |  64 validation  |  64 test  |  1 epoch", 75, 486, size=16, color=MUTED)
    para(c, "Held-out response NLL  /  lower is better", 54, 414, size=17, bold=True)
    base, adapter = lora["base_response_loss"]["nll"], lora["adapter_response_loss"]["nll"]
    for x, label, value in [(54, "BASE", base), (370, "RELOADED ADAPTER", adapter)]:
        card(c, x, 377, 296, 116)
        para(c, label, x + 20, 355, width=250, size=12, color=MUTED, bold=True)
        para(c, f"{value:.3f}", x + 20, 323, width=250, size=38, color=TEAL, bold=True)
    para(c, "Likelihood is one measurement.", 54, 223, size=23, color=TEAL, bold=True)
    para(
        c,
        "Normalized instruction groups stay separate. Prompt and padding tokens are masked. The selected adapter reloads onto a fresh base at the same immutable revision. These NLL values do not establish generated-code correctness or robust model improvement.",
        54,
        176,
        size=15,
    )
    c.showPage()

    frame(c, 4, "03 / Aircraft damage", "Classification is not<br/>caption validation.")
    para(
        c,
        f"Frozen VGG16 + a new classifier head: <b>{aircraft['test']['accuracy']:.0%} accuracy</b> and <b>{aircraft['test']['macro_f1']:.3f} macro F1</b> across all 50 test images.",
        54,
        630,
        size=20,
    )
    source = (
        ROOT
        / ".lab-cache/02-deep-learning-with-keras/aircraft-damage-classification-and-captioning/aircraft_damage_dataset_v1"
    )
    for x, row in zip([54, 370], aircraft["captions"], strict=False):
        card(c, x, 548, 296, 302)
        c.drawImage(
            str(source / row["file"]),
            x + 15,
            368,
            width=266,
            height=160,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )
        para(
            c,
            f"TRUE: {row['label'].upper()}  |  CLASSIFIER: {row['classifier_prediction'].upper()}",
            x + 15,
            351,
            width=266,
            size=10,
            color=TEAL,
            bold=True,
        )
        para(c, "BLIP: " + escape(row["caption"]), x + 15, 324, width=266, size=14)
    para(
        c,
        "Read the picture and the generated text together.",
        54,
        218,
        size=22,
        color=TEAL,
        bold=True,
    )
    para(
        c,
        "BLIP is a separate pretrained captioner. These two examples are qualitative inspection, not a caption benchmark. Correct VGG preprocessing and complete batches fix execution issues; human quality-review fields remain empty.",
        54,
        171,
        size=14,
    )
    para(
        c,
        "Images: Youssef Donia / Roboflow, CC BY 4.0; IBM course folder adaptation.",
        54,
        91,
        size=9,
        color=MUTED,
    )
    c.linkURL(
        "https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk",
        (54, 76, 290, 94),
        relative=0,
        thickness=0,
    )
    c.linkURL(
        "https://creativecommons.org/licenses/by/4.0/", (290, 76, 370, 94), relative=0, thickness=0
    )
    c.showPage()

    frame(c, 5, "Engineering lessons", "Make the evidence<br/>part of the project.")
    lessons = [
        (
            "01",
            "Split before comparing",
            "Separate source groups, record dataset hashes and keep the held-out boundary visible.",
        ),
        (
            "02",
            "Test the saved artifact",
            "A successful training loop is incomplete evidence. Reload the selected checkpoint and verify predictions.",
        ),
        (
            "03",
            "Keep failures in the story",
            "Document where a caption is misleading and where a larger architecture does not win.",
        ),
    ]
    y = 635
    for number, title, body in lessons:
        para(c, number, 54, y, width=55, size=25, color=TEAL, bold=True)
        para(c, title, 120, y, width=546, size=22, bold=True)
        para(c, body, 120, y - 38, width=546, size=16)
        y -= 115
    para(c, "Explore the notebooks", 54, 274, size=21, bold=True)
    for y, slug, label in [
        (231, "satellite", "Satellite CNN vs Transformer"),
        (193, "instruction_tuning", "LoRA instruction tuning and verified reload"),
        (155, "aircraft", "Aircraft damage and caption reliability"),
    ]:
        meta = json.loads((ROOT / f"kaggle/{slug}/kernel-metadata.json").read_text())
        url = "https://www.kaggle.com/code/" + meta["id"]
        para(c, label + "  >", 54, y, size=16, color=TEAL)
        c.linkURL(url, (54, y - 25, 666, y + 2), relative=0, thickness=0)
    para(
        c,
        "Mevzuat RAG remains a separate capstone. Data/model ownership stays with its providers.",
        54,
        100,
        size=11,
        color=MUTED,
    )
    c.save()
    print(output)


if __name__ == "__main__":
    build()
