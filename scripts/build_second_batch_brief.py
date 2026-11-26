"""Create a five-page supplement from completed public second-batch Kaggle evidence."""

from __future__ import annotations

import hashlib
import json

from build_linkedin_brief import (
    HEIGHT,
    MINT,
    MUTED,
    NAVY,
    ROOT,
    TEAL,
    WIDTH,
    card,
    frame,
    para,
)
from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas


def build():
    """Finalize presentation only after all four published experiments actually complete."""
    base = ROOT / "reports/kaggle-second-batch"
    ledger = json.loads((base / "release.json").read_text())
    expected = {"translation", "fraud", "cartpole", "preference_alignment"}
    if (
        len(ledger) != 4
        or {r["project"] for r in ledger} != expected
        or not all(r["status"] == "COMPLETE" and not r["is_private"] for r in ledger)
    ):
        raise ValueError("Four completed public experiments are required.")
    reports = {}
    for row in ledger:
        path = base / row["project"] / "metrics.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["files"]["metrics.json"]:
            raise ValueError("Metrics changed after verification.")
        reports[row["project"]] = json.loads(path.read_text())
    out = ROOT / "output/pdf/ibm-ai-engineering-additional-projects.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(out), pagesize=(WIDTH, HEIGHT))
    c.setAuthor("Kerem Ozcan")
    c.setTitle("IBM AI Engineering | Additional measured studies")
    c.setFillColor(NAVY)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    para(c, "IBM AI ENGINEERING  /  ADDITIONAL STUDIES", 54, 825, size=13, color=MINT, bold=True)
    para(
        c,
        "Four more studies.<br/>Fresh training. Real tests.",
        54,
        758,
        size=45,
        color=colors.white,
        bold=True,
    )
    para(
        c,
        "Translation, fraud detection, reinforcement learning<br/>and preference alignment.",
        54,
        587,
        size=22,
        color=colors.HexColor("#CEDCDF"),
    )
    links = {r["project"]: r["url"] for r in ledger}
    rows = [
        ("translation", "01 / GRU vs Transformer", "Whole-test-set translation evaluation"),
        ("fraud", "02 / Fraud review budget", "Actual true and false alerts in 100 reviews"),
        ("cartpole", "03 / CartPole target networks", "A controlled comparison across three seeds"),
        (
            "preference_alignment",
            "04 / SFT vs DPO",
            "Frozen reference and independently reloaded adapters",
        ),
    ]
    for top, (slug, title, description) in zip([485, 385, 285, 185], rows, strict=False):
        card(c, 54, top, 612, 88, colors.HexColor("#203E56"))
        para(c, title, 73, top - 14, size=19, color=MINT, bold=True)
        para(c, description + "  >", 73, top - 48, size=14, color=colors.white)
        c.linkURL(links[slug], (54, top - 88, 666, top), relative=0, thickness=0)
    para(
        c,
        "Kerem Ozcan  |  Measured on Kaggle, 26 November 2026",
        54,
        66,
        size=13,
        color=colors.white,
        bold=True,
    )
    para(
        c,
        "Prepared with AI assistance. Provider/model attribution accompanies every notebook.",
        54,
        40,
        size=10,
        color=MINT,
    )
    c.showPage()

    t = reports["translation"]
    frame(c, 2, "01 / Translation", "Fresh models.<br/>The same held-out text.")
    para(
        c,
        "German to English on Multi30k. Both models use the same training pairs, initial embeddings, learning rate and six-epoch budget.",
        54,
        629,
        size=19,
    )
    card(c, 54, 544, 612, 94)
    for x, role in [(74, "train"), (280, "validation"), (486, "test")]:
        para(c, f"{t['selected_split_counts'][role]:,}", x, 527, width=160, size=30, bold=True)
        para(c, role.upper(), x, 484, width=160, size=11, color=MUTED)
    para(c, "Case-insensitive corpus BLEU  /  higher is better", 54, 415, size=17, bold=True)
    for y, key, label in [(357, "gru", "GRU"), (287, "transformer", "Transformer")]:
        score = t["models"][key]["corpus_bleu"]
        para(c, label, 54, y + 23, width=170, size=16)
        c.setFillColor(colors.HexColor("#DDE7E4"))
        c.roundRect(240, y, 350, 24, 4, fill=1, stroke=0)
        c.setFillColor(TEAL if key == "transformer" else NAVY)
        c.roundRect(240, y, 350 * score / 30, 24, 4, fill=1, stroke=0)
        para(c, f"{score:.2f}", 607, y + 23, width=60, size=18, bold=True)
    para(
        c,
        "Bar scale: 0-30 BLEU. Full scoring signature saved in the notebook.",
        240,
        261,
        width=426,
        size=10,
        color=MUTED,
    )
    para(c, "A comparison within this training setting.", 54, 219, size=23, color=TEAL, bold=True)
    para(
        c,
        "Validation selects checkpoints; fresh reloads verify predictions. Different model capacities/computation, one seed and short captions limit the conclusion. Corpus BLEU does not replace human translation review.",
        54,
        170,
        size=15,
    )
    c.linkURL(links["translation"], (54, 76, 666, 92), relative=0, thickness=0)
    para(
        c,
        "Source: Multi30k research dataset. Open the notebook for provenance and outputs.",
        54,
        88,
        size=9,
        color=MUTED,
    )
    c.showPage()

    f = reports["fraud"]
    lr = f["models"]["logistic_regression"]
    top = lr["top_budget_test"]
    fixed = lr["fixed_threshold_test"]
    frame(c, 3, "02 / Fraud detection", "A useful alert<br/>has a review cost.")
    para(
        c,
        f"Chronological evaluation on {f['split_counts']['test']['samples']:,} later transactions, including {f['split_counts']['test']['fraud']} fraud cases. Preprocessing is fitted on training data only.",
        54,
        629,
        size=19,
    )
    for x, value, label in [
        (54, f"{lr['average_precision']:.3f}", "LOGISTIC AVERAGE PRECISION"),
        (370, f"{top['true_alerts']} /100", "FRAUD FOUND IN 100 REVIEWS"),
    ]:
        card(c, x, 539, 296, 117)
        para(c, label, x + 18, 518, width=260, size=11, color=MUTED, bold=True)
        para(c, value, x + 18, 481, width=260, size=36, color=TEAL, bold=True)
    para(c, "The fixed top 100 ranking", 54, 381, size=22, bold=True)
    para(
        c,
        f"{top['true_alerts']} true alerts  |  {top['false_alerts']} false alerts  |  {top['missed_fraud']} missed fraud cases",
        54,
        337,
        size=19,
    )
    card(c, 54, 281, 612, 91)
    para(
        c,
        f"The frozen validation cutoff produced {fixed['alerts']} test alerts.",
        74,
        262,
        width=572,
        size=18,
        color=TEAL,
        bold=True,
    )
    para(
        c,
        "Future workload can drift. A fixed cutoff and a fixed ranking budget are different operating policies.",
        74,
        227,
        width=572,
        size=14,
    )
    para(c, "Keep the operating policy visible.", 54, 155, size=22, color=TEAL, bold=True)
    para(
        c,
        "Historical two-day data; customer groups and upstream PCA provenance are unverified. Cost units are illustrative, with no production savings claim.",
        54,
        116,
        size=13,
    )
    c.showPage()

    r = reports["cartpole"]
    frame(c, 4, "03 / Reinforcement learning", "One target changed.<br/>Learning changed.")
    para(
        c,
        "Online-target DQN versus a lagged target network. Match architecture, replay, exploration, reward and environment-step budgets.",
        54,
        629,
        size=19,
    )
    card(c, 54, 544, 612, 94)
    for x, value, label in [
        (74, "3", "TRAINING SEEDS"),
        (280, "20,000", "STEPS PER SEED"),
        (486, "500", "EPISODE LIMIT"),
    ]:
        para(c, value, x, 527, width=160, size=30, bold=True)
        para(c, label, x, 484, width=160, size=10, color=MUTED)
    para(c, "Greedy test return  / 20 episodes per training seed", 54, 415, size=17, bold=True)
    for y, key, label in [
        (357, "online_target", "Online target"),
        (287, "lagged_target", "Lagged target"),
    ]:
        result = r["models"][key]
        score = result["mean_of_seed_means"]
        para(c, label, 54, y + 23, width=170, size=16)
        c.setFillColor(colors.HexColor("#DDE7E4"))
        c.roundRect(240, y, 350, 24, 4, fill=1, stroke=0)
        c.setFillColor(TEAL if key == "lagged_target" else NAVY)
        c.roundRect(240, y, 350 * score / 500, 24, 4, fill=1, stroke=0)
        para(c, f"{score:.1f}", 607, y + 23, width=60, size=18, bold=True)
    para(
        c,
        "Bar scale: 0-500 return. Per-seed scores and variation are saved.",
        240,
        261,
        width=426,
        size=10,
        color=MUTED,
    )
    para(c, "Retain the failed baseline.", 54, 219, size=23, color=TEAL, bold=True)
    para(
        c,
        "Validation reset seeds select checkpoints; separate test seeds evaluate them. Truncations bootstrap; genuine terminations do not. This bounded CartPole study does not establish a solved task or general RL robustness.",
        54,
        170,
        size=15,
    )
    c.showPage()

    a = reports["preference_alignment"]
    sft = a["models"]["sft"]
    aligned = a["models"]["selected_dpo"]
    frame(c, 5, "04 / Preference alignment", "Update the policy.<br/>Keep the reference frozen.")
    para(
        c,
        "OPT-350M with a new preference-domain SFT baseline, then real DPO updates against cached frozen-reference log probabilities.",
        54,
        629,
        size=19,
    )
    card(c, 54, 544, 612, 94)
    para(
        c,
        f"{a['sft_optimizer_steps']} SFT  + {a['dpo_optimizer_steps']} DPO updates",
        74,
        526,
        size=28,
        bold=True,
    )
    para(c, "256 train  | 64 validation  | 64 test pairs", 74, 481, size=16, color=MUTED)
    para(c, "Held-out reference-adjusted DPO loss", 54, 415, size=17, bold=True)
    for x, label, value in [
        (54, "SFT REFERENCE", sft["reference_adjusted_dpo_loss"]),
        (370, "SELECTED CHECKPOINT", aligned["reference_adjusted_dpo_loss"]),
    ]:
        card(c, x, 377, 296, 115)
        para(c, label, x + 18, 356, width=260, size=11, color=MUTED, bold=True)
        para(c, f"{value:.3f}", x + 18, 320, width=260, size=38, color=TEAL, bold=True)
    selected = "DPO candidate" if a["selected_stage"] == "dpo" else "SFT baseline retained"
    para(
        c,
        f"Chosen-response NLL: {sft['chosen_response_nll']:.3f} (SFT) vs {aligned['chosen_response_nll']:.3f} (selected).",
        54,
        247,
        size=14,
        color=MUTED,
    )
    para(c, "Validation selection: " + selected, 54, 223, size=21, color=TEAL, bold=True)
    para(
        c,
        "Prompt/padding masks preserve response-only learning. SFT and selected-DPO adapters reload onto fresh identical-revision bases. Likelihood margins are not generated-response win rates or human-validated quality. Small budgets, synthetic labels and response truncation limit conclusions.",
        54,
        176,
        size=14,
    )
    para(
        c,
        "Data: Argilla UltraFeedback (MIT). Base: Meta OPT-350M research/noncommercial terms.",
        54,
        88,
        size=9,
        color=MUTED,
    )
    c.save()
    print(out)


if __name__ == "__main__":
    build()
