from __future__ import annotations

import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reconcile.evidence import build_document_evidence, build_open_bank_after_evidence  # noqa: E402
from reconcile.matching import MatchSettings, reconcile  # noqa: E402
from reconcile.parsers import parse_accountable_file, parse_bank_file, parse_paypal_file  # noqa: E402
from reconcile.paypal import match_docs_to_paypal, match_paypal_transfers_to_bank  # noqa: E402


OUTPUT_DIR = ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "finanz_reconciliation_belegketten_fyrst_paypal.pdf"


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)

    settings = MatchSettings()
    accountable = parse_accountable_file("Daten.xlsx", (ROOT / "Daten.xlsx").read_bytes())
    bank_parsed = parse_bank_file("fyrst.csv", (ROOT / "fyrst.csv").read_bytes())
    paypal_parsed = parse_paypal_file("paypal.CSV", (ROOT / "paypal.CSV").read_bytes())

    docs = accountable.frame
    bank = bank_parsed.frame
    paypal = paypal_parsed.frame

    matches, links = reconcile(docs, bank, settings)
    paypal_doc_matches, paypal_doc_links = match_docs_to_paypal(docs, paypal, settings)
    paypal_bank_matches = match_paypal_transfers_to_bank(paypal, bank, settings)

    evidence = build_document_evidence(docs, bank, paypal, matches, links, paypal_doc_matches, paypal_doc_links, paypal_bank_matches)
    open_doc_ids = set(evidence.loc[evidence["evidence_level"] == "offen", "doc_id"])
    open_docs = docs[docs["doc_id"].isin(open_doc_ids)].copy()
    open_bank = build_open_bank_after_evidence(bank, matches, paypal_bank_matches)

    write_detail_exports(docs, bank, paypal, matches, links, paypal_doc_matches, paypal_doc_links, paypal_bank_matches, evidence, open_docs, open_bank)
    build_pdf(docs, bank, paypal, matches, links, paypal_doc_matches, paypal_doc_links, paypal_bank_matches, evidence, open_docs, open_bank)
    print(REPORT_PATH)
    return 0


def build_pdf(
    docs: pd.DataFrame,
    bank: pd.DataFrame,
    paypal: pd.DataFrame,
    matches: pd.DataFrame,
    links: pd.DataFrame,
    paypal_doc_matches: pd.DataFrame,
    paypal_doc_links: pd.DataFrame,
    paypal_bank_matches: pd.DataFrame,
    evidence: pd.DataFrame,
    open_docs: pd.DataFrame,
    open_bank: pd.DataFrame,
) -> None:
    pdf = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=landscape(A4),
        rightMargin=1.0 * cm,
        leftMargin=1.0 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
        title="Finanz-Reconciliation Review",
    )

    styles = get_styles()
    story: list = []

    story.append(Paragraph("Finanz-Reconciliation Belegketten-Review", styles["Title"]))
    story.append(Paragraph("Scope: Daten.xlsx + fyrst.csv + paypal.CSV", styles["Subtitle"]))
    story.append(Paragraph(f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}", styles["Small"]))
    story.append(Spacer(1, 0.35 * cm))

    story.append(
        Paragraph(
            "Hinweis: Dieser Bericht ist eine technische Reconciliation-Übersicht und keine Steuerberatung. "
            "Die App und dieser Bericht nutzen dieselbe lokale Parser- und Matching-Engine; dieser PDF-Lauf wurde per Skript erzeugt, nicht durch einen Browser-Klick. "
            "Die Hauptsicht ist belegzentriert: Accountable-Beleg → PayPal/Plattformdetail → FYRST-Bankumsatz.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    summary_rows = [
        ["Kennzahl", "Wert"],
        ["Accountable-Belege gesamt", len(docs)],
        ["FYRST-Bankumsätze gesamt", len(bank)],
        ["PayPal-Zeilen gesamt", len(paypal)],
        ["Belege über FYRST-Match verknüpft", links["doc_id"].nunique() if not links.empty else 0],
        ["Belege über PayPal-Detail verknüpft", paypal_doc_links["doc_id"].nunique() if not paypal_doc_links.empty else 0],
        ["Vollständig belegte Belege", len(evidence[evidence["evidence_level"] == "vollständig belegt"])],
        ["Teilweise belegte Belege", len(evidence[evidence["evidence_level"] == "teilweise belegt"])],
        ["Offene Accountable-Belege", len(open_docs)],
        ["Offene FYRST-Umsätze", len(open_bank)],
    ]
    story.append(Paragraph("1. Executive Summary", styles["H1"]))
    story.append(make_table(summary_rows, [9 * cm, 4 * cm], header=True))
    story.append(Spacer(1, 0.35 * cm))

    story.append(
        Paragraph(
            "Interpretation: PayPal wird als Zwischenkonto behandelt, nicht als zusätzliche Bank. "
            "Ein Beleg ist vollständig belegt, wenn er entweder direkt zu FYRST passt oder über PayPal bis zu FYRST zurückverfolgt werden kann. "
            "Die verbleibenden offenen Fälle sind kein pauschales Etsy-Problem: Bei den offenen Belegen sind viele Printful-/Fulfillment-Ausgaben "
            "und Belege ohne klares Plattformsignal dabei. Bei den offenen FYRST-Umsätzen liegen dagegen auffällig viele Etsy-/eBay-Sammelzahlungen, "
            "für die Plattform-Detaildaten die Beweiskette deutlich verbessern würden.",
            styles["Body"],
        )
    )

    story.append(Paragraph("2. Offene Fälle nach Plattform", styles["H1"]))
    story.append(Paragraph("Offene Accountable-Belege", styles["H2"]))
    story.append(counter_table(open_docs.get("platform", pd.Series(dtype=str))))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Offene FYRST-Umsätze", styles["H2"]))
    story.append(counter_table(open_bank.get("platform", pd.Series(dtype=str))))

    story.append(PageBreak())
    story.append(Paragraph("3. Match-Übersicht", styles["H1"]))
    story.append(Paragraph("Belegketten nach Status", styles["H2"]))
    story.append(counter_table(evidence["evidence_level"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("FYRST zu Accountable", styles["H2"]))
    story.append(
        make_table(
            [
                ["Methode", "Anzahl", "Ø Konfidenz"],
                *[
                    [method, count, f"{matches[matches['method'] == method]['confidence'].mean():.0%}"]
                    for method, count in matches["method"].value_counts().items()
                ],
            ],
            [8 * cm, 3 * cm, 4 * cm],
            header=True,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("PayPal-Brücke", styles["H2"]))
    story.append(
        make_table(
            [
                ["Bereich", "Anzahl"],
                ["PayPal-Detailmatches zu Accountable", len(paypal_doc_matches)],
                ["PayPal-Banktransfer-Brücken", len(paypal_bank_matches)],
            ],
            [8 * cm, 3 * cm],
            header=True,
        )
    )

    story.append(PageBreak())
    add_detail_section(
        story,
        styles,
        "Anhang A: Alle Accountable-Belegketten",
        evidence,
        ["doc_ref", "doc_type", "date", "amount", "platform", "counterparty", "evidence_level", "evidence_path", "fyrst_amounts", "paypal_amounts", "bridge_fyrst_amounts", "evidence_note"],
        ["Beleg", "Typ", "Datum", "Betrag", "Plattform", "Gegenpartei", "Status", "Nachweiskette", "FYRST", "PayPal", "PayPal→FYRST", "Hinweis"],
        [2.4 * cm, 1.5 * cm, 1.7 * cm, 1.8 * cm, 1.7 * cm, 3.2 * cm, 2.5 * cm, 4.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 5.0 * cm],
        max_rows=None,
    )

    add_detail_section(
        story,
        styles,
        "Anhang B: Offene Accountable-Belege",
        evidence[evidence["evidence_level"] == "offen"],
        ["doc_ref", "doc_type", "date", "amount", "platform", "counterparty", "description", "evidence_note"],
        ["Beleg", "Typ", "Datum", "Betrag", "Plattform", "Gegenpartei", "Beschreibung", "Warum offen"],
        [2.6 * cm, 1.5 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 4.0 * cm, 9.0 * cm, 5.0 * cm],
        max_rows=None,
    )

    add_detail_section(
        story,
        styles,
        "Anhang C: Offene FYRST-Umsätze",
        open_bank,
        ["date", "amount", "platform", "counterparty", "description"],
        ["Datum", "Betrag", "Plattform", "Gegenpartei", "Beschreibung"],
        [2.0 * cm, 2.1 * cm, 2.0 * cm, 5.0 * cm, 13.0 * cm],
        max_rows=None,
    )

    add_detail_section(
        story,
        styles,
        "Anhang D: FYRST-Accountable-Matches (technische Sicht)",
        matches,
        ["bank_date", "bank_amount", "bank_counterparty", "method", "confidence", "doc_count", "doc_sum", "gap", "explanation"],
        ["Datum", "Bank", "Gegenpartei", "Methode", "Konf.", "Belege", "Belegsumme", "Diff.", "Hinweis"],
        [2.0 * cm, 2.1 * cm, 4.4 * cm, 2.8 * cm, 1.5 * cm, 1.2 * cm, 2.0 * cm, 1.5 * cm, 8.0 * cm],
        max_rows=None,
    )

    add_detail_section(
        story,
        styles,
        "Anhang E: PayPal-Detailmatches zu Accountable (technische Sicht)",
        paypal_doc_matches,
        ["bank_date", "bank_amount", "bank_counterparty", "confidence", "doc_count", "doc_sum", "gap", "explanation"],
        ["Datum", "PayPal", "Gegenpartei", "Konf.", "Belege", "Belegsumme", "Diff.", "Hinweis"],
        [2.0 * cm, 2.1 * cm, 4.4 * cm, 1.5 * cm, 1.2 * cm, 2.0 * cm, 1.5 * cm, 10.0 * cm],
        max_rows=None,
    )

    add_detail_section(
        story,
        styles,
        "Anhang F: PayPal-zu-FYRST-Brücken (technische Sicht)",
        paypal_bank_matches,
        ["paypal_date", "paypal_amount", "paypal_description", "bank_date", "bank_amount", "bank_counterparty", "confidence", "day_diff"],
        ["PayPal-Datum", "PayPal", "PayPal-Text", "Bankdatum", "Bank", "Bank-Gegenpartei", "Konf.", "Tage"],
        [2.0 * cm, 2.1 * cm, 4.3 * cm, 2.0 * cm, 2.1 * cm, 5.0 * cm, 1.5 * cm, 1.2 * cm],
        max_rows=None,
    )

    pdf.build(story, onFirstPage=footer, onLaterPages=footer)


def add_detail_section(
    story: list,
    styles: dict,
    title: str,
    frame: pd.DataFrame,
    cols: list[str],
    headers: list[str],
    widths: list[float],
    max_rows: int | None,
) -> None:
    story.append(PageBreak())
    story.append(Paragraph(title, styles["H1"]))
    if frame.empty:
        story.append(Paragraph("Keine Zeilen.", styles["Body"]))
        return
    subset = frame.copy()
    if max_rows is not None:
        subset = subset.head(max_rows)
    rows = [headers]
    for _, row in subset.iterrows():
        rows.append([format_cell(row.get(col, ""), styles["Tiny"]) for col in cols])
    story.append(make_table(rows, widths, header=True))


def write_detail_exports(
    docs: pd.DataFrame,
    bank: pd.DataFrame,
    paypal: pd.DataFrame,
    matches: pd.DataFrame,
    links: pd.DataFrame,
    paypal_doc_matches: pd.DataFrame,
    paypal_doc_links: pd.DataFrame,
    paypal_bank_matches: pd.DataFrame,
    evidence: pd.DataFrame,
    open_docs: pd.DataFrame,
    open_bank: pd.DataFrame,
) -> None:
    exports = {
        "report_all_docs.csv": docs,
        "report_all_fyrst_transactions.csv": bank,
        "report_all_paypal_transactions.csv": paypal,
        "report_fyrst_matches.csv": matches,
        "report_fyrst_match_links.csv": links,
        "report_paypal_doc_matches.csv": paypal_doc_matches,
        "report_paypal_doc_links.csv": paypal_doc_links,
        "report_paypal_bank_bridge.csv": paypal_bank_matches,
        "report_document_evidence_chains.csv": evidence,
        "report_open_docs.csv": open_docs,
        "report_open_fyrst_transactions.csv": open_bank,
    }
    for name, frame in exports.items():
        frame.to_csv(OUTPUT_DIR / name, sep=";", index=False, encoding="utf-8-sig")


def counter_table(series: pd.Series) -> Table:
    values = ["(ohne Plattform)" if not str(value) or str(value) == "nan" else str(value) for value in series.fillna("")]
    counts = Counter(values)
    rows = [["Plattform", "Anzahl"], *[[key, value] for key, value in counts.most_common()]]
    return make_table(rows, [7 * cm, 3 * cm], header=True)


def make_table(rows: list[list[object]], widths: list[float], header: bool = False) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.2),
        ("LEADING", (0, 0), (-1, -1), 7.2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(style))
    return table


def format_cell(value: object, style: ParagraphStyle) -> Paragraph:
    if pd.isna(value):
        text = ""
    elif isinstance(value, pd.Timestamp):
        text = value.strftime("%Y-%m-%d")
    elif isinstance(value, float):
        if math.isfinite(value):
            text = f"{value:.2f}" if abs(value) >= 1 else f"{value:.3f}"
        else:
            text = ""
    else:
        text = str(value)
    return Paragraph(escape_pdf_text(text), style)


def escape_pdf_text(text: str, limit: int = 260) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def get_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#111827")),
        "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11, leading=14, textColor=colors.HexColor("#4B5563")),
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#111827")),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=13, spaceBefore=5, spaceAfter=4, textColor=colors.HexColor("#1F2937")),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=12, textColor=colors.HexColor("#1F2937"), alignment=TA_LEFT),
        "Small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=10, textColor=colors.HexColor("#6B7280")),
        "Tiny": ParagraphStyle("Tiny", parent=base["BodyText"], fontName="Helvetica", fontSize=5.8, leading=6.8, textColor=colors.HexColor("#111827")),
    }


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(1.0 * cm, 0.45 * cm, "Finanz-Reconciliation Review - technische Übersicht, keine Steuerberatung")
    canvas.drawRightString(landscape(A4)[0] - 1.0 * cm, 0.45 * cm, f"Seite {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    raise SystemExit(main())
