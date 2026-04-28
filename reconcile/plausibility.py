from __future__ import annotations

from pathlib import Path

import pandas as pd


PLAUSIBILITY_COLUMNS = ["bereich", "wert", "betrag", "hinweis"]


def build_overall_plausibility_report(
    docs: pd.DataFrame,
    bank: pd.DataFrame,
    paypal: pd.DataFrame,
    evidence: pd.DataFrame,
    bank_claim_usage: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    doc_income = sum_where(docs, docs["doc_type"].eq("income"), "signed_amount") if not docs.empty else 0.0
    doc_expense = sum_where(docs, docs["doc_type"].eq("expense"), "signed_amount") if not docs.empty else 0.0
    doc_net = round(doc_income + doc_expense, 2)

    bank_income = sum_where(bank, bank["amount"].astype(float) > 0, "amount") if not bank.empty else 0.0
    bank_expense = sum_where(bank, bank["amount"].astype(float) < 0, "amount") if not bank.empty else 0.0
    bank_net = round(bank_income + bank_expense, 2)

    paypal_commercial_net = 0.0
    paypal_transfer_net = 0.0
    if not paypal.empty:
        paypal_commercial_net = sum_where(paypal, paypal["category"].eq("commercial") & paypal["currency"].eq("EUR"), "amount")
        paypal_transfer_net = sum_where(paypal, paypal["category"].eq("bank_transfer") & paypal["currency"].eq("EUR"), "amount")

    complete_sum = sum_where(evidence, evidence["evidence_level"].astype(str).str.contains("vollständig|vollstaendig|vollst", na=False, regex=True), "amount") if not evidence.empty else 0.0
    partial_sum = sum_where(evidence, evidence["evidence_level"].eq("teilweise belegt"), "amount") if not evidence.empty else 0.0
    open_sum = sum_where(evidence, evidence["evidence_level"].eq("offen"), "amount") if not evidence.empty else 0.0
    gross_platform_count = int((bank_claim_usage["usage_status"] == "gross_platform_over_bank").sum()) if not bank_claim_usage.empty else 0
    gross_platform_delta = sum_where(bank_claim_usage, bank_claim_usage["usage_status"].eq("gross_platform_over_bank"), "delta_bank_minus_claim") if not bank_claim_usage.empty else 0.0
    hard_over_claim_count = int((bank_claim_usage["usage_status"] == "over_claim").sum()) if not bank_claim_usage.empty else 0
    hard_over_claim_delta = sum_where(bank_claim_usage, bank_claim_usage["usage_status"].eq("over_claim"), "delta_bank_minus_claim") if not bank_claim_usage.empty else 0.0

    rows.extend(
        [
            row("Accountable", "Einnahmen laut Belegen", doc_income, "Summe der Accountable-Ausgangsrechnungen."),
            row("Accountable", "Ausgaben laut Belegen", doc_expense, "Negative Summe der Accountable-Eingangsrechnungen/Ausgaben."),
            row("Accountable", "Netto Belege", doc_net, "Einnahmen minus Ausgaben laut Accountable."),
            row("FYRST", "Bankeingänge", bank_income, "Alle positiven FYRST-Zeilen im Upload, inklusive Umbuchungen und ggf. privaten/technischen Zahlungen."),
            row("FYRST", "Bankausgänge", bank_expense, "Alle negativen FYRST-Zeilen im Upload."),
            row("FYRST", "Netto Bank", bank_net, "Bankeingänge minus Bankausgänge im hochgeladenen FYRST-Zeitraum."),
            row("Kontrollgleichung", "Accountable-Netto minus FYRST-Netto", round(doc_net - bank_net, 2), "Das muss nur nahe 0 sein, wenn exakt dieselben Konten, derselbe Zeitraum, reine Geschäftsvorgänge und keine offenen PayPal-/Plattform-/Privatkonto-Salden vorliegen."),
            row("PayPal", "EUR commercial netto", paypal_commercial_net, "Nur PayPal-EUR-Geschäftsvorgänge; Fremdwährungen laufen zusätzlich über Umrechnungszeilen."),
            row("PayPal", "EUR Banktransfer netto", paypal_transfer_net, "PayPal-Transfers zum/vom Bankkonto. Das ist kein Umsatz, sondern Brücke."),
            row("Belegstatus", "vollständig belegte Belegsumme", complete_sum, "Summe der Belege mit vollständigem Zahlungsnachweis."),
            row("Belegstatus", "teilweise belegte Belegsumme", partial_sum, "Summe der Belege mit Detailspur, aber nicht vollständig geschlossen."),
            row("Belegstatus", "offene Belegsumme", open_sum, "Summe der noch offenen Belege."),
            row("Umsatz-Auslastung", "Brutto-Plattform über Bank Anzahl", float(gross_platform_count), "Meist Etsy/eBay: Brutto-Ausgangsrechnungen liegen über Netto-Auszahlung, weil Gebühren/Reserven gegengerechnet werden müssen."),
            row("Umsatz-Auslastung", "Brutto-Plattform Differenzsumme", gross_platform_delta, "Negative Werte zeigen, wie viel Gebühr/Reserve/Timing noch gegen die Brutto-Belege laufen muss."),
            row("Umsatz-Auslastung", "harte Überbeanspruchung Anzahl", float(hard_over_claim_count), "Bankzeilen außerhalb der typischen Plattform-Brutto/Netto-Logik, die rechnerisch zu stark beansprucht werden."),
            row("Umsatz-Auslastung", "harte Überbeanspruchung Differenzsumme", hard_over_claim_delta, "Negative Werte zeigen echte rechnerische Überbeanspruchung in der Kontrollrechnung."),
        ]
    )
    return pd.DataFrame(rows, columns=PLAUSIBILITY_COLUMNS)


def build_overall_plausibility_pdf(report: pd.DataFrame, path: Path) -> None:
    load_reportlab()
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        rightMargin=0.8 * cm,
        leftMargin=0.8 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
        title="Gesamt-Plausibilitaet",
    )
    styles = getSampleStyleSheet()
    story: list = []
    story.append(Paragraph("Gesamt-Plausibilitaetsbericht", styles["Title"]))
    story.append(
        Paragraph(
            "Dieser Bericht vergleicht die Belegwelt mit der Zahlungswelt. Er ist eine Plausibilitaetskontrolle, kein Steuerabschluss.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(report_table(report))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "Wichtig: Accountable-Netto minus FYRST-Netto muss nicht exakt 0 sein, wenn PayPal als Zwischenkonto genutzt wird, "
            "Fremdwaehrungen umgerechnet werden, Privat-/Nebenkonten fehlen, Steuern/Privatentnahmen enthalten sind oder Zahlungen und Belege in unterschiedliche Perioden fallen.",
            styles["BodyText"],
        )
    )
    pdf.build(story)


def row(bereich: str, wert: str, betrag: float, hinweis: str) -> dict[str, object]:
    return {"bereich": bereich, "wert": wert, "betrag": round(float(betrag), 2), "hinweis": hinweis}


def sum_where(frame: pd.DataFrame, mask, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return round(float(frame.loc[mask, column].astype(float).sum()), 2)


def report_table(frame: pd.DataFrame) -> Table:
    load_reportlab()
    data = [["Bereich", "Wert", "Betrag", "Hinweis"]]
    for _, row_data in frame.iterrows():
        data.append([row_data["bereich"], row_data["wert"], money(row_data["betrag"]), row_data["hinweis"]])
    table = Table(data, colWidths=[4 * cm, 6 * cm, 3 * cm, 15 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (2, 1), (2, -1), "RIGHT"),
            ]
        )
    )
    return table


def money(value: object) -> str:
    try:
        return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return ""


def load_reportlab() -> None:
    global colors, A4, landscape, getSampleStyleSheet, cm, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
