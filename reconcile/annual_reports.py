from __future__ import annotations

from pathlib import Path

import pandas as pd


ETSY_ANNUAL_COLUMNS = [
    "level",
    "shop",
    "period",
    "sales",
    "fees",
    "taxes",
    "adjustments",
    "payouts",
    "calculated_balance_change",
    "status",
    "note",
]

ETSY_ACCOUNTABLE_COLUMNS = [
    "level",
    "period",
    "statement_sales",
    "accountable_income",
    "sales_diff",
    "statement_costs",
    "accountable_expenses",
    "costs_diff",
    "statement_net_before_payout",
    "accountable_net",
    "net_diff",
    "statement_payouts",
    "statement_balance_change",
    "status",
    "note",
]


def build_etsy_annual_reconciliation(platform_transactions: pd.DataFrame) -> pd.DataFrame:
    if platform_transactions.empty:
        return pd.DataFrame(columns=ETSY_ANNUAL_COLUMNS)
    work = platform_transactions[
        (platform_transactions["platform"].astype(str).str.lower() == "etsy")
        & (platform_transactions["source"].astype(str) == "Etsy Statement")
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=ETSY_ANNUAL_COLUMNS)

    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work[work["date"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=ETSY_ANNUAL_COLUMNS)

    work["shop"] = work["shop"].fillna("").replace("", "default")
    work["period"] = work["date"].dt.to_period("M").astype(str)

    rows: list[dict[str, object]] = []
    for (shop, period), group in work.groupby(["shop", "period"], dropna=False):
        rows.append(etsy_reconciliation_row("month", shop, period, group))

    for shop, group in work.groupby("shop", dropna=False):
        year = str(int(group["date"].dt.year.min())) if group["date"].dt.year.nunique() == 1 else "all"
        rows.append(etsy_reconciliation_row("year", shop, year, group))

    total_year = str(int(work["date"].dt.year.min())) if work["date"].dt.year.nunique() == 1 else "all"
    rows.append(etsy_reconciliation_row("all_shops_year", "alle", total_year, work))
    return pd.DataFrame(rows, columns=ETSY_ANNUAL_COLUMNS)


def etsy_reconciliation_row(level: str, shop: str, period: str, group: pd.DataFrame) -> dict[str, object]:
    sales = amount_sum(group, ["ledger_order"])
    fees = amount_sum(group, ["ledger_fee"])
    taxes = amount_sum(group, ["ledger_tax"])
    adjustments = amount_sum(group, ["ledger_adjustment"])
    payouts = amount_sum(group, ["payout"])
    balance = round(sales + fees + taxes + adjustments + payouts, 2)
    status = "balanced" if abs(balance) <= 0.05 else "balance_change"
    note = (
        "Verkaeufe, Gebuehren, Steuern/Adjustments und Auszahlungen heben sich rechnerisch auf."
        if status == "balanced"
        else "Nicht automatisch falsch: Der Saldo entspricht meist Anfangs-/Endbestand, Auszahlungs-Timing, PayPal-Zahlungen, Reserven oder Korrekturen im Etsy-Zahlungskonto."
    )
    return {
        "level": level,
        "shop": shop,
        "period": period,
        "sales": sales,
        "fees": fees,
        "taxes": taxes,
        "adjustments": adjustments,
        "payouts": payouts,
        "calculated_balance_change": balance,
        "status": status,
        "note": note,
    }


def amount_sum(group: pd.DataFrame, categories: list[str]) -> float:
    return round(float(group.loc[group["category"].isin(categories), "amount"].sum()), 2)


def build_etsy_accountable_comparison(evidence: pd.DataFrame, platform_transactions: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty or platform_transactions.empty:
        return pd.DataFrame(columns=ETSY_ACCOUNTABLE_COLUMNS)

    statement = platform_transactions[
        (platform_transactions["platform"].astype(str).str.lower() == "etsy")
        & (platform_transactions["source"].astype(str) == "Etsy Statement")
    ].copy()
    if statement.empty:
        return pd.DataFrame(columns=ETSY_ACCOUNTABLE_COLUMNS)

    statement["date"] = pd.to_datetime(statement["date"], errors="coerce")
    statement = statement[statement["date"].notna()].copy()
    if statement.empty:
        return pd.DataFrame(columns=ETSY_ACCOUNTABLE_COLUMNS)
    statement["period"] = statement["date"].dt.to_period("M").astype(str)

    accountable = evidence[evidence.apply(is_etsy_accountable_row, axis=1)].copy()
    if not accountable.empty:
        accountable["date"] = pd.to_datetime(accountable["date"], errors="coerce")
        accountable = accountable[accountable["date"].notna()].copy()
        accountable["period"] = accountable["date"].dt.to_period("M").astype(str)

    periods = sorted(set(statement["period"]) | (set(accountable["period"]) if not accountable.empty else set()))
    rows = [etsy_accountable_comparison_row("month", period, statement[statement["period"] == period], accountable[accountable["period"] == period] if not accountable.empty else pd.DataFrame()) for period in periods]

    years = sorted(set(period[:4] for period in periods))
    for year in years:
        statement_year = statement[statement["period"].astype(str).str.startswith(year)]
        accountable_year = accountable[accountable["period"].astype(str).str.startswith(year)] if not accountable.empty else pd.DataFrame()
        rows.append(etsy_accountable_comparison_row("year", year, statement_year, accountable_year))

    rows.append(etsy_accountable_comparison_row("all", "alle", statement, accountable))
    return pd.DataFrame(rows, columns=ETSY_ACCOUNTABLE_COLUMNS)


def etsy_accountable_comparison_row(level: str, period: str, statement: pd.DataFrame, accountable: pd.DataFrame) -> dict[str, object]:
    statement_sales = amount_sum(statement, ["ledger_order"])
    statement_costs = amount_sum(statement, ["ledger_fee", "ledger_tax", "ledger_adjustment"])
    statement_payouts = amount_sum(statement, ["payout"])
    statement_net = round(statement_sales + statement_costs, 2)
    statement_balance = round(statement_net + statement_payouts, 2)

    if accountable.empty:
        accountable_income = 0.0
        accountable_expenses = 0.0
    else:
        accountable_income = round(float(accountable.loc[accountable["doc_type"].eq("income"), "amount"].sum()), 2)
        accountable_expenses = round(float(accountable.loc[accountable["doc_type"].eq("expense"), "amount"].sum()), 2)
    accountable_net = round(accountable_income + accountable_expenses, 2)

    sales_diff = round(accountable_income - statement_sales, 2)
    costs_diff = round(accountable_expenses - statement_costs, 2)
    net_diff = round(accountable_net - statement_net, 2)
    status = "ok" if abs(net_diff) <= 5.0 else "prüfen"
    note = "Accountable-Etsy-Belege liegen nahe an Etsy-Verkäufen abzüglich Etsy-Kosten." if status == "ok" else "Abweichung zwischen Etsy-Monatsabrechnung und Accountable-Etsy-Belegen prüfen."

    return {
        "level": level,
        "period": period,
        "statement_sales": statement_sales,
        "accountable_income": accountable_income,
        "sales_diff": sales_diff,
        "statement_costs": statement_costs,
        "accountable_expenses": accountable_expenses,
        "costs_diff": costs_diff,
        "statement_net_before_payout": statement_net,
        "accountable_net": accountable_net,
        "net_diff": net_diff,
        "statement_payouts": statement_payouts,
        "statement_balance_change": statement_balance,
        "status": status,
        "note": note,
    }


def is_etsy_accountable_row(row: pd.Series) -> bool:
    doc_type = str(row.get("doc_type") or "")
    doc_platform = str(row.get("platform") or "").lower()
    row_platforms = str(row.get("platforms") or "").lower()
    text = " ".join(
        str(row.get(field) or "").lower()
        for field in ["counterparty", "description", "fyrst_counterparties", "evidence_path", "evidence_note"]
    )

    if doc_type == "expense":
        return doc_platform == "etsy" or "etsy" in text

    if doc_type == "income":
        if row_platforms:
            return "etsy" in row_platforms
        return "etsy" in text

    return False


def build_etsy_annual_pdf(report: pd.DataFrame, path: Path) -> None:
    load_reportlab()
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        rightMargin=0.8 * cm,
        leftMargin=0.8 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
        title="Etsy Jahresabgleich",
    )
    styles = getSampleStyleSheet()
    story: list = []
    story.append(Paragraph("Etsy Jahresabgleich", styles["Title"]))
    story.append(
        Paragraph(
            "Separater Plausibilitaetsbericht: Etsy-Verkaeufe plus Gebuehren, Steuern/Adjustments und Auszahlungen aus den Monatsabrechnungen.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    if report.empty:
        story.append(Paragraph("Keine Etsy-Monatsabrechnungen im Import gefunden.", styles["BodyText"]))
        pdf.build(story)
        return

    annual = report[report["level"].isin(["year", "all_shops_year"])].copy()
    monthly = report[report["level"] == "month"].copy()
    story.append(Paragraph("Jahressummen", styles["Heading2"]))
    story.append(report_table(annual))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Monate", styles["Heading2"]))
    story.append(report_table(monthly))
    pdf.build(story)


def report_table(frame: pd.DataFrame) -> Table:
    load_reportlab()
    if frame.empty:
        return Table([["Keine Daten"]])
    cols = ["level", "shop", "period", "sales", "fees", "taxes", "adjustments", "payouts", "calculated_balance_change", "status"]
    headers = ["Ebene", "Shop", "Zeitraum", "Verkaeufe", "Gebuehren", "Steuern", "Korrekturen", "Auszahlungen", "Saldo", "Status"]
    data = [headers]
    for _, row in frame[cols].iterrows():
        data.append(
            [
                row["level"],
                row["shop"],
                row["period"],
                money(row["sales"]),
                money(row["fees"]),
                money(row["taxes"]),
                money(row["adjustments"]),
                money(row["payouts"]),
                money(row["calculated_balance_change"]),
                row["status"],
            ]
        )
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (3, 1), (8, -1), "RIGHT"),
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
