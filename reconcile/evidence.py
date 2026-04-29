from __future__ import annotations

import pandas as pd


def build_document_evidence(
    docs: pd.DataFrame,
    bank: pd.DataFrame,
    paypal: pd.DataFrame,
    matches: pd.DataFrame,
    links: pd.DataFrame,
    paypal_doc_matches: pd.DataFrame,
    paypal_doc_links: pd.DataFrame,
    paypal_bank_matches: pd.DataFrame,
    platform_transactions: pd.DataFrame | None = None,
    platform_doc_matches: pd.DataFrame | None = None,
    platform_doc_links: pd.DataFrame | None = None,
    platform_bank_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one evidence row per Accountable document.

    This is the tax-review view: start with the document, then show the payment chain.
    PayPal is treated as an intermediate ledger, not as another bank account.
    """
    if docs.empty:
        return pd.DataFrame()

    platform_transactions = platform_transactions if platform_transactions is not None else pd.DataFrame()
    platform_doc_matches = platform_doc_matches if platform_doc_matches is not None else pd.DataFrame()
    platform_doc_links = platform_doc_links if platform_doc_links is not None else pd.DataFrame()
    platform_bank_matches = platform_bank_matches if platform_bank_matches is not None else pd.DataFrame()

    bank_by_id = index_by(bank, "tx_id")
    paypal_by_id = index_by(paypal, "pp_id")
    platform_by_id = index_by(platform_transactions, "tx_id")
    fyrst_matches_by_id = index_by(matches, "match_id")
    paypal_matches_by_id = index_by(paypal_doc_matches, "match_id")
    platform_matches_by_id = index_by(platform_doc_matches, "match_id")
    bridges_by_pp = group_by(paypal_bank_matches, "pp_id")
    transfer_by_related = group_paypal_transfers_by_related(paypal)
    platform_bridges_by_payout = group_by(platform_bank_matches, "payout_id")
    platform_bridges_by_tx = group_by(platform_bank_matches, "platform_tx_id")

    fyrst_links_by_doc = group_by(links, "doc_id")
    paypal_links_by_doc = group_by(paypal_doc_links, "doc_id")
    platform_links_by_doc = group_by(platform_doc_links, "doc_id")

    rows: list[dict[str, object]] = []
    for _, doc in docs.sort_values(["doc_type", "date", "doc_ref"]).iterrows():
        doc_id = doc["doc_id"]
        direct_matches = []
        for _, link in fyrst_links_by_doc.get(doc_id, pd.DataFrame()).iterrows():
            match = fyrst_matches_by_id.get(link["match_id"])
            if match is None:
                continue
            tx = bank_by_id.get(match.get("tx_id"))
            direct_matches.append((match, tx))

        paypal_details = []
        bridged_bank_rows = []
        for _, link in paypal_links_by_doc.get(doc_id, pd.DataFrame()).iterrows():
            match = paypal_matches_by_id.get(link["match_id"])
            if match is None:
                continue
            pp = paypal_by_id.get(match.get("tx_id"))
            if pp is None:
                continue
            paypal_details.append((match, pp))
            related_transfers = transfer_by_related.get(str(pp.get("transaction_code") or ""), [])
            for transfer in related_transfers:
                for _, bridge in bridges_by_pp.get(transfer.get("pp_id"), pd.DataFrame()).iterrows():
                    bridged_bank_rows.append((transfer, bridge))

        platform_details = []
        platform_bridge_rows = []
        for _, link in platform_links_by_doc.get(doc_id, pd.DataFrame()).iterrows():
            match = platform_matches_by_id.get(link["match_id"])
            if match is None:
                continue
            platform_tx = platform_by_id.get(match.get("tx_id"))
            if platform_tx is None:
                continue
            platform_details.append((match, platform_tx))
            payout_id = str(platform_tx.get("payout_id") or "")
            platform_category = str(platform_tx.get("category") or "")
            for _, bridge in platform_bridges_by_tx.get(platform_tx.get("tx_id"), pd.DataFrame()).iterrows():
                platform_bridge_rows.append((platform_tx, bridge))
            if payout_id and platform_category not in {"charge"}:
                for _, bridge in platform_bridges_by_payout.get(payout_id, pd.DataFrame()).iterrows():
                    platform_bridge_rows.append((platform_tx, bridge))

        if paypal_details:
            direct_matches = []

        has_paypal_fx_package = any(str(match.get("method")) == "paypal_fx_batch" for match, _ in paypal_details)

        if paypal_details and bridged_bank_rows:
            level = "vollständig belegt"
            if has_paypal_fx_package:
                path = "Accountable → PayPal-FX-Paket → FYRST"
                note = "PayPal-Fremdwährungspaket erklärt mehrere Belege netto; passende PayPal-Bankumbuchung ist in FYRST gefunden."
            else:
                path = "Accountable → PayPal-Detail → FYRST"
                note = "PayPal erklärt den Einzelbetrag; passende PayPal-Bankumbuchung ist in FYRST gefunden."
        elif paypal_details:
            level = "vollständig belegt"
            if has_paypal_fx_package:
                path = "Accountable → PayPal-FX-Paket"
                note = "PayPal-Fremdwährungspaket erklärt mehrere Belege netto über eine EUR-Umrechnung. Eine spätere Umbuchung zu FYRST kann als Sammelumbuchung laufen."
            else:
                path = "Accountable → PayPal-Konto"
                note = "PayPal-Transaktion passt zum Beleg. Eine spätere Umbuchung zu FYRST kann als Sammelumbuchung laufen und wird nicht zwingend je Einzelbeleg aufgeteilt."
        elif platform_details and platform_bridge_rows:
            level = "vollständig belegt"
            path = "Accountable → Plattformdetail → FYRST"
            note = "Plattformdaten erklären den Einzelbetrag; passende Plattform-Auszahlung ist in FYRST gefunden."
        elif direct_matches:
            level = "vollständig belegt"
            if any(str(match.get("method")) == "platform_package_net" for match, _ in direct_matches):
                path = "Accountable → Plattform-Abrechnungspaket → FYRST"
                note = "Beleg ist Teil eines Plattformpakets; Accountable-Belege schließen den FYRST-Umsatz rechnerisch."
            elif any(str(match.get("method")) in {"batch_exact", "batch_net_fee_gap"} for match, _ in direct_matches):
                if platform_details:
                    path = "Accountable → Plattformdetail + FYRST-Sammelzahlung"
                    note = "Plattformdaten stützen den Beleg; FYRST zeigt die Sammel-/Nettozahlung."
                else:
                    path = "Accountable → FYRST-Sammelzahlung"
                    note = "Beleg ist Teil einer FYRST-Sammel-/Nettozahlung. Details hängen von Plattformabrechnung ab."
            else:
                if platform_details:
                    path = "Accountable → Plattformdetail + FYRST direkt"
                    note = "Betrag und Datum passen zu FYRST; Plattformdaten liefern zusätzliche Bestell-/Gebührendetails."
                else:
                    path = "Accountable → FYRST direkt"
                    note = "Betrag und Datum passen direkt zu einem FYRST-Umsatz."
        elif paypal_details:
            level = "teilweise belegt"
            path = "Accountable → PayPal-Detail"
            note = "PayPal-Detail passt, aber die zugehörige FYRST-Umbuchung wurde nicht eindeutig gefunden."
        elif platform_details:
            level = "teilweise belegt"
            path = "Accountable → Plattformdetail"
            note = "Plattform-Bestellung oder -Gebühr passt; die zugehörige FYRST-Auszahlung ist noch nicht eindeutig verbunden."
        else:
            level = "offen"
            path = "kein belastbarer Zahlungsnachweis"
            note = open_doc_note(doc)

        level = normalize_evidence_level(level)
        rows.append(
            {
                "doc_id": doc_id,
                "doc_ref": doc.get("doc_ref", ""),
                "doc_type": doc.get("doc_type", ""),
                "date": doc.get("date"),
                "date_source": doc.get("date_source", ""),
                "amount": doc.get("signed_amount"),
                "platform": doc.get("platform", ""),
                "counterparty": doc.get("counterparty", ""),
                "description": doc.get("description", ""),
                "evidence_level": level,
                "evidence_path": path,
                "evidence_note": note,
                "fyrst_methods": join_unique(match.get("method") for match, _ in direct_matches),
                "fyrst_tx_ids": join_unique(match.get("tx_id") for match, _ in direct_matches),
                "fyrst_dates": join_unique(format_date(tx.get("date")) for _, tx in direct_matches if tx is not None),
                "fyrst_amounts": join_unique(format_amount(tx.get("amount")) for _, tx in direct_matches if tx is not None),
                "fyrst_counterparties": join_unique(tx.get("counterparty") for _, tx in direct_matches if tx is not None),
                "paypal_detail_ids": join_unique(pp.get("pp_id") for _, pp in paypal_details),
                "paypal_dates": join_unique(format_date(pp.get("date")) for _, pp in paypal_details),
                "paypal_amounts": join_unique(format_amount(pp.get("amount")) for _, pp in paypal_details),
                "paypal_counterparties": join_unique(pp.get("counterparty") for _, pp in paypal_details),
                "paypal_transfer_ids": join_unique(transfer.get("pp_id") for transfer, _ in bridged_bank_rows),
                "bridge_fyrst_tx_ids": join_unique(bridge.get("tx_id") for _, bridge in bridged_bank_rows),
                "bridge_fyrst_dates": join_unique(format_date(bridge.get("bank_date")) for _, bridge in bridged_bank_rows),
                "bridge_fyrst_amounts": join_unique(format_amount(bridge.get("bank_amount")) for _, bridge in bridged_bank_rows),
                "platform_detail_ids": join_unique(platform_tx.get("tx_id") for _, platform_tx in platform_details),
                "platforms": join_unique(platform_tx.get("platform") for _, platform_tx in platform_details),
                "platform_categories": join_unique(platform_tx.get("category") for _, platform_tx in platform_details),
                "platform_order_ids": join_unique(platform_tx.get("order_id") for _, platform_tx in platform_details),
                "platform_dates": join_unique(format_date(platform_tx.get("date")) for _, platform_tx in platform_details),
                "platform_amounts": join_unique(format_amount(platform_tx.get("amount")) for _, platform_tx in platform_details),
                "platform_counterparties": join_unique(platform_tx.get("counterparty") for _, platform_tx in platform_details),
                "platform_payout_ids": join_unique(platform_tx.get("payout_id") for _, platform_tx in platform_details),
                "platform_bridge_fyrst_tx_ids": join_unique(bridge.get("tx_id") for _, bridge in platform_bridge_rows),
                "platform_bridge_fyrst_dates": join_unique(format_date(bridge.get("bank_date")) for _, bridge in platform_bridge_rows),
                "platform_bridge_fyrst_amounts": join_unique(format_amount(bridge.get("bank_amount")) for _, bridge in platform_bridge_rows),
                "confidence": max_confidence(direct_matches, paypal_details, bridged_bank_rows, platform_details, platform_bridge_rows),
            }
        )

    return pd.DataFrame(rows)


def build_open_bank_after_evidence(
    bank: pd.DataFrame,
    matches: pd.DataFrame,
    paypal_bank_matches: pd.DataFrame,
    platform_bank_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if bank.empty:
        return bank.copy()
    linked = set(matches["tx_id"]) if not matches.empty else set()
    if not paypal_bank_matches.empty:
        linked |= set(paypal_bank_matches["tx_id"])
    if platform_bank_matches is not None and not platform_bank_matches.empty:
        linked |= set(platform_bank_matches["tx_id"])
    return bank[~bank["tx_id"].isin(linked)].copy()


def normalize_evidence_level(value: object) -> str:
    text = str(value or "")
    return text.replace("vollstÃ¤ndig", "vollständig")


def index_by(frame: pd.DataFrame, column: str) -> dict[object, pd.Series]:
    if frame.empty or column not in frame.columns:
        return {}
    return {row[column]: row for _, row in frame.iterrows()}


def group_by(frame: pd.DataFrame, column: str) -> dict[object, pd.DataFrame]:
    if frame.empty or column not in frame.columns:
        return {}
    return {key: group.copy() for key, group in frame.groupby(column, dropna=False)}


def group_paypal_transfers_by_related(paypal: pd.DataFrame) -> dict[str, list[pd.Series]]:
    if paypal.empty:
        return {}
    transfers = paypal[paypal["category"] == "bank_transfer"].copy()
    grouped: dict[str, list[pd.Series]] = {}
    for _, row in transfers.iterrows():
        related = str(row.get("related_code") or "")
        if related:
            grouped.setdefault(related, []).append(row)
    return grouped


def join_unique(values) -> str:
    seen = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value)
        if not text or text == "nan":
            continue
        if text not in seen:
            seen.append(text)
    return " | ".join(seen)


def format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def format_amount(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def max_confidence(
    direct_matches,
    paypal_details,
    bridged_bank_rows,
    platform_details=None,
    platform_bridge_rows=None,
) -> float | None:
    values: list[float] = []
    for match, _ in direct_matches:
        if "confidence" in match and not pd.isna(match["confidence"]):
            values.append(float(match["confidence"]))
    for match, _ in paypal_details:
        if "confidence" in match and not pd.isna(match["confidence"]):
            values.append(float(match["confidence"]))
    for _, bridge in bridged_bank_rows:
        if "confidence" in bridge and not pd.isna(bridge["confidence"]):
            values.append(float(bridge["confidence"]))
    for match, _ in platform_details or []:
        if "confidence" in match and not pd.isna(match["confidence"]):
            values.append(float(match["confidence"]))
    for _, bridge in platform_bridge_rows or []:
        if "confidence" in bridge and not pd.isna(bridge["confidence"]):
            values.append(float(bridge["confidence"]))
    return round(max(values), 3) if values else None


def open_doc_note(doc: pd.Series) -> str:
    platform = str(doc.get("platform") or "")
    doc_type = str(doc.get("doc_type") or "")
    if doc_type == "income" and platform in {"etsy", "ebay", "redbubble", "juniqe"}:
        return "Plattform-/Marktplatzdetails fehlen oder reichen nicht bis zur FYRST-Auszahlung."
    if doc_type == "expense" and platform in {"printful", "gelato", "etsy", "ebay"}:
        return "Lieferanten-/Plattformausgabe offen; PayPal/FYRST enthält keinen eindeutigen Einzelbezug."
    if platform:
        return f"Plattform {platform} erkannt, aber keine vollständige Zahlungskette gefunden."
    return "Kein eindeutiger Plattform-, PayPal- oder FYRST-Bezug gefunden."
