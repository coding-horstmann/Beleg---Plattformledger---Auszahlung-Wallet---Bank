from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .matching import MatchSettings, empty_links, empty_matches, make_link_row, make_match_row
from .parsers import clean_text, stable_id


PACKAGE_STATUS_AUTO = "auto_matched"


def build_platform_package_matches(
    docs: pd.DataFrame,
    bank: pd.DataFrame,
    existing_matches: pd.DataFrame,
    existing_links: pd.DataFrame,
    platform_transactions: pd.DataFrame,
    platform_doc_matches: pd.DataFrame,
    platform_doc_links: pd.DataFrame,
    platform_bank_matches: pd.DataFrame,
    settings: MatchSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build payout-level reconciliation packages.

    A package is stricter than a fuzzy match: Accountable documents must add up
    to the FYRST transfer. When the uploaded data is not enough to close the
    package, the function still emits a diagnostic row for review.
    """
    if docs.empty or bank.empty or platform_transactions.empty or platform_bank_matches.empty:
        return empty_matches(), empty_links(), empty_package_report()

    docs_work = docs.copy()
    docs_work["date"] = pd.to_datetime(docs_work["date"], errors="coerce")
    docs_work["_platform_key"] = docs_work.apply(doc_platform_key, axis=1)
    bank_by_id = {row["tx_id"]: row for _, row in bank.iterrows()} if "tx_id" in bank.columns else {}

    package_doc_ids, package_detail_ids = docs_by_platform_payout(
        platform_transactions,
        platform_doc_matches,
        platform_doc_links,
    )
    bridge_by_payout = {
        str(key): group.copy()
        for key, group in platform_bank_matches.groupby("payout_id", dropna=False)
        if str(key or "")
    }
    existing_txs = set(existing_matches["tx_id"]) if not existing_matches.empty else set()
    auto_linked_fee_docs: set[str] = set()

    payout_rows = platform_transactions[
        platform_transactions["category"].astype(str).isin(["payout", "bank_transfer"])
    ].copy()
    if not payout_rows.empty:
        payout_rows["_source_rank"] = payout_rows["source"].astype(str).eq("Etsy Statement").astype(int)
        payout_rows = (
            payout_rows.sort_values(["platform", "payout_id", "_source_rank"], ascending=[True, True, False])
            .drop_duplicates(["platform", "payout_id"], keep="first")
            .drop(columns=["_source_rank"], errors="ignore")
        )
    detail_rows = platform_transactions[
        platform_transactions["category"].astype(str).isin(
            ["order", "fee", "charge", "refund", "adjustment", "ledger_order", "ledger_fee", "ledger_tax", "ledger_adjustment"]
        )
    ].copy()

    report_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []

    for _, payout in payout_rows.sort_values(["platform", "date", "amount"]).iterrows():
        payout_id = str(payout.get("payout_id") or "")
        platform = canonical_platform(payout.get("platform"), payout.get("description"), payout.get("source_file"))
        package_details = detail_rows[detail_rows["payout_id"].astype(str).eq(payout_id)].copy() if payout_id else detail_rows.iloc[0:0]
        if platform == "etsy" and not package_details.empty:
            statement_rows = package_details[package_details["source"].astype(str).eq("Etsy Statement")].copy()
            if not statement_rows.empty:
                package_details = statement_rows
        bridge_rows = bridge_by_payout.get(payout_id, pd.DataFrame())
        if bridge_rows.empty:
            report_rows.append(package_report_row(payout, platform, payout_id, package_details, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "no_bank_bridge"))
            continue

        linked_docs = docs_work[docs_work["doc_id"].isin(package_doc_ids.get(payout_id, set()))].copy()
        bridge = bridge_rows.sort_values("bank_date").iloc[0]
        bank_amount = float(bridge.get("bank_amount") or 0)
        bank_tx_id = str(bridge.get("tx_id") or "")

        fee_candidates = candidate_fee_docs_for_package(
            docs_work,
            linked_docs,
            platform,
            payout.get("date"),
            auto_linked_fee_docs,
        )
        selected_fee_docs = pd.DataFrame(columns=docs_work.columns)
        linked_sum = float(linked_docs["signed_amount"].sum()) if not linked_docs.empty else 0.0
        target_gap = round(bank_amount - linked_sum, 2)
        if abs(target_gap) > settings.tolerance_cents / 100:
            selected_ids = find_subset_for_target(fee_candidates, target_gap, settings)
            if selected_ids:
                selected_fee_docs = fee_candidates[fee_candidates["doc_id"].isin(selected_ids)].copy()

        selected_docs = pd.concat([linked_docs, selected_fee_docs], ignore_index=True) if not selected_fee_docs.empty else linked_docs
        package_sum = round(float(selected_docs["signed_amount"].sum()), 2) if not selected_docs.empty else 0.0
        gap = round(bank_amount - package_sum, 2)
        can_auto_match = (
            bank_tx_id
            and bank_tx_id not in existing_txs
            and len(selected_docs) > 0
            and abs(gap) <= settings.tolerance_cents / 100
        )
        status = PACKAGE_STATUS_AUTO if can_auto_match else package_status(linked_docs, fee_candidates, selected_fee_docs, gap)

        report_rows.append(
            package_report_row(
                payout,
                platform,
                payout_id,
                package_details,
                linked_docs,
                fee_candidates,
                selected_fee_docs,
                bridge,
                status,
                package_sum=package_sum,
                gap=gap,
            )
        )

        if not can_auto_match:
            continue

        tx = bank_by_id.get(bank_tx_id)
        if tx is None:
            tx = tx_like_from_bridge(bridge)
        match_id = stable_id("match", "platform_package_net", bank_tx_id, payout_id, ",".join(sorted(selected_docs["doc_id"].astype(str))))
        payload = {
            "method": "platform_package_net",
            "confidence": 0.93,
            "doc_sum": package_sum,
            "gap": gap,
            "amount_diff": abs(gap),
            "day_diff": average_day_diff(selected_docs, tx.get("date") if "date" in tx else bridge.get("bank_date")),
            "text_score": 1.0,
            "explanation": f"{platform_label(platform)}-Abrechnungspaket: Accountable-Belege ergeben {package_sum:.2f} EUR und schließen den FYRST-Umsatz innerhalb der Toleranz.",
        }
        match_rows.append(make_match_row(match_id, tx, len(selected_docs), payload))
        for _, doc in selected_docs.iterrows():
            link_rows.append(make_link_row(match_id, doc))
            if doc.get("doc_type") == "expense":
                auto_linked_fee_docs.add(str(doc.get("doc_id")))

    return (
        pd.DataFrame(match_rows) if match_rows else empty_matches(),
        pd.DataFrame(link_rows) if link_rows else empty_links(),
        pd.DataFrame(report_rows) if report_rows else empty_package_report(),
    )


def docs_by_platform_payout(
    platform_transactions: pd.DataFrame,
    platform_doc_matches: pd.DataFrame,
    platform_doc_links: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    docs_by_payout: dict[str, set[str]] = defaultdict(set)
    details_by_payout: dict[str, set[str]] = defaultdict(set)
    if platform_transactions.empty or platform_doc_matches.empty or platform_doc_links.empty:
        return docs_by_payout, details_by_payout

    tx_by_id = {row["tx_id"]: row for _, row in platform_transactions.iterrows()}
    match_by_id = {row["match_id"]: row for _, row in platform_doc_matches.iterrows()}
    for _, link in platform_doc_links.iterrows():
        match = match_by_id.get(link.get("match_id"))
        if match is None:
            continue
        tx = tx_by_id.get(match.get("tx_id"))
        if tx is None:
            continue
        payout_id = str(tx.get("payout_id") or "")
        if not payout_id:
            continue
        docs_by_payout[payout_id].add(str(link.get("doc_id")))
        details_by_payout[payout_id].add(str(tx.get("tx_id")))
    return docs_by_payout, details_by_payout


def candidate_fee_docs_for_package(
    docs: pd.DataFrame,
    linked_docs: pd.DataFrame,
    platform: str,
    payout_date: object,
    already_used: set[str],
) -> pd.DataFrame:
    if docs.empty or not platform:
        return pd.DataFrame(columns=docs.columns)
    payout_ts = pd.Timestamp(payout_date) if payout_date is not None and not pd.isna(payout_date) else pd.NaT
    candidates = docs[
        (docs["doc_type"] == "expense")
        & (docs["_platform_key"] == platform)
        & (~docs["doc_id"].isin(already_used))
    ].copy()
    if candidates.empty:
        return candidates
    if not linked_docs.empty:
        candidates = candidates[~candidates["doc_id"].isin(set(linked_docs["doc_id"]))].copy()
    if pd.isna(payout_ts):
        return candidates

    month_start = payout_ts.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    if payout_ts.day <= 7:
        month_start = (month_start - pd.offsets.MonthBegin(1)).normalize()
    candidates = candidates[(candidates["date"] >= month_start) & (candidates["date"] <= month_end)].copy()
    return candidates.sort_values(["date", "signed_amount"])


def find_subset_for_target(candidates: pd.DataFrame, target: float, settings: MatchSettings) -> set[str]:
    if candidates.empty:
        return set()
    target_cents = int(round(float(target) * 100))
    tolerance = int(settings.tolerance_cents)
    work = candidates.copy()
    work["_distance"] = (work["signed_amount"].astype(float) - float(target)).abs()
    work = work.sort_values(["_distance", "date"]).head(18)
    states: dict[int, tuple[str, ...]] = {0: ()}
    best: tuple[int, tuple[str, ...]] | None = None
    for _, row in work.iterrows():
        amount = int(round(float(row["signed_amount"]) * 100))
        doc_id = str(row["doc_id"])
        updates: dict[int, tuple[str, ...]] = {}
        for total, ids in states.items():
            new_total = total + amount
            if new_total not in states and new_total not in updates:
                updates[new_total] = ids + (doc_id,)
                if abs(new_total - target_cents) <= tolerance:
                    if best is None or abs(new_total - target_cents) < abs(best[0] - target_cents):
                        best = (new_total, updates[new_total])
        states.update(updates)
        if len(states) > settings.max_subset_states:
            keep = sorted(states.items(), key=lambda item: abs(item[0] - target_cents))[: settings.max_subset_states]
            states = dict(keep)
    return set(best[1]) if best else set()


def package_report_row(
    payout: pd.Series,
    platform: str,
    payout_id: str,
    detail_rows: pd.DataFrame,
    linked_docs: pd.DataFrame,
    fee_candidates: pd.DataFrame,
    selected_fee_docs: pd.DataFrame,
    bridge: pd.Series | None,
    status: str,
    package_sum: float | None = None,
    gap: float | None = None,
) -> dict[str, object]:
    bank_amount = float(bridge.get("bank_amount") or 0) if bridge is not None else 0.0
    linked_doc_sum = round(float(linked_docs["signed_amount"].sum()), 2) if not linked_docs.empty else 0.0
    candidate_fee_sum = round(float(fee_candidates["signed_amount"].sum()), 2) if not fee_candidates.empty else 0.0
    selected_fee_sum = round(float(selected_fee_docs["signed_amount"].sum()), 2) if not selected_fee_docs.empty else 0.0
    detail_net = round(float(detail_rows["amount"].sum()), 2) if not detail_rows.empty else 0.0
    order_sum = round(float(detail_rows.loc[detail_rows["category"].isin(["order", "ledger_order"]), "amount"].sum()), 2) if not detail_rows.empty else 0.0
    fee_sum = round(
        float(
            detail_rows.loc[
                detail_rows["category"].isin(["fee", "charge", "refund", "adjustment", "ledger_fee", "ledger_tax", "ledger_adjustment"]),
                "amount",
            ].sum()
        ),
        2,
    ) if not detail_rows.empty else 0.0
    package_sum_value = linked_doc_sum + selected_fee_sum if package_sum is None else package_sum
    gap_value = round(bank_amount - package_sum_value, 2) if gap is None and bridge is not None else gap
    return {
        "package_id": stable_id("package", platform, payout_id, payout.get("date"), payout.get("amount")),
        "platform": platform,
        "payout_id": payout_id,
        "payout_date": payout.get("date"),
        "platform_payout_amount": round(float(payout.get("amount") or 0), 2),
        "bank_tx_id": "" if bridge is None else bridge.get("tx_id", ""),
        "bank_date": "" if bridge is None else bridge.get("bank_date", ""),
        "bank_amount": bank_amount if bridge is not None else "",
        "platform_detail_rows": len(detail_rows),
        "platform_order_amount": order_sum,
        "platform_fee_amount": fee_sum,
        "platform_detail_net": detail_net,
        "linked_doc_count": len(linked_docs),
        "linked_doc_sum": linked_doc_sum,
        "candidate_fee_doc_count": len(fee_candidates),
        "candidate_fee_doc_sum": candidate_fee_sum,
        "selected_fee_doc_count": len(selected_fee_docs),
        "selected_fee_doc_sum": selected_fee_sum,
        "package_doc_sum": package_sum_value,
        "gap_to_bank": gap_value if gap_value is not None else "",
        "package_status": status,
        "note": package_note(status),
        "linked_doc_refs": join_unique(linked_docs.get("doc_ref", [])),
        "candidate_fee_doc_refs": join_unique(fee_candidates.get("doc_ref", [])),
        "selected_fee_doc_refs": join_unique(selected_fee_docs.get("doc_ref", [])),
    }


def package_status(linked_docs: pd.DataFrame, fee_candidates: pd.DataFrame, selected_fee_docs: pd.DataFrame, gap: float) -> str:
    if linked_docs.empty:
        return "no_accountable_sales_link"
    if abs(gap) <= 0.05:
        return PACKAGE_STATUS_AUTO
    if not fee_candidates.empty and selected_fee_docs.empty:
        return "candidate_fees_do_not_close"
    return "package_gap"


def package_note(status: str) -> str:
    notes = {
        PACKAGE_STATUS_AUTO: "Accountable-Belege schließen den FYRST-Umsatz rechnerisch.",
        "no_bank_bridge": "Plattform-Auszahlung vorhanden, aber keine passende FYRST-Gegenbuchung im Upload.",
        "no_accountable_sales_link": "FYRST-Auszahlung gefunden, aber keine Accountable-Verkaufsbelege mit dieser Plattformauszahlung verknüpft.",
        "candidate_fees_do_not_close": "Es gibt passende Gebührenkandidaten, aber ihre Summe schließt die Auszahlung nicht.",
        "package_gap": "Verkäufe/Gebühren und FYRST-Auszahlung haben eine Restdifferenz.",
    }
    return notes.get(status, "Paket benötigt manuelle Kontrolle.")


def tx_like_from_bridge(bridge: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "tx_id": bridge.get("tx_id", ""),
            "source": bridge.get("bank_source", "FYRST"),
            "date": bridge.get("bank_date"),
            "amount": bridge.get("bank_amount", 0),
            "counterparty": bridge.get("bank_counterparty", ""),
            "description": bridge.get("bank_description", ""),
        }
    )


def average_day_diff(docs: pd.DataFrame, bank_date: object) -> float:
    if docs.empty or bank_date is None or pd.isna(bank_date):
        return 0.0
    bank_ts = pd.Timestamp(bank_date).normalize()
    return round(float(docs["date"].apply(lambda value: abs((pd.Timestamp(value).normalize() - bank_ts).days)).mean()), 1)


def doc_platform_key(row: pd.Series) -> str:
    return canonical_platform(row.get("platform"), row.get("counterparty"), row.get("description"))


def canonical_platform(*values: object) -> str:
    text = " ".join(clean_text(value).lower() for value in values if clean_text(value))
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    rules = [
        ("printler", ["printler"]),
        ("printful", ["printful"]),
        ("gelato", ["gelato"]),
        ("etsy", ["etsy"]),
        ("ebay", ["ebay"]),
        ("shopify", ["shopify"]),
        ("paypal", ["paypal"]),
        ("redbubble", ["redbubble"]),
        ("spreadshirt", ["sprd", "spreadshirt"]),
        ("juniqe", ["juniqe"]),
        ("teepublic", ["teepublic", "tee public"]),
        ("ionos", ["ionos", "1&1"]),
    ]
    for key, needles in rules:
        if any(needle in text for needle in needles):
            return key
    return ""


def platform_label(platform: str) -> str:
    labels = {
        "etsy": "Etsy",
        "ebay": "eBay",
        "shopify": "Shopify",
        "paypal": "PayPal",
        "printful": "Printful",
        "printler": "Printler",
        "gelato": "Gelato",
        "spreadshirt": "Spreadshirt",
    }
    return labels.get(platform, platform or "Plattform")


def join_unique(values) -> str:
    seen: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text == "nan" or text in seen:
            continue
        seen.append(text)
    return " | ".join(seen)


def empty_package_report() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "package_id",
            "platform",
            "payout_id",
            "payout_date",
            "platform_payout_amount",
            "bank_tx_id",
            "bank_date",
            "bank_amount",
            "platform_detail_rows",
            "platform_order_amount",
            "platform_fee_amount",
            "platform_detail_net",
            "linked_doc_count",
            "linked_doc_sum",
            "candidate_fee_doc_count",
            "candidate_fee_doc_sum",
            "selected_fee_doc_count",
            "selected_fee_doc_sum",
            "package_doc_sum",
            "gap_to_bank",
            "package_status",
            "note",
            "linked_doc_refs",
            "candidate_fee_doc_refs",
            "selected_fee_doc_refs",
        ]
    )
