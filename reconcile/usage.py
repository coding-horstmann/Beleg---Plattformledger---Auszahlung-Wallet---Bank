from __future__ import annotations

from collections import defaultdict

import pandas as pd


def build_bank_claim_usage(evidence: pd.DataFrame, bank: pd.DataFrame, tolerance: float = 0.05) -> pd.DataFrame:
    columns = [
        "tx_id",
        "bank_date",
        "bank_amount",
        "bank_counterparty",
        "bank_description",
        "claim_doc_count",
        "claim_doc_net_sum",
        "claim_doc_income_sum",
        "claim_doc_expense_sum",
        "delta_bank_minus_claim",
        "usage_status",
        "linked_doc_refs",
        "evidence_paths",
    ]
    if evidence.empty or bank.empty:
        return pd.DataFrame(columns=columns)

    claims: dict[str, dict[str, object]] = defaultdict(lambda: {"doc_ids": set(), "refs": [], "paths": set(), "amounts": []})
    for _, row in evidence.iterrows():
        tx_ids = split_ids(row.get("fyrst_tx_ids")) + split_ids(row.get("bridge_fyrst_tx_ids")) + split_ids(row.get("platform_bridge_fyrst_tx_ids"))
        for tx_id in set(tx_ids):
            if not tx_id:
                continue
            claim = claims[tx_id]
            doc_id = str(row.get("doc_id") or row.get("doc_ref") or "")
            if doc_id in claim["doc_ids"]:
                continue
            claim["doc_ids"].add(doc_id)
            claim["refs"].append(str(row.get("doc_ref") or ""))
            claim["paths"].add(str(row.get("evidence_path") or ""))
            try:
                claim["amounts"].append(float(row.get("amount") or 0))
            except Exception:
                claim["amounts"].append(0.0)

    rows: list[dict[str, object]] = []
    for _, tx in bank.sort_values(["date", "amount"]).iterrows():
        tx_id = str(tx.get("tx_id") or "")
        claim = claims.get(tx_id)
        amounts = claim["amounts"] if claim else []
        net_sum = round(sum(amounts), 2)
        income_sum = round(sum(amount for amount in amounts if amount > 0), 2)
        expense_sum = round(sum(amount for amount in amounts if amount < 0), 2)
        bank_amount = round(float(tx.get("amount") or 0), 2)
        delta = round(bank_amount - net_sum, 2)
        status = usage_status(bank_amount, net_sum, amounts, tolerance)
        paths_text = " | ".join(sorted(claim["paths"])) if claim else ""
        if status == "over_claim" and looks_like_paypal_balance_bridge(tx.get("counterparty", ""), tx.get("description", ""), paths_text):
            status = "paypal_balance_bridge"
        elif status == "over_claim" and looks_like_platform_gross_vs_net(tx.get("counterparty", ""), paths_text):
            status = "gross_platform_over_bank"
        rows.append(
            {
                "tx_id": tx_id,
                "bank_date": tx.get("date"),
                "bank_amount": bank_amount,
                "bank_counterparty": tx.get("counterparty", ""),
                "bank_description": tx.get("description", ""),
                "claim_doc_count": len(amounts),
                "claim_doc_net_sum": net_sum,
                "claim_doc_income_sum": income_sum,
                "claim_doc_expense_sum": expense_sum,
                "delta_bank_minus_claim": delta,
                "usage_status": status,
                "linked_doc_refs": " | ".join(claim["refs"]) if claim else "",
                "evidence_paths": paths_text,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def usage_status(bank_amount: float, net_sum: float, amounts: list[float], tolerance: float) -> str:
    if not amounts:
        return "no_claim"
    delta = round(bank_amount - net_sum, 2)
    if abs(delta) <= tolerance:
        return "balanced"
    if bank_amount > 0 and net_sum > bank_amount + tolerance:
        return "over_claim"
    if bank_amount < 0 and net_sum < bank_amount - tolerance:
        return "over_claim"
    return "under_or_timing_gap"


def looks_like_platform_gross_vs_net(counterparty: object, evidence_paths: str) -> bool:
    text = f"{counterparty} {evidence_paths}".lower()
    platform_counterparty = any(token in text for token in ["etsy", "ebay", "shopify"])
    platform_path = "plattformdetail" in text or "plattform" in text
    return platform_counterparty or platform_path


def looks_like_paypal_balance_bridge(counterparty: object, description: object, evidence_paths: str) -> bool:
    text = f"{counterparty} {description} {evidence_paths}".lower()
    paypal_bank_row = "paypal" in text
    paypal_evidence = "paypal-detail" in text or "paypal-fx" in text or "paypal-konto" in text
    return paypal_bank_row and paypal_evidence


def split_ids(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]
