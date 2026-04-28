from __future__ import annotations

import pandas as pd

from .matching import MatchSettings, days_between, text_similarity, token_set


LEFTOVER_COLUMNS = [
    "doc_ref",
    "doc_type",
    "doc_date",
    "doc_amount",
    "doc_counterparty",
    "bank_date",
    "bank_amount",
    "bank_counterparty",
    "amount_diff",
    "day_diff",
    "counterparty_score",
    "text_score",
    "score",
    "recommendation",
]

KNOWN_VENDOR_TOKENS = {
    "cursor",
    "google",
    "pinterest",
    "etsy",
    "ebay",
    "paypal",
    "ionos",
    "printful",
    "gelato",
    "microsoft",
    "runware",
    "ideogram",
    "europosters",
    "redbubble",
    "printler",
    "sprd",
}


def build_leftover_candidate_report(open_docs: pd.DataFrame, open_bank: pd.DataFrame, settings: MatchSettings) -> pd.DataFrame:
    if open_docs.empty or open_bank.empty:
        return pd.DataFrame(columns=LEFTOVER_COLUMNS)

    docs = open_docs.copy()
    bank = open_bank.copy()
    docs["date"] = pd.to_datetime(docs["date"], errors="coerce")
    bank["date"] = pd.to_datetime(bank["date"], errors="coerce")

    rows: list[dict[str, object]] = []
    for _, doc in docs.iterrows():
        doc_amount = float(doc.get("signed_amount") or 0)
        if abs(doc_amount) < 0.005:
            continue
        candidates = bank[(bank["amount"].astype(float) * doc_amount) > 0].copy()
        if candidates.empty:
            continue
        candidates["_day_diff"] = candidates["date"].apply(lambda value: days_between(value, doc.get("date")))
        candidates = candidates[candidates["_day_diff"] <= 90].copy()
        if candidates.empty:
            continue
        candidates["_amount_diff"] = candidates["amount"].astype(float).apply(lambda value: abs(value - doc_amount))
        candidates["_text_score"] = candidates["text"].apply(lambda value: text_similarity(doc.get("text", ""), value))
        candidates["_counterparty_score"] = candidates["counterparty"].apply(lambda value: counterparty_overlap(doc.get("counterparty", ""), value))
        candidates = candidates[candidates.apply(lambda row: leftover_candidate_allowed(doc, row, settings), axis=1)].copy()
        if candidates.empty:
            continue
        candidates["_score"] = candidates.apply(lambda row: leftover_score(doc_amount, row, settings), axis=1)
        candidates = candidates.sort_values(["_score", "_amount_diff", "_day_diff"], ascending=[False, True, True]).head(3)
        for _, tx in candidates.iterrows():
            if float(tx["_score"]) < 0.35:
                continue
            rows.append(
                {
                    "doc_ref": doc.get("doc_ref", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "doc_date": doc.get("date"),
                    "doc_amount": round(doc_amount, 2),
                    "doc_counterparty": doc.get("counterparty", ""),
                    "bank_date": tx.get("date"),
                    "bank_amount": round(float(tx.get("amount") or 0), 2),
                    "bank_counterparty": tx.get("counterparty", ""),
                    "amount_diff": round(float(tx["_amount_diff"]), 2),
                    "day_diff": int(tx["_day_diff"]),
                    "counterparty_score": round(float(tx["_counterparty_score"]), 3),
                    "text_score": round(float(tx["_text_score"]), 3),
                    "score": round(float(tx["_score"]), 3),
                    "recommendation": recommendation_for(float(tx["_amount_diff"]), int(tx["_day_diff"]), float(tx["_text_score"]), float(tx["_counterparty_score"])),
                }
            )
    return pd.DataFrame(rows, columns=LEFTOVER_COLUMNS).sort_values(["score", "doc_ref"], ascending=[False, True])


def leftover_candidate_allowed(doc: pd.Series, tx: pd.Series, settings: MatchSettings) -> bool:
    amount_diff = float(tx["_amount_diff"])
    text_score = float(tx["_text_score"])
    counterparty_score = float(tx["_counterparty_score"])
    if obvious_vendor_mismatch(doc.get("counterparty", ""), tx.get("counterparty", "")):
        return False
    return amount_diff <= settings.tolerance_cents / 100 or counterparty_score >= 0.35 or text_score >= 0.18


def leftover_score(doc_amount: float, tx: pd.Series, settings: MatchSettings) -> float:
    amount_diff = float(tx["_amount_diff"])
    day_diff = int(tx["_day_diff"])
    text_score = float(tx["_text_score"])
    counterparty_score = float(tx["_counterparty_score"])
    amount_scale = max(abs(doc_amount), abs(float(tx.get("amount") or 0)), 1)
    amount_score = 1 - min(1, amount_diff / max(amount_scale * 0.2, settings.tolerance_cents / 100, 0.01))
    date_score = 1 - min(1, day_diff / 90)
    return amount_score * 0.45 + date_score * 0.15 + counterparty_score * 0.25 + text_score * 0.15


def counterparty_overlap(left: object, right: object) -> float:
    left_tokens = {token for token in token_set(str(left)) if len(token) >= 4}
    right_tokens = {token for token in token_set(str(right)) if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def obvious_vendor_mismatch(left: object, right: object) -> bool:
    left_known = token_set(str(left)) & KNOWN_VENDOR_TOKENS
    right_known = token_set(str(right)) & KNOWN_VENDOR_TOKENS
    return bool(left_known and right_known and left_known.isdisjoint(right_known))


def recommendation_for(amount_diff: float, day_diff: int, text_score: float, counterparty_score: float) -> str:
    if amount_diff <= 0.05 and day_diff <= 14 and (counterparty_score >= 0.35 or text_score >= 0.18):
        return "starker Prüfkandidat"
    if amount_diff <= 5 and day_diff <= 45 and (counterparty_score >= 0.35 or text_score >= 0.18):
        return "möglicher Prüfkandidat"
    if text_score >= 0.35:
        return "Text ähnlich, Betrag/Datum prüfen"
    return "schwach, nur zur Sichtprüfung"
