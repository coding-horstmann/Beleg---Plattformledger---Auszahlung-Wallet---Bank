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

HYPOTHESIS_COLUMNS = [
    "hypothesis_source",
    "source_id",
    "source_date",
    "source_amount",
    "source_rest_amount",
    "source_counterparty",
    "source_context",
    "doc_ref",
    "doc_type",
    "doc_date",
    "doc_amount",
    "doc_counterparty",
    "amount_diff",
    "day_diff",
    "counterparty_score",
    "text_score",
    "score",
    "recommendation",
    "caution",
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


def build_hypothesis_candidate_report(
    open_docs: pd.DataFrame,
    open_bank: pd.DataFrame,
    settlement_detail_report: pd.DataFrame,
    settings: MatchSettings,
) -> pd.DataFrame:
    """Suggest manual review candidates without turning them into confirmed matches."""
    if open_docs.empty:
        return pd.DataFrame(columns=HYPOTHESIS_COLUMNS)

    sources = pd.concat(
        [
            open_bank_sources(open_bank),
            settlement_residual_sources(settlement_detail_report, settings),
        ],
        ignore_index=True,
    )
    if sources.empty:
        return pd.DataFrame(columns=HYPOTHESIS_COLUMNS)

    docs = open_docs.copy()
    docs["date"] = pd.to_datetime(docs["date"], errors="coerce")
    sources["source_date"] = pd.to_datetime(sources["source_date"], errors="coerce")

    rows: list[dict[str, object]] = []
    for _, doc in docs.iterrows():
        doc_amount = float(doc.get("signed_amount") or doc.get("amount") or 0)
        if abs(doc_amount) < 0.005:
            continue
        candidates = sources[(sources["source_rest_amount"].astype(float) * doc_amount) > 0].copy()
        if candidates.empty:
            continue
        candidates["_day_diff"] = candidates["source_date"].apply(lambda value: days_between(value, doc.get("date")))
        candidates = candidates[candidates["_day_diff"] <= 180].copy()
        if candidates.empty:
            continue
        candidates["_amount_diff"] = candidates["source_rest_amount"].astype(float).apply(lambda value: abs(value - doc_amount))
        candidates["_text_score"] = candidates["source_context"].apply(lambda value: text_similarity(doc.get("text", ""), value))
        candidates["_counterparty_score"] = candidates["source_counterparty"].apply(lambda value: counterparty_overlap(doc.get("counterparty", ""), value))
        candidates = candidates[candidates.apply(lambda row: hypothesis_candidate_allowed(doc, row, settings), axis=1)].copy()
        if candidates.empty:
            continue
        candidates["_score"] = candidates.apply(lambda row: hypothesis_score(doc_amount, row, settings), axis=1)
        candidates = candidates.sort_values(["_score", "_amount_diff", "_day_diff"], ascending=[False, True, True]).head(5)
        for _, source in candidates.iterrows():
            if float(source["_score"]) < 0.32:
                continue
            rows.append(
                {
                    "hypothesis_source": source.get("hypothesis_source", ""),
                    "source_id": source.get("source_id", ""),
                    "source_date": source.get("source_date"),
                    "source_amount": round(float(source.get("source_amount") or 0), 2),
                    "source_rest_amount": round(float(source.get("source_rest_amount") or 0), 2),
                    "source_counterparty": source.get("source_counterparty", ""),
                    "source_context": source.get("source_context", ""),
                    "doc_ref": doc.get("doc_ref", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "doc_date": doc.get("date"),
                    "doc_amount": round(doc_amount, 2),
                    "doc_counterparty": doc.get("counterparty", ""),
                    "amount_diff": round(float(source["_amount_diff"]), 2),
                    "day_diff": int(source["_day_diff"]),
                    "counterparty_score": round(float(source["_counterparty_score"]), 3),
                    "text_score": round(float(source["_text_score"]), 3),
                    "score": round(float(source["_score"]), 3),
                    "recommendation": recommendation_for(float(source["_amount_diff"]), int(source["_day_diff"]), float(source["_text_score"]), float(source["_counterparty_score"])),
                    "caution": "Nicht automatisch bestaetigt; nur manuell pruefen.",
                }
            )
    if not rows:
        return pd.DataFrame(columns=HYPOTHESIS_COLUMNS)
    return pd.DataFrame(rows, columns=HYPOTHESIS_COLUMNS).sort_values(["score", "doc_ref"], ascending=[False, True])


def open_bank_sources(open_bank: pd.DataFrame) -> pd.DataFrame:
    if open_bank.empty:
        return pd.DataFrame(columns=HYPOTHESIS_COLUMNS[:7])
    rows: list[dict[str, object]] = []
    for _, tx in open_bank.iterrows():
        amount = float(tx.get("amount") or 0)
        if abs(amount) < 0.005:
            continue
        rows.append(
            {
                "hypothesis_source": "Offener FYRST-Umsatz",
                "source_id": tx.get("tx_id", ""),
                "source_date": tx.get("date"),
                "source_amount": amount,
                "source_rest_amount": amount,
                "source_counterparty": tx.get("counterparty", ""),
                "source_context": " ".join(str(value) for value in [tx.get("counterparty", ""), tx.get("description", ""), tx.get("text", "")] if str(value).strip()),
            }
        )
    return pd.DataFrame(rows)


def settlement_residual_sources(settlement_detail_report: pd.DataFrame, settings: MatchSettings) -> pd.DataFrame:
    if settlement_detail_report.empty:
        return pd.DataFrame(columns=HYPOTHESIS_COLUMNS[:7])
    work = settlement_detail_report.copy()
    group_key = work["tx_id"].astype(str) if "tx_id" in work else work.index.astype(str)
    rows: list[dict[str, object]] = []
    tolerance = max(settings.tolerance_cents / 100, 0.01)
    for _, group in work.groupby(group_key, sort=False):
        first = group.iloc[0]
        rest = to_float(first.get("Rest"))
        if abs(rest) <= tolerance:
            continue
        rows.append(
            {
                "hypothesis_source": "Restbetrag Sammelumsatz",
                "source_id": first.get("tx_id", ""),
                "source_date": first.get("FYRST-Datum"),
                "source_amount": to_float(first.get("FYRST-Betrag")),
                "source_rest_amount": rest,
                "source_counterparty": first.get("FYRST-Gegenpartei", ""),
                "source_context": " ".join(
                    str(value)
                    for value in [
                        first.get("Plattform", ""),
                        first.get("FYRST-Gegenpartei", ""),
                        first.get("Verwendungszweck", ""),
                        " ".join(str(value) for value in group.get("Detail", pd.Series(dtype=str)).dropna().head(8)),
                        " ".join(str(value) for value in group.get("Nachweiskette", pd.Series(dtype=str)).dropna().head(8)),
                    ]
                    if str(value).strip()
                ),
            }
        )
    return pd.DataFrame(rows)


def leftover_candidate_allowed(doc: pd.Series, tx: pd.Series, settings: MatchSettings) -> bool:
    amount_diff = float(tx["_amount_diff"])
    text_score = float(tx["_text_score"])
    counterparty_score = float(tx["_counterparty_score"])
    if obvious_vendor_mismatch(doc.get("counterparty", ""), tx.get("counterparty", "")):
        return False
    return amount_diff <= settings.tolerance_cents / 100 or counterparty_score >= 0.35 or text_score >= 0.18


def hypothesis_candidate_allowed(doc: pd.Series, source: pd.Series, settings: MatchSettings) -> bool:
    amount_diff = float(source["_amount_diff"])
    text_score = float(source["_text_score"])
    counterparty_score = float(source["_counterparty_score"])
    if obvious_vendor_mismatch(doc.get("counterparty", ""), source.get("source_counterparty", "")):
        return False
    doc_amount = abs(float(doc.get("signed_amount") or doc.get("amount") or 0))
    return (
        amount_diff <= max(settings.tolerance_cents / 100, doc_amount * 0.03, 1.0)
        or counterparty_score >= 0.35
        or text_score >= 0.18
    )


def leftover_score(doc_amount: float, tx: pd.Series, settings: MatchSettings) -> float:
    amount_diff = float(tx["_amount_diff"])
    day_diff = int(tx["_day_diff"])
    text_score = float(tx["_text_score"])
    counterparty_score = float(tx["_counterparty_score"])
    amount_scale = max(abs(doc_amount), abs(float(tx.get("amount") or 0)), 1)
    amount_score = 1 - min(1, amount_diff / max(amount_scale * 0.2, settings.tolerance_cents / 100, 0.01))
    date_score = 1 - min(1, day_diff / 90)
    return amount_score * 0.45 + date_score * 0.15 + counterparty_score * 0.25 + text_score * 0.15


def hypothesis_score(doc_amount: float, source: pd.Series, settings: MatchSettings) -> float:
    amount_diff = float(source["_amount_diff"])
    day_diff = int(source["_day_diff"])
    text_score = float(source["_text_score"])
    counterparty_score = float(source["_counterparty_score"])
    source_rest = float(source.get("source_rest_amount") or 0)
    amount_scale = max(abs(doc_amount), abs(source_rest), 1)
    amount_score = 1 - min(1, amount_diff / max(amount_scale * 0.2, settings.tolerance_cents / 100, 0.01))
    date_score = 1 - min(1, day_diff / 180)
    source_bonus = 0.06 if str(source.get("hypothesis_source", "")).startswith("Restbetrag") else 0.0
    return min(1, amount_score * 0.48 + date_score * 0.12 + counterparty_score * 0.22 + text_score * 0.18 + source_bonus)


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

def to_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(str(value).replace(",", "."))
    except Exception:
        return 0.0
