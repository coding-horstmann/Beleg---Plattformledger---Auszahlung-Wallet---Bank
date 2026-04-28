from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

import pandas as pd

from .parsers import stable_id


@dataclass
class MatchSettings:
    direct_window_days: int = 7
    batch_window_days: int = 14
    tolerance_cents: int = 5
    max_batch_candidates: int = 8
    max_fee_abs: float = 35.0
    max_fee_pct: float = 0.18
    max_subset_states: int = 5_000
    include_fee_documents: bool = True
    batch_outgoing: bool = False


PLATFORM_KEYWORDS = {
    "etsy",
    "ebay",
    "paypal",
    "shopify",
    "gelato",
    "printful",
    "redbubble",
    "teepublic",
    "spreadshirt",
    "sprd",
    "juniqe",
    "amazon",
    "stripe",
    "klarna",
    "sumup",
    "ionos",
}


def is_paypal_bank_row(row: pd.Series) -> bool:
    text = normalize_text(f"{row.get('counterparty', '')} {row.get('description', '')} {row.get('text', '')}")
    return "paypal" in text


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower()
    text = text.replace("ß", "ss")
    text = re.sub(r"[^a-z0-9äöü]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if len(token) > 2}


def text_similarity(a: object, b: object) -> float:
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return 0.0

    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        overlap = 0.0
    else:
        overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    if overlap == 0:
        seq = 0.0
    else:
        seq = SequenceMatcher(None, left[:240], right[:240]).ratio()
    return round((seq * 0.45) + (overlap * 0.55), 4)


def platform_overlap(a: object, b: object) -> bool:
    left = token_set(str(a))
    right = token_set(str(b))
    return bool((left & right) & PLATFORM_KEYWORDS)


def cents(amount: float) -> int:
    return int(round(float(amount) * 100))


def days_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    if pd.isna(a) or pd.isna(b):
        return 99999
    return abs(int((pd.Timestamp(a).normalize() - pd.Timestamp(b).normalize()).days))


def reconcile(docs: pd.DataFrame, bank: pd.DataFrame, settings: MatchSettings | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or MatchSettings()
    docs = docs.copy()
    bank = bank.copy()

    if docs.empty or bank.empty:
        return empty_matches(), empty_links()

    docs["date"] = pd.to_datetime(docs["date"], errors="coerce")
    bank["date"] = pd.to_datetime(bank["date"], errors="coerce")

    matches: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    linked_docs: set[str] = set()
    linked_txs: set[str] = set()

    direct_matches, direct_links = direct_one_to_one(docs, bank, linked_docs, linked_txs, settings)
    matches.extend(direct_matches)
    links.extend(direct_links)

    batch_matches, batch_links = batch_match(docs, bank, linked_docs, linked_txs, settings)
    matches.extend(batch_matches)
    links.extend(batch_links)

    match_frame = pd.DataFrame(matches) if matches else empty_matches()
    link_frame = pd.DataFrame(links) if links else empty_links()
    return match_frame, link_frame


def direct_one_to_one(
    docs: pd.DataFrame,
    bank: pd.DataFrame,
    linked_docs: set[str],
    linked_txs: set[str],
    settings: MatchSettings,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates: list[tuple[float, dict[str, object], pd.Series, pd.Series]] = []
    tol = settings.tolerance_cents / 100
    bank_indexed = bank.copy()
    bank_indexed["_amount_cents"] = bank_indexed["amount"].apply(cents)
    bank_indexed["_date"] = pd.to_datetime(bank_indexed["date"], errors="coerce")

    for _, doc in docs.iterrows():
        doc_amount = float(doc["signed_amount"])
        if math.isclose(doc_amount, 0, abs_tol=0.001):
            continue
        target_cents = cents(doc_amount)
        amount_mask = bank_indexed["_amount_cents"].between(target_cents - settings.tolerance_cents, target_cents + settings.tolerance_cents)
        direction_mask = (bank_indexed["amount"] * doc_amount) > 0
        possible = bank_indexed[amount_mask & direction_mask].copy()
        if possible.empty:
            continue
        possible["_day_diff"] = (possible["_date"] - pd.Timestamp(doc["date"])).dt.days.abs()
        possible = possible[possible["_day_diff"] <= settings.direct_window_days]
        for _, tx in possible.iterrows():
            if tx["tx_id"] in linked_txs:
                continue
            amount_diff = abs(float(tx["amount"]) - doc_amount)
            if amount_diff > tol:
                continue
            day_diff = int(tx["_day_diff"])
            sim = text_similarity(doc.get("text", ""), tx.get("text", ""))
            amount_score = 1 - min(1, amount_diff / max(tol, 0.01))
            date_score = 1 - min(1, day_diff / max(settings.direct_window_days, 1))
            score = (amount_score * 0.55) + (date_score * 0.25) + (sim * 0.20)
            payload = {
                "method": "direct",
                "confidence": round(max(0.55, min(0.99, score)), 3),
                "doc_sum": round(doc_amount, 2),
                "gap": round(float(tx["amount"]) - doc_amount, 2),
                "explanation": f"1:1-Betrag passt innerhalb {settings.tolerance_cents} Cent; Datumsabstand {day_diff} Tage; Textscore {sim:.2f}.",
                "amount_diff": round(amount_diff, 2),
                "day_diff": day_diff,
                "text_score": sim,
            }
            candidates.append((score, payload, doc, tx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    matches: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    for _, payload, doc, tx in candidates:
        if doc["doc_id"] in linked_docs or tx["tx_id"] in linked_txs:
            continue
        match_id = stable_id("match", "direct", doc["doc_id"], tx["tx_id"])
        matches.append(make_match_row(match_id, tx, 1, payload))
        links.append(make_link_row(match_id, doc))
        linked_docs.add(doc["doc_id"])
        linked_txs.add(tx["tx_id"])
    return matches, links


def batch_match(
    docs: pd.DataFrame,
    bank: pd.DataFrame,
    linked_docs: set[str],
    linked_txs: set[str],
    settings: MatchSettings,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matches: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    remaining_docs = docs[~docs["doc_id"].isin(linked_docs)].copy()
    remaining_bank = bank[~bank["tx_id"].isin(linked_txs)].copy()

    # PayPal rows in FYRST are account transfers. Their detail allocation must come
    # from the PayPal ledger, otherwise a transfer can accidentally carry unrelated
    # invoices with a plausible fee gap.
    remaining_bank = remaining_bank[~remaining_bank.apply(is_paypal_bank_row, axis=1)].copy()
    remaining_bank = remaining_bank[remaining_bank["amount"].abs() >= 0.01].sort_values("date")
    for _, tx in remaining_bank.iterrows():
        if tx["tx_id"] in linked_txs:
            continue
        if float(tx["amount"]) < 0 and not settings.batch_outgoing:
            continue
        candidates = candidate_docs_for_tx(tx, remaining_docs[~remaining_docs["doc_id"].isin(linked_docs)], settings)
        if len(candidates) < 2:
            continue
        if not batch_feasible(candidates, float(tx["amount"]), settings):
            continue

        exact = find_signed_subset(candidates, float(tx["amount"]), settings)
        method = "batch_exact"
        explanation = ""
        selected: pd.DataFrame | None = None
        gap = 0.0

        if exact:
            selected_ids, selected_sum = exact
            selected = candidates[candidates["doc_id"].isin(selected_ids)].copy()
            gap = round(float(tx["amount"]) - selected_sum, 2)
            explanation = f"Mehrere Belege ergeben zusammen {selected_sum:.2f} EUR und treffen den Bankbetrag innerhalb der Toleranz."
        elif float(tx["amount"]) > 0:
            net = find_net_payout_subset(candidates[candidates["signed_amount"] > 0], float(tx["amount"]), settings)
            if net:
                selected_ids, selected_sum, gap = net
                selected = candidates[candidates["doc_id"].isin(selected_ids)].copy()
                method = "batch_net_fee_gap"
                explanation = f"Positive Belege ergeben {selected_sum:.2f} EUR; Differenz zum Bankeingang {gap:.2f} EUR als moegliche Plattform-/Zahlungsgebuehr."

        if selected is None or selected.empty or len(selected) < 2:
            continue

        doc_sum = round(float(selected["signed_amount"].sum()), 2)
        sim = text_similarity(" ".join(selected["text"].astype(str).head(8)), tx.get("text", ""))
        avg_days = float(selected["date"].apply(lambda d: days_between(d, tx["date"])).mean())
        fee_cap = fee_cap_for(float(tx["amount"]), settings)
        gap_abs = abs(float(tx["amount"]) - doc_sum) if method == "batch_exact" else abs(gap)
        amount_score = 1 - min(1, gap_abs / max(fee_cap, settings.tolerance_cents / 100, 0.01))
        date_score = 1 - min(1, avg_days / max(settings.batch_window_days, 1))
        confidence = round(max(0.58, min(0.95, amount_score * 0.48 + date_score * 0.28 + sim * 0.24)), 3)

        payload = {
            "method": method,
            "confidence": confidence,
            "doc_sum": doc_sum if method == "batch_exact" else round(float(selected["signed_amount"].sum()), 2),
            "gap": round(float(tx["amount"]) - float(selected["signed_amount"].sum()), 2),
            "explanation": f"{explanation} Durchschnittlicher Datumsabstand {avg_days:.1f} Tage; Textscore {sim:.2f}.",
            "amount_diff": round(gap_abs, 2),
            "day_diff": round(avg_days, 1),
            "text_score": sim,
        }
        match_id = stable_id("match", method, tx["tx_id"], ",".join(sorted(selected["doc_id"].astype(str))))
        matches.append(make_match_row(match_id, tx, len(selected), payload))
        for _, doc in selected.iterrows():
            links.append(make_link_row(match_id, doc))
            linked_docs.add(doc["doc_id"])
        linked_txs.add(tx["tx_id"])

    return matches, links


def candidate_docs_for_tx(tx: pd.Series, docs: pd.DataFrame, settings: MatchSettings) -> pd.DataFrame:
    if docs.empty:
        return docs
    tx_amount = float(tx["amount"])
    tx_date = tx["date"]
    window = settings.batch_window_days
    candidates = docs[docs["date"].apply(lambda d: days_between(d, tx_date) <= window)].copy()
    if candidates.empty:
        return candidates

    if tx_amount > 0:
        positive = candidates[candidates["signed_amount"] > 0].copy()
        if settings.include_fee_documents:
            tx_text = tx.get("text", "")
            fee_docs = candidates[
                (candidates["signed_amount"] < 0)
                & candidates["text"].apply(lambda text: platform_overlap(text, tx_text) or text_similarity(text, tx_text) >= 0.18)
            ].copy()
            candidates = pd.concat([positive, fee_docs], ignore_index=True)
        else:
            candidates = positive
    else:
        candidates = candidates[candidates["signed_amount"] < 0].copy()

    if candidates.empty:
        return candidates

    target_abs = abs(tx_amount)
    cap = target_abs + fee_cap_for(tx_amount, settings)
    candidates = candidates[candidates["signed_amount"].abs() <= max(cap, 1.0)].copy()
    if candidates.empty:
        return candidates

    tx_text = tx.get("text", "")
    candidates["candidate_score"] = candidates.apply(
        lambda row: (
            text_similarity(row.get("text", ""), tx_text) * 0.52
            + (1 - min(1, days_between(row["date"], tx_date) / max(window, 1))) * 0.32
            + (1 - min(1, abs(abs(float(row["signed_amount"])) - target_abs) / max(target_abs, 1))) * 0.16
        ),
        axis=1,
    )
    candidates = candidates.sort_values(["candidate_score", "date"], ascending=[False, True])
    return candidates.head(settings.max_batch_candidates).copy()


def batch_feasible(candidates: pd.DataFrame, target: float, settings: MatchSettings) -> bool:
    tolerance = settings.tolerance_cents / 100
    if target > 0:
        positive_sum = float(candidates.loc[candidates["signed_amount"] > 0, "signed_amount"].sum())
        if positive_sum < target - tolerance:
            return False
        return True
    negative_abs = abs(float(candidates.loc[candidates["signed_amount"] < 0, "signed_amount"].sum()))
    return negative_abs >= abs(target) - tolerance


def fee_cap_for(amount: float, settings: MatchSettings) -> float:
    return max(float(settings.max_fee_abs), abs(float(amount)) * float(settings.max_fee_pct))


def find_signed_subset(candidates: pd.DataFrame, target: float, settings: MatchSettings) -> tuple[set[str], float] | None:
    target_cents = cents(target)
    tolerance = settings.tolerance_cents
    cap = cents(abs(target) + fee_cap_for(target, settings))
    items = [(row["doc_id"], cents(row["signed_amount"])) for _, row in candidates.iterrows()]
    states: dict[int, tuple[str, ...]] = {0: ()}
    best: tuple[int, tuple[str, ...]] | None = None

    for doc_id, amount in items:
        updates: dict[int, tuple[str, ...]] = {}
        for total, ids in states.items():
            new_total = total + amount
            if abs(new_total) > cap:
                continue
            if new_total not in states and new_total not in updates:
                updates[new_total] = ids + (doc_id,)
                diff = abs(new_total - target_cents)
                if diff <= tolerance and len(updates[new_total]) >= 2:
                    if best is None or diff < abs(best[0] - target_cents) or len(updates[new_total]) < len(best[1]):
                        best = (new_total, updates[new_total])
        states.update(updates)
        if len(states) > settings.max_subset_states:
            keep = sorted(states.items(), key=lambda item: abs(item[0] - target_cents))[: settings.max_subset_states]
            states = dict(keep)

    if best is None:
        for total, ids in states.items():
            if len(ids) >= 2 and abs(total - target_cents) <= tolerance:
                best = (total, ids)
                break

    if best is None:
        return None
    return set(best[1]), best[0] / 100


def find_net_payout_subset(candidates: pd.DataFrame, target: float, settings: MatchSettings) -> tuple[set[str], float, float] | None:
    if candidates.empty:
        return None
    target_cents = cents(target)
    fee_cap_cents = cents(fee_cap_for(target, settings))
    upper = target_cents + fee_cap_cents
    items = [(row["doc_id"], cents(abs(float(row["signed_amount"])))) for _, row in candidates.iterrows()]
    states: dict[int, tuple[str, ...]] = {0: ()}
    best: tuple[int, tuple[str, ...]] | None = None

    for doc_id, amount in items:
        updates: dict[int, tuple[str, ...]] = {}
        for total, ids in states.items():
            new_total = total + amount
            if new_total > upper:
                continue
            if new_total not in states and new_total not in updates:
                updates[new_total] = ids + (doc_id,)
                if target_cents <= new_total <= upper and len(updates[new_total]) >= 2:
                    if best is None or (new_total - target_cents) < (best[0] - target_cents):
                        best = (new_total, updates[new_total])
        states.update(updates)
        if len(states) > settings.max_subset_states:
            keep = sorted(
                states.items(),
                key=lambda item: abs(max(target_cents, min(item[0], upper)) - item[0]) + abs(item[0] - target_cents),
            )[: settings.max_subset_states]
            states = dict(keep)

    if best is None:
        return None
    selected_sum = best[0] / 100
    fee_gap = round(selected_sum - target, 2)
    return set(best[1]), selected_sum, fee_gap


def make_match_row(match_id: str, tx: pd.Series, doc_count: int, payload: dict[str, object]) -> dict[str, object]:
    return {
        "match_id": match_id,
        "method": payload["method"],
        "confidence": payload["confidence"],
        "status": "suggested",
        "tx_id": tx["tx_id"],
        "bank_source": tx.get("source", ""),
        "bank_date": tx["date"],
        "bank_amount": round(float(tx["amount"]), 2),
        "bank_counterparty": tx.get("counterparty", ""),
        "bank_description": tx.get("description", ""),
        "doc_count": doc_count,
        "doc_sum": payload["doc_sum"],
        "gap": payload["gap"],
        "amount_diff": payload["amount_diff"],
        "day_diff": payload["day_diff"],
        "text_score": payload["text_score"],
        "explanation": payload["explanation"],
    }


def make_link_row(match_id: str, doc: pd.Series) -> dict[str, object]:
    return {
        "match_id": match_id,
        "doc_id": doc["doc_id"],
        "doc_ref": doc.get("doc_ref", ""),
        "doc_type": doc.get("doc_type", ""),
        "doc_date": doc.get("date"),
        "doc_amount": doc.get("signed_amount"),
        "doc_counterparty": doc.get("counterparty", ""),
        "doc_description": doc.get("description", ""),
        "source_sheet": doc.get("source_sheet", ""),
        "source_row": doc.get("source_row", ""),
    }


def empty_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "method",
            "confidence",
            "status",
            "tx_id",
            "bank_source",
            "bank_date",
            "bank_amount",
            "bank_counterparty",
            "bank_description",
            "doc_count",
            "doc_sum",
            "gap",
            "amount_diff",
            "day_diff",
            "text_score",
            "explanation",
        ]
    )


def empty_links() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "doc_id",
            "doc_ref",
            "doc_type",
            "doc_date",
            "doc_amount",
            "doc_counterparty",
            "doc_description",
            "source_sheet",
            "source_row",
        ]
    )


def unmatched_docs(docs: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    if docs.empty:
        return docs
    linked = set(links["doc_id"]) if not links.empty else set()
    return docs[~docs["doc_id"].isin(linked)].copy()


def unmatched_bank(bank: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if bank.empty:
        return bank
    linked = set(matches["tx_id"]) if not matches.empty else set()
    return bank[~bank["tx_id"].isin(linked)].copy()


def top_candidates_for_doc(doc: pd.Series, bank: pd.DataFrame, limit: int = 5, window_days: int = 21) -> pd.DataFrame:
    if bank.empty:
        return bank
    signed_amount = float(doc["signed_amount"])
    candidates = bank[(bank["amount"] * signed_amount) > 0].copy()
    candidates["day_diff"] = candidates["date"].apply(lambda d: days_between(d, doc["date"]))
    candidates = candidates[candidates["day_diff"] <= window_days]
    if candidates.empty:
        return candidates
    candidates["amount_diff"] = (candidates["amount"] - signed_amount).abs()
    candidates["text_score"] = candidates["text"].apply(lambda text: text_similarity(doc.get("text", ""), text))
    candidates["candidate_score"] = (
        (1 - candidates["amount_diff"] / candidates["amount"].abs().clip(lower=1)).clip(lower=0) * 0.45
        + (1 - candidates["day_diff"] / max(window_days, 1)).clip(lower=0) * 0.30
        + candidates["text_score"] * 0.25
    )
    return candidates.sort_values("candidate_score", ascending=False).head(limit)
