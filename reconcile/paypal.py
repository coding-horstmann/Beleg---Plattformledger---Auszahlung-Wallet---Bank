from __future__ import annotations

import pandas as pd

from .matching import MatchSettings, days_between, empty_links, empty_matches, make_link_row, make_match_row, text_similarity
from .parsers import clean_text, stable_id


PAYPAL_DOC_WINDOW_DAYS = 45
PAYPAL_DOC_TOLERANCE_CENTS = 15


def paypal_document_transactions(paypal: pd.DataFrame) -> pd.DataFrame:
    if paypal.empty:
        return pd.DataFrame()
    detail = paypal[paypal["category"].isin(["commercial", "conversion", "fee"])].copy()
    if detail.empty:
        return pd.DataFrame()
    detail["tx_id"] = detail["pp_id"]
    detail["source"] = "PayPal"
    detail["source_file"] = detail["source_file"]
    detail["source_row"] = detail["source_row"]
    detail["value_date"] = detail["date"]
    detail["raw_summary"] = detail["raw_summary"]
    return detail[
        [
            "tx_id",
            "source_file",
            "source",
            "source_row",
            "date",
            "value_date",
            "amount",
            "currency",
            "direction",
            "counterparty",
            "description",
            "platform",
            "text",
            "raw_summary",
        ]
    ].copy()


def match_docs_to_paypal(
    docs: pd.DataFrame,
    paypal: pd.DataFrame,
    settings: MatchSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paypal_bank_like = paypal_document_transactions(paypal)
    if docs.empty or paypal_bank_like.empty:
        return empty_matches(), empty_links()

    docs_work = docs.copy()
    docs_work["date"] = pd.to_datetime(docs_work["date"], errors="coerce")
    paypal_work = paypal_bank_like.copy()
    paypal_work["date"] = pd.to_datetime(paypal_work["date"], errors="coerce")
    paypal_lookup = paypal.set_index("pp_id") if "pp_id" in paypal.columns else pd.DataFrame()
    tolerance = max(settings.tolerance_cents, PAYPAL_DOC_TOLERANCE_CENTS) / 100
    window = max(settings.direct_window_days, PAYPAL_DOC_WINDOW_DAYS)

    candidates: list[tuple[float, dict[str, object], pd.Series, pd.Series]] = []
    used_docs: set[str] = set()
    used_paypal: set[str] = set()
    fx_matches, fx_links = paypal_fx_counterparty_batches(docs_work, paypal, settings, used_docs, used_paypal)

    option_work = paypal_option_rows(paypal_work, paypal_lookup, used_paypal)

    for _, doc in docs_work.iterrows():
        if str(doc.get("doc_id")) in used_docs:
            continue
        doc_amount = float(doc.get("signed_amount") or 0)
        if abs(doc_amount) < 0.005:
            continue
        target_cents = int(round(doc_amount * 100))
        possible = option_work[
            (~option_work["tx_id"].astype(str).isin(used_paypal))
            & (option_work["_match_amount_cents"].between(target_cents - int(round(tolerance * 100)), target_cents + int(round(tolerance * 100))))
            & ((option_work["_match_amount"].astype(float) * doc_amount) > 0)
        ].copy()
        for _, pp in possible.iterrows():
            if str(pp.get("tx_id")) in used_paypal:
                continue
            amount_kind = str(pp.get("_match_amount_kind") or "Netto")
            amount = float(pp.get("_match_amount") or 0)
            amount_diff = abs(amount - doc_amount)
            if amount_diff > tolerance:
                continue
            day_diff = days_between(pp.get("date"), doc.get("date"))
            if day_diff > window:
                continue
            sim = text_similarity(doc.get("text", ""), pp.get("text", ""))
            if day_diff > settings.direct_window_days and sim < 0.06 and not counterparty_overlap(doc, pp):
                continue
            amount_score = 1 - min(1, amount_diff / max(tolerance, 0.01))
            date_score = 1 - min(1, day_diff / max(window, 1))
            score = amount_score * 0.58 + date_score * 0.22 + sim * 0.20
            tx = pp.copy()
            tx["amount"] = round(amount, 2)
            payload = {
                "method": "paypal_detail",
                "confidence": round(max(0.60, min(0.98, score)), 3),
                "doc_sum": round(doc_amount, 2),
                "gap": round(amount - doc_amount, 2),
                "amount_diff": round(amount_diff, 2),
                "day_diff": day_diff,
                "text_score": sim,
                "explanation": (
                    f"PayPal-{amount_kind} passt innerhalb {round(tolerance, 2):.2f} EUR; "
                    f"Datumsabstand {day_diff} Tage; Textscore {sim:.2f}."
                ),
            }
            candidates.append((score, payload, doc, tx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    matches: list[dict[str, object]] = list(fx_matches)
    links: list[dict[str, object]] = list(fx_links)
    for _, payload, doc, tx in candidates:
        doc_id = str(doc.get("doc_id"))
        pp_id = str(tx.get("tx_id"))
        if doc_id in used_docs or pp_id in used_paypal:
            continue
        match_id = stable_id("match", "paypal_detail", doc_id, pp_id)
        row = make_match_row(match_id, tx, 1, payload)
        row["bank_source"] = "PayPal"
        matches.append(row)
        links.append(make_link_row(match_id, doc))
        used_docs.add(doc_id)
        used_paypal.add(pp_id)

    batch_matches, batch_links = paypal_batch_matches(docs_work, paypal_work, paypal_lookup, used_docs, used_paypal, settings)
    matches.extend(batch_matches)
    links.extend(batch_links)

    return (
        pd.DataFrame(matches) if matches else empty_matches(),
        pd.DataFrame(links) if links else empty_links(),
    )


def paypal_option_rows(paypal_work: pd.DataFrame, paypal_lookup: pd.DataFrame, used_paypal: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, pp in paypal_work.iterrows():
        if str(pp.get("tx_id")) in used_paypal:
            continue
        for amount_kind, amount in paypal_amount_options(pp, paypal_lookup):
            if abs(float(amount)) < 0.005:
                continue
            payload = pp.to_dict()
            payload["_match_amount_kind"] = amount_kind
            payload["_match_amount"] = float(amount)
            payload["_match_amount_cents"] = int(round(float(amount) * 100))
            rows.append(payload)
    if not rows:
        return pd.DataFrame(columns=list(paypal_work.columns) + ["_match_amount_kind", "_match_amount", "_match_amount_cents"])
    return pd.DataFrame(rows)


def paypal_fx_counterparty_batches(
    docs: pd.DataFrame,
    paypal: pd.DataFrame,
    settings: MatchSettings,
    used_docs: set[str],
    used_paypal: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if docs.empty or paypal.empty:
        return [], []
    commercial = paypal[paypal["category"].eq("commercial") & paypal["transaction_code"].astype(str).ne("")].copy()
    conversions = paypal[
        paypal["category"].eq("conversion")
        & paypal["currency"].astype(str).str.upper().eq("EUR")
        & (paypal["amount"].astype(float).abs() > 0.005)
        & paypal["related_code"].astype(str).ne("")
    ].copy()
    if commercial.empty or conversions.empty:
        return [], []

    commercial_by_code = {row["transaction_code"]: row for _, row in commercial.iterrows()}
    tolerance_cents = max(settings.tolerance_cents, 1000)
    window = max(settings.direct_window_days, 45)
    matches: list[dict[str, object]] = []
    links: list[dict[str, object]] = []

    for _, conv in conversions.sort_values("date").iterrows():
        conv_id = str(conv.get("pp_id"))
        if conv_id in used_paypal:
            continue
        source = commercial_by_code.get(str(conv.get("related_code") or ""))
        if source is None:
            continue
        counterparty = clean_text(source.get("counterparty"))
        if not counterparty:
            continue
        target = float(conv.get("amount") or 0)
        if target <= 0:
            continue
        candidates = docs[
            (~docs["doc_id"].astype(str).isin(used_docs))
            & (docs["date"].apply(lambda value: days_between(value, conv.get("date")) <= window))
            & (docs.apply(lambda doc: counterparty_overlap(doc, source), axis=1))
        ].copy()
        if len(candidates) < 2:
            continue
        candidates["_day_diff"] = candidates["date"].apply(lambda value: days_between(value, conv.get("date")))
        candidates["_amount_abs"] = candidates["signed_amount"].astype(float).abs()
        candidates = candidates.sort_values(["_day_diff", "_amount_abs"], ascending=[True, False]).head(18)
        selected_ids = subset_for_amount(candidates, target, tolerance_cents)
        if not selected_ids:
            continue
        selected = candidates[candidates["doc_id"].isin(selected_ids)].copy()
        doc_sum = round(float(selected["signed_amount"].sum()), 2)
        gap = round(target - doc_sum, 2)
        tx = conversion_tx_like(conv, source, target)
        match_id = stable_id("match", "paypal_fx_batch", conv_id, ",".join(sorted(selected_ids)))
        payload = {
            "method": "paypal_fx_batch",
            "confidence": 0.88 if abs(gap) <= 10 else 0.82,
            "doc_sum": doc_sum,
            "gap": gap,
            "amount_diff": abs(gap),
            "day_diff": round(float(selected["_day_diff"].mean()), 1),
            "text_score": text_similarity(" ".join(selected["text"].astype(str).head(8)), tx.get("text", "")),
            "explanation": f"PayPal-Fremdwaehrungspaket: {counterparty} wurde in EUR umgerechnet; mehrere Accountable-Belege ergeben netto {doc_sum:.2f} EUR zur EUR-Umrechnung {target:.2f} EUR.",
        }
        matches.append(make_match_row(match_id, tx, len(selected), payload))
        for _, doc in selected.iterrows():
            links.append(make_link_row(match_id, doc))
            used_docs.add(str(doc.get("doc_id")))
        used_paypal.add(conv_id)
    return matches, links


def conversion_tx_like(conversion: pd.Series, source: pd.Series, target: float) -> pd.Series:
    return pd.Series(
        {
            "tx_id": conversion.get("pp_id"),
            "source": "PayPal",
            "date": conversion.get("date"),
            "amount": round(float(target), 2),
            "counterparty": source.get("counterparty", ""),
            "description": f"{source.get('description', '')} / {conversion.get('description', '')}".strip(" /"),
            "text": " ".join(clean_text(value) for value in [source.get("counterparty"), source.get("description"), conversion.get("description"), source.get("transaction_code")] if clean_text(value)),
        }
    )


def paypal_batch_matches(
    docs: pd.DataFrame,
    paypal_work: pd.DataFrame,
    paypal_lookup: pd.DataFrame,
    used_docs: set[str],
    used_paypal: set[str],
    settings: MatchSettings,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tolerance_cents = max(settings.tolerance_cents, PAYPAL_DOC_TOLERANCE_CENTS)
    window = max(settings.direct_window_days, PAYPAL_DOC_WINDOW_DAYS)
    matches: list[dict[str, object]] = []
    links: list[dict[str, object]] = []

    for _, pp in paypal_work.sort_values("date").iterrows():
        pp_id = str(pp.get("tx_id"))
        if pp_id in used_paypal:
            continue
        for amount_kind, amount in paypal_amount_options(pp, paypal_lookup):
            if amount <= 0:
                continue
            candidates = docs[
                (~docs["doc_id"].astype(str).isin(used_docs))
                & (docs["signed_amount"].astype(float) > 0)
                & (docs["date"].apply(lambda value: days_between(value, pp.get("date")) <= window))
                & (docs.apply(lambda doc: counterparty_overlap(doc, pp), axis=1))
            ].copy()
            if len(candidates) < 2:
                continue
            candidates["_day_diff"] = candidates["date"].apply(lambda value: days_between(value, pp.get("date")))
            candidates["_score"] = candidates.apply(
                lambda doc: text_similarity(doc.get("text", ""), pp.get("text", "")) * 0.65
                + max(0, 1 - float(doc["_day_diff"]) / max(window, 1)) * 0.35,
                axis=1,
            )
            candidates = candidates.sort_values(["_score", "_day_diff"], ascending=[False, True]).head(14)
            selected_ids = subset_for_amount(candidates, amount, tolerance_cents)
            if not selected_ids:
                continue
            selected = candidates[candidates["doc_id"].isin(selected_ids)].copy()
            doc_sum = round(float(selected["signed_amount"].sum()), 2)
            tx = pp.copy()
            tx["amount"] = round(float(amount), 2)
            avg_day_diff = round(float(selected["_day_diff"].mean()), 1)
            sim = text_similarity(" ".join(selected["text"].astype(str).head(8)), pp.get("text", ""))
            match_id = stable_id("match", "paypal_batch", pp_id, ",".join(sorted(selected_ids)))
            payload = {
                "method": "paypal_batch",
                "confidence": 0.90,
                "doc_sum": doc_sum,
                "gap": round(float(amount) - doc_sum, 2),
                "amount_diff": abs(round(float(amount) - doc_sum, 2)),
                "day_diff": avg_day_diff,
                "text_score": sim,
                "explanation": f"Mehrere Accountable-Belege derselben PayPal-Gegenpartei ergeben zusammen den PayPal-{amount_kind}betrag.",
            }
            matches.append(make_match_row(match_id, tx, len(selected), payload))
            for _, doc in selected.iterrows():
                links.append(make_link_row(match_id, doc))
                used_docs.add(str(doc.get("doc_id")))
            used_paypal.add(pp_id)
            break
    return matches, links


def subset_for_amount(candidates: pd.DataFrame, target: float, tolerance_cents: int) -> set[str]:
    target_cents = int(round(float(target) * 100))
    items = [
        (str(row.get("doc_id")), int(round(float(row.get("signed_amount") or 0) * 100)))
        for _, row in candidates.iterrows()
    ]
    amounts = [amount for _, amount in items]
    allow_negative_offsets = any(amount < 0 for amount in amounts)
    suffix_positive = [0] * (len(items) + 1)
    suffix_negative = [0] * (len(items) + 1)
    for idx in range(len(items) - 1, -1, -1):
        amount = items[idx][1]
        suffix_positive[idx] = suffix_positive[idx + 1] + max(amount, 0)
        suffix_negative[idx] = suffix_negative[idx + 1] + min(amount, 0)

    states: dict[int, tuple[str, ...]] = {0: ()}
    best: tuple[int, tuple[str, ...]] | None = None
    for idx, (doc_id, amount) in enumerate(items):
        updates: dict[int, tuple[str, ...]] = {}
        for total, ids in states.items():
            new_total = total + amount
            if not allow_negative_offsets and new_total > target_cents + tolerance_cents:
                continue
            if allow_negative_offsets:
                min_reachable = new_total + suffix_negative[idx + 1]
                max_reachable = new_total + suffix_positive[idx + 1]
                if max_reachable < target_cents - tolerance_cents or min_reachable > target_cents + tolerance_cents:
                    continue
            if new_total not in states and new_total not in updates:
                updates[new_total] = ids + (doc_id,)
                if len(updates[new_total]) >= 2 and abs(new_total - target_cents) <= tolerance_cents:
                    if best is None or abs(new_total - target_cents) < abs(best[0] - target_cents):
                        best = (new_total, updates[new_total])
        states.update(updates)
        if len(states) > 4000:
            states = dict(
                sorted(
                    states.items(),
                    key=lambda item: (
                        abs(item[0] - target_cents),
                        -len(item[1]) if allow_negative_offsets else len(item[1]),
                    ),
                )[:4000]
            )
    return set(best[1]) if best else set()


def paypal_amount_options(pp: pd.Series, paypal_lookup: pd.DataFrame) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = [("Netto", float(pp.get("amount") or 0))]
    pp_id = pp.get("tx_id")
    if not paypal_lookup.empty and pp_id in paypal_lookup.index:
        original = paypal_lookup.loc[pp_id]
        gross = original.get("gross")
        if gross is not None and not pd.isna(gross):
            gross_value = float(gross)
            if abs(gross_value) >= 0.005 and abs(gross_value - values[0][1]) >= 0.01:
                values.append(("Brutto", gross_value))
    return values


def counterparty_overlap(doc: pd.Series, pp: pd.Series) -> bool:
    left = tokens(doc.get("counterparty", ""), doc.get("description", ""))
    right = tokens(pp.get("counterparty", ""), pp.get("description", ""))
    return bool(left and right and left.intersection(right))


def tokens(*values: object) -> set[str]:
    text = " ".join(clean_text(value).lower() for value in values if clean_text(value))
    return {token for token in text.replace("/", " ").replace(".", " ").split() if len(token) >= 4}


def match_paypal_transfers_to_bank(
    paypal: pd.DataFrame,
    bank: pd.DataFrame,
    settings: MatchSettings,
) -> pd.DataFrame:
    if paypal.empty or bank.empty:
        return empty_transfer_matches()

    transfers = paypal[(paypal["category"] == "bank_transfer") & (paypal["amount"].abs() > 0)].copy()
    if transfers.empty:
        return empty_transfer_matches()

    bank_candidates = bank[bank["text"].str.contains("paypal", case=False, na=False)].copy()
    if bank_candidates.empty:
        return empty_transfer_matches()
    bank_candidates["_amount_cents"] = (bank_candidates["amount"].astype(float) * 100).round().astype(int)
    bank_candidates["_date"] = pd.to_datetime(bank_candidates["date"], errors="coerce")

    rows: list[dict[str, object]] = []
    used_bank: set[str] = set()
    tolerance_cents = int(settings.tolerance_cents)
    tolerance = tolerance_cents / 100
    for _, pp in transfers.sort_values("date").iterrows():
        target = -float(pp["amount"])
        target_cents = int(round(target * 100))
        possible = bank_candidates[
            (~bank_candidates["tx_id"].isin(used_bank))
            & bank_candidates["_amount_cents"].between(target_cents - tolerance_cents, target_cents + tolerance_cents)
        ].copy()
        if possible.empty:
            continue
        possible["amount_diff"] = (possible["amount"] - target).abs()
        possible["day_diff"] = (possible["_date"] - pd.Timestamp(pp["date"])).dt.days.abs()
        possible = possible[
            (possible["amount_diff"] <= tolerance)
            & (possible["day_diff"] <= max(settings.direct_window_days, 10))
        ].copy()
        if possible.empty:
            continue
        possible["text_score"] = possible["text"].apply(lambda text: text_similarity(pp.get("text", ""), text))
        possible["score"] = (
            (1 - possible["amount_diff"] / max(tolerance, 0.01)).clip(lower=0) * 0.55
            + (1 - possible["day_diff"] / max(settings.direct_window_days, 10)).clip(lower=0) * 0.30
            + possible["text_score"] * 0.15
        )
        best = possible.sort_values("score", ascending=False).iloc[0]
        used_bank.add(best["tx_id"])
        confidence = round(max(0.65, min(0.98, float(best["score"]))), 3)
        rows.append(
            {
                "bridge_id": stable_id("ppbank", pp["pp_id"], best["tx_id"]),
                "pp_id": pp["pp_id"],
                "tx_id": best["tx_id"],
                "paypal_date": pp["date"],
                "paypal_amount": round(float(pp["amount"]), 2),
                "paypal_description": pp["description"],
                "paypal_counterparty": pp["counterparty"],
                "bank_date": best["date"],
                "bank_amount": round(float(best["amount"]), 2),
                "bank_source": best["source"],
                "bank_counterparty": best["counterparty"],
                "bank_description": best["description"],
                "amount_diff": round(float(best["amount_diff"]), 2),
                "day_diff": int(best["day_diff"]),
                "confidence": confidence,
                "explanation": "PayPal-Transfer mit Gegenbuchung auf Bankkonto; PayPal-Vorzeichen ist spiegelbildlich zur Bank.",
            }
        )
    return pd.DataFrame(rows) if rows else empty_transfer_matches()


def empty_transfer_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bridge_id",
            "pp_id",
            "tx_id",
            "paypal_date",
            "paypal_amount",
            "paypal_description",
            "paypal_counterparty",
            "bank_date",
            "bank_amount",
            "bank_source",
            "bank_counterparty",
            "bank_description",
            "amount_diff",
            "day_diff",
            "confidence",
            "explanation",
        ]
    )
