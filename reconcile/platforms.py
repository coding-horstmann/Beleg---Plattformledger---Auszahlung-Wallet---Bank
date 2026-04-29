from __future__ import annotations

import io
import re
from itertools import combinations

import pandas as pd

from .matching import MatchSettings, direct_one_to_one, empty_links, empty_matches, make_link_row, make_match_row, text_similarity
from .parsers import ParseResult, clean_text, combine_fields, decode_text, parse_amount, stable_id


PLATFORM_COLUMNS = [
    "tx_id",
    "source_file",
    "source",
    "source_row",
    "date",
    "payout_date",
    "amount",
    "gross_amount",
    "fee_amount",
    "currency",
    "direction",
    "platform",
    "category",
    "counterparty",
    "order_id",
    "transaction_id",
    "payout_id",
    "description",
    "item_name",
    "quantity",
    "shop",
    "text",
    "raw_summary",
]

BANNED_TEXT_SNIPPETS = [
    "kunstgalerie" + ".niklas",
]


def has_real_text(value: object) -> bool:
    text = clean_text(value).lower()
    return bool(text and text not in {"--", "nan", "nat", "none"})


def parse_platform_file(name: str, data: bytes) -> ParseResult:
    text, encoding = decode_text(data)
    lines = text.splitlines()
    header_idx = find_platform_header(lines)
    if header_idx is None:
        return ParseResult(empty_platform_transactions(), [f"{name}: Plattform-CSV nicht erkannt."])

    header = lines[header_idx].lower()
    try:
        if "datum" in header and "art" in header and "gebühren & steuern" in header and "netto" in header:
            frame = normalize_etsy_statement(name, text, header_idx)
            label = "Etsy Monatsabrechnung"
        elif "sale date" in header and "order id" in header:
            frame = normalize_etsy_sold_items(name, text, header_idx)
            label = "Etsy SoldOrderItems"
        elif "payment id" in header and "gross amount" in header and "net amount" in header:
            frame = normalize_etsy_payment_sales(name, text, header_idx)
            label = "Etsy Payments Verkäufe"
        elif "date" in header and "amount" in header and "bank account ending digits" in header:
            frame = normalize_etsy_payouts(name, text, header_idx)
            label = "Etsy Payments Überweisungen"
        elif "financial status" in header and "lineitem name" in header:
            frame = normalize_shopify_orders(name, text, header_idx)
            label = "Shopify Orders"
        elif "bill #" in header and "charge category" in header and "store name" in header:
            frame = normalize_shopify_fees(name, text, header_idx)
            label = "Shopify Gebuehren"
        elif "reference id" in header and "total charge" in header and "product charge" in header:
            frame = normalize_gelato_statement(name, text, header_idx)
            label = "Gelato Kontoauszug"
        elif "printful-id" in header and "gesamtsumme" in header:
            frame = normalize_printful_orders(name, text, header_idx)
            label = "Printful Bestellungen"
        elif "aktion" in header and "zahlungsinstrument" in header and "betrag" in header:
            frame = normalize_printful_wallet(name, text, header_idx)
            label = "Printful Geldboerse"
        elif "datum der transaktionserstellung" in header and "auszahlung nr" in header:
            frame = normalize_ebay_transactions(name, text, header_idx)
            label = "eBay Transaktionsbericht"
        elif "verkaufsprotokollnummer" in header and "angebotstitel" in header and "gesamtbetrag" in header:
            frame = normalize_ebay_sales_report(name, text, header_idx)
            label = "eBay Verkaufsprotokoll"
        else:
            return ParseResult(empty_platform_transactions(), [f"{name}: Plattform-CSV-Header erkannt, aber nicht zugeordnet."])
    except Exception as exc:  # pragma: no cover - defensive UI path
        return ParseResult(empty_platform_transactions(), [f"{name}: Plattform-CSV konnte nicht gelesen werden ({exc})."])

    counts = frame["category"].value_counts().to_dict() if not frame.empty else {}
    return ParseResult(frame, [f"{name}: {len(frame)} Plattformzeilen importiert ({label}, {encoding}); Kategorien: {counts}."])


def assign_platform_payout_ids(platform_transactions: pd.DataFrame) -> pd.DataFrame:
    """Attach platform detail rows to their payout when exports provide the link indirectly.

    Etsy's payment sales export contains "Funds Available" dates, while the payout
    export contains the actual bank transfer date. The bank transfer often happens
    a few days later, so a same-day key would leave good evidence chains partial.

    eBay's sales report can contain the exact order amount without the payout id,
    while the settlement report contains the same order id with the payout id.
    """
    if platform_transactions.empty or "platform" not in platform_transactions.columns:
        return platform_transactions

    result = platform_transactions.copy()
    result["_date"] = pd.to_datetime(result["date"], errors="coerce")
    result["_payout_date"] = pd.to_datetime(result["payout_date"], errors="coerce")

    etsy_mask = result["platform"].astype(str).str.lower().eq("etsy")
    if etsy_mask.any():
        result["_etsy_account_key"] = result["source_file"].apply(etsy_account_key)
        payout_rows = result[
            etsy_mask
            & result["category"].astype(str).eq("payout")
            & result["_date"].notna()
            & result["payout_id"].apply(has_real_text)
        ][["_etsy_account_key", "_date", "payout_id"]].copy()

        if not payout_rows.empty:
            payout_rows = payout_rows.sort_values(["_etsy_account_key", "_date"])
            detail_idx = result[
                etsy_mask
                & result["category"].astype(str).isin(["order", "fee", "ledger_order", "ledger_fee", "ledger_tax", "ledger_adjustment"])
                & result["_payout_date"].notna()
            ].index

            for idx in detail_idx:
                account_key = result.at[idx, "_etsy_account_key"]
                available_date = result.at[idx, "_payout_date"]
                candidates = payout_rows[
                    (payout_rows["_etsy_account_key"] == account_key)
                    & (payout_rows["_date"] >= available_date)
                ]
                if candidates.empty:
                    continue
                result.at[idx, "payout_id"] = candidates.iloc[0]["payout_id"]

    result = assign_ebay_order_payout_ids(result)

    return result.drop(columns=["_etsy_account_key", "_date", "_payout_date"], errors="ignore")


def assign_ebay_order_payout_ids(platform_transactions: pd.DataFrame) -> pd.DataFrame:
    result = platform_transactions.copy()
    ebay_mask = result["platform"].astype(str).str.lower().eq("ebay")
    if not ebay_mask.any() or "order_id" not in result.columns:
        return result

    reference_rows = result[
        ebay_mask
        & result["category"].astype(str).eq("order")
        & (result["amount"].astype(float) > 0)
        & result["order_id"].apply(has_real_text)
        & result["payout_id"].apply(has_real_text)
    ].copy()
    if reference_rows.empty:
        return result

    refs: dict[str, dict[str, object]] = {}
    ambiguous: set[str] = set()
    for _, row in reference_rows.sort_values(["_date", "_payout_date"]).iterrows():
        order_id = clean_text(row.get("order_id"))
        payout_id = clean_text(row.get("payout_id"))
        if order_id in refs and refs[order_id]["payout_id"] != payout_id:
            ambiguous.add(order_id)
            continue
        refs[order_id] = {
            "payout_id": payout_id,
            "payout_date": row.get("payout_date"),
        }

    target_idx = result[
        ebay_mask
        & result["category"].astype(str).eq("order")
        & result["order_id"].apply(has_real_text)
        & ~result["payout_id"].apply(has_real_text)
    ].index
    for idx in target_idx:
        order_id = clean_text(result.at[idx, "order_id"])
        if order_id in ambiguous or order_id not in refs:
            continue
        result.at[idx, "payout_id"] = refs[order_id]["payout_id"]
        if pd.isna(result.at[idx, "payout_date"]):
            result.at[idx, "payout_date"] = refs[order_id]["payout_date"]
    return result


def deduplicate_platform_transactions(platform_transactions: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate platform exports uploaded under different file names.

    eBay and Etsy exports are often downloaded repeatedly with different local
    names. The generated tx_id includes the source file name, so we need a
    business-key de-duplication step after import.
    """
    if platform_transactions.empty:
        return platform_transactions
    result = platform_transactions.copy()
    for column in ["date", "payout_date"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    key_columns = [
        "source",
        "source_row",
        "platform",
        "category",
        "date",
        "payout_date",
        "amount",
        "gross_amount",
        "fee_amount",
        "currency",
        "counterparty",
        "order_id",
        "transaction_id",
        "payout_id",
        "description",
        "shop",
    ]
    available = [column for column in key_columns if column in result.columns]
    if not available:
        return result
    work = result.copy()
    for column in available:
        if pd.api.types.is_datetime64_any_dtype(work[column]):
            work[column] = work[column].dt.strftime("%Y-%m-%d").fillna("")
        elif column in {"amount", "gross_amount", "fee_amount"}:
            work[column] = pd.to_numeric(work[column], errors="coerce").round(2).astype(str).replace("nan", "")
        else:
            work[column] = work[column].apply(lambda value: clean_text(value).lower())
    keep_index = ~work.duplicated(subset=available, keep="first")
    return result.loc[keep_index].reset_index(drop=True)


def find_platform_header(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines[:120]):
        lower = line.lower()
        if "datum" in lower and "art" in lower and "gebühren & steuern" in lower and "netto" in lower:
            return idx
        if '"sale date"' in lower and "order id" in lower:
            return idx
        if "payment id" in lower and "gross amount" in lower and "net amount" in lower:
            return idx
        if "date" in lower and "amount" in lower and "bank account ending digits" in lower:
            return idx
        if "financial status" in lower and "lineitem name" in lower:
            return idx
        if "bill #" in lower and "charge category" in lower and "store name" in lower:
            return idx
        if "reference id" in lower and "total charge" in lower and "product charge" in lower:
            return idx
        if "printful-id" in lower and "gesamtsumme" in lower:
            return idx
        if "aktion" in lower and "zahlungsinstrument" in lower and "betrag" in lower:
            return idx
        if "datum der transaktionserstellung" in lower and "auszahlung nr" in lower:
            return idx
        if "verkaufsprotokollnummer" in lower and "angebotstitel" in lower and "gesamtbetrag" in lower:
            return idx
    return None


def normalize_etsy_sold_items(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=",", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    group_col = "Order ID" if "Order ID" in raw.columns else "Transaction ID"
    for order_id, group in raw.groupby(group_col, dropna=False):
        first = first_nonempty_row(group)
        item_total = sum_amounts(group.get("Item Total"))
        shipping = max_amount(group.get("Order Shipping"))
        sales_tax = max_amount(group.get("Order Sales Tax"))
        amount = round(item_total + shipping + sales_tax, 2)
        if abs(amount) < 0.01:
            continue
        date = parse_us_date(first.get("Date Paid"))
        if pd.isna(date):
            date = parse_us_date(first.get("Sale Date"))
        item_names = join_limited(group.get("Item Name", pd.Series(dtype=str)))
        buyer = clean_text(first.get("Buyer"))
        ship_name = clean_text(first.get("Ship Name"))
        transaction_ids = join_limited(group.get("Transaction ID", pd.Series(dtype=str)), limit=12)
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Etsy",
                source_row=int(group.index.min()) + 2,
                date=date,
                payout_date=pd.NaT,
                amount=amount,
                gross_amount=amount,
                fee_amount=None,
                currency=clean_text(first.get("Currency")) or "EUR",
                platform="etsy",
                category="order",
                counterparty=ship_name or buyer,
                order_id=clean_text(order_id),
                transaction_id=transaction_ids,
                payout_id="",
                description=item_names,
                item_name=item_names,
                quantity=sum_amounts(group.get("Quantity")),
                shop="",
                raw_summary=combine_fields(first, first.index[: min(len(first.index), 18)]),
            )
        )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def normalize_etsy_payment_sales(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=",", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    for idx, row in raw.iterrows():
        gross = parse_amount(row.get("Gross Amount"))
        net = parse_amount(row.get("Net Amount"))
        fee = parse_amount(row.get("Fees"))
        if gross is None or abs(float(gross)) < 0.01:
            continue
        order_date = parse_us_date(row.get("Order Date"))
        funds_available = parse_us_date(row.get("Funds Available"))
        payout_id = etsy_payout_key(file_name, funds_available) if not pd.isna(funds_available) else ""
        buyer = first_clean(row, ["Buyer", "Buyer Name", "Buyer Username"])
        payment_id = clean_text(row.get("Payment ID"))
        order_id = clean_text(row.get("Order ID"))
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Etsy",
                source_row=idx + 2,
                date=order_date if not pd.isna(order_date) else funds_available,
                payout_date=funds_available,
                amount=round(abs(float(gross)), 2),
                gross_amount=round(abs(float(gross)), 2),
                fee_amount=round(-abs(float(fee)), 2) if fee is not None else None,
                currency=clean_text(row.get("Currency")) or "EUR",
                platform="etsy",
                category="order",
                counterparty=buyer,
                order_id=order_id,
                transaction_id=payment_id,
                payout_id=payout_id,
                description=join_nonempty(["Etsy Payment", order_id, buyer]),
                item_name="",
                quantity=None,
                shop="",
                raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
            )
        )
        if fee is not None and abs(float(fee)) >= 0.01:
            rows.append(
                make_platform_row(
                    file_name=file_name,
                    source="Etsy",
                    source_row=idx + 2,
                    date=order_date if not pd.isna(order_date) else funds_available,
                    payout_date=funds_available,
                    amount=round(-abs(float(fee)), 2),
                    gross_amount=None,
                    fee_amount=round(-abs(float(fee)), 2),
                    currency=clean_text(row.get("Currency")) or "EUR",
                    platform="etsy",
                    category="fee",
                    counterparty="Etsy",
                    order_id=order_id,
                    transaction_id=payment_id,
                    payout_id=payout_id,
                    description=join_nonempty(["Etsy Gebühren", order_id]),
                    item_name="",
                    quantity=None,
                    shop="",
                    raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
                )
            )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def normalize_etsy_payouts(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=",", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    for idx, row in raw.iterrows():
        amount = parse_amount(row.get("Amount"))
        date = parse_english_date(row.get("Date"))
        if amount is None or pd.isna(date) or abs(float(amount)) < 0.01:
            continue
        status = clean_text(row.get("Status"))
        payout_id = etsy_payout_key(file_name, date)
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Etsy",
                source_row=idx + 2,
                date=date,
                payout_date=date,
                amount=round(-abs(float(amount)), 2),
                gross_amount=None,
                fee_amount=None,
                currency=clean_text(row.get("Currency")) or "EUR",
                platform="etsy",
                category="payout",
                counterparty="Etsy",
                order_id="",
                transaction_id="",
                payout_id=payout_id,
                description=join_nonempty(["Etsy Überweisung", status, row.get("Bank Account Ending Digits")]),
                item_name="",
                quantity=None,
                shop="",
                raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
            )
        )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def normalize_etsy_statement(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=",", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    fee_total = 0.0
    fee_dates: list[pd.Timestamp] = []
    account_key = etsy_account_key(file_name)
    for idx, row in raw.iterrows():
        date = parse_statement_date(row.get("Datum"))
        if pd.isna(date):
            continue
        art = clean_text(row.get("Art")).lower()
        title = clean_text(row.get("Titel"))
        info = clean_text(row.get("Info"))
        currency = clean_text(row.get("Währung")) or "EUR"
        gross = parse_amount(row.get("Betrag"))
        fee_or_tax = parse_amount(row.get("Gebühren & Steuern"))
        net = parse_amount(row.get("Netto"))
        source_row = idx + header_idx + 2

        if "überweisung" in art or "ueberweisung" in art:
            transfer_amount = parse_amount(title)
            if transfer_amount is None or abs(float(transfer_amount)) < 0.01:
                continue
            rows.append(
                make_platform_row(
                    file_name=file_name,
                    source="Etsy Statement",
                    source_row=source_row,
                    date=date,
                    payout_date=date,
                    amount=round(-abs(float(transfer_amount)), 2),
                    gross_amount=None,
                    fee_amount=None,
                    currency=currency,
                    platform="etsy",
                    category="payout",
                    counterparty="Etsy",
                    order_id="",
                    transaction_id=etsy_payout_key(file_name, date),
                    payout_id=etsy_payout_key(file_name, date),
                    description=title,
                    item_name="",
                    quantity=None,
                    shop=account_key,
                    raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
                )
            )
            continue

        amount = first_number(net, gross, fee_or_tax)
        if amount is None or abs(float(amount)) < 0.01:
            continue

        if art == "sale":
            category = "ledger_order"
            amount = abs(float(gross if gross is not None else amount))
            counterparty = ""
            order_id = extract_order_id(title)
            description = join_nonempty([title, info])
            gross_amount = amount
            fee_amount = None
        elif art in {"fee", "marketing"}:
            category = "ledger_fee"
            amount = -abs(float(amount))
            counterparty = "Etsy"
            order_id = extract_order_id(info) or extract_order_id(title)
            description = join_nonempty([title, info])
            gross_amount = None
            fee_amount = amount
            fee_total += amount
            fee_dates.append(date)
        elif art == "tax":
            category = "ledger_tax"
            amount = float(amount)
            counterparty = "Etsy"
            order_id = extract_order_id(info) or extract_order_id(title)
            description = join_nonempty([title, info])
            gross_amount = None
            fee_amount = amount
        else:
            category = "ledger_adjustment"
            amount = float(amount)
            counterparty = "Etsy"
            order_id = extract_order_id(info) or extract_order_id(title)
            description = join_nonempty([title, info, art])
            gross_amount = None
            fee_amount = amount if amount < 0 else None

        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Etsy Statement",
                source_row=source_row,
                date=date,
                payout_date=date,
                amount=round(float(amount), 2),
                gross_amount=round(float(gross_amount), 2) if gross_amount is not None else None,
                fee_amount=round(float(fee_amount), 2) if fee_amount is not None else None,
                currency=currency,
                platform="etsy",
                category=category,
                counterparty=counterparty,
                order_id=order_id,
                transaction_id=extract_statement_transaction_id(title, info),
                payout_id="",
                description=description,
                item_name=title,
                quantity=None,
                shop=account_key,
                raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
            )
        )

    if abs(fee_total) >= 0.01 and fee_dates:
        month_date = max(fee_dates) + pd.offsets.MonthEnd(0)
        month_key = pd.Timestamp(month_date).strftime("%Y-%m")
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Etsy Statement",
                source_row=len(raw) + header_idx + 2,
                date=month_date,
                payout_date=pd.NaT,
                amount=round(fee_total, 2),
                gross_amount=None,
                fee_amount=round(fee_total, 2),
                currency="EUR",
                platform="etsy",
                category="fee_month",
                counterparty="Etsy",
                order_id="",
                transaction_id=f"etsy-fees-{account_key}-{month_key}",
                payout_id="",
                description=f"Etsy Monatsgebuehren {account_key} {month_key}",
                item_name="",
                quantity=None,
                shop=account_key,
                raw_summary=f"Etsy statement fee month {account_key} {month_key}",
            )
        )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def normalize_shopify_orders(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=",", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    for order_id, group in raw.groupby("Name", dropna=False):
        first = first_nonempty_row(group)
        total = first_amount(group.get("Total"))
        if total is None:
            subtotal = first_amount(group.get("Subtotal")) or 0.0
            shipping = first_amount(group.get("Shipping")) or 0.0
            taxes = first_amount(group.get("Taxes")) or 0.0
            total = subtotal + shipping + taxes
        if total is None or abs(total) < 0.01:
            continue

        date = parse_isoish_date(first.get("Paid at"))
        if pd.isna(date):
            date = parse_isoish_date(first.get("Created at"))
        item_names = join_limited(group.get("Lineitem name", pd.Series(dtype=str)))
        counterparty = first_clean(first, ["Billing Name", "Shipping Name", "Email"])
        payment_id = first_clean(first, ["Payment ID", "Payment Reference", "Payment References"])
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Shopify",
                source_row=int(group.index.min()) + 2,
                date=date,
                payout_date=pd.NaT,
                amount=round(float(total), 2),
                gross_amount=round(float(total), 2),
                fee_amount=None,
                currency=clean_text(first.get("Currency")) or "EUR",
                platform="shopify",
                category="order",
                counterparty=counterparty,
                order_id=clean_text(order_id),
                transaction_id=payment_id,
                payout_id="",
                description=item_names,
                item_name=item_names,
                quantity=sum_amounts(group.get("Lineitem quantity")),
                shop="",
                raw_summary=combine_fields(first, first.index[: min(len(first.index), 18)]),
            )
        )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def normalize_shopify_fees(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=",", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    for idx, row in raw.iterrows():
        amount = parse_amount(row.get("Amount"))
        if amount is None or abs(float(amount)) < 0.01:
            continue
        date = parse_isoish_date(row.get("Date"))
        if pd.isna(date):
            date = parse_isoish_date(row.get("Start of billing cycle"))
        if pd.isna(date):
            continue
        store = first_clean(row, ["Store Name", ".myshopify.com URL"])
        category = first_clean(row, ["Charge category"])
        description = join_nonempty([category, row.get("Description"), row.get("App"), store])
        bill_id = clean_text(row.get("Bill #"))
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Shopify Billing",
                source_row=idx + header_idx + 2,
                date=date,
                payout_date=pd.NaT,
                amount=round(-abs(float(amount)), 2),
                gross_amount=None,
                fee_amount=round(-abs(float(amount)), 2),
                currency=clean_text(row.get("Currency")) or "EUR",
                platform="shopify",
                category="fee",
                counterparty="Shopify",
                order_id=clean_text(row.get("Order")),
                transaction_id=bill_id,
                payout_id="",
                description=description,
                item_name=description,
                quantity=None,
                shop=store,
                raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
            )
        )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def normalize_gelato_statement(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=",", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    for idx, row in raw.iterrows():
        total = parse_amount(row.get("Total Charge"))
        if total is None or abs(float(total)) < 0.01:
            continue
        date = parse_isoish_date(row.get("Date"))
        if pd.isna(date):
            continue
        ref_id = clean_text(row.get("Reference ID"))
        description = join_nonempty(
            [
                "Gelato charge",
                ref_id,
                row.get("VAT Country"),
                row.get("Printhouse Country"),
                row.get("Product Charge"),
                row.get("Shipping Charge"),
            ]
        )
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Gelato",
                source_row=idx + header_idx + 2,
                date=date,
                payout_date=pd.NaT,
                amount=round(-abs(float(total)), 2),
                gross_amount=None,
                fee_amount=None,
                currency=clean_text(row.get("Currency")) or "EUR",
                platform="gelato",
                category="charge",
                counterparty="Gelato",
                order_id=ref_id,
                transaction_id=ref_id,
                payout_id="",
                description=description,
                item_name="",
                quantity=None,
                shop="",
                raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
            )
        )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def normalize_printful_orders(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=",", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    for idx, row in raw.iterrows():
        total = parse_amount(row.get("Gesamtsumme"))
        if total is None or abs(float(total)) < 0.01:
            continue
        date = parse_english_date(row.get("Datum"))
        if pd.isna(date):
            continue
        order_label = clean_text(row.get("Bestellung"))
        order_id = extract_order_id(order_label) or order_label
        printful_id = clean_text(row.get("Printful-ID"))
        description = join_nonempty(
            [
                "Printful order",
                order_label,
                row.get("Status"),
                row.get("Versand aus"),
                row.get("Geliefert an"),
                row.get("Produkte"),
                row.get("Versand"),
            ]
        )
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Printful",
                source_row=idx + header_idx + 2,
                date=date,
                payout_date=pd.NaT,
                amount=round(-abs(float(total)), 2),
                gross_amount=None,
                fee_amount=None,
                currency="EUR",
                platform="printful",
                category="charge",
                counterparty="Printful",
                order_id=order_id,
                transaction_id=printful_id,
                payout_id="",
                description=description,
                item_name=clean_text(row.get("Produkte")),
                quantity=None,
                shop="",
                raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
            )
        )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def normalize_printful_wallet(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=",", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    for idx, row in raw.iterrows():
        amount = parse_amount(row.get("Betrag"))
        if amount is None or abs(float(amount)) < 0.01:
            continue
        date = parse_english_date(row.get("Datum"))
        if pd.isna(date):
            continue
        description = join_nonempty(["Printful wallet", row.get("Aktion"), row.get("Zahlungsinstrument")])
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="Printful Wallet",
                source_row=idx + header_idx + 2,
                date=date,
                payout_date=pd.NaT,
                amount=round(-abs(float(amount)), 2),
                gross_amount=None,
                fee_amount=None,
                currency="EUR",
                platform="printful",
                category="wallet_deposit",
                counterparty="Printful",
                order_id="",
                transaction_id=stable_id("printful-wallet", file_name, idx, date, amount),
                payout_id="",
                description=description,
                item_name="",
                quantity=None,
                shop="",
                raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
            )
        )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def normalize_ebay_transactions(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=";", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    fee_cols = [
        "Fixer Anteil der Verkaufsprovision",
        "Variabler Anteil der Verkaufsprovision",
        "Gebühr für gesetzliche Betriebskosten",
        "Gebühr für sehr hohe Quote an „nicht wie beschriebenen Artikeln“",
        "Gebühr für unterdurchschnittlichen Servicestatus",
        "Internationale Gebühr",
    ]
    for idx, row in raw.iterrows():
        tx_type = clean_text(row.get("Typ"))
        category = ebay_category(tx_type)
        date = parse_german_date(row.get("Datum der Transaktionserstellung"))
        payout_date = parse_german_date(row.get("Auszahlungsdatum"))
        payout_id = clean_text(row.get("Auszahlung Nr.")).replace("--", "")
        currency = clean_text(row.get("Auszahlungswährung")) or clean_text(row.get("Transaktionswährung")) or "EUR"
        counterparty = first_clean(row, ["Name des Käufers", "Nutzername des Käufers", "Auszahlungsmethode"])
        description = first_clean(row, ["Angebotstitel", "Beschreibung", "Typ"])
        order_id = clean_text(row.get("Bestellnummer")).replace("--", "")
        transaction_id = clean_text(row.get("Transaktionsnummer")).replace("--", "")
        net_amount = parse_amount(row.get("Betrag abzügl. Kosten"))
        gross_amount = parse_amount(row.get("Transaktionsbetrag (inkl. Kosten)"))

        if category == "order":
            gross = gross_amount if gross_amount is not None else net_amount
            if gross is not None and abs(gross) >= 0.01:
                rows.append(
                    make_platform_row(
                        file_name=file_name,
                        source="eBay",
                        source_row=idx + 2,
                        date=date,
                        payout_date=payout_date,
                        amount=abs(round(float(gross), 2)),
                        gross_amount=abs(round(float(gross), 2)),
                        fee_amount=sum_fee_columns(row, fee_cols),
                        currency=currency,
                        platform="ebay",
                        category="order",
                        counterparty=counterparty,
                        order_id=order_id,
                        transaction_id=transaction_id,
                        payout_id=payout_id,
                        description=description,
                        item_name=description,
                        quantity=parse_amount(row.get("Stückzahl")),
                        shop="",
                        raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
                    )
                )
            fee_total = sum_fee_columns(row, fee_cols)
            if fee_total is not None and abs(fee_total) >= 0.01:
                fee_value = round(-abs(float(fee_total)), 2)
                rows.append(
                    make_platform_row(
                        file_name=file_name,
                        source="eBay",
                        source_row=idx + 2,
                        date=date,
                        payout_date=payout_date,
                        amount=fee_value,
                        gross_amount=None,
                        fee_amount=fee_value,
                        currency=currency,
                        platform="ebay",
                        category="fee",
                        counterparty="eBay",
                        order_id=order_id,
                        transaction_id=transaction_id,
                        payout_id=payout_id,
                        description=f"eBay Gebühren zu Bestellung {order_id}".strip(),
                        item_name=description,
                        quantity=None,
                        shop="",
                        raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
                    )
                )
            continue

        amount = net_amount if net_amount is not None else gross_amount
        if amount is None or abs(amount) < 0.01:
            continue
        amount_value = round(float(amount), 2)
        if category in {"fee", "charge", "refund", "payout"}:
            amount_value = round(-abs(amount_value), 2)
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="eBay",
                source_row=idx + 2,
                date=date,
                payout_date=payout_date,
                amount=amount_value,
                gross_amount=round(float(gross_amount), 2) if gross_amount is not None else None,
                fee_amount=None,
                currency=currency,
                platform="ebay",
                category=category,
                counterparty=counterparty or "eBay",
                order_id=order_id,
                transaction_id=transaction_id,
                payout_id=payout_id,
                description=description,
                item_name=description,
                quantity=parse_amount(row.get("Stückzahl")),
                shop="",
                raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
            )
        )
    if not rows:
        return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)
    frame = pd.DataFrame(rows, columns=PLATFORM_COLUMNS)
    payout_rows = ebay_synthetic_payout_rows(file_name, frame, len(raw) + header_idx + 2)
    if payout_rows:
        frame = pd.concat([frame, pd.DataFrame(payout_rows, columns=PLATFORM_COLUMNS)], ignore_index=True)
    monthly_rows = ebay_monthly_fee_rows(file_name, frame, len(raw) + header_idx + 2 + len(payout_rows))
    if monthly_rows:
        frame = pd.concat([frame, pd.DataFrame(monthly_rows, columns=PLATFORM_COLUMNS)], ignore_index=True)
    return frame


def normalize_ebay_sales_report(file_name: str, text: str, header_idx: int) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), sep=";", skiprows=header_idx, dtype=str)
    raw.columns = [clean_text(col).strip('"') for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return empty_platform_transactions()

    rows: list[dict[str, object]] = []
    for idx, row in raw.iterrows():
        order_id = first_clean(row, ["Bestellnummer"])
        title = first_clean(row, ["Angebotstitel"])
        if not order_id and not title:
            continue
        total = parse_amount(first_clean(row, ["Gesamtbetrag", "Gesamtbetrag inkl. der von eBay eingezogenen Steuer und Gebühren", "Verkauft für"]))
        if total is None or abs(float(total)) < 0.01:
            continue
        payment_date = parse_german_date(first_clean(row, ["Zahlungsdatum", "Verkauft am"]))
        buyer = first_clean(row, ["Name des Käufers", "Nutzername des Käufers", "Name des Empfängers"])
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="eBay",
                source_row=idx + 2,
                date=payment_date,
                payout_date=pd.NaT,
                amount=round(abs(float(total)), 2),
                gross_amount=round(abs(float(total)), 2),
                fee_amount=None,
                currency="EUR",
                platform="ebay",
                category="order",
                counterparty=buyer,
                order_id=order_id,
                transaction_id=first_clean(row, ["Transaktionsnummer", "PayPal-Transaktionsnummer", "Verkaufsprotokollnummer"]),
                payout_id="",
                description=title,
                item_name=title,
                quantity=parse_amount(row.get("Anzahl")),
                shop="",
                raw_summary=combine_fields(row, row.index[: min(len(row.index), 18)]),
            )
        )
    return pd.DataFrame(rows, columns=PLATFORM_COLUMNS)


def match_docs_to_platform(
    docs: pd.DataFrame,
    platform_transactions: pd.DataFrame,
    settings: MatchSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if docs.empty or platform_transactions.empty:
        return empty_matches(), empty_links()

    details = platform_transactions[platform_transactions["category"].isin(["order", "fee", "fee_month", "charge", "refund", "adjustment"])].copy()
    details = details[details["amount"].abs() > 0.005].copy()
    if details.empty:
        return empty_matches(), empty_links()
    details["_has_payout"] = details["payout_id"].astype(str).ne("").astype(int)
    details["_source_rank"] = details["source"].astype(str).isin(["eBay", "Etsy", "Etsy Statement"]).astype(int)
    details = details.sort_values(["_has_payout", "_source_rank"], ascending=[False, False]).drop(
        columns=["_has_payout", "_source_rank"],
        errors="ignore",
    )

    bank_like = details.rename(columns={"payout_date": "value_date"}).copy()
    bank_like["source"] = "Plattform"
    relaxed_settings = MatchSettings(
        direct_window_days=max(settings.direct_window_days, 21),
        batch_window_days=settings.batch_window_days,
        tolerance_cents=max(settings.tolerance_cents, 15),
        max_batch_candidates=settings.max_batch_candidates,
        max_fee_abs=settings.max_fee_abs,
        max_fee_pct=settings.max_fee_pct,
        max_subset_states=settings.max_subset_states,
        include_fee_documents=settings.include_fee_documents,
        batch_outgoing=settings.batch_outgoing,
    )
    linked_docs: set[str] = set()
    linked_txs: set[str] = set()
    pre_matches, pre_links = platform_counterparty_one_to_one(docs, bank_like, linked_docs, linked_txs, relaxed_settings)
    split_matches, split_links = platform_order_split_merge_support(docs, bank_like, linked_docs, linked_txs, relaxed_settings)
    direct_matches, direct_links = platform_direct_one_to_one(docs, bank_like, linked_docs, linked_txs, relaxed_settings)
    monthly_fee_matches, monthly_fee_links = platform_monthly_fee_support(docs, bank_like, linked_docs, linked_txs)
    matches = pre_matches + split_matches + direct_matches + monthly_fee_matches
    links = pre_links + split_links + direct_links + monthly_fee_links
    if not matches:
        return empty_matches(), empty_links()

    match_frame = pd.DataFrame(matches)
    link_frame = pd.DataFrame(links)
    detail_index = details.set_index("tx_id")
    doc_index = docs.set_index("doc_id")

    keep_match_ids: set[str] = set()
    for _, link in link_frame.iterrows():
        match = match_frame[match_frame["match_id"] == link["match_id"]].iloc[0]
        tx = detail_index.loc[match["tx_id"]]
        doc = doc_index.loc[link["doc_id"]]
        if not platform_match_allowed(doc, tx):
            continue
        doc_platform = clean_text(doc.get("platform"))
        same_platform = bool(doc_platform and doc_platform == clean_text(tx.get("platform")))
        useful_text = float(match.get("text_score") or 0) >= 0.08
        amount_diff_raw = match.get("amount_diff")
        day_diff_raw = match.get("day_diff")
        amount_diff_value = 999.0 if amount_diff_raw is None or pd.isna(amount_diff_raw) else float(amount_diff_raw)
        day_diff_value = 999 if day_diff_raw is None or pd.isna(day_diff_raw) else int(day_diff_raw)
        strong_amount_date = amount_diff_value <= relaxed_settings.tolerance_cents / 100 and day_diff_value <= 7
        strong_order_text_amount_match = (
            str(tx.get("category") or "") == "order"
            and strong_amount_date
            and useful_text
            and float(match.get("text_score") or 0) >= 0.25
        )
        if str(tx.get("category") or "") == "order" and clean_text(tx.get("counterparty")) and clean_text(doc.get("counterparty")):
            if counterparty_match_score(doc.get("counterparty"), tx.get("counterparty")) < 0.4 and not strong_order_text_amount_match:
                continue
        if same_platform or useful_text or strong_amount_date:
            keep_match_ids.add(link["match_id"])

    match_frame = match_frame[match_frame["match_id"].isin(keep_match_ids)].copy()
    link_frame = link_frame[link_frame["match_id"].isin(keep_match_ids)].copy()
    if match_frame.empty:
        return empty_matches(), empty_links()

    match_frame["method"] = "platform_detail"
    match_frame["platform"] = match_frame["tx_id"].map(detail_index["platform"])
    match_frame["platform_category"] = match_frame["tx_id"].map(detail_index["category"])
    match_frame["order_id"] = match_frame["tx_id"].map(detail_index["order_id"])
    match_frame["payout_id"] = match_frame["tx_id"].map(detail_index["payout_id"])
    match_frame["bank_source"] = match_frame["platform"].apply(lambda value: f"Plattform: {value}")
    match_frame["explanation"] = match_frame["explanation"].str.replace(
        "1:1-Betrag",
        "Plattform-Detailbetrag",
        regex=False,
    )
    return match_frame, link_frame


def platform_direct_one_to_one(
    docs: pd.DataFrame,
    platform_like: pd.DataFrame,
    linked_docs: set[str],
    linked_txs: set[str],
    settings: MatchSettings,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matches: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    if docs.empty or platform_like.empty:
        return matches, links

    for platform, tx_group in platform_like.groupby(platform_like["platform"].astype(str).str.lower(), dropna=False):
        platform_key = clean_text(platform).lower()
        doc_group = docs[docs.apply(lambda doc: doc_platform_can_match(doc, platform_key), axis=1)].copy()
        if doc_group.empty:
            continue
        group_matches, group_links = direct_one_to_one(doc_group, tx_group.copy(), linked_docs, linked_txs, settings)
        matches.extend(group_matches)
        links.extend(group_links)
    return matches, links


def platform_order_split_merge_support(
    docs: pd.DataFrame,
    platform_like: pd.DataFrame,
    linked_docs: set[str],
    linked_txs: set[str],
    settings: MatchSettings,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if docs.empty or platform_like.empty:
        return [], []

    docs_work = docs.copy()
    docs_work["date"] = pd.to_datetime(docs_work["date"], errors="coerce")
    tx_work = platform_like[
        platform_like["category"].astype(str).eq("order")
        & (platform_like["amount"].astype(float).abs() > 0.005)
    ].copy()
    tx_work["date"] = pd.to_datetime(tx_work["date"], errors="coerce")
    if tx_work.empty:
        return [], []

    matches: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    tolerance = max(settings.tolerance_cents, 15) / 100
    window_days = max(settings.direct_window_days, 21)

    # One marketplace order can become multiple Accountable invoice lines.
    for _, tx in tx_work.sort_values("date").iterrows():
        tx_id = str(tx.get("tx_id"))
        if tx_id in linked_txs:
            continue
        tx_amount = float(tx.get("amount") or 0)
        candidates = docs_work[
            (~docs_work["doc_id"].astype(str).isin(linked_docs))
            & (docs_work["signed_amount"].astype(float) * tx_amount > 0)
            & docs_work.apply(lambda doc: platform_match_allowed(doc, tx), axis=1)
            & docs_work["date"].apply(lambda value: date_distance(value, tx.get("date")) <= window_days)
        ].copy()
        candidates = order_counterparty_candidates(candidates, tx)
        subset = find_amount_subset(candidates, tx_amount, tolerance, max_items=4)
        if subset is None or len(subset) < 2:
            continue
        selected = candidates[candidates["doc_id"].astype(str).isin(subset)].copy()
        score = split_merge_score(selected, tx, tx_amount, tolerance, window_days)
        match_id = stable_id("match", "platform_order_split_docs", tx_id, ",".join(sorted(subset)))
        payload = {
            "method": "platform_detail",
            "confidence": round(max(0.76, min(0.96, score)), 3),
            "doc_sum": round(float(selected["signed_amount"].sum()), 2),
            "gap": round(tx_amount - float(selected["signed_amount"].sum()), 2),
            "amount_diff": round(abs(tx_amount - float(selected["signed_amount"].sum())), 2),
            "day_diff": round(float(selected["date"].apply(lambda value: date_distance(value, tx.get("date"))).mean()), 1),
            "text_score": text_similarity(" ".join(selected["text"].astype(str).head(4)), tx.get("text", "")),
            "explanation": "Eine Plattformbestellung passt rechnerisch zu mehreren Accountable-Belegzeilen.",
        }
        matches.append(make_match_row(match_id, tx, len(selected), payload))
        for _, doc in selected.iterrows():
            links.append(make_link_row(match_id, doc))
            linked_docs.add(str(doc.get("doc_id")))
        linked_txs.add(tx_id)

    # One Accountable invoice can cover multiple marketplace order rows.
    for _, doc in docs_work.sort_values("date").iterrows():
        doc_id = str(doc.get("doc_id"))
        if doc_id in linked_docs:
            continue
        doc_amount = float(doc.get("signed_amount") or 0)
        if abs(doc_amount) < 0.005:
            continue
        candidates = tx_work[
            (~tx_work["tx_id"].astype(str).isin(linked_txs))
            & (tx_work["amount"].astype(float) * doc_amount > 0)
            & tx_work.apply(lambda tx: platform_match_allowed(doc, tx), axis=1)
            & tx_work["date"].apply(lambda value: date_distance(value, doc.get("date")) <= window_days)
        ].copy()
        candidates = doc_counterparty_tx_candidates(candidates, doc)
        subset = find_tx_amount_subset(candidates, doc_amount, tolerance, max_items=4)
        if subset is None or len(subset) < 2:
            continue
        selected = candidates[candidates["tx_id"].astype(str).isin(subset)].copy()
        total = float(selected["amount"].sum())
        for _, tx in selected.iterrows():
            match_id = stable_id("match", "platform_doc_split_orders", doc_id, tx.get("tx_id"))
            payload = {
                "method": "platform_detail",
                "confidence": round(max(0.74, min(0.94, split_merge_score(pd.DataFrame([doc]), tx, float(tx.get("amount") or 0), tolerance, window_days))), 3),
                "doc_sum": round(doc_amount, 2),
                "gap": round(float(tx.get("amount") or 0) - doc_amount, 2),
                "amount_diff": round(abs(total - doc_amount), 2),
                "day_diff": date_distance(doc.get("date"), tx.get("date")),
                "text_score": text_similarity(doc.get("text", ""), tx.get("text", "")),
                "explanation": "Ein Accountable-Beleg passt rechnerisch zu mehreren Plattformbestellungen.",
            }
            matches.append(make_match_row(match_id, tx, 1, payload))
            links.append(make_link_row(match_id, doc))
            linked_txs.add(str(tx.get("tx_id")))
        linked_docs.add(doc_id)

    return matches, links


def platform_match_allowed(doc: pd.Series, tx: pd.Series) -> bool:
    return doc_platform_can_match(doc, clean_text(tx.get("platform")).lower())


def doc_platform_can_match(doc: pd.Series, platform_key: str) -> bool:
    doc_platform = clean_text(doc.get("platform")).lower()
    if not doc_platform:
        return True
    return doc_platform == clean_text(platform_key).lower()


def order_counterparty_candidates(candidates: pd.DataFrame, tx: pd.Series) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    tx_counterparty = tx.get("counterparty", "")
    tx_text = tx.get("text", "")
    order_id = clean_text(tx.get("order_id"))
    work = candidates.copy()
    work["_counterparty_score"] = work["counterparty"].apply(lambda value: counterparty_match_score(value, tx_counterparty))
    work["_text_score"] = work["text"].apply(lambda value: text_similarity(value, tx_text))
    work["_order_id_hit"] = work["text"].apply(lambda value: bool(order_id and order_id in clean_text(value)))
    work = work[(work["_counterparty_score"] >= 0.55) | (work["_text_score"] >= 0.22) | work["_order_id_hit"]].copy()
    if work.empty:
        return work
    return work.sort_values(["_counterparty_score", "_text_score", "date"], ascending=[False, False, True]).head(10)


def doc_counterparty_tx_candidates(candidates: pd.DataFrame, doc: pd.Series) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    doc_counterparty = doc.get("counterparty", "")
    doc_text = doc.get("text", "")
    work = candidates.copy()
    work["_counterparty_score"] = work["counterparty"].apply(lambda value: counterparty_match_score(doc_counterparty, value))
    work["_text_score"] = work["text"].apply(lambda value: text_similarity(doc_text, value))
    work = work[(work["_counterparty_score"] >= 0.55) | (work["_text_score"] >= 0.22)].copy()
    if work.empty:
        return work
    return work.sort_values(["_counterparty_score", "_text_score", "date"], ascending=[False, False, True]).head(10)


def find_amount_subset(candidates: pd.DataFrame, target: float, tolerance: float, max_items: int) -> set[str] | None:
    items = [(str(row["doc_id"]), amount_cents(row["signed_amount"])) for _, row in candidates.head(10).iterrows()]
    return find_subset_ids(items, amount_cents(target), amount_cents(tolerance), max_items)


def find_tx_amount_subset(candidates: pd.DataFrame, target: float, tolerance: float, max_items: int) -> set[str] | None:
    items = [(str(row["tx_id"]), amount_cents(row["amount"])) for _, row in candidates.head(10).iterrows()]
    return find_subset_ids(items, amount_cents(target), amount_cents(tolerance), max_items)


def find_subset_ids(items: list[tuple[str, int]], target_cents: int, tolerance_cents: int, max_items: int) -> set[str] | None:
    best: tuple[int, tuple[str, ...]] | None = None
    for size in range(2, min(max_items, len(items)) + 1):
        for combo in combinations(items, size):
            total = sum(amount for _, amount in combo)
            diff = abs(total - target_cents)
            if diff <= tolerance_cents:
                ids = tuple(item_id for item_id, _ in combo)
                if best is None or diff < best[0] or (diff == best[0] and len(ids) < len(best[1])):
                    best = (diff, ids)
        if best is not None:
            break
    if best is None:
        return None
    return set(best[1])


def split_merge_score(selected_docs: pd.DataFrame, tx: pd.Series, target: float, tolerance: float, window_days: int) -> float:
    selected_sum = round(float(selected_docs["signed_amount"].sum()), 2)
    amount_diff = abs(float(target) - selected_sum)
    amount_score = 1 - min(1, amount_diff / max(tolerance, 0.01))
    day_score = 1 - min(1, float(selected_docs["date"].apply(lambda value: date_distance(value, tx.get("date"))).mean()) / max(window_days, 1))
    counterparty_score = float(selected_docs["counterparty"].apply(lambda value: counterparty_match_score(value, tx.get("counterparty", ""))).max())
    text_score = text_similarity(" ".join(selected_docs["text"].astype(str).head(4)), tx.get("text", ""))
    return amount_score * 0.42 + counterparty_score * 0.30 + day_score * 0.18 + text_score * 0.10


def amount_cents(value: object) -> int:
    return int(round(float(value) * 100))


def date_distance(left: object, right: object) -> int:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return 99999
    return abs(int((pd.Timestamp(left).normalize() - pd.Timestamp(right).normalize()).days))


def platform_monthly_fee_support(
    docs: pd.DataFrame,
    platform_like: pd.DataFrame,
    linked_docs: set[str],
    linked_txs: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Add conservative support links for marketplace monthly fee invoices.

    Accountable can represent eBay/Etsy platform invoices slightly differently
    from the platform statement, especially around VAT/reverse-charge display or
    small end-of-month corrections. These links are intentionally only platform
    detail support. They do not claim a bank payment by themselves.
    """
    if docs.empty or platform_like.empty:
        return [], []

    docs_work = docs.copy()
    docs_work["date"] = pd.to_datetime(docs_work["date"], errors="coerce")
    tx_work = platform_like[
        platform_like["category"].astype(str).eq("fee_month")
        & (platform_like["amount"].astype(float) < -0.005)
        & (~platform_like["tx_id"].astype(str).isin(linked_txs))
    ].copy()
    tx_work["date"] = pd.to_datetime(tx_work["date"], errors="coerce")
    if tx_work.empty:
        return [], []

    candidates: list[tuple[float, pd.Series, pd.Series, float, str]] = []
    for _, doc in docs_work.iterrows():
        doc_id = str(doc.get("doc_id"))
        if doc_id in linked_docs:
            continue
        if str(doc.get("doc_type") or "") != "expense":
            continue
        doc_amount = float(doc.get("signed_amount") or 0)
        if doc_amount >= -0.005 or pd.isna(doc.get("date")):
            continue
        doc_platform = clean_text(doc.get("platform")).lower()
        if doc_platform not in {"etsy", "ebay"}:
            continue
        same_month = tx_work[
            (tx_work["platform"].astype(str).str.lower().eq(doc_platform))
            & (tx_work["date"].dt.to_period("M") == pd.Timestamp(doc.get("date")).to_period("M"))
        ].copy()
        for _, tx in same_month.iterrows():
            tx_amount = float(tx.get("amount") or 0)
            ok, diff, basis = monthly_fee_amount_plausible(abs(doc_amount), abs(tx_amount))
            if not ok:
                continue
            score = 1 - min(1, diff / max(abs(doc_amount), abs(tx_amount), 1))
            text_score = text_similarity(doc.get("text", ""), tx.get("text", ""))
            combined = score * 0.70 + text_score * 0.30
            candidates.append((combined, doc, tx, diff, basis))

    candidates.sort(key=lambda item: item[0], reverse=True)
    matches: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    for score, doc, tx, diff, basis in candidates:
        doc_id = str(doc.get("doc_id"))
        tx_id = str(tx.get("tx_id"))
        if doc_id in linked_docs or tx_id in linked_txs:
            continue
        match_id = stable_id("match", "platform_monthly_fee_support", doc_id, tx_id)
        payload = {
            "method": "platform_detail",
            "confidence": round(max(0.62, min(0.84, score)), 3),
            "doc_sum": round(float(doc.get("signed_amount") or 0), 2),
            "gap": round(float(tx.get("amount") or 0) - float(doc.get("signed_amount") or 0), 2),
            "amount_diff": round(diff, 2),
            "day_diff": 0,
            "text_score": text_similarity(doc.get("text", ""), tx.get("text", "")),
            "explanation": f"Monatsgebühr derselben Plattform und desselben Monats; Betrag plausibel über {basis}, Differenz {diff:.2f} EUR.",
        }
        matches.append(make_match_row(match_id, tx, 1, payload))
        links.append(make_link_row(match_id, doc))
        linked_docs.add(doc_id)
        linked_txs.add(tx_id)
    return matches, links


def monthly_fee_amount_plausible(doc_abs: float, tx_abs: float) -> tuple[bool, float, str]:
    direct_diff = abs(tx_abs - doc_abs)
    if direct_diff <= 5.0:
        return True, direct_diff, "direkte Monatsdifferenz"
    if direct_diff / max(doc_abs, tx_abs, 1) <= 0.16:
        return True, direct_diff, "relative Monatsdifferenz"
    reverse_charge_diff = abs((tx_abs / 1.19) - doc_abs)
    if reverse_charge_diff <= 2.0:
        return True, reverse_charge_diff, "19%-USt-/Reverse-Charge-Naeherung"
    return False, min(direct_diff, reverse_charge_diff), ""


def match_platform_payouts_to_bank(
    platform_transactions: pd.DataFrame,
    bank: pd.DataFrame,
    settings: MatchSettings,
) -> pd.DataFrame:
    if platform_transactions.empty or bank.empty:
        return empty_platform_bank_matches()

    payouts = platform_transactions[
        (platform_transactions["category"].isin(["payout", "bank_transfer"]))
        & (platform_transactions["amount"].abs() > 0.005)
    ].copy()
    if payouts.empty:
        return empty_platform_bank_matches()

    bank_work = bank.copy()
    bank_work["_amount_cents"] = (bank_work["amount"].astype(float) * 100).round().astype(int)
    bank_work["_date"] = pd.to_datetime(bank_work["date"], errors="coerce")
    tolerance_cents = int(settings.tolerance_cents)
    tolerance = tolerance_cents / 100
    rows: list[dict[str, object]] = []
    used_bank: set[str] = set()

    for _, platform_tx in payouts.sort_values("date").iterrows():
        platform_name = clean_text(platform_tx.get("platform"))
        target = -float(platform_tx["amount"])
        target_cents = int(round(target * 100))
        possible = bank_work[
            (~bank_work["tx_id"].isin(used_bank))
            & bank_work["_amount_cents"].between(target_cents - tolerance_cents, target_cents + tolerance_cents)
        ].copy()
        if possible.empty:
            continue
        if platform_name:
            possible = possible[
                possible["text"].str.contains(platform_name, case=False, na=False)
                | possible["platform"].astype(str).str.contains(platform_name, case=False, na=False)
            ].copy()
        if possible.empty:
            continue
        possible["amount_diff"] = (possible["amount"] - target).abs()
        possible["day_diff"] = (possible["_date"] - pd.Timestamp(platform_tx["date"])).dt.days.abs()
        possible = possible[
            (possible["amount_diff"] <= tolerance)
            & (possible["day_diff"] <= max(settings.direct_window_days, 14))
        ].copy()
        if possible.empty:
            continue
        possible["text_score"] = possible["text"].apply(lambda text: text_similarity(platform_tx.get("text", ""), text))
        possible["score"] = (
            (1 - possible["amount_diff"] / max(tolerance, 0.01)).clip(lower=0) * 0.55
            + (1 - possible["day_diff"] / max(settings.direct_window_days, 14)).clip(lower=0) * 0.30
            + possible["text_score"] * 0.15
        )
        best = possible.sort_values("score", ascending=False).iloc[0]
        used_bank.add(best["tx_id"])
        confidence = round(max(0.65, min(0.98, float(best["score"]))), 3)
        rows.append(
            {
                "bridge_id": stable_id("platformbank", platform_tx["tx_id"], best["tx_id"]),
                "platform_tx_id": platform_tx["tx_id"],
                "tx_id": best["tx_id"],
                "platform": platform_tx.get("platform", ""),
                "platform_date": platform_tx["date"],
                "platform_amount": round(float(platform_tx["amount"]), 2),
                "platform_category": platform_tx.get("category", ""),
                "platform_description": platform_tx.get("description", ""),
                "payout_id": platform_tx.get("payout_id", ""),
                "bank_date": best["date"],
                "bank_amount": round(float(best["amount"]), 2),
                "bank_source": best["source"],
                "bank_counterparty": best["counterparty"],
                "bank_description": best["description"],
                "amount_diff": round(float(best["amount_diff"]), 2),
                "day_diff": int(best["day_diff"]),
                "confidence": confidence,
                "explanation": "Plattform-Auszahlung mit Gegenbuchung auf FYRST; Plattform-Vorzeichen ist spiegelbildlich zur Bank.",
            }
        )

    return pd.DataFrame(rows) if rows else empty_platform_bank_matches()


def match_platform_details_to_bank(
    docs: pd.DataFrame,
    platform_transactions: pd.DataFrame,
    bank: pd.DataFrame,
    settings: MatchSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match supplier/platform charge details directly to bank rows.

    This is intentionally narrow: it only handles platform detail rows that are
    themselves payment-like charges. Monthly marketplace ledgers are not closed
    by this function.
    """
    if docs.empty or platform_transactions.empty or bank.empty:
        return empty_matches(), empty_links(), empty_platform_bank_matches()

    details = platform_transactions[
        platform_transactions["category"].astype(str).eq("charge")
        & (platform_transactions["amount"].astype(float).abs() > 0.005)
    ].copy()
    if details.empty:
        return empty_matches(), empty_links(), empty_platform_bank_matches()

    docs_work = docs.copy()
    docs_work["date"] = pd.to_datetime(docs_work["date"], errors="coerce")
    details["date"] = pd.to_datetime(details["date"], errors="coerce")
    bank_work = bank.copy()
    bank_work["date"] = pd.to_datetime(bank_work["date"], errors="coerce")

    tolerance = max(settings.tolerance_cents, 10) / 100
    doc_detail_window = max(settings.direct_window_days, 45)
    detail_bank_window = max(settings.direct_window_days, 21)

    chain_candidates: list[tuple[float, pd.Series, pd.Series, pd.Series, float, int, int]] = []
    for _, doc in docs_work.iterrows():
        doc_amount = float(doc.get("signed_amount") or 0)
        if doc_amount >= -0.005:
            continue
        possible_details = details[
            (details["amount"].astype(float) * doc_amount > 0)
            & (details["amount"].astype(float).sub(doc_amount).abs() <= tolerance)
            & details.apply(lambda tx: platform_match_allowed(doc, tx), axis=1)
        ].copy()
        if possible_details.empty:
            continue
        possible_details["_doc_day_diff"] = possible_details["date"].apply(lambda value: date_distance(value, doc.get("date")))
        possible_details = possible_details[possible_details["_doc_day_diff"] <= doc_detail_window].copy()
        if possible_details.empty:
            continue

        for _, detail in possible_details.iterrows():
            bank_candidates = bank_work[
                (bank_work["amount"].astype(float).sub(float(detail.get("amount") or 0)).abs() <= tolerance)
                & bank_work.apply(lambda tx: platform_detail_bank_text_hit(detail, tx), axis=1)
            ].copy()
            if bank_candidates.empty:
                continue
            bank_candidates["_bank_day_diff"] = bank_candidates["date"].apply(lambda value: date_distance(value, detail.get("date")))
            bank_candidates = bank_candidates[bank_candidates["_bank_day_diff"] <= detail_bank_window].copy()
            if bank_candidates.empty:
                continue
            bank_candidates["_text_score"] = bank_candidates["text"].apply(lambda text: text_similarity(detail.get("text", ""), text))
            best_bank = bank_candidates.sort_values(["_text_score", "_bank_day_diff"], ascending=[False, True]).iloc[0]
            doc_day_diff = int(detail["_doc_day_diff"])
            bank_day_diff = int(best_bank["_bank_day_diff"])
            amount_diff = abs(float(detail.get("amount") or 0) - doc_amount)
            bank_amount_diff = abs(float(best_bank.get("amount") or 0) - float(detail.get("amount") or 0))
            amount_score = 1 - min(1, max(amount_diff, bank_amount_diff) / max(tolerance, 0.01))
            doc_date_score = 1 - min(1, doc_day_diff / max(doc_detail_window, 1))
            bank_date_score = 1 - min(1, bank_day_diff / max(detail_bank_window, 1))
            score = amount_score * 0.45 + doc_date_score * 0.20 + bank_date_score * 0.20 + float(best_bank["_text_score"]) * 0.15
            chain_candidates.append((score, doc, detail, best_bank, max(amount_diff, bank_amount_diff), doc_day_diff, bank_day_diff))

    chain_candidates.sort(key=lambda item: item[0], reverse=True)
    doc_matches: list[dict[str, object]] = []
    doc_links: list[dict[str, object]] = []
    bank_rows: list[dict[str, object]] = []
    used_docs: set[str] = set()
    used_details: set[str] = set()
    used_bank: set[str] = set()

    for score, doc, detail, bank_tx, amount_diff, doc_day_diff, bank_day_diff in chain_candidates:
        doc_id = str(doc.get("doc_id"))
        detail_id = str(detail.get("tx_id"))
        bank_id = str(bank_tx.get("tx_id"))
        if doc_id in used_docs or detail_id in used_details or bank_id in used_bank:
            continue
        confidence = round(max(0.78, min(0.97, score)), 3)
        match_id = stable_id("match", "platform_detail_bank_chain", doc_id, detail_id, bank_id)
        payload = {
            "method": "platform_detail_bank_chain",
            "confidence": confidence,
            "doc_sum": round(float(doc.get("signed_amount") or 0), 2),
            "gap": round(float(detail.get("amount") or 0) - float(doc.get("signed_amount") or 0), 2),
            "amount_diff": round(amount_diff, 2),
            "day_diff": doc_day_diff,
            "text_score": text_similarity(doc.get("text", ""), detail.get("text", "")),
            "explanation": "Plattform-Detail passt zum Beleg und dieselbe Plattform-Zahlung ist direkt in FYRST gefunden.",
        }
        doc_matches.append(make_match_row(match_id, detail, 1, payload))
        doc_links.append(make_link_row(match_id, doc))
        bank_rows.append(
            {
                "bridge_id": stable_id("platformbank", "detail", detail_id, bank_id),
                "platform_tx_id": detail_id,
                "tx_id": bank_id,
                "platform": detail.get("platform", ""),
                "platform_date": detail.get("date"),
                "platform_amount": round(float(detail.get("amount") or 0), 2),
                "platform_category": detail.get("category", ""),
                "platform_description": detail.get("description", ""),
                "payout_id": detail.get("payout_id", ""),
                "bank_date": bank_tx.get("date"),
                "bank_amount": round(float(bank_tx.get("amount") or 0), 2),
                "bank_source": bank_tx.get("source", ""),
                "bank_counterparty": bank_tx.get("counterparty", ""),
                "bank_description": bank_tx.get("description", ""),
                "amount_diff": round(abs(float(bank_tx.get("amount") or 0) - float(detail.get("amount") or 0)), 2),
                "day_diff": bank_day_diff,
                "confidence": confidence,
                "explanation": "Plattform-Detailzahlung direkt mit FYRST-Umsatz verbunden; gleiches Vorzeichen.",
            }
        )
        used_docs.add(doc_id)
        used_details.add(detail_id)
        used_bank.add(bank_id)

    return (
        pd.DataFrame(doc_matches) if doc_matches else empty_matches(),
        pd.DataFrame(doc_links) if doc_links else empty_links(),
        pd.DataFrame(bank_rows) if bank_rows else empty_platform_bank_matches(),
    )


def platform_detail_bank_text_hit(detail: pd.Series, bank_tx: pd.Series) -> bool:
    text = clean_text(f"{bank_tx.get('counterparty', '')} {bank_tx.get('description', '')} {bank_tx.get('text', '')}").lower()
    platform = clean_text(detail.get("platform")).lower()
    counterparty = clean_text(detail.get("counterparty")).lower()
    payout_id = clean_text(detail.get("payout_id")).lower()
    return bool(platform and platform in text) or bool(counterparty and counterparty in text) or bool(payout_id and payout_id in text)


def platform_counterparty_one_to_one(
    docs: pd.DataFrame,
    platform_like: pd.DataFrame,
    linked_docs: set[str],
    linked_txs: set[str],
    settings: MatchSettings,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if docs.empty or platform_like.empty:
        return [], []

    docs_work = docs.copy()
    docs_work["date"] = pd.to_datetime(docs_work["date"], errors="coerce")
    tx_work = platform_like.copy()
    tx_work["date"] = pd.to_datetime(tx_work["date"], errors="coerce")
    tolerance = settings.tolerance_cents / 100
    candidates: list[tuple[float, dict[str, object], pd.Series, pd.Series]] = []

    for _, tx in tx_work.iterrows():
        tx_counterparty = clean_text(tx.get("counterparty"))
        if not tx_counterparty or tx_counterparty.lower() in {"etsy", "ebay", "shopify", "gelato", "printful"}:
            continue
        tx_amount = float(tx.get("amount") or 0)
        if abs(tx_amount) < 0.005:
            continue
        for _, doc in docs_work.iterrows():
            doc_amount = float(doc.get("signed_amount") or 0)
            if tx_amount * doc_amount <= 0:
                continue
            amount_diff = abs(tx_amount - doc_amount)
            if amount_diff > tolerance:
                continue
            day_diff = abs((pd.Timestamp(tx.get("date")).normalize() - pd.Timestamp(doc.get("date")).normalize()).days) if not pd.isna(tx.get("date")) and not pd.isna(doc.get("date")) else 99999
            if day_diff > settings.direct_window_days:
                continue
            counterparty_score = counterparty_match_score(doc.get("counterparty"), tx_counterparty)
            if counterparty_score < 0.6:
                continue
            sim = text_similarity(doc.get("text", ""), tx.get("text", ""))
            amount_score = 1 - min(1, amount_diff / max(tolerance, 0.01))
            date_score = 1 - min(1, day_diff / max(settings.direct_window_days, 1))
            score = counterparty_score * 0.50 + amount_score * 0.25 + date_score * 0.15 + sim * 0.10
            payload = {
                "method": "platform_detail",
                "confidence": round(max(0.72, min(0.99, score)), 3),
                "doc_sum": round(doc_amount, 2),
                "gap": round(tx_amount - doc_amount, 2),
                "amount_diff": round(amount_diff, 2),
                "day_diff": day_diff,
                "text_score": sim,
                "explanation": f"Plattform-Kunde und Accountable-Kunde stimmen ueberein; Betrag passt innerhalb {settings.tolerance_cents} Cent; Datumsabstand {day_diff} Tage.",
            }
            candidates.append((score, payload, doc, tx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    matches: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    for _, payload, doc, tx in candidates:
        doc_id = str(doc.get("doc_id"))
        tx_id = str(tx.get("tx_id"))
        if doc_id in linked_docs or tx_id in linked_txs:
            continue
        match_id = stable_id("match", "platform_detail_counterparty", doc_id, tx_id)
        matches.append(make_match_row(match_id, tx, 1, payload))
        links.append(make_link_row(match_id, doc))
        linked_docs.add(doc_id)
        linked_txs.add(tx_id)
    return matches, links


def counterparty_match_score(left: object, right: object) -> float:
    left_tokens = significant_tokens(left)
    right_tokens = significant_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    return len(overlap) / max(1, min(len(left_tokens), len(right_tokens)))


def significant_tokens(value: object) -> set[str]:
    text = clean_text(value).lower()
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    return {token for token in re.sub(r"[^a-z0-9]+", " ", text).split() if len(token) >= 4}


def empty_platform_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=PLATFORM_COLUMNS)


def empty_platform_bank_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bridge_id",
            "platform_tx_id",
            "tx_id",
            "platform",
            "platform_date",
            "platform_amount",
            "platform_category",
            "platform_description",
            "payout_id",
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


def make_platform_row(**kwargs) -> dict[str, object]:
    amount = float(kwargs["amount"])
    tx_id = stable_id(
        "plat",
        kwargs["source"],
        kwargs["file_name"],
        kwargs["source_row"],
        kwargs.get("category"),
        kwargs.get("order_id"),
        kwargs.get("transaction_id"),
        kwargs.get("payout_id"),
        kwargs.get("date"),
        round(amount, 2),
    )
    text = " ".join(
        scrub_text(value)
        for value in [
            kwargs.get("platform"),
            kwargs.get("category"),
            kwargs.get("counterparty"),
            kwargs.get("order_id"),
            kwargs.get("transaction_id"),
            kwargs.get("payout_id"),
            kwargs.get("description"),
            kwargs.get("item_name"),
        ]
        if clean_text(value)
    )
    return {
        "tx_id": tx_id,
        "source_file": kwargs["file_name"],
        "source": kwargs["source"],
        "source_row": kwargs["source_row"],
        "date": kwargs["date"],
        "payout_date": kwargs["payout_date"],
        "amount": round(amount, 2),
        "gross_amount": kwargs.get("gross_amount"),
        "fee_amount": kwargs.get("fee_amount"),
        "currency": kwargs.get("currency") or "EUR",
        "direction": "income" if amount > 0 else "expense" if amount < 0 else "zero",
        "platform": kwargs.get("platform", ""),
        "category": kwargs.get("category", ""),
        "counterparty": scrub_text(kwargs.get("counterparty", "")),
        "order_id": scrub_text(kwargs.get("order_id", "")),
        "transaction_id": scrub_text(kwargs.get("transaction_id", "")),
        "payout_id": scrub_text(kwargs.get("payout_id", "")),
        "description": scrub_text(kwargs.get("description", "")),
        "item_name": scrub_text(kwargs.get("item_name", "")),
        "quantity": kwargs.get("quantity"),
        "shop": scrub_text(kwargs.get("shop", "")),
        "text": text,
        "raw_summary": scrub_text(kwargs.get("raw_summary", "")),
    }


def scrub_text(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    for snippet in BANNED_TEXT_SNIPPETS:
        text = re.sub(re.escape(snippet), "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" |;-")


def ebay_category(value: str) -> str:
    normalized = value.lower()
    if "bestellung" in normalized:
        return "order"
    if "auszahlung" in normalized:
        return "payout"
    if "gebühr" in normalized or "gebuehr" in normalized:
        return "fee"
    if "belastung" in normalized:
        return "charge"
    if "rückerstattung" in normalized or "rueckerstattung" in normalized:
        return "refund"
    return "adjustment"


def parse_statement_date(value: object) -> pd.Timestamp | pd.NaT:
    text = clean_text(value)
    if not text:
        return pd.NaT
    text = re.sub(r"^(\d{1,2})\.\s+", r"\1 ", text)
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, dayfirst=False, errors="coerce")
    return normalize_timestamp(parsed)


def parse_us_date(value: object) -> pd.Timestamp | pd.NaT:
    text = clean_text(value)
    if not text:
        return pd.NaT
    parsed = pd.to_datetime(text, dayfirst=False, errors="coerce")
    return normalize_timestamp(parsed)


def parse_isoish_date(value: object) -> pd.Timestamp | pd.NaT:
    text = clean_text(value)
    if not text:
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce")
    return normalize_timestamp(parsed)


def parse_german_date(value: object) -> pd.Timestamp | pd.NaT:
    text = clean_text(value).replace("--", "")
    if not text:
        return pd.NaT
    replacements = {
        "März": "Mar",
        "Mär": "Mar",
        "Mrz": "Mar",
        "Mai": "May",
        "Okt": "Oct",
        "Dez": "Dec",
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    text = re.sub(r"\b(MEZ|MESZ)\b", "", text).strip()
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return normalize_timestamp(parsed)


def parse_english_date(value: object) -> pd.Timestamp | pd.NaT:
    text = clean_text(value)
    if not text:
        return pd.NaT
    parsed = pd.to_datetime(text, dayfirst=False, errors="coerce")
    return normalize_timestamp(parsed)


def normalize_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def sum_amounts(values: pd.Series | None) -> float:
    if values is None:
        return 0.0
    total = 0.0
    for value in values:
        amount = parse_amount(value)
        if amount is not None:
            total += float(amount)
    return round(total, 2)


def max_amount(values: pd.Series | None) -> float:
    if values is None:
        return 0.0
    amounts = [float(amount) for value in values if (amount := parse_amount(value)) is not None]
    return round(max(amounts), 2) if amounts else 0.0


def first_amount(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    for value in values:
        amount = parse_amount(value)
        if amount is not None:
            return float(amount)
    return None


def first_number(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None


def sum_fee_columns(row: pd.Series, columns: list[str]) -> float | None:
    values = [parse_amount(row.get(col)) for col in columns]
    amounts = [float(value) for value in values if value is not None]
    if not amounts:
        return None
    return round(sum(amounts), 2)


def ebay_synthetic_payout_rows(file_name: str, frame: pd.DataFrame, start_row: int) -> list[dict[str, object]]:
    if frame.empty or "payout_id" not in frame.columns:
        return []
    work = frame[
        frame["payout_id"].apply(has_real_text)
        & ~frame["category"].astype(str).eq("payout")
    ].copy()
    if work.empty:
        return []
    work["_payout_date"] = pd.to_datetime(work["payout_date"], errors="coerce")
    work = work[work["_payout_date"].notna()].copy()
    if work.empty:
        return []

    existing_payout_ids = set(
        frame.loc[frame["category"].astype(str).eq("payout"), "payout_id"].astype(str)
    )
    rows: list[dict[str, object]] = []
    grouped = work.groupby(["payout_id", "_payout_date"], dropna=False)
    for offset, ((payout_id, payout_date), group) in enumerate(grouped, start=1):
        payout_id = clean_text(payout_id)
        if not payout_id or payout_id in existing_payout_ids:
            continue
        detail_net = round(float(group["amount"].sum()), 2)
        if abs(detail_net) < 0.01:
            continue
        payout_amount = round(-detail_net, 2)
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="eBay",
                source_row=start_row + offset,
                date=payout_date,
                payout_date=payout_date,
                amount=payout_amount,
                gross_amount=None,
                fee_amount=None,
                currency="EUR",
                platform="ebay",
                category="payout",
                counterparty="eBay",
                order_id="",
                transaction_id=f"ebay-payout-{payout_id}",
                payout_id=payout_id,
                description=f"eBay Auszahlung {payout_id}",
                item_name="",
                quantity=None,
                shop="",
                raw_summary=f"eBay synthetic payout {payout_id} from {len(group)} detail rows",
            )
        )
    return rows


def ebay_monthly_fee_rows(file_name: str, frame: pd.DataFrame, start_row: int) -> list[dict[str, object]]:
    fees = frame[frame["category"].astype(str).eq("fee")].copy()
    if fees.empty:
        return []
    fees["_date"] = pd.to_datetime(fees["date"], errors="coerce")
    fees = fees[fees["_date"].notna()].copy()
    if fees.empty:
        return []

    rows: list[dict[str, object]] = []
    for offset, (period, group) in enumerate(fees.groupby(fees["_date"].dt.to_period("M")), start=1):
        amount = round(float(group["amount"].sum()), 2)
        if abs(amount) < 0.01:
            continue
        month_date = period.to_timestamp(how="end").normalize()
        month_key = str(period)
        rows.append(
            make_platform_row(
                file_name=file_name,
                source="eBay",
                source_row=start_row + offset,
                date=month_date,
                payout_date=pd.NaT,
                amount=amount,
                gross_amount=None,
                fee_amount=amount,
                currency="EUR",
                platform="ebay",
                category="fee_month",
                counterparty="eBay",
                order_id="",
                transaction_id=f"ebay-fees-{month_key}",
                payout_id="",
                description=f"eBay Monatsgebuehren {month_key}",
                item_name="",
                quantity=None,
                shop="",
                raw_summary=f"eBay monthly fee aggregate {month_key}",
            )
        )
    return rows


def first_nonempty_row(group: pd.DataFrame) -> pd.Series:
    for _, row in group.iterrows():
        if any(clean_text(value) for value in row.values):
            return row
    return group.iloc[0]


def first_clean(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        value = clean_text(row.get(column))
        if value and value != "--":
            return value
    return ""


def join_nonempty(values) -> str:
    return " ".join(clean_text(value) for value in values if clean_text(value))


def extract_order_id(*values: object) -> str:
    text = " ".join(clean_text(value) for value in values if clean_text(value))
    for pattern in [
        r"Order\s*#\s*([A-Za-z0-9-]+)",
        r"Bestellung\s+([A-Za-z0-9-]+)",
        r"order\s+([A-Za-z0-9-]+)",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def extract_statement_transaction_id(*values: object) -> str:
    text = " ".join(clean_text(value) for value in values if clean_text(value))
    article = re.search(r"Artikel\s+Nr\.\s*([A-Za-z0-9-]+)", text, flags=re.IGNORECASE)
    if article:
        return article.group(1)
    order_id = extract_order_id(text)
    if order_id:
        return order_id
    return text[:80]


def etsy_payout_key(file_name: str, date: object) -> str:
    if date is None or pd.isna(date):
        return ""
    return f"etsy-{etsy_account_key(file_name)}-{pd.Timestamp(date).strftime('%Y-%m-%d')}"


def etsy_account_key(file_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", file_name.lower()).strip()
    tokens = normalized.split()
    if "frida" in tokens:
        return "frida"
    if "ff" in tokens:
        return "ff"
    for noise in ["etsy", "payment", "payments", "verk", "ufe", "verkaufe", "verk", "berweisungen", "ueberweisungen", "csv"]:
        tokens = [token for token in tokens if token != noise]
    return re.sub(r"[^a-z0-9]+", "-", " ".join(tokens)).strip("-") or "default"


def join_limited(values: pd.Series, limit: int = 5) -> str:
    seen: list[str] = []
    for value in values:
        text = clean_text(value)
        if not text or text in seen:
            continue
        seen.append(text)
        if len(seen) >= limit:
            break
    return " | ".join(seen)
