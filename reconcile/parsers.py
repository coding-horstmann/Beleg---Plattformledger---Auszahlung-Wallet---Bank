from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class ParseResult:
    frame: pd.DataFrame
    issues: list[str]


def decode_text(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin1", errors="replace"), "latin1-replace"


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8', errors='ignore')).hexdigest()[:12]}"


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_amount(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = clean_text(value)
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    if text.endswith("-"):
        negative = True
        text = text[:-1]

    text = (
        text.replace("\xa0", "")
        .replace("EUR", "")
        .replace("€", "")
        .replace(" ", "")
        .replace("'", "")
    )
    text = re.sub(r"[^0-9,.\-+]", "", text)
    if not text or text in {"-", "+", ".", ","}:
        return None

    if "," in text:
        text = text.replace(".", "").replace(",", ".")

    try:
        number = float(text)
    except ValueError:
        return None
    return -abs(number) if negative else number


def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).normalize()


def first_existing(row: pd.Series, names: Iterable[str]) -> object:
    for name in names:
        if name in row and not pd.isna(row[name]) and clean_text(row[name]):
            return row[name]
    return None


def first_existing_with_name(row: pd.Series, names: Iterable[str]) -> tuple[object, str]:
    for name in names:
        if name in row and not pd.isna(row[name]) and clean_text(row[name]):
            return row[name], name
    return None, ""


def combine_fields(row: pd.Series, names: Iterable[str]) -> str:
    return " | ".join(clean_text(row[name]) for name in names if name in row and clean_text(row[name]))


def detect_platform(*values: object) -> str:
    text = normalize_for_category(" ".join(clean_text(value) for value in values if clean_text(value)))
    rules = [
        ("etsy", ["etsy"]),
        ("ebay", ["ebay"]),
        ("paypal", ["paypal"]),
        ("shopify", ["shopify"]),
        ("printler", ["printler"]),
        ("gelato", ["gelato"]),
        ("printful", ["printful"]),
        ("redbubble", ["redbubble"]),
        ("teepublic", ["teepublic", "tee public"]),
        ("spreadshirt", ["sprd", "spreadshirt"]),
        ("juniqe", ["juniqe"]),
        ("amazon", ["amazon"]),
        ("stripe", ["stripe"]),
        ("ionos", ["ionos"]),
    ]
    for platform, needles in rules:
        if any(needle in text for needle in needles):
            return platform
    return ""


def find_csv_header(lines: list[str], required: Iterable[str]) -> int | None:
    required_lower = [term.lower() for term in required]
    for idx, line in enumerate(lines[:120]):
        lower = line.lower()
        if all(term in lower for term in required_lower):
            return idx
    return None


def parse_bank_file(name: str, data: bytes) -> ParseResult:
    text, encoding = decode_text(data)
    issues: list[str] = []

    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            message = payload.get("message") or payload
            return ParseResult(pd.DataFrame(), [f"{name}: keine Bank-CSV, sondern API/Export-Fehler: {message}"])
        except json.JSONDecodeError:
            pass

    lines = text.splitlines()
    header_idx = find_csv_header(lines, ["Buchungstag", "Betrag"])
    source_hint = "FYRST"
    if header_idx is None:
        header_idx = find_csv_header(lines, ["Buchungsdatum", "Betrag"])
        source_hint = "DKB"

    if header_idx is None:
        return parse_generic_bank_file(name, data)

    try:
        raw = pd.read_csv(io.StringIO(text), sep=";", skiprows=header_idx, dtype=str)
    except Exception as exc:  # pragma: no cover - defensive UI path
        return ParseResult(pd.DataFrame(), [f"{name}: CSV konnte nicht gelesen werden ({exc})."])

    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)

    if "Buchungstag" in raw.columns:
        result = normalize_fyrst_bank(raw, name)
    elif "Buchungsdatum" in raw.columns:
        result = normalize_dkb_bank(raw, name)
    else:
        return ParseResult(pd.DataFrame(), [f"{name}: Header erkannt, aber Spalten sind unbekannt: {list(raw.columns)}"])

    result.issues.insert(0, f"{name}: {len(result.frame)} Bankumsätze importiert ({source_hint}, {encoding}).")
    return result


def parse_generic_bank_file(name: str, data: bytes) -> ParseResult:
    text, encoding = decode_text(data)
    issues: list[str] = []
    best: pd.DataFrame | None = None
    best_sep = ";"
    best_skip = 0

    for sep in (";", ",", "\t"):
        for skip in range(0, 20):
            try:
                candidate = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skip, dtype=str, nrows=20)
            except Exception:
                continue
            cols = [clean_text(c).lower() for c in candidate.columns]
            has_date = any("datum" in c or "date" in c or "buchung" in c for c in cols)
            has_amount = any("betrag" in c or "amount" in c or "haben" in c or "soll" in c for c in cols)
            if has_date and has_amount and len(cols) >= 3:
                best = candidate
                best_sep = sep
                best_skip = skip
                break
        if best is not None:
            break

    if best is None:
        return ParseResult(pd.DataFrame(), [f"{name}: CSV-Format nicht erkannt."])

    raw = pd.read_csv(io.StringIO(text), sep=best_sep, skiprows=best_skip, dtype=str)
    raw.columns = [clean_text(col) for col in raw.columns]
    raw = raw.dropna(how="all").reset_index(drop=True)

    lower = {col.lower(): col for col in raw.columns}
    date_col = pick_column(lower, ["buchungsdatum", "buchungstag", "date", "datum", "wertstellung"])
    amount_col = pick_column(lower, ["betrag (€)", "betrag", "amount", "umsatz"])
    credit_col = pick_column(lower, ["haben", "credit"])
    debit_col = pick_column(lower, ["soll", "debit"])

    if date_col is None or (amount_col is None and credit_col is None and debit_col is None):
        return ParseResult(pd.DataFrame(), [f"{name}: Generisches Mapping nicht möglich: {list(raw.columns)}"])

    rows = []
    for idx, row in raw.iterrows():
        amount = parse_amount(row[amount_col]) if amount_col else None
        if amount is None:
            credit = parse_amount(row[credit_col]) if credit_col else None
            debit = parse_amount(row[debit_col]) if debit_col else None
            if credit is not None:
                amount = abs(credit)
            elif debit is not None:
                amount = -abs(debit)
        date = parse_date(row[date_col])
        if amount is None or pd.isna(date):
            continue
        text_fields = [col for col in raw.columns if col != amount_col]
        rows.append(make_bank_row(name, "Generic", idx, row, date, date, amount, "", combine_fields(row, text_fields), "EUR"))

    frame = pd.DataFrame(rows)
    issues.append(f"{name}: {len(frame)} Bankumsätze generisch importiert ({encoding}). Bitte Mapping prüfen.")
    return ParseResult(frame, issues)


def pick_column(lower_to_original: dict[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    for key, original in lower_to_original.items():
        if any(candidate in key for candidate in candidates):
            return original
    return None


def normalize_fyrst_bank(raw: pd.DataFrame, file_name: str) -> ParseResult:
    rows = []
    issues: list[str] = []
    for idx, row in raw.iterrows():
        amount = parse_amount(row.get("Betrag"))
        if amount is None:
            credit = parse_amount(row.get("Haben"))
            debit = parse_amount(row.get("Soll"))
            if credit is not None:
                amount = abs(credit)
            elif debit is not None:
                amount = -abs(debit)
        date = parse_date(row.get("Buchungstag"))
        value_date = parse_date(row.get("Wert"))
        if amount is None or pd.isna(date):
            continue
        counterparty = clean_text(row.get("Begünstigter / Auftraggeber"))
        description = combine_fields(row, ["Umsatzart", "Verwendungszweck", "Kundenreferenz", "Mandatsreferenz", "Gläubiger ID"])
        currency = clean_text(row.get("Währung")) or "EUR"
        rows.append(make_bank_row(file_name, "FYRST", idx, row, date, value_date, amount, counterparty, description, currency))
    return ParseResult(pd.DataFrame(rows), issues)


def normalize_dkb_bank(raw: pd.DataFrame, file_name: str) -> ParseResult:
    rows = []
    issues: list[str] = []
    for idx, row in raw.iterrows():
        amount = parse_amount(row.get("Betrag (€)"))
        date = parse_date(row.get("Buchungsdatum"))
        value_date = parse_date(row.get("Wertstellung"))
        if amount is None or pd.isna(date):
            continue
        payer = clean_text(row.get("Zahlungspflichtige*r"))
        payee = clean_text(row.get("Zahlungsempfänger*in"))
        counterparty = payee if amount < 0 else payer
        description = combine_fields(row, ["Umsatztyp", "Verwendungszweck", "Kundenreferenz", "Mandatsreferenz", "Gläubiger-ID"])
        rows.append(make_bank_row(file_name, "DKB", idx, row, date, value_date, amount, counterparty, description, "EUR"))
    return ParseResult(pd.DataFrame(rows), issues)


def make_bank_row(
    file_name: str,
    source: str,
    idx: int,
    row: pd.Series,
    date: pd.Timestamp,
    value_date: pd.Timestamp | pd.NaT,
    amount: float,
    counterparty: str,
    description: str,
    currency: str,
) -> dict[str, object]:
    text = f"{counterparty} {description}".strip()
    tx_id = stable_id("tx", source, file_name, idx, date.date(), round(amount, 2), text[:160])
    return {
        "tx_id": tx_id,
        "source_file": file_name,
        "source": source,
        "source_row": idx + 1,
        "date": date,
        "value_date": value_date if not pd.isna(value_date) else date,
        "amount": round(float(amount), 2),
        "currency": currency,
        "direction": "income" if amount > 0 else "expense" if amount < 0 else "zero",
        "counterparty": counterparty,
        "description": description,
        "platform": detect_platform(counterparty, description),
        "text": text,
        "raw_summary": combine_fields(row, row.index[: min(len(row.index), 18)]),
    }


def parse_accountable_file(name: str, data: bytes) -> ParseResult:
    suffix = Path(name).suffix.lower()
    issues: list[str] = []
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            excel = pd.ExcelFile(io.BytesIO(data))
        except Exception as exc:
            return ParseResult(pd.DataFrame(), [f"{name}: Excel-Datei konnte nicht gelesen werden ({exc})."])

        frames = []
        if "Rechnungen" in excel.sheet_names:
            invoices = pd.read_excel(excel, sheet_name="Rechnungen")
            frames.append(normalize_accountable_invoices(invoices, name))
        else:
            issues.append(f"{name}: Blatt 'Rechnungen' fehlt.")

        if "Ausgaben" in excel.sheet_names:
            expenses = pd.read_excel(excel, sheet_name="Ausgaben")
            frames.append(normalize_accountable_expenses(expenses, name))
        else:
            issues.append(f"{name}: Blatt 'Ausgaben' fehlt.")

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        issues.insert(0, f"{name}: {len(frame)} Accountable-Belege importiert ({', '.join(excel.sheet_names)}).")
        return ParseResult(frame, issues)

    if suffix == ".csv":
        text, encoding = decode_text(data)
        try:
            raw = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception as exc:
            return ParseResult(pd.DataFrame(), [f"{name}: CSV konnte nicht gelesen werden ({exc})."])
        raw.columns = [clean_text(col) for col in raw.columns]
        lower = {c.lower() for c in raw.columns}
        if "einkommenszahl" in lower or "kundenname" in lower:
            frame = normalize_accountable_invoices(raw, name)
        elif "name des lieferanten" in lower:
            frame = normalize_accountable_expenses(raw, name)
        else:
            return ParseResult(pd.DataFrame(), [f"{name}: Accountable-CSV nicht erkannt: {list(raw.columns)}"])
        return ParseResult(frame, [f"{name}: {len(frame)} Accountable-Belege aus CSV importiert ({encoding})."])

    return ParseResult(pd.DataFrame(), [f"{name}: Dateityp wird fuer Accountable nicht unterstuetzt."])


def parse_paypal_file(name: str, data: bytes) -> ParseResult:
    text, encoding = decode_text(data)
    issues: list[str] = []
    try:
        raw = pd.read_csv(io.StringIO(text), sep=",", dtype=str)
    except Exception as exc:
        return ParseResult(pd.DataFrame(), [f"{name}: PayPal-CSV konnte nicht gelesen werden ({exc})."])

    raw.columns = [clean_text(col) for col in raw.columns]
    required = {"Datum", "Beschreibung", "Netto", "Transaktionscode"}
    missing = sorted(required - set(raw.columns))
    if missing:
        return ParseResult(pd.DataFrame(), [f"{name}: PayPal-CSV nicht erkannt. Fehlende Spalten: {', '.join(missing)}"])

    rows = []
    for idx, row in raw.dropna(how="all").reset_index(drop=True).iterrows():
        net = parse_amount(row.get("Netto"))
        gross = parse_amount(row.get("Brutto"))
        fee = parse_amount(row.get("Entgelt"))
        date = parse_date(row.get("Datum"))
        if net is None or pd.isna(date):
            continue
        description = clean_text(row.get("Beschreibung"))
        transaction_code = clean_text(row.get("Transaktionscode"))
        related_code = clean_text(row.get("Zugehöriger Transaktionscode"))
        name_value = clean_text(row.get("Name"))
        email = clean_text(row.get("Absender E-Mail-Adresse"))
        invoice_number = clean_text(row.get("Rechnungsnummer"))
        category = classify_paypal_description(description)
        pp_id = stable_id("pp", name, idx, transaction_code, date.date(), round(float(net), 2))
        text_blob = combine_fields(
            row,
            [
                "Beschreibung",
                "Name",
                "Absender E-Mail-Adresse",
                "Rechnungsnummer",
                "Transaktionscode",
                "Zugehöriger Transaktionscode",
            ],
        )
        rows.append(
            {
                "pp_id": pp_id,
                "source_file": name,
                "source": "PayPal",
                "source_row": idx + 2,
                "date": date,
                "time": clean_text(row.get("Uhrzeit")),
                "description": description,
                "category": category,
                "amount": round(float(net), 2),
                "gross": round(float(gross), 2) if gross is not None else None,
                "fee": round(float(fee), 2) if fee is not None else None,
                "currency": clean_text(row.get("Währung")) or "EUR",
                "balance": parse_amount(row.get("Guthaben")),
                "transaction_code": transaction_code,
                "related_code": related_code,
                "counterparty": name_value or email,
                "email": email,
                "invoice_number": invoice_number,
                "platform": detect_platform(description, name_value, email, invoice_number),
                "direction": "income" if net > 0 else "expense" if net < 0 else "zero",
                "text": text_blob,
                "raw_summary": combine_fields(row, row.index[: min(len(row.index), 18)]),
            }
        )

    frame = pd.DataFrame(rows)
    frame = enrich_paypal_conversions(frame)
    if not frame.empty:
        counts = frame["category"].value_counts().to_dict()
        issues.append(f"{name}: {len(frame)} PayPal-Zeilen importiert ({encoding}); Kategorien: {counts}.")
    else:
        issues.append(f"{name}: PayPal-CSV gelesen, aber keine verwertbaren Transaktionen gefunden.")
    return ParseResult(frame, issues)


def enrich_paypal_conversions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "transaction_code" not in frame.columns or "related_code" not in frame.columns:
        return frame
    result = frame.copy()
    commercial = result[result["category"].eq("commercial") & result["transaction_code"].astype(str).ne("")]
    by_code = {row["transaction_code"]: row for _, row in commercial.iterrows()}
    for idx, row in result[result["category"].eq("conversion")].iterrows():
        related = str(row.get("related_code") or "")
        source = by_code.get(related)
        if source is None:
            continue
        for column in ["counterparty", "email", "invoice_number", "platform"]:
            if column in result.columns and not clean_text(result.at[idx, column]):
                result.at[idx, column] = source.get(column, "")
        result.at[idx, "text"] = " ".join(
            clean_text(value)
            for value in [row.get("text"), source.get("counterparty"), source.get("description"), source.get("transaction_code")]
            if clean_text(value)
        )
    return result


def classify_paypal_description(description: str) -> str:
    lower = normalize_for_category(description)
    transfer_terms = [
        "bankgutschrift auf paypal konto",
        "allgemeine abbuchung bankkonto",
        "abbuchung bankkonto",
        "bankkonto",
    ]
    if any(term in lower for term in transfer_terms):
        return "bank_transfer"
    if "wahrungsumrechnung" in lower or "waehrungsumrechnung" in lower:
        return "conversion"
    if "entgelt" in lower or "gebuhr" in lower or "gebuehr" in lower:
        return "fee"
    return "commercial"


def normalize_for_category(value: object) -> str:
    text = clean_text(value).lower()
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("–", " ")
        .replace("-", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_accountable_invoices(raw: pd.DataFrame, file_name: str) -> pd.DataFrame:
    rows = []
    raw = raw.dropna(how="all").reset_index(drop=True)
    for idx, row in raw.iterrows():
        amount = parse_amount(row.get("Gesamtbetrag"))
        date_value, date_source = first_existing_with_name(row, ["Rechnungsdatum", "Zahlungsdatum", "Fälligkeitsdatum"])
        date = parse_date(date_value)
        if amount is None or pd.isna(date):
            continue
        doc_ref = clean_text(row.get("Einkommenszahl")) or f"invoice-{idx + 1}"
        counterparty = clean_text(row.get("Kundenname"))
        description = combine_fields(row, ["Name Des Artikels", "Status", "USt.-Zeitraum"])
        doc_id = stable_id("doc", file_name, "Rechnungen", idx, doc_ref, round(amount, 2))
        rows.append(
            {
                "doc_id": doc_id,
                "doc_ref": doc_ref,
                "doc_type": "income",
                "source_file": file_name,
                "source_sheet": "Rechnungen",
                "source_row": idx + 2,
                "date": date,
                "date_source": date_source,
                "amount": round(abs(float(amount)), 2),
                "signed_amount": round(abs(float(amount)), 2),
                "currency": clean_text(row.get("Währung")) or "EUR",
                "counterparty": counterparty,
                "description": description,
                "platform": detect_platform(counterparty, description),
                "status": clean_text(row.get("Status")),
                "document_available": "",
                "tax_period": clean_text(row.get("USt.-Zeitraum")),
                "text": f"{doc_ref} {counterparty} {description}".strip(),
                "raw_summary": combine_fields(row, row.index[: min(len(row.index), 18)]),
            }
        )
    return pd.DataFrame(rows)


def normalize_accountable_expenses(raw: pd.DataFrame, file_name: str) -> pd.DataFrame:
    rows = []
    raw = raw.dropna(how="all").reset_index(drop=True)
    for idx, row in raw.iterrows():
        amount = parse_amount(row.get("Gesamtbetrag"))
        date_value, date_source = first_existing_with_name(row, ["Buchungsdatum", "Zahlungsdatum", "Kodierungsdatum"])
        date = parse_date(date_value)
        if amount is None or pd.isna(date):
            continue
        vendor = clean_text(row.get("Name Des Lieferanten")) or "Unbekannter Lieferant"
        code = clean_text(row.get("Buchungscode"))
        doc_ref = f"expense-{idx + 1}" if not code else f"expense-{idx + 1}-{code}"
        description = combine_fields(row, ["Codebeschreibung", "Tags & Notizen", "USt.-Zeitraum", "Bezahlt"])
        doc_id = stable_id("doc", file_name, "Ausgaben", idx, vendor, round(amount, 2))
        rows.append(
            {
                "doc_id": doc_id,
                "doc_ref": doc_ref,
                "doc_type": "expense",
                "source_file": file_name,
                "source_sheet": "Ausgaben",
                "source_row": idx + 2,
                "date": date,
                "date_source": date_source,
                "amount": round(abs(float(amount)), 2),
                "signed_amount": round(-abs(float(amount)), 2),
                "currency": clean_text(row.get("Währung")) or "EUR",
                "counterparty": vendor,
                "description": description,
                "platform": detect_platform(vendor, description),
                "status": clean_text(row.get("Bezahlt")),
                "document_available": clean_text(row.get("Dokument Vorhanden")),
                "tax_period": clean_text(row.get("USt.-Zeitraum")),
                "text": f"{doc_ref} {vendor} {description}".strip(),
                "raw_summary": combine_fields(row, row.index[: min(len(row.index), 18)]),
            }
        )
    return pd.DataFrame(rows)
