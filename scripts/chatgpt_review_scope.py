from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from reconcile.matching import MatchSettings, reconcile, unmatched_bank
from reconcile.parsers import parse_accountable_file, parse_bank_file, parse_paypal_file
from reconcile.paypal import match_docs_to_paypal, match_paypal_transfers_to_bank


def main() -> int:
    settings = MatchSettings()
    accountable = parse_accountable_file("Daten.xlsx", (ROOT / "Daten.xlsx").read_bytes()).frame
    bank = parse_bank_file("fyrst.csv", (ROOT / "fyrst.csv").read_bytes()).frame
    paypal = parse_paypal_file("paypal.CSV", (ROOT / "paypal.CSV").read_bytes()).frame

    matches, links = reconcile(accountable, bank, settings)
    paypal_doc_matches, paypal_doc_links = match_docs_to_paypal(accountable, paypal, settings)
    paypal_bank_matches = match_paypal_transfers_to_bank(paypal, bank, settings)

    linked_docs = set(links["doc_id"]) | set(paypal_doc_links["doc_id"])
    linked_bank = set(matches["tx_id"]) | set(paypal_bank_matches["tx_id"])
    unresolved_docs = accountable[~accountable["doc_id"].isin(linked_docs)].copy()
    unresolved_bank = bank[~bank["tx_id"].isin(linked_bank)].copy()

    rows: list[dict[str, object]] = []
    for _, doc in unresolved_docs.iterrows():
        platform = str(doc.get("platform") or "")
        rows.append(
            {
                "case_type": "open_accountable_doc",
                "case_id": doc["doc_id"],
                "date": doc["date"],
                "amount": doc["signed_amount"],
                "platform": platform,
                "counterparty": doc["counterparty"],
                "reference": doc["doc_ref"],
                "recommendation": doc_recommendation(doc),
                "confidence": "review",
                "reason": doc_reason(doc),
            }
        )

    for _, tx in unresolved_bank.iterrows():
        platform = str(tx.get("platform") or "")
        rows.append(
            {
                "case_type": "open_fyrst_transaction",
                "case_id": tx["tx_id"],
                "date": tx["date"],
                "amount": tx["amount"],
                "platform": platform,
                "counterparty": tx["counterparty"],
                "reference": tx["description"],
                "recommendation": bank_recommendation(tx),
                "confidence": "review",
                "reason": bank_reason(tx),
            }
        )

    review = pd.DataFrame(rows).sort_values(["case_type", "date", "amount"])
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    review.to_csv(out / "chatgpt_review_open_cases.csv", sep=";", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"metric": "accountable_docs_total", "value": len(accountable)},
            {"metric": "fyrst_transactions_total", "value": len(bank)},
            {"metric": "paypal_rows_total", "value": len(paypal)},
            {"metric": "docs_linked_by_bank", "value": links["doc_id"].nunique()},
            {"metric": "docs_linked_by_paypal", "value": paypal_doc_links["doc_id"].nunique()},
            {"metric": "docs_linked_union", "value": len(linked_docs)},
            {"metric": "docs_still_open", "value": len(unresolved_docs)},
            {"metric": "fyrst_still_open", "value": len(unresolved_bank)},
        ]
    )
    summary.to_csv(out / "chatgpt_review_summary.csv", sep=";", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print("review_file", out / "chatgpt_review_open_cases.csv")
    return 0


def doc_recommendation(doc: pd.Series) -> str:
    platform = str(doc.get("platform") or "")
    doc_type = str(doc.get("doc_type") or "")
    if doc_type == "income" and platform in {"redbubble", "juniqe"}:
        return "Plattform-Auszahlung suchen; nicht blind mit Etsy/eBay/FYRST-Batch matchen."
    if doc_type == "income":
        return "Ohne Plattform-/Shop-Zahlungsdaten offen lassen oder mit passender Auszahlungsreferenz manuell prüfen."
    if platform in {"etsy", "ebay"}:
        return "Wahrscheinlich Marktplatzgebühr/netted payout; Etsy/eBay-Abrechnung nötig."
    if platform in {"printful", "gelato"}:
        return "Wahrscheinlich Produktions-/Fulfillmentkosten; PayPal/FYRST-Detail prüfen, sonst Lieferantenexport nötig."
    if platform == "ionos":
        return "Software/Hosting-Ausgabe; Zahlungsweg prüfen, falls nicht in FYRST/PayPal sichtbar."
    return "Kein belastbarer automatischer Match; manuell klassifizieren oder passenden Zahlungsnachweis nachladen."


def doc_reason(doc: pd.Series) -> str:
    platform = str(doc.get("platform") or "")
    if platform:
        return f"Beleg ist als {platform} erkennbar, aber nach Bank- und PayPal-Brücke nicht eindeutig belegt."
    return "Kein Plattformsignal im Belegtext und kein eindeutiger Betrag-/Datums-Treffer in FYRST oder PayPal."


def bank_recommendation(tx: pd.Series) -> str:
    platform = str(tx.get("platform") or "")
    amount = float(tx.get("amount") or 0)
    if platform in {"etsy", "ebay"} and amount > 0:
        return "Plattform-Sammelauszahlung. Für echtes Matching Etsy/eBay-CSV mit Orders, Gebühren und Auszahlungs-ID nachladen."
    if platform in {"etsy", "ebay"} and amount < 0:
        return "Plattformgebühr/Lastschrift. Mit Marktplatzabrechnung oder passendem Accountable-Ausgabenbeleg prüfen."
    if "STEUERVERWALTUNG" in str(tx.get("counterparty") or "").upper():
        return "Steuerzahlung/-erstattung; nicht mit Umsatzbelegen matchen, sondern Steuerkonto/Buchung prüfen."
    if amount < 0 and "werbung" in str(tx.get("description") or "").lower():
        return "Mögliche Umbuchung/Privat- oder Werbebudget-Zahlung; nicht als Plattformbeleg automatisch matchen."
    return "Kein sicherer Auto-Match. Manuell prüfen oder fehlende Plattform-/Zahlungsdaten nachladen."


def bank_reason(tx: pd.Series) -> str:
    platform = str(tx.get("platform") or "")
    if platform in {"etsy", "ebay"}:
        return f"FYRST-Text zeigt {platform}; Bankbetrag ist wahrscheinlich aggregiert und enthält nicht alle Einzelbelege."
    return "Banktext/Betrag hat nach den gelieferten Dateien keinen eindeutigen Gegenbeleg."


if __name__ == "__main__":
    raise SystemExit(main())
