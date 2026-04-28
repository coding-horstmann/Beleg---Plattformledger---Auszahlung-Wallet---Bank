from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reconcile.evidence import build_document_evidence, build_open_bank_after_evidence
from reconcile.matching import MatchSettings, reconcile
from reconcile.parsers import parse_accountable_file, parse_bank_file, parse_paypal_file
from reconcile.paypal import match_docs_to_paypal, match_paypal_transfers_to_bank
from reconcile.platforms import assign_platform_payout_ids, match_docs_to_platform, match_platform_payouts_to_bank, parse_platform_file


def main() -> int:
    accountable_path = ROOT / "Daten.xlsx"
    bank_paths = [
        ROOT / "fyrst.csv",
    ]
    paypal_path = ROOT / "paypal.CSV"
    platform_paths = [
        ROOT / "ebay bestellungen.csv",
        ROOT / "EtsySoldOrderItems2026-2.csv",
        ROOT / "shopify einfache csv bestellungen.csv",
        Path.home() / "Desktop" / "alt" / "rechnunungstool" / "ebay bestellungen.csv",
        Path.home() / "Desktop" / "alt" / "rechnunungstool" / "EtsySoldOrderItems2026-2.csv",
        Path.home() / "Desktop" / "alt" / "rechnunungstool" / "shopify einfache csv bestellungen.csv",
    ]

    if not accountable_path.exists():
        print("Daten.xlsx fehlt im Arbeitsordner.")
        return 2

    accountable = parse_accountable_file(accountable_path.name, accountable_path.read_bytes())
    bank_frames = []
    issues = list(accountable.issues)
    for path in bank_paths:
        if not path.exists():
            continue
        parsed = parse_bank_file(path.name, path.read_bytes())
        issues.extend(parsed.issues)
        if not parsed.frame.empty:
            bank_frames.append(parsed.frame)

    bank = pd.concat(bank_frames, ignore_index=True) if bank_frames else pd.DataFrame()
    matches, links = reconcile(accountable.frame, bank, MatchSettings())
    paypal = pd.DataFrame()
    paypal_doc_matches = pd.DataFrame()
    paypal_doc_links = pd.DataFrame()
    paypal_bank_matches = pd.DataFrame()
    platform_frames = []
    platform_doc_matches = pd.DataFrame()
    platform_doc_links = pd.DataFrame()
    platform_bank_matches = pd.DataFrame()
    if paypal_path.exists():
        parsed_paypal = parse_paypal_file(paypal_path.name, paypal_path.read_bytes())
        issues.extend(parsed_paypal.issues)
        paypal = parsed_paypal.frame
        paypal_doc_matches, paypal_doc_links = match_docs_to_paypal(accountable.frame, paypal, MatchSettings())
        paypal_bank_matches = match_paypal_transfers_to_bank(paypal, bank, MatchSettings())

    seen_platform_paths: set[str] = set()
    for path in platform_paths:
        if not path.exists():
            continue
        resolved = str(path.resolve()).lower()
        if resolved in seen_platform_paths:
            continue
        seen_platform_paths.add(resolved)
        parsed_platform = parse_platform_file(path.name, path.read_bytes())
        issues.extend(parsed_platform.issues)
        if not parsed_platform.frame.empty:
            platform_frames.append(parsed_platform.frame)
    platform_transactions = pd.concat(platform_frames, ignore_index=True) if platform_frames else pd.DataFrame()
    platform_transactions = assign_platform_payout_ids(platform_transactions)
    if not platform_transactions.empty:
        platform_doc_matches, platform_doc_links = match_docs_to_platform(accountable.frame, platform_transactions, MatchSettings())
        platform_bank_matches = match_platform_payouts_to_bank(platform_transactions, bank, MatchSettings())

    evidence = build_document_evidence(
        accountable.frame,
        bank,
        paypal,
        matches,
        links,
        paypal_doc_matches,
        paypal_doc_links,
        paypal_bank_matches,
        platform_transactions,
        platform_doc_matches,
        platform_doc_links,
        platform_bank_matches,
    )
    open_doc_ids = set(evidence.loc[evidence["evidence_level"] == "offen", "doc_id"])
    open_docs = accountable.frame[accountable.frame["doc_id"].isin(open_doc_ids)].copy()
    open_bank = build_open_bank_after_evidence(bank, matches, paypal_bank_matches, platform_bank_matches)

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    matches.to_csv(out / "matches_smoke.csv", index=False, sep=";")
    links.to_csv(out / "match_links_smoke.csv", index=False, sep=";")
    open_docs.to_csv(out / "open_docs_smoke.csv", index=False, sep=";")
    open_bank.to_csv(out / "open_bank_smoke.csv", index=False, sep=";")
    paypal.to_csv(out / "paypal_smoke.csv", index=False, sep=";")
    paypal_doc_matches.to_csv(out / "paypal_doc_matches_smoke.csv", index=False, sep=";")
    paypal_doc_links.to_csv(out / "paypal_doc_links_smoke.csv", index=False, sep=";")
    paypal_bank_matches.to_csv(out / "paypal_bank_bridge_smoke.csv", index=False, sep=";")
    platform_transactions.to_csv(out / "platform_transactions_smoke.csv", index=False, sep=";")
    platform_doc_matches.to_csv(out / "platform_doc_matches_smoke.csv", index=False, sep=";")
    platform_doc_links.to_csv(out / "platform_doc_links_smoke.csv", index=False, sep=";")
    platform_bank_matches.to_csv(out / "platform_bank_bridge_smoke.csv", index=False, sep=";")
    evidence.to_csv(out / "document_evidence_chains_smoke.csv", index=False, sep=";")

    print("\n".join(issues))
    print(
        {
            "docs": len(accountable.frame),
            "bank": len(bank),
            "matches": len(matches),
            "linked_docs": links["doc_id"].nunique() if not links.empty else 0,
            "evidence_complete": int((evidence["evidence_level"] == "vollständig belegt").sum()) if not evidence.empty else 0,
            "evidence_partial": int((evidence["evidence_level"] == "teilweise belegt").sum()) if not evidence.empty else 0,
            "open_docs": len(open_docs),
            "open_bank": len(open_bank),
            "paypal": len(paypal),
            "paypal_doc_matches": len(paypal_doc_matches),
            "paypal_doc_links": paypal_doc_links["doc_id"].nunique() if not paypal_doc_links.empty else 0,
            "paypal_bank_bridge": len(paypal_bank_matches),
            "platform": len(platform_transactions),
            "platform_doc_matches": len(platform_doc_matches),
            "platform_doc_links": platform_doc_links["doc_id"].nunique() if not platform_doc_links.empty else 0,
            "platform_bank_bridge": len(platform_bank_matches),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
