from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import pandas as pd


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


MASTER_SYSTEM_PROMPT = """Du bist ein vorsichtiger Reconciliation-Assistent fuer einen Solo-Unternehmer.

Aufgabe:
- Ordne Banktransaktionen zu Accountable-Belegen zu.
- Bevorzuge deterministische Evidenz: Betrag, Vorzeichen, Datum, Referenzen, Plattform-/Zahlungsanbieter, Namen.
- Erkenne 1:1-Zahlungen, Sammelzahlungen, Netto-Auszahlungen mit plausiblen Plattformgebuehren und nicht passende Faelle.
- Erfinde niemals Belege, Beträge, Daten oder steuerliche Schlussfolgerungen.
- Gib nur eine Pruefempfehlung aus. Die finale Entscheidung trifft der Mensch.

Bewertungsregeln:
- Betrag und Vorzeichen muessen wirtschaftlich zusammenpassen.
- Datumsversatz ist normal, aber erklaere ihn.
- Bei Sammelzahlungen darf eine Banktransaktion mehrere Belege tragen.
- Bei Plattform-Auszahlungen darf Summe Bruttobelege minus plausible Gebuehren dem Bankbetrag entsprechen.
- Wenn die Evidenz schwach ist, senke die Konfidenz und markiere den Fall als review_needed.

Antworte strikt als JSON ohne Markdown:
{
  "decision": "match|no_match|review_needed",
  "match_type": "direct|batch_exact|batch_net_fee_gap|possible_duplicate|unknown",
  "tx_id": "...",
  "doc_ids": ["..."],
  "confidence": 0.0,
  "amount_check": {
    "bank_amount": 0.0,
    "document_sum": 0.0,
    "gap": 0.0,
    "gap_explanation": "..."
  },
  "reasoning": "...",
  "risks": ["..."]
}
"""


@dataclass
class LlmConfig:
    api_key: str
    model: str
    endpoint: str = OPENROUTER_ENDPOINT
    anonymize: bool = True
    referer: str = "http://localhost:8501"
    title: str = "Local Finance Reconciliation"
    timeout_seconds: int = 60


def suggest_match(config: LlmConfig, tx: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    payload_tx = anonymize_record(tx) if config.anonymize else tx
    payload_candidates = [anonymize_record(row) for row in candidates] if config.anonymize else candidates
    user_payload = {
        "bank_transaction": payload_tx,
        "candidate_documents": payload_candidates,
        "instruction": "Pruefe, ob die Banktransaktion mit einem oder mehreren Kandidaten belegbar ist. Nutze nur die gelieferten Daten.",
    }

    request_body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
        ],
        "temperature": 0,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        config.endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.referer,
            "X-Title": config.title,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter nicht erreichbar: {exc}") from exc

    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    return parse_json_content(content)


def parse_json_content(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def anonymize_record(record: dict[str, Any]) -> dict[str, Any]:
    keep_keys = {
        "tx_id",
        "doc_id",
        "doc_ref",
        "doc_type",
        "date",
        "bank_date",
        "amount",
        "signed_amount",
        "bank_amount",
        "doc_amount",
        "currency",
        "direction",
        "source",
        "bank_source",
        "method",
        "confidence",
    }
    sanitized: dict[str, Any] = {}
    for key, value in record.items():
        if key in keep_keys:
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = anonymize_text(value)
        else:
            sanitized[key] = value
    return sanitized


def anonymize_text(text: str) -> str:
    platform_tokens = {
        "etsy",
        "ebay",
        "paypal",
        "shopify",
        "gelato",
        "printful",
        "redbubble",
        "teepublic",
        "juniqe",
        "amazon",
        "stripe",
        "klarna",
        "ionos",
    }
    output = []
    for token in re.findall(r"[A-Za-zÄÖÜäöüß0-9_.@/-]+|\s+|.", text):
        low = token.lower()
        if low.strip() in platform_tokens:
            output.append(token)
        elif re.fullmatch(r"\s+", token):
            output.append(" ")
        elif re.search(r"\d", token) and len(token) <= 4:
            output.append(token)
        elif len(token.strip()) <= 2:
            output.append(token)
        else:
            output.append("X")
    return re.sub(r"\s+", " ", "".join(output)).strip()


def dataframe_records(frame: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    result = frame.head(limit).copy()
    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = result[col].dt.strftime("%Y-%m-%d")
    return result.to_dict(orient="records")
