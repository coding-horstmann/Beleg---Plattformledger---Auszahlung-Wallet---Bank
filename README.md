# Buchhaltungs Buddy

Lokales Tool zum Abgleich von Accountable-Belegen mit Bankumsätzen aus FYRST und optionalen Plattformdetails wie PayPal.

## Was das Tool macht

- Importiert Accountable Excel-Dateien mit den Blättern `Rechnungen` und `Ausgaben`.
- Importiert Bank-CSV-Dateien von FYRST; unbekannte CSVs werden vorsichtig generisch erkannt.
- Importiert optional PayPal-CSV-Dateien als Zwischenkonto. PayPal wird nicht als zusätzliches Bankkonto gezählt, sondern als Brücke zwischen Belegdetails und Banktransfer ausgewertet.
- Importiert optional Plattform-CSV-Dateien als Detailspur:
  - Etsy `SoldOrderItems`
  - eBay Transaktionsbericht
  - Shopify Orders
- Nutzt Plattformdaten als Stütze für Bestellung, Kunde, Gebühr und ggf. Auszahlung. Plattformdaten sind kein weiteres Bankkonto.
- Normalisiert alle Beträge als Vorzeichenlogik: Einnahmen positiv, Ausgaben negativ.
- Verknüpft zuerst deterministisch:
  - 1:1 Match nach Betrag, Datum und Textähnlichkeit.
  - Batch-Match für Sammelzahlungen.
  - Netto-Auszahlungen, bei denen eine Differenz als mögliche Plattform-/Zahlungsgebühr markiert wird.
- Zeigt offene Belege und offene Bankumsätze zur manuellen Prüfung.
- Zeigt PayPal-Brücken separat: Accountable-Beleg zu PayPal-Detail und PayPal-Transfer zu Bankumsatz.
- Kann einzelne Restfälle per OpenRouter API prüfen lassen.
- Exportiert Matches, Links und normalisierte Daten als CSV oder speichert einen Run lokal in SQLite.
- Kann geladene Daten über `Geladene Daten zurücksetzen` aus der App-Session entfernen. Originaldateien werden dabei nicht gelöscht.

## Start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Wenn du die mitgelieferten Beispieldateien im Arbeitsordner behalten hast, lädt die App `Daten.xlsx`, `fyrst.csv`, optional `paypal.CSV` und, falls vorhanden, die Beispiel-CSV-Dateien für Etsy/eBay/Shopify. Sonst lade im Sidebar-Import eine Accountable-Datei und deine FYRST-Bank-CSV hoch.

PayPal-CSV ist optional. Sie ist besonders hilfreich, wenn PayPal selbst Sammelzahlungen, Gebühren, Währungsumrechnungen oder Lastschrift-Finanzierungen enthält. Lade sie nicht als Bank-CSV hoch, sondern in das separate PayPal-Feld.

Etsy-/eBay-/Shopify-CSV-Dateien sind fachlich sinnvoll, wenn sie Auszahlungen, Gebühren und Bestellbezug enthalten. Sie werden im Feld `Plattform-CSV optional` geladen. Die App nutzt sie als Plattform-Detaildaten, nicht als weiteres Bankkonto. Für einen echten Match müssen die Zeiträume zusammenpassen: 2026-Plattformdateien können 2025-Accountable-Belege nicht belastbar belegen.

## OpenRouter

Der API-Key wird nur im Streamlit-Formular verwendet und nicht gespeichert. Die App sendet Daten erst, wenn du im Tab `KI-Fallback` auf `KI-Vorschlag erzeugen` klickst. Standardmäßig werden Freitextfelder grob anonymisiert; Beträge, Daten, IDs und Plattformnamen bleiben erhalten.

Das normale Komplett-Matching nutzt keine KI. Das ist Absicht: Summen, Zeitfenster, Gebührenlücken und Plattform-Batches sollen reproduzierbar berechnet werden. KI ist für Restfälle und Begründungen gedacht, nicht als unsichtbarer Entscheider.

Empfohlener Ablauf:

1. Importieren.
2. Deterministische Matches prüfen.
3. Niedrige Konfidenzen und offene Fälle manuell ansehen.
4. KI nur für Restfälle verwenden.
5. Export oder SQLite-Speicherung erstellen.

## Grenzen

Das Tool ersetzt keine Steuerberatung und keine GoBD-/Verfahrensdokumentation. Es hilft dir, Nachweise strukturiert vorzubereiten: welcher Beleg wurde durch welchen Bankumsatz oder welche Sammelzahlung plausibel belegt.
