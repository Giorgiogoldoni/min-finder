#!/usr/bin/env python3
"""compute_rev1_signals.py

Segnale REV1 — INDIPENDENTE dal motore esistente di min-finder
(calc_inversion_signals in fetch_min_finder.py, NON modificato).

Regola d'ingresso REV1 (validata fuori campione su universo core
EUR/Europa in core-backtest — non ancora verificata specificamente su
tutto l'universo esteso di questo repo, trattare con la stessa cautela
riservata a un segnale nuovo):
  RSI(14) < 35 + AO in miglioramento (ao[i] > ao[i-1])

Regola d'uscita di riferimento (per chi opera manualmente, non
applicata qui — questo script segnala solo candidati d'ingresso):
  RSI(14) torna sopra 60

Riusa la cache prezzi GIÀ PRESENTE in data/min_finder_checkpoint.json
(popolata dal motore esistente) — NESSUN nuovo fetch yfinance.
Per questo copre solo i ticker già in cache in quel momento (parziale
sull'universo totale, cresce mano a mano che il motore esistente
scansiona più strumenti nei prossimi run).

Output: data/rev1_signals.json
"""

import json
import os
import datetime
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "data", "min_finder_checkpoint.json")
UNIVERSE_PATH = os.path.join(BASE_DIR, "data", "etf_universe.json")
OUT_PATH = os.path.join(BASE_DIR, "data", "rev1_signals.json")

RSI_ENTRY_THRESHOLD = 35
RSI_EXIT_REFERENCE = 60  # solo promemoria descrittivo, non applicato qui
MIN_BARS = 50


def calc_rsi(close, n=14):
    result = [None] * len(close)
    if len(close) < n + 2:
        return result
    for i in range(n, len(close)):
        gains = losses = 0.0
        for j in range(i - n + 1, i + 1):
            d = close[j] - close[j - 1]
            if d > 0:
                gains += d
            else:
                losses += -d
        ag, al = gains / n, losses / n
        result[i] = 100 - 100 / (1 + ag / al) if al > 0 else 100.0
    return result


def calc_ao(high, low):
    mid = [(h + l) / 2 for h, l in zip(high, low)]
    result = [None] * len(mid)
    for i in range(33, len(mid)):
        sma5 = sum(mid[i - 4:i + 1]) / 5
        sma34 = sum(mid[i - 33:i + 1]) / 34
        result[i] = round(sma5 - sma34, 4)
    return result


def sanitize_nan(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    return obj


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    print(f"compute_rev1_signals.py — {now.isoformat()}")

    if not os.path.exists(CKPT_PATH):
        print(f"[ERROR] {CKPT_PATH} non trovato")
        return

    with open(CKPT_PATH, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    prices = ckpt.get("prices", {})
    print(f"Ticker in cache prezzi: {len(prices)}")

    names = {}
    if os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
            universe = json.load(f)
        for e in universe:
            tk = e.get("TICKER")
            if tk:
                names[tk] = e.get("NOME")

    candidates = []
    skipped_short = 0
    for ticker, rec in prices.items():
        closes = rec.get("c", [])
        highs = rec.get("h", [])
        lows = rec.get("l", [])
        if len(closes) < MIN_BARS:
            skipped_short += 1
            continue

        rsi = calc_rsi(closes)
        ao = calc_ao(highs, lows)

        i = len(closes) - 1
        if rsi[i] is None or ao[i] is None or ao[i - 1] is None:
            continue

        if rsi[i] < RSI_ENTRY_THRESHOLD and ao[i] > ao[i - 1]:
            base_ticker = ticker.split(".")[0]
            candidates.append({
                "ticker": ticker,
                "name": names.get(base_ticker) or names.get(ticker),
                "rsi": round(rsi[i], 2),
                "ao": ao[i],
                "ao_prev": ao[i - 1],
                "price": round(closes[i], 4),
                "date": rec.get("d", [None] * len(closes))[i] if rec.get("d") else None,
            })

    candidates.sort(key=lambda c: c["rsi"])

    output = {
        "generated_at": now.isoformat(),
        "tickers_in_cache": len(prices),
        "tickers_skipped_short_history": skipped_short,
        "rev1_count": len(candidates),
        "entry_rule": f"RSI(14) < {RSI_ENTRY_THRESHOLD} + AO in miglioramento",
        "exit_reference": f"RSI(14) torna sopra {RSI_EXIT_REFERENCE} (promemoria, non un'uscita automatica)",
        "note": (
            "Segnale sperimentale, indipendente dal motore principale di min-finder. "
            "Validato fuori campione su universo core EUR/Europa (core-backtest) — non "
            "ancora verificato specificamente su tutto l'universo esteso di questo repo. "
            "Copre solo i ticker già presenti nella cache prezzi del motore esistente "
            "in questo momento, non l'intero universo."
        ),
        "candidates": candidates,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sanitize_nan(output), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    print(f"\nCompletato: {len(candidates)} candidati REV1 su {len(prices)} ticker in cache")
    print(f"Saltati per storico insufficiente: {skipped_short}")


if __name__ == "__main__":
    main()
