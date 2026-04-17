#!/usr/bin/env python3
"""
RAPTOR MIN FINDER — update_min_finder_live.py
══════════════════════════════════════════════
Script giornaliero (09:30 e 16:00 IT) — aggiorna prezzi e segnali
SOLO per i candidati identificati dallo script notturno.

Input:  data/min_finder.json (generato da fetch_min_finder.py)
Output: data/min_finder_live.json
"""

import json, os, time
from datetime import datetime, timezone
from pathlib import Path

def install(pkg):
    os.system(f"pip install {pkg} --break-system-packages -q")

try:
    import yfinance as yf
except ImportError:
    install("yfinance"); import yfinance as yf

try:
    import numpy as np
except ImportError:
    install("numpy"); import numpy as np

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
BASE_FILE = DATA_DIR / "min_finder.json"
LIVE_FILE = DATA_DIR / "min_finder_live.json"

BATCH_SIZE  = 50
BATCH_DELAY = 1.0

def fetch_current_prices(tickers: list) -> dict:
    """Scarica prezzi degli ultimi 5 giorni per calcolare variazione live."""
    result = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        try:
            if len(batch) == 1:
                hist = yf.Ticker(batch[0]).history(period="5d", auto_adjust=True)
                if not hist.empty:
                    closes = hist['Close'].dropna()
                    if len(closes) >= 2:
                        result[batch[0]] = {
                            "price":  round(float(closes.iloc[-1]), 4),
                            "prev":   round(float(closes.iloc[-2]), 4),
                            "ret_1d": round((closes.iloc[-1]/closes.iloc[-2]-1)*100, 2),
                        }
            else:
                data = yf.download(batch, period="5d", interval="1d",
                                   group_by="ticker", auto_adjust=True,
                                   progress=False, threads=True)
                for t in batch:
                    try:
                        closes = data[t]['Close'].dropna() if len(batch) > 1 else data['Close'].dropna()
                        if len(closes) >= 2:
                            result[t] = {
                                "price":  round(float(closes.iloc[-1]), 4),
                                "prev":   round(float(closes.iloc[-2]), 4),
                                "ret_1d": round((closes.iloc[-1]/closes.iloc[-2]-1)*100, 2),
                            }
                    except: pass
        except Exception as e:
            print(f"  err batch {i+1}: {e}")
        if i < len(batches) - 1:
            time.sleep(BATCH_DELAY)
    return result

def update_entry(entry: dict, live: dict) -> dict:
    """Aggiorna un record con i prezzi live."""
    t = entry['ticker']
    if t in live:
        entry['price']     = live[t]['price']
        entry['ret_1d']    = live[t]['ret_1d']
        entry['live_updated'] = True

        # Ricalcola distanza dal 52w low con prezzo aggiornato
        if entry.get('low_52w') and entry['low_52w'] > 0:
            entry['dist_52w_low'] = round((entry['price'] - entry['low_52w']) / entry['low_52w'] * 100, 2)
    else:
        entry['ret_1d']       = None
        entry['live_updated'] = False
    return entry

def main():
    print("=" * 60)
    print(f"RAPTOR MIN FINDER LIVE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Carica base
    if not BASE_FILE.exists():
        print(f"❌ File base non trovato: {BASE_FILE}")
        print("   Esegui prima fetch_min_finder.py")
        return

    with open(BASE_FILE) as f:
        base = json.load(f)

    candidates = base.get("candidates", [])
    print(f"\n  Candidati da aggiornare: {len(candidates)}")

    # 2. Scarica prezzi live
    print(f"\n  Download prezzi live...")
    live_prices = fetch_current_prices(candidates)
    print(f"  Aggiornati: {len(live_prices)}/{len(candidates)}")

    # 3. Aggiorna tutte le liste
    now_str = datetime.now(timezone.utc).isoformat()

    def update_list(lst):
        return [update_entry(dict(e), live_prices) for e in lst]

    lista1 = update_list(base.get("lista1_min_storico",  []))
    lista2 = update_list(base.get("lista2_pullback",     []))
    lista3 = update_list(base.get("lista3_compressione", []))
    lista4 = update_list(base.get("lista4_min_relativo", []))
    master = update_list(base.get("lista_master",        []))

    # 4. Salva
    output = {
        "generated":     now_str,
        "base_generated": base.get("generated"),
        "version":       "1.0",
        "universe_tot":  base.get("universe_tot", 0),
        "analyzed":      base.get("analyzed", 0),
        "live_updated":  len(live_prices),
        "stats":         base.get("stats", {}),
        "soglie":        base.get("soglie", {}),
        "lista1_min_storico":  lista1,
        "lista2_pullback":     lista2,
        "lista3_compressione": lista3,
        "lista4_min_relativo": lista4,
        "lista_master":        master,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

    print(f"\n✅ Salvato: {LIVE_FILE}")
    print(f"   📉 Min Storico: {len(lista1)} · 🔄 Pullback: {len(lista2)} · 🔥 Compressione: {len(lista3)} · 📊 Min Relativo: {len(lista4)} · ⭐ Master: {len(master)}")

if __name__ == "__main__":
    main()
