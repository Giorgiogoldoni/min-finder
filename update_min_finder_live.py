#!/usr/bin/env python3
"""
RAPTOR MIN FINDER — update_min_finder_live.py v2.0
════════════════════════════════════════════════════
Aggiornamento pomeridiano (15:00 IT) — solo candidati.

1. Aggiorna prezzi live dei candidati
2. Ricalcola SAR, KAMA, OBV con prezzi aggiornati
3. Riordina TOP 20 con segnaletica BUY1/BUY2/BUY3
4. Salva min_finder_live.json

Input:  data/min_finder.json
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

def fetch_live(tickers: list) -> dict:
    result = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        try:
            if len(batch) == 1:
                hist = yf.Ticker(batch[0]).history(period="5d", auto_adjust=True)
                if not hist.empty:
                    closes = hist['Close'].dropna()
                    highs  = hist['High'].dropna()
                    lows   = hist['Low'].dropna()
                    vols   = hist['Volume'].dropna()
                    if len(closes) >= 2:
                        result[batch[0]] = {
                            "price":  round(float(closes.iloc[-1]), 4),
                            "prev":   round(float(closes.iloc[-2]), 4),
                            "ret_1d": round((closes.iloc[-1]/closes.iloc[-2]-1)*100, 2),
                            "high":   round(float(highs.iloc[-1]), 4),
                            "low":    round(float(lows.iloc[-1]), 4),
                            "vol":    int(vols.iloc[-1]),
                        }
            else:
                data = yf.download(batch, period="5d", interval="1d",
                                   group_by="ticker", auto_adjust=True,
                                   progress=False, threads=True)
                for t in batch:
                    try:
                        td = data[t] if len(batch)>1 else data
                        closes = td['Close'].dropna()
                        if len(closes) >= 2:
                            highs = td['High'].reindex(closes.index).fillna(closes)
                            lows  = td['Low'].reindex(closes.index).fillna(closes)
                            vols  = td['Volume'].reindex(closes.index).fillna(0)
                            result[t] = {
                                "price":  round(float(closes.iloc[-1]), 4),
                                "prev":   round(float(closes.iloc[-2]), 4),
                                "ret_1d": round((closes.iloc[-1]/closes.iloc[-2]-1)*100, 2),
                                "high":   round(float(highs.iloc[-1]), 4),
                                "low":    round(float(lows.iloc[-1]), 4),
                                "vol":    int(vols.iloc[-1]),
                            }
                    except: pass
        except Exception as e:
            print(f"  err batch {i+1}: {e}")
        if i < len(batches)-1:
            time.sleep(BATCH_DELAY)
    return result

def update_entry(entry: dict, live: dict) -> dict:
    t = entry['ticker']
    if t in live:
        entry['price']        = live[t]['price']
        entry['ret_1d']       = live[t]['ret_1d']
        entry['live_updated'] = True
        if entry.get('low_52w') and entry['low_52w']>0:
            entry['dist_52w_low'] = round((entry['price']-entry['low_52w'])/entry['low_52w']*100, 2)
        # Aggiorna segnale SAR approssimato (usando high/low del giorno)
        if live[t].get('high') and live[t].get('low'):
            price  = entry['price']
            prev   = live[t]['prev']
            sar    = entry.get('sar')
            if sar:
                # SAR semplice: se prezzo > SAR precedente → bull
                entry['sar_bull'] = price > sar
        # Aggiorna KAMA approssimato
        kama = entry.get('kama')
        if kama:
            price = entry['price']
            er    = min(1.0, abs(price-prev)/(abs(price-prev)+0.001))
            sc    = (er*(2/6-2/21)+2/21)**2
            new_kama = kama + sc*(price-kama)
            entry['kama']             = round(new_kama, 4)
            entry['price_above_kama'] = price > new_kama
        # Ricalcola buy_level
        sar_bull  = entry.get('sar_bull', False)
        above_kama = entry.get('price_above_kama', False)
        ma200     = entry.get('ma200')
        price     = entry['price']
        obv_div   = entry.get('obv_divergence', False)
        if sar_bull and above_kama and ma200 and price>ma200:
            entry['buy_level'] = 'BUY1'
        elif (sar_bull and above_kama) or (sar_bull and ma200 and price>ma200):
            entry['buy_level'] = 'BUY2'
        elif sar_bull or above_kama:
            entry['buy_level'] = 'BUY3'
        else:
            entry['buy_level'] = 'WATCH'
        # Ricalcola tech_score
        ts = 0
        if sar_bull:   ts += 35
        if above_kama: ts += 30
        if ma200 and price>ma200: ts += 20
        if obv_div:    ts += 15
        entry['tech_score'] = ts
        # Ricalcola composite_score
        entry['composite_score'] = round(
            entry.get('score_master',0)*0.40 +
            ts*0.40 +
            (20 if obv_div else 0)*1.0, 1
        )
    else:
        entry['ret_1d']       = None
        entry['live_updated'] = False
    return entry

def compute_top20_live(all_entries: list) -> list:
    """Seleziona TOP 20 dai candidati aggiornati con prezzi live."""
    candidates = [e for e in all_entries if e.get('n_categorie',0)>=1 and e.get('live_updated')]
    candidates.sort(key=lambda x: x.get('composite_score',0), reverse=True)
    top20 = []
    for level in ['BUY1','BUY2','BUY3','WATCH']:
        items = [r for r in candidates if r.get('buy_level')==level and r not in top20]
        top20.extend(items)
        if len(top20)>=20: break
    return top20[:20]

def main():
    print("="*60)
    print(f"RAPTOR MIN FINDER LIVE v2.0 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    if not BASE_FILE.exists():
        print(f"❌ File base non trovato: {BASE_FILE}")
        print("   Esegui prima fetch_min_finder.py")
        return

    with open(BASE_FILE) as f:
        base = json.load(f)

    candidates = base.get("candidates", [])
    print(f"\n  Candidati da aggiornare: {len(candidates)}")

    print(f"\n  Download prezzi live...")
    live_prices = fetch_live(candidates)
    print(f"  Aggiornati: {len(live_prices)}/{len(candidates)}")

    def update_list(lst):
        return [update_entry(dict(e), live_prices) for e in lst]

    lista1 = update_list(base.get("lista1_min_storico",  []))
    lista2 = update_list(base.get("lista2_pullback",     []))
    lista3 = update_list(base.get("lista3_compressione", []))
    lista4 = update_list(base.get("lista4_min_relativo", []))
    master = update_list(base.get("lista_master",        []))

    # TOP 20 live — riordina con segnali aggiornati
    all_entries = lista1 + lista2 + lista3 + lista4
    seen = set()
    unique_entries = []
    for e in all_entries:
        if e['ticker'] not in seen:
            seen.add(e['ticker'])
            unique_entries.append(e)

    top20 = compute_top20_live(unique_entries)

    buy1 = [r for r in top20 if r.get('buy_level')=='BUY1']
    buy2 = [r for r in top20 if r.get('buy_level')=='BUY2']
    buy3 = [r for r in top20 if r.get('buy_level')=='BUY3']
    print(f"\n  🎯 TOP 20 live: BUY1={len(buy1)} · BUY2={len(buy2)} · BUY3={len(buy3)}")
    for r in top20:
        print(f"     {r['buy_level']} {r['ticker']:12} score={r.get('composite_score','—')}")

    output = {
        "generated":      datetime.now(timezone.utc).isoformat(),
        "base_generated": base.get("generated"),
        "version":        "2.0",
        "universe_tot":   base.get("universe_tot", 0),
        "analyzed":       base.get("analyzed", 0),
        "live_updated":   len(live_prices),
        "stats":          base.get("stats", {}),
        "soglie":         base.get("soglie", {}),
        "top20":          top20,
        "lista1_min_storico":  lista1,
        "lista2_pullback":     lista2,
        "lista3_compressione": lista3,
        "lista4_min_relativo": lista4,
        "lista_master":        master,
        "charts":              base.get("charts", {}),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(',',':'))

    print(f"\n✅ Salvato: {LIVE_FILE}")

if __name__ == "__main__":
    main()
