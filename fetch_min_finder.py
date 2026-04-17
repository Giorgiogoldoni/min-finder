#!/usr/bin/env python3
"""
RAPTOR MIN FINDER — fetch_min_finder.py v2.0
═════════════════════════════════════════════
Scansione incrementale a 3 passate con checkpoint.

Passata 1: tutti i 2.253 ETF  — batch 25, delay 2s
Passata 2: solo i falliti P1  — batch 10, delay 4s
Passata 3: solo i falliti P2  — batch  5, delay 6s

Dopo 3 fallimenti consecutivi → ticker in blacklist permanente.
Output: data/min_finder.json
        data/min_finder_checkpoint.json
        data/min_finder_blacklist.json
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

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUT_FILE   = DATA_DIR / "min_finder.json"
CKPT_FILE  = DATA_DIR / "min_finder_checkpoint.json"
BLIST_FILE = DATA_DIR / "min_finder_blacklist.json"
ETF_FILE   = DATA_DIR / "etf_universe.json"

SOGLIA_MIN_STORICO  = 0.05
SOGLIA_PULLBACK_MIN = 0.05
SOGLIA_PULLBACK_MAX = 0.15
SOGLIA_ATR_COMPRES  = 0.30
SOGLIA_MIN_RELATIVO = 0.08
MAX_FAILS           = 3

PASSATE = [
    (25, 2.0, "Passata 1 — tutti"),
    (10, 4.0, "Passata 2 — falliti P1"),
    ( 5, 6.0, "Passata 3 — falliti P2"),
]

def load_universe():
    if not ETF_FILE.exists():
        print("  ⚠ etf_universe.json non trovato!")
        return []
    with open(ETF_FILE) as f:
        data = json.load(f)
    universe = []
    for d in data:
        t = str(d.get('TICKER', '')).strip()
        if not t or len(t) < 2 or t == 'nan': continue
        universe.append({
            "t": t,
            "n": str(d.get('NOME', '')).strip().replace('nan', '') or t,
            "b": str(d.get('BORSA', '')).strip().replace('nan', '') or '',
            "c": str(d.get('CATEGORIA', '')).strip().replace('nan', '') or 'Altro',
        })
    print(f"  Universo: {len(universe)} ETF")
    return universe

def load_checkpoint():
    if CKPT_FILE.exists():
        try:
            with open(CKPT_FILE) as f:
                ck = json.load(f)
            print(f"  Checkpoint: {len(ck.get('prices',{}))} prezzi · {len(ck.get('fails',{}))} fail")
            return ck
        except Exception as e:
            print(f"  ⚠ Checkpoint corrotto: {e}")
    return {"prices": {}, "fails": {}}

def save_checkpoint(ck):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_FILE, "w") as f:
        json.dump(ck, f, separators=(',', ':'))

def load_blacklist():
    if BLIST_FILE.exists():
        try:
            with open(BLIST_FILE) as f:
                return set(json.load(f))
        except: pass
    return set()

def save_blacklist(blist):
    with open(BLIST_FILE, "w") as f:
        json.dump(sorted(blist), f)

def fetch_batch(tickers: list) -> dict:
    result = {}
    if not tickers: return result
    try:
        if len(tickers) == 1:
            hist = yf.Ticker(tickers[0]).history(period="1y", auto_adjust=True)
            if not hist.empty and 'Close' in hist.columns:
                closes = hist['Close'].dropna()
                if len(closes) > 20:
                    result[tickers[0]] = closes
        else:
            data = yf.download(
                tickers, period="1y", interval='1d',
                group_by='ticker', auto_adjust=True,
                progress=False, threads=True
            )
            for t in tickers:
                try:
                    if hasattr(data.columns, 'levels') and t in data.columns.get_level_values(0):
                        closes = data[t]['Close'].dropna()
                    elif not hasattr(data.columns, 'levels') and len(tickers) == 1:
                        closes = data['Close'].dropna()
                    else:
                        continue
                    if len(closes) > 20:
                        result[t] = closes
                except Exception:
                    pass
    except Exception as e:
        err = str(e)
        if 'Rate' in err or 'Too Many' in err or '429' in err:
            print(f"  ⚠ Rate limit! Attendo 45s...")
            time.sleep(45)
    return result

def scan_incremental(tickers_todo, ck, batch_size, delay, name):
    total   = len(tickers_todo)
    batches = [tickers_todo[i:i+batch_size] for i in range(0, total, batch_size)]
    n_ok    = 0
    falliti = []

    print(f"\n  {name}: {total} ticker · {len(batches)} batch · size={batch_size} · delay={delay}s")

    for i, batch in enumerate(batches):
        closes = fetch_batch(batch)
        ok_this = 0
        for t in batch:
            if t in closes:
                arr = closes[t].values.astype(float)
                ck["prices"][t] = {"values": arr.tolist()}
                ck["fails"].pop(t, None)
                n_ok += 1
                ok_this += 1
            else:
                ck["fails"][t] = ck["fails"].get(t, 0) + 1
                falliti.append(t)

        if (i + 1) % 5 == 0 or i == len(batches) - 1:
            print(f"  Batch {i+1}/{len(batches)} → ok cumulativo: {n_ok}")
            save_checkpoint(ck)

        if i < len(batches) - 1:
            time.sleep(delay)

    save_checkpoint(ck)
    print(f"  ✅ {name}: {n_ok} nuovi · {len(falliti)} falliti")
    return ck, falliti

def calc_atr(arr, period=14):
    if len(arr) < period + 1: return None, None
    diffs = np.abs(np.diff(arr))
    return float(np.mean(diffs[-period:])), float(np.mean(diffs[-63:]) if len(diffs) >= 63 else np.mean(diffs))

def calc_ma(arr, period):
    if len(arr) < period: return None
    return float(np.mean(arr[-period:]))

def analyze_etf(ticker, price_data, etf_info):
    try:
        arr   = np.array(price_data["values"], dtype=float)
        n     = len(arr)
        price = float(arr[-1])
        if price <= 0 or n < 50: return None

        low_52w  = float(np.min(arr[-252:])) if n >= 252 else float(np.min(arr))
        high_52w = float(np.max(arr[-252:])) if n >= 252 else float(np.max(arr))
        dist_low = (price - low_52w) / low_52w if low_52w > 0 else 1.0

        ret_1w = float(price / arr[-6]  - 1) if n > 6  else None
        ret_4w = float(price / arr[-21] - 1) if n > 21 else None
        ret_3m = float(price / arr[-63] - 1) if n > 63 else None

        ma50  = calc_ma(arr, 50)
        ma200 = calc_ma(arr, 200)
        ma200_rising = False
        if ma200 and n >= 210:
            ma200_prev = calc_ma(arr[:-5], 200)
            ma200_rising = bool(ma200 > (ma200_prev or 0))

        atr_c, atr_3m = calc_atr(arr, 14)
        atr_ratio = float(atr_c / atr_3m) if atr_c and atr_3m and atr_3m > 0 else None

        high_4w     = float(np.max(arr[-21:])) if n > 21 else price
        drawdown_4w = float((high_4w - price) / high_4w) if high_4w > 0 else 0.0

        is1 = bool(dist_low <= SOGLIA_MIN_STORICO)
        s1  = int(100 - dist_low / SOGLIA_MIN_STORICO * 50) if is1 else 0

        is2 = bool(
            SOGLIA_PULLBACK_MIN <= drawdown_4w <= SOGLIA_PULLBACK_MAX and
            ma200 is not None and price > ma200 * 0.95 and ma200_rising
        )
        s2 = 0
        if is2:
            d = abs(price - ma200) / ma200 if ma200 else 1
            s2 = int(min(100, 70 + (1 - d * 10) * 30))

        is3 = bool(atr_ratio is not None and atr_ratio < (1 - SOGLIA_ATR_COMPRES))
        s3  = int(min(100, (1 - atr_ratio) * 150)) if is3 else 0

        n_cat = sum([is1, is2, is3])
        sm    = max(s1, s2, s3)

        return {
            "ticker":           ticker,
            "name":             etf_info.get("n", ticker),
            "borsa":            etf_info.get("b", ""),
            "categoria":        etf_info.get("c", ""),
            "price":            round(price, 4),
            "low_52w":          round(low_52w, 4),
            "high_52w":         round(high_52w, 4),
            "dist_52w_low":     round(dist_low * 100, 2),
            "dist_52w_high":    round((high_52w - price) / high_52w * 100, 2) if high_52w > 0 else 0,
            "ret_1w":           round(ret_1w * 100, 2) if ret_1w is not None else None,
            "ret_4w":           round(ret_4w * 100, 2) if ret_4w is not None else None,
            "ret_3m":           round(ret_3m * 100, 2) if ret_3m is not None else None,
            "ma50":             round(ma50, 4)  if ma50  else None,
            "ma200":            round(ma200, 4) if ma200 else None,
            "ma200_rising":     ma200_rising,
            "drawdown_4w":      round(drawdown_4w * 100, 2),
            "atr_ratio":        round(atr_ratio, 3) if atr_ratio else None,
            "is_min_storico":   is1,
            "is_pullback":      is2,
            "is_compressione":  is3,
            "is_min_relativo":  False,
            "score_min_storico":   s1,
            "score_pullback":      s2,
            "score_compressione":  s3,
            "score_min_relativo":  0,
            "score_master":        sm,
            "n_categorie":         n_cat,
            "cat_mean_ret3m":      0.0,
        }
    except Exception:
        return None

def compute_min_relativi(results):
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        if r and r.get('ret_3m') is not None:
            by_cat[r['categoria']].append(r['ret_3m'])
    cat_means = {c: float(np.mean(v)) for c, v in by_cat.items() if len(v) >= 5}
    for r in results:
        if not r: continue
        cat  = r['categoria']
        ret3 = r.get('ret_3m')
        mean = cat_means.get(cat, 0.0)
        r['cat_mean_ret3m'] = round(mean, 2)
        if ret3 is not None and cat in cat_means:
            diff = mean - ret3
            if diff >= SOGLIA_MIN_RELATIVO * 100:
                r['is_min_relativo']    = True
                r['score_min_relativo'] = int(min(100, diff * 5))
                r['n_categorie']       += 1
                r['score_master']       = max(r['score_master'], r['score_min_relativo'])
    return results

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main():
    t0 = time.time()
    print("=" * 60)
    print(f"RAPTOR MIN FINDER v2.0 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    universe    = load_universe()
    ticker_map  = {e["t"]: e for e in universe}
    all_tickers = [e["t"] for e in universe]
    ck          = load_checkpoint()
    blist       = load_blacklist()

    # ── 3 PASSATE ────────────────────────────────────────────────
    already   = set(ck["prices"].keys())
    todo_p1   = [t for t in all_tickers if t not in already and t not in blist]
    print(f"\n  Cache: {len(already)} · Da scaricare: {len(todo_p1)} · Blacklist: {len(blist)}")

    falliti = todo_p1
    for idx, (bs, dl, nm) in enumerate(PASSATE):
        if not falliti:
            print(f"\n  ⚡ {nm} saltata — nessun ticker da riprovare")
            continue
        # Filtra quelli già scaricati in una passata precedente di questo run
        todo = [t for t in falliti if t not in ck["prices"] and t not in blist]
        if not todo:
            print(f"\n  ⚡ {nm} saltata — tutti già scaricati")
            falliti = []
            continue
        ck, falliti = scan_incremental(todo, ck, bs, dl, nm)

    # ── AGGIORNA BLACKLIST ────────────────────────────────────────
    new_bl = 0
    for t, n in ck["fails"].items():
        if n >= MAX_FAILS and t not in blist:
            blist.add(t); new_bl += 1
    if new_bl:
        print(f"\n  🚫 Blacklist: +{new_bl} ticker (≥{MAX_FAILS} fail)")
        save_blacklist(blist)

    # ── ANALISI ───────────────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    print(f"\n  Prezzi in cache: {len(ck['prices'])}/{len(all_tickers)} · Tempo: {elapsed:.1f} min")
    print("  Analisi ETF...")

    results = []
    for ticker, pd in ck["prices"].items():
        info = ticker_map.get(ticker, {"n": ticker, "b": "", "c": "Altro"})
        r = analyze_etf(ticker, pd, info)
        if r: results.append(r)
    print(f"  Analizzati: {len(results)} ETF")

    results = compute_min_relativi(results)

    lista1 = sorted([r for r in results if r['is_min_storico']],  key=lambda x: x['score_min_storico'],  reverse=True)
    lista2 = sorted([r for r in results if r['is_pullback']],     key=lambda x: x['score_pullback'],     reverse=True)
    lista3 = sorted([r for r in results if r['is_compressione']], key=lambda x: x['score_compressione'], reverse=True)
    lista4 = sorted([r for r in results if r['is_min_relativo']], key=lambda x: x['score_min_relativo'], reverse=True)
    master  = sorted([r for r in results if r['n_categorie'] >= 2], key=lambda x: x['score_master'],     reverse=True)
    cands   = list({r['ticker'] for r in lista1 + lista2 + lista3 + lista4 + master})

    print(f"\n  📉 {len(lista1)} · 🔄 {len(lista2)} · 🔥 {len(lista3)} · 📊 {len(lista4)} · ⭐ {len(master)}")

    output = make_serializable({
        "generated":   datetime.now(timezone.utc).isoformat(),
        "version":     "2.0",
        "universe_tot": len(all_tickers),
        "analyzed":    len(results),
        "cached":      len(ck["prices"]),
        "blacklisted": len(blist),
        "candidates":  cands,
        "stats": {
            "min_storico": len(lista1), "pullback": len(lista2),
            "compressione": len(lista3), "min_relativo": len(lista4), "master": len(master),
        },
        "soglie": {
            "min_storico_pct": SOGLIA_MIN_STORICO*100,
            "pullback_min_pct": SOGLIA_PULLBACK_MIN*100,
            "pullback_max_pct": SOGLIA_PULLBACK_MAX*100,
            "atr_compres_pct": SOGLIA_ATR_COMPRES*100,
            "min_relativo_pct": SOGLIA_MIN_RELATIVO*100,
        },
        "lista1_min_storico":  lista1[:200],
        "lista2_pullback":     lista2[:200],
        "lista3_compressione": lista3[:200],
        "lista4_min_relativo": lista4[:200],
        "lista_master":        master[:100],
        "elapsed_min":         round(elapsed, 1),
    })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

    print(f"\n✅ {OUT_FILE.name} ({OUT_FILE.stat().st_size/1024:.0f} KB) · {elapsed:.1f} min totali")

if __name__ == "__main__":
    main()
