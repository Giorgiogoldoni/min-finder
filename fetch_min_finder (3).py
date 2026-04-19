#!/usr/bin/env python3
"""
RAPTOR MIN FINDER — fetch_min_finder.py v3.0
═════════════════════════════════════════════
Scansione notturna completa su 2.253 ETF.

Passata 1: tutti i ticker — batch 25, delay 2s
Passata 2: falliti P1    — batch 10, delay 4s
Passata 3: falliti P2    — batch  5, delay 6s

Per ogni candidato (nelle 4 liste) calcola:
  - SAR Parabolico
  - OBV (On Balance Volume)
  - KAMA (10)
  - Bande di Bollinger (20)
  - Dati chart completi (90 barre)

Output: data/min_finder.json
        data/min_finder_checkpoint.json
        data/min_finder_blacklist.json
"""

import json, os, time, math
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


# Correzione scala per ticker con dati Yahoo anomali
# Formato: "TICKER": fattore_moltiplicativo
PRICE_SCALE_FIX = {
    "LCCN.MI": 10000,   # Amundi MSCI China — Yahoo scala /10000
}

PASSATE = [
    (25, 2.0, "Passata 1 — tutti"),
    (10, 4.0, "Passata 2 — falliti P1"),
    ( 5, 6.0, "Passata 3 — falliti P2"),
]

# ── UNIVERSE ──────────────────────────────────────────────────
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

# ── CHECKPOINT ────────────────────────────────────────────────
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

# ── FETCH ────────────────────────────────────────────────────
def fetch_batch(tickers: list) -> dict:
    result = {}
    if not tickers: return result
    try:
        if len(tickers) == 1:
            hist = yf.Ticker(tickers[0]).history(period="1y", auto_adjust=True)
            if not hist.empty and 'Close' in hist.columns:
                closes = hist['Close'].dropna()
                highs  = hist['High'].dropna()
                lows   = hist['Low'].dropna()
                vols   = hist['Volume'].dropna()
                if len(closes) > 20:
                    result[tickers[0]] = {
                        "c": closes.values.tolist(),
                        "h": highs.values.tolist(),
                        "l": lows.values.tolist(),
                        "v": vols.values.tolist(),
                        "d": [str(d.date()) for d in closes.index],
                    }
        else:
            data = yf.download(
                tickers, period="1y", interval='1d',
                group_by='ticker', auto_adjust=True,
                progress=False, threads=True
            )
            for t in tickers:
                try:
                    if hasattr(data.columns, 'levels') and t in data.columns.get_level_values(0):
                        td = data[t]
                    else:
                        continue
                    closes = td['Close'].dropna()
                    if len(closes) > 20:
                        highs = td['High'].reindex(closes.index).fillna(closes)
                        lows  = td['Low'].reindex(closes.index).fillna(closes)
                        vols  = td['Volume'].reindex(closes.index).fillna(0)
                        result[t] = {
                            "c": closes.values.tolist(),
                            "h": highs.values.tolist(),
                            "l": lows.values.tolist(),
                            "v": vols.values.tolist(),
                            "d": [str(d.date()) for d in closes.index],
                        }
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
    print(f"\n  {name}: {total} ticker · {len(batches)} batch")
    for i, batch in enumerate(batches):
        prices = fetch_batch(batch)
        for t in batch:
            if t in prices:
                ck["prices"][t] = prices[t]
                ck["fails"].pop(t, None)
                n_ok += 1
            else:
                ck["fails"][t] = ck["fails"].get(t, 0) + 1
                falliti.append(t)
        if (i + 1) % 5 == 0 or i == len(batches) - 1:
            print(f"  Batch {i+1}/{len(batches)} → ok: {n_ok}")
            save_checkpoint(ck)
        if i < len(batches) - 1:
            time.sleep(delay)
    save_checkpoint(ck)
    print(f"  ✅ {name}: {n_ok} nuovi · {len(falliti)} falliti")
    return ck, falliti

# ── INDICATORI ────────────────────────────────────────────────
def calc_atr(h, l, c, period=14):
    h, l, c = np.array(h), np.array(l), np.array(c)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr = np.concatenate([[h[0]-l[0]], tr])
    atr = np.full(len(c), np.nan)
    if len(c) >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(c)):
            atr[i] = (atr[i-1]*(period-1)+tr[i])/period
    return atr

def calc_sar(h, l, af_step=0.02, af_max=0.2):
    h, l = np.array(h), np.array(l)
    sar  = np.full(len(h), np.nan)
    bull_arr = np.zeros(len(h), dtype=bool)
    if len(h) < 2: return sar, bull_arr
    bull = True; af = af_step; ep = h[0]; sar[0] = l[0]; bull_arr[0] = True
    for i in range(1, len(h)):
        prev = sar[i-1]
        if bull:
            sar[i] = prev + af*(ep-prev)
            sar[i] = min(sar[i], l[i-1], l[i-2] if i>1 else l[i-1])
            if l[i] < sar[i]:
                bull=False; af=af_step; ep=l[i]; sar[i]=ep
            else:
                if h[i]>ep: ep=h[i]; af=min(af+af_step,af_max)
        else:
            sar[i] = prev + af*(ep-prev)
            sar[i] = max(sar[i], h[i-1], h[i-2] if i>1 else h[i-1])
            if h[i] > sar[i]:
                bull=True; af=af_step; ep=h[i]; sar[i]=ep
            else:
                if l[i]<ep: ep=l[i]; af=min(af+af_step,af_max)
        bull_arr[i] = bull
    return sar, bull_arr

def calc_kama(c, n=10, fast=5, slow=20):
    c = np.array(c, dtype=float)
    fs, ss = 2/(fast+1), 2/(slow+1)
    kama = np.full(len(c), np.nan)
    if len(c) <= n: return kama
    kama[n] = c[n]
    for i in range(n+1, len(c)):
        direction  = abs(c[i]-c[i-n])
        volatility = np.sum(np.abs(np.diff(c[i-n:i+1])))
        er  = direction/volatility if volatility != 0 else 0
        sc  = (er*(fs-ss)+ss)**2
        kama[i] = kama[i-1] + sc*(c[i]-kama[i-1])
    return kama

def calc_obv(c, v):
    c, v = np.array(c), np.array(v)
    obv = np.zeros(len(c))
    for i in range(1, len(c)):
        if   c[i] > c[i-1]: obv[i] = obv[i-1] + v[i]
        elif c[i] < c[i-1]: obv[i] = obv[i-1] - v[i]
        else:                obv[i] = obv[i-1]
    return obv

def calc_ma(arr, period):
    arr = np.array(arr)
    if len(arr) < period: return None
    return float(np.mean(arr[-period:]))

def calc_bb(c, period=20, n_std=2.0):
    c = np.array(c)
    upper = np.full(len(c), np.nan)
    lower = np.full(len(c), np.nan)
    mid   = np.full(len(c), np.nan)
    for i in range(period-1, len(c)):
        w = c[i-period+1:i+1]
        m, s = np.mean(w), np.std(w, ddof=1)
        mid[i]   = m
        upper[i] = m + n_std*s
        lower[i] = m - n_std*s
    return mid, upper, lower

def clean(arr, n=90):
    arr = np.array(arr, dtype=float)[-n:]
    return [round(float(v),4) if not math.isnan(v) else None for v in arr]

# ── ANALISI ETF ───────────────────────────────────────────────
def analyze_etf(ticker, pd, etf_info):
    try:
        c = np.array(pd["c"], dtype=float)
        h = np.array(pd.get("h", pd["c"]), dtype=float)
        l = np.array(pd.get("l", pd["c"]), dtype=float)
        v = np.array(pd.get("v", [0]*len(c)), dtype=float)
        d = pd.get("d", [])
        n = len(c)
        price = float(c[-1])
        if price <= 0 or n < 50: return None

        # Correzione scala per ticker con dati Yahoo anomali
        scale = PRICE_SCALE_FIX.get(ticker, 1)
        if scale != 1:
            c = c * scale
            h = h * scale
            l = l * scale
            price = float(c[-1])
            print(f"    ⚙ Scala corretta ×{scale} → {price:.2f}")
        elif price < 0.10:
            # Prezzo anomalo senza correzione nota → tenta ×10000
            candidate = price * 10000
            if 1.0 < candidate < 50000:
                c = c * 10000
                h = h * 10000
                l = l * 10000
                price = float(c[-1])
                print(f"    ⚙ Scala auto-corretta ×10000 → {price:.2f}")
            else:
                print(f"    ⚠ Prezzo anomalo {price:.6f} — skip")
                return None

        low_52w  = float(np.min(c[-252:])) if n>=252 else float(np.min(c))
        high_52w = float(np.max(c[-252:])) if n>=252 else float(np.max(c))
        dist_low = (price-low_52w)/low_52w if low_52w>0 else 1.0

        ret_1w = float(price/c[-6]-1)   if n>6  else None
        ret_4w = float(price/c[-21]-1)  if n>21 else None
        ret_3m = float(price/c[-63]-1)  if n>63 else None

        ma50  = calc_ma(c, 50)
        ma200 = calc_ma(c, 200)
        ma200_rising = False
        if ma200 and n>=210:
            ma200_prev = calc_ma(c[:-5], 200)
            ma200_rising = bool(ma200 > (ma200_prev or 0))

        atr_arr  = calc_atr(h, l, c, 14)
        atr_c    = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else None
        atr_3m_val = float(np.mean(np.abs(np.diff(c[-63:])))) if n>63 else None
        atr_ratio  = float(atr_c/atr_3m_val) if atr_c and atr_3m_val and atr_3m_val>0 else None

        high_4w     = float(np.max(c[-21:])) if n>21 else price
        drawdown_4w = float((high_4w-price)/high_4w) if high_4w>0 else 0.0

        # ── INDICATORI TECNICI ──
        sar_arr, bull_arr = calc_sar(h, l)
        sar_bull = bool(bull_arr[-1])
        sar_val  = float(sar_arr[-1]) if not np.isnan(sar_arr[-1]) else None

        kama_arr = calc_kama(c, n=10, fast=5, slow=20)
        kama_val = float(kama_arr[-1]) if not np.isnan(kama_arr[-1]) else None
        kama_prev = float(kama_arr[-2]) if len(kama_arr)>1 and not np.isnan(kama_arr[-2]) else kama_val
        price_crossed_kama = kama_val and (price > kama_val) and (c[-2] <= kama_prev if len(c)>1 else False)
        price_above_kama   = kama_val and price > kama_val

        obv_arr = calc_obv(c, v)
        # OBV divergenza: prezzo scende ma OBV tiene (ultimi 20gg)
        obv_div = False
        if n > 20:
            price_trend = float(c[-1] - c[-20])
            obv_trend   = float(obv_arr[-1] - obv_arr[-20])
            obv_div     = bool(price_trend < 0 and obv_trend > 0)

        bb_mid, bb_upper, bb_lower = calc_bb(c, 20, 2.0)
        bb_width = float((bb_upper[-1]-bb_lower[-1])/bb_mid[-1]) if not np.isnan(bb_mid[-1]) and bb_mid[-1]>0 else None

        # ── CRITERI CATEGORIE ──
        is1 = bool(dist_low <= SOGLIA_MIN_STORICO)
        s1  = int(100 - dist_low/SOGLIA_MIN_STORICO*50) if is1 else 0

        is2 = bool(
            SOGLIA_PULLBACK_MIN <= drawdown_4w <= SOGLIA_PULLBACK_MAX and
            ma200 is not None and price > ma200*0.95 and ma200_rising
        )
        s2 = 0
        if is2:
            d_ma = abs(price-ma200)/ma200 if ma200 else 1
            s2 = int(min(100, 70+(1-d_ma*10)*30))

        is3 = bool(atr_ratio is not None and atr_ratio < (1-SOGLIA_ATR_COMPRES))
        s3  = int(min(100,(1-atr_ratio)*150)) if is3 else 0

        n_cat = sum([is1, is2, is3])
        sm    = max(s1, s2, s3)

        # ── SEGNALE TECNICO ──
        tech_score = 0
        if sar_bull:         tech_score += 35
        if price_above_kama: tech_score += 30
        if ma200 and price > ma200: tech_score += 20
        if obv_div:          tech_score += 15

        # BUY1/BUY2/BUY3
        if sar_bull and price_above_kama and ma200 and price > ma200:
            buy_level = "BUY1"
        elif (sar_bull and price_above_kama) or (sar_bull and ma200 and price > ma200):
            buy_level = "BUY2"
        elif sar_bull or price_above_kama:
            buy_level = "BUY3"
        else:
            buy_level = "WATCH"

        # Chart data (solo ultimi 90 barre)
        n90 = min(90, n)
        chart = {
            "dates":   d[-n90:],
            "closes":  clean(c, n90),
            "highs":   clean(h, n90),
            "lows":    clean(l, n90),
            "volumes": [int(x) for x in v[-n90:]],
            "ma50":    clean(np.array([calc_ma(c[:i+1], 50) if i>=49 else np.nan for i in range(n)])[-n90:], n90),
            "ma200":   clean(np.array([calc_ma(c[:i+1], 200) if i>=199 else np.nan for i in range(n)])[-n90:], n90),
            "sar":     clean(sar_arr, n90),
            "sar_bull":[bool(x) for x in bull_arr[-n90:]],
            "kama":    clean(kama_arr, n90),
            "obv":     clean(obv_arr, n90),
            "bb_upper":clean(bb_upper, n90),
            "bb_lower":clean(bb_lower, n90),
            "bb_mid":  clean(bb_mid, n90),
        }

        return {
            "ticker":           ticker,
            "name":             etf_info.get("n", ticker),
            "borsa":            etf_info.get("b", ""),
            "categoria":        etf_info.get("c", ""),
            "price":            round(price, 4),
            "low_52w":          round(low_52w, 4),
            "high_52w":         round(high_52w, 4),
            "dist_52w_low":     round(dist_low*100, 2),
            "dist_52w_high":    round((high_52w-price)/high_52w*100, 2) if high_52w>0 else 0,
            "ret_1w":           round(ret_1w*100, 2) if ret_1w is not None else None,
            "ret_4w":           round(ret_4w*100, 2) if ret_4w is not None else None,
            "ret_3m":           round(ret_3m*100, 2) if ret_3m is not None else None,
            "ma50":             round(ma50, 4)  if ma50  else None,
            "ma200":            round(ma200, 4) if ma200 else None,
            "ma200_rising":     ma200_rising,
            "drawdown_4w":      round(drawdown_4w*100, 2),
            "atr":              round(atr_c, 4) if atr_c else None,
            "atr_ratio":        round(atr_ratio, 3) if atr_ratio else None,
            "sar_bull":         sar_bull,
            "sar":              round(sar_val, 4) if sar_val else None,
            "kama":             round(kama_val, 4) if kama_val else None,
            "price_above_kama": price_above_kama,
            "price_crossed_kama": price_crossed_kama,
            "obv_divergence":   obv_div,
            "bb_width":         round(bb_width, 4) if bb_width else None,
            "tech_score":       tech_score,
            "buy_level":        buy_level,
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
            "chart":               chart,
        }
    except Exception as e:
        print(f"  ⚠ {ticker}: {e}")
        return None

# ── MIN RELATIVI ─────────────────────────────────────────────
def compute_min_relativi(results):
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        if r and r.get('ret_3m') is not None:
            by_cat[r['categoria']].append(r['ret_3m'])
    cat_means = {c: float(np.mean(v)) for c, v in by_cat.items() if len(v)>=5}
    for r in results:
        if not r: continue
        cat  = r['categoria']
        ret3 = r.get('ret_3m')
        mean = cat_means.get(cat, 0.0)
        r['cat_mean_ret3m'] = round(mean, 2)
        if ret3 is not None and cat in cat_means:
            diff = mean - ret3
            if diff >= SOGLIA_MIN_RELATIVO*100:
                r['is_min_relativo']    = True
                r['score_min_relativo'] = int(min(100, diff*5))
                r['n_categorie']       += 1
                r['score_master']       = max(r['score_master'], r['score_min_relativo'])
    return results

# ── TOP 20 ────────────────────────────────────────────────────
def compute_top20(results):
    """
    Seleziona i TOP 20 candidati operativi.
    Score composito = score_master×40% + tech_score×40% + obv_div×20%
    """
    candidates = [r for r in results if r and r.get('n_categorie',0) >= 1]
    for r in candidates:
        composite = (
            r.get('score_master', 0) * 0.40 +
            r.get('tech_score', 0)   * 0.40 +
            (20 if r.get('obv_divergence') else 0) * 1.0
        )
        r['composite_score'] = round(composite, 1)

    sorted_cands = sorted(candidates, key=lambda x: x['composite_score'], reverse=True)

    # Dividi per livello buy
    top20 = []
    for level in ['BUY1', 'BUY2', 'BUY3', 'WATCH']:
        level_items = [r for r in sorted_cands if r.get('buy_level')==level and r not in top20]
        top20.extend(level_items)
        if len(top20) >= 20: break

    return [{
        "ticker":           r["ticker"],
        "name":             r["name"],
        "borsa":            r["borsa"],
        "categoria":        r["categoria"],
        "price":            r["price"],
        "buy_level":        r["buy_level"],
        "composite_score":  r["composite_score"],
        "tech_score":       r["tech_score"],
        "score_master":     r["score_master"],
        "n_categorie":      r["n_categorie"],
        "sar_bull":         r["sar_bull"],
        "price_above_kama": r["price_above_kama"],
        "obv_divergence":   r["obv_divergence"],
        "ma200":            r.get("ma200"),
        "atr":              r.get("atr"),
        "ret_1w":           r.get("ret_1w"),
        "ret_3m":           r.get("ret_3m"),
        "is_min_storico":   r.get("is_min_storico"),
        "is_pullback":      r.get("is_pullback"),
        "is_compressione":  r.get("is_compressione"),
        "is_min_relativo":  r.get("is_min_relativo"),
    } for r in top20[:20]]

def make_serializable(obj):
    if isinstance(obj, dict):  return {k: make_serializable(v) for k,v in obj.items()}
    if isinstance(obj, list):  return [make_serializable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    if isinstance(obj, np.integer):       return int(obj)
    if isinstance(obj, np.floating):      return float(obj)
    if isinstance(obj, np.ndarray):       return obj.tolist()
    return obj

# ── MAIN ─────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("="*60)
    print(f"RAPTOR MIN FINDER v3.0 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    universe    = load_universe()
    ticker_map  = {e["t"]: e for e in universe}
    all_tickers = [e["t"] for e in universe]
    ck          = load_checkpoint()
    blist       = load_blacklist()

    # ── 3 PASSATE ──
    already = set(ck["prices"].keys())
    todo_p1 = [t for t in all_tickers if t not in already and t not in blist]
    print(f"\n  Cache: {len(already)} · Da scaricare: {len(todo_p1)} · Blacklist: {len(blist)}")

    falliti = todo_p1
    for bs, dl, nm in PASSATE:
        if not falliti: print(f"\n  ⚡ {nm} saltata"); continue
        todo = [t for t in falliti if t not in ck["prices"] and t not in blist]
        if not todo: falliti=[]; continue
        ck, falliti = scan_incremental(todo, ck, bs, dl, nm)

    # Aggiorna blacklist
    new_bl = 0
    for t, n in ck["fails"].items():
        if n >= MAX_FAILS and t not in blist:
            blist.add(t); new_bl += 1
    if new_bl:
        print(f"\n  🚫 Blacklist: +{new_bl} ticker")
        save_blacklist(blist)

    # ── ANALISI ──
    elapsed = (time.time()-t0)/60
    print(f"\n  Prezzi in cache: {len(ck['prices'])}/{len(all_tickers)} · Tempo: {elapsed:.1f} min")
    print("  Analisi ETF...")

    results = []
    for ticker, pd in ck["prices"].items():
        info = ticker_map.get(ticker, {"n":ticker,"b":"","c":"Altro"})
        r = analyze_etf(ticker, pd, info)
        if r: results.append(r)
    print(f"  Analizzati: {len(results)} ETF")

    results = compute_min_relativi(results)

    lista1 = sorted([r for r in results if r['is_min_storico']],  key=lambda x: x['score_min_storico'],  reverse=True)
    lista2 = sorted([r for r in results if r['is_pullback']],     key=lambda x: x['score_pullback'],     reverse=True)
    lista3 = sorted([r for r in results if r['is_compressione']], key=lambda x: x['score_compressione'], reverse=True)
    lista4 = sorted([r for r in results if r['is_min_relativo']], key=lambda x: x['score_min_relativo'], reverse=True)
    master  = sorted([r for r in results if r['n_categorie']>=2], key=lambda x: x['score_master'],       reverse=True)
    top20   = compute_top20(results)
    cands   = list({r['ticker'] for r in lista1+lista2+lista3+lista4+master})

    print(f"\n  📉 {len(lista1)} · 🔄 {len(lista2)} · 🔥 {len(lista3)} · 📊 {len(lista4)} · ⭐ {len(master)} · 🎯 TOP20: {len(top20)}")

    # Rimuovi chart dai record nelle liste (troppo pesante) — tienilo solo nel dict per ticker
    def strip_chart(lst, max_items=200):
        out = []
        for r in lst[:max_items]:
            rc = dict(r)
            rc.pop('chart', None)
            out.append(rc)
        return out

    # Chart separato indicizzato per ticker (solo candidati)
    charts = {}
    for r in results:
        if r.get('n_categorie',0) >= 1 and 'chart' in r:
            charts[r['ticker']] = r['chart']

    output = make_serializable({
        "generated":    datetime.now(timezone.utc).isoformat(),
        "version":      "3.0",
        "universe_tot": len(all_tickers),
        "analyzed":     len(results),
        "cached":       len(ck["prices"]),
        "blacklisted":  len(blist),
        "candidates":   cands,
        "stats": {
            "min_storico":  len(lista1), "pullback":  len(lista2),
            "compressione": len(lista3), "min_relativo": len(lista4),
            "master": len(master), "top20": len(top20),
        },
        "soglie": {
            "min_storico_pct":  SOGLIA_MIN_STORICO*100,
            "pullback_min_pct": SOGLIA_PULLBACK_MIN*100,
            "pullback_max_pct": SOGLIA_PULLBACK_MAX*100,
            "atr_compres_pct":  SOGLIA_ATR_COMPRES*100,
            "min_relativo_pct": SOGLIA_MIN_RELATIVO*100,
        },
        "top20":               top20,
        "lista1_min_storico":  strip_chart(lista1),
        "lista2_pullback":     strip_chart(lista2),
        "lista3_compressione": strip_chart(lista3),
        "lista4_min_relativo": strip_chart(lista4),
        "lista_master":        strip_chart(master, 100),
        "charts":              charts,
        "elapsed_min":         round(elapsed, 1),
    })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(',',':'))

    size_kb = OUT_FILE.stat().st_size/1024
    print(f"\n✅ {OUT_FILE.name} ({size_kb:.0f} KB) · {elapsed:.1f} min totali")
    print(f"   📊 Charts salvati: {len(charts)} candidati")

if __name__ == "__main__":
    main()
