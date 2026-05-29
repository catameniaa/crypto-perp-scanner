#!/usr/bin/env python3
import time, sys, argparse, os, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import ccxt

# Ayarlar
OUT_DIR = Path("public")
STRONG = 2
DAMPEN = 0.5
LIMIT = 100

# Vektörize İndikatörler
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, float('nan')))

def atr(df, n=10):
    h, l, c = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def supertrend(df, n=10, m=3):
    hl2 = (df['high'] + df['low']) / 2
    a = atr(df, n)
    ub = hl2 + m * a
    lb = hl2 - m * a
    st = np.full(len(df), np.nan)
    d = np.ones(len(df))
    for i in range(1, len(df)):
        p = st[i-1] if not np.isnan(st[i-1]) else lb.iloc[i]
        c = df['close'].iloc[i]
        if c > p:
            d[i] = 1
            st[i] = max(lb.iloc[i], p) if d[i-1] == 1 else lb.iloc[i]
        else:
            d[i] = -1
            st[i] = min(ub.iloc[i], p) if d[i-1] == -1 else ub.iloc[i]
    return pd.Series(d, index=df.index)

def bollinger(s, n=20, m=2):
    mid = s.rolling(n).mean()
    std = s.rolling(n).std()
    return mid - m * std, mid + m * std

def divergence(close, rsi_s, lb=14):
    if len(close) < lb + 1: return 0
    c, r = close.iloc[-lb:], rsi_s.iloc[-lb:]
    pu = c.iloc[-1] > c.iloc[0]
    ru = r.iloc[-1] > r.iloc[0]
    if not pu and ru: return 1  # Fiyat düşerken RSI yükseliyor (Bullish Div)
    if pu and not ru: return -1 # Fiyat çıkarken RSI düşüyor (Bearish Div)
    return 0

# Veri Çekme İşlemleri (Binance)
def get_ex():
    return ccxt.binance({'options': {'defaultType': 'future'}, 'enableRateLimit': True})

def fetch_ohlcv(ex, sym, tf='4h', lim=300):
    try:
        raw = ex.fetch_ohlcv(sym, tf, limit=lim)
        if not raw or len(raw) < 60: return None
        df = pd.DataFrame(raw, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        return df.set_index('ts').astype(float)
    except:
        return None

def fetch_funding(ex, sym):
    try:
        data = ex.fetch_funding_rate(sym)
        rate = data.get('fundingRate')
        return float(rate) * 100 * 3 * 365 if rate else None
    except:
        return None

def fetch_oi(ex, sym):
    try:
        history = ex.fetch_open_interest_history(sym, '1h', limit=24)
        if not history or len(history) < 2: return None
        o = float(history[0]['openInterestAmount'])
        n = float(history[-1]['openInterestAmount'])
        return (n - o) / o * 100 if o else None
    except:
        return None

def fetch_cvd_binance(sym):
    # Binance public REST klines contains taker buy volume
    # raw index 5: volume, index 9: taker buy base volume
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym.replace('/USDT:USDT', 'USDT')}&interval=4h&limit=14"
        res = requests.get(url, timeout=5).json()
        if type(res) is list and len(res) == 14:
            vols = np.array([float(k[5]) for k in res])
            taker_vols = np.array([float(k[9]) for k in res])
            return pd.Series(taker_vols / np.where(vols==0, 1, vols))
        return None
    except:
        return None

def btc_regime(ex):
    df = fetch_ohlcv(ex, 'BTC/USDT:USDT', '1d', 220)
    if df is None or len(df) < 200:
        return {'bull': True, 'price': None, 'ema200': None}
    e = ema(df['close'], 200)
    p = df['close'].iloc[-1]
    return {'bull': p > e.iloc[-1], 'price': p, 'ema200': e.iloc[-1]}

def scan_coin(ex, sym, bull):
    df = fetch_ohlcv(ex, sym)
    if df is None: return None

    close = df['close']
    r14 = rsi(close)
    bb_lo, bb_hi = bollinger(close)
    st = supertrend(df)
    fund = fetch_funding(ex, sym)
    oi = fetch_oi(ex, sym)
    cvd = fetch_cvd_binance(sym)

    lc, lr = close.iloc[-1], r14.iloc[-1]
    sc = {}

    sc['trend'] = 1 if st.iloc[-1] == 1 else -1
    sc['rsi'] = 1 if lr < 30 else (-1 if lr > 70 else 0)
    sc['div'] = divergence(close, r14)

    if cvd is not None:
        cvd_slope = cvd.iloc[-1] - cvd.iloc[0]
        pu = close.iloc[-1] > close.iloc[-14]
        sc['cvd'] = 1 if (not pu and cvd_slope > 0) else (-1 if (pu and cvd_slope < 0) else 0)
    else:
        sc['cvd'] = 0

    if fund is not None:
        sc['fund'] = 1 if fund < -10 else (-1 if fund > 50 else 0)
    else:
        sc['fund'] = 0

    if oi is not None:
        pc = (lc / close.iloc[-2] - 1) * 100
        sc['oi'] = 1 if (oi > 2 and pc > 0) else (-1 if (oi > 2 and pc < 0) else 0)
    else:
        sc['oi'] = 0

    sc['bb'] = 1 if lc < bb_lo.iloc[-1] else (-1 if lc > bb_hi.iloc[-1] else 0)

    raw = sum(sc.values())
    net = raw * DAMPEN if (not bull and raw > 0) else float(raw)

    return {
        'symbol': sym, 'close': round(lc, 6), 'rsi': round(lr, 1),
        'raw_net': raw, 'net': net, 'scores': sc,
        'funding': round(fund, 2) if fund is not None else None,
        'oi': round(oi, 2) if oi is not None else None,
    }

def harvest_stats(ex, sym, days=30):
    try:
        history = ex.fetch_funding_rate_history(sym, limit=days * 3)
        if not history: return None
        arr = [float(h['fundingRate']) * 100 * 3 * 365 for h in history]
        a = np.array(arr)
        pp = float((a > 0).mean() * 100)
        return {
            'symbol': sym, 'mean': round(float(a.mean()), 1),
            'std': round(float(a.std()), 1), 'pos_pct': round(pp, 0),
            'score': round(pp - a.std() * 0.5, 1),
        }
    except:
        return None

# HTML ve Markdown Üreticileri
def pills(sc):
    names = {'trend':'Trend', 'rsi':'RSI', 'div':'Div', 'cvd':'CVD', 'fund':'Fund', 'oi':'OI', 'bb':'BB'}
    out = ''
    for k, v in sc.items():
        cls = 'bull' if v > 0 else ('bear' if v < 0 else 'neu')
        arrow = '▲' if v > 0 else ('▼' if v < 0 else '–')
        out += f'<span class="p {cls}">{arrow}{names.get(k, k)}</span>'
    return out

def trows(items):
    if not items: return "<tr><td colspan='8' class='empty'>Sinyal yok</td></tr>"
    rows = ''
    for r in items:
        nc = 'bull' if r['net'] > 0 else 'bear'
        fn = f"{r['funding']:+.1f}%" if r['funding'] is not None else '-'
        oi = f"{r['oi']:+.1f}%" if r['oi'] is not None else '-'
        sym = r['symbol'].replace('/USDT:USDT', '')
        rows += f"<tr><td><b><a href='https://www.binance.com/en/futures/{sym}' target='_blank' style='color:inherit;text-decoration:none'>{sym}</a></b></td><td>{r['close']}</td><td>{r['rsi']}</td><td class='{nc}'><b>{r['raw_net']:+d}</b></td><td class='{nc}'>{r['net']:+.1f}</td><td>{fn}</td><td>{oi}</td><td>{pills(r['scores'])}</td></tr>"
    return rows

def generate_md(sl, ss, ts):
    md = f"### Sinyal Özeti ({ts})\n\n**UYARI:** Bu araç bir KARAR DESTEK ARACIDIR. Backtest 365 günlük Binance perpetual verisinde Sharpe -0.7 ile negatif sonuç verdi. Sinyaller tek başına alfa garantisi vermez.\n\n"
    md += f"**Güçlü Long Sinyalleri:** {len(sl)} adet\n"
    md += f"**Güçlü Short Sinyalleri:** {len(ss)} adet\n\n"
    md += "Lütfen paper trade defteri (Notion/Excel) tutun, 5-10 gün sonra fiyatı kontrol edin. Hit-rate %55+ değilse otomatik emir bağlamayın."
    (OUT_DIR / "dashboard_summary.md").write_text(md, encoding='utf-8')

def write_dashboard_html(sl, ss, rl, rs, hv, regime, ts):
    bull = regime['bull']
    bp = f"${regime['price']:,.0f}" if regime['price'] else 'N/A'
    ep = f"${regime['ema200']:,.0f}" if regime['ema200'] else 'N/A'
    rlbl = 'BULL' if bull else 'BEAR'
    rcls = 'bull' if bull else 'bear'

    html = f"""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="600"><title>Binance Perp Dashboard</title><style>
    *{{box-sizing:border-box;margin:0;padding:0}}:root{{--bg:#0d1117;--card:#161b22;--border:#30363d;--green:#3fb950;--red:#f85149;--text:#c9d1d9;--muted:#8b949e}}
    body{{background:var(--bg);color:var(--text);font-family:-apple-system,sans-serif;font-size:14px;padding:12px}}
    h1{{font-size:1.1em;margin-bottom:12px}} h2{{font-size:.8em;color:var(--muted);text-transform:uppercase;margin:14px 0 6px}}
    .kpis{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}}
    .kpi{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px;flex:1;min-width:90px}}
    .kpi .l{{font-size:.7em;color:var(--muted);margin-bottom:3px}} .kpi .v{{font-size:1em;font-weight:600}}
    .bull{{color:var(--green)!important}} .bear{{color:var(--red)!important}}
    table{{width:100%;border-collapse:collapse;background:var(--card);border-radius:8px;overflow:hidden;margin-bottom:10px}}
    th{{background:#21262d;padding:7px;text-align:left;font-size:.7em;color:var(--muted);white-space:nowrap}}
    td{{padding:6px 7px;border-top:1px solid var(--border);font-size:.8em}} tr:hover td{{background:#1c2128}}
    .p{{display:inline-block;font-size:.65em;padding:2px 4px;border-radius:3px;margin:1px}}
    .p.bull{{background:#1e3a2a;color:var(--green)}} .p.bear{{background:#3a1e1e;color:var(--red)}} .p.neu{{background:#21262d;color:var(--muted)}}
    .warn{{background:#272115;border:1px solid #b08000;border-radius:8px;padding:10px;margin-bottom:12px;font-size:.8em;color:#d29922;line-height:1.5}}
    .btn{{background:#238636;color:#fff;border:none;border-radius:6px;padding:11px;font-size:.9em;cursor:pointer;width:100%;margin-bottom:12px}}
    @media(max-width:500px){{th:nth-child(n+6),td:nth-child(n+6){{display:none}}}}</style></head><body>
    <h1>Binance Perp Sinyal Motoru v2</h1>
    <div class="warn"><b>DİKKAT:</b> Backtest 365 günlük veride Sharpe -0.7 verdi. Bu sistem kar garantisi vermez. Paper trade (sanal bakiye) tutun, 30 işlem sonrası hit-rate %55'in altındaysa gerçek işlem açmayın! Trade botu değil, radar aracıdır.</div>
    <div class="kpis"><div class="kpi"><div class="l">BTC Fiyat</div><div class="v">{bp}</div></div><div class="kpi"><div class="l">EMA200</div><div class="v">{ep}</div></div><div class="kpi"><div class="l">Rejim</div><div class="v {rcls}">{rlbl}</div></div><div class="kpi"><div class="l">Son Tarama</div><div class="v" style="font-size:.7em">{ts}</div></div></div>
    <button class="btn" onclick="window.location.reload()">Verileri Yenile (Browser Cache Temizle)</button>
    <p style="color:var(--muted);font-size:.7em;margin-bottom:10px">Not: Arka planda GitHub Actions her 15 dakikada bir verileri günceller. Manuel tetikleme için GitHub reponuzdan Actions tabını kullanın.</p>
    <h2>Güçlü Long (Net &gt;= {STRONG})</h2><table><tr><th>Sembol</th><th>Fiyat</th><th>RSI</th><th>Ham</th><th>Net</th><th>Fund</th><th>OI</th><th>Sinyaller</th></tr>{trows(sl)}</table>
    <h2>Güçlü Short (Net &lt;= -{STRONG})</h2><table><tr><th>Sembol</th><th>Fiyat</th><th>RSI</th><th>Ham</th><th>Net</th><th>Fund</th><th>OI</th><th>Sinyaller</th></tr>{trows(ss)}</table>
    <h2>Ham Long</h2><table><tr><th>Sembol</th><th>Fiyat</th><th>RSI</th><th>Ham</th><th>Net</th><th>Fund</th><th>OI</th><th>Sinyaller</th></tr>{trows(rl)}</table>
    <h2>Ham Short</h2><table><tr><th>Sembol</th><th>Fiyat</th><th>RSI</th><th>Ham</th><th>Net</th><th>Fund</th><th>OI</th><th>Sinyaller</th></tr>{trows(rs)}</table>
    <br><br><p style="text-align:center;color:var(--muted);font-size:0.8em;">Makine öğrenimi/Algoritmik radar testi</p>
    </body></html>"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.html").write_text(html, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()

    ex = get_ex()
    regime = btc_regime(ex)
    
    try:
        markets = ex.load_markets()
        tickers = ex.fetch_tickers()
        
        def _vol(t): return t.get('quoteVolume') or t.get('baseVolume') or 0
        
        syms = sorted(
            [k for k in tickers.keys() if k.endswith('USDT') and markets[k]['swap']],
            key=lambda s: _vol(tickers[s]),
            reverse=True
        )[:args.limit]
    except Exception as e:
        print(f"Hata: {e}")
        return

    results = []
    for sym in syms:
        r = scan_coin(ex, sym, regime['bull'])
        if r: results.append(r)
        time.sleep(0.1)  # Rate limit koruması

    sl = sorted([r for r in results if r['net'] >= STRONG], key=lambda x: -x['net'])
    ss = sorted([r for r in results if r['net'] <= -STRONG], key=lambda x: x['net'])
    rl = sorted([r for r in results if r['raw_net'] >= STRONG], key=lambda x: -x['raw_net'])
    rs = sorted([r for r in results if r['raw_net'] <= -STRONG], key=lambda x: x['raw_net'])

    hv = [] # Funding Harvest opsiyonel olarak eklenebilir.

    now_tr = datetime.now(timezone.utc) + timedelta(hours=3)
    ts = now_tr.strftime('%d.%m.%Y %H:%M TSİ')

    write_dashboard_html(sl, ss, rl, rs, hv, regime, ts)
    generate_md(sl, ss, ts)
    print("GitHub Pages için dosyalar /public klasörüne yazıldı.")

if __name__ == '__main__':
    main()
