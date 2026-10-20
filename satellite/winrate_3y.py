"""Meilleur win rate parmi les stratégies rentables sur CHAQUE fenêtre glissante de 3 ans depuis 1980.
Instrument: S&P 500 (^GSPC, prix, sans dividendes), long seulement, 1 position à la fois, frais 0,05 % par côté.
Fenêtre = trades entrés dans les 36 mois, départ chaque janvier (1980-82, 1981-83, ...) ;
réussie si rendement composé > 0 avec >= 3 trades. On affiche aussi la part des fenêtres mensuelles réussies."""
import itertools, sys
import numpy as np, pandas as pd, yfinance as yf

FEE = 0.0005
px = yf.download("^GSPC", start="1979-01-01", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
v, idx = px.values, px.index

def rsi(s, n):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn)).values
R = {n: rsi(px, n) for n in (2, 3, 4, 5)}
SMA = {m: px.rolling(m).mean().values for m in (3, 5, 10, 100, 200)}
first = np.searchsorted(idx, pd.Timestamp("1980-01-01"))

me = pd.Series(range(len(idx)), idx).groupby([idx.year, idx.month]).transform("max").values
TOM = {k: np.where(me - np.arange(len(idx)) == k)[0] for k in (0, 1, 2, 3)}  # k jours avant la fin du mois

def sim(sig, ex, stop, maxd):
    out, i = [], 0
    for s in sig:
        if s < first or s < i or s + 1 >= len(v): continue
        e = v[s + 1]; j = s + 1
        while j < len(v) - 1:
            if j > s + 1 and ((ex[0] == "sma" and v[j] > SMA[ex[1]][j]) or (ex[0] == "tp" and v[j] >= e * (1 + ex[1]))
                              or (stop and v[j] <= e * (1 - stop)) or j - (s + 1) >= maxd): break
            j += 1
        out.append((idx[s + 1], v[j] / e - 1 - 2 * FEE)); i = j + 1
    return pd.DataFrame(out, columns=["d", "r"])

YEARLY = pd.date_range("1980-01-01", "2023-01-01", freq="YS")
MONTHLY = pd.date_range("1980-01-01", "2023-06-01", freq="MS")
def windows(t, starts=YEARLY, strict=True):
    worst, ok, tot = 9, 0, 0
    d = t.d.values; lr = np.log1p(t.r.values); cs = np.concatenate([[0], np.cumsum(lr)])
    for s in starts:
        a, b = np.searchsorted(d, np.datetime64(s)), np.searchsorted(d, np.datetime64(s + pd.DateOffset(years=3)))
        w = np.expm1(cs[b] - cs[a]) if b - a >= 3 else -1
        worst = min(worst, w); ok += w > 0; tot += 1
        if strict and w <= 0: return None
    return worst if strict else ok / tot

combos = []
for n, thr, trend, ex, stop, maxd in itertools.product(
        (2, 3, 4, 5), (5, 10, 15, 20, 30), (200, 100, None),
        [("sma", m) for m in (3, 5, 10)] + [("tp", p) for p in (0.01, 0.02, 0.03)],
        (0.03, 0.05, 0.10, None), (5, 10, 20)):
    sig = np.where((R[n] < thr) & ((v > SMA[trend]) if trend else True))[0]
    combos.append((f"RSI({n})<{thr} filtre SMA{trend} sortie {ex[0]}{ex[1]} stop {stop} max {maxd}j", sig, ex, stop, maxd))
for k, h, trend, stop in itertools.product((0, 1, 2, 3), (2, 3, 4, 5, 6), (200, None), (0.03, 0.05, None)):
    sig = TOM[k] if not trend else TOM[k][v[TOM[k]] > SMA[trend][TOM[k]]]
    combos.append((f"Fin de mois : achat {k + 1}e dernier jour, garde {h}j, filtre SMA{trend}, stop {stop}", sig, ("none", 0), stop, h))
res, n_tot = [], 0
for name, sig, ex, stop, maxd in combos:
    t = sim(sig, ex, stop, maxd); n_tot += 1
    if len(t) < 60: continue
    w = windows(t)
    if w is None: continue
    res.append((name, len(t), (t.r > 0).mean(), t.r.mean(), w, windows(t, MONTHLY, False), t))
res.sort(key=lambda r: -r[2])
print(f"{n_tot} stratégies testées ; {len(res)} rentables sur TOUTES les fenêtres de 3 ans débutant en janvier (1980-2023)")
print("stratégie | trades | win rate | rdt moyen/trade | pire fenêtre | % fenêtres mensuelles OK")
for name, n, wr, m, w, mo, _ in res[:12]:
    print(name, "|", n, f"| {wr:.1%} | {m:+.2%} | {w:+.1%} | {mo:.0%}")
if res:
    t = res[0][6]; t["y"] = t.d.dt.year
    print("\nMeilleure, par année : win rate et rendement composé")
    print(t.groupby("y").r.agg(trades="size", wr=lambda r: (r > 0).mean(), ret=lambda r: (1 + r).prod() - 1).round(3).to_string())
