"""Cherche une stratégie à win rate >= 55 % ET espérance positive, validée hors échantillon.
Entrée: RSI(n) < seuil et clôture > SMA200 ; sortie: clôture > SMA(m), ou stop, ou durée max.
Train 2001-2014 / Test 2015-2026, 25 titres, 1 position par titre, frais 0,05 % par côté."""
import itertools
import numpy as np, pandas as pd
from rotation_satellite import TICKERS, load

px = load(TICKERS, 2001).drop(columns="SPY")
FEE = 0.0005

def rsi(s, n):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def trades(n, thr, m, stop, maxd):
    out = []
    for t in px:
        s = px[t].dropna()
        if len(s) < 400: continue
        r, sma200, sm = rsi(s, n), s.rolling(200).mean(), s.rolling(m).mean()
        v, idx, i = s.values, s.index, 200
        while i < len(s) - 1:
            if r.iloc[i] < thr and v[i] > sma200.iloc[i]:
                e = v[i + 1]; j = i + 1
                while j < len(s) - 1:
                    if v[j] > sm.iloc[j] or v[j] <= e * (1 - stop) or j - (i + 1) >= maxd: break
                    j += 1
                out.append((idx[i + 1], v[j] / e - 1 - 2 * FEE)); i = j + 1
            else: i += 1
    return pd.DataFrame(out, columns=["d", "r"])

def summ(x):
    if len(x) < 30: return None
    w = x.r > 0
    return len(x), w.mean(), x.r.mean(), x.r[w].sum() / -x.r[~w].sum()

res = []
for n, thr, m, stop, maxd in itertools.product([2, 3, 5], [5, 10, 20], [3, 5, 10], [0.05, 0.10, 1.0], [5, 10]):
    t = trades(n, thr, m, stop, maxd)
    a, b = summ(t[t.d < "2015-01-01"]), summ(t[t.d >= "2015-01-01"])
    if a and b: res.append(((n, thr, m, stop, maxd), a, b))
ok = [r for r in res if r[1][1] >= .55 and r[2][1] >= .55 and r[1][2] > 0 and r[2][2] > 0]
ok.sort(key=lambda r: -(r[1][2] + r[2][2]))
print(f"{len(res)} combos testés, {len(ok)} avec WR>=55 % et espérance>0 en train ET test")
print("params(rsi_n,seuil,sma_exit,stop,max_jours) | train(n,WR,exp,PF) | test(n,WR,exp,PF)")
for p, a, b in ok[:8]:
    print(p, "|", f"{a[0]} {a[1]:.1%} {a[2]:+.2%} {a[3]:.2f}", "|", f"{b[0]} {b[1]:.1%} {b[2]:+.2%} {b[3]:.2f}")
