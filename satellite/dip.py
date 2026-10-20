"""Repli court sur le S&P 500 : RSI(n) < seuil au-dessus de la SMA200, sortie sur cible de gain, stop ou durée max.
Une seule position, 100 % du capital, frais 0,05 % par côté."""
import pandas as pd
import yfinance as yf
from meanrev import rsi

COST = 0.0005


def load_index(first_year, symbol="^GSPC"):
    s = yf.download(symbol, start=f"{first_year - 2}-01-01", auto_adjust=True, progress=False)["Close"]
    return s.squeeze().dropna().rename(symbol)


def run_dip(px, start_year, capital, n=3, thr=20, trend=200, tp=0.01, stop=0.05, maxd=20):
    v, idx = px.values, px.index
    r = rsi(px, n).values
    sma = px.rolling(trend).mean().values if trend else None
    first = idx.searchsorted(pd.Timestamp(f"{start_year}-01-01"))
    cash, sh, e, k0, journal, i = capital, 0.0, 0.0, None, [], first
    equity, inmkt = {}, {}
    cur = None
    for k in range(first, len(v)):
        d, p = idx[k], v[k]
        if cur is not None and k > k0:
            why = None
            if p >= e * (1 + tp):
                why = f"cible +{tp:.1%}"
            elif stop and p <= e * (1 - stop):
                why = f"stop {stop:.0%}"
            elif k - k0 >= maxd:
                why = f"{maxd} jours"
            if why:
                cash = sh * p * (1 - COST)
                cur.update(exit_date=d.date(), exit_price=round(p, 2), exit_reason=why,
                           ret_pct=round((p / e - 1) * 100, 2), cap_out=cash)
                sh, cur = 0.0, None
        elif cur is None and k > first and r[k - 1] < thr and (sma is None or v[k - 1] > sma[k - 1]):
            e, k0 = p, k
            sh = cash * (1 - COST) / p
            cur = dict(year=d.year, winner=px.name, prev_year_ret_pct=round(r[k - 1], 1), entry_date=d.date(),
                       entry_price=round(p, 2), exit_date=None, exit_price=None, exit_reason="", ret_pct=None,
                       cap_in=cash)
            journal.append(cur)
            cash = 0.0
        equity[d] = cash + sh * p
        inmkt[d] = cur is not None
    if cur is not None:
        cur.update(exit_reason="ouverte", exit_price=round(v[-1], 2), ret_pct=round((v[-1] / e - 1) * 100, 2),
                   cap_out=sh * v[-1])
    return pd.Series(equity), pd.DataFrame(journal), pd.Series(inmkt)
