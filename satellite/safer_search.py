"""Cherche une stratégie plus sûre que le S&P 500 (drawdown et volatilité plus bas) qui le bat (CAGR plus haut).
Données : VFINX (fonds S&P 500, dividendes réinvestis) depuis 1980, VUSTX (obligations long terme) depuis 1986,
^IRX (bons du Trésor 3 mois) pour le cash. Signal à la clôture, exécution à la clôture suivante (1 jour de retard),
0,10 % de coût par changement de position. Effet de levier : coût d'emprunt = T-bill + 1 %/an sur la part empruntée."""
import numpy as np, pandas as pd, yfinance as yf

def get(s, start="1979-01-01"):
    return yf.download(s, start=start, auto_adjust=True, progress=False)["Close"].squeeze().dropna()

spx, bnd, irx = get("VFINX"), get("VUSTX"), get("^IRX")
idx = spx.index
cash_d = (irx.reindex(idx).ffill() / 100 / 252).fillna(0)
r_spx = spx.pct_change().fillna(0)
r_bnd = bnd.reindex(idx).ffill().pct_change().fillna(0)
SW = 0.001

def stats(r, label, start):
    r = r[r.index >= start]
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    vol = r.std() * np.sqrt(252)
    sharpe = (r - cash_d.reindex(r.index)).mean() / r.std() * np.sqrt(252)
    ye = eq.resample("YE").last(); yr = ye.pct_change(); yr.iloc[0] = ye.iloc[0] - 1
    return dict(strat=label, cagr=cagr, mdd=mdd, vol=vol, sharpe=sharpe, worst_year=yr.min(), final=eq.iloc[-1])

def trend(n, lev=1.0, off="cash", monthly=False):
    sig = (spx > spx.rolling(n).mean()).astype(float)
    if monthly:  # ne regarde le signal qu'au dernier jour du mois
        me = sig.groupby([idx.year, idx.month]).transform(lambda s: s.iloc[-1])
        last = pd.Series(idx, idx).groupby([idx.year, idx.month]).transform("max") == idx
        sig = sig.where(last).ffill()
    pos = sig.shift(2).fillna(0)
    alt = cash_d if off == "cash" else r_bnd
    on = lev * r_spx - (lev - 1) * (cash_d + 0.01 / 252)
    r = pos * on + (1 - pos) * alt - pos.diff().abs().fillna(0) * SW
    return r

def dual(look=252, lev=1.0):
    """Momentum dual mensuel : S&P si son rendement 12 mois bat les T-bills, sinon obligations."""
    m12 = spx.pct_change(look)
    c12 = cash_d.rolling(look).sum()
    last = pd.Series(idx, idx).groupby([idx.year, idx.month]).transform("max") == idx
    sig = (m12 > c12).astype(float).where(last).ffill()
    pos = sig.shift(2).fillna(0)
    on = lev * r_spx - (lev - 1) * (cash_d + 0.01 / 252)
    return pos * on + (1 - pos) * r_bnd - pos.diff().abs().fillna(0) * SW

START = "1987-01-01"   # période commune (obligations disponibles)
rows = [stats(r_spx, "S&P 500 acheté et gardé", START)]
for n in (100, 150, 200, 250):
    for lev in (1.0, 1.5, 2.0):
        for off in ("cash", "obligations"):
            for mo in (False, True):
                rows.append(stats(trend(n, lev, off, mo), f"SMA{n} {'mensuel' if mo else 'quotidien'} x{lev} hors marché={off}", START))
for look in (126, 189, 252):
    for lev in (1.0, 1.5, 2.0):
        rows.append(stats(dual(look, lev), f"Momentum dual {look // 21} mois x{lev}", START))
df = pd.DataFrame(rows)
b = df.iloc[0]
ok = df[(df.cagr > b.cagr) & (df.mdd > b.mdd) & (df.vol < b.vol)].sort_values("sharpe", ascending=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
fmt = lambda d: d.assign(cagr=d.cagr.map("{:.1%}".format), mdd=d.mdd.map("{:.1%}".format), vol=d.vol.map("{:.1%}".format),
                         sharpe=d.sharpe.map("{:.2f}".format), worst_year=d.worst_year.map("{:.1%}".format), final=d.final.map("{:.1f}x".format))
print(f"Période {START} – {idx[-1].date()}  |  {len(df) - 1} stratégies testées, {len(ok)} battent le S&P 500 avec moins de drawdown ET de volatilité\n")
print(fmt(pd.concat([df.iloc[[0]], ok.head(15)])).to_string(index=False))

print("\nRelâché : CAGR > S&P et drawdown moins profond (volatilité libre)")
ok2 = df[(df.cagr > b.cagr) & (df.mdd > b.mdd)].sort_values("cagr", ascending=False)
print(fmt(ok2.head(12)).to_string(index=False))

print("\nRobustesse : deux moitiés de l'historique")
cands = {"S&P 500": r_spx, "SMA200 mensuel → obligations": trend(200, 1, "obligations", True),
         "SMA150 mensuel → obligations": trend(150, 1, "obligations", True),
         "Momentum dual 12 mois": dual(252), "SMA200 quotidien x1.5 → obligations": trend(200, 1.5, "obligations"),
         "SMA200 mensuel x1.5 → obligations": trend(200, 1.5, "obligations", True)}
for a_, b_ in (("1987-01-01", "2006-12-31"), ("2007-01-01", "2026-12-31")):
    rows = [stats(r[(r.index <= b_)], k, a_) for k, r in cands.items()]
    print(f"\n{a_[:4]}–{b_[:4]}"); print(fmt(pd.DataFrame(rows)).to_string(index=False))
