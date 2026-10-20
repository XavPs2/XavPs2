"""Recherche plus poussée : rotation mensuelle par momentum entre secteurs / classes d'actifs.
Objectif : rendement nettement supérieur au S&P 500 (au moins +3 pts/an) avec MOINS de drawdown ET de volatilité.
Données : fonds sectoriels Fidelity Select / Vanguard (frais déjà inclus dans les prix), dividendes réinvestis.
Signal en fin de mois, exécution à la clôture suivante, 0,10 % de coût par unité de rotation.
Validation : deux moitiés (1988-2006 / 2007-2026) et stabilité des paramètres voisins."""
import itertools
import numpy as np, pandas as pd, yfinance as yf

SECTORS = ("FSPTX FSELX FBIOX FSPHX FSENX FSRBX FSCHX FDFAX FSUTX FSRFX FSAGX FSCSX FSRPX FSHOX FBMPX "
           "FSDPX FSTCX FSDAX FSLBX FSVLX FSPCX FSHCX FIDSX VGPMX VGENX VGHCX").split()
ASSETS = "VFINX NAESX VWIGX VWEHX VGPMX".split()
BOND, SPX = "VUSTX", "VFINX"
px = yf.download(sorted(set(SECTORS + ASSETS + [BOND, SPX])), start="1984-01-01", auto_adjust=True, progress=False)["Close"]
irx = yf.download("^IRX", start="1984-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
px = px.ffill(limit=5)
idx = px.index
R = px.pct_change()
cash = (irx.reindex(idx).ffill() / 100 / 252).fillna(0)
month_end = pd.Series(idx, idx).groupby([idx.year, idx.month]).transform("max") == idx
ME = idx[month_end.values]
START = "1988-01-01"
COST = 0.001

def score(kind):
    if kind == "12m": return px.pct_change(252)
    if kind == "6m": return px.pct_change(126)
    if kind == "3m": return px.pct_change(63)
    if kind == "12-1": return px.shift(21).pct_change(231)
    return (px.pct_change(21) + px.pct_change(63) + px.pct_change(126) + px.pct_change(252)) / 4   # "mix"

SC = {k: score(k) for k in ("12m", "6m", "3m", "12-1", "mix")}
CASH_LB = {k: cash.rolling({"12m": 252, "6m": 126, "3m": 63, "12-1": 231, "mix": 115}[k]).sum() for k in SC}
SMA200 = px[SPX].rolling(200).mean()

def sim(universe, kind, top, absf, regime, vt):
    s = SC[kind][universe]
    W = pd.DataFrame(0.0, index=ME, columns=universe + [BOND, "CASH"])
    for d in ME:
        row = s.loc[d].dropna()
        if len(row) < top: continue
        pick = row.nlargest(top)
        for t, v in pick.items():
            if absf != "none" and v < CASH_LB[kind].loc[d]:
                W.loc[d, BOND if absf == "bonds" else "CASH"] += 1 / top
            else:
                W.loc[d, t] += 1 / top
        if regime and px.loc[d, SPX] < SMA200.loc[d]:
            W.loc[d] = 0.0; W.loc[d, BOND] = 1.0
    W = W.reindex(idx).ffill().fillna(0).shift(2).fillna(0)
    rets = R[universe + [BOND]].fillna(0).copy(); rets["CASH"] = cash
    r = (W * rets[W.columns]).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0) * COST
    if vt:  # ciblage de volatilité 15 %, levier max 1,5x, emprunt au T-bill + 1 %
        lev = (0.15 / (r.rolling(63).std() * np.sqrt(252))).clip(upper=1.5).shift(2).fillna(1)
        r = lev * r - (lev - 1).clip(lower=0) * (cash + 0.01 / 252)
    return r

def stats(r, a=START, b="2100-01-01"):
    r = r[(r.index >= a) & (r.index <= b)]
    eq = (1 + r).cumprod(); yrs = len(r) / 252
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, mdd=(eq / eq.cummax() - 1).min(), vol=r.std() * np.sqrt(252),
                sharpe=(r - cash.reindex(r.index)).mean() / r.std() * np.sqrt(252))

bench = {k: stats(R[SPX].fillna(0), *p) for k, p in (("all", (START,)), ("h1", (START, "2006-12-31")), ("h2", ("2007-01-01",)))}
rows = []
UNIS = {"secteurs": SECTORS, "classes d'actifs": ASSETS, "secteurs+actifs": sorted(set(SECTORS + ASSETS))}
for (un, uni), kind, top, absf, regime, vt in itertools.product(UNIS.items(), SC, (1, 2, 3, 4, 5), ("none", "bonds", "cash"), (False, True), (False, True)):
    if top > len(uni) - 1: continue
    r = sim(uni, kind, top, absf, regime, vt)
    a, h1, h2 = stats(r), stats(r, START, "2006-12-31"), stats(r, "2007-01-01")
    rows.append(dict(univers=un, score=kind, top=top, abs=absf, regime="SMA200" if regime else "-", voltgt="15%" if vt else "-",
                     cagr=a["cagr"], mdd=a["mdd"], vol=a["vol"], sharpe=a["sharpe"],
                     cagr_h1=h1["cagr"], mdd_h1=h1["mdd"], cagr_h2=h2["cagr"], mdd_h2=h2["mdd"]))
df = pd.DataFrame(rows)
df.to_csv("rotation_search_results.csv", index=False)
B = bench["all"]
good = df[(df.cagr >= B["cagr"] + 0.03) & (df.mdd > B["mdd"]) & (df.vol <= B["vol"])]
robust = good[(good.cagr_h1 > bench["h1"]["cagr"]) & (good.cagr_h2 > bench["h2"]["cagr"]) &
              (good.mdd_h1 > bench["h1"]["mdd"]) & (good.mdd_h2 > bench["h2"]["mdd"])]
print(f"S&P 500 {START[:4]}-2026 : CAGR {B['cagr']:.1%}  MaxDD {B['mdd']:.1%}  vol {B['vol']:.1%}  Sharpe {B['sharpe']:.2f}")
print(f"   1988-2006 : CAGR {bench['h1']['cagr']:.1%} DD {bench['h1']['mdd']:.1%} | 2007-2026 : CAGR {bench['h2']['cagr']:.1%} DD {bench['h2']['mdd']:.1%}")
print(f"{len(df)} stratégies testées ; {len(good)} font ≥ +3 pts/an avec moins de DD et de vol ; {len(robust)} le font aussi dans CHAQUE moitié\n")
pd.set_option("display.width", 250)
p = lambda d: d.assign(**{c: d[c].map("{:.1%}".format) for c in ("cagr", "mdd", "vol", "cagr_h1", "mdd_h1", "cagr_h2", "mdd_h2")}, sharpe=d.sharpe.map("{:.2f}".format))
print(p(robust.sort_values("sharpe", ascending=False).head(20)).to_string(index=False))
print("\nFamilles robustes (combien de variantes passent, par univers / filtre) :")
print(robust.groupby(["univers", "abs", "regime", "voltgt"]).size().sort_values(ascending=False).head(12).to_string())
