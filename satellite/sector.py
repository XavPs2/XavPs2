"""Rotation sectorielle : chaque fin de mois, les `top` secteurs au meilleur momentum à parts égales ;
si le S&P 500 est sous sa SMA200, 100 % en obligations du Trésor long terme (VUSTX).
Exécution à la clôture suivant la fin de mois, 0,10 % de coût sur le montant échangé."""
import numpy as np
import pandas as pd
import yfinance as yf

SECTORS = ("FSPTX FSELX FBIOX FSPHX FSENX FSRBX FSCHX FDFAX FSUTX FSRFX FSAGX FSCSX FSRPX FSHOX FBMPX "
           "FSDPX FSTCX FSDAX FSLBX FSVLX FSPCX FSHCX FIDSX VGPMX VGENX VGHCX").split()
BOND, SPX, CASH = "VUSTX", "VFINX", "CASH"
COST = 0.001
LOOK = {"12m": 252, "6m": 126, "3m": 63}


def load_sector(first_year=1984):
    px = yf.download(SECTORS + [BOND, SPX], start=f"{first_year}-01-01", auto_adjust=True, progress=False)["Close"]
    irx = yf.download("^IRX", start=f"{first_year}-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
    px = px.ffill(limit=5)
    px[CASH] = (1 + (irx.reindex(px.index).ffill() / 100 / 252).fillna(0)).cumprod()
    return px


def momentum(px, kind):
    if kind == "mix":
        return sum(px.pct_change(n) for n in (21, 63, 126, 252)) / 4
    return px.pct_change(LOOK[kind])


def run_sector(px, start_year, capital, score="12m", top=5, regime=True, off=BOND):
    mom = momentum(px[SECTORS], score)
    sma = px[SPX].rolling(200).mean()
    idx = px.index
    is_me = pd.Series(idx, idx).groupby([idx.year, idx.month]).transform("max") == idx
    first = idx.searchsorted(pd.Timestamp(f"{start_year}-01-01"))
    cols = list(px.columns)
    P = px.values
    col = {c: i for i, c in enumerate(cols)}
    sh = np.zeros(len(cols))
    cash = capital
    open_ = {}          # ticker -> trade (dict) en cours
    journal, equity, inmkt = [], {}, {}
    target = None
    for k in range(first, len(idx)):
        d, p = idx[k], np.nan_to_num(P[k])
        if target is not None:      # exécution de la cible décidée à la fin de mois précédente
            val = cash + (sh * p).sum()
            want = {t: val * w for t, w in target.items()}
            for t in list(open_):   # positions à réduire ou fermer
                cur = sh[col[t]] * p[col[t]]
                goal = want.get(t, 0.0)
                if goal < cur:
                    sell = cur - goal
                    sh[col[t]] -= sell / p[col[t]]
                    cash += sell * (1 - COST)
                    open_[t]["cap_out"] += sell * (1 - COST)
                    if goal == 0:
                        tr = open_.pop(t)
                        tr.update(exit_date=d.date(), exit_price=round(p[col[t]], 2),
                                  ret_pct=round((p[col[t]] / tr["entry_price"] - 1) * 100, 1),
                                  exit_reason=target_reason.get(t, "sortie du top"))
                        sh[col[t]] = 0.0
            for t, goal in want.items():   # achats
                cur = sh[col[t]] * p[col[t]]
                if goal > cur and p[col[t]] > 0:
                    buy = min(goal - cur, cash)
                    sh[col[t]] += buy * (1 - COST) / p[col[t]]
                    cash -= buy
                    if t not in open_:
                        open_[t] = dict(year=d.year, winner=t, prev_year_ret_pct=signal.get(t), entry_date=d.date(),
                                        entry_price=round(p[col[t]], 2), exit_date=None, exit_price=None,
                                        exit_reason="", ret_pct=None, cap_in=0.0, cap_out=0.0)
                        journal.append(open_[t])
                    open_[t]["cap_in"] += buy
            target = None
        val = cash + (sh * p).sum()
        equity[d] = val
        inmkt[d] = sum(sh[col[t]] * p[col[t]] for t in open_ if t in SECTORS) / val
        if is_me.iloc[k] and k + 1 < len(idx):   # décision de fin de mois
            risk_off = regime and P[k, col[SPX]] < sma.iloc[k]
            m = mom.iloc[k].dropna()
            if risk_off or len(m) < top:
                target, signal = {off: 1.0}, {off: None}
                target_reason = {t: "filtre SMA200 → " + ("obligations" if off == BOND else "cash") for t in open_}
            else:
                pick = m.nlargest(top)
                target = {t: 1 / top for t in pick.index}
                signal = {t: round(v * 100, 1) for t, v in pick.items()}
                target_reason = {t: ("retour en actions" if t in (BOND, CASH) else "sortie du top " + str(top)) for t in open_}
    for t, tr in open_.items():
        last = P[-1, col[t]]
        tr.update(exit_reason="ouverte", exit_price=round(last, 2), ret_pct=round((last / tr["entry_price"] - 1) * 100, 1),
                  cap_out=tr["cap_out"] + sh[col[t]] * last)
    return pd.Series(equity), pd.DataFrame(journal), pd.Series(inmkt)
