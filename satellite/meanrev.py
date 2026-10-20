"""Stratégie de repli (RSI court) en tendance haussière, capital partagé entre `slots` positions."""
import numpy as np
import pandas as pd

COST = 0.0005  # par côté


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def run_meanrev(px, start_year, capital, n=5, thr=20, m=10, stop=0.10, maxd=10, slots=5):
    st = px.drop(columns="SPY")
    R = st.apply(lambda s: rsi(s.dropna(), n)).reindex(st.index)
    T = st > st.rolling(200).mean()
    X = st > st.rolling(m).mean()
    days = st.index[st.index.year >= start_year]
    cash, pos, equity, inmkt, journal = capital, {}, {}, {}, []
    for k, d in enumerate(days):
        i = st.index.get_loc(d)
        row = st.iloc[i]
        for t in list(pos):
            p = row[t]
            if pd.isna(p):
                continue
            q = pos[t]
            held = i - q["i"]
            why = None
            if p <= q["e"] * (1 - stop):
                why = f"stop {stop:.0%}"
            elif X[t].iloc[i]:
                why = f"clôture > SMA{m}"
            elif held >= maxd:
                why = f"{maxd} jours"
            if why:
                out = q["sh"] * p * (1 - COST)
                cash += out
                journal.append(dict(year=q["d"].year, winner=t, prev_year_ret_pct=q["rsi"], entry_date=q["d"].date(),
                                    entry_price=round(q["e"], 2), exit_date=d.date(), exit_price=round(p, 2),
                                    exit_reason=why, ret_pct=round((p / q["e"] - 1) * 100, 1),
                                    cap_in=q["cap"], cap_out=out))
                del pos[t]
        if i > 0:
            prev = i - 1
            sig = [t for t in st.columns if t not in pos and pd.notna(row[t]) and pd.notna(R[t].iloc[prev])
                   and R[t].iloc[prev] < thr and T[t].iloc[prev]]
            sig.sort(key=lambda t: R[t].iloc[prev])
            eqv = cash + sum(q["sh"] * row[t] for t, q in pos.items() if pd.notna(row[t]))
            for t in sig:
                if len(pos) >= slots or cash <= 1:
                    break
                size = min(cash, eqv / slots)
                pos[t] = dict(sh=size * (1 - COST) / row[t], e=row[t], i=i, d=d, cap=size, rsi=round(R[t].iloc[prev], 1))
                cash -= size
        inv = sum(q["sh"] * row[t] for t, q in pos.items() if pd.notna(row[t]))
        equity[d] = cash + inv
        inmkt[d] = inv / equity[d]
    last = days[-1]
    for t, q in pos.items():  # positions ouvertes : marquées à la dernière clôture
        p = st[t].iloc[-1]
        journal.append(dict(year=q["d"].year, winner=t, prev_year_ret_pct=q["rsi"], entry_date=q["d"].date(),
                            entry_price=round(q["e"], 2), exit_date=None, exit_price=round(p, 2),
                            exit_reason="ouverte", ret_pct=round((p / q["e"] - 1) * 100, 1),
                            cap_in=q["cap"], cap_out=q["sh"] * p))
    jr = pd.DataFrame(journal).sort_values("entry_date").reset_index(drop=True)
    return pd.Series(equity), jr, pd.Series(inmkt)
