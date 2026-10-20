#!/usr/bin/env python3
"""Satellite "gagnant de l'an passé" : 100 % sur le meilleur titre de l'année précédente,
rotation au 1er jour de bourse de janvier, stop à 35 % sous le plus haut depuis l'entrée.

Usage: python rotation_satellite.py [--start 2001] [--capital 10000] [--stop 0.35]
"""
import argparse
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "META", "JPM", "XOM", "JNJ",
           "PG", "WMT", "KO", "PEP", "HD", "UNH", "V", "MA", "CVX", "PFE", "CSCO",
           "ORCL", "INTC", "AMD", "NFLX"]
FEE = 10.0          # $ par transaction
KILL = 0.50         # arrêt si la valeur tombe à 50 % du capital initial


def load(tickers, first_year):
    px = yf.download(tickers + ["SPY"], start=f"{first_year - 1}-01-01", auto_adjust=True,
                     progress=False)["Close"]
    return px.dropna(how="all")


def run(px, start_year, capital, stop_pct):
    stocks = px.drop(columns="SPY")
    last_year = px.index[-1].year
    cash, shares, tkr, peak = capital, 0.0, None, 0.0
    killed, journal, equity = False, [], {}
    cur = None  # entrée en cours (ligne du journal)

    def close_pos(d, p, reason):
        nonlocal cash, shares, tkr, cur
        cash += shares * p - FEE
        cur.update(exit_date=d.date(), exit_price=round(p, 2), exit_reason=reason,
                   ret_pct=round((p / cur["entry_price"] - 1) * 100, 1))
        shares, tkr, cur = 0.0, None, None

    for y in range(start_year, last_year + 1):
        yr = stocks[stocks.index.year == y]
        if yr.empty:
            continue
        prev = stocks[stocks.index.year == y - 1]
        prev2 = stocks[stocks.index.year == y - 2]
        # rendement de l'année civile y-1 ; nécessite un historique complet
        base = prev2.iloc[-1] if not prev2.empty else prev.iloc[0]
        ret = (prev.iloc[-1] / base - 1).dropna()
        ret = ret[prev[ret.index].notna().sum() > 200]
        winner = ret.idxmax() if len(ret) else None
        for d, row in yr.iterrows():
            if killed:
                equity[d] = cash
                continue
            if d == yr.index[0] and winner is not None:
                if tkr != winner:
                    if tkr is not None:
                        close_pos(d, row[tkr], "rotation")
                    p = row[winner]
                    if pd.notna(p):
                        cash -= FEE
                        shares, tkr, peak = (cash) / p, winner, p
                        cash = 0.0
                        cur = dict(year=y, winner=winner, prev_year_ret_pct=round(ret[winner] * 100, 1),
                                   entry_date=d.date(), entry_price=round(p, 2), exit_date=None,
                                   exit_price=None, exit_reason="", ret_pct=None)
                        journal.append(cur)
                else:
                    # gagnant identique : on garde, aucune transaction
                    journal.append(dict(year=y, winner=winner, prev_year_ret_pct=round(ret[winner] * 100, 1),
                                        entry_date="(gardé)", entry_price=None, exit_date=None,
                                        exit_price=None, exit_reason="", ret_pct=None))
            if tkr is not None:
                p = row[tkr]
                if pd.isna(p):               # rachat/delist : sortie au dernier prix connu
                    close_pos(d, stocks[tkr].dropna().iloc[-1], "delist/rachat")
                else:
                    peak = max(peak, p)
                    if p <= (1 - stop_pct) * peak:
                        close_pos(d, p, f"stop {stop_pct:.0%} (plus haut {peak:.2f})")
            eq = cash + (shares * row[tkr] if tkr is not None and pd.notna(row[tkr]) else 0.0)
            equity[d] = eq
            if eq <= KILL * capital:
                if tkr is not None:
                    close_pos(d, row[tkr], "KILL SWITCH")
                killed = True
                equity[d] = cash
    # position encore ouverte : on la marque à la dernière clôture
    if cur is not None:
        cur["exit_reason"] = "ouverte"
        cur["exit_price"] = round(stocks[tkr].dropna().iloc[-1], 2)
        cur["ret_pct"] = round((cur["exit_price"] / cur["entry_price"] - 1) * 100, 1)
    return pd.Series(equity), pd.DataFrame(journal)


def stats(eq, label):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    return f"{label:<22} final {eq.iloc[-1]:>12,.0f}  CAGR {cagr:6.1%}  MaxDD {mdd:6.1%}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2001)
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--stop", type=float, default=0.35)
    a = ap.parse_args()
    px = load(TICKERS, a.start)
    eq, jr = run(px, a.start, a.capital, a.stop)
    eq_nostop, _ = run(px, a.start, a.capital, 0.999)
    spy = px["SPY"][px.index >= eq.index[0]]
    spy = spy / spy.iloc[0] * a.capital
    jr.to_csv("journal.csv", index=False)
    eq.to_csv("equity.csv", header=["equity"])
    pd.set_option("display.width", 200)
    print(jr.to_string(index=False))
    print()
    print(stats(eq, f"Stop {a.stop:.0%}"))
    print(stats(eq_nostop, "Sans stop (kill actif)"))
    print(stats(spy, "SPY buy & hold"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = eq.plot(label=f"Rotation + stop {a.stop:.0%}", logy=True, figsize=(10, 5))
    eq_nostop.plot(ax=ax, label="Sans stop (kill actif)", alpha=.6); spy.plot(ax=ax, label="SPY")
    ax.legend(); ax.set_title("Satellite gagnant de l'an passé (log)"); plt.tight_layout()
    plt.savefig("equity.png", dpi=120)
