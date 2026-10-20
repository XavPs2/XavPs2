#!/usr/bin/env python3
"""Rapport HTML façon TextToQuant pour les stratégies du satellite.

Usage:
  python report.py                      # rotation  -> report.html
  python report.py --strategy meanrev   # repli RSI sur 25 titres -> report_meanrev.html
  python report.py --strategy dip       # repli RSI(3) sur le S&P 500 depuis 1980 -> report_dip.html
  python report.py --strategy sector    # rotation sectorielle + filtre SMA200 depuis 1988 -> report_sector.html
  python report.py --strategy all       # les quatre
Options: --start ANNÉE --capital MONTANT. Graphiques Chart.js chargés depuis cdnjs (connexion requise).
"""
import argparse
import json
import math
import numpy as np
import pandas as pd
from rotation_satellite import TICKERS, FEE, KILL, load, run
from meanrev import run_meanrev
from dip import load_index, run_dip
from sector import SECTORS, load_sector, run_sector

FILES = {"rotation": "report.html", "meanrev": "report_meanrev.html", "dip": "report_dip.html", "sector": "report_sector.html"}
TITLES = {"rotation": "ROTATION", "meanrev": "REPLI RSI · 25 TITRES", "dip": "REPLI RSI(3) · S&P 500", "sector": "ROTATION SECTORIELLE"}


def fnum(x, nd=2):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def pct(x, nd=1):
    return fnum(None if x is None else x * 100, nd)


# ---------------------------------------------------------------- stratégies
def prepare(name, a, cache):
    """Retourne (runner, prix, benchmark, infos). runner(**kw) -> (equity, journal, in_market)."""
    if name in ("rotation", "meanrev"):
        start = a.start or 2001
        if "stocks" not in cache:
            cache["stocks"] = load(TICKERS, min(start, 2001))
        px = cache["stocks"]
        prices, bench = px.drop(columns="SPY"), px["SPY"]
    elif name == "sector":
        start = a.start or 1988
        prices = load_sector(min(start, 1988) - 4)
        bench = prices["VFINX"]
    else:
        start = a.start or 1980
        idx = load_index(start)
        prices, bench = idx.to_frame(), idx
    cap = a.capital
    if name == "rotation":
        kw = dict(stop=0.35)
        runner = lambda stop=0.35: run(px, start, cap, stop)
        variants = [(f"stop {s:.0%}" if s < 1 else "sans stop", dict(stop=s)) for s in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.999)]
        info = dict(chips=[f"{len(TICKERS)} titres", "1d", "Long", f"Frais {FEE:.0f} $/trans.", "Stop 35 % du plus haut", f"Kill {KILL:.0%}"],
                    universe=" · ".join(TICKERS), prevlabel="RDT ANNÉE PRÉC.", bench_label="SPY",
                    entry=["1er jour de bourse de janvier : 100 % sur le meilleur rendement total de l'année précédente"],
                    exit=["Clôture ≤ 65 % du plus haut depuis l'entrée → cash jusqu'en janvier",
                          "Rotation en janvier (sauf si même gagnant)", f"Valeur ≤ {KILL:.0%} du capital initial → arrêt"],
                    assumptions=f"entrée à la clôture du 1er jour de janvier · {FEE:.0f} $/transaction · fractions d'action · aucun impôt ni change · univers à biais de survivant",
                    note="Une seule position : 100 % du satellite sur le gagnant.", compound=True)
        lines = lambda t, p: ([round(x * (1 - kw["stop"]), 2) for x in p.cummax()], None)
    elif name == "meanrev":
        kw = dict(n=5, thr=20, m=10, stop=0.10, maxd=10, slots=5)
        runner = lambda **k: run_meanrev(px, start, cap, **{**kw, **k})
        variants = [(f"RSI(5) < {t}", dict(thr=t)) for t in (10, 15, 20, 25, 30)] + [("10 positions", dict(slots=10))]
        info = dict(chips=[f"{len(TICKERS)} titres", "1d", "Long", "5 positions max", "Frais 0,05 %/côté", "Stop 10 %"],
                    universe=" · ".join(TICKERS), prevlabel="RSI SIGNAL", bench_label="SPY",
                    entry=["RSI(5) < 20 la veille et clôture > SMA200 : achat à la clôture suivante, max 5 positions (capital ÷ 5 chacune)"],
                    exit=["Clôture > SMA10", "Durée max 10 jours", "Stop 10 % sous l'entrée", "Frais 0,05 % par côté"],
                    assumptions="signal à la clôture, achat à la clôture suivante · fractions d'action · aucun impôt ni change · univers à biais de survivant",
                    note="Capital partagé entre 5 positions maximum.", compound=False)
        lines = lambda t, p: ([round(t["entry_price"] * (1 - kw["stop"]), 2)] * len(p), None)
    elif name == "sector":
        kw = dict(score="12m", top=5, regime=True, off="VUSTX")
        runner = lambda **k: run_sector(prices, start, cap, **{**kw, **k})
        variants = [(f"momentum {sc} · top {t}", dict(score=sc, top=t)) for sc in ("12m", "6m", "3m", "mix") for t in (3, 5, 7)]
        variants += [("hors marché = cash", dict(off="CASH")), ("sans filtre SMA200", dict(regime=False))]
        info = dict(chips=[f"{len(SECTORS)} secteurs US", "1d", "Long", "Top 5 à parts égales", "Mensuel", "Coût 0,10 %"],
                    universe=" · ".join(SECTORS) + " — obligations : VUSTX", prevlabel="MOMENTUM 12M %", bench_label="S&P 500 TR",
                    entry=["Fin de mois : les 5 secteurs au meilleur rendement sur 12 mois, 20 % chacun, achat à la clôture suivante"],
                    exit=["Le secteur sort du top 5 à une fin de mois", "S&P 500 sous sa SMA200 en fin de mois → 100 % obligations du Trésor long terme (VUSTX)",
                          "Rééquilibrage à parts égales chaque mois", "Coût 0,10 % du montant échangé"],
                    assumptions="fonds sectoriels Fidelity Select / Vanguard, frais et dividendes inclus · référence : fonds S&P 500 VFINX dividendes réinvestis · fonds encore existants (biais de survivant) · aucun impôt",
                    note="Le P/L par trade inclut les achats et ventes de rééquilibrage mensuel.", compound=False)
        lines = lambda t, p: (None, None)
    else:
        kw = dict(n=3, thr=20, trend=200, tp=0.01, stop=0.05, maxd=20)
        runner = lambda **k: run_dip(prices.iloc[:, 0], start, cap, **{**kw, **k})
        variants = [(f"RSI(3) < {t} · cible {g:.1%}", dict(thr=t, tp=g)) for t in (10, 15, 20, 25, 30) for g in (0.005, 0.01, 0.02)]
        variants += [("stop 3 %", dict(stop=0.03)), ("stop 10 %", dict(stop=0.10)), ("sans filtre SMA200", dict(trend=None))]
        info = dict(chips=["S&P 500 (^GSPC)", "1d", "Long", "100 % du capital", "Frais 0,05 %/côté", "Cible +1 %", "Stop 5 %"],
                    universe="Indice S&P 500 en prix (sans dividendes) — pour investir : SPY, VOO ou VFV.TO", prevlabel="RSI SIGNAL",
                    bench_label="S&P 500",
                    entry=["RSI(3) < 20 la veille et clôture > SMA200 : achat à la clôture suivante"],
                    exit=["Cible +1 % au-dessus de l'entrée", "Stop 5 % sous l'entrée", "Durée max 20 jours", "Frais 0,05 % par côté"],
                    assumptions="indice en prix sans dividendes (stratégie et référence) · signal à la clôture, achat à la clôture suivante · aucun impôt",
                    note="Une seule position à la fois sur l'indice.", compound=True)
        lines = lambda t, p: ([round(t["entry_price"] * (1 - kw["stop"]), 2)] * len(p),
                              [round(t["entry_price"] * (1 + kw["tp"]), 2)] * len(p))
    info["start"] = start
    return runner, variants, prices, bench, lines, info


# ---------------------------------------------------------------- analyse
def trades_of(jr, eq):
    tr = jr[jr["entry_date"] != "(gardé)"].copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    tr["exit_dt"] = pd.to_datetime(tr["exit_date"]).fillna(eq.index[-1])
    tr["net"] = tr["cap_out"] / tr["cap_in"] - 1
    tr["pnl"] = tr["cap_out"] - tr["cap_in"]
    return tr.reset_index(drop=True)


def windows3(eq):
    out = []
    for y in range(eq.index[0].year, eq.index[-1].year - 2):
        before = eq[eq.index < f"{y}-01-01"]
        a = before.iloc[-1] if len(before) else eq.iloc[0]
        b = eq[eq.index < f"{y + 3}-01-01"].iloc[-1]
        out.append((y, b / a - 1))
    return out


def yearly(eq):
    ye = eq.resample("YE").last()
    base = pd.concat([pd.Series([eq.iloc[0]], index=[eq.index[0] - pd.Timedelta(days=1)]), ye])
    return ye / base.shift(1).iloc[1:].values - 1


def core(eq, tr):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = eq / eq.cummax() - 1
    sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() else np.nan
    w = windows3(eq)
    return dict(cagr=cagr, mdd=dd.min(), sharpe=sharpe, trades=len(tr),
                wr=(tr["net"] > 0).mean() if len(tr) else np.nan, avg=tr["net"].mean() if len(tr) else np.nan,
                win3=int(sum(x > 0 for _, x in w)), n3=len(w), final=eq.iloc[-1])


def drawdowns(eq, k=5):
    dd = eq / eq.cummax() - 1
    out, start = [], None
    for d, v in dd.items():
        if v < 0 and start is None:
            start = d
        elif v == 0 and start is not None:
            seg = dd[start:d]
            out.append((start, seg.idxmin(), d, seg.min()))
            start = None
    if start is not None:
        seg = dd[start:]
        out.append((start, seg.idxmin(), None, seg.min()))
    out.sort(key=lambda x: x[3])
    return [dict(peak=str(s.date()), trough=str(t.date()), rec=str(e.date()) if e is not None else "non récupéré",
                 depth=pct(v), days=((e or eq.index[-1]) - s).days) for s, t, e, v in out[:k]]


def monte_carlo(eq, n=1000, block=21, seed=7):
    r = eq.pct_change().dropna().values
    L = len(r)
    rng = np.random.default_rng(seed)
    nb = L // block + 1
    starts = rng.integers(0, L - block, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :L]
    paths = np.cumprod(1 + r[idx], axis=1)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = paths[:, -1] ** (1 / yrs) - 1
    mdd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
    step = max(1, L // 400)
    q = np.percentile(paths[:, ::step], [5, 25, 50, 75, 95], axis=0) * eq.iloc[0]
    hist, edges = np.histogram(mdd * 100, bins=12)
    p = lambda v, qq: pct(np.percentile(v, qq))
    return dict(t=[d.strftime("%Y-%m-%d") for d in eq.index[1:][::step]],
                q=[[fnum(x, 0) for x in row] for row in q],
                cagr=[p(cagr, 5), p(cagr, 50), p(cagr, 95)], mdd=[p(mdd, 5), p(mdd, 50), p(mdd, 95)],
                p_loss=pct((paths[:, -1] < 1).mean()), p_cagr_neg=pct((cagr < 0).mean()),
                p_dd50=pct((mdd < -0.5).mean()), n=n, block=block,
                hist=[int(x) for x in hist], edges=[round(float(x), 1) for x in edges])


def holdout(eq, tr, frac=0.7):
    cut = eq.index[int(len(eq) * frac)]
    res = {}
    for lab, e, t in (("IN-SAMPLE", eq[:cut], tr[tr.entry_date < cut]), ("OUT-OF-SAMPLE", eq[cut:], tr[tr.entry_date >= cut])):
        c = core(e, t)
        res[lab] = dict(period=f"{e.index[0].date()} – {e.index[-1].date()}", cagr=pct(c["cagr"]), mdd=pct(c["mdd"]),
                        sharpe=fnum(c["sharpe"]), trades=c["trades"], wr=pct(c["wr"]), avg=pct(c["avg"], 2))
    return res


def build(name, a, cache):
    runner, variants, prices, bench, lines, info = prepare(name, a, cache)
    eq, jr, inmkt = runner()
    cap = a.capital
    bh = bench.reindex(eq.index).ffill()
    bh = bh / bh.iloc[0] * cap
    tr = trades_of(jr, eq)
    tr["bars"] = [int(((eq.index >= e) & (eq.index <= x)).sum()) for e, x in zip(tr.entry_date, tr.exit_dt)]

    mae, mfe, paths = [], [], []
    for i, t in tr.iterrows():
        s = prices[t["winner"]].dropna()
        p = s.loc[t.entry_date:t.exit_dt]
        rel = p / p.iloc[0] - 1
        mae.append(rel.min()); mfe.append(rel.max())
        if i >= len(tr) - 100:  # chemins de prix des 100 derniers trades seulement
            a0, b0 = s.index.get_loc(p.index[0]), s.index.get_loc(p.index[-1])
            ctx = s.iloc[max(0, a0 - 15): b0 + 16]
            stp, tgt = lines(t, p)
            inside = (ctx.index >= p.index[0]) & (ctx.index <= p.index[-1])
            st_full = [None] * len(ctx); tg_full = [None] * len(ctx)
            pos = np.where(inside)[0]
            for k_, j in enumerate(pos):
                st_full[j] = stp[k_] if stp else None
                tg_full[j] = tgt[k_] if tgt else None
            step = max(1, len(ctx) // 300)
            paths.append(dict(t=[d.strftime("%Y-%m-%d") for d in ctx.index[::step]],
                              p=[round(float(v), 2) for v in ctx.values[::step]], s=st_full[::step], g=tg_full[::step],
                              ein=t.entry_date.strftime("%Y-%m-%d"), eout=t.exit_dt.strftime("%Y-%m-%d")))
        else:
            paths.append(None)
    tr["mae"], tr["mfe"] = mae, mfe

    # métriques de la courbe
    r = eq.pct_change().dropna()
    c = core(eq, tr)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    dn = r[r < 0]
    sortino = r.mean() / math.sqrt((dn ** 2).sum() / len(r)) * math.sqrt(252) if len(dn) else np.nan
    omega = r[r > 0].sum() / -dn.sum() if len(dn) else np.nan
    rb = bh.pct_change().dropna().reindex(r.index).fillna(0)
    beta = np.cov(r, rb)[0, 1] / rb.var()
    alpha = (r.mean() - beta * rb.mean()) * 252
    vol = r.std() * math.sqrt(252)
    ulcer = math.sqrt((dd ** 2).mean()) * 100
    cb = core(bh, tr.iloc[0:0])
    yr_s, yr_b = yearly(eq), yearly(bh)

    wins, losses = tr[tr.net > 0], tr[tr.net <= 0]
    wr = len(wins) / len(tr)
    pf = wins.pnl.sum() / -losses.pnl.sum() if len(losses) and losses.pnl.sum() else np.nan
    avg_w = wins.net.mean() if len(wins) else 0
    avg_l = losses.net.mean() if len(losses) else 0
    payoff = avg_w / -avg_l if avg_l else np.nan
    kelly = wr - (1 - wr) / payoff if payoff and not np.isnan(payoff) else np.nan
    sgn = (tr.net > 0).tolist()

    def longest(v):
        best = cur = 0
        for s in sgn:
            cur = cur + 1 if s == v else 0
            best = max(best, cur)
        return best

    # note maison en 4 piliers (pas la formule de TextToQuant)
    prof = max(0, min(100, c["cagr"] / 0.30 * 100))
    risk = max(0, min(100, (1 + c["mdd"] / 0.8) * 100))
    cons = max(0, min(100, c["win3"] / max(1, c["n3"]) * 70 + (30 if pf and pf > 1.5 else 15 if pf and pf > 1 else 0)))
    edge = max(0, min(100, (c["sharpe"] / 1.5) * 100)) if not np.isnan(c["sharpe"]) else 0
    score = round((prof + risk + cons + edge) / 4)
    letter = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "F"

    # calendrier mensuel (courbe de capital)
    me = eq.resample("ME").last()
    mret = me.pct_change()
    mret.iloc[0] = me.iloc[0] / eq.iloc[0] - 1
    cal = {}
    for d, v in mret.items():
        cal.setdefault(str(d.year), {})[str(d.month)] = pct(v)
    for y in cal:
        cal[y]["ytd"] = pct(yr_s[yr_s.index.year == int(y)].iloc[0])
    seas = [[v for d, v in mret.items() if d.month == m] for m in range(1, 13)]

    def excl(n, best):
        s = tr.sort_values("net", ascending=not best)
        if info["compound"]:
            return float((1 + tr.drop(s.index[:n]).net).prod() - 1)
        return float((eq.iloc[-1] - s.pnl.iloc[:n].sum()) / cap - 1)

    reasons = {}
    for _, t in tr.iterrows():
        why = str(t.exit_reason)
        k = "Stop" if why.startswith("stop") else "Cible" if why.startswith("cible") else "Filtre SMA200" if why.startswith("filtre") else "Sortie du top" if why.startswith("sortie") else {"rotation": "Rotation", "ouverte": "Ouverte"}.get(why, why)
        g = reasons.setdefault(k, {"n": 0, "w": 0, "pnl": 0.0})
        g["n"] += 1; g["w"] += int(t.net > 0); g["pnl"] += float(t.pnl)

    # sensibilité des paramètres
    sens = []
    for lab, k in variants:
        e2, j2, _ = runner(**k)
        c2 = core(e2, trades_of(j2, e2))
        sens.append(dict(label=lab, trades=c2["trades"], wr=pct(c2["wr"]), avg=pct(c2["avg"], 2), cagr=pct(c2["cagr"]), mdd=pct(c2["mdd"]),
                         sharpe=fnum(c2["sharpe"]), win3=f'{c2["win3"]}/{c2["n3"]}', final=fnum(c2["final"], 0)))
        print("  variante", lab, "ok")

    step = max(1, len(eq) // 1500)
    ei = eq.index[::step].union(eq.index[-1:])
    roll = eq.pct_change(252)
    rollb = bh.pct_change(252)
    hist, edges = np.histogram(tr.net * 100, bins=12)
    dh, de = np.histogram(tr.bars, bins=min(12, max(3, tr.bars.nunique())))
    w3s, w3b = windows3(eq), dict(windows3(bh))
    trades = [dict(n=i + 1, tkr=t.winner, prev=fnum(t.prev_year_ret_pct, 1), ein=t.entry_date.strftime("%Y-%m-%d"),
                   pin=t.entry_price, eout=t.exit_dt.strftime("%Y-%m-%d"), pout=t.exit_price, why=t.exit_reason,
                   net=pct(t.net, 2), pnl=fnum(t.pnl, 0), bars=int(t.bars), mae=pct(t.mae), mfe=pct(t.mfe), path=paths[i])
              for i, t in tr.iterrows()]
    kept = [int(y) for y in jr[jr.entry_date == "(gardé)"]["year"]]
    sl = slice(None) if len(tr) <= 1500 else slice(-1500, None)
    return dict(
        params=dict(strategy=name, title=TITLES[name], files=FILES, titles=TITLES, capital=cap,
                    period=f"{eq.index[0].date()} – {eq.index[-1].date()}", kept=kept, **info),
        head=dict(ret=pct(eq.iloc[-1] / eq.iloc[0] - 1), win=pct(wr), trades=len(tr)),
        score=dict(score=score, letter=letter, prof=round(prof), risk=round(risk), cons=round(cons), edge=round(edge)),
        m=dict(net=fnum(eq.iloc[-1] - cap, 0), cagr=pct(c["cagr"]), wins=len(wins), n=len(tr), mdd=pct(c["mdd"]),
               trough=str(dd.idxmin().date()), pf=fnum(pf), sharpe=fnum(c["sharpe"]), exp=pct(tr.net.mean(), 2),
               sortino=fnum(sortino), kelly=pct(kelly), omega=fnum(omega), alpha=pct(alpha), beta=fnum(beta),
               avg_w=pct(avg_w, 2), avg_l=pct(avg_l, 2), payoff=fnum(payoff), inmkt=pct(inmkt.mean()),
               dur=fnum(tr.bars.mean(), 1), tpm=fnum(len(tr) / (yrs * 12)), sw=longest(True), sl=longest(False),
               best=pct(tr.net.max(), 2), worst=pct(tr.net.min(), 2), med=pct(tr.net.median(), 2), mean=pct(tr.net.mean(), 2),
               ex_best=pct(excl(5, True)), ex_worst=pct(excl(5, False)), avg_mae=pct(tr.mae.mean()),
               avg_mfe=pct(tr.mfe.mean()), med_hold=fnum(tr.bars.median(), 0), vol=pct(vol), calmar=fnum(c["cagr"] / -c["mdd"]),
               ulcer=fnum(ulcer, 1), bcagr=pct(cb["cagr"]), bmdd=pct(cb["mdd"]), bsharpe=fnum(cb["sharpe"]),
               posy=pct((yr_s > 0).mean(), 0), besty=pct(yr_s.max()), worsty=pct(yr_s.min()),
               beaty=pct((yr_s.values > yr_b.values).mean(), 0), win3=c["win3"], n3=c["n3"], bwin3=cb["win3"]),
        eq=dict(t=[d.strftime("%Y-%m-%d") for d in ei], s=[fnum(v, 0) for v in eq.loc[ei]], b=[fnum(v, 0) for v in bh.loc[ei]],
                dd=[pct(v) for v in dd.loc[ei]], r12=[pct(v) for v in roll.loc[ei]], b12=[pct(v) for v in rollb.loc[ei]]),
        years=dict(y=[int(d.year) for d in yr_s.index], s=[pct(v) for v in yr_s], b=[pct(v) for v in yr_b]),
        w3=dict(y=[f"{y}–{y + 2}" for y, _ in w3s], s=[pct(v) for _, v in w3s], b=[pct(w3b[y]) for y, _ in w3s]),
        dds=drawdowns(eq), mc=monte_carlo(eq), ho=holdout(eq, tr), sens=sens,
        cal=cal, seas=[pct(np.mean(v)) if v else None for v in seas], seas_p=[pct(np.mean([x > 0 for x in v]), 0) if v else None for v in seas],
        hist=dict(h=[int(x) for x in hist], e=[round(float(x), 1) for x in edges]),
        dur=dict(h=[int(x) for x in dh], e=[round(float(x)) for x in de]),
        scat=dict(mae=[pct(v) for v in tr.mae.iloc[sl]], net=[pct(v, 2) for v in tr.net.iloc[sl]]),
        reasons=reasons, trades=trades[-100:], ntrades=len(trades))


HTML = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Satellite – Rapport</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0a141c;--panel:#0d1a23;--line:#1b2c38;--tx:#d5e0e6;--mut:#6f8593;--up:#1dc7a0;--dn:#d6386a;--warn:#f0b429;--bl:#3b82f6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:center;gap:28px;padding:10px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header b{letter-spacing:.06em}.tabs a{color:var(--mut);margin-right:18px;text-decoration:none}.tabs a.on{color:var(--tx);border-bottom:2px solid var(--up);padding-bottom:6px}
.wrap{display:grid;grid-template-columns:250px 1fr;min-height:calc(100vh - 44px)}
aside{border-right:1px solid var(--line);padding:14px;font-size:12px}
aside h4{margin:16px 0 6px;color:var(--mut);font-size:10px;letter-spacing:.12em;font-weight:400}
.q{border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:10px;color:var(--mut);line-height:1.6}
.chip{display:inline-block;border:1px solid var(--line);border-radius:12px;padding:1px 8px;margin:2px 2px 2px 0;font-size:11px;color:var(--mut)}
.rule{border:1px solid #17795f;background:#0c2a27;border-radius:6px;padding:7px 9px;margin:5px 0;color:var(--tx)}
.rule i{font-style:normal;float:right;color:var(--up);font-size:10px}.rule.x{border-color:#7b2a45;background:#2a1220}
main{padding:14px 20px;min-width:0}
.top{display:flex;align-items:center;gap:26px;flex-wrap:wrap;margin-bottom:8px}.top .big{font-size:15px;font-weight:700}
.k{color:var(--mut);font-size:10px;letter-spacing:.1em;margin-right:6px}
.bar{border:1px solid var(--line);background:#14180f;padding:5px 10px;font-size:11px;color:#c6b25a;margin-bottom:8px}
.panel{border:1px solid var(--line);background:var(--panel);border-radius:4px;margin:12px 0}
.ph{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.ph .tag{border:1px solid var(--line);font-size:10px;color:var(--mut);padding:1px 6px}.ph b{letter-spacing:.04em}.ph .r{margin-left:auto;color:var(--mut)}
.warn{color:var(--warn);font-size:10px}
.grid{display:grid;grid-template-columns:230px repeat(4,1fr)}
.grade{padding:18px;border-right:1px solid var(--line);grid-row:span 2}
.grade .L{font-size:44px;font-weight:700;line-height:1}.grade small{color:var(--mut)}
.cell{padding:12px 14px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.cell .l{color:var(--mut);font-size:10px;letter-spacing:.1em}.cell .v{font-size:20px;margin:3px 0}.cell .s{color:var(--mut);font-size:10px}
.row{display:grid;grid-template-columns:repeat(5,1fr);border-top:1px solid var(--line)}
.row div{padding:9px 14px;display:flex;justify-content:space-between;gap:8px;border-right:1px solid var(--line)}
.row div span{color:var(--mut);font-size:10px;letter-spacing:.1em}
.up{color:var(--up)}.dn{color:var(--dn)}
.pil{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;padding:12px 14px}
.pil .t{display:flex;justify-content:space-between;color:var(--mut);font-size:10px;letter-spacing:.1em}
.pil .tr{height:3px;background:var(--line);margin-top:6px}.pil .tr i{display:block;height:3px;background:var(--up)}
.cv{padding:10px 14px}canvas{max-width:100%}
select{background:var(--bg);color:var(--tx);border:1px solid var(--line);padding:3px 6px;font:inherit}
.cal{width:100%;border-collapse:separate;border-spacing:3px;padding:8px}.cal th{color:var(--mut);font-weight:400;font-size:10px}
.cal td{text-align:center;padding:6px 0;border:1px solid var(--line);font-size:11px}
.g2{display:grid;grid-template-columns:1fr 1fr}.g2>div{padding:14px;border-right:1px solid var(--line)}
.g3{display:grid;grid-template-columns:repeat(3,1fr)}.g3>div{padding:14px;border-right:1px solid var(--line)}
.big2{font-size:24px}.note{color:var(--mut);font-size:11px;border-left:2px solid var(--line);padding-left:10px;margin-top:10px}
table.t{width:100%;border-collapse:collapse;font-size:12px}.t th{color:var(--mut);font-weight:400;font-size:10px;text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
.t td{text-align:right;padding:6px 8px;border-bottom:1px solid #112029}.t td.l,.t th.l{text-align:left}
.t.click tr{cursor:pointer}.t.click tr:hover{background:#12242f}.t tr.base td{background:#0c2a27}
.scroll{max-height:520px;overflow:auto}
@media(max-width:900px){.wrap{grid-template-columns:1fr}aside{display:none}.grid,.row,.pil,.g2,.g3{grid-template-columns:1fr 1fr}.grade{grid-row:auto}}
</style></head><body>
<header><b>◆ SATELLITE</b><div class="tabs" id="tabs"></div></header>
<div class="wrap"><aside>
<h4>ACTIVE QUERY</h4><div class="q" id="qbox"></div>
<h4>ENTRY</h4><div id="re"></div>
<h4>EXIT</h4><div id="rx"></div>
<h4>RERUN</h4><div class="q" id="rr"></div>
</aside><main id="m"></main></div>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s),f=(x,d=2,suf='')=>x==null?'–':Number(x).toFixed(d)+suf;
const cls=x=>x==null?'':x>=0?'up':'dn',sg=(x,d=1,s='%')=>x==null?'–':(x>=0?'+':'')+Number(x).toFixed(d)+s;
const money=x=>x==null?'–':Math.round(x).toLocaleString('fr-CA')+' $';
const P=D.params,M=D.m,S=D.score,MC=D.mc,BL=P.bench_label;
document.title=P.title+' – Rapport';
$('#tabs').innerHTML=Object.entries(P.files).map(([k,v])=>`<a href="${v}" class="${k==P.strategy?'on':''}">${P.titles[k]}</a>`).join('');
$('#qbox').innerHTML=P.chips.map(c=>`<span class="chip">${c}</span>`).join('')+`<span class="chip">${P.period}</span><span class="chip">Capital ${money(P.capital)}</span><div style="margin-top:8px;font-size:11px">${P.universe}</div>`;
$('#re').innerHTML=P.entry.map(r=>`<div class="rule">${r}<i>LONG</i></div>`).join('');
$('#rx').innerHTML=P.exit.map(r=>`<div class="rule x">${r}</div>`).join('');
$('#rr').textContent=`python report.py --strategy ${P.strategy} --start ${P.start} --capital ${P.capital}`;
const cell=(l,v,s,c='')=>`<div class="cell"><div class="l">${l}</div><div class="v ${c}">${v}</div><div class="s">${s}</div></div>`;
const rw=(l,v,c='')=>`<div><span>${l}</span><b class="${c}">${v}</b></div>`;
const panel=(tag,title,body,right='')=>`<div class="panel"><div class="ph"><span class="tag">${tag}</span><b>${title}</b>${right}</div>${body}</div>`;
const H=D.ho['IN-SAMPLE'],O=D.ho['OUT-OF-SAMPLE'];
let h=`<div class="top"><span class="big">SATELLITE · ${P.title}</span><span>1d</span><span style="color:var(--mut)">${P.period}</span>
<span><span class="k">RETURN</span><b class="${cls(D.head.ret)}">${sg(D.head.ret)}</b></span><span><span class="k">CAGR</span><b class="${cls(M.cagr)}">${sg(M.cagr)}</b></span><span><span class="k">WIN</span><b>${f(D.head.win,2,'%')}</b></span><span><span class="k">TRADES</span><b>${D.head.trades}</b></span><span><span class="k">MAX DD</span><b class="dn">${f(M.mdd,1,'%')}</b></span></div>
<div class="bar">● Hypothèses : ${P.assumptions}</div>`;
h+=panel('METRICS','PERFORMANCE',`
<div class="grid"><div class="grade"><div class="k">STRATEGY GRADE (maison)</div><div class="L ${S.score>=50?'up':'dn'}">${S.letter} <small>${S.score}/100</small></div><div class="k" style="margin-top:8px">${S.score>=65?'DEPLOY':'AVOID'}</div><div class="note">Note maison : rendement, drawdown, régularité sur 3 ans, Sharpe. Ce n'est pas la formule de TextToQuant.</div></div>
${cell('NET PROFIT',sg(M.net,0,' $'),sg(D.head.ret)+' | CAGR '+sg(M.cagr),cls(M.net))}
${cell('WIN RATE',f(D.head.win,2,'%'),M.wins+' / '+M.n)}
${cell('MAX DD',f(M.mdd,2,'%'),'TROUGH '+M.trough,'dn')}
${cell('PROFIT FAC.',f(M.pf),'≥ 1.0')}
${cell('SHARPE',f(M.sharpe),BL+' : '+f(M.bsharpe))}
${cell('EXPECTANCY',sg(M.exp,2),'PAR TRADE | %',cls(M.exp))}
${cell('SORTINO',f(M.sortino),'DOWNSIDE ONLY')}
${cell('KELLY',sg(M.kelly,1),M.kelly<0?'NO EDGE':'FRACTION',cls(M.kelly))}</div>
<div class="row">${rw('OMEGA',f(M.omega))}${rw('ALPHA vs '+BL,sg(M.alpha,1,'%/an'),cls(M.alpha))}${rw('BETA',f(M.beta))}${rw('AVG WIN',sg(M.avg_w,2),'up')}${rw('AVG LOSS',sg(M.avg_l,2),'dn')}</div>
<div class="row">${rw('PAYOFF',f(M.payoff)+' : 1')}${rw('IN MARKET',f(M.inmkt,1,'%'))}${rw('AVG DURATION',f(M.dur,0)+' bars')}${rw('TRADES/MO',f(M.tpm))}${rw('STREAK W / L',`<span class="up">${M.sw}</span> / <span class="dn">${M.sl}</span>`)}</div>
<div class="row">${rw('VOLATILITÉ',f(M.vol,1,'%/an'))}${rw('CALMAR',f(M.calmar))}${rw('ULCER',f(M.ulcer,1))}${rw('ANNÉES +',f(M.posy,0,'%'))}${rw('BAT '+BL,f(M.beaty,0,'% des ans'))}</div>
<div class="row">${rw('MEILLEURE ANNÉE',sg(M.besty),'up')}${rw('PIRE ANNÉE',sg(M.worsty),'dn')}${rw('3 ANS RENTABLES',M.win3+' / '+M.n3)}${rw(BL+' CAGR',sg(M.bcagr))}${rw(BL+' MAX DD',f(M.bmdd,1,'%'),'dn')}</div>
<div class="pil">${[['PROFIT',S.prof],['RISK',S.risk],['CONSIST',S.cons],['EDGE',S.edge]].map(([k,v])=>`<div><div class="t"><span>${k}</span><b>${v}</b></div><div class="tr"><i style="width:${v}%"></i></div></div>`).join('')}</div>`,D.head.trades<30?`<span class="warn">⚠ ${D.head.trades} TRADES</span>`:'');
h+=panel('CURVE','EQUITY CURVE',`<div class="cv"><canvas id="c1" height="110"></canvas></div><div class="cv"><canvas id="c2" height="38"></canvas></div><div class="cv"><div class="k">RENDEMENT GLISSANT 12 MOIS</div><canvas id="c7" height="45"></canvas></div>`,
 `<span class="r">START ${money(P.capital)} · END ${money(D.eq.s.at(-1))} · ${BL} ${money(D.eq.b.at(-1))}</span><label style="color:var(--mut)"><input type="checkbox" id="lg" checked> log</label>`);
h+=panel('YEARS','RENDEMENTS ANNUELS',`<div class="cv"><canvas id="c8" height="55"></canvas></div>`,`<span class="r">${M.posy}% d'années positives · bat ${BL} ${M.beaty}% des années</span>`);
h+=panel('ROBUST','FENÊTRES DE 3 ANS',`<div class="cv"><canvas id="c9" height="55"></canvas></div><div class="cv note">Rendement sur chaque période de 3 ans commençant en janvier. Stratégie rentable sur <b>${M.win3}/${M.n3}</b> périodes ; ${BL} sur <b>${M.bwin3}/${M.n3}</b>.</div>`);
h+=panel('DRAWDOWN','5 PIRES DRAWDOWNS',`<table class="t"><thead><tr><th class="l">#</th><th class="l">SOMMET</th><th class="l">CREUX</th><th class="l">RÉCUPÉRATION</th><th>PROFONDEUR</th><th>DURÉE (jours)</th></tr></thead><tbody>${D.dds.map((d,i)=>`<tr><td class="l">${i+1}</td><td class="l">${d.peak}</td><td class="l">${d.trough}</td><td class="l">${d.rec}</td><td class="dn">${f(d.depth,1,'%')}</td><td>${d.days}</td></tr>`).join('')}</tbody></table>`);
h+=panel('MONTE CARLO','SIMULATION MONTE CARLO',`<div class="g3"><div><div class="k">CAGR (5e / médian / 95e)</div><div class="big2">${sg(MC.cagr[0])} / ${sg(MC.cagr[1])} / ${sg(MC.cagr[2])}</div></div>
<div><div class="k">MAX DD (5e / médian / 95e)</div><div class="big2 dn">${f(MC.mdd[0],1,'%')} / ${f(MC.mdd[1],1,'%')} / ${f(MC.mdd[2],1,'%')}</div></div>
<div><div class="k">RISQUES</div><div>Perte sur toute la période : <b class="dn">${f(MC.p_loss,1,'%')}</b><br>Drawdown pire que −50 % : <b class="dn">${f(MC.p_dd50,1,'%')}</b></div></div></div>
<div class="g2" style="border-top:1px solid var(--line)"><div><div class="k">ÉVENTAIL DES COURBES (5e–95e centile)</div><canvas id="c10" height="110"></canvas></div><div><div class="k">DISTRIBUTION DU MAX DRAWDOWN</div><canvas id="c11" height="110"></canvas></div></div>
<div class="cv note">${MC.n} séquences obtenues en rééchantillonnant les rendements quotidiens par blocs de ${MC.block} jours. Montre la plage de résultats plausibles si l'ordre des événements avait été différent.</div>`);
h+=panel('HOLDOUT','IN-SAMPLE vs OUT-OF-SAMPLE (70 / 30)',`<table class="t"><thead><tr><th class="l">SEGMENT</th><th class="l">PÉRIODE</th><th>TRADES</th><th>WIN RATE</th><th>RDT MOYEN/TRADE</th><th>CAGR</th><th>MAX DD</th><th>SHARPE</th></tr></thead><tbody>${[['IN-SAMPLE',H],['OUT-OF-SAMPLE',O]].map(([k,v])=>`<tr><td class="l">${k}</td><td class="l">${v.period}</td><td>${v.trades}</td><td>${f(v.wr,1,'%')}</td><td class="${cls(v.avg)}">${sg(v.avg,2)}</td><td class="${cls(v.cagr)}">${sg(v.cagr)}</td><td class="dn">${f(v.mdd,1,'%')}</td><td>${f(v.sharpe)}</td></tr>`).join('')}</tbody></table><div class="cv note">Les 30 % les plus récents de l'historique comparés aux 70 % plus anciens. Un écart fort signale une stratégie qui dépend d'une époque.</div>`);
h+=panel('SENSITIVITY','SENSIBILITÉ DES PARAMÈTRES',`<div class="scroll"><table class="t"><thead><tr><th class="l">VARIANTE</th><th>TRADES</th><th>WIN RATE</th><th>RDT MOYEN</th><th>CAGR</th><th>MAX DD</th><th>SHARPE</th><th>3 ANS +</th><th>FINAL</th></tr></thead><tbody>${D.sens.map(v=>`<tr><td class="l">${v.label}</td><td>${v.trades}</td><td>${f(v.wr,1,'%')}</td><td class="${cls(v.avg)}">${sg(v.avg,2)}</td><td class="${cls(v.cagr)}">${sg(v.cagr)}</td><td class="dn">${f(v.mdd,1,'%')}</td><td>${f(v.sharpe)}</td><td>${v.win3}</td><td>${money(v.final)}</td></tr>`).join('')}</tbody></table></div><div class="cv note">Si les résultats s'effondrent dès qu'on bouge un paramètre, la stratégie est probablement surajustée.</div>`);
h+=panel('PRICE','POSITION',`<div class="cv"><canvas id="c3" height="80"></canvas></div>`,`<select id="sel"></select><span style="color:var(--mut)" id="info"></span>`);
h+=panel('CALENDAR','MONTHLY RETURNS CALENDAR',`<div style="overflow:auto" class="scroll"><table class="cal" id="cal"></table></div>`,`<span class="r">courbe de capital, en %</span>`);
h+=panel('STATS','ADVANCED STATISTICS',`
<div class="g2"><div><div class="k">01 RETURNS PER TRADE ${D.ntrades>D.trades.length?'(100 derniers)':''}</div><div style="margin:8px 0">BEST <b class="up">${sg(M.best,2)}</b> · WORST <b class="dn">${sg(M.worst,2)}</b></div><canvas id="c4" height="90"></canvas></div>
<div><div class="k">02 RETURN DISTRIBUTION</div><div style="margin:8px 0">MEDIAN <b>${sg(M.med,2)}</b> · MEAN <b>${sg(M.mean,2)}</b></div><canvas id="c5" height="90"></canvas></div></div>
<div class="g2" style="border-top:1px solid var(--line)"><div><div class="k">03 RETURN CONCENTRATION</div><div style="margin:8px 0">SANS LES 5 MEILLEURS <b class="${cls(M.ex_best)}">${sg(M.ex_best)}</b><br>SANS LES 5 PIRES <b class="${cls(M.ex_worst)}">${sg(M.ex_worst)}</b></div><div class="note">Rendement total en retirant les trades extrêmes : mesure la dépendance à quelques coups.</div></div>
<div><div class="k">04 WINNING &amp; LOSING STREAKS</div><div style="margin:8px 0">LONGEST WIN RUN <b class="up">${M.sw} trades</b> · LONGEST LOSS RUN <b class="dn">${M.sl} trades</b></div><div class="note">${P.note}${P.kept.length?' Gagnant identique gardé en '+P.kept.join(', ')+'.':''}</div></div></div>
<div class="g2" style="border-top:1px solid var(--line)"><div><div class="k">05 MAE vs RÉSULTAT DU TRADE</div><canvas id="c12" height="110"></canvas></div><div><div class="k">06 DURÉE DES TRADES (jours de bourse)</div><canvas id="c13" height="110"></canvas></div></div>
<div style="padding:14px;border-top:1px solid var(--line)"><div class="k">07 SEASONALITY (rendement mensuel moyen, % de mois positifs)</div><canvas id="c6" height="50" style="margin-top:8px"></canvas></div>`,D.head.trades<30?`<span class="warn">THIN SAMPLE | ${D.head.trades} TRADES</span>`:'');
h+=panel('PROFILE','TRADE PROFILE',`<div class="row"><div><span>AVG MAE</span><b class="dn">${sg(M.avg_mae)}</b></div><div><span>AVG MFE</span><b class="up">${sg(M.avg_mfe)}</b></div><div><span>MEDIAN HOLD</span><b>${M.med_hold} bars</b></div><div style="grid-column:span 2;flex-wrap:wrap;justify-content:flex-end"><span>EXITS BY REASON</span>${Object.entries(D.reasons).map(([k,v])=>`<b>${k} ${v.n} · WR ${Math.round(v.w/v.n*100)}% · ${sg(v.pnl,0,' $')}</b>`).join(' | ')}</div></div>`);
h+=panel('JOURNAL','TRADES',`<div class="scroll"><table class="t click"><thead><tr><th class="l">#</th><th class="l">TITRE</th><th>${P.prevlabel}</th><th class="l">ENTRÉE</th><th>PRIX</th><th class="l">SORTIE</th><th>PRIX</th><th class="l">MOTIF</th><th>JOURS</th><th>NET %</th><th>P/L $</th><th>MAE</th><th>MFE</th></tr></thead><tbody id="tb"></tbody></table></div>`,
 `<span style="color:var(--mut)">${D.ntrades>D.trades.length?D.trades.length+' derniers trades sur '+D.ntrades+' · ':''}clique sur une ligne pour voir le prix</span>`);
$('#m').innerHTML=h;

Chart.defaults.color='#6f8593';Chart.defaults.borderColor='#1b2c38';Chart.defaults.font.family='ui-monospace,Menlo,Consolas,monospace';Chart.defaults.font.size=10;
const x={ticks:{maxTicksLimit:10,autoSkip:true}},nol={legend:{display:false}},UP='#1dc7a0',DN='#d6386a',BLU='#3b82f6';
const opt=(extra={})=>({animation:false,interaction:{mode:'index',intersect:false},elements:{point:{radius:0}},scales:{x},...extra});
const line=(id,labels,ds,extra)=>new Chart($(id),{type:'line',data:{labels,datasets:ds},options:opt(extra)});
const bars=(id,labels,ds,extra={})=>new Chart($(id),{type:'bar',data:{labels,datasets:ds},options:{animation:false,plugins:ds.length>1?{}:nol,...extra}});
const colr=a=>a.map(v=>v>=0?UP:DN);
let c1=line('#c1',D.eq.t,[{label:'Strategy equity',data:D.eq.s,borderColor:UP,borderWidth:1.5},{label:'Buy & Hold '+BL,data:D.eq.b,borderColor:BLU,borderWidth:1.2}],{scales:{x,y:{type:'logarithmic'}}});
$('#lg').onchange=e=>{c1.options.scales.y.type=e.target.checked?'logarithmic':'linear';c1.update()};
line('#c2',D.eq.t,[{label:'Drawdown %',data:D.eq.dd,borderColor:DN,backgroundColor:'rgba(214,56,106,.25)',fill:true,borderWidth:1}],{plugins:nol});
line('#c7',D.eq.t,[{label:'Stratégie 12 mois %',data:D.eq.r12,borderColor:UP,borderWidth:1},{label:BL+' 12 mois %',data:D.eq.b12,borderColor:BLU,borderWidth:1}]);
bars('#c8',D.years.y,[{label:'Stratégie %',data:D.years.s,backgroundColor:colr(D.years.s)},{label:BL+' %',data:D.years.b,backgroundColor:'rgba(59,130,246,.55)'}]);
bars('#c9',D.w3.y,[{label:'Stratégie 3 ans %',data:D.w3.s,backgroundColor:colr(D.w3.s)},{label:BL+' 3 ans %',data:D.w3.b,backgroundColor:'rgba(59,130,246,.55)'}]);
const mq=MC.q;line('#c10',MC.t,[{label:'95e',data:mq[4],borderColor:'rgba(29,199,160,.35)',borderWidth:1,fill:'+4',backgroundColor:'rgba(29,199,160,.08)'},{label:'75e',data:mq[3],borderColor:'rgba(29,199,160,.6)',borderWidth:1},{label:'médiane',data:mq[2],borderColor:UP,borderWidth:1.6},{label:'25e',data:mq[1],borderColor:'rgba(214,56,106,.6)',borderWidth:1},{label:'5e',data:mq[0],borderColor:'rgba(214,56,106,.4)',borderWidth:1}],{scales:{x,y:{type:'logarithmic'}}});
bars('#c11',MC.hist.map((_,i)=>MC.edges[i]+'→'+MC.edges[i+1]+'%'),[{data:MC.hist,backgroundColor:DN}]);
const T=D.trades;let c3;
function showTrade(i){const t=T[i];$('#sel').value=i;$('#info').textContent=`${t.tkr} · ${t.ein} → ${t.why=='ouverte'?'ouverte':t.eout} · ${t.why} · ${sg(t.net,2)}`;
 const pa=t.path,ds=[{label:t.tkr+' (prix ajusté)',data:pa.p,borderColor:'#d5e0e6',borderWidth:1.3,pointRadius:pa.t.map(d=>d==pa.ein||d==pa.eout?4:0),pointBackgroundColor:pa.t.map(d=>d==pa.ein?UP:DN)},{label:'Stop',data:pa.s,borderColor:DN,borderDash:[4,3],borderWidth:1}];
 if(pa.g.some(v=>v!=null))ds.push({label:'Cible',data:pa.g,borderColor:UP,borderDash:[4,3],borderWidth:1});
 if(c3)c3.destroy();c3=new Chart($('#c3'),{type:'line',data:{labels:pa.t,datasets:ds},options:{animation:false,elements:{point:{radius:0}},scales:{x}}})}
$('#sel').innerHTML=T.map((t,i)=>`<option value="${i}">#${t.n} ${t.tkr} ${t.ein}</option>`).join('');$('#sel').onchange=e=>showTrade(+e.target.value);showTrade(T.length-1);
$('#tb').innerHTML=T.map((t,i)=>`<tr onclick="showTrade(${i});document.querySelector('#c3').scrollIntoView({behavior:'smooth',block:'center'})"><td class="l">${t.n}</td><td class="l">${t.tkr}</td><td>${P.prevlabel.startsWith('RSI')?f(t.prev,1):sg(t.prev)}</td><td class="l">${t.ein}</td><td>${f(t.pin)}</td><td class="l">${t.why=='ouverte'?'(ouverte)':t.eout}</td><td>${f(t.pout)}</td><td class="l">${t.why}</td><td>${t.bars}</td><td class="${cls(t.net)}">${sg(t.net,2)}</td><td class="${cls(t.pnl)}">${sg(t.pnl,0,'')}</td><td class="dn">${sg(t.mae)}</td><td class="up">${sg(t.mfe)}</td></tr>`).join('');
const ys=Object.keys(D.cal).reverse();const col=v=>v==null?'':`background:rgba(${v>=0?'29,199,160':'214,56,106'},${Math.min(.8,Math.abs(v)/15+.1)})`;
$('#cal').innerHTML='<tr><th></th>'+['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','YTD'].map(m=>`<th>${m}</th>`).join('')+'</tr>'+ys.map(y=>`<tr><th>${y}</th>`+[...Array(12)].map((_,i)=>{const v=D.cal[y][i+1];return `<td style="${col(v)}">${v==null?'·':sg(v,1,'')}</td>`}).join('')+`<td class="${cls(D.cal[y].ytd)}"><b>${sg(D.cal[y].ytd)}</b></td></tr>`).join('');
bars('#c4',T.map(t=>'#'+t.n+' '+t.tkr),[{data:T.map(t=>t.net),backgroundColor:colr(T.map(t=>t.net))}]);
bars('#c5',D.hist.h.map((_,i)=>D.hist.e[i]+'→'+D.hist.e[i+1]),[{data:D.hist.h,backgroundColor:D.hist.h.map((_,i)=>D.hist.e[i+1]>0?UP:DN)}]);
new Chart($('#c12'),{type:'scatter',data:{datasets:[{data:D.scat.mae.map((m,i)=>({x:m,y:D.scat.net[i]})),pointRadius:2.5,backgroundColor:D.scat.net.map(v=>v>=0?'rgba(29,199,160,.7)':'rgba(214,56,106,.7)')}]},options:{animation:false,plugins:nol,scales:{x:{title:{display:true,text:'MAE % (pire moment du trade)'}},y:{title:{display:true,text:'Résultat net %'}}}}});
bars('#c13',D.dur.h.map((_,i)=>D.dur.e[i]+'–'+D.dur.e[i+1]),[{data:D.dur.h,backgroundColor:BLU}]);
bars('#c6',['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'].map((m,i)=>m+' '+(D.seas_p[i]??'–')+'%+'),[{data:D.seas,backgroundColor:colr(D.seas.map(v=>v??0))}]);
</script></body></html>"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["rotation", "meanrev", "dip", "sector", "all"], default="rotation")
    ap.add_argument("--start", type=int, default=None, help="année de départ (défaut : 2001 ; 1980 pour dip ; 1988 pour sector)")
    ap.add_argument("--capital", type=float, default=10000)
    a = ap.parse_args()
    cache = {}
    for name in (["rotation", "meanrev", "dip", "sector"] if a.strategy == "all" else [a.strategy]):
        print(f"{name} : calcul en cours…")
        data = build(name, a, cache)
        with open(FILES[name], "w", encoding="utf-8") as fh:
            fh.write(HTML.replace("__DATA__", json.dumps(data, default=str)))
        print(f"{FILES[name]} écrit —", data["head"], data["score"])
