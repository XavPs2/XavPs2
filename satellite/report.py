#!/usr/bin/env python3
"""Rapport HTML façon TextToQuant pour le satellite « gagnant de l'an passé ».

Usage: python report.py [--start 2001] [--capital 10000] [--stop 0.35]
Écrit report.html (graphiques Chart.js chargés depuis cdnjs, donc connexion requise).
"""
import argparse
import json
import math
import numpy as np
import pandas as pd
from rotation_satellite import TICKERS, FEE, KILL, load, run

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fnum(x, nd=2):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), nd)


def build(a):
    px = load(TICKERS, a.start)
    stocks = px.drop(columns="SPY")
    eq, jr, inmkt = run(px, a.start, a.capital, a.stop)
    spy = px["SPY"].reindex(eq.index).ffill()
    bh = spy / spy.iloc[0] * a.capital

    tr = jr[jr["entry_date"] != "(gardé)"].copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    last_day = eq.index[-1]
    tr["exit_dt"] = pd.to_datetime(tr["exit_date"]).fillna(last_day)
    tr["net"] = tr["cap_out"] / tr["cap_in"] - 1
    tr["pnl"] = tr["cap_out"] - tr["cap_in"]
    tr["bars"] = [int(((eq.index >= e) & (eq.index <= x)).sum()) for e, x in zip(tr["entry_date"], tr["exit_dt"])]
    mae, mfe, paths = [], [], []
    for _, t in tr.iterrows():
        p = stocks[t["winner"]].loc[t["entry_date"]:t["exit_dt"]].dropna()
        rel = p / p.iloc[0] - 1
        mae.append(rel.min()); mfe.append(rel.max())
        step = max(1, len(p) // 250)
        paths.append({"t": [d.strftime("%Y-%m-%d") for d in p.index[::step]],
                      "p": [round(v, 2) for v in p.values[::step]]})
    tr["mae"], tr["mfe"] = mae, mfe

    # --- métriques sur la courbe d'équité quotidienne
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() else np.nan
    dn = r[r < 0]
    sortino = r.mean() / math.sqrt((dn ** 2).sum() / len(r)) * math.sqrt(252) if len(dn) else np.nan
    omega = r[r > 0].sum() / -r[r < 0].sum() if len(dn) else np.nan
    rb = bh.pct_change().dropna().reindex(r.index).fillna(0)
    beta = np.cov(r, rb)[0, 1] / rb.var()
    alpha = (r.mean() - beta * rb.mean()) * 252

    wins, losses = tr[tr["net"] > 0], tr[tr["net"] <= 0]
    wr = len(wins) / len(tr)
    pf = wins["pnl"].sum() / -losses["pnl"].sum() if len(losses) and losses["pnl"].sum() else np.nan
    avg_w = wins["net"].mean() if len(wins) else 0
    avg_l = losses["net"].mean() if len(losses) else 0
    payoff = avg_w / -avg_l if avg_l else np.nan
    kelly = wr - (1 - wr) / payoff if payoff and not np.isnan(payoff) else np.nan
    exp = tr["net"].mean()
    sgn = (tr["net"] > 0).tolist()
    def longest(v):
        best = cur = 0
        for s in sgn:
            cur = cur + 1 if s == v else 0
            best = max(best, cur)
        return best
    # un cap en 4 piliers (heuristique maison, pas la formule de TextToQuant)
    prof = max(0, min(100, cagr / 0.30 * 100))
    risk = max(0, min(100, (1 + dd.min() / 0.8) * 100))
    cons = max(0, min(100, wr * 100 + (20 if pf and pf > 1.5 else 0)))
    edge = max(0, min(100, (sharpe / 1.5) * 100)) if not np.isnan(sharpe) else 0
    score = round((prof + risk + cons + edge) / 4)
    letter = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "F"

    # --- calendrier de rendements réalisés (à la clôture du trade)
    cal, cur_cap = {}, a.capital
    for _, t in tr.sort_values("exit_dt").iterrows():
        if pd.isna(pd.to_datetime(t["exit_date"])) :
            continue  # position ouverte : non réalisée
        d = t["exit_dt"]
        cal.setdefault(d.year, {}).setdefault(d.month, []).append(t["net"])
    calendar = {}
    for y, ms in cal.items():
        row = {}
        for m, v in ms.items():
            row[m] = float(np.prod([1 + x for x in v]) - 1)
        row["ytd"] = float(np.prod([1 + x for x in row.values()]) - 1)
        calendar[y] = row
    seas = {m: [] for m in range(1, 13)}
    for y, ms in cal.items():
        for m, v in ms.items():
            seas[m] += v

    def excl(n, best):
        s = tr.sort_values("net", ascending=not best)
        return float((1 + tr.drop(s.index[:n])["net"]).prod() - 1)

    reasons = {}
    for _, t in tr.iterrows():
        k = "Stop" if str(t["exit_reason"]).startswith("stop") else {"rotation": "Rotation", "ouverte": "Ouverte"}.get(t["exit_reason"], t["exit_reason"])
        g = reasons.setdefault(k, {"n": 0, "w": 0, "pnl": 0.0})
        g["n"] += 1; g["w"] += int(t["net"] > 0); g["pnl"] += float(t["pnl"])

    step = 5
    ei = eq.index[::step].union(eq.index[-1:])
    hist, edges = np.histogram(tr["net"] * 100, bins=10)
    trades = [{
        "n": i + 1, "tkr": t["winner"], "prev": fnum(t["prev_year_ret_pct"], 1),
        "ein": t["entry_date"].strftime("%Y-%m-%d"), "pin": t["entry_price"],
        "eout": t["exit_dt"].strftime("%Y-%m-%d"), "pout": t["exit_price"],
        "why": t["exit_reason"], "net": fnum(t["net"] * 100, 1), "pnl": fnum(t["pnl"], 0),
        "bars": t["bars"], "mae": fnum(t["mae"] * 100, 1), "mfe": fnum(t["mfe"] * 100, 1), "path": paths[i]}
        for i, (_, t) in enumerate(tr.iterrows())]
    kept = [int(y) for y in jr[jr["entry_date"] == "(gardé)"]["year"]]
    return dict(
        params=dict(start=a.start, capital=a.capital, stop=a.stop, fee=FEE, kill=KILL, tickers=TICKERS, kept=kept,
                    period=f"{eq.index[0].date()} – {eq.index[-1].date()}"),
        head=dict(ret=fnum((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1), win=fnum(wr * 100, 1), trades=len(tr)),
        score=dict(score=score, letter=letter, prof=round(prof), risk=round(risk), cons=round(cons), edge=round(edge)),
        m=dict(net=fnum(eq.iloc[-1] - a.capital, 0), cagr=fnum(cagr * 100, 1), wins=len(wins), n=len(tr),
               mdd=fnum(dd.min() * 100, 1), trough=str(dd.idxmin().date()), pf=fnum(pf), sharpe=fnum(sharpe),
               exp=fnum(exp * 100, 1), sortino=fnum(sortino), kelly=fnum(kelly * 100, 1), omega=fnum(omega),
               alpha=fnum(alpha * 100, 1), avg_w=fnum(avg_w * 100, 1), avg_l=fnum(avg_l * 100, 1), payoff=fnum(payoff),
               inmkt=fnum(inmkt.mean() * 100, 1), dur=fnum(tr["bars"].mean(), 1),
               tpm=fnum(len(tr) / (yrs * 12)), sw=longest(True), sl=longest(False),
               best=fnum(tr["net"].max() * 100, 1), worst=fnum(tr["net"].min() * 100, 1),
               med=fnum(tr["net"].median() * 100, 1), mean=fnum(exp * 100, 1),
               ex_best=fnum(excl(5, True) * 100, 1), ex_worst=fnum(excl(5, False) * 100, 1),
               avg_mae=fnum(tr["mae"].mean() * 100, 1), avg_mfe=fnum(tr["mfe"].mean() * 100, 1),
               med_hold=fnum(tr["bars"].median(), 0)),
        eq=dict(t=[d.strftime("%Y-%m-%d") for d in ei], s=[fnum(v, 0) for v in eq.loc[ei]],
                b=[fnum(v, 0) for v in bh.loc[ei]], dd=[fnum(v * 100, 1) for v in dd.loc[ei]]),
        cal={str(y): {str(k): fnum(v * 100, 1) for k, v in row.items()} for y, row in sorted(calendar.items())},
        seas=[fnum(np.mean(seas[m]) * 100, 1) if seas[m] else None for m in range(1, 13)],
        seas_n=[len(seas[m]) for m in range(1, 13)],
        hist=dict(h=[int(x) for x in hist], e=[round(float(x), 1) for x in edges]),
        reasons=reasons, trades=trades)


HTML = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Satellite Rotation – Rapport</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0a141c;--panel:#0d1a23;--line:#1b2c38;--tx:#d5e0e6;--mut:#6f8593;--up:#1dc7a0;--dn:#d6386a;--warn:#f0b429;--bl:#3b82f6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:center;gap:28px;padding:10px 20px;border-bottom:1px solid var(--line)}
header b{letter-spacing:.06em}.tabs span{color:var(--mut);margin-right:18px}.tabs .on{color:var(--tx);border-bottom:2px solid var(--up);padding-bottom:6px}
.wrap{display:grid;grid-template-columns:250px 1fr;min-height:calc(100vh - 44px)}
aside{border-right:1px solid var(--line);padding:14px;font-size:12px}
aside h4{margin:16px 0 6px;color:var(--mut);font-size:10px;letter-spacing:.12em;font-weight:400}
.q{border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:10px;color:var(--mut);line-height:1.6}
.chip{display:inline-block;border:1px solid var(--line);border-radius:12px;padding:1px 8px;margin:2px 2px 2px 0;font-size:11px;color:var(--mut)}
.rule{border:1px solid #17795f;background:#0c2a27;border-radius:6px;padding:7px 9px;margin:5px 0;color:var(--tx)}
.rule i{font-style:normal;float:right;color:var(--up);font-size:10px}.rule.x{border-color:#7b2a45;background:#2a1220}.rule.x i{color:var(--dn)}
main{padding:14px 20px;min-width:0}
.top{display:flex;align-items:center;gap:26px;flex-wrap:wrap;margin-bottom:8px}.top .big{font-size:15px;font-weight:700}
.k{color:var(--mut);font-size:10px;letter-spacing:.1em;margin-right:6px}
.bar{border:1px solid var(--line);background:#14180f;padding:5px 10px;font-size:11px;color:#c6b25a;margin-bottom:8px}
.panel{border:1px solid var(--line);background:var(--panel);border-radius:4px;margin:12px 0}
.ph{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}
.ph .tag{border:1px solid var(--line);font-size:10px;color:var(--mut);padding:1px 6px}.ph b{letter-spacing:.04em}
.warn{color:var(--warn);font-size:10px}
.grid{display:grid;grid-template-columns:230px repeat(4,1fr)}
.grade{padding:18px;border-right:1px solid var(--line);grid-row:span 2}
.grade .L{font-size:44px;font-weight:700;line-height:1}.grade small{color:var(--mut)}
.cell{padding:12px 14px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.cell .l{color:var(--mut);font-size:10px;letter-spacing:.1em}.cell .v{font-size:20px;margin:3px 0}.cell .s{color:var(--mut);font-size:10px}
.row{display:grid;grid-template-columns:repeat(5,1fr);border-top:1px solid var(--line)}
.row div{padding:9px 14px;display:flex;justify-content:space-between;border-right:1px solid var(--line)}
.row div span{color:var(--mut);font-size:10px;letter-spacing:.1em}
.up{color:var(--up)}.dn{color:var(--dn)}
.pil{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;padding:12px 14px}
.pil .t{display:flex;justify-content:space-between;color:var(--mut);font-size:10px;letter-spacing:.1em}
.pil .tr{height:3px;background:var(--line);margin-top:6px}.pil .tr i{display:block;height:3px;background:var(--up)}
.cv{padding:10px 14px}canvas{max-width:100%}
select{background:var(--bg);color:var(--tx);border:1px solid var(--line);padding:3px 6px;font:inherit}
.cal{width:100%;border-collapse:separate;border-spacing:4px;padding:8px}.cal th{color:var(--mut);font-weight:400;font-size:10px}
.cal td{text-align:center;padding:10px 0;border:1px solid var(--line)}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:0}.g2>div{padding:14px;border-right:1px solid var(--line)}
.big2{font-size:24px}.note{color:var(--mut);font-size:11px;border-left:2px solid var(--line);padding-left:10px;margin-top:10px}
table.t{width:100%;border-collapse:collapse;font-size:12px}.t th{color:var(--mut);font-weight:400;font-size:10px;text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
.t td{text-align:right;padding:6px 8px;border-bottom:1px solid #112029}.t td:first-child,.t th:first-child,.t td:nth-child(2),.t th:nth-child(2){text-align:left}
.t th:nth-child(8){text-align:left}.t tr{cursor:pointer}.t tr:hover{background:#12242f}
@media(max-width:900px){.wrap{grid-template-columns:1fr}aside{display:none}.grid,.row,.pil,.g2{grid-template-columns:1fr 1fr}.grade{grid-row:auto}}
</style></head><body>
<header><b>◆ SATELLITE</b><div class="tabs"><span class="on">Terminal</span><span>Journal</span></div></header>
<div class="wrap"><aside>
<h4>ACTIVE QUERY</h4><div class="q" id="qbox"></div>
<h4>ENTRY</h4><div class="rule">1er jour de bourse de janvier : 100 % sur le meilleur rendement total de l'année précédente<i>LONG</i></div>
<h4>EXIT</h4><div class="rule x" id="rx"></div><div class="rule x">Rotation en janvier (sauf si même gagnant)<i>ROT</i></div><div class="rule x" id="rk"></div>
<h4>RERUN</h4><div class="q">python report.py --start 2001 --capital 10000 --stop 0.35</div>
</aside><main id="m"></main></div>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s),f=(x,d=2,suf='')=>x==null?'–':Number(x).toFixed(d)+suf;
const cls=x=>x==null?'':x>=0?'up':'dn',sg=(x,d=1,s='%')=>x==null?'–':(x>=0?'+':'')+Number(x).toFixed(d)+s;
const P=D.params,M=D.m,S=D.score;
$('#qbox').innerHTML=`<span class="chip">${P.tickers.length} titres</span><span class="chip">1d</span><span class="chip">${P.period}</span><span class="chip">Long</span><span class="chip">Capital ${P.capital.toLocaleString()} $</span><span class="chip">Frais ${P.fee} $/trans.</span><span class="chip">Stop ${(P.stop*100).toFixed(0)} % du plus haut</span><span class="chip">Kill ${(P.kill*100).toFixed(0)} %</span><div style="margin-top:8px;font-size:11px">${P.tickers.join(' · ')}</div>`;
$('#rx').innerHTML=`Clôture ≤ ${(100-P.stop*100).toFixed(0)} % du plus haut depuis l'entrée → cash jusqu'en janvier<i>STOP</i>`;
$('#rk').innerHTML=`Valeur du satellite ≤ ${(P.kill*100).toFixed(0)} % du capital initial → arrêt<i>KILL</i>`;
const cell=(l,v,s,c='')=>`<div class="cell"><div class="l">${l}</div><div class="v ${c}">${v}</div><div class="s">${s}</div></div>`;
const rw=(l,v,c='')=>`<div><span>${l}</span><b class="${c}">${v}</b></div>`;
let h=`<div class="top"><span class="big">SATELLITE</span><span>1d</span><span style="color:var(--mut)">${P.period}</span>
<span><span class="k">RETURN</span><b class="${cls(D.head.ret)}">${sg(D.head.ret)}</b></span><span><span class="k">WIN</span><b>${f(D.head.win,2,'%')}</b></span><span><span class="k">TRADES</span><b>${D.head.trades}</b></span></div>
<div class="bar">● Hypothèses : entrée à la clôture du 1er jour de janvier · ${P.fee} $/transaction · fractions d'action permises · aucun impôt ni change · univers à biais de survivant</div>
<div class="panel"><div class="ph"><span class="tag">METRICS</span><b>PERFORMANCE</b>${D.head.trades<30?`<span class="warn">⚠ ${D.head.trades} TRADES</span>`:''}</div>
<div class="grid"><div class="grade"><div class="k">STRATEGY GRADE (maison)</div><div class="L ${S.score>=50?'up':'dn'}">${S.letter} <small>${S.score}/100</small></div><div class="k" style="margin-top:8px">${S.score>=65?'DEPLOY':'AVOID'}</div></div>
${cell('NET PROFIT',sg(M.net,0,' $'),sg(D.head.ret)+' | CAGR '+sg(M.cagr),cls(M.net))}
${cell('WIN RATE',f(D.head.win,2,'%'),M.wins+' / '+M.n)}
${cell('MAX DD',f(M.mdd,2,'%'),'TROUGH '+M.trough,'dn')}
${cell('PROFIT FAC.',f(M.pf),'≥ 1.0')}
${cell('SHARPE',f(M.sharpe),'sur courbe quotidienne')}
${cell('EXPECTANCY',sg(M.exp,1),'PAR TRADE | %',cls(M.exp))}
${cell('SORTINO',f(M.sortino),'DOWNSIDE ONLY')}
${cell('KELLY',sg(M.kelly,1),M.kelly<0?'NO EDGE':'FRACTION',cls(M.kelly))}</div>
<div class="row">${rw('OMEGA',f(M.omega))}${rw('ALPHA vs SPY',sg(M.alpha,1,'%/an'),cls(M.alpha))}${rw('AVG WIN',sg(M.avg_w),'up')}${rw('AVG LOSS',sg(M.avg_l),'dn')}${rw('PAYOFF',f(M.payoff)+' : 1')}</div>
<div class="row">${rw('IN MARKET',f(M.inmkt,1,'%'))}${rw('AVG DURATION',f(M.dur,0)+' bars')}${rw('TRADES/MO',f(M.tpm))}${rw('STREAK W',M.sw,'up')}${rw('STREAK L',M.sl,'dn')}</div>
<div class="pil">${[['PROFIT',S.prof],['RISK',S.risk],['CONSIST',S.cons],['EDGE',S.edge]].map(([k,v])=>`<div><div class="t"><span>${k}</span><b>${v}</b></div><div class="tr"><i style="width:${v}%"></i></div></div>`).join('')}</div></div>
<div class="panel"><div class="ph"><span class="tag">CURVE</span><b>EQUITY CURVE</b><span style="margin-left:auto;color:var(--mut)">START ${P.capital.toLocaleString()} · END ${D.eq.s.at(-1).toLocaleString()} · MAX DD ${f(M.mdd,1,'%')}</span><label style="color:var(--mut)"><input type="checkbox" id="lg" checked> log</label></div><div class="cv"><canvas id="c1" height="110"></canvas></div><div class="cv"><canvas id="c2" height="40"></canvas></div></div>
<div class="panel"><div class="ph"><span class="tag">PRICE</span><b>POSITION</b><select id="sel"></select><span style="color:var(--mut)" id="info"></span></div><div class="cv"><canvas id="c3" height="80"></canvas></div></div>
<div class="panel"><div class="ph"><span class="tag">CALENDAR</span><b>REALIZED RETURNS CALENDAR</b></div><div style="overflow:auto"><table class="cal" id="cal"></table></div></div>
<div class="panel"><div class="ph"><span class="tag">STATS</span><b>ADVANCED STATISTICS</b><span class="warn">${D.head.trades<30?'THIN SAMPLE | '+D.head.trades+' TRADES':''}</span></div>
<div class="g2"><div><div class="k">01 RETURNS OVER TIME</div><div class="k" style="margin-top:10px">NET RETURN</div><div class="big2 ${cls(D.head.ret)}">${sg(D.head.ret)}</div>BEST <b class="up">${sg(M.best)}</b> · WORST <b class="dn">${sg(M.worst)}</b><canvas id="c4" height="90" style="margin-top:10px"></canvas></div>
<div><div class="k">02 RETURN DISTRIBUTION</div><div style="margin:8px 0">MEDIAN <b>${sg(M.med)}</b> · MEAN <b>${sg(M.mean)}</b></div><canvas id="c5" height="90"></canvas></div></div>
<div class="g2" style="border-top:1px solid var(--line)"><div><div class="k">03 RETURN CONCENTRATION</div><div style="margin:8px 0">SANS LES 5 MEILLEURS <b class="${cls(M.ex_best)}">${sg(M.ex_best)}</b><br>SANS LES 5 PIRES <b class="${cls(M.ex_worst)}">${sg(M.ex_worst)}</b></div><div class="note">Rendement composé du satellite en retirant les trades extrêmes : mesure la dépendance à quelques coups.</div></div>
<div><div class="k">04 WINNING &amp; LOSING STREAKS</div><div style="margin:8px 0">LONGEST WIN RUN <b class="up">${M.sw} trades</b> · LONGEST LOSS RUN <b class="dn">${M.sl} trades</b></div><div class="note">Gagnant identique gardé en ${P.kept.length} janvier : ${P.kept.join(', ')||'–'}.</div></div></div>
<div style="padding:14px;border-top:1px solid var(--line)"><div class="k">06 SEASONALITY (mois de sortie)</div><canvas id="c6" height="50" style="margin-top:8px"></canvas></div></div>
<div class="panel"><div class="ph"><span class="tag">PROFILE</span><b>TRADE PROFILE</b></div><div class="row"><div><span>AVG MAE</span><b class="dn">${sg(M.avg_mae)}</b></div><div><span>AVG MFE</span><b class="up">${sg(M.avg_mfe)}</b></div><div><span>MEDIAN HOLD</span><b>${M.med_hold} bars</b></div><div style="grid-column:span 2;gap:8px;flex-wrap:wrap;justify-content:flex-end"><span>EXITS BY REASON</span>${Object.entries(D.reasons).map(([k,v])=>`<b>${k} ${v.n} · WR ${Math.round(v.w/v.n*100)}% · ${sg(v.pnl,0,' $')}</b>`).join(' | ')}</div></div></div>
<div class="panel"><div class="ph"><span class="tag">JOURNAL</span><b>TRADES</b><span style="color:var(--mut)">clique sur une ligne pour voir le prix</span></div><table class="t"><thead><tr><th>#</th><th>TITRE</th><th>RDT ANNÉE PRÉC.</th><th>ENTRÉE</th><th>PRIX</th><th>SORTIE</th><th>PRIX</th><th>MOTIF</th><th>NET %</th><th>P/L $</th><th>MAE</th><th>MFE</th></tr></thead><tbody id="tb"></tbody></table></div>`;
$('#m').innerHTML=h;
Chart.defaults.color='#6f8593';Chart.defaults.borderColor='#1b2c38';Chart.defaults.font.family='ui-monospace,Menlo,Consolas,monospace';Chart.defaults.font.size=10;
const x={ticks:{maxTicksLimit:10,autoSkip:true}};
const line=(id,ds,extra={})=>new Chart($(id),{type:'line',data:{labels:D.eq.t,datasets:ds},options:{animation:false,interaction:{mode:'index',intersect:false},elements:{point:{radius:0}},scales:{x},...extra}});
let c1=line('#c1',[{label:'Strategy equity',data:D.eq.s,borderColor:'#1dc7a0',borderWidth:1.5},{label:'Buy & Hold SPY',data:D.eq.b,borderColor:'#3b82f6',borderWidth:1.2}],{scales:{x,y:{type:'logarithmic'}}});
$('#lg').onchange=e=>{c1.options.scales.y.type=e.target.checked?'logarithmic':'linear';c1.update()};
line('#c2',[{label:'Drawdown %',data:D.eq.dd,borderColor:'#d6386a',backgroundColor:'rgba(214,56,106,.25)',fill:true,borderWidth:1}],{plugins:{legend:{display:false}}});
const T=D.trades;let c3;
function showTrade(i){const t=T[i];$('#sel').value=i;$('#info').textContent=`${t.tkr} · ${t.why} · ${sg(t.net)}`;
 const stopLine=t.path.p.map(()=>null);let pk=0;const peaks=t.path.p.map(p=>{pk=Math.max(pk,p);return +(pk*(1-P.stop)).toFixed(2)});
 if(c3)c3.destroy();c3=new Chart($('#c3'),{type:'line',data:{labels:t.path.t,datasets:[{label:t.tkr+' (prix ajusté)',data:t.path.p,borderColor:'#d5e0e6',borderWidth:1.3},{label:'Niveau de stop',data:peaks,borderColor:'#d6386a',borderDash:[4,3],borderWidth:1}]},options:{animation:false,elements:{point:{radius:0}},scales:{x}}})}
$('#sel').innerHTML=T.map((t,i)=>`<option value="${i}">#${t.n} ${t.tkr} ${t.ein}</option>`).join('');$('#sel').onchange=e=>showTrade(+e.target.value);showTrade(T.length-1);
$('#tb').innerHTML=T.map((t,i)=>`<tr onclick="showTrade(${i});scrollTo({top:$('#c3').offsetTop-300,behavior:'smooth'})"><td>${t.n}</td><td>${t.tkr}</td><td>${sg(t.prev)}</td><td>${t.ein}</td><td>${f(t.pin)}</td><td>${t.why=='ouverte'?'(ouverte)':t.eout}</td><td>${f(t.pout)}</td><td style="text-align:left">${t.why}</td><td class="${cls(t.net)}">${sg(t.net)}</td><td class="${cls(t.pnl)}">${sg(t.pnl,0,'')}</td><td class="dn">${sg(t.mae)}</td><td class="up">${sg(t.mfe)}</td></tr>`).join('');
const ys=Object.keys(D.cal);const col=v=>v==null?'':`background:rgba(${v>=0?'29,199,160':'214,56,106'},${Math.min(.8,Math.abs(v)/40+.12)})`;
$('#cal').innerHTML='<tr><th></th>'+['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','YTD'].map(m=>`<th>${m}</th>`).join('')+'</tr>'+ys.map(y=>`<tr><th>${y}</th>`+[...Array(12)].map((_,i)=>{const v=D.cal[y][i+1];return `<td style="${col(v)}">${v==null?'·':sg(v,1,'')}</td>`}).join('')+`<td class="${cls(D.cal[y].ytd)}">${sg(D.cal[y].ytd)}</td></tr>`).join('');
new Chart($('#c4'),{type:'bar',data:{labels:T.map(t=>'#'+t.n+' '+t.tkr),datasets:[{data:T.map(t=>t.net),backgroundColor:T.map(t=>t.net>=0?'#1dc7a0':'#d6386a')}]},options:{animation:false,plugins:{legend:{display:false}}}});
new Chart($('#c5'),{type:'bar',data:{labels:D.hist.h.map((_,i)=>D.hist.e[i]+'→'+D.hist.e[i+1]),datasets:[{data:D.hist.h,backgroundColor:D.hist.h.map((_,i)=>D.hist.e[i+1]>0?'#1dc7a0':'#d6386a')}]},options:{animation:false,plugins:{legend:{display:false}}}});
new Chart($('#c6'),{type:'bar',data:{labels:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'].map((m,i)=>m+' n='+D.seas_n[i]),datasets:[{data:D.seas,backgroundColor:D.seas.map(v=>v>=0?'#1dc7a0':'#d6386a')}]},options:{animation:false,plugins:{legend:{display:false}}}});
</script></body></html>"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2001)
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--stop", type=float, default=0.35)
    a = ap.parse_args()
    data = build(a)
    open("report.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(data, default=str)))
    print("report.html écrit —", data["head"], data["score"])
