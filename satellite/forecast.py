#!/usr/bin/env python3
"""Analyse quantitative prospective (horizon 36 mois) de la rotation sectorielle.

1. Significativité : Sharpe probabiliste (PSR) et dégonflé (DSR, Bailey & López de Prado 2014).
2. Décomposition : CAPM et Treynor-Mazuy, erreurs Newey-West ; rupture structurelle (Chow, sup-Wald d'Andrews).
3. Projection : bootstrap stationnaire (Politis & Romano 1994) sur deux régimes d'échantillon,
   chaîne de Markov cachée à 2 états (EM), modèle structurel bayésien (alpha rétréci, prime de risque prospective).
4. Ensemble pondéré, risque de queue, état courant du signal, scénarios de crise.

Usage: python forecast.py  ->  forecast.json + report_forecast.html
Dépendances : numpy, pandas, yfinance (aucune autre).
"""
import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from sector import CASH, SPX, load_sector, momentum, run_sector, SECTORS

ND = NormalDist()
H = 36                      # horizon en mois
HD = 756                    # horizon en jours de bourse
NSIM = 10000
SPLIT = "2015-01-01"        # coupure pré / post pour la décroissance d'alpha
N_TRIALS = 840              # nombre de variantes testées dans rotation_search.py
rng = np.random.default_rng(2026)


# ------------------------------------------------------------------ outils statistiques
def moments(x):
    x = np.asarray(x)
    m, s = x.mean(), x.std(ddof=1)
    z = (x - m) / s
    return m, s, (z ** 3).mean(), (z ** 4).mean()


def sharpe_se(x):
    """Erreur type du Sharpe par période, non normale (Mertens 2002 / Lo 2002)."""
    m, s, g3, g4 = moments(x)
    sr = m / s
    return sr, math.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (len(x) - 1))


def ols_nw(y, X, lags=6, w=None):
    """MCO (ou MCP si w) avec covariance HAC de Newey-West (noyau de Bartlett)."""
    y, X = np.asarray(y, float), np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    if w is not None:
        sw = np.sqrt(w)
        y, X = y * sw, X * sw[:, None]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    xe = X * e[:, None]
    S = xe.T @ xe
    for l in range(1, lags + 1):
        g = xe[l:].T @ xe[:-l]
        S += (1 - l / (lags + 1)) * (g + g.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, np.sqrt(np.diag(V)), e, r2


def max_dd(paths):
    """Drawdown max de chaque trajectoire de valeur (lignes)."""
    return (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)


def stationary_bootstrap_idx(n_obs, n_paths, length, mean_block):
    """Indices du bootstrap stationnaire : blocs de longueur géométrique de moyenne `mean_block`."""
    p = 1 / mean_block
    idx = np.empty((n_paths, length), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, n_paths)
    for t in range(1, length):
        jump = rng.random(n_paths) < p
        idx[:, t] = np.where(jump, rng.integers(0, n_obs, n_paths), (idx[:, t - 1] + 1) % n_obs)
    return idx


# ------------------------------------------------------------------ chaîne de Markov cachée
def hmm_fit(Y, k=2, iters=300, restarts=8):
    """HMM gaussien multivarié par EM (Baum-Welch normalisé). Y : (T, d)."""
    T, d = Y.shape
    best = None
    for r in range(restarts):
        lab = rng.integers(0, k, T) if r else (Y[:, 1] < np.quantile(Y[:, 1], 0.25)).astype(int)
        mu = np.array([Y[lab == j].mean(0) if (lab == j).sum() > d else Y.mean(0) for j in range(k)])
        cov = np.array([np.cov(Y.T) for _ in range(k)])
        A = np.full((k, k), 0.1 / (k - 1)); np.fill_diagonal(A, 0.9)
        pi = np.full(k, 1 / k)
        ll_old = -np.inf
        for _ in range(iters):
            B = np.column_stack([_mvn_pdf(Y, mu[j], cov[j]) for j in range(k)]) + 1e-300
            alpha, c = np.zeros((T, k)), np.zeros(T)
            alpha[0] = pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[t]; c[t] = alpha[t].sum(); alpha[t] /= c[t]
            beta = np.ones((T, k))
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gam = alpha * beta
            xi = np.zeros((k, k))
            for t in range(T - 1):
                xi += (alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]) / c[t + 1]
            pi = gam[0]
            A = xi / xi.sum(1, keepdims=True)
            for j in range(k):
                w = gam[:, j]
                mu[j] = (w[:, None] * Y).sum(0) / w.sum()
                D = Y - mu[j]
                cov[j] = (w[:, None] * D).T @ D / w.sum() + 1e-8 * np.eye(d)
            ll = np.log(c).sum()
            if abs(ll - ll_old) < 1e-8:
                break
            ll_old = ll
        if best is None or ll > best["ll"]:
            best = dict(ll=ll, mu=mu.copy(), cov=cov.copy(), A=A.copy(), filt=alpha[-1].copy(), gam=gam.copy())
    return best


def _mvn_pdf(Y, m, S):
    d = Y.shape[1]
    D = Y - m
    Si = np.linalg.inv(S)
    return np.exp(-0.5 * np.einsum("ij,jk,ik->i", D, Si, D)) / math.sqrt((2 * math.pi) ** d * np.linalg.det(S))


# ------------------------------------------------------------------ résumé d'une distribution simulée
def summarize(name, S, B, method):
    """S, B : trajectoires de valeur (n, H+1) normalisées à 1, stratégie et S&P 500, pas mensuel."""
    rs, rb = S[:, -1] - 1, B[:, -1] - 1
    dd = max_dd(S)
    q = lambda v, p: float(np.percentile(v, p))
    worst = np.sort(rs)[: max(1, len(rs) // 20)]
    return dict(name=name, method=method, n=len(rs),
                med=q(rs, 50), mean=float(rs.mean()), p5=q(rs, 5), p25=q(rs, 25), p75=q(rs, 75), p95=q(rs, 95),
                cagr_med=(1 + q(rs, 50)) ** (1 / 3) - 1, p_loss=float((rs < 0).mean()), cvar5=float(worst.mean()),
                p_beat=float((rs > rb).mean()), excess_med=q(rs - rb, 50), b_med=q(rb, 50),
                dd_med=q(dd, 50), dd_p5=q(dd, 5), p_dd30=float((dd < -0.30).mean()))


def build():
    px = load_sector(1984)
    eq, jr, inmkt = run_sector(px, 1988, 10000)
    bench, cidx = px[SPX].reindex(eq.index).ffill(), px[CASH].reindex(eq.index).ffill()
    me = lambda s: s.resample("ME").last()
    rs = me(eq).pct_change().iloc[1:]
    rb = me(bench).pct_change().reindex(rs.index)
    rf = me(cidx).pct_change().reindex(rs.index)
    xs, xb = rs - rf, rb - rf
    T = len(rs)
    out = dict(period=f"{rs.index[0]:%Y-%m} – {rs.index[-1]:%Y-%m}", months=T)

    # ---------------- 1. descriptives et significativité
    desc = {}
    for lab, r, x in (("strat", rs, xs), ("bench", rb, xb)):
        m, s, g3, g4 = moments(r)
        sr, se = sharpe_se(x)
        desc[lab] = dict(mean=m * 12, vol=s * math.sqrt(12), skew=g3, kurt=g4 - 3, sr=sr * math.sqrt(12),
                         sr_se=se * math.sqrt(12), ac1=float(pd.Series(r).autocorr(1)))
    sr_s, se_s = sharpe_se(xs)
    sr_b = sharpe_se(xb)[0]
    trials = pd.read_csv("rotation_search_results.csv")["sharpe"].dropna() / math.sqrt(12)  # en unité mensuelle
    gamma = 0.5772156649
    def sr0(n):
        return trials.std() * ((1 - gamma) * ND.inv_cdf(1 - 1 / n) + gamma * ND.inv_cdf(1 - 1 / (n * math.e)))
    sig = dict(psr0=ND.cdf(sr_s / se_s), psr_bench=ND.cdf((sr_s - sr_b) / se_s),
               sr0_840=sr0(N_TRIALS) * math.sqrt(12), dsr_840=ND.cdf((sr_s - sr0(N_TRIALS)) / se_s),
               sr0_30=sr0(30) * math.sqrt(12), dsr_30=ND.cdf((sr_s - sr0(30)) / se_s),
               trials_sd=trials.std() * math.sqrt(12),
               min_trl=int(math.ceil(1 + (1 - moments(xs)[2] * sr_s + (moments(xs)[3] - 1) / 4 * sr_s ** 2) * (ND.inv_cdf(0.95) / (sr_s - sr_b)) ** 2)) if sr_s > sr_b else None)
    out["desc"], out["sig"] = desc, sig

    # ---------------- 2. alpha, timing et rupture
    def reg(mask, tm=False):
        X = np.column_stack([xb[mask], xb[mask] ** 2]) if tm else xb[mask].values
        b, se, e, r2 = ols_nw(xs[mask], X)
        res = dict(alpha=b[0] * 12, alpha_t=b[0] / se[0], beta=b[1], beta_t=(b[1] - 1) / se[1], r2=r2,
                   ir=b[0] / e.std(ddof=1) * math.sqrt(12), te=e.std(ddof=1) * math.sqrt(12), n=int(mask.sum()))
        if tm:
            res.update(gamma=b[2], gamma_t=b[2] / se[2])
        return res
    allm = np.ones(T, bool)
    pre, post = (rs.index < SPLIT), (rs.index >= SPLIT)
    out["reg"] = dict(full=reg(allm), pre=reg(pre), post=reg(post), tm=reg(allm, tm=True))
    D = post.astype(float)
    b, se, _, _ = ols_nw(xs, np.column_stack([D, xb, D * xb]))
    out["chow"] = dict(dalpha=b[1] * 12, dalpha_t=b[1] / se[1], dbeta=b[3], dbeta_t=b[3] / se[3])
    waldv, dates = [], []
    for i in range(int(0.15 * T), int(0.85 * T)):
        Di = (np.arange(T) >= i).astype(float)
        bi, sei, _, _ = ols_nw(xs, np.column_stack([Di, xb]))
        waldv.append((bi[1] / sei[1]) ** 2); dates.append(rs.index[i])
    k = int(np.argmax(waldv))
    out["supwald"] = dict(stat=waldv[k], date=f"{dates[k]:%Y-%m}", cv10=7.12, cv5=8.85, cv1=12.35,
                          series=dict(t=[f"{d:%Y-%m}" for d in dates], w=[round(v, 2) for v in waldv]))
    roll = []
    for i in range(60, T + 1):
        bi, sei, ei, _ = ols_nw(xs.iloc[i - 60:i], xb.iloc[i - 60:i].values, lags=3)
        roll.append((f"{rs.index[i - 1]:%Y-%m}", bi[0] * 12, bi[1], bi[0] / ei.std(ddof=1) * math.sqrt(12)))
    out["rolling"] = dict(t=[r[0] for r in roll], alpha=[round(r[1] * 100, 2) for r in roll],
                          beta=[round(r[2], 3) for r in roll], ir=[round(r[3], 3) for r in roll])
    rel = np.log((1 + rs).cumprod() / (1 + rb).cumprod())
    out["rel"] = dict(t=[f"{d:%Y-%m}" for d in rel.index], v=[round(v, 4) for v in rel])

    # ---------------- 3. projections sur 36 mois
    models = {}
    paths_m = {}

    # M1 : fenêtres glissantes historiques (référence descriptive, non projective)
    cs, cb = (1 + rs).cumprod(), (1 + rb).cumprod()
    w_s = (cs.shift(-H) / cs - 1).dropna()
    w_b = (cb.shift(-H) / cb - 1).reindex(w_s.index)
    S1 = np.column_stack([np.ones(len(w_s)), 1 + w_s.values]); B1 = np.column_stack([np.ones(len(w_b)), 1 + w_b.values])
    m1 = summarize("Fenêtres historiques", S1, B1, "Toutes les fenêtres de 36 mois de 1988 à 2023 (chevauchantes)")
    dd_hist = [((eq.loc[d:d + pd.DateOffset(months=H)] / eq.loc[d:d + pd.DateOffset(months=H)].cummax()) - 1).min()
               for d in w_s.index]
    m1.update(dd_med=float(np.median(dd_hist)), dd_p5=float(np.percentile(dd_hist, 5)),
              p_dd30=float((np.array(dd_hist) < -0.30).mean()), n_eff=round(len(w_s) / H, 1))
    models["M1"] = m1

    # M2 : bootstrap stationnaire journalier conjoint (stratégie, S&P), blocs moyens de 63 jours
    rd = eq.pct_change().dropna()
    bd = bench.pct_change().reindex(rd.index).fillna(0)
    for key, start, lab in (("M2a", None, "Bootstrap stationnaire, 1988-2026"), ("M2b", SPLIT, "Bootstrap stationnaire, 2015-2026")):
        a, bb = (rd, bd) if start is None else (rd[rd.index >= start], bd[bd.index >= start])
        idx = stationary_bootstrap_idx(len(a), NSIM, HD, 63)
        Sd = np.cumprod(1 + a.values[idx], axis=1)
        Bd = np.cumprod(1 + bb.values[idx], axis=1)
        dd = max_dd(Sd)
        cols = np.arange(20, HD, 21)
        S = np.column_stack([np.ones(NSIM), Sd[:, cols]]); B = np.column_stack([np.ones(NSIM), Bd[:, cols]])
        s = summarize(lab, S, B, "Rééchantillonnage par blocs aléatoires (longueur géométrique, moyenne 3 mois) des rendements quotidiens conjoints")
        s.update(dd_med=float(np.median(dd)), dd_p5=float(np.percentile(dd, 5)), p_dd30=float((dd < -0.30).mean()))
        models[key], paths_m[key] = s, (S, B)

    # M3 : HMM bivarié à 2 états sur les rendements mensuels
    Y = np.column_stack([rs.values, rb.values])
    hm = hmm_fit(Y)
    calm = int(np.argmax(hm["mu"][:, 1] / np.sqrt(hm["cov"][:, 1, 1])))
    st = rng.choice(2, NSIM, p=hm["filt"] / hm["filt"].sum())
    L = [np.linalg.cholesky(hm["cov"][j]) for j in range(2)]
    R3 = np.zeros((NSIM, H, 2))
    for t in range(H):
        st = (rng.random(NSIM) < hm["A"][st, 1]).astype(int)   # transition depuis l'état du mois précédent
        z = rng.standard_normal((NSIM, 2))
        for j in range(2):
            m_ = st == j
            R3[m_, t] = hm["mu"][j] + z[m_] @ L[j].T
    S = np.column_stack([np.ones(NSIM), np.cumprod(1 + R3[:, :, 0], axis=1)])
    B = np.column_stack([np.ones(NSIM), np.cumprod(1 + R3[:, :, 1], axis=1)])
    models["M3"] = summarize("Chaîne de Markov cachée (2 états)", S, B, "Régimes latents estimés par EM, départ à la probabilité filtrée actuelle")
    paths_m["M3"] = (S, B)
    states = []
    for j in (calm, 1 - calm):
        sd = np.sqrt(np.diag(hm["cov"][j]))
        states.append(dict(label="calme" if j == calm else "stress", mu_s=hm["mu"][j][0] * 12, mu_b=hm["mu"][j][1] * 12,
                           vol_s=sd[0] * math.sqrt(12), vol_b=sd[1] * math.sqrt(12), corr=hm["cov"][j][0, 1] / sd[0] / sd[1],
                           stay=hm["A"][j, j], dur=1 / (1 - hm["A"][j, j]), p_now=float(hm["filt"][j] / hm["filt"].sum())))
    out["hmm"] = dict(states=states, ll=hm["ll"],
                      p_stress=dict(t=[f"{d:%Y-%m}" for d in rs.index], v=[round(float(v), 3) for v in hm["gam"][:, 1 - calm]]))

    # M4 : modèle structurel bayésien
    rf_now = float(px[CASH].pct_change(21).iloc[-1] * 12)            # rendement annualisé actuel des T-bills
    erp0, erp_sd = 0.03, 0.02                                        # prime de risque actions prospective : 3 % ± 2 %
    hl = 96                                                          # demi-vie de 8 ans pour l'alpha
    wts = 0.5 ** ((T - 1 - np.arange(T)) / hl)
    bw, sew, _, _ = ols_nw(xs, xb.values, w=wts)
    a_hat, a_se = bw[0] * 12, sew[0] * 12
    tau = 0.02                                                       # a priori sceptique : alpha ~ N(0, 2 %²)
    post_var = 1 / (1 / a_se ** 2 + 1 / tau ** 2)
    a_post, a_post_sd = post_var * a_hat / a_se ** 2, math.sqrt(post_var)
    bfull, _, efull, _ = ols_nw(xs, xb.values)
    beta = bfull[1]
    xb_c = xb.values - xb.values.mean()                              # chocs de marché centrés (forme empirique conservée)
    idx = stationary_bootstrap_idx(T, NSIM, H, 12)
    erp = rng.normal(erp0, erp_sd, NSIM)[:, None] / 12
    m_ex = erp + xb_c[idx]
    alpha_d = rng.normal(a_post, a_post_sd, NSIM)[:, None] / 12
    s_ex = alpha_d + beta * m_ex + efull[idx]                        # résidus appariés : conserve l'effet du filtre
    S = np.column_stack([np.ones(NSIM), np.cumprod(1 + rf_now / 12 + s_ex, axis=1)])
    B = np.column_stack([np.ones(NSIM), np.cumprod(1 + rf_now / 12 + m_ex, axis=1)])
    models["M4"] = summarize("Structurel bayésien", S, B, "α rétréci (a priori sceptique), β historique, prime de risque prospective, résidus appariés")
    paths_m["M4"] = (S, B)
    out["m4"] = dict(rf=rf_now, erp0=erp0, erp_sd=erp_sd, a_hat=a_hat, a_se=a_se, tau=tau, a_post=a_post,
                     a_post_sd=a_post_sd, beta=beta, hl_years=hl / 12, er_mkt=rf_now + erp0,
                     p_alpha_pos=ND.cdf(a_post / a_post_sd))
    sens = []
    for e_ in (0.0, 0.015, 0.03, 0.045, 0.06):
        for a_ in (0.0, a_post, a_hat):
            s_ = np.cumprod(1 + rf_now / 12 + a_ / 12 + beta * (e_ / 12 + xb_c[idx]) + efull[idx], axis=1)[:, -1] - 1
            b_ = np.cumprod(1 + rf_now / 12 + e_ / 12 + xb_c[idx], axis=1)[:, -1] - 1
            sens.append(dict(erp=e_, alpha=a_, med=float(np.median(s_)), p_beat=float((s_ > b_).mean()), p_loss=float((s_ < 0).mean())))
    out["sens"] = sens

    # Ensemble
    W = {"M2a": 0.20, "M2b": 0.25, "M3": 0.20, "M4": 0.35}
    Ss, Bs = [], []
    for k_, w_ in W.items():
        S_, B_ = paths_m[k_]
        pick = rng.choice(NSIM, int(NSIM * w_), replace=False)
        Ss.append(S_[pick]); Bs.append(B_[pick])
    SE, BE = np.vstack(Ss), np.vstack(Bs)
    ens = summarize("Ensemble pondéré", SE, BE, "M2a 20 % · M2b 25 % · M3 20 % · M4 35 %")
    ens["dd_med"] = float(sum(models[k_]["dd_med"] * w_ for k_, w_ in W.items()))
    ens["dd_p5"] = float(sum(models[k_]["dd_p5"] * w_ for k_, w_ in W.items()))
    ens["p_dd30"] = float(sum(models[k_]["p_dd30"] * w_ for k_, w_ in W.items()))
    models["ENS"] = ens
    out["models"], out["weights"] = models, W
    pct = [5, 25, 50, 75, 95]
    out["fan"] = dict(m=list(range(H + 1)), s=[[round(float(v) * 10000) for v in np.percentile(SE, p, axis=0)] for p in pct],
                      b=[[round(float(v) * 10000) for v in np.percentile(BE, p, axis=0)] for p in pct])
    hs, edges = np.histogram(np.clip(SE[:, -1] - 1, -0.6, 2.0) * 100, bins=40, range=(-60, 200))
    hb, _ = np.histogram(np.clip(BE[:, -1] - 1, -0.6, 2.0) * 100, bins=40, range=(-60, 200))
    out["hist"] = dict(e=[round(float(x)) for x in edges[:-1]], s=hs.tolist(), b=hb.tolist())
    ex = (SE[:, -1] - BE[:, -1]) * 100
    he, ee = np.histogram(np.clip(ex, -100, 150), bins=40, range=(-100, 150))
    out["hist_ex"] = dict(e=[round(float(x)) for x in ee[:-1]], h=he.tolist())

    # ---------------- 4. état courant
    last_me = rs.index[-1] if rs.index[-1] <= px.index[-1] else rs.index[-2]
    d_me = px.index[px.index <= last_me][-1]
    sma = px[SPX].rolling(200).mean()
    dist = px[SPX] / sma - 1
    mom = momentum(px[SECTORS], "12m")
    top = mom.loc[d_me].nlargest(5)
    hold = [dict(t=t, mom=float(v)) for t, v in top.items()]
    mes = px.index[pd.Series(px.index, px.index).groupby([px.index.year, px.index.month]).transform("max").values == px.index]
    dme = dist.reindex(mes).dropna()
    now = float(dist.iloc[-1])
    lo, hi = now - 0.03, now + 0.03
    hits = []
    for i, (d, v) in enumerate(dme.items()):
        if lo <= v <= hi and i + 12 < len(dme):
            hits.append(bool((dme.iloc[i + 1:i + 13] < 0).any()))
    r63, b63 = rd.iloc[-63:], bd.iloc[-63:]
    out["now"] = dict(date=f"{px.index[-1]:%Y-%m-%d}", signal_date=f"{d_me:%Y-%m-%d}", dist=now,
                      risk_on=bool(dist.loc[d_me] > 0), hold=hold, p_switch12=float(np.mean(hits)) if hits else None,
                      n_hist=len(hits), vol63=float(r63.std() * math.sqrt(252)), beta63=float(np.cov(r63, b63)[0, 1] / b63.var()),
                      bvol63=float(b63.std() * math.sqrt(252)),
                      dd_now=float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1))

    # ---------------- 5. épisodes de crise
    episodes = [("Krach d'octobre 1987*", "1987-08-25", "1987-12-04"), ("Récession 1990", "1990-07-16", "1990-10-11"),
                ("Bulle techno 2000-02", "2000-03-24", "2002-10-09"), ("Crise financière 2007-09", "2007-10-09", "2009-03-09"),
                ("Dette US 2011", "2011-04-29", "2011-10-03"), ("Resserrement 2018", "2018-09-20", "2018-12-24"),
                ("COVID 2020", "2020-02-19", "2020-03-23"), ("Inflation 2022", "2022-01-03", "2022-10-12")]
    ep = []
    for lab, a_, b_ in episodes:
        if pd.Timestamp(a_) < eq.index[0]:
            continue
        e_ = eq.loc[a_:b_]; bb_ = bench.loc[a_:b_]
        ep.append(dict(label=lab, start=a_, end=b_, s=float(e_.iloc[-1] / e_.iloc[0] - 1), b=float(bb_.iloc[-1] / bb_.iloc[0] - 1),
                       s_dd=float((e_ / e_.cummax() - 1).min())))
    out["episodes"] = ep
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == "__main__":
    res = clean(build())
    with open("forecast.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False)
    from forecast_html import HTML
    with open("report_forecast.html", "w", encoding="utf-8") as fh:
        fh.write(HTML.replace("__DATA__", json.dumps(res, ensure_ascii=False)))
    print("report_forecast.html écrit")
    print(json.dumps({k: res[k] for k in ("desc", "sig", "reg", "chow", "m4", "now")}, indent=1, ensure_ascii=False)[:6000])
    for k, m in res["models"].items():
        print(k, f"med {m['med']:+.1%} p5 {m['p5']:+.1%} p95 {m['p95']:+.1%} ploss {m['p_loss']:.1%} pbeat {m['p_beat']:.1%} dd {m['dd_med']:.1%} bmed {m['b_med']:+.1%}")
    print("supwald", res["supwald"]["stat"], res["supwald"]["date"])
    print(res["hmm"]["states"])
    print(res["episodes"])
