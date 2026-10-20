"""Gabarit HTML de la note de recherche prospective (rempli par forecast.py)."""
import re

HTML = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perspectives 36 mois – Note quantitative</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<style>
:root{--bg:#0a141c;--panel:#0d1a23;--line:#1b2c38;--tx:#d5e0e6;--mut:#6f8593;--up:#1dc7a0;--dn:#d6386a;--warn:#f0b429;--bl:#3b82f6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:center;gap:28px;padding:10px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header b{letter-spacing:.06em}.tabs a{color:var(--mut);margin-right:18px;text-decoration:none}.tabs a.on{color:var(--tx);border-bottom:2px solid var(--up);padding-bottom:6px}
main{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
h1{font:600 24px/1.3 Georgia,"Times New Roman",serif;margin:6px 0 4px;letter-spacing:.01em}
h2{font:600 18px/1.3 Georgia,"Times New Roman",serif;margin:0;color:var(--tx)}
h3{font-size:12px;letter-spacing:.12em;color:var(--mut);font-weight:400;margin:18px 0 8px}
.sub{color:var(--mut);font-size:12px}
p,li{color:#c3d0d7}p{margin:8px 0}
.panel{border:1px solid var(--line);background:var(--panel);border-radius:4px;margin:16px 0;padding:16px 18px}
.ph{display:flex;align-items:baseline;gap:12px;margin-bottom:10px;border-bottom:1px solid var(--line);padding-bottom:10px}
.num{border:1px solid var(--line);font-size:11px;color:var(--mut);padding:1px 7px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border:1px solid var(--line);margin:12px 0}
.kpi{padding:12px 14px;border-right:1px solid var(--line)}.kpi:last-child{border-right:0}
.kpi .l{color:var(--mut);font-size:10px;letter-spacing:.1em}.kpi .v{font-size:22px;margin:2px 0}.kpi .s{color:var(--mut);font-size:11px}
.up{color:var(--up)}.dn{color:var(--dn)}.wa{color:var(--warn)}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin:8px 0}
th{color:var(--mut);font-weight:400;font-size:10.5px;letter-spacing:.06em;text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
td{text-align:right;padding:6px 8px;border-bottom:1px solid #112029}th.l,td.l{text-align:left}
tr.hi td{background:#0c2a27}#sigt td:nth-child(3),#sigt th:nth-child(3){text-align:left}
.verdict{border-left:3px solid var(--warn);background:#14180f;padding:10px 14px;margin:12px 0;color:#e1d49a}
.note{color:var(--mut);font-size:12px;border-left:2px solid var(--line);padding-left:10px;margin-top:10px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.eq{overflow-x:auto;margin:8px 0;padding:6px 10px;background:#0a1720;border:1px solid var(--line)}
.katex{color:#e6edf1}
@media(max-width:900px){.kpis{grid-template-columns:1fr 1fr}.g2{grid-template-columns:1fr}}
</style></head><body>
<header><b>◆ SATELLITE</b><div class="tabs">
<a href="report.html">ROTATION</a><a href="report_meanrev.html">REPLI RSI · 25 TITRES</a><a href="report_dip.html">REPLI RSI(3) · S&amp;P 500</a><a href="report_sector.html">ROTATION SECTORIELLE</a><a href="report_forecast.html" class="on">PERSPECTIVES 36 MOIS</a></div></header>
<main id="m"></main>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s);
const P=(x,d=1)=>x==null?'–':(100*x).toFixed(d)+' %';
const SP=(x,d=1)=>x==null?'–':(x>=0?'+':'')+(100*x).toFixed(d)+' %';
const F=(x,d=2)=>x==null?'–':Number(x).toFixed(d);
const C=x=>x==null?'':x>=0?'up':'dn';
const M=D.models,E=M.ENS,R=D.reg,S=D.sig,Q=D.m4,N=D.now,SW=D.supwald,HM=D.hmm.states;
const ann=x=>Math.pow(1+x,1/3)-1;
const sigtxt=t=>Math.abs(t)>=2.58?'significatif à 1 %':Math.abs(t)>=1.96?'significatif à 5 %':Math.abs(t)>=1.645?'significatif à 10 %':'non significatif';
const swtxt=SW.stat>=SW.cv1?'au seuil de 1 %':SW.stat>=SW.cv5?'au seuil de 5 %':SW.stat>=SW.cv10?'au seuil de 10 % (mais pas de 5 %)':'non significative';
const names={FSELX:'Semi-conducteurs',FSPTX:'Technologie',FSENX:'Énergie',FBIOX:'Biotechnologie',FSPHX:'Santé',VGPMX:'Mines et métaux',VGENX:'Énergie (Vanguard)',FSRBX:'Banques',FSCSX:'Logiciels',FSAGX:'Or',FSUTX:'Services publics',FSDAX:'Défense',FIDSX:'Services financiers',FSLBX:'Courtage',FSPCX:'Assurance',FSHCX:'Services de santé',VGHCX:'Santé (Vanguard)',FSDPX:'Matériaux',FSCHX:'Chimie',FDFAX:'Consommation de base',FSRFX:'Transport',FSRPX:'Commerce de détail',FSHOX:'Construction',FBMPX:'Médias',FSTCX:'Télécoms',FSVLX:'Crédit à la consommation'};
const row=(cells,cls='')=>`<tr class="${cls}">${cells.map((c,i)=>`<td class="${i==0?'l':''}">${c}</td>`).join('')}</tr>`;
const panel=(n,t,body)=>`<div class="panel"><div class="ph"><span class="num">${n}</span><h2>${t}</h2></div>${body}</div>`;
let h=`<div class="sub">NOTE DE RECHERCHE QUANTITATIVE · ${N.date} · HORIZON 36 MOIS</div>
<h1>Rotation sectorielle par momentum avec filtre de tendance : propriétés statistiques et distribution prospective des rendements</h1>
<div class="sub">Échantillon mensuel ${D.period} (T = ${D.months}) · ${Object.keys(M).length-1} modèles de projection · ${(E.n).toLocaleString('fr-CA')} trajectoires simulées dans l'ensemble</div>`;

h+=panel('0','Résumé exécutif',`
<div class="kpis">
<div class="kpi"><div class="l">RENDEMENT MÉDIAN 36 MOIS</div><div class="v ${C(E.med)}">${SP(E.med)}</div><div class="s">${SP(ann(E.med))} / an · moyenne ${SP(E.mean)}</div></div>
<div class="kpi"><div class="l">INTERVALLE 90 %</div><div class="v">${SP(E.p5,0)} ; ${SP(E.p95,0)}</div><div class="s">5e et 95e centiles</div></div>
<div class="kpi"><div class="l">P(BATTRE LE S&amp;P 500)</div><div class="v">${P(E.p_beat,0)}</div><div class="s">écart médian ${SP(E.excess_med)}</div></div>
<div class="kpi"><div class="l">P(PERTE SUR 36 MOIS)</div><div class="v dn">${P(E.p_loss,0)}</div><div class="s">CVaR 5 % : ${SP(E.cvar5)}</div></div></div>
<p><b>Constat principal.</b> L'avantage historique est statistiquement robuste sur l'ensemble de l'échantillon : alpha CAPM de ${SP(R.full.alpha)} par an (t de Newey-West = ${F(R.full.alpha_t)}), Sharpe dégonflé de ${F(S.dsr_840,3)} après correction pour ${(840).toLocaleString('fr-CA')} essais. Mais il n'est <b>pas stationnaire</b> : l'alpha passe de ${SP(R.pre.alpha)} avant 2015 à ${SP(R.post.alpha)} après (t = ${F(R.post.alpha_t)}), un écart ${sigtxt(D.chow.dalpha_t)} (t = ${F(D.chow.dalpha_t)}), et le test de rupture d'Andrews situe la cassure en ${SW.date}, ${swtxt}.</p>
<p><b>Projection.</b> L'ensemble pondéré donne un rendement médian de ${SP(E.med)} sur 36 mois (${SP(ann(E.med))} par an) contre ${SP(E.b_med)} pour le S&amp;P 500 sous les mêmes hypothèses, avec une probabilité de surperformance de ${P(E.p_beat,0)}. Les modèles calibrés sur tout l'historique (M2a, M3) projettent une forte surperformance ; ceux qui intègrent la rupture post-2015 et une prime de risque prospective réaliste (M2b, M4) projettent une sous-performance. <b>La dispersion entre modèles est elle-même le résultat le plus informatif.</b></p>
<div class="verdict">Verdict : l'espérance de surperformance sur 3 ans est proche de zéro une fois la rupture structurelle prise en compte ; l'intérêt résiduel de la stratégie tient surtout à sa protection en marché baissier (bêta moyen ${F(R.full.beta)}). Pondération recommandée : satellite ≤ 10 %, avec des règles de révision fixées à l'avance (section 10).</div>`);

h+=panel('1','Données et méthode',`
<p><b>Stratégie.</b> À chaque fin de mois \(t\), si \(P^{SPX}_t &lt; \mathrm{SMA}_{200}(P^{SPX})_t\), le portefeuille est investi à 100 % en Trésor long terme (VUSTX). Sinon, il détient à parts égales les \(k=5\) secteurs (sur 26 fonds sectoriels américains) au plus fort rendement total sur 12 mois :</p>
<div class="eq">$$w_{i,t+1}=\frac{1}{k}\,\mathbb{1}\!\left\{\operatorname{rang}\!\left(\tfrac{P_{i,t}}{P_{i,t-252}}-1\right)\le k\right\}\cdot \mathbb{1}\!\left\{P^{SPX}_t\ge \mathrm{SMA}_{200,t}\right\}$$</div>
<p>Exécution à la clôture suivant le signal, coût de 10 pb par unité échangée, dividendes réinvestis. Référence : fonds indiciel S&amp;P 500 VFINX, rendement total. Taux sans risque : bons du Trésor à 3 mois (^IRX). Rendements excédentaires \(x_t=r_t-r^f_t\).</p>
<table><thead><tr><th class="l">STATISTIQUE (annualisée)</th><th>STRATÉGIE</th><th>S&amp;P 500 TR</th></tr></thead><tbody>
${row(['Rendement arithmétique moyen',P(D.desc.strat.mean),P(D.desc.bench.mean)])}
${row(['Volatilité',P(D.desc.strat.vol),P(D.desc.bench.vol)])}
${row(['Asymétrie (mensuelle)',F(D.desc.strat.skew),F(D.desc.bench.skew)])}
${row(['Kurtosis excédentaire (mensuelle)',F(D.desc.strat.kurt),F(D.desc.bench.kurt)])}
${row(['Autocorrélation d\'ordre 1',F(D.desc.strat.ac1),F(D.desc.bench.ac1)])}
${row(['Ratio de Sharpe ± erreur type',F(D.desc.strat.sr)+' ± '+F(D.desc.strat.sr_se),F(D.desc.bench.sr)+' ± '+F(D.desc.bench.sr_se)])}
</tbody></table>
<p class="note">L'asymétrie positive de la stratégie (contre une asymétrie négative de l'indice) est la signature d'un filtre qui coupe la queue gauche. La kurtosis plus élevée signale néanmoins des mois extrêmes plus fréquents des deux côtés : la concentration sur 5 secteurs se paie en risque spécifique. Les Sharpe sont calculés ici sur rendements <b>mensuels</b> excédentaires ; le rapport principal, calculé sur rendements quotidiens, affiche des valeurs légèrement différentes.</p>`);

h+=panel('2','Significativité statistique et surajustement',`
<p>L'erreur type du Sharpe tient compte de la non-normalité (Mertens, 2002) ; le Sharpe probabiliste (PSR) et le Sharpe dégonflé (DSR, Bailey et López de Prado, 2014) corrigent pour la sélection parmi \(N\) variantes :</p>
<div class="eq">$$\widehat{\sigma}(\widehat{SR})=\sqrt{\frac{1-\hat\gamma_3\widehat{SR}+\frac{\hat\gamma_4-1}{4}\widehat{SR}^2}{T-1}},\qquad \mathrm{PSR}(SR^*)=\Phi\!\left(\frac{\widehat{SR}-SR^*}{\widehat{\sigma}(\widehat{SR})}\right)$$</div>
<div class="eq">$$SR_0=\sqrt{\mathbb{V}[\widehat{SR}_n]}\left((1-\gamma_E)\,\Phi^{-1}\!\left(1-\tfrac1N\right)+\gamma_E\,\Phi^{-1}\!\left(1-\tfrac1{Ne}\right)\right),\qquad \mathrm{DSR}=\mathrm{PSR}(SR_0)$$</div>
<table id="sigt"><thead><tr><th class="l">TEST</th><th>VALEUR</th><th class="l">LECTURE</th></tr></thead><tbody>
${row(['PSR(0) : P(Sharpe réel &gt; 0)',F(S.psr0,4),'Quasi-certitude d\'un Sharpe positif'])}
${row(['PSR(SR du S&amp;P) : P(Sharpe réel &gt; Sharpe de l\'indice)',F(S.psr_bench,3),'Surperformance ajustée du risque probable sur l\'échantillon'])}
${row(['Seuil SR₀ attendu du hasard, N = 840 essais',F(S.sr0_840),'Écart type des Sharpe entre essais : '+F(S.trials_sd)])}
${row(['DSR, N = 840',F(S.dsr_840,4),'Robuste au surajustement par multiplicité'])}
${row(['DSR, N = 30 (essais effectivement indépendants)',F(S.dsr_30,4),''])}
${row(['Historique minimal pour SR &gt; SR(S&amp;P) à 95 % (MinTRL)',S.min_trl?S.min_trl+' mois':'–','Échantillon disponible : '+D.months+' mois'])}
</tbody></table>
<p class="note">Lecture : le risque principal n'est pas la multiplicité des essais (DSR ≈ 1) mais la <b>non-stationnarité</b>. Même sous l'hypothèse de stationnarité, il faut environ ${S.min_trl?Math.round(S.min_trl/12):'–'} ans de données pour conclure à 95 % que la stratégie bat l'indice en Sharpe : un historique de 3 ans ne permettra jamais de confirmer ou d'infirmer l'avantage.</p>`);

h+=panel('3','Décomposition du rendement : alpha, bêta et market timing',`
<div class="eq">$$x^{S}_t=\alpha+\beta\,x^{M}_t+\varepsilon_t\qquad\text{(CAPM)}\qquad\qquad x^{S}_t=\alpha+\beta\,x^{M}_t+\gamma\,(x^{M}_t)^2+\varepsilon_t\qquad\text{(Treynor-Mazuy)}$$</div>
<p>Erreurs types HAC de Newey-West (noyau de Bartlett, 6 retards). Le t du bêta teste \(H_0:\beta=1\).</p>
<table><thead><tr><th class="l">ÉCHANTILLON</th><th>N</th><th>α ANNUEL</th><th>t(α)</th><th>β</th><th>t(β=1)</th><th>R²</th><th>RATIO D'INFORMATION</th><th>TRACKING ERROR</th></tr></thead><tbody>
${[['1988–2026 (complet)',R.full],['1988–2014',R.pre],['2015–2026',R.post]].map(([l,r],i)=>row([l,r.n,`<span class="${C(r.alpha)}">${SP(r.alpha)}</span>`,F(r.alpha_t),F(r.beta),F(r.beta_t),F(r.r2),F(r.ir),P(r.te)],i==2?'hi':'')).join('')}
</tbody></table>
<p><b>Treynor-Mazuy :</b> \(\hat\gamma\) = ${F(R.tm.gamma)} (t = ${F(R.tm.gamma_t)}), ${sigtxt(R.tm.gamma_t)}. À fréquence mensuelle, le filtre SMA200 ne produit pas de convexité mesurable : il agit avec retard, après le début des baisses, ce qui limite l'ampleur des krachs prolongés (2000-02, 2008) sans capturer les chocs rapides (1990, 2018). La bonne tenue de 2020 vient de la composition sectorielle du moment (technologie, santé), pas du filtre.</p>
<p><b>Interprétation.</b> Le bêta d'environ ${F(R.full.beta)} est stable dans le temps ; ce qui disparaît après 2015, c'est l'alpha. La stratégie est passée d'un rendement « bêta 0,5 + alpha 14 % » à un rendement « bêta 0,5 + alpha ≈ 0 ». Avec un bêta de 0,5, un alpha nul implique une sous-performance mécanique en marché haussier, exactement ce qu'on observe depuis 2017.</p>`);

h+=panel('4','Stabilité temporelle : rupture structurelle et décroissance de l\'alpha',`
<p><b>Test de Chow</b> (coupure imposée en janvier 2015) : \(\Delta\alpha\) = ${SP(D.chow.dalpha)} par an, t = ${F(D.chow.dalpha_t)} (${sigtxt(D.chow.dalpha_t)}) ; \(\Delta\beta\) = ${F(D.chow.dbeta)}, t = ${F(D.chow.dbeta_t)} (${sigtxt(D.chow.dbeta_t)}).</p>
<p><b>Test sup-Wald d'Andrews (1993)</b> sur une rupture d'intercept à date inconnue (élagage de 15 %) : statistique maximale ${F(SW.stat)} en ${SW.date}, contre des valeurs critiques de ${SW.cv10} (10 %), ${SW.cv5} (5 %) et ${SW.cv1} (1 %). La rupture est détectée ${swtxt}. Le fait que la date estimée tombe d'elle-même près de la coupure de 2015, sans avoir été imposée, renforce le diagnostic.</p>
<canvas id="cw" height="70"></canvas>
<div class="g2" style="margin-top:14px"><div><h3>ALPHA ROULANT SUR 60 MOIS (% / AN)</h3><canvas id="ca" height="130"></canvas></div><div><h3>RENDEMENT RELATIF CUMULÉ, log(STRATÉGIE / S&amp;P)</h3><canvas id="cr" height="130"></canvas></div></div>
<p class="note">Hypothèses plausibles pour la décroissance : (i) concentration de la capitalisation dans quelques méga-capitalisations, mal captée par une pondération égale de secteurs ; (ii) diffusion des stratégies de momentum sectoriel via les FNB, qui comprime la prime ; (iii) régime de taux nuls jusqu'en 2021 puis choc de 2022, où le refuge obligataire a échoué.</p>`);

h+=panel('5','Régimes latents : chaîne de Markov cachée à deux états',`
<div class="eq">$$s_t\in\{1,2\},\quad \Pr(s_{t+1}=j\mid s_t=i)=A_{ij},\qquad \begin{pmatrix}r^S_t\\ r^M_t\end{pmatrix}\Big|\,s_t=j\ \sim\ \mathcal N(\mu_j,\Sigma_j)$$</div>
<p>Estimation par maximum de vraisemblance (algorithme EM de Baum-Welch, 8 initialisations, log-vraisemblance = ${F(D.hmm.ll,1)}).</p>
<table><thead><tr><th class="l">ÉTAT</th><th>μ STRATÉGIE</th><th>μ S&amp;P</th><th>σ STRATÉGIE</th><th>σ S&amp;P</th><th>CORRÉLATION</th><th>PERSISTANCE A<sub>jj</sub></th><th>DURÉE MOYENNE</th><th>PROBABILITÉ ACTUELLE</th></tr></thead><tbody>
${HM.map(s=>row([s.label,SP(s.mu_s),SP(s.mu_b),P(s.vol_s),P(s.vol_b),F(s.corr),F(s.stay,3),F(s.dur,1)+' mois',P(s.p_now,0)])).join('')}
</tbody></table>
<canvas id="ch" height="60"></canvas>
<p class="note">L'état de stress (S&amp;P à ${SP(HM[1].mu_b)} par an, volatilité de ${P(HM[1].vol_b,0)}) est celui où la stratégie se distingue le plus : rendement de ${SP(HM[1].mu_s)} et corrélation de seulement ${F(HM[1].corr)}. En régime calme, elle se comporte comme un indice à bêta élevé (corrélation ${F(HM[0].corr)}) et l'écart de rendement est faible. La probabilité filtrée actuelle d'être en régime calme est de ${P(HM[0].p_now,0)}.</p>`);

const ord=['M1','M2a','M2b','M3','M4','ENS'];
h+=panel('6','Modèles de projection sur 36 mois',`
<p><b>M1, fenêtres historiques :</b> toutes les fenêtres de 36 mois de l'échantillon. Purement descriptif ; les fenêtres se chevauchent (taille effective ≈ ${M.M1.n_eff} observations indépendantes).</p>
<p><b>M2, bootstrap stationnaire</b> (Politis et Romano, 1994) des rendements quotidiens conjoints (stratégie, S&amp;P), blocs de longueur géométrique de moyenne \(\bar L\) = 63 jours. Cela préserve l'autocorrélation, la dépendance croisée et le regroupement de volatilité : \(L_b\sim\mathrm{Géom}(1/\bar L)\). M2a utilise 1988-2026, M2b seulement 2015-2026 (régime post-rupture).</p>
<p><b>M3, chaîne de Markov cachée</b> (section 5) simulée à partir de la probabilité filtrée actuelle.</p>
<p><b>M4, modèle structurel bayésien.</b> Prime de risque actions prospective \(\pi\sim\mathcal N(${(Q.erp0*100).toFixed(1)}\,\%,\ ${(Q.erp_sd*100).toFixed(1)}\,\%^2)\) au-dessus du taux sans risque actuel (\(r^f\) = ${P(Q.rf,2)}), soit un rendement attendu du marché de ${P(Q.er_mkt)}, inférieur à la moyenne historique pour tenir compte des valorisations élevées. Chocs de marché tirés par bootstrap en blocs de 12 mois puis recentrés. L'alpha est estimé par moindres carrés pondérés exponentiellement (demi-vie de ${Q.hl_years} ans) puis rétréci vers zéro par un a priori sceptique :</p>
<div class="eq">$$\alpha\sim\mathcal N(0,\tau^2),\ \tau=${(Q.tau*100).toFixed(0)}\,\%\quad\Rightarrow\quad \alpha\mid\text{données}\sim\mathcal N\!\left(\frac{\hat\alpha/\hat\sigma_\alpha^2}{1/\hat\sigma_\alpha^2+1/\tau^2},\ \frac{1}{1/\hat\sigma_\alpha^2+1/\tau^2}\right)$$</div>
<div class="eq">$$x^S_{t}=\alpha+\hat\beta\,(\pi/12+\tilde x^M_{t})+\tilde\varepsilon_{t},\qquad (\tilde x^M_t,\tilde\varepsilon_t)\ \text{tirés conjointement}$$</div>
<p>\(\hat\alpha\) = ${SP(Q.a_hat)} ± ${P(Q.a_se)} ⇒ a posteriori ${SP(Q.a_post)} ± ${P(Q.a_post_sd)} ; P(α &gt; 0 | données) = ${P(Q.p_alpha_pos,0)} ; \(\hat\beta\) = ${F(Q.beta)}. Les résidus sont tirés aux mêmes dates que les chocs de marché, ce qui conserve la non-linéarité introduite par le filtre.</p>
<table><thead><tr><th class="l">MODÈLE</th><th>MÉDIANE</th><th>MOYENNE</th><th>P5</th><th>P95</th><th>MÉDIANE / AN</th><th>P(PERTE)</th><th>CVaR 5 %</th><th>P(&gt; S&amp;P)</th><th>ÉCART MÉDIAN</th><th>S&amp;P MÉDIAN</th><th>DD MÉDIAN</th><th>P(DD &lt; −30 %)</th></tr></thead><tbody>
${ord.map(k=>{const m=M[k];return row([`<b>${k}</b> ${m.name}`,`<span class="${C(m.med)}">${SP(m.med)}</span>`,SP(m.mean),SP(m.p5),SP(m.p95),SP(m.cagr_med),P(m.p_loss),SP(m.cvar5),P(m.p_beat,0),`<span class="${C(m.excess_med)}">${SP(m.excess_med)}</span>`,SP(m.b_med),P(m.dd_med),P(m.p_dd30)],k=='ENS'?'hi':'')}).join('')}
</tbody></table>
<p class="note">Pondération de l'ensemble : ${Object.entries(D.weights).map(([k,v])=>k+' '+(v*100).toFixed(0)+' %').join(' · ')}. Le poids le plus élevé va à M4, le seul modèle qui intègre à la fois l'incertitude sur l'alpha et une prime de risque prospective ; M2b reçoit plus de poids que M2a pour refléter la rupture. Les drawdowns de M3 et M4 sont mesurés sur une grille mensuelle et sous-estiment le drawdown quotidien d'environ 15 à 25 % ; ceux de M2 sont quotidiens. Le drawdown de l'ensemble est la moyenne pondérée des modèles.</p>`);

h+=panel('7','Distribution prospective (ensemble)',`
<h3>ÉVENTAIL DE VALEUR D'UN PLACEMENT DE 10 000 $ (5e, 25e, 50e, 75e, 95e CENTILES)</h3><canvas id="cf" height="105"></canvas>
<div class="g2" style="margin-top:14px"><div><h3>DISTRIBUTION DU RENDEMENT TOTAL À 36 MOIS (%)</h3><canvas id="chs" height="150"></canvas></div><div><h3>DISTRIBUTION DE L'ÉCART STRATÉGIE − S&amp;P À 36 MOIS (points de %)</h3><canvas id="che" height="150"></canvas></div></div>
<p class="note">Valeur médiane de 10 000 $ après 36 mois : ${Math.round(D.fan.s[2].at(-1)).toLocaleString('fr-CA')} $ pour la stratégie contre ${Math.round(D.fan.b[2].at(-1)).toLocaleString('fr-CA')} $ pour le S&amp;P 500. La distribution de l'écart est presque centrée sur zéro et très large : c'est une loterie symétrique à l'horizon de 3 ans, pas un avantage exploitable avec confiance.</p>`);

const sens=D.sens,erps=[...new Set(sens.map(s=>s.erp))],al=[...new Set(sens.map(s=>s.alpha))];
const alab=a=>a===0?'α = 0':Math.abs(a-Q.a_post)<1e-9?'α a posteriori ('+SP(a)+')':'α échantillon pondéré ('+SP(a)+')';
h+=panel('8','Sensibilité du modèle structurel',`
<p>Rendement médian à 36 mois, probabilité de battre l'indice et probabilité de perte, selon la prime de risque actions et l'hypothèse d'alpha (bêta et résidus inchangés).</p>
<table><thead><tr><th class="l">PRIME DE RISQUE π</th>${al.map(a=>`<th>${alab(a)}</th>`).join('')}</tr></thead><tbody>
${erps.map(e=>row([P(e)+' (marché '+P(Q.rf+e)+')',...al.map(a=>{const s=sens.find(z=>z.erp===e&&z.alpha===a);return `<span class="${C(s.med)}">${SP(s.med,0)}</span> · ${P(s.p_beat,0)} &gt; S&amp;P · ${P(s.p_loss,0)} perte`})])).join('')}
</tbody></table>
<p class="note">Lecture : avec un bêta de ${F(Q.beta)}, la stratégie bat l'indice surtout quand la prime de risque est faible, c'est-à-dire en marché médiocre. Plus le marché est fort, plus il faut d'alpha pour compenser le bêta réduit : le point mort se situe vers α ≈ (1 − β)·π.</p>`);

h+=panel('9','État actuel du signal et exposition courante',`
<div class="kpis">
<div class="kpi"><div class="l">FILTRE SMA200 (${N.signal_date})</div><div class="v ${N.risk_on?'up':'dn'}">${N.risk_on?'ACTIONS':'OBLIGATIONS'}</div><div class="s">S&amp;P à ${SP(N.dist)} de sa SMA200 au ${N.date}</div></div>
<div class="kpi"><div class="l">P(BASCULE DANS 12 MOIS)</div><div class="v wa">${P(N.p_switch12,0)}</div><div class="s">fréquence historique à distance ±3 pts (n = ${N.n_hist})</div></div>
<div class="kpi"><div class="l">BÊTA 63 JOURS</div><div class="v ${N.beta63>1?'dn':''}">${F(N.beta63)}</div><div class="s">contre ${F(R.full.beta)} en moyenne historique</div></div>
<div class="kpi"><div class="l">VOLATILITÉ 63 JOURS</div><div class="v">${P(N.vol63)}</div><div class="s">S&amp;P : ${P(N.bvol63)} · drawdown actuel ${P(N.dd_now)}</div></div></div>
<table><thead><tr><th class="l">POSITION (signal du ${N.signal_date})</th><th class="l">SECTEUR</th><th>POIDS</th><th>MOMENTUM 12 MOIS</th></tr></thead><tbody>
${N.hold.map(x=>row([x.t,`<span class="l">${names[x.t]||x.t}</span>`,'20 %',SP(x.mom)])).join('')}
</tbody></table>
<p class="note">Point de vigilance : le portefeuille actuel est concentré sur la technologie au sens large (semi-conducteurs et technologie, 40 %) et affiche un bêta à court terme de ${F(N.beta63)}, bien au-dessus de sa moyenne historique de ${F(R.full.beta)}. À court terme, la stratégie est donc <b>plus</b> risquée que l'indice, pas moins : sa protection ne s'active qu'après une cassure de la SMA200, avec retard. La probabilité historique de ${P(N.p_switch12,0)} de bascule dans les 12 prochains mois signale un risque élevé de faux signal (vente puis rachat plus haut).</p>`);

h+=panel('10','Tests de résistance historiques',`
<table><thead><tr><th class="l">ÉPISODE</th><th class="l">PÉRIODE</th><th>STRATÉGIE</th><th>S&amp;P 500 TR</th><th>ÉCART</th><th>DD MAX STRATÉGIE</th></tr></thead><tbody>
${D.episodes.map(e=>row([e.label,`<span class="l">${e.start} → ${e.end}</span>`,`<span class="${C(e.s)}">${SP(e.s)}</span>`,`<span class="${C(e.b)}">${SP(e.b)}</span>`,`<span class="${C(e.s-e.b)}">${SP(e.s-e.b)}</span>`,P(e.s_dd)])).join('')}
</tbody></table>
<p class="note">Profil asymétrique très net : la stratégie excelle dans les baisses <b>lentes</b> (2000-02, 2008-09) où le filtre a le temps d'agir, et n'apporte rien, ou fait pire, dans les chocs <b>rapides</b> (2018) et quand actions et obligations baissent ensemble (2022).</p>`);

h+=panel('11','Limites de l\'analyse',`
<ul>
<li><b>Biais de survivant :</b> les 26 fonds existent encore aujourd'hui ; les secteurs fermés ou fusionnés sont absents.</li>
<li><b>Biais de sélection :</b> le DSR corrige la multiplicité des 840 essais, mais pas les choix de conception faits avant (univers, coût, horizon).</li>
<li><b>Non-stationnarité :</b> tous les modèles supposent que l'avenir ressemble à une partie du passé. Une rupture future (régime de taux, structure du marché) n'est captée par aucun.</li>
<li><b>Mise en œuvre :</b> FNB sectoriels plutôt que fonds Fidelity ; écarts de suivi, frais de change (gambit de Norbert), retenue fiscale américaine dans le CELI.</li>
<li><b>Hypothèse de prime de risque :</b> les résultats de M4 dépendent fortement de π (section 8). Le choix π = ${P(Q.erp0)} ± ${P(Q.erp_sd)} est une hypothèse de travail, pas une prévision.</li>
<li><b>Granularité :</b> M3 et M4 sont mensuels ; leurs drawdowns sont sous-estimés.</li>
</ul>`);

h+=panel('12','Conclusions et recommandations',`
<ol>
<li><b>Calibrer les attentes :</b> espérance de rendement d'environ ${SP(ann(E.med))} par an (médiane), avec 90 % de chances de finir entre ${SP(E.p5,0)} et ${SP(E.p95,0)} sur 3 ans. Probabilité de battre le S&amp;P 500 : environ ${P(E.p_beat,0)}, soit essentiellement un pile ou face.</li>
<li><b>Taille de position :</b> avec un alpha a posteriori de ${SP(Q.a_post)} ± ${P(Q.a_post_sd)}, le critère de Kelly appliqué à l'alpha seul justifie une allocation active modeste. Garder le satellite à 10 % ou moins du capital.</li>
<li><b>Refuge :</b> remplacer le Trésor long terme par des bons du Trésor ou du cash (FNB à très court terme). Cela réduit le risque de corrélation positive actions-obligations observé en 2022, au prix d'environ 2 points de rendement par an selon le backtest.</li>
<li><b>Règles de révision fixées à l'avance :</b> (a) si la valeur réelle passe sous la bande du 10e centile de l'éventail pendant 6 mois consécutifs, suspendre et réévaluer ; (b) recalculer l'alpha roulant sur 36 mois chaque année ; s'il est négatif avec t &lt; −1,5, arrêter ; (c) kill switch existant à −50 % maintenu.</li>
<li><b>Ne pas juger la stratégie sur 3 ans :</b> le MinTRL de ${S.min_trl?Math.round(S.min_trl/12):'–'} ans montre qu'aucun résultat sur 36 mois, bon ou mauvais, ne sera statistiquement concluant. Juger le processus (respect des règles), pas le résultat à court terme.</li>
</ol>
<p class="note">Reproductibilité : <code>python forecast.py</code> (graine aléatoire fixe 2026, ${(E.n).toLocaleString('fr-CA')} trajectoires dans l'ensemble, 10 000 par modèle). Références : Andrews (1993) Econometrica 61(4) ; Bailey &amp; López de Prado (2014) J. Portfolio Management 40(5) ; Lo (2002) Financial Analysts Journal 58(4) ; Newey &amp; West (1987) Econometrica 55(3) ; Politis &amp; Romano (1994) JASA 89(428) ; Treynor &amp; Mazuy (1966) Harvard Business Review 44.</p>`);
$('#m').innerHTML=h;
renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\(',right:'\)',display:false}],throwOnError:false});

Chart.defaults.color='#6f8593';Chart.defaults.borderColor='#1b2c38';Chart.defaults.font.family='ui-monospace,Menlo,Consolas,monospace';Chart.defaults.font.size=10;
const UP='#1dc7a0',DN='#d6386a',BL='#3b82f6',WA='#f0b429',x={ticks:{maxTicksLimit:12}};
const o=(e={})=>({animation:false,interaction:{mode:'index',intersect:false},elements:{point:{radius:0}},scales:{x},...e});
new Chart($('#cw'),{type:'line',data:{labels:SW.series.t,datasets:[{label:'Statistique de Wald',data:SW.series.w,borderColor:WA,borderWidth:1.4},{label:'VC 5 %',data:SW.series.t.map(()=>SW.cv5),borderColor:DN,borderDash:[4,3],borderWidth:1},{label:'VC 10 %',data:SW.series.t.map(()=>SW.cv10),borderColor:'rgba(214,56,106,.5)',borderDash:[2,3],borderWidth:1}]},options:o()});
new Chart($('#ca'),{type:'line',data:{labels:D.rolling.t,datasets:[{label:'α 60 mois (%/an)',data:D.rolling.alpha,borderColor:UP,borderWidth:1.3,fill:{target:{value:0},above:'rgba(29,199,160,.12)',below:'rgba(214,56,106,.18)'}}]},options:o({plugins:{legend:{display:false}}})});
new Chart($('#cr'),{type:'line',data:{labels:D.rel.t,datasets:[{label:'log relatif',data:D.rel.v,borderColor:BL,borderWidth:1.3}]},options:o({plugins:{legend:{display:false}}})});
new Chart($('#ch'),{type:'line',data:{labels:D.hmm.p_stress.t,datasets:[{label:'P(régime de stress | données)',data:D.hmm.p_stress.v,borderColor:DN,backgroundColor:'rgba(214,56,106,.25)',fill:true,borderWidth:1}]},options:o({scales:{x,y:{min:0,max:1}}})});
const f=D.fan,ml=f.m.map(m=>'M'+m);
new Chart($('#cf'),{type:'line',data:{labels:ml,datasets:[
{label:'Stratégie 95e',data:f.s[4],borderColor:'rgba(29,199,160,.3)',borderWidth:1,fill:'+4',backgroundColor:'rgba(29,199,160,.07)'},
{label:'Stratégie 75e',data:f.s[3],borderColor:'rgba(29,199,160,.55)',borderWidth:1,fill:'+2',backgroundColor:'rgba(29,199,160,.12)'},
{label:'Stratégie médiane',data:f.s[2],borderColor:UP,borderWidth:2},
{label:'Stratégie 25e',data:f.s[1],borderColor:'rgba(29,199,160,.55)',borderWidth:1},
{label:'Stratégie 5e',data:f.s[0],borderColor:'rgba(29,199,160,.3)',borderWidth:1},
{label:'S&P médiane',data:f.b[2],borderColor:BL,borderWidth:2,borderDash:[5,3]},
{label:'S&P 5e / 95e',data:f.b[0],borderColor:'rgba(59,130,246,.45)',borderWidth:1,borderDash:[2,3]},
{label:' ',data:f.b[4],borderColor:'rgba(59,130,246,.45)',borderWidth:1,borderDash:[2,3]}]},options:o()});
new Chart($('#chs'),{type:'bar',data:{labels:D.hist.e,datasets:[{label:'Stratégie',data:D.hist.s,backgroundColor:'rgba(29,199,160,.7)'},{label:'S&P 500',data:D.hist.b,backgroundColor:'rgba(59,130,246,.5)'}]},options:{animation:false,scales:{x:{stacked:false,ticks:{maxTicksLimit:14}}}}});
new Chart($('#che'),{type:'bar',data:{labels:D.hist_ex.e,datasets:[{label:'Écart',data:D.hist_ex.h,backgroundColor:D.hist_ex.e.map(v=>v>=0?'rgba(29,199,160,.7)':'rgba(214,56,106,.7)')}]},options:{animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{maxTicksLimit:14}}}}});
</script></body></html>"""

# Le LaTeX vit dans des gabarits JavaScript : chaque barre oblique doit y être doublée.
HTML = re.sub(r"\\(?=[A-Za-z{},!%|;()])", r"\\\\", HTML)
