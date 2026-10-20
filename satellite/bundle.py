#!/usr/bin/env python3
"""Regroupe les rapports HTML en un seul fichier : rapports_satellite.html.
Chaque rapport est intégré en clair dans un <iframe srcdoc>, sans encodage ni décodage JavaScript
(un contenu encodé en base64 puis décodé est bloqué par Gmail et les antivirus comme du « HTML smuggling »).
Usage: python bundle.py (après report.py --strategy all et forecast.py)."""
import html

REPORTS = [("report_sector.html", "Rotation sectorielle"), ("report_forecast.html", "Perspectives 36 mois"),
           ("report.html", "Rotation gagnant annuel"), ("report_meanrev.html", "Repli RSI · 25 titres"),
           ("report_dip.html", "Repli RSI(3) · S&P 500")]
HIDE = "<style>body>header{display:none!important}</style>"   # la barre d'onglets interne ne sert plus

tabs, frames = [], []
for i, (f, title) in enumerate(REPORTS):
    doc = open(f, encoding="utf-8").read().replace("</head>", HIDE + "</head>", 1)
    on = " on" if i == 0 else ""
    tabs.append(f'<button class="tab{on}" data-i="{i}">{html.escape(title)}</button>')
    frames.append(f'<iframe class="pg{on}" title="{html.escape(title)}" srcdoc="{html.escape(doc, quote=True)}"></iframe>')

page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rapports satellite</title>
<style>
:root{{--bg:#0a141c;--line:#1b2c38;--tx:#d5e0e6;--mut:#6f8593;--up:#1dc7a0}}
html,body{{margin:0;height:100%;background:var(--bg);color:var(--tx);font:13px ui-monospace,Menlo,Consolas,monospace;overflow:hidden}}
header{{display:flex;align-items:center;gap:18px;padding:10px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap;height:46px;box-sizing:border-box}}
header b{{letter-spacing:.06em}}
.tab{{background:none;border:0;border-bottom:2px solid transparent;color:var(--mut);font:inherit;cursor:pointer;padding:0 0 6px}}
.tab.on{{color:var(--tx);border-bottom-color:var(--up)}}
.pg{{position:absolute;top:46px;left:0;width:100%;height:calc(100% - 46px);border:0;visibility:hidden}}
.pg.on{{visibility:visible}}
</style></head><body>
<header><b>◆ SATELLITE</b>{"".join(tabs)}</header>
{"".join(frames)}
<script>
const tabs=document.querySelectorAll('.tab'),pgs=document.querySelectorAll('.pg');
tabs.forEach(t=>t.addEventListener('click',()=>{{const i=+t.dataset.i;
  tabs.forEach((x,j)=>x.classList.toggle('on',j===i));pgs.forEach((x,j)=>x.classList.toggle('on',j===i));}}));
</script></body></html>"""
open("rapports_satellite.html", "w", encoding="utf-8").write(page)
print("rapports_satellite.html écrit")
