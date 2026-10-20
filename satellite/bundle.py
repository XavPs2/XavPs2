#!/usr/bin/env python3
"""Regroupe les rapports HTML en un seul fichier autonome : rapports_satellite.html.
Usage: python bundle.py (après report.py --strategy all et forecast.py)."""
import base64

REPORTS = [("report_sector.html", "Rotation sectorielle"), ("report_forecast.html", "Perspectives 36 mois"),
           ("report.html", "Rotation gagnant annuel"), ("report_meanrev.html", "Repli RSI · 25 titres"),
           ("report_dip.html", "Repli RSI(3) · S&P 500")]
HIDE = "<style>body>header{display:none!important}</style>"   # la barre d'onglets interne ne sert plus

docs = []
for f, title in REPORTS:
    html = open(f, encoding="utf-8").read().replace("</head>", HIDE + "</head>", 1)
    docs.append((title, base64.b64encode(html.encode("utf-8")).decode("ascii")))

tabs = "".join(f'<a href="#" data-i="{i}">{t}</a>' for i, (t, _) in enumerate(docs))
data = ",".join(f'"{b}"' for _, b in docs)
page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rapports satellite</title>
<style>
:root{{--bg:#0a141c;--line:#1b2c38;--tx:#d5e0e6;--mut:#6f8593;--up:#1dc7a0}}
html,body{{margin:0;height:100%;background:var(--bg);color:var(--tx);font:13px ui-monospace,Menlo,Consolas,monospace}}
header{{display:flex;align-items:center;gap:22px;padding:10px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap}}
header b{{letter-spacing:.06em}}header a{{color:var(--mut);text-decoration:none;padding-bottom:6px}}
header a.on{{color:var(--tx);border-bottom:2px solid var(--up)}}
iframe{{border:0;width:100%;height:calc(100vh - 46px);display:none}}iframe.on{{display:block}}
</style></head><body>
<header><b>◆ SATELLITE</b>{tabs}</header>
<div id="frames"></div>
<script>
const DOCS=[{data}];
const dec=b=>new TextDecoder().decode(Uint8Array.from(atob(b),c=>c.charCodeAt(0)));
const frames=[];
function show(i){{
  document.querySelectorAll('header a').forEach((a,j)=>a.classList.toggle('on',j==i));
  frames.forEach(f=>f&&f.classList.remove('on'));
  if(!frames[i]){{const f=document.createElement('iframe');f.className='on';document.getElementById('frames').appendChild(f);f.srcdoc=dec(DOCS[i]);frames[i]=f;}}
  frames[i].classList.add('on');
}}
document.querySelectorAll('header a').forEach(a=>a.onclick=e=>{{e.preventDefault();show(+a.dataset.i)}});
show(0);
</script></body></html>"""
open("rapports_satellite.html", "w", encoding="utf-8").write(page)
print("rapports_satellite.html écrit")
