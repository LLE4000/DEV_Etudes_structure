# -*- coding: utf-8 -*-
"""S08 — Le catalogue CORNIERES du classeur face au catalogue du site (acier/donnees/profiles.json, lecture seule).

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/s08_catalogue.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openpyxl  # noqa: E402
import s_commun as K  # noqa: E402

S = K.Sortie("s08_catalogue")
wb = openpyxl.load_workbook(K.XLSX, data_only=True)
ws = wb["CORNIERES"]
S("# S08 — catalogues")
S(f"en-têtes : {[c.value for c in ws[1]][:7]}")
lignes = []
for i, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=7, values_only=True), start=2):
    if row[0] is None:
        continue
    lignes.append(dict(ligne=i, nom=row[0], larg=row[1], haut=row[2], e=row[3], r=row[4], masse=row[5], acier=row[6]))
corn = [l_ for l_ in lignes if l_["nom"] != "NON STANDARD"]
S(f"{len(lignes)} lignes nommées (dont « NON STANDARD ») ; {len(corn)} cornières ; {len(set(l_['nom'] for l_ in corn))} désignations distinctes")
S(f"colonne G (ACIER) : {[l_['acier'] for l_ in lignes if l_['acier'] is not None]}")
vus = {}
doubles = []
for l_ in corn:
    if l_["nom"] in vus:
        p = vus[l_["nom"]]
        doubles.append((l_["nom"], p["ligne"], (p["larg"], p["haut"]), l_["ligne"], (l_["larg"], l_["haut"])))
    else:
        vus[l_["nom"]] = l_
S(f"{len(doubles)} désignations en double (RECHERCHEV exacte lit la première) :")
for nom, l1, d1, l2, d2 in doubles:
    S(f"   {nom} : ligne {l1} {d1[0]}×{d1[1]} lue ; ligne {l2} {d2[0]}×{d2[1]} jamais atteinte")
S.v("doubles", doubles)

site = {}
for q in K.profils_site(("L", "LI")):
    site[q["nom"].replace(" ", "")] = q
S(f"catalogue du site : {len(site)} cornières (familles L et LI)")
pres, abs_, ray = [], [], {"r1": [], "r2": [], "autre": []}
masse = []
for l_ in corn:
    q = site.get(l_["nom"])
    if q is None:
        abs_.append(l_)
        continue
    d = q["dims"]
    dims_ok = sorted([d["h"], d["b"]]) == sorted([float(l_["larg"]), float(l_["haut"])]) and abs(d["t"] - l_["e"]) < 1e-9
    pres.append((l_, q, dims_ok))
    if abs(l_["r"] - d["r1"]) < 1e-9:
        ray["r1"].append(l_)
    elif abs(l_["r"] - d["r2"]) < 1e-9:
        ray["r2"].append((l_, d))
    else:
        ray["autre"].append((l_, d))
    if l_["masse"]:
        masse.append((l_["masse"] / q["masse"] - 1, l_["nom"], l_["masse"], q["masse"]))
noms_pres = sorted(set(l_["nom"] for l_, _, _ in pres))
noms_abs = sorted(set(l_["nom"] for l_ in abs_))
S(f"présentes au site : {len(noms_pres)} désignations ({len(pres)} lignes) ; dimensions identiques : {sum(1 for *_, ok in pres if ok)} lignes sur {len(pres)}")
S(f"absentes du site : {len(noms_abs)} désignations ({len(abs_)} lignes) : {', '.join(noms_abs)}")
S(f"rayon = r1 (congé) : {len(ray['r1'])} lignes ; = r2 (bout d'aile) : {len(ray['r2'])} lignes "
  + "; ".join(f"{l_['nom']} ligne {l_['ligne']} : {l_['r']} au lieu de r1 = {d['r1']}" for l_, d in ray["r2"]))
S("rayon différent des deux : " + "; ".join(f"{l_['nom']} : {l_['r']} (r1 = {d['r1']}, r2 = {d['r2']})" for l_, d in ray["autre"]))
masse.sort()
S(f"masse (colonne F, non utilisée par le calcul) : écart de {100 * masse[0][0]:+.2f} % ({masse[0][1]} : {masse[0][2]} contre {masse[0][3]:.3f}) "
  f"à {100 * masse[-1][0]:+.2f} % ({masse[-1][1]})")
# rayons des inégales absentes : r du classeur face aux r2 typiques
for l_ in abs_:
    if l_["larg"] != l_["haut"]:
        S(f"   inégale absente du site : {l_['nom']} (ligne {l_['ligne']}) r = {l_['r']}")
_cor = K.profils_site(("L", "LI"))
_p300 = [q for q in _cor if q["nom"] == "L 300x300x35"][0]
S(f"catalogue du site : {len(_cor)} cornières (L {sum(q['famille'] == 'L' for q in _cor)}, LI {sum(q['famille'] == 'LI' for q in _cor)}) ; "
  f"« courant » = true : {sum(q.get('courant') is True for q in _cor)}, false : {sum(q.get('courant') is False for q in _cor)} "
  f"(docs/acier/PROFILES.md §2 : cornières complémentaires hors liste EN 10056-1) ; L 300x300x35 : courant = {_p300.get('courant')}, "
  f"r1 = {_p300['dims']['r1']:g}")
S.v("pres", noms_pres)
S.v("abs", noms_abs)
S.v("r2", [(l_["nom"], l_["ligne"], l_["r"], d["r1"]) for l_, d in ray["r2"]])
S.v("autre", [(l_["nom"], l_["r"], d["r1"]) for l_, d in ray["autre"]])
S.v("n", dict(lignes=len(corn), noms=len(set(l_['nom'] for l_ in corn)), site=len(site), pres=len(pres), abs=len(abs_),
              r1=len(ray["r1"])))
S.ecrire()
