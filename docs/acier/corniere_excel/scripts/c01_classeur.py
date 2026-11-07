# -*- coding: utf-8 -*-
"""c01 — Le classeur relu et reproduit.

1. ouvre le classeur (openpyxl, formules ET valeurs en cache) ;
2. recalcule la feuille P8 avec ``commun.classeur`` (recopie formule par formule) et compare CHAQUE valeur en cache ;
3. contrôle les résultats historiques donnés par l'ingénieur ;
4. relit la feuille CORNIERES : doublons de nom (RECHERCHEV), rayons face au catalogue du site
   (/home/user/integ/acier/donnees/profiles.json, LECTURE seule), unités.

Lancement : /usr/bin/python3.12 -I c01_classeur.py   (depuis un autre dossier que source/)
"""
import json
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import commun as K  # noqa: E402

import openpyxl  # noqa: E402

RACINE = os.path.dirname(ICI)
XLSX = os.path.join(RACINE, "source", "realex_cornieres.xlsx")
SORTIE = os.path.join(RACINE, "sorties", "c01_classeur.json")
CATALOGUE = "/home/user/integ/acier/donnees/profiles.json"

wb_f = openpyxl.load_workbook(XLSX, data_only=False)
wb_v = openpyxl.load_workbook(XLSX, data_only=True)
print("Feuilles :", wb_f.sheetnames)
p8 = [n for n in wb_f.sheetnames if n.strip() == "P8"][0]
Ff, Fv = wb_f[p8], wb_v[p8]

res = {}
# ---------------------------------------------------------------- 1. formules clés relues dans le fichier
cles = ["C19", "C20", "C23", "C24", "C25", "C32", "C35", "C42", "C45", "C55", "C56", "C57", "C63", "C64", "C65",
        "G31", "G34", "G37", "G44", "G54", "G64", "H31", "H34", "D3", "C82", "C83", "C85", "C86", "C88", "C90",
        "C75", "C76", "C8", "C14", "C17", "G85", "G86"]
formules = {k: (Ff[k].value, Fv[k].value) for k in cles}
for k, (f, v) in formules.items():
    print(f"  P8!{k:4s} {str(f):55s} => {v}")
res["formules"] = {k: [str(f), v] for k, (f, v) in formules.items()}

# ---------------------------------------------------------------- 2. recalcul et comparaison au cache
c = K.classeur(**K.CAS)
corresp = {
    "C19": c["C19_Xe"], "C20": c["C20_Xer"], "C23": c["C23_Ye"], "C24": c["C24_Yer"], "C25": c["C25_YH"],
    "A20": c["A20_larg_moins_X"], "A21": c["A21_haut_moins_Y"],
    "C32": c["S1"]["W"], "C35": c["S1"]["M"], "G31": c["S1"]["sigma"], "G34": c["S1"]["tau"],
    "G37": c["S1"]["sigma_eq"], "I37": c["S1"]["taux"], "I31": c["S1"]["taux_sigma"], "I34": c["S1"]["taux_tau"],
    "C42": c["S2"]["W"], "C45": c["S2"]["M"], "G41": c["S2"]["sigma"], "G44": c["S2"]["tau"],
    "G47": c["S2"]["sigma_eq"], "I47": c["S2"]["taux"],
    "C55": c["MMax"], "C56": c["C56_rapport3"], "C57": c["S3"]["M"], "G51": c["S3"]["sigma"],
    "G54": c["S3"]["tau"], "G57": c["S3"]["sigma_eq"], "I57": c["S3"]["taux"],
    "C63": c["MMax"], "C64": c["C64_rapport4"], "C65": c["S4"]["M"], "G61": c["S4"]["sigma"],
    "G64": c["S4"]["tau"], "G67": c["S4"]["sigma_eq"], "I67": c["S4"]["taux"],
    "H31": c["adm"], "H34": c["tadm"], "D3": c["D3_taux_max"],
    "C78": c["MMax"], "C82": c["N_ELS"], "C83": c["V_ELS"], "C85": c["N_ELU"], "C86": c["V_ELU"],
    "C88": c["C88_pression"], "C90": c["C90_arrachement"],
}
ecart_max = 0.0
for k, val in corresp.items():
    cache = Fv[k].value
    ec = abs(val - cache) / max(abs(cache), 1e-12)
    ecart_max = max(ecart_max, ec)
    if ec > 1e-12:
        print("  ÉCART", k, val, cache)
print(f"Recalcul de {len(corresp)} cellules de P8 : écart relatif maximal au cache = {ecart_max:.1e}")
res["reproduction"] = dict(cellules=len(corresp), ecart_relatif_max=ecart_max)

# ---------------------------------------------------------------- 3. résultats historiques de l'ingénieur
hist = {"M1": c["S1"]["M"], "M2": c["S2"]["M"], "M3": c["S3"]["M"], "M4": c["S4"]["M"], "taux": c["D3_taux_max"]}
gouv = max(("S1", "S2", "S3", "S4"), key=lambda s: c[s]["taux"])
for k, v in hist.items():
    print(f"  {k} = {v:.6f}   (historique {K.HISTO[k]})")
print("  section gouvernante :", gouv)
res["historique"] = dict(calcule=hist, donne=K.HISTO, gouvernante=gouv,
                         taux_sections={s: c[s]["taux"] for s in ("S1", "S2", "S3", "S4")})

# rapport ELU/ELS des ancrages et valeurs HILTI notées en G85/G86
rap = K.CAS["R_elu"] / K.CAS["R"]
g85, g86 = Fv["G85"].value, Fv["G86"].value
z_hilti = K.CAS["R_elu"] * (K.CAS["X"] + K.CAS["Xj"]) / (K.CAS["n_anc"] * g85)
res["ancrages"] = dict(rapport_ELU_ELS=rap, N_ELU_classeur=c["N_ELU"], G85=g85, G86=g86,
                       G85_sur_C85=g85 / c["N_ELU"], bras_implicite_G85_mm=z_hilti,
                       pivot_implicite_G85_mm=K.CAS["Y"] - z_hilti, bras_classeur_mm=c["z_ancrage_mm"])
print(f"Rapport C71/C70 = {rap:.4f} ; G85/C85 = {g85 / c['N_ELU']:.4f} ; bras implicite de G85 = {z_hilti:.2f} mm "
      f"(classeur : {c['z_ancrage_mm']:.1f} mm)")

# ---------------------------------------------------------------- 4. feuille CORNIERES
Fc = wb_v["CORNIERES"]
lignes = []
for row in range(2, Fc.max_row + 1):
    nom = Fc.cell(row, 1).value
    if nom is None:
        continue
    vals = [Fc.cell(row, j).value for j in range(2, 7)]
    lignes.append((row, nom, *vals))
noms = [l[1] for l in lignes]
doublons = sorted({n for n in noms if noms.count(n) > 1})
inaccessibles = []
for n in doublons:
    rows = [l for l in lignes if l[1] == n]
    premier = rows[0]
    for l in rows[1:]:
        inaccessibles.append(dict(nom=n, ligne=l[0], larg=l[2], haut=l[3],
                                  rendu_par_RECHERCHEV=dict(ligne=premier[0], larg=premier[2], haut=premier[3])))
print(f"CORNIERES : {len(lignes)} lignes, {len(doublons)} noms en double ; lignes jamais atteintes par "
      f"RECHERCHEV(...;FAUX) : {[d['ligne'] for d in inaccessibles]}")
res["catalogue_doublons"] = inaccessibles

with open(CATALOGUE, encoding="utf-8") as f:
    cat = json.load(f)["profils"]
Ls = {p["nom"]: p for p in cat if p.get("famille") in ("L", "LI")}


def cherche(larg, haut, t):
    def nm(v):
        return ("%g" % v)
    for a, b in ((larg, haut), (haut, larg)):
        n = f"L {nm(a)}x{nm(b)}x{nm(t)}"
        if n in Ls:
            return Ls[n]
    return None


comp = []
for (row, nom, larg, haut, t, r, masse) in lignes:
    if nom == "NON STANDARD":
        continue
    p = cherche(larg, haut, t)
    if p is None:
        comp.append(dict(ligne=row, nom=nom, r_classeur=r, absent=True))
        continue
    d = p["dims"]
    comp.append(dict(ligne=row, nom=nom, larg=larg, haut=haut, r_classeur=r, r1_site=d["r1"], r2_site=d["r2"],
                     masse_classeur=masse, masse_site=round(p["masse"], 3), egales=(larg == haut)))
absents = [x for x in comp if x.get("absent")]
pres = [x for x in comp if not x.get("absent")]
r_ok = [x for x in pres if abs(x["r_classeur"] - x["r1_site"]) < 1e-9]
r_r2 = [x for x in pres if abs(x["r_classeur"] - x["r1_site"]) >= 1e-9 and abs(x["r_classeur"] - x["r2_site"]) < 1e-9]
r_autre = [x for x in pres if abs(x["r_classeur"] - x["r1_site"]) >= 1e-9 and abs(x["r_classeur"] - x["r2_site"]) >= 1e-9]
print(f"Rayons : {len(pres)} lignes trouvées au catalogue du site ({len(absents)} absentes) ; "
      f"r = r1 (congé) : {len(r_ok)} ; r = r2 (arrondi de bout d'aile) : {len(r_r2)} ; autre : {len(r_autre)}")
for x in r_r2 + r_autre:
    print(f"   ligne {x['ligne']:2d} {x['nom']:12s} ({'égales' if x['egales'] else 'inégales'}) r classeur = "
          f"{x['r_classeur']}  r1 site = {x['r1_site']}  r2 site = {x['r2_site']}")
for x in absents:
    print(f"   ligne {x['ligne']:2d} {x['nom']:12s} absente du catalogue du site")
res["catalogue_rayons"] = dict(trouvees=len(pres), absentes=[x["nom"] for x in absents],
                               r_egal_r1=len(r_ok), r_egal_r2=[(x["ligne"], x["nom"]) for x in r_r2],
                               r_autre=[(x["ligne"], x["nom"], x["r_classeur"], x["r1_site"], x["r2_site"])
                                        for x in r_autre],
                               inegales_toutes_fausses=all(not x["egales"] for x in r_r2 + r_autre),
                               nb_inegales=len([x for x in pres if not x["egales"]]),
                               nb_egales_justes=len([x for x in r_ok if x["egales"]]),
                               nb_egales=len([x for x in pres if x["egales"]]))

# masses
ecm = [(x["nom"], x["masse_classeur"], x["masse_site"], (x["masse_classeur"] - x["masse_site"]) / x["masse_site"])
       for x in pres]
pire = max(ecm, key=lambda t: abs(t[3]))
print(f"Masses : écart relatif max classeur/site = {pire[3] * 100:+.1f} % ({pire[0]} : {pire[1]} contre {pire[2]})")
res["catalogue_masses_pire"] = pire

# cornière du cas
p = cherche(120, 120, 15)
print("Cas : L120x120x15 au site :", p["dims"], "masse", p["masse"])
res["cas_site"] = p["dims"]

# feuille E10A BD : liée à P8 ?
txt = []
for row in Ff.iter_rows():
    for cell in row:
        if isinstance(cell.value, str) and cell.value.startswith("=") and "E10A" in cell.value:
            txt.append(cell.coordinate)
print("Formules de P8 qui lisent la feuille E10A BD :", txt or "aucune")
res["P8_lit_E10A"] = txt

with open(SORTIE, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1, default=str)
print("→", SORTIE)
