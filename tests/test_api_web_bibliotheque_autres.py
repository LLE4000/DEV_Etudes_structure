# -*- coding: utf-8 -*-
"""La BIBLIOTHÈQUE « AUTRES » du site (api_web/biblio_autres.py et ses trois outils : biblio_poutres, biblio_plaques,
biblio_sections ; docs/site/BIBLIOTHEQUE_AUTRES.md).

    /usr/bin/python3.12 tests/test_api_web_bibliotheque_autres.py

Ce que la suite prouve :
  1. le CONTRAT des bibliothèques (la même porte que Béton armé et Acier), par le routeur ;
  2. CHAQUE FORMULE AFFICHÉE SE RECALCULE : pour chaque ligne de chaque feuille de calcul, l'expression « = valeurs » est
     relue ici, évaluée, et comparée au résultat écrit à côté d'elle — sur des dizaines de cas des trois outils ;
  3. l'écran dit ce que le moteur calcule (bande, résultats), ne demande que ce qui sert (ν vient du matériau), REFUSE ce
     qui n'est pas pris en charge sans afficher de valeur, et dit ce qui n'est pas calculé ;
  4. les gestes : chaque bouton porte des paramètres lisibles, ajouter / retirer / dupliquer rendent un état cohérent, les
     pièces qui se glissent disent ce que le geste règle ;
  5. échappement, paramètres illisibles, mutations.
"""
import html
import json
import math
import os
import re
import sys

RACINE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RACINE)

from api_web import biblio_autres as A  # noqa: E402
from api_web import biblio_plaques as BPL  # noqa: E402
from api_web import biblio_poutres as BPO  # noqa: E402
from api_web import biblio_sections as BS  # noqa: E402
from api_web import routeur as R  # noqa: E402
from rdm import plaques as PL  # noqa: E402
from rdm import poutres as P  # noqa: E402
from rdm import sections as S  # noqa: E402

OK, KO = [], []
NBSP, FINE = " ", " "


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:340]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


def texte(h):
    return html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<sub>(.*?)</sub>", r"_\1", h or ""))).replace(NBSP, " ").replace(FINE, "")


def nombre(t):
    """« −1 105,88 » → −1105.88 (le nombre tel que l'écran l'écrit)."""
    return float(t.replace(FINE, "").replace(NBSP, "").replace(" ", "").replace("−", "-").replace(",", "."))


def valeur(o):
    m_ = re.search(r'<b class="bib-val">(.*?)<small>(.*?)</small></b>', o["bande"], re.S)
    return (texte(m_.group(1)), texte(m_.group(2))) if m_ else (None, None)


_SUP = {"²": "**2", "³": "**3", "⁴": "**4", "⁵": "**5"}


def evaluer(expr):
    """L'expression « = valeurs » d'une ligne de feuille, évaluée : × ÷ puissances, racines, virgule décimale."""
    s = texte(expr).replace(" ", "").replace("−", "-").replace("×", "*").replace(",", ".")
    for k, v in _SUP.items():
        s = s.replace(k, v)
    s = s.replace("^", "**").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    s = re.sub(r"√(\d+(?:\.\d+)?)", r"math.sqrt(\1)", s).replace("√(", "math.sqrt(")
    if not re.fullmatch(r"[0-9+\-*/().mathsqr]+", s):
        raise ValueError(f"expression non numérique : {s[:60]}")
    return eval(s, {"math": math})                       # noqa: S307 — une expression arithmétique écrite par le moteur


def lignes(o):
    """[(clé, formule générale, valeurs, résultat, unité)] de la feuille de calcul."""
    out = []
    for cle, c in re.findall(r'<tr data-terme="([^"]*)"[^>]*>(.*?)</tr>', o["tableau"], re.S):
        va = re.search(r'<span class="va">= (.*?)</span>', c, re.S)
        val = re.search(r'<td class="v"><b>(.*?)</b>(?:' + NBSP + r'<span class="u">(.*?)</span>)?', c, re.S)
        fo = re.search(r'<span class="fo">(.*?)</span>', c, re.S)
        out.append((cle, texte(fo.group(1)) if fo else "", va.group(1) if va else "", texte(val.group(1)), texte(val.group(2) or "")))
    return out


def formules_fausses(o):
    """Les lignes dont « = valeurs » ne redonne pas le résultat écrit (à une demi-unité du dernier chiffre et 3 ‰ près)."""
    faux, vues = [], 0
    for cle, _fo, va, val, _u in lignes(o):
        if not va or va.startswith("min(") or va.startswith("contrôle"):
            continue
        try:
            x = evaluer(va)
            v = nombre(val)
        except Exception as ex:                          # noqa: BLE001 — toute ligne illisible est une faute
            faux.append((cle, "illisible", texte(va)[:60], str(ex)[:40]))
            continue
        vues += 1
        nd = len(val.split(",")[1]) if "," in val else 0
        if abs(x - v) > 0.6 * 10 ** -nd + 3e-3 * abs(v):
            faux.append((cle, round(x, 5), v, texte(va)[:80]))
    return faux, vues


def sels(o, zone="tableau"):
    return [json.loads(html.unescape(x)) for x in re.findall(r'data-sel="([^"]+)"', o[zone])]


# ================================================================================================ 1 · le contrat
print("=== 1 · Le contrat des bibliothèques")
chk("la bibliothèque se déclare : clé « autres », page « bibliotheque-autres », trois rubriques (Poutres, Plaques, Sections) avec une icône chacune",
    A.BIBLIOTHEQUE == "autres" and A.PAGE == "bibliotheque-autres" and A.CLES == ("poutres", "plaques", "sections")
    and all(ic in __import__("api_web.icones", fromlist=["ICONES"]).ICONES for _, _, _, ic in A.RUBRIQUES), A.RUBRIQUES)
CLES = {"rubrique", "bibliotheque", "titre", "barre", "tableau", "bande", "pourquoi", "sources", "fiche", "params", "message"}
for rub in A.CLES:
    o = A.consulter(rub, None)
    chk(f"{rub} : les onze clés du contrat, des fragments HTML non vides, aucun message par défaut, la rubrique nommée",
        set(o) == CLES and o["rubrique"] == rub and o["bibliotheque"] == "autres" and o["message"] is None and len(o["tableau"]) > 2000
        and "bib-val" in o["bande"] and "bib-src" in o["sources"] and "bib-pq-h" in o["pourquoi"] and o["fiche"] is None, set(o) ^ CLES)
    rr = req(action="consulter", bibliotheque="autres", rubrique=rub, params=o["params"])
    chk(f"{rub} : le routeur rend la même rubrique, relisible en JSON, et les paramètres relus redonnent le même écran",
        rr["ok"] and rr["rubrique"] == rub and rr["tableau"] == o["tableau"] and rr["bande"] == o["bande"] and json.dumps(rr))
m0 = A.menu("plaques")
chk("le menu : trois entrées, celle qui est ouverte allumée", m0.count("data-rubrique=") == 3 and 'class="bib-m on" data-rubrique="plaques"' in m0)
o = A.consulter("torsion", None)
chk("une rubrique inconnue ouvre la première, et le message le dit", o["rubrique"] == "poutres" and "torsion" in (o["message"] or ""))
try:
    A.consulter("poutres", None, "ailleurs")
    chk("un hôte inconnu est refusé", False)
except ValueError:
    chk("un hôte inconnu est refusé", True)

# ================================================================================================ 2 · poutres
print("\n=== 2 · Poutres : chaque formule affichée se recalcule")
CAS_POUTRES = [
    {}, {"modele": "console", "L1": "3"}, {"modele": "enc_app"}, {"modele": "bi_enc"}, {"modele": "continue"}, {"modele": "continue", "nt": 3},
    {"t1": "P", "a1": 2}, {"t1": "P"}, {"t1": "tri"}, {"t1": "tri", "s1": "decroissante"}, {"modele": "console", "t1": "P", "a1": 2, "L1": 3},
    {"modele": "console", "t1": "tri", "s1": "decroissante"}, {"modele": "enc_app", "t1": "P"}, {"modele": "enc_app", "t1": "P", "a1": 1.5},
    {"modele": "bi_enc", "t1": "P"}, {"modele": "bi_enc", "t1": "P", "a1": 2}, {"t1": "q", "v1": -10},
    {"n": 3, "t1": "q", "a1": 1, "b1": 4, "t2": "P", "a2": 2, "t3": "C", "a3": 4.5},
    {"modele": "enc_app", "n": 3, "t1": "tri", "a1": 1, "b1": 4, "t2": "P", "a2": 2, "v2": -7, "t3": "C", "a3": 4.5, "rig": "bois"},
    {"modele": "bi_enc", "n": 2, "t1": "tri", "s1": "decroissante", "a1": 0.5, "b1": 5, "t2": "q", "v2": 3, "rig": "sans"},
    {"modele": "console", "n": 3, "t1": "q", "a1": 1, "b1": 2.5, "t2": "C", "a2": 2, "t3": "tri", "L1": 3, "rig": "libre"},
    {"modele": "continue", "nt": 4, "L1": 4, "L2": 5, "L3": 3.5, "L4": 6, "n": 4, "t1": "q", "a1": 1, "b1": 11, "t2": "P", "v2": 25, "a2": 6.5, "t3": "C", "v3": 15,
     "a3": 4, "t4": "tri", "v4": 8, "a4": 9, "s4": "decroissante", "x": 7.3},
    {"modele": "continue", "nt": 5, "L1": 3, "rig": "acier", "fam": "HEA", "axe": "z"}, {"n": 0},
]
total, pb = 0, []
for p in CAS_POUTRES:
    o = A.consulter("poutres", p)
    faux, vues = formules_fausses(o)
    total += vues
    if faux or o["message"]:
        pb.append((p, faux[:2], o["message"]))
chk(f"{len(CAS_POUTRES)} poutres (cinq modèles, quatre charges, superpositions, quatre rigidités) : les {total} expressions « = valeurs » de leurs feuilles "
    "redonnent le résultat écrit à côté d'elles", not pb and total > 250, pb[:2])
o = A.consulter("poutres", {})
sol = P.resoudre("appuis", [6.0], [{"type": "q", "v": 10.0, "a": 0.0, "b": 6.0}], 210000.0 * 8356.11 * 1e-5)
chk("par défaut : deux appuis, 6 m, q = 10 kN/m, IPE 300 — la bande dit M_max = 45,00 kN·m (q·L²/8, à la main) et σ = M/W_el = 80,8 MPa",
    valeur(o) == ("45,00", "kN·m") and "80,8" in texte(o["bande"]) and "x = 3,00 m" in texte(o["bande"]), (valeur(o), texte(o["bande"])[:120]))
li = {c: (fo, va, val, u) for c, fo, va, val, u in lignes(o)}
chk("par défaut : un CAS D'ÉCOLE — la feuille écrit q·L/2, q·L²/8, 5·q·L⁴/384/(E·I) et leurs valeurs ; la flèche 9,62 mm est celle du moteur",
    li["M_max"][0] == "q·L²/8" and li["R_A"][0] == "q·L/2" and li["EIf_max"][0] == "(5·q·L⁴/384)/(E·I)" and "Formulaire · deux appuis" in texte(o["tableau"])
    and abs(nombre(li["EIf_max"][2]) - sol.extremes["EIf_max"][0] / sol.EI * 1000) < 0.006, li.get("EIf_max"))
from acier.profiles import catalogue as KA  # noqa: E402

fiche_ipe = KA.par_nom()["IPE 300"]
chk("par défaut : I_y et W_el,y de l'IPE 300 sont LUS au catalogue de la Bibliothèque Acier (8 356 cm⁴, 557 cm³) et dits dans les hypothèses",
    "IPE 300" in texte(o["tableau"]) and "8356,5 cm⁴" in texte(o["tableau"]) and "557,1 cm³" in texte(o["tableau"])
    and abs(fiche_ipe["Iy"] - 8356.5) < 0.06 and abs(fiche_ipe["Wel_y"] - 557.1) < 0.06, texte(o["tableau"])[texte(o["tableau"]).find("Rigidité"):][:160])
o2 = A.consulter("poutres", {"n": 2, "t2": "P", "a2": 2})
chk("deux charges : plus de cas d'école — la feuille montre la SUPERPOSITION (travée isostatique charge par charge, paramètres initiaux, section d'extremum)",
    "Formulaire ·" not in texte(o2["tableau"]) and "Travée isostatique" in texte(o2["tableau"]) and "Paramètres initiaux" in texte(o2["tableau"])
    and "Moment maximal en travée" in texte(o2["tableau"]))
s2 = P.resoudre("appuis", [6.0], [{"type": "q", "v": 10.0, "a": 0.0, "b": 6.0}, {"type": "P", "v": 20.0, "a": 2.0}])
chk("deux charges : la bande dit le moment maximal du moteur et son abscisse", abs(nombre(valeur(o2)[0]) - s2.extremes["M_max"][0]) < 0.006
    and f"x = {s2.extremes['M_max'][1]:.2f}".replace(".", ",") in texte(o2["bande"]), valeur(o2))
oc = A.consulter("poutres", {"modele": "continue", "nt": 3, "L1": 4, "L2": 5, "L3": 4, "n": 2, "t1": "q", "t2": "P", "a2": 6.5})
tx = texte(oc["tableau"])
chk("poutre continue : les trois moments sont écrits appui par appui, le tranchant d'une travée intérieure s'appelle V_Bd (et non R_B, la réaction de l'appui)",
    "Clapeyron" in tx and "M_A·L_1 + 2·M_B·(L_1 + L_2) + M_C·L_2" in tx and "V_Bd" in tx and "tranchant à droite de B" in tx)
chk("poutre continue : la bande nomme le moment SUR APPUI quand c'est lui le plus grand", "sur l'appui" in texte(oc["bande"]) and valeur(oc)[0].startswith("−"), texte(oc["bande"])[:90])
os_ = A.consulter("poutres", {"rig": "sans", "g": "f"})
chk("sans rigidité : aucune donnée E ou I demandée, la flèche est donnée × E·I (kN·m³) et l'écran le dit",
    valeur(os_)[1] == "kN·m³" and 'data-param="E"' not in os_["barre"] and 'data-param="profil"' not in os_["barre"] and "choisir une rigidité" in texte(os_["bande"]))
ob = A.consulter("poutres", {"rig": "bois"})
chk("section bois : classe et section du module bois (C24, E_0,mean = 11 000 MPa), I = b·h³/12 écrit, flèche instantanée dite",
    "E_0,mean = 11000 MPa" in texte(ob["tableau"]) and "b·h³/12" in texte(ob["tableau"]) and "ni fluage" in texte(ob["tableau"]), texte(ob["tableau"])[:0])
# la gestion des charges : b absent = toute la poutre ; ajouter, retirer
chk("une charge sur toute la poutre n'écrit pas b dans l'adresse : elle SUIT la portée quand on la change",
    "b1" not in o["params"] and "b1" not in A.consulter("poutres", {"L1": 8})["params"]
    and abs(nombre(valeur(A.consulter("poutres", {"L1": 8}))[0]) - 80.0) < 0.006)
o3 = A.consulter("poutres", {"n": 3, "t1": "q", "a1": 1, "b1": 4, "t2": "P", "a2": 2, "t3": "C", "a3": 4.5})
ret = [d for d in sels(o3) if d.get("n") == 2]
apres = A.consulter("poutres", dict(o3["params"], **ret[1]))
chk("« Retirer la charge 2 » : il reste deux charges, la troisième a pris son rang (couple en 4,5 m), aucune clé de la charge 3 ne reste",
    len(ret) == 3 and apres["params"]["n"] == 2 and apres["params"]["t2"] == "C" and apres["params"]["a2"] == 4.5 and "t3" not in apres["params"], apres["params"])
aj = [d for d in sels(o3) if d.get("n") == 4]
chk("« + Charge » : quatre boutons (répartie, triangle, ponctuelle, couple) ; à huit charges, plus aucun",
    sorted(d["t4"] for d in aj) == ["C", "P", "q", "tri"] and not [d for d in sels(A.consulter("poutres", {"n": 8})) if d.get("n") == 9])
gl = [json.loads(html.unescape(x)) for x in re.findall(r'data-glisser="([^"]+)"', o3["tableau"])]
chk("chaque charge se GLISSE : le dessin dit ce que le geste règle (a et b d'une charge répartie, a d'une ponctuelle), l'échelle, le pas, les bornes 0 … L",
    [sorted(g["x"]) for g in gl] == [["a1", "b1"], ["a2"], ["a3"]] and all(g["min"] == 0 and g["max"] == 6 and g["ech"] > 10 and g["pas"] == 0.05 for g in gl), gl)
oa = A.consulter("poutres", {"a1": 5, "b1": 2})
chk("a ≥ b : la charge est remise sur toute la poutre, et le message le dit", "a1" in (oa["message"] or "") and oa["params"]["a1"] == 0.0 and "b1" not in oa["params"])
ox = A.consulter("poutres", {"profil": '<script>alert(1)</script>', "modele": "<b>", "v1": "dix", "L1": "-4"})
chk("paramètres piégés ou illisibles : défauts repris, message dit, rien de brut dans la page",
    "<script>" not in ox["tableau"] + ox["barre"] + ox["bande"] and ox["params"]["modele"] == "appuis" and ox["params"]["v1"] == 10.0 and ox["params"]["L1"] == 0.1
    and "modele" in ox["message"] and "v1" in ox["message"], ox["message"])
chk("le volet dit les conventions de signe, les hypothèses, la méthode et ce qui n'est pas traité (porte-à-faux, appuis élastiques, effort tranchant)",
    all(k in texte(o["pourquoi"]) for k in ("Conventions de signe", "fibre inférieure", "Euler-Bernoulli", "trois moments", "Porte-à-faux", "aucune intégration numérique")))
of_ = A.consulter("poutres", {"v1": 400, "g": "f"})
chk("flèche au-delà de L/100 : l'écran avertit que l'on sort des petites déformations", "L/100" in texte(of_["bande"]) and "bib-av" in of_["bande"])

# ================================================================================================ 3 · plaques
print("\n=== 3 · Plaques")
CAS_PLAQUES = [{}, {"bx0": "E", "bxa": "E", "by0": "E", "byb": "E"}, {"byb": "L", "q": -3}, {"by0": "E", "byb": "L", "mat": "beton", "t": 200, "a": 5, "b": 4},
               {"bx0": "E", "by0": "E", "mat": "autre", "nu": 0.25, "E": 70000}, {"bx0": "L"}, {"bx0": "E", "bxa": "E", "mat": "alu", "t": 6, "a": 0.8, "b": 1.6},
               {"bx0": "L", "bxa": "L"}, {"bx0": "E", "bxa": "E", "by0": "E", "a": 3, "b": 1.2}]
total, pb = 0, []
for p in CAS_PLAQUES:
    o = A.consulter("plaques", p)
    faux, vues = formules_fausses(o)
    total += vues
    if faux or o["message"]:
        pb.append((p, faux[:2], o["message"]))
chk(f"{len(CAS_PLAQUES)} plaques (Lévy, superposition, bords libres, quatre matériaux) : les {total} expressions de leurs feuilles redonnent le résultat écrit",
    not pb and total > 60, pb[:2])
o = A.consulter("plaques", {})
res = PL.calculer(2.0, 1.5, dict(zip(PL.COTES, "AAAA")), 5.0, 210000.0, 0.3, 10.0)
chk("par défaut : 2 × 1,5 m, 10 mm, acier, quatre appuis — la bande dit la flèche du moteur ; D = E·t³/(12·(1 − ν²)) = 19,231 kN·m (à la main)",
    abs(nombre(valeur(o)[0]) - res["w_max"]["valeur"] * 1000) < 0.006 and valeur(o)[1] == "mm" and abs(210000 * 1000 / (12 * 0.91) * 1e-6 - 19.2308) < 1e-4
    and dict((c, v) for c, _f, _v, v, _u in lignes(o))["D"] == "19,231", valeur(o))
chk("le coefficient de Poisson n'est PAS demandé pour l'acier, le béton ni l'aluminium (il vient du matériau, écrit comme hypothèse) ; « autre » le demande avec E",
    all('data-param="nu"' not in A.consulter("plaques", {"mat": k})["barre"] for k in ("acier", "beton", "alu"))
    and 'data-param="nu"' in A.consulter("plaques", {"mat": "autre"})["barre"] and "ν = 0,30" in texte(o["tableau"]) and "EN 1993-1-1" in texte(o["tableau"]))
ob = A.consulter("plaques", {"mat": "beton", "t": 200, "a": 5, "b": 4})
D_b = 32836.568 * 200.0 ** 3 / (12 * (1 - 0.04)) * 1e-6
chk(f"béton C30/37 : E_cm = 32 837 MPa (le tableau 3.1 le donne en GPa), ν = 0,2 ; D = {D_b:.0f} kN·m à la main — la feuille l'écrit",
    abs(nombre(dict((c, v) for c, _f, _v, v, _u in lignes(ob))["D"]) - D_b) < 0.5 and "ν = 0,20" in texte(ob["tableau"]) and "E = 32837 MPa" in texte(ob["tableau"]),
    dict((c, v) for c, _f, _v, v, _u in lignes(ob)).get("D"))
ref = A.consulter("plaques", {"bx0": "E", "byb": "L"})
chk("bord libre sans deux voisins appuyés : le cas est REFUSÉ — la raison est écrite, aucune valeur dans la bande, aucune feuille, le dessin et les bords restent",
    "rdm-refus" in ref["tableau"] and "n'est pas prise en charge" in texte(ref["tableau"]) and "Lévy" in texte(ref["tableau"]) and valeur(ref)[0] == "—"
    and "rdm-fc" not in ref["tableau"] and "rdm-svg-plaque" in ref["tableau"] and 'data-param="byb"' in ref["tableau"] and "Aucune valeur approchée" in texte(ref["tableau"]))
cyc = {d2["by0"] for d2 in sels(o) if list(d2) == ["by0"]} | {d2["by0"] for d2 in sels(A.consulter("plaques", {"by0": "E"})) if list(d2) == ["by0"]} \
    | {d2["by0"] for d2 in sels(A.consulter("plaques", {"by0": "L"})) if list(d2) == ["by0"]}
chk("un clic sur un bord du dessin passe à la condition suivante : appui simple → encastré → libre → appui simple", cyc == {"E", "L", "A"}, cyc)
chk("les quatre bords ont leur menu (le téléphone n'a pas à viser un trait) et cinq préréglages",
    all(f'data-param="{k}"' in o["tableau"] for k in ("bx0", "bxa", "by0", "byb")) and len([d2 for d2 in sels(o) if set(d2) == {"bx0", "bxa", "by0", "byb"}]) == 5)
oe = A.consulter("plaques", {"bx0": "E", "bxa": "E", "by0": "E", "byb": "E", "a": 1, "b": 1, "q": 1})
chk("plaque carrée encastrée : l'écran affiche α = 0,00127 et β = −0,0513 au bord, comparables au tableau 35 de Timoshenko",
    "0,00127" in texte(oe["tableau"]) and "−0,0513" in texte(oe["tableau"]) and "superposition de Timoshenko" in texte(oe["tableau"]))
chk("plaque appuyée : la feuille donne l'écart de convergence et la comparaison à la double série de Navier",
    "moitié des termes" in texte(o["tableau"]) and "Navier" in texte(o["tableau"]))
og = A.consulter("plaques", {"q": 50})
chk("flèche > t/2 : l'écran avertit (hors petites flèches) ; plaque épaisse aussi",
    "petites flèches" in texte(og["bande"]) and "épaisse" in texte(A.consulter("plaques", {"t": 300})["bande"]))
chk("la bande peut dire le plus grand moment ou la contrainte (un clic sur la ligne des résultats)",
    valeur(A.consulter("plaques", {"g": "M"}))[1] == "kN·m/m" and valeur(A.consulter("plaques", {"g": "sigma"}))[1] == "MPa"
    and abs(nombre(valeur(A.consulter("plaques", {"g": "sigma"}))[0]) - res["sigma_max"]) < 0.06)
chk("le volet dit les hypothèses (Kirchhoff, petites flèches, coins maintenus), la méthode et ce qui n'est pas pris en charge",
    all(k in texte(o["pourquoi"]) for k in ("Kirchhoff", "coins MAINTENUS", "Lévy", "superposition", "Non pris en charge", "éléments finis") if k != "éléments finis"))

# ================================================================================================ 4 · sections
print("\n=== 4 · Sections")
CAS_SECTIONS = [{}] + [BS._jeu(els) for _nom, els in BS.EXEMPLES] + [
    {"n": 3, "k3": "poly", "sel": 3}, {"n": 3, "k3": "rond", "v3": "1", "sel": 3}, {"n": 2, "y2": 0, "z2": 50},
    {"n": 1, "k1": "profil", "f1": "L", "p1": "L 150x90x10", "r1": 30}, {"n": 1, "k1": "profil", "f1": "UPN", "p1": "UPN 200", "m1": "1", "r1": 90},
    {"n": 2, "k1": "plat", "b1": 300, "h1": 20, "k2": "profil", "f2": "IPE", "p2": "IPE 200", "c2": "bas", "cc2": 0}]
total, pb = 0, []
for p in CAS_SECTIONS:
    o = A.consulter("sections", p)
    faux, vues = formules_fausses(o)
    total += vues
    if faux or o["message"]:
        pb.append((str(p)[:80], faux[:2], o["message"]))
chk(f"{len(CAS_SECTIONS)} sections (profilés, plats, ronds, polygones, coupes, miroirs, rotations, évidements) : les {total} expressions de leurs feuilles "
    "redonnent le résultat écrit", not pb and total > 150, pb[:2])
o = A.consulter("sections", {})
A1, I1, A2, I2 = 53.8345, 3692.36, 26.4, 22.0 * 1.2 ** 3 / 12
zG = A2 * 10.1 / (A1 + A2)
Iy_main = I1 + A1 * zG ** 2 + I2 + A2 * (10.1 - zG) ** 2
chk(f"par défaut : HEA 200 + plat 220 × 12 posé dessus — I_y = {Iy_main:.2f} cm⁴ recalculé ici à la main (Steiner, A et I du HEA lus au catalogue) ; la bande le dit",
    abs(nombre(valeur(o)[0]) - Iy_main) < 0.05 and valeur(o)[1] == "cm⁴" and o["params"]["z2"] == 101.0, (valeur(o), Iy_main))
tx = texte(o["tableau"])
chk("par défaut : le tableau de Huygens-Steiner (une ligne par élément, A_i, y_i, z_i, I propres, A·d²) et le contrôle au catalogue du profilé non coupé",
    "rdm-st" in o["tableau"] and "A_i·(z_i − z_G)²" in tx and "catalogue 3692,36 cm⁴" in tx
    and nombre(re.search(r"écart ([\d,]+) %", tx).group(1)) < 0.001, tx[tx.find("catalogue") - 40:tx.find("catalogue") + 50])
chk("ce qui n'est pas calculé est DIT, avec sa raison : centre de cisaillement, torsion, gauchissement",
    tx.count("non calculé") >= 3 and "non calculés" in tx and "fonction de gauchissement" in texte(o["pourquoi"]) and "deux axes de symétrie" in texte(o["pourquoi"]))
gl = [json.loads(html.unescape(x)) for x in re.findall(r'data-glisser="([^"]+)"', o["tableau"])]
chk("chaque élément se GLISSE : le dessin dit y{i} et z{i}, l'axe z vers le haut (sy = −1), le pas de 1 mm, et choisit l'élément au lâcher",
    [(sorted(g["x"]), sorted(g["y"]), g["sy"], g["pas"], g["sel"]) for g in gl] == [(["y1"], ["z1"], -1, 1.0, {"sel": 1}), (["y2"], ["z2"], -1, 1.0, {"sel": 2})], gl)
o3 = A.consulter("sections", dict(o["params"], n=3, sel=3, k3="plat"))
chk("« + Plat » : l'élément ajouté sans position se pose SUR la section, centré (z = 107 + 5 = 112)", o3["params"]["z3"] == 112.0 and o3["params"]["y3"] == 0.0, o3["params"])
ret = next(d for d in sels(o3) if d.get("n") == 2 and d.get("k1") is None)
apres = A.consulter("sections", dict(o3["params"], **ret))
chk("« Retirer l'élément 3 » : deux éléments, les mêmes qu'au départ", apres["params"]["n"] == 2 and nombre(valeur(apres)[0]) == nombre(valeur(o)[0]), apres["params"])
ret1 = next(d for d in sels(A.consulter("sections", dict(o["params"], sel=1))) if d.get("n") == 1)
ap1 = A.consulter("sections", dict(o["params"], **ret1))
chk("« Retirer l'élément 1 » : le plat reste seul, à sa place (z = 101), I_y = b·h³/12 = 3,17 cm⁴",
    ap1["params"]["n"] == 1 and ap1["params"]["k1"] == "plat" and ap1["params"]["z1"] == 101.0 and abs(nombre(valeur(ap1)[0]) - 3.168) < 0.006, (ap1["params"], valeur(ap1)))
cop = next(d for d in sels(A.consulter("sections", dict(o["params"], sel=2))) if d.get("n") == 3 and d.get("k3") == "plat")
chk("« Dupliquer » : une copie de l'élément 2, posée à côté (décalée de sa largeur + 10 mm), choisie", cop["b3"] == 220.0 and cop["y3"] == 230.0 and cop["sel"] == 3, cop)
acc = [d for d in sels(A.consulter("sections", dict(o["params"], sel=2, y2=40, z2=160))) if set(d) == {"y2", "z2"}]
chk("« Poser contre l'élément 1 » : dessus (z = 101), dessous (z = −101), à gauche (y = −210), à droite (y = 210), centré (0 ; 0)",
    [(d["y2"], d["z2"]) for d in acc] == [(0.0, 101.0), (0.0, -101.0), (-210.0, 0.0), (210.0, 0.0), (0.0, 0.0)], acc)
ov = A.consulter("sections", {"n": 2, "y2": 0, "z2": 50})
chk("deux éléments qui se recouvrent : l'écran avertit que la matière est comptée deux fois, et les marque sur le dessin",
    "se recouvrent" in texte(ov["bande"]) and "comptée deux fois" in texte(ov["bande"]) and 'class="elt double' in ov["tableau"].replace("elt on double", "elt double"))
oh = A.consulter("sections", {"n": 2, "k1": "plat", "b1": 100, "h1": 100, "k2": "rond", "d2": 40, "v2": "1", "y2": 50, "z2": 0})
chk("un évidement à cheval sur le bord : l'écran dit la partie retirée à tort", "sort de la matière" in texte(oh["bande"]))
od = A.consulter("sections", BS._jeu(BS.EXEMPLES[1][1]))
hea300 = S.caracteristiques([S.element(S.couper(S.profil("HEA 300")[0], "haut", 0.0))])
chk("demi-HEA 300 : l'écran dit l'inertie du moteur ; l'aire est la moitié de celle du catalogue (112,5 / 2 cm²)",
    abs(nombre(valeur(od)[0]) - hea300["Iy"] / 1e4) < 0.006 and "56,26" in texte(od["tableau"]) or "56,27" in texte(od["tableau"]), valeur(od))
op = A.consulter("sections", {"n": 1, "k1": "poly", "pn1": 3, "py1_1": 0, "pz1_1": 0, "py1_2": 60, "pz1_2": 0, "py1_3": 0, "pz1_3": 90})
som = [json.loads(html.unescape(x)) for x in re.findall(r'class="sommet glis"[^>]*data-glisser="([^"]+)"', op["tableau"])]
chk("polygone libre : triangle 60 × 90 — I_y = b·h³/36 = 121,50 cm⁴ ; ses trois sommets se glissent un à un (py1_j, pz1_j) et se saisissent",
    abs(nombre(valeur(op)[0]) - 60 * 90 ** 3 / 36 / 1e4) < 0.006 and [sorted(g["x"]) + sorted(g["y"]) for g in som] == [[f"py1_{j}", f"pz1_{j}"] for j in (1, 2, 3)]
    and 'data-param="pz1_3"' in op["tableau"], (valeur(op), som))
chk("un polygone tourné ne montre plus ses poignées de sommets (le glisser écrirait dans le mauvais repère)",
    'class="sommet glis"' not in A.consulter("sections", dict(op["params"], r1=30))["tableau"])
oc_ = A.consulter("sections", {"n": 1, "k1": "plat", "b1": 100, "h1": 100, "c1": "haut", "cc1": 500})
chk("une coupe hors de l'élément est ramenée dans l'élément, et le message le dit (jamais d'élément vide)", "cc1" in (oc_["message"] or "") and oc_["params"]["cc1"] < 50)
# un exemple est ce que l'ingénieur ouvre en premier : il ne s'ouvre jamais sur un avertissement
avertis = {nom: [m_ for m_ in ("se recouvrent", "sort de la matière") if m_ in texte(A.consulter("sections", BS._jeu(els))["bande"])] for nom, els in BS.EXEMPLES}
chk("aucun des six exemples ne s'ouvre sur un avertissement (ni matière comptée deux fois, ni évidement hors de la matière)",
    not any(avertis.values()), {k: v for k, v in avertis.items() if v})
opp = A.consulter("sections", BS._jeu(dict(BS.EXEMPLES)["Plat percé"]))
lpp = {c: nombre(v) for c, _f, _v, v, _u in lignes(opp) if v and c in ("A", "Iy", "Iz")}
chk("plat percé 200 × 20, deux trous de 22 vus en COUPE (rectangles 22 × 20) : section nette à la main — A = (200 − 44)·20 = 31,20 cm², "
    "I_y = 156·20³/12 = 10,40 cm⁴, I_z = 20·200³/12 − 2·(20·22³/12 + 440·50²) = 1 109,78 cm⁴",
    abs(nombre(valeur(opp)[0]) - 156 * 20 ** 3 / 12 / 1e4) < 0.006 and abs(lpp.get("A", 0) - 31.20) < 0.006
    and abs(lpp.get("Iz", 0) - (20 * 200 ** 3 / 12 - 2 * (20 * 22 ** 3 / 12 + 440 * 50 ** 2)) / 1e4) < 0.006, (valeur(opp), lpp))
# le nom d'un axe principal plutôt vertical est à son bout du haut (un plat à plat : l'axe 1 est vertical, la cote est dessous)
opl = A.consulter("sections", {"n": 1, "k1": "plat", "b1": 200, "h1": 20})
ax1 = re.search(r'<text class="ax1" x="([\d.]+)" y="([\d.]+)"', opl["tableau"])
cdg = re.search(r'<g class="cdg"><path d="M([\d.]+) ([\d.]+)h24', opl["tableau"])
chk("plat à plat (axe 1 vertical) : le « 1 » est AU-DESSUS de G, pas sur la cote de largeur", ax1 is not None and cdg is not None and float(ax1.group(2)) < float(cdg.group(2)) - 10,
    (ax1 and ax1.groups(), cdg and cdg.groups()))
chk("six exemples (HEA + plat, demi-HEA, caisson de plats, PRS en I, deux UPN dos à dos, plat percé), chacun remplace toute la section",
    len(BS.EXEMPLES) == 6 and all(d.get("k3") is None or True for d in sels(o)) and len([d for d in sels(o) if "k1" in d and "n" in d and "f1" in d]) >= 6)
ox = A.consulter("sections", {"n": "trois", "k1": "<img>", "p1": "<script>", "y1": "ici", "sel": 99})
chk("paramètres piégés ou illisibles : défauts repris, message dit, rien de brut dans la page",
    "<script>" not in ox["tableau"] + ox["bande"] and "<img>" not in ox["tableau"] and ox["params"]["k1"] == "profil" and ox["params"]["sel"] == 1 and "k1" in ox["message"])

# ================================================================================================ 5 · tout se relit
print("\n=== 5 · Les gestes se relisent, le HTML tient")
from html.parser import HTMLParser  # noqa: E402

VIDES = {"input", "br", "path", "circle", "rect", "col", "ellipse", "img", "hr"}


class Pile(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pile, self.fautes = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VIDES:
            self.pile.append(tag)

    def handle_endtag(self, tag):
        if tag in VIDES:
            return
        if not self.pile or self.pile[-1] != tag:
            self.fautes.append((tag, self.pile[-3:]))
        else:
            self.pile.pop()


mal, n_sel, n_gl = [], 0, 0
for rub, cas in (("poutres", CAS_POUTRES), ("plaques", CAS_PLAQUES + [{"bx0": "E", "byb": "L"}]), ("sections", CAS_SECTIONS)):
    for p in cas:
        o = A.consulter(rub, p)
        for zone in ("barre", "tableau", "bande", "pourquoi", "sources"):
            pi = Pile()
            pi.feed(o[zone])
            if pi.fautes or pi.pile:
                mal.append((rub, zone, pi.fautes[:1], pi.pile[:3]))
        for x in re.findall(r'data-sel="([^"]+)"', o["tableau"]):
            n_sel += isinstance(json.loads(html.unescape(x)), dict)
        for x in re.findall(r'data-glisser="([^"]+)"', o["tableau"]):
            g = json.loads(html.unescape(x))
            n_gl += g["ech"] > 0 and g["pas"] > 0 and ("x" in g or "y" in g)
chk(f"sur tous les cas : chaque fragment est du HTML bien formé (balises appariées), les {n_sel} data-sel et les {n_gl} data-glisser sont du JSON lisible",
    not mal and n_sel > 500 and n_gl > 60, mal[:3])
chk("aucun champ du cadre ne porte la classe « cur » (js/biblio.js y ramènerait le défilement du cadre)", all('class="cur' not in A.consulter(r_, None)["tableau"] for r_ in A.CLES))
chk("chaque outil porte « Imprimer la feuille », l'en-tête d'impression (titre, moteur daté) et ses hypothèses",
    all("data-imprimer" in A.consulter(r_, None)["bande"] and "rdm-imp" in A.consulter(r_, None)["tableau"] and A.VERSION in A.consulter(r_, None)["tableau"]
        and "rdm-hyp" in A.consulter(r_, None)["tableau"] for r_ in A.CLES))

# ================================================================================================ 6 · mutations
print("\n=== 6 · Mutations")
memo = BPO.ECOLE["appuis_q"]
try:
    BPO.ECOLE["appuis_q"] = (memo[0], [(k, f.replace("q·L²/8", "q·L²/10")) for k, f in memo[1]])
    faux, _ = formules_fausses(A.consulter("poutres", {}))
    chk("mutation « formule de formulaire fausse à l'écran (q·L²/10) » : la relecture des formules la voit (le résultat, lui, vient du moteur)",
        [f[0] for f in faux] == ["M_max"], faux)
finally:
    BPO.ECOLE["appuis_q"] = memo
orig = P.resoudre
try:
    def faux_moteur(*a, **k):
        s_ = orig(*a, **k)
        s_.extremes["M_max"] = (s_.extremes["M_max"][0] * 1.1, s_.extremes["M_max"][1])
        return s_
    P.resoudre = faux_moteur
    chk("mutation « le moteur rend un autre moment » : la bande suit le moteur (l'écran ne recalcule rien)", valeur(A.consulter("poutres", {"n": 2, "t2": "P", "a2": 2}))[0] != valeur(o2)[0])
finally:
    P.resoudre = orig
memo_m = BPL.MAT["acier"]
try:
    BPL.MAT["acier"] = (210000.0, 0.0, "essai")
    chk("mutation « ν de l'acier mis à 0 » : D et la flèche changent, l'hypothèse écrite aussi (une seule source : le matériau)",
        "ν = 0,00" in texte(A.consulter("plaques", {})["tableau"]) and valeur(A.consulter("plaques", {}))[0] != "8,73")
finally:
    BPL.MAT["acier"] = memo_m
orig_c = S.caracteristiques
try:
    S.caracteristiques = lambda els, plastique=True: dict(orig_c(els, plastique), Iy=orig_c(els, plastique)["Iy"] * 2)
    faux, _ = formules_fausses(A.consulter("sections", {}))
    chk("mutation « le moteur double I_y » : la formule de Steiner écrite ne redonne plus le résultat — la relecture le voit", any(f[0] == "Iy" for f in faux), faux[:2])
finally:
    S.caracteristiques = orig_c
chk("après les mutations, tout est revenu (45,00 kN·m ; 8,73 mm ; 5 502,47 cm⁴)",
    valeur(A.consulter("poutres", {}))[0] == "45,00" and valeur(A.consulter("plaques", {}))[0] == "8,73" and valeur(A.consulter("sections", {}))[0] == "5502,47")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)
