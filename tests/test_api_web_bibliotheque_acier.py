# -*- coding: utf-8 -*-
"""La BIBLIOTHÈQUE ACIER (api_web/biblio_acier.py, lot BIB5 de docs/site/BIBLIOTHEQUES.md ; action « consulter » du
routeur, gabarit site/web/construire.page_bibliotheque), vérifiée en Python ordinaire. La suite sœur de
tests/test_api_web_bibliotheque.py (la béton armé).

1. Les trois rubriques par défaut : tout ce que la page pose, une seule case allumée, la bande sans « Inclure » ni
   « Utiliser » (une consultation de l'acier n'écrit rien dans une note), aucun script, chaque paramètre de la barre
   relu, la fiche figée en JSON pur ; aucun <label> de la barre ne contient de bouton.
2. Profilés = le catalogue (R2) : pour plusieurs réglages, les lignes du tableau (noms, ordre, cellules) sont celles
   que le module « Catalogue des profilés » écrit ; les nombres sont ceux du fichier acier/donnees/profiles.json, relu
   ici à part ; la fiche porte la section, chaque propriété de la fiche du catalogue et ses formules ; le plus léger
   qui satisfait les filtres est celui qu'on trouve en lisant le fichier.
3. Les gestes : chaque ligne porte le profilé qu'un clic ouvre ; chaque en-tête le tri ; chaque pastille de famille et
   de colonne la liste qu'un clic donne — les régler donne bien cela.
4. Les paramètres : refus doux (familles, profil, colonnes, filtres en double, bornes illisibles), « aucune » famille,
   min > max, la recherche (texte LIBRE) échappée partout et bornée, les lignes, le profilé ouvert hors des lignes.
5. Nuances (lot A, 16/11/2026) : LA table du site, acier/materiaux.py (NBN EN 1993-1-1 ANB:2018, 3.2.1(1) : option a ;
   tableaux 3.3 à 3.12 ANB) — le tableau est la table ATTENDUE écrite ici à part (désignation, f_y par palier jusqu'à
   80 mm, f_u sous 3 mm et de 3 à 100 mm, β_w, f_vw,d) ET rend exactement MA.fy / MA.fu (R2) ; la règle, la ligne de
   l'ANB de chaque nuance et la borne des modules dites ; plus aucune trace des anciennes tables ; chaque palier, la
   fiche, « S355 (fu 490) », les adresses d'avant. Boulons : les valeurs de bibliotheques.json (relu à part), F_t,Rd,
   F_v,Rd recalculés ici. Chaque case règle des paramètres qui allument une case qui les porte.
6. Le routeur et la page : « consulter » sert l'acier, la page porte le menu des trois rubriques, les identifiants
   que les scripts cherchent, la feuille ; js/biblio.js porte les gestes de l'acier (lignes, en-têtes, pastilles,
   menu « Colonnes » gardé ouvert).
7. Mutations : la page ne garde aucune copie (catalogue, table du site, resistances) ; le défaut A1 remis (f_y nominal,
   f_u de l'ancienne bibliothèque) fait rougir la table écrite et le lien R2 ; les trois copies de la table retirées en
   reprise (référence de f_u sous 3 mm écrite à la main, bornes lues dans MA.TABLEAUX, palier décidé sur un texte)
   remises une à une font rougir le lien R2 ; l'échappement retient le piège ; un paramètre fait de boutons dans un
   <label> est vu.

    python3 tests/test_api_web_bibliotheque_acier.py
"""
import html
import json
import math
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier import bibliotheques as BQ  # noqa: E402
from acier import materiaux as MA  # noqa: E402
from acier import resistances as RS  # noqa: E402
from acier.flambement import normes as NO  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402
from acier.profiles import consultation as CO  # noqa: E402
from api_web import biblio_acier as B  # noqa: E402
from api_web import biblio_beton as BB  # noqa: E402
from api_web import catalogue as CW  # noqa: E402
from api_web import routeur as R  # noqa: E402

sys.path.insert(0, os.path.join(RACINE, "site", "web"))
import construire as SW  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


EN_LIGNE = re.compile(r"<script|\son[a-z]+\s*=", re.I)
CUR = re.compile(r'<t[dh][^>]*class="[^"]*\bcur\b[^"]*"')
CASES = re.compile(r'<td([^>]*)data-sel="([^"]*)"', re.S)
VAL = re.compile(r'<b class="bib-val">(.*?)<small>(.*?)</small></b>', re.S)
LIGNE = re.compile(r'<tr([^>]*)>(.*?)</tr>', re.S)
CELLULE = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.S)
PIEGE = '<img src=x onerror=alert(1)>'
DONNEES = os.path.join(RACINE, "acier", "donnees")


ESPACES = re.compile(r"[\s   ]+")


def sans_espaces(s: str) -> str:
    """Un texte sans ses espaces (ordinaires, insécables, fines) : on compare des chiffres, pas une typographie."""
    return ESPACES.sub("", s)


def valeur(out) -> tuple:
    m = VAL.search(out["bande"])
    return (sans_espaces(html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))),
            ESPACES.sub(" ", html.unescape(re.sub(r"<[^>]+>", "", m.group(2)))).strip()) if m else (None, None)


def fragments(out) -> str:
    return "".join(out[k] for k in ("barre", "tableau", "bande", "pourquoi", "sources"))


def texte(h: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", h)).strip()


def lignes_tableau(t: str) -> list:
    """Les lignes du corps d'un tableau de profilés : [(nom, [cellules en texte])] (les marques ★ et « perso » ôtées)."""
    corps = t.split("<tbody>", 1)[1] if "<tbody>" in t else ""
    out = []
    for attrs, li in LIGNE.findall(corps):
        if "saut" in attrs:
            continue
        cs = [texte(re.sub(r'<(?:i class="mk"|span class="cat-(?:tag|k)")[^>]*>.*?</(?:i|span)>', "", c)) for c in CELLULE.findall(li)]
        out.append((cs[0], [sans_espaces(c) for c in cs[1:]]))
    return out


def sel(attr: str) -> dict:
    return json.loads(html.unescape(attr))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


# les profilés relus À PART (le fichier livré, sans le module du catalogue)
with open(os.path.join(DONNEES, "profiles.json"), encoding="utf-8") as f:
    FICHIER = {p["nom"]: {**p, **p["dims"]} for p in json.load(f)["profils"]}
with open(os.path.join(DONNEES, "bibliotheques.json"), encoding="utf-8") as f:
    BIBLIO = json.load(f)


def nombre_fr(v: float, n: int) -> str:
    """Un nombre écrit comme le catalogue l'écrit (virgule, espace fine des milliers) — écrit ici à part."""
    return f"{v:,.{n}f}".replace(",", "").replace(".", ",")


# ============================================================================================ 1. les trois rubriques
print("=== 1 · Les trois rubriques, par défaut")
CLES_REPONSE = ("titre", "barre", "tableau", "bande", "pourquoi", "sources", "fiche", "params", "message", "rubrique", "bibliotheque")
for r in B.CLES:
    for hote in B.HOTES:
        out = B.consulter(r, {}, hote=hote)
        tag = f"{r} ({hote})"
        chk(f"{tag} : la réponse porte tout ce que la page pose", tuple(sorted(out)) == tuple(sorted(CLES_REPONSE)) and out["bibliotheque"] == "acier", sorted(out))
        chk(f"{tag} : barre, tableau, bande, fiche et sources sont des fragments non vides",
            all(isinstance(out[k], str) and len(out[k]) > 40 for k in ("barre", "tableau", "bande", "pourquoi", "sources")))
        chk(f"{tag} : une seule case allumée", len(CUR.findall(out["tableau"])) == 1, len(CUR.findall(out["tableau"])))
        chk(f"{tag} : la bande ouvre la fiche (data-pourquoi), sans « Inclure » ni « Utiliser » (rien n'est écrit dans une note)",
            'data-pourquoi="1"' in out["bande"] and "data-inclure" not in out["bande"] and "data-utiliser" not in out["bande"])
        chk(f"{tag} : aucun script ni gestionnaire en ligne", not EN_LIGNE.search(fragments(out)))
        chk(f"{tag} : chaque paramètre de la barre est relu (data-param ↔ params)",
            set(re.findall(r'data-param="([^"]+)"', out["barre"])) <= set(out["params"]), set(re.findall(r'data-param="([^"]+)"', out["barre"])) - set(out["params"]))
        chk(f"{tag} : aucun <label> de la barre ne contient de bouton (un clic sur son libellé irait au premier bouton)",
            not any("<button" in x for x in re.findall(r"<label\b.*?</label>", out["barre"], re.S)))
        f = out["fiche"]
        chk(f"{tag} : la fiche figée porte titre, symbole, valeur, unité, lignes, sources, et se relit en JSON",
            all(k in f for k in ("titre", "symbole", "valeur", "unite", "params", "lignes", "sources", "moteur")) and f["lignes"] and f["sources"]
            and f["moteur"] == f"{B.TITRE} {B.VERSION}" and json.loads(json.dumps(f)) == f)
        chk(f"{tag} : aucun message par défaut", out["message"] is None, out["message"])
chk("le menu porte les trois rubriques dans l'ordre, la première allumée",
    re.findall(r'data-rubrique="([^"]+)"', B.menu()) == ["profils", "nuances", "boulons"] and 'class="bib-m on" data-rubrique="profils"' in B.menu())
chk("rubrique inconnue ou absente : « Profilés » s'ouvre, et le message le dit",
    B.consulter(None, {})["rubrique"] == "profils" and B.consulter("zzz", {})["rubrique"] == "profils" and "inconnue" in B.consulter("zzz", {})["message"])
try:
    B.consulter("profils", {}, hote="pdf")
    chk("hôte inconnu : refusé", False)
except ValueError as e:
    chk("hôte inconnu : refusé", "hôte" in str(e))

# ============================================================================================ 2. profilés = le catalogue
print("\n=== 2 · Profilés : le tableau du catalogue, ses nombres, sa fiche")
REGLAGES = [
    {},
    {"familles": "IPE,HEA,HEB", "f1": "Iy", "min1": "8000", "tri": "masse"},
    {"familles": "CHS,SHS,RHS", "recherche": "100", "courants": "1", "tri": "iz", "ordre": "decroissant", "colonnes": "masse,A,Iy,iz,Wt"},
    {"familles": "L,LI", "f1": "A", "min1": "10", "max1": "12", "f2": "Iv", "min2": "20", "lignes": "100"},
    {"familles": "HD,HL,HP", "recherche": "HD 400", "colonnes": "h,b,tw,tf,masse,It,Iw"},
]
for p in REGLAGES:
    out = B.consulter("profils", p)
    L, rech = B._lire_profils(p)
    v = CW._vue(rech)
    lb = lignes_tableau(out["tableau"])
    lc = lignes_tableau(CW.tableau(rech, v))
    noms_attendus = [q["nom"] for q in CO.liste(rech["familles"], rech["recherche"], rech["filtres"], rech["courants"], [], rech["tri"], rech["decroissant"])]
    n = min(rech["lignes"] or len(noms_attendus), len(noms_attendus))
    chk(f"profilés {p} : {len(lb)} lignes, les mêmes que le module « Catalogue des profilés » (noms, ordre, CHAQUE cellule)", lb and lb == lc, (lb[:2], lc[:2]))
    chk(f"profilés {p} : l'ordre et le nombre de lignes sont ceux de la liste du catalogue (acier/profiles/consultation)", [x[0] for x in lb][:n] == noms_attendus[:n])
    # les en-têtes : les colonnes du catalogue (cochées et non vides), la colonne de tri en tête
    th = re.findall(r'data-sel="([^"]*)"', out["tableau"].split("<tbody>")[0])
    chk(f"profilés {p} : les en-têtes sont les colonnes du catalogue (tri en tête) et chacun porte son tri",
        [sel(x)["tri"] for x in th][1:] == CW.colonnes_affichees(rech, v))
# les nombres : relus dans le FICHIER, écrits ici à part
out = B.consulter("profils", {"familles": "IPE", "profil": "IPE 300", "colonnes": "h,masse,A,Iy,Wpl_y,iz,It"})
li = dict(lignes_tableau(out["tableau"]))["IPE 300"]
f300 = FICHIER["IPE 300"]
attendu = ["IPE", nombre_fr(f300["h"], 1), nombre_fr(f300["masse"], 1), nombre_fr(f300["A"], 2), nombre_fr(f300["Iy"], 1), nombre_fr(f300["Wpl_y"], 1),
           nombre_fr(f300["iz"], 2), nombre_fr(f300["It"], 2)]
chk("IPE 300 : chaque cellule est le nombre du fichier profiles.json (relu à part), aux décimales du catalogue", li == attendu, (li, attendu))
chk("IPE 300 : la bande donne le nom, la masse du fichier, la famille et sa norme, la vignette", valeur(out) == ("IPE300", f"{nombre_fr(f300['masse'], 1)} kg/m")
    and "Poutrelles IPE" in out["bande"] and "EN 10365" in out["bande"] and 'class="bib-vig"' in out["bande"] and "<svg" in out["bande"], valeur(out))
chk("IPE 300 : la ligne ouverte est allumée (son nom porte la case) et la bande cite I_y du fichier",
    re.search(r'<tr class="on"[^>]*><th class="nm cur">IPE 300</th>', out["tableau"]) is not None and nombre_fr(f300["Iy"], 1) in sans_espaces(texte(out["bande"])))
pq = out["pourquoi"]
props = [c for _g, ps in CO.fiche(CAT.par_nom()["IPE 300"]) for c, *_ in ps]
chk(f"IPE 300 : la fiche porte la section à l'échelle (le dessin du catalogue) et ses {len(props)} propriétés, groupe par groupe",
    'id="bib_sec"' in pq and pq.count('<td class="s">') == len(props) and pq.count('<tr class="g">') == len(CO.fiche(CAT.par_nom()["IPE 300"])))
chk("IPE 300 : la fiche donne les formules du catalogue (W_el,y = I_y/max(z_sup ; z_inf), substituée)",
    "I<sub>y</sub>/max(z<sub>sup</sub> ; z<sub>inf</sub>)" in pq and "Saint-Venant" in pq)
# le plus léger qui satisfait les filtres : lu dans le FICHIER
cand = [q for q in FICHIER.values() if q["famille"] in ("IPE", "HEA", "HEB") and q.get("Iy") is not None and q["Iy"] >= 8000]
leger = min(cand, key=lambda q: q["masse"])["nom"]
o1 = B.consulter("profils", {"familles": "IPE,HEA,HEB", "f1": "Iy", "min1": "8000"})
o2 = B.consulter("profils", {"familles": "IPE,HEA,HEB", "f1": "Iy", "min1": "8000", "profil": "HEB 300"})
chk(f"I_y ≥ 8 000 cm⁴ (IPE, HEA, HEB) : le plus léger est {leger} (lu dans le fichier) ; il porte ★ dans le tableau",
    f'{leger}<i class="mk"' in o1["tableau"] and o1["tableau"].count('class="mk"') == 1)
chk(f"… HEB 300 ouvert : la bande nomme {leger} et offre de l'ouvrir (data-sel)", f"<b>{leger}</b>" in o2["bande"]
    and sel(re.search(r'<button[^>]*data-sel="([^"]*)"', o2["bande"]).group(1)) == {"profil": leger} and valeur(o2)[0] == "HEB300")
chk("… sans filtre, aucune étoile (le plus léger de tout n'a pas de sens)", 'class="mk"' not in B.consulter("profils", {})["tableau"])

# ============================================================================================ 3. les gestes
print("\n=== 3 · Les gestes : ce que porte chaque ligne, chaque en-tête, chaque pastille")
base = {"familles": "IPE,HEA,UPN,CHS", "lignes": "500"}
out = B.consulter("profils", base)
rangs = re.findall(r'<tr[^>]*data-sel="([^"]*)"', out["tableau"].split("<tbody>")[1])
rates = []
for s in rangs[:: max(1, len(rangs) // 40)]:
    d = sel(s)
    o2 = B.consulter("profils", {**out["params"], **d})
    allumee = re.findall(r'<tr class="on"[^>]*><th class="nm cur">(.*?)</th>', o2["tableau"])
    if allumee != [html.escape(d["profil"])] or valeur(o2)[0] != d["profil"].replace(" ", ""):
        rates.append((d, allumee))
chk(f"profilés : {len(rangs)} lignes cliquables ; ouvrir le profilé d'une ligne allume CETTE ligne et la bande le nomme (quarante essayées)", rangs and not rates, rates[:3])
ths = re.findall(r'<th class="n(?: on)?"[^>]*data-sel="([^"]*)"', out["tableau"].split("<tbody>")[0])
rates = []
for s in ths:
    d = sel(s)
    o2 = B.consulter("profils", {**out["params"], **d})
    on = re.findall(r'<th class="n on"[^>]*data-sel="([^"]*)"', o2["tableau"].split("<tbody>")[0])
    if [sel(x)["tri"] for x in on] != [d["tri"]] or o2["params"]["tri"] != d["tri"]:
        rates.append(d)
chk(f"profilés : chacun des {len(ths)} en-têtes trie par sa colonne (marquée) ; un second clic renverse l'ordre",
    ths and not rates and sel(re.findall(r'<th class="n on"[^>]*data-sel="([^"]*)"', B.consulter("profils", {**base, "tri": "Iy"})["tableau"])[0])["ordre"] == "decroissant", rates[:3])
puces = re.findall(r'data-valeur="([^"]*)" aria-pressed="(true|false)"[^>]*title="([^"]*)"><span class="bib-pt">([^<]*)</span>',
                   out["barre"].split('data-param="familles"')[1].split("</div>")[0])
rates = []
for v_, on, _t, f in puces:
    o2 = B.consulter("profils", {**out["params"], "familles": html.unescape(v_)})
    avant = set(out["params"]["familles"].split(","))
    apres = set(o2["params"]["familles"].split(",")) - {B.AUCUNE}
    if apres != (avant ^ {f}) - {B.AUCUNE}:
        rates.append((f, on, sorted(apres)))
chk(f"profilés : chacune des {len(puces)} pastilles de famille ajoute ou retire SA famille, rien d'autre", len(puces) == len(CO.toutes_familles()) and not rates, rates[:3])
cols = re.findall(r'data-valeur="([^"]*)" aria-pressed="(true|false)"', out["barre"].split('data-param="colonnes"')[1])
rates = []
for v_, on in cols[:len(CO.cles_colonnes())]:
    o2 = B.consulter("profils", {**out["params"], "colonnes": html.unescape(v_)})
    diff = set(out["params"]["colonnes"].split(",")) ^ set(o2["params"]["colonnes"].split(","))
    if len(diff - {B.AUCUNE}) != 1:
        rates.append((v_, on))
chk(f"profilés : chacune des {len(CO.cles_colonnes())} pastilles du menu « Colonnes » ajoute ou retire UNE colonne ; le menu reste ouvert d'une réponse à l'autre (data-garde)",
    len(cols) >= len(CO.cles_colonnes()) and not rates and 'details class="bib-plus" data-garde="colonnes"' in out["barre"], rates[:3])

# ============================================================================================ 4. les paramètres
print("\n=== 4 · Les paramètres : refus doux, texte libre échappé, cas limites")
m = B.consulter("profils", {"familles": "IPE,ZZZ", "profil": "IPE 9999", "colonnes": "h,zz", "f1": "Iy", "f2": "Iy", "min1": "abc"})
chk("famille, profilé et colonne inconnus, deux filtres sur I_y, borne illisible : chacun ignoré et dit, le reste tient",
    m["params"]["familles"] == "IPE" and m["params"]["colonnes"] == "h" and m["params"]["profil"] == "IPE 80" and m["params"]["f2"] == "" and m["params"]["min1"] is None
    and all(x in m["message"] for x in ("familles", "profil", "colonnes", "f2", "min1")), m["message"])
v0 = B.consulter("profils", {"familles": "aucune"})
chk("« aucune » famille : le tableau le dit avec sa raison, la bande « Aucun profilé », la fiche grisée, pas de case allumée",
    "bib-vide" in v0["tableau"] and "aucune famille cochée" in v0["tableau"] and "Aucun profilé" in v0["bande"] and "disabled" in v0["bande"]
    and not CUR.search(v0["tableau"]) and v0["params"]["familles"] == "aucune")
v1 = B.consulter("profils", {"f1": "Iy", "min1": "5000", "max1": "100"})
chk("min > max : le tableau vide dit « Iy (cm⁴) : min > max »", "min &gt; max" in v1["tableau"] or "min > max" in texte(v1["tableau"]))
p1 = B.consulter("profils", {"recherche": PIEGE})
chk("la recherche (texte libre) piégée : jamais brute dans un fragment, écrite échappée dans son champ",
    PIEGE not in fragments(p1) and html.escape(PIEGE) in p1["barre"] and p1["params"]["recherche"] == PIEGE)
p2 = B.consulter("profils", {"recherche": "x" * 200})
chk("une recherche de 200 caractères est coupée à 60, et le message le dit", len(p2["params"]["recherche"]) == B.RECHERCHE_MAX and "recherche" in p2["message"])
p3 = B.consulter("profils", {"recherche": "hea 3"})
chk("« hea 3 » trouve les HEA 3xx (casse et espaces ignorés, la règle du catalogue)",
    {x[0] for x in lignes_tableau(p3["tableau"])} == {n for n in FICHIER if n.startswith("HEA 3")})
p4 = B.consulter("profils", {"lignes": "100", "profil": "RHS 300x200x10"})
lb = lignes_tableau(p4["tableau"])
chk("100 lignes, le profilé ouvert plus loin : « … n lignes … » puis sa ligne, allumée", len(lb) == 101 and lb[-1][0] == "RHS 300x200x10"
    and 'class="saut"' in p4["tableau"] and "Lignes · 100 sur" in p4["barre"])
chk("les paramètres s'écrivent comme l'adresse les donne (chaînes) : même réponse qu'avec des nombres et des listes",
    B.consulter("profils", {"lignes": "500", "courants": "1", "min1": "1500", "f1": "Iy"})["tableau"]
    == B.consulter("profils", {"lignes": 500, "courants": True, "min1": 1500, "f1": "Iy"})["tableau"])
chk("« filtres = effacer » vide les trois filtres", all(not B.consulter("profils", {"f1": "Iy", "min1": "10", "filtres": "effacer"})["params"][f"f{i}"] for i in (1, 2, 3)))

# ============================================================================================ 5. nuances, boulons
print("\n=== 5 · Nuances : LA table du site (acier/materiaux.py) ; boulons : la bibliothèque des moteurs, relue à part")
# Lot A (16/11/2026) : la rubrique montrait côte à côte deux tables qui ne sont plus celles du site (le f_y « t ≤ 40 » et
# le f_u de bibliotheques.json, la colonne « 40 < t ≤ 80 » du tableau 3.1 de l'EN — l'option b). Elle montre LA table du
# site : NBN EN 1993-1-1 ANB:2018, 3.2.1(1) : option a ; tableaux 3.3 à 3.12 ANB.
REGLE = "NBN EN 1993-1-1 ANB:2018, 3.2.1(1) : option a ; tableaux 3.3 à 3.12 ANB"
# LA TABLE ATTENDUE, écrite ici à part (valeurs de contrôle des règles du lot A ; tableaux 3.3 ANB, EN 10025-2, et
# 3.4 ANB, EN 10025-3 ; f_vw,d = f_u / (√3 · β_w · 1,25) avec le f_u de 3 à 100 mm, au dixième) :
# nuance → (désignation, f_y ≤ 16, 16–40, 40–63, 63–80, f_u < 3, f_u 3–100, β_w, f_vw,d)
ATTENDU = {
    "S235": ("S235JR", "235", "225", "215", "215", "360", "360", "0,80", "207,8"),
    "S275": ("S275JR", "275", "265", "255", "245", "430", "410", "0,85", "222,8"),
    "S355": ("S355JR", "355", "345", "335", "325", "510", "470", "0,90", "241,2"),
    "S355 (fu 490)": ("S355JR", "355", "345", "335", "325", "510", "470", "0,90", "241,2"),
    "S355 N/NL": ("S355N", "355", "345", "335", "325", "470", "470", "0,90", "241,2"),
    "S460 N/NL": ("S460N", "460", "440", "430", "410", "540", "540", "1,00", "249,4"),
}
NOM_NUANCE = re.compile(r'<th class="nm"[^>]*>(.*?)<br><small>(.*?)</small></th>', re.S)


def lignes_nuances(t: str) -> list:
    """Les lignes du tableau des nuances : [(nom, désignation, [cellules en texte])]."""
    out = []
    for _attrs, li in LIGNE.findall(t.split("<tbody>", 1)[1]):
        m = NOM_NUANCE.search(li)
        cs = [sans_espaces(texte(c)) for c in re.findall(r"<td[^>]*>(.*?)</td>", li, re.S)]
        out.append((html.unescape(m.group(1)) if m else None, html.unescape(m.group(2)) if m else None, cs))
    return out


def table_ecrite(out) -> bool:
    """Le tableau rendu = LA TABLE ATTENDUE écrite ci-dessus, nuance par nuance, cellule par cellule."""
    li = lignes_nuances(out["tableau"])
    return [x[0] for x in li] == list(ATTENDU) and all((d, *c) == ATTENDU[n] for n, d, c in li)


def table_du_site(out) -> bool:
    """R2 : la rubrique rend EXACTEMENT ce que la table du site donne — MA.fy à la borne de chaque palier, MA.fu sous
    3 mm et de 3 à 100 mm, MA.ligne(…)["designation"] ; β_w de bibliotheques.json ; f_vw,d recalculé ici."""
    li = lignes_nuances(out["tableau"])
    if [x[0] for x in li] != [a["n"] for a in BIBLIO["aciers"]]:
        return False
    for (n, d, c), a in zip(li, BIBLIO["aciers"]):
        fu = MA.fu(n, 3.0)
        attendu = ([MA.ligne(n)["designation"]] + [nombre_fr(MA.fy(n, t), 0) for t in (16.0, 40.0, 63.0, 80.0)]
                   + [nombre_fr(MA.fu(n, 2.0), 0), nombre_fr(fu, 0), f"{a['bw']:.2f}".replace(".", ","),
                      nombre_fr(fu / (math.sqrt(3) * a["bw"] * 1.25), 1)])
        if [d] + c != attendu or MA.fu(n, 100.0) != fu:
            return False
    return True


n0 = B.consulter("nuances", {})
chk("nuances : par défaut S355, t ≤ 16 mm → f_y = 355 MPa, sans message", valeur(n0) == ("355", "MPa") and n0["params"] == {"nuance": "S355", "epaisseur": "16"}
    and n0["message"] is None, (valeur(n0), n0["params"]))
chk("nuances : le tableau est LA TABLE ATTENDUE (écrite ici à part) — désignation, f_y par palier (≤ 16, 16–40, 40–63, 63–80 mm), f_u sous 3 mm et "
    "de 3 à 100 mm, β_w, f_vw,d — pour les six nuances", table_ecrite(n0), lignes_nuances(n0["tableau"])[:3])
chk("nuances, R2 : la rubrique rend EXACTEMENT MA.fy / MA.fu (acier/materiaux.py, la table du site) et MA.ligne — la seule source de f_y et f_u ; "
    "β_w de bibliotheques.json (qui ne porte plus ni f_y ni f_u)", table_du_site(n0) and all(set(a) == {"n", "bw"} for a in BIBLIO["aciers"]))
# les colonnes : les paliers de l'ANB jusqu'à la borne des modules, les mêmes pour les six nuances, nommés en entier
PAL = [x for x, _ in MA.paliers("S355")][:4]
th2 = re.findall(r'<th class="k[^"]*" title="([^"]*)"><b>(.*?)</b></th>', n0["tableau"])
segs = [(k, html.unescape(t_)) for k, t_ in re.findall(r'<button type="button"(?: class="on")? data-valeur="([^"]*)">(.*?)</button>', n0["barre"].split('data-param="epaisseur"')[1])]
chk(f"nuances : les colonnes de f_y sont les paliers de l'ANB jusqu'à {NO.T_MAX_TABLEAU:g} mm (la borne des modules), les mêmes pour les six nuances, "
    "nommés en entier (infobulle de l'en-tête, boutons de la barre) ; la clé d'un palier est sa borne supérieure",
    NO.T_MAX_TABLEAU == 80.0 and PAL == ["t ≤ 16 mm", "16 < t ≤ 40 mm", "40 < t ≤ 63 mm", "63 < t ≤ 80 mm"]
    and all([x for x, _ in MA.paliers(a["n"])][:4] == PAL for a in BIBLIO["aciers"])
    and [html.unescape(t_) for t_, _ in th2[:4]] == PAL and [html.unescape(b) for _, b in th2] == ["≤ 16", "16–40", "40–63", "63–80", "< 3", "3–100"]
    and [html.unescape(t_) for t_, _ in th2[4:]] == ["t < 3 mm", "3 ≤ t ≤ 100 mm"]
    and segs == [("16", "t ≤ 16"), ("40", "16 < t ≤ 40"), ("63", "40 < t ≤ 63"), ("80", "63 < t ≤ 80")], (th2, segs))
# la règle citée, la ligne de l'ANB de chaque nuance, la borne des modules dite en une phrase ; plus aucune trace des
# deux anciennes tables (le tableau 3.1 de l'EN, le f_y et le f_u de bibliotheques.json)
src = texte(n0["sources"])
lignes_anb = {(MA.ligne(a["n"])["tableau"], MA.ligne(a["n"])["norme"], MA.ligne(a["n"])["designation"]) for a in BIBLIO["aciers"]}
PHRASE_80 = "Au-delà de 80 mm, les tableaux de l'ANB continuent ; la rubrique s'arrête à 80 mm, la borne des modules Flambement et Barres."
chk("nuances : les sources citent la règle (« " + REGLE + " »), la ligne de l'ANB de chaque nuance (tableau, norme, désignation) et disent "
    "qu'au-delà de 80 mm les tableaux continuent et que la rubrique s'arrête à la borne des modules Flambement et Barres",
    REGLE in src and REGLE in texte(n0["pourquoi"]) and lignes_anb == {("3.3 ANB", "EN 10025-2:2004", "S235JR"), ("3.3 ANB", "EN 10025-2:2004", "S275JR"),
                                                                     ("3.3 ANB", "EN 10025-2:2004", "S355JR"), ("3.4 ANB", "EN 10025-3:2004", "S355N"),
                                                                     ("3.4 ANB", "EN 10025-3:2004", "S460N")}
    and all(f"tableau {t_}, {no}" in src and d in src for t_, no, d in lignes_anb)
    and PHRASE_80 in src and "les modules du site s'arrêtent" not in src, src[:400])
# (lot A, reprise F4) la phrase dit vrai : Flambement et Barres s'arrêtent à 80 mm (normes.T_MAX_TABLEAU, leur borne de
# domaine) ; les Assemblages, les Soudures et les Garde-corps lisent la table jusqu'au bout (bibliotheques.acier : une
# tôle S355 de 100 mm y vaut 315 MPa, tableau 3.3 ANB) — « les modules du site s'arrêtent à 80 mm » était faux
try:
    NO.limite_elastique("S355", 80.5)
    flamb_80 = False
except NO.HorsDomaine:
    flamb_80 = True
chk("nuances : la borne dite est vraie — le Flambement (et les Barres, même fonction) refusent 80,5 mm, les Assemblages, les Soudures et les "
    "Garde-corps lisent la table au-delà (bibliotheques.acier S355, 100 mm → 315 MPa)",
    flamb_80 and NO.limite_elastique("S355", 80.0) == 325 and BQ.acier("S355", 100.0)["fy"] == 315.0)
VIEUX = re.compile(r"tableau 3\.1(?![\d])|(?<!< )t ≤ 40(?![\d])|40 < t ≤ 80")
vieux = [(p_, k, m.group(0)) for p_ in ({}, {"epaisseur": "40"}, {"epaisseur": "80"}, {"nuance": "S460 N/NL", "epaisseur": "63"})
         for k, f in ((k, texte(B.consulter("nuances", p_)[k])) for k in ("barre", "tableau", "bande", "pourquoi", "sources")) for m in VIEUX.finditer(f)]
chk("nuances : plus aucune trace des anciennes tables (« tableau 3.1 », « t ≤ 40 », « 40 < t ≤ 80 ») dans aucun fragment", not vieux, vieux[:4])
# chaque palier : la bande, la fiche et sa référence imprimée (MA.source) ; ε de la paroi de ce palier
rates = []
for k, t_, fy_, eps, eps3 in (("16", 16.0, "355", "0,81", "0,814"), ("40", 40.0, "345", "0,83", "0,825"), ("63", 63.0, "335", "0,84", "0,838"),
                             ("80", 80.0, "325", "0,85", "0,850")):
    o = B.consulter("nuances", {"nuance": "S355", "epaisseur": k})
    lf = {li["symbole"]: li for li in reversed(o["fiche"]["lignes"])}
    ph = texte(o["bande"].split('class="bib-phr">')[1].split("</span></div>")[0])
    if not (valeur(o) == (fy_, "MPa") and o["message"] is None and lf["f_y"]["regle"].startswith(MA.source("S355", t_))
            and f"{PAL[['16', '40', '63', '80'].index(k)]}" in ph and "S355JR, tableau 3.3 ANB" in ph and re.search(r"fu\s*=\s*470\s*MPa", ph)
            and f"ε = {eps}" in ph and lf["ε"]["texte"] == eps3):
        rates.append((k, valeur(o), ph, lf["f_y"]["regle"]))
chk("nuances S355, chaque palier : la bande donne f_y (355, 345, 335, 325), la phrase la ligne de l'ANB, le palier, f_u = 470 et ε de ce palier ; "
    "la fiche cite MA.source", not rates, rates)
# la fiche : f_u de l'épaisseur retenue (3 à 100 mm) ET f_u sous 3 mm quand il diffère ; f_vw,d avec le premier
n40 = B.consulter("nuances", {"nuance": "S355", "epaisseur": "40"})
fus = [li for li in n40["fiche"]["lignes"] if li["symbole"] == "f_u"]
fvw = 470 / (math.sqrt(3) * 0.9 * 1.25)
l_fvw = next(li for li in n40["fiche"]["lignes"] if li["symbole"] == "f_vw,d")
chk(f"nuances S355, 16 < t ≤ 40 mm : la fiche donne f_u = 470 (3 ≤ t ≤ 100 mm) et 510 (t < 3 mm), chacun avec sa référence ; f_vw,d = {nombre_fr(fvw, 1)} MPa "
    "(470 / (√3 · 0,9 · 1,25), recalculé ici) — la justification aussi",
    [(li["valeur"], li["regle"].split(" — ")[0]) for li in fus] == [(470.0, MA.source("S355", 40.0, grandeur="fu")), (510.0, "tableau 3.3 ANB (EN 10025-2:2004, S355JR), t < 3 mm")]
    and abs(l_fvw["valeur"] - fvw) < 1e-9 and l_fvw["regle"].startswith("f_u / (√3 · β_w · γ_M2) = 470 / (√3 · 0,9 · 1,25)")
    and f"{fvw:.1f}".replace(".", ",") in n40["pourquoi"], ([(li["valeur"], li["regle"]) for li in fus], l_fvw))
nn = B.consulter("nuances", {"nuance": "S355 N/NL"})
chk("nuances S355 N/NL : une seule ligne f_u dans la fiche (le tableau 3.4 ANB n'a qu'un palier jusqu'à 100 mm : 470)",
    [li["valeur"] for li in nn["fiche"]["lignes"] if li["symbole"] == "f_u"] == [470.0])
# « S355 (fu 490) » : un nom des notes enregistrées (aucun nom n'est retiré ni renommé) ; la page dit qu'il suit S355JR
n490 = B.consulter("nuances", {"nuance": "S355 (fu 490)", "epaisseur": "40"})
av490 = " ".join(texte(a) for a in n490["fiche"]["avertissements"])
chk("nuances « S355 (fu 490) » : la ligne montre f_u = 470 (pas 490) et l'avertissement dit qu'elle suit la ligne S355JR du tableau 3.3 ANB",
    lignes_nuances(n490["tableau"])[3][2][5] == "470" and "S355JR" in av490 and "tableau 3.3 ANB" in av490 and re.search(r"fu\s*=\s*470", av490)
    and "notes enregistrées" in av490 and "S355JR" in texte(n490["bande"]), av490)
chk("nuances S235 : aucun avertissement (un nom de la table n'en appelle pas)", not B.consulter("nuances", {"nuance": "S235"})["fiche"]["avertissements"])


# (lot A, reprise F5) « S355 N/NL » et « S460 N/NL » sont des désignations réelles de l'EN 10025-3 (sous-nuances N et
# NL, mêmes R_eH et R_m) : une simple mention « lu sur la ligne S355N / S460N du tableau 3.4 ANB », sans triangle ; seul
# « S355 (fu 490) », qui porte dans son nom une valeur qui n'est pas la sienne, garde l'avertissement
def alias_sans_triangle() -> list:
    out = []
    for nom, des in (("S355 N/NL", "S355N"), ("S460 N/NL", "S460N")):
        o = B.consulter("nuances", {"nuance": nom, "epaisseur": "40"})
        ph = texte(o["bande"].split('class="bib-phr">')[1].split("</span></div>")[0])
        if o["fiche"]["avertissements"] or 'class="bib-av"' in o["bande"]:
            out.append(f"{nom} : avertissement (triangle) — {[texte(a) for a in o['fiche']['avertissements']]}")
        if f"lu sur la ligne {des} du tableau 3.4 ANB" not in ph or f"lu sur la ligne {des} du tableau 3.4 ANB" not in texte(o["fiche"]["phrase"]):
            out.append(f"{nom} : la mention « lu sur la ligne {des} du tableau 3.4 ANB » manque à la bande ou à la fiche — « {ph} »")
    o = B.consulter("nuances", {"nuance": "S355 (fu 490)"})
    if len(o["fiche"]["avertissements"]) != 1 or 'class="bib-av"' not in o["bande"]:
        out.append("« S355 (fu 490) » : l'avertissement manque")
    return out


ec_alias = alias_sans_triangle()
chk("nuances « S355 N/NL », « S460 N/NL » : pas d'avertissement, la mention « lu sur la ligne S355N / S460N du tableau 3.4 ANB » dans la bande et "
    "la fiche ; « S355 (fu 490) » seul garde son avertissement", not ec_alias, ec_alias)
# une adresse d'avant le lot A (« epaisseur=40 » : t ≤ 40 ; « 80 » : 40 < t ≤ 80) ouvre le palier qui finit là — le plus
# défavorable de l'ancienne colonne ; une épaisseur au-delà de la borne des modules n'est pas offerte (refus doux)
o80 = B.consulter("nuances", {"nuance": "S275", "epaisseur": "80"})
o100 = B.consulter("nuances", {"nuance": "S275", "epaisseur": "100"})
chk("nuances : « epaisseur=80 » (adresse d'avant) → 63 < t ≤ 80 mm, S275 → 245 MPa (et non plus 255 du tableau 3.1) ; « 100 » est refusé "
    "doucement (message, t ≤ 16 mm)", valeur(o80) == ("245", "MPa") and o80["message"] is None
    and valeur(o100) == ("275", "MPa") and o100["params"]["epaisseur"] == "16" and "epaisseur" in (o100["message"] or ""), (valeur(o80), valeur(o100), o100["message"]))
# un « < » de texte (« 16 < t ≤ 40 mm ») entre ÉCHAPPÉ dans le HTML : un « < » nu n'ouvre pas de balise dans le
# navigateur, mais casse toute relecture du fragment (un texte « 16 < t … > » s'y perdrait)
NU = re.compile(r"<(?![A-Za-z/!])")
nus = [(r, k, f[max(0, m.start() - 20):m.start() + 20]) for r, p_ in (("nuances", {"epaisseur": "80"}), ("nuances", {}), ("boulons", {}), ("boulons", {"vue": "classes"}),
                                                                     ("profils", {}), ("profils", {"f1": "Iy", "min1": "5", "max1": "1"}))
       for k, f in ((k, B.consulter(r, p_)[k]) for k in ("barre", "tableau", "bande", "pourquoi", "sources")) for m in NU.finditer(f)]
chk("aucun « < » nu dans les fragments des trois rubriques (« 16 < t ≤ 40 mm » est échappé : barre, bande, tableau)", not nus, nus[:4])
chk("nuances : la justification de S355 (t ≤ 16 mm) donne f_vw,d = 241,2 MPa et ε = 0,814", "241,2" in n0["pourquoi"] and "0,814" in n0["pourquoi"])


# (lot A, reprise F6 — R2) la rubrique ne recompose rien de la table du site : la référence de f_u sous 3 mm EST
# MA.source, les bornes des paliers viennent de MA.bornes (jamais de la structure interne MA.TABLEAUX), et le palier
# de f_u retenu (la ligne « f_u sous 3 mm » de la fiche, la colonne allumée) se décide sur les BORNES, jamais en
# comparant le texte d'une référence. Preuve : sous une table qui imprime ses références autrement et ne montre pas sa
# structure interne, la rubrique suit — mêmes lignes, mêmes colonnes allumées, références imprimées par la table.
def rubrique_suit_la_table() -> list:
    """Les écarts de la rubrique sous une table du site qui imprime ses références entre crochets et sans TABLEAUX."""
    vrai = B.MA
    faux = types.ModuleType("materiaux_autre_format")
    faux.__dict__.update({k: v for k, v in vrai.__dict__.items() if k != "TABLEAUX"})
    faux.source = lambda *a, **k: "[" + vrai.source(*a, **k) + "]"
    B.MA = faux
    out = []
    try:
        for nom, n_fu in (("S355", 2), ("S235", 2), ("S355 N/NL", 1), ("S460 N/NL", 1)):
            o = B.consulter("nuances", {"nuance": nom, "epaisseur": "40"})
            fus = [li for li in o["fiche"]["lignes"] if li["symbole"] == "f_u"]
            if len(fus) != n_fu:
                out.append(f"{nom} : {len(fus)} ligne(s) f_u dans la fiche au lieu de {n_fu}")
            elif n_fu == 2 and fus[1]["regle"] != faux.source(nom, 1.5, "lamine", "fu") + " — acier/materiaux.py":
                out.append(f"{nom} : référence de f_u sous 3 mm « {fus[1]['regle']} » — pas celle de MA.source")
        on = [html.unescape(x) for x in re.findall(r'<th class="k on" title="([^"]*)"', B.consulter("nuances", {"epaisseur": "40"})["tableau"])]
        if on != ["16 < t ≤ 40 mm", "3 ≤ t ≤ 100 mm"]:
            out.append(f"colonnes allumées : {on}")
    except Exception as err:  # noqa: BLE001 — une rubrique qui lit la structure interne casse ici
        out.append(f"{type(err).__name__} : {err}")
    finally:
        B.MA = vrai
    return out


ec_table = rubrique_suit_la_table()
chk("nuances, R2 : sous une table qui imprime autrement et ne montre pas sa structure interne, la rubrique suit — f_u sous 3 mm cité par "
    "MA.source, bornes des paliers par MA.bornes, ligne « f_u sous 3 mm » et colonne allumée décidées sur les bornes (pas sur un texte)",
    not ec_table, ec_table)
b0 = B.consulter("boulons", {})
m16 = next(x for x in BIBLIO["boulons"] if x["n"] == "M16")
c88 = next(x for x in BIBLIO["classes"] if x["n"] == "8.8")
fv = c88["av"] * c88["fub"] * m16["As"] / 1.25 / 1000
ft = 0.9 * c88["fub"] * m16["As"] / 1.25 / 1000
chk(f"boulons : M16 8.8 → F_v,Rd = {fv:.1f} kN (α_v · f_ub · A_s / γ_M2, recalculé ici) ; F_t,Rd = {ft:.1f} kN dans la phrase",
    valeur(b0) == (f"{fv:.1f}".replace(".", ","), "kN") and f"{ft:.1f}".replace(".", ",") in texte(b0["bande"]))
b1 = B.consulter("boulons", {"boulon": "M24", "classe": "10.9", "vue": "classes"})
chk("boulons : vue « classes » → f_ub de la classe (1 000 MPa pour 10.9), α_v = 0,5 dans le tableau", valeur(b1) == ("1000", "MPa") and "0,5" in b1["tableau"])
chk("boulons : chaque boulon du fichier est une ligne, avec d_0 et A_s du fichier",
    [x[0] for x in lignes_tableau(b0["tableau"])] == [x["n"] for x in BIBLIO["boulons"]]
    and all(c[1] == str(x["d0"]) and c[3] == f"{x['As']:.1f}".replace(".", ",") for (_, c), x in zip(lignes_tableau(b0["tableau"]), BIBLIO["boulons"])))
for r in ("nuances", "boulons"):
    for vue in ({}, {"vue": "classes"}) if r == "boulons" else ({}, {"epaisseur": "80"}):
        out = B.consulter(r, vue)
        cases = CASES.findall(out["tableau"])
        rates = []
        for _a, s in cases:
            d = sel(s)
            o2 = B.consulter(r, {**out["params"], **d})
            cur = [sel(x) for a_, x in CASES.findall(o2["tableau"]) if re.search(r'class="[^"]*\bcur\b', a_)]
            if len(cur) != 1 or not all(cur[0].get(k) == v_ for k, v_ in d.items()):
                rates.append((d, cur))
        chk(f"{r} {vue} : {len(cases)} cases ; les paramètres d'une case allument une case qui les porte (toutes essayées)", cases and not rates, rates[:3])

# ============================================================================================ 6. routeur, page, scripts
print("\n=== 6 · Le routeur, la page bibliotheque-acier, les scripts")
rr = req(action="consulter", bibliotheque="acier", rubrique="boulons", params={"boulon": "M20"})
chk("routeur : « consulter » sert la Bibliothèque Acier (ok, fragments, paramètres relus)", rr["ok"] and rr["bibliotheque"] == "acier" and rr["params"]["boulon"] == "M20")
rr = req(action="consulter", bibliotheque="acier", rubrique="profils", hote="note")
chk("routeur : l'hôte « note » passe (la bande reste sans « Utiliser »)", rr["ok"] and "data-utiliser" not in rr["bande"])
chk("routeur : les bibliothèques du routeur sont celles du générateur (une par famille déclarée)",
    set(R.BIBLIOTHEQUES) == set(SW.BIBLIOTHEQUES) and all(R.BIBLIOTHEQUES[k] == SW.BIBLIOTHEQUES[k][0] for k in R.BIBLIOTHEQUES))
page = SW.page_bibliotheque("acier")
js = "".join(open(os.path.join(RACINE, "site", "web", "js", f), encoding="utf-8").read() for f in ("bibliotheque.js", "biblio.js"))
ids = sorted(set(re.findall(r'\$\("#([\w-]+)', js)) | set(re.findall(r'getElementById\("([\w-]+)"', js)))
chk(f"bibliotheque-acier.html : les {len(ids)} identifiants que js/bibliotheque.js et js/biblio.js cherchent existent, une fois chacun",
    all(page.count(f'id="{i}"') == 1 for i in ids), [i for i in ids if page.count(f'id="{i}"') != 1])
# (S4 : le seul script de module de la page est js/app.js, qui monte js/bibliotheque.js — body data-page="bibliotheque" ;
# le fichier construit porte la VERSION du site, recalculée ici comme le fait test_site_web)
chk("la page porte le menu des trois rubriques (RUBRIQUES, une source), data-bibliotheque, la feuille et le script (js/app.js)",
    B.menu() in page and 'data-bibliotheque="acier"' in page and 'data-page="bibliotheque"' in page and 'href="css/bibliotheque.css"' in page
    and 'src="js/app.js"' in page and "bibliotheque.js" not in page and "etabli.js" not in page
    and 'import { ouvrirBibliotheque } from "./bibliotheque.js";' in open(os.path.join(RACINE, "site", "web", "js", "app.js"), encoding="utf-8").read())
chk("le générateur écrit la page et le fichier construit est le sien",
    SW.PAGES_BIBLIOTHEQUES.get("acier") == "bibliotheque-acier.html"
    and open(os.path.join(RACINE, "site", "web", "bibliotheque-acier.html"), encoding="utf-8").read()
    == SW.poser_version(page, SW.version_site(os.path.join(RACINE, "site", "web"))))
bjs = open(os.path.join(RACINE, "site", "web", "js", "biblio.js"), encoding="utf-8").read()
chk("js/biblio.js : une ligne (tr[data-sel]), un en-tête (th[data-sel]), un bouton de la bande règlent ; une pastille règle le paramètre "
    "du bloc qui la contient ; un menu « data-garde » ouvert le reste ; le tableau garde son défilement dans la même rubrique",
    # (11/11/2026, le test SUIT : le câblage ne nomme plus la case, la ligne et l'en-tête un à un — TOUT élément du cadre
    #  qui porte data-sel règle, un élément d'un dessin compris (Bibliothèque Autres). Le geste lui-même — la ligne
    #  cliquée, l'en-tête qui trie — est joué dans le vrai navigateur : verifier_site.js, filtre « bibliotheque ».)
    "#bib-tableau [data-sel], #bib-bande [data-sel]" in bjs
    and '.bib-barre [data-param] button[data-valeur]' in bjs and 'closest("[data-param]")' in bjs and "details[data-garde][open]" in bjs
    and "affichee" in bjs)

# ============================================================================================ 7. mutations
print("\n=== 7 · Mutations : aucune copie ; l'échappement ; le <label>")
garde = CAT.profils
try:
    autre = [dict(q) for q in garde()]
    next(q for q in autre if q["nom"] == "IPE 300")["Iy"] = 9999.0
    CAT.profils = lambda: autre
    o = B.consulter("profils", {"familles": "IPE", "profil": "IPE 300", "colonnes": "Iy"})
    chk("mutation « I_y de l'IPE 300 = 9 999 dans le catalogue chargé » : le tableau ET la bande suivent (la page ne garde aucune copie) — "
        "et la comparaison au fichier, plus haut, rougirait", dict(lignes_tableau(o["tableau"]))["IPE 300"][1] == "9999,0" and "9999,0" in sans_espaces(texte(o["bande"])))
finally:
    CAT.profils = garde
garde = CO.liste
try:
    CO.liste = lambda *a, **k: garde(*a, **k)[1:]
    chk("mutation « la liste du catalogue perd sa première ligne » : la bibliothèque la perd aussi (elle lit la liste du catalogue)",
        lignes_tableau(B.consulter("profils", {"familles": "IPE"})["tableau"])[0][0] == "IPE 100")
finally:
    CO.liste = garde
# (lot A) la table du site change : la rubrique la suit (elle n'en garde aucune copie) — et la table écrite plus haut
# rougirait
garde = MA.TABLEAUX["3.3"]
try:
    norme, lignes = garde
    reh, rm = lignes["S355JR"]
    MA.TABLEAUX["3.3"] = (norme, {**lignes, "S355JR": (((16.0, True, 350.0),) + tuple(reh[1:]), rm)})
    o = B.consulter("nuances", {})
    chk("mutation « la table du site : S355JR, t ≤ 16 mm = 350 » : la bande et le tableau suivent (la rubrique ne garde aucune copie) — et la "
        "table écrite ici rougirait", valeur(o)[0] == "350" and table_du_site(o) and not table_ecrite(o))
finally:
    MA.TABLEAUX["3.3"] = garde
# (lot A, le défaut A1 remis) la rubrique revient à des valeurs sans épaisseur — le f_y nominal pour tout palier, le f_u
# de l'ancienne bibliothèque (tableau 3.1 de l'EN : S355 510) : la table écrite ici ET le lien R2 avec la table du site
# rougissent, f_vw,d aussi
vrai = B.MA
try:
    ANCIEN_FU = {"S235": 360.0, "S275": 430.0, "S355": 510.0, "S355 (fu 490)": 490.0, "S355 N/NL": 490.0, "S460 N/NL": 540.0}
    faux = types.ModuleType("materiaux_sans_epaisseur")
    faux.__dict__.update(vrai.__dict__)
    faux.fy = lambda nom, t, produit="lamine": vrai.nominale(nom)
    faux.fu = lambda nom, t, produit="lamine": ANCIEN_FU[nom]
    B.MA = faux
    o = B.consulter("nuances", {"epaisseur": "40"})
    chk("mutation « le défaut A1 remis : f_y nominal quel que soit le palier, f_u de l'ancienne bibliothèque (S355 510) » : la table écrite ici ET "
        "le lien R2 avec la table du site rougissent ; f_vw,d redevient 261,7", valeur(o)[0] == "355" and not table_ecrite(o) and not table_du_site(o)
        and "261,7" in o["tableau"])
finally:
    B.MA = vrai
# (lot A, reprise F4) la phrase fausse remise : le contrôle des sources rougit
vraie_phrase = B.PHRASE_BORNE
try:
    B.PHRASE_BORNE = "Au-delà de {t} mm, les tableaux de l'ANB continuent, mais les modules du site s'arrêtent à {t} mm."
    src_m = texte(B.consulter("nuances", {})["sources"])
    chk("mutation « les modules du site s'arrêtent à 80 mm » (la phrase d'avant, fausse pour les Assemblages, les Soudures et les Garde-corps) : "
        "le contrôle des sources rougit", PHRASE_80 not in src_m and "les modules du site s'arrêtent" in src_m)
finally:
    B.PHRASE_BORNE = vraie_phrase
# (lot A, reprise F5) l'avertissement rallumé pour tout autre nom (le code d'avant : « nuance in MA.ALIAS ») : rouge
vrai_contredit = B._nom_contredit
try:
    B._nom_contredit = lambda nom, lg: nom in MA.ALIAS
    ec = alias_sans_triangle()
    chk("mutation « l'avertissement des noms des notes allumé pour S355 N/NL et S460 N/NL » : le contrôle des alias rougit", len(ec) >= 2, ec[:2])
finally:
    B._nom_contredit = vrai_contredit
# (lot A, reprise F6) les trois copies retirées, remises une à une : le lien R2 ci-dessus rougit à chaque fois
vrai_lire, vrai_paliers = B._lire_nuance, B._paliers


def lire_ref_a_la_main(nom, t, fy_cles, fu_t):
    """La référence de f_u sous 3 mm composée à la main, au format de MA.source recopié (le code d'avant)."""
    x = vrai_lire(nom, t, fy_cles, fu_t)
    if x is not None:
        lg = x["ligne"]
        x["src_fu_mince"] = f"tableau {lg['tableau']} ({lg['norme']}, {lg['designation']}), {x['fu_mince'][0]}"
    return x


def paliers_par_tableaux(nom, grandeur):
    """Les bornes lues dans la structure interne MA.TABLEAUX (le code d'avant)."""
    lg = B.MA.ligne(nom, B.PRODUIT)
    bandes = B.MA.TABLEAUX[lg["tableau"].removesuffix(" ANB")][1][lg["designation"]][0 if grandeur == "fy" else 1]
    return [(sup, inc, txt, v) for (sup, inc, _), (txt, v) in zip(bandes, B.MA.paliers(nom, B.PRODUIT, grandeur))]


def lire_palier_par_le_texte(nom, t, fy_cles, fu_t):
    """Le palier de f_u retenu décidé en comparant le texte de la référence imprimée (endswith, le code d'avant)."""
    x = vrai_lire(nom, t, fy_cles, fu_t)
    if x is not None:
        x["fu_palier"] = 0 if x["src_fu"].endswith(", " + x["fu_mince"][0]) else 1
    return x


for nom_mut, attr, mutant in (("la référence de f_u sous 3 mm composée à la main (format de MA.source recopié)", "_lire_nuance", lire_ref_a_la_main),
                              ("les bornes des paliers lues dans MA.TABLEAUX (la structure interne de la table)", "_paliers", paliers_par_tableaux),
                              ("le palier de f_u retenu décidé sur le texte de la référence (endswith)", "_lire_nuance", lire_palier_par_le_texte)):
    setattr(B, attr, mutant)
    try:
        ec = rubrique_suit_la_table()
        chk(f"mutation « {nom_mut} » : le lien R2 avec la table rougit, et le rendu ordinaire reste celui d'avant (la copie était invisible)",
            ec and B.consulter("nuances", {"epaisseur": "40"}) == n40, ec[:2])
    finally:
        B._lire_nuance, B._paliers = vrai_lire, vrai_paliers
garde = RS.Fv_Rd
try:
    RS.Fv_Rd = lambda a_v, f_ub, A, g: 2 * garde(a_v, f_ub, A, g)
    chk("mutation « F_v,Rd doublé dans acier/resistances.py » : la bande suit (la bibliothèque appelle le moteur)", valeur(B.consulter("boulons", {}))[0] == "120,6")
finally:
    RS.Fv_Rd = garde
garde = B.C.esc
try:
    B.C.esc = lambda s: str(s)
    chk("mutation « sans échappement » : le piège de la recherche entre dans la barre — c'est l'échappement qui le retient (le contrôle plus haut rougirait)",
        PIEGE in B.consulter("profils", {"recherche": PIEGE})["barre"])
finally:
    B.C.esc = garde
garde = B._bloc
try:
    B._bloc = lambda libelle, controle, aide="": BB._champ("", libelle, controle, aide)
    o = B.consulter("profils", {"tri": "Iy"})
    chk("mutation « les pastilles et les segments dans un <label> » : le contrôle des <label> le voit",
        any("<button" in x for x in re.findall(r"<label\b.*?</label>", o["barre"], re.S)))
finally:
    B._bloc = garde

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
