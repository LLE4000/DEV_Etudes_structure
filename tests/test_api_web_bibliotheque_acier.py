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
5. Nuances et boulons : les valeurs de acier/donnees/bibliotheques.json (relu à part), f_y au-delà de 40 mm et ε par
   le moteur du flambement, f_vw,d, F_t,Rd, F_v,Rd recalculés ici ; les deux copies de f_y (bibliotheques.json et
   acier/flambement/normes.py) NOMMÉES et liées (R2) ; chaque case règle des paramètres qui allument une case qui les
   porte.
6. Le routeur et la page : « consulter » sert l'acier, la page porte le menu des trois rubriques, les identifiants
   que les scripts cherchent, la feuille ; js/biblio.js porte les gestes de l'acier (lignes, en-têtes, pastilles,
   menu « Colonnes » gardé ouvert).
7. Mutations : la page ne garde aucune copie (catalogue, bibliotheques.json, resistances) ; l'échappement retient le
   piège ; un paramètre fait de boutons dans un <label> est vu.

    python3 tests/test_api_web_bibliotheque_acier.py
"""
import html
import json
import math
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier import bibliotheques as BQ  # noqa: E402
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
print("\n=== 5 · Nuances et boulons : la bibliothèque des moteurs, relue à part")
n0 = B.consulter("nuances", {})
s355 = next(a for a in BIBLIO["aciers"] if a["n"] == "S355")
chk("nuances : S355 par défaut → f_y = 355 MPa (bibliotheques.json, relu à part)", valeur(n0) == (str(s355["fy"]), "MPa"))
chk("nuances : chaque ligne du tableau est un acier du fichier, avec f_y, f_u, β_w du fichier",
    [x[0] for x in lignes_tableau(n0["tableau"])] == [a["n"] for a in BIBLIO["aciers"]]
    and all(c[0] == str(a["fy"]) and c[2] == str(a["fu"]) and c[3] == f"{a['bw']:.2f}".replace(".", ",") for (_, c), a in zip(lignes_tableau(n0["tableau"]), BIBLIO["aciers"])))
n1 = B.consulter("nuances", {"nuance": "S275", "epaisseur": "80"})
chk("nuances : S275, 40 < t ≤ 80 mm → 255 MPa (le moteur du flambement, tableau 3.1) et l'avertissement sur f_u",
    valeur(n1) == ("255", "MPa") and valeur(n1)[0] == str(int(NO.limite_elastique("S275", 60.0))) and "f<sub>u</sub> n'est connu" in n1["bande"])
# (relecture BIB5) à 40 < t ≤ 80 mm, f_u n'est connu du dépôt que pour t ≤ 40 mm, et f_vw,d qu'on en tire vaut pour
# une paroi de t ≤ 40 mm (le tableau 3.1 donne un f_u plus faible au-delà : S355 510 → 470 MPa, f_vw,d +8,5 %) : rien ne
# doit se lire comme valable pour la paroi épaisse — l'en-tête de f_u, la phrase de la bande, la ligne f_vw,d de la
# justification et de la fiche, l'avertissement disent leur domaine
TOUT_40 = re.compile(r"t\s*≤\s*40")
n80 = B.consulter("nuances", {"nuance": "S355", "epaisseur": "80"})
th_fu = [texte(x) for x in re.findall(r"<th[^>]*>.*?</th>", n80["tableau"].split("</thead>")[0], re.S) if texte(x).startswith("fu")]
ph80 = texte(n80["bande"].split('class="bib-phr">')[1].split("</span></div>")[0])
l_fvw = next(li for li in n80["fiche"]["lignes"] if li["symbole"] == "f_vw,d")
pq_fvw = next(texte(r) for r in re.findall(r"<tr[^>]*>(.*?)</tr>", n80["pourquoi"], re.S) if "vw,d" in r)
chk("nuances 40 < t ≤ 80 mm : f_u et f_vw,d disent leur domaine (t ≤ 40 mm) — en-tête de f_u, phrase de la bande, ligne f_vw,d de la "
    "justification et de la fiche, avertissement",
    th_fu and all(TOUT_40.search(x) for x in th_fu) and re.search(r"fu\s*\(t ≤ 40 mm\)\s*=\s*510", ph80) is not None
    and TOUT_40.search(l_fvw["nom"]) and TOUT_40.search(pq_fvw) and any("vw,d" in a for a in n80["fiche"]["avertissements"]),
    (th_fu, ph80, l_fvw["nom"], pq_fvw[:120], n80["fiche"]["avertissements"]))
chk("nuances t ≤ 40 mm : la ligne f_vw,d n'est pas alourdie (le domaine est celui de la page)",
    not TOUT_40.search(next(li for li in n0["fiche"]["lignes"] if li["symbole"] == "f_vw,d")["nom"]))
# un « < » de texte (« 40 < t ≤ 80 mm ») entre ÉCHAPPÉ dans le HTML : un « < » nu n'ouvre pas de balise dans le
# navigateur, mais casse toute relecture du fragment (un texte « 40 < t … > » s'y perdrait)
NU = re.compile(r"<(?![A-Za-z/!])")
nus = [(r, k, f[max(0, m.start() - 20):m.start() + 20]) for r, p_ in (("nuances", {"epaisseur": "80"}), ("nuances", {}), ("boulons", {}), ("boulons", {"vue": "classes"}),
                                                                     ("profils", {}), ("profils", {"f1": "Iy", "min1": "5", "max1": "1"}))
       for k, f in ((k, B.consulter(r, p_)[k]) for k in ("barre", "tableau", "bande", "pourquoi", "sources")) for m in NU.finditer(f)]
chk("aucun « < » nu dans les fragments des trois rubriques (« 40 < t ≤ 80 mm » est échappé : barre, bande, tableau)", not nus, nus[:4])
fvw = s355["fu"] / (math.sqrt(3) * s355["bw"] * 1.25)
chk(f"nuances : la justification de S355 donne f_vw,d = {fvw:.1f} MPa (f_u / (√3 · β_w · 1,25), recalculé ici) et ε = 0,81",
    f"{fvw:.1f}".replace(".", ",") in n0["pourquoi"] and "0,814" in n0["pourquoi"])
liens = [(a["n"], a["fy"], NO.NUANCES.get(B._base(a["n"]), (None,))[0]) for a in BQ.DB["aciers"]]
chk("R2 — deux copies NOMMÉES de f_y (t ≤ 40 mm) : acier/donnees/bibliotheques.json (assemblages, soudures, garde-corps) et "
    "acier/flambement/normes.NUANCES (flambement) disent la même chose pour chaque nuance", all(b_ is not None and float(a_) == b_ for _, a_, b_ in liens), liens)
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
chk("la page porte le menu des trois rubriques (RUBRIQUES, une source), data-bibliotheque, la feuille et le script",
    B.menu() in page and 'data-bibliotheque="acier"' in page and 'href="css/bibliotheque.css"' in page and 'src="js/bibliotheque.js"' in page and "etabli.js" not in page)
chk("le générateur écrit la page et le fichier construit est le sien",
    SW.PAGES_BIBLIOTHEQUES.get("acier") == "bibliotheque-acier.html"
    and open(os.path.join(RACINE, "site", "web", "bibliotheque-acier.html"), encoding="utf-8").read() == page)
bjs = open(os.path.join(RACINE, "site", "web", "js", "biblio.js"), encoding="utf-8").read()
chk("js/biblio.js : une ligne (tr[data-sel]), un en-tête (th[data-sel]), un bouton de la bande règlent ; une pastille règle le paramètre "
    "du bloc qui la contient ; un menu « data-garde » ouvert le reste ; le tableau garde son défilement dans la même rubrique",
    "#bib-tableau tr[data-sel]" in bjs and "#bib-tableau th[data-sel]" in bjs and "#bib-bande [data-sel]" in bjs
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
garde = BQ.DB["aciers"]
try:
    BQ.DB["aciers"] = [{**a, "fy": 350} if a["n"] == "S355" else a for a in garde]
    chk("mutation « S355 : f_y = 350 dans bibliotheques.json » : la bande suit — et le lien R2 avec normes.NUANCES rougirait",
        valeur(B.consulter("nuances", {}))[0] == "350" and not all(float(a["fy"]) == NO.NUANCES[B._base(a["n"])][0] for a in BQ.DB["aciers"]))
finally:
    BQ.DB["aciers"] = garde
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
