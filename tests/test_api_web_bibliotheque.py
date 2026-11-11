# -*- coding: utf-8 -*-
"""La page BIBLIOTHÈQUE BÉTON ARMÉ (api_web/biblio_beton.py, action « consulter » du routeur, gabarit
site/web/construire.page_bibliotheque), vérifiée en Python ordinaire : chaque valeur de la bande est celle du MOTEUR
(beton/bibliotheque, beton/donnees, beton/armatures), lue ici indépendamment de la page.

1. Les six rubriques se rendent avec leurs défauts : barre, tableau (une seule case allumée), bande (« Pourquoi ? »),
   justification, sources, fiche figée, paramètres relus ; aucun script ni « on…= » ; l'hôte « note » ajoute
   « Inclure dans la note » et « Utiliser cette valeur », la page non.
2. Les valeurs : l'exemple de la NIT 260 (dalle XC1 C30/37 Ø 12 → 22 mm), l'ancrage Ø 16 C30/37 (580 mm), B500B
   (f_yd 434,8), C30/37 (f_cd 20,0), 4Ø12 (452 mm²) ; les réductions de l'ancrage changent la bande, pas le tableau.
3. Les cases : chaque case porte les paramètres qu'un clic règle (JSON) ; les régler allume CETTE case — sur toutes
   les cases de chaque rubrique.
4. Les paramètres : écrits comme l'adresse les donne (des chaînes) ; un paramètre illisible reprend son défaut et le
   message le dit ; un texte piégé est échappé ; une rubrique inconnue ouvre la première ; le routeur répond à
   « consulter » et refuse une bibliothèque inconnue sans exception.
5. La page : le gabarit porte le menu des six rubriques, les identifiants que js/bibliotheque.js cherche, la feuille
   css/bibliotheque.css ; le menu vient de RUBRIQUES (une source).
6. Mutations : abîmer un moteur change la bande (la page ne garde aucune copie) ; retirer l'échappement fait rougir.

    python3 tests/test_api_web_bibliotheque.py
"""
import html
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import biblio_beton as B  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.bibliotheque import ancrages as AN  # noqa: E402
from beton.bibliotheque import enrobage as EN  # noqa: E402
from beton.donnees import ec2 as EC  # noqa: E402

sys.path.insert(0, os.path.join(RACINE, "site", "web"))
import construire as SW  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


EN_LIGNE = re.compile(r"<script|\son[a-z]+\s*=", re.I)
CASES = re.compile(r'<td([^>]*)data-sel="([^"]*)"', re.S)
CUR = re.compile(r'<td[^>]*class="[^"]*\bcur\b[^"]*"')
VAL = re.compile(r'<b class="bib-val">(.*?)<small>(.*?)</small></b>', re.S)
PIEGE = '<img src=x onerror=alert(1)>'


def valeur(out) -> tuple:
    m = VAL.search(out["bande"])
    return (m.group(1).replace(" ", "").replace(" ", ""), m.group(2)) if m else (None, None)


def fragments(out) -> str:
    return "".join(out[k] for k in ("barre", "tableau", "bande", "pourquoi", "sources"))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


# ============================================================================================ 1. les six rubriques
print("=== 1 · Les six rubriques, par défaut")
CLES_REPONSE = ("titre", "barre", "tableau", "bande", "pourquoi", "sources", "fiche", "params", "message", "rubrique", "bibliotheque")
for r in B.CLES:
    out = B.consulter(r, {})
    chk(f"{r} : la réponse porte tout ce que la page pose", tuple(sorted(out)) == tuple(sorted(CLES_REPONSE)), sorted(out))
    chk(f"{r} : barre, tableau, bande, justification et sources sont des fragments non vides",
        all(isinstance(out[k], str) and len(out[k]) > 40 for k in ("barre", "tableau", "bande", "pourquoi", "sources")))
    chk(f"{r} : une seule case allumée dans le tableau", len(CUR.findall(out["tableau"])) == 1, len(CUR.findall(out["tableau"])))
    chk(f"{r} : la bande offre « Pourquoi ? » et rien d'autre pour la page", 'data-pourquoi="1"' in out["bande"] and "data-inclure" not in out["bande"] and "data-utiliser" not in out["bande"])
    chk(f"{r} : aucun script ni gestionnaire en ligne", not EN_LIGNE.search(fragments(out)))
    chk(f"{r} : chaque paramètre de la barre est relu (data-param ↔ params)",
        set(re.findall(r'data-param="([^"]+)"', out["barre"])) <= set(out["params"]), set(re.findall(r'data-param="([^"]+)"', out["barre"])) - set(out["params"]))
    f = out["fiche"]
    chk(f"{r} : la fiche figée porte titre, symbole, valeur, unité, paramètres, lignes, sources",
        all(k in f for k in ("titre", "symbole", "valeur", "unite", "params", "lignes", "sources", "avertissements", "extrait")) and f["lignes"] and f["sources"]
        and all({"symbole", "nom", "valeur", "unite", "regle"} <= set(li) for li in f["lignes"]))
    chk(f"{r} : la fiche se relit en JSON (aucun objet Python)", json.loads(json.dumps(f)) == f)
    chk(f"{r} : aucun message par défaut", out["message"] is None, out["message"])
    note = B.consulter(r, {}, hote="note")
    chk(f"{r} : dans une note, la bande ajoute « Inclure dans la note » (fiche, extrait) et « Utiliser cette valeur »",
        note["bande"].count("data-inclure=") == 2 and 'data-utiliser="1"' in note["bande"] and note["tableau"] == out["tableau"])
chk("le menu porte les six rubriques dans l'ordre, la première allumée", re.findall(r'data-rubrique="([^"]+)"', B.menu()) == list(B.CLES) and 'class="bib-m on" data-rubrique="betons"' in B.menu())
chk("rubrique inconnue ou absente : la première s'ouvre, et le message le dit",
    B.consulter(None, {})["rubrique"] == "betons" and B.consulter(None, {})["message"] is None and B.consulter("zzz", {})["rubrique"] == "betons" and "inconnue" in B.consulter("zzz", {})["message"])
try:
    B.consulter("betons", {}, hote="pdf")
    chk("hôte inconnu : refusé", False)
except ValueError as e:
    chk("hôte inconnu : refusé", "hôte" in str(e))

# ============================================================================================ 2. les valeurs
print("\n=== 2 · Les valeurs de la bande sont celles des moteurs")
# (10/11/2026 : la rubrique a deux onglets ; le tableau 4.4N est dans « Tableaux », que l'ancien « vue= » ouvre aussi)
ex = B.consulter("enrobages", {"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": "12", "phi_ext": "0", "onglet": "tableaux"})
chk("enrobages : l'exemple de la NIT 260 (dalle XC1 C30/37 Ø 12) → 22 mm, S2 allumée", valeur(ex) == ("22", "mm") and 'class="on"><th class="nm">S2' in ex["tableau"])
chk("enrobages : la case allumée est [S2 ; XC1] (X0 = 10 la précède), la ligne S2 est dite « calculée », la colonne XC1 est marquée",
    re.search(r'<tr class="on"><th class="nm">S2 <small>calculée</small></th><td[^>]*>10</td><td class="cur"[^>]*>10</td>', ex["tableau"]) is not None
    and 'th class="on"><span class="s">XC1</span>' in ex["tableau"], ex["tableau"][:600])
p4 = B.consulter("enrobages", {"element": "poutre", "exposition": "XC4", "classe": "C30/37", "phi": "20", "phi_etrier": "8", "feu": "R60", "b": "250"})
chk("enrobages : poutre XC4 C30/37 R60 b 250 → 40 mm (la durabilité gouverne), la phrase le dit", valeur(p4) == ("40", "mm") and "la durabilité gouverne" in p4["bande"])
chk("enrobages : la justification cite a = 30 et a_sd = 40 (tableau 5.5)", "a<sub>sd</sub>" in p4["pourquoi"] and "tableau 5.5" in p4["pourquoi"])
pf = B.consulter("enrobages", {"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": "16", "phi_etrier": "6", "feu": "R120", "b": "240", "vue": "feu"})
chk("enrobages : R120 b 240 Ø 16 → le feu gouverne (56 mm) ; l'ancienne « vue=feu » ouvre « Tableaux » et allume la case (240, 60)",
    valeur(pf) == ("56", "mm") and "le feu gouverne" in pf["bande"] and pf["params"]["onglet"] == "tableaux"
    and re.search(r'<td class="cur"[^>]*data-sel="[^"]*R120[^"]*"[^>]*><b>60</b><small>≥\u00a0240</small>', pf["tableau"]) is not None, pf["tableau"][-900:])
pp = B.consulter("enrobages", {"element": "poteau", "feu": "R90", "mu_fi": "0.7", "b": "350", "onglet": "tableaux"})
chk("enrobages : poteau R 90, μ_fi = 0,7, b = 350 → le tableau 5.2a ENTIER, la case 350/53 allumée, 35 mm (53 − 10 − 8 : le feu gouverne)",
    valeur(pp) == ("35", "mm") and len(re.findall(r'<tr[^>]*><th class="nm">R\u00a0?\d+', pp["tableau"])) == 6
    and re.search(r'<td class="cur"[^>]*><b>53</b><small>≥\u00a0350</small>', pp["tableau"]) is not None and "le feu gouverne" in pp["bande"], pp["tableau"][:400])
ps = B.consulter("enrobages", {"contre_sol": "direct", "exposition": "XC2"})
chk("enrobages : coulé contre le sol → 75 mm, « le coulage contre le sol gouverne »", valeur(ps) == ("75", "mm") and "contre le sol gouverne" in ps["bande"])

a = B.consulter("ancrages", {})
chk("ancrages : Ø 16 C30/37 B500 par défaut → 580 mm (572 calculés), case [16 ; C30/37] allumée", valeur(a) == ("580", "mm") and "572" in a["bande"]
    and re.search(r'<tr class="on"><th class="nm">16</th>.*?<td class="cur"[^>]*data-sel="[^"]*C30/37', a["tableau"], re.S) is not None)
a2 = B.consulter("ancrages", {"grandeur": "l0"})
chk("ancrages : l_0 → 860 mm (858 calculés) ; le tableau l_0 a ses coefficients (53,6 Ø en C30/37)", valeur(a2) == ("860", "mm") and "53,6" in a2["tableau"])
a3 = B.consulter("ancrages", {"taux": "0,5", "forme": "courbe", "c_d": "60"})
chk("ancrages : réductions (σ_sd/f_yd 0,5, crochet c_d 60) → la bande change, le tableau non", valeur(a3)[0] != valeur(a)[0] and a3["tableau"] == a["tableau"] and "la case du tableau" in a3["bande"])
r_a3 = AN.calculer({"phi": 16, "classe": "C30/37", "taux": 0.5, "forme": "courbe", "c_d": 60})
chk("ancrages : la bande = arrondi au cm supérieur du moteur", valeur(a3)[0] == str(AN.arrondi_cm(r_a3["lbd"]["valeur"])), (valeur(a3), r_a3["lbd"]["valeur"]))
a4 = B.consulter("ancrages", {"phi": "40", "classe": "C20/25"})
chk("ancrages : Ø 40 → l'avertissement « gros diamètre » dans la bande et la marque « gros Ø » sur les lignes Ø 40 et 50",
    "gros diamètre" in a4["bande"] and a4["tableau"].count(">gros Ø</small>") == 2)
a5 = B.consulter("ancrages", {"conditions": "mediocres"})
chk("ancrages : conditions médiocres → le tableau entier change (× 1/0,7) et la colonne des coefficients aussi", a5["tableau"] != a["tableau"] and valeur(a5) == ("820", "mm"))

s = B.consulter("aciers", {})
chk("aciers : B500B, γ_s 1,15 → f_yd 434,8 MPa", valeur(s) == ("434,8", "MPa"))
s2 = B.consulter("aciers", {"gamma_s": "1.5", "nuance": "B400C"})
chk("aciers : B400C, γ_s 1,5 (bureau) → 266,7 MPa, ligne B400C allumée", valeur(s2) == ("266,7", "MPa") and '<tr class="on"><th class="nm">B400C' in s2["tableau"])
s3 = B.consulter("aciers", {"vue": "treillis", "treillis": "8/8/150/150"})
chk("aciers : treillis 8/8/150/150 → 335 mm²/m (fils porteurs)", valeur(s3) == ("335", "mm²/m") and '<tr class="on"><th class="nm">8/8/150/150' in s3["tableau"])

b = B.consulter("betons", {})
chk("bétons : C30/37, α_cc 1,0, γ_c 1,5 → f_cd 20,0 MPa ; 14 classes dans le tableau", valeur(b) == ("20,0", "MPa") and len(re.findall(r'<th class="nm">C\d+/\d+</th>', b["tableau"])) == 14)
b2 = B.consulter("betons", {"alpha_cc": "0.85", "gamma_c": "1.2", "classe": "C50/60"})
chk("bétons : C50/60, α_cc 0,85, γ_c 1,2 → f_cd 35,4 MPa", valeur(b2) == ("35,4", "MPa"))
b3 = B.consulter("betons", {"groupe": "bureau", "classe": "C60/75"})
chk("bétons : méthode du bureau pour C60/75 → cases vides (—) et l'avertissement « hors des tableaux du bureau »", 'class="vide"' in b3["tableau"] and "hors des tableaux du bureau" in b3["bande"])
b4 = B.consulter("betons", {"groupe": "bureau", "classe": "C30/37"})
chk("bétons : méthode du bureau pour C30/37 → α_b, μ et τ_adm présents, sans avertissement", "α<sub>b</sub>" in b4["tableau"] and "bib-av" not in b4["bande"])

c = B.consulter("sections", {})
chk("sections : 4Ø12 → 452 mm², case allumée sur la ligne Ø 12", valeur(c) == ("452", "mm²") and re.search(r'<tr class="on"><th class="nm">12</th>.*?<td class="cur"', c["tableau"], re.S) is not None)
c2 = B.consulter("sections", {"As_req": "650", "d": "10", "n": "4"})
chk("sections : A_s requis 650 mm² → la plus économique est marquée ★ et nommée (5Ø14 = 770 ? non : 6Ø12 = 679)", "★" in c2["tableau"] and ("6Ø12" in c2["bande"] or "4Ø16" in c2["bande"]) and "ne suffit pas" in c2["bande"])
c3 = B.consulter("sections", {"mode": "nappe", "d": "10", "espacement": "e15"})
chk("sections : par mètre, Ø 10 à 15 cm → 524 mm²/m", valeur(c3) == ("524", "mm²/m"))

# ============================================================================================ 3. les cases
print("\n=== 3 · Chaque case porte les paramètres qu'un clic règle ; les régler l'allume")
for r in B.CLES:
    out = B.consulter(r, {"onglet": "tableaux"} if r == "enrobages" else {})
    cases = CASES.findall(out["tableau"])
    rates = []
    for attrs, sel in cases[:: max(1, len(cases) // 40)]:       # une quarantaine de cases par rubrique
        d = json.loads(html.unescape(sel))
        p = {**out["params"], **d}
        o2 = B.consulter(r, p)
        allumee = [html.unescape(s) for a_, s in CASES.findall(o2["tableau"]) if "cur" in (re.search(r'class="([^"]*)"', a_) or [None, ""])[1]]
        if allumee != [html.unescape(sel)]:
            rates.append((d, allumee[:2]))
    chk(f"{r} : {len(cases)} cases cliquables ; régler les paramètres d'une case l'allume (quarante essayées)", cases and not rates, rates[:3])

# ============================================================================================ 4. les paramètres
print("\n=== 4 · Les paramètres : des chaînes, des refus doux, l'échappement, le routeur")
chk("les paramètres s'écrivent comme l'adresse les donne (chaînes) : même réponse qu'avec des nombres",
    B.consulter("ancrages", {"phi": "20", "taux": "0.8", "soude": "1"})["bande"] == B.consulter("ancrages", {"phi": 20, "taux": 0.8, "soude": True})["bande"])
m = B.consulter("ancrages", {"phi": "abc", "taux": "9", "classe": PIEGE})
chk("un paramètre illisible reprend son défaut, un hors bornes est ramené, le message le dit",
    m["params"]["phi"] == 16 and m["params"]["taux"] == 1.0 and m["params"]["classe"] == "C30/37" and "phi" in m["message"] and "taux" in m["message"])
chk("un texte piégé n'entre dans aucun fragment ni dans les paramètres relus (le message, du TEXTE que la page pose en textContent, le cite tronqué)",
    PIEGE not in fragments(m) and PIEGE not in json.dumps(m["params"]) and "« classe » ignoré" in m["message"])
e = B.consulter("enrobages", {"feu": "R45", "b": "10", "duree": "75", "qualite": "oui"})
chk("enrobages : feu R45 ignoré, b ramené à 20, durée 75 ignorée, qualité « oui » lue", e["params"]["feu"] == "" and e["params"]["b"] == 20 and e["params"]["duree"] == 50 and e["params"]["qualite"] is True)
rr = req(action="consulter", bibliotheque="beton", rubrique="ancrages", params={"phi": "25"})
chk("routeur : l'action « consulter » rend la rubrique (ok, fragments, paramètres relus)", rr["ok"] and rr["rubrique"] == "ancrages" and rr["params"]["phi"] == 25 and "bib-tab" in rr["tableau"])
# (la Bibliothèque Acier existe depuis BIB5 : tests/test_api_web_bibliotheque_acier.py ; l'inconnue est celle du bois)
rr2 = req(action="consulter", bibliotheque="bois", rubrique="profils")
chk("routeur : une bibliothèque inconnue est refusée sans exception", rr2["ok"] is False and "bibliothèque" in rr2["erreur"])
rr3 = req(action="consulter", rubrique="betons")
chk("routeur : sans bibliothèque nommée, la béton armé répond", rr3["ok"] and rr3["bibliotheque"] == "beton")
rr4 = req(action="consulter", bibliotheque="beton", rubrique="betons", hote="note")
chk("routeur : l'hôte « note » passe", rr4["ok"] and "data-utiliser" in rr4["bande"])
chk("la réponse du routeur se relit telle quelle en JSON (pas d'infini)", json.dumps(rr) and True)

# ============================================================================================ 5. la page
print("\n=== 5 · La page bibliotheque-beton")
page = SW.page_bibliotheque("beton")
js = open(os.path.join(RACINE, "site", "web", "js", "bibliotheque.js"), encoding="utf-8").read()
ids = sorted(set(re.findall(r'\$\("#([\w-]+)', js)) | set(re.findall(r'getElementById\("([\w-]+)"', js)))
absents = [i for i in ids if f'id="{i}"' not in page]
chk(f"js/bibliotheque.js → bibliotheque-beton.html : les {len(ids)} identifiants cherchés existent", not absents, absents)
chk("la page porte le menu des six rubriques (RUBRIQUES, une source) et son corps", B.menu() in page and 'data-bibliotheque="beton"' in page and 'id="bib-corps"' in page)
# (S4 : la page n'a plus qu'un script de module, js/app.js, qui monte l'écran de js/bibliotheque.js — body
# data-page="bibliotheque" ; le fichier construit porte la VERSION du site, recalculée ici comme le fait test_site_web ;
# le démarrage n'est plus une carte #demarrage de la page mais le bandeau partagé de js/commun.js, le même que l'établi)
_app = open(os.path.join(RACINE, "site", "web", "js", "app.js"), encoding="utf-8").read()
chk("la page charge css/bibliotheque.css et js/app.js, qui monte js/bibliotheque.js ; ni l'établi ni un second script",
    'href="css/bibliotheque.css"' in page and 'src="js/app.js"' in page and 'data-page="bibliotheque"' in page
    and "bibliotheque.js" not in page and "etabli.js" not in page
    and 'import { ouvrirBibliotheque } from "./bibliotheque.js";' in _app and "bibliotheque: ouvrirBibliotheque" in _app)
chk("la page n'a aucun script ni gestionnaire en ligne", not EN_LIGNE.search(page.replace('<script type="module" src="js/app.js"></script>', "").replace('<script src="js/theme.js"></script>', "")))
chk("le générateur écrit la page (PAGES_BIBLIOTHEQUES) et le fichier construit est le sien",
    SW.PAGES_BIBLIOTHEQUES.get("beton") == "bibliotheque-beton.html"
    and open(os.path.join(RACINE, "site", "web", "bibliotheque-beton.html"), encoding="utf-8").read()
    == SW.poser_version(page, SW.version_site(os.path.join(RACINE, "site", "web"))))
chk("aucun lien de la page ne vise une adresse en « .html »", not [h for h in re.findall(r'<a [^>]*href="([^"]+)"', page) if h.endswith(".html")])
chk("le démarrage du moteur est annoncé (le même bandeau que l'établi, js/commun.js ; « Chargement… » du tableau dans la page, sans carte #demarrage)",
    'id="demarrage"' not in page and '<div class="bib-tableau" id="bib-tableau"><div class="attente">Chargement…</div></div>' in page
    and "bandeauDemarrage(" in js and "lectureSeule(" in js and '"#bib-tableau"' in js
    and "bandeauDemarrage(" in open(os.path.join(RACINE, "site", "web", "js", "etabli.js"), encoding="utf-8").read())

# ============================================================================================ 6. mutations
print("\n=== 6 · Mutations : la page ne garde aucune copie des moteurs ; l'échappement tient")
garde = EN.C_MIN_DUR
try:
    EN.C_MIN_DUR = {**garde, "S2": (10, 20, 15, 20, 25, 30, 35)}
    chk("mutation « 4.4N S2/XC1 = 20 » : l'exemple de la NIT passe de 22 à 30 mm sur la page", valeur(B.consulter("enrobages", {"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": "12", "phi_etrier": "0"})) == ("30", "mm"))
finally:
    EN.C_MIN_DUR = garde
garde = AN.GAMMA_C
try:
    AN.GAMMA_C = 1.0
    chk("mutation « γ_c = 1,0 dans le moteur des ancrages » : la bande et le tableau suivent", valeur(B.consulter("ancrages", {})) == ("390", "mm"))
finally:
    AN.GAMMA_C = garde
garde = EC.GAMMA_C
try:
    EC.GAMMA_C = 1.2
    chk("mutation « γ_c = 1,2 dans ec2 » : f_cd ne change pas (la page passe γ_c explicitement) — la bande garde 20,0", valeur(B.consulter("betons", {})) == ("20,0", "MPa"))
finally:
    EC.GAMMA_C = garde
garde = B.C.esc
try:
    B.C.esc = lambda s: str(s)
    chk("sans échappement, un texte piégé n'entre toujours dans aucun fragment : chaque paramètre est un CHOIX ou un NOMBRE relu, jamais un texte recopié",
        all(PIEGE not in fragments(B.consulter(r, {k: PIEGE for k in B.consulter(r, {})["params"]})) for r in B.CLES))
finally:
    B.C.esc = garde


# ============================================================================================ 7. enrobages : deux onglets
print("\n=== 7 · Enrobages : « Calcul » et « Tableaux » — les mêmes paramètres (l'adresse), chaque nombre affiché = le moteur")
import types  # noqa: E402

from api_web import biblio_enrobages as BE  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import fiches as FI  # noqa: E402
from api_web import poutre as PO  # noqa: E402
from beton.bibliotheque import feu as FEU  # noqa: E402

NB = " "
TD = re.compile(r'<td([^>]*)>(.*?)</td>', re.S)
LIGNE = re.compile(r'<tr[^>]*><th class="nm">([^<]*)</th>(.*?)</tr>', re.S)


def texte(h: str) -> str:
    """Le texte que l'œil lit : les balises de mise en forme dans un mot (<i>, <sub>) tombent, les autres séparent."""
    h = re.sub(r"</?(i|sub)[^>]*>", "", h)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h)).replace(NB, " ")).strip()


def classe_de(attrs: str) -> str:
    m = re.search(r'class="([^"]*)"', attrs)
    return m.group(1) if m else ""


def page(rend, params):
    return rend(params) if rend else B.consulter("enrobages", params)


# les variantes de chaque élément : (paramètres de l'écran, paramètres du tableau dans le moteur)
VARIANTES = (("poteau", {}, {}), ("voile", {"porteur": "1"}, {"porteur": True}), ("voile", {"porteur": "0"}, {"porteur": False}),
             ("poutre", {"appuis": "simples"}, {"appuis": "simples"}), ("poutre", {"appuis": "continue"}, {"appuis": "continue"}),
             ("dalle", {"portee": "une"}, {"portee": "une"}), ("dalle", {"portee": "plancher_dalle"}, {"portee": "plancher_dalle"}))


def attendu(c: dict, sorte: str) -> str:
    rep = c["repere"]
    return {"combinaison": f"{c['a']}{rep} ≥ {c['dimension']}", "dimension": f"{c['dimension']}", "a": f"{c['a']}{rep}"}[sorte]


def ecarts_tableaux(rend=None) -> list:
    """Chaque case du tableau du feu affiché, relue contre feu.tableau (le moteur) : la même durée, la même colonne,
    la même combinaison, case par case — et l'état de vérification du tableau écrit dessous."""
    out = []
    for el, ps, pm in VARIANTES:
        o = page(rend, {"element": el, "onglet": "tableaux", "vue": "feu", "feu": "R60", **ps})
        tf = FEU.tableau(el, **pm)
        larg = {c["cle"]: max(1, max(len(tf["lignes"][r][c["cle"]]) for r in tf["durees"])) for c in tf["colonnes"]}
        lignes = LIGNE.findall(o["tableau"].split("<tbody>")[1])
        if [texte(n) for n, _ in lignes] != [tf["libelles"][r] for r in tf["durees"]]:
            out.append((el, ps, "durées", [texte(n) for n, _ in lignes]))
            continue
        for (n, corps), r in zip(lignes, tf["durees"]):
            vus = [texte(x) for _, x in TD.findall(corps)]
            prevus = []
            for c in tf["colonnes"]:
                combs = tf["lignes"][r][c["cle"]]
                prevus += [attendu(combs[k], c["sorte"]) if k < len(combs) else ("—" if not combs and k == 0 else "") for k in range(larg[c["cle"]])]
            if vus != prevus:
                out.append((el, ps, r, vus, prevus))
        if C.esc(tf["verification"]) not in o["tableau"] or f"tableau {tf['numero']}" not in o["tableau"].lower():
            out.append((el, ps, "vérification ou numéro absent"))
    return out


def rates_cases(rend=None) -> list:
    """Chaque case cliquable du tableau du feu, de chaque élément et de chaque variante : régler ses paramètres
    (data-sel, comme le fait js/biblio.js) l'allume, et elle seule."""
    out = []
    for el, ps, _ in VARIANTES:
        base = page(rend, {"element": el, "onglet": "tableaux", "vue": "feu", "feu": "R60", **ps})
        for i, (attrs, sel) in enumerate(CASES.findall(base["tableau"])):
            d = json.loads(html.unescape(sel))
            o2 = page(rend, {**{k: v for k, v in base["params"].items()}, **d})
            allumees = [j for j, (a_, s_) in enumerate(CASES.findall(o2["tableau"])) if "cur" in classe_de(a_).split()]
            if allumees != [i]:                   # la case allumée est CELLE qu'on a cliquée (sa place, pas son contenu)
                out.append((el, d, allumees[:2]))
    return out


chk("tableaux du feu : chaque case affichée (poteaux 5.2a, voiles 5.4 et 5.3, poutres 5.5 et 5.6, dalles 5.8 et 5.9) = feu.tableau, case par case ; l'état de vérification sous chaque tableau",
    not ecarts_tableaux(), ecarts_tableaux()[:2])
_r = rates_cases()
_n = sum(len(CASES.findall(B.consulter("enrobages", {"element": el, "onglet": "tableaux", "vue": "feu", "feu": "R60", **ps})["tableau"])) for el, ps, _ in VARIANTES)
chk(f"tableaux du feu : les {_n} cases cliquables des sept tableaux — régler les paramètres d'une case l'allume, elle seule", _n > 100 and not _r, _r[:3])


def case_allumee(o) -> str:
    m = [x for a_, x in TD.findall(o["tableau"]) if "cur" in classe_de(a_).split()]
    return texte(m[0]) if len(m) == 1 else f"{len(m)} cases"


def poteau450(rend=None) -> bool:
    return case_allumee(page(rend, {"element": "poteau", "feu": "R90", "mu_fi": "0.7", "b": "450", "onglet": "tableaux"})) == "40** ≥ 450"


chk("case allumée : poteau R 90, μ_fi = 0,7, b = 450 → 450/40** (au moins 8 barres) ; b = 349 → 350/53 allumée « non atteinte »",
    poteau450() and case_allumee(B.consulter("enrobages", {"element": "poteau", "feu": "R90", "mu_fi": "0.7", "b": "349", "onglet": "tableaux"})) == "53 ≥ 350"
    and 'class="cur ko"' in B.consulter("enrobages", {"element": "poteau", "feu": "R90", "mu_fi": "0.7", "b": "349", "onglet": "tableaux"})["tableau"])
vo = B.consulter("enrobages", {"element": "voile", "feu": "R180", "mu_voile": "0.7", "faces": "1", "e": "210", "phi": "20", "onglet": "tableaux"})
chk("case allumée : voile porteur REI 180, μ_fi = 0,7, une face, e = 210 → 210/50 ; c_nom = 50 − 10 = 40 mm (aucun étrier retiré)",
    case_allumee(vo) == "50 ≥ 210" and valeur(vo) == ("40", "mm"), (case_allumee(vo), valeur(vo)))
da = B.consulter("enrobages", {"element": "dalle", "feu": "R90", "portee": "continue", "h": "120", "onglet": "tableaux"})
chk("case allumée : dalle continue R 90 → la colonne « deux directions, l_y/l_x ≤ 1,5 » du tableau 5.8 (a = 15*) ; l'épaisseur 100 marquée sur sa ligne",
    case_allumee(da) == "15*" and re.search(r'<td class="col"><b>100</b></td>', da["tableau"]) is not None, case_allumee(da))
r24 = B.consulter("enrobages", {"element": "poteau", "feu": "R240", "mu_fi": "0.7", "b": "450", "onglet": "tableaux"})
chk("poteau R 240 à μ_fi = 0,7 : la case est vide (« — »), aucune case allumée, « durée non atteinte » dans la bande, c_nom de la durabilité seule",
    case_allumee(r24) == "0 cases" and "non atteinte" in r24["bande"] and valeur(r24) == ("30", "mm"))

# ---- la feuille de calcul : chaque terme du moteur, sa valeur, dans l'ordre ; c_nom allumé
CAS_FEUILLE = (
    ({"element": "poutre", "feu": "R60", "b": "250", "exposition": "XC4"}, {"element": "poutre", "exposition": "XC4", "feu": {"r": "R60", "appuis": "simples", "b": 250}}),
    ({"element": "poteau", "feu": "R90", "b": "350", "mu_fi": "0.7"},
     {"element": "poteau", "feu": {"r": "R90", "methode": "tableau", "exposition": "plusieurs", "b": 350, "mu_fi": 0.7}}),
    ({"element": "poteau", "feu": "R90", "b": "350", "mu_fi": "0.7", "methode": "formule"},
     {"element": "poteau", "feu": {"r": "R90", "methode": "formule", "exposition": "plusieurs", "b": 350, "mu_fi": 0.7, "l0_fi": 3.0, "barres": "4"}}),
    ({"element": "poteau", "feu": "R240", "b": "450", "mu_fi": "0.7"},
     {"element": "poteau", "feu": {"r": "R240", "methode": "tableau", "exposition": "plusieurs", "b": 450, "mu_fi": 0.7}}),
    ({"element": "dalle", "feu": "R90", "h": "100", "phi": "12", "contre_sol": "prepare"},
     {"element": "dalle", "phi": 12, "contre_sol": "prepare", "feu": {"r": "R90", "portee": "une", "h": 100}}),
    ({"element": "voile", "feu": "R180", "e": "210", "phi": "16", "phi_ext": "10"},
     {"element": "voile", "phi": 16, "phi_ext": 10, "feu": {"r": "R180", "porteur": True, "mu_fi": 0.7, "faces": 1, "e": 210}}),
    ({"element": "voile", "feu": "R60", "e": "100", "porteur": "0"}, {"element": "voile", "feu": {"r": "R60", "porteur": False, "e": 100}}),
    ({"element": "poutre", "classe_structurale": "S5", "qualite": "1", "duree": "100", "dg": "40"},
     {"element": "poutre", "classe_structurale": "S5", "qualite": True, "duree": 100, "dg": 40}),
)
TERME = re.compile(r'<tr data-terme="([^"]+)"[^>]*>.*?<td class="(v|cur)"><b>(.*?)</b>', re.S)


def valeur_prevue(t: dict) -> str:
    v, u = t["valeur"], t.get("unite", "")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return C.nb(float(v), 1) + (f" {u}" if u else "")
    return str(v) + (f" {u}" if u and str(v)[:1].isdigit() else "")


def ecarts_feuille(rend=None) -> list:
    out = []
    for ps, pm in CAS_FEUILLE:
        o = page(rend, ps)
        res = EN.calculer(pm)
        vus = [(cle, cls, texte(v)) for cle, cls, v in TERME.findall(o["tableau"])]
        prevus = [(t["cle"], "cur" if t["cle"] == "cnom" else "v", valeur_prevue(t)) for t in res["termes"]]
        if vus != prevus or valeur(o)[0] != C.nb(res["c_nom"], 1):
            out.append((ps, [x for x in zip(vus, prevus) if x[0] != x[1]][:2] or (len(vus), len(prevus)), valeur(o)))
        for t in res["termes"]:
            if C.esc(t["regle"])[:60].replace("&#x27;", "'") not in html.unescape(o["tableau"]).replace("<sub>", "_").replace("</sub>", "") and \
                    html.unescape(C.esc(t["regle"]))[:60] not in texte(o["tableau"]).replace(" ", ""):
                pass
    return out


chk(f"calcul : la feuille porte chaque terme du moteur, dans son ordre, avec sa valeur (huit situations : poutre, poteau au tableau et à la formule, R 240, dalle, voile porteur et non porteur, sans feu) ; la ligne c_nom est la case allumée",
    not ecarts_feuille(), ecarts_feuille()[:2])
po = B.consulter("enrobages", {"element": "poteau", "feu": "R90", "b": "350", "mu_fi": "0.7"})
chk("calcul : l'onglet s'ouvre par défaut (onglet=calcul), dessin à gauche, feuille à droite ; le dessin porte c_nom = 35 et a ≥ 53 · R 90",
    po["params"]["onglet"] == "calcul" and 'class="bib-calc"' in po["tableau"] and 'id="bib-enr-detail"' in po["tableau"]
    and "a ≥ 53 · R 90" in po["tableau"] and "enrobage nominal 35 mm" in po["tableau"])
pt = B.consulter("enrobages", {"element": "poutre", "feu": "R60", "b": "200"})
chk("calcul : poutre R 60, b = 200 → chaque terme porte sa référence (écrite) et sa règle complète (au survol de sa ligne), a_sd = 40 repéré sur la face latérale du dessin, les conditions d'emploi et l'état de vérification sous le dessin",
    all(C.esc(t["regle"]) + '"' in pt["tableau"] and texte(C.symb_math(t["reference"])) in texte(re.search(f'data-terme="{t["cle"]}".*?</tr>', pt["tableau"], re.S).group(0))
        for t in EN.calculer({"element": "poutre", "feu": {"r": "R60", "b": 200}})["termes"])
    and "sd</tspan>" in pt["tableau"] and " ≥ 40" in pt["tableau"] and C.esc(FEU.VERIFICATIONS["5.5"]) in pt["tableau"]
    and "poutre exposée au feu sur trois côtés" in pt["tableau"])
dl = B.consulter("enrobages", {"element": "dalle", "feu": "R90", "phi": "12", "h": "120"})
chk("calcul : dalle — « Ø nappe ext. » à la place de l'étrier (0 par défaut), c_nom,feu = a − Ø/2 = 30 − 6 = 24 sans étrier ; aucun étrier dessiné",
    'data-param="phi_ext"' in dl["barre"] and 'data-param="phi_etrier"' not in dl["barre"] and "étrier" not in texte(dl["tableau"]).split("Conditions")[0]
    and re.search(r'data-terme="cfeu".*?a − Ø/2.*?30 − 6.*?<b>24 mm</b>', texte(dl["tableau"]).replace(" ", " ") + "", re.S) is None
    and "24" in texte(re.search(r'data-terme="cfeu".*?</tr>', dl["tableau"], re.S).group(0)), texte(re.search(r'data-terme="cfeu".*?</tr>', dl["tableau"], re.S).group(0)))

# ---- le poteau : tableau par défaut, formule au choix (mention, comparaison, durée obtenue), un côté
pd_ = B.consulter("enrobages", {"element": "poteau", "feu": "R90"})
chk("poteau : la méthode par défaut est le tableau 5.2a (FEU.METHODE_DEFAUT) ; ses champs propres à la formule n'apparaissent pas",
    pd_["params"]["methode"] == "tableau" == FEU.METHODE_DEFAUT and all(f'data-param="{k}"' not in pd_["barre"] for k in ("h_grand", "l0_fi", "barres")))
pf7 = B.consulter("enrobages", {"element": "poteau", "feu": "R90", "b": "350", "mu_fi": "0.7", "methode": "formule"})
fo = texte(re.search(r'data-terme="comparaison".*?</tr>', pf7["tableau"], re.S).group(0))
chk("poteau, formule (5.7) : a = 47 mm (46,7 arrondi au mm supérieur), la mention de l'annexe belge sous le dessin (et dans « Pourquoi ? »), la ligne « tableau 5.2a de l'EN : 350/53 » en comparaison, la durée R obtenue",
    "47 mm" in texte(re.search(r'data-terme="a".*?</tr>', pf7["tableau"], re.S).group(0)) and "5.2.a-ANB" in pf7["tableau"] and "5.2.a-ANB" in pf7["pourquoi"] and "350/53" in fo
    and 'data-terme="R_obtenue"' in pf7["tableau"] and all(f'data-param="{k}"' in pf7["barre"] for k in ("h_grand", "l0_fi", "barres")), fo)
pft = B.consulter("enrobages", {**pf7["params"], "onglet": "tableaux", "vue": "feu"})
chk("poteau, formule, onglet « Tableaux » : le tableau 5.2a, la case de comparaison marquée en pointillé (aucune case « retenue »), la méthode dite",
    case_allumee(pft) == "0 cases" and re.search(r'<td class="cmp"[^>]*><b>53</b>', pft["tableau"]) is not None and "formule (5.7)" in pft["tableau"])
rf = B.consulter("enrobages", {"element": "poteau", "feu": "R90", "b": "150", "methode": "formule"})
chk("poteau, formule hors domaine (b = 150 : b′ < 200 mm) : refusée, le message le dit, le tableau 5.2a est lu (methode=tableau dans l'adresse)",
    rf["params"]["methode"] == "tableau" and "formule (5.7) refusée" in (rf["message"] or "") and "b′" in rf["message"], rf["message"])
r45 = B.consulter("enrobages", {"element": "poteau", "feu": "R90", "b": "400", "mu_fi": "0.45"})
chk("poteau, tableau, μ_fi = 0,45 : la colonne du DESSUS (0,5) est lue, et le message le dit",
    r45["params"]["mu_fi"] == 0.5 and "0,45" in (r45["message"] or "") and case_allumee(B.consulter("enrobages", {**r45["params"], "onglet": "tableaux"})) == "38 ≥ 400")
un = B.consulter("enrobages", {"element": "poteau", "feu": "R90", "expo_feu": "un", "mu_fi": "0.5"})
chk("poteau exposé sur un seul côté : « Formule (5.7) » et μ_fi 0,2 / 0,5 grisés avec leur raison ; la colonne 0,7 lue, la mention le dit sous le dessin",
    re.search(r'<button type="button"[^>]*disabled title="la formule \(5\.7\) vaut[^"]*" data-valeur="formule">', un["barre"]) is not None
    and len(re.findall(r'disabled title="exposé sur un seul côté', un["barre"])) == 2 and "colonne μfi = 0,7" in texte(un["tableau"]))


# ---- l'adresse : les paramètres relus rouvrent le même écran ; l'ancien « vue= » est lu
def en_adresse(p: dict) -> dict:
    """Les paramètres tels que js/bibliotheque.js les écrit dans l'adresse puis les relit : des chaînes."""
    return {k: ("true" if v is True else "false" if v is False else str(v)) for k, v in p.items() if v is not None and v != ""}


def ecarts_adresse(rend=None) -> list:
    out = []
    # chaque adresse : l'aller-retour rend le même écran, et la case qu'elle désigne est allumée
    for ps, case in (({"element": "poteau", "feu": "R90", "b": "350", "mu_fi": "0.7", "onglet": "tableaux"}, "53 ≥ 350"), (pf7["params"], None),
                     ({"element": "voile", "feu": "R120", "porteur": "1", "mu_voile": "0.35", "faces": "2", "e": "160"}, "25 ≥ 160"),
                     ({"element": "dalle", "feu": "R60", "portee": "plancher_dalle", "h": "180", "onglet": "tableaux", "vue": "feu"}, "15*"),
                     ({"element": "poutre", "feu": "R30", "appuis": "continue", "b": "160", "qualite": "1", "classe_structurale": "S3"}, "12* ≥ 160")):
        o = page(rend, ps)
        o2 = page(rend, en_adresse(o["params"]))
        if (o2["tableau"], o2["bande"], o2["barre"]) != (o["tableau"], o["bande"], o["barre"]) or o2["message"]:
            out.append(("aller-retour", ps, o2["message"]))
        if case and case_allumee(page(rend, {**en_adresse(o["params"]), "onglet": "tableaux", "vue": "feu"})) != case:
            out.append(("case", ps, case))
    for ps, onglet, quoi in (({"vue": "feu", "feu": "R60"}, "tableaux", "bib-feu"), ({"vue": "durabilite"}, "tableaux", "bib-enr"), ({}, "calcul", "bib-calc"),
                             ({"onglet": "calcul", "vue": "feu"}, "calcul", "bib-calc"), ({"onglet": "tableaux"}, "tableaux", "bib-enr")):
        o = page(rend, ps)
        if o["params"]["onglet"] != onglet or quoi not in o["tableau"]:
            out.append(("onglet", ps, o["params"].get("onglet")))
    return out


chk("l'adresse : les paramètres relus rouvrent le même écran (barre, tableau, bande), sans message ; « vue=feu » ou « vue=durabilite » sans onglet ouvre « Tableaux » sur ce tableau ; sans rien, « Calcul »",
    not ecarts_adresse(), ecarts_adresse()[:2])
chk("les deux onglets partagent les paramètres : la même bande, la même fiche, l'onglet seul change",
    all(B.consulter("enrobages", {**pf7["params"], "onglet": "calcul"})[k] == B.consulter("enrobages", {**pf7["params"], "onglet": "tableaux"})[k] for k in ("bande", "pourquoi", "sources"))
    and 'data-param="onglet"' in pf7["barre"] and 'data-valeur="tableaux"' in pf7["barre"])

# ---- l'échappement : un texte piégé dans chaque paramètre, dans chaque élément et chaque méthode
_etats = [pf7["params"], vo["params"], da["params"], pt["params"], {**pf7["params"], "onglet": "tableaux"}]
chk("échappement : un texte piégé dans chaque paramètre des enrobages (poteau à la formule, voile, dalle, poutre ; les deux onglets) n'entre dans aucun fragment",
    all(PIEGE not in fragments(B.consulter("enrobages", {k: PIEGE for k in e_})) and PIEGE not in fragments(B.consulter("enrobages", {**e_, "methode": PIEGE, "h_grand": PIEGE}))
        for e_ in _etats))

# ---- le panneau d'une poutre : il s'ouvre sur « Calcul », prérempli ; la fiche se relit ; avec le feu, la fiche porte son tableau
e = PO.nouvel_etat()
for chemin, v in ((["beton"], "C25/30"), (["b"], "30"), (["sections", "0", "inf", "0", "d"], "20"), (["sections", "0", "etriers", "0", "d"], "8")):
    assert PO.modifier(e, chemin, v) is None, chemin
rp = req(module="poutre", action="bibliotheque", etat=e)
chk("panneau d'une poutre : l'onglet « Calcul », poutre sur appuis simples (le défaut), b = 300, C25/30, Ø 20, étrier 8 ; une seule case allumée (c_nom) ; « Utiliser cette valeur »",
    rp["ok"] and rp["params"]["onglet"] == "calcul" and rp["params"]["element"] == "poutre" and rp["params"]["appuis"] == "simples" and rp["params"]["b"] == 300
    and rp["params"]["classe"] == "C25/30" and rp["params"]["phi"] == 20 and rp["tableau"].count('class="cur"') == 1 and 'data-utiliser="1"' in rp["bande"],
    {k: rp["params"].get(k) for k in ("onglet", "element", "appuis", "b")})
rpf = req(module="poutre", action="bibliotheque", etat=e, rubrique="enrobages", params={"feu": "R90"})
fi = FI.relire({**rpf["fiche"], "id": "f1", "sorte": "fiche", "bibliotheque": "beton", "rubrique": "enrobages", "date": "2026-11-10"})
chk("panneau, feu R 90 : la fiche « Inclure dans la note » se relit (api_web/fiches.relire) et porte le tableau 5.5 entier, la case (R 90 ; colonne 3 : 300/40) marquée",
    rpf["ok"] and "5.5" in fi["extrait"]["titre"] and fi["extrait"]["case"] == [2, 3] and fi["extrait"]["lignes"][2][3] == "300/40" and len(fi["extrait"]["lignes"]) == 6,
    (fi["extrait"]["titre"], fi["extrait"]["case"], fi["extrait"]["lignes"][2]))
for ps in (pf7["params"], vo["params"], r24["params"], {}):
    try:
        FI.relire({**B.consulter("enrobages", ps)["fiche"], "id": "f1", "sorte": "extrait", "bibliotheque": "beton", "rubrique": "enrobages", "date": "2026-11-10"})
        ok_ = True
    except ValueError as x:
        ok_ = str(x)
    chk(f"fiche figée relue sans refus ({ps.get('element', 'défaut')}{(' ' + ps['methode']) if ps.get('methode') else ''})", ok_ is True, ok_)

# ---- mutations de l'écran : chacune fait rougir un contrôle ci-dessus
print("\n=== 7 bis · Mutations de l'écran des enrobages")
SRC_BE = open(os.path.join(RACINE, "api_web", "biblio_enrobages.py"), encoding="utf-8").read()


def mutant(avant: str, apres: str):
    assert SRC_BE.count(avant) == 1, avant
    m = types.ModuleType("biblio_enrobages_mutant")
    exec(compile(SRC_BE.replace(avant, apres), "biblio_enrobages_mutant", "exec"), m.__dict__)
    return lambda params: m.rendre(params, "page")


for nom, avant, apres, controle in (
        ("une case ne règle plus sa dimension (le clic allume une autre case)", 'sel.update({"appuis": pa["appuis"], cle_dim: comb["dimension"]})',
         'sel.update({"appuis": pa["appuis"]})', rates_cases),
        ("la case allumée est toujours la première combinaison de la colonne", '0 if tf["numero"] in ("5.5", "5.6") else f["indice"]', "0",
         lambda r: [] if poteau450(r) else ["450/40 non allumée"]),
        ("la colonne μ_fi lue est celle du DESSOUS (le défaut D1 d'hier)", "col = next(m for m in MU_POTEAU if m >= mu - 1e-9)",
         "col = next(m for m in reversed(MU_POTEAU) if m <= mu + 1e-9)",
         lambda r: [] if case_allumee(page(r, {"element": "poteau", "feu": "R90", "b": "400", "mu_fi": "0.45", "onglet": "tableaux"})) == "38 ≥ 400" else ["colonne"]),
        ("la feuille oublie le groupe de l'enrobage retenu", ', ("retenu", "", ""))', ')', ecarts_feuille),
        ("une combinaison du tableau perd son repère « ** »", "return f\"<b>{comb['a']}{marque}</b>\", f\"{comb['a']}{rep}\"",
         "return f\"<b>{comb['a']}</b>\", f\"{comb['a']}{rep}\"", ecarts_tableaux),
        ("le tableau du feu écrit b_min à la place de a", "<b>{comb['a']}{marque}</b><small>≥{NBSP}{comb['dimension']}</small>",
         "<b>{comb['dimension']}{marque}</b><small>≥{NBSP}{comb['dimension']}</small>", ecarts_tableaux),
        ("l'ancien « vue= » n'ouvre plus « Tableaux »", 'onglet = "tableaux" if "vue" in brut else "calcul"', 'onglet = "calcul"', ecarts_adresse),
        ("le μ_fi d'un voile relu dans l'adresse n'atteint plus le moteur", 'f["mu_fi"] = L.choix("mu_voile", FEU.MU_VOILE, 0.7)',
         'f["mu_fi"] = L.choix("mu_voile", FEU.MU_VOILE, 0.7) and 0.7', ecarts_adresse),
):
    chk(f"mutation « {nom} » : le contrôle rougit", bool(controle(mutant(avant, apres))))
garde = FEU.POTEAU_5_2A["R90"]
try:
    FEU.POTEAU_5_2A["R90"] = {**garde, "mu07": ((350, 45, ""), (450, 40, "**"))}
    chk("mutation du MOTEUR « 5.2a R 90 μ 0,7 : 350/45 » : la case allumée et la bande suivent (l'écran ne garde aucune copie)",
        case_allumee(B.consulter("enrobages", {"element": "poteau", "feu": "R90", "mu_fi": "0.7", "b": "350", "onglet": "tableaux"})) == "45 ≥ 350"
        and valeur(B.consulter("enrobages", {"element": "poteau", "feu": "R90", "mu_fi": "0.7", "b": "350"})) == ("30", "mm"))
finally:
    FEU.POTEAU_5_2A["R90"] = garde

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
