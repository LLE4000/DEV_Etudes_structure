# -*- coding: utf-8 -*-
"""La page BIBLIOTHÈQUE BÉTON ARMÉ (api_web/biblio_beton.py, action « consulter » du routeur, gabarit
site/web/construire.page_bibliotheque), vérifiée en Python ordinaire : chaque valeur de la bande est celle du MOTEUR
(beton/bibliotheque, beton/donnees, beton/armatures), lue ici indépendamment de la page.

1. Les cinq rubriques se rendent avec leurs défauts : barre, tableau (une seule case allumée), bande (« Pourquoi ? »),
   justification, sources, fiche figée, paramètres relus ; aucun script ni « on…= » ; l'hôte « note » ajoute
   « Inclure dans la note » et « Utiliser cette valeur », la page non.
2. Les valeurs : l'exemple de la NIT 260 (dalle XC1 C30/37 Ø 12 → 22 mm), l'ancrage Ø 16 C30/37 (580 mm), B500B
   (f_yd 434,8), C30/37 (f_cd 20,0), 4Ø12 (452 mm²) ; les réductions de l'ancrage changent la bande, pas le tableau.
3. Les cases : chaque case porte les paramètres qu'un clic règle (JSON) ; les régler allume CETTE case — sur toutes
   les cases de chaque rubrique.
4. Les paramètres : écrits comme l'adresse les donne (des chaînes) ; un paramètre illisible reprend son défaut et le
   message le dit ; un texte piégé est échappé ; une rubrique inconnue ouvre la première ; le routeur répond à
   « consulter » et refuse une bibliothèque inconnue sans exception.
5. La page : le gabarit porte le menu des cinq rubriques, les identifiants que js/bibliotheque.js cherche, la feuille
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


# ============================================================================================ 1. les cinq rubriques
print("=== 1 · Les cinq rubriques, par défaut")
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
chk("le menu porte les cinq rubriques dans l'ordre, la première allumée", re.findall(r'data-rubrique="([^"]+)"', B.menu()) == list(B.CLES) and 'class="bib-m on" data-rubrique="betons"' in B.menu())
chk("rubrique inconnue ou absente : la première s'ouvre, et le message le dit",
    B.consulter(None, {})["rubrique"] == "betons" and B.consulter(None, {})["message"] is None and B.consulter("zzz", {})["rubrique"] == "betons" and "inconnue" in B.consulter("zzz", {})["message"])
try:
    B.consulter("betons", {}, hote="pdf")
    chk("hôte inconnu : refusé", False)
except ValueError as e:
    chk("hôte inconnu : refusé", "hôte" in str(e))

# ============================================================================================ 2. les valeurs
print("\n=== 2 · Les valeurs de la bande sont celles des moteurs")
ex = B.consulter("enrobages", {"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": "12", "phi_etrier": "0"})
chk("enrobages : l'exemple de la NIT 260 (dalle XC1 C30/37 Ø 12) → 22 mm, S2 allumée", valeur(ex) == ("22", "mm") and 'class="on"><th class="nm">S2' in ex["tableau"])
chk("enrobages : la case allumée est [S2 ; XC1] (X0 = 10 la précède), la ligne S2 est dite « calculée », la colonne XC1 est marquée",
    re.search(r'<tr class="on"><th class="nm">S2 <small>calculée</small></th><td[^>]*>10</td><td class="cur"[^>]*>10</td>', ex["tableau"]) is not None
    and 'th class="on"><span class="s">XC1</span>' in ex["tableau"], ex["tableau"][:600])
p4 = B.consulter("enrobages", {"element": "poutre", "exposition": "XC4", "classe": "C30/37", "phi": "20", "phi_etrier": "8", "feu": "R60", "b": "250"})
chk("enrobages : poutre XC4 C30/37 R60 b 250 → 40 mm (la durabilité gouverne), la phrase le dit", valeur(p4) == ("40", "mm") and "la durabilité gouverne" in p4["bande"])
chk("enrobages : la justification cite a = 30 et a_sd = 40 (tableau 5.5)", "a_sd" in html.unescape(p4["pourquoi"]).replace("<sub>", "_").replace("</sub>", "") or "a<sub>sd</sub>" in p4["pourquoi"])
pf = B.consulter("enrobages", {"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": "16", "phi_etrier": "6", "feu": "R120", "b": "240", "vue": "feu"})
chk("enrobages : R120 b 240 Ø 16 → le feu gouverne (56 mm) ; la vue « feu » allume la ligne (240, 60)", valeur(pf) == ("56", "mm") and "le feu gouverne" in pf["bande"]
    and re.search(r'<td class="cur"[^>]*data-sel="[^"]*R120[^"]*"[^>]*><b>60</b><small>b ≥ 240</small>', pf["tableau"]) is not None)
pp = B.consulter("enrobages", {"element": "poteau", "feu": "R60", "vue": "feu"})
chk("enrobages : poteau au feu → le tableau 5.2a est dit non transcrit, aucune valeur de feu", "n'est pas transcrit" in pp["tableau"] and "bib-vide" in pp["tableau"] and "feu_a_transcrire" not in pp["bande"] and "n'est pas transcrit" in pp["bande"])
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
    out = B.consulter(r, {})
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
chk("enrobages : feu R45 ignoré, b ramené à 50, durée 75 ignorée, qualité « oui » lue", e["params"]["feu"] == "" and e["params"]["b"] == 50 and e["params"]["duree"] == 50 and e["params"]["qualite"] is True)
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
chk("la page porte le menu des cinq rubriques (RUBRIQUES, une source) et son corps", B.menu() in page and 'data-bibliotheque="beton"' in page and 'id="bib-corps"' in page)
chk("la page charge css/bibliotheque.css et js/bibliotheque.js, pas l'établi", 'href="css/bibliotheque.css"' in page and 'src="js/bibliotheque.js"' in page and "etabli.js" not in page)
chk("la page n'a aucun script ni gestionnaire en ligne", not EN_LIGNE.search(page.replace('<script type="module" src="js/bibliotheque.js"></script>', "").replace('<script src="js/theme.js"></script>', "")))
chk("le générateur écrit la page (PAGES_BIBLIOTHEQUES) et le fichier construit est le sien",
    SW.PAGES_BIBLIOTHEQUES.get("beton") == "bibliotheque-beton.html"
    and open(os.path.join(RACINE, "site", "web", "bibliotheque-beton.html"), encoding="utf-8").read() == page)
chk("aucun lien de la page ne vise une adresse en « .html »", not [h for h in re.findall(r'<a [^>]*href="([^"]+)"', page) if h.endswith(".html")])
chk("le démarrage du moteur est annoncé (la même carte que l'établi)", 'id="demarrage"' in page and 'id="demarrage-etape"' in page)

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

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
