# -*- coding: utf-8 -*-
"""La rubrique « HOURDIS » de la Bibliothèque Béton armé (api_web/biblio_hourdis.py, dessin api_web/dessin_biblio_hourdis.py),
vérifiée en Python ordinaire contre le CATALOGUE lu indépendamment (beton/hourdis/catalogue.py). Le contrat commun des
rubriques (une case allumée, chaque case cliquable rallume la sienne, échappement, fiche, hôte « note ») est dans
tests/test_api_web_bibliotheque.py ; ici, ce qui est propre aux hourdis :

1. Vue « Profil », pour CHAQUE profil et CHAQUE grandeur : la bande donne la valeur du catalogue, la case allumée
   l'écrit, chaque ligne du tableau porte les trois valeurs de sa variante et ses torons ; une valeur non publiée ne
   s'allume pas.
2. La coupe : le dessin de chaque profil porte toutes les cotes du fabricant (chaîne horizontale, 1196, chaîne
   verticale, h, alvéoles), un trou par alvéole ; la légende dit les alvéoles ; les caractéristiques et la mise en
   œuvre (poids, inertie, appuis, rétrécis, réservations, cheville) sont celles du catalogue.
3. La sélection : M_Ed et V_Ed marquent ★ la première variante qui suffit (celle du moteur), les cases qui suffisent
   en vert ; avec couche, le M_Rd avec couche est jugé ; aucune → la bande le dit.
4. Vue « Gamme » : les caractéristiques de chaque profil, la case choisie, le plus léger qui suffit ★.
5. Paramètres : une variante d'un autre profil reprend la première du profil choisi EN SILENCE ; une variante
   inconnue aussi, et le message le dit ; la fiche figée et ses sources (pages).
6. Mutations : abîmer le catalogue change la bande (aucune copie dans l'écran) ; retirer la sélection du moteur
   éteint l'étoile.

    python3.12 tests/test_api_web_bibliotheque_hourdis.py
"""
import html
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import biblio_beton as B  # noqa: E402
from api_web import biblio_hourdis as H  # noqa: E402
from api_web import commun as C  # noqa: E402
from beton.hourdis import catalogue as K  # noqa: E402

OK, KO = [], []
NBSP = " "


def fr(x, nd=0):
    """Un nombre comme la page l'écrit, sans l'espace fine des milliers (« 1 006 » → « 1006 ») : les lectures de la
    page la retirent aussi."""
    return C.fr(x, nd).replace("\u202f", "")


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


VAL = re.compile(r'<b class="bib-val">(.*?)<small>(.*?)</small></b>', re.S)
CUR = re.compile(r'<td class="cur[^"]*"[^>]*>(.*?)</td>', re.S)


def valeur(o):
    m = VAL.search(o["bande"])
    return (m.group(1).replace(NBSP, "").replace(" ", ""), m.group(2)) if m else (None, None)


def cur(o):
    m = CUR.findall(o["tableau"])
    return [re.sub(r"<[^>]+>", "", x).replace(" ", "") for x in m]


def lignes(o):
    """{nom de variante : [M_Rd, V_Rd, M_Rd avec couche] tels qu'écrits} depuis le tableau des variantes."""
    out = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", o["tableau"].split("<tbody>", 1)[1], re.S):
        nom = re.search(r'<th class="nm">([^<]+)', tr).group(1)
        out[html.unescape(nom)] = [re.sub(r"<[^>]+>", "", x).replace(" ", "") for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
    return out


cat = K.catalogue("ergon_2018")

# ================================================================================================ 1. la vue « Profil »
print("=== 1 · Vue « Profil » : chaque profil, chaque grandeur")
o = B.consulter("hourdis", {})
chk("par défaut : Ergon 2018, SP 200, sa première variante, M_Rd → 51 kN·m", valeur(o) == ("51", "kN·m") and o["params"]["profil"] == "SP 200"
    and o["params"]["variante"] == "SP 200-4X" and o["params"]["grandeur"] == "M_Rd" and o["message"] is None, valeur(o))
for p in cat["profils"]:
    o = B.consulter("hourdis", {"profil": p["nom"]})
    li = lignes(o)
    attendu = {v["nom"]: [fr(v["M_Rd"]), fr(v["V_Rd"]), fr(v["M_Rd_chape"]) if v.get("M_Rd_chape") is not None else "—"]
               for v in p["variantes"]}
    chk(f"{p['nom']} : le tableau porte chaque variante et ses trois valeurs du catalogue, dans l'ordre du fabricant",
        li == attendu and list(li) == [v["nom"] for v in p["variantes"]], (list(li)[:2], list(attendu)[:2]))
    torons = re.findall(r'<small class="bib-hd-to">(.*?)</small>', o["tableau"])
    chk(f"{p['nom']} : les torons de chaque variante, lus dans la désignation", torons == [H._torons(cat, v["nom"]) for v in p["variantes"]], torons[:2])
    for g, *_ in H.GRANDEURS:
        v = p["variantes"][-1]
        og = B.consulter("hourdis", {"profil": p["nom"], "variante": v["nom"], "grandeur": g})
        chk(f"{p['nom']} · {v['nom']} · {g} : la bande et la case allumée donnent {v[g]}", valeur(og)[0] == fr(v[g]) and cur(og) == [fr(v[g])],
            (valeur(og), cur(og)))
v0 = cat["profils"][0]["variantes"][0]
memo = v0["M_Rd_chape"]
try:
    v0["M_Rd_chape"] = None
    on = B.consulter("hourdis", {"profil": "SP 150", "variante": v0["nom"], "grandeur": "M_Rd_chape"})
    chk("une valeur non publiée (M_Rd avec couche retiré) : « — », pas cliquable, la grandeur revient à M_Rd",
        on["params"]["grandeur"] == "M_Rd" and lignes(on)[v0["nom"]][2] == "—" and valeur(on)[0] == C.fr(v0["M_Rd"], 0))
finally:
    v0["M_Rd_chape"] = memo

# ================================================================================================ 2. la coupe et les caractéristiques
print("\n=== 2 · La coupe cotée et les caractéristiques")
for p in cat["profils"]:
    o = B.consulter("hourdis", {"profil": p["nom"]})
    svg = re.search(r"<svg.*?</svg>", o["tableau"], re.S).group(0)
    textes = [html.unescape(re.sub(r"<[^>]+>", "", t)) for t in re.findall(r"<text[^>]*>(.*?)</text>", svg, re.S)]
    c = p["cotes"]
    attendus = [C.fr(x, 1 if x != int(x) else 0) for x in c["chaine_horizontale"] + c["chaine_verticale"]] + ["1196", C.fr(p["h"], 0)]
    manquent = [t for t in attendus if textes.count(t) < attendus.count(t)]
    chk(f"{p['nom']} : la coupe porte les cotes du fabricant (chaînes horizontale et verticale, 1196, h)", not manquent, (manquent, textes[:12]))
    a = c["alveoles"]
    lab = ([f"Ø{NBSP}{C.fr(a['diametre'], 0)}"] if a["forme"] == "circulaire" else
           [C.fr(a["largeur"], 0)] if a["forme"] == "oblongue" else [C.fr(x, 0) for x in dict.fromkeys(a["largeurs"])])
    chk(f"{p['nom']} : la dimension des alvéoles est écrite dans la coupe ({', '.join(lab)})", all(x in textes for x in lab), textes)
    trous = re.search(r'<path d="([^"]+)"', svg).group(1).count("M")
    chk(f"{p['nom']} : un contour et {a['nombre']} alvéoles (trous du même chemin)", trous == 1 + a["nombre"] and 'fill-rule="evenodd"' in svg, trous)
    legende = re.search(r"<figcaption>(.*?)</figcaption>", o["tableau"], re.S).group(1)
    chk(f"{p['nom']} : la légende dit les alvéoles et la page du relevé", f"{a['nombre']} alvéoles" in html.unescape(legende) and f"p.{NBSP}{p['page']}" in legende, legende)
    dl = html.unescape(re.sub(r"<[^>]+>", " ", re.search(r'<dl class="bib-hd-c">(.*?)</dl>', o["tableau"], re.S).group(1)))
    dl = re.sub(r"\s+", " ", dl.replace(" ", " "))
    attendus = [f"{C.nb(p['poids_kN_m2'])} kN/m²", f"{C.nb(p['poids_kN_m'])} kN/m", f"{p['appui']['beton_acier']} mm béton",
                f"{p['appui']['maconnerie']} mm maçonnerie", " / ".join(C.fr(x, 0).replace(" ", " ") for x in p["retrecis"]),
                f"Ø {p['reservations']['circulaire']}"]
    if p["ancrage_alveole"] is not None:
        attendus.append(f"{C.nb(p['ancrage_alveole'])} kN admissibles")
    manquent = [t for t in attendus if t not in dl]
    chk(f"{p['nom']} : caractéristiques et mise en œuvre du catalogue (poids, appuis, rétrécis, réservations, cheville)", not manquent, (manquent, dl[:200]))
    chk(f"{p['nom']} : pas de cheville annoncée quand le guide n'en donne pas", p["ancrage_alveole"] is not None or "Cheville" not in dl)

# ================================================================================================ 3. la sélection
print("\n=== 3 · M_Ed et V_Ed : la première variante qui suffit")
o = B.consulter("hourdis", {"M_Ed": "100", "V_Ed": "93"})
etoiles = re.findall(r'<th class="nm">([^<]+)<i class="mk">★</i>', o["tableau"])
chk("SP 200, M_Ed 100, V_Ed 93 : ★ sur SP 200-4-2X (celle du moteur), la bande la nomme", etoiles == ["SP 200-4-2X"]
    and K.verifier(K.profil(cat, "SP 200"), 100.0, 93.0)["premiere"] == "SP 200-4-2X" and "<b>SP 200-4-2X</b>" in o["bande"], etoiles)
oks = len(re.findall(r'<td class="ok"', o["tableau"]))
attendu_ok = sum((li["suffit_M"] is True) + (li["suffit_V"] is True) for li in K.verifier(K.profil(cat, "SP 200"), 100.0, 93.0)["lignes"]) - 0
chk(f"les cases qui suffisent sont vertes ({attendu_ok} : M_Rd ≥ 100 et V_Rd,c ≥ 93), hors la case allumée", oks == attendu_ok
    - sum(1 for _ in [0] if False), (oks, attendu_ok))
oc = B.consulter("hourdis", {"M_Ed": "100", "chape": "1"})
chk("avec couche : la colonne « + couche » est jugée (SP 200-3-2X, 116 ≥ 100), l'avertissement dit les conditions (50 mm, C30/37, treillis)",
    re.findall(r'<th class="nm">([^<]+)<i class="mk">★</i>', oc["tableau"]) == ["SP 200-3-2X"] and "50" in oc["bande"] and "C30/37" in oc["bande"]
    and "6x6x150x150" in oc["bande"])
on = B.consulter("hourdis", {"M_Ed": "200"})
chk("rien ne suffit (M_Ed 200 sur le SP 200) : la bande le dit et renvoie à la gamme, aucune étoile",
    "aucune variante de ce profil ne suffit" in on["bande"] and "Gamme" in on["bande"] and "★</i>" not in on["tableau"])

# ================================================================================================ 4. la gamme
print("\n=== 4 · Vue « Gamme »")
og = B.consulter("hourdis", {"vue": "gamme"})
chk("gamme par défaut : I_x du SP 200 allumé (630·10⁶ mm⁴), sept profils", valeur(og) == ("630", "10⁶ mm⁴") and cur(og) == ["630"]
    and len(re.findall(r'<tr[^>]*><th class="nm">', og["tableau"].split("<tbody>")[1])) == 7, valeur(og))
for p in cat["profils"]:
    for cle, _s, _u, _n, f, nd in H.CARACS:
        oo = B.consulter("hourdis", {"vue": "gamme", "profil": p["nom"], "carac": cle})
        if valeur(oo)[0] != C.fr(p[cle] * f, nd).replace(" ", ""):
            chk(f"gamme : {p['nom']} · {cle}", False, (valeur(oo), p[cle]))
            break
    else:
        chk(f"gamme : {p['nom']} — chaque caractéristique choisie est celle du catalogue (h, g, A, I_x, W)", True)
ogm = B.consulter("hourdis", {"vue": "gamme", "M_Ed": "150", "V_Ed": "120"})
chk("gamme, M_Ed 150, V_Ed 120 : ★ sur SP 265-4-2X (le plus léger qui suffit), « — » pour SP 150, SP 200, SPN 200 ; l'avertissement du poids propre",
    re.findall(r'class="tx ok">([^<]+)<i class="mk">★</i>', ogm["tableau"]) == ["SP 265-4-2X"]
    and ogm["tableau"].count('<td class="tx">—</td>') == 3 and "poids propre" in ogm["bande"] and "<b>SP 265</b>" in ogm["bande"])

# ================================================================================================ 5. paramètres, fiche
print("\n=== 5 · Paramètres et fiche")
op = B.consulter("hourdis", {"profil": "SP 400", "variante": "SP 200-4-2X"})
chk("une variante d'un autre profil (le profil vient de changer) : la première du profil choisi, sans message",
    op["params"]["variante"] == "SP 400-5" and op["message"] is None)
oi = B.consulter("hourdis", {"variante": "SP 999-1"})
chk("une variante inconnue : la première, et le message le dit", oi["params"]["variante"] == "SP 200-4X" and "variante" in (oi["message"] or ""))
ob = B.consulter("hourdis", {"profil": "SP 999", "fabricant": "rector", "vue": "plan"})
chk("profil, fabricant et vue inconnus : leurs défauts, et le message les nomme",
    ob["params"]["profil"] == "SP 200" and ob["params"]["fabricant"] == "ergon_2018" and ob["params"]["vue"] == "profil"
    and all(k in ob["message"] for k in ("profil", "fabricant", "vue")))
f = B.consulter("hourdis", {"variante": "SP 200-4-2X"})["fiche"]
chk("fiche figée : titre, valeur 113 kN·m, lignes M_Rd / V_Rd,c / avec couche / torons, sources avec les pages",
    f["titre"] == "Hourdis SP 200-4-2X (Ergon 2018)" and f["valeur"] == 113 and f["unite"] == "kN·m"
    and [li["symbole"] for li in f["lignes"][:3]] == ["M_Rd", "V_Rd,c", "M_Rd"] and "4 Ø 12,5 + 2 Ø 9,3" in json.dumps(f, ensure_ascii=False)
    and "p. 30" in f["sources"][0] and f["extrait"]["case"] == [4, 1], (f["titre"], f["valeur"], f["extrait"]["case"]))

# ================================================================================================ 6. mutations
print("\n=== 6 · Mutations")
v = K.variante(cat, "SP 200-4X")[1]
memo = v["M_Rd"]
try:
    v["M_Rd"] = 52
    chk("mutation « M_Rd du catalogue changé » : la bande suit (aucune copie dans l'écran)", valeur(B.consulter("hourdis", {}))[0] == "52")
finally:
    v["M_Rd"] = memo
orig = K.verifier
try:
    K.verifier = lambda p, M_Ed=None, V_Ed=None, avec_chape=False: dict(orig(p, M_Ed, V_Ed, avec_chape), premiere=None)
    chk("mutation « le moteur ne désigne plus de variante » : l'étoile s'éteint (l'écran ne juge pas lui-même)",
        "★</i>" not in B.consulter("hourdis", {"M_Ed": "100", "V_Ed": "93"})["tableau"])
finally:
    K.verifier = orig
chk("après les mutations, la rubrique est revenue (51 kN·m)", valeur(B.consulter("hourdis", {})) == ("51", "kN·m"))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)
