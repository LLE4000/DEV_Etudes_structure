# -*- coding: utf-8 -*-
"""Le module « Catalogue des profilés » du site (api_web/catalogue.py) : ce que l'écran du site AFFICHE est ce que le
moteur extrait (acier/profiles/consultation.py, prouvé égal à l'écran Streamlit par tests/test_catalogue_moteur.py)
calcule — liste, cellules, compteur, fiche, comparaison, export ; chaque saisie et chaque refus ; les sélecteurs (retenu
✓, plus léger ★, familles) ; la fenêtre de calcul, dont CHAQUE formule écrite avec ses valeurs redonne la valeur du
catalogue ; les dessins à l'échelle (le contour dessiné est celui du calcul) ; l'échappement ; les notes abîmées.

    python3.12 tests/test_api_web_catalogue.py
"""
import copy
import html
import json
import math
import os
import re
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier.profiles import catalogue as CAT  # noqa: E402
from acier.profiles import consultation as CO  # noqa: E402
from acier.profiles import familles as F  # noqa: E402
from api_web import catalogue as M  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import dessin_catalogue as D  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:300]}]" if info and not cond else ""))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


def saisir(etat, *saisies):
    """Plusieurs saisies de suite ; rend (état, dernier message)."""
    e, msg = copy.deepcopy(etat), None
    for chemin, valeur in saisies:
        msg = M.modifier(e, chemin, valeur)
        if msg:
            break
    return e, msg


def r0(e):
    return e["recherches"][e["actif"]]


def texte(h):
    """Le texte d'un fragment HTML (balises ôtées, entités rendues)."""
    return html.unescape(re.sub(r"<[^>]+>", "", h))


def lignes_vues(h):
    """Les lignes du tableau de la vue : [[cellule, …]] (texte), en-tête exclu."""
    corps = re.search(r"<tbody>(.*?)</tbody>", h, re.S).group(1)
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", corps, re.S):
        out.append([texte(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)])
    return out


def vue(res, cle):
    return next((v for v in res["vues"] if v["cle"] == cle), None)


E0 = M.nouvel_etat()

# ====================================================================== 1. la note neuve : le tableau de l'écran
print("=== 1. Note neuve : le tableau ===")
reg = CO.reglages_initiaux()
r = r0(E0)
chk("réglages d'ouverture = ceux de l'écran (préférences enregistrées)", all(r[k] == reg[k] for k in reg), {k: r[k] for k in reg})
res = M.resultat(E0)
lst = CO.liste(r["familles"], "", r["filtres"], r["courants"], [], r["tri"], r["decroissant"])
cols = CO.colonnes_visibles(lst, r["colonnes"])
vt = vue(res, "tableau")
lv = lignes_vues(vt["svg"])
chk(f"le tableau montre les {M.LIGNES_DEFAUT} premières lignes de la liste de l'écran ({len(lst)})",
    [x[0] for x in lv] == [p["nom"] for p in lst[:M.LIGNES_DEFAUT]], [x[0] for x in lv][:3])
lt = CO.lignes_tableau(lst[:M.LIGNES_DEFAUT], cols)
ecarts = []
for ligne_vue, ligne in zip(lv, lt):
    attendu = [ligne["Profilé"], ligne["Famille"]] + [M._valeur(ligne[CO.entete(c)], c) for c in cols]
    if ligne_vue != attendu:
        ecarts.append((ligne_vue[:4], attendu[:4]))
chk(f"chaque cellule ({len(lt) * (len(cols) + 2)}) = la valeur arrondie de l'écran, écrite à décimales fixes", not ecarts, ecarts[:2])
tete = re.search(r"<thead>(.*?)</thead>", vt["svg"], re.S).group(1)
chk("les en-têtes portent le symbole (indices) et l'unité de chaque colonne",
    all(M.symbole(c) in tete for c in cols) and "cm⁴" in tete and "kg/m" in tete and "W<sub>pl,y</sub>" in tete)
chk("la légende du tableau est le compteur de l'écran",
    vt["legende"] == CO.compteur(len(lst), r["familles"], 0).replace("**", "") and f"<b>{len(lst)} profilés</b>" in res["phrase"])
chk("plus de 200 lignes : le pied le dit, avec le moyen d'en voir plus", f"Lignes 1 à {M.LIGNES_DEFAUT} sur {len(lst)}" in vt["svg"])
chk("la fiche s'ouvre sur le premier profilé (comme l'écran)", r["fiche"] == lst[0]["nom"] and "cat-v" in res["resultat"])
chk("résumé de la note : « Consultation », sans taux", res["resume"]["niveau"] == "i" and res["resume"]["taux"] is None
    and C.STATUTS["i"][1] in res["statut_html"])
chk("deux vues : le tableau (un bloc HTML) puis la section (un dessin)", [v["cle"] for v in res["vues"]] == ["tableau", "section"]
    and vt["svg"].startswith("<div") and vue(res, "section")["svg"].startswith("<svg"))
t0 = time.time()
for _ in range(5):
    M.resultat(E0)
chk("une réponse complète se calcule vite (< 0,3 s en CPython)", (time.time() - t0) / 5 < 0.3, round((time.time() - t0) / 5, 3))

# ====================================================================== 2. les saisies
print("\n=== 2. Saisies ===")
e, msg = saisir(E0, (["recherche"], "heb 3"))
r = r0(e)
attendus = [p["nom"] for p in CO.liste(r["familles"], "heb 3", r["filtres"], False, [], "", False)]
chk("recherche « heb 3 » : la liste de l'écran", not msg and [x[0] for x in lignes_vues(vue(M.resultat(e), "tableau")["svg"])] == attendus
    and attendus == ["HEB 300", "HEB 320", "HEB 340", "HEB 360"], attendus)
e, msg = saisir(E0, (["tri"], "masse"), (["ordre"], "decroissant"))
res = M.resultat(e)
noms = [x[0] for x in lignes_vues(vue(res, "tableau")["svg"])]
r = r0(e)
chk("tri par masse décroissante : l'ordre de l'écran (CAT.trier) ; le profilé ouvert (le plus léger) ajouté sous les 200 lignes",
    noms[:200] == [p["nom"] for p in CO.liste(r["familles"], "", r["filtres"], False, [], "masse", True)][:200] and noms[-1] == r["fiche"])
chk("la colonne triée vient juste après le nom, marquée ↓", re.search(r'<th class="fa">Famille</th><th class="n tri"[^>]*><span class="s">Masse<i>↓</i>', res["vues"][0]["svg"]) is not None)
e2, msg = saisir(E0, (["tri"], "Wt"))
res2 = M.resultat(e2)
chk("tri par une colonne NON cochée (W_t) : elle est montrée quand même, et le pied le dit",
    "W<sub>t</sub><i>↑</i>" in res2["vues"][0]["svg"] and "sans être cochée" in res2["vues"][0]["svg"]
    and "Wt" not in r0(e2)["colonnes"])
chk("l'export garde l'ordre et les colonnes de l'écran (sans la colonne de tri ajoutée)",
    "Wt" not in CO.texte_tableau(CO.lignes_tableau(M._vue(r0(e2))["liste"], M._vue(r0(e2))["cols"]))[:200])
e, msg = saisir(E0, (["ordre"], "en travers"))
chk("ordre inconnu : refusé, note intacte", msg and e == E0, msg)
e, msg = saisir(E0, (["tri"], "inconnu"))
chk("tri sur une propriété inconnue : refusé", msg and e == E0, msg)
e, _ = saisir(E0, (["familles", "aucune"], 1))
res = M.resultat(e)
chk("« Aucune » famille : tableau vide, message de l'écran et sa cause, pas de fiche, pas de vue Section",
    CO.MESSAGE_VIDE in texte(res["resultat"]) and "aucune famille cochée" in texte(res["vues"][0]["svg"])
    and r0(e)["fiche"] is None and [v["cle"] for v in res["vues"]] == ["tableau"])
e, _ = saisir(e, (["familles", "HEA"], True), (["familles", "IPE"], True))
chk("familles cochées dans le désordre : rangées comme les pastilles de l'écran", r0(e)["familles"] == ["IPE", "HEA"])
e, _ = saisir(e, (["familles", "toutes"], 1))
chk("« Toutes les familles » : toutes, groupe par groupe", r0(e)["familles"] == CO.toutes_familles())
e, msg = saisir(E0, (["familles", "XYZ"], True))
chk("famille inconnue : refusée", msg and e == E0)
e, _ = saisir(E0, (["courants"], True))
chk("séries courantes seulement : aucune famille « non courante » au tableau",
    all(CAT.par_nom()[p["nom"]]["courant"] for p in M._vue(r0(e))["liste"]))
e, _ = saisir(E0, (["colonnes", "Iw"], True), (["colonnes", "h"], False), (["colonnes", "Wt"], True))
chk("colonnes cochées : rangées dans l'ordre du catalogue", r0(e)["colonnes"] == CO.colonnes_ordonnees(set(r0(E0)["colonnes"]) - {"h"} | {"Iw", "Wt"}))
e, msg = saisir(E0, (["colonnes", "xx"], True))
chk("colonne inconnue : refusée", msg and e == E0)
for n in (100, 500, 0):
    e, _ = saisir(E0, (["lignes"], str(n)))
    nb = len(lignes_vues(vue(M.resultat(e), "tableau")["svg"]))
    chk(f"« Lignes : {n or 'toutes'} » : {min(n or 10 ** 6, len(M._vue(r0(e))['liste']))} lignes", nb == min(n or 10 ** 6, len(M._vue(r0(e))["liste"])), nb)
e, msg = saisir(E0, (["lignes"], "7"))
chk("un nombre de lignes hors des choix : refusé", msg and e == E0)
dernier = M._vue(r0(E0))["noms"][-1]
e, _ = saisir(E0, (["fiche"], dernier))
lv = lignes_vues(vue(M.resultat(e), "tableau")["svg"])
chk("le profilé ouvert au-delà des 200 lignes reste visible (ligne « … » puis lui)", lv[-1][0] == dernier and lv[-2][0].startswith("…"), lv[-2:])
e, msg = saisir(E0, (["fiche"], "IPE 9999"))
chk("ouvrir un profilé hors du tableau : refusé", msg and e == E0)

# les filtres numériques
e, msg = saisir(E0, (["filtres", "0", "cle"], "Iy"), (["filtres", "0", "min"], "1 500"), (["filtres", "0", "max"], "10000,5"))
r = r0(e)
chk("filtre I_y : « 1 500 » et « 10000,5 » lus comme des nombres", not msg and r["filtres"][0] == {"cle": "Iy", "min": 1500.0, "max": 10000.5}, r["filtres"][0])
liste = M._vue(r)["liste"]
chk("filtre I_y ≥ 1 500 et ≤ 10 000,5 : chaque profilé du tableau le satisfait, à la valeur exacte",
    liste and all(1500 <= p["Iy"] <= 10000.5 for p in liste) and len(liste) == len(CO.liste(r["familles"], "", r["filtres"], False, [], "", False)))
res = M.resultat(e)
leger = min(liste, key=lambda p: p["masse"])
chk("★ le plus léger qui satisfait les filtres : nommé au résultat et proposé au cadre Données (bouton qui l'ouvre)",
    f"<b>{leger['nom']}</b>" in res["donnees"] and f'data-chemin="fiche" data-valeur="{leger["nom"]}"' in res["donnees"]
    and leger["nom"] in texte(res["resultat"]) or r["fiche"] == leger["nom"])
e3, _ = saisir(e, (["fiche"], leger["nom"]))
chk("ouvert, le plus léger : le résultat le dit, plus de bouton", "est le plus léger qui satisfait les filtres" in texte(M.resultat(e3)["resultat"])
    and 'data-chemin="fiche" data-valeur' not in M.resultat(e3)["donnees"])
chk("sans borne, pas d'étoile au résultat (le plus léger de onze familles serait une petite cornière)",
    "plus léger" not in texte(M.resultat(E0)["resultat"]) and "cat-leger" not in M.resultat(E0)["donnees"])
e4, msg = saisir(e, (["filtres", "1", "cle"], "Iy"))
chk("une propriété filtrée deux fois : refusée, avec la raison (l'écran Streamlit ne garderait que la seconde)",
    msg and "déjà filtrée" in msg and "Streamlit" in msg and e4 == e, msg)
e4, msg = saisir(e, (["filtres", "0", "min"], "beaucoup"))
chk("une borne illisible : refusée", msg and e4 == e)
e4, msg = saisir(e, (["filtres", "0", "max"], ""))
chk("une borne effacée (vide) : plus de borne", not msg and r0(e4)["filtres"][0]["max"] is None)
e4, _ = saisir(e, (["filtres", "0", "-"], 1))
chk("« × » d'un filtre : il est vidé", r0(e4)["filtres"][0] == {"cle": "", "min": None, "max": None})
e4, _ = saisir(e, (["filtres", "effacer"], 1))
chk("« Effacer les filtres » : les trois filtres vides de l'écran", r0(e4)["filtres"] == CO.filtres_vides())
e4, msg = saisir(E0, (["filtres", "0", "cle"], "masse"), (["filtres", "0", "min"], "50"), (["filtres", "0", "max"], "20"))
chk("min > max : tableau vide, et la cause est nommée", "min > max" in texte(M.resultat(e4)["vues"][0]["svg"]))
e4, msg = saisir(E0, (["filtres", "5", "cle"], "Iy"))
chk("un quatrième filtre : refusé", msg and e4 == E0)
e4, _ = saisir(E0, (["recherche"], "HEB"), (["courants"], True), (["tri"], "Iy"), (["filtres", "0", "cle"], "A"), (["reglages", "retablir"], 1))
r = r0(e4)
chk("« Rétablir les réglages livrés » : les défauts livrés de l'écran ; la recherche reste (comme l'écran)",
    all(r[k] == v for k, v in CO.reglages_livres().items()) and r["recherche"] == "HEB")

# la comparaison
e, _ = saisir(E0, (["recherche"], "300"))
for n in ("IPE 300", "HEA 300", "HEB 300"):
    M.choisir_section(e, "compare", n)
res = M.resultat(e)
vc = vue(res, "comparaison")
chk("trois profilés comparés : la vue « Comparaison · 3 », sections à la même échelle", vc and vc["court"] == "Comparaison · 3"
    and vc["svg"].count('fill="url(#cat_comp_h)"') == 3)
comp = [CAT.par_nom()[n] for n in ("IPE 300", "HEA 300", "HEB 300")]
lignes_c = CO.comparaison(comp, CO.colonnes_comparaison(M._vue(r0(e))["cols"]))
lv = lignes_vues(vc["svg"])
chk("le tableau de comparaison = celui de l'écran (lignes, valeurs écrites)",
    [x[1:] for x in lv] == [vals for _c, _e, vals in lignes_c] and len(lv) == len(lignes_c), (lv[:2], lignes_c[:2]))
chk("les comparés sont numérotés dans le tableau", re.search(r'IPE 300<span class="cat-k">1</span>', vue(res, "tableau")["svg"]) is not None)
e2, _ = saisir(e, (["recherche"], "HEB 300"))
chk("une recherche qui chasse deux comparés : ils quittent la comparaison (comme l'écran)", r0(e2)["compare"] == ["HEB 300"]
    and vue(M.resultat(e2), "comparaison") is None)
e2, msg = saisir(e, (["fiche"], "IPE 300"), (["compare", "+"], 1))
chk("« Comparer » le profilé ouvert déjà comparé : refusé", msg and "déjà" in msg)
autres = [n for n in M._vue(r0(e))["noms"] if n not in r0(e)["compare"]]
for n in autres[:3]:
    M.choisir_section(e, "compare", n)
chk("au-delà de cinq (max_selections de l'écran), le plus ancien laisse sa place",
    r0(e)["compare"] == ["HEA 300", "HEB 300"] + autres[:3], r0(e)["compare"])
e2, msg = saisir(e, (["fiche"], autres[3]), (["compare", "+"], 1))
chk("« + » sur cinq comparés : refusé", msg and "5" in msg, msg)
e2, _ = saisir(e, (["compare", "0", "-"], 1))
chk("« × » d'un comparé : retiré", r0(e2)["compare"] == ["HEB 300"] + autres[:3])
e2, _ = saisir(e, (["compare", "effacer"], 1))
chk("« Vider la comparaison »", r0(e2)["compare"] == [])

# le profilé personnalisé
e, msg = saisir(E0, (["perso", "dims", "PRS", "h"], "600"), (["perso", "dims", "PRS", "tw"], "10"), (["perso", "+"], 1))
p = CO.calculer_perso("PRS", {"h": 600.0, "b": 200.0, "tw": 10.0, "tf": 12.0})
r = r0(e)
vp = M._vue(r)
chk("PRS 600x200x10x12 : calculé par le moteur de l'écran (chaque grandeur), au tableau, fiche ouverte",
    not msg and vp["p"] == p and r["fiche"] == "PRS 600x200x10x12" and vp["liste"][-1]["nom"] == "PRS 600x200x10x12")
res = M.resultat(e)
chk("la fiche d'un personnalisé : pastille « personnalisé », toutes ses propriétés", "tag perso" in res["resultat"]
    and all(v in texte(res["resultat"]) for _g, props in CO.fiche(p) for _c, _l, v, _u, _a in props))
chk("le compteur compte le personnalisé (« · 1 personnalisé »)", "1 personnalisé" in res["phrase"])
e2, msg = saisir(e, (["perso", "+"], 1))
chk("le même nom ajouté deux fois : remplacé, toujours un seul", not msg and len(r0(e2)["persos"]) == 1)
for dims, attendu in (({"h": "24", "tf": "12"}, "Dimensions impossibles : h doit dépasser 2·tf"),):
    e2, msg = saisir(E0, *[(["perso", "dims", "PRS", k], v) for k, v in dims.items()], (["perso", "+"], 1))
    chk(f"PRS h = 2·t_f : refusé avec le message de l'écran (« {attendu} »)", msg == attendu, msg)
e2, msg = saisir(E0, (["perso", "forme"], "CAISSON"), (["perso", "dims", "CAISSON", "t"], "100"), (["perso", "dims", "CAISSON", "h"], "200"),
                 (["perso", "+"], 1))
chk("caisson 2·t = h : refusé avec le message de l'écran", msg == "Dimensions impossibles : t doit être inférieure à la demi-dimension", msg)
e2, msg = saisir(E0, (["perso", "dims", "PRS", "tf"], "0"))
chk("une dimension sous 0,1 mm (l'écran la remonterait en silence) : refusée avec un message", msg and "0,1" in msg and e2 == E0, msg)
e2, msg = saisir(E0, (["perso", "dims", "PRS", "tw"], "300"))
chk("PRS t_w ≥ b : l'avertissement s'affiche AVANT le calcul", not msg and "ce n'est plus un I" in texte(M.resultat(e2)["donnees"]))
e2, msg = saisir(e2, (["perso", "+"], 1))
chk("PRS t_w ≥ b : calculé comme l'écran (c_f/t_f négatif), et le résultat l'écrit en rouge",
    not msg and round(M._vue(r0(e2))["p"]["cf_tf"], 2) == -4.17 and 'class="msg grave"' in M.resultat(e2)["resultat"])
e2, msg = saisir(E0, (["perso", "forme"], "ZZZ"))
chk("forme inconnue : refusée", msg and e2 == E0)
e3, _ = saisir(e, (["persos", "0", "-"], 1))
chk("« × » d'un personnalisé : retiré du tableau, la fiche passe au premier", r0(e3)["persos"] == [] and r0(e3)["fiche"] == M._vue(r0(E0))["noms"][0])
for fam in CO.formes_perso():
    e3, msg = saisir(E0, (["perso", "forme"], fam), (["perso", "+"], 1))
    pp = M._vue(r0(e3))["p"]
    chk(f"forme {fam} aux dimensions proposées par l'écran : calculée, dessinée", not msg and pp["famille"] == fam
        and vue(M.resultat(e3), "section")["svg"].startswith("<svg"), msg)

# l'export
e, _ = saisir(E0, (["recherche"], "HEB"), (["export", "tableau"], True), (["export", "fiche"], True))
res = M.resultat(e)
v_ = M._vue(r0(e))
tsv = CO.texte_tableau(CO.lignes_tableau(v_["liste"], v_["cols"]), "\t")
zone = re.search(r'<textarea data-chemin="export/texte_tableau"[^>]*>(.*?)</textarea>', res["donnees"], re.S)
chk("export du tableau : le texte des cellules de l'export CSV de l'écran, en tabulations (pour Excel)",
    zone and html.unescape(zone.group(1)) == tsv and tsv.replace("\t", ";") == CO.texte_tableau(CO.lignes_tableau(v_["liste"], v_["cols"]), ";"))
zone = re.search(r'<textarea data-chemin="export/texte_fiche"[^>]*>(.*?)</textarea>', res["donnees"], re.S)
chk("export de la fiche : le JSON de l'écran", zone and html.unescape(zone.group(1)) == CO.fiche_json(v_["p"]))
e2, msg = saisir(e, (["export", "texte_tableau"], "n'importe quoi"))
chk("retoucher le texte exporté ne change rien", not msg and e2 == e)
chk("sans case cochée, aucun texte d'export (la réponse reste légère)", "texte_tableau" not in M.resultat(E0)["donnees"])

# ====================================================================== 3. le résultat : la fiche
print("\n=== 3. La fiche de chaque famille ===")
mauvaises = []
for fam in CO.toutes_familles():
    serie = [q for q in CAT.profils() if q["famille"] == fam]
    for q in (serie[0], serie[len(serie) // 2], serie[-1]):
        e, _ = saisir(E0, (["familles", "toutes"], 1), (["fiche"], q["nom"]))
        res = M.resultat(e)
        t = res["resultat"]
        tab = re.search(r'<div class="cat-fiche">(.*)</table>', t, re.S).group(1)
        vus = [(texte(a), texte(b), texte(c_)) for a, b, c_ in re.findall(r'<td class="s"[^>]*>(.*?)</td><td class="n">(.*?)</td><td class="u">(.*?)</td>', tab, re.S)]
        att = [(texte(M.symbole(c)), v, u) for _g, props in CO.fiche(q) for c, _l, v, u, _a in props]
        if vus != att or f"<b>{q['nom']}</b>" not in t or (CO.methode_it(q) and M._methode_txt(CO.methode_it(q)) not in texte(t)):
            mauvaises.append((q["nom"], vus[:2], att[:2]))
        if f"{M._masse(q['masse'])} kg/m · A = {C.fr(q['A'], 2)} cm²" not in t:
            mauvaises.append((q["nom"], "en-tête"))
chk(f"la fiche (en-tête, TOUTES les propriétés et leurs unités, méthode de I_t) = celle de l'écran — 3 profilés × "
    f"{len(CO.toutes_familles())} familles", not mauvaises, mauvaises[:2])

# ====================================================================== 4. la fenêtre de calcul : chaque formule redonne la valeur
print("\n=== 4. Fenêtre de calcul ===")


def evaluer(t):
    """Une formule écrite avec ses valeurs (« (46 − 3,8)/2 − 5 », « √(80,14/7,64) ») évaluée."""
    s = t.replace("m²", "").replace("kg/m³", "").replace("10⁻⁴", "1e-4")
    for a, b in (("×", "*"), ("−", "-"), ("π", "pi"), ("√", "sqrt"), ("²", "**2"), ("⁴", "**4"), ("³", "**3")):
        s = s.replace(a, b)
    for _ in range(3):
        s = re.sub(r"(\d)[   ](\d)", r"\1\2", s)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return eval(s, {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi})  # noqa: S307 — nos propres chaînes


n_fx, pires, ratees = 0, [], []
tous = CAT.profils() + [CO.calculer_perso(f, CO.dims_defaut(f)) for f in CO.formes_perso()]
for q in tous:
    for _titre, lignes in M.formules(q):
        for li in lignes:
            sub = li["parts"][2] if len(li["parts"]) > 2 else ""
            if not re.match(r"^[\d(√π]", sub):
                continue
            try:
                x = evaluer(sub)
            except Exception as e:  # noqa: BLE001
                ratees.append((q["nom"], li["cle"], sub, str(e)))
                continue
            n_fx += 1
            v = q[li["cle"]]
            ec = abs(x - v) / max(abs(v), 1e-9)
            if ec > 2e-3:
                pires.append((round(ec, 5), q["nom"], li["cle"], sub, x, v))
chk(f"CHAQUE formule écrite avec ses valeurs ({n_fx} lignes, {len(tous)} profilés) redonne la valeur du catalogue à 0,2 % près",
    n_fx > 13000 and not pires and not ratees, (pires[:2], ratees[:2]))
chk("le résultat écrit au bout de chaque formule est la valeur de la fiche, avec son unité",
    all(li["res"] == CAT.formater(q[li["cle"]], li["cle"]) + M._u(li["cle"]) for q in tous[::40] for _t, ls in M.formules(q) for li in ls))
q = CAT.par_nom()["L 100x65x8"]
iu = next(li for _t, ls in M.formules(q) for li in ls if li["cle"] == "Iv")["parts"][2]
chk("un I_yz négatif s'écrit entre parenthèses avant d'être élevé au carré (sinon « −42,2² » se lirait −(42,2²))",
    "(−" in iu and "+ −" not in iu, iu)
e, _ = saisir(E0, (["familles", "toutes"], 1), (["fiche"], "SHS 100x100x5"))
chk("un creux carré n'a pas de bloc « axes principaux » (I_y = I_z : axes quelconques)",
    "Axes principaux" not in "".join(t for t, _ in M.formules(CAT.par_nom()["SHS 100x100x5"])))
res = M.resultat(E0)
calc = set(re.findall(r'data-calcul="([^"]+)"', res["resultat"]))
chk("les lignes « Profilé » et « Méthode et sources » ouvrent leur fenêtre", calc == {"fiche", "methode"}
    and 'id="ck-fiche"' in res["volets"] and 'id="ck-methode"' in res["volets"])
chk("la fenêtre de calcul n'a pas de tableau de vérification (un catalogue ne vérifie rien)", 'class="vtab"' not in res["volets"])
m = CO.methode_et_sources()
chk("« Méthode et sources » : sources, conventions (en indices), calcul, contrôle, chaque définition, chaque famille",
    all(C.esc(s) in res["volets"] for s in m["sources"]) and C.esc(m["calcul"]) in res["volets"] and C.esc(m["controle"]) in res["volets"]
    and "W<sub>el</sub>" in res["volets"] and all(C.esc(t) in res["volets"] for _c, _l, t in m["definitions"])
    and all(C.esc(d) in res["volets"] for _f, _l, _n, d in m["familles"]))

# ====================================================================== 5. les sélecteurs
print("\n=== 5. Sélecteurs ===")
e, _ = saisir(E0, (["filtres", "0", "cle"], "Iy"), (["filtres", "0", "min"], "2000"), (["fiche"], "HEB 200"))
sel = M.selecteur(e, "fiche")
rows = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)"', sel)
liste = M._vue(r0(e))["liste"]
chk("sélecteur « fiche » : la liste du tableau, dans son ordre", [x[2] for x in rows] == [p["nom"] for p in liste])
chk("le retenu ✓ est la fiche ouverte, l'optimal ★ le plus léger qui satisfait les filtres",
    [x[2] for x in rows if x[0]] == ["HEB 200"] and [x[2] for x in rows if x[1]] == [min(liste, key=lambda p: p["masse"])["nom"]]
    and "le plus léger qui satisfait les filtres" in sel)
chk("la colonne de grandeur est la masse (kg/m), sans pastille de taux", "<span>kg/m</span>" in sel and 'class="chip' not in sel)
chk("les familles du tableau en pastilles", all(f'data-famille="{f}"' in sel for f in {p["famille"] for p in liste}))
sel2 = M.selecteur(e, "fiche", ["HEA"])
chk("une pastille de famille resserre la liste (HE A seulement)", set(re.findall(r'data-choix="([A-Z]+)', sel2)) == {"HEA"})
r_ = req(module="catalogue", action="choisir", etat=e, role="fiche", nom="HEA 300")
chk("un clic dans le sélecteur ouvre la fiche", r_["ok"] and r0(r_["etat"])["fiche"] == "HEA 300")
r_ = req(module="catalogue", action="choisir", etat=e, role="fiche", nom="IPE 9999")
chk("un nom hors du tableau : rien ne change", r_["ok"] and r_["etat"] == e)
selc = M.selecteur(saisir(E0, (["recherche"], "IPE 3"))[0], "compare")
chk("sélecteur « compare » : « un clic ajoute (au plus 5) »", "au plus 5" in selc)

# ====================================================================== 6. les dessins
print("\n=== 6. Dessins ===")
pb = []
for fam in list(CO.toutes_familles()) + CO.formes_perso():
    q = next((x for x in CAT.profils() if x["famille"] == fam), None) or CO.calculer_perso(fam, CO.dims_defaut(fam))
    svg = D.section("t", q)
    anneaux = F.contours(fam, {k: float(v) for k, v in q["dims"].items()})
    chemin = re.search(r'<path d="([^"]+)" fill="#E4E9F0"', svg).group(1)
    pts = re.findall(r"[ML](-?[\d.]+),(-?[\d.]+)", chemin)
    vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    if len(pts) != sum(len(a) for a in anneaux) or chemin.count("M") != len(anneaux):
        pb.append((q["nom"], "contour"))
    x0, z0 = anneaux[0][0]
    if abs(float(pts[0][0]) - x0) > 0.006 or abs(float(pts[0][1]) + z0) > 0.006:
        pb.append((q["nom"], "point", pts[0], (x0, z0)))
    if not all(vb[0] <= float(x) <= vb[0] + vb[2] and vb[1] <= float(y) <= vb[1] + vb[3] for x, y in pts):
        pb.append((q["nom"], "hors cadre"))
    if "nan" in svg.lower() or "inf" in svg.lower():
        pb.append((q["nom"], "nan"))
    d = {k: float(v) for k, v in q["dims"].items()}
    cote = f"Ø {D._n(d['d'])}" if F.FAMILLES[fam]["forme"] in ("creux_rond", "plein_rond") else f"b = {D._n(d['b'])}"
    if cote not in svg:
        pb.append((q["nom"], "cote"))
chk("la section de chaque famille (21 du catalogue + 5 personnalisées) : le contour DU CALCUL (mêmes sommets, en mm), "
    "dans son cadre, cotée", not pb, pb[:3])
q = CAT.par_nom()["UPN 200"]
svg = D.section("t", q)
chk("UPN : le centre de cisaillement S est repéré, à y_s du centre de gravité", f'cx="{D._f(float(q["yc"]) + float(q["y_s"]))}"' in svg and ">S<" in svg)
q = CAT.par_nom()["L 100x65x8"]
chk("cornière : les axes principaux u et v sont tracés", ">u<" in D.section("t", q) and ">v<" in D.section("t", q))
chk("la légende de la section : les dimensions en mm, à la française", D.legende(CAT.par_nom()["UPN 200"]) == "h 200 · b 75 · tw 8,5 · tf 11,5 · r1 12 · r2 6 mm")

# ====================================================================== 7. la note : éléments, relecture, échappement
print("\n=== 7. Note, relecture, échappement ===")
e = copy.deepcopy(E0)
M.gerer_elements(e, "ajouter")
M.gerer_elements(e, "dupliquer")
chk("ajouter puis copier : trois recherches, la copie ouverte et nommée", len(e["recherches"]) == 3 and e["actif"] == 2
    and e["recherches"][2]["nom"].endswith("(copie)"))
M.gerer_elements(e, "supprimer", 0)
chk("supprimer la première : deux, la bonne reste ouverte", len(e["recherches"]) == 2 and e["actif"] == 1)
M.gerer_elements(e, "supprimer", 0)
M.gerer_elements(e, "supprimer", 0)
chk("la dernière ne se supprime pas", len(e["recherches"]) == 1)
e, _ = saisir(E0, (["recherche"], "IPE"), (["filtres", "0", "cle"], "Iy"), (["filtres", "0", "min"], "500"), (["perso", "+"], 1),
              (["export", "fiche"], True))
chk("relecture (.stab) stable : l'état relu est l'état", M.normaliser(json.loads(json.dumps(e))) == e)
for abime, quoi in (({"recherches": [{"filtres": [{"cle": "zz", "min": 1, "max": None}]}]}, "propriété inconnue"),
                    ({"recherches": [{"filtres": [{"cle": "Iy", "min": 1}, {"cle": "Iy", "max": 2}]}]}, "même propriété"),
                    ({"recherches": [{"persos": [{"famille": "PRS", "dims": {"h": 24, "b": 200, "tw": 8, "tf": 12}}]}]}, "2·tf"),
                    ({"recherches": [{"persos": [{"famille": "PRS", "dims": {"h": 400, "b": 200, "tw": 0, "tf": 12}}]}]}, "0,1"),
                    ({"recherches": [{"filtres": [{"cle": "Iy", "min": "beaucoup"}]}]}, "nombre"),
                    ({"recherches": "abîmé"}, "liste")):
    try:
        M.normaliser(abime)
        chk(f"note abîmée ({quoi}) : refusée", False)
    except ValueError as x:
        chk(f"note abîmée ({quoi}) : erreur lisible", quoi in str(x), str(x))
r_ = req(module="catalogue", action="calculer", etat={"recherches": [{"filtres": [{"cle": "zz"}]}]})
chk("une note abîmée rend une erreur au navigateur, jamais une exception", r_["ok"] is False and "propriété inconnue" in r_["erreur"])
PIEGE = '<img src=x onerror=alert(1)>'
e, _ = saisir(E0, (["nom"], PIEGE), (["recherche"], PIEGE))
res = M.resultat(e)
frag = res["resultat"] + res["volets"] + res["donnees"] + res["elements"] + res["phrase"] + "".join(v["svg"] for v in res["vues"])
chk("le nom et la recherche piégés sont échappés partout", "<img" not in frag and "&lt;img" in res["donnees"])
e5, _ = saisir(E0, (["recherche"], "300"), (["filtres", "0", "cle"], "Iy"), (["filtres", "0", "min"], "500"), (["perso", "+"], 1),
               (["export", "tableau"], True), (["export", "fiche"], True))
for n in ("IPE 300", "HEB 300"):
    M.choisir_section(e5, "compare", n)
r5 = M.resultat(e5)
frag5 = r5["resultat"] + r5["volets"] + r5["donnees"] + r5["elements"] + r5["phrase"] + "".join(v["svg"] for v in r5["vues"])
frag5 += M.selecteur(e5, "fiche") + M.selecteur(e5, "compare")
chk("aucun fragment (vues, fenêtres, sélecteurs, export) ne porte de script ni de « on…= »", not re.search(r"<script|\son[a-z]+\s*=", frag5, re.I))
chk("la réponse est du JSON, l'état aussi", json.loads(json.dumps(res)) == res)

# ====================================================================== 8. relecture : refus propres, ergonomie
print("\n=== 8. Relecture : refus propres et ergonomie ===")
JARGON = re.compile(r"index|literal|out of range|object|Error|NoneType|not iterable", re.I)
mauvais = []
for chemin in (["compare", "9", "-"], ["persos", "0", "-"], ["filtres", "x", "cle"], ["filtres", "²", "cle"], [], "abc", None,
               ["compare", "-1", "-"], ["persos", "0"], ["familles", ["x"]], ["filtres", "7", "min"]):
    out = req(module="catalogue", action="modifier", etat=E0, chemin=chemin, valeur="Iy")
    if not out["ok"] or not out["message"] or JARGON.search(out["message"]) or out["etat"] != E0:
        mauvais.append((chemin, out.get("message") or out.get("erreur")))
chk("un chemin que l'écran n'envoie pas (rang absent, « ² », vide, texte) : refusé en français, note intacte",
    not mauvais, mauvais[:3])
e, msg = saisir(E0, (["perso", "dims", "PRS", "h"], "1e300"))
e2, msg2 = saisir(E0, (["perso", "dims", "PRS", "h"], "10 000"), (["perso", "+"], 1))
chk("dimension au-delà de 10 m : refusée (l'écran Streamlit calcule jusqu'à déborder : 1e300 mm le fait tomber) ; 10 m admis",
    msg == "h doit valoir au plus 10 000 mm" and e == E0 and not msg2 and r0(e2)["fiche"] == "PRS 10000x200x8x12", (msg, msg2))
for abime, quoi in (({"recherches": [{"persos": [{"famille": "PRS", "dims": {"h": float("inf"), "b": 200, "tw": 8, "tf": 12}}]}]}, "nombre fini"),
                    ({"recherches": [{"persos": [{"famille": "PRS", "dims": {"h": 20000, "b": 200, "tw": 8, "tf": 12}}]}]}, "au plus 10 000 mm"),
                    ({"recherches": [{"filtres": [{"cle": "Iy", "min": float("nan")}]}]}, "nombre fini"),
                    ({"recherches": [{"filtres": [{"cle": "Iy", "max": float("inf")}]}]}, "nombre fini"),
                    ({"recherches": [{"familles": "IPE"}]}, "une liste est attendue"),
                    ({"recherches": [{"colonnes": 5}]}, "une liste est attendue")):
    out = json.loads(R.traiter(json.dumps({"module": "catalogue", "action": "calculer", "etat": abime}, allow_nan=True)))
    chk(f"note abîmée ({quoi}) : erreur lisible, pas d'exception Python", out["ok"] is False and quoi in out["erreur"]
        and out["erreur"].startswith("ValueError"), out.get("erreur"))
e = M.normaliser({"recherches": [{"perso": {"forme": "PRS", "dims": {"PRS": {"h": float("inf"), "b": 1e300, "tw": 9, "tf": 12}}}},
                                 {"compare": ["IPE 80", "IPE 80", "IPE 100"]}]})
chk("note abîmée : formulaire hors bornes → valeurs proposées ; comparaison sans doublon",
    e["recherches"][0]["perso"]["dims"]["PRS"] == {"h": 400.0, "b": 200.0, "tw": 9.0, "tf": 12.0}
    and e["recherches"][1]["compare"] == ["IPE 80", "IPE 100"], e["recherches"][0]["perso"]["dims"]["PRS"])
# une borne se vide au clavier : le champ part comme texte (l'établi refuse un nombre vide), garde son unité
e, _ = saisir(E0, (["filtres", "0", "cle"], "Iy"), (["filtres", "0", "min"], "1 500"), (["filtres", "0", "max"], "9000"))
champ = re.search(r'<div class="field"><input data-chemin="filtres/0/min"[^>]*>(<span class="u">[^<]*</span>)?', M.resultat(e)["donnees"])
chk("borne de filtre : champ facultatif (texte, « — » quand vide), clavier décimal, unité cm⁴",
    champ and 'data-sorte="texte"' in champ.group(0) and 'placeholder="—"' in champ.group(0) and 'inputmode="decimal"' in champ.group(0)
    and "cm⁴" in champ.group(0), champ and champ.group(0))
e2, msg = saisir(e, (["filtres", "0", "min"], ""))
chk("borne effacée au clavier : la borne min tombe, la borne max reste", not msg and r0(e2)["filtres"][0] == {"cle": "Iy", "min": None, "max": 9000.0})
# le sélecteur « Comparer » : un second clic retire
e, _ = saisir(E0, (["recherche"], "IPE"))
for n in ("IPE 100", "IPE 120", "IPE 100"):
    M.choisir_section(e, "compare", n)
chk("sélecteur « Comparer » : un clic sur un profilé coché ✓ le retire", r0(e)["compare"] == ["IPE 120"], r0(e)["compare"])
chk("sélecteur « Comparer » : le sous-titre le dit", "ajoute ou retire" in M.selecteur(e, "compare"))
chk("comparaison à moins de deux profilés : l'aide de l'écran (« Choisir au moins deux… ») est écrite",
    CO.AIDE_COMPARER in M.resultat(e)["donnees"] and CO.AIDE_COMPARER not in M.resultat(saisir(e, (["fiche"], "IPE 200"), (["compare", "+"], 1))[0])["donnees"])
# les cases des colonnes portent l'unité (l'écran : « Iy (cm⁴) »)
d0 = M.resultat(E0)["donnees"]
chk("cases des colonnes : symbole ET unité (l'en-tête « Libellé (unité) » de l'écran)",
    all(f'data-chemin="colonnes/{c}"' in d0 for c in CO.cles_colonnes())
    and re.search(r'data-chemin="colonnes/Iy"[^>]*><span>I<sub>y</sub><small>cm⁴</small></span>', d0) is not None)
# les titres de la toile sont écrits en capitales par l'établi : « × » entre deux chiffres
e, _ = saisir(E0, (["familles", "toutes"], 1), (["fiche"], "L 100x100x10"))
chk("titre de la vue Section : « L 100×100×10 » (en capitales, « x » deviendrait « X »)",
    vue(M.resultat(e), "section")["titre"] == "Section L 100×100×10")
# I_t des PRS : Σ b·t³/3 surestime (non conservatif) — affiché comme l'écran, et ÉCRIT
e, _ = saisir(E0, (["perso", "+"], 1))
t = texte(M.resultat(e)["resultat"])
chk("PRS : la réserve sur I_t (Σ b·t³/3 surestime, non conservatif pour le déversement) est écrite au résultat",
    "surestime" in t and "déversement" in t and CO.reserve_torsion(M._vue(r0(e))["p"]))
chk("PRS : la réserve sur A_v,z (formule des laminés contre η·h_w·t_w d'un soudé) est écrite au résultat, en indices",
    "formule des laminés" in t and "A<sub>v,z</sub> suit" in M.resultat(e)["resultat"])
chk("PRS t_w ≥ b : l'avertissement écrit c_f/t_f et A_v,z en indices (pas de « _ » brut)",
    "c<sub>f</sub>/t<sub>f</sub>" in M.resultat(saisir(E0, (["perso", "dims", "PRS", "tw"], "300"))[0])["donnees"]
    and "_" not in texte(M.resultat(saisir(E0, (["perso", "dims", "PRS", "tw"], "300"), (["perso", "+"], 1))[0])["resultat"]))
chk("profilé laminé (Saint-Venant) : pas de réserve", CO.reserve_torsion(CAT.par_nom()["IPE 300"]) is None
    and "surestime" not in texte(M.resultat(E0)["resultat"]))
# un dessin qui échoue ne cache pas la fiche
_sec, _vig = D.section, D.vignette
D.section = D.vignette = lambda *a, **k: (_ for _ in ()).throw(ValueError("contour impossible"))
try:
    out = req(module="catalogue", action="calculer", etat=E0)
finally:
    D.section, D.vignette = _sec, _vig
chk("un dessin qui échoue : la réponse reste complète (« Dessin indisponible », la fiche est là)",
    out["ok"] and "Dessin indisponible" in vue(out, "section")["svg"] and "cat-fiche" in out["resultat"], out.get("erreur"))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
