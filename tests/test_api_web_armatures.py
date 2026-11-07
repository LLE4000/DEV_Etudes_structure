# -*- coding: utf-8 -*-
"""L'API du site pour le tableau des armatures (api_web/armatures.py), vérifiée en Python ordinaire.

- la vue « Tableau » du site porte, case par case, les 180 nombres de l'image que l'écran Streamlit affiche (la
  transcription est celle de tests/test_armatures_moteur.py, qui la relit dans les pixels : elle est lue ici dans ce
  fichier, sans copie) et son en-tête ;
- ce que l'écran affiche est ce que le moteur calcule : la disposition retenue (section, masse, entraxe), la plus
  économique, les taux A_s,req / A_s et leur couleur, les cases qui suffisent, la retenue (✓) et l'optimale (★) ;
- la grande fenêtre de calcul écrit les formules avec leurs valeurs substituées ;
- chaque saisie va au bon endroit ; une saisie hors domaine est refusée avec un message, et le repère reste intact ;
- le sélecteur est trié par section, marque la retenue et la plus économique ; un clic retient la disposition (barres,
  colonnes du tableau, espacement libre) ;
- les repères de la note s'ajoutent, se copient, se suppriment (jamais le dernier) ; la note est une CONSULTATION ;
- un nom saisi est échappé partout où il s'affiche ; les dessins sont à l'échelle.

    python3 tests/test_api_web_armatures.py
"""
import ast
import copy
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import armatures as A  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.armatures import moteur as MO  # noqa: E402
from beton.armatures import tableau as T  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def _fines(x):
    """L'espace fine insécable des milliers (C.fr) lue comme une espace : « 1 018 » s'écrit simplement ici."""
    if isinstance(x, str):
        return x.replace("\u202f", " ")
    if isinstance(x, dict):
        return {k: _fines(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_fines(v) for v in x]
    return x


def req(**k):
    return _fines(json.loads(R.traiter(json.dumps({"module": "armatures", **k}))))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée."""
    return req(action="modifier", etat=etat, chemin=chemin.split("/"), valeur=valeur)


def transcription():
    """La transcription de l'image, lue dans tests/test_armatures_moteur.py (sa source unique, relue dans les pixels)."""
    with open(os.path.join(RACINE, "tests", "test_armatures_moteur.py"), encoding="utf-8") as fh:
        arbre = ast.parse(fh.read())
    for n in arbre.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRANSCRIPTION" for t in n.targets):
            return [li.split(";") for li in ast.literal_eval(n.value).splitlines()]
    raise AssertionError("TRANSCRIPTION introuvable")


LIGNES = transcription()
CELLULE = re.compile(r'<td class="([^"]*)" title="([^"]*)">(?:<i class="mk">([★✓])</i>)?([\d ]+)</td>')
MASSE = re.compile(r'<td class="m">([\d,]+)</td>')
RANGEE = re.compile(r'<tr class="(cur)?"><th class="d( hc)?">(\d+)</th>(.*?)</tr>', re.S)
LIGNE = re.compile(r'<div class="el( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>(.*?)</div><div class="l2">.*?<span class="pct (\w)">([^<]+)</span>', re.S)


def nombre(s):
    return int(s.replace(" ", ""))


def rangees(html):
    """Les rangées de la vue Tableau : {Ø: {"cur", "hc", "cases": [(classes, titre, marque, valeur)] × 14, "masse"}}."""
    out = {}
    for cur, hc, d, corps in RANGEE.findall(html):
        cases = [(set(c.split()), t, mk, nombre(v)) for c, t, mk, v in CELLULE.findall(corps)]
        out[int(d)] = {"cur": bool(cur), "hc": bool(hc), "cases": cases, "masse": MASSE.findall(corps)}
    return out


def vue(r, cle):
    return next(v for v in r["vues"] if v["cle"] == cle)


# ---------------------------------------------------------------------------------------------------------- 1. neuve
print("=== 1 · Une note neuve : consultation, le tableau de l'image")
r = req(action="nouveau")
chk("nouveau : la requête aboutit", r["ok"], r.get("erreur"))
etat0 = r["etat"]
chk("l'état : un repère « Repère 1 », les défauts du moteur", etat0 == {"version": 1, "reperes": [MO.normaliser(MO.nouveau_besoin())], "actif": 0}, etat0)
chk("le résumé : une CONSULTATION (niveau i, sans taux), le repère nommé",
    r["resume"]["niveau"] == "i" and r["resume"]["taux"] is None and r["resume"]["statut"] == "Consultation"
    and r["resume"]["gouvernant"] == "Repère 1 · 4Ø12 · 452 mm²", r["resume"])
chk("la pastille d'état dit « Consultation »", "Consultation" in r["statut_html"])
chk("trois vues : le tableau (bloc HTML), la nappe d'un mètre et les diamètres (SVG)",
    [(v["cle"], v["svg"][:4]) for v in r["vues"]] == [("tableau", "<div"), ("nappe", "<svg"), ("diametres", "<svg")])
chk("la vue Tableau porte la légende de l'image de l'écran Streamlit", vue(r, "tableau")["legende"] == T.LEGENDE_IMAGE)
tab = vue(r, "tableau")["svg"]
chk("… et la porte VISIBLE dans le tableau (le cadre de la toile n'écrit, pour la première vue, que la phrase)",
    f"<b>{T.LEGENDE_IMAGE}.</b>" in tab)
chk("la vue Tableau a pour titre celui de l'écran Streamlit : « Tableau des sections d’armatures »", vue(r, "tableau")["titre"] == T.TITRE_ECRAN)
rg = rangees(tab)
chk("la vue Tableau : 12 rangées, de Ø6 à Ø40, dans l'ordre de l'image", list(rg) == [int(li[0]) for li in LIGNES], list(rg))
ecarts = []
for li in LIGNES:
    d = int(li[0])
    vals = [c[3] for c in rg.get(d, {}).get("cases", [])]
    att = [int(x) for x in li[1:11]] + [int(x) for x in li[13:17]]
    if vals != att:
        ecarts.append((d, vals, att))
    if rg.get(d, {}).get("masse") != [li[11]]:
        ecarts.append((d, "masse", rg.get(d, {}).get("masse"), li[11]))
chk("chaque case de la vue Tableau = la valeur de l'IMAGE (120 sections, 12 masses, 48 sections par mètre)", not ecarts, ecarts[:3])
chk("l'en-tête : n = 1 … 10 et 100,0 … 10,0 cm ; espacements fixes 15, 30, 35, 40 cm ; masse kg/m",
    all(f"<b>{n}</b><small>{e}</small>" in tab for n, e in zip(range(1, 11), "100,0 50,0 33,3 25,0 20,0 16,7 14,3 12,5 11,1 10,0".split()))
    and all(f"<b>{e}</b><small>cm</small>" in tab for e in (15, 30, 35, 40)) and "Masse<small>kg/m</small>" in tab)
chk("le pied du tableau : arrondi au mm², ρ = 7 860 kg/m³ (la valeur de l'image)", "7 860 kg/m³" in tab and "arrondies au mm²" in tab)
chk("le survol d'une case donne sa valeur exacte et son taux (Ø12, 4 barres : 452,4 mm² · 88 %)",
    any(f"4Ø12 = 452,4 mm² · 88{C.NBSP}% · suffit" in c[1] for c in rg[12]["cases"]), [c[1] for c in rg[12]["cases"]][:4])

# ---------------------------------------------------------------------------------------------------------- 2. marques
print("\n=== 2 · Ce que le tableau surligne = ce que le moteur calcule")
res = MO.calculer(etat0["reperes"][0])
toutes = [(d, j, c) for d, x in rg.items() for j, c in enumerate(x["cases"])]
ok_crit = [(d, j) for d, j, c in toutes if "ok" in c[0] and "hc" not in c[0] and "hs" not in c[0]]
chk(f"les cases « suffit » dans les critères : {res['suffisantes']} — celles du moteur", len(ok_crit) == res["suffisantes"] == 91, len(ok_crit))
chk("une case ≥ A_s,req et une seule surlignée « la plus économique » (★) : 2Ø16",
    [(d, j) for d, j, c in toutes if "opt" in c[0]] == [(16, 1)] and [c[2] for d, j, c in toutes if "opt" in c[0]] == ["★"])
chk("la retenue (✓) : 4Ø12, sa rangée et sa colonne marquées",
    [(d, j) for d, j, c in toutes if "cur" in c[0]] == [(12, 3)] and rg[12]["cur"] and '<th class="n cur"><b>4</b>' in tab)
chk("en mode barres, les colonnes « espacement fixe » sont hors sujet (grisées)", all("hs" in c[0] for d, x in rg.items() for c in x["cases"][10:]))
chk("la légende : 91 cases suffisent ≥ 400 mm² ; la plus économique 2Ø16 = 402 mm² ; retenue 4Ø12",
    "91 cases suffisent · ≥ 400 mm²" in tab and "la plus économique · 2Ø16 = 402 mm²" in tab and "retenue · 4Ø12" in tab)

# ---------------------------------------------------------------------------------------------------------- 3. résultat
print("\n=== 3 · Le cadre Résultat et les fenêtres de calcul")
lignes = LIGNE.findall(r["resultat"])
chk("deux lignes : la retenue (gouvernante) et la plus économique", [(g, n) for g, n, *_ in lignes] == [(" gov", "Retenue"), ("", "Économique")], lignes)
t_ret = C.pct(400 / T.section(4, 12))
t_opt = C.pct(400 / T.section(2, 16))
chk(f"la retenue : 4Ø12 · 452 mm², {t_ret} % (A_s,req / A_s), sa couleur C.niveau",
    "4Ø12 · 452 mm²" in lignes[0][2] and lignes[0][3] == C.niveau(400 / T.section(4, 12)) and lignes[0][4] == f"{t_ret}{C.NBSP}%", lignes[0])
chk(f"la plus économique : ★ 2Ø16 · 402 mm², {t_opt} % (vert foncé : 95–100 %)",
    "2Ø16 · 402 mm²" in lignes[1][2] and lignes[1][3] == "f" and lignes[1][4] == f"{t_opt}{C.NBSP}%", lignes[1])
chk("l'en-tête : la section retenue en grand (452 mm²), le besoin et le taux", '<span class="chip xl p">452<small>mm²</small></span>' in r["resultat"]
    and "taux 88" in r["resultat"] and "suffit" in r["resultat"])
chk("la fiche de la disposition : section, masse des barres 3,556 kg/m, lues par mètre e = 25 cm, une barre 113,1 mm² · 0,889 kg/m",
    all(s in r["resultat"] for s in ("452 mm²", "3,556 kg/m", "4 barres/m · e = 25 cm", "113,1 mm² · 0,889 kg/m")))
chk("l'aide de la disposition écrit la masse au gramme, comme la fiche (3,556 kg/m, non 3,56)", "3,556 kg/m" in r["donnees"] and "3,56 kg/m" not in r["donnees"])
chk("le groupe de la fiche : « Barres retenues · 4Ø12 »", '<div class="elgrp">Barres retenues · 4Ø12</div>' in r["resultat"])
chk("un repère « ☆ » sur la retenue : il existe plus économique (titre)", "La plus économique : 2Ø16 · 402 mm²" in lignes[0][2])
chk("deux fenêtres de calcul : la retenue et la plus économique", 'id="ck-disp"' in r["volets"] and 'id="ck-opt"' in r["volets"])
ck = r["volets"].split('id="ck-opt"')[0]
for attendu in ("π × 12² / 4", "113,1 mm²", "7 860 × 113,1 × 10⁻⁶", "0,889 kg/m", "4 × 113,1", "452 mm²", "4 × 0,889", "3,556 kg/m",
                "100 / 4", "25,0 cm", "400 / 452", f"88{C.NBSP}%"):
    chk(f"la fenêtre de la retenue écrit « {attendu} »", attendu in ck)
chk("… le tableau de la fenêtre : A_s,req = 400 mm² face à A_s = 452 mm², taux 88 %",
    "A<sub>s,req</sub> = 400 mm²" in ck and "A<sub>s</sub> = 452 mm²" in ck and f'<span class="chip p">88{C.NBSP}%</span>' in ck)

# ---------------------------------------------------------------------------------------------------------- 4. saisies
print("\n=== 4 · Les saisies, les refus")
e = saisir(etat0, "As_req", "600")
chk("A_s,req = 600 : la retenue 4Ø12 ne suffit pas — rouge, message", e["ok"] and not e["message"] and e["etat"]["reperes"][0]["As_req"] == 600
    and LIGNE.findall(e["resultat"])[0][3] == "r" and "ne suffit pas" in e["resultat"])
chk("… la plus économique devient 3Ø16 (603)", "3Ø16 · 603 mm²" in e["resultat"])
for chemin, val, mot in (("As_req", "-5", "négative"), ("As_req", "abc", "illisible"), ("n", "2,5", "entier"), ("n", "0", "1 à"),
                         ("n", "201", "1 à"), ("d", "13", "diamètre"), ("d_min", "13", "diamètre"), ("n_max", "11", "recherche"),
                         ("e_min", "0", "positifs"), ("inconnu", "1", "inconnu"), ("mode", "treillis", "mode")):
    e = saisir(etat0, chemin, val)
    chk(f"{chemin} = « {val} » est refusé avec un message (« {mot} ») et le repère reste intact",
        e["ok"] and e["message"] and mot in e["message"] and e["etat"] == etat0, (e.get("message"), e.get("erreur")))
e1 = saisir(etat0, "d_max", "10")["etat"]
e = saisir(e1, "d_min", "12")
chk("Ø mini 12 au-dessus du Ø maxi 10 : refusé avec un message, le repère intact", e["message"] and "maxi" in e["message"] and e["etat"] == e1, e.get("message"))
e = saisir(etat0, "n", "14")
chk("14 barres : accepté, hors du tableau — signalé ; aucune case retenue dans la vue", e["ok"] and not e["message"] and "n&#x27;est pas une case du tableau" in e["resultat"]
    and not any("cur" in c[0] for x in rangees(vue(e, "tableau")["svg"]).values() for c in x["cases"])
    and "(hors du tableau)" in vue(e, "tableau")["svg"])
# relecture : la case écrit 1 257 pour 1Ø40 (1 256,64 mm²) ; face à 1 257 requis, l'écran ne doit jamais écrire
# « 1 257 < 1 257 » — la section s'écrit alors au dixième, partout où elle est comparée au besoin
e = saisir(saisir(saisir(etat0, "d", "40")["etat"], "n", "1")["etat"], "As_req", "1257")
ck = e["volets"].split('id="ck-opt"')[0]
chk("1Ø40 face à 1 257 mm² : l'en-tête, la ligne, la fenêtre et le message écrivent 1 256,6 mm² < 1 257 mm²",
    ">1 256,6<small>mm²</small>" in e["resultat"] and "1Ø40 · 1 256,6 mm²" in e["resultat"] and "1 256,6 mm² &lt; 1 257 mm² requis" in e["resultat"]
    and "A<sub>s</sub> = 1 256,6 mm²" in ck and "1 257 / 1 256,6" in ck and "1 257 mm² &lt; 1 257" not in e["resultat"], e["resultat"][:400])
chk("… et la section requise s'écrit avec l'espace des milliers (« 1 257 », non « 1257 »)",
    "A<sub>s,req</sub> = 1 257 mm²" in e["resultat"] and "1257" not in e["resultat"] + e["phrase"] + ck)
e = saisir(etat0, "As_req", "1000000")
chk("A_s,req = 1 000 000 mm² : le taux s'écrit « > 999 % » (et non un nombre de sept chiffres), la jauge à 999 %",
    f">{C.NBSP}999{C.NBSP}%" in e["resultat"] and f">{C.NBSP}999{C.NBSP}%" in e["elements"] and not re.search(r"\d{5,}\s?%", e["resultat"] + e["elements"]))
for chemin, val, mot in (("As_req", "1e300", "au plus"), ("As_req", "1000001", "au plus"), ("nom", "x" * 121, "caractères")):
    e = saisir(etat0, chemin, val)
    chk(f"{chemin} = « {val[:12]}… » est refusé avec un message (« {mot} ») et le repère reste intact",
        e["ok"] and e["message"] and mot in e["message"] and e["etat"] == etat0, e.get("message"))
e = saisir(etat0, "As_req", "-0")
chk("« -0 » est lu 0 : le champ ne montre jamais « -0 »", e["etat"]["reperes"][0]["As_req"] == 0 and 'value="-0"' not in e["donnees"])
e = saisir(etat0, "As_req", "1 250,5")
chk("une saisie « 1 250,5 » (espace des milliers, virgule) est lue 1250,5", e["etat"]["reperes"][0]["As_req"] == 1250.5)
e = saisir(etat0, "nom", "Dalle D1 · nappe x")
chk("le nom du repère s'affiche (résultat, éléments)", "Dalle D1 · nappe x" in e["resultat"] and "Dalle D1 · nappe x" in e["elements"])
e = saisir(etat0, "optimal", "1")
chk("« Retenir la plus économique » : 2Ø16 devient la retenue", (e["etat"]["reperes"][0]["d"], e["etat"]["reperes"][0]["n"]) == (16, 2)
    and "Retenir la plus économique" not in e["donnees"])
e0 = saisir(etat0, "As_req", "0")["etat"]
e = saisir(e0, "optimal", "1")
chk("… sans section requise : refusé avec un message", e["message"] and "section requise" in e["message"] and e["etat"] == e0)
r0 = req(action="calculer", etat=e0)
chk("sans section requise : ni ★ ni case « suffit », taux « à compléter », pas de couleur", "opt" not in "".join(" ".join(c[0]) for x in rangees(vue(r0, "tableau")["svg"]).values() for c in x["cases"])
    and not any("ok" in c[0] for x in rangees(vue(r0, "tableau")["svg"]).values() for c in x["cases"]) and "à compléter" in r0["resultat"]
    and 'id="ck-opt"' not in r0["volets"])

# la nappe
en = saisir(etat0, "mode", "nappe")["etat"]
rn = req(action="calculer", etat=en)
chk("mode Nappe : l'unité devient mm²/m (champ, résultat), la disposition Ø12 à 15 cm (la case e15)",
    'data-chemin="As_req"' in rn["donnees"] and "mm²/m" in rn["donnees"] and "Ø12 · 15 cm · 754 mm²/m" in rn["resultat"])
rgn = rangees(vue(rn, "tableau")["svg"])
chk("… les colonnes « espacement fixe » servent ; la retenue est la case Ø12 · 15 cm", not any("hs" in c[0] for x in rgn.values() for c in x["cases"])
    and [(d, j) for d, x in rgn.items() for j, c in enumerate(x["cases"]) if "cur" in c[0]] == [(12, 10)])
chk("… les colonnes 1 et 2 barres/m (100 et 50 cm) sont hors des critères (e ≤ 40 cm)", all("hc" in x["cases"][0][0] and "hc" in x["cases"][1][0] for x in rgn.values()))
chk("… A_s,req = 400 mm²/m : la plus économique Ø8 à 12,5 cm (402) — Ø16 à 50 cm, même section, est hors des critères (e ≤ 40 cm)",
    "★ Ø8 · 12,5 cm" in rn["phrase"], rn["phrase"])
r5 = req(action="calculer", etat=saisir(en, "As_req", "500")["etat"])
chk("… A_s,req = 500 mm²/m : Ø16 à 40 cm (503) — le moins de barres par mètre à section égale avec Ø8 à 10 cm", "★ Ø16 · 40 cm" in r5["phrase"], r5["phrase"])
chk("… la fiche : section par mètre, masse de la nappe en kg/m², barres par mètre, jour entre barres",
    all(s in rn["resultat"] for s in ("754 mm²/m", "5,93 kg/m²", "6,67 · e = 15 cm", "e − Ø = 138 mm")))
el = saisir(en, "espacement", "libre")["etat"]
rl = req(action="calculer", etat=el)
chk("espacement « autre » : le champ e apparaît", 'data-chemin="e_libre"' in rl["donnees"])
e = saisir(el, "e_libre", "12")
chk("e = 12 cm : Ø12 · 12 cm = 942 mm²/m, hors du tableau (signalé)", "Ø12 · 12 cm · 942 mm²/m" in e["resultat"] and "n&#x27;est pas une case du tableau" in e["resultat"])
e = saisir(el, "e_libre", "0")
chk("e = 0 : refusé, message", e["message"] and "espacement" in e["message"] and e["etat"] == el)
e = saisir(el, "e_libre", "250")
chk("e = 250 cm : refusé (au-delà d'un mètre, le mètre de nappe n'a plus de barre à dessiner)", e["message"] and "100 cm" in e["message"] and e["etat"] == el)
e = saisir(el, "e_libre", "16,75")
chk("e = 16,75 cm : le nom garde ses deux décimales (« Ø12 · 16,75 cm », non « 16,8 ») ; l'aide donne ses barres par mètre",
    "Ø12 · 16,75 cm · 675 mm²/m" in e["resultat"] and "16,8" not in e["resultat"] + e["donnees"] and "5,97 barres/m" in e["donnees"])
e = saisir(en, "espacement", "n6")
chk("espacement 16,7 cm (6 barres/m) : A_s = 6·a = 679 mm²/m (la colonne « 6 » de l'image)", "Ø12 · 16,7 cm · 679 mm²/m" in e["resultat"])

# relecture : la disposition RETENUE (en-tête, ligne, fenêtre de calcul, fiche) a la valeur de sa case de l'IMAGE,
# pour seize dispositions : barres aux coins du tableau, colonnes « n barres par mètre », espacements fixes, libre
def case_image(d, colonne):
    li = LIGNES[T.DIAMETRES.index(d)]
    return int(li[int(colonne[1:])]) if colonne[0] == "n" else int(li[13 + T.ESPACEMENTS_FIXES.index(int(colonne[1:]))])


DISPOSITIONS = [({}, "4Ø12", 12, "n4"), ({"d": "6", "n": "1"}, "1Ø6", 6, "n1"), ({"d": "40", "n": "10"}, "10Ø40", 40, "n10"),
                ({"d": "14", "n": "9"}, "9Ø14", 14, "n9"), ({"d": "28", "n": "7"}, "7Ø28", 28, "n7"), ({"d": "25", "n": "3"}, "3Ø25", 25, "n3"),
                ({"mode": "nappe", "d": "10", "espacement": "e15"}, "Ø10 · 15 cm", 10, "e15"),
                ({"mode": "nappe", "d": "6", "espacement": "e40"}, "Ø6 · 40 cm", 6, "e40"),
                ({"mode": "nappe", "d": "40", "espacement": "e35"}, "Ø40 · 35 cm", 40, "e35"),
                ({"mode": "nappe", "d": "32", "espacement": "e30"}, "Ø32 · 30 cm", 32, "e30"),
                ({"mode": "nappe", "d": "20", "espacement": "n6"}, "Ø20 · 16,7 cm", 20, "n6"),
                ({"mode": "nappe", "d": "16", "espacement": "n7"}, "Ø16 · 14,3 cm", 16, "n7"),
                ({"mode": "nappe", "d": "8", "espacement": "n10"}, "Ø8 · 10 cm", 8, "n10"),
                ({"mode": "nappe", "d": "30", "espacement": "n9"}, "Ø30 · 11,1 cm", 30, "n9"),
                ({"mode": "nappe", "d": "12", "espacement": "libre", "e_libre": "15"}, "Ø12 · 15 cm", 12, "e15"),
                ({"mode": "nappe", "d": "25", "espacement": "libre", "e_libre": "30"}, "Ø25 · 30 cm", 25, "e30")]
fautes = []
for champs, nom, d, col in DISPOSITIONS:
    e = etat0
    for k, v in champs.items():
        e = saisir(e, k, v)["etat"]
    x = req(action="calculer", etat=e)
    att = f"{case_image(d, col):,}".replace(",", " ")
    u = "mm²" if col[0] == "n" and "mode" not in champs else "mm²/m"
    ck = x["volets"].split('id="ck-opt"')[0]
    masse = LIGNES[T.DIAMETRES.index(d)][11]
    if not (f">{att}<small>{u}</small>" in x["resultat"] and f"{nom} · {att} {u}" in x["resultat"] and f"A<sub>s</sub> = {att} {u}" in ck
            and f"{masse} kg/m" in x["resultat"]):
        fautes.append((nom, att, masse))
chk(f"{len(DISPOSITIONS)} dispositions (coins, colonnes 100/n, espacements fixes, libre) : l'en-tête, la ligne « Retenue », la fenêtre "
    "de calcul et la barre seule portent la valeur de la case de l'IMAGE et sa masse", not fautes and len(DISPOSITIONS) == 16, fautes)

# ---------------------------------------------------------------------------------------------------------- 5. sélecteur
print("\n=== 5 · Le sélecteur des dispositions")
s = req(action="selecteur", etat=etat0, role="disp")
rows = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)".*?<span class="kg">([^<]+)</span>', s["html"])
chk("barres : 120 dispositions, triées par section croissante", len(rows) == 120 and [nombre(x[3]) for x in rows] == sorted(nombre(x[3]) for x in rows), len(rows))
chk("… la retenue ✓ 4Ø12, la plus économique ★ 2Ø16 (avant 8Ø8, même section)",
    [x[2] for x in rows if x[0]] == ["4Ø12"] and [x[2] for x in rows if x[1]] == ["2Ø16"]
    and [x[2] for x in rows].index("2Ø16") < [x[2] for x in rows].index("8Ø8"))
chk("… chaque ligne a la section de sa case de l'image", all(nombre(x[3]) == int(LIGNES[T.DIAMETRES.index(int(x[2].split("Ø")[1]))][int(x[2].split("Ø")[0])])
                                                           for x in rows))
c = req(action="choisir", etat=etat0, role="disp", nom="3Ø20")
chk("un clic sur 3Ø20 la retient", (c["etat"]["reperes"][0]["n"], c["etat"]["reperes"][0]["d"]) == (3, 20))
s = req(action="selecteur", etat=en, role="disp")
noms = re.findall(r'data-choix="([^"]+)"', s["html"])
chk("nappe : 12 Ø × 12 espacements de 10 à 40 cm = 144 dispositions", len(noms) == 144, len(noms))
c = req(action="choisir", etat=en, role="disp", nom="Ø10 · 16,7 cm")
chk("un clic sur « Ø10 · 16,7 cm » retient la colonne 6 barres/m", (c["etat"]["reperes"][0]["d"], c["etat"]["reperes"][0]["espacement"]) == (10, "n6"))
c = req(action="choisir", etat=en, role="disp", nom="Ø10 · 35 cm")
chk("« Ø10 · 35 cm » retient l'espacement fixe de 35 cm", c["etat"]["reperes"][0]["espacement"] == "e35")
e = saisir(el, "e_libre", "12")["etat"]
s = req(action="selecteur", etat=e, role="disp")
chk("une retenue hors du tableau (Ø12 · 12 cm) paraît dans la liste, marquée ✓", re.search(r'class="pk-row cur[^"]*" data-choix="Ø12 · 12 cm"', s["html"]) is not None)
c = req(action="choisir", etat=en, role="disp", nom="Ø16 · 12 cm")
chk("« Ø16 · 12 cm » (hors colonnes) retient un espacement libre de 12 cm", c["etat"]["reperes"][0]["espacement"] == "libre" and c["etat"]["reperes"][0]["e_libre"] == 12)
c = req(action="choisir", etat=etat0, role="disp", nom="n'importe quoi")
chk("un nom illisible est une erreur lisible, jamais une exception", c["ok"] is False and "illisible" in c["erreur"])

# ---------------------------------------------------------------------------------------------------------- 6. éléments
print("\n=== 6 · Les repères de la note")
a = req(action="ajouter", etat=etat0)
chk("ajouter : « Repère 2 », ouvert", [x["nom"] for x in a["etat"]["reperes"]] == ["Repère 1", "Repère 2"] and a["etat"]["actif"] == 1)
e2 = saisir(a["etat"], "As_req", "1000")["etat"]
rr = req(action="calculer", etat=e2)
chk("le résumé nomme le repère le plus sollicité (taux le plus haut), sans verdict", rr["resume"]["gouvernant"].startswith("Repère 2 · ") and rr["resume"]["niveau"] == "i")
chk("chaque repère a son taux dans la liste (88 % ; Repère 2 : rouge)", '88' in rr["elements"] and 'class="pct r"' in rr["elements"])
d = req(action="dupliquer", etat=e2)
chk("copier : « Repère 2 (copie) », mêmes valeurs", d["etat"]["reperes"][2]["nom"] == "Repère 2 (copie)" and d["etat"]["reperes"][2]["As_req"] == 1000)
s_ = req(action="supprimer", etat=d["etat"], index=0)
chk("supprimer le premier : deux restent, l'ouvert suit", [x["nom"] for x in s_["etat"]["reperes"]] == ["Repère 2", "Repère 2 (copie)"] and s_["etat"]["actif"] == 1)
s_ = req(action="supprimer", etat=etat0, index=0)
chk("le dernier repère ne se supprime pas", len(s_["etat"]["reperes"]) == 1)
k = req(action="ajouter", etat=req(action="ajouter", etat=etat0)["etat"])
k = req(action="supprimer", etat=k["etat"], index=1)
k = req(action="ajouter", etat=k["etat"])
chk("un repère ajouté prend un nom libre (pas deux « Repère 3 »)", len({x["nom"] for x in k["etat"]["reperes"]}) == 3, [x["nom"] for x in k["etat"]["reperes"]])
x = req(action="calculer", etat={"version": 1, "reperes": [{"d": 13}], "actif": 0})
chk("un fichier .stab abîmé (Ø13) : erreur lisible", x["ok"] is False and "Ø13" in x["erreur"])
x = req(action="calculer", etat={"version": 1, "reperes": [{}, {}], "actif": 9})
chk("un repère actif hors de la liste est ramené au dernier", x["ok"] and x["etat"]["actif"] == 1)

# ---------------------------------------------------------------------------------------------------------- 7. sûreté, dessins
print("\n=== 7 · Échappement, dessins")
PIEGE = '<img src=x onerror=alert(1)>'
e = saisir(etat0, "nom", PIEGE)
frag = e["resultat"] + e["volets"] + e["donnees"] + e["elements"] + e["phrase"] + "".join(v["svg"] for v in e["vues"]) + json.dumps(e["resume"])
chk("un nom piégé est échappé partout (résultat, fenêtres, données, éléments, phrase, vues)", "<img" not in frag.replace(json.dumps(e["resume"]), "") and "&lt;img" in e["resultat"])
chk("… le résumé le porte en TEXTE (la page l'échappe)", PIEGE in e["resume"]["gouvernant"])
nap = vue(r, "nappe")
chk("la nappe d'un mètre (4Ø12 lus par mètre) : 4 barres de rayon 6 mm, l'entraxe 250 mm coté", len(re.findall(r'<circle cx="[\d.]+" cy="[\d.]+" r="6"', nap["svg"])) == 4
    and "e = 250" in nap["svg"] and "1 000 mm" in nap["svg"])
dia = vue(r, "diametres")
chk("les diamètres : douze cercles à l'échelle 2:1 ; Ø12 retenu encadré ; Ø16 (la plus économique) étoilé",
    [float(x) for x in re.findall(r'<circle cx="[\d.]+" cy="[\d.]+" r="([\d.]+)"', dia["svg"])] == [float(dd) for dd in T.DIAMETRES]
    and dia["svg"].count("<polygon") == 1)
chk("aucun script, aucun « on…= » dans les fragments", not re.search(r"<script|\son[a-z]+\s*=", r["resultat"] + r["volets"] + r["donnees"] + tab, re.I))
chk("la phrase sous le titre : le repère, la retenue, le besoin, la plus économique",
    r["phrase"] == "Repère 1 · 4Ø12 retenu · A<sub>s,req</sub> = 400 mm² · ★ 2Ø16", r["phrase"])
e = copy.deepcopy(etat0)
chk("normaliser est stable et l'état survit au JSON", A.normaliser(json.loads(json.dumps(e))) == e)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:240])
sys.exit(1 if KO else 0)
