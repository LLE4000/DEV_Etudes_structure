# -*- coding: utf-8 -*-
"""L'API du site pour les poutres béton armé (api_web/poutre.py), vérifiée en Python ordinaire.

- le résultat affiché est celui du moteur beton/poutre : taux, couleur et vérification gouvernante, section par section ;
- une vérification que le moteur déclare non conforme est rouge même sous 100 % (acier excessif) ;
- chaque champ du cadre Données pointe un endroit réel de la poutre, et le réécrire avec sa valeur ne change rien ;
- les boutons « Ajouter » et « Retirer » (sections, lits, lignes d'étriers) passent par le même chemin, avec leurs bornes ;
- une saisie hors domaine est refusée avec un message, et la poutre reste intacte ;
- le sélecteur d'armatures est trié par section d'acier, marque la disposition retenue et l'optimale du moteur ;
- les poutres de la note s'ajoutent, se copient, se suppriment (jamais la dernière) ; le résumé est la plus chargée ;
- un nom saisi est échappé partout où il s'affiche ;
- l'établi compact (site, 11/2026) : chaque vérification sur deux lignes ouvre SA fenêtre de détail ; une section sans
  effort est « Données incomplètes », jamais « Conforme » ; sans V, τ et étriers sont « non effectuée » ; les dispositions
  constructives disent leur état en mots, sans taux ; chaque pièce du dessin a sa ligne dans les données (et
  réciproquement) ; les clés du ferraillage détaillé (position, répartition, entraxe, barres entourées, latérales,
  copie de section) se saisissent, se vident (automatique) et se refusent comme les autres ;
- la deuxième passe (11/2026) : à gauche, depuis le 09/11/2026, la colonne COMPACTE (la ligne de la poutre ouverte : son
  nom en champ, la liste des poutres « nom · b × h cm » sans taux, copier, supprimer ; puis ses rubriques) ; les
  barres qu'un cadre entoure se choisissent (« de-à ») et ne désignent jamais une barre absente ; à droite un BLOC par
  section, coloré par son état, et la LISTE DE CONTRÔLE des dispositions (une observation « à examiner » se justifie,
  une non-conformité non) ; au centre l'en-tête technique de chaque vue et les poignées du cadre choisi.

    python3 tests/test_api_web_poutre.py
"""
import copy
import itertools
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import poutre as P  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.poutre import entree as EN  # noqa: E402
from beton.poutre import moteur as MO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:240]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée."""
    out = req(module="poutre", action="modifier", etat=etat, chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


def niveau_de(v):
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


# une vérification À TAUX (l'établi compact, commun.ligne_verif) : deux lignes, le taux à droite de la première
LIGNE = re.compile(r'<div class="el vc( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>(?:(?!</div>).)*?<span class="pct (\w)">([^<]+)</span>', re.S)

# 1. une poutre neuve : le résultat est celui du moteur
r = req(module="poutre", action="nouveau")
chk("nouveau : la requête aboutit", r["ok"], r.get("erreur"))
p0 = EN.nouvelle_poutre()
p0["nom"] = "V1"
chk("la poutre neuve est celle du module (20 × 40, C30/37, B500)", r["etat"]["poutres"] == [EN.normaliser(p0)])
ref = MO.verifier(p0)
chk("une poutre neuve, sans effort, n'affiche pas de taux : « Données incomplètes »", r["resume"]["taux"] is None and r["resume"]["niveau"] == "n",
    (r["resume"], ref["taux_max"]))

# 2. une poutre chargée : M = 122 kN·m, V = 60 kN, puis une seconde section sur appui
r = saisir(r["etat"], "sections/0/M_inf", "122")
r = saisir(r["etat"], "sections/0/V", "60")
r = req(module="poutre", action="modifier", etat=r["etat"], chemin=["sections", "+"], valeur=1)
r = saisir(r["etat"], "sections/1/M_sup", "45")
r = saisir(r["etat"], "sections/1/V", "35")
etat = r["etat"]
p = etat["poutres"][0]
chk("« Ajouter une section » crée la section B", [s["nom"] for s in p["sections"]] == ["A", "B"], [s["nom"] for s in p["sections"]])
ref = MO.verifier(p)
attendu = [(v, s) for s in ref["sections"] for v in s["verifications"]]
lignes = LIGNE.findall(r["resultat"])
chk(f"une ligne de résultat par vérification du moteur ({len(attendu)})", len(lignes) == len(attendu), (len(lignes), len(attendu)))
chk("chaque ligne porte le nom court, le taux et la couleur de sa vérification",
    all(nom == C.esc(P.COURTS[v["cle"]]) and n == niveau_de(v) and t == f"{C.pc(C.pct(v['taux']))}{C.NBSP}%"
        for (_, nom, n, t), (v, _s) in zip(lignes, attendu)), lignes[:3])
g = ref["gouvernant"]
gov = [i for i, (x, *_) in enumerate(lignes) if x]
chk("une seule ligne gouvernante, celle du moteur",
    len(gov) == 1 and attendu[gov[0]][1]["nom"] == g["section"] and attendu[gov[0]][0]["cle"] == g["cle"], (gov, g))
niv = "r" if ref["statut"] == MO.NOK else C.niveau(ref["taux_max"])
niv = {MO.NOK: "r", MO.WARN: "v"}.get(ref["statut_complet"], niv)
chk("le résumé : taux, couleur et gouvernant du moteur",
    r["resume"]["taux"] == C.pct(ref["taux_max"]) and r["resume"]["niveau"] == niv and r["resume"]["gouvernant"].endswith(f"section {g['section']} · {g['nom']}"),
    (r["resume"], ref["taux_max"], g))
grp = re.findall(r'<section class="et-bloc (\w+)" data-section="(\d+)">.*?<span class="n">Section ([^<]+)</span><span class="e">([^<]+)</span>', r["resultat"], re.S)
etat_sec = {MO.NOK: ("nok", "Non conforme"), MO.WARN: ("warn", "À vérifier")}
chk("un BLOC par section, son en-tête coloré par son état (méthode et dispositions)",
    grp == [(*etat_sec.get(s["statut_complet"], ("ok", "Conforme"))[:1], str(i), s["nom"], etat_sec.get(s["statut_complet"], ("ok", "Conforme"))[1])
            for i, s in enumerate(ref["sections"])], grp)
cles_v = [f"s{i}-{v['cle']}" for i, s in enumerate(ref["sections"]) for v in s["verifications"]]
ouvre = re.findall(r'<div class="el vc[^"]*" data-calcul="(s\d-\w+)"', r["resultat"])
chk("chaque vérification ouvre SA fenêtre de détail, et chaque section a la sienne (tout le calcul)",
    all(f'id="ck-{k}"' in r["volets"] for k in cles_v) and all(f'id="ck-s{i}"' in r["volets"] for i in range(len(p["sections"])))
    and ouvre == cles_v and re.findall(r'class="et-tout" data-calcul="(s\d)"', r["resultat"]) == [f"s{i}" for i in range(len(p["sections"]))], (ouvre, cles_v))
chk("une ligne « Dispositions constructives » par section, qui déplie SA liste de contrôle (pas de fenêtre à part)",
    re.findall(r'class="el vc et-dispo" data-dispo="(\d)"', r["resultat"]) == [str(i) for i in range(len(p["sections"]))]
    and re.findall(r'class="et-obs-l" data-dispo-l="(\d)"', r["resultat"]) == [str(i) for i in range(len(p["sections"]))], ouvre)
chk("une vue par section, et « Toutes » quand il y en a plusieurs", [v["cle"] for v in r["vues"]] == ["s0", "s1", "toutes"], [v["cle"] for v in r["vues"]])
chk("le choix d'armatures s'ouvre sur les deux faces de chaque section, depuis le résultat et depuis les données",
    re.findall(r'data-pk="([^"]+)"', r["resultat"]) == ["inf-0", "sup-0", "inf-1", "sup-1"]
    and re.findall(r'data-pk="([^"]+)"', r["donnees"]) == ["inf-0", "sup-0", "inf-1", "sup-1"], re.findall(r'data-pk="([^"]+)"', r["resultat"]))
# la fenêtre de calcul : chaque formule du moteur a sa ligne, et son résultat est celui du moteur
fx = sum(len(v["formules"]) for v in ref["sections"][0]["verifications"])
ck0 = r["volets"].split('id="ck-s1-hauteur"')[0]
chk(f"la fenêtre de la section A détaille les {fx} formules du moteur", ck0.count('class="fx') >= fx, ck0.count('class="fx'))
v_inf = next(v for v in ref["sections"][0]["verifications"] if v["cle"] == "inf")
f_req = next(f for f in v_inf["formules"] if f["cle"] == "As_req")
chk("A_s,req s'écrit avec les valeurs du moteur et son résultat", f"{C.fr(f_req['resultat'], 0)} mm²" in ck0 and "122 × 10⁶" in ck0, f_req["resultat"])

# 3. rouge même sous 100 % : l'acier excessif (A_s > A_s,max) est non conforme quel que soit son taux
e = copy.deepcopy(etat)
for j in range(EN.MAX_LITS):
    if j:
        req_ = req(module="poutre", action="modifier", etat=e, chemin=["sections", "0", "inf", "+"], valeur=1)
        e = req_["etat"]
    e = saisir(e, f"sections/0/inf/{j}/n", "6")["etat"]
    out = saisir(e, f"sections/0/inf/{j}/d", "32")
    e = out["etat"]
v = next(v for v in MO.verifier(e["poutres"][0])["sections"][0]["verifications"] if v["cle"] == "inf")
ligne_inf = [x for x in LIGNE.findall(out["resultat"]) if x[1] == "Armatures inférieures"][0]
chk("acier excessif : le moteur le déclare non conforme sous 100 %", v["etat"] == MO.NOK and v["taux"] < 1, (v["etat"], v["taux"]))
chk("… et la ligne est rouge, pas verte", ligne_inf[2] == "r", ligne_inf)

# 4. chaque champ pointe la poutre ; le réécrire avec sa propre valeur ne change rien
champs = re.findall(r'<input data-chemin="([^"]+)" data-sorte="(\w+)"[^>]*?value="([^"]*)"', r["donnees"])
champs += [(c_, "choix", v_) for c_, v_ in re.findall(r'<select data-chemin="([^"]+)" data-sorte="choix"[^>]*>.*?<option value="([^"]*)" selected', r["donnees"], re.S)]
champs += re.findall(r'<button type="button" class="on" data-chemin="([^"]+)" data-sorte="(choix)" data-valeur="([^"]*)"', r["donnees"])
# poutre (nom, béton, acier, b, h, enrobage) + par section (nom, M_inf, M_sup, V ; par lit : n, Ø, répartition, position
# [+ entraxe horizontal] ; y_G imposé des deux faces ; latérales n et Ø ; par ligne d'étriers : type, Ø — ses barres se
# CHOISISSENT, data-plage, plus bas ; pas) + options (γs, deux jeux, d_g, Ø et écart des latérales automatiques)
n_attendu = 6 + 6 + sum(4 + sum(4 + (lit.get("rep") == "entraxe") for f in ("inf", "sup") for lit in s_[f]) + 2 + 2
                        + sum(2 for _e in s_["etriers"]) + 1 for s_ in p["sections"])
chk(f"le cadre Données porte tous les champs de la poutre ({len(champs)})", len(champs) == n_attendu, (len(champs), n_attendu))
vus = {c_ for c_, _, _ in champs}
chk("les champs couvrent la poutre, les efforts, les lits, les étriers et la méthode",
    {"nom", "beton", "fyk", "b", "h", "enrobage", "gamma_s", "jeu_premier_lit", "jeu_entre_lits", "sections/0/M_inf", "sections/1/M_sup",
     "sections/0/V", "sections/0/inf/0/n", "sections/0/inf/0/d", "sections/1/sup/0/d", "sections/0/etriers/0/type", "sections/0/etriers/0/d",
     "sections/1/pas"} <= vus, sorted(vus))
mauvais = []
for chemin, sorte, valeur in champs:
    e2 = copy.deepcopy(etat)
    msg = P.modifier(e2, chemin.split("/"), valeur.replace("&amp;", "&"))
    if msg or e2 != etat:
        mauvais.append((chemin, valeur, msg))
chk("réécrire chaque champ avec sa valeur affichée ne change rien", not mauvais, mauvais[:3])

# 5. les saisies : nombres, entiers, refus
e = copy.deepcopy(etat)
P.modifier(e, ["h"], "62,5")
chk("« 62,5 » se lit 62,5 cm", e["poutres"][0]["h"] == 62.5)
P.modifier(e, ["sections", "0", "inf", "0", "n"], "3")
P.modifier(e, ["fyk"], "400")
chk("le nombre de barres et la nuance restent des entiers",
    e["poutres"][0]["sections"][0]["inf"][0]["n"] == 3 and isinstance(e["poutres"][0]["sections"][0]["inf"][0]["n"], int)
    and e["poutres"][0]["fyk"] == 400 and isinstance(e["poutres"][0]["fyk"], int))
for chemin, valeur, quoi in ((["h"], "douze", "un nombre illisible"), (["h"], "0", "une hauteur nulle"), (["b"], "-20", "une largeur négative"),
                             (["sections", "0", "V"], "-5", "un effort tranchant négatif"), (["beton"], "C99/999", "une classe de béton inconnue"),
                             (["fyk"], "235", "une nuance inconnue"), (["sections", "0", "inf", "0", "d"], "13", "un diamètre hors liste"),
                             (["sections", "0", "pas"], "0", "un pas nul")):
    avant = copy.deepcopy(e)
    msg = P.modifier(e, chemin, valeur)
    if not (msg and e == avant):
        chk(f"{quoi} est refusé, la poutre reste intacte", False, (chemin, valeur, msg))
        break
else:
    chk("les saisies hors domaine (illisible, nulle, négative, classe, nuance, Ø, pas) sont refusées, la poutre intacte", True)
out = req(module="poutre", action="modifier", etat=etat, chemin=["h"], valeur="0")
chk("refusée par l'écran : un message, et l'état renvoyé est l'ancien", out["ok"] and out["message"] and out["etat"] == etat, out.get("message"))

# 6. ajouter, retirer : sections, lits, lignes d'étriers
e = copy.deepcopy(etat)
for _ in range(EN.MAX_LITS - 1):
    P.modifier(e, ["sections", "0", "sup", "+"], 1)
chk(f"« Ajouter un lit » jusqu'à {EN.MAX_LITS} lits", len(e["poutres"][0]["sections"][0]["sup"]) == EN.MAX_LITS)
msg = P.modifier(e, ["sections", "0", "sup", "+"], 1)
chk(f"un {EN.MAX_LITS + 1}ᵉ lit est refusé avec un message lisible (pas celui, technique, du moteur)",
    msg == f"Au plus {EN.MAX_LITS} lits par face" and len(e["poutres"][0]["sections"][0]["sup"]) == EN.MAX_LITS, msg)
h = P.formulaire(e)
chk("au maximum, le bouton « Ajouter un lit » de cette face disparaît", 'data-chemin="sections/0/sup/+"' not in h and 'data-chemin="sections/0/inf/+"' in h)
P.modifier(e, ["sections", "0", "sup", "1", "-"], 1)
chk("« Retirer ce lit » retire le bon lit", len(e["poutres"][0]["sections"][0]["sup"]) == EN.MAX_LITS - 1)
P.modifier(e, ["sections", "0", "etriers", "+"], 1)
P.modifier(e, ["sections", "0", "etriers", "1", "type"], "epingle")
chk("une ligne d'étriers s'ajoute, et devient une épingle", [x["type"] for x in e["poutres"][0]["sections"][0]["etriers"]] == ["etrier", "epingle"])
P.modifier(e, ["sections", "0", "-"], 1)
chk("retirer la section A garde la section B", [s["nom"] for s in e["poutres"][0]["sections"]] == ["B"])
msg = P.modifier(e, ["sections", "0", "-"], 1)
chk("la dernière section ne se retire pas", msg and len(e["poutres"][0]["sections"]) == 1, msg)
msg = P.modifier(e, ["sections", "0", "inf", "0", "-"], 1)
chk("le dernier lit d'une face ne se retire pas", msg and len(e["poutres"][0]["sections"][0]["inf"]) == 1, msg)
h = P.formulaire(e)
chk("une section seule : « Supprimer la section » reste dans son menu ⋯, grisé (jamais actif)",
    re.search(r'<button type="button" role="menuitem" class="danger" data-chemin="sections/0/-" data-valeur="1" disabled>', h) is not None, h[:200])
P.modifier(e, ["sections", "+"], 1)
chk("la section ajoutée prend la première lettre libre (A)", [s["nom"] for s in e["poutres"][0]["sections"]] == ["B", "A"])

# 7. le sélecteur d'armatures
s = req(module="poutre", action="selecteur", etat=etat, role="inf-0")
rows = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)".*?<span class="kg">([^<]+)</span>', s["html"], re.S)
aires = [float(re.sub(r"[\s  ]", "", a).replace(",", ".")) for *_, a in rows]
cands = [c_ for c_ in MO.candidats_armatures(etat["poutres"][0], "A", "inf") if c_["possible"] or c_["retenue"]]
chk(f"le sélecteur liste les dispositions du moteur qui se POSENT ({len(rows)})", [n for _, _, n, _ in rows] == [c_["nom"] for c_ in cands] and len(rows) > 20, len(rows))
chk("… aucune disposition impossible (barres qui se chevauchent) n'est proposée",
    not [n for _, _, n, _ in rows if not next(c_ for c_ in MO.candidats_armatures(etat["poutres"][0], "A", "inf") if c_["nom"] == n)["possible"]])
chk("triées par section d'acier croissante", aires == sorted(aires), aires[:6])
chk("une seule disposition retenue : 2Ø16, celle de la section", [n for cur, _, n, _ in rows if cur] == ["2Ø16"], [n for cur, _, n, _ in rows if cur])
best = next((c_["nom"] for c_ in cands if c_["ok"] and c_["possible"] and not c_["serree"]), None)
chk("l'étoile est la plus petite section qui tient et se pose à l'espacement de l'EC2", [n for _, opt, n, _ in rows if opt] == ([best] if best else []),
    ([n for _, opt, n, _ in rows if opt], best))
r2 = req(module="poutre", action="choisir", etat=etat, role="inf-0", nom=best)
v2 = next(v for v in MO.verifier(r2["etat"]["poutres"][0])["sections"][0]["verifications"] if v["cle"] == "inf")
chk(f"choisir {best} le pose en un lit et recalcule : la face inférieure passe",
    r2["ok"] and r2["etat"]["poutres"][0]["sections"][0]["inf"] == [{"n": int(best.split("Ø")[0]), "d": int(best.split("Ø")[1])}] and v2["etat"] == MO.OK,
    (r2.get("erreur"), v2["etat"]))
s = req(module="poutre", action="selecteur", etat=etat, role="sup-1")
chk("le sélecteur de la face supérieure de la section B se titre ainsi", "Armatures supérieures · section B" in s["html"])

# 8. les poutres de la note
e = copy.deepcopy(etat)
P.gerer_elements(e, "ajouter")
P.gerer_elements(e, "dupliquer")
chk("ajouter puis copier : trois poutres, la dernière ouverte, nommée V2 (copie)",
    len(e["poutres"]) == 3 and e["actif"] == 2 and e["poutres"][2]["nom"] == "V2 (copie)", [x["nom"] for x in e["poutres"]])
e["actif"] = 1
rn = req(module="poutre", action="calculer", etat=e)
t_max = max(MO.verifier(x)["taux_max"] for x in e["poutres"])
chk("le résumé de la note est sa poutre la plus chargée (ici V1, fermée)",
    rn["resume"]["taux"] == C.pct(t_max) and rn["resume"]["gouvernant"].startswith("V1 ·"), (rn["resume"], t_max))
# la colonne de gauche COMPACTE (demande de l'ingénieur du 09/11/2026, etabli_compact.arbre_compact) : la LIGNE de la
# poutre ouverte (que l'écran pose dans l'en-tête du panneau) — son nom en champ, la LISTE des poutres en menu (chacune
# « nom · b × h cm », l'ouverte cochée), copier, supprimer — puis ses rubriques ; plus d'arborescence
ligne_ = re.search(r'<div class="et-elsel" data-elements="\d+">.*?(?=<div class="sec)', rn["donnees"], re.S)
ligne_ = ligne_.group(0) if ligne_ else ""
pts_ = re.findall(r'aria-checked="(true|false)" class="et-elt-i( on)?" data-activer="(\d)" title="[^"]*">.*?<span class="nom">([^<]+)</span>'
                  r'<span class="dim">([^<]+)</span>', ligne_, re.S)
chk("la LISTE des poutres (la ligne de la poutre ouverte) montre chaque poutre « nom · b × h cm », l'ouverte cochée, leur compte ; "
    "les rubriques de la seule ouverte — plus d'arborescence, sans taux ni pastille d'état",
    [(c, o, i, n) for c, o, i, n, _ in pts_] == [("false", "", "0", "V1"), ("true", " on", "1", "V2"), ("false", "", "2", "V2 (copie)")]
    and all(d == f"· {C.nb(x['b'])} × {C.nb(x['h'])} cm" for (*_, d), x in zip(pts_, e["poutres"]))
    and re.search(r'<div class="et-arbre et-arbre-c" data-poutre="1"><div class="et-elsel" data-elements="3">', rn["donnees"]) is not None
    and '<span class="n">3</span></button>' in ligne_ and "et-arbre-h" not in rn["donnees"]
    and not re.search(r'class="et-pt(?: on)?"|class="et-pt-(?:l|corps|nom|add)\b', rn["donnees"])
    and rn["donnees"].count('data-sec="pt"') == 1 and rn["elements"] == ""
    and not re.search(r'class="(chip|status|pct)\b', ligne_), (pts_, ligne_[:300]))
rub1 = re.search(r'<div class="sec[^"]*" data-sec="pt".*?(?=<div class="sec[^"]*" data-sec="so")', rn["donnees"], re.S).group(0)
chk("le NOM de la poutre ouverte se modifie dans SA ligne (champ « nom » : un clic, et on le renomme) ; la rubrique « Poutre » n'a "
    "plus de ligne Nom, et ses données tiennent en trois lignes (béton | acier, b | h, enrobage)",
    ligne_.startswith('<div class="et-elsel" data-elements="3"><input data-chemin="nom" data-sorte="texte" value="V2" class="et-elt-nom"')
    and rn["donnees"].count('data-chemin="nom"') == 1 and 'data-chemin="nom"' not in rub1 and 'data-chemin="nom"' not in rn["resultat"]
    and rub1.count('class="row2 n2"') == 2 and re.search(r'<select data-chemin="fyk" data-sorte="choix">', rub1) is not None
    and len(re.findall(r'<div class="row(?:2 n2)?"', rub1)) == 3, (ligne_[:300], re.findall(r'<div class="row[^"]*"', rub1)))
P.gerer_elements(e, "supprimer", 0)
chk("supprimer la première garde la poutre ouverte", len(e["poutres"]) == 2 and e["actif"] == 0 and e["poutres"][0]["nom"] == "V2")
P.gerer_elements(e, "supprimer", 0)
P.gerer_elements(e, "supprimer", 0)
chk("la dernière poutre ne se supprime pas", len(e["poutres"]) == 1)

# 9. toutes les classes de béton, les deux aciers, des efforts du nul au démesuré se calculent
fautes = []
for beton, fyk, (m_inf, m_sup, v_) in itertools.product(EN.classes_beton(), EN.NUANCES, ((0, 0, 0), (60, 0, 20), (0, 90, 70), (400, 300, 400))):
    e = P.nouvel_etat()
    for chemin, x in ((["beton"], beton), (["fyk"], fyk), (["sections", "0", "M_inf"], m_inf), (["sections", "0", "M_sup"], m_sup), (["sections", "0", "V"], v_)):
        msg = P.modifier(e, chemin, x)
        if msg:
            fautes.append((beton, fyk, chemin, msg))
    out = req(module="poutre", action="calculer", etat=e)
    if not out["ok"]:
        fautes.append((beton, fyk, m_inf, m_sup, v_, out.get("erreur")))
n_cas = len(EN.classes_beton()) * len(EN.NUANCES) * 4
chk(f"les {n_cas} combinaisons béton × acier × efforts se calculent", not fautes, fautes[:3])

# 10. l'échappement. Le contrat : les fragments HTML (liste, données, résultat, fenêtres, dessins, phrase) sont échappés
# ici ; les champs TEXTE (titres des vues, résumé) sont des données, que la page échappe en les posant
# (verifier_site.js : « .stab piégé »).
def fragments_html(out):
    return {"liste des poutres": out["elements"], "cadre Données": out["donnees"], "résultat": out["resultat"], "fenêtres de calcul": out["volets"],
            "dessins": "".join(v["svg"] for v in out["vues"]), "phrase": out["phrase"], "statut": out["statut_html"]}


e = copy.deepcopy(etat)
P.modifier(e, ["nom"], '<script>alert("x")</script>')
out = req(module="poutre", action="calculer", etat=e)
fuites = [k for k, h in fragments_html(out).items() if "<script>" in h]
chk("un nom de poutre saisi est échappé dans chaque fragment HTML", not fuites and "&lt;script&gt;" in out["donnees"] and "&lt;script&gt;" in out["volets"]
    and "&lt;script&gt;" in out["resultat"], fuites)
# un nom de section ne se saisit pas à l'écran, mais un fichier .stab peut en porter un
e = copy.deepcopy(etat)
e["poutres"][0]["sections"][0]["nom"] = '<img src=x onerror=alert(1)>'
out = req(module="poutre", action="calculer", etat=e)
fuites = [k for k, h in fragments_html(out).items() if "<img" in h]
chk("un nom de section venu d'un fichier est échappé dans chaque fragment HTML", out["ok"] and not fuites, fuites or out.get("erreur"))

# 11. deux sections du même nom (possible dans un fichier .stab, pas à l'écran) : le site désigne une section par son
# nom (sélecteur, ligne gouvernante) — la seconde prend la première lettre libre, et son sélecteur est bien le sien
e = copy.deepcopy(etat)
e["poutres"][0]["sections"][1]["nom"] = "A"
e["poutres"][0]["sections"][1]["M_inf"] = 200.0
out = req(module="poutre", action="calculer", etat=e)
noms = [s_["nom"] for s_ in out["etat"]["poutres"][0]["sections"]] if out["ok"] else out.get("erreur")
chk("deux sections du même nom venues d'un fichier : la seconde prend la première lettre libre (B)", noms == ["A", "B"], noms)
s = req(module="poutre", action="selecteur", etat=out["etat"], role="inf-1")
taux_affiches = re.findall(r'class="chip \w">([^<]+)<', s["html"])
# l'attendu : la seconde section SEULE dans sa poutre (aucun nom à confondre)
p2 = dict(out["etat"]["poutres"][0])
p2["sections"] = [p2["sections"][1]]
attendus = [f"{C.pc(C.pct(min(c_['taux'], 9.99)))}{C.NBSP}%" for c_ in MO.candidats_armatures(p2, p2["sections"][0]["nom"], "inf")
            if c_["possible"] or c_["retenue"]]
chk("… et le sélecteur de la seconde calcule SA section (M = 200 kN·m), pas la première", taux_affiches == attendus, (taux_affiches[:3], attendus[:3]))

# 12. l'établi compact (site/web/js/compact.js) : une pièce du dessin (<g data-sel="s<i>.<pièce>">) et la ligne des
# données qui la règle (data-cible) portent la MÊME clé — le béton, commun aux sections, porte « beton ». Chaque pièce
# a sa ligne, chaque lit, cadre et groupe de latérales a sa pièce ; les rubriques sont les quatre de l'écran, « Options »
# seule secondaire.
def cibles_de(vue):
    return set(re.findall(r'data-sel="([^"]+)"', vue["svg"]))


for nom_cas, e_cas in (("une section", req(module="poutre", action="nouveau")["etat"]), ("deux sections", etat)):
    out = req(module="poutre", action="calculer", etat=e_cas)
    pt = out["etat"]["poutres"][out["etat"].get("actif", 0)]
    lignes_d = set(re.findall(r'data-cible="([^"]+)"', out["donnees"]))
    pieces = set().union(*(cibles_de(v) for v in out["vues"]))
    sans_ligne = sorted(k for k in pieces if not (k in lignes_d or (k.endswith(".beton") and "beton" in lignes_d)))
    chk(f"dessin ↔ données ({nom_cas}) : chaque pièce du dessin a sa ligne dans les données", not sans_ligne, sans_ligne)
    attendues = {f"s{i}.lit-{f}-{j}" for i, s_ in enumerate(pt["sections"]) for f in ("inf", "sup") for j in range(len(s_[f]))} \
        | {f"s{i}.cadre-{k}" for i, s_ in enumerate(pt["sections"]) for k in range(len(s_["etriers"]))}
    chk(f"dessin ↔ données ({nom_cas}) : chaque lit et chaque cadre a sa pièce ET sa ligne", attendues <= pieces and attendues <= lignes_d,
        (sorted(attendues - pieces), sorted(attendues - lignes_d)))
    rubriques = re.findall(r'<div class="sec[^"]*" data-sec="([^"]+)"', out["donnees"])
    secondaires = re.findall(r'data-sec="([^"]+)" data-secondaire="1"', out["donnees"])
    chk(f"établi compact ({nom_cas}) : quatre rubriques (poutre, sections, armatures, options), « Options » seule secondaire",
        rubriques == ["pt", "so", "arm", "op"] and secondaires == ["op"], (rubriques, secondaires))
    blocs = re.findall(r'<div class="et-arm[^"]*" data-section="(\d+)"', out["donnees"])
    chk(f"établi compact ({nom_cas}) : la rubrique Armatures porte un bloc par section", blocs == [str(i) for i in range(len(pt["sections"]))], blocs)

# 13. les états : jamais « Conforme » sans sollicitation ; « non effectuée » sans effort tranchant ; les dispositions en mots
out = req(module="poutre", action="nouveau")
chk("une poutre neuve (aucun effort) : « Données incomplètes », sans taux, jamais « Conforme »",
    out["resume"]["niveau"] == "n" and out["resume"]["statut"] == "Données incomplètes" and out["resume"]["taux"] is None
    and "Conforme" not in out["statut_html"] and "Données incomplètes" in out["resultat"], (out["resume"], out["statut_html"]))
# la pastille de l'en-tête du téléphone n'a la place que d'un mot : la forme courte (commun.STATUTS_COURTS), lue par
# js/etabli.js ; une note calculée n'en a pas (son taux s'y écrit)
_js_etabli = open(os.path.join(RACINE, "site", "web", "js", "etabli.js"), encoding="utf-8").read()
chk("téléphone : la pastille d'une note sans effort dit « Incomplet » (le statut entier au survol)",
    out["resume"].get("court") == "Incomplet" == C.STATUTS_COURTS["n"] and "res.court || res.statut" in _js_etabli, out["resume"])
non_eff =re.findall(r'<div class="el vc fixe"><div class="l1"><b>([^<]+)</b><span class="et-mot c">non effectuée</span>', out["resultat"])
chk("sans effort tranchant : τ et étriers « non effectuée » (et pourquoi), sans taux", non_eff == ["Effort tranchant", "Espacement des étriers"]
    and "sans effort tranchant (V = 0)" in out["resultat"], non_eff)
dispo = re.findall(r'<b>Dispositions constructives</b><span class="et-mot (\w)">([^<]+)</span>', out["resultat"])
chk("les dispositions constructives disent leur état en mots, sans pourcentage", dispo == [("p", "Conforme")], dispo)
e = P.nouvel_etat()
for chemin, valeur in ((["sections", "0", "M_inf"], "40"), (["sections", "0", "V"], "20"), (["sections", "0", "pas"], "20"), (["sections", "0", "inf", "+"], 1)):
    P.modifier(e, chemin, valeur)
out = req(module="poutre", action="calculer", etat=e)
d0 = MO.verifier(out["etat"]["poutres"][0])["sections"][0]["dispositions"]
chk("deux lits Ø16 au jeu du bureau (1 cm) : 14 mm libres < 25 mm, NON CONFORME (EC2 8.2(2) : un minimum, pas un avis)",
    any(it["cle"] == "entre_lits" and abs(it["libre"] - 14.0) < 1e-9 and it["mini"] == 25.0 and it["etat"] == "nok" for it in d0["items"])
    and out["resume"]["niveau"] == "r", (out["resume"], [i for i in d0["items"] if i["etat"] != "ok"]))
chk("… et les Options le disent, avec le jeu qui suffirait (2,1 cm : 1 + 2,1 + 1 = 4,1 cm d'entraxe, 41 − 16 = 25 mm libres)",
    f"gardent 14{C.NBSP}mm libres" in out["donnees"] and "non conforme" in out["donnees"] and f"Porter ce jeu à 2,1{C.NBSP}cm" in out["donnees"])
# une non-conformité trouvée SANS AUCUN EFFORT (deux lits trop serrés) n'est pas cachée par « Données incomplètes » :
# le bloc de la section est rouge, comme la note (l'ordre de poutre._etat_bloc : non conforme d'abord)
e = P.nouvel_etat()
for chemin, valeur in ((["sections", "0", "pas"], "20"), (["sections", "0", "inf", "+"], 1)):
    P.modifier(e, chemin, valeur)
out = req(module="poutre", action="calculer", etat=e)
r0 = MO.verifier(out["etat"]["poutres"][0])
blocs_etat = re.findall(r'<section class="et-bloc (\w+)" data-section="\d+">.*?<span class="e">([^<]+)</span>', out["resultat"])
chk("sans effort mais deux lits trop serrés : le bloc de la section est « Non conforme » (rouge), jamais « Données incomplètes »",
    not r0["sollicitee"] and r0["statut_complet"] == MO.NOK and blocs_etat == [("nok", "Non conforme")] and out["resume"]["niveau"] == "r",
    (blocs_etat, out["resume"]))

# 14. le ferraillage détaillé se saisit, se vide (automatique), se refuse
e = copy.deepcopy(etat)
pts = lambda: e["poutres"][0]["sections"][0]  # noqa: E731
for chemin, valeur in ((["sections", "0", "inf", "+"], 1), (["sections", "0", "inf", "1", "pos"], "4,5"), (["sections", "0", "inf", "1", "rep"], "entraxe"),
                       (["sections", "0", "inf", "1", "sh"], "6"), (["sections", "0", "laterales", "n"], "2"), (["sections", "0", "etriers", "+"], "epingle"),
                       (["sections", "0", "etriers", "1", "de"], "2")):
    msg = P.modifier(e, chemin, valeur)
    chk(f"saisie du détail {'/'.join(map(str, chemin))} = {valeur!r}", msg is None, msg)
chk("… le lit 2 garde position, répartition et entraxe ; l'épingle sa barre ; les latérales leur nombre",
    pts()["inf"][1] == {"n": 2, "d": 16, "pos": 4.5, "rep": "entraxe", "sh": 6.0} and pts()["etriers"][1] == {"type": "epingle", "d": 10, "de": 2}
    and pts()["laterales"] == {"n": 2}, (pts()["inf"], pts()["etriers"], pts().get("laterales")))
geo = MO.verifier(e["poutres"][0])["sections"][0]["geometrie"]["inf"]["lits"]
chk("… et le calcul les lit : le lit 2 à 4,5 cm du lit 1, ses deux barres à 6 cm d'entraxe, centrées",
    abs(geo[1]["distance"] - geo[0]["distance"] - 4.5) < 1e-9 and abs(geo[1]["x"][1] - geo[1]["x"][0] - 6.0) < 1e-9
    and abs((geo[1]["x"][0] + geo[1]["x"][1]) / 2 - e["poutres"][0]["b"] / 2) < 1e-9, geo[1])
P.modifier(e, ["sections", "0", "inf", "1", "pos"], "")
P.modifier(e, ["sections", "0", "laterales", "n"], " ")
chk("vider un champ automatique rend la règle du module (la clé disparaît)", "pos" not in pts()["inf"][1] and "laterales" not in pts(), (pts()["inf"][1], pts().get("laterales")))
avant = copy.deepcopy(e)
msg = P.modifier(e, ["sections", "0", "inf", "1", "pos"], "haut")
chk("un texte dans un champ automatique est refusé avec un message, la poutre intacte", msg and "n'est pas un nombre" in msg and e == avant, msg)
msg = P.modifier(e, ["sections", "0", "inf", "1", "rep"], "en quinconce")
chk("une répartition inconnue est refusée, la poutre intacte", msg and e == avant, msg)
P.modifier(e, ["sections", "0", "dup"], 1)
chk("« Copier la section » insère sa copie juste après, sous la première lettre libre",
    [x["nom"] for x in e["poutres"][0]["sections"]] == ["A", "C", "B"] and e["poutres"][0]["sections"][1]["inf"] == e["poutres"][0]["sections"][0]["inf"],
    [x["nom"] for x in e["poutres"][0]["sections"]])
e = P.nouvel_etat()
P.modifier(e, ["sections", "0", "inf", "0", "n"], "4")
P.modifier(e, ["sections", "0", "etriers", "+"], "etrier")
chk("un 2ᵉ cadre se place à l'intérieur du premier (barres 2 à n − 1 du lit 1)", e["poutres"][0]["sections"][0]["etriers"][1] == {"type": "etrier", "d": 10, "de": 2, "a": 3},
    e["poutres"][0]["sections"][0]["etriers"])
d1 = MO.verifier(e["poutres"][0])["sections"][0]["dispositions"]
chk("… et un cadre intérieur sous deux barres hautes demande des barres de montage (« À vérifier »)",
    any(it["cle"] == "angles_cadre" and it["etat"] == "warn" and "C2" in it["texte"] for it in d1["items"]), [i["texte"] for i in d1["items"] if i["etat"] != "ok"])

# un nom piégé ne sort pas de son attribut : la cible est une clé du moteur, jamais une saisie
from api_web import dessin_poutre as DP  # noqa: E402
g = DP._cible({"beton": 'x" onclick="alert(1)'}, "beton", "<rect/>")
chk("une cible de pièce est échappée dans son attribut", 'onclick="' not in g and "&quot;" in g, g)
chk("sans cible, le dessin est le même, à l'octet près", DP._cible(None, "beton", "<rect/>") == "<rect/>" and DP._cible({}, "beton", "<rect/>") == "<rect/>")

# 15. (deuxième passe) les barres qu'un cadre entoure : « plage », jamais une barre absente
def sec0(e_):
    return e_["poutres"][0]["sections"][0]


e = P.nouvel_etat()
for chemin, valeur in ((["sections", "0", "inf", "0", "n"], "4"), (["sections", "0", "inf", "0", "d"], "20"), (["sections", "0", "etriers", "+"], "etrier")):
    P.modifier(e, chemin, valeur)
m1 = P.modifier(e, ["sections", "0", "etriers", "0", "plage"], "1-3")
m2 = P.modifier(e, ["sections", "0", "etriers", "1", "plage"], "4–2")
chk("le cas du bureau : deux cadres sur les barres 1 → 3 et 2 → 4 (l'ordre des bouts, le tiret long : indifférents)",
    m1 is None and m2 is None and sec0(e)["etriers"] == [{"type": "etrier", "d": 10, "de": 1, "a": 3}, {"type": "etrier", "d": 10, "de": 2, "a": 4}],
    (m1, m2, sec0(e)["etriers"]))
res0 = MO.verifier(e["poutres"][0])["sections"][0]
xs0 = res0["geometrie"]["inf"]["lits"][0]["x"]
cds = res0["geometrie"]["etriers"]["cadres"]
c0, b0 = res0["geometrie"]["enrobage_calcul"], e["poutres"][0]["b"]
chk("… le moteur pose chaque cadre autour de SES barres, dans l'enrobage (jamais hors de la poutre)",
    all(c0 - 1e-9 <= k["x0"] < k["x1"] <= b0 - c0 + 1e-9 for k in cds) and cds[0]["x0"] < xs0[0] - 1.0 and cds[0]["x1"] > xs0[2] + 1.0
    and xs0[0] < cds[1]["x0"] < xs0[1] - 1.0 and cds[1]["x1"] > xs0[3] + 1.0 and xs0[2] < cds[0]["x1"] < xs0[3], (cds, xs0))
out = req(module="poutre", action="calculer", etat=e)
g_cadres = re.findall(r'<g class="pc cadre" data-sel="s0\.cadre-(\d)" data-cadre="\d" data-type="(\w+)" data-brins="\d" data-de="(\d+)" data-a="(\d+)">', out["vues"][0]["svg"])
chk("… et le dessin dit, pour chaque cadre, les barres que le moteur lui a données", g_cadres == [("0", "etrier", "1", "3"), ("1", "etrier", "2", "4")], g_cadres)
pick = re.findall(r'<div class="et-barres" data-plage="sections/0/etriers/(\d)/plage" data-type="(\w+)" data-de="(\d)" data-a="(\d)" data-n="(\d)">(.*?)</div>', out["donnees"])
chk("… et les données : le choix des barres de chaque cadre, ses bouts éclairés, aucune barre de trop",
    [(k, t, d_, a_, n_) for k, t, d_, a_, n_, _ in pick] == [("0", "etrier", "1", "3", "4"), ("1", "etrier", "2", "4", "4")]
    and all(re.findall(r'data-barre="(\d)"', h_) == ["1", "2", "3", "4"] for *_, h_ in pick)
    and re.findall(r'class="b on bout" data-barre="(\d)"', pick[1][5]) == ["2", "4"], pick)
refus = []
for valeur in ("3-3", "2-7", "0-2", "x", "1-2-3", "5"):
    avant = copy.deepcopy(e)
    msg = P.modifier(e, ["sections", "0", "etriers", "1", "plage"], valeur)
    if not msg or e != avant:
        refus.append((valeur, msg))
chk("une plage impossible (une seule barre, une barre absente, la barre 0, un texte, trois bouts) est refusée, la poutre intacte", not refus, refus)
out = req(module="poutre", action="modifier", etat=e, chemin=["sections", "0", "etriers", "1", "plage"], valeur="2-7")
chk("… refusée par l'écran : le message dit les barres qui existent, l'état renvoyé est l'ancien",
    out["ok"] and out["message"] == "Le lit 1 inférieur compte 4 barres : barres 1 à 4" and out["etat"] == e, out.get("message"))
e2 = copy.deepcopy(e)
P.modifier(e2, ["sections", "0", "etriers", "1", "plage"], "")
e3 = copy.deepcopy(e)
P.modifier(e3, ["sections", "0", "etriers", "1", "plage"], "1-4")
chk("« Toute la largeur » (vide) et « 1-4 » sur quatre barres écrivent le même cadre complet (sans de ni à)",
    sec0(e2)["etriers"][1] == sec0(e3)["etriers"][1] == {"type": "etrier", "d": 10}, (sec0(e2)["etriers"], sec0(e3)["etriers"]))
e4 = copy.deepcopy(e)
P.modifier(e4, ["sections", "0", "etriers", "1", "type"], "epingle")
m4 = P.modifier(e4, ["sections", "0", "etriers", "1", "plage"], "3")
chk("une épingle se pose sur UNE barre (« 3 »)", m4 is None and sec0(e4)["etriers"][1] == {"type": "epingle", "d": 10, "de": 3}, (m4, sec0(e4)["etriers"]))
msg = P.modifier(e, ["sections", "0", "inf", "0", "n"], "3")
chk("retirer une barre au lit 1 ramène le cadre qui l'entourait sur les barres qui restent, et le dit",
    msg == "Le lit 1 inférieur compte moins de barres — C2 : cadre ramené aux barres 2 à 3." and sec0(e)["etriers"][1] == {"type": "etrier", "d": 10, "de": 2, "a": 3}
    and sec0(e)["etriers"][0] == {"type": "etrier", "d": 10, "de": 1, "a": 3}, (msg, sec0(e)["etriers"]))
out = req(module="poutre", action="choisir", etat=e, role="inf-0", nom="2Ø20")
chk("… le sélecteur aussi : 2Ø20 posés, les deux cadres reviennent sur toute la largeur (aucune barre absente)",
    out["ok"] and [x for x in out["etat"]["poutres"][0]["sections"][0]["etriers"]] == [{"type": "etrier", "d": 10}, {"type": "etrier", "d": 10}],
    out["etat"]["poutres"][0]["sections"][0]["etriers"] if out["ok"] else out.get("erreur"))

# 16. la liste de contrôle : une observation « à examiner » se justifie (et se garde) ; une non-conformité, jamais
e = P.nouvel_etat()
for chemin, valeur in ((["b"], "40"), (["h"], "60"), (["sections", "0", "M_inf"], "150"), (["sections", "0", "V"], "80"), (["sections", "0", "pas"], "20"),
                       (["sections", "0", "inf", "0", "n"], "5"), (["sections", "0", "inf", "0", "d"], "20"), (["sections", "0", "sup", "0", "d"], "16"),
                       (["sections", "0", "etriers", "+"], "etrier")):
    m_ = P.modifier(e, chemin, valeur)
    assert m_ is None, (chemin, m_)
out = req(module="poutre", action="calculer", etat=e)
d_ = MO.verifier(e["poutres"][0])["sections"][0]["dispositions"]
warn = [it for it in d_["items"] if it["etat"] == "warn"]
boutons = re.findall(r'data-justifier="0" data-obs="([^"]+)" data-empreinte="([^"]*)"', out["resultat"])
chk("un cadre intérieur sous deux barres hautes : « à examiner », et SEULES ces observations ont « Marquer examinée… »",
    warn and sorted(boutons) == sorted((it["id"], it["empreinte"]) for it in warn) and MO.verifier(e["poutres"][0])["statut"] != MO.NOK
    and 'class="et-bloc warn"' in out["resultat"], (boutons, [it["id"] for it in warn]))
cible = warn[0]
out = req(module="poutre", action="modifier", etat=e, chemin=["sections", "0", "justifs", cible["id"]],
          valeur={"texte": "  Barres de montage Ø8 au plan.  ", "empreinte": cible["empreinte"]})
s_j = out["etat"]["poutres"][0]["sections"][0]
chk("… la justification se garde dans la note (texte nettoyé, empreinte de ce qui a été examiné)",
    out["ok"] and not out["message"] and s_j.get("justifs") == {cible["id"]: {"texte": "Barres de montage Ø8 au plan.", "empreinte": cible["empreinte"]}}, s_j.get("justifs"))
obs_j = re.search(r'<div class="et-obs j" data-obs="' + re.escape(cible["id"]) + r'".*?</div></div>', out["resultat"], re.S)
chk("… l'observation passe « Examinée — justifiée », son texte cité, « Annuler » à côté",
    obs_j is not None and "Examinée — justifiée" in obs_j.group(0) and "« Barres de montage Ø8 au plan. »" in obs_j.group(0)
    and f'data-dejustifier="0" data-obs="{cible["id"]}"' in obs_j.group(0), obs_j.group(0)[:300] if obs_j else out["resultat"][:300])
reste = len(warn) - 1
chk("… et l'en-tête de la section le compte", (f'class="et-bloc ok"' in out["resultat"] and "Conforme · 1 justification" in out["resultat"]) if not reste
    else 'class="et-bloc warn"' in out["resultat"], re.findall(r'<span class="e">([^<]+)</span>', out["resultat"]))
e_j = out["etat"]
P.modifier(e_j, ["b"], "45")
it_ = next(x for x in MO.verifier(e_j["poutres"][0])["sections"][0]["dispositions"]["items"] if x["id"] == cible["id"])
chk("la géométrie change : la justification ne vaut plus (« à examiner » de nouveau, « justification à revoir »)",
    it_["etat_eff"] == "warn" and it_.get("justification_perimee") == "Barres de montage Ø8 au plan.", it_)
out = req(module="poutre", action="modifier", etat=out["etat"], chemin=["sections", "0", "justifs", cible["id"]], valeur=None)
chk("« Annuler » retire la justification (la clé disparaît de la note)", out["ok"] and "justifs" not in out["etat"]["poutres"][0]["sections"][0], out["etat"]["poutres"][0]["sections"][0].get("justifs"))
# une non-conformité : 2 lits Ø20 au jeu du bureau (10 mm libres < 25 mm)
e5 = P.nouvel_etat()
for chemin, valeur in ((["sections", "0", "M_inf"], "40"), (["sections", "0", "V"], "20"), (["sections", "0", "inf", "0", "d"], "20"), (["sections", "0", "inf", "+"], 1)):
    P.modifier(e5, chemin, valeur)
nok = next(x for x in MO.verifier(e5["poutres"][0])["sections"][0]["dispositions"]["items"] if x["cle"] == "entre_lits")
out = req(module="poutre", action="modifier", etat=e5, chemin=["sections", "0", "justifs", nok["id"]], valeur={"texte": "vu", "empreinte": nok["empreinte"]})
nok2 = next(x for x in MO.verifier(out["etat"]["poutres"][0])["sections"][0]["dispositions"]["items"] if x["id"] == nok["id"])
chk("une NON-CONFORMITÉ (10 mm < 25 mm) ne se justifie pas : aucun bouton, et même forcée dans la note elle reste non conforme",
    nok["etat"] == "nok" and nok2["etat_eff"] == "nok" and f'data-obs="{nok["id"]}" data-empreinte' not in out["resultat"]
    and 'class="et-bloc nok"' in out["resultat"], (nok["etat"], nok2["etat_eff"]))
grp_ = re.findall(r'<div class="et-obs-g (\w)">([^<]+)<span>(\d+)</span>', out["resultat"])
items5 = MO.verifier(out["etat"]["poutres"][0])["sections"][0]["dispositions"]["items"]
chk("la liste de contrôle se range par état : non conformes, à examiner, justifiées, conformes automatiquement",
    [(t, str(sum(x["etat_eff"] == e_ for x in items5))) for e_, t in P.GROUPES_OBS if any(x["etat_eff"] == e_ for x in items5)] == [g_[1:] for g_ in grp_]
    and grp_[0][:2] == ("r", "Non conformes"), grp_)
conformes = re.findall(r'<div class="et-obs p"[^>]*>.*?</div></div>', out["resultat"], re.S)
chk("une observation conforme tient en une ligne (ni mot d'état, ni bouton)", conformes and not any('class="et-mot' in x or "<button" in x for x in conformes), conformes[:1])
pieces_ = set().union(*(set(re.findall(r'data-sel="([^"]+)"', v["svg"])) for v in out["vues"]))
mal = [k for k in re.findall(r'data-piece="([^"]+)"', out["resultat"]) if k not in pieces_]
chk("chaque observation désigne une pièce du dessin (un clic la montre)", not mal and 'data-piece="s0.lit-inf-1"' in out["resultat"], mal)

# 17. le dessin : l'en-tête technique de chaque vue ; les poignées des cadres cachées tant que rien n'est choisi
out = req(module="poutre", action="calculer", etat=etat)
ent = out["vues"][0]["entete"]
chk("chaque vue a son en-tête : la section et ses matériaux, les aciers, les étriers, et la bande des trois efforts",
    all("entete" in v for v in out["vues"]) and "<b>SECTION A</b>" in ent and "· 20 × 40 cm · C30/37 · B500" in ent
    and "Inf. 2Ø16 · Sup. 2Ø16" in ent and "1 cadre Ø10 · e = 30 cm" in ent and ent.count("<span><i>") == 3 and ">122,0<" in ent and ">60,0<" in ent, ent)
chk("le dessin ne porte plus « mm · à l'échelle » (il s'ajuste à la feuille) : l'écran ne l'écrit pas pour l'établi compact",
    "const echelle = (v) => !COMPACT &&" in _js_etabli)
svg0 = out["vues"][0]["svg"]
chk("les poignées des cadres sont cachées tant que le cadre n'est pas choisi ; une zone de saisie élargit chaque acier",
    svg0.count('<g class="poignees" visibility="hidden">') == len(re.findall(r'<g class="pc cadre" data-sel="[^"]+" data-cadre=', svg0)) > 0
    and 'class="zone"' in svg0
    and 'pointer-events="stroke"' in svg0, svg0.count('class="poignees"'))

# 18. le résultat : le nom de la poutre, puis les sections — plus de grand verdict global
chk("le résultat commence par le nom de la poutre, sans grand indicateur global (ni verdict, ni pastille XL)",
    out["resultat"].startswith('<div class="pbody"><div class="et-rp"><b>V1</b><span>20 × 40 cm · C30/37 · B500</span></div>')
    and 'class="verdict' not in out["resultat"] and "chip xl" not in out["resultat"], out["resultat"][:160])
tau_ = re.search(r'<b>Effort tranchant</b><span class="prof"><span>([^<]+)</span></span>', out["resultat"])
chk("l'effort tranchant nomme ce que demande sa bande à côté de son titre (jamais « pas besoin d'étriers » seul)",
    tau_ is not None and tau_.group(1) in P.BESOINS_COURTS.values(), tau_.group(0) if tau_ else "")

# 19. la ligne d'une poutre : son nom (un clic le renomme), copier CETTE poutre (la copie s'insère après elle), supprimer
# (jamais la dernière)
e = P.nouvel_etat()
out = req(module="poutre", action="calculer", etat=e)
chk("une poutre seule : sa ligne offre le nom en champ (plus de « Renommer »), « Copier la poutre » et « Supprimer » éteint",
    '<input data-chemin="nom" data-sorte="texte" value="V1" class="et-elt-nom"' in out["donnees"] and "data-renommer" not in out["donnees"]
    and re.search(r'<button type="button" class="et-elt-b" data-dupliquer-index="0" [^>]*aria-label="Copier la poutre">', out["donnees"]) is not None
    and re.search(r'class="et-elt-b danger" data-supprimer="0" data-confirmer="Supprimer la poutre « V1 » de la note \?" '
                  r'title="Une note garde au moins une poutre" aria-label="Supprimer la poutre" disabled>', out["donnees"]) is not None)
P.gerer_elements(e, "ajouter")
out = req(module="poutre", action="dupliquer", etat=e, index=0)
chk("« Copier la poutre » (V1, V2 ouverte) : la copie de V1 s'insère après elle, et s'ouvre",
    out["ok"] and [x["nom"] for x in out["etat"]["poutres"]] == ["V1", "V1 (copie)", "V2"] and out["etat"]["actif"] == 1, [x["nom"] for x in out["etat"]["poutres"]])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
