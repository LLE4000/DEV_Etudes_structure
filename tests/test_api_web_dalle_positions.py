# -*- coding: utf-8 -*-
"""Les NIVEAUX réels des deux directions d'une dalle (api_web/dalle_positions.py) et leur dessin (api_web/dessin_dalle.py,
api_web/dessin_predalle.py), vérifiés à la main — docs/site/DALLES_FUSION.md, § 4 ; la règle de position du 13/11/2026
(beton/position.py, tests/test_position.py) : le site calcule des POSITIONS RÉELLES (c + Ø/2 ; une couche posée sur une
autre : le dessus de celle-ci + Ø/2 ; les Ø réels, aucun arrondi) ; le moteur en fait l'axe de CALCUL (⌈réel + jeu⌉ au
0,5 cm). Les nappes sont TOUJOURS posées l'une sur l'autre (décision de l'ingénieur du 09/11/2026) et un RENFORT de la
direction secondaire se pose sur sa nappe, comme sa base (S3).

(a) ce qui reste à la position du moteur : la direction principale (base ET renforts), quel que soit le type ; la clé
    « empilement » d'une note d'avant ne change rien ;
(b) dalle coulée c = 3 cm, principale Y, bases Ø12 en Y et Ø10 en X : inf. Y 3,6 ; inf. X 3,6 + 0,6 + 0,5 = 4,7 ; sup.
    pareil ; S3 : un renfort secondaire 3 Ø12 à 4,2 + 0,6 = 4,8 (Ø25 : 5,45), sur la principale saisie ; le jeu de pose
    n'entre que dans le calcul ; un renfort saisi n'est pas déplacé ; les moteurs posent un renfort dans le LIT de sa base ;
(c) prédalle : les deux BASES inférieures dans la peau (la secondaire posée sur la principale, 4,7), les renforts
    inférieurs sur la prédalle (6 + Ø/2), les faces supérieures empilées ;
(d) pour_moteur (la position réelle d'une couche empilée, rien d'autre), annoter (positions, détail de l'axe, empilement,
    formule « e »), candidats_base (chaque candidat à SA position empilée) ;
(e) les dessins : chaque couche à sa position RÉELLE ; les cotes de d sont celles du calcul ; les légendes ;
(f) mutations (en mémoire, le dépôt intact) : chacune fait rougir le contrôle qui la garde ;
(g) S5 : la nappe secondaire repose sur la couche principale la plus HAUTE de son lit de contact ; les cas limites du lit
    (contact exact, 0,01 cm au-dessus, chaîne, base saisie, égalités, treillis, jeu) ; le balayage de tous les Ø ;
    A_s,req ne baisse jamais ; les chiffres de la règle (d inf. X 14,5 → 14,0 → 13,0) ;
(h) DAL5b : un treillis unique a ses fils soudés l'un sur l'autre, ses renforts principaux posés sur lui ; dans la peau
    d'une prédalle aussi (et l'avertissement « secondaire_hors_predalle ») ;
(i) DAL5c : les flèches (un pas de 0,5 cm, sur la grille), jamais au travers d'une couche ni de l'enrobage.

    /usr/bin/python3.12 tests/test_api_web_dalle_positions.py
"""
import copy
import html
import json
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import dalle_positions as POS  # noqa: E402
from api_web import dessin_dalle as DD  # noqa: E402
from api_web import dessin_predalle as DP  # noqa: E402
from beton import treillis as TR  # noqa: E402
from beton.dalle import entree as EN_D  # noqa: E402
from beton.dalle import moteur as MO_D  # noqa: E402
from beton.predalle import entree as EN_P  # noqa: E402
from beton.predalle import moteur as MO_P  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def eq(a, b):
    """Deux positions égales aux arrondis des flottants près (un nombre, ou une liste)."""
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return abs(a - b) < 1e-9


BARRES_12 = {"type": "barres", "treillis": "10/10/100/100", "d": 12, "esp": 150, "n": 3, "dist": None}


def dalle(type_="coulee", principale="Y"):
    """Dalle C ouverte c = 3 cm (jeu de pose 1 cm, arrondi 0,5 cm : les défauts), principale Y : bases Ø12/150 en Y,
    treillis 10/10/100/100 (Ø10) en X, un renfort 3 Ø12 en inf. Y ; moments sur les quatre faces (inf. plus chargées).
    En prédalle : h = 22, h_pre = 6."""
    d = EN_D.nouvelle_dalle() if type_ == "coulee" else EN_P.nouvelle_predalle()
    s = d["sections"][0]
    for f in ("inf_y", "sup_y"):
        s["faces"][f]["couches"][0] = dict(BARRES_12)
    s["faces"]["inf_y"]["couches"].append(EN_D.nouvelle_couche(base=False))
    s.update(Mx_inf=15.0, My_inf=30.0, Mx_sup=5.0, My_sup=8.0, V=40.0)
    d["dir_principale"] = principale
    d = (EN_D if type_ == "coulee" else EN_P).normaliser(d)
    d.update(verrou=False, type=type_)
    return d


def dalle_s3(type_="coulee", d_inf=12, d_sup=16):
    """La même, avec un RENFORT de la direction secondaire (X) sur chaque face : 3 Ø``d_inf`` en inf. X, 3 Ø``d_sup`` en
    sup. X (S3)."""
    d = dalle(type_)
    s = d["sections"][0]
    for f, dd in (("inf_x", d_inf), ("sup_x", d_sup)):
        s["faces"][f]["couches"].append(dict(EN_D.nouvelle_couche(base=False), d=dd))
    return d


def moteur_de(d):
    """La dalle brute, sans les clés du site : ce que le moteur SEUL calculerait (il ne pose rien sur rien)."""
    return {k: v for k, v in d.items() if k not in POS.CLES_SITE}


def autos_moteur(d):
    """La position RÉELLE automatique DU MOTEUR de chaque couche, par face (au contact de l'enrobage ; un renfort
    inférieur de prédalle sur la prédalle)."""
    s = d["sections"][0]
    if d["type"] == "predalle":
        return {f: [MO_P.position_auto(c, j, f, d["dir_principale"], d["enrobage"], d["h_pre"])
                    for j, c in enumerate(s["faces"][f]["couches"])] for f in EN_D.FACES}
    return {f: [MO_D.position_auto(c, d["enrobage"]) for c in s["faces"][f]["couches"]] for f in EN_D.FACES}


def autos_site(POSm, d):
    s = d["sections"][0]
    return {f: [POSm.position_auto(d, s, f, j) for j in range(len(s["faces"][f]["couches"]))] for f in EN_D.FACES}


# ------------------------------------------------------------------ (a) ce qui reste à la position du moteur
print("=== (a) direction principale : exactement la position du moteur ===")
for type_ in ("coulee", "predalle"):
    for p in ("Y", "X"):
        d = dalle_s3(type_)
        d["dir_principale"] = p
        a_s, a_m = autos_site(POS, d), autos_moteur(d)
        garde = [f for f in EN_D.FACES if f.endswith(p.lower())]
        chk(f"{type_}, principale {p} : position_auto == position du moteur pour {', '.join(garde)} (base et renforts) ; la "
            "secondaire posée sur elle (sa base, au moins)",
            all(eq(a_s[f], a_m[f]) for f in garde) and all(not eq(a_s[f], a_m[f]) for f in EN_D.FACES if f not in garde), (a_s, a_m))
chk("la clé « empilement » d'une note d'avant (vraie, fausse, absente) ne change rien : les mêmes positions",
    autos_site(POS, dict(dalle(), empilement=False)) == autos_site(POS, dict(dalle(), empilement=True)) == autos_site(POS, dalle())
    and not hasattr(POS, "empilement") and POS.CLES_SITE == {"verrou", "type"})
chk("sans la clé « type » : dalle coulée", POS.type_dalle({"nom": "x"}) == "coulee" and POS.moteur({"nom": "x"}) is MO_D)

# ------------------------------------------------------------------ (b) dalle coulée : la secondaire posée sur la principale
print("\n=== (b) dalle coulée : la direction secondaire (base ET renforts) posée sur la principale ===")
d = dalle()
s = d["sections"][0]
a = autos_site(POS, d)
chk("inf. Y (principale, Ø12) : 3 + 0,6 = 3,6 cm — la position du moteur", eq(a["inf_y"][0], 3.6) and eq(MO_D.position_auto(BARRES_12, 3.0), 3.6), a)
chk("inf. X (secondaire, Ø10) : posée sur la base Y, 3 + 1,2 + 0,5 = 4,7 cm", eq(a["inf_x"][0], 4.7), a)
chk("sup. Y 3,6 et sup. X 4,7 cm (la même règle depuis le dessus)", eq((a["sup_y"][0], a["sup_x"][0]), (3.6, 4.7)), a)
chk("le renfort 3 Ø12 en inf. Y (principale) : la position du moteur (3 + 0,6 = 3,6), pas empilé",
    eq(a["inf_y"][1], 3.6) and eq(autos_moteur(d)["inf_y"][1], 3.6)
    and not POS.empilee(d, s, "inf_y", 1) and POS.empilee(d, s, "inf_x", 0) and not POS.empilee(d, s, "inf_y", 0))
dx = dalle(principale="X")
ax_ = autos_site(POS, dx)
chk("principale X : la base X (Ø10) à 3,5, la base Y (Ø12) posée dessus à 3,5 + 0,5 + 0,6 = 4,6 cm ; le renfort inf. Y (devenu "
    "secondaire) aussi : 4,6 (S3)",
    eq((ax_["inf_x"][0], ax_["inf_y"][0], ax_["sup_x"][0], ax_["sup_y"][0], ax_["inf_y"][1]), (3.5, 4.6, 3.5, 4.6, 4.6)), ax_)
di = dalle()
di["sections"][0]["faces"]["inf_y"]["couches"][0]["dist"] = 5.0
chk("la base principale SAISIE à 5,0 cm (une position réelle) : la secondaire repose dessus, 5,0 + 0,6 + 0,5 = 6,1 ; ses axes de "
    "calcul : 5,0 + 1 (jeu) = 6,0 et le renfort 3,6 + 1 = 4,6 → 5,0",
    eq(POS.position_auto(di, di["sections"][0], "inf_x", 0), 6.1) and eq(POS.positions(di, di["sections"][0])["inf_y"], [5.0, 3.6])
    and POS.axes(di, di["sections"][0])["inf_y"] == [6.0, 5.0], POS.axes(di, di["sections"][0]))
d0 = dalle()
d0["jeu_premier_lit"] = 0.0
chk("le jeu de pose n'entre que dans le CALCUL : les positions ne bougent pas (3,6 et 4,7) ; les axes ⌈3,6 + 1⌉ = 5,0 et "
    "⌈4,7 + 1⌉ = 6,0 avec le jeu de 1 cm, ⌈3,6⌉ = 4,0 et ⌈4,7⌉ = 5,0 sans",
    autos_site(POS, d0) == autos_site(POS, d) and POS.axes(d, s)["inf_y"][0] == 5.0 and POS.axes(d, s)["inf_x"][0] == 6.0
    and POS.axes(d0, d0["sections"][0])["inf_y"][0] == 4.0 and POS.axes(d0, d0["sections"][0])["inf_x"][0] == 5.0)
chk("positions(d, s) : la position saisie sinon l'automatique du site ; axes(d, s) : leur axe de calcul, pour les quatre faces",
    all(eq(POS.positions(d, s)[f], v) for f, v in {"inf_x": [4.7], "sup_x": [4.7], "inf_y": [3.6, 3.6], "sup_y": [3.6]}.items())
    and POS.axes(d, s) == {"inf_x": [6.0], "sup_x": [6.0], "inf_y": [5.0, 5.0], "sup_y": [5.0]}, (POS.positions(d, s), POS.axes(d, s)))
# S3 : le renfort de la direction secondaire sur sa nappe
d3 = dalle_s3()
s3 = d3["sections"][0]
a3 = autos_site(POS, d3)
chk("S3 — renfort 3 Ø12 en inf. X (secondaire) : 4,2 + 0,6 = 4,8 cm, posé sur la base Y comme la base X (le moteur seul : 3,6, au "
    "niveau de la base Y, sous sa propre nappe) ; renfort 3 Ø16 en sup. X : 4,2 + 0,8 = 5,0 (depuis le dessus)",
    eq(a3["inf_x"], [4.7, 4.8]) and eq(a3["sup_x"], [4.7, 5.0]) and eq(autos_moteur(d3)["inf_x"][1], 3.6)
    and POS.empilee(d3, s3, "inf_x", 1) and POS.empilee(d3, s3, "sup_x", 1), (a3, autos_moteur(d3)))
d25 = dalle_s3(d_inf=25)
chk("S3 — un renfort Ø25 en inf. X : 4,2 + 1,25 = 5,45 cm (le moteur seul : 3 + 1,25 = 4,25)",
    eq(autos_site(POS, d25)["inf_x"][1], 5.45) and eq(autos_moteur(d25)["inf_x"][1], 4.25))
d3i = dalle_s3()
d3i["sections"][0]["faces"]["inf_y"]["couches"][0]["dist"] = 5.0
chk("S3 — sur la base principale SAISIE à 5,0 : le renfort secondaire 5,0 + 0,6 + 0,6 = 6,2",
    eq(POS.position_auto(d3i, d3i["sections"][0], "inf_x", 1), 6.2))
d3s = dalle_s3()
d3s["sections"][0]["faces"]["inf_x"]["couches"][1]["dist"] = 8.0
chk("S3 — un renfort secondaire SAISI (8 cm) n'est pas déplacé : positions() le garde, il n'est pas « empilé », pour_moteur n'y "
    "touche pas", eq(POS.positions(d3s, d3s["sections"][0])["inf_x"], [4.7, 8.0]) and not POS.empilee(d3s, d3s["sections"][0], "inf_x", 1)
    and POS.pour_moteur(d3s)["sections"][0]["faces"]["inf_x"]["couches"][1]["dist"] == 8.0)
# la preuve que la formule du renfort est celle de sa base : les deux moteurs posent un renfort dans le LIT de sa base
# (la position moins le demi-diamètre est la même pour toutes les couches d'une face : l'enrobage)
lits = []
for type_ in ("coulee", "predalle"):
    for f in EN_D.FACES:
        if type_ == "predalle" and f.startswith("inf"):
            continue
        dd_ = dalle(type_)
        for diam in EN_D.DIAMETRES:
            base, renf = dd_["sections"][0]["faces"][f]["couches"][0], dict(EN_D.nouvelle_couche(base=False), d=diam)
            lit_b = POS._reel_moteur(dd_, base, 0, f) - POS._demi(base)
            lit_r = POS._reel_moteur(dd_, renf, 1, f) - POS._demi(renf)
            if abs(lit_b - lit_r) > 1e-12:
                lits.append((type_, f, diam, lit_b, lit_r))
chk("les moteurs posent un renfort dans le LIT de la base de sa face (dalle : toutes les faces ; prédalle : les supérieures), pour "
    "chaque Ø : son dessous est celui de la base — empilé, il le reste (position − Ø/2 = celle de la base secondaire − Ø_s/2)",
    not lits and all(abs((a3[f][1] - POS._demi(s3["faces"][f]["couches"][1])) - (a3[f][0] - POS._demi(s3["faces"][f]["couches"][0]))) < 1e-12
                     for f in ("inf_x", "sup_x")), lits)

# ------------------------------------------------------------------ (c) prédalle
print("\n=== (c) prédalle : les deux bases dans la peau, les renforts inférieurs sur la prédalle, empilement en haut ===")
dp = dalle("predalle")
ap_ = autos_site(POS, dp)
chk("inf. Y (principale) dans la peau : c + Ø/2 = 3,6 ; renfort inf. Y sur la prédalle : h_p + Ø/2 = 6 + 0,6 = 6,6 — la position du "
    "moteur ; inf. X (secondaire) DANS la peau, posée sur la base Y : 3 + 1,2 + 0,5 = 4,7 (le moteur seul : 3,5 ; l'écran "
    "Streamlit la posait sur la prédalle, 6,5)",
    eq(ap_["inf_y"], [3.6, 6.6]) and eq(ap_["inf_y"], autos_moteur(dp)["inf_y"]) and eq(ap_["inf_x"], [4.7])
    and eq(autos_moteur(dp)["inf_x"], [3.5]), ap_)
chk("sup. Y 3,6 (moteur) ; sup. X empilée 4,7 (le moteur dirait 3,5)",
    eq((ap_["sup_y"][0], ap_["sup_x"][0]), (3.6, 4.7)) and eq(autos_moteur(dp)["sup_x"][0], 3.5), ap_)
chk("empilee : vrai pour les deux bases secondaires (inf. X dans la peau, sup. X)",
    [f for f in EN_D.FACES if POS.empilee(dp, dp["sections"][0], f, 0)] == [f for f in EN_D.FACES if f in ("inf_x", "sup_x")])
dp3 = dalle_s3("predalle")
ap3 = autos_site(POS, dp3)
chk("S3 en prédalle — le renfort sup. X (3 Ø16) posé sur la base sup. Y : 4,2 + 0,8 = 5,0 (le moteur : 3 + 0,8 = 3,8) ; le renfort "
    "inf. X (3 Ø12) reste SUR la prédalle à 6 + 0,6 = 6,6, la position du moteur (posé au chantier)",
    eq(ap3["sup_x"], [4.7, 5.0]) and eq(autos_moteur(dp3)["sup_x"][1], 3.8) and eq(ap3["inf_x"], [4.7, 6.6])
    and eq(autos_moteur(dp3)["inf_x"], [3.5, 6.6])
    and not POS.empilee(dp3, dp3["sections"][0], "inf_x", 1) and POS.empilee(dp3, dp3["sections"][0], "sup_x", 1), ap3)

# ------------------------------------------------------------------ (d) pour_moteur, annoter, candidats_base
print("\n=== (d) pour_moteur : la copie que le moteur lit ; annoter ; le sélecteur ===")
d = dalle()
avant = copy.deepcopy(d)
pm = POS.pour_moteur(d)
chk("pour_moteur : dist = la position RÉELLE (4,7) pour inf. X et sup. X seulement, les autres couches intactes (None)",
    {f: [c["dist"] for c in pm["sections"][0]["faces"][f]["couches"]] for f in EN_D.FACES}
    == {"inf_x": [4.7], "sup_x": [4.7], "inf_y": [None, None], "sup_y": [None]}, pm["sections"][0]["faces"])
chk("pour_moteur : sans verrou ni type ; la dalle d'origine n'est pas modifiée",
    not (set(pm) & POS.CLES_SITE) and d == avant and d["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] is None)
pm3 = POS.pour_moteur(dalle_s3())
chk("pour_moteur (S3) : les renforts secondaires reçoivent leur position empilée (4,8 et 5,0), le renfort principal reste automatique",
    {f: [c["dist"] for c in pm3["sections"][0]["faces"][f]["couches"]] for f in ("inf_y", "sup_y")} == {"inf_y": [None, None], "sup_y": [None]}
    and eq([c["dist"] for c in pm3["sections"][0]["faces"]["inf_x"]["couches"]], [4.7, 4.8])
    and eq([c["dist"] for c in pm3["sections"][0]["faces"]["sup_x"]["couches"]], [4.7, 5.0]), pm3["sections"][0]["faces"])
r_emp = MO_D.verifier(pm)
r_mot = MO_D.verifier(moteur_de(d))
g_e, g_m = r_emp["sections"][0]["grandeurs"], r_mot["sections"][0]["grandeurs"]
chk("verifier(pour_moteur(d)) : d utile de la direction secondaire plus petit (inf. X 20 − ⌈4,7 + 1⌉ = 14,0 contre 20 − ⌈3,5 + 1⌉ "
    "= 15,5 ; sup. X pareil), la principale inchangée",
    g_e["d_inf_x"] == 14.0 < g_m["d_inf_x"] == 15.5 and g_e["d_sup_x"] == 14.0 < g_m["d_sup_x"] and g_e["d_inf_y"] == g_m["d_inf_y"]
    and g_e["As_req_inf_x"] > g_m["As_req_inf_x"], (g_e["d_inf_x"], g_m["d_inf_x"]))
chk("… et l'avertissement « nappes_confondues » du moteur disparaît (les bases ne sont plus au même niveau : 3,5 et 3,6)",
    [x["code"] for x in r_mot["avertissements"]] == ["nappes_confondues", "nappes_confondues"] and not r_emp["avertissements"])
s = d["sections"][0]
chk("h_utile(d, s, f) du site = d du calcul, pour les quatre faces", all(abs(POS.h_utile(d, s, f) - g_e[f"d_{f}"]) < 1e-12 for f in EN_D.FACES))
chk("yG_calcule(d, s, f) du site = y_G calculé du moteur", all(abs(POS.yG_calcule(d, s, f) - g_e[f"yG_calcule_{f}"]) < 1e-12 for f in EN_D.FACES))
r3e, r3m = MO_D.verifier(pm3), MO_D.verifier(moteur_de(dalle_s3()))
chk("S3 — le sens de la sécurité : le renfort secondaire monte, y_G inf. X monte, A_s,req inf. X augmente (sup. X pareil)",
    all(r3e["sections"][0]["grandeurs"][f"yG_{f}"] > r3m["sections"][0]["grandeurs"][f"yG_{f}"]
        and r3e["sections"][0]["grandeurs"][f"As_req_{f}"] > r3m["sections"][0]["grandeurs"][f"As_req_{f}"] for f in ("inf_x", "sup_x")),
    {f: (r3e["sections"][0]["grandeurs"][f"As_req_{f}"], r3m["sections"][0]["grandeurs"][f"As_req_{f}"]) for f in ("inf_x", "sup_x")})
for type_ in ("coulee", "predalle"):
    d0 = dalle(type_)
    for f in ("inf_x", "sup_x"):                                  # tous les axes secondaires saisis : rien à empiler
        d0["sections"][0]["faces"][f]["couches"][0]["dist"] = 7.0
    MO = POS.moteur(d0)
    chk(f"{type_}, positions secondaires SAISIES (rien d'empilé) : pour_moteur n'écrit rien, verifier(pour_moteur(d)) == verifier(d) "
        "à l'octet (JSON)", json.dumps(MO.verifier(POS.pour_moteur(d0)), sort_keys=True) == json.dumps(MO.verifier(moteur_de(d0)), sort_keys=True))
dp = dalle("predalle")
pmp = POS.pour_moteur(dp)
chk("prédalle : pour_moteur renseigne inf. X (4,7, dans la peau) et sup. X (4,7), h_pre passe au moteur",
    {f: [c["dist"] for c in pmp["sections"][0]["faces"][f]["couches"]] for f in EN_D.FACES}
    == {"inf_x": [4.7], "sup_x": [4.7], "inf_y": [None, None], "sup_y": [None]} and pmp["h_pre"] == 6.0)
# annoter : la notion d'automatique du site posée sur le résultat
r = POS.annoter(MO_D.verifier(pm), d)
cx, cy = r["sections"][0]["faces"]["inf_x"]["couches"][0], r["sections"][0]["faces"]["inf_y"]["couches"][0]
chk("annoter (dalle) : inf. X auto (dans l'état), position 4,7 (le site), axe ⌈5,7⌉ = 6,0, celui du moteur seul ⌈4,5⌉ = 4,5, "
    "empilée ; inf. Y auto, 3,6 → 5,0, pas empilée",
    (cx["auto"], cx["e_auto"], cx["e_auto_moteur"], cx["empilee"], cx["e"]) == (True, 6.0, 4.5, True, 6.0) and eq(cx["reel_auto"], 4.7)
    and eq(cx["e_reel"], 4.7) and (cy["auto"], cy["e_auto"], cy["e_auto_moteur"], cy["empilee"]) == (True, 5.0, 5.0, False), cx)
chk("annoter : le détail de l'axe d'inf. X — c + Ø_Y + Ø/2 + jeu = 3 + 1,2 + 0,5 + 1 = 5,7 → 6,0 (« façon A » : tout depuis le béton)",
    [(t["symbole"], round(t["valeur"], 9)) for t in cx["detail"]["termes"]] == [("c", 3.0), ("Ø_Y", 1.2), ("Ø/2", 0.5)]
    and cx["detail"]["jeu"] == 1.0 and eq(cx["detail"]["brut"], 5.7) and cx["detail"]["axe"] == 6.0, cx["detail"])
r3 = POS.annoter(MO_D.verifier(pm3), dalle_s3())
c3 = r3["sections"][0]["faces"]["inf_x"]["couches"][1]
chk("annoter (S3) : le renfort inf. X auto, position 4,8 (« auto · 4,8 » à l'écran), axe ⌈5,8⌉ = 6,0, celui du moteur seul ⌈4,6⌉ = "
    "5,0, empilé — et le moteur l'a utilisé à 6,0",
    (c3["auto"], c3["e_auto"], c3["e_auto_moteur"], c3["empilee"], c3["e"]) == (True, 6.0, 5.0, True, 6.0) and eq(c3["reel_auto"], 4.8), c3)
rp = POS.annoter(MO_P.verifier(pmp), dp)
csx = rp["sections"][0]["faces"]["sup_x"]["couches"][0]
fe = [fo for v in rp["sections"][0]["verifications"] if v["cle"] == "sup_x" for fo in v["formules"] if fo["cle"] == "e"]
chk("annoter (prédalle) : sup. X auto, empilée, axe 6,0 ; sa formule « e » (celle du site, qui remplace celle du moteur) porte "
    "empilee, « sur » (la base Y) et les termes de sa position : c 3, Ø_Y 1,2, Ø/2 0,5",
    (csx["auto"], csx["empilee"], csx["e_auto"], csx["pose"]) == (True, True, 6.0, "superieure")
    and len(fe) == 1 and fe[0]["empilee"] and fe[0]["auto"] and fe[0]["sur"] == "la base Y"
    and [(t["symbole"], round(t["valeur"], 9)) for t in fe[0]["termes"]] == [("c", 3.0), ("Ø_Y", 1.2), ("Ø/2", 0.5)]
    and fe[0]["resultat"] == 6.0, fe)
dp3 = dalle_s3("predalle")
rp3 = POS.annoter(MO_P.verifier(POS.pour_moteur(dp3)), dp3)
fe3 = [fo for v in rp3["sections"][0]["verifications"] if v["cle"] == "sup_x" for fo in v["formules"] if fo["cle"] == "e" and fo["couche"] == 1]
chk("annoter (prédalle, S3) : la formule « e » du renfort sup. X (Ø16) dit c + Ø_Y + Ø/2 = 3 + 1,2 + 0,8 = 5,0 → ⌈6,0⌉ = 6,0",
    len(fe3) == 1 and fe3[0]["empilee"]
    and [(t["symbole"], round(t["valeur"], 9)) for t in fe3[0]["termes"]] == [("c", 3.0), ("Ø_Y", 1.2), ("Ø/2", 0.8)]
    and fe3[0]["resultat"] == 6.0, fe3)
chk("chaque vérification de face commence par la formule « e » de chacune de ses couches, la dalle comme la prédalle",
    all([fo["couche"] for fo in v["formules"] if fo["cle"] == "e"] == list(range(len(res["sections"][0]["faces"][v["cle"]]["couches"])))
        and v["formules"][0]["cle"] == "e" for res in (r, rp3) for v in res["sections"][0]["verifications"] if v["cle"] in EN_D.FACES))
# le sélecteur : chaque candidat à SA position empilée
cands = POS.candidats_base(d, 0, "inf_x")
figes = MO_D.candidats_base(pm, 0, "inf_x")
par_nom = {c["nom"]: c for c in cands}
chk("candidats_base, base secondaire inf. X (posée sur la base Y, dessus 4,2) : un Ø8 à 4,2 + 0,4 = 4,6 → axe 6,0, un Ø12 à 4,8 → "
    "6,0, un Ø25 à 5,45 → 6,5 — le moteur sur la position figée (4,7) les mettrait tous à 6,0",
    (par_nom["Ø8/150"]["e"], par_nom["Ø12/150"]["e"], par_nom["Ø25/150"]["e"]) == (6.0, 6.0, 6.5)
    and {c["nom"] for c in cands} == {c["nom"] for c in figes} and all(abs(c["yG"] - 6.0) < 1e-9 for c in figes if c["nom"] == "Ø25/150"))
chk("… même tri que le moteur (section au mètre), la retenue et une optimale marquées, mêmes clés plus « e »",
    [c["nom"] for c in cands] == [c["nom"] for c in figes] and sum(c["retenue"] for c in cands) == 1 and sum(c["optimal"] for c in cands) == 1
    and set(cands[0]) == set(figes[0]) | {"e"})
cp = POS.candidats_base(d, 0, "inf_y")
chk("candidats_base, base principale inf. Y : un Ø25 (position 4,25, axe ⌈5,25⌉ = 5,5) pousse la secondaire à 5,5 + 0,5 = 6,0",
    next(c for c in cp if c["nom"] == "Ø25/150")["e"] == 5.5 and len(cp) == len(figes))
d3 = dalle_s3()
cp3 = {c["nom"]: c for c in POS.candidats_base(d3, 0, "inf_y")}
essai = copy.deepcopy(d3)
essai["sections"][0]["faces"]["inf_y"]["couches"][0].update(MO_D.lire_candidat("Ø25/150"))
r_essai = MO_D.verifier(POS.pour_moteur(essai))["sections"][0]
chk("candidats_base (S3) : la base principale candidate Ø25 (dessus 5,5) pousse AUSSI le renfort secondaire à 5,5 + 0,6 = 6,1, axe "
    "⌈7,1⌉ = 7,5 ; le statut de section du candidat est celui de la dalle recomposée",
    r_essai["faces"]["inf_x"]["couches"][1]["e"] == 7.5 and cp3["Ø25/150"]["statut_section"] == r_essai["statut"])
try:
    POS.candidats_base(d, 3, "inf_x")
    chk("candidats_base : section introuvable → ValueError", False)
except ValueError as e:
    chk("candidats_base : section introuvable → ValueError", "introuvable" in str(e))

# ------------------------------------------------------------------ (e) les dessins
print("\n=== (e) les dessins : chaque couche à sa position RÉELLE, lue dans le résultat ===")
GROUPE = r'<g class="pc couche" data-sel="s0\.{}">(.*?)</g>'


def niveaux(svg, cle_points, cle_traits):
    """Dans la PREMIÈRE coupe (⊥ principale) : les cy des cercles de la couche ``cle_points`` et les y1 des traits de ``cle_traits``."""
    coupe = svg.split("data-coupe=")[1]
    pts = re.search(GROUPE.format(re.escape(cle_points)), coupe).group(1)
    trs = re.search(GROUPE.format(re.escape(cle_traits)), coupe).group(1)
    return ({float(x) for x in re.findall(r'<circle cx="[\d.]+" cy="([\d.]+)"', pts)},
            {float(x) for x in re.findall(r'<line x1="[\d.]+" y1="([\d.]+)"', trs)})


def dessin(POSm, DDm, d, MO):
    """Le résultat du moteur sur la copie du site, annoté, et la coupe de la section 0 (pièces cliquables : i = 0)."""
    r = POSm.annoter(MO.verifier(POSm.pour_moteur(d)), d)
    cle = "predalle" if d["type"] == "predalle" else "dalle"
    return r, DDm.svg_coupe("t", r["sections"][0], r[cle], 0)


def dessin_brut(DDm, d, MO):
    """Le résultat du moteur SEUL (la dalle brute) et sa coupe : le témoin des niveaux qu'il pose sans le site."""
    r = MO.verifier(moteur_de(d))
    cle = "predalle" if d["type"] == "predalle" else "dalle"
    return r, DDm.svg_coupe("t", r["sections"][0], r[cle], 0)


for type_, DDm, MO in (("coulee", DD, MO_D), ("predalle", DP, MO_P)):
    r1, sv1 = dessin(POS, DDm, dalle_s3(type_), MO)
    r0, sv0 = dessin_brut(DDm, dalle_s3(type_), MO)
    h = r1[("predalle" if type_ == "predalle" else "dalle")]["h"] * 10.0
    cy1, y1 = niveaux(sv1, "inf_y-0", "inf_x-0")
    cy0, y0 = niveaux(sv0, "inf_y-0", "inf_x-0")
    sy1, sx1 = niveaux(sv1, "sup_y-0", "sup_x-0")
    sy0, sx0 = niveaux(sv0, "sup_y-0", "sup_x-0")
    _, rx1 = niveaux(sv1, "sup_y-0", "sup_x-1")
    _, rx0 = niveaux(sv0, "sup_y-0", "sup_x-1")
    _, ri1 = niveaux(sv1, "inf_y-0", "inf_x-1")
    chk(f"coupe ⊥ Y ({type_}) : les barres Y (points, position 36 mm) 11 mm plus bas que les barres X (trait, 47 mm : posées dessus, "
        "au contact) ; le moteur seul : X à 35 et Y à 36 (au contact de l'enrobage l'une et l'autre), en bas et en haut ; en haut, "
        "sup. X 11 mm sous sup. Y",
        len(cy1) == 1 and len(y1) == 1 and abs((min(cy1) - min(y1)) - 11.0) < 0.06 and abs((min(cy0) - min(y0)) - (-1.0)) < 0.06
        and abs((min(sx1) - min(sy1)) - 11.0) < 0.06 and abs((min(sx0) - min(sy0)) - (-1.0)) < 0.06, (cy1, y1, cy0, y0, sy1, sx1))
    if type_ == "coulee":
        chk("S3 sur la coupe : le renfort inf. X (trait, 48 mm) 1 mm au-dessus de la base X (47 : le même lit, Ø12 sur Ø10), soit 12 mm "
            "au-dessus des barres Y ; en haut, le renfort sup. X (Ø16) à 50, 14 mm sous sup. Y (moteur seul : 38, au contact de "
            "l'enrobage)",
            abs((min(y1) - min(ri1)) - 1.0) < 0.06 and abs((min(cy1) - min(ri1)) - 12.0) < 0.06
            and abs(min(rx1) - min(sy1) - 14.0) < 0.06 and abs(min(rx0) - min(sy0) - 2.0) < 0.06, (y1, ri1, rx1, sy1, rx0, sy0))
    else:
        chk("S3 sur la coupe de la prédalle : le renfort sup. X à 50 (posé sur la base sup. Y, 14 mm sous elle), le moteur seul : 38",
            abs(min(rx1) - min(sy1) - 14.0) < 0.06 and abs(min(rx0) - min(sy0) - 2.0) < 0.06, (rx1, sy1, rx0, sy0))
        chk("la peau hachurée et le plan de reprise restent (prédalle 60 mm, coulé 160 mm)", 'stroke="#8C97A6"' in sv1 and "prédalle préfabriquée · 60 mm" in sv1
            and "béton coulé en place · 160 mm" in sv1)
        chk("légende prédalle : les couches empilées (inf. X dans la peau, sup. X, son renfort) disent « sur la base Y », jamais « axe saisi » "
            "(le site les a posées, pas l'ingénieur) ; la base Y « dans la prédalle », les renforts inférieurs « sur la prédalle »",
            "axe à 47 · sur la base Y" in sv1 and "axe à 50 · sur la base Y" in sv1 and "axe saisi" not in sv1
            and "axe à 36 · dans la prédalle" in sv1 and "axe à 66 · sur la prédalle" in sv1, re.findall(r"axe à \d+[^<]*", sv1)[:8])
    # chaque couche est dessinée à sa position RÉELLE : les y des points et des traits sont h − e_reel (inf.) ou e_reel (sup.), en mm
    yc = None
    faute = []
    for cote in ("inf", "sup"):
        for dk in ("x", "y"):
            for j, c in enumerate(r1["sections"][0]["faces"][f"{cote}_{dk}"]["couches"]):
                coupe = sv1.split("data-coupe=")[1]                                    # ⊥ Y : Y en points, X en traits
                m = re.search(GROUPE.format(re.escape(f"{cote}_{dk}-{j}")), coupe)
                ys = [float(x) for x in re.findall(r'cy="([\d.]+)"' if dk == "y" else r'y1="([\d.]+)"', m.group(1))]
                attendu = (h - c["e_reel"] * 10.0) if cote == "inf" else c["e_reel"] * 10.0
                if yc is None:
                    yc = ys[0] - attendu                                              # le dessus du béton, lu une fois
                if any(abs(y - (yc + attendu)) > 0.06 for y in ys):
                    faute.append((cote, dk, j, c["e_reel"], ys[:2]))
    chk(f"{type_} : chaque couche (bases, renforts des deux directions) est dessinée à sa position RÉELLE (h − e_réel ou e_réel), au "
        "dixième de mm — jamais à l'axe de calcul (jeu et arrondi compris)", not faute, faute)
    # les cotes de d : celles du calcul, pour la face tendue de chaque direction (ici les inférieures : M_inf > M_sup)
    g1 = r1["sections"][0]["grandeurs"]
    coupes = sv1.split("data-coupe=")
    chk(f"{type_} : la cote d de la coupe ⊥ Y est h − y_G inf. Y du calcul ({DDm._mm(g1['d_inf_y'] * 10)}), celle de la coupe ⊥ X "
        f"h − y_G inf. X ({DDm._mm(g1['d_inf_x'] * 10)}) ; d inf. X plus petit que celui du moteur seul",
        f"d = {DDm._mm(g1['d_inf_y'] * 10)}" in coupes[1] and f"d = {DDm._mm(g1['d_inf_x'] * 10)}" in coupes[2]
        and g1["d_inf_x"] < r0["sections"][0]["grandeurs"]["d_inf_x"], re.findall(r"d = [\d ,]+", sv1))
    chk(f"{type_} : le jeu de pose est rappelé sous la légende, discrètement : « Barres dessinées à leur position réelle · le calcul "
        "ajoute un jeu de pose de 10 mm et l'arrondi au 5 mm »",
        "Barres dessinées à leur position réelle · le calcul ajoute un jeu de pose de 10 mm et l'arrondi au 5 mm" in html.unescape(sv1))
    # (une prédalle a aussi, depuis le lot 5, la barre de couture du dessin du joint : s0.joint-0)
    chk(f"{type_} : les pièces cliquables gardent leurs clés (s0.<face>-<j>, s0.beton{', s0.joint-0' if type_ == 'predalle' else ''})",
        {k for k in re.findall(r'data-sel="([^"]+)"', sv1)} == {f"s0.{f}-{j}" for f in EN_D.FACES for j in range(len(r1["sections"][0]["faces"][f]["couches"]))}
        | {"s0.beton"} | ({"s0.joint-0"} if type_ == "predalle" else set()))
r1, sv1 = dessin(POS, DD, dalle_s3(), MO_D)
chk("légende dalle : les couches empilées disent « axe à 47 · sur la base Y » (la base X) et « axe à 48 · sur la base Y » (son "
    "renfort, S3) ; la principale « axe à 36 » sans plus",
    "X · Treillis 10/10/100/100 · axe à 47 · sur la base Y" in sv1 and "X · 3 Ø12 · axe à 48 · sur la base Y" in sv1
    and "Y · Ø12/150 · axe à 36<" in sv1, re.findall(r"axe à \d+[^<]*", sv1)[:6])
d_sans = dalle_s3()
d_sans.update(jeu_premier_lit=0.0, arrondi=0.0)
chk("sans jeu ni arrondi, le dessin ne rappelle rien (la ligne n'a rien à dire)", "dessinées à leur position réelle" not in dessin(POS, DD, d_sans, MO_D)[1])
r0, sv0 = dessin_brut(DD, dalle(), MO_D)
chk("le moteur seul (non annoté) : aucune mention « sur la base », aucun « axe saisi » (rien que le calcul n'ait fait)",
    "sur la base" not in sv0 and "axe saisi" not in sv0)
# le dessin sans annotation ne ment pas sur les niveaux (les y sont ceux des positions) ; seule la légende de la prédalle lirait « saisi »
r_brut = MO_P.verifier(POS.pour_moteur(dalle("predalle")))
sv_brut = DP.svg_coupe("t", r_brut["sections"][0], r_brut["predalle"], 0)
sv_ann = DP.svg_coupe("t", POS.annoter(copy.deepcopy(r_brut), dalle("predalle"))["sections"][0], r_brut["predalle"], 0)
chk("prédalle SANS annoter : les niveaux sont justes (les mêmes qu'annoté), mais la légende dit « axe saisi » de la couche que le site a "
    "posée — c'est pourquoi api_web/dalle.py doit passer par annoter",
    niveaux(sv_brut, "sup_y-0", "sup_x-0") == niveaux(sv_ann, "sup_y-0", "sup_x-0") and "axe à 47 · axe saisi" in sv_brut
    and "axe à 47 · sur la base Y" in sv_ann)
# la cote de d d'une prédalle suit la face TENDUE (comme la dalle) : sup. quand son moment l'emporte
da = dalle("predalle")
da["sections"][0].update(My_inf=5.0, My_sup=30.0)                      # (M_x,inf = 15 > M_x,sup = 5 : X reste tendue en bas)
ra, sva = dessin(POS, DP, da, MO_P)
ga = ra["sections"][0]["grandeurs"]
chk("prédalle, M_y,sup > M_y,inf : la coupe ⊥ Y cote d = h − y_G SUP. Y (comme la coupe des dalles), la coupe ⊥ X d inf. X",
    f"d = {DP._mm(ga['d_sup_y'] * 10)}" in sva.split("data-coupe=")[1] and f"d = {DP._mm(ga['d_inf_y'] * 10)}" not in sva.split("data-coupe=")[1]
    and f"d = {DP._mm(ga['d_inf_x'] * 10)}" in sva.split("data-coupe=")[2], re.findall(r"d = [\d ,]+", sva))
# svg_toutes et svg_plan tournent sur un résultat annoté
chk("svg_plan et svg_toutes (dalle et prédalle) acceptent le résultat annoté et gardent les pièces",
    all("data-sel=" in m.svg_plan("t", r_["sections"][0], r_[k], 0) and "s0." in m.svg_toutes("t", r_["sections"], r_[k])
        for m, r_, k in ((DD, r1, "dalle"), (DP, ra, "predalle"))))

# ------------------------------------------------------------------ (f) mutations, en mémoire
print("\n=== (f) mutations (en mémoire : le dépôt n'est pas touché) ===")


def muter(fichier, ancien, nouveau, nom):
    """Le module ``fichier`` rechargé depuis sa source où ``ancien`` (présent une fois) devient ``nouveau``."""
    chemin = os.path.join(RACINE, "api_web", fichier)
    with open(chemin, encoding="utf-8") as fh:
        src = fh.read()
    assert src.count(ancien) == 1, (fichier, ancien, src.count(ancien))
    m = types.ModuleType(nom)
    m.__file__ = chemin
    m.__package__ = "api_web"          # ses imports relatifs (dessin_dalle lit dalle_nappes, DAL5) : les modules du paquet
    exec(compile(src.replace(ancien, nouveau), chemin, "exec"), m.__dict__)
    return m


REGLE = '    return P.sur_couche(appui(d, s, cote)["dessus"], MO_D.couche_diametre(c), P.JEU_COUCHES)\n'
M1 = muter("dalle_positions.py", REGLE, '    return appui(d, s, cote)["dessus"]\n', "positions_m1")
chk("mutation 1 — la règle sans le Ø/2 de la couche posée : inf. X vaudrait 4,2 et non 4,7 → le contrôle (b) rougit",
    eq(autos_site(M1, dalle())["inf_x"][0], 4.2))
M2 = muter("dalle_positions.py", '                    c_m["dist"] = position_auto(d, s_d, f, j)\n', '                    c_m["dist"] = None\n',
           "positions_m2")
pm2 = M2.pour_moteur(dalle())
chk("mutation 2 — pour_moteur n'écrit plus dist : le moteur garde ses positions, d utile inf. X 15,5 et non 14,0 → le contrôle (d) rougit",
    pm2["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] is None and MO_D.verifier(pm2)["sections"][0]["grandeurs"]["d_inf_x"] == 15.5)
M3 = muter("dessin_dalle.py", '                y = (h_mm - c["e_reel"] * 10.0) if cote == "inf" else c["e_reel"] * 10.0\n',
           '                y = (h_mm - c["e"] * 10.0) if cote == "inf" else c["e"] * 10.0\n', "dessin_m3")
rm, svm = dessin(POS, M3, dalle(), MO_D)
cym, ym = niveaux(svm, "inf_y-0", "inf_x-0")
chk("mutation 3 — le dessin lit l'axe de CALCUL (jeu et arrondi compris) au lieu de la position réelle : X et Y à 60 et 50 (écart 10 mm, "
    "pas 11) → le contrôle (e) rougit", abs((min(cym) - min(ym)) - 10.0) < 0.06, (cym, ym))
GARDE = '    if _peau(d, cote) and j > 0:\n'
M4 = muter("dalle_positions.py", GARDE, '    if j > 0:\n', "positions_m4")
chk("mutation 4 — S3 retiré (un renfort secondaire reste à la position du moteur) : le renfort inf. X retombe à 3,6 (au niveau de la "
    "base Y) et le sup. X de la prédalle à 3,8 → les contrôles S3 de (b), (c) et (d) rougissent",
    eq(autos_site(M4, dalle_s3())["inf_x"][1], 3.6) and eq(autos_site(M4, dalle_s3("predalle"))["sup_x"][1], 3.8)
    and not M4.empilee(dalle_s3(), dalle_s3()["sections"][0], "inf_x", 1))
M5 = muter("dalle_positions.py", GARDE, '    if not d.get("empilement") or (_peau(d, cote) and j > 0):\n', "positions_m5")
chk("mutation 5 — l'option revenue (clé « empilement » absente lue fausse) : toute la direction secondaire revient à la position du "
    "moteur, la base inf. X à 3,5 → les contrôles (a) et (b) rougissent",
    autos_site(M5, dalle()) == autos_moteur(dalle()) and eq(autos_site(M5, dalle())["inf_x"][0], 3.5))

# ------------------------------------------------------------------ (g) S5 : la couche principale la plus haute porte
print("\n=== (g) S5 : la nappe secondaire repose sur la couche principale la plus haute de son lit (renfort compris) ===")


def renfort(diam, n=3, dist=None, type_="n_barres", treillis="10/10/100/100"):
    return {"type": type_, "treillis": treillis, "d": diam, "esp": 150, "n": n, "dist": dist}


def dalle_s5(type_="coulee", principale="Y", cote="inf", renforts=((12,),), base=None, sec_renfort=None, treillis=False,
             **modifs):
    """La note neuve du site (c = 3 cm, jeu de pose 1 cm, arrondi 0,5 cm ; prédalle h = 22, h_p = 6) aux bases de BARRES
    Ø10/100 partout (S5 vaut pour des barres ; un treillis unique pose ses renforts principaux sur lui : section (h)) — ou,
    ``treillis`` vrai, la note neuve telle quelle (treillis 10/10/100/100 partout, « Mixte ») ; des RENFORTS sur la face
    principale ``cote`` (``renforts`` : (Ø, n, dist) ; par défaut le 3 Ø12 de « + Renfort »), ``base`` : une autre base
    principale, ``sec_renfort`` : le Ø d'un renfort S3 sur la secondaire de la même face ; moments sur les quatre faces
    (M_x,inf = 20)."""
    d = EN_D.nouvelle_dalle() if type_ == "coulee" else EN_P.nouvelle_predalle()
    d["dir_principale"] = principale
    p = principale.lower()
    q = "x" if p == "y" else "y"
    s = d["sections"][0]
    if not treillis:
        for f_ in EN_D.FACES:
            s["faces"][f_]["couches"][0] = {"type": "barres", "treillis": "10/10/100/100", "d": 10, "esp": 100, "n": 3, "dist": None}
    if base:
        s["faces"][f"{cote}_{p}"]["couches"][0] = dict(base)
    for r_ in renforts:
        s["faces"][f"{cote}_{p}"]["couches"].append(renfort(*r_))
    if sec_renfort:
        s["faces"][f"{cote}_{q}"]["couches"].append(renfort(sec_renfort))
    s.update(Mx_inf=20.0, My_inf=30.0, Mx_sup=12.0, My_sup=15.0, V=40.0)
    if type_ == "predalle":
        d.update(h=22, h_pre=6)
    d.update(modifs)
    d = (EN_D if type_ == "coulee" else EN_P).normaliser(d)
    d.update(verrou=False, type=type_)
    return d


def ax(d, f, j=0):
    return POS.position_auto(d, d["sections"][0], f, j)


def ap(d, cote="inf"):
    return POS.appui(d, d["sections"][0], cote)


def nom(d, cote="inf"):
    return POS.nom_appui(d, d["sections"][0], cote)


def appui_eq(a, attendu):
    return set(a) == set(attendu) and all(eq(a[k], v) if isinstance(v, float) else a[k] == v for k, v in attendu.items())


# la règle
d12 = dalle_s5()
chk("des barres Ø10/100 et le renfort de « + Renfort » (base inf. Y Ø10 à 3,5, dessus 4,0 ; renfort 3 Ø12 à 3,6, dessus 4,2 : le même "
    "dessous 3,0) : le lit contient les deux, le renfort porte — la base inf. X à 4,2 + 0,5 = 4,7 (sur la base seule : 4,5), « le "
    "renfort Y » ; la face sup. (sans renfort) reste sur sa base, 4,5",
    POS.lit_contact(d12, d12["sections"][0], "inf") == [0, 1] and ap(d12)["couche"] == 1 and eq(ap(d12)["dessus"], 4.2)
    and eq(ax(d12, "inf_x"), 4.7) and nom(d12) == "le renfort Y" and eq(ax(d12, "sup_x"), 4.5) and ap(d12, "sup")["couche"] == 0
    and nom(d12, "sup") == "la base Y", (ap(d12), ax(d12, "inf_x"), ax(d12, "sup_x")))
d25 = dalle_s5(renforts=((25,),), sec_renfort=16)
chk("un renfort principal 3 Ø25 (3 + 1,25 = 4,25, dessus 5,5) : la base inf. X à 6,0 et son renfort S3 3 Ø16 à 5,5 + 0,8 = 6,3 — la "
    "même nappe secondaire, sur le même renfort (le moteur seul : 3,5 et 3,8)",
    eq(ax(d25, "inf_x"), 6.0) and eq(ax(d25, "inf_x", 1), 6.3)
    and appui_eq(ap(d25), {"couche": 1, "e": 4.25, "d": 25.0, "demi": 1.25, "dessus": 5.5, "lit": [0, 1]})
    and eq(autos_moteur(d25)["inf_x"], [3.5, 3.8]), (ap(d25), POS.positions(d25, d25["sections"][0])["inf_x"]))
dsup = dalle_s5(cote="sup", renforts=((20,),))
dpx = dalle_s5(principale="X", renforts=((25,),))
chk("la même règle en haut (renfort sup. Y Ø20 à 4,0, dessus 5,0 : sup. X 5,5 depuis le dessus) et principale X (renfort inf. X Ø25, "
    "dessus 5,5 : la base inf. Y, secondaire, à 5,5 + 0,5 = 6,0, « le renfort X »)",
    eq(ax(dsup, "sup_x"), 5.5) and nom(dsup, "sup") == "le renfort Y" and eq(ax(dsup, "inf_x"), 4.5)
    and eq(ax(dpx, "inf_y"), 6.0) and nom(dpx) == "le renfort X", (ax(dsup, "sup_x"), ax(dpx, "inf_y")))
dps = dalle_s5("predalle", cote="sup", renforts=((20,),))
dpi = dalle_s5("predalle", renforts=((20,),))
rpi = MO_P.verifier(POS.pour_moteur(dpi))
chk("prédalle : en haut, la règle des dalles (renfort sup. Y Ø20 : sup. X 5,5) ; en BAS, la base inf. X dans la peau, posée sur la "
    "base Y SEULE (4,0 + 0,5 = 4,5 : le renfort Y Ø20 est sur la prédalle, 6 + 1,0 = 7,0, posé au chantier) — plus de croisement sur "
    "la prédalle (un seul renfort y est posé)",
    eq(ax(dps, "sup_x"), 5.5) and eq(ax(dpi, "inf_x"), 4.5) and eq(autos_moteur(dpi)["inf_y"], [3.5, 7.0])
    and POS.empilee(dpi, dpi["sections"][0], "inf_x", 0) and POS.lit_contact(dpi, dpi["sections"][0], "inf") == [0, 1]
    and ap(dpi)["couche"] == 0 and "croisement_sur_predalle" not in [a["code"] for a in rpi["avertissements"]],
    (ax(dps, "sup_x"), ax(dpi, "inf_x"), [a["code"] for a in rpi["avertissements"]]))
# les cas limites du lit de contact (base Ø10 : dessus 4,0)
T6 = {"type": "treillis", "treillis": "6/6/150/150", "d": 12, "esp": 150, "n": 3, "dist": None}
B10 = {"type": "barres", "treillis": "10/10/100/100", "d": 10, "esp": 150, "n": 3, "dist": None}
B16 = {"type": "barres", "treillis": "10/10/100/100", "d": 16, "esp": 150, "n": 3, "dist": None}
limites = {
    "2e lit : renfort Ø25 SAISI à 9 cm (dessous 7,75 > dessus du lit 4,0) — il ne porte rien": (dalle_s5(renforts=((25, 3, 9.0),)), 4.5, [0], 0, "la base Y"),
    "contact exact : renfort Ø12 saisi à 4,6 (dessous 4,0 = dessus de la base) — du lit, il porte (5,2 + 0,5)":
        (dalle_s5(renforts=((12, 3, 4.6),)), 5.7, [0, 1], 1, "le renfort Y"),
    "0,01 cm au-dessus : renfort Ø12 saisi à 4,61 (dessous 4,01) — un autre lit (aucune marge)": (dalle_s5(renforts=((12, 3, 4.61),)), 4.5, [0], 0, "la base Y"),
    "chaîne : R1 Ø25 automatique (dessus 5,5), R2 Ø12 saisi à 6,1 (dessous 5,5 = dessus de R1) — R2 entre par R1 et porte (6,7 + 0,5)":
        (dalle_s5(renforts=((25,), (12, 3, 6.1))), 7.2, [0, 1, 2], 2, "le renfort R2 Y"),
    "le même R2 sans R1 (dessous 5,5 > 4,0) — hors du lit": (dalle_s5(renforts=((12, 3, 6.1),)), 4.5, [0], 0, "la base Y"),
    "base saisie à 6,0 (dessus 6,5), renfort Ø12 automatique (dessus 4,2) — la base porte": (
        dalle_s5(base=dict(B10, dist=6.0), renforts=((12,),)), 7.0, [0, 1], 0, "la base Y"),
    "plusieurs renforts : 3 Ø12 et 2 Ø25 — le plus haut (R2, dessus 5,5) porte": (dalle_s5(renforts=((12,), (25, 2))), 6.0, [0, 1, 2], 2, "le renfort R2 Y"),
    "plusieurs renforts à égalité : 3 Ø12 et 2 Ø12 (dessus 4,2 l'un et l'autre) — le premier": (dalle_s5(renforts=((12,), (12, 2))), 4.7, [0, 1, 2], 1, "le renfort R1 Y"),
    "dessus égaux : base Ø16/150, renfort Ø16 (dessus 4,6 l'un et l'autre) — la base porte": (
        dalle_s5(base=B16, renforts=((16,),)), 5.1, [0, 1], 0, "la base Y"),
    "un renfort TREILLIS compte par son fil : treillis 12/12/150/150 (Ø12, dessus 4,2) sur une base treillis 6/6/150/150 (Ø6, dessus 3,6)": (
        dalle_s5(base=T6, renforts=((12, 3, None, "treillis", "12/12/150/150"),)), 4.7, [0, 1], 1, "le renfort Y"),
    "le jeu de pose (2 cm) ne change aucune position : renfort Ø25 (dessus 5,5) — 6,0": (
        dalle_s5(renforts=((25,),), jeu_premier_lit=2.0), 6.0, [0, 1], 1, "le renfort Y"),
}
fautes = []
for lib, (dd, attendu, lit, k, mot) in limites.items():
    s_ = dd["sections"][0]
    vu = (ax(dd, "inf_x"), POS.lit_contact(dd, s_, "inf"), ap(dd)["couche"], nom(dd))
    if not (eq(vu[0], attendu) and vu[1:] == (lit, k, mot)):
        fautes.append((lib, vu))
chk(f"cas limites du lit de contact ({len(limites)}) : {' ; '.join(limites)}", not fautes, fautes)
dsl = dalle_s5(renforts=((25, 3, 9.0),))
chk("un renfort principal saisi n'est jamais déplacé (positions() le garde à 9,0) ; pour_moteur non plus",
    eq(POS.positions(dsl, dsl["sections"][0])["inf_y"], [3.5, 9.0]) and POS.pour_moteur(dsl)["sections"][0]["faces"]["inf_y"]["couches"][1]["dist"] == 9.0)
# relecture S5 : une couche principale AUTOMATIQUE est du lit quoi qu'il arrive (posée sur les cales du premier lit) — même sous une base
# SAISIE plus bas qu'elle ; sans cela, le renfort automatique sortait du lit et la nappe secondaire le traversait
dbb = dalle_s5(base=dict(B10, d=8, dist=3.4), renforts=((20,),), jeu_premier_lit=1.0)
dbb0 = dalle_s5(base=dict(B10, dist=2.4), renforts=((20,),))
gbb = MO_D.verifier(POS.pour_moteur(dbb))["sections"][0]["grandeurs"]
chk("relecture S5 — base inf. Y Ø8 SAISIE à 3,4 (dessus 3,8), renfort 3 Ø20 AUTOMATIQUE à 4,0 (sur les cales : dessous 3,0, dessus "
    "5,0) : le renfort est du lit et porte, la base inf. X à 5,0 + 0,5 = 5,5 (et non 4,3, au travers du renfort), d = 20 − ⌈6,5⌉ = 13,5, "
    "A_s,req 493,8 mm² ; base Ø10 saisie à 2,4 (dessus 2,9, SOUS le dessous du renfort) : 5,5 (et non 3,4) ; un renfort SAISI au-dessus "
    "du lit reste un 2e lit",
    (POS.lit_contact(dbb, dbb["sections"][0], "inf"), nom(dbb), gbb["d_inf_x"], round(gbb["As_req_inf_x"], 1))
    == ([0, 1], "le renfort Y", 13.5, 493.8) and eq(ax(dbb, "inf_x"), 5.5) and eq(ax(dbb0, "inf_x"), 5.5)
    and eq(ax(dalle_s5(base=dict(B10, dist=2.4), renforts=((20, 3, 6.0),)), "inf_x"), 3.4),
    (POS.lit_contact(dbb, dbb["sections"][0], "inf"), ax(dbb, "inf_x"), gbb["As_req_inf_x"], ax(dbb0, "inf_x")))
# le balayage : tous les Ø, toutes les faces où le site empile — jamais plus bas que sur la base seule, jamais à travers une couche du lit
bal, n_bal = [], 0
for type_ in ("coulee", "predalle"):
    for cote in ("inf", "sup"):
        if type_ == "predalle" and cote == "inf":
            continue
        bases = [dict(T6, treillis=t) for t in TR.liste_choix()] + [dict(B10, d=x) for x in EN_D.DIAMETRES]
        for b_ in bases:
            for x in EN_D.DIAMETRES:
                dd = dalle_s5(type_, cote=cote, base=b_, renforts=((x,),))
                s_ = dd["sections"][0]
                cs = s_["faces"][f"{cote}_y"]["couches"]
                e_b, e_r = (POS._reel_moteur(dd, c, j, f"{cote}_y") for j, c in enumerate(cs))
                h_b, h_r, h_s = POS._demi(cs[0]), POS._demi(cs[1]), POS._demi(s_["faces"][f"{cote}_x"]["couches"][0])
                a = ax(dd, f"{cote}_x")
                n_bal += 1
                if not (abs(a - (max(e_b + h_b, e_r + h_r) + h_s)) < 1e-9 and a >= e_b + h_b + h_s - 1e-9
                        and a - h_s >= max(e_b + h_b, e_r + h_r) - 1e-9):
                    bal.append((type_, cote, b_["type"], b_.get("treillis"), b_["d"], x, a))
chk(f"balayage ({n_bal} cas : dalle en bas et en haut, prédalle en haut ; chaque treillis et chaque Ø de base, chaque Ø de renfort) : "
    "la position secondaire = max(dessus de la base, dessus du renfort) + Ø_s/2, jamais sous la base seule, jamais à travers le renfort",
    not bal and n_bal == 3 * (len(TR.liste_choix()) + len(EN_D.DIAMETRES)) * len(EN_D.DIAMETRES), bal[:4])
# le sens de la sécurité, mesuré par le moteur : A_s,req de la secondaire ne baisse jamais (même dalle, renfort principal hors du lit)
baisse, n_mon = [], 0
for x in EN_D.DIAMETRES:
    for type_, cote in (("coulee", "inf"), ("coulee", "sup"), ("predalle", "sup")):
        dd, dh = dalle_s5(type_, cote=cote, renforts=((x,),)), dalle_s5(type_, cote=cote, renforts=((x, 3, 15.0),))
        M = POS.moteur(dd)
        g_, gh = M.verifier(POS.pour_moteur(dd))["sections"][0]["grandeurs"], M.verifier(POS.pour_moteur(dh))["sections"][0]["grandeurs"]
        monte = POS.axes(dd, dd["sections"][0])[f"{cote}_x"][0] > POS.axes(dh, dh["sections"][0])[f"{cote}_x"][0]
        n_mon += 1
        if g_[f"As_req_{cote}_x"] < gh[f"As_req_{cote}_x"] or monte != (g_[f"As_req_{cote}_x"] > gh[f"As_req_{cote}_x"]):
            baisse.append((type_, cote, x, g_[f"As_req_{cote}_x"], gh[f"As_req_{cote}_x"]))
chk(f"A_s,req de la nappe secondaire ne baisse jamais ({n_mon} cas : chaque Ø de renfort principal, dalle inf./sup., prédalle sup.) ; "
    "il monte exactement quand son axe de calcul monte (le renfort dépasse la base et l'arrondi le montre)", not baisse, baisse)
# ce que le moteur reçoit, ce qu'il calcule (h = 20, M_x,inf = 20 kN·m)
g12 = MO_D.verifier(POS.pour_moteur(d12))["sections"][0]["grandeurs"]
g25 = MO_D.verifier(POS.pour_moteur(dalle_s5(renforts=((25,),))))["sections"][0]["grandeurs"]
g00 = MO_D.verifier(POS.pour_moteur(dalle_s5(renforts=())))["sections"][0]["grandeurs"]
chk("pour_moteur et le moteur (h = 20, M_x,inf = 20) : la base inf. X à 4,5 (sur la base Y), 4,7 (3 Ø12), 6,0 (3 Ø25) ; axes 5,5, "
    "6,0, 7,0 ; d inf. X 14,5 → 14,0 → 13,0 ; A_s,req = 20·10⁶ / (333,3 × 0,9 × d) = 459,8 → 476,2 (+ 3,6 %) → 512,8 mm² (+ 11,5 %)",
    eq(POS.pour_moteur(d12)["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"], 4.7)
    and (g00["d_inf_x"], g12["d_inf_x"], g25["d_inf_x"]) == (14.5, 14.0, 13.0)
    and (round(g00["As_req_inf_x"], 1), round(g12["As_req_inf_x"], 1), round(g25["As_req_inf_x"], 1)) == (459.8, 476.2, 512.8)
    and round(100 * (g12["As_req_inf_x"] / g00["As_req_inf_x"] - 1), 1) == 3.6 and round(100 * (g25["As_req_inf_x"] / g00["As_req_inf_x"] - 1), 1) == 11.5,
    (g00["As_req_inf_x"], g12["As_req_inf_x"], g25["As_req_inf_x"]))
# annoter : appui et « sur » ; la formule « e » et la pose disent la couche qui porte
r25 = POS.annoter(MO_D.verifier(POS.pour_moteur(d25)), d25)
cs25 = r25["sections"][0]["faces"]["inf_x"]["couches"]
chk("annoter : la base et le renfort inf. X empilés portent appui = 1, « le renfort Y » (formule, pose) et « renfort Y » (légendes) ; la "
    "base principale, non empilée, rien",
    [(c["appui"], c["sur"], c["sur_court"]) for c in cs25] == [(1, "le renfort Y", "renfort Y")] * 2 and "sur" not in r25["sections"][0]["faces"]["inf_y"]["couches"][0]
    and "sur" not in POS.annoter(MO_D.verifier(POS.pour_moteur(dalle())), dalle())["sections"][0]["faces"]["inf_y"]["couches"][0])
from api_web import dalle as P_D  # noqa: E402  (les formules de la fenêtre de calcul : celles de l'écran des dalles)
from api_web import dalle_predalle as AP  # noqa: E402  (le type prédalle à l'affichage)
rps = POS.annoter(MO_P.verifier(POS.pour_moteur(dps)), dps)
fe5 = [fo for v in rps["sections"][0]["verifications"] if v["cle"] == "sup_x" for fo in v["formules"] if fo["cle"] == "e"]
gx5 = rps["sections"][0]["faces"]["sup_x"]["couches"][0]
chk("prédalle (renfort sup. Y Ø20) : la formule « e » de la base sup. X prend les termes du RENFORT (c 3, son Ø entier 2,0) et s'écrit "
    "« e₁ = c + Ø_Y + Ø/2 + jeu = 3 + 2 + 0,5 + 1 = 6,5 cm · posée sur le renfort Y » ; la pose dit « posée sur le renfort Y » ; une "
    "base porteuse dit « posée sur la base Y »",
    len(fe5) == 1 and [(t["symbole"], round(t["valeur"], 9)) for t in fe5[0]["termes"]] == [("c", 3.0), ("Ø_Y", 2.0), ("Ø/2", 0.5)]
    and fe5[0]["sur"] == "le renfort Y" and P_D._formule(fe5[0])["res"] == "6,5 cm · posée sur le renfort Y"
    and P_D._formule(fe5[0])["parts"] == ["e_1", "c + Ø_Y + Ø/2 + jeu", "3 + 2 + 0,5 + 1"]
    and "posée sur le renfort Y (e" in AP.pose_couche(dps["sections"][0]["faces"]["sup_x"]["couches"][0], gx5)
    and P_D._formule(fe[0])["res"] == "5,7 → 6,0 cm · posée sur la base Y", (fe5, P_D._formule(fe5[0]) if fe5 else None, P_D._formule(fe[0])))
# le sélecteur : un candidat de la base secondaire repose sur le renfort principal ; un candidat de la base principale ne fait pas
# descendre la secondaire sous le renfort, un plus gros la fait monter
c5 = {c["nom"]: c for c in POS.candidats_base(d12, 0, "inf_x")}
chk("candidats_base, base secondaire inf. X sur le renfort 3 Ø12 (dessus 4,2) : Ø8 à 4,6 → axe 6,0, Ø12 à 4,8 → 6,0, Ø25 à 5,45 → 6,5",
    (c5["Ø8/150"]["e"], c5["Ø12/150"]["e"], c5["Ø25/150"]["e"]) == (6.0, 6.0, 6.5), {k: c5[k]["e"] for k in ("Ø8/150", "Ø12/150", "Ø25/150")})
essais = {}
for cand in ("Ø8/150", "Ø25/150"):
    e_ = copy.deepcopy(dalle_s5(renforts=((16,),)))
    e_["sections"][0]["faces"]["inf_y"]["couches"][0].update(MO_D.lire_candidat(cand))
    essais[cand] = MO_D.verifier(POS.pour_moteur(e_))["sections"][0]
cp5 = {c["nom"]: c for c in POS.candidats_base(dalle_s5(renforts=((16,),)), 0, "inf_y")}
chk("candidats_base, base principale sous un renfort Ø16 (dessus 4,6) : Ø8 (dessus 3,8) laisse la secondaire sur le renfort (5,1 → "
    "axe 6,5), Ø25 (dessus 5,5) la porte (6,0 → 7,0) — le statut de chaque candidat est celui de la dalle recomposée",
    essais["Ø8/150"]["faces"]["inf_x"]["couches"][0]["e"] == 6.5 and essais["Ø25/150"]["faces"]["inf_x"]["couches"][0]["e"] == 7.0
    and all(cp5[k]["statut_section"] == essais[k]["statut"] for k in essais))
# la coupe : la nappe X ne traverse plus le renfort Y ; la légende dit sur quoi elle repose


def bords(svg, cle_points, cle_trait):
    """Dans la coupe ⊥ Y : (haut des cercles de ``cle_points`` : min(cy − r), bas du trait de ``cle_trait`` : y1 + épaisseur/2), mm."""
    coupe = svg.split("data-coupe=")[1]
    pts = re.search(GROUPE.format(re.escape(cle_points)), coupe).group(1)
    trs = re.search(GROUPE.format(re.escape(cle_trait)), coupe).group(1)
    haut = min(float(cy) - float(r_) for cy, r_ in re.findall(r'<circle cx="[\d.]+" cy="([\d.]+)" r="([\d.]+)"', pts))
    y1, ep = re.search(r'<line x1="[\d.]+" y1="([\d.]+)" x2="[\d.]+" y2="[\d.]+" stroke="[^"]+" stroke-width="([\d.]+)"', trs).groups()
    return haut, float(y1) + float(ep) / 2


d20 = dalle_s5(renforts=((20,),), sec_renfort=12)
r20, sv20 = dessin(POS, DD, d20, MO_D)
haut20, bas20 = bords(sv20, "inf_y-1", "inf_x-0")
chk("coupe ⊥ Y (renfort principal 3 Ø20 sous la nappe X) : le bas du trait de la base X (55 mm, Ø10) touche le haut des points du "
    "renfort Y (40 mm, Ø20) — au contact exact, sans le traverser",
    abs(bas20 - haut20) < 0.06 and eq(r20["sections"][0]["faces"]["inf_x"]["couches"][0]["e_reel"], 5.5), (haut20, bas20))
chk("… la légende commune dit sur quoi la nappe X repose, en forme courte (la ligne tient dans sa colonne) : « X · Ø10/100 · axe à 55 "
    "· sur renfort Y », « R1 X · 3 Ø12 · axe à 56 · sur renfort Y » ; la prédalle aussi (en haut) ; plusieurs renforts : « sur renfort "
    "Y » encore (le repère R2 est dans la forme longue : « le renfort R2 Y »)",
    "X · Ø10/100 · axe à 55 · sur renfort Y" in sv20 and "R1 X · 3 Ø12 · axe à 56 · sur renfort Y" in sv20
    and "X · Ø10/100 · axe à 55 · sur renfort Y" in dessin(POS, DP, dps, MO_P)[1]
    and "X · Ø10/100 · axe à 60 · sur renfort Y" in dessin(POS, DD, dalle_s5(renforts=((12,), (25, 2))), MO_D)[1]
    and POS.nom_appui(dalle_s5(renforts=((12,), (25, 2))), dalle_s5(renforts=((12,), (25, 2)))["sections"][0], "inf", court=True) == "renfort Y",
    re.findall(r"axe à \d+[^<]*", sv20)[:6])
# les mutations S5 (en mémoire)
LIT = "            if auto[p[0]] or p[1] - p[2] <= dessus + TOL_LIT:\n"
M6 = muter("dalle_positions.py", LIT, "            if False:\n", "positions_m6")
chk("mutation 6 — retour à la BASE SEULE (le lit réduit à la base) : la base inf. X retombe à 4,5 (3 Ø12 et Ø25), dans le renfort → "
    "les contrôles (g) rougissent", eq(M6.position_auto(d12, d12["sections"][0], "inf_x", 0), 4.5)
    and eq(M6.position_auto(d25, d25["sections"][0], "inf_x", 0), 4.5) and M6.nom_appui(d12, d12["sections"][0], "inf") == "la base Y")
M7 = muter("dalle_positions.py", "TOL_LIT = 1e-6 ", "TOL_LIT = 0.5 ", "positions_m7")
d461 = limites["0,01 cm au-dessus : renfort Ø12 saisi à 4,61 (dessous 4,01) — un autre lit (aucune marge)"][0]
d50 = dalle_s5(renforts=((12, 3, 5.0),))
chk("mutation 7 — une marge de 0,5 cm au lit : le renfort saisi à 4,61 (et à 5,0, dessous 4,4) entre dans le lit, la nappe X monte à "
    "5,71 (6,1) → le contrôle des cas limites rougit", eq(M7.position_auto(d461, d461["sections"][0], "inf_x", 0), 5.71)
    and eq(M7.position_auto(d50, d50["sections"][0], "inf_x", 0), 6.1) and eq(POS.position_auto(d50, d50["sections"][0], "inf_x", 0), 4.5))
M8 = muter("dalle_positions.py", "                dessus = max(dessus, p[1] + p[2])\n", "                pass\n", "positions_m8")
dch = limites["chaîne : R1 Ø25 automatique (dessus 5,5), R2 Ø12 saisi à 6,1 (dessous 5,5 = dessus de R1) — R2 entre par R1 et porte (6,7 + 0,5)"][0]
chk("mutation 8 — le lit sans chaîne (le dessus du lit reste celui de la base) : R2 sort du lit, la nappe X redescend à 6,0, au travers de "
    "R2 → le contrôle de la chaîne rougit", eq(M8.position_auto(dch, dch["sections"][0], "inf_x", 0), 6.0)
    and M8.lit_contact(dch, dch["sections"][0], "inf") == [0, 1])
M9 = muter("dalle_positions.py", "    k = next(j for j in lit if pts[j][1] + pts[j][2] >= haut - TOL_LIT)\n",
           "    k = next(j for j in reversed(lit) if pts[j][1] + pts[j][2] >= haut - TOL_LIT)\n", "positions_m9")
chk("mutation 9 — l'égalité donnée au dernier renfort au lieu de la base : dalle() (base et renfort Ø12) dirait « sur le renfort Y » → les "
    "légendes et les égalités des cas limites rougissent", M9.nom_appui(dalle(), dalle()["sections"][0], "inf") == "le renfort Y"
    and M9.nom_appui(dalle(), dalle()["sections"][0], "inf") != POS.nom_appui(dalle(), dalle()["sections"][0], "inf"))
M10 = muter("dessin_dalle.py", """        return f"sur {c.get('sur_court') or 'la base ' + principale.upper()}"\n""",
            """        return f"sur la base {principale.upper()}"\n""", "dessin_m10")
chk("mutation 10 — la légende d'avant (« sur la base Y » pour toute couche empilée) : la coupe dirait « sur la base Y » d'un renfort X que "
    "le renfort Y porte → le contrôle de la légende rougit", "R1 X · 3 Ø12 · axe à 56 · sur la base Y" in dessin(POS, M10, d20, MO_D)[1]
    and "sur renfort Y" not in dessin(POS, M10, d20, MO_D)[1])
M11 = muter("dalle_positions.py", LIT, "            if p[1] - p[2] <= dessus + TOL_LIT:\n", "positions_m11")
chk("mutation 11 — une couche automatique jugée sur son seul dessous (le lit d'avant la relecture) : sous la base Ø10 saisie à 2,4 "
    "(dessus 2,9), le renfort Ø20 automatique (dessous 3,0) sort du lit, la base inf. X retombe à 3,4, au travers du renfort → le "
    "contrôle « relecture S5 » rougit",
    M11.lit_contact(dbb0, dbb0["sections"][0], "inf") == [0] and eq(M11.position_auto(dbb0, dbb0["sections"][0], "inf_x", 0), 3.4))

# ------------------------------------------------------------------ (h) DAL5b : le treillis posé tel qu'il est
print("\n=== (h) DAL5b : un treillis unique a ses fils soudés l'un sur l'autre ; ses renforts principaux posés sur lui ===")
t12 = dalle_s5(treillis=True)
s12 = t12["sections"][0]
at12 = POS.appui_treillis(t12, s12, "inf")
chk("la note neuve « Mixte » (treillis 10/10/100/100, 3 Ø12 en inf. Y) : un treillis unique ; les fils X SOUDÉS sur les fils Y (3,5 + 0,5 "
    "+ 0,5 = 4,5 — le renfort ne les soulève pas) ; le renfort Y posé SUR le treillis (5,0 + 0,6 = 5,6) — « le treillis »",
    POS.treillis(t12, s12, "inf") and eq(ax(t12, "inf_y", 1), 5.6) and eq(ax(t12, "inf_x"), 4.5)
    and appui_eq(POS.appui(t12, s12, "inf"), {"couche": 0, "e": 3.5, "d": 10.0, "demi": 0.5, "dessus": 4.0, "lit": [0]})
    and appui_eq(at12, {"couche": 0, "e": 4.5, "d": 10.0, "demi": 0.5, "dessus": 5.0, "lit": [0], "face": "inf_x"})
    and POS.nom_support(t12, s12, "inf_y", 1) == "le treillis" == POS.nom_support(t12, s12, "inf_y", 1, court=True)
    and POS.empilee(t12, s12, "inf_y", 1) and POS.empilee(t12, s12, "inf_x", 0) and not POS.empilee(t12, s12, "inf_y", 0),
    (POS.positions(t12, s12)["inf_y"], POS.positions(t12, s12)["inf_x"], at12))
t2 = dalle_s5(treillis=True, sec_renfort=12)
s2 = t2["sections"][0]
chk("avec un renfort X 3 Ø12 aussi : il est dans le plan des fils X, posé sur les fils Y (4,0 + 0,6 = 4,6, dessus 5,2) ; le renfort Y "
    "monte sur lui (5,2 + 0,6 = 5,8) — « le renfort X », « renfort X » (légende)",
    eq(ax(t2, "inf_x", 1), 4.6) and eq(ax(t2, "inf_y", 1), 5.8) and POS.appui_treillis(t2, s2, "inf")["couche"] == 1
    and POS.nom_support(t2, s2, "inf_y", 1) == "le renfort X" and POS.nom_support(t2, s2, "inf_y", 1, court=True) == "renfort X",
    POS.positions(t2, s2))
t2b = dalle_s5(treillis=True, renforts=((12,), (16, 2)), sec_renfort=12)
t2c = dalle_s5(treillis=True)
t2c["sections"][0]["faces"]["inf_x"]["couches"].append(renfort(12, dist=12.0))
chk("deux renforts Y : tous deux sur la couche X la plus haute (le renfort X, dessus 5,2 : 5,8 et 6,0) ; un renfort X SAISI à 12 cm "
    "(dessous 11,4, hors du lit des fils X) ne porte rien : le renfort Y reste sur le treillis, 5,6",
    eq([ax(t2b, "inf_y", k) for k in (1, 2)], [5.8, 6.0]) and eq(ax(t2c, "inf_y", 1), 5.6)
    and POS.appui_treillis(t2c, t2c["sections"][0], "inf")["lit"] == [0], (POS.positions(t2b, t2b["sections"][0])["inf_y"], POS.positions(t2c, t2c["sections"][0])))
ts = dalle_s5(treillis=True, cote="sup")
tx = dalle_s5(treillis=True, principale="X", renforts=((25,),))
chk("la même règle en haut (chapeaux 3 Ø12 en sup. Y : 5,6 depuis le dessus, sous les fils X à 4,5) et principale X (renfort inf. X "
    "Ø25 : 5,0 + 1,25 = 6,25 sur les fils Y à 4,5)",
    eq(ax(ts, "sup_y", 1), 5.6) and eq(ax(ts, "sup_x"), 4.5) and eq(ax(tx, "inf_x", 1), 6.25) and eq(ax(tx, "inf_y"), 4.5),
    (POS.positions(ts, ts["sections"][0])["sup_y"], POS.positions(tx, tx["sections"][0])["inf_x"]))
tp = dalle_s5("predalle", treillis=True)
sp = tp["sections"][0]
rtp = POS.annoter(MO_P.verifier(POS.pour_moteur(tp)), tp)
cx = rtp["sections"][0]["faces"]["inf_x"]["couches"][0]
chk("prédalle : le treillis de la nappe inférieure est DANS la peau, ses fils X soudés sur les fils Y (4,5) — leur pose dit « predalle » ; "
    "le renfort Y reste SUR la prédalle (6 + 0,6 = 6,6, la position du moteur) ; aucun avertissement de peau (dessus des fils 5,0 ≤ 6)",
    eq(ax(tp, "inf_x"), 4.5) and eq(ax(tp, "inf_y", 1), 6.6) and eq(autos_moteur(tp)["inf_y"][1], 6.6) and cx["pose"] == "predalle"
    and cx["empilee"] and cx["sur"] == "la base Y" and not POS.empilee(tp, sp, "inf_y", 1)
    and not {"croisement_sur_predalle", "secondaire_hors_predalle"} & {a_["code"] for a_ in rtp["avertissements"]},
    (POS.positions(tp, sp), [a_["code"] for a_ in rtp["avertissements"]]))
tm = dalle_s5("predalle", treillis=True, h_pre=4.5)
rtm = POS.annoter(MO_P.verifier(POS.pour_moteur(tm)), tm)
av_ = [a_ for a_ in rtm["avertissements"] if a_["code"] == "secondaire_hors_predalle"]
chk("prédalle de 4,5 cm : les fils X soudés (position 4,5, Ø10) ont leur dessus à 5,0 cm — hors de la peau : l'avertissement le dit (grave "
    "à l'écran, comme la base hors de la peau)",
    len(av_) == 1 and eq(av_[0]["haut"], 5.0) and "sort de la peau de 4,5 cm" in av_[0]["message"]
    and 'class="msg grave"' in AP.avertissements(rtm, 0), [a_["message"] for a_ in av_])
db = dalle_s5("predalle", h_pre=4.5, base=dict(B10, d=12))
rdb = POS.annoter(MO_P.verifier(POS.pour_moteur(db)), db)
chk("… des barres aussi : base Y Ø12 (dessus 4,2) et base X Ø10 posée dessus (4,7, dessus 5,2) dans une peau de 4,5 cm — l'avertissement",
    [a_["code"] for a_ in rdb["avertissements"] if a_["code"] == "secondaire_hors_predalle"] == ["secondaire_hors_predalle"])
dp_sup = dalle_s5("predalle", treillis=True, cote="sup", renforts=((20,),))
rps2 = POS.annoter(MO_P.verifier(POS.pour_moteur(dp_sup)), dp_sup)
fo_e = [fo for v in rps2["sections"][0]["verifications"] if v["cle"] == "sup_y" for fo in v["formules"]
        if fo.get("cle") == "e" and fo.get("couche") == 1]
chk("prédalle, chapeaux 3 Ø20 en sup. Y sur le treillis : 5,0 + 1,0 = 6,0 ; sa formule « e » dit c + Ø_Y + Ø_X + Ø/2 = 3 + 1 + 1 + 1 "
    "et « posée sur le treillis » (appui_sec) ; la pose aussi",
    eq(ax(dp_sup, "sup_y", 1), 6.0) and len(fo_e) == 1 and fo_e[0].get("appui_sec") is True
    and [(t["symbole"], round(t["valeur"], 9)) for t in fo_e[0]["termes"]] == [("c", 3.0), ("Ø_Y", 1.0), ("Ø_X", 1.0), ("Ø/2", 1.0)]
    and "· posée sur le treillis" in P_D._formule(fo_e[0])["res"]
    and "posée sur le treillis" in AP.pose_couche(dp_sup["sections"][0]["faces"]["sup_y"]["couches"][1], rps2["sections"][0]["faces"]["sup_y"]["couches"][1]),
    fo_e)
# le dessin : le renfort Y au-dessus des fils X, au contact ; les fils Y au contact des fils X
rt12, sv12 = dessin(POS, DD, t12, MO_D)
coupe12 = sv12.split("data-coupe=")[1]
cy_r = [float(x) for x in re.findall(r'cy="([\d.]+)"', re.search(GROUPE.format("inf_y-1"), coupe12).group(1))]
cy_b = [float(x) for x in re.findall(r'cy="([\d.]+)"', re.search(GROUPE.format("inf_y-0"), coupe12).group(1))]
y_x = float(re.search(r'y1="([\d.]+)"', re.search(GROUPE.format("inf_x-0"), coupe12).group(1)).group(1))
chk("coupe ⊥ Y : les fils Y (points, Ø10) touchent le trait des fils X (Ø10) — 10 mm d'axe à axe, en contact ; le renfort Y (Ø12, 56 mm) "
    "posé sur le trait des fils X, au contact (le bas de ses points au haut du trait) ; la légende dit « R1 Y · 3 Ø12 · axe à 56 · sur le "
    "treillis » et « axes : Y à 35 · X à 45 »",
    abs((min(cy_b) - y_x) - 10.0) < 0.06 and abs((y_x - 5.0) - (max(cy_r) + 6.0)) < 0.06
    and "R1 Y · 3 Ø12 · axe à 56 · sur le treillis" in sv12 and "axes : Y à 35 · X à 45" in sv12,
    (cy_b[:1], y_x, cy_r[:1], re.findall(r"axes? [^<]+", sv12)))
# le sélecteur de la base : un treillis candidat sur un treillis unique pose sa transposée sur l'autre direction, comme le choix
cands_t = [c for c in POS.candidats_base(t12, 0, "inf_y", ("treillis",))]
ecarts_t = []
for c in cands_t:
    e = copy.deepcopy(t12)
    e["sections"][0]["faces"]["inf_y"]["couches"][0].update(type="treillis", treillis=c["treillis"])
    e["sections"][0]["faces"]["inf_x"]["couches"][0].update(type="treillis", treillis=POS.NAP.transposee(c["treillis"]))
    g_ = MO_D.verifier(POS.pour_moteur(e))["sections"][0]["grandeurs"]
    if abs(c["yG"] - g_["yG_inf_y"]) > 1e-9 or c["statut_section"] != MO_D.verifier(POS.pour_moteur(e))["sections"][0]["statut"]:
        ecarts_t.append((c["nom"], c["yG"], g_["yG_inf_y"]))
chk(f"candidats_base sur un treillis unique ({len(cands_t)} treillis) : chaque candidat est calculé comme le choix le poserait (sa "
    "transposée sur l'autre direction : la nappe reste un treillis, le renfort Y sur ses fils X) — y_G et statut identiques",
    len(cands_t) == len(TR.liste_choix()) and not ecarts_t, ecarts_t[:3])
# les mutations DAL5b (en mémoire)
M12 = muter("dalle_positions.py", "        if j > 0 and treillis(d, s, cote) and not _peau(d, cote):\n", "        if False:\n", "positions_m12")
chk("mutation 12 — le renfort principal laissé à la position du moteur (dans le plan des fils) : 3,6 au lieu de 5,6 → le contrôle de la "
    "note neuve rougit", eq(M12.position_auto(t12, s12, "inf_y", 1), 3.6))
M13 = muter("dalle_positions.py", "    lit = [0] if treillis(d, s, cote) or _peau(d, cote) else lit_contact(d, s, cote)\n",
            "    lit = [0] if _peau(d, cote) else lit_contact(d, s, cote)\n", "positions_m13")
chk("mutation 13 — S5 appliqué au treillis (les fils X soulevés par le renfort Y, à sa position du moteur, dessus 4,2) : 4,7 au lieu de "
    "4,5 → le contrôle de la note neuve rougit", eq(M13.position_auto(t12, s12, "inf_x", 0), 4.7))
M14 = muter("dalle_positions.py", "    if _peau(d, cote) and j > 0:\n", "    if _peau(d, cote):\n", "positions_m14")
chk("mutation 14 — la base secondaire de la peau laissée à la position du moteur (au niveau de la principale, 3,5) au lieu d'être posée "
    "sur elle (4,5) → le contrôle de la prédalle rougit", eq(M14.position_auto(tp, sp, "inf_x", 0), 3.5))
M15 = muter("dalle_positions.py", '            autre.update({"type": "treillis", "treillis": NAP.transposee(cand["treillis"])})\n', "            pass\n",
            "positions_m15")
ec15 = [c["nom"] for c, c15 in zip(cands_t, M15.candidats_base(t12, 0, "inf_y", ("treillis",))) if abs(c["yG"] - c15["yG"]) > 1e-9]
chk("mutation 15 — le sélecteur sans la transposée : un treillis candidat d'un autre Ø rend la nappe « incohérente » pendant l'essai "
    "(S5, renfort dans le plan des fils) — ses y_G diffèrent du choix → le contrôle du sélecteur rougit", len(ec15) >= 5, ec15)

# ------------------------------------------------------------------ (i) DAL5c : monter, descendre une couche
print("\n=== (i) DAL5c : les flèches — un pas de 0,5 cm sur la grille, jamais au travers d'une couche ni de l'enrobage ===")
from api_web import plaque_compact as PCx  # noqa: E402


def dep(d, f, j, sens, i=0):
    try:
        return POS.deplacer(d, i, f, j, sens)
    except ValueError as e:
        return str(e)


u = dalle_s5(treillis=True)                    # treillis 10/10/100/100, renfort 3 Ø12 en inf. Y (5,6 sur le treillis)
chk("le pas : POS.PAS_AXE = 0,5 cm, et l'écran le dit (plaque_compact.PAS_TXT « 0,5 cm » — deux copies nommées, liées ici)",
    POS.PAS_AXE == 0.5 and PCx.PAS_TXT == "0,5 cm")
chk("▲ monte, ▼ descend, sur la grille de 0,5 cm — en BAS la position grandit (renfort inf. Y : 5,6 → 6,0), en HAUT elle diminue (base "
    "sup. Y : 3,5 → 4,0 pour ▼, ses fils X soudés la suivent) ; rien n'est écrit dans la dalle (l'appelant écrit la position rendue)",
    dep(u, "inf_y", 1, 1) == 6.0 and dep(u, "sup_y", 0, -1) == 4.0 and u["sections"][0]["faces"]["inf_y"]["couches"][1]["dist"] is None)
m1 = dep(u, "inf_y", 1, -1)
chk("▼ refusé quand la couche traverserait une autre direction : le renfort Y (5,6) descendrait à 5,0, à 0,5 cm d'axe des fils X "
    "(4,5) — il faut ½ (12 + 10) = 1,1 cm ; le message le dit, en mots",
    m1 == "Impossible de descendre le renfort Y de 0,5 cm : la couche croiserait les fils X du treillis (axe à 4,5 cm de son parement).", m1)
m2 = dep(u, "inf_y", 0, -1)
m3 = dep(u, "sup_y", 0, 1)
chk("▼ des fils porteurs inférieurs (3,5) et ▲ des supérieurs (3,5 depuis le dessus) refusés : ils entreraient dans l'enrobage (3 cm)",
    "entrerait dans l'enrobage (3,0 cm)" in m2 and m2.startswith("Impossible de descendre les fils Y du treillis") and
    "entrerait dans l'enrobage" in m3 and m3.startswith("Impossible de monter les fils Y du treillis"), (m2, m3))
m4 = dep(u, "inf_x", 0, 1)
chk("les fils X d'un treillis unique ne se déplacent pas seuls : soudés sur les fils Y, le treillis se déplace par eux",
    m4.startswith("Les fils X du treillis sont soudés sur les fils Y"), m4)
u2 = copy.deepcopy(u)
u2["sections"][0]["faces"]["inf_y"]["couches"][0]["dist"] = dep(u, "inf_y", 0, 1)
chk("▲ des fils Y (3,5 → 4,0) : le treillis entier suit — les fils X soudés (5,0) et le renfort posé dessus (5,5 + 0,6 = 6,1) montent avec eux",
    eq(POS.positions(u2, u2["sections"][0])["inf_y"], [4.0, 6.1]) and eq(POS.positions(u2, u2["sections"][0])["inf_x"], [5.0]))
ub = dalle_s5(renforts=((12,),), sec_renfort=12)              # des barres Ø10/100 : S3 et S5
mb1, mb2, mb3 = dep(ub, "inf_y", 1, 1), dep(ub, "inf_x", 1, -1), dep(ub, "inf_x", 0, -1)
chk("des barres : le renfort Y (3,6) se monte à 4,0 (la grille ; il porte alors la base X, qui le suit, S5 : au contact exact, pas au "
    "travers) ; le renfort X (4,8) descendu à 4,5 croiserait la base Y (3,5 : 1,0 cm d'axe, il en faut 1,1) — refusé ; la base X (4,7) "
    "descendue à 4,0 aussi",
    mb1 == 4.0 and mb2 == "Impossible de descendre le renfort X de 0,5 cm : la couche croiserait la base Y (axe à 3,5 cm de son parement)."
    and mb3 == "Impossible de descendre la base X de 0,5 cm : la couche croiserait la base Y (axe à 3,5 cm de son parement).",
    (mb1, mb2, mb3))
ug = copy.deepcopy(u)
ug["sections"][0]["faces"]["inf_y"]["couches"][1]["dist"] = 6.3
chk("une position hors de la grille revient sur elle : 6,3 → ▲ 7,0, ▼ 6,0 (le pas arrondi au 0,5 cm)",
    dep(ug, "inf_y", 1, 1) == 7.0 and dep(ug, "inf_y", 1, -1) == 6.0, (dep(ug, "inf_y", 1, 1), dep(ug, "inf_y", 1, -1)))
u6 = copy.deepcopy(u)
u6["sections"][0]["faces"]["inf_y"]["couches"][1]["dist"] = 6.0     # le renfort Y monté d'un cran (▲ depuis 5,6)
uq = dalle_s5(cote="sup", base=dict(B10, d=12, dist=4.0), renforts=())  # la base sup. Y Ø12 saisie à 4,0 (automatique : 3,6)
chk("le CRAN de la position automatique (réelle, hors de la grille) : le renfort Y monté à 6,0, ▼ le ramène à 5,6 — son automatique, "
    "que l'appelant écrit vide (sans le cran : 5,5, au travers des fils X, refusé) ; en haut, la base sup. Y Ø12 saisie à 4,0, ▲ la "
    "ramène à 3,6 (et non 3,5) ; un pas qui ne l'atteint pas reste sur la grille (6,0 ▲ → 6,5)",
    eq(dep(u6, "inf_y", 1, -1), 5.6) and eq(dep(uq, "sup_y", 0, 1), 3.6) and dep(u6, "inf_y", 1, 1) == 6.5,
    (dep(u6, "inf_y", 1, -1), dep(uq, "sup_y", 0, 1), dep(u6, "inf_y", 1, 1)))
uc = copy.deepcopy(u)
uc["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] = 3.5     # les fils X saisis au travers des fils Y : un conflit déjà là
chk("… les fils X saisis au niveau des fils Y (un conflit déjà là) : le renfort Y se monte encore, de sa position automatique (4,6, sur les "
    "fils X saisis) au pas suivant de la grille (5,0)",
    (("croise", "inf_x", 0, "inf_y", 0) in POS.conflits(uc, uc["sections"][0]) or ("croise", "inf_y", 0, "inf_x", 0) in POS.conflits(uc, uc["sections"][0]))
    and eq(POS.position_auto(uc, uc["sections"][0], "inf_y", 1), 4.6) and dep(uc, "inf_y", 1, 1) == 5.0, dep(uc, "inf_y", 1, 1))
up = dalle_s5("predalle", treillis=True)
chk("prédalle : le renfort inf. Y posé sur la prédalle (6,6) monte (7,0) et descend (6,0 : 1,5 cm d'axe au-dessus des fils X de la "
    "peau, 4,5) ; les fils de la peau ne descendent pas dans l'enrobage",
    dep(up, "inf_y", 1, 1) == 7.0 and dep(up, "inf_y", 1, -1) == 6.0 and "enrobage" in dep(up, "inf_y", 0, -1),
    (dep(up, "inf_y", 1, 1), dep(up, "inf_y", 1, -1), dep(up, "inf_y", 0, -1)))
M16 = muter("dalle_positions.py", "    nouveaux = sorted(conflits(essai, s2) - conflits(d, s))\n", "    nouveaux = []\n", "positions_m16")
chk("mutation 16 — les flèches sans contrôle : le renfort Y descendrait à 5,0, au travers des fils X → le contrôle (i) rougit",
    M16.deplacer(u, 0, "inf_y", 1, -1) == 5.0)
M17 = muter("dalle_positions.py", "    delta = PAS_AXE if (sens > 0) == (cote == \"inf\") else -PAS_AXE\n", "    delta = PAS_AXE * sens\n", "positions_m17")


def dep_m(M, d, f, j, sens):
    try:
        return M.deplacer(d, 0, f, j, sens)
    except ValueError as e:
        return str(e)


chk("mutation 17 — ▼ qui diminuerait la position partout (en haut, la couche MONTERAIT) : la base sup. Y « descendue » irait à 3,0, dans "
    "l'enrobage du dessus (refusée) au lieu de 4,0 → le contrôle du sens rougit",
    dep_m(M17, u, "sup_y", 0, -1) != 4.0 and "enrobage" in str(dep_m(M17, u, "sup_y", 0, -1)))

M18 = muter("dalle_positions.py", "        e2 = auto\n", "        pass\n", "positions_m18")
chk("mutation 18 — sans le cran : le renfort Y monté à 6,0 ne redescend plus à son automatique (5,6) — ▼ vise 5,5, au travers des fils X, "
    "refusé → le contrôle du cran rougit", dep_m(M18, u6, "inf_y", 1, -1) != 5.6 and "croiserait" in str(dep_m(M18, u6, "inf_y", 1, -1)))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
