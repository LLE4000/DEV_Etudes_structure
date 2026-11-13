# -*- coding: utf-8 -*-
"""Les NIVEAUX réels des deux directions d'une dalle (api_web/dalle_positions.py) et leur dessin (api_web/dessin_dalle.py,
api_web/dessin_predalle.py), vérifiés en Python ordinaire — docs/site/DALLES_FUSION.md, § 4.

Les nappes sont TOUJOURS posées l'une sur l'autre (décision de l'ingénieur du 09/11/2026 : plus d'option « nappes
empilées ») et un RENFORT de la direction secondaire se pose sur sa nappe, comme sa base (S3, même décision).

(a) ce qui reste à l'automatique du moteur : la direction principale (base ET renforts) et les faces inférieures d'une
    prédalle (sa règle de pose) — exactement ``distance_auto`` ; la clé « empilement » d'une note d'avant ne change rien ;
(b) dalle coulée c = 3 cm, principale Y, bases Ø12 en Y et Ø10 en X : inf. Y 4,0 ; inf. X 4,0 + 1,0 + 0,5 = 5,5 ; sup.
    pareil ; le renfort de la principale à l'automatique du moteur ; S3 : un renfort secondaire 3 Ø12 à 4,0 + 1,0 + 1,0 =
    6,0 (Ø25 : 6,5), sur la principale saisie, avec le jeu ; un renfort saisi n'est pas déplacé ; la preuve que les moteurs
    posent un renfort dans le LIT de sa base (son décalage relatif est nul, il le reste empilé) ;
(c) prédalle : faces inférieures à la règle du moteur (renfort inf. X sur la prédalle compris : S3 n'y change rien),
    faces supérieures empilées, renfort sup. X compris (S3) ;
(d) pour_moteur : ``dist`` renseigné là où il faut seulement, la dalle intacte, un d utile plus petit pour la direction
    secondaire ; rien d'empilé (axes secondaires saisis) : verifier(pour_moteur(d)) == verifier(d) à l'octet ; annoter :
    auto / e_auto / empilee du SITE, la formule « e » d'un renfort empilé de prédalle ; candidats_base : chaque candidat à
    son axe empilé, et le renfort secondaire suit la base principale candidate ;
(e) les dessins : barres X et Y à des y différents, le renfort secondaire dans le lit de sa base ; chaque couche à
    l'axe que le moteur a reçu ; les cotes de d sont celles du calcul ; la légende de la prédalle ne dit pas « axe
    saisi » d'un axe que le site a posé ;
(f) mutations (en mémoire, le dépôt intact) : la règle sans ⌈Ø/2⌉, pour_moteur qui n'écrit rien, le dessin qui lit
    l'automatique du moteur, S3 retiré (renfort secondaire au niveau de la principale), l'option revenue (clé absente :
    nappes au même niveau) — chacune fait rougir le contrôle qui la garde ;
(g) S5 (relecture du 09/11/2026) : la nappe secondaire, base ET renforts, repose sur la couche principale la plus HAUTE
    de son lit de contact (un renfort principal plus épais que la base, posé dans son lit) — le mode par défaut
    « Treillis + renforts » (3 Ø12 sur un treillis Ø10 : 4,5 → 5,5, A_s,req + 6,9 %), un renfort Ø25 (6,5, + 14,8 %),
    les faces supérieures, la principale X, la prédalle en haut (et en bas : la règle du type, inchangée, et
    l'avertissement de croisement) ; les cas limites du lit (2e lit saisi, contact exact, 0,01 cm au-dessus, chaîne de
    renforts, base saisie plus haut, plusieurs renforts, demis égaux, treillis par son fil, jeu) ; le balayage de tous
    les Ø (jamais plus bas que sur la base seule, jamais à travers une couche du lit) ; pour_moteur, A_s,req, annoter
    (appui, « sur », formule « e » de la prédalle), candidats_base, la coupe (la nappe X ne traverse plus le renfort Y,
    la légende dit « sur renfort Y », forme courte qui tient dans sa colonne) ; une couche principale AUTOMATIQUE est du lit quoi qu'il
    arrive, même sous une base saisie plus bas que l'automatique (relecture S5) ; mutations : retour à la base seule, une marge de 0,5 cm,
    le lit sans chaîne, l'égalité donnée au renfort, la légende d'avant, la couche automatique jugée sur son seul dessous.

    /usr/bin/python3.12 tests/test_api_web_dalle_positions.py
"""
import copy
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


BARRES_12 = {"type": "barres", "treillis": "10/10/100/100", "d": 12, "esp": 150, "n": 3, "dist": None}


def dalle(type_="coulee", principale="Y"):
    """Dalle C ouverte c = 3 cm (jeu 0), principale Y : bases Ø12/150 en Y, treillis 10/10/100/100 (Ø10) en X, un renfort
    3 Ø12 en inf. Y ; moments sur les quatre faces (inf. plus chargées). En prédalle : h = 22, h_pre = 6."""
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
    """La dalle brute, sans les clés du site : ce que le moteur SEUL calculerait (l'écran Streamlit)."""
    return {k: v for k, v in d.items() if k not in POS.CLES_SITE}


def autos_moteur(d):
    """L'axe automatique DU MOTEUR de chaque couche, par face (la règle de l'écran Streamlit)."""
    s = d["sections"][0]
    if d["type"] == "predalle":
        return {f: [MO_P.distance_auto(c, j, f, d["dir_principale"], d["enrobage"], d["jeu_premier_lit"], d["h_pre"])
                    for j, c in enumerate(s["faces"][f]["couches"])] for f in EN_D.FACES}
    return {f: [MO_D.distance_auto(c, d["enrobage"], d["jeu_premier_lit"]) for c in s["faces"][f]["couches"]] for f in EN_D.FACES}


def autos_site(POSm, d):
    s = d["sections"][0]
    return {f: [POSm.axe_auto(d, s, f, j) for j in range(len(s["faces"][f]["couches"]))] for f in EN_D.FACES}


# ------------------------------------------------------------------ (a) ce qui reste à l'automatique du moteur
print("=== (a) direction principale et faces inférieures d'une prédalle : exactement l'automatique du moteur ===")
for type_ in ("coulee", "predalle"):
    for p in ("Y", "X"):
        d = dalle_s3(type_)
        d["dir_principale"] = p
        a_s, a_m = autos_site(POS, d), autos_moteur(d)
        garde = [f for f in EN_D.FACES if f.endswith(p.lower()) or (type_ == "predalle" and f.startswith("inf"))]
        chk(f"{type_}, principale {p} : axe_auto == distance_auto du moteur pour {', '.join(garde)} (base et renforts)",
            all(a_s[f] == a_m[f] for f in garde) and all(a_s[f] != a_m[f] for f in EN_D.FACES if f not in garde), (a_s, a_m))
chk("la clé « empilement » d'une note d'avant (vraie, fausse, absente) ne change rien : les mêmes axes",
    autos_site(POS, dict(dalle(), empilement=False)) == autos_site(POS, dict(dalle(), empilement=True)) == autos_site(POS, dalle())
    and not hasattr(POS, "empilement") and POS.CLES_SITE == {"verrou", "type"})
chk("sans la clé « type » : dalle coulée", POS.type_dalle({"nom": "x"}) == "coulee" and POS.moteur({"nom": "x"}) is MO_D)

# ------------------------------------------------------------------ (b) dalle coulée : la secondaire posée sur la principale
print("\n=== (b) dalle coulée : la direction secondaire (base ET renforts) posée sur la principale ===")
d = dalle()
s = d["sections"][0]
a = autos_site(POS, d)
chk("inf. Y (principale, Ø12) : 3 + ⌈0,6⌉₀,₅ = 4,0 cm — l'automatique du moteur", a["inf_y"][0] == 4.0 == MO_D.distance_auto(BARRES_12, 3.0, 0.0), a)
chk("inf. X (secondaire, Ø10) : 4,0 + 1,0 + 0,5 = 5,5 cm", a["inf_x"][0] == 5.5, a)
chk("sup. Y 4,0 et sup. X 5,5 cm (la même règle depuis le dessus)", (a["sup_y"][0], a["sup_x"][0]) == (4.0, 5.5), a)
chk("le renfort 3 Ø12 en inf. Y (principale) : l'automatique du moteur (3 + 1,0 = 4,0), pas empilé",
    a["inf_y"][1] == 4.0 == autos_moteur(d)["inf_y"][1]
    and not POS.empilee(d, s, "inf_y", 1) and POS.empilee(d, s, "inf_x", 0) and not POS.empilee(d, s, "inf_y", 0))
dx = dalle(principale="X")
ax = autos_site(POS, dx)
chk("principale X : la base X (Ø10) à 3,5, la base Y (Ø12) posée dessus à 3,5 + 0,5 + 1,0 = 5,0 cm ; le renfort inf. Y (devenu "
    "secondaire) aussi : 3,5 + 0,5 + 1,0 = 5,0 (S3)",
    (ax["inf_x"][0], ax["inf_y"][0], ax["sup_x"][0], ax["sup_y"][0], ax["inf_y"][1]) == (3.5, 5.0, 3.5, 5.0, 5.0), ax)
di = dalle()
di["sections"][0]["faces"]["inf_y"]["couches"][0]["dist"] = 5.0
chk("la base principale SAISIE à 5,0 cm : la secondaire repose dessus, 5,0 + 1,0 + 0,5 = 6,5 (et non sur l'automatique 4,0)",
    POS.axe_auto(di, di["sections"][0], "inf_x", 0) == 6.5 and POS.axes(di, di["sections"][0])["inf_y"] == [5.0, 4.0])
dj = dalle()
dj["jeu_premier_lit"] = 1.0
chk("jeu premier lit 1 cm : il entre une fois, par l'axe de la base principale (5,0 → secondaire 6,5)",
    autos_site(POS, dj)["inf_y"][0] == 5.0 and autos_site(POS, dj)["inf_x"][0] == 6.5)
chk("axes(d, s) : l'axe imposé sinon l'automatique du site, pour les quatre faces",
    POS.axes(d, s) == {"inf_x": [5.5], "sup_x": [5.5], "inf_y": [4.0, 4.0], "sup_y": [4.0]}, POS.axes(d, s))
# S3 : le renfort de la direction secondaire sur sa nappe
d3 = dalle_s3()
s3 = d3["sections"][0]
a3 = autos_site(POS, d3)
chk("S3 — renfort 3 Ø12 en inf. X (secondaire) : 4,0 + 1,0 + 1,0 = 6,0 cm, posé sur la base Y comme la base X (le moteur seul : 4,0, au niveau "
    "de la base Y, sous sa propre nappe) ; renfort 3 Ø16 en sup. X : 4,0 + 1,0 + 1,0 = 6,0 (depuis le dessus)",
    a3["inf_x"] == [5.5, 6.0] and a3["sup_x"] == [5.5, 6.0] and autos_moteur(d3)["inf_x"][1] == 4.0
    and POS.empilee(d3, s3, "inf_x", 1) and POS.empilee(d3, s3, "sup_x", 1), (a3, autos_moteur(d3)))
d25 = dalle_s3(d_inf=25)
chk("S3 — un renfort Ø25 en inf. X : 4,0 + 1,0 + 1,5 = 6,5 cm (le moteur seul : 3 + 1,5 = 4,5)",
    autos_site(POS, d25)["inf_x"][1] == 6.5 and autos_moteur(d25)["inf_x"][1] == 4.5)
d3i = dalle_s3()
d3i["sections"][0]["faces"]["inf_y"]["couches"][0]["dist"] = 5.0
d3j = dalle_s3()
d3j["jeu_premier_lit"] = 1.0
chk("S3 — sur la base principale SAISIE à 5,0 : le renfort secondaire 5,0 + 1,0 + 1,0 = 7,0 ; avec un jeu de 1 cm : 5,0 + 1,0 + 1,0 = 7,0 "
    "(le jeu entre une fois, par la base principale)",
    POS.axe_auto(d3i, d3i["sections"][0], "inf_x", 1) == 7.0 and autos_site(POS, d3j)["inf_x"][1] == 7.0)
d3s = dalle_s3()
d3s["sections"][0]["faces"]["inf_x"]["couches"][1]["dist"] = 8.0
chk("S3 — un renfort secondaire SAISI (8 cm) n'est pas déplacé : axes() le garde, il n'est pas « empilé », pour_moteur n'y touche pas",
    POS.axes(d3s, d3s["sections"][0])["inf_x"] == [5.5, 8.0] and not POS.empilee(d3s, d3s["sections"][0], "inf_x", 1)
    and POS.pour_moteur(d3s)["sections"][0]["faces"]["inf_x"]["couches"][1]["dist"] == 8.0)
# la preuve que la formule du renfort est celle de sa base : les deux moteurs posent un renfort dans le LIT de sa base
# (l'automatique moins le demi-diamètre est le même pour toutes les couches d'une face), aux faces où la règle s'applique
lits = []
for type_ in ("coulee", "predalle"):
    for f in EN_D.FACES:
        if type_ == "predalle" and f.startswith("inf"):
            continue
        dd_ = dalle(type_)
        for diam in EN_D.DIAMETRES:
            base, renf = dd_["sections"][0]["faces"][f]["couches"][0], dict(EN_D.nouvelle_couche(base=False), d=diam)
            lit_b = POS._auto_moteur(dd_, base, 0, f) - POS._demi(base)
            lit_r = POS._auto_moteur(dd_, renf, 1, f) - POS._demi(renf)
            if abs(lit_b - lit_r) > 1e-12:
                lits.append((type_, f, diam, lit_b, lit_r))
chk("les moteurs posent un renfort dans le LIT de la base de sa face (dalle : toutes les faces ; prédalle : les supérieures), pour chaque Ø : "
    "son décalage relatif à la base est nul — empilé, il le reste (axe du renfort − ⌈Ø/2⌉ = axe de la base secondaire − ⌈Ø_s/2⌉)",
    not lits and all(abs((a3[f][1] - POS._demi(s3["faces"][f]["couches"][1])) - (a3[f][0] - POS._demi(s3["faces"][f]["couches"][0]))) < 1e-12
                     for f in ("inf_x", "sup_x")), lits)

# ------------------------------------------------------------------ (c) prédalle
print("\n=== (c) prédalle : règle du moteur en bas, empilement en haut (renforts compris) ===")
dp = dalle("predalle")
ap = autos_site(POS, dp)
chk("inf. Y (principale) dans la peau : c + ⌈Ø/2⌉ = 4,0 ; inf. X sur la prédalle : h_pre + ⌈Ø/2⌉ = 6 + 0,5 = 6,5 ; renfort inf. Y : 6 + 1,0 = 7,0 — "
    "la règle du moteur, telle quelle", (ap["inf_y"], ap["inf_x"]) == ([4.0, 7.0], [6.5]) and ap["inf_y"] == autos_moteur(dp)["inf_y"]
    and ap["inf_x"] == autos_moteur(dp)["inf_x"], ap)
chk("sup. Y 4,0 (moteur) ; sup. X empilée 4,0 + 1,0 + 0,5 = 5,5 (le moteur dirait 3,5)",
    (ap["sup_y"][0], ap["sup_x"][0]) == (4.0, 5.5) and autos_moteur(dp)["sup_x"][0] == 3.5, ap)
chk("empilee : vrai pour sup. X seulement", [f for f in EN_D.FACES if POS.empilee(dp, dp["sections"][0], f, 0)] == ["sup_x"])
dp3 = dalle_s3("predalle")
ap3 = autos_site(POS, dp3)
chk("S3 en prédalle — le renfort sup. X (3 Ø16) posé sur la base sup. Y : 4,0 + 1,0 + 1,0 = 6,0 (le moteur : 3 + 1,0 = 4,0) ; le renfort inf. X "
    "(3 Ø12) reste SUR la prédalle à 6 + 1,0 = 7,0, la règle du moteur, déjà dans le lit de sa base (6,5 : h_pre + ⌈Ø/2⌉ l'une et l'autre)",
    ap3["sup_x"] == [5.5, 6.0] and autos_moteur(dp3)["sup_x"][1] == 4.0 and ap3["inf_x"] == [6.5, 7.0] == autos_moteur(dp3)["inf_x"]
    and not POS.empilee(dp3, dp3["sections"][0], "inf_x", 1) and POS.empilee(dp3, dp3["sections"][0], "sup_x", 1), ap3)

# ------------------------------------------------------------------ (d) pour_moteur, annoter, candidats_base
print("\n=== (d) pour_moteur : la copie que le moteur lit ; annoter ; le sélecteur ===")
d = dalle()
avant = copy.deepcopy(d)
pm = POS.pour_moteur(d)
chk("pour_moteur : dist renseigné pour inf. X et sup. X (5,5) seulement, les autres couches intactes (None)",
    {f: [c["dist"] for c in pm["sections"][0]["faces"][f]["couches"]] for f in EN_D.FACES}
    == {"inf_x": [5.5], "sup_x": [5.5], "inf_y": [None, None], "sup_y": [None]}, pm["sections"][0]["faces"])
chk("pour_moteur : sans verrou ni type ; la dalle d'origine n'est pas modifiée",
    not (set(pm) & POS.CLES_SITE) and d == avant and d["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] is None)
pm3 = POS.pour_moteur(dalle_s3())
chk("pour_moteur (S3) : les renforts secondaires reçoivent leur axe empilé (6,0), le renfort principal reste automatique",
    {f: [c["dist"] for c in pm3["sections"][0]["faces"][f]["couches"]] for f in EN_D.FACES}
    == {"inf_x": [5.5, 6.0], "sup_x": [5.5, 6.0], "inf_y": [None, None], "sup_y": [None]}, pm3["sections"][0]["faces"])
r_emp = MO_D.verifier(pm)
r_mot = MO_D.verifier(moteur_de(d))
g_e, g_m = r_emp["sections"][0]["grandeurs"], r_mot["sections"][0]["grandeurs"]
chk("verifier(pour_moteur(d)) : d utile de la direction secondaire plus petit (inf. X 14,5 contre 16,5 ; sup. X pareil), la principale inchangée",
    g_e["d_inf_x"] == 14.5 < g_m["d_inf_x"] == 16.5 and g_e["d_sup_x"] == 14.5 < g_m["d_sup_x"] and g_e["d_inf_y"] == g_m["d_inf_y"]
    and g_e["As_req_inf_x"] > g_m["As_req_inf_x"], (g_e["d_inf_x"], g_m["d_inf_x"]))
chk("… et l'avertissement « nappes_confondues » du moteur disparaît (les bases ne sont plus au même niveau)",
    [x["code"] for x in r_mot["avertissements"]] == ["nappes_confondues", "nappes_confondues"] and not r_emp["avertissements"])
s = d["sections"][0]
chk("h_utile(d, s, f) du site = d du calcul, pour les quatre faces", all(abs(POS.h_utile(d, s, f) - g_e[f"d_{f}"]) < 1e-12 for f in EN_D.FACES))
chk("yG_calcule(d, s, f) du site = y_G calculé du moteur", all(abs(POS.yG_calcule(d, s, f) - g_e[f"yG_calcule_{f}"]) < 1e-12 for f in EN_D.FACES))
r3e, r3m = MO_D.verifier(pm3), MO_D.verifier(moteur_de(dalle_s3()))
chk("S3 — le sens de la sécurité : le renfort secondaire monte, y_G inf. X monte, d inf. X baisse, A_s,req inf. X augmente (sup. X pareil)",
    all(r3e["sections"][0]["grandeurs"][f"yG_{f}"] > r3m["sections"][0]["grandeurs"][f"yG_{f}"]
        and r3e["sections"][0]["grandeurs"][f"As_req_{f}"] > r3m["sections"][0]["grandeurs"][f"As_req_{f}"] for f in ("inf_x", "sup_x")),
    {f: (r3e["sections"][0]["grandeurs"][f"As_req_{f}"], r3m["sections"][0]["grandeurs"][f"As_req_{f}"]) for f in ("inf_x", "sup_x")})
for type_ in ("coulee", "predalle"):
    d0 = dalle(type_)
    for f in ("inf_x", "sup_x"):                                  # tous les axes secondaires saisis : rien à empiler
        d0["sections"][0]["faces"][f]["couches"][0]["dist"] = 7.0
    MO = POS.moteur(d0)
    chk(f"{type_}, axes secondaires SAISIS (rien d'empilé) : pour_moteur n'écrit rien, verifier(pour_moteur(d)) == verifier(d) à l'octet (JSON)",
        json.dumps(MO.verifier(POS.pour_moteur(d0)), sort_keys=True) == json.dumps(MO.verifier(moteur_de(d0)), sort_keys=True))
dp = dalle("predalle")
pmp = POS.pour_moteur(dp)
chk("prédalle : pour_moteur renseigne sup. X (5,5) seulement, h_pre passe au moteur",
    {f: [c["dist"] for c in pmp["sections"][0]["faces"][f]["couches"]] for f in EN_D.FACES}
    == {"inf_x": [None], "sup_x": [5.5], "inf_y": [None, None], "sup_y": [None]} and pmp["h_pre"] == 6.0)
# annoter : la notion d'automatique du site posée sur le résultat
r = POS.annoter(MO_D.verifier(pm), d)
cx, cy = r["sections"][0]["faces"]["inf_x"]["couches"][0], r["sections"][0]["faces"]["inf_y"]["couches"][0]
chk("annoter (dalle) : inf. X auto (dans l'état), e_auto 5,5 (le site), e_auto_moteur 3,5, empilee ; inf. Y auto, 4,0, pas empilée",
    (cx["auto"], cx["e_auto"], cx["e_auto_moteur"], cx["empilee"], cx["e"]) == (True, 5.5, 3.5, True, 5.5)
    and (cy["auto"], cy["e_auto"], cy["e_auto_moteur"], cy["empilee"]) == (True, 4.0, 4.0, False), cx)
r3 = POS.annoter(MO_D.verifier(pm3), dalle_s3())
c3 = r3["sections"][0]["faces"]["inf_x"]["couches"][1]
chk("annoter (S3) : le renfort inf. X auto, e_auto 6,0 (« auto · 6 » à l'écran), e_auto_moteur 4,0, empilé — et le moteur l'a utilisé à 6,0",
    (c3["auto"], c3["e_auto"], c3["e_auto_moteur"], c3["empilee"], c3["e"]) == (True, 6.0, 4.0, True, 6.0), c3)
rp = POS.annoter(MO_P.verifier(pmp), dp)
csx = rp["sections"][0]["faces"]["sup_x"]["couches"][0]
fe = [fo for v in rp["sections"][0]["verifications"] if v["cle"] == "sup_x" for fo in v["formules"] if fo["cle"] == "e"]
chk("annoter (prédalle) : sup. X auto, empilée, e_auto 5,5 ; sa formule « e » porte empilee et les termes e_p, demi_p, demi_s (le terme "
    "« saisi » du moteur gardé)", (csx["auto"], csx["empilee"], csx["e_auto"], csx["pose"]) == (True, True, 5.5, "superieure")
    and len(fe) == 1 and fe[0]["empilee"] and fe[0]["auto"]
    and {t["symbole"]: t["valeur"] for t in fe[0]["termes"]} == {"saisi": 5.5, "e_p": 4.0, "d_p": 12.0, "demi_p": 1.0, "d_s": 10.0, "demi_s": 0.5},
    fe)
dp3 = dalle_s3("predalle")
rp3 = POS.annoter(MO_P.verifier(POS.pour_moteur(dp3)), dp3)
fe3 = [fo for v in rp3["sections"][0]["verifications"] if v["cle"] == "sup_x" for fo in v["formules"] if fo["cle"] == "e" and fo["couche"] == 1]
chk("annoter (prédalle, S3) : la formule « e » du renfort sup. X dit e_p + ⌈Ø_p/2⌉ + ⌈Ø/2⌉ = 4,0 + 1,0 + 1,0 (Ø16 : demi 1,0) — ses termes, le "
    "renfort comme couche posée",
    len(fe3) == 1 and fe3[0]["empilee"] and {t["symbole"]: t["valeur"] for t in fe3[0]["termes"]}
    == {"saisi": 6.0, "e_p": 4.0, "d_p": 12.0, "demi_p": 1.0, "d_s": 16.0, "demi_s": 1.0}, fe3)
# le sélecteur : chaque candidat à SON axe empilé
cands = POS.candidats_base(d, 0, "inf_x")
figes = MO_D.candidats_base(pm, 0, "inf_x")
par_nom = {c["nom"]: c for c in cands}
chk("candidats_base, base secondaire inf. X : un Ø8 posé à 5,5, un Ø12 à 4,0 + 1,0 + 1,0 = 6,0, un Ø25 à 6,5 — le moteur sur l'axe figé les "
    "mettrait tous à 5,5", (par_nom["Ø8/150"]["e"], par_nom["Ø12/150"]["e"], par_nom["Ø25/150"]["e"]) == (5.5, 6.0, 6.5)
    and {c["nom"] for c in cands} == {c["nom"] for c in figes} and all(abs(c["yG"] - 5.5) < 1e-9 for c in figes if c["nom"] == "Ø12/150"))
chk("… même tri que le moteur (section au mètre), la retenue et une optimale marquées, mêmes clés plus « e »",
    [c["nom"] for c in cands] == [c["nom"] for c in figes] and sum(c["retenue"] for c in cands) == 1 and sum(c["optimal"] for c in cands) == 1
    and set(cands[0]) == set(figes[0]) | {"e"})
cp = POS.candidats_base(d, 0, "inf_y")
chk("candidats_base, base principale inf. Y : un Ø25 (axe 4,5) pousse la secondaire à 4,5 + 1,5 + 0,5 = 6,5 — le statut de section le sait",
    next(c for c in cp if c["nom"] == "Ø25/150")["e"] == 4.5 and len(cp) == len(figes))
d3 = dalle_s3()
cp3 = {c["nom"]: c for c in POS.candidats_base(d3, 0, "inf_y")}
essai = copy.deepcopy(d3)
essai["sections"][0]["faces"]["inf_y"]["couches"][0].update(MO_D.lire_candidat("Ø25/150"))
r_essai = MO_D.verifier(POS.pour_moteur(essai))["sections"][0]
chk("candidats_base (S3) : la base principale candidate Ø25 (axe 4,5) pousse AUSSI le renfort secondaire à 4,5 + 1,5 + 1,0 = 7,0 ; le "
    "statut de section du candidat est celui de la dalle recomposée",
    r_essai["faces"]["inf_x"]["couches"][1]["e"] == 7.0 and cp3["Ø25/150"]["statut_section"] == r_essai["statut"])
try:
    POS.candidats_base(d, 3, "inf_x")
    chk("candidats_base : section introuvable → ValueError", False)
except ValueError as e:
    chk("candidats_base : section introuvable → ValueError", "introuvable" in str(e))

# ------------------------------------------------------------------ (e) les dessins
print("\n=== (e) les dessins : chaque couche à son axe réel, lu dans le résultat ===")
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
    """Le résultat du moteur SEUL (la dalle brute, comme l'écran Streamlit) et sa coupe : le témoin des niveaux d'avant."""
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
    ex1, ex0 = r1["sections"][0]["faces"]["inf_x"]["couches"][0]["e"], r0["sections"][0]["faces"]["inf_x"]["couches"][0]["e"]
    if type_ == "coulee":
        _, ri1 = niveaux(sv1, "inf_y-0", "inf_x-1")
        chk("coupe ⊥ Y (dalle) : les barres Y (points) 15 mm plus bas que les barres X (trait) — e 4,0 contre 5,5 ; le moteur seul : X et Y au "
            "plan du moteur (Y Ø12 à 40, X Ø10 à 35 : l'écart n'est que les 5 mm de Ø), en bas et en haut",
            len(cy1) == 1 and len(y1) == 1 and abs((min(cy1) - min(y1)) - 15.0) < 1e-6 and ex1 == 5.5
            and abs((min(cy0) - min(y0)) - (-5.0)) < 1e-6 and ex0 == 3.5 and abs((min(sx0) - min(sy0)) - (-5.0)) < 1e-6, (cy1, y1, cy0, y0))
        chk("en haut : sup. X 15 mm sous sup. Y (depuis le dessus : 55 contre 40 mm)", abs((min(sx1) - min(sy1)) - 15.0) < 1e-6, (sy1, sx1))
        chk("S3 sur la coupe : le renfort inf. X (trait) 5 mm au-dessus de la base X (axe 60 contre 55 : le même lit, Ø12 sur Ø10), soit 20 mm "
            "au-dessus des barres Y ; en haut, le renfort sup. X à 60 (moteur seul : 40, le niveau des barres Y)",
            abs((min(y1) - min(ri1)) - 5.0) < 1e-6 and abs((min(cy1) - min(ri1)) - 20.0) < 1e-6
            and abs(min(rx1) - min(sy1) - 20.0) < 1e-6 and abs(min(rx0) - min(sy0)) < 1e-6, (y1, ri1, rx1, sy1))
    else:
        chk("coupe ⊥ Y (prédalle) : en bas la règle du moteur (Y dans la peau à 40, X sur la prédalle à 65 : 25 mm), en haut l'empilement "
            "(sup. X 15 mm sous sup. Y) ; le moteur seul : en haut X et Y au même niveau (35 et 40 : les 5 mm de Ø)",
            abs((min(cy1) - min(y1)) - 25.0) < 1e-6 and abs((min(sx1) - min(sy1)) - 15.0) < 1e-6
            and abs((min(sx0) - min(sy0)) - (-5.0)) < 1e-6, (cy1, y1, sy1, sx1))
        chk("S3 sur la coupe de la prédalle : le renfort sup. X à 60 (posé sur la base sup. Y, 20 mm sous elle), le moteur seul : 40",
            abs(min(rx1) - min(sy1) - 20.0) < 1e-6 and abs(min(rx0) - min(sy0)) < 1e-6, (rx1, sy1, rx0, sy0))
        chk("la peau hachurée et le plan de reprise restent (prédalle 60 mm, coulé 160 mm)", 'stroke="#8C97A6"' in sv1 and "prédalle préfabriquée · 60 mm" in sv1
            and "béton coulé en place · 160 mm" in sv1)
        chk("légende prédalle : les couches empilées (sup. X, son renfort) disent « sur la base Y », jamais « axe saisi » (le site les a posées, "
            "pas l'ingénieur) ; les inférieures disent la pose du moteur",
            "axe à 55 · sur la base Y" in sv1 and "axe à 60 · sur la base Y" in sv1 and "axe saisi" not in sv1
            and "axe à 40 · dans la prédalle" in sv1 and "axe à 65 · sur la prédalle" in sv1, re.findall(r"axe à \d+[^<]*", sv1)[:8])
    # chaque couche est dessinée à son e : les y des points et des traits sont h − e (inf.) ou e (sup.), en mm, à 0,05 près (arrondi SVG)
    yc = None
    faute = []
    for cote in ("inf", "sup"):
        for dk in ("x", "y"):
            for j, c in enumerate(r1["sections"][0]["faces"][f"{cote}_{dk}"]["couches"]):
                coupe = sv1.split("data-coupe=")[1]                                    # ⊥ Y : Y en points, X en traits
                m = re.search(GROUPE.format(re.escape(f"{cote}_{dk}-{j}")), coupe)
                ys = [float(x) for x in re.findall(r'cy="([\d.]+)"' if dk == "y" else r'y1="([\d.]+)"', m.group(1))]
                attendu = (h - c["e"] * 10.0) if cote == "inf" else c["e"] * 10.0
                if yc is None:
                    yc = ys[0] - attendu                                              # le dessus du béton, lu une fois
                if any(abs(y - (yc + attendu)) > 0.06 for y in ys):
                    faute.append((cote, dk, j, c["e"], ys[:2]))
    chk(f"{type_} : chaque couche (bases, renforts des deux directions) est à h − e ou e du résultat du moteur, au dixième de mm",
        not faute, faute)
    # les cotes de d : celles du calcul, pour la face tendue de chaque direction (ici les inférieures : M_inf > M_sup)
    g1 = r1["sections"][0]["grandeurs"]
    coupes = sv1.split("data-coupe=")
    f_emp = "d_inf_x" if type_ == "coulee" else "d_sup_x"               # la face que l'empilement abaisse (prédalle : en haut seulement)
    chk(f"{type_} : la cote d de la coupe ⊥ Y est h − y_G inf. Y du calcul ({DDm._mm(g1['d_inf_y'] * 10)}), celle de la coupe ⊥ X h − y_G inf. X "
        f"({DDm._mm(g1['d_inf_x'] * 10)}) ; {f_emp} plus petit que celui du moteur seul",
        f"d = {DDm._mm(g1['d_inf_y'] * 10)}" in coupes[1] and f"d = {DDm._mm(g1['d_inf_x'] * 10)}" in coupes[2]
        and g1[f_emp] < r0["sections"][0]["grandeurs"][f_emp], re.findall(r"d = [\d ,]+", sv1))
    chk(f"{type_} : les pièces cliquables gardent leurs clés (s0.<face>-<j>, s0.beton)",
        {k for k in re.findall(r'data-sel="([^"]+)"', sv1)} == {f"s0.{f}-{j}" for f in EN_D.FACES for j in range(len(r1["sections"][0]["faces"][f]["couches"]))} | {"s0.beton"})
r1, sv1 = dessin(POS, DD, dalle_s3(), MO_D)
chk("légende dalle : les couches empilées disent « axe à 55 · sur la base Y » (la base X) et « axe à 60 · sur la base Y » (son renfort, S3) — "
    "court : la ligne tient dans sa colonne ; la principale « axe à 40 » sans plus",
    "X · Treillis 10/10/100/100 · axe à 55 · sur la base Y" in sv1 and "X · 3 Ø12 · axe à 60 · sur la base Y" in sv1
    and "Y · Ø12/150 · axe à 40<" in sv1, re.findall(r"axe à \d+[^<]*", sv1)[:6])
r0, sv0 = dessin_brut(DD, dalle(), MO_D)
chk("le moteur seul (non annoté) : aucune mention « sur la base », aucun « axe saisi » (rien que le calcul n'ait fait)",
    "sur la base" not in sv0 and "axe saisi" not in sv0)
# le dessin sans annotation ne ment pas sur les niveaux (les y sont ceux de e) ; seule la légende de la prédalle lirait « saisi »
r_brut = MO_P.verifier(POS.pour_moteur(dalle("predalle")))
sv_brut = DP.svg_coupe("t", r_brut["sections"][0], r_brut["predalle"], 0)
sv_ann = DP.svg_coupe("t", POS.annoter(copy.deepcopy(r_brut), dalle("predalle"))["sections"][0], r_brut["predalle"], 0)
chk("prédalle SANS annoter : les niveaux sont justes (les mêmes qu'annoté), mais la légende dit « axe saisi » de la couche que le site a "
    "posée — c'est pourquoi api_web/dalle.py doit passer par annoter",
    niveaux(sv_brut, "sup_y-0", "sup_x-0") == niveaux(sv_ann, "sup_y-0", "sup_x-0") and "axe à 55 · axe saisi" in sv_brut
    and "axe à 55 · sur la base Y" in sv_ann)
# la cote de d d'une prédalle suit la face TENDUE (comme la dalle) : sup. quand son moment l'emporte
da = dalle("predalle")
da["sections"][0].update(My_inf=5.0, My_sup=30.0)                      # (M_x,inf = 15 > M_x,sup = 5 : X reste tendue en bas)
ra, sva = dessin(POS, DP, da, MO_P)
ga = ra["sections"][0]["grandeurs"]
chk("prédalle, M_y,sup > M_y,inf : la coupe ⊥ Y cote d = h − y_G SUP. Y (comme la coupe des dalles), la coupe ⊥ X d inf. X",
    f"d = {DP._mm(ga['d_sup_y'] * 10)}" in sva.split("data-coupe=")[1] and f"d = {DP._mm(ga['d_inf_y'] * 10)}" not in sva.split("data-coupe=")[1]
    and f"d = {DP._mm(ga['d_inf_x'] * 10)}" in sva.split("data-coupe=")[2], re.findall(r"d = [\d ,]+", sva))
# svg_toutes et svg_plan tournent sur un résultat annoté (plan : la couche la plus haute recouvre, l'ordre lit e)
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


REGLE = '    return appui(d, s, cote)["dessus"] + _demi(c)\n'
M1 = muter("dalle_positions.py", REGLE, '    return appui(d, s, cote)["dessus"]\n', "positions_m1")
chk("mutation 1 — la règle sans ⌈Ø/2⌉ de la couche posée : inf. X vaudrait 5,0 et non 5,5 → le contrôle (b) rougit",
    autos_site(M1, dalle())["inf_x"][0] == 5.0 != 5.5)
M2 = muter("dalle_positions.py", "                        c_m[\"dist\"] = a\n", "                        c_m[\"dist\"] = None\n", "positions_m2")
pm2 = M2.pour_moteur(dalle())
chk("mutation 2 — pour_moteur n'écrit plus dist : le moteur garde ses axes, d utile inf. X 16,5 et non 14,5 → le contrôle (d) rougit",
    pm2["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] is None and MO_D.verifier(pm2)["sections"][0]["grandeurs"]["d_inf_x"] == 16.5)
M3 = muter("dessin_dalle.py", '                y = (h_mm - c["e"] * 10.0) if cote == "inf" else c["e"] * 10.0\n',
           '                y = (h_mm - c["e_auto_moteur"] * 10.0) if cote == "inf" else c["e_auto_moteur"] * 10.0\n', "dessin_m3")
rm, svm = dessin(POS, M3, dalle(), MO_D)
cym, ym = niveaux(svm, "inf_y-0", "inf_x-0")
chk("mutation 3 — le dessin lit l'automatique du moteur au lieu de e (l'axe utilisé) : X et Y reviennent au plan du moteur (écart −5 mm, "
    "pas 15) → le contrôle (e) rougit", abs((min(cym) - min(ym)) - (-5.0)) < 1e-6, (cym, ym))
GARDE = '    if dk == principale(d) or (cote == "inf" and type_dalle(d) == "predalle"):\n'
M4 = muter("dalle_positions.py", GARDE, '    if j > 0 or dk == principale(d) or (cote == "inf" and type_dalle(d) == "predalle"):\n', "positions_m4")
chk("mutation 4 — S3 retiré (un renfort reste à l'automatique du moteur) : le renfort inf. X retombe à 4,0 (au niveau de la base Y) et le "
    "sup. X de la prédalle à 4,0 → les contrôles S3 de (b), (c) et (d) rougissent",
    autos_site(M4, dalle_s3())["inf_x"][1] == 4.0 != 6.0 and autos_site(M4, dalle_s3("predalle"))["sup_x"][1] == 4.0
    and not M4.empilee(dalle_s3(), dalle_s3()["sections"][0], "inf_x", 1))
M5 = muter("dalle_positions.py", GARDE, '    if not d.get("empilement") or dk == principale(d) or (cote == "inf" and type_dalle(d) == "predalle"):\n',
           "positions_m5")
chk("mutation 5 — l'option revenue (clé « empilement » absente lue fausse) : toutes les couches reviennent à l'automatique du moteur, la base "
    "inf. X à 3,5 → les contrôles (a) et (b) rougissent",
    autos_site(M5, dalle()) == autos_moteur(dalle()) and autos_site(M5, dalle())["inf_x"][0] == 3.5)

# ------------------------------------------------------------------ (g) S5 : la couche principale la plus haute porte
print("\n=== (g) S5 : la nappe secondaire repose sur la couche principale la plus haute de son lit (renfort compris) ===")


def renfort(diam, n=3, dist=None, type_="n_barres", treillis="10/10/100/100"):
    return {"type": type_, "treillis": treillis, "d": diam, "esp": 150, "n": n, "dist": dist}


def dalle_s5(type_="coulee", principale="Y", cote="inf", renforts=((12,),), base=None, sec_renfort=None, **modifs):
    """La note NEUVE du site (treillis 10/10/100/100 partout, c = 3 cm, jeu 0 ; prédalle h = 22, h_p = 6) en mode
    « Treillis + renforts » : des RENFORTS sur la face principale ``cote`` (``renforts`` : (Ø, n, dist) ; par défaut le
    3 Ø12 de « + Renfort »), ``base`` : une autre base principale, ``sec_renfort`` : le Ø d'un renfort S3 sur la
    secondaire de la même face ; moments sur les quatre faces (M_x,inf = 20)."""
    d = EN_D.nouvelle_dalle() if type_ == "coulee" else EN_P.nouvelle_predalle()
    d["dir_principale"] = principale
    p = principale.lower()
    q = "x" if p == "y" else "y"
    s = d["sections"][0]
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
    return POS.axe_auto(d, d["sections"][0], f, j)


def ap(d, cote="inf"):
    return POS.appui(d, d["sections"][0], cote)


def nom(d, cote="inf"):
    return POS.nom_appui(d, d["sections"][0], cote)


# la règle
d12 = dalle_s5()
chk("le mode par défaut « Treillis + renforts » (base inf. Y treillis Ø10 à 3,5, renfort 3 Ø12 à 4,0 : le même dessous 3,0) : le lit "
    "contient les deux, le renfort porte (dessus 5,0 contre 4,0) — la base inf. X à 5,0 + 0,5 = 5,5 (sur la base seule : 4,5), « le "
    "renfort Y » ; la face sup. (sans renfort) reste sur sa base, 4,5",
    POS.lit_contact(d12, d12["sections"][0], "inf") == [0, 1] and ap(d12)["couche"] == 1 and ap(d12)["dessus"] == 5.0
    and ax(d12, "inf_x") == 5.5 and nom(d12) == "le renfort Y" and ax(d12, "sup_x") == 4.5 and ap(d12, "sup")["couche"] == 0
    and nom(d12, "sup") == "la base Y", (ap(d12), ax(d12, "inf_x"), ax(d12, "sup_x")))
d25 = dalle_s5(renforts=((25,),), sec_renfort=16)
chk("un renfort principal 3 Ø25 (3 + 1,5 = 4,5, dessus 6,0) : la base inf. X à 6,5 et son renfort S3 3 Ø16 à 6,0 + 1,0 = 7,0 — la même "
    "nappe secondaire, sur le même renfort (le moteur seul : 3,5 et 4,0)",
    ax(d25, "inf_x") == 6.5 and ax(d25, "inf_x", 1) == 7.0 and ap(d25) == {"couche": 1, "e": 4.5, "d": 25.0, "demi": 1.5, "dessus": 6.0, "lit": [0, 1]}
    and autos_moteur(d25)["inf_x"] == [3.5, 4.0], (ap(d25), POS.axes(d25, d25["sections"][0])["inf_x"]))
dsup = dalle_s5(cote="sup", renforts=((20,),))
dpx = dalle_s5(principale="X", renforts=((25,),))
chk("la même règle en haut (renfort sup. Y Ø20 à 4,0, dessus 5,0 : sup. X 5,5 depuis le dessus) et principale X (renfort inf. X Ø25 : "
    "la base inf. Y, secondaire, à 6,0 + 0,5 = 6,5, « le renfort X »)",
    ax(dsup, "sup_x") == 5.5 and nom(dsup, "sup") == "le renfort Y" and ax(dsup, "inf_x") == 4.5
    and ax(dpx, "inf_y") == 6.5 and nom(dpx) == "le renfort X", (ax(dsup, "sup_x"), ax(dpx, "inf_y")))
dps = dalle_s5("predalle", cote="sup", renforts=((20,),))
dpi = dalle_s5("predalle", renforts=((20,),))
rpi = MO_P.verifier(POS.pour_moteur(dpi))
chk("prédalle : en haut, la règle des dalles (renfort sup. Y Ø20 : sup. X 5,5) ; en BAS, la règle du type, inchangée (base inf. Y dans la "
    "peau à 3,5, renfort inf. Y Ø20 et base inf. X tous deux SUR la prédalle, 7,0 et 6,5 : l'axe inf. X n'est pas empilé) — et le moteur le "
    "dit : « croisement_sur_predalle »",
    ax(dps, "sup_x") == 5.5 and ax(dpi, "inf_x") == 6.5 == autos_moteur(dpi)["inf_x"][0] and autos_moteur(dpi)["inf_y"] == [3.5, 7.0]
    and not POS.empilee(dpi, dpi["sections"][0], "inf_x", 0)
    and "croisement_sur_predalle" in [a["code"] for a in rpi["avertissements"]], (ax(dps, "sup_x"), ax(dpi, "inf_x")))
# les cas limites du lit de contact
T6 = {"type": "treillis", "treillis": "6/6/150/150", "d": 12, "esp": 150, "n": 3, "dist": None}
B10 = {"type": "barres", "treillis": "10/10/100/100", "d": 10, "esp": 150, "n": 3, "dist": None}
B16 = {"type": "barres", "treillis": "10/10/100/100", "d": 16, "esp": 150, "n": 3, "dist": None}
limites = {
    "2e lit : renfort Ø25 SAISI à 9 cm (dessous 7,5 > dessus du lit 4,0) — il ne porte rien": (dalle_s5(renforts=((25, 3, 9.0),)), 4.5, [0], 0, "la base Y"),
    "contact exact : renfort Ø12 saisi à 5,0 (dessous 4,0 = dessus de la base) — du lit, il porte": (dalle_s5(renforts=((12, 3, 5.0),)), 6.5, [0, 1], 1, "le renfort Y"),
    "0,01 cm au-dessus : renfort Ø12 saisi à 5,01 (dessous 4,01) — un autre lit (aucune marge)": (dalle_s5(renforts=((12, 3, 5.01),)), 4.5, [0], 0, "la base Y"),
    "chaîne : R1 Ø25 automatique (dessus 6,0), R2 Ø12 saisi à 6,5 (dessous 5,5 ≤ 6,0) — R2 entre par R1 et porte (7,5 + 0,5)":
        (dalle_s5(renforts=((25,), (12, 3, 6.5))), 8.0, [0, 1, 2], 2, "le renfort R2 Y"),
    "le même R2 sans R1 (dessous 5,5 > 4,0) — hors du lit": (dalle_s5(renforts=((12, 3, 6.5),)), 4.5, [0], 0, "la base Y"),
    "base saisie à 6,0 (dessus 6,5), renfort Ø12 automatique (dessus 5,0) — la base porte": (
        dalle_s5(base=dict(B10, dist=6.0), renforts=((12,),)), 7.0, [0, 1], 0, "la base Y"),
    "plusieurs renforts : 3 Ø12 et 2 Ø25 — le plus haut (R2) porte": (dalle_s5(renforts=((12,), (25, 2))), 6.5, [0, 1, 2], 2, "le renfort R2 Y"),
    "plusieurs renforts à égalité : 3 Ø12 et 2 Ø20 (dessus 5,0 l'un et l'autre) — le premier": (dalle_s5(renforts=((12,), (20, 2))), 5.5, [0, 1, 2], 1, "le renfort R1 Y"),
    "demis arrondis égaux : base Ø16/150, renfort Ø20 (dessus 5,0 et 5,0) — la base porte, rien ne change": (
        dalle_s5(base=B16, renforts=((20,),)), 5.5, [0, 1], 0, "la base Y"),
    "un renfort TREILLIS compte par son fil : treillis 12/12/150/150 (Ø12) sur une base treillis 6/6/150/150 (Ø6)": (
        dalle_s5(base=T6, renforts=((12, 3, None, "treillis", "12/12/150/150"),)), 5.5, [0, 1], 1, "le renfort Y"),
    "jeu premier lit 1 cm : renfort Ø25 à 5,5 (dessus 7,0) — le jeu entre une fois, par la principale": (
        dalle_s5(renforts=((25,),), jeu_premier_lit=1.0), 7.5, [0, 1], 1, "le renfort Y"),
}
fautes = []
for lib, (dd, attendu, lit, k, mot) in limites.items():
    s_ = dd["sections"][0]
    vu = (ax(dd, "inf_x"), POS.lit_contact(dd, s_, "inf"), ap(dd)["couche"], nom(dd))
    if vu != (attendu, lit, k, mot):
        fautes.append((lib, vu))
chk(f"cas limites du lit de contact ({len(limites)}) : {' ; '.join(limites)}", not fautes, fautes)
dsl = dalle_s5(renforts=((25, 3, 9.0),))
chk("un renfort principal saisi n'est jamais déplacé (axes() le garde à 9,0) ; un axe secondaire saisi non plus",
    POS.axes(dsl, dsl["sections"][0])["inf_y"] == [3.5, 9.0] and POS.pour_moteur(dsl)["sections"][0]["faces"]["inf_y"]["couches"][1]["dist"] == 9.0)
# relecture S5 : une couche principale AUTOMATIQUE est du lit quoi qu'il arrive (posée sur les cales du premier lit) — même sous une base
# SAISIE plus bas que l'automatique ; sans cela, le renfort automatique sortait du lit et la nappe secondaire le traversait
dbb = dalle_s5(base=dict(B10, d=8, dist=3.4), renforts=((20,),), jeu_premier_lit=1.0)
dbb0 = dalle_s5(base=dict(B10, dist=2.4), renforts=((20,),))
gbb = MO_D.verifier(POS.pour_moteur(dbb))["sections"][0]["grandeurs"]
chk("relecture S5 — base inf. Y Ø8 SAISIE à 3,4 (dessus 3,9) avec un jeu de 1 cm, renfort 3 Ø20 AUTOMATIQUE à 5,0 (sur les cales : dessous "
    "4,0, dessus 6,0) : le renfort est du lit et porte, la base inf. X à 6,0 + 0,5 = 6,5 (et non 4,4, au travers du renfort), d 13,5, "
    "A_s,req 493,8 mm² (et non 427,4) ; sans jeu, base Ø10 saisie à 2,4 : 5,5 (et non 3,4) ; un renfort SAISI au-dessus du lit reste un 2e lit",
    (POS.lit_contact(dbb, dbb["sections"][0], "inf"), ax(dbb, "inf_x"), nom(dbb), gbb["d_inf_x"], round(gbb["As_req_inf_x"], 1))
    == ([0, 1], 6.5, "le renfort Y", 13.5, 493.8) and ax(dbb0, "inf_x") == 5.5
    and ax(dalle_s5(base=dict(B10, dist=2.4), renforts=((20, 3, 6.0),)), "inf_x") == 3.4,
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
                e_b, e_r = (POS._auto_moteur(dd, c, j, f"{cote}_y") for j, c in enumerate(cs))
                h_b, h_r, h_s = POS._demi(cs[0]), POS._demi(cs[1]), POS._demi(s_["faces"][f"{cote}_x"]["couches"][0])
                a = ax(dd, f"{cote}_x")
                n_bal += 1
                if not (a == max(e_b + h_b, e_r + h_r) + h_s and a >= e_b + h_b + h_s and a - h_s >= max(e_b + h_b, e_r + h_r) - 1e-12):
                    bal.append((type_, cote, b_["type"], b_.get("treillis"), b_["d"], x, a))
chk(f"balayage ({n_bal} cas : dalle en bas et en haut, prédalle en haut ; chaque treillis et chaque Ø de base, chaque Ø de renfort) : "
    "l'axe secondaire = max(dessus de la base, dessus du renfort) + ⌈Ø_s/2⌉, jamais sous la base seule, jamais à travers le renfort",
    not bal and n_bal == 3 * (len(TR.liste_choix()) + len(EN_D.DIAMETRES)) * len(EN_D.DIAMETRES), bal[:4])
# le sens de la sécurité, mesuré par le moteur : A_s,req de la secondaire ne baisse jamais (même dalle, renfort principal hors du lit)
baisse, n_mon = [], 0
for x in EN_D.DIAMETRES:
    for type_, cote in (("coulee", "inf"), ("coulee", "sup"), ("predalle", "sup")):
        dd, dh = dalle_s5(type_, cote=cote, renforts=((x,),)), dalle_s5(type_, cote=cote, renforts=((x, 3, 15.0),))
        M = POS.moteur(dd)
        g_, gh = M.verifier(POS.pour_moteur(dd))["sections"][0]["grandeurs"], M.verifier(POS.pour_moteur(dh))["sections"][0]["grandeurs"]
        n_mon += 1
        if g_[f"As_req_{cote}_x"] < gh[f"As_req_{cote}_x"] or (POS._demi(renfort(x)) > 0.5) != (g_[f"As_req_{cote}_x"] > gh[f"As_req_{cote}_x"]):
            baisse.append((type_, cote, x, g_[f"As_req_{cote}_x"], gh[f"As_req_{cote}_x"]))
chk(f"A_s,req de la nappe secondaire ne baisse jamais ({n_mon} cas : chaque Ø de renfort principal, dalle inf./sup., prédalle sup.) ; "
    "il monte dès que le renfort dépasse la base (⌈Ø/2⌉ > 0,5 : Ø ≥ 12), il est égal sinon (Ø6 à Ø10)", not baisse, baisse)
# ce que le moteur reçoit, ce qu'il calcule (les mesures de la relecture du 09/11/2026)
g12 = MO_D.verifier(POS.pour_moteur(d12))["sections"][0]["grandeurs"]
g25 = MO_D.verifier(POS.pour_moteur(dalle_s5(renforts=((25,),))))["sections"][0]["grandeurs"]
g00 = MO_D.verifier(POS.pour_moteur(dalle_s5(renforts=())))["sections"][0]["grandeurs"]
chk("pour_moteur et le moteur (h = 20, M_x,inf = 20) : d inf. X 15,5 → 14,5 (3 Ø12) et 13,5 (3 Ø25) ; A_s,req inf. X 430,1 → 459,8 mm² "
    "(+ 6,9 %) et 493,8 mm² (+ 14,8 %) — les chiffres de la relecture",
    POS.pour_moteur(d12)["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] == 5.5
    and (g00["d_inf_x"], g12["d_inf_x"], g25["d_inf_x"]) == (15.5, 14.5, 13.5)
    and (round(g00["As_req_inf_x"], 1), round(g12["As_req_inf_x"], 1), round(g25["As_req_inf_x"], 1)) == (430.1, 459.8, 493.8)
    and round(100 * (g12["As_req_inf_x"] / g00["As_req_inf_x"] - 1), 1) == 6.9 and round(100 * (g25["As_req_inf_x"] / g00["As_req_inf_x"] - 1), 1) == 14.8,
    (g00["As_req_inf_x"], g12["As_req_inf_x"], g25["As_req_inf_x"]))
# annoter : appui et « sur » ; la formule « e » de la prédalle et sa pose disent la couche qui porte
r25 = POS.annoter(MO_D.verifier(POS.pour_moteur(d25)), d25)
cs25 = r25["sections"][0]["faces"]["inf_x"]["couches"]
chk("annoter : la base et le renfort inf. X empilés portent appui = 1, « le renfort Y » (formule, pose) et « renfort Y » (légendes) ; la "
    "base principale, non empilée, rien",
    [(c["appui"], c["sur"], c["sur_court"]) for c in cs25] == [(1, "le renfort Y", "renfort Y")] * 2 and "sur" not in r25["sections"][0]["faces"]["inf_y"]["couches"][0]
    and "sur" not in POS.annoter(MO_D.verifier(POS.pour_moteur(dalle())), dalle())["sections"][0]["faces"]["inf_y"]["couches"][0])
from api_web import predalle as AP  # noqa: E402
rps = POS.annoter(MO_P.verifier(POS.pour_moteur(dps)), dps)
fe5 = [fo for v in rps["sections"][0]["verifications"] if v["cle"] == "sup_x" for fo in v["formules"] if fo["cle"] == "e"]
gx5 = rps["sections"][0]["faces"]["sup_x"]["couches"][0]
chk("prédalle (renfort sup. Y Ø20) : la formule « e » de la base sup. X prend les termes du RENFORT (e_p 4,0, Ø_p 20, ⌈Ø_p/2⌉ 1,0) et "
    "s'écrit « 4 + 1 + 0,5 = 5,5 cm · posée sur le renfort Y » ; la pose dit « posée sur le renfort Y » ; une base porteuse garde « la base "
    "principale »",
    len(fe5) == 1 and {t["symbole"]: t["valeur"] for t in fe5[0]["termes"]} == {"saisi": 5.5, "e_p": 4.0, "d_p": 20.0, "demi_p": 1.0, "d_s": 10.0, "demi_s": 0.5}
    and fe5[0]["sur"] == "le renfort Y" and AP._formule(fe5[0])["res"] == "5,5 cm · posée sur le renfort Y"
    and AP._formule(fe5[0])["parts"] == ["e_1", "e_p + ⌈Ø_p/2⌉ + ⌈Ø_s/2⌉", "4 + 1 + 0,5"] and "posée sur le renfort Y (e" in AP.pose_couche(dps["sections"][0]["faces"]["sup_x"]["couches"][0], gx5)
    and "posée sur la base principale" in AP._formule(fe[0])["res"], (fe5, AP._formule(fe5[0]) if fe5 else None))
# le sélecteur : un candidat de la base secondaire repose sur le renfort principal ; un candidat de la base principale ne fait pas
# descendre la secondaire sous le renfort, un plus gros la fait monter
c5 = {c["nom"]: c for c in POS.candidats_base(d12, 0, "inf_x")}
chk("candidats_base, base secondaire inf. X sur le renfort 3 Ø12 (dessus 5,0) : Ø8 à 5,5, Ø12 à 6,0, Ø25 à 6,5 (sur la base seule : 4,5, 5,0, 5,5)",
    (c5["Ø8/150"]["e"], c5["Ø12/150"]["e"], c5["Ø25/150"]["e"]) == (5.5, 6.0, 6.5), {k: c5[k]["e"] for k in ("Ø8/150", "Ø12/150", "Ø25/150")})
essais = {}
for cand in ("Ø8/150", "Ø25/150"):
    e_ = copy.deepcopy(dalle_s5(renforts=((16,),)))
    e_["sections"][0]["faces"]["inf_y"]["couches"][0].update(MO_D.lire_candidat(cand))
    essais[cand] = MO_D.verifier(POS.pour_moteur(e_))["sections"][0]
cp5 = {c["nom"]: c for c in POS.candidats_base(dalle_s5(renforts=((16,),)), 0, "inf_y")}
chk("candidats_base, base principale sous un renfort Ø16 (dessus 5,0) : Ø8 (dessus 4,0) laisse la secondaire sur le renfort (5,5), Ø25 "
    "(dessus 6,0) la porte (6,5) — le statut de chaque candidat est celui de la dalle recomposée",
    essais["Ø8/150"]["faces"]["inf_x"]["couches"][0]["e"] == 5.5 and essais["Ø25/150"]["faces"]["inf_x"]["couches"][0]["e"] == 6.5
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
chk("coupe ⊥ Y (renfort principal 3 Ø20 sous la nappe X) : le bas du trait de la base X (axe 55, Ø10) touche le haut des points du renfort Y "
    "(axe 40, Ø20) — au contact, sans le traverser (avant S5 : axe 45, 10 mm dans le renfort)",
    abs(bas20 - haut20) < 0.06 and r20["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 5.5, (haut20, bas20))
chk("… la légende commune (DAL5) dit l'axe des deux fils du treillis (« axes : Y à 35 · X à 55 ») et sur quoi le renfort X repose, en "
    "forme courte (la ligne tient dans sa colonne) : « R1 X · 3 Ø12 · axe à 60 · sur renfort Y » ; la prédalle aussi (en haut : « axes : "
    "Y à 35 · X à 55 ») ; plusieurs renforts : X à 65 (le repère R2 est dans la forme longue, formule et pose : « le renfort R2 Y »)",
    "axes : Y à 35 · X à 55" in sv20 and "R1 X · 3 Ø12 · axe à 60 · sur renfort Y" in sv20
    and "axes : Y à 35 · X à 55" in dessin(POS, DP, dps, MO_P)[1]
    and "axes : Y à 35 · X à 65" in dessin(POS, DD, dalle_s5(renforts=((12,), (25, 2))), MO_D)[1]
    and POS.nom_appui(dalle_s5(renforts=((12,), (25, 2))), dalle_s5(renforts=((12,), (25, 2)))["sections"][0], "inf", court=True) == "renfort Y",
    re.findall(r"axe à \d+[^<]*", sv20)[:6])
# les mutations S5 (en mémoire)
LIT = "            if auto[p[0]] or p[1] - p[2] <= dessus + TOL_LIT:\n"
M6 = muter("dalle_positions.py", LIT, "            if False:\n", "positions_m6")
chk("mutation 6 — retour à la BASE SEULE (le lit réduit à la base) : la base inf. X retombe à 4,5 (3 Ø12) et 4,5 (Ø25), dans le renfort → "
    "les contrôles (g) rougissent", M6.axe_auto(d12, d12["sections"][0], "inf_x", 0) == 4.5 != 5.5
    and M6.axe_auto(d25, d25["sections"][0], "inf_x", 0) == 4.5 and M6.nom_appui(d12, d12["sections"][0], "inf") == "la base Y")
M7 = muter("dalle_positions.py", "TOL_LIT = 1e-6 ", "TOL_LIT = 0.5 ", "positions_m7")
d501, d9 = limites["0,01 cm au-dessus : renfort Ø12 saisi à 5,01 (dessous 4,01) — un autre lit (aucune marge)"][0], dalle_s5(renforts=((12, 3, 5.5),))
chk("mutation 7 — une marge de 0,5 cm au lit : le renfort saisi à 5,01 (et à 5,5) entre dans le lit, la nappe X monte à 6,51 (7,0) → le "
    "contrôle des cas limites rougit", M7.axe_auto(d501, d501["sections"][0], "inf_x", 0) > 6.5 and M7.axe_auto(d9, d9["sections"][0], "inf_x", 0) == 7.0
    and POS.axe_auto(d9, d9["sections"][0], "inf_x", 0) == 4.5)
M8 = muter("dalle_positions.py", "                dessus = max(dessus, p[1] + p[2])\n", "                pass\n", "positions_m8")
dch = limites["chaîne : R1 Ø25 automatique (dessus 6,0), R2 Ø12 saisi à 6,5 (dessous 5,5 ≤ 6,0) — R2 entre par R1 et porte (7,5 + 0,5)"][0]
chk("mutation 8 — le lit sans chaîne (le dessus du lit reste celui de la base) : R2 sort du lit, la nappe X redescend à 6,5, au travers de "
    "R2 → le contrôle de la chaîne rougit", M8.axe_auto(dch, dch["sections"][0], "inf_x", 0) == 6.5 and M8.lit_contact(dch, dch["sections"][0], "inf") == [0, 1])
M9 = muter("dalle_positions.py", "    k = next(j for j in lit if pts[j][1] + pts[j][2] >= haut - TOL_LIT)\n",
           "    k = next(j for j in reversed(lit) if pts[j][1] + pts[j][2] >= haut - TOL_LIT)\n", "positions_m9")
chk("mutation 9 — l'égalité donnée au dernier renfort au lieu de la base : dalle() (base et renfort Ø12) dirait « sur le renfort Y » → les "
    "légendes d'avant S5 et les égalités des cas limites rougissent", M9.nom_appui(dalle(), dalle()["sections"][0], "inf") == "le renfort Y"
    and M9.nom_appui(dalle(), dalle()["sections"][0], "inf") != POS.nom_appui(dalle(), dalle()["sections"][0], "inf"))
M10 = muter("dessin_dalle.py", """        return f"sur {c.get('sur_court') or 'la base ' + principale.upper()}"\n""",
            """        return f"sur la base {principale.upper()}"\n""", "dessin_m10")
chk("mutation 10 — la légende d'avant (« sur la base Y » pour toute couche empilée) : la coupe dirait « sur la base Y » d'un renfort X que "
    "le renfort Y porte → le contrôle de la légende rougit", "R1 X · 3 Ø12 · axe à 60 · sur la base Y" in dessin(POS, M10, d20, MO_D)[1]
    and "sur renfort Y" not in dessin(POS, M10, d20, MO_D)[1])
M11 = muter("dalle_positions.py", LIT, "            if p[1] - p[2] <= dessus + TOL_LIT:\n", "positions_m11")
chk("mutation 11 — une couche automatique jugée sur son seul dessous (le lit d'avant la relecture) : sous la base Ø8 saisie à 3,4, le renfort "
    "Ø20 automatique sort du lit, la base inf. X retombe à 4,4, au travers du renfort → le contrôle « relecture S5 » rougit",
    M11.lit_contact(dbb, dbb["sections"][0], "inf") == [0] and M11.axe_auto(dbb, dbb["sections"][0], "inf_x", 0) == 4.4
    and M11.axe_auto(dbb0, dbb0["sections"][0], "inf_x", 0) == 3.4)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
