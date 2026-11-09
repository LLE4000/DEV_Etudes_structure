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
    nappes au même niveau) — chacune fait rougir le contrôle qui la garde.

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
    coupe = svg.split("Coupe ⊥")[1]
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
            "e = 55 · sur la base Y" in sv1 and "e = 60 · sur la base Y" in sv1 and "axe saisi" not in sv1
            and "e = 40 · dans la prédalle" in sv1 and "e = 65 · sur la prédalle" in sv1, re.findall(r"e = \d+[^<]*", sv1)[:8])
    # chaque couche est dessinée à son e : les y des points et des traits sont h − e (inf.) ou e (sup.), en mm, à 0,05 près (arrondi SVG)
    yc = None
    faute = []
    for cote in ("inf", "sup"):
        for dk in ("x", "y"):
            for j, c in enumerate(r1["sections"][0]["faces"][f"{cote}_{dk}"]["couches"]):
                coupe = sv1.split("Coupe ⊥")[1]                                    # ⊥ Y : Y en points, X en traits
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
    coupes = sv1.split("Coupe ⊥")
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
    niveaux(sv_brut, "sup_y-0", "sup_x-0") == niveaux(sv_ann, "sup_y-0", "sup_x-0") and "e = 55 · axe saisi" in sv_brut
    and "e = 55 · sur la base Y" in sv_ann)
# la cote de d d'une prédalle suit la face TENDUE (comme la dalle) : sup. quand son moment l'emporte
da = dalle("predalle")
da["sections"][0].update(My_inf=5.0, My_sup=30.0)                      # (M_x,inf = 15 > M_x,sup = 5 : X reste tendue en bas)
ra, sva = dessin(POS, DP, da, MO_P)
ga = ra["sections"][0]["grandeurs"]
chk("prédalle, M_y,sup > M_y,inf : la coupe ⊥ Y cote d = h − y_G SUP. Y (comme la coupe des dalles), la coupe ⊥ X d inf. X",
    f"d = {DP._mm(ga['d_sup_y'] * 10)}" in sva.split("Coupe ⊥")[1] and f"d = {DP._mm(ga['d_inf_y'] * 10)}" not in sva.split("Coupe ⊥")[1]
    and f"d = {DP._mm(ga['d_inf_x'] * 10)}" in sva.split("Coupe ⊥")[2], re.findall(r"d = [\d ,]+", sva))
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
    exec(compile(src.replace(ancien, nouveau), chemin, "exec"), m.__dict__)
    return m


REGLE = "    return _axe_principal(d, s, cote) + _demi(_base_principale(d, s, cote)) + _demi(c)\n"
M1 = muter("dalle_positions.py", REGLE, "    return _axe_principal(d, s, cote) + _demi(_base_principale(d, s, cote))\n", "positions_m1")
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

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
