# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : le PRINCIPE D'ARMATURES (saf/fondation_grue/armatures.py) et les EFFORTS SCIA recopiés
(saf/fondation_grue/scia.py). Cahier des charges du lot 5, version 4 (15/11/2026), § 7, § 8, § 13 et annexe A : les tests T19,
A1 à A4 (sous-lot 5b) et E1 à E4 côté moteur (sous-lot 5c ; l'écran : tests/test_api_web_saf.py).

Ce que la suite prouve :
  1. T19, A1 — avec mRd imposé à 97,3 kNm/m et l'ancrage de 0,80 m, les renforts des bandes et les zones rouges sont, à
     l'écriture près, la sortie de l'annexe A (8 cas, bandes de 0,50 et de 1,00 m, plots, SCIA × 1,10) ; le module s'en sert
     tel quel (son ancrage : 50 Ø).
  2. LE MOTEUR DALLES — mRd = A_s du treillis ÷ A_s,req(1 kNm/m) (Ø12 e 15 : 97,3 en x, 95,0 en y, la couche y sur la couche
     x) ; A3 : le nombre de barres de chaque renfort, ⌈(A_s(m_dim) − A_s du treillis) × largeur ÷ aire d'une barre⌉ avec
     A_s() du moteur ; la carte de l'écran montre les mêmes barres ; A4 : la hauteur de béton et l'effort tranchant, les deux
     seuls cas (avec le sol) où le module dit d'épaisser ; le Ø automatique (le plus léger), imposé ; châssis tourné.
  3. SCIA (E1 à E4) — les valeurs grises (T14), la symétrie « x = y », un champ vide reste vide (r = 1), r = 1,10 → l'échelle
     du principe, l'écart hors de 0,85 à 1,15.
  4. LA PARITÉ AVEC LE CALCUL DE RÉFÉRENCE — tests/fondation_grue_ref.py (``principe_armatures``, scipy.ndimage) sur des
     dalles qui ne sont pas celles du cahier (contour en L, châssis tourné de 30°, grue décalée, mRd de 60 à 140).
  5. A2 — changer un treillis ou la largeur des bandes refait le principe, jamais le calcul du sol.
  6. PURETÉ (numpy seul) et MUTATIONS (arbre miroir, contrôles rapides).

    /usr/bin/python3.12 tests/test_saf_grue_armatures.py
    /usr/bin/python3.12 tests/test_saf_grue_armatures.py --rapide [--racine dossier]   (les contrôles rapides seuls)
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ARGS = sys.argv[1:]
RACINE_MOTEUR = ARGS[ARGS.index("--racine") + 1] if "--racine" in ARGS else RACINE
RAPIDE = "--rapide" in ARGS
os.chdir(RACINE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, RACINE_MOTEUR)
sys.path.insert(1, ICI)

import numpy as np  # noqa: E402

from saf.fondation_grue import armatures as AR  # noqa: E402
from saf.fondation_grue import efforts as EF  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402
from saf.fondation_grue import scia as SC  # noqa: E402
from saf.fondation_grue import verification as V  # noqa: E402

OK, KO = [], []
JEUX = [(350, 440, 440, 350), (150, 150, 550, 550), (135, 100, 800, 135)]
KS = (2.0, 5.0, 10.0)
MRD_ESSAI = {c: 97.3 for c in AR.CLES}          # T19 : le point d'essai du cahier
PREF = {"type": "prefabrique", "a": 2.0, "b": 2.0, "h": 0.35}
COULE = {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}
L_T7 = [(0, 0), (12, 0), (12, 7), (7, 7), (7, 12), (0, 12)]

T19_ATTENDU = """
T19 dalle type 8 x 8 x 0,50, bandes 0,50 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 0.00 a 2.05, y 6.75 a 7.25, m_dim 143.2 | 4 zone(s) rouge(s), 0.64 m2, max 117.4 ; 1re x 0.00 a 2.10, y 0.30 a 0.70
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 0.75 a 1.25, y 0.00 a 2.05, m_dim 143.2 | 4 zone(s) rouge(s), 0.64 m2, max 117.4 ; 1re x 0.30 a 0.70, y 0.00 a 2.10
    sup_x : 2 renfort(s) de bande ; 1er BX1 x 1.90 a 6.10, y 6.75 a 7.25, m_dim 126.7 | 4 zone(s) rouge(s), 8.96 m2, max 140.8 ; 1re x 1.70 a 6.30, y 0.00 a 0.70
    sup_y : 2 renfort(s) de bande ; 1er BY1 x 0.75 a 1.25, y 1.90 a 6.10, m_dim 126.7 | 4 zone(s) rouge(s), 8.96 m2, max 140.8 ; 1re x 0.00 a 0.70, y 1.70 a 6.30
T19 dalle type, bandes 1,00 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 0.00 a 2.00, y 6.50 a 7.50, m_dim 127.1 | 4 zone(s) rouge(s), 0.16 m2, max 100.8 ; 1re x 0.10 a 1.90, y 0.30 a 0.50
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 0.50 a 1.50, y 0.00 a 2.00, m_dim 127.1 | 4 zone(s) rouge(s), 0.16 m2, max 100.8 ; 1re x 0.30 a 0.50, y 0.10 a 1.90
    sup_x : 2 renfort(s) de bande ; 1er BX1 x 1.95 a 6.05, y 6.50 a 7.50, m_dim 126.5 | 4 zone(s) rouge(s), 6.88 m2, max 140.8 ; 1re x 1.70 a 6.30, y 0.00 a 0.50
    sup_y : 2 renfort(s) de bande ; 1er BY1 x 0.50 a 1.50, y 1.95 a 6.05, m_dim 126.5 | 4 zone(s) rouge(s), 6.88 m2, max 140.8 ; 1re x 0.00 a 0.50, y 1.70 a 6.30
T19 dalle type, plots coules en place, bandes 0,50 :
    inf_x : bandes : treillis suffisant | 4 zone(s) rouge(s), 1.28 m2, max 199.6 ; 1re x 0.00 a 2.90, y 0.00 a 0.30
    inf_y : bandes : treillis suffisant | 4 zone(s) rouge(s), 1.28 m2, max 199.6 ; 1re x 0.00 a 0.30, y 0.00 a 2.90
    sup_x : 2 renfort(s) de bande ; 1er BX1 x 1.60 a 6.40, y 6.75 a 7.25, m_dim 129.5 | 4 zone(s) rouge(s), 13.76 m2, max 169.0 ; 1re x 0.90 a 7.10, y 0.00 a 0.70
    sup_y : 2 renfort(s) de bande ; 1er BY1 x 0.75 a 1.25, y 1.60 a 6.40, m_dim 129.5 | 4 zone(s) rouge(s), 13.76 m2, max 169.0 ; 1re x 0.00 a 0.70, y 0.90 a 7.10
T19 dalle 10,70 x 10,70 x 0,50, bandes 0,50 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 1.10 a 3.60, y 8.10 a 8.60, m_dim 186.7 | 8 zone(s) rouge(s), 4.08 m2, max 167.8 ; 1re x 1.12 a 3.53, y 1.31 a 2.12
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 2.10 a 2.60, y 1.10 a 3.60, m_dim 186.7 | 8 zone(s) rouge(s), 4.08 m2, max 167.8 ; 1re x 1.31 a 2.12, y 1.12 a 3.53
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
T19 dalle 10,70, bandes 1,00 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 1.10 a 3.60, y 7.85 a 8.85, m_dim 170.1 | 8 zone(s) rouge(s), 2.12 m2, max 134.1 ; 1re x 1.12 a 3.53, y 1.31 a 1.92
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 1.85 a 2.85, y 1.10 a 3.60, m_dim 170.1 | 8 zone(s) rouge(s), 2.12 m2, max 134.1 ; 1re x 1.31 a 1.92, y 1.12 a 3.53
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
T19 dalle 10,70, plots prefabriques, bandes 0,50 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 1.05 a 3.65, y 8.10 a 8.60, m_dim 128.2 | 8 zone(s) rouge(s), 4.08 m2, max 125.6 ; 1re x 0.92 a 3.73, y 1.51 a 2.12
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 2.10 a 2.60, y 1.05 a 3.65, m_dim 128.2 | 8 zone(s) rouge(s), 4.08 m2, max 125.6 ; 1re x 1.51 a 2.12, y 0.92 a 3.73
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
T19 dalle 10,70, plots coules en place, bandes 0,50 :
    inf_x : bandes : treillis suffisant | 8 zone(s) rouge(s), 0.33 m2, max 110.8 ; 1re x 2.53 a 4.33, y 1.31 a 1.51
    inf_y : bandes : treillis suffisant | 8 zone(s) rouge(s), 0.33 m2, max 110.8 ; 1re x 3.33 a 3.53, y 0.51 a 2.31
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
T19 dalle 10,70, bandes 0,50, SCIA = module x 1,10 en bas :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 1.05 a 3.65, y 8.10 a 8.60, m_dim 205.4 | 8 zone(s) rouge(s), 5.71 m2, max 184.5 ; 1re x 0.92 a 3.73, y 1.31 a 2.12
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 2.10 a 2.60, y 1.05 a 3.65, m_dim 205.4 | 8 zone(s) rouge(s), 5.71 m2, max 184.5 ; 1re x 1.31 a 2.12, y 0.92 a 3.73
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
"""


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info != "" and not cond else ""))


def projet(L=8.0, h=0.5, largeur=0.5, plots=None, **k):
    """La dalle type du cahier (L × L × h, châssis 6 × 6 m centré, appuis 0,5 × 0,5 m, maille 0,20 m, 3 jeux), modifiée par
    ``k`` (dalle, grue, armatures… fusionnés)."""
    el = PR.nouveau()
    el["reactions"] = [{"nom": "", "P": list(j)} for j in JEUX]
    el["dalle"]["Lx"] = el["dalle"]["Ly"] = L
    el["dalle"]["h"] = h
    el["bandes"]["largeur"] = largeur
    if plots:
        el["grue"]["plots"] = dict(plots)
    for cle, v in k.items():
        el[cle].update(v)
    return PR.lire(el)


_EF = {}


def efforts(el, d_v=None):
    """(modèle, enveloppe des efforts) — d_v = h − 0,06 m (0,44 m pour h = 0,50 : celui de T14 et T19)."""
    cle = repr(el)
    if cle not in _EF:
        m = PR.modele(el)
        _EF[cle] = (m, EF.enveloppe(m, V.par_k(m, list(KS), (0.1, 0.1)), d_v if d_v is not None else el["dalle"]["h"] - 0.06))
    return _EF[cle]


def f2(v):
    return f"{round(v, 6) + 0.0:.2f}"


def lignes_t19(nom, pa):
    """L'écriture de l'annexe A (T19) d'un principe."""
    out = [f"T19 {nom} :"]
    for cle in AR.CLES:
        rb = [r for r in pa["bandes"] if r["cle"] == cle]
        if rb:
            r = rb[0]
            tb = (f"{len(rb)} renfort(s) de bande ; 1er {r['bande']} x {f2(r['x0'])} a {f2(r['x1'])}, "
                  f"y {f2(r['y0'])} a {f2(r['y1'])}, m_dim {r['m_dim']:.1f}")
        else:
            tb = "bandes : treillis suffisant"
        rl = pa["locaux"][cle]
        if rl:
            r = rl[0]
            tl = (f"{len(rl)} zone(s) rouge(s), {sum(x['aire'] for x in rl):.2f} m2, max {max(x['m_dim'] for x in rl):.1f} ; "
                  f"1re x {f2(r['x0'])} a {f2(r['x1'])}, y {f2(r['y0'])} a {f2(r['y1'])}")
        else:
            tl = "hors bandes : treillis suffisant"
        out.append(f"    {cle} : {tb} | {tl}")
    return out


CAS_T19 = [("dalle type 8 x 8 x 0,50, bandes 0,50", dict(), None),
           ("dalle type, bandes 1,00", dict(largeur=1.0), None),
           ("dalle type, plots coules en place, bandes 0,50", dict(plots=COULE), None),
           ("dalle 10,70 x 10,70 x 0,50, bandes 0,50", dict(L=10.7), None),
           ("dalle 10,70, bandes 1,00", dict(L=10.7, largeur=1.0), None),
           ("dalle 10,70, plots prefabriques, bandes 0,50", dict(L=10.7, plots=PREF), None),
           ("dalle 10,70, plots coules en place, bandes 0,50", dict(L=10.7, plots=COULE), None),
           ("dalle 10,70, bandes 0,50, SCIA = module x 1,10 en bas", dict(L=10.7), {"inf_x": 1.1, "inf_y": 1.1})]


def _geo(pa_bandes, pa_locaux):
    """La géométrie d'un principe, arrondie au µm (pour comparer deux principes)."""
    b = [(r["bande"], r["cle"], round(r["x0"], 6), round(r["x1"], 6), round(r["y0"], 6), round(r["y1"], 6), round(r["m_dim"], 6)) for r in pa_bandes]
    loc = {c: [(round(r["x0"], 6), round(r["x1"], 6), round(r["y0"], 6), round(r["y1"], 6), round(r["aire"], 6), round(r["m_dim"], 6))
               for r in lst] for c, lst in pa_locaux.items()}
    return b, loc


# ------------------------------------------------------------------ les contrôles RAPIDES (rejoués par les mutations)
def rapides():
    print("== 1. T19, A1 : le principe de l'annexe A (mRd = 97,3 kNm/m, ancrage 0,80 m)")
    attendu = T19_ATTENDU.strip("\n").split("\n")
    blocs = [attendu[i * 5:(i + 1) * 5] for i in range(8)]
    for (nom, k, ech), bloc in zip(CAS_T19, blocs):
        m, ef = efforts(projet(**k))
        obtenu = lignes_t19(nom, AR.principe(ef, m["contour"], MRD_ESSAI, 0.80, ech))
        chk(f"T19 {nom} : renforts des bandes et zones rouges de l'annexe A", obtenu == bloc,
            [(a, b) for a, b in zip(obtenu, bloc) if a != b][:2])
    # A1 : le module se sert du principe tel quel (son mRd, son ancrage 50 Ø, son Ø)
    el = projet()
    m, ef = efforts(el)
    fer = AR.calculer(el, m, ef)
    res = fer["res"]
    pa = AR.principe(ef, m["contour"], res["mRd"], 50 * 12 / 1000, None, fer["mRd_bandes"])
    rb = [r for r in fer["renforts"] if r["type"] == "bande"]
    rl = {c: [r for r in fer["renforts"] if r["type"] == "local" and r["cle"] == c] for c in AR.CLES}
    chk("A1 le principe du module (Ø12 partout, ancrage 50 Ø = 0,60 m, mRd du moteur Dalles) est celui de l'annexe : mêmes renforts "
        "de bande, mêmes renforts locaux, dans l'ordre des repères (le bas puis le haut, les bandes puis les locaux, x puis y)",
        set(fer["diametres"].values()) == {12} and _geo(rb, rl) == _geo(sorted(pa["bandes"], key=lambda r: (r["cle"][:3] != "inf", r["cle"])), pa["locaux"])
        and [r["rep"] for r in fer["renforts"]] == [f"R{i}" for i in range(1, len(fer["renforts"]) + 1)]
        and all(r["ancrage"] == 0.6 for r in fer["renforts"]), (fer["diametres"], len(rb)))
    print("== 2. le moteur Dalles : mRd, A3 (les barres), A4 (la hauteur), le Ø, châssis tourné")
    s1 = AR.section(el, {"inf_x": 1.0})["grandeurs"]
    s50 = AR.section(el, {"inf_x": 50.0})["grandeurs"]
    s_r = AR.section(el, {c: res["mRd"][c] for c in AR.CLES})["grandeurs"]
    chk("mRd du treillis Ø12 e 15 (enrobage 5 cm, h = 50 cm) par le moteur Dalles : 97,3 kNm/m en x (d = 43,0 cm), 95,0 en y (d = 42,0 cm : "
        "la couche y sur la couche x), les deux faces ; A_s = 754 mm²/m ; A_s,req est linéaire en m (50 kNm/m : 50 × celui de 1 kNm/m), et à "
        "m = mRd le moteur demande exactement le treillis",
        f2(res["mRd"]["inf_x"]) == "97.26" and f2(res["mRd"]["inf_y"]) == "95.00" and res["mRd"]["sup_x"] == res["mRd"]["inf_x"]
        and (res["d"]["inf_x"], res["d"]["inf_y"]) == (43.0, 42.0) and round(res["As"]["inf_x"]) == 754
        and abs(s50["As_req_inf_x"] - 50 * s1["As_req_inf_x"]) < 1e-9 * s50["As_req_inf_x"]
        and all(abs(s_r[f"As_req_{c}"] - res["As"][c]) < 1e-6 for c in AR.CLES), (res["mRd"], res["d"]))
    # A3 : la dalle 10,70 sans plot, Ø12 e 15
    e7 = projet(L=10.7)
    m7, ef7 = efforts(e7)
    fer7 = AR.calculer(e7, m7, ef7)
    faux = []
    for r in fer7["renforts"]:
        g = AR.section(e7, {r["cle_section"]: r["m_dim"]})["grandeurs"]
        besoin = max(g[f"As_req_{r['cle_section']}"], g[f"As_min_{r['cle_section']}"])
        n = math.ceil((besoin - 1000 * math.pi * 12 ** 2 / 4 / 150) * r["largeur"] / (math.pi * r["d"] ** 2 / 4) - 1e-9)
        if n != r["n"] or abs(r["besoin"] - besoin) > 1e-9:
            faux.append((r["rep"], r["n"], n))
    from api_web import saf_armatures as SAW
    sc7 = SC.rapports(e7, ef7, None)
    carte = SAW.carte(e7, fer7, ef7, sc7).replace(" ", " ")
    chk(f"A3 dalle 10,70 × 10,70 × 0,50 sans plot, Ø12 e 15 : pour chacun des {len(fer7['renforts'])} renforts, n = ⌈(A_s(m_dim) − 754) × "
        "largeur ÷ aire d'une barre⌉ avec A_s() du moteur Dalles (max(A_s,req ; A_s,min)) ; la carte ARMATURES de l'écran écrit ces barres",
        len(fer7["renforts"]) > 8 and not faux and all(f"{r['n']} Ø{r['d']}" in carte for r in fer7["renforts"] if r["n"]), faux[:4])
    chk("A3 un renfort de bande de la dalle 10,70 : sur toute la largeur de la bande (0,50 m), de la limite + l'ancrage, m_dim = 186,7 kNm/m "
        "(T19) ; n barres de Ø12",
        (lambda r: r["largeur"] == 0.5 and f"{r['m_dim']:.1f}" == "186.7" and r["n"] >= 2 and r["d"] == 12)(fer7["renforts"][0]),
        fer7["renforts"][0])
    # le besoin est max(A_s,req ; A_s,min) du moteur : une dalle de 1,00 m sous un treillis Ø8 e 25 en bas — A_s,min gouverne
    em = projet(h=1.0, armatures={"inf": {"x": {"d": 8, "e": 250}, "y": {"d": 8, "e": 250}, "xy": False}})
    ferm = AR.calculer(em, *efforts(em))
    gouv = [r for r in ferm["renforts"] if r["As_min"] > r["As_req"] + 1.0]
    chk("le besoin d'un renfort est max(A_s,req ; A_s,min) du moteur Dalles : h = 1,00 m, treillis du bas Ø8 e 25 — A_s,min gouverne des "
        "renforts du bas (le besoin vaut A_s,min, pas A_s,req)",
        gouv and all(abs(r["besoin"] - max(r["As_req"], r["As_min"])) < 1e-9 for r in ferm["renforts"])
        and all(abs(r["besoin"] - r["As_min"]) < 1e-9 for r in gouv), [(r["rep"], round(r["As_req"]), round(r["As_min"])) for r in ferm["renforts"][:4]])
    # l'étiquetage des zones rouges : 4-connexité (deux nœuds qui ne se touchent que par un coin font deux zones), numérotés
    # dans l'ordre du balayage ligne par ligne — scipy.ndimage.label
    lab, nb = AR.etiqueter(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], bool))
    chk("les zones rouges se regroupent en 4-connexité (gauche, droite, haut, bas ; un coin ne relie pas), numérotées dans l'ordre du "
        "balayage : quatre zones", nb == 4 and lab.tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3], [4, 4, 0]], (nb, lab.tolist()))
    # A4 : la hauteur de béton
    e2 = projet(h=0.20)
    m2, ef2 = efforts(e2)
    fer2 = AR.calculer(e2, m2, ef2)
    textes = [t for t, _ in fer2["alertes"]]
    chk("A4 h = 0,20 m sur la dalle type : le moteur Dalles dit que d ne suffit pas pour m_dim (d_min > d) — pastille rouge « Hauteur de "
        "béton insuffisante : épaissir la dalle », la carte non conforme",
        fer2["hauteur"]["etat"] == "nok" and fer2["hauteur"]["hu_min"] > fer2["hauteur"]["d"] and fer2["etat"] == "nok"
        and ("Hauteur de béton insuffisante : épaissir la dalle", "r") in fer2["alertes"], (fer2["hauteur"], textes))
    tous = []
    for k in (dict(), dict(L=10.7), dict(plots=COULE), dict(L=10.7, plots=PREF), dict(h=0.20), dict(h=0.30),
              dict(armatures={"inf": {"x": {"d": 8, "e": 250}, "y": {"d": 8, "e": 250}, "xy": False}})):
        e_ = projet(**k)
        tous += [(t, n) for t, n in AR.calculer(e_, *efforts(e_))["alertes"]]
    epaissir = {t for t, _ in tous if "paissir" in t}
    chk("A4 aucune autre situation ne propose d'épaissir la dalle : sur sept dalles (h = 0,20 à 0,50, plots, treillis Ø8 e 25…), « épaissir » "
        "n'est dit que par la hauteur de béton et l'effort tranchant",
        epaissir <= {"Hauteur de béton insuffisante : épaissir la dalle", "Effort tranchant : τ > τ_adm — épaissir la dalle"}
        and fer["hauteur"]["etat"] == "ok" and not any("paissir" in t for t, _ in fer["alertes"]), sorted({t for t, _ in tous}))
    chk("la hauteur de béton de la dalle type : d_min = h_u,min (le plus exigeant des renforts et du treillis : un renfort en y) ≤ d = h − y_G "
        "= 42,0 cm, taux 61 %",
        fer["hauteur"]["etat"] == "ok" and abs(fer["hauteur"]["taux"] - fer["hauteur"]["hu_min"] / fer["hauteur"]["d"]) < 1e-12
        and round(fer["hauteur"]["taux"] * 100) == 61 and fer["hauteur"]["d"] == 42.0
        and next(r for r in fer["renforts"] if r["rep"] == fer["hauteur"]["rep"])["sens"] == "y", fer["hauteur"])
    # l'effort tranchant : V à d ÷ largeur au moteur Dalles (τ = v ÷ (0,75 × h)), V SCIA direct
    t1 = fer["tranchant"]["BX1"]
    bx1 = next(b for b in ef["bandes"] if b["nom"] == "BX1")
    fer_v = AR.calculer(el, m, ef, SC.rapports(dict(el, scia=dict(el["scia"], bx_v=120.0)), ef))
    chk("l'effort tranchant de BX1 : V à d = 94,0 kN (T14) ÷ 0,50 m → τ = 188,0 ÷ (0,75 × 500) = 0,50 N/mm² ≤ τ_adm,I du moteur ; V SCIA "
        "saisi (120 kN) va directement au moteur",
        abs(t1["V"] - bx1["V_d"]) < 1e-12 and f"{t1['V']:.1f}" == "94.0" and abs(t1["tau"] - t1["v"] / 375.0) < 1e-6 and t1["etat"] == "ok"
        and fer_v["tranchant"]["BX1"]["V"] == 120.0 and fer_v["tranchant"]["BX1"]["scia"] and abs(fer_v["tranchant"]["BX1"]["tau"] - 0.64) < 1e-6,
        (t1, fer_v["tranchant"]["BX1"]))
    # le Ø des renforts : automatique (le plus léger, à poids égal le plus gros), imposé
    par_d = {d: AR.calculer(projet(armatures={"renfort": d}), m, ef) for d in AR.DIAMETRES_RENFORT}
    choisi_ok = []
    for cle in AR.CLES:
        notes = {d: (any(r["serre"] for r in f["renforts"] if r["cle"] == cle), round(sum(r["poids"] for r in f["renforts"] if r["cle"] == cle), 3), -d)
                 for d, f in par_d.items()}
        choisi_ok.append(fer["diametres"][cle] == min(notes, key=notes.get))
    chk("le Ø automatique des renforts, face par face et direction par direction : le plus léger (Σ n × aire × longueur) de Ø12, Ø16, Ø20 — à "
        "poids égal le plus gros, un espacement trop serré en dernier ; imposé (Ø16) : tous les renforts en Ø16, ancrage 50 × 16 = 0,80 m",
        all(choisi_ok) and {r["d"] for r in par_d[16]["renforts"]} == {16} and {r["ancrage"] for r in par_d[16]["renforts"]} == {0.8},
        (fer["diametres"], choisi_ok))
    # châssis tourné : une bande prend le plus faible des deux directions de la face
    et = projet(L=12.0, grue={"beta": 30.0})
    mt, eft = efforts(et)
    fert = AR.calculer(et, mt, eft)
    chk("châssis tourné de 30° (dalle 12 × 12) : chaque bande prend le m_Rd le plus faible des deux directions de sa face (95,0 kNm/m en bas "
        "comme en haut) ; châssis droit : sa direction",
        fert["tourne"] and fert["mRd_bandes"]["inf_x"] == fert["mRd_bandes"]["inf_y"] == min(fert["res"]["mRd"]["inf_x"], fert["res"]["mRd"]["inf_y"])
        and not fer["tourne"] and fer["mRd_bandes"] == fer["res"]["mRd"]
        and all(r["cle_section"] in ("inf_y", "sup_y") for r in fert["renforts"] if r["type"] == "bande"), fert["mRd_bandes"])
    # le U de fermeture, les plots coulés, hors du moteur Dalles
    ec = projet(plots=COULE)
    ferc = AR.calculer(ec, *efforts(ec))
    hd = [AR.calculer(e_, *efforts(e_)) for e_ in (projet(dalle={"beton": "C45/55"}), projet(h=1.05))]
    chk("le U de fermeture (« Ø12 e 15 cm sur tout le pourtour (32,0 m) ») ; plots coulés en place : « Non vérifiés » nomme leur ferraillage, "
        "aucun renfort de bande dans un plot ; béton C45/55 ou h = 1,05 m : hors du moteur Dalles, les armatures « non vérifiées » (à vérifier, "
        "jamais conformes)",
        fer["u"] == "U de fermeture : Ø12 e 15 cm sur tout le pourtour (32,0 m)" and ferc["non_verifies"] == ["le poinçonnement", "le ferraillage des plots coulés en place"]
        and fer["non_verifies"] == ["le poinçonnement"] and ferc["plot_coule"]
        and all(h_["hors_domaine"] and h_["etat"] == "v" and not h_["renforts"] for h_ in hd)
        and "C45/55" in hd[0]["hors_domaine"] and "1,00 m" in hd[1]["hors_domaine"], (fer["u"], ferc["non_verifies"], [h_["hors_domaine"] for h_ in hd]))
    print("== 3. SCIA : E1 à E4 côté moteur")
    mo = SC.module(ef)
    sym = [SC.symetrique(projet(**k)) for k in (dict(), dict(grue={"ey": 0.5}), dict(grue={"beta": 90.0}), dict(grue={"beta": 45.0}),
                                                  dict(dalle={"Ly": 10.0}))]
    chk("E1 les valeurs grises sont celles du module (T14) : M+ le plus fort des bandes BX (71,6 kN·m) et BY, V à d (94,0 kN), m le plus "
        "négatif de la carte (−140,8 kN·m/m) ; « x = y » cochée d'office pour la dalle type (carrée, châssis carré centré), β = 90° ; pas "
        "pour une grue décalée, β = 45°, une dalle 8 × 10",
        (f"{mo['bx_m']:.1f}", f"{mo['bx_v']:.1f}", f"{mo['mx']:.1f}") == ("71.6", "94.0", "-140.8") and f2(mo["by_m"]) == f2(mo["bx_m"])
        and mo["bx_m"] == max(b["M_plus"] for b in ef["bandes"] if b["nom"].startswith("BX")) and mo["mx"] == ef["extremes"]["sup_x"]
        and sym == [True, False, True, False, False], (mo, sym))
    r0 = SC.rapports(el, ef, None)
    chk("E2 rien de saisi : r = 1 partout (les moments du module gardés), « à saisir », 3 champs à taper (x = y)",
        r0["echelle"] == {"inf_x": 1.0, "inf_y": 1.0, "sup_x": 1.0, "sup_y": 1.0} and not r0["saisi"] and not r0["complet"]
        and all(li["r"] is None for li in r0["lignes"]) and r0["a_taper"] == ["bx_m", "bx_v", "mx"], r0["echelle"])
    m7_ = SC.module(ef7)
    r11 = SC.rapports(dict(e7, scia=dict(e7["scia"], bx_m=m7_["bx_m"] * 1.10)), ef7)
    fer11 = AR.calculer(e7, m7, ef7, r11)
    pa11 = AR.principe(ef7, m7["contour"], fer11["res"]["mRd"], 0.60, {"inf_x": 1.1, "inf_y": 1.1}, fer11["mRd_bandes"])
    rb11 = [r for r in fer11["renforts"] if r["type"] == "bande"]
    rl11 = {c: [r for r in fer11["renforts"] if r["type"] == "local" and r["cle"] == c] for c in AR.CLES}
    chk("E3 dalle 10,70, bandes 0,50 : M+ SCIA = 1,10 × la valeur grise → r = 1,10, l'échelle du bas en x ET en y (x = y), le haut à 1 ; le "
        "principe du module est celui de T19 « SCIA = module × 1,10 en bas » (avec son mRd et son ancrage), et il a plus de barres",
        abs(r11["echelle"]["inf_x"] - 1.1) < 1e-12 and abs(r11["echelle"]["inf_y"] - 1.1) < 1e-9 and r11["echelle"]["sup_x"] == 1.0
        and _geo(rb11, rl11) == _geo(sorted(pa11["bandes"], key=lambda r: (r["cle"][:3] != "inf", r["cle"])), pa11["locaux"])
        and sum(r["n"] for r in fer11["renforts"]) > sum(r["n"] for r in fer7["renforts"]), r11["echelle"])
    def alerte(f):
        r_ = SC.rapports(dict(el, scia=dict(el["scia"], bx_m=mo["bx_m"] * f)), ef)
        return r_["lignes"][0]["alerte"], r_["alertes"]
    a130, a115, a084, a086 = alerte(1.30), alerte(1.15), alerte(0.84), alerte(0.86)
    chk("E4 r hors de 0,85 à 1,15 : l'alerte (« Écart module / SCIA : +30,0 % sur bandes BX : M+ le plus fort — vérifier le modèle SCIA (k, "
        "bandes, unités) ») ; r = 1,15 et 0,86 : pas d'alerte ; 0,84 : « −16,0 % »",
        a130[0] and a130[1] == ["Écart module / SCIA : +30,0 % sur bandes BX : M+ le plus fort — vérifier le modèle SCIA (k, bandes, unités)"]
        and not a115[0] and not a086[0] and a084[0] and "-16,0 %" in a084[1][0], (a130, a084))
    rs = SC.rapports(dict(el, scia=dict(el["scia"], sym=False, bx_m=80.0, by_m=60.0, mx=-150.0)), ef)
    rsig = SC.rapports(dict(el, scia=dict(el["scia"], sigma={"10": 100.0})), ef, [{"k": 10.0, "sigma_max": 110.3}])
    chk("« x = y » décochée : chaque direction a son r (BY : 60 ÷ 71,6) ; m SCIA négatif → r du haut ; σ max SCIA d'un k : comparé "
        "seulement (r = 100 ÷ 110,3), l'échelle des moments ne bouge pas",
        abs(rs["echelle"]["inf_y"] - 60.0 / mo["by_m"]) < 1e-12 and abs(rs["echelle"]["inf_x"] - 80.0 / mo["bx_m"]) < 1e-12
        and abs(rs["echelle"]["sup_x"] - 150.0 / -mo["mx"]) < 1e-12 and rs["echelle"]["sup_y"] == 1.0 and len(rs["a_taper"]) == 6
        and abs(next(s for s in rsig["sigma"] if s["k"] == 10.0)["r"] - 100 / 110.3) < 1e-12
        and rsig["echelle"] == {"inf_x": 1.0, "inf_y": 1.0, "sup_x": 1.0, "sup_y": 1.0}, (rs["echelle"], rsig["sigma"]))


def complets():
    print("== 4. la parité avec le calcul de référence (tests/fondation_grue_ref.py : principe_armatures, scipy.ndimage)")
    import fondation_grue_ref as REF
    cas = [("dalle en L 12 × 12, châssis tourné de 30°", projet(dalle={"forme": "contour", "sommets": [list(p) for p in L_T7]}, grue={"beta": 30.0}), (60.0, 80.0, 90.0, 70.0)),
           ("dalle 10 × 9, grue décalée (0,6 ; −0,4)", projet(L=10.0, dalle={"Ly": 9.0}, grue={"ex": 0.6, "ey": -0.4}), (140.0, 120.0, 100.0, 110.0)),
           ("dalle type, bandes 1,00, plots coulés", projet(largeur=1.0, plots=COULE), (75.0, 85.0, 65.0, 95.0)),
           ("dalle 10,70, plots préfabriqués", projet(L=10.7, plots=PREF), (97.3, 95.0, 97.3, 95.0))]
    for nom, el, mr in cas:
        m, ef = efforts(el)
        mRd = dict(zip(AR.CLES, mr))
        for anc, ech in ((0.80, None), (0.60, {"inf_x": 1.2, "sup_y": 0.9})):
            a = AR.principe(ef, m["contour"], mRd, anc, ech)
            b = REF.principe_armatures(ef, m["contour"], mRd, anc, ech)
            chk(f"{nom}, mRd {mr}, ancrage {anc} m{' , échelle SCIA' if ech else ''} : le principe du module = celui de la référence "
                f"({len(a['bandes'])} renforts de bande, {sum(len(v) for v in a['locaux'].values())} renforts locaux)",
                _geo(a["bandes"], a["locaux"]) == _geo(b["bandes"], b["locaux"]), (len(a["bandes"]), len(b["bandes"])))
    rng = np.random.default_rng(5)
    bon = True
    for _ in range(40):
        msk = rng.random((int(rng.integers(3, 15)), int(rng.integers(3, 15)))) < rng.uniform(0.2, 0.7)
        from scipy import ndimage
        la, na = AR.etiqueter(msk)
        lb, nb = ndimage.label(msk)
        bon &= na == nb and bool((la == lb).all())
    chk("l'étiquetage des zones rouges (numpy seul) est celui de scipy.ndimage.label : 40 grilles tirées au hasard, mêmes groupes, mêmes "
        "numéros", bon)
    print("== 5. A2 : changer un treillis ou la largeur des bandes ne refait pas le calcul du sol")
    from api_web import saf as M
    appels = []
    vrai = M.V.par_k

    def compte(*a, **k):
        appels.append(1)
        return vrai(*a, **k)
    M.V.par_k = compte
    try:
        e = M._etat_type()
        f = PR.lire(e["fondations"][0])
        c0 = M.calculer(f)
        z0 = M.armatures(f, c0)
        n0 = len(appels)
        f16 = PR.lire(dict(e["fondations"][0], armatures=dict(e["fondations"][0]["armatures"], inf={"x": {"d": 16, "e": 150}, "y": {"d": 16, "e": 150}, "xy": False})))
        t0 = time.perf_counter()
        z16 = M.armatures(f16, M.calculer(f16))
        dt = time.perf_counter() - t0
        f100 = PR.lire(dict(e["fondations"][0], bandes={"largeur": 1.0}))
        z100 = M.armatures(f100, M.calculer(f100))
    finally:
        M.V.par_k = vrai
    chk(f"A2 un treillis changé (bas Ø16 e 15) : les zones rouges et les renforts du bas suivent ({z0['zones']['inf_x']['aire']:.2f} → "
        f"{z16['zones']['inf_x']['aire']:.2f} m²), en {dt:.2f} s ; la largeur des bandes (1,00 m) : les renforts des bandes suivent ; le calcul du "
        "sol n'est PAS refait (aucun appel de verification.par_k après le premier)",
        n0 == 1 and len(appels) == 1 and z16["zones"]["inf_x"]["aire"] < z0["zones"]["inf_x"]["aire"] and dt < 1.0
        and {r["largeur"] for r in z100["renforts"] if r["type"] == "bande"} == {1.0}, (n0, len(appels)))


MUTATIONS = [
    ("armatures.py", '''            dep = (m > mRd_b[cle] + 1e-9) & ~bd["dans_plot"]''', '''            dep = (m > mRd_b[cle] + 1e-9)''',
     "un renfort de bande posé dans un plot coulé"),
    ("armatures.py", '''                s0, s1 = max(0.0, float(s[i0]) - ancrage), min(L, float(s[i1]) + ancrage)''',
     '''                s0, s1 = max(0.0, float(s[i0])), min(L, float(s[i1]))''', "un renfort de bande sans ancrage"),
    ("armatures.py", '''        rouge = (m > mRd[cle] + 1e-9) & g["actif"] & ~dans_bande[cle[-1]]''', '''        rouge = (m > mRd[cle] + 1e-9) & g["actif"]''',
     "une zone rouge dans une bande de sa direction"),
    ("armatures.py", '''        rouge = (m > mRd[cle] + 1e-9) & g["actif"] & ~dans_bande[cle[-1]]''',
     '''        rouge = (m > mRd[cle] + 1e-9) & g["actif"] & ~dans_bande["x"] & ~dans_bande["y"]''', "une bande qui couvre aussi les barres de l'autre direction"),
    ("armatures.py", '''                    for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):''',
     '''                    for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1), (j + 1, i + 1), (j - 1, i - 1), (j + 1, i - 1), (j - 1, i + 1)):''',
     "des zones rouges réunies en 8-connexité (scipy : 4)"),
    ("armatures.py", '''            r[lo] = max(LO, r[lo] - ancrage)''', '''            r[lo] = r[lo] - ancrage''', "un renfort local qui sort de la dalle"),
    ("armatures.py", '''    return int(math.ceil(renfort_pm * largeur / aire_barre(d_mm) - 1e-9))''', '''    return int(round(renfort_pm * largeur / aire_barre(d_mm)))''',
     "le nombre de barres arrondi au plus près"),
    ("armatures.py", '''    renfort = max(0.0, besoin - As_tr)''', '''    renfort = besoin''', "un renfort qui ne compte pas le treillis"),
    ("armatures.py", '''    besoin = g[f"besoin_{cle_sec}"]''', '''    besoin = g[f"As_req_{cle_sec}"]''', "le besoin sans A_s,min"),
    ("armatures.py", '''    return {"mRd": {f: g[f"As_{f}"] / g[f"As_req_{f}"] for f in CLES}''', '''    return {"mRd": {f: 97.3 for f in CLES}''',
     "un mRd forfaitaire au lieu de celui du moteur Dalles"),
    ("armatures.py", '''"dist": round(POS.sur_couche(dessus_x, ty["d"]), 6)}]''', '''"dist": None}]''', "la couche y posée sur l'enrobage (d trop fort)"),
    ("armatures.py", '''        m = min(mRd[f"{face}_x"], mRd[f"{face}_y"])''', '''        m = max(mRd[f"{face}_x"], mRd[f"{face}_y"])''',
     "une bande de châssis tourné qui prend la direction la plus forte"),
    ("armatures.py", '''            note = (serre, round(poids, 3), -d)''', '''            note = (serre, -round(poids, 3), -d)''', "le Ø le plus lourd"),
    ("armatures.py", '''               "etat": "ok" if h_min <= h_cm + 1e-9 else "nok", "rep": rep_h}''', '''               "etat": "ok", "rep": rep_h}''',
     "une hauteur de béton insuffisante tue"),
    ("armatures.py", '''            V = abs(V_s)''', '''            pass''', "V SCIA ignoré"),
    ("armatures.py", '''    if pl and pl["type"] == "coule":
        nv.append''', '''    if False:
        nv.append''', "le ferraillage des plots coulés dit vérifié"),
    ("armatures.py", '''    if h_moteur(el) > H_MAX_MOTEUR:''', '''    if False:''', "une dalle de plus de 1,00 m passée au moteur Dalles"),
    ("armatures.py", '''        v = V / bd["largeur"]''', '''        v = V''', "l'effort tranchant de la bande entière pris pour un mètre"),
    ("scia.py", '''    echelle = {"inf_x": r["bx_m"] or 1.0, "inf_y": r["by_m"] or 1.0, "sup_x": r["mx"] or 1.0, "sup_y": r["my"] or 1.0}''',
     '''    echelle = {"inf_x": r["bx_m"] or 1.0, "inf_y": r["by_m"] or 1.0, "sup_x": r["bx_m"] or 1.0, "sup_y": r["by_m"] or 1.0}''',
     "le haut recalé sur les bandes au lieu de la carte"),
    ("scia.py", '''                       "alerte": e is not None and e > ECART + 1e-12})''', '''                       "alerte": e is not None and e > 0.5})''',
     "l'écart de 15 % toléré jusqu'à 50 %"),
    ("scia.py", '''        for y, x in JUMELLES.items():
            s[y] = s[x]''', '''        for y, x in JUMELLES.items():
            pass''', "« x = y » qui ne recopie pas la direction y"),
    ("scia.py", '''    if abs(ch["a"] - ch["b"]) > 1e-9 or abs(g["ex"]) > 1e-9 or abs(g["ey"]) > 1e-9:''',
     '''    if abs(ch["a"] - ch["b"]) > 1e-9 or abs(g["ex"]) > 1e-9:''', "une grue décalée en y dite symétrique"),
    ("scia.py", '''        r = s / m if (s is not None and m not in (None, 0.0)) else None''', '''        r = m / s if (s is not None and m not in (None, 0.0)) else None''',
     "r = module ÷ SCIA"),
]
TEMOIN = ("armatures.py", "CLES = (", "CLES = (")


def muter(fichier, ancien, nouveau):
    """Les contrôles rapides rejoués sur une copie du moteur où ``ancien`` est remplacé : (texte trouvé, code de sortie)."""
    with tempfile.TemporaryDirectory() as tmp:
        for nom in os.listdir(RACINE):
            if nom in (".git", ".temoin_tests", "__pycache__", "node_modules", "saf"):
                continue
            os.symlink(os.path.join(RACINE, nom), os.path.join(tmp, nom))
        shutil.copytree(os.path.join(RACINE, "saf"), os.path.join(tmp, "saf"), ignore=shutil.ignore_patterns("__pycache__"))
        cible = os.path.join(tmp, "saf", "fondation_grue", fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rapide", "--racine", tmp], capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    t0 = time.time()
    rapides()
    print(f"   (contrôles rapides : {time.time() - t0:.1f} s)")
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    complets()
    print("== 6. mutations")
    trouve, code = muter(*TEMOIN)
    chk("témoin : le moteur recopié sans rien changer passe les contrôles rapides", trouve and code == 0, code)
    for fichier, ancien, nouveau, nom in MUTATIONS:
        trouve, code = muter(fichier, ancien, nouveau)
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print("== 7. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('scipy', 'streamlit', 'pandas', 'matplotlib', 'reportlab', 'PIL', 'fitz', 'openpyxl'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from saf.fondation_grue import projet as PR, verification as V, efforts as EF, armatures as AR, scia as SC\n"
            "el = PR.nouveau(); el['reactions'] = [{'nom': '', 'P': [350, 440, 440, 350]}, {'nom': '', 'P': [150, 150, 550, 550]}, "
            "{'nom': '', 'P': [135, 100, 800, 135]}]\n"
            "el = PR.lire(el); m = PR.modele(el)\n"
            "ef = EF.enveloppe(m, V.par_k(m, [2.0, 5.0, 10.0], (0.1, 0.1)), 0.44)\n"
            "fer = AR.calculer(el, m, ef, SC.rapports(el, ef))\n"
            "print(len(fer['renforts']), round(fer['res']['mRd']['inf_x'], 1))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le principe d'armatures se charge et calcule (dalle type : 28 renforts, mRd 97,3 kNm/m) avec numpy seul : scipy, streamlit, pandas, "
        "matplotlib, reportlab, openpyxl refusés", p.returncode == 0 and p.stdout.strip() == "28 97.3", (p.stdout.strip(), p.stderr[-300:]))
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
