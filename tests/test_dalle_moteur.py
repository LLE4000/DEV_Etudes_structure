# -*- coding: utf-8 -*-
"""Moteur « Dalle béton armé » (beton/dalle) : la PARITÉ avec l'écran Streamlit modules/dalle.py v2.2, qui reste la
référence, et la copie tenue de la base des treillis (beton/treillis.py ↔ modules/treillis.py). Chaque contrôle rougit
si l'on retire ce qu'il protège.

  1. Données : classes de béton, défauts de l'écran LUS dans son état (_ensure_global_defaults, _init_dalles_if_needed,
     _ensure_defaults_for_dalle, _add_section, _add_couche, _next_section_name), listes ; la base des treillis
     (beton/treillis.py) rend exactement ce que rend modules/treillis.py, fonction par fonction.
  2. À la main : les cas des tests de l'écran (tests/test_dalle_bidir.py) recalculés ici en N et en mm.
  3. Parité (plus de 60 notes, plusieurs dalles et sections) : chaque note est chargée dans l'état de l'écran piloté
     SANS navigateur, puis on compare au moteur :
       a) TOUTES les clés de _dimensionnement_compute_states (direction par direction) et ses états ;
       b) chaque couche (_couche_as_per_m, _dist_couche_eff, _auto_dist_couche, _couche_label, _couche_diam_mm) et
          chaque face (_layers_geometry, avec et sans le centre de gravité saisi) ;
       c) ce que l'ÉCRAN affiche : render_dimensionnement_right exécuté avec un faux « st » qui enregistre, et un espion
          sur open_bloc_left_right — titre, texte de droite, état et POURCENTAGE EXACT (flottant) de chaque bloc ; les
          lignes de la hauteur, du tranchant, Aₛ,req / Aₛ,min / Aₛ,max / Aₛ fourni ET leurs infobulles (formules et
          valeurs) ; l'icône de chaque section et de chaque dalle ;
       d) la NOTE PDF (modules/export_pdf_dalle._compute_section, seconde copie du calcul dans l'écran) ;
       e) le FICHIER : ce que « 💾 Enregistrer » écrit (_build_save_payload), relu par beton/dalle/fichier.py, redonne
          la note.
     Tolérance 1e-9 relative ; l'écart maximal et la part des comparaisons identiques au bit près sont imprimés.
  4. Couverture : chaque état, chaque motif, chaque terme de Aₛ,min, chaque type de couche, chaque famille qui
     gouverne la hauteur, les bornes exactes (≤ et non <)… sont effectivement atteints.
  5. Sélecteur de la nappe de base : chaque nappe proposée est repassée par l'écran.
  6. Fichiers de l'écran : migration v1 → v2, saisies que l'écran remplace en silence (relevées), refus.
  7. Format : JSON, normalisation (défauts, ValueError hors du domaine de l'écran), avertissements, formules
     recalculées depuis leurs seuls termes, grandeurs documentées.
  8. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, pandas,
     reportlab, numpy… sous Python 3.12 et 3.13 (celui de Pyodide) s'il est installé — résultats identiques.
  9. Mutations : 25 altérations du moteur (coefficients, bornes ≤ / <, règle de l'audit C-1, axe et centre de gravité
     saisis, enrobage nul, largeur de bande, direction principale…) ; chacune fait rougir la parité de la section 3.

    /usr/bin/python3.12 tests/test_dalle_moteur.py       (streamlit requis pour la parité, reportlab pour la note)
"""
import ast
import contextlib
import copy
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from beton import treillis as TRB  # noqa: E402
from beton.dalle import entree as E  # noqa: E402
from beton.dalle import fichier as FI  # noqa: E402
from beton.dalle import moteur as M  # noqa: E402

OK, KO = [], []


def lire(*chemin):
    f = os.path.join(RACINE, *chemin)
    if not os.path.exists(f):
        return ""
    with open(f, encoding="utf-8") as fh:
        return fh.read()


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


TOL = 1e-9
ECART = {"max": 0.0, "n": 0, "bit": 0}


def _nombre(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def egal(a, b):
    """1e-9 relatif pour deux nombres ; égalité stricte pour le reste (textes, booléens, None)."""
    if not (_nombre(a) and _nombre(b)):
        return a == b
    ECART["n"] += 1
    if a == b:
        ECART["bit"] += 1
        return True
    e = abs(a - b) / max(abs(a), abs(b))
    ECART["max"] = max(ECART["max"], e)
    return e <= TOL


def ecarts_json(a, b, chemin="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{chemin} : clés {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b)):
            ecarts_json(a[k], b[k], f"{chemin}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{chemin} : {len(a)} ≠ {len(b)} éléments")
        for i, (x, y) in enumerate(zip(a, b)):
            ecarts_json(x, y, f"{chemin}[{i}]", out)
    elif not egal(a, b):
        out.append(f"{chemin} : {a!r} ≠ {b!r}")
    return out


# ------------------------------------------------------------------ les cas, au format du moteur
def couche(t="treillis", des="10/10/100/100", d=12, esp=150, n=3, dist=None):
    return {"type": t, "treillis": des, "d": d, "esp": esp, "n": n, "dist": dist}


def tr(des, dist=None):
    return couche("treillis", des, dist=dist)


def ba(d, esp, dist=None):
    return couche("barres", d=d, esp=esp, dist=dist)


def nb(n, d, dist=None):
    return couche("n_barres", d=d, n=n, dist=dist)


def fa(*couches, yG=None):
    return {"couches": list(couches) or [tr("10/10/100/100")], "yG": yG}


def sec(nom, Mx_inf=0.0, Mx_sup=0.0, My_inf=0.0, My_sup=0.0, V=0.0, **faces):
    return {"nom": nom, "Mx_inf": Mx_inf, "Mx_sup": Mx_sup, "My_inf": My_inf, "My_sup": My_sup, "V": V,
            "faces": {f: faces.get(f) or fa() for f in E.FACES}}


def dal(nom, sections, **kw):
    d = {"nom": nom, "sections": sections}
    d.update(kw)
    return M.normaliser(d)


def note(*dalles, gamma_s=None, jeu=None):
    """Une NOTE de l'écran : ses dalles partagent γs et le jeu premier lit (paramètres globaux de l'écran)."""
    out = [copy.deepcopy(d) for d in dalles]
    gs = out[0]["gamma_s"] if gamma_s is None else gamma_s
    j = out[0]["jeu_premier_lit"] if jeu is None else jeu
    for d in out:
        d["gamma_s"], d["jeu_premier_lit"] = gs, j
    return [M.normaliser(d) for d in out]


CL = M.classes_beton()
BIDIR = sec("A", 28, 12, 35, 9, 60)          # le cas complet de tests/test_dalle_bidir.py
CAS = [  # (description, [dalles de la note])
    ("défauts de l'écran (nouvelle_dalle)", note(M.nouvelle_dalle())),
    ("bidirectionnelle complète : Mx 28/12, My 35/9, V 60 (test_dalle_bidir)", note(dal("D1", [BIDIR]))),
    ("même dalle, direction principale X", note(dal("D1X", [BIDIR], dir_principale="X"))),
    ("renfort « n barres » 3 Ø12 sur inf. Y (785 + 339 mm²/m)",
     note(dal("RN", [sec("A", 28, 12, 35, 9, 60, inf_y=fa(tr("10/10/100/100"), nb(3, 12)))]))),
    ("renfort posé à 8,0 cm (axe saisi) : y_G pondéré 4,86 cm",
     note(dal("RN8", [sec("A", 28, 12, 35, 9, 60, inf_y=fa(tr("10/10/100/100"), nb(3, 12, dist=8.0)),
                          sup_y=fa(tr("8/8/150/150")), sup_x=fa(ba(10, 200)))]))),
    ("audit C-1 : la famille inf. Y posée à 8,0 cm gouverne la hauteur malgré My,sup > My,inf",
     note(dal("C-1", [sec("A", 0, 0, 30, 32, inf_y=fa(tr("10/10/100/100", dist=8.0)))]))),
    ("barres Ø12/150 partout, C25/30, h = 25 cm",
     note(dal("B12", [sec("A", 40, 25, 55, 30, 80, **{f: fa(ba(12, 150)) for f in E.FACES})], h=25, beton="C25/30"))),
    ("base « n barres » 8 Ø10 en bas, barres Ø8/200 en haut",
     note(dal("NB", [sec("A", 20, 5, 22, 6, 30, inf_x=fa(nb(8, 10)), inf_y=fa(nb(8, 10)), sup_x=fa(ba(8, 200)),
                         sup_y=fa(ba(8, 200)))]))),
    ("quatre couches (le maximum) mêlant les trois types, deux axes saisis",
     note(dal("Q4", [sec("A", 45, 10, 60, 15, 70, inf_y=fa(tr("12/12/150/150"), ba(10, 200, dist=5.0), nb(4, 16),
                                                            tr("6/6/150/150", dist=6.5)))], h=25))),
]
for beton in CL:  # les cinq bétons : fctm, α_b, μ et τ_adm différents
    CAS.append((f"béton {beton}, V = 90 kN", note(dal(f"B-{beton}", [sec("A", 30, 15, 42, 18, 90)], beton=beton, h=22))))
CAS += [
    ("B400 (μ_a400)", note(dal("B400", [sec("A", 28, 12, 35, 9, 60)], fyk=400))),
    ("γs = 1,15", note(dal("GS115", [sec("A", 28, 12, 35, 9, 60)]), gamma_s=1.15)),
    ("γs aux bornes de l'écran : 1,0", note(dal("GS1", [sec("A", 30, 0, 30, 0, 40)]), gamma_s=1.0)),
    ("γs aux bornes de l'écran : 2,0", note(dal("GS2", [sec("A", 30, 0, 30, 0, 40)]), gamma_s=2.0)),
    ("jeu premier lit 0,5 cm", note(dal("J05", [BIDIR]), jeu=0.5)),
    ("jeu premier lit 1,0 cm, enrobage 2,5 cm", note(dal("J10", [BIDIR], enrobage=2.5), jeu=1.0)),
    ("enrobage nul : l'écran le lit 3,0 cm", note(dal("E0", [BIDIR], enrobage=0.0))),
    ("enrobage 5 cm", note(dal("E5", [BIDIR], enrobage=5.0))),
    ("enrobage 20 cm (borne) sur 20 cm : d ≤ 0 partout", note(dal("E20", [sec("A", 5, 5, 5, 5, 10)], enrobage=20.0))),
    ("largeur de bande minimale (20 cm) et « n barres »", note(dal("B20", [sec("A", 8, 0, 9, 0, 15, inf_y=fa(nb(2, 10)))], b=20))),
    ("largeur de bande maximale (500 cm), n barres et barres", note(dal("B500", [sec("A", 140, 60, 175, 45, 300,
                                                                                      inf_y=fa(tr("10/10/100/100"), nb(12, 12)),
                                                                                      sup_x=fa(ba(10, 125)))], b=500))),
    ("bande de 150 cm : As = As/m × 1,50", note(dal("B150", [sec("A", 40, 15, 52, 20, 80, inf_y=fa(nb(9, 12)))], b=150))),
    ("épaisseur minimale (5 cm) : hauteur insuffisante et d < 0", note(dal("H5", [sec("A", 2, 0, 3, 0, 5)], h=5))),
    ("épaisseur maximale (100 cm)", note(dal("H100", [sec("A", 300, 120, 450, 200, 600, inf_y=fa(ba(25, 100)), inf_x=fa(ba(20, 100)))],
                                             h=100))),
    ("τ au-delà de τ_adm,I (non conforme)", note(dal("TAU", [sec("A", 20, 0, 25, 0, 150)]))),
    ("acier insuffisant", note(dal("ASINS", [sec("A", 45, 0, 60, 0, 40)]))),
    ("acier excessif : As > As,max (bande de 20 cm, h = 8 cm, Ø25/100)",
     note(dal("ASMAX", [sec("A", 1, 0, 1, 0, 2, inf_x=fa(ba(25, 100)), inf_y=fa(ba(25, 100)), sup_x=fa(ba(6, 250)),
                            sup_y=fa(ba(6, 250)))], b=20, h=8))),
    ("As,min = 0,0013·b·h (C20/25 : 0,26·fctm/fyk plus petit)", note(dal("PL", [sec("A", 10, 0, 12, 0, 20)], beton="C20/25"))),
    ("As,min = 0,25·As,req de la face opposée (grand moment sur appui)", note(dal("OPP", [sec("A", 0, 60, 0, 70, 50,
                                                                                            sup_x=fa(ba(12, 100)), sup_y=fa(ba(12, 100)))],
                                                                                 h=25))),
    ("y_G saisis (inf. X et sup. Y)", note(dal("YG", [sec("A", 25, 10, 32, 14, 50, inf_x=fa(tr("10/10/100/100"), nb(3, 12), yG=5.2),
                                                         sup_y=fa(tr("8/8/100/100"), yG=4.25))]))),
    ("axe saisi au-delà de h : d < 0 avec moment", note(dal("D0", [sec("A", 10, 0, 0, 0, 0, inf_x=fa(tr("10/10/100/100", dist=21.0)))]))),
    ("axe saisi au-delà de h SANS moment : seul le contrôle de d refuse", note(dal("D0M0", [sec("A", 0, 0, 0, 0, 0,
                                                                                               sup_y=fa(tr("8/8/150/150", dist=20.5)))]))),
    ("moments nuls, effort tranchant seul", note(dal("V", [sec("A", V=70)]))),
    ("égalité Mx,inf = My,inf sur les mêmes couches : inf. X gouverne (ordre de l'écran)", note(dal("EG", [sec("A", 30, 0, 30, 0)]))),
    ("sup. X gouverne la hauteur", note(dal("SX", [sec("A", 10, 40, 12, 8, sup_x=fa(ba(12, 125)))]))),
    ("sup. Y gouverne la hauteur", note(dal("SY", [sec("A", 10, 8, 12, 41, sup_y=fa(ba(12, 125)))]))),
    ("treillis hors de la base (« 9/9/150/150 », ancien fichier)", note(dal("OLD", [sec("A", 20, 0, 25, 0, inf_y=fa(tr("9/9/150/150")))]))),
    ("espacements et nombres aux bornes (25 et 500 mm, 1 et 50 barres)",
     note(dal("BOR", [sec("A", 15, 5, 18, 6, 20, inf_x=fa(ba(6, 500)), inf_y=fa(ba(6, 25)), sup_x=fa(nb(1, 8)), sup_y=fa(nb(50, 6)))]))),
    ("valeurs non entières (M = 37,25 ; V = 48,4 ; enrobage 2,75 ; jeu 0,25)",
     note(dal("DEC", [sec("A", 37.25, 12.75, 41.5, 9.125, 48.4)], enrobage=2.75), jeu=0.25)),
    ("nom de section qui commence par « section » (titre gardé tel quel)", note(dal("NOM", [sec("Section travée", 20, 0, 22, 0)]))),
    ("trois sections aux statuts mêlés, principale X",
     note(dal("MIX", [sec("A", 20, 0, 25, 0, 40), sec("B", 0, 45, 0, 55, 120), sec("C", 60, 0, 70, 0, 20)], dir_principale="X"))),
    ("deux dalles dans la note (γs et jeu partagés), principales différentes",
     note(dal("P1", [sec("A", 28, 12, 35, 9, 60), sec("B", 0, 30, 0, 38, 80)]),
          dal("P2", [sec("A", 15, 0, 18, 0, 30, inf_y=fa(ba(10, 150)))], dir_principale="X", h=18, beton="C25/30"), gamma_s=1.35)),
    ("trois dalles dans la note, l'une non conforme", note(dal("T1", [BIDIR]), dal("T2", [sec("A", 80, 0, 90, 0, 40)], h=18),
                                                           dal("T3", [sec("A", 5, 0, 6, 0)], b=50, fyk=400), jeu=0.5)),
]


def aleatoire(rng, k):
    """Une dalle tirée au sort (graine fixe) : dimensions, matériaux, couches (1 à 4 par face, trois types, axes et y_G
    saisis parfois), sollicitations de part et d'autre de la capacité."""
    b, h = rng.choice((20, 50, 100, 100, 100, 120, 150, 250, 500)), rng.choice((12, 15, 16, 18, 20, 22, 25, 30, 40))
    beton, fyk = rng.choice(sorted(CL)), rng.choice((400, 500))
    m_ref = (0.85 * h) ** 2 * CL[beton]["alpha_b"] * 10 * b * CL[beton][f"mu_a{fyk}"] / 1e6 / 100   # ≈ M qui remplit h
    v_ref = 0.016 * CL[beton]["fck_cube"] / 1.05 * 0.75 * b * h * 100 / 1e3                          # V à τ_adm,I

    def une_couche(i):
        t = rng.choices(("treillis", "barres", "n_barres"), (5, 3, 2) if i == 0 else (2, 3, 4))[0]
        return couche(t, rng.choice(TRB.liste_choix()), rng.choice(E.DIAMETRES[:7]), rng.choice((75, 100, 125, 150, 200, 250, 300)),
                      rng.randint(1, 12), round(rng.uniform(2.5, 0.6 * h), 1) if rng.random() < 0.12 else None)

    def une_face():
        cs = [une_couche(i) for i in range(rng.choices((1, 2, 3, 4), (6, 3, 1, 1))[0])]
        return {"couches": cs, "yG": round(rng.uniform(3.0, 0.5 * h), 2) if rng.random() < 0.08 else None}
    secs = []
    for i in range(rng.randint(1, 3)):
        mom = [round(rng.uniform(0, 1.4 * m_ref * b), 2) if rng.random() < 0.7 else 0.0 for _ in range(4)]
        secs.append(sec("ABC"[i], *mom, round(rng.uniform(0, 1.3 * v_ref), 1) if rng.random() < 0.75 else 0.0,
                        **{f: une_face() for f in E.FACES}))
    return M.normaliser({"nom": f"Aléa {k}", "b": b, "h": h, "beton": beton, "fyk": fyk, "dir_principale": rng.choice("XY"),
                         "enrobage": rng.choice((2.0, 2.5, 3.0, 3.5, 4.0)), "gamma_s": rng.choice((1.15, 1.3, 1.5)),
                         "jeu_premier_lit": rng.choice((0.0, 0.0, 0.5, 1.0)), "sections": secs})


RNG = random.Random(20261107)
CAS += [(f"tirage {k:02d} (graine 20261107)", note(aleatoire(RNG, k))) for k in range(1, 25)]
CAS += [(f"tirage multiple {k:02d} : deux dalles dans la note", note(aleatoire(RNG, 100 + k), aleatoire(RNG, 200 + k)))
        for k in range(1, 7)]


def cas(debut):
    return next(n for d, n in CAS if d.startswith(debut))


# ------------------------------------------------------------------ 1. données
print("=== 1. Données : classes de béton, défauts de l'écran, base des treillis ===")
FICHIER = json.loads(lire("beton_classes.json"))
chk("classes_beton() rend beton_classes.json, lu depuis le paquet (beton/dalle/../../)",
    M.classes_beton() == FICHIER and os.path.samefile(E.CHEMIN_CLASSES, os.path.join(RACINE, "beton_classes.json")))
copie = M.classes_beton()
copie["C30/37"]["alpha_b"] = 1.0
chk("classes_beton() rend une copie : la modifier ne change pas le calcul",
    M.verifier(M.nouvelle_dalle())["materiaux"]["alpha_b"] == FICHIER["C30/37"]["alpha_b"])

from modules import treillis as TRS  # noqa: E402 — la base de l'écran (sans streamlit)

chk("beton/treillis.py : la même base que modules/treillis.py (treillis, défaut, liste triée)",
    TRB.TREILLIS_STANDARDS == TRS.TREILLIS_STANDARDS and TRB.TREILLIS_DEFAUT == TRS.TREILLIS_DEFAUT
    and TRB.liste_choix() == TRS.liste_choix())
ENTREES = (["10/10/100/100", "5/5/150/150", "9/9/150/150", "12/12/100/100", " 8 / 8 /150/150", "10.5/10/100/100", "abc", "",
            "10/10/100", "0/10/100/100", "-6/6/100/100", "6/6/100/100/1", None, 12, "1e1/10/100/100"]
           + [TRB.designation(t) for t in TRB.TREILLIS_STANDARDS])
diff = [(f, x) for x in ENTREES for f in ("parse_designation", "as_treillis_mm2_m")
        if getattr(TRB, f)(x) != getattr(TRS, f)(x)]
diff += [("aire_barre_mm2", d) for d in (5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 6.5) if TRB.aire_barre_mm2(d) != TRS.aire_barre_mm2(d)]
diff += [("as_barres_mm2_m", (d, e)) for d in (6, 8, 12, 40) for e in (0, -5, 25, 100, 150.0, 333, 500)
         if TRB.as_barres_mm2_m(d, e) != TRS.as_barres_mm2_m(d, e)]
diff += [("designation", t) for t in TRB.TREILLIS_STANDARDS + [(9.0, 9, 150.0, 150)] if TRB.designation(t) != TRS.designation(t)]
chk(f"beton/treillis.py : chaque fonction rend, AU BIT PRÈS, ce que rend modules/treillis.py ({len(ENTREES) * 2 + 12 + 28 + 11} appels)",
    not diff, diff[:4])
fonctions = sorted(n.name for n in ast.parse(lire("modules", "treillis.py")).body if isinstance(n, ast.FunctionDef))
chk(f"beton/treillis.py porte chaque fonction de modules/treillis.py ({', '.join(fonctions)})",
    all(callable(getattr(TRB, f, None)) for f in fonctions), fonctions)
chk("est_canonique : la base est canonique, une désignation mal écrite ne l'est pas",
    all(TRB.est_canonique(TRB.designation(t)) for t in TRB.TREILLIS_STANDARDS)
    and not any(TRB.est_canonique(x) for x in (" 8 / 8 /150/150", "10.5/10/100/100", "abc", "", None, 12, "1e1/10/100/100")))

try:
    logging.disable(logging.WARNING)
    import streamlit as st  # noqa: E402
    import modules.dalle as D  # noqa: E402
    D.BETON_DATA = json.loads(lire("beton_classes.json"))
except ImportError as e:  # pragma: no cover — la parité EXIGE l'écran
    st = D = None
    chk("streamlit et modules/dalle.py se chargent (indispensables à la parité)", False, e)
try:
    import modules.export_pdf_dalle as XP  # noqa: E402 — la note PDF (reportlab)
except ImportError as e:  # pragma: no cover
    XP = None
    chk("modules/export_pdf_dalle.py se charge (reportlab : la note est la seconde copie du calcul)", False, e)

TYPES_ECRAN = {"treillis": "Treillis", "barres": "Barres", "n_barres": "n barres"}
DIR_TXT = lambda x: repr(float(x)).replace(".", ",")  # noqa: E731 — un nombre saisi à la française, exact


def lire_session(did):
    """La dalle ``did`` telle que l'état de l'écran la porte (après ses synchronisations), au format du moteur."""
    ss = st.session_state
    d = next(x for x in ss.dalles if int(x["id"]) == did)
    out = {"nom": ss[f"meta_dalle_nom_{did}"], "b": ss[D.KD("b", did)], "h": ss[D.KD("h", did)], "beton": ss[D.KD("beton", did)],
           "fyk": ss[D.KD("fyk", did)], "enrobage": ss[D.KD("enrobage_beton", did)], "dir_principale": ss[D.KD("dir_principale", did)],
           "gamma_s": ss["gamma_s"], "jeu_premier_lit": ss["jeu_enrobage_cm"], "sections": []}
    for s in d["sections"]:
        sid = int(s["id"])
        S = {"nom": ss[f"meta_dal{did}_nom_{sid}"], **{k: ss[D.KS(k, did, sid)] for k in E.MOMENTS + ("V",)}, "faces": {}}
        for which in E.FACES:
            couches = []
            for i in range(1, D._get_ncouches(did, sid, which) + 1):
                typ, des, dd, esp, n = D._couche_data(did, sid, which, i)
                auto = bool(ss.get(D.KS(f"dist_auto_{which}_c{i}", did, sid), True))
                try:
                    saisi = float(str(ss.get(D.KS(f"dist_axe_{which}_c{i}", did, sid), "") or "").strip().replace(",", "."))
                except ValueError:
                    saisi = None
                # une saisie illisible ou nulle est lue « automatique » par _dist_couche_eff : le moteur ne la représente pas
                couches.append({"type": {v: k for k, v in TYPES_ECRAN.items()}[typ], "treillis": des, "d": dd,
                                "esp": int(ss[D.KS(f"esp_barres_{which}_c{i}", did, sid)]), "n": n,
                                "dist": None if auto or saisi is None or saisi <= 0 else saisi})
            S["faces"][which] = {"couches": couches, "yG": D._ycdg_manual(did, sid, which)}
        out["sections"].append(S)
    return out


if D is not None:
    ENTETE = lire("modules", "dalle.py")[:300]
    chk(f"la référence de parité est bien celle du moteur : {M.REFERENCE}",
        "VERSION 2.0" in ENTETE and D.DALLE_VERSION == "2.2", D.DALLE_VERSION)
    chk("les listes du moteur sont celles de l'écran (Ø, types, couches, directions, faces)",
        list(E.DIAMETRES) == D.DIAM_OPTS and [E.LIBELLES_TYPES[t] for t in E.TYPES] == D.TYPES_ARMATURE
        and E.MAX_COUCHES == D.MAX_COUCHES and E.DIRECTIONS == D.DIR_KEYS and E.FACES == D.FACES_DIR)
    st.session_state.clear()
    D._ensure_global_defaults()
    D._init_dalles_if_needed()
    for which in E.FACES:
        D._sync_dist_state(1, 1, which, 1)
        D._sync_ycdg_state(1, 1, which)
    defaut = lire_session(1)
    # (la règle de position du 13/11/2026, beton/position.py : le jeu de pose du bureau vaut 1 cm — l'écran disait 0 — et
    #  l'arrondi est une clé nouvelle ; tout le reste est l'état de l'écran)
    neuve = M.nouvelle_dalle()
    chk("nouvelle_dalle() : le jeu de pose du bureau (1 cm) et l'arrondi (0,5 cm) de la règle du 13/11/2026",
        neuve["jeu_premier_lit"] == 1.0 and neuve["arrondi"] == 0.5 and defaut["jeu_premier_lit"] == 0.0,
        (neuve["jeu_premier_lit"], neuve.get("arrondi")))
    neuve = {k: v for k, v in neuve.items() if k not in ("jeu_premier_lit", "arrondi")}
    defaut.pop("jeu_premier_lit")
    chk("nouvelle_dalle() = l'état de l'écran au premier affichage (bande 100 × 20, C30/37, B500, Y, treillis "
        "10/10/100/100…), hors jeu de pose et arrondi", not ecarts_json(neuve, defaut), ecarts_json(neuve, defaut))
    D._add_section(1)
    D._add_couche(1, 2, "inf_y")
    for which in E.FACES:
        for i in range(1, D._get_ncouches(1, 2, which) + 1):
            D._sync_dist_state(1, 2, which, i)
        D._sync_ycdg_state(1, 2, which)
    s2 = lire_session(1)["sections"][1]
    attendu = M.nouvelle_section(M.nom_section_suivant(M.nouvelle_dalle()))
    attendu["faces"]["inf_y"]["couches"].append(M.nouvelle_couche(base=False))
    chk("nouvelle_section() = la section que « ➕ » ajoute, nouvelle_couche(False) = le renfort que « ＋ » ajoute (3 Ø12)",
        not ecarts_json(attendu, s2), ecarts_json(attendu, s2))
    for noms in (["A", "B", "D"], ["B"], [chr(65 + i) for i in range(26)], [" A ", "C"]):
        st.session_state.dalles[0]["sections"] = [{"id": i + 1, "nom": n} for i, n in enumerate(noms)]
        for i, n in enumerate(noms):
            st.session_state[f"meta_dal1_nom_{i + 1}"] = n
        attendu = D._next_section_name(1)
        obtenu = M.nom_section_suivant({"sections": [{"nom": n} for n in noms]})
        chk(f"nom_section_suivant({noms if len(noms) < 5 else 'A…Z'}) = {attendu!r} comme _next_section_name", obtenu == attendu, obtenu)

# ------------------------------------------------------------------ 2. à la main
print("\n=== 2. Contrôles à la main (formules du bureau écrites en N et en mm) ===")


def proche(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b))


def fr(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


r1 = M.verifier(cas("bidirectionnelle complète")[0])["sections"][0]
g1 = r1["grandeurs"]
v1 = {v["cle"]: v for v in r1["verifications"]}
fyd = 500 / 1.5
# la règle de position du 13/11/2026 (beton/position.py) : ⌈c + Ø/2 + jeu de pose⌉ au pas de 5 mm, une fois — le moteur
# seul ne pose pas une direction sur l'autre (le site le fait : tests/test_api_web_dalle_positions.py)
e = math.ceil((30 + 5 + 10) / 5) * 5                                    # ⌈30 + 5 + 10⌉₅ = 45 mm
As10 = math.pi * 10 ** 2 / 4 * 1000 / 100                               # treillis 10/10/100/100 : 785,4 mm²/m
req = {f: m * 1e6 / (fyd * 0.9 * (200 - e)) for f, m in (("inf_y", 35), ("inf_x", 28), ("sup_y", 9), ("sup_x", 12))}
fctm = 0.30 * 30 ** (2 / 3)
chk("H1 axes de calcul à 45 mm (⌈30 + Ø10/2 + 10 (jeu)⌉ au 5 mm), d = 155 mm sur les quatre faces",
    all(proche(g1[f"yG_{f}"] * 10, e) and proche(g1[f"d_{f}"] * 10, 200 - e) for f in E.FACES))
chk("H1 As,req indépendants des quatre familles : 753 / 602 / 194 / 258 mm² (d = 155 mm ; 707 / 566 / 182 / 242 à "
    "d = 165 mm, l'écran Streamlit : test_dalle_bidir)",
    all(proche(g1[f"As_req_{f}"], req[f]) for f in req) and [round(req[f]) for f in ("inf_y", "inf_x", "sup_y", "sup_x")] == [753, 602, 194, 258],
    {f: g1[f"As_req_{f}"] for f in req})
As_min = {f: max(0.26 * fctm / 500 * 1000 * 200, 0.0013 * 1000 * 200, 0.25 * req[o])
          for f, o in (("inf_y", "sup_y"), ("sup_y", "inf_y"), ("inf_x", "sup_x"), ("sup_x", "inf_x"))}
chk(f"H1 As,min = max(0,26·fctm/fyk·b·h = {fr(0.26 * fctm / 500 * 2e5, 1)} ; 0,0013·b·h = 260 ; 0,25·As,req opposé de la même direction)",
    all(proche(g1[f"As_min_{f}"], As_min[f]) for f in As_min), {f: g1[f"As_min_{f}"] for f in As_min})
chk("H1 taux = max(As,req ; As,min) / As fourni (785,4 mm² sur la bande d'un mètre)",
    all(proche(v1[f]["taux"], max(req[f], As_min[f]) / As10) for f in req))
hu = math.sqrt(35e6 / (12.96 * 1000 * 0.1709))                          # mm
chk(f"H1 hauteur : hu,min(35) + y_G = {fr(hu / 10, 2)} + 4,5 = {fr(hu / 10 + 4.5, 2)} cm (inf. Y gouverne)",
    proche(g1["h_min"], hu / 10 + 4.5) and g1["face_hauteur"] == "inf_y" and fr(g1["hu_min"], 1) == "12,6")
tau = 60e3 / (0.75 * 1000 * 200)
chk(f"H1 τ = V/(0,75·b·h) = {fr(tau, 4)} ≤ τ_adm,I = 0,016·37/1,05 = {fr(0.016 * 37 / 1.05, 4)} N/mm², sans étriers",
    proche(g1["tau"], tau) and proche(v1["tau"]["taux"], tau / (0.016 * 37 / 1.05)) and v1["tau"]["etat"] == "ok")
r2 = M.verifier(cas("renfort « n barres »")[0])["sections"][0]
chk("H2 renfort 3 Ø12 dans la bande : 3·π·12²/4 = 339 mm²/m, As fourni = 785 + 339 = 1 125 mm²/m",
    proche(r2["faces"]["inf_y"]["couches"][1]["As_pm"], 3 * math.pi * 144 / 4) and round(r2["grandeurs"]["As_pm_inf_y"]) == 1125
    and r2["faces"]["inf_y"]["libelle"] == "Treillis 10/10/100/100 + 3 Ø12")
r3 = M.verifier(cas("renfort posé à 8,0 cm")[0])["sections"][0]
# le renfort posé à 8,0 cm (position saisie, réelle) : axe de calcul ⌈8,0 + 1⌉ = 9,0 cm ; le treillis 4,5 cm
yG = (As10 * 4.5 + 3 * math.pi * 144 / 4 * 9.0) / (As10 + 3 * math.pi * 144 / 4)
chk(f"H3 position saisie 8,0 cm (axe de calcul 9,0) : y_G = Σ As·e / Σ As = {fr(yG, 3)} cm (« 5,9 » à l'écran), "
    "As,req = 35·10⁶ / (333,3 × 0,9 × 141,4) = 825 mm²",
    proche(r3["grandeurs"]["yG_inf_y"], yG) and fr(yG, 1) == "5,9" and round(r3["grandeurs"]["As_req_inf_y"]) == 825,
    (r3["grandeurs"]["yG_inf_y"], r3["grandeurs"]["As_req_inf_y"]))
r4 = M.verifier(cas("audit C-1")[0])["sections"][0]
chk("H4 audit C-1 : la famille posée à 8,0 cm (axe de calcul 9,0) gouverne, hu,min(30) + 9,0 = 11,6 + 9,0 = 20,6 cm "
    "(et non hu,min(32) + 4,5 = 12,0 + 4,5 = 16,5)",
    r4["grandeurs"]["face_hauteur"] == "inf_y" and r4["grandeurs"]["M_hauteur"] == 30
    and (fr(r4["grandeurs"]["hu_min"], 1), fr(r4["grandeurs"]["h_min"], 1)) == ("11,6", "20,6"),
    (r4["grandeurs"]["hu_min"], r4["grandeurs"]["h_min"]))

# ------------------------------------------------------------------ 3. parité
ICONES = {"🟢": "ok", "🟡": "warn", "🔴": "nok"}
CORR_HAUT = (  # clé de _dimensionnement_compute_states ↔ grandeur du moteur
    ("etat_h", "etat_hauteur"), ("etat_inf", "etat_inf"), ("etat_sup", "etat_sup"), ("etat_tau", "etat_tau"),
    ("principale", "principale"), ("V_val", "V"), ("tau", "tau"), ("tau_adm", "tau_adm"), ("M_max", "M_hauteur"),
    ("hmin_calc", "hu_min"), ("e_cdg_gov", "yG_hauteur"), ("h_min_dalle", "h_min"), ("tau_1", "tau_1"), ("tau_2", "tau_2"),
    ("tau_4", "tau_4"), ("fyd", "fyd"), ("gamma_s", "gamma_s"), ("fyk", "fyk"), ("alpha_b", "alpha_b"), ("mu_val", "mu"),
    ("beton", "beton"), ("b", "b"), ("h", "h"), ("fctm", "fctm"), ("As_min_ec", "As_min_ec"),
    ("As_min_plancher", "As_min_plancher"), ("As_max", "As_max"))
CORR_DIR = (  # clé de r["dirs"][dk] ↔ grandeur du moteur, {} = la face
    ("M_inf_val", "M_inf_{}"), ("M_sup_val", "M_sup_{}"), ("As_inf_total", "As_inf_{}"), ("As_sup_total", "As_sup_{}"),
    ("As_inf_pm", "As_pm_inf_{}"), ("As_sup_pm", "As_pm_sup_{}"), ("e_cdg_inf", "yG_inf_{}"), ("e_cdg_sup", "yG_sup_{}"),
    ("d_utile_inf", "d_inf_{}"), ("d_utile_sup", "d_sup_{}"), ("geom_inf_ok", "geometrie_inf_{}_ok"),
    ("geom_sup_ok", "geometrie_sup_{}_ok"), ("As_formule_inf", "As_req_inf_{}"), ("As_formule_sup", "As_req_sup_{}"),
    ("As_min_inf_eff", "As_min_inf_{}"), ("As_min_sup_eff", "As_min_sup_{}"), ("As_req_inf_final", "As_req_inf_{}"),
    ("As_req_sup_final", "As_req_sup_{}"), ("etat_inf", "etat_inf_{}"), ("etat_sup", "etat_sup_{}"))


class FauxSt:
    """Un « st » qui ENREGISTRE ce que l'écran afficherait. L'état reste le vrai st.session_state ; les widgets
    rendent leur valeur de session (rien n'est saisi)."""

    def __init__(self):
        self.ev = []
        self.session_state = st.session_state

    def markdown(self, body="", unsafe_allow_html=False, help=None, **_):
        self.ev.append(("md", str(body), help or ""))

    def expander(self, label, expanded=False, **_):
        self.ev.append(("exp", label))
        return contextlib.nullcontext()

    def container(self, **_):
        return contextlib.nullcontext()

    def columns(self, spec, **_):
        return [contextlib.nullcontext() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def _widget(self, *_, key=None, **__):
        return self.session_state.get(key) if key else None

    selectbox = text_input = number_input = checkbox = _widget

    def button(self, *_, **__):
        return False


def charger(dalles):
    """L'état de l'écran pour la note ``dalles`` (format du moteur), puis le début d'un run (_ensure_global_defaults,
    _init_dalles_if_needed : défauts, migrations, coercitions) et les synchronisations des champs « Dist. axe » et
    « CDG » que render_dimensionnement_section fait avant de calculer."""
    ss = st.session_state
    ss.clear()
    D._ensure_global_defaults()
    # la POSITION de chaque couche : la règle du 13/11/2026 (beton/position.py — position réelle, jeu de pose, arrondi),
    # que l'écran ne connaît pas ; il reçoit l'axe de calcul du moteur comme un axe saisi (« Dist. axe », comme la note
    # PDF : api_web/note_dalle.valeurs) — la parité porte alors sur tout le reste (formules, états, écran), au bit près ;
    # les positions sont prouvées à la main (tests/test_position.py, tests/test_api_web_dalle_positions.py)
    axes = [[{f: [c["e"] for c in sec_["faces"][f]["couches"]] for f in E.FACES} for sec_ in M.verifier(d)["sections"]]
            for d in dalles]
    ss["gamma_s"], ss["jeu_enrobage_cm"] = dalles[0]["gamma_s"], dalles[0]["jeu_premier_lit"]
    ss.dalles = [{"id": k + 1, "nom": d["nom"], "sections": [{"id": i + 1, "nom": s["nom"]} for i, s in enumerate(d["sections"])]}
                 for k, d in enumerate(dalles)]
    for k, d in enumerate(dalles):
        did = k + 1
        ss[f"meta_dalle_nom_{did}"] = d["nom"]
        for cle_ecran, cle in (("b", "b"), ("h", "h"), ("enrobage_beton", "enrobage"), ("beton", "beton"), ("fyk", "fyk"),
                               ("dir_principale", "dir_principale")):
            ss[D.KD(cle_ecran, did)] = d[cle]
        for i, s in enumerate(d["sections"]):
            sid = i + 1
            ss[f"meta_dal{did}_nom_{sid}"] = s["nom"]
            for x in E.MOMENTS + ("V",):
                ss[D.KS(x, did, sid)] = s[x]
            for which in E.FACES:
                f = s["faces"][which]
                ss[D.KS(f"ncouches_{which}", did, sid)] = len(f["couches"])
                for j, c in enumerate(f["couches"], start=1):
                    for cle_ecran, v in (("arm_type", TYPES_ECRAN[c["type"]]), ("treillis", c["treillis"]), ("ø_barres", c["d"]),
                                         ("esp_barres", c["esp"]), ("n_barres", c["n"])):
                        ss[D.KS(f"{cle_ecran}_{which}_c{j}", did, sid)] = v
                    # le champ « Dist. axe » saisi à la française : l'axe de calcul du moteur
                    ss[D.KS(f"dist_axe_{which}_c{j}", did, sid)] = DIR_TXT(axes[k][i][which][j - 1])
                    ss[D.KS(f"dist_auto_{which}_c{j}", did, sid)] = False
                if f["yG"] is not None:         # le champ « CDG (cm) »
                    ss[D.KS(f"ycdg_{which}", did, sid)] = DIR_TXT(f["yG"])
                    ss[D.KS(f"ycdg_auto_{which}", did, sid)] = False
    D._ensure_global_defaults()
    D._init_dalles_if_needed()
    for k, d in enumerate(dalles):
        for i in range(len(d["sections"])):
            for which in E.FACES:
                for j in range(1, D._get_ncouches(k + 1, i + 1, which) + 1):
                    D._sync_dist_state(k + 1, i + 1, which, j)
                D._sync_ycdg_state(k + 1, i + 1, which)


def ecran(dalles):
    """L'écran sur la note : le calcul (section par section) et ce que render_dimensionnement_right affiche."""
    charger(dalles)
    calc = []
    for k, d in enumerate(dalles):
        for i, s in enumerate(d["sections"]):
            did, sid = k + 1, i + 1
            c = {"r": D._dimensionnement_compute_states(did, sid, D.BETON_DATA), "faces": {}}
            for which in E.FACES:
                As, yG, detail, As_pm = D._layers_geometry(did, sid, which)
                _, yG_auto, _, _ = D._layers_geometry(did, sid, which, use_manual=False)
                c["faces"][which] = {"As": As, "yG": yG, "detail": detail, "As_pm": As_pm, "yG_calcule": yG_auto,
                                     "yG_impose": D._ycdg_manual(did, sid, which) is not None,
                                     "couches": [{"As_pm": D._couche_as_per_m(did, sid, which, j), "e": D._dist_couche_eff(did, sid, which, j),
                                                  "e_auto": D._auto_dist_couche(did, sid, which, j), "libelle": D._couche_label(did, sid, which, j),
                                                  "d": D._couche_diam_mm(did, sid, which, j)}
                                                 for j in range(1, D._get_ncouches(did, sid, which) + 1)]}
            if XP is not None:
                c["pdf"] = XP._compute_section(dict(st.session_state), D.BETON_DATA, did, sid)
            calc.append(c)
    payload = json.loads(json.dumps(D._build_save_payload(), ensure_ascii=False))
    faux = FauxSt()
    vrai_st, vrai_bloc = D.st, D.open_bloc_left_right

    def espion(left, right="", etat="ok", pct=None):
        faux.ev.append(("bloc", left, right, etat, pct))
        return vrai_bloc(left, right, etat, pct)
    D.st, D.open_bloc_left_right = faux, espion
    try:
        D.render_dimensionnement_right(D.BETON_DATA)
    finally:
        D.st, D.open_bloc_left_right = vrai_st, vrai_bloc
    titres_dalles, ecrans = [], []
    for ev in faux.ev:
        if ev[0] == "exp":
            ecrans.append({"titre": ev[1], "blocs": [], "md": []})
        elif ecrans:
            if ev[0] == "bloc":
                ecrans[-1]["blocs"].append({"bloc": ev[1:], "md": []})
            elif ecrans[-1]["blocs"] and not ev[1].startswith(('<div style="background-color', "</div>")):
                ecrans[-1]["blocs"][-1]["md"].append(ev[1:])     # (le HTML du bandeau et la fermeture du bloc sont écartés)
    # les expanders se suivent : dalle, ses sections, dalle suivante…
    out, k = [], 0
    for d in dalles:
        titres_dalles.append(ecrans[k]["titre"])
        out += ecrans[k + 1:k + 1 + len(d["sections"])]
        k += 1 + len(d["sections"])
    for c, e in zip(calc, out):
        c["ecran"] = e
    return titres_dalles, calc, payload


def _fr(x, nd=1):
    return f"{float(x):.{nd}f}".replace(".", ",")


def comparer(es, md, ecarts):
    """Une section du moteur (es) contre l'écran (md : calcul, couches, affichage, note PDF)."""
    def cmp(nom, a, b):
        if not egal(a, b):
            ecarts.append(f"{es['nom']}.{nom} : moteur {a!r} ≠ écran {b!r}")
    g, r = es["grandeurs"], md["r"]
    rendues = set(r) - {"dirs", "etat_global"}
    if rendues != {k for k, _ in CORR_HAUT}:
        ecarts.append(f"clés de _dimensionnement_compute_states non comparées : {sorted(rendues ^ {k for k, _ in CORR_HAUT})}")
    for cle_r, cle_m in CORR_HAUT:
        cmp(cle_m, g[cle_m], r[cle_r])
    cmp("statut", es["statut"], r["etat_global"])
    for dk in E.DIRECTIONS:
        if set(r["dirs"][dk]) != {k for k, _ in CORR_DIR} | {"inf_detail", "sup_detail"}:
            ecarts.append(f"clés de dirs[{dk}] non comparées : {sorted(set(r['dirs'][dk]) ^ ({k for k, _ in CORR_DIR} | {'inf_detail', 'sup_detail'}))}")
        for cle_r, cle_m in CORR_DIR:
            cmp(f"{cle_m.format(dk)}", g[cle_m.format(dk)], r["dirs"][dk][cle_r])
        for cote in ("inf", "sup"):
            cmp(f"libellé {cote}_{dk}", es["faces"][f"{cote}_{dk}"]["libelle"], r["dirs"][dk][f"{cote}_detail"])
    # le terme de As,min que le moteur NOMME désigne bien la valeur que l'écran a retenue
    for f in E.FACES:
        cote, dk = f.split("_")
        opp = ("sup" if cote == "inf" else "inf")
        terme = {"ec": r["As_min_ec"], "plancher": r["As_min_plancher"], "opposee": 0.25 * r["dirs"][dk][f"As_formule_{opp}"]}
        cmp(f"As_min_terme_{f}", terme[g[f"As_min_terme_{f}"]], r["dirs"][dk][f"As_min_{cote}_eff"])
        cmp(f"besoin_{f}", g[f"besoin_{f}"], max(r["dirs"][dk][f"As_req_{cote}_final"], r["dirs"][dk][f"As_min_{cote}_eff"]))
        cmp(f"d_calc_{f}", g[f"d_calc_{f}"], max(r["dirs"][dk][f"d_utile_{cote}"], 0.1))
        # chaque face, chaque couche
        geo, mf = es["faces"][f], md["faces"][f]
        for cle in ("As", "yG", "As_pm", "yG_calcule"):
            cmp(f"{f}.{cle}", geo[cle], mf[cle])
        cmp(f"{f}.libellé", geo["libelle"], mf["detail"])
        cmp(f"{f}.yG_impose", geo["yG_impose"], mf["yG_impose"])
        cmp(f"{f}.couches", len(geo["couches"]), len(mf["couches"]))
        for j, (a, b) in enumerate(zip(geo["couches"], mf["couches"]), start=1):
            for cle in ("As_pm", "e", "libelle", "d"):      # (« e_auto » : la règle du 13/11/2026, que l'écran ignore)
                cmp(f"{f}.couche {j}.{cle}", a[cle], b[cle])
    for v in es["verifications"]:
        cmp(f"motifs {v['cle']} (vides si et seulement si l'état est ok)", not v["motifs"], v["etat"] == "ok")
    # ---- l'écran
    e = md["ecran"]
    cmp("icône de la section", es["statut"], ICONES.get(e["titre"][:1]))
    cmp("titre de la section", es["titre"], e["titre"][2:])
    blocs = e["blocs"]
    cmp("blocs (titres, dans l'ordre de l'écran)", [v["titre"] for v in es["verifications"]], [b["bloc"][0] for b in blocs])
    for v, b in zip(es["verifications"], blocs):
        _, droite, etat, pct = b["bloc"]
        lignes = [x[0] for x in b["md"]]
        aides = {x[0]: x[1] for x in b["md"]}
        cmp(f"% {v['cle']}", v["taux"], pct / 100.0)
        cmp(f"état affiché {v['cle']}", v["etat"], etat)
        if v["cle"] == "hauteur":
            cmp("bandeau hauteur", f"{g['beton']} • Section {g['b']:.0f}×{g['h']:.0f} cm", droite)
            M_h = g["M_hauteur"]
            m_txt = _fr(M_h, 0) if abs(M_h - round(M_h)) < 1e-9 else _fr(M_h, 1)
            l1 = (f"hᵤ,min = √( {m_txt}·10⁶ / ({_fr(g['alpha_b'], 2)} · {_fr(g['b'] * 10, 0)} · {_fr(g['mu'], 4)}) ) = "
                  f"**{_fr(g['hu_min'], 1)} cm**") if M_h > 0 else "hᵤ,min = **0,0 cm** — aucun moment appliqué"
            ok = v["etat"] == "ok"
            l2 = (f"hᵤ,min + d₁ = {_fr(g['hu_min'], 1)} + {_fr(g['yG_hauteur'], 1)} = **{_fr(g['h_min'], 1)} cm** "
                  f"{'≤' if ok else '>'} h = **{_fr(g['h'], 0)} cm** {'✅' if ok else '❌'}")
            cmp("lignes de la hauteur", [l1, l2], lignes[:2])
        elif v["cle"] == "tau":
            cmp("bandeau τ (sans texte)", "", droite)
            ok = v["etat"] == "ok"
            cmp("ligne τ (point décimal, comme l'écran)",
                f"τ = {g['tau']:.2f} N/mm² {'≤' if ok else '>'} τ_adm,I = {g['tau_adm']:.2f} N/mm² {'✅' if ok else '❌'}", lignes[0])
        else:
            f = v["cle"]
            cote, dk = f.split("_")
            opp = ("sup_" if cote == "inf" else "inf_") + dk
            cmp(f"bandeau {f}", f"B{int(g['fyk'])} • {v['libelle']} ({v['limite']:.0f} mm²)", droite)
            req, mini, besoin = g[f"As_req_{f}"], g[f"As_min_{f}"], g[f"besoin_{f}"]
            attendu = {
                f"**Aₛ,req = {req:.0f} mm²**": ("**Aₛ,req = M / (fyd · 0,9 · d)**\n\n" f"M = {_fr(g[f'M_{f}'], 1)} kN·m\n\n"
                                                f"fyd = {_fr(g['fyk'], 0)} / {_fr(g['gamma_s'], 2)} = {_fr(g['fyd'], 0)} N/mm²\n\n"
                                                f"d = {_fr(g[f'd_{f}'], 1)} cm\n\n" f"→ Aₛ,req = {_fr(req, 0)} mm²"),
                f"**Aₛ,min = {mini:.0f} mm²**": ("**Aₛ,min = max( 0,26·fctm/fyk·b·h ; 0,0013·b·h ; 0,25·Aₛ,req face opposée, même direction )**\n\n"
                                                 f"0,26 · {_fr(g['fctm'], 1)} / {_fr(g['fyk'], 0)} · b · h = {_fr(g['As_min_ec'], 0)} mm²\n\n"
                                                 f"0,0013 · b · h = {_fr(g['As_min_plancher'], 0)} mm²\n\n"
                                                 f"0,25 · {_fr(g[f'As_req_{opp}'], 0)} = {_fr(g[f'As_min_opposee_{f}'], 0)} mm²\n\n"
                                                 f"→ Aₛ,min = {_fr(mini, 0)} mm²"),
                f"**Aₛ,max = {g['As_max']:.0f} mm²**": ("**Aₛ,max = 0,04 · b · h**\n\n"
                                                         f"0,04 · {_fr(g['b'] * 10, 0)} · {_fr(g['h'] * 10, 0)} = {_fr(g['As_max'], 0)} mm²"),
                f"**Aₛ fourni = {v['As_pm']:.0f} mm²/m**": ("**Aₛ fourni = Σ couches × largeur de bande**\n\n" f"{v['libelle']}\n\n"
                                                            f"Σ = {_fr(v['As_pm'], 0)} mm²/m × {_fr(g['b'] / 100.0, 2)} m = {_fr(v['limite'], 0)} mm²")}
            cmp(f"lignes et infobulles {f} (Aₛ,req, Aₛ,min, Aₛ,max, Aₛ fourni)", attendu, {k: aides.get(k) for k in lignes[:4]})
            cmp(f"besoin {f} (valeur dimensionnante de l'écran)", besoin, max(req, mini))
            geom = "❌ **Position des couches incompatible avec l'épaisseur : d utile ≤ 0.**" in lignes
            cmp(f"message « d utile ≤ 0 » {f}", "d_utile_negatif" in v["motifs"], geom)
            cmp(f"lignes du tableau des couches {f}", [str(j) for j in range(1, len(es["faces"][f]["couches"]) + 1)],
                [x for x in lignes if x.isdigit()])
    # ---- la note PDF (seconde copie du calcul dans l'écran)
    R = md.get("pdf")
    if R is not None:
        for cle_r, cle_m in (("beton", "beton"), ("fck", "fck"), ("alpha_b", "alpha_b"), ("fctm", "fctm"), ("fyk", "fyk"), ("fyd", "fyd"),
                             ("gamma_s", "gamma_s"), ("mu", "mu"), ("b", "b"), ("h", "h"), ("enrob_beton", "enrobage_calcul"),
                             ("principale", "principale"), ("V", "V"), ("M_max", "M_hauteur"), ("hmin", "hu_min"), ("etat_h", "etat_hauteur"),
                             ("e_cdg_gov", "yG_hauteur"), ("h_min_dalle", "h_min"), ("As_min_ec", "As_min_ec"),
                             ("As_min_plancher", "As_min_plancher"), ("As_max", "As_max"), ("etat_inf", "etat_inf"), ("etat_sup", "etat_sup")):
            cmp(f"note PDF {cle_r}", g[cle_m], R[cle_r])
        cmp("note PDF fck_cube", g["fck_cube"], R["fck_cube"])
        cmp("note PDF etat_global", es["statut"], R["etat_global"])
        cmp("note PDF tranchant", (g["tau"], g["tau_adm"], g["etat_tau"]) if g["V"] > 0 else None,
            (R["shear"]["tau"], R["shear"]["tau_adm"], R["shear"]["etat_tau"]) if R["shear"] else None)
        for dk in E.DIRECTIONS:
            Rd = R["dirs"][dk]
            for cote in ("inf", "sup"):
                f = f"{cote}_{dk}"
                for cle_r, cle_m in ((f"M_{cote}", f"M_{f}"), (f"As_req_{cote}", f"As_req_{f}"), (f"As_min_{cote}", f"As_min_{f}"),
                                     (f"As_{cote}", f"As_{f}"), (f"etat_{cote}", f"etat_{f}"), (f"geom_{cote}_ok", f"geometrie_{f}_ok"),
                                     ("di" if cote == "inf" else "ds", f"d_{f}")):
                    cmp(f"note PDF {dk}.{cle_r}", g[cle_m], Rd[cle_r])
                geo, Rg = es["faces"][f], Rd[f"geo_{cote}"]
                cmp(f"note PDF {f} As, As/m, y_G, détail, couches",
                    (geo["As"], geo["As_pm"], geo["yG"], geo["libelle"], len(geo["couches"])),
                    (Rg["As"], Rg["As_pm"], Rg["e_cdg"], Rg["detail"], Rg["nc"]))
                for a, b in zip(geo["couches"], Rg["couches"]):
                    cmp(f"note PDF {f} couche {b['i']}", (a["As_pm"], a["e"], a["d"], a["esp"], a["n"], a["libelle"], a["valeur"]),
                        (b["As_pm"], b["e"], b["d"], b["esp"], b["n"], b["label"], b["valeur"]))


def faits(dalles, res, calc):
    """Ce que la note atteint (couverture), lu dans les sorties CONTRÔLÉES contre l'écran."""
    out = set()
    if len(dalles) > 1:
        out.add("plusieurs dalles")
    k = 0
    for d, r in zip(dalles, res):
        out |= {f"béton {d['beton']}", f"B{d['fyk']}", f"principale {d['dir_principale']}", f"dalle {r['statut']}"}
        if len(d["sections"]) > 1:
            out.add("plusieurs sections")
        for s, es in zip(d["sections"], r["sections"]):
            md = calc[k]
            k += 1
            g = es["grandeurs"]
            out |= {f"statut {es['statut']}", f"hauteur {g['face_hauteur']}", f"état hauteur {g['etat_hauteur']}"}
            for v in es["verifications"]:
                out |= {f"motif {m}" for m in v["motifs"]} | {f"état {v['cle'][:3]} {v['etat']}"}
                if v["taux"] > 1:
                    out.add(f"taux {v['cle'][:3]} > 1")
                bl = next(b for b in md["ecran"]["blocs"] if b["bloc"][0] == v["titre"])
                if bl["bloc"][3] == 100.0 and v["etat"] == "ok":
                    out.add(f"borne {v['cle'][:3]}")
            out |= {f"As,min {g[f'As_min_terme_{f}']}" for f in E.FACES}
            out |= {f"{len(s['faces'][f]['couches'])} couche(s)" for f in E.FACES}
            out |= {f"couche {c['type']}" + (" en renfort" if j else "") for f in E.FACES for j, c in enumerate(s["faces"][f]["couches"])}
            out |= {"axe saisi" for f in E.FACES for c in s["faces"][f]["couches"] if c["dist"] is not None}
            out |= {"y_G saisi" for f in E.FACES if s["faces"][f]["yG"] is not None}
            out |= {"V = 0" if not s["V"] else "V > 0"}
            if not any(s[m] for m in E.MOMENTS):
                out.add("sans moment")
            if any(c["type"] == "treillis" and c["treillis"] not in TRB.liste_choix() for f in E.FACES for c in s["faces"][f]["couches"]):
                out.add("treillis hors de la base")
        out |= {f"avertissement {a['code']}" for a in r["avertissements"]}
    return out


def avec(dalles, **champs):
    """Une copie de la note dont la première section de la première dalle reçoit ``champs`` (h, V, yG_inf_x…)."""
    q = copy.deepcopy(dalles)
    for k, x in champs.items():
        if k in ("b", "h"):
            q[0][k] = x
        elif k.startswith("yG_"):
            q[0]["sections"][0]["faces"][k[3:]]["yG"] = x
        else:
            q[0]["sections"][0][k] = x
    return [M.normaliser(d) for d in q]


if D is not None:
    print("\n=== 3. Parité avec modules/dalle.py, note par note (calcul, couches, écran, note PDF, fichier) ===")
    # les bornes EXACTES, construites avec l'écran : τ = τ_adm,I (V flottant) et h = h_min (y_G saisi flottant)
    base = cas("τ au-delà")
    charger(base)
    r0 = D._dimensionnement_compute_states(1, 1, D.BETON_DATA)
    den = 0.75 * r0["b"] * r0["h"] * 100
    V0 = r0["tau_1"] * den / 1e3
    V = None
    for sens in (math.inf, -math.inf):
        x = V0
        for _ in range(64):
            if x * 1e3 / den == r0["tau_1"]:
                V = x
                break
            x = math.nextafter(x, sens)
        if V is not None:
            break
    if V is not None:
        CAS.append((f"borne exacte : τ = τ_adm,I (V = {V!r}, construit avec l'écran)", avec(base, V=V)))
    hu0 = r0["hmin_calc"]
    y = None
    for sens in (math.inf, -math.inf):
        x = 20.0 - hu0
        for _ in range(64):
            if hu0 + x == 20.0:
                y = x
                break
            x = math.nextafter(x, sens)
        if y is not None:
            break
    if y is not None and r0["dirs"]["y"]["M_inf_val"] == r0["M_max"]:
        CAS.append((f"borne exacte : h = h_min (y_G saisi {y!r}, construit avec l'écran)", avec(base, yG_inf_y=y, V=10.0)))
    COUV = set()
    ECRANS = []          # ce que l'écran a rendu, note par note : les mutations (section 9) s'y comparent
    for k, (desc, dalles) in enumerate(CAS):
        res = [M.verifier(d) for d in dalles]
        titres, calc, payload = ecran(dalles)
        ECRANS.append((desc, dalles, titres, calc))
        ecarts = []
        es_tous = [s for r in res for s in r["sections"]]
        for es, md in zip(es_tous, calc):
            comparer(es, md, ecarts)
        if len(es_tous) != len(calc):
            ecarts.append(f"{len(es_tous)} sections au moteur, {len(calc)} à l'écran")
        for r, t in zip(res, titres):     # l'icône de chaque dalle : la fusion de ses sections
            if not egal(r["statut"], ICONES.get(t[:1])) or t[2:] != r["dalle"]["nom"]:
                ecarts.append(f"dalle {r['dalle']['nom']} : moteur {r['statut']} ≠ écran {t}")
        # le gouvernant de chaque dalle : le plus grand % de ses blocs
        k0 = 0
        for r in res:
            pcts = [(i, b["bloc"][0], b["bloc"][3]) for i, md in enumerate(calc[k0:k0 + len(r["sections"])]) for b in md["ecran"]["blocs"]]
            k0 += len(r["sections"])
            gv = max(pcts, key=lambda x: x[2])
            if not (egal(r["taux_max"], gv[2] / 100.0) and (r["gouvernant"]["section"], r["gouvernant"]["titre"]) == gv[:2]):
                ecarts.append(f"gouvernant : moteur {r['gouvernant']} ≠ écran {gv}")
        # le fichier que l'écran écrit, relu par le moteur, redonne la note — chaque couche avec l'axe que l'écran a reçu
        # (l'axe de calcul du moteur, « Dist. axe » saisi : voir charger)
        lu = FI.lire(payload)
        attendues = copy.deepcopy(dalles)
        for d_, r_ in zip(attendues, res):
            for s_, rs_ in zip(d_["sections"], r_["sections"]):
                for f_ in E.FACES:
                    for c_, cr_ in zip(s_["faces"][f_]["couches"], rs_["faces"][f_]["couches"]):
                        c_["dist"] = cr_["e"]
        diff = ecarts_json(lu["dalles"], attendues)
        if diff or lu["remarques"]:
            ecarts.append(f"fichier relu : {diff[:3]} {lu['remarques'][:2]}")
        COUV |= faits(dalles, res, calc)
        n_g = sum(len(s["grandeurs"]) for s in es_tous)
        chk(f"note {k + 1:02d} · {desc} — {len(dalles)} dalle(s), {len(es_tous)} section(s), {n_g} grandeurs, % de l'écran, note PDF, fichier",
            not ecarts, ecarts[:6])
    n_sec = sum(len(d["sections"]) for _, ds in CAS for d in ds)
    chk(f"{len(CAS)} notes ({sum(len(ds) for _, ds in CAS)} dalles, {n_sec} sections) comparées : au moins 60 notes", len(CAS) >= 60)
    print(f"     {ECART['n']} comparaisons numériques : {ECART['bit']} identiques au bit près, écart relatif maximal "
          f"{ECART['max']:.2e} (tolérance {TOL:g})")
    chk("l'écart relatif maximal reste sous 1e-9 (c'est le même calcul)", ECART["max"] <= TOL, ECART["max"])

    print("\n=== 4. Couverture : ce que les notes atteignent réellement ===")
    ATTENDUS = (["plusieurs dalles", "plusieurs sections", "V = 0", "V > 0", "sans moment", "dalle ok", "dalle nok", "statut ok", "statut nok",
                 "principale X", "principale Y", "B400", "B500", "axe saisi", "y_G saisi", "treillis hors de la base"]
                + [f"béton {b}" for b in CL]
                + [f"état {k} {e}" for k in ("hau", "inf", "sup") for e in ("ok", "nok")] + ["état tau ok", "état tau nok"]
                + [f"hauteur {f}" for f in E.FACES + (None,)] + ["état hauteur ok", "état hauteur nok"]
                + [f"motif {m}" for m in ("hauteur_insuffisante", "d_utile_negatif", "acier_insuffisant", "acier_excessif", "cisaillement_excessif")]
                + [f"taux {k} > 1" for k in ("hau", "inf", "sup", "tau")]
                + ["As,min ec", "As,min plancher", "As,min opposee"] + [f"{n} couche(s)" for n in (1, 2, 3, 4)]
                + [f"couche {t}" for t in E.TYPES] + [f"couche {t} en renfort" for t in E.TYPES]
                + ["avertissement enrobage_nul", "avertissement d_utile_negatif", "avertissement nappes_confondues", "borne tau", "borne hau"])
    manque = [a for a in ATTENDUS if a not in COUV]
    chk(f"les notes atteignent les {len(ATTENDUS)} situations attendues (états, motifs, termes, types, bornes…)", not manque, manque)

    print("\n=== 5. Sélecteur de la nappe de base : chaque nappe repassée par l'écran ===")
    for desc, i_s, face in (("bidirectionnelle complète", 0, "inf_y"), ("renfort posé à 8,0 cm", 0, "inf_y"), ("y_G saisis", 0, "inf_x"),
                            ("trois sections", 1, "sup_x"), ("largeur de bande maximale", 0, "sup_x"), ("acier excessif", 0, "inf_x")):
        dalles = cas(desc)
        cands = M.candidats_base(dalles[0], i_s, face)
        ecarts = []
        for c in cands:
            q = copy.deepcopy(dalles)
            c1 = q[0]["sections"][i_s]["faces"][face]["couches"][0]
            c1.update({k: c[k] for k in ("type", "treillis", "d", "esp")})
            charger(q)
            r = D._dimensionnement_compute_states(1, i_s + 1, D.BETON_DATA)
            cote, dk = face.split("_")
            R = r["dirs"][dk]
            for nom_c, a, b in (("As de la face", c["As"], R[f"As_{cote}_total"]), ("y_G", c["yG"], R[f"e_cdg_{cote}"]),
                                ("taux", c["taux"], max(R[f"As_req_{cote}_final"], R[f"As_min_{cote}_eff"]) / R[f"As_{cote}_total"]),
                                ("état", c["etat"], R[f"etat_{cote}"]), ("statut", c["statut_section"], r["etat_global"]),
                                ("As/m de la nappe", c["As_pm"], D._couche_as_per_m(1, i_s + 1, face, 1))):
                if not egal(a, b):
                    ecarts.append(f"{c['nom']}.{nom_c} : {a!r} ≠ {b!r}")
        forme = (len(cands) == len(TRB.TREILLIS_STANDARDS) + len(M.CANDIDATS_D) * len(M.CANDIDATS_ESP)
                 and [c["As_pm"] for c in cands] == sorted(c["As_pm"] for c in cands))
        premier = next((c for c in cands if c["ok"]), None)
        b0 = dalles[0]["sections"][i_s]["faces"][face]["couches"][0]
        nom0 = M.candidat_nom(b0) if b0["type"] != "n_barres" else None
        marques = (sum(c["optimal"] for c in cands) == (premier is not None) and (premier is None or premier["optimal"])
                   and [c["nom"] for c in cands if c["retenue"]] == ([nom0] if nom0 in [c["nom"] for c in cands] else []))
        chk(f"sélecteur {dalles[0]['nom']} · section {i_s + 1} · {face} : {len(cands)} nappes triées par section, optimale = "
            f"{premier['nom'] if premier else 'aucune'} ; As, y_G, taux, états identiques à l'écran",
            not ecarts and forme and marques, (ecarts[:4], forme, marques))
    fams = M.candidats_base(cas("défauts")[0], 0, "inf_y", ["treillis"])
    chk("le filtre des familles : « treillis » seul ne propose que les dix treillis de la base",
        [c["nom"] for c in fams] == [f"Treillis {x}" for x in TRB.liste_choix()])

    print("\n=== 6. Fichiers de l'écran : migration v1 → v2, saisies remplacées en silence ===")

    def ouvrir(payload):
        """L'écran ouvre un fichier (_load_from_payload), fait un run (raw → valeurs) et ses synchronisations."""
        st.session_state.clear()
        D._load_from_payload(copy.deepcopy(payload))
        D._sync_float_raw_keys()
        D._init_dalles_if_needed()
        for d in st.session_state.dalles:
            for s in d["sections"]:
                for which in E.FACES:
                    for j in range(1, D._get_ncouches(int(d["id"]), int(s["id"]), which) + 1):
                        D._sync_dist_state(int(d["id"]), int(s["id"]), which, j)
                    D._sync_ycdg_state(int(d["id"]), int(s["id"]), which)
        return [M.normaliser(lire_session(int(d["id"]))) for d in st.session_state.dalles]
    V1 = {"dalles": [{"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "A"}]}],
          "values": {"dal1_sec1_M_inf": 28.0, "dal1_sec1_M_sup": 12.0, "dal1_sec1_V": 55.0, "dal1_sec1_ncouches_inf": 2,
                     "dal1_sec1_arm_type_inf_c1": "Treillis", "dal1_sec1_treillis_inf_c1": "10/10/100/100",
                     "dal1_sec1_arm_type_inf_c2": "Barres", "dal1_sec1_ø_barres_inf_c2": 12, "dal1_sec1_esp_barres_inf_c2": 150,
                     "dal1_sec1_ncouches_sup": 1, "dal1_sec1_treillis_sup_c1": "8/8/150/150", "gamma_s": 1.5}}
    PIEGES = [("fichier v1 (test_dalle_bidir) : M_inf → Mx_inf, couches → direction X, renfort compris", V1, False)]
    v1b = copy.deepcopy(V1)
    v1b["values"].update({"dal1_sec1_M_inf_raw": "31,5", "dal1_b": 120, "dal1_h": 22.0, "dal1_dir_principale": "x",
                          "meta_dalle_nom_1": "Plancher haut", "meta_dal1_nom_1": "Travée"})
    PIEGES.append(("fichier v1 avec texte saisi « 31,5 », b = 120, h = 22.0, principale « x » minuscule, noms", v1b, False))
    deux = {"dalles": [{"id": 3, "nom": "Sans dalle 1", "sections": [{"id": 2, "nom": "B"}]}], "values": {"dal3_sec2_My_inf": 12.0}}
    PIEGES.append(("dalle 1 et section 1 absentes : l'écran les recrée (dalle 1 neuve en tête, section A en tête)", deux, False))
    neg = {"dalles": [{"id": 1, "nom": "D", "sections": [{"id": 1, "nom": "A"}]}],
           "values": {"dal1_sec1_Mx_inf": 20.0, "dal1_sec1_Mx_inf_raw": "-20,00", "dal1_sec1_V": 30.0, "dal1_beton": "C16/20",
                      "dal1_fyk": 450, "dal1_sec1_arm_type_inf_x_c1": "Grille", "dal1_sec1_ø_barres_sup_y_c1": 14,
                      "dal1_sec1_dist_axe_inf_y_c1": "abc", "dal1_sec1_dist_auto_inf_y_c1": False,
                      "dal1_sec1_ycdg_sup_x": "0", "dal1_sec1_ycdg_auto_sup_x": False, "dal1_sec1_esp_barres_inf_x_c1": 7,
                      "dal1_sec1_ncouches_sup_y": 9}}
    PIEGES.append(("saisies que l'écran remplace en silence (négatif, classe, nuance, type, Ø, axe, y_G, bornes) : mêmes valeurs", neg, True))
    # « dalles » qui n'est pas une liste (ou des dalles qui ne sont pas des objets) : l'écran repart de la dalle 1 neuve
    PIEGES.append(("« dalles » n'est pas une liste : la dalle 1 neuve, comme l'écran", {"dalles": 5, "values": {"dal1_sec1_V": 12.0}}, False))
    PIEGES.append(("une dalle qui n'est pas un objet est sautée, comme l'écran", {"dalles": [5, "x", {"id": 2, "nom": "D2"}]}, False))
    for desc, payload, remarques in PIEGES:
        attendu = ouvrir(payload)
        lu = FI.lire(copy.deepcopy(payload))
        diff = ecarts_json(lu["dalles"], attendu)
        chk(f"{desc} — {len(attendu)} dalle(s)", not diff and bool(lu["remarques"]) == remarques, (diff[:4], lu["remarques"][:3]))
    lu = FI.lire(neg)
    chk("… et chaque remplacement silencieux de l'écran est relevé (négatif, classe, nuance, type, Ø, axe, y_G)",
        len(lu["remarques"]) >= 7, lu["remarques"])
    for nom, payload in (("treillis illisible", {"values": {"dal1_sec1_treillis_inf_x_c1": "10/10/abc/100"}}),
                         ("largeur hors des bornes de l'écran", {"values": {"dal1_b": 800}}),
                         ("pas un fichier de dalle", [1, 2])):
        try:
            FI.lire(payload)
            chk(f"fichier : {nom} → ValueError (le moteur ne peut pas le représenter)", False)
        except ValueError as e_:
            chk(f"fichier : {nom} → ValueError (« {str(e_)[:70]}… »)", True)

# ------------------------------------------------------------------ 7. format
print("\n=== 7. Format : JSON, normalisation, domaine de l'écran, formules ===")
tous = [M.verifier(d) for _, ds in CAS for d in ds]
chk("la sortie de verifier() est du JSON strict (sans NaN) et se relit à l'identique",
    all(json.loads(json.dumps(r, allow_nan=False)) == r for r in tous))
chk("normaliser() est idempotente et {} donne nouvelle_dalle()",
    all(M.normaliser(M.normaliser(d)) == M.normaliser(d) for _, ds in CAS for d in ds) and M.normaliser({}) == M.nouvelle_dalle())
entree = copy.deepcopy(cas("renfort posé")[0])
M.verifier(entree)
M.candidats_base(entree, 0, "inf_y")
chk("verifier() et candidats_base() ne modifient pas la dalle reçue", entree == cas("renfort posé")[0])
chk("les libellés de l'écran sont admis (« Treillis », « Barres », « n barres »)",
    [c["type"] for c in M.normaliser({"sections": [{"faces": {"inf_x": {"couches": [{"type": "Treillis"}, {"type": "Barres"},
                                                                                     {"type": "n barres"}]}}}]})
     ["sections"][0]["faces"]["inf_x"]["couches"]] == ["treillis", "barres", "n_barres"])
chk("sans nom, une section prend la première lettre libre",
    [s["nom"] for s in M.normaliser({"sections": [{"nom": "A"}, {}, {"nom": "C"}, {}]})["sections"]] == ["A", "B", "C", "D"])


def f_inf_x(**c):
    return {"sections": [{"faces": {"inf_x": {"couches": [dict(c)]}}}]}


REFUS = (("largeur sous 20 cm", {"b": 15}), ("largeur au-delà de 500 cm", {"b": 600}), ("largeur non entière", {"b": 100.5}),
         ("épaisseur sous 5 cm", {"h": 4}), ("épaisseur au-delà de 100 cm", {"h": 101}), ("classe inconnue", {"beton": "C16/20"}),
         ("nuance 450", {"fyk": 450}), ("enrobage négatif", {"enrobage": -1}), ("enrobage au-delà de 20 cm", {"enrobage": 21}),
         ("direction « Z »", {"dir_principale": "Z"}), ("γs sous 1", {"gamma_s": 0.9}), ("γs au-delà de 2", {"gamma_s": 2.5}),
         ("jeu négatif", {"jeu_premier_lit": -0.5}), ("largeur en texte", {"b": "100"}), ("épaisseur booléenne", {"h": True}),
         ("enrobage infini", {"enrobage": float("inf")}), ("clé inconnue", {"largeur": 100}), ("aucune section", {"sections": []}),
         ("moment négatif", {"sections": [{"My_inf": -10}]}), ("V négatif", {"sections": [{"V": -1}]}),
         ("face inconnue", {"sections": [{"faces": {"inf_z": {}}}]}), ("aucune couche", {"sections": [{"faces": {"inf_x": {"couches": []}}}]}),
         ("cinq couches", {"sections": [{"faces": {"inf_x": {"couches": [{}] * 5}}}]}), ("type « grille »", f_inf_x(type="grille")),
         ("treillis illisible", f_inf_x(treillis="10/10/abc/100")), ("treillis mal écrit", f_inf_x(treillis="10.5/10/100/100")),
         ("Ø14 (non proposé)", f_inf_x(d=14)), ("espacement 20 mm", f_inf_x(esp=20)), ("espacement 600 mm", f_inf_x(esp=600)),
         ("0 barre", f_inf_x(n=0)), ("51 barres", f_inf_x(n=51)), ("2,5 barres", f_inf_x(n=2.5)), ("axe nul", f_inf_x(dist=0)),
         ("axe négatif", f_inf_x(dist=-3)), ("y_G nul", {"sections": [{"faces": {"inf_x": {"yG": 0}}}]}),
         ("clé de couche inconnue", f_inf_x(diam=12)), ("clé de section inconnue", {"sections": [{"M_inf": 10}]}))
refusees = []
for nom, champ in REFUS:
    try:
        M.verifier(dict(M.nouvelle_dalle(), **champ))
        refusees.append(f"{nom} : accepté")
    except ValueError:
        pass
    except Exception as e_:  # noqa: BLE001 — une autre exception n'est pas un refus : c'est une panne
        refusees.append(f"{nom} : {type(e_).__name__} {e_}")
chk(f"{len(REFUS)} saisies hors du domaine de l'écran lèvent ValueError (jamais de substitution silencieuse)", not refusees, refusees)
for nom, appel in (("face inconnue", lambda: M.candidats_base(M.nouvelle_dalle(), 0, "milieu")),
                   ("section inconnue", lambda: M.candidats_base(M.nouvelle_dalle(), 3, "inf_x")),
                   ("nappe illisible", lambda: M.lire_candidat("Treillis 10/10")), ("nappe inconnue", lambda: M.lire_candidat("HA12"))):
    try:
        appel()
        chk(f"{nom} → ValueError", False)
    except ValueError:
        chk(f"{nom} → ValueError", True)
chk("lire_candidat relit chaque nom du sélecteur", all(M.candidat_nom(dict(M.nouvelle_couche(), **M.lire_candidat(c["nom"]))) == c["nom"]
                                                         for c in M.candidats_base(M.nouvelle_dalle(), 0, "inf_x")))
r_e0 = M.verifier(cas("enrobage nul")[0])
r_e3 = M.verifier(dict(cas("enrobage nul")[0], enrobage=3.0))
chk("enrobage nul : avertissement « enrobage_nul » et calcul à 3,0 cm (comme l'écran) ; aucun à 3,0 cm",
    "enrobage_nul" in [a["code"] for a in r_e0["avertissements"]] and "enrobage_nul" not in [a["code"] for a in r_e3["avertissements"]]
    and r_e0["sections"][0]["grandeurs"]["h_min"] == r_e3["sections"][0]["grandeurs"]["h_min"])
r_nc = M.verifier(cas("renfort posé")[0])
chk("nappes confondues : signalées quand les axes X et Y d'une face sont plus proches que (Øx + Øy)/2, pas quand elles sont écartées",
    [a["face"] for a in M.verifier(M.nouvelle_dalle())["avertissements"] if a["code"] == "nappes_confondues"] == ["inf", "sup"]
    and not [a for a in M.verifier(cas("audit C-1")[0])["avertissements"] if a["code"] == "nappes_confondues" and a["face"] == "inf"]
    and all(a["code"] != "nappes_confondues" or abs(a["e_x"] - a["e_y"]) < a["ecart_mini"] for a in r_nc["avertissements"]))
# « formule = valeurs = résultat » : chaque résultat se retrouve à partir des seuls termes que le moteur donne
FORMULES = {
    "hu_min": lambda t: math.sqrt(t["M"] * 1e6 / (t["alpha_b"] * t["b"] * t["mu"])) / 10,
    "h_min": lambda t: t["hu_min"] + t["yG"],
    "d": lambda t: t["h"] - t["yG"],
    "yG": lambda t: sum(a * e for a, e in t.values()) / sum(a for a, _ in t.values()),
    "As": lambda t: t["As_pm"] * t["b"] / 100,
    "As_req": lambda t: t["M"] * 1e6 / (t["fyd"] * t["k"] * t["d"] * 10) if t["M"] > 0 else 0.0,
    "As_min_ec": lambda t: t["k"] * t["fctm"] / t["fyk"] * t["b"] * t["h"],
    "As_min_plancher": lambda t: t["k"] * t["b"] * t["h"],
    "As_min_opposee": lambda t: t["k"] * t["As_req_opp"],
    "As_min": lambda t: max(t["As_min_ec"], t["As_min_plancher"], t["As_min_opposee"]),
    "besoin": lambda t: max(t["As_req"], t["As_min"]),
    "As_max": lambda t: t["k"] * t["b"] * t["h"],
    "tau": lambda t: t["V"] * 1e3 / (t["k"] * t["b"] * t["h"]),
    "tau_lim": lambda t: t["k"] * t["fck_cube"] / t["gamma"],
}
VALEUR_LIMITE = {"hauteur": ("h_min", None), "tau": ("tau", "tau_lim")}
fautes, n_formules = [], 0
for r in tous:
    for s in r["sections"]:
        for v in s["verifications"]:
            resultats = {}
            for f in v["formules"]:
                n_formules += 1
                t = {x["symbole"]: x["valeur"] for x in f["termes"]}
                resultats[f["cle"]] = f["resultat"]
                if not (proche(FORMULES[f["cle"]](t), f["resultat"]) or FORMULES[f["cle"]](t) == f["resultat"]):
                    fautes.append(f"{r['dalle']['nom']}/{s['nom']}/{v['cle']}/{f['cle']} : {FORMULES[f['cle']](t)!r} ≠ {f['resultat']!r}")
            cv, cl = VALEUR_LIMITE.get(v["cle"], ("besoin", "As"))
            if v["valeur"] != resultats[cv] or (cl and v["limite"] != resultats[cl]) or v["taux"] != v["valeur"] / v["limite"]:
                fautes.append(f"{r['dalle']['nom']}/{s['nom']}/{v['cle']} : valeur, limite et taux ≠ formules")
chk(f"« formule = valeurs = résultat » : les {n_formules} formules se recalculent à partir de leurs seuls termes, et valeur / limite = taux",
    not fautes and n_formules > 1500, fautes[:4])
doc = lire("docs", "beton", "DALLE_MOTEUR.md")
cles = sorted({re.sub(r"_(inf|sup)_[xy]$", "_<f>", k).replace("geometrie_inf_x_ok", "geometrie_<f>_ok")
               for k in tous[1]["sections"][0]["grandeurs"]})
cles = sorted({re.sub(r"geometrie_(inf|sup)_[xy]_ok", "geometrie_<f>_ok", k) for k in cles})
sans_doc = [k for k in cles if f"`{k}`" not in doc or not re.search(rf"(?<![\w<]){re.escape(k)}(?![\w>])", M.__doc__)]
chk(f"les {len(cles)} grandeurs (par face : _<f>) sont nommées dans la docstring du moteur ET dans docs/beton/DALLE_MOTEUR.md",
    not sans_doc, sans_doc)

# ------------------------------------------------------------------ 8. pureté
print("\n=== 8. Pureté : le moteur se charge et calcule SANS streamlit, avec la seule bibliothèque standard ===")
non_std = []
for chemin in [os.path.join("beton", "treillis.py")] + [os.path.join("beton", "dalle", f) for f in os.listdir(os.path.join(RACINE, "beton", "dalle"))
                                                        if f.endswith(".py")]:
    arbre = ast.parse(lire(chemin))
    for n in ast.walk(arbre):
        noms = [a.name for a in n.names] if isinstance(n, ast.Import) else \
            [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
        non_std += [f"{chemin}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names | {"beton"}]
chk("aucun fichier de beton/dalle ni beton/treillis.py n'importe autre chose que la bibliothèque standard", not non_std, non_std)

SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("streamlit", "pandas", "reportlab", "numpy", "matplotlib", "scipy", "pymupdf", "fitz", "modules")
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("chargement refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
try:
    import streamlit  # noqa: F401
    refus = False
except ImportError:
    refus = True
sys.path.insert(0, sys.argv[1])
from beton.dalle import moteur as M, fichier as F
dalles = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.verifier(d) for d in dalles],
          "cands": M.candidats_base(dalles[1], 0, "inf_y"), "fichier": F.lire({"values": {"dal1_sec1_M_inf": 12.0}})}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"beton"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
dalles = [d for _, ds in CAS for d in ds]
attendu = {"res": json.loads(json.dumps(tous)), "cands": json.loads(json.dumps(M.candidats_base(dalles[1], 0, "inf_y"))),
           "fichier": json.loads(json.dumps(FI.lire({"values": {"dal1_sec1_M_inf": 12.0}})))}
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(dalles), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans streamlit", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : streamlit, pandas, reportlab, numpy, modules/… REFUSÉS par sys.meta_path, et le moteur calcule quand même "
        f"({len(out['res'])} dalles, le sélecteur, la lecture d'un fichier)", out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : le moteur n'a chargé que la bibliothèque standard", not out["tiers"], out["tiers"])
    diff = ecarts_json({"res": out["res"], "cands": out["cands"], "fichier": out["fichier"]}, attendu)
    chk(f"Python {v} : résultats identiques à ceux de ce processus", not diff, diff[:3])
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")

# ------------------------------------------------------------------ 9. mutations
# Le moteur est ALTÉRÉ (une constante, un opérateur, une règle du bureau), rechargé sous un autre nom, et comparé à ce
# que l'écran a rendu en section 3 : chaque altération doit faire rougir au moins une note. Une mutation qui passe
# serait un trou de la preuve (un cas qui manque).
if D is not None:
    print("\n=== 9. Mutations : le moteur altéré, la parité avec l'écran rougit ===")
    import types

    SOURCE = lire("beton", "dalle", "moteur.py")
    MUTATIONS = [
        ("bras de levier 0,9·d → 0,85·d", "K_BRAS_DE_LEVIER = 0.9 ", "K_BRAS_DE_LEVIER = 0.85 "),
        ("As,min : 0,25·As,req opposé → 0,30", "K_AS_MIN_OPPOSEE = 0.25", "K_AS_MIN_OPPOSEE = 0.30"),
        ("As,min : la face opposée → la face elle-même", "opposee = K_AS_MIN_OPPOSEE * req[opp]", "opposee = K_AS_MIN_OPPOSEE * req[f]"),
        ("As,min,EC : 0,26 → 0,25", "K_AS_MIN_EC = 0.26", "K_AS_MIN_EC = 0.25"),
        ("As,min plancher : 0,0013 → 0,0012", "K_AS_MIN_PLANCHER = 0.0013", "K_AS_MIN_PLANCHER = 0.0012"),
        ("As,max : 0,04 → 0,045", "K_AS_MAX = 0.04 ", "K_AS_MAX = 0.045 "),
        ("As ≤ As,max oublié", " and As <= As_max) else NOK", ") else NOK"),
        ("τ = V/(0,75·b·h) → V/(b·h)", "K_TAU_SECTION = 0.75", "K_TAU_SECTION = 1.0"),
        ("τ ≤ τ_adm,I → τ < τ_adm,I (la borne)", 'etat_tau = OK if tau <= mat["tau_1"] else NOK', 'etat_tau = OK if tau < mat["tau_1"] else NOK'),
        ("τ_adm,I → τ_adm,II (une dalle à étriers)", 'g.update({"tau": tau, "tau_adm": mat["tau_1"]', 'g.update({"tau": tau, "tau_adm": mat["tau_2"]'),
        ("h_min ≤ h → h_min < h (la borne)", '"etat_hauteur": OK if h_min <= h else NOK', '"etat_hauteur": OK if h_min < h else NOK'),
        ("hauteur : la famille au plus grand moment (avant l'audit C-1)",
         "max(actives, key=lambda x: _hu_min(x[1], alpha_b, b, mu) + x[2])", "max(actives, key=lambda x: x[1])"),
        ("h_u,min : b en cm au lieu de mm", "math.sqrt((m * 1e6) / (alpha_b * b * 10 * mu)) / 10", "math.sqrt((m * 1e6) / (alpha_b * b * mu)) / 10"),
        ("fctm : 0,30·fck^(2/3) → 0,30·fck^0,66", "K_FCTM * (fck ** (2.0 / 3.0))", "K_FCTM * (fck ** 0.66)"),
        ("fyd = fyk / γs → fyk / 1,15", "fyd = fyk / gamma_s", "fyd = fyk / 1.15"),
        ("axe : le jeu de pose oublié (règle du 13/11/2026)", "e = P.axe_calcul(reel, jeu, arrondi)",
         "e = P.axe_calcul(reel, 0.0, arrondi)"),
        ("axe : l'arrondi oublié", "e = P.axe_calcul(reel, jeu, arrondi)", "e = P.axe_calcul(reel, jeu, 0.0)"),
        ("enrobage nul lu 0 (et non 3,0 cm comme l'écran)", "P.sur_parement(float(enrobage or ENROBAGE_SI_NUL), couche_diametre(c))",
         "P.sur_parement(float(enrobage), couche_diametre(c))"),
        ("position saisie ignorée", 'reel = reel_auto if c["dist"] is None else float(c["dist"])', "reel = reel_auto"),
        ("centre de gravité saisi ignoré", '"yG": yG_calcule if yG is None else yG', '"yG": yG_calcule'),
        ("« n barres » : la largeur de bande ignorée", "/ max(0.01, bb / 100.0)", "/ 1.0"),
        ("As de la face : la largeur de bande ignorée", '"As": As_pm_tot * (float(b) / 100.0)', '"As": As_pm_tot'),
        ("d de calcul : plancher 0,1 cm → 0,5 cm", "D_CALCUL_MIN = 0.1 ", "D_CALCUL_MIN = 0.5 "),
        ("ordre des vérifications : X avant Y quelle que soit la principale", 'for f in ordre_faces(g["principale"]):', "for f in E.FACES:"),
        ("direction principale ignorée (toujours Y)", 'principale = d["dir_principale"].lower()', 'principale = "y"'),
    ]

    def mutant(avant, apres):
        """Le moteur dont la source porte UNE altération, chargé sous un autre nom (le vrai reste intact)."""
        if SOURCE.count(avant) != 1:
            return None
        mod = types.ModuleType("beton.dalle._mutant")
        mod.__package__ = "beton.dalle"
        exec(compile(SOURCE.replace(avant, apres), "beton/dalle/moteur.py (altéré)", "exec"), mod.__dict__)
        return mod

    for nom, avant, apres in MUTATIONS:
        mod = mutant(avant, apres)
        if mod is None:
            chk(f"mutation « {nom} » : le texte altéré est dans moteur.py, une seule fois", False, avant)
            continue
        rouges = 0
        for desc, dalles, _titres, calc in ECRANS:
            try:
                es_tous = [s_ for d_ in dalles for s_ in mod.verifier(d_)["sections"]]
            except Exception:  # noqa: BLE001 — un moteur altéré qui casse rougit aussi
                rouges += 1
                continue
            ecarts = []
            for es, md in zip(es_tous, calc):
                comparer(es, md, ecarts)
            rouges += bool(ecarts)
        chk(f"mutation « {nom} » : {rouges} note(s) sur {len(ECRANS)} rougissent", rouges > 0)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
