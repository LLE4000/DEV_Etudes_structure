# -*- coding: utf-8 -*-
"""Moteur « Poutre béton armé » (beton/poutre) : la PARITÉ avec l'écran Streamlit modules/poutre.py v2.40,
qui reste la référence. Chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Données : classes_beton() est beton_classes.json ; nouvelle_poutre(), nouvelle_section() et
     nom_section_suivant() sont les défauts du module, LUS dans son état (_ensure_global_defaults,
     _init_beams_if_needed, _ensure_defaults_for_beam, _add_section, _next_section_name).
  2. À la main : deux sections recalculées ici avec les formules du bureau, écrites en N et en mm.
  3. Parité (plus de 40 poutres, sections multiples) : chaque poutre est chargée dans l'état du module piloté
     SANS écran, puis on compare au moteur :
       a) toutes les grandeurs et les six états de _dimensionnement_compute_states ;
       b) _get_dist_lit (chaque lit), _layers_geometry (y_G calculé), _shear_lines_total_Ast_mm2,
          _stirrup_diam_mm, _shear_lines_summary ;
       c) ce que l'ÉCRAN affiche : render_dimensionnement_right est exécuté avec un faux « st » qui
          enregistre au lieu d'afficher et un espion sur open_bloc_left_right — titre, état et POURCENTAGE
          EXACT (flottant) de chaque bloc, d'où τ et s_lim que le module ne calcule QUE dans son code
          d'écran ; la ligne « τ = … ≤ τ_adm_… → … » (bande et besoin) ; s_th et s_max tronqués ; la
          ligne « hᵤ,min + d₁ » ; l'icône d'état de chaque section et de la poutre.
     Tolérance 1e-9 relative ; l'écart maximal observé est imprimé.
  4. Couverture : chaque bande de τ, chaque état, chaque motif de non-conformité, chaque terme de As,min,
     les bornes exactes (≤ et non <)… sont effectivement atteints par les cas.
  5. Sélecteur d'armatures : chaque disposition est repassée par le module.
  6. Format : JSON, normalisation (défauts, ValueError hors du domaine de l'écran), avertissement.
  7. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit,
     pandas, reportlab (et numpy, matplotlib, scipy, pymupdf) et n'y charge que la bibliothèque standard,
     sous Python 3.12 et sous 3.13 (celui de Pyodide) s'il est installé — résultats identiques.

    /usr/bin/python3.12 tests/test_poutre_moteur.py       (streamlit requis pour la parité)
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

from beton.poutre import entree as E  # noqa: E402
from beton.poutre import moteur as M  # noqa: E402

OK, KO = [], []


def lire(*chemin):
    """Le texte d'un fichier du dépôt ("" s'il n'existe pas)."""
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
    """Les écarts entre deux sorties JSON : mêmes clés, mêmes longueurs, nombres à 1e-9."""
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
def sec(nom, M_inf=0.0, M_sup=0.0, V=0.0, inf=((2, 16),), sup=((2, 16),), et=(("etrier", 10),), pas=30.0,
        yG_inf=None, yG_sup=None):
    return {"nom": nom, "M_inf": M_inf, "M_sup": M_sup, "V": V,
            "inf": [{"n": n, "d": d} for n, d in inf], "sup": [{"n": n, "d": d} for n, d in sup],
            "etriers": [{"type": t, "d": d} for t, d in et], "pas": pas, "yG_inf": yG_inf, "yG_sup": yG_sup}


def poutre(nom, sections, **kw):
    p = {"nom": nom, "sections": sections}
    p.update(kw)
    return M.normaliser(p)


CL = M.classes_beton()
L2 = ((2, 16), (2, 16))
CAS = [  # (description, poutre)
    ("défauts du module (nouvelle_poutre)", M.nouvelle_poutre()),
    ("V1 du site 30×50 : travée A et appui B (étrier + épingle)",
     poutre("V1", [sec("A", 122, 0, 60, ((3, 20),), ((2, 16),), (("etrier", 8),), 20),
                   sec("B", 0, 95, 120, ((2, 20),), ((4, 16),), (("etrier", 8), ("epingle", 8)), 10)],
            b=30, h=50, jeu_entre_lits=1.5)),
    ("travée seule (M_inf), C25/30, B500, V = 0",
     poutre("T1", [sec("A", 85, 0, 0, ((3, 16),), ((2, 12),))], b=25, h=50, beton="C25/30")),
    ("même travée en B400 (μ_a400)",
     poutre("T1-B400", [sec("A", 85, 0, 0, ((3, 16),), ((2, 12),))], b=25, h=50, beton="C25/30", fyk=400)),
    ("appui seul (M_sup) avec effort tranchant",
     poutre("A1", [sec("A", 0, 140, 150, ((2, 12),), ((4, 20),), (("etrier", 8),), 15)], b=30, h=55)),
    ("travée et appui sur la même section, étrier + épingle Ø10",
     poutre("TA", [sec("A", 90, 140, 180, ((3, 20),), ((3, 25),), (("etrier", 10), ("epingle", 10)), 12)],
            b=35, h=60)),
    ("moments nuls, effort tranchant seul",
     poutre("V", [sec("A", 0, 0, 80, pas=20)], b=25, h=45)),
    ("deux lits inférieurs (2Ø20 + 2Ø16), jeu entre lits 2 cm",
     poutre("2L", [sec("A", 220, 0, 140, ((2, 20), (2, 16)), ((2, 12),), (("etrier", 10),), 15)],
            b=30, h=60, jeu_entre_lits=2.0)),
    ("audit C-1 : trois lits en bas, la famille la plus exigeante gouverne la hauteur",
     poutre("C-1", [sec("A", 150, 160, 120, ((2, 16),) * 3, ((2, 16),))], b=30, h=60)),
    ("quatre lits en bas, deux en haut, deux étriers Ø12, grande section",
     poutre("4L", [sec("A", 600, 250, 400, ((3, 25), (3, 25), (2, 20), (2, 16)), ((3, 20), (2, 16)),
                       (("etrier", 12), ("etrier", 12)), 10)], b=40, h=80)),
]
for beton in CL:  # les cinq bétons : bandes de τ différentes pour le même V
    CAS.append((f"béton {beton}, V = 170 kN",
                poutre(f"B-{beton}", [sec("A", 150, 60, 170, ((4, 20),), ((2, 16),), (("etrier", 10),), 12)],
                       b=30, h=50, beton=beton)))
CAS += [
    ("γs = 1,15 (fyd = 435 N/mm²)", poutre("GS115", [sec("A", 150, 50, 160, ((3, 20),), ((2, 12),), pas=15)],
                                          b=30, h=50, gamma_s=1.15)),
    ("γs = 1,35, B400, C35/45", poutre("GS135", [sec("A", 110, 40, 140, ((3, 20),), ((2, 12),), pas=18)],
                                       b=30, h=50, gamma_s=1.35, fyk=400, beton="C35/45")),
    ("enrobage 2,5 cm, jeu premier lit 0,5 cm", poutre("E25", [sec("A", 100, 30, 90, ((3, 16),), ((2, 12),), pas=20)],
                                                      b=25, h=50, enrobage=2.5, jeu_premier_lit=0.5)),
    ("enrobage 5 cm, jeu premier lit 1 cm, jeu entre lits nul, deux lits",
     poutre("E5", [sec("A", 180, 0, 100, ((3, 20), (3, 16)), ((2, 12),), pas=20)], b=30, h=60, enrobage=5.0,
            jeu_premier_lit=1.0, jeu_entre_lits=0.0)),
    ("enrobage 4 cm, jeu entre lits 2,5 cm, deux lits en haut",
     poutre("E4", [sec("A", 60, 200, 150, ((2, 16),), ((3, 20), (2, 20)), pas=15)], b=30, h=60, enrobage=4.0,
            jeu_entre_lits=2.5)),
    ("étriers Ø6", poutre("ET6", [sec("A", 40, 0, 35, ((2, 12),), ((2, 10),), (("etrier", 6),), 15)], b=20, h=35)),
    ("étrier Ø12 + deux épingles Ø8 (le Ø max place le lit 1)",
     poutre("ET12", [sec("A", 160, 0, 200, ((4, 20),), ((2, 12),), (("etrier", 12), ("epingle", 8), ("epingle", 8)), 15)],
            b=35, h=55)),
    ("τ dans la bande I (pas besoin d'étriers)", poutre("TAU-I", [sec("A", 80, 0, 50, ((3, 16),), pas=25)], b=30, h=50)),
    ("τ dans la bande II (besoin d'étriers)", poutre("TAU-II", [sec("A", 80, 0, 100, ((3, 16),), pas=20)], b=30, h=50)),
    ("τ dans la bande IV (barres inclinées : warn)", poutre("TAU-IV", [sec("A", 80, 0, 200, ((3, 16),), pas=10)], b=30, h=50)),
    ("τ au-delà de τ_adm,IV (inacceptable)", poutre("TAU-X", [sec("A", 80, 0, 300, ((3, 16),), pas=8)], b=30, h=50)),
    ("pas trop grand : pas > s_th", poutre("PAS", [sec("A", 80, 0, 100, ((3, 16),), et=(("etrier", 8),), pas=20)], b=30, h=50)),
    ("s_max gouverne : pas de 32 cm > 30 cm", poutre("SMAX", [sec("A", 80, 0, 50, ((3, 16),), pas=32)], b=30, h=50)),
    ("borne naturelle : pas = s_max = 30 cm", poutre("SMAX30", [sec("A", 80, 0, 50, ((3, 16),), pas=30)], b=30, h=50)),
    ("hauteur insuffisante (20 × 40, M = 100)", poutre("HMIN", [sec("A", 100, 0, 40, ((4, 20),))], b=20, h=40)),
    ("acier insuffisant (2Ø16 pour M = 150)", poutre("ASINS", [sec("A", 150, 0, 60, ((2, 16),))], b=30, h=50)),
    ("acier excessif : As > As,max (15 × 25, 6Ø20)", poutre("ASMAX", [sec("A", 20, 0, 10, ((6, 20),), ((2, 8),), (("etrier", 6),), 10)],
                                                         b=15, h=25)),
    ("As,min de la face sup. = 0,25·As,req de la face inf.", poutre("OPP", [sec("A", 200, 0, 0, ((4, 20),), ((2, 16),))], b=30, h=50)),
    ("géométrie impossible en flexion : y_G > h (h = 8 cm, quatre lits)",
     poutre("GEO", [sec("A", 5, 0, 2, L2 * 2, ((2, 8),), (("etrier", 6),), 10)], b=20, h=8)),
    ("géométrie impossible au cisaillement : h = 5 cm, lit 1 à 5,0 cm (d = 0)",
     poutre("GEO5", [sec("A", 0, 0, 3, ((2, 16),), ((2, 16),), (("etrier", 10),), 10)], b=20, h=5)),
    # d ≤ 0 donne d = 0,1 cm dans les formules : s_max = 0,075 cm refuse déjà tout pas réaliste, As,req explose dès
    # que M > 0 ; ces deux cas sont ceux où SEUL le contrôle de d refuse (pas de 0,05 cm ; aucun moment)
    ("d = 0 et pas de 0,05 cm : seul le contrôle de d refuse le pas",
     poutre("GEO5P", [sec("A", 0, 0, 3, ((2, 16),), ((2, 16),), (("etrier", 10),), 0.05)], b=20, h=5)),
    ("y_G > h sans moment (h = 6 cm, quatre lits 2Ø6) : seul le contrôle de d refuse la face",
     poutre("GEO6", [sec("A", 0, 0, 0, ((2, 6),) * 4, ((2, 6),), (("etrier", 6),), 10)], b=20, h=6)),
    ("y_G imposés en bas et en haut", poutre("YG", [sec("A", 120, 80, 90, ((3, 20), (2, 16)), ((3, 16),), pas=20,
                                                       yG_inf=7.37, yG_sup=4.83)], b=30, h=55)),
    ("enrobage nul : le module le lit 3,0 cm", poutre("E0", [sec("A", 90, 20, 70, ((3, 16),), ((2, 12),), pas=20)],
                                                   b=30, h=50, enrobage=0.0)),
    ("grande poutre 100 × 120, Ø32 et Ø40",
     poutre("XL", [sec("A", 1500, 400, 900, ((6, 32), (6, 32), (4, 40)), ((4, 25),),
                       (("etrier", 12), ("etrier", 12), ("epingle", 12), ("epingle", 12)), 12)], b=100, h=120)),
    ("petite poutre 15 × 25, étrier Ø6", poutre("XS", [sec("A", 8, 0, 15, ((2, 10),), ((2, 8),), (("etrier", 6),), 10)],
                                                b=15, h=25)),
    ("valeurs non entières (22,5 × 47,5 ; M = 37,25 ; pas 17,5)",
     poutre("DEC", [sec("A", 37.25, 12.75, 48.4, ((3, 12),), ((2, 10),), (("etrier", 8),), 17.5)], b=22.5, h=47.5,
            enrobage=2.75, jeu_premier_lit=0.25, jeu_entre_lits=1.25)),
    ("égalité M_inf = M_sup sur les mêmes lits : l'inférieure gouverne la hauteur",
     poutre("EGAL", [sec("A", 80, 80, 60, ((3, 16),), ((3, 16),), pas=20)], b=30, h=50)),
    ("M_sup gouverne la hauteur", poutre("MSUP", [sec("A", 50, 150, 100, ((2, 16),), ((4, 20),), pas=15)], b=30, h=55)),
    ("cinq sections aux statuts mêlés (ok, warn, nok, V = 0)",
     poutre("MIX", [sec("A", 80, 0, 50, ((3, 16),), pas=25), sec("B", 80, 0, 200, ((3, 16),), pas=10),
                    sec("C", 150, 0, 60, ((2, 16),)), sec("D", 60, 0, 0, ((3, 12),)),
                    sec("E", 0, 120, 140, ((2, 12),), ((4, 16),), (("etrier", 8),), 12)], b=30, h=50)),
    ("deux sections, poutre « warn » (barres inclinées, rien de non conforme)",
     poutre("WARN", [sec("A", 80, 0, 200, ((3, 16),), pas=10), sec("B", 50, 0, 40, ((3, 16),), pas=25)], b=30, h=50)),
]


def aleatoire(rng, k):
    """Une poutre tirée au sort (graine fixe) : dimensions, matériaux, jeux, lits, étriers, efforts."""
    b, h = float(rng.choice(range(15, 65, 5))), float(rng.choice(range(25, 105, 5)))
    beton, fyk = rng.choice(sorted(CL)), rng.choice((400, 500))
    p = {"nom": f"Aléa {k}", "b": b, "h": h, "beton": beton, "fyk": fyk, "gamma_s": rng.choice((1.15, 1.3, 1.5)),
         "enrobage": rng.choice((2.0, 2.5, 3.0, 3.5, 4.0, 5.0)), "jeu_premier_lit": rng.choice((0.0, 0.5, 1.0)),
         "jeu_entre_lits": rng.choice((0.0, 1.0, 1.5, 2.0)), "sections": []}
    m_ref = (8.5 * h) ** 2 * CL[beton]["alpha_b"] * 10 * b * CL[beton][f"mu_a{fyk}"] / 1e6     # ≈ M qui remplit h
    v_ref = 0.064 * CL[beton]["fck_cube"] / 1.05 * 0.75 * b * h * 100 / 1e3                    # V à τ_adm,IV
    for i in range(rng.randint(1, 3)):
        def lits():
            return [(rng.randint(2, 6), rng.choice(E.DIAMETRES[1:-1])) for _ in range(rng.choices((1, 2, 3, 4), (6, 3, 1, 1))[0])]
        et = [("etrier" if j == 0 else rng.choice(("etrier", "epingle")), rng.choice(E.DIAMETRES_ETRIERS))
              for j in range(rng.randint(1, 3))]
        p["sections"].append(sec("ABC"[i], round(rng.uniform(0, 1.3 * m_ref), 2) if rng.random() < 0.85 else 0.0,
                                 round(rng.uniform(0, 0.6 * m_ref), 2) if rng.random() < 0.5 else 0.0,
                                 round(rng.uniform(0, 1.2 * v_ref), 1) if rng.random() < 0.85 else 0.0,
                                 lits(), lits(), et, float(rng.choice(range(5, 36))),
                                 yG_inf=round(rng.uniform(3.5, 9.0), 2) if rng.random() < 0.1 else None))
    return M.normaliser(p)


RNG = random.Random(20261106)
CAS += [(f"tirage {k:02d} (graine 20261106)", aleatoire(RNG, k)) for k in range(1, 25)]


def cas(debut):
    """La poutre du cas dont la description commence par ``debut``."""
    return next(p for d, p in CAS if d.startswith(debut))

# ------------------------------------------------------------------ 1. données
print("=== 1. Données : classes de béton, défauts du module ===")
FICHIER = json.loads(lire("beton_classes.json"))
chk("classes_beton() rend beton_classes.json, lu depuis le paquet (beton/poutre/../../)",
    M.classes_beton() == FICHIER and os.path.samefile(E.CHEMIN_CLASSES, os.path.join(RACINE, "beton_classes.json")))
copie = M.classes_beton()
copie["C30/37"]["alpha_b"] = 1.0
chk("classes_beton() rend une copie : la modifier ne change pas le calcul",
    M.classes_beton() == FICHIER and M.verifier(M.nouvelle_poutre())["materiaux"]["alpha_b"] == FICHIER["C30/37"]["alpha_b"])

try:
    logging.disable(logging.WARNING)
    import streamlit as st  # noqa: E402
    import modules.poutre as P  # noqa: E402
    P.BETON_DATA = json.loads(lire("beton_classes.json"))
except ImportError as e:  # pragma: no cover — la parité EXIGE le module
    st = P = None
    chk("streamlit et modules/poutre.py se chargent (indispensables à la parité)", False, e)

if P is not None:
    KB, KS = P.KB, P.KS
    ENTETE = lire("modules", "poutre.py")[:200]
    chk(f"la référence de parité est bien celle du moteur : {M.REFERENCE}",
        "VERSION 2.40" in ENTETE and P.APP_VERSION == "1.2.41", (ENTETE[:60], P.APP_VERSION))
    chk("les listes du moteur sont celles du module (Ø, Ø d'étriers, lits par face)",
        list(E.DIAMETRES) == P.DIAM_OPTS and list(E.DIAMETRES_ETRIERS) == P.SHEAR_DIAM_OPTS and E.MAX_LITS == P.MAX_LITS)

    def lire_section(sid):
        """Une section telle que l'état du module la porte, au format du moteur."""
        ss = st.session_state
        lits = {face: [{"n": ss[KS(f"n_as_{face}" + ("" if j == 1 else f"_l{j}"), 1, sid)],
                        "d": ss[KS(f"ø_as_{face}" + ("" if j == 1 else f"_l{j}"), 1, sid)]}
                       for j in range(1, ss[KS(f"nlits_{face}", 1, sid)] + 1)] for face in ("inf", "sup")}
        return {"nom": ss[f"meta_b1_nom_{sid}"], "M_inf": ss[KS("M_inf", 1, sid)], "M_sup": ss[KS("M_sup", 1, sid)],
                "V": ss[KS("V", 1, sid)], "inf": lits["inf"], "sup": lits["sup"],
                "etriers": [{"type": {"Étrier": "etrier", "Épingle": "epingle"}[ss[KS(f"shear_line{i}_type", 1, sid)]],
                             "d": ss[KS(f"shear_line{i}_d", 1, sid)]} for i in range(ss[KS("shear_n_lines", 1, sid)])],
                "pas": ss[KS("shear_pas", 1, sid)], "yG_inf": None, "yG_sup": None}

    st.session_state.clear()
    P._ensure_global_defaults()
    P._init_beams_if_needed()
    ss = st.session_state
    defaut_module = {"nom": ss["meta_beam_nom_1"], "b": ss[KB("b", 1)], "h": ss[KB("h", 1)], "beton": ss[KB("beton", 1)],
                     "fyk": ss[KB("fyk", 1)], "gamma_s": ss["gamma_s"], "enrobage": ss[KB("enrobage_beton", 1)],
                     "jeu_premier_lit": ss["jeu_enrobage_cm"], "jeu_entre_lits": ss["jeu_entre_lits_cm"],
                     "sections": [lire_section(1)]}
    chk("nouvelle_poutre() = l'état du module au premier affichage (20 × 40, C30/37, B500, γs 1,5, 2Ø16, étrier Ø10…)",
        not ecarts_json(M.nouvelle_poutre(), defaut_module), ecarts_json(M.nouvelle_poutre(), defaut_module))
    P._add_section(1)
    chk("nouvelle_section() = la section que le bouton ➕ du module ajoute, et nom_section_suivant() = sa lettre",
        not ecarts_json(M.nouvelle_section(M.nom_section_suivant(M.nouvelle_poutre())), lire_section(2)),
        ecarts_json(M.nouvelle_section(M.nom_section_suivant(M.nouvelle_poutre())), lire_section(2)))
    for noms in (["A", "B", "D"], ["B"], [chr(65 + i) for i in range(26)], [" A ", "C"]):
        st.session_state.beams[0]["sections"] = [{"id": i + 1, "nom": n} for i, n in enumerate(noms)]
        for i, n in enumerate(noms):
            st.session_state[f"meta_b1_nom_{i + 1}"] = n
        attendu = P._next_section_name(1)
        obtenu = M.nom_section_suivant({"sections": [{"nom": n} for n in noms]})
        chk(f"nom_section_suivant({noms if len(noms) < 5 else 'A…Z'}) = {attendu!r} comme _next_section_name",
            obtenu == attendu, obtenu)

# ------------------------------------------------------------------ 2. à la main
print("\n=== 2. Contrôles à la main (formules du bureau écrites en N et en mm) ===")


def arr5(x_mm):
    """Arrondi aux 5 mm supérieurs."""
    return math.ceil(x_mm / 5.0) * 5.0


def proche(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b))


def fr(x, nd):
    """Un nombre à la française, pour les libellés."""
    return f"{x:.{nd}f}".replace(".", ",")


# H1 : la section A de V1 — C30/37 (fck 30, fck,cube 37, α_b 12,96, μ_a500 0,1709), B500, γs 1,5, 300 × 500 mm,
#      enrobage 30 mm, étrier Ø8, 3Ø20 en bas, 2Ø16 en haut, M_inf = 122 kN·m, V = 60 kN, pas 200 mm
r1 = M.verifier(cas("V1 du site"))["sections"][0]
g1 = r1["grandeurs"]
v1 = {v["cle"]: v for v in r1["verifications"]}
fyd = 500 / 1.5
e_inf, e_sup = 30 + arr5(8 + 20 / 2), 30 + arr5(8 + 16 / 2)           # 50 et 50 mm
hu = math.sqrt(122e6 / (12.96 * 300 * 0.1709))                          # mm
As_req = 122e6 / (fyd * 0.9 * (500 - e_inf))                            # mm²
fctm = 0.30 * 30 ** (2 / 3)
As_min_inf = max(0.26 * fctm / 500 * 300 * 500, 0.0013 * 300 * 500, 0.25 * 0.0)
As_min_sup = max(0.26 * fctm / 500 * 300 * 500, 0.0013 * 300 * 500, 0.25 * As_req)
As3, As2 = 3 * math.pi * 20 ** 2 / 4, 2 * math.pi * 16 ** 2 / 4
tau, tau_1 = 60e3 / (0.75 * 300 * 500), 0.016 * 37 / 1.05
s_th = 2 * math.pi * 8 ** 2 / 4 * fyd * (500 - min(e_inf, e_sup)) / 60e3    # mm
s_lim = min(s_th, min(0.75 * (500 - min(e_inf, e_sup)), 300.0))
chk("H1 lits : axe à 50 mm en bas (30 + ⌈8 + 10⌉₅) et en haut (30 + ⌈8 + 8⌉₅)",
    proche(g1["dist_lit1_inf"] * 10, e_inf) and proche(g1["dist_lit1_sup"] * 10, e_sup), (g1["dist_lit1_inf"], g1["dist_lit1_sup"]))
chk(f"H1 hauteur : h_min = √(M/(α_b·b·μ)) + y_G = {fr(hu / 10, 2)} + 5,0 = {fr(hu / 10 + 5, 2)} cm, taux h_min/h",
    proche(g1["h_min"], hu / 10 + e_inf / 10) and proche(v1["hauteur"]["taux"], (hu + e_inf) / 500), g1["h_min"])
chk(f"H1 As,req = M/(fyd·0,9·d) = {fr(As_req, 1)} mm², taux = As,req/As = {fr(As_req / As3, 4)}",
    proche(g1["As_req_inf"], As_req) and proche(v1["inf"]["taux"], As_req / As3), (g1["As_req_inf"], v1["inf"]["taux"]))
chk(f"H1 As,min : {fr(As_min_inf, 2)} mm² en bas, {fr(As_min_sup, 2)} mm² en haut (0,25·As,req inf. = {fr(0.25 * As_req, 2)})",
    proche(g1["As_min_inf"], As_min_inf) and proche(g1["As_min_sup"], As_min_sup) and proche(v1["sup"]["taux"], As_min_sup / As2),
    (g1["As_min_inf"], g1["As_min_sup"]))
chk(f"H1 τ = V/(0,75·b·h) = {fr(tau, 4)} ≤ τ_adm,I = 0,016·37/1,05 = {fr(tau_1, 4)} N/mm² : bande I, taux τ/τ_adm,I",
    proche(g1["tau"], tau) and g1["bande_tau"] == "I" and proche(v1["tau"]["taux"], tau / tau_1) and v1["tau"]["etat"] == "ok")
chk(f"H1 s_th = Ast·fyd·d/V = {fr(s_th / 10, 3)} cm, s_lim = min(s_th ; 30 cm), taux = 20/s_lim",
    proche(g1["s_th"] * 10, s_th) and proche(g1["s_lim"] * 10, s_lim) and proche(v1["pas"]["taux"], 200 / s_lim)
    and g1["s_th_tronque"] == math.floor(s_th) / 10, (g1["s_th"], g1["s_th_tronque"]))

# H2 : audit C-1 — 300 × 600, trois lits 2Ø16 en bas (d₁ = 80 mm), un lit en haut (d₁ = 50 mm) ;
#      M_inf = 150 < M_sup = 160 kN·m, mais c'est M_inf qui exige le plus : 47,5 + 8,0 = 55,5 cm
r2 = M.verifier(cas("audit C-1"))["sections"][0]
g2 = r2["grandeurs"]
e1 = 30 + arr5(10 + 8)                                                   # étrier Ø10 par défaut
lits = [e1, e1 + arr5(8) + 10 + arr5(8), e1 + 2 * (arr5(8) + 10 + arr5(8))]
hu150, hu160 = (math.sqrt(m * 1e6 / (12.96 * 300 * 0.1709)) for m in (150, 160))
chk("H2 lits du bas à 50, 80, 110 mm ; y_G = 80 mm", [round(x["distance"] * 10, 9) for x in r2["geometrie"]["inf"]["lits"]] == lits
    and proche(g2["yG_inf"] * 10, sum(lits) / 3), r2["geometrie"]["inf"]["lits"])
chk(f"H2 la famille inférieure gouverne : {fr(hu150 / 10, 2)} + 8,0 = {fr((hu150 + 80) / 10, 2)} cm > {fr((hu160 + 50) / 10, 2)} cm",
    g2["face_hauteur"] == "inf" and proche(g2["h_min"], (hu150 + 80) / 10) and g2["M_hauteur"] == 150, (g2["face_hauteur"], g2["h_min"]))
tau2 = 120e3 / (0.75 * 300 * 600)
chk(f"H2 τ = {fr(tau2, 4)} N/mm² entre τ_adm,I et τ_adm,II = 0,032·37/1,05 : bande II, état ok",
    proche(g2["tau"], tau2) and g2["bande_tau"] == "II" and proche(g2["tau_lim"], 0.032 * 37 / 1.05))

# ------------------------------------------------------------------ 3. parité
ICONES = {"🟢": "ok", "🟡": "warn", "🔴": "nok"}
BESOINS = {"Pas besoin d’étriers": "pas_d_etriers", "Besoin d’étriers": "etriers",
           "Besoin de barres inclinées et d’étriers": "barres_inclinees_et_etriers", "Pas acceptable": "inacceptable"}
LIMITES = {"I": "τ_adm_I", "II": "τ_adm_II", "IV": "τ_adm_IV", "hors": "τ_adm_IV"}
TAU_MODULE = {"τ_adm_I": "tau_1", "τ_adm_II": "tau_2", "τ_adm_IV": "tau_4"}
RE_TAU = re.compile(r"^τ = (\d+\.\d\d) N/mm² ≤ (τ_adm_(?:I|II|IV)) = (\d+\.\d\d) N/mm² → (.+)$")
RE_HMIN = re.compile(r"^Hauteur minimale de la poutre : hᵤ,min \+ d₁ = ([\d,]+) \+ ([\d,]+) = \*\*([\d,]+) cm\*\*$")
# grandeur du moteur ↔ clé de _dimensionnement_compute_states
CORRESPONDANCE = (
    ("beton", "beton"), ("fyk", "fyk"), ("gamma_s", "gamma_s"), ("fyd", "fyd"), ("alpha_b", "alpha_b"), ("mu", "mu_val"),
    ("fctm", "fctm"), ("b", "b"), ("h", "h"), ("M_inf", "M_inf_val"), ("M_sup", "M_sup_val"), ("V", "V_val"),
    ("dist_lit1_inf", "dist_l1_inf"), ("dist_lit1_sup", "dist_l1_sup"), ("yG_inf", "e_cdg_inf"), ("yG_sup", "e_cdg_sup"),
    ("d_inf", "d_utile_inf"), ("d_sup", "d_utile_sup"), ("d_cisaillement", "d_utile_shear"),
    ("geometrie_inf_ok", "geom_inf_ok"), ("geometrie_sup_ok", "geom_sup_ok"), ("geometrie_cisaillement_ok", "geom_shear_ok"),
    ("M_hauteur", "M_max"), ("hu_min", "hmin_calc"), ("yG_hauteur", "e_cdg_gov"), ("h_min", "h_min_poutre"),
    ("tau_1", "tau_1"), ("tau_2", "tau_2"), ("tau_4", "tau_4"),
    ("As_min_ec", "As_min_ec"), ("As_min_plancher", "As_min_plancher"), ("As_min_base", "As_min_formula"), ("As_max", "As_max"),
    ("As_req_inf", "As_formule_inf"), ("As_req_sup", "As_formule_sup"), ("As_req_inf", "As_req_inf_final"),
    ("As_req_sup", "As_req_sup_final"), ("As_min_inf", "As_min_inf_eff"), ("As_min_sup", "As_min_sup_eff"),
    ("As_inf", "As_inf_total"), ("As_sup", "As_sup_total"))


class FauxSt:
    """Un « st » qui ENREGISTRE ce que l'écran du module afficherait. L'état reste le vrai st.session_state ;
    les widgets rendent leur valeur de session (rien n'est saisi)."""

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


def charger(p):
    """L'état du module pour la poutre p (au format du moteur), puis le début d'un run de son écran
    (_ensure_global_defaults, _init_beams_if_needed : défauts, migrations, coercitions)."""
    ss = st.session_state
    ss.clear()
    P._ensure_global_defaults()
    P._init_beams_if_needed()
    ss.beams[0]["sections"] = [{"id": i + 1, "nom": s["nom"]} for i, s in enumerate(p["sections"])]
    ss["meta_beam_nom_1"] = p["nom"]
    ss["gamma_s"], ss["jeu_enrobage_cm"], ss["jeu_entre_lits_cm"] = p["gamma_s"], p["jeu_premier_lit"], p["jeu_entre_lits"]
    for cle_module, cle in (("b", "b"), ("h", "h"), ("beton", "beton"), ("fyk", "fyk"), ("enrobage_beton", "enrobage")):
        ss[KB(cle_module, 1)] = p[cle]
    for i, s in enumerate(p["sections"]):
        sid = i + 1
        ss[f"meta_b1_nom_{sid}"] = s["nom"]
        for k in ("M_inf", "M_sup", "V"):
            ss[KS(k, 1, sid)] = s[k]
        for face in ("inf", "sup"):
            ss[KS(f"nlits_{face}", 1, sid)] = len(s[face])
            for j, lit in enumerate(s[face], start=1):
                suf = "" if j == 1 else f"_l{j}"
                ss[KS(f"n_as_{face}{suf}", 1, sid)] = lit["n"]
                ss[KS(f"ø_as_{face}{suf}", 1, sid)] = lit["d"]
            if s[f"yG_{face}"] is not None:  # le champ « CDG (cm) » saisi à la française
                ss[KS(f"ycdg_{face}", 1, sid)] = repr(float(s[f"yG_{face}"])).replace(".", ",")
        ss[KS("shear_n_lines", 1, sid)] = len(s["etriers"])
        for j, e in enumerate(s["etriers"]):
            ss[KS(f"shear_line{j}_type", 1, sid)] = {"etrier": "Étrier", "epingle": "Épingle"}[e["type"]]
            ss[KS(f"shear_line{j}_d", 1, sid)] = e["d"]
        ss[KS("shear_pas", 1, sid)] = s["pas"]
    P._ensure_global_defaults()
    P._init_beams_if_needed()


def module(p):
    """Le module sur la poutre p : son calcul, section par section, puis ce que son écran affiche."""
    charger(p)
    calc = []
    for sid, s in enumerate(p["sections"], start=1):
        r = P._dimensionnement_compute_states(1, sid, P.BETON_DATA)
        calc.append({"r": r, "Ast": P._shear_lines_total_Ast_mm2(1, sid), "d_etrier": P._stirrup_diam_mm(1, sid),
                     "resume": P._shear_lines_summary(1, sid), "pas": float(st.session_state[KS("shear_pas", 1, sid)]),
                     "dist": {f: [P._get_dist_lit(1, sid, f, j) for j in range(1, len(s[f]) + 1)] for f in ("inf", "sup")},
                     "yG_auto": {f: P._layers_geometry(1, sid, f, use_manual=False)[1] for f in ("inf", "sup")},
                     "impose": {f: not st.session_state.get(KS(f"ycdg_auto_{f}", 1, sid), True) for f in ("inf", "sup")}})
    faux = FauxSt()
    vrai_st, vrai_bloc = P.st, P.open_bloc_left_right

    def espion(left, right="", etat="ok", pct=None):
        faux.ev.append(("bloc", left, right, etat, pct))
        return vrai_bloc(left, right, etat, pct)

    P.st, P.open_bloc_left_right = faux, espion
    try:
        P.render_dimensionnement_right(P.BETON_DATA)
    finally:
        P.st, P.open_bloc_left_right = vrai_st, vrai_bloc
    titre_poutre, ecrans = None, []
    for ev in faux.ev:
        if ev[0] == "exp":
            if titre_poutre is None:
                titre_poutre = ev[1]
            else:
                ecrans.append({"titre": ev[1], "blocs": [], "md": []})
        elif ecrans:
            ecrans[-1]["blocs" if ev[0] == "bloc" else "md"].append(ev[1:])
    for c, e in zip(calc, ecrans):
        c["ecran"] = e
    return titre_poutre, calc


def comparer(es, md, ecarts):
    """Une section du moteur (es) contre le module (md : calcul et écran) ; les écarts vont dans ``ecarts``."""
    def cmp(nom, a, b):
        if not egal(a, b):
            ecarts.append(f"{es['nom']}.{nom} : moteur {a!r} ≠ module {b!r}")
    g, r, geo = es["grandeurs"], md["r"], es["geometrie"]
    for cle_m, cle_r in CORRESPONDANCE:
        cmp(cle_m, g[cle_m], r[cle_r])
    for face in ("inf", "sup"):
        cmp(f"d_calc_{face}", g[f"d_calc_{face}"], max(r[f"d_utile_{face}"], 0.1))
        cmp(f"besoin_{face}", g[f"besoin_{face}"], max(r[f"As_req_{face}_final"], r[f"As_min_{face}_eff"]))
        cmp(f"libelle_{face}", geo[face]["libelle"], r[f"{face}_detail"])
        cmp(f"yG_calcule_{face}", g[f"yG_calcule_{face}"], md["yG_auto"][face])
        cmp(f"yG_impose_{face}", g[f"yG_impose_{face}"], md["impose"][face])
        cmp(f"lits_{face}", len(geo[face]["lits"]), len(md["dist"][face]))
        for j, (lit, dist) in enumerate(zip(geo[face]["lits"], md["dist"][face]), start=1):
            cmp(f"distance lit {j} {face}", lit["distance"], dist)
    # les noms que le moteur AJOUTE doivent désigner ce que le module a retenu
    fh = g["face_hauteur"]
    cmp("face_hauteur (M, y_G de la famille retenue par le module)",
        (0.0, g["yG_inf"]) if fh is None else (g[f"M_{fh}"], g[f"yG_{fh}"]), (r["M_max"], r["e_cdg_gov"]))
    for face, opp in (("inf", "sup"), ("sup", "inf")):
        terme = {"ec": r["As_min_ec"], "plancher": r["As_min_plancher"], "opposee": 0.25 * r[f"As_formule_{opp}"]}
        cmp(f"As_min_terme_{face} (le terme égal à As,min du module)", terme[g[f"As_min_terme_{face}"]], r[f"As_min_{face}_eff"])
        cmp(f"lits {face} : Σ As du lit = As de la face", sum(x["As"] for x in geo[face]["lits"]), r[f"As_{face}_total"])
    for x in es["verifications"]:
        cmp(f"motifs {x['cle']} (vides si et seulement si l'état est ok)", not x["motifs"], x["etat"] == "ok")
    cmp("d_calc_cisaillement", g["d_calc_cisaillement"], max(r["d_utile_shear"], 0.1))
    cmp("Ast", g["Ast"], md["Ast"])
    cmp("d_etrier", g["d_etrier"], md["d_etrier"])
    cmp("resume étriers", geo["etriers"]["resume"], md["resume"])
    cmp("pas", g["pas"], md["pas"])
    v = {x["cle"]: x for x in es["verifications"]}
    for cle, cle_r in (("hauteur", "etat_h"), ("inf", "etat_inf"), ("sup", "etat_sup"), ("tau", "etat_tau"), ("pas", "etat_pas")):
        cmp(f"état {cle}", v[cle]["etat"] if cle in v else "ok", r[cle_r])
    cmp("statut", es["statut"], r["etat_global"])
    # ---- l'écran
    e = md["ecran"]
    cmp("icône de la section", es["statut"], ICONES.get(e["titre"][:1]))
    cmp("titre de la section", f"Section {es['nom']}", e["titre"][2:])
    cmp("blocs", [x["nom"] for x in es["verifications"]], [b[0] for b in e["blocs"]])
    blocs = {b[0]: b for b in e["blocs"]}
    rien = (None, "", None, float("nan"))
    for x in es["verifications"]:
        _, droite, etat, pct = blocs.get(x["nom"], rien)
        cmp(f"% {x['cle']}", x["taux"], pct / 100.0)
        cmp(f"état affiché {x['cle']}", x.get("etat_bande", x["etat"]), etat)
        if x["cle"] == "hauteur":
            cmp("bandeau hauteur", f"{g['beton']} • Section {g['b']:.0f}×{g['h']:.0f} cm", droite)
        if x["cle"] in ("inf", "sup"):
            cmp(f"bandeau {x['cle']}", f"B{int(g['fyk'])} • {x['libelle']} ({x['limite']:.0f} mm²)", droite)
        if x["cle"] == "pas":
            cmp("bandeau des étriers", geo["etriers"]["resume"], droite)
    lignes = {}
    for body, aide in e["md"]:
        if RE_TAU.match(body):
            lignes["tau"] = RE_TAU.match(body).groups()
        elif RE_HMIN.match(body):
            lignes["hmin"] = RE_HMIN.match(body).groups()
        elif body.startswith("**Pas théorique = "):
            lignes["s_th"] = (body[len("**Pas théorique = "):].split(" ")[0], re.search(r"→ s,th = ([\d,]+) cm", aide).group(1))
        elif body.startswith("**Pas maximal = "):
            lignes["s_max"] = body[len("**Pas maximal = "):].split(" ")[0]
        elif body.startswith("**Asw = "):
            lignes["Asw"] = body[len("**Asw = "):].split(" ")[0]
    cmp("ligne hᵤ,min + d₁ = …", (P._fr(g["hu_min"], 1), P._fr(g["yG_hauteur"], 1), P._fr(g["h_min"], 1)), lignes.get("hmin"))
    if g["V"] > 0:
        tau_txt, nom_lim, _, besoin_txt = lignes.get("tau", ("", "τ_adm_I", "", ""))
        cmp("bande de τ", LIMITES[g["bande_tau"]], nom_lim)
        cmp("besoin (τ)", g["besoin_tau"], BESOINS.get(besoin_txt))
        cmp("τ_adm de la bande", g["tau_lim"], r[TAU_MODULE[nom_lim]])
        cmp("τ (code d'écran)", g["tau"], blocs.get(M.NOMS["tau"], rien)[3] / 100.0 * r[TAU_MODULE[nom_lim]])
        cmp("τ affiché", f"{g['tau']:.2f}", tau_txt)
        cmp("s_lim (code d'écran)", g["s_lim"], md["pas"] / (blocs.get(M.NOMS["pas"], rien)[3] / 100.0))
        cmp("s_th tronqué (écran, aide)", (f"{g['s_th_tronque']:.1f}", P._fr(g["s_th_tronque"], 1)), lignes.get("s_th"))
        cmp("s_max tronqué", f"{g['s_max_tronque']:.1f}", lignes.get("s_max"))
        cmp("Asw affiché", f"{g['Ast']:.0f}", lignes.get("Asw"))
    else:
        cmp("sans V : ni τ ni étriers à l'écran", [k for k in ("tau", "s_th", "s_max", "Asw") if k in lignes], [])


def faits(p, res, titre_poutre, calc):
    """Ce que le cas atteint (couverture), lu dans les sorties CONTRÔLÉES contre le module."""
    out = {f"béton {p['beton']}", f"B{p['fyk']}", f"γs {p['gamma_s']}", f"enrobage {p['enrobage']}",
           f"jeu 1 {p['jeu_premier_lit']}", f"jeu L {p['jeu_entre_lits']}", f"poutre {ICONES.get(titre_poutre[:1])}"}
    if len(p["sections"]) > 1:
        out.add("plusieurs sections")
    for s, es, md in zip(p["sections"], res["sections"], calc):
        g = es["grandeurs"]
        v = {x["cle"]: x for x in es["verifications"]}
        out |= {f"lits {len(s['inf'])}", f"lits {len(s['sup'])}", f"statut {es['statut']}", f"hauteur {g['face_hauteur']}"}
        out |= {f"étrier Ø{e['d']}" for e in s["etriers"]} | {e["type"] for e in s["etriers"]}
        out |= {f"{len(s['etriers'])} lignes"} if len(s["etriers"]) > 1 else set()
        out |= {f"motif {m}" for x in v.values() for m in x["motifs"]}
        out |= {f"état {k} {x['etat']}" for k, x in v.items()} | {f"taux {k} > 1" for k, x in v.items() if x["taux"] > 1}
        out |= {f"As,min {g[f'As_min_terme_{f}']}" for f in ("inf", "sup")}
        out |= {f"y_G imposé {f}" for f in ("inf", "sup") if md["impose"][f]}
        if not s["M_inf"] and not s["M_sup"]:
            out.add("sans moment")
        if not s["V"]:
            out.add("V = 0")
        else:
            out.add(f"bande {g['bande_tau']}")
            out.add("s_lim = s_th" if g["s_th"] < g["s_max"] else "s_lim = s_max")
            if v["tau"]["etat_bande"] != v["tau"]["etat"]:
                out.add("écran τ ≠ état (d = 0)")
        blocs = {b[0]: b for b in md["ecran"]["blocs"]}
        for x in v.values():  # une borne exacte : le pourcentage vaut 100 % pile et l'état reste « ok » (≤, pas <)
            if blocs[x["nom"]][3] == 100.0 and x["etat"] != "nok":
                out.add(f"borne {x['cle']}" + (f" {g['bande_tau']}" if x["cle"] == "tau" else ""))
    if res["avertissements"]:
        out |= {f"avertissement {a['code']}" for a in res["avertissements"]}
    return out


def avec(p, **sections):
    """Une copie de p dont la première section reçoit ``sections`` (et la poutre, les clés b/h)."""
    q = copy.deepcopy(p)
    for k, x in sections.items():
        (q if k in ("b", "h") else q["sections"][0])[k] = x
    return q


if P is not None:
    print("\n=== 3. Parité avec modules/poutre.py, cas par cas (calcul + écran) ===")
    # les bornes EXACTES, construites avec le module lui-même : pas = s_lim, h = h_min, τ = τ_adm,I/II/IV
    base = cas("pas trop grand")
    charger(base)
    r0 = P._dimensionnement_compute_states(1, 1, P.BETON_DATA)
    d_sh = max(r0["d_utile_shear"], 0.1)                               # le code d'écran du module, mot pour mot
    pas_th = P._shear_lines_total_Ast_mm2(1, 1) * r0["fyd"] * (d_sh * 10.0) / (100 * 1e3) / 10.0
    CAS.append(("borne exacte : pas = s_th (construit avec le module)", avec(base, pas=min(pas_th, min(0.75 * d_sh, 30.0)))))
    CAS.append(("borne exacte : h = h_min (construit avec le module)", avec(base, h=r0["h_min_poutre"])))
    for bande, k in (("I", 0.016), ("II", 0.032), ("IV", 0.064)):
        cible, den = k * 37 / 1.05, 0.75 * 30.0 * 50.0 * 100
        V0 = cible * den / 1e3
        essais = [V0]
        for sens in (math.inf, -math.inf):
            V = V0
            for _ in range(64):
                V = math.nextafter(V, sens)
                essais.append(V)
        V = next((x for x in essais if x * 1e3 / den == cible), None)
        if V is not None:
            CAS.append((f"borne exacte : τ = τ_adm,{bande} (V = {V!r})", avec(base, V=V, pas=5.0)))
        else:
            print(f"     (aucun V flottant ne donne τ = τ_adm,{bande} exactement pour 30 × 50 : borne non construite)")

    COMPAREES = {c for _, c in CORRESPONDANCE} | {"inf_detail", "sup_detail", "etat_global", "etat_h", "etat_inf",
                                                  "etat_sup", "etat_tau", "etat_pas"}
    charger(M.nouvelle_poutre())
    rendues = set(P._dimensionnement_compute_states(1, 1, P.BETON_DATA))
    chk(f"chacune des {len(rendues)} clés rendues par _dimensionnement_compute_states est comparée au moteur",
        rendues == COMPAREES, sorted(rendues ^ COMPAREES))
    COUV = set()
    for k, (desc, p) in enumerate(CAS):
        res = M.verifier(p)
        titre_poutre, calc = module(p)
        ecarts = []
        for es, md in zip(res["sections"], calc):
            comparer(es, md, ecarts)
        # la poutre : statut (icône du module), taux maximal et vérification gouvernante (le plus grand % de l'écran)
        if not egal(res["statut"], ICONES.get(titre_poutre[:1])):
            ecarts.append(f"statut de la poutre : moteur {res['statut']} ≠ module {titre_poutre}")
        pcts = [(md["ecran"]["titre"][2:], b[0], b[3]) for md in calc for b in md["ecran"]["blocs"]]
        gouv = max(pcts, key=lambda x: x[2])
        if not (egal(res["taux_max"], gouv[2] / 100.0) and (f"Section {res['gouvernant']['section']}", res["gouvernant"]["nom"]) == gouv[:2]):
            ecarts.append(f"gouvernant : moteur {res['gouvernant']} ≠ écran {gouv}")
        COUV |= faits(p, res, titre_poutre, calc)
        n_g = sum(len(es["grandeurs"]) for es in res["sections"])
        chk(f"cas {k + 1:02d} · {desc} — {len(p['sections'])} section(s), {n_g} grandeurs, états, % de l'écran",
            not ecarts, ecarts[:6])
    chk(f"{len(CAS)} poutres ({sum(len(p['sections']) for _, p in CAS)} sections) comparées : au moins 40",
        len(CAS) >= 40)
    print(f"     {ECART['n']} comparaisons numériques : {ECART['bit']} identiques au bit près, écart relatif maximal "
          f"{ECART['max']:.2e} (tolérance {TOL:g})")
    chk("l'écart relatif maximal reste sous 1e-9 (c'est le même calcul)", ECART["max"] <= TOL, ECART["max"])

    print("\n=== 4. Couverture : ce que les cas atteignent réellement ===")
    ATTENDUS = (
        ["bande I", "bande II", "bande IV", "bande hors", "V = 0", "sans moment", "plusieurs sections"]
        + [f"état {k} {e}" for k in ("hauteur", "inf", "sup", "pas") for e in ("ok", "nok")]
        + ["état tau ok", "état tau warn", "état tau nok", "statut ok", "statut warn", "statut nok",
           "poutre ok", "poutre warn", "poutre nok"]
        + [f"motif {m}" for m in ("hauteur_insuffisante", "d_utile_negatif", "acier_insuffisant", "acier_excessif",
                                  "barres_inclinees", "cisaillement_excessif", "pas_trop_grand")]
        + [f"taux {k} > 1" for k in ("hauteur", "inf", "sup", "tau", "pas")]
        + ["As,min ec", "As,min plancher", "As,min opposee", "s_lim = s_th", "s_lim = s_max",
           "hauteur inf", "hauteur sup", "hauteur None", "écran τ ≠ état (d = 0)"]
        + ["lits 1", "lits 2", "lits 3", "lits 4", "epingle", "3 lignes"] + [f"étrier Ø{d}" for d in E.DIAMETRES_ETRIERS]
        + [f"béton {b}" for b in CL] + ["B400", "B500", "γs 1.15", "γs 1.35", "γs 1.5"]
        + ["enrobage 0.0", "enrobage 2.5", "enrobage 4.0", "enrobage 5.0", "jeu 1 0.5", "jeu 1 1.0", "jeu L 0.0",
           "jeu L 2.0", "jeu L 2.5", "y_G imposé inf", "y_G imposé sup", "avertissement enrobage_nul"]
        + ["borne pas", "borne hauteur", "borne tau I", "borne tau II", "borne tau IV"])
    manque = [a for a in ATTENDUS if a not in COUV]
    chk(f"les cas atteignent les {len(ATTENDUS)} situations attendues (bandes, états, motifs, termes, bornes…)",
        not manque, manque)

    print("\n=== 5. Sélecteur d'armatures : chaque disposition repassée par le module ===")
    for desc, nom, face in (("V1 du site", "A", "inf"), ("V1 du site", "B", "sup"), ("même travée en B400", "A", "inf"),
                            ("audit C-1", "A", "inf"), ("y_G imposés", "A", "sup"), ("cinq sections", "E", "sup")):
        p = cas(desc)
        cands = M.candidats_armatures(p, nom, face)
        sid = [s["nom"] for s in p["sections"]].index(nom) + 1
        ecarts = []
        for c in cands:
            q = copy.deepcopy(p)
            q["sections"][sid - 1][face] = [{"n": c["n"], "d": c["d"]}]
            charger(q)
            r = P._dimensionnement_compute_states(1, sid, P.BETON_DATA)
            for nom_c, a, b in (("aire", c["aire"], r[f"As_{face}_total"]),
                                ("taux", c["taux"], max(r[f"As_req_{face}_final"], r[f"As_min_{face}_eff"]) / r[f"As_{face}_total"]),
                                ("état", c["etat"], r[f"etat_{face}"]), ("statut", c["statut_section"], r["etat_global"]),
                                ("distance", c["distance"], P._get_dist_lit(1, sid, face, 1)), ("yG", c["yG"], r[f"e_cdg_{face}"])):
                if not egal(a, b):
                    ecarts.append(f"{c['nom']}.{nom_c} : {a!r} ≠ {b!r}")
        couples = [(c["n"], c["d"]) for c in cands]
        forme = (len(cands) == 30 and sorted(couples) == [(n, d) for n in range(2, 7) for d in (10, 12, 16, 20, 25, 32)]
                 and all(d in P.DIAM_OPTS for _, d in couples)
                 and [c["aire"] for c in cands] == sorted(c["aire"] for c in cands))
        premier_ok = next((c for c in cands if c["ok"]), None)
        actuel = p["sections"][sid - 1][face]
        retenue = [f"{x['n']}Ø{x['d']}" for x in actuel if len(actuel) == 1 and 2 <= x["n"] <= 6 and x["d"] in M.CANDIDATS_D]
        marques = (sum(c["optimal"] for c in cands) == (premier_ok is not None) and (premier_ok is None or premier_ok["optimal"])
                   and [c["nom"] for c in cands if c["retenue"]] == retenue)
        chk(f"sélecteur {p['nom']} · section {nom} · {face} : 30 dispositions (n 2…6, Ø 10…32) triées par aire, "
            f"optimal = {premier_ok['nom'] if premier_ok else 'aucune'} ; aire, taux, états identiques au module",
            not ecarts and forme and marques, (ecarts[:4], forme, marques))

# ------------------------------------------------------------------ 6. format
print("\n=== 6. Format : JSON, normalisation, domaine de l'écran ===")
tous = [M.verifier(p) for _, p in CAS]
chk("la sortie de verifier() est du JSON strict (sans NaN) et se relit à l'identique",
    all(json.loads(json.dumps(r, allow_nan=False)) == r for r in tous))
chk("normaliser() est idempotente et {} donne nouvelle_poutre()",
    all(M.normaliser(M.normaliser(p)) == M.normaliser(p) for _, p in CAS) and M.normaliser({}) == M.nouvelle_poutre())
entree = copy.deepcopy(cas("V1 du site"))
M.verifier(entree)
M.candidats_armatures(entree, "A", "inf")
chk("verifier() et candidats_armatures() ne modifient pas la poutre reçue", entree == cas("V1 du site"))
chk("les types d'étriers du module (« Étrier », « Épingle ») sont admis",
    M.normaliser(avec(M.nouvelle_poutre(), etriers=[{"type": "Étrier", "d": 8}, {"type": "Épingle", "d": 8}]))
    ["sections"][0]["etriers"] == [{"type": "etrier", "d": 8}, {"type": "epingle", "d": 8}])
chk("sans nom, une section prend la première lettre libre",
    [s["nom"] for s in M.normaliser({"sections": [{"nom": "A"}, {}, {"nom": "C"}, {}]})["sections"]] == ["A", "B", "C", "D"])
REFUS = (("largeur nulle", {"b": 0}), ("hauteur négative", {"h": -40}), ("classe inconnue", {"beton": "C16/20"}),
         ("nuance 450", {"fyk": 450}), ("γs nul", {"gamma_s": 0}), ("enrobage négatif", {"enrobage": -1}),
         ("jeu négatif", {"jeu_entre_lits": -0.5}), ("largeur en texte", {"b": "30"}), ("hauteur booléenne", {"h": True}),
         ("largeur infinie", {"b": float("inf")}), ("clé inconnue", {"largeur": 30}), ("aucune section", {"sections": []}),
         ("moment négatif", {"sections": [{"M_inf": -10}]}), ("V négatif", {"sections": [{"V": -1}]}),
         ("Ø14 (non proposé)", {"sections": [{"inf": [{"n": 2, "d": 14}]}]}), ("aucun lit", {"sections": [{"inf": []}]}),
         ("cinq lits", {"sections": [{"inf": [{"n": 2, "d": 16}] * 5}]}), ("zéro barre", {"sections": [{"sup": [{"n": 0, "d": 16}]}]}),
         ("2,5 barres", {"sections": [{"sup": [{"n": 2.5, "d": 16}]}]}), ("étrier Ø14", {"sections": [{"etriers": [{"type": "etrier", "d": 14}]}]}),
         ("type « cadre »", {"sections": [{"etriers": [{"type": "cadre", "d": 8}]}]}), ("aucun étrier", {"sections": [{"etriers": []}]}),
         ("pas nul", {"sections": [{"pas": 0}]}), ("y_G nul", {"sections": [{"yG_inf": 0}]}),
         ("clé de section inconnue", {"sections": [{"Minf": 10}]}), ("clé de lit inconnue", {"sections": [{"inf": [{"n": 2, "o": 16}]}]}))
refusees = []
for nom, champ in REFUS:
    try:
        M.verifier(dict(M.nouvelle_poutre(), **champ))
        refusees.append(f"{nom} : accepté")
    except ValueError:
        pass
    except Exception as e:  # noqa: BLE001 — une autre exception n'est pas un refus : c'est une panne
        refusees.append(f"{nom} : {type(e).__name__} {e}")
chk(f"{len(REFUS)} saisies hors du domaine de l'écran lèvent ValueError (jamais de substitution silencieuse)",
    not refusees, refusees)
for nom, appel in (("face inconnue", lambda: M.candidats_armatures(M.nouvelle_poutre(), "A", "milieu")),
                   ("section inconnue", lambda: M.candidats_armatures(M.nouvelle_poutre(), "Z", "inf"))):
    try:
        appel()
        chk(f"candidats_armatures : {nom} → ValueError", False)
    except ValueError:
        chk(f"candidats_armatures : {nom} → ValueError", True)
r_e0 = M.verifier(dict(cas("V1 du site"), enrobage=0.0))
r_e3 = M.verifier(dict(cas("V1 du site"), enrobage=3.0))
chk("enrobage nul : avertissement « enrobage_nul » et calcul à 3,0 cm (comme le module) ; aucun à 3,0 cm",
    [a["code"] for a in r_e0["avertissements"]] == ["enrobage_nul"] and not r_e3["avertissements"]
    and r_e0["sections"][0]["grandeurs"]["enrobage_calcul"] == 3.0
    and r_e0["sections"][0]["grandeurs"]["h_min"] == r_e3["sections"][0]["grandeurs"]["h_min"])
# « formule = valeurs = résultat » : chaque résultat se retrouve à partir des seuls termes que le moteur donne
FORMULES = {
    "hu_min": lambda t: math.sqrt(t["M"] * 1e6 / (t["alpha_b"] * t["b"] * t["mu"])) / 10,     # b en mm → cm
    "h_min": lambda t: t["hu_min"] + t["yG"],
    "As_req": lambda t: t["M"] * 1e6 / (t["fyd"] * t["k"] * t["d"] * 10),                     # d en cm → mm²
    "As_min_ec": lambda t: t["k"] * t["fctm"] / t["fyk"] * t["b"] * t["h"],
    "As_min_plancher": lambda t: t["k"] * t["b"] * t["h"],
    "As_min_opposee": lambda t: t["k"] * t["As_req_opp"],
    "As_min": lambda t: max(t["As_min_ec"], t["As_min_plancher"], t["As_min_opposee"]),
    "besoin": lambda t: max(t["As_req"], t["As_min"]),
    "As_max": lambda t: t["k"] * t["b"] * t["h"],
    "tau": lambda t: t["V"] * 1e3 / (t["k"] * t["b"] * t["h"]),
    "tau_lim": lambda t: t["k"] * t["fck_cube"] / t["gamma"],
    "s_th": lambda t: t["Ast"] * t["fyd"] * t["d"] / (t["V"] * 1e3),                           # d en cm → cm
    "s_max": lambda t: min(t["k"] * t["d"], t["plafond"]),
    "s_lim": lambda t: min(t["s_th"], t["s_max"]),
}
VALEUR_LIMITE = {"hauteur": ("h_min", None), "inf": ("besoin", None), "sup": ("besoin", None), "tau": ("tau", "tau_lim"),
                 "pas": (None, "s_lim")}
fautes, n_formules = [], 0
for (desc, p), res in zip(CAS, tous):
    for s in res["sections"]:
        for v in s["verifications"]:
            resultats = {}
            for f in v["formules"]:
                n_formules += 1
                t = {x["symbole"]: x["valeur"] for x in f["termes"]}
                resultats[f["cle"]] = f["resultat"]
                if not proche(FORMULES[f["cle"]](t), f["resultat"]):
                    fautes.append(f"{p['nom']}/{s['nom']}/{f['cle']} : {FORMULES[f['cle']](t)!r} ≠ {f['resultat']!r}")
                if "resultat_tronque" in f and f["resultat_tronque"] != math.floor(f["resultat"] * 10) / 10:
                    fautes.append(f"{p['nom']}/{s['nom']}/{f['cle']} : tronqué {f['resultat_tronque']!r}")
            cv, cl = VALEUR_LIMITE[v["cle"]]
            if (cv and v["valeur"] != resultats[cv]) or (cl and v["limite"] != resultats[cl]) or v["taux"] != v["valeur"] / v["limite"]:
                fautes.append(f"{p['nom']}/{s['nom']}/{v['cle']} : valeur, limite et taux ≠ formules")
chk(f"« formule = valeurs = résultat » : les {n_formules} formules se recalculent à partir de leurs seuls termes, "
    "et valeur / limite = taux", not fautes and n_formules > 1000, fautes[:4])

doc = lire("docs", "beton", "POUTRE_MOTEUR.md")
cles = sorted(tous[1]["sections"][0]["grandeurs"])
sans_doc = [k for k in cles if f"`{k}`" not in doc or k not in M.__doc__]
chk(f"les {len(cles)} grandeurs sont nommées dans la docstring du moteur ET dans docs/beton/POUTRE_MOTEUR.md",
    not sans_doc, sans_doc)

# ------------------------------------------------------------------ 7. pureté
print("\n=== 7. Pureté : le moteur se charge et calcule SANS streamlit, avec la seule bibliothèque standard ===")
def _importations(sous_dossier):
    """(fichier, module importé) pour chaque importation absolue des sources de beton/<sous_dossier>."""
    out = []
    for dossier, _, fichiers in os.walk(os.path.join(RACINE, "beton", sous_dossier)):
        for f in fichiers:
            if f.endswith(".py"):
                arbre = ast.parse(lire(os.path.relpath(os.path.join(dossier, f), RACINE)))
                for n in ast.walk(arbre):
                    noms = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                        [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
                    out += [(os.path.relpath(os.path.join(dossier, f), RACINE), x.split(".")[0]) for x in noms]
    return out


# le moteur POUTRE : la seule bibliothèque standard (le paquet beton/ porte aussi d'autres moteurs du site, dont
# beton/maturite qui emploie numpy, permis sur le site : la règle propre au moteur poutre ne s'étend pas à eux)
non_std = [f"{f}: {x}" for f, x in _importations("poutre") if x not in sys.stdlib_module_names | {"beton"}]
chk("aucun fichier de beton/poutre/ n'importe autre chose que la bibliothèque standard (lecture des sources)", not non_std, non_std)
# tout beton/ : jamais une bibliothèque absente du navigateur (règle du site, docs/site/MODULES_SITE.md §1)
REFUSEES = {"streamlit", "pandas", "matplotlib", "scipy", "reportlab", "PIL", "fitz", "requests"}
refusees = [f"{f}: {x}" for f, x in _importations("") if x in REFUSEES]
chk("aucun fichier de beton/ n'importe une bibliothèque absente du navigateur (streamlit, pandas, matplotlib…)",
    not refusees, refusees)

SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("streamlit", "pandas", "reportlab", "numpy", "matplotlib", "scipy", "pymupdf", "fitz")
class Refus:
    """Refuse tout chargement des bibliothèques interdites."""
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
from beton.poutre import moteur as M
poutres = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.verifier(p) for p in poutres],
          "cands": M.candidats_armatures(poutres[1], "A", "inf")}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"beton"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
poutres = [p for _, p in CAS]
attendu = {"res": json.loads(json.dumps(tous)), "cands": json.loads(json.dumps(M.candidats_armatures(poutres[1], "A", "inf")))}
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(poutres), capture_output=True, text=True,
                        check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans streamlit", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : streamlit, pandas, reportlab, numpy… REFUSÉS par sys.meta_path, et le moteur calcule quand même "
        f"({len(out['res'])} poutres, le sélecteur)", out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : le moteur n'a chargé que la bibliothèque standard", not out["tiers"], out["tiers"])
    diff = ecarts_json({"res": out["res"], "cands": out["cands"]}, attendu)
    chk(f"Python {v} : résultats identiques à ceux de ce processus", not diff, diff[:3])
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
