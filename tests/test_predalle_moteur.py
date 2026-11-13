# -*- coding: utf-8 -*-
"""Moteur « Prédalle béton armé » (beton/predalle) : la PARITÉ avec l'écran Streamlit modules/predalle.py v1.1, qui
reste la référence. Chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Données : défauts de l'écran LUS dans son état (h = 22, peau de 6 cm…), bornes de la peau (_h_pre_max) pour
     chaque épaisseur, listes, sections et renforts ajoutés.
  2. À la main : les cas de tests/test_predalle.py (l'écran) recalculés ici en cm et en mm : base principale DANS la
     peau sans jeu (3,5 cm), base secondaire SUR la peau (6,5 cm), renfort sur la peau (7,0 cm), audit C-1.
  3. Parité (plus de 60 notes, plusieurs prédalles et sections) : chaque note est chargée dans l'état de l'écran piloté
     SANS navigateur, puis on compare au moteur :
       a) TOUTES les clés de _dimensionnement_compute_states (direction par direction) et ses états ;
       b) chaque couche (_couche_as_per_m, _dist_couche_eff, _auto_dist_couche — LA règle prédalle —, _couche_label,
          _couche_diam_mm, le drapeau « automatique ») et chaque face (_layers_geometry, avec et sans CDG saisi) ;
       c) ce que l'ÉCRAN affiche : render_dimensionnement_right exécuté avec un faux « st » qui enregistre, et un espion
          sur open_bloc_left_right — titre, texte de droite, état et POURCENTAGE EXACT de chaque bloc ; lignes de la
          hauteur et du tranchant, Aₛ,req / Aₛ,min / Aₛ,max / Aₛ fourni ET leurs infobulles ; l'icône de chaque section
          et de chaque prédalle ;
       d) la NOTE PDF : ce que l'écran lui envoie (_valeurs_pour_export, axes FIGÉS) relu par
          modules/export_pdf_dalle._compute_section ;
       e) le FICHIER : ce que « 💾 Enregistrer » écrit (_build_save_payload), relu par beton/predalle/fichier.py.
  4. Couverture : états, motifs, termes de Aₛ,min, types, poses (dans / sur la prédalle, dessus), bornes…
  5. Sélecteur de la nappe de base : chaque nappe proposée est repassée par l'écran (le Ø déplace l'axe).
  6. Fichiers de l'écran : migration v1, peau ramenée dans ses bornes, fichier du module Dalle, remplacements muets.
  7. Format : JSON, normalisation (défauts, ValueError hors du domaine — la peau surtout), avertissements propres à la
     prédalle, formules recalculées depuis leurs seuls termes (axes compris).
  8. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, numpy… sous
     Python 3.12 et 3.13 s'il est installé.
  9. Mutations : altérations de la règle prédalle (beton/predalle/moteur.py) ET des formules partagées
     (beton/dalle/moteur.py, injecté dans le moteur prédalle) ; chacune fait rougir la parité de la section 3.

    /usr/bin/python3.12 tests/test_predalle_moteur.py      (streamlit requis pour la parité, reportlab pour la note)
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
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from beton import treillis as TRB  # noqa: E402
from beton.dalle import moteur as MD  # noqa: E402
from beton.predalle import entree as E  # noqa: E402
from beton.predalle import fichier as FI  # noqa: E402
from beton.predalle import moteur as M  # noqa: E402

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


def pre(nom, sections, **kw):
    p = {"nom": nom, "sections": sections}
    p.update(kw)
    return M.normaliser(p)


def note(*predalles, gamma_s=None, jeu=None):
    """Une NOTE de l'écran : ses prédalles partagent γs et le jeu premier lit (paramètres globaux de l'écran)."""
    out = [copy.deepcopy(p) for p in predalles]
    gs = out[0]["gamma_s"] if gamma_s is None else gamma_s
    j = out[0]["jeu_premier_lit"] if jeu is None else jeu
    for p in out:
        p["gamma_s"], p["jeu_premier_lit"] = gs, j
    return [M.normaliser(p) for p in out]


CL = M.classes_beton()
TP = sec("A", 30, 12, 30, 9, 60)                 # le cas 1 de tests/test_predalle.py
TP_R = sec("A", 30, 12, 30, 9, 60, inf_y=fa(tr("10/10/100/100"), nb(3, 12)))   # + renfort principal 3 Ø12
CAS = [  # (description, [prédalles de la note])
    ("défauts de l'écran (nouvelle_predalle)", note(M.nouvelle_predalle())),
    ("test_predalle 1 : Mx = My = 30, sup 12 / 9, V 60 (h 22, peau 6, Y principale)", note(pre("P1", [TP]))),
    ("test_predalle 2 : renfort inférieur principal 3 Ø12 posé SUR la prédalle", note(pre("P2", [TP_R]))),
    ("test_predalle 3 : même prédalle, direction principale X", note(pre("P3", [TP_R], dir_principale="X"))),
    ("audit C-1 : la direction secondaire (sur la peau) gouverne la hauteur", note(pre("C-1", [sec("A", 55, 0, 60, 0, 30)], h=20))),
    ("peau minimale 3 cm : la base Ø12 sort de la peau", note(pre("H3", [sec("A", 20, 5, 25, 6, 30, inf_y=fa(tr("12/12/150/150")))], h_pre=3.0))),
    ("peau maximale 15 cm sur 40 cm", note(pre("H15", [sec("A", 60, 20, 80, 25, 90)], h=40, h_pre=15.0))),
    ("peau = h − 1 (h 10, peau 9 cm) : un centimètre coulé en place", note(pre("H9", [sec("A", 4, 0, 5, 0, 8)], h=10, h_pre=9.0))),
    ("peau fractionnaire 5,5 cm, base Ø20/150 principale (4,0 cm)",
     note(pre("H55", [sec("A", 25, 10, 40, 12, 50, inf_y=fa(ba(20, 150)))], h_pre=5.5))),
    ("renforts partout : inf. secondaire, sup. principale et secondaire",
     note(pre("RT", [sec("A", 35, 18, 42, 22, 70, inf_x=fa(tr("8/8/150/150"), nb(4, 10)), sup_y=fa(tr("8/8/150/150"), ba(10, 200)),
                         sup_x=fa(tr("6/6/150/150"), nb(2, 12)))]))),
    ("axe saisi sur la base secondaire (8,0 cm) et sur un renfort (9,5 cm)",
     note(pre("AX", [sec("A", 28, 12, 35, 9, 60, inf_x=fa(tr("10/10/100/100", dist=8.0)), inf_y=fa(tr("10/10/100/100"), nb(3, 12, dist=9.5)))]))),
    ("axe saisi sur la base principale (dans la peau : 4,2 cm)",
     note(pre("AXP", [sec("A", 28, 12, 35, 9, 60, inf_y=fa(tr("10/10/100/100", dist=4.2)))]))),
    ("CDG saisi : inf. Y 5,25 cm, sup. X 3,0 cm",
     note(pre("YG", [sec("A", 28, 12, 35, 9, 60, inf_y=fa(tr("10/10/100/100"), nb(3, 12), yG=5.25), sup_x=fa(yG=3.0))]))),
    ("quatre couches (le maximum) mêlant les trois types sur la nappe principale",
     note(pre("Q4", [sec("A", 45, 10, 60, 15, 70, inf_y=fa(tr("12/12/150/150"), ba(10, 200, dist=5.0), nb(4, 16), tr("6/6/150/150")))], h=25))),
    ("base « n barres » 8 Ø10 en bas, barres Ø8/200 en haut",
     note(pre("NB", [sec("A", 20, 5, 22, 6, 30, inf_x=fa(nb(8, 10)), inf_y=fa(nb(8, 10)), sup_x=fa(ba(8, 200)), sup_y=fa(ba(8, 200)))]))),
]
for beton in CL:  # les cinq bétons
    CAS.append((f"béton {beton}, V = 90 kN", note(pre(f"B-{beton}", [sec("A", 30, 15, 42, 18, 90)], beton=beton, h=24))))
CAS += [
    ("B400 (μ_a400)", note(pre("B400", [TP], fyk=400))),
    ("γs = 1,15", note(pre("GS115", [TP]), gamma_s=1.15)),
    ("γs aux bornes de l'écran : 1,0", note(pre("GS1", [sec("A", 30, 0, 30, 0, 40)]), gamma_s=1.0)),
    ("γs aux bornes de l'écran : 2,0", note(pre("GS2", [sec("A", 30, 0, 30, 0, 40)]), gamma_s=2.0)),
    ("jeu premier lit 0,5 cm (il ne touche que les nappes supérieures)", note(pre("J05", [TP_R]), jeu=0.5)),
    ("jeu premier lit 1,0 cm, enrobage 2,5 cm", note(pre("J10", [TP], enrobage=2.5), jeu=1.0)),
    ("enrobage nul : l'écran le lit 3,0 cm", note(pre("E0", [TP], enrobage=0.0))),
    ("enrobage 5 cm (la base principale à 5,5 cm, sur une peau de 6)", note(pre("E5", [TP], enrobage=5.0))),
    # (la règle du 13/11/2026 : barres Ø20 en haut, réel 20 + 1,0 = 21,0, calcul ⌈21,0 + 1⌉ = 22,0 → d = 22 − 22 = 0 ≤ 0)
    ("enrobage 20 cm (borne) sur 22 cm : d ≤ 0 en haut", note(pre("E20", [sec("A", 5, 5, 5, 5, 10, sup_y=fa(ba(20, 150)))],
                                                               enrobage=20.0))),
    ("largeur de bande minimale (20 cm) et « n barres »", note(pre("B20", [sec("A", 8, 0, 9, 0, 15, inf_y=fa(nb(2, 10)))], b=20))),
    ("largeur de bande maximale (500 cm)", note(pre("B500", [sec("A", 140, 60, 175, 45, 300, inf_y=fa(tr("10/10/100/100"), nb(12, 12)),
                                                                sup_x=fa(ba(10, 125)))], b=500))),
    ("bande de 150 cm", note(pre("B150", [sec("A", 40, 15, 52, 20, 80, inf_y=fa(nb(9, 12)))], b=150))),
    ("épaisseur minimale (5 cm, peau 3) : hauteur insuffisante", note(pre("H5", [sec("A", 2, 0, 3, 0, 5)], h=5, h_pre=3.0))),
    ("épaisseur maximale (100 cm, peau 15)", note(pre("H100", [sec("A", 300, 120, 450, 200, 600, inf_y=fa(ba(25, 100)), inf_x=fa(ba(20, 100)))],
                                                       h=100, h_pre=15.0))),
    ("τ au-delà de τ_adm,I (non conforme)", note(pre("TAU", [sec("A", 20, 0, 25, 0, 150)]))),
    ("acier insuffisant", note(pre("ASINS", [sec("A", 45, 0, 60, 0, 40)]))),
    ("acier excessif : As > As,max (bande de 20 cm, h = 8 cm, Ø25/100)",
     note(pre("ASMAX", [sec("A", 1, 0, 1, 0, 2, inf_x=fa(ba(25, 100)), inf_y=fa(ba(25, 100)), sup_x=fa(ba(6, 250)), sup_y=fa(ba(6, 250)))],
              b=20, h=8, h_pre=4.0))),
    ("sans moment : la hauteur prend l'axe inf. X (sur la peau)", note(pre("SM", [sec("A", V=20)]))),
    ("trois sections (travée, appui, rive)", note(pre("T3", [TP, sec("B", 10, 35, 12, 40, 80, sup_y=fa(tr("10/10/100/100"), nb(4, 12))),
                                                                  sec("C", 5, 0, 8, 0, 15)]))),
    ("trois prédalles dans la note, l'une non conforme", note(pre("T1", [TP]), pre("T2", [sec("A", 80, 0, 90, 0, 40)], h=18),
                                                               pre("T3", [sec("A", 5, 0, 6, 0)], b=50, fyk=400, h_pre=4.5), jeu=0.5)),
]


def aleatoire(rng, k):
    """Une prédalle tirée au sort (graine fixe) : dimensions, peau, matériaux, couches (1 à 4 par face, trois types, axes
    et y_G saisis parfois), sollicitations de part et d'autre de la capacité."""
    b, h = rng.choice((20, 50, 100, 100, 100, 120, 150, 250, 500)), rng.choice((12, 15, 16, 18, 20, 22, 25, 30, 40))
    h_pre = rng.choice((3.0, 4.0, 5.0, 5.5, 6.0, 6.0, 7.0, 8.0, E.h_pre_max(h)))
    h_pre = min(h_pre, E.h_pre_max(h))
    beton, fyk = rng.choice(sorted(CL)), rng.choice((400, 500))
    m_ref = (0.8 * h) ** 2 * CL[beton]["alpha_b"] * 10 * b * CL[beton][f"mu_a{fyk}"] / 1e6 / 100
    v_ref = 0.016 * CL[beton]["fck_cube"] / 1.05 * 0.75 * b * h * 100 / 1e3

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
    return M.normaliser({"nom": f"Aléa {k}", "b": b, "h": h, "h_pre": h_pre, "beton": beton, "fyk": fyk,
                         "dir_principale": rng.choice("XY"), "enrobage": rng.choice((2.0, 2.5, 3.0, 3.5, 4.0)),
                         "gamma_s": rng.choice((1.15, 1.3, 1.5)), "jeu_premier_lit": rng.choice((0.0, 0.0, 0.5, 1.0)), "sections": secs})


RNG = random.Random(20261107)
CAS += [(f"tirage {k:02d} (graine 20261107)", note(aleatoire(RNG, k))) for k in range(1, 25)]
CAS += [(f"tirage multiple {k:02d} : deux prédalles dans la note", note(aleatoire(RNG, 100 + k), aleatoire(RNG, 200 + k)))
        for k in range(1, 7)]


def cas(debut):
    return next(n for d, n in CAS if d.startswith(debut))


# ------------------------------------------------------------------ 1. données
print("=== 1. Données : défauts de l'écran, bornes de la peau, listes ===")
try:
    logging.disable(logging.WARNING)
    import streamlit as st  # noqa: E402
    import modules.predalle as D  # noqa: E402
    D.BETON_DATA = json.loads(lire("beton_classes.json"))
except ImportError as e:  # pragma: no cover — la parité EXIGE l'écran
    st = D = None
    chk("streamlit et modules/predalle.py se chargent (indispensables à la parité)", False, e)
try:
    import modules.export_pdf_dalle as XP  # noqa: E402 — la note PDF (reportlab), commune aux dalles et prédalles
except ImportError as e:  # pragma: no cover
    XP = None
    chk("modules/export_pdf_dalle.py se charge (reportlab : la note est la seconde copie du calcul)", False, e)

TYPES_ECRAN = {"treillis": "Treillis", "barres": "Barres", "n_barres": "n barres"}
DIR_TXT = lambda x: repr(float(x)).replace(".", ",")  # noqa: E731 — un nombre saisi à la française, exact


def lire_session(did):
    """La prédalle ``did`` telle que l'état de l'écran la porte (après ses synchronisations), au format du moteur."""
    ss = st.session_state
    d = next(x for x in ss.predalles if int(x["id"]) == did)
    out = {"nom": ss[f"meta_predalle_nom_{did}"], "b": ss[D.KD("b", did)], "h": ss[D.KD("h", did)], "h_pre": float(ss[D.KD("h_pre", did)] or 6.0),
           "beton": ss[D.KD("beton", did)], "fyk": ss[D.KD("fyk", did)], "enrobage": ss[D.KD("enrobage_beton", did)],
           "dir_principale": ss[D.KD("dir_principale", did)], "gamma_s": ss["gamma_s"], "jeu_premier_lit": ss["jeu_enrobage_cm"],
           "sections": []}
    for s in d["sections"]:
        sid = int(s["id"])
        S = {"nom": ss[f"meta_pre{did}_nom_{sid}"], **{k: ss[D.KS(k, did, sid)] for k in E.MOMENTS + ("V",)}, "faces": {}}
        for which in E.FACES:
            couches = []
            for i in range(1, D._get_ncouches(did, sid, which) + 1):
                typ, des, dd, esp, n = D._couche_data(did, sid, which, i)
                auto = bool(ss.get(D.KS(f"dist_auto_{which}_c{i}", did, sid), True))
                try:
                    saisi = float(str(ss.get(D.KS(f"dist_axe_{which}_c{i}", did, sid), "") or "").strip().replace(",", "."))
                except ValueError:
                    saisi = None
                couches.append({"type": {v: k for k, v in TYPES_ECRAN.items()}[typ], "treillis": des, "d": dd,
                                "esp": int(ss[D.KS(f"esp_barres_{which}_c{i}", did, sid)]), "n": n,
                                "dist": None if auto or saisi is None or saisi <= 0 else saisi})
            S["faces"][which] = {"couches": couches, "yG": D._ycdg_manual(did, sid, which)}
        out["sections"].append(S)
    return out


def synchroniser():
    """Les synchronisations de render_dimensionnement_section : axes de chaque couche, puis CDG de chaque face."""
    for d in st.session_state.predalles:
        did = int(d["id"])
        for s in d["sections"]:
            sid = int(s["id"])
            for which in E.FACES:
                for j in range(1, D._get_ncouches(did, sid, which) + 1):
                    D._sync_dist_state(did, sid, which, j)
                D._sync_ycdg_state(did, sid, which)


if D is not None:
    ENTETE = lire("modules", "predalle.py")[:400]
    chk(f"la référence de parité est bien celle du moteur : {M.REFERENCE}",
        "PRÉDALLE EN BÉTON ARMÉ" in ENTETE and D.PREDALLE_VERSION == "1.1" and M.REFERENCE.endswith(D.PREDALLE_VERSION), D.PREDALLE_VERSION)
    chk("les listes du moteur sont celles de l'écran (Ø, types, couches, directions, faces)",
        list(E.DIAMETRES) == D.DIAM_OPTS and [E.LIBELLES_TYPES[t] for t in E.TYPES] == D.TYPES_ARMATURE
        and E.MAX_COUCHES == D.MAX_COUCHES and E.DIRECTIONS == D.DIR_KEYS and E.FACES == D.FACES_DIR)
    chk("les bornes de la peau sont celles de l'écran (3 et 15 cm)", (E.H_PRE_MIN, E.H_PRE_MAX) == (D.H_PRE_MIN, D.H_PRE_MAX))
    hs = [x / 2 for x in range(2 * E.H_MIN, 2 * E.H_MAX + 1)] + [4.0, 3.5, 16.0, 16.25]
    diff = [h for h in hs if E.h_pre_max(h) != D._h_pre_max(h)]
    chk(f"h_pre_max(h) = _h_pre_max(h) de l'écran pour {len(hs)} épaisseurs (5 à 100 cm, demi-centimètres)", not diff, diff[:5])
    st.session_state.clear()
    D._ensure_global_defaults()
    D._init_dalles_if_needed()
    synchroniser()
    defaut = lire_session(1)
    # (la règle de position du 13/11/2026 : le jeu de pose du bureau vaut 1 cm — l'écran disait 0 — et l'arrondi est une
    #  clé nouvelle ; tout le reste est l'état de l'écran)
    neuve = M.nouvelle_predalle()
    chk("nouvelle_predalle() : le jeu de pose du bureau (1 cm) et l'arrondi (0,5 cm) de la règle du 13/11/2026",
        neuve["jeu_premier_lit"] == 1.0 and neuve["arrondi"] == 0.5 and defaut["jeu_premier_lit"] == 0.0,
        (neuve["jeu_premier_lit"], neuve.get("arrondi")))
    neuve = {k: v for k, v in neuve.items() if k not in ("jeu_premier_lit", "arrondi")}
    defaut.pop("jeu_premier_lit")
    chk("nouvelle_predalle() = l'état de l'écran au premier affichage (bande 100 × 22, peau 6, C30/37, B500, Y…), hors jeu "
        "de pose et arrondi", not ecarts_json(neuve, defaut), ecarts_json(neuve, defaut))
    D._add_section(1)
    D._add_couche(1, 2, "inf_y")
    synchroniser()
    s2 = lire_session(1)["sections"][1]
    attendu = M.nouvelle_section(M.nom_section_suivant(M.nouvelle_predalle()))
    attendu["faces"]["inf_y"]["couches"].append(M.nouvelle_couche(base=False))
    chk("nouvelle_section() = la section que « ➕ » ajoute, nouvelle_couche(False) = le renfort que « ＋ » ajoute (3 Ø12)",
        not ecarts_json(attendu, s2), ecarts_json(attendu, s2))
    for noms in (["A", "B", "D"], ["B"], [chr(65 + i) for i in range(26)], [" A ", "C"]):
        st.session_state.predalles[0]["sections"] = [{"id": i + 1, "nom": n} for i, n in enumerate(noms)]
        for i, n in enumerate(noms):
            st.session_state[f"meta_pre1_nom_{i + 1}"] = n
        attendu = D._next_section_name(1)
        obtenu = M.nom_section_suivant({"sections": [{"nom": n} for n in noms]})
        chk(f"nom_section_suivant({noms if len(noms) < 5 else 'A…Z'}) = {attendu!r} comme _next_section_name", obtenu == attendu, obtenu)

# ------------------------------------------------------------------ 2. à la main
print("\n=== 2. Contrôles à la main (les cas de tests/test_predalle.py) ===")


def proche(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b))


def fr(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


# La règle du 13/11/2026 (beton/position.py) : position réelle (c + Ø/2 ; un renfort inférieur sur la peau, h_p + Ø/2),
# axe de calcul ⌈réel + jeu de pose⌉ au 0,5 cm — le jeu (1 cm) pour TOUTE couche, celles de la peau comprises. Les deux
# BASES inférieures sont DANS la peau (pose « predalle ») ; le moteur seul les pose au même niveau (la base secondaire
# posée sur la principale, c'est le site : tests/test_api_web_dalle_positions.py), et le dit (« nappes_confondues »).
r1 = M.verifier(cas("test_predalle 1")[0])["sections"][0]
g1 = r1["grandeurs"]
fyd = 500 / 1.5
As10 = math.pi * 10 ** 2 / 4 * 1000 / 100                          # treillis 10/10/100/100 : 785,4 mm²/m
e_peau = math.ceil((30 + 5 + 10) / 5) * 5                          # mm : ⌈3,0 + Ø10/2 + 1 (jeu)⌉ = 4,5 cm
chk("H1 les deux bases inférieures DANS la peau : réel 3 + 0,5 = 3,5 cm, calcul 3,5 + 1 (jeu) = 4,5 cm",
    [r1["faces"][f]["couches"][0]["e_reel"] for f in ("inf_y", "inf_x")] == [3.5, 3.5]
    and [r1["faces"][f]["couches"][0]["e"] for f in ("inf_y", "inf_x")] == [4.5, 4.5]
    and [r1["faces"][f]["couches"][0]["pose"] for f in ("inf_y", "inf_x")] == ["predalle", "predalle"])
req = {f: m * 1e6 / (fyd * 0.9 * (220 - e_peau)) for f, m in (("inf_y", 30), ("inf_x", 30), ("sup_y", 9), ("sup_x", 12))}
chk("H1 As,req = M / (f_yd · 0,9 · (220 − 45)) : 571 / 571 / 171 / 229 mm²",
    all(proche(g1[f"As_req_{f}"], req[f]) for f in req) and [round(req[f]) for f in ("inf_y", "inf_x", "sup_y", "sup_x")] == [571, 571, 171, 229],
    {f: g1[f"As_req_{f}"] for f in req})
hu = math.sqrt(30e6 / (12.96 * 1000 * 0.1709))                     # mm
chk(f"H1 hauteur : hu,min(30) + 4,5 = {fr(hu / 10 + 4.5, 2)} cm ≤ 22 (« 16,1 » à l'écran)",
    proche(g1["h_min"], hu / 10 + 4.5) and g1["M_hauteur"] == 30 and fr(g1["h_min"], 1) == "16,1", g1["h_min"])
r2 = M.verifier(cas("test_predalle 2")[0])["sections"][0]
c2 = r2["faces"]["inf_y"]["couches"]
yG = (As10 * 4.5 + 3 * math.pi * 144 / 4 * 8.0) / (As10 + 3 * math.pi * 144 / 4)
As_req2 = 30e6 / (fyd * 0.9 * (220 - yG * 10))
chk(f"H2 renfort 3 Ø12 SUR la peau : réel 6 + 0,6 = 6,6 cm, calcul ⌈7,6⌉ = 8,0 cm ; y_G = {fr(yG, 3)} (« 5,6 »), "
    f"As,req = {fr(As_req2, 0)} mm²",
    c2[1]["e_reel"] == 6.6 and c2[1]["e"] == 8.0 and c2[1]["pose"] == "sur_predalle" and proche(r2["grandeurs"]["yG_inf_y"], yG)
    and fr(yG, 1) == "5,6" and round(r2["grandeurs"]["As_req_inf_y"]) == 608
    and r2["faces"]["inf_y"]["libelle"] == "Treillis 10/10/100/100 + 3 Ø12", (c2[1]["e"], r2["grandeurs"]["As_req_inf_y"]))
r3 = M.verifier(cas("test_predalle 3")[0])["sections"][0]
chk("H3 X principale : les deux bases dans la peau (4,5 au calcul), le renfort Y sur la peau (8,0)",
    [r3["faces"]["inf_x"]["couches"][0]["e"], r3["faces"]["inf_y"]["couches"][0]["e"], r3["faces"]["inf_y"]["couches"][1]["e"]] == [4.5, 4.5, 8.0])
r4 = M.verifier(cas("audit C-1")[0])["sections"][0]
chk("H4 audit C-1 (moteur seul, bases au même niveau) : la famille au plus grand hu,min + y_G gouverne, inf. Y : "
    "16,5 + 4,5 = 21,0 cm > h = 20 cm",
    r4["grandeurs"]["face_hauteur"] == "inf_y" and (fr(r4["grandeurs"]["hu_min"], 1), fr(r4["grandeurs"]["yG_hauteur"], 1),
                                                    fr(r4["grandeurs"]["h_min"], 1)) == ("16,5", "4,5", "21,0")
    and r4["verifications"][0]["etat"] == "nok", (r4["grandeurs"]["face_hauteur"], r4["grandeurs"]["h_min"]))
r5 = M.verifier(cas("peau fractionnaire")[0])["sections"][0]
chk("H5 base principale Ø20 : réel 3,0 + 1,0 = 4,0 cm, calcul ⌈5,0⌉ = 5,0 ; base secondaire Ø10 : réel 3,5, calcul 4,5",
    (r5["faces"]["inf_y"]["couches"][0]["e_reel"], r5["faces"]["inf_y"]["couches"][0]["e"]) == (4.0, 5.0)
    and (r5["faces"]["inf_x"]["couches"][0]["e_reel"], r5["faces"]["inf_x"]["couches"][0]["e"]) == (3.5, 4.5))
r6 = M.verifier(cas("jeu premier lit 0,5")[0])["sections"][0]
chk("H6 le jeu de pose (0,5) vaut pour TOUTES les couches, la peau comprise : sup. ⌈3,5 + 0,5⌉ = 4,0 ; inf. 4,0 / 4,0 ; "
    "le renfort sur la peau ⌈6,6 + 0,5⌉ = 7,5",
    r6["faces"]["sup_y"]["couches"][0]["e"] == 4.0 and r6["faces"]["sup_x"]["couches"][0]["e"] == 4.0
    and [r6["faces"]["inf_y"]["couches"][0]["e"], r6["faces"]["inf_x"]["couches"][0]["e"], r6["faces"]["inf_y"]["couches"][1]["e"]] == [4.0, 4.0, 7.5])
chk("H7 coulé en place = h − peau, rendu par le moteur (22 − 6 = 16 cm)", g1["h_pre"] == 6.0 and g1["h_coule"] == 16.0)

# ------------------------------------------------------------------ 3. parité
ICONES = {"🟢": "ok", "🟡": "warn", "🔴": "nok"}
CORR_HAUT = (
    ("etat_h", "etat_hauteur"), ("etat_inf", "etat_inf"), ("etat_sup", "etat_sup"), ("etat_tau", "etat_tau"),
    ("principale", "principale"), ("V_val", "V"), ("tau", "tau"), ("tau_adm", "tau_adm"), ("M_max", "M_hauteur"),
    ("hmin_calc", "hu_min"), ("e_cdg_gov", "yG_hauteur"), ("h_min_dalle", "h_min"), ("tau_1", "tau_1"), ("tau_2", "tau_2"),
    ("tau_4", "tau_4"), ("fyd", "fyd"), ("gamma_s", "gamma_s"), ("fyk", "fyk"), ("alpha_b", "alpha_b"), ("mu_val", "mu"),
    ("beton", "beton"), ("b", "b"), ("h", "h"), ("fctm", "fctm"), ("As_min_ec", "As_min_ec"),
    ("As_min_plancher", "As_min_plancher"), ("As_max", "As_max"))
CORR_DIR = (
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


def charger(predalles):
    """L'état de l'écran pour la note ``predalles`` (format du moteur), puis le début d'un run (_ensure_global_defaults,
    _init_dalles_if_needed) et les synchronisations des champs « Dist. axe » et « CDG »."""
    ss = st.session_state
    ss.clear()
    D._ensure_global_defaults()
    # la POSITION de chaque couche : la règle du 13/11/2026 (beton/position.py — position réelle, jeu de pose, arrondi ;
    # la base secondaire dans la peau), que l'écran ne connaît pas ; il reçoit l'axe de calcul du moteur comme un axe
    # saisi (« Dist. axe »), comme la note PDF — la parité porte alors sur tout le reste (formules, états, écran) ; les
    # positions sont prouvées à la main (section 2, tests/test_position.py, tests/test_api_web_dalle_positions.py)
    axes = [[{f: [c["e"] for c in sec_["faces"][f]["couches"]] for f in E.FACES} for sec_ in M.verifier(p_)["sections"]]
            for p_ in predalles]
    ss["gamma_s"], ss["jeu_enrobage_cm"] = predalles[0]["gamma_s"], predalles[0]["jeu_premier_lit"]
    ss.predalles = [{"id": k + 1, "nom": p["nom"], "sections": [{"id": i + 1, "nom": s["nom"]} for i, s in enumerate(p["sections"])]}
                    for k, p in enumerate(predalles)]
    for k, p in enumerate(predalles):
        did = k + 1
        ss[f"meta_predalle_nom_{did}"] = p["nom"]
        for cle_ecran, cle in (("b", "b"), ("h", "h"), ("h_pre", "h_pre"), ("enrobage_beton", "enrobage"), ("beton", "beton"),
                               ("fyk", "fyk"), ("dir_principale", "dir_principale")):
            ss[D.KD(cle_ecran, did)] = p[cle]
        for i, s in enumerate(p["sections"]):
            sid = i + 1
            ss[f"meta_pre{did}_nom_{sid}"] = s["nom"]
            for x in E.MOMENTS + ("V",):
                ss[D.KS(x, did, sid)] = s[x]
            for which in E.FACES:
                f = s["faces"][which]
                ss[D.KS(f"ncouches_{which}", did, sid)] = len(f["couches"])
                for j, c in enumerate(f["couches"], start=1):
                    for cle_ecran, v in (("arm_type", TYPES_ECRAN[c["type"]]), ("treillis", c["treillis"]), ("ø_barres", c["d"]),
                                         ("esp_barres", c["esp"]), ("n_barres", c["n"])):
                        ss[D.KS(f"{cle_ecran}_{which}_c{j}", did, sid)] = v
                    ss[D.KS(f"dist_axe_{which}_c{j}", did, sid)] = DIR_TXT(axes[k][i][which][j - 1])
                    ss[D.KS(f"dist_auto_{which}_c{j}", did, sid)] = False
                if f["yG"] is not None:
                    ss[D.KS(f"ycdg_{which}", did, sid)] = DIR_TXT(f["yG"])
                    ss[D.KS(f"ycdg_auto_{which}", did, sid)] = False
    D._ensure_global_defaults()
    D._init_dalles_if_needed()
    synchroniser()


def ecran(predalles):
    """L'écran sur la note : le calcul (section par section), la note PDF et ce que render_dimensionnement_right affiche."""
    charger(predalles)
    calc = []
    export = D._valeurs_pour_export()
    for k, p in enumerate(predalles):
        for i, s in enumerate(p["sections"]):
            did, sid = k + 1, i + 1
            c = {"r": D._dimensionnement_compute_states(did, sid, D.BETON_DATA), "faces": {}}
            for which in E.FACES:
                As, yG, detail, As_pm = D._layers_geometry(did, sid, which)
                _, yG_auto, _, _ = D._layers_geometry(did, sid, which, use_manual=False)
                c["faces"][which] = {"As": As, "yG": yG, "detail": detail, "As_pm": As_pm, "yG_calcule": yG_auto,
                                     "yG_impose": D._ycdg_manual(did, sid, which) is not None,
                                     "couches": [{"As_pm": D._couche_as_per_m(did, sid, which, j), "e": D._dist_couche_eff(did, sid, which, j),
                                                  "e_auto": D._auto_dist_couche(did, sid, which, j), "libelle": D._couche_label(did, sid, which, j),
                                                  "d": D._couche_diam_mm(did, sid, which, j),
                                                  "auto": bool(st.session_state.get(D.KS(f"dist_auto_{which}_c{j}", did, sid), True))}
                                                 for j in range(1, D._get_ncouches(did, sid, which) + 1)]}
            if XP is not None:
                c["pdf"] = XP._compute_section(export, D.BETON_DATA, did, sid)
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
    titres, ecrans = [], []
    for ev in faux.ev:
        if ev[0] == "exp":
            ecrans.append({"titre": ev[1], "blocs": [], "md": []})
        elif ecrans:
            if ev[0] == "bloc":
                ecrans[-1]["blocs"].append({"bloc": ev[1:], "md": []})
            elif ecrans[-1]["blocs"] and not ev[1].startswith(('<div style="background-color', "</div>")):
                ecrans[-1]["blocs"][-1]["md"].append(ev[1:])
    out, k = [], 0
    for p in predalles:
        titres.append(ecrans[k]["titre"])
        out += ecrans[k + 1:k + 1 + len(p["sections"])]
        k += 1 + len(p["sections"])
    for c, e in zip(calc, out):
        c["ecran"] = e
    return titres, calc, payload


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
        for cle_r, cle_m in CORR_DIR:
            cmp(f"{cle_m.format(dk)}", g[cle_m.format(dk)], r["dirs"][dk][cle_r])
        for cote in ("inf", "sup"):
            cmp(f"libellé {cote}_{dk}", es["faces"][f"{cote}_{dk}"]["libelle"], r["dirs"][dk][f"{cote}_detail"])
    for f in E.FACES:
        cote, dk = f.split("_")
        opp = ("sup" if cote == "inf" else "inf")
        terme = {"ec": r["As_min_ec"], "plancher": r["As_min_plancher"], "opposee": 0.25 * r["dirs"][dk][f"As_formule_{opp}"]}
        cmp(f"As_min_terme_{f}", terme[g[f"As_min_terme_{f}"]], r["dirs"][dk][f"As_min_{cote}_eff"])
        cmp(f"besoin_{f}", g[f"besoin_{f}"], max(r["dirs"][dk][f"As_req_{cote}_final"], r["dirs"][dk][f"As_min_{cote}_eff"]))
        geo, mf = es["faces"][f], md["faces"][f]
        for cle in ("As", "yG", "As_pm", "yG_calcule"):
            cmp(f"{f}.{cle}", geo[cle], mf[cle])
        cmp(f"{f}.libellé", geo["libelle"], mf["detail"])
        cmp(f"{f}.yG_impose", geo["yG_impose"], mf["yG_impose"])
        cmp(f"{f}.couches", len(geo["couches"]), len(mf["couches"]))
        for j, (a, b) in enumerate(zip(geo["couches"], mf["couches"]), start=1):
            for cle in ("As_pm", "e", "libelle", "d"):      # (« e_auto », « auto » : la règle du 13/11/2026, que l'écran ignore)
                cmp(f"{f}.couche {j}.{cle}", a[cle], b[cle])
    for v in es["verifications"]:
        cmp(f"motifs {v['cle']} (vides si et seulement si l'état est ok)", not v["motifs"], v["etat"] == "ok")
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
            opp = ("sup_" if v["face"] == "inf" else "inf_") + v["direction"]
            cmp(f"bandeau {f}", f"B{int(g['fyk'])} • {v['libelle']} ({v['limite']:.0f} mm²)", droite)
            req_, mini, besoin = g[f"As_req_{f}"], g[f"As_min_{f}"], g[f"besoin_{f}"]
            attendu = {
                f"**Aₛ,req = {req_:.0f} mm²**": ("**Aₛ,req = M / (fyd · 0,9 · d)**\n\n" f"M = {_fr(g[f'M_{f}'], 1)} kN·m\n\n"
                                                 f"fyd = {_fr(g['fyk'], 0)} / {_fr(g['gamma_s'], 2)} = {_fr(g['fyd'], 0)} N/mm²\n\n"
                                                 f"d = {_fr(g[f'd_{f}'], 1)} cm\n\n" f"→ Aₛ,req = {_fr(req_, 0)} mm²"),
                f"**Aₛ,min = {mini:.0f} mm²**": ("**Aₛ,min = max( 0,26·fctm/fyk·b·h ; 0,0013·b·h ; 0,25·Aₛ,req face opposée, même direction )**\n\n"
                                                 f"0,26 · {_fr(g['fctm'], 1)} / {_fr(g['fyk'], 0)} · b · h = {_fr(g['As_min_ec'], 0)} mm²\n\n"
                                                 f"0,0013 · b · h = {_fr(g['As_min_plancher'], 0)} mm²\n\n"
                                                 f"0,25 · {_fr(g[f'As_req_{opp}'], 0)} = {_fr(g[f'As_min_opposee_{f}'], 0)} mm²\n\n"
                                                 f"→ Aₛ,min = {_fr(mini, 0)} mm²"),
                f"**Aₛ,max = {g['As_max']:.0f} mm²**": ("**Aₛ,max = 0,04 · b · h**\n\n"
                                                         f"0,04 · {_fr(g['b'] * 10, 0)} · {_fr(g['h'] * 10, 0)} = {_fr(g['As_max'], 0)} mm²"),
                f"**Aₛ fourni = {v['As_pm']:.0f} mm²/m**": ("**Aₛ fourni = Σ couches × largeur de bande**\n\n" f"{v['libelle']}\n\n"
                                                            f"Σ = {_fr(v['As_pm'], 0)} mm²/m × {_fr(g['b'] / 100.0, 2)} m = {_fr(v['limite'], 0)} mm²")}
            cmp(f"lignes et infobulles {f}", attendu, {k: aides.get(k) for k in lignes[:4]})
            cmp(f"besoin {f}", besoin, max(req_, mini))
            geom = "❌ **Position des couches incompatible avec l'épaisseur : d utile ≤ 0.**" in lignes
            cmp(f"message « d utile ≤ 0 » {f}", "d_utile_negatif" in v["motifs"], geom)
            cmp(f"lignes du tableau des couches {f}", [str(j) for j in range(1, len(es["faces"][f]["couches"]) + 1)],
                [x for x in lignes if x.isdigit()])
    R = md.get("pdf")
    if R is not None:
        for cle_r, cle_m in (("beton", "beton"), ("fck", "fck"), ("alpha_b", "alpha_b"), ("fctm", "fctm"), ("fyk", "fyk"), ("fyd", "fyd"),
                             ("gamma_s", "gamma_s"), ("mu", "mu"), ("b", "b"), ("h", "h"), ("enrob_beton", "enrobage_calcul"),
                             ("principale", "principale"), ("V", "V"), ("M_max", "M_hauteur"), ("hmin", "hu_min"), ("etat_h", "etat_hauteur"),
                             ("e_cdg_gov", "yG_hauteur"), ("h_min_dalle", "h_min"), ("As_min_ec", "As_min_ec"),
                             ("As_min_plancher", "As_min_plancher"), ("As_max", "As_max"), ("etat_inf", "etat_inf"), ("etat_sup", "etat_sup")):
            cmp(f"note PDF {cle_r}", g[cle_m], R[cle_r])
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
                # la note reçoit les axes FIGÉS au centième (« 3,50 ») : y_G et d au centième près, le reste exact
                cmp(f"note PDF {f} As, As/m, détail, couches", (geo["As"], geo["As_pm"], geo["libelle"], len(geo["couches"])),
                    (Rg["As"], Rg["As_pm"], Rg["detail"], Rg["nc"]))
                for a, b in zip(geo["couches"], Rg["couches"]):
                    cmp(f"note PDF {f} couche {b['i']} (axe au centième)", (a["As_pm"], round(a["e"], 2), a["d"], a["libelle"]),
                        (b["As_pm"], round(b["e"], 2), b["d"], b["label"]))


def faits(predalles, res, calc):
    """Ce que la note atteint (couverture), lu dans les sorties CONTRÔLÉES contre l'écran."""
    out = set()
    if len(predalles) > 1:
        out.add("plusieurs prédalles")
    k = 0
    for p, r in zip(predalles, res):
        out |= {f"béton {p['beton']}", f"B{p['fyk']}", f"principale {p['dir_principale']}", f"prédalle {r['statut']}"}
        if p["h_pre"] == E.H_PRE_MIN:
            out.add("peau minimale")
        if p["h_pre"] == E.h_pre_max(p["h"]):
            out.add("peau maximale")
        if p["jeu_premier_lit"]:
            out.add("jeu premier lit")
        if len(p["sections"]) > 1:
            out.add("plusieurs sections")
        for s, es in zip(p["sections"], r["sections"]):
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
            out |= {f"pose {c['pose']}" + ("" if c["auto"] else " saisie") for f in E.FACES for c in es["faces"][f]["couches"]}
            out |= {"y_G saisi" for f in E.FACES if s["faces"][f]["yG"] is not None}
            out |= {"V = 0" if not s["V"] else "V > 0"}
            if not any(s[m] for m in E.MOMENTS):
                out.add("sans moment")
        out |= {f"avertissement {a['code']}" for a in r["avertissements"]}
    return out


def avec(predalles, **champs):
    """Une copie de la note dont la première section de la première prédalle reçoit ``champs`` (V, yG_inf_y…)."""
    q = copy.deepcopy(predalles)
    for k, x in champs.items():
        if k.startswith("yG_"):
            q[0]["sections"][0]["faces"][k[3:]]["yG"] = x
        else:
            q[0]["sections"][0][k] = x
    return [M.normaliser(p) for p in q]


ECRANS = []
if D is not None:
    print("\n=== 3. Parité avec modules/predalle.py, note par note (calcul, couches, écran, note PDF, fichier) ===")
    base = cas("τ au-delà")
    charger(base)
    r0 = D._dimensionnement_compute_states(1, 1, D.BETON_DATA)
    den = 0.75 * r0["b"] * r0["h"] * 100
    V = None
    for sens in (math.inf, -math.inf):
        x = r0["tau_1"] * den / 1e3
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
        x = 22.0 - hu0
        for _ in range(64):
            if hu0 + x == 22.0:
                y = x
                break
            x = math.nextafter(x, sens)
        if y is not None:
            break
    face_h = "inf_" + ("y" if r0["dirs"]["y"]["M_inf_val"] == r0["M_max"] else "x")
    if y is not None:
        CAS.append((f"borne exacte : h = h_min (y_G {face_h} saisi {y!r}, construit avec l'écran)", avec(base, **{f"yG_{face_h}": y}, V=10.0)))
    COUV = set()
    for k, (desc, predalles) in enumerate(CAS):
        res = [M.verifier(p) for p in predalles]
        titres, calc, payload = ecran(predalles)
        ECRANS.append((desc, predalles, titres, calc))
        ecarts = []
        es_tous = [s for r in res for s in r["sections"]]
        for es, md in zip(es_tous, calc):
            comparer(es, md, ecarts)
        if len(es_tous) != len(calc):
            ecarts.append(f"{len(es_tous)} sections au moteur, {len(calc)} à l'écran")
        for r, t in zip(res, titres):
            if not egal(r["statut"], ICONES.get(t[:1])) or t[2:] != r["predalle"]["nom"]:
                ecarts.append(f"prédalle {r['predalle']['nom']} : moteur {r['statut']} ≠ écran {t}")
        k0 = 0
        for r in res:
            pcts = [(i, b["bloc"][0], b["bloc"][3]) for i, md in enumerate(calc[k0:k0 + len(r["sections"])]) for b in md["ecran"]["blocs"]]
            k0 += len(r["sections"])
            gv = max(pcts, key=lambda x: x[2])
            if not (egal(r["taux_max"], gv[2] / 100.0) and (r["gouvernant"]["section"], r["gouvernant"]["titre"]) == gv[:2]):
                ecarts.append(f"gouvernant : moteur {r['gouvernant']} ≠ écran {gv}")
        # (chaque couche avec l'axe que l'écran a reçu : l'axe de calcul du moteur, « Dist. axe » saisi — voir charger)
        lu = FI.lire(payload)
        attendues = copy.deepcopy(predalles)
        for p_, r_ in zip(attendues, res):
            for s_, rs_ in zip(p_["sections"], r_["sections"]):
                for f_ in E.FACES:
                    for c_, cr_ in zip(s_["faces"][f_]["couches"], rs_["faces"][f_]["couches"]):
                        c_["dist"] = cr_["e"]
        diff = ecarts_json(lu["predalles"], attendues)
        if diff or lu["remarques"]:
            ecarts.append(f"fichier relu : {diff[:3]} {lu['remarques'][:2]}")
        COUV |= faits(predalles, res, calc)
        n_g = sum(len(s["grandeurs"]) for s in es_tous)
        chk(f"note {k + 1:02d} · {desc} — {len(predalles)} prédalle(s), {len(es_tous)} section(s), {n_g} grandeurs, % de l'écran, note PDF, fichier",
            not ecarts, ecarts[:6])
    n_sec = sum(len(p["sections"]) for _, ps in CAS for p in ps)
    chk(f"{len(CAS)} notes ({sum(len(ps) for _, ps in CAS)} prédalles, {n_sec} sections) comparées : au moins 60 notes", len(CAS) >= 60)
    print(f"     {ECART['n']} comparaisons numériques : {ECART['bit']} identiques au bit près, écart relatif maximal "
          f"{ECART['max']:.2e} (tolérance {TOL:g})")
    chk("l'écart relatif maximal reste sous 1e-9 (c'est le même calcul)", ECART["max"] <= TOL, ECART["max"])
    chk("au moins 99 % des comparaisons numériques sont identiques AU BIT PRÈS", ECART["bit"] >= 0.99 * ECART["n"], ECART)

    print("\n=== 4. Couverture : ce que les notes atteignent réellement ===")
    ATTENDUS = (["plusieurs prédalles", "plusieurs sections", "V = 0", "V > 0", "sans moment", "prédalle ok", "prédalle nok",
                 "statut ok", "statut nok", "principale X", "principale Y", "B400", "B500", "y_G saisi", "peau minimale",
                 "peau maximale", "jeu premier lit"]
                + [f"béton {b}" for b in CL]
                + [f"état {k} {e}" for k in ("hau", "inf", "sup") for e in ("ok", "nok")] + ["état tau ok", "état tau nok"]
                + [f"hauteur {f}" for f in E.FACES + (None,)] + ["état hauteur ok", "état hauteur nok"]
                + [f"motif {m}" for m in ("hauteur_insuffisante", "d_utile_negatif", "acier_insuffisant", "acier_excessif", "cisaillement_excessif")]
                + [f"taux {k} > 1" for k in ("hau", "inf", "sup", "tau")]
                + ["As,min ec", "As,min plancher", "As,min opposee"] + [f"{n} couche(s)" for n in (1, 2, 3, 4)]
                + [f"couche {t}" for t in E.TYPES] + [f"couche {t} en renfort" for t in E.TYPES]
                + [f"pose {p}" for p in M.POSES] + [f"pose {p} saisie" for p in M.POSES]
                + ["avertissement enrobage_nul", "avertissement d_utile_negatif", "avertissement nappes_confondues",
                   "avertissement base_hors_predalle", "avertissement croisement_sur_predalle", "borne tau", "borne hau"])
    manque = [a for a in ATTENDUS if a not in COUV]
    chk(f"les notes atteignent les {len(ATTENDUS)} situations attendues (états, motifs, termes, types, poses, bornes…)", not manque, manque)

    print("\n=== 5. Sélecteur de la nappe de base : chaque nappe repassée par l'écran ===")
    for desc, i_s, face in (("test_predalle 2", 0, "inf_y"), ("test_predalle 1", 0, "inf_x"), ("CDG saisi", 0, "inf_y"),
                            ("trois sections", 1, "sup_y"), ("peau minimale", 0, "inf_y"), ("axe saisi sur la base principale", 0, "inf_y"),
                            ("test_predalle 3", 0, "inf_x")):
        predalles = cas(desc)
        cands = M.candidats_base(predalles[0], i_s, face)
        ecarts = []
        for c in cands:
            q = copy.deepcopy(predalles)
            c1 = q[0]["sections"][i_s]["faces"][face]["couches"][0]
            c1.update({k: c[k] for k in ("type", "treillis", "d", "esp")})
            charger(q)
            r = D._dimensionnement_compute_states(1, i_s + 1, D.BETON_DATA)
            cote, dk = face.split("_")
            R = r["dirs"][dk]
            for nom_c, a, b in (("As de la face", c["As"], R[f"As_{cote}_total"]), ("y_G", c["yG"], R[f"e_cdg_{cote}"]),
                                ("axe de la nappe", c["e"], D._dist_couche_eff(1, i_s + 1, face, 1)),
                                ("taux", c["taux"], max(R[f"As_req_{cote}_final"], R[f"As_min_{cote}_eff"]) / R[f"As_{cote}_total"]),
                                ("état", c["etat"], R[f"etat_{cote}"]), ("statut", c["statut_section"], r["etat_global"]),
                                ("As/m de la nappe", c["As_pm"], D._couche_as_per_m(1, i_s + 1, face, 1))):
                if not egal(a, b):
                    ecarts.append(f"{c['nom']}.{nom_c} : {a!r} ≠ {b!r}")
        forme = (len(cands) == len(TRB.TREILLIS_STANDARDS) + len(MD.CANDIDATS_D) * len(MD.CANDIDATS_ESP)
                 and [c["As_pm"] for c in cands] == sorted(c["As_pm"] for c in cands))
        premier = next((c for c in cands if c["ok"]), None)
        b0 = predalles[0]["sections"][i_s]["faces"][face]["couches"][0]
        nom0 = M.candidat_nom(b0) if b0["type"] != "n_barres" else None
        marques = (sum(c["optimal"] for c in cands) == (premier is not None) and (premier is None or premier["optimal"])
                   and [c["nom"] for c in cands if c["retenue"]] == ([nom0] if nom0 in [c["nom"] for c in cands] else []))
        axes = {round(c["e"], 6) for c in cands}
        chk(f"sélecteur {predalles[0]['nom']} · section {i_s + 1} · {face} : {len(cands)} nappes, optimale = "
            f"{premier['nom'] if premier else 'aucune'} ; As, axe ({len(axes)} niveau(x) selon le Ø), y_G, taux, états identiques à l'écran",
            not ecarts and forme and marques, (ecarts[:4], forme, marques))
    fams = M.candidats_base(cas("défauts")[0], 0, "inf_y", ["treillis"])
    chk("le filtre des familles : « treillis » seul ne propose que les dix treillis de la base",
        [c["nom"] for c in fams] == [f"Treillis {x}" for x in TRB.liste_choix()])

    print("\n=== 6. Fichiers de l'écran : migration v1, peau ramenée, fichier Dalle, remplacements muets ===")

    def ouvrir(payload):
        """L'écran ouvre un fichier (_load_from_payload), fait un run (raw → valeurs) et ses synchronisations."""
        st.session_state.clear()
        D._load_from_payload(copy.deepcopy(payload))
        D._sync_float_raw_keys()
        D._init_dalles_if_needed()
        synchroniser()
        return [M.normaliser(lire_session(int(d["id"]))) for d in st.session_state.predalles]
    V1 = {"predalles": [{"id": 1, "nom": "Prédalle 1", "sections": [{"id": 1, "nom": "A"}]}],
          "values": {"pre1_sec1_M_inf": 28.0, "pre1_sec1_M_sup": 12.0, "pre1_sec1_V": 55.0, "pre1_sec1_ncouches_inf": 2,
                     "pre1_sec1_arm_type_inf_c1": "Treillis", "pre1_sec1_treillis_inf_c1": "10/10/100/100",
                     "pre1_sec1_arm_type_inf_c2": "Barres", "pre1_sec1_ø_barres_inf_c2": 12, "pre1_sec1_esp_barres_inf_c2": 150,
                     "pre1_sec1_ncouches_sup": 1, "pre1_sec1_treillis_sup_c1": "8/8/150/150", "gamma_s": 1.5}}
    PIEGES = [("fichier v1 : M_inf → Mx_inf, couches → direction X (secondaire : sur la peau), renfort compris", V1, False)]
    v1b = copy.deepcopy(V1)
    v1b["values"].update({"pre1_sec1_M_inf_raw": "31,5", "pre1_b": 120, "pre1_h": 24.0, "pre1_h_pre": 7.5, "pre1_dir_principale": "x",
                          "meta_predalle_nom_1": "Plancher haut", "meta_pre1_nom_1": "Travée"})
    PIEGES.append(("fichier v1 avec texte saisi « 31,5 », b = 120, h = 24.0, peau 7,5, principale « x » minuscule, noms", v1b, False))
    peau = {"predalles": [{"id": 1, "nom": "P", "sections": [{"id": 1, "nom": "A"}]}],
            "values": {"pre1_h": 8, "pre1_h_pre": 12.0, "pre1_sec1_My_inf": 10.0}}
    PIEGES.append(("audit I-5 : peau de 12 cm sur 8 cm, ramenée à 7 cm par l'écran (relevé)", peau, True))
    peau0 = {"predalles": [{"id": 1, "nom": "P", "sections": [{"id": 1, "nom": "A"}]}], "values": {"pre1_h_pre": 0, "pre1_sec1_My_inf": 10.0}}
    PIEGES.append(("peau nulle : calculée 6 cm (« or 6.0 »), comme l'écran (relevé)", peau0, True))
    deux = {"predalles": [{"id": 3, "nom": "Sans prédalle 1", "sections": [{"id": 2, "nom": "B"}]}], "values": {"pre3_sec2_My_inf": 12.0}}
    PIEGES.append(("prédalle 1 et section 1 absentes : l'écran les recrée", deux, False))
    dalle = {"version": "dalle-2.0", "dalles": [{"id": 1, "nom": "Plancher", "sections": [{"id": 1, "nom": "A"}, {"id": 2, "nom": "Appui"}]}],
             "values": {"dal1_sec1_Mx_inf": 28.0, "dal1_h": 20, "gamma_s": 1.3}}
    PIEGES.append(("fichier du module Dalle ouvert ici : seuls les noms et γs sont repris (relevé)", dalle, True))
    neg = {"predalles": [{"id": 1, "nom": "D", "sections": [{"id": 1, "nom": "A"}]}],
           "values": {"pre1_sec1_Mx_inf": 20.0, "pre1_sec1_Mx_inf_raw": "-20,00", "pre1_sec1_V": 30.0, "pre1_beton": "C16/20",
                      "pre1_fyk": 450, "pre1_sec1_arm_type_inf_x_c1": "Grille", "pre1_sec1_ø_barres_sup_y_c1": 14,
                      "pre1_sec1_dist_axe_inf_y_c1": "abc", "pre1_sec1_dist_auto_inf_y_c1": False,
                      "pre1_sec1_ycdg_sup_x": "0", "pre1_sec1_ycdg_auto_sup_x": False, "pre1_sec1_esp_barres_inf_x_c1": 7,
                      "pre1_sec1_ncouches_sup_y": 9}}
    PIEGES.append(("saisies que l'écran remplace en silence (négatif, classe, nuance, type, Ø, axe, y_G, bornes) : mêmes valeurs", neg, True))
    for desc, payload, remarques in PIEGES:
        attendu = ouvrir(payload)
        lu = FI.lire(copy.deepcopy(payload))
        diff = ecarts_json(lu["predalles"], attendu)
        chk(f"{desc} — {len(attendu)} prédalle(s)", not diff and bool(lu["remarques"]) == remarques, (diff[:4], lu["remarques"][:3]))
    lu = FI.lire(neg)
    chk("… et chaque remplacement silencieux de l'écran est relevé (négatif, classe, nuance, type, Ø, axe, y_G)",
        len(lu["remarques"]) >= 7, lu["remarques"])
    for nom, payload in (("treillis illisible", {"values": {"pre1_sec1_treillis_inf_x_c1": "10/10/abc/100"}}),
                         ("largeur hors des bornes de l'écran", {"values": {"pre1_b": 800}}),
                         ("pas un fichier de prédalle", [1, 2])):
        try:
            FI.lire(payload)
            chk(f"fichier : {nom} → ValueError (le moteur ne peut pas le représenter)", False)
        except ValueError as e_:
            chk(f"fichier : {nom} → ValueError (« {str(e_)[:70]}… »)", True)

# ------------------------------------------------------------------ 7. format
print("\n=== 7. Format : JSON, normalisation, domaine de l'écran, avertissements, formules ===")
tous = [M.verifier(p) for _, ps in CAS for p in ps]
chk("la sortie de verifier() est du JSON strict (sans NaN) et se relit à l'identique",
    all(json.loads(json.dumps(r, allow_nan=False)) == r for r in tous))
chk("normaliser() est idempotente et {} donne nouvelle_predalle()",
    all(M.normaliser(M.normaliser(p)) == M.normaliser(p) for _, ps in CAS for p in ps) and M.normaliser({}) == M.nouvelle_predalle())
entree = copy.deepcopy(cas("test_predalle 2")[0])
M.verifier(entree)
M.candidats_base(entree, 0, "inf_y")
chk("verifier() et candidats_base() ne modifient pas la prédalle reçue", entree == cas("test_predalle 2")[0])


def f_inf_x(**c):
    return {"sections": [{"faces": {"inf_x": {"couches": [dict(c)]}}}]}


REFUS = (("peau sous 3 cm", {"h_pre": 2.5}), ("peau au-delà de 15 cm", {"h": 40, "h_pre": 16.0}),
         ("peau = h (rien de coulé en place)", {"h": 10, "h_pre": 10.0}), ("peau au-delà de h − 1", {"h": 10, "h_pre": 9.5}),
         ("peau en texte", {"h_pre": "6"}), ("peau booléenne", {"h_pre": True}), ("peau infinie", {"h_pre": float("inf")}),
         ("peau NaN", {"h_pre": float("nan")}), ("épaisseur réduite sous la peau + 1 (h 6, peau 6 par défaut)", {"h": 6}),
         ("largeur sous 20 cm", {"b": 15}), ("épaisseur au-delà de 100 cm", {"h": 101}), ("classe inconnue", {"beton": "C16/20"}),
         ("nuance 450", {"fyk": 450}), ("enrobage négatif", {"enrobage": -1}), ("direction « Z »", {"dir_principale": "Z"}),
         ("γs sous 1", {"gamma_s": 0.9}), ("jeu négatif", {"jeu_premier_lit": -0.5}), ("clé inconnue", {"peau": 6}),
         ("aucune section", {"sections": []}), ("moment négatif", {"sections": [{"My_inf": -10}]}),
         ("treillis illisible", f_inf_x(treillis="10/10/abc/100")), ("Ø14 (non proposé)", f_inf_x(d=14)), ("axe nul", f_inf_x(dist=0)),
         ("y_G négatif", {"sections": [{"faces": {"inf_x": {"yG": -2}}}]}))
refusees = []
for nom, champ in REFUS:
    try:
        M.verifier(dict(M.nouvelle_predalle(), **champ))
        refusees.append(f"{nom} : accepté")
    except ValueError:
        pass
    except Exception as e_:  # noqa: BLE001
        refusees.append(f"{nom} : {type(e_).__name__} {e_}")
chk(f"{len(REFUS)} saisies hors du domaine de l'écran lèvent ValueError (la peau d'abord ; jamais de substitution silencieuse)",
    not refusees, refusees)
for nom, appel in (("face inconnue", lambda: M.candidats_base(M.nouvelle_predalle(), 0, "milieu")),
                   ("section inconnue", lambda: M.candidats_base(M.nouvelle_predalle(), 3, "inf_x"))):
    try:
        appel()
        chk(f"sélecteur : {nom} → ValueError", False)
    except ValueError:
        chk(f"sélecteur : {nom} → ValueError", True)
chk("les bornes de la peau sont ADMISES (3 cm ; 15 cm sur 40 ; h − 1 sur 10 ; 4 cm sur 5)",
    all(M.normaliser(dict(M.nouvelle_predalle(), **x))["h_pre"] == x["h_pre"]
        for x in ({"h_pre": 3.0}, {"h": 40, "h_pre": 15.0}, {"h": 10, "h_pre": 9.0}, {"h": 5, "h_pre": 4.0})))
codes = {a["code"] for r in tous for a in r["avertissements"]}
r_hors = M.verifier(cas("peau minimale")[0])
a_hors = [a for a in r_hors["avertissements"] if a["code"] == "base_hors_predalle"]
chk("« base_hors_predalle » : la base Ø12 posée à 3,6 cm sur une peau de 3 cm dépasse (3 + 1,2 = 4,2 cm) — signalé ; "
    "jamais sur la peau de 6 cm par défaut",
    a_hors and proche(a_hors[0]["haut"], 4.2) and not [a for a in M.verifier(cas("défauts")[0])["avertissements"] if a["code"] == "base_hors_predalle"],
    a_hors)
# (depuis le 13/11/2026 les deux BASES sont dans la peau : seuls deux RENFORTS croisés se posent tous deux sur la prédalle)
CROISES = sec("A", 30, 12, 30, 9, 60, inf_x=fa(tr("10/10/100/100"), nb(4, 10)), inf_y=fa(tr("10/10/100/100"), nb(3, 12)))
r_cr = M.verifier(note(pre("CR", [CROISES]))[0])
a_cr = [a for a in r_cr["avertissements"] if a["code"] == "croisement_sur_predalle"]
saisi = copy.deepcopy(CROISES)
saisi["faces"]["inf_x"]["couches"][1]["dist"] = 8.0
chk("« croisement_sur_predalle » : le renfort X (6 + 0,5 = 6,5) et le renfort Y (6 + 0,6 = 6,6), posés tous deux sur la peau, "
    "se croisent (1 mm d'axe à axe, il en faut 11) — signalé ; absent si l'axe de l'un est saisi",
    a_cr and a_cr[0]["paires"] == [[1, 1]]
    and not [a for a in M.verifier(note(pre("CR", [saisi]))[0])["avertissements"] if a["code"] == "croisement_sur_predalle"], a_cr)
def axe_attendu(f):
    """L'axe de calcul d'une couche recalculé ICI, sans beton/position.py : Σ termes (la position réelle) + jeu, arrondi
    au pas SUPÉRIEUR (le quotient arrondi à 1e-6)."""
    det = f["detail"]
    brut = sum(x["valeur"] for x in f["termes"]) + det["jeu"]
    return math.ceil(round(brut / det["pas"], 6)) * det["pas"] if det["pas"] else brut


FORMULES = {
    "e": None,              # l'axe : recalculé par axe_attendu (ses termes, le jeu, l'arrondi : f["detail"])
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
fautes, n_formules, n_axes = [], 0, 0
for r in tous:
    for s in r["sections"]:
        for v in s["verifications"]:
            resultats = {}
            for f in v["formules"]:
                n_formules += 1
                n_axes += f["cle"] == "e"
                t = {x["symbole"]: x["valeur"] for x in f["termes"]}
                resultats[f["cle"]] = f["resultat"]
                calcule = axe_attendu(f) if f["cle"] == "e" else FORMULES[f["cle"]](t)
                if not (calcule == f["resultat"] or proche(calcule, f["resultat"])):
                    fautes.append(f"{r['predalle']['nom']}/{s['nom']}/{v['cle']}/{f['cle']} : {calcule!r} ≠ {f['resultat']!r}")
            cv, cl = VALEUR_LIMITE.get(v["cle"], ("besoin", "As"))
            if v["valeur"] != resultats[cv] or (cl and v["limite"] != resultats[cl]) or v["taux"] != v["valeur"] / v["limite"]:
                fautes.append(f"{r['predalle']['nom']}/{s['nom']}/{v['cle']} : valeur, limite et taux ≠ formules")
chk(f"« formule = valeurs = résultat » : les {n_formules} formules (dont {n_axes} axes de couche et leur règle) se recalculent "
    "à partir de leurs seuls termes, et valeur / limite = taux", not fautes and n_formules > 1500 and n_axes > 300, fautes[:4])
doc = lire("docs", "beton", "PREDALLE_MOTEUR.md")
manque_doc = [k for k in ("h_pre", "h_coule", "e_auto", "pose", "base_hors_predalle", "croisement_sur_predalle", "predalle",
                          "sur_predalle", "superieure") if f"`{k}`" not in doc or k not in (M.__doc__ + M.pose.__doc__)]
chk("ce que le moteur prédalle ajoute (h_pre, h_coule, e_auto, pose, avertissements) est nommé dans sa docstring ET dans "
    "docs/beton/PREDALLE_MOTEUR.md", not manque_doc, manque_doc)

# ------------------------------------------------------------------ 8. pureté
print("\n=== 8. Pureté : le moteur se charge et calcule SANS streamlit, avec la seule bibliothèque standard ===")
non_std = []
for chemin in [os.path.join("beton", "predalle", f) for f in os.listdir(os.path.join(RACINE, "beton", "predalle")) if f.endswith(".py")]:
    arbre = ast.parse(lire(chemin))
    for n in ast.walk(arbre):
        noms = [a.name for a in n.names] if isinstance(n, ast.Import) else \
            [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
        non_std += [f"{chemin}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names | {"beton"}]
chk("aucun fichier de beton/predalle n'importe autre chose que la bibliothèque standard (et beton/)", not non_std, non_std)
prives = [f"{n}" for n in re.findall(r"\bMD\.(_\w+)", lire("beton", "predalle", "moteur.py") + lire("beton", "predalle", "fichier.py"))]
chk("le moteur prédalle n'emploie que l'API PUBLIQUE du moteur des dalles (aucun « MD._… »)", not prives, prives)

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
from beton.predalle import moteur as M, fichier as F
predalles = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.verifier(p) for p in predalles],
          "cands": M.candidats_base(predalles[2], 0, "inf_y"), "fichier": F.lire({"values": {"pre1_sec1_M_inf": 12.0}})}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"beton"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
predalles = [p for _, ps in CAS for p in ps]
attendu = {"res": json.loads(json.dumps(tous)), "cands": json.loads(json.dumps(M.candidats_base(predalles[2], 0, "inf_y"))),
           "fichier": json.loads(json.dumps(FI.lire({"values": {"pre1_sec1_M_inf": 12.0}})))}
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(predalles), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans streamlit", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : streamlit, pandas, reportlab, numpy, modules/… REFUSÉS par sys.meta_path, et le moteur calcule quand même "
        f"({len(out['res'])} prédalles, le sélecteur, la lecture d'un fichier)", out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : le moteur n'a chargé que la bibliothèque standard", not out["tiers"], out["tiers"])
    diff = ecarts_json({"res": out["res"], "cands": out["cands"], "fichier": out["fichier"]}, attendu)
    chk(f"Python {v} : résultats identiques à ceux de ce processus", not diff, diff[:3])
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")

# ------------------------------------------------------------------ 9. mutations
# Le moteur est ALTÉRÉ, rechargé sous un autre nom, et comparé à ce que l'écran a rendu en section 3 : chaque altération
# doit faire rougir au moins une note. Deux familles : la RÈGLE PRÉDALLE (beton/predalle/moteur.py) et les FORMULES
# PARTAGÉES (beton/dalle/moteur.py, injecté dans le moteur prédalle à la place du vrai) — la parité prédalle garde les deux.
if D is not None:
    print("\n=== 9. Mutations : le moteur altéré, la parité avec l'écran rougit ===")
    SOURCE_P = lire("beton", "predalle", "moteur.py")
    SOURCE_D = lire("beton", "dalle", "moteur.py")
    MUTATIONS_P = [   # (la règle du 13/11/2026 : beton/predalle/moteur.pose et position_auto)
        ("toute couche inférieure sur la peau (même les bases)", 'return "predalle" if rang == 0 else "sur_predalle"',
         'return "sur_predalle"'),
        ("les renforts inférieurs dans la peau avec la base (et non sur la peau)",
         'return "predalle" if rang == 0 else "sur_predalle"', 'return "predalle"'),
        ("sur la peau : enrobage + demi-Ø (la règle de la dalle)", "return float(h_pre or H_PRE_SI_NUL) + d / 20.0",
         "return float(enrobage or MD.ENROBAGE_SI_NUL) + d / 20.0"),
        ("sur la peau : h_pre + Ø entier (et non le demi-Ø)", "return float(h_pre or H_PRE_SI_NUL) + d / 20.0",
         "return float(h_pre or H_PRE_SI_NUL) + d / 10.0"),
        ("peau lue 6 cm quelle qu'elle soit", "return float(h_pre or H_PRE_SI_NUL) +", "return float(H_PRE_SI_NUL) +"),
        ("axe saisi écrasé par l'automatique", '                if c["dist"] is None:\n                    c["dist"] = a[f][j]',
         '                if True:\n                    c["dist"] = a[f][j]'),
        ("enrobage nul lu 0 (et non 3,0 cm comme l'écran)", "P.sur_parement(float(enrobage or MD.ENROBAGE_SI_NUL), d)",
         "P.sur_parement(float(enrobage), d)"),
    ]
    MUTATIONS_D = [
        ("formules partagées : bras de levier 0,9·d → 0,85·d", "K_BRAS_DE_LEVIER = 0.9 ", "K_BRAS_DE_LEVIER = 0.85 "),
        ("formules partagées : As,min avec la face opposée → la face elle-même", "opposee = K_AS_MIN_OPPOSEE * req[opp]",
         "opposee = K_AS_MIN_OPPOSEE * req[f]"),
        ("formules partagées : τ_adm,I → τ_adm,II", 'g.update({"tau": tau, "tau_adm": mat["tau_1"]', 'g.update({"tau": tau, "tau_adm": mat["tau_2"]'),
        ("formules partagées : hauteur par le plus grand moment (avant l'audit C-1)",
         "max(actives, key=lambda x: _hu_min(x[1], alpha_b, b, mu) + x[2])", "max(actives, key=lambda x: x[1])"),
        ("formules partagées : As ≤ As,max oublié", " and As <= As_max) else NOK", ") else NOK"),
        ("formules partagées : « n barres » sans la largeur de bande", "/ max(0.01, bb / 100.0)", "/ 1.0"),
        ("formules partagées : y_G saisi ignoré", '"yG": yG_calcule if yG is None else yG', '"yG": yG_calcule'),
        ("position : le jeu de pose oublié", "e = P.axe_calcul(reel, jeu, arrondi)", "e = P.axe_calcul(reel, 0.0, arrondi)"),
        ("position : l'arrondi oublié", "e = P.axe_calcul(reel, jeu, arrondi)", "e = P.axe_calcul(reel, jeu, 0.0)"),
    ]

    def charger_source(source, nom, paquet):
        mod = types.ModuleType(nom)
        mod.__package__ = paquet
        exec(compile(source, f"{nom} (altéré)", "exec"), mod.__dict__)
        return mod

    def mutant(mutation_p=None, mutation_d=None):
        """Le moteur prédalle dont la source (ou celle du moteur des dalles qu'il emploie) porte UNE altération."""
        sp, sd = SOURCE_P, SOURCE_D
        if mutation_p:
            if sp.count(mutation_p[0]) != 1:
                return None
            sp = sp.replace(*mutation_p)
        if mutation_d:
            if sd.count(mutation_d[0]) != 1:
                return None
            sd = sd.replace(*mutation_d)
        mod = charger_source(sp, "beton.predalle._mutant", "beton.predalle")
        if mutation_d:
            mod.MD = charger_source(sd, "beton.dalle._mutant", "beton.dalle")
        return mod

    temoin = mutant()
    rouges_temoin = 0
    for desc, ps, _t, calc in ECRANS[:6]:
        ecarts = []
        for es, md in zip([s_ for p_ in ps for s_ in temoin.verifier(p_)["sections"]], calc):
            comparer(es, md, ecarts)
        rouges_temoin += bool(ecarts)
    chk("témoin : le moteur rechargé SANS altération reste vert (le mécanisme ne rougit pas tout seul)", rouges_temoin == 0, rouges_temoin)
    for famille, liste in (("p", MUTATIONS_P), ("d", MUTATIONS_D)):
        for nom, avant, apres in liste:
            mod = mutant((avant, apres) if famille == "p" else None, (avant, apres) if famille == "d" else None)
            if mod is None:
                chk(f"mutation « {nom} » : le texte altéré est dans le moteur, une seule fois", False, avant)
                continue
            rouges = 0
            for desc, ps, _titres, calc in ECRANS:
                try:
                    es_tous = [s_ for p_ in ps for s_ in mod.verifier(p_)["sections"]]
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
