# -*- coding: utf-8 -*-
"""Moteur « Taux d'armature » (beton/taux), REFAIT sur la conception du 10/11/2026
(docs/site/LOT_Q_QUATRE_MODULES.md § 3) : plus aucune parité avec l'écran Streamlit — des contrôles À LA MAIN (chaque
formule recalculée ici avec ``math`` seul, sur des valeurs écrites en clair), des invariants, des refus et des
MUTATIONS. Un quantitatif ne doit jamais sous-estimer : chaque contrôle rougit si l'on retire ce qu'il protège.

  1. La poutre TYPE d'une note neuve, déroulée à la main de bout en bout : chaque repère, chaque total (85,8 kg/m³).
  2. Les formules une à une : masse linéique (EN 10080 à 0,5 %), barre filante = L − 2c, bouts bornés à l'enrobage,
     retours (automatiques, saisis), recouvrements (14 m : un de 50 Ø ; 23,9 m : DEUX — l'écart avec la conception,
     prouvé par la pose), barres de peau, cadre, cadre intérieur, épingle, retour de max(10 Ø ; 70 mm).
  3. Le compte des étriers : le cas d'hier (L = 1 m, zone 0–0,3 m, s = 10 cm : 4 étriers, pas 3), deux zones jointives
     qui ne doublent pas l'étrier commun, deux suites disjointes, et une GRILLE (L de 1 à 12 m, zones de 0,3 à 6 m, s
     de 5 à 30 cm) contre un oracle EXACT (la pose simulée en millimètres entiers) : jamais un de moins.
  4. Les talons : comptés dans le béton (deux talons : aire = âme + les deux ; le polygone par la formule du lacet) et
     leur cadre dans l'acier.
  5. Le bordereau : Σ repères = Σ familles = Σ diamètres = total ; les parts ; le CSV se relit et redonne les nombres.
  6. Les refus de la normalisation : un message en français qui nomme le champ, l'entrée intacte.
  7. Les gestes de la saisie (beton/taux/saisie.py) : ajouts, limite de deux zones qui suit, longueur, type employé.
  8. Recevoir une poutre (beton/taux/depuis_poutre.py) : trois poutres, la masse au mètre de chaque tronçon à la main,
     l'invariant « masse détaillée (sans retours, recouvrements ni fermeture) = Σ masse au mètre × l_i, aux abouts près ».
  9. La géométrie de la coupe (barre d'angle à c + Ø cadre + Ø/2, rien ne se chevauche, rien ne sort du cadre) et les
     compositions de la poutre.
 10. Le format : JSON strict, idempotence, chaque formule RECALCULÉE à partir de ses seuls termes ; avertissements.
 11. Pureté : bibliothèque standard seule, sous Python 3.12 et 3.13 (Pyodide), numpy et consorts REFUSÉS.
 12. Mutations : le moteur rechargé depuis ses sources, modifié d'UNE règle, est rejeté par ces contrôles.

    /usr/bin/python3.12 tests/test_taux_moteur.py
"""
import ast
import copy
import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from beton.poutre import entree as EP  # noqa: E402
from beton.taux import depuis_poutre as DP  # noqa: E402
from beton.taux import entree as EN  # noqa: E402
from beton.taux import geometrie as GE  # noqa: E402
from beton.taux import moteur as M  # noqa: E402
from beton.taux import saisie as SA  # noqa: E402

OK, KO = [], []
REEL = {"EN": EN, "GE": GE, "M": M, "SA": SA, "DP": DP}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:700]}]" if info and not cond else ""))


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as fh:
        return fh.read()


def pres(a, b, tol=1e-9):
    """a = b à ``tol`` près (relatif au-delà de 1)."""
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))


def refuse(fonction, *mots):
    """La fonction lève ValueError, et son message porte chacun des ``mots`` ; rend le message, ou None."""
    try:
        fonction()
    except ValueError as e:
        return str(e) if all(mot in str(e) for mot in mots) else None
    return None


# ====================================================================== la poutre type, écrite ICI (une copie nommée :
# le contrôle « défauts » la lie à entree.nouvelle_poutre)
def barre(famille="inf", lit=1, n=2, d=16, de=None, a=None, retours=0, l_retour=None):
    return {"famille": famille, "lit": None if famille == "peau" else lit, "n": n, "d": d, "de": de, "a": a, "retours": retours,
            "l_retour": l_retour}


def cadre(d=8):
    return {"forme": "cadre", "d": d, "largeur": None}


def epingle(d=8):
    return {"forme": "epingle", "d": d, "largeur": None}


def interieur(w, d=8):
    return {"forme": "cadre_interieur", "d": d, "largeur": w}


def zone(de, a, s, t="E1"):
    return {"de": de, "a": a, "s": s, "type": t}


def poutre(**k):
    p = {"nom": "P1", "forme": "rectangulaire", "b": 30.0, "h": 60.0, "L": 6.0, "enrobage": 3.0,
         "talon_gauche": {"b": 15.0, "h": 20.0}, "talon_droit": {"b": 15.0, "h": 20.0},
         "barres": [barre("inf", 1, 3, 20), barre("inf", 2, 2, 16, 1.2, 4.8), barre("sup", 1, 2, 12)],
         "types_etriers": [{"nom": "E1", "lignes": [cadre(8)], "cadre_talon": True}],
         "zones": [zone(0.0, 1.5, 15.0), zone(1.5, 4.5, 25.0), zone(4.5, 6.0, 15.0)],
         "majoration": 5.0, "recouvrements": {"actifs": True, "l_com": 12.0, "k": 50.0}, "rho_beton": 2500.0, "rho_acier": 7850.0,
         "origine": None}
    p.update(copy.deepcopy(k))
    return p


TYPE = poutre()


def m_lin(d):
    """π·Ø²/4·ρ_s, à la main (Ø en mm, 7 850 kg/m³)."""
    return math.pi * (d / 1000.0) ** 2 / 4.0 * 7850.0


# ====================================================================== 1. la poutre type, à la main
# 30 × 60 cm, L = 6,00 m, c = 3 cm ; ① 3 Ø 20 filantes, ② 2 Ø 16 de 1,20 à 4,80 m, ③ 2 Ø 12 filantes ;
# cadres Ø 8 : 15 cm sur 1,50 m à chaque about, 25 cm sur les 3,00 m de travée
MAIN = {}
MAIN["l1"] = 6.00 - 2 * 0.03                              # 5,94 m
MAIN["L1"] = 3 * MAIN["l1"]                               # 17,82 m
MAIN["M1"] = MAIN["L1"] * m_lin(20)
MAIN["l2"] = 4.80 - 1.20                                  # 3,60 m
MAIN["L2"] = 2 * MAIN["l2"]                               # 7,20 m
MAIN["M2"] = MAIN["L2"] * m_lin(16)
MAIN["l3"] = 6.00 - 2 * 0.03
MAIN["L3"] = 2 * MAIN["l3"]                               # 11,88 m
MAIN["M3"] = MAIN["L3"] * m_lin(12)
MAIN["l_r"] = max(10 * 8, 70)                             # 80 mm
MAIN["l4"] = 2 * ((30 - 2 * 3) + (60 - 2 * 3)) / 100.0 + 2 * MAIN["l_r"] / 1000.0      # 1,56 + 0,16 = 1,72 m
MAIN["n4"] = math.ceil(1.5 / 0.15 - 1e-9) + math.ceil(3.0 / 0.25 - 1e-9) + math.ceil(1.5 / 0.15 - 1e-9) + 1   # 10 + 12 + 10 + 1
MAIN["L4"] = MAIN["n4"] * MAIN["l4"]                      # 56,76 m
MAIN["M4"] = MAIN["L4"] * m_lin(8)
MAIN["Ms"] = MAIN["M1"] + MAIN["M2"] + MAIN["M3"] + MAIN["M4"]
MAIN["Mmaj"] = MAIN["Ms"] * 1.05
MAIN["Vc"] = 0.30 * 0.60 * 6.00                           # 1,080 m³
MAIN["taux"] = MAIN["Mmaj"] / MAIN["Vc"]
ECRIT = {"l1": 5.94, "L1": 17.82, "l2": 3.6, "L2": 7.2, "L3": 11.88, "l4": 1.72, "n4": 33, "L4": 56.76, "Vc": 1.08}
ARRONDIS = {"M1": (43.947, 3), "M2": (11.364, 3), "M3": (10.547, 3), "M4": (22.397, 3), "Ms": (88.255, 3), "Mmaj": (92.667, 3),
            "taux": (85.80, 2)}


def fautes_type(m):
    """La poutre type contre le calcul à la main (``MAIN``), repère par repère, total par total."""
    f = []
    r = m["M"].calculer(TYPE)
    rp = r["reperes"]
    if [(x["repere"], x["famille"], x["d"], x["n"]) for x in rp] != [(1, "inf", 20, 3), (2, "renfort", 16, 2), (3, "sup", 12, 2), (4, "etriers", 8, MAIN["n4"])]:
        f.append(f"repères : {[(x['repere'], x['famille'], x['d'], x['n']) for x in rp]}")
        return f
    for k, (lu, lt, ms) in enumerate((("l1", "L1", "M1"), ("l2", "L2", "M2"), ("l3", "L3", "M3"), ("l4", "L4", "M4"))):
        x = rp[k]
        if not (pres(x["l_unitaire"], MAIN[lu]) and pres(x["l_totale"], MAIN[lt]) and pres(x["masse"], MAIN[ms])):
            f.append(f"repère {k + 1} : l = {x['l_unitaire']}, L = {x['l_totale']}, M = {x['masse']} (à la main : {MAIN[lu]}, {MAIN[lt]}, {MAIN[ms]})")
        if not pres(x["masse_lineique"], m_lin(x["d"])):
            f.append(f"repère {k + 1} : masse linéique {x['masse_lineique']}")
    t, bt = r["totaux"], r["beton"]
    attendus = (("masse", t["masse"], MAIN["Ms"]), ("masse majorée", t["masse_majoree"], MAIN["Mmaj"]), ("volume", bt["volume"], MAIN["Vc"]),
                ("taux majoré", t["taux_majore"], MAIN["taux"]), ("taux net", t["taux"], MAIN["Ms"] / MAIN["Vc"]),
                ("kg/m majoré", t["kg_m_majore"], MAIN["Mmaj"] / 6.0), ("kg/m", t["kg_m"], MAIN["Ms"] / 6.0),
                ("supplément", t["supplement"], MAIN["Ms"] * 0.05), ("masse du béton", bt["masse"], 1.08 * 2500.0),
                ("aire", bt["aire_cm2"], 1800.0), ("longueur totale", t["longueur"], MAIN["L1"] + MAIN["L2"] + MAIN["L3"] + MAIN["L4"]))
    f += [f"{nom} : {v} (à la main : {a})" for nom, v, a in attendus if not pres(v, a)]
    if [(z["cours"], z["fermeture"], z["n"]) for z in r["zones"]] != [(10, 0, 10), (12, 0, 12), (10, 1, 11)]:
        f.append(f"zones : {[(z['cours'], z['fermeture'], z['n']) for z in r['zones']]}")
    if r["avertissements"] or r["trous"]:
        f.append(f"la poutre type n'appelle aucun avertissement : {r['avertissements']}")
    return f


# ====================================================================== 2. les formules, une à une
# EN 10080 (tableau des masses nominales), kg/m ; et les six valeurs au millième de la consigne
NOMINALES = {6: 0.222, 8: 0.395, 10: 0.617, 12: 0.888, 14: 1.21, 16: 1.58, 20: 2.47, 25: 3.85, 28: 4.83, 32: 6.31, 40: 9.86}
AU_MILLIEME = {8: 0.395, 12: 0.888, 16: 1.578, 20: 2.466, 25: 3.853, 32: 6.313}


def fautes_masses(m):
    f = []
    for d, nom in NOMINALES.items():
        v = m["M"].masse_lineique(d)
        if abs(v - nom) / nom > 0.005:
            f.append(f"Ø {d} : {v:.4f} kg/m, nominal {nom} ({100 * (v - nom) / nom:+.2f} %)")
        if not pres(v, m_lin(d)):
            f.append(f"Ø {d} : {v} ≠ π·Ø²/4·7 850 = {m_lin(d)}")
    f += [f"Ø {d} : {m['M'].masse_lineique(d):.4f} ≠ {nom} au millième" for d, nom in AU_MILLIEME.items()
          if abs(m["M"].masse_lineique(d) - nom) > 0.0006]
    if tuple(m["EN"].DIAMETRES) != tuple(NOMINALES):
        f.append(f"Ø permis : {m['EN'].DIAMETRES}")
    if not pres(m["M"].masse_lineique(20, 8000.0), math.pi * 0.02 ** 2 / 4 * 8000.0):
        f.append("la masse volumique de l'acier n'entre pas dans la masse linéique")
    return f


def un_repere(m, p, k=0):
    return m["M"].calculer(p)["reperes"][k]


def fautes_barres(m):
    """La barre filante, les bouts bornés, les retours, la peau."""
    f = []
    # filante : L − 2c — L = 7,35 m, c = 2,5 cm → 7,30 m
    r = un_repere(m, poutre(L=7.35, enrobage=2.5, barres=[barre("inf", 1, 4, 25)], zones=[]))
    if not (pres(r["l_unitaire"], 7.35 - 2 * 0.025) and pres(r["l_unitaire"], 7.30) and pres(r["l_totale"], 4 * 7.30)):
        f.append(f"filante : {r['l_unitaire']} m (L − 2c = 7,30 m)")
    # bouts bornés à l'enrobage : de 0 à 7,35 → 7,30 ; de 0 à 2 → 1,975 ; de 5 au bout → 2,325 ; de 1 à 3 → 2 (rien à borner)
    for de, a, attendu in ((0.0, 7.35, 7.30), (0.0, 2.0, 2.0 - 0.025), (5.0, None, 7.35 - 0.025 - 5.0), (1.0, 3.0, 2.0), (None, 2.0, 1.975)):
        r = un_repere(m, poutre(L=7.35, enrobage=2.5, barres=[barre("inf", 1, 2, 16, de, a)], zones=[]))
        if not pres(r["l_unitaire"], attendu):
            f.append(f"barre de {de} à {a} : {r['l_unitaire']} m (attendu {attendu})")
    # retours : automatiques h − 2c = 60 − 6 = 54 cm ; 2 retours → 5,94 + 1,08 = 7,02 m
    r = un_repere(m, poutre(barres=[barre("inf", 1, 3, 20, retours=2)]))
    if not (pres(r["l_unitaire"], 5.94 + 2 * 0.54) and pres(r["l_unitaire"], 7.02) and r["detail"]["l_retour_auto"] and pres(r["detail"]["l_retour"], 54.0)):
        f.append(f"2 retours automatiques : {r['l_unitaire']} m (5,94 + 2 × 0,54 = 7,02)")
    # saisis : 1 retour de 30 cm → 6,24 m ; 0 retour : la longueur saisie ne compte pas
    r = un_repere(m, poutre(barres=[barre("inf", 1, 3, 20, retours=1, l_retour=30.0)]))
    if not pres(r["l_unitaire"], 6.24):
        f.append(f"1 retour de 30 cm : {r['l_unitaire']} m (6,24)")
    r = un_repere(m, poutre(barres=[barre("inf", 1, 3, 20, retours=0, l_retour=30.0)]))
    if not pres(r["l_unitaire"], 5.94):
        f.append(f"0 retour : {r['l_unitaire']} m (5,94)")
    # la peau : n par face, DEUX faces ; son retour automatique est b − 2c = 24 cm
    r = un_repere(m, poutre(barres=[barre("peau", None, 2, 10)]))
    if not (r["n"] == 4 and pres(r["l_totale"], 4 * 5.94) and pres(r["masse"], 4 * 5.94 * m_lin(10)) and r["famille"] == "peau"):
        f.append(f"peau 2 × 2 Ø 10 : n = {r['n']}, L = {r['l_totale']}")
    r = un_repere(m, poutre(barres=[barre("peau", None, 1, 10, retours=1)]))
    if not pres(r["l_unitaire"], 5.94 + 0.24):
        f.append(f"peau, retour automatique b − 2c : {r['l_unitaire']} m (6,18)")
    return f


def pose_recouvrements(l_0, l_com, l_rec):
    """L'oracle des recouvrements, par la POSE : on aboute des barres de l_com (la dernière coupée) qui se recouvrent
    de l_rec jusqu'à couvrir l_0 ; rend le nombre de recouvrements."""
    n, couvert = 1, l_com
    while couvert < l_0 - 1e-9:
        n += 1
        couvert += l_com - l_rec
    return n - 1


def fautes_recouvrements(m):
    f = []
    # la barre de 14 m : L = 14,06 m, c = 3 cm ; l_com = 12 m, 50 Ø de Ø 20 = 1,00 m → UN recouvrement, l = 15,00 m
    p14 = poutre(L=14.06, barres=[barre("inf", 1, 3, 20)], zones=[])
    r = un_repere(m, p14)
    if not (r["detail"]["n_rec"] == 1 and pres(r["detail"]["l_rec"], 50 * 0.020) and pres(r["l_unitaire"], 14.0 + 1.0) and pres(r["l_totale"], 45.0)):
        f.append(f"barre de 14 m : {r['detail']['n_rec']} recouvrement(s), l = {r['l_unitaire']} m (attendu 1 et 15,00 m)")
    # 12,00 m tout juste : aucun ; 12,01 m : un
    for L, n in ((12.06, 0), (12.07, 1)):
        r = un_repere(m, poutre(L=L, barres=[barre("inf", 1, 2, 20)], zones=[]))
        if r["detail"]["n_rec"] != n:
            f.append(f"barre de {L - 0.06:.2f} m : {r['detail']['n_rec']} recouvrement(s) (attendu {n})")
    # 23,90 m en Ø 20 : deux barres de 12 m se recouvrant de 1 m ne couvrent que 23 m — il en faut TROIS, deux recouvrements
    r = un_repere(m, poutre(L=23.96, barres=[barre("inf", 1, 2, 20)], zones=[]))
    if not (r["detail"]["n_rec"] == 2 == pose_recouvrements(23.9, 12.0, 1.0) and pres(r["l_unitaire"], 23.9 + 2 * 1.0)):
        f.append(f"barre de 23,90 m : {r['detail']['n_rec']} recouvrement(s) (la pose en demande {pose_recouvrements(23.9, 12.0, 1.0)})")
    # les retours comptent dans la longueur à couvrir : 11,50 m droits + 2 × 0,54 = 12,58 m > 12 m → un recouvrement
    r = un_repere(m, poutre(L=11.56, barres=[barre("inf", 1, 2, 20, retours=2)], zones=[]))
    if r["detail"]["n_rec"] != 1:
        f.append(f"11,50 m + 2 retours de 54 cm : {r['detail']['n_rec']} recouvrement(s) (attendu 1)")
    # la grille : jamais moins que la pose, jamais moins que la formule de la conception
    for l_com in (6.0, 12.0, 14.0):
        for d in (12, 16, 20, 25, 32, 40):
            for k in (40.0, 50.0, 60.0):
                rec = {"actifs": True, "l_com": l_com, "k": k}
                for dm in range(50, 421):
                    l_0 = dm / 10.0
                    n, l_rec = m["M"].recouvrements(l_0, d, rec)
                    if n != pose_recouvrements(l_0, l_com, k * d / 1000.0) or n < math.ceil(l_0 / l_com - 1e-9) - 1 or not pres(l_rec, k * d / 1000.0):
                        f.append(f"l₀ = {l_0} m, Ø {d}, l_com = {l_com}, k = {k} : {n} (pose : {pose_recouvrements(l_0, l_com, k * d / 1000.0)})")
                        break
    # l'option désactivée : aucun recouvrement, et c'est DIT (avertissement grave qui nomme le repère)
    res = m["M"].calculer(poutre(L=14.06, barres=[barre("inf", 1, 3, 20)], zones=[], recouvrements={"actifs": False, "l_com": 12.0, "k": 50.0}))
    av = [a for a in res["avertissements"] if a["code"] == "barre_longue"]
    if not (res["reperes"][0]["detail"]["n_rec"] == 0 and pres(res["reperes"][0]["l_unitaire"], 14.0) and len(av) == 1 and av[0]["grave"]
            and "Repère 1" in av[0]["message"] and "14,00 m" in av[0]["message"]):
        f.append(f"recouvrements désactivés : {res['reperes'][0]['l_unitaire']} m, avertissements {av}")
    if any(a["code"] == "barre_longue" for a in m["M"].calculer(p14)["avertissements"]):
        f.append("recouvrements comptés : pas d'avertissement « barre longue »")
    return f


def fautes_etriers_longueurs(m):
    """Cadre, cadre intérieur, épingle : 40 × 80 cm, c = 4 cm."""
    f = []
    base = dict(b=40.0, h=80.0, enrobage=4.0, barres=[], zones=[zone(0.0, 6.0, 20.0)])
    cas = (([cadre(10)], [2 * ((40 - 8) + (80 - 8)) / 100.0 + 2 * 0.100]),                       # 2,08 + 0,20 = 2,28 m
           ([cadre(6)], [2.08 + 2 * 0.070]),                                                      # Ø 6 : retour de 70 mm, pas 60
           ([cadre(10), interieur(15.0, 8)], [2.28, 2 * (15 + 72) / 100.0 + 2 * 0.080]),          # 1,74 + 0,16 = 1,90 m
           ([cadre(10), epingle(8), epingle(8)], [2.28, 0.72 + 0.16, 0.72 + 0.16]),               # 0,88 m
           ([cadre(12), epingle(16)], [2.08 + 0.24, 0.72 + 0.32]))
    ecrits = ([2.28], [2.22], [2.28, 1.90], [2.28, 0.88, 0.88], [2.32, 1.04])
    for (lignes, attendues), en_clair in zip(cas, ecrits):
        res = m["M"].calculer(poutre(types_etriers=[{"nom": "E1", "lignes": lignes, "cadre_talon": False}], **base))
        ls = [x["l_unitaire"] for x in res["reperes"]]
        if not (len(ls) == len(attendues) and all(pres(a, b) for a, b in zip(ls, attendues)) and all(pres(a, b) for a, b in zip(ls, en_clair))):
            f.append(f"{[(li['forme'], li['d']) for li in lignes]} : {ls} (attendu {en_clair})")
        n = math.ceil(6.0 / 0.20 - 1e-9) + 1                                                       # 31 cours
        if any(x["n"] != 31 or n != 31 for x in res["reperes"]):
            f.append(f"chaque ligne d'un type est posée à chaque cours (31) : {[x['n'] for x in res['reperes']]}")
    for d, attendu in ((6, 70.0), (8, 80.0), (10, 100.0), (16, 160.0), (20, 200.0)):
        if not pres(m["M"].retour_etrier(d), attendu) or not pres(m["M"].retour_etrier(d), max(10 * d, 70)):
            f.append(f"retour d'un cadre Ø {d} : {m['M'].retour_etrier(d)} mm (max(10 Ø ; 70) = {attendu})")
    return f


# ====================================================================== 3. le compte des étriers
def pose(zones_mm):
    """L'ORACLE : la pose réelle, simulée en millimètres ENTIERS (exacte) — un étrier au bout gauche de chaque suite de
    zones jointives, puis, dans chaque zone, un étrier tous les s au plus, le dernier au bout de la zone (c'est aussi le
    premier de la zone suivante quand elle la touche). ``zones_mm`` : [(de, a, s)], de gauche à droite."""
    n, fin = 0, None
    for de, a, s in zones_mm:
        if fin is None or de != fin:
            n += 1
        x = de
        while x < a:
            x = min(x + s, a)
            n += 1
        fin = a
    return n


def compte(m, zones_mm):
    return sum(k["n"] for k in m["M"].compter_etriers([{"de": de / 1000.0, "a": a / 1000.0, "s": s / 10.0} for de, a, s in zones_mm]))


PAS_MM = (50, 75, 100, 120, 125, 150, 175, 200, 220, 250, 300)


def grille():
    """Les cas de la grille (mm) : une zone depuis l'about, une zone au milieu, deux zones jointives, deux disjointes."""
    for L in (1000, 1500, 2000, 2400, 3000, 3600, 4000, 5000, 6000, 7500, 9000, 10000, 12000):
        for l1 in range(300, min(L, 6000) + 1, 100):
            for s1 in PAS_MM:
                yield [(0, l1, s1)]
                if l1 % 300 == 0 and L - l1 >= 300:
                    yield [(L - l1, L, s1)]
                    for l2 in range(300, min(L - l1, 6000) + 1, 700):
                        for s2 in PAS_MM[::3]:
                            yield [(0, l1, s1), (l1, l1 + l2, s2)]
                            if L - l1 - l2 >= 250:
                                yield [(0, l1, s1), (L - l2, L, s2)]


def fautes_compte(m, complet=True):
    f = []
    c = m["M"].compter_etriers
    # le cas d'hier : L = 1 m, zone de 0 à 0,3 m, s = 10 cm → 4 étriers (0 ; 0,1 ; 0,2 ; 0,3), pas 3
    k = c([{"de": 0.0, "a": 0.3, "s": 10.0}])[0]
    if not (k["cours"] == 3 and k["fermeture"] == 1 and k["n"] == 4 == pose([(0, 300, 100)])):
        f.append(f"le cas d'hier (0–0,3 m à 10 cm) : {k['n']} étriers (4)")
    # 0,9 / 0,1 = 9,000000000000002 en flottants : 9 intervalles, 10 étriers — pas 11
    k = c([{"de": 1.2, "a": 2.1, "s": 10.0}])[0]
    if k["n"] != 10:
        f.append(f"1,2–2,1 m à 10 cm : {k['n']} étriers (10)")
    # deux zones jointives ne doublent pas l'étrier commun : 0–1,5 à 15 puis 1,5–4,5 à 25 → 10 + 12 + 1 = 23
    ks = c([{"de": 0.0, "a": 1.5, "s": 15.0}, {"de": 1.5, "a": 4.5, "s": 25.0}])
    if not ([x["n"] for x in ks] == [10, 13] and [x["fermeture"] for x in ks] == [0, 1] and ks[0]["suite"] == ks[1]["suite"]
            and sum(x["n"] for x in ks) == 23 == pose([(0, 1500, 150), (1500, 4500, 250)])):
        f.append(f"deux zones jointives : {[x['n'] for x in ks]} (10 + 13 = 23, un seul étrier à la jonction)")
    # l'ordre des zones ne compte pas
    ks = c([{"de": 1.5, "a": 4.5, "s": 25.0}, {"de": 0.0, "a": 1.5, "s": 15.0}])
    if [x["n"] for x in ks] != [13, 10]:
        f.append(f"zones données de droite à gauche : {[x['n'] for x in ks]} (13, 10)")
    # deux suites DISJOINTES ferment chacune : 0–1 à 20 (5 + 1) et 3–4 à 20 (5 + 1) → 12
    ks = c([{"de": 0.0, "a": 1.0, "s": 20.0}, {"de": 3.0, "a": 4.0, "s": 20.0}])
    if not ([x["n"] for x in ks] == [6, 6] and ks[0]["suite"] != ks[1]["suite"] and pose([(0, 1000, 200), (3000, 4000, 200)]) == 12):
        f.append(f"deux suites disjointes : {[x['n'] for x in ks]} (6 et 6)")
    # une zone plus courte que son pas : deux étriers (un à chaque bout)
    if c([{"de": 0.0, "a": 0.1, "s": 25.0}])[0]["n"] != 2:
        f.append("zone de 10 cm au pas de 25 : 2 étriers")
    # la grille : le moteur = la pose exacte (donc jamais un de moins)
    n = moins = plus = 0
    for zs in grille():
        n += 1
        if not complet and n % 7:
            continue
        a, b = compte(m, zs), pose(zs)
        moins += a < b
        plus += a > b
        if a != b and len(f) < 6:
            f.append(f"grille {zs} : {a} étriers, la pose en demande {b}")
    if moins or plus:
        f.append(f"grille : {moins} cas avec MOINS d'étriers que la pose, {plus} avec plus, sur {n}")
    fautes_compte.n = n
    return f


def ancienne_formule():
    """Ce que l'ancien écran perdait sur les zones de la grille dont la longueur est un MULTIPLE EXACT du pas (là où
    int(L / s) + 1 et la pose devraient s'accorder) : la troncature d'un quotient de flottants (0,3 / 0,1 =
    2,9999999999999996) y retirait un étrier. Rend (zones au quotient entier, zones où il manquait un étrier)."""
    n = moins = 0
    for zs in grille():
        de, a, s = zs[0]
        if len(zs) != 1 or (a - de) % s:
            continue
        n += 1
        moins += int((a / 1000.0 - de / 1000.0) / (s / 1000.0)) + 1 < pose(zs)
    return n, moins


# ====================================================================== 4. les talons
def fautes_talons(m):
    f = []
    # deux talons : âme 30 × 60, talon gauche 15 × 20, talon droit 20 × 25 → 1 800 + 300 + 500 = 2 600 cm²
    p = poutre(forme="deux_talons", talon_gauche={"b": 15.0, "h": 20.0}, talon_droit={"b": 20.0, "h": 25.0})
    r = m["M"].calculer(p)
    sec = r["section"]
    if not (pres(sec["aire_cm2"], 30 * 60 + 15 * 20 + 20 * 25) and pres(sec["aire_cm2"], 2600.0) and pres(r["beton"]["volume"], 0.26 * 6.0)
            and pres(r["beton"]["masse"], 0.26 * 6.0 * 2500.0) and pres(sec["b_tot"], 65.0) and pres(sec["h_talon"], 25.0)):
        f.append(f"deux talons : aire {sec['aire_cm2']} cm² (2 600), V = {r['beton']['volume']} m³ (1,560)")
    for forme, aire, btot in (("rectangulaire", 1800.0, 30.0), ("talon_gauche", 2100.0, 45.0), ("talon_droit", 2300.0, 50.0), ("deux_talons", 2600.0, 65.0)):
        s = m["GE"].section(dict(p, forme=forme))
        if not (pres(s["aire_cm2"], aire) and pres(m["GE"].aire_polygone(s["polygone"]), aire) and pres(s["b_tot"], btot)):
            f.append(f"{forme} : aire {s['aire_cm2']} cm², polygone {m['GE'].aire_polygone(s['polygone'])} (attendu {aire})")
    # le cadre de talon dans l'acier : 2·[(65 − 6) + (25 − 6)] + 2 × 8 cm = 156 + 16 = 172 cm, à chaque cours (33)
    rp = r["reperes"]
    talon = [x for x in rp if x["detail"].get("ligne") == "talon"]
    if not (len(talon) == 1 and pres(talon[0]["l_unitaire"], 2 * ((65 - 6) + (25 - 6)) / 100.0 + 0.16) and pres(talon[0]["l_unitaire"], 1.72)
            and talon[0]["n"] == 33 and pres(talon[0]["masse"], 33 * 1.72 * m_lin(8)) and talon[0]["famille"] == "etriers"):
        f.append(f"cadre de talon : {[(x['l_unitaire'], x['n']) for x in talon]} (1,72 m × 33)")
    # un seul talon, à gauche : 2·[(45 − 6) + (20 − 6)] + 16 = 122 cm ; décoché : pas de cadre de talon, et c'est dit
    r1 = m["M"].calculer(dict(p, forme="talon_gauche"))
    t1 = [x for x in r1["reperes"] if x["detail"].get("ligne") == "talon"]
    if not (len(t1) == 1 and pres(t1[0]["l_unitaire"], 1.22)):
        f.append(f"cadre de talon, talon gauche seul : {[x['l_unitaire'] for x in t1]} (1,22 m)")
    r2 = m["M"].calculer(dict(p, types_etriers=[{"nom": "E1", "lignes": [cadre(8)], "cadre_talon": False}]))
    if any(x["detail"].get("ligne") == "talon" for x in r2["reperes"]) or not any(a["code"] == "talon_sans_cadre" for a in r2["avertissements"]):
        f.append("cadre de talon décoché : il n'est pas compté, et un avertissement le dit")
    if not pres(r["totaux"]["masse"] - r2["totaux"]["masse"], 33 * 1.72 * m_lin(8)):
        f.append("le cadre de talon pèse 33 × 1,72 m de Ø 8 dans le total")
    # rectangulaire : la case est sans objet (aucun cadre de talon compté, aucun avertissement)
    r3 = m["M"].calculer(TYPE)
    if any(x["detail"].get("ligne") == "talon" for x in r3["reperes"]) or any(a["code"] == "talon_sans_cadre" for a in r3["avertissements"]):
        f.append("rectangulaire : pas de cadre de talon")
    return f


# ====================================================================== 5. le bordereau
RICHE = poutre(nom="Poutre riche", forme="deux_talons", L=14.06, b=40.0, h=80.0, enrobage=4.0,
               talon_gauche={"b": 15.0, "h": 20.0}, talon_droit={"b": 20.0, "h": 25.0},
               barres=[barre("inf", 1, 5, 25), barre("inf", 2, 3, 20, 3.0, 11.0), barre("sup", 1, 3, 16, retours=2),
                       barre("sup", 1, 2, 16, None, 2.5, retours=1, l_retour=40.0), barre("peau", None, 2, 12)],
               types_etriers=[{"nom": "E1", "lignes": [cadre(10), interieur(14.0, 8), epingle(8)], "cadre_talon": True},
                              {"nom": "E2", "lignes": [cadre(8)], "cadre_talon": False},
                              {"nom": "Inutile", "lignes": [cadre(6)], "cadre_talon": False}],
               zones=[zone(0.0, 2.0, 10.0, "E1"), zone(2.0, 12.0, 20.0, "E2"), zone(12.5, 14.06, 12.5, "E1")], majoration=7.5)


def lire_csv(texte):
    return list(csv.reader(io.StringIO(texte), delimiter=";"))


def nb_csv(s):
    return float(s.replace(",", "."))


def fautes_bordereau(m):
    f = []
    for nom, p in (("type", TYPE), ("riche", RICHE)):
        r = m["M"].calculer(p)
        rp, t = r["reperes"], r["totaux"]
        s_rep = sum(x["masse"] for x in rp)
        s_fam = sum(x["masse"] for x in r["familles"])
        s_dia = sum(x["masse"] for x in r["diametres"])
        if not (pres(s_rep, t["masse"]) and pres(s_fam, t["masse"]) and pres(s_dia, t["masse"])):
            f.append(f"{nom} : Σ repères {s_rep}, Σ familles {s_fam}, Σ diamètres {s_dia}, total {t['masse']}")
        if not (pres(sum(x["l_totale"] for x in rp), t["longueur"]) and pres(sum(x["longueur"] for x in r["diametres"]), t["longueur"])
                and pres(sum(x["longueur"] for x in r["familles"]), t["longueur"])):
            f.append(f"{nom} : les longueurs ne se retrouvent pas")
        if not (pres(sum(x["part"] for x in rp), 1.0) and pres(sum(x["part"] for x in r["familles"]), 1.0) and pres(sum(x["part"] for x in r["diametres"]), 1.0)):
            f.append(f"{nom} : les parts ne font pas 1")
        if any(not pres(x["masse"], x["l_totale"] * x["masse_lineique"]) or not pres(x["l_totale"], x["n"] * x["l_unitaire"]) for x in rp):
            f.append(f"{nom} : un repère n'est pas n × l × m")
        if [x["repere"] for x in rp] != list(range(1, len(rp) + 1)) or [x["cle"] for x in r["familles"]] != ["inf", "sup", "renfort", "peau", "etriers"]:
            f.append(f"{nom} : repères 1, 2, 3… et cinq familles dans l'ordre")
        if [x["d"] for x in r["diametres"]] != sorted({x["d"] for x in rp}):
            f.append(f"{nom} : diamètres triés")
        if not (pres(t["masse_majoree"], t["masse"] * (1 + p["majoration"] / 100.0)) and pres(t["taux_majore"], t["masse_majoree"] / r["beton"]["volume"])
                and pres(t["kg_m_majore"], t["masse_majoree"] / p["L"])):
            f.append(f"{nom} : majoration, taux ou kg/m")
        # le CSV : « ; », virgule décimale, une ligne d'en-tête, les repères, la synthèse par Ø, le total
        texte = m["M"].bordereau_csv(r)
        lignes = lire_csv(texte)
        n = len(rp)
        if lignes[0] != list(m["M"].EN_TETE_CSV) or any(len(li) != 8 for li in lignes) or "." in "".join(c for li in lignes[1:] for c in li[2:]):
            f.append(f"{nom} : CSV — en-tête, huit cases par ligne, virgule décimale")
            continue
        for x, li in zip(rp, lignes[1:1 + n]):
            if not (int(li[0]) == x["repere"] and li[1].startswith(x["designation"]) and int(li[2]) == x["d"] and int(li[3]) == x["n"]
                    and abs(nb_csv(li[4]) - x["l_unitaire"]) <= 5.1e-4 and abs(nb_csv(li[5]) - x["l_totale"]) <= 5.1e-4
                    and abs(nb_csv(li[6]) - x["masse_lineique"]) <= 5.1e-4 and abs(nb_csv(li[7]) - x["masse"]) <= 5.1e-4):
                f.append(f"{nom} : CSV, repère {x['repere']} : {li}")
        for x, li in zip(r["diametres"], lignes[1 + n:1 + n + len(r["diametres"])]):
            if not (li[0] == f"Ø {x['d']}" and abs(nb_csv(li[5]) - x["longueur"]) <= 5.1e-4 and abs(nb_csv(li[7]) - x["masse"]) <= 5.1e-4):
                f.append(f"{nom} : CSV, Ø {x['d']} : {li}")
        fin = lignes[1 + n + len(r["diametres"]):]
        if not ([li[0] for li in fin] == ["Total", "Majoration", "Total majoré"] and abs(nb_csv(fin[0][7]) - t["masse"]) <= 5.1e-4
                and abs(nb_csv(fin[0][5]) - t["longueur"]) <= 5.1e-4 and abs(nb_csv(fin[1][7]) - t["supplement"]) <= 5.1e-4
                and abs(nb_csv(fin[2][7]) - t["masse_majoree"]) <= 5.1e-4):
            f.append(f"{nom} : CSV, total : {fin}")
        # relu, le CSV redonne les totaux : Σ des repères lus = total lu (au gramme par ligne près)
        if abs(sum(nb_csv(li[7]) for li in lignes[1:1 + n]) - nb_csv(fin[0][7])) > 0.0006 * n:
            f.append(f"{nom} : CSV, la somme des repères lus n'est pas le total lu")
    # le type inemployé n'a pas de repère ; un nom de type piégé reste UNE case du CSV
    r = m["M"].calculer(RICHE)
    if any("Inutile" in x["designation"] for x in r["reperes"]) or not any(a["code"] == "type_inemploye" and "Inutile" in a["message"] for a in r["avertissements"]):
        f.append("type inemployé : sans repère, et dit")
    p = poutre(types_etriers=[{"nom": 'a;b"c', "lignes": [cadre(8)], "cadre_talon": False}], zones=[zone(0.0, 6.0, 15.0, 'a;b"c')])
    li = lire_csv(m["M"].bordereau_csv(m["M"].calculer(p)))[4]
    if not (len(li) == 8 and li[1].startswith('Cadres a;b"c')):
        f.append(f"CSV : un nom de type avec « ; » et un guillemet reste une case : {li}")
    return f


# ====================================================================== 6. les refus
def refus_attendus():
    """(ce qu'on abîme, la poutre abîmée, les mots que le message doit porter)."""
    t1 = [{"nom": "E1", "lignes": [cadre(8)], "cadre_talon": False}]
    return [
        ("largeur de 5 cm", poutre(b=5.0), ("largeur b", "10", "300")),
        ("largeur de 301 cm", poutre(b=301.0), ("largeur b",)),
        ("hauteur de 9 cm", poutre(h=9.0), ("hauteur h",)),
        ("longueur de 0,4 m", poutre(L=0.4, zones=[], barres=[]), ("longueur L", "0,5")),
        ("longueur de 61 m", poutre(L=61.0), ("longueur L", "60")),
        ("longueur nulle", poutre(L=0.0), ("longueur L",)),
        ("enrobage de 0,5 cm", poutre(enrobage=0.5), ("enrobage c",)),
        ("enrobage de 16 cm", poutre(enrobage=16.0), ("enrobage c",)),
        ("2c ≥ b", poutre(b=10.0, enrobage=5.0, barres=[], zones=[]), ("enrobage c", "largeur b")),
        ("2c ≥ h", poutre(h=12.0, enrobage=6.0, barres=[], zones=[]), ("enrobage c", "hauteur h")),
        ("forme inconnue", poutre(forme="en_T"), ("forme", "en_T")),
        ("clé de l'ancien module", dict(poutre(), acier="B500A"), ("clé(s) inconnue(s)", "acier")),
        ("largeur en texte", poutre(b="30"), ("largeur b", "nombre")),
        ("largeur booléenne", poutre(b=True), ("largeur b", "nombre")),
        ("longueur infinie", poutre(L=float("inf")), ("longueur L",)),
        ("hauteur NaN", poutre(h=float("nan")), ("hauteur h",)),
        ("talon aussi haut que la poutre", poutre(forme="talon_gauche", talon_gauche={"b": 15.0, "h": 60.0}), ("talon gauche", "moins haut")),
        ("talon de 2 cm de large", poutre(forme="talon_droit", talon_droit={"b": 2.0, "h": 20.0}), ("talon droit", "largeur")),
        ("cadre de talon dans un talon trop bas", poutre(forme="talon_gauche", talon_gauche={"b": 15.0, "h": 6.0}), ("cadre de talon", "enrobage")),
        ("barres : pas une liste", poutre(barres="3Ø20"), ("barres", "liste")),
        ("famille inconnue", poutre(barres=[barre("talon")]), ("barres n° 1", "famille")),
        ("zéro barre", poutre(barres=[barre(n=0)]), ("barres n° 1", "nombre de barres")),
        ("41 barres", poutre(barres=[barre(n=41)]), ("barres n° 1", "40")),
        ("2,5 barres", poutre(barres=[barre(n=2.5)]), ("barres n° 1", "entier")),
        ("Ø 18", poutre(barres=[barre(d=18)]), ("barres n° 1", "Ø")),
        ("« de » au-delà de L", poutre(barres=[barre(de=7.0)]), ("barres n° 1", "« de »", "hors de la poutre")),
        ("« à » ≤ « de »", poutre(barres=[barre(), barre(de=3.0, a=3.0)]), ("barres n° 2", "« à »", "« de »")),
        ("barre dans l'enrobage d'about", poutre(barres=[barre(de=0.0, a=0.02)]), ("barres n° 1", "enrobage")),
        ("3 retours", poutre(barres=[barre(retours=3)]), ("barres n° 1", "retours")),
        ("retour négatif", poutre(barres=[barre(retours=1, l_retour=-1.0)]), ("barres n° 1", "retour")),
        ("lit 0", poutre(barres=[barre(lit=0)]), ("barres n° 1", "lit")),
        ("type sans nom", poutre(types_etriers=[{"nom": " ", "lignes": [cadre()], "cadre_talon": False}], zones=[]), ("type", "nom")),
        ("deux types du même nom", poutre(types_etriers=t1 + t1), ("deux types", "« E1 »")),
        ("type sans ligne", poutre(types_etriers=[{"nom": "E1", "lignes": [], "cadre_talon": False}]), ("« E1 »", "au moins une ligne")),
        ("forme de ligne inconnue", poutre(types_etriers=[{"nom": "E1", "lignes": [{"forme": "U", "d": 8, "largeur": None}], "cadre_talon": False}]), ("« E1 »", "forme")),
        ("cadre intérieur sans largeur", poutre(types_etriers=[{"nom": "E1", "lignes": [{"forme": "cadre_interieur", "d": 8, "largeur": None}], "cadre_talon": False}]),
         ("« E1 »", "ligne 1", "largeur")),
        ("cadre intérieur plus large que l'âme", poutre(types_etriers=[{"nom": "E1", "lignes": [cadre(), interieur(25.0)], "cadre_talon": False}]),
         ("« E1 »", "ligne 2", "largeur", "24")),
        ("cadre Ø 25", poutre(types_etriers=[{"nom": "E1", "lignes": [cadre(25)], "cadre_talon": False}]), ("« E1 »", "Ø")),
        ("zone d'un type inconnu", poutre(zones=[zone(0.0, 6.0, 15.0, "E9")]), ("zone Z1", "« E9 »")),
        ("zone « à » ≤ « de »", poutre(zones=[zone(2.0, 2.0, 15.0)]), ("zone Z1", "« à »")),
        ("zone au-delà de L", poutre(zones=[zone(0.0, 6.5, 15.0)]), ("zone Z1", "hors de la poutre")),
        ("pas de 4 cm", poutre(zones=[zone(0.0, 6.0, 4.0)]), ("zone Z1", "pas", "5")),
        ("pas de 51 cm", poutre(zones=[zone(0.0, 6.0, 51.0)]), ("zone Z1", "pas", "50")),
        ("deux zones qui se recouvrent", poutre(zones=[zone(0.0, 2.0, 15.0), zone(1.5, 4.5, 25.0)]), ("Z1", "Z2", "se recouvrent", "de 1,5 à 2 m")),
        ("zones qui se recouvrent, données dans le désordre", poutre(zones=[zone(3.0, 6.0, 15.0), zone(0.0, 3.5, 25.0)]), ("Z1", "Z2", "se recouvrent")),
        ("majoration de 51 %", poutre(majoration=51.0), ("majoration",)),
        ("majoration négative", poutre(majoration=-1.0), ("majoration",)),
        ("longueur commerciale de 5 m", poutre(recouvrements={"actifs": True, "l_com": 5.0, "k": 50.0}), ("longueur commerciale",)),
        ("recouvrement de 10 Ø", poutre(recouvrements={"actifs": True, "l_com": 12.0, "k": 10.0}), ("recouvrements",)),
        ("recouvrements « oui »", poutre(recouvrements={"actifs": "oui", "l_com": 12.0, "k": 50.0}), ("recouvrements", "vrai ou faux")),
        ("béton de 1 000 kg/m³", poutre(rho_beton=1000.0), ("béton",)),
        ("acier de 7 000 kg/m³", poutre(rho_acier=7000.0), ("acier",)),
        ("origine d'un module inconnu", poutre(origine={"module": "dalle", "nom": "D1", "date": ""}), ("origine", "module")),
        ("nom de 61 caractères", poutre(nom="x" * 61), ("nom", "60")),
        ("nom sur deux lignes", poutre(nom="P1\nP2"), ("nom", "seule ligne")),
    ]


def fautes_refus(m):
    f = []
    for quoi, p, mots in refus_attendus():
        avant = copy.deepcopy(p)
        msg = refuse(lambda: m["EN"].normaliser(p), *mots)
        if msg is None:
            try:
                m["EN"].normaliser(p)
                f.append(f"{quoi} : ADMIS")
            except ValueError as e:
                f.append(f"{quoi} : message sans {mots} — « {e} »")
        elif repr(p) != repr(avant):
            f.append(f"{quoi} : l'entrée a été modifiée")
    for mauvais in (None, [1, 2], "poutre", 3):
        if refuse(lambda: m["EN"].normaliser(mauvais), "objet") is None:
            f.append(f"une poutre qui n'est pas un objet ({mauvais!r}) : refusée")
    return f


def fautes_normalisation(m):
    """Défauts, idempotence, copie, zones rangées, « sans objet » retiré."""
    f = []
    EN_ = m["EN"]
    if EN_.nouvelle_poutre() != TYPE:
        f.append("entree.nouvelle_poutre() n'est plus la poutre type écrite dans ce test (conception § 3.5)")
    n1 = EN_.normaliser(TYPE)
    if n1 != TYPE or EN_.normaliser(n1) != n1 or n1 is TYPE or n1["barres"] is TYPE["barres"]:
        f.append("normaliser(poutre type) : identique, idempotente, une copie")
    vide = EN_.normaliser({})
    if not (vide["b"] == 30.0 and vide["h"] == 60.0 and vide["L"] == 6.0 and vide["enrobage"] == 3.0 and vide["barres"] == [] and vide["zones"] == []
            and vide["types_etriers"] == [] and vide["majoration"] == 5.0 and vide["recouvrements"] == {"actifs": True, "l_com": 12.0, "k": 50.0}
            and vide["rho_beton"] == 2500.0 and vide["rho_acier"] == 7850.0 and vide["origine"] is None and vide["forme"] == "rectangulaire"):
        f.append(f"clés absentes → défauts, listes VIDES : {vide}")
    z = EN_.normaliser(poutre(zones=[zone(4.5, 6.0, 15.0), zone(0.0, 1.5, 15.0), zone(1.5, 4.5, 25.0)]))["zones"]
    if [x["de"] for x in z] != [0.0, 1.5, 4.5]:
        f.append(f"les zones sont rendues de gauche à droite : {[x['de'] for x in z]}")
    b = EN_.normaliser(poutre(barres=[barre("peau", 3, 1, 10)], types_etriers=[{"nom": "E1", "lignes": [dict(cadre(), largeur=12.0)], "cadre_talon": False}]))
    if b["barres"][0]["lit"] is not None or b["types_etriers"][0]["lignes"][0]["largeur"] is not None:
        f.append("le lit d'une barre de peau et la largeur d'un cadre sont sans objet : rendus à None")
    o = EN_.normaliser(poutre(origine={"module": "poutre", "nom": " V1 ", "date": "10/11/2026"}))["origine"]
    if o != {"module": "poutre", "nom": "V1", "date": "10/11/2026"}:
        f.append(f"origine : {o}")
    if EN_.etendue(barre(), 6.0, 3.0) != (0.03, 5.97) or EN_.etendue(barre(de=1.2, a=4.8), 6.0, 3.0) != (1.2, 4.8):
        f.append("etendue : filante de c à L − c, renfort à ses abscisses")
    return f


# ====================================================================== 7. les gestes de la saisie
def fautes_saisie(m):
    f = []
    EN_, SA_ = m["EN"], m["SA"]

    def neuve(**k):
        return EN_.normaliser(poutre(**k))
    # + Lit inférieur : au-dessus du dernier lit filant, avec ses barres ; + Lit supérieur ; + Renfort ; + Barres de peau
    p = neuve()
    SA_.ajouter_barre(p, "inf")
    SA_.ajouter_barre(p, "sup")
    SA_.ajouter_barre(p, "renfort")
    SA_.ajouter_barre(p, "peau")
    attendus = [barre("inf", 2, 3, 20), barre("sup", 2, 2, 12), barre("inf", 2, 2, 16, 1.2, 4.8), barre("peau", None, 1, 10)]
    if p["barres"][3:] != attendus or EN_.normaliser(p) != p:
        f.append(f"ajouts de barres : {p['barres'][3:]}")
    q = neuve(barres=[], L=7.35)
    SA_.ajouter_barre(q, "inf")
    SA_.ajouter_barre(q, "renfort")
    if q["barres"] != [barre("inf", 1, 2, 16), barre("inf", 2, 2, 16, 1.47, 5.88)]:
        f.append(f"sur une poutre sans barre : {q['barres']}")
    if refuse(lambda: SA_.ajouter_barre(neuve(), "talon"), "sorte") is None:
        f.append("une sorte de barres inconnue est refusée")
    # + Type, + Ligne, la forme d'une ligne ; retirer
    p = neuve()
    SA_.ajouter_type(p)
    SA_.ajouter_ligne(p, 1)
    SA_.changer_forme_ligne(p, 1, 1, "cadre_interieur")
    if p["types_etriers"][1] != {"nom": "E2", "lignes": [cadre(8), interieur(12.0)], "cadre_talon": True} or EN_.normaliser(p) != p:
        f.append(f"+ Type, + Ligne, cadre intérieur (largeur proposée (30 − 6)/2 = 12 cm) : {p['types_etriers'][1]}")
    msg = refuse(lambda: SA_.retirer_type(neuve(), 0), "« E1 »", "Z1, Z2, Z3")
    if msg is None:
        f.append("retirer un type que des zones emploient : refusé, les zones nommées")
    SA_.retirer_type(p, 1)
    if [t["nom"] for t in p["types_etriers"]] != ["E1"]:
        f.append("un type inemployé se retire")
    if refuse(lambda: SA_.retirer_ligne(neuve(), 0, 0), "au moins une ligne") is None:
        f.append("la dernière ligne d'un type ne se retire pas")
    # renommer : les zones suivent ; un nom déjà pris est refusé
    p = neuve()
    SA_.renommer_type(p, 0, "Cadre")
    if {z["type"] for z in p["zones"]} != {"Cadre"} or EN_.normaliser(p) != p:
        f.append("renommer un type : ses zones le suivent")
    p = neuve()
    SA_.ajouter_type(p)
    if refuse(lambda: SA_.renommer_type(p, 1, "E1"), "existe déjà") is None:
        f.append("renommer un type du nom d'un autre : refusé")
    # + Zone : la suite (de la fin de la dernière à L, le pas et le type de la précédente) ; rien à proposer quand tout est couvert
    if refuse(lambda: SA_.ajouter_zone(neuve()), "couvrent déjà toute la poutre") is None:
        f.append("+ Zone sur une poutre couverte : refusé")
    p = neuve(zones=[zone(0.0, 1.5, 15.0), zone(1.5, 4.5, 25.0)])
    if SA_.ajouter_zone(p) != zone(4.5, 6.0, 25.0):
        f.append(f"+ Zone propose la suite : {p['zones'][-1]}")
    p = neuve(zones=[zone(0.0, 1.5, 15.0), zone(4.5, 6.0, 12.5)])
    if SA_.ajouter_zone(p) != zone(1.5, 4.5, 15.0):
        f.append(f"+ Zone, la dernière va au bout : la première portion sans étriers : {p['zones'][-1]}")
    p = neuve(zones=[], types_etriers=[])
    if SA_.ajouter_zone(p) != zone(0.0, 6.0, 15.0) or [t["nom"] for t in p["types_etriers"]] != ["E1"] or EN_.normaliser(p) != p:
        f.append(f"+ Zone sans aucun type : E1 est créé, la zone couvre la poutre : {p['zones']}")
    # la limite de deux zones jointives : la voisine SUIT, et c'est dit
    p = neuve()
    dits = SA_.deplacer_borne(p, 0, "a", 2.0)
    if not ([(z["de"], z["a"]) for z in p["zones"]] == [(0.0, 2.0), (2.0, 4.5), (4.5, 6.0)] and len(dits) == 1 and "Z2" in dits[0] and "2 m" in dits[0]
            and EN_.normaliser(p) == p):
        f.append(f"« à » de Z1 porté à 2 m : {[(z['de'], z['a']) for z in p['zones']]} — dit : {dits}")
    p = neuve()
    dits = SA_.deplacer_borne(p, 2, "de", 4.0)
    if not ([(z["de"], z["a"]) for z in p["zones"]] == [(0.0, 1.5), (1.5, 4.0), (4.0, 6.0)] and len(dits) == 1 and "Z2" in dits[0]):
        f.append(f"« de » de Z3 ramené à 4 m : {[(z['de'], z['a']) for z in p['zones']]} — dit : {dits}")
    p = neuve(zones=[zone(0.0, 1.5, 15.0), zone(2.0, 6.0, 25.0)])
    if SA_.deplacer_borne(p, 0, "a", 1.8) or [(z["de"], z["a"]) for z in p["zones"]] != [(0.0, 1.8), (2.0, 6.0)]:
        f.append("une zone qui ne touche pas sa voisine ne l'entraîne pas")
    p = neuve()
    SA_.deplacer_borne(p, 0, "a", 5.0)
    if refuse(lambda: EN_.normaliser(p), "Z2") is None:
        f.append("une limite poussée au-delà de la zone voisine : la poutre est refusée (la zone voisine n'a plus de longueur)")
    # la longueur : ce qui se compte depuis l'about droit le suit
    p = neuve()
    dits = SA_.changer_longueur(p, 7.2)
    if not (p["L"] == 7.2 and [(z["de"], z["a"]) for z in p["zones"]] == [(0.0, 1.5), (1.5, 5.7), (5.7, 7.2)] and (p["barres"][1]["de"], p["barres"][1]["a"]) == (1.2, 6.0)
            and p["barres"][0]["a"] is None and len(dits) == 3 and EN_.normaliser(p) == p):
        f.append(f"L de 6 à 7,2 m : zones {[(z['de'], z['a']) for z in p['zones']]}, renfort {(p['barres'][1]['de'], p['barres'][1]['a'])}, dit {dits}")
    r = m["M"].calculer(p)
    if [z["n"] for z in r["zones"]] != [10, math.ceil(4.2 / 0.25 - 1e-9), 11] or r["trous"]:
        f.append(f"L = 7,2 m : étriers {[z['n'] for z in r['zones']]}, sans trou")
    p = neuve()
    SA_.changer_longueur(p, 5.0)
    if [(z["de"], z["a"]) for z in p["zones"]] != [(0.0, 1.5), (1.5, 3.5), (3.5, 5.0)] or (p["barres"][1]["de"], p["barres"][1]["a"]) != (1.2, 3.8) or EN_.normaliser(p) != p:
        f.append(f"L de 6 à 5 m : {[(z['de'], z['a']) for z in p['zones']]}")
    p = neuve()
    SA_.changer_longueur(p, 2.5)
    if refuse(lambda: EN_.normaliser(p), "Z") is None:
        f.append("L de 6 à 2,5 m : les zones d'about se rejoignent, la poutre est refusée (rien n'est écrasé en silence)")
    p = neuve()
    if SA_.changer_longueur(p, 6.0) or p != neuve():
        f.append("la même longueur ne déplace rien")
    return f


# ====================================================================== 8. recevoir une poutre de « Poutres béton armé »
def section_ba(nom, inf, sup, etriers, pas, **k):
    return dict({"nom": nom, "inf": [{"n": n, "d": d} for n, d in inf], "sup": [{"n": n, "d": d} for n, d in sup], "etriers": etriers, "pas": pas}, **k)


def poutre_ba(sections, b=30.0, h=60.0, enrobage=3.0, nom="V1"):
    p = EP.nouvelle_poutre()
    p.update({"nom": nom, "b": b, "h": h, "enrobage": enrobage, "sections": sections})
    return p


E8 = [{"type": "etrier", "d": 8}]
# 1 · une section : 30 × 60, 3 Ø 20 / 2 Ø 12, cadre Ø 8 au pas de 15 cm, sur 6,00 m
BA1 = (poutre_ba([section_ba("A", [(3, 20)], [(2, 12)], E8, 15.0)]), [6.0])
# 2 · trois sections, celle du milieu renforcée (un second lit 2 Ø 16) ; le Ø supérieur change sur l'appui droit
BA2 = (poutre_ba([section_ba("A", [(3, 20)], [(2, 12)], E8, 15.0), section_ba("B", [(3, 20), (2, 16)], [(2, 12)], E8, 25.0),
                  section_ba("C", [(3, 20)], [(2, 16)], E8, 15.0)]), [1.5, 3.0, 1.5])
# 3 · 40 × 70, quatre sections : cadre Ø 10 + cadre intérieur Ø 8 (barres 2 à 3) + épingle Ø 8 aux abouts ; 4 puis 6 puis 6
#     puis 4 Ø 20 en bas (le surplus de deux tronçons voisins réuni), des latérales imposées
E3 = [{"type": "etrier", "d": 10}, {"type": "etrier", "d": 8, "de": 2, "a": 3}, {"type": "epingle", "d": 8}]
BA3 = (poutre_ba([section_ba("A", [(4, 20)], [(3, 16)], E3, 10.0), section_ba("B", [(6, 20)], [(3, 16)], [{"type": "etrier", "d": 10}], 20.0),
                  section_ba("C", [(6, 20)], [(3, 16)], [{"type": "etrier", "d": 10}], 20.0, laterales={"n": 2, "d": 12}),
                  section_ba("D", [(4, 20)], [(3, 16)], E3, 10.0)], b=40.0, h=70.0), [1.0, 2.0, 2.0, 1.0])


def masse_au_metre(m, res, x):
    """La masse au mètre courant du module DÉTAILLÉ à l'abscisse x, lue dans le résultat du moteur : les barres qui
    passent en x, et les cours de la zone qui contient x (sans retour, recouvrement ni fermeture)."""
    barres = sum(r["n"] * r["masse_lineique"] for r in res["reperes"] if r["sorte"] == "barre" and r["detail"]["x1"] - 1e-9 <= x <= r["detail"]["x2"] + 1e-9)
    z = next((z for z in res["zones"] if z["de"] <= x <= z["a"]), None)
    etriers = 0.0
    if z:
        t = next(t for t in res["types"] if t["nom"] == z["type"])
        etriers = t["masse_cours"] / (z["s"] / 100.0)
    return barres, etriers


def fautes_depuis_poutre(m):
    f = []
    conv = m["DP"].convertir
    # ---- 1 · une section. Latérales AUTOMATIQUES de la poutre : lits 1 à 3 + 2,0 = 5,0 cm et 3 + 1,5 = 4,5 cm des
    #      parements (arrondis au demi-centimètre supérieur de 0,8 + 1,0 et 0,8 + 0,6) ; écart 60 − 9,5 = 50,5 cm > 30 :
    #      ⌈50,5 / 30⌉ − 1 = 1 Ø 10 par face
    p = conv(*BA1, date="10/11/2026")
    attendu = poutre(nom="V1", barres=[barre("inf", 1, 3, 20), barre("sup", 1, 2, 12), barre("peau", None, 1, 10)],
                     types_etriers=[{"nom": "E1", "lignes": [cadre(8)], "cadre_talon": False}], zones=[zone(0.0, 6.0, 15.0)],
                     origine={"module": "poutre", "nom": "V1", "date": "10/11/2026"})
    if p != attendu or m["EN"].normaliser(p) != p:
        f.append(f"une section : {p}")
    res = m["M"].calculer(p)
    mb = 3 * m_lin(20) + 2 * m_lin(12) + 2 * 1 * m_lin(10)                       # kg/m de barres
    me = (2 * ((30 - 6) + (60 - 6)) / 100.0 + 0.16) * m_lin(8) / 0.15           # kg/m d'étriers : 1,72 m tous les 15 cm
    lu = masse_au_metre(m, res, 3.0)
    if not (pres(lu[0], mb) and pres(lu[1], me)):
        f.append(f"une section : masse au mètre {lu} (à la main : {mb}, {me})")
    f += invariant(m, res, [6.0], [mb], [me], "une section")
    # ---- 2 · trois sections, celle du milieu renforcée ; le Ø supérieur change en C
    p = conv(*BA2)
    attendu = [barre("inf", 1, 3, 20), barre("inf", 2, 2, 16, 1.5, 4.5), barre("sup", 1, 2, 12, 0.0, 4.5), barre("sup", 1, 2, 16, 4.5, 6.0),
               barre("peau", None, 1, 10)]
    if p["barres"] != attendu or p["zones"] != [zone(0.0, 1.5, 15.0), zone(1.5, 4.5, 25.0), zone(4.5, 6.0, 15.0)] or len(p["types_etriers"]) != 1 or p["L"] != 6.0:
        f.append(f"trois sections : barres {p['barres']}, zones {p['zones']}")
    res = m["M"].calculer(p)
    ml = 1.72 * m_lin(8)
    mbs = [3 * m_lin(20) + 2 * m_lin(12) + 2 * m_lin(10), 3 * m_lin(20) + 2 * m_lin(16) + 2 * m_lin(12) + 2 * m_lin(10), 3 * m_lin(20) + 2 * m_lin(16) + 2 * m_lin(10)]
    mes = [ml / 0.15, ml / 0.25, ml / 0.15]
    for x, a, b_ in zip((0.75, 3.0, 5.25), mbs, mes):
        lu = masse_au_metre(m, res, x)
        if not (pres(lu[0], a) and pres(lu[1], b_)):
            f.append(f"trois sections, x = {x} m : masse au mètre {lu} (à la main : {a}, {b_})")
    f += invariant(m, res, [1.5, 3.0, 1.5], mbs, mes, "trois sections")
    if [z["n"] for z in res["zones"]] != [10, 12, 11]:
        f.append(f"trois sections : cours {[z['n'] for z in res['zones']]} (10, 12, 10 + 1 de fermeture)")
    # ---- 3 · cadres + cadre intérieur + épingle ; surplus réuni ; latérales imposées
    #      lit 1 inférieur de 4 Ø 20 dans 40 cm : axes à 3 + 1,0 + 1,0 = 5 cm, soit 5 · 15 · 25 · 35 cm ; le cadre intérieur
    #      entoure les barres 2 et 3 : (25 + 1,0 + 0,8) − (15 − 1,0 − 0,8) = 13,6 cm hors tout
    p = conv(*BA3)
    t1 = {"nom": "E1", "lignes": [cadre(10), interieur(13.6, 8), epingle(8)], "cadre_talon": False}
    t2 = {"nom": "E2", "lignes": [cadre(10)], "cadre_talon": False}
    if p["types_etriers"] != [t1, t2] or [z["type"] for z in p["zones"]] != ["E1", "E2", "E2", "E1"] or [z["s"] for z in p["zones"]] != [10.0, 20.0, 20.0, 10.0]:
        f.append(f"cadres + épingle + cadre intérieur : types {p['types_etriers']}, zones {p['zones']}")
    # latérales : automatiques en A, B, D (70 − 5,0 − 5,0 = 60 cm > 30 : 1 Ø 10 par face), imposées 2 Ø 12 en C → le Ø
    # change : des barres par tronçon ; les 2 Ø 20 de surplus de B et C réunis de 1 à 5 m
    attendu = [barre("inf", 1, 4, 20), barre("inf", 1, 2, 20, 1.0, 5.0), barre("sup", 1, 3, 16), barre("peau", None, 1, 10, 0.0, 3.0),
               barre("peau", None, 2, 12, 3.0, 5.0), barre("peau", None, 1, 10, 5.0, 6.0)]
    if p["barres"] != attendu:
        f.append(f"cadres + épingle + cadre intérieur : barres {p['barres']}")
    res = m["M"].calculer(p)
    cours1 = (2 * ((40 - 6) + (70 - 6)) / 100.0 + 0.20) * m_lin(10) + (2 * (13.6 + (70 - 6)) / 100.0 + 0.16) * m_lin(8) + ((70 - 6) / 100.0 + 0.16) * m_lin(8)
    cours2 = (2 * ((40 - 6) + (70 - 6)) / 100.0 + 0.20) * m_lin(10)
    mbs = [4 * m_lin(20) + 3 * m_lin(16) + 2 * m_lin(10), 6 * m_lin(20) + 3 * m_lin(16) + 2 * m_lin(10), 6 * m_lin(20) + 3 * m_lin(16) + 4 * m_lin(12),
           4 * m_lin(20) + 3 * m_lin(16) + 2 * m_lin(10)]
    mes = [cours1 / 0.10, cours2 / 0.20, cours2 / 0.20, cours1 / 0.10]
    for x, a, b_ in zip((0.5, 2.0, 4.0, 5.5), mbs, mes):
        lu = masse_au_metre(m, res, x)
        if not (pres(lu[0], a) and pres(lu[1], b_)):
            f.append(f"quatre sections, x = {x} m : masse au mètre {lu} (à la main : {a}, {b_})")
    f += invariant(m, res, [1.0, 2.0, 2.0, 1.0], mbs, mes, "quatre sections")
    # ---- les refus : une longueur qui manque, nulle, de trop ; une poutre hors du domaine de ce module
    for longueurs, mots in ((None, ("Longueurs", "Longueur (m)")), ([1.5, 3.0], ("2 donnée(s) pour 3 section(s)",)), ([1.5, None, 1.5], ("Section B", "manque")),
                            ([1.5, 0.0, 1.5], ("Section B", "nulle")), ([1.5, -1.0, 1.5], ("Section B",)), ([1.5, "3", 1.5], ("Section B",)),
                            ([1.5, True, 1.5], ("Section B",))):
        if refuse(lambda: conv(BA2[0], longueurs), *mots) is None:
            f.append(f"longueurs {longueurs} : refus attendu avec {mots}")
    grande = poutre_ba([section_ba("A", [(3, 20)], [(2, 12)], E8, 60.0)])
    if refuse(lambda: conv(grande, [6.0]), "« V1 »", "pas", "50") is None:
        f.append("un pas de 60 cm (hors du domaine de ce module) : refusé, la poutre nommée")
    avant = copy.deepcopy(BA3[0])
    conv(*BA3)
    if BA3[0] != avant:
        f.append("convertir a modifié la poutre reçue")
    return f


def invariant(m, res, ls, mbs, mes, nom):
    """L'INVARIANT : sans retours, recouvrements ni cours de fermeture, la masse du module détaillé vaut
    Σ (masse au mètre de la section × l_i) — aux ABOUTS près pour les barres, qui s'arrêtent à l'enrobage : il manque
    c × (masse au mètre des barres de la première et de la dernière section). Les longueurs sont des multiples du pas."""
    f = []
    c = res["poutre"]["enrobage"] / 100.0
    barres = sum(r["masse"] for r in res["reperes"] if r["sorte"] == "barre")
    attendu_b = sum(a * l for a, l in zip(mbs, ls)) - c * (mbs[0] + mbs[-1])
    if any(r["detail"]["retours"] or r["detail"]["n_rec"] for r in res["reperes"] if r["sorte"] == "barre") or not pres(barres, attendu_b):
        f.append(f"{nom} : barres {barres} kg, Σ m·l − c·(m₁ + mₙ) = {attendu_b} kg")
    sans_fermeture = sum(z["cours"] * next(t for t in res["types"] if t["nom"] == z["type"])["masse_cours"] for z in res["zones"])
    attendu_e = sum(a * l for a, l in zip(mes, ls))
    if not pres(sans_fermeture, attendu_e):
        f.append(f"{nom} : étriers sans fermeture {sans_fermeture} kg, Σ m·l = {attendu_e} kg")
    etriers = sum(r["masse"] for r in res["reperes"] if r["sorte"] == "etrier")
    if etriers < attendu_e - 1e-9 or res["totaux"]["masse"] < attendu_b + attendu_e - 1e-9:
        f.append(f"{nom} : le module détaillé compte MOINS que le taux au mètre")
    return f


# ====================================================================== 9. la géométrie des coupes, les compositions
def fautes_geometrie(m):
    f = []
    r = m["M"].calculer(TYPE)
    cp = r["coupes"]
    if not ([c["cle"] for c in cp] == ["A", "B"] and cp[0]["plages"] == [[0.0, 1.2], [4.8, 6.0]] and cp[1]["plages"] == [[1.2, 4.8]]
            and cp[0]["reperes"] == [1, 3] and cp[1]["reperes"] == [1, 2, 3] and cp[0]["type"] == cp[1]["type"] == "E1" and cp[1]["pas"] == [15.0, 25.0]
            and pres(cp[0]["x"], 0.6) and pres(cp[1]["x"], 3.0) and r["coupes_omises"] == 0):
        f.append(f"compositions de la poutre type : {[(c['cle'], c['plages'], c['reperes'], c['x']) for c in cp]}")
    g = cp[1]["geometrie"]
    par = {b["index"]: b for b in g["barres"]}
    # la barre d'angle au contact du cadre : axe à c + Ø cadre + Ø/2 = 3 + 0,8 + 1,0 = 4,8 cm des deux parements
    if not (pres(par[0]["points"][0][0], 4.8) and pres(par[0]["points"][-1][0], 30 - 4.8) and pres(par[0]["points"][0][1], 4.8) and len(par[0]["points"]) == 3
            and pres(par[0]["points"][1][0], 15.0)):
        f.append(f"lit inférieur 1 (3 Ø 20) : {par[0]['points']} (angles à 4,8 cm, une barre au milieu)")
    # le second lit : au-dessus du premier, à l'espacement libre de max(Ø ; d_g + 5 ; 20) = 25 mm — axe à 3,8 + 2,0 + 2,5 + 0,8 = 9,1 cm
    if not (pres(par[1]["points"][0][1], 9.1) and pres(par[2]["points"][0][1], 60 - (3 + 0.8 + 0.6)) and r["elevation"]["niveaux"] == [[4.8], [9.1], [55.6]]):
        f.append(f"lit 2 à 9,1 cm, lit supérieur à 55,6 cm : {par[1]['points']}, {par[2]['points']}, {r['elevation']}")
    if g["cadres"] != [{"sorte": "cadre", "d": 8, "x0": 3.0, "x1": 27.0, "y0": 3.0, "y1": 57.0}]:
        f.append(f"le cadre : nus extérieurs à l'enrobage : {g['cadres']}")
    # une poutre chargée : aucune barre ne chevauche une autre, toutes tiennent dans le cadre ; épingle sur une barre ;
    # peau au contact des montants ; cadre de talon
    for nom, p in (("riche", RICHE), ("type + peau + épingle", poutre(barres=TYPE["barres"] + [barre("peau", None, 2, 10), barre("inf", 1, 2, 16, 1.0, 5.0)],
                                                                         types_etriers=[{"nom": "E1", "lignes": [cadre(8), epingle(8), interieur(10.0)], "cadre_talon": True}]))):
        res = m["M"].calculer(p)
        for c in res["coupes"]:
            g = c["geometrie"]
            pts = [(x, y, b["d"]) for b in g["barres"] for x, y in b["points"]]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) * 10 < (pts[i][2] + pts[j][2]) / 2.0 - 1e-6:
                        f.append(f"{nom}, coupe {c['cle']} : deux barres se chevauchent ({pts[i]}, {pts[j]})")
            if sum(len(b["points"]) for b in g["barres"]) != sum((2 if res["poutre"]["barres"][i]["famille"] == "peau" else 1) * res["poutre"]["barres"][i]["n"] for i in c["barres"]):
                f.append(f"{nom}, coupe {c['cle']} : toutes les barres présentes sont dessinées, et elles seules")
            cadres = [k for k in g["cadres"] if k["sorte"] == "cadre"]
            talons = [k for k in g["cadres"] if k["sorte"] == "talon"]
            if cadres:
                k = cadres[0]
                boite = talons[0] if talons else k
                for x, y, d in pts:
                    dedans_ame = k["x0"] + k["d"] / 10 - 1e-6 <= x - d / 20 and x + d / 20 <= k["x1"] - k["d"] / 10 + 1e-6
                    dedans_talon = talons and boite["x0"] <= x - d / 20 and x + d / 20 <= boite["x1"] and y <= boite["y1"]
                    if not (dedans_ame or dedans_talon) or not (k["y0"] <= y - d / 20 and y + d / 20 <= k["y1"]):
                        f.append(f"{nom}, coupe {c['cle']} : une barre hors du cadre ({x}, {y}, Ø {d})")
                for e in g["epingles"]:
                    if not (k["x0"] < e["x"] < k["x1"]) or not any(abs(e["x"] - x) < 1e-6 for x, y, d in pts):
                        f.append(f"{nom}, coupe {c['cle']} : l'épingle (x = {e['x']}) n'est pas sur une barre du cadre")
                for k2 in g["cadres"]:
                    if k2["sorte"] == "cadre_interieur" and not (k["x0"] - 1e-9 <= k2["x0"] < k2["x1"] <= k["x1"] + 1e-9 and pres(k2["x1"] - k2["x0"], [li["largeur"] for t in res["poutre"]["types_etriers"] for li in t["lignes"] if li["forme"] == "cadre_interieur"][0])):
                        f.append(f"{nom}, coupe {c['cle']} : cadre intérieur hors du cadre, ou d'une autre largeur ({k2})")
            if talons and not (pres(talons[0]["x0"], res["section"]["c"]) and pres(talons[0]["x1"], res["section"]["b_tot"] - res["section"]["c"])
                               and pres(talons[0]["y1"], res["section"]["h_talon"] - res["section"]["c"])):
                f.append(f"{nom}, coupe {c['cle']} : cadre de talon {talons[0]}")
    # au plus quatre coupes, les plus longues ; le reste est compté
    p = poutre(barres=[barre("inf", 1, 3, 20)] + [barre("inf", 2, 2, 16, 0.5 * k, 0.5 * k + 0.3) for k in range(1, 9)], zones=[zone(0.0, 6.0, 15.0)])
    res = m["M"].calculer(p)
    if len(res["coupes"]) != 4 or res["coupes_omises"] != 5 or [c["cle"] for c in res["coupes"]] != ["A", "B", "C", "D"]:
        f.append(f"neuf compositions : quatre coupes montrées, cinq comptées à part ({len(res['coupes'])}, {res['coupes_omises']})")
    # une portion sans étriers est une composition à part, sans cadre
    res = m["M"].calculer(poutre(zones=[zone(0.0, 1.5, 15.0)]))
    sans = [c for c in res["coupes"] if c["type"] is None]
    if not sans or any(c["geometrie"]["cadres"] for c in sans):
        f.append("une portion sans étriers : une coupe sans cadre")
    return f


# ====================================================================== 10. le format, les formules, les avertissements
EVAL = {
    "x1": lambda t: max(t["de"], t["c"]),
    "x2": lambda t: min(t["à"], t["L"] - t["c"]),
    "l_droite": lambda t: t["x_2"] - t["x_1"],
    "l_retour_auto": lambda t: (t["h"] if "h" in t else t["b"]) - 2 * t["c"],
    "l_dev": lambda t: t["l_d"] + t["r"] * t["l_r"] / 100.0,
    "l_rec": lambda t: t["k"] * t["Ø"] / 1000.0,
    "n_rec": lambda t: math.ceil((t["l_0"] - t["l_com"]) / (t["l_com"] - t["l_rec"]) - 1e-9),
    "l_unitaire": lambda t: t["l_d"] + t.get("r", 0) * t.get("l_r", 0.0) / 100.0 + t.get("n_rec", 0) * t.get("l_rec", 0.0),
    "n_peau": lambda t: t["n_face"] * t["faces"],
    "l_totale": lambda t: t["n"] * t["l"],
    "m_lin": lambda t: math.pi * (t["Ø"] / 1000.0) ** 2 / 4.0 * t["ρ_s"],
    "masse": lambda t: t["L"] * t["m"],
    "l_r_etrier": lambda t: max(10 * t["Ø"], 70),
    "contour_cadre": lambda t: 2 * ((t["b"] - 2 * t["c"]) + (t["h"] - 2 * t["c"])) / 100.0,
    "contour_interieur": lambda t: 2 * (t["w"] + (t["h"] - 2 * t["c"])) / 100.0,
    "contour_epingle": lambda t: (t["h"] - 2 * t["c"]) / 100.0,
    "contour_talon": lambda t: 2 * ((t["b_tot"] - 2 * t["c"]) + (t["h_t"] - 2 * t["c"])) / 100.0,
    "l_etrier": lambda t: t["p"] + 2 * t["l_r"] / 1000.0,
    "n_cours": lambda t: sum(t.values()),
    "l_zone": lambda t: t["à"] - t["de"],
    "quotient": lambda t: t["l_z"] / (t["s"] / 100.0),
    "cours": lambda t: math.ceil(t["q"] - 1e-9),
    "n_zone": lambda t: t["n_c"] + t["fermeture"],
    "A_c": lambda t: t["b"] * t["h"] + t.get("b_g", 0) * t.get("h_g", 0) + t.get("b_d", 0) * t.get("h_d", 0),
    "V_c": lambda t: t["A_c"] / 1e4 * t["L"],
    "M_c": lambda t: t["V_c"] * t["ρ_c"],
    "L_d": lambda t: sum(t.values()),
    "M_s": lambda t: sum(t.values()),
    "M_maj": lambda t: t["M_s"] * (1 + t["maj"] / 100.0),
    "taux": lambda t: t["M_s,maj"] / t["V_c"],
    "taux_net": lambda t: t["M_s"] / t["V_c"],
    "kg_m": lambda t: t["M_s,maj"] / t["L"],
}


def formules_de(res):
    out = list(res["beton"]["formules"]) + list(res["totaux"]["formules"])
    for bloc in res["reperes"] + res["zones"] + res["diametres"]:
        out += bloc["formules"]
    return out


def fautes_formules(m):
    """Chaque formule du moteur, recalculée ICI à partir de ses seuls termes, retombe sur son résultat ; et chaque
    grandeur du résultat est le résultat de sa formule."""
    f, n = [], 0
    for nom, p in (("type", TYPE), ("riche", RICHE), ("peau + retours", poutre(barres=[barre("peau", None, 2, 10, retours=2), barre("sup", 2, 3, 25, 1.0, None, 1, 35.0)]))):
        res = m["M"].calculer(p)
        for fx in formules_de(res):
            n += 1
            if fx["cle"] not in EVAL:
                f.append(f"{nom} : formule inconnue du test « {fx['cle']} »")
                continue
            t = {x["symbole"]: x["valeur"] for x in fx["termes"]}
            if len(t) != len(fx["termes"]):
                f.append(f"{nom} : {fx['cle']} — deux termes du même symbole")
            v = EVAL[fx["cle"]](t)
            if not pres(v, fx["resultat"]):
                f.append(f"{nom} : {fx['cle']} = {v} à partir de ses termes, le moteur écrit {fx['resultat']}")
        for r in res["reperes"]:
            derniers = {fx["cle"]: fx["resultat"] for fx in r["formules"]}
            if not (pres(derniers["masse"], r["masse"]) and pres(derniers["l_totale"], r["l_totale"]) and pres(derniers["m_lin"], r["masse_lineique"])
                    and pres(derniers.get("l_unitaire", derniers.get("l_etrier")), r["l_unitaire"])):
                f.append(f"{nom} : repère {r['repere']} — ses formules ne rendent pas ses grandeurs")
        tf = {fx["cle"]: fx["resultat"] for fx in res["totaux"]["formules"]}
        if not (pres(tf["taux"], res["totaux"]["taux_majore"]) and pres(tf["M_s"], res["totaux"]["masse"]) and pres(tf["kg_m"], res["totaux"]["kg_m_majore"])):
            f.append(f"{nom} : totaux — les formules ne rendent pas les totaux")
    fautes_formules.n = n
    return f


def fautes_format(m):
    f = []
    for nom, p in (("type", TYPE), ("riche", RICHE)):
        res = m["M"].calculer(p)
        try:
            txt = json.dumps(res, allow_nan=False)
        except ValueError as e:
            f.append(f"{nom} : la sortie n'est pas du JSON strict ({e})")
            continue
        if json.loads(txt) != json.loads(json.dumps(m["M"].calculer(json.loads(json.dumps(p))))):
            f.append(f"{nom} : le calcul change après un aller-retour JSON de la poutre")
        if res["poutre"] != m["EN"].normaliser(p):
            f.append(f"{nom} : la sortie porte la poutre normalisée")
    # les invariants : doubler L (zones et renfort aux mêmes proportions) double l'acier à l'about près
    r1 = m["M"].calculer(poutre(majoration=0.0))
    r2 = m["M"].calculer(poutre(majoration=10.0))
    if not (pres(r2["totaux"]["masse"], r1["totaux"]["masse"]) and pres(r2["totaux"]["masse_majoree"], 1.1 * r1["totaux"]["masse"])
            and pres(r1["totaux"]["taux"], r1["totaux"]["taux_majore"])):
        f.append("la majoration ne touche que le total majoré")
    r3 = m["M"].calculer(poutre(rho_acier=7850.0 * 1.01, rho_beton=2400.0))
    if not (pres(r3["totaux"]["masse"], 1.01 * r1["totaux"]["masse"]) and pres(r3["beton"]["masse"], 1.08 * 2400.0) and pres(r3["beton"]["volume"], 1.08)):
        f.append("les masses volumiques : l'acier proportionnel à ρ_s, le béton à ρ_c")
    s = m["M"].synthese_note([m["M"].calculer(TYPE), m["M"].calculer(RICHE)])
    ra, rb = m["M"].calculer(TYPE), m["M"].calculer(RICHE)
    if not (s["n"] == 2 and pres(s["volume"], ra["beton"]["volume"] + rb["beton"]["volume"]) and pres(s["masse_majoree"], ra["totaux"]["masse_majoree"] + rb["totaux"]["masse_majoree"])
            and pres(s["taux_majore"], s["masse_majoree"] / s["volume"]) and pres(s["taux"], (ra["totaux"]["masse"] + rb["totaux"]["masse"]) / s["volume"])):
        f.append(f"synthèse de la note : {s}")
    return f


def fautes_avertissements(m):
    f = []

    def codes(p):
        return [(a["code"], a["grave"]) for a in m["M"].calculer(p)["avertissements"]]

    def messages(p, code):
        return [a["message"] for a in m["M"].calculer(p)["avertissements"] if a["code"] == code]
    # une portion sans étriers : dite avec ses abscisses, grave ; sans aucune zone : toute la longueur
    ms = messages(poutre(zones=[zone(0.5, 1.5, 15.0), zone(4.5, 6.0, 15.0)]), "sans_etriers")
    if not (len(ms) == 2 and "de 0,00 à 0,50 m" in ms[0] and "de 1,50 à 4,50 m (3,00 m)" in ms[1]):
        f.append(f"portions sans étriers : {ms}")
    if m["M"].calculer(poutre(zones=[zone(0.5, 1.5, 15.0), zone(4.5, 6.0, 15.0)]))["trous"] != [[0.0, 0.5], [1.5, 4.5]]:
        f.append("trous : [[0 ; 0,5], [1,5 ; 4,5]]")
    # … et au bout droit (la poutre type sans sa dernière zone) : de 4,50 à 6,00 m
    ms = messages(poutre(zones=[zone(0.0, 1.5, 15.0), zone(1.5, 4.5, 25.0)]), "sans_etriers")
    if not (len(ms) == 1 and "de 4,50 à 6,00 m (1,50 m)" in ms[0]) or m["M"].calculer(poutre(zones=[zone(0.0, 1.5, 15.0), zone(1.5, 4.5, 25.0)]))["trous"] != [[4.5, 6.0]]:
        f.append(f"portion sans étriers au bout droit : {ms}")
    ms = messages(poutre(zones=[]), "sans_etriers")
    if not (len(ms) == 1 and "toute la longueur" in ms[0] and "6,00 m" in ms[0]) or ("sans_etriers", True) not in codes(poutre(zones=[])):
        f.append(f"aucune zone : {ms}")
    if ("retour_nul", True) not in codes(poutre(barres=[barre(retours=2, l_retour=0.0)])) or any(c == "retour_nul" for c, _ in codes(poutre(barres=[barre(retours=0, l_retour=0.0)]))):
        f.append("un retour saisi à 0 cm est dit (grave) ; sans retour, rien")
    if ("epingle_inutile", False) not in codes(poutre(b=25.0, types_etriers=[{"nom": "E1", "lignes": [cadre(), epingle()], "cadre_talon": False}])) \
            or any(c == "epingle_inutile" for c, _ in codes(poutre(types_etriers=[{"nom": "E1", "lignes": [cadre(), epingle()], "cadre_talon": False}]))):
        f.append("épingle dans une âme de moins de 20 cm entre nus de cadre : dite ; à 24 cm : rien")
    if ("sans_barres", False) not in codes(poutre(barres=[])):
        f.append("aucune barre : dit")
    # 300 × 300 cm, cadre Ø 20 : 2·(294 + 294) cm + 2 × 20 cm = 12,16 m > 12 m ; en Ø 8 : 11,92 m, rien
    gros = dict(b=300.0, h=300.0, barres=[])
    if ("cadre_long", True) not in codes(poutre(types_etriers=[{"nom": "E1", "lignes": [cadre(20)], "cadre_talon": False}], **gros)) \
            or any(c == "cadre_long" for c, _ in codes(poutre(**gros))):
        f.append("un cadre de 12,16 m, plus long que la longueur commerciale : dit ; un cadre de 11,92 m : rien")
    return f


TOUS = (("la poutre type à la main", fautes_type), ("masses linéiques", fautes_masses), ("barres", fautes_barres), ("recouvrements", fautes_recouvrements),
        ("longueurs des étriers", fautes_etriers_longueurs), ("compte des étriers", fautes_compte), ("talons", fautes_talons), ("bordereau et CSV", fautes_bordereau),
        ("refus", fautes_refus), ("normalisation", fautes_normalisation), ("saisie", fautes_saisie), ("depuis une poutre", fautes_depuis_poutre),
        ("géométrie", fautes_geometrie), ("formules", fautes_formules), ("format", fautes_format), ("avertissements", fautes_avertissements))


# ====================================================================== les contrôles, sur le moteur du dépôt
print("=== 1. La poutre type, à la main (math seul) ===")
chk("le calcul à la main retombe sur les valeurs écrites en clair (5,94 m ; 17,82 m ; 3,60 m ; 1,72 m ; 33 cours ; 56,76 m ; 1,080 m³)",
    all(pres(MAIN[k], v) for k, v in ECRIT.items()), {k: MAIN[k] for k in ECRIT})
chk("… et sur les masses arrondies : 43,947 + 11,364 + 10,547 + 22,397 = 88,255 kg ; × 1,05 = 92,667 kg ; / 1,080 m³ = 85,80 kg/m³",
    all(round(MAIN[k], nd) == v for k, (v, nd) in ARRONDIS.items()), {k: MAIN[k] for k in ARRONDIS})
ff = fautes_type(REEL)
chk("le moteur rend la poutre type du calcul à la main : chaque repère (l, L, m, M), chaque total, 85,8 kg/m³, aucun avertissement", not ff, ff)

print("\n=== 2. Les formules, une à une ===")
ff = fautes_masses(REEL)
chk(f"masse linéique π·Ø²/4·7 850 : les {len(NOMINALES)} Ø à moins de 0,5 % des valeurs nominales de l'EN 10080, six au millième", not ff, ff)
ff = fautes_barres(REEL)
chk("barre filante = L − 2c ; bouts bornés à l'enrobage ; retours automatiques (h − 2c, b − 2c pour la peau) ou saisis ; peau sur deux faces", not ff, ff)
ff = fautes_recouvrements(REEL)
chk("recouvrements : 14 m → un de 50 Ø ; 23,90 m → DEUX (la pose le prouve) ; grille contre la pose ; désactivés : dit", not ff, ff)
chk("l'écart avec la conception est réel : ⌈23,9 / 12⌉ − 1 = 1 recouvrement, quand deux barres de 12 m se recouvrant de 1 m ne couvrent que 23 m",
    math.ceil(23.9 / 12.0) - 1 == 1 and 2 * 12.0 - 1.0 < 23.9 and pose_recouvrements(23.9, 12.0, 1.0) == 2)
ff = fautes_etriers_longueurs(REEL)
chk("cadre 2·[(b − 2c) + (h − 2c)] + 2·l_r ; cadre intérieur 2·[w + (h − 2c)] + 2·l_r ; épingle (h − 2c) + 2·l_r ; l_r = max(10 Ø ; 70 mm)", not ff, ff)

print("\n=== 3. Le compte des étriers ===")
ff = fautes_compte(REEL)
chk(f"le cas d'hier (4 étriers), les zones jointives, les suites disjointes, et la grille : {fautes_compte.n} cas = la pose exacte (jamais un de moins)",
    not ff and fautes_compte.n > 20000, ff)
n_anc, moins_anc = ancienne_formule()
chk(f"l'ancienne formule int(L/s) + 1 perdait un étrier dans {moins_anc} des {n_anc} zones de cette grille dont la longueur est un multiple exact du pas "
    f"({100 * moins_anc / n_anc:.1f} %, par la seule troncature des flottants) : le défaut était réel", moins_anc > 0, (n_anc, moins_anc))

print("\n=== 4. Les talons ===")
ff = fautes_talons(REEL)
chk("deux talons : aire = âme + les deux (2 600 cm²), polygone par le lacet ; le cadre de talon compté à chaque cours (1,72 m × 33)", not ff, ff)

print("\n=== 5. Le bordereau ===")
ff = fautes_bordereau(REEL)
chk("Σ repères = Σ familles = Σ diamètres = total ; les parts font 1 ; le CSV (« ; », virgule) se relit et redonne les mêmes nombres", not ff, ff)

print("\n=== 6. Les refus ===")
ff = fautes_refus(REEL)
chk(f"{len(refus_attendus())} saisies hors domaine : chacune refusée par ValueError, en français, le champ nommé, l'entrée intacte", not ff, ff)
ff = fautes_normalisation(REEL)
chk("normalisation : la poutre type, les défauts d'une clé absente, l'idempotence, les zones de gauche à droite", not ff, ff)

print("\n=== 7. Les gestes de la saisie ===")
ff = fautes_saisie(REEL)
chk("ajouts (barres, type, ligne, zone), type employé non retiré, limite de deux zones qui suit, longueur : ce qui tient à l'about droit le suit", not ff, ff)

print("\n=== 8. Recevoir une poutre de « Poutres béton armé » ===")
ff = fautes_depuis_poutre(REEL)
chk("trois poutres converties : barres communes filantes, surplus en renfort réuni, Ø qui change par tronçon, latérales, types, zones ; "
    "la masse au mètre de chaque tronçon à la main ; l'invariant aux abouts près ; les refus", not ff, ff)

print("\n=== 9. La géométrie ===")
ff = fautes_geometrie(REEL)
chk("coupes : barre d'angle à c + Ø cadre + Ø/2, second lit à l'espacement libre de l'EC2, rien ne se chevauche ni ne sort du cadre ; compositions", not ff, ff)

print("\n=== 10. Le format ===")
ff = fautes_formules(REEL)
chk(f"chaque formule du moteur ({fautes_formules.n}) recalculée à partir de ses seuls termes retombe sur son résultat", not ff and fautes_formules.n > 150, ff)
ff = fautes_format(REEL)
chk("JSON strict, stable après un aller-retour ; majoration, masses volumiques, synthèse de la note", not ff, ff)
ff = fautes_avertissements(REEL)
chk("avertissements : portion sans étriers (ses abscisses), retour de 0 cm, épingle sans utilité, cadre trop long, aucune barre", not ff, ff)

# ====================================================================== 11. pureté
print("\n=== 11. Pureté : la bibliothèque standard seule, sous Python 3.12 et 3.13 ===")
non_std = []
for fichier in sorted(os.listdir(os.path.join(RACINE, "beton", "taux"))):
    if fichier.endswith(".py"):
        for n in ast.walk(ast.parse(lire("beton", "taux", fichier))):
            noms = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
            non_std += [f"{fichier}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names | {"beton"}]
chk("aucun fichier de beton/taux n'importe autre chose que la bibliothèque standard et le moteur des poutres (lecture des sources)", not non_std, non_std)
SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("streamlit", "pandas", "reportlab", "numpy", "matplotlib", "scipy", "pymupdf", "fitz", "PIL")
class Refus:
    """Refuse tout chargement des bibliothèques interdites."""
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("chargement refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
try:
    import numpy  # noqa: F401
    refus = False
except ImportError:
    refus = True
sys.path.insert(0, sys.argv[1])
from beton.taux import moteur as M, depuis_poutre as DP
entree = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.calculer(p) for p in entree["poutres"]],
          "conv": [DP.convertir(p, ls) for p, ls in entree["ba"]], "csv": [M.bordereau_csv(M.calculer(p)) for p in entree["poutres"]]}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"beton"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
ENTREE = {"poutres": [TYPE, RICHE], "ba": [list(BA1), list(BA2), list(BA3)]}
ICI = {"res": [M.calculer(p) for p in ENTREE["poutres"]], "conv": [DP.convertir(p, ls) for p, ls in ENTREE["ba"]],
       "csv": [M.bordereau_csv(M.calculer(p)) for p in ENTREE["poutres"]]}
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(ENTREE), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans bibliothèque tierce", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : numpy, streamlit, pandas… REFUSÉS par sys.meta_path, et le moteur calcule, convertit et écrit le CSV quand même",
        out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : le moteur n'a chargé que la bibliothèque standard", not out["tiers"], out["tiers"])
    chk(f"Python {v} : résultats identiques à ceux de ce processus, au bit près",
        all(out[k] == json.loads(json.dumps(ICI[k])) for k in ("res", "conv", "csv")))
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")

# ====================================================================== 12. mutations
print("\n=== 12. Mutations : le moteur modifié d'UNE règle est rejeté ===")
PAQUET = sys.modules["beton.taux"]
NOMS_MODULES = ("entree", "geometrie", "moteur", "saisie", "depuis_poutre")
ALIAS = {"entree": "EN", "geometrie": "GE", "moteur": "M", "saisie": "SA", "depuis_poutre": "DP"}


def charger_mutant(mutations):
    """Les cinq modules du moteur rechargés depuis leurs SOURCES après des remplacements [(module, avant, après)] ;
    chaque « avant » doit figurer UNE fois dans sa source (sinon la mutation est périmée). Rend les modules mutants ;
    ceux du dépôt sont remis en place."""
    sauve = {n: sys.modules[f"beton.taux.{n}"] for n in NOMS_MODULES}
    sauve_attr = {n: getattr(PAQUET, n) for n in NOMS_MODULES}
    mods = {}
    try:
        for n in NOMS_MODULES:
            chemin = os.path.join(RACINE, "beton", "taux", n + ".py")
            src = lire("beton", "taux", n + ".py")
            for module, avant, apres in mutations:
                if module == n:
                    if src.count(avant) != 1:
                        raise AssertionError(f"mutation périmée : « {avant} » ({src.count(avant)} fois dans {n}.py)")
                    src = src.replace(avant, apres)
            mod = types.ModuleType(f"beton.taux.{n}")
            mod.__package__, mod.__file__ = "beton.taux", chemin
            sys.modules[mod.__name__] = mod
            setattr(PAQUET, n, mod)
            exec(compile(src, chemin, "exec"), mod.__dict__)  # noqa: S102 — le moteur lui-même, muté
            mods[ALIAS[n]] = mod
    finally:
        sys.modules.update({f"beton.taux.{n}": sauve[n] for n in NOMS_MODULES})
        for n in NOMS_MODULES:
            setattr(PAQUET, n, sauve_attr[n])
    return mods


def rejets(mods):
    """Ce que les contrôles de ce fichier reprochent au moteur ``mods`` (la grille des étriers allégée d'un facteur 7)."""
    out = []
    for nom, fonction in TOUS:
        try:
            ff_ = fonction(mods, complet=False) if fonction is fautes_compte else fonction(mods)
        except AssertionError:
            raise
        except Exception as err:  # noqa: BLE001 — un mutant qui casse est rejeté
            ff_ = [f"{type(err).__name__} : {err}"]
        out += [f"{nom} : {x}" for x in ff_[:2]]
    return out


MUTATIONS = (
    ("étriers : l'ancienne formule, le quotient TRONQUÉ (int(l/s) + 1)", [("moteur", "return int(math.ceil(l_z / (s_cm / 100.0) - ARRONDI))", "return int(l_z / (s_cm / 100.0))")]),
    ("étriers : l'arrondi supérieur sans sa tolérance (0,9 / 0,1 = 9,000000000000002 compte 10)", [("moteur", "return int(math.ceil(l_z / (s_cm / 100.0) - ARRONDI))", "return int(math.ceil(l_z / (s_cm / 100.0)))")]),
    ("étriers : le cours de fermeture oublié", [("moteur", "f = 1 if i == suite[-1] else 0", "f = 0")]),
    ("étriers : un cours de fermeture par ZONE (l'étrier commun doublé)", [("moteur", "f = 1 if i == suite[-1] else 0", "f = 1")]),
    ("étriers : deux zones jointives prises pour deux suites", [("moteur", 'if suites and zones[i]["de"] <= zones[suites[-1][-1]]["a"] + EN.EPS_X:', "if False:")]),
    ("étriers : le nombre d'un type oublie les cours de fermeture", [("moteur", 'n_t = sum(z["n"] for z in ses_zones)', 'n_t = sum(z["cours"] for z in ses_zones)')]),
    ("barre filante comptée sur L, sans l'enrobage d'about", [("entree", "return max(de, c), min(a, L - c)", "return de, a")]),
    ("retours oubliés dans la longueur d'une barre", [("moteur", "l_0 = l_d + r * l_r / 100.0", "l_0 = l_d")]),
    ("retour automatique de 0 cm (l'ancien écran)", [("moteur", 'l_r = (cote - 2.0 * c) if auto else b["l_retour"]', 'l_r = 0.0 if auto else b["l_retour"]')]),
    ("recouvrements : la formule de la conception, ⌈l₀/l_com⌉ − 1", [("moteur", 'return int(math.ceil((l_0 - rec["l_com"]) / (rec["l_com"] - l_rec) - ARRONDI)), l_rec',
                                                                    'return int(math.ceil(l_0 / rec["l_com"] - ARRONDI)) - 1, l_rec')]),
    ("recouvrements jamais comptés", [("moteur", 'return int(math.ceil((l_0 - rec["l_com"]) / (rec["l_com"] - l_rec) - ARRONDI)), l_rec', "return 0, l_rec")]),
    ("recouvrements : la longueur des retours oubliée dans la longueur à couvrir", [("moteur", 'n_rec, l_rec = recouvrements(l_0, b["d"], p["recouvrements"])',
                                                                                    'n_rec, l_rec = recouvrements(l_d, b["d"], p["recouvrements"])')]),
    ("barres de peau sur une seule face", [("moteur", 'n = 2 * b["n"] if peau else b["n"]', 'n = b["n"]')]),
    ("cadres sans leurs retours", [("moteur", "l_u = contour + 2.0 * l_r / 1000.0", "l_u = contour")]),
    ("retour d'un cadre sans le minimum de 70 mm", [("moteur", "return max(RETOUR_K * d_mm, RETOUR_MIN)", "return RETOUR_K * d_mm")]),
    ("retour d'un cadre de 8 Ø (l'ancien écran)", [("moteur", "RETOUR_K = 10.0        # × Ø", "RETOUR_K = 8.0        # × Ø")]),
    ("épingle comptée comme un cadre sans largeur (deux brins)", [("moteur", "        return h / 100.0", "        return 2.0 * h / 100.0")]),
    ("cadre intérieur compté à la largeur de l'âme", [("moteur", "        return 2.0 * (largeur + h) / 100.0", "        return 2.0 * (w + h) / 100.0")]),
    ("cadre de talon jamais compté (l'ancien écran)", [("moteur", 'if (t["cadre_talon"] and EN.a_talon(p["forme"])) else None', "if False else None")]),
    ("cadre de talon au contour de l'âme", [("moteur", "        return 2.0 * (wt + ht) / 100.0", "        return 2.0 * (w + ht) / 100.0")]),
    ("cadre de talon : le plus BAS des deux talons", [("geometrie", '"h_talon": max(hg, hd)', '"h_talon": min(hg, hd)')]),
    ("deux talons calculée sans ses talons (l'ancien écran)", [("geometrie", '"aire_cm2": b * h + bg * hg + bd * hd', '"aire_cm2": b * h')]),
    ("deux talons : le talon droit oublié", [("entree", '"deux_talons": ("talon_gauche", "talon_droit")}\nLIBELLES_TALON', '"deux_talons": ("talon_gauche",)}\nLIBELLES_TALON')]),
    ("masse linéique : π·Ø²/2", [("moteur", "return math.pi * (d_mm / 1000.0) ** 2 / 4.0 * rho", "return math.pi * (d_mm / 1000.0) ** 2 / 2.0 * rho")]),
    ("volume de béton : A_c en dm²", [("moteur", "volume = aire / 1.0e4 * L", "volume = aire / 1.0e3 * L")]),
    ("majoration oubliée", [("moteur", "masse_maj = masse * (1.0 + maj / 100.0)", "masse_maj = masse")]),
    ("le taux affiché n'est pas majoré", [("moteur", '"taux": masse / volume, "taux_majore": masse_maj / volume, "kg_m"', '"taux": masse / volume, "taux_majore": masse / volume, "kg_m"')]),
    ("la masse par diamètre oublie un repère", [("moteur", 'rs = [r for r in reperes if r["d"] == d]\n        l_d', 'rs = [r for r in reperes if r["d"] == d][:1]\n        l_d')]),
    ("zones qui se recouvrent admises (étriers comptés deux fois)", [("entree", 'if v["de"] < u["a"] - EPS_X:', 'if False and v["de"] < u["a"] - EPS_X:')]),
    ("zone « à » ≤ « de » admise", [("entree", '    if out["a"] <= out["de"] + EPS_X:\n        raise ValueError(f"{ch} : « à » ({nombre_fr(out[\'a\'])} m) doit dépasser « de » ({nombre_fr(out[\'de\'])} m)")\n    if not isinstance(z["type"], str)',
                                     '    if not isinstance(z["type"], str)')]),
    ("une valeur hors bornes admise", [("entree", "    if x < lo or x > hi:\n        u_", "    if False:\n        u_")]),
    ("une clé inconnue admise", [("entree", "    if inconnues:\n", "    if False:\n")]),
    ("portion sans étriers au bout droit : muette", [("moteur", "    if x < L - EN.EPS_X:\n        trous.append([x, L])", "    if False:\n        trous.append([x, L])")]),
    ("avertissement « barre longue » muet", [("moteur", 'if not rec["actifs"] and d["l_dev"] > rec["l_com"] + EN.EPS_X:', "if False:")]),
    ("CSV au point décimal", [("moteur", '    return f"{x:.{nd}f}".replace(".", ",")', '    return f"{x:.{nd}f}"')]),
    ("CSV séparé par des virgules", [("moteur", 'w = csv.writer(sortie, delimiter=";", lineterminator="\\r\\n")', 'w = csv.writer(sortie, delimiter=",", lineterminator="\\r\\n")')]),
    ("coupe : la barre d'angle à c + Ø/2 (la convention de l'ancien écran)", [("geometrie", "        nu = c + d_cadre / 10.0                       # le nu du lit", "        nu = c                                        # le nu du lit")]),
    ("coupe : deux lits au même niveau", [("geometrie", "                nu += (d_max + DP.s_min(d_max, d_suiv, D_GRANULAT)) / 10.0", "                nu += 0.0")]),
    ("saisie : un type employé se retire", [("saisie", "    if zones:\n        un = len(zones) == 1", "    if False:\n        un = len(zones) == 1")]),
    ("saisie : la zone voisine ne suit plus la limite", [("saisie", "            v[autre] = x\n", "            pass\n")]),
    ("saisie : la longueur ne déplace plus ce qui tient à l'about droit", [("saisie", "        return _mm(x + delta) if x > L0 / 2.0 + EN.EPS_X else x", "        return x")]),
    ("saisie : « + Zone » ne propose plus la suite", [("saisie", 'z = {"de": _mm(de), "a": _mm(a),', 'z = {"de": 0.0, "a": _mm(p["L"]),')]),
    ("depuis une poutre : les filantes prennent le PLUS GRAND nombre de barres", [("depuis_poutre", "        commun = min(ns)", "        commun = max(ns)")]),
    ("depuis une poutre : le surplus n'est pas réuni entre tronçons voisins", [("depuis_poutre", "        if out and out[-1][2] == v:", "        if False:")]),
    ("depuis une poutre : un cadre intérieur devient un cadre de toute la largeur", [("depuis_poutre", '            elif emprise["complet"]:', "            elif True:")]),
    ("depuis une poutre : une longueur nulle admise", [("depuis_poutre", "        if l <= 0:", "        if l < 0:")]),
    ("depuis une poutre : les latérales oubliées", [("depuis_poutre", '    barres += _lignes("peau", None, laterales, xs)', "    pass")]),
    ("depuis une poutre : chaque section reçoit le pas de la première", [("depuis_poutre", '"s": float(s["pas"]), "type": noms[cle]', '"s": float(sections[0]["pas"]), "type": noms[cle]')]),
)
TEMOIN = ("témoin : un commentaire changé", [("moteur", "# ---- le béton", "# ---- le béton (témoin)")])
R0 = M.calculer(TYPE)
survivants, n_rejets = [], []
for nom, muts in MUTATIONS:
    try:
        rej = rejets(charger_mutant(muts))
    except AssertionError as err:
        survivants.append(f"{nom} : {err}")
        continue
    n_rejets.append(len(rej))
    if not rej:
        survivants.append(nom)
chk(f"{len(MUTATIONS)} mutations volontaires (étriers : troncature, tolérance, fermeture, jonction ; barres : about, retours, recouvrements, peau ; "
    f"cadres : retours, minimum, épingle, intérieur, talon ; béton ; masse linéique ; majoration ; refus ; avertissements ; CSV ; coupe ; saisie ; "
    f"poutre reçue) : TOUTES rejetées (de {min(n_rejets or [0])} à {max(n_rejets or [0])} reproches chacune)", not survivants, survivants)
chk("la mutation témoin (un commentaire) n'est rejetée par aucun contrôle : les rejets viennent bien des mutations", not rejets(charger_mutant(TEMOIN[1])))
chk("après les mutations, le moteur chargé est intact (mêmes résultats)", M.calculer(TYPE) == R0 and sys.modules["beton.taux.moteur"] is M)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)
