# -*- coding: utf-8 -*-
"""Moteur « Maturité du béton » (beton/maturite), REFAIT sur la norme le 10/11/2026 (docs/site/LOT_Q_QUATRE_MODULES.md
§ 2) : il ne reproduit plus l'écran Streamlit. Chaque formule est recalculée ICI, à la main (``math`` seul, valeurs
écrites en clair), et chaque contrôle rougit si l'on retire ce qu'il protège (§ 13, mutations).

  1. Les tables et les lois : classes, durcissement (s du ciment, s du bon de livraison, s(r), r(s)), les trois lois de
     température (chaque point du tableau belge, son interpolation, l'exemple de la norme, le plafond, le prolongement ;
     (B.10) ; Arrhenius), β_cc, f_cm(t), f_ck(t), la forme fermée face à un balayage numérique fin.
  2. Le temps : l'heure de fin de bétonnage, l'âge équivalent d'un profil de six jours, son inverse exact, la
     prolongation, le relevé posé sur le calendrier du coulage (jours d'avant, jours manquants, jours sans donnée).
  3. Un exemple complet déroulé de bout en bout, à la main.
  4. Les objectifs : le défaut d'ouverture (10,73 j ; 6,46 j en ciment N), le défaut n° 1 de l'ancien écran (jour 8, plus
     4,8 j), les deux bases, la variante prudente, l'opération datée, les instants à l'heure près.
  5. Le gel avant 5 MPa et le rappel « temps froid » ; la minimale horaire de la météo.
  6. L'essai sur éprouvettes et le calage de la courbe.
  7. Le sélecteur de classe.
  8. L'entrée : le coulage neuf, la normalisation, le refus de chaque borne, une note de l'ancien format.
  9. Le format : JSON strict, « symbole = expression = valeurs = résultat » recalculé à partir des seuls termes, les
     invariants sur une grille et des tirages.
 10. Le relevé saisi (collé, lu dans un fichier) : formats, refus, aller-retour.
 11. La météo du lieu (beton/maturite/meteo.py) : le PLAN des appels et la LECTURE des réponses, contre les fonctions de
     l'écran Streamlit (_plages_endpoints, meteo_journaliere, meteo_horaire, geocode_open_meteo) rejouées avec un faux
     « requests » (tests/faux_open_meteo.py, au format documenté d'Open-Meteo — le vrai service n'est jamais appelé) :
     mêmes adresses, mêmes paramètres un à un, mêmes températures (à 1e-12 près : la moyenne de la prolongation n'est
     plus celle de numpy), mêmes refus ; et des mutations du plan et de la lecture.
 12. La pureté : le moteur se charge et calcule dans un sous-processus où numpy, streamlit, matplotlib, pandas,
     requests, scipy sont REFUSÉS à l'import.
 13. Les mutations : chaque défaut plausible glissé dans une copie du moteur est rejeté par les contrôles 1 à 9.

    /usr/bin/python3.12 tests/test_maturite_moteur.py     (streamlit, matplotlib et pandas requis pour le § 11 seul)
"""
import ast
import copy
import datetime as _dt
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

import faux_open_meteo as FO  # noqa: E402  (tests/ : le faux Open-Meteo, au format du service)
from beton.maturite import entree as EN  # noqa: E402
from beton.maturite import lois as L  # noqa: E402
from beton.maturite import meteo as MT  # noqa: E402
from beton.maturite import moteur as M  # noqa: E402
from beton.maturite import releve as RL  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as f:
        return f.read()


def pres(a, b, tol=1e-9):
    """Deux nombres égaux à ``tol`` près (relatif au-delà de 1) ; None = None."""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def dec(a, b, nd):
    """``a`` arrondi à ``nd`` décimales vaut ``b`` (la valeur écrite dans la conception)."""
    return a is not None and abs(a - b) <= 0.5 * 10 ** -nd + 1e-12


# le moteur RÉEL, vu comme un jeu de modules : les contrôles 1 à 9 le reçoivent, et les mutants aussi (§ 13)
REEL = types.SimpleNamespace(L=L, EN=EN, M=M, RL=RL, MT=MT)

# ------------------------------------------------------------------ les valeurs de norme, RECOPIÉES ici (R2 : deux
# copies nommées, liées par ce test — une valeur changée dans le moteur sans l'être ici rougit)
CLASSES_NORME = {"C16/20": (16, 20), "C20/25": (20, 25), "C25/30": (25, 30), "C30/37": (30, 37), "C35/45": (35, 45),
                 "C40/50": (40, 50), "C45/55": (45, 55), "C50/60": (50, 60)}            # EN 1992-1-1 tableau 3.1
S_CIMENT = {"R": 0.20, "N": 0.25, "S": 0.38}                                             # EN 1992-1-1 §3.1.2(6)
CIMENTS_NORME = {"R": ("CEM 42,5 R", "CEM 52,5 N", "CEM 52,5 R"), "N": ("CEM 32,5 R", "CEM 42,5 N"), "S": ("CEM 32,5 N",)}
R_BETON = {"rapide": 0.50, "moyen": 0.30, "lent": 0.15}                                  # NBN EN 206, borne basse de la classe
TABLE_NBN = [(-5, 0.15), (0, 0.30), (5, 0.45), (10, 0.60), (15, 0.80), (20, 1.00)]        # NBN B 15-400, âge fictif
RAC14 = math.sqrt(14.0)


# ------------------------------------------------------------------ le calcul À LA MAIN (math seul), indépendant du moteur
def k_nbn_main(T):
    """La loi belge écrite à la main : le tableau, l'interpolation linéaire, 1 au-dessus de 20 °C, la droite de −5 à
    0 °C prolongée jusqu'à 0 à −10 °C."""
    pts = [(-10, 0.0)] + TABLE_NBN
    if T <= -10:
        return 0.0
    if T >= 20:
        return 1.0
    for (Ta, ka), (Tb, kb) in zip(pts, pts[1:]):
        if Ta <= T <= Tb:
            return ka + (kb - ka) * (T - Ta) / (Tb - Ta)


def k_ec2_main(T):
    return 0.0 if T < 0 else math.exp(-4000.0 * (1.0 / (273.0 + T) - 1.0 / 293.0))


def k_arr_main(T, Ea_kJ):
    return 0.0 if T < 0 else math.exp(-(Ea_kJ * 1000.0 / 8.314) * (1.0 / (273.0 + T) - 1.0 / 293.0))


def f_main(t_T, fck, s, base):
    """La résistance à l'âge équivalent t_T, à la main : f_cm(t) = (f_ck + 8)·exp{s·[1 − √(28/t_T)]}, plafonnée à 28 j ;
    f_ck(t) = max(0 ; f_cm(t) − 8)."""
    if t_T <= 0:
        return 0.0
    fcm_t = (fck + 8.0) * min(1.0, math.exp(s * (1.0 - math.sqrt(28.0 / t_T))))
    return fcm_t if base == "fcm" else max(0.0, fcm_t - 8.0)


def tT_main(cible, fck, s, base):
    """La forme fermée, à la main : t_T = 28/(1 − ln β/s)², β = (cible + 8)/f_cm ou cible/f_cm."""
    beta = (cible + 8.0) / (fck + 8.0) if base == "fck" else cible / (fck + 8.0)
    return 28.0 / (1.0 - math.log(beta) / s) ** 2


def te_main(t, ks, k_fin, heure):
    """L'âge équivalent à l'âge réel t, à la main et AUTREMENT que le moteur : la somme des recouvrements de [0 ; t]
    avec chaque jour civil — le jour 0 dure (24 − heure)/24, les suivants un jour."""
    f0 = (24.0 - heure) / 24.0
    total = 0.0
    for j, k in enumerate(ks):
        debut, fin = (0.0, f0) if j == 0 else (f0 + j - 1, f0 + j)
        total += k * max(0.0, min(t, fin) - debut)
    fin_connus = f0 + len(ks) - 1 if ks else 0.0
    return total + k_fin * max(0.0, t - fin_connus)


def balayage(f, cible, t_max, pas=0.001):
    """Le premier instant d'une grille de pas ``pas`` où f(t) ≥ cible (un balayage numérique fin) ; None sinon."""
    for i in range(1, int(round(t_max / pas)) + 1):
        if f(i * pas) >= cible:
            return i * pas
    return None


def coulage(m, **k):
    """Un coulage neuf modifié : « a__b=v » règle e[a][b]."""
    e = m.M.nouvel_element()
    for cle, v in k.items():
        if "__" in cle:
            a, b = cle.split("__")
            e[a][b] = v
        else:
            e[cle] = v
    return e


def objectif(m, nom="Objectif", **k):
    o = m.EN.nouvel_objectif(nom)
    o.update(k)
    return o


def releve(temps, sources=None, debut=None):
    """Un relevé collé : une ligne par jour, « — » pour un jour sans donnée ; ``debut`` (date) : lignes datées."""
    lignes = []
    for i, t in enumerate(temps):
        x = "—" if t is None else repr(float(t)).replace(".", ",")
        mot = {"m": "mesuré", "p": "prévision", "x": "prolongé"}.get((sources or "")[i:i + 1], "")
        lignes.append(";".join(v for v in ((debut + _dt.timedelta(days=i)).isoformat() if debut else "", x, mot) if v))
    return "\n".join(lignes)


def codes(r):
    return [a["code"] for a in r["avertissements"]]


AUJ_M = _dt.date(2026, 11, 7)                      # le jour des récupérations d'exemple (le moteur ne lit jamais la date)


def coulage_meteo(m, lieu, debut, fin, auj=AUJ_M, nulls=(), horaire=False, nulls_h=(), **k):
    """Un coulage en mode « météo du lieu », comme le site le laisse après « Récupérer la météo » : le plan du moteur,
    les réponses du faux Open-Meteo (ce que le navigateur rend), relues et GARDÉES, le relevé écrit."""
    srv = FO.FauxService(auj, nulls, nulls_h)
    p = m.MT.plan(lieu, debut, fin, auj)
    v = FO.recuperer(p, srv)
    rec = m.MT.enregistrement(p, v["geocodage"], v["reponses"], f"{auj.isoformat()}T09:15")
    texte = m.MT.releve(m.MT.lire_journalier(p, v["reponses"]))
    if horaire:
        ph = m.MT.plan_horaire(rec, auj)
        rec["horaire"] = m.MT.enregistrement_horaire(ph, FO.recuperer(ph, srv)["reponses"], f"{auj.isoformat()}T09:20")
    e = coulage(m, **k)
    e["temperature"].update(mode="meteo", lieu=lieu, debut=p["debut"], fin=p["fin"], releve=texte, meteo=rec)
    return e


# ====================================================================== les contrôles 1 à 9, sur un jeu de modules
def controles(m):
    """Les contrôles À LA MAIN du moteur ``m`` (ses modules L, EN, M, RL, MT) : [(nom, vrai/faux, information)]. Le
    moteur réel les passe tous (§ 1 à 9) ; un moteur mutant doit en faire rougir un au moins (§ 13)."""
    out = []

    def v(nom, cond, info=""):
        out.append((nom, bool(cond), info))

    def groupe(nom, f):
        try:
            f()
        except Exception as x:  # noqa: BLE001 — un moteur (mutant) qui plante est rejeté aussi, et c'est dit
            v(f"{nom} : aucune exception", False, f"{type(x).__name__} : {x}")
    Lm, ENm, Mm, RLm = m.L, m.EN, m.M, m.RL

    # ------------------------------------------------------------- 1. les tables et les lois
    def tables():
        v("1 · classes : C16/20 à C50/60, f_ck et f_ck,cube du tableau 3.1 de l'EN 1992-1-1 (copie indépendante)",
          {k: (float(a), float(b)) for k, (a, b) in CLASSES_NORME.items()} == dict(Lm.CLASSES) and list(Lm.CLASSES) == list(CLASSES_NORME), Lm.CLASSES)
        v("1 · f_cm = f_ck + 8 MPa ; l'âge de référence est 28 j ; « indicatif » jusqu'à 3 jours équivalents compris",
          Lm.DELTA_FCM == 8.0 and Lm.AGE_REFERENCE_J == 28.0 and Lm.SEUIL_INDICATIF_J == 3.0 and Lm.indicatif(3.0) and not Lm.indicatif(3.001))
        v("1 · f_ck/f_ck,cube : 0,78 (C35/45) à 0,83 (C25/30, C50/60) ; C30/37 : 30/37",
          dec(Lm.rapport_cube("C35/45"), 0.78, 2) and dec(Lm.rapport_cube("C25/30"), 0.83, 2) and Lm.rapport_cube("C30/37") == 30.0 / 37.0
          and min(Lm.rapport_cube(c) for c in Lm.CLASSES) == 35.0 / 45.0 and max(Lm.rapport_cube(c) for c in Lm.CLASSES) == 25.0 / 30.0)
        v("1 · ciment (EN 1992-1-1 §3.1.2(6)) : s = 0,20 (R) · 0,25 (N) · 0,38 (S), et les ciments de chaque classe",
          {k: x["s"] for k, x in Lm.CIMENTS.items()} == S_CIMENT and {k: x["ciments"] for k, x in Lm.CIMENTS.items()} == CIMENTS_NORME)
        v("1 · béton (NBN EN 206) : rapide r ≥ 0,50 · moyen 0,30 ≤ r < 0,50 · lent 0,15 ≤ r < 0,30",
          dict(Lm.BETONS) == {"rapide": (0.50, None), "moyen": (0.30, 0.50), "lent": (0.15, 0.30)})
        # s(r) = −ln r/(√14 − 1) : (3.2) écrite à 2 jours
        s_att = {r: -math.log(r) / (RAC14 - 1.0) for r in (0.50, 0.30, 0.15)}
        v("1 · s(r) = −ln r/(√14 − 1) : 0,50 → 0,2528 ; 0,30 → 0,4391 ; 0,15 → 0,6920",
          all(pres(Lm.s_de_r(r), s, 1e-13) for r, s in s_att.items()) and dec(Lm.s_de_r(0.50), 0.2528, 4) and dec(Lm.s_de_r(0.30), 0.4391, 4)
          and dec(Lm.s_de_r(0.15), 0.6920, 4), [Lm.s_de_r(r) for r in s_att])
        v("1 · r(s) = exp{s·(1 − √14)} des trois ciments : 0,578 · 0,504 · 0,353 ; s(r(s)) = s",
          all(pres(Lm.r_de_s(s), math.exp(s * (1.0 - RAC14)), 1e-13) for s in S_CIMENT.values()) and dec(Lm.r_de_s(0.20), 0.578, 3)
          and dec(Lm.r_de_s(0.25), 0.504, 3) and dec(Lm.r_de_s(0.38), 0.353, 3) and all(pres(Lm.s_de_r(Lm.r_de_s(s)), s, 1e-12) for s in (0.2, 0.25, 0.38, 0.6)),
          [Lm.r_de_s(s) for s in S_CIMENT.values()])
        d = {(mode, cle): Lm.durcissement({"mode": mode, "ciment": cle if mode == "ciment" else "N", "beton": cle if mode == "beton" else "moyen", "r": 0.42})
             for mode, cle in [("ciment", c) for c in "RNS"] + [("beton", b) for b in R_BETON] + [("r", "r")]}
        v("1 · durcissement : le ciment donne s (R, N, S) et le r qu'il suppose ; le béton donne s À LA BORNE BASSE de sa classe ; « r connu » donne s(r)",
          all(d[("ciment", c)]["s"] == S_CIMENT[c] and pres(d[("ciment", c)]["r"], math.exp(S_CIMENT[c] * (1 - RAC14)), 1e-13) for c in "RNS")
          and all(d[("beton", b)]["r"] == R_BETON[b] and pres(d[("beton", b)]["s"], -math.log(R_BETON[b]) / (RAC14 - 1), 1e-13) for b in R_BETON)
          and d[("r", "r")]["r"] == 0.42 and pres(d[("r", "r")]["s"], -math.log(0.42) / (RAC14 - 1), 1e-13)
          and all(x["libelle"] and x["source"] for x in d.values()), {k: (x["s"], x["r"]) for k, x in d.items()})
        v("1 · la classe S de l'EC2 n'est PAS « lente » : elle suppose r = 0,35, un béton « moyen » ; N suppose r = 0,50, la frontière du « rapide »",
          R_BETON["moyen"] <= Lm.r_de_s(0.38) < R_BETON["rapide"] and dec(Lm.r_de_s(0.25), 0.50, 2) and Lm.s_de_r(0.30) > 0.38 > Lm.s_de_r(0.50))
        # la loi belge
        v("1 · NBN B 15-400 : chaque point du tableau (−5 → 0,15 ; 0 → 0,30 ; 5 → 0,45 ; 10 → 0,60 ; 15 → 0,80 ; 20 → 1,00)",
          all(pres(Lm.k_nbn(T), k, 1e-13) for T, k in TABLE_NBN) and [(a, b) for a, b in Lm.K_NBN_NORME] == [(float(a), b) for a, b in TABLE_NBN],
          [Lm.k_nbn(T) for T, _ in TABLE_NBN])
        v("1 · NBN B 15-400 : interpolation LINÉAIRE (7 °C → 0,51 ; 12,5 °C → 0,70 ; 2 °C → 0,36)",
          pres(Lm.k_nbn(7.0), 0.51, 1e-12) and pres(Lm.k_nbn(12.5), 0.70, 1e-12) and pres(Lm.k_nbn(2.0), 0.36, 1e-12), [Lm.k_nbn(T) for T in (7, 12.5, 2)])
        v("1 · NBN B 15-400 : l'exemple de la norme, trois jours à 5, 7 et 5 °C = 1,41 j",
          pres(Lm.k_nbn(5.0) + Lm.k_nbn(7.0) + Lm.k_nbn(5.0), 1.41, 1e-12), Lm.k_nbn(5.0) + Lm.k_nbn(7.0) + Lm.k_nbn(5.0))
        v("1 · NBN B 15-400 : le plafond (25 °C → 1 ; 60 °C → 1) et le prolongement (−7,5 °C → 0,075 ; −10 °C → 0 ; −12 °C → 0)",
          Lm.k_nbn(25.0) == 1.0 and Lm.k_nbn(60.0) == 1.0 and pres(Lm.k_nbn(-7.5), 0.075, 1e-12) and Lm.k_nbn(-10.0) == 0.0 and Lm.k_nbn(-12.0) == 0.0,
          [Lm.k_nbn(T) for T in (25, 60, -7.5, -10, -12)])
        grille = [x / 10.0 for x in range(-200, 601)]
        v("1 · NBN B 15-400 : la loi écrite à la main (tableau, interpolation, plafond, prolongement) sur 801 températures de −20 à 60 °C ; jamais décroissante",
          all(pres(Lm.k_nbn(T), k_nbn_main(T), 1e-12) for T in grille) and all(Lm.k_nbn(a) <= Lm.k_nbn(b) + 1e-15 for a, b in zip(grille, grille[1:])))
        # (B.10)
        v("1 · EC2 (B.10) : 0 °C → 0,37 ; 5 → 0,48 ; 10 → 0,62 ; 15 → 0,79 ; 20 → 1 EXACTEMENT ; 30 → 1,57",
          dec(Lm.k_ec2(0.0), 0.37, 2) and dec(Lm.k_ec2(5.0), 0.48, 2) and dec(Lm.k_ec2(10.0), 0.62, 2) and dec(Lm.k_ec2(15.0), 0.79, 2)
          and Lm.k_ec2(20.0) == 1.0 and dec(Lm.k_ec2(30.0), 1.57, 2), [Lm.k_ec2(T) for T in (0, 5, 10, 15, 20, 30)])
        v("1 · EC2 (B.10) : exp{−4000·[1/(273 + T) − 1/293]} à la main sur 801 températures ; 0 sous 0 °C (−1 °C → 0 ; −0,1 °C → 0), 0 °C compris dans le domaine",
          all(pres(Lm.k_ec2(T), k_ec2_main(T), 1e-12) for T in grille) and Lm.k_ec2(-1.0) == 0.0 and Lm.k_ec2(-0.1) == 0.0 and Lm.k_ec2(0.0) > 0.36)
        v("1 · Arrhenius à 45 kJ/mol (R = 8,314) : 5 °C → 0,37 ; 10 → 0,52 ; 30 → 1,84 ; 0 → 0,26 ; 15 → 0,73 ; 0 sous 0 °C ; 20 °C → 1",
          dec(Lm.k_arrhenius(5.0, 45.0), 0.37, 2) and dec(Lm.k_arrhenius(10.0, 45.0), 0.52, 2) and dec(Lm.k_arrhenius(30.0, 45.0), 1.84, 2)
          and dec(Lm.k_arrhenius(0.0, 45.0), 0.26, 2) and dec(Lm.k_arrhenius(15.0, 45.0), 0.73, 2) and Lm.k_arrhenius(-1.0, 45.0) == 0.0
          and Lm.k_arrhenius(20.0, 45.0) == 1.0 and all(pres(Lm.k_arrhenius(T, Ea), k_arr_main(T, Ea), 1e-12) for T in grille for Ea in (20.0, 45.0, 60.0)),
          [Lm.k_arrhenius(T, 45.0) for T in (5, 10, 30)])
        v("1 · l'énergie d'activation que (B.10) suppose : 4000 × 8,314 = 33,256 kJ/mol — saisie en « E_a », elle redonne (B.10)",
          pres(Lm.EA_EC2_KJ, 33.256, 1e-12) and all(pres(Lm.k_arrhenius(T, Lm.EA_EC2_KJ), Lm.k_ec2(T), 1e-12) for T in grille))
        v("1 · k_T choisit la loi : nbn, ec2, perso (E_a)",
          Lm.k_T(7.0, {"mode": "nbn", "Ea_kJ": 45.0}) == Lm.k_nbn(7.0) and Lm.k_T(7.0, {"mode": "ec2", "Ea_kJ": 45.0}) == Lm.k_ec2(7.0)
          and Lm.k_T(7.0, {"mode": "perso", "Ea_kJ": 45.0}) == Lm.k_arrhenius(7.0, 45.0))
        # la loi belge face à (B.10), balayée au dixième de degré de 0 à 20 °C — MESURÉ, et non l'invariant de la conception
        # (« jamais plus rapide que (B.10) sous 20 °C ») : il est FAUX de 12,3 à 19,9 °C, où le tableau belge dépasse (B.10)
        # de 1,5 % au plus (0,80 contre 0,789 à 15 °C — le tableau de la conception le montre lui-même ; les deux lois se
        # croisent à 12,27 °C)
        balay = [x / 10.0 for x in range(0, 201)]
        rapports = [(T, Lm.k_nbn(T) / Lm.k_ec2(T)) for T in balay]
        depasse = [T for T, q in rapports if q > 1.0 + 1e-12]
        v("1 · la loi belge n'est jamais plus rapide que (B.10) de 0 à 12,2 °C (balayage au dixième de degré) ; par temps froid elle compte nettement moins "
          "(0,30 contre 0,37 à 0 °C, 0,45 contre 0,48 à 5 °C)",
          all(q <= 1.0 for T, q in rapports if T <= 12.2) and Lm.k_nbn(0.0) / Lm.k_ec2(0.0) < 0.82 and Lm.k_nbn(5.0) / Lm.k_ec2(5.0) < 0.95, depasse[:3])
        v("1 · … mais de 12,3 à 19,9 °C le tableau belge DÉPASSE (B.10), de 1,5 % au plus (0,836 contre 0,824 à 15,9 °C ; 0,80 contre 0,789 à 15 °C) : l'invariant "
          "de la conception (« jamais plus rapide sous 20 °C ») est faux, l'écart est mesuré et borné ici",
          depasse and depasse[0] == 12.3 and depasse[-1] == 19.9 and 1.0147 < max(q for _, q in rapports) < 1.0148
          and max(rapports, key=lambda x: x[1])[0] == 15.9 and dec(Lm.k_nbn(15.0) / Lm.k_ec2(15.0), 1.014, 3) and dec(Lm.k_ec2(15.9), 0.824, 3),
          (depasse[:1], depasse[-1:], max(rapports, key=lambda x: x[1])))
        v("1 · au-dessus de 20 °C la loi belge ne compte aucun bonus (30 °C : 1,00 contre 1,57 pour (B.10))", Lm.k_nbn(30.0) == 1.0 and Lm.k_ec2(30.0) > 1.5)
        # β_cc, f_cm(t), f_ck(t)
        v("1 · β_cc(3 j ; s = 0,25) = exp(0,25·(1 − √(28/3))) = 0,598 ; β_cc(28 j) = 1 ; β_cc(0) = 0 ; à 3 jours, 0,60 en ciment N contre 0,41 en béton « moyen »",
          pres(Lm.beta_cc(3.0, 0.25), math.exp(0.25 * (1.0 - math.sqrt(28.0 / 3.0))), 1e-13) and dec(Lm.beta_cc(3.0, 0.25), 0.598, 3)
          and Lm.beta_cc(28.0, 0.25) == 1.0 and Lm.beta_cc(0.0, 0.25) == 0.0 and dec(Lm.beta_cc(3.0, 0.25), 0.60, 2)
          and dec(Lm.beta_cc(3.0, Lm.s_de_r(0.30)), 0.41, 2), Lm.beta_cc(3.0, 0.25))
        b3 = math.exp(0.25 * (1.0 - math.sqrt(28.0 / 3.0)))
        v("1 · C30/37, ciment N, 3 jours équivalents : f_cm(3 j) = 38·β_cc = 22,73 MPa ; f_ck(3 j) = f_cm(3 j) − 8 = 14,73 MPa",
          pres(Lm.fcm_t(3.0, 30.0, 0.25), 38.0 * b3, 1e-12) and pres(Lm.fck_t(3.0, 30.0, 0.25), 38.0 * b3 - 8.0, 1e-12)
          and dec(Lm.fcm_t(3.0, 30.0, 0.25), 22.73, 2) and dec(Lm.fck_t(3.0, 30.0, 0.25), 14.73, 2), (Lm.fcm_t(3.0, 30.0, 0.25), Lm.fck_t(3.0, 30.0, 0.25)))
        v("1 · f_ck(t) n'est JAMAIS négative : à 0,2 jour équivalent (ciment N), f_cm(t) = 2,53 MPa < 8 → f_ck(t) = 0 (et non −5,47)",
          dec(Lm.fcm_t(0.2, 30.0, 0.25), 2.53, 2) and Lm.fck_t(0.2, 30.0, 0.25) == 0.0 and Lm.fck_t(0.0, 30.0, 0.25) == 0.0
          and all(Lm.fck_t(x / 100.0, fck, s) >= 0.0 for x in range(0, 400) for fck in (16.0, 50.0) for s in (0.2, 0.69)))
        v("1 · au-delà de 28 jours équivalents : f_ck(t) = f_ck (§3.1.2(5)) et f_cm(t) = f_cm — aucun gain compté (β_cc borné à 1, alors que (3.2) donne 1,03 à 40 j)",
          Lm.fck_t(28.0, 30.0, 0.25) == 30.0 and Lm.fck_t(60.0, 30.0, 0.25) == 30.0 and Lm.fcm_t(60.0, 30.0, 0.25) == 38.0 and Lm.beta_retenu(40.0, 0.25) == 1.0
          and dec(Lm.beta_cc(40.0, 0.25), 1.04, 2) and Lm.beta_retenu(10.0, 0.25) == Lm.beta_cc(10.0, 0.25), Lm.beta_cc(40.0, 0.25))
        v("1 · f_cm(t) et f_ck(t) sont celles écrites à la main, pour chaque classe, quatre s, 60 âges équivalents",
          all(pres(Lm.resistance(x / 2.0, base, fck, s), f_main(x / 2.0, fck, s, base), 1e-12) for x in range(1, 61) for (fck, _) in CLASSES_NORME.values()
              for s in (0.20, 0.25, 0.38, 0.6920) for base in ("fck", "fcm")))
        # la forme fermée : t_T = 28/(1 − ln β_req/s)², face à la formule à la main et à un BALAYAGE numérique au millième de jour
        cas = [(21.0, 30.0, Lm.s_de_r(0.30), "fck"), (21.0, 30.0, 0.25, "fck"), (15.0, 30.0, 0.25, "fck"), (5.0, 30.0, Lm.s_de_r(0.30), "fcm"),
               (10.0, 25.0, 0.38, "fcm"), (35.0, 50.0, 0.20, "fck"), (8.0, 16.0, Lm.s_de_r(0.15), "fck"), (30.0, 40.0, 0.25, "fcm"), (1.0, 20.0, 0.25, "fck")]
        ecarts_ff = []
        for cible, fck, s, base in cas:
            t_ff = Lm.t_T_requis(cible, base, fck, s)
            t_bal = balayage(lambda x: f_main(x, fck, s, base), cible, 28.0)
            if not (pres(t_ff, tT_main(cible, fck, s, base), 1e-12) and t_bal is not None and abs(t_ff - t_bal) <= 0.002 and t_bal >= t_ff - 1e-9):
                ecarts_ff.append((cible, fck, base, t_ff, t_bal))
        v(f"1 · l'âge équivalent d'une cible : la forme fermée 28/(1 − ln β/s)² = la formule à la main, et = un balayage numérique au pas de 0,001 j à "
          f"0,002 j près ({len(cas)} cas, les deux bases)", not ecarts_ff, ecarts_ff[:2])
        v("1 · β requis : (cible + 8)/f_cm face à la caractéristique, cible/f_cm face à la moyenne (C30/37, 21 MPa : 29/38 et 21/38)",
          Lm.beta_requis(21.0, "fck", 30.0) == 29.0 / 38.0 and Lm.beta_requis(21.0, "fcm", 30.0) == 21.0 / 38.0)
        v("1 · une cible au-dessus de la classe n'est PAS atteignable : f_ck + 0,1 face à la caractéristique, f_cm + 0,1 face à la moyenne ; la classe exacte l'est à 28 j",
          Lm.t_T_requis(30.1, "fck", 30.0, 0.25) is None and Lm.t_T_requis(38.1, "fcm", 30.0, 0.25) is None
          and pres(Lm.t_T_requis(30.0, "fck", 30.0, 0.25), 28.0, 1e-9) and pres(Lm.t_T_requis(38.0, "fcm", 30.0, 0.25), 28.0, 1e-9)
          and Lm.t_T_requis(35.0, "fcm", 30.0, 0.25) < 28.0)
        v("1 · la résistance moyenne atteint une cible AVANT la caractéristique (même cible, même béton) — 9 classes × cibles",
          all(Lm.t_T_requis(c, "fcm", fck, s) < Lm.t_T_requis(c, "fck", fck, s) for (fck, _) in CLASSES_NORME.values() for s in (0.2, 0.44, 0.69)
              for c in (2.0, 5.0, 10.0, fck * 0.7, fck)))
    groupe("1 · tables et lois", tables)

    # ------------------------------------------------------------- 2. le temps
    T6 = [12.0, 8.0, 5.0, 7.0, 10.0, 15.0]                       # un relevé de six jours (°C), le jour du bétonnage d'abord
    K6 = [0.68, 0.54, 0.45, 0.51, 0.60, 0.80]                    # k de la loi belge, à la main : 0,60 + 0,04·2 ; 0,45 + 0,03·3 ; … ; 0,80

    def temps():
        v("2 · la fraction du jour du bétonnage : (24 − heure)/24 — 16 h laisse un tiers de jour, midi une moitié, 0 h le jour entier, 23 h une heure",
          Lm.fraction_premier_jour(16) == 8.0 / 24.0 and Lm.fraction_premier_jour(12) == 0.5 and Lm.fraction_premier_jour(0) == 1.0
          and Lm.fraction_premier_jour(23) == 1.0 / 24.0)
        v("2 · les six jours à la main : k = 0,68 · 0,54 · 0,45 · 0,51 · 0,60 · 0,80 (loi belge)", all(pres(Lm.k_nbn(T), k, 1e-12) for T, k in zip(T6, K6)))
        r = Mm.calculer(coulage(m, heure=16, temperature__mode="releve", temperature__releve=releve(T6)))
        f0 = 8.0 / 24.0
        fins = [f0 * 0.68]                                        # t_T cumulé en fin de jour, à la main
        for k in K6[1:]:
            fins.append(fins[-1] + k)
        v("2 · profil de 6 jours, fin de bétonnage à 16 h : t_T en fin de jour = 0,2267 · 0,7667 · 1,2167 · 1,7267 · 2,3267 · 3,1267 j (le jour 0 ne compte qu'un tiers)",
          all(pres(r["profil"][j]["t_T"], fins[j], 1e-12) for j in range(6)) and dec(fins[0], 0.2267, 4) and dec(fins[5], 3.1267, 4)
          and [x["source"] for x in r["profil"][:6]] == ["mesure"] * 6 and all(pres(r["profil"][j]["fin"], f0 + j, 1e-12) for j in range(6))
          and r["profil"][0]["debut"] == 0.0 and pres(r["profil"][1]["debut"], f0, 1e-12), [x["t_T"] for x in r["profil"][:6]])
        prol = (8.0 + 5.0 + 7.0 + 10.0 + 15.0) / 5.0              # la moyenne des cinq derniers jours : 9,0 °C → k = 0,57
        v("2 · au-delà du relevé : PROLONGÉ à la moyenne des cinq derniers jours (9,0 °C, k = 0,57), source « prolonge » ; la variante prudente l'abaisse de 3 °C (6 °C, k = 0,48)",
          pres(r["temperature"]["T_fin"], prol, 1e-12) and r["temperature"]["source_fin"] == "prolonge" and pres(r["profil"][6]["k"], 0.57, 1e-12)
          and r["profil"][6]["source"] == "prolonge" and pres(r["profil"][6]["T_prudent"], 6.0, 1e-12) and pres(r["profil"][6]["k_prudent"], 0.48, 1e-12)
          and pres(r["profil"][6]["t_T"], fins[5] + 0.57, 1e-12) and r["profil"][6]["T_brut"] is None, r["profil"][6])
        ks = [x["k"] for x in r["profil"][:6]]
        pts = [0.0, 0.2, f0, 1.0, 2.0, f0 + 5, 7.0, 20.0]
        att = [0.0, 0.136, f0 * 0.68, f0 * 0.68 + 0.54 * (1 - f0), f0 * 0.68 + 0.54 + 0.45 * (2 - f0 - 1), fins[5], fins[5] + 0.57 * (7 - f0 - 5),
               fins[5] + 0.57 * (20 - f0 - 5)]
        v("2 · l'âge équivalent en cours de journée : t_T(0,2 j) = 0,136 ; t_T(2 j) = 1,0667 ; t_T(7 j) = 4,0767 (à la main, et par recouvrement des jours)",
          all(pres(Lm.age_equivalent(t, ks, 0.57, f0), a, 1e-12) and pres(te_main(t, K6, 0.57, 16), a, 1e-12) for t, a in zip(pts, att))
          and dec(att[4], 1.0667, 4) and dec(att[6], 4.0767, 4), [Lm.age_equivalent(t, ks, 0.57, f0) for t in pts])
        v("2 · l'inverse est EXACT : age_reel(age_equivalent(t)) = t sur 400 âges (0,05 à 20 j), à 1e-11 près",
          all(pres(Lm.age_reel(Lm.age_equivalent(i / 20.0, ks, 0.57, f0), ks, 0.57, f0), i / 20.0, 1e-11) for i in range(1, 401)))
        gel = [0.5, 0.0, 0.0, 0.7]                                # deux jours sans durcissement : le palier se franchit au jour d'après
        v("2 · l'inverse rend le PREMIER âge où t_T est atteint, par-dessus les jours qui ne comptent rien ; None si rien ne compte plus (k = 0 jusqu'au bout)",
          pres(Lm.age_reel(0.5 * 0.5, gel, 0.0, 0.5), 0.5, 1e-12) and pres(Lm.age_reel(0.25 + 0.07, gel, 0.0, 0.5), 2.5 + 0.1, 1e-12)
          and Lm.age_reel(0.25 + 0.7 + 0.01, gel, 0.0, 0.5) is None and pres(Lm.age_reel(0.25 + 0.7 + 0.01, gel, 0.1, 0.5), 3.5 + 0.1, 1e-12)
          and Lm.age_reel(0.0, gel, 0.0, 0.5) == 0.0 and Lm.age_reel(1.0, [], 0.0, 0.5) is None and pres(Lm.age_reel(1.0, [], 0.4, 0.5), 2.5, 1e-12))
        r0 = Mm.calculer(coulage(m, heure=0, temperature__mode="releve", temperature__releve=releve(T6)))
        v("2 · l'heure du coulage compte : à 0 h le jour 0 est entier (t_T = 0,68 en fin de jour), à 16 h il vaut un tiers — 0,4533 jour équivalent d'écart, gardé ensuite",
          pres(r0["profil"][0]["t_T"], 0.68, 1e-12) and pres(r0["profil"][5]["t_T"] - r["profil"][5]["t_T"], 0.68 * (2.0 / 3.0), 1e-12)
          and dec(0.68 * 2.0 / 3.0, 0.4533, 4) and r0["temps"]["f0"] == 1.0 and r["temps"]["f0"] == f0)
        # le relevé posé sur le CALENDRIER du coulage
        d = _dt.date(2026, 11, 2)
        meme = Mm.calculer(coulage(m, date="2026-11-02", heure=16, temperature__mode="releve", temperature__releve=releve(T6, debut=d)))
        v("2 · relevé daté qui commence le jour du bétonnage : le même calcul que sans dates, chaque jour à sa date",
          [x["t_T"] for x in meme["profil"][:7]] == [x["t_T"] for x in r["profil"][:7]] and meme["profil"][0]["date"] == "2026-11-02"
          and meme["profil"][6]["date"] == "2026-11-08" and meme["temps"]["date_origine"] == "saisie" and not codes(meme) == ["x"])
        avant = Mm.calculer(coulage(m, date="2026-11-04", heure=16, temperature__mode="releve", temperature__releve=releve(T6, debut=d)))
        v("2 · relevé qui commence DEUX JOURS AVANT le bétonnage : ces deux jours ne comptent pas (le jour 0 est le 04/11, à 5 °C), et c'est dit",
          avant["synthese"]["n_avant"] == 2 and avant["profil"][0]["T"] == 5.0 and avant["profil"][0]["date"] == "2026-11-04"
          and pres(avant["profil"][0]["t_T"], f0 * 0.45, 1e-12) and pres(avant["profil"][3]["t_T"], f0 * 0.45 + 0.51 + 0.60 + 0.80, 1e-12)
          and avant["synthese"]["n_jours"] == 4 and "releve_avant" in codes(avant), (avant["synthese"], codes(avant)))
        apres = Mm.calculer(coulage(m, date="2026-11-01", heure=16, temperature__mode="releve", temperature__releve=releve(T6, debut=d)))
        v("2 · relevé qui commence UN JOUR APRÈS le bétonnage : ce jour, sans température, ne compte AUCUN durcissement (k = 0), en grave — jamais une température inventée",
          apres["synthese"]["n_manquants"] == 1 and apres["profil"][0]["source"] == "sans_donnee" and apres["profil"][0]["T"] is None
          and apres["profil"][0]["k"] == 0.0 and apres["profil"][0]["t_T"] == 0.0 and pres(apres["profil"][1]["t_T"], 0.68, 1e-12)
          and any(a["code"] == "releve_tardif" and a["niveau"] == "grave" for a in apres["avertissements"]), (apres["profil"][:2], codes(apres)))
        sans = Mm.calculer(coulage(m, heure=16, temperature__mode="releve", temperature__releve=releve(T6, debut=d)))
        v("2 · sans date de bétonnage, la première ligne d'un relevé DATÉ la donne (et l'origine le dit) ; sans aucune date : des âges, pas d'instants",
          sans["temps"]["date"] == "2026-11-02" and sans["temps"]["date_origine"] == "releve" and sans["objectifs"][0]["instant"] is not None
          and r["temps"]["date"] is None and r["temps"]["date_origine"] is None and r["objectifs"][0]["instant"] is None
          and r["objectifs"][0]["t"] is not None and all(x["date"] is None for x in r["profil"]))
        trou = Mm.calculer(coulage(m, heure=16, temperature__mode="releve", temperature__releve=releve([12.0, None, 5.0, 7.0, 10.0, 15.0])))
        v("2 · un jour SANS DONNÉE (« — ») ne compte aucun durcissement (k = 0, du côté sûr), il n'est pas remplacé par une moyenne ; c'est dit",
          trou["profil"][1]["source"] == "sans_donnee" and trou["profil"][1]["k"] == 0.0 and trou["profil"][1]["k_prudent"] == 0.0
          and pres(trou["profil"][5]["t_T"], fins[5] - 0.54, 1e-12) and "sans_donnee" in codes(trou) and trou["synthese"]["n"]["sans_donnee"] == 1,
          (trou["profil"][1], codes(trou)))
        v("2 · la prolongation : la moyenne des cinq DERNIERS jours qui portent une température (un jour sans donnée parmi eux n'y entre pas)",
          pres(Lm.temperature_prolongee([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 5.0, 1e-12)
          and pres(Lm.temperature_prolongee([1.0, 2.0, 3.0, 4.0, float("nan"), 6.0, 7.0]), (3.0 + 4.0 + 6.0 + 7.0) / 4.0, 1e-12)
          and pres(Lm.temperature_prolongee([4.0, 8.0]), 6.0, 1e-12)
          and pres(Lm.temperature_prolongee([9.0] + [float("nan")] * 5), 9.0, 1e-12))
        cst = Mm.calculer(coulage(m, temperature__T=8.0))
        v("2 · température constante : chaque jour à sa source « constante », k = 0,54 ; la variante prudente les abaisse TOUS de la marge (5 °C, k = 0,45)",
          all(x["source"] == "constante" and pres(x["k"], 0.54, 1e-12) and pres(x["k_prudent"], 0.45, 1e-12) and x["T"] == 8.0 for x in cst["profil"])
          and pres(cst["profil"][0]["t_T"], 0.5 * 0.54, 1e-12) and cst["temperature"]["source"] == "constante" and cst["synthese"]["n_jours"] == 0)
        vide = Mm.calculer(coulage(m, temperature__mode="releve", temperature__T=11.0))
        v("2 · mode relevé sans relevé : le calcul se fait à la température constante SAISIE (11 °C, jamais un 20 °C inventé), et c'est dit",
          vide["temperature"]["source"] == "constante" and vide["profil"][0]["T"] == 11.0 and "releve_vide" in codes(vide), codes(vide))
        mix = Mm.calculer(coulage(m, heure=16, temperature__mode="releve", temperature__releve=releve(T6, "mmmppx")))
        v("2 · la variante prudente abaisse de la marge les jours PRÉVUS et PROLONGÉS, jamais les jours mesurés",
          [pres(x["T_prudent"], x["T"] - (3.0 if x["source"] != "mesure" else 0.0), 1e-12) for x in mix["profil"][:8]] == [True] * 8
          and [x["source"] for x in mix["profil"][:7]] == ["mesure"] * 3 + ["prevision"] * 2 + ["prolonge"] * 2
          and mix["synthese"]["n"] == {"mesure": 3, "prevision": 2, "prolonge": 1}, [(x["source"], x["T"], x["T_prudent"]) for x in mix["profil"][:7]])
        d = _dt.date(2026, 11, 2)
        tard = Mm.calculer(coulage(m, date="2026-11-20", heure=16, temperature__mode="releve", temperature__releve=releve(T6, debut=d)))
        v("2 · un relevé tout entier AVANT le bétonnage : aucun de ses jours ne compte, tout est prolongé à la moyenne de ses cinq derniers jours (9 °C), et c'est dit "
          "(« prolongé », six jours non comptés)",
          tard["synthese"]["n_avant"] == 6 and tard["synthese"]["n_jours"] == 0 and tard["profil"][0]["source"] == "prolonge" and tard["profil"][0]["T"] == 9.0
          and "prolonge" in codes(tard) and "releve_avant" in codes(tard) and tard["objectifs"][0]["au_dela"] and tard["synthese"]["fin_connus"] == 0.0, codes(tard))
        met = Mm.calculer(coulage(m, heure=16, temperature__mode="releve", temperature__releve=releve([15.0] * 6 + [9.0] * 30, "m" * 6 + "x" * 30)))
        v("2 · un objectif atteint pendant des jours « prolongé » du relevé lui-même (une météo au-delà de son horizon) repose aussi sur des températures prolongées : dit",
          met["objectifs"][0]["source"] == "prolonge" and met["objectifs"][0]["au_dela"] and "prolonge" in codes(met)
          and not Mm.calculer(coulage(m, heure=16, temperature__mode="releve", temperature__releve=releve([15.0] * 36)))["objectifs"][0]["au_dela"])
    groupe("2 · le temps", temps)

    # ------------------------------------------------------------- 3. un exemple complet, déroulé à la main
    def exemple():
        """C30/37, béton à durcissement moyen, fin du bétonnage le lundi 02/11/2026 à 16 h, relevé de six jours MESURÉS
        (12, 8, 5, 7, 10, 15 °C), loi belge, marge 3 °C. Deux objectifs : « Hors gel » 5 MPa (moyenne) et « Décoffrage »
        70 % de f_ck prévu à J + 14."""
        e = coulage(m, date="2026-11-02", heure=16, temperature__mode="releve", temperature__releve=releve(T6))
        e["objectifs"] = [objectif(m, "Hors gel", mode="mpa", mpa=5.0, base="fcm"), objectif(m, "Décoffrage", mode="pct", pct=70.0, age=14.0)]
        r = Mm.calculer(e)
        s = -math.log(0.30) / (RAC14 - 1.0)                       # 0,43914
        fcm = 30.0 + 8.0
        f0 = 8.0 / 24.0
        cum = [f0 * 0.68, f0 * 0.68 + 0.54, f0 * 0.68 + 0.54 + 0.45]
        fin6 = f0 * 0.68 + 0.54 + 0.45 + 0.51 + 0.60 + 0.80       # 3,12667 j équivalents à la fin du relevé (âge réel 5,3333 j)
        # « Hors gel » : 5 MPa de résistance moyenne
        tT1 = 28.0 / (1.0 - math.log(5.0 / fcm) / s) ** 2          # 0,88700 j
        t1 = (f0 + 1.0) + (tT1 - cum[1]) / 0.45                    # atteint le 3ᵉ jour (k = 0,45) : 1,6007 j
        o1, o2 = r["objectifs"]
        v("3 · exemple : s = 0,43914 (béton moyen), f_cm = 38 MPa", pres(r["materiau"]["s"], s, 1e-13) and r["materiau"]["fcm"] == 38.0 and dec(s, 0.43914, 5))
        v("3 · exemple · « Hors gel » 5 MPa (moyenne) : t_T requis = 28/(1 − ln(5/38)/s)² = 0,8870 j ; atteint à t = 1,6007 j, le 3ᵉ jour (k = 0,45)",
          pres(o1["t_T"], tT1, 1e-12) and pres(o1["t"], t1, 1e-12) and dec(tT1, 0.8870, 4) and dec(t1, 1.6007, 4) and o1["base"] == "fcm" and o1["cible"] == 5.0,
          (o1["t_T"], o1["t"]))
        v("3 · exemple · « Hors gel » : 02/11 16 h + 1,6007 j = 04/11 à 6 h 25 → annoncé « 04/11 vers 7 h » (l'heure SUPÉRIEURE, jamais le jour tronqué) ; indicatif "
          "(t_T ≤ 3 j) ; la variante prudente est la même (tous les jours jusque-là sont mesurés)",
          o1["instant"] == "2026-11-04T07:00" and o1["indicatif"] and pres(o1["prudent"]["t"], t1, 1e-12) and o1["prudent"]["instant"] == o1["instant"]
          and o1["etat"] == "date" and not o1["au_dela"], (o1["instant"], o1["prudent"]))
        # « Décoffrage » : 70 % de f_ck = 21 MPa de résistance caractéristique, prévu à J + 14
        tT2 = 28.0 / (1.0 - math.log((21.0 + 8.0) / fcm) / s) ** 2  # 10,72865 j
        t2 = (f0 + 5.0) + (tT2 - fin6) / 0.57                      # au-delà du relevé, à 9 °C (k = 0,57) : 18,6701 j
        t2p = (f0 + 5.0) + (tT2 - fin6) / 0.48                     # prudent : 6 °C (k = 0,48) : 21,1708 j
        v("3 · exemple · « Décoffrage » 70 % de f_ck = 21 MPa (caractéristique) : t_T requis = 10,7286 j ; atteint à 18,67 j (au-delà du relevé, prolongé à 9 °C) ; "
          "prudent 21,17 j (6 °C)",
          o2["cible"] == 21.0 and o2["base"] == "fck" and pres(o2["t_T"], tT2, 1e-12) and pres(o2["t"], t2, 1e-12) and pres(o2["prudent"]["t"], t2p, 1e-12)
          and dec(tT2, 10.7286, 4) and dec(t2, 18.67, 2) and dec(t2p, 21.17, 2) and o2["au_dela"] and not o2["indicatif"], (o2["t"], o2["prudent"]))
        te14 = fin6 + 0.57 * (14.0 - f0 - 5.0)                     # l'âge équivalent à 14 jours réels : 8,0667 j
        b14 = math.exp(s * (1.0 - math.sqrt(28.0 / te14)))
        f14 = fcm * b14 - 8.0                                      # f_ck(14 j) = 18,01 MPa
        te14p = fin6 + 0.48 * (14.0 - f0 - 5.0)
        f14p = fcm * math.exp(s * (1.0 - math.sqrt(28.0 / te14p))) - 8.0
        v("3 · exemple · à J + 14 : t_T = 8,0667 j, f_ck(14 j) = 38·β_cc − 8 = 18,01 MPa pour 21 demandés → η = 117 %, non conforme ; prudent 16,93 MPa",
          pres(o2["t_T_prevu"], te14, 1e-12) and pres(o2["f_prevu"], f14, 1e-12) and pres(o2["taux"], 21.0 / f14, 1e-12) and o2["etat"] == "nok"
          and pres(o2["f_prevu_prudent"], f14p, 1e-12) and o2["prudent_ok"] is False and dec(te14, 8.0667, 4) and dec(f14, 18.01, 2)
          and dec(21.0 / f14, 1.17, 2) and dec(f14p, 16.93, 2) and o2["instant_prevu"] == "2026-11-16T16:00",
          (o2["t_T_prevu"], o2["f_prevu"], o2["taux"], o2["f_prevu_prudent"]))
        v("3 · exemple · le résumé du coulage : son objectif daté, 117 %, non conforme ; l'avertissement « prolongé » nomme « Décoffrage » ; pas de gel ; pas de "
          "rappel « temps froid » (5 °C n'est pas SOUS 5 °C)",
          r["resume"]["date"] and pres(r["resume"]["taux"], 21.0 / f14, 1e-12) and r["resume"]["etat"] == "nok" and r["resume"]["objectif"] == 1
          and any(a["code"] == "prolonge" and "Décoffrage" in a["message"] and "Hors gel" not in a["message"] and "9,0 °C" in a["message"] for a in r["avertissements"])
          and o2["source"] == "prolonge" and o1["source"] == "mesure"
          and not r["gel"]["alerte"] and r["gel"]["premier_jour"] is None and not r["temps_froid"]["rappel"], (r["resume"], codes(r)))
        v("3 · exemple · la synthèse : six jours mesurés, T moy 9,5 °C, de 5 à 15 °C ; âge équivalent à 28 j réels = 16,05 j ; t_T = 3 j à l'âge réel 5,17 j",
          r["synthese"]["n"] == {"mesure": 6} and pres(r["synthese"]["T_moy"], 9.5, 1e-12) and r["synthese"]["T_min"] == 5.0 and r["synthese"]["T_max"] == 15.0
          and pres(r["synthese"]["t_T_28"], fin6 + 0.57 * (28.0 - f0 - 5.0), 1e-12) and dec(r["synthese"]["t_T_28"], 16.05, 2)
          and pres(r["synthese"]["t_3"], (f0 + 4.0) + (3.0 - (fin6 - 0.80)) / 0.80, 1e-12) and dec(r["synthese"]["t_3"], 5.17, 2)
          and pres(r["synthese"]["fin_connus"], f0 + 5.0, 1e-12) and r["synthese"]["date_fin_connus"] == "2026-11-07", r["synthese"])
        # … et le même objectif par un balayage numérique de l'âge RÉEL au millième de jour (l'intégration à la main, par recouvrement)
        bal = balayage(lambda t: f_main(te_main(t, K6, 0.57, 16), 30.0, s, "fck"), 21.0, 40.0)
        bal1 = balayage(lambda t: f_main(te_main(t, K6, 0.57, 16), 30.0, s, "fcm"), 5.0, 40.0)
        v("3 · exemple · l'âge RÉEL de chaque objectif par un balayage numérique au pas de 0,001 j (intégration à la main) : égal à 0,002 j près",
          abs(bal - o2["t"]) <= 0.002 and abs(bal1 - o1["t"]) <= 0.002, (bal, o2["t"], bal1, o1["t"]))
    groupe("3 · exemple complet", exemple)

    # ------------------------------------------------------------- 4. les objectifs
    def objectifs():
        r = Mm.calculer(Mm.nouvel_element())
        o = r["objectifs"][0]
        tT = tT_main(21.0, 30.0, -math.log(0.30) / (RAC14 - 1.0), "fck")
        v("4 · le défaut d'ouverture : C30/37, béton « moyen », 20 °C, 70 % de f_ck = 21 MPa → J + 10,73 (t = t_T : k = 1) ; indicatif non ; prudent (17 °C, k = 0,88) J + 12,19",
          pres(o["t"], tT, 1e-12) and dec(o["t"], 10.73, 2) and pres(o["t_T"], tT, 1e-12) and pres(o["prudent"]["t"], tT / 0.88, 1e-12)
          and dec(o["prudent"]["t"], 12.19, 2) and o["etat"] == "date" and not o["indicatif"] and o["nom"] == "Décoffrage" and codes(r) == [],
          (o["t"], o["prudent"], codes(r)))
        rn = Mm.calculer(coulage(m, durcissement__mode="ciment", durcissement__ciment="N"))
        v("4 · … et 6,46 j si l'on choisit le ciment N (s = 0,25) : quatre jours plus tôt — la lecture par le ciment est optimiste pour un béton « moyen »",
          pres(rn["objectifs"][0]["t"], tT_main(21.0, 30.0, 0.25, "fck"), 1e-12) and dec(rn["objectifs"][0]["t"], 6.46, 2)
          and 4.2 < o["t"] - rn["objectifs"][0]["t"] < 4.3, rn["objectifs"][0]["t"])
        # le défaut n° 1 de l'ancien écran : 7 j à 2 °C puis 7 j à 20 °C, C30/37 ciment N, loi (B.10), 15 MPa caractéristique
        e1 = coulage(m, heure=0, durcissement__mode="ciment", durcissement__ciment="N", loi__mode="ec2", temperature__mode="releve",
                     temperature__releve=releve([2.0] * 7 + [20.0] * 7))
        e1["objectifs"] = [objectif(m, "Seuil", mode="mpa", mpa=15.0, base="fck")]
        o1 = Mm.calculer(e1)["objectifs"][0]
        tT1 = 28.0 / (1.0 - math.log(23.0 / 38.0) / 0.25) ** 2      # 3,0938 j équivalents
        k2 = math.exp(-4000.0 * (1.0 / 275.0 - 1.0 / 293.0))        # 0,4092 à 2 °C
        t1 = 7.0 + (tT1 - 7.0 * k2) / 1.0                          # sept jours à 2 °C n'y suffisent pas (2,864 j) : atteint le 8ᵉ jour, à 20 °C
        k11 = math.exp(-4000.0 * (1.0 / 284.0 - 1.0 / 293.0))       # ce que faisait l'ancien écran : la température MOYENNE (11 °C)
        v("4 · le défaut n° 1 de l'ancien écran (7 j à 2 °C puis 7 j à 20 °C, ciment N, (B.10), 15 MPa caractéristique) : atteint à 7,23 j — le 8ᵉ jour —, "
          "PAS à 4,8 j (ce que donnait la température moyenne de 11 °C : trois jours trop tôt)",
          pres(o1["t"], t1, 1e-12) and dec(o1["t"], 7.23, 2) and 7.0 < o1["t"] < 8.0 and dec(tT1 / k11, 4.8, 1) and o1["t"] - tT1 / k11 > 2.4
          and dec(tT1, 3.0938, 4) and dec(7.0 * k2, 2.864, 3), (o1["t"], tT1 / k11))
        e1b = copy.deepcopy(e1)
        e1b["heure"] = 12
        v("4 · … et à midi le premier jour ne compte qu'une moitié : 6,5 j à 2 °C, atteint à 6,93 j (toujours le 8ᵉ jour civil)",
          pres(Mm.calculer(e1b)["objectifs"][0]["t"], 6.5 + (tT1 - 6.5 * k2), 1e-12) and dec(6.5 + (tT1 - 6.5 * k2), 6.93, 2))
        # les deux bases, la cible en % qui suit la classe
        e2 = Mm.nouvel_element()
        e2["objectifs"] = [objectif(m, "A", mode="mpa", mpa=12.0, base="fcm"), objectif(m, "B", mode="mpa", mpa=12.0, base="fck"),
                           objectif(m, "C", mode="pct", pct=40.0, base="fcm")]
        r2 = Mm.calculer(e2)
        a, b, c = r2["objectifs"]
        s = r2["materiau"]["s"]
        v("4 · 12 MPa comparés à la MOYENNE (β = 12/38) sont atteints avant 12 MPa comparés à la CARACTÉRISTIQUE (β = 20/38) : 2,13 j contre 4,62 j",
          pres(a["t"], tT_main(12.0, 30.0, s, "fcm"), 1e-12) and pres(b["t"], tT_main(12.0, 30.0, s, "fck"), 1e-12) and a["t"] < b["t"]
          and dec(a["t"], 2.13, 2) and dec(b["t"], 4.62, 2), (a["t"], b["t"]))
        v("4 · une cible en % est TOUJOURS comparée à la caractéristique (sa base est forcée à « fck » dès la normalisation) : 40 % de 30 = 12 MPa → le même âge que B",
          c["base"] == "fck" and c["cible"] == 12.0 and pres(c["t"], b["t"], 1e-12) and r2["entree"]["objectifs"][2]["base"] == "fck")
        e3 = copy.deepcopy(e2)
        e3["classe"] = "C40/50"
        r3 = Mm.calculer(e3)
        v("4 · les objectifs en % suivent la classe (40 % de C40/50 = 16 MPa) ; en MPa ils sont fixes (12 MPa, atteints plus tôt avec une classe plus forte)",
          r3["objectifs"][2]["cible"] == 16.0 and r3["objectifs"][0]["cible"] == 12.0 and r3["objectifs"][1]["t"] < b["t"])
        e4 = Mm.nouvel_element()
        e4["objectifs"] = [objectif(m, "Trop", mode="mpa", mpa=31.0, base="fck"), objectif(m, "Trop moyen", mode="mpa", mpa=38.5, base="fcm"),
                           objectif(m, "Juste", mode="pct", pct=100.0), objectif(m, "Daté trop", mode="mpa", mpa=31.0, base="fck", age=20.0)]
        r4 = Mm.calculer(e4)
        v("4 · une cible au-dessus de la classe (31 MPa caractéristique, 38,5 MPa moyenne pour un C30/37) : NON ATTEIGNABLE, en grave, sans âge ; 100 % de f_ck : 28 j",
          [x["etat"] for x in r4["objectifs"]] == ["non_atteignable", "non_atteignable", "date", "nok"] and r4["objectifs"][0]["t"] is None
          and not r4["objectifs"][0]["atteignable"] and pres(r4["objectifs"][2]["t"], 28.0, 1e-9)
          and sum(1 for x in r4["avertissements"] if x["code"] == "non_atteignable" and x["niveau"] == "grave") == 3, [x["etat"] for x in r4["objectifs"]])
        gele = Mm.calculer(coulage(m, temperature__T=-5.0, loi__mode="ec2"))
        v("4 · −5 °C constants sous (B.10) : aucun durcissement compté, l'objectif n'est JAMAIS atteint (état « jamais », en grave) ; sous la loi belge il l'est (k = 0,15 : 71,5 j)",
          gele["objectifs"][0]["etat"] == "jamais" and gele["objectifs"][0]["t"] is None and gele["objectifs"][0]["atteignable"] and "jamais" in codes(gele)
          and "loi_sous_zero" in codes(gele) and pres(Mm.calculer(coulage(m, temperature__T=-5.0))["objectifs"][0]["t"], tT / 0.15, 1e-9)
          and dec(tT / 0.15, 71.5, 1), codes(gele))
        # l'opération datée : conforme, attention (la variante prudente ne passe pas), non conforme
        etats = {}
        for age in (11.0, 13.0, 9.0):
            ed = Mm.nouvel_element()
            ed["objectifs"][0]["age"] = age
            od = Mm.calculer(ed)["objectifs"][0]
            f = f_main(age, 30.0, s, "fck")
            fp = f_main(age * 0.88, 30.0, s, "fck")
            etats[age] = (od["etat"], pres(od["f_prevu"], f, 1e-12) and pres(od["taux"], 21.0 / f, 1e-12) and pres(od["f_prevu_prudent"], fp, 1e-12)
                          and od["prudent_ok"] == (fp >= 21.0))
        v("4 · opération datée (70 % de f_ck à 20 °C, atteint à 10,73 j ; prudent 12,19 j) : prévue à J + 13 → conforme ; à J + 11 → conforme, ATTENTION (la "
          "variante prudente ne passe pas) ; à J + 9 → non conforme ; η = cible / f(t prévu), à la main",
          etats == {13.0: ("ok", True), 11.0: ("attention", True), 9.0: ("nok", True)}, etats)
        # les instants
        ei = coulage(m, date="2026-11-02", heure=16)
        oi = Mm.calculer(ei)["objectifs"][0]
        v("4 · l'instant à l'heure près : 02/11 16 h + 10,7286 j = 13/11 à 9 h 29 → « 13/11 vers 10 h » (heure supérieure) ; prudent 14/11 vers 21 h",
          oi["instant"] == "2026-11-13T10:00" and oi["prudent"]["instant"] == "2026-11-14T21:00", (oi["instant"], oi["prudent"]["instant"]))
        ej = coulage(m, date="2026-11-02", heure=6, temperature__T=20.0)
        ej["objectifs"] = [objectif(m, "X", mode="mpa", mpa=float(f_main(7.93, 30.0, s, "fck")), base="fck")]
        oj = Mm.calculer(ej)["objectifs"][0]
        v("4 · la date n'est JAMAIS tronquée au jour : 7,93 j après le 02/11 à 6 h → le 10/11 vers 5 h (l'ancien écran écrivait « le 09/11 »)",
          dec(oj["t"], 7.93, 2) and oj["instant"] == "2026-11-10T05:00", (oj["t"], oj["instant"]))
        ek = coulage(m, date="2026-11-02", heure=8)
        ek["objectifs"][0]["age"] = 3.0
        ok_ = Mm.calculer(ek)["objectifs"][0]
        v("4 · un âge entier tombe à l'heure du bétonnage (J + 3 depuis le 02/11 à 8 h : le 05/11 à 8 h) ; un instant exact n'est pas repoussé d'une heure",
          ok_["instant_prevu"] == "2026-11-05T08:00" and Mm._instant("2026-11-02", 8, 2.0) == "2026-11-04T08:00"
          and Mm._instant("2026-11-02", 8, 2.0 + 1e-9) == "2026-11-04T08:00" and Mm._instant("2026-11-02", 8, 2.0 + 0.001) == "2026-11-04T09:00"
          and Mm._instant(None, 8, 2.0) is None and Mm._instant("2026-11-02", 8, None) is None)
        v("4 · avant 3 jours équivalents la valeur est INDICATIVE (5 MPa moyenne : t_T = 0,89 j → indicatif, avertissement) ; 70 % de f_ck (10,7 j) ne l'est pas",
          Mm.calculer(dict(Mm.nouvel_element(), objectifs=[objectif(m, "Hors gel", mode="mpa", mpa=5.0, base="fcm")]))["objectifs"][0]["indicatif"]
          and "indicatif" in codes(Mm.calculer(dict(Mm.nouvel_element(), objectifs=[objectif(m, "Hors gel", mode="mpa", mpa=5.0, base="fcm")])))
          and not o["indicatif"])
        v("4 · un objectif sans nom s'appelle « Objectif n » ; huit objectifs se calculent, dans l'ordre",
          [x["nom"] for x in Mm.calculer(dict(Mm.nouvel_element(), objectifs=[objectif(m, "", pct=10.0 * (i + 1)) for i in range(8)]))["objectifs"]]
          == [f"Objectif {i + 1}" for i in range(8)])
    groupe("4 · objectifs", objectifs)

    # ------------------------------------------------------------- 5. le gel, le temps froid
    def gel():
        e = coulage(m, heure=12, temperature__mode="releve", temperature__releve=releve([10.0, -2.0, 8.0, 8.0, 8.0, 8.0]))
        r = Mm.calculer(e)
        s = r["materiau"]["s"]
        tT5 = 28.0 / (1.0 - math.log(5.0 / 38.0) / s) ** 2          # 0,8870 j : la résistance MOYENNE atteint 5 MPa
        cum1 = 0.5 * 0.60 + 0.24                                   # fin du 2ᵉ jour (10 °C une demi-journée, puis −2 °C : k = 0,24)
        t5 = 1.5 + (tT5 - cum1) / 0.54                             # atteints le 3ᵉ jour, à 8 °C
        g = r["gel"]
        v("5 · un jour à −2 °C le 2ᵉ jour : ALERTE — le gel vient (au début du jour, t = 0,5 j) avant que le béton ait 5 MPa (f_cm(t) = 5 à 2,14 j)",
          g["alerte"] and g["premier_jour"] == 1 and g["T"] == -2.0 and g["t"] == 0.5 and g["base"] == "moyenne" and pres(g["t_5MPa"], t5, 1e-12)
          and dec(t5, 2.14, 2) and any(a["code"] == "gel" and a["niveau"] == "grave" and "PROTÉGER" in a["message"] for a in r["avertissements"])
          and r["profil"][1]["gel"] and not r["profil"][0]["gel"], g)
        e2 = copy.deepcopy(e)
        e2["loi"]["mode"] = "ec2"
        r2 = Mm.calculer(e2)
        v("5 · ce jour à −2 °C compte k = 0,24 sous la loi belge (0,30 − 0,03·2) mais 0 sous (B.10), hors de son domaine ; l'alerte de gel ne dépend PAS de la loi",
          pres(r["profil"][1]["k"], 0.24, 1e-12) and r2["profil"][1]["k"] == 0.0 and r2["gel"]["alerte"] and r2["gel"]["premier_jour"] == 1
          and r2["gel"]["t"] == g["t"] and "loi_sous_zero" in codes(r2) and "loi_sous_zero" not in codes(r), (r["profil"][1]["k"], r2["profil"][1]["k"]))
        tard = Mm.calculer(coulage(m, heure=12, temperature__mode="releve", temperature__releve=releve([20.0, 20.0, 20.0, -3.0, 8.0])))
        v("5 · le gel APRÈS 5 MPa (trois jours à 20 °C, 5 MPa à 0,89 j, puis −3 °C) : le jour de gel est relevé, sans alerte",
          not tard["gel"]["alerte"] and tard["gel"]["premier_jour"] == 3 and tard["gel"]["t"] == 2.5 and pres(tard["gel"]["t_5MPa"], tT5, 1e-12)
          and "gel" not in codes(tard), tard["gel"])
        zero = Mm.calculer(coulage(m, temperature__T=0.0))
        v("5 · une moyenne journalière de 0 °C est un jour de gel (≤ 0) ; 0,1 °C n'en est pas un ; à température constante le gel est là dès le bétonnage (t = 0)",
          zero["gel"]["alerte"] and zero["gel"]["premier_jour"] == 0 and zero["gel"]["t"] == 0.0
          and not Mm.calculer(coulage(m, temperature__T=0.1))["gel"]["alerte"])
        prud = Mm.calculer(coulage(m, heure=12, temperature__mode="releve", temperature__releve=releve([6.0, 2.0, 8.0, 8.0, 8.0], "mpppp")))
        v("5 · un jour PRÉVU à 2 °C : pas de gel au nominal, gel dans la variante prudente (2 − 3 = −1 °C) → « gel possible », sans l'alerte grave",
          not prud["gel"]["alerte"] and prud["gel"]["prudent"]["alerte"] and prud["gel"]["prudent"]["premier_jour"] == 1
          and "gel_prudent" in codes(prud) and "gel" not in codes(prud), (prud["gel"], codes(prud)))
        v("5 · le rappel « temps froid » : une température SOUS 5 °C dans les 72 premières heures (le jour à −2 °C, puis à 4,9 °C au 3ᵉ jour) ; à 5,0 °C non ; au 5ᵉ jour non",
          r["temps_froid"]["rappel"] and r["temps_froid"]["jour"] == 1 and "temps_froid" in codes(r)
          and Mm.calculer(coulage(m, temperature__mode="releve", temperature__releve=releve([10.0, 10.0, 4.9, 10.0])))["temps_froid"]["jour"] == 2
          and not Mm.calculer(coulage(m, temperature__mode="releve", temperature__releve=releve([10.0, 10.0, 5.0, 10.0])))["temps_froid"]["rappel"]
          and not Mm.calculer(coulage(m, temperature__mode="releve", temperature__releve=releve([10.0, 10.0, 10.0, 10.0, 4.0, 10.0])))["temps_froid"]["rappel"]
          and Mm.calculer(coulage(m, heure=12, temperature__mode="releve", temperature__releve=releve([10.0, 10.0, 10.0, 4.0, 10.0])))["temps_froid"]["jour"] == 3
          and not Mm.calculer(coulage(m, heure=0, temperature__mode="releve", temperature__releve=releve([10.0, 10.0, 10.0, 4.0, 10.0])))["temps_froid"]["rappel"],
          r["temps_froid"])
        v("5 · les seuils : 5 MPa (EN 13670 §8.5(12)), comparés à la résistance MOYENNE ; 5 °C dans les 72 heures (NIT 285 §9.4)",
          Mm.SEUIL_GEL_MPA == 5.0 and Mm.SEUIL_FROID_C == 5.0 and Mm.HEURES_FROID == 72.0 and g["seuil"] == 5.0
          and pres(Mm.calculer(Mm.nouvel_element())["gel"]["t_5MPa"], tT5, 1e-12))
        # la minimale horaire de la météo (détail horaire chargé) : elle sert l'alerte, et le moteur dit laquelle il a utilisée
        J = lambda n: (AUJ_M + _dt.timedelta(days=n)).isoformat()  # noqa: E731
        em = coulage_meteo(m, "Bruxelles, Belgique", J(-12), J(0), horaire=True, heure=8)
        rm = Mm.calculer(em)
        eb = copy.deepcopy(em)
        eb["temperature"]["meteo"]["horaire"] = None
        rb = Mm.calculer(eb)
        d0 = _dt.date.fromisoformat(J(-12))
        mins = [min(FO.t_heure(d0 + _dt.timedelta(days=j), h) for h in range(8 if j == 0 else 0, 24)) for j in range(13)]
        moys = [FO.t_jour(d0 + _dt.timedelta(days=j)) for j in range(13)]
        j_min = next((j for j, x in enumerate(mins) if x < 0), None)
        j_moy = next((j for j, x in enumerate(moys) if x <= 0), None)
        v("5 · météo avec détail horaire : le gel se lit sur la MINIMALE horaire du jour (comptée dès l'heure de fin du bétonnage le jour 0) — le premier jour de "
          "gel est celui que donnent les valeurs horaires du faux service, et le moteur dit « minimale » ; sans le détail horaire : la moyenne, dite « moyenne »",
          j_min is not None and rm["gel"]["premier_jour"] == j_min and rm["gel"]["base"] == "minimale" and pres(rm["gel"]["T"], mins[j_min], 1e-12)
          and rb["gel"]["base"] == "moyenne" and rb["gel"]["premier_jour"] == j_moy and (j_moy is None or j_moy > j_min)
          and [pres(x["T_gel"], mins[j], 1e-12) for j, x in enumerate(rm["profil"][:13])] == [True] * 13
          and all(x["base_gel"] == "minimale" for x in rm["profil"][:13]) and rm["profil"][13]["base_gel"] == "moyenne"
          and rm["tableau"]["horaire"] and not rb["tableau"]["horaire"], (j_min, j_moy, rm["gel"], rb["gel"]))
        v("5 · … le tableau garde T min h et T max h de chaque jour (les 24 heures), et le calcul d'âge équivalent reste celui des moyennes journalières",
          all(pres(x["T_min_h"], min(FO.t_heure(d0 + _dt.timedelta(days=j), h) for h in range(24)), 1e-12)
              and pres(x["T_max_h"], max(FO.t_heure(d0 + _dt.timedelta(days=j), h) for h in range(24)), 1e-12) for j, x in enumerate(rm["profil"][:13]))
          and [x["t_T"] for x in rm["profil"]] == [x["t_T"] for x in rb["profil"]] and rm["profil"][13]["T_min_h"] is None
          and rm["temps"]["date"] == J(-12) and rm["temps"]["date_origine"] == "releve")
    groupe("5 · gel et temps froid", gel)

    # ------------------------------------------------------------- 6. l'essai sur éprouvettes
    def essai():
        e = coulage(m, essai__actif=True, essai__age=7.0, essai__mpa=25.0, essai__eprouvette="cube")
        r = Mm.calculer(e)
        x = r["essai"]
        s = -math.log(0.30) / (RAC14 - 1.0)
        f_cyl = 25.0 * 30.0 / 37.0                                 # le cube de 150 mm ramené au cylindre : 20,27 MPa
        f_prev = 38.0 * math.exp(s * (1.0 - math.sqrt(28.0 / 7.0)))  # f_cm(7 j) prévu : 24,49 MPa
        v("6 · essai : 25 MPa sur cubes à 7 j (20 °C) → 20,27 MPa sur cylindre (× 30/37), comparés à la MOYENNE prévue f_cm(7 j) = 24,49 MPa : écart −17,2 %",
          pres(x["f_cyl"], f_cyl, 1e-12) and pres(x["f_prevu"], f_prev, 1e-12) and pres(x["ecart"], (f_cyl - f_prev) / f_prev * 100.0, 1e-10)
          and dec(f_cyl, 20.27, 2) and dec(f_prev, 24.49, 2) and dec(x["ecart"], -17.2, 1) and x["t_T"] == 7.0 and x["rapport_cube"] == 30.0 / 37.0
          and not x["cale"] and r["materiau"]["s"] == Lm.s_de_r(0.30) and not r["materiau"]["cale"] and "essai_sous" in codes(r), x)
        s_cal = math.log(f_cyl / 38.0) / (1.0 - math.sqrt(28.0 / 7.0))  # 0,6284 : le s qui fait passer la courbe par la mesure
        v("6 · le s calé : ln(f_mes/f_cm)/(1 − √(28/t_T)) = 0,6284, et le r qu'il suppose (0,179) ; sans « caler », la courbe garde s = 0,439",
          pres(x["s_cale"], s_cal, 1e-12) and dec(s_cal, 0.6284, 4) and pres(x["r_cale"], math.exp(s_cal * (1.0 - RAC14)), 1e-12) and dec(x["r_cale"], 0.179, 3)
          and x["calage_refuse"] is None, (x["s_cale"], x["r_cale"]))
        e2 = copy.deepcopy(e)
        e2["essai"]["caler"] = True
        r2 = Mm.calculer(e2)
        tT_cal = 28.0 / (1.0 - math.log(29.0 / 38.0) / s_cal) ** 2
        v("6 · « caler la courbe sur l'essai » : le s retenu est le s calé (dit « calé »), la courbe passe par la mesure à 1e-9 près, et TOUTES les échéances "
          "suivent (70 % de f_ck : 13,69 j au lieu de 10,73 j)",
          r2["materiau"]["cale"] and pres(r2["materiau"]["s"], s_cal, 1e-12) and abs(Lm.fcm_t(r2["essai"]["t_T"], 30.0, r2["materiau"]["s"]) - f_cyl) < 1e-9
          and pres(r2["objectifs"][0]["t"], tT_cal, 1e-12) and dec(tT_cal, 13.69, 2) and "cale" in codes(r2) and r2["materiau"]["durcissement"]["s"] == Lm.s_de_r(0.30)
          and pres(r2["essai"]["f_prevu"], f_prev, 1e-12) and pres(r2["materiau"]["r"], math.exp(s_cal * (1.0 - RAC14)), 1e-12),
          (r2["materiau"]["s"], r2["objectifs"][0]["t"]))
        pts = [(t, f) for t, f in zip(r2["courbes"]["t"], r2["courbes"]["fcm"])]
        v("6 · … la courbe TRACÉE f_cm(t) passe par le point de l'essai (interpolée entre ses deux points voisins, à 0,05 MPa près)",
          any(a[0] <= 7.0 <= b[0] and abs(a[1] + (b[1] - a[1]) * (7.0 - a[0]) / (b[0] - a[0]) - f_cyl) < 0.05 for a, b in zip(pts, pts[1:]))
          and r2["courbes"]["essai"] == {"t": 7.0, "f": f_cyl})
        cyl = Mm.calculer(coulage(m, essai__actif=True, essai__age=7.0, essai__mpa=25.0, essai__eprouvette="cylindre"))["essai"]
        v("6 · un cylindre n'est pas converti (25 MPa restent 25 MPa : +2,1 % sur la prévision)",
          cyl["f_cyl"] == 25.0 and cyl["rapport_cube"] == 1.0 and pres(cyl["ecart"], (25.0 - f_prev) / f_prev * 100.0, 1e-10) and dec(cyl["ecart"], 2.1, 1))
        refus = {}
        for nom, k in (("28 j", dict(essai__age=28.0, essai__mpa=40.0)), ("hors bornes", dict(essai__age=3.0, essai__mpa=40.0)),
                       ("trop lent", dict(essai__age=7.0, essai__mpa=12.0))):
            rr = Mm.calculer(coulage(m, essai__actif=True, essai__caler=True, **k))
            refus[nom] = (rr["essai"]["calage_refuse"] is not None and not rr["essai"]["cale"] and not rr["materiau"]["cale"]
                          and rr["materiau"]["s"] == Lm.s_de_r(0.30) and any(a["code"] == "calage_refuse" and a["niveau"] == "grave" for a in rr["avertissements"]))
        v("6 · le calage est REFUSÉ, avec sa raison, et la courbe garde son s : un essai à 28 jours équivalents (il ne dit rien du durcissement), un s calé hors de "
          "[0,10 ; 0,80] (mesure trop forte : s < 0,10 ; trop faible : s > 0,80) — jamais ramené à une borne",
          all(refus.values()) and Lm.S_CALE_MIN == 0.10 and Lm.S_CALE_MAX == 0.80, refus)
        froid = Mm.calculer(coulage(m, temperature__T=-5.0, loi__mode="ec2", essai__actif=True, essai__caler=True))
        v("6 · un essai dont l'âge équivalent est nul (gel sous (B.10)) : pas d'écart, calage refusé, aucune exception",
          froid["essai"]["t_T"] == 0.0 and froid["essai"]["ecart"] is None and froid["essai"]["calage_refuse"] and not froid["essai"]["cale"])
        v("6 · sans essai : rien (None), aucun avertissement d'essai", Mm.calculer(Mm.nouvel_element())["essai"] is None)
    groupe("6 · essai", essai)

    # ------------------------------------------------------------- 7. le sélecteur de classe
    def selecteur():
        c = Mm.candidats_classes(Mm.nouvel_element())
        s = Lm.s_de_r(0.30)
        v("7 · sélecteur : les huit classes dans l'ordre, chacune à l'âge où le PREMIER objectif est atteint (70 % de SA f_ck, à la main) ; ✓ la classe du "
          "coulage ; aucune ★ sans âge prévu",
          [x["classe"] for x in c] == list(CLASSES_NORME) and all(pres(x["t"], tT_main(0.7 * fck, fck, s, "fck"), 1e-12) for x, (fck, _) in zip(c, CLASSES_NORME.values()))
          and [x["classe"] for x in c if x["retenue"]] == ["C30/37"] and not any(x["optimal"] for x in c) and all(x["ok"] for x in c),
          [(x["classe"], x["t"]) for x in c])
        e = Mm.nouvel_element()
        e["objectifs"] = [objectif(m, "Décoffrage", mode="mpa", mpa=15.0, base="fck", age=5.0)]
        c2 = Mm.candidats_classes(e)
        att = {cl: 15.0 / f_main(5.0, fck, s, "fck") for cl, (fck, _) in CLASSES_NORME.items()}
        prem = next(cl for cl in CLASSES_NORME if att[cl] <= 1.0)
        v("7 · sélecteur, objectif daté (15 MPa caractéristique à J + 5) : le taux de chaque classe à la main ; ★ la plus petite qui tient (C35/45, 96 %), une seule ; "
          "le C30/37 du coulage ne tient pas (117 %)",
          all(pres(x["taux"], att[x["classe"]], 1e-12) for x in c2) and [x["classe"] for x in c2 if x["optimal"]] == [prem] and prem == "C35/45"
          and [x["ok"] for x in c2] == [att[cl] <= 1.0 for cl in CLASSES_NORME] and dec(att["C30/37"], 1.17, 2) and dec(att["C35/45"], 0.96, 2), (prem, att))
        e3 = coulage(m, essai__actif=True, essai__age=7.0, essai__mpa=25.0, essai__caler=True)
        s_cal = Mm.calculer(e3)["materiau"]["s"]
        v("7 · sélecteur : le MÊME durcissement pour toutes les classes — le s retenu du coulage, calé sur l'essai s'il l'est",
          all(pres(x["t"], tT_main(0.7 * fck, fck, s_cal, "fck"), 1e-12) for x, (fck, _) in zip(Mm.candidats_classes(e3), CLASSES_NORME.values())))
        e4 = Mm.nouvel_element()
        e4["objectifs"] = [objectif(m, "Fort", mode="mpa", mpa=28.0, base="fck")]
        c4 = Mm.candidats_classes(e4)
        v("7 · sélecteur : une classe qui n'atteint pas la cible (28 MPa : C16/20 à C25/30) est dite non atteignable, sans âge",
          [x["classe"] for x in c4 if not x["atteignable"]] == ["C16/20", "C20/25", "C25/30"] and all(x["t"] is None and not x["ok"] for x in c4[:3])
          and all(x["t"] is not None for x in c4[3:]))
    groupe("7 · sélecteur", selecteur)

    # ------------------------------------------------------------- 8. l'entrée
    def entree():
        d = Mm.nouvel_element()
        v("8 · le coulage NEUF : C30/37, béton à durcissement moyen, midi sans date, 20 °C constante, loi belge, marge 3 °C, un objectif « Décoffrage » à 70 % "
          "de f_ck non daté, aucun essai",
          d["classe"] == "C30/37" and d["durcissement"] == {"mode": "beton", "ciment": "N", "beton": "moyen", "r": 0.30} and d["date"] is None
          and d["heure"] == 12 and d["temperature"]["mode"] == "constante" and d["temperature"]["T"] == 20.0 and d["loi"]["mode"] == "nbn"
          and d["marge"] == 3.0 and d["objectifs"] == [{"nom": "Décoffrage", "mode": "pct", "pct": 70.0, "mpa": 21.0, "base": "fck", "age": None}]
          and d["essai"]["actif"] is False and d["nom"] == "Coulage 1", d)
        v("8 · normaliser : {} donne le coulage neuf ; idempotente ; une clé absente prend son défaut ; le coulage reçu n'est pas modifié",
          ENm.normaliser({}) == d and ENm.normaliser(ENm.normaliser(d)) == ENm.normaliser(d) and ENm.normaliser({"classe": "C40/50"})["heure"] == 12
          and ENm.normaliser({"objectifs": [{"nom": "X"}]})["objectifs"][0]["pct"] == 70.0
          and ENm.normaliser({"classe": "C40/50", "objectifs": [{}]})["objectifs"][0]["mpa"] == 28.0)
        avant = copy.deepcopy(d)
        Mm.calculer(d)
        Mm.candidats_classes(d)
        v("8 · calculer et candidats_classes ne modifient pas le coulage reçu", d == avant)
        REFUS = (("classe inconnue", {"classe": "C12/15"}), ("classe C55/67", {"classe": "C55/67"}), ("nom trop long", {"nom": "x" * 81}), ("nom en nombre", {"nom": 3}),
                 ("durcissement inconnu", {"durcissement": {"mode": "vite"}}), ("ciment inconnu", {"durcissement": {"ciment": "X"}}),
                 ("béton inconnu", {"durcissement": {"beton": "très lent"}}), ("r sous 0,15", {"durcissement": {"r": 0.149}}), ("r au-dessus de 0,70", {"durcissement": {"r": 0.701}}),
                 ("r en texte", {"durcissement": {"r": "0,3"}}), ("clé de durcissement inconnue", {"durcissement": {"s": 0.3}}),
                 ("date illisible", {"date": "demain"}), ("date en nombre", {"date": 20261102}), ("date impossible", {"date": "2026-02-30"}),
                 ("heure 24", {"heure": 24}), ("heure −1", {"heure": -1}), ("heure 13,5", {"heure": 13.5}), ("heure en texte", {"heure": "midi"}), ("heure booléenne", {"heure": True}),
                 ("T sous −20 °C", {"temperature": {"T": -20.1}}), ("T au-dessus de 60 °C", {"temperature": {"T": 60.1}}), ("T en texte", {"temperature": {"T": "20"}}),
                 ("T infinie", {"temperature": {"T": float("inf")}}), ("T NaN", {"temperature": {"T": float("nan")}}), ("mode inconnu", {"temperature": {"mode": "météo"}}),
                 ("relevé illisible", {"temperature": {"mode": "releve", "releve": "douze"}}), ("relevé en nombre", {"temperature": {"releve": 12}}),
                 ("clé de température inconnue", {"temperature": {"cure": 3}}), ("météo abîmée", {"temperature": {"meteo": {"lieu": "x"}}}),
                 ("début illisible", {"temperature": {"debut": "hier"}}), ("marge négative", {"marge": -0.1}), ("marge 10,1", {"marge": 10.1}), ("marge en texte", {"marge": "3"}),
                 ("aucun objectif", {"objectifs": []}), ("neuf objectifs", {"objectifs": [{}] * 9}), ("objectifs non liste", {"objectifs": {"nom": "x"}}),
                 ("objectif non objet", {"objectifs": ["x"]}), ("cible 4,9 %", {"objectifs": [{"pct": 4.9}]}), ("cible 100,1 %", {"objectifs": [{"pct": 100.1}]}),
                 ("cible 0,4 MPa", {"objectifs": [{"mpa": 0.4}]}), ("cible 100,1 MPa", {"objectifs": [{"mpa": 100.1}]}), ("mode de cible inconnu", {"objectifs": [{"mode": "kN"}]}),
                 ("base inconnue", {"objectifs": [{"base": "fctm"}]}), ("âge prévu 0,09 j", {"objectifs": [{"age": 0.09}]}), ("âge prévu 365,1 j", {"objectifs": [{"age": 365.1}]}),
                 ("âge prévu en texte", {"objectifs": [{"age": "3"}]}), ("nom d'objectif trop long", {"objectifs": [{"nom": "x" * 41}]}),
                 ("clé d'objectif inconnue", {"objectifs": [{"seuil": 3}]}), ("loi inconnue", {"loi": {"mode": "saul"}}), ("E_a 19,9", {"loi": {"Ea_kJ": 19.9}}),
                 ("E_a 60,1", {"loi": {"Ea_kJ": 60.1}}), ("essai : case en texte", {"essai": {"actif": "oui"}}), ("essai : âge 0,09", {"essai": {"age": 0.09}}),
                 ("essai : âge 366", {"essai": {"age": 366.0}}), ("essai : 0,4 MPa", {"essai": {"mpa": 0.4}}), ("essai : 150,1 MPa", {"essai": {"mpa": 150.1}}),
                 ("essai : éprouvette inconnue", {"essai": {"eprouvette": "prisme"}}), ("essai : caler en nombre", {"essai": {"caler": 1}}),
                 ("clé inconnue", {"beton": "C30/37"}), ("clé de l'ancien format", {"ciment": "N"}), ("coulage non objet", None), ("coulage liste", [1]))
        acceptes = []
        for nom, champ in REFUS:
            try:
                ENm.normaliser(champ)
                acceptes.append(f"{nom} : accepté")
            except ValueError as x:
                if not str(x).strip() or "Traceback" in str(x):
                    acceptes.append(f"{nom} : message vide")
            except Exception as x:  # noqa: BLE001 — une autre exception n'est pas un refus : c'est une panne
                acceptes.append(f"{nom} : {type(x).__name__} {x}")
        v(f"8 · {len(REFUS)} saisies hors domaine lèvent ValueError avec un message (jamais une valeur ramenée à une borne en silence)", not acceptes, acceptes)
        BORNES = (({"temperature": {"T": -20.0}}, {"temperature": {"T": 60.0}}), ({"marge": 0.0}, {"marge": 10.0}), ({"loi": {"Ea_kJ": 20.0}}, {"loi": {"Ea_kJ": 60.0}}),
                  ({"durcissement": {"r": 0.15}}, {"durcissement": {"r": 0.70}}), ({"objectifs": [{"age": 0.1}]}, {"objectifs": [{"age": 365.0}]}),
                  ({"objectifs": [{"mpa": 0.5}]}, {"objectifs": [{"mpa": 100.0}]}), ({"objectifs": [{"pct": 5.0}]}, {"objectifs": [{"pct": 100.0}]}),
                  ({"heure": 0}, {"heure": 23}))
        refusees = []
        for bas, haut in BORNES:
            for x in (bas, haut):
                try:
                    ENm.normaliser(x)
                except ValueError as err:
                    refusees.append((x, str(err)))
        v("8 · les bornes elles-mêmes sont ADMISES : T de −20 à 60 °C, marge 0 à 10 °C, E_a 20 à 60 kJ/mol, r 0,15 à 0,70, âge prévu 0,1 à 365 j, cible 0,5 à "
          "100 MPa ou 5 à 100 %, heure 0 à 23", not refusees, refusees[:2])
        msg = ""
        try:
            ENm.normaliser({"temperature": {"T": -25}})
        except ValueError as x:
            msg = str(x)
        v("8 · le message d'un refus nomme le champ et la borne (« temperature/T : -25 est sous le minimum -20 »)", msg == "temperature/T : -25 est sous le minimum -20", msg)
        v("8 · une date se lit JJ/MM/AAAA ou AAAA-MM-JJ et se garde AAAA-MM-JJ ; vide = sans date",
          ENm.normaliser({"date": "02/11/2026"})["date"] == "2026-11-02" and ENm.normaliser({"date": "2026-11-02"})["date"] == "2026-11-02"
          and ENm.normaliser({"date": "  "})["date"] is None and ENm.normaliser({"date": None})["date"] is None)
        ancien = {"nom": "Voile R+1", "classe": "C30/37", "ciment": "N", "date_coulage": None, "age": 28, "seuil": {"mode": "pct", "pct": 75},
                  "temperature": {"mode": "constante", "T": 20.0}}
        v("8 · un coulage du module d'AVANT le 10/11/2026 (ses clés : ciment, seuil, age…) est reconnu ; relu, il garde son NOM et tout le reste est neuf ; le "
          "normaliser tel quel est refusé (clé inconnue), jamais lu de travers",
          ENm.est_ancien(ancien) and not ENm.est_ancien(d) and not ENm.est_ancien(None) and ENm.depuis_ancien(ancien) == dict(d, nom="Voile R+1")
          and ENm.depuis_ancien({"nom": "", "ciment": "N"}, "Coulage 3")["nom"] == "Coulage 3" and len(ENm.depuis_ancien({"nom": "x" * 300, "age": 3})["nom"]) == 80)
        mo = {k: (nom, mode, val, base) for k, nom, mode, val, base, _src in Mm.MODELES_OBJECTIF}
        v("8 · les six modèles d'objectif, chacun avec sa source : hors gel 5 MPa et décoffrage vertical 5 MPa (moyenne), reprise de bétonnage 10 MPa (moyenne), "
          "50, 70 et 100 % de f_ck (caractéristique)",
          mo == {"hors_gel": ("Hors gel", "mpa", 5.0, "fcm"), "decoffrage_vertical": ("Décoffrage vertical", "mpa", 5.0, "fcm"),
                 "reprise": ("Reprise de bétonnage", "mpa", 10.0, "fcm"), "pct50": ("50 % de f_ck", "pct", 50.0, "fck"),
                 "pct70": ("70 % de f_ck", "pct", 70.0, "fck"), "pct100": ("100 % de f_ck", "pct", 100.0, "fck")}
          and all(len(x[5]) > 20 for x in Mm.MODELES_OBJECTIF) and "EN 13670" in Mm.MODELES_OBJECTIF[0][5], mo)
        v("8 · un modèle donne un objectif que la normalisation accepte tel quel ; un modèle inconnu est refusé",
          all(ENm.normaliser_objectif(Mm.modele_objectif(k, 30.0), "o", 30.0) == Mm.modele_objectif(k, 30.0) for k in mo)
          and Mm.modele_objectif("hors_gel", 30.0)["mpa"] == 5.0 and Mm.modele_objectif("pct50", 30.0)["pct"] == 50.0)
        try:
            Mm.modele_objectif("inconnu", 30.0)
            v("8 · modèle inconnu refusé", False)
        except ValueError:
            pass
        v("8 · ce que le module ne couvre pas est écrit (les délais minimaux de la NBN B 15-400 et le tableau F.8-ANB, non relus ; l'EN 1992-1-1:2023…)",
          len(Mm.NON_COUVERT) >= 4 and any("F.8-ANB" in x for x in Mm.NON_COUVERT) and any("2023" in x for x in Mm.NON_COUVERT)
          and Mm.calculer(d)["non_couvert"] == list(Mm.NON_COUVERT))
    groupe("8 · entrée", entree)

    # ------------------------------------------------------------- 9. le format, les formules, les invariants
    def format_():
        rng = random.Random(20261110)
        cas = [("neuf", Mm.nouvel_element())]
        for i in range(60):
            e = Mm.nouvel_element()
            e["classe"] = rng.choice(list(CLASSES_NORME))
            e["durcissement"].update(mode=rng.choice(("ciment", "beton", "r")), ciment=rng.choice("RNS"), beton=rng.choice(list(R_BETON)), r=round(rng.uniform(0.15, 0.7), 3))
            e["date"] = rng.choice([None, "2026-11-02", "2026-10-30", "2026-11-06"])
            e["heure"] = rng.randint(0, 23)
            if rng.random() < 0.6:
                n = rng.randint(1, 30)
                d0 = _dt.date(2026, 11, 2) + _dt.timedelta(days=rng.randint(-3, 3))
                e["temperature"].update(mode="releve", releve=releve([None if rng.random() < 0.08 else round(rng.uniform(-12, 30), 1) for _ in range(n)],
                                                                    "".join(rng.choice("mmmpx") for _ in range(n)), d0 if rng.random() < 0.5 else None))
            else:
                e["temperature"]["T"] = round(rng.uniform(-20, 60), 1)
            e["marge"] = rng.choice([0.0, 3.0, 7.5])
            e["objectifs"] = [objectif(m, f"O{k}", mode=rng.choice(("pct", "mpa")), pct=round(rng.uniform(5, 100), 1), mpa=round(rng.uniform(0.5, 60), 1),
                                       base=rng.choice(("fck", "fcm")), age=rng.choice([None, round(rng.uniform(0.1, 40), 1)])) for k in range(rng.randint(1, 5))]
            e["loi"].update(mode=rng.choice(("nbn", "ec2", "perso")), Ea_kJ=round(rng.uniform(20, 60), 1))
            e["essai"].update(actif=rng.random() < 0.5, age=round(rng.uniform(0.5, 30), 1), mpa=round(rng.uniform(2, 60), 1), eprouvette=rng.choice(("cube", "cylindre")),
                              caler=rng.random() < 0.5)
            cas.append((f"tirage {i:02d}", e))
        res = [(nom, e, Mm.calculer(e)) for nom, e in cas]
        v(f"9 · la sortie de calculer() est du JSON STRICT (ni NaN ni infini) et se relit à l'identique ({len(res)} coulages : le neuf et 60 tirages)",
          all(json.loads(json.dumps(r, allow_nan=False)) == r for _, _, r in res))
        v("9 · l'entrée rendue est la normalisée, stable à la relecture", all(ENm.normaliser(r["entree"]) == r["entree"] for _, _, r in res))
        # « symbole = expression = valeurs = résultat » : chaque résultat se retrouve à partir des SEULS termes que le moteur donne
        # (la variante prudente porte l'indice « _prud » : T_prud, k_prud — le terme se lit sous l'un ou l'autre symbole,
        # jamais les deux dans la même formule)
        def T_(t):
            assert ("T" in t) != ("T_prud" in t), t
            return t["T"] if "T" in t else t["T_prud"]

        def k_(t):
            assert ("k" in t) != ("k_prud" in t), t
            return t["k"] if "k" in t else t["k_prud"]

        def nbn(t):
            return t["k_a"] + (t["k_b"] - t["k_a"]) * (T_(t) - t["T_a"]) / (t["T_b"] - t["T_a"])
        FORMULES = {
            "cible_pct": lambda t: t["p"] * t["f_ck"] / 100.0,
            "fcm": lambda t: t["f_ck"] + 8.0,
            "beta_req_fck": lambda t: (t["f_cible"] + 8.0) / t["f_cm"],
            "beta_req_fcm": lambda t: (t.get("f_cible", t.get("f_gel"))) / t["f_cm"],
            "t_T_req": lambda t: 28.0 / (1.0 - math.log(t["β_req"]) / t["s"]) ** 2,
            "t_T_req_28": lambda t: 28.0,
            "T_prudent": lambda t: t["T"] - t["marge"],
            "k_nbn": nbn, "k_nbn_nul": lambda t: 0.0, "k_nbn_plafond": lambda t: 1.0, "k_sous_zero": lambda t: 0.0, "k_sans_donnee": lambda t: 0.0,
            "k_ec2": lambda t: math.exp(-4000.0 * (1.0 / (273.0 + T_(t)) - 1.0 / 293.0)),
            "k_perso": lambda t: math.exp(-(t["E_a"] * 1000.0 / 8.314) * (1.0 / (273.0 + T_(t)) - 1.0 / 293.0)),
            "t_atteint": lambda t: t["t_d"] + (t["t_T,req"] - t["t_T,d"]) / k_(t),
            "t_T_a": lambda t: t["t_T,d"] + t["k"] * (t["t"] - t["t_d"]),
            "beta": lambda t: math.exp(t["s"] * (1.0 - math.sqrt(28.0 / t["t_T"]))) if t["t_T"] > 0 else 0.0,
            "beta_28": lambda t: 1.0,
            "fcm_t": lambda t: t["β_cc"] * t["f_cm"],
            "fck_t": lambda t: max(0.0, t["f_cm(t)"] - 8.0),
            "taux": lambda t: t["f_cible"] / t["f(t_p)"],
            "rapport_cube": lambda t: t["f_ck"] / t["f_ck,cube"],
            "f_cyl": lambda t: t["f_cube"] * t["f_ck/f_ck,cube"],
            "ecart": lambda t: (t["f_mes"] - t["f_cm(t)"]) / t["f_cm(t)"] * 100.0,
            "s_cale": lambda t: math.log(t["f_mes"] / t["f_cm"]) / (1.0 - math.sqrt(28.0 / t["t_T"])),
            "r_de_s": lambda t: math.exp(t["s"] * (1.0 - RAC14)),
            "s_de_r": lambda t: -math.log(t["r"]) / (RAC14 - 1.0),
        }
        fautes, n_f, vues = [], 0, set()
        for nom, _, r in res:
            blocs = [("materiau", r["materiau"]["formules"]), ("gel", r["gel"]["formules"]), ("k", [r["temperature"]["formule_k"]])]
            blocs += [(f"objectif {i}", o["formules"]) for i, o in enumerate(r["objectifs"])]
            if r["essai"]:
                blocs.append(("essai", r["essai"]["formules"]))
            for bloc, fx in blocs:
                for f in fx:
                    n_f += 1
                    vues.add(f["cle"])
                    t = {y["symbole"]: y["valeur"] for y in f["termes"]}
                    if f["cle"] not in FORMULES:
                        fautes.append(f"{nom}/{bloc} : formule inconnue « {f['cle']} »")
                    elif not pres(FORMULES[f["cle"]](t), f["resultat"], 1e-10):
                        fautes.append(f"{nom}/{bloc}/{f['cle']} : {FORMULES[f['cle']](t)!r} ≠ {f['resultat']!r}")
        v(f"9 · « symbole = expression = valeurs = résultat » : les {n_f} formules ({len(vues)} sortes) se recalculent à partir de leurs seuls termes",
          not fautes and n_f > 700 and len(vues) >= 22, (fautes[:3], n_f, sorted(set(FORMULES) - vues)))
        # la NOTATION de la variante prudente (relecture du 11/11/2026 : « k,p » et « t,p » s'écrivaient sans indice, et la
        # formule de k y nommait « T » une température abaissée) : T_prud, k_prud, t_prud — et un terme qui reprend une de ces
        # grandeurs porte SON symbole. t_p reste l'âge PRÉVU d'une opération.
        sy0 = {f["symbole"]: f for f in Mm.calculer(Mm.nouvel_element())["objectifs"][0]["formules"]}
        v("9 · la variante prudente s'écrit en indice : T_prud = T − marge, k_prud calculé sur T_prud, t_prud sur k_prud (le coulage neuf, 20 °C "
          "constante : 17 °C, k = 0,88, J + 12,19) ; aucun symbole « ,p »",
          {"T_prud", "k_prud", "t_prud", "k", "t"} <= set(sy0) and not any(",p" in s for s in sy0)
          and [y["symbole"] for y in sy0["T_prud"]["termes"]] == ["T", "marge"] and sy0["T_prud"]["resultat"] == 17.0
          and sy0["k_prud"]["termes"][0]["symbole"] == "T_prud" and sy0["k_prud"]["termes"][0]["valeur"] == 17.0 and pres(sy0["k_prud"]["resultat"], 0.88, 1e-12)
          and sy0["t_prud"]["termes"][-1]["symbole"] == "k_prud" and pres(sy0["t_prud"]["termes"][-1]["valeur"], 0.88, 1e-12)
          and pres(sy0["t_prud"]["resultat"], 10.728645672973535 / 0.88, 1e-9)
          and sy0["k"]["termes"][0]["symbole"] == "T" and sy0["t"]["termes"][-1]["symbole"] == "k", sorted(sy0))
        e_mes = coulage(m, date="2026-11-02", heure=16, temperature__mode="releve", temperature__releve=releve(T6))
        e_mes["objectifs"] = [objectif(m, "Hors gel", mode="mpa", mpa=5.0, base="fcm")]
        sy1 = {f["symbole"]: f for f in Mm.calculer(e_mes)["objectifs"][0]["formules"]}
        v("9 · … et un jour MESURÉ n'est pas abaissé : pas de ligne T_prud, k_prud se calcule sur T (5 °C : 0,45), t_prud = t",
          "T_prud" not in sy1 and sy1["k_prud"]["termes"][0]["symbole"] == "T" and sy1["k_prud"]["termes"][0]["valeur"] == 5.0
          and pres(sy1["k_prud"]["resultat"], 0.45, 1e-12) and pres(sy1["t_prud"]["resultat"], sy1["t"]["resultat"], 1e-12), sorted(sy1))
        # les invariants
        inv = []
        for nom, e, r in res:
            for i, o in enumerate(r["objectifs"]):
                if o["t"] is not None and (o["prudent"]["t"] is None or o["prudent"]["t"] < o["t"] - 1e-9):
                    if o["prudent"]["t"] is not None:
                        inv.append(f"{nom}/{i} : prudent {o['prudent']['t']} < nominal {o['t']}")
                if o["f_prevu"] is not None and o["f_prevu_prudent"] > o["f_prevu"] + 1e-9:
                    inv.append(f"{nom}/{i} : f prudent > f nominal")
                if o["t"] is not None and not pres(Lm.resistance(Lm.age_equivalent(o["t"], [x["k"] for x in r["profil"]][:r["synthese"]["n_jours"]],
                                                                                 r["temperature"]["k_fin"], r["temps"]["f0"]), o["base"], r["materiau"]["fck"], r["materiau"]["s"]),
                                                   o["cible"], 1e-8):
                    inv.append(f"{nom}/{i} : la résistance à l'âge trouvé n'est pas la cible")
                if (o["mode"] == "pct") != (o["cible"] == r["materiau"]["fck"] * o["pct"] / 100.0 and o["mode"] == "pct"):
                    inv.append(f"{nom}/{i} : cible en %")
            tts = [x["t_T"] for x in r["profil"]]
            if any(b < a - 1e-12 for a, b in zip(tts, tts[1:])) or any(x["t_T_prudent"] > x["t_T"] + 1e-12 for x in r["profil"]):
                inv.append(f"{nom} : t_T décroît, ou le prudent dépasse le nominal")
            if any(b < a - 1e-9 for a, b in zip(r["courbes"]["fck"], r["courbes"]["fck"][1:])) or min(r["courbes"]["fck"]) < 0 \
                    or any(abs(c - k - 8.0) > 1e-9 and k > 0 for c, k in zip(r["courbes"]["fcm"], r["courbes"]["fck"])):
                inv.append(f"{nom} : la courbe f_ck décroît, est négative, ou n'est pas f_cm − 8")
            if not (28.0 <= r["courbes"]["horizon"] <= 90.0 and r["courbes"]["t"][0] == 0.0 and r["courbes"]["t"][-1] == r["courbes"]["horizon"]
                    and r["profil"][-1]["fin"] >= r["courbes"]["horizon"] - 1e-9 and r["profil"][-2]["fin"] < r["courbes"]["horizon"]):
                inv.append(f"{nom} : horizon {r['courbes']['horizon']}")
            if [x["t_T"] for x in r["tableau"]["lignes"]] != tts or any(not pres(x["fck"], max(0.0, x["fcm"] - 8.0), 1e-12) for x in r["tableau"]["lignes"]):
                inv.append(f"{nom} : tableau")
        v("9 · invariants (61 coulages) : la variante prudente n'est jamais plus tôt que la nominale ; la résistance à l'âge trouvé EST la cible ; t_T croît ; "
          "f_ck(t) = max(0 ; f_cm(t) − 8) croît et n'est jamais négative ; l'horizon de 28 à 90 j ; le tableau est le profil", not inv, inv[:3])
        mono = []
        base = Mm.nouvel_element()
        ts = [Mm.calculer(dict(base, temperature=dict(base["temperature"], T=float(T))))["objectifs"][0]["t"] for T in range(-9, 41)]
        if any(b > a + 1e-9 for a, b in zip(ts, ts[1:])):
            mono.append("plus chaud → plus tard")
        tc = [Mm.calculer(dict(base, objectifs=[objectif(m, "X", pct=float(p))]))["objectifs"][0]["t"] for p in range(5, 101, 5)]
        if any(b <= a for a, b in zip(tc, tc[1:])):
            mono.append("cible plus haute → pas plus tard")
        tr = [Mm.calculer(dict(base, durcissement={"mode": "r", "ciment": "N", "beton": "moyen", "r": r_ / 100.0}))["objectifs"][0]["t"] for r_ in range(15, 71, 5)]
        if any(b >= a for a, b in zip(tr, tr[1:])):
            mono.append("r plus grand → pas plus tôt")
        tm = [Mm.calculer(dict(base, marge=float(mg)))["objectifs"][0]["prudent"]["t"] for mg in range(0, 11)]
        if any(b < a for a, b in zip(tm, tm[1:])) or tm[0] != Mm.calculer(base)["objectifs"][0]["t"]:
            mono.append("marge plus grande → prudent pas plus tard (ou marge 0 ≠ nominal)")
        v("9 · monotonies : plus chaud → pas plus tard (de −9 à 40 °C) ; cible plus haute → plus tard ; r plus grand → plus tôt ; marge plus grande → prudent "
          "plus tard, et marge 0 = nominal", not mono, mono)
        r0 = Mm.calculer(Mm.nouvel_element())
        v("9 · la courbe : 0 à l'horizon (28 j), ses points à chaque changement de jour et aux repères t_T = 3 j et 28 j ; les deux courbes f_cm et f_ck ; la "
          "variante prudente en plus quand elle diffère (aucune avec une marge nulle)",
          r0["courbes"]["horizon"] == 28.0 and 3.0 in r0["courbes"]["t"] and 0.5 in r0["courbes"]["t"] and 27.5 in r0["courbes"]["t"]
          and r0["courbes"]["fcm"][-1] == 38.0 and r0["courbes"]["fck"][-1] == 30.0 and r0["courbes"]["fck_prudent"] is not None
          and r0["courbes"]["fck_prudent"][-1] < 30.0 and Mm.calculer(coulage(m, marge=0.0))["courbes"]["fck_prudent"] is None
          and r0["courbes"]["t_3"] == 3.0 and r0["courbes"]["t_28"] == 28.0 and r0["courbes"]["objectifs"][0]["t"] == r0["objectifs"][0]["t"])
        loin = Mm.calculer(coulage(m, temperature__T=2.0))
        v("9 · l'horizon suit le dernier objectif + 20 % (2 °C : atteint à 29,8 j, prudent — à −1 °C — à 39,7 j → horizon 47,7 j), 90 j au plus (−5 °C : 71,5 j → 90 j)",
          pres(loin["courbes"]["horizon"], 1.2 * loin["objectifs"][0]["prudent"]["t"], 1e-12) and dec(loin["courbes"]["horizon"], 47.7, 1) and dec(loin["objectifs"][0]["t"], 29.8, 1)
          and Mm.calculer(coulage(m, temperature__T=-5.0))["courbes"]["horizon"] == 90.0, loin["courbes"]["horizon"])
        long_ = Mm.calculer(coulage(m, temperature__mode="releve", temperature__releve=releve([15.0] * 45)))
        v("9 · un relevé plus long que l'horizon l'allonge (45 jours connus → 45 lignes au tableau)", len(long_["profil"]) == 45 and long_["synthese"]["n_jours"] == 45)
        v("9 · le résumé d'un coulage non daté : son premier objectif, sans taux ; daté : l'objectif le plus chargé et le pire état",
          r0["resume"] == {"date": False, "taux": None, "etat": "date", "objectif": 0, "nom": "Décoffrage"})
    groupe("9 · format", format_)
    return out


# ====================================================================== 1 à 9 : le moteur réel
print("=== 1 à 9. Les contrôles à la main (math seul), les invariants, le format ===")
RES = controles(REEL)
for nom, ok, info in RES:
    chk(nom, ok, info)
chk(f"les contrôles à la main sont nombreux ({len(RES)}) et couvrent les neuf sections", len(RES) >= 110 and {n[:1] for n, _, _ in RES} == set("123456789"))

# ====================================================================== 10. le relevé saisi
print("\n=== 10. Le relevé saisi (collé, lu dans un fichier) ===")


def ecarts(a, b, chemin="", out=None, tol=0.0):
    """Les écarts entre deux structures ; les nombres à ``tol`` près (0 : au bit près), NaN = NaN."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{chemin} : clés {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b)):
            ecarts(a[k], b[k], f"{chemin}.{k}", out, tol)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{chemin} : {len(a)} ≠ {len(b)} éléments")
        for i, (x, y) in enumerate(zip(a, b)):
            ecarts(x, y, f"{chemin}[{i}]", out, tol)
    elif isinstance(a, float) and isinstance(b, float) and not isinstance(a, bool):
        if not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol * max(1.0, abs(a), abs(b))):
            out.append(f"{chemin} : {a!r} ≠ {b!r}")
    elif a != b:
        out.append(f"{chemin} : {a!r} ≠ {b!r}")
    return out


r = RL.lire("12,5\n13.0\n—\n14")
chk("une température par ligne, virgule ou point ; « — » = jour sans donnée ; tout « mesuré » par défaut",
    r["temps"][:2] == [12.5, 13.0] and math.isnan(r["temps"][2]) and r["dates"] is None and not any(r["prev"]))
r = RL.lire("Date;T moy (°C);Source\n02/11/2026;8;mesuré\n2026-11-03;9,5;prévision\n04.11.2026;10;prolongé")
chk("en-tête ignoré, dates JJ/MM/AAAA, AAAA-MM-JJ, JJ.MM.AAAA ; sources : prévision → prévu, prolongé → prévu ET prolongé",
    r["dates"] == ["2026-11-02", "2026-11-03", "2026-11-04"] and r["prev"] == [False, True, True] and r["extra"] == [False, False, True])
r = RL.lire("2026-11-02,8.5,mesuré\n2026-11-03,9.5,p")
chk("un CSV à virgules (nombres à point) se lit aussi", r["temps"] == [8.5, 9.5] and r["prev"] == [False, True])
REFUS_RELEVE = (("ligne illisible au milieu", "12\nabc\n13"), ("deux températures sur une ligne", "12 13"),
                ("dates non consécutives", "2026-11-02;8\n2026-11-04;9"), ("dates dans le désordre", "2026-11-03;8\n2026-11-02;9"),
                ("date sur une ligne sur deux", "2026-11-02;8\n9"), ("source inconnue", "8;demain"),
                ("température invraisemblable", "180"), ("date seule", "2026-11-02"), ("trop de jours", "10\n" * 401),
                ("date impossible", "31/02/2026;8"))
mal = []
for nom, txt in REFUS_RELEVE:
    try:
        RL.lire(txt)
        mal.append(f"{nom} : accepté")
    except ValueError as x:
        if "ligne" not in str(x) and "jours" not in str(x):
            mal.append(f"{nom} : message sans numéro de ligne ({x})")
chk(f"{len(REFUS_RELEVE)} relevés mal formés refusés, chacun avec sa ligne (jamais lus de travers)", not mal, mal)
rr = RL.lire(releve([10, None, 11.25, -3], "mpxm", debut=_dt.date(2026, 10, 12)))
chk("aller-retour : lire(ecrire(r)) = r", ecarts(RL.lire(RL.ecrire(rr)), rr) == [], ecarts(RL.lire(RL.ecrire(rr)), rr))
chk("le tableau « Données météo » de l'écran Streamlit se colle tel quel (Date, T moy, Source, T min h, T max h)",
    RL.lire("Date\tT moy (°C)\tSource\tT min h\tT max h\n2026-11-02\t9.4\tmesuré\t4.1\t13.0\n2026-11-03\t—\tprévision\t\t")["prev"] == [False, True])
syn = RL.synthese(RL.lire(releve([10.0, None, 12.0, 11.0, 9.0, 9.0, 9.0], "mmmppxx")))
chk("ce que le relevé contient, tel qu'il est écrit : 7 jours — 3 mesurés, 2 prévus, 2 prolongés, 1 sans donnée —, T moy 10,0 (des jours qui portent une "
    "température), de 9 à 12 °C ; None sans aucune température",
    syn == {"n_jours": 7, "n_mes": 3, "n_prev": 2, "n_extra": 2, "n_nan": 1, "T_moy": 60.0 / 6.0, "T_min": 9.0, "T_max": 12.0}
    and RL.synthese(RL.lire("—\n—")) is None, syn)

# ====================================================================== 11. la météo du lieu : le plan et la lecture
# Le module du site n'appelle jamais Open-Meteo : le moteur fait le PLAN (meteo.plan), le navigateur fait les appels et
# rend les réponses BRUTES, que le moteur LIT (meteo.lire_*). Ici, pour chaque scénario, l'écran Streamlit va chercher sa
# météo (geocode_open_meteo, meteo_journaliere, meteo_horaire) chez un faux « requests » ; le site fait son plan, et le
# même faux service (tests/faux_open_meteo.py, au format documenté) rend les réponses que le navigateur rendrait. On
# compare les APPELS (adresse, paramètres un à un et dans l'ordre) et ce qui est LU, refus compris. (La lecture de la
# météo est GARDÉE de l'écran Streamlit ; seul le CALCUL de maturité ne s'y compare plus.)
try:
    logging.disable(logging.WARNING)
    import modules.age_beton as AB  # noqa: E402
except ImportError as e:  # pragma: no cover — la parité de la météo EXIGE le module
    AB = None
    chk("streamlit, matplotlib, pandas et modules/age_beton.py se chargent (indispensables au § 11)", False, e)
J = lambda n: (AUJ_M + _dt.timedelta(days=n)).isoformat()  # noqa: E731 — un jour relatif au jour des récupérations
TOL_METEO = 1e-12          # la moyenne de la prolongation : somme / nombre, et non plus numpy.mean


def date_figee(a: _dt.date):
    """Un module datetime de substitution dont date.today() est figé (l'écran partage les dates selon ce jour)."""
    class DateFixe(_dt.date):
        @classmethod
        def today(cls):
            return cls(a.year, a.month, a.day)
    return types.SimpleNamespace(date=DateFixe, timedelta=_dt.timedelta, datetime=_dt.datetime)


def _ecran_meteo(faux_requests, f):
    """Une fonction météo de l'écran, appelée avec le faux « requests » ; les caches vidés."""
    vrai = (AB._dt, AB._req if AB._HAS_REQUESTS else None, AB._HAS_REQUESTS)
    AB._req, AB._HAS_REQUESTS = faux_requests, True
    for g in (AB.meteo_journaliere, AB.meteo_horaire, AB.geocode_open_meteo):
        g.clear()
    try:
        return f()
    finally:
        AB._dt, AB._HAS_REQUESTS = vrai[0], vrai[2]
        if vrai[1] is not None:
            AB._req = vrai[1]


if AB is not None:
    print("\n=== 11. La météo du lieu : le plan des appels et la lecture des réponses, contre l'écran (faux requests) ===")
    chk("les adresses d'Open-Meteo et les constantes sont celles de l'écran (archive, prévisions, géocodage ; retard 6 j, horizon 16 j)",
        (MT.ADRESSES["archive"], MT.ADRESSES["prevision"], MT.RETARD_ARCHIVE_J, MT.HORIZON_PREVISION_J)
        == (AB.URL_ARCHIVE, AB.URL_FORECAST, AB.RETARD_ARCHIVE_J, AB.HORIZON_PREVISION_J)
        and 'geocoding-api.open-meteo.com/v1/search"' in lire("modules", "age_beton.py").replace(MT.ADRESSES["geocodage"], "geocoding-api.open-meteo.com/v1/search"))
    # 11.1 les plages : _plages_endpoints, sur une grille de dates (passé lointain, à cheval, récent, futur, au-delà, bornes)
    mal, n = [], 0
    for auj in (_dt.date(2026, 11, 7), _dt.date(2026, 2, 28), _dt.date(2024, 12, 31)):
        for a in range(-130, 30, 3):
            for b in (0, 1, 2, 5, 6, 7, 9, 15, 16, 17, 22, 40, 90):
                d0 = auj + _dt.timedelta(days=a)
                d1 = d0 + _dt.timedelta(days=b)
                n += 1
                if MT.plages(d0, d1, auj) != AB._plages_endpoints(d0, d1, auj):
                    mal.append((auj, d0, d1))
    for auj in (_dt.date(2026, 11, 7),):                       # les bornes exactes : J−6 (recente), J+16 (horizon)
        for d0, d1 in ((J(-6), J(-6)), (J(-7), J(-7)), (J(-7), J(-6)), (J(16), J(16)), (J(17), J(17)), (J(-6), J(16)), (J(-30), J(-7))):
            n += 1
            if MT.plages(_dt.date.fromisoformat(d0), _dt.date.fromisoformat(d1), auj) != AB._plages_endpoints(_dt.date.fromisoformat(d0), _dt.date.fromisoformat(d1), auj):
                mal.append((d0, d1))
    chk(f"plages : le partage archive / prévisions / prolongation est celui de l'écran ({n} périodes, bornes J−6 et J+16 comprises)", not mal, mal[:3])

    def scenario(lieu, debut, fin, auj, nulls=(), nulls_h=()):
        """L'écran et le site sur un même scénario : (appels, lu) de chaque côté, et les refus."""
        d0, d1 = _dt.date.fromisoformat(debut), _dt.date.fromisoformat(fin)
        # l'écran : un faux requests sur le faux service
        srv_e = FO.FauxService(auj, nulls, nulls_h)
        ecran_ = {"appels": None, "geo": None, "j": None, "h": None, "refus": None}

        def f():
            AB._dt = date_figee(auj)
            ecran_["geo"] = AB.geocode_open_meteo(lieu)
            if ecran_["geo"] is None:
                return
            lat, lon, _ = ecran_["geo"]
            try:
                ecran_["j"] = AB.meteo_journaliere(lat, lon, debut, fin)
            except RuntimeError as x:
                ecran_["refus"] = str(x)
            ecran_["h"] = AB.meteo_horaire(lat, lon, debut, fin)
        _ecran_meteo(FO.FauxRequests(srv_e), f)
        ecran_["appels"] = srv_e.appels
        # le site : le plan, ce que le navigateur rend, la lecture
        srv_s = FO.FauxService(auj, nulls, nulls_h)
        site = {"appels": None, "geo": None, "j": None, "h": None, "refus": None}
        try:
            p = MT.plan(lieu, d0, d1, auj)
        except ValueError as x:
            site["refus"] = str(x)
            p = None
            # le site ne demande RIEN quand le plan est impossible ; l'écran, lui, a géocodé avant de refuser : seul le
            # géocodage est attendu dans son journal
            code, geo = srv_s.reponse(MT.ADRESSES["geocodage"], {"name": lieu, **MT.GEOCODAGE})
            site["geo"] = MT.lire_geocodage(geo)
        if p is not None:
            v = FO.recuperer(p, srv_s)
            site["geo"] = MT.lire_geocodage(v["geocodage"])
            if site["geo"] is not None:
                try:
                    site["j"] = MT.lire_journalier(p, v["reponses"])
                except ValueError as x:
                    site["refus"] = str(x)
                pos = {"latitude": site["geo"]["latitude"], "longitude": site["geo"]["longitude"]}
                ph = MT.plan(None, d0, d1, auj, horaire=True, position=pos)
                site["h"] = MT.lire_horaire(ph, FO.recuperer(ph, srv_s)["reponses"])
        site["appels"] = srv_s.appels
        return ecran_, site, p

    SCEN = [("Bruxelles, Belgique", J(-60), J(-31), AUJ_M, (), ()), ("Bruxelles, Belgique", J(-20), J(0), AUJ_M, (J(-12),), ()),
            ("Namur", J(-3), J(10), AUJ_M, (), (J(2),)), ("Namur", J(3), J(14), AUJ_M, (), ()),
            ("Bruxelles", J(-4), J(30), AUJ_M, (J(13), J(16)), ()), ("Vaduz", J(-6), J(16), AUJ_M, (), ()),
            ("Bruxelles", J(-7), J(-7), AUJ_M, (), ()), ("Bruxelles", J(16), J(20), AUJ_M, (), ()),
            ("Liège", J(-10), J(0), AUJ_M, (), ()),                                          # lieu introuvable (le faux ne le connaît pas)
            ("Bruxelles", J(17), J(25), AUJ_M, (), ()),                                       # entièrement au-delà de l'horizon : refus
            ("Bruxelles", J(10), J(30), AUJ_M, tuple(J(k) for k in range(10, 17)), ()),       # rien d'exploitable pour prolonger : refus
            ("Namur", "2026-02-20", "2026-03-20", _dt.date(2026, 3, 2), (), ("2026-02-25",))]
    rng = random.Random(20261107)
    for k in range(18):                                                                      # des tirages (graine fixe)
        auj = AUJ_M + _dt.timedelta(days=rng.randint(-400, 0))
        a = rng.randint(-150, 20)
        nul = tuple((auj + _dt.timedelta(days=a + rng.randint(0, 30))).isoformat() for _ in range(rng.randint(0, 3)))
        SCEN.append((rng.choice(["Bruxelles, Belgique", "Namur", "Vaduz"]), (auj + _dt.timedelta(days=a)).isoformat(),
                     (auj + _dt.timedelta(days=a + rng.randint(0, 60))).isoformat(), auj, nul, nul[:1]))

    def comparer_meteo(sc):
        lieu, debut, fin, auj, nulls, nulls_h = sc
        e_, s_, p = scenario(lieu, debut, fin, auj, nulls, nulls_h)
        out = []
        # les appels : mêmes adresses, mêmes paramètres, dans le même ordre (le site ne demande rien si le plan est impossible)
        att = [(u, list(q.items())) for u, q in e_["appels"]]
        lu = [(u, list(q.items())) for u, q in s_["appels"]]
        if p is not None and att != lu:
            out.append(f"appels : écran {att[:2]} … ≠ site {lu[:2]} …")
        if p is None and (lu != att[:1] or not s_["refus"]):
            out.append(f"plan impossible : le site devait refuser sans autre appel que le géocodage ({lu})")
        # le lieu
        if (e_["geo"] is None) != (s_["geo"] is None) or (e_["geo"] and (e_["geo"][0], e_["geo"][1], e_["geo"][2])
                                                         != (s_["geo"]["latitude"], s_["geo"]["longitude"], s_["geo"]["label"])):
            out.append(f"géocodage : {e_['geo']} ≠ {s_['geo']}")
        # le journalier (à 1e-12 près : la moyenne de la prolongation), et les refus des deux côtés
        if (e_["refus"] is None) != (s_["refus"] is None):
            out.append(f"refus : écran {e_['refus']!r} / site {s_['refus']!r}")
        elif e_["j"] is not None or s_["j"] is not None:
            out += ecarts(s_["j"], e_["j"], "journalier", tol=TOL_METEO)
        # le détail horaire : une période sans jour servi rend, à l'écran, une série vide (sans appel) ; le site refuse le
        # plan (le bouton est grisé, le message dit pourquoi) — les deux ne montrent rien
        if s_["h"] is None and e_["h"] == {"h_times": [], "h_temps": []}:
            pass
        elif e_["h"] is not None or s_["h"] is not None:
            out += ecarts(s_["h"], e_["h"], "horaire")
        return out, e_, s_

    mal, vus = [], set()
    for sc in SCEN:
        d, e_, s_ = comparer_meteo(sc)
        if d:
            mal.append((sc[:3], d[:2]))
        j = e_["j"]
        vus |= {"lieu introuvable" for _ in [0] if e_["geo"] is None} | {"refus de l'écran" for _ in [0] if e_["refus"]}
        if j:
            vus |= {"jours prolongés" for _ in [0] if j["n_extra"]} | {"jours prévus" for _ in [0] if any(j["d_prev"])}
            vus |= {"jour sans donnée" for _ in [0] if any(math.isnan(t) for t in j["d_temps"])}
            vus |= {"archive et prévisions" for _ in [0] if len([u for u, _ in e_["appels"] if "daily" in str(_)]) >= 2}
        if e_["h"] and any(math.isnan(t) for t in e_["h"]["h_temps"]):
            vus.add("heure sans valeur")
    chk(f"{len(SCEN)} scénarios : mêmes appels (adresses, paramètres un à un et dans l'ordre), même lieu trouvé, mêmes températures "
        "journalières (à 1e-12 près) et horaires, mêmes refus — le site lit les réponses comme l'écran", not mal, mal[:2])
    manque = {"lieu introuvable", "refus de l'écran", "jours prolongés", "jours prévus", "jour sans donnée", "archive et prévisions", "heure sans valeur"} - vus
    chk("les scénarios atteignent : lieu introuvable, refus, jours prolongés, prévus, sans donnée, archive et prévisions, heure sans valeur",
        not manque, manque)
    # 11.2 la chaîne : les températures lues → le relevé de la note → relu, ce sont celles que l'écran a reçues
    mal = []
    for sc in SCEN:
        e_, s_, p = scenario(*sc)
        if s_["j"] and not s_["refus"]:
            rel = RL.lire(MT.releve(s_["j"]))
            att = {"dates": e_["j"]["d_dates"], "temps": e_["j"]["d_temps"], "prev": e_["j"]["d_prev"], "extra": e_["j"]["d_extra"]}
            if ecarts({k: rel[k] for k in att}, att, tol=TOL_METEO):
                mal.append(sc[:3])
    chk("le relevé écrit par la météo se relit en exactement les jours, températures (à 1e-12) et sources que l'écran a reçus", not mal, mal[:2])
    # 11.3 les refus propres au site : réponse mal formée, trouée, en nombre faux — un message, jamais une lecture de travers
    p = MT.plan("Bruxelles", J(-10), J(0), AUJ_M)
    bon = FO.recuperer(p, FO.FauxService(AUJ_M))["reponses"]
    abimes = []
    r1 = copy.deepcopy(bon); r1[0]["daily"]["temperature_2m_mean"][2] = "12,5"
    r2 = copy.deepcopy(bon); r2[0]["daily"]["time"].pop(); r2[0]["daily"]["temperature_2m_mean"].pop()
    r3 = copy.deepcopy(bon); r3[1]["daily"]["time"][0] = "07/11/2026"
    r4 = copy.deepcopy(bon); r4[0]["daily"]["temperature_2m_mean"].pop()
    r5 = copy.deepcopy(bon); r5[0]["daily"]["temperature_2m_mean"][0] = True
    for nom, reps, etape, motif in (("une température en texte", r1, "lire", "« 12,5 » n'est pas une température"),
                                    ("un jour manquant entre l'archive et les prévisions", r2, "releve", f"le {_dt.date.fromisoformat(J(-7)):%d/%m/%Y} manque"),
                                    ("une date au mauvais format", r3, "lire", "instant « 07/11/2026 »"),
                                    ("des listes de longueurs différentes", r4, "lire", "ne se répondent pas"),
                                    ("un booléen pour température", r5, "lire", "« True » n'est pas une température"),
                                    ("une réponse de moins", bon[:1], "lire", "devait rendre 2 réponse(s), il en a rendu 1"),
                                    ("pas de réponses", None, "lire", "il en a rendu aucune")):
        try:
            j = MT.lire_journalier(p, reps)
            if etape == "releve":
                MT.releve(j)
            abimes.append(f"{nom} : accepté")
        except ValueError as x:
            if motif not in str(x):
                abimes.append(f"{nom} : message « {x} » (attendu : « {motif} »)")
    chk("7 réponses mal formées (texte, jour manquant, date, longueurs, booléen, nombre de réponses) sont refusées, chacune avec le message qui "
        "dit ce qui ne va pas (le jour qui manque, la valeur illisible…)", not abimes, abimes)
    # 11.3 bis : une réponse dont les JOURS ne sont pas ceux demandés — le relevé commencerait un autre jour que le coulage,
    # et tous les âges glisseraient sans un mot (une archive décalée d'un jour, la veille du coulage — un jour chaud AVANT le
    # coulage compterait : non conservatif —, un détail horaire d'une autre année ; une date qui n'existe pas)
    d0 = _dt.date.fromisoformat(p["debut"])
    r6 = copy.deepcopy(bon); r6[0]["daily"]["time"].pop(0); r6[0]["daily"]["temperature_2m_mean"].pop(0)
    r7 = copy.deepcopy(bon); r7[0]["daily"]["time"].insert(0, (d0 - _dt.timedelta(days=1)).isoformat()); r7[0]["daily"]["temperature_2m_mean"].insert(0, 30.0)
    r8 = copy.deepcopy(bon); r8[0]["daily"]["time"][0] = "2026-02-30"
    r9 = copy.deepcopy(bon); r9[1]["daily"]["time"] = r9[1]["daily"]["time"][::-1]; r9[1]["daily"]["temperature_2m_mean"] = r9[1]["daily"]["temperature_2m_mean"][::-1]
    ph = MT.plan(None, d0, _dt.date.fromisoformat(p["fin"]), AUJ_M, horaire=True, position={"latitude": 50.85045, "longitude": 4.34878})
    bon_h = FO.recuperer(ph, FO.FauxService(AUJ_M))["reponses"]
    h1 = copy.deepcopy(bon_h); h1[0]["hourly"]["time"] = [x.replace("2026-", "2025-", 1) for x in h1[0]["hourly"]["time"]]
    h2 = copy.deepcopy(bon_h); h2[0]["hourly"]["time"][5] = h2[0]["hourly"]["time"][5][:11] + "25:00"
    abimes = []
    for nom, lire_, reps, motif in (
            ("l'archive commence un jour plus tard", MT.lire_journalier, r6, f"le {d0:%d/%m/%Y} manque"),
            ("l'archive rend la veille du coulage", MT.lire_journalier, r7, f"hors des dates demandées (archive : le {d0 - _dt.timedelta(days=1):%d/%m/%Y}"),
            ("un 30 février", MT.lire_journalier, r8, "instant impossible « 2026-02-30 »"),
            ("des prévisions dans le désordre", MT.lire_journalier, r9, "les jours ne se suivent pas"),
            ("un détail horaire d'une autre année", MT.lire_horaire, h1, "hors des dates demandées"),
            ("une 25e heure", MT.lire_horaire, h2, "instant impossible")):
        try:
            lire_(ph if lire_ is MT.lire_horaire else p, reps)
            abimes.append(f"{nom} : accepté")
        except ValueError as x:
            if motif not in str(x):
                abimes.append(f"{nom} : message « {x} » (attendu : « {motif} »)")
    chk("6 réponses dont les jours ne sont pas ceux demandés (archive décalée d'un jour, veille du coulage, 30 février, jours dans le "
        "désordre, horaire d'une autre année, 25e heure) sont refusées en français, chacune avec le jour en cause — le relevé commence "
        "toujours au coulage", not abimes, abimes)
    chk("une réponse conforme (les jours demandés, exactement) est lue : le relevé commence au coulage et va jour par jour jusqu'à la fin",
        MT.lire_journalier(p, bon)["d_dates"][0] == p["debut"] and MT.lire_journalier(p, bon)["d_dates"][-1] == p["fin"]
        and len(MT.lire_horaire(ph, bon_h)["h_times"]) == 24 * len(MT.lire_journalier(p, bon)["d_dates"]))
    for mal_p, attendu in (((None, J(0), J(5)), "lieu"), (("Namur", J(5), J(0)), "précède"), (("Namur", J(-500), J(0)), "400"),
                           (("Namur", J(17), J(30)), "horizon")):
        try:
            MT.plan(*mal_p, AUJ_M)
            chk(f"plan impossible ({attendu}) : refusé", False, "accepté")
        except ValueError as x:
            chk(f"plan impossible ({attendu}) : refusé, avec un message en français", attendu in str(x), str(x))
    # 11.4 ce que la note garde : réduit, relu strictement, et la récupération se relit à l'identique hors ligne
    v = FO.recuperer(p, FO.FauxService(AUJ_M))
    rec = MT.enregistrement(p, v["geocodage"], v["reponses"], "2026-11-07T10:05")
    chk("la note garde la récupération réduite à ce qui est lu (instants et températures), et la relit à l'identique",
        set(rec["reponses"][0]) == {"daily"} and set(rec["reponses"][0]["daily"]) == {"time", "temperature_2m_mean"}
        and MT.normaliser(json.loads(json.dumps(rec))) == rec
        and ecarts(MT.lire_journalier(MT.plan_enregistre(rec), rec["reponses"]), MT.lire_journalier(p, v["reponses"])) == [])
    refus = []
    for nom, f_ in (("clé inconnue", lambda r: r.update(x=1)), ("lieu trouvé sans latitude", lambda r: r["geocodage"].pop("latitude")),
                    ("latitude 120°", lambda r: r["geocodage"].update(latitude=120)), ("jour de récupération illisible", lambda r: r.update(aujourdhui="hier")),
                    ("une réponse de trop", lambda r: r["reponses"].append(r["reponses"][0])),
                    ("une température en texte", lambda r: r["reponses"][0]["daily"]["temperature_2m_mean"].__setitem__(0, "x")),
                    ("heure de récupération illisible", lambda r: r.update(recupere="07/11/2026")),
                    ("détail horaire sans réponses", lambda r: r.update(horaire={"aujourdhui": "2026-11-07", "recupere": "2026-11-07T10:06", "reponses": []}))):
        r_ = copy.deepcopy(rec)
        f_(r_)
        try:
            MT.normaliser(r_)
            refus.append(f"{nom} : accepté")
        except ValueError:
            pass
    chk("8 récupérations gardées abîmées (un .stab venu d'ailleurs) sont refusées par la relecture stricte", not refus, refus)
    rec_h = dict(copy.deepcopy(rec), horaire=MT.enregistrement_horaire(ph, bon_h, "2026-11-07T10:06"))
    chk("une note qui garde le détail horaire tel que reçu se relit à l'identique", MT.normaliser(json.loads(json.dumps(rec_h))) == rec_h)
    refus = []
    for nom, f_ in (("détail horaire d'une autre année", lambda r: r["horaire"]["reponses"][0]["hourly"].update(time=h1[0]["hourly"]["time"])),
                    ("journalier qui commence un jour plus tard", lambda r: [r["reponses"][0]["daily"][k].pop(0) for k in ("time", "temperature_2m_mean")])):
        r_ = copy.deepcopy(rec_h)
        f_(r_)
        try:
            MT.normaliser(r_)
            refus.append(f"{nom} : accepté")
        except ValueError as x:
            if "temperature/meteo" not in str(x):
                refus.append(f"{nom} : message « {x} »")
    chk("une note dont la récupération gardée porte des jours non demandés (horaire d'une autre année, journalier décalé) est refusée à la relecture",
        not refus, refus)
    # 11.4 bis la minimale horaire comptée dès une heure : ce qui sert l'alerte de gel du moteur
    h = MT.lire_horaire(ph, bon_h)
    jour0 = p["debut"]
    mins = MT.minimales_apres(h)
    apres = MT.minimales_apres(h, f"{jour0}T14:00")
    chk("minimales_apres : la plus basse valeur horaire de chaque jour (celle des extrêmes) ; comptée dès 14 h le premier jour, les heures d'avant n'y entrent pas",
        all(mins[d] == MT.extremes_horaires(h)[d][0] for d in mins) and len(mins) == 11
        and apres[jour0] == min(FO.t_heure(d0, hh) for hh in range(14, 24)) and apres[jour0] > mins[jour0]
        and all(apres[d] == mins[d] for d in mins if d != jour0), (mins[jour0], apres[jour0]))

    # 11.5 mutations : un défaut plausible du plan ou de la lecture, glissé dans meteo.py, fait rougir la parité
    def _prevision_large(p_, reps):
        j_ = MT_LIRE(p_, reps)
        auj_ = _dt.date.fromisoformat(p_["aujourdhui"])
        return dict(j_, d_prev=[x or _dt.date.fromisoformat(d) >= auj_ for d, x in zip(j_["d_dates"], j_["d_prev"])])

    def _geo_sans_region(rep):
        g = MT_GEO(rep)
        return g and dict(g, label=", ".join(x for x in (g["nom"], g["pays"]) if x))
    MT_LIRE, MT_GEO = MT.lire_journalier, MT.lire_geocodage
    MUT_METEO = [("archive fiable jusqu'à J−5 au lieu de J−6", [(MT, "RETARD_ARCHIVE_J", 5)]),
                 ("horizon des prévisions 15 j au lieu de 16", [(MT, "HORIZON_PREVISION_J", 15)]),
                 ("prolongation sur les 4 derniers jours au lieu de 5", [(MT, "JOURS_PROLONGATION", 4)]),
                 ("fuseau UTC au lieu de « auto »", [(MT, "FUSEAU", "UTC")]),
                 ("géocodage : deux résultats demandés", [(MT, "GEOCODAGE", {"count": 2, "language": "fr"})]),
                 ("variable journalière : le maximum au lieu de la moyenne", [(MT, "JOURNALIER", {"daily": "temperature_2m_max"})]),
                 ("prévision dès le jour de la récupération (≥ au lieu de >)", [(MT, "lire_journalier", _prevision_large)]),
                 ("nom du lieu sans sa région", [(MT, "lire_geocodage", _geo_sans_region)])]
    vues = []
    for nom, patchs in MUT_METEO:
        anciens = [(mod, att, getattr(mod, att)) for mod, att, _ in patchs]
        for mod, att, val in patchs:
            setattr(mod, att, val)
        trouve = None
        try:
            for sc in SCEN:
                try:
                    d = comparer_meteo(sc)[0]
                except Exception as x:  # noqa: BLE001 — une mutation qui fait planter est vue aussi
                    d = [f"{type(x).__name__} : {x}"]
                if d:
                    trouve = (sc[:3], d[0])
                    break
        finally:
            for mod, att, val in anciens:
                setattr(mod, att, val)
        vues.append(trouve is not None)
        chk(f"mutation météo « {nom} » : vue par la parité" + (f" ({str(trouve[1])[:70]})" if trouve else ""), trouve is not None)
    chk(f"les {len(MUT_METEO)} mutations de la météo rougissent toutes ; après elles, la parité est intacte",
        all(vues) and not comparer_meteo(SCEN[1])[0])

# ====================================================================== 12. la pureté
print("\n=== 12. Pureté : le moteur se charge et calcule SANS numpy (refusé à l'import), ni streamlit, ni matplotlib ===")
non_std = []
for dossier, _, fichiers in os.walk(os.path.join(RACINE, "beton", "maturite")):
    for f in fichiers:
        if f.endswith(".py"):
            arbre = ast.parse(lire(os.path.relpath(os.path.join(dossier, f), RACINE)))
            for n in ast.walk(arbre):
                noms = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                    [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
                non_std += [f"{f}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names | {"beton"}]
chk("aucun fichier de beton/maturite n'importe autre chose que la bibliothèque standard (lecture des sources) : plus de numpy",
    not non_std and sorted(f for f in os.listdir(os.path.join(RACINE, "beton", "maturite")) if f.endswith(".py"))
    == ["__init__.py", "entree.py", "lois.py", "meteo.py", "moteur.py", "releve.py"], non_std)
SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("numpy", "streamlit", "pandas", "reportlab", "matplotlib", "scipy", "requests", "pymupdf", "fitz", "PIL")
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("chargement refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
refus = []
for nom in ("numpy", "matplotlib"):
    try:
        __import__(nom)
    except ImportError:
        refus.append(nom)
sys.path.insert(0, sys.argv[1])
from beton.maturite import moteur as M
coulages = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.calculer(e) for e in coulages],
          "cands": M.candidats_classes(coulages[1])}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"beton"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
COULAGES = [M.nouvel_element(),
            coulage(REEL, date="2026-11-02", heure=16, temperature__mode="releve", temperature__releve=releve([12.0, 8.0, 5.0, None, -3.0, 15.0], "mmmppx"),
                    essai__actif=True, essai__caler=True, loi__mode="ec2"),
            coulage_meteo(REEL, "Namur", J(-3), J(25), horaire=True, heure=7, loi__mode="perso", loi__Ea_kJ=45.0),
            coulage(REEL, temperature__T=-8.0, durcissement__mode="ciment", durcissement__ciment="S")]
COULAGES[1]["objectifs"] = [objectif(REEL, "Hors gel", mode="mpa", mpa=5.0, base="fcm"), objectif(REEL, "Décoffrage", pct=70.0, age=9.0)]
attendu = {"res": json.loads(json.dumps([M.calculer(e) for e in COULAGES])), "cands": json.loads(json.dumps(M.candidats_classes(COULAGES[1])))}
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(COULAGES), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans numpy", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    vers = ".".join(map(str, out["version"]))
    chk(f"Python {vers} : numpy, streamlit, matplotlib, pandas, requests… REFUSÉS à l'import, et le moteur se charge et calcule quand même "
        f"({len(out['res'])} coulages : constante, relevé, météo avec détail horaire, gel)", out["refus"] == ["numpy", "matplotlib"] and not out["interdits"],
        (out["refus"], out["interdits"]))
    chk(f"Python {vers} : le moteur n'a chargé que la bibliothèque standard", not out["tiers"], out["tiers"])
    diff = ecarts({"res": out["res"], "cands": out["cands"]}, attendu, tol=1e-12)
    chk(f"Python {vers} : résultats identiques à ceux de ce processus", not diff, diff[:3])
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")

# ====================================================================== 13. les mutations
print("\n=== 13. Mutations : le moteur modifié d'UN défaut plausible est rejeté par les contrôles 1 à 9 ===")
PAQUET = sys.modules["beton.maturite"]
NOMS_MODULES = ("lois", "releve", "meteo", "entree", "moteur")       # dans l'ordre de leurs dépendances
ALIAS = {"lois": "L", "releve": "RL", "meteo": "MT", "entree": "EN", "moteur": "M"}


def charger_mutant(mutations):
    """Les cinq modules du moteur rechargés depuis leurs SOURCES après des remplacements [(module, avant, après)] ; chaque
    « avant » doit figurer UNE fois dans la source (sinon la mutation est périmée). Rend le jeu de modules mutant."""
    sauve = {n: sys.modules[f"beton.maturite.{n}"] for n in NOMS_MODULES}
    sauve_attr = {n: getattr(PAQUET, n) for n in NOMS_MODULES}
    mods = {}
    try:
        for n in NOMS_MODULES:
            chemin = os.path.join(RACINE, "beton", "maturite", n + ".py")
            src = lire("beton", "maturite", n + ".py")
            for module, avant, apres in mutations:
                if module == n:
                    if src.count(avant) != 1:
                        raise AssertionError(f"mutation périmée : « {avant} » ({src.count(avant)} fois dans {n}.py)")
                    src = src.replace(avant, apres)
            mod = types.ModuleType(f"beton.maturite.{n}")
            mod.__package__, mod.__file__ = "beton.maturite", chemin
            sys.modules[mod.__name__] = mod
            setattr(PAQUET, n, mod)
            exec(compile(src, chemin, "exec"), mod.__dict__)
            mods[n] = mod
    finally:
        sys.modules.update({f"beton.maturite.{n}": sauve[n] for n in NOMS_MODULES})
        for n in NOMS_MODULES:
            setattr(PAQUET, n, sauve_attr[n])
    return types.SimpleNamespace(**{ALIAS[n]: mods[n] for n in NOMS_MODULES})


def rejets(mutations):
    """Les contrôles 1 à 9 que le moteur mutant fait rougir (leurs noms)."""
    return [nom for nom, ok, _ in controles(charger_mutant(mutations)) if not ok]


MUTATIONS = [
    ("la référence 293 K changée (293,15 : l'ancien moteur)", [("lois", "T_REFERENCE_C = 20.0 ", "T_REFERENCE_C = 20.15 ")]),
    ("β_cc : 27 au lieu de 28 sous la racine", [("lois", "AGE_REFERENCE_J = 28.0 ", "AGE_REFERENCE_J = 27.0 ")]),
    ("f_ck(t) : le « − 8 » oublié", [("lois", "    return max(0.0, fcm_t(t_T, fck, s) - DELTA_FCM)", "    return max(0.0, fcm_t(t_T, fck, s))")]),
    ("f_cm = f_ck + 7", [("lois", "DELTA_FCM = 8.0 ", "DELTA_FCM = 7.0 ")]),
    ("f_ck(t) laissée négative au très jeune âge", [("lois", "    return max(0.0, fcm_t(t_T, fck, s) - DELTA_FCM)", "    return fcm_t(t_T, fck, s) - DELTA_FCM")]),
    ("le gain d'après 28 jours compté (β_cc non borné à 1)", [("lois", "    return min(beta_cc(t_T, s), 1.0)", "    return beta_cc(t_T, s)")]),
    ("le premier jour compté en entier, quelle que soit l'heure", [("lois", "    return (HEURES_PAR_JOUR - heure) / HEURES_PAR_JOUR", "    return 1.0")]),
    ("un point de la table belge changé (0,48 à 5 °C : la valeur de (B.10))", [("lois", "(5.0, 0.45)", "(5.0, 0.48)")]),
    ("la table belge : l'interpolation remplacée par un palier", [("lois", "            return ka + (kb - ka) * (T - Ta) / (Tb - Ta)", "            return ka")]),
    ("la table belge : le plafond à 1 retiré (la droite continue au-dessus de 20 °C)",
     [("lois", "    if T >= K_NBN[-1][0]:\n        return K_NBN[-1][1]\n    for (Ta, ka), (Tb, kb) in zip(K_NBN, K_NBN[1:]):",
       "    if T >= K_NBN[-1][0]:\n        return K_NBN[-1][1] + 0.04 * (T - K_NBN[-1][0])\n    for (Ta, ka), (Tb, kb) in zip(K_NBN, K_NBN[1:]):")]),
    ("la table belge : pas de prolongement sous −5 °C (k = 0,15 jusqu'à −10 °C)", [("lois", "K_NBN = ((T_NUL_NBN, 0.0),) + K_NBN_NORME", "K_NBN = ((T_NUL_NBN, 0.15),) + K_NBN_NORME")]),
    ("(B.10) laissée courir sous 0 °C", [("lois", "T_MIN_EC2, T_MAX_EC2 = 0.0, 80.0", "T_MIN_EC2, T_MAX_EC2 = -100.0, 80.0")]),
    ("(B.10) : E_a/R = 4500 K", [("lois", "EA_SUR_R_EC2 = 4000.0 ", "EA_SUR_R_EC2 = 4500.0 ")]),
    ("Arrhenius : R = 8,0", [("lois", "R_GAZ = 8.314 ", "R_GAZ = 8.0 ")]),
    ("la marge ignorée sur les jours prévus du relevé",
     [("moteur", "        c[\"T_prudent\"] = None if T is None else (T - marge if c[\"source\"] in NON_MESURES else T)", "        c[\"T_prudent\"] = T")]),
    ("la marge ignorée au-delà du relevé et à température constante", [("moteur", "    out[\"T_fin_prudent\"] = out[\"T_fin\"] - marge", "    out[\"T_fin_prudent\"] = out[\"T_fin\"]")]),
    ("la marge appliquée aux jours MESURÉS aussi", [("moteur", "(T - marge if c[\"source\"] in NON_MESURES else T)", "(T - marge)")]),
    ("la prolongation à 20 °C au lieu de la moyenne des cinq derniers jours", [("lois", "    return sum(v) / len(v)", "    return 20.0")]),
    ("la prolongation sur TOUT le relevé au lieu des cinq derniers jours", [("lois", "JOURS_PROLONGATION = 5 ", "JOURS_PROLONGATION = 500 ")]),
    ("la date tronquée au jour", [("moteur", "    n = math.ceil(heures - 1e-6) if plafond else math.floor(heures + 0.5)", "    n = 24 * math.floor(t)")]),
    ("l'instant arrondi à l'heure INFÉRIEURE", [("moteur", "    n = math.ceil(heures - 1e-6) if plafond else", "    n = math.floor(heures) if plafond else")]),
    ("le cube non converti en cylindre", [("moteur", "    rapport = L.rapport_cube(e[\"classe\"]) if cube else 1.0", "    rapport = 1.0")]),
    ("la base moyenne et la base caractéristique confondues (β requis)", [("lois", "        return (cible + DELTA_FCM) / fcm", "        return cible / fcm")]),
    ("la base moyenne et la base caractéristique confondues (résistance à un âge)",
     [("lois", "    if base == \"fck\":\n        return fck_t(t_T, fck, s)", "    if base == \"fck\":\n        return fcm_t(t_T, fck, s)")]),
    ("s pris à la borne HAUTE de la classe du bon de livraison",
     [("lois", "        return {\"mode\": mode, \"cle\": b, \"s\": s_de_r(bas), \"r\": bas,", "        return {\"mode\": mode, \"cle\": b, \"s\": s_de_r(haut or 0.70), \"r\": haut or 0.70,")]),
    ("un pourcentage de f_ck comparé à la moyenne (base non forcée)", [("entree", "    if o[\"mode\"] == \"pct\":\n        o[\"base\"] = \"fck\"\n", "")]),
    ("un jour sans donnée compté comme un jour à 20 °C", [("moteur", "        c[\"k\"] = 0.0 if T is None else L.k_T(T, loi)", "        c[\"k\"] = 1.0 if T is None else L.k_T(T, loi)")]),
    ("les jours du relevé d'AVANT le bétonnage comptés", [("moteur", "            for i in range(out[\"n_avant\"], n):", "            for i in range(0, n):")]),
    ("le gel comparé à 5 MPa de résistance CARACTÉRISTIQUE", [("moteur", "    tT5 = L.t_T_requis(SEUIL_GEL_MPA, \"fcm\", fck, s)", "    tT5 = L.t_T_requis(SEUIL_GEL_MPA, \"fck\", fck, s)")]),
    ("le gel : une moyenne de 0 °C n'est plus un jour de gel", [("moteur", "(T < T_GEL_C if base == \"minimale\" else T <= T_GEL_C)", "(T < T_GEL_C)")]),
    ("le gel lu sur la moyenne même quand la minimale horaire est connue",
     [("moteur", "    if jr[\"connu\"] and p[\"minimales\"] and date in p[\"minimales\"]:", "    if False and jr[\"connu\"] and p[\"minimales\"] and date in p[\"minimales\"]:")]),
    ("le temps froid : 48 heures au lieu de 72", [("moteur", "HEURES_FROID = 72.0 ", "HEURES_FROID = 48.0 ")]),
    ("le calage hors de [0,10 ; 0,80] accepté", [("moteur", "        if L.S_CALE_MIN <= sc <= L.S_CALE_MAX:", "        if True:")]),
    ("l'essai comparé avec le s déjà calé (écart toujours nul)", [("moteur", "    beta = L.beta_retenu(tT, s)\n    f_prevu = beta * fcm\n    out = {\"age\": x[\"age\"]",
                                                                      "    beta = f_cyl / fcm\n    f_prevu = beta * fcm\n    out = {\"age\": x[\"age\"]")]),
    ("le taux inversé (f / cible)", [("moteur", "        out[\"taux\"] = cible / out[\"f_prevu\"] if out[\"f_prevu\"] > 0 else None", "        out[\"taux\"] = out[\"f_prevu\"] / cible if out[\"f_prevu\"] > 0 else None")]),
    ("« attention » jamais dit (la variante prudente ne compte pas)", [("moteur", "(\"ok\" if out[\"prudent_ok\"] else \"attention\")", "\"ok\"")]),
    ("une ★ au sélecteur sans âge prévu", [("moteur", "    if date:\n        for c in out:\n            if c[\"ok\"]:", "    if True:\n        for c in out:\n            if c[\"ok\"]:")]),
    ("le sélecteur : le s du durcissement choisi, même calé sur l'essai", [("moteur", "    _avert, _fck, _dur, p, _essai_, s = _base(e)", "    _avert, _fck, _dur, p, _essai_, s = _base(e)\n    s = _dur[\"s\"]")]),
    ("l'inverse rend la FIN du jour où la cible est atteinte", [("lois", "            return debut + (t_T - total) / k\n        total += k * duree\n        debut += duree\n    if k_fin > 0.0:",
                                                                  "            return debut + duree\n        total += k * duree\n        debut += duree\n    if k_fin > 0.0:")]),
    ("l'horizon figé à 28 jours", [("moteur", "    horizon = min(max([HORIZON_MIN_J] + [MARGE_HORIZON * x for x in ages]), HORIZON_MAX_J)", "    horizon = HORIZON_MIN_J")]),
    ("une cible au-dessus de la classe dite atteignable (à 28 jours)", [("lois", "    if b > 1.0 + 1e-12:\n        return None", "    if b > 1.0 + 1e-12:\n        return AGE_REFERENCE_J")]),
    ("T de −25 °C admise (borne basse de la température retirée)", [("entree", "T_MIN, T_MAX = -20.0, 60.0 ", "T_MIN, T_MAX = -200.0, 60.0 ")]),
    ("neuf objectifs admis", [("entree", "MAX_OBJECTIFS = 8", "MAX_OBJECTIFS = 9")]),
    ("le défaut d'ouverture en ciment N (optimiste pour le béton courant)", [("entree", "\"durcissement\": {\"mode\": \"beton\", \"ciment\": \"N\"", "\"durcissement\": {\"mode\": \"ciment\", \"ciment\": \"N\"")]),
]
TEMOIN = ("témoin : un commentaire ajouté, aucun calcul changé", [("lois", "DELTA_FCM = 8.0 ", "DELTA_FCM = 8.0  # (témoin) ")])
non_vues = []
for nom, mut in MUTATIONS:
    try:
        rj = rejets(mut)
    except AssertionError as x:
        rj = []
        chk(f"mutation « {nom} » : applicable", False, x)
    if not rj:
        non_vues.append(nom)
    chk(f"mutation « {nom} » : rejetée ({len(rj)} contrôle(s) : {rj[0][:60] if rj else '—'}…)", bool(rj))
chk(f"les {len(MUTATIONS)} mutations (une par formule, table ou règle : référence 293, 28 → 27, + 8, premier jour entier, table belge, palier, plafond, "
    "(B.10) sous 0 °C, marge, prolongation, date tronquée, cube, bases confondues, borne haute de la classe, gel, calage, sélecteur…) sont TOUTES rejetées",
    not non_vues and len(MUTATIONS) >= 14, non_vues)
chk("la mutation témoin (un commentaire) n'est rejetée par aucun contrôle : les rejets viennent bien des mutations", not rejets(TEMOIN[1]))
chk("après les mutations, le moteur chargé est intact (mêmes résultats)",
    json.loads(json.dumps([M.calculer(e) for e in COULAGES])) == attendu["res"] and sys.modules["beton.maturite.lois"] is L and PAQUET.moteur is M)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
