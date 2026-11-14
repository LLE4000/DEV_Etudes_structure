# -*- coding: utf-8 -*-
"""Barres en acier — LA BARRE CHARGÉE (acier/barre/combinaisons.py, statique.py, entree.py, moteur.py) : les cas de
charge, les combinaisons, les efforts le long de la barre, les sections à vérifier, les tronçons de déversement, et les
deux méthodes rendues côte à côte. La spécification : docs/acier/BARRES_ACIER.md § 5.8 et § 9.

Les références :
  1. COMBINAISONS — (6.10) écrite à la main pour G, G + Q, G + Q + neige, deux cas d'une même nature ; la méthode
     historique à 100 %.
  2. STATIQUE — la somme pondérée des cas = le formulaire résolu d'un coup sur les charges pondérées ; le poids propre ;
     l'extremum du moment ; les tronçons (deux appuis, console, poutre continue, maintiens).
  3. R2 DE BOUT EN BOUT (brief § 11.3) — IPE 300, 6 m : q_Ed 3,267 kN/m, P_Ed 36 kN → M_Ed 68,70 kN·m, V_Ed 27,80 kN,
     puis le déversement, égal à l'appel direct de l'Eurocode 3 sur le même tronçon.
  4. L'ENVELOPPE — la disposition défavorable des charges d'une poutre continue ; les efforts d'une section pris
     ENSEMBLE ; la méthode historique dans chaque section ; la console ; les maintiens d'une semelle.
  5. EFFORTS IMPOSÉS — le classeur HEB 200 du bureau : les deux méthodes = leurs appels directs.
  6. SORTIE — formules recalculées par leur gabarit, JSON strict, la barre reçue n'est pas modifiée, refus motivés.
  7. MUTATIONS — chaque altération plausible du texte fait rougir un contrôle.
  8. PURETÉ.

    /usr/bin/python3.12 tests/test_barre_moteur.py
"""
import copy
import importlib.util
import json
import math
import os
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.barre import combinaisons as CO  # noqa: E402
from acier.barre import ec3 as EC  # noqa: E402
from acier.barre import entree as EN  # noqa: E402
from acier.barre import formules as FO  # noqa: E402
from acier.barre import historique as HI  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.barre import parametres as PA  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from acier.barre import section as SE  # noqa: E402
from acier.barre import statique as ST  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402
from rdm import poutres as PO  # noqa: E402

OK, KO = [], []
FICHIERS = ("combinaisons", "statique", "entree", "moteur")
CHEMINS = {k: os.path.join(RACINE, "acier", "barre", k + ".py") for k in FICHIERS}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def verif(r, id_):
    return next((v for v in r["verifications"] if v["id"] == id_), None)


def q(v, a, b):
    return {"type": "q", "v": v, "a": a, "b": b}


def P_(v, a):
    return {"type": "P", "v": v, "a": a}


def barre(**k):
    b = EN.nouvelle_barre("B")
    b.update(k)
    return b


R2 = dict(poids_propre=False, cas=[{"nom": "G", "nature": "permanent", "charges": [q(2.42, 0.0, 6.0), P_(10.0, 3.0)]},
                                   {"nom": "Q", "nature": "exploitation_A", "charges": [P_(15.0, 3.0)]}])
CONTINUE = dict(profil="HEA 200", modele="continue", travees=[5.0, 5.0], poids_propre=False,
                cas=[{"nom": "G", "nature": "permanent", "charges": [q(6.0, 0.0, 10.0)]},
                     {"nom": "Q1", "nature": "exploitation_B", "charges": [q(8.0, 0.0, 5.0)]},
                     {"nom": "Q2", "nature": "exploitation_B", "charges": [q(8.0, 5.0, 10.0)]}])


# ------------------------------------------------------------------ les contrôles de référence (rejoués par les mutations)
def controles(M=None):
    M = M or {"combinaisons": CO, "statique": ST, "entree": EN, "moteur": MO}
    co, st, en, mo = M["combinaisons"], M["statique"], M["entree"], M["moteur"]
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    Pm = PA.jeu()
    # ---- 1. les combinaisons
    G, Q, S = {"nom": "G", "nature": "permanent"}, {"nom": "Q", "nature": "exploitation_A"}, {"nom": "S", "nature": "neige"}
    c("un seul cas permanent : une seule combinaison, 1,35·G", [(x["nom"], x["facteurs"]) for x in co.elu([G], Pm)] == [("1,35·G", [1.35])])
    c("G + Q : 1,35·G ; 1,35·G + 1,5·Q ; G + 1,5·Q (G favorable)",
      [(x["nom"], x["facteurs"]) for x in co.elu([G, Q], Pm)] == [("1,35·G", [1.35, 0.0]), ("1,35·G + 1,5·Q", [1.35, 1.5]), ("G + 1,5·Q", [1.0, 1.5])])
    noms = {x["nom"]: x["facteurs"] for x in co.elu([G, Q, S], Pm)}
    c("G + Q (catégorie A) + neige : neuf combinaisons — chaque arrangement, chaque dominante ; l'accompagnement reçoit 1,5·ψ₀ "
      "(neige 1,5·0,5 = 0,75 ; exploitation 1,5·0,7 = 1,05)",
      len(noms) == 9 and all(proche(a, b) for a, b in zip(noms["1,35·G + 1,5·Q + 0,75·S"], [1.35, 1.5, 0.75]))
      and all(proche(a, b) for a, b in zip(noms["1,35·G + 1,05·Q + 1,5·S"], [1.35, 1.05, 1.5])) and "G + 1,5·S" in noms and "1,35·G + 1,5·Q" in noms, sorted(noms))
    Q1, Q2 = {"nom": "Q1", "nature": "exploitation_B"}, {"nom": "Q2", "nature": "exploitation_B"}
    noms = {x["nom"]: x["facteurs"] for x in co.elu([G, Q1, Q2], Pm)}
    c("deux cas de la MÊME nature (une charge d'exploitation sur chaque travée) : une seule action disposée librement — 1,5 sur les deux, "
      "et chaque travée chargée seule ; sept combinaisons",
      len(noms) == 7 and noms["1,35·G + 1,5·Q1 + 1,5·Q2"] == [1.35, 1.5, 1.5] and noms["1,35·G + 1,5·Q1"] == [1.35, 1.5, 0.0]
      and noms["G + 1,5·Q2"] == [1.0, 0.0, 1.5], sorted(noms))
    H = {"nom": "T", "nature": "exploitation_H"}
    noms = [x["nom"] for x in co.elu([G, H, S], Pm)]
    c("toiture (catégorie H, ψ₀ = 0) en accompagnement : elle disparaît de la combinaison, sans doublon",
      "1,35·G + 1,5·S" in noms and len(noms) == len(set(noms)) and not any("0·T" in n for n in noms), noms)
    c("méthode historique : G + ΣQ à 100 %, pour chaque arrangement — G ; G + Q ; G + S ; G + Q + S",
      [(x["nom"], x["facteurs"]) for x in co.service([G, Q, S])] == [("G", [1.0, 0.0, 0.0]), ("G + Q", [1.0, 1.0, 0.0]), ("G + S", [1.0, 0.0, 1.0]),
                                                                    ("G + Q + S", [1.0, 1.0, 1.0])])
    c("domaine : aucun cas, nature inconnue, plus de quatre cas variables — refusés",
      all(_refuse(f) for f in (lambda: co.elu([], Pm), lambda: co.elu([{"nom": "X", "nature": "seisme"}], Pm),
                               lambda: co.elu([G] + [dict(Q, nom=f"Q{i}") for i in range(5)], Pm))))

    # ---- 2. la statique
    b = en.normaliser(barre(travees=[6.0], cas=[{"nom": "G", "nature": "permanent", "N": 30.0, "charges": [q(4.0, 0.0, 6.0), P_(10.0, 2.0)]},
                                               {"nom": "Q", "nature": "exploitation_A", "N": 20.0, "charges": [q(3.0, 1.0, 5.0)]}]))
    s = SE.du_catalogue(b["profil"])
    cas = st.cas_complets(b, s)
    g = st.poids_propre(s)
    c("poids propre : la masse linéique du catalogue × 9,81 — IPE 300 : 42,2 kg/m → 0,414 kN/m, un cas permanent en tête",
      proche(g, CAT.par_nom()["IPE 300"]["masse"] * 9.81 / 1000.0) and round(g, 3) == 0.414 and cas[0]["nom"] == st.POIDS_PROPRE
      and cas[0]["nature"] == "permanent" and cas[0]["charges"] == [q(g, 0.0, 6.0)] and len(cas) == 3
      and len(st.cas_complets(dict(b, poids_propre=False), s)) == 2, g)
    sols = st.resoudre(b, cas, 1.0e4)
    f = [1.35, 1.35, 1.5]
    comb = st.Combinaison("x", sols, f, [cc.get("N", 0.0) for cc in cas])
    direct = PO.resoudre("appuis", [6.0], [q(1.35 * g, 0.0, 6.0), q(1.35 * 4.0, 0.0, 6.0), P_(13.5, 2.0), q(4.5, 1.0, 5.0)], 1.0e4)
    ecarts = []
    for x in (0.0, 1.0, 2.0, 2.5, 3.7, 5.0, 6.0):
        e, d = comb.lire(x), direct.evaluer(x)
        if not all(proche(e[k], d[k], 1e-9) for k in ("M_g", "M_d", "V_g", "V_d", "f")):
            ecarts.append((x, e, d))
    c("une combinaison = la somme pondérée des cas : M, V (des deux côtés d'une section) et la flèche égalent le formulaire résolu d'un coup "
      "sur les charges pondérées ; N = Σ γ·N (1,35·30 + 1,5·20 = 70,5 kN)", not ecarts and proche(comb.N, 70.5), ecarts[:1])
    xs = st.sections(b, comb, st.ruptures(sols))
    Mmax = max(comb.M(6.0 * i / 6000.0) for i in range(6001))
    c("les sections à vérifier : les appuis, chaque rupture de chargement (1, 2 et 5 m) et l'extremum du moment de la travée, au millionième",
      {0.0, 1.0, 2.0, 5.0, 6.0} <= set(xs) and max(comb.M(x) for x in xs) >= Mmax and proche(max(comb.M(x) for x in xs), Mmax, 1e-6)
      and len(xs) == 6, (xs, Mmax))
    tr = st.troncons(b, s)
    c("deux appuis : un tronçon, d'une fourche à l'autre", [(t["x1"], t["x2"], t["appuis"], t["maintiens"]) for t in tr] == [(0.0, 6.0, ("fourche", "fourche"), [])])
    bm = en.normaliser(barre(maintiens=[{"x": 4.0, "genre": "bas"}, {"x": 2.0, "genre": "complet"}, {"x": 3.0, "genre": "haut"}]))
    tr = st.troncons(bm, s)
    cote = (300.0 - 10.7) / 2.0
    c("maintiens : un maintien COMPLET coupe la barre (0–2 m, 2–6 m) ; un maintien d'une seule semelle reste dans son tronçon, à la cote du plan "
      "moyen de la semelle (+144,65 mm en haut, −144,65 mm en bas), en abscisse locale",
      [(t["x1"], t["x2"]) for t in tr] == [(0.0, 2.0), (2.0, 6.0)] and tr[0]["maintiens"] == []
      and all(proche(a, b) for (a, _), (b, _) in zip(tr[1]["maintiens"], [(1000.0, 0), (2000.0, 0)]))
      and [round(z, 2) for _, z in tr[1]["maintiens"]] == [round(cote, 2), -round(cote, 2)], tr)
    bc = en.normaliser(barre(modele="console", travees=[2.0], cas=[{"nom": "G", "nature": "permanent", "charges": [P_(8.0, 2.0)]}]))
    c("console : un tronçon, ENCASTRÉ à la racine et LIBRE au bout ; un maintien complet au bout en fait une fourche",
      [t["appuis"] for t in st.troncons(bc, s)] == [("encastre", "libre")]
      and [t["appuis"] for t in st.troncons(dict(bc, maintiens=[{"x": 2.0, "genre": "complet"}]), s)] == [("encastre", "fourche")])
    bk = en.normaliser(barre(**CONTINUE))
    c("poutre continue : un tronçon par travée (l'appui intermédiaire tient la poutre)",
      [(t["x1"], t["x2"]) for t in st.troncons(bk, SE.du_catalogue("HEA 200"))] == [(0.0, 5.0), (5.0, 10.0)])
    casx = [{"nom": "G", "nature": "permanent", "charges": [q(4.0, 0.0, 6.0), P_(10.0, 1.0), P_(7.0, 3.5), {"type": "tri", "v": 6.0, "a": 2.0, "b": 6.0, "sens": "croissante"}]}]
    rep, pon = st.charges_du_troncon(casx, [1.5], {"x1": 2.0, "x2": 6.0}, 150.0)
    c("les charges d'un tronçon (2 à 6 m) : pondérées, en N et mm, en abscisse locale, à leur cote ; la charge hors du tronçon est écartée ; "
      "une charge triangulaire garde ses deux intensités d'extrémité",
      pon == [(1500.0, 10500.0, 150.0)] and len(rep) == 2 and all(proche(a, b) for a, b in zip(rep[0], (0.0, 4000.0, 6.0, 6.0, 150.0)))
      and all(proche(a, b) for a, b in zip(rep[1], (0.0, 4000.0, 0.0, 9.0, 150.0))), (rep, pon))

    # ---- 3. R2 de bout en bout
    r = mo.verifier(barre(**R2))
    stq = r["statique"]
    i3 = stq["x"].index(3.0)
    e3 = r["methodes"]["ec3"]
    c("R2 (IPE 300 S235, 6 m, G = 2,42 kN/m + 10 kN à mi-portée, Q = 15 kN à mi-portée) : q_Ed = 1,35·2,42 = 3,267 kN/m, P_Ed = 36 kN → "
      "M_Ed = 68,70 kN·m à mi-portée, V_Ed = 27,80 kN à l'appui ; combinaison gouvernante 1,35·G + 1,5·Q",
      proche(stq["enveloppes"]["ec3"]["M_max"][i3], 3.267 * 36 / 8 + 36.0 * 6 / 4) and proche(stq["enveloppes"]["ec3"]["V_max"][0], 3.267 * 3 + 18.0)
      and round(verif(e3, "E-My")["valeur"], 2) == 68.70 and round(verif(e3, "E-Vz")["valeur"], 2) == 27.80 and verif(e3, "E-My")["x"] == 3.0
      and verif(e3, "E-Vz")["x"] in (0.0, 6.0) and e3["combinaison"] == "1,35·G + 1,5·Q", (verif(e3, "E-My"), e3["combinaison"]))
    s3 = SE.du_catalogue("IPE 300")

    def M23(x):
        return 3.267 * x * (6000.0 - x) / 2.0 + 36000.0 * (x if x <= 3000.0 else 6000.0 - x) / 2.0

    direct = EC.verifier({"section": s3, "nuance": "S235", "MyEd": M23(3000.0), "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}},
                          "deversement": {"troncon": {"L": 6000.0, "moment": M23, "reparties": [(0.0, 6000.0, 3.267, 150.0)],
                                                      "ponctuelles": [(3000.0, 36000.0, 150.0)]}}})
    Dm, Dd = e3["grandeurs"]["deversement"], direct["grandeurs"]["deversement"]
    c("R2 : le déversement de la barre chargée (tronçon de 6 m, charges sur la semelle haute, diagramme réel) = l'appel direct de l'Eurocode 3 "
      "sur le même tronçon, à 10⁻⁸ ; M_cr = 86,4 kN·m — à moins de 1 % de la référence R3 (86,98, propriétés des catalogues) ; taux 89,5 %",
      proche(Dm["M_cr"], Dd["M_cr"], 1e-8) and proche(verif(e3, "E-Mb")["taux"], verif(direct, "E-Mb")["taux"], 1e-8)
      and abs(Dm["M_cr"] / 86.98e6 - 1) < 0.01 and round(verif(e3, "E-Mb")["taux"], 3) == 0.895 and verif(e3, "E-Mb")["troncon"] == [0.0, 6.0]
      and Dm["kc"] == 1.0, (Dm["M_cr"], verif(e3, "E-Mb")["taux"]))
    rc_, rb_ = (mo.verifier(barre(niveau=n_, **R2))["methodes"]["ec3"]["grandeurs"]["deversement"]["M_cr"] for n_ in ("centre", "bas"))
    c("le niveau d'application des charges : semelle haute < centre de cisaillement < semelle basse", Dm["M_cr"] < rc_ < rb_, (Dm["M_cr"], rc_, rb_))
    h3 = r["methodes"]["historique"]
    Ms, Vs = 2.42 * 36 / 8 + 25.0 * 6 / 4, 2.42 * 3 + 12.5
    hd = HI.verifier({"section": s3, "fy": 235.0, "P": 0.0, "My": Ms * 1e6, "Vz": 12.5e3, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}},
                      "deversement": {"Lc": 6000.0}})
    c("R2, méthode historique : charges de SERVICE (G + Q : M = 48,39 kN·m), dans la section de mi-portée avec l'effort tranchant qui s'y "
      "trouve (12,5 kN, et non les 19,76 kN de l'appui) = l'appel direct ; le déversement de la doc sur 6 m rend la barre non conforme",
      h3["combinaison"] == "G + Q" and proche(Ms, 48.39) and proche(verif(h3, "H-SIGMA")["valeur"], verif(hd, "H-SIGMA")["valeur"], 1e-8)
      and proche(verif(h3, "H-EQ")["valeur"], verif(hd, "H-EQ")["valeur"], 1e-8) and verif(h3, "H-SIGMA")["x"] == 3.0 and h3["statut"] == RE.NOK
      and proche(Vs, 19.76), (verif(h3, "H-EQ"), Ms, Vs))
    c("R2 : les deux méthodes côte à côte — la comparaison a une ligne « flexion et contraintes » et la ligne du plus défavorable ; "
      "le statut de la barre est le plus sévère des deux", [x["sujet"] for x in r["comparaison"]] == ["Flexion et contraintes, déversement compris",
                                                                                              "Effort tranchant seul", "Le plus défavorable"]
      and proche(r["comparaison"][-1]["historique"], h3["taux_max"]) and proche(r["comparaison"][-1]["ec3"], e3["taux_max"])
      and r["comparaison"][1]["historique"] is None and r["statut"] == RE.NOK and e3["statut"] == RE.OK, r["comparaison"])
    fl = stq["fleche"]
    EI = 210000.0 * s3["Iy"] / 1e9
    c("la flèche de service (information) : 5·q·L⁴/(384·E·I) + P·L³/(48·E·I) = 8,7 mm à mi-portée, L/687",
      proche(fl["valeur"], (5 * 2.42 * 6 ** 4 / (384 * EI) + 25.0 * 6 ** 3 / (48 * EI)) * 1e3, 1e-9) and fl["x"] == 3.0 and round(fl["rapport"]) == 687, fl)
    r1 = mo.verifier(barre(poids_propre=False, cas=[{"nom": "G", "nature": "permanent", "charges": [q(5.0, 0.0, 6.0)]}]))
    r2 = mo.verifier(barre(poids_propre=False, cas=[{"nom": "G", "nature": "permanent", "charges": [P_(20.0, 3.0)]}]))
    c("k_c du tableau 6.6 reconnu sur une poutre sur deux appuis : charges uniformes sur toute la portée 0,94 ; une seule charge à mi-portée 0,86 ; "
      "le mélange des deux (R2) : hors tableau, k_c = 1",
      r1["methodes"]["ec3"]["grandeurs"]["deversement"]["kc"] == 0.94 and r2["methodes"]["ec3"]["grandeurs"]["deversement"]["kc"] == 0.86)

    # ---- 4. l'enveloppe
    r = mo.verifier(barre(**CONTINUE))
    env = r["statique"]["enveloppes"]["ec3"]
    xg = r["statique"]["x"]
    q1 = 1.35 * 6 + 1.5 * 8
    MB_tout = -q1 * 25 / 8
    RA = q1 * 5 / 2 - (q1 + 1.35 * 6) * 25 / 16 / 5
    Mtravee = RA ** 2 / (2 * q1)
    e = r["methodes"]["ec3"]
    c("poutre continue 2 × 5 m, une charge d'exploitation par travée : le moment sur appui vient des DEUX travées chargées (−q·L²/8 = −62,8 kN·m), "
      "le moment en travée d'UNE seule travée chargée (42,7 kN·m, contre 35,3 toutes chargées) — l'enveloppe trouve la disposition défavorable",
      proche(env["M_min"][xg.index(5.0)], MB_tout) and abs(max(m for x, m in zip(xg, env["M_max"]) if x <= 5.0) / Mtravee - 1) < 2e-3
      and round(Mtravee, 1) == 42.7 and round(MB_tout, 1) == -62.8 and verif(e, "E-My")["x"] == 5.0
      and proche(verif(e, "E-My")["valeur"], -MB_tout) and verif(e, "E-My")["combinaison"] == "1,35·G + 1,5·Q1 + 1,5·Q2",
      (env["M_min"][xg.index(5.0)], MB_tout, Mtravee))
    c("le bilan de chaque combinaison est rendu (sept combinaisons), avec sa vérification gouvernante et son taux",
      len(e["combinaisons"]) == 7 and all(x["gouvernant"] and x["taux"] is not None for x in e["combinaisons"])
      and proche(max(x["taux"] for x in e["combinaisons"]), e["taux_max"]))
    # les efforts d'une section sont pris ENSEMBLE : une forte charge près d'un appui
    br = barre(profil="IPE 300", travees=[2.0], poids_propre=False, cas=[{"nom": "G", "nature": "permanent", "charges": [P_(400.0, 0.5)]}])
    r = mo.verifier(br)
    e = r["methodes"]["ec3"]
    Vg, Mx = 1.35 * 400 * 1.5 / 2.0, 1.35 * 400 * 1.5 / 2.0 * 0.5
    rs = __import__("acier.barre.ec3_section", fromlist=["x"]).verifier({"section": s3, "nuance": "S235", "MyEd": Mx * 1e6, "VzEd": Vg * 1e3})
    c("charge de 400 kN à 0,5 m d'un appui (portée 2 m) : sous la charge, V = 405 kN à gauche (> 0,5·V_pl) AVEC M = 202,5 kN·m → « flexion et "
      "effort tranchant » vérifiée là, égale à l'appel direct de la section ; elle est refusée, et la barre avec",
      verif(e, "E-MV") is not None and verif(e, "E-MV")["x"] == 0.5 and proche(verif(e, "E-MV")["limite"], verif(rs, "E-MV")["limite"], 1e-9)
      and proche(Vg, 405.0) and proche(Mx, 202.5) and verif(e, "E-MV")["statut"] == RE.NOK and e["statut"] == RE.NOK, verif(e, "E-MV"))
    # la console
    r = mo.verifier(barre(profil="IPE 240", modele="console", travees=[2.0], poids_propre=False,
                          cas=[{"nom": "G", "nature": "permanent", "N": 50.0, "charges": [P_(8.0, 2.0)]}]))
    e = r["methodes"]["ec3"]
    c("console de 2 m sous une charge en bout et un effort normal : le moment est à l'encastrement (1,35·8·2 = 21,6 kN·m, x = 0) ; elle flambe "
      "sur DEUX fois sa longueur autour de l'axe fort ; son tronçon de déversement est encastré puis libre",
      proche(verif(e, "E-My")["valeur"], 21.6) and verif(e, "E-My")["x"] == 0.0 and e["grandeurs"]["flambement"]["y"]["Lcr"] == 4000.0
      and e["grandeurs"]["deversement"]["troncon"]["appuis"] == ["encastre", "libre"] and verif(e, "E-Mb") is not None, e["grandeurs"]["flambement"]["y"])
    # la longueur de flambement de l'axe faible, les maintiens d'une semelle
    r = mo.verifier(barre(maintiens=[{"x": 2.0, "genre": "complet"}], cas=[{"nom": "G", "nature": "permanent", "N": 100.0, "charges": [q(5.0, 0.0, 6.0)]}]))
    c("un maintien complet à 2 m : l'axe faible flambe sur le plus long tronçon (4 m), l'axe fort sur la portée (6 m)",
      r["methodes"]["ec3"]["grandeurs"]["flambement"]["z"]["Lcr"] == 4000.0 and r["methodes"]["ec3"]["grandeurs"]["flambement"]["y"]["Lcr"] == 6000.0)
    Lc = {g_: mo.verifier(barre(maintiens=[{"x": 3.0, "genre": g_}]))["methodes"]["historique"]["grandeurs"]["deversement"].get("Lc") for g_ in ("haut", "bas")}
    Mcr = {g_: mo.verifier(barre(maintiens=[{"x": 3.0, "genre": g_}]))["methodes"]["ec3"]["grandeurs"]["deversement"]["M_cr"] for g_ in ("haut", "bas")}
    nu = mo.verifier(barre())["methodes"]["ec3"]["grandeurs"]["deversement"]["M_cr"]
    c("un maintien d'une seule semelle à mi-portée, moment positif partout : la semelle HAUTE (comprimée) ramène la longueur de déversement "
      "historique à 3 m et relève beaucoup M_cr ; la semelle BASSE (tendue) laisse 6 m et aide peu",
      Lc == {"haut": 3000.0, "bas": 6000.0} and Mcr["haut"] > 1.8 * nu and nu < Mcr["bas"] < 1.3 * nu, (Lc, Mcr, nu))
    r = mo.verifier(barre(profil="IPE 750x147", nuance="S355", travees=[8.0], methodes=["ec3"], maintiens=[{"x": 2.0}, {"x": 4.0}, {"x": 6.0}],
                          cas=[{"nom": "G", "nature": "permanent", "charges": [q(20.0, 0.0, 8.0)]}]))
    c("une vérification non couverte (voilement par cisaillement d'une âme élancée, lot 4) traverse l'enveloppe : rien ne dépasse, mais la "
      "méthode et la barre sont rendues « non couvert », jamais « ok »",
      verif(r["methodes"]["ec3"], "E-Vb")["statut"] == RE.NON_COUVERT and r["methodes"]["ec3"]["statut"] == RE.NON_COUVERT
      and r["methodes"]["ec3"]["taux_max"] < 1.0 and r["statut"] == RE.NON_COUVERT, (r["methodes"]["ec3"]["statut"], r["statut"]))
    r = mo.verifier(barre(profil="HD 400x818.0", nuance="S355"))
    c("une méthode qui REFUSE la barre (HD 400x818 : semelle de 97 mm, hors du tableau 3.1 de l'Eurocode) est rendue « non couvert » avec son "
      "motif ; l'autre méthode reste calculée",
      r["methodes"]["ec3"]["statut"] == RE.NON_COUVERT and "Tableau 3.1" in r["methodes"]["ec3"]["alertes"][0]["message"]
      and r["methodes"]["historique"]["taux_max"] is not None and r["statut"] in (RE.NON_COUVERT, RE.NOK), r["methodes"]["ec3"]["statut"])
    r = mo.verifier(barre(cas=[{"nom": "G", "nature": "permanent", "N": 150.0, "charges": [q(1.0, 0.0, 6.0)]},
                               {"nom": "Q", "nature": "exploitation_A", "N": 120.0, "charges": []}]))
    x661 = verif(r["methodes"]["ec3"], "E-661")
    c("un REFUS sans taux passe devant tout taux : N_Ed dépasse la charge critique dans la combinaison complète (382 kN > N_cr,z = 348 kN) alors "
      "qu'un taux existe sous G seul — l'enveloppe garde le refus, et sa combinaison",
      x661["statut"] == RE.NOK and x661["taux"] is None and x661["combinaison"].endswith("1,5·Q") and r["methodes"]["ec3"]["statut"] == RE.NOK, x661)
    kc = [mo.verifier(barre(poids_propre=False, cas=[{"nom": "G", "nature": "permanent", "charges": [q(5.0, 0.0, 6.0)]}], **k))["methodes"]["ec3"]
          ["grandeurs"]["deversement"]["kc"] for k in ({"maintiens": [{"x": 3.0}]}, {"maintiens": [{"x": 3.0, "genre": "bas"}]})]
    kcc = mo.verifier(barre(**CONTINUE))["methodes"]["ec3"]["grandeurs"]["deversement"]["kc"]
    c("k_c n'est lu dans le tableau 6.6 que si le tronçon EST la portée d'une poutre sur deux appuis, sans maintien intérieur : sinon k_c = 1",
      kc == [1.0, 1.0] and kcc == 1.0, (kc, kcc))
    r = mo.verifier({"profil": "IPE 300", "nuance": "S235", "cas": [{"nom": "G", "nature": "permanent", "charges": [q(5.0, 0.0, 6.0)]}]})
    rh = mo.verifier(barre(poids_propre=True, niveau="haut", cas=[{"nom": "G", "nature": "permanent", "charges": [q(5.0, 0.0, 6.0)]}]))
    hh = r["methodes"]["historique"]["grandeurs"]
    c("une barre donnée sans précision : poids propre compté, charges sur la semelle haute, s = 1,5, déversement historique actif, les deux méthodes",
      r["statique"]["cas"][0]["nom"] == st.POIDS_PROPRE and r["barre"]["niveau"] == "haut"
      and proche(r["methodes"]["ec3"]["grandeurs"]["deversement"]["M_cr"], rh["methodes"]["ec3"]["grandeurs"]["deversement"]["M_cr"])
      and proche(hh["sigma_adm"], 235.0 / 1.5) and hh["deversement"]["phi"] < 1.0 and list(r["methodes"]) == ["historique", "ec3"], hh["sigma_adm"])

    # ---- 5. efforts imposés
    be = barre(profil="HEB 200", mode="efforts", efforts={"service": {"N": 432.72, "My": 10.0, "Vz": 5.0}, "elu": {"N": 600.0, "My": 20.0, "Mz": 4.0}},
               flambement={"y": {"L": 7.2}, "z": {"L": 7.2, "u": 0.9}}, diagramme={"y": {"forme": "reparti"}})
    r = mo.verifier(be)
    sh = SE.du_catalogue("HEB 200")
    axes = {"y": {"L": 7200.0, "u": 1.0}, "z": {"L": 7200.0, "u": 0.9}}
    hd = HI.verifier({"section": sh, "fy": 235.0, "s": 1.5, "P": 432.72e3, "My": 10e6, "Vz": 5e3, "axes": axes})
    ed = EC.verifier({"section": sh, "nuance": "S235", "NEd": 600e3, "MyEd": 20e6, "MzEd": 4e6, "axes": axes,
                      "flexion": {"y": {"diagramme": "reparti"}}, "deversement": {"niveau": "haut"}})
    c("efforts imposés (le mode des classeurs) : les efforts de SERVICE vont à la méthode historique, ceux de CALCUL à l'Eurocode 3 ; chacune "
      "= son appel direct ; pas de statique",
      r["statique"] is None and [(x["id"], x["taux"]) for x in r["methodes"]["historique"]["verifications"]] == [(x["id"], x["taux"]) for x in hd["verifications"]]
      and [(x["id"], x["taux"]) for x in r["methodes"]["ec3"]["verifications"]] == [(x["id"], x["taux"]) for x in ed["verifications"]])
    r = mo.verifier(barre(profil="HEB 200", mode="efforts", efforts={"service": {"N": 432.72}}, flambement={"y": {"L": 7.2}, "z": {"L": 7.2}}))
    c("le classeur HEB 200 du bureau (432,72 kN, 7,20 m) : méthode historique 112 % (non conforme), Eurocode 3 sans effort de calcul : rien à vérifier",
      round(r["methodes"]["historique"]["taux_max"], 2) == 1.12 and r["methodes"]["historique"]["gouvernant"] == "H-N-z"
      and r["methodes"]["ec3"]["verifications"] == [] and r["statut"] == RE.NOK)
    r1, r2 = (mo.verifier(barre(profil="HEB 200", mode="efforts", efforts={"service": {"N": 100.0, "My": 30.0}}, flambement={"y": {"L": 5.0}, "z": {"L": 5.0}},
                                diagramme={"y": {"forme": "reparti"}}, historique={"C": c_}, methodes=["historique"])) for c_ in ("unite", "norme"))
    c("coefficient C de la méthode historique : 1 par défaut (les classeurs) ; « norme » : 0,85 sous une charge appliquée sur la barre",
      r1["methodes"]["historique"]["grandeurs"]["C_y"] == 1.0 and r2["methodes"]["historique"]["grandeurs"]["C_y"] == 0.85
      and list(r1["methodes"]) == ["historique"] and r1["comparaison"] == [])
    refus = [k for k, d in (("mode inconnu", {"mode": "x"}), ("nuance inconnue", {"nuance": "S999"}), ("profilé inconnu", {"profil": "IPE 999"}),
                            ("méthode inconnue", {"methodes": ["aci"]}), ("aucune méthode", {"methodes": []}), ("s hors de [1 ; 3]", {"historique": {"s": 0.5}}),
                            ("portée nulle", {"travees": [0.0]}), ("continue à une travée", {"modele": "continue"}), ("cas sans nature", {"cas": [{"nom": "G"}]}),
                            ("deux cas de même nom", {"cas": [dict(G, charges=[]), dict(G, charges=[])]}),
                            ("charge hors de la poutre", {"cas": [dict(G, charges=[P_(5.0, 9.0)])]}), ("maintien hors de la barre", {"maintiens": [{"x": 9.0}]}),
                            ("genre de maintien inconnu", {"maintiens": [{"x": 1.0, "genre": "x"}]}), ("niveau inconnu", {"niveau": "milieu"}),
                            ("efforts imposés sans longueur", {"mode": "efforts"}), ("effort texte", {"mode": "efforts", "efforts": {"elu": {"N": "12"}},
                                                                                                   "flambement": {"y": {"L": 3.0}, "z": {"L": 3.0}}}))
             if not _refuse(lambda d=d: mo.verifier(barre(**d)))]
    c("domaine : mode, nuance, profilé, méthode, coefficient s, portée, modèle, cas, charges, maintiens, niveau, longueurs, efforts — refusés avec un message",
      not refus, refus)
    return C


def _refuse(f):
    try:
        f()
    except RE.Refus as e:
        return bool(str(e))
    return False


def charger(mutation=None):
    """Les quatre modules, chargés depuis leur TEXTE (une mutation appliquée à l'un d'eux) et liés entre eux."""
    mods = {}
    for k in FICHIERS:
        src = open(CHEMINS[k], encoding="utf-8").read()
        if mutation and mutation[0] == k:
            if src.count(mutation[1]) != 1:
                return None
            src = src.replace(mutation[1], mutation[2])
        spec = importlib.util.spec_from_file_location(f"acier.barre.{k}_mute", CHEMINS[k])
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "acier.barre"
        exec(compile(src, CHEMINS[k], "exec"), mod.__dict__)
        mods[k] = mod
    mods["entree"].CO, mods["entree"].ST = mods["combinaisons"], mods["statique"]
    m = mods["moteur"]
    m.CO, m.ST, m.EN = mods["combinaisons"], mods["statique"], mods["entree"]
    return mods


MUTATIONS = [
    ("combinaisons", "l'accompagnement reçoit γ_Q sans ψ₀", 'f[i] = P["gamma_Q"] * P[cle][0]', 'f[i] = P["gamma_Q"]'),
    ("combinaisons", "ψ₁ au lieu de ψ₀", 'f[i] = P["gamma_Q"] * P[cle][0]', 'f[i] = P["gamma_Q"] * P[cle][1]'),
    ("combinaisons", "deux cas de même nature : un seul dominant", 'if cas[i]["nature"] == dom:', "if i == s[0]:"),
    ("combinaisons", "G favorable (1,00) oublié", 'for gG in (P["gamma_G_sup"], P["gamma_G_inf"]):', 'for gG in (P["gamma_G_sup"],):'),
    ("combinaisons", "un seul arrangement : tous les cas variables présents", "    for k in range(len(var) + 1):", "    for k in (len(var),):"),
    ("combinaisons", "méthode historique pondérée (1,35)", "            f[i] = 1.0\n", "            f[i] = 1.35\n"),
    ("statique", "pesanteur à 10 m/s²", "G_PESANTEUR = 9.81 ", "G_PESANTEUR = 10.0 "),
    ("statique", "poids propre oublié", '    if barre.get("poids_propre", True):', "    if False:"),
    ("statique", "effort normal non pondéré", "self.N = sum(f * n for f, n in zip(facteurs, N_cas))", "self.N = sum(n for f, n in zip(facteurs, N_cas))"),
    ("statique", "efforts des cas additionnés sans leurs facteurs", "                o[k] += f * e[k]\n", "                o[k] += e[k]\n"),
    ("statique", "extremum du moment non cherché", "            if 0 < i < BALAYAGE and vals[i] > 0:", "            if False:"),
    ("statique", "bout libre d'une console pris en fourche", "        tenus[round(L, 9)] = MC.LIBRE\n", "        tenus[round(L, 9)] = MC.FOURCHE\n"),
    ("statique", "racine d'une console en fourche", "tenus = {0.0: MC.ENCASTRE if console else MC.FOURCHE}", "tenus = {0.0: MC.FOURCHE}"),
    ("statique", "maintien d'une semelle compté comme complet", '        if genre == "complet":\n            tenus.setdefault', "        if True:\n            tenus.setdefault"),
    ("statique", "semelles haute et basse échangées", 'cote if genre == "haut" else -cote', '-cote if genre == "haut" else cote'),
    ("statique", "charges ponctuelles du tronçon non pondérées", "pon.append(((ch[\"a\"] - a) * 1e3, f * ch[\"v\"] * 1e3, zg))", "pon.append(((ch[\"a\"] - a) * 1e3, ch[\"v\"] * 1e3, zg))"),
    ("statique", "charges hors du tronçon gardées", '                if a - 1e-9 <= ch["a"] <= b + 1e-9:\n', "                if True:\n"),
    ("statique", "charge triangulaire prise uniforme", "                    q1, q2 = q(u), q(v)\n", "                    q1 = q2 = ch[\"v\"]\n"),
    ("statique", "appui intermédiaire d'une poutre continue non tenu", "            tenus[round(x, 9)] = MC.FOURCHE\n", "            pass\n"),
    ("moteur", "un seul côté de la section vérifié", "    return [d] if (abs(g[0] - d[0]) < EPS and abs(g[1] - d[1]) < EPS) else [g, d]", "    return [d]"),
    ("moteur", "k_c du tableau appliqué à tout tronçon", '    if b["modele"] != "appuis" or tr["maintiens"] or tr["x2"] - tr["x1"] < L - EPS:\n        return None', "    if False:\n        return None"),
    ("moteur", "niveau d'application des charges ignoré", 'zg = {"haut": 1.0, "centre": 0.0, "bas": -1.0}[b["niveau"]] * (s.get("h") or 0.0) / 2.0', "zg = 0.0"),
    ("moteur", "console : flambement sur une fois sa longueur", "                L = 2.0 * L   ", "                L = 1.0 * L   "),
    ("moteur", "axe faible : flambement sur toute la barre malgré les maintiens", 'max(t["x2"] - t["x1"] for t in trs)', 'sum(b["travees"])'),
    ("moteur", "l'enveloppe garde l'occurrence la plus FAVORABLE", '            if courant is None or rang > courant["rang"]:', '            if courant is None or rang < courant["rang"]:'),
    ("moteur", "un refus sans taux passe derrière les taux", "        return (2, 0.0)\n", "        return (0, 0.0)\n"),
    ("moteur", "méthode historique : un maintien d'une semelle tendue compté", "        if (z > 0 and haut) or (z < 0 and bas):", "        if True:"),
    ("moteur", "méthode historique sur les combinaisons pondérées", "        for co in CO.service(cas):", "        for co in CO.elu(cas, P):"),
    ("moteur", "Eurocode 3 sur les combinaisons de service", "        for co in CO.elu(cas, P):\n            C = ST.Combinaison", "        for co in CO.service(cas):\n            C = ST.Combinaison"),
    ("moteur", "une méthode qui refuse fait tomber toute la barre", "            except RE.Refus as e:\n                methodes[nom] = _refusee(nom, norme, str(e))\n    return {\"statique\": statique",
     "            except ZeroDivisionError as e:\n                methodes[nom] = _refusee(nom, norme, str(e))\n    return {\"statique\": statique"),
    ("moteur", "une méthode « non couverte » rend la barre « ok »", "statut = RE.NOK if RE.NOK in statuts else RE.NON_COUVERT if RE.NON_COUVERT in statuts else RE.OK",
     "statut = RE.NOK if RE.NOK in statuts else RE.OK"),
    ("moteur", "efforts imposés : les efforts de calcul donnés à la méthode historique", 'e, h = b["efforts"]["service"], b["historique"]', 'e, h = b["efforts"]["elu"], b["historique"]'),
    ("moteur", "coefficient C « norme » : toujours 1", "    if forme in (DV.REPARTI, DV.PONCTUEL, DV.GENERAL):\n        return {\"mode\": \"charges\"}", "    if False:\n        return {\"mode\": \"charges\"}"),
    ("moteur", "déversement : le plus grand moment du tronçon remplacé par zéro", '"NEd": C.N * 1e3, "MyEd": Mmax * 1e6, "axes": axes,', '"NEd": C.N * 1e3, "MyEd": 0.0, "axes": axes,'),
    ("moteur", "méthode historique : l'effort tranchant de la section oublié", '"P": C.N * 1e3, "My": M * 1e6, "Vz": V * 1e3, "axes": axes,', '"P": C.N * 1e3, "My": M * 1e6, "Vz": 0.0, "axes": axes,'),
    ("moteur", "la flèche rendue en mètres", '        fleche["valeur"] *= 1e3  ', '        fleche["valeur"] *= 1.0  '),
    ("entree", "le poids propre éteint par défaut", '"poids_propre": True, "niveau": "haut", "maintiens": [],', '"poids_propre": False, "niveau": "haut", "maintiens": [],'),
    ("entree", "les charges au centre de cisaillement par défaut", '"poids_propre": True, "niveau": "haut", "maintiens": [],', '"poids_propre": True, "niveau": "centre", "maintiens": [],'),
    ("entree", "coefficient de sécurité 1,0 par défaut", '"historique": {"s": 1.5, "C": "unite", "deversement": True, "tolerance": False},\n            "parametres": {}}', '"historique": {"s": 1.0, "C": "unite", "deversement": True, "tolerance": False},\n            "parametres": {}}'),
    ("entree", "déversement historique éteint par défaut", '"historique": {"s": 1.5, "C": "unite", "deversement": True, "tolerance": False},\n            "parametres": {}}', '"historique": {"s": 1.5, "C": "unite", "deversement": False, "tolerance": False},\n            "parametres": {}}'),
]


def main():
    print("== 1 à 5. les combinaisons, la statique, R2, l'enveloppe, les efforts imposés")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 6. la sortie")
    faux, n = [], 0
    cas = [barre(**R2), barre(**CONTINUE), barre(profil="HEB 200", travees=[4.0], cas=[{"nom": "G", "nature": "permanent", "N": 300.0, "charges": [q(5.0, 0.0, 4.0)]},
                                                                                       {"nom": "Q", "nature": "exploitation_A", "N": 200.0, "charges": [q(4.0, 0.0, 4.0)]},
                                                                                       {"nom": "S", "nature": "neige", "charges": [P_(10.0, 1.0)]}]),
           barre(profil="IPE 240", modele="console", travees=[2.0], cas=[{"nom": "G", "nature": "permanent", "charges": [P_(8.0, 2.0), {"type": "tri", "v": 4.0, "a": 0.0, "b": 2.0}]}]),
           barre(profil="HEB 200", mode="efforts", efforts={"service": {"N": 300.0, "My": 20.0, "Mz": 5.0, "Vz": 30.0}, "elu": {"N": 420.0, "My": 28.0, "Mz": 7.0, "Vz": 42.0}},
                 flambement={"y": {"L": 5.0}, "z": {"L": 5.0}}),
           barre(profil="RHS 200x100x6.3", nuance="S355", maintiens=[{"x": 2.5, "genre": "complet"}])]
    for b in cas:
        r = MO.verifier(b)
        json.dumps(r, allow_nan=False)
        for m, x in r["methodes"].items():
            cles = [f["cle"] for f in x["calcul"]]
            if len(cles) != len(set(cles)):
                faux.append((b["profil"], m, "clé répétée"))
            for f in x["calcul"]:
                n += 1
                if f["gabarit"] not in FO.GABARITS or not proche(FO.evaluer(f), f["resultat"]):
                    faux.append((b["profil"], m, f["cle"]))
            for v in x["verifications"]:
                if any(k not in set(cles) for k in v["formules"]):
                    faux.append((b["profil"], m, v["id"], "formules absentes"))
                if r["statique"] is not None and (v["combinaison"] is None or (v["x"] is None and v["troncon"] is None)):
                    faux.append((b["profil"], m, v["id"], "sans combinaison ni lieu"))
    chk(f"six barres (deux appuis, continue, poteau-poutre à trois cas, console, efforts imposés, tube) : JSON strict ; {n} formules recalculées par "
        "leur gabarit, clés uniques ; chaque vérification d'une barre chargée dit sa combinaison et son lieu", n > 250 and not faux, faux[:4])
    b0 = barre(**CONTINUE)
    b1 = copy.deepcopy(b0)
    MO.verifier(b1)
    chk("verifier ne modifie pas la barre reçue", b1 == b0)
    r = MO.verifier(barre(**R2))
    chk("la sortie : la barre relue, le profilé, sa masse, la statique (abscisses, enveloppes, réactions par cas, flèche, tronçons), les méthodes, "
        "le statut, la comparaison", set(r) == {"barre", "profil", "masse", "statique", "methodes", "statut", "comparaison"}
        and set(r["statique"]) == {"L", "x", "troncons", "cas", "enveloppes", "fleche"} and len(r["statique"]["x"]) == len(r["statique"]["enveloppes"]["ec3"]["M_max"])
        and [a["R"] for a in r["statique"]["cas"][0]["reactions"]] == [2.42 * 3 + 5.0] * 2, sorted(r["statique"]))
    plantes, n = [], 0
    for i, p in enumerate(CAT.profils()):
        if p["famille"] not in SE.FAMILLES or SE.FAMILLES[p["famille"]][0] not in ("I",) + SE.TUBES or i % 12:
            continue
        try:
            json.dumps(MO.verifier(barre(profil=p["nom"], nuance="S355")), allow_nan=False)
            n += 1
        except Exception as e:  # noqa: BLE001
            plantes.append((p["nom"], type(e).__name__, str(e)[:80]))
    chk(f"un profilé sur douze du catalogue ({n}), la barre par défaut (6 m, G + Q, poids propre) : aucun plantage, JSON strict", n > 80 and not plantes, plantes[:3])

    print("== 7. mutations")
    for fichier, nom, ancien, nouveau in MUTATIONS:
        mods = charger((fichier, ancien, nouveau))
        if mods is None:
            chk(f"mutation « {nom} » : le texte à muter est dans le moteur, une seule fois", False, "texte absent ou répété")
            continue
        try:
            rouges = [n_ for n_, ok, _ in controles(mods) if not ok]
        except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
            rouges = [f"exception {type(e).__name__}: {e}"]
        chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")

    print("== 8. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab', 'PIL', 'fitz'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from acier.barre import moteur as M, entree as E\n"
            "r = M.verifier(E.nouvelle_barre())\n"
            "print(r['statut'], sorted(m for m in sys.modules if m.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab')))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le moteur se charge et calcule sans streamlit, pandas, matplotlib, scipy, reportlab (refusés ; numpy ne sert qu'au catalogue)",
        p.returncode == 0 and p.stdout.strip().endswith("[]"), (p.stdout.strip(), p.stderr[-300:]))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
