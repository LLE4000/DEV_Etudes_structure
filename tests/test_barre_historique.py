# -*- coding: utf-8 -*-
"""Barres en acier — la MÉTHODE HISTORIQUE (acier/barre/historique.py) : ancienne norme belge NBN B 51-001, contraintes
admissibles. La spécification et ses sources : docs/acier/BARRES_ACIER.md.

Il n'y a pas d'écran dont reproduire les chiffres : la référence, ce sont les SIX CLASSEURS « Flambement » du bureau et
sa documentation théorique (26 pages). Ils ne sont pas dans le dépôt (public) ; leurs données et leurs résultats sont
recopiés ici, cellule par cellule, depuis les fichiers reçus le 14/11/2026.

  1. La norme : les 108 valeurs du tableau φ(λ̄) (doc p. 5), λ_E des quatre nuances, le coefficient C, le choix des
     courbes (doc p. 4).
  2. Les QUATRE classeurs récents et les DEUX anciens, à 10⁻⁹ près (données → chaque cellule de résultat) ; le choix
     automatique des courbes redonne celui de l'utilisateur du classeur.
  3. Les cinq EXEMPLES de la doc (p. 6, 8, 14, 23, 25), recalculés sans arrondi intermédiaire (valeurs écrites ici).
  4. Ce que le module CORRIGE ou AJOUTE (docs § 4.3) : P_cr avec u ; u et L par axe ; φ_min ; une seule courbe par
     axe ; les garde-fous (λ̄ > 1,8, λ > 200, PE ≤ 0 et l'inertie minimale, nuance) ; C ; le déversement, sans la
     dispense ; la traction et la flexion simple ; les formes non couvertes ; la tolérance de 5 % éteinte.
  5. Les FORMULES : chaque bloc du moteur se recalcule par son gabarit (acier/barre/formules.py) à 10⁻⁹.
  6. Le CATALOGUE : chaque I, H et tube se calcule ; la sortie est du JSON strict ; ``verifier`` ne modifie rien.
  7. MUTATIONS : chaque altération plausible du moteur fait rougir au moins un contrôle des § 1 à 4.
  8. PURETÉ : le moteur se charge et calcule sans streamlit, pandas, matplotlib, scipy, reportlab.

    /usr/bin/python3.12 tests/test_barre_historique.py
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

from acier.barre import formules as FO  # noqa: E402
from acier.barre import historique as HI  # noqa: E402
from acier.barre import section as SE  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402

OK, KO = [], []
CHEMIN_MOTEUR = os.path.join(RACINE, "acier", "barre", "historique.py")


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ------------------------------------------------------------------ 1. le tableau de la doc p. 5 (lu sur la numérisation)
TABLEAU = """0.05 0.999 0.999 0.998|0.10 0.998 0.996 0.993|0.15 0.995 0.992 0.985|0.20 0.990 0.985 0.973|0.25 0.985 0.976 0.958
0.30 0.978 0.966 0.940|0.35 0.969 0.953 0.919|0.40 0.959 0.938 0.896|0.45 0.947 0.920 0.870|0.50 0.933 0.900 0.843
0.55 0.917 0.878 0.814|0.60 0.899 0.853 0.783|0.65 0.878 0.826 0.752|0.70 0.854 0.797 0.720|0.75 0.828 0.766 0.688
0.80 0.799 0.734 0.657|0.85 0.768 0.701 0.626|0.90 0.736 0.667 0.595|0.95 0.702 0.634 0.566|1.00 0.668 0.601 0.537
1.05 0.634 0.569 0.510|1.10 0.600 0.537 0.484|1.15 0.567 0.508 0.459|1.20 0.536 0.479 0.436|1.25 0.505 0.453 0.413
1.30 0.477 0.427 0.392|1.35 0.450 0.404 0.373|1.40 0.425 0.382 0.354|1.45 0.402 0.361 0.337|1.50 0.380 0.342 0.320
1.55 0.359 0.324 0.305|1.60 0.340 0.307 0.290|1.65 0.323 0.291 0.277|1.70 0.306 0.277 0.264|1.75 0.291 0.263 0.252
1.80 0.276 0.251 0.241"""
LIGNES_TABLEAU = [tuple(float(x) for x in li.split()) for bloc in TABLEAU.splitlines() for li in bloc.split("|")]
LAMBDA_E_DOC = {235: 93.9, 255: 90.2, 295: 83.8, 355: 76.4}


# ------------------------------------------------------------------ 2. les classeurs (unités des classeurs)
def section_classeur(c):
    """La section d'un classeur : cm, cm⁴, cm³ → N et mm."""
    return SE.saisie(c["forme"], c["fabrication"], nom=c["nom"], h=c.get("h"), b=c.get("b"), tf=c.get("tf"), tw=c.get("tw"),
                     t=c.get("t"), A=c["A"], Iy=c["I"][0] * 1e4, Iz=c["I"][1] * 1e4, iy=c["i"][0] * 10, iz=c["i"][1] * 10,
                     Wel_y=c["W"][0] * 1e3, Wel_z=c["W"][1] * 1e3, A_ame=c.get("ame"), A_ailes=c.get("ailes"))


def barre_classeur(c, **k):
    """La barre d'un classeur : P en N, M en N·m, L en cm ; sans déversement (les classeurs n'en ont pas)."""
    b = {"section": section_classeur(c), "fy": c["fy"], "s": 1.5, "P": c["P"], "My": c.get("My", 0.0) * 1e3, "Mz": c.get("Mz", 0.0) * 1e3,
         "axes": {"y": {"L": c["L"] * 10.0, "u": 1.0}, "z": {"L": c["L"] * 10.0, "u": 1.0}}, "deversement": {"actif": False}}
    b.update(k)
    return b


# I, i, W : (axe fort, axe faible). Les quatre classeurs RÉCENTS.
RECENTS = {
    "HEB 200": dict(nom="HEB 200 (classeur)", forme="I", fabrication="lamine", h=200, b=200, tf=15, tw=9, A=7810.0, I=(5696, 2003), i=(8.54, 5.07),
                    W=(569.6, 200.3), ame=2483.0, ailes=6000.0, fy=235.0, P=432720.0, L=720.0, courbes=("b", "c"),
                    attendu=dict(lE=93.9129729381402, lam=(84.3091334894614, 142.011834319527), lb=(0.897736817947348, 1.51216418644386),
                                 Ap=(0.946485992441219, 1.71649276023813), Bp=(0.594777368992093, 0.862063478589805),
                                 phi=(0.668818545805598, 0.316435900170424), Padm=(818344.078696203, 387180.419585192),
                                 reserve=(1.89116305855103, 0.894759705086873), Pcr=(2277319.83032543, 800820.15803052),
                                 PE=(0.714980745630601, 0.189480942142739), sP=(82.8414376847553, 175.093565094615), sM=(0.0, 0.0),
                                 sigma=175.093565094615, sigma_eq=175.093565094615)),
    "HEA 120": dict(nom="HEA 120 (classeur)", forme="I", fabrication="lamine", h=114, b=120, tf=8, tw=5, A=2534.0, I=(606, 231), i=(4.89, 3.02),
                    W=(106, 38.5), ame=846.0, ailes=960.0, fy=235.0, P=112000.0, L=100.0, courbes=("b", "c"),
                    attendu=dict(lE=93.9129729381402, lam=(20.4498977505112, 33.112582781457), lb=(0.217753704421448, 0.352587951861219),
                                 phi=(0.982155599733816, 0.918065629832997), Padm=(389909.225390327, 364465.934606168),
                                 reserve=(3.48133236955649, 3.25416013041221), Pcr=(12560058.5608263, 4787745.09496845),
                                 PE=(0.98662426618583, 0.964910412591398), sP=(45.0019274334974, 48.1435025899667), sM=(0.0, 0.0),
                                 sigma=48.1435025899667, sigma_eq=48.1435025899667)),
    # le tube 168,3 × 6 : M de l'axe faible = 0,05·P + poids propre (78 600·A/10⁶ N/m)·L²/8 = 1750 + 1213,191 N·m
    "CHS 168,3 × 6": dict(nom="CHS 168,3 × 6 (classeur)", forme="tube_rond", fabrication="creux_froid", t=6, A=2520.0, I=(1009, 1009),
                          i=(5.74, 5.74), W=(1009 / (16.83 / 2), 1009 / (16.83 / 2)), ame=1260.0, ailes=1260.0, fy=235.0, P=35000.0,
                          L=700.0, Mz=35000 * 0.05 + 78600 * 2520 / 1e6 * 7.0 ** 2 / 8, courbes=("c", "c"),
                          attendu=dict(lE=93.9129729381402, lam=(121.951219512195, 121.951219512195), lb=(1.29855562758644, 1.29855562758644),
                                       phi=(0.393034042848461, 0.393034042848461), Padm=(155169.840116573, 155169.840116573),
                                       reserve=(4.43342400333065, 4.43342400333065), Pcr=(426789.893172821, 426789.893172821),
                                       PE=(0.876988652168619, 0.876988652168619), sP=(35.3376231438657, 35.3376231438657),
                                       sM=(0.0, 28.1791978417059), sigma=63.5168209855716, sigma_eq=63.5168209855716)),
    "Tube Ø610 × 12,5": dict(nom="Tube Ø610 × 12,5 (classeur)", forme="tube_rond", fabrication="creux_chaud", t=12.5, A=23464.0,
                             I=(104754, 104754), i=(21.13, 21.13), W=(3506, 3506), ame=11732.0, ailes=11732.0, fy=450.0, P=5000000.0,
                             L=1000.0, courbes=("a", "a"),
                             attendu=dict(lE=67.8661578931284, lam=(47.3260766682442, 47.3260766682442), lb=(0.697344274929642, 0.697344274929642),
                                          phi=(0.855232757982717, 0.855232757982717), Padm=(6020154.42999194, 6020154.42999194),
                                          reserve=(1.20403088599839, 1.20403088599839), Pcr=(21711491.328066, 21711491.328066),
                                          PE=(0.654560808989436, 0.654560808989436), sP=(249.163043480599, 249.163043480599), sM=(0.0, 0.0),
                                          sigma=249.163043480599, sigma_eq=249.163043480599)),
}
# Les deux ANCIENS classeurs : un seul plan de flexion (« fort »), la contrainte y prend φ de l'axe FORT. I_z n'y est pas
# saisi : pris à A·i_z² (il n'entre dans aucune cellule comparée).
ANCIENS = {
    "FLAMBECBD (HEA 160)": dict(nom="HEA 160 (FLAMBECBD)", forme="I", fabrication="lamine", h=152, b=160, tf=9, tw=6, A=3877.0,
                                I=(615.6, 3877 * 3.98 ** 2 / 100), i=(3.98, 3.98), W=(76.95, 76.95), fy=235.0, P=300000.0, L=175.0,
                                My=3000.0, courbes=("b", "c"),
                                attendu=dict(lb=(0.468197820499131, 0.468197820499131), phi=(0.913104362989606, 0.860491946411508),
                                             Padm=(554616.546398677, 522659.939943862), Pcr=4166213.80752727, PE=0.891988260615198,
                                             sP=84.7432344115728, sM=43.7072509776528, sigma_classeur=128.450485389226, sigma_module=133.63)),
    "FLAMBEBAD (IPE 270)": dict(nom="IPE 270 (FLAMBEBAD)", forme="I", fabrication="lamine", h=270, b=135, tf=10.2, tw=6.6, A=4594.0,
                                I=(5790, 4594 * 3.02 ** 2 / 100), i=(11.23, 3.02), W=(428, 62.2), fy=355.0, P=185000.0, L=340.0,
                                My=361 * 3.4 ** 2 / 8, courbes=("a", "b"),
                                attendu=dict(lb=(0.396235896401517, 1.47342023728114), phi=(0.960096134130251, 0.351913857981606),
                                             Padm=(1043861.32151267, 382617.169044308), Pcr=10381013.8332911, PE=0.973268506866827,
                                             sP=41.9436302802048, sM=1.25227182465356, sigma_classeur=43.1959021048583, sigma_module=115.68)),
}


# ------------------------------------------------------------------ 3. les exemples de la doc (sections de la doc)
def hea(nom, h, b, tw, tf, A, Iy, iy, iz, Wy, Iz=None, Wz=None):
    """Un HEA avec les valeurs que la DOC imprime (cm, cm², cm⁴, cm³)."""
    return SE.saisie("I", "lamine", nom=nom, h=h, b=b, tw=tw, tf=tf, A=A * 100, Iy=Iy * 1e4, Iz=(Iz or A * iz ** 2) * 1e4, iy=iy * 10, iz=iz * 10,
                     Wel_y=Wy * 1e3, Wel_z=(Wz or 1.0) * 1e3)


HEA200_DOC = dict(nom="HEA 200 (doc p. 23)", h=190, b=200, tw=6.5, tf=10, A=53.8, Iy=3692, iy=8.28, iz=4.98, Wy=389)
HEA220_DOC = dict(nom="HEA 220 (doc p. 14)", h=210, b=220, tw=7, tf=11, A=64.3, Iy=5410, iy=9.17, iz=5.51, Wy=515)
HEA300_DOC = dict(nom="HEA 300 (doc p. 24)", h=290, b=300, tw=8.5, tf=14, A=112.5, Iy=18263, iy=12.74, iz=7.49, Wy=1260)


def controles(H=HI):
    """Les contrôles des § 1 à 4 : [(nom, condition, info)] — la même liste sert au rapport et aux mutations.
    ``H`` : le moteur (celui du dépôt, ou une copie mutée)."""
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    # ---- 1. la norme
    ecarts = [(lb, n, lu, round(H.phi(lb, n), 4)) for lb, a, b, cc in LIGNES_TABLEAU for n, lu in (("a", a), ("b", b), ("c", cc))
              if abs(round(H.phi(lb, n), 3) - lu) > 1e-9]
    c("doc p. 5 : les 108 valeurs du tableau φ(λ̄), courbes a, b, c, de 0,05 à 1,80, au millième", len(LIGNES_TABLEAU) == 36 and not ecarts, ecarts[:4])
    c("doc p. 5 : λ_E = 93,9 ; 90,2 ; 83,8 ; 76,4 pour AE235, 255, 295, 355",
      all(round(H.lambda_E(fy), 1) == v for fy, v in LAMBDA_E_DOC.items()), {fy: H.lambda_E(fy) for fy in LAMBDA_E_DOC})
    c("doc p. 20 : C = √(0,3·(1 + ψ²) + 0,4·ψ) vaut 1 pour ψ = 1 et 0,447 pour ψ = −1 ; doc p. 21 : 0,85 sous charges ; 1 par défaut",
      proche(H.coefficient_C("moments", 1.0), 1.0) and proche(H.coefficient_C("moments", -1.0), math.sqrt(0.2))
      and proche(H.coefficient_C("moments", 0.0), math.sqrt(0.3)) and H.coefficient_C("charges") == 0.85 and H.coefficient_C() == 1.0)
    attendus = {("I", "lamine", 300, 150): ("a", "b"), ("I", "lamine", 200, 200): ("b", "c"), ("I", "lamine", 240, 200): ("b", "c"),
                ("I", "lamine", 241, 200): ("a", "b"), ("I", "soude_oxycoupe", 300, 300): ("a", "b"), ("I", "soude_larges_plats", 300, 300): ("b", "c"),
                ("I", "lamine_semelles_additionnelles", 300, 300): ("b", "a"), ("I", "recuit", 300, 300): ("a", "b"),
                ("tube_rond", "creux_chaud", None, None): ("a", "a"), ("tube_carre", "creux_froid", None, None): ("c", "c"),
                ("autre", "caisson", None, None): ("b", "b"), ("autre", "caisson_recuit", None, None): ("a", "a"),
                ("U", "lamine", None, None): ("c", "c"), ("L", "lamine", None, None): ("c", "c"), ("autre", "lamine", None, None): ("c", "c")}
    faux = []
    for (forme, fab, h, b), att in attendus.items():
        s = SE.saisie(forme, fab, h=h, b=b, A=1000.0, Iy=1e6, Iz=1e5, iy=30.0, iz=10.0)
        r = H.courbes(s)
        if (r["y"], r["z"]) != att:
            faux.append((forme, fab, h, b, r["y"], r["z"]))
    c("doc p. 4 : la courbe de chaque profil (axe fort, axe faible) — I laminé selon h/b (limite 1,2 comprise), soudé, recuit, "
      "tubes, caissons, té, U, cornière", not faux, faux)

    # ---- 2. les classeurs récents
    for nom, cl in RECENTS.items():
        r = H.verifier(barre_classeur(cl))
        g, a = r["grandeurs"], cl["attendu"]
        ok = (proche(g["lambda_E"], a["lE"]) and all(proche(g[ax]["lambda"], a["lam"][k]) and proche(g[ax]["lambda_bar"], a["lb"][k])
                                                     and proche(g[ax]["phi"], a["phi"][k]) and proche(g[ax]["P_adm"], a["Padm"][k])
                                                     and proche(g[ax]["reserve"], a["reserve"][k]) and proche(g[ax]["P_cr"], a["Pcr"][k])
                                                     and proche(g[ax]["PE"], a["PE"][k]) for k, ax in enumerate("yz"))
              and ("Ap" not in a or all(proche(g[ax]["A_prime"], a["Ap"][k]) and proche(g[ax]["B_prime"], a["Bp"][k]) for k, ax in enumerate("yz")))
              and proche(cl["P"] / (cl["A"] * g["y"]["phi"]), a["sP"][0]) and proche(g["sigma_P"], a["sP"][1])
              and proche(g["sigma_My"], a["sM"][0]) and proche(g["sigma_Mz"], a["sM"][1])
              and proche(g["sigma"], a["sigma"]) and proche(g["sigma_eq"], a["sigma_eq"]))
        c(f"classeur {nom} : λ, λ̄, A′, B′, φ, P_adm, réserve, P_cr, PE des deux axes, termes P et M, σ et σ_eq à 10⁻⁹",
          ok, (g["y"]["phi"], g["z"]["phi"], g.get("sigma")))
        c(f"classeur {nom} : le choix AUTOMATIQUE des courbes (doc p. 4) est celui de l'utilisateur du classeur {cl['courbes']}",
          (g["y"]["courbe"], g["z"]["courbe"]) == cl["courbes"], (g["y"]["courbe"], g["z"]["courbe"]))
    r = H.verifier(barre_classeur(RECENTS["HEB 200"]))
    vz = next(v for v in r["verifications"] if v["id"] == "H-N-z")
    c("classeur HEB 200 : le verdict que le classeur ne rend pas — P = 432,7 kN > P_adm,z = 387,2 kN, taux 1,118, NON CONFORME ; "
      "l'axe faible gouverne ; en compression simple la ligne des contraintes n'est pas répétée",
      r["statut"] == H.NOK and r["gouvernant"] == "H-N-z" and proche(vz["taux"], 432720 / 387180.419585192) and round(vz["taux"], 3) == 1.118
      and [v["id"] for v in r["verifications"]] == ["H-N-z", "H-N-y"] and proche(r["grandeurs"]["sigma"] / r["grandeurs"]["sigma_adm"], vz["taux"]),
      (r["statut"], r["gouvernant"], vz["taux"], [v["id"] for v in r["verifications"]]))
    r = H.verifier(barre_classeur(RECENTS["Tube Ø610 × 12,5"]))
    c("classeur du tube Ø610 : f_y = 450 N/mm² est hors des nuances AE235 à AE355 de la doc — DIT (extrapolation) ; conforme à 83 %",
      any(x["code"] == "nuance" for x in r["alertes"]) and r["statut"] == H.OK and round(r["taux_max"], 3) == 0.831, (r["alertes"], r["taux_max"]))

    # ---- 2 bis. les anciens classeurs
    for nom, cl in ANCIENS.items():
        r = H.verifier(barre_classeur(cl))
        g, a = r["grandeurs"], cl["attendu"]
        ok = (all(proche(g[ax]["lambda_bar"], a["lb"][k]) and proche(g[ax]["phi"], a["phi"][k]) and proche(g[ax]["P_adm"], a["Padm"][k])
                  for k, ax in enumerate("yz")) and proche(g["y"]["P_cr"], a["Pcr"]) and proche(g["y"]["PE"], a["PE"])
              and proche(cl["P"] / (cl["A"] * g["y"]["phi"]), a["sP"]) and proche(g["sigma_axe_y"] - a["sP"], a["sM"], 1e-8)
              and proche(g["sigma_axe_y"], a["sigma_classeur"]))
        c(f"ancien classeur {nom} : λ̄, φ, P_adm, P_cr, PE, termes P et M, σ du classeur (φ de l'axe FORT) à 10⁻⁹", ok,
          (g["y"]["phi"], g["z"]["phi"], g["sigma_axe_y"]))
        c(f"ancien classeur {nom} : le module prend φ_min (l'axe faible) — σ = {a['sigma_module']} N/mm² au lieu de "
          f"{a['sigma_classeur']:.2f} (correction 3)", abs(g["sigma"] - a["sigma_module"]) < 0.01 and g["phi_min"] == g["z"]["phi"]
          and g["sigma"] > g["sigma_axe_y"], g["sigma"])

    # ---- 3. les exemples de la doc
    caisson = SE.saisie("autre", "caisson", nom="caisson 100 × 100 × 1 cm", A=400e2, Iy=(101 ** 4 - 99 ** 4) / 12 * 1e4,
                        Iz=(101 ** 4 - 99 ** 4) / 12 * 1e4, iy=math.sqrt((101 ** 4 - 99 ** 4) / 12 / 400) * 10, iz=math.sqrt((101 ** 4 - 99 ** 4) / 12 / 400) * 10)
    r = H.verifier({"section": caisson, "fy": 235.0, "P": 4000e3, "axes": {"y": {"L": 20000.0}, "z": {"L": 20000.0}}})
    g = r["grandeurs"]
    c("doc p. 6, caisson soudé non recuit, 20 m, 4000 kN : courbe b, λ̄ = 0,522, φ = 0,891, P_adm = 5582 kN (la doc, arrondie : 0,54 ; 0,88 ; "
      "5500) — conforme", g["z"]["courbe"] == "b" and round(g["z"]["lambda_bar"], 3) == 0.522 and round(g["z"]["phi"], 3) == 0.891
      and round(g["z"]["P_adm"] / 1e3) == 5582 and r["statut"] == H.OK, (g["z"]["lambda_bar"], g["z"]["phi"], g["z"]["P_adm"]))
    corniere = SE.saisie("L", "lamine", nom="L 60 × 60 × 8", A=903.0, Iy=903 * 22.9 ** 2, Iz=903 * 11.7 ** 2, iy=22.9, iz=11.7)
    r = H.verifier({"section": corniere, "fy": 235.0, "P": 30e3, "axes": {"y": {"L": 2000.0}, "z": {"L": 2000.0}}})
    g = r["grandeurs"]
    c("doc p. 8, cornière L 60 × 60 × 8, 2 m, 30 kN : courbe c, λ = 170,9, λ̄ = 1,820, φ = 0,236, P_adm = 33,4 kN (la doc : 171 ; 1,82 ; "
      "0,23 ; 32,6) ; λ̄ > 1,8 est DIT", g["z"]["courbe"] == "c" and round(g["z"]["lambda"], 1) == 170.9 and round(g["z"]["lambda_bar"], 3) == 1.820
      and round(g["z"]["phi"], 3) == 0.236 and round(g["z"]["P_adm"] / 1e3, 1) == 33.4 and any(x["code"] == "tableau_z" for x in r["alertes"]),
      (g["z"]["lambda"], g["z"]["phi"], g["z"]["P_adm"], r["alertes"]))
    # p. 14 : flexion composée dans le seul plan vertical — la doc ne regarde que l'axe fort (φ_b = 0,86) : le flambement
    # de l'axe faible est écarté ici (L_z = 0), comme le fait l'énoncé
    r = H.verifier({"section": hea(**HEA220_DOC), "fy": 235.0, "P": 500e3, "My": 25e6, "axes": {"y": {"L": 5000.0}, "z": {"L": 0.0}},
                    "deversement": {"actif": False}})
    g = r["grandeurs"]
    c("doc p. 14, HEA 220, 5 m, P = 500 kN, M = 25 kN·m (plan vertical seul) : P_cr = 4485 kN, φ_b = 0,863, σ = 90,1 + 58,3 = 148,4 N/mm² "
      "(la doc : 4496 ; 0,86 ; 90,4 + 58,3 = 149)", round(g["y"]["P_cr"] / 1e3) == 4485 and round(g["y"]["phi"], 3) == 0.863
      and round(g["sigma_P"], 1) == 90.1 and round(g["sigma_My"], 1) == 58.3 and round(g["sigma"], 1) == 148.4 and r["statut"] == H.OK,
      (g["y"]["P_cr"], g["y"]["phi"], g["sigma_P"], g["sigma_My"]))
    bc = {"section": hea(**HEA200_DOC), "fy": 235.0, "P": 38e3, "My": 48e6, "axes": {"y": {"L": 8000.0}, "z": {"L": 8000.0}}}
    r = H.verifier(dict(bc, deversement={"actif": False}))
    g = r["grandeurs"]
    c("doc p. 23, poutre HEA 200, 8 m, P = 38 kN, M = 48 kN·m, AVEC la dispense de la doc (C/φ_ed = 1) : φ_c = 0,2614, P_cr = 1196 kN, "
      "PE = 0,9523, σ = 27,0 + 129,6 = 156,6 N/mm² (la doc : 0,264 ; 1196 ; 0,95 ; 26 + 130 = 156)",
      round(g["z"]["phi"], 4) == 0.2614 and round(g["y"]["P_cr"] / 1e3) == 1196 and round(g["y"]["PE"], 4) == 0.9523
      and round(g["sigma_P"], 1) == 27.0 and round(g["sigma_My"], 1) == 129.6 and round(g["sigma"], 1) == 156.6,
      (g["z"]["phi"], g["y"]["P_cr"], g["y"]["PE"], g["sigma"]))
    r = H.verifier(bc)
    g = r["grandeurs"]
    c("… le même SANS la dispense (annotée « FAUX » p. 20 ; le module ne la reprend pas) : sur 8 m sans maintien, r_d = 55,2 mm, "
      "λ̄_ed = 1,542, φ_ed = 0,307 → σ_My = 422 N/mm², NON CONFORME ; P/(A·φ) = 27 N/mm² est pourtant > 0,15·σ_adm = 23,5",
      g["deversement"]["cas"] == "I" and round(g["deversement"]["rd"], 1) == 55.2 and round(g["deversement"]["lambda_bar"], 3) == 1.542
      and round(g["deversement"]["phi"], 3) == 0.307 and round(g["sigma_My"]) == 422 and r["statut"] == H.NOK,
      (g["deversement"], g["sigma_My"]))
    ab = {"section": hea(**HEA300_DOC), "fy": 235.0, "P": 168e3, "My": 110e6, "axes": {"y": {"L": 6000.0, "u": 1.5}, "z": {"L": 6000.0, "u": 1.5}},
          "C": {"y": {"mode": "charges"}}, "deversement": {"actif": True, "Lc": 6000.0}}
    r = H.verifier(ab)
    g = r["grandeurs"]
    dv = g["deversement"]
    c("doc p. 24-25, colonne HEA 300, 6 m, u = 1,5, P = 168 kN, M = 110 kN·m, C = 0,85 : φ_c = 0,4009, I_d = 3150 cm⁴, A_d = 45,7 cm², "
      "r_d = 8,30 cm, λ̄_ed = 0,770, φ_ed = 0,676, P_cr = 4673 kN, PE = 0,9461, σ = 37,3 + 116,0 = 153,3 N/mm² (la doc : 0,4 ; 3150 ; 45,7 ; "
      "8,3 ; 0,77 ; 0,7 ; 4673 ; 0,95 ; 37,3 + 111,5 = 148,8) — conforme",
      round(g["z"]["phi"], 4) == 0.4009 and round(dv["Id"] / 1e4) == 3150 and round(dv["Ad"] / 100, 1) == 45.7 and round(dv["rd"] / 10, 2) == 8.30
      and round(dv["lambda_bar"], 3) == 0.770 and round(dv["phi"], 3) == 0.676 and round(g["y"]["P_cr"] / 1e3) == 4673
      and round(g["y"]["PE"], 4) == 0.9461 and round(g["sigma_P"], 1) == 37.3 and round(g["sigma_My"], 1) == 116.0
      and round(g["sigma"], 1) == 153.3 and r["statut"] == H.OK and g["C_y"] == 0.85,
      (g["z"]["phi"], dv, g["y"]["P_cr"], g["y"]["PE"], g["sigma_P"], g["sigma_My"]))
    c("doc p. 19 : la courbe du déversement est celle de l'axe FAIBLE du profil (HEA 300 : c)", dv["courbe"] == "c" == g["z"]["courbe"])

    # ---- 4. ce que le module corrige ou ajoute
    heb = RECENTS["HEB 200"]
    r1 = H.verifier(barre_classeur(heb, P=100e3, Mz=5e6, axes={"y": {"L": 7200.0, "u": 1.0}, "z": {"L": 3600.0, "u": 2.0}}))
    r2 = H.verifier(barre_classeur(heb, P=100e3, Mz=5e6, axes={"y": {"L": 7200.0, "u": 1.0}, "z": {"L": 7200.0, "u": 1.0}}))
    c("correction 1 — P_cr = π²·E·I/(u·L)² : u = 2 sur 3,60 m vaut u = 1 sur 7,20 m (P_cr, PE, σ égaux) ; le classeur, qui oublie u, "
      "rendrait P_cr quatre fois trop fort", proche(r1["grandeurs"]["z"]["P_cr"], r2["grandeurs"]["z"]["P_cr"])
      and proche(r1["grandeurs"]["z"]["P_cr"], math.pi ** 2 * 210000 * 2003e4 / 7200 ** 2) and proche(r1["grandeurs"]["sigma"], r2["grandeurs"]["sigma"]),
      (r1["grandeurs"]["z"]["P_cr"], r2["grandeurs"]["z"]["P_cr"]))
    r = H.verifier(barre_classeur(heb, P=100e3, axes={"y": {"L": 9000.0, "u": 1.0}, "z": {"L": 3000.0, "u": 0.7}}))
    g = r["grandeurs"]
    c("correction 2 — u et L par axe : axe fort 9 m, u = 1 ; axe faible 3 m, u = 0,7", proche(g["y"]["lambda"], 9000 / 85.4)
      and proche(g["z"]["lambda"], 0.7 * 3000 / 50.7) and proche(g["z"]["P_cr"], math.pi ** 2 * 210000 * 2003e4 / 2100 ** 2), (g["y"]["lambda"], g["z"]["lambda"]))
    r = H.verifier(barre_classeur(heb, P=300e3, My=20e6, axes={"y": {"L": 12000.0, "u": 1.0}, "z": {"L": 2000.0, "u": 1.0}}))
    g = r["grandeurs"]
    c("correction 3 — φ_min : quand l'axe FORT est le plus élancé (12 m contre 2 m), c'est SON φ qui divise P (le classeur prenait "
      "toujours l'axe faible)", g["y"]["phi"] < g["z"]["phi"] and g["phi_min"] == g["y"]["phi"] and g["axe_phi_min"] == "y"
      and proche(g["sigma_P"], 300e3 / (7810 * g["y"]["phi"])), (g["y"]["phi"], g["z"]["phi"], g["phi_min"]))
    r = H.verifier(barre_classeur(heb, axes={"y": {"L": 7200.0, "courbe": "a"}, "z": {"L": 7200.0}}))
    refus = []
    for mauvais in ("d", "ab", "", 1):
        try:
            H.verifier(barre_classeur(heb, axes={"y": {"L": 7200.0, "courbe": mauvais}, "z": {"L": 7200.0}}))
            refus.append(mauvais)
        except H.Refus:
            pass
    c("garde-fou — une seule courbe par axe : « a » imposée à l'axe fort est prise ET signalée (la norme donne b) ; « d », « ab », "
      "« », 1 sont refusés (le classeur rendait φ = 1 sans choix et additionnait α et β avec deux cases cochées)",
      r["grandeurs"]["y"]["courbe"] == "a" and r["grandeurs"]["y"]["courbe_forcee"] and any(x["code"] == "courbe_y" for x in r["alertes"])
      and proche(r["grandeurs"]["y"]["phi"], H.phi(r["grandeurs"]["y"]["lambda_bar"], "a")) and not refus, refus)
    r = H.verifier(barre_classeur(heb, P=50e3, axes={"y": {"L": 7200.0}, "z": {"L": 10500.0}}))
    codes = {x["code"] for x in r["alertes"]}
    c("garde-fous — λ̄ = 2,21 > 1,8 (fin du tableau) et λ = 207 > 200 (doc p. 7) sont DITS, le calcul continue",
      {"tableau_z", "elancement_z"} <= codes and r["grandeurs"]["z"]["lambda"] > 200 and r["grandeurs"]["z"]["phi"] > 0, codes)
    r = H.verifier(barre_classeur(heb, P=600e3, Mz=5e6))
    g = r["grandeurs"]
    v = next(x for x in r["verifications"] if x["id"] == "H-SIGMA")
    Imin = 1.5 * 600e3 * 7200 ** 2 / (math.pi ** 2 * 210000)
    c("garde-fou — PE ≤ 0 avec un moment : ARRÊT, non conforme sans taux, et l'inertie minimale I_min = s·P·(u·L)²/(π²·E) = 2251 cm⁴ "
      "est dite (le profil en a 2003)", g["z"]["PE"] <= 0 and v["statut"] == H.NOK and v["taux"] is None and r["statut"] == H.NOK
      and r["gouvernant"] == "H-SIGMA" and proche(g["z"]["I_min"], Imin) and round(Imin / 1e4) == 2251 and "sigma" not in g
      and any("2 251" in m for m in v["messages"]), (g["z"]["PE"], v, Imin))
    r = H.verifier(barre_classeur(heb, P=600e3))
    c("… en compression SIMPLE, PE ≤ 0 n'arrête rien : P > P_adm le dit déjà (A·φ·f_y ≤ P_cr toujours), l'alerte reste",
      r["statut"] == H.NOK and r["gouvernant"] == "H-N-z" and r["taux_max"] > 1 and any(x["code"] == "second_ordre_z" for x in r["alertes"])
      and all(H.phi(lb / 100, cc) <= 1 / (lb / 100) ** 2 + 1e-12 for lb in range(5, 400, 5) for cc in "abc"))
    base = barre_classeur(heb, P=200e3, My=30e6, axes={"y": {"L": 4000.0}, "z": {"L": 4000.0}})
    s0 = H.verifier(base)["grandeurs"]
    s1 = H.verifier(dict(base, C={"y": {"mode": "charges"}}))["grandeurs"]
    s2 = H.verifier(dict(base, C={"y": {"mode": "moments", "psi": -1.0}}))["grandeurs"]
    c("coefficient C : 1 par défaut (classeurs) ; 0,85 sous charges et √0,2 pour ψ = −1 réduisent d'autant le SEUL terme de moment",
      s0["C_y"] == 1.0 and proche(s1["sigma_My"], 0.85 * s0["sigma_My"]) and proche(s2["sigma_My"], math.sqrt(0.2) * s0["sigma_My"])
      and proche(s1["sigma_P"], s0["sigma_P"]), (s0["sigma_My"], s1["sigma_My"], s2["sigma_My"]))
    try:
        H.verifier(dict(base, C={"y": {"mode": "moments", "psi": 1.5}}))
        ok = False
    except H.Refus:
        ok = True
    c("coefficient C : ψ hors de [−1 ; 1] est refusé (|M₂| ≤ |M₁|)", ok)
    r_dev = H.verifier(dict(base, deversement={"actif": True, "Lc": 4000.0}))
    r_sans = H.verifier(dict(base, deversement={"actif": False}))
    r_defaut = H.verifier({k: v for k, v in base.items() if k != "deversement"})
    gd = r_dev["grandeurs"]
    Id, Ad = 15 * 200 ** 3 / 12, 200 * 15 + (200 - 30) * 9 / 6
    c("déversement (doc p. 19) : I_d = t_f·b³/12, A_d = b·t_f + h_w·t_w/6 (h_w = h − 2·t_f), λ_ed = L_c/√(I_d/A_d), φ_ed de la courbe "
      "de l'axe faible ; il divise le SEUL terme de moment de l'axe fort", proche(gd["deversement"]["Id"], Id) and proche(gd["deversement"]["Ad"], Ad)
      and proche(gd["deversement"]["lambda"], 4000 / math.sqrt(Id / Ad)) and proche(gd["deversement"]["phi"], H.phi(4000 / math.sqrt(Id / Ad) / H.lambda_E(235), "c"))
      and proche(gd["sigma_My"], r_sans["grandeurs"]["sigma_My"] / gd["deversement"]["phi"]) and proche(gd["sigma_P"], r_sans["grandeurs"]["sigma_P"]),
      gd["deversement"])
    c("déversement : ACTIVÉ par défaut, sur la longueur de l'axe faible ; éteint (semelle maintenue en continu), φ_ed = 1 — c'est le classeur",
      proche(r_defaut["grandeurs"]["sigma"], gd["sigma"]) and r_sans["grandeurs"]["deversement"] == {"cas": "maintenu", "phi": 1.0}
      and any("maintenue en continu" in h for h in r_sans["hypotheses"]) and any("tous les 4,00 m" in h for h in r_dev["hypotheses"]))
    petit = H.verifier(dict(base, P=10e3, deversement={"actif": True, "Lc": 4000.0}))["grandeurs"]
    c("déversement : PAS de dispense quand P/(A·φ) ≤ 0,15·σ_adm (doc p. 20, « FAUX ») — φ_ed reste appliqué",
      petit["sigma_P"] <= 0.15 * petit["sigma_adm"] and petit["deversement"]["phi"] < 1 and proche(petit["deversement"]["phi"], gd["deversement"]["phi"])
      and proche(petit["sigma_My"], 30e6 / (petit["deversement"]["phi"] * 569.6e3 * petit["y"]["PE"])), (petit["sigma_P"], petit["sigma_My"]))
    rhs = SE.du_catalogue("RHS 200x100x6.3")
    rr = H.verifier({"section": rhs, "fy": 235.0, "My": 20e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}})["grandeurs"]
    lam_f = math.sqrt(math.pi * math.sqrt(2.6) * 6000 * rhs["Wel_y"] / math.sqrt(rhs["Iz"] * rhs["It"]))
    rc = H.verifier({"section": SE.du_catalogue("SHS 150x150x6.3"), "fy": 235.0, "My": 20e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}})["grandeurs"]
    rt = H.verifier({"section": SE.du_catalogue("CHS 168.3x6"), "fy": 235.0, "My": 10e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}})["grandeurs"]
    c("déversement des tubes (doc p. 17-18) : rectangulaire — λ_ed = √(π·√(E/G)·L_c·W_y/√(I_z·I_t)), E/G = 2,6, « proche de l'unité » "
      "(φ_ed = 0,98 ici) ; rond et carré — sans objet, φ_ed = 1", rr["deversement"]["cas"] == "ferme" and proche(rr["deversement"]["lambda"], lam_f)
      and 0.95 < rr["deversement"]["phi"] < 1 and rc["deversement"] == {"cas": "sans_objet", "phi": 1.0} == rt["deversement"],
      (rr["deversement"], rc["deversement"]))
    r = H.verifier(barre_classeur(heb, P=-300e3, My=20e6))
    g = r["grandeurs"]
    c("traction (hors doc, dit) : σ = |N|/A + M/W, sans φ ni second ordre ; aucune ligne de flambement",
      proche(g["sigma"], 300e3 / 7810 + 20e6 / 569.6e3) and g["phi_min"] == 1.0 and g["y"]["PE"] == 1.0
      and [v["id"] for v in r["verifications"]] == ["H-SIGMA"] and r["verifications"][0]["titre"] == "Traction et flexion"
      and any(x["code"] == "traction" for x in r["alertes"]), (g["sigma"], [v["id"] for v in r["verifications"]]))
    r = H.verifier(barre_classeur(heb, P=0.0, My=40e6, Mz=10e6, Vz=80e3, Vy=20e3))
    g = r["grandeurs"]
    tau = 80e3 / 2483 + 20e3 / 6000
    sig = 40e6 / 569.6e3 + 10e6 / 200.3e3
    c("flexion déviée et effort tranchant : σ = M_y/W_y + M_z/W_z ; τ = V_z/A_âme + V_y/A_ailes ; σ_eq = √(σ² + 3·τ²) ; deux lignes",
      proche(g["sigma"], sig) and proche(g["tau"], tau) and proche(g["sigma_eq"], math.sqrt(sig ** 2 + 3 * tau ** 2))
      and [v["id"] for v in r["verifications"]] == ["H-SIGMA", "H-EQ"] and r["verifications"][0]["titre"] == "Flexion déviée",
      (g["sigma"], g["tau"], g["sigma_eq"]))
    r = H.verifier(barre_classeur(heb, P=200e3, My=50e6, Mz=10e6, axes={"y": {"L": 4000.0}, "z": {"L": 4000.0}}, deversement={"actif": True, "Lc": 2000.0}))
    g = r["grandeurs"]
    att = 200e3 / (7810 * g["phi_min"]) + 50e6 / (g["deversement"]["phi"] * 569.6e3 * g["y"]["PE"]) + 10e6 / (200.3e3 * g["z"]["PE"])
    c("flexion composée déviée (doc p. 26) : σ = P/(A·φ_min) + C_y·M_y/(φ_ed·W_y·PE_y) + C_z·M_z/(W_z·PE_z), φ_ed sur le seul axe fort",
      proche(g["sigma"], att) and r["verifications"][-1]["titre"] == "Flexion composée déviée" and "doc p. 26" in r["verifications"][-1]["reference"], (g["sigma"], att))
    cat = H.aires_cisaillement(SE.du_catalogue("HEB 200"))
    c("aires de cisaillement d'un profilé du catalogue : « âme » = A_v,z (HEB 200 : 2483 mm², la valeur du classeur) ; « ailes » = 2·b·t_f "
      "(6000) ; tube rond : A/2 chacune (classeur du tube 168,3)", round(cat["ame"]) == 2483 and proche(cat["ailes"], 6000.0)
      and H.aires_cisaillement(SE.du_catalogue("CHS 168.3x6")) == {"ame": SE.du_catalogue("CHS 168.3x6")["A"] / 2, "ailes": SE.du_catalogue("CHS 168.3x6")["A"] / 2}, cat)
    u = H.verifier({"section": SE.du_catalogue("UPN 200"), "fy": 235.0, "P": 100e3, "My": 5e6, "axes": {"y": {"L": 3000.0}, "z": {"L": 3000.0}}})
    c("cornière, U, té : seule la compression simple est vérifiée — la flexion est dite NON COUVERTE, le statut n'est jamais « ok »",
      u["statut"] == H.NON_COUVERT and [v["statut"] for v in u["verifications"]] == [H.OK, H.OK, H.NON_COUVERT]
      and u["grandeurs"]["z"]["courbe"] == "c" and any(x["code"] == "forme" for x in u["alertes"]), [v["statut"] for v in u["verifications"]])
    tol = barre_classeur(heb, P=0.0, My=569.6e3 * 235 / 1.5 * 1.03)
    c("tolérance de 5 % (doc p. 13, 23) : ÉTEINTE par défaut — 103 % est non conforme ; allumée, la limite passe à 1,05·σ_adm et c'est DIT",
      H.verifier(tol)["statut"] == H.NOK and H.verifier(dict(tol, tolerance=True))["statut"] == H.OK
      and any(x["code"] == "tolerance" for x in H.verifier(dict(tol, tolerance=True))["alertes"]))
    refus = []
    for nom, k in (("f_y nul", dict(fy=0.0)), ("s = 0,5", dict(s=0.5)), ("L négatif", dict(axes={"y": {"L": -1.0}, "z": {"L": 100.0}})),
                   ("u nul", dict(axes={"y": {"L": 100.0, "u": 0.0}, "z": {"L": 100.0}})), ("P texte", dict(P="100")),
                   ("M infini", dict(My=float("inf"))), ("L_c négatif", dict(deversement={"Lc": -5.0})), ("mode de C inconnu", dict(C={"y": {"mode": "x"}}))):
        try:
            H.verifier(barre_classeur(heb, **k))
            refus.append(nom)
        except H.Refus as e:
            if not str(e):
                refus.append(nom + " (sans message)")
    c("domaine : f_y nul, s < 1, L négatif, u nul, texte, infini, L_c négatif, mode de C inconnu — refusés avec un message", not refus, refus)
    return C


# ------------------------------------------------------------------ 5 et 6. formules, catalogue, sortie
def cas_formules():
    heb = RECENTS["HEB 200"]
    return [barre_classeur(cl) for cl in list(RECENTS.values()) + list(ANCIENS.values())] + [
        barre_classeur(heb, P=200e3, My=50e6, Mz=10e6, Vz=60e3, Vy=10e3, axes={"y": {"L": 4000.0, "u": 1.2}, "z": {"L": 4000.0, "u": 0.8}},
                       C={"y": {"mode": "charges"}, "z": {"mode": "moments", "psi": 0.3}}, deversement={"actif": True, "Lc": 2500.0}),
        barre_classeur(heb, P=-150e3, My=30e6, Vz=40e3),
        {"section": SE.du_catalogue("RHS 200x100x6.3"), "fy": 355.0, "P": 80e3, "My": 20e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}},
        {"section": SE.du_catalogue("IPE 300"), "fy": 235.0, "My": 60e6, "Vz": 50e3, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}},
    ]


def mutations():
    """Chaque altération plausible du moteur — écrite sur son TEXTE — fait rougir au moins un contrôle de référence."""
    source = open(CHEMIN_MOTEUR, encoding="utf-8").read()
    cas = [
        ("φ sans la racine : 1/(A′ + A′)", "return 1.0 / (a + math.sqrt(a * a - b))", "return 1.0 / (a + a)"),
        ("α et β de la courbe b échangés", '"b": (0.554, 0.738)', '"b": (0.738, 0.554)'),
        ("courbe c avec le β de la courbe a", '"c": (0.532, 0.377)', '"c": (0.532, 0.795)'),
        ("λ_E avec E = 200 000", "E = 210000.0 ", "E = 200000.0 "),
        ("P_adm sans le coefficient de sécurité", "Padm = A * ph * fy / sec", "Padm = A * ph * fy"),
        ("P_cr sans u (le défaut du classeur)", "Pcr = math.pi ** 2 * E * I / Lf ** 2 if Lf > 0 else None", "Pcr = math.pi ** 2 * E * I / ax[\"L\"] ** 2 if Lf > 0 else None"),
        ("PE sans le coefficient de sécurité", "else 1.0 - sec * P / Pcr", "else 1.0 - P / Pcr"),
        ("φ de l'axe faible au lieu de φ_min (le défaut du classeur)", 'phi_min = min(G["y"]["phi"], G["z"]["phi"]) if comprime else 1.0',
         'phi_min = G["z"]["phi"] if comprime else 1.0'),
        ("limite h/b à 1,0 au lieu de 1,2", "RAPPORT_HB = 1.2", "RAPPORT_HB = 1.0"),
        ("I laminé h/b ≤ 1,2 : courbes de h/b > 1,2", 'return {"y": "b", "z": "c", "regle": "I ou H laminé, h/b ≤ 1,2", "message": None}',
         'return {"y": "a", "z": "b", "regle": "I ou H laminé, h/b ≤ 1,2", "message": None}'),
        ("caisson non recuit en courbe a", '"caisson": ("b", "b")', '"caisson": ("a", "a")'),
        ("C sous charges = 1", "C_CHARGES = 0.85", "C_CHARGES = 1.0"),
        ("C des moments d'extrémité sans le terme 0,4·ψ", "return math.sqrt(0.3 * (1.0 + psi * psi) + 0.4 * psi)", "return math.sqrt(0.3 * (1.0 + psi * psi))"),
        ("déversement : âme entière au lieu d'un sixième", 'Ad = s["b"] * s["tf"] + hw * s["tw"] / 6.0', 'Ad = s["b"] * s["tf"] + hw * s["tw"]'),
        ("déversement : inertie de la semelle en b²", 'Id = s["tf"] * s["b"] ** 3 / 12.0', 'Id = s["tf"] * s["b"] ** 2 / 12.0'),
        ("déversement : courbe de l'axe fort", 'dev = deversement(s, Lc, fy, G["z"]["courbe"])', 'dev = deversement(s, Lc, fy, G["y"]["courbe"])'),
        ("déversement oublié dans le terme de moment", 'sMy = Cy * My / (phi_ed * s["Wel_y"] * G["y"]["PE"]) if My > 0 else 0.0',
         'sMy = Cy * My / (s["Wel_y"] * G["y"]["PE"]) if My > 0 else 0.0'),
        ("déversement appliqué aussi à l'axe faible", 'sMz = Cz * Mz / (s["Wel_z"] * G["z"]["PE"]) if Mz > 0 else 0.0',
         'sMz = Cz * Mz / (phi_ed * s["Wel_z"] * G["z"]["PE"]) if Mz > 0 else 0.0'),
        ("la dispense de la doc p. 20 reprise (φ_ed = 1 si P/(A·φ) ≤ 0,15·σ_adm)", "    phi_ed = dev[\"phi\"]\n",
         "    phi_ed = 1.0 if (comprime and P / (A * phi_min) <= 0.15 * sigma_adm) else dev[\"phi\"]\n"),
        ("second ordre oublié (PE = 1)", "PE = 1.0 if (Pcr is None or not comprime) else 1.0 - sec * P / Pcr", "PE = 1.0"),
        ("C oublié dans le terme de moment fort", "sMy = Cy * My / (phi_ed", "sMy = My / (phi_ed"),
        ("tube rectangulaire : E/G = 2 au lieu de 2,6", "E_SUR_G = 2.6", "E_SUR_G = 2.0"),
        ("σ_eq avec τ² au lieu de 3·τ²", "sigma_eq = math.sqrt(sigma * sigma + 3.0 * tau * tau)", "sigma_eq = math.sqrt(sigma * sigma + tau * tau)"),
        ("aires de cisaillement échangées", 'tz = Vz / aires["ame"] if Vz > 0 else 0.0', 'tz = Vz / aires["ailes"] if Vz > 0 else 0.0'),
        ("tolérance de 5 % allumée par défaut", '"tolerance": bool(d.get("tolerance", False))', '"tolerance": bool(d.get("tolerance", True))'),
        ("déversement éteint par défaut", '"actif": bool(dev.get("actif", True))', '"actif": bool(dev.get("actif", False))'),
        ("alerte λ̄ > 1,8 retirée", "if lb > LAMBDA_BAR_TABLEAU:\n                alerte(f\"tableau_{a}\"", "if False:\n                alerte(f\"tableau_{a}\""),
        ("arrêt PE ≤ 0 retiré", 'instable = comprime and ((My > 0 and G["y"]["PE"] <= 0) or (Mz > 0 and G["z"]["PE"] <= 0))', "instable = False"),
        ("une forme non couverte rendue « ok »", "statut = NON_COUVERT            #", "statut = OK            #"),
    ]
    res = []
    for nom, ancien, nouveau in cas:
        if source.count(ancien) != 1:
            res.append((nom, None))
            continue
        spec = importlib.util.spec_from_file_location("acier.barre.historique_mute", CHEMIN_MOTEUR)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "acier.barre"
        try:
            exec(compile(source.replace(ancien, nouveau), CHEMIN_MOTEUR, "exec"), mod.__dict__)
            rouges = [n for n, ok, _ in controles(mod) if not ok]
        except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
            rouges = [f"exception {type(e).__name__}: {e}"]
        res.append((nom, rouges))
    return res


def main():
    print("== 1 à 4. la norme, les classeurs, la doc, les corrections")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 5. les formules")
    faux, n = [], 0
    for b in cas_formules():
        r = HI.verifier(b)
        cles = {f["cle"] for f in r["calcul"]}
        for f in r["calcul"]:
            n += 1
            if f["gabarit"] not in FO.GABARITS:
                faux.append((b["section"]["nom"], f["cle"], "gabarit absent"))
            elif not proche(FO.evaluer(f), f["resultat"]):
                faux.append((b["section"]["nom"], f["cle"], FO.evaluer(f), f["resultat"]))
        for v in r["verifications"]:
            manquent = [k for k in v["formules"] if k not in cles]
            if manquent:
                faux.append((b["section"]["nom"], v["id"], "formules absentes", manquent))
    chk(f"chaque bloc de formule du moteur se recalcule par son gabarit, sur ses seuls termes, à 10⁻⁹ ({n} formules, {len(cas_formules())} barres) ; "
        "chaque vérification nomme des formules qui existent", n > 150 and not faux, faux[:4])
    utilises = {f["gabarit"] for b in cas_formules() for f in HI.verifier(b)["calcul"]}
    chk("chaque gabarit de la méthode historique sert à un contrôle (aucun n'est mort)", utilises == set(FO.GABARITS), set(FO.GABARITS) ^ utilises)
    r = HI.verifier(barre_classeur(RECENTS["HEB 200"]))
    pcr = next(f for f in r["calcul"] if f["cle"] == "P_cr_z")
    chk("la charge critique s'écrit comme dans le classeur : π²·E·I/(u·L)² avec E = 21 000 kN/cm², I en cm⁴, L en cm → 800,8 kN",
        [(t["symbole"], t["valeur"], t["unite"]) for t in pcr["termes"]] == [("E", 21000.0, "kN/cm²"), ("I_z", 2003.0, "cm⁴"), ("u", 1.0, ""), ("L", 720.0, "cm")]
        and round(pcr["resultat"], 1) == 800.8, pcr)

    print("== 6. le catalogue, la sortie")
    plantes, n, formes = [], 0, set()
    for p in CAT.profils():
        if p["famille"] not in SE.FAMILLES:
            continue
        s = SE.du_catalogue(p["nom"])
        try:
            r = HI.verifier({"section": s, "fy": 235.0, "P": 0.05 * s["A"] * 235, "My": 0.1 * (s["Wel_y"] or 0) * 235, "Vz": 0.05 * s["A"] * 235 / 3,
                             "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}}})
            json.dumps(r, allow_nan=False)
            n += 1
            formes.add((s["forme"], any(v["statut"] == HI.NON_COUVERT for v in r["verifications"])))
        except Exception as e:  # noqa: BLE001
            plantes.append((p["nom"], type(e).__name__, str(e)[:80]))
    chk(f"chaque profilé du catalogue ({n}) se calcule en compression + flexion + effort tranchant ; la sortie est du JSON strict ; "
        "I et tubes couverts, U et cornières dits non couverts", n > 1300 and not plantes
        and formes == {("I", False), ("tube_rond", False), ("tube_carre", False), ("tube_rect", False), ("U", True), ("L", True)}, (plantes[:3], formes))
    faux = []
    for nom in ("HEB 200", "HEA 120", "IPE 270", "HEA 160", "IPE 300", "HEM 300"):
        s = SE.du_catalogue(nom)
        cb = HI.courbes(s)
        att = ("a", "b") if s["h"] / s["b"] > 1.2 else ("b", "c")
        if (cb["y"], cb["z"]) != att:
            faux.append((nom, cb))
    chk("profilés du catalogue : la courbe suit h/b (HEB 200, HEA 120, HEA 160 : b et c ; IPE 270, IPE 300 : a et b)", not faux, faux)
    cl = RECENTS["HEB 200"]
    r_cat = HI.verifier({"section": SE.du_catalogue("HEB 200"), "fy": 235.0, "P": cl["P"], "axes": {"y": {"L": 7200.0}, "z": {"L": 7200.0}}})
    chk("HEB 200 du CATALOGUE (A = 78,08 cm², i_z = 5,065 cm, calculés sur le contour exact) : à moins de 0,3 % du classeur (taux 1,120 contre 1,118)",
        abs(r_cat["taux_max"] / 1.11762 - 1) < 0.003, r_cat["taux_max"])
    froid = HI.verifier({"section": SE.du_catalogue("SHS-F 200x200x8" if "SHS-F 200x200x8" in CAT.par_nom() else
                                                    next(p["nom"] for p in CAT.profils() if p["famille"] == "SHS-F" and p["t"] > 6)),
                         "fy": 235.0, "P": 100e3, "axes": {"y": {"L": 3000.0}, "z": {"L": 3000.0}}})
    chk("creux fini à froid de plus de 6 mm : la doc ne donne pas de courbe — c est gardée et c'est DIT",
        froid["grandeurs"]["z"]["courbe"] == "c" and any(x["code"] == "courbe" for x in froid["alertes"]), froid["alertes"])
    b0 = barre_classeur(RECENTS["HEA 120"], My=2.0, Vz=1000.0)
    b1 = copy.deepcopy(b0)
    HI.verifier(b1)
    chk("verifier ne modifie pas la barre reçue", b1 == b0)

    print("== 7. mutations")
    for nom, rouges in mutations():
        if rouges is None:
            chk(f"mutation « {nom} » : le texte à muter est dans le moteur, une seule fois", False, "texte absent ou répété")
        else:
            chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")

    print("== 8. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab', 'PIL', 'fitz'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from acier.barre import historique as H, section as S\n"
            "r = H.verifier({'section': S.du_catalogue('HEB 200'), 'fy': 235.0, 'P': 432720.0, 'My': 1e7, 'axes': {'y': {'L': 7200.0}, 'z': {'L': 7200.0}}})\n"
            "print(r['statut'], sorted(m for m in sys.modules if m.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab')))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le moteur se charge et calcule sans streamlit, pandas, matplotlib, scipy, reportlab (refusés)",
        p.returncode == 0 and p.stdout.strip() == "nok []", (p.stdout.strip(), p.stderr[-300:]))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
