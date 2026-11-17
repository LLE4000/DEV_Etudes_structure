# -*- coding: utf-8 -*-
"""Barres en acier — la MÉTHODE ELS (acier/barre/historique.py, historique_deversement.py ; clé interne « historique ») :
ancienne norme belge NBN B 51-001, contraintes admissibles. La spécification : brief v12 § 6 et § 18.6 ; ses sources :
docs/acier/BARRES_ACIER.md.

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
  4 bis. LE LOT B (brief v12) : le déversement des classeurs « Déversement » par défaut (11.1 bis, tout le tableau ;
     les lectures cohérentes de 11.1) ; la variante du fascicule éteinte ; chaque correction de 6.7 dite dans sa
     vérification quand elle joue ; les PRS (11.8 PRS-A, 11.9 PRS-C) ; la f_y nominale (test 37) ; au-delà de 80 mm,
     hors domaine (19.1 point 5) ; les tests 3, 5, 6, 7, 8, 11, 12, 16 de 11.5 ; le verdict bloqué.
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

from acier.barre import ec3_section as ES  # noqa: E402
from acier.barre import formules as FO  # noqa: E402
from acier.barre import historique as HI  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.barre import section as SE  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402

OK, KO = [], []
CHEMIN_MOTEUR = os.path.join(RACINE, "acier", "barre", "historique.py")
CHEMIN_DEVERSEMENT = os.path.join(RACINE, "acier", "barre", "historique_deversement.py")
CHEMIN_BARRE = os.path.join(RACINE, "acier", "barre", "moteur.py")


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


def controles(H=HI, MO_=MO):
    """Les contrôles des § 1 à 4 bis : [(nom, condition, info)] — la même liste sert au rapport et aux mutations.
    ``H`` : le moteur de la méthode (celui du dépôt, ou une copie mutée) ; ``MO_`` : le moteur des barres (la f_y)."""
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
    c("classeur du tube Ø610 : f_y = 450 N/mm² est hors des nuances AE235 à AE355 de la doc — HORS DOMAINE (brief § 6.2 point 8, test 7 du "
      "§ 11.5) : le calcul est rendu (83 %), aucune vérification n'est « ok », l'alerte le dit — chacune porte le message, qui nomme la "
      "méthode par son nom entier « méthode ELS (NBN B 51-001) » (celui de l'écran et de la note ; troisième passe du lot A : la norme, "
      "et la raison après un tiret, sans double parenthèse)",
      any(x["code"] == "nuance" for x in r["alertes"]) and r["statut"] == H.HORS_DOMAINE and round(r["taux_max"], 3) == 0.831
      and all(v["statut"] == H.HORS_DOMAINE for v in r["verifications"]) and r["grandeurs"]["hors_domaine"] == ["nuance"]
      and all("Hors du domaine de la méthode ELS (NBN B 51-001) — nuance hors AE235 à AE355 : calcul pour information" in " ".join(v["messages"])
              for v in r["verifications"]),
      (r["alertes"], r["statut"], r["taux_max"]))
    # ---- le domaine se lit sur la NUANCE (brief § 13.2 point 5) : une S235 épaisse (f_y 225) reste dans le domaine
    heb = section_classeur(RECENTS["HEB 200"])
    dans = H.verifier(dict(barre_classeur(RECENTS["HEB 200"], P=200e3), fy=225.0, nuance="S235"))
    hors = H.verifier(dict(barre_classeur(RECENTS["HEB 200"], P=200e3), fy=440.0, nuance="S460"))
    c("domaine lu sur la nuance : S235 à f_y = 225 (paroi de plus de 16 mm) → dans le domaine, « ok » ; S460 à f_y = 440 → hors domaine",
      dans["statut"] == H.OK and not any(x["code"] == "nuance" for x in dans["alertes"]) and hors["statut"] == H.HORS_DOMAINE
      and H.domaine(heb, 225.0, "S235") == [] and H.domaine(heb, 440.0, "S460") == ["nuance"] and H.domaine(heb, 225.0) == ["nuance"],
      (dans["statut"], hors["statut"]))
    froid = SE.saisie("tube_carre", "creux_froid", nom="SHS 150x150x8 formé à froid", h=150.0, b=150.0, t=8.0, A=4320.0, Iy=1.39e7, Iz=1.39e7,
                      iy=56.7, iz=56.7, Wel_y=1.85e5, Wel_z=1.85e5)
    f1 = H.verifier({"section": froid, "fy": 235.0, "nuance": "S235", "P": 100e3, "axes": {"y": {"L": 3000.0}, "z": {"L": 3000.0}}})
    f2 = H.verifier({"section": froid, "fy": 235.0, "nuance": "S235", "P": 900e3, "axes": {"y": {"L": 3000.0}, "z": {"L": 3000.0}}})
    mince = H.verifier({"section": dict(froid, t=6.0), "fy": 235.0, "nuance": "S235", "P": 100e3, "axes": {"y": {"L": 3000.0}, "z": {"L": 3000.0}}})
    c("creux formé à froid de 8 mm (> 6 mm, test 16 du § 11.5) : HORS DOMAINE, jamais « ok » — mais un taux de plus de 100 % reste « nok » ; "
      "à 6 mm, dans le domaine",
      f1["statut"] == H.HORS_DOMAINE and all(v["statut"] == H.HORS_DOMAINE for v in f1["verifications"]) and f1["taux_max"] < 1.0
      and f2["statut"] == H.NOK and mince["statut"] == H.OK and any(x["code"] == "courbe" and x["niveau"] == "attention" for x in f1["alertes"]),
      (f1["statut"], f2["statut"], mince["statut"]))

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
    r = H.verifier(dict(bc, deversement={"variante": "fascicule"}))
    g = r["grandeurs"]
    c("… le même SANS la dispense (annotée « FAUX » p. 20 ; le module ne la reprend pas), variante du fascicule p. 19 : sur 8 m sans maintien, r_d = 55,2 mm, "
      "λ̄_ed = 1,542, φ_ed = 0,307 → σ_My = 422 N/mm², NON CONFORME ; P/(A·φ) = 27 N/mm² est pourtant > 0,15·σ_adm = 23,5",
      g["deversement"]["cas"] == "I" and round(g["deversement"]["rd"], 1) == 55.2 and round(g["deversement"]["lambda_bar"], 3) == 1.542
      and round(g["deversement"]["phi"], 3) == 0.307 and round(g["sigma_My"]) == 422 and r["statut"] == H.NOK,
      (g["deversement"], g["sigma_My"]))
    ab = {"section": hea(**HEA300_DOC), "fy": 235.0, "P": 168e3, "My": 110e6, "axes": {"y": {"L": 6000.0, "u": 1.5}, "z": {"L": 6000.0, "u": 1.5}},
          "C": {"y": {"mode": "charges"}}, "deversement": {"actif": True, "Lc": 6000.0, "variante": "fascicule"}}
    r = H.verifier(ab)
    g = r["grandeurs"]
    dv = g["deversement"]
    c("doc p. 24-25, colonne HEA 300, 6 m, u = 1,5, P = 168 kN, M = 110 kN·m, C = 0,85, variante du fascicule p. 19 : φ_c = 0,4009, I_d = 3150 cm⁴, A_d = 45,7 cm², "
      "r_d = 8,30 cm, λ̄_ed = 0,770, φ_ed = 0,676, P_cr = 4673 kN, PE = 0,9461, σ = 37,3 + 116,0 = 153,3 N/mm² (la doc : 0,4 ; 3150 ; 45,7 ; "
      "8,3 ; 0,77 ; 0,7 ; 4673 ; 0,95 ; 37,3 + 111,5 = 148,8) — conforme",
      round(g["z"]["phi"], 4) == 0.4009 and round(dv["Id"] / 1e4) == 3150 and round(dv["Ad"] / 100, 1) == 45.7 and round(dv["rd"] / 10, 2) == 8.30
      and round(dv["lambda_bar"], 3) == 0.770 and round(dv["phi"], 3) == 0.676 and round(g["y"]["P_cr"] / 1e3) == 4673
      and round(g["y"]["PE"], 4) == 0.9461 and round(g["sigma_P"], 1) == 37.3 and round(g["sigma_My"], 1) == 116.0
      and round(g["sigma"], 1) == 153.3 and r["statut"] == H.OK and g["C_y"] == 0.85
      and any(h.startswith("Déversement : variante du fascicule p. 19") for h in r["hypotheses"]),
      (g["z"]["phi"], dv, g["y"]["P_cr"], g["y"]["PE"], g["sigma_P"], g["sigma_My"]))
    c("doc p. 19 : la courbe du déversement est celle de l'axe FAIBLE du profil (HEA 300 : c)", dv["courbe"] == "c" == g["z"]["courbe"])
    # les valeurs du brief (11.1) : recalculées avec le CATALOGUE (A et W exacts ; la doc imprime A et W arrondis) — l'écart
    # de 0,1 N/mm² vient des données, pas de la formule
    g14 = H.verifier({"section": SE.du_catalogue("HEA 220"), "fy": 235.0, "P": 500e3, "My": 25e6, "axes": {"y": {"L": 5000.0}, "z": {"L": 0.0}},
                      "deversement": {"actif": False}})["grandeurs"]
    g23 = H.verifier(dict(bc, section=SE.du_catalogue("HEA 200"), deversement={"actif": False}))["grandeurs"]
    g25 = H.verifier(dict(ab, section=SE.du_catalogue("HEA 300")))["grandeurs"]
    c("11.1 — les exemples de la doc aux valeurs du brief, profilés du CATALOGUE (A, W exacts) : p. 14 σ = 148,3 N/mm² (P_cr 4485 kN) ; p. 23 "
      "27,0 + 129,7 = 156,7 ; p. 25 φ_ed 0,676, 37,3 + 116,1 = 153,3 (à 0,05 près) — avec les A et W arrondis de la doc : 148,4 et 156,6",
      a_pres(g14["sigma"], 148.3, 0.05) and round(g14["y"]["P_cr"] / 1e3) == 4485 and a_pres(g23["sigma_P"], 27.0, 0.05)
      and a_pres(g23["sigma_My"], 129.7, 0.05) and a_pres(g23["sigma"], 156.7, 0.05) and round(g25["deversement"]["phi"], 3) == 0.676
      and a_pres(g25["sigma_P"], 37.3, 0.05) and a_pres(g25["sigma_My"], 116.1, 0.05) and a_pres(g25["sigma"], 153.3, 0.05),
      (g14["sigma"], g23["sigma_My"], g23["sigma"], g25["sigma_My"], g25["sigma"]))

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
    r_dev = H.verifier(dict(base, deversement={"actif": True, "Lc": 4000.0, "variante": "fascicule"}))
    r_sans = H.verifier(dict(base, deversement={"actif": False}))
    r_cl = H.verifier(dict(base, deversement={"actif": True, "Lc": 4000.0}))
    r_defaut = H.verifier({k: v for k, v in base.items() if k != "deversement"})
    gd = r_dev["grandeurs"]
    Id, Ad = 15 * 200 ** 3 / 12, 200 * 15 + (200 - 30) * 9 / 6
    c("déversement, variante du fascicule (doc p. 19) : I_d = t_f·b³/12, A_d = b·t_f + h_w·t_w/6 (h_w = h − 2·t_f), λ_ed = L_d/√(I_d/A_d), φ_ed de la courbe "
      "de l'axe faible ; il divise le SEUL terme de moment de l'axe fort", proche(gd["deversement"]["Id"], Id) and proche(gd["deversement"]["Ad"], Ad)
      and proche(gd["deversement"]["lambda"], 4000 / math.sqrt(Id / Ad)) and proche(gd["deversement"]["phi"], H.phi(4000 / math.sqrt(Id / Ad) / H.lambda_E(235), "c"))
      and proche(gd["sigma_My"], r_sans["grandeurs"]["sigma_My"] / gd["deversement"]["phi"]) and proche(gd["sigma_P"], r_sans["grandeurs"]["sigma_P"]),
      gd["deversement"])
    c("déversement : ACTIVÉ par défaut, la méthode des CLASSEURS (brief v12 § 6.4), sur la longueur de l'axe faible ; la variante du fascicule "
      "n'est ni calculée ni dite tant qu'on ne la choisit pas ; éteint (semelle maintenue en continu), φ_ed = 1 — c'est le classeur Flambement",
      proche(r_defaut["grandeurs"]["sigma"], r_cl["grandeurs"]["sigma"]) and r_cl["grandeurs"]["deversement"]["cas"] == "classeurs"
      and r_defaut["grandeurs"]["deversement"]["cas"] == "classeurs" and "Id" not in r_defaut["grandeurs"]["deversement"]
      and not any("fascicule" in h for h in r_defaut["hypotheses"]) and not any(f["cle"] in ("I_d", "A_d", "r_d") for f in r_defaut["calcul"])
      and r_sans["grandeurs"]["deversement"] == {"cas": "maintenu", "phi": 1.0}
      and any("maintenue en continu" in h for h in r_sans["hypotheses"]) and any("L_d = 4,00 m" in h for h in r_cl["hypotheses"])
      and any("L_d = 4,00 m" in h and "fascicule" in h for h in r_dev["hypotheses"]))
    petit = H.verifier(dict(base, P=10e3, deversement={"actif": True, "Lc": 4000.0, "variante": "fascicule"}))["grandeurs"]
    c("déversement : PAS de dispense quand P/(A·φ) ≤ 0,15·σ_adm (doc p. 20, « FAUX ») — φ_ed reste appliqué",
      petit["sigma_P"] <= 0.15 * petit["sigma_adm"] and petit["deversement"]["phi"] < 1 and proche(petit["deversement"]["phi"], gd["deversement"]["phi"])
      and proche(petit["sigma_My"], 30e6 / (petit["deversement"]["phi"] * 569.6e3 * petit["y"]["PE"])), (petit["sigma_P"], petit["sigma_My"]))
    rhs = SE.du_catalogue("RHS 200x100x6.3")
    rr_ = H.verifier({"section": rhs, "fy": 235.0, "My": 20e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}})
    rr = rr_["grandeurs"]
    lam_f = math.sqrt(math.pi * math.sqrt(2.6) * 6000 * rhs["Wel_y"] / math.sqrt(rhs["Iz"] * rhs["It"]))
    rc = H.verifier({"section": SE.du_catalogue("SHS 150x150x6.3"), "fy": 235.0, "My": 20e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}})["grandeurs"]
    rt = H.verifier({"section": SE.du_catalogue("CHS 168.3x6"), "fy": 235.0, "My": 10e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}})["grandeurs"]
    c("déversement des tubes (doc p. 17-18) : rectangulaire — λ_ed = √(π·√(E/G)·L_c·W_y/√(I_z·I_t)), E/G = 2,6, « proche de l'unité » "
      "(φ_ed = 0,98 ici) ; rond et carré — sans objet, φ_ed = 1 ; l'hypothèse en une phrase, « facteur » en minuscule après « ; »",
      rr["deversement"]["cas"] == "ferme" and proche(rr["deversement"]["lambda"], lam_f)
      and 0.95 < rr["deversement"]["phi"] < 1 and rc["deversement"] == {"cas": "sans_objet", "phi": 1.0} == rt["deversement"]
      and "Déversement (doc p. 18, profil fermé) : L_d = 6,00 m ; facteur π·√2,6 = 5,0657 (5,1 dans la doc)." in rr_["hypotheses"],
      (rr["deversement"], rc["deversement"], rr_["hypotheses"]))
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
      proche(g["sigma"], att) and next(v for v in r["verifications"] if v["id"] == "H-SIGMA")["titre"] == "Flexion composée déviée"
      and "doc p. 26" in next(v for v in r["verifications"] if v["id"] == "H-SIGMA")["reference"], (g["sigma"], att))
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
                   ("M infini", dict(My=float("inf"))), ("L_c négatif", dict(deversement={"Lc": -5.0})), ("mode de C inconnu", dict(C={"y": {"mode": "x"}})),
                   ("C_d = 0,9", dict(deversement={"Cd": 0.9})), ("C_d = 3,5", dict(deversement={"Cd": 3.5})), ("C_d texte", dict(deversement={"Cd": "1"})),
                   ("variante inconnue", dict(deversement={"variante": "eurocode"})), ("classe e", dict(deversement={"classe": "e"})),
                   ("semelle « milieu »", dict(deversement={"semelle": "milieu"}))):
        try:
            H.verifier(barre_classeur(heb, **k))
            refus.append(nom)
        except H.Refus as e:
            if not str(e):
                refus.append(nom + " (sans message)")
    c("domaine : f_y nul, s < 1, L négatif, u nul, texte, infini, L_d négatif, mode de C inconnu, C_d hors de [1 ; 3] (1 : le moment uniforme, le "
      "cas le plus défavorable), variante, classe ou semelle inconnues — refusés avec un message", not refus, refus)
    controles_lot_b(H, MO_, c)
    return C


# ------------------------------------------------------------------ 4 bis. le lot B (brief v12)
def section_prs(b1, t1, hw, tw, b2, t2, bords="soude_oxycoupe", nom="PRS"):
    """Un PRS en I au CONTRAT de acier/barre/prs.py (règles du lot B, § 4 ; le module est écrit par le bloc B2) : ses
    propriétés calculées ici sur les trois rectangles, cordons non comptés ; 1 = semelle HAUTE ; ``bords`` None : non saisis."""
    h = t1 + hw + t2
    rects = [(b2, t2, t2 / 2.0), (tw, hw, t2 + hw / 2.0), (b1, t1, t2 + hw + t1 / 2.0)]       # (largeur, hauteur, z depuis le bas)
    A = sum(b * t for b, t, _ in rects)
    zg = sum(b * t * z for b, t, z in rects) / A
    Iy = sum(b * t ** 3 / 12.0 + b * t * (z - zg) ** 2 for b, t, z in rects)
    Iz = sum(t * b ** 3 / 12.0 for b, t, _ in rects)
    Wh, Wb = Iy / (h - zg), Iy / zg
    return {"nom": nom, "famille": None, "forme": "I", "fabrication": bords, "axes": ("y-y", "z-z"), "h": h, "b": max(b1, b2), "tf": max(t1, t2),
            "tw": tw, "r": None, "t": None, "d": None, "A": A, "Iy": Iy, "Iz": Iz, "iy": math.sqrt(Iy / A), "iz": math.sqrt(Iz / A),
            "Wel_y": min(Wh, Wb), "Wel_y_haut": Wh, "Wel_y_bas": Wb, "Wel_z": 2.0 * Iz / max(b1, b2), "Wpl_y": None, "Wpl_z": None, "It": None,
            "Iw": None, "Avz": 1.2 * hw * tw, "Avy": A - hw * tw, "masse": A * 7850e-6, "cf_tf": None, "cw_tw": None,
            "prs": {"b1": b1, "t1": t1, "hw": hw, "tw": tw, "b2": b2, "t2": t2, "a": None}, "monosymetrique": (b1, t1) != (b2, t2), "z_g": zg,
            "eta": 1.2}


def lamine(h, b, tf, tw=10.0, nom="I laminé"):
    """Un I laminé donné par h, b, t_f (les cas du tableau 11.1 bis) : ses propriétés approchées, le déversement des
    classeurs n'en lit que h, b, t_f et W."""
    s = section_prs(b, tf, h - 2 * tf, tw, b, tf, bords="lamine", nom=nom)
    s.pop("prs")
    return dict(s, monosymetrique=False)


PRS_A = dict(b1=300, t1=30, hw=540, tw=15, b2=300, t2=30)
PRS_C = dict(b1=300, t1=30, hw=1420, tw=25, b2=500, t2=50)
# 11.1 bis : (nom, section, f_y, l (mm), C_d, attendu {λ_e,d, λ̄, classe, φ_d, σ_adm,d}, corrections attendues)
CAS_11_1_BIS = [
    ("fichier de travail, HE 200 B S235, l = 4 m", lamine(200, 200, 15, 9), 235.0, 4000.0, 1.0,
     dict(lam=63.687, lb=0.6781, classe="c", phi=0.7385, sad=115.70), ["axe_faible"]),
    ("DEVERSEMENT_CAD S355, h = b = 180, t_f 14, l = 4 m", lamine(180, 180, 14, 8.5), 355.0, 4000.0, 1.0,
     dict(lam=65.922, lb=0.8628, classe="c", phi=0.6232, sad=147.50), ["axe_faible", "fy_nuance"]),
    ("φ_d = 1 : HE 200 B S235, l = 0,30 m", lamine(200, 200, 15, 9), 235.0, 300.0, 1.0,
     dict(lam=17.44, lb=0.1857, classe="c", phi=1.0, sad=156.67), ["phid_plafond", "axe_faible"]),
    ("t_f = 50 mm et h/b = 2 : h 600, b 300, S235, l = 6 m", lamine(600, 300, 50, 20), 235.0, 6000.0, 1.0,
     dict(lam=60.42, lb=0.6433, classe="c", phi=0.7597, sad=119.03), ["seuils_epaisseur", "axe_faible"]),
    ("t_f = 80 mm : h = b = 300, S235, l = 6 m", lamine(300, 300, 80, 40), 235.0, 6000.0, 1.0,
     dict(lam=33.78, lb=0.3596, classe="c", phi=0.9186, sad=143.92), ["seuils_epaisseur", "axe_faible"]),
    ("IPE 600 S235, l = 12 m, C_d = 1", lamine(600, 220, 19, 12), 235.0, 12000.0, 1.0, dict(classe="b", sad=42.58), ["axe_faible"]),
    ("IPE 600 S235, l = 12 m, C_d = 1,12", lamine(600, 220, 19, 12), 235.0, 12000.0, 1.12, dict(classe="b", sad=46.86), ["axe_faible"]),
]
PHI_D = {"a": (0.9256, 0.6682, 0.3735, 0.2233), "b": (0.8845, 0.5975, 0.3424, 0.2096), "c": (0.8433, 0.5403, 0.3147, 0.1963),
         "d": (0.7802, 0.4680, 0.2771, 0.1769)}


def a_pres(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


def verif_de(r, id_):
    return next((v for v in r["verifications"] if v["id"] == id_), None)


def correction_dite(r, cle, id_):
    """La correction ``cle`` joue : elle est NOMMÉE (``corrections``) et sa ligne est dans la vérification ``id_``."""
    v = verif_de(r, id_)
    c = next((x for x in r.get("corrections") or [] if x["id"] == cle), None)
    return c is not None and v is not None and id_ in c["verifications"] and f"Correction du classeur : {c['texte']}" in v["messages"]


def controles_lot_b(H, MO_, c):
    DE = H.DE
    # ---- 11.1 bis : la fonction des classeurs, cas par cas (σ_adm,d à 0,01 N/mm², φ_d à 0,0001)
    for nom, s, fy, l, Cd, att, corr in CAS_11_1_BIS:
        r = H.verifier({"section": s, "fy": fy, "My": 0.5 * s["Wel_y"] * fy / 1.5 * 0.5, "axes": {"y": {"L": l}, "z": {"L": l}},
                        "deversement": {"Lc": l, "Cd": Cd}})
        dv = r["grandeurs"]["deversement"]
        v = verif_de(r, "H-DEV")
        ok = (dv["cas"] == "classeurs" and dv["classe"] == att["classe"] and a_pres(dv["sigma_adm_d"], att["sad"], 0.01)
              and a_pres(v["limite"], att["sad"], 0.01) and ("lam" not in att or a_pres(dv["lambda"], att["lam"], 0.01))
              and ("lb" not in att or a_pres(dv["lambda_bar"], att["lb"], 0.0001)) and ("phi" not in att or a_pres(dv["phi"], att["phi"], 0.0001))
              and sorted(x["id"] for x in r["corrections"]) == sorted(corr) and all(correction_dite(r, k, "H-DEV") for k in corr))
        c(f"11.1 bis — {nom} : λ_e,d, λ̄, classe {att['classe']}, φ_d, σ_adm,d = {att['sad']:.2f} N/mm² ; corrections dites : {', '.join(corr)}",
          ok, (dv.get("lambda"), dv.get("lambda_bar"), dv.get("classe"), dv.get("phi"), dv.get("sigma_adm_d"), [x["id"] for x in r["corrections"]]))
    d = DE.classeurs(lamine(200, 200, 15, 9), 300.0, H.lambda_E(235.0))
    c("11.1 bis — φ_d = 1 sous λ̄ = 0,2 (NBN B 51-002) : la formule prolongée donnerait 1,0073 ; la correction le dit avec ce chiffre",
      d["phi"] == 1.0 and round(d["phi_prolonge"], 4) == 1.0073 and "phid_plafond" in d["corrections"]
      and "1,0073" in DE.texte_correction("phid_plafond", prolonge=H._fr(d["phi_prolonge"], 4)), d)
    u = SE.saisie("U", "lamine", nom="UPN 140 (DEVERSEMENT_CAD)", h=140.0, b=60.0, tf=10.0, tw=7.0, A=2040.0, Iy=6.05e6, Iz=0.627e6, iy=54.5, iz=17.5,
                  Wel_y=86.4e3, Wel_z=14.8e3)
    d = DE.classeurs(u, 3800.0, H.lambda_E(235.0))
    c("11.1 bis — U (dimensions d'un UPN 140), S235, l = 3,80 m, fonction seule : classe c (le classeur donnait b, celle d'un I laminé) ; "
      "λ_e,d 116,13 ; λ̄ 1,2366 ; φ_d 0,4169 ; σ_adm,d 65,32 N/mm² ; la correction « classe du type » est nommée",
      d["classe"] == "c" and d["classe_lamine"] == "b" and a_pres(d["lambda"], 116.13, 0.01) and a_pres(d["lambda_bar"], 1.2366, 0.0001)
      and a_pres(d["phi"], 0.4169, 0.0001) and a_pres(235.0 / 1.5 * d["phi"], 65.32, 0.01) and "classe_type" in d["corrections"]
      and "axe_faible" not in d["corrections"], d)
    faux = [(k, lb, round(DE.phi_d(lb, k), 4), att) for k, vals in PHI_D.items() for lb, att in zip((0.5, 1.0, 1.5, 2.0), vals)
            if abs(DE.phi_d(lb, k) - att) > 0.0001]
    c("11.1 bis — la formule de φ_d (NBN B 51-002, n du tableau 11 : 0,206 / 0,339 / 0,489 / 0,756) à λ̄ = 0,5 / 1,0 / 1,5 / 2,0, classes a à d, à 0,0001",
      not faux, faux)
    r1 = H.verifier({"section": lamine(600, 220, 19, 12), "fy": 235.0, "My": 50e6, "axes": {"y": {"L": 12000.0}, "z": {"L": 12000.0}},
                     "deversement": {"Lc": 12000.0, "Cd": 1.12, "niveau": "haut"}})
    r2 = H.verifier({"section": lamine(600, 220, 19, 12), "fy": 235.0, "My": 50e6, "axes": {"y": {"L": 12000.0}, "z": {"L": 12000.0}},
                     "deversement": {"Lc": 12000.0, "Cd": 1.12, "niveau": "centre"}})
    c("11.1 bis — C_d = 1,12 avec la charge sur la semelle haute : le message de 6.4 (« gardez C_d = 1 »), sans changer la valeur ; au centre "
      "de cisaillement, pas de message", any(x["code"] == "cd_haut" and x["message"] == DE.MESSAGE_CD for x in r1["alertes"])
      and not any(x["code"] == "cd_haut" for x in r2["alertes"]) and proche(r1["grandeurs"]["deversement"]["phi"], r2["grandeurs"]["deversement"]["phi"]))
    tableau = {(600, 300, 40): "b", (600, 300, 40.5): "c", (600, 300, 90): "d", (300, 300, 79): "c", (300, 300, 81): "d", (240, 200, 50): "c",
               (241, 200, 50): "c", (241, 200, 30): "b"}
    faux = [(k, DE.classe(lamine(*k), *k)["classe"]) for k, att in tableau.items() if DE.classe(lamine(*k), *k)["classe"] != att]
    eur = [k for k in ((600, 300, 90), (300, 300, 95)) if "seuils_eurocode" not in DE.classe(lamine(*k), *k)["corrections"]]
    pas = [k for k in ((600, 300, 40), (300, 300, 79), (600, 300, 105)) if set(DE.classe(lamine(*k), *k)["corrections"]) & {"seuils_epaisseur", "seuils_eurocode"}]
    c("tableau 12, I laminé : h/b > 1,2 → b (t_f ≤ 40), c (≤ 80), d au-delà (pas de ligne : note 3, prudent) ; h/b ≤ 1,2 → c (≤ 80), d au-delà ; "
      "les corrections 2 (40 < t_f ≤ 80) et 3 (80 < t_f ≤ 100, la feuille « Calcul automatique ») jouent seulement là",
      not faux and not eur and not pas, (faux, eur, pas))
    pa = section_prs(**PRS_A)
    soudes = {("soude_oxycoupe", 30): "c", ("soude_oxycoupe", 40): "c", ("soude_oxycoupe", 41): "d", ("soude_larges_plats", 30): "d"}
    faux = [(k, DE.classe(dict(pa, fabrication=k[0]), 600, 300, k[1])["classe"]) for k, att in soudes.items()
            if DE.classe(dict(pa, fabrication=k[0]), 600, 300, k[1])["classe"] != att]
    c("tableau 12, I soudé (PRS, t_f de la semelle comprimée) : oxycoupé → c jusqu'à 40 mm, d au-delà ; larges plats → d (note 3)", not faux, faux)
    textes = {}
    for fab in ("recuit", "lamine_semelles_additionnelles"):
        try:
            DE.classe(dict(lamine(300, 300, 20, 10), fabrication=fab), 300, 300, 20)
        except H.Refus as e:
            textes[fab] = str(e)
    c("tableau 12 : un profil que la règle des classeurs ne couvre pas (I recuit, à semelles additionnelles) — la classe est « à fournir », "
      "rien n'est deviné ; le message (une ligne) nomme le profil, jamais une clé interne",
      textes.get("recuit", "").startswith("Classe de déversement à fournir") and "pour un I recuit" in textes["recuit"]
      and "pour un I à semelles additionnelles" in textes.get("lamine_semelles_additionnelles", "")
      and all("_" not in t and "(I," not in t and len(t) < 120 for t in textes.values()), textes)

    # ---- 11.1 : les lectures cohérentes (méthode de la doc en entier, C = 1)
    vraies = SE.saisie("I", "lamine", nom="HEA 160", h=152, b=160, tf=9, tw=6, A=3877.0, Iy=1673e4, Iz=615.6e4, iy=65.7, iz=39.8, Wel_y=220.1e3,
                       Wel_z=76.95e3)
    melange = SE.saisie("I", "lamine", nom="HEA 160 (FLAMBECBD)", h=152, b=160, tf=9, tw=6, A=3877.0, Iy=615.6e4, Iz=3877 * 3.98 ** 2 * 100,
                        iy=39.8, iz=39.8, Wel_y=76.95e3, Wel_z=76.95e3)
    ax = {"y": {"L": 1750.0}, "z": {"L": 1750.0}}
    fa = H.verifier({"section": vraies, "fy": 235.0, "P": 300e3, "Mz": 3e6, "axes": ax})["grandeurs"]
    fo = H.verifier({"section": vraies, "fy": 235.0, "P": 300e3, "My": 3e6, "axes": ax, "deversement": {"Lc": 1750.0}})
    ff = H.verifier({"section": vraies, "fy": 235.0, "P": 300e3, "My": 3e6, "axes": ax, "deversement": {"Lc": 1750.0, "variante": "fascicule"}})
    mx = H.verifier({"section": melange, "fy": 235.0, "P": 300e3, "My": 3e6, "axes": ax, "deversement": {"Lc": 1750.0, "variante": "fascicule"}})
    go, gf = fo["grandeurs"], ff["grandeurs"]
    c("11.1 / test 5 du § 11.5 — FLAMBECBD : axe faible 89,92 + 43,71 = 133,63 N/mm² ; axe fort, vraies valeurs, L_d = 1,75 m, classeurs (classe c) : "
      "λ_e,d 53,01, λ̄ 0,5644, φ_ed 0,8066, 89,92 + 17,60 = 107,52 ; variante fascicule : φ_ed 0,8850, 105,96",
      a_pres(fa["sigma_P"], 89.92, 0.01) and a_pres(fa["sigma_Mz"], 43.71, 0.01) and a_pres(fa["sigma"], 133.63, 0.01)
      and go["deversement"]["classe"] == "c" and a_pres(go["deversement"]["lambda"], 53.01, 0.01) and a_pres(go["deversement"]["lambda_bar"], 0.5644, 0.0001)
      and a_pres(go["deversement"]["phi"], 0.8066, 0.0001) and a_pres(go["sigma_My"], 17.60, 0.01) and a_pres(go["sigma"], 107.52, 0.01)
      and a_pres(gf["deversement"]["phi"], 0.8850, 0.0001) and a_pres(gf["sigma"], 105.96, 0.01), (fa["sigma"], go["sigma"], gf["sigma"]))
    c("test 5 du § 11.5 — le MÉLANGE « valeurs d'axe faible dans les cases fortes + déversement » (139,31 N/mm²) ne correspond à aucun cas : "
      "le verdict est BLOQUÉ (non couvert, sans taux), le calcul reste montré, le message le dit",
      a_pres(mx["grandeurs"]["sigma"], 139.31, 0.01) and mx["statut"] == H.NON_COUVERT and mx["taux_max"] is None
      and all(v["statut"] == H.NON_COUVERT and v["taux"] is None and any("Verdict bloqué" in m for m in v["messages"]) for v in mx["verifications"])
      and any(x["code"] == "bloque" for x in mx["alertes"]), (mx["statut"], mx["grandeurs"]["sigma"]))
    ipe = SE.saisie("I", "lamine", nom="IPE 270 (FLAMBEBAD)", h=270, b=135, tf=10.2, tw=6.6, A=4594.0, Iy=5790e4, Iz=4594 * 3.02 ** 2 * 100, iy=112.3,
                    iz=30.2, Wel_y=428e3, Wel_z=62.2e3)
    ax = {"y": {"L": 3400.0}, "z": {"L": 3400.0}}
    bc = H.verifier({"section": ipe, "fy": 355.0, "P": 185e3, "My": 361 * 3.4 ** 2 / 8 * 1e3, "axes": ax, "deversement": {"Lc": 3400.0}})["grandeurs"]
    bf = H.verifier({"section": ipe, "fy": 355.0, "P": 185e3, "My": 361 * 3.4 ** 2 / 8 * 1e3, "axes": ax,
                     "deversement": {"Lc": 3400.0, "variante": "fascicule"}})["grandeurs"]
    c("11.1 — FLAMBEBAD : φ_min et L_d = 3,40 m ; classeurs (IPE 270, S355, h/b = 2,0, classe b) : λ_e,d 100,70, λ̄ 1,3179, φ_ed 0,4186, "
      "114,43 + 2,99 = 117,42 N/mm² ; variante fascicule : φ_ed 0,4524 (courbe b), 117,20",
      bc["deversement"]["classe"] == "b" and a_pres(bc["deversement"]["lambda"], 100.70, 0.01) and a_pres(bc["deversement"]["lambda_bar"], 1.3179, 0.0001)
      and a_pres(bc["deversement"]["phi"], 0.4186, 0.0001) and a_pres(bc["sigma_P"], 114.43, 0.01) and a_pres(bc["sigma_My"], 2.99, 0.01)
      and a_pres(bc["sigma"], 117.42, 0.01) and a_pres(bf["deversement"]["phi"], 0.4524, 0.0001) and a_pres(bf["sigma"], 117.20, 0.01),
      (bc["sigma"], bf["sigma"]))
    A = 3059.3
    chs = SE.saisie("tube_rond", "creux_froid", nom="CHS 168,3 × 6", t=6, A=A, Iy=1009e4, Iz=1009e4, iy=57.4, iz=57.4, Wel_y=1009e4 / 84.15,
                    Wel_z=1009e4 / 84.15)
    M = (35000 * 0.05 + 78600 * A / 1e6 * 7.0 ** 2 / 8) * 1e3
    b_ = {"section": chs, "fy": 235.0, "P": 35000.0, "My": M, "axes": {"y": {"L": 7000.0}, "z": {"L": 7000.0}}}
    c1, c2 = H.verifier(b_), H.verifier(dict(b_, C={"y": {"mode": "charges"}}))
    c("11.1 / test 3 du § 11.5 — CHS 168,3 × 6 avec A = 3059,3 mm² : 59,75 N/mm² (φ_ed = 1, taux 0,3814) ; avec C = 0,85 : 55,15 — un tube rond "
      "n'a pas de déversement : φ_ed = 1, présenté comme une règle complémentaire (EC3 § 6.3.2.1(2))",
      a_pres(c1["grandeurs"]["sigma"], 59.75, 0.01) and round(c1["taux_max"], 4) == 0.3814 and a_pres(c2["grandeurs"]["sigma"], 55.15, 0.01)
      and c1["grandeurs"]["deversement"] == {"cas": "sans_objet", "phi": 1.0}
      and any("règle complémentaire" in h and "6.3.2.1(2)" in h for h in c1["hypotheses"]), (c1["grandeurs"]["sigma"], c2["grandeurs"]["sigma"]))
    sec = lambda fab, W: SE.saisie("tube_rond", fab, nom="Tube Ø610 × 12,5", t=12.5, A=23464.0, Iy=104754e4, Iz=104754e4, iy=211.3, iz=211.3,  # noqa: E731
                                   Wel_y=W * 1e3, Wel_z=W * 1e3)
    q = 78600 * 23464 / 1e6
    tx = {}
    for cle, fab, W, M, C in (("a", "creux_chaud", 3506, 0.0, "unite"), ("c", "creux_froid", 3506, 0.0, "unite"), ("c+pp", "creux_froid", 3434.6, q * 100 / 8, "unite"),
                              ("c+pp C", "creux_froid", 3434.6, q * 100 / 8, "charges"), ("a+pp", "creux_chaud", 3434.6, q * 100 / 8, "unite")):
        r = H.verifier({"section": sec(fab, W), "fy": 450.0, "P": 5e6, "My": M * 1e3, "axes": {"y": {"L": 10000.0}, "z": {"L": 10000.0}},
                        "C": {"y": {"mode": C}}})
        tx[cle] = (r["grandeurs"]["sigma"], r["taux_max"], r["statut"])
    c("11.1 / test 6 du § 11.5 — REALEX : le verdict change avec la fabrication et la position — courbe a 0,831 ; courbe c 295,14 N/mm² (0,984) ; "
      "c + poids propre (q = 1844 N/m, W = 3434,6 cm³) 305,40 (1,018, ne passe pas ; C = 0,85 : 303,86) ; a + poids propre 259,42 (0,865) ; "
      "f_y 450 : hors domaine, et le 1,018 reste « nok »",
      round(tx["a"][1], 3) == 0.831 and a_pres(tx["c"][0], 295.14, 0.01) and round(tx["c"][1], 3) == 0.984 and a_pres(tx["c+pp"][0], 305.40, 0.01)
      and round(tx["c+pp"][1], 3) == 1.018 and tx["c+pp"][2] == H.NOK and a_pres(tx["c+pp C"][0], 303.86, 0.01) and a_pres(tx["a+pp"][0], 259.42, 0.01)
      and round(tx["a+pp"][1], 3) == 0.865 and tx["a"][2] == tx["c"][2] == tx["a+pp"][2] == H.HORS_DOMAINE, tx)

    # ---- PRS-A (11.8, méthode ELS) : f_y nominale 355, L_d = 10 m
    pa_ = section_prs(**PRS_A)
    c("PRS-A au contrat de prs.py : A = 26 100 mm², I_y = 1660,23·10⁶ mm⁴, I_z = 135,152·10⁶ mm⁴, W_el,y = 5534,1·10³ mm³ (le dictionnaire des tests)",
      proche(pa_["A"], 26100.0) and abs(pa_["Iy"] / 1660.23e6 - 1) < 1e-5 and abs(pa_["Iz"] / 135.152e6 - 1) < 1e-5 and abs(pa_["Wel_y"] / 5534.1e3 - 1) < 1e-4)
    res = {}
    for bords in ("soude_oxycoupe", "soude_larges_plats"):
        for var in ("classeurs", "fascicule"):
            r = H.verifier({"section": section_prs(**PRS_A, bords=bords), "fy": 355.0, "nuance": "S355", "My": 1000e6,
                            "axes": {"y": {"L": 10000.0}, "z": {"L": 10000.0}}, "deversement": {"Lc": 10000.0, "variante": var}})
            res[(bords, var)] = r
    o, lp = res[("soude_oxycoupe", "classeurs")], res[("soude_larges_plats", "classeurs")]
    do, dl = o["grandeurs"]["deversement"], lp["grandeurs"]["deversement"]
    c("11.8 PRS-A, méthode ELS, classeurs : λ_e,d = 100,698 ; λ̄ = 1,3179 ; oxycoupé → classe c : φ_d 0,3816, σ_adm,d 90,31 N/mm² ; larges plats → d : "
      "0,3329, 78,79 ; PRS bisymétrique : calcul normal, avec le message « formule prudente », dans le domaine",
      a_pres(do["lambda"], 100.698, 0.001) and a_pres(do["lambda_bar"], 1.3179, 0.0001) and do["classe"] == "c" and a_pres(do["phi"], 0.3816, 0.0001)
      and a_pres(do["sigma_adm_d"], 90.31, 0.01) and dl["classe"] == "d" and a_pres(dl["phi"], 0.3329, 0.0001) and a_pres(dl["sigma_adm_d"], 78.79, 0.01)
      and DE.MESSAGE_PRS in verif_de(o, "H-DEV")["messages"] and "prs_mono" not in o["grandeurs"]["hors_domaine"]
      and not any(x["code"] == "prs_mono" for x in o["alertes"]), (do, dl))
    fo_, fl_ = res[("soude_oxycoupe", "fascicule")]["grandeurs"]["deversement"], res[("soude_larges_plats", "fascicule")]["grandeurs"]["deversement"]
    c("11.8 PRS-A, variante fascicule p. 19 : I_d = 67,5·10⁶ mm⁴ ; A_d = 10 350 mm² ; r_d = 80,757 mm ; λ_ed = 123,83 ; λ̄ = 1,6206 ; φ_ed = 0,3005 "
      "(oxycoupé, courbe b) ou 0,2847 (larges plats, courbe c)",
      proche(fo_["Id"], 67.5e6) and proche(fo_["Ad"], 10350.0) and a_pres(fo_["rd"], 80.757, 0.001) and a_pres(fo_["lambda"], 123.83, 0.01)
      and a_pres(fo_["lambda_bar"], 1.6206, 0.0001) and fo_["courbe"] == "b" and a_pres(fo_["phi"], 0.3005, 0.0001) and fl_["courbe"] == "c"
      and a_pres(fl_["phi"], 0.2847, 0.0001), (fo_, fl_))
    g_o = res[("soude_oxycoupe", "classeurs")]["grandeurs"]
    c("18.6 — PRS en méthode ELS : courbes de flambement du fascicule pour un I soudé (oxycoupé : a fort, b faible ; larges plats : b et c)",
      (g_o["y"]["courbe"], g_o["z"]["courbe"]) == ("a", "b") and (lp["grandeurs"]["y"]["courbe"], lp["grandeurs"]["z"]["courbe"]) == ("b", "c"))
    lam_ = H.verifier({"section": lamine(600, 220, 19, 12), "fy": 235.0, "My": 50e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}})
    c("18.6 — PRS-A : « classes du flambement (fascicule) et du déversement (tableau 12) différentes » est DIT, une fois, dans les hypothèses "
      "de la méthode (oxycoupé : courbe b, classe c) ; pas pour un laminé ; la ligne du déversement ne porte que « formule prudente » et ses "
      "corrections",
      o["hypotheses"].count(DE.MESSAGE_CLASSES) == 1 and DE.MESSAGE_CLASSES not in lam_["hypotheses"]
      and all(m == DE.MESSAGE_PRS or m.startswith("Correction du classeur : ") for m in verif_de(o, "H-DEV")["messages"]),
      (o["hypotheses"], verif_de(o, "H-DEV")["messages"]))

    # ---- PRS-C (11.9) : la poutre de 28 m du bureau, monosymétrique, S235 nominale
    pc = section_prs(**PRS_C)
    q = 5.456 + 0.25 * 26 * 3.20 + 0.75 * 3.20
    Mmid = (q * 28 ** 2 / 8 + 0.75 * 3.0 * 3.0 * 28 / 4 - 0.75 * 3.0 * 1.5 ** 2 / 2) * 1e6
    sad = {}
    for nom, n in (("8 maintiens", 8), ("aux tiers", 2), ("17 maintiens", 17), ("18 maintiens", 18)):
        l = 28000.0 / (n + 1)
        r = H.verifier({"section": pc, "fy": 235.0, "nuance": "S235", "My": Mmid, "axes": {"y": {"L": 28000.0}, "z": {"L": l}},
                        "deversement": {"Lc": l}})
        sad[nom] = r
    r8, g8 = sad["8 maintiens"], sad["8 maintiens"]["grandeurs"]
    d8 = g8["deversement"]
    v8 = verif_de(r8, "H-DEV")
    c("11.9 PRS-C, méthode ELS : M = 2853,0 kN·m au milieu (charges de service du bureau) ; W_el,y,min = 24,368·10⁶ mm³ (fibre haute) ; "
      "σ = M/W = 117,08 N/mm²", a_pres(Mmid / 1e6, 2853.0, 0.05) and abs(pc["Wel_y"] / 24.368e6 - 1) < 1e-4 and pc["Wel_y"] == pc["Wel_y_haut"]
      and a_pres(v8["valeur"], 117.08, 0.01), (Mmid, pc["Wel_y"], v8["valeur"]))
    c("11.9 PRS-C, déversement des classeurs, semelle haute comprimée (b 300, t_f 30 → classe c), C_d = 1, 8 maintiens (l = 3,111 m) : λ_e,d 88,81 ; "
      "λ̄ 0,9456 ; φ_d 0,5724 ; σ_adm,d 89,68 N/mm² → NE PASSE PAS (taux 1,306)",
      d8["semelle"] == "haut" and (d8["b"], d8["tf"]) == (300, 30) and d8["classe"] == "c" and a_pres(d8["lambda"], 88.81, 0.01)
      and a_pres(d8["lambda_bar"], 0.9456, 0.0001) and a_pres(d8["phi"], 0.5724, 0.0001) and a_pres(d8["sigma_adm_d"], 89.68, 0.01)
      and v8["statut"] == H.NOK and r8["statut"] == H.NOK and a_pres(v8["taux"], 117.08 / 89.68, 0.001), (d8, v8["taux"]))
    st = {k: (r["grandeurs"]["deversement"]["sigma_adm_d"], verif_de(r, "H-DEV")["statut"]) for k, r in sad.items()}
    c("11.9 PRS-C : maintiens aux tiers → σ_adm,d 42,91 ; il faut 18 maintiens (l = 1,474 m : σ_adm,d 118,3 ≥ 117,08 ; 17 ne suffisent pas)",
      a_pres(st["aux tiers"][0], 42.91, 0.01) and a_pres(st["18 maintiens"][0], 118.3, 0.05) and st["17 maintiens"][0] < 117.08 <= st["18 maintiens"][0]
      and st["17 maintiens"][1] == H.NOK, st)
    r18 = sad["18 maintiens"]
    c("11.9 / 18.6 — PRS monosymétrique : « hors domaine » (la formule suppose deux semelles égales) — avec 18 maintiens la ligne passe et n'est "
      "jamais « vérifiée » ; à 8 maintiens elle reste « nok » ; l'alerte et le message de 6.4 (« formule prudente ») le disent ; W = W_el,y,min",
      verif_de(r18, "H-DEV")["statut"] == H.HORS_DOMAINE and r18["statut"] == H.HORS_DOMAINE and "prs_mono" in r18["grandeurs"]["hors_domaine"]
      and any("PRS monosymétrique" in m for m in verif_de(r18, "H-DEV")["messages"]) and any(x["code"] == "prs_mono" for x in r8["alertes"])
      and DE.MESSAGE_PRS in v8["messages"] and proche(v8["valeur"], Mmid / pc["Wel_y_haut"]), (verif_de(r18, "H-DEV")["statut"], r18["statut"]))
    l18 = 28000.0 / 19
    base18 = {"section": pc, "fy": 235.0, "nuance": "S235", "My": Mmid, "axes": {"y": {"L": 28000.0}, "z": {"L": l18}}, "deversement": {"Lc": l18}}
    f18 = H.verifier(dict(base18, deversement={"Lc": l18, "variante": "fascicule"}))
    vf = verif_de(f18, "H-SIGMA")
    c("18.6 — PRS-C monosymétrique, 18 maintiens, VARIANTE du fascicule (I_d et A_d établis pour deux semelles égales) : « hors domaine » "
      "aussi — la ligne des contraintes passe (taux < 1) et n'est jamais « vérifiée » ; son message ne nomme pas les classeurs",
      f18["grandeurs"]["deversement"]["cas"] == "I" and vf is not None and vf["statut"] == H.HORS_DOMAINE and vf["taux"] < 1.0
      and f18["statut"] == H.HORS_DOMAINE and "prs_mono" in f18["grandeurs"]["hors_domaine"]
      and any("PRS monosymétrique" in m and "classeurs" not in m for m in vf["messages"]), (vf and vf["statut"], f18["statut"]))
    c18 = H.verifier(dict(base18, Vz=400e3, C={"y": {"mode": "charges"}}))
    st18 = {i: (verif_de(c18, i)["statut"], round(verif_de(c18, i)["taux"], 3)) for i in ("H-SIGMA", "H-DEV", "H-EQ") if verif_de(c18, i)}
    c("18.6 — … classeurs, C = 0,85 et V_z = 400 kN : H-SIGMA, H-DEV et H-EQ passent et sont TOUS « hors domaine » (φ_ed de la formule "
      "des deux semelles égales entre dans H10 et dans σ_eq)",
      set(st18) == {"H-SIGMA", "H-DEV", "H-EQ"} and all(x[0] == H.HORS_DOMAINE and x[1] < 1.0 for x in st18.values()), st18)
    # (intégration du lot B, point 4) une console à bout libre : la formule des classeurs lit l entre deux sections tenues —
    # « hors domaine » (brief § 6.2, point 8), avec la longueur que le moteur propose ; sans l'indication, dans le domaine
    con = {"section": SE.du_catalogue("IPE 240"), "fy": 235.0, "My": -16e6, "axes": {"y": {"L": 4200.0}, "z": {"L": 4200.0}}}
    cl_, cd_ = H.verifier(dict(con, deversement={"Lc": 4200.0, "bout_libre": True})), H.verifier(dict(con, deversement={"Lc": 4200.0}))
    vl_ = verif_de(cl_, "H-DEV")
    c("console à bout libre (« bout_libre ») : H-DEV passe et reste « hors domaine », son message le dit (« console à bout libre … L_d = 2,1·L, "
      "À VÉRIFIER ») ; la même barre sans l'indication : « ok »",
      vl_["statut"] == H.HORS_DOMAINE and vl_["taux"] < 1.0 and "console" in cl_["grandeurs"]["hors_domaine"]
      and any("console à bout libre" in m and "À VÉRIFIER" in m for m in vl_["messages"]) and verif_de(cd_, "H-DEV")["statut"] == H.OK
      and "console" not in cd_["grandeurs"]["hors_domaine"], (vl_["statut"], vl_["taux"], vl_["messages"][-1:]))
    rb = H.verifier({"section": pc, "fy": 235.0, "My": -Mmid, "axes": {"y": {"L": 28000.0}, "z": {"L": 3111.0}}, "deversement": {"Lc": 3111.0}})
    db = rb["grandeurs"]["deversement"]
    rbC = H.verifier({"section": pc, "fy": 235.0, "My": -Mmid, "axes": {"y": {"L": 28000.0}, "z": {"L": 3111.0}}, "deversement": {"Lc": 3111.0},
                      "C": {"y": {"mode": "charges"}}})
    wmin = min(pc["Wel_y_haut"], pc["Wel_y_bas"])
    c("18.6 — la semelle COMPRIMÉE décide : un moment négatif comprime la semelle basse (500 × 50 → oxycoupé, t_f > 40 : classe d) ; W reste "
      "W_el,y,min (fibre haute, ici tendue) dans le déversement (σ_M = |M|/W_el,y,min) et dans H10 (C = 0,85)",
      db["semelle"] == "bas" and (db["b"], db["tf"]) == (500, 50) and db["classe"] == "d" and pc["Wel_y"] == wmin < pc["Wel_y_bas"]
      and proche(verif_de(rb, "H-DEV")["valeur"], Mmid / wmin)
      and proche(rbC["grandeurs"]["sigma_My"], 0.85 * Mmid / (rbC["grandeurs"]["deversement"]["phi"] * wmin)),
      (db, verif_de(rb, "H-DEV")["valeur"], Mmid / wmin))
    ac = H.aires_cisaillement(pc)
    c("PRS, contrôle hérité H11 : « section âme » = h_w·t_w (1420 × 25 = 35 500 mm² ; l'A_v,z = 1,20·h_w·t_w de l'Eurocode 3 n'est pas prise, côté "
      "sûr) ; « section ailes » = b₁·t₁ + b₂·t₂ (9000 + 25 000)", ac == {"ame": 35500.0, "ailes": 34000.0}, ac)
    nb = H.verifier({"section": section_prs(**PRS_A, bords=None), "fy": 355.0, "My": 1000e6, "axes": {"y": {"L": 10000.0}, "z": {"L": 10000.0}}})
    c("18.2 / 18.6 — bords des semelles non saisis : verdict de la méthode ELS BLOQUÉ (non couvert), calcul montré en larges plats (classe d, "
      "courbes b et c : le cas le plus défavorable), le message le dit",
      nb["statut"] == H.NON_COUVERT and nb["taux_max"] is None and nb["grandeurs"]["deversement"]["classe"] == "d"
      and (nb["grandeurs"]["y"]["courbe"], nb["grandeurs"]["z"]["courbe"]) == ("b", "c") and any("Bords des semelles" in x["message"] for x in nb["alertes"])
      and all(v["statut"] == H.NON_COUVERT for v in nb["verifications"]), (nb["statut"], nb["grandeurs"]["deversement"].get("classe")))

    # ---- les corrections du flambement (6.7) : dites quand elles jouent, jamais sinon
    heb = section_classeur(RECENTS["HEB 200"])
    k = {"section": heb, "fy": 235.0, "deversement": {"actif": False}}
    pu = H.verifier(dict(k, P=100e3, My=20e6, axes={"y": {"L": 3000.0, "u": 2.0}, "z": {"L": 3000.0, "u": 1.0}}))
    p1 = H.verifier(dict(k, P=100e3, My=20e6, axes={"y": {"L": 6000.0, "u": 1.0}, "z": {"L": 3000.0, "u": 1.0}}))
    fort = H.verifier(dict(k, P=300e3, My=20e6, axes={"y": {"L": 12000.0}, "z": {"L": 2000.0}}))
    vs = H.verifier(dict(k, My=20e6, Vz=80e3, Vy=-20e3, axes={"y": {"L": 3000.0}, "z": {"L": 3000.0}}))
    vm = H.verifier(dict(k, My=20e6, Vz=80e3, Vy=20e3, axes={"y": {"L": 3000.0}, "z": {"L": 3000.0}}))
    c("6.7 — corrections du flambement : « P_cr avec (u·L)² » quand u ≠ 1 divise un moment ; « φ_min » quand l'axe fort est le plus élancé ; "
      "« |V_y| + |V_z| » quand V_y et V_z ont des signes contraires (τ = 80/2483 + 20/6000) — chacune dans SA vérification, et seulement là",
      correction_dite(pu, "pcr_u", "H-SIGMA") and not p1["corrections"] and correction_dite(fort, "phi_min", "H-SIGMA")
      and correction_dite(vs, "cisaillement_abs", "H-EQ") and proche(vs["grandeurs"]["tau"], 80e3 / 2483 + 20e3 / 6000) and not vm["corrections"]
      and not any("Correction du classeur" in m for r in (p1, vm) for v in r["verifications"] for m in v["messages"]),
      ([x["id"] for x in pu["corrections"]], [x["id"] for x in fort["corrections"]], [x["id"] for x in vs["corrections"]]))
    c("6.7 — les choix du bureau ne sont jamais présentés comme des corrections : aucune ligne « Correction du classeur » pour f_y nominale, s, C, C_d",
      all("nominale" not in t and "C_d" not in t and " s " not in t for t in DE.CORRECTIONS.values()) and set(DE.CORRECTIONS) == {
          "pcr_u", "phi_min", "une_courbe", "cisaillement_abs", "phid_plafond", "seuils_epaisseur", "seuils_eurocode", "fy_nuance",
          "classe_type", "axe_faible"})

    # ---- les corrections du déversement : dites seulement quand elles jouent (relecture du lot B)
    ipe = SE.du_catalogue("IPE 300")
    dev_ = {"My": 60e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 3000.0}}}
    s275 = H.verifier(dict(dev_, section=ipe, fy=275.0, nuance="S275"))
    s355 = H.verifier(dict(dev_, section=ipe, fy=355.0, nuance="S355"))
    s460 = H.verifier(dict(dev_, section=ipe, fy=460.0, nuance="S460"))
    c("6.7 correction 4 — « σ_adm,d avec la f_y de la nuance » n'est dite qu'à f_y = 355 N/mm² (la feuille S355 de DEVERSEMENT_CAD, la seule "
      "erreur connue) : IPE 300 en S355, dite ; en S275 et en S460, rien",
      correction_dite(s355, "fy_nuance", "H-DEV") and not any(x["id"] == "fy_nuance" for r in (s275, s460) for x in r["corrections"])
      and not any("DEVERSEMENT_CAD" in m for r in (s275, s460) for v in r["verifications"] for m in v["messages"]),
      ([x["id"] for x in s275["corrections"]], [x["id"] for x in s355["corrections"]]))
    s460i = H.verifier(dict(dev_, section=ipe, fy=460.0, nuance="S460", deversement={"classe": "c"}))
    c("6.4 — S460 (f_y > 430 N/mm²) : la classe est lue dans la colonne « f_y ≤ 430 » du tableau 12 (l'autre n'est pas dans le brief) et "
      "l'hypothèse le dit, « À VÉRIFIER » ; pas en S355, ni quand la classe est imposée (rien n'est lu) ; la méthode reste hors domaine",
      any(DE.MESSAGE_COLONNE in h and "À VÉRIFIER" in h for h in s460["hypotheses"]) and not any(DE.MESSAGE_COLONNE in h for h in s355["hypotheses"])
      and not any(DE.MESSAGE_COLONNE in h for h in s460i["hypotheses"]) and s460i["grandeurs"]["deversement"]["classe"] == "c"
      and s460["statut"] in (H.HORS_DOMAINE, H.NOK), (s460["hypotheses"], s460i["hypotheses"]))
    s355i = H.verifier(dict(dev_, section=ipe, fy=355.0, nuance="S355", deversement={"classe": "c"}))
    a460, a355 = ([x["message"] for x in r_["alertes"] if x["code"] == "classe_d"] for r_ in (s460i, s355i))
    c("6.4 — S460, classe c imposée : l'alerte qui la compare au tableau 12 (b) dit la colonne lue (f_y ≤ 430, À VÉRIFIER) ; en S355, non",
      len(a460) == len(a355) == 1 and DE.MESSAGE_COLONNE in a460[0] and "À VÉRIFIER" in a460[0] and DE.MESSAGE_COLONNE not in a355[0]
      and len(a460[0]) < 110, (a460, a355))
    pai = H.verifier({"section": section_prs(**PRS_A), "fy": 235.0, "My": 100e6, "axes": {"y": {"L": 8000.0}, "z": {"L": 8000.0}},
                      "deversement": {"Lc": 4000.0, "classe": "d"}})
    c("18.6 — PRS-A, classe d imposée : « classes du flambement (fascicule) et du déversement (tableau 12) différentes » n'est PAS dit (la "
      "classe employée n'est pas celle du tableau 12) ; l'alerte compare avec le tableau 12 (c)",
      DE.MESSAGE_CLASSES not in pai["hypotheses"] and pai["grandeurs"]["deversement"]["classe"] == "d"
      and any(x["code"] == "classe_d" and "donne c" in x["message"] for x in pai["alertes"]), (pai["hypotheses"], pai["alertes"]))
    k80 = [("axe_faible" in DE.classe(lamine(300, 300, t), 300, 300, t)["corrections"]) for t in (79.0, 80.0, 81.0, 95.0)]
    k80 += [("axe_faible" in DE.classe(dict(section_prs(**PRS_A), fabrication="soude_oxycoupe"), 600, 300, 90.0)["corrections"])]
    c("6.7 correction 6 — « la ligne axe faible seule » : dite pour un I jusqu'à t_f = 80 mm, PAS pour un laminé plus épais (les deux lignes y "
      "donnent sans doute d : À VÉRIFIER) ; un PRS oxycoupé de 90 mm la garde (tableau 12 : d en axe faible)", k80 == [True, True, False, False, True], k80)
    r03 = H.verifier({"section": lamine(200, 200, 15, 9), "fy": 235.0, "My": 20e6, "axes": {"y": {"L": 300.0}, "z": {"L": 300.0}},
                      "deversement": {"Lc": 300.0}})
    m03 = verif_de(r03, "H-DEV")["messages"]
    c("6.4 — sous λ̄ = 0,2, UNE ligne dit φ_d = 1 : la correction 1 (avec la formule prolongée, 1,0073) ; pas de seconde ligne « ≤ 0,2 : φ_d = 1 »",
      correction_dite(r03, "phid_plafond", "H-DEV") and not any("≤ 0,2 : φ_d = 1" in m for m in m03), m03)
    plat = section_prs(600, 30, 140, 15, 600, 30)
    rp = H.verifier({"section": plat, "fy": 235.0, "My": 50e6, "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}}})
    c("un PRS plus raide autour de z que de y (semelles 600 × 30, âme 140 × 15 : I_y/I_z = 0,25) : verdict bloqué avec SON message, qui dit "
      "pourquoi (« les charges (plan de l'âme) fléchissent ce PRS autour de son axe faible ») ; une saisie aux cases échangées (FLAMBECBD) "
      "garde le sien",
      plat["Iy"] < plat["Iz"] and rp["statut"] == H.NON_COUVERT
      and any("fléchissent ce PRS autour de son axe faible" in m and len(m) < 110 for m in rp["grandeurs"]["bloque"])
      and not any("cases de l'axe fort" in m for m in rp["grandeurs"]["bloque"]) and any("cases de l'axe fort" in m for m in mx["grandeurs"]["bloque"]),
      rp["grandeurs"]["bloque"])
    try:
        rr = H.verifier({"section": dict(lamine(300, 300, 20, 10), fabrication="recuit"), "fy": 235.0, "P": 200e3, "My": 50e6,
                         "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}}})
        vs_ = {v["id"]: v for v in rr["verifications"]}
        ok = (rr["grandeurs"]["deversement"]["cas"] == "a_fournir" and vs_["H-N-z"]["statut"] == H.OK and vs_["H-N-y"]["statut"] == H.OK
              and vs_["H-SIGMA"]["statut"] == H.NON_COUVERT and any("à fournir" in m for m in vs_["H-SIGMA"]["messages"])
              and "H-DEV" not in vs_ and "H-EQ" not in vs_ and rr["statut"] == H.NON_COUVERT and any(x["code"] == "a_fournir" for x in rr["alertes"]))
        info = (rr["statut"], list(vs_))
    except H.Refus as e:
        ok, info = False, str(e)
    c("règle 1 du brief — une classe de déversement « à fournir » (I recuit : la règle des classeurs ne le couvre pas) BLOQUE la flexion seule "
      "(non couverte, « à fournir ») ; le flambement se calcule et passe ; la méthode n'est plus refusée en entier", ok, info)
    env = MO_.Enveloppe(H.METHODE, H.NORME)
    env.verser(rb, {"combinaison": "c1", "x": 25.0})
    env.verser(r8, {"combinaison": "c1", "x": 14.0})
    out = env.rendre()
    ct = {x["id"]: x["texte"] for x in out["corrections"]}
    c8 = {x["id"]: x["texte"] for x in r8["corrections"]}
    c("barre chargée — une correction dont le texte change d'un tronçon à l'autre (PRS-C : « classe du type » c/b en haut, d/c en bas) garde "
      "le texte de l'occurrence qui GOUVERNE (ici la semelle haute, versée en second)",
      out["gouvernant"] == "H-DEV" and ct.get("classe_type") == c8.get("classe_type") != {x["id"]: x["texte"] for x in rb["corrections"]}.get("classe_type"),
      (ct.get("classe_type"), c8.get("classe_type")))
    # la ligne qui gouverne prise là où rien ne fléchit (H-EQ à l'appui) : les hypothèses du déversement restent dites
    pa_ = section_prs(**PRS_A)
    ax_ = {"y": {"L": 8000.0}, "z": {"L": 8000.0}}
    appui = H.verifier({"section": pa_, "fy": 235.0, "Vz": 600e3, "axes": ax_})
    env = MO_.Enveloppe(H.METHODE, H.NORME)
    env.verser(appui, {"combinaison": "c1", "x": 0.0})
    env.verser(H.verifier({"section": pa_, "fy": 235.0, "My": 100e6, "axes": ax_, "deversement": {"Lc": 4000.0}}), {"combinaison": "c1", "x": 4.0})
    hp_ = env.rendre()
    env = MO_.Enveloppe(H.METHODE, H.NORME)
    env.verser(appui, {"combinaison": "c1", "x": 0.0})
    env.verser(H.verifier({"section": pa_, "fy": 235.0, "My": 100e6, "axes": ax_, "deversement": {"Lc": 4000.0, "variante": "fascicule"}}),
               {"combinaison": "c1", "x": 4.0})
    hf_ = env.rendre()
    c("barre chargée — H-EQ gouverne à l'appui (M = 0) : les hypothèses gardent celles de la ligne du déversement (L_d et classe, origine du 3,9, "
      "« classes différentes » d'un PRS : une fois chacune) et, variante du fascicule, celles de la ligne des contraintes",
      hp_["gouvernant"] == hf_["gouvernant"] == "H-EQ" and DE.MESSAGE_3_9 not in appui["hypotheses"]
      and hp_["hypotheses"].count(DE.MESSAGE_3_9) == 1 and hp_["hypotheses"].count(DE.MESSAGE_CLASSES) == 1
      and sum(h.startswith("Déversement (classeurs") for h in hp_["hypotheses"]) == 1
      and any("variante du fascicule" in h for h in hf_["hypotheses"]) and len(hf_["hypotheses"]) == len(set(hf_["hypotheses"])),
      (hp_["gouvernant"], hp_["hypotheses"], hf_["hypotheses"]))

    # ---- la ligne qui répéterait l'autre : flexion simple d'axe fort, C = 1, classeurs
    base = {"section": heb, "fy": 235.0, "My": 40e6, "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}}}
    seul = H.verifier(base)
    avecN = H.verifier(dict(base, P=100e3))
    avecC = H.verifier(dict(base, C={"y": {"mode": "charges"}}))
    vd = verif_de(seul, "H-DEV")
    c("6.4 — flexion simple d'axe fort avec C = 1 : la ligne du déversement M/W ≤ σ_adm,d (le taux des classeurs) est seule — H10 y redonne le même "
      "taux (σ/σ_adm = (M/W)/σ_adm,d), la ligne des contraintes n'est pas répétée ; avec N, ou avec C = 0,85, les deux lignes sont rendues",
      [v["id"] for v in seul["verifications"]] == ["H-DEV"] and proche(seul["grandeurs"]["sigma"] / seul["grandeurs"]["sigma_adm"], vd["taux"])
      and [v["id"] for v in avecN["verifications"]] == ["H-N-z", "H-N-y", "H-SIGMA", "H-DEV"]
      and [v["id"] for v in avecC["verifications"]] == ["H-SIGMA", "H-DEV"]
      and proche(verif_de(avecC, "H-SIGMA")["taux"], 0.85 * vd["taux"]) and proche(verif_de(avecC, "H-DEV")["taux"], vd["taux"])
      and proche(avecN["grandeurs"]["sigma_My"], 40e6 / (avecN["grandeurs"]["deversement"]["phi"] * heb["Wel_y"] * avecN["grandeurs"]["y"]["PE"])),
      ([v["id"] for v in seul["verifications"]], [v["id"] for v in avecN["verifications"]]))
    c("6.4 — la ligne du déversement nomme sa référence (NBN B 51-002, § 4.3.3.5.3) ; l'origine du 3,9 (√(3π·√2,6), doc p. 18) est dite UNE "
      "fois, en une ligne, dans les hypothèses de la méthode (pas sur la ligne : la note la cite)",
      "4.3.3.5.3" in vd["reference"] and seul["hypotheses"].count(DE.MESSAGE_3_9) == 1 and DE.MESSAGE_3_9 not in vd["messages"]
      and "3,899" not in DE.MESSAGE_3_9 and len(DE.MESSAGE_3_9) < 100, (seul["hypotheses"], vd["messages"]))

    # ---- profils fermés (6.4, test 11 et 16 du § 11.5)
    rhs = SE.du_catalogue("RHS 200x100x6.3")
    b_ = {"section": rhs, "fy": 235.0, "My": 20e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}}
    l1 = H.verifier(b_)["grandeurs"]["deversement"]
    l2 = H.verifier(dict(b_, deversement={"facteur_doc": True}))["grandeurs"]["deversement"]
    l3 = DE.ferme(rhs, 6000.0, H.lambda_E(235.0), e_sur_g=210000.0 / 81000.0)
    c("test 11 du § 11.5 — G de la méthode ELS = E/2,6 = 80 769,23 N/mm² ; λ_ed(5,1)/λ_ed(π·√2,6) = 1,003383 (variante « facteur de la doc ») ; "
      "λ_ed(G = 81 000)/λ_ed(G = 80 769) = 0,999287", round(H.G, 2) == 80769.23 and l1["cas"] == l2["cas"] == "ferme"
      and round(l2["lambda"] / l1["lambda"], 6) == 1.003383 and round(l3["lambda"] / l1["lambda"], 6) == 0.999287 and l2["facteur"] == 5.1, (l1, l2))
    inconnu = dict(rhs, fabrication=None)
    ri = H.verifier(dict(b_, section=inconnu))
    froid = SE.du_catalogue(next(p["nom"] for p in CAT.profils() if p["famille"] == "RHS-F" and p["t"] > 6))
    rf = H.verifier(dict(b_, section=froid, My=5e6))
    c("test 16 du § 11.5 — RHS de fabrication inconnue → verdict BLOQUÉ (calcul en formé à froid, le cas défavorable) ; RHS formé à froid de plus "
      "de 6 mm → « hors domaine », jamais « vérifié »", ri["statut"] == H.NON_COUVERT and ri["grandeurs"]["z"]["courbe"] == "c"
      and any("Fabrication du tube" in x["message"] for x in ri["alertes"]) and rf["statut"] == H.HORS_DOMAINE
      and all(v["statut"] == H.HORS_DOMAINE for v in rf["verifications"]), (ri["statut"], rf["statut"]))
    cais = SE.saisie("autre", "caisson", nom="caisson 300 × 200 × 10", A=9600.0, Iy=1.21e8, Iz=6.44e7, iy=112.3, iz=81.9, Wel_y=8.07e5,
                     Wel_z=6.44e5, It=1.29e8)
    rc = H.verifier({"section": cais, "fy": 235.0, "My": 50e6, "axes": {"y": {"L": 8000.0}, "z": {"L": 8000.0}}})
    c("6.4 — un caisson soudé non recuit fléchi : la formule des profils fermés (doc p. 18), courbe b ; la flexion est couverte",
      rc["grandeurs"]["deversement"]["cas"] == "ferme" and rc["grandeurs"]["deversement"]["courbe"] == "b" and rc["statut"] == H.OK
      and proche(rc["grandeurs"]["deversement"]["lambda"], math.sqrt(math.pi * math.sqrt(2.6) * 8000 * 8.07e5 / math.sqrt(6.44e7 * 1.29e8))))
    shs = H.verifier({"section": SE.du_catalogue("SHS 150x150x6.3"), "fy": 235.0, "My": 20e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}})
    c("test 3 du § 11.5 — SHS : φ_ed = 1 (aucun déversement), la règle complémentaire dite ; φ_ed = 1 n'est pas φ = 1 : le flambement sous P reste vérifié",
      shs["grandeurs"]["deversement"] == {"cas": "sans_objet", "phi": 1.0} and any("règle complémentaire" in h for h in shs["hypotheses"])
      and H.verifier({"section": SE.du_catalogue("SHS 150x150x6.3"), "fy": 235.0, "P": 300e3, "My": 5e6,
                      "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}})["grandeurs"]["phi_min"] < 1)

    # ---- test 12 du § 11.5 (P67) : le contrôle hérité du classeur
    cat = SE.du_catalogue("HEB 200")
    a_ = cat["Wel_y"] / cat["Wpl_y"]
    k_ = (cat["h"] - 2 * cat["tf"]) ** 2 * cat["tw"] / (4 * cat["Wpl_y"])
    Mpl, Vpl = cat["Wpl_y"] * 235.0, cat["Avz"] * 235.0 / math.sqrt(3)

    def herite(m, v, C="unite", Lc=0.0):
        r = H.verifier({"section": cat, "fy": 235.0, "My": m * Mpl / 1.5, "Vz": v * Vpl / 1.5, "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}},
                        "C": {"y": {"mode": C}}, "deversement": {"Lc": Lc}})
        return verif_de(r, "H-EQ")["taux"], r
    rho = lambda v: (2 * v - 1) ** 2 if v > 0.5 else 0.0  # noqa: E731
    vs_ = [i / 10000 for i in range(10001)]
    marge = min(1 - k_ * rho(v) - a_ * math.sqrt(1 - v * v) for v in vs_)
    t1, _ = herite(0.8, 0.3)
    c("test 12 du § 11.5 — HEB 200, C_y = 1, φ_ed = 1 (a = 0,8865 ; k = 0,1012) : hérité M/M_pl,y ≤ a·√(1 − v²), toujours sous § 6.2.8 "
      "(1 − k·ρ ≥ 0,8988) ; le moteur rend bien le taux √((m/a)² + v²)",
      round(a_, 4) == 0.8865 and round(k_, 4) == 0.1012 and round(min(1 - k_ * rho(v) for v in vs_), 4) == 0.8988 and marge > 0.1
      and proche(t1, math.sqrt((0.8 / a_) ** 2 + 0.3 ** 2), 1e-9), (a_, k_, marge, t1))
    lb_cible = None
    lo, hi = 0.2, 0.5
    for _ in range(80):                       # λ̄_d tel que φ_d = 0,99 (classe c de la HEB 200) — puis L_d
        lb_cible = (lo + hi) / 2
        lo, hi = (lb_cible, hi) if DE.phi_d(lb_cible, "c") > 0.99 else (lo, lb_cible)
    Ld = (lb_cible * H.lambda_E(235.0) / 3.9) ** 2 * cat["b"] * cat["tf"] / cat["h"]
    t2, r2 = herite(1.02, 0.10, "charges", Ld)
    c("test 12 du § 11.5 — hors domaine (contre-exemple) : HEB 200, C_y = 0,85, φ_ed = 0,99 (a_eff = 1,0326), v = 0,10, M/M_pl,y = 1,02 : "
      "taux hérité 0,9929 (passe) quand l'EC3 donne 1,02 : le module ne dit nulle part « plus sévère que l'EC3 »",
      round(r2["grandeurs"]["deversement"]["phi"], 6) == 0.99 and abs(a_ * 0.99 / 0.85 - 1.0326) <= 1e-4 and round(t2, 4) == 0.9929
      and not any("plus sévère" in m for v in r2["verifications"] for m in v["messages"])
      and not any("plus sévère" in t for t in list(DE.CORRECTIONS.values()) + [DE.MESSAGE_CD, DE.MESSAGE_PRS, DE.MESSAGE_3_9]), (t2,))
    az = cat["Wel_z"] / cat["Wpl_z"]
    f_ = 2 * cat["tf"] * cat["b"] ** 2 / 4 / cat["Wpl_z"]
    Vply, Mplz = 2 * cat["b"] * cat["tf"] * 235.0 / math.sqrt(3), cat["Wpl_z"] * 235.0

    def herite_z(v):
        m = 1 - f_ * rho(v)                    # M_z à la limite de § 6.2.8 (la réduction porte sur les semelles)
        r = H.verifier({"section": cat, "fy": 235.0, "Mz": m * Mplz / 1.5, "Vy": v * Vply / 1.5, "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}}})
        return verif_de(r, "H-EQ")["taux"]
    c("test 12 du § 11.5 — axe faible + V_y : le contrôle hérité est moins sévère que § 6.2.8 dès v = 0,9505 (à v = 0,95 il l'est encore, à 0,951 plus) ; "
      "τ_max/(V/A_v,z) = 1,556 et (V/A_w)/τ_max = 1,043 : ce n'est pas un contrôle élastique local",
      herite_z(0.95) > 1.0 > herite_z(0.951) and round(az, 4) == round(cat["Wel_z"] / cat["Wpl_z"], 4)
      and round(cat["Avz"] * (cat["Wpl_y"] / 2) / (cat["Iy"] * cat["tw"]), 3) == 1.556
      and round((cat["Iy"] * cat["tw"] / (cat["Wpl_y"] / 2)) / ((cat["h"] - 2 * cat["tf"]) * cat["tw"]), 3) == 1.043, (herite_z(0.95), herite_z(0.951)))

    # ---- f_y nominale (test 37) et au-delà de 80 mm (19.1 point 5) : le moteur des barres
    hp = dict(MO_.EN.nouvelle_barre(), profil="HEB 300", mode="efforts", Lc=5.0, flambement={"y": {"L": 5.0, "u": 1.0}, "z": {"L": 5.0, "u": 1.0}},
              efforts={"service": {"N": 1000.0, "My": 50.0, "Mz": 0.0, "Vz": 0.0, "Vy": 0.0}, "elu": {"N": 1400.0, "My": 70.0, "Mz": 0.0, "Vz": 0.0, "Vy": 0.0}})
    r = MO_.verifier(hp)
    h_ = r["methodes"]["historique"]
    pa30 = section_prs(**PRS_A)
    c("test 37 du § 11.5 — f_y NOMINALE de la méthode ELS : HEB 300 S235 (t_f = 19 mm) → 235 N/mm² en méthode ELS, 225 en EC3 ; plats de 30 mm en "
      "S355 → 355 en méthode ELS, 345 en EC3 ; la note écrit « f_y nominale, comme les classeurs du bureau » (une ligne)",
      h_["grandeurs"]["fy"] == 235.0 and r["methodes"]["ec3"]["grandeurs"]["fy"] == 225.0 and MO_._fy_historique("S355", pa30)["fy"] == 355.0
      and ES.limite_elastique("S355", pa30) == 345.0 and "(f_y nominale, comme les classeurs du bureau)." in h_["hypotheses"][0]
      and "tableau" not in h_["hypotheses"][0]
      and H.verifier({"section": heb, "fy": 235.0, "fy_source": H.SOURCE_NOMINALE, "fy_ref": H.FY_NOMINALE, "P": 100e3,
                      "axes": {"y": {"L": 3000.0}, "z": {"L": 3000.0}}})["hypotheses"][0].endswith("= 156,7 N/mm² (f_y nominale, comme les classeurs du bureau)."),
      (h_["grandeurs"]["fy"], h_["hypotheses"][0]))
    hd = MO_.verifier(dict(MO_.EN.nouvelle_barre(), profil="HD 400x818.0", nuance="S355", maintiens=[{"x": 3.0, "genre": "complet"}]))
    hh = hd["methodes"]["historique"]
    c("19.1 point 5 — au-delà de 80 mm (HD 400x818 : t_f 97 mm), la méthode ELS se CALCULE avec la f_y nominale (355) : statut « hors domaine » "
      "(paroi que la doc ne prévoit pas), jamais « vérifié » ; l'EC3 reste refusé (non couvert)",
      hh["statut"] == H.HORS_DOMAINE and hh["taux_max"] is not None and hh["grandeurs"]["fy"] == 355.0 and "epaisseur" in hh["grandeurs"]["hors_domaine"]
      and any(x["code"] == "epaisseur" for x in hh["alertes"]) and hd["methodes"]["ec3"]["statut"] == H.NON_COUVERT
      and all(v["statut"] != H.OK for v in hh["verifications"]), (hh["statut"], hh.get("taux_max")))
    hl = H.verifier({"section": SE.du_catalogue("HL 920x970"), "fy": 235.0, "nuance": "S235", "My": 500e6,
                     "axes": {"y": {"L": 8000.0}, "z": {"L": 4000.0}}})
    c("19.1 point 5 / correction 3 — HL 920x970 (t_f 89,9 mm, h/b > 1,2) : classe d (pas de ligne au tableau 12 : note 3, prudent), la correction "
      "« seuils de 40 et 80 mm » est dite (la feuille « Calcul automatique » prenait 100 mm)", hl["grandeurs"]["deversement"]["classe"] == "d"
      and correction_dite(hl, "seuils_eurocode", "H-DEV") and hl["statut"] in (H.HORS_DOMAINE, H.NOK), hl["grandeurs"]["deversement"].get("classe"))
    r = MO_.verifier(dict(MO_.EN.nouvelle_barre(), maintiens=[{"x": 3.0, "genre": "complet"}]))
    h_ = r["methodes"]["historique"]
    c("la barre chargée prend le déversement des classeurs (L_d du tronçon : 3 m), C_d et la variante de la barre ; la ligne du déversement "
      "gouverne une poutre (IPE 300, 6 m, tenue à mi-portée), nomme ses corrections, et l'origine du 3,9 est dite une fois",
      verif_de(h_, "H-DEV") is not None and h_["gouvernant"] == "H-DEV" and h_["grandeurs"]["deversement"]["cas"] == "classeurs"
      and proche(h_["grandeurs"]["deversement"]["Lc"], 3000.0) and any(x["id"] == "axe_faible" for x in h_["corrections"])
      and r["barre"]["historique"]["Cd"] == 1.0 and r["barre"]["historique"]["variante"] == "classeurs"
      and h_["hypotheses"].count(DE.MESSAGE_3_9) == 1 and len(h_["hypotheses"]) == len(set(h_["hypotheses"])), (h_["gouvernant"], h_["hypotheses"]))
    nb_ = MO_.EN.nouvelle_barre()
    court = MO_.verifier(dict(nb_, travees=[1.0], cas=[{"nom": "G", "nature": "permanent", "N": 0.0,
                                                        "charges": [{"type": "q", "v": 400.0, "a": 0.0, "b": 1.0}]}]))["methodes"]["historique"]
    c("barre chargée — IPE 300 de 1 m sous 400 kN/m : H-EQ gouverne à l'appui (x = 0, M = 0), H-DEV est montrée ; ses hypothèses restent dites "
      "(« Déversement (classeurs… », origine du 3,9 : une fois), à leur place (avant les courbes, comme pour une section seule)",
      court["gouvernant"] == "H-EQ" and verif_de(court, "H-EQ")["x"] == 0.0 and verif_de(court, "H-DEV") is not None
      and court["hypotheses"].count(DE.MESSAGE_3_9) == 1 and sum(h.startswith("Déversement (classeurs") for h in court["hypotheses"]) == 1
      and [h[:12] for h in court["hypotheses"]] == ["Charges de s", "Déversement ", DE.MESSAGE_3_9[:12], "Courbes de f"],
      (court["gouvernant"], court["hypotheses"]))
    cont = MO_.verifier(dict(nb_, modele="continue", travees=[1.7, 2.2, 2.0], methodes=["historique"], cas=[
        {"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 102.0, "a": 0.0, "b": 5.9}, {"type": "P", "v": 326.0, "a": 2.28}]}]
    ))["methodes"]["historique"]
    dvs = [h for h in cont["hypotheses"] if h.startswith("Déversement")]
    eq_ = verif_de(cont, "H-EQ")
    c("barre chargée — poutre continue (1,7 + 2,2 + 2,0 m, IPE 300, sans maintien) : H-EQ gouverne à DROITE de l'appui B (V = 361,8 kN), "
      "avec le L_d de la travée 2 (2,20 m) : 223 % (214 % avec celui de la travée 1 : le défaut corrigé) ; H-DEV dans le même tronçon "
      "(160 %) : une seule ligne « Déversement »",
      cont["gouvernant"] == "H-EQ" and eq_["x"] == 1.7 and eq_["troncon"] == [1.7, 3.9] and round(eq_["taux"], 4) == 2.2315
      and verif_de(cont, "H-DEV")["troncon"] == [1.7, 3.9] and round(verif_de(cont, "H-DEV")["taux"], 3) == 1.598
      and len(dvs) == 1 and dvs[0].startswith("Déversement (classeurs, NBN B 51-002) : L_d = 2,20 m, ")
      and cont["hypotheses"].count(DE.MESSAGE_3_9) == 1 and len(cont["hypotheses"]) == len(set(cont["hypotheses"])),
      ([(v["id"], v["taux"], v["x"], v["troncon"]) for v in cont["verifications"]], cont["hypotheses"]))

    # ---- le SENS de dessin ne change rien : à une frontière de tronçons, chaque côté avec SON tronçon ; les bouts des
    #      tronçons (un maintien seul) sont des sections vérifiées
    def appui_P(trav, xP):
        return MO_.verifier(dict(nb_, modele="continue", travees=trav, methodes=["historique"], cas=[{"nom": "G", "nature": "permanent", "N": 0.0,
            "charges": [{"type": "q", "v": 30.0, "a": 0.0, "b": 6.0}, {"type": "P", "v": 300.0, "a": xP}]}]))["methodes"]["historique"]
    deux, deux_m = appui_P([2.0, 4.0], 1.75), appui_P([4.0, 2.0], 4.25)
    dvs = [h for h in deux["hypotheses"] if h.startswith("Déversement")]
    c("barre chargée — poutre continue 2 + 4 m (IPE 300, 30 kN/m, 300 kN à 1,75 m) : à l'appui, le côté gauche (L_d = 2 m) donne H-EQ "
      "175 %, le côté droit (L_d = 4 m) H-DEV 147 % ; deux lignes « Déversement » de moins de 110 caractères, chacune avec son tronçon et "
      "sa vérification, sans redire L_d ni la norme, celle de la gouvernante d'abord ; la poutre retournée donne les mêmes taux",
      deux["gouvernant"] == "H-EQ" and verif_de(deux, "H-EQ")["troncon"] == [0.0, 2.0] and verif_de(deux, "H-DEV")["troncon"] == [2.0, 6.0]
      and verif_de(deux, "H-EQ")["x"] == verif_de(deux, "H-DEV")["x"] == 2.0
      and round(deux["taux_max"], 4) == 1.7514 and round(verif_de(deux, "H-DEV")["taux"], 4) == 1.4724
      and dvs == ["Déversement sur 0–2 m (contrainte équivalente) : C_d = 1, classe b (h/b > 1,2, t_f ≤ 40 mm).",
                  "Déversement sur 2–6 m (déversement) : C_d = 1, classe b (h/b > 1,2, t_f ≤ 40 mm)."]
      and all(len(h) < 110 for h in dvs) and deux["hypotheses"].count(DE.MESSAGE_3_9) == 1
      and proche(deux_m["taux_max"], deux["taux_max"]) and proche(verif_de(deux_m, "H-DEV")["taux"], verif_de(deux, "H-DEV")["taux"]),
      (deux["gouvernant"], deux["taux_max"], deux_m["taux_max"], dvs))

    def lignes_dv(N=0.0, **k):
        b_ = dict(nb_, modele="continue", travees=[2.0, 4.0], methodes=["historique"], cas=[{"nom": "G", "nature": "permanent", "N": N,
                  "charges": [{"type": "q", "v": 30.0, "a": 0.0, "b": 6.0}, {"type": "P", "v": 300.0, "a": 1.75}]}], **k)
        return [h for h in MO_.verifier(b_)["methodes"]["historique"]["hypotheses"] if h.startswith("Déversement")]
    cd_, fa_, rh_ = (lignes_dv(50.0, historique=dict(nb_["historique"], Cd=1.13)),
                     lignes_dv(historique=dict(nb_["historique"], variante="fascicule")), lignes_dv(profil="RHS 200x100x6.3"))
    c("… la même poutre, lignes « Déversement » de moins de 110 caractères : C_d = 1,13 et N = 50 kN (deux titres : leur premier mot) ; "
      "variante du fascicule (L_d tu en fin de ligne) ; RHS 200x100x6.3 (le sujet garde « profil fermé », sans sa source)",
      cd_ == ["Déversement sur 0–2 m (contrainte équivalente) : C_d = 1,13, classe b (h/b > 1,2, t_f ≤ 40 mm).",
              "Déversement sur 2–6 m (flexion, déversement) : C_d = 1,13, classe b (h/b > 1,2, t_f ≤ 40 mm)."]
      and fa_ == ["Déversement sur 0–2 m (contrainte équivalente) : variante du fascicule p. 19.",
                  "Déversement sur 2–6 m (flexion) : variante du fascicule p. 19."]
      and rh_ == ["Déversement (profil fermé) sur 0–2 m (contrainte équivalente) : facteur π·√2,6 = 5,0657 (5,1 dans la doc).",
                  "Déversement (profil fermé) sur 2–6 m (flexion) : facteur π·√2,6 = 5,0657 (5,1 dans la doc)."]
      and all(len(h) < 110 for h in cd_ + fa_ + rh_), (cd_, fa_, rh_))
    # (intégration du lot B, point 28 a) le cas le plus long relevé : HEB 300 (h/b ≤ 1,2 : classe c), C_d = 1,13, N = 50 kN (deux titres),
    # L_d gardé par un maintien de la semelle haute (2,5 m sur le tronçon de 5 m) — 117 caractères avec la règle longue
    b28 = dict(nb_, profil="HEB 300", travees=[6.0], methodes=["historique"], historique=dict(nb_["historique"], Cd=1.13),
               maintiens=[{"x": 1.0, "genre": "complet"}, {"x": 3.5, "genre": "haut"}],
               cas=[{"nom": "G", "nature": "permanent", "N": 50.0, "charges": [{"type": "q", "v": 60.0, "a": 0.0, "b": 6.0},
                                                                              {"type": "P", "v": 600.0, "a": 0.4}]}])
    l28 = [h for h in MO_.verifier(b28)["methodes"]["historique"]["hypotheses"] if h.startswith("Déversement")]
    c("HEB 300, C_d = 1,13, N = 50 kN, maintien de la semelle haute dans le tronçon (L_d = 2,5 m gardé) : les deux lignes « Déversement » "
      "étiquetées restent sous 110 caractères — la règle de classe dit les seuls critères qui décident (« h/b ≤ 1,2, t_f ≤ 80 mm »)",
      l28 == ["Déversement sur 0–1 m (contrainte équivalente) : C_d = 1,13, classe c (h/b ≤ 1,2, t_f ≤ 80 mm).",
              "Déversement sur 1–6 m (flexion, déversement) : L_d = 2,50 m, C_d = 1,13, classe c (h/b ≤ 1,2, t_f ≤ 80 mm)."]
      and all(len(h) < 110 for h in l28), [(len(h), h) for h in l28])
    g28 = H.verifier({"section": SE.du_catalogue("HEB 300"), "fy": 235.0, "My": 100e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}},
                      "deversement": {"Lc": 2500.0, "Cd": 1.13}})
    h28 = [h for h in g28["hypotheses"] if h.startswith("Déversement")]
    c("… la règle de classe de la méthode : HEB 300 « h/b ≤ 1,2, t_f ≤ 80 mm » (classe c jusqu'à 80 mm, sans seuil à 40), sans « I laminé »",
      g28["grandeurs"]["deversement"]["regle"] == "h/b ≤ 1,2, t_f ≤ 80 mm"
      and h28 == ["Déversement (classeurs, NBN B 51-002) : L_d = 2,50 m, C_d = 1,13, classe c (h/b ≤ 1,2, t_f ≤ 80 mm)."], h28)

    def els(**k):
        return MO_.verifier(dict(nb_, poids_propre=False, methodes=["historique"], **k))["methodes"]["historique"]
    q20 = [{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 20.0, "a": 0.0, "b": 8.0}]}]
    r26, r62 = els(modele="continue", travees=[2.0, 6.0], cas=q20), els(modele="continue", travees=[6.0, 2.0], cas=q20)
    pm = [els(travees=[6.0], maintiens=[{"x": a, "genre": "complet"}],
              cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "P", "v": 40.0, "a": a}]}]) for a in (2.0, 4.0)]
    c("la poutre RETOURNÉE donne le même taux ELS : IPE 300 continue 2 + 6 m et 6 + 2 m sous 20 kN/m → 210 % (H-EQ à l'appui, côté de la "
      "travée de 6 m ; 173 % avant pour 2 + 6 m) ; 6 m, 40 kN et un maintien au même point, à 2 m ou à 4 m → 119 %, « nok » "
      "(85 % « ok » avant à 2 m)",
      proche(r26["taux_max"], r62["taux_max"]) and round(r26["taux_max"], 4) == 2.1011 and verif_de(r26, "H-EQ")["troncon"] == [2.0, 8.0]
      and verif_de(r62, "H-EQ")["troncon"] == [0.0, 6.0] and proche(pm[0]["taux_max"], pm[1]["taux_max"]) and round(pm[0]["taux_max"], 4) == 1.1886
      and pm[0]["statut"] == pm[1]["statut"] == H.NOK,
      (r26["taux_max"], r62["taux_max"], pm[0]["taux_max"], pm[1]["taux_max"]))
    m45 = els(profil="IPE 400", travees=[10.0], maintiens=[{"x": 4.5, "genre": "complet"}, {"x": 5.5, "genre": "complet"}],
              cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 15.0, "a": 0.0, "b": 10.0}]}])
    s4 = SE.du_catalogue("IPE 400")
    lb4 = 3.9 * math.sqrt(4500.0 * s4["h"] / (s4["b"] * s4["tf"])) / (math.pi * math.sqrt(210000.0 / 235.0))
    F4 = 1.0 + 0.339 * (lb4 - 0.2) + lb4 * lb4
    main4 = (15.0 * 4.5 * 5.5 / 2.0 * 1e6 / s4["Wel_y"]) / (235.0 / 1.5 * (F4 - math.sqrt(F4 * F4 - 4.0 * lb4 * lb4)) / (2.0 * lb4 * lb4))
    dv4 = verif_de(m45, "H-DEV")
    c("un maintien qui n'est ni un appui ni une charge borne un tronçon VÉRIFIÉ : IPE 400 de 10 m, 15 kN/m, maintiens à 4,5 et 5,5 m → "
      "H-DEV 198 % à x = 4,5 m avec L_d = 4,5 m (à la main : M = 185,6 kN·m, classe b) ; 119 % avant, pris dans le tronçon central de 1 m",
      dv4["x"] == 4.5 and dv4["troncon"] == [0.0, 4.5] and proche(dv4["taux"], main4, 1e-6) and round(dv4["taux"], 3) == 1.979,
      (dv4["taux"], main4, dv4["x"], dv4["troncon"]))
    rcl = MO_.verifier(dict(nb_, maintiens=[{"x": 3.0, "genre": "complet"}], historique={"classe_d": "d"}))
    rf1 = MO_.verifier(dict(nb_, profil="RHS 200x100x6.3", historique={"facteur_doc": True}))
    rf0 = MO_.verifier(dict(nb_, profil="RHS 200x100x6.3"))
    try:
        MO_.EN.normaliser(dict(nb_, historique={"classe_d": "e"}))
        refuse = False
    except H.Refus as e:
        refuse = "classe de déversement" in str(e)
    try:
        vide = MO_.EN.normaliser(dict(nb_, historique={"classe_d": ""}))["historique"]["classe_d"] is None
    except H.Refus:
        vide = False
    dcl, df1, df0 = (x["methodes"]["historique"]["grandeurs"]["deversement"] for x in (rcl, rf1, rf0))
    c("modèle d'entrée — les deux choix de 6.4 sont des champs de la barre, avec leur défaut (« facteur_doc » False, « classe_d » None : le "
      "tableau 12) : une classe imposée (d) et le facteur 5,1 de la doc (RHS) passent à la méthode ; une classe inconnue est refusée ; "
      "« classe_d » vide (\"\", une liste sans choix) vaut None",
      nb_["historique"]["facteur_doc"] is False and nb_["historique"]["classe_d"] is None and dcl["classe"] == "d" and dcl["classe_norme"] == "b"
      and any(x["code"] == "classe_d" for x in rcl["methodes"]["historique"]["alertes"]) and df1["cas"] == df0["cas"] == "ferme"
      and df1["facteur"] == 5.1 and abs(df0["facteur"] - math.pi * math.sqrt(2.6)) < 1e-12 and refuse and vide,
      (dcl.get("classe"), df1.get("facteur"), refuse, vide))
    pi_ = DE.classeurs(dict(section_prs(**PRS_A)), 4000.0, H.lambda_E(235.0), classe_imposee="b")
    c("6.4 — une classe IMPOSÉE : aucune correction de classe (ni « ligne axe faible » sur l'IPE 300 en d, ni « classe du type » qui citerait la "
      "classe imposée sur un PRS) ; l'alerte compare avec le tableau 12",
      rcl["methodes"]["historique"]["corrections"] == [] and pi_["corrections"] == [] and pi_["classe_norme"] == "c",
      (rcl["methodes"]["historique"]["corrections"], pi_["corrections"]))
    rv = MO_.verifier(dict(MO_.EN.nouvelle_barre(), maintiens=[{"x": 3.0, "genre": "complet"}],
                           historique={"s": 1.5, "C": "unite", "deversement": True, "tolerance": False, "Cd": 1.0, "variante": "fascicule"}))
    c("… la variante du fascicule choisie à la barre : calculée, dite, et la ligne des contraintes la porte (pas de ligne des classeurs)",
      rv["methodes"]["historique"]["grandeurs"]["deversement"]["cas"] == "I" and verif_de(rv["methodes"]["historique"], "H-DEV") is None
      and any("fascicule" in hh_ for hh_ in rv["methodes"]["historique"]["hypotheses"]))


# ------------------------------------------------------------------ 5 et 6. formules, catalogue, sortie
def cas_formules():
    heb = RECENTS["HEB 200"]
    return [barre_classeur(cl) for cl in list(RECENTS.values()) + list(ANCIENS.values())] + [
        barre_classeur(heb, P=200e3, My=50e6, Mz=10e6, Vz=60e3, Vy=10e3, axes={"y": {"L": 4000.0, "u": 1.2}, "z": {"L": 4000.0, "u": 0.8}},
                       C={"y": {"mode": "charges"}, "z": {"mode": "moments", "psi": 0.3}}, deversement={"actif": True, "Lc": 2500.0}),
        barre_classeur(heb, P=-150e3, My=30e6, Vz=40e3),
        {"section": SE.du_catalogue("RHS 200x100x6.3"), "fy": 355.0, "P": 80e3, "My": 20e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}},
        {"section": SE.du_catalogue("IPE 300"), "fy": 235.0, "My": 60e6, "Vz": 50e3, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}}},
        {"section": SE.du_catalogue("IPE 300"), "fy": 235.0, "P": 50e3, "My": 60e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}},
         "deversement": {"variante": "fascicule"}},
        {"section": SE.du_catalogue("RHS 200x100x6.3"), "fy": 235.0, "My": 20e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}},
         "deversement": {"facteur_doc": True}},
        {"section": SE.du_catalogue("HEB 200"), "fy": 235.0, "My": 60e6, "axes": {"y": {"L": 300.0}, "z": {"L": 300.0}}},
        {"section": section_prs(**PRS_C), "fy": 235.0, "nuance": "S235", "My": 2853e6, "Vz": 400e3, "axes": {"y": {"L": 28000.0}, "z": {"L": 3111.0}}},
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
        ("deuxième passe — le message « hors domaine » redit « la méthode historique »", 'v["messages"].append(f"Hors du domaine de la méthode ELS ({NORME}) — "',
         'v["messages"].append(f"Hors du domaine de la méthode historique ({NORME}) — "'),
        ("troisième passe — le message « hors domaine » dit « méthode ELS » sans la norme", 'v["messages"].append(f"Hors du domaine de la méthode ELS ({NORME}) — "',
         'v["messages"].append("Hors du domaine de la méthode ELS — "'),
        ("I laminé h/b ≤ 1,2 : courbes de h/b > 1,2", 'return {"y": "b", "z": "c", "regle": "I ou H laminé, h/b ≤ 1,2", "message": None}',
         'return {"y": "a", "z": "b", "regle": "I ou H laminé, h/b ≤ 1,2", "message": None}'),
        ("caisson non recuit en courbe a", '"caisson": ("b", "b")', '"caisson": ("a", "a")'),
        ("C sous charges = 1", "C_CHARGES = 0.85", "C_CHARGES = 1.0"),
        ("C des moments d'extrémité sans le terme 0,4·ψ", "return math.sqrt(0.3 * (1.0 + psi * psi) + 0.4 * psi)", "return math.sqrt(0.3 * (1.0 + psi * psi))"),
        ("déversement : courbe de l'axe fort", 'dev = deversement(s, Ld, fy, G["z"]["courbe"]', 'dev = deversement(s, Ld, fy, G["y"]["courbe"]'),
        ("déversement oublié dans le terme de moment", 'sMy = Cy * My / (phi_ed * s["Wel_y"] * G["y"]["PE"]) if My > 0 else 0.0',
         'sMy = Cy * My / (s["Wel_y"] * G["y"]["PE"]) if My > 0 else 0.0'),
        ("déversement appliqué aussi à l'axe faible", 'sMz = Cz * Mz / (s["Wel_z"] * G["z"]["PE"]) if Mz > 0 else 0.0',
         'sMz = Cz * Mz / (phi_ed * s["Wel_z"] * G["z"]["PE"]) if Mz > 0 else 0.0'),
        ("la dispense de la doc p. 20 reprise (φ_ed = 1 si P/(A·φ) ≤ 0,15·σ_adm)", "    phi_ed = dev[\"phi\"]\n",
         "    phi_ed = 1.0 if (comprime and P / (A * phi_min) <= 0.15 * sigma_adm) else dev[\"phi\"]\n"),
        ("second ordre oublié (PE = 1)", "PE = 1.0 if (Pcr is None or not comprime) else 1.0 - sec * P / Pcr", "PE = 1.0"),
        ("C oublié dans le terme de moment fort", "sMy = Cy * My / (phi_ed", "sMy = My / (phi_ed"),
        ("σ_eq avec τ² au lieu de 3·τ²", "sigma_eq = math.sqrt(sigma * sigma + 3.0 * tau * tau)", "sigma_eq = math.sqrt(sigma * sigma + tau * tau)"),
        ("aires de cisaillement échangées", 'tz = Vz / aires["ame"] if Vz > 0 else 0.0', 'tz = Vz / aires["ailes"] if Vz > 0 else 0.0'),
        ("tolérance de 5 % allumée par défaut", '"tolerance": bool(d.get("tolerance", False))', '"tolerance": bool(d.get("tolerance", True))'),
        ("déversement éteint par défaut", '"actif": bool(dev.get("actif", True))', '"actif": bool(dev.get("actif", False))'),
        ("alerte λ̄ > 1,8 retirée", "if lb > LAMBDA_BAR_TABLEAU:\n                alerte(f\"tableau_{a}\"", "if False:\n                alerte(f\"tableau_{a}\""),
        ("arrêt PE ≤ 0 retiré", 'instable = comprime and ((My > 0 and G["y"]["PE"] <= 0) or (Mz > 0 and G["z"]["PE"] <= 0))', "instable = False"),
        # (la synthèse est celle de acier/barre/resultat.py depuis le lot A : « une forme non couverte rendue ok » se mute là,
        #  tests/test_barre_ec3_section.py)
        ("hors domaine oublié (le verdict reste « ok »)", 'if raisons and v["statut"] == OK:', 'if False and v["statut"] == OK:'),
        ("domaine lu sur f_y et non sur la nuance", "ref = MA.nominale(nuance) if nuance else fy", "ref = fy"),
        ("creux formé à froid de plus de 6 mm laissé dans le domaine", '        hors.append("froid")', "        pass"),
        ("un « nok » hors domaine rendu « hors domaine »", 'if raisons and v["statut"] == OK:', 'if raisons:'),
        # ---- lot B (brief v12) : la méthode ELS
        ("6.7 — « une seule courbe par axe » retirée (une courbe inconnue passe)", "        if courbe is not None and courbe not in COURBES:\n",
         "        if False:\n"),
        ("lot B — la variante du fascicule redevient le défaut", '"variante": dev.get("variante", "classeurs")', '"variante": dev.get("variante", "fascicule")'),
        ("lot B — φ_ed des classeurs oublié dans le terme de moment fort (H10)", '    phi_ed = dev["phi"]\n', '    phi_ed = 1.0 if dev["cas"] == "classeurs" else dev["phi"]\n'),
        ("lot B — la ligne du déversement compare M/W à f_y/s (sans φ_d)", "              phi_ed * limite, \"N/mm²\"", "              limite, \"N/mm²\""),
        ("lot B — correction 4 retirée : σ_adm,d avec 157 N/mm² (la f_y du S235)", "              phi_ed * limite, \"N/mm²\"",
         "              phi_ed * DE.FY_CLASSEURS / sec, \"N/mm²\""),
        ("lot B — la ligne des contraintes répétée en flexion simple (C = 1)", 'repete = dev["cas"] == "classeurs" and P == 0 and Mz == 0 and Cy == 1.0',
         "repete = False"),
        ("lot B — la ligne des contraintes ôtée aussi avec C ≠ 1", 'repete = dev["cas"] == "classeurs" and P == 0 and Mz == 0 and Cy == 1.0',
         'repete = dev["cas"] == "classeurs" and P == 0 and Mz == 0'),
        ("lot B — la semelle comprimée toujours la haute", 'semelle = dv["semelle"] or ("haut" if d["My"] >= 0 else "bas")', 'semelle = dv["semelle"] or "haut"'),
        ("lot B — PRS monosymétrique laissé dans le domaine", 'hors_dev = ["prs_mono"] if (mono and dev["cas"] in ("classeurs", "I")) else []', "hors_dev = []"),
        ("intégration, point 4 — console à bout libre laissée dans le domaine", '    if dv["bout_libre"] and dev["cas"] in ("classeurs", "I", "ferme"):',
         "    if False:"),
        ("intégration, point 4 — l'indication « bout_libre » perdue à la lecture", '"bout_libre": bool(dev.get("bout_libre", False))}',
         '"bout_libre": False}'),
        ("lot B — paroi de plus de 80 mm laissée dans le domaine", '        hors.append("epaisseur")', "        pass"),
        ("lot B — le verdict bloqué oublié", "        if bloque and v[\"statut\"] in (OK, NOK, HORS_DOMAINE):", "        if False:"),
        ("lot B — I_y ≤ I_z (valeurs d'axe faible dans les cases fortes) non bloqué", 'if s["forme"] == "I" and (s["Iy"] <= s["Iz"] or s["iy"] <= s["iz"]):',
         "if False:"),
        ("lot B — bords des semelles non saisis : calcul en oxycoupé (le cas favorable)", 's = dict(s0, fabrication="soude_larges_plats")',
         's = dict(s0, fabrication="soude_oxycoupe")'),
        ("lot B — fabrication du tube inconnue : verdict rendu", '            s = dict(s0, fabrication="creux_froid")\n            bloque.append(',
         '            s = dict(s0, fabrication="creux_froid")\n            (lambda x: None)('),
        ("lot B — la correction « P_cr avec u » n'est plus dite", 'a_corriger.append(("pcr_u", "H-SIGMA", {}))', "pass"),
        ("lot B — la correction « φ_min » n'est plus dite", 'a_corriger.append(("phi_min", "H-SIGMA", {}))', "pass"),
        ("lot B — |V_y| + |V_z| : les valeurs signées du classeur", 'My, Mz, Vz, Vy = abs(d["My"]), abs(d["Mz"]), abs(d["Vz"]), abs(d["Vy"])',
         'My, Mz, Vz, Vy = abs(d["My"]), abs(d["Mz"]), d["Vz"], d["Vy"]'),
        ("lot B — la correction « |V_y| + |V_z| » n'est plus dite", 'a_corriger.append(("cisaillement_abs", "H-EQ", {}))', "pass"),
        ("lot B — la correction « f_y de la nuance » n'est plus dite", 'a_corriger.append(("fy_nuance", "H-DEV", {"fy": _fr(fy)}))', "pass"),
        ("lot B — les corrections du déversement ne sont plus dites", '            a_corriger.append((k, "H-DEV", {"prolonge"', '            (lambda *x: None)((k, "H-DEV", {"prolonge"'),
        ("lot B — « Correction du classeur » absente de la vérification", 'par_id[vid]["messages"].append(f"Correction du classeur : {c[\'texte\']}")', "pass"),
        ("lot B — PRS : le message « formule prudente » oublié", "            msg_dev.append(DE.MESSAGE_PRS)\n", "            pass\n"),
        ("lot B — l'origine du 3,9 n'est plus dite", "        hyp.append(DE.MESSAGE_3_9)", "        str(DE.MESSAGE_3_9)"),
        ("lot B — l'origine du 3,9 redite sur la ligne du déversement (deux fois par méthode)", "        if prs:\n            msg_dev.append(DE.MESSAGE_PRS)\n",
         "        msg_dev.append(DE.MESSAGE_3_9)\n        if prs:\n            msg_dev.append(DE.MESSAGE_PRS)\n"),
        # (relecture du lot B) ce que la contre-vérification a vu sans garde
        ("relecture — PRS monosymétrique : la variante du fascicule laissée dans le domaine",
         'hors_dev = ["prs_mono"] if (mono and dev["cas"] in ("classeurs", "I")) else []', 'hors_dev = ["prs_mono"] if (mono and dev["cas"] == "classeurs") else []'),
        ("relecture — PRS monosymétrique : H-SIGMA et H-EQ laissés dans le domaine", 'ids_dev = {"H-DEV", "H-SIGMA", "H-EQ"} if', 'ids_dev = {"H-DEV"} if'),
        ("relecture — « classes différentes » des PRS n'est plus dit", "            hyp.append(DE.MESSAGE_CLASSES)", "            pass"),
        ("relecture — PRS monosymétrique : W de la semelle comprimée (et non W_el,y,min) dans σ_M", 'sMd = My / s["Wel_y"] if s.get("Wel_y") else None',
         'sMd = My / s.get("Wel_y_" + semelle, s["Wel_y"]) if s.get("Wel_y") else None'),
        ("relecture — PRS monosymétrique : W de la semelle comprimée dans H10", 'sMy = Cy * My / (phi_ed * s["Wel_y"] * G["y"]["PE"]) if My > 0 else 0.0',
         'sMy = Cy * My / (phi_ed * s.get("Wel_y_" + semelle, s["Wel_y"]) * G["y"]["PE"]) if My > 0 else 0.0'),
        ("relecture — correction 4 dite pour toute f_y ≠ 235 (S275, S460)", "if fy == DE.FY_FEUILLE_S355:", "if fy != DE.FY_CLASSEURS:"),
        ("relecture — S460 : la colonne « f_y ≤ 430 » du tableau 12 prise sans le dire", '        col = f" — {DE.MESSAGE_COLONNE} (À VÉRIFIER)" if',
         '        col = "" if'),
        ("relecture — λ̄ ≤ 0,2 : deux lignes pour φ_d = 1", 'elif "phid_plafond" not in dev["corrections"]:', "else:"),
        ("relecture — PRS à I_y ≤ I_z : le message des cases échangées", "bloque.append(MESSAGE_PRS_AXE_FAIBLE if",
         'bloque.append("I_y ≤ I_z : des valeurs de l\'axe faible dans les cases de l\'axe fort ? Verdict bloqué." if'),
        ("fin du bloc — PRS à I_y ≤ I_z : la phrase ambiguë revient (elle ne dit pas pourquoi)",
         'MESSAGE_PRS_AXE_FAIBLE = "I_y ≤ I_z : les charges (plan de l\'âme) fléchissent ce PRS autour de son axe faible — verdict bloqué."',
         'MESSAGE_PRS_AXE_FAIBLE = "I_y ≤ I_z : l\'axe de l\'âme n\'est pas l\'axe fort de ce PRS — verdict bloqué."'),
        ("fin du bloc — S460 : « classe lue pour f_y ≤ 430 » dite aussi d'une classe imposée", 'if fy > DE.FY_COLONNE_12 and dv["classe"] is None else ""',
         'if fy > DE.FY_COLONNE_12 else ""'),
        ("reprise — PRS, classe imposée : « classes du flambement et du déversement (tableau 12) différentes » dit quand même",
         'if prs and dv["classe"] is None:', "if prs:"),
        ("reprise — S460, classe imposée : l'alerte tait la colonne « f_y ≤ 430 » du tableau 12", 'col_i = f" ; {DE.MESSAGE_COLONNE}, À VÉRIFIER" if',
         'col_i = "" if'),
        ("reprise — la colonne « f_y ≤ 430 » dite dans l'alerte à toute f_y (S355)", 'À VÉRIFIER" if fy > DE.FY_COLONNE_12 else ""',
         'À VÉRIFIER" if True else ""'),
        ("relecture — classe de déversement à fournir : toute la méthode refusée", "except DE.AFournir as e:", "except ZeroDivisionError as e:"),
        ("relecture — classe à fournir : la flexion jugée quand même (φ_ed = 1)", '            dev = {"cas": "a_fournir", "phi": None}',
         '            dev = {"cas": "maintenu", "phi": 1.0}'),
        ("lot B — C_d > 1 sous une charge sur la semelle haute : sans message", 'if dev["Cd"] > 1.0 and dv["niveau"] == "haut":', "if False:"),
        ("lot B — le facteur 5,1 de la doc ignoré", 'dev = DE.ferme(s, Lc, lE, DE.FACTEUR_DOC if facteur_doc else None)', "dev = DE.ferme(s, Lc, lE, None)"),
        ("lot B — un caisson laissé non couvert en flexion", 'couvert = s["forme"] in ("I",) + SE.TUBES or s.get("fabrication") in DE.CAISSONS',
         'couvert = s["forme"] in ("I",) + SE.TUBES'),
        ("lot B — f_y nominale écrite « f_y : » comme une source de tableau", 'f" ({d[\'fy_source\']})." if nominale else', 'f" (f_y : {d[\'fy_source\']})." if nominale else'),
        ("lot B — PRS : l'A_v,z de l'Eurocode 3 (1,20·h_w·t_w) prise pour l'âme du contrôle hérité", 'ame = p["hw"] * p["tw"] if ame is None else ame',
         'ame = s["Avz"] if ame is None else ame'),
        ("lot B — tube rond ou carré : la règle complémentaire n'est plus dite", '        hyp.append("Tube rond ou carré : pas de déversement', '        str("Tube rond ou carré : pas de déversement'),
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
    # le DÉVERSEMENT de la méthode ELS (acier/barre/historique_deversement.py) : ses mutations, rejouées sur la méthode
    source_de = open(CHEMIN_DEVERSEMENT, encoding="utf-8").read()
    cas_de = [
        ("lot B — 3,9 remplacé par 3,899", "K_CLASSEURS = 3.9 ", "K_CLASSEURS = 3.899 "),
        ("lot B — C_d oublié dans λ_e,d", 'math.sqrt(Ld * g["h"] / (g["b"] * g["tf"] * Cd))', 'math.sqrt(Ld * g["h"] / (g["b"] * g["tf"]))'),
        ("lot B — λ_e,d sans la hauteur (l/b·t_f)", 'math.sqrt(Ld * g["h"] / (g["b"] * g["tf"] * Cd))', 'math.sqrt(Ld * 100.0 / (g["b"] * g["tf"] * Cd))'),
        ("lot B — n de la classe c pris à 0,339", '"c": 0.489', '"c": 0.339'),
        ("lot B — n de la classe d pris à 0,489", '"d": 0.756', '"d": 0.489'),
        ("lot B — φ_d : 4·λ̄² remplacé par λ̄²", "(Phi - math.sqrt(Phi * Phi - 4.0 * lambda_bar * lambda_bar))", "(Phi - math.sqrt(Phi * Phi - lambda_bar * lambda_bar))"),
        ("lot B — correction 1 retirée : φ_d sans le palier de 0,2", "    if lambda_bar <= PALIER:\n        return 1.0\n", "\n"),
        ("lot B — correction 2 retirée : « tf < 8 » (t_f = 80 mm sans classe c)", 'return "c" if tf <= TF_80 else "d"\n', 'return "c" if tf < TF_80 else "d"\n'),
        ("lot B — correction 2 retirée : h/b > 1,2 et 40 < t_f ≤ 80 sans la classe c", '"b" if tf <= TF_40 else "c" if tf <= TF_80 else "d"',
         '"b" if tf <= TF_40 else "d"'),
        ("lot B — correction 3 retirée : le seuil de 100 mm de l'Eurocode", "TF_40, TF_80 = 40.0, 80.0 ", "TF_40, TF_80 = 40.0, 100.0 "),
        ("lot B — correction 5 retirée : un U reçoit la classe d'un I laminé", 'cl, regle = "c", "U"', 'cl, regle = lam, "U"'),
        ("lot B — PRS oxycoupé en classe d'un I laminé", 'cl, regle = ("c", "oxycoupé, t_f ≤ 40 mm") if tf <= TF_40',
         'cl, regle = (lam, "oxycoupé, t_f ≤ 40 mm") if tf <= TF_40'),
        ("lot B — PRS en larges plats en classe c", 'cl, regle = "d", "larges plats"', 'cl, regle = "c", "larges plats"'),
        # (intégration du lot B, point 28 a ; les deux mutations du PRS ci-dessus suivent la règle courte — le harnais suit)
        ("intégration, point 28 a — la règle de classe longue revenue (« I laminé, … » : la ligne étiquetée dépasse 110 caractères)",
         "regle = f\"h/b {'>' if elance else '≤'} 1,2, {ep}\"", "regle = f\"I laminé, h/b {'>' if elance else '≤'} 1,2, {ep}\""),
        ("lot B — la correction 6 (axe faible seul) n'est plus dite", '        corr.append("axe_faible")', "        pass"),
        ("lot B — PRS : la semelle haute toujours, même comprimée en bas", 'b, tf = (p["b1"], p["t1"]) if semelle == "haut" else (p["b2"], p["t2"])',
         'b, tf = (p["b1"], p["t1"])'),
        ("lot B — fascicule : âme entière au lieu d'un sixième", 'Ad = g["b"] * g["tf"] + g["hw"] * g["tw"] / 6.0', 'Ad = g["b"] * g["tf"] + g["hw"] * g["tw"]'),
        ("lot B — fascicule : inertie de la semelle en b²", 'Id = g["tf"] * g["b"] ** 3 / 12.0', 'Id = g["tf"] * g["b"] ** 2 / 12.0'),
        ("tube rectangulaire : E/G = 2 au lieu de 2,6", "E_SUR_G = 2.6 ", "E_SUR_G = 2.0 "),
        ("lot B — le facteur de la doc 5,1 remplacé par 5,0657", "FACTEUR_DOC = 5.1 ", "FACTEUR_DOC = 5.0657 "),
        ("lot B — C_d < 1 admis", "CD_MIN, CD_MAX = 1.0, 3.0\n", "CD_MIN, CD_MAX = 0.5, 3.0\n"),
        ("relecture — une classe imposée : celle du tableau 12 n'est plus lue (l'alerte « classe imposée » se tait)",
         "        cl = classe(s, g[\"h\"], g[\"b\"], g[\"tf\"])\n    except AFournir:", "        cl = None\n    except AFournir:"),
        ("relecture — correction 6 dite aussi pour un laminé de plus de 80 mm", 'if forme == "I" and not (fab == "lamine" and tf > TF_80):', 'if forme == "I":'),
        ("relecture — correction 4 dite à toute f_y (la constante de la feuille S355 perdue)", "FY_FEUILLE_S355 = 355.0", "FY_FEUILLE_S355 = 275.0"),
        ("fin du bloc — une classe imposée : les corrections de classe du tableau 12 dites quand même",
         'corr = list(cl["corrections"]) if (cl and classe_imposee is None) else []', 'corr = list(cl["corrections"]) if cl else []'),
        ("fin du bloc — « à fournir » : les clés internes du profil dans le message", "pour {PROFILS_A_FOURNIR.get(fab, 'ce profil')} — ",
         "({forme}, {fab}) — "),
        ("reprise 2 — profil fermé : « Facteur » avec une majuscule au milieu de la phrase", 'MESSAGE_FERME = "facteur π·√2,6', 'MESSAGE_FERME = "Facteur π·√2,6'),
    ]
    for nom, ancien, nouveau in cas_de:
        if source_de.count(ancien) != 1:
            res.append((nom, None))
            continue
        try:
            spec = importlib.util.spec_from_file_location("acier.barre.historique_deversement_mute", CHEMIN_DEVERSEMENT)
            de_m = importlib.util.module_from_spec(spec)
            de_m.__package__ = "acier.barre"
            exec(compile(source_de.replace(ancien, nouveau), CHEMIN_DEVERSEMENT, "exec"), de_m.__dict__)
            spec = importlib.util.spec_from_file_location("acier.barre.historique_mute", CHEMIN_MOTEUR)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "acier.barre"
            mod.__dict__["DE"] = de_m
            exec(compile(source.replace("from . import historique_deversement as DE\n", ""), CHEMIN_MOTEUR, "exec"), mod.__dict__)
            rouges = [n for n, ok, _ in controles(mod) if not ok]
        except Exception as e:  # noqa: BLE001
            rouges = [f"exception {type(e).__name__}: {e}"]
        res.append((nom, rouges))
    # le MOTEUR DES BARRES (acier/barre/moteur.py) : la f_y qu'il passe à la méthode ELS
    source_mo = open(CHEMIN_BARRE, encoding="utf-8").read()
    cas_mo = [
        ("lot B — A9 revient : la f_y de l'Eurocode 3 selon l'épaisseur", 'return {"fy": MA.nominale(nuance), "nuance": nuance,',
         'return {"fy": ES.limite_elastique(nuance, s), "nuance": nuance,'),
        # (intégration du lot B, point 4, le harnais suit : l'appel passe aussi la console à bout libre)
        ("lot B — la barre chargée oublie C_d et la variante", '"deversement": _deversement_historique(b, Lc * 1e3, libre is not None)})',
         '"deversement": {"actif": b["historique"]["deversement"], "Lc": Lc * 1e3}})'),
        ("sens de dessin — à une frontière, le tronçon de GAUCHE pour les deux côtés (le défaut d'avant)",
         "    if gauche is not None and droite is not None and gauche is not droite:\n", "    if False:\n"),
        ("sens de dessin — à une frontière, les côtés échangés (le gauche avec le tronçon de droite)",
         'return [(gauche, (e["M_g"], e["V_g"])), (droite, (e["M_d"], e["V_d"]))]', 'return [(droite, (e["M_g"], e["V_g"])), (gauche, (e["M_d"], e["V_d"]))]'),
        ("sens de dessin — les bouts des tronçons hors des sections vérifiées (un maintien seul jamais vérifié)",
         'for x in sorted(set(xs) | {tr[k] for tr in trs for k in ("x1", "x2")}):', "for x in xs:"),
        ("relecture — la barre oublie le facteur 5,1 de la doc", '"facteur_doc": h["facteur_doc"],\n', '"facteur_doc": False,\n'),
        ("relecture — la barre oublie la classe de déversement imposée", '"classe": h["classe_d"], ', '"classe": None, '),
    ]
    for nom, ancien, nouveau in cas_mo:
        if source_mo.count(ancien) != 1:
            res.append((nom, None))
            continue
        try:
            spec = importlib.util.spec_from_file_location("acier.barre.moteur_mute", CHEMIN_BARRE)
            mo_m = importlib.util.module_from_spec(spec)
            mo_m.__package__ = "acier.barre"
            exec(compile(source_mo.replace(ancien, nouveau), CHEMIN_BARRE, "exec"), mo_m.__dict__)
            rouges = [n for n, ok, _ in controles(HI, mo_m) if not ok]
        except Exception as e:  # noqa: BLE001
            rouges = [f"exception {type(e).__name__}: {e}"]
        res.append((nom, rouges))
    # l'ENVELOPPE de la barre chargée (acier/barre/enveloppe.py) : le texte des corrections
    chemin_en = os.path.join(RACINE, "acier", "barre", "enveloppe.py")
    source_en = open(chemin_en, encoding="utf-8").read()
    import_en = "from .enveloppe import IDS_BARRE, ORDRE, Enveloppe          # noqa: F401 — ORDRE, relu par les tests\n"
    cas_en = [
        ("relecture — l'enveloppe garde le texte de la PREMIÈRE occurrence d'une correction", "                    corrections.setdefault(c[\"id\"], c)\n",
         "                    pass\n"),
        ("fin du bloc — les hypothèses de la seule gouvernante (H-EQ à l'appui : le déversement et le 3,9 perdus)",
         'HYPOTHESES_AUSSI = {"historique": ("H-SIGMA", "H-DEV")}', "HYPOTHESES_AUSSI = {}"),
        ("fin du bloc — les hypothèses de la ligne du déversement oubliées", 'HYPOTHESES_AUSSI = {"historique": ("H-SIGMA", "H-DEV")}',
         'HYPOTHESES_AUSSI = {"historique": ("H-SIGMA",)}'),
        ("fin du bloc — les hypothèses de la ligne des contraintes oubliées (variante du fascicule)", 'HYPOTHESES_AUSSI = {"historique": ("H-SIGMA", "H-DEV")}',
         'HYPOTHESES_AUSSI = {"historique": ("H-DEV",)}'),
        ("fin du bloc — hypothèses répétées (gouvernante et ligne du déversement)", "                if j is None:\n", "                if True:\n"),
        ("fin du bloc — hypothèses ajoutées en fin de liste (après les courbes de flambement)", "j = pareils[-1] + 1 if pareils else i\n",
         "j = pareils[-1] + 1 if pareils else len(lignes)\n"),
        ("reprise — deux lignes « Déversement » (deux tronçons) sans leur tronçon ni leur vérification",
         '                h = self._etiquetee(s, h[len(s) + 3:], ids)\n', "                pass\n"),
        ("sens de dessin — la ligne étiquetée redit L_d (la longueur de son tronçon)", "                if corps.startswith(Ld + sep):\n",
         "                if False:\n"),
        ("sens de dessin — la ligne étiquetée redit la norme (« (classeurs, NBN B 51-002) »)",
         "return f\"{_sans_source(s)} {self._origine(ids)} : {corps}\"", "return f\"{s} {self._origine(ids)} : {corps}\""),
        ("reprise 2 — la ligne étiquetée redit L_d en fin (variante du fascicule)", '            if corps.endswith(f", {Ld}."):\n',
         "            if False:\n"),
        ("reprise 2 — le sujet étiqueté perd aussi ce qui décrit le cas (« profil fermé »)",
         "    return f\"{tete} ({', '.join(gardes)})\" if gardes else tete\n", "    return tete\n"),
        ("reprise 2 — deux titres gardés entiers (« (flexion composée, déversement) » : plus de 110 caractères)",
         "        par_troncon = {t: ts if len(ts) == 1 else [x.split(\" \")[0] for x in ts] for t, ts in par_troncon.items()}\n", ""),
        ("reprise — la ligne « Déversement » d'un autre tronçon placée avant celle de la gouvernante", "j = pareils[-1] + 1 if pareils else i\n",
         "j = i\n"),
    ]
    for nom, ancien, nouveau in cas_en:
        if source_en.count(ancien) != 1 or source_mo.count(import_en) != 1:
            res.append((nom, None))
            continue
        try:
            spec = importlib.util.spec_from_file_location("acier.barre.enveloppe_mute", chemin_en)
            en_m = importlib.util.module_from_spec(spec)
            en_m.__package__ = "acier.barre"
            exec(compile(source_en.replace(ancien, nouveau), chemin_en, "exec"), en_m.__dict__)
            spec = importlib.util.spec_from_file_location("acier.barre.moteur_mute", CHEMIN_BARRE)
            mo_m = importlib.util.module_from_spec(spec)
            mo_m.__package__ = "acier.barre"
            mo_m.__dict__.update(IDS_BARRE=en_m.IDS_BARRE, ORDRE=en_m.ORDRE, Enveloppe=en_m.Enveloppe)
            exec(compile(source_mo.replace(import_en, ""), CHEMIN_BARRE, "exec"), mo_m.__dict__)
            rouges = [n for n, ok, _ in controles(HI, mo_m) if not ok]
        except Exception as e:  # noqa: BLE001
            rouges = [f"exception {type(e).__name__}: {e}"]
        res.append((nom, rouges))
    # le MODÈLE D'ENTRÉE de la barre (acier/barre/entree.py) : ses mutations, rejouées sur le moteur des barres
    chemin_ent = os.path.join(RACINE, "acier", "barre", "entree.py")
    source_ent = open(chemin_ent, encoding="utf-8").read()
    import_ent = "from . import entree as EN\n"
    cas_ent = [
        ("fin du bloc — « classe_d » vide (\"\", une liste sans choix) refusée", 'if h["classe_d"] == "":', "if False:"),
    ]
    for nom, ancien, nouveau in cas_ent:
        if source_ent.count(ancien) != 1 or source_mo.count(import_ent) != 1:
            res.append((nom, None))
            continue
        try:
            spec = importlib.util.spec_from_file_location("acier.barre.entree_mute", chemin_ent)
            ent_m = importlib.util.module_from_spec(spec)
            ent_m.__package__ = "acier.barre"
            exec(compile(source_ent.replace(ancien, nouveau), chemin_ent, "exec"), ent_m.__dict__)
            spec = importlib.util.spec_from_file_location("acier.barre.moteur_mute", CHEMIN_BARRE)
            mo_m = importlib.util.module_from_spec(spec)
            mo_m.__package__ = "acier.barre"
            mo_m.__dict__["EN"] = ent_m
            exec(compile(source_mo.replace(import_ent, ""), CHEMIN_BARRE, "exec"), mo_m.__dict__)
            rouges = [n for n, ok, _ in controles(HI, mo_m) if not ok]
        except Exception as e:  # noqa: BLE001
            rouges = [f"exception {type(e).__name__}: {e}"]
        res.append((nom, rouges))
    # la SYNTHÈSE des statuts est celle de acier/barre/resultat.py (lot A) : ses mutations, rejouées sur la méthode historique
    chemin_re = os.path.join(RACINE, "acier", "barre", "resultat.py")
    source_re = open(chemin_re, encoding="utf-8").read()
    cas_re = [
        ("une forme non couverte rendue « ok » (synthèse)", '    elif any(v["statut"] == NON_COUVERT for v in verifs):\n        statut = NON_COUVERT',
         '    elif False:\n        statut = NON_COUVERT'),
        ("lot A — hors domaine rendu « ok » par la synthèse", '    elif any(v["statut"] == HORS_DOMAINE for v in verifs):\n        statut = HORS_DOMAINE',
         '    elif False:\n        statut = HORS_DOMAINE'),
        ("lot A — un taux hors domaine ignoré par la synthèse", 'juges = [v for v in verifs if v["statut"] in (OK, NOK, HORS_DOMAINE)]',
         'juges = [v for v in verifs if v["statut"] in (OK, NOK)]'),
    ]
    for nom, ancien, nouveau in cas_re:
        if source_re.count(ancien) != 1:
            res.append((nom, None))
            continue
        try:
            spec = importlib.util.spec_from_file_location("acier.barre.resultat_mute", chemin_re)
            re_m = importlib.util.module_from_spec(spec)
            re_m.__package__ = "acier.barre"
            exec(compile(source_re.replace(ancien, nouveau), chemin_re, "exec"), re_m.__dict__)
            spec = importlib.util.spec_from_file_location("acier.barre.historique_mute", CHEMIN_MOTEUR)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "acier.barre"
            mod.__dict__["RE"] = re_m
            exec(compile(source.replace("from . import resultat as RE\n", ""), CHEMIN_MOTEUR, "exec"), mod.__dict__)
            rouges = [n for n, ok, _ in controles(mod) if not ok]
        except Exception as e:  # noqa: BLE001
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
    chk("chaque gabarit de la méthode historique sert à un contrôle (aucun n'est mort)", utilises == set(FO.GABARITS_HISTORIQUE),
        set(FO.GABARITS_HISTORIQUE) ^ utilises)
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
