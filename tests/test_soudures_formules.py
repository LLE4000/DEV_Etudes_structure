# -*- coding: utf-8 -*-
"""Les formules de la note (acier/soudures/formules.py) : chaque substitution
imprimée, ré-évaluée avec ses nombres tels qu'imprimés, redonne le résultat
du moteur — pour chaque méthode (directionnelle, simplifiée, ELS, pleine
pénétration à l'ELU et en service) et sur plusieurs configurations ; les
critères en typographie mathématique se composent ; la note n'imprime que
la méthode choisie.

Lancement : python3 tests/test_soudures_formules.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.soudures import etat as ET  # noqa: E402
from acier.soudures import formules as FO  # noqa: E402
from ndc_pdf import mathx as M  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def verif(config, params=None, efforts=None, methode=None):
    cas = ET.nouveau_cas(config)
    cas["params"].update(params or {})
    cas["efforts"].update(efforts or {})
    cas["methode"].update(methode or {})
    return ET.verifier(cas)


print("=== 1. Chaque substitution redonne le résultat du moteur ===")
CAS = [("C3 directionnelle", verif("C3")), ("C3 simplifiée", verif("C3", methode={"ec": "simplifiee"})),
       ("C3 ELS", verif("C3", methode={"mode": "els"})), ("C1", verif("C1")), ("C2 M", verif("C2", {}, {"M": 2.0})),
       ("C2 pleine (plats de 30)", verif("C2", {"jonction": "pleine", "tB": 30, "tA": 30}, {"N": 400, "M": 10})),
       ("C2 pleine ELS", verif("C2", {"jonction": "pleine", "tB": 30, "tA": 30}, {"N": 400}, {"mode": "els"})),
       ("C4", verif("C4")), ("C5", verif("C5")), ("C5 semelles pleines", verif("C5", {"jonction_semelles": "pleine"})),
       ("C6 CHS tronçons", verif("C6", {"profil": "CHS 114.3x5", "troncons": {"n": 4, "l": 60}})), ("C7 pleine", verif("C7")),
       ("C7 partielle", verif("C7", {"penetration": "partielle"}, {"M": 2.0})), ("C8", verif("C8"))]
n_subs = 0
for nom, r in CAS:
    t = FO.toutes(r)
    subs = t["groupe"] + t["point"] + t["critere"]
    n_subs += len(subs)
    mauvaises = [(FO.rendre(s), v) for s in subs if not FO.verifier(s)[0] for v in [FO.verifier(s)[1]]]
    chk(f"{nom} : {len(subs)} substitutions, toutes vérifiées, dont le taux η", subs and not mauvaises and any(s.nom == "η" for s in t["critere"]), str(mauvaises[:2]))
chk(f"{n_subs} substitutions en tout (≥ 150)", n_subs >= 150, str(n_subs))

print("\n=== 2. Ce qui s'écrit ===")
L = FO.lignes(verif("C3"))
chk("C3 directionnelle : A_w = 2·a·l_eff = 2 × 5 × 190 = 1900 mm²", "A_w = 2·a·l_eff = 2 × 5 × 190 = 1900 mm²" in L, str(L[:1]))
chk("|f| = √(f_x² + f_y² + f_z²) = √(0² + −157,9² + 747,9²) = 764 N/mm", "|f| = √(f_x² + f_y² + f_z²) = √(0² + −157,9² + 747,9²) = 764 N/mm" in L, str(L[4:5]))
chk("σ_c = √(σ⊥² + 3·(τ⊥² + τ∥²)) = √(105,8² + 3 × (105,8² + 31,58²)) = 218,5 N/mm²",
    "σ_c = √(σ⊥² + 3·(τ⊥² + τ∥²)) = √(105,8² + 3 × (105,8² + 31,58²)) = 218,5 N/mm²" in L, str(L[8:9]))
chk("lim_c = fu/(βw·γM2) = 360/(0,8 × 1,25) = 360 N/mm² ; η_c = σ_c/lim_c = 218,5/360 = 60,7 % ; η = max(η_c, η⊥) = 60,7 %",
    any(l.startswith("lim_c = f_u/(β_w·γ_M2) = 360/(0,8 × 1,25) = 360 N/mm²") for l in L)
    and "η_c = σ_c/lim_c = 218,5/360 = 60,7 %" in L and any(l.startswith("η = max(η_c, η⊥)") and l.endswith("= 60,7 %") for l in L), str(L[-6:]))
L = FO.lignes(verif("C3", methode={"ec": "simplifiee"}))
chk("simplifiée : f_vw,d = f_u/(√3·β_w·γ_M2) = 360/(√3 × 0,8 × 1,25) = 207,8 N/mm² ; F_w,Rd = 207,8 × 5 = 1039 N/mm ; η = 764,4/1039 = 73,6 %",
    "f_vw,d = f_u/(√3·β_w·γ_M2) = 360/(√3 × 0,8 × 1,25) = 207,8 N/mm²" in L and "F_w,Rd = f_vw,d·a = 207,8 × 5 = 1039 N/mm" in L
    and "η = |f|/F_w,Rd = 764,4/1039 = 73,6 %" in L, str(L[-3:]))
L = FO.lignes(verif("C3", methode={"mode": "els"}))
chk("ELS : τ_adm = k·g/10 = 100 × 9,81/10 = 98,1 N/mm² ; F_adm = 491 N/mm ; η = 155,8 % ; k équivalent Eurocode = 141 kg/(mm·cm)",
    "τ_adm = k·g/10 = 100 × 9,81/10 = 98,1 N/mm²" in L and "F_adm = τ_adm·a = 98,1 × 5 = 491 N/mm" in L
    and "η = |f|/F_adm = 764,4/490,5 = 155,8 %" in L and any(l.startswith("k équivalent Eurocode") and l.endswith("= 141 kg/(mm·cm)") for l in L), str(L[-4:]))
L = FO.lignes(verif("C2", {"jonction": "pleine", "tB": 30, "tA": 30}, {"N": 400}))
chk("pleine pénétration : A_w = a·l_eff + 2·a·l_eff = 20 × 200 + 2 × 5 × 200 = 6000 mm² (les trois lignes de gorge = t·l), lim = f_y/γ_M0 = 235/1 = 235 N/mm², η = 100/235 = 42,6 %",
    "A_w = a·l_eff + 2·a·l_eff = 20 × 200 + 2 × 5 × 200 = 6000 mm²" in L and "lim = f_y/γ_M0 = 235/1 = 235 N/mm²" in L
    and "η = σ_c/lim = 100/235 = 42,6 %" in L, str(L))
L = FO.lignes(verif("C2", {"jonction": "pleine", "tB": 30, "tA": 30}, {"N": 400}, {"mode": "els"}))
chk("pleine pénétration en ELS : lim = f_y/γ_F = 235/1,5 = 156,7 N/mm²", "lim = f_y/γ_F = 235/1,5 = 156,7 N/mm²" in L, str(L[-2:]))
chk("refus : aucune substitution", FO.toutes(verif("C3", {"a": 2.0})) == {"groupe": [], "point": [], "critere": []})

print("\n=== 3. Les critères en typographie mathématique et la méthode du point ===")
chk("les cinq critères se composent (mini-LaTeX de ndc_pdf/mathx)", all(M.parse(e) is not None for e in FO.CRITERES_MATH.values()))
chk("méthode du point : directionnelle | simplifiee | els | pleine | pleine_els selon le choix et la nature du cordon",
    FO.methode_du_point(verif("C3"), verif("C3")["gouvernant"]) == "directionnelle"
    and FO.methode_du_point(verif("C3", methode={"ec": "simplifiee"}), verif("C3", methode={"ec": "simplifiee"})["gouvernant"]) == "simplifiee"
    and FO.methode_du_point(verif("C3", methode={"mode": "els"}), verif("C3", methode={"mode": "els"})["gouvernant"]) == "els"
    and FO.methode_du_point(verif("C7"), verif("C7")["gouvernant"]) == "pleine"
    and FO.methode_du_point(verif("C7", methode={"mode": "els"}), verif("C7", methode={"mode": "els"})["gouvernant"]) == "pleine_els")
chk("C5 semelles pleines + âme d'angle : le point gouvernant (semelle) est jugé au métal de base, le critère dit « pièce »",
    FO.methode_du_point(verif("C5", {"jonction_semelles": "pleine"}), verif("C5", {"jonction_semelles": "pleine"})["gouvernant"]) == "pleine"
    and "pièce" in verif("C5", {"jonction_semelles": "pleine"})["gouvernant"]["critere"])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)
