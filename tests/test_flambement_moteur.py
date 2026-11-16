# -*- coding: utf-8 -*-
"""Moteur « Flambement » (acier/flambement) : flambement par flexion d'une barre comprimée, EN 1993-1-1 §6.3.1.

NOUVEAU CALCUL : l'écran Streamlit modules/flambement.py n'a AUCUN calcul (« Page en construction ») ; il n'y a donc pas
d'écran dont reproduire les chiffres. La seule expression de flambement que l'application Streamlit porte déjà est
ec3_buckling_chi (modules/choix_profile.py, option « Vérifier flambement ») : le moteur est prouvé IDENTIQUE à elle, au
bit près. Le reste est contrôlé contre des valeurs publiées et des calculs à la main. Chaque contrôle rougit si l'on
retire ce qu'il protège (§6, mutations).

  1. La norme : f_y selon l'épaisseur, option a de l'ANB belge (NBN EN 1993-1-1 ANB:2018 3.2.1(1) : tableaux 3.3 ANB,
     3.4 ANB, 3.8 ANB à 3.11 ANB — paliers 16, 40, 63, 80 mm, selon le produit), ε, λ₁ = 93,9·ε (éq. 6.50), Tableau 6.1
     (α), Tableau 6.2 (chaque ligne : laminés en I selon h/b et t_f, S460, creux finis à chaud et formés à froid, U,
     cornières).
  2. Valeurs PUBLIÉES : le tableau des coefficients de réduction χ(λ̄) des courbes a, b, c, d (ENV 1993-1-1:1992,
     tableau 5.5.2, repris par les manuels) et a0, à 10⁻⁴.
  3. À la main : HEA 200 S235 bi-articulé de 4 m (N_cr, λ̄, φ, χ, N_b,Rd recalculés ici, sans le moteur), le même par
     λ̄ = L_cr / (i·λ₁) ; IPE 400 S355 de classe 4 (âme réduite, EN 1993-1-5 §4.4) ; cornière (axes principaux u, v) ;
     tube rectangulaire formé à froid (paroi b − 3t, EN 1993-1-5 §4.4(2)) ; longueurs de flambement (cas types,
     maintiens, k et L_cr saisis) ; γ_M1 ; f_y des profilés épais ; ε de CHAQUE paroi avec la f_y de son épaisseur
     (HEB 300 S235, IPE 600 S460 de classe 4 : brief § 7.1, EN 1993-1-1 Tableau 5.2), pour la classe comme pour λ̄_p
     (IPE 600 et IPE 550 S235 : âme de classe 4 avec son ε, de classe 3 avec celui de la section).
  4. Parité : ec3_buckling_chi de l'écran Streamlit (faux « st ») sur plus de 600 cas (profilés du catalogue, longueurs,
     nuances, courbes) — χ, λ̄, N_cr et N_b,Rd ÉGAUX (==) à ceux du moteur.
  5. Domaine : saisies refusées avec un message (traction, longueurs nulles ou absurdes, k et N démesurés, γ < 1,
     maintiens) — le calcul aboutit aux bornes du domaine, sans débordement ; hors du domaine du module
     (t > 80 mm), tube rond de classe 4 ; couverture (chaque courbe, chaque classe atteintes par le catalogue).
  6. Mutations : chaque altération plausible du moteur (α, φ, χ ≤ 1, seuil h/b, f_y et ses paliers, ε par paroi — pour
     λ̄_p et, seule, pour la classe —, produit, classe 4, k, maintiens, γ_M1…)
     fait rougir au moins un contrôle des §1–4.
  7. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, pandas,
     matplotlib, scipy, reportlab, PIL, fitz.

    /usr/bin/python3.12 tests/test_flambement_moteur.py
"""
import copy
import importlib.util
import inspect
import math
import os
import random
import re
import subprocess
import sys
import textwrap
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.flambement import entree as EN  # noqa: E402
from acier.flambement import moteur as MO  # noqa: E402
from acier.flambement import normes as NO  # noqa: E402
from acier.flambement import section as SE  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402

OK, KO = [], []


def _refuse(f):
    """La fonction lève HorsDomaine (une donnée que la table ou le module ne couvre pas)."""
    try:
        f()
    except NO.HorsDomaine:
        return True
    return False


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def proche(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def barre(**k):
    b = EN.nouvelle_barre()
    axes = k.pop("axes", None)
    b.update(k)
    if axes:
        for a, v in axes.items():
            b["axes"][a].update(v)
    return b


# ------------------------------------------------------------------ 2. le tableau publié des χ
# ENV 1993-1-1:1992, tableau 5.5.2 « Coefficients de réduction χ » (courbes a, b, c, d) ; la courbe a0 (EN 1993-1-1
# Tableau 6.1, α = 0,13) dans les tableaux des manuels qui l'ajoutent. λ̄ → χ, quatre décimales.
TABLE_CHI = {
    "a0": {0.2: 1.0, 0.3: 0.9859, 0.4: 0.9701, 0.5: 0.9513, 0.6: 0.9276, 0.7: 0.8961, 0.8: 0.8533, 0.9: 0.7961, 1.0: 0.7253},
    "a": {0.2: 1.0, 0.3: 0.9775, 0.4: 0.9528, 0.5: 0.9243, 0.6: 0.8900, 0.7: 0.8477, 0.8: 0.7957, 0.9: 0.7339, 1.0: 0.6656, 1.5: 0.3724,
          2.0: 0.2229},
    "b": {0.2: 1.0, 0.3: 0.9641, 0.4: 0.9261, 0.5: 0.8842, 0.6: 0.8371, 0.7: 0.7837, 0.8: 0.7245, 0.9: 0.6612, 1.0: 0.5970, 1.2: 0.4781,
          1.5: 0.3422, 2.0: 0.2095, 3.0: 0.0994},
    "c": {0.2: 1.0, 0.3: 0.9491, 0.4: 0.8973, 0.5: 0.8430, 0.6: 0.7854, 0.7: 0.7247, 0.8: 0.6622, 0.9: 0.5998, 1.0: 0.5399, 1.5: 0.3145,
          2.0: 0.1962},
    "d": {0.2: 1.0, 0.3: 0.9235, 0.4: 0.8504, 0.5: 0.7793, 0.6: 0.7100, 0.7: 0.6431, 0.8: 0.5797, 0.9: 0.5208, 1.0: 0.4671, 1.5: 0.2766,
          2.0: 0.1766},
}


# ------------------------------------------------------------------ les contrôles de référence (rejoués sous mutation)
def controles():
    """Les contrôles des §1 à 3 : [(nom, condition, info)] — la même liste sert au rapport et aux mutations."""
    out = []

    def c(nom, cond, info=""):
        out.append((nom, bool(cond), info))

    # 1. la norme : f_y selon l'épaisseur, option a de l'ANB (NBN EN 1993-1-1 ANB:2018 3.2.1(1)) ; les valeurs sont écrites
    # ICI, lues dans les tableaux 3.3 ANB (EN 10025-2:2004, S235JR, S275JR, S355JR) et 3.4 ANB (EN 10025-3:2004, S460N)
    # — brief § 7.1 —, sans relire la table du site
    nus = ("S235", "S275", "S355", "S460")
    for t_lus, attendu, bande in (((3.0, 16.0), [235, 275, 355, 460], "t ≤ 16 mm"), ((16.5, 40.0), [225, 265, 345, 440], "16 < t ≤ 40 mm"),
                                  ((40.5, 63.0), [215, 255, 335, 430], "40 < t ≤ 63 mm"), ((63.5, 80.0), [215, 245, 325, 410], "63 < t ≤ 80 mm")):
        lus = [[NO.limite_elastique(n, t) for n in nus] for t in t_lus]
        c(f"tableaux 3.3 ANB et 3.4 ANB, {bande} : {', '.join(map(str, attendu))} (lu à {' et '.join(f'{t:g}'.replace('.', ',') for t in t_lus)} mm)",
          all(x == attendu for x in lus), lus)
    c("creux : tableaux 3.8 ANB (S355J0H, fini à chaud) et 3.10 ANB (formé à froid) à 20 mm = 345 ; S460NH fini à chaud (3.9 ANB) "
      "s'arrête à 65 mm, formé à froid (3.10 ANB, 3.11 ANB) à 40 mm : refusés au-delà",
      NO.limite_elastique("S355", 20.0, produit="creux_chaud") == 345 and NO.limite_elastique("S355", 20.0, produit="creux_froid") == 345
      and NO.limite_elastique("S460", 65.0, produit="creux_chaud") == 430 and _refuse(lambda: NO.limite_elastique("S460", 66.0, produit="creux_chaud"))
      and _refuse(lambda: NO.limite_elastique("S355", 40.5, froid=True)) and _refuse(lambda: NO.limite_elastique("S355", 80.5)))
    c("le produit suit la famille : CHS, SHS, RHS finis à chaud ; CHS-F, SHS-F, RHS-F formés à froid ; laminés sinon (I, H, U, cornières)",
      [SE.produit(CAT.par_nom()[n]) for n in ("SHS 400x400x20", "RHS 200x100x5 (froid)", "HEA 200", "L 100x100x10", "UPN 200")]
      == ["creux_chaud", "creux_froid", "lamine", "lamine", "lamine"])
    # le refus au-delà de 80 mm : le nombre écrit à la française (troisième passe du lot A : « 89.9 mm » s'imprimait avec un
    # point, à l'écran et dans la note) — les 13 HD et HL de paroi > 80 mm du catalogue
    epais80 = [p for p in CAT.profils() if p["famille"] in SE.FABRICATION and SE.epaisseur(p) > NO.T_MAX_TABLEAU]
    motifs = {p["nom"]: MO.verifier(barre(profil=p["nom"]))["motif"] for p in epais80}
    c("refus au-delà de 80 mm, à la française : « épaisseur 89,9 mm > 80 mm : hors du domaine du module » (HL 920x970), "
      "« 115,1 mm » (HL 920x1269), « 97 mm » (HD 400x818) — aucun point décimal sur les 13 HD et HL concernés",
      len(motifs) == 13 and motifs.get("HL 920x970") == "épaisseur 89,9 mm > 80 mm : hors du domaine du module"
      and motifs.get("HL 920x1269", "").startswith("épaisseur 115,1 mm > 80 mm") and motifs.get("HD 400x818.0", "").startswith("épaisseur 97 mm >")
      and not any(re.search(r"\d\.\d", m_ or "") for m_ in motifs.values()), {k: motifs[k] for k in list(motifs)[:2]})
    r = MO.verifier(barre(profil="SHS 400x400x20", nuance="S355"))
    c("SHS 400×400×20 S355 (fini à chaud, t = 20 mm) : f_y = 345, lue au tableau 3.8 ANB (EN 10210-1:2006, S355J0H), 16 < t ≤ 40 mm",
      r["section"]["fy"] == 345 and r["section"]["source_fy"] == "tableau 3.8 ANB (EN 10210-1:2006, S355J0H), 16 < t ≤ 40 mm", r["section"].get("source_fy"))
    c("ε = √(235/f_y) : 1 en S235, 0,8136 en S355", proche(NO.epsilon(235), 1.0) and abs(NO.epsilon(355) - 0.81362) < 1e-5)
    c("λ₁ = π·√(E/f_y) = 93,9·ε (éq. 6.50)", abs(NO.lambda_1(235) - 93.9129) < 1e-3 and abs(NO.lambda_1(355) / NO.epsilon(355) - 93.9129) < 1e-3)
    c("Tableau 6.1 : α = 0,13 ; 0,21 ; 0,34 ; 0,49 ; 0,76", [NO.ALPHA[k] for k in ("a0", "a", "b", "c", "d")] == [0.13, 0.21, 0.34, 0.49, 0.76])
    t62 = [NO.courbes("lamine_I", "S235", 300, 150, 10.7)[:2], NO.courbes("lamine_I", "S235", 1000, 300, 45)[:2],
           NO.courbes("lamine_I", "S355", 190, 200, 10)[:2], NO.courbes("lamine_I", "S235", 500, 450, 105)[:2],
           NO.courbes("lamine_I", "S460", 300, 150, 10.7)[:2], NO.courbes("lamine_I", "S460", 1000, 300, 45)[:2],
           NO.courbes("lamine_I", "S460", 190, 200, 10)[:2], NO.courbes("lamine_I", "S460", 500, 450, 105)[:2],
           NO.courbes("creux_chaud", "S355")[:2], NO.courbes("creux_chaud", "S460")[:2], NO.courbes("creux_froid", "S460")[:2],
           NO.courbes("U", "S235")[:2], NO.courbes("L", "S460")[:2]]
    attendu = [("a", "b"), ("b", "c"), ("b", "c"), ("d", "d"), ("a0", "a0"), ("a", "a"), ("a", "a"), ("c", "c"), ("a", "a"), ("a0", "a0"),
               ("c", "c"), ("c", "c"), ("b", "b")]
    c("Tableau 6.2 : chaque ligne (laminés en I selon h/b et t_f, S460, creux chaud/froid, U, cornière)", t62 == attendu, t62)
    c("Tableau 6.2 : h/b = 1,2 exactement relève de « h/b ≤ 1,2 »", NO.courbes("lamine_I", "S235", 240, 200, 10)[:2] == ("b", "c"))
    c("Tableau 5.2 : limites 33/38/42·ε (interne), 9/10/14·ε (console), 50/70/90·ε² (tube), bornes incluses",
      [NO.classe_paroi("interne", x, 1.0) for x in (33, 33.01, 38, 42, 42.01)] == [1, 2, 2, 3, 4]
      and [NO.classe_paroi("console", x, 1.0) for x in (9, 10, 14, 14.01)] == [1, 2, 3, 4]
      and [NO.classe_paroi("tube", x, 0.9) for x in (50 * 0.81, 70 * 0.81, 90 * 0.81, 90 * 0.81 + 0.01)] == [1, 2, 3, 4])
    c("Tableau 5.2 (cornières) : h/t ≤ 15·ε ET (b + h)/2t ≤ 11,5·ε", NO.classe_corniere(11.5, 11.5, 1.0) == 3 and NO.classe_corniere(12, 11.5, 1.0) == 3
      and NO.classe_corniere(15.01, 11, 1.0) == 4 and NO.classe_corniere(13, 11.6, 1.0) == 4)
    c("EN 1993-1-5 §4.4 : ρ interne (λ̄_p − 0,22)/λ̄_p² au-delà de 0,673 ; console (λ̄_p − 0,188)/λ̄_p² au-delà de 0,748",
      NO.rho("interne", 0.673) == 1.0 and proche(NO.rho("interne", 1.0), 0.78) and NO.rho("console", 0.748) == 1.0
      and proche(NO.rho("console", 1.0), 0.812) and proche(NO.elancement_paroi("interne", 56.8, 1.0), 1.0))
    # 2. les valeurs publiées de χ
    ecarts = [(k, lb, round(NO.phi_chi(lb, NO.ALPHA[k])[1], 4), v) for k, t in TABLE_CHI.items() for lb, v in t.items()
              if abs(NO.phi_chi(lb, NO.ALPHA[k])[1] - v) > 0.5e-4 + 1e-12]
    c(f"χ(λ̄) = le tableau publié des coefficients de réduction ({sum(len(t) for t in TABLE_CHI.values())} valeurs, à 10⁻⁴)", not ecarts, ecarts[:4])
    c("χ ≤ 1 : χ = 1 pour λ̄ ≤ 0,2 (§6.3.1.2(4))", all(NO.phi_chi(lb, 0.76)[1] == 1.0 for lb in (0.0, 0.05, 0.1, 0.19)))
    # 3. à la main : HEA 200, S235, L = 4 m bi-articulé autour des deux axes, N_Ed = 500 kN
    r = MO.verifier(barre(profil="HEA 200", nuance="S235", N=500.0, L=4.0))
    p = CAT.par_nom()["HEA 200"]
    A = p["A"] * 100.0                                         # 5 383,45 mm²
    for a, I, crb in (("y", p["Iy"], 0.34), ("z", p["Iz"], 0.49)):
        Ncr = math.pi ** 2 * 210000.0 * I * 1e4 / 4000.0 ** 2
        lb = math.sqrt(A * 235.0 / Ncr)
        ph = 0.5 * (1 + crb * (lb - 0.2) + lb ** 2)
        chi = min(1.0, 1 / (ph + math.sqrt(ph ** 2 - lb ** 2)))
        Nb = chi * A * 235.0 / 1.0 / 1000.0
        ax = r["axes"][a]
        c(f"HEA 200 S235, 4 m, axe {a} : N_cr, λ̄, φ, χ, N_b,Rd recalculés à la main", proche(ax["Ncr"], Ncr / 1000) and proche(ax["lambdabar"], lb)
          and proche(ax["phi"], ph) and proche(ax["chi"], chi) and proche(ax["NbRd"], Nb) and proche(ax["taux"], 500 / Nb), (ax["NbRd"], Nb))
        lb2 = 4000.0 / (p["i" + a] * 10 * 93.9129 * 1.0)
        c(f"HEA 200 axe {a} : λ̄ = L_cr / (i·λ₁) donne le même élancement", abs(ax["lambdabar"] - lb2) < 1e-5)
    c("HEA 200 S235 : courbes b (y) et c (z) (h/b = 0,95 ≤ 1,2, t_f = 10 ≤ 100 mm), classe 1",
      (r["axes"]["y"]["courbe"], r["axes"]["z"]["courbe"], r["section"]["classe"]) == ("b", "c", 1))
    c("HEA 200 S235 : χ_z = 0,628, N_b,z,Rd = 794,0 kN, η = 63 % — z gouverne",
      round(r["axes"]["z"]["chi"], 3) == 0.628 and round(r["axes"]["z"]["NbRd"], 1) == 794.0 and r["gouvernant"] == "z")
    c("HEA 200 S235 : N_c,Rd = A·f_y/γ_M0 = 1 265,1 kN", round(r["section"]["NcRd"], 1) == 1265.1)
    # IPE 400 S355 : âme de classe 4 (c/t = 38,5 > 42·ε = 34,2), semelles de classe 1
    r = MO.verifier(barre(profil="IPE 400", nuance="S355", N=900.0, L=7.0))
    q = CAT.par_nom()["IPE 400"]
    eps = math.sqrt(235 / 355)
    lp = q["cw_tw"] / (28.4 * eps * 2.0)
    rho = (lp - 0.22) / lp ** 2
    Aeff = q["A"] * 100 - (1 - rho) * q["cw_tw"] * q["tw"] * q["tw"]
    c("IPE 400 S355 : classe 4 (âme), A_eff = A − (1 − ρ)·c_w·t_w recalculée à la main (EN 1993-1-5 §4.4)",
      r["section"]["classe"] == 4 and proche(r["section"]["A_eff"], Aeff) and 0.88 < rho < 0.89, (r["section"]["A_eff"], Aeff))
    Ncr = math.pi ** 2 * 210000 * q["Iz"] * 1e4 / 7000 ** 2
    lb = math.sqrt(Aeff * 355 / Ncr)
    c("IPE 400 S355 classe 4 : λ̄_z = √(A_eff·f_y/N_cr) (§6.3.1.3 éq. 6.51)", proche(r["axes"]["z"]["lambdabar"], lb))
    c("IPE 400 (h/b = 2,7 > 1,2 ; t_f = 13,5 ≤ 40) en S355 : courbes a / b", (r["axes"]["y"]["courbe"], r["axes"]["z"]["courbe"]) == ("a", "b"))
    r460 = MO.verifier(barre(profil="IPE 400", nuance="S460", N=900.0, L=7.0))
    c("… et en S460 : courbes a0 / a0", (r460["axes"]["y"]["courbe"], r460["axes"]["z"]["courbe"]) == ("a0", "a0"))
    # cornière : axes principaux u (fort) et v (faible)
    r = MO.verifier(barre(profil="L 100x100x10", N=50.0, L=2.0))
    q = CAT.par_nom()["L 100x100x10"]
    Ncr_v = math.pi ** 2 * 210000 * q["Iv"] * 1e4 / 2000 ** 2
    c("Cornière L 100×100×10 : flambement autour de u-u et v-v (I_u, I_v du catalogue), courbe b, v gouverne",
      (r["axes"]["y"]["nom"], r["axes"]["z"]["nom"]) == ("u-u", "v-v") and proche(r["axes"]["z"]["Ncr"], Ncr_v / 1000)
      and r["axes"]["z"]["courbe"] == "b" and r["gouvernant"] == "z")
    # cornière de classe 4 : la largeur de référence est h pour les DEUX ailes (EN 1993-1-5 §4.4(2))
    r = MO.verifier(barre(profil="L 100x65x7", nuance="S355", N=50.0, L=2.0))
    q = CAT.par_nom()["L 100x65x7"]
    eps = math.sqrt(235 / 355)
    lp = (q["h"] / q["t"]) / (28.4 * eps * math.sqrt(0.43))
    rho = min(1.0, (lp - 0.188) / lp ** 2) if lp > 0.748 else 1.0
    Aeff = q["A"] * 100 - (1 - rho) * (q["h"] + q["b"]) * q["t"]
    c("Cornière L 100×65×7 S355 (classe 4) : ρ des deux ailes avec b̄ = h, A_eff recalculée (c/t du catalogue à 10⁻⁶ près)",
      r["section"]["classe"] == 4 and proche(r["section"]["A_eff"], Aeff, 1e-5) and rho < 1, (r["section"]["A_eff"], Aeff, rho))
    # tube rectangulaire : la paroi vaut b − 3t (EN 1993-1-5 §4.4(2)), formé à froid COMME fini à chaud — le catalogue
    # donne pour les formés à froid la largeur plate nominale b − 2·r_o = b − 5t ici (r_o = 2,5t), plus favorable
    q = CAT.par_nom()["SHS 350x350x10 (froid)"]
    eps = math.sqrt(235 / 460)
    ct = (350.0 - 3 * 10.0) / 10.0                            # 32,0 > 42·ε = 30,02 : classe 4
    lp = ct / (28.4 * eps * 2.0)
    rho = (lp - 0.22) / lp ** 2
    Aeff = q["A"] * 100 - 4 * (1 - rho) * 320.0 * 10.0
    r = MO.verifier(barre(profil="SHS 350x350x10 (froid)", nuance="S460", N=1000.0, L=6.0))
    c("SHS 350×350×10 formé à froid S460 : paroi b − 3t = 320 mm, c/t = 32 > 42·ε = 30,0 → classe 4, A_eff recalculée "
      "(le catalogue, b − 5t, la dirait de classe 3)", r["section"]["classe"] == 4 and proche(r["section"]["A_eff"], Aeff)
      and q["cf_tf"] <= 42 * eps < ct, (r["section"]["classe"], r["section"]["A_eff"], Aeff, q["cf_tf"]))
    q = CAT.par_nom()["RHS 200x100x5"]
    pa = SE.parois(q)
    c("RHS fini à chaud (r_o = 1,5t) : b − 3t et h − 3t sont la largeur du catalogue", proche(pa[0]["c_t"], q["cf_tf"]) and proche(pa[1]["c_t"], q["cw_tw"])
      and proche(pa[0]["c"], 85.0) and proche(pa[1]["c"], 185.0))
    # longueurs de flambement
    L = 6.0
    cas = {"AA": 6.0, "EA": 4.2, "EE": 3.0, "EG": 6.0, "EL": 12.0}
    ok = all(proche(EN.longueur_flambement(dict(EN.AXE_DEFAUT, appuis=k), L), v) for k, v in cas.items())
    c("L_cr des cas types : k = 1 ; 0,7 ; 0,5 ; 1 ; 2 (valeurs théoriques d'Euler)", ok)
    c("L_cr entre maintiens : L / (n + 1)", proche(EN.longueur_flambement(dict(EN.AXE_DEFAUT, appuis="AA", maintiens=2), L), 2.0))
    c("L_cr saisie, coefficient saisi", proche(EN.longueur_flambement(dict(EN.AXE_DEFAUT, appuis="L", Lcr=3.3), L), 3.3)
      and proche(EN.longueur_flambement(dict(EN.AXE_DEFAUT, appuis="k", k=0.85), L), 5.1))
    # ε paroi par paroi (brief § 7.1 ; EN 1993-1-1 Tableau 5.2 : ε = √(235/f_y) DE LA PAROI) : HEB 300 S235, semelle de
    # 19 mm (225, tableau 3.3 ANB, 16 < t ≤ 40), âme de 11 mm (235, t ≤ 16) ; la section se calcule avec la plus petite
    r = MO.verifier(barre(profil="HEB 300", nuance="S235", N=500.0, L=4.0))
    q = CAT.par_nom()["HEB 300"]
    sem, ame = r["section"]["parois"]
    A = q["A"] * 100.0
    Ncr = math.pi ** 2 * 210000.0 * q["Iz"] * 1e4 / 4000.0 ** 2
    lb = math.sqrt(A * 225.0 / Ncr)
    ph = 0.5 * (1 + 0.49 * (lb - 0.2) + lb ** 2)
    Nb = min(1.0, 1 / (ph + math.sqrt(ph ** 2 - lb ** 2))) * A * 225.0 / 1000.0
    c("HEB 300 S235 : f_y de calcul 225 (t_f = 19 mm) ; semelle f_y = 225, ε = √(235/225) ; âme f_y = 235, ε = 1 ; "
      "N_c,Rd = A × 225, N_b,z,Rd recalculé à la main avec 225",
      r["section"]["fy"] == 225 and (sem["nom"], sem["fy"], ame["nom"], ame["fy"]) == ("semelle", 225, "âme", 235)
      and proche(sem["epsilon"], math.sqrt(235 / 225)) and ame["epsilon"] == 1.0 and proche(r["section"]["NcRd"], A * 225 / 1000)
      and proche(r["axes"]["z"]["NbRd"], Nb) and r["section"]["fy_par_paroi"], (r["section"]["fy"], sem.get("fy"), ame.get("fy"), sem.get("epsilon"), ame.get("epsilon")))
    # IPE 600 S460 : âme de 12 mm (f_y = 460, ε = 0,7148) de classe 4, semelles de 19 mm (440) ; A_eff avec l'ε DE L'ÂME,
    # N_c,Rd avec la f_y de calcul 440 (tableau 3.4 ANB, S460N, 16 < t ≤ 40 mm)
    r = MO.verifier(barre(profil="IPE 600", nuance="S460", N=900.0, L=7.0))
    q = CAT.par_nom()["IPE 600"]
    eps_w = math.sqrt(235 / 460)
    lp = q["cw_tw"] / (28.4 * eps_w * 2.0)
    rho = (lp - 0.22) / lp ** 2
    Aeff = q["A"] * 100 - (1 - rho) * q["cw_tw"] * q["tw"] * q["tw"]
    c("IPE 600 S460 classe 4 : ρ de l'âme avec ε_w = √(235/460) (t_w = 12 mm), A_eff recalculée à la main ; N_c,Rd = A_eff × 440",
      r["section"]["classe"] == 4 and proche(r["section"]["A_eff"], Aeff) and r["section"]["fy"] == 440
      and proche(r["section"]["NcRd"], Aeff * 440 / 1000), (r["section"]["A_eff"], Aeff, r["section"]["fy"]))
    # la CLASSIFICATION avec l'ε de chaque paroi (brief § 7.1 ; EN 1993-1-1 Tableau 5.2) : IPE 600 et IPE 550 S235, âme de
    # 12 et 11,1 mm (f_y = 235, ε_w = 1, tableau 3.3 ANB, t ≤ 16 mm), semelles de 19 et 17,2 mm (225) ; c/t de l'âme 42,83
    # et 42,13 > 42·ε_w = 42 → classe 4. Avec l'ε de la SECTION (√(235/225) = 1,022 : 42,92) l'âme serait de classe 3 et
    # A_eff = A (IPE 600 : N_b,z,Rd 2 333,3 kN, +1,4 %, du côté non sûr). Recalcul à la main : λ̄_p de l'âme avec ε_w,
    # ρ = (λ̄_p − 0,22)/λ̄_p² (EN 1993-1-5 §4.4), N_cr,z, λ̄, φ, χ (courbe b : h/b > 1,2, t_f ≤ 40), f_y de calcul 225.
    for nom, aeff_txt, nb_txt in (("IPE 600", 15224.0, 2301.0), ("IPE 550", 13174.2, 1909.9)):
        r = MO.verifier(barre(profil=nom, nuance="S235", N=500.0, L=4.0))
        q = CAT.par_nom()[nom]
        ct_w = q["cw_tw"]
        lp = ct_w / (28.4 * 1.0 * 2.0)
        rho = (lp - 0.22) / lp ** 2
        Aeff = q["A"] * 100 - (1 - rho) * ct_w * q["tw"] * q["tw"]
        Ncr = math.pi ** 2 * 210000.0 * q["Iz"] * 1e4 / 4000.0 ** 2
        lb = math.sqrt(Aeff * 225.0 / Ncr)
        ph = 0.5 * (1 + 0.34 * (lb - 0.2) + lb ** 2)
        Nb = min(1.0, 1 / (ph + math.sqrt(ph ** 2 - lb ** 2))) * Aeff * 225.0 / 1000.0
        ame = r["section"]["parois"][1]
        fr1 = lambda x: f"{x:,.1f}".replace(",", " ").replace(".", ",")      # noqa: E731 — « 15 224,0 »
        c(f"{nom} S235, N 500 kN, L 4 m : classe 4 (âme t_w = {q['tw']:g} mm, f_y 235, ε_w = 1 ; c/t = {ct_w:.2f} > 42·ε_w), ".replace(".", ",")
          + f"A_eff = {fr1(aeff_txt)} mm², N_b,z,Rd = {fr1(nb_txt)} kN recalculés à la main",
          42.0 < ct_w <= 42.0 * math.sqrt(235.0 / 225.0) and r["section"]["classe"] == 4 and (ame["nom"], ame["classe"], ame["epsilon"]) == ("âme", 4, 1.0)
          and proche(r["section"]["A_eff"], Aeff) and round(r["section"]["A_eff"], 1) == aeff_txt
          and proche(r["axes"]["z"]["NbRd"], Nb) and round(r["axes"]["z"]["NbRd"], 1) == nb_txt and r["axes"]["z"]["courbe"] == "b",
          (r["section"]["classe"], ame.get("classe"), r["section"]["A_eff"], Aeff, r["axes"]["z"]["NbRd"], Nb))
    # γ_M1 et f_y épais
    r1 = MO.verifier(barre(profil="HEB 300", N=100.0, L=5.0))
    r2 = MO.verifier(barre(profil="HEB 300", N=100.0, L=5.0, gamma_M1=1.1, gamma_M0=1.05))
    c("γ_M1 divise N_b,Rd ; γ_M0 divise N_c,Rd", proche(r2["axes"]["z"]["NbRd"], r1["axes"]["z"]["NbRd"] / 1.1)
      and proche(r2["section"]["NcRd"], r1["section"]["NcRd"] / 1.05))
    r = MO.verifier(barre(profil="HEM 1000", nuance="S355", N=100.0, L=5.0))
    r2 = MO.verifier(barre(profil="HL 920x588", nuance="S355"))
    c("HEM 1000 (t_f = 40 mm) : f_y = 345 (16 < t ≤ 40) ; HL 920x588 (t_f = 55,9 mm) : f_y = 335 (40 < t ≤ 63, tableau 3.3 ANB), "
      "courbes b / c", r["section"]["fy"] == 345 and r2["section"]["fy"] == 335 and (r2["axes"]["y"]["courbe"], r2["axes"]["z"]["courbe"]) == ("b", "c"),
      (r["section"]["fy"], r2["section"]["fy"]))
    return out


# ------------------------------------------------------------------ 4. parité avec l'écran Streamlit (ec3_buckling_chi)
def ecran_choix_profile():
    """modules/choix_profile.py chargé avec un faux « streamlit » (le module ne fait que décorer une fonction au
    chargement) : sa fonction ec3_buckling_chi, telle quelle."""
    st = types.ModuleType("streamlit")
    st.cache_data = lambda f=None, **k: f if f else (lambda g: g)
    ancien = sys.modules.get("streamlit")
    sys.modules["streamlit"] = st
    try:
        spec = importlib.util.spec_from_file_location("ecran_choix_profile", os.path.join(RACINE, "modules", "choix_profile.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        if ancien is None:
            del sys.modules["streamlit"]
        else:
            sys.modules["streamlit"] = ancien
    return m


def parite():
    m = ecran_choix_profile()
    alea = random.Random(20261107)
    noms = [p["nom"] for p in CAT.profils() if p["famille"] in SE.FABRICATION]
    n, ecarts, n_moteur = 0, [], 0
    for _ in range(320):
        nom = alea.choice(noms)
        nuance = alea.choice(list(NO.NUANCES))
        L = alea.choice([0.8, 1.5, 2.5, 3.0, 4.0, 5.25, 6.0, 8.0, 12.0])
        b = barre(profil=nom, nuance=nuance, N=alea.choice([10.0, 150.0, 800.0]), L=L,
                  axes={"y": {"appuis": alea.choice(["AA", "EA", "EE", "EG", "EL"])}, "z": {"appuis": "AA", "maintiens": alea.choice([0, 1, 2])}})
        r = MO.verifier(b)
        if r["statut"] == MO.IMPOSSIBLE:
            continue
        p = CAT.par_nom()[nom]
        fy = r["section"]["fy"]
        for a in ("y", "z"):
            ax = r["axes"][a]
            I = ax["I"]
            # A de l'écran : l'aire de la section (cm²) ; en classe 4, le moteur emploie A_eff — on passe A_eff à l'écran
            A_cm2 = r["section"]["A_eff"] / 100.0 if r["section"]["classe"] == 4 else p["A"]
            e = m.ec3_buckling_chi(A_cm2, I, ax["Lcr"], fy=fy, alpha=ax["alpha"])
            Nb = e["chi"] * (A_cm2 * 1e2) * fy / b["gamma_M1"]           # la ligne de l'écran, N
            n += 1
            if r["section"]["classe"] == 4:
                ok = proche(e["chi"], ax["chi"], 1e-12) and proche(e["lambdabar"], ax["lambdabar"], 1e-12)
            else:
                n_moteur += 1
                ok = (e["chi"] == ax["chi"] and e["lambdabar"] == ax["lambdabar"] and e["Ncr_N"] == ax["Ncr_N"]
                      and Nb == ax["NbRd_N"])
            if not ok:
                ecarts.append((nom, nuance, a, e, ax["chi"], ax["NbRd"]))
    return n, n_moteur, ecarts, m


def parite_menu(m):
    """Le menu de l'écran (« a0 (0.13) », … « d (0.76) ») : ses α sont ceux du Tableau 6.1 du moteur ; relu dans le
    SOURCE (le dictionnaire est local à show())."""
    src = open(os.path.join(RACINE, "modules", "choix_profile.py"), encoding="utf-8").read()
    import re
    menu = dict((k, float(v)) for k, v in re.findall(r'"(a0|a|b|c|d) \(([0-9.]+)\)":', src))
    return menu


# ------------------------------------------------------------------ 6. mutations
def mutations():
    """Chaque altération plausible du moteur doit faire rougir au moins un contrôle de référence."""
    sauf = {"phi_chi": NO.phi_chi, "ALPHA": dict(NO.ALPHA), "courbes": NO.courbes, "limite_elastique": NO.limite_elastique,
            "rho": NO.rho, "K_APPUIS": dict(EN.K_APPUIS), "longueur_flambement": EN.longueur_flambement, "euler": NO.euler,
            "classe_paroi": NO.classe_paroi, "LIMITES_CLASSE": dict(NO.LIMITES_CLASSE), "parois": SE.parois, "elancement_paroi": NO.elancement_paroi,
            "classe_corniere": NO.classe_corniere, "t_fy": MO._t_fy, "produit": SE.produit, "section": MO._section}
    _axe = MO._axe
    src_section = textwrap.dedent(inspect.getsource(MO._section))

    def section_mutee(avant, apres):
        """``MO._section`` réécrite d'UN remplacement dans sa source (« avant » doit y figurer une fois : sinon la mutation
        est périmée et le test s'arrête) — pour une altération qu'aucune fonction de normes.py ne porte seule."""
        if src_section.count(avant) != 1:
            raise AssertionError(f"mutation périmée : « {avant} » ({src_section.count(avant)} fois dans _section)")
        ns = dict(MO.__dict__)
        exec(compile(src_section.replace(avant, apres), MO.__file__, "exec"), ns)
        return ns["_section"]

    def restaure():
        NO.phi_chi = sauf["phi_chi"]
        NO.ALPHA.clear()
        NO.ALPHA.update(sauf["ALPHA"])
        NO.courbes = sauf["courbes"]
        NO.limite_elastique = sauf["limite_elastique"]
        NO.rho = sauf["rho"]
        NO.euler = sauf["euler"]
        NO.classe_paroi = sauf["classe_paroi"]
        NO.classe_corniere = sauf["classe_corniere"]
        NO.elancement_paroi = sauf["elancement_paroi"]
        NO.LIMITES_CLASSE.clear()
        NO.LIMITES_CLASSE.update(sauf["LIMITES_CLASSE"])
        EN.K_APPUIS.clear()
        EN.K_APPUIS.update(sauf["K_APPUIS"])
        EN.longueur_flambement = sauf["longueur_flambement"]
        SE.parois = sauf["parois"]
        SE.produit = sauf["produit"]
        MO._axe = _axe
        MO._t_fy = sauf["t_fy"]
        MO._section = sauf["section"]

    def phi_02(lb, alpha):
        phi = 0.5 * (1.0 + alpha * (lb - 0.25) + lb ** 2)
        return phi, min(1.0, 1.0 / (phi + math.sqrt(max(phi ** 2 - lb ** 2, 0.0))))

    def chi_sans_plafond(lb, alpha):
        phi = 0.5 * (1.0 + alpha * (lb - 0.2) + lb ** 2)
        return phi, 1.0 / (phi + math.sqrt(max(phi ** 2 - lb ** 2, 0.0)))

    def chi_signe(lb, alpha):
        phi = 0.5 * (1.0 + alpha * (lb - 0.2) + lb ** 2)
        return phi, min(1.0, 1.0 / (phi + math.sqrt(max(phi ** 2 + lb ** 2, 0.0))))

    def courbes_12(fab, nuance, h=None, b=None, tf=None):
        if fab == "lamine_I" and h / b > 1.0 and h / b <= 1.2:
            h = 1.3 * b
        return sauf["courbes"](fab, nuance, h, b, tf)

    def courbes_s460(fab, nuance, h=None, b=None, tf=None):
        return sauf["courbes"](fab, "S355" if nuance == "S460" else nuance, h, b, tf)

    def courbes_inverse(fab, nuance, h=None, b=None, tf=None):
        y, z, t = sauf["courbes"](fab, nuance, h, b, tf)
        return z, y, t

    def fy_sans_epaisseur(nuance, t, froid=False, produit=None):
        return NO.NUANCES[nuance]                       # la valeur nominale (le nombre du nom), quelle que soit t

    def fy_sans_40(nuance, t, froid=False, produit=None):
        return sauf["limite_elastique"](nuance, min(t, 40.0), froid, produit)

    def fy_point(nuance, t, froid=False, produit=None):
        if t > NO.T_MAX_TABLEAU:                         # l'ancien refus : le nombre écrit par Python (« 89.9 »)
            raise NO.HorsDomaine(f"épaisseur {t:g} mm > {NO.T_MAX_TABLEAU:g} mm : hors du domaine du module")
        return sauf["limite_elastique"](nuance, t, froid, produit)

    def fy_sans_16(nuance, t, froid=False, produit=None):
        return sauf["limite_elastique"](nuance, 16.0 if 16.0 < t <= 40.0 else t, froid, produit)

    def rho_un(t, lp):
        return 1.0

    def euler_sans_pi(A, I, L, fy, E=NO.E_ACIER):
        Ncr = 9.8696 * E * I / L ** 2 * 1.02
        return Ncr, math.sqrt(A * fy / Ncr)

    def lcr_maintiens(ax, L):
        if ax["appuis"] == "AA":
            return L / max(ax["maintiens"], 1)
        return sauf["longueur_flambement"](ax, L)

    def parois_corniere_b(p):
        out = sauf["parois"](p)
        if SE.FABRICATION[p["famille"]] == "L":
            out[1]["c_t"] = p["cw_tw"]          # b̄ = b pour la petite aile (au lieu de h)
        return out

    def parois_tube_catalogue(p):
        out = sauf["parois"](p)
        if SE.FABRICATION[p["famille"]] == "creux_froid" and not p["famille"].startswith("CHS"):
            out[0].update(c_t=p["cf_tf"], c=p["cf_tf"] * p["t"])           # la largeur plate b − 2·r_o du catalogue
            out[1].update(c_t=p["cw_tw"], c=p["cw_tw"] * p["t"])
        return out

    def axe_sans_gamma(*a, **k):
        a = list(a)
        a[-1] = 1.0
        return _axe(*a, **k)

    def classe_42(t, ct, eps):
        if t == "interne":
            return 3 if ct <= 45 * eps else 4
        return sauf["classe_paroi"](t, ct, eps)

    def corniere_15(h_t, bh_2t, eps):
        return 3 if h_t <= 15 * eps else 4

    cas = [
        ("α de la courbe b = 0,35", lambda: NO.ALPHA.__setitem__("b", 0.35)),
        ("α de la courbe a0 = 0,21", lambda: NO.ALPHA.__setitem__("a0", 0.21)),
        ("φ avec (λ̄ − 0,25)", lambda: setattr(NO, "phi_chi", phi_02)),
        ("χ sans le plafond 1", lambda: setattr(NO, "phi_chi", chi_sans_plafond)),
        ("χ avec √(φ² + λ̄²)", lambda: setattr(NO, "phi_chi", chi_signe)),
        ("seuil h/b à 1,0 au lieu de 1,2", lambda: setattr(NO, "courbes", courbes_12)),
        ("S460 traité comme S355 dans le Tableau 6.2", lambda: setattr(NO, "courbes", courbes_s460)),
        ("courbes des axes y et z permutées", lambda: setattr(NO, "courbes", courbes_inverse)),
        ("f_y nominale, sans épaisseur", lambda: setattr(NO, "limite_elastique", fy_sans_epaisseur)),
        ("f_y sans la réduction au-delà de 40 mm", lambda: setattr(NO, "limite_elastique", fy_sans_40)),
        ("palier à 16 mm oublié (16 < t ≤ 40 lu comme t ≤ 16)", lambda: setattr(NO, "limite_elastique", fy_sans_16)),
        ("troisième passe — refus de 80 mm écrit avec un point décimal (« 89.9 mm »)", lambda: setattr(NO, "limite_elastique", fy_point)),
        ("ε de la semelle donné à l'âme (une seule f_y, celle de la paroi la plus épaisse)", lambda: setattr(MO, "_t_fy", lambda p, pa: SE.epaisseur(p))),
        ("classe de chaque paroi avec l'ε de la SECTION (λ̄_p par paroi gardé)",
         lambda: setattr(MO, "_section", section_mutee('NO.classe_paroi(pa["type"], pa["c_t"], d["epsilon"])', 'NO.classe_paroi(pa["type"], pa["c_t"], eps)'))),
        ("produit oublié (les creux lus comme des laminés)", lambda: setattr(SE, "produit", lambda p: "lamine")),
        ("classe 4 sans réduction (ρ = 1)", lambda: setattr(NO, "rho", rho_un)),
        ("N_cr approché (π² arrondi, +2 %)", lambda: setattr(NO, "euler", euler_sans_pi)),
        ("k encastrée–articulée = 0,8", lambda: EN.K_APPUIS.__setitem__("EA", 0.8)),
        ("L_cr = L / n entre maintiens", lambda: setattr(EN, "longueur_flambement", lcr_maintiens)),
        ("cornière : b̄ = b pour la petite aile", lambda: setattr(SE, "parois", parois_corniere_b)),
        ("tube formé à froid : largeur plate du catalogue (b − 2·r_o) au lieu de b − 3t", lambda: setattr(SE, "parois", parois_tube_catalogue)),
        ("γ_M1 ignoré", lambda: setattr(MO, "_axe", axe_sans_gamma)),
        ("limite de classe 3 interne à 45·ε", lambda: setattr(NO, "classe_paroi", classe_42)),
        ("cornière : critère (b + h)/2t oublié", lambda: setattr(NO, "classe_corniere", corniere_15)),
    ]
    res = []
    for nom, appliquer in cas:
        try:
            appliquer()
            try:
                rouges = [n for n, ok, _ in controles() if not ok]
            except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
                rouges = [f"exception {type(e).__name__}"]
        finally:
            restaure()
        res.append((nom, rouges))
    return res


# ------------------------------------------------------------------ 7. pureté
SONDE = r"""
import sys
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("streamlit", "pandas", "matplotlib", "scipy", "reportlab", "PIL", "fitz"):
            raise ImportError("refusé : " + name)
sys.meta_path.insert(0, Refus())
sys.path.insert(0, %r)
from acier.flambement import moteur as MO, entree as EN
r = MO.verifier(EN.nouvelle_barre())
c = MO.candidats(EN.nouvelle_barre(), ["IPE", "HEA", "L", "CHS-F"])
lourds = sorted(m for m in sys.modules if m.split(".")[0] in ("streamlit", "pandas", "matplotlib", "scipy", "reportlab", "PIL", "fitz"))
print(repr(r["taux_max"]), len(c), lourds)
"""


def main():
    print("1–3. La norme, les valeurs publiées, les calculs à la main")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("4. Parité avec l'écran Streamlit (modules/choix_profile.py, ec3_buckling_chi)")
    n, n_moteur, ecarts, m = parite()
    chk(f"ec3_buckling_chi = le moteur : {n} axes calculés, dont {n_moteur} AU BIT PRÈS (χ, λ̄, N_cr, N_b,Rd ==) ; classe 4 à 10⁻¹²",
        n >= 600 and n_moteur >= 400 and not ecarts, ecarts[:2])
    menu = parite_menu(m)
    chk("le menu des courbes de l'écran (« a0 (0.13) » … « d (0.76) ») = le Tableau 6.1 du moteur", menu == NO.ALPHA, menu)
    chk("ec3_buckling_chi rend None pour L_cr ≤ 0 ; le moteur REFUSE L ≤ 0 avec un message (pas de résultat muet)",
        m.ec3_buckling_chi(10, 100, 0, 235) is None and EN.verifier_valeur("L", 0) and EN.verifier_valeur("Lcr", -1))

    print("5. Domaine, refus, couverture")
    for cle, x in (("N", -1), ("L", 0), ("k", 0), ("Lcr", 0), ("maintiens", 10), ("maintiens", -1), ("gamma_M1", 0.95), ("gamma_M0", 1.6),
                   ("N", 2e6), ("N", 1e300), ("L", 150), ("L", 1e200), ("L", 1e-300), ("L", 0.005), ("k", 1e200), ("k", 0.001), ("k", 11),
                   ("Lcr", 1e200), ("Lcr", 1e-300), ("Lcr", 1001)):
        chk(f"saisie refusée avec un message : {cle} = {x}", bool(EN.verifier_valeur(cle, x)))
    for cle, x in (("N", 0), ("N", 1e6), ("L", 0.1), ("L", 0.01), ("L", 100), ("k", 0.85), ("k", 10), ("k", 0.01), ("Lcr", 0.001), ("Lcr", 1000),
                   ("maintiens", 9), ("gamma_M1", 1.0), ("gamma_M1", 1.5)):
        chk(f"saisie admise : {cle} = {x}", EN.verifier_valeur(cle, x) is None)
    # aux bornes du domaine, le calcul aboutit (ni débordement, ni division par zéro)
    extremes = []
    for prof in ("CHS 26.9x3.2", "HD 400x1086.0", "HEA 200"):
        for L, k, N in ((100.0, 10.0, 1e6), (0.01, 0.01, 1e6), (100.0, 10.0, 0.0)):
            r = MO.verifier(barre(profil=prof, N=N, L=L, axes={"y": {"appuis": "k", "k": k}, "z": {"appuis": "k", "k": k}}))
            extremes.append(r["statut"] == MO.IMPOSSIBLE or (math.isfinite(r["taux_max"]) and 0 <= r["axes"]["z"]["chi"] <= 1))
    chk("aux bornes du domaine (L = 0,01 à 100 m, k = 0,01 à 10, N jusqu'à 10⁶ kN), le calcul aboutit sans débordement", all(extremes))
    for b in ({"profil": "HEA 999"}, {"nuance": "S690"}, {"N": -10}, {"L": 0}, {"axes": {"y": {"appuis": "XX"}}}, {"axes": {"z": {"maintiens": 1.5}}},
              {"gamma_M1": 0.9}, {"N": "cent"}, {"N": float("nan")}, {"N": 1e308}, {"L": 1e200}, {"axes": {"y": {"appuis": "L", "Lcr": 1e300}}}):
        try:
            EN.normaliser(dict(EN.nouvelle_barre(), **b))
            chk(f"barre abîmée refusée : {b}", False)
        except ValueError as e:
            chk(f"barre abîmée refusée avec un message : {b} → « {e} »", bool(str(e)))
    r = MO.verifier(barre(profil="HD 400x1086.0"))
    chk("HD 400×1086 (t_f > 80 mm) : hors du domaine du module (80 mm), calcul refusé avec le motif", r["statut"] == MO.IMPOSSIBLE and "80 mm" in r["motif"]
        and r["taux_max"] is None)
    tubes4 = [p["nom"] for p in CAT.profils() if p["famille"] == "CHS" and MO.verifier(barre(profil=p["nom"], nuance="S460"))["statut"] == MO.IMPOSSIBLE]
    chk(f"tubes ronds de classe 4 (d/t > 90·ε²) refusés : EN 1993-1-6 hors du champ ({len(tubes4)} en S460)", len(tubes4) > 0
        and "classe 4" in MO.verifier(barre(profil=tubes4[0], nuance="S460"))["motif"])
    vus_c, vus_cl, vus_fy = set(), set(), set()
    for p in CAT.profils():
        if p["famille"] not in SE.FABRICATION:
            continue
        for nu in ("S235", "S460"):
            r = MO.verifier(barre(profil=p["nom"], nuance=nu))
            if r["statut"] != MO.IMPOSSIBLE:
                vus_c |= {r["axes"]["y"]["courbe"], r["axes"]["z"]["courbe"]}
                vus_cl.add(r["section"]["classe"])
                vus_fy.add(r["section"]["fy"])
    chk("couverture : les courbes a0, a, b, c sont atteintes par le catalogue", vus_c == {"a0", "a", "b", "c"}, vus_c)
    epais = [p["nom"] for p in CAT.profils() if p["famille"] in SE.FABRICATION and (p.get("tf") or 0) > 100]
    chk(f"courbe d (laminés de t_f > 100 mm) : les {len(epais)} profilés du catalogue concernés sont hors du domaine du module (80 mm), refusés",
        epais and all(MO.verifier(barre(profil=n))["statut"] == MO.IMPOSSIBLE for n in epais))
    chk("couverture : les classes 1, 2, 3 et 4 sont atteintes", vus_cl >= {1, 2, 3, 4}, vus_cl)
    chk("couverture : chaque palier de f_y atteint (S235 : 235, 225, 215 ; S460 : 460, 440, 430, 410)", {235.0, 225.0, 215.0, 460.0, 440.0, 430.0, 410.0} <= vus_fy, vus_fy)
    c = MO.candidats(EN.nouvelle_barre(), ["IPE", "HEA", "HEB"])
    chk("candidats : triés par masse croissante", [x["masse"] for x in c] == sorted(x["masse"] for x in c))
    best = MO.plus_leger(c)
    chk("le plus léger qui tient : η ≤ 1 et aucun plus léger ne tient", best["taux"] <= 1
        and all(x["taux"] > 1 for x in c if x["masse"] < best["masse"]), best)
    chk("l'avertissement « nouveau calcul » est le premier message du moteur", MO.verifier(EN.nouvelle_barre())["avertissements"][0] == MO.NOUVEAU)
    r = MO.verifier(barre(profil="UPN 200"))
    chk("U : le flambement par flexion-torsion non vérifié est DIT", any("flexion-torsion" in a for a in r["avertissements"]))
    r = MO.verifier(barre(profil="L 100x100x10"))
    chk("cornière : torsion, flexion-torsion et excentricité des attaches non prises en compte, DIT", any("annexe BB" in a for a in r["avertissements"]))
    b0 = EN.nouvelle_barre()
    b1 = copy.deepcopy(b0)
    MO.verifier(b1)
    chk("verifier ne modifie pas la barre reçue", b1 == b0)

    print("6. Mutations")
    for nom, rouges in mutations():
        chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    chk("hors mutation, tous les contrôles de référence sont au vert", all(ok for _, ok, _ in controles()))

    print("7. Pureté")
    for py in ("/usr/bin/python3.12", "/usr/bin/python3.13"):
        if not os.path.exists(py):
            continue
        p = subprocess.run([py, "-c", SONDE % RACINE], capture_output=True, text=True, cwd=RACINE, timeout=300)
        sortie = p.stdout.strip()
        chk(f"{os.path.basename(py)} : le moteur calcule sans streamlit, pandas, matplotlib, scipy, reportlab (refusés)",
            p.returncode == 0 and sortie.endswith("[]"), (sortie, p.stderr[-400:]))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
