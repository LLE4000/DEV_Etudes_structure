# -*- coding: utf-8 -*-
"""Moteur « Flambement » (acier/flambement) : flambement par flexion d'une barre comprimée, EN 1993-1-1 §6.3.1.

NOUVEAU CALCUL : l'écran Streamlit modules/flambement.py n'a AUCUN calcul (« Page en construction ») ; il n'y a donc pas
d'écran dont reproduire les chiffres. La seule expression de flambement que l'application Streamlit porte déjà est
ec3_buckling_chi (modules/choix_profile.py, option « Vérifier flambement ») : le moteur est prouvé IDENTIQUE à elle, au
bit près. Le reste est contrôlé contre des valeurs publiées et des calculs à la main. Chaque contrôle rougit si l'on
retire ce qu'il protège (§6, mutations).

  1. La norme : Tableau 3.1 (f_y selon l'épaisseur), ε, λ₁ = 93,9·ε (éq. 6.50), Tableau 6.1 (α), Tableau 6.2 (chaque
     ligne : laminés en I selon h/b et t_f, S460, creux finis à chaud et formés à froid, U, cornières).
  2. Valeurs PUBLIÉES : le tableau des coefficients de réduction χ(λ̄) des courbes a, b, c, d (ENV 1993-1-1:1992,
     tableau 5.5.2, repris par les manuels) et a0, à 10⁻⁴.
  3. À la main : HEA 200 S235 bi-articulé de 4 m (N_cr, λ̄, φ, χ, N_b,Rd recalculés ici, sans le moteur), le même par
     λ̄ = L_cr / (i·λ₁) ; IPE 400 S355 de classe 4 (âme réduite, EN 1993-1-5 §4.4) ; cornière (axes principaux u, v) ;
     tube rectangulaire formé à froid (paroi b − 3t, EN 1993-1-5 §4.4(2)) ; longueurs de flambement (cas types,
     maintiens, k et L_cr saisis) ; γ_M1 ; f_y de 40 à 80 mm.
  4. Parité : ec3_buckling_chi de l'écran Streamlit (faux « st ») sur plus de 600 cas (profilés du catalogue, longueurs,
     nuances, courbes) — χ, λ̄, N_cr et N_b,Rd ÉGAUX (==) à ceux du moteur.
  5. Domaine : saisies refusées avec un message (traction, longueurs nulles ou absurdes, k et N démesurés, γ < 1,
     maintiens) — le calcul aboutit aux bornes du domaine, sans débordement ; hors du Tableau 3.1
     (t > 80 mm), tube rond de classe 4 ; couverture (chaque courbe, chaque classe atteintes par le catalogue).
  6. Mutations : chaque altération plausible du moteur (α, φ, χ ≤ 1, seuil h/b, f_y, classe 4, k, maintiens, γ_M1…)
     fait rougir au moins un contrôle des §1–4.
  7. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, pandas,
     matplotlib, scipy, reportlab, PIL, fitz.

    /usr/bin/python3.12 tests/test_flambement_moteur.py
"""
import copy
import importlib.util
import math
import os
import random
import subprocess
import sys
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

    # 1. la norme
    c("Tableau 3.1 : S235, S275, S355, S460 jusqu'à 40 mm", [NO.limite_elastique(n, 40.0) for n in ("S235", "S275", "S355", "S460")] == [235, 275, 355, 460])
    c("Tableau 3.1 : de 40 à 80 mm (215, 255, 335, 430)", [NO.limite_elastique(n, 40.5) for n in ("S235", "S275", "S355", "S460")] == [215, 255, 335, 430])
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
    # γ_M1 et f_y épais
    r1 = MO.verifier(barre(profil="HEB 300", N=100.0, L=5.0))
    r2 = MO.verifier(barre(profil="HEB 300", N=100.0, L=5.0, gamma_M1=1.1, gamma_M0=1.05))
    c("γ_M1 divise N_b,Rd ; γ_M0 divise N_c,Rd", proche(r2["axes"]["z"]["NbRd"], r1["axes"]["z"]["NbRd"] / 1.1)
      and proche(r2["section"]["NcRd"], r1["section"]["NcRd"] / 1.05))
    r = MO.verifier(barre(profil="HEM 1000", nuance="S355", N=100.0, L=5.0))
    r2 = MO.verifier(barre(profil="HL 920x588", nuance="S355"))
    c("HEM 1000 (t_f = 40 mm) : f_y = 355 ; HL 920x588 (t_f = 55,9 mm) : f_y = 335 (Tableau 3.1, 40 < t ≤ 80), courbes b / c",
      r["section"]["fy"] == 355 and r2["section"]["fy"] == 335 and (r2["axes"]["y"]["courbe"], r2["axes"]["z"]["courbe"]) == ("b", "c"))
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
            "classe_corniere": NO.classe_corniere}
    _axe = MO._axe

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
        MO._axe = _axe

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

    def fy_sans_epaisseur(nuance, t, froid=False):
        return NO.NUANCES[nuance][0]

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
        ("f_y sans la réduction au-delà de 40 mm", lambda: setattr(NO, "limite_elastique", fy_sans_epaisseur)),
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
    chk("HD 400×1086 (t_f > 80 mm) : hors du Tableau 3.1, calcul refusé avec le motif", r["statut"] == MO.IMPOSSIBLE and "80 mm" in r["motif"]
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
    chk(f"courbe d (laminés de t_f > 100 mm) : les {len(epais)} profilés du catalogue concernés sont hors du Tableau 3.1, refusés",
        epais and all(MO.verifier(barre(profil=n))["statut"] == MO.IMPOSSIBLE for n in epais))
    chk("couverture : les classes 1, 2, 3 et 4 sont atteintes", vus_cl >= {1, 2, 3, 4}, vus_cl)
    chk("couverture : f_y réduit (40 < t ≤ 80 mm) atteint", {215.0, 430.0} <= vus_fy, vus_fy)
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
