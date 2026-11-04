# -*- coding: utf-8 -*-
"""Moteur de sections ``acier/profiles/geometrie.py`` et registre
``acier/profiles/familles.py`` : verrous analytiques (formes fermées,
séries exactes, formules classiques dérivées dans le test) et contrôle
contre les valeurs que l'application portait déjà (``profiles_test.json``
avant la transcription, puis le catalogue livré).

Lancement : python3 tests/test_profiles_geometrie.py
"""
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import numpy as np  # noqa: E402

from acier.profiles import familles as F  # noqa: E402
from acier.profiles import geometrie as G  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


print("=== 1. Intégration du contour : formes fermées ===")
b, h = 100.0, 50.0
p = G.proprietes([G.contour_rectangle(b, h)])
chk("rectangle : A, I_y = b·h³/12, I_z = h·b³/12, W_el, W_pl = b·h²/4, i",
    pres(p["A"], b * h) and pres(p["Iy"], b * h ** 3 / 12) and pres(p["Iz"], h * b ** 3 / 12)
    and pres(p["Wel_y"], b * h * h / 6) and pres(p["Wpl_y"], b * h * h / 4, 1e-6) and pres(p["Wpl_z"], h * b * b / 4, 1e-6)
    and pres(p["iy"], h / math.sqrt(12)) and pres(p["Iyz"], 0.0, 1e-9))
R = 40.0
p = G.proprietes([G.cercle(R)])
chk("cercle : A = π·R² (rayon corrigé), I = π·R⁴/4 à 10⁻⁵, W_pl = 4·R³/3 à 10⁻⁵",
    pres(p["A"], math.pi * R * R, 1e-9) and pres(p["Iy"], math.pi * R ** 4 / 4, 1e-5) and pres(p["Wpl_y"], 4 * R ** 3 / 3, 1e-5))
d, t = 100.0, 5.0
p = G.proprietes(G.contours_creux_rond(d, t))
chk("tube rond : A et I = π·(D⁴ − d⁴)/64 (trou soustrait par l'orientation)",
    pres(p["A"], math.pi * ((d / 2) ** 2 - (d / 2 - t) ** 2), 1e-9) and pres(p["Iy"], math.pi * (d ** 4 - (d - 2 * t) ** 4) / 64, 1e-5))
rect = G.contour_rectangle(60.0, 20.0)
rect_d = G.contour_rectangle(60.0, 20.0)
rect_d[:, 1] += 100.0
p = G.proprietes([rect_d])
chk("translation : I au centre de gravité inchangé, zc = 100", pres(p["Iy"], 60 * 20 ** 3 / 12) and pres(p["zc"], 100.0))

print("\n=== 2. Congés et arrondis ===")
# un carré 100 × 100 aux quatre coins arrondis r = 10 : A = 100² − (4 − π)·r²
c = G.arrondir([(0, 0), (100, 0), (100, 100), (0, 100)], [10, 10, 10, 10])
pa = G.proprietes([c])
chk("quatre arrondis convexes : aire = 100² − (4 − π)·r² à 10⁻⁴",
    pres(pa["A"], 100.0 ** 2 - (4 - math.pi) * 100.0, 1e-4))
# un I : A = 2·b·tf + (h − 2tf)·tw + (4 − π)·r² (congés rentrants ajoutés) — formule de acier/bibliotheques.aire_profil
h, b, tw, tf, r = 96.0, 100.0, 5.0, 8.0, 12.0
pi_ = G.proprietes([G.contour_I(h, b, tw, tf, r)])
chk("I : aire = 2·b·tf + (h − 2tf)·tw + (4 − π)·r² à 10⁻⁴",
    pres(pi_["A"], 2 * b * tf + (h - 2 * tf) * tw + (4 - math.pi) * r * r, 1e-4))
chk("I : I_z = 2·tf·b³/12 + (h − 2tf)·tw³/12 + congés (> sans congés, < + 1 %)",
    2 * tf * b ** 3 / 12 + (h - 2 * tf) * tw ** 3 / 12 < pi_["Iz"] < 1.01 * (2 * tf * b ** 3 / 12 + (h - 2 * tf) * tw ** 3 / 12))
pl = G.proprietes([G.contour_L(100, 100, 10, 12, 6)])
chk("cornière égale : α = 45°, I_u + I_v = I_y + I_z, yc = zc",
    pres(pl["alpha_deg"], 45.0, 1e-6) and pres(pl["Iu"] + pl["Iv"], pl["Iy"] + pl["Iz"]) and pres(pl["yc"], pl["zc"]))
pli = G.proprietes([G.contour_L(150, 75, 9, 12, 6)])
chk("cornière inégale : 0 < α < 45° et I_u > I_y > I_z > I_v", 0.0 < pli["alpha_deg"] < 45.0 and pli["Iu"] > pli["Iy"] > pli["Iz"] > pli["Iv"])

print("\n=== 3. Torsion de Saint-Venant ===")
a_, c_ = 100.0, 10.0
serie = sum(math.tanh(n * math.pi * a_ / (2 * c_)) / n ** 5 for n in range(1, 60, 2))
J_exact = a_ * c_ ** 3 / 3 * (1 - 192 * c_ / (math.pi ** 5 * a_) * serie)
J20 = G.torsion_saint_venant([G.contour_rectangle(a_, c_)], c_ / 20)
J40 = G.torsion_saint_venant([G.contour_rectangle(a_, c_)], c_ / 40)
chk("rectangle 100 × 10 : I_t à 0,6 % de la série exacte au pas t/20, 0,2 % au pas t/40, convergence en O(h²)",
    pres(J20, J_exact, 6e-3) and pres(J40, J_exact, 2e-3) and abs(J40 - J_exact) < 0.4 * abs(J20 - J_exact),
    f"{J20:.0f} {J40:.0f} vs {J_exact:.0f}")
Jc = G.torsion_saint_venant([G.cercle(20.0)], 0.25)
chk("cercle : I_t = π·R⁴/2 à 0,3 %", pres(Jc, math.pi * 20.0 ** 4 / 2, 3e-3))
Jr = G.torsion_saint_venant([G.contour_rectangle(a_, c_)], 0.5, solveur=None)
chk("série exacte du rectangle (familles._torsion_rectangle) = formule du test", pres(F._torsion_rectangle(a_, c_), J_exact))
It_mince = G.torsion_mince(G.ligne_moyenne_I(h, b, tw, tf))
chk("I : Σ b·t³/3 < Saint-Venant avec congés (les congés raidissent)", It_mince < G.torsion_saint_venant([G.contour_I(h, b, tw, tf, r)], 0.5))
It_t, Wt_t = G.torsion_creux_rond(100.0, 5.0)
chk("tube rond : I_t = π·(D⁴ − d⁴)/32, W_t = 2·I_t/D", pres(It_t, math.pi * (100 ** 4 - 90 ** 4) / 32) and pres(Wt_t, 2 * It_t / 100))
It_b, Wt_b = G.torsion_creux_rect(100.0, 100.0, 5.0, 7.5, 5.0)
Am = (100 - 5) ** 2
chk("tube carré : Bredt mince 4·A_m²·t/p_m encadre la formule EN 10210 (coins arrondis) à 5 %",
    pres(It_b, 4 * Am * Am * 5.0 / (4 * (100 - 5)), 5e-2))

print("\n=== 4. Gauchissement et centre de cisaillement (Vlasov, ligne moyenne) ===")
gi = G.gauchissement(G.ligne_moyenne_I(h, b, tw, tf))
hm = h - tf
chk("I doublement symétrique : centre de cisaillement au centre, I_w = I_z,m·h_m²/4 (I_z de la ligne moyenne)",
    pres(gi["y_s"], 0.0, 1e-9) and pres(gi["z_s"], 0.0, 1e-9) and pres(gi["Iw"], gi["Iz_m"] * hm * hm / 4, 1e-6))
hh, bb, tww, tff = 200.0, 80.0, 8.0, 12.0
gu = G.gauchissement(G.ligne_moyenne_U(hh, bb, tww, tff))
bp, hp = bb - tww / 2, hh - tff
# dérivation : flux dans les semelles → H = V·tf·b'²·h'/(4·I_y) ; e·V = H·h' avec I_y = tw·h'³/12 + tf·b'·h'²/2
e = 3 * tff * bp ** 2 / (tww * hp + 6 * tff * bp)
chk("U : centre de cisaillement à e = 3·b'²·t_f/(6·b'·t_f + h'·t_w) de l'axe de l'âme, côté dos",
    pres(gu["y_s"], tww / 2 - e, 1e-9), f"{gu['y_s']:.4f} vs {tww / 2 - e:.4f}")
Iw_classique = tff * bp ** 3 * hp ** 2 / 12 * (3 * bp * tff + 2 * hp * tww) / (6 * bp * tff + hp * tww)
chk("U : I_w = t_f·b'³·h'²/12 · (3·b'·t_f + 2·h'·t_w)/(6·b'·t_f + h'·t_w) (formule classique) à 10⁻⁶",
    pres(gu["Iw"], Iw_classique, 1e-6), f"{gu['Iw']:.4g} vs {Iw_classique:.4g}")
gl = G.gauchissement(G.ligne_moyenne_L(100, 100, 10))
chk("L : centre de cisaillement au croisement des ailes, I_w de la ligne moyenne nul",
    pres(gl["y_s"], 5.0, 1e-6) and pres(gl["z_s"], 5.0, 1e-6) and abs(gl["Iw"]) < 1e-6 * (100 * 10) ** 3)

print("\n=== 5. Familles : calcul complet et conventions ===")
p = F.calculer("HEA", dict(h=96, b=100, tw=5, tf=8, r=12))
chk("HEA 100 : A, I_y, W_el,y, W_pl,y, A_v,z, I_z, W_pl,z, masse en unités de catalogue",
    pres(p["A"], 21.24, 2e-3) and pres(p["Iy"], 349.2, 2e-3) and pres(p["Wel_y"], 72.76, 2e-3) and pres(p["Wpl_y"], 83.01, 2e-3)
    and pres(p["Avz"], 7.56, 2e-3) and pres(p["Iz"], 133.8, 2e-3) and pres(p["Wpl_z"], 41.14, 2e-3) and pres(p["masse"], 16.7, 3e-3),
    str({k: round(p[k], 3) for k in ("A", "Iy", "Wel_y", "Wpl_y", "Avz", "Iz", "Wpl_z", "masse")}))
chk("HEA 100 : I_t de Saint-Venant à 2 % de 5,24 cm⁴, I_w = I_z·(h − t_f)²/4 ≈ 2 590 cm⁶",
    pres(p["It"], 5.24, 2e-2) and pres(p["Iw"], 133.8 * (9.6 - 0.8) ** 2 / 4, 3e-3), f"{p['It']:.3f} {p['Iw']:.0f}")
chk("HEA 100 : élancements c_f/t_f = ((b − t_w)/2 − r)/t_f et c_w/t_w = d/t_w, d = h − 2t_f − 2r = 56 mm",
    pres(p["cf_tf"], ((100 - 5) / 2 - 12) / 8) and pres(p["cw_tw"], 56 / 5) and pres(p["d_ame"], 56.0))
pu = F.calculer("UPN", dict(h=200, b=75, tw=8.5, tf=11.5, r1=11.5, r2=6))
pu4 = F.calculer("UPN", dict(h=400, b=110, tw=14, tf=18, r1=18, r2=9))
chk("UPN : t_f à b/2 du bout (h ≤ 300) puis à (b − t_w)/2 (h > 300) — UPN 200 A = 32,2 cm², UPN 400 A = 91,5 cm² à 0,3 %",
    pres(pu["A"], 32.2, 3e-3) and pres(pu4["A"], 91.5, 3e-3), f"{pu['A']:.2f} {pu4['A']:.2f}")
chk("UPN 200 : centre de gravité e_y = 2,01 cm, centre de cisaillement côté dos (y_s < 0)",
    pres(pu["yc"] / 10, 2.01, 5e-3) and pu["y_s"] < 0)
pn = F.calculer("IPN", dict(h=200, b=90, tw=7.5, tf=11.3, r1=7.5, r2=4.5))
chk("IPN 200 : t_f à b/4 de l'axe — A = 33,4 cm², I_y = 2 140 cm⁴ à 0,3 %", pres(pn["A"], 33.4, 3e-3) and pres(pn["Iy"], 2140, 3e-3))
ps = F.calculer("SHS", dict(h=100, b=100, t=5))
chk("SHS 100 × 5 EN 10210 : r_ext = 1,5·t, A = 18,7 cm², W_pl = 66,4 cm³, I_t à 1 % de 436 cm⁴, c/t = (h − 3t)/t",
    pres(ps["A"], 18.7, 3e-3) and pres(ps["Wpl_y"], 66.4, 3e-3) and pres(ps["It"], 436, 1e-2) and pres(ps["cf_tf"], 85 / 5))
pf = F.calculer("SHS-F", dict(h=100, b=100, t=5))
# Bredt : I_t ∝ A_m²/p_m ; arrondir les coins ôte (4 − π)·r_c² d'aire mais 2·(4 − π)·r_c de périmètre,
# d'où d(I_t)/I_t = 2·(4 − π)·r_c·(1/p − r_c/A) > 0 tant que r_c < A/p : I_t MONTE légèrement avec r_c
chk("SHS 100 × 5 EN 10219 : coins plus grands (r_ext = 2·t) → aire plus petite, I_t à moins de 1 % (Bredt : A_m²/p_m)",
    pf["A"] < ps["A"] and ps["It"] < pf["It"] < 1.01 * ps["It"] and F.rayons("SHS-F", 5.0) == (10.0, 5.0)
    and F.rayons("SHS-F", 8.0) == (20.0, 12.0) and F.rayons("SHS-F", 12.0) == (36.0, 24.0), f"{ps['It']:.1f} {pf['It']:.1f}")
pc = F.calculer("CHS", dict(d=100, t=5))
chk("CHS 100 × 5 : A_v = 2·A/π, I_t exact, W_t = 2·I_t/d", pres(pc["Avz"], 2 * pc["A"] / math.pi) and pres(pc["Wt"], 2 * pc["It"] / 10))
pp = F.calculer("PRS", dict(h=400, b=200, tw=8, tf=12))
chk("PRS : sans congé, I_t = Σ L·t³/3 sur la ligne moyenne (âme de h − t_f), I_w = I_z·(h − t_f)²/4",
    pres(pp["It"], (2 * 200 * 12 ** 3 + (400 - 12) * 8 ** 3) / 3 * 1e-4, 1e-9) and "mince" in pp["methode_It"]
    and pres(pp["Iw"], pp["Iz"] * (40 - 1.2) ** 2 / 4, 1e-9), f"{pp['It']:.4f}")
ppl = F.calculer("PLAT", dict(h=100, b=10))
chk("PLAT 100 × 10 : I_t par la série exacte du rectangle", pres(ppl["It"], J_exact * 1e-4, 1e-9) and "série" in ppl["methode_It"])
pr = F.calculer("ROND", dict(d=20))
chk("ROND 20 : I_t = π·d⁴/32, W_t = I_t/(d/2)", pres(pr["It"], math.pi * 20 ** 4 / 32 * 1e-4) and pres(pr["Wt"], pr["It"] / 1.0))
chk("toutes les familles ont libellé, groupe, norme, dims, forme, description ; groupes connus",
    all(all(k in f for k in ("libelle", "groupe", "norme", "dims", "forme", "description")) and f["groupe"] in F.GROUPES
        for f in F.FAMILLES.values()))
chk("le moteur n'importe que numpy et la bibliothèque standard",
    not any(m in open(os.path.join("acier", "profiles", f), encoding="utf-8").read()
            for f in ("geometrie.py", "familles.py") for m in ("import scipy", "import shapely", "import streamlit", "import pandas")))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
