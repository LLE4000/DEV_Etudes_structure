# -*- coding: utf-8 -*-
"""c05 — Cisaillement et contrainte équivalente.

1. aile horizontale : τ moyen R/(b·t) du classeur, τ réel parabolique, combinaison point par point ;
2. aile verticale : quelle force est un effort normal, laquelle est un effort tranchant (équilibre du corps libre
   au-dessus de la section 3) ; formule corrigée σ = M/W + N/A, τ = T/A ; écart au classeur, et condition sur Y.
"""
import json
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import commun as K  # noqa: E402

RACINE = os.path.dirname(ICI)
p = K.CAS
e, r, X, Y, R = p["e"], p["r"], p["X"], p["Y"], p["R"]
b = p["b_cm"] * 10
A = b * e
W = b * e ** 2 / 6
xl = K.classeur(**p)
adm = xl["adm"]
res = {}

# ---------------------------------------------------------------- 1. aile horizontale, section 1
s1, t1 = xl["S1"]["sigma"], xl["S1"]["tau"]
pts = []
for i in range(0, 201):
    z = -e / 2 + e * i / 200                       # cote depuis le plan moyen
    s = s1 * z / (e / 2)
    t = 1.5 * t1 * (1 - (2 * z / e) ** 2)
    pts.append((z, s, t, math.sqrt(s * s + 3 * t * t)))
eq_point = max(p_[3] for p_ in pts)
print(f"S1 : σ_max = {s1:.2f}, τ_moy = {t1:.2f}, τ_max (parabole) = {1.5 * t1:.2f} MPa")
print(f"   classeur √(σ_max² + 3τ_moy²) = {xl['S1']['sigma_eq']:.2f} ; maximum point par point de √(σ² + 3τ²) = "
      f"{eq_point:.2f} MPa → classeur {(xl['S1']['sigma_eq'] / eq_point - 1) * 100:+.2f} % (côté sûr)")
print(f"   critère de cisaillement seul : τ_max/τ_adm = {1.5 * t1 / xl['tadm'] * 100:.1f} % (classeur, τ moyen : "
      f"{xl['S1']['taux_tau'] * 100:.1f} %)")
res["S1"] = dict(sigma=s1, tau_moy=t1, tau_max=1.5 * t1, eq_classeur=xl["S1"]["sigma_eq"], eq_point=eq_point)

# ---------------------------------------------------------------- 2. aile verticale, section 3
# corps libre AU-DESSUS de la section 3 : la cheville (T horizontal, R vertical) → dans l'aile verticale,
# R est un EFFORT NORMAL (traction), T est l'EFFORT TRANCHANT.
T_cl = R * X / xl["C25_YH"]                      # la traction que suppose le classeur (pivot e/2, moment au nu du mur)
N = R
print(f"\nS3 : le classeur prend τ = R/(b·e) = {xl['S3']['tau']:.2f} MPa (R = {R} kN). Équilibre du corps libre "
      f"au-dessus de S3 : effort normal N = R = {N} kN (traction), effort tranchant T = R·X/YH = {T_cl:.2f} kN.")
lignes = []
for nom, x_V, y_c in (("statique du classeur (x_V = 0, y_c = e/2)", 0.0, e / 2),
                      ("tige au milieu de l'épaisseur (x_V = e/2, y_c = e/2)", e / 2, e / 2),
                      ("tige au milieu, pivot au talon (x_V = e/2, y_c = 0)", e / 2, 0.0),
                      ("tige au milieu, pivot à 15 mm (support souple)", e / 2, 15.0)):
    T = R * (X - x_V) / (Y - y_c)
    M = T * (Y - (e + r)) + R * (x_V - e / 2)    # kN·mm, autour du centre de la section 3
    sig = abs(M) * 1e3 / W + N * 1e3 / A
    tau = T * 1e3 / A
    eq = math.sqrt(sig ** 2 + 3 * tau ** 2)
    lignes.append(dict(nom=nom, x_V=x_V, y_c=y_c, T=T, M=M / 1000, sigma=sig, tau=tau, eq=eq, taux=eq / adm))
    print(f"   {nom:52s} T = {T:6.2f} kN, M = {M / 1000:.4f} kN·m, σ = {sig:6.2f}, τ = {tau:5.2f}, σ_eq = {eq:6.2f} "
          f"→ {eq / adm * 100:6.2f} % (classeur {xl['S3']['taux'] * 100:.2f} %)")
# variante « on garde le moment du classeur et on ajoute seulement N » (correction partielle, ce qu'on serait tenté
# de faire)
sig = xl["S3"]["sigma"] + N * 1e3 / A
tau = T_cl * 1e3 / A
eq = math.sqrt(sig ** 2 + 3 * tau ** 2)
print(f"   moment du classeur + N/A, τ = T/A (correction partielle)       σ = {sig:.2f}, τ = {tau:.2f}, σ_eq = {eq:.2f} "
      f"→ {eq / adm * 100:.2f} %")
lignes.append(dict(nom="moment du classeur + N/A", sigma=sig, tau=tau, eq=eq, taux=eq / adm))
res["S3"] = lignes
# identité : avec x_V = 0 et le pivot du classeur, σ_vrai = σ_classeur − 2R/(b·t) exactement
d = lignes[0]["sigma"] - xl["S3"]["sigma"]
print(f"   identité (x_V = 0) : σ_cohérent − σ_classeur = {d:.3f} MPa = −2R/(b·e) = {-2 * R * 1e3 / A:.3f}")
# condition sur Y pour x_V = e/2 : σ_cohérent > σ_classeur  ⇔  Y > 1,25 e + 1,5 r  est FAUX (côté sûr) ;
Ylim = 1.25 * e + 1.5 * r
print(f"   avec x_V = e/2 et le pivot du classeur, le classeur n'est plus du côté sûr que si Y < 1,25e + 1,5r = {Ylim:.2f} mm")
for Yv in (30.0, 35.0, Ylim, 45.0, 85.0):
    c = K.classeur(**{**p, "Y": Yv})
    T = R * (X - e / 2) / (Yv - e / 2)
    M = T * (Yv - (e + r))
    sig = abs(M) * 1e3 / W + N * 1e3 / A
    print(f"     Y = {Yv:6.2f} : σ cohérent = {sig:7.2f} MPa, σ classeur = {c['S3']['sigma']:7.2f} MPa "
          f"({(sig / c['S3']['sigma'] - 1) * 100:+.1f} %)")
res["Y_limite"] = Ylim
json.dump(res, open(os.path.join(RACINE, "sorties", "c05_cisaillement.json"), "w"), ensure_ascii=False, indent=1)
