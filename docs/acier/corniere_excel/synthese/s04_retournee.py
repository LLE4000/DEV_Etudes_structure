# -*- coding: utf-8 -*-
"""S04 — Cornière retournée (configuration B : aile horizontale en haut) et ailes inégales.

Configuration B. Repère : y depuis le BOUT LIBRE de l'aile verticale (en bas), la face supérieure de l'aile
horizontale est à y = haut. Y_B = distance de la face supérieure à l'axe de l'ancrage (la même cote que Y en A :
c'est la même cornière retournée) ; y_a = haut − Y_B = hauteur de l'ancrage au-dessus du bout libre.
R pousse vers le bas à u = X' ; la cornière tourne autour du BOUT LIBRE, qui s'appuie sur le mur (contact à y = c).
  équilibre : H·(y_a − c) = R·(X' − u_V)
  aile verticale, entre l'ancrage et l'angle : M = R·(X' − e/2) constant, N = −R (compression), V = 0
  aile verticale, entre le bout libre et l'ancrage : M = H·(y − c), V = H, N = 0
Statique de la partie située au-dessus de la coupe : seule la charge R y agit (certain, sans hypothèse de pivot).

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/s04_retournee.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_commun as K  # noqa: E402

S = K.Sortie("s04_retournee")
C = K.CAS
e, r, L, R, X = C["e"], C["r"], C["L"], C["R"], C["X"]
haut = C["haut"]
h = K.historique(C)
sadm = h["sadm"]
Xp = X + C["Xj"]

S("# S04 — cornière retournée (B) et ailes inégales")
S("")
S("## 1. Configuration B, cas de contrôle (même cornière, même charge, retournée)")
M_B = R * (Xp - e / 2)
W, A = L * e * e / 6, L * e
sigB = M_B / W + R / A
tauxB = sigB / sadm
tauxB_cl = math.sqrt(sigB ** 2 + 3 * (R / A) ** 2) / sadm
S(f"M3 = M4 = R·(X' − e/2) = {M_B / 1e6:.5f} kN·m, indépendant de la position de l'ancrage (tant qu'il est sous la section 3)")
S(f"section 3 : σ = M/W + R/A = {M_B / W:.2f} + {R / A:.3f} = {sigB:.2f} N/mm² (face côté congé COMPRIMÉE), τ = 0 → taux {100 * tauxB:.2f} % "
  f"(avec le τ = R/(L·e) du classeur : {100 * tauxB_cl:.2f} %)")
S(f"configuration A, même équilibre (P0) : taux 3 = 76,91 % (s02) ; classeur : {100 * h['taux'][2]:.2f} % ; "
  f"M3(B)/M3(classeur A) = {M_B / h['M'][2]:.3f}")
S(f"formule du classeur Mmax·Ye/YH appliquée à B : {h['M'][2] / 1e6:.5f} kN·m, soit {100 * (1 - h['M'][2] / M_B):.1f} % sous le vrai M3")
S.v("B", dict(M=M_B, sig=sigB, taux=tauxB, taux_cl=tauxB_cl))

S("")
S("| Y_B (mm sous la face supérieure) | y_a (mm au-dessus du bout libre) | c (mm) | u_V | H (kN) | par ancrage (kN) | H(B)/H(A, P0) | H(B)/classeur A |")
H_A0 = K.corps_libre_A(C, 0.0, e / 2)["H"]
lignesB = {}
for YB in (45.0, 60.0, 85.0, 95.0):
    for c, uV in ((0.0, e / 2), (0.0, 0.0), (10.0, e / 2)):
        ya = haut - YB
        Hb = R * (Xp - uV) / (ya - c)
        lignesB[f"{YB:.0f}_{c:.0f}_{uV:.1f}"] = Hb
        S(f"| {YB:.0f} | {ya:.0f} | {c:.0f} | {uV:.1f} | {Hb / 1e3:.2f} | {Hb / 3e3:.2f} | {Hb / H_A0:.3f} | {Hb / R:.3f} |")
S.v("H_B", lignesB)
S(f"référence A : P0 H = {H_A0 / 1e3:.3f} kN ; classeur (C82 × 3) = {3 * h['N_els'] / 1e3:.3f} kN")

d0, n = 18.0, 3
bnet = L - n * d0
sig_net = M_B / (bnet * e * e / 6) + R / (bnet * e)
S(f"section nette au droit de l'ancrage en B ({n} trous Ø{d0:.0f} sur {L:.0f} mm, hypothèse) : σ = {sig_net:.1f} N/mm² → {100 * sig_net / sadm:.1f} %")
S(f"en A, le moment est nul à l'ancrage : la section nette n'y travaille qu'en effort normal ({R / (bnet * e):.2f} N/mm²)")
for YB in (85.0, 45.0):
    Hb = R * (Xp - e / 2) / (haut - YB)
    S(f"compression du mur au bout libre (Y_B = {YB:.0f}) : {Hb / 1e3:.2f} kN, soit {Hb / L:.1f} N/mm de cornière")
S.v("net", sig_net / sadm)

# diagramme de l'aile verticale pour la figure (c), A et B
YB = 85.0
ya = haut - YB
HB = R * (Xp - e / 2) / ya
fbA = K.corps_libre_A(C, 0.0, e / 2)
diag = dict(
    A_classeur=[(y, h["H"] * max(C["Y"] - y, 0) / 1e6) for y in [e + 0.5 * k for k in range(0, int((C["Y"] - e) / 0.5) + 1)]],
    A_P0=[(y, fbA["M"](y) / 1e6) for y in [e + 0.5 * k for k in range(0, int((C["Y"] - e) / 0.5) + 1)]],
    B=[(y, (HB * y if y < ya else M_B) / 1e6) for y in [0.5 * k for k in range(0, int((haut - e) / 0.5) + 1)]],
    HB=HB, MB=M_B / 1e6, ya=ya, HA0=fbA["H"], HA=h["H"],
)
with open(os.path.join(K.SORTIES, "s04_diagrammes.json"), "w", encoding="utf-8") as fh:
    json.dump(diag, fh)

S("")
S("## 1b. Configuration A : effet de la hauteur d'ancrage Y (X = 75 mm, mêmes charges)")
for Yv in (65.0, 85.0, 105.0):
    pY = dict(C, Y=Yv)
    hY = K.historique(pY)
    cl = K.corps_libre_A(pY, 0.0, e / 2)
    t0 = K.taux_section_v(pY, cl["M"](e + r), R, cl["H"])[0]
    S(f"   Y = {Yv:.0f} mm : taux 3 du classeur {100 * hY['taux'][2]:.2f} % ; taux 3 en P0 {100 * t0:.2f} % ; traction P0 H = {cl['H'] / 1e3:.2f} kN ; "
      f"traction du classeur n·C82 = {3 * hY['N_els'] / 1e3:.2f} kN")
S("   → monter l'ancrage soulage l'ancrage mais charge la section 3 (Ye/YH → 1)")

S("")
S("## 2. Ce que deviennent les jeux en B")
for dy in (-10.0, 10.0):
    ya2 = haut - (YB + dy)
    S(f"ancrage décalé de {dy:+.0f} mm vers le {'bas' if dy > 0 else 'haut'} (Y_B = {YB + dy:.0f}) : H = {R * (Xp - e / 2) / ya2 / 1e3:.2f} kN ; M3 inchangé {M_B / 1e6:.4f} kN·m")
Xj20 = R * (Xp + 20 - e / 2)
S(f"Xj = 20 mm : M3 = {Xj20 / 1e6:.4f} kN·m, taux 3 = {100 * (Xj20 / W + R / A) / sadm:.2f} %, H = {R * (Xp + 20 - e / 2) / ya / 1e3:.2f} kN")

S("")
S("## 3. Ailes inégales — L150x100x12 (catalogue du site : r1 = 12, r2 = 6), configuration A, même charge")
prof = [q for q in K.profils_site(("LI",)) if q["nom"] == "L 150x100x12"][0]
S(f"catalogue du site : h = {prof['dims']['h']}, b = {prof['dims']['b']}, t = {prof['dims']['t']}, r1 = {prof['dims']['r1']}, r2 = {prof['dims']['r2']}")
ineg = {}
for nom, larg, hauteur in (("grande aile horizontale (150 × 100)", 150.0, 100.0), ("grande aile verticale (100 × 150)", 100.0, 150.0)):
    for rr in (12.0, 6.0):
        p = dict(C, larg=larg, haut=hauteur, e=12.0, r=rr, Y=hauteur - 35.0)
        hh = K.historique(p)
        fb = K.corps_libre_A(p, 0.0, 6.0)
        t0 = K.taux_section_v(p, fb["M"](12.0 + rr), R, fb["H"])[0]
        ineg[f"{nom}_{rr:.0f}"] = dict(t1=hh["taux"][0], t3=hh["taux"][2], tmax=hh["taux_max"], H=3 * hh["N_els"], t3_P0=t0, H_P0=fb["H"])
        S(f"{nom}, r = {rr:.0f}, Y = {p['Y']:.0f} : classeur taux 1 = {100 * hh['taux'][0]:.1f} %, taux 3 = {100 * hh['taux'][2]:.1f} %, "
          f"traction totale {3 * hh['N_els'] / 1e3:.2f} kN | P0 : taux 3 = {100 * t0:.1f} %, H = {fb['H'] / 1e3:.2f} kN")
S.v("inegales", ineg)
S.ecrire()
