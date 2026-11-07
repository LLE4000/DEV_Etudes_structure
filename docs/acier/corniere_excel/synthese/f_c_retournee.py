# -*- coding: utf-8 -*-
"""Figure (c) — Configuration A (classeur) et configuration B (cornière retournée), même cornière, même charge.

Le diagramme de moment de l'aile verticale est porté côté mur (comme sur le schéma de la note).
Données : s_commun (classeur, corps libre) et sorties/s04_retournee.json.
Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/f_c_retournee.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_commun as K  # noqa: E402
import svg as G  # noqa: E402
from svg import fr  # noqa: E402

C = K.CAS
e, r, X, Y, R, L = C["e"], C["r"], C["X"], C["Y"], C["R"], C["L"]
larg, haut, r2 = C["larg"], C["haut"], 6.5
h = K.historique(C)
sB = json.load(open(os.path.join(K.SORTIES, "s04_retournee.json"), encoding="utf-8"))
s2 = json.load(open(os.path.join(K.SORTIES, "s02_controle.json"), encoding="utf-8"))
MB = sB["B"]["M"]
HB = sB["H_B"]["85_0_7.5"]
fb0 = K.corps_libre_A(C, 0.0, e / 2)
kM = 12.0 / 1e6                     # échelle des moments : 12 mm de dessin par kN·m

D = G.Dessin(960, 1110)
D.texte_px(20, 26, "Cornière retournée : ce qui change dans l'aile verticale", taille=17, gras=True, halo=False)
D.texte_px(20, 50, "Même cornière, même charge, même trou d'ancrage (85 mm de la face extérieure de l'aile horizontale).",
           taille=13, coul=G.ENCRE2, halo=False)
D.texte_px(20, 70, "Moment de l'aile verticale porté côté mur (12 mm de dessin par kN·m) ; cotes en mm, à l'échelle.",
           taille=13, coul=G.ENCRE2, halo=False)
ECH = 2.6


def panneau(oy, titre, retourne):
    ox = 290
    D.repere(ox=ox, oy=oy, ech=ECH)
    yv = (lambda y: haut - y) if retourne else (lambda y: y)
    x0, y0 = D.P(-100, 136 if not retourne else 138)
    x1, y1 = D.P(0, -14)
    D.rect_px(x0, y0, x1 - x0, y1 - y0, fill="url(#hachBeton)", stroke="none")
    D.ligne((0, 136 if not retourne else 138), (0, -14), coul=G.ENCRE, ep=1.5)
    D.texte_px(20, 102 if not retourne else D.P(0, 170)[1], titre, taille=16, gras=True, halo=False)
    cont = [(u, yv(y)) for u, y in G.contour_L(larg, haut, e, r, r2)]
    D.poly(cont, fill=G.ACIER, stroke="none")
    D.poly(cont, fill="url(#hachAcier)", stroke=G.ENCRE, ep=1.8)
    ya = Y if not retourne else haut - Y
    D.poly([(-40, ya - 8), (e + 17, ya - 8), (e + 17, ya + 8), (-40, ya + 8)], fill="#cfcac0", stroke=G.ENCRE2, ep=0.7)
    D.poly([(e, ya - 15), (e + 3, ya - 15), (e + 3, ya + 15), (e, ya + 15)], fill="#b9b4aa", stroke=G.ENCRE, ep=0.9)
    D.poly([(e + 3, ya - 12), (e + 16, ya - 12), (e + 16, ya + 12), (e + 3, ya + 12)], fill="#8f8a80", stroke=G.ENCRE, ep=0.9)
    yface = e if not retourne else haut
    D.fleche((X, yface + 44), (X, yface + 0.8), coul=G.CHARGE, ep=3.4, marq="flR")
    D.texte(X + 3, yface + 42, f"R = {fr(R / 1e3, 2)} kN", taille=14, coul=G.CHARGE, gras=True)
    for yy, n in ((e + r, 3), (e, 4)):
        yy = yv(yy)
        D.ligne((-2, yy), (e if n == 3 else e + r, yy), coul=G.SECTION, ep=2.6)
        D.pastille(e + (8 if n == 3 else r + 7), yy, n, r_px=9, taille=12)
    D.rond(0, 0, 5, fill=G.COMP, stroke="#fff", ep=1.3)
    D.fleche((-14, 0.5), (-1.6, 0.5), coul=G.COMP, ep=2.4, marq="flV")
    D.fleche((e + 36, ya), (e + 18, ya), coul=G.ENCRE2, ep=2.0, marq="flG")
    D.texte(e + 38, ya + 5, "traction H", taille=13, coul=G.ENCRE2)
    if not retourne:
        poly = [(0, e / 2), (-h["Mmax"] * kM, e / 2), (0, Y)]
        D.poly(poly, fill="#ffffff", stroke="none", op=0.85)
        D.poly(poly, fill="url(#hachMomentV)", stroke=G.SECTION, ep=1.6)
        D.poly([(-fb0["M"](e) * kM, e), (0, Y)], stroke=G.COMP, ep=1.8, ferme=False, tirets="6 3")
        D.texte(-h["M"][2] * kM - 3, e + r, f"M3 = {fr(h['M'][2] / 1e6, 3)}", taille=13, coul=G.SECTION, gras=True, ancre="end")
        D.texte(-h["Mmax"] * kM - 3, e / 2, f"Mmax = {fr(h['Mmax'] / 1e6, 3)}", taille=12, coul=G.SECTION, ancre="end")
        D.texte(-9, 66, "nul à l'ancrage,", taille=12, coul=G.SECTION, ancre="end")
        D.texte(-9, 60, "maximal vers l'angle", taille=12, coul=G.SECTION, ancre="end")
        D.texte(4, -8, "contact à l'arête inférieure", taille=12, coul=G.COMP)
        D.texte(e + 38, ya - 1, f"{fr(h['H'] / 1e3, 2)} kN (classeur)", taille=12, coul=G.ENCRE2)
        D.texte(e + 38, ya - 7, f"{fr(fb0['H'] / 1e3, 2)} kN (équilibre P0)", taille=12, coul=G.COMP)
        D.texte(e + 4, 70, "aile verticale TENDUE (N = +R)", taille=12, coul=G.ENCRE2)
        D.texte(e + 4, 64, "face du congé tendue", taille=12, coul=G.ENCRE2)
        D.cote((-92, 0), (-92, ya), 0, f"Y = {fr(ya, 0)}", ecart_txt=10, taille=13)
        D.ligne((-1, ya), (-95, ya), coul=G.COTE, ep=0.5)
        D.ligne((-1, 0), (-95, 0), coul=G.COTE, ep=0.5)
    else:
        poly = [(0, 0), (-HB * ya * kM, ya), (-MB * kM, haut - e), (0, haut - e)]
        D.poly(poly, fill="#ffffff", stroke="none", op=0.85)
        D.poly(poly, fill="url(#hachMomentV)", stroke=G.SECTION, ep=1.6)
        D.texte(-MB * kM - 3, haut - e - r, f"M3 = M4 = {fr(MB / 1e6, 3)}", taille=13, coul=G.SECTION, gras=True, ancre="end")
        D.texte(-MB * kM / 2, (ya + haut - e) / 2, "constant de l'ancrage à l'angle", taille=12, coul=G.SECTION, gras=True,
                ancre="middle", rot=-90)
        D.texte(-MB * kM - 3, ya / 2, "nul au bout libre", taille=12, coul=G.SECTION, ancre="end")
        D.texte(4, -8, "contact au BOUT LIBRE (maillage : y = 0)", taille=12, coul=G.COMP)
        D.texte(e + 38, ya - 1, f"{fr(HB / 1e3, 2)} kN (bras {fr(ya, 0)} mm)", taille=12, coul=G.ENCRE2)
        D.texte(e + 4, 86, "aile verticale COMPRIMÉE (N = −R)", taille=12, coul=G.ENCRE2)
        D.texte(e + 4, 80, "face du congé comprimée", taille=12, coul=G.ENCRE2)
        D.texte(e + 4, 74, "trou d'ancrage dans le moment maximal", taille=12, coul=G.CHARGE)
        D.cote((-92, ya), (-92, haut), 0, f"Y_B = {fr(haut - ya, 0)}", ecart_txt=10, taille=13)
        D.cote((-84, 0), (-84, ya), 0, f"bras {fr(ya, 0)}", ecart_txt=-10, taille=13)
        D.ligne((-1, ya), (-95, ya), coul=G.COTE, ep=0.5)
        D.ligne((-1, haut), (-95, haut), coul=G.COTE, ep=0.5)
        D.ligne((-1, 0), (-87, 0), coul=G.COTE, ep=0.5)


def bilan(y, titre, lignes, coul_titre):
    D.rect_px(640, y, 300, 30 + 27 * len(lignes), fill="#ffffff", stroke="#c9c5bb", ep=0.8, rx=4)
    D.texte_px(654, y + 18, titre, taille=14, gras=True, coul=coul_titre, halo=False)
    for i, (t, c, g) in enumerate(lignes):
        D.texte_px(654, y + 45 + 27 * i, t, taille=13, coul=c, gras=g, halo=False)


panneau(470, "A — aile horizontale en bas (classeur)", False)
tA = K.taux_section_v(C, fb0["M"](e + r), R, fb0["H"])[0]
bilan(110, "Configuration A", [
    (f"M3 = {fr(h['M'][2] / 1e6, 3)} kN·m (classeur)", G.ENCRE, False),
    (f"M3 = {fr(fb0['M'](e + r) / 1e6, 3)} kN·m (équilibre P0)", G.COMP, False),
    (f"taux 3 = {fr(100 * h['taux'][2], 2)} % (classeur)", G.ENCRE, True),
    (f"taux 3 = {fr(100 * tA, 2)} % (P0, N = +R, τ = H/A)", G.COMP, False),
    (f"traction : {fr(3 * h['N_els'] / 1e3, 2)} kN (C82 × 3)", G.ENCRE, False),
    ("ancrage idéal : le plus HAUT possible", G.ENCRE2, False),
    ("tirets verts sur le dessin : équilibre P0", G.COMP, False)], G.ENCRE)
panneau(1020, "B — aile horizontale en haut (retournée)", True)
bilan(600, "Configuration B", [
    (f"M3 = R·(X − e/2) = {fr(MB / 1e6, 3)} kN·m", G.ENCRE, False),
    ("indépendant de la position de l'ancrage", G.ENCRE2, False),
    (f"taux 3 = {fr(100 * sB['B']['taux'], 2)} % (σ = M/W + R/A)", G.CHARGE, True),
    (f"traction : {fr(HB / 1e3, 2)} kN (×{fr(HB / fb0['H'], 2)} sur A, P0)", G.CHARGE, False),
    (f"section nette au trou : {fr(100 * sB['net'], 1)} %", G.CHARGE, False),
    ("ancrage idéal : PRÈS de l'angle", G.ENCRE2, False)], G.CHARGE)

out = os.path.join(K.FIGURES, "fig_c_retournee.svg")
D.ecrire(out, "Cornière retournée")
print(out)
