# -*- coding: utf-8 -*-
"""Figure (a) — Cas de contrôle, configuration A : géométrie à l'échelle, bras de levier, jeux, sections 1 à 4,
ancrage, résultante de compression. Les valeurs écrites sont recalculées par s_commun.historique.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/f_a_coupe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_commun as K  # noqa: E402
import svg as G  # noqa: E402
from svg import fr  # noqa: E402

C = K.CAS
e, r, X, Y, R = C["e"], C["r"], C["X"], C["Y"], C["R"]
larg, haut, r2 = C["larg"], C["haut"], 6.5
h = K.historique(C)
D = G.Dessin(940, 900)
ech = 3.55
D.repere(ox=405, oy=560, ech=ech)

# ---------------------------------------------------------------- mur
x0, y0 = D.P(-107, 132)
x1, y1 = D.P(0, -58)
D.rect_px(x0, y0, x1 - x0, y1 - y0, fill="url(#hachBeton)", stroke="none")
D.ligne((0, 132), (0, -58), coul=G.ENCRE, ep=1.6)
D.texte(-100, 128, "MUR (support)", taille=13, coul=G.ENCRE2, gras=True, ancre="start")

# ---------------------------------------------------------------- ancrage (tige dans le mur, trou, rondelle, écrou)
d, dt = 16.0, 18.0
D.poly([(-72, Y - d / 2), (0, Y - d / 2), (0, Y + d / 2), (-72, Y + d / 2)], fill="#d9d6cf", stroke=G.ENCRE2, ep=0.8, tirets="5 3")
D.poly([(0, Y - dt / 2), (e, Y - dt / 2), (e, Y + dt / 2), (0, Y + dt / 2)], fill="#ffffff", stroke="none")

# ---------------------------------------------------------------- cornière
cont = G.contour_L(larg, haut, e, r, r2)
D.poly(cont, fill=G.ACIER, stroke="none")
D.poly(cont, fill="url(#hachAcier)", stroke=G.ENCRE, ep=2.0)
# trou de l'ancrage (traits cachés) et tige à travers l'aile
D.poly([(0, Y - dt / 2), (e, Y - dt / 2), (e, Y + dt / 2), (0, Y + dt / 2)], fill="#ffffff", stroke="none")
D.ligne((0, Y - dt / 2), (e, Y - dt / 2), coul=G.ENCRE2, ep=0.8, tirets="4 3")
D.ligne((0, Y + dt / 2), (e, Y + dt / 2), coul=G.ENCRE2, ep=0.8, tirets="4 3")
D.poly([(0, Y - d / 2), (e + 17, Y - d / 2), (e + 17, Y + d / 2), (0, Y + d / 2)], fill="#cfcac0", stroke=G.ENCRE2, ep=0.8)
D.poly([(e, Y - 15), (e + 3, Y - 15), (e + 3, Y + 15), (e, Y + 15)], fill="#b9b4aa", stroke=G.ENCRE, ep=1.0)
D.poly([(e + 3, Y - 12), (e + 16, Y - 12), (e + 16, Y + 12), (e + 3, Y + 12)], fill="#8f8a80", stroke=G.ENCRE, ep=1.0)
D.ligne((e + 3, Y - 5), (e + 16, Y - 5), coul="#5e5a52", ep=0.7)
D.ligne((e + 3, Y + 5), (e + 16, Y + 5), coul="#5e5a52", ep=0.7)
# axe de l'ancrage
D.ligne((-82, Y), (e + 26, Y), coul=G.ENCRE2, ep=0.7, tirets="12 3 2 3")
D.texte(e + 28, Y + 15, "ancrage Ø16", taille=14, coul=G.ENCRE, gras=True)
D.texte(e + 28, Y + 9, "(n = 3 sur 300 mm)", taille=13, coul=G.ENCRE2)
# efforts de l'ancrage sur la cornière
D.fleche((e + 40, Y - 0), (e + 18, Y - 0), coul=G.ENCRE2, ep=2.0, marq="flG")
D.texte(e + 42, Y + 1, f"N : traction ({fr(h['N_els'] / 1e3, 2)} kN/ancrage, C82)", taille=13, coul=G.ENCRE2, base="middle")
D.fleche((e + 9.5, Y - 40), (e + 9.5, Y - 13), coul=G.ENCRE2, ep=2.0, marq="flG")
D.texte(e + 12, Y - 34, f"V = R/n = {fr(h['V_els'] / 1e3, 2)} kN", taille=13, coul=G.ENCRE2)
# Yj : axe d'ancrage décalé
Yj = 20.0
D.ligne((-82, Y + Yj), (e + 4, Y + Yj), coul=G.COTE, ep=0.8, tirets="3 3")
D.cote((-78, Y), (-78, Y + Yj), 0, "", fl=True, ext=False)
D.texte(-75, Y + Yj / 2, "Yj", taille=15, coul=G.COTE, gras=True, ancre="start")
D.texte(-100, Y + Yj + 4, "axe de l'ancrage relevé de Yj (0 ici)", taille=12, coul=G.COTE, ancre="start")

# ---------------------------------------------------------------- charge et jeu Xj
D.fleche((X, 72), (X, e + 0.6), coul=G.CHARGE, ep=4.0, marq="flR")
D.texte(X + 3, 76, f"R = {fr(R / 1e3, 2)} kN", taille=17, coul=G.CHARGE, gras=True)
D.texte(X + 3, 70, "réaction ELS (C12) sur 300 mm", taille=12, coul=G.CHARGE)
Xj = 20.0
D.fleche((X + Xj, 52), (X + Xj, e + 0.6), coul=G.CHARGE, ep=1.4, marq="flR", tirets="6 4")
D.texte(X + Xj + 3, 46, "R déplacée de Xj", taille=12, coul=G.CHARGE)
D.texte(X + Xj + 3, 41, "(0 ici ; 20 mm → 112 %)", taille=12, coul=G.CHARGE)

# ---------------------------------------------------------------- sections 1 à 4
for (a, b, n) in (((e + r, -4.5), (e + r, e), 1), ((e, -4.5), (e, e + r), 2), ((-4.5, e + r), (e, e + r), 3), ((-4.5, e), (e + r, e), 4)):
    D.ligne(a, b, coul=G.SECTION, ep=3.2)
for (u, y, n) in ((e + r, -8.3, 1), (e, -8.3, 2), (-8.3, e + r, 3), (-8.3, e, 4)):
    D.pastille(u, y, n, r_px=11, taille=14)

# ---------------------------------------------------------------- résultante de compression (pivots)
D.rond(0, e / 2, 6, fill=G.COMP, stroke="#fff", ep=1.5)
D.fleche((-14, e / 2), (-1.8, e / 2), coul=G.COMP, ep=2.6, marq="flV")
D.rond(0, 10.0, 3.6, fill="#ffffff", stroke=G.COMP, ep=1.6)
D.ligne((-1.2, 6.2), (-9, -12), coul=G.COMP, ep=0.8)
D.texte(-10, -13, f"C = H = {fr(h['H'] / 1e3, 2)} kN", taille=14, coul=G.COMP, gras=True, ancre="end")
D.texte(-10, -18.5, "pivot du classeur : e/2 = 7,5 mm (●)", taille=12, coul=G.COMP, ancre="end")
D.texte(-10, -23, "ancrages : 1 cm, C76 (○)", taille=12, coul=G.COMP, ancre="end")
D.texte(-10, -27.5, "maillage : arête inférieure (y = 0)", taille=12, coul=G.COMP, ancre="end")

# ---------------------------------------------------------------- cotes verticales (dans le mur)
for (u, y_bas, txt) in ((-17, e + r, f"Ye = {fr(h['Ye'], 0)}"), (-28, e, f"Ye+r = {fr(h['Yer'], 0)}"),
                        (-39, e / 2, f"YH = {fr(h['YH'], 1)}"), (-50, 0.0, f"Y = {fr(Y, 0)}")):
    D.ligne((-1, y_bas), (u - 2, y_bas), coul=G.COTE, ep=0.6)
    D.cote((u, Y), (u, y_bas), 0, txt, ecart_txt=-10, taille=14)
D.ligne((-1, Y), (-82, Y), coul=G.COTE, ep=0.5)
D.cote((-63, 0), (-63, haut), 0, f"haut. = {fr(haut, 0)}", ecart_txt=10, taille=14)
D.ligne((-1, haut), (-66, haut), coul=G.COTE, ep=0.6)
D.ligne((-1, 0), (-66, 0), coul=G.COTE, ep=0.6)

# ---------------------------------------------------------------- cotes horizontales (sous la cornière)
for (u0, yy, txt) in ((e + r, -15, f"Xe = {fr(h['Xe'], 0)}"), (e, -24, f"Xe+r = {fr(h['Xer'], 0)}"),
                      (0.0, -33, f"X = {fr(X, 0)}")):
    D.ligne((u0, -10.5 if u0 > 0 else -1), (u0, yy - 2), coul=G.COTE, ep=0.6)
    D.cote((u0, yy), (X, yy), 0, txt, ecart_txt=-9, taille=14)
D.ligne((X, -1), (X, -35), coul=G.COTE, ep=0.6)
D.ligne((X + Xj, -1), (X + Xj, -35), coul=G.COTE, ep=0.6, tirets="3 3")
D.cote((X, -33), (X + Xj, -33), 0, "Xj", ecart_txt=-9, taille=14)
D.cote((0, -43), (larg, -43), 0, f"larg. = {fr(larg, 0)}", ecart_txt=-9, taille=14)
D.ligne((larg, -1), (larg, -45), coul=G.COTE, ep=0.6)
D.ligne((0, -1), (0, -45), coul=G.COTE, ep=0.6)
# e et r
D.cote((larg + 9, 0), (larg + 9, e), 0, f"e = {fr(e, 0)}", ecart_txt=-11, taille=14)
D.ligne((larg + 1, 0), (larg + 11, 0), coul=G.COTE, ep=0.6)
D.ligne((larg + 1, e), (larg + 11, e), coul=G.COTE, ep=0.6)
import math  # noqa: E402
pm = (e + r - r * math.cos(math.pi / 4), e + r - r * math.sin(math.pi / 4))
D.fleche((pm[0] + 22, pm[1] + 22), (pm[0] + 0.8, pm[1] + 0.8), coul=G.COTE, ep=1.0, marq="flC")
D.texte(pm[0] + 23, pm[1] + 24, f"r = {fr(r, 0)} (congé r1)", taille=14, coul=G.COTE)

# ---------------------------------------------------------------- titre et cartouche
D.texte_px(20, 22, "Coupe de la cornière — cas de contrôle (configuration A, celle du classeur)", taille=17, gras=True, halo=False)
D.texte_px(20, 44, "L120×120×15, r = 13, S355, longueur reprise 300 mm, X = 75, Y = 85, Xj = Yj = 0 ; cotes en mm, à l'échelle",
           taille=13, coul=G.ENCRE2, halo=False)
bx, by = 20, 792
D.rect_px(bx, by, 900, 92, fill="#ffffff", stroke="#c9c5bb", ep=0.8, rx=4)
leg = [(G.SECTION, "1, 2, 3, 4 : sections vérifiées par le classeur"),
       (G.COTE, "bras de levier et jeux (P8!C19 à C25, C14, C17)"),
       (G.CHARGE, "charge R et charge déplacée de Xj (tirets)"),
       (G.COMP, "résultante de compression contre le mur"),
       (G.ENCRE2, "efforts de l'ancrage sur la cornière")]
for i, (c, t) in enumerate(leg):
    cx, cy = bx + 12 + 450 * (i // 3), by + 20 + 26 * (i % 3)
    D.ligne_px(cx, cy, cx + 26, cy, coul=c, ep=3)
    D.texte_px(cx + 34, cy, t, taille=13, coul=G.ENCRE, halo=False)

out = os.path.join(K.FIGURES, "fig_a_coupe_A.svg")
D.ecrire(out, "Coupe du cas de contrôle")
print(out)
