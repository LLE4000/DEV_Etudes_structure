# -*- coding: utf-8 -*-
"""Figure (d) — Cornière longue (2,00 m, 3 ancrages) : élévation à l'échelle, efforts par ancrage selon les raideurs,
couple de torsion et moment longitudinal le long de la cornière. Données : sorties/s05_courbes.json (s05_longue.py).

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/f_d_longue.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_commun as K  # noqa: E402
import svg as G  # noqa: E402
from svg import fr  # noqa: E402

d = json.load(open(os.path.join(K.SORTIES, "s05_courbes.json"), encoding="utf-8"))
res = d["res"]
R, Htot = d["R"], d["H_tot"]
anc = d["ancrages"]
NOMS = [("ancrages rigides", "#1f4e8c", "rigides"),
        ("ancrages raides (100 / 30 kN/mm) + aile", G.SECTION, "raides 100 / 30 kN/mm"),
        ("ancrages souples (10 / 5 kN/mm) + aile", G.COMP, "souples 10 / 5 kN/mm")]
Ltot = 2000.0
W = 960
D = G.Dessin(W, 1310)
x0, ech = 70, 0.42                       # px par mm le long de z


def Z(z):
    return x0 + z * ech


D.texte_px(20, 26, "Cornière longue : 2,00 m, trois ancrages, charge localisée (exemple d'illustration)", taille=17, gras=True, halo=False)
D.texte_px(20, 50, "L120×120×15, ancrages Ø16 à 200 / 1 000 / 1 800 mm, Y = 85 ; R = 42,76 kN répartis sur 300 mm centrés à 400 mm, X = 75.",
           taille=13, coul=G.ENCRE2, halo=False)
D.texte_px(20, 70, "Élévation à l'échelle (vue face au mur) ; efforts du modèle poutre + torsion sur appuis élastiques (s05_longue.py).",
           taille=13, coul=G.ENCRE2, halo=False)

# ---------------------------------------------------------------- élévation
yb = 245                                  # pixel du dessous de la cornière
Y = lambda y: yb - y * ech
D.rect_px(Z(-60), Y(260), (Ltot + 120) * ech, (260 + 60) * ech, fill="#f4f2ed", stroke="none")
D.texte_px(Z(2050), Y(245), "mur (vue de face)", taille=12, coul=G.ENCRE2, ancre="end")
D.rect_px(Z(0), Y(120), Ltot * ech, 120 * ech, fill=G.ACIER, stroke=G.ENCRE, ep=1.6)
D.ligne_px(Z(0), Y(15), Z(Ltot), Y(15), coul=G.ENCRE, ep=1.4)
D.ligne_px(Z(0), Y(28), Z(Ltot), Y(28), coul=G.MUET, ep=0.7)
D.texte_px(Z(1400), Y(45), "face intérieure de l'aile verticale ; bande du bas : chant de l'aile horizontale", taille=11, coul=G.ENCRE2, ancre="middle")
for i, z in enumerate(anc):
    D.add(f'<circle cx="{Z(z):.1f}" cy="{Y(85):.1f}" r="{15 * ech:.1f}" fill="#b9b4aa" stroke="{G.ENCRE}" stroke-width="0.8"/>')
    D.add(f'<circle cx="{Z(z):.1f}" cy="{Y(85):.1f}" r="{8 * ech:.1f}" fill="#7f7a70" stroke="{G.ENCRE}" stroke-width="0.6"/>')
    D.texte_px(Z(z) - 13, Y(85), f"A{i + 1}", taille=14, coul=G.ENCRE, gras=True, ancre="end")
z1, z2 = 250.0, 550.0
D.rect_px(Z(z1), Y(15) - 4, (z2 - z1) * ech, 4, fill=G.CHARGE, stroke="none", op=0.6)
for k in range(7):
    z = z1 + k * (z2 - z1) / 6
    D.ligne_px(Z(z), Y(210), Z(z), Y(16), coul=G.CHARGE, ep=1.6, fleche="flR")
D.ligne_px(Z(z1), Y(210), Z(z2), Y(210), coul=G.CHARGE, ep=1.6)
D.texte_px(Z(z2) + 8, Y(205), f"R = {fr(R / 1e3, 2)} kN sur 300 mm", taille=13, coul=G.CHARGE, gras=True)
D.texte_px(Z(z2) + 8, Y(178), "posée sur l'aile horizontale, à X = 75 mm du mur", taille=12, coul=G.CHARGE)
# cotes
yc = Y(-38)
pts = [0.0] + anc + [Ltot]
for a, b in zip(pts[:-1], pts[1:]):
    D.ligne_px(Z(a), yc, Z(b), yc, coul=G.COTE, ep=0.9, fleche="flC", debut="flC")
    D.texte_px((Z(a) + Z(b)) / 2, yc + 13, fr(b - a, 0), taille=13, coul=G.COTE, ancre="middle")
for z in pts:
    D.ligne_px(Z(z), Y(-2), Z(z), yc + 6, coul=G.COTE, ep=0.5)
yc2 = Y(-78)
D.ligne_px(Z(0), yc2, Z(Ltot), yc2, coul=G.COTE, ep=0.9, fleche="flC", debut="flC")
D.texte_px((Z(0) + Z(Ltot)) / 2, yc2 + 13, "longueur totale 2 000", taille=13, coul=G.COTE, ancre="middle")
yc3 = Y(232)
D.ligne_px(Z(0), yc3, Z(z1), yc3, coul=G.COTE, ep=0.8, fleche="flC", debut="flC")
D.texte_px((Z(0) + Z(z1)) / 2, yc3 - 10, "250", taille=12, coul=G.COTE, ancre="middle")
D.ligne_px(Z(z1), yc3, Z(z2), yc3, coul=G.COTE, ep=0.8, fleche="flC", debut="flC")
D.texte_px((Z(z1) + Z(z2)) / 2, yc3 - 10, "largeur d'appui 300", taille=12, coul=G.COTE, ancre="middle")
for z in (0.0, z1, z2):
    D.ligne_px(Z(z), yc3 - 5, Z(z), Y(212) if z else Y(122), coul=G.COTE, ep=0.5)


# ---------------------------------------------------------------- graphiques
class Gr:
    def __init__(self, y, hh, vmin, vmax, titre, unite, pas, nd=0):
        self.y, self.h, self.vmin, self.vmax = y, hh, vmin, vmax
        D.texte_px(20, y - 16, titre, taille=14, gras=True, halo=False)
        v = vmin
        while v <= vmax + 1e-9:
            D.ligne_px(Z(0), self.V(v), Z(Ltot), self.V(v), coul="#e3e1da" if abs(v) > 1e-9 else "#9c988e", ep=0.7 if abs(v) > 1e-9 else 1.0)
            D.texte_px(Z(0) - 6, self.V(v), fr(v, nd).replace("-", "−"), taille=12, coul=G.ENCRE2, ancre="end", halo=False)
            v += pas
        D.texte_px(Z(0) - 44, y + hh / 2, unite, taille=12, coul=G.ENCRE2, ancre="middle", rot=-90, halo=False)
        for z in anc:
            D.ligne_px(Z(z), y, Z(z), y + hh, coul="#cfcbc2", ep=0.8, tirets="3 3")

    def V(self, v):
        return self.y + self.h - (v - self.vmin) / (self.vmax - self.vmin) * self.h


def barres(gr, cle, ref, ref_txt, cl_val=None):
    for i, z in enumerate(anc):
        for j, (nom, coul, lab) in enumerate(NOMS):
            v = res[nom][cle][i] / 1e3
            x = Z(z) + (j - 1) * 26 - 11
            y0p, y1p = gr.V(0), gr.V(v)
            D.rect_px(x, min(y0p, y1p), 22, abs(y1p - y0p), fill=coul, stroke="none", op=0.9)
            D.texte_px(x + 11, (y1p - 9) if v >= 0 else (y1p + 10), fr(v, 1), taille=11, coul=coul, ancre="middle")
    D.ligne_px(Z(0), gr.V(ref), Z(Ltot), gr.V(ref), coul=G.ENCRE2, ep=1.3, tirets="7 4")
    D.texte_px(Z(Ltot), gr.V(ref) - 9, ref_txt, taille=12, coul=G.ENCRE2, ancre="end")
    if cl_val is not None:
        D.ligne_px(Z(0), gr.V(cl_val), Z(Ltot), gr.V(cl_val), coul=G.CHARGE, ep=1.0, tirets="2 3")
        D.texte_px(Z(Ltot), gr.V(cl_val) - 9, f"classeur : {fr(cl_val, 2)} kN", taille=12, coul=G.CHARGE, ancre="end")


g1 = Gr(350, 170, 0, 30, "Traction N par ancrage (kN) — la cornière, souple en torsion, renvoie le couple vers l'ancrage voisin", "N (kN)", 10)
barres(g1, "N", Htot / 3e3, f"« R/n » : H/3 = {fr(Htot / 3e3, 2)} kN (P0)", 14.2533)
g2 = Gr(580, 190, -5, 35, "Cisaillement V par ancrage (kN) — la charge n'est pas au centre du groupe d'ancrages", "V (kN)", 10)
barres(g2, "V", R / 3e3, f"« R/n » = {fr(R / 3e3, 2)} kN")
for i, v in enumerate(d["V_levier"]):
    D.add(f'<circle cx="{Z(anc[i]) + 50:.1f}" cy="{g2.V(v / 1e3):.1f}" r="4" fill="#fff" stroke="{G.ENCRE}" stroke-width="1.4"/>')


def courbe(gr, cle, coul, fac):
    z = res[NOMS[0][0]]["z"]
    pts = " ".join(f"{Z(a):.1f},{gr.V(b * fac):.1f}" for a, b in zip(z, res[cle[0]][cle[1]]))
    D.add(f'<polyline points="{pts}" fill="none" stroke="{coul}" stroke-width="2"/>')


g3 = Gr(830, 140, -2.5, 1.5, "Couple de torsion T(z) (kN·m) — τ = T·e/It", "T (kN·m)", 0.5, nd=1)
for nom, coul, lab in NOMS:
    courbe(g3, (nom, "Tz"), coul, 1e-6)
for nom, coul, lab in NOMS:
    tmax = res[nom]["tau"]
D.texte_px(Z(1200), g3.V(-1.6), "τ max : " + " ; ".join(f"{fr(res[n]['tau'], 0)}" for n, _, _ in NOMS) + " N/mm² (τ_adm = 136,6)",
           taille=12, coul=G.ENCRE)
g4 = Gr(1030, 150, -3, 5, "Moment de flexion longitudinale M(z) (kN·m) — non vérifié par le classeur", "M (kN·m)", 2)
for nom, coul, lab in NOMS:
    courbe(g4, (nom, "Mvz"), coul, 1e-6)
D.texte_px(Z(1100), g4.V(3.6), "|M| max ≈ " + " ; ".join(f"{fr(res[n]['Ml'] / 1e6, 2)}" for n, _, _ in NOMS) + " kN·m ; σz ≈ "
           + fr(res[NOMS[0][0]]["sz"], 0) + " N/mm²", taille=12, coul=G.ENCRE)
D.texte_px(Z(Ltot) / 2 + 30, 1214, "z (mm) le long de la cornière", taille=12, coul=G.ENCRE2, ancre="middle", halo=False)
for k in range(0, 2001, 200):
    D.texte_px(Z(k), 1196, fr(k, 0), taille=11, coul=G.ENCRE2, ancre="middle", halo=False)
# légende
ly = 1246
for j, (nom, coul, lab) in enumerate(NOMS):
    D.rect_px(70 + 300 * j, ly - 7, 22, 14, fill=coul, stroke="none")
    D.texte_px(98 + 300 * j, ly, f"ancrages {lab}", taille=13, coul=G.ENCRE, halo=False)
D.add(f'<circle cx="81" cy="{ly + 26}" r="4" fill="#fff" stroke="{G.ENCRE}" stroke-width="1.4"/>')
D.texte_px(98, ly + 26, "règle du levier : limite d'ancrages très souples devant la cornière (poutre continue si les ancrages sont raides)",
           taille=12, coul=G.ENCRE2, halo=False)
D.texte_px(98, ly + 46, "raides et souples : raideur de l'ancrage en série avec la flexion locale de l'aile",
           taille=12, coul=G.ENCRE2, halo=False)

out = os.path.join(K.FIGURES, "fig_d_longue.svg")
D.ecrire(out, "Cornière longue")
print(out)
