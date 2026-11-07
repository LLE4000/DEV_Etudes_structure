# -*- coding: utf-8 -*-
"""Figure (e) — Maillage de la coupe (prototype numpy, cas de contrôle) : déformée amplifiée et contrainte de
von Mises ; zoom sur l'angle, sections 1 à 4 et pic de surface. Données : sorties/s07_maillage.json (s07_maillage.py).

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/f_e_maillage.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import s_commun as K  # noqa: E402
import svg as G  # noqa: E402
from svg import fr  # noqa: E402

d = json.load(open(os.path.join(K.SORTIES, "s07_figure.json"), encoding="utf-8"))
nd = np.array(d["noeuds"])
U = np.array(d["U"])
el = np.array(d["elements"])
vm = np.array(d["vm"])
ech_def = d["ech"]
C = K.CAS
e, r = C["e"], C["r"]
VMAX = 210.0
PAL = [(0.0, (255, 247, 236)), (0.25, (253, 212, 158)), (0.5, (252, 141, 89)), (0.75, (215, 48, 31)), (1.0, (127, 0, 0))]


def couleur(v):
    t = min(max(v / VMAX, 0.0), 1.0)
    for (a, ca), (b, cb) in zip(PAL[:-1], PAL[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            c = [round(ca[k] + f * (cb[k] - ca[k])) for k in range(3)]
            return "#%02x%02x%02x" % tuple(c)
    return "#7f0000"


SOUS = [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)]
BORD = [0, 1, 2, 5, 8, 7, 6, 3]
D = G.Dessin(1000, 740)
D.texte_px(20, 26, "Maillage de la coupe (prototype, numpy seul) — cas de contrôle, contrainte de von Mises", taille=17, gras=True, halo=False)
D.texte_px(20, 50, f"{len(nd)} nœuds, {len(el)} éléments à 9 nœuds, déformation plane ; mur raide, contact unilatéral (s07_maillage.py).",
           taille=13, coul=G.ENCRE2, halo=False)
D.texte_px(20, 70, "Ancrage : 30 kN/mm par cheville (hypothèse) ; charge q = R/L répartie sur 10 mm autour de X = 75 mm.",
           taille=13, coul=G.ENCRE2, halo=False)


def dessiner(ox, oy, ech, xy, fen=None, aretes=True):
    D.repere(ox, oy, ech)
    for k, E in enumerate(el):
        for q in SOUS:
            pts = [xy[E[i]] for i in q]
            if fen and not any(fen[0] <= p[0] <= fen[1] and fen[2] <= p[1] <= fen[3] for p in pts):
                continue
            v = float(np.mean([vm[E[i]] for i in q]))
            c = couleur(v)
            D.poly([tuple(p) for p in pts], fill=c, stroke=c, ep=0.35)
    if aretes:
        for E in el:
            pts = [xy[E[i]] for i in BORD]
            if fen and not any(fen[0] <= p[0] <= fen[1] and fen[2] <= p[1] <= fen[3] for p in pts):
                continue
            D.poly([tuple(p) for p in pts], fill="none", stroke="#3d3a35", ep=0.25, op=1.0)


# ---------------------------------------------------------------- vue générale, déformée amplifiée
xy_def = nd + ech_def * U.reshape(-1, 2)
ox, oy, ech = 105, 632, 4.0
D.repere(ox, oy, ech)
x0, y0 = D.P(-18, 132)
x1, y1 = D.P(0, -8)
D.rect_px(x0, y0, x1 - x0, y1 - y0, fill="url(#hachBeton)", stroke="none")
D.ligne((0, 132), (0, -8), coul=G.ENCRE, ep=1.4)
cont = G.contour_L(120, 120, e, r, 0.0)
D.poly(cont, fill="none", stroke=G.MUET, ep=1.0, tirets="4 3")
dessiner(ox, oy, ech, xy_def)
D.fleche((75, 52), (75, 16.5), coul=G.CHARGE, ep=3.0, marq="flR")
D.texte(77, 52, f"R = {fr(C['R'] / 1e3, 2)} kN", taille=13, coul=G.CHARGE, gras=True)
D.fleche((e + 34, 85), (e + 6, 85), coul=G.ENCRE2, ep=2.2, marq="flG")
D.texte(e + 36, 89, f"ancrage : H = {fr(d['H'] / 1e3, 2)} kN", taille=13, coul=G.ENCRE)
D.texte(e + 36, 83, "(équilibre P0 : 33,96 ; classeur : 42,76)", taille=12, coul=G.ENCRE2)
D.rond(0, 0, 5.5, fill=G.COMP, stroke="#fff", ep=1.4)
D.texte(4, -4.5, f"contact trouvé à y = {fr(d['yc'], 2)} mm : l'arête inférieure", taille=12, coul=G.COMP, gras=True)
D.texte(52, 124, f"déformée amplifiée ×{fr(ech_def, 0)} (déplacement maximal {fr(d['umax'], 2)} mm)", taille=12, coul=G.ENCRE2)
D.texte(52, 118, "tirets gris : forme non déformée", taille=12, coul=G.ENCRE2)

# ---------------------------------------------------------------- zoom sur l'angle, non déformé
ox2, oy2, ech2 = 640, 632, 8.6
fen = (-1.0, 40.0, -1.0, 40.0)
D.repere(ox2, oy2, ech2)
x0, y0 = D.P(-5, 41)
x1, y1 = D.P(0, -5)
D.rect_px(x0, y0, x1 - x0, y1 - y0, fill="url(#hachBeton)", stroke="none")
D.add(f'<clipPath id="fenetre"><rect x="{D.P(-5, 41)[0]:.1f}" y="{D.P(-5, 41)[1]:.1f}" width="{46 * ech2:.1f}" '
      f'height="{46 * ech2:.1f}"/></clipPath><g clip-path="url(#fenetre)">')
dessiner(ox2, oy2, ech2, nd, fen=fen)
D.add("</g>")
D.rect_px(D.P(-5, 41)[0], D.P(-5, 41)[1], 46 * ech2, 46 * ech2, fill="none", stroke="#9c988e", ep=0.8)
D.texte_px(D.P(-5, 41)[0], D.P(-5, 41)[1] - 12, "Zoom sur l'angle (forme réelle, sans amplification)", taille=14, gras=True, halo=False)
a = e + r
# coupes prolongées hors de la matière jusqu'à leur numéro, comme sur la figure (a) : 1 et 2 sous l'aile horizontale
# (coupes u = e + r et u = e), 3 et 4 côté mur (coupes y = e + r et y = e)
dP = 3.3
for (p, q) in (((a, -dP), (a, e)), ((e, -dP), (e, a)), ((-dP, a), (e, a)), ((-dP, e), (a, e))):
    D.ligne(p, q, coul="#1d1d1b", ep=1.6, tirets="5 3")
pk = d["pic"]
D.rond(pk["u"], pk["y"], 6, fill="none", stroke="#1d1d1b", ep=2.0)
D.ligne((pk["u"] - 0.3, pk["y"] + 0.9), (22, 32), coul="#1d1d1b", ep=1.0)
D.texte(16.6, 36.2, f"pic de surface {fr(pk['sig'], 0)} N/mm²", taille=13, coul="#1d1d1b", gras=True)
D.texte(16.6, 33.9, "(principale, 3 mm dans l'arc)", taille=12, coul="#1d1d1b")
D.pastille(a, -dP, 1, r_px=10, taille=12)
D.pastille(e, -dP, 2, r_px=10, taille=12)
D.pastille(-dP, a, 3, r_px=10, taille=12)
D.pastille(-dP, e, 4, r_px=10, taille=12)

# ---------------------------------------------------------------- échelle de couleurs
cx, cy, cw = 640, 700, 320
for k in range(64):
    D.rect_px(cx + k * cw / 64, cy, cw / 64 + 0.6, 12, fill=couleur(VMAX * k / 63), stroke="none")
D.rect_px(cx, cy, cw, 12, fill="none", stroke="#9c988e", ep=0.6)
for v in (0, 50, 100, 150, 200):
    D.texte_px(cx + cw * v / VMAX, cy + 26, fr(v, 0), taille=12, coul=G.ENCRE2, ancre="middle", halo=False)
D.texte_px(cx - 8, cy + 6, "von Mises (N/mm²)", taille=12, coul=G.ENCRE2, ancre="end", halo=False)

out = os.path.join(K.FIGURES, "fig_e_maillage.svg")
D.ecrire(out, "Maillage de la coupe")
print(out)
