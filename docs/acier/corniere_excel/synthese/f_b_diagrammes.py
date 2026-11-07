# -*- coding: utf-8 -*-
"""Figure (b) — Diagrammes continus : moment, épaisseur locale, contrainte équivalente et taux, dans chaque aile.

Données : sorties/s03_courbes.json (s03_continus.py) et s_commun.historique (valeurs des sections).
Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/f_b_diagrammes.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_commun as K  # noqa: E402
import svg as G  # noqa: E402
from svg import fr  # noqa: E402

C = K.CAS
e, r, X, Y, R = C["e"], C["r"], C["X"], C["Y"], C["R"]
h = K.historique(C)
cb = json.load(open(os.path.join(K.SORTIES, "s03_courbes.json"), encoding="utf-8"))
mx = json.load(open(os.path.join(K.SORTIES, "s03_continus.json"), encoding="utf-8"))
sadm = h["sadm"]

W = 920
D = G.Dessin(W, 1290)


class Graphe:
    def __init__(self, x, y, w, hh, xmin, xmax, ymin, ymax):
        self.x, self.y, self.w, self.h = x, y, w, hh
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax

    def X(self, v):
        return self.x + (v - self.xmin) / (self.xmax - self.xmin) * self.w

    def Y(self, v):
        return self.y + self.h - (v - self.ymin) / (self.ymax - self.ymin) * self.h

    def cadre(self, pas_x, pas_y, fmt_y=lambda v: fr(v, 0), titre_y="", grille=True, axe_x=True):
        if grille:
            v = self.ymin
            while v <= self.ymax + 1e-9:
                D.ligne_px(self.x, self.Y(v), self.x + self.w, self.Y(v), coul="#e3e1da", ep=0.7)
                D.texte_px(self.x - 7, self.Y(v), fmt_y(v), taille=12, coul=G.ENCRE2, ancre="end", halo=False)
                v += pas_y
        v = self.xmin
        while v <= self.xmax + 1e-9:
            D.ligne_px(self.X(v), self.y, self.X(v), self.y + self.h, coul="#efede7", ep=0.6)
            if axe_x:
                D.texte_px(self.X(v), self.y + self.h + 13, fr(v, 0), taille=12, coul=G.ENCRE2, ancre="middle", halo=False)
            v += pas_x
        D.ligne_px(self.x, self.y + self.h, self.x + self.w, self.y + self.h, coul="#9c988e", ep=0.9)
        D.ligne_px(self.x, self.y, self.x, self.y + self.h, coul="#9c988e", ep=0.9)
        if titre_y:
            D.texte_px(self.x - 46, self.y + self.h / 2, titre_y, taille=12, coul=G.ENCRE2, ancre="middle", rot=-90, halo=False)

    def zones(self, etiquettes=True):
        x0, x1, x2 = self.X(self.xmin), self.X(e), self.X(e + r)
        D.rect_px(x0, self.y, x1 - x0, self.h, fill="url(#hachGris)", stroke="none")
        D.rect_px(x1, self.y, x2 - x1, self.h, fill="#fde3d3", stroke="none", op=0.75)
        if etiquettes:
            D.texte_px((x0 + x1) / 2, self.y + self.h - 12, "angle", taille=11, coul=G.ENCRE2, ancre="middle")
            D.texte_px((x1 + x2) / 2, self.y + self.h - 12, "congé", taille=11, coul=G.SECTION, ancre="middle")

    def courbe(self, pts, coul, ep=2.2, tirets=None):
        p = " ".join(f"{self.X(a):.2f},{self.Y(b):.2f}" for a, b in pts)
        d = f' stroke-dasharray="{tirets}"' if tirets else ""
        D.add(f'<polyline points="{p}" fill="none" stroke="{coul}" stroke-width="{ep}" stroke-linejoin="round"{d}/>')

    def aire(self, pts, coul, op=0.12):
        p = " ".join(f"{self.X(a):.2f},{self.Y(b):.2f}" for a, b in pts)
        p += f" {self.X(pts[-1][0]):.2f},{self.Y(0):.2f} {self.X(pts[0][0]):.2f},{self.Y(0):.2f}"
        D.add(f'<polygon points="{p}" fill="{coul}" fill-opacity="{op}" stroke="none"/>')

    def section(self, xv, num):
        D.ligne_px(self.X(xv), self.y, self.X(xv), self.y + self.h, coul=G.SECTION, ep=1.4, tirets="5 3")

    def point(self, xv, yv, coul, txt=None, dx=8, dy=-10, ancre="start", taille=13, gras=False):
        D.add(f'<circle cx="{self.X(xv):.2f}" cy="{self.Y(yv):.2f}" r="4.2" fill="{coul}" stroke="#fff" stroke-width="1.3"/>')
        if txt:
            D.texte_px(self.X(xv) + dx, self.Y(yv) + dy, txt, taille=taille, coul=coul, ancre=ancre, gras=gras)


def badge(g, xv, num):
    D.add(f'<circle cx="{g.X(xv):.2f}" cy="{g.y - 11:.2f}" r="10" fill="{G.SECTION}" stroke="#fff" stroke-width="1.3"/>')
    D.texte_px(g.X(xv), g.y - 10.5, str(num), taille=13, coul="#fff", ancre="middle", gras=True, halo=False)


D.texte_px(20, 24, "Diagrammes continus — cas de contrôle (méthode du classeur, sans interpolation)", taille=17, gras=True, halo=False)
D.texte_px(20, 46, "Le moment vient de l'équilibre ; la contrainte vient du moment ET de l'épaisseur locale t. "
           "Abscisse : distance au dos de l'autre aile (mm).", taille=13, coul=G.ENCRE2, halo=False)

gx, gw = 92, W - 92 - 40
# ================================================================= aile horizontale
y0 = 92
D.texte_px(20, y0 - 8, "AILE HORIZONTALE — abscisse u depuis le nu du mur ; charge R à u = X = 75 mm", taille=15, gras=True,
           coul=G.ENCRE, halo=False)
g1 = Graphe(gx, y0 + 26, gw, 170, 0, 120, 0, 3.5)
g1.zones()
g1.cadre(10, 0.5, fmt_y=lambda v: fr(v, 1), titre_y="M (kN·m)")
pts = [(p[0], p[1]) for p in cb["h"]]
coin = [(0.0, cb["Mmax"]), (e, R * (X - e) / 1e6)]
g1.aire(pts, G.CHARGE, 0.08)
g1.courbe(coin, G.CHARGE, ep=1.6, tirets="4 3")
g1.courbe(pts + [(120, 0)], G.CHARGE)
for xv, n in ((e, 2), (e + r, 1)):
    g1.section(xv, n)
    badge(g1, xv, n)
g1.point(e + r, h["M"][0] / 1e6, G.CHARGE, f"M1 = {fr(h['M'][0] / 1e6, 3)}", dx=8, dy=-14)
g1.point(e, h["M"][1] / 1e6, G.CHARGE, f"M2 = {fr(h['M'][1] / 1e6, 3)}", dx=8, dy=-14)
g1.point(0, cb["Mmax"], G.CHARGE)
D.texte_px(g1.X(17), g1.Y(3.32), f"Mmax = R·X = {fr(cb['Mmax'], 3)} au nu du mur (u = 0)", taille=13, coul=G.CHARGE)
D.ligne_px(g1.X(X), g1.Y(1.4), g1.X(X), g1.Y(0.08), coul=G.CHARGE, ep=2.6, fleche="flR")
D.texte_px(g1.X(X) + 6, g1.Y(1.3), "R", taille=14, coul=G.CHARGE, gras=True)
D.texte_px(g1.X(42), g1.Y(2.55), "M(u) = R·(X + Xj − u) : statique exacte du porte-à-faux", taille=13, coul=G.CHARGE)

g2 = Graphe(gx, y0 + 230, gw, 64, 0, 120, 10, 30)
g2.zones(etiquettes=False)
g2.cadre(10, 10, titre_y="t (mm)", axe_x=False)
g2.courbe([(p[0], p[2]) for p in cb["h"]] + [(120, e)], G.COTE, ep=2.0)
D.texte_px(g2.X(48), g2.Y(24), "épaisseur locale t(u) = e + r − √(r² − (e + r − u)²) dans le congé ; e au-delà", taille=12, coul=G.COTE)

g3 = Graphe(gx, y0 + 320, gw, 170, 0, 120, 0, 250)
g3.zones(etiquettes=False)
g3.cadre(10, 50, titre_y="σ_éq (N/mm²)")
D.ligne_px(g3.X(0), g3.Y(sadm), g3.X(120), g3.Y(sadm), coul=G.CHARGE, ep=1.2, tirets="7 4")
D.texte_px(g3.X(119), g3.Y(sadm) - 9, f"σ_adm = f_y/1,5 = {fr(sadm, 1)} (taux 100 %)", taille=12, coul=G.CHARGE, ancre="end")
g3.courbe([(p[0], p[4]) for p in cb["h"]], G.ENCRE, ep=2.2)
for xv, n in ((e, 2), (e + r, 1)):
    g3.section(xv, n)
m = mx["max_h"]["max"]
g3.point(e + r, h["seq"][0], G.SECTION)
D.texte_px(g3.X(13.6), g3.Y(180), f"S1 : {fr(100 * h['taux'][0], 2)} %", taille=13, coul=G.SECTION, ancre="end", gras=True)
g3.point(e, h["seq"][1], G.SECTION, f"S2 : {fr(h['seq'][1], 1)} → {fr(100 * h['taux'][1], 2)} %", dx=8, dy=12)
g3.point(m["s"], m["seq"], G.CHARGE)
D.texte_px(g3.X(36), g3.Y(219), f"maximum : {fr(m['seq'], 1)} → {fr(100 * m['taux'], 2)} % à u = {fr(m['s'], 1)} "
           f"({fr(mx['max_h']['recul'], 1)} mm dans le congé)", taille=13, coul=G.CHARGE, gras=True)
D.ligne_px(g3.X(35.5), g3.Y(219), g3.X(m["s"]) + 4, g3.Y(m["seq"]), coul=G.CHARGE, ep=0.8)
D.texte_px(gx + gw / 2, y0 + 520, "u (mm)", taille=12, coul=G.ENCRE2, ancre="middle", halo=False)

# ================================================================= aile verticale
y0 = 660
D.texte_px(20, y0 - 8, "AILE VERTICALE — abscisse y depuis le dessous de la cornière ; ancrage à y = Y = 85 mm", taille=15,
           gras=True, coul=G.ENCRE, halo=False)
g4 = Graphe(gx, y0 + 26, gw, 170, 0, 120, 0, 3.5)
g4.zones()
g4.cadre(10, 0.5, fmt_y=lambda v: fr(v, 1), titre_y="M (kN·m)")
pv = [(p[0], p[1]) for p in cb["v"]]
p0 = [(p[0], p[1]) for p in cb["v0"]]
g4.aire(pv, G.CHARGE, 0.08)
g4.courbe([(e / 2, cb["Mmax"]), (e, cb["H"] * (Y - e) / 1e6)], G.CHARGE, ep=1.6, tirets="4 3")
g4.courbe(pv + [(120, 0)], G.CHARGE)
g4.courbe(p0, G.COMP, ep=2.0, tirets="8 4")
for xv, n in ((e, 4), (e + r, 3)):
    g4.section(xv, n)
    badge(g4, xv, n)
g4.point(e + r, h["M"][2] / 1e6, G.CHARGE, f"M3 = {fr(h['M'][2] / 1e6, 3)}", dx=10, dy=-14)
g4.point(e, h["M"][3] / 1e6, G.CHARGE, f"M4 = {fr(h['M'][3] / 1e6, 3)}", dx=-8, dy=8, ancre="end")
g4.point(e / 2, cb["Mmax"], G.CHARGE)
D.texte_px(g4.X(17), g4.Y(3.32), f"Mmax = {fr(cb['Mmax'], 3)} au pivot du classeur (y = e/2)", taille=13, coul=G.CHARGE)
g4.point(e + r, cb["v0"][int(round((e + r - e) / 0.1))][1], G.COMP, f"{fr(cb['v0'][int(round(r / 0.1))][1], 3)}", dx=8, dy=14)
D.ligne_px(g4.X(Y), g4.Y(1.0), g4.X(Y), g4.Y(0.06), coul=G.ENCRE2, ep=2.2, fleche="flG")
D.texte_px(g4.X(Y) + 6, g4.Y(0.95), "ancrage", taille=13, coul=G.ENCRE2, gras=True)
D.texte_px(g4.X(42), g4.Y(2.75), f"classeur : M(y) = H·(Y + Yj − y), H = Mmax/YH = {fr(cb['H'] / 1e3, 2)} kN", taille=13, coul=G.CHARGE)
D.texte_px(g4.X(42), g4.Y(2.42), f"équilibre cohérent P0 (pivot à l'arête, V sur l'axe) : H = {fr(cb['H0'] / 1e3, 2)} kN",
           taille=13, coul=G.COMP)

g5 = Graphe(gx, y0 + 230, gw, 64, 0, 120, 10, 30)
g5.zones(etiquettes=False)
g5.cadre(10, 10, titre_y="t (mm)", axe_x=False)
g5.courbe([(p[0], p[2]) for p in cb["v"]] + [(120, e)], G.COTE, ep=2.0)

g6 = Graphe(gx, y0 + 320, gw, 170, 0, 120, 0, 250)
g6.zones(etiquettes=False)
g6.cadre(10, 50, titre_y="σ_éq (N/mm²)")
D.ligne_px(g6.X(0), g6.Y(sadm), g6.X(120), g6.Y(sadm), coul=G.CHARGE, ep=1.2, tirets="7 4")
D.texte_px(g6.X(119), g6.Y(sadm) - 9, f"σ_adm = {fr(sadm, 1)}", taille=12, coul=G.CHARGE, ancre="end")
g6.courbe([(p[0], p[4]) for p in cb["v"]], G.ENCRE, ep=2.2)
g6.courbe([(p[0], p[4]) for p in cb["v0"]], G.COMP, ep=1.8, tirets="8 4")
for xv, n in ((e, 4), (e + r, 3)):
    g6.section(xv, n)
m = mx["max_v"]["max"]
g6.point(e + r, h["seq"][2], G.SECTION)
D.texte_px(g6.X(13.6), g6.Y(203), f"S3 : {fr(100 * h['taux'][2], 2)} %", taille=13, coul=G.SECTION, ancre="end", gras=True)
g6.point(e, h["seq"][3], G.SECTION, f"S4 : {fr(h['seq'][3], 1)} → {fr(100 * h['taux'][3], 2)} %", dx=8, dy=12)
g6.point(m["s"], m["seq"], G.CHARGE)
D.texte_px(g6.X(36), g6.Y(219), f"maximum : {fr(m['seq'], 1)} → {fr(100 * m['taux'], 2)} % à y = {fr(m['s'], 1)} "
           f"({fr(mx['max_v']['recul'], 1)} mm dans le congé)", taille=13, coul=G.CHARGE, gras=True)
D.ligne_px(g6.X(35.5), g6.Y(219), g6.X(m["s"]) + 4, g6.Y(m["seq"]), coul=G.CHARGE, ep=0.8)
m0 = mx["max_v0"]["max"]
D.texte_px(g6.X(60), g6.Y(150), f"P0 (avec N = R et τ = H/A) : maximum {fr(100 * m0['taux'], 2)} %, "
           f"section 3 {fr(100 * mx['max_v0']['pied']['taux'], 2)} %", taille=13, coul=G.COMP)
D.texte_px(gx + gw / 2, y0 + 520, "y (mm)", taille=12, coul=G.ENCRE2, ancre="middle", halo=False)

# légende
ly = 1210
D.rect_px(20, ly - 18, W - 40, 70, fill="#fff", stroke="#c9c5bb", ep=0.8, rx=4)
items = [("#fde3d3", "congé (e ≤ s ≤ e + r) : épaisseur variable"), ("url(#hachGris)", "bloc d'angle (s < e) : la poutre ne s'y applique pas"),
         (G.SECTION, "sections 1 à 4 du classeur (tirets)"), (G.COMP, "équilibre cohérent P0 (tirets verts)")]
for i, (c, t) in enumerate(items):
    cx, cy = 34 + 440 * (i % 2), ly + 22 * (i // 2)
    D.rect_px(cx, cy - 7, 22, 14, fill=c, stroke="#9c988e", ep=0.6)
    D.texte_px(cx + 30, cy, t, taille=13, coul=G.ENCRE, halo=False)

out = os.path.join(K.FIGURES, "fig_b_diagrammes.svg")
D.ecrire(out, "Diagrammes continus")
print(out)
