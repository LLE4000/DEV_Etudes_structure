# -*- coding: utf-8 -*-
"""E01 — Diagrammes continus exacts dans les deux ailes (configuration A).

1. Reproduction de M1..M4 et du taux maximal (méthode du classeur, symboles).
2. Lecture des formules comme un équilibre : modèle « E » (celui que les
   formules imposent) et corps libres physiques (P) paramétrés par la hauteur
   du pivot yP et la ligne d'action du cisaillement d'ancrage uV.
3. Diagrammes continus M, N, V ; épaisseur variable dans le congé ; contrainte
   et taux en fonction de la position ; maxima.
4. Figures : schéma coté avec les diagrammes, courbes M / σ_éq / taux, N et V.

Lancer : /usr/bin/python3.12 -I e01_diagrammes.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch  # noqa: E402

import ext_commun as K  # noqa: E402

c = K.CAS
e, r, L = c["e"], c["r"], c["L"]
R = c["R"]
out = []
P = out.append

# ---------------------------------------------------------------- 1. reproduction
ex = K.excel(c)
cache = K.releve_excel()
P("# E01 — diagrammes continus (cas de contrôle L120x120x15, configuration A)")
P("")
P("## 1. Reproduction de la méthode du classeur")
for i, (cell, nom) in enumerate([("C35", "M1"), ("C45", "M2"), ("C57", "M3"), ("C65", "M4")]):
    m = ex["M"][i] / 1e6
    P(f"{nom} = {m:.6f} kN·m   cache {cell} = {cache[cell]:.6f}   écart {m - cache[cell]:+.2e}   "
      f"attendu {K.ATTENDU[nom]:.5f}")
for i, cell in enumerate(["I37", "I47", "I57", "I67"]):
    P(f"taux section {i + 1} = {ex['taux'][i] * 100:.4f} %   cache {cell} = {cache[cell] * 100:.4f} %")
P(f"taux max = {ex['taux_max'] * 100:.4f} % (section {ex['section_max']})   cache D3 = {cache['D3'] * 100:.4f} %")
P(f"N ancrage ELS (C82) = {ex['N_anc'] / 1e3:.4f} kN (cache {cache['C82']:.4f}) ; "
  f"V ancrage ELS (C83) = {ex['V_anc'] / 1e3:.4f} kN (cache {cache['C83']:.4f})")

# ---------------------------------------------------------------- 2. corps libres
Xp, Yp = c["X"] + c["Xj"], c["Y"] + c["Yj"]
H_E = ex["Mmax"] / (Yp - e / 2)          # effort du couple implicite des sections 3 et 4


def Mv_E(y):
    """Modèle E : M(y) = Mmax·(Y' − y)/(Y' − e/2) (C56·C55, C64·C63)."""
    return H_E * max(Yp - y, 0.0)


P("")
P("## 2. Les formules lues comme un équilibre")
P(f"Modèle E : H_E = Mmax/(Y' − e/2) = {ex['Mmax'] / 1e6:.5f} / {(Yp - e / 2) / 1e3:.4f} = {H_E / 1e3:.4f} kN")
# moment au nœud filaire (intersection des lignes moyennes, u = e/2, y = e/2)
Mh_noeud = R * (Xp - e / 2)
Mv_noeud_E = H_E * (Yp - e / 2)
P(f"Nœud filaire (u = y = e/2) : aile horizontale R·(X' − e/2) = {Mh_noeud / 1e6:.5f} kN·m ; "
  f"aile verticale du modèle E H_E·(Y' − e/2) = {Mv_noeud_E / 1e6:.5f} kN·m ; "
  f"saut = R·e/2 = {(Mv_noeud_E - Mh_noeud) / 1e6:.5f} kN·m ({(Mv_noeud_E / Mh_noeud - 1) * 100:.2f} %)")
P("Le modèle E n'est donc pas l'équilibre d'un nœud filaire : son aile verticale reçoit R·X' "
  "(bras compté depuis le nu du mur) quand l'aile horizontale ne délivre que R·(X' − e/2) au nœud.")

variantes = []
for nom, yP, uV in [("P0  pivot y=0, V sur l'axe de l'aile", 0.0, e / 2),
                    ("P1  pivot y=e/2, V sur l'axe (nœud filaire)", e / 2, e / 2),
                    ("P10 pivot y=10 mm (C76), V sur l'axe", 10.0, e / 2),
                    ("P15 pivot y=e, V sur l'axe", e, e / 2),
                    ("Q1  pivot y=e/2, V au nu du mur", e / 2, 0.0)]:
    fb = K.corps_libre_A(c, yP=yP, uV=uV)
    M3 = K.moment_aile_v(e + r, fb, e)
    M4 = K.moment_aile_v(e, fb, e)
    s3 = M3 / (L * e ** 2 / 6)
    t3 = math.sqrt(s3 ** 2 + 3 * (R / (L * e)) ** 2) / ex["sadm"]
    variantes.append((nom, fb, M3, M4, t3))
    P(f"{nom:45s} H = {fb['H'] / 1e3:7.3f} kN   M3 = {M3 / 1e6:.5f}   M4 = {M4 / 1e6:.5f} kN·m   "
      f"taux 3 (convention classeur) = {t3 * 100:6.2f} %   N/ancrage = {fb['H'] / 3e3:.3f} kN")

yP_star = Yp - (Xp - e / 2) * (Yp - e / 2) / Xp
fb_star = K.corps_libre_A(c, yP=yP_star, uV=e / 2)
P(f"Pivot qui redonne EXACTEMENT M3 et M4 du classeur avec V sur l'axe de l'aile : "
  f"yP* = Y' − (X' − e/2)(Y' − e/2)/X' = {yP_star:.4f} mm ; H = {fb_star['H'] / 1e3:.4f} kN ; "
  f"M3 = {K.moment_aile_v(e + r, fb_star, e) / 1e6:.6f} ; M4 = {K.moment_aile_v(e, fb_star, e) / 1e6:.6f} kN·m")

# ---------------------------------------------------------------- 3. balayage continu
def scan_h(n=4001):
    """Aile horizontale : u de e (nu intérieur de l'aile verticale) à X'."""
    us = np.linspace(e, Xp, n)
    t = np.array([K.t_conge(u, e, r) for u in us])
    M = R * (Xp - us)
    W = L * t ** 2 / 6
    sig = M / W
    tau = R / (L * t)
    seq = np.sqrt(sig ** 2 + 3 * tau ** 2)
    return us, t, M, sig, tau, seq


def scan_v(Mfun, Hphys=None, n=4001, avec_N=False):
    """Aile verticale : y de e (face supérieure de l'aile horizontale) à Y'.

    avec_N=False : convention du classeur (σ = M/W, τ = R/(L·t)).
    avec_N=True  : équilibre complet (σ = M/W + R/(L·t), τ = H/(L·t)).
    """
    ys = np.linspace(e, Yp, n)
    t = np.array([K.t_conge(y, e, r) for y in ys])
    M = np.array([Mfun(y) for y in ys])
    W = L * t ** 2 / 6
    A = L * t
    if avec_N:
        sig = M / W + R / A
        tau = Hphys / A
    else:
        sig = M / W
        tau = R / A
    seq = np.sqrt(sig ** 2 + 3 * tau ** 2)
    return ys, t, M, sig, tau, seq


P("")
P("## 3. Balayage continu (pas 0,02 mm environ)")
us, th, Mh, sh, tauh, seqh = scan_h()
i = int(np.argmax(seqh))
P(f"Aile horizontale : σ_éq max = {seqh[i]:.3f} N/mm² à u = {us[i]:.3f} mm "
  f"({us[i] - (e + r):+.3f} mm du pied du congé), t = {th[i]:.3f} mm, M = {Mh[i] / 1e6:.5f} kN·m, "
  f"taux = {seqh[i] / ex['sadm'] * 100:.3f} % (section 1 : {ex['taux'][0] * 100:.3f} %, "
  f"rapport {seqh[i] / ex['seq'][0]:.4f})")
for u0 in (e, e + r / 2, e + r - 2, e + r):
    j = int(np.argmin(abs(us - u0)))
    P(f"   u = {us[j]:6.2f}  t = {th[j]:6.3f}  M = {Mh[j] / 1e6:.5f}  σ = {sh[j]:7.3f}  σ_éq = {seqh[j]:7.3f}  "
      f"taux = {seqh[j] / ex['sadm'] * 100:6.2f} %")

ysE, tvE, MvE, svE, tauvE, seqvE = scan_v(Mv_E)
k = int(np.argmax(seqvE))
P(f"Aile verticale, modèle E : σ_éq max = {seqvE[k]:.3f} N/mm² à y = {ysE[k]:.3f} mm "
  f"({ysE[k] - (e + r):+.3f} mm du pied du congé), t = {tvE[k]:.3f}, M = {MvE[k] / 1e6:.5f} kN·m, "
  f"taux = {seqvE[k] / ex['sadm'] * 100:.3f} % (section 3 : {ex['taux'][2] * 100:.3f} %, "
  f"rapport {seqvE[k] / ex['seq'][2]:.4f})")
for y0 in (e, e + r / 2, e + r - 2, e + r):
    j = int(np.argmin(abs(ysE - y0)))
    P(f"   y = {ysE[j]:6.2f}  t = {tvE[j]:6.3f}  M = {MvE[j] / 1e6:.5f}  σ = {svE[j]:7.3f}  σ_éq = {seqvE[j]:7.3f}  "
      f"taux = {seqvE[j] / ex['sadm'] * 100:6.2f} %")

# équilibre complet (N et vrai tranchant) sur le modèle E et sur P1
fbP1 = variantes[1][1]
ysC, tvC, MvC, svC, tauvC, seqvC = scan_v(Mv_E, Hphys=H_E, avec_N=True)
kC = int(np.argmax(seqvC))
P(f"Aile verticale, modèle E + effort normal R et tranchant H_E : σ_éq max = {seqvC[kC]:.3f} à y = {ysC[kC]:.3f} "
  f"(taux {seqvC[kC] / ex['sadm'] * 100:.3f} %) ; en section 3 : "
  f"{seqvC[int(np.argmin(abs(ysC - (e + r))))] / ex['sadm'] * 100:.3f} %")
ysP, tvP, MvP, svP, tauvP, seqvP = scan_v(lambda y: K.moment_aile_v(y, fbP1, e), Hphys=fbP1["H"], avec_N=True)
kP = int(np.argmax(seqvP))
P(f"Aile verticale, P1 (nœud filaire) + N + vrai tranchant : σ_éq max = {seqvP[kP]:.3f} à y = {ysP[kP]:.3f} "
  f"(taux {seqvP[kP] / ex['sadm'] * 100:.3f} %) ; section 3 : "
  f"{seqvP[int(np.argmin(abs(ysP - (e + r))))] / ex['sadm'] * 100:.3f} %")

# ---------------------------------------------------------------- N et V
ef = K.efforts_A(K.corps_libre_A(c))
P("")
P("## 4. Effort normal et tranchant (équilibre de la cornière isolée, frottement nul)")
P(f"Aile horizontale (e ≤ u < X') : V = R = {R / 1e3:.3f} kN ; N = 0 ; au-delà de la charge : 0.")
P(f"Aile verticale (e ≤ y < Y') : N = +R = {R / 1e3:.3f} kN (traction) ; V = H "
  f"(modèle E : {H_E / 1e3:.3f} kN ; P1 : {fbP1['H'] / 1e3:.3f} kN) ; au-dessus de l'ancrage : 0.")
P(f"Contraintes moyennes en section 3 (L·e = {L * e:.0f} mm²) : σ_N = R/(L·e) = {R / (L * e):.3f} ; "
  f"τ_H(E) = {H_E / (L * e):.3f} ; τ du classeur = R/(L·e) = {R / (L * e):.3f} N/mm²")

# ---------------------------------------------------------------- figures
COL = K.COUL
os.makedirs(K.FIGURES, exist_ok=True)

# --- figure A : schéma à l'échelle avec diagrammes
def contour_L():
    """Contour de la cornière (bouts d'ailes droits, congé intérieur)."""
    B, Hh = c["larg"], c["haut"]
    pts = [(0, 0), (B, 0), (B, e), (e + r, e)]
    for a in np.linspace(-math.pi / 2, -math.pi, 25)[1:-1]:
        pts.append((e + r + r * math.cos(a), e + r + r * math.sin(a)))
    pts += [(e, e + r), (e, Hh), (0, Hh)]
    return pts


fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=150)
ax.set_facecolor(COL["fond"])
# mur
ax.add_patch(Rectangle((-70, -45), 70, 185, facecolor=COL["beton"], edgecolor="none", zorder=0))
ax.plot([0, 0], [-45, 140], color=COL["encre2"], lw=1.0)
ax.add_patch(Polygon(contour_L(), closed=True, facecolor=COL["acier"], edgecolor=COL["encre"], lw=1.2, zorder=3))
# ancrage
ax.add_patch(Rectangle((e, Yp - 9), 10, 18, facecolor=COL["encre2"], edgecolor=COL["encre"], lw=0.8, zorder=4))
ax.plot([-60, e + 10], [Yp, Yp], color=COL["encre2"], lw=0.8, ls=(0, (6, 3)), zorder=4)
# diagramme aile horizontale : sous l'aile, échelle 1 kN·m -> 12 mm
sc = 12.0
uu = np.linspace(0, Xp, 200)
Mh_full = R * (Xp - uu) / 1e6
poly = [(u, -sc * m) for u, m in zip(uu, Mh_full)] + [(Xp, 0), (0, 0)]
ax.add_patch(Polygon(poly, closed=True, facecolor=COL["s1"], alpha=0.18, edgecolor=COL["s1"], lw=1.2, zorder=2))
# zone de l'angle (u < e) hachurée : la théorie des poutres n'y vaut pas
ax.add_patch(Rectangle((0, -sc * Mh_full[0] - 2), e, sc * Mh_full[0] + 2, facecolor="none",
                       edgecolor=COL["muet"], hatch="////", lw=0, zorder=2))
# diagramme aile verticale (modèle E), à gauche dans le mur
yy = np.linspace(e / 2, Yp, 200)
Mv_full = np.array([Mv_E(y) for y in yy]) / 1e6
polyv = [(-sc * m, y) for y, m in zip(yy, Mv_full)] + [(0, Yp), (0, e / 2)]
ax.add_patch(Polygon(polyv, closed=True, facecolor=COL["s2"], alpha=0.20, edgecolor=COL["s2"], lw=1.2, zorder=2))
Mv_P1 = np.array([K.moment_aile_v(y, fbP1, e) for y in yy]) / 1e6
ax.plot(-sc * Mv_P1, yy, color=COL["s3"], lw=1.4, ls=(0, (5, 2)), zorder=3)
# sections
secs = [("1", e + r, "h"), ("2", e, "h"), ("3", e + r, "v"), ("4", e, "v")]
for j, (lab, s, aile) in enumerate(secs):
    if aile == "h":
        ax.plot([s, s], [0, K.t_conge(s, e, r)], color=COL["s1"], lw=2.0, zorder=5)
        ax.text(s + 1.0, 1.5, lab, fontsize=9, fontweight="bold", color=COL["s1"], ha="left", zorder=6)
        ax.plot([s, s], [-sc * R * (Xp - s) / 1e6, 0], color=COL["encre2"], lw=0.8, ls=":", zorder=5)
        ax.text(s + 0.8, -sc * R * (Xp - s) / 1e6 - 1.5, f"M{lab} = {K.fr(ex['M'][j] / 1e6, 3)}", fontsize=7.5,
                color=COL["encre"], ha="left", va="top", zorder=6)
    else:
        ax.plot([0, K.t_conge(s, e, r)], [s, s], color=COL["s2"], lw=2.0, zorder=5)
        ax.text(1.0, s + 1.0, lab, fontsize=9, fontweight="bold", color=COL["s2"], va="bottom", zorder=6)
        ax.plot([-sc * Mv_E(s) / 1e6, 0], [s, s], color=COL["encre2"], lw=0.8, ls=":", zorder=5)
        ax.text(-sc * Mv_E(s) / 1e6 - 1.5, s + (3.0 if lab == "3" else 1.5), f"M{lab} = {K.fr(ex['M'][j] / 1e6, 3)}",
                fontsize=7.5, color=COL["encre"], ha="right", va="center", zorder=6)
# charge
ax.annotate("", xy=(Xp, e), xytext=(Xp, e + 42), arrowprops=dict(arrowstyle="-|>", color=COL["critique"], lw=2.2,
                                                                  mutation_scale=16), zorder=6)
ax.text(Xp + 2, e + 36, f"R = {K.fr(R / 1e3, 2)} kN\n(sur L = {L:.0f} mm)", color=COL["encre"], fontsize=8,
        va="center")
ax.annotate(f"ancrage, Y = {Yp:.0f} mm", xy=(e + 10, Yp), xytext=(e + 22, Yp + 12), fontsize=8, color=COL["encre"],
            arrowprops=dict(arrowstyle="-", color=COL["muet"], lw=0.7))
ax.plot([0], [e / 2], "o", ms=6, color=COL["s2"], zorder=7)
ax.annotate("pivot du classeur\n(y = e/2)", xy=(0, e / 2), xytext=(-62, -14), fontsize=7.5, color=COL["encre2"],
            arrowprops=dict(arrowstyle="-", color=COL["muet"], lw=0.7))
ax.text(-sc * Mv_E(e / 2) / 1e6 - 1.5, e / 2 - 3.5, f"R·X' = {K.fr(ex['Mmax'] / 1e6, 3)}", fontsize=7.5,
        color=COL["encre"], ha="right", va="center")
ax.text(1, -sc * ex["Mmax"] / 1e6 - 1.5, f"R·X' = {K.fr(ex['Mmax'] / 1e6, 3)}", fontsize=7.5,
        color=COL["encre"], ha="left", va="top")
ax.text(-68, 132, "Moments en kN·m (1 kN·m = 12 mm de dessin)\n"
        "bleu : aile horizontale, M(u) = R·(X' − u)\n"
        "orange : aile verticale, formules du classeur\n"
        "vert tireté : nœud filaire (pivot e/2, V sur l'axe)\n"
        "hachures : bloc d'angle (u < e),\n   hors théorie des poutres",
        fontsize=7.5, color=COL["encre2"], va="top")
ax.set_xlim(-72, 135)
ax.set_ylim(-48, 140)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Cornière L120×120×15, cas de contrôle — moments dans les deux ailes (à l'échelle)",
             fontsize=10, color=COL["encre"])
fig.tight_layout()
fA = os.path.join(K.FIGURES, "e01_schema_moments.png")
fig.savefig(fA, facecolor=COL["fond"])
plt.close(fig)

# --- figure B : courbes M, σ_éq, taux
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.6), dpi=150)
for a in axs.flat:
    K.style_axes(a)
# (a) M horizontal
a = axs[0, 0]
uu = np.linspace(0, Xp + 10, 400)
a.axvspan(0, e, color=COL["neutre"], zorder=0)
a.axvspan(e, e + r, color="#e9f0fa", zorder=0)
a.plot(uu[uu >= e], R * np.clip(Xp - uu[uu >= e], 0, None) / 1e6, color=COL["s1"], lw=2)
a.plot(uu[uu <= e], R * (Xp - uu[uu <= e]) / 1e6, color=COL["s1"], lw=1.4, ls=":")
for j, s in ((0, e + r), (1, e)):
    a.plot([s], [ex["M"][j] / 1e6], "o", ms=8, color=COL["s1"], mec=COL["fond"], mew=2)
    a.annotate(f"section {j + 1} : {K.fr(ex['M'][j] / 1e6, 5)}", (s, ex["M"][j] / 1e6), xytext=(8, 6),
               textcoords="offset points", fontsize=8, color=COL["encre"])
a.text(1, 0.15, "bloc\nd'angle", fontsize=7, color=COL["encre2"])
a.text(e + 0.5, 0.15, "congé", fontsize=7, color=COL["encre2"])
a.axvline(Xp, color=COL["muet"], lw=0.8, ls="--")
a.text(Xp + 1, 2.9, "charge R", fontsize=7.5, color=COL["encre2"])
a.set_xlabel("u : distance au nu du mur (mm)", fontsize=8.5, color=COL["encre2"])
a.set_ylabel("M (kN·m, sur L = 300 mm)", fontsize=8.5, color=COL["encre2"])
a.set_title("Aile horizontale : moment (statique exacte)", fontsize=9.5, color=COL["encre"], loc="left")
# (b) M vertical (y en ordonnée)
a = axs[0, 1]
yy = np.linspace(0, Yp + 15, 400)
a.axhspan(0, e, color=COL["neutre"], zorder=0)
a.axhspan(e, e + r, color="#fbece4", zorder=0)
yl = yy[yy >= e]
a.plot([Mv_E(y) / 1e6 for y in yl], yl, color=COL["s2"], lw=2, label="formules du classeur (modèle E)")
for nom, fb, *_ in (variantes[1], variantes[0]):
    lab = "nœud filaire : pivot e/2, V sur l'axe" if nom.startswith("P1") else "pivot à l'arête (y = 0), V sur l'axe"
    colv = COL["s3"] if nom.startswith("P1") else COL["s4"]
    a.plot([K.moment_aile_v(y, fb, e) / 1e6 for y in yl], yl, color=colv, lw=1.6, ls=(0, (5, 2)), label=lab)
a.text(0.05, 4, "bloc d'angle (y < e)", fontsize=7, color=COL["encre2"])
a.text(0.05, e + 4, "congé", fontsize=7, color=COL["encre2"])
for j, s in ((2, e + r), (3, e)):
    a.plot([ex["M"][j] / 1e6], [s], "o", ms=8, color=COL["s2"], mec=COL["fond"], mew=2)
    a.annotate(f"section {j + 1} : {K.fr(ex['M'][j] / 1e6, 5)}", (ex["M"][j] / 1e6, s),
               xytext=(8, 4 if j == 2 else -12), textcoords="offset points", fontsize=8, color=COL["encre"])
a.axhline(Yp, color=COL["muet"], lw=0.8, ls="--")
a.text(0.05, Yp + 1.5, "ancrage", fontsize=7.5, color=COL["encre2"])
a.set_xlim(-0.05, 3.6)
a.set_ylabel("y : hauteur depuis le dessous (mm)", fontsize=8.5, color=COL["encre2"])
a.set_xlabel("M (kN·m, sur L = 300 mm)", fontsize=8.5, color=COL["encre2"])
a.set_title("Aile verticale : moment selon l'équilibre retenu", fontsize=9.5, color=COL["encre"], loc="left")
a.legend(fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))
# (c) taux horizontal
a = axs[1, 0]
a.axvspan(e, e + r, color="#e9f0fa", zorder=0)
a.plot(us, seqh / ex["sadm"] * 100, color=COL["s1"], lw=2)
a.plot([us[i]], [seqh[i] / ex["sadm"] * 100], "o", ms=8, color=COL["s1"], mec=COL["fond"], mew=2)
a.annotate(f"maximum {K.fr(seqh[i] / ex['sadm'] * 100, 2)} % à u = {K.fr(us[i], 1)} mm",
           (us[i], seqh[i] / ex["sadm"] * 100), xytext=(-20, 14), textcoords="offset points", fontsize=8,
           color=COL["encre"])
for j, s in ((0, e + r), (1, e)):
    a.plot([s], [ex["taux"][j] * 100], "s", ms=6, color=COL["encre2"])
    a.annotate(f"section {j + 1} : {K.fr(ex['taux'][j] * 100, 2)} %", (s, ex["taux"][j] * 100), xytext=(8, -4),
               textcoords="offset points", fontsize=7.5, color=COL["encre2"])
a.axhline(100, color=COL["critique"], lw=0.9, ls="--")
a.set_xlim(e - 2, Xp + 5)
a.set_ylim(0, 110)
a.set_xlabel("u (mm)", fontsize=8.5, color=COL["encre2"])
a.set_ylabel("taux σ_éq / σ_adm (%)", fontsize=8.5, color=COL["encre2"])
a.set_title("Aile horizontale : taux le long de l'aile", fontsize=9.5, color=COL["encre"], loc="left")
# (d) taux vertical (y en ordonnée)
a = axs[1, 1]
a.axhspan(e, e + r, color="#fbece4", zorder=0)
a.plot(seqvE / ex["sadm"] * 100, ysE, color=COL["s2"], lw=2, label="classeur (σ = M/W, τ = R/A)")
a.plot(seqvC / ex["sadm"] * 100, ysC, color=COL["s3"], lw=1.6, ls=(0, (5, 2)),
       label="même M + effort normal R + tranchant H")
a.plot([seqvE[k] / ex["sadm"] * 100], [ysE[k]], "o", ms=8, color=COL["s2"], mec=COL["fond"], mew=2)
a.annotate(f"maximum {K.fr(seqvE[k] / ex['sadm'] * 100, 2)} % à y = {K.fr(ysE[k], 1)} mm",
           (seqvE[k] / ex["sadm"] * 100, ysE[k]), xytext=(-175, -4), textcoords="offset points", fontsize=8,
           color=COL["encre"])
for j, s in ((2, e + r), (3, e)):
    a.plot([ex["taux"][j] * 100], [s], "s", ms=6, color=COL["encre2"])
    a.annotate(f"section {j + 1} : {K.fr(ex['taux'][j] * 100, 2)} %", (ex["taux"][j] * 100, s),
               xytext=(-95 if j == 2 else 8, 6 if j == 2 else -14), textcoords="offset points", fontsize=7.5,
               color=COL["encre2"])
a.axvline(100, color=COL["critique"], lw=0.9, ls="--")
a.set_ylim(e - 6, Yp + 5)
a.set_xlim(0, 110)
a.set_ylabel("y (mm)", fontsize=8.5, color=COL["encre2"])
a.set_xlabel("taux σ_éq / σ_adm (%)", fontsize=8.5, color=COL["encre2"])
a.set_title("Aile verticale : taux le long de l'aile", fontsize=9.5, color=COL["encre"], loc="left")
a.legend(fontsize=7.5, frameon=False, loc="upper right")
fig.suptitle("Diagrammes continus — L120×120×15, R = 42,76 kN sur 300 mm, X = 75, Y = 85 (σ_adm = 355/1,5)",
             fontsize=10.5, color=COL["encre"])
fig.tight_layout()
fB = os.path.join(K.FIGURES, "e01_diagrammes.png")
fig.savefig(fB, facecolor=COL["fond"])
plt.close(fig)

# --- figure C : N et V
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
for a in axs:
    K.style_axes(a)
a = axs[0]
uu = np.linspace(e, Xp + 15, 400)
a.plot(uu, [R / 1e3 if u < Xp else 0 for u in uu], color=COL["s1"], lw=2, label="V (tranchant)")
a.plot(uu, [0 for u in uu], color=COL["s2"], lw=1.6, ls=(0, (5, 2)), label="N (normal)")
a.set_xlabel("u (mm)", fontsize=8.5, color=COL["encre2"])
a.set_ylabel("kN (sur L = 300 mm)", fontsize=8.5, color=COL["encre2"])
a.set_title("Aile horizontale", fontsize=9.5, color=COL["encre"], loc="left")
a.legend(fontsize=7.5, frameon=False)
a = axs[1]
yy = np.linspace(e, Yp + 15, 400)
a.plot([H_E / 1e3 if y < Yp else 0 for y in yy], yy, color=COL["s1"], lw=2, label="V = H (tranchant, modèle E)")
a.plot([R / 1e3 if y < Yp else 0 for y in yy], yy, color=COL["s2"], lw=1.6, ls=(0, (5, 2)),
       label="N = R (traction)")
a.set_ylabel("y (mm)", fontsize=8.5, color=COL["encre2"])
a.set_xlabel("kN (sur L = 300 mm)", fontsize=8.5, color=COL["encre2"])
a.set_title("Aile verticale", fontsize=9.5, color=COL["encre"], loc="left")
a.legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout()
fC = os.path.join(K.FIGURES, "e01_efforts_NV.png")
fig.savefig(fC, facecolor=COL["fond"])
plt.close(fig)

P("")
P("## Figures")
for f in (fA, fB, fC):
    P(f)
K.ecrire(os.path.join(K.ICI, "e01_sortie.txt"), out)
print("\n".join(out))
