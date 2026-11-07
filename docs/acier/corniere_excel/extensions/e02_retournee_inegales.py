# -*- coding: utf-8 -*-
"""E02 — Cornière RETOURNÉE (configuration B) et ailes INÉGALES.

Configuration B : aile horizontale en HAUT, aile verticale qui descend le long du
mur depuis l'angle ; Y_B = distance de la face supérieure (dos de l'aile
horizontale) à l'axe de l'ancrage. Le contact du mur est au BOUT LIBRE de
l'aile verticale (y = 0), sous l'ancrage.

Corps libre (frottement nul, V = R à l'ancrage, ligne d'action uV) :
  moments autour du bout comprimé : H·(y_a − c) = R·(X' − uV),  y_a = H_aile − Y_B
  aile verticale, entre le bout et l'ancrage : M = H·(y − c), V = H, N = 0
  entre l'ancrage et l'angle             : M = R·(X' − uV) constant, V = 0, N = −R (compression)

Lancer : /usr/bin/python3.12 -I e02_retournee_inegales.py
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ext_commun as K  # noqa: E402

out = []
P = out.append
COL = K.COUL


def verif_A(B, Hh, e, r, X, Y, R=42760.0, L=300.0, fy=355.0, gM=1.5, uV=None, yP=None, d0=18.0, n_anc=3,
            jeu_anc=10.0):
    """Configuration A : méthode du classeur (uV, yP = None) ou corps libre (uV, yP donnés).
    Renvoie M1..M4 (N·mm), σ nominales (M/W [+ N/A dans l'aile verticale pour le corps libre]),
    taux σ_éq (τ = R/A comme le classeur), traction totale d'ancrage."""
    sadm = fy / gM
    Xe, Xer = X - e - r, X - e
    if uV is None:
        Ye, Yer, YH = Y - e - r, Y - e, Y - e / 2
        M = [R * Xe, R * Xer, R * X * Ye / YH, R * X * Yer / YH]
        Nv = 0.0
        Htot = R * X / (Y - jeu_anc)          # C82 × n
    else:
        Hc = R * (X - uV) / (Y - yP)
        M = [R * Xe, R * Xer, Hc * (Y - e - r), Hc * (Y - e)]
        Nv = R
        Htot = Hc
    t = [e, e + r, e, e + r]
    sig = [m / (L * ti ** 2 / 6) + (Nv / (L * ti) if k >= 2 else 0.0) for k, (m, ti) in enumerate(zip(M, t))]
    tau = [R / (L * ti) for ti in t]
    taux = [math.sqrt(s ** 2 + 3 * ta ** 2) / sadm for s, ta in zip(sig, tau)]
    return dict(M=M, sig=sig, taux=taux, H=Htot, ok_geo=(X <= B and Y > e + r))


def verif_B(B, Hh, e, r, X, YB, R=42760.0, L=300.0, fy=355.0, gM=1.5, uV=0.0, c=0.0, d0=18.0, n_anc=3):
    """Configuration B, corps libre. uV = 0 : convention du classeur (bras R·X depuis le nu du mur) ;
    uV = e/2 : cisaillement sur l'axe de l'aile."""
    sadm = fy / gM
    ya = Hh - YB                       # hauteur de l'ancrage au-dessus du bout libre
    Hc = R * (X - uV) / (ya - c)
    Mc = R * (X - uV)                  # moment constant entre l'ancrage et l'angle
    M = [R * (X - e - r), R * (X - e), Mc, Mc]
    t = [e, e + r, e, e + r]
    sig = [m / (L * ti ** 2 / 6) + (R / (L * ti) if k >= 2 else 0.0) for k, (m, ti) in enumerate(zip(M, t))]
    tau = [R / (L * ti) for ti in t]
    taux = [math.sqrt(s ** 2 + 3 * ta ** 2) / sadm for s, ta in zip(sig, tau)]
    # section nette au droit de l'ancrage (M = Mc, trous d0)
    Wn = (L - n_anc * d0) * e ** 2 / 6
    s_net = Mc / Wn + R / ((L - n_anc * d0) * e)
    return dict(M=M, sig=sig, taux=taux, H=Hc, ya=ya, s_net=s_net, taux_net=s_net / sadm,
                C_lin=Hc / L, ok_geo=(ya - c > 0 and YB > e + r))


c = K.CAS
e, r, X, Y, R, L = c["e"], c["r"], c["X"], c["Y"], c["R"], c["L"]
P("# E02 — cornière retournée (B) et ailes inégales")
P("")
P("## 1. Cas de contrôle L120×120×15 : configuration A face à B")
A_xl = verif_A(120, 120, e, r, X, Y)
A_P0 = verif_A(120, 120, e, r, X, Y, uV=e / 2, yP=0.0)
P("| | M1 | M2 | M3 | M4 (kN·m) | taux 1 | taux 3 | traction totale ancrages (kN) |")
P("|---|---|---|---|---|---|---|---|")


def ligne(nom, d):
    P(f"| {nom} | " + " | ".join(f"{m / 1e6:.4f}" for m in d["M"]) + f" | {d['taux'][0] * 100:.2f} % | "
      f"{d['taux'][2] * 100:.2f} % | {d['H'] / 1e3:.2f} |")


ligne("A, méthode du classeur", A_xl)
ligne("A, corps libre (pivot y = 0, V sur l'axe, + N)", A_P0)
res = {}
for YB in (85.0, 60.0, 45.0):
    for uV, lab in ((0.0, "convention classeur (uV = 0)"), (e / 2, "V sur l'axe (uV = e/2)")):
        d = verif_B(120, 120, e, r, X, YB, uV=uV)
        res[(YB, uV)] = d
        ligne(f"B, ancrage à {YB:.0f} mm sous le dessus (y_a = {d['ya']:.0f}), {lab}", d)
P("")
for YB in (85.0, 45.0):
    d = res[(YB, e / 2)]
    P(f"B, Y_B = {YB:.0f} : section nette au droit de l'ancrage (3 trous Ø18 sur 300 mm) σ = {d['s_net']:.1f} N/mm² "
      f"→ {d['taux_net'] * 100:.1f} % ; compression du mur au bout libre C = H = {d['H'] / 1e3:.2f} kN, "
      f"soit {d['C_lin']:.1f} N/mm de cornière")
dB = res[(85.0, e / 2)]
P(f"Rapport traction B (Y_B = 85, V sur l'axe) / A (corps libre pivot 0) = {dB['H'] / A_P0['H']:.3f} ; "
  f"B / A classeur = {dB['H'] / A_xl['H']:.3f}")
P(f"Rapport moment section 3, B / A classeur = {dB['M'][2] / A_xl['M'][2]:.3f} (V sur l'axe) ; "
  f"{res[(85.0, 0.0)]['M'][2] / A_xl['M'][2]:.3f} (convention classeur)")
# cohérence avec le maillage (e05)
try:
    ef = json.load(open(os.path.join(K.ICI, "e05_resultats.json")))
    for nom, YB in (("B-Y85", 85.0), ("B-Y45", 45.0)):
        P(f"maillage {nom} : H = {ef[nom]['H'] / 1e3:.3f} kN ; corps libre (uV = e/2, c = 0) : "
          f"{res[(YB, e / 2)]['H'] / 1e3:.3f} kN ; écart {ef[nom]['H'] / res[(YB, e / 2)]['H'] * 100 - 100:+.2f} %")
except FileNotFoundError:
    P("(lancer e05 d'abord pour la comparaison au maillage)")

# meilleure position de l'ancrage en B
P("")
P("## 2. Configuration B : position de l'ancrage")
P("Le bras de levier est la distance de l'ancrage au BOUT LIBRE (contact) : l'ancrage doit être le plus PRÈS DE L'ANGLE "
  "possible, à l'inverse de A où il doit être le plus haut possible (loin de l'angle).")
for YB in (45.0, 50.0, 60.0, 70.0, 85.0, 95.0):
    d = verif_B(120, 120, e, r, X, YB, uV=e / 2)
    P(f"  Y_B = {YB:5.1f} mm (rondelle Ø30 : bord haut à {YB - 15:.0f} mm du dessus ; pied du congé à {e + r:.0f}) : "
      f"H = {d['H'] / 1e3:6.2f} kN ; taux section 3 = {d['taux'][2] * 100:6.2f} % (indépendant de Y_B)")

# ---------------------------------------------------------------- ailes inégales
P("")
P("## 3. Ailes inégales — L150×100×12 (catalogue du site : r1 = 12 mm), configuration A")
dims = dict(e=12.0, r=12.0)
cas_ineg = [
    ("grande aile horizontale (150 × 100)", 150.0, 100.0),
    ("grande aile verticale (100 × 150)", 100.0, 150.0),
]
tab = []
for nom, Bh, Hv in cas_ineg:
    for Xv in (75.0,):
        Yv = Hv - 35.0                     # même distance de l'ancrage au bord libre que le cas de contrôle
        dx = verif_A(Bh, Hv, dims["e"], dims["r"], Xv, Yv)
        dp = verif_A(Bh, Hv, dims["e"], dims["r"], Xv, Yv, uV=dims["e"] / 2, yP=0.0)
        tab.append((nom, Xv, Yv, dx, dp))
        P(f"{nom} : X = {Xv:.0f}, Y = {Yv:.0f} (bord libre à 35 mm) → classeur : taux 1 = {dx['taux'][0] * 100:.1f} %, "
          f"taux 3 = {dx['taux'][2] * 100:.1f} %, max = {max(dx['taux']) * 100:.1f} %, traction totale "
          f"{dx['H'] / 1e3:.2f} kN | corps libre (pivot 0) : taux 3 = {dp['taux'][2] * 100:.1f} %, H = {dp['H'] / 1e3:.2f} kN")
P("Avec le rayon du classeur pour cette cornière (6 mm, CORNIERES!E6) au lieu de r1 = 12 :")
for nom, Bh, Hv in cas_ineg:
    Yv = Hv - 35.0
    d6 = verif_A(Bh, Hv, 12.0, 6.0, 75.0, Yv)
    d12 = verif_A(Bh, Hv, 12.0, 12.0, 75.0, Yv)
    P(f"  {nom} : taux 1 = {d6['taux'][0] * 100:.1f} % (r = 6) contre {d12['taux'][0] * 100:.1f} % (r = 12) ; "
      f"taux 3 = {d6['taux'][2] * 100:.1f} % contre {d12['taux'][2] * 100:.1f} %")

# ---------------------------------------------------------------- catalogue du classeur
P("")
P("## 4. Catalogue CORNIERES du classeur : doublons de nom et rayons")
lignes = {}
feuille = None
for l_ in open(K.RELEVE, encoding="utf-8"):
    m = re.match(r"===== FEUILLE (\S+)", l_)
    if m:
        feuille = m.group(1)
        continue
    if feuille != "CORNIERES":
        continue
    m = re.match(r"([A-G])(\d+)\tVALEUR (.+)$", l_.rstrip("\n"))
    if m:
        col, row, v = m.group(1), int(m.group(2)), m.group(3)
        lignes.setdefault(row, {})[col] = v.strip("'")
cat = []
for row in sorted(lignes):
    if row == 1:
        continue
    d = lignes[row]
    if "B" not in d:
        continue
    cat.append(dict(ligne=row, nom=d["A"], larg=float(d["B"]), haut=float(d["C"]), e=float(d["D"]), r=float(d["E"])))
noms = {}
for x in cat:
    noms.setdefault(x["nom"], []).append(x)
doublons = {k: v for k, v in noms.items() if len(v) > 1}
P(f"{len(cat)} lignes de cornières ; {len(doublons)} noms en double (VLOOKUP exact, P8!C4:C7, ne lit que la première) :")
for k, v in doublons.items():
    P(f"  {k} : ligne {v[0]['ligne']} ({v[0]['larg']:.0f} × {v[0]['haut']:.0f}) lue ; "
      f"ligne {v[1]['ligne']} ({v[1]['larg']:.0f} × {v[1]['haut']:.0f}) inaccessible")
site = json.load(open("/home/user/integ/acier/donnees/profiles.json"))["profils"]
idx = {}
for p_ in site:
    if p_["famille"] in ("L", "LI"):
        dd = p_["dims"]
        idx[(dd["h"], dd["b"], dd["t"])] = dd
        idx[(dd["b"], dd["h"], dd["t"])] = dd
ecarts = []
for x in cat:
    k = (x["larg"], x["haut"], x["e"])
    if k in idx:
        dd = idx[k]
        if abs(dd["r1"] - x["r"]) > 1e-9:
            ecarts.append((x["nom"], x["ligne"], x["r"], dd["r1"], dd["r2"]))
P(f"Rayon du classeur ≠ r1 du catalogue du site (EN 10056-1) pour {len(ecarts)} lignes présentes dans les deux :")
for nm, li, rx, r1, r2 in ecarts:
    P(f"  {nm} (ligne {li}) : classeur r = {rx:g} ; site r1 = {r1:g}, r2 = {r2:g}" + ("  ← égal à r2 (rayon de bout d'aile)" if abs(rx - r2) < 1e-9 else ""))
absents = [x["nom"] for x in cat if (x["larg"], x["haut"], x["e"]) not in idx]
P(f"Lignes du classeur absentes du catalogue du site : {len(absents)} ({', '.join(sorted(set(absents)))})")

# ---------------------------------------------------------------- figure : A et B côte à côte
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=150)
for a in axs:
    K.style_axes(a)
ys = np.linspace(0, 120, 600)
# A : y depuis le dessous ; aile verticale de e à Y
H_A = A_P0["H"]
MA = [H_A * (Y - y) / 1e6 if e <= y <= Y else np.nan for y in ys]
MA_xl = [R * X * (Y - y) / (Y - e / 2) / 1e6 if e <= y <= Y else np.nan for y in ys]
a = axs[0]
a.plot(MA_xl, ys, color=COL["s2"], lw=2, label="formules du classeur")
a.plot(MA, ys, color=COL["s1"], lw=1.6, ls=(0, (5, 2)), label="corps libre, pivot à l'arête")
a.axhspan(0, e, color=COL["neutre"])
a.axhline(Y, color=COL["muet"], lw=0.8, ls="--")
a.text(0.05, Y + 2, "ancrage", fontsize=7.5, color=COL["encre2"])
a.text(0.05, 3, "aile horizontale (en bas)", fontsize=7.5, color=COL["encre2"])
a.set_title("A : aile horizontale en bas — M nul à l'ancrage", fontsize=9.5, color=COL["encre"], loc="left")
a.set_xlabel("M dans l'aile verticale (kN·m sur 300 mm)", fontsize=8.5, color=COL["encre2"])
a.set_ylabel("hauteur (mm)", fontsize=8.5, color=COL["encre2"])
a.set_xlim(0, 3.6)
a.legend(fontsize=7.5, frameon=False, loc="center right")
a = axs[1]
for YB, colr in ((85.0, COL["s1"]), (45.0, COL["s3"])):
    d = res[(YB, e / 2)]
    ya = d["ya"]
    Hc = d["H"]
    MB = [(Hc * y if y <= ya else R * (X - e / 2)) / 1e6 if y <= 120 - e else np.nan for y in ys]
    a.plot(MB, ys, color=colr, lw=2, label=f"ancrage à {YB:.0f} mm sous le dessus : H = {K.fr(Hc / 1e3, 1)} kN")
    a.plot([0], [ya], marker=">", color=colr, ms=7)
a.axhspan(120 - e, 120, color=COL["neutre"])
a.text(0.05, 120 - e + 3, "aile horizontale (en haut)", fontsize=7.5, color=COL["encre2"])
a.text(0.05, 2, "bout libre : contact du mur", fontsize=7.5, color=COL["encre2"])
a.set_title("B : aile horizontale en haut — M maximal de l'ancrage à l'angle", fontsize=9.5, color=COL["encre"], loc="left")
a.set_xlabel("M dans l'aile verticale (kN·m sur 300 mm)", fontsize=8.5, color=COL["encre2"])
a.set_xlim(0, 3.6)
a.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.suptitle("Même cornière, même charge (42,76 kN à 75 mm) : le retournement change le schéma de l'aile verticale",
             fontsize=10, color=COL["encre"])
fig.tight_layout()
f1 = os.path.join(K.FIGURES, "e02_A_contre_B.png")
fig.savefig(f1, facecolor=COL["fond"])
plt.close(fig)
P("")
P("## Figures")
P(f1)
K.ecrire(os.path.join(K.ICI, "e02_sortie.txt"), out)
print("\n".join(out))
