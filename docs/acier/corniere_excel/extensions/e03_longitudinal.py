# -*- coding: utf-8 -*-
"""E03 — Cornière LONGUE : 2,00 m, trois ancrages, charge localisée.

Modèle longitudinal (poutre à section indéformable sur appuis élastiques) :
  inconnues le long de z : U (translation horizontale du centre de cisaillement S),
  V (translation verticale de S), φ (rotation autour de z, positive dans le sens
  trigonométrique du repère u→y de la coupe) ;
  énergie : ½E[I_yy U''² + 2 I_uy U''V'' + I_uu V''²] + ½ G·It·φ'²  (gauchissement
  d'une cornière négligé, It du catalogue) ;
  appuis : à chaque ancrage, un tirant horizontal à la hauteur Y (raideur k_N en
  série avec la flexion locale de l'aile verticale) et un appui vertical k_V ;
  le long du mur, un contact UNILATÉRAL continu au pivot (u = 0, y = yP).
  charge : q(z) verticale à u = X' → force −q en S et couple −q·(X' − u_S).

Section L120×120×15 : propriétés du catalogue du site (acier/donnees/profiles.json),
signe du produit d'inertie recalculé sur le contour dans le repère de la coupe.

Lancer : /usr/bin/python3.12 -I e03_longitudinal.py
"""
import json
import math
import os
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
c = K.CAS
e, r, Xc, Yc, R = c["e"], c["r"], c["X"], c["Y"], c["R"]

# ---------------------------------------------------------------- section
prof = [p_ for p_ in json.load(open("/home/user/integ/acier/donnees/profiles.json"))["profils"]
        if p_["nom"] == "L 120x120x15"][0]


def contour():
    B = H = 120.0
    pts = [(0, 0), (B, 0), (B, e), (e + r, e)]
    for a in np.linspace(-math.pi / 2, -math.pi, 41)[1:-1]:
        pts.append((e + r + r * math.cos(a), e + r + r * math.sin(a)))
    pts += [(e, e + r), (e, H), (0, H)]
    return np.array(pts, float)


def proprietes(pts):
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    cx = ((x + x1) * cr).sum() / (6 * A)
    cy = ((y + y1) * cr).sum() / (6 * A)
    Ixx = ((y ** 2 + y * y1 + y1 ** 2) * cr).sum() / 12 - A * cy ** 2      # ∫(y − yG)²
    Iyy = ((x ** 2 + x * x1 + x1 ** 2) * cr).sum() / 12 - A * cx ** 2      # ∫(u − uG)²
    Ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / 24 - A * cx * cy
    return A, cx, cy, Ixx, Iyy, Ixy


A_p, uG, yG, Iuu_p, Iyy_p, Iuy_p = proprietes(contour())
P("# E03 — cornière longue (2,00 m, 3 ancrages)")
P("")
P("## 1. Section L120×120×15")
P(f"contour recalculé (sans rayons de bout d'aile) : A = {A_p / 100:.3f} cm², centre de gravité ({uG:.2f} ; {yG:.2f}) mm, "
  f"∫(y−yG)² = {Iuu_p / 1e4:.2f} cm⁴, ∫(u−uG)² = {Iyy_p / 1e4:.2f}, ∫(u−uG)(y−yG) = {Iuy_p / 1e4:.2f} cm⁴")
P(f"catalogue du site : A = {prof['A']} cm², Iy = Iz = {prof['Iy']} cm⁴, Iyz = {prof['Iyz']} cm⁴, It = {prof['It']} cm⁴, "
  f"yc = zc = {prof['yc']} mm")
E_, G_ = 210000.0, 81000.0
Iuu = prof["Iy"] * 1e4
Iyy = prof["Iz"] * 1e4
Iuy = math.copysign(abs(prof["Iyz"]) * 1e4, Iuy_p)
It = prof["It"] * 1e4
uS = yS = e / 2
P(f"retenu : I_uu = I_yy = {Iuu / 1e4:.1f} cm⁴, I_uy = {Iuy / 1e4:.1f} cm⁴ (signe du contour), It = {It / 1e4:.2f} cm⁴, "
  f"G·It = {G_ * It / 1e9:.3f} kN·m², E·I = {E_ * Iuu / 1e9:.1f} kN·m², centre de cisaillement S = ({uS} ; {yS}) mm")


# ---------------------------------------------------------------- modèle
def modele(Ltot=2000.0, ancrages=(200.0, 1000.0, 1800.0), charges=((400.0, 300.0, 42760.0),), Xp=75.0, Y=85.0,
           yP=0.0, kN=30e3, kV=10e3, k_pl=None, kw=1e5, h=5.0, rigide=1.0, H_aile=120.0, levier=True):
    """charges : (position du centre, largeur, effort total N) ; largeur 0 = ponctuelle.
    kN, kV : N/mm par ancrage ; k_pl : raideur locale de l'aile (N/mm, en série avec kN) ; kw : N/mm par mm."""
    n_el = int(round(Ltot / h))
    zs = np.linspace(0, Ltot, n_el + 1)
    nn = len(zs)
    ndof = 5 * nn                       # U, U', V, V', φ
    Kg = np.zeros((ndof, ndof))
    pat = np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                    [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]]) * E_ / h ** 3 * rigide
    for i in range(n_el):
        a, b = 5 * i, 5 * (i + 1)
        iu = [a, a + 1, b, b + 1]
        iv = [a + 2, a + 3, b + 2, b + 3]
        Kg[np.ix_(iu, iu)] += Iyy * pat
        Kg[np.ix_(iv, iv)] += Iuu * pat
        Kg[np.ix_(iu, iv)] += Iuy * pat
        Kg[np.ix_(iv, iu)] += Iuy * pat
        ip = [a + 4, b + 4]
        Kg[np.ix_(ip, ip)] += G_ * It / h * np.array([[1, -1], [-1, 1]]) * rigide
    F = np.zeros(ndof)
    for zc, larg, Ptot in charges:
        if larg <= 0:
            k = int(np.argmin(abs(zs - zc)))
            w = np.zeros(nn)
            w[k] = 1.0
        else:
            w = np.clip(np.minimum(zs + h / 2, zc + larg / 2) - np.maximum(zs - h / 2, zc - larg / 2), 0, None)
            w /= w.sum()
        F[2::5] += -Ptot * w
        F[4::5] += -Ptot * w * (Xp - uS)
    kNe = kN if k_pl is None else 1.0 / (1.0 / kN + 1.0 / k_pl)
    idx_a = [int(np.argmin(abs(zs - za))) for za in ancrages]
    # vecteurs de liaison (déplacement d'un point en fonction des inconnues du nœud)
    def g_u(yy):     # déplacement horizontal du point (·, yy) : U − φ·(yy − yS)
        v = np.zeros(5)
        v[0] = 1.0
        v[4] = -(yy - yS)
        return v
    gv = np.zeros(5)
    gv[2] = 1.0       # vertical au point (uS, ·) : V
    wz = np.full(nn, h)
    wz[0] = wz[-1] = h / 2
    actif_bas = np.ones(nn, dtype=bool)
    actif_haut = np.zeros(nn, dtype=bool)
    actif_anc = np.ones(len(idx_a), dtype=bool)
    for it in range(50):
        Kt = Kg.copy()
        for j, k in enumerate(idx_a):
            sl = slice(5 * k, 5 * k + 5)
            if actif_anc[j]:
                Kt[sl, sl] += kNe * np.outer(g_u(Y), g_u(Y))
            Kt[sl, sl] += kV * np.outer(gv, gv)
        gb, gh = g_u(yP), g_u(H_aile)
        for k in range(nn):
            sl = slice(5 * k, 5 * k + 5)
            if actif_bas[k]:
                Kt[sl, sl] += kw * wz[k] * np.outer(gb, gb)
            if actif_haut[k]:
                Kt[sl, sl] += kw * wz[k] * np.outer(gh, gh)
        Ug = np.linalg.solve(Kt, F)
        Un = Ug.reshape(nn, 5)
        db = Un @ gb
        dh = Un @ gh
        da = np.array([Un[k] @ g_u(Y) for k in idx_a])
        nb, nh, na = db < 0, (dh < 0) & levier, da > 0   # contact = enfoncement ; ancrage = décollement
        if np.array_equal(nb, actif_bas) and np.array_equal(nh, actif_haut) and np.array_equal(na, actif_anc):
            break
        actif_bas, actif_haut, actif_anc = nb, nh, na
    N_anc = np.where(actif_anc, kNe * da, 0.0)
    V_anc = np.array([-kV * Un[k] @ gv for k in idx_a])
    C_bas = np.where(actif_bas, -kw * wz * db, 0.0)
    C_haut = np.where(actif_haut, -kw * wz * dh, 0.0)
    # efforts internes : couple de torsion T = G·It·φ', moments M_u = −E(I_yy U'' + I_uy V'') ...
    phi = Un[:, 4]
    Tz = G_ * It * np.gradient(phi, zs) * rigide
    # courbures par différences (suffisant pour l'ordre de grandeur) ; contraintes σz aux fibres extrêmes
    U2 = np.gradient(Un[:, 1], zs)
    V2 = np.gradient(Un[:, 3], zs)
    fibres = [(0, 0), (120, 0), (0, 120), (120, e), (e, 120)]
    sz = np.max([np.abs(E_ * rigide * ((uf - uG) * U2 + (yf - yG) * V2)) for uf, yf in fibres], axis=0)
    # moment de flexion « vertical » M = E(I_uu V'' + I_uy U'')
    Mv = E_ * rigide * (Iuu * V2 + Iuy * U2)
    return dict(z=zs, U=Un, N=N_anc, V=V_anc, C_bas=C_bas, C_haut=C_haut, T=Tz, sz=sz, Mv=Mv, it=it + 1,
                ancrages=list(ancrages), kNe=kNe, idx=idx_a)


# ---------------------------------------------------------------- raideurs
Y = Yc
yP = 0.0
z_lev = Y - yP
m_tee = (Y - e) - 0.8 * r
e_bord = 120.0 - Y
leff = min(2 * math.pi * m_tee, 4 * m_tee + 1.25 * e_bord)
k_pl_cons = 3 * E_ * (leff * e ** 3 / 12) / z_lev ** 3
P("")
P("## 2. Raideurs")
P(f"tronçon en T : m = (Y − e) − 0,8·r = {m_tee:.1f} mm ; 2πm = {2 * math.pi * m_tee:.1f} ; 4m + 1,25·e_bord = "
  f"{4 * m_tee + 1.25 * e_bord:.1f} → l_eff = {leff:.1f} mm")
P(f"flexion locale de l'aile verticale (console de {z_lev:.0f} mm, largeur l_eff) : k_pl = 3EI/z³ = {k_pl_cons / 1e3:.1f} kN/mm")
k_theta = lambda kNe: kNe * z_lev ** 2
P(f"G·It / 800 mm = {G_ * It / 800 / 1e6:.2f} kN·m/rad ; G·It / 200 mm = {G_ * It / 200 / 1e6:.2f} kN·m/rad")
for kN_ in (10e3, 30e3, 100e3):
    kNe = 1 / (1 / kN_ + 1 / k_pl_cons)
    P(f"ancrage {kN_ / 1e3:.0f} kN/mm + aile → {kNe / 1e3:.1f} kN/mm ; k_θ = {k_theta(kNe) / 1e6:.1f} kN·m/rad ; "
      f"rapport k_θ·s/(G·It) (s = 800) = {k_theta(kNe) * 800 / (G_ * It):.2f}")

# ---------------------------------------------------------------- cas
Xp = Xc
Rtot = R
H_tot = R * (Xp - e / 2) / (Y - yP)        # traction totale d'une coupe (corps libre pivot 0, V sur l'axe)
cas = [
    ("rigides", dict(kN=1e9, kV=1e9, k_pl=None, levier=False)),
    ("rigides, avec effet de levier", dict(kN=1e9, kV=1e9, k_pl=None, levier=True)),
    ("raides (100/30 kN/mm) + aile", dict(kN=100e3, kV=30e3, k_pl=k_pl_cons)),
    ("souples (10/5 kN/mm) + aile", dict(kN=10e3, kV=5e3, k_pl=k_pl_cons)),
    ("cornière 1000 fois plus raide, ancrages souples", dict(kN=10e3, kV=5e3, k_pl=k_pl_cons, rigide=1e3)),
]
P("")
P("## 3. Répartition entre les trois ancrages (z = 200, 1000, 1800 mm) — charge 42,76 kN sur 300 mm centrée à z = 400")
P(f"Référence « R/n » : V = {R / 3e3:.2f} kN et N = H/3 = {H_tot / 3e3:.2f} kN par ancrage "
  f"(H = R·(X − e/2)/Y = {H_tot / 1e3:.2f} kN, corps libre de la coupe, pivot à l'arête) ; "
  f"le classeur donnerait N = V = {R * Xp / (Y - 10) / 3e3:.2f} kN.")
P("| ancrages | N1 | N2 | N3 (kN) | V1 | V2 | V3 (kN) | N1/(H/3) | V1/(R/3) | couple max (kN·m) | τ torsion max (N/mm²) | σz max (N/mm²) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
res = {}
for nom, kw_ in cas:
    for forme, ch in (("répartie 300 mm", ((400.0, 300.0, R),)), ("ponctuelle", ((400.0, 0.0, R),))):
        m = modele(charges=ch, Xp=Xp, Y=Y, yP=yP, **kw_)
        tau = np.abs(m["T"]).max() * e / It
        res[(nom, forme)] = m
        if forme == "répartie 300 mm" or nom.startswith("rigides"):
            P(f"| {nom} — {forme} | " + " | ".join(f"{x / 1e3:.2f}" for x in m["N"]) + " | "
              + " | ".join(f"{x / 1e3:.2f}" for x in m["V"]) + f" | {m['N'][0] / (H_tot / 3):.2f} | "
              f"{m['V'][0] / (R / 3):.2f} | {np.abs(m['T']).max() / 1e6:.3f} | {tau:.1f} | {m['sz'][5:-5].max():.1f} |")
for (nom, forme), m in res.items():
    if forme == "répartie 300 mm":
        P(f"contrôle d'équilibre {nom} : ΣV = {m['V'].sum() / 1e3:.3f} kN ; ΣN − ΣC = "
          f"{(m['N'].sum() - m['C_bas'].sum() - m['C_haut'].sum()) / 1e3:.4f} kN ; contact en tête (levier) "
          f"{m['C_haut'].sum() / 1e3:.2f} kN ; itérations de contact {m['it']}")

# ---------------------------------------------------------------- conséquences locales
P("")
P("## 4. Conséquences locales (ELS, méthode des contraintes admissibles, σ_adm = 236,7)")
sadm = 355 / 1.5
for nom in ("rigides", "rigides, avec effet de levier", "raides (100/30 kN/mm) + aile", "souples (10/5 kN/mm) + aile"):
    m = res[(nom, "répartie 300 mm")]
    N1 = m["N"].max()
    s3 = 6 * N1 * (Y - e - r) / (leff * e ** 2) + m["V"].max() / (leff * e)
    tau = np.abs(m["T"]).max() * e / It
    P(f"{nom} : ancrage le plus chargé N = {N1 / 1e3:.2f} kN → aile verticale, section 3 sur l_eff = {leff:.0f} mm : "
      f"σ ≈ {s3:.0f} N/mm² ({s3 / sadm * 100:.0f} %) ; torsion entre charge et ancrage : τ = T·e/It = {tau:.0f} N/mm² "
      f"({tau / (sadm / math.sqrt(3)) * 100:.0f} % de τ_adm = {sadm / math.sqrt(3):.1f})")
P(f"Pour mémoire, cas de contrôle (3 ancrages sur 300 mm, charge au droit) : section 3 = 88,86 % (classeur).")

# variante : ancrage au droit de la charge
P("")
P("## 5. Variante : ancrages à 400, 1000, 1600 mm (un ancrage au droit de la charge)")
for nom, kw_ in [cas[0]] + cas[2:4]:
    m = modele(ancrages=(400.0, 1000.0, 1600.0), charges=((400.0, 300.0, R),), Xp=Xp, Y=Y, yP=yP, **kw_)
    tau = np.abs(m["T"]).max() * e / It
    P(f"{nom} : N = " + ", ".join(f"{x / 1e3:.2f}" for x in m["N"]) + " kN ; V = "
      + ", ".join(f"{x / 1e3:.2f}" for x in m["V"]) + f" kN ; τ torsion max = {tau:.0f} N/mm²")

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=150)
for a in axs.flat:
    K.style_axes(a)
noms = ["rigides", "raides (100/30 kN/mm) + aile", "souples (10/5 kN/mm) + aile"]
cols = [COL["s1"], COL["s2"], COL["s3"]]
lab = ["ancrages rigides", "ancrages raides (100 et 30 kN/mm)", "ancrages souples (10 et 5 kN/mm)"]
# (a) efforts par ancrage : N
a = axs[0, 0]
w = 0.22
for j, (nm, cl, lb) in enumerate(zip(noms, cols, lab)):
    m = res[(nm, "répartie 300 mm")]
    a.bar(np.arange(3) + (j - 1) * w, m["N"] / 1e3, width=w * 0.92, color=cl, label=lb)
a.axhline(H_tot / 3e3, color=COL["encre2"], lw=1, ls="--")
a.text(2.45, H_tot / 3e3 + 0.6, "H/3", fontsize=7.5, color=COL["encre2"], ha="right")
a.set_xticks(range(3), ["ancrage 1\nz = 200", "ancrage 2\nz = 1000", "ancrage 3\nz = 1800"], fontsize=8)
a.set_ylabel("traction N (kN, ELS)", fontsize=8.5, color=COL["encre2"])
a.set_title("Traction par ancrage", fontsize=9.5, color=COL["encre"], loc="left")
a.legend(fontsize=7, frameon=False, loc="upper right")
# (b) cisaillement V
a = axs[0, 1]
for j, (nm, cl, lb) in enumerate(zip(noms, cols, lab)):
    m = res[(nm, "répartie 300 mm")]
    a.bar(np.arange(3) + (j - 1) * w, m["V"] / 1e3, width=w * 0.92, color=cl, label=lb)
a.axhline(R / 3e3, color=COL["encre2"], lw=1, ls="--")
a.axhline(0, color=COL["axe"], lw=0.8)
a.text(2.45, R / 3e3 + 0.6, "R/3", fontsize=7.5, color=COL["encre2"], ha="right")
a.set_xticks(range(3), ["ancrage 1", "ancrage 2", "ancrage 3"], fontsize=8)
a.set_ylabel("cisaillement V (kN, ELS)", fontsize=8.5, color=COL["encre2"])
a.set_title("Cisaillement par ancrage", fontsize=9.5, color=COL["encre"], loc="left")
# (c) couple de torsion
a = axs[1, 0]
for nm, cl, lb in zip(noms, cols, lab):
    m = res[(nm, "répartie 300 mm")]
    a.plot(m["z"], m["T"] / 1e6, color=cl, lw=1.8, label=lb)
a.axvspan(250, 550, color=COL["neutre"], zorder=0)
for za in (200, 1000, 1800):
    a.axvline(za, color=COL["muet"], lw=0.8, ls=":")
a.set_xlabel("z le long de la cornière (mm)", fontsize=8.5, color=COL["encre2"])
a.set_ylabel("couple de torsion (kN·m)", fontsize=8.5, color=COL["encre2"])
a.set_title("Couple de torsion le long de la cornière (zone grise : charge)", fontsize=9.5, color=COL["encre"], loc="left")
# (d) moment de flexion longitudinal vertical
a = axs[1, 1]
for nm, cl, lb in zip(noms, cols, lab):
    m = res[(nm, "répartie 300 mm")]
    a.plot(m["z"][3:-3], m["Mv"][3:-3] / 1e6, color=cl, lw=1.8, label=lb)
a.axvspan(250, 550, color=COL["neutre"], zorder=0)
for za in (200, 1000, 1800):
    a.axvline(za, color=COL["muet"], lw=0.8, ls=":")
a.set_xlabel("z (mm)", fontsize=8.5, color=COL["encre2"])
a.set_ylabel("moment longitudinal (kN·m)", fontsize=8.5, color=COL["encre2"])
a.set_title("Flexion longitudinale (plan vertical)", fontsize=9.5, color=COL["encre"], loc="left")
fig.suptitle("Cornière L120×120×15 de 2,00 m, 3 ancrages, 42,76 kN répartis sur 300 mm centrés à 0,40 m",
             fontsize=10.5, color=COL["encre"])
fig.tight_layout()
f1 = os.path.join(K.FIGURES, "e03_longitudinal.png")
fig.savefig(f1, facecolor=COL["fond"])
plt.close(fig)
P("")
P("## Figures")
P(f1)
K.ecrire(os.path.join(K.ICI, "e03_sortie.txt"), out)
print("\n".join(out))
