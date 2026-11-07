# -*- coding: utf-8 -*-
"""c02 — Le congé et la section la plus sollicitée (théorie des poutres, coupe par coupe).

1. où sont les sections 2 et 4 du classeur, et quelle hauteur de matière il y a réellement à cet endroit ;
2. trois familles de coupes : verticales dans l'aile horizontale, horizontales dans l'aile verticale, radiales
   issues du centre du congé (0° → coupe horizontale au pied du congé, 45° → bissectrice, 90° → coupe verticale) ;
3. balayage continu σ(position) et maximum réel, face au maximum des quatre sections ;
4. conditions où la section 2 (resp. 4) gouvernerait ; balayage du catalogue du site.

Toutes les grandeurs sont PAR MM de longueur de cornière (q = R / b), comme le classeur (qui divise par b).
"""
import json
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import commun as K  # noqa: E402

RACINE = os.path.dirname(ICI)
SORTIE = os.path.join(RACINE, "sorties", "c02_conge.json")
p = K.CAS
e, r, X, Y, Xj, Yj = p["e"], p["r"], p["X"], p["Y"], p["Xj"], p["Yj"]
b = p["b_cm"] * 10
q = p["R"] * 1000 / b                      # N/mm
adm = p["fy"] / p["gM"]
xl = K.classeur(**p)
res = {"q_N_par_mm": q}

# ---------------------------------------------------------------- 1. la section 2 : où, quelle épaisseur
x2 = X - xl["C20_Xer"]                     # abscisse de la section 2 : X - (X - e) = e
x1 = X - xl["C19_Xe"]                      # abscisse de la section 1 : e + r
t_x2 = K.epaisseur_verticale(e + 1e-9, e, r)
t_x1 = K.epaisseur_verticale(x1, e, r)
print(f"Section 1 à x = {x1} mm (pied du congé sur l'aile horizontale) : matière {t_x1:.6f} mm (classeur : e = {e})")
print(f"Section 2 à x = {x2} mm (aplomb de la face intérieure de l'aile verticale) : matière {t_x2:.6f} mm "
      f"(classeur : e + r = {e + r})")
# pente de la face du congé au sommet de la coupe 2 : dt/dx → -∞ (face verticale) : bord libre de normale x
pente = [(K.epaisseur_verticale(e + d, e, r) - (e + r)) / d for d in (1e-2, 1e-4, 1e-6)]
print("  pente de la face du congé au sommet de la coupe 2 (dt/dx pour dx = 1e-2, 1e-4, 1e-6) :",
      [round(v, 1) for v in pente], "→ la face y est VERTICALE : bord libre de normale x, σ_xx = 0 en fibre haute")
res["section2"] = dict(x=x2, matiere=t_x2, pente_face=pente)
res["section1"] = dict(x=x1, matiere=t_x1)

# ---------------------------------------------------------------- 2. coupes verticales (aile horizontale)
def coupe_verticale(x):
    t = K.epaisseur_verticale(x, e, r) if x > e else e + r
    M = q * (X + Xj - x)                   # N·mm/mm : statique EXACTE (corps libre à droite : la seule charge)
    s = 6 * M / t ** 2
    tau = q / t
    return dict(x=x, t=t, M=M, sigma=s, tau=tau, sigma_eq=math.sqrt(s ** 2 + 3 * tau ** 2))


pas = 0.01
xs = [e + 1e-6] + [e + pas * i for i in range(1, int((X - e) / pas) + 1)]
VC = [coupe_verticale(x) for x in xs]
vmax = max(VC, key=lambda d: d["sigma_eq"])
print(f"\nCoupes verticales x ∈ ]e, X] : σ_eq max = {vmax['sigma_eq']:.2f} MPa à x = {vmax['x']:.2f} mm "
      f"(t = {vmax['t']:.3f} mm), soit {vmax['x'] - e:.2f} mm de la face de l'aile verticale et "
      f"{x1 - vmax['x']:.2f} mm EN DEÇÀ du pied du congé ; section 1 : {xl['S1']['sigma_eq']:.2f} MPa → "
      f"{(vmax['sigma_eq'] / xl['S1']['sigma_eq'] - 1) * 100:+.2f} %")
res["verticales"] = dict(max=vmax, S1=xl["S1"]["sigma_eq"], S2=xl["S2"]["sigma_eq"],
                         ecart_pct=(vmax["sigma_eq"] / xl["S1"]["sigma_eq"] - 1) * 100)
c2 = coupe_verticale(e + 1e-9)
print(f"  à x = e + 0 : t = {c2['t']:.3f}, σ_poutre = {c2['sigma']:.2f} MPa (classeur S2 : {xl['S2']['sigma']:.2f})")

# ---------------------------------------------------------------- 3. coupes horizontales (aile verticale)
# trois statiques : (a) celle du classeur ; (b) et (c) l'équilibre complet du corps libre, réaction verticale de
# la cheville appliquée au nu du mur (x_V = 0) ou au milieu de l'épaisseur (x_V = e/2), pivot y_c = e/2.
def coupe_horizontale(y, variante, y_c=e / 2, x_V=0.0):
    t = K.epaisseur_verticale(y, e, r) if y > e else e + r          # symétrie du congé
    if variante == "classeur":
        M = q * (X + Xj) * (Y - y + Yj) / (Y - e / 2 + Yj)          # C56·C55 (C64·C63) par mm
        N, V = 0.0, q                                                 # τ du classeur : R / (b t)
    else:
        T = q * (X + Xj - x_V) / (Y - y_c)                            # moment autour du point de compression
        # corps libre AU-DESSUS de la coupe (centre G = (t/2, y)) : T vers le mur à la hauteur Y, réaction verticale
        # V = q vers le haut à l'abscisse x_V. Moment en G : T·(Y − y) + q·(x_V − t/2).
        M = T * (Y - y) + q * (x_V - t / 2)
        N, V = q, T                                                   # traction R, cisaillement T
    s = 6 * abs(M) / t ** 2 + N / t
    tau = V / t
    return dict(y=y, t=t, M=M, N=N, V=V, sigma=s, tau=tau, sigma_eq=math.sqrt(s ** 2 + 3 * tau ** 2))


ys = [e + 1e-6] + [e + pas * i for i in range(1, int((Y - e) / pas))]
VARIANTES = (("classeur", "classeur", {}),
             ("equilibre_xV0_yc7.5", "equilibre", dict(x_V=0.0, y_c=e / 2)),
             ("equilibre_xVe2_yc7.5", "equilibre", dict(x_V=e / 2, y_c=e / 2)),
             ("equilibre_xVe2_yc0", "equilibre", dict(x_V=e / 2, y_c=0.0)),
             ("equilibre_xVe2_yc10", "equilibre", dict(x_V=e / 2, y_c=10.0)),
             ("equilibre_xVe2_yc15", "equilibre", dict(x_V=e / 2, y_c=15.0)))
for var, mode, kw in VARIANTES:
    HC = [coupe_horizontale(y, mode, **kw) for y in ys]
    hmax = max(HC, key=lambda d: d["sigma_eq"])
    h3 = coupe_horizontale(e + r, mode, **kw)
    print(f"Coupes horizontales [{var}] : σ_eq max = {hmax['sigma_eq']:.2f} MPa à y = {hmax['y']:.2f} mm "
          f"(t = {hmax['t']:.3f}) ; au pied du congé (y = e + r) : {h3['sigma_eq']:.2f} MPa "
          f"[σ = {h3['sigma']:.2f}, τ = {h3['tau']:.2f}] ; classeur S3 : {xl['S3']['sigma_eq']:.2f} → max/S3 = "
          f"{(hmax['sigma_eq'] / xl['S3']['sigma_eq'] - 1) * 100:+.2f} %")
    res["horizontales_" + var] = dict(max=hmax, pied=h3, S3=xl["S3"]["sigma_eq"],
                                      ecart_pct=(hmax["sigma_eq"] / xl["S3"]["sigma_eq"] - 1) * 100,
                                      taux_max=hmax["sigma_eq"] / adm)

# ---------------------------------------------------------------- 4. coupes radiales issues du centre du congé
def coupe_rad(alpha, y_c=e / 2, x_V=0.0):
    P, Q, t = K.coupe_radiale(alpha, e, r)
    G = ((P[0] + Q[0]) / 2, (P[1] + Q[1]) / 2)
    a = math.radians(alpha)
    m = (-math.sin(a), math.cos(a))          # normale qui va du corps « aile horizontale » vers l'autre corps
    d = (-math.cos(a), -math.sin(a))
    forces = [((X + Xj, e), (0.0, -q))]       # corps « aile horizontale » : la charge
    T = q * (X + Xj - x_V) / (Y - y_c)
    if Q[0] < 1e-9 and y_c < Q[1]:            # sortie sur le mur, au-dessus du point de compression
        forces.append(((0.0, y_c), (T, 0.0)))
    Fx = -sum(f[1][0] for f in forces)
    Fy = -sum(f[1][1] for f in forces)
    M = sum((pt[0] - G[0]) * f[1] - (pt[1] - G[1]) * f[0] for pt, f in forces)
    N = Fx * m[0] + Fy * m[1]
    V = abs(Fx * d[0] + Fy * d[1])
    s = 6 * abs(M) / t ** 2 + abs(N) / t
    tau = V / t
    return dict(alpha=alpha, t=t, P=P, Q=Q, M=M, N=N, V=V, sigma=s, tau=tau,
                sigma_eq=math.sqrt(s ** 2 + 3 * tau ** 2))


RC = [coupe_rad(a / 10) for a in range(0, 901)]
rmax = max(RC, key=lambda d: d["sigma_eq"])
r45 = coupe_rad(45.0)
print(f"\nCoupes radiales : t(0°) = {RC[0]['t']:.2f}, t(45°) = {r45['t']:.3f} mm = √2(e+r) − r, t(90°) = "
      f"{RC[-1]['t']:.2f}")
print(f"  bissectrice 45° : M = {r45['M'] / 1e3:.3f} kN·m/m, N = {r45['N']:.1f} N/mm, σ_eq = {r45['sigma_eq']:.2f} MPa "
      f"(W/W(e) = {(r45['t'] / e) ** 2:.3f}, W/W(e+r) = {(r45['t'] / (e + r)) ** 2:.3f})")
print(f"  max des coupes radiales : {rmax['sigma_eq']:.2f} MPa à α = {rmax['alpha']:.1f}° (t = {rmax['t']:.3f})")
res["radiales"] = dict(r45=dict(t=r45["t"], M=r45["M"], N=r45["N"], sigma_eq=r45["sigma_eq"]),
                       max=dict(alpha=rmax["alpha"], t=rmax["t"], sigma_eq=rmax["sigma_eq"]),
                       W45_sur_We=(r45["t"] / e) ** 2, W45_sur_Wer=(r45["t"] / (e + r)) ** 2)

# ---------------------------------------------------------------- 5. quand la section 2 (4) gouvernerait-elle ?
X_lim = e + (e + r) ** 2 / (2 * e + r)     # σ2 > σ1 ⇔ (X+Xj) − e < (e+r)² / (2e + r)
print(f"\nSection 2 > section 1 seulement si X + Xj < {X_lim:.2f} mm (charge à moins de {X_lim - e - r:.2f} mm du "
      f"pied du congé) ; idem section 4 > section 3 si Y + Yj < {X_lim:.2f} mm")
res["S2_gouverne_si_X_inf"] = X_lim
rapW = ((e + r) / e) ** 2
print(f"W(e+r)/W(e) = {rapW:.3f} ; M2/M1 = {xl['S2']['M'] / xl['S1']['M']:.3f} ; M4/M3 = "
      f"{xl['S4']['M'] / xl['S3']['M']:.3f}")
res["W_rapport"] = rapW

# ---------------------------------------------------------------- 6. balayage du catalogue du site (congé exact r1)
CAT = "/home/user/integ/acier/donnees/profiles.json"
with open(CAT, encoding="utf-8") as f:
    cat = [pp for pp in json.load(f)["profils"] if pp.get("famille") == "L"]
pire = None
tab = []
for pp in cat:
    d = pp["dims"]
    ee, rr, aa = d["t"], d["r1"], d["b"]
    for dist in (5.0, 10.0, 20.0, 0.5 * aa):           # distance de la charge au pied du congé
        XX = ee + rr + dist
        if XX > aa:
            continue
        s1 = 6 * (XX - ee - rr) / ee ** 2
        s2 = 6 * (XX - ee) / (ee + rr) ** 2                  # la section 2 du classeur (« e + r »)
        best = 0.0
        n = 2000
        for i in range(1, n + 1):
            x = ee + rr * i / n
            t = K.epaisseur_verticale(x, ee, rr) if x < ee + rr else ee
            best = max(best, 6 * (XX - x) / t ** 2)
        ratio = best / max(s1, s2)                           # face au MAX des sections 1 et 2, comme le classeur
        tab.append((pp["nom"], dist, ratio))
        if pire is None or ratio > pire[2]:
            pire = (pp["nom"], dist, ratio)
for dist in (5.0, 10.0, 20.0):
    rs = [t[2] for t in tab if t[1] == dist]
    pir = max((t for t in tab if t[1] == dist), key=lambda t: t[2])
    print(f"Catalogue L ({len(rs)} cornières), charge à {dist:.0f} mm du pied du congé : σ_max(congé)/max(σ_S1, σ_S2) "
          f"de {min(rs):.3f} à {max(rs):.3f} (pire : {pir[0]})")
rs = [t[2] for t in tab if t[1] not in (5.0, 10.0, 20.0)]
print(f"Catalogue L, charge au milieu de l'aile restante : σ_max(congé)/max(σ_S1, σ_S2) de {min(rs):.3f} à "
      f"{max(rs):.3f}")
res["catalogue"] = {str(dd): [min(t[2] for t in tab if t[1] == dd), max(t[2] for t in tab if t[1] == dd)]
                    for dd in (5.0, 10.0, 20.0)}
res["catalogue"]["milieu"] = [min(rs), max(rs)]

# courbes pour la figure (c03 les trace avec les EF)
res["courbe_verticale"] = [(d["x"], d["t"], d["sigma"], d["sigma_eq"]) for d in VC[::20]]
res["courbe_horizontale_classeur"] = [(d["y"], d["t"], d["sigma_eq"]) for d in
                                      [coupe_horizontale(y, "classeur") for y in ys[::20]]]
with open(SORTIE, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1, default=float)
print("→", SORTIE)

# ---------------------------------------------------------------- 7. convention « 0,8 r » (EN 1993-1-8, tronçon en T)
# la ligne de rotule d'une semelle laminée est prise à 0,8 r de la face de l'âme, avec l'épaisseur courante.
x08 = e + 0.8 * r
s08h = 6 * q * (X + Xj - x08) / e ** 2
s08v = 6 * q * (X + Xj) * (Y - x08 + Yj) / (Y - e / 2 + Yj) / e ** 2      # statique du classeur
print(f"\nConvention 0,8 r : section à {x08:.1f} mm du nu extérieur, épaisseur e → aile horizontale σ = {s08h:.2f} MPa "
      f"(S1 {xl['S1']['sigma']:.2f}, {(s08h / xl['S1']['sigma'] - 1) * 100:+.1f} %) ; aile verticale σ = {s08v:.2f} MPa "
      f"(S3 {xl['S3']['sigma']:.2f}, {(s08v / xl['S3']['sigma'] - 1) * 100:+.1f} %)")
res["convention_08r"] = dict(x=x08, sigma_h=s08h, sigma_v=s08v)
with open(SORTIE, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1, default=float)
