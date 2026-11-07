# -*- coding: utf-8 -*-
"""S07 — Maillage de la coupe (prototype extensions/ef_l.py, numpy seul), relancé ici.

Cas A-ref : configuration A, tranche de 1 mm en déformation plane, q = R/L = 42 760/300 N/mm répartie sur 10 mm
autour de X ; mur raide (Winkler 1e5 N/mm³, contact unilatéral) ; ancrage = ressorts unilatéraux sous l'écrou
(± 3 mm), 300 N/mm par mm de cornière (soit 30 kN/mm par cheville, 3 chevilles sur 300 mm — HYPOTHÈSE) ;
cisaillement de l'ancrage au nœud de mi-épaisseur. Cas B-Y85 : la même cornière retournée.

Sorties : réactions, pivot trouvé, sections 1 à 4 (résultantes et fibres), pic à la surface du congé, données de la
figure (e) (maillage déformé, von Mises nodal).

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/s07_maillage.py
"""
import json
import math
import os
import sys
import time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(1, os.path.join(os.path.dirname(ICI), "extensions"))
import numpy as np  # noqa: E402
import ef_l  # noqa: E402
import s_commun as K  # noqa: E402

S = K.Sortie("s07_maillage")
C = K.CAS
e, r, L, R, X = C["e"], C["r"], C["L"], C["R"], C["X"]
q = R / L
h = K.historique(C)
GEO = dict(B=C["larg"], H=C["haut"], e=e, r=r, Y=C["Y"])
MAIL = dict(n_t=10, n_c=16, n_leg=40, grad=4.0)
a = e + r

S("# S07 — maillage de la coupe (cas de contrôle)")
t0 = time.perf_counter()
pb = ef_l.Probleme(GEO, "A", maillage=MAIL, solveur="bande")
pb.resoudre(q, X, k_anc=300.0, k_mur=1e5, b_rond=3.0)
dt = time.perf_counter() - t0
rr = pb.reactions
yv, fv = rr["mur_y"], rr["mur_f"]
bas = yv < C["Y"]
C_bas = fv[bas].sum() * L
y_c = (fv[bas] * yv[bas]).sum() / max(fv[bas].sum(), 1e-30)
S(f"maillage : {pb.n} nœuds, {len(pb.elements)} éléments à 9 nœuds, {2 * pb.n} inconnues ; calcul {dt:.2f} s (CPython, numpy seul)")
S(f"traction de l'ancrage H = {rr['H'] * L / 1e3:.3f} kN à y = {rr['y_H']:.2f} mm ; compression du mur {C_bas / 1e3:.3f} kN centrée à y = {y_c:.2f} mm ; "
  f"contact au-dessus de l'ancrage {fv[~bas].sum() * L / 1e3:.3f} kN ; V = {rr['V'] * L / 1e3:.3f} kN ; itérations de contact {pb.iterations}")
H_P0 = K.corps_libre_A(C, 0.0, e / 2)["H"]
S(f"corps libre P0 : H = {H_P0 / 1e3:.3f} kN ; classeur : H = {h['H'] / 1e3:.3f} kN (sections), {3 * h['N_els'] / 1e3:.3f} kN (ancrages, C82 × 3) ; "
  f"le classeur dépasse le maillage de {100 * (3 * h['N_els'] / (rr['H'] * L) - 1):.0f} % sur la traction")
S.v("A", dict(H=rr["H"] * L, yH=rr["y_H"], C=C_bas, yc=y_c, V=rr["V"] * L, n=pb.n, t=dt))

cuts = {1: dict(debut=(a, 0.0), fin=(a, e), normale="x", t=e), 2: dict(debut=(e, 0.0), fin=(e, a), normale="x", t=a),
        3: dict(debut=(0.0, a), fin=(e, a), normale="y", t=e), 4: dict(debut=(0.0, e), fin=(a, e), normale="y", t=a)}
S("| section | M classeur (kN·m) | σ classeur | M maillage | N maillage (kN) | σ poutre (M, N du maillage) | fibre côté congé | fibre opposée |")
sec = {}
for k, cu in cuts.items():
    cp = pb.coupe(cu["debut"], cu["fin"], n=60, normale=cu["normale"])
    t = cu["t"]
    d = 1e-4
    if cu["normale"] == "x":
        s_int, s_ext = pb.contrainte_en((cu["debut"][0], t - d))[0], pb.contrainte_en((cu["debut"][0], d))[0]
    else:
        s_int, s_ext = pb.contrainte_en((t - d, cu["debut"][1]))[1], pb.contrainte_en((d, cu["debut"][1]))[1]
    s_p = abs(cp["M"]) / (t * t / 6) + cp["N"] / t
    sec[k] = dict(M=abs(cp["M"]) * L, N=cp["N"] * L, s_int=s_int, s_ext=s_ext, s_p=s_p)
    S(f"| {k} | {h['M'][k - 1] / 1e6:.4f} | {h['sig'][k - 1]:.2f} | {abs(cp['M']) * L / 1e6:.4f} | {cp['N'] * L / 1e3:.2f} | {s_p:.2f} | "
      f"{s_int:.2f} | {s_ext:.2f} |")
S.v("sections", sec)
S(f"S1 : fibre côté congé / poutre = {sec[1]['s_int'] / sec[1]['s_p']:.3f} ; S3 : {sec[3]['s_int'] / sec[3]['s_p']:.3f} ; "
  f"S3 fibre maillée / σ3 du classeur = {sec[3]['s_int'] / h['sig'][2]:.3f}")
S(f"S2 : |fibre opposée| / σ2 du classeur = {abs(sec[2]['s_ext']) / h['sig'][1]:.3f} (fibre côté congé {sec[2]['s_int']:.2f}) ; "
  f"S4 : {abs(sec[4]['s_ext']) / h['sig'][3]:.3f} (fibre côté congé {sec[4]['s_int']:.2f})")
S(f"M3 classeur / M3 maillé = {h['M'][2] / sec[3]['M']:.3f} ; H sections classeur / H maillé = {h['H'] / (rr['H'] * L):.3f}")
S(f"M3 maillé / M3 classeur − 1 = {100 * (sec[3]['M'] / h['M'][2] - 1):.1f} % ; M1 maillé / M1 classeur − 1 = {100 * (sec[1]['M'] / h['M'][0] - 1):.2f} %")
S(f"section 1 : M maillage / statique exacte R·(X − u) = {sec[1]['M'] / (R * (X - a)):.4f} (charge répartie sur 10 mm)")

# pic le long de la surface intérieure (arc du congé), contrainte tangentielle
pts = []
Cc = (a, a)
for th in np.linspace(0, math.pi / 2, 181):
    ang = 3 * math.pi / 2 - th
    Pp = (Cc[0] + (r + 1e-4) * math.cos(ang), Cc[1] + (r + 1e-4) * math.sin(ang))
    s = pb.contrainte_en(Pp)
    tv = np.array([-math.sin(ang), math.cos(ang)])
    pts.append((r * th, Pp, s[0] * tv[0] ** 2 + s[1] * tv[1] ** 2 + 2 * s[2] * tv[0] * tv[1]))
k = max(range(len(pts)), key=lambda i: pts[i][2])
s_arc, P_arc, sig_pk = pts[k]
u_pk = P_arc[0]
t_pk = K.t_conge(u_pk, e, r)
s_poutre = R * (X - u_pk) / L / (t_pk ** 2 / 6)
S(f"pic à la surface du congé : {sig_pk:.1f} N/mm² à {s_arc:.2f} mm le long de l'arc depuis le pied côté aile horizontale "
  f"(u = {P_arc[0]:.2f}, y = {P_arc[1]:.2f}) ; poutre au même point {s_poutre:.1f} → rapport {sig_pk / s_poutre:.2f} ; "
  f"{100 * sig_pk / h['sadm']:.1f} % de σ_adm ; ×{sig_pk / h['sig'][0]:.2f} sur σ1 du classeur")
s_n, vm = pb.contraintes_noeuds()
S(f"von Mises nodal maximal (déformation plane) : {vm.max():.1f} N/mm²")
S.v("pic", dict(sig=sig_pk, s=s_arc, u=P_arc[0], y=P_arc[1], rapport=sig_pk / s_poutre, poutre=s_poutre, vm=float(vm.max())))

# configuration B
pbB = ef_l.Probleme(GEO, "B", maillage=MAIL, solveur="bande")
pbB.resoudre(q, X, k_anc=300.0, k_mur=1e5, b_rond=3.0)
rb = pbB.reactions
ybv, fbv = rb["mur_y"], rb["mur_f"]
yA = C["haut"] - C["Y"]
bb = ybv < yA
cpB = pbB.coupe((0.0, C["haut"] - a), (e, C["haut"] - a), n=60, normale="y")
HB = R * (X - e / 2) / (yA - 0.0)
S(f"B (Y_B = 85) : H = {rb['H'] * L / 1e3:.3f} kN (corps libre {HB / 1e3:.3f}) ; contact au bout libre {fbv[bb].sum() * L / 1e3:.3f} kN à y = "
  f"{(fbv[bb] * ybv[bb]).sum() / max(fbv[bb].sum(), 1e-30):.2f} ; section 3 : M = {abs(cpB['M']) * L / 1e6:.4f} kN·m "
  f"(corps libre {R * (X - e / 2) / 1e6:.4f}), N = {cpB['N'] * L / 1e3:.2f} kN")
S(f"B : H maillé / corps libre − 1 = {100 * (rb['H'] * L / HB - 1):.2f} %")
S.v("B", dict(H=rb["H"] * L, M3=abs(cpB["M"]) * L, N3=cpB["N"] * L))


# mur souple (même modèle d'ancrage que le cas de référence : écrou ± 3 mm, 30 kN/mm par cheville, V à mi-épaisseur)
S(f"mur raide : M3 maillé = {sec[3]['M'] / 1e6:.4f} kN·m (classeur {100 * (h['M'][2] / sec[3]['M'] - 1):+.0f} %)")
S("mur souple (module de réaction de Winkler, maillage fin, écrou ± 3 mm) : pivot trouvé et section 3 NOMINALE |M|/W + N/A "
  f"face au classeur ({h['sig'][2]:.2f} N/mm²)")


def mur_souple(km, b_rond=3.0):
    pbm = ef_l.Probleme(GEO, "A", maillage=MAIL, solveur="bande")
    pbm.resoudre(q, X, k_anc=300.0, k_mur=km, b_rond=b_rond)
    rm = pbm.reactions
    bm = rm["mur_y"] < C["Y"]
    ycm = (rm["mur_f"][bm] * rm["mur_y"][bm]).sum() / rm["mur_f"][bm].sum()
    cpm = pbm.coupe((0.0, a), (e, a), n=60, normale="y")
    snom = abs(cpm["M"]) / (e * e / 6) + cpm["N"] / e
    return dict(H=rm["H"] * L, yH=rm["y_H"], yc=ycm, M3=abs(cpm["M"]) * L, N3=cpm["N"] * L, snom=snom)


souple = {1e5: dict(H=rr["H"] * L, yH=rr["y_H"], yc=y_c, M3=sec[3]["M"], N3=sec[3]["N"], snom=sec[3]["s_p"])}
for km in (1e3, 300.0, 100.0, 30.0, 20.0, 10.0, 5.0):
    souple[km] = mur_souple(km)
for km, o in souple.items():
    S(f"   mur {km:>8g} N/mm³ : H = {o['H'] / 1e3:.2f} kN (à y = {o['yH']:.2f}) ; pivot à {o['yc']:.2f} mm ; M3 = {o['M3'] / 1e6:.4f} kN·m ; "
      f"N3 = {o['N3'] / 1e3:.2f} kN ; σ3 nominale {o['snom']:.1f} N/mm² ({100 * (o['snom'] / h['sig'][2] - 1):+.1f} % sur le classeur) ; "
      f"taux poutre (même pivot, V sur l'axe) {100 * K.taux_section_v(C, K.corps_libre_A(C, o['yc'], e / 2)['M'](a), R, K.corps_libre_A(C, o['yc'], e / 2)['H'])[0]:.2f} %")
o15 = mur_souple(30.0, b_rond=15.0)
S(f"   variante rondelle ÉLASTIQUE ± 15 mm, mur 30 N/mm³ : H = {o15['H'] / 1e3:.2f} kN à y = {o15['yH']:.2f} ; pivot {o15['yc']:.2f} mm ; "
  f"σ3 nominale {o15['snom']:.1f} N/mm² ({100 * (o15['snom'] / h['sig'][2] - 1):+.1f} %)")
s_r, s_30 = souple[1e5]["snom"], souple[30.0]["snom"]
S(f"   de mur raide à 30 N/mm³ : σ3 nominale {s_r:.1f} → {s_30:.1f} N/mm² ({100 * (s_30 / s_r - 1):+.1f} %, soit "
  f"{100 * (s_30 - s_r) / h['sadm']:.1f} points de σ_adm)")
ks_ = sorted(souple)
for k1, k2 in zip(ks_[:-1], ks_[1:]):
    s1, s2 = souple[k1]["snom"] - h["sig"][2], souple[k2]["snom"] - h["sig"][2]
    if s1 * s2 <= 0:
        f_ = s1 / (s1 - s2)
        kseuil = math.exp(math.log(k1) + f_ * (math.log(k2) - math.log(k1)))
        S(f"   σ3 nominale = classeur vers {kseuil:.0f} N/mm³ (interpolation logarithmique entre {k1:g} et {k2:g})")
        S.v("k_seuil_mur", kseuil)


def _sig_pivot(yP):
    cl = K.corps_libre_A(C, yP, e / 2)
    return cl["M"](a) / (L * e * e / 6) + R / (L * e)


lo, hi = 0.0, 40.0
for _ in range(100):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if _sig_pivot(mid) < h["sig"][2] else (lo, mid)
S(f"   poutre, résultante d'ancrage et cisaillement sur l'axe : σ3 nominale = classeur pour un pivot à {lo:.2f} mm")
S.v("mur_souple", {str(k): v for k, v in souple.items()})

S("effet de levier (maillage FIN, mur raide, écrou ± 3 mm) : raideur axiale d'une cheville → traction totale, contact au-dessus de l'ancrage")
lev = {}
for kc in (30.0, 100.0, 200.0, 250.0, 300.0, 1000.0, 1e5):
    pg = ef_l.Probleme(GEO, "A", maillage=MAIL, solveur="bande")
    pg.resoudre(q, X, k_anc=kc * 1e3 * 3 / L, k_mur=1e5, b_rond=3.0)
    rg = pg.reactions
    haut_ = rg["mur_y"] >= C["Y"]
    lev[kc] = (rg["H"] * L, rg["mur_f"][haut_].sum() * L)
    S(f"   {kc:>9.0f} kN/mm : H = {rg['H'] * L / 1e3:6.2f} kN ; contact en tête {rg['mur_f'][haut_].sum() * L / 1e3:6.2f} kN")
pr = ef_l.Probleme(GEO, "A", maillage=MAIL, solveur="bande")
pr.resoudre(q, X, k_anc=1e7, k_mur=1e5, b_rond=3.0)
rr2 = pr.reactions
ht = rr2["mur_y"] >= C["Y"]
S(f"ancrage infiniment raide (maillage fin) : H = {rr2['H'] * L / 1e3:.2f} kN, dont contact en tête {rr2['mur_f'][ht].sum() * L / 1e3:.2f} kN ; "
  f"{rr2['H'] * L / (3 * h['N_els']):.2f} fois la traction du classeur")
pg = ef_l.Probleme(GEO, "A", maillage=dict(n_t=6, n_c=8, n_leg=20, grad=4.0), solveur="bande")
pg.resoudre(q, X, k_anc=30.0 * 1e3 * 3 / L, k_mur=1e5, b_rond=3.0)
S(f"(pour mémoire, maillage gros n_t = 6, n_c = 8, n_leg = 20, 30 kN/mm : H = {pg.reactions['H'] * L / 1e3:.2f} kN, "
  f"{100 * (lev[30.0][0] / (pg.reactions['H'] * L) - 1):.1f} % sous le maillage fin)")
S.v("levier", {str(k): v for k, v in lev.items()})
S.v("rigide", rr2["H"] * L)

# données de la figure (e)
U = pb.U
ech = 6.0 / np.abs(U).max()
fig = dict(noeuds=pb.noeuds.tolist(), U=U.tolist(), elements=pb.elements.tolist(), vm=vm.tolist(), ech=ech,
           umax=float(np.abs(U).max()), H=rr["H"] * L, yc=y_c, pic=dict(u=P_arc[0], y=P_arc[1], sig=sig_pk))
with open(os.path.join(K.SORTIES, "s07_figure.json"), "w", encoding="utf-8") as fh:
    json.dump(fig, fh)
S(f"déplacement maximal {np.abs(U).max():.4f} mm ; amplification du dessin ×{ech:.0f}")
S.ecrire()
