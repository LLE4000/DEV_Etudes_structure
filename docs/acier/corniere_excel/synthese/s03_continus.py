# -*- coding: utf-8 -*-
"""S03 — Diagrammes continus exacts et congé.

Aile horizontale : M(u) = R·(X' − u) (statique exacte du porte-à-faux, P8!C35 et C45 en sont deux lectures).
Aile verticale (méthode du classeur) : M(y) = H·(Y' − y), H = Mmax/(YH + Yj) (P8!C55·C56 et C63·C64).
Aile verticale (équilibre cohérent P0 : pivot à l'arête, V sur l'axe) : M(y) = H0·(Y' − y), N = R, V = H0.
Épaisseur dans le congé : t(s) = e + r − √(r² − (e + r − s)²), s compté depuis le dos de l'autre aile.
Contrainte : σ = M/(L·t²/6) [+ N/(L·t)], τ = V/(L·t), σ_éq = √(σ² + 3τ²).

Sorties : maxima, généralisation au catalogue (rapport pic dans le congé / pied du congé), courbes pour la figure (b).

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/s03_continus.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_commun as K  # noqa: E402

S = K.Sortie("s03_continus")
C = K.CAS
e, r, L, R = C["e"], C["r"], C["L"], C["R"]
h = K.historique(C)
sadm = h["sadm"]
Xp, Yp = C["X"] + C["Xj"], C["Y"] + C["Yj"]
H = h["H"]
fb0 = K.corps_libre_A(C, 0.0, e / 2)


def balayer(debut, fin, pas, Mf, Nf, Vf, tauf=None):
    pts = []
    k = 0
    while True:
        s = debut + k * pas
        if s > fin + 1e-9:
            break
        t = K.t_conge(s, e, r)
        M = Mf(s)
        W, A = L * t * t / 6, L * t
        sig = M / W + Nf(s) / A
        tau = Vf(s) / A
        seq = math.sqrt(sig * sig + 3 * tau * tau)
        pts.append(dict(s=s, t=t, M=M, sig=sig, tau=tau, seq=seq, taux=seq / sadm))
        k += 1
    return pts


def resume(nom, pts, s_pied, s_nu):
    imax = max(range(len(pts)), key=lambda i: pts[i]["seq"])
    pm = pts[imax]
    pied = min(pts, key=lambda q: abs(q["s"] - s_pied))
    nu = min(pts, key=lambda q: abs(q["s"] - s_nu))
    mil = min(pts, key=lambda q: abs(q["s"] - (s_nu + s_pied) / 2))
    S(f"{nom} :")
    for lab, q in (("nu de l'autre aile (section 2 ou 4)", nu), ("milieu du congé", mil), ("MAXIMUM", pm), ("pied du congé (section 1 ou 3)", pied)):
        S(f"   {lab:38s} s = {q['s']:6.2f} mm  t = {q['t']:6.3f}  M = {q['M'] / 1e6:.5f} kN·m  σ = {q['sig']:7.3f}  "
          f"σ_éq = {q['seq']:7.3f} N/mm²  taux = {100 * q['taux']:6.2f} %")
    S(f"   maximum à {s_pied - pm['s']:.2f} mm avant le pied du congé, {100 * (pm['seq'] / pied['seq'] - 1):+.2f} % sur la section de référence")
    return dict(max=pm, pied=pied, nu=nu, mil=mil, ecart=pm["seq"] / pied["seq"] - 1, recul=s_pied - pm["s"])


S("# S03 — diagrammes continus et congé (cas de contrôle)")
S("")
S("## 1. Maxima le long des ailes (pas de 0,01 mm)")
tau_cl = lambda s: R                      # le classeur prend R comme « τ » dans les quatre sections
ah = balayer(e, Xp, 0.01, lambda u: R * max(Xp - u, 0.0), lambda u: 0.0, tau_cl)
res_h = resume("Aile horizontale (statique exacte ; τ = R/(L·t))", ah, e + r, e)
av = balayer(e, Yp, 0.01, lambda y: H * max(Yp - y, 0.0), lambda y: 0.0, tau_cl)
res_v = resume("Aile verticale, méthode du classeur (M = H·(Y' − y), τ = R/(L·t))", av, e + r, e)
av0 = balayer(e, Yp, 0.01, lambda y: fb0["M"](y), lambda y: R, lambda y: fb0["H"])
res_v0 = resume("Aile verticale, équilibre P0 (pivot à l'arête, V sur l'axe ; σ = M/W + N/A, τ = H/A)", av0, e + r, e)
S.v("max_h", res_h)
S.v("max_v", res_v)
S.v("max_v0", res_v0)

S("")
S("## 2. Quand la section « e + r » dépasserait-elle la section e ?")
Xs = e + r * (e + r) ** 2 / ((e + r) ** 2 - e * e)
S(f"σ2 > σ1 ⇔ X + Xj < e + r·(e+r)²/((e+r)² − e²) = {Xs:.2f} mm (charge à {Xs - e - r:.2f} mm du pied du congé) ; même seuil pour Y + Yj")
S.v("seuil_S2", Xs)


def _bis(f, lo, hi):
    flo = f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (flo > 0):
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return (lo + hi) / 2


_a = e + r
_W1, _W2, _A1, _A2 = L * e * e / 6, L * _a * _a / 6, L * e, L * _a
_eqh = lambda Xp_: (math.hypot(R * (Xp_ - e) / _W2, math.sqrt(3) * R / _A2)
                    - math.hypot(R * (Xp_ - e - r) / _W1, math.sqrt(3) * R / _A1))
_Mm = R * (C["X"] + C["Xj"])
_eqv = lambda Yp_: (math.hypot(_Mm * (Yp_ - e) / (Yp_ - e / 2) / _W2, math.sqrt(3) * R / _A2)
                    - math.hypot(_Mm * (Yp_ - e - r) / (Yp_ - e / 2) / _W1, math.sqrt(3) * R / _A1))
Xs_eq, Ys_eq = _bis(_eqh, _a, 200.0), _bis(_eqv, _a, 200.0)
S(f"avec le critère du classeur (σ_éq, cisaillement compris) : σ_éq,2 > σ_éq,1 ⇔ X + Xj < {Xs_eq:.2f} mm ; "
  f"σ_éq,4 > σ_éq,3 ⇔ Y + Yj < {Ys_eq:.2f} mm (à X = {C['X']:.0f} mm) ; le seuil en flexion seule ({Xs:.2f} mm) est une condition nécessaire")
S.v("seuil_S2_eq", Xs_eq)
S.v("seuil_S4_eq", Ys_eq)

S("")
S("## 3. Convention « 0,8·r » (analogie EN 1993-1-8, à confirmer)")
u08 = e + 0.8 * r
s08 = R * (Xp - u08) / (L * e * e / 6)
s08v = H * (Yp - u08) / (L * e * e / 6)
S(f"coupe à e + 0,8·r = {u08:.1f} mm avec l'épaisseur e : σ = {s08:.1f} N/mm² ({100 * (s08 / h['sig'][0] - 1):+.1f} % sur S1) ; "
  f"aile verticale {s08v:.1f} ({100 * (s08v / h['sig'][2] - 1):+.1f} % sur S3)")

S("")
S("## 4. Généralisation : rapport pic dans le congé / contrainte au pied du congé (flexion seule)")
S("La charge (ou l'ancrage) est à la distance s du pied du congé ; le rapport ne dépend que de e, r et s.")


def rapport(ee, rr, s):
    Xq = ee + rr + s
    best = 0.0
    u = ee
    while u <= ee + rr + 1e-9:
        t = K.t_conge(u, ee, rr)
        best = max(best, (Xq - u) / t ** 2)
        u += 0.005
    pied = s / ee ** 2
    s2 = (Xq - ee) / (ee + rr) ** 2
    return best / pied, best / max(pied, s2)


cat_site = [(q["nom"], q["dims"]["t"], q["dims"]["r1"], min(q["dims"]["h"], q["dims"]["b"])) for q in K.profils_site(("L",))]
cl = K.Classeur()
vus, cat_cl = set(), []
for row in cl.cat:
    if row[0] and row[0] != "NON STANDARD" and (row[0], row[3], row[4]) not in vus:
        vus.add((row[0], row[3], row[4]))
        cat_cl.append((row[0], float(row[3]), float(row[4]), float(min(row[1], row[2]))))
gen = {}
for nom_cat, cat in (("catalogue du site, cornières égales (r = r1)", cat_site), ("catalogue du classeur (rayon de la colonne E)", cat_cl)):
    S(f"{nom_cat} : {len(cat)} cornières")
    gen[nom_cat] = {}
    for s in (40.0, 20.0, 10.0, 5.0):
        vals = [(rapport(ee, rr, s)[0], rapport(ee, rr, s)[1], nom) for nom, ee, rr, aile in cat if ee + rr + s <= aile]
        if not vals:
            continue
        mx = max(vals)
        mn = min(vals)
        mx2 = max(vals, key=lambda v: v[1])
        S(f"   s = {s:4.0f} mm : pic/pied de {mn[0]:.3f} ({mn[2]}) à {mx[0]:.3f} ({mx[2]}) ; pic/max(S1, S2) au plus {mx2[1]:.3f} ({mx2[2]}) — {len(vals)} cornières")
        gen[nom_cat][s] = dict(min=mn, max=mx, max2=mx2, n=len(vals))
    ee, rr = 15.0, 13.0
S(f"cas de contrôle (e = 15, r = 13) : s = 47 mm → {rapport(15, 13, 47)[0]:.4f} ; s = 57 mm → {rapport(15, 13, 57)[0]:.4f}")
S.v("generalisation", gen)

# ------------------------------------------------------------------ courbes pour la figure (b)
courbes = dict(
    h=[(q["s"], q["M"] / 1e6, q["t"], q["sig"], q["seq"], q["taux"]) for q in ah[::10]] + [(ah[-1]["s"], 0.0, e, 0.0, ah[-1]["seq"], ah[-1]["taux"])],
    v=[(q["s"], q["M"] / 1e6, q["t"], q["sig"], q["seq"], q["taux"]) for q in av[::10]],
    v0=[(q["s"], q["M"] / 1e6, q["t"], q["sig"], q["seq"], q["taux"]) for q in av0[::10]],
    Mmax=h["Mmax"] / 1e6, M=[m / 1e6 for m in h["M"]], seq=h["seq"], taux=h["taux"], H=H, H0=fb0["H"],
    M0_angle=fb0["M"](e) / 1e6, M0_nu=fb0["M"](0) / 1e6,
)
with open(os.path.join(K.SORTIES, "s03_courbes.json"), "w", encoding="utf-8") as fh:
    json.dump(courbes, fh)
S("")
S(f"courbes écrites : {os.path.join(K.SORTIES, 's03_courbes.json')}")
S(f"P0 : H0 = {fb0['H'] / 1e3:.3f} kN ; M0 au pied du congé (section 3) = {fb0['M'](e + r) / 1e6:.5f} ; au nu (section 4) = {fb0['M'](e) / 1e6:.5f} kN·m")
S.ecrire()
