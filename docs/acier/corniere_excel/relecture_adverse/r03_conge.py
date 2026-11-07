# -*- coding: utf-8 -*-
"""Relecture adverse — 3. Le congé : épaisseur locale, maximum de contrainte, seuil « e + r », généralisation.
Arguments : chemin du xlsx, chemin de profiles.json."""
import json
import math
import sys
import openpyxl

XLSX, PROFILS = sys.argv[1], sys.argv[2]
SADM = 355 / 1.5
R, X, Y, e, r, L = 42.76, 75.0, 85.0, 15.0, 13.0, 300.0


def t_loc(s, e, r):
    """Épaisseur de matière à la distance s du dos de l'autre aile (congé de rayon r centré en (e+r, e+r))."""
    if s >= e + r:
        return e
    return e + r - math.sqrt(max(r * r - (e + r - s) ** 2, 0.0))


def balayage(Mfun, Nfun, Vfun, e, r, pas=0.001):
    best = None
    s = e
    while s <= e + r + 1e-9:
        t = t_loc(s, e, r)
        sig = abs(Mfun(s)) * 1e6 / (L * t * t / 6) + Nfun(s) * 1e3 / (L * t)
        tau = Vfun(s) * 1e3 / (L * t)
        seq = math.sqrt(sig * sig + 3 * tau * tau)
        if best is None or seq > best[1]:
            best = (s, seq, t, Mfun(s), sig)
        s += pas
    return best


# aile horizontale (statique exacte)
bh = balayage(lambda u: R * (X - u) / 1000, lambda u: 0.0, lambda u: R, e, r)
s1 = math.sqrt((R * 47 / 1000 * 1e6 / 11250) ** 2 + 3 * (R * 1e3 / 4500) ** 2)
print(f"horizontale : max à u = {bh[0]:.3f} (à {28-bh[0]:.3f} du pied), t {bh[2]:.4f}, M {bh[3]:.5f}, σeq {bh[1]:.3f},"
      f" taux {bh[1]/SADM*100:.3f} ; S1 {s1:.3f} ; rapport σeq {bh[1]/s1:.5f}")
# aile verticale, classeur
H = R * X / (Y - e / 2)
bv = balayage(lambda y: H * (Y - y) / 1000, lambda y: 0.0, lambda y: R, e, r)
s3 = math.sqrt((H * 57 / 1000 * 1e6 / 11250) ** 2 + 3 * (R * 1e3 / 4500) ** 2)
print(f"verticale classeur : max à y = {bv[0]:.3f} (à {28-bv[0]:.3f}), t {bv[2]:.4f}, M {bv[3]:.5f}, σeq {bv[1]:.3f},"
      f" taux {bv[1]/SADM*100:.3f} ; S3 {s3:.3f} ; rapport {bv[1]/s3:.5f}")
H0 = R * (X - e / 2) / Y
bp = balayage(lambda y: H0 * (Y - y) / 1000, lambda y: R, lambda y: H0, e, r)
print(f"verticale P0 : max à y = {bp[0]:.3f}, t {bp[2]:.4f}, M {bp[3]:.5f}, σeq {bp[1]:.3f}, taux {bp[1]/SADM*100:.3f}")


def rapport_flexion(e, r, s, pas=None):
    """max σ (flexion seule) dans le congé / σ au pied du congé ; charge (ou ancrage) à s du pied."""
    pas = pas or r / 4000
    best = 1.0
    d = 0.0
    while d <= r:
        t = e + r - math.sqrt(max(r * r - (r - d) ** 2, 0.0))   # d : distance au pied, vers l'autre aile
        t = e + r - math.sqrt(max(r * r - d * d, 0)) if False else t_loc(e + r - d, e, r)
        best = max(best, (s + d) / s * (e / t) ** 2)
        d += pas
    return best


print(f"rapports flexion seule cas de contrôle : s=47 -> {rapport_flexion(15,13,47):.5f} ; s=57 -> {rapport_flexion(15,13,57):.5f}")

# seuil « e + r » : section 2 > section 1
def taux12(Xp):
    a = R * (Xp - 28) / 1000 * 1e6 / 11250
    b = R * (Xp - 15) / 1000 * 1e6 / 39200
    return a, b
lo, hi = 28.001, 80
for _ in range(200):
    m = (lo + hi) / 2
    a, b = taux12(m)
    if b > a:
        lo = m
    else:
        hi = m
print(f"seuil flexion seule X' = {lo:.3f}")
def seq12(Xp):
    a, b = taux12(Xp)
    return math.sqrt(a*a + 3*(R*1e3/4500)**2), math.sqrt(b*b + 3*(R*1e3/8400)**2)
lo, hi = 28.001, 80
for _ in range(200):
    m = (lo + hi) / 2
    a, b = seq12(m)
    lo, hi = (m, hi) if b > a else (lo, m)
print(f"seuil σeq X' = {lo:.3f}")
def seq34(Yp):
    Mm = R * X / 1000
    a = Mm * (Yp - 28) / (Yp - 7.5) * 1e6 / 11250
    b = Mm * (Yp - 15) / (Yp - 7.5) * 1e6 / 39200
    return math.sqrt(a*a + 3*(R*1e3/4500)**2), math.sqrt(b*b + 3*(R*1e3/8400)**2)
lo, hi = 28.001, 80
for _ in range(200):
    m = (lo + hi) / 2
    a, b = seq34(m)
    lo, hi = (m, hi) if b > a else (lo, m)
print(f"seuil σeq Y' (S4 > S3, X = 75) = {lo:.3f} ; flexion seule : même formule que X' -> 33,233")

# coupe e + 0,8 r
print(f"e+0,8r : S1 {(75-15-10.4)/47:.4f}  S3 {(85-15-10.4)/57:.4f}")

# généralisation : catalogues
P = json.load(open(PROFILS))["profils"]
egales = [p for p in P if p["famille"] == "L"]
print("cornières égales du site :", len(egales))
wb = openpyxl.load_workbook(XLSX, data_only=True)
ws = wb["CORNIERES"]
vus, classeur = set(), []
for row in ws.iter_rows(min_row=2, max_row=58, values_only=True):
    nom, b, h, t, rr = row[:5]
    if nom in vus:
        continue
    vus.add(nom)
    classeur.append((nom, t, rr))
print("désignations du classeur :", len(classeur))
for s in (40, 20, 10, 5):
    site = max(((rapport_flexion(p["dims"]["t"], p["dims"]["r1"], s), p["nom"]) for p in egales))
    cls = max(((rapport_flexion(t, rr, s), nom) for nom, t, rr in classeur))
    tous_site = max(((rapport_flexion(p["dims"]["t"], p["dims"]["r1"], s), p["nom"]) for p in P if p["famille"] in ("L", "LI")))
    print(f"  s = {s:2d} : site égales {site[0]:.3f} ({site[1]}) ; classeur {cls[0]:.3f} ({cls[1]}) ; site L+LI {tous_site[0]:.3f} ({tous_site[1]})")
