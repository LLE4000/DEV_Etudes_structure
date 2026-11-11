# -*- coding: utf-8 -*-
"""Le moteur des SECTIONS QUELCONQUES (rdm/sections.py, docs/site/BIBLIOTHEQUE_AUTRES.md § 5).

    /usr/bin/python3.12 tests/test_rdm_sections.py

Ce que la suite prouve :
  1. les formes simples contre leurs formules (b·h³/12, π·d⁴/64…), écrites ici ;
  2. chaque profilé du CATALOGUE, non coupé, retrouve les valeurs du catalogue (A, I_y, I_z, W_el, W_pl) ; et le cœur en
     bibliothèque standard donne les mêmes nombres que acier/profiles/geometrie.py (numpy), écrit séparément ;
  3. des sections composées recalculées À LA MAIN (Huygens-Steiner écrit dans le test) : PRS en I, caisson de plats,
     HEA + plat, demi-HEA, section en Z (axes principaux), évidement ;
  4. les invariants : translation, rotation (I_1, I_2 conservés ; I_y + I_z conservé), miroir ; la rotation directe des
     inerties retrouve I_1, I_2 et annule I_uv ;
  5. coupes, recouvrements, refus, mutations.
"""
import math
import os
import sys

RACINE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RACINE)

from rdm import sections as S  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def pres(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


def plat(b, h, y=0.0, z=0.0, ang=0.0, nom="plat", ev=False):
    return S.element(S.transformer(S.rectangle(b, h), ang, False, y, z), nom, ev)


# ================================================================================================ 1 · formes simples
print("=== 1 · Formes simples")
c = S.caracteristiques([plat(200.0, 10.0)])
chk("rectangle 200 × 10 : A = b·h, I_y = b·h³/12, I_z = h·b³/12, W_el = b·h²/6, W_pl = b·h²/4, I_yz = 0, centre à l'origine",
    pres(c["A"], 2000.0) and pres(c["Iy"], 200 * 1000 / 12) and pres(c["Iz"], 10 * 8e6 / 12) and pres(c["Wel_y"], 200 * 100 / 6)
    and pres(c["Wpl_y"], 200 * 100 / 4, 1e-9) and pres(c["Wpl_z"], 10 * 40000 / 4, 1e-9) and abs(c["Iyz"]) < 1e-6 and abs(c["yG"]) < 1e-12, c)
d = 100.0
c = S.caracteristiques([S.element(S.rond(d), "rond")])
chk("rond Ø 100 (polygone inscrit de 180 côtés) : A = π·d²/4 à 2·10⁻⁴, I = π·d⁴/64 à 4·10⁻⁴, W_pl = d³/6 à 4·10⁻⁴, isotrope",
    abs(c["A"] / (math.pi * d * d / 4) - 1) < 2.1e-4 and abs(c["Iy"] / (math.pi * d ** 4 / 64) - 1) < 4.2e-4 and abs(c["Wpl_y"] / (d ** 3 / 6) - 1) < 4.2e-4
    and c["isotrope"] and pres(c["Iy"], c["Iz"], 1e-9), (c["A"], c["Iy"]))
tri = S.element(S.polygone([(0, 0), (60, 0), (0, 90)]), "triangle")
c = S.caracteristiques([tri])
chk("triangle rectangle b = 60, h = 90 : A = b·h/2, centre à (b/3 ; h/3), I_y = b·h³/36, I_z = h·b³/36, I_yz = −b²·h²/72",
    pres(c["A"], 2700.0) and pres(c["yG"], 20.0) and pres(c["zG"], 30.0) and pres(c["Iy"], 60 * 90 ** 3 / 36) and pres(c["Iz"], 90 * 60 ** 3 / 36)
    and pres(c["Iyz"], -(60 ** 2) * 90 ** 2 / 72), (c["Iy"], c["Iyz"]))
chk("polygone donné dans le sens horaire : remis dans le sens trigonométrique (aire positive)", S.element(S.polygone([(0, 0), (0, 90), (60, 0)]))["A"] > 0)

# ================================================================================================ 2 · le catalogue
print("\n=== 2 · Les profilés du catalogue, non coupés, retrouvent le catalogue")
from acier.profiles import catalogue as KA  # noqa: E402
from acier.profiles import familles as FA  # noqa: E402
from acier.profiles import geometrie as GE  # noqa: E402

echantillon = ["IPE 300", "IPE 80", "IPN 200", "HEA 200", "HEB 300", "HEM 100", "UPE 200", "UPN 200", "L 100x100x10", "L 150x90x10",
               "CHS 114.3x5", "SHS 100x100x5", "RHS 200x100x6", "HL 920x344", "HD 400x1086.0", "HP 200x43", "RHS 50x25x2 (froid)"]
pire = {}
for nom in echantillon:
    contours, p = S.profil(nom)
    c = S.caracteristiques([S.element(contours, nom)])
    for k_cat, v, ech in (("A", c["A"] / 100, 1), ("Iy", c["Iy"] / 1e4, 1), ("Iz", c["Iz"] / 1e4, 1), ("Wel_y", c["Wel_y"] / 1e3, 1), ("Wel_z", c["Wel_z"] / 1e3, 1),
                          ("Wpl_y", c["Wpl_y"] / 1e3, 1), ("Wpl_z", c["Wpl_z"] / 1e3, 1), ("iy", c["iy"] / 10, 1)):
        if p.get(k_cat):
            pire[k_cat] = max(pire.get(k_cat, 0.0), abs(v / p[k_cat] - 1))
chk(f"{len(echantillon)} profilés (I, H, U, L, creux ronds et rectangulaires, laminés et formés à froid) : A, I_y, I_z, W_el, W_pl, i_y du catalogue "
    f"retrouvés à 10⁻⁵ près (pire écart {max(pire.values()):.1e})", max(pire.values()) < 1e-5 and len(pire) == 8, pire)
tous = KA.profils()
pire_A = max(abs(S.caracteristiques([S.element(S.profil(p["nom"])[0], p["nom"])], plastique=False)["Iy"] / 1e4 / p["Iy"] - 1) for p in tous[::7])
chk(f"un profilé sur sept du catalogue ({len(tous[::7])} profilés) : I_y retrouvé (pire écart {pire_A:.1e})", pire_A < 1e-5, pire_A)
p = KA.par_nom()["L 150x90x10"]
c = S.caracteristiques([S.element(S.profil("L 150x90x10")[0], "L")])
chk("cornière inégale L 150×90×10 : axes principaux du catalogue — I_u, I_v à 10⁻⁵ ; angle α au catalogue près (à 180° près)",
    abs(c["I1"] / 1e4 / p["Iu"] - 1) < 1e-5 and abs(c["I2"] / 1e4 / p["Iv"] - 1) < 1e-5
    and min(abs(((c["alpha_deg"] - p["alpha_deg"]) + 90) % 180 - 90), 180) < 1e-3, (c["alpha_deg"], p["alpha_deg"], c["I1"] / 1e4, p["Iu"]))
import numpy as np  # noqa: E402

rings = FA.contours("UPN", KA.par_nom()["UPN 200"]["dims"])
g = GE.proprietes(rings)
c = S.caracteristiques([S.element([[(float(q[0]), float(q[1])) for q in r_] for r_ in rings], "UPN")])
chk("cœur en bibliothèque standard = acier/profiles/geometrie.py (numpy, écrit séparément) sur l'UPN 200 : A, centre, I_y, I_z, I_yz, W_pl",
    pres(c["A"], g["A"], 1e-12) and pres(c["yG"], g["yc"], 1e-10) and pres(c["Iy"], g["Iy"], 1e-11) and pres(c["Iz"], g["Iz"], 1e-11)
    and abs(c["Iyz"] - g["Iyz"]) < 1e-6 and pres(c["Wpl_y"], g["Wpl_y"], 1e-9), (c["A"], g["A"]))

# ================================================================================================ 3 · sections composées, à la main
print("\n=== 3 · Sections composées, recalculées à la main")
# PRS en I : semelles 200 × 15, âme 10 × 300 (hauteur totale 330)
bf, tf, tw, hw = 200.0, 15.0, 10.0, 300.0
els = [plat(bf, tf, 0, (hw + tf) / 2, nom="semelle sup."), plat(tw, hw, nom="âme"), plat(bf, tf, 0, -(hw + tf) / 2, nom="semelle inf.")]
c = S.caracteristiques(els)
A_m = 2 * bf * tf + tw * hw
Iy_m = 2 * (bf * tf ** 3 / 12 + bf * tf * ((hw + tf) / 2) ** 2) + tw * hw ** 3 / 12
Iz_m = 2 * tf * bf ** 3 / 12 + hw * tw ** 3 / 12
Wpl_m = 2 * bf * tf * (hw + tf) / 2 + tw * hw ** 2 / 4
chk(f"PRS en I (2 × 200×15 + 10×300) : A = {A_m:.0f} mm², I_y = Σ (b·t³/12 + A·d²) + t_w·h_w³/12 = {Iy_m / 1e4:.1f} cm⁴, I_z, W_el = I_y/165, W_pl = {Wpl_m / 1e3:.1f} cm³",
    pres(c["A"], A_m) and pres(c["Iy"], Iy_m) and pres(c["Iz"], Iz_m) and pres(c["Wel_y"], Iy_m / 165.0) and pres(c["Wpl_y"], Wpl_m, 1e-9)
    and abs(c["zG"]) < 1e-9, (c["Iy"], Iy_m))
chk("PRS en I : le détail par élément est celui de Steiner (I propre + A·d² de chaque élément, somme = I_y)",
    pres(c["lignes"][0]["Iy"], bf * tf ** 3 / 12) and pres(c["lignes"][0]["dz"], (hw + tf) / 2) and pres(sum(li["Iy_G"] for li in c["lignes"]), c["Iy"]))
# caisson de quatre plats 300 × 200 extérieur, t = 10
B_, H_, t = 300.0, 200.0, 10.0
els = [plat(B_, t, 0, (H_ - t) / 2), plat(B_, t, 0, -(H_ - t) / 2), plat(t, H_ - 2 * t, (B_ - t) / 2, 0), plat(t, H_ - 2 * t, -(B_ - t) / 2, 0)]
c = S.caracteristiques(els)
chk("caisson de quatre plats (300 × 200 ext., t = 10) : I_y = (B·H³ − b·h³)/12, I_z = (H·B³ − h·b³)/12, A = B·H − b·h",
    pres(c["A"], B_ * H_ - (B_ - 2 * t) * (H_ - 2 * t)) and pres(c["Iy"], (B_ * H_ ** 3 - (B_ - 2 * t) * (H_ - 2 * t) ** 3) / 12)
    and pres(c["Iz"], (H_ * B_ ** 3 - (H_ - 2 * t) * (B_ - 2 * t) ** 3) / 12), (c["Iy"],))
c2 = S.caracteristiques([plat(B_, H_), plat(B_ - 2 * t, H_ - 2 * t, ev=True, nom="évidement")])
chk("le même caisson par un rectangle plein et un ÉVIDEMENT : mêmes A, I_y, I_z, W_pl", pres(c2["A"], c["A"]) and pres(c2["Iy"], c["Iy"]) and pres(c2["Iz"], c["Iz"])
    and pres(c2["Wpl_y"], c["Wpl_y"], 1e-8))
# HEA 200 + plat 220 × 12 soudé sur la semelle supérieure
hea, ph = S.profil("HEA 200")
e1 = S.element(hea, "HEA 200")
e2 = plat(220.0, 12.0, 0, 95.0 + 6.0, nom="plat")
c = S.caracteristiques([e1, e2])
A1, I1 = ph["A"] * 100, ph["Iy"] * 1e4
A2, I2 = 220.0 * 12.0, 220.0 * 12 ** 3 / 12
zG = A2 * 101.0 / (A1 + A2)
Iy_m = I1 + A1 * zG ** 2 + I2 + A2 * (101.0 - zG) ** 2
chk(f"HEA 200 + plat 220×12 sur la semelle : z_G = A₂·101/(A₁ + A₂) = {zG:.2f} mm, I_y = I₁ + A₁·z_G² + I₂ + A₂·(101 − z_G)² = {Iy_m / 1e4:.1f} cm⁴ "
    "(A₁, I₁ lus au catalogue)", abs(c["zG"] - zG) < 1e-3 and abs(c["Iy"] / Iy_m - 1) < 1e-5 and abs(c["fibres"]["z_sup"] - (107.0 - zG)) < 1e-3, (c["zG"], zG, c["Iy"], Iy_m))
# demi-HEA 200 (coupé à mi-hauteur) : la moitié de l'aire ; I_y(entier) = 2·(I_demi + A_demi·d²)
demi = S.element(S.couper(hea, "haut", 0.0), "demi-HEA")
cd = S.caracteristiques([demi])
chk("demi-HEA 200 (coupe à mi-hauteur, on garde le haut) : A/2 ; et 2·(I_y,demi + A_demi·z_G²) redonne I_y du profilé entier ; I_z/2",
    pres(cd["A"], e1["A"] / 2, 1e-9) and pres(2 * (cd["Iy"] + cd["A"] * cd["zG"] ** 2), e1["Iy"], 1e-9) and pres(cd["Iz"], e1["Iz"] / 2, 1e-9)
    and cd["zG"] > 0 and abs(cd["boite"][2]) < 1e-9, (cd["A"], cd["zG"]))
té = S.caracteristiques([S.element(S.couper(S.transformer(S.rectangle(100, 100), 0, False, 0, 0), "bas", 20.0), "bande")])
chk("coupe d'un carré 100 × 100 à z = 20, on garde le bas : rectangle 100 × 70 (A = 7 000, centre à z = −15)", pres(té["A"], 7000.0) and pres(té["zG"], -15.0))
# section en Z : semelles 80 × 10, âme 10 × 180 — I_yz ≠ 0, axes principaux
els = [plat(80, 10, 35, 95, nom="aile haute"), plat(10, 180, 0, 0, nom="âme"), plat(80, 10, -35, -95, nom="aile basse")]
c = S.caracteristiques(els)
Iyz_m = 2 * (800.0 * 35.0 * 95.0)
Iy_m = 2 * (80 * 1000 / 12 + 800 * 95 ** 2) + 10 * 180 ** 3 / 12
Iz_m = 2 * (10 * 80 ** 3 / 12 + 800 * 35 ** 2) + 180 * 1000 / 12
R_m = math.hypot((Iy_m - Iz_m) / 2, Iyz_m)
chk(f"section en Z : I_yz = 2·A·y·z = {Iyz_m / 1e4:.1f} cm⁴ ; I_1,2 = (I_y + I_z)/2 ± √(((I_y − I_z)/2)² + I_yz²) ; tan 2α = −2·I_yz/(I_y − I_z)",
    pres(c["Iyz"], Iyz_m) and pres(c["Iy"], Iy_m) and pres(c["Iz"], Iz_m) and pres(c["I1"], (Iy_m + Iz_m) / 2 + R_m) and pres(c["I2"], (Iy_m + Iz_m) / 2 - R_m)
    and pres(math.tan(2 * c["alpha"]), -2 * Iyz_m / (Iy_m - Iz_m)) and c["alpha"] < 0, (c["Iyz"], c["alpha_deg"]))
a_ = c["alpha"]
Iu = c["Iy"] * math.cos(a_) ** 2 + c["Iz"] * math.sin(a_) ** 2 - c["Iyz"] * math.sin(2 * a_)
Iuv = (c["Iy"] - c["Iz"]) / 2 * math.sin(2 * a_) + c["Iyz"] * math.cos(2 * a_)
chk("section en Z : la rotation directe des inerties à l'angle α retrouve I_1 et annule le produit d'inertie", pres(Iu, c["I1"]) and abs(Iuv) < 1e-6 * c["I1"], (Iu, c["I1"], Iuv))
# rond évidé = tube
c = S.caracteristiques([S.element(S.rond(100.0), "rond"), S.element(S.rond(80.0), "trou", True)])
chk("rond Ø 100 évidé d'un Ø 80 : I = π·(D⁴ − d⁴)/64 à 5·10⁻⁴ (arcs discrétisés)", abs(c["Iy"] / (math.pi * (100 ** 4 - 80 ** 4) / 64) - 1) < 5e-4, c["Iy"])

# ================================================================================================ 4 · invariants
print("\n=== 4 · Invariants")
base = [S.profil("UPN 200")[0], S.rectangle(150, 12), S.polygone([(0, 0), (60, 0), (0, 90)])]
poses = [(0, False, 0, 0), (0, False, 40, 130), (0, False, -80, -60)]


def section(angle, miroir, dy, dz):
    els_ = []
    for r_, (a0, m0, y0, z0) in zip(base, poses):
        loc = S.transformer(r_, a0, m0, y0, z0)
        els_.append(S.element(S.transformer(loc, angle, miroir, dy, dz)))
    return S.caracteristiques(els_)


c0 = section(0, False, 0, 0)
ct = section(0, False, 123.4, -56.7)
chk("translation : A, I_y, I_z, I_yz, W_pl inchangés ; le centre de gravité suit", pres(ct["Iy"], c0["Iy"], 1e-9) and pres(ct["Iz"], c0["Iz"], 1e-9)
    and pres(ct["Iyz"], c0["Iyz"], 1e-8) and pres(ct["yG"], c0["yG"] + 123.4, 1e-9) and pres(ct["Wpl_y"], c0["Wpl_y"], 1e-7))
for ang in (30.0, 90.0, 137.0):
    cr = section(ang, False, 0, 0)
    chk(f"rotation de {ang:.0f}° : A, I_1, I_2 et I_y + I_z conservés ; l'angle des axes principaux tourne d'autant",
        pres(cr["A"], c0["A"]) and pres(cr["I1"], c0["I1"], 1e-9) and pres(cr["I2"], c0["I2"], 1e-9) and pres(cr["Iy"] + cr["Iz"], c0["Iy"] + c0["Iz"], 1e-10)
        and abs(((cr["alpha_deg"] - c0["alpha_deg"] - ang) + 90) % 180 - 90) < 1e-6, (cr["alpha_deg"], c0["alpha_deg"]))
cm = section(0, True, 0, 0)
chk("miroir : I_y, I_z conservés, I_yz change de signe, y_G change de signe", pres(cm["Iy"], c0["Iy"], 1e-10) and pres(cm["Iz"], c0["Iz"], 1e-10)
    and pres(cm["Iyz"], -c0["Iyz"], 1e-9) and pres(cm["yG"], -c0["yG"], 1e-9))
c90 = section(90.0, False, 0, 0)
chk("rotation de 90° : I_y et I_z s'échangent", pres(c90["Iy"], c0["Iz"], 1e-9) and pres(c90["Iz"], c0["Iy"], 1e-9))

# ================================================================================================ 5 · recouvrements, refus
print("\n=== 5 · Recouvrements, refus")
r_ = S.recouvrements([plat(100, 100), plat(100, 100, 50, 50)])
chk("deux carrés 100 × 100 décalés de (50 ; 50) : recouvrement détecté, aire ≈ 2 500 mm²", len(r_) == 1 and r_[0]["sorte"] == "double" and abs(r_[0]["aire"] - 2500) < 30, r_)
chk("deux plats qui se TOUCHENT (bord à bord) : aucun recouvrement signalé", S.recouvrements([plat(100, 10, 0, 5), plat(100, 10, 0, -5)]) == [])
tube = S.element(S.profil("CHS 114.3x5")[0], "tube")
chk("un rond Ø 60 DANS le vide d'un tube Ø 114,3 : aucun recouvrement (le trou du tube est respecté)", S.recouvrements([tube, S.element(S.rond(60.0), "rond")]) == [])
chk("un rond Ø 110 dans le même tube : il mord la paroi — recouvrement détecté", len(S.recouvrements([tube, S.element(S.rond(110.0), "rond")])) == 1)
chk("un évidement entièrement dans la matière : rien à signaler", S.recouvrements([plat(100, 100), S.element(S.rond(40.0), "trou", True)]) == [])
r_ = S.recouvrements([plat(100, 100), S.element(S.transformer(S.rond(40.0), 0, False, 50, 0), "trou", True)])
chk("un évidement à cheval sur le bord : sa partie hors matière est signalée (la moitié du disque ≈ 628 mm²)",
    len(r_) == 1 and r_[0]["sorte"] == "evidement_hors" and abs(r_[0]["aire"] - math.pi * 400 / 2) < 15, r_)
for nom, fn in (("section vide", lambda: S.caracteristiques([])), ("polygone de deux sommets", lambda: S.polygone([(0, 0), (1, 1)])),
                ("profilé inconnu", lambda: S.profil("IPE 9999")), ("côté de coupe inconnu", lambda: S.couper(S.rectangle(1, 1), "diagonale", 0)),
                ("élément entièrement coupé", lambda: S.element(S.couper(S.rectangle(10, 10), "haut", 50.0), "x")),
                ("évidement plus grand que la matière", lambda: S.caracteristiques([plat(10, 10), plat(20, 20, ev=True)]))):
    try:
        fn()
        chk(f"refus : {nom}", False, "accepté")
    except S.ErreurDonnees as ex:
        chk(f"refus : {nom} ({ex})", True)
chk("ce qui n'est pas calculé est NOMMÉ avec sa raison : centre de cisaillement, torsion, gauchissement",
    set(S.NON_CALCULES) == {"centre_cisaillement", "torsion", "gauchissement"} and all(len(v) > 80 for v in S.NON_CALCULES.values()))

# ================================================================================================ 6 · mutations
print("\n=== 6 · Mutations")
src = open(os.path.join(RACINE, "rdm", "sections.py"), encoding="utf-8").read()


def controle(M):
    """Les contrôles à la main rejoués sur un moteur donné : (PRS, Z, triangle, demi-profilé)."""
    def pl(b, h, y=0.0, z=0.0):
        return M["element"](M["transformer"](M["rectangle"](b, h), 0.0, False, y, z))
    c_ = M["caracteristiques"]([pl(200, 15, 0, 157.5), pl(10, 300), pl(200, 15, 0, -157.5)])
    z_ = M["caracteristiques"]([pl(80, 10, 35, 95), pl(10, 180), pl(80, 10, -35, -95)])
    t_ = M["caracteristiques"]([M["element"](M["polygone"]([(0, 0), (60, 0), (0, 90)]))])
    d_ = M["caracteristiques"]([M["element"](M["couper"](M["rectangle"](100, 100), "bas", 20.0))])
    return (pres(c_["Iy"], 2 * (200 * 15 ** 3 / 12 + 3000 * 157.5 ** 2) + 10 * 300 ** 3 / 12) and pres(c_["Wpl_y"], 3000 * 315 + 10 * 300 ** 2 / 4, 1e-8)
            and pres(z_["Iyz"], 2 * 800 * 35 * 95) and pres(z_["I1"], (z_["Iy"] + z_["Iz"]) / 2 + math.hypot((z_["Iy"] - z_["Iz"]) / 2, z_["Iyz"]))
            and pres(math.tan(2 * z_["alpha"]), -2 * z_["Iyz"] / (z_["Iy"] - z_["Iz"]))
            and pres(t_["Iyz"], -(60 ** 2) * 90 ** 2 / 72) and pres(t_["Iy"], 60 * 90 ** 3 / 36) and pres(d_["A"], 7000.0) and pres(d_["zG"], -15.0))


chk("le contrôle des mutations est vert sur le moteur réel", controle({k: getattr(S, k) for k in ("element", "transformer", "rectangle", "caracteristiques", "polygone", "couper")}))
for nom, avant, apres in (
        ("Green : ∫z² dA divisé par 6 au lieu de 12", '"Jzz": Jzz / 12', '"Jzz": Jzz / 6'),
        ("Green : produit d'inertie divisé par 12 au lieu de 24", '"Jyz": Jyz / 24', '"Jyz": Jyz / 12'),
        ("Steiner : A·d² oublié sur I_y", 'li["Iy_G"] = li["Iy"] + li["A"] * li["dz"] ** 2', 'li["Iy_G"] = li["Iy"]'),
        ("Steiner : produit d'inertie sans le terme A·dy·dz", 'li["Iyz_G"] = li["Iyz"] + li["A"] * li["dy"] * li["dz"]', 'li["Iyz_G"] = li["Iyz"]'),
        ("axes principaux : signe de I_yz perdu", "alpha = 0.0 if isotrope else 0.5 * math.atan2(-2.0 * Iyz, Iy - Iz)", "alpha = 0.0 if isotrope else 0.5 * math.atan2(2.0 * Iyz, Iy - Iz)"),
        ("inertie propre : I_y sans le retour au centre de gravité", '"Iy": t["Jzz"] - A * zg * zg,', '"Iy": t["Jzz"],'),
        ("coupe : le côté gardé inversé", 'sup = cote in ("haut", "droite")', 'sup = cote in ("bas", "gauche")'),
        ("module plastique : demi-aire remplacée par l'aire", "if aire_sup(mid) > A / 2:", "if aire_sup(mid) > A:")):
    assert src.count(avant) == 1, nom
    espace = {"__name__": "rdm_sections_mute"}
    exec(compile(src.replace(avant, apres), "sections_mute.py", "exec"), espace)
    try:
        vu = not controle(espace)
    except Exception:
        vu = True
    chk(f"mutation « {nom} » : vue par les contrôles à la main", vu)

import subprocess  # noqa: E402

sonde = ("import sys\nclass R:\n    def find_spec(self, n, p=None, t=None):\n"
         "        if n.split('.')[0] in ('numpy','scipy','streamlit','matplotlib','reportlab','pandas'):\n"
         "            raise ImportError(n)\nsys.meta_path.insert(0, R())\n"
         f"sys.path.insert(0, {RACINE!r})\nfrom rdm import sections as S\n"
         "print(S.caracteristiques([S.element(S.rectangle(200.0, 10.0))])['Iy'])")
rr = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True)
chk("le cœur (formes simples, Steiner, axes principaux, coupes) charge et calcule sans numpy ; seul le contour d'un profilé du catalogue le demande",
    rr.returncode == 0 and abs(float(rr.stdout.strip()) - 200 * 1000 / 12) < 1e-6, rr.stderr[-300:])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)
