# -*- coding: utf-8 -*-
"""Benchmark du pied de poteau — V1 (lot A) : CALCULS MANUELS INDÉPENDANTS,
formule par formule, sur le cas par défaut (VAL-A) et reproduction du script
autonome de l'étude du 30/10/2026 (VAL-B, massif large : fjd = 34 MPa) —
l'étude avait chiffré la répartition élastique et le domaine sans ce module.

AUCUN EXEMPLE PUBLIÉ n'est encore rejoué (BENCH vide) : l'exemple de pied de
poteau du guide « Steel Buildings in Europe, Part 5 » et celui de SCI P398
seront rejoués au lot B, avec les chiffres de la source sous les yeux — aucun
nombre n'est reconstitué de mémoire. Seuils : écart ≤ 0,5 % sur un calcul
manuel (formules exactes), sinon « À CONTRÔLER ».
"""
import math

from acier.js import mx
from .moteur import compute

PI = math.pi


def rd(k):
    return lambda R: R.ck[k].Rd


def ed(k):
    return lambda R: R.ck[k].Ed


def va(k):
    return lambda R: R[k]


SRC = dict(
    titre="Aucun exemple publié rejoué en V1 — calculs manuels (VAL-A, VAL-B)",
    org="Bureau d'Études Valens — étude de méthode du 30/10/2026 (docs/assemblages/PIED_POTEAU_ETUDE.md)",
    annee="2026",
    norme="EN 1993-1-8:2005 §6.2.5, §6.2.4, §6.2.8, §6.2.2 ; EN 1992-1-1 §6.7 ; EN 1992-4:2018 §7.2 ; "
          "γM0 = 1,00 (ANB), γM2 = 1,25, γMc = 1,5, αcc = 0,85 (hypothèse)",
    url="")

BENCH = []

NOCOVER = [
    ["Exemple publié (pied articulé, aire efficace — MSB Part 5 ; pied encastré N + M — SCI P398)",
     "À rejouer au lot B avec le document sous les yeux ; en V1 les formules sont ancrées par VAL-A "
     "(calcul manuel) et la répartition par VAL-B (script indépendant de l'étude)."],
    ["Flexion biaxiale (niveau 2, domaine exact) et tiges de coin",
     "Aucune source publiée : borne inférieure (théorème statique) — à valider par éléments finis hors outil "
     "avant promotion (campagne RV, lot C)."],
    ["Tiges post-installées (valeurs ETA), raidisseurs, bêche",
     "Hors V1."],
]

# --- VAL-A : cas par défaut, recalculé à la main (N, mm, MPa)
_fcd = 0.85 * 30 / 1.5
_ratio = math.sqrt(900 * 900 / (500 * 500))                 # Ac0 = platine, Ac1 = 900 × 900 (massif, 3·Ac0 et hc non atteints)
_fjd = 2 / 3 * _fcd * _ratio
_c = 25 * math.sqrt(345 / (3 * _fjd * 1.0))
_A_eff = 2 * (300 + 2 * _c) * (14 + 2 * _c) + (8.5 + 2 * _c) * (290 - 2 * 14 - 2 * _c)
_NRd = _fjd * _A_eff / 1000
_m = 200 - 145 - 0.8 * 8 * math.sqrt(2)
_leff1 = 2 * 0.25 * 300                                     # tiges de coin : plafond 0,25·b par tige
_Mpl1 = 0.25 * _leff1 * 25 ** 2 * 345 / 1.0
_Lb = 8 * 24 + 30 + 25 + 0.6 * 24
_Lb_star = 8.8 * _m ** 3 * 353 / (_leff1 * 25 ** 3)
_Ft_EC3 = 0.9 * 800 * 353 / 1.25 / 1000
_NRd_s = 353 * 800 / max(1.4, 1.2 * 800 / 640) / 1000
_FT_row = min(2 * _Mpl1 / _m / 1000, 2 * min(_Ft_EC3, _NRd_s))   # Lb > Lb* : mode 1-2 sans effet de levier
_F2vb = (0.44 - 0.0003 * 640) * 800 * 353 / 1.25 / 1000
_N0 = 7.7 * math.sqrt(30) * 300 ** 1.5
_cone = _N0 * (700 * 700) / (900 * 900) * (0.7 + 0.3 * 250 / 450) / 1.5 / 1000      # une tige tendue à (−200 ; −200)
_pull = 7.5 * PI / 4 * (40 ** 2 - 24 ** 2) * 30 / 1.5 / 1000
_lf = min(300, 12 * 24)
_alpha, _beta = 0.1 * math.sqrt(_lf / 250), 0.1 * (24 / 250) ** 0.2
_V0 = 1.7 * 24 ** _alpha * _lf ** _beta * math.sqrt(30) * 250 ** 1.5
_edge = _V0 * (900 * 375) / (4.5 * 250 ** 2) * (0.7 + 0.3 * 250 / 375) / 1.5 / 1000
_pry = 2 * _N0 * 1.0 * (0.7 + 0.3 * 250 / 450) / 1.5 / 1000
_FC = _fjd * (300 + 2 * _c) * (14 + 2 * _c) / 1000
_zC = 145 - 7
_M1x = ((300 + _FT_row) * _zC + _FT_row * 200) / 1000           # tiges à leur résistance (N + ΣT ≤ FC,Rd)
_zCy = (150 + _c) / 2
_M1y = ((300 + _FT_row) * _zCy + _FT_row * 200) / 1000

VALID = [
    dict(id="VAL-A", titre="Cas par défaut (HEA 300, platine 500×500×25 S355, 4 × M24 8.8 en coin, C30/37, "
                           "massif 900×900×600, N = 300 kN, Mx = 50 kNm, My = 30 kNm, Vy = 40 kN)",
         inp={}, rows=[
        ["fjd = βj·fcd·√(Ac1/Ac0), Ac0 = platine, Ac1 = massif", _fjd, 20.40, va("fjd"), "MPa"],
        ["c = tp·√(fy/(3·fjd·γM0)), fy = 345 (tp = 25 > 16)", _c, 59.36, va("c"), "mm"],
        ["Aeff = 2 ailes + âme élargies de c (dans la platine)", _A_eff, 129367, va("A_eff"), "mm²"],
        ["NRd = fjd·Aeff", _NRd, 2639.1, rd("ncp"), "kN"],
        ["Tronçon en T de coin : m = 200 − 145 − 0,8·a·√2", _m, 45.95, lambda R: R.rangees[0]["m"], "mm"],
        ["Σleff,1 plafonnée à 0,25·b par tige (coin)", _leff1, 150, lambda R: R.rangees[0]["leff1"], "mm"],
        ["Lb* = 8,8·m³·As/(Σleff,1·tp³) (< Lb = 261 : sans effet de levier)", _Lb_star, 128.6,
         lambda R: R.rangees[0]["Lb_star"], "mm"],
        ["FT,Rd rangée = min(2·Mpl,1/m ; 2·Ft,anc)", _FT_row, 351.96, lambda R: R.rangees[0]["FT_Rd"] / 1000, "kN"],
        ["Ft,Rd = 0,9·fub·As/γM2 (Tableau 3.4)", _Ft_EC3, 203.33, lambda R: R.Ft_EC3 / 1000, "kN"],
        ["NRd,s = As·fuk/γMs,N, γMs,N = 1,2·fuk/fyk = 1,5 (EN 1992-4)", _NRd_s, 188.27, rd("ta"), "kN"],
        ["F2,vb,Rd = αb·fub·As/γM2, αb = 0,44 − 0,0003·fyb", _F2vb, 56.03, lambda R: R.F2vb / 1000, "kN"],
        ["Cône : N⁰Rk,c·(Ac,N/A⁰c,N)·ψs,N/γMc (une tige tendue, c = 250, Ac,N = 700²)", _cone, 76.60, rd("cone"), "kN"],
        ["Arrachement : k2·Ah·fck/γMc (k2 = 7,5, dh = 40)", _pull, 120.64, rd("pull"), "kN"],
        ["Bord : V⁰Rk,c·(Ac,V/A⁰c,V)·ψs,V/γMc (c1 = 250, rangée de 2 à 400)", _edge, 53.13, rd("edge"), "kN"],
        ["Pry-out : k8·NRk,c(groupe)/γMc (k8 = 2)", _pry, 253.23, rd("pry"), "kN"],
        ["Niveau 1, axe x : (N + ΣT)·zC + ΣT·zT, zC = h/2 − tf/2", _M1x, 160.36, lambda R: R.MRd1["x+"], "kNm"],
        ["Niveau 1, axe y : zC = (b/2 + c)/2 (demi-bandes d'aile)", _M1y, 138.64, lambda R: R.MRd1["y+"], "kNm"]]),
    dict(id="VAL-B", titre="Répartition élastique de l'étude (script autonome du 30/10/2026) — massif large, "
                           "fjd = 34 MPa, c = 46 mm, kc = Ecm/(1,275·√(beff·leff)), kb = 2·E·As/Lb",
         inp=dict(bc_u=3000, lc_u=3000, hc_u=2000), rows=[
        ["fjd (√(Ac1/Ac0) plafonné à 3)", 34.0, 34.0, va("fjd"), "MPa"],
        ["c", 46.0, 46.0, lambda R: round(R.c, 1), "mm"],
        ["Pression maximale (coin comprimé)", 14.9, 14.9, lambda R: round(R.pmax, 1), "MPa"],
        ["Résultante comprimée C", 349.6, 349.6, lambda R: round(R.C, 1), "kN"],
        ["Position xC", 57.0, 57.0, lambda R: round(R.xC), "mm"],
        ["Position yC", 115.0, 115.0, lambda R: round(R.yC), "mm"],
        ["Tige tendue (−200 ; −200)", 49.6, 49.6, lambda R: round(R.tiges[0]["T"], 1), "kN"],
        ["Les trois autres tiges", 0.0, 0.0, lambda R: round(sum(t["T"] for t in R.tiges[1:]), 3), "kN"]]),
]


def run_bench():
    """Rejoue les exemples publiés (aucun en V1) et les calculs manuels ;
    retourne ``bench``, ``valid``, ``worst``, ``ok`` et ``statut``."""
    out, worst, n_unexp = [], 0.0, 0
    for b in BENCH:
        R = compute(b["inp"])
        rows = []
        for r in b["rows"]:
            v = r[3](R)
            ec = (v - r[2]) / r[2] if r[2] else 0.0
            a = abs(ec)
            st = "OK" if a <= 0.01 else ("Écart expliqué" if r[5] else "À expliquer") if a <= 0.03 \
                else ("Écart > 3 % documenté" if r[5] else "À vérifier")
            if a > 0.01 and not r[5]:
                n_unexp += 1
            worst = mx(worst, a)
            rows.append(dict(page=r[0], lab=r[1], ref=r[2], val=v, unit=r[4], ec=ec, st=st, exp=r[5]))
        out.append(dict(id=b["id"], titre=b["titre"], data=b["data"], rows=rows))
    nv, val = 0, []
    for c in VALID:
        R = compute(c["inp"])
        rows = []
        for lab, main, ref, fn, unit in c["rows"]:
            v = fn(R)
            ec = (v - main) / main if main else abs(v)
            ok = abs(ec) <= 0.005
            nv += 0 if ok else 1
            worst = mx(worst, abs(ec))
            rows.append(dict(lab=lab, main=main, ref=ref, val=v, unit=unit, ec=ec, ok=ok))
        val.append(dict(id=c["id"], titre=c["titre"], rows=rows))
    ok = n_unexp == 0 and nv == 0
    statut = ("VALIDÉ" if BENCH else "VALIDÉ (calculs manuels, aucun exemple publié)") if ok else "À CONTRÔLER"
    return dict(bench=out, valid=val, worst=worst, ok=ok, statut=statut)
