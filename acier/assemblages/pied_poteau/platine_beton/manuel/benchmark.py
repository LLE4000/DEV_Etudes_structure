# -*- coding: utf-8 -*-
"""Benchmark du calcul manuel — des CALCULS À LA MAIN, formule par formule,
rejoués par le moteur à chaque ouverture (les mêmes que
tests/test_platine_manuelle.py) : suspente sur boulons d'angle, boulons
devant l'aile, compression centrée, pression diamétrale. Seuil : écart
≤ 0,5 %, sinon « À CONTRÔLER ». Aucun exemple publié : le calcul manuel
d'une platine libre n'a pas de fiche de référence dans les guides."""
import math

from .moteur import compute

_CORD = 0.8 * 8 * math.sqrt(2)                       # cordon a = 8 mm
_m1 = math.hypot(50, 55) - _CORD                     # boulon (−200 ; −200) → angle (−150 ; −145)
_Mpl1 = 0.25 * _m1 * 25 ** 2 * 345
_e1 = 50 / (55 / math.hypot(50, 55))
_n1 = min(_e1, 1.25 * _m1)
_FT3 = min(0.9 * 800 * 353 / 1.25, 353 * 800 / 1.5)
_FT2 = (2 * _Mpl1 + _n1 * _FT3) / (_m1 + _n1)
_tT1 = math.sqrt(25e3 / 355)
_b_ab = 14 + 2 * (50 - _CORD)
_tS1 = math.sqrt(4 * 25e3 * (50 - _CORD) / (_b_ab * 355))
_m2 = 215 - 145 - _CORD
_cp2 = min(math.pi * _m2, math.pi * _m2 / 2 + 100, math.pi * _m2 / 2 + 150)
_nc2 = min(2 * _m2 + 0.625 * 35, 75 + _m2 + 0.3125 * 35, 50 + _m2 + 0.3125 * 35)
_b2 = 300 + 2 * _m2
_tS2 = math.sqrt(4 * 2 * 25e3 * _m2 / (_b2 * 355))
_tT2 = math.sqrt(25e3 * _m2 / (min(_cp2, _nc2) * 355))
_fjd = 2 / 3 * (0.85 * 30 / 1.5) * math.sqrt(900 * 900 / (500 * 500))
_c3 = (-1763 + math.sqrt(1763 ** 2 + 16 * (2000e3 / _fjd - 10627))) / 8
_tc3 = _c3 / math.sqrt(345 / (3 * _fjd))
# Platine S355 : fu = 510 MPa (EN 1993-1-1:2005 Tableau 3.1, la table des aciers du module — pas 490, valeur
# d'autres références). Le harnais d'écran l'a attrapé : à 490, Fb,Rd sortait 4 % trop bas et le benchmark
# disait « À CONTRÔLER ».
_FU4 = 510
_ab4 = min(50 / (3 * 26), 800 / _FU4, 1.0)
_k14 = min(2.8 * 50 / 26 - 1.7, 1.4 * 400 / 26 - 1.7, 2.5)
_Fb4 = _k14 * _ab4 * _FU4 * 24 * 25 / 1.25 / 1e3
_tb4 = 50e3 * 1.25 / (_k14 * _ab4 * _FU4 * 24)

CAS = [
    dict(id="VAL-M1", titre="Suspente N = −100 kN, quatre boulons d'angle (±200 ; ±200), HEA 300, platine 500×500×25 S355",
         data="T = N/4 ; angle : bras diagonal − cordon, leff = m, effet de levier (Lb ≤ Lb*), mode 2 ; "
              "épaisseur tT = √(T/fy) en mode 1 ; section de l'about d'aile b = 14 + 2·m⊥.",
         inp=dict(N_Ed=-100, Mx_Ed=0, My_Ed=0, Vy_Ed=0), rows=[
             ["T de chaque boulon = N/4", 25.0, lambda R: R.tiges[0]["T"], "kN"],
             ["Bras m = √(50² + 55²) − 0,8·a·√2", _m1, lambda R: R.tiges[0]["m"], "mm"],
             ["leff,1 = m (angle : modèle d'ingénieur)", _m1, lambda R: R.tiges[0]["leff1"], "mm"],
             ["Mpl,1 = 0,25·leff·tp²·fy", _Mpl1 / 1e6, lambda R: R.tiges[0]["Mpl1"], "kNm"],
             ["FT,Rd = (2·Mpl + n·Ft)/(m + n) (mode 2)", _FT2 / 1e3, lambda R: R.tiges[0]["FT_Rd"], "kN"],
             ["tT = √(T·m/(leff·fy)) = √(T/fy)", _tT1, lambda R: R.plaque["t_T"], "mm"],
             ["tS (about d'aile) = √(4·T·m⊥/(b·fy))", _tS1, lambda R: R.plaque["t_S"], "mm"]]),
    dict(id="VAL-M2", titre="Suspente N = −100 kN, boulons devant les ailes (±100 ; ±215)",
         data="m = 215 − 145 − cordon ; leff,1 = min(motifs circulaire et non circulaire, L/2 + m) ; section au nu "
              "de l'aile b = 300 + 2·m, M = 2·T·m ; tS = √(4·M/(b·fy)) gouverne.",
         inp=dict(tiges=[dict(x=-100, y=-215), dict(x=100, y=-215), dict(x=-100, y=215), dict(x=100, y=215)],
                  N_Ed=-100, Mx_Ed=0, My_Ed=0, Vy_Ed=0), rows=[
             ["Bras m = 215 − 145 − 0,8·a·√2", _m2, lambda R: R.tiges[0]["m"], "mm"],
             ["leff,1 = min(cp ; nc ; L/2 + m)", min(_cp2, _nc2), lambda R: R.tiges[0]["leff1"], "mm"],
             ["Section : b = 300 + 2·m", _b2, lambda R: R.sections[0]["b"], "mm"],
             ["Section : M = 2·T·m", 2 * 25e3 * _m2 / 1e6, lambda R: R.sections[0]["M"], "kNm"],
             ["tS = √(4·M/(b·fy))", _tS2, lambda R: R.plaque["t_S"], "mm"],
             ["tT = √(T·m/(leff·fy))", _tT2, lambda R: R.plaque["t_T"], "mm"]]),
    dict(id="VAL-M3", titre="Compression centrée N = 2 000 kN, HEA 300, bande c",
         data="L'épaisseur requise côté comprimé est celle dont la bande c porte N à fjd : Aeff(c) = N/fjd → c = 45,0 "
              "→ tp = c/√(fy/(3·fjd)) ; pression uniforme → bande m = p·c²/2.",
         inp=dict(N_Ed=2000, Mx_Ed=0, My_Ed=0, Vy_Ed=0), rows=[
             ["fjd = βj·fcd·√(Ac1/Ac0)", _fjd, lambda R: R.fjd, "MPa"],
             ["tc : Aeff(c) = N/fjd", _tc3, lambda R: R.plaque["t_c"], "mm"],
             ["bande m = p·c²/2 (à tp = 25)", 2000e3 / (2 * (300 + 2 * 25 * math.sqrt(345 / (3 * _fjd))) * (14 + 2 * 25 * math.sqrt(345 / (3 * _fjd)))
                                              + (8.5 + 2 * 25 * math.sqrt(345 / (3 * _fjd))) * (262 - 2 * 25 * math.sqrt(345 / (3 * _fjd))))
              * (25 * math.sqrt(345 / (3 * _fjd))) ** 2 / 2 / 1e3, lambda R: R.console["m_Ed"], "kNm/m"]]),
    dict(id="VAL-M4", titre="Cisaillement Vy = 200 kN, N = 0 : pression diamétrale (Tableau 3.4)",
         data="50 kN par boulon ; boulon vers le bord : e1 = 50, p1 = ∞, e2 = 50, p2 = 400 → αb = 0,64, k1 = 2,5 ; "
              "fu = 510 MPa (S355) ; Fb,Rd = k1·αb·fu·d·tp/γM2 = 392,3 kN ; tb = Fv·γM2/(k1·αb·fu·d) = 3,19 mm.",
         inp=dict(N_Ed=0, Mx_Ed=0, My_Ed=0, Vx_Ed=0, Vy_Ed=200), rows=[
             ["Fv par boulon", 50.0, lambda R: R.appuis[2]["Fv"], "kN"],
             ["αb = min(e1/(3d0) ; fub/fu ; 1)", _ab4, lambda R: R.appuis[2]["ab"], "-"],
             ["k1 = min(2,8·e2/d0 − 1,7 ; 1,4·p2/d0 − 1,7 ; 2,5)", _k14, lambda R: R.appuis[2]["k1"], "-"],
             ["Fb,Rd", _Fb4, lambda R: R.appuis[2]["Fb"], "kN"],
             ["tb", _tb4, lambda R: R.appuis[2]["t_b"], "mm"]]),
]


def run_bench_manuel():
    """Rejoue les cas ; ``cas`` (lignes avec écart et statut), ``ok``, ``statut``."""
    out, nko = [], 0
    for c in CAS:
        R = compute(c["inp"])
        rows = []
        for lab, main, fn, unit in c["rows"]:
            v = fn(R)
            ec = (v - main) / main if main else abs(v)
            ok = abs(ec) <= 0.005
            nko += 0 if ok else 1
            rows.append(dict(lab=lab, main=main, val=v, unit=unit, ec=ec, ok=ok))
        out.append(dict(id=c["id"], titre=c["titre"], data=c["data"], rows=rows))
    return dict(cas=out, ok=nko == 0, statut="VALIDÉ (calculs manuels)" if nko == 0 else "À CONTRÔLER")
