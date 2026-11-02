# -*- coding: utf-8 -*-
"""Benchmark du module doubles cornières sur poteau — quatre cas, et aucun
nombre de mémoire.

- **BM1 — HÉRITAGE** : l'exemple publié du guide (MSB Part 5 §4.4, *Worked
  Example: Angle Web Cleats*) rejoué avec ``appui = Aile du poteau`` et un
  poteau PERSO d'aile 15 mm S275. Les SEIZE valeurs publiées doivent sortir
  INCHANGÉES : du côté cornières, boulons et âme portée, rien ne dépend du
  support (démontré par l'étude, script 00). C'est le contrôle d'héritage.
- **BM1'** : le TYING du même exemple publié (p. 5-78/79) — l'ANCRE des
  modes 1 / 2 / 3 du tronçon en T, la formule même que l'aile du poteau
  réutilise. Le guide applique fu/γMu : reproduit avec un acier de cornière
  PERSO fy = fu = 430 MPa et γM0 = γM2 = 1,1, comme le module poutre–poutre.
- **BM2** : le cas par défaut, appui sur l'AILE (étude §11.4).
- **BM3** : appui sur l'AILE en TYING (NEd = 300 kN) — le tronçon en T des
  ailes A et celui de l'AILE DU POTEAU, côte à côte : **les cornières
  gouvernent** (rapport 1,915).
- **BM4** : appui dans l'ÂME — une L100 NE PASSE PAS (104,3 > 104 mm), une
  **L90×90×9** passe ; pression diamétrale et cisaillement local de l'âme.

Chaque valeur de référence est écrite ici comme une EXPRESSION LITTÉRALE
indépendante (jamais une constante recopiée) : le moteur est relancé et
comparé à elle. Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à
vérifier. Un écart qui porte une explication (``exp``) ne compte pas comme
inattendu. Aucune formule n'a été ajustée pour coller à la source.
"""
import math

from acier.bibliotheques import PERSO
from acier.js import dv, mx
from .entrees import APPUI_AILE, APPUI_AME, NAT_TYING, SOUDEE
from .moteur import compute, leff_aile_poteau

S3 = math.sqrt(3)

E_ARR = ("Arrondis intermédiaires de la source (αb = 0,61 pour 0,606 ; Fv,Rd = 94 pour 94,08 ; "
         "m = 31 pour 31,2).")
E_AIRE = ("La source utilise l'aire catalogue A = 11700 mm² ; l'outil calcule A = 2b·tf + (h − 2tf)·tw "
          "+ (4 − π)·r² = 11729 mm² (+0,25 %), d'où Av = 6030 au lieu de 6001 mm².")

# --- les données de l'exemple publié, reprises telles quelles du module
# --- poutre–poutre, SUPPORT remplacé : un poteau PERSO dont l'aile a
# --- l'épaisseur de l'âme support de l'exemple (15 mm) et sa nuance (S275).
# --- Aucune des seize valeurs publiées ne dépend du support ; hP, bP, twP et
# --- rP ne servent qu'au dessin, au gauge minimal et — en cas (b) — au
# --- cisaillement local, inactif ici.
WE = dict(appui=APPUI_AILE,
          prof_S=PERSO, hS_u=547, bS_u=210, twS_u=9, tfS_u=15.7, rS_u=24, nu_S="S275", nu_C="S275",
          prof_P=PERSO, hP_u=300, bP_u=300, twP_u=11, tfP_u=15, rP_u=27, nu_P="S275",
          d_nt=0, d_nb=0, l_n=0, g_h=10, boulon_u="M20", classe="8.8", z_C=50, e2b_u=40,
          corn_u=PERSO, k1_u=90, k2_u=90, kt_u=10, kr_u=11, V_Ed=450, LC_u=430,
          n1S_u=6, n2S_u=1, p1S_u=70, e1S_u=40, n1P_u=6, n2P_u=1, p1P_u=70, e1P_u=40, gA_u=50)
# tying de l'exemple publié : l'ancre des modes 1 / 2 / 3
WE_T = dict(WE, N_Ed=370, nu_C=PERSO, fy_u=430, fu_u=430, bw_u=0.85, g_M0=1.1, g_M2=1.1)

# --- BM2 : le cas par défaut (appui sur l'aile) — aucune entrée à forcer
BM2 = {}
# --- BM3 : appui sur l'aile, TYING NEd = 300 kN
BM3 = dict(N_Ed=300, nat_N=NAT_TYING)
# --- BM4 : appui dans l'âme, L90×90×9 (une L100 ne passe pas)
BM4 = dict(appui=APPUI_AME, corn_u="L90x90x9")


def vr(k):
    """Résistance du groupe exprimée en VEd : VEd / η de la vérification k."""
    return lambda R: dv(R.u.V_Ed, R.ck[k].eta)


def rd(k):
    return lambda R: R.ck[k].Rd


def et(k):
    return lambda R: R.ck[k].eta


def va(k):
    return lambda R: R[k]


# =====================================================================
#  Calculs manuels indépendants (expressions littérales, hors moteur)
# =====================================================================
def _troncon(leff, t, m, n, ew, f, gamma, SFt):
    """Modes 1 (méthode 2), 2 et 3 du tronçon en T, réécrits ici in extenso —
    la valeur de référence ne vient PAS de la fonction du moteur."""
    Mpl = 0.25 * leff * t * t * f / gamma                       # N·mm
    FT1 = (8 * n - 2 * ew) * Mpl / (2 * m * n - ew * (m + n)) / 1000
    FT2 = (2 * Mpl + n * SFt * 1000) / (m + n) / 1000
    return Mpl / 1e6, FT1, FT2, SFt


# --- géométrie et matériaux de référence (étude §11.1)
_fy, _fu, _bw = 355.0, 510.0, 0.9                 # S355
_d, _d0, _As, _dm, _dw = 20.0, 22.0, 245.0, 31.8, 37.0        # M20
_fub, _av = 800.0, 0.6                            # classe 8.8, filet cisaillé
_Fv = _av * _fub * _As / 1.25 / 1000              # 94,08 kN
_Fvu = _av * _fub * _As / 1.1 / 1000              # tying, γMu = 1,1
_Ftu = 0.9 * _fub * _As / 1.1 / 1000              # 160,364 kN
_hP, _bP, _twP, _tfP, _rP = 300.0, 300.0, 11.0, 19.0, 27.0        # HEB 300
_hS, _bS, _twS, _tfS, _rS = 400.0, 180.0, 8.6, 13.5, 21.0         # IPE 400
_V = 200.0
# --- 2 L100×100×10 (rc 12), hc 290, 4 + 4 M20, p1 70, e1 40, gA 50, gh 10, e2,b 40
_bA, _bB, _tc, _rc = 100.0, 100.0, 10.0, 12.0
_hc, _n1, _p1, _e1, _e2b, _gh, _zc = 290.0, 4, 70.0, 40.0, 40.0, 10.0, 40.0
_gA = 50.0
_p3 = _twS + 2 * _gA                              # gauge sur le poteau
_p3min = _twP + 2 * _rP + _dw                     # gauge mini (dégagement du congé)
_eP = (_bP - _p3) / 2                             # pince au bout d'aile
_z = _gh + _e2b                                   # bras du groupe S
_MS = _V * _z / 1000
_Ip = _n1 * _p1 ** 2 * (_n1 ** 2 - 1) / 12
_Fz = _V / _n1
_Fx = _MS * 1000 * ((_n1 - 1) * _p1 / 2) / _Ip
_FS = math.hypot(_Fz, _Fx)
_FP = (_V / 2) / _n1
_e2A = _bA - _gA
_e2B = _bB - _gh - _e2b
_e1bot = _hc - _e1 - (_n1 - 1) * _p1
# cornières : cisaillement brut (facteur 1,27), net, rupture de bloc, flexion
_VRdg_C = 2 * _hc * _tc * _fy / (1.27 * S3 * 1.0) / 1000
_VRdn_C = 2 * _tc * (_hc - _n1 * _d0) * _fu / (S3 * 1.25) / 1000
_Ant_B = _tc * (_e2B - 0.5 * _d0)
_Anv_B = _tc * (_hc - _e1 - (_n1 - 0.5) * _d0)
_VRdb_C = 2 * (0.5 * _fu * _Ant_B / 1.25 + _fy * _Anv_B / S3) / 1000
_MRd_C = _tc * _hc ** 2 / 6 * _fy / 1.0 / 1e6
# pression diamétrale des ailes (e1 = min(e1 ; e1'), e2 = pince de l'aile)
_abv_C = min(min(_e1, _e1bot) / (3 * _d0), _p1 / (3 * _d0) - 0.25, _fub / _fu, 1)
_k1v_C = min(2.8 * _e2B / _d0 - 1.7, 2.5)
_Fbv_C = _k1v_C * _abv_C * _fu * _d * _tc / 1.25 / 1000
_abh_C = min(_e2B / (3 * _d0), _fub / _fu, 1)
_k1h_C = min(2.8 * min(_e1, _e1bot) / _d0 - 1.7, 1.4 * _p1 / _d0 - 1.7, 2.5)
_Fbh_C = _k1h_C * _abh_C * _fu * _d * _tc / 1.25 / 1000
_ipd_B = math.hypot(_Fz / 2 / _Fbv_C, _Fx / 2 / _Fbh_C)
# âme de la poutre portée : aucune pince au-delà du grugeage (poutre non grugée)
_e1b = _zc + _e1
_abv_S = min(_e1b / (3 * _d0), _p1 / (3 * _d0) - 0.25, _fub / _fu, 1)
_Fbv_S = min(2.8 * _e2b / _d0 - 1.7, 2.5) * _abv_S * _fu * _d * _twS / 1.25 / 1000
_abh_S = min(_e2b / (3 * _d0), _fub / _fu, 1)
_Fbh_S = min(2.8 * _e1b / _d0 - 1.7, 1.4 * _p1 / _d0 - 1.7, 2.5) * _abh_S * _fu * _d * _twS / 1.25 / 1000
_ipd_S = math.hypot(_Fz / _Fbv_S, _Fx / _Fbh_S)
_A_S = 2 * _bS * _tfS + (_hS - 2 * _tfS) * _twS + (4 - math.pi) * _rS ** 2
_Av_S = max(_A_S - 2 * _bS * _tfS + (_twS + 2 * _rS) * _tfS, (_hS - 2 * _tfS) * _twS)
_VRdg_S = _Av_S * _fy / S3 / 1000
# --- POTEAU, cas (a) : pression diamétrale de l'aile (t = tfP)
# αb : aucune pince d'extrémité verticale ; k1 : gauge p3 et bout d'aile
_abv_P = min(_p1 / (3 * _d0) - 0.25, _fub / _fu, 1)
_k1v_P = min(2.8 * _eP / _d0 - 1.7, 1.4 * _p3 / _d0 - 1.7, 2.5)
_Fbv_Pa = _k1v_P * _abv_P * _fu * _d * _tfP / 1.25 / 1000
# --- POTEAU, cas (b) : pression diamétrale de l'âme (t = twP), aucun bord libre
_k1v_Pb = min(1.4 * _p3 / _d0 - 1.7, 2.5)
_Fbv_Pb = _k1v_Pb * _abv_P * _fu * _d * _twP / 1.25 / 1000
# --- POTEAU, cas (b) : cisaillement local, un seul côté chargé (et = 5d)
_et = 5 * _d
_eb = min(_p3 / 2, 5 * _d)
_Av_P = _twP * (_et + (_n1 - 1) * _p1 + _eb)
_VRd_lP = _Av_P * _fy / (S3 * 1.0) / 1000
_VRd_nP = (_Av_P - _n1 * _d0 * _twP) * _fu / (S3 * 1.25) / 1000
# --- géométrie du cas (b)
_libre_droit = _hP - 2 * _tfP - 2 * _rP
_demi_droit = (_hP - 2 * _tfP) / 2 - _rP
_libre_ailes = _hP - 2 * _tfP - 2 * 5.0
_bA90, _tc90, _rc90 = 90.0, 9.0, 11.0              # L90×90×9 (cas b)
_bA_max = _demi_droit - _twS / 2
_gA_min = _tc + _rc + _d0 / 2                      # congé de la L100×100×10
_gA_min90 = _tc90 + _rc90 + _d0 / 2                # congé de la L90×90×9 : 31,0 mm
_p3b = _twS + 2 * _gA
_Av_Pb = _twP * (_et + (_n1 - 1) * _p1 + min(_p3b / 2, 5 * _d))
_VRd_lPb = _Av_Pb * _fy / (S3 * 1.0) / 1000
_VRd_nPb = (_Av_Pb - _n1 * _d0 * _twP) * _fu / (S3 * 1.25) / 1000
# --- TYING, cas (a), NEd = 300 kN, γMu = 1,1
_Nty = 300.0
_SFt = 2 * _n1 * _Ftu                              # 1 282,91 kN
_ew = _dw / 4
# ailes A des cornières (MSB P5 §4.3.1.1)
_m_A = (_p3 - _twS - 2 * _tc - 1.6 * _rc) / 2
_n_A = min(_e2A, 1.25 * _m_A)
_leff_A = 2 * min(_e1, _e1bot, 0.5 * (_p3 - _twS - 2 * _rc) + _d0 / 2) \
    + (_n1 - 1) * min(_p1, _p3 - _twS - 2 * _rc + _d0)
_MplA, _FT1_A, _FT2_A, _FT3_A = _troncon(_leff_A, _tc, _m_A, _n_A, _ew, _fu, 1.1, _SFt)
_FT_A = min(_FT1_A, _FT2_A, _FT3_A)
# aile du poteau : Tableau 6.4, aile NON RAIDIE, toutes les rangées intérieures
_m_P = (_p3 - _twP) / 2 - 0.8 * _rP
_n_P = min(_eP, 1.25 * _m_P)
_cp_i = _n1 * 2 * math.pi * _m_P
_nc_i = _n1 * (4 * _m_P + 1.25 * _eP)
_cp_g = _n1 * 2 * _p1
_nc_g = _n1 * _p1
_leff_P = min(min(_cp_i, _nc_i), min(_cp_g, _nc_g))
_MplP, _FT1_P, _FT2_P, _FT3_P = _troncon(_leff_P, _tfP, _m_P, _n_P, _ew, _fu, 1.1, _SFt)
_FT_P = min(_FT1_P, _FT2_P, _FT3_P)
_Ft_wc = 1.0 * _leff_P * _twP * _fu / 1.1 / 1000   # §6.2.6.3, ω = 1
_Bp_A = 0.6 * math.pi * _dm * _tc * _fu / 1.1 / 1000
_Bp_Pa = 0.6 * math.pi * _dm * _tfP * _fu / 1.1 / 1000
_Bp_Pb = 0.6 * math.pi * _dm * _twP * _fu / 1.1 / 1000
_tnB_u = 2 * 0.9 * _tc * (_hc - _n1 * _d0) * _fu / 1.1 / 1000
_tnS_u = 0.9 * _twS * (_hc - _n1 * _d0) * _fu / 1.1 / 1000
_bvT_S = _Nty / _n1                                # effort par boulon du groupe S


SRC = dict(
    titre="Steel Buildings in Europe – Multi-Storey Steel Buildings – Part 5: Joint Design "
          "— §4.4 Worked Example: Angle Web Cleats (BM1, BM1') ; BM2 à BM4 : étude du 01/11/2026, "
          "docs/assemblages/POUTRE_POTEAU.md §11.4, recalculée à la main",
    org="ArcelorMittal / Peiner Träger / Corus – contenu technique CTICM et SCI (projet RFCS SECHALO, RFS2-CT-2008-0030)",
    annee="feuilles de calcul datées 06/2009",
    norme="EN 1993-1-8:2005 et EN 1993-1-1:2005, valeurs recommandées (γM0 = 1,0 ; γM2 = 1,25 ; γMu = 1,1) ; "
          "le module applique par défaut l'ANB belge (γM0 = 1,00 ; γM2 = 1,25)",
    url="https://steelconstruction.info/images/5/53/SBE_MS5.pdf")

BENCH = [
    dict(id="BM1", titre="HÉRITAGE — §4.4 Worked Example: Angle Web Cleats sur un POTEAU (appui sur "
                         "l'aile), VEd = 450 kN, p. 5-68 à 5-77",
         data="IPE A 550 S275 portée (h 547, b 210, tw 9, tf 15,7, r 24) sur poteau PERSO S275 d'aile "
              "tfP = 15 mm, 2 L90x90x10 S275, hc = 430, M20 8.8, n1 = 6, p1 = 70, e1 = 40, e2 = 40, "
              "gh = 10, e2,b = 40 (z = 50), gA = 50 (p3 = 109), poutre non grugée. Les seize valeurs "
              "publiées ne dépendent PAS du support : elles doivent sortir inchangées",
         inp=WE, rows=[
             ["p. 5-70", "Cisaillement d'un boulon Fv,Rd (1 plan)", 94, va("Fv_Rd"), "kN", E_ARR],
             ["p. 5-70", "Groupe S excentré – cisaillement des boulons VRd", 962, vr("bv_S"), "kN", E_ARR],
             ["p. 5-71", "Pression diamétrale cornière Fb,ver,Rd", 105, va("Fbv_B"), "kN", E_ARR],
             ["p. 5-71", "Pression diamétrale cornière Fb,hor,Rd", 105, va("Fbh_B"), "kN", E_ARR],
             ["p. 5-71", "Groupe S – pression diamétrale sur les cornières VRd", 1075, vr("pdB"), "kN", E_ARR],
             ["p. 5-72", "Pression diamétrale âme Fb,ver,Rd", 125, va("Fbv_S"), "kN", ""],
             ["p. 5-73", "Pression diamétrale âme Fb,hor,Rd", 94, va("Fbh_S"), "kN", ""],
             ["p. 5-73", "Groupe S – pression diamétrale sur l'âme VRd", 583, vr("pdS"), "kN", ""],
             ["p. 5-74", "Côté support – FRd = 0,8·ns·Fv,Rd", 902, vr("bv_P"), "kN", E_ARR],
             ["p. 5-74", "Cornières – cisaillement section brute VRd,g", 1076, rd("cgB"), "kN", ""],
             ["p. 5-75", "Cornières – cisaillement section nette VRd,n", 1184, rd("cnB"), "kN", ""],
             ["p. 5-75", "Cornières – rupture de bloc VRd,b (côté poutre portée)", 954, rd("cbB"), "kN", ""],
             ["p. 5-76", "Cornières – rupture de bloc VRd,b (côté support)", 954, rd("cbA"), "kN", ""],
             ["p. 5-77", "Âme – cisaillement section brute VRd,g", 953, rd("vgS"), "kN", E_AIRE],
             ["p. 5-77", "Âme – cisaillement section nette VRd,n", 956, rd("vnS"), "kN", E_AIRE],
             ["p. 5-77", "Âme – rupture de bloc VRd,b", 501, rd("vbS"), "kN", ""]]),
    dict(id="BM1'", titre="HÉRITAGE — TYING du même exemple publié (p. 5-78/79) : l'ANCRE des modes "
                          "1 / 2 / 3 du tronçon en T, la formule même que l'AILE DU POTEAU réutilise. "
                          "Le guide applique fu/γMu : reproduit avec un acier de cornière PERSO "
                          "fy = fu = 430 MPa et γM0 = γM2 = 1,1",
         data="Données BM1 + NEd = 370 kN, acier de cornière PERSO fy = fu = 430 MPa, γM0 = γM2 = 1,1",
         inp=WE_T, rows=[
             ["p. 5-79", "Traction d'un boulon Ft,Rd,u", 160, va("Ft_Rd"), "kN", ""],
             ["p. 5-78", "Longueur efficace Σleff des ailes A", 430, va("leff_T"), "mm", ""],
             ["p. 5-78", "Moment plastique Mpl,1,Rd,u", 4.2, va("Mpl_T"), "kNm", ""],
             ["p. 5-79", "Tronçon en T – mode 1", 696, va("FT_1"), "kN", E_ARR],
             ["p. 5-79", "Tronçon en T – mode 2", 1190, va("FT_2"), "kN", E_ARR],
             ["p. 5-79", "Tronçon en T – mode 3", 1920, va("FT_3"), "kN", E_ARR]]),
    dict(id="BM2", titre="Cas (a) AILE — le cas par défaut du module : le côté poteau n'apporte que "
                         "la PRESSION DIAMÉTRALE de l'aile (t = tfP), sans pince d'extrémité verticale",
         data="IPE 400 S355 sur HEB 300 S355, 2 L100x100x10 S355, hc = 290, 4 + 4 M20 8.8, p1 = 70, "
              "e1 = 40, gA = 50, gh = 10, e2,b = 40, zc = 40, VEd = 200 kN",
         inp=BM2, rows=[
             ["§11.4", "Gauge sur le poteau p3 = tw,S + 2·gA", _p3, va("p_3"), "mm", ""],
             ["§11.4", "Gauge minimal (dégagement du congé) twP + 2·rP + dw", _p3min, va("p3_congeP"), "mm", ""],
             ["§11.4", "Pince au bout d'aile du poteau e = (bP − p3)/2", _eP, va("e_P_fl"), "mm", ""],
             ["§11.4", "Bras du groupe S z = gh + e2,b", _z, va("z_S"), "mm", ""],
             ["§11.4", "Moment d'excentricité MS = VEd·z", _MS, va("M_S"), "kNm", ""],
             ["§11.4", "Inertie polaire Ip = n·p1²(n1² − 1)/12", _Ip, va("Ip_S"), "mm²", ""],
             ["§11.4", "Effort du boulon le plus sollicité du groupe S", _FS, va("F_S"), "kN", ""],
             ["§11.4", "Taux du groupe S (2 plans : 2·Fv,Rd = 188,16 kN)", _FS / (2 * _Fv), et("bv_S"), "-", ""],
             ["§11.4", "Taux du groupe P (k_rot·Fv,Rd = 0,8 × 94,08)", _FP / (0.8 * _Fv), et("bv_P"), "-", ""],
             ["§11.4", "Cornières – cisaillement brut VRd,g (1/1,27)", _VRdg_C, rd("cgB"), "kN", ""],
             ["§11.4", "Cornières – cisaillement net VRd,n", _VRdn_C, rd("cnB"), "kN", ""],
             ["§11.4", "Ailes B – rupture de bloc Veff,2,Rd (2 cornières)", _VRdb_C, rd("cbB"), "kN", ""],
             ["§11.4", "Ailes B – flexion dans leur plan MRd = Wel·fy/γM0", _MRd_C, rd("flB"), "kNm", ""],
             ["§11.4", "Ailes B – taux de pression diamétrale (interaction, effort F/2)", _ipd_B, et("pdB"), "-", ""],
             ["§11.4", "Ailes A – Fb,ver,Rd", _Fbv_C, va("Fbv_A"), "kN", ""],
             ["§11.4", "Âme portée – taux de pression diamétrale", _ipd_S, et("pdS"), "-", ""],
             ["§11.4", "Âme portée – cisaillement brut VRd,g", _VRdg_S, rd("vgS"), "kN", ""],
             ["§11.4", "AILE DU POTEAU – Fb,Rd (αb = 0,810606 ; k1 = 2,5 ; t = 19 mm)", _Fbv_Pa, va("Fbv_P"), "kN", ""],
             ["§11.4", "AILE DU POTEAU – taux de pression diamétrale", _FP / _Fbv_Pa, et("pdP"), "-", ""]]),
    dict(id="BM3", titre="Cas (a) TYING, NEd = 300 kN — le tronçon en T des ailes A et celui de "
                         "l'AILE DU POTEAU : les CORNIÈRES gouvernent (rapport 1,915)",
         data="Données BM2 + NEd = 300 kN en tying (γMu = 1,1) ; Ft,Rd,u = 160,364 kN, "
              "ΣFt,Rd,u = 2·n1P·Ft,Rd,u = 1 282,91 kN, ew = dw/4 = 9,25 mm",
         inp=BM3, rows=[
             ["§11.4", "Boulons du groupe S en cisaillement – effort NEd/n", _bvT_S, va("N_par_S"), "kN", ""],
             ["§11.4", "Boulons du groupe S – résistance 2·αv·fub·A/γMu", 2 * _Fvu, rd("bvT_S"), "kN", ""],
             ["§11.4", "Ailes A – bras m = (p3 − tw,S − 2tc − 1,6·rc)/2", _m_A, va("m_T"), "mm", ""],
             ["§11.4", "Ailes A – bras n = min(e2 ; 1,25·m)", _n_A, va("n_T"), "mm", ""],
             ["§11.4", "Ailes A – Σleff", _leff_A, va("leff_T"), "mm", ""],
             ["§11.4", "Ailes A – Mpl,1 (fu/γMu)", _MplA, va("Mpl_T"), "kNm", ""],
             ["§11.4", "Ailes A – mode 1", _FT1_A, va("FT_1"), "kN", ""],
             ["§11.4", "Ailes A – mode 2", _FT2_A, va("FT_2"), "kN", ""],
             ["§11.4", "Ailes A – FT,Rd (taux η = NEd/FT,Rd)", _FT_A, rd("tsA"), "kN", ""],
             ["§11.4", "AILE DU POTEAU – bras m = (p3 − twP)/2 − 0,8·rP", _m_P, va("m_P"), "mm", ""],
             ["§11.4", "AILE DU POTEAU – bras n = min(e ; 1,25·m)", _n_P, va("n_Pf"), "mm", ""],
             ["§11.4", "AILE DU POTEAU – Σleff (Tab. 6.4, rangées en groupe : n1·p1)", _leff_P, va("leff1_P"), "mm", ""],
             ["§11.4", "AILE DU POTEAU – Mpl,1 (fu/γMu)", _MplP, va("Mpl_P"), "kNm", ""],
             ["§11.4", "AILE DU POTEAU – mode 1", _FT1_P, va("FTP_1"), "kN", ""],
             ["§11.4", "AILE DU POTEAU – mode 2", _FT2_P, va("FTP_2"), "kN", ""],
             ["§11.4", "AILE DU POTEAU – FT,Rd", _FT_P, rd("tfP"), "kN", ""],
             ["§11.4", "Âme du poteau en traction transversale (ω = 1, beff = Σleff)", _Ft_wc, rd("twtP"), "kN", ""],
             ["§11.4", "Poinçonnement de l'aile A de la cornière (t = 10 mm)", _Bp_A, va("Bp_Rd"), "kN", ""],
             ["§11.4", "Poinçonnement de l'aile du poteau (t = 19 mm)", _Bp_Pa, rd("ppP"), "kN", ""],
             ["§11.4", "Ailes B – traction section nette (2 × 0,9·Anet·fu/γMu)", _tnB_u, rd("tnB"), "kN", ""],
             ["§11.4", "Âme portée – traction section nette (hauteur participante hc)", _tnS_u, rd("tnS"), "kN", ""]]),
    dict(id="BM4", titre="Cas (b) ÂME DU POTEAU — 2 L90×90×9 : une L100 NE PASSE PAS "
                         "(4,3 + 100 = 104,3 > 104 mm), une L90 passe (94,3 mm)",
         data="Données BM2, appui = Âme du poteau, 2 L90x90x9 S355 ; la semelle de l'IPE 400 "
              "(bS = 180) passe entre les ailes du HEB 300 (hP − 2tfP − 2j = 252 mm) : aucune "
              "recoupe n'est nécessaire",
         inp=BM4, rows=[
             ["§11.4", "Largeur droite de l'âme entre congés hP − 2tfP − 2rP", _libre_droit, va("libre_droit"), "mm", ""],
             ["§11.4", "Demi-largeur droite (hP − 2tfP)/2 − rP", _demi_droit, va("demi_droit"), "mm", ""],
             ["§11.4", "Longueur d'aile A maximale admise = demi-largeur − tw,S/2", _bA_max, va("bA_max"), "mm", ""],
             ["§11.4", "gA minimal (congé de la cornière) = tc + rc + d0/2", _gA_min90, va("gA_min"), "mm", ""],
             ["§11.4", "Passage de la semelle hP − 2tfP − 2j", _libre_ailes, va("libre_ailes"), "mm", ""],
             ["§11.4", "ÂME DU POTEAU – Fb,Rd (αb = 0,810606 ; k1 = 2,5 ; t = 11 mm)", _Fbv_Pb, va("Fbv_P"), "kN", ""],
             ["§11.4", "ÂME DU POTEAU – taux de pression diamétrale", _FP / _Fbv_Pb, et("pdP"), "-", ""],
             ["§11.4", "ÂME DU POTEAU – et = 5·d", _et, va("et_P"), "mm", ""],
             ["§11.4", "ÂME DU POTEAU – eb = min(p3/2 ; 5d)", min(_p3b / 2, 5 * _d), va("eb_P"), "mm", ""],
             ["§11.4", "ÂME DU POTEAU – Av = twP·(et + (n1 − 1)·p1 + eb)", _Av_Pb, va("Av_P"), "mm²", ""],
             ["§11.4", "ÂME DU POTEAU – cisaillement local BRUT", _VRd_lPb, rd("vlP"), "kN", ""],
             ["§11.4", "ÂME DU POTEAU – cisaillement local NET", _VRd_nPb, rd("vnP"), "kN", ""],
             ["§11.4", "ÂME DU POTEAU – taux brut / net (Ed = VEd/2 = 100 kN)",
              _V / 2 / _VRd_lPb, et("vlP"), "-", ""]]),
]

NOCOVER = [
    ["Pression diamétrale sur l'aile / l'âme du poteau", "Aucun exemple publié : formule directe du "
     "Tab. 3.4 avec αb SANS pince d'extrémité verticale (le poteau continue au-dessus et en dessous) "
     "et k1 du gauge p3 — en cas (a) le seul bord libre est le bout d'aile, e = (bP − p3)/2 ; en "
     "cas (b) il n'y en a aucun. Calcul manuel indépendant (BM2, BM4, VAL-B, VAL-C)."],
    ["Tronçon en T de l'AILE DU POTEAU", "Aucun exemple publié consulté pour une attache SIMPLE. Les "
     "formules des modes 1 / 2 / 3 sont ancrées par l'exemple publié §4.4 (BM1') ; les longueurs "
     "efficaces viennent du Tableau 6.4 (aile NON RAIDIE, toutes les rangées intérieures). Calcul "
     "manuel indépendant (BM3, VAL-A). Point tranché P-1 : c'est le NUMÉRO du tableau qui est en cause."],
    ["Âme du poteau en traction transversale", "§6.2.6.3 avec ω = 1 et beff,t,wc = Σleff de l'aile : "
     "interprétation de l'outil (point tranché P-5), calcul manuel indépendant (BM3, VAL-A)."],
    ["Cisaillement local de l'âme du poteau (cas b)", "Même modèle que les trois modules "
     "poutre–poutre (SCI P358, élément support chargé d'un seul côté), avec et = 5·d et "
     "eb = min(p3/2 ; 5d) — un poteau n'a pas de face horizontale (point tranché P-8). Calcul manuel "
     "indépendant (BM4, VAL-C)."],
    ["Règles géométriques du poteau", "Gauge minimal p3 ≥ twP + 2·rP + dw (cas a), dégagement de "
     "l'écrou et longueur d'aile A maximale par la partie droite de l'âme (cas b), passage de la "
     "semelle entre les ailes et recoupe en largeur : géométrie, pas de résistance — calculs manuels "
     "(BM2, BM4). La recoupe en LARGEUR n'est pas un grugeage (point tranché P-3)."],
    ["Tying — boulons du groupe S en cisaillement, poinçonnement du poteau", "La bascule nat_N / γMu "
     "vient du module platine d'about (MSB P5 §2.3). La force de chaînage est dans le PLAN de l'âme "
     "portée : elle cisaille le groupe S (étude §7.2) — ligne ajoutée à la liste du §8.2, qui ne la "
     "portait pas. Formules directes du Tab. 3.4, calcul manuel indépendant (BM3, VAL-A)."],
    ["Glissement (catégories B et C)", "Formule directe §3.9, aucun exemple publié."],
]

VALID = [
    dict(id="VAL-A", titre="TYING sur le cas (a) par défaut (NEd = 300 kN, γMu = 1,1) : le tronçon en T "
                           "des ailes A ET celui de l'AILE DU POTEAU, modes 1 / 2 / 3 réécrits in extenso",
         inp=BM3, rows=[
        ["Traction d'un boulon Ft,Rd,u = 0,9·fub·As/γMu", _Ftu, 160.3636364, va("Ft_Rd_g"), "kN"],
        ["ΣFt,Rd,u = 2·n1P·Ft,Rd,u", _SFt, 1282.909091, va("SFt_T"), "kN"],
        ["Ailes A – m", _m_A, 30.4, va("m_T"), "mm"],
        ["Ailes A – n = min(e2 ; 1,25·m)", _n_A, 38.0, va("n_T"), "mm"],
        ["Ailes A – Σleff", _leff_A, 290.0, va("leff_T"), "mm"],
        ["Ailes A – Mpl,1 = 0,25·Σleff·tc²·fu/γMu", _MplA, 3.361363636, va("Mpl_T"), "kNm"],
        ["Ailes A – mode 1", _FT1_A, 572.0149, va("FT_1"), "kN"],
        ["Ailes A – mode 2", _FT2_A, 811.0129, va("FT_2"), "kN"],
        ["Aile du poteau – m = (p3 − twP)/2 − 0,8·rP", _m_P, 27.2, va("m_P"), "mm"],
        ["Aile du poteau – n = min(e ; 1,25·m)", _n_P, 34.0, va("n_Pf"), "mm"],
        ["Aile du poteau – Σcp rangées isolées = n1·2πm", _cp_i, 683.6106, lambda R: 4 * 2 * math.pi * R.m_P, "mm"],
        ["Aile du poteau – Σnc rangées isolées = n1·(4m + 1,25e)", _nc_i, 913.7, lambda R: 4 * (4 * R.m_P + 1.25 * R.e_P_fl), "mm"],
        ["Aile du poteau – Σleff,1 = Σleff,2 = n1·p1 (en groupe)", _leff_P, 280.0, va("leff1_P"), "mm"],
        ["Aile du poteau – Mpl,1 = 0,25·Σleff·tfP²·fu/γMu", _MplP, 11.71609091, va("Mpl_P"), "kNm"],
        ["Aile du poteau – mode 1", _FT1_P, 2314.0290, va("FTP_1"), "kN"],
        ["Aile du poteau – mode 2", _FT2_P, 1095.6060, va("FTP_2"), "kN"],
        ["Aile du poteau – FT,Rd = min(FT,1 ; FT,2 ; FT,3)", _FT_P, 1095.6060, rd("tfP"), "kN"],
        ["Rapport aile du poteau / cornières", _FT_P / _FT_A, 1.915345, lambda R: R.FTP_Rd / R.FT_Rd, "-"],
        ["Âme du poteau en traction transversale (ω = 1)", _Ft_wc, 1428.0, rd("twtP"), "kN"],
        ["Poinçonnement de l'aile du poteau (t = tfP)", _Bp_Pa, 528.0307, rd("ppP"), "kN"],
        ["Boulons du groupe S en cisaillement – 2·αv·fub·A/γMu", 2 * _Fvu, 213.8181818, rd("bvT_S"), "kN"]]),
    dict(id="VAL-B", titre="Cas (a) par défaut : le modèle (bras, groupe excentré, cornières) et la "
                           "PRESSION DIAMÉTRALE de l'aile du poteau — calculs indépendants",
         inp=BM2, rows=[
        ["Gauge p3 = tw,S + 2·gA", _p3, 108.6, va("p_3"), "mm"],
        ["Gauge minimal twP + 2·rP + dw", _p3min, 102.0, va("p3_congeP"), "mm"],
        ["Pince au bout d'aile e = (bP − p3)/2", _eP, 95.7, va("e_P_fl"), "mm"],
        ["Moment d'excentricité MS = VEd·z", _MS, 10.0, va("M_S"), "kNm"],
        ["Effort maximal sur un boulon du groupe S", _FS, 65.85388898, va("F_S"), "kN"],
        ["Pince e2 de l'aile B", _e2B, 50.0, va("e2a_S"), "mm"],
        ["Pince e2 de l'aile A", _e2A, 50.0, va("e2a_P"), "mm"],
        ["Pince basse e1' = hc − e1 − 3·p1", _e1bot, 40.0, va("e1bot_S"), "mm"],
        ["Ailes B – taux de pression diamétrale (interaction)", _ipd_B, 0.2451766, et("pdB"), "-"],
        ["Âme portée – taux de pression diamétrale (interaction)", _ipd_S, 0.5348614, et("pdS"), "-"],
        ["Aile du poteau – αb (aucune pince d'extrémité)", _abv_P, 0.8106061, va("abv_P"), "-"],
        ["Aile du poteau – k1 (bout d'aile et gauge)", _k1v_P, 2.5, va("k1v_P"), "-"],
        ["Aile du poteau – Fb,Rd", _Fbv_Pa, 314.1909, va("Fbv_P"), "kN"],
        ["Excentricité apportée au poteau e_col = hP/2 + z", _hP / 2 + _z, 200.0, va("e_col"), "mm"],
        ["Moment apporté au poteau VEd·e_col (information)", _V * (_hP / 2 + _z) / 1000, 40.0,
         va("M_col"), "kNm"]]),
    dict(id="VAL-C", titre="Cas (b) ÂME DU POTEAU : la géométrie entre congés, la pression diamétrale "
                           "de l'âme et le cisaillement local d'un seul côté chargé (et = 5·d)",
         inp=BM4, rows=[
        ["Largeur droite entre congés hP − 2tfP − 2rP", _libre_droit, 208.0, va("libre_droit"), "mm"],
        ["Demi-largeur droite (hP − 2tfP)/2 − rP", _demi_droit, 104.0, va("demi_droit"), "mm"],
        ["Aile A maximale = demi-largeur − tw,S/2", _bA_max, 99.7, va("bA_max"), "mm"],
        ["Gauge maximal 2·(demi-largeur − dw/2)", 2 * (_demi_droit - _dw / 2), 171.0, va("p3_max_ame"), "mm"],
        ["Passage de la semelle hP − 2tfP − 2j", _libre_ailes, 252.0, va("libre_ailes"), "mm"],
        ["Recoupe minimale (bP − twP)/2 + j − gh", (_bP - _twP) / 2 + 5 - _gh, 139.5, va("rec_min"), "mm"],
        ["Âme du poteau – k1 = min(1,4·p3/d0 − 1,7 ; 2,5)", _k1v_Pb, 2.5, va("k1v_P"), "-"],
        ["Âme du poteau – Fb,Rd (t = twP)", _Fbv_Pb, 181.8996, va("Fbv_P"), "kN"],
        ["Borne et = 5·d", _et, 100.0, va("et_P"), "mm"],
        ["Borne eb = min(p3/2 ; 5d)", min(_p3b / 2, 5 * _d), 54.3, va("eb_P"), "mm"],
        ["Aire de cisaillement Av = twP·(et + (n1 − 1)·p1 + eb)", _Av_Pb, 4007.3, va("Av_P"), "mm²"],
        ["Cisaillement local brut VRd = Av·fy,P/(√3·γM0)", _VRd_lPb, 821.3336, rd("vlP"), "kN"],
        ["Cisaillement local net VRd,n", _VRd_nPb, 715.9342, rd("vnP"), "kN"],
        ["Poinçonnement de l'âme (t = twP, γM2)",
         0.6 * math.pi * _dm * _twP * _fu / 1.25 / 1000, 269.0176, va("Bp_P"), "kN"]]),
]


def run_bench():
    """Rejoue l'exemple publié, les quatre cas de l'étude et les calculs
    manuels ; retourne ``bench``, ``valid``, ``worst``, ``ok`` et ``statut``
    (« VALIDÉ » / « À CONTRÔLER »)."""
    out = []; worst = 0; n_unexp = 0
    for b in BENCH:
        R = compute(b["inp"]); rows = []
        for r in b["rows"]:
            v = r[3](R); ec = (v - r[2]) / r[2]; a = abs(ec)
            if a <= 0.01:
                st = "OK"
            elif a <= 0.03:
                st = "Écart expliqué" if r[5] else "À expliquer"
            else:
                st = "Écart > 3 % documenté" if r[5] else "À vérifier"
            if a > 0.01 and not r[5]:
                n_unexp += 1
            worst = mx(worst, a)
            rows.append(dict(page=r[0], lab=r[1], ref=r[2], val=v, unit=r[4], ec=ec, st=st, exp=r[5]))
        out.append(dict(id=b["id"], titre=b["titre"], data=b["data"], rows=rows))
    nv = 0; val = []
    for c in VALID:
        R = compute(c["inp"]); rows = []
        for r in c["rows"]:
            v = r[3](R); ec = (v - r[1]) / r[1] if r[1] != 0 else v
            if abs(ec) > 0.001:
                nv += 1
            rows.append(dict(lab=r[0], hand=r[1], xls=r[2], val=v, unit=r[4], ec=ec,
                             st="OK" if abs(ec) <= 0.001 else "À vérifier"))
        val.append(dict(id=c["id"], titre=c["titre"], rows=rows))
    ok = n_unexp == 0 and nv == 0
    return dict(bench=out, valid=val, worst=worst, ok=ok, statut="VALIDÉ" if ok else "À CONTRÔLER")
