# -*- coding: utf-8 -*-
"""Benchmark du module platine d'about sur poteau — quatre cas, et aucun
nombre de mémoire.

- **BM1 — HÉRITAGE** : l'exemple publié du guide (MSB Part 5 §2.4, *Worked
  Example: Partial depth end plate*, VEd = 550 kN et tying 275 kN) rejoué
  avec ``appui = Aile du poteau`` et un poteau PERSO d'aile 15 mm S275. Les
  NEUF valeurs publiées doivent sortir INCHANGÉES : du côté platine, boulons
  et âme portée, rien ne dépend du support (démontré par l'étude, script 00).
  C'est le contrôle d'héritage du module.
- **BM2** : le cas par défaut, appui sur l'AILE (étude §11.3).
- **BM3** : le même en TYING (NEd = 250 kN, γMu = 1,1) — le tableau du §7.3 :
  les DEUX tronçons en T, celui de la platine et celui de l'AILE DU POTEAU,
  qui est 3,66 fois plus forte.
- **BM4** : appui dans l'ÂME du poteau — la platine 290 × 220 du cas (a) ne
  tient pas entre les congés (198 mm) : p3 = 110 et bp = 190.

Chaque valeur de référence est écrite ici comme une EXPRESSION LITTÉRALE
indépendante (jamais une constante recopiée) : le moteur est relancé et
comparé à elle. Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à
vérifier. Un écart qui porte une explication (``exp``) ne compte pas comme
inattendu. Aucune formule n'a été ajustée pour coller à la source.
"""
import math

from acier.bibliotheques import PERSO
from acier.js import mx
from .entrees import APPUI_AILE, APPUI_AME, NAT_TYING
from .moteur import compute

S3 = math.sqrt(3)

E_AB = "Arrondi de la source : αb = 0,61 pour e1/3d0 = 40/66 = 0,606."
E_FV = "Arrondi de la source : Fv,Rd = 94 kN pour 94,08 kN."
E_M = "Arrondi de la source : m = 59 mm pour 59,16 mm (a = 5,6 mm)."

# --- les données de l'exemple publié, reprises telles quelles du module
# --- poutre–poutre, SUPPORT remplacé : un poteau PERSO dont l'aile a
# --- l'épaisseur de l'âme support de l'exemple (15 mm) et sa nuance (S275).
# --- Aucune des neuf valeurs publiées ne dépend du support ; hP, bP, twP et
# --- rP ne servent qu'au dessin, au gauge et — en cas (b) — au cisaillement
# --- local (inactif ici).
WE = dict(appui=APPUI_AILE,
          prof_S=PERSO, hS_u=547, bS_u=210, twS_u=9, tfS_u=15.7, rS_u=24, nu_S="S275",
          prof_P=PERSO, hP_u=300, bP_u=300, twP_u=11, tfP_u=15, rP_u=27, nu_P="S275",
          nu_pl="S275", d_nt=0, d_nb=0, l_n=0, z_C=50,
          hp_u=430, tp_u=12, bp_u=200, boulon_u="M20", classe="8.8",
          n1_u=6, p1_u=70, e1_u=40, p3_u=140, a_w=5.6,
          V_Ed=550, N_Ed=275, nat_N=NAT_TYING, g_Mu=1.1)

# --- BM2 : le cas par défaut (appui sur l'aile) — aucune entrée à forcer
BM2 = {}
# --- BM3 : le cas par défaut en TYING (NEd = 250 kN seul, γMu = 1,1)
BM3 = dict(N_Ed=250, nat_N=NAT_TYING)
# --- BM4 : appui dans l'âme — platine 290 × 190 × 10, p3 = 110
BM4 = dict(appui=APPUI_AME, bp_u=190, p3_u=110)
# --- BM4 en tying : le poinçonnement de l'ÂME du poteau (t = 11, γMu)
BM4T = dict(BM4, N_Ed=250, nat_N=NAT_TYING)


def vr(k):
    """Résistance du groupe exprimée en VEd : VEd / η de la vérification k."""
    return lambda R: R.u.V_Ed / R.ck[k].eta


def rd(k):
    return lambda R: R.ck[k].Rd


def et(k):
    return lambda R: R.ck[k].eta


def va(k):
    return lambda R: R[k]


# =====================================================================
#  Calculs manuels indépendants (expressions littérales, hors moteur)
# =====================================================================
# --- géométrie et matériaux de référence (étude §11.1)
_fy, _fu, _bw = 355.0, 510.0, 0.9                 # S355
_d, _d0, _As, _dm, _dw = 20.0, 22.0, 245.0, 31.8, 37.0        # M20
_fub, _av = 800.0, 0.6                            # classe 8.8, filet cisaillé
_Fv = _av * _fub * _As / 1.25 / 1000              # 94,08 kN
_Ftu = 0.9 * _fub * _As / 1.1 / 1000              # tying, γMu = 1,1
_hP, _bP, _twP, _tfP, _rP = 300.0, 300.0, 11.0, 19.0, 27.0        # HEB 300
_hS, _bS, _twS, _tfS, _rS = 400.0, 180.0, 8.6, 13.5, 21.0         # IPE 400
_V, _Nty = 200.0, 250.0
# --- platine 290 × 220 × 10, 4 × 2 M20, p1 70, e1 40, p3 140, a 5, zp 40
_hp, _tp, _bp, _n1, _p1, _e1, _p3, _a = 290.0, 10.0, 220.0, 4, 70.0, 40.0, 140.0, 5.0
_ns = 2 * _n1
_e2 = (_bp - _p3) / 2                             # pince latérale de la platine
_eP = (_bP - _p3) / 2                             # pince au bout d'aile du poteau
_e1bot = _hp - _e1 - (_n1 - 1) * _p1
_FvEd = _V / _ns
# gauge minimal : l'écrou dégage le congé du poteau (étude §4.3)
_p3_congeP = _twP + 2 * _rP + _dw
# platine : pression diamétrale (Tab. 3.4), deux files
_k1_L = min(2.8 * _e2 / _d0 - 1.7, 1.4 * _p3 / _d0 - 1.7, 2.5)
_abb_L = min(_e1 / (3 * _d0), _fub / _fu, 1)
_abi_L = min(_p1 / (3 * _d0) - 0.25, _fub / _fu, 1)
_Fbb_L = _k1_L * _abb_L * _fu * _d * _tp / 1.25 / 1000
_Fbi_L = _k1_L * _abi_L * _fu * _d * _tp / 1.25 / 1000
_FRd_g = 0.8 * _ns * _Fv                          # Fv,Rd ≤ (Fb,Rd)min
_VRdg_L = 2 * (_hp * _tp) * _fy / (1.27 * S3 * 1.0) / 1000
_VRdn_L = 2 * _tp * (_hp - _n1 * _d0) * _fu / (S3 * 1.25) / 1000
_Ant_L = _tp * (_e2 - 0.5 * _d0)
_Anv_L = _tp * (_hp - _e1 - (_n1 - 0.5) * _d0)
_VRdb_L = 2 * (1.0 * _fu * _Ant_L / 1.25 + _fy * _Anv_L / S3) / 1000   # k = 1 (hp ≥ 1,36·p3)
# âme de la poutre portée au droit de la platine
_Av_w = _hp * _twS
_VRd_wS = _Av_w * _fy / S3 / 1000
# aile du poteau : pression diamétrale, AUCUNE pince d'extrémité verticale
_k1_P = min(2.8 * _eP / _d0 - 1.7, 1.4 * _p3 / _d0 - 1.7, 2.5)
_Fb_P = _k1_P * _abi_L * _fu * _d * _tfP / 1.25 / 1000
_VRd_pdP = _ns * _Fb_P
# cordons (deux cordons verticaux pleine hauteur)
_fvw = _fu / (S3 * _bw * 1.25)
_qz = _V * 1000 / (2 * _hp)
_k_plein = _bw * 1.25 * _fy / (math.sqrt(2) * _fu * 1.0)
_a_plein = _k_plein * _twS
# --- TYING (NEd = 250 kN, γMu = 1,1) : les DEUX tronçons en T
_SFt = _ns * _Ftu
_ew = _dw / 4
# platine : m, n, Σleff (bords libres haut et bas : les rangées en groupe
# donnent 2·min(2m + 0,625e + p1/2 ; e1 + p1/2) + (n1 − 2)·p1 = hp)
_mT = (_p3 - _twS) / 2 - 0.8 * _a * math.sqrt(2)
_nT = min(_e2, 1.25 * _mT)
_leffT = 2 * min(2 * _mT + 0.625 * _e2 + 0.5 * _p1, _e1 + 0.5 * _p1) + (_n1 - 2) * _p1
_MplT = 0.25 * _leffT * _tp ** 2 * _fu / 1.1                 # N·mm
_FT1_L = (8 * _nT - 2 * _ew) * _MplT / (2 * _mT * _nT - _ew * (_mT + _nT)) / 1000
_FT2_L = (2 * _MplT + _nT * _SFt * 1000) / (_mT + _nT) / 1000
_FT3_L = _SFt
# aile du poteau : m du poteau LAMINÉ (congé), toutes les rangées INTÉRIEURES
_mP = (_p3 - _twP) / 2 - 0.8 * _rP
_nP = min(_eP, 1.25 * _mP)
_leffP = min(_n1 * 2 * math.pi * _mP, _n1 * (4 * _mP + 1.25 * _eP), _n1 * 2 * _p1, _n1 * _p1)
_MplP = 0.25 * _leffP * _tfP ** 2 * _fu / 1.1                # N·mm
_FT1_P = (8 * _nP - 2 * _ew) * _MplP / (2 * _mP * _nP - _ew * (_mP + _nP)) / 1000
_FT2_P = (2 * _MplP + _nP * _SFt * 1000) / (_mP + _nP) / 1000
_FT3_P = _SFt
_FT_L = min(_FT1_L, _FT2_L, _FT3_L)
_FT_P = min(_FT1_P, _FT2_P, _FT3_P)
# poinçonnement et âme du poteau en traction transversale (ω = 1)
_Bp_pl = 0.6 * math.pi * _dm * _tp * _fu / 1.1 / 1000
_Bp_fl = 0.6 * math.pi * _dm * _tfP * _fu / 1.1 / 1000
_Bp_we = 0.6 * math.pi * _dm * _twP * _fu / 1.1 / 1000
_Ft_wc = 1.0 * _leffP * _twP * _fu / 1.1 / 1000
_tnS_u = _Av_w * _fu / 1.1 / 1000
_qx_ty = _Nty * 1000 / (2 * _hp)
_Fw_ty = _fu / (S3 * _bw * 1.1) * _a
_FtEd = _Nty / _ns
# --- cas (b) : géométrie du poteau et cisaillement local de l'âme
_p3b, _bpb = 110.0, 190.0
_libre_droit = _hP - 2 * _tfP - 2 * _rP
_demi_droit = (_hP - 2 * _tfP) / 2 - _rP
_libre_ailes = _hP - 2 * _tfP - 2 * 5.0
_bp_maxP = _libre_droit - 2 * 5.0
_p3_maxP = 2 * (_demi_droit - _dw / 2)
_marge_b = _demi_droit - (_p3b / 2 + _dw / 2)
_rec_min = (_bP - _twP) / 2 + 5.0 - _tp
_k1_Pb = min(1.4 * _p3b / _d0 - 1.7, 2.5)
_Fb_Pb = _k1_Pb * _abi_L * _fu * _d * _twP / 1.25 / 1000
_VRd_pdPb = _ns * _Fb_Pb
_Av_Pb = _twP * _hp
_VRd_lP = _Av_Pb * _fy / S3 / 1000
_VRd_nP = _twP * (_hp - _n1 * _d0) * _fu / (S3 * 1.25) / 1000


SRC = dict(
    titre="Steel Buildings in Europe – Multi-Storey Steel Buildings – Part 5: Joint Design "
          "— §2.4 Worked Example: Partial depth end plate (BM1) ; BM2 à BM4 : étude du 01/11/2026, "
          "docs/assemblages/POUTRE_POTEAU.md §11.3, recalculée à la main",
    org="ArcelorMittal / Peiner Träger / Corus – contenu technique CTICM et SCI (projet RFCS SECHALO, RFS2-CT-2008-0030)",
    annee="feuilles de calcul 1 à 7 du §2.4, datées 06/2009",
    norme="EN 1993-1-8:2005 et EN 1993-1-1:2005, valeurs recommandées ; tying avec fu/γMu, γMu = 1,1 "
          "(MSB P5 §2.3)",
    url="https://steelconstruction.info/images/5/53/SBE_MS5.pdf")

BENCH = [
    dict(id="BM1", titre="HÉRITAGE — §2.4 Worked Example: Partial depth end plate sur un POTEAU (appui "
                         "sur l'aile), VEd = 550 kN et force de chaînage 275 kN",
         data="IPE A 550 S275 portée (h 547, b 210, tw 9, tf 15,7, r 24) sur poteau PERSO S275 d'aile "
              "tfP = 15 mm, platine 430×200×12 S275, M20 8.8, 6 rangées × 2 files, p3 = 140, e2 = 30, "
              "e1 = 40, p1 = 70 (40 + 5 × 70 + 40 = 430), cordon 8 mm (a = 5,6), poutre non grugée. "
              "Les neuf valeurs publiées ne dépendent PAS du support : elles doivent sortir inchangées",
         inp=WE, rows=[
             ["§2.4", "Âme de la poutre – cisaillement sur hp (Av = 430 × 9 = 3870 mm²)", 614, rd("vwS"), "kN", ""],
             ["§2.4", "Un boulon en cisaillement Fv,Rd", 94, va("Fv_Rd"), "kN", E_FV],
             ["§2.4", "Pression diamétrale sur la platine – boulons d'extrémité", 107, va("Fbb_L"), "kN", E_AB],
             ["§2.4", "Pression diamétrale sur la platine – boulons intérieurs", 142, va("Fbi_L"), "kN", ""],
             ["§2.4", "Groupe de boulons FRd = 0,8·ns·Fv,Rd (Fv,Rd ≤ (Fb,Rd)min)", 902, rd("bg"), "kN", E_FV],
             ["§2.4", "Platine – cisaillement, min(brut ; net ; bloc) = rupture de bloc", 1182,
              lambda R: min(R.ck["pcg"].Rd, R.ck["pcn"].Rd, R.ck["pcb"].Rd), "kN", ""],
             ["§2.4", "Tying – platine en flexion (tronçon en T, mode 1)", 493, rd("ptT"), "kN", E_M],
             ["§2.4", "Tying – âme de la poutre en traction tw·hp·fu/γMu", 1513, rd("tnS"), "kN", ""],
             ["§2.2", "Cordon à pleine résistance en S275 : a ≥ 0,48·tw", 0.48, va("k_plein"), "-", ""]]),
    dict(id="BM2", titre="Cas (a) AILE — le cas par défaut du module : la pression diamétrale de l'aile "
                         "est la SEULE vérification du poteau à l'ELU",
         data="IPE 400 S355 sur HEB 300 S355, platine 290×220×10 S355, 4 × 2 M20 8.8, p1 = 70, e1 = 40, "
              "p3 = 140, zp = 40, a = 5, VEd = 200 kN",
         inp=BM2, rows=[
             ["§11.3", "Gauge minimal p3 ≥ twP + 2·rP + dw (le p3 = 90 usuel est REFUSÉ)",
              _p3_congeP, va("p3_congeP"), "mm", ""],
             ["§11.3", "Pince au bout d'aile e = (bP − p3)/2", _eP, va("e_P"), "mm", ""],
             ["§11.3", "Pince latérale de la platine e2 = (bp − p3)/2", _e2, va("e_2"), "mm", ""],
             ["§11.3", "Effort tranchant par boulon Fv,Ed = VEd/ns (groupe centré)", _FvEd, va("Fv_Ed"), "kN", ""],
             ["§11.3", "Platine – Fb,Rd boulons d'extrémité", _Fbb_L, va("Fbb_L"), "kN", ""],
             ["§11.3", "Platine – Fb,Rd boulons intérieurs", _Fbi_L, va("Fbi_L"), "kN", ""],
             ["§11.3", "Groupe de boulons FRd = 0,8·ns·Fv,Rd", _FRd_g, rd("bg"), "kN", ""],
             ["§11.3", "Platine – cisaillement brut VRd,g (2 plans, 1/1,27)", _VRdg_L, rd("pcg"), "kN", ""],
             ["§11.3", "Platine – cisaillement net VRd,n (2 plans)", _VRdn_L, rd("pcn"), "kN", ""],
             ["§11.3", "Platine – rupture de bloc VRd,b (2 blocs, k = 1)", _VRdb_L, rd("pcb"), "kN", ""],
             ["§11.3", "Âme portée – cisaillement sur hp (Av = hp·tw = 2494 mm²)", _VRd_wS, rd("vwS"), "kN", ""],
             ["§11.3", "Âme portée – taux (dimensionnante)", _V / _VRd_wS, et("vwS"), "-", ""],
             ["§11.3", "AILE DU POTEAU – Fb,Rd par boulon (αb du seul p1, k1 du gauge)", _Fb_P, va("Fb_P"), "kN", ""],
             ["§11.3", "Aile du poteau – VRd = ns·Fb,Rd", _VRd_pdP, rd("pdP"), "kN", ""],
             ["§11.3", "Cordons – qz = VEd/(2hp)", _qz, va("qz_w"), "N/mm", ""],
             ["§11.3", "Gorge de pleine résistance recommandée (0,553727·tw)", _a_plein, va("a_plein"), "mm", ""]]),
    dict(id="BM3", titre="Cas (a) TYING (NEd = 250 kN seul, γMu = 1,1) — les DEUX tronçons en T : "
                         "celui de la platine et celui de l'AILE DU POTEAU, le composant NEUF",
         data="Même assemblage qu'en BM2, NEd = 250 kN en tying : Ft,Rd,u = 160,364 kN par boulon, "
              "ΣFt,Rd,u = 1 282,91 kN, ew = dw/4 = 9,25 mm. L'aile du poteau (tfP = 19) est 3,66 fois "
              "plus forte que la platine (tp = 10) : avec une aile laminée épaisse, le composant neuf "
              "ne gouverne pas — il gouvernerait avec une aile mince ou un p3 plus grand",
         inp=BM3, rows=[
             ["§7.3", "Platine – m = (p3 − tw)/2 − 0,8·a·√2", _mT, va("m_T"), "mm", ""],
             ["§7.3", "Platine – n = min(e2 ; 1,25·m)", _nT, va("nT"), "mm", ""],
             ["§7.3", "Platine – Σleff (bords libres) = hp", _leffT, va("leff_T"), "mm", ""],
             ["§7.3", "Platine – FT,1 (mode 1, méthode 2)", _FT1_L, va("FT1"), "kN", ""],
             ["§7.3", "Platine – FT,2", _FT2_L, va("FT2"), "kN", ""],
             ["§7.3", "Platine – FT,Rd (dimensionnante en tying)", _FT_L, rd("ptT"), "kN", ""],
             ["§7.3", "AILE DU POTEAU – m = (p3 − twP)/2 − 0,8·rP", _mP, va("m_fP"), "mm", ""],
             ["§7.3", "Aile du poteau – n = min(e ; 1,25·m)", _nP, va("n_fP"), "mm", ""],
             ["§7.3", "Aile du poteau – Σleff = n1·p1 (Tab. 6.4, rangées en groupe)", _leffP, va("leff_fP"), "mm", ""],
             ["§7.3", "Aile du poteau – Mpl,1 = 0,25·Σleff·tfP²·fu/γMu", _MplP / 1e6,
              lambda R: R.Mpl1_fP / 1e6, "kNm", ""],
             ["§7.3", "Aile du poteau – FT,1", _FT1_P, va("FT1_P"), "kN", ""],
             ["§7.3", "Aile du poteau – FT,2 (déterminant)", _FT2_P, va("FT2_P"), "kN", ""],
             ["§7.3", "Aile du poteau – FT,Rd", _FT_P, rd("tfP"), "kN", ""],
             ["§7.3", "Rapport aile du poteau / platine", _FT_P / _FT_L,
              lambda R: R.FT_Rd_P / R.FT_Rd, "-", ""],
             ["§11.3", "Ft,Ed = NEd/ns", _FtEd, lambda R: R.ck["pp"].Ed, "kN", ""],
             ["§11.3", "Poinçonnement de la platine (t = 10)", _Bp_pl, va("Bp_pl"), "kN", ""],
             ["§11.3", "Poinçonnement de l'aile (t = 19)", _Bp_fl, va("Bp_P"), "kN", ""],
             ["§7.4", "Âme du poteau en traction transversale (ω = 1, beff = Σleff)", _Ft_wc, rd("twtP"), "kN", ""],
             ["§11.3", "Âme portée en traction sur hp (fu/γMu)", _tnS_u, rd("tnS"), "kN", ""],
             ["§11.3", "Cordons – qx = NEd/(2hp)", _qx_ty, va("qx_ty"), "N/mm", ""],
             ["§11.3", "Cordons – Fw,Rd,u = fu/(√3·βw·γMu)·a", _Fw_ty, rd("wT"), "N/mm", ""]]),
    dict(id="BM4", titre="Cas (b) ÂME DU POTEAU — la platine 290 × 220 du cas (a) NE TIENT PAS entre les "
                         "congés (198 mm) : p3 = 110 et bp = 190",
         data="Même poutre et même poteau, appui = Âme du poteau, platine 290×190×10, p3 = 110, "
              "e2 = 40. L'écrou dégage le congé (p3/2 + dw/2 = 73,5 ≤ 104, marge 30,5 mm) ; la semelle "
              "de l'IPE 400 (bS = 180) passe entre les ailes (252 mm), aucune recoupe n'est nécessaire — "
              "la limite 198 mm est celle de la PIÈCE d'attache, entre congés",
         inp=BM4, rows=[
             ["§11.3", "Largeur droite de l'âme entre congés hP − 2tfP − 2rP", _libre_droit, va("libre_droit"), "mm", ""],
             ["§11.3", "Demi-largeur droite (hP − 2tfP)/2 − rP", _demi_droit, va("demi_droit"), "mm", ""],
             ["§11.3", "Largeur maximale de la platine (bp ≤ ℓ − 2·j)", _bp_maxP, va("bp_maxP"), "mm", ""],
             ["§11.3", "Entraxe maximal des files p3 = 2·(demi-largeur − dw/2)", _p3_maxP, va("p3_maxP"), "mm", ""],
             ["§11.3", "Passage de la semelle hP − 2tfP − 2j", _libre_ailes, va("libre_ailes"), "mm", ""],
             ["§11.3", "Recoupe minimale si la poutre ne passait pas : (bP − twP)/2 + j − tp",
              _rec_min, va("rec_min"), "mm", ""],
             ["§11.3", "ÂME DU POTEAU – Fb,Rd par boulon (k1 du seul gauge, aucun bord libre)",
              _Fb_Pb, va("Fb_P"), "kN", ""],
             ["§11.3", "Âme du poteau – VRd = ns·Fb,Rd", _VRd_pdPb, rd("pdP"), "kN", ""],
             ["§11.3", "Âme du poteau – cisaillement local BRUT (Av = twP·hp)", _VRd_lP, rd("vlP"), "kN", ""],
             ["§11.3", "Âme du poteau – taux (Ed = VEd/2 = 100 kN)", _V / 2 / _VRd_lP, et("vlP"), "-", ""],
             ["§11.3", "Âme du poteau – cisaillement local NET", _VRd_nP, rd("vnP"), "kN", ""],
             ["§11.3", "Âme du poteau – taux net", _V / 2 / _VRd_nP, et("vnP"), "-", ""],
             ["§5.4", "Poinçonnement de l'âme (t = 11, γMu) — cas (b) en tying",
              _Bp_we, lambda R: compute(BM4T).Bp_P, "kN", ""]]),
]

NOCOVER = [
    ["Pression diamétrale sur l'AILE ou l'ÂME du poteau", "Aucun exemple publié : formule directe du "
     "Tab. 3.4 avec αb SANS pince d'extrémité (le poteau continue au-dessus et en dessous) et k1 du "
     "gauge — c'est le motif bearing(e1 = None) du moteur des doubles cornières, transposé sans "
     "changement de formule. Calcul manuel indépendant (BM2, BM4)."],
    ["Tronçon en T de l'AILE DU POTEAU", "Aucun exemple publié consulté pour une attache SIMPLE ; les "
     "formules des modes 1 / 2 / 3 sont ancrées par l'exemple §2.4 (BM1 et VAL-A), les longueurs "
     "efficaces viennent du Tableau 6.4 (aile non raidie, toutes les rangées intérieures — le NUMÉRO du "
     "tableau est le point P-1 de l'étude). Calcul manuel indépendant (BM3, VAL-B)."],
    ["Âme du poteau en traction transversale", "§6.2.6.3 avec ω = 1 et beff,t,wc = Σleff de l'aile : "
     "interprétation de l'outil (point P-5) — les guides ne listent pas ce composant pour les attaches "
     "simples. Calcul manuel indépendant (BM3, VAL-B)."],
    ["Cisaillement local de l'âme du poteau (cas b)", "Aucun exemple publié consulté : c'est le modèle "
     "même des modules poutre–poutre (SCI P358, élément support chargé d'un seul côté, Av = twP·hp), "
     "transposé sans changement de formule. Calcul manuel indépendant (BM4, VAL-C)."],
    ["Règles géométriques du cas (b) et gauge minimal", "Largeur droite entre congés, largeur de la "
     "platine, dégagement de l'écrou, passage de la semelle, recoupe en largeur, gauge "
     "p3 ≥ twP + 2·rP + dw : géométrie, pas de résistance — calculs manuels (BM2, BM4, VAL-C). La "
     "recoupe en LARGEUR n'est pas un grugeage (point tranché P-3)."],
    ["Traction ELU combinée avec VEd (interaction, tronçons en T avec fy/γM0, poinçonnement)",
     "Aucun exemple publié consulté : formules directes de l'EN 1993-1-8 (Tab. 3.4, §6.2.4) ; les "
     "tronçons en T sont ceux du tying, ancrés par l'exemple publié et VAL-A."],
    ["Glissement (catégories B et C)", "Formule directe §3.9, aucun exemple publié."],
]

VALID = [
    dict(id="VAL-A", titre="Tronçon en T de l'EXEMPLE PUBLIÉ (tying, fu/γMu) — m, Σleff et les trois "
                           "modes : l'ancre de la formule que l'aile du poteau réutilise",
         inp=WE, rows=[
        ["m = (p3 − tw)/2 − 0,8·a·√2", (140 - 9) / 2 - 0.8 * 5.6 * math.sqrt(2), 59.1643, va("m_T"), "mm"],
        ["n = min(e2 ; 1,25·m)", min(30.0, 1.25 * ((140 - 9) / 2 - 0.8 * 5.6 * math.sqrt(2))), 30.0, va("nT"), "mm"],
        ["Σleff (rangées en groupe, bords libres) = hp",
         2 * min(2 * ((140 - 9) / 2 - 0.8 * 5.6 * math.sqrt(2)) + 0.625 * 30 + 35, 40 + 35) + 4 * 70,
         430.0, va("leff_T"), "mm"],
        ["Mode 1, méthode 2 : (8n − 2ew)·Mpl,1/(2mn − ew(m + n))",
         (8 * 30.0 - 2 * 9.25) * (0.25 * 430 * 12 ** 2 * 430 / 1.1)
         / (2 * ((140 - 9) / 2 - 0.8 * 5.6 * math.sqrt(2)) * 30.0
            - 9.25 * (((140 - 9) / 2 - 0.8 * 5.6 * math.sqrt(2)) + 30.0)) / 1000,
         491.86, va("FT1"), "kN"],
        ["Mode 2 : (2·Mpl,2 + n·ΣFt)/(m + n)",
         (2 * (0.25 * 430 * 12 ** 2 * 430 / 1.1) + 30.0 * 12 * (0.9 * 800 * 245 / 1.1))
         / (((140 - 9) / 2 - 0.8 * 5.6 * math.sqrt(2)) + 30.0) / 1000, 783.20, va("FT2"), "kN"],
        ["Mode 3 : ΣFt,Rd = 12 × 0,9·fub·As/γMu", 12 * (0.9 * 800 * 245 / 1.1) / 1000, 1924.36,
         va("FT3"), "kN"]]),
    dict(id="VAL-B", titre="Tronçon en T de l'AILE DU POTEAU et âme en traction transversale (cas a, "
                           "tying NEd = 250 kN) — le composant NEUF, calculé hors moteur",
         inp=BM3, rows=[
        ["m = (p3 − twP)/2 − 0,8·rP (poteau laminé : le congé)", _mP, 42.900, va("m_fP"), "mm"],
        ["n = min(e ; 1,25·m) avec e = (bP − p3)/2 = 80", _nP, 53.625, va("n_fP"), "mm"],
        ["Σleff : Tab. 6.4, aucune rangée d'extrémité → n1·p1", _leffP, 280.0, va("leff_fP"), "mm"],
        ["Mpl,1 = 0,25·Σleff·tfP²·fu/γMu", _MplP / 1e6, 11.7161, lambda R: R.Mpl1_fP / 1e6, "kNm"],
        ["FT,1 (mode 1, méthode 2)", _FT1_P, 1296.99, va("FT1_P"), "kN"],
        ["FT,2", _FT2_P, 955.485, va("FT2_P"), "kN"],
        ["FT,3 = ΣFt,Rd,u", _FT3_P, 1282.91, va("FT3_P"), "kN"],
        ["FT,Rd de l'aile = min des trois modes", _FT_P, 955.485, rd("tfP"), "kN"],
        ["Rapport aile / platine (3,66 : l'aile ne gouverne pas)", _FT_P / _FT_L, 3.6562,
         lambda R: R.FT_Rd_P / R.FT_Rd, "-"],
        ["Âme du poteau en traction transversale ω·beff·twP·fu/γMu", _Ft_wc, 1428.00, rd("twtP"), "kN"]]),
    dict(id="VAL-C", titre="Cas (b) : la géométrie du poteau (largeur droite, dégagement de l'écrou, "
                           "recoupe) et le cisaillement local de l'âme — calculs indépendants",
         inp=BM4, rows=[
        ["Largeur droite entre congés hP − 2tfP − 2rP", _libre_droit, 208.0, va("libre_droit"), "mm"],
        ["Demi-largeur droite (hP − 2tfP)/2 − rP", _demi_droit, 104.0, va("demi_droit"), "mm"],
        ["Marge de l'écrou : demi-largeur − (p3/2 + dw/2)", _marge_b, 30.5,
         lambda R: R.demi_droit - (R.p_3 / 2 + R.d_w / 2), "mm"],
        ["Largeur maximale de la platine bp ≤ ℓ − 2·j", _bp_maxP, 198.0, va("bp_maxP"), "mm"],
        ["Recoupe minimale (bP − twP)/2 + j − tp", _rec_min, 139.5, va("rec_min"), "mm"],
        ["Âme du poteau – Fb,Rd par boulon (t = 11)", _Fb_Pb, 181.900, va("Fb_P"), "kN"],
        ["Âme du poteau – cisaillement local brut", _VRd_lP, 653.820, rd("vlP"), "kN"],
        ["Âme du poteau – cisaillement local net", _VRd_nP, 523.412, rd("vnP"), "kN"],
        ["Excentricité apportée au poteau e_col = tp (cas b)", _tp, 10.0, va("e_col"), "mm"]]),
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
