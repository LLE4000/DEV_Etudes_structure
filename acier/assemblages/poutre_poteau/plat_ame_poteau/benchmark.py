# -*- coding: utf-8 -*-
"""Benchmark du module plat d'âme sur poteau — quatre cas, et aucun nombre
de mémoire.

- **BM1 — HÉRITAGE** : l'exemple publié du guide (MSB Part 5 §3.4, *Worked
  Example: Fin Plate*) rejoué avec ``appui = Aile du poteau`` et un poteau
  PERSO d'aile 15 mm S275. Les DIX valeurs publiées doivent sortir
  INCHANGÉES : du côté pièce d'attache, boulons et âme portée, rien ne
  dépend du support (démontré par l'étude, script 00). C'est le contrôle
  d'héritage du module.
- **BM2** : le cas par défaut, appui sur l'AILE (étude §11.2).
- **BM3** : appui dans l'ÂME, variante **b1** (la poutre reste au-delà des
  bouts d'aile) — il NE PASSE PAS : boulons 185 %, déversement 111 %.
- **BM3'** : le même, redimensionné (5 × 2 M24, plat 320 × 12).
- **BM4** : appui dans l'ÂME, variante b0/b2 (about au contact de l'âme).

Chaque valeur de référence est écrite ici comme une EXPRESSION LITTÉRALE
indépendante (jamais une constante recopiée) : le moteur est relancé et
comparé à elle. Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à
vérifier. Un écart qui porte une explication (``exp``) ne compte pas comme
inattendu. Aucune formule n'a été ajustée pour coller à la source.
"""
import math

from acier.bibliotheques import PERSO
from acier.js import dv, mx
from .entrees import APPUI_AILE, APPUI_AME
from .moteur import compute, pb_bs5950

S3 = math.sqrt(3)

E_ARR = "Arrondis intermédiaires de la source (αb = 0,61 pour 0,606 ; Fv,Rd = 94 pour 94,08)."
E_AIRE = ("La source utilise l'aire catalogue A = 11700 mm² ; l'outil calcule A = 2b·tf + (h − 2tf)·tw "
          "+ (4 − π)·r² = 11729 mm² (+0,25 %), d'où Av = 6030 au lieu de 6001 mm².")

# --- les données de l'exemple publié, reprises telles quelles du module
# --- poutre–poutre, SUPPORT remplacé : un poteau PERSO dont l'aile a
# --- l'épaisseur de l'âme support de l'exemple (15 mm) et sa nuance (S275).
# --- Aucune des dix valeurs publiées ne dépend du support : hP, bP, twP et
# --- rP sont donc libres (ils ne servent qu'au dessin et, en cas (b), au
# --- cisaillement local — inactif ici).
WE = dict(appui=APPUI_AILE,
          prof_S=PERSO, hS_u=547, bS_u=210, twS_u=9, tfS_u=15.7, rS_u=24, nu_S="S275",
          prof_P=PERSO, hP_u=300, bP_u=300, twP_u=11, tfP_u=15, rP_u=27, nu_P="S275",
          nu_pl="S275", d_nt=0, d_nb=0, l_n=0, g_h=10, z_C=50,
          hp_u=360, tp_u=10, bp_u=160, boulon_u="M20", classe="8.8",
          n1_u=5, n2_u="2", p1_u=70, p2_u=60, e1_u=40, e2b_u=40, V_Ed=350, g_M1=1.0)

# --- BM2 : le cas par défaut (appui sur l'aile) — aucune entrée à forcer
BM2 = {}
# --- BM3 : appui dans l'âme, variante b1 (about au-delà des bouts d'aile)
BM3 = dict(appui=APPUI_AME, g_h=155, bp_u=245)
# --- BM3' : le même, redimensionné
BM3P = dict(appui=APPUI_AME, g_h=155, boulon_u="M24", n1_u=5, n2_u="2", p1_u=60,
            p2_u=60, hp_u=320, tp_u=12, bp_u=295)
# --- BM4 : appui dans l'âme, about au contact (la poutre passe entre les ailes)
BM4 = dict(appui=APPUI_AME)


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
def _pb(lam, py):
    """Perry-Robertson de BS 5950-1 Annexe B.2.1, réécrit ici in extenso —
    la valeur de référence ne vient PAS de la fonction du moteur."""
    lam0 = 0.4 * math.sqrt(math.pi ** 2 * 205000 / py)
    eta = 0.007 * (lam - lam0)
    pE = math.pi ** 2 * 205000 / lam ** 2
    phi = (py + (eta + 1) * pE) / 2
    return pE * py / (phi + math.sqrt(phi * phi - pE * py))


# --- géométrie et matériaux de référence (étude §11.1)
_fy, _fu, _bw = 355.0, 510.0, 0.9                 # S355
_d, _d0, _As, _dw = 20.0, 22.0, 245.0, 37.0       # M20
_fub, _av = 800.0, 0.6                            # classe 8.8, filet cisaillé
_Fv = _av * _fub * _As / 1.25 / 1000              # 94,08 kN
_Fvu = _av * _fub * _As / 1.1 / 1000              # tying, γMu = 1,1
_hP, _bP, _twP, _tfP, _rP = 300.0, 300.0, 11.0, 19.0, 27.0        # HEB 300
_hS, _bS, _twS, _tfS, _rS = 400.0, 180.0, 8.6, 13.5, 21.0         # IPE 400
_V = 200.0
# --- plat 290 × 100 × 10, 4 M20, p1 70, e1 40, gh 10, e2,b 40, zp 40
_hp, _tp, _bp, _n1, _p1, _e1, _e2b, _gh, _zp = 290.0, 10.0, 100.0, 4, 70.0, 40.0, 40.0, 10.0, 40.0
_z2 = _gh + _e2b                                  # bras du groupe, cas (a) et b0/b2
_MS2 = _V * _z2 / 1000
_Ip2 = _n1 * _p1 ** 2 * (_n1 ** 2 - 1) / 12
_Fz2 = _V / _n1
_Fx2 = _MS2 * 1000 * ((_n1 - 1) * _p1 / 2) / _Ip2
_F2 = math.hypot(_Fz2, _Fx2)
_e2p = _bp - _gh - _e2b                           # pince du plat au bord libre
_e1bot = _hp - _e1 - (_n1 - 1) * _p1
# plat : pression diamétrale dans les deux directions (Tab. 3.4)
_abv_L = min(min(_e1, _e1bot) / (3 * _d0), _p1 / (3 * _d0) - 0.25, _fub / _fu, 1)
_k1v_L = min(2.8 * _e2p / _d0 - 1.7, 2.5)
_Fbv_L = _k1v_L * _abv_L * _fu * _d * _tp / 1.25 / 1000
_abh_L = min(_e2p / (3 * _d0), _fub / _fu, 1)
_k1h_L = min(2.8 * min(_e1, _e1bot) / _d0 - 1.7, 1.4 * _p1 / _d0 - 1.7, 2.5)
_Fbh_L = _k1h_L * _abh_L * _fu * _d * _tp / 1.25 / 1000
_ipd_L = math.hypot(_Fz2 / _Fbv_L, _Fx2 / _Fbh_L)
_VRdg_L = _hp * _tp * _fy / (1.27 * S3 * 1.0) / 1000
_VRdn_L = (_hp - _n1 * _d0) * _tp * _fu / (S3 * 1.25) / 1000
_Ant_L = _tp * (_e2p - 0.5 * _d0)
_Anv_L = _tp * (_hp - _e1 - (_n1 - 0.5) * _d0)
_Veff_L = 0.5 * _fu * _Ant_L / 1.25 / 1000 + _fy * _Anv_L / S3 / 1000
# âme de la poutre portée : aucune pince au-delà du grugeage (poutre non grugée)
_e1b = _zp + _e1
_abv_S = min(_e1b / (3 * _d0), _p1 / (3 * _d0) - 0.25, _fub / _fu, 1)
_Fbv_S = min(2.8 * _e2b / _d0 - 1.7, 2.5) * _abv_S * _fu * _d * _twS / 1.25 / 1000
_abh_S = min(_e2b / (3 * _d0), _fub / _fu, 1)
_Fbh_S = min(2.8 * _e1b / _d0 - 1.7, 1.4 * _p1 / _d0 - 1.7, 2.5) * _abh_S * _fu * _d * _twS / 1.25 / 1000
_ipd_S = math.hypot(_Fz2 / _Fbv_S, _Fx2 / _Fbh_S)
_A_S = 2 * _bS * _tfS + (_hS - 2 * _tfS) * _twS + (4 - math.pi) * _rS ** 2
_Av_S = max(_A_S - 2 * _bS * _tfS + (_twS + 2 * _rS) * _tfS, (_hS - 2 * _tfS) * _twS)
_VRdg_S = _Av_S * _fy / S3 / 1000
# cordons (deux cordons verticaux pleine hauteur)
_fvw = _fu / (S3 * _bw * 1.25)
_qz2 = _V * 1000 / (2 * _hp)
_qx2 = 3 * _MS2 * 1e6 / _hp ** 2
_a_plein = _tp * _fy * _bw * 1.25 / (2 * _fu * 1.0)
# poteau, cas (b) : cisaillement local de l'âme, un seul côté chargé
_VRd_lP = _twP * _hp * _fy / (S3 * 1.0) / 1000
# géométrie du cas (b)
_libre_droit = _hP - 2 * _tfP - 2 * _rP
_demi_droit = (_hP - 2 * _tfP) / 2 - _rP
_libre_ailes = _hP - 2 * _tfP - 2 * 5.0
_gh_b1 = (_bP - _twP) / 2 + 5.0
_rec_min = _gh_b1 - _gh
# --- BM3 : variante b1, gh = 155
_gh3 = 155.0
_z3 = _gh3 + _e2b
_MS3 = _V * _z3 / 1000
_Fx3 = _MS3 * 1000 * ((_n1 - 1) * _p1 / 2) / _Ip2
_F3 = math.hypot(_Fz2, _Fx3)
_lam3 = 2.8 * math.sqrt(_z3 * _hp / (1.5 * _tp ** 2))
_pb3 = _pb(_lam3, _fy)
_Wel3 = _tp * _hp ** 2 / 6
_Mb3 = _Wel3 * _pb3 / 1.1 / 1e6
# --- BM3' : 5 × 2 M24, p1 = p2 = 60, plat 320 × 12
_d4, _d04, _As4 = 24.0, 26.0, 353.0
_n1p, _n2p, _p1p, _p2p, _hpp, _tpp, _bpp = 5, 2, 60.0, 60.0, 320.0, 12.0, 295.0
_Fv4 = _av * _fub * _As4 / 1.25 / 1000
_Ipp = (_n1p * _n2p) * (_p1p ** 2 * (_n1p ** 2 - 1) + _p2p ** 2 * (_n2p ** 2 - 1)) / 12
_zp4 = _gh3 + _e2b + (_n2p - 1) * _p2p / 2
_MS4 = _V * _zp4 / 1000
_Fz4 = _V / (_n1p * _n2p) + _MS4 * 1000 * ((_n2p - 1) * _p2p / 2) / _Ipp
_Fx4 = _MS4 * 1000 * ((_n1p - 1) * _p1p / 2) / _Ipp
_F4 = math.hypot(_Fz4, _Fx4)
_lam4 = 2.8 * math.sqrt(_zp4 * _hpp / (1.5 * _tpp ** 2))
_pb4 = _pb(_lam4, _fy)
_Wel4 = _tpp * _hpp ** 2 / 6
_Mb4 = _Wel4 * _pb4 / 1.1 / 1e6
_e1b4 = _zp + _e1
_abv_S4 = min(_e1b4 / (3 * _d04), _p1p / (3 * _d04) - 0.25, _fub / _fu, 1)
_k1v_S4 = min(2.8 * _e2b / _d04 - 1.7, 1.4 * _p2p / _d04 - 1.7, 2.5)
_Fbv_S4 = _k1v_S4 * _abv_S4 * _fu * _d4 * _twS / 1.25 / 1000
_abh_S4 = min(_e2b / (3 * _d04), _p2p / (3 * _d04) - 0.25, _fub / _fu, 1)
_k1h_S4 = min(2.8 * _e1b4 / _d04 - 1.7, 1.4 * _p1p / _d04 - 1.7, 2.5)
_Fbh_S4 = _k1h_S4 * _abh_S4 * _fu * _d4 * _twS / 1.25 / 1000
_ipd_S4 = math.hypot(_Fz4 / _Fbv_S4, _Fx4 / _Fbh_S4)
# --- tying (NEd = 250 kN, γMu = 1,1) sur le cas par défaut
_Nty = 250.0
_ptn_u = 0.9 * (_hp - _n1 * _d0) * _tp * _fu / 1.1 / 1000
_tnS_u = 0.9 * _twS * (_hp - _n1 * _d0) * _fu / 1.1 / 1000
_qx_ty = _Nty * 1000 / (2 * _hp)
_Fw_ty = _fu / (S3 * _bw * 1.1) * 5.0


SRC = dict(
    titre="Steel Buildings in Europe – Multi-Storey Steel Buildings – Part 5: Joint Design "
          "— §3.4 Worked Example: Fin Plate (BM1) ; BM2 à BM4 : étude du 01/11/2026, "
          "docs/assemblages/POUTRE_POTEAU.md §11.2, recalculée à la main",
    org="ArcelorMittal / Peiner Träger / Corus – contenu technique CTICM et SCI (projet RFCS SECHALO, RFS2-CT-2008-0030)",
    annee="feuilles de calcul datées 06/2009",
    norme="EN 1993-1-8:2005 et EN 1993-1-1:2005, valeurs recommandées (le module applique par défaut "
          "l'ANB belge : γM1 = 1,10 — l'exemple est rejoué avec γM1 = 1,0 comme la source)",
    url="https://steelconstruction.info/images/5/53/SBE_MS5.pdf")

BENCH = [
    dict(id="BM1", titre="HÉRITAGE — §3.4 Worked Example: Fin Plate sur un POTEAU (appui sur l'aile), "
                         "VEd = 350 kN, 2 files de boulons, z = 80 mm, p. 5-38 à 5-45",
         data="IPE A 550 S275 portée (h 547, b 210, tw 9, tf 15,7, r 24) sur poteau PERSO S275 d'aile "
              "tfP = 15 mm, plat 360×160×10 S275, M20 8.8, n1 = 5, n2 = 2, p1 = 70, p2 = 60, e1 = 40, "
              "e2 = 50, e2,b = 40, e1,b = 90, poutre non grugée. Les dix valeurs publiées ne dépendent "
              "PAS du support : elles doivent sortir inchangées",
         inp=WE, rows=[
             ["p. 5-40", "Inertie polaire du groupe de boulons I", 107000, va("Ip"), "mm²", ""],
             ["p. 5-40", "Groupe excentré – cisaillement des boulons VRd", 584, vr("bv"), "kN", E_ARR],
             ["p. 5-41", "Pression diamétrale sur le plat VRd", 605, vr("pdL"), "kN", E_ARR],
             ["p. 5-42", "Pression diamétrale sur l'âme VRd", 624, vr("pdS"), "kN", ""],
             ["p. 5-43", "Plat – cisaillement section brute VRd,g", 450, rd("pcg"), "kN", ""],
             ["p. 5-43", "Plat – cisaillement section nette VRd,n", 497, rd("pcn"), "kN", ""],
             ["p. 5-43", "Plat – rupture de bloc VRd,b", 483, rd("pcb"), "kN", ""],
             ["p. 5-45", "Âme – cisaillement section brute VRd,g", 953, rd("vgS"), "kN", E_AIRE],
             ["p. 5-45", "Âme – cisaillement section nette VRd,n", 995, rd("vnS"), "kN", E_AIRE],
             ["p. 5-45", "Âme – rupture de bloc VRd,b (non grugée : (n1 − 1)·d0)", 507, rd("vbS"), "kN", ""]]),
    dict(id="BM2", titre="Cas (a) AILE — le cas par défaut du module : aucune vérification locale de "
                         "l'aile, le groupe « Poteau porteur » est VIDE",
         data="IPE 400 S355 sur HEB 300 S355, plat 290×100×10 S355, 4 M20 8.8, p1 = 70, e1 = 40, "
              "gh = 10, e2,b = 40, zp = 40, a = 5, VEd = 200 kN",
         inp=BM2, rows=[
             ["§11.2", "Bras du groupe z = gh + e2,b", _z2, va("z_p"), "mm", ""],
             ["§11.2", "Moment d'excentricité MS = VEd·z", _MS2, va("M_S"), "kNm", ""],
             ["§11.2", "Inertie polaire Ip = n·p1²(n1² − 1)/12", _Ip2, va("Ip"), "mm²", ""],
             ["§11.2", "Effort du boulon le plus sollicité F,Ed", _F2, va("F_b"), "kN", ""],
             ["§11.2", "Taux des boulons (Fv,Rd = 94,08 kN)", _F2 / _Fv, et("bv"), "-", ""],
             ["§11.2", "Plat – Fb,ver,Rd", _Fbv_L, va("Fbv_L"), "kN", ""],
             ["§11.2", "Plat – Fb,hor,Rd", _Fbh_L, va("Fbh_L"), "kN", ""],
             ["§11.2", "Plat – taux de pression diamétrale (interaction)", _ipd_L, et("pdL"), "-", ""],
             ["§11.2", "Plat – cisaillement brut VRd,g (1/1,27)", _VRdg_L, rd("pcg"), "kN", ""],
             ["§11.2", "Plat – cisaillement net VRd,n", _VRdn_L, rd("pcn"), "kN", ""],
             ["§11.2", "Plat – rupture de bloc Veff,2,Rd", _Veff_L, rd("pcb"), "kN", ""],
             ["§11.2", "Âme portée – taux de pression diamétrale", _ipd_S, et("pdS"), "-", ""],
             ["§11.2", "Âme portée – cisaillement brut VRd,g", _VRdg_S, rd("vgS"), "kN", ""],
             ["§11.2", "Cordons – Fw,Ed = √(qz² + qx²)", math.hypot(_qz2, _qx2), va("qw"), "N/mm", ""],
             ["§11.2", "Cordons – Fw,Rd = fvw,d·a", _fvw * 5, rd("w"), "N/mm", ""],
             ["§11.2", "Gorge de pleine résistance recommandée", _a_plein, va("a_plein"), "mm", ""]]),
    dict(id="BM3", titre="Cas (b) ÂME, variante b1 (la poutre reste au-delà des bouts d'aile) — "
                         "l'attache NE PASSE PAS : deux taux au-dessus de 1",
         data="Même pièce, même poutre, même poteau, appui = Âme du poteau, gh = 155 mm (≥ (bP − twP)/2 "
              "+ j = 149,5 mm), plat 290×245×10. z passe de 50 à 195 mm (× 3,9) et le plat devient LONG",
         inp=BM3, rows=[
             ["§11.2", "gh minimal de la variante b1 = (bP − twP)/2 + j", _gh_b1, va("gh_b1"), "mm", ""],
             ["§11.2", "Bras du groupe z = gh + e2,b", _z3, va("z_p"), "mm", ""],
             ["§11.2", "Moment d'excentricité MS", _MS3, va("M_S"), "kNm", ""],
             ["§11.2", "Effort du boulon le plus sollicité F,Ed", _F3, va("F_b"), "kN", ""],
             ["§11.2", "Taux des boulons (NON VÉRIFIÉ)", _F3 / _Fv, et("bv"), "-", ""],
             ["§11.2", "Élancement de déversement λLT", _lam3, va("lam_LT"), "-", ""],
             ["§11.2", "Contrainte de déversement pb", _pb3, va("pb_LT"), "MPa", ""],
             ["§11.2", "Module élastique Wel = tp·hp²/6", _Wel3, va("Wel_p"), "mm³", ""],
             ["§11.2", "Moment résistant Mb,Rd = Wel·pb/γM1 (γM1 = 1,10 ANB)", _Mb3, rd("plt"), "kNm", ""],
             ["§11.2", "Taux de déversement du plat (NON VÉRIFIÉ)", _MS3 / _Mb3, et("plt"), "-", ""],
             ["§11.2", "Âme du poteau – cisaillement local VRd = twP·hp·fy/(√3·γM0)", _VRd_lP, rd("vlP"), "kN", ""],
             ["§11.2", "Âme du poteau – taux (Ed = VEd/2 = 100 kN)", _V / 2 / _VRd_lP, et("vlP"), "-", ""]]),
    dict(id="BM3'", titre="Cas (b) ÂME, variante b1 REDIMENSIONNÉE — les deux taux de l'étude "
                          "reviennent sous 1 ; l'outil trouve DEUX autres défauts, voir les deux "
                          "dernières lignes",
         data="5 × 2 M24 8.8, p1 = p2 = 60, plat 320×295×12, gh = 155, z = 225 mm. ATTENTION : avec un "
              "M24 (d0 = 26) l'entraxe p2 = 60 mm reste SOUS le minimum 2,4·d0 = 62,4 mm du Tableau 3.3, "
              "et la pression diamétrale de l'âme portée (IPE 400, tw 8,6) dépasse 1 — l'étude ne "
              "vérifiait que les deux taux ci-dessous",
         inp=BM3P, rows=[
             ["§11.2", "Inertie polaire Ip = n·[p1²(n1² − 1) + p2²(n2² − 1)]/12", _Ipp, va("Ip"), "mm²", ""],
             ["§11.2", "Bras du groupe z", _zp4, va("z_p"), "mm", ""],
             ["§11.2", "Composantes Fz / Fx — ici Fz", _Fz4, va("Fz_b"), "kN", ""],
             ["§11.2", "Composante Fx", _Fx4, va("Fx_b"), "kN", ""],
             ["§11.2", "Effort du boulon le plus sollicité F,Ed", _F4, va("F_b"), "kN", ""],
             ["§11.2", "Taux des boulons (Fv,Rd = 135,552 kN pour un M24 filet)", _F4 / _Fv4, et("bv"), "-", ""],
             ["§11.2", "Élancement λLT", _lam4, va("lam_LT"), "-", ""],
             ["§11.2", "Contrainte pb", _pb4, va("pb_LT"), "MPa", ""],
             ["§11.2", "Moment résistant Mb,Rd", _Mb4, rd("plt"), "kNm", ""],
             ["§11.2", "Taux de déversement du plat", _MS4 / _Mb4, et("plt"), "-", ""],
             ["calcul manuel", "Âme portée – taux de pression diamétrale : DÉFAUT trouvé par l'outil",
              _ipd_S4, et("pdS"), "-", ""],
             ["Tab. 3.3", "Entraxe p2 minimal 2,4·d0 (p2 = 60 mm saisi) : NON CONFORME",
              2.4 * _d04, lambda R: [x.min for x in R.dist if x.id == "p2"][0], "mm", ""]]),
    dict(id="BM4", titre="Cas (b) ÂME, about au contact de l'âme (b0 / b2) — tous les taux de BM2 "
                         "reviennent, et le cisaillement local de l'âme du poteau s'ajoute",
         data="Même pièce qu'en BM2, appui = Âme du poteau, gh = 10, j = 5 : la semelle de l'IPE 400 "
              "(bS = 180) passe entre les ailes du HEB 300 (hP − 2tfP − 2j = 252 mm), aucune recoupe "
              "n'est nécessaire ; une HEA 300 portée (bS = 300) l'exigerait",
         inp=BM4, rows=[
             ["§11.2", "Bras du groupe z (retombé à gh + e2,b)", _z2, va("z_p"), "mm", ""],
             ["§11.2", "Taux des boulons (identique à BM2)", _F2 / _Fv, et("bv"), "-", ""],
             ["§11.2", "Largeur droite de l'âme entre congés hP − 2tfP − 2rP", _libre_droit, va("libre_droit"), "mm", ""],
             ["§11.2", "Demi-largeur droite (hP − 2tfP)/2 − rP", _demi_droit, va("demi_droit"), "mm", ""],
             ["§11.2", "Passage de la semelle hP − 2tfP − 2j", _libre_ailes, va("libre_ailes"), "mm", ""],
             ["§11.2", "Recoupe minimale si la poutre ne passait pas : (bP − twP)/2 + j − gh",
              _rec_min, va("rec_min"), "mm", ""],
             ["§11.2", "Âme du poteau – cisaillement local VRd", _VRd_lP, rd("vlP"), "kN", ""],
             ["§11.2", "Âme du poteau – taux (Ed = VEd/2)", _V / 2 / _VRd_lP, et("vlP"), "-", ""]]),
]

NOCOVER = [
    ["Cisaillement local de l'âme du poteau (cas b)", "Aucun exemple publié consulté : c'est le modèle "
     "même des trois modules poutre–poutre (SCI P358, élément support chargé d'un seul côté, "
     "Av = twP·hp), transposé sans changement de formule. Calcul manuel indépendant (BM3, BM4)."],
    ["Déversement du plat LONG en âme de poteau", "La courbe est celle du module plat d'âme "
     "(BS 5950-1 Annexe B.2.1), ancrée par son VAL-A ; son application à z ≈ 200 mm est NEUVE : "
     "calcul manuel indépendant (BM3, VAL-A)."],
    ["Règles géométriques du cas (b)", "Largeur droite entre congés, passage de la semelle entre "
     "ailes, recoupe des semelles en largeur : géométrie, pas de résistance — calculs manuels (BM4). "
     "La recoupe en LARGEUR n'est pas un grugeage (point tranché P-3 de l'étude)."],
    ["Aucune vérification locale de l'aile sous VEd seul (cas a)", "Aucun modèle publié n'a été trouvé "
     "pour une aile de poteau chargée par un plat soudé sur chant au droit de l'âme : le groupe "
     "« Poteau porteur » est VIDE et la limite est écrite (point tranché P-2)."],
    ["Tying — boulons en cisaillement, plat et âme en traction", "La bascule nat_N / γMu vient du "
     "module platine d'about (MSB P5 §2.3) ; pour un plat d'âme, la force de chaînage est dans le PLAN "
     "du plat : elle cisaille les boulons. Aucun exemple publié consulté pour ce composant — formules "
     "directes du Tab. 3.4 et de l'EC3-1-1 §6.2.3(2), calcul manuel indépendant (VAL-B)."],
    ["Soudure (double cordon sous V et VEd·z, et sous la force de chaînage)", "Aucun exemple publié "
     "consulté : calcul manuel indépendant (BM2, VAL-B). Résistance : EN 1993-1-8 §4.5.3.3."],
    ["Glissement (catégories B et C)", "Formule directe §3.9, aucun exemple publié."],
]

VALID = [
    dict(id="VAL-A", titre="Déversement du plat LONG sur le cas (b) variante b1 (plat 290×10, "
                           "z = 195 > tp/0,15 = 66,7) — la courbe de BS 5950-1 réécrite in extenso",
         inp=BM3, rows=[
        ["Élancement λLT = 2,8·√(z·hp/(1,5·tp²))", _lam3, 54.36616595, va("lam_LT"), "-"],
        ["Contrainte de déversement pb (Perry-Robertson, E = 205 000)", _pb3, 276.5156658, va("pb_LT"), "MPa"],
        ["Moment résistant Mb,Rd = Wel·pb/γM1 (γM1 = 1,10 ANB)", _Mb3, 35.23479923, rd("plt"), "kNm"],
        ["Plateau : pb = py pour λ ≤ λL0", 355, 355, lambda R: pb_bs5950(20, 355), "MPa"]]),
    dict(id="VAL-B", titre="Cas (a) par défaut : le modèle (bras, groupe, cordons) et le TYING "
                           "(NEd = 250 kN, γMu = 1,1) — calculs indépendants",
         inp=BM2, rows=[
        ["Moment d'excentricité MS = VEd·z", _MS2, 10.0, va("M_S"), "kNm"],
        ["Effort maximal sur un boulon F,Ed", _F2, 65.85388898, va("F_b"), "kN"],
        ["Taux – pression diamétrale de l'âme portée (interaction)", _ipd_S, 0.5348613587, et("pdS"), "-"],
        ["Pince e2 du plat au bord libre", _e2p, 50.0, va("e2_p"), "mm"],
        ["Pince basse du plat e1' = hp − e1 − 3·p1", _e1bot, 40.0, va("e1bot"), "mm"],
        ["Soudure – qz = VEd/(2hp)", _qz2, 344.8275862, va("qz_w"), "N/mm"],
        ["Soudure – qx = 3·MS/hp²", _qx2, 356.7182, va("qx_w"), "N/mm"],
        ["Soudure – résistance Fw,Rd = fvw,d·a (a = 5)", _fvw * 5, 1308.66061, rd("w"), "N/mm"],
        ["Excentricité apportée au poteau e_col = hP/2 + z", _hP / 2 + _z2, 200.0, va("e_col"), "mm"],
        ["Moment apporté au poteau VEd·e_col (information)", _V * (_hP / 2 + _z2) / 1000, 40.0,
         va("M_col"), "kNm"]]),
    dict(id="VAL-C", titre="TYING du plat d'âme (NEd = 250 kN seul, γMu = 1,1) : les boulons en "
                           "CISAILLEMENT — la force de chaînage est dans le plan du plat",
         inp=dict(BM2, N_Ed=250, nat_N="Tying (accidentel)"), rows=[
        ["Un boulon en cisaillement Fv,Rd,u = αv·fub·A/γMu", _Fvu, 106.9090909, va("Fv_Rd_u"), "kN"],
        ["Le groupe : n·Fv,Rd,u", 4 * _Fvu, 427.6363636, lambda R: R.n_b * R.Fv_Rd_u, "kN"],
        ["Plat – traction section nette 0,9·Anet·fu/γMu", _ptn_u, 842.8909091, rd("ptn"), "kN"],
        ["Âme portée – traction section nette (hauteur participante hp)", _tnS_u, 724.8858182, rd("tnS"), "kN"],
        ["Cordons – qx = NEd/(2hp)", _qx_ty, 431.0344828, va("qx_ty"), "N/mm"],
        ["Cordons – Fw,Rd,u = fu/(√3·βw·γMu)·a", _Fw_ty, 1487.114330, rd("wT"), "N/mm"]]),
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
