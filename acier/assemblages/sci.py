# -*- coding: utf-8 -*-
"""Règles ADOPTÉES par le lot de clôture du 27/11/2026 — briques pures, sans écran.

Le bureau a retenu, pour les platines d'about de hauteur partielle et pour les
plats d'âme, la procédure du guide « Joints in Steel Construction : Simple
Joints to Eurocode 3 » (SCI P358, édition de 2014), lue dans le texte le
27/11/2026. Ce fichier porte les règles de ce guide que les moteurs
n'appliquaient pas, ou appliquaient autrement ; chacune cite sa page et
l'exemple publié qui l'ancre au banc (``benchmark.py`` de chaque module).
Les COEFFICIENTS PARTIELS restent ceux du projet (annexe belge) : le guide
est écrit pour l'annexe britannique (γM2 = 1,1 pour les sections nettes),
qui ne se reprend pas.

Registre : ``docs/assemblages/ECARTS_ET_CORRECTIONS.md`` §8 (ancien et nouveau
résultat de chaque règle, test qui la garde).

PURETÉ : bibliothèque standard seulement (``math``) et ``acier.js`` (la
sémantique numérique commune, comme ``mecanique.py``).

Unités : mm, mm², MPa, kN, kNm.
"""
import math

from acier.js import INF, dv, mn, mx

S3 = math.sqrt(3)
K_ROT = 0.8              # un boulon ne porte pas plus de 0,8·Fv,Rd : traction parasite due à la rotation (Check 8)
K_AV = 0.9               # aire de cisaillement d'une section RECTANGULAIRE (Check 4 p. 17, Check 5 p. 18 ; p. 121, 124)
K_CORDON_PLATINE = 0.8   # platine d'about : gorge du détail éprouvé = 0,8 × pleine résistance (annexe C.2, p. 279)
E_ACIER = 210000.0       # EN 1993-1-1 §3.2.6
ALPHA_LT_PLAT = 0.76     # imperfection de la Table 5.2 (p. 120) : expression (6.56) de l'EN 1993-1-1
SEUIL_PLAT_LONG = 0.15   # plat LONG si zp > tp/0,15 (SCI, p. 109) — le MSB P5 le juge sur z, au centre du groupe
N_2D = 2.0               # « all end and edge distances ≥ 2d » : plats d'âme (SCI §5.3 (5), p. 109 ; MSB P5 §3.1)


# =========================================================================
#  1. Platine d'about — groupe de boulons (Check 8, p. 21-22)
# =========================================================================
def groupe_boulons(Fv, Fb, n):
    """Résistance du groupe en cisaillement : ``(FRd, branche)``.

    ``Fb`` : les pressions diamétrales d'UN boulon à considérer (platine en
    extrémité, platine en partie courante, élément porteur) — le guide n'en
    garde qu'une, la plus petite : Fb,Rd = min(Fb,Rd,p ; Fb,Rd,2), avec
    αb,p = min(e1/3d0 ; p1/3d0 − 1/4 ; fub/fu,p ; 1).

    - Fb,Rd ≤ 0,8·Fv,Rd → FRd = n·Fb,Rd ;
    - Fb,Rd > 0,8·Fv,Rd → FRd = 0,8·n·Fv,Rd (« the reduction factor 0.8
      allows for the presence of tension in the bolts »).

    La règle est monotone en épaisseur et ne dépasse jamais 0,8·n·Fv,Rd.

    Référence : SCI P358 Check 8, p. 21-22. Ancre : exemple 1 du guide,
    p. 38-40 (602 et 902 kN) ; A1 : 598,15 kN.
    Remplace : MSB P5 §2.2.4 plafonné (``mecanique.groupe_boulons``)."""
    fb = min(Fb)
    if fb <= K_ROT * Fv:
        return n * fb, "n·Fb,Rd (plus petite pression diamétrale)"
    return K_ROT * n * Fv, "0,8·n·Fv,Rd (cisaillement des boulons)"


# =========================================================================
#  2. Platine d'about courte — flexion dans son plan (Check 9, p. 23)
# =========================================================================
def platine_flexion_plan(tp, hp, fy_p, p3, tw, gM0):
    """VRd,ip = 2·tp·hp²·fy,p/(3·(p3 − tw)·γM0), en kN — à vérifier EN PLUS du
    cisaillement de la platine quand hp < 1,36·p3 (platine courte devant
    l'entraxe des files). ``INF`` si p3 ≤ tw.

    Référence : SCI P358 Check 9, p. 23. Absente du MSB P5."""
    return dv(2 * tp * hp * hp * fy_p, 3 * (p3 - tw) * gM0) / 1000 if p3 > tw else INF


# =========================================================================
#  3. Platine d'about — cisaillement local de l'élément porteur (Check 10, p. 24)
# =========================================================================
def longueurs_ame_porteuse(e1t, e1b, p3, d, poteau=False):
    """``(et, eb)`` : les longueurs cisaillées au-dessus de la première rangée
    et sous la dernière, mesurées sur l'élément PORTEUR.

    - et = min(e1,t ; 5d) — e1,t : de la première rangée au dessus de la
      poutre porteuse ; ``None`` (poteau qui continue) : 5d ;
    - eb = min(e1,b ; p3/2 ; 5d) pour une poutre porteuse ; min(p3/2 ; 5d)
      pour un poteau.

    Av = t2·(et + (n1 − 1)·p1 + eb) ; Av,net = Av − n1·d0·t2 ; l'effort par
    plan est VEd/2.

    Référence : SCI P358 Check 10, p. 24. Ancre : exemple 1, p. 42 (et = 90,
    eb = 70 : 742 et 795 kN). Remplace Av = t2·hp (les pinces de la platine)."""
    et = 5.0 * d if e1t is None else mn(mx(e1t, 0.0), 5.0 * d)
    eb = mn(p3 / 2.0, 5.0 * d) if (poteau or e1b is None) else mn(mx(e1b, 0.0), p3 / 2.0, 5.0 * d)
    return et, eb


# =========================================================================
#  4. Platine d'about en CHAÎNAGE — longueur efficace (Check 11, p. 26)
# =========================================================================
def leff_chainage(n1, p1, e1h, e1b, p3, tw, a, d0):
    """Σleff = e1A(haut) + e1A(bas) + (n1 − 1)·p1A, avec
    e1A = e1 mais ≤ 0,5·(p3 − tw − 2·a·√2) + d0/2 et
    p1A = p1 mais ≤ p3 − tw − 2·a·√2 + d0.

    Les deux guides impriment la même règle. Elle vaut pour le CHAÎNAGE
    (situation accidentelle, fu/γMu) ; la traction à l'ELU garde les
    mécanismes de l'EN 1993-1-8 (``mecanique.leff_platine_bords_libres``).

    Référence : SCI P358 Check 11, p. 26 ; MSB P5 §2.3.1, p. 5-13. Ancre :
    exemple 2 du SCI, p. 47 (290 mm)."""
    libre = p3 - tw - 2 * a * math.sqrt(2)
    plafond_e = 0.5 * libre + d0 / 2.0
    plafond_p = libre + d0
    return mn(e1h, plafond_e) + mn(e1b, plafond_e) + (n1 - 1) * mn(p1, plafond_p)


# =========================================================================
#  5. CHAÎNAGE sur l'âme d'un poteau en I ou H (Check 14)
# =========================================================================
def ame_poteau_platine(fu2, t2, d2, n1, p1, p3, d0, gMu):
    """Résistance au chaînage de l'âme du poteau sous une PLATINE D'ABOUT :
    FRd,u = 8·Mpl/(1 − β1)·(η1 + 1,5·(1 − β1)^0,5·(1 − γ1)^0,5), en kN, avec
    Mpl = fu,2·t2²/(4·γMu), η1 = ((n1 − 1)·p1 − n1·d0/2)/d2, β1 = p3/d2,
    γ1 = d0/d2 ; d2 : hauteur droite de l'âme. Le 1,5 tient compte de la
    compression du poteau. 0 si la géométrie sort de la formule (β1 ≥ 1).

    Référence : SCI P358 Check 14, p. 29. Ancre : exemple 2, p. 49 (405 kN)."""
    if d2 <= 0 or p3 >= d2 or d0 >= d2:
        return 0.0
    eta1 = ((n1 - 1) * p1 - n1 * d0 / 2.0) / d2
    b1, g1 = p3 / d2, d0 / d2
    Mpl = dv(fu2 * t2 * t2, 4 * gMu)
    return mx(0.0, 8 * Mpl / (1 - b1) * (eta1 + 1.5 * math.sqrt(1 - b1) * math.sqrt(1 - g1))) / 1000


def ame_poteau_plat(fu2, t2, dc, hp, tp, a, gMu):
    """Résistance au chaînage de l'âme du poteau sous un PLAT D'ÂME :
    FRd,u = 8·Mpl/(γMu·(1 − β1))·(η1 + 1,5·(1 − β1)^0,5), en kN, avec
    Mpl = fu,2·t2²/4, η1 = hp/dc, β1 = (tp + 2s)/dc, s = a/0,7 (côté du
    cordon) ; dc : hauteur droite de l'âme.

    Référence : SCI P358 Check 14, p. 133. Ancre : exemple 2 des plats
    d'âme, p. 159 (412 kN) ; cas par défaut du module : 332,13 kN."""
    if dc <= 0:
        return 0.0
    s = a / 0.7
    b1 = (tp + 2 * s) / dc
    if b1 >= 1:
        return 0.0
    eta1 = hp / dc
    Mpl = fu2 * t2 * t2 / 4.0
    return 8 * Mpl / (gMu * (1 - b1)) * (eta1 + 1.5 * math.sqrt(1 - b1)) / 1000


# =========================================================================
#  6. Plat d'âme LONG — déversement du plat (Check 3, p. 119-120)
# =========================================================================
def chi_LT(lam_bar, alpha=ALPHA_LT_PLAT):
    """χLT de l'expression (6.56) de l'EN 1993-1-1 avec αLT = 0,76 — la
    formule qui engendre la Table 5.2 du guide (λ̄ = 1,00 → 0,47)."""
    if lam_bar <= 0:
        return 1.0
    phi = 0.5 * (1 + alpha * (lam_bar - 0.2) + lam_bar * lam_bar)
    rac = phi * phi - lam_bar * lam_bar
    return mn(1.0, dv(1.0, phi + math.sqrt(mx(rac, 0.0))))


def plat_long(tp, hp, z, zp, fy_p, gM0, gM1):
    """Résistance du plat en flexion et au déversement, exprimée en VEd (kN).

    - Wel,p = tp·hp²/6 ; z : bras de levier (nu du porteur → centre du
      groupe) ; zp : nu du porteur → PREMIÈRE file ;
    - plat COURT (zp ≤ tp/0,15) : VRd = Wel,p·fy,p/(z·γM0) ;
    - plat LONG (zp > tp/0,15) : VRd = min(Wel,p·χLT·fy,p/(0,6·z·γM1) ;
      Wel,p·fy,p/(z·γM0)), λ̄LT = 2,8·√(zp·hp/(1,5·tp²))/(π·√(E/fy,p)) — le 0,6
      vient du diagramme triangulaire du moment dans le plat.

    Rend ``{long, Wel, lam, lam_bar, chi, VRd_el, VRd_LT, VRd}``.

    Référence : SCI P358 Check 3, p. 119-120. Ancre : exemple 1 du guide,
    533 UKB (743 kN). Remplace : la courbe BS 5950-1 des profilés laminés,
    sans le 0,6, avec le bras z (``mecanique.pb_bs5950``)."""
    Wel = tp * hp * hp / 6.0
    VRd_el = dv(Wel * fy_p, z * gM0) / 1000 if z > 0 else INF
    long_ = bool(tp > 0 and zp > tp / SEUIL_PLAT_LONG)
    lam = 2.8 * math.sqrt(dv(zp * hp, 1.5 * tp * tp)) if tp > 0 else INF
    lam_bar = dv(lam, math.pi * math.sqrt(dv(E_ACIER, fy_p))) if fy_p > 0 else INF
    chi = chi_LT(lam_bar)
    VRd_LT = dv(Wel * chi * fy_p, 0.6 * z * gM1) / 1000 if z > 0 else INF
    return dict(long=long_, Wel=Wel, lam=lam, lam_bar=lam_bar, chi=chi, VRd_el=VRd_el, VRd_LT=VRd_LT,
                VRd=mn(VRd_LT, VRd_el) if long_ else VRd_el)


# =========================================================================
#  7. Plat d'âme LONG sur une poutre NON grugée — section ABCD de l'âme portée
# =========================================================================
def abcd_sci(V, tw, fy, n1, p1, n2, p2, e2b, zp, hb, gM0):
    """Section ABCD de l'âme portée (AB et CD cisaillées, BC fléchie), SCI :
    VEd·zp (une file) ou VEd·(zp + p2) (deux files)
        ≤ Mc,BC,Rd + Vpl,AB,Rd·(n1 − 1)·p1.

    - Vpl,AB,Rd = tw·e2,b·fy/(√3·γM0) (une file), tw·(e2,b + p2)·… (deux) ;
    - VBC,Ed = VEd·(n1 − 1)·p1/hb ; Vpl,BC,Rd = tw·(n1 − 1)·p1·fy/(√3·γM0) ;
    - Mc,BC,Rd = fy·tw·((n1 − 1)·p1)²/(6·γM0), réduit par
      1 − (2·VBC,Ed/Vpl,BC,Rd − 1)² si VBC,Ed > 0,5·Vpl,BC,Rd.

    À appeler quand zp > tp/0,15. Rend ``{MEd, MRd, Mc_BC, V_AB, VBC,
    Vpl_BC, fort}`` (kNm, kN) ; MRd = 0 s'il n'y a qu'une rangée.

    Référence : SCI P358 Check 4 (suite), p. 123."""
    hBC = (n1 - 1) * p1
    bras = zp if n2 == 1 else zp + p2
    MEd = V * bras / 1000
    if hBC <= 0:
        return dict(MEd=MEd, MRd=0.0, Mc_BC=0.0, V_AB=0.0, VBC=0.0, Vpl_BC=0.0, fort=False, bras=bras)
    V_AB = dv(tw * (e2b if n2 == 1 else e2b + p2) * fy, S3 * gM0) / 1000
    Vpl_BC = dv(tw * hBC * fy, S3 * gM0) / 1000
    VBC = dv(V * hBC, hb)
    Mc = dv(fy * tw * hBC * hBC, 6 * gM0) / 1e6
    fort = bool(VBC > 0.5 * Vpl_BC)
    if fort:
        Mc *= mx(0.0, 1 - (2 * dv(VBC, Vpl_BC) - 1) ** 2)
    return dict(MEd=MEd, MRd=Mc + V_AB * hBC / 1000, Mc_BC=Mc, V_AB=V_AB, VBC=VBC, Vpl_BC=Vpl_BC, fort=fort, bras=bras)


def abcd_msb(V, tw, fy, fu, n1, p1, n2, p2, e2b, zp, d0, VRd_min, gM0, gM2):
    """La même section selon le MSB P5, pour INFORMATION (le MSB la demande
    dès que z > tp/0,15, z au centre du groupe : un plat peut être « long »
    pour lui et « court » pour le SCI).

    - Fpl,AB,Rd = min(e2,b·tw·fy/(√3·γM0) ; (e2,b − d0/2)·tw·fu/(√3·γM2))
      (une file) ; (e2,b + p2) et (e2,b + p2 − 3·d0/2) (deux files) ;
    - Fpl,BC,Rd = min((n1 − 1)·p1·tw·fy/(√3·γM0) ;
      ((n1 − 1)·p1 − (n1 − 1)·d0)·tw·fu/(√3·γM2)) ;
    - VBC,Ed = VEd − (VRd,min − Fpl,BC,Rd) ≥ 0, VRd,min = min(VRd,g ; VRd,n) ;
    - Mc,BC,Rd = fy·tw·((n1 − 1)·p1)²/(6·γM0) si VBC,Ed ≤ 0,5·Fpl,BC,Rd,
      sinon fy·tw·((n1 − 1)·p1)²/(4·γM0)·[1 − (2·VEd/VRd,min − 1)²] ;
    - bras : zp (une file) ; zp + p2 (deux files) — l'équation imprimée porte
      « zp + p2/2 », le texte, le schéma et l'exemple du guide zp + p2.

    Référence : MSB P5 §3.2.5.3, p. 5-29 et 5-30. Ancre : exemple §3.4
    (38 kNm demandés, 66 kNm résistants)."""
    hBC = (n1 - 1) * p1
    bras = zp if n2 == 1 else zp + p2
    MEd = V * bras / 1000
    if hBC <= 0:
        return dict(MEd=MEd, MRd=0.0, bras=bras)
    lAB = e2b if n2 == 1 else e2b + p2
    trous = 0.5 * d0 if n2 == 1 else 1.5 * d0
    F_AB = mn(dv(lAB * tw * fy, S3 * gM0), dv((lAB - trous) * tw * fu, S3 * gM2)) / 1000
    F_BC = mn(dv(hBC * tw * fy, S3 * gM0), dv((hBC - (n1 - 1) * d0) * tw * fu, S3 * gM2)) / 1000
    VBC = mx(0.0, V - (VRd_min - F_BC))
    if VBC <= 0.5 * F_BC:
        Mc = dv(fy * tw * hBC * hBC, 6 * gM0) / 1e6
    else:
        Mc = dv(fy * tw * hBC * hBC, 4 * gM0) / 1e6 * mx(0.0, 1 - (2 * dv(V, VRd_min) - 1) ** 2)
    return dict(MEd=MEd, MRd=Mc + F_AB * hBC / 1000, Mc_BC=Mc, F_AB=F_AB, F_BC=F_BC, VBC=VBC, bras=bras)


# =========================================================================
#  8. Plat d'âme — poinçonnement de l'âme ou de la paroi porteuse (Check 10, p. 128)
# =========================================================================
def poinconnement(t2, fu2, hp, tp, fy_p, gM2):
    """Le plat doit plastifier avant que l'âme porteuse ne poinçonne.

    - forme PRUDENTE : tp ≤ t2·fu,2/(fy,p·γM2) ;
    - forme RIGOUREUSE : tp ≤ t2·fu,2·tp·hp²/(VEd·6·z·γM2), soit
      VEd·z ≤ MRd = t2·fu,2·hp²/(6·γM2).

    Rend ``(tp_max_prudent en mm, MRd en kNm)``.

    Référence : SCI P358 Check 10, p. 128 (un seul côté chargé)."""
    return dv(t2 * fu2, fy_p * gM2), dv(t2 * fu2 * hp * hp, 6 * gM2) / 1e6


# =========================================================================
#  9. Cordons
# =========================================================================
def leff_cordon(L, a, n=1):
    """Longueur efficace de ``n`` cordons de longueur ``L`` : la longueur à
    pleine gorge, L − 2a par cordon (les deux extrémités).

    Référence : EN 1993-1-8 §4.5.1(1)."""
    return n * mx(0.0, L - 2 * a)


def gorge_detail_platine(k_plein, tw):
    """Gorge qu'exige le détail éprouvé d'une PLATINE D'ABOUT : 0,8 fois la
    pleine résistance de l'âme (« the end plate should yield before the weld
    fractures »), soit 0,40·tw en S275 et 0,48·tw en S355 avec les fu du
    site. Le plat d'âme, lui, demande la pleine résistance (annexe C.3).

    Référence : SCI P358 Check 2, p. 16, et annexe C.2, p. 279."""
    return K_CORDON_PLATINE * k_plein * tw


# =========================================================================
#  10. Le statut « à vérifier » : ce que le calcul ne couvre pas
# =========================================================================
DONNEE = "donnée"


def donnee(raison):
    """Marque une raison qui tient aux DONNÉES que le prédimensionnement garde — un effort (traction, sens de VEd,
    HEd), l'appui, la fixation des cornières, la classe des boulons, le maintien de la poutre : aucune proposition ne
    peut la lever. Une raison non marquée tient au DÉTAIL de l'attache (épaisseur, gorge, pinces, plat long…).
    ``raison`` faux (False, None, "") : rendu tel quel."""
    return (raison, DONNEE) if raison else raison


def poser_statut(R, raisons, nom="ASSEMBLAGE"):
    """Écrit le verdict de ``R`` : ``<nom> VÉRIFIÉ``, ``NON VÉRIFIÉ``, ou —
    quand toutes les vérifications passent mais qu'un point HORS du calcul
    reste à justifier — ``<nom> À VÉRIFIER (raisons)``. ``R.a_verifier`` porte
    les raisons (l'écran du site, la note PDF et la liste des propositions le
    lisent : ocre, jamais vert) ; ``R.justifier`` en garde la liste.

    Un point « à justifier » n'est PAS une résistance dépassée : c'est une
    hypothèse du modèle (articulation, maintien, sens de l'effort, mode de
    ruine non calculé) que le calcul ne prouve pas."""
    R.justifier = [r[0] if isinstance(r, tuple) else r for r in raisons if r]
    # les raisons du DÉTAIL seules (non marquées ``donnee``) : une proposition du prédimensionnement qui en porte une
    # n'est pas une solution vérifiée (``propositions._valide``)
    R.justifier_detail = [r for r in raisons if r and not isinstance(r, tuple)]
    R.a_verifier = " ; ".join(R.justifier) if R.verified else ""
    if R.verified and R.a_verifier:
        R.statut = nom + " À VÉRIFIER (" + R.a_verifier + ")"
    else:
        R.statut = nom + (" VÉRIFIÉ" if R.verified else " NON VÉRIFIÉ")
