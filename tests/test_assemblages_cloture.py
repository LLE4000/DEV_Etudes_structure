# -*- coding: utf-8 -*-
"""Lot de clôture des assemblages (27/11/2026) — l'ANCIEN et le NOUVEAU résultat de chaque règle des six attaches
de poutre, sur un cas chiffré à la main ou imprimé dans un guide. Chaque contrôle rougit si on retire la règle.

Le lot reprend les arbitrages A1 à A11 du contrôle indépendant (docs/assemblages/ECARTS_ET_CORRECTIONS.md §8) :

  1. RÈGLES PARTAGÉES (acier/assemblages/sci.py) contre leurs valeurs publiées : groupe de boulons, platine courte,
     âme porteuse, Σleff en chaînage, âme du poteau en chaînage (deux formules), χLT du plat long, section ABCD
     (SCI et MSB), poinçonnement, cordons.
  2. PLATINE D'ABOUT : groupe (ancienne règle à trois branches / règle du SCI), Av = 0,9·hp·tw (89,5 → 99,5 %),
     platine courte (105,0 %), βLf (903,2 → 891,9 kN), cordons hp − 2a, statut.
  3. PLATINE D'ABOUT SUR POTEAU : âme du poteau en chaînage (370 / 362,47 kN = 102,1 %).
  4. PLAT D'ÂME : plat long (seuils du SCI et du MSB, cas N1 à 121,2 %), âme du poteau en chaînage (400 / 332,13 kN
     = 120,4 % ; 300 kN passe), poinçonnement, détail éprouvé (2d, gorge), cordons.
  5. DOUBLES CORNIÈRES : grugeage court à deux files.
  6. STATUT : effort tranchant inversé, traction, poutre non maintenue — jamais « vérifié » en vert.
  7. ÉCRAN ET PROPOSITIONS : une ligne « à vérifier » ne passe pas pour vérifiée.
  8. NOTES : les 216 combinaisons d'options des trois attaches poutre–poteau s'impriment (44 étaient refusées).
  9. AIDE : la formule de la gorge de pleine résistance est la même dans les textes et dans le moteur.

Le pied de poteau a ses propres contrôles : tests/test_pied_poteau_moteur.py §8 et tests/test_platine_manuelle.py.

Lancement : /usr/bin/python3.12 tests/test_assemblages_cloture.py (depuis la racine).
"""
import itertools
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages import materiau as MT  # noqa: E402
from acier.assemblages import mecanique as MEC  # noqa: E402
from acier.assemblages import note_kit as KIT  # noqa: E402
from acier.assemblages import propositions as PR  # noqa: E402
from acier.assemblages import sci as SCI  # noqa: E402
from acier.assemblages.poutre_poutre.platine_about import moteur as PA, benchmark as B_PA, note as N_PA  # noqa: E402
from acier.bibliotheques import PERSO  # noqa: E402
from acier.assemblages.poutre_poutre.plat_ame import moteur as PL, entrees as E_PL, note as N_PL  # noqa: E402
from acier.assemblages.poutre_poutre.plat_ame import methode as ME_PL, ecran_saisie as ES_PL  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres import moteur as DC, note as N_DC  # noqa: E402
from acier.assemblages.poutre_poteau.platine_about_poteau import moteur as PAP, entrees as E_PAP, note as N_PAP  # noqa: E402
from acier.assemblages.poutre_poteau.plat_ame_poteau import moteur as PLP, entrees as E_PLP, note as N_PLP  # noqa: E402
from acier.assemblages.poutre_poteau.plat_ame_poteau import methode as ME_PLP, ecran_saisie as ES_PLP  # noqa: E402
from acier.assemblages.poutre_poteau.doubles_cornieres_poteau import moteur as DCP, entrees as E_DCP, note as N_DCP  # noqa: E402

OK, KO = [], []
S3 = math.sqrt(3)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def taux(R, cle):
    c = R.ck[cle]
    return c.eta if c.active else None


def ids(R):
    return [a.id for a in R.alerts]


# ================================================================
print("=== 1. Les règles partagées contre leurs valeurs publiées (sci.py) ===")
g1 = SCI.groupe_boulons(94.08, [70.0, 90.0], 8)
g2 = SCI.groupe_boulons(94.08, [85.1, 120.0], 8)
chk("groupe de boulons (SCI P358, Check 8) : Fb,min = 70 ≤ 0,8·Fv = 75,3 → n·Fb,min = 560 kN ; Fb,min = 85,1 > 75,3 → "
    "0,8·n·Fv = 602,1 kN (les 602 de l'exemple 1 du SCI)",
    pres(g1[0], 560.0, 1e-9) and pres(g2[0], 0.8 * 8 * 94.08, 1e-9) and pres(g2[0], 602, 1e-3), f"{g1} {g2}")
chk("platine courte (Check 9) : VRd = 2·tp·hp²·fy/(3·(p3 − tw)·γM0) = 2 × 10 × 80² × 355/(3 × 132,5) = 114,3 kN",
    pres(SCI.platine_flexion_plan(10, 80, 355, 140, 7.5, 1.0), 114.31, 1e-4))
chk("âme porteuse (Check 10) : et = min(e1,t ; 5d), eb = min(e1,b ; p3/2 ; 5d) ; un poteau qui continue : et = 5d, "
    "eb = min(p3/2 ; 5d)",
    SCI.longueurs_ame_porteuse(160, 250, 140, 20) == (100, 70) and SCI.longueurs_ame_porteuse(60, 30, 140, 20) == (60, 30)
    and SCI.longueurs_ame_porteuse(None, None, 100, 20, poteau=True) == (100, 50))
libre = 90 - 8.6 - 2 * 5.6 * math.sqrt(2)
chk("Σleff en chaînage (Check 11 ; MSB P5 §2.3.1) : exemple 2 du SCI, 290 mm (ses pinces de 40 mm restent sous le "
    "plafond) ; pinces de 60 et entraxes de 100 : e1A = 0,5·(p3 − tw − 2·a·√2) + d0/2 = 43,8 mm et p1A = 87,6 mm "
    "plafonnent → 262,7 mm au lieu de 320",
    pres(SCI.leff_chainage(4, 70, 40, 40, 90, 8.6, 8 * 0.7, 22), 290, 5e-3)
    and pres(SCI.leff_chainage(3, 100, 60, 60, 90, 8.6, 5.6, 22), 2 * (0.5 * libre + 11) + 2 * (libre + 22), 1e-9)
    and pres(SCI.leff_chainage(3, 100, 60, 60, 90, 8.6, 5.6, 22), 262.7, 1e-3),
    str(SCI.leff_chainage(3, 100, 60, 60, 90, 8.6, 5.6, 22)))
chk("âme du poteau en chaînage sous une platine d'about (Check 14, p. 29) : cas du dossier, 362,47 kN "
    "(η1 = 0,7981, β1 = 100/208, γ1 = 22/208)",
    pres(SCI.ame_poteau_platine(470, 11, 208, 4, 70, 100, 22, 1.1), 362.47, 1e-4))
chk("âme du poteau en chaînage sous un plat d'âme (Check 14, p. 133) : cas du dossier, 332,13 kN "
    "(β1 = (10 + 2 × 8,571)/208 = 0,13049, η1 = 290/208)",
    pres(SCI.ame_poteau_plat(470, 11, 208, 290, 10, 6, 1.1), 332.13, 1e-4))
chk("âme du poteau en chaînage — les exemples 2 PUBLIÉS du SCI P358 (305 × 305 × 137 UKC, fu = 410, tw = 13,8, "
    "d = 246,7) : platine d'about p. 49, 4 rangées à 70, p3 = 90, d0 = 22 → 405 kN ; plat d'âme p. 159, hp = 290, "
    "tp = 10, cordon de 8 mm de côté (a = 5,6) → 412 kN",
    pres(SCI.ame_poteau_platine(410, 13.8, 246.7, 4, 70, 90, 22, 1.1), 405, 2e-3)
    and pres(SCI.ame_poteau_plat(410, 13.8, 246.7, 290, 10, 5.6, 1.1), 412, 2e-3),
    f"{SCI.ame_poteau_platine(410, 13.8, 246.7, 4, 70, 90, 22, 1.1):.2f} / {SCI.ame_poteau_plat(410, 13.8, 246.7, 290, 10, 5.6, 1.1):.2f}")
chk("χLT du plat long (EN 1993-1-1 §6.3.2.2, αLT = 0,76) : le Tableau 5.2 du SCI (p. 120) — 0,25 → 0,96 ; 0,50 → "
    "0,78 ; 0,75 → 0,61 ; 1,00 → 0,47 ; 1,25 → 0,36, retrouvés à moins de 1,5 % ; sous 0,2 : 1",
    all(pres(SCI.chi_LT(l), x, 0.015) for l, x in ((0.25, 0.96), (0.50, 0.78), (0.75, 0.61), (1.00, 0.47), (1.25, 0.36)))
    and SCI.chi_LT(0.2) == 1.0, f"{[round(SCI.chi_LT(l), 4) for l in (0.25, 0.5, 0.75, 1.0, 1.25)]}")
ab = SCI.abcd_sci(250, 8.5, 355, 3, 70, 2, 60, 40, 80, 300, 1.0)
chk("section ABCD d'un plat long (Check 4, p. 123) — cas repère du contrôle indépendant : VBC = 116,67 kN ≤ 0,5 × 243,90, "
    "Mc,BC = 9,857 kNm, Vpl,AB = 174,22 kN, disponible 34,25 kNm pour 35,00 demandés : 102,2 %",
    pres(ab["VBC"], 116.67, 1e-4) and pres(ab["Mc_BC"], 9.857, 1e-4) and pres(ab["V_AB"], 174.22, 1e-4)
    and pres(ab["MRd"], 34.25, 1e-4) and pres(ab["MEd"], 35.0, 1e-9) and pres(ab["MEd"] / ab["MRd"], 1.022, 1e-3), str(ab))
po = SCI.poinconnement(12.5, 470, 300, 10, 275, 1.25)
chk("poinçonnement de l'âme porteuse (Check 10) : borne prudente tp ≤ t2·fu,2/(fy,p·γM2) = 12,5 × 470/(275 × 1,25) = "
    "17,1 mm ; forme rigoureuse VEd·z ≤ MRd = t2·fu,2·hp²/(6·γM2) = 70,5 kNm (tp s'élimine)",
    pres(po[0], 12.5 * 470 / (275 * 1.25), 1e-9) and pres(po[1], 12.5 * 470 * 300 ** 2 / (6 * 1.25) / 1e6, 1e-9)
    and pres(po[1], 70.5, 1e-9), str(po))
chk("cordons : longueur efficace L − 2a par cordon (EN 1993-1-8 §4.5.1(1)) — 290 mm, a = 6 → 278 ; deux cordons → 556",
    SCI.leff_cordon(290, 6) == 278 and SCI.leff_cordon(290, 6, 2) == 556)

# ================================================================
print("\n=== 2. Platine d'about ===")
bm1 = [b for b in B_PA.BENCH if b["id"] == "BM1"][0]            # exemple du MSB P5 §2.4 (matériau du guide)
Rb = PA.compute(bm1["inp"], MT.EXEMPLES)
anc = PA.groupe_boulons(Rb.Fv_Rd, Rb.Fbb_L, Rb.Fbi_L, 4, Rb.n_b - 4)
chk("groupe de boulons — exemple du MSB : la règle du SCI rend 0,8·n·Fv = 903,2 kN ; l'ancienne règle à trois branches, "
    "plafonnée, rendait la même valeur ICI ; elle additionnait les pressions diamétrales ailleurs (cas ci-dessous)",
    pres(Rb.FRd_g, 903.2, 1e-4) and pres(anc[0], Rb.FRd_g, 1e-9), f"{Rb.FRd_g:.1f} / {anc[0]:.1f}")
ga = PA.groupe_boulons(94.08, 70.0, 90.0, 4, 4)
gn = SCI.groupe_boulons(94.08, [70.0, 90.0], 8)
chk("groupe de boulons — pressions diamétrales 70 (extrémité) et 90 (intérieur) sous Fv = 94,08 kN, 8 boulons : "
    "l'ancienne règle additionnait (640 kN, plafonnés à 602,1) ; le SCI prend n·Fb,min = 560 kN — 7 % de moins",
    pres(ga[0], 0.8 * 8 * 94.08, 1e-9) and pres(gn[0], 560.0, 1e-9) and gn[0] < ga[0], f"{ga} → {gn}")
chk("âme portée au droit de la platine — exemple du MSB, VEd = 550 kN : Av = 0,9·hp·tw → 99,5 % (avant : hp·tw, 89,5 %)",
    pres(Rb.Av_w, 0.9 * Rb.At_w, 1e-12) and pres(taux(Rb, "vwS"), 0.995, 1e-3)
    and pres(550 / (Rb.At_w * Rb.fy_S / S3 / 1000), 0.895, 1e-3), f"{taux(Rb, 'vwS'):.4f}")
Rbl = PA.compute(dict(bm1["inp"], opt_blf="Oui"), MT.EXEMPLES)
chk("βLf — six rangées à 70 mm (Lj = 350 > 15·d = 300) : « Non » par défaut (903,2 kN, comme l'exemple publié) et un "
    "avis « blf_off » ; « Oui » : βLf = 0,9875 → 891,9 kN",
    "blf_off" in ids(Rb) and pres(Rbl.bLf, 0.9875, 1e-9) and pres(Rbl.FRd_g, 891.9, 1e-4) and "blf_off" not in ids(Rbl),
    f"{Rb.FRd_g:.1f} → {Rbl.FRd_g:.1f}")
chk("βLf — l'avis écrit la condition de l'exception : âme soudée sur toute la hauteur, transfert uniforme, §3.8(2) "
    "(A10 : « documenter les conditions de l'exception »)",
    any(a.id == "blf_off" and "l'exception du §3.8(2)" in a.msg for a in Rb.alerts), str([a.msg for a in Rb.alerts if a.id == "blf_off"]))
COURTE = dict(prof_S="IPE 330", nu_S="S355", prof_P="HEB 500", nu_P="S355", d_top=60, d_nt=0, d_nb=0, l_n=0, hp_u=80,
              bp_u=220, tp_u=10, nu_pl="S355", n1_u=1, e1_u=40, p3_u=140, a_w=5, V_Ed=120, boulon_u="M20", classe="8.8")
Rc = PA.compute(COURTE)
chk("platine courte (une rangée, hp = 80 < 1,36 × 140) sous 120 kN : flexion dans son plan 120 / 114,3 = 105,0 % — "
    "vérification absente avant le lot ; inactive dès que hp ≥ 1,36·p3",
    not Rc.hp_136 and pres(Rc.VRd_ip, 114.31, 1e-4) and taux(Rc, "pip") is not None
    and pres(taux(Rc, "pip") or 0, 1.050, 1e-3) and not Rc.verified
    and taux(PA.compute(dict(COURTE, hp_u=200, n1_u=2)), "pip") is None, f"{taux(Rc, 'pip')}")
Rd = PA.compute()
chk("cordons de la platine : Lw = 2·(hp − 2a) (avant : 2·hp) ; âme porteuse sur et + (n1 − 1)·p1 + eb (avant : hp)",
    pres(Rd.Lw, 2 * (Rd.h_p - 2 * Rd.a_w), 1e-12) and pres(Rd.Lw1, Rd.h_p - 2 * Rd.a_w, 1e-12)
    and pres(Rd.Av_P, Rd.tw_P * (Rd.et_P + (Rd.n_1 - 1) * Rd.p_1 + Rd.eb_P), 1e-12))
Rg = PA.compute(dict(a_w=3))
Rt = PA.compute(dict(tp_u=20))
chk("articulation — gorge de 3 mm sous le détail éprouvé (0,8 × pleine résistance = 4,1 mm) : ASSEMBLAGE À VÉRIFIER, "
    "raison nommée ; platine de 20 mm au-delà de la limite de ductilité (11,0 mm) : de même ; le cas par défaut : VÉRIFIÉ",
    Rg.verified and Rg.a_verifier == "articulation à justifier : gorge de 3 mm < 4,1 mm (détail éprouvé)"
    and Rt.verified and Rt.a_verifier == "articulation à justifier : platine de 20 mm > 11,0 mm"
    and Rd.statut == "ASSEMBLAGE VÉRIFIÉ" and pres(Rd.a_detail, 0.8 * Rd.a_plein, 1e-9), f"{Rg.statut} | {Rt.statut}")

# ================================================================
print("\n=== 3. Platine d'about sur poteau : l'âme du poteau en chaînage ===")
AME_PAP = dict(appui=E_PAP.APPUI_AME, nat_N=E_PAP.NAT_TYING, p3_u=100, bp_u=180)     # platine 290 × 180 × 10, 4 rangées
R370, R300 = PAP.compute(dict(AME_PAP, N_Ed=370)), PAP.compute(dict(AME_PAP, N_Ed=300))
chk("chaînage de 370 kN, appui dans l'âme du HEB 300 : FRd,u = 362,47 kN → 102,1 %, c'est la ligne qui gouverne — NON "
    "VÉRIFIÉ (avant le lot : « ASSEMBLAGE VÉRIFIÉ » à 97,6 %, avec un simple avis) ; à 300 kN : 82,8 %, VÉRIFIÉ",
    pres(R370.FRd_ame, 362.47, 1e-4) and pres(taux(R370, "ahP"), 1.021, 1e-3) and not R370.verified and R370.gov.key == "ahP"
    and not [a for a in R370.alerts if a.block] and pres(taux(R300, "ahP"), 0.828, 1e-3) and R300.verified and not R300.a_verifier,
    f"{R370.statut} {taux(R370, 'ahP')} | {R300.statut} {taux(R300, 'ahP')}")
# l'exemple 2 du SCI P358 (p. 44-49) dans le moteur : chaînage de 175 kN, platine 290 × 150 × 10 sur l'âme d'un
# 305 × 305 × 137 UKC, matériau et coefficients du guide (S275 : fu = 410 ; γMu = 1,1)
EX2 = dict(appui=E_PAP.APPUI_AME, prof_P=PERSO, hP_u=320.5, bP_u=309.2, twP_u=13.8, tfP_u=21.7, rP_u=15.2, nu_P="S275",
           prof_S=PERSO, hS_u=412.8, bS_u=179.5, twS_u=9.5, tfS_u=16.0, rS_u=10.2, nu_S="S275", nu_pl="S275",
           boulon_u="M20", classe="8.8", n1_u=4, p1_u=70, e1_u=40, p3_u=90, bp_u=150, hp_u=290, tp_u=10, a_w=4.24,
           d_nt=0, d_nb=0, l_n=0, z_C=50, V_Ed=0, N_Ed=175, nat_N=E_PAP.NAT_TYING, g_Mu=1.1, g_M2n=1.1, g_M2=1.25)
Rx2 = PAP.compute(EX2, lambda nom, t: {"fy": 275.0 if t <= 16 else 265.0, "fu": 410.0, "bw": 0.85})
Ftu = 0.9 * Rx2.f_ub * Rx2.As_b / 1.1 / 1000
FT_guide = MEC.troncon_T(Rx2.leff_T, Rx2.leff2_T, Rx2.t_p, Rx2.m_T, Rx2.nT, 33 / 4, Rx2.fu_L, 1.1, Rx2.n_b * Ftu)["FT_Rd"]
chk("exemple 2 du SCI dans le moteur : âme du poteau FRd,u = 405,4 kN (405 publiés, p. 49) ; tronçon en T, Σleff = "
    "290 mm, 380,5 kN avec ew = 33/4 = 8,25 mm (les hypothèses du guide, sans rondelle : 381 publiés, p. 47-48) et "
    "393,3 kN avec la rondelle du site (dw = 37 → ew = 9,25) : l'écart de 3,2 % est la rondelle, pas la formule",
    pres(Rx2.ck["ahP"].Rd, 405, 2e-3) and pres(Rx2.leff_T, 290, 1e-9) and pres(FT_guide, 381, 3e-3)
    and pres(Rx2.ew_T, 9.25, 1e-9) and pres(Rx2.ck["ptT"].Rd, 393.3, 1e-3),
    f"{Rx2.ck['ahP'].Rd:.2f} | {FT_guide:.2f} | {Rx2.ck['ptT'].Rd:.2f}")
Rel_t = PAP.compute(dict(N_Ed=60, nat_N=E_PAP.NAT_ELU))
m_, e_, n1_, p_ = Rel_t.m_T, Rel_t.e_2, Rel_t.n_1, Rel_t.p_1
t66 = min(n1_ * 2 * math.pi * m_, n1_ * (4 * m_ + 1.25 * e_), 2 * (math.pi * m_ + p_) + (n1_ - 2) * 2 * p_,
          2 * (2 * m_ + 0.625 * e_ + 0.5 * p_) + (n1_ - 2) * p_)
chk("tronçon en T à l'ELU (A2) : Σleff du Tableau 6.4 avec les bords libres de la platine (e1) = 290 mm = hp, la valeur "
    "des guides ; les motifs du Tableau 6.6 (rangées intérieures et d'extrémité, sans terme de bord) donneraient "
    "500 mm, plus que la platine — l'adaptation est la lecture physique, du côté sûr",
    pres(Rel_t.leff_T, 290, 1e-9) and pres(Rel_t.leff_T, Rel_t.h_p, 1e-9) and t66 > Rel_t.h_p + 100, f"{Rel_t.leff_T} / {t66:.1f}")
Rel = PAP.compute(dict(AME_PAP, N_Ed=60, nat_N=E_PAP.NAT_ELU))
chk("traction ELU, appui dans l'âme : la formule du chaînage ne s'applique pas — pas de ligne « ahP » — et le statut "
    "est « ASSEMBLAGE À VÉRIFIER (âme du poteau sous la traction ELU : flexion hors plan non vérifiée) »",
    taux(Rel, "ahP") is None and Rel.verified
    and Rel.a_verifier == "âme du poteau sous la traction ELU : flexion hors plan non vérifiée", Rel.statut)

# ================================================================
print("\n=== 4. Plat d'âme ===")
N1 = dict(prof_S="IPE 300", hp_u=220, bp_u=150, tp_u=10, n1_u=3, n2_u="2", boulon_u="M20", V_Ed=250, e1_u=40, p1_u=70,
          p2_u=60, e2b_u=40, g_h=10)
Rn7, Rn6 = PLP.compute(dict(N1, a_w=7)), PLP.compute(dict(N1, a_w=6))
chk("plat long — cas N1 (plat 220 × 150 × 10 sur l'aile, 3 × 2 M20, 250 kN) : zp = 50 ≤ 66,7 < z = 80 mm → court pour le "
    "SCI, long pour le MSB ; section ABCD du MSB 27,50 / 22,69 kNm = 121,2 %",
    Rn7.g_B == 50 and Rn7.z_p == 80 and not Rn7.long_p and Rn7.long_msb and pres(Rn7.abcd_msb.MEd, 27.50, 1e-9)
    and pres(Rn7.abcd_msb.MRd, 22.69, 5e-4) and pres(Rn7.eta_abcd_msb, 1.212, 1e-3), f"{Rn7.abcd_msb}")
chk("… avant le lot : « ASSEMBLAGE VÉRIFIÉ » à 99,6 %. Maintenant, gorge de 7 mm : ASSEMBLAGE À VÉRIFIER (section ABCD "
    "… selon le MSB P5) ; gorge de 6 mm : les cordons (hp − 2a) sont à 104,4 % — NON VÉRIFIÉ",
    Rn7.verified and "MSB P5" in Rn7.a_verifier and Rn7.statut.startswith("ASSEMBLAGE À VÉRIFIER")
    and pres(Rn7.eta_max, 0.996, 1e-3) and not Rn6.verified and pres(taux(Rn6, "w"), 1.044, 1e-3),
    f"{Rn7.statut} | {taux(Rn6, 'w')}")
AME_PLP = dict(appui=E_PLP.APPUI_AME, nat_N=E_PLP.NAT_TYING)
R400, R300p = PLP.compute(dict(AME_PLP, N_Ed=400)), PLP.compute(dict(AME_PLP, N_Ed=300))
chk("âme du poteau en chaînage sous le plat d'âme par défaut : 400 / 332,13 kN = 120,4 %, NON VÉRIFIÉ (avant le lot : "
    "« ASSEMBLAGE VÉRIFIÉ » à 93,5 %) ; à 300 kN : 90,3 %, la ligne passe",
    pres(R400.FRd_ame, 332.13, 1e-4) and "ahP" in R400.ck and pres(taux(R400, "ahP"), 1.204, 1e-3) and not R400.verified
    and R400.gov.key == "ahP" and pres(taux(R300p, "ahP"), 0.903, 1e-3) and R300p.verified,
    f"{taux(R400, 'ahP') if 'ahP' in R400.ck else None} {R400.statut} | {R300p.statut}")
PL3 = dict(prof_S="HEB 200", hS_u=600, bS_u=100, twS_u=5, tfS_u=8, rS_u=0, prof_P="IPE 500", d_top=0, d_nt=50.0, d_nb=40.0,
           l_n=150.0, boulon_u="M12", classe="8.8", filet="Oui", cat="C", opt_blf="Oui", expo="Oui", lt_ok="Oui", nu_S="S355",
           nu_P="S460 N/NL", V_Ed=200, N_Ed=30, n1_u=2, n2_u="2", p1_u=50, e1_u=25, e2b_u=40, p2_u=70, bp_u=220, tp_u=20,
           a_w=6, g_h=15, nu_pl="S460 N/NL", M_Ed=15, z_C=50, hp_u=100.0, trou="Surdimensionné", d0_u=15)
PL10 = dict(prof_P="IPE 400", d_nt=40, d_nb=50, l_n=100, e1_u=35, p1_u=60)
R3, R10 = PL.compute(PL3), PL.compute(PL10)
chk("poinçonnement de l'âme porteuse — fiche PL-3 du dossier (plat de 20 mm sur une âme d'IPE 500) : la forme "
    "rigoureuse est dépassée (plus de 400 %), ligne absente avant le lot ; fiche PL-10 : le plat dépasse la borne "
    "prudente (tp > tp,max) mais la forme rigoureuse passe (32,1 %)",
    taux(R3, "poi") > 4.0 and not R3.verified and R10.t_p > R10.tp_poi and pres(taux(R10, "poi"), 0.321, 2e-3),
    f"{taux(R3, 'poi')} | {R10.t_p} > {R10.tp_poi:.2f}, {taux(R10, 'poi')}")
d0 = E_PL.defaults()
Rdef, R35 = PL.compute(), PL.compute(dict(e1_u=35, p1_u=60))
chk("détail éprouvé — le cas par défaut (M20) tient les pinces de 2d : e1 = 40, p1 = 55, statut sans raison ; l'ancien "
    "défaut (e1 = 35 < 40) : ASSEMBLAGE À VÉRIFIER (articulation à justifier : e1 du plat = 35 mm < 2d = 40 mm), avis "
    "« pince_2d » — la saisie reste admise (Tableau 3.3)",
    d0["e1_u"] == 40 and d0["p1_u"] == 55 and Rdef.rec_2d and Rdef.verified and not Rdef.a_verifier
    and R35.verified and not R35.rec_2d and "e1 du plat = 35 mm < 2d = 40 mm" in R35.a_verifier and "pince_2d" in ids(R35)
    and R35.dist_ok, f"{Rdef.statut} | {R35.statut}")
Rgo = PL.compute(dict(a_w=4))
chk("gorge sous la pleine résistance (a = 4 mm < 6,0) : ASSEMBLAGE À VÉRIFIER, raison nommée ; boulons 10.9 ou "
    "précontraints : hors du détail éprouvé",
    Rgo.verified and Rgo.a_verifier == "articulation à justifier : gorge de 4 mm < 6,0 mm (pleine résistance)" and not Rgo.rec_a
    and "boulons hors du détail éprouvé" in " ".join(PL.compute(dict(classe="10.9")).justifier))
chk("cordons du plat : longueur efficace hp − 2a par cordon (avant : hp) ; qz = VEd/(2·Lw)",
    pres(Rdef.Lw1, Rdef.h_p - 2 * Rdef.a_w, 1e-12))
lam = 2.8 / (math.pi * math.sqrt(210000 / 355.0)) * math.sqrt(120 * 290 / (1.5 * 10 ** 2))
pl = SCI.plat_long(10, 290, 150, 120, 355, 1.0, 1.1)
chk("déversement du plat long (Check 3) : λ̄LT sur zp, χLT de l'EN 1993-1-1 ; l'ancienne résistance prenait la courbe "
    "BS 5950-1 (pb_bs5950), gardée pour ce test — les deux diffèrent",
    pl["long"] and pres(pl["lam_bar"], lam, 1e-9) and pres(pl["chi"], SCI.chi_LT(lam), 1e-12)
    and pres(pl["VRd"], min(pl["VRd_LT"], pl["VRd_el"]), 1e-12)
    and abs(pl["chi"] * 355 - PL.pb_bs5950(pl["lam"], 355)) > 1.0, f"{pl['chi'] * 355:.1f} / {PL.pb_bs5950(pl['lam'], 355):.1f}")

Rw1, Rw2 = PL.compute(dict(n1_u=1, hp_u=70, e1_u=35, a_w=10)), PL.compute(dict(n1_u=1, hp_u=80, e1_u=40, a_w=10))
chk("longueur minimale d'un cordon (EN 1993-1-8 §4.5.1(2)) : a = 10 → max(30 ; 6a) = 60 mm ; hp = 70 → hp − 2a = 50 mm "
    "refusé (« w_court », bloquante) ; hp = 80 → 60 mm admis",
    pres(Rw1.Lw1, 50, 1e-9) and "w_court" in [a.id for a in Rw1.alerts if a.block]
    and pres(Rw2.Lw1, 60, 1e-9) and "w_court" not in ids(Rw2), f"{Rw1.Lw1} {Rw2.Lw1}")
# ================================================================
print("\n=== 5. Doubles cornières : grugeage court à deux files ===")
DC2 = dict(n2S_u=2, p2_S=60, e2b_u=40, d_nt=50)
Rc135, Rc150 = DC.compute(dict(DC2, l_n=135)), DC.compute(dict(DC2, l_n=150))
chk("xN = ln − (e2,b + p2) = 35 mm < 2d = 40 : bras max(gh + ln ; gh + e2,b + p2) et résistance de la section au droit "
    "de la 2ᵉ file (MSB P5 §4.2.4.2) — plus faible que la résistance au bout du grugeage ; xN = 50 mm : règle inactive",
    Rc135.court_N and pres(Rc135.xN, 35, 1e-12) and pres(Rc135.bras_N, max(Rc135.u.g_h + 135, Rc135.u.g_h + 100), 1e-12)
    and pres(Rc135.ck["mN"].Rd, Rc135.Mc_2, 1e-9) and Rc135.Mc_2 <= Rc135.Mv_N + 1e-9
    and not Rc150.court_N and pres(Rc150.ck["mN"].Rd, Rc150.Mv_N, 1e-9), f"{Rc135.ck['mN'].Rd:.2f} / {Rc135.Mv_N:.2f}")
Rp135 = DCP.compute(dict(DC2, l_n=135))
chk("la même règle sur le poteau (doubles cornières sur poteau)",
    Rp135.court_N and pres(Rp135.ck["mN"].Rd, Rp135.Mc_2, 1e-9))

# ================================================================
print("\n=== 6. Statut : ce que le calcul ne couvre pas n'est jamais « vérifié » en vert ===")
for nom, M, base in (("platine d'about", PA, dict()), ("plat d'âme", PL, dict()), ("doubles cornières", DC, dict(d_nt=0, l_n=0)),
                     ("platine d'about sur poteau", PAP, dict()), ("plat d'âme sur poteau", PLP, dict()),
                     ("doubles cornières sur poteau", DCP, dict())):
    V0 = M.compute(base).u.V_Ed
    cas = [("sans grugeage", dict(base)), ("grugeage haut", dict(base, d_nt=50, l_n=150)),
           ("grugeage bas", dict(base, d_nb=50, l_n=150)), ("double grugeage", dict(base, d_nt=50, d_nb=50, l_n=150))]
    bons, verifies = [], 0
    for lib, u in cas:
        Rp, Rm = M.compute(dict(u, V_Ed=V0)), M.compute(dict(u, V_Ed=-V0))
        memes = all(pres(a.eta, b.eta, 1e-9) for a, b in zip(Rp.checks, Rm.checks) if a.active and a.eta is not None)
        verifies += bool(Rp.verified)
        bons.append(Rm.V_inverse and not Rp.V_inverse and "V_inv" in ids(Rm) and "V_inv" not in ids(Rp) and memes
                    and Rm.verified == Rp.verified
                    and (not Rp.verified or (Rm.statut.startswith("ASSEMBLAGE À VÉRIFIER")
                                             and "effort tranchant de sens inversé" in Rm.a_verifier))
                    and "effort tranchant de sens inversé" in Rm.justifier
                    and "effort tranchant de sens inversé" not in Rp.justifier)
    chk(f"{nom} — VEd < 0 (sans grugeage, haut, bas, double) : mêmes taux qu'en positif, avis « V_inv » ; un cas "
        f"vérifié en positif devient « ASSEMBLAGE À VÉRIFIER (… effort tranchant de sens inversé) » ({verifies} cas "
        "vérifiés sur 4)", all(bons) and verifies >= 1, str(bons))
Rnm = PL.compute(dict(d_nt=50, l_n=150, lt_ok="Non"))
chk("poutre grugée NON maintenue au déversement : « ASSEMBLAGE À VÉRIFIER (poutre grugée non maintenue au "
    "déversement) » ; maintenue : « ASSEMBLAGE VÉRIFIÉ »",
    Rnm.verified and Rnm.a_verifier == "poutre grugée non maintenue au déversement"
    and PL.compute(dict(d_nt=50, l_n=150)).statut == "ASSEMBLAGE VÉRIFIÉ", Rnm.statut)
Rtr = PA.compute(dict(N_Ed=40))
chk("traction sur une attache poutre–poutre : « ASSEMBLAGE À VÉRIFIER (âme porteuse sous la traction non vérifiée) »",
    Rtr.verified and Rtr.a_verifier == "âme porteuse sous la traction non vérifiée", Rtr.statut)
Rso = DC.compute(dict(fix_P="Soudée", fix_S="Soudée"))
chk("doubles cornières soudées des deux côtés : « ASSEMBLAGE À VÉRIFIER (articulation à justifier : cornières soudées "
    "des deux côtés) » ; le cas par défaut : « ASSEMBLAGE VÉRIFIÉ »",
    Rso.verified and Rso.a_verifier == "articulation à justifier : cornières soudées des deux côtés"
    and DC.compute().statut == "ASSEMBLAGE VÉRIFIÉ", Rso.statut)
for R in (R35, Rn7, Rnm, Rso):
    if R.verified and R.a_verifier:
        chk("un statut « À VÉRIFIER » : vérifié reste vrai pour le moteur, la raison est écrite une fois, le statut la porte "
            f"({R.statut[:60]}…)", R.statut == "ASSEMBLAGE À VÉRIFIER (" + R.a_verifier + ")"
            and R.a_verifier == " ; ".join(R.justifier))

# ================================================================
print("\n=== 7. Écran et propositions : une ligne « à vérifier » ne passe pas pour vérifiée ===")
from api_web import assemblages as A  # noqa: E402

lv, lo, lr = A._ligne_sel("a", "1", Rdef, False, "x"), A._ligne_sel("b", "2", R35, False, "x"), A._ligne_sel("c", "3", R400, False, "x")
chk("sélecteur : vérifié → vert, « _ok » ; à vérifier → orange (niveau « v »), « _ok » FAUX, « _ok2 » vrai, le statut "
    "est écrit sur la ligne ; non vérifié → rouge",
    lv["_ok"] and lv["niveau"] != "v" and not lo["_ok"] and lo["_ok2"] and lo["niveau"] == "v"
    and "à vérifier" in lo["prop"] and not lr["_ok"] and not lr["_ok2"] and lr["niveau"] == "r", f"{lv} {lo}")
chk("l'étoile « le plus petit qui tient » va à la première ligne vérifiée SANS raison ; à défaut seulement, à la "
    "première ligne « à vérifier »",
    [r["opt"] for r in A._marquer_optimal([dict(lo, opt=False), dict(lv, opt=False)])] == [False, True]
    and [r["opt"] for r in A._marquer_optimal([dict(lr, opt=False), dict(lo, opt=False)])] == [False, True])
cfg = A.CONFIGS["plat_ame"]
X = A.chercher_solutions(cfg, A.valeurs_normales(cfg, dict(E_PL.defaults(), V_Ed=110)))
chk("propositions du plat d'âme (110 kN) : chacune porte « a_verifier » ; les solutions sans raison passent devant ; la "
    "conseillée tient le détail éprouvé (pinces de 2d, gorge de pleine résistance)",
    len(X.solutions) >= 1 and all(s.a_verifier == (s.R.get("a_verifier") or "") for s in X.solutions)
    and [bool(s.a_verifier) for s in X.solutions if s.cible] == sorted(bool(s.a_verifier) for s in X.solutions if s.cible)
    and not X.solutions[0].a_verifier, str([(s.boulon, s.a_verifier) for s in X.solutions]))
# Le contrôle ci-dessus est devenu VIDE quand le filtre du détail (_valide) est arrivé : plus aucune solution proposée
# n'y porte de raison, et la passe de mutations du lot l'a montré (le rang retiré, le test restait vert). Le rang est
# la seconde ligne de défense : on lève le filtre pour le voir travailler.
_valide_rang = PR._valide
PR._valide = lambda R: bool(R is not None and R.verified)
try:
    X_ord = A.chercher_solutions(cfg, A.valeurs_normales(cfg, dict(E_PL.defaults(), V_Ed=110)))
finally:
    PR._valide = _valide_rang
chk("le RANG des propositions (seconde ligne de défense) : filtre du détail levé, la solution « à vérifier » (M30, section "
    "ABCD à 105 % selon le MSB) passe derrière toutes les solutions sans raison, même le M12 à réserve",
    any(s.a_verifier for s in X_ord.solutions if s.cible)
    and [bool(s.a_verifier) for s in X_ord.solutions if s.cible] == sorted(bool(s.a_verifier) for s in X_ord.solutions
                                                                           if s.cible),
    str([(s.boulon, s.a_verifier[:30]) for s in X_ord.solutions]))

cfg_pap = A.CONFIGS["platine_about_poteau"]
u_pap = A.valeurs_normales(cfg_pap, A.nouvel_assemblage("platine_about_poteau")["values"])
X_pap = A.chercher_solutions(cfg_pap, u_pap)
_valide = PR._valide
PR._valide = lambda R: bool(R is not None and R.verified)          # mutation : le détail « à justifier » accepté
try:
    X_mut = A.chercher_solutions(cfg_pap, u_pap)
finally:
    PR._valide = _valide
chk("une solution dont le DÉTAIL est à justifier n'est pas proposée (platine d'about sur poteau, défaut) : aucune n'a de "
    "raison de détail, le M12 en est absent (platine de 8 mm > 6,5 mm) ; MUTATION — accepter le détail : le M12 revient, "
    "« À VÉRIFIER (articulation à justifier : platine de 8 mm > 6,5 mm) »",
    X_pap.solutions and not any(s.R.get("justifier_detail") for s in X_pap.solutions)
    and "M12" not in [s.boulon for s in X_pap.solutions]
    and any(s.boulon == "M12" and "platine de 8 mm > 6,5 mm" in s.a_verifier for s in X_mut.solutions),
    str([(s.boulon, s.a_verifier) for s in X_mut.solutions]))
Rtr_pa = PA.compute(dict(N_Ed=40))
chk("une raison qui tient aux DONNÉES (une traction gardée) est marquée telle : elle n'est pas « du détail », la solution "
    "reste proposée et dit sa raison ; une raison du détail (gorge) l'est",
    Rtr_pa.justifier == ["âme porteuse sous la traction non vérifiée"] and Rtr_pa.justifier_detail == []
    and Rg.justifier_detail == ["articulation à justifier : gorge de 3 mm < 4,1 mm (détail éprouvé)"])

# ================================================================
print("\n=== 8. Notes : les hypothèses ne refusent plus la note (défaut D1) ===")
Rd1 = PLP.compute(dict(d_nt=40, l_n=150, N_Ed=250, nat_N=E_PLP.NAT_TYING))
try:
    KIT.empaqueter(N_PLP.segments_hypotheses(Rd1), n_max=3, larg=110, une_fois=KIT.MSB)
    refus = ""
except ValueError as e:
    refus = str(e)
l4 = N_PLP.lignes_hypotheses(Rd1)
chk("reproduction la plus simple (plat d'âme sur poteau, grugeage 40 × 150, chaînage de 250 kN) : l'ancien "
    "empaqueteur refusait (« ne tiennent pas en 3 lignes ») ; la note prend une quatrième ligne",
    "3 lignes" in refus and len(l4) == 4 and all(len(x) <= 110 for x in l4), f"{refus} | {[len(x) for x in l4]}")
bilan = {}
for nom, M, NO, E in (("plat d'âme sur poteau", PLP, N_PLP, E_PLP), ("platine d'about sur poteau", PAP, N_PAP, E_PAP),
                      ("doubles cornières sur poteau", DCP, N_DCP, E_DCP)):
    grug = [dict(), dict(d_nt=40, l_n=150), dict(d_nt=40, d_nb=40, l_n=150)]
    efforts = [dict(), dict(N_Ed=60), dict(N_Ed=250, nat_N=E.NAT_TYING)]
    appuis = [dict(appui=E.APPUI_AILE), dict(appui=E.APPUI_AME)]
    cats = [dict(cat=c) for c in ("A", "B", "C")]
    fix = [dict(), dict(fix_S=E.SOUDEE)] if M is DCP else [dict()]
    n = avant = refusees = longues = 0
    for g, ef, ap, c, f in itertools.product(grug, efforts, appuis, cats, fix):
        R = M.compute(dict(g, **ef, **ap, **c, **f))
        n += 1
        try:
            KIT.empaqueter(NO.segments_hypotheses(R), n_max=3, larg=110, une_fois=KIT.MSB)
        except ValueError:
            avant += 1
        try:
            ls = NO.lignes_hypotheses(R)
            longues += any(len(x) > 110 for x in ls) or len(ls) > 4
        except ValueError:
            refusees += 1
    bilan[nom] = (n, avant, refusees, longues)
chk("216 combinaisons d'options (grugeage × effort normal × appui × catégorie × fixation) : la révision en ligne refusait "
    "44 notes (28 + 10 + 6, mesuré sur d207162) ; l'ancien empaqueteur, sur les hypothèses d'aujourd'hui (L19), en "
    "refuserait 58 (36 + 16 + 6) ; aucune n'est refusée, et jamais plus de 4 lignes de 110 caractères",
    bilan == {"plat d'âme sur poteau": (54, 36, 0, 0), "platine d'about sur poteau": (54, 16, 0, 0),
              "doubles cornières sur poteau": (108, 6, 0, 0)}, str(bilan))
# hypothèses de la note (L19) : ce que le calcul fait s'y lit
def _hyp(NO, R):
    return " ".join(x if isinstance(x, str) else x[0] for x in NO.segments_hypotheses(R))


Rgr = PA.compute(dict(d_nt=50, l_n=150))
h_pa, h_pl, h_dc = _hyp(N_PA, Rgr), _hyp(N_PL, PL.compute(dict(d_nt=50, l_n=150))), _hyp(N_DC, DC.compute())
h_ty = _hyp(N_PAP, PAP.compute(dict(appui=E_PAP.APPUI_AME, nat_N=E_PAP.NAT_TYING, N_Ed=250)))
h_elu = _hyp(N_PAP, PAP.compute(dict(appui=E_PAP.APPUI_AME, N_Ed=60)))
h_v = _hyp(N_PAP, PAP.compute(dict(appui=E_PAP.APPUI_AME)))
h_plp = _hyp(N_PLP, PLP.compute(dict(appui=E_PLP.APPUI_AME, nat_N=E_PLP.NAT_TYING, N_Ed=250)))
chk("hypothèses des notes : « cordons sur hp − 2a » (platine, plat) — le calcul ; la poutre grugée « maintenue au "
    "déversement (donnée de la saisie) », en dernier segment (le kit le raccourcit d'abord : la charte garde trois lignes)",
    "cordons sur hp − 2a" in h_pa and "cordons sur hp − 2a" in h_pl
    and all(h.endswith("poutre grugée maintenue au déversement (donnée de la saisie)") for h in (h_pa, h_pl, h_dc))
    and "cordons sur hp (" not in h_pa and "pleine hauteur" not in h_pl
    and not _hyp(N_PA, PA.compute(dict(d_nt=0, d_nb=0, l_n=0))).endswith("(donnée de la saisie)")
    and N_PA.lignes_hypotheses(Rgr)[-1].endswith("maintenue au déversement."), f"{h_pa} || {h_pl}")
chk("hypothèses — appui dans l'âme : au chaînage « flexion hors plan au chaînage (SCI P358 Check 14) » (platine, plat) ; "
    "sous une traction ELU « NON vérifiée » ; sans traction, ni l'un ni l'autre",
    "flexion hors plan au chaînage (SCI P358 Check 14)" in h_ty and "flexion hors plan au chaînage (SCI P358 Check 14)" in h_plp
    and "flexion hors plan sous la traction ELU NON vérifiée" in h_elu and "flexion hors plan" not in h_v, f"{h_ty} || {h_elu} || {h_v}")
chk("le plat d'âme poutre–poutre par défaut garde ses trois lignes, maintien au déversement compris (le secours ne sert "
    "que si elles ne tiennent pas)",
    len(N_PL.lignes_hypotheses(Rdef)) <= 3 and "maintenue au déversement" in N_PL.lignes_hypotheses(Rdef)[-1],
    str(N_PL.lignes_hypotheses(Rdef)))

import inspect as _insp  # noqa: E402
from acier.assemblages.poutre_poutre.plat_ame import schemas as SC_PL, ecran_saisie as ES_PL2, notation as NT_PL  # noqa: E402
chk("notation (A7) : « zp » est celui du SCI P358 (face d'appui → PREMIÈRE file) ; la position verticale du plat (z_C) "
    "s'écrit « zc », comme aux doubles cornières — saisie, dessin, légende ; le dessin cote zp à côté de z à deux files",
    ES_PL2._LIBELLE["z_C"] == "zc" and ES_PLP._LIBELLE["z_C"] == "zc" and "zc : position du plat" in NT_PL.NOTATIONS
    and "zp : position" not in NT_PL.NOTATIONS and 'id="zc"' in _insp.getsource(SC_PL) and 'sym="zp", val=R.g_B' in _insp.getsource(SC_PL),
    f"{ES_PL2._LIBELLE.get('z_C')} | {NT_PL.NOTATIONS[:120]}")
# A1 (second contrôleur) : « reporter la justification de l'articulation dans le verdict de la note révisée ». Le
# détail éprouvé TENU s'écrit à la conclusion (``note_kit.phrase_articulation``, valeurs nommées par le moteur) ; non
# tenu, le statut « À VÉRIFIER (articulation à justifier : …) » le dit déjà. Mesuré sur 192 notes (exemples du dossier
# et grille d'options) : 43 gagnent la phrase, aucune ne change de disposition, aucun débordement ; sur cinq notes du
# plat sur poteau aux options cumulées, elle prend la place du dernier bloc de formules FACULTATIF.
import pymupdf  # noqa: E402
U_A1 = dict(prof_P="HEA 400", nu_P="S355", prof_S="HEB 340", nu_S="S355", nu_pl="S235", boulon_u="M24", classe="8.8",
            n1_u=3, p1_u=75, e1_u=45, p3_u=90, hp_u=240, bp_u=190, tp_u=10, a_w=9, d_nt=50, d_nb=0, l_n=150, z_C=50, V_Ed=450)
R_A1 = PA.compute(U_A1)
t_A1 = pymupdf.open(stream=N_PA.generer_note(R_A1, {}), filetype="pdf")[0].get_text().replace("\xa0", " ")
t_A1 = " ".join(t_A1.split())
chk("A1 révisée : « ASSEMBLAGE VÉRIFIÉ » et, au verdict, l'articulation justifiée par le détail éprouvé — platine de "
    "10 mm ≤ 0,36·d·√(fub/fy) = 15,9 mm, gorge de 9 mm ≥ 0,8 × pleine résistance = 6,7 mm",
    R_A1.statut == "ASSEMBLAGE VÉRIFIÉ" and KIT.phrase_articulation(R_A1) == "articulation : détail éprouvé tenu (tp = 10 ≤ "
    "15,9\u00a0mm ; a = 9 ≥ 6,7\u00a0mm)"
    and "(γM0 : ANB belge). Articulation : détail éprouvé tenu (tp = 10 ≤ 15,9 mm ; a = 9 ≥ 6,7 mm)." in t_A1,
    KIT.phrase_articulation(R_A1))
chk("plat d'âme par défaut : la phrase nomme la plus petite pince, 2d, la gorge de pleine résistance et les boulons ; une "
    "platine de 20 mm (articulation à justifier) et les doubles cornières n'en ont pas — rien n'est écrit deux fois",
    KIT.phrase_articulation(Rdef) == "articulation : détail éprouvé tenu (e1 du plat = 40 ≥ 2d = 40\u00a0mm ; a = 6 ≥ "
    "6,0\u00a0mm ; 8.8 non précontraints)"
    and KIT.phrase_articulation(Rt) == "" and Rt.a_verifier.startswith("articulation à justifier")
    and KIT.phrase_articulation(DC.compute()) == "", KIT.phrase_articulation(Rdef))
# ================================================================
print("\n=== 9. Aide : la gorge de pleine résistance, la même formule partout (défaut D4) ===")
FORMULE = "tp·fy·βw·γM2/(√2·fu·γM0)"
textes = {"aide de la saisie (poutre)": ES_PL.AIDES["a_w"], "aide de la saisie (poteau)": ES_PLP.AIDES["a_w"],
          "méthode (poutre)": ME_PL.METHODE, "méthode (poteau)": ME_PLP.METHODE,
          "champ (poutre)": " ".join(str(c.get("h") or "") for c in E_PL.CHAMPS.values()),
          "champ (poteau)": " ".join(str(c.get("h") or "") for c in E_PLP.CHAMPS.values())}
chk("les six textes écrivent a ≥ " + FORMULE + " — le diviseur « 2 » de l'ancienne rédaction (0,39·tp en S355) a "
    "disparu ; c'est la formule du moteur (mecanique.a_pleine_resistance_plat : 0,55·tp en S355 avec les fu de l'EN)",
    all(FORMULE in t and "γM2/(2·fu·γM0)" not in t for t in textes.values())
    and pres(MEC.a_pleine_resistance_plat(10, 355, 0.9, 510, 1.25, 1.0), 10 * 355 * 0.9 * 1.25 / (math.sqrt(2) * 510), 1e-12),
    str([k for k, t in textes.items() if FORMULE not in t or "γM2/(2·fu·γM0)" in t]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)
