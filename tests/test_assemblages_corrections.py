# -*- coding: utf-8 -*-
"""Corrections du 25/11/2026 aux DOUBLES CORNIÈRES poutre–poutre (docs/assemblages/ECARTS_ET_CORRECTIONS.md §7).

Le moteur des doubles cornières est la transcription d'un corrigé de référence (HTML), que la parité rejoue. Une
correction s'y écrit donc À CÔTÉ du corrigé : le moteur l'applique partout, sauf quand il rejoue le corrigé
(``parite.materiau_du_corrige`` porte « formules du corrigé »). Cette suite documente, pour chacune, l'ANCIEN résultat
(le corrigé) et le NOUVEAU, avec un calcul à la main — chaque garantie rougit si on retire la correction :

  C1  rupture de bloc des ailes A : le plan cisaillé va vers la plus PETITE pince (les boulons P poussent l'aile A
      vers le haut) — sans effet quand les pinces haute et basse sont égales ;
  C2  ailes A SOUDÉES : l'excentricité de VEd/2 (du talon au cordon) n'est plus une option ;
  C3  cordons P : MEd et le couple HEd·z les tendent, comme ils tendent les boulons P ;
  C4  tronçon en T des ailes A : l'effort est la traction réelle des boulons P (NEd, MEd, HEd·z) ;
  C5  ailes A SOUDÉES en traction : flexion de l'aile, une rotule plastique au talon.

Lancement : python3 tests/test_assemblages_corrections.py
"""
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages import materiau as MT  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, formules  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres.parite import materiau_du_corrige  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)
CORRIGE = materiau_du_corrige(REF)                          # matériau ET formules du corrigé
MAIN = MT.fixe({a["n"]: (a["fy"], a["fu"]) for a in REF["bibliotheques"]["aciers"]})   # même matériau, moteur corrigé
S3 = math.sqrt(3)


def ancien(u):
    return moteur.compute(u, CORRIGE)


def nouveau(u):
    return moteur.compute(u, MAIN)


print("=== 0. Le corrigé reste rejoué tel quel ===")
A0, N0 = ancien({}), nouveau({})
chk("mode corrigé : aucune correction, aucun scalaire de plus (l'instantané garde sa forme)",
    "corrections" not in A0 and N0.get("corrections") == "C1-C5"
    and set(N0) - set(A0) == {"corrections"}, str(sorted(set(N0) - set(A0))))
chk("cas par défaut (boulonné, pinces égales, sans NEd ni MEd) : les corrections ne changent AUCUNE vérification",
    all(pres(a.Ed, n.Ed) and pres(a.Rd, n.Rd) and a.active == n.active for a, n in zip(A0.checks, N0.checks))
    and [c.key for c in A0.checks] == [c.key for c in N0.checks])

print("\n=== C1. Rupture de bloc des ailes A, pinces dissymétriques ===")
u1 = dict(n1P_u=2, p1P_u=60, e1P_u=35)
A1, N1 = ancien(u1), nouveau(u1)
# L100×100×10, hc = 190 : pinces 35 (haut) / 95 (bas) ; Ant = tc·(e2 − d0/2) ; Anv :
#   corrigé : tc·(hc − e1 − 1,5·d0) = 10·(190 − 35 − 33) = 1 220 mm²
#   corrigé du 25/11 : tc·(e1,min + p1 − 1,5·d0) = 10·(35 + 60 − 33) = 620 mm²
chk("ailes A, 2 rangées, pinces 35 / 95 : Anv 1 220 → 620 mm² ; le bloc tombe de 638,8 à 392,9 kN (S355, fu = 510)",
    N1.e1bot_P == 95 and A1.Anv_A == 1220 and N1.Anv_A == 620
    and pres(A1.ck["cbA"].Rd, 2 * (0.5 * 510 * N1.Ant_A / 1.25 + 355 * 1220 / S3) / 1000)
    and pres(N1.ck["cbA"].Rd, 2 * (0.5 * 510 * N1.Ant_A / 1.25 + 355 * 620 / S3) / 1000)
    and pres(N1.ck["cbA"].Rd, 392.87, 1e-4) and pres(A1.ck["cbA"].Rd, 638.82, 1e-4),
    f"{A1.Anv_A} {N1.Anv_A} {A1.ck['cbA'].Rd} {N1.ck['cbA'].Rd}")
N1m = nouveau(dict(n1P_u=2, p1P_u=60, e1P_u=95))
chk("le même bloc pinces échangées (95 / 35) ; l'aile B, poussée vers le bas, garde hc − e1",
    N1m.e1bot_P == 35 and N1m.Anv_A == 620 and pres(N1m.ck["cbA"].Rd, N1.ck["cbA"].Rd)
    and N1.Anv_B == A1.Anv_B, f"{N1m.Anv_A} {N1.Anv_B} {A1.Anv_B}")
chk("pinces égales : aucune différence (cas par défaut)", A0.Anv_A == N0.Anv_A and A0.e1bot_P == A0.e1_P)

print("\n=== C2. Ailes A soudées : l'excentricité n'est plus une option ===")
u2 = dict(fix_P="Soudée")
A2, N2 = ancien(u2), nouveau(u2)
A2oui = ancien(dict(fix_P="Soudée", opt_exc="Oui"))
chk("ailes A soudées, option « Non » : MP = 0 dans le corrigé ; (VEd/2)·(bA − xg) désormais — le cordon se calcule "
    "comme avec l'option « Oui » du corrigé",
    A2.M_P == 0 and pres(N2.M_P, 125 / 2 * N2.ew_P / 1000) and N2.M_P > 0
    and pres(N2.ck["wP"].Ed, A2oui.ck["wP"].Ed) and N2.ck["wP"].Ed > 2 * A2.ck["wP"].Ed,
    f"{A2.M_P} {N2.M_P} {A2.ck['wP'].Ed} {N2.ck['wP'].Ed} {A2oui.ck['wP'].Ed}")
chk("ailes A BOULONNÉES : l'option reste une option (« Non » : MP = 0, la pratique des guides)",
    N0.M_P == 0 and nouveau(dict(opt_exc="Oui")).M_P > 0)
chk("le libellé du modèle et la ligne d'hypothèse le disent",
    N2.mod_P == "Soudé – groupe de cordons avec moment" and A2.mod_P == "Soudé – cisaillement centré")

print("\n=== C3. Cordons P : MEd et le couple HEd·z ===")
u3 = dict(fix_P="Soudée", N_Ed=60, H_Ed=20, M_Ed=5)
A3, N3 = ancien(u3), nouveau(u3)
Lw, Lc, lh = N3.Lw_P, N3.L_C, N3.u.lh_P
Ixx = Lc ** 3 / 12 + 2 * lh * (Lc / 2) ** 2
qn_main = 60 / 2 * 1000 / Lw + 5 / 2 * 1e6 * (Lc / 2) / Ixx + 20 * N3.zeff * 1000 / ((N3.tw_S + 2 * N3.ew_P) * Lw)
chk("qn = NEd/(2·ΣL) + (MEd/2)·(hc/2)/Ixx + HEd·z/((tw + 2·(bA − xg))·ΣL) ; le corrigé n'avait que le premier terme",
    pres(N3.qn_P, qn_main) and pres(A3.qn_P, 60 / 2 * 1000 / Lw) and N3.qn_P > 1.5 * A3.qn_P,
    f"{A3.qn_P} {N3.qn_P} {qn_main}")
chk("sans MEd ni HEd : qn inchangé", pres(nouveau(dict(fix_P="Soudée", N_Ed=60)).qn_P, 60 / 2 * 1000 / Lw))

print("\n=== C4. Tronçon en T : la traction réelle des boulons P ===")
u4 = dict(M_Ed=5)
A4, N4 = ancien(u4), nouveau(u4)
chk("MEd = 5 kNm, NEd = 0 : le corrigé laissait le tronçon en T INACTIF ; il reçoit 2·nP·Ft,Ed",
    not A4.ck["tsA"].active and N4.ck["tsA"].active and N4.Ft_P > 0
    and pres(N4.ck["tsA"].Ed, 2 * N4.n_P * N4.Ft_P) and pres(N4.ck["tsA"].Rd, A4.ck["tsA"].Rd),
    f"{A4.ck['tsA'].active} {N4.ck['tsA'].active} {N4.ck['tsA'].Ed}")
A4n, N4n = ancien(dict(N_Ed=80)), nouveau(dict(N_Ed=80))
chk("NEd seul : rien ne change (2·nP·NEd/(2·nP) = NEd)",
    pres(N4n.ck["tsA"].Ed, 80) and pres(A4n.ck["tsA"].Ed, 80) and pres(N4n.ck["tsA"].eta, A4n.ck["tsA"].eta))

print("\n=== C5. Ailes A soudées en traction ===")
u5 = dict(fix_P="Soudée", N_Ed=50)
A5, N5 = ancien(u5), nouveau(u5)
# L100×100×10 (rc = 12), hc = 190, S355 : Mpl = 0,25·190·10²·355/1e6 = 1,686 kNm ; m = 100 − 10 − 0,8·12 = 80,4 mm
#   NRd = 2·Mpl/m = 41,95 kN → NEd = 50 kN : η = 1,19, NON VÉRIFIÉ (le corrigé : aucune vérification, VÉRIFIÉ)
chk("NEd = 50 kN : flexion de l'aile A, NRd = 2·Mpl/m = 41,95 kN (η = 1,19) — le corrigé rendait VÉRIFIÉ sans la vérifier",
    not A5.ck["tsA"].active and A5.verified and N5.ck["tsA"].active and not N5.verified
    and pres(N5.Mpl_wA, 0.25 * N5.L_C * N5.t_C ** 2 * 355 / 1e6) and pres(N5.m_wA, N5.b_A - N5.t_C - 0.8 * N5.r_C)
    and pres(N5.ck["tsA"].Rd, 2 * N5.Mpl_wA * 1000 / N5.m_wA) and pres(N5.ck["tsA"].Ed, 50)
    and pres(N5.ck["tsA"].Rd, 41.95, 1e-3) and N5.gov.key == "tsA",
    f"{N5.ck['tsA'].Ed} {N5.ck['tsA'].Rd} {N5.verified} {N5.gov.key if N5.gov else None}")
chk("sans traction : la ligne reste inactive ; boulonnée : le tronçon en T du Tableau 6.2, comme avant",
    not nouveau(dict(fix_P="Soudée")).ck["tsA"].active and "tronçon en T" in N4n.ck["tsA"].label
    and "soudées" in N5.ck["tsA"].label)

print("\n=== 6. Les formules affichées se recalculent ===")
nb, vues, sans = 0, 0, []
for R in (N1, N1m, N2, N3, N4, N5):
    for c in R.checks:
        if not c.active:
            continue
        L = formules.lignes(R, c)
        if not L:
            sans.append(c.key)
        for ligne in L or []:
            for s in ligne:
                vues += 1
                ok, v = formules.verifier(s)
                if not ok:
                    nb += 1
                    print("      ", c.key, s.nom, "moteur", s.res, "substitution", v)
chk(f"aucune substitution affichée ne s'écarte de son résultat (cas C1 à C5, {vues} substitutions)",
    nb == 0 and vues > 400 and not sans, f"{nb} écart(s), {vues} vues, sans gabarit {sorted(set(sans))}")
chk("les gabarits corrigés sont bien ceux affichés : Anv par e1,min (C1), NT = 2·nP·Ft,Ed (C4), NT = 2·qn·ΣL et "
    "NRd = 2·Mpl/m (C5), qn à trois termes (C3)",
    any(s.nom == "Anv" and "e1m" in s.template for l in formules.lignes(N1, N1.ck["cbA"]) for s in l)
    and any(s.nom == "NT" and "nP" in s.template for l in formules.lignes(N4, N4.ck["tsA"]) for s in l)
    and any(s.nom == "NRd" and "Mpl" in s.template for l in formules.lignes(N5, N5.ck["tsA"]) for s in l)
    and any(s.nom == "qn" and "qnM" in s.template for l in formules.lignes(N3, N3.ck["wP"]) for s in l))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
