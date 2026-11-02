# -*- coding: utf-8 -*-
"""Charges sur plancher — moteur (plancher/charges : résistances,
vérification, balayage, dimensionnement, bibliothèque, pureté).

Garanties, chacune rougit si on retire ce qu'elle verrouille :
  1. COPIE TENUE des formules BA : les expressions de ``modules/poutre.py``
     sont retrouvées MOT POUR MOT dans son texte, et les inversions de
     ``resistances.py`` les retournent exactement (hᵤ,min(M_adm,béton) = d,
     As,req(M_adm,acier) = As, τ(V_adm) = τ) ; l'exemple du § 2.1 de
     l'étude (64,0 · 34,9 · 84,6) est retrouvé.
  2. Classes de béton : arrondi vers le bas de fck(t), Ecm du Tableau 3.1.
  3. Hourdis : nature Rd → M_adm = M_Rd et coefficients γG / γQ sur les
     efforts (ELU, P-2 tranché le 02/11/2026), service → valeur telle
     quelle et coefficients 1, nature inconnue → erreur ; chape : valeurs
     « avec chape » seulement si collaborante ET classe exigée atteinte
     (P-17), sinon valeurs nues et note.
  4. Effets locaux EN 1992-1-1 : §6.7 recalculé à la main (plafond 3·fcd·Ac0),
     §6.4 recalculé à la main (u1, k, v_Rd,c, plancher v_min).
  5. PIPELINE hourdis : charge seule à mi-portée → M_Q = P·L/4, V_Q = P/2,
     M_Ed = γQ·M_Q contre M_Rd ; POIDS PROPRES dans les efforts : M_Ed =
     γG·M_G + γQ·M_Q, V de même, service → coefficients 1, γG = γQ = 1
     retrouve le service ; charge sur un joint → moitié-moitié ; plateau → empreinte
     a + 2c et pression cohérente, membrure à travers la chape ; filière →
     Σ réactions = P, un élément hors filière ne reçoit rien, filière rigide
     sur trois éléments égaux → trois tiers ; parts annexe C → parts de M,
     efforts de joint, non vérifié sans paramètre et vérifié avec (P-15) ;
     tranchant 100 % sur l'élément chargé (P-3) ; fissuration en
     information (P-11).
  6. PIPELINE dalle et prédalle : charge seule → m = (P·L/4)/t_y + w·L²/8 et
     v_A = (P/2)/(t_y + 0,5·L/2) + w·L/2 ; largeur plafonnée à la prédalle et
     note P-16 à cheval sur un joint ; étais en place P-8 ; refus P-7.
  7. REFUS : engin sans charge (P-13), charge hors plancher, plateau sans f,
     prédalle sans chape ; le résultat reste un dict de statut « refus ».
  8. Balayage : la position la plus défavorable d'une charge seule est à
     mi-portée ; dimensionnement : une filière trouvée fait baisser le pire
     taux et fait monter ceux des voisins (P-14).
  9. Bibliothèque : gabarits sans charge bloquent, profilés avec Iy/Wel,
     filière = E·Iy, plateaux ; résultat JSON ; CAS GELÉS ``tests/ref_chp.json``.
 10. PURETÉ : aucun fichier du paquet n'importe streamlit.

Lancement : python3 tests/test_charges_plancher_moteur.py (depuis la racine).
"""
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from plancher.charges import modele as M  # noqa: E402
from plancher.charges import resistances as RS  # noqa: E402
from plancher.charges import verification as V  # noqa: E402
from plancher.charges import bibliotheque as BIB  # noqa: E402
from plancher.charges.parametres import Parametres  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))


def pres(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def verif(res, famille, objet=None):
    return [v for v in res["verifications"] if v["famille"] == famille and (objet is None or objet in v["objet"])]


# 1. copie tenue des formules BA
with open("modules/poutre.py", "r", encoding="utf-8") as fh:
    src = fh.read()
for frag in ("fyd = fyk / gamma_s", "math.sqrt((m * 1e6) / (alpha_b * b * 10 * mu_val)) / 10",
             "(M_inf_val * 1e6) / (fyd * 0.9 * d_calc_inf * 10)", "V_val * 1e3 / (0.75 * b * h * 100)",
             "tau_1 = 0.016 * fck_cube / 1.05"):
    chk(f"poutre.py porte « {frag[:34]}… »", frag in src)
v = RS.BETON["C30/37"]
f = RS.fyd(500.0, 1.15)
mb = RS.m_adm_beton_kNm(v["alpha_b"], v["mu_a500"], 100.0, 17.0)
ms = RS.m_adm_acier_kNm(524.0, f, 17.0)
chk("hᵤ,min(M_adm,béton) = d", pres(RS.hu_min_cm(mb, v["alpha_b"], 100.0, v["mu_a500"]), 17.0, 1e-9))
chk("As,req(M_adm,acier) = As", pres(RS.as_req_mm2(ms, f, 17.0), 524.0, 1e-9))
tau1 = RS.tau_adm_I(v["fck_cube"])
chk("τ(V_adm) = τadm,I", pres(RS.tau_N_mm2(RS.v_adm_kN(tau1, 100.0, 20.0), 100.0, 20.0), tau1, 1e-9))
chk("exemple de l'étude : 64,0 · 34,9 · 84,6", (round(mb, 1), round(ms, 1), round(RS.v_adm_kN(tau1, 100.0, 20.0), 1)) == (64.0, 34.9, 84.6),
    str((mb, ms, RS.v_adm_kN(tau1, 100.0, 20.0))))

# 2. classes
chk("classe arrondie vers le bas", RS.classe_pour_fck(24.0) == "C20/25" and RS.classe_pour_fck(30.0) == "C30/37"
    and RS.classe_pour_fck(19.0) is None and RS.classe_pour_fck(60.0) == "C40/50")
chk("Ecm Tableau 3.1", pres(RS.Ecm(30.0), 22.0 * (38.0 / 10.0) ** 0.3 * 1e3))

# 3. hourdis : natures et chape
h = M.Hourdis(nom="H", L=6.0, n_elements=4, M_Rd=90.0, V_Rd=95.0, nature=M.RD)
r = RS.resistance_hourdis(h, 1.35, 1.5)
chk("nature Rd : M_adm = M_Rd, V_adm = V_Rd, coefficients γG = 1,35 et γQ = 1,5 sur les efforts (P-2)",
    pres(r["M_adm"], 90.0) and pres(r["V_adm"], 95.0) and r["gamma_G"] == 1.35 and r["gamma_Q"] == 1.5 and "ELU" in r["methode"])
h.nature = M.SERVICE
rs_ = RS.resistance_hourdis(h, 1.35, 1.5)
chk("nature service : tel quel, coefficients 1", pres(rs_["M_adm"], 90.0) and rs_["gamma_G"] == 1.0 and rs_["gamma_Q"] == 1.0)
h.nature = "Rk"
try:
    RS.resistance_hourdis(h, 1.35, 1.5)
    chk("nature inconnue : erreur", False)
except ValueError:
    chk("nature inconnue : erreur", True)
hc = M.Hourdis(nom="H", L=6.0, n_elements=4, M_Rd=90.0, V_Rd=95.0, M_Rd_chape=112.0, V_Rd_chape=104.0,
               chape=M.Chape(0.05, 24.0, True, 20.0))
chk("chape admise : valeurs avec chape", RS.resistance_hourdis(hc)["avec_chape"] and pres(RS.resistance_hourdis(hc)["M_adm"], 112.0))
hc.chape = M.Chape(0.05, 18.0, True, 20.0)
rc = RS.resistance_hourdis(hc)
chk("classe exigée non atteinte : valeurs nues + note (P-17)", not rc["avec_chape"] and pres(rc["M_adm"], 90.0) and "P-17" in rc["note"])
hc.chape = M.Chape(0.05, 30.0, True, None)
chk("classe exigée inconnue : refus des valeurs avec chape", not RS.resistance_hourdis(hc)["avec_chape"])
hc.chape = M.Chape(0.05, 30.0, False, 20.0)
chk("chape non collaborante : valeurs nues", not RS.resistance_hourdis(hc)["avec_chape"])
chk("chape comptée en poids (permanent)", pres(hc.g_perm(), 2.9 + 1.25) and pres(hc.g_total(), 2.9 + 1.25))

# 4. effets locaux
pl = RS.pression_localisee_Rd(300.0, 300.0, 200.0, 30.0)
fcd = 30.0 / 1.5
chk("§6.7 : F_Rdu = Ac0·fcd·√(Ac1/Ac0)", pres(pl["F_Rdu_kN"], 300 * 300 * fcd * math.sqrt((500 * 500) / (300 * 300)) / 1e3))
pl2 = RS.pression_localisee_Rd(100.0, 100.0, 1000.0, 30.0)
chk("§6.7 : plafond 3·fcd·Ac0", pres(pl2["F_Rdu_kN"], 3 * fcd * 100 * 100 / 1e3))
po = RS.poinconnement(60.0, 150.0, 150.0, 170.0, 30.0, 524.0 / (1000 * 170.0))
u1 = 2 * (150 + 150) + 2 * math.pi * 340
k = min(1 + math.sqrt(200 / 170), 2.0)
vrdc = max(0.18 / 1.5 * k * (100 * 524.0 / (1000 * 170.0) * 30.0) ** (1 / 3), 0.035 * k ** 1.5 * math.sqrt(30.0))
chk("§6.4 : u1, k, v_Rd,c à la main", pres(po["u1_mm"], u1) and pres(po["k"], k) and pres(po["v_Rd_c"], vrdc)
    and pres(po["v_Ed"], 60e3 / (u1 * 170.0)))
po2 = RS.poinconnement(60.0, 150.0, 150.0, 170.0, 30.0, 0.0001)
chk("§6.4 : plancher v_min", pres(po2["v_Rd_c"], 0.035 * k ** 1.5 * math.sqrt(30.0)))
chk("§6.4 : u0 non vérifié si coef None", "taux_u0" not in RS.poinconnement(60.0, 150.0, 150.0, 170.0, 30.0, 0.003, coef_vrd_max=None))

# 5. pipeline hourdis
H = M.Hourdis(nom="H", L=6.0, n_elements=5, g=0.0, q_presentes=0.0, M_Rd=90.0, V_Rd=95.0, I=1.0e-3, p_loc_adm=0.7)
e = M.Empreinte("E", 3.0, 3.0, 0.3, 0.3, 20.0)          # centre de H3 (2,4–3,6)
r = V.verifier(H, [e], {})
h3 = next(x for x in r["elements"] if x["nom"] == "H3")
chk("charge seule : M_Q = P·L/4, M_Ed = γQ·M_Q contre M_Rd", pres(h3["M_Q"], 30.0) and pres(h3["M_max"], 1.5 * 30.0)
    and pres(h3["taux_M"], 1.5 * 30.0 / 90.0) and h3["M_G"] == 0.0)
chk("charge seule : V_Ed = γQ·P/2 contre V_Rd", pres(h3["V_max"], 1.5 * 10.0) and pres(h3["taux_V"], 1.5 * 10.0 / 95.0))
chk("autres éléments vides sans annexe C", all(x["M_max"] == 0.0 for x in r["elements"] if x["nom"] != "H3"))
chk("statut ok, gouverne flexion H3", r["statut"] == "ok" and r["gouverne"]["objet"] == "H3" and r["gouverne"]["famille"] == "flexion")
chk("membrure : pression = P/(a·b) contre p_loc_adm", pres(verif(r, "local")[0]["E"], 20e3 / (300 * 300)) and pres(verif(r, "local")[0]["taux"], 20e3 / 9e4 / 0.7))
ej = M.Empreinte("E", 3.0, 2.4, 0.3, 0.3, 20.0)          # sur le joint H2 | H3
rj = V.verifier(H, [ej], {})
m2 = next(x["M_Q"] for x in rj["elements"] if x["nom"] == "H2")
m3 = next(x["M_Q"] for x in rj["elements"] if x["nom"] == "H3")
chk("charge sur un joint : moitié-moitié (levier)", pres(m2, 15.0) and pres(m3, 15.0))
# plateau
plat = M.Plateau("tôle", 235.0, 20.0, 600.0, 600.0)
rp = V.verifier(H, [e], {"E": plat})
pt = rp["patches"][0]
chk("plateau : empreinte a + 2c, pression P/aire", pres(pt["a"], (300 + 2 * verif(rp, "plateau")[0]["E"]) / 1e3)
    and pres(pt["pression"], 20e3 / (pt["a"] * 1e3 * pt["b"] * 1e3)))
chk("plateau : M de l'élément inchangé (résultante)", pres(next(x["M_Q"] for x in rp["elements"] if x["nom"] == "H3"), 30.0))
# tôle de 20 mm : c plafonné par le bord (plateau entier) ; tôle de 6 mm : bande partielle, calculée sous
# la pression ELU γQ·P / aire efficace (P-2) — plus courte que sous la pression de service
e95 = M.Empreinte("E", 3.0, 3.0, 0.3, 0.3, 95.0)          # 95 kN : la bande d'une tôle de 6 mm ne couvre pas le plateau
rp6 = V.verifier(H, [e95], {"E": M.Plateau("tôle fine", 235.0, 6.0, 600.0, 600.0)})
pt6, c6 = rp6["patches"][0], verif(rp6, "plateau")[0]["E"]
p_Ed = 1.5 * 95e3 / (pt6["a"] * 1e3 * pt6["b"] * 1e3)
chk("plateau : bande plastique calculée sous la pression ELU γQ·P (P-2)",
    not verif(rp6, "plateau")[0]["complet"] and pres(c6, 6.0 * math.sqrt(235.0 / (3 * p_Ed)), 1e-3)
    and c6 < 6.0 * math.sqrt(235.0 / (3 * p_Ed / 1.5)) and "γQ" in verif(rp6, "plateau")[0]["methode"],
    f"c = {c6}, a_eff = {pt6['a']}")
Hc = M.Hourdis(**{**H.__dict__, "chape": M.Chape(0.05, 30.0, True, 20.0)})
rc = V.verifier(Hc, [e], {"E": plat})
ptc = rc["patches"][0]
chk("membrure à travers la chape : (a + 2t)(b + 2t)", pres(verif(rc, "local")[0]["E"], 20e3 / ((ptc["a"] * 1e3 + 100) * (ptc["b"] * 1e3 + 100))))
Hn = M.Hourdis(**{**H.__dict__, "p_loc_adm": None})
rn = V.verifier(Hn, [e], {})
chk("membrure sans donnée ni répartiteur : nok « répartiteur exigé » (P-5)", verif(rn, "local")[0]["statut"] == "nok" and rn["statut"] == "nok")
rn2 = V.verifier(Hn, [e], {"E": M.Plateau("épais", 235.0, 100.0, 600.0, 600.0)})
chk("membrure sans donnée, plateau complet : non vérifié (P-5)", verif(rn2, "local")[0]["statut"] == "non_verifie" and rn2["statut"] == "ok")
# filière
fil = M.Filiere("fil", EI=1.0e9, W_el=1.0e6, fy=235.0, longueur=3.6, x=3.0, y=3.0, refs=["E"])   # rigide, H2 H3 H4 entiers
rf = V.verifier(H, [e], {"E": fil})
ms = {x["nom"]: x["M_Q"] for x in rf["elements"]}
chk("filière rigide sur trois éléments : trois tiers", all(pres(ms[n], 10.0, 1e-4) for n in ("H2", "H3", "H4")), str(ms))
chk("filière : éléments hors filière vides", ms["H1"] == 0.0 and ms["H5"] == 0.0)
chk("filière : Σ réactions = P", pres(sum(R for _, R in verif(rf, "filiere")[0]["reactions"]), 20.0, 1e-9))
fil2 = M.Filiere("fil", EI=1.0, W_el=100.0, fy=235.0, longueur=3.6, x=3.0, y=3.0, refs=["E"])      # très souple
rf2 = V.verifier(H, [e], {"E": fil2})
chk("filière très souple : la charge reste sur H3", next(x["M_Q"] for x in rf2["elements"] if x["nom"] == "H3") > 29.0)
chk("filière : flexion vérifiée à l'ELU (γQ·M ≤ W·fy/γM0)", verif(rf, "filiere")[0]["taux"] is not None
    and pres(verif(rf, "filiere")[0]["taux"], 1.5 * verif(rf, "filiere")[0]["E"] / verif(rf, "filiere")[0]["R"]))
# annexe C et joints
Ha = M.Hourdis(**{**H.__dict__, "parts_annexe_C": {0: 0.5, 1: 0.25}})
ra = V.verifier(Ha, [e], {})
ma = {x["nom"]: x["M_Q"] for x in ra["elements"]}
chk("annexe C : parts de moment 0,25 · 0,5 · 0,25", pres(ma["H3"], 15.0) and pres(ma["H2"], 7.5) and pres(ma["H4"], 7.5))
va = {x["nom"]: x["V_max"] for x in ra["elements"]}
chk("tranchant : 100 % de la réaction sur l'élément chargé, rien aux voisins (P-3), V_Ed = γQ·P/2",
    pres(va["H3"], 1.5 * 10.0) and va["H2"] == 0.0 and va["H4"] == 0.0
    and "sans partage par les joints" in verif(ra, "tranchant")[0]["methode"])
js = verif(ra, "joint")
chk("joints : effort transmis = part du voisin, non vérifié sans paramètre (P-15)",
    len(js) == 2 and all(pres(j["E"], 5.0) and j["statut"] == "non_verifie" for j in js), str(js))
raj = V.verifier(Ha, [e], {}, Parametres(joint_adm_kN_m=5.0, joint_longueur_m=1.0))
chk("joints : vérifiés avec paramètres, taux = γQ·F/(adm·l)", all(pres(j["taux"], 1.5 * 5.0 / 5.0) for j in verif(raj, "joint")))
Hp = M.Hourdis(**{**H.__dict__, "parts_annexe_C": {0: 0.5, 1: 0.25}, "etat": M.POSES})
chk("annexe C inactive sur des hourdis posés", next(x["M_Q"] for x in V.verifier(Hp, [e], {})["elements"] if x["nom"] == "H3") == 30.0)
Hf = M.Hourdis(**{**H.__dict__, "M_cr": 25.0})
chk("fissuration : ligne d'information (P-11)", verif(V.verifier(Hf, [e], {}), "fissuration")[0]["statut"] == "info")
rr = V.verifier(M.Hourdis(**{**H.__dict__, "nature": M.SERVICE}), [e], {})
chk("nature service dans le pipeline", pres(next(x["taux_M"] for x in rr["elements"] if x["nom"] == "H3"), 30.0 / 90.0))
# charge dynamique et charges réparties présentes
Hg = M.Hourdis(**{**H.__dict__, "g": 2.0, "q_presentes": 1.0})            # b = 1,2 m : w_g = 2,4 kN/m, w_q = 1,2 kN/m
rg = V.verifier(Hg, [M.Empreinte("E", 3.0, 3.0, 0.3, 0.3, 20.0, phi=1.4)], {})
hg3 = next(x for x in rg["elements"] if x["nom"] == "H3")
M_G, M_Q = 2.4 * 36 / 8, 1.2 * 36 / 8 + 1.4 * 30.0
chk("P-2 : POIDS PROPRES dans les efforts — M_G = w_g·L²/8, M_Q = w_q·L²/8 + φ·P·L/4", pres(hg3["M_G"], M_G) and pres(hg3["M_Q"], M_Q))
chk("P-2 : M_Ed = 1,35·M_G + 1,5·M_Q (nature Rd)", pres(hg3["M_max"], 1.35 * M_G + 1.5 * M_Q))
chk("P-2 : V_Ed = 1,35·w_g·L/2 + 1,5·(w_q·L/2 + φ·P/2)", pres(hg3["V_max"], 1.35 * 2.4 * 3.0 + 1.5 * (1.2 * 3.0 + 1.4 * 10.0)))
rgs = V.verifier(M.Hourdis(**{**Hg.__dict__, "nature": M.SERVICE}), [M.Empreinte("E", 3.0, 3.0, 0.3, 0.3, 20.0, phi=1.4)], {})
chk("P-2 : nature service → M = M_G + M_Q, coefficients 1", pres(next(x["M_max"] for x in rgs["elements"] if x["nom"] == "H3"), M_G + M_Q))
rg1 = V.verifier(Hg, [M.Empreinte("E", 3.0, 3.0, 0.3, 0.3, 20.0, phi=1.4)], {}, Parametres(gamma_G=1.0, gamma_Q=1.0))
chk("P-2 : γG = γQ = 1 retrouve le moment de service", pres(next(x["M_max"] for x in rg1["elements"] if x["nom"] == "H3"), M_G + M_Q))
chk("P-2 : l'ancien ÷ 1,5 uniforme n'est plus là (taux ≠ (M_G + M_Q)/(M_Rd/1,5))",
    not pres(hg3["taux_M"], (M_G + M_Q) / (90.0 / 1.5)) and pres(hg3["taux_M"], (1.35 * M_G + 1.5 * M_Q) / 90.0))
rgf = V.verifier(M.Hourdis(**{**Hg.__dict__, "M_cr": 25.0}), [M.Empreinte("E", 3.0, 3.0, 0.3, 0.3, 20.0, phi=1.4)], {})
chk("P-11 : la fissuration compare le moment de SERVICE M_G + M_Q",
    pres(next(v for v in verif(rgf, "fissuration") if v["objet"] == "H3")["E"], M_G + M_Q))
chk("hypothèses : γG, γQ, poids propres et tranchant P-3 écrits en toutes lettres",
    any("γG = 1.35" in t and "γQ = 1.5" in t for t in rg["hypotheses"]) and any("sans partage par les joints" in t for t in rg["hypotheses"])
    and any("poids propres" in t for t in rg["hypotheses"]))

# 6. pipeline dalle et prédalle
D = M.DalleBA(nom="D", L=6.0, largeur=6.0, h=0.20, d=0.17, beton="C30/37", As_inf=524.0)
rd = V.verifier(D, [M.Empreinte("E", 3.0, 3.0, 0.15, 0.15, 40.0)], {})
ty = 0.15 + 0.2
w = 25 * 0.2
chk("dalle : m = (P·L/4)/t_y + w·L²/8", pres(verif(rd, "flexion")[0]["E"], 60.0 / ty + w * 36 / 8))
chk("dalle : v_A = (P/2)/(t_y + 0,5·L/2) + w·L/2", pres(verif(rd, "tranchant")[0]["E"], 20.0 / (ty + 1.5) + w * 3.0))
chk("dalle : résistances de resistance_dalle", pres(verif(rd, "flexion")[0]["R"], RS.resistance_dalle(D)["M_adm"]))
chk("dalle : ligne P-4 (k non transcrit)", any("P-4" in v["methode"] for v in verif(rd, "hypothese")))
chk("dalle : §6.7 et §6.4 présents", len(verif(rd, "local")) == 3)
chk("dalle : effets locaux à l'ELU, E = γQ·P (P-2, P-9)", pres(verif(rd, "local")[0]["E"], 1.5 * 40.0) and "γQ" in verif(rd, "local")[0]["methode"])
# taux par bande : la bande chargée porte le taux global, une bande non chargée porte celui du poids
# propre seul — jamais une valeur « concentrée » sur une largeur nulle (défaut du 02/11/2026, note PDF)
bandes = {b["nom"]: b for b in rd["elements"]}
chk("dalle : taux par bande ≤ taux global, bande chargée = taux global, bande vide = poids propre seul",
    max(b["taux_M"] for b in bandes.values()) <= verif(rd, "flexion")[0]["taux"] * (1 + 1e-9)
    and pres(bandes["B3"]["taux_M"], verif(rd, "flexion")[0]["taux"]) and pres(bandes["B1"]["M_max"], w * 36 / 8),
    str({k: round(b["taux_M"], 3) for k, b in bandes.items()}))
chk("dalle : flexion et tranchant par mètre en service, coefficients 1 (méthode de l'application)",
    any("coefficients 1" in t for t in rd["hypotheses"]))
rdk = V.verifier(D, [M.Empreinte("E", 3.0, 3.0, 0.15, 0.15, 40.0)], {}, Parametres(k_flexion_heft240=2.5))
chk("dalle : k transcrit élargit la bande", verif(rdk, "flexion")[0]["E"] < verif(rd, "flexion")[0]["E"])
P1 = M.Predalle(nom="P", L=6.0, largeur=4.8, h=0.20, d=0.17, b_predalle=2.4, joints_armes=False)
rpd = V.verifier(P1, [M.Empreinte("E", 3.0, 1.2, 0.15, 0.15, 40.0)], {})
chk("prédalle joints non armés : largeur plafonnée", any("plafonnée" in v["methode"] for v in verif(rpd, "hypothese")))
rpj = V.verifier(M.Predalle(**{**P1.__dict__, "joints_armes": True}), [M.Empreinte("E", 3.0, 2.4, 0.15, 0.15, 40.0)], {})
chk("prédalle : charge à cheval sur un joint signalée (P-16)", any("P-16" in v["methode"] for v in verif(rpj, "hypothese")))
rpe = V.verifier(M.Predalle(**{**P1.__dict__, "portee_etayee": 2.0}), [M.Empreinte("E", 1.0, 1.2, 0.15, 0.15, 40.0)], {})
chk("prédalle étayée : portée réduite et avertissement (P-8)", any("P-8" in v["methode"] and v["statut"] == "attention" for v in verif(rpe, "hypothese"))
    and verif(rpe, "flexion")[0]["E"] < verif(rpd, "flexion")[0]["E"])

# 7. refus
gab = M.Gabarit("g", "pieds", [M.AppuiGabarit("P1", 0.0, 0.0, 0.3, 0.3, 0.0)])
chk("refus : engin sans charge (P-13)", V.verifier(H, [M.Engin("E", gab, 3.0, 3.0)], {})["statut"] == "refus")
chk("refus : charge hors plancher", V.verifier(H, [M.Empreinte("E", 7.0, 3.0, 0.3, 0.3, 10.0)], {})["statut"] == "refus")
chk("refus : plateau sans f", V.verifier(H, [e], {"E": M.Plateau("osb", None, 18.0, 600.0, 600.0)})["statut"] == "refus")
chk("refus : prédalle sans chape (P-7)", V.verifier(M.Predalle(**{**P1.__dict__, "chape_durcie": False}), [e], {})["statut"] == "refus")
chk("refus : message présent", "P-7" in V.verifier(M.Predalle(**{**P1.__dict__, "chape_durcie": False}), [e], {})["message"])
# hourdis sans résistances de catalogue (gabarit vide, M_Rd = V_Rd = 0) : REFUS nommé, jamais un
# taux infini présenté comme « non conforme » (écran, 02/11/2026)
r0 = V.verifier(M.Hourdis(**{**H.__dict__, "M_Rd": 0.0, "V_Rd": 0.0}), [e], {})
chk("refus : hourdis sans M_Rd / V_Rd (catalogue à renseigner)", r0["statut"] == "refus" and "M_Rd et V_Rd" in r0["message"])

# 8. balayage et dimensionnement
gab1 = M.Gabarit("g", "pieds", [M.AppuiGabarit("P", 0.0, 0.0, 0.3, 0.3, 20.0)])
b = V.balayer(H, [M.Engin("E", gab1, 1.0, 3.0)], {}, "E")
chk("balayage : pire position à mi-portée", pres(b["pire"]["x"], 3.0, 1e-6) and len(b["points"]) == 61, str(b["pire"]))
gab2 = M.Gabarit("g", "pieds", [M.AppuiGabarit("P", 0.0, 0.0, 0.3, 0.3, 50.0)])
eng = M.Engin("E", gab2, 3.0, 3.0)
cands = [(n, 210000e3 * BIB.profils()[n]["Iy"] * 1e-8, BIB.profils()[n]["Wel"]) for n in BIB.candidats_filieres(3.6)]
d = V.dimensionner_filiere(H, [eng], {}, [["E.P"]], cands,
                           lambda n, l, x, y, refs: BIB.filiere_profil(n, l, x, y, refs=refs))
chk("dimensionnement : trouvé, pire taux ≤ 1, voisins montés (P-14)",
    d["trouve"] and d["taux_max"] <= 1.0 and d["avant"] > 1.0 and len(d["taux_montes"]) >= 2, str({k: d.get(k) for k in ("trouve", "taux_max", "avant")}))

# 9. bibliothèque, JSON, cas gelés
gabs = BIB.gabarits()
chk("bibliothèque : gabarits génériques sans charge", len(gabs) >= 5 and all(not g.charges_renseignees() for g in gabs.values()))
chk("bibliothèque : profilés avec Iy et Wel", all(p["Iy"] > 0 and p["Wel"] > 0 for p in BIB.profils().values()))
ff = BIB.filiere_profil("HEB 140", 3.6, 3.0, 3.0)
chk("filière catalogue : EI = E·Iy, W_el et b", pres(ff.EI, 210000e3 * BIB.profils()["HEB 140"]["Iy"] * 1e-8) and ff.W_el == BIB.profils()["HEB 140"]["Wel"]
    and pres(ff.b_semelle, BIB.profils()["HEB 140"]["b"] / 1000.0))
chk("bibliothèque : plateaux et types de hourdis", len(BIB.plateaux()) >= 4 and len(BIB.types_hourdis()) >= 1)
cas = {}
Hx = M.Hourdis(nom="SP", L=7.2, n_elements=8, I=1.05e-3, M_Rd=90.0, V_Rd=95.0, p_loc_adm=0.7, q_presentes=1.0,
               parts_annexe_C={0: 0.52, 1: 0.24})
gm = gabs["Télescopique rotatif (type MRT 2150) · levage sur stabilisateurs"]
for a in gm.appuis:
    a.P = 60.0 if a.nom in ("P2", "P4") else 40.0
engx = M.Engin("E1", gm, 3.0, 4.8)
plx = BIB.plateaux()["Tôle acier S235 · 600 × 600 × 20"]
cas["hourdis_mrt_plateaux"] = V.verifier(Hx, [engx], {x.ref: plx for x in engx.empreintes()})
fx = BIB.filiere_profil("HEB 140", 3.6, engx.empreintes()[1].x, engx.empreintes()[1].y, refs=["E1.P2"])
cas["hourdis_mrt_filiere"] = V.verifier(Hx, [engx], {"E1.P2": fx})
cas["dalle_etai"] = V.verifier(D, [M.Empreinte("E", 3.0, 3.0, 0.15, 0.15, 40.0), M.ChargeLineaire("F", 2.0, 4.0, 2.0, 0.1, 8.0)], {})
cas["predalle_etais_filiere"] = V.verifier(P1, [M.Empreinte("E1", 2.0, 2.4, 0.15, 0.15, 30.0), M.Empreinte("E2", 2.0, 3.6, 0.15, 0.15, 30.0)],
                                           {"E1": BIB.filiere_profil("IPE 140", 2.4, 2.0, 3.0, refs=["E1", "E2"]), "E2": BIB.filiere_profil("IPE 140", 2.4, 2.0, 3.0, refs=["E1", "E2"])})
txt = json.dumps(cas, ensure_ascii=False, sort_keys=True, default=float)
chk("résultats sérialisables en JSON", len(txt) > 1000)
REF = os.path.join("tests", "ref_chp.json")


def _plat(o, pref=""):
    """Aplatit un résultat en {chemin: valeur} pour la comparaison."""
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_plat(v, f"{pref}/{k}"))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            out.update(_plat(v, f"{pref}[{i}]"))
    else:
        out[pref] = o
    return out


if os.path.exists(REF):
    with open(REF, "r", encoding="utf-8") as fh:
        ref = json.load(fh)
    a, b_ = _plat(json.loads(txt)), _plat(ref)
    diff = [k for k in set(a) | set(b_) if k not in a or k not in b_
            or (isinstance(a[k], float) and isinstance(b_[k], (int, float)) and not pres(a[k], b_[k], 1e-9))
            or (not isinstance(a[k], float) and a[k] != b_[k])]
    chk("cas gelés ref_chp.json : identiques", not diff, str(sorted(diff)[:8]))
else:
    with open(REF, "w", encoding="utf-8") as fh:
        fh.write(txt)
    chk("cas gelés ref_chp.json : ÉCRITS (première exécution)", True)

# 10. pureté
imp = []
for dp, _, fs in os.walk("plancher"):
    for f_ in fs:
        if f_.endswith(".py"):
            with open(os.path.join(dp, f_), "r", encoding="utf-8") as fh:
                if "streamlit" in fh.read():
                    imp.append(f_)
chk("pureté : aucun streamlit dans plancher/", not imp, str(imp))

print(f"RÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("  ÉCHEC :", nom, info)
sys.exit(1 if KO else 0)
