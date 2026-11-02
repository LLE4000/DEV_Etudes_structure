# -*- coding: utf-8 -*-
"""`acier/assemblages/mecanique.py` est une COPIE : ce test la TIENT.

Les briques de calcul partagées des attaches articulées (pression
diamétrale, groupe excentré, efforts par boulon, poutre grugée, déversement
du plat long, règles de ductilité, groupe de fixations, tronçon en T,
longueurs efficaces) existent désormais DEUX fois : dans les moteurs des
trois attaches poutre–poutre, qui ne sont pas touchés, et dans
`mecanique.py`, que les attaches poutre–poteau réutilisent
(`docs/assemblages/POUTRE_POTEAU.md` §10.1 et §10.2).

Une copie non tenue est une seconde vérité. Ce test est le lien : pour
chacun des **56 cas figés** du dépôt, il recalcule chaque grandeur par
`mecanique.py` et la compare à celle du moteur d'origine. Si une formule
change d'un seul côté, il rougit et NOMME le cas et la grandeur.

  1. PLAT D'ÂME — les 12 cas de `tests/ref_plat_ame.json` : `section_grugee`,
     `bearing` (plat et âme portée), `groupe_excentre`, `bolt_table`,
     `lam_LT`, `pb_bs5950`, `a_pleine_resistance_plat`.
  2. PLATINE D'ABOUT — les 13 cas de `tests/ref_platine_about.json` :
     `section_grugee`, `portance_bout_interieur` (platine et âme porteuse),
     `groupe_boulons`, `tp_ductile`, `coef_pleine_resistance`,
     `leff_platine_bords_libres`, `troncon_T` (ELU et tying). Puis, parce
     que les cas figés n'atteignent pas toutes les branches du Tableau 6.4,
     2 000 GÉOMÉTRIES TIRÉES (graine fixe) comparées directement à
     `platine_about.tronçon_en_T`, et les branches restantes recalculées à
     la main.
  3. DOUBLES CORNIÈRES — les 31 cas de parité de
     `acier/reference/reference_double_corniere.json` : `section_grugee`,
     `bearing` (ailes A et B, âme portée, âme porteuse avec le « gauge »),
     `groupe_excentre` (groupes S et P), `bolt_table`, `troncon_T`.
  4. LA BRIQUE NEUVE — `leff_aile_poteau` (EN 1993-1-8 Tableau 6.4, aile de
     poteau NON RAIDIE, aucune rangée d'extrémité) contre les nombres
     recalculés à la main de l'étude §7.3, et `troncon_T` sur les deux
     tronçons en T d'aile de poteau du §7.3 (platine et cornières).
  5. PURETÉ — `mecanique.py` n'importe que `math`, `acier.js` et
     `acier.resultats` : ni Streamlit, ni interface, ni dessin.

Écart admis sur les comparaisons de copie : 1e-12 en relatif. Les deux
copies n'associent pas toujours les mêmes produits (`0,25·l·t²·(f/γ)` d'un
côté, `(0,25·l·t²·f)/γ` de l'autre) : l'écart de représentation binaire est
de l'ordre de 1e-16, mille fois sous le seuil. Les ancres manuelles du §7.3
sont imprimées à 5 ou 6 chiffres : écart admis 1e-4 en relatif.

Lancement : python3 tests/test_assemblages_mecanique.py (depuis la racine).
"""
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages import mecanique as MC  # noqa: E402
from acier.js import mn  # noqa: E402
from acier import resistances as RS  # noqa: E402

OK, KO = [], []
TOTAL = 0


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def proche(a, b, tol=1e-12):
    """Égalité à ``tol`` en relatif ; égalité stricte hors nombres ; NaN et
    ±Infinity doivent coïncider des deux côtés."""
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
            or not isinstance(b, (int, float)):
        return a == b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


class Lien:
    """Un cas figé : compare les grandeurs de `mecanique` à celles du moteur
    et retient les écarts, avec leur nom."""

    def __init__(self, cas, tol=1e-12):
        self.cas = cas
        self.tol = tol
        self.n = 0
        self.ecarts = []

    def eq(self, nom, calcule, moteur):
        self.n += 1
        if not proche(calcule, moteur, self.tol):
            self.ecarts.append(f"{nom} : mecanique {calcule!r} ≠ moteur {moteur!r}")

    def eq_dict(self, R, paires, src):
        """``paires`` : (clé de `mecanique`, clé du moteur)."""
        for k_mc, k_R in paires:
            self.eq(k_R, src[k_mc], R[k_R])

    def eq_bearing(self, R, key, **kw):
        b = MC.bearing(**kw)
        for champ, prefixe in (("abv", "abv_"), ("k1v", "k1v_"), ("abh", "abh_"),
                               ("k1h", "k1h_"), ("Fbv", "Fbv_"), ("Fbh", "Fbh_"),
                               ("ipd", "ipd_")):
            self.eq(prefixe + key, b[champ], R[prefixe + key])

    def eq_table(self, nom, calculee, moteur):
        self.n += 1
        if len(calculee) != len(moteur):
            self.ecarts.append(f"{nom} : {len(calculee)} boulons ≠ {len(moteur)}")
            return
        for k, (a, b) in enumerate(zip(calculee, moteur)):
            for champ in ("i", "j", "x", "y", "fx", "fz", "f"):
                if not proche(getattr(a, champ), getattr(b, champ), self.tol):
                    self.ecarts.append(f"{nom}[{k}].{champ} : {getattr(a, champ)!r} "
                                       f"≠ {getattr(b, champ)!r}")

    def verdict(self, titre):
        global TOTAL
        TOTAL += self.n
        chk(f"{titre} — {self.cas} : {self.n} grandeurs, {len(self.ecarts)} écart(s)",
            not self.ecarts, " | ".join(self.ecarts[:3]))


GRUGEE = [("cas_g", "cas_g"), ("Asec", "Asec_S"), ("h_T", "h_T"), ("A_Tee", "A_Tee"),
          ("Av", "Av_S"), ("VRd_g", "VRd_gS"), ("hw_T", "hw_T"), ("zb_T", "zb_T"),
          ("I_N", "I_N"), ("W_N", "W_N"), ("rho_N", "rho_N"), ("Mv_N", "Mv_N"),
          ("el_N", "el_N"), ("lim_N", "lim_N"), ("ln_max", "ln_max"), ("val_N", "val_N")]


def profil_S(R):
    return {"h": R.h_S, "b": R.b_S, "tw": R.tw_S, "tf": R.tf_S, "r": R.r_S}


# =========================================================================
print("=== 1. Plat d'âme — les 12 cas figés de tests/ref_plat_ame.json ===")
from acier.assemblages.poutre_poutre.plat_ame import moteur as M_fpl  # noqa: E402

with open("tests/ref_plat_ame.json", encoding="utf-8") as fh:
    REF_FPL = json.load(fh)
chk("12 cas figés chargés", len(REF_FPL) == 12, str(len(REF_FPL)))
for nom, cas in REF_FPL.items():
    R = M_fpl.compute(cas["inp"])
    u = R.u
    L = Lien(nom)
    # poutre grugée : le plus gros bloc dupliqué du dépôt
    L.eq_dict(R, GRUGEE, MC.section_grugee(profil_S(R), R.fy_S, u.d_nt, u.d_nb, u.g_M0,
                                           V=u.V_Ed, maintenue=u.lt_ok == "Oui"))
    # groupe de boulons excentré
    g = MC.groupe_excentre(R.n_1, R.n_2, R.p_1, u.p2_u, u.V_Ed, u.N_Ed, R.M_S)
    L.eq_dict(R, [("n", "n_b"), ("Ip", "Ip"), ("xm", "xm"), ("ym", "ym"),
                  ("Fz", "Fz_b"), ("Fx", "Fx_b"), ("F", "F_b")], g)
    L.eq_table("tab", MC.bolt_table(R.n_1, R.n_2, R.p_1, u.p2_u, u.V_Ed, u.N_Ed,
                                    R.M_S, R.Ip), R.tab)
    # pression diamétrale : plat (deux bords libres) et âme de la poutre portée
    commun = dict(fub=R.f_ub, d=R.d_b, d0=R.d_0, p1=R.p_1, n1=R.n_1, p2=u.p2_u,
                  n2=R.n_2, gM2=u.g_M2, ktrou=R.k_trou, Fz=R.Fz_b, Fx=R.Fx_b)
    L.eq_bearing(R, "L", t=R.t_p, fu=R.fu_L, e1=mn(R.e_1, R.e1bot), e2=R.e2_p, **commun)
    L.eq_bearing(R, "S", t=R.tw_S, fu=R.fu_S, e2=R.e2b, **commun,
                 e1=mn(R.e1b_S, R.h_e) if u.d_nb > 0 else R.e1b_S)
    # déversement du plat LONG et gorge de pleine résistance
    L.eq("lam_LT", MC.lam_LT(R.z_p, R.h_p, R.t_p), R.lam_LT)
    L.eq("pb_LT", MC.pb_bs5950(R.lam_LT, R.fy_L), R.pb_LT)
    L.eq("a_plein", MC.a_pleine_resistance_plat(R.t_p, R.fy_L, R.bww, R.fuw,
                                                u.g_M2, u.g_M0), R.a_plein)
    L.verdict("plat d'âme")

# =========================================================================
print("\n=== 2. Platine d'about — les 13 cas figés de tests/ref_platine_about.json ===")
from acier.assemblages.poutre_poutre.platine_about import moteur as M_pab  # noqa: E402

with open("tests/ref_platine_about.json", encoding="utf-8") as fh:
    REF_PAB = json.load(fh)
chk("13 cas figés chargés", len(REF_PAB) == 13, str(len(REF_PAB)))
for nom, cas in REF_PAB.items():
    R = M_pab.compute(cas["inp"])
    u = R.u
    L = Lien(nom)
    L.eq_dict(R, GRUGEE, MC.section_grugee(profil_S(R), R.fy_S, u.d_nt, u.d_nb, u.g_M0,
                                           V=u.V_Ed, maintenue=u.lt_ok == "Oui"))
    # pression diamétrale du groupe à deux files : boulons d'extrémité et intérieurs
    p = MC.portance_bout_interieur(R.t_p, R.fu_L, R.f_ub, R.d_b, R.d_0, R.e1_min,
                                   R.e_2, R.p_1, R.p_3, u.g_M2, R.k_trou)
    L.eq_dict(R, [("k1", "k1_L"), ("ab_bout", "abb_L"), ("ab_int", "abi_L"),
                  ("Fb_bout", "Fbb_L"), ("Fb_int", "Fbi_L")], p)
    # âme porteuse : aucun bord libre, tous les boulons sont « intérieurs »
    pP = MC.portance_bout_interieur(R.tw_P, R.fu_P, R.f_ub, R.d_b, R.d_0, None, None,
                                    R.p_1, R.p_3, u.g_M2, R.k_trou)
    L.eq("k1_P", pP["k1"], R.k1_P)
    if R.n_1 > 1:         # à une seule rangée, le moteur écrase abi_P (pas de p1)
        L.eq_dict(R, [("ab_int", "abi_P"), ("Fb_int", "Fbi_P")], pP)
    n_bout = 2 if R.n_1 > 0 else 0
    FRd, branche = MC.groupe_boulons(R.Fv_Rd * R.bLf, R.Fbb_L, R.Fbi_L,
                                     n_bout, R.n_b - n_bout)
    L.eq("FRd_g", FRd, R.FRd_g)
    L.eq("branche_g", branche, R.branche_g)
    # règles de détail
    L.eq("tp_max", MC.tp_ductile(R.d_b, R.f_ub, R.fy_L), R.tp_max)
    L.eq("k_plein", MC.coef_pleine_resistance(R.fy_S, R.fuw, R.bww, u.g_M2, u.g_M0),
         R.k_plein)
    # tronçon en T de la platine : longueurs efficaces, puis les trois modes
    l1, l2 = MC.leff_platine_bords_libres(R.n_1, R.p_1, R.e_1, R.e1bot, R.m_T, R.e_2)
    L.eq("leff_T", l1, R.leff_T)
    L.eq("leff2_T", l2, R.leff2_T)
    if R.tying:
        f, gamma, Ft = R.fu_L, u.g_Mu, RS.Ft_Rd(R.f_ub, R.As_b, u.g_Mu)
    else:
        f, gamma, Ft = R.fy_L, u.g_M0, R.Ft_Rd
    t = MC.troncon_T(R.leff_T, R.leff2_T, R.t_p, R.m_T, R.nT, R.ew_T, f, gamma,
                     R.n_b * Ft)
    L.eq_dict(R, [("FT1", "FT1"), ("FT2", "FT2"), ("FT3", "FT3"),
                  ("FT_Rd", "FT_Rd")], t)
    L.verdict("platine d'about")

# =========================================================================
print("\n=== 3. Doubles cornières — les 31 cas de parité ===")
from acier.assemblages.poutre_poutre.doubles_cornieres import moteur as M_dc  # noqa: E402

with open(os.path.join("acier", "reference", "reference_double_corniere.json"),
          encoding="utf-8") as fh:
    REF_DC = json.load(fh)["cas"]
chk("31 cas de parité chargés", len(REF_DC) == 31, str(len(REF_DC)))
degeneres = []
for cas in REF_DC:
    R = M_dc.compute(cas["inputs"])
    u = R.u
    H = abs(u.H_Ed)
    L = Lien(cas["id"])
    L.eq_dict(R, GRUGEE, MC.section_grugee(
        profil_S(R), R.fy_S, u.d_nt, u.d_nb, u.g_M0, V=u.V_Ed,
        maintenue=u.lt_ok == "Oui", renvoi_non_maintenue=" (SCI P358 Check 7)"))
    # groupes excentrés S (deux plans) et P (sur l'élément porteur)
    gS = MC.groupe_excentre(R.n1_S, R.n2_S, R.p1_S, u.p2_S, u.V_Ed, u.N_Ed, R.M_S)
    L.eq_dict(R, [("n", "n_S"), ("Ip", "Ip_S"), ("xm", "xm_S"), ("ym", "ym_S"),
                  ("Fz", "Fz_S"), ("Fx", "Fx_S"), ("F", "F_S")], gS)
    gP = MC.groupe_excentre(R.n1_P, R.n2_P, R.p1_P, u.p2_P, u.V_Ed / 2, H / 2, R.M_P)
    L.eq_dict(R, [("n", "n_P"), ("Ip", "Ip_P"), ("xm", "xm_P"), ("ym", "ym_P"),
                  ("Fz", "Fz_P"), ("Fx", "Fx_P"), ("F", "F_P")], gP)
    # la table du groupe S garde le SIGNE de NEd dans ce moteur (voir la
    # docstring de mecanique.bolt_table) ; celle du groupe P reçoit |HEd|/2
    L.eq_table("tabS", MC.bolt_table(R.n1_S, R.n2_S, R.p1_S, u.p2_S, u.V_Ed, u.N_Ed,
                                     R.M_S, R.Ip_S, horizontal_signe=True), R.tabS)
    L.eq_table("tabP", MC.bolt_table(R.n1_P, R.n2_P, R.p1_P, u.p2_P, u.V_Ed / 2, H / 2,
                                     R.M_P, R.Ip_P), R.tabP)
    # pression diamétrale : ailes B et A, âme portée, âme porteuse (gauge p3)
    L.eq_bearing(R, "B", t=R.t_C, fu=R.fu_C, fub=R.f_ub, d=R.d_b, d0=R.d_0,
                 e1=mn(R.e1_S, R.e1bot_S), p1=R.p1_S, n1=R.n1_S, e2=R.e2a_S,
                 p2=u.p2_S, n2=R.n2_S, gM2=u.g_M2, ktrou=R.k_trou,
                 Fz=R.Fz_S / 2, Fx=R.Fx_S / 2)
    L.eq_bearing(R, "A", t=R.t_C, fu=R.fu_C, fub=R.f_ub, d=R.d_b, d0=R.d_0,
                 e1=mn(R.e1_P, R.e1bot_P), p1=R.p1_P, n1=R.n1_P, e2=R.e2a_P,
                 p2=u.p2_P, n2=R.n2_P, gM2=u.g_M2, ktrou=R.k_trou,
                 Fz=R.Fz_P, Fx=R.Fx_P)
    L.eq_bearing(R, "S", t=R.tw_S, fu=R.fu_S, fub=R.f_ub, d=R.d_b, d0=R.d_0,
                 e1=mn(R.e1b_S, R.h_e) if u.d_nb > 0 else R.e1b_S,
                 p1=R.p1_S, n1=R.n1_S, e2=R.e2b_S, p2=u.p2_S, n2=R.n2_S,
                 gM2=u.g_M2, ktrou=R.k_trou, Fz=R.Fz_S, Fx=R.Fx_S)
    L.eq_bearing(R, "P", t=R.tw_P, fu=R.fu_P, fub=R.f_ub, d=R.d_b, d0=R.d_0,
                 e1=None, p1=R.p1_P, n1=R.n1_P, e2=None, p2=u.p2_P, n2=R.n2_P,
                 gM2=u.g_M2, ktrou=R.k_trou, Fz=R.Fz_P, Fx=R.Fx_P,
                 exV=1.4 * R.p_3 / R.d_0 - 1.7, exH=R.p_3 / (3 * R.d_0) - 0.25)
    # tronçon en T des ailes A (une seule Σleff pour les modes 1 et 2)
    D = 2 * R.m_T * R.n_T - R.ew_T * (R.m_T + R.n_T)
    t = MC.troncon_T(R.leff_T, R.leff_T, R.t_C, R.m_T, R.n_T, R.ew_T, R.fy_C,
                     u.g_M0, R.SFt_T)
    if D > 0:
        L.eq_dict(R, [("FT1", "FT_1"), ("FT2", "FT_2"), ("FT3", "FT_3")], t)
        L.eq("FT_Rd", t["FT_Rd"], mn(R.FT_1, R.FT_2, R.FT_3))
    else:
        # le moteur des cornières n'a PAS le garde-fou « méthode 1 » que
        # `platine_about` porte et que `mecanique` reprend : sa FT,1 devient
        # négative. Divergence NOMMÉE, sans conséquence sur les cas figés
        # (la vérification tsA y est inactive) — voir le contrôle ci-dessous.
        L.eq_dict(R, [("FT2", "FT_2"), ("FT3", "FT_3")], t)
        degeneres.append((cas["id"], R.FT_1, t["FT1"], R.ck["tsA"].active))
    L.verdict("doubles cornières")
chk("le seul cas figé où 2mn − ew(m + n) dégénère est « bloc_petite_corniere » : le "
    "moteur des cornières y rend une FT,1 NÉGATIVE (garde-fou « méthode 1 » absent de "
    "cette copie, présent dans platine_about et dans mecanique), et sa vérification "
    "tsA y est inactive — aucun nombre publié n'en dépend",
    [c[0] for c in degeneres] == ["bloc_petite_corniere"]
    and all(c[1] <= 0 and c[2] > 0 and not c[3] for c in degeneres), str(degeneres))

# =========================================================================
print("\n=== 4. La brique NEUVE : leff_aile_poteau (Tableau 6.4, aile non raidie) ===")
# Étude §7.3 : n1 = 4 rangées de p1 = 70 mm, m = 42,9 mm, e = 80 mm — aucune
# rangée d'extrémité (le poteau continue au-dessus et en dessous).
n1, p1, m, e = 4, 70.0, 42.9, 80.0
cp_i = n1 * 2 * math.pi * m                       # rangées isolées, 2πm
nc_i = n1 * (4 * m + 1.25 * e)                    # rangées isolées, 4m + 1,25e
cp_g = n1 * 2 * p1                                # en groupe, 2p
nc_g = n1 * p1                                    # en groupe, p
chk("les quatre sommes du §7.3 recalculées : Σcp isolées 1 078,19 · Σnc isolées "
    "1 086,40 · Σcp groupe 560 · Σnc groupe 280 mm",
    all(proche(a, b, 1e-4) for a, b in ((cp_i, 1078.19), (nc_i, 1086.40),
                                        (cp_g, 560.0), (nc_g, 280.0))),
    f"{cp_i:.2f} {nc_i:.2f} {cp_g:.2f} {nc_g:.2f}")
l1, l2 = MC.leff_aile_poteau(n1, p1, m, e)
chk("leff_aile_poteau : Σleff,1 = Σleff,2 = 280 mm (= n1·p1, le groupe gouverne)",
    proche(l1, 280.0, 1e-12) and proche(l2, 280.0, 1e-12), f"{l1} / {l2}")
chk("Σleff,1 = min[min(Σcp ; Σnc) isolées ; min(Σcp ; Σnc) groupe] et "
    "Σleff,2 = min(Σnc isolées ; Σnc groupe) — la composition, pas le résultat",
    proche(l1, min(min(cp_i, nc_i), min(cp_g, nc_g)), 1e-12)
    and proche(l2, min(nc_i, nc_g), 1e-12))
chk("en groupe, nc = p est toujours sous cp = 2p : la branche « en groupe » vaut "
    "n1·p1, et c'est pourquoi Σleff,1 = Σleff,2 au §7.3",
    proche(l1, n1 * p1, 1e-12) and proche(l2, n1 * p1, 1e-12))
# p1 large et m grand : les rangées ISOLÉES gouvernent par 4m + 1,25e
l1b, l2b = MC.leff_aile_poteau(2, 400.0, 60.0, 80.0)
chk("avec p1 = 400 mm et m = 60 mm, les rangées ISOLÉES gouvernent par 4m + 1,25e "
    "(Σleff,1 = Σleff,2 = 2 × 340 = 680 mm ; 2πm = 376,99 > 340)",
    proche(l1b, 680.0, 1e-12) and proche(l2b, 680.0, 1e-12), f"{l1b} / {l2b}")
# p1 large et m modéré : la CIRCULAIRE 2πm des rangées isolées gouverne le mode 1
l1c, l2c = MC.leff_aile_poteau(2, 300.0, 42.9, 80.0)
chk("avec p1 = 300 mm et m = 42,9 mm, la CIRCULAIRE gouverne le mode 1 "
    "(Σleff,1 = 2·2π·42,9 = 539,10 mm) et 4m + 1,25e le mode 2 (543,20 mm)",
    proche(l1c, 2 * 2 * math.pi * 42.9, 1e-12) and proche(l2c, 543.2, 1e-12),
    f"{l1c} / {l2c}")
# une seule rangée : il n'y a pas de groupe, p1 n'a aucun sens
l1d, l2d = MC.leff_aile_poteau(1, 70.0, 42.9, 80.0)
chk("une SEULE rangée : aucun groupe, Σleff = min(2πm ; 4m + 1,25e) = 269,55 mm "
    "(et non n1·p1 = 70 mm)",
    proche(l1d, 2 * math.pi * 42.9, 1e-12) and proche(l2d, 271.6, 1e-12),
    f"{l1d} / {l2d}")

print("\n--- leff_platine_bords_libres et troncon_T : 2 000 géométries tirées, "
      "comparées à la fonction d'origine ---")
# Les 13 cas figés n'atteignent pas toutes les branches du Tableau 6.4 (ni la
# branche à UNE rangée). On appelle donc DIRECTEMENT `platine_about.tronçon_en_T`
# sur un R synthétique — elle ne lit que ces onze grandeurs — et on balaie le
# domaine. Tirage à graine fixe : la suite est reproductible.
import random  # noqa: E402

from acier.js import AttrDict  # noqa: E402

random.seed(20261101)
ecarts_tirage, varies = [], set()
for _ in range(2000):
    n_1 = random.choice([1, 2, 3, 4, 6])
    Rs = AttrDict(p_3=random.choice([60.0, 90.0, 140.0, 220.0, 300.0]),
                  tw_S=random.choice([6.0, 8.6, 11.0, 18.0]),
                  a_w=random.choice([3.0, 5.0, 8.0]),
                  e_2=random.choice([20.0, 40.0, 60.0, 100.0, 200.0]),
                  d_w=random.choice([25.0, 37.0, 50.0]),
                  p_1=random.choice([50.0, 70.0, 100.0, 180.0, 300.0]),
                  n_1=n_1, e_1=random.choice([15.0, 30.0, 40.0, 80.0, 150.0]),
                  e1bot=random.choice([15.0, 30.0, 60.0, 120.0]),
                  t_p=random.choice([8.0, 10.0, 15.0]), n_b=2 * n_1)
    f, gamma, Ft = random.choice([(355.0, 1.0, 141.12), (510.0, 1.1, 160.364)])
    ori = M_pab.tronçon_en_T(Rs, f, gamma, Ft)
    l1, l2 = MC.leff_platine_bords_libres(Rs.n_1, Rs.p_1, Rs.e_1, Rs.e1bot,
                                          ori["m_T"], Rs.e_2)
    t = MC.troncon_T(l1, l2, Rs.t_p, ori["m_T"], ori["nT"], ori["ew_T"], f, gamma,
                     Rs.n_b * Ft)
    varies.add((round(l1, 6), round(l2, 6)))
    for nom, a, b in (("leff_T", l1, ori["leff_T"]), ("leff2_T", l2, ori["leff2_T"]),
                      ("FT1", t["FT1"], ori["FT1"]), ("FT2", t["FT2"], ori["FT2"]),
                      ("FT3", t["FT3"], ori["FT3"]),
                      ("FT_Rd", t["FT_Rd"], ori["FT_Rd"])):
        if not proche(a, b):
            ecarts_tirage.append(f"{nom} : {a!r} ≠ {b!r} pour {dict(Rs)}")
chk(f"2 000 géométries tirées : leff et les trois modes identiques à l'origine "
    f"({len(varies)} couples (Σleff,1 ; Σleff,2) distincts)",
    not ecarts_tirage and len(varies) > 200, " | ".join(ecarts_tirage[:2])[:260])

print("\n--- leff_platine_bords_libres : les branches que les 13 cas figés n'atteignent "
      "pas ---")
# bords libres : les cas figés ne font jamais gouverner le terme
# « 2m + 0,625e + e1 » (ni la branche à UNE rangée) — recalculé à la main ici.
# n1 = 2, p1 = 400, e1 = 30, m = 60, e = 40 :
#   nc,seul = min(4m + 1,25e = 290 ; 2m + 0,625e + e1 = 175) = 175
#   nc,grp  = min(2m + 0,625e + 0,5p = 345 ; e1 + 0,5p = 230) = 230
#   Σnc isolées = 350 ; Σnc groupe = 460 ; Σcp isolées = 496,99 ; Σcp groupe = 920
chk("bords libres, n1 = 2 : le terme 2m + 0,625e + e1 gouverne — Σleff,1 = Σleff,2 "
    "= 2 × 175 = 350 mm",
    all(proche(a, 350.0, 1e-12)
        for a in MC.leff_platine_bords_libres(2, 400.0, 30.0, 30.0, 60.0, 40.0)),
    str(MC.leff_platine_bords_libres(2, 400.0, 30.0, 30.0, 60.0, 40.0)))
# n1 = 1 : aucun groupe. m = 20, e = 100, e1h = e1b = 20 :
#   cp = min(2πm = 125,66 ; πm + e1h + e1b = 102,83) = 102,83
#   nc = min(4m + 1,25e = 205 ; 2m + 0,625e + 0,5(e1h + e1b) = 122,5) = 122,5
un = MC.leff_platine_bords_libres(1, 70.0, 20.0, 20.0, 20.0, 100.0)
chk("bords libres, n1 = 1 : aucun groupe — Σleff,1 = πm + e1h + e1b = 102,832 mm, "
    "Σleff,2 = 2m + 0,625e + 0,5(e1h + e1b) = 122,5 mm",
    proche(un[0], math.pi * 20.0 + 40.0, 1e-12) and proche(un[1], 122.5, 1e-12),
    str(un))

print("\n--- troncon_T sur les deux ailes de poteau du §7.3 (HEB 300 S355, tying) ---")
Ft_u = RS.Ft_Rd(800.0, 245.0, 1.1)                # M20 8.8, γMu = 1,1
SFt = 8 * Ft_u
chk("Ft,Rd,u = 160,364 kN et ΣFt,Rd,u = 1 282,91 kN (M20 8.8, γMu = 1,10)",
    proche(Ft_u, 160.364, 1e-4) and proche(SFt, 1282.91, 1e-4), f"{Ft_u} {SFt}")
# platine d'about, p3 = 140 : m = (140 − 11)/2 − 0,8·27 ; e = (300 − 140)/2
m_a = (140.0 - 11.0) / 2 - 0.8 * 27.0
n_a = min((300.0 - 140.0) / 2, 1.25 * m_a)
TA = MC.troncon_T(280.0, 280.0, 19.0, m_a, n_a, 37.0 / 4, 510.0, 1.1, SFt)
chk("aile du poteau, p3 = 140 : m 42,90 · n 53,625 · Mpl,1 11,7161 kNm · "
    "FT,1 1 296,99 · FT,2 955,485 · FT,3 1 282,91 → FT,Rd 955,485 kN",
    all(proche(a, b, 1e-4) for a, b in ((m_a, 42.90), (n_a, 53.625),
                                        (TA["Mpl1"] / 1e6, 11.7161),
                                        (TA["FT1"], 1296.99), (TA["FT2"], 955.485),
                                        (TA["FT3"], 1282.91), (TA["FT_Rd"], 955.485))),
    f"m {m_a} n {n_a} Mpl {TA['Mpl1'] / 1e6:.4f} "
    f"FT {TA['FT1']:.2f} / {TA['FT2']:.3f} / {TA['FT3']:.2f}")
# doubles cornières, p3 = tw,S + 2·gA = 8,6 + 100 = 108,6
m_b = (108.6 - 11.0) / 2 - 0.8 * 27.0
n_b_ = min((300.0 - 108.6) / 2, 1.25 * m_b)
TB = MC.troncon_T(280.0, 280.0, 19.0, m_b, n_b_, 37.0 / 4, 510.0, 1.1, SFt)
chk("aile du poteau, p3 = 108,6 : m 27,20 · n 34,00 · FT,1 2 314,03 · FT,2 1 095,61 "
    "· FT,3 1 282,91 → FT,Rd 1 095,61 kN (les cornières gouvernent, rapport 1,915)",
    all(proche(a, b, 1e-4) for a, b in ((m_b, 27.20), (n_b_, 34.00),
                                        (TB["FT1"], 2314.03), (TB["FT2"], 1095.61),
                                        (TB["FT_Rd"], 1095.61)))
    and proche(TB["FT_Rd"] / 572.015, 1.915, 1e-3),
    f"m {m_b} n {n_b_} FT {TB['FT1']:.2f} / {TB['FT2']:.2f} / {TB['FT3']:.2f}")
chk("troncon_T retombe sur la MÉTHODE 1 (4·Mpl,1/m) quand 2mn − ew(m + n) dégénère",
    proche(MC.troncon_T(280.0, 280.0, 19.0, 5.0, 5.0, 9.25, 510.0, 1.1, SFt)["FT1"],
           4 * (0.25 * 280.0 * 19.0 * 19.0 * 510.0 / 1.1) / 5.0 / 1000, 1e-12))

# =========================================================================
print("\n=== 5. geometrie_poteau — les sept largeurs, sur les 50 cas figés "
      "des trois modules de la famille ===")
# `geometrie_poteau` était écrite EN LIGNE, mot pour mot, dans les trois
# moteurs poutre–poteau (consolidation du 02/11/2026). Les trois l'appellent
# désormais : comparer la brique au moteur serait circulaire. Ce qui la TIENT,
# c'est une SECONDE ÉCRITURE des sept formules, ici, confrontée au moteur sur
# chacun de ses cas figés — si l'une des deux change, le test rougit et nomme
# le cas et la grandeur. Les cas figés portent déjà ces grandeurs dans leurs
# `scal` : la chaîne complète est donc tenue (brique → moteur → ref_*.json).
#
# L'argument `gh` de la brique est la distance FACE D'APPUI → ABOUT de la
# poutre portée : le jeu `gh` pour un plat d'âme et des cornières,
# l'épaisseur `tp` de la platine pour une platine d'about (écart assumé du
# module `pac_`, docs/assemblages/PLATINE_ABOUT_POTEAU.md).
from acier.assemblages.poutre_poteau.plat_ame_poteau import moteur as M_fpc  # noqa: E402
from acier.assemblages.poutre_poteau.doubles_cornieres_poteau import moteur as M_dcc  # noqa: E402
from acier.assemblages.poutre_poteau.platine_about_poteau import moteur as M_pac  # noqa: E402

# (nom du module, moteur, fichier de régression, nom local du gauge maximal,
#  d'où vient l'about : « gh » ou « tp »)
FAMILLE = [("fpc_", M_fpc, "tests/ref_fpc.json", None, "gh"),
           ("dcc_", M_dcc, "tests/ref_dcc.json", "p3_max_ame", "gh"),
           ("pac_", M_pac, "tests/ref_pac.json", "p3_maxP", "tp")]
n_geo, n_cas_geo, varies_geo = 0, 0, set()
for court, Mx, fichier, nom_p3max, about in FAMILLE:
    with open(fichier, encoding="utf-8") as fh:
        REF_X = json.load(fh)
    for nom, cas in REF_X.items():
        R = Mx.compute(cas["inp"])
        n_cas_geo += 1
        # seconde écriture des sept formules (étude §3.2.2, §4.3, §5.3)
        hP, bP, twP, tfP, rP = R.h_P, R.b_P, R.tw_P, R.tf_P, R.r_P
        j = R.u.j_lat
        gh = R.t_p if about == "tp" else R.u.g_h
        att = {"libre_droit": hP - 2 * tfP - 2 * rP,
               "demi_droit": 0.5 * (hP - 2 * tfP) - rP,
               "libre_ailes": hP - 2 * tfP - 2 * j,
               "p3_congeP": twP + 2 * rP + R.d_w,
               "gh_b1": 0.5 * (bP - twP) + j,
               "rec_min": max(0.0, 0.5 * (bP - twP) + j - gh),
               "passe_S": bool(R.b_S <= hP - 2 * tfP - 2 * j + 1e-9)}
        if nom_p3max:
            att[nom_p3max] = 2 * (0.5 * (hP - 2 * tfP) - rP - R.d_w / 2)
        for cle, v in att.items():
            if cle not in R:
                continue                      # fpc_ ne nomme pas demi_droit ni p3_maxP
            varies_geo.add((court, cle))
            obtenu = R[cle]
            if isinstance(v, bool):
                ok = bool(obtenu) == v
            else:
                ok = proche(obtenu, v, 1e-12)
            if not ok:
                n_geo += 1
                print("   ", court, nom, cle, "attendu", v, "obtenu", obtenu)
chk(f"les sept largeurs du poteau coïncident à 1e-12 sur les {n_cas_geo} cas "
    f"figés de la famille ({len(varies_geo)} couples module × grandeur)",
    n_geo == 0 and n_cas_geo >= 50 and len(varies_geo) >= 20,
    f"{n_geo} écart(s), {n_cas_geo} cas, {len(varies_geo)} couples")
# les cas figés ne font pas varier tout le domaine : trois géométries à la main
g1 = MC.geometrie_poteau(dict(h=300.0, b=300.0, tw=11.0, tf=19.0, r=27.0),
                         5.0, 10.0, 180.0, 37.0)
chk("HEB 300, j = 5, gh = 10, bS = 180, dw = 37 : libre droit 208 · demi 104 · "
    "libre entre ailes 252 · p3 congé 102 · p3 max 171 · gh b1 149,5 · "
    "recoupe 139,5 · la semelle passe",
    all(proche(g1[k], v, 1e-12) for k, v in (("libre_droit", 208.0), ("demi_droit", 104.0),
                                             ("libre_ailes", 252.0), ("p3_congeP", 102.0),
                                             ("p3_maxP", 171.0), ("gh_b1", 149.5),
                                             ("rec_min", 139.5)))
    and g1["passe_S"] is True, str(g1))
g2 = MC.geometrie_poteau(dict(h=300.0, b=300.0, tw=11.0, tf=19.0, r=27.0),
                         5.0, 200.0, 300.0, 37.0)
chk("la même, gh = 200 > gh,b1 : recoupe nulle ; bS = 300 > 252 : la semelle ne passe pas",
    proche(g2["rec_min"], 0.0, 1e-12) and g2["passe_S"] is False, str(g2))
g3 = MC.geometrie_poteau(dict(h=400.0, b=180.0, tw=8.6, tf=13.5, r=21.0), 5.0, 0.0, 0.0, 0.0)
chk("IPE 400 porteur, arguments facultatifs omis : libre droit 331 · demi 165,5 "
    "(le prédimensionnement les lit ainsi)",
    proche(g3["libre_droit"], 331.0, 1e-12) and proche(g3["demi_droit"], 165.5, 1e-12),
    str(g3))

# =========================================================================
print("\n=== 6. Pureté de mecanique.py ===")
import inspect  # noqa: E402

SRC = inspect.getsource(MC)
imports = sorted({l.strip() for l in SRC.splitlines()
                  if l.startswith("import ") or l.startswith("from ")})
chk("aucun import de Streamlit, d'interface, de dessin ni de base de profilés",
    imports == ["from acier.js import INF, dv, hyp, mn, mx",
                "from acier.resultats import EffortBoulon", "import math"], str(imports))
chk("ni « streamlit » ni « st. » dans le fichier",
    "streamlit" not in SRC and "\nimport st" not in SRC)
chk("les 15 briques de l'étude §10.2 sont présentes",
    all(callable(getattr(MC, f, None)) for f in (
        "bearing", "interaction_pd", "portance_bout_interieur", "groupe_boulons",
        "groupe_excentre", "bolt_table", "section_grugee", "lam_LT", "pb_bs5950",
        "tp_ductile", "coef_pleine_resistance", "a_pleine_resistance_plat",
        "troncon_T", "leff_aile_poteau", "leff_platine_bords_libres")),
    str([f for f in ("bearing", "interaction_pd", "portance_bout_interieur",
                     "groupe_boulons", "groupe_excentre", "bolt_table",
                     "section_grugee", "lam_LT", "pb_bs5950", "tp_ductile",
                     "coef_pleine_resistance", "a_pleine_resistance_plat",
                     "troncon_T", "leff_aile_poteau", "leff_platine_bords_libres")
          if not callable(getattr(MC, f, None))]))
BRIQUES = [(n, f) for n, f in inspect.getmembers(MC, inspect.isfunction)
           if f.__module__ == MC.__name__ and not n.startswith("_")]
sans_ref = [n for n, f in BRIQUES if "Référence" not in (f.__doc__ or "")]
chk(f"chacune des {len(BRIQUES)} briques porte sa référence normative dans sa "
    "docstring", not sans_ref, str(sans_ref))
sans_src = [n for n, f in BRIQUES if "Origine de la copie" not in (f.__doc__ or "")]
chk("chaque brique nomme le module d'origine de la copie", not sans_src, str(sans_src))

chk(f"les 56 cas figés sont liés ({len(REF_FPL)} + {len(REF_PAB)} + {len(REF_DC)})",
    len(REF_FPL) + len(REF_PAB) + len(REF_DC) == 56)
chk(f"au moins 2 900 grandeurs comparées sur les 56 cas ({TOTAL})", TOTAL >= 2900,
    str(TOTAL))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
