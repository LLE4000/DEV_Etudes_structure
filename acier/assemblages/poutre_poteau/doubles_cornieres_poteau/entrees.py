# -*- coding: utf-8 -*-
"""Données d'entrée de l'assemblage poutre–poteau par doubles cornières d'âme.

- ``defaults()``          : les 88 clés et leur valeur par défaut ;
- ``NUM``                 : les clés normalisées en nombre par le moteur ;
- ``GROUPES``             : les groupes du formulaire (clé, libellé long, type,
                            options, unité, aide, condition d'affichage) — une
                            VUE : les libellés et aides de l'écran sont ceux
                            de ``ecran_saisie.LIBELLE`` / ``AIDES`` (charte
                            du 31/10/2026) ;
- ``PILOTEES_PAR_PREDIM`` : les clés remplacées par la proposition en mode
                            prédimensionnement (marquées ◆) ;
- ``UNITE``               : unité de chaque clé (``ecran_saisie.LIBELLE`` la
                            lit).

Les 88 clés sont celles du grand frère poutre–poutre ``doubles_cornieres``
(84), MOINS ``d_top`` (un poteau n'a pas de dessus), PLUS les cinq clés de la
famille poutre–poteau : ``appui`` (la clé centrale), ``nat_N``, ``g_Mu``,
``rec_S`` et ``j_lat`` — `docs/assemblages/POUTRE_POTEAU.md` §10.6.

Coefficients partiels : défauts de l'ANNEXE NATIONALE BELGE (NBN EN
1993-1-1 ANB : γM0 = 1,00 ; γM2 = 1,25). γMu = 1,1 (tying, situation
accidentelle) est la valeur des guides (MSB P5 §2.3, SCI P358) :
l'EN 1993-1-8 ne la fixe pas et l'ANB belge n'a pas été consultée — point
P-9 de l'étude. Texte NBN payant non consulté ici, valeurs modifiables.
"""
from acier.bibliotheques import DB, PERSO, ORI_P, ORI_S, noms

MODE_VERIF = "VÉRIFICATION"
MODE_PREDIM = "PRÉDIMENSIONNEMENT"
BOULONNEE, SOUDEE = "Boulonnée", "Soudée"

# Cas d'appui — LA clé centrale du module (POUTRE_POTEAU.md §1.2) : sur
# l'AILE du poteau (cas a : l'attache vit dans le plan de l'âme du poteau)
# ou dans son ÂME (cas b : l'attache est perpendiculaire à l'âme).
APPUI_AILE = "Aile du poteau"
APPUI_AME = "Âme du poteau"

# Nature de l'effort normal de traction : ELU fondamental (fy/γM0, γM2), ou
# force de chaînage « tying » en situation accidentelle (MSB P5 §2.3 :
# fu/γMu, vérifiée SEULE, sans VEd) — mécanisme repris de ``platine_about``.
NAT_ELU = "ELU fondamental"
NAT_TYING = "Tying (accidentel)"

# Annotation de FABRICATION, hors moteur et hors CLES : composition de la
# boulonnerie par boulon (même mécanisme que les modules existants).
VISSERIE_DEFAUT = "1 rondelle + 1 écrou"


def defaults():
    """Les 88 entrées et leur valeur par défaut — le cas BM2 de l'étude
    (§11.4) : IPE 400 S355 sur HEB 300 S355, 2 L100×100×10 S355, hc = 290,
    4 + 4 M20 8.8, gA = 50, appui sur l'aile, VEd = 200 kN."""
    return {
        "id_projet": "", "id_rep": "", "id_red": "", "id_date": "",
        "mode_calc": MODE_VERIF, "appui": APPUI_AILE,
        "fix_P": BOULONNEE, "fix_S": BOULONNEE,
        "prof_P": "HEB 300", "hP_u": None, "bP_u": None, "twP_u": None,
        "tfP_u": None, "rP_u": None, "nu_P": "S355",
        "prof_S": "IPE 400", "hS_u": None, "bS_u": None, "twS_u": None,
        "tfS_u": None, "rS_u": None, "nu_S": "S355",
        "d_nt": 0, "d_nb": 0, "l_n": 0, "r_n": 10, "lt_ok": "Oui",
        "g_h": 10, "rec_S": 0, "j_lat": 5,
        "corn_u": "L100x100x10", "k1_u": None, "k2_u": None, "kt_u": None,
        "kr_u": None, "orient": ORI_P, "LC_u": 290, "z_C": 40, "nu_C": "S355",
        "boulon_u": "M20", "classe": "8.8", "trou": "Normal", "d0_u": None,
        "filet": "Oui", "cat": "A", "mu_s": 0.3, "k_s": 1, "k_ser": 1,
        "n1S_u": 4, "n2S_u": 1, "p1S_u": 70, "p2_S": 60, "e1S_u": 40,
        "e2b_u": 40, "n1P_u": 4, "n2P_u": 1, "p1P_u": 70, "p2_P": 60,
        "e1P_u": 40, "gA_u": 50,
        "a_P": 5, "lh_P": 0, "a_S": 5, "lh_S": 40,
        "V_Ed": 200, "N_Ed": 0, "H_Ed": 0, "M_Ed": 0, "nat_N": NAT_ELU,
        "g_M0": 1, "g_M2n": 1.25, "g_M2": 1.25, "g_M3": 1.25, "g_M3s": 1.1,
        "g_Mu": 1.1,
        "opt_exc": "Non", "k_rot": 0.8, "opt_blf": "Non", "expo": "Non",
        "fy_u": None, "fu_u": None, "bw_u": None,
        "eta_c": 0.9, "k_e1": 1.6, "k_p1": 2.7, "k_e2": 1.8,
        "pd_dmin": "M16", "pd_dmax": "M24",
    }


CLES = tuple(defaults().keys())          # 88 clés

# Clés passées par N() en tête de compute()
NUM = ["d_nt", "d_nb", "l_n", "r_n", "g_h", "rec_S", "j_lat", "LC_u", "z_C",
       "mu_s", "k_s", "k_ser", "n1S_u", "n2S_u", "p1S_u", "p2_S", "e1S_u",
       "e2b_u", "n1P_u", "n2P_u", "p1P_u", "p2_P", "e1P_u", "gA_u", "a_P",
       "lh_P", "a_S", "lh_S", "V_Ed", "N_Ed", "H_Ed", "M_Ed", "g_M0", "g_M2n",
       "g_M2", "g_M3", "g_M3s", "g_Mu", "k_rot", "eta_c", "k_e1", "k_p1",
       "k_e2"]

# Champs remplacés par la proposition en mode prédimensionnement (◆)
PILOTEES_PAR_PREDIM = ("corn_u", "LC_u", "boulon_u", "n1S_u", "n2S_u", "p1S_u",
                       "e1S_u", "e2b_u", "n1P_u", "n2P_u", "p1P_u", "e1P_u",
                       "gA_u")

PROFILS = [PERSO] + noms(DB["profils"])
ACIERS = noms(DB["aciers"]) + [PERSO]
CORNIERES = [PERSO] + noms(DB["cornieres"])
BOULONS = noms(DB["boulons"])
CLASSES = noms(DB["classes"])
OUI_NON = ["Oui", "Non"]
FIXATIONS = [BOULONNEE, SOUDEE]


# --------------------------------------------------------------- conditions
def boulonne_S(u):
    return u.get("fix_S") == BOULONNEE


def boulonne_P(u):
    return u.get("fix_P") == BOULONNEE


def perso_P(u):
    return u.get("prof_P") == PERSO


def perso_S(u):
    return u.get("prof_S") == PERSO


def perso_C(u):
    return u.get("corn_u") == PERSO


def _cat_non_A(u):
    return u.get("cat") != "A"


def _cat_B(u):
    return u.get("cat") == "B"


def _deux_files_S(u):
    return str(u.get("n2S_u")) == "2"


def _deux_files_P(u):
    return str(u.get("n2P_u")) == "2"


def _soude_un_cote(u):
    return not boulonne_S(u) or not boulonne_P(u)


def _soude_P(u):
    return not boulonne_P(u)


def _soude_S(u):
    return not boulonne_S(u)


def _acier_perso(u):
    return PERSO in (u.get("nu_P"), u.get("nu_S"), u.get("nu_C"))


def mode_predim(u):
    return u.get("mode_calc") != MODE_VERIF


def sur_aile(u):
    """Cas (a) : l'attache est sur l'AILE du poteau, dans le plan de son âme."""
    return u.get("appui") != APPUI_AME


def sur_ame(u):
    """Cas (b) : l'attache est dans l'ÂME du poteau, perpendiculaire à elle."""
    return u.get("appui") == APPUI_AME


def tying(u):
    return u.get("nat_N") == NAT_TYING


def traction(u):
    """NEd > 0 : la nature de la traction (ELU / tying) n'a de sens qu'alors —
    le moteur fait NEd = max(0, N_Ed), une compression n'a pas de nature."""
    try:
        return float(str(u.get("N_Ed")).strip().replace(",", ".")) > 0
    except (TypeError, ValueError):
        return False


def _tying_N(u):
    """γMu ne sert qu'au tying effectif : nature Tying ET NEd > 0."""
    return tying(u) and traction(u)


def corniere_inegale(u):
    """L'orientation (grande aile côté poteau ou côté poutre portée) n'a de
    sens que pour une cornière à ailes inégales — ou Personnalisé ; en
    prédimensionnement la cornière proposée est à ailes égales, le champ
    n'est pas rendu."""
    if mode_predim(u):
        return False
    if u.get("corn_u") == PERSO:
        return True
    c = next((x for x in DB["cornieres"] if x["n"] == u.get("corn_u")), None)
    return c is not None and c["a1"] != c["a2"]


def _dims(X, cond):
    return [dict(k=f"{d}{X}_u", l=f"{d} (personnalisé)", t="n", un="mm", si=cond)
            for d in ("h", "b", "tw", "tf", "r")]


def _c(k, l, t="n", o=None, un=None, h=None, si=None):
    """Un champ : clé, libellé, type (n nombre / s choix / x texte), options,
    unité, aide, condition d'affichage."""
    return dict(k=k, l=l, t=t, o=o, un=un, h=h, si=si)


# ------------------------------------------------------------------ groupes
GROUPES = [
    dict(t="Identification",
         f=[_c("id_projet", "Projet", "x"), _c("id_rep", "Repère de l'assemblage", "x"),
            _c("id_red", "Rédacteur", "x"), _c("id_date", "Date", "x")]),
    dict(t="Mode de calcul et cas d'appui",
         f=[_c("mode_calc", "Mode de calcul", "s", [MODE_VERIF, MODE_PREDIM],
               h="Prédimensionnement : boulons, rangées et cornière sont proposés (onglet Prédim) puis vérifiés en détail. Les champs marqués ◆ sont alors remplacés."),
            _c("appui", "Cas d'appui sur le poteau", "s", [APPUI_AILE, APPUI_AME],
               h="Aile : les ailes A portent sur la face extérieure d'une aile, de part et d'autre de l'âme du poteau. "
                 "Âme : les ailes A portent sur l'âme, entre les ailes.")]),
    dict(t="Poteau porteur",
         f=[_c("prof_P", "Profilé", "s", PROFILS)] + _dims("P", perso_P) +
           [_c("nu_P", "Nuance d'acier", "s", ACIERS)]),
    dict(t="Poutre portée et grugeage",
         f=[_c("prof_S", "Profilé", "s", PROFILS)] + _dims("S", perso_S) +
           [_c("nu_S", "Nuance d'acier", "s", ACIERS),
            _c("d_nt", "Profondeur du grugeage supérieur dc,sup", un="mm",
               h="0 = semelle supérieure non grugée. Un poteau n'oblige pas à gruger : le grugeage sert à passer une platine de tête, un gousset ou un raidisseur."),
            _c("d_nb", "Profondeur du grugeage inférieur dc,inf", un="mm"),
            _c("l_n", "Longueur du grugeage c depuis l'about", un="mm"),
            _c("r_n", "Rayon du grugeage (information d'exécution)", un="mm", h="Non utilisé dans les résistances."),
            _c("g_h", "Jeu gh entre la face d'appui du poteau et l'about de la poutre", un="mm"),
            _c("rec_S", "Recoupe des semelles en largeur, depuis l'about", un="mm", si=sur_ame,
               h="Cas (b) : longueur recoupée pour que la poutre entre entre les ailes du poteau. 0 = aucune."),
            _c("j_lat", "Jeu latéral entre la semelle de la poutre et l'aile du poteau", un="mm", si=sur_ame,
               h="Cas (b) : jeu de montage en largeur, entre la semelle de la poutre portée et l'aile du poteau."),
            _c("lt_ok", "Poutre portée maintenue au déversement", "s", OUI_NON,
               h="Condition d'application de la règle de stabilité locale du grugeage (MSB Part 5 §4.2.5).")]),
    dict(t="Cornières (2 cornières identiques)",
         f=[_c("corn_u", "Cornière", "s", CORNIERES,
               h="Dimensions usuelles EN 10056-1, à contrôler avec le catalogue du fournisseur."),
            _c("k1_u", "Grande aile", un="mm", si=perso_C), _c("k2_u", "Petite aile", un="mm", si=perso_C),
            _c("kt_u", "Épaisseur", un="mm", si=perso_C), _c("kr_u", "Rayon de congé", un="mm", si=perso_C),
            _c("orient", "Orientation (cornière inégale)", "s", [ORI_P, ORI_S], si=corniere_inegale),
            _c("LC_u", "Hauteur des cornières hc", un="mm",
               h="Recommandation des guides : hc ≥ 0,6·h de la portée (maintien en torsion)."),
            _c("z_C", "Dessus de la poutre portée → dessus des cornières zc", un="mm"),
            _c("nu_C", "Nuance d'acier", "s", ACIERS)]),
    dict(t="Boulons",
         f=[_c("boulon_u", "Diamètre", "s", BOULONS), _c("classe", "Classe", "s", CLASSES),
            _c("trou", "Type de trou", "s", ["Normal", "Surdimensionné"],
               h="Surdimensionné : Fb,Rd × 0,8 (EN 1993-1-8 Tableau 3.4) et d0 à saisir."),
            _c("d0_u", "Diamètre de trou d0 imposé (vide = trou normal)", un="mm"),
            _c("filet", "Plan de cisaillement dans la partie filetée", "s", OUI_NON),
            _c("cat", "Catégorie d'assemblage en cisaillement", "s", ["A", "B", "C"],
               h="A : pression diamétrale. B : sans glissement à l'ELS. C : sans glissement à l'ELU (boulons précontraints 8.8 ou 10.9)."),
            _c("mu_s", "Coefficient de frottement μ", un="-", si=_cat_non_A),
            _c("k_s", "Coefficient ks (Tableau 3.6)", un="-", si=_cat_non_A),
            _c("k_ser", "Rapport efforts ELS / ELU", un="-", si=_cat_B, h="1,0 = hypothèse sécuritaire.")]),
    dict(t="Groupe P – ailes A sur le poteau (1 plan, par cornière)",
         f=[_c("fix_P", "Fixation des ailes A sur le poteau", "s", FIXATIONS),
            _c("n1P_u", "Nombre de rangées n1", un="-", si=boulonne_P),
            _c("n2P_u", "Nombre de files verticales n2 par cornière", "s", ["1", "2"], si=boulonne_P),
            _c("p1P_u", "Entraxe vertical p1", un="mm", si=boulonne_P),
            _c("p2_P", "Entraxe horizontal p2", un="mm", si=_deux_files_P),
            _c("e1P_u", "Pince e1 : dessus de cornière → 1re rangée", un="mm", si=boulonne_P),
            _c("gA_u", "Trusquinage gA : talon de cornière → 1re file", un="mm", si=boulonne_P)]),
    dict(t="Groupe S – boulons dans l'âme de la poutre portée (2 plans)",
         f=[_c("fix_S", "Fixation des ailes B sur la poutre portée", "s", FIXATIONS,
               h="Soudée des deux côtés : assemblage peu déformable, à réserver à des cas particuliers."),
            _c("n1S_u", "Nombre de rangées n1", un="-", si=boulonne_S),
            _c("n2S_u", "Nombre de files verticales n2", "s", ["1", "2"], si=boulonne_S),
            _c("p1S_u", "Entraxe vertical p1", un="mm", si=boulonne_S),
            _c("p2_S", "Entraxe horizontal p2", un="mm", si=_deux_files_S),
            _c("e1S_u", "Pince e1 : dessus de cornière → 1re rangée", un="mm", si=boulonne_S),
            _c("e2b_u", "Distance e2,b : about de poutre → 1re file", un="mm", si=boulonne_S)]),
    dict(t="Soudures (cordons d'angle)", si=_soude_un_cote,
         f=[_c("a_P", "Gorge a – ailes A sur le poteau", un="mm", si=_soude_P,
               h="Cordon vertical en bout d'aile sur toute la hauteur hc."),
            _c("lh_P", "Retours horizontaux haut et bas – ailes A", un="mm", si=_soude_P),
            _c("a_S", "Gorge a – ailes B sur l'âme portée", un="mm", si=_soude_S),
            _c("lh_S", "Retours horizontaux haut et bas – ailes B", un="mm", si=_soude_S)]),
    dict(t="Efforts de calcul ELU",
         f=[_c("V_Ed", "VEd – effort tranchant vertical", un="kN"),
            _c("N_Ed", "NEd – effort normal dans la poutre portée (traction +)", un="kN"),
            _c("H_Ed", "HEd – effort horizontal perpendiculaire à l'âme portée", un="kN"),
            _c("M_Ed", "MEd – moment éventuel à la face d'appui", un="kNm",
               h="Assemblage articulé : normalement 0. Pris en valeur absolue comme moment parasite."),
            _c("nat_N", "Nature de NEd", "s", [NAT_ELU, NAT_TYING], si=traction,
               h="ELU fondamental : NEd est combiné avec VEd, résistances fy/γM0 et γM2. Tying : force de "
                 "chaînage accidentelle, vérifiée SEULE, résistances ultimes fu/γMu (MSB Part 5 §2.3).")]),
    dict(t="Coefficients partiels",
         f=[_c("g_M0", "γM0", un="-"),
            _c("g_M2n", "γM2 – sections nettes, rupture de bloc", un="-"),
            _c("g_M2", "γM2 – boulons, pression diamétrale, soudures", un="-"),
            _c("g_M3", "γM3 – glissement ELU", un="-"), _c("g_M3s", "γM3,ser – glissement ELS", un="-"),
            _c("g_Mu", "γMu – tying (situation accidentelle)", un="-", si=_tying_N,
               h="1,1 : valeur des guides (MSB P5 §2.3, SCI P358) ; l'EN 1993-1-8 ne la fixe pas — à confirmer pour la Belgique.")]),
    dict(t="Options du modèle mécanique",
         f=[_c("opt_exc", "Excentricité gA reprise par chaque groupe P dans son plan", "s", OUI_NON,
               h="Non : pratique ECCS n°126 / MSB Part 5. Oui : variante conservatrice, M = (VEd/2)·eP par cornière."),
            _c("k_rot", "Facteur sur Fv,Rd des boulons P", un="-", h="0,8 : MSB Part 5 §4.2.1.2 / SCI P358. 1,0 pour désactiver."),
            _c("opt_blf", "Appliquer βLf des assemblages longs (Lj > 15d)", "s", OUI_NON,
               h="EN 1993-1-8 §3.8. MSB Part 5 ne l'applique pas aux cornières d'âme ; Oui = variante conservatrice."),
            _c("expo", "Assemblage exposé aux intempéries", "s", OUI_NON,
               h="Active les pinces maximales 4t + 40 mm (Tableau 3.3).")]),
    dict(t="Acier personnalisé", si=_acier_perso,
         f=[_c("fy_u", "fy", un="MPa"), _c("fu_u", "fu", un="MPa"), _c("bw_u", "βw", un="-")]),
    dict(t="Règles du prédimensionnement", si=mode_predim,
         f=[_c("eta_c", "Taux d'utilisation cible", un="-"),
            _c("k_e1", "Pince e1 = k × d0 (minimum normatif 1,2)", un="-"),
            _c("k_p1", "Entraxe p1 = k × d0 (minimum normatif 2,2)", un="-"),
            _c("k_e2", "Pince e2 = k × d0 (minimum normatif 1,2)", un="-"),
            _c("pd_dmin", "Plus petit diamètre admis", "s", BOULONS),
            _c("pd_dmax", "Plus grand diamètre admis", "s", BOULONS)]),
]

UNITE = {}
CHAMPS = {}
for _g in GROUPES:
    for _f in _g["f"]:
        UNITE[_f["k"]] = " " + _f["un"] if _f.get("un") and _f["un"] != "-" else ""
        CHAMPS[_f["k"]] = _f


def groupes_visibles(u):
    """Les groupes et champs affichables pour l'état ``u``."""
    out = []
    for g in GROUPES:
        if g.get("si") and not g["si"](u):
            continue
        champs = [f for f in g["f"] if not f.get("si") or f["si"](u)]
        out.append((g, champs))
    return out


def champ_visible(u, k):
    """Le champ ``k`` est-il affiché dans la configuration ``u`` ?"""
    return any(f["k"] == k for _, fs in groupes_visibles(u) for f in fs)


def charger_json(objet):
    """Fusion d'un fichier enregistré sur les défauts, clé par clé (les clés
    inconnues sont ignorées) : un fichier sans ``appui`` reçoit donc
    APPUI_AILE, et un fichier sans ``nat_N`` reçoit NAT_ELU."""
    d = defaults()
    if not isinstance(objet, dict):
        raise ValueError("Structure JSON inattendue")
    for k in d:
        if k in objet:
            d[k] = objet[k]
    return d
