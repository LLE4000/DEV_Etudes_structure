# -*- coding: utf-8 -*-
"""Données d'entrée de l'assemblage à doubles cornières d'âme.

- ``defaults()``          : les 84 clés et leur valeur par défaut (noms du
                            moteur de référence, conservés pour la parité) ;
- ``NUM``                 : les 41 clés normalisées en nombre par le moteur ;
- ``GROUPES``             : les 14 groupes du formulaire, avec pour chaque
                            champ sa clé, son libellé, son type, ses options,
                            son unité, son aide et sa condition d'affichage —
                            libellés et aides repris mot pour mot ;
- ``PILOTEES_PAR_PREDIM`` : les 13 clés remplacées par la proposition en
                            mode prédimensionnement (marquées ◆) ;
- ``COURT`` / ``UNITE``   : libellé court et unité de chaque clé (chips
                            d'alerte, panneau des cotes).

Les conditions d'affichage sont des fonctions du dictionnaire des saisies
``u`` : elles ne dépendent d'aucune interface.
"""
from acier.bibliotheques import DB, PERSO, ORI_P, ORI_S, noms

MODE_VERIF = "VÉRIFICATION"
MODE_PREDIM = "PRÉDIMENSIONNEMENT"
BOULONNEE, SOUDEE = "Boulonnée", "Soudée"


def defaults():
    """Les 84 entrées et leur valeur par défaut (``defaults()`` du moteur JS)."""
    return {
        "id_projet": "", "id_rep": "", "id_red": "", "id_date": "",
        "mode_calc": MODE_VERIF, "fix_P": BOULONNEE, "fix_S": BOULONNEE,
        "prof_P": "HEA 400", "hP_u": None, "bP_u": None, "twP_u": None,
        "tfP_u": None, "rP_u": None, "nu_P": "S355",
        "prof_S": "HEA 300", "hS_u": None, "bS_u": None, "twS_u": None,
        "tfS_u": None, "rS_u": None, "nu_S": "S355",
        "d_top": 0, "d_nt": 50, "d_nb": 0, "l_n": 150, "r_n": 10, "g_h": 10,
        "lt_ok": "Oui",
        "corn_u": "L100x100x10", "k1_u": None, "k2_u": None, "kt_u": None,
        "kr_u": None, "orient": ORI_P, "LC_u": 190, "z_C": 50, "nu_C": "S355",
        "boulon_u": "M20", "classe": "8.8", "trou": "Normal", "d0_u": None,
        "filet": "Oui", "cat": "A", "mu_s": 0.3, "k_s": 1, "k_ser": 1,
        "n1S_u": 3, "n2S_u": 1, "p1S_u": 60, "p2_S": 60, "e1S_u": 35,
        "e2b_u": 40, "n1P_u": 3, "n2P_u": 1, "p1P_u": 60, "p2_P": 60,
        "e1P_u": 35, "gA_u": 55,
        "a_P": 5, "lh_P": 0, "a_S": 5, "lh_S": 40,
        "V_Ed": 125, "N_Ed": 0, "H_Ed": 0, "M_Ed": 0,
        "g_M0": 1, "g_M2n": 1.25, "g_M2": 1.25, "g_M3": 1.25, "g_M3s": 1.1,
        "opt_exc": "Non", "k_rot": 0.8, "opt_blf": "Non", "expo": "Non",
        "fy_u": None, "fu_u": None, "bw_u": None,
        "eta_c": 0.9, "k_e1": 1.6, "k_p1": 2.7, "k_e2": 1.8,
        "pd_dmin": "M16", "pd_dmax": "M24",
    }


CLES = tuple(defaults().keys())          # 84 clés, dans l'ordre du moteur

# Clés passées par N() en tête de compute() (nombre, 0 si vide)
NUM = ["d_top", "d_nt", "d_nb", "l_n", "r_n", "g_h", "LC_u", "z_C", "mu_s",
       "k_s", "k_ser", "n1S_u", "n2S_u", "p1S_u", "p2_S", "e1S_u", "e2b_u",
       "n1P_u", "n2P_u", "p1P_u", "p2_P", "e1P_u", "gA_u", "a_P", "lh_P",
       "a_S", "lh_S", "V_Ed", "N_Ed", "H_Ed", "M_Ed", "g_M0", "g_M2n",
       "g_M2", "g_M3", "g_M3s", "k_rot", "eta_c", "k_e1", "k_p1", "k_e2"]

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


def _dims(X, cond):
    """Les cinq dimensions d'un profilé personnalisé."""
    return [dict(k=f"{d}{X}_u", l=f"{d} (personnalisé)", t="n", un="mm", si=cond)
            for d in ("h", "b", "tw", "tf", "r")]


def _c(k, l, t="n", o=None, un=None, h=None, si=None):
    """Un champ : clé, libellé, type (n nombre / s choix / x texte), options,
    unité, aide, condition d'affichage."""
    return dict(k=k, l=l, t=t, o=o, un=un, h=h, si=si)


# ------------------------------------------------------------------ groupes
# Ordre, titres, libellés, aides et notes : ceux du formulaire de référence.
GROUPES = [
    dict(t="Identification",
         note="Reprise dans le rapport et l'export texte. Un champ vide n'est pas imprimé.",
         f=[_c("id_projet", "Projet", "x"), _c("id_rep", "Repère de l'assemblage", "x"),
            _c("id_red", "Rédacteur", "x"), _c("id_date", "Date", "x")]),
    dict(t="Mode de calcul et fixations",
         f=[_c("mode_calc", "Mode de calcul", "s", [MODE_VERIF, MODE_PREDIM],
               h="Prédimensionnement : boulons, rangées et cornière sont proposés (onglet Prédim) puis vérifiés en détail. Les champs marqués ◆ sont alors remplacés."),
            _c("fix_P", "Cornières sur la poutre principale (ailes A)", "s", FIXATIONS),
            _c("fix_S", "Cornières sur la poutre secondaire (ailes B)", "s", FIXATIONS,
               h="Soudée des deux côtés : assemblage peu déformable, à réserver à des cas particuliers.")]),
    dict(t="Poutre principale (porteuse)",
         f=[_c("prof_P", "Profilé", "s", PROFILS)] + _dims("P", perso_P) +
           [_c("nu_P", "Nuance d'acier", "s", ACIERS,
               h="S355 : fu = 510 MPa (EN 1993-1-1:2005 Tab. 3.1). Ligne « fu 490 » : valeur retenue par d'autres références (annexe nationale française, EN 1993-1-1:2022). À caler sur ton ANB.")]),
    dict(t="Poutre secondaire (portée) et grugeage",
         f=[_c("prof_S", "Profilé", "s", PROFILS)] + _dims("S", perso_S) +
           [_c("nu_S", "Nuance d'acier", "s", ACIERS),
            _c("d_top", "Décalage : dessus de la secondaire sous le dessus de la principale", un="mm",
               h="0 = dessus des semelles au même niveau."),
            _c("d_nt", "Profondeur du grugeage supérieur dnt", un="mm", h="0 = semelle supérieure non grugée."),
            _c("d_nb", "Profondeur du grugeage inférieur dnb", un="mm"),
            _c("l_n", "Longueur du grugeage ln depuis l'about", un="mm"),
            _c("r_n", "Rayon du grugeage (information d'exécution)", un="mm", h="Non utilisé dans les résistances."),
            _c("g_h", "Jeu gh entre l'âme porteuse et l'about de la secondaire", un="mm"),
            _c("lt_ok", "Poutre secondaire maintenue au déversement", "s", OUI_NON,
               h="Condition d'application de la règle de stabilité locale du grugeage (MSB Part 5 §4.2.5).")]),
    dict(t="Cornières (2 cornières identiques)",
         f=[_c("corn_u", "Cornière", "s", CORNIERES,
               h="Dimensions usuelles EN 10056-1, à contrôler avec le catalogue du fournisseur."),
            _c("k1_u", "Grande aile", un="mm", si=perso_C), _c("k2_u", "Petite aile", un="mm", si=perso_C),
            _c("kt_u", "Épaisseur", un="mm", si=perso_C), _c("kr_u", "Rayon de congé", un="mm", si=perso_C),
            _c("orient", "Orientation (cornière inégale)", "s", [ORI_P, ORI_S]),
            _c("LC_u", "Longueur des cornières Lc (hauteur)", un="mm"),
            _c("z_C", "Dessus de la poutre secondaire → dessus des cornières zc", un="mm"),
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
    dict(t="Groupe S – boulons dans l'âme de la poutre secondaire (2 plans)", si=boulonne_S,
         f=[_c("n1S_u", "Nombre de rangées n1", un="-"),
            _c("n2S_u", "Nombre de files verticales n2", "s", ["1", "2"]),
            _c("p1S_u", "Entraxe vertical p1", un="mm"),
            _c("p2_S", "Entraxe horizontal p2", un="mm", si=_deux_files_S),
            _c("e1S_u", "Pince e1 : dessus de cornière → 1re rangée", un="mm"),
            _c("e2b_u", "Distance e2,b : about de poutre → 1re file", un="mm")]),
    dict(t="Groupe P – boulons dans l'âme de la poutre principale (1 plan, par cornière)", si=boulonne_P,
         f=[_c("n1P_u", "Nombre de rangées n1", un="-"),
            _c("n2P_u", "Nombre de files verticales n2 par cornière", "s", ["1", "2"]),
            _c("p1P_u", "Entraxe vertical p1", un="mm"),
            _c("p2_P", "Entraxe horizontal p2", un="mm", si=_deux_files_P),
            _c("e1P_u", "Pince e1 : dessus de cornière → 1re rangée", un="mm"),
            _c("gA_u", "Trusquinage gA : talon de cornière → 1re file", un="mm")]),
    dict(t="Soudures (cordons d'angle)", si=_soude_un_cote,
         f=[_c("a_P", "Gorge a – ailes A sur l'âme principale", un="mm", si=_soude_P,
               h="Cordon vertical en bout d'aile sur toute la hauteur Lc."),
            _c("lh_P", "Retours horizontaux haut et bas – ailes A", un="mm", si=_soude_P),
            _c("a_S", "Gorge a – ailes B sur l'âme secondaire", un="mm", si=_soude_S),
            _c("lh_S", "Retours horizontaux haut et bas – ailes B", un="mm", si=_soude_S)]),
    dict(t="Efforts de calcul ELU",
         f=[_c("V_Ed", "VEd – effort tranchant vertical", un="kN"),
            _c("N_Ed", "NEd – effort normal dans la secondaire (traction +)", un="kN"),
            _c("H_Ed", "HEd – effort horizontal perpendiculaire à l'âme secondaire", un="kN"),
            _c("M_Ed", "MEd – moment éventuel à la face de l'âme porteuse", un="kNm",
               h="Assemblage articulé : normalement 0. Pris en valeur absolue comme moment parasite.")]),
    dict(t="Coefficients partiels",
         note="Valeurs recommandées de l'EN 1993 : à confirmer par rapport à l'ANB applicable.",
         f=[_c("g_M0", "γM0", un="-"), _c("g_M2n", "γM2 – sections nettes, rupture de bloc", un="-"),
            _c("g_M2", "γM2 – boulons, pression diamétrale, soudures", un="-"),
            _c("g_M3", "γM3 – glissement ELU", un="-"), _c("g_M3s", "γM3,ser – glissement ELS", un="-")]),
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
         note="Règles de détail non normatives, modifiables.",
         f=[_c("eta_c", "Taux d'utilisation cible", un="-"),
            _c("k_e1", "Pince e1 = k × d0 (minimum normatif 1,2)", un="-"),
            _c("k_p1", "Entraxe p1 = k × d0 (minimum normatif 2,2)", un="-"),
            _c("k_e2", "Pince e2 = k × d0 (minimum normatif 1,2)", un="-"),
            _c("pd_dmin", "Plus petit diamètre admis", "s", BOULONS),
            _c("pd_dmax", "Plus grand diamètre admis", "s", BOULONS)]),
]

# Libellés courts (chips d'alerte, panneau des cotes) — ceux de l'interface
# de référence, complétés par le libellé du champ
COURT = {"LC_u": "Lc", "n1S_u": "n1 groupe S", "n2S_u": "n2 groupe S", "p1S_u": "p1 groupe S",
         "p2_S": "p2 groupe S", "e1S_u": "e1 groupe S", "e2b_u": "e2,b", "n1P_u": "n1 groupe P",
         "n2P_u": "n2 groupe P", "p1P_u": "p1 groupe P", "p2_P": "p2 groupe P", "e1P_u": "e1 groupe P",
         "gA_u": "gA", "z_C": "zc", "d_nt": "dnt", "d_nb": "dnb", "l_n": "ln", "g_h": "gh",
         "d_top": "décalage", "corn_u": "cornière", "orient": "orientation",
         "prof_P": "poutre principale", "prof_S": "poutre secondaire", "boulon_u": "boulon",
         "trou": "type de trou", "d0_u": "d0", "M_Ed": "MEd", "N_Ed": "NEd", "H_Ed": "HEd",
         "V_Ed": "VEd", "fix_P": "fixation ailes A", "fix_S": "fixation ailes B", "cat": "catégorie",
         "classe": "classe", "k1_u": "grande aile", "k2_u": "petite aile",
         "lh_S": "retours lh (ailes B)", "a_S": "gorge a (ailes B)", "a_P": "gorge a (ailes A)"}
UNITE = {}
CHAMPS = {}
for _g in GROUPES:
    for _f in _g["f"]:
        UNITE[_f["k"]] = " " + _f["un"] if _f.get("un") and _f["un"] != "-" else ""
        COURT.setdefault(_f["k"], _f["l"])
        CHAMPS[_f["k"]] = _f


def groupes_visibles(u):
    """Les groupes et champs affichables pour l'état ``u`` (conditions du
    formulaire de référence), sous la forme ``[(groupe, [champ, …]), …]``."""
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
    """Fusion d'un fichier enregistré sur les défauts, clé par clé (comme le
    chargement du HTML : les clés inconnues sont ignorées)."""
    d = defaults()
    if not isinstance(objet, dict):
        raise ValueError("Structure JSON inattendue")
    for k in d:
        if k in objet and objet[k] is not None or (k in objet and objet[k] is None):
            d[k] = objet[k]
    return d
