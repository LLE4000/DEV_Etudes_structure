# -*- coding: utf-8 -*-
"""Données d'entrée de l'assemblage par platine d'about de hauteur partielle.

- ``defaults()``          : les clés et leur valeur par défaut ;
- ``NUM``                 : les clés normalisées en nombre par le moteur ;
- ``GROUPES``             : les groupes du formulaire (clé, libellé, type,
                            options, unité, aide, condition d'affichage) ;
- ``PILOTEES_PAR_PREDIM`` : les clés remplacées par la proposition en mode
                            prédimensionnement (marquées ◆) ;
- ``COURT`` / ``UNITE``   : libellé court et unité (chips, panneau des cotes).

Coefficients partiels : défauts de l'ANNEXE NATIONALE BELGE (NBN EN
1993-1-1 ANB : γM0 = 1,00). Aucune vérification de cet assemblage ne fait
intervenir γM1 (pas d'instabilité de plat) : il n'est pas demandé. Pour
l'EN 1993-1-8, aucune divergence belge relevée : valeurs recommandées
(γM2 = 1,25). γMu = 1,1 (tying, situation accidentelle) est la valeur du
guide MSB P5 §2.3 / SCI P358 — l'EN 1993-1-8 ne la fixe pas.
"""
from acier.bibliotheques import DB, PERSO, noms

MODE_VERIF = "VÉRIFICATION"
MODE_PREDIM = "PRÉDIMENSIONNEMENT"

# Nature de l'effort normal de traction : ELU fondamental (fy/γM0, γM2), ou
# force de chaînage « tying » en situation accidentelle (MSB P5 §2.3 : fu/γMu,
# vérifiée indépendamment de VEd)
NAT_ELU = "ELU fondamental"
NAT_TYING = "Tying (accidentel)"

# Annotation de FABRICATION, hors moteur et hors CLES : composition de la
# boulonnerie par boulon (même mécanisme que les deux autres assemblages).
VISSERIE_DEFAUT = "1 rondelle + 1 écrou"


def defaults():
    """Les entrées et leur valeur par défaut — mêmes poutres et même effort
    que les deux autres assemblages de la famille (comparaison directe)."""
    return {
        "id_projet": "", "id_rep": "", "id_red": "", "id_date": "",
        "mode_calc": MODE_VERIF,
        "prof_P": "HEA 400", "hP_u": None, "bP_u": None, "twP_u": None,
        "tfP_u": None, "rP_u": None, "nu_P": "S355",
        "prof_S": "HEA 300", "hS_u": None, "bS_u": None, "twS_u": None,
        "tfS_u": None, "rS_u": None, "nu_S": "S355",
        "d_top": 0, "d_nt": 50, "d_nb": 0, "l_n": 150, "r_n": 10,
        "lt_ok": "Oui",
        "hp_u": 190, "tp_u": 10, "bp_u": 150, "z_C": 50, "nu_pl": "S355",
        "a_w": 5,
        "boulon_u": "M20", "classe": "8.8", "trou": "Normal", "d0_u": None,
        "filet": "Oui", "cat": "A", "mu_s": 0.3, "k_s": 1, "k_ser": 1,
        "n1_u": 3, "p1_u": 60, "e1_u": 35, "p3_u": 90,
        "V_Ed": 125, "N_Ed": 0, "nat_N": NAT_ELU,
        "g_M0": 1, "g_M2n": 1.25, "g_M2": 1.25, "g_M3": 1.25, "g_M3s": 1.1,
        "g_Mu": 1.1,
        "opt_blf": "Non", "expo": "Non",
        "fy_u": None, "fu_u": None, "bw_u": None,
        "eta_c": 0.9, "k_e1": 1.6, "k_p1": 2.7, "k_e2": 1.4,
        "pd_dmin": "M16", "pd_dmax": "M24",
    }


CLES = tuple(defaults().keys())

# Clés passées par N() en tête de compute()
NUM = ["d_top", "d_nt", "d_nb", "l_n", "r_n", "hp_u", "tp_u", "bp_u", "z_C",
       "a_w", "mu_s", "k_s", "k_ser", "n1_u", "p1_u", "e1_u", "p3_u", "V_Ed",
       "N_Ed", "g_M0", "g_M2n", "g_M2", "g_M3", "g_M3s", "g_Mu", "eta_c",
       "k_e1", "k_p1", "k_e2"]

# Champs remplacés par la proposition en mode prédimensionnement (◆)
PILOTEES_PAR_PREDIM = ("boulon_u", "n1_u", "p1_u", "e1_u", "p3_u",
                       "hp_u", "tp_u", "bp_u", "a_w")

PROFILS = [PERSO] + noms(DB["profils"])
ACIERS = noms(DB["aciers"]) + [PERSO]
BOULONS = noms(DB["boulons"])
CLASSES = noms(DB["classes"])
OUI_NON = ["Oui", "Non"]


# --------------------------------------------------------------- conditions
def perso_P(u):
    return u.get("prof_P") == PERSO


def perso_S(u):
    return u.get("prof_S") == PERSO


def _cat_non_A(u):
    return u.get("cat") != "A"


def _cat_B(u):
    return u.get("cat") == "B"


def _acier_perso(u):
    return PERSO in (u.get("nu_P"), u.get("nu_S"), u.get("nu_pl"))


def mode_predim(u):
    return u.get("mode_calc") != MODE_VERIF


def tying(u):
    return u.get("nat_N") == NAT_TYING


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
         note="Reprise dans la note et l'export texte. Un champ vide n'est pas imprimé.",
         f=[_c("id_projet", "Projet", "x"), _c("id_rep", "Repère de l'assemblage", "x"),
            _c("id_red", "Rédacteur", "x"), _c("id_date", "Date", "x")]),
    dict(t="Mode de calcul",
         f=[_c("mode_calc", "Mode de calcul", "s", [MODE_VERIF, MODE_PREDIM],
               h="Prédimensionnement : boulons, rangées, entraxe des files et platine sont proposés "
                 "(onglet Prédim) puis vérifiés en détail. Les champs marqués ◆ sont alors remplacés.")]),
    dict(t="Poutre principale (porteuse)",
         f=[_c("prof_P", "Profilé", "s", PROFILS)] + _dims("P", perso_P) +
           [_c("nu_P", "Nuance d'acier", "s", ACIERS)]),
    dict(t="Poutre secondaire (portée) et grugeage",
         f=[_c("prof_S", "Profilé", "s", PROFILS)] + _dims("S", perso_S) +
           [_c("nu_S", "Nuance d'acier", "s", ACIERS),
            _c("d_top", "Décalage : dessus de la secondaire sous le dessus de la principale", un="mm",
               h="0 = dessus des semelles au même niveau."),
            _c("d_nt", "Profondeur du grugeage supérieur dnt", un="mm", h="0 = semelle supérieure non grugée."),
            _c("d_nb", "Profondeur du grugeage inférieur dnb", un="mm"),
            _c("l_n", "Longueur du grugeage c, depuis la face arrière de la platine", un="mm"),
            _c("r_n", "Rayon du grugeage (information d'exécution)", un="mm", h="Non utilisé dans les résistances."),
            _c("lt_ok", "Poutre secondaire maintenue au déversement", "s", OUI_NON,
               h="Condition d'application de la règle de stabilité locale du grugeage. "
                 "Non : la stabilité d'ensemble de la poutre grugée est à justifier à part.")]),
    dict(t="Platine d'about",
         f=[_c("hp_u", "Hauteur de la platine hp", un="mm",
               h="Recommandation des guides : hp ≥ 0,6·h de la portée (maintien en torsion à l'appui)."),
            _c("tp_u", "Épaisseur de la platine tp", un="mm",
               h="Platine MINCE : c'est sa déformation qui donne la rotation de l'articulation — "
                 "tp ≤ 0,36·d·√(fub/fy) (EN 1993-1-8 §6.4.2(2), ECCS n°126) ; 10 ou 12 mm usuels, "
                 "moins de 8 mm déconseillé (MSB P5 §2.1)."),
            _c("bp_u", "Largeur de la platine bp", un="mm"),
            _c("z_C", "Dessus de la poutre secondaire → dessus de la platine zp", un="mm"),
            _c("nu_pl", "Nuance d'acier de la platine", "s", ACIERS)]),
    dict(t="Soudure de la platine sur l'âme de la portée",
         f=[_c("a_w", "Gorge a (cordon d'angle DOUBLE, toute la hauteur hp)", un="mm",
               h="Deux cordons verticaux, un de chaque côté de l'âme de la poutre portée. Pleine "
                 "résistance recommandée (MSB P5 §2.2) : a ≥ 0,46 / 0,48 / 0,55·tw en S235 / S275 / S355.")]),
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
    dict(t="Groupe de boulons – platine et âme de la porteuse (2 files, 1 plan)",
         f=[_c("n1_u", "Nombre de rangées n1", un="-"),
            _c("p1_u", "Entraxe vertical p1", un="mm"),
            _c("e1_u", "Pince e1 : dessus de la platine → 1re rangée", un="mm"),
            _c("p3_u", "Entraxe des deux files p3 (« gauge »)", un="mm",
               h="Symétrique par rapport à l'âme de la poutre portée. La rondelle doit dégager le cordon.")]),
    dict(t="Efforts de calcul ELU",
         f=[_c("V_Ed", "VEd – effort tranchant vertical", un="kN"),
            _c("N_Ed", "NEd – effort normal dans la secondaire (traction +)", un="kN",
               h="Traction : boulons tendus, platine fléchie (tronçon en T, EN 1993-1-8 §6.2.4), âme "
                 "portée et cordons tendus. Compression : transmise par contact, sans effet sur les boulons."),
            _c("nat_N", "Nature de NEd", "s", [NAT_ELU, NAT_TYING],
               h="ELU fondamental : fy/γM0 et γM2, combiné avec VEd. Tying : force de chaînage en "
                 "situation accidentelle, vérifiée seule avec fu/γMu (MSB P5 §2.3).")]),
    dict(t="Coefficients partiels",
         note="Défauts : ANB BELGE pour l'EN 1993-1-1 (γM0 1,00) ; valeurs recommandées pour "
              "l'EN 1993-1-8 (γM2 1,25). À confirmer sur le texte NBN.",
         f=[_c("g_M0", "γM0", un="-"),
            _c("g_M2n", "γM2 – sections nettes, rupture de bloc", un="-"),
            _c("g_M2", "γM2 – boulons, pression diamétrale, soudures", un="-"),
            _c("g_M3", "γM3 – glissement ELU", un="-"), _c("g_M3s", "γM3,ser – glissement ELS", un="-"),
            _c("g_Mu", "γMu – tying (situation accidentelle)", un="-", si=tying,
               h="1,1 : valeur du guide (MSB P5 §2.3, SCI P358). L'EN 1993-1-8 ne la fixe pas : "
                 "à confirmer pour la Belgique.")]),
    dict(t="Options du modèle mécanique",
         f=[_c("opt_blf", "Appliquer βLf des assemblages longs (Lj > 15d)", "s", OUI_NON,
               h="EN 1993-1-8 §3.8. Les guides ne l'appliquent pas aux attaches d'âme ; Oui = variante conservatrice."),
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

# Libellés courts (chips d'alerte, panneau des cotes)
COURT = {"hp_u": "hp", "tp_u": "tp", "bp_u": "bp", "z_C": "zp", "a_w": "a",
         "n1_u": "n1", "p1_u": "p1", "e1_u": "e1", "p3_u": "p3",
         "d_nt": "dc,sup", "d_nb": "dc,inf", "l_n": "c", "d_top": "Δz",
         "prof_P": "poutre principale", "prof_S": "poutre secondaire",
         "boulon_u": "boulon", "trou": "type de trou", "d0_u": "d0",
         "N_Ed": "NEd", "V_Ed": "VEd", "cat": "catégorie", "classe": "classe",
         "nu_pl": "nuance de la platine"}
UNITE = {}
CHAMPS = {}
for _g in GROUPES:
    for _f in _g["f"]:
        UNITE[_f["k"]] = " " + _f["un"] if _f.get("un") and _f["un"] != "-" else ""
        COURT.setdefault(_f["k"], _f["l"])
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
    inconnues sont ignorées)."""
    d = defaults()
    if not isinstance(objet, dict):
        raise ValueError("Structure JSON inattendue")
    for k in d:
        if k in objet:
            d[k] = objet[k]
    return d
