# -*- coding: utf-8 -*-
"""Poutre bois — les DONNÉES du moteur : ce que les normes fixent, chaque valeur UNE fois, sa source écrite à côté.

Refait le 10/11/2026 (docs/site/LOT_Q_QUATRE_MODULES.md § 1) : le moteur calcule selon l'Eurocode 5, il ne reproduit
plus l'écran Streamlit ``modules/poutre_bois.py`` (dont la base mêlait trois éditions de l'EN 338).

Les normes :
- EN 1995-1-1:2004 + A1:2008 + A2:2014 (Eurocode 5) et son annexe belge NBN EN 1995-1-1 ANB (γ_M, k_cr) ;
- EN 338:2016, tableau 1 (bois massif résineux, classes C) ; EN 14080:2013, tableaux 4 et 5 (lamellé-collé) ;
- NBN EN 1990 ANB (γ_G, γ_Q, ψ₂ ; expression 6.10).

Les valeurs ont été croisées sur des sources indépendantes le 10/11/2026 (rapport « SOURCES_BOIS » du lot Q, résumé
dans docs/bois/POUTRE_BOIS_MOTEUR.md § 7). ``tests/test_poutre_bois_moteur.py`` relit chaque table contre une
transcription indépendante écrite dans le test (deux copies nommées, liées par le test).

Unités : N/mm² (résistances, modules), kg/m³ (masses volumiques), mm (dimensions), m (portée, entraxes), kN/m et
kN/m² (charges).
"""

REFERENCE = ("EN 1995-1-1:2004 + A1:2008 + A2:2014 et NBN EN 1995-1-1 ANB ; EN 338:2016 ; EN 14080:2013 ; "
             "NBN EN 1990 ANB")

# ------------------------------------------------------------------ les trois types de bois
# Le TYPE d'un bois est celui de sa classe de résistance : les classes C (résineux massifs) et D (feuillus massifs) de
# l'EN 338, les classes GL (lamellé-collé résineux) de l'EN 14080. C'est lui que les coefficients lisent : γ_M, k_h,
# k_c,90, les sections standard. (24/11/2026 : les feuillus arrivent, avec les essences — voir ESSENCES.)
MASSIF, FEUILLU, LAMELLE = "massif", "feuillu", "lamelle"
TYPES = (MASSIF, FEUILLU, LAMELLE)
LIBELLES_TYPE = {MASSIF: "bois massif résineux", FEUILLU: "bois massif feuillu", LAMELLE: "lamellé-collé résineux"}
COURTS_TYPE = {MASSIF: "bois massif", FEUILLU: "bois feuillu", LAMELLE: "lamellé-collé"}

# ------------------------------------------------------------------ les classes de résistance
# Par classe : f_m,k, f_t,0,k, f_c,0,k, f_c,90,k, f_v,k (N/mm²), E_0,mean, E_0,05, G_mean (N/mm²), ρ_k, ρ_mean (kg/m³).
# f_t,0,k et f_c,0,k ne servent à aucune vérification d'une poutre fléchie : ils sont gardés pour la fiche.
PROPRIETES = ("fm_k", "ft0_k", "fc0_k", "fc90_k", "fv_k", "E0_mean", "E0_05", "G_mean", "rho_k", "rho_mean")

# EN 338:2016, tableau 1 — résineux, classes définies par des essais de flexion sur chant. Une ligne par propriété,
# une colonne par classe, comme le tableau de la norme (et comme la conception § 1.7).
_CLASSES_C = ("C14", "C16", "C18", "C20", "C22", "C24", "C27", "C30", "C35", "C40", "C45", "C50")
_EN_338_2016_TABLEAU_1 = {
    "fm_k":     (14, 16, 18, 20, 22, 24, 27, 30, 35, 40, 45, 50),
    "ft0_k":    (7.2, 8.5, 10, 11.5, 13, 14.5, 16.5, 19, 22.5, 26, 30, 33.5),
    "fc0_k":    (16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 29, 30),
    "fc90_k":   (2.0, 2.2, 2.2, 2.3, 2.4, 2.5, 2.5, 2.7, 2.7, 2.8, 2.9, 3.0),
    "fv_k":     (3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
    "E0_mean":  (7000, 8000, 9000, 9500, 10000, 11000, 11500, 12000, 13000, 14000, 15000, 16000),
    "E0_05":    (4700, 5400, 6000, 6400, 6700, 7400, 7700, 8000, 8700, 9400, 10100, 10700),
    "G_mean":   (440, 500, 560, 590, 630, 690, 720, 750, 810, 880, 940, 1000),
    "rho_k":    (290, 310, 320, 330, 340, 350, 360, 380, 390, 400, 410, 430),
    "rho_mean": (350, 370, 380, 400, 410, 420, 430, 460, 470, 480, 490, 520),
}

# EN 338:2016, tableau 3 — FEUILLUS, classes définies par des essais de flexion sur chant (les huit classes que le
# marché belge rencontre : chêne D24 et D30, hêtre D35 et D40, bois tropicaux D50 à D70). Transcription du 24/11/2026,
# recoupée par les relations de l'EN 384:2016 que la norme a utilisées pour bâtir le tableau (f_t,0,k = 0,6·f_m,k ;
# f_c,0,k = 4,3·√f_m,k ; f_c,90,k = 0,010·ρ_k jusqu'à D50, 0,015·ρ_k au-delà ; E_0,05 = 0,84·E_0,mean ; G_mean = E_0,mean/16 ;
# ρ_mean = 1,2·ρ_k — tests/test_poutre_bois_moteur.py les rejoue) et par trois lectures partielles (D40 : 40 / 13 000 /
# 660 ; ρ_mean de D30 et D70) ; les f_v,k ne se déduisent d'aucune relation : À RECOUPER sur le texte de la norme
# (docs/bois/POUTRE_BOIS_MOTEUR.md § 4).
_CLASSES_D = ("D18", "D24", "D30", "D35", "D40", "D50", "D60", "D70")
_EN_338_2016_TABLEAU_3 = {
    "fm_k":     (18, 24, 30, 35, 40, 50, 60, 70),
    "ft0_k":    (11, 14, 18, 21, 24, 30, 36, 42),
    "fc0_k":    (18, 21, 24, 25, 27, 30, 33, 36),
    "fc90_k":   (4.8, 4.9, 5.3, 5.4, 5.5, 6.2, 10.5, 12.0),
    "fv_k":     (3.5, 3.7, 3.9, 4.1, 4.2, 4.5, 4.8, 5.0),
    "E0_mean":  (9500, 10000, 11000, 12000, 13000, 14000, 17000, 20000),
    "E0_05":    (8000, 8400, 9200, 10100, 10900, 11800, 14300, 16800),
    "G_mean":   (590, 630, 690, 750, 810, 880, 1060, 1250),
    "rho_k":    (475, 485, 530, 540, 550, 620, 700, 800),
    "rho_mean": (570, 580, 640, 650, 660, 740, 840, 960),
}

# EN 14080:2013 — lamellé-collé résineux : tableau 5 (homogène, « h ») et tableau 4 (panaché, « c »).
# (GL28h : f_t,0,k = 22,3 — des transcriptions écrivent 22,4 = 0,8 × 28 ; la norme dit 22,3. GL30h et GL30c ajoutés le
# 24/11/2026 : f_m, f_t,0, f_c,0, E_0,mean et E_0,05 recoupés sur une transcription ouverte ; ρ du GL30c de mémoire.)
_CLASSES_GL = ("GL24h", "GL28h", "GL30h", "GL32h", "GL24c", "GL28c", "GL30c", "GL32c")
_EN_14080_2013 = {
    "fm_k":     (24, 28, 30, 32, 24, 28, 30, 32),
    "ft0_k":    (19.2, 22.3, 24, 25.6, 17, 19.5, 19.5, 19.5),
    "fc0_k":    (24, 28, 30, 32, 21.5, 24, 24.5, 24.5),
    "fc90_k":   (2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5),
    "fv_k":     (3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5),
    "E0_mean":  (11500, 12600, 13600, 14200, 11000, 12500, 13000, 13500),
    "E0_05":    (9600, 10500, 11300, 11800, 9100, 10400, 10800, 11200),
    "G_mean":   (650, 650, 650, 650, 650, 650, 650, 650),
    "rho_k":    (385, 425, 430, 440, 365, 390, 390, 400),
    "rho_mean": (420, 460, 480, 490, 400, 420, 430, 440),
}
NORMES_TYPE = {MASSIF: "EN 338:2016, tableau 1", FEUILLU: "EN 338:2016, tableau 3", LAMELLE: "EN 14080:2013"}


def _classes() -> dict:
    """Les tables des deux normes, retournées par classe : {"C24": {"type", "norme", "fm_k": 24.0, …}, …}."""
    out = {}
    for noms, table, typ in ((_CLASSES_C, _EN_338_2016_TABLEAU_1, MASSIF), (_CLASSES_D, _EN_338_2016_TABLEAU_3, FEUILLU),
                             (_CLASSES_GL, _EN_14080_2013, LAMELLE)):
        for k, nom in enumerate(noms):
            if typ == LAMELLE:
                norme = "EN 14080:2013, tableau 5" if nom.endswith("h") else "EN 14080:2013, tableau 4"
            else:
                norme = NORMES_TYPE[typ]
            out[nom] = dict({"type": typ, "norme": norme}, **{p: float(table[p][k]) for p in PROPRIETES})
    return out


CLASSES = _classes()                       # dans l'ordre des menus : les C, les D, puis les GL
CLASSE_DEFAUT = "C24"
CLASSE_DEFAUT_TYPE = {MASSIF: "C24", FEUILLU: "D30", LAMELLE: "GL24h"}      # la classe qu'un bois prend en changeant de type

# ------------------------------------------------------------------ les essences et leur classement visuel
# L'ESSENCE ne calcule rien : l'Eurocode 5 lit la CLASSE de résistance (EN 338, EN 14080), qui porte toutes les valeurs.
# Elle fixe le TYPE (résineux, feuillu, lamellé-collé — donc k_c,90, k_h, les sections proposées), elle est écrite dans la
# note, et elle dit quelles QUALITÉS de classement visuel (EN 1912 : une règle nationale, une qualité → une classe) mènent
# à chaque classe. (clé, libellé, nom latin, type)
ESSENCES = (
    ("epicea", "Épicéa", "Picea abies", MASSIF),
    ("sapin", "Sapin", "Abies alba", MASSIF),
    ("pin_sylvestre", "Pin sylvestre", "Pinus sylvestris", MASSIF),
    ("pin_maritime", "Pin maritime", "Pinus pinaster", MASSIF),
    ("douglas", "Douglas", "Pseudotsuga menziesii", MASSIF),
    ("meleze", "Mélèze", "Larix decidua", MASSIF),
    ("resineux", "Résineux (essence non précisée)", "", MASSIF),
    ("chene", "Chêne", "Quercus robur, Q. petraea", FEUILLU),
    ("hetre", "Hêtre", "Fagus sylvatica", FEUILLU),
    ("chataignier", "Châtaignier", "Castanea sativa", FEUILLU),
    ("azobe", "Azobé", "Lophira alata", FEUILLU),
    ("feuillu", "Feuillu (essence non précisée)", "", FEUILLU),
    ("lamelle", "Lamellé-collé (épicéa, sapin)", "Picea abies, Abies alba", LAMELLE),
)
CLES_ESSENCE = tuple(e[0] for e in ESSENCES)
LIBELLES_ESSENCE = {e[0]: e[1] for e in ESSENCES}
LATIN_ESSENCE = {e[0]: e[2] for e in ESSENCES}
TYPE_ESSENCE = {e[0]: e[3] for e in ESSENCES}
ESSENCE_DEFAUT = "epicea"
ESSENCE_DEFAUT_TYPE = {MASSIF: "resineux", FEUILLU: "feuillu", LAMELLE: "lamelle"}      # une note sans essence : « non précisée »
# Les règles de classement visuel et leurs qualités → la classe EN 338 (EN 1912, l'affectation des qualités visuelles et
# des essences aux classes de résistance) : (règle, qualité, classe), par essence. Résineux français (NF B 52-001-1 :
# ST-I, ST-II, ST-III → C30, C24, C18) : recoupé le 24/11/2026 sur les fiches du classement structurel de France Bois
# Forêt et du CTBA ; chêne (D30, D24) et hêtre (H2 → D35, H3 → D24) : le rapport des qualités visuelles approuvées (CEN,
# 2024) qui complète l'EN 1912, provenance Belgique et France. Les règles allemande (DIN 4074-1 : S7, S10, S13 → C16, C24,
# C30), nordique (INSTA 142 : T1, T2, T3 → C18, C24, C30) et britannique (BS 4978 : GS, SS → C16, C24) sont celles du
# tableau 1 de l'EN 1912:2012, de mémoire : À RECOUPER (docs/bois/POUTRE_BOIS_MOTEUR.md § 4). Le classement belge
# (NBN B 16-520 : S6, S8, S10) n'a pas de correspondance confirmée : il n'est pas écrit.
_FR = (("NF B 52-001-1", "ST-I", "C30"), ("NF B 52-001-1", "ST-II", "C24"), ("NF B 52-001-1", "ST-III", "C18"))
_DE = (("DIN 4074-1", "S13", "C30"), ("DIN 4074-1", "S10", "C24"), ("DIN 4074-1", "S7", "C16"))
_NORD = (("INSTA 142", "T3", "C30"), ("INSTA 142", "T2", "C24"), ("INSTA 142", "T1", "C18"))
_UK = (("BS 4978", "SS", "C24"), ("BS 4978", "GS", "C16"))
CLASSEMENTS = {
    "epicea": _FR + _DE + _NORD + _UK,
    "sapin": _FR + _DE + _UK,
    "pin_sylvestre": _FR + _DE + _NORD + _UK,
    "pin_maritime": _FR,
    "douglas": _FR + _DE,
    "meleze": _FR + _DE,
    "chene": (("NF B 52-001-1:2018", "", "D30"), ("NF B 52-001-1:2018", "", "D24")),
    "hetre": (("NF B 52-001-1:2018", "H2", "D35"), ("NF B 52-001-1:2018", "H3", "D24")),
}
REGLES_CLASSEMENT = {"NF B 52-001-1": "France", "NF B 52-001-1:2018": "France", "DIN 4074-1": "Allemagne", "INSTA 142": "pays nordiques",
                     "BS 4978": "Royaume-Uni"}


def qualites(essence: str, classe: str) -> tuple:
    """Les qualités de classement visuel qui mènent à ``classe`` pour ``essence`` : (« ST-II (NF B 52-001-1) », …)."""
    return tuple((f"{q} ({r})" if q else r) for r, q, c in CLASSEMENTS.get(essence, ()) if c == classe)


def classes_du_type(typ: str) -> tuple:
    """Les classes de résistance d'un type de bois, dans l'ordre des menus."""
    return tuple(c for c, m in CLASSES.items() if m["type"] == typ)

# ------------------------------------------------------------------ les coefficients de l'Eurocode 5
# γ_M — EN 1995-1-1 tableau 2.3, valeurs retenues par l'annexe belge (NBN EN 1995-1-1 ANB) : bois massif 1,30 (résineux
# et feuillus), lamellé-collé 1,25 (fiche Buildwise de la NBN EN 1995-1-1, recoupée le 24/11/2026).
GAMMA_M = {MASSIF: 1.30, FEUILLU: 1.30, LAMELLE: 1.25}

# k_mod — EN 1995-1-1 tableau 3.1 (bois massif EN 14081-1 et lamellé-collé EN 14080 : les mêmes valeurs), par classe
# de service puis par classe de durée de charge.
DUREES = ("permanent", "long", "moyen", "court", "instantane")          # de la plus longue à la plus courte
LIBELLES_DUREE = {"permanent": "permanent", "long": "long terme", "moyen": "moyen terme", "court": "court terme",
                  "instantane": "instantané"}
# EN 1995-1-1 tableau 2.1 : l'ordre de grandeur de la durée cumulée de la charge caractéristique
PERIODES_DUREE = {"permanent": "plus de 10 ans", "long": "6 mois à 10 ans", "moyen": "1 semaine à 6 mois",
                  "court": "moins d'une semaine", "instantane": "instantanée"}
CLASSES_SERVICE = (1, 2, 3)
# EN 1995-1-1 §2.3.1.3 : l'ambiance de chaque classe de service
LIBELLES_SERVICE = {1: "intérieur chauffé", 2: "abrité, non chauffé", 3: "exposé aux intempéries"}
KMOD = {
    1: {"permanent": 0.60, "long": 0.70, "moyen": 0.80, "court": 0.90, "instantane": 1.10},
    2: {"permanent": 0.60, "long": 0.70, "moyen": 0.80, "court": 0.90, "instantane": 1.10},
    3: {"permanent": 0.50, "long": 0.55, "moyen": 0.65, "court": 0.70, "instantane": 0.90},
}

# k_def — EN 1995-1-1 tableau 3.2 (bois massif et lamellé-collé), par classe de service.
KDEF = {1: 0.60, 2: 0.80, 3: 2.00}

# k_cr — EN 1995-1-1 §6.1.7(2) (amendement A1:2008) : 0,67 pour le bois massif et le lamellé-collé ; valeur de
# l'annexe belge (fiche Buildwise de la NBN EN 1995-1-1). 0,67 est la valeur la plus sévère rencontrée.
K_CR = 0.67

# k_sys — EN 1995-1-1 §6.6(2) : pièces semblables, régulièrement espacées, liées par un système continu de
# répartition des charges ; appliqué ici à f_m,d et à f_v,d (pas à la compression d'appui).
K_SYS = 1.1
ENTRAXE_SYSTEME = 0.60          # m — au-delà, l'effet système est signalé (conception § 1.4 : « 60 cm environ »)

# k_h — EN 1995-1-1 §3.2(3) (bois massif, résineux ou feuillu, de masse volumique ρ_k ≤ 700 kg/m³) et §3.3(3)
# (lamellé-collé) : sous la hauteur de référence, f_m,k est majoré de k_h = min((h_ref/h)^exposant ; plafond) ; à partir
# de h_ref, k_h = 1. Un bois massif plus lourd que RHO_K_MAX_K_H (D70 : ρ_k = 800) n'a pas de majoration.
K_H = {MASSIF: {"h_ref": 150.0, "exposant": 0.2, "plafond": 1.3, "article": "§3.2(3)"},
       FEUILLU: {"h_ref": 150.0, "exposant": 0.2, "plafond": 1.3, "article": "§3.2(3)"},
       LAMELLE: {"h_ref": 600.0, "exposant": 0.1, "plafond": 1.1, "article": "§3.3(3)"}}
RHO_K_MAX_K_H = 700.0           # kg/m³ — §3.2(3) : le bois massif plus lourd garde k_h = 1

# Déversement — EN 1995-1-1 §6.3.3 : σ_m,crit = 0,78·b²·E_0,05/(h·l_ef) (6.32, résineux, section rectangulaire) ;
# λ_rel,m = √(f_m,k/σ_m,crit) (6.30) ; k_crit (6.34) : 1 jusqu'à 0,75 ; 1,56 − 0,75·λ jusqu'à 1,4 ; 1/λ² au-delà.
# Tableau 6.1 : poutre sur deux appuis, charge uniformément répartie : l_ef/l = 0,9 ; charge appliquée sur la fibre
# comprimée : l_ef augmentée de 2·h ; moment constant (pris entre deux entretoises, côté sûr) : l_ef/l = 1,0.
COEF_SIGMA_CRIT = 0.78
LAMBDA_PALIER, LAMBDA_ELANCE = 0.75, 1.4
K_CRIT_ORIGINE, K_CRIT_PENTE = 1.56, 0.75
LEF_CHARGE_REPARTIE = 0.9
LEF_FIBRE_COMPRIMEE = 2.0       # × h
MAINTIENS = ("continu", "appuis", "entretoises")
LIBELLES_MAINTIEN = {"continu": "fibre comprimée maintenue sur toute la longueur", "appuis": "maintien latéral aux appuis seulement",
                     "entretoises": "entretoises régulièrement espacées"}

# Cisaillement — section rectangulaire : τ_max = 1,5·V/A (répartition parabolique).
COEF_TAU_RECTANGLE = 1.5

# Compression perpendiculaire au fil à l'appui — EN 1995-1-1 §6.1.5 (amendement A1:2008) : la longueur de contact
# est augmentée de 30 mm de chaque côté, sans dépasser a (le débord, côté about), l (la longueur de contact) ni
# l_1/2 (la demi-distance libre, côté travée) ; k_c,90 sur appuis discrets, si l_1 ≥ 2·h : 1,5 (massif résineux),
# 1,75 (lamellé-collé résineux, si l ≤ 400 mm) ; 1,0 sinon — le §6.1.5(4) ne majore que les RÉSINEUX : un feuillu garde 1,0.
MAJORATION_APPUI = 30.0         # mm
K_C90 = {MASSIF: 1.5, FEUILLU: 1.0, LAMELLE: 1.75}
K_C90_DEFAUT = 1.0
L1_SUR_H_MIN = 2.0              # l_1 ≥ 2·h
A_MAX_K_C90 = {LAMELLE: 400.0}  # mm — lamellé-collé : la majoration ne vaut que pour l ≤ 400 mm

# Flèche — poutre sur deux appuis sous charge répartie : flexion 5·p·L⁴/(384·E·I) ; effort tranchant
# κ·p·L²/(8·G·A) avec κ = 1,2 (section rectangulaire, Timoshenko : 6/5).
FLECHE_NUM, FLECHE_DEN = 5.0, 384.0
FORME_TRANCHANT = 1.2

G_PESANTEUR = 9.81              # m/s² — le poids propre : ρ_mean·b·h·g

# ------------------------------------------------------------------ la charge variable : sa NATURE fixe sa durée et ψ₂
# Classe de durée : EN 1995-1-1 tableau 2.2 (exploitation des planchers : moyen terme ; stockage : long terme ;
# entretien d'une toiture, chantier de moins d'une semaine : court terme, tableau 2.1) ; ψ₂ : NBN EN 1990 ANB,
# tableau A1.1 (catégories A et B : 0,3 ; C et D : 0,6 ; E : 0,8 ; H : 0 ; neige en Belgique, altitude ≤ 1 000 m : 0).
# LA NEIGE EN COURT TERME est la valeur de la conception du lot Q : le tableau 2.2 de l'EN cite la neige en exemple
# du moyen ET du court terme et renvoie à l'annexe nationale, et le croisement des sources du 10/11/2026 n'a pas pu
# lire le texte de l'ANB belge (seul un extrait de la STS 31 dit la neige de courte durée). Plus sévère : la nature
# « autre » avec « moyen terme » (k_mod 0,80 au lieu de 0,90).
# (clé, libellé du menu, durée, ψ₂) — « autre » : durée et ψ₂ au choix.
NATURES = (
    ("A", "A · habitation", "moyen", 0.3),
    ("B", "B · bureaux", "moyen", 0.3),
    ("C", "C · lieux de réunion", "moyen", 0.6),
    ("D", "D · commerces", "moyen", 0.6),
    ("E", "E · stockage", "long", 0.8),
    ("H", "H · toiture (entretien)", "court", 0.0),
    ("neige", "Neige (Belgique, ≤ 1 000 m)", "court", 0.0),
    ("chantier", "Chantier, coffrage (< 1 semaine)", "court", 0.0),
    ("autre", "Autre (durée et ψ₂ au choix)", None, None),
)
CLES_NATURE = tuple(n[0] for n in NATURES)
LIBELLES_NATURE = {n[0]: n[1] for n in NATURES}
DUREE_NATURE = {n[0]: n[2] for n in NATURES}
PSI2_NATURE = {n[0]: n[3] for n in NATURES}
NATURE_LIBRE = "autre"

# γ_G, γ_Q — NBN EN 1990 ANB, expression (6.10) : 1,35 et 1,50.
GAMMA_G, GAMMA_Q = 1.35, 1.50

# ------------------------------------------------------------------ les limites de flèche (diviseurs de L)
# EN 1995-1-1 tableau 7.2 (poutre sur deux appuis) : w_inst de L/300 à L/500, w_net,fin de L/250 à L/350 ; les menus
# offrent aussi des limites plus larges (ouvrages provisoires, toitures), au choix du bureau (conception § 1.4).
TABLEAU_7_2 = {"inst": (300, 500), "net_fin": (250, 350)}      # les diviseurs de L recommandés par la norme
LIMITES_INST = (200, 250, 300, 350, 400, 500)
LIMITES_FIN = (150, 200, 250, 300, 350)
LIMITE_INST_DEFAUT, LIMITE_FIN_DEFAUT = 300, 250

# ------------------------------------------------------------------ les sections standard (b, h) en mm
# Bois massif : les sections courantes du commerce belge (liste du lot Q, 10/11/2026 — non recoupée sur un tarif de
# négociant : le rapport de sources n'a pu confirmer que les largeurs de 63 et 75 mm).
SECTIONS_MASSIF = (
    (38, 89), (38, 140), (38, 184), (38, 235),
    (45, 95), (45, 120), (45, 145), (45, 170), (45, 195), (45, 220), (45, 245),
    (63, 150), (63, 175), (63, 200), (63, 225),
    (75, 150), (75, 175), (75, 200), (75, 225),
    (100, 200), (100, 225), (100, 250), (100, 300),
    (150, 150), (150, 300), (200, 200),
)
# Lamellé-collé : largeurs et hauteurs courantes (lamelles de 40 mm), limitées aux élancements h/b ≤ 6.
LARGEURS_LAMELLE = (80, 100, 120, 140, 160, 180, 200)
HAUTEURS_LAMELLE = (160, 200, 240, 280, 320, 360, 400, 480, 560, 640)
ELANCEMENT_MAX_LAMELLE = 6
SECTIONS_LAMELLE = tuple((b, h) for b in LARGEURS_LAMELLE for h in HAUTEURS_LAMELLE if h <= ELANCEMENT_MAX_LAMELLE * b)
EPAISSEUR_LAMELLE = 40.0        # mm — pour le dessin de la section (lamelles horizontales)
# Les feuillus se débitent aux mêmes sections que les résineux (une liste à recouper chez un négociant de chêne).
SECTIONS = {MASSIF: SECTIONS_MASSIF, FEUILLU: SECTIONS_MASSIF, LAMELLE: SECTIONS_LAMELLE}
SECTION_DEFAUT = {MASSIF: "75x225", FEUILLU: "100x200", LAMELLE: "120x400"}

# ------------------------------------------------------------------ la saisie : modes, bornes, la poutre neuve
MODES_CHARGE = ("lineaire", "surfacique")       # G, Q en kN/m — ou g, q en kN/m² × entraxe
NOM_MAX = 60                                    # caractères

# Les BORNES des champs numériques : chemin → (minimum, maximum, unité, libellé pour les messages). Fermées.
BORNES = {
    "b": (20.0, 400.0, "mm", "la largeur b"),
    "h": (40.0, 2500.0, "mm", "la hauteur h"),
    "L": (0.5, 30.0, "m", "la portée L"),
    "charges/G": (0.0, 200.0, "kN/m", "la charge permanente G"),
    "charges/Q": (0.0, 200.0, "kN/m", "la charge variable Q"),
    "charges/g": (0.0, 50.0, "kN/m²", "la charge permanente g"),
    "charges/q": (0.0, 50.0, "kN/m²", "la charge variable q"),
    "charges/entraxe": (0.1, 6.0, "m", "l'entraxe des pièces"),
    "charges/psi2": (0.0, 1.0, "", "ψ₂"),
    "appui/a": (20.0, 600.0, "mm", "la longueur d'appui a"),
    "appui/debord": (0.0, 2000.0, "mm", "le débord"),
    "maintien/entraxe": (0.2, 30.0, "m", "l'entraxe des entretoises"),
    "gamma_G": (1.0, 1.5, "", "γ_G"),
    "gamma_Q": (1.0, 2.0, "", "γ_Q"),
    "contre_fleche": (0.0, 200.0, "mm", "la contre-flèche"),
}

# La poutre NEUVE (conception § 1.5) : C24, 75 × 225, L = 4,00 m, G = 0,6 et Q = 0,8 kN/m (habitation), classe de
# service 1, appui de 100 mm, maintien continu — conforme, et le sélecteur propose plus léger. Les charges
# surfaciques en réserve donnent les mêmes charges linéaires (1,5 et 2,0 kN/m² × 0,40 m).
DEFAUTS = {
    "nom": "P1",
    "essence": ESSENCE_DEFAUT,
    "classe": CLASSE_DEFAUT,
    "standard": True,
    "section": SECTION_DEFAUT[MASSIF],
    "b": 75.0,
    "h": 225.0,
    "L": 4.0,
    "charges": {"mode": "lineaire", "G": 0.6, "Q": 0.8, "g": 1.5, "q": 2.0, "entraxe": 0.4, "poids_propre": True,
                "nature": "A", "duree": DUREE_NATURE["A"], "psi2": PSI2_NATURE["A"]},
    "appui": {"verifier": True, "a": 100.0, "debord": 0.0},
    "maintien": {"type": "continu", "entraxe": 1.0},
    "service": 1,
    "gamma_G": GAMMA_G,
    "gamma_Q": GAMMA_Q,
    "systeme": False,
    "lim_inst": LIMITE_INST_DEFAUT,
    "lim_fin": LIMITE_FIN_DEFAUT,
    "contre_fleche": 0.0,
}

# Ce que le module NE COUVRE PAS, dit à l'écran (conception § 1.4).
NON_COUVERT = (
    "les charges concentrées et les charges partielles",
    "les porte-à-faux et les poutres continues",
    "les entailles d'appui (k_v, §6.5)",
    "les vibrations des planchers (§7.3)",
    "la flexion déviée et la flexion composée (effort normal)",
    "la résistance au feu (EN 1995-1-2)",
    "les perçages et les trémies",
    "le bois mis en œuvre humide (k_def majoré de 1,0, §3.2(4))",
)
