# -*- coding: utf-8 -*-
# =============================================================
#  sol_base.py — Base de données des sols (contexte belge)
#  VERSION 2.0
#
#  Évolutions vs la table intégrée à raideur_sol v4.0 :
#   1. Les k tabulés sont désormais NOMMÉS pour ce qu'ils sont : des
#      valeurs de PLAQUE 0,30 m (Terzaghi 1955). Utilisées telles quelles
#      pour une semelle ou un radier, elles surestiment k d'un facteur 3
#      à 30 selon la largeur. La correction de taille est appliquée à
#      l'affichage (sol_theorie.k_terzaghi_taille).
#   2. Ajout du module OEDOMÉTRIQUE M par type de sol : c'est lui qui
#      entre dans le calcul de tassement, pas le module de Young E.
#   3. Les mots-clés de reconnaissance portent leurs ACCENTS : sans cela
#      « craie altérée » ne correspondait pas à « craie alter » et
#      retombait sur « Craie saine » (E médian ×23).
#   4. Le repli générique d'une roche pointe vers la variante ALTÉRÉE et
#      non saine : à profondeur de fondation, un log qui dit « schiste »
#      désigne presque toujours un matériau altéré, et c'est l'hypothèse
#      prudente.
# =============================================================

# Chaque entrée :
#   gamma      poids volumique (kN/m³)
#   qc_min/max résistance de cône usuelle (MPa) — None si refus de pointe
#   alpha_qc   coefficient indicatif qc -> E (littérature) — None si sans objet
#   rf_typ     rapport de frottement typique (%)
#   E_min/max  module de Young (MPa)
#   M_min/max  module OEDOMÉTRIQUE (MPa) — celui du calcul de tassement
#   kp_min/max coefficient de réaction à la PLAQUE 0,30 m (MN/m³)
#   nature     "sable" ou "argile" — pilote la correction de taille Terzaghi
#   cpt_ok     False si le CPT n'a pas de sens (refus de pointe)
SOIL_DB = {
    "Remblais / terre végétale": dict(
        category="Remblai", gamma=17.0, nature="sable",
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=2.0, E_max=10.0, M_min=3.0, M_max=13.0,
        kp_min=1, kp_max=8, cpt_ok=False,
        desc="Matériau rapporté, hétérogène, non contrôlé. Ne jamais retenir "
             "comme assise de fondation sans reconnaissance spécifique (souvent à purger)."),
    "Tourbe": dict(
        category="Sol organique", gamma=10.0, nature="argile",
        qc_min=0.1, qc_max=0.5, alpha_qc=4.0, rf_typ=4.0,
        E_min=0.5, E_max=2.5, M_min=0.6, M_max=3.4,
        kp_min=1, kp_max=5, cpt_ok=True,
        desc="Sol très organique, très compressible, souvent saturé. Portance très "
             "faible : à éviter comme assise (substitution, pieux, colonnes)."),
    "Argile très molle": dict(
        category="Argile", gamma=16.0, nature="argile",
        qc_min=0.3, qc_max=1.0, alpha_qc=4.0, rf_typ=3.5,
        E_min=1.0, E_max=5.0, M_min=1.3, M_max=6.7,
        kp_min=2, kp_max=10, cpt_ok=True,
        desc="Argile plastique peu consolidée, forte compressibilité, faibles résistances."),
    "Argile molle à moyenne": dict(
        category="Argile", gamma=18.0, nature="argile",
        qc_min=1.0, qc_max=2.5, alpha_qc=5.0, rf_typ=3.0,
        E_min=5.0, E_max=15.0, M_min=6.7, M_max=20.0,
        kp_min=10, kp_max=40, cpt_ok=True,
        desc="Normalement à légèrement surconsolidée, tassements notables sous charge."),
    "Argile ferme / raide": dict(
        category="Argile", gamma=19.0, nature="argile",
        qc_min=2.5, qc_max=5.0, alpha_qc=6.0, rf_typ=2.5,
        E_min=15.0, E_max=40.0, M_min=20.0, M_max=54.0,
        kp_min=20, kp_max=80, cpt_ok=True,
        desc="Argile raide bien consolidée, tassements plus limités."),
    "Argile surconsolidée (Boom/Ypresienne)": dict(
        category="Argile", gamma=20.0, nature="argile",
        qc_min=3.0, qc_max=8.0, alpha_qc=7.0, rf_typ=2.5,
        E_min=25.0, E_max=80.0, M_min=34.0, M_max=108.0,
        kp_min=40, kp_max=150, cpt_ok=True,
        desc="Argiles profondes surconsolidées (bassin belge), raides, faible "
             "compressibilité résiduelle mais sensibles au gonflement/retrait si décomprimées."),
    "Limon (loess)": dict(
        category="Limon", gamma=18.0, nature="argile",
        qc_min=1.0, qc_max=3.0, alpha_qc=4.0, rf_typ=2.0,
        E_min=8.0, E_max=25.0, M_min=11.0, M_max=34.0,
        kp_min=15, kp_max=60, cpt_ok=True,
        desc="Très répandu en Hesbaye/Brabant. Comportement intermédiaire argile/sable, "
             "sensible à l'eau (collapsibilité possible à l'état non saturé)."),
    "Sable lâche": dict(
        category="Sable", gamma=18.0, nature="sable",
        qc_min=1.0, qc_max=5.0, alpha_qc=3.5, rf_typ=0.6,
        E_min=5.0, E_max=15.0, M_min=6.7, M_max=20.0,
        kp_min=10, kp_max=30, cpt_ok=True,
        desc="Peu compacté, tassements importants, risque de liquéfaction si saturé et sismique."),
    "Sable moyennement compact": dict(
        category="Sable", gamma=19.0, nature="sable",
        qc_min=5.0, qc_max=12.0, alpha_qc=5.0, rf_typ=0.5,
        E_min=15.0, E_max=40.0, M_min=20.0, M_max=54.0,
        kp_min=30, kp_max=80, cpt_ok=True,
        desc="Sable courant sous bâtiments, portance correcte, tassements modérés."),
    "Sable dense": dict(
        category="Sable", gamma=20.0, nature="sable",
        qc_min=12.0, qc_max=25.0, alpha_qc=6.0, rf_typ=0.4,
        E_min=40.0, E_max=80.0, M_min=54.0, M_max=108.0,
        kp_min=80, kp_max=150, cpt_ok=True,
        desc="Très compact, bonne portance, tassements faibles."),
    "Sable graveleux / grave compacte": dict(
        category="Sable/grave", gamma=21.0, nature="sable",
        qc_min=15.0, qc_max=30.0, alpha_qc=4.0, rf_typ=0.4,
        E_min=50.0, E_max=120.0, M_min=67.0, M_max=162.0,
        kp_min=100, kp_max=200, cpt_ok=True,
        desc="Granulométrie étalée bien compactée, très bonne portance."),
    "Sable argileux / argile sableuse": dict(
        category="Sable", gamma=19.0, nature="argile",
        qc_min=3.0, qc_max=8.0, alpha_qc=4.5, rf_typ=1.5,
        E_min=12.0, E_max=35.0, M_min=16.0, M_max=47.0,
        kp_min=20, kp_max=60, cpt_ok=True,
        desc="Mélange intermédiaire (fréquent dans les formations bruxelliennes/yprésiennes) : "
             "comportement plastique, drainage lent."),
    "Craie altérée": dict(
        category="Craie", gamma=18.0, nature="argile",
        qc_min=1.5, qc_max=5.0, alpha_qc=3.0, rf_typ=1.5,
        E_min=15.0, E_max=60.0, M_min=20.0, M_max=81.0,
        kp_min=30, kp_max=100, cpt_ok=True,
        desc="Craie remaniée/fissurée (Hesbaye, Tournaisis) — comportement dispersé, "
             "attention aux dissolutions/cavités (karst crayeux)."),
    "Craie saine": dict(
        category="Craie", gamma=20.0, nature="argile",
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=200.0, E_max=1500.0, M_min=269.0, M_max=2019.0,
        kp_min=150, kp_max=500, cpt_ok=False,
        desc="Craie compacte non remaniée. Souvent refus au pénétromètre : caractériser "
             "par carottage/RQD ou essai de plaque plutôt que par CPT."),
    "Calcaire fracturé / altéré": dict(
        category="Calcaire", gamma=21.0, nature="argile",
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=100.0, E_max=800.0, M_min=135.0, M_max=1077.0,
        kp_min=100, kp_max=400, cpt_ok=False,
        desc="Massif calcaire fissuré ou altéré en surface. Grande dispersion : "
             "attention aux karst/cavités, RQD indispensable."),
    "Calcaire sain": dict(
        category="Calcaire", gamma=23.0, nature="argile",
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=2000.0, E_max=15000.0, M_min=2692.0, M_max=20192.0,
        kp_min=1000, kp_max=3000, cpt_ok=False,
        desc="Massif rocheux sain, peu fracturé. Refus au pénétromètre — "
             "caractérisation par RQD/GSI/essai en place."),
    "Schiste houiller décomposé (W4-W5)": dict(
        category="Schiste houiller", gamma=18.0, nature="argile",
        qc_min=0.5, qc_max=3.0, alpha_qc=2.5, rf_typ=2.0,
        E_min=5.0, E_max=30.0, M_min=6.7, M_max=40.0,
        kp_min=10, kp_max=50, cpt_ok=True,
        desc="Roche entièrement à fortement décomposée (aspect de sol résiduel), "
             "classification ISO 14689 W4-W5. Comportement proche d'un sol fin ferme : "
             "un CPT reste indicatif, à recouper avec le log de sondage."),
    "Schiste houiller altéré (W3)": dict(
        category="Schiste houiller", gamma=20.0, nature="argile",
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=100.0, E_max=800.0, M_min=135.0, M_max=1077.0,
        kp_min=100, kp_max=400, cpt_ok=False,
        desc="Roche modérément altérée (W3), matrice affaiblie mais structure "
             "rocheuse conservée. Refus probable au CPT : caractériser par RQD/"
             "pressiomètre. Grande dispersion selon le degré de fracturation."),
    "Schiste houiller sain (W1-W2)": dict(
        category="Schiste houiller", gamma=25.0, nature="argile",
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=1000.0, E_max=8000.0, M_min=1346.0, M_max=10769.0,
        kp_min=800, kp_max=3000, cpt_ok=False,
        desc="Roche saine à faiblement altérée (W1-W2), massif carbonifère typique "
             "des bassins wallons. Refus au pénétromètre — caractériser par RQD/GSI "
             "ou essai de plaque ; anisotropie de feuilletage à prendre en compte."),
    "Grès altéré": dict(
        category="Grès", gamma=20.0, nature="argile",
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=150.0, E_max=1000.0, M_min=202.0, M_max=1346.0,
        kp_min=150, kp_max=500, cpt_ok=False,
        desc="Grès fracturé/altéré en surface. RQD recommandé."),
    "Grès sain": dict(
        category="Grès", gamma=24.0, nature="argile",
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=3000.0, E_max=20000.0, M_min=4038.0, M_max=26923.0,
        kp_min=1500, kp_max=4000, cpt_ok=False,
        desc="Massif rocheux sain. Refus au pénétromètre — RQD/GSI ou essai en place."),
    "Personnalisé": dict(
        category="—", gamma=None, nature="sable",
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=None, E_max=None, M_min=None, M_max=None,
        kp_min=None, kp_max=None, cpt_ok=True,
        desc="Valeurs saisies manuellement — aucune valeur suggérée."),
}

ROCK_CATEGORIES = {"Craie", "Calcaire", "Schiste houiller", "Grès"}

# Mots-clés -> type SOIL_DB. ORDRE IMPORTANT : du plus spécifique au plus
# générique. Chaque variante altérée porte ses formes accentuée ET non
# accentuée ; le repli générique d'une roche pointe vers la variante
# ALTÉRÉE (hypothèse prudente et la plus probable à profondeur de
# fondation).
_SOIL_KEYWORDS = [
    (("remblai", "debris", "débris", "bricaillon", "revetement", "revêtement",
      "terre vegetale", "terre végétale", "asphalte"), "Remblais / terre végétale"),
    (("tourbe", "organique"), "Tourbe"),
    (("argile sableuse", "sable argileux", "argilo-sableux", "sablo-argileux"),
     "Sable argileux / argile sableuse"),
    (("boom", "ypres", "yprési", "surconsolid"), "Argile surconsolidée (Boom/Ypresienne)"),
    (("argile tres molle", "argile très molle"), "Argile très molle"),
    (("argile molle", "argile moyenne"), "Argile molle à moyenne"),
    (("argile ferme", "argile raide", "argile"), "Argile ferme / raide"),
    (("limon", "loess", "silt"), "Limon (loess)"),
    (("gravier", "grave", "graveleux"), "Sable graveleux / grave compacte"),
    (("sable lache", "sable lâche", "sable peu compact"), "Sable lâche"),
    (("sable dense", "sable tres compact", "sable très compact"), "Sable dense"),
    (("sable",), "Sable moyennement compact"),
    # --- roches : variantes altérées AVANT le repli générique ---
    (("craie alter", "craie altér", "craie remani", "craie fissur"), "Craie altérée"),
    (("craie saine", "craie compacte"), "Craie saine"),
    (("craie",), "Craie altérée"),
    (("calcaire alter", "calcaire altér", "calcaire fractur", "calcaire karst"),
     "Calcaire fracturé / altéré"),
    (("calcaire sain", "calcaire compact"), "Calcaire sain"),
    (("calcaire",), "Calcaire fracturé / altéré"),
    (("schiste decompos", "schiste décompos", "w4", "w5"),
     "Schiste houiller décomposé (W4-W5)"),
    (("schiste alter", "schiste altér", "w3"), "Schiste houiller altéré (W3)"),
    (("schiste sain", "w1", "w2"), "Schiste houiller sain (W1-W2)"),
    (("schiste", "houiller"), "Schiste houiller altéré (W3)"),
    (("gres alter", "grès altér", "gres fractur", "grès fractur"), "Grès altéré"),
    (("gres sain", "grès sain"), "Grès sain"),
    (("gres", "grès"), "Grès altéré"),
]


def soil_types_list():
    return ["—"] + list(SOIL_DB.keys())


def match_soil_type(label: str) -> str:
    """Mappe un libellé libre (log de forage, import) vers le type SOIL_DB
    le plus proche. '—' si vide, 'Personnalisé' si aucun mot-clé ne sort."""
    low = (label or "").strip().lower()
    if not low:
        return "—"
    if label in SOIL_DB:
        return label
    for keys, target in _SOIL_KEYWORDS:
        if any(k in low for k in keys):
            return target
    return "Personnalisé"


def _mid(lo, hi):
    if lo is None or hi is None:
        return None
    return round((lo + hi) / 2.0, 1)


def soil_default_qc(soil_type: str):
    d = SOIL_DB.get(soil_type)
    if not d or not d.get("cpt_ok", False):
        return None
    return _mid(d.get("qc_min"), d.get("qc_max"))


def soil_default_Rf(soil_type: str):
    d = SOIL_DB.get(soil_type)
    return d.get("rf_typ") if d else None


def soil_default_E(soil_type: str):
    d = SOIL_DB.get(soil_type)
    return _mid(d.get("E_min"), d.get("E_max")) if d else None


def soil_default_M(soil_type: str):
    d = SOIL_DB.get(soil_type)
    return _mid(d.get("M_min"), d.get("M_max")) if d else None


def soil_gamma(soil_type: str, defaut=19.0):
    d = SOIL_DB.get(soil_type) or {}
    g = d.get("gamma")
    return float(g) if g else defaut


def soil_nature(soil_type: str):
    d = SOIL_DB.get(soil_type) or {}
    return d.get("nature", "sable")


def is_rock(soil_type: str) -> bool:
    d = SOIL_DB.get(soil_type)
    return bool(d and d.get("category") in ROCK_CATEGORIES)


def suggest_M_from_qc(qc_MPa, soil_type: str, nu=0.30):
    """M ≈ α·qc converti en module oedométrique. None si non pertinent."""
    d = SOIL_DB.get(soil_type, {})
    alpha = d.get("alpha_qc")
    if alpha is None or not qc_MPa or qc_MPa <= 0:
        return None
    E = alpha * float(qc_MPa)
    return round(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)), 1)
