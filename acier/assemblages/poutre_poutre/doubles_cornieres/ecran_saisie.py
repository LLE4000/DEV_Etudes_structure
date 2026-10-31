# -*- coding: utf-8 -*-
"""La carte de données des doubles cornières d'âme — les DONNÉES seulement :
le kit (``acier/assemblages/carte.py``) rend la carte sur les trois formats.

Charte ergonomique du 31/10/2026 (docs/assemblages/ERGONOMIE.md §4.1) :

    EN-TÊTE     [ Vérification | Prédimensionnement ] ?   (+ Taux cible · Ø min · Ø max en prédim)
    PROFILÉS    | Principale ▾ | Nuance ▾ | Secondaire ▾ | Nuance ▾ |   ↳ perso (si Personnalisé)
    CORNIÈRES   | Cornière ▾ | Nuance ▾ | Grande aile côté ▾ (cornière inégale ou Personnalisé) |
      ↳ perso    | Grande aile | Petite aile | Épaisseur | Congé |       (si Personnalisé)
    BOULONS     | Boulon ▾ | Classe ▾ | Trou ▾ ? | Catégorie ▾ ? |
      ↳ AILES A (groupe P)   | Fixation ▾ | Rangées n1 | Files n2 ▾ |
      ↳ AILES B (groupe S) ? | Fixation ▾ | Rangées n1 | Files n2 ▾ |
      ↳ glissement (cat ≠ A) | μ | ks (Tab. 3.6) | ELS / ELU ? (cat B) |
    EFFORTS ELU ? | VEd (kN) | NEd (kN) | HEd (kN) | MEd (kNm) |
    ▸ Paramètres avancés   ▸ Cartouche (note et plan)

Le bloc MODE ET FIXATIONS disparaît : le mode est dans l'en-tête, chaque
fixation est la première case du groupe de boulons qu'elle gouverne ; la
VISSERIE sort au cartouche. La géométrie (entraxes, pinces, hauteur et
position des cornières, grugeage, jeu, décalage, gorges, retours) ne se
modifie que sur le dessin (P6). Une forme par clé : ``LIBELLE`` est lu par
la carte, les fenêtres d'objet du dessin, le panneau Cotes et les chips
d'alerte (P5) ; un « ? » n'existe que pour une entrée de ``AIDES`` (P3).

Clés de session ``asm_<clé>`` : source unique, écrite AVANT l'instanciation
des widgets (le dessin est rendu avant la carte)."""
import streamlit as st

from acier.assemblages import carte as kit
from acier.assemblages.carte import L
from acier.assemblages.ui_commun import titre_bloc, _esc, C_TRAIT
from acier.bibliotheques import ORI_P, ORI_S
from .entrees import (mode_predim, PILOTEES_PAR_PREDIM, CHAMPS, UNITE, champ_visible, perso_P, perso_S, perso_C,
                      _cat_non_A, _acier_perso, boulonne_P, boulonne_S, VISSERIE_DEFAUT, MODE_VERIF, MODE_PREDIM)
from .schemas import CLE_PAR_COTE

PREFIXE = "asm_"

# Pas des champs numériques (convenance du compteur ; la saisie reste libre)
PAS = {"V_Ed": 5.0, "N_Ed": 5.0, "H_Ed": 5.0, "M_Ed": 1.0,
       "mu_s": 0.05, "k_s": 0.05, "k_ser": 0.05, "g_M0": 0.05, "g_M2n": 0.05, "g_M2": 0.05,
       "g_M3": 0.05, "g_M3s": 0.05, "k_rot": 0.05, "eta_c": 0.05, "k_e1": 0.1, "k_p1": 0.1,
       "k_e2": 0.1, "bw_u": 0.05, "fy_u": 5.0, "fu_u": 5.0,
       "LC_u": 5.0, "z_C": 5.0, "p1S_u": 5.0, "p2_S": 5.0, "e1S_u": 5.0, "e2b_u": 5.0,
       "p1P_u": 5.0, "p2_P": 5.0, "e1P_u": 5.0, "gA_u": 5.0, "d_top": 5.0, "d_nt": 5.0,
       "d_nb": 5.0, "l_n": 5.0, "k1_u": 5.0, "k2_u": 5.0, "hP_u": 5.0, "bP_u": 5.0,
       "hS_u": 5.0, "bS_u": 5.0, "lh_P": 5.0, "lh_S": 5.0,
       "twP_u": 0.5, "tfP_u": 0.5, "rP_u": 0.5, "twS_u": 0.5, "tfS_u": 0.5, "rS_u": 0.5,
       "kt_u": 0.5, "kr_u": 0.5, "a_P": 0.5, "a_S": 0.5}

# Libellés sans unité (≤ 16 dans une rangée de 4, ≤ 20 dans une rangée de 3 ou une continuation) ;
# l'unité vient de UNITE.
_LIBELLE = {
    "mode_calc": "Mode",
    "prof_P": "Principale", "nu_P": "Nuance", "prof_S": "Secondaire", "nu_S": "Nuance",
    "hP_u": "h", "bP_u": "b", "twP_u": "tw", "tfP_u": "tf", "rP_u": "r",
    "hS_u": "h", "bS_u": "b", "twS_u": "tw", "tfS_u": "tf", "rS_u": "r",
    "corn_u": "Cornière", "nu_C": "Nuance", "orient": "Grande aile côté",
    "k1_u": "Grande aile", "k2_u": "Petite aile", "kt_u": "Épaisseur", "kr_u": "Congé",
    "boulon_u": "Boulon", "classe": "Classe", "trou": "Trou", "cat": "Catégorie",
    # « Fixation ailes A / B » et non « Fixation » : la fenêtre Cornières et l'alerte « soudée des deux
    # côtés » montrent les DEUX fixations ensemble — une forme par clé (P5), jamais deux « Fixation ».
    "fix_P": "Fixation ailes A", "n1P_u": "Rangées n1", "n2P_u": "Files n2",
    "fix_S": "Fixation ailes B", "n1S_u": "Rangées n1", "n2S_u": "Files n2",
    "mu_s": "μ", "k_s": "ks (Tab. 3.6)", "k_ser": "ELS / ELU",
    "V_Ed": "VEd", "N_Ed": "NEd", "H_Ed": "HEd", "M_Ed": "MEd",
    "r_n": "Rayon grugeage", "lt_ok": "Déversement empêché",
    "d0_u": "d0 imposé", "filet": "Filetage cisaillé",
    "g_M0": "γM0", "g_M2": "γM2 boulons", "g_M2n": "γM2 nettes", "g_M3": "γM3 (ELU)", "g_M3s": "γM3,ser (ELS)",
    "opt_exc": "Excentricité gA", "k_rot": "k sur Fv,Rd (P)", "opt_blf": "βLf joints longs", "expo": "Intempéries",
    "fy_u": "fy", "fu_u": "fu", "bw_u": "βw",
    "eta_c": "Taux cible", "k_e1": "e1 = k·d0", "k_p1": "p1 = k·d0", "k_e2": "e2 = k·d0",
    "pd_dmin": "Ø minimal", "pd_dmax": "Ø maximal",
    "id_projet": "Projet", "id_rep": "Repère", "id_red": "Rédacteur", "id_date": "Date",
    "visserie": "Visserie par boulon",
    # géométrie : fenêtres d'objet, panneau Cotes, chips d'alerte
    "e2b_u": "e2,b", "p2_S": "p2 groupe S", "lh_S": "ℓh ailes B", "g_h": "gh", "l_n": "c",
    "e1S_u": "e1 groupe S", "p1S_u": "p1 groupe S", "LC_u": "hc", "z_C": "zc",
    "d_nt": "dc,sup", "d_nb": "dc,inf", "e1P_u": "e1 groupe P", "p1P_u": "p1 groupe P", "d_top": "Δz",
    "a_S": "a ailes B", "gA_u": "gA", "p2_P": "p2 groupe P", "a_P": "a ailes A", "lh_P": "ℓh ailes A",
    # tête de la colonne dessin
    "ui_niveau": "Niveau de cotation", "ui_composant": "Dessin interactif",
}
LIBELLE = {k: v + (" (" + UNITE[k].strip() + ")" if UNITE.get(k) else "") for k, v in _LIBELLE.items()}

# Les aides (« ? ») — textes de la charte §3 ; une clé absente n'a pas de « ? ».
AIDES = {
    "mode_calc": "Prédimensionnement : boulons, rangées et pièce d'attache proposés (onglet Prédim) puis "
                 "vérifiés ; les champs ◆ sont alors pilotés par la proposition.",
    "trou": "Surdimensionné : Fb,Rd × 0,8 (EN 1993-1-8 Tab. 3.4) et d0 à saisir dans Paramètres avancés › TROUS.",
    "cat": "A : pression diamétrale. B : sans glissement à l'ELS. C : sans glissement à l'ELU (boulons "
           "précontraints 8.8 ou 10.9).",
    "k_ser": "Rapport des efforts ELS / ELU ; 1,0 = hypothèse sécuritaire.",
    "r_n": "Information d'exécution : non utilisé dans les résistances ; dessiné sur l'élévation.",
    "lt_ok": "Poutre secondaire maintenue au déversement : condition de la règle de stabilité locale du "
             "grugeage (MSB P5 §4.2.5).",
    "d0_u": "Vide = d0 normal du boulon (Tab. 3.3) ; à saisir pour un trou surdimensionné.",
    "opt_exc": "Non : pratique ECCS n°126 / MSB P5. Oui : variante conservatrice, M = (VEd/2)·eP par cornière.",
    "k_rot": "0,8 : MSB P5 §4.2.1.2 / SCI P358 ; 1,0 pour désactiver.",
    "opt_blf": "βLf des assemblages longs, EN 1993-1-8 §3.8 (Lj > 15d). Les guides ne l'appliquent pas aux "
               "attaches d'âme ; Oui = variante conservatrice.",
    "expo": "Assemblage exposé aux intempéries : active les pinces maximales 4t + 40 mm (Tab. 3.3).",
    "visserie": "Rondelles et écrous par boulon, portés au cartouche du plan. Ex. : « 1 rondelle + 2 écrous "
                "(contre-écrou) », « 2 rondelles + 1 écrou ».",
    "ui_composant": "Décoché : dessins fixes ; les cotes se modifient dans le panneau « Cotes » en tête de la "
                    "colonne de droite.",
    # aides portées par les fenêtres d'objet et le panneau Cotes (géométrie)
    "d_top": "0 = dessus des semelles au même niveau ; aussi par glissement vertical de la poutre secondaire.",
    "d_nt": "0 = semelle supérieure non grugée.",
    "d_nb": "0 = semelle inférieure non grugée.",
    "g_h": "Jeu entre l'âme porteuse et l'about de la secondaire ; aussi par glissement horizontal de la poutre.",
    "gA_u": "Trusquinage : talon de la cornière → 1re file.",
    "a_P": "Cordon vertical en bout d'aile sur toute la hauteur hc.",
    # aides de BLOC (titre_bloc(help=)) et de sous-titre « ↳ » : les conventions communes, dites une fois
    "EFFORTS ELU": "Efforts dans la secondaire : NEd > 0 traction ; HEd perpendiculaire à l'âme ; MEd à la face "
                   "de l'âme porteuse (articulé : 0, pris en valeur absolue).",
    "AILES B": "Ailes B sur l'âme de la secondaire (groupe S, deux plans). Soudées des deux côtés : assemblage "
               "peu déformable, à réserver à des cas particuliers.",
    # « COEFFICIENTS » (12 lettres) ne tient pas dans la case des titres de 76 px (charte §2 : ≤ 11
    # caractères, jamais coupé) : le sous-bloc s'appelle « COEFF. γ », son « ? » dit l'ANB une fois.
    "COEFF. γ": "Valeurs recommandées de l'EN 1993-1-8 ; γM0 selon l'ANB belge (1,00). γM2 boulons : boulons, "
                "pression diamétrale, soudures ; γM2 nettes : sections nettes.",
    "PRÉDIM": "Pinces et entraxes proposés en multiples de d0 ; minima normatifs e ≥ 1,2·d0, p1 ≥ 2,2·d0 "
              "(Tab. 3.3). Règles de détail modifiables.",
}

# Options courtes à l'AFFICHAGE ; les valeurs brutes (fichiers, tests) ne changent pas (P4).
AFFICHAGE = {"mode_calc": {MODE_VERIF: "Vérification", MODE_PREDIM: "Prédimensionnement"},
             "orient": {ORI_P: "Principale", ORI_S: "Secondaire"}}

# Ce qui se modifie sur le dessin — jamais dans la carte (CLE_PAR_COTE porte les 19 cotes, lh_P compris)
GEOMETRIE_DESSIN = frozenset(CLE_PAR_COTE.values())

EN_TETE = [("mode_calc", None), ("eta_c", mode_predim), ("pd_dmin", mode_predim), ("pd_dmax", mode_predim)]
SEGMENTS = frozenset({"mode_calc"})

# Sous-titres des continuations du bloc BOULONS : une fixation gouverne un groupe de boulons
SUITE_A, SUITE_B = "AILES A (groupe P)", "AILES B (groupe S)"

BLOCS = [
    ("PROFILÉS", [L(["prof_P", "nu_P", "prof_S", "nu_S"]),
                  L(["hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"], perso_P, "principale"),
                  L(["hS_u", "bS_u", "twS_u", "tfS_u", "rS_u"], perso_S, "secondaire")]),
    # orient n'est rendu que pour une cornière inégale ou Personnalisé (entrees.corniere_inegale)
    ("CORNIÈRES", [L(["corn_u", "nu_C", "orient"]),
                   L(["k1_u", "k2_u", "kt_u", "kr_u"], perso_C, "perso")]),
    # n1 / n2 d'un groupe ne sont rendus que si sa fixation est boulonnée (entrees.GROUPES)
    ("BOULONS", [L(["boulon_u", "classe", "trou", "cat"]),
                 L(["fix_P", "n1P_u", "n2P_u"], None, SUITE_A),
                 L(["fix_S", "n1S_u", "n2S_u"], None, SUITE_B),
                 L(["mu_s", "k_s", "k_ser"], _cat_non_A, "glissement")]),
    ("EFFORTS ELU", [L(["V_Ed", "N_Ed", "H_Ed", "M_Ed"])]),
]
AVANCES = [
    ("GRUGEAGE", [L(["r_n", "lt_ok"])]),
    ("TROUS", [L(["d0_u", "filet"])]),
    ("COEFF. γ", [L(["g_M0", "g_M2", "g_M2n"]), L(["g_M3", "g_M3s"], _cat_non_A, "glissement")]),
    ("MODÈLE", [L(["opt_exc", "k_rot", "opt_blf", "expo"])]),
    ("ACIER PERSO", [L(["fy_u", "fu_u", "bw_u"], _acier_perso)]),
    ("PRÉDIM", [L(["k_e1", "k_p1", "k_e2"], mode_predim)]),
]


def _hors_projet_nom(u):
    # la condition même de interface._infos : dans un projet la case est écrasée par le projet
    return not st.session_state.get("nom_projet")


def _hors_projet_date(u):
    return not st.session_state.get("date")


def _avec_boulons(u):
    # la visserie n'a de sens que s'il y a des boulons (condition inchangée)
    return boulonne_P(u) or boulonne_S(u)


CARTOUCHE = [("", [L(["id_rep", "id_red", "visserie"]), L(["id_projet", "id_date"], None, "hors projet")])]

D = kit.Description(
    PREFIXE=PREFIXE, CHAMPS=CHAMPS, champ_visible=champ_visible, LIBELLE=LIBELLE, AIDES=AIDES,
    AFFICHAGE=AFFICHAGE, PAS=PAS, BLOCS=BLOCS, AVANCES=AVANCES, PILOTEES_PAR_PREDIM=PILOTEES_PAR_PREDIM,
    mode_predim=mode_predim, VISSERIE_DEFAUT=VISSERIE_DEFAUT, GEOMETRIE_DESSIN=GEOMETRIE_DESSIN,
    EN_TETE=EN_TETE, SEGMENTS=SEGMENTS, CARTOUCHE=CARTOUCHE,
    VISIBLE_SUPPL={"id_projet": _hors_projet_nom, "id_date": _hors_projet_date, "visserie": _avec_boulons})


def K(k):
    return PREFIXE + k


def valeur_widget(k, v):
    """Lue aussi par interface.py et acier/projet_ui.py (hydratation)."""
    return kit.valeur_widget(D, k, v)


# ------------------------------------------------- briques locales (kit gelé)
# BRIQUES DE KIT MANQUANTES, écrites ici le temps de la phase où le kit est gelé (amendement 7) et
# signalées pour la consolidation : un sous-titre « ↳ » qui porte un « ? » (le kit ne le prévoit que
# sur le titre du bloc) et qui tient sur DEUX lignes dans la case de 76 px (« ↳ AILES A (groupe P) »
# déborderait en nowrap) ; le bloc qui les rend ; le compte des « ? » qui les inclut.
def _aide_suite(suite):
    """Le « ? » d'un sous-titre : AIDES[« AILES B »] pour « AILES B (groupe S) »."""
    return D.AIDES.get(str(suite).split(" (", 1)[0]) if suite else None


def _kit_sous_titre(txt, help=None):
    """Sous-titre gris d'une continuation, coupé seulement avant « ( » (espaces insécables ailleurs) :
    une ligne quand la case est pleine largeur (≤ 1350 px), deux dans la case de 76 px."""
    tete, _, reste = str(txt).partition(" (")
    html = "&nbsp;".join(_esc(tete).split(" ")) + (" " + "&nbsp;".join(_esc("(" + reste).split(" ")) if reste else "")
    st.markdown(f"<div style='font-size:0.72em;font-weight:600;color:{C_TRAIT['mut']};line-height:1.15;"
                f"padding:0 0 9px 0;margin-bottom:16px;'>{html}</div>", unsafe_allow_html=True, help=help)


def _kit_bloc(titre, lignes, u, chauds=frozenset()):
    """Le bloc du kit (mêmes clés lg_ / ls_ / bt_, donc même CSS) dont les sous-titres « ↳ » peuvent
    porter un « ? » et passer sur deux lignes."""
    vis = kit.lignes_visibles(D, lignes, u)
    if not vis:
        return
    ident = kit.slug(titre)
    t = ("🔴 " if kit.contient_chaud(D, lignes, chauds, u) else "") + titre
    for n, (lg, cles) in enumerate(vis):
        suite = lg["suite"]
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom",
                          key=f"{PREFIXE}{'lg' if suite is None else 'ls'}_{ident}_{n}"):
            if n == 0 or suite is not None:
                with st.container(key=f"{PREFIXE}bt_{ident}_{n}", width=kit.LARG_TITRE):
                    if n == 0:
                        titre_bloc(t, help=D.AIDES.get(titre))
                    if suite is not None:
                        _kit_sous_titre("↳ " + suite, help=_aide_suite(suite))
            for k in cles:
                kit.widget(D, k, u, chauds, width="stretch" if suite is None else kit.LARG_SUITE)


def aides_visibles(u):
    """Les « ? » visibles dans la CARTE (kit.aides_visibles) + ceux des sous-titres « ↳ » visibles."""
    out = kit.aides_visibles(D, u)
    for _, lignes in D.BLOCS:
        for lg, _cles in kit.lignes_visibles(D, lignes, u):
            if _aide_suite(lg["suite"]):
                out.append(lg["suite"].split(" (", 1)[0])
    return out


def carte(u, chauds=frozenset()):
    """La carte complète : en-tête, blocs par objet (sous-titres à « ? » par la brique locale), puis
    les deux replis du kit."""
    with st.container(gap=None):
        kit.en_tete(D, u, chauds)
        for titre, lignes in D.BLOCS:
            _kit_bloc(titre, lignes, u, chauds)
    kit.avances(D, u, chauds)
    kit.cartouche(D, u, chauds)
