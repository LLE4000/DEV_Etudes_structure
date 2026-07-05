# ===========================
#  VERSION 2.36
# ===========================
#  poutre.py (Streamlit)
#
#  Évolutions vs 2.35 :
#   1. TAUX D'ARMATURE (application uniquement, pas dans le PDF) :
#      - activé par défaut, case sur la ligne du titre ;
#      - majoration 5 % par défaut ;
#      - nouveaux paramètres : Retour d'étrier (10 cm — compté 2×
#        dans la longueur d'un étrier) et Arrondi supérieur (kg/m³) ;
#      - TA affiché dans l'en-tête de chaque section (droite) +
#        icône ⓘ avec le tableau détaillé du calcul au mètre courant.
#   2. SECTIONS : libellé "Section" + valeur lettre seule (A, B, C...).
#      Migration : "Section A" -> "A". La copie prend toujours la
#      lettre suivante disponible (jamais de "copy").
#   3. Récap armatures : "2Ø16 (402.12 mm²)" (tiret supprimé).
#   4. En-tête : "Version 1.001 (?)" à droite du titre (l'historique
#      des versions viendra plus tard).
#
#  Évolutions vs 2.34 :
#   1. Ajout d'armature d'effort tranchant : défaut = Étrier (2 brins),
#      positionné automatiquement de la 1re à la dernière barre inf.
#      (épingle : barre centrale, modifiable).
#   2. POURCENTAGE D'UTILISATION dans les bandeaux verts/rouges
#      (application uniquement, pas dans le PDF), affiché juste avant
#      l'icône ✅/❌ :
#        hauteur : (hmin + dist axe lit 1)/h · armatures : As dimensionnant
#        (max(As,req ; As,min)) / As choisi · τ : τ/τ_adm ·
#        étriers : pas retenu / pas admissible.  Conforme si ≤ 100 %.
#      En-tête hauteur : "... — (hmin = X cm)".
#   3. "Coefficient acier γs" -> "Coefficient acier ELS".
#
#  Évolutions vs 2.33 :
#   1. ÉTRIERS : suppression du type "Étriers (3 brins)". Migration
#      automatique : une ancienne ligne 3 brins devient 1 étrier
#      (2 brins) + 1 épingle centrale de même Ø — Ast strictement
#      conservé (3 × aire).
#   2. PARAMÈTRES AVANCÉS, nouvelles rubriques :
#      - "Armatures technologiques" : Ø (10 mm) + espacement vertical
#        maximal (30 cm). Utilisées par la coupe du PDF pour placer
#        automatiquement des barres latérales de peau quand l'écart
#        vertical entre lit 1 inf. et lit 1 sup. dépasse l'espacement.
#        DESSIN UNIQUEMENT, aucun calcul modifié.
#      - "Taux d'armature" : case d'activation + % de majoration —
#        préparés pour une évolution future, sans effet pour l'instant.
#   3. Icône ⚙️ : cadre légèrement agrandi (l'icône ne déborde plus).
#   4. NOM DU PDF automatique : AAA_NDC Partie#Indice_Date.pdf
#      (AAA = 3 premières lettres du projet).
#
#  Évolutions vs 2.32 :
#   1. ÉTRIERS / ÉPINGLES POSITIONNÉS INDIVIDUELLEMENT :
#      - le champ "Nbr. cadres" est supprimé : 1 ligne = 1 étrier
#        (ou 1 épingle). Migration automatique : une ancienne ligne
#        avec n cadres est dupliquée en n lignes identiques (Ast et
#        vérifications strictement conservés).
#      - chaque ligne porte une position "de barre X → à barre Y"
#        (barres du lit 1 inférieur). Par défaut, un étrier englobe
#        toutes les barres (1 → n) : comportement actuel conservé.
#        Une épingle est par défaut sur la barre centrale.
#      - ces positions ne servent QU'AU DESSIN (coupe de section du
#        PDF) : aucun calcul (brins, cisaillement, pas) n'est modifié.
#      - interface discrète : les sélecteurs de position n'apparaissent
#        que s'il y a plusieurs barres au lit 1 inférieur, sur la même
#        ligne que le type et le Ø.
#   2. PARAMÈTRES AVANCÉS : la case + le texte sont remplacés par une
#      icône ⚙️ qui ouvre/referme le panneau.
#
#  Évolutions vs 2.31 :
#   1. FIX BUG RECALCUL (étriers rouges à tort) : les champs décimaux
#      FR (pas choisi, moments, V...) ne synchronisaient leur valeur
#      numérique qu'au rendu du widget, c.-à-d. APRÈS les calculs du
#      run -> les vérifications utilisaient la valeur du run précédent.
#      Symptôme : pas correct mais vérification rouge, qui repassait
#      au vert en touchant n'importe quel autre champ. Corrigé par
#      _sync_float_raw_keys() en tout début de run.
#   2. EXPANDERS qui se referment seuls : le libellé des expanders
#      contient l'icône d'état (🟢/🔴) ; quand l'état change, Streamlit
#      considère que c'est un NOUVEL expander et le remet à son état
#      par défaut. Les expanders du dimensionnement sont donc tous à
#      expanded=True (limitation Streamlit : pas de key sur expander).
#   3. INFOS PROJET : la case "Ajouter" est remplacée par un bouton
#      ➕ / ➖ compact.
#   4. EFFORT TRANCHANT : la conclusion "Détermination des étriers"
#      (vert/rouge) est remontée AVANT le tableau de saisie.
#   5. AJOUT DE SECTION : bouton ➕ sur la ligne du nom de section
#      (infobulle "Ajouter une section") ; le bouton "Ajouter une
#      section à vérifier" est supprimé.
#   6. POUTRES : infobulle du 📋 raccourcie en "Copier la poutre" ;
#      le bouton "Ajouter une poutre" est supprimé (la duplication
#      couvre le besoin).
#   7. INFOBULLES PÉDAGOGIQUES : formules + valeurs numériques au
#      survol de h,min, As,req, As,min, As,max et du pas théorique.
#   8. LIBELLÉS sollicitations : "M inf (kN·m)", "M sup (kN·m)",
#      "V (kN)".
#
#  Évolutions vs 2.30 :
#   a. DISTANCE AUTO LIT 1 = enrobage
#                          + (Ø étrier + demi-Ø barre lit 1) arrondi
#                            ensemble au 0,5 cm supérieur
#                          + jeu premier lit.
#      La valeur par défaut est désormais bien affichée (plus de 0,00 :
#      les clés ne sont plus pré-créées à 0 et les anciennes valeurs
#      nulles sont réinitialisées en mode auto).
#   b. SOLLICITATIONS : M inférieur, M supérieur et V sur une seule
#      ligne. Suppression de la case "Ajouter un moment supérieur" et
#      du concept d'effort tranchant réduit (on vérifie plusieurs
#      sections à la place).
#   c. Tableau des lits : suppression du titre de colonne "Lit".
#   d. "Sections" à la même taille que "Caractéristiques de la poutre".
#   e. Largeur / Hauteur : pas de 5 cm.
#   f. Coefficient acier γs : défaut 1,5.
#   g. Bouton 📋 "Copier la poutre" (toutes les données) à droite du
#      titre, à gauche du cadenas — même style que pour les sections.
#
#  Évolutions vs 2.20 :
#   1. LITS D'ARMATURES : présentation en tableau
#         | Lit | Nb barres | Ø | Distance axe lit | Action |
#      - le bouton "+ Lit" est intégré sur la ligne du lit 1,
#        dans la même colonne (même taille) que les poubelles ;
#      - les poubelles restent uniquement pour les lits 2, 3, 4.
#   2. LIBELLÉS : "Distance axe lit / parement" -> "Distance axe lit"
#      (le mot "parement" est supprimé partout).
#   3. DISTANCE AUTO DU LIT 1 : le Ø étrier provient désormais de la
#      configuration des étriers (partie Effort tranchant, Ø max des
#      lignes). Le paramètre avancé "Diamètre étrier" est supprimé.
#         dist lit 1 = enrobage béton
#                    + Ø étrier arrondi au 0,5 cm sup.
#                    + demi-Ø barre lit 1 arrondi au 0,5 cm sup.
#                    + jeu premier lit
#      (le demi-Ø barre est conservé : la distance est mesurée à l'AXE)
#   4. VERROUILLAGE PAR POUTRE : le statut "En cours / Validé" est
#      remplacé par un cadenas 🔓/🔒 propre à chaque poutre. Poutre
#      verrouillée = caractéristiques, sections, sollicitations,
#      dimensionnement, armatures et efforts tranchants bloqués.
#   5. SUPPRESSION du verrouillage par section ("Bloquer le
#      dimensionnement de cette section") : doublon du cadenas poutre.
#   6. "Sollicitations" (colonne gauche) renommé "Sections".
#   7. NOM DE SECTION : l'en-tête devient directement le champ
#      éditable (bloc bordé, plus de double affichage expander+champ).
#   8. yG (centre de gravité des lits, distance axe) affiché dans le
#      résumé "Choix : ...".
#   9. PARAMÈTRES AVANCÉS réorganisés en 3 colonnes :
#      Affichage / Coefficients matériaux / Jeux d'armatures.
#      - Nouveau coefficient acier γs (défaut 1.15) : fyd = fyk / γs
#        (remplace l'ancien fyd = fyk / 1.5).
#      - "Tolérance de dépassement" supprimée.
# ===========================
import streamlit as st
from datetime import datetime
from string import ascii_uppercase
import json
import math
import re
from copy import deepcopy

# ============================================================
#  STYLES BLOCS
# ============================================================
C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌"}

# Données béton (chargées dans show())
BETON_DATA = {}

MAX_LITS = 4  # nombre maximal de lits d'armatures par face

APP_VERSION = "1.001"  # version affichée dans l'en-tête de l'application

RHO_ACIER = 7850.0  # kg/m³ (taux d'armature)

# Largeurs de colonnes du tableau des lits (Lit | Nb | Ø | Dist | Action)
LIT_COLS = [0.6, 1.0, 1.0, 1.2, 0.55]


def open_bloc_left_right(left: str, right: str = "", etat: str = "ok", pct=None):
    """
    Header de bloc : texte à gauche + texte à droite (aligné contre l'icône à droite).
    pct (optionnel) : pourcentage d'utilisation affiché juste avant l'icône.
    NB : le rendu (seule la barre d'en-tête est colorée) repose sur
    l'auto-fermeture du HTML par Streamlit. Ne pas modifier.
    """
    right_html = f"<div style='font-weight:600;opacity:0.9;white-space:nowrap;'>{right}</div>" if right else ""
    pct_html = ""
    if pct is not None:
        try:
            pct_html = f"<div style='font-weight:700;white-space:nowrap;'>{float(pct):.0f}\u202f%</div>"
        except Exception:
            pct_html = ""
    st.markdown(
        f"""
        <div style="
            background-color:{C_COULEURS.get(etat, '#f6f6f6')};
            padding:12px 14px 10px 14px;
            border-radius:10px;
            border:1px solid #d9d9d9;
            margin:10px 0 12px 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="font-weight:700;">{left}</div>
            <div style="display:flex;align-items:center;gap:10px;">
              {right_html}
              {pct_html}
              <div style="font-size:20px;line-height:1;">{C_ICONES.get(etat, '')}</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )


def close_bloc():
    st.markdown("</div>", unsafe_allow_html=True)


def small_italic_label_right(txt: str):
    """Libellé italique aligné à droite pour être collé visuellement à une checkbox."""
    st.markdown(
        f"<div style='text-align:right;font-style:italic;opacity:0.75;white-space:nowrap;padding-right:0px;margin-right:0px;'>{txt}</div>",
        unsafe_allow_html=True,
    )


# ============================================================
#  UTILITAIRES SESSION / CLÉS
# ============================================================
def KB(base: str, beam_id: int) -> str:
    return f"b{beam_id}_{base}"


def KS(base: str, beam_id: int, sec_id: int) -> str:
    return f"b{beam_id}_sec{sec_id}_{base}"


# Clés globales persistées (sauvegarde JSON + épinglage anti-nettoyage)
PERSISTED_GLOBAL_KEYS = {
    "units_len",
    "units_as",
    "gamma_s",              # coefficient acier γs (fyd = fyk / γs)
    "jeu_enrobage_cm",      # "Jeu premier lit (cm)" (clé conservée pour compat anciens JSON)
    "jeu_entre_lits_cm",    # "Jeu entre lits (cm)"
    "techno_d_mm",          # armatures technologiques : Ø (dessin PDF)
    "techno_s_max_cm",      # armatures technologiques : espacement vertical max
    "taux_arm_enable",      # taux d'armature : activation
    "taux_arm_major_pct",   # % de majoration
    "taux_retour_etrier_cm",  # retour d'étrier (compté 2× par étrier)
    "taux_arrondi_kgm3",    # arrondi supérieur du TA (kg/m³)
    "nom_projet",
    "partie",
    "date",
    "indice",
    "chk_infos_projet",
}

# Clés à ne jamais épingler ni sauvegarder (transitoires / widgets boutons)
_TRANSIENT_MARKERS = ("btn", "open_uploader", "show_open_uploader", "pdf_bytes", "show_param_avances")


def _is_transient_key(k: str) -> bool:
    return any(m in k for m in _TRANSIENT_MARKERS)


def _pin_persistent_state():
    """
    FIX PERSISTANCE :
    Streamlit supprime en fin de run l'état des widgets qui n'ont pas été
    rendus (champs conditionnels : M_sup masqué, lits repliés, cisaillement
    quand V=0, infos projet quand la case est décochée...).
    Ré-affecter chaque clé à elle-même en début de run la marque comme
    "gérée par l'application" et empêche ce nettoyage.
    """
    for k in list(st.session_state.keys()):
        if _is_transient_key(k):
            continue
        if re.match(r"^b\d+_", k) or k.startswith("meta_") or k in PERSISTED_GLOBAL_KEYS:
            st.session_state[k] = st.session_state[k]
        elif k.endswith("_raw") and (re.match(r"^b\d+_", k) or k[:-4] in PERSISTED_GLOBAL_KEYS):
            st.session_state[k] = st.session_state[k]


def _sync_float_raw_keys():
    """
    FIX BUG RECALCUL :
    les champs décimaux FR stockent la saisie dans '<clé>_raw' et la
    valeur numérique dans '<clé>'. La valeur numérique n'était mise à
    jour qu'au rendu du widget, c.-à-d. APRÈS les calculs du run en
    cours -> les vérifications utilisaient la valeur du run précédent
    (symptôme : pas correct mais statut rouge jusqu'à la modification
    d'un autre champ). On synchronise donc toutes les clés *_raw vers
    leur clé numérique en tout début de run, AVANT tout calcul.
    """
    for k in list(st.session_state.keys()):
        if not k.endswith("_raw"):
            continue
        base = k[:-4]
        try:
            val = float(str(st.session_state[k]).strip().replace(",", "."))
        except Exception:
            continue
        st.session_state[base] = max(0.0, float(val))


def _fr(x, nd=1):
    """Format FR (virgule décimale) pour les infobulles."""
    try:
        return f"{float(x):.{nd}f}".replace(".", ",")
    except Exception:
        return str(x)


def _ensure_global_defaults():
    """Défauts des paramètres globaux (avant tout calcul / rendu)."""
    st.session_state.setdefault("units_len", "cm")
    st.session_state.setdefault("units_as", "mm²")
    st.session_state.setdefault("jeu_enrobage_cm", 1.0)   # jeu premier lit
    st.session_state.setdefault("jeu_entre_lits_cm", 1.0)

    # Armatures technologiques (dessin PDF uniquement)
    st.session_state.setdefault("techno_d_mm", 10)
    _coerce_int_choice("techno_d_mm", [8, 10, 12, 16], 10)
    st.session_state.setdefault("techno_s_max_cm", 30.0)

    # Taux d'armature (application uniquement)
    st.session_state.setdefault("taux_arm_enable", True)
    st.session_state.setdefault("taux_arm_major_pct", 5.0)
    st.session_state.setdefault("taux_retour_etrier_cm", 10.0)
    st.session_state.setdefault("taux_arrondi_kgm3", 5)

    # Coefficient acier γs (défaut 1.5)
    try:
        gs = float(st.session_state.get("gamma_s", 1.5) or 1.5)
    except Exception:
        gs = 1.5
    if gs <= 0:
        gs = 1.5
    st.session_state["gamma_s"] = gs


# ============================================================
#  RESET
# ============================================================
def _reset_module():
    current_page = st.session_state.get("page")
    st.session_state.clear()
    if current_page:
        st.session_state.page = current_page
    st.rerun()


# ============================================================
#  SAISIE DÉCIMALE FR (texte)
# ============================================================
def float_input_fr_simple(label, key, default=0.0, min_value=0.0, disabled: bool = False,
                          label_visibility: str = "visible"):
    """
    Champ décimal FR (virgule). La valeur numérique vit dans `key`,
    la saisie texte dans `key_raw`.
    """
    if key not in st.session_state:
        st.session_state[key] = float(default)
    raw_key = f"{key}_raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = f"{float(st.session_state[key]):.2f}".replace(".", ",")

    raw = st.text_input(label, key=raw_key, disabled=disabled, label_visibility=label_visibility)

    try:
        val = float(str(raw).strip().replace(",", "."))
    except Exception:
        val = float(st.session_state[key])

    val = max(min_value, val)
    st.session_state[key] = float(val)
    return val


# ============================================================
#  COERCITIONS (pré-rendu uniquement)
# ============================================================
def _coerce_int_choice(key: str, options: list, default: int):
    """Force une valeur de session à être un entier appartenant aux options.
    À n'appeler qu'AVANT l'instanciation du widget correspondant."""
    cur = st.session_state.get(key, default)
    try:
        cur = int(float(cur))
    except Exception:
        cur = default
    if cur not in options:
        cur = default
    if st.session_state.get(key) != cur:
        st.session_state[key] = cur


# ============================================================
#  NOMS DE SECTIONS PAR LETTRES (A, B, ..., Z, AA, AB, ...)
# ============================================================
def _letter_sequence():
    for c in ascii_uppercase:
        yield c
    for a in ascii_uppercase:
        for b in ascii_uppercase:
            yield a + b


def _next_section_name(beam_id: int) -> str:
    """Première lettre non utilisée dans la poutre (A, B, ..., AA, AB...)."""
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    used = set()
    for s in beam.get("sections", []):
        sid = int(s.get("id"))
        used.add(str(st.session_state.get(f"meta_b{beam_id}_nom_{sid}", s.get("nom", ""))).strip())
        used.add(str(s.get("nom", "")).strip())
    for L in _letter_sequence():
        if L not in used:
            return L
    return f"S{len(beam.get('sections', [])) + 1}"  # repli improbable


# ============================================================
#  POUTRES / SECTIONS : INIT / ADD / DELETE / DUPLICATE / COPY
# ============================================================
def _init_beams_if_needed():
    if "beams" not in st.session_state or not isinstance(st.session_state.beams, list) or len(st.session_state.beams) == 0:
        st.session_state.beams = [{"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "A"}]}]

    for b in st.session_state.beams:
        b["id"] = int(b.get("id", 0))
        b["nom"] = str(b.get("nom", f"Poutre {b['id']}"))
        if "sections" not in b or not isinstance(b["sections"], list) or len(b["sections"]) == 0:
            b["sections"] = [{"id": 1, "nom": "A"}]
        for s in b["sections"]:
            s["id"] = int(s.get("id", 0))
            s["nom"] = str(s.get("nom", f"Section {s['id']}"))

    if not any(int(b.get("id", 0)) == 1 for b in st.session_state.beams):
        st.session_state.beams.insert(0, {"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "A"}]})

    for b in st.session_state.beams:
        if not any(int(s.get("id", 0)) == 1 for s in b["sections"]):
            b["sections"].insert(0, {"id": 1, "nom": "A"})

    # Synchronisation des noms (labels d'expander à jour immédiatement)
    for b in st.session_state.beams:
        bid = int(b["id"])
        key_nom = f"meta_beam_nom_{bid}"
        if key_nom not in st.session_state:
            st.session_state[key_nom] = str(b.get("nom", f"Poutre {bid}"))
        b["nom"] = str(st.session_state.get(key_nom, b.get("nom")))

        for s in b.get("sections", []):
            sid = int(s["id"])
            key_snom = f"meta_b{bid}_nom_{sid}"
            if key_snom not in st.session_state:
                st.session_state[key_snom] = str(s.get("nom", "A"))
            raw = str(st.session_state.get(key_snom, s.get("nom", "A")))
            # Migration v2.36 : "Section A" -> "A" (le libellé "Section"
            # est désormais affiché à part).
            if raw.lower().startswith("section "):
                raw = raw[8:].strip() or raw
                st.session_state[key_snom] = raw
            s["nom"] = raw

    # Defaults + migrations
    for b in st.session_state.beams:
        _ensure_defaults_for_beam(int(b["id"]))


def _next_beam_id() -> int:
    ids = [int(b.get("id", 0)) for b in st.session_state.beams]
    return (max(ids) + 1) if ids else 1


def _next_section_id(beam_id: int) -> int:
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    ids = [int(s.get("id", 0)) for s in beam["sections"]]
    return (max(ids) + 1) if ids else 1


DIAM_OPTS = [6, 8, 10, 12, 16, 20, 25, 32, 40]
SHEAR_DIAM_OPTS = [6, 8, 10, 12]


def _migrate_second_lit(beam_id: int, sec_id: int, which: str):
    """
    Migration des anciennes clés 'second lit' (checkbox + *_2) vers le
    système multi-lits. Idempotent, fonctionne aussi pour les anciens
    fichiers JSON rechargés.
    """
    old_flag = KS(f"ajouter_second_lit_{which}", beam_id, sec_id)
    nkey = KS(f"nlits_{which}", beam_id, sec_id)
    if bool(st.session_state.get(old_flag, False)):
        if int(st.session_state.get(nkey, 1) or 1) < 2:
            st.session_state[nkey] = 2
            st.session_state[KS(f"n_as_{which}_l2", beam_id, sec_id)] = int(
                st.session_state.get(KS(f"n_as_{which}_2", beam_id, sec_id), 2) or 2
            )
            st.session_state[KS(f"ø_as_{which}_l2", beam_id, sec_id)] = int(
                st.session_state.get(KS(f"ø_as_{which}_2", beam_id, sec_id), 16) or 16
            )
        st.session_state[old_flag] = False


def _migrate_shear_cadres(beam_id: int, sec_id: int):
    """
    Migration v2.33 : 1 ligne = 1 étrier/épingle. Les anciennes lignes
    avec 'Nbr. cadres' n>1 sont dupliquées en n lignes identiques
    (Ast total et dessin strictement conservés). Idempotent : les clés
    *_n sont retirées au passage.
    """
    nk = KS("shear_n_lines", beam_id, sec_id)
    n_lines = max(1, int(st.session_state.get(nk, 1) or 1))
    lines = []
    expanded = False
    for i in range(n_lines):
        typ = st.session_state.get(KS(f"shear_line{i}_type", beam_id, sec_id), "Étriers (2 brins)")
        d = st.session_state.get(KS(f"shear_line{i}_d", beam_id, sec_id), 8)
        f = st.session_state.get(KS(f"shear_line{i}_from", beam_id, sec_id), None)
        t = st.session_state.get(KS(f"shear_line{i}_to", beam_id, sec_id), None)
        n_raw = st.session_state.pop(KS(f"shear_line{i}_n", beam_id, sec_id), None)
        try:
            n = max(1, int(float(n_raw))) if n_raw is not None else 1
        except Exception:
            n = 1
        if n > 1:
            expanded = True
        # Migration v2.34 : "Étriers (3 brins)" n'existe plus ->
        # 1 étrier (2 brins) + 1 épingle centrale de même Ø (Ast conservé).
        if "3 brins" in str(typ):
            expanded = True
            try:
                mid = (int(float(f)) + int(float(t))) // 2 if (f is not None and t is not None) else None
            except Exception:
                mid = None
            for _ in range(n):
                lines.append(("Étriers (2 brins)", d, f, t))
                lines.append(("Épingles (1 brin)", d, mid, mid))
            continue
        for _ in range(n):
            lines.append((typ, d, f, t))
    if not expanded:
        return
    for i, (typ, d, f, t) in enumerate(lines):
        st.session_state[KS(f"shear_line{i}_type", beam_id, sec_id)] = typ
        st.session_state[KS(f"shear_line{i}_d", beam_id, sec_id)] = d
        if f is not None:
            st.session_state[KS(f"shear_line{i}_from", beam_id, sec_id)] = f
        if t is not None:
            st.session_state[KS(f"shear_line{i}_to", beam_id, sec_id)] = t
    st.session_state[nk] = len(lines)


def _ensure_defaults_for_beam(beam_id: int):
    # Poutre
    st.session_state.setdefault(KB("b", beam_id), 20)
    st.session_state.setdefault(KB("h", beam_id), 40)
    st.session_state.setdefault(KB("enrobage_beton", beam_id), 3.0)

    if BETON_DATA:
        default_beton = "C30/37" if "C30/37" in BETON_DATA else list(BETON_DATA.keys())[0]
        st.session_state.setdefault(KB("beton", beam_id), default_beton)
        if st.session_state.get(KB("beton", beam_id)) not in BETON_DATA:
            st.session_state[KB("beton", beam_id)] = default_beton

    # Acier par poutre : ENTIER 400 ou 500
    fkey = KB("fyk", beam_id)
    cur = st.session_state.get(fkey, 500)
    try:
        cur = int(float(cur))
    except Exception:
        cur = 500
    if cur not in (400, 500):
        cur = 500
    st.session_state[fkey] = cur

    # Verrouillage par poutre (cadenas). Migration de l'ancien statut
    # "En cours / Validé" : Validé -> verrouillé.
    st.session_state.setdefault(KB("lock_data", beam_id), False)
    old_statut = st.session_state.pop(KB("statut", beam_id), None)
    if old_statut is not None:
        st.session_state[KB("lock_data", beam_id)] = (str(old_statut) == "Validé")

    # Sections
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    for s in beam.get("sections", []):
        sid = int(s["id"])
        st.session_state.setdefault(KS("M_inf", beam_id, sid), 0.0)
        st.session_state.setdefault(KS("M_sup", beam_id, sid), 0.0)
        st.session_state.setdefault(KS("V", beam_id, sid), 0.0)

        # Migration : M_sup était optionnel (case à cocher). Si la case
        # était décochée, la valeur masquée ne comptait pas -> on la
        # remet à zéro pour ne pas changer les résultats des anciens
        # fichiers. Le concept d'effort tranchant réduit est supprimé.
        old_msup_flag = st.session_state.pop(KS("ajouter_moment_sup", beam_id, sid), None)
        if old_msup_flag is False:
            st.session_state[KS("M_sup", beam_id, sid)] = 0.0
            st.session_state[KS("M_sup", beam_id, sid) + "_raw"] = "0,00"
        st.session_state.pop(KS("ajouter_effort_reduit", beam_id, sid), None)

        for which in ("inf", "sup"):
            st.session_state.setdefault(KS(f"n_as_{which}", beam_id, sid), 2)
            st.session_state.setdefault(KS(f"ø_as_{which}", beam_id, sid), 16)
            st.session_state.setdefault(KS(f"nlits_{which}", beam_id, sid), 1)

            # Migration ancien "second lit"
            _migrate_second_lit(beam_id, sid, which)

            # Coercitions + defaults des lits existants
            _coerce_int_choice(KS(f"ø_as_{which}", beam_id, sid), DIAM_OPTS, 16)
            nl = max(1, min(MAX_LITS, int(st.session_state.get(KS(f"nlits_{which}", beam_id, sid), 1) or 1)))
            st.session_state[KS(f"nlits_{which}", beam_id, sid)] = nl
            for i in range(2, nl + 1):
                st.session_state.setdefault(KS(f"n_as_{which}_l{i}", beam_id, sid), 2)
                st.session_state.setdefault(KS(f"ø_as_{which}_l{i}", beam_id, sid), 16)
                st.session_state.setdefault(KS(f"dist_{which}_l{i}_override", beam_id, sid), False)
                _coerce_int_choice(KS(f"ø_as_{which}_l{i}", beam_id, sid), DIAM_OPTS, 16)

        # Distance axe lit 1 (modifiable par face)
        # NB : clés 'enrob_calc_*' conservées pour compat anciens JSON.
        # Les valeurs ne sont PLUS pré-créées à 0 : _get_dist_lit les
        # initialise directement à la valeur automatique. Migration :
        # une ancienne valeur nulle (ou négative) repasse en mode auto.
        st.session_state.setdefault(KS("enrob_calc_inf_override", beam_id, sid), False)
        st.session_state.setdefault(KS("enrob_calc_sup_override", beam_id, sid), False)
        for which in ("inf", "sup"):
            kv = KS(f"enrob_calc_{which}", beam_id, sid)
            if kv in st.session_state:
                try:
                    v = float(st.session_state.get(kv) or 0.0)
                except Exception:
                    v = 0.0
                if v <= 0.0:
                    st.session_state.pop(kv, None)
                    st.session_state[KS(f"enrob_calc_{which}_override", beam_id, sid)] = False

        # Cisaillement : 1 ligne = 1 étrier/épingle, positionné "de barre
        # X à barre Y" (barres du lit 1 inférieur). Migration : les
        # anciennes lignes avec "Nbr. cadres" n>1 sont dupliquées en n
        # lignes identiques (Ast et vérifications conservés).
        st.session_state.setdefault(KS("shear_n_lines", beam_id, sid), 1)
        st.session_state.setdefault(KS("shear_pas", beam_id, sid), 30.0)
        _migrate_shear_cadres(beam_id, sid)

        n_bars = max(1, int(st.session_state.get(KS("n_as_inf", beam_id, sid), 2) or 2))
        n_lines = max(1, int(st.session_state.get(KS("shear_n_lines", beam_id, sid), 1) or 1))
        st.session_state[KS("shear_n_lines", beam_id, sid)] = n_lines
        for i in range(n_lines):
            st.session_state.setdefault(KS(f"shear_line{i}_type", beam_id, sid), "Étriers (2 brins)")
            st.session_state.setdefault(KS(f"shear_line{i}_d", beam_id, sid), 8)
            _coerce_int_choice(KS(f"shear_line{i}_d", beam_id, sid), SHEAR_DIAM_OPTS, 8)
            # garde-fou : seuls 2 types existent désormais
            if str(st.session_state.get(KS(f"shear_line{i}_type", beam_id, sid), "")) not in ("Étriers (2 brins)", "Épingles (1 brin)"):
                st.session_state[KS(f"shear_line{i}_type", beam_id, sid)] = "Étriers (2 brins)"

            # Positions par défaut : étrier = toutes les barres (1 -> n),
            # épingle = barre centrale. Clamp + swap pré-rendu.
            typ = str(st.session_state.get(KS(f"shear_line{i}_type", beam_id, sid), ""))
            if _brins_from_type(typ) == 1:
                mid = (n_bars + 1) // 2
                df, dt = mid, mid
            else:
                df, dt = 1, n_bars
            kf = KS(f"shear_line{i}_from", beam_id, sid)
            kt = KS(f"shear_line{i}_to", beam_id, sid)
            st.session_state.setdefault(kf, df)
            st.session_state.setdefault(kt, dt)
            try:
                f = max(1, min(n_bars, int(float(st.session_state.get(kf, df)))))
            except Exception:
                f = df
            try:
                t = max(1, min(n_bars, int(float(st.session_state.get(kt, dt)))))
            except Exception:
                t = dt
            if f > t:
                f, t = t, f
            if st.session_state.get(kf) != f:
                st.session_state[kf] = f
            if st.session_state.get(kt) != t:
                st.session_state[kt] = t


def _add_beam():
    new_id = _next_beam_id()
    st.session_state.beams.append({"id": new_id, "nom": f"Poutre {new_id}", "sections": [{"id": 1, "nom": "A"}]})
    st.session_state[f"meta_beam_nom_{new_id}"] = f"Poutre {new_id}"
    st.session_state[f"meta_b{new_id}_nom_1"] = "A"
    _ensure_defaults_for_beam(new_id)


def _delete_beam(beam_id: int):
    if beam_id == 1:
        return
    st.session_state.beams = [b for b in st.session_state.beams if int(b.get("id")) != beam_id]
    prefix = f"b{beam_id}_"
    for k in [k for k in list(st.session_state.keys()) if k.startswith(prefix)]:
        del st.session_state[k]
    st.session_state.pop(f"meta_beam_nom_{beam_id}", None)
    for k in list(st.session_state.keys()):
        if k.startswith(f"meta_b{beam_id}_nom_"):
            del st.session_state[k]


def _duplicate_beam(src_beam_id: int):
    """Disponible si tu veux brancher un bouton 'Dupliquer la poutre' plus tard."""
    src = next(b for b in st.session_state.beams if int(b.get("id")) == src_beam_id)
    new_id = _next_beam_id()
    st.session_state.beams.append({"id": new_id, "nom": f"{src.get('nom','Poutre')} (copie)", "sections": deepcopy(src["sections"])})

    src_prefix = f"b{src_beam_id}_"
    dst_prefix = f"b{new_id}_"
    for k in list(st.session_state.keys()):
        if k.startswith(src_prefix) and not _is_transient_key(k):
            st.session_state[dst_prefix + k[len(src_prefix):]] = deepcopy(st.session_state[k])

    st.session_state[f"meta_beam_nom_{new_id}"] = f"{st.session_state.get(f'meta_beam_nom_{src_beam_id}', src.get('nom','Poutre'))} (copie)"
    for s in src.get("sections", []):
        sid = int(s.get("id"))
        st.session_state[f"meta_b{new_id}_nom_{sid}"] = st.session_state.get(f"meta_b{src_beam_id}_nom_{sid}", s.get("nom", f"Section {sid}"))

    _ensure_defaults_for_beam(new_id)


def _add_section(beam_id: int):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    new_id = _next_section_id(beam_id)
    name = _next_section_name(beam_id)
    beam["sections"].append({"id": new_id, "nom": name})
    st.session_state[f"meta_b{beam_id}_nom_{new_id}"] = name
    _ensure_defaults_for_beam(beam_id)


def _copy_section(beam_id: int, src_sec_id: int):
    """
    Copie intégrale d'une section (sollicitations, armatures inf./sup.,
    lits, distances, cisaillement, options) vers une nouvelle section
    nommée avec la première lettre disponible. Callback on_click.
    """
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    new_id = _next_section_id(beam_id)
    name = _next_section_name(beam_id)
    beam["sections"].append({"id": new_id, "nom": name})

    src_prefix = f"b{beam_id}_sec{src_sec_id}_"
    dst_prefix = f"b{beam_id}_sec{new_id}_"
    for k in list(st.session_state.keys()):
        if k.startswith(src_prefix) and not _is_transient_key(k):
            st.session_state[dst_prefix + k[len(src_prefix):]] = deepcopy(st.session_state[k])

    st.session_state[f"meta_b{beam_id}_nom_{new_id}"] = name
    _ensure_defaults_for_beam(beam_id)


def _delete_section(beam_id: int, sec_id: int):
    if sec_id == 1:
        return
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    beam["sections"] = [s for s in beam["sections"] if int(s.get("id")) != sec_id]
    prefix = f"b{beam_id}_sec{sec_id}_"
    for k in [k for k in list(st.session_state.keys()) if k.startswith(prefix)]:
        del st.session_state[k]
    st.session_state.pop(f"meta_b{beam_id}_nom_{sec_id}", None)


# ============================================================
#  SAVE / LOAD JSON (beams + valeurs)
# ============================================================
def _build_save_payload():
    beams = []
    for b in st.session_state.beams:
        beams.append(
            {
                "id": int(b.get("id")),
                "nom": str(b.get("nom")),
                "sections": [{"id": int(s.get("id")), "nom": str(s.get("nom"))} for s in b.get("sections", [])],
            }
        )

    values = {}
    for k in list(st.session_state.keys()):
        if _is_transient_key(k):
            continue
        if k in PERSISTED_GLOBAL_KEYS or (k.endswith("_raw") and k[:-4] in PERSISTED_GLOBAL_KEYS):
            values[k] = st.session_state[k]
        elif re.match(r"^b\d+_", k):
            values[k] = st.session_state[k]
        elif k.startswith("meta_beam_nom_") or (k.startswith("meta_b") and "_nom_" in k):
            values[k] = st.session_state[k]

    return {"version": "2.36", "beams": beams, "values": values}


def _load_from_payload(payload: dict):
    beams = payload.get("beams", None)
    values = payload.get("values", {})

    if isinstance(beams, list) and len(beams) > 0:
        cleaned = []
        for b in beams:
            try:
                bid = int(b.get("id"))
            except Exception:
                continue
            secs = b.get("sections", [])
            if not isinstance(secs, list) or len(secs) == 0:
                secs = [{"id": 1, "nom": "A"}]
            cleaned_secs = []
            for s in secs:
                try:
                    sid = int(s.get("id"))
                except Exception:
                    continue
                cleaned_secs.append({"id": sid, "nom": str(s.get("nom", f"Section {sid}"))})
            cleaned.append({"id": bid, "nom": str(b.get("nom", f"Poutre {bid}")), "sections": cleaned_secs})
        st.session_state.beams = cleaned if cleaned else [{"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "A"}]}]
    else:
        st.session_state.beams = [{"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "A"}]}]

    if isinstance(values, dict):
        for k, v in values.items():
            if _is_transient_key(k):
                continue
            if k in PERSISTED_GLOBAL_KEYS or (k.endswith("_raw") and k[:-4] in PERSISTED_GLOBAL_KEYS):
                st.session_state[k] = v
            elif re.match(r"^b\d+_", k):
                st.session_state[k] = v
            elif k.startswith("meta_beam_nom_") or (k.startswith("meta_b") and "_nom_" in k):
                st.session_state[k] = v

    _ensure_global_defaults()
    _init_beams_if_needed()


# ============================================================
#  OUTILS CALCUL
# ============================================================
def _bar_area_mm2(diam_mm: float) -> float:
    return math.pi * (diam_mm / 2.0) ** 2


def _status_merge(*states: str) -> str:
    if any(s == "nok" for s in states):
        return "nok"
    if any(s == "warn" for s in states):
        return "warn"
    return "ok"


def _status_icon_label(state: str, label: str) -> str:
    if state == "ok":
        return f"🟢 {label}"
    if state == "warn":
        return f"🟡 {label}"
    return f"🔴 {label}"


def _brins_from_type(type_txt: str) -> int:
    if "3 brins" in type_txt:
        return 3
    if "2 brins" in type_txt:
        return 2
    return 1


def _get_fyk_and_mu_ref(beam_id: int):
    """Acier par poutre : 400 ou 500 (lecture seule — coercition dans _ensure_defaults)."""
    try:
        fyk_i = int(float(st.session_state.get(KB("fyk", beam_id), 500)))
    except Exception:
        fyk_i = 500
    if fyk_i not in (400, 500):
        fyk_i = 500
    return float(fyk_i), str(fyk_i)


def _get_gamma_s() -> float:
    try:
        gs = float(st.session_state.get("gamma_s", 1.5) or 1.5)
    except Exception:
        gs = 1.5
    return gs if gs > 0 else 1.5


def _round_up_to_half_cm(x_cm: float) -> float:
    try:
        return math.ceil(float(x_cm) * 2.0) / 2.0
    except Exception:
        return x_cm


def _stirrup_diam_mm(beam_id: int, sec_id: int) -> float:
    """
    Ø étrier (mm) pris directement dans la configuration des étriers de la
    section (partie Effort tranchant). S'il y a plusieurs lignes, on prend
    le Ø maximal (cas défavorable pour la distance du lit 1).
    """
    n_lines = max(1, int(st.session_state.get(KS("shear_n_lines", beam_id, sec_id), 1) or 1))
    diams = []
    for i in range(n_lines):
        try:
            diams.append(float(st.session_state.get(KS(f"shear_line{i}_d", beam_id, sec_id), 8) or 8))
        except Exception:
            pass
    return max(diams) if diams else 8.0


# ============================================================
#  DISTANCES AXE LIT
# ============================================================
def _get_nlits(beam_id: int, sec_id: int, which: str) -> int:
    try:
        nl = int(st.session_state.get(KS(f"nlits_{which}", beam_id, sec_id), 1) or 1)
    except Exception:
        nl = 1
    return max(1, min(MAX_LITS, nl))


def _lit_bars(beam_id: int, sec_id: int, which: str, i: int):
    """Retourne (n, Ø) du lit i (i>=1)."""
    if i == 1:
        n = int(st.session_state.get(KS(f"n_as_{which}", beam_id, sec_id), 2) or 2)
        d = int(st.session_state.get(KS(f"ø_as_{which}", beam_id, sec_id), 16) or 16)
    else:
        n = int(st.session_state.get(KS(f"n_as_{which}_l{i}", beam_id, sec_id), 2) or 2)
        d = int(st.session_state.get(KS(f"ø_as_{which}_l{i}", beam_id, sec_id), 16) or 16)
    return n, d


def _dist_keys(beam_id: int, sec_id: int, which: str, i: int):
    """Clés (valeur, override) de la distance axe lit i."""
    if i == 1:
        # clés historiques conservées pour la compatibilité des anciens JSON
        return KS(f"enrob_calc_{which}", beam_id, sec_id), KS(f"enrob_calc_{which}_override", beam_id, sec_id)
    return KS(f"dist_{which}_l{i}", beam_id, sec_id), KS(f"dist_{which}_l{i}_override", beam_id, sec_id)


def _auto_dist_lit(beam_id: int, sec_id: int, which: str, i: int) -> float:
    """
    Distance automatique axe lit i (cm).

    Lit 1 :
      enrobage béton
      + (Ø étrier + demi-Ø barre lit 1) arrondi ENSEMBLE au 0,5 cm sup.
        (Ø étrier lu dans la configuration des étriers de la section,
         partie Effort tranchant — plus de paramètre avancé dédié)
      + jeu premier lit
      Ex (étrier Ø8, barre Ø16) : 3,0 + arr(0,8+0,8=1,6)=2,0 + 1,0 = 6,0 cm

    Lit i (i >= 2) :
      distance lit (i-1)   [valeur réelle, override compris]
      + demi-Ø lit (i-1) arrondi au 0,5 cm sup.
      + jeu entre lits (paramètre global)
      + demi-Ø lit i arrondi au 0,5 cm sup.
    """
    if i == 1:
        enrob_beton = float(st.session_state.get(KB("enrobage_beton", beam_id), 3.0) or 3.0)
        d_etrier = _stirrup_diam_mm(beam_id, sec_id)
        jeu1 = float(st.session_state.get("jeu_enrobage_cm", 1.0) or 0.0)
        _, d1 = _lit_bars(beam_id, sec_id, which, 1)
        return (
            enrob_beton
            + _round_up_to_half_cm(d_etrier / 10.0 + d1 / 20.0)
            + jeu1
        )

    prev_dist = _get_dist_lit(beam_id, sec_id, which, i - 1)
    _, d_prev = _lit_bars(beam_id, sec_id, which, i - 1)
    _, d_i = _lit_bars(beam_id, sec_id, which, i)
    jeuL = float(st.session_state.get("jeu_entre_lits_cm", 1.0) or 0.0)
    return (
        prev_dist
        + _round_up_to_half_cm(d_prev / 20.0)
        + jeuL
        + _round_up_to_half_cm(d_i / 20.0)
    )


def _get_dist_lit(beam_id: int, sec_id: int, which: str, i: int) -> float:
    """
    Distance axe lit i effectivement utilisée :
    valeur auto, sauf override manuel de l'utilisateur.
    NB : peut écrire dans la clé du widget -> à n'appeler qu'AVANT le
    rendu des widgets de la section.
    """
    key_val, key_ovr = _dist_keys(beam_id, sec_id, which, i)
    auto_val = _auto_dist_lit(beam_id, sec_id, which, i)

    if key_ovr not in st.session_state:
        st.session_state[key_ovr] = False
    if key_val not in st.session_state:
        st.session_state[key_val] = float(auto_val)

    if not bool(st.session_state.get(key_ovr, False)):
        if abs(float(st.session_state.get(key_val, auto_val) or auto_val) - float(auto_val)) > 1e-9:
            st.session_state[key_val] = float(auto_val)

    try:
        return float(st.session_state.get(key_val, auto_val) or auto_val)
    except Exception:
        return float(auto_val)


def _layers_geometry(beam_id: int, sec_id: int, which: str):
    """
    Pour une face :
      - As_total (mm²)
      - e_cdg (cm) = yG : distance parement -> centre de gravité des lits
        yG = Σ(As,i · y_i) / Σ(As,i)   (positions réelles, overrides compris)
      - detail : chaîne '2Ø16 + 2Ø20 ...'
    """
    nl = _get_nlits(beam_id, sec_id, which)
    parts = []
    As_tot = 0.0
    somme_As_e = 0.0

    for i in range(1, nl + 1):
        n, d = _lit_bars(beam_id, sec_id, which, i)
        e = _get_dist_lit(beam_id, sec_id, which, i)
        As_i = n * _bar_area_mm2(d)
        As_tot += As_i
        somme_As_e += As_i * e
        parts.append(f"{n}Ø{d}")

    e_cdg = (somme_As_e / As_tot) if As_tot > 0 else _get_dist_lit(beam_id, sec_id, which, 1)
    return As_tot, e_cdg, " + ".join(parts)


# ============================================================
#  CISAILLEMENT : aires, résumé, callbacks (1 ligne = 1 étrier)
# ============================================================
def _shear_lines_total_Ast_mm2(beam_id: int, sec_id: int, reduced: bool = False) -> float:
    """Ast = Σ brins × aire(Ø) — 1 ligne = 1 étrier/épingle (v2.33)."""
    n_lines = max(1, int(st.session_state.get(KS("shear_n_lines", beam_id, sec_id), 1) or 1))
    Ast = 0.0
    for i in range(n_lines):
        typ = str(st.session_state.get(KS(f"shear_line{i}_type", beam_id, sec_id), "Étriers (2 brins)"))
        diam = float(st.session_state.get(KS(f"shear_line{i}_d", beam_id, sec_id), 8) or 8)
        Ast += _brins_from_type(typ) * _bar_area_mm2(diam)
    return Ast


def _shear_line_pos(beam_id: int, sec_id: int, i: int):
    """Position (from, to) de la ligne i (barres du lit 1 inférieur)."""
    try:
        f = int(float(st.session_state.get(KS(f"shear_line{i}_from", beam_id, sec_id), 1) or 1))
    except Exception:
        f = 1
    try:
        t = int(float(st.session_state.get(KS(f"shear_line{i}_to", beam_id, sec_id), f) or f))
    except Exception:
        t = f
    return (f, t) if f <= t else (t, f)


def _shear_lines_summary(beam_id: int, sec_id: int, reduced: bool = False) -> str:
    """Résumé compact : lignes identiques regroupées, position affichée
    quand elle n'est pas triviale (étrier partiel, épingle)."""
    n_lines = max(1, int(st.session_state.get(KS("shear_n_lines", beam_id, sec_id), 1) or 1))
    n_bars = max(1, int(st.session_state.get(KS("n_as_inf", beam_id, sec_id), 2) or 2))
    order = []
    counts = {}
    for i in range(n_lines):
        typ = str(st.session_state.get(KS(f"shear_line{i}_type", beam_id, sec_id), "Étriers (2 brins)"))
        diam = int(float(st.session_state.get(KS(f"shear_line{i}_d", beam_id, sec_id), 8) or 8))
        f, t = _shear_line_pos(beam_id, sec_id, i)
        key = (typ, diam, f, t)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1
    parts = []
    for (typ, diam, f, t) in order:
        n = counts[(typ, diam, f, t)]
        lab = f"{n}× {typ} Ø{diam}"
        brins = _brins_from_type(typ)
        full = (f <= 1 and t >= n_bars)
        if brins == 1:
            lab += f" (b{f})" if f == t else f" ({f}→{t})"
        elif not full:
            lab += f" ({f}→{t})"
        parts.append(lab)
    return " + ".join(parts)


def _delete_shear_line(beam_id: int, sec_id: int, reduced: bool, i: int):
    """Callback on_click : mutation légale des clés (avant instanciation des widgets)."""
    nk = KS("shear_n_lines", beam_id, sec_id)
    prefix = "shear_line"
    n_lines = max(1, int(st.session_state.get(nk, 1) or 1))
    if n_lines <= 1 or i <= 0 or i >= n_lines:
        return
    for j in range(i, n_lines - 1):
        for suf in ("type", "d", "from", "to"):
            st.session_state[KS(f"{prefix}{j}_{suf}", beam_id, sec_id)] = st.session_state.get(
                KS(f"{prefix}{j+1}_{suf}", beam_id, sec_id)
            )
    for suf in ("type", "d", "from", "to"):
        st.session_state.pop(KS(f"{prefix}{n_lines-1}_{suf}", beam_id, sec_id), None)
    st.session_state[nk] = n_lines - 1


def _add_shear_line(beam_id: int, sec_id: int, reduced: bool = False):
    nk = KS("shear_n_lines", beam_id, sec_id)
    new_i = max(1, int(st.session_state.get(nk, 1) or 1))
    st.session_state[nk] = new_i + 1
    n_bars = max(1, int(st.session_state.get(KS("n_as_inf", beam_id, sec_id), 2) or 2))
    # Défaut : Étrier (2 brins) englobant toutes les barres (1 -> n).
    st.session_state.setdefault(KS(f"shear_line{new_i}_type", beam_id, sec_id), "Étriers (2 brins)")
    st.session_state.setdefault(KS(f"shear_line{new_i}_d", beam_id, sec_id), 8)
    st.session_state.setdefault(KS(f"shear_line{new_i}_from", beam_id, sec_id), 1)
    st.session_state.setdefault(KS(f"shear_line{new_i}_to", beam_id, sec_id), n_bars)


# ============================================================
#  LITS : callbacks ajout / suppression
# ============================================================
def _add_lit(beam_id: int, sec_id: int, which: str):
    nk = KS(f"nlits_{which}", beam_id, sec_id)
    nl = _get_nlits(beam_id, sec_id, which)
    if nl >= MAX_LITS:
        return
    i = nl + 1
    _, prev_d = _lit_bars(beam_id, sec_id, which, nl)
    st.session_state.setdefault(KS(f"n_as_{which}_l{i}", beam_id, sec_id), 2)
    st.session_state[KS(f"ø_as_{which}_l{i}", beam_id, sec_id)] = prev_d
    st.session_state[KS(f"dist_{which}_l{i}_override", beam_id, sec_id)] = False
    st.session_state.pop(KS(f"dist_{which}_l{i}", beam_id, sec_id), None)  # sera recalculée en auto
    st.session_state[nk] = i


def _delete_lit(beam_id: int, sec_id: int, which: str, i: int):
    """Callback on_click : suppression du lit i (i>=2) avec décalage des suivants."""
    nk = KS(f"nlits_{which}", beam_id, sec_id)
    nl = _get_nlits(beam_id, sec_id, which)
    if i < 2 or i > nl:
        return
    for j in range(i, nl):
        for suf in ("n_as", "ø_as"):
            st.session_state[KS(f"{suf}_{which}_l{j}", beam_id, sec_id)] = st.session_state.get(
                KS(f"{suf}_{which}_l{j+1}", beam_id, sec_id)
            )
        st.session_state[KS(f"dist_{which}_l{j}", beam_id, sec_id)] = st.session_state.get(
            KS(f"dist_{which}_l{j+1}", beam_id, sec_id)
        )
        st.session_state[KS(f"dist_{which}_l{j}_override", beam_id, sec_id)] = bool(
            st.session_state.get(KS(f"dist_{which}_l{j+1}_override", beam_id, sec_id), False)
        )
    for suf in ("n_as", "ø_as", "dist"):
        st.session_state.pop(KS(f"{suf}_{which}_l{nl}", beam_id, sec_id), None)
    st.session_state.pop(KS(f"dist_{which}_l{nl}_override", beam_id, sec_id), None)
    st.session_state[nk] = nl - 1


# ============================================================
#  UI : SECTIONS (SOLLICITATIONS) PAR POUTRE
# ============================================================
def _render_section_inputs(beam_id: int, sec_id: int, disabled: bool):
    c1, c2, c3 = st.columns(3)
    with c1:
        float_input_fr_simple("M inf (kN·m)", key=KS("M_inf", beam_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)
    with c2:
        float_input_fr_simple("M sup (kN·m)", key=KS("M_sup", beam_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)
    with c3:
        float_input_fr_simple("V (kN)", key=KS("V", beam_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)


def render_solicitations_for_beam(beam_id: int, data_locked: bool = False):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    st.markdown("#### Sections")

    for sec in beam.get("sections", []):
        sec_id = int(sec.get("id"))
        sec_name_key = f"meta_b{beam_id}_nom_{sec_id}"
        st.session_state.setdefault(sec_name_key, sec.get("nom", f"Section {sec_id}"))

        # Bloc bordé : l'en-tête EST le champ de nom (plus de double affichage)
        with st.container(border=True):
            cL, cN, cA, cC, cD = st.columns([1.1, 4.3, 0.8, 0.8, 0.8], vertical_alignment="center")
            with cL:
                st.markdown("**Section**")
            with cN:
                st.text_input(
                    "Nom de la section",
                    key=sec_name_key,
                    disabled=data_locked,
                    label_visibility="collapsed",
                )
            with cA:
                st.button(
                    "➕",
                    key=f"add_sec_btn_{beam_id}_{sec_id}",
                    help="Ajouter une section",
                    use_container_width=True,
                    on_click=_add_section,
                    args=(beam_id,),
                    disabled=data_locked,
                )
            with cC:
                st.button(
                    "📋",
                    key=f"copy_sec_btn_{beam_id}_{sec_id}",
                    help="Copier la section (toutes les données)",
                    use_container_width=True,
                    on_click=_copy_section,
                    args=(beam_id, sec_id),
                    disabled=data_locked,
                )
            with cD:
                if sec_id != 1:
                    st.button(
                        "🗑️",
                        key=f"del_sec_btn_{beam_id}_{sec_id}",
                        help="Supprimer la section",
                        use_container_width=True,
                        on_click=_delete_section,
                        args=(beam_id, sec_id),
                        disabled=data_locked,
                    )

            _render_section_inputs(beam_id, sec_id, disabled=data_locked)

    if beam_id != 1:
        st.button(
            "🗑️ Supprimer la poutre",
            key=f"del_beam_btn_{beam_id}",
            on_click=_delete_beam,
            args=(beam_id,),
            disabled=data_locked,
            use_container_width=True,
        )


def _toggle_beam_lock(beam_id: int):
    k = KB("lock_data", beam_id)
    st.session_state[k] = not bool(st.session_state.get(k, False))


def render_caracteristiques_beam(beam_id: int):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)

    beam_name_key = f"meta_beam_nom_{beam_id}"
    st.session_state.setdefault(beam_name_key, beam.get("nom", f"Poutre {beam_id}"))

    lock_key = KB("lock_data", beam_id)
    st.session_state.setdefault(lock_key, False)
    data_locked = bool(st.session_state.get(lock_key, False))

    with st.expander(st.session_state.get(beam_name_key, beam.get("nom", f"Poutre {beam_id}")), expanded=True):
        t1, tC, tL = st.columns([6, 0.8, 0.8], vertical_alignment="center")
        with t1:
            st.markdown("#### Caractéristiques de la poutre")
        with tC:
            st.button(
                "📋",
                key=f"btn_copy_beam_{beam_id}",
                help="Copier la poutre",
                use_container_width=True,
                on_click=_duplicate_beam,
                args=(beam_id,),
            )
        with tL:
            st.button(
                "🔒" if data_locked else "🔓",
                key=f"btn_lock_beam_{beam_id}",
                help="Poutre verrouillée — cliquer pour déverrouiller" if data_locked
                     else "Poutre éditable — cliquer pour verrouiller",
                use_container_width=True,
                on_click=_toggle_beam_lock,
                args=(beam_id,),
            )

        c1, c2, c3 = st.columns([2.6, 1.6, 1.4], vertical_alignment="center")
        with c1:
            st.text_input("Nom de la poutre", key=beam_name_key, disabled=data_locked)
        with c2:
            st.selectbox("Classe de béton", list(BETON_DATA.keys()), key=KB("beton", beam_id), disabled=data_locked)
        with c3:
            st.selectbox("Qualité acier (B)", [400, 500], key=KB("fyk", beam_id), disabled=data_locked)

        cB, cH, cE = st.columns(3)
        with cB:
            st.number_input("Larg. (cm)", min_value=5, max_value=200, step=5, key=KB("b", beam_id), disabled=data_locked)
        with cH:
            st.number_input("Haut. (cm)", min_value=5, max_value=300, step=5, key=KB("h", beam_id), disabled=data_locked)
        with cE:
            st.number_input("Enrob. béton (cm)", min_value=0.0, max_value=20.0, step=0.5, key=KB("enrobage_beton", beam_id), disabled=data_locked)

        render_solicitations_for_beam(beam_id, data_locked=data_locked)


# ============================================================
#  CALCUL DES ÉTATS D'UNE SECTION
# ============================================================
def _dimensionnement_compute_states(beam_id: int, sec_id: int, beton_data: dict):
    beton = str(st.session_state.get(KB("beton", beam_id), "C30/37"))
    if beton not in beton_data:
        beton = list(beton_data.keys())[0]
    fck_cube = beton_data[beton]["fck_cube"]
    alpha_b = beton_data[beton]["alpha_b"]

    # fyd = fyk / γs (γs paramétrable, défaut 1.15)
    fyk, mu_ref = _get_fyk_and_mu_ref(beam_id)
    gamma_s = _get_gamma_s()
    fyd = fyk / gamma_s

    mu_key = f"mu_a{mu_ref}"
    if mu_key not in beton_data[beton]:
        mu_key = "mu_a500" if "mu_a500" in beton_data[beton] else [k for k in beton_data[beton].keys() if k.startswith("mu_a")][0]
    mu_val = beton_data[beton][mu_key]

    b = float(st.session_state.get(KB("b", beam_id), 20))
    h = float(st.session_state.get(KB("h", beam_id), 40))

    # --- Lits : aire totale + centre de gravité (positions réelles) ---
    As_inf_total, e_cdg_inf, inf_detail = _layers_geometry(beam_id, sec_id, "inf")
    As_sup_total, e_cdg_sup, sup_detail = _layers_geometry(beam_id, sec_id, "sup")

    # Distance lit 1 (pour hmin et cisaillement — logique conservée)
    dist_l1_inf = _get_dist_lit(beam_id, sec_id, "inf", 1)
    dist_l1_sup = _get_dist_lit(beam_id, sec_id, "sup", 1)

    # d utile FLEXION : c.d.g. des lits (inf. depuis le bas, sup. depuis le haut)
    d_utile_inf = h - e_cdg_inf  # cm
    d_utile_sup = h - e_cdg_sup  # cm
    # d utile CISAILLEMENT : inchangé (lit 1, min des deux faces)
    d_utile_for_shear = h - min(dist_l1_inf, dist_l1_sup)  # cm

    geom_inf_ok = d_utile_inf > 0.0
    geom_sup_ok = d_utile_sup > 0.0
    geom_shear_ok = d_utile_for_shear > 0.0
    d_calc_inf = max(d_utile_inf, 0.1)
    d_calc_sup = max(d_utile_sup, 0.1)
    d_calc_shear = max(d_utile_for_shear, 0.1)

    M_inf_val = float(st.session_state.get(KS("M_inf", beam_id, sec_id), 0.0) or 0.0)
    M_sup_val = float(st.session_state.get(KS("M_sup", beam_id, sec_id), 0.0) or 0.0)
    V_val = float(st.session_state.get(KS("V", beam_id, sec_id), 0.0) or 0.0)

    # --- Hauteur (logique conservée : distance lit 1 inf.) ---
    M_max = max(M_inf_val, M_sup_val)
    if M_max > 0:
        hmin_calc = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10  # cm
    else:
        hmin_calc = 0.0
    etat_h = "ok" if (hmin_calc + dist_l1_inf <= h) else "nok"

    # --- As min/max ---
    # As,min retenu = max( 0,26·fctm/fyk·b·h ; 0,0013·b·h ; 0,25·As,req face opposée )
    #   (h utilisé partout, en cohérence avec le rapport PDF)
    fck_cyl = float(beton_data[beton].get("fck", 0.8 * fck_cube) or (0.8 * fck_cube))
    fctm = 0.30 * (fck_cyl ** (2.0 / 3.0)) if fck_cyl > 0 else 0.0
    As_min_ec = 0.26 * fctm / fyk * b * h * 1e2     # mm²  (b,h en cm -> ·1e2)
    As_min_plancher = 0.0013 * b * h * 1e2          # mm²
    As_min_base = max(As_min_ec, As_min_plancher)   # partie indépendante de la face
    As_min_formula = As_min_base                    # (compat clé exportée)
    As_max = 0.04 * b * h * 1e2  # mm²

    As_formule_inf = (M_inf_val * 1e6) / (fyd * 0.9 * d_calc_inf * 10) if M_inf_val > 0 else 0.0
    As_formule_sup = (M_sup_val * 1e6) / (fyd * 0.9 * d_calc_sup * 10) if M_sup_val > 0 else 0.0

    As_min_inf_eff = max(As_min_base, 0.25 * As_formule_sup)
    As_min_sup_eff = max(As_min_base, 0.25 * As_formule_inf)

    As_req_inf_final = As_formule_inf
    As_req_sup_final = As_formule_sup

    # As,min effectif CONTRAIGNANT dans le statut
    etat_inf = "ok" if (geom_inf_ok and As_inf_total >= max(As_req_inf_final, As_min_inf_eff) and As_inf_total <= As_max) else "nok"
    etat_sup = "ok" if (geom_sup_ok and As_sup_total >= max(As_req_sup_final, As_min_sup_eff) and As_sup_total <= As_max) else "nok"

    # --- Tranchant : τ = V / (0.75·b·h) (inchangé) ---
    tau_1 = 0.016 * fck_cube / 1.05
    tau_2 = 0.032 * fck_cube / 1.05
    tau_4 = 0.064 * fck_cube / 1.05

    def _shear_state(tau):
        if tau <= tau_1:
            return "ok"
        if tau <= tau_2:
            return "ok"
        if tau <= tau_4:
            return "warn"
        return "nok"

    if V_val > 0:
        tau = V_val * 1e3 / (0.75 * b * h * 100)
        etat_tau = _shear_state(tau)
        if not geom_shear_ok:
            etat_tau = "nok"
    else:
        etat_tau = "ok"

    def _pas_state(V_kn: float, pas_key: str, reduced: bool):
        pas = float(st.session_state.get(KS(pas_key, beam_id, sec_id), 30.0) or 30.0)
        Ast_e = _shear_lines_total_Ast_mm2(beam_id, sec_id, reduced=reduced)
        # s = Ast·fyd·d / V  (d en mm, V en N) -> résultat en mm, puis /10 -> cm
        pas_th = Ast_e * fyd * (d_calc_shear * 10.0) / (V_kn * 1e3) / 10.0
        s_max = min(0.75 * d_calc_shear, 30.0)
        pas_lim = min(pas_th, s_max)
        etat = "ok" if pas <= pas_lim else "nok"
        if not geom_shear_ok:
            etat = "nok"
        return etat

    etat_pas = _pas_state(V_val, "shear_pas", reduced=False) if V_val > 0 else "ok"

    etat_global = _status_merge(etat_h, etat_inf, etat_sup, etat_tau, etat_pas)

    return {
        "etat_global": etat_global,
        "etat_h": etat_h,
        "etat_inf": etat_inf,
        "etat_sup": etat_sup,
        "etat_tau": etat_tau,
        "etat_pas": etat_pas,
        "M_inf_val": M_inf_val,
        "M_sup_val": M_sup_val,
        "V_val": V_val,
        "hmin_calc": hmin_calc,
        "tau_1": tau_1,
        "tau_2": tau_2,
        "tau_4": tau_4,
        "fyd": fyd,
        "gamma_s": gamma_s,
        "fyk": fyk,
        "alpha_b": alpha_b,
        "mu_val": mu_val,
        "beton": beton,
        "b": b,
        "h": h,
        "dist_l1_inf": dist_l1_inf,
        "dist_l1_sup": dist_l1_sup,
        "e_cdg_inf": e_cdg_inf,
        "e_cdg_sup": e_cdg_sup,
        "fctm": fctm,
        "As_min_ec": As_min_ec,
        "As_min_plancher": As_min_plancher,
        "As_min_formula": As_min_formula,
        "As_max": As_max,
        "As_formule_inf": As_formule_inf,
        "As_formule_sup": As_formule_sup,
        "As_min_inf_eff": As_min_inf_eff,
        "As_min_sup_eff": As_min_sup_eff,
        "As_req_inf_final": As_req_inf_final,
        "As_req_sup_final": As_req_sup_final,
        "As_inf_total": As_inf_total,
        "As_sup_total": As_sup_total,
        "inf_detail": inf_detail,
        "sup_detail": sup_detail,
        "d_utile_inf": d_utile_inf,
        "d_utile_sup": d_utile_sup,
        "d_utile_shear": d_utile_for_shear,
        "geom_inf_ok": geom_inf_ok,
        "geom_sup_ok": geom_sup_ok,
        "geom_shear_ok": geom_shear_ok,
    }


# ============================================================
#  UI : CISAILLEMENT (lignes)
# ============================================================
def _render_shear_lines_ui(beam_id: int, sec_id: int, disabled: bool):
    n_key = KS("shear_n_lines", beam_id, sec_id)
    pas_key = KS("shear_pas", beam_id, sec_id)
    prefix = "shear_line"
    add_btn_key = KS("btn_add_shear_line", beam_id, sec_id)
    del_btn_prefix = KS("btn_del_shear_line_", beam_id, sec_id)

    n_lines = max(1, int(st.session_state.get(n_key, 1) or 1))
    st.session_state[n_key] = n_lines

    # Barres du lit 1 inférieur : référence des positions "de -> à".
    n_bars = max(1, int(st.session_state.get(KS("n_as_inf", beam_id, sec_id), 2) or 2))
    bar_opts = list(range(1, n_bars + 1))
    show_pos = n_bars > 1  # positions discrètes : visibles seulement si plusieurs barres

    for i in range(n_lines):
        st.session_state.setdefault(KS(f"{prefix}{i}_type", beam_id, sec_id), "Étriers (2 brins)")
        st.session_state.setdefault(KS(f"{prefix}{i}_d", beam_id, sec_id), 8)

        if show_pos:
            c0, c1, cF, cT, c3, c4 = st.columns([2.6, 1.3, 0.9, 0.9, 1.6, 0.65], vertical_alignment="center")
        else:
            c0, c1, c3, c4 = st.columns([2.6, 1.3, 3.4, 0.65], vertical_alignment="center")

        with c0:
            st.selectbox(
                "Type",
                ["Étriers (2 brins)", "Épingles (1 brin)"],
                key=KS(f"{prefix}{i}_type", beam_id, sec_id),
                label_visibility="visible" if i == 0 else "collapsed",
                disabled=disabled,
            )
        with c1:
            st.selectbox(
                "Ø (mm)",
                SHEAR_DIAM_OPTS,
                key=KS(f"{prefix}{i}_d", beam_id, sec_id),
                label_visibility="visible" if i == 0 else "collapsed",
                disabled=disabled,
            )
        if show_pos:
            with cF:
                st.selectbox(
                    "de",
                    bar_opts,
                    key=KS(f"{prefix}{i}_from", beam_id, sec_id),
                    label_visibility="visible" if i == 0 else "collapsed",
                    disabled=disabled,
                    help="Barre de début (lit 1 inf.) — dessin uniquement" if i == 0 else None,
                )
            with cT:
                st.selectbox(
                    "à",
                    bar_opts,
                    key=KS(f"{prefix}{i}_to", beam_id, sec_id),
                    label_visibility="visible" if i == 0 else "collapsed",
                    disabled=disabled,
                    help="Barre de fin (lit 1 inf.) — dessin uniquement" if i == 0 else None,
                )
        with c3:
            if i == 0:
                float_input_fr_simple("Pas choisi (cm)", key=pas_key, default=30.0, min_value=1.0, disabled=disabled)
            else:
                st.markdown("")
        with c4:
            if i > 0:
                st.button(
                    "🗑️",
                    key=f"{del_btn_prefix}{i}",
                    use_container_width=True,
                    disabled=disabled,
                    on_click=_delete_shear_line,
                    args=(beam_id, sec_id, False, i),
                )

    st.button(
        "➕ Ajouter armature d'effort tranchant",
        key=add_btn_key,
        use_container_width=True,
        disabled=disabled,
        on_click=_add_shear_line,
        args=(beam_id, sec_id, False),
    )


# ============================================================
#  UI : TABLEAU DES LITS (Lit | Nb barres | Ø | Distance | Action)
# ============================================================
def _render_lit_row(beam_id: int, sec_id: int, which: str, i: int, nl: int, disabled: bool):
    """Ligne du tableau des lits. i=1 : bouton '+ Lit' ; i>=2 : poubelle."""
    suffix = " (sup.)" if which == "sup" else " (inf.)"

    if i == 1:
        key_val, key_ovr = _dist_keys(beam_id, sec_id, which, 1)
        key_n = KS(f"n_as_{which}", beam_id, sec_id)
        key_d = KS(f"ø_as_{which}", beam_id, sec_id)
    else:
        st.session_state.setdefault(KS(f"n_as_{which}_l{i}", beam_id, sec_id), 2)
        st.session_state.setdefault(KS(f"ø_as_{which}_l{i}", beam_id, sec_id), 16)
        key_val, key_ovr = _dist_keys(beam_id, sec_id, which, i)
        key_n = KS(f"n_as_{which}_l{i}", beam_id, sec_id)
        key_d = KS(f"ø_as_{which}_l{i}", beam_id, sec_id)

    auto_i = _auto_dist_lit(beam_id, sec_id, which, i)
    st.session_state.setdefault(key_val, float(auto_i))
    st.session_state.setdefault(key_ovr, False)

    c0, c1, c2, c3, c4 = st.columns(LIT_COLS, vertical_alignment="center")
    with c0:
        st.markdown(f"Lit {i}")
    with c1:
        st.number_input(
            f"Nb barres (lit {i}){suffix}",
            min_value=1,
            max_value=50,
            step=1,
            key=key_n,
            disabled=disabled,
            label_visibility="collapsed",
        )
    with c2:
        st.selectbox(
            f"Ø (mm) (lit {i}){suffix}",
            DIAM_OPTS,
            key=key_d,
            disabled=disabled,
            label_visibility="collapsed",
        )
    with c3:
        val_d = st.number_input(
            f"Distance axe lit {i} (cm){suffix}",
            min_value=0.0,
            max_value=300.0,
            step=0.5,
            key=key_val,
            disabled=disabled,
            label_visibility="collapsed",
        )
        # L'override est une clé non-widget : mutation légale après rendu.
        st.session_state[key_ovr] = bool(abs(float(val_d) - float(auto_i)) > 1e-6)
    with c4:
        if i == 1:
            st.button(
                "＋ Lit",
                key=KS(f"btn_add_lit_{which}", beam_id, sec_id),
                use_container_width=True,
                disabled=disabled or (nl >= MAX_LITS),
                help="Ajouter un lit",
                on_click=_add_lit,
                args=(beam_id, sec_id, which),
            )
        else:
            st.button(
                "🗑️",
                key=KS(f"btn_del_lit_{which}_{i}", beam_id, sec_id),
                use_container_width=True,
                disabled=disabled,
                help="Supprimer ce lit",
                on_click=_delete_lit,
                args=(beam_id, sec_id, which, i),
            )


def _render_lits_table(beam_id: int, sec_id: int, which: str, disabled: bool):
    nl = _get_nlits(beam_id, sec_id, which)

    # En-tête du tableau
    h0, h1, h2, h3, h4 = st.columns(LIT_COLS, vertical_alignment="bottom")
    with h0:
        st.markdown("")
    with h1:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Nb barres</div>", unsafe_allow_html=True)
    with h2:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Ø (mm)</div>", unsafe_allow_html=True)
    with h3:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Distance axe lit (cm)</div>", unsafe_allow_html=True)
    with h4:
        st.markdown("")

    for i in range(1, nl + 1):
        _render_lit_row(beam_id, sec_id, which, i, nl, disabled)


def _render_face_armatures(beam_id: int, sec_id: int, which: str, states: dict, dim_locked: bool, units_as: str):
    """Bloc 'Armatures inférieures' / 'Armatures supérieures' complet."""
    is_inf = (which == "inf")
    titre = "Armatures inférieures" if is_inf else "Armatures supérieures"
    unit_as_txt = "mm²" if units_as == "mm²" else "cm²"

    As_total = states["As_inf_total"] if is_inf else states["As_sup_total"]
    detail = states["inf_detail"] if is_inf else states["sup_detail"]
    etat = states["etat_inf"] if is_inf else states["etat_sup"]
    As_req = states["As_req_inf_final"] if is_inf else states["As_req_sup_final"]
    As_min_eff = states["As_min_inf_eff"] if is_inf else states["As_min_sup_eff"]
    As_max = states["As_max"]
    geom_ok = states["geom_inf_ok"] if is_inf else states["geom_sup_ok"]
    nl = _get_nlits(beam_id, sec_id, which)

    As_disp = As_total if units_as == "mm²" else As_total / 100.0
    right = f"{detail} — As={As_disp:.2f} {unit_as_txt}"
    besoin_dim = max(As_req, As_min_eff)      # valeur dimensionnante
    pct_as = (besoin_dim / As_total * 100.0) if As_total > 0 else None

    open_bloc_left_right(titre, right, etat, pct=pct_as)

    # Infobulles pédagogiques : formule + valeurs numériques
    M_face = states["M_inf_val"] if is_inf else states["M_sup_val"]
    d_face = states["d_utile_inf"] if is_inf else states["d_utile_sup"]
    As_req_opp = states["As_formule_sup"] if is_inf else states["As_formule_inf"]
    fyk = states["fyk"]; gs = states["gamma_s"]; fyd = states["fyd"]

    help_req = (
        "**Aₛ,req = M / (fyd · 0,9 · d)**\n\n"
        f"M = {_fr(M_face, 1)} kN·m\n\n"
        f"fyd = {_fr(fyk, 0)} / {_fr(gs, 2)} = {_fr(fyd, 0)} N/mm²\n\n"
        f"d = {_fr(d_face, 1)} cm\n\n"
        f"→ Aₛ,req = {_fr(As_req, 0)} mm²"
    )
    help_min = (
        "**Aₛ,min = max( 0,26·fctm/fyk·b·h ; 0,0013·b·h ; 0,25·Aₛ,req face opposée )**\n\n"
        f"0,26 · {_fr(states['fctm'], 1)} / {_fr(fyk, 0)} · b · h = {_fr(states['As_min_ec'], 0)} mm²\n\n"
        f"0,0013 · b · h = {_fr(states['As_min_plancher'], 0)} mm²\n\n"
        f"0,25 · {_fr(As_req_opp, 0)} = {_fr(0.25 * As_req_opp, 0)} mm²\n\n"
        f"→ Aₛ,min = {_fr(As_min_eff, 0)} mm²"
    )
    help_max = (
        "**Aₛ,max = 0,04 · b · h**\n\n"
        f"0,04 · {_fr(states['b'] * 10, 0)} · {_fr(states['h'] * 10, 0)} = {_fr(As_max, 0)} mm²"
    )

    ca1, ca2, ca3 = st.columns(3)
    with ca1:
        st.markdown(f"**Aₛ,req,{'inf' if is_inf else 'sup'} = {As_req:.0f} mm²**", help=help_req)
    with ca2:
        st.markdown(f"**Aₛ,min,{'inf' if is_inf else 'sup'} = {As_min_eff:.0f} mm²**", help=help_min)
    with ca3:
        st.markdown(f"**Aₛ,max = {As_max:.0f} mm²**", help=help_max)

    if not geom_ok:
        st.markdown("❌ **Position des lits incompatible avec la hauteur : d utile ≤ 0.**")

    # ---- Tableau des lits : Lit | Nb barres | Ø | Distance axe lit | Action ----
    _render_lits_table(beam_id, sec_id, which, disabled=dim_locked)

    # ---- Récap : choix + d utile + yG (c.d.g. des lits) ----
    As_total2, e_cdg2, detail2 = _layers_geometry(beam_id, sec_id, which)
    As_disp2 = As_total2 if units_as == "mm²" else As_total2 / 100.0
    if nl > 1:
        d_eff2 = states["h"] - e_cdg2
        st.markdown(
            f"<div style='margin-top:6px;font-weight:600;'>Choix : {detail2} ({As_disp2:.2f} {unit_as_txt}) — "
            f"d utile = {d_eff2:.1f} cm (c.d.g. des {nl} lits) — yG = {e_cdg2:.1f} cm</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='margin-top:6px;font-weight:600;'>Choix : {detail2} ({As_disp2:.2f} {unit_as_txt}) — "
            f"yG = {e_cdg2:.1f} cm</div>",
            unsafe_allow_html=True,
        )
    close_bloc()


# ============================================================
#  TAUX D'ARMATURE (application uniquement — pas exporté en PDF
#  tant que le calcul global de la poutre n'est pas disponible)
# ============================================================
def _masse_lin_kg_m(d_mm: float) -> float:
    """Masse linéique d'une barre (kg/m) : ρ·π·d²/4."""
    return RHO_ACIER * math.pi * (d_mm / 1000.0) ** 2 / 4.0


def _taux_armature_section(beam_id: int, sec_id: int):
    """
    Taux d'armature d'une section, au mètre courant de poutre.
    Retourne (ta_arrondi_kgm3, detail_markdown) ou (None, None) si
    le calcul est désactivé.
    """
    if not bool(st.session_state.get("taux_arm_enable", True)):
        return None, None

    b = float(st.session_state.get(KB("b", beam_id), 20))          # cm
    h = float(st.session_state.get(KB("h", beam_id), 40))          # cm
    enrob = float(st.session_state.get(KB("enrobage_beton", beam_id), 3.0) or 3.0)
    maj = float(st.session_state.get("taux_arm_major_pct", 5.0) or 0.0)
    retour = float(st.session_state.get("taux_retour_etrier_cm", 10.0) or 0.0)
    arrondi = max(1, int(st.session_state.get("taux_arrondi_kgm3", 5) or 5))

    rows = []   # (famille, Ø, As_unitaire_mm², kg/m, L_m_par_m, poids_kg_par_m)

    # ---- Barres longitudinales (tous lits, deux faces) : L = 1 m/barre ----
    for which, face in (("inf", "inf."), ("sup", "sup.")):
        nl = _get_nlits(beam_id, sec_id, which)
        for i in range(1, nl + 1):
            n, dmm = _lit_bars(beam_id, sec_id, which, i)
            kgm = _masse_lin_kg_m(dmm)
            L = float(n)  # n barres × 1 m
            rows.append((f"Lit {i} ({face})", dmm, _bar_area_mm2(dmm), kgm, L, L * kgm))

    # ---- Étriers / épingles : n/m = 100 / pas ----
    pas = float(st.session_state.get(KS("shear_pas", beam_id, sec_id), 30.0) or 30.0)
    n_par_m = (100.0 / pas) if pas > 0 else 0.0
    n_lines = max(1, int(st.session_state.get(KS("shear_n_lines", beam_id, sec_id), 1) or 1))

    # positions réelles des barres du lit 1 inf. (largeur des étriers partiels)
    n1, d1 = _lit_bars(beam_id, sec_id, "inf", 1)
    d_et_max = _stirrup_diam_mm(beam_id, sec_id)
    inset = enrob + d_et_max / 10.0 + d1 / 20.0                     # axe barre (cm)
    if n1 > 1:
        xs = [inset + (b - 2 * inset) * k / (n1 - 1) for k in range(n1)]
    else:
        xs = [b / 2.0]

    for i in range(n_lines):
        typ = str(st.session_state.get(KS(f"shear_line{i}_type", beam_id, sec_id), "Étriers (2 brins)"))
        dmm = float(st.session_state.get(KS(f"shear_line{i}_d", beam_id, sec_id), 8) or 8)
        f, t = _shear_line_pos(beam_id, sec_id, i)
        f = max(1, min(n1, f)); t = max(1, min(n1, t))
        kgm = _masse_lin_kg_m(dmm)
        if _brins_from_type(typ) == 1:
            # épingle : brin vertical (ou agrafe horizontale si f != t)
            L_un = ((h - 2 * enrob) if f == t else abs(xs[t - 1] - xs[f - 1])) + 2 * retour   # cm
            fam = f"Épingle Ø{int(dmm)}"
        else:
            w_ext = (b - 2 * enrob) if (f <= 1 and t >= n1) else (abs(xs[t - 1] - xs[f - 1]) + d1 / 10.0 + 2 * dmm / 10.0)
            L_un = 2 * (w_ext + (h - 2 * enrob)) + 2 * retour                                 # cm
            fam = f"Étrier Ø{int(dmm)}"
        L = n_par_m * L_un / 100.0                                                            # m / m courant
        rows.append((fam, dmm, _bar_area_mm2(dmm), kgm, L, L * kgm))

    # ---- Armatures de peau (si déclenchées) ----
    t_d = float(st.session_state.get("techno_d_mm", 10) or 10)
    t_smax = float(st.session_state.get("techno_s_max_cm", 30.0) or 30.0)
    e_inf1 = _get_dist_lit(beam_id, sec_id, "inf", 1)
    e_sup1 = _get_dist_lit(beam_id, sec_id, "sup", 1)
    d_vert = h - e_inf1 - e_sup1
    if t_smax > 0 and d_vert > t_smax:
        n_side = max(0, int(math.ceil(d_vert / t_smax)) - 1)
        if n_side > 0:
            kgm = _masse_lin_kg_m(t_d)
            L = 2.0 * n_side
            rows.append((f"Peau Ø{int(t_d)}", t_d, _bar_area_mm2(t_d), kgm, L, L * kgm))

    poids = sum(r[5] for r in rows)
    poids_maj = poids * (1.0 + maj / 100.0)
    vol = (b / 100.0) * (h / 100.0) * 1.0                            # m³ / m courant
    ta_brut = poids_maj / vol if vol > 0 else 0.0
    ta_arr = math.ceil(ta_brut / arrondi) * arrondi if ta_brut > 0 else 0

    # ---- Tableau détaillé (markdown, pour l'infobulle) ----
    md = "**Calcul du taux d'armature (par mètre courant)**\n\n"
    md += "| Famille | Ø (mm) | As (mm²) | kg/m | L (m/m) | Poids (kg/m) |\n"
    md += "|---|---|---|---|---|---|\n"
    for fam, dmm, As, kgm, L, p in rows:
        md += f"| {fam} | {int(dmm)} | {As:.0f} | {kgm:.3f} | {L:.2f} | {p:.2f} |\n"
    md += f"\nPoids total : **{poids:.1f} kg/m** — majoration {maj:.0f} % → **{poids_maj:.1f} kg/m**\n\n"
    md += f"Volume de béton : **{vol:.3f} m³/m**\n\n"
    md += f"TA = {poids_maj:.1f} / {vol:.3f} = **{ta_brut:.0f} kg/m³** → arrondi ({arrondi}) : **{ta_arr:.0f} kg/m³**"
    return ta_arr, md


# ============================================================
#  UI : DIMENSIONNEMENT D'UNE SECTION
# ============================================================
def render_dimensionnement_section(beam_id: int, sec_id: int, beton_data: dict):
    beam_locked = bool(st.session_state.get(KB("lock_data", beam_id), False))
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    sec = next(s for s in beam["sections"] if int(s.get("id")) == sec_id)
    sec_nom = str(st.session_state.get(f"meta_b{beam_id}_nom_{sec_id}", sec.get("nom", f"Section {sec_id}")))

    states = _dimensionnement_compute_states(beam_id, sec_id, beton_data)
    ta_val, ta_md = _taux_armature_section(beam_id, sec_id)
    sec_label = sec_nom if sec_nom.lower().startswith("section") else f"Section {sec_nom}"
    title = _status_icon_label(states["etat_global"], sec_label)
    if ta_val is not None:
        title += f" — TA = {ta_val:.0f} kg/m³"

    # NB : expanded=True pour tous — le libellé contient l'icône d'état,
    # et Streamlit remet un expander à son état par défaut dès que son
    # libellé change ; avec False, les sections se refermaient toutes
    # seules à chaque changement d'état.
    with st.expander(title, expanded=True):
        if ta_val is not None:
            # icône ⓘ : tableau détaillé du calcul au survol
            st.markdown(f"**Taux d'armature : TA = {ta_val:.0f} kg/m³**", help=ta_md)
        if beam_locked:
            st.caption("🔒 Poutre verrouillée — édition bloquée.")

        dim_locked = beam_locked

        units_len = st.session_state.get("units_len", "cm")
        units_as = st.session_state.get("units_as", "mm²")

        beton = states["beton"]
        b = states["b"]
        h = states["h"]
        fyd = states["fyd"]
        hmin_calc = states["hmin_calc"]
        dist_l1_inf = states["dist_l1_inf"]
        V_val = states["V_val"]

        # ---- Vérification de la hauteur ----
        if units_len == "mm":
            right_h = f"{beton} — {b*10:.0f}×{h*10:.0f} mm — (hmin = {hmin_calc*10:.0f} mm)"
        else:
            right_h = f"{beton} — {b:.0f}×{h:.0f} cm — (hmin = {hmin_calc:.1f} cm)"
        pct_h = ((hmin_calc + dist_l1_inf) / h * 100.0) if h > 0 else None

        M_max_val = max(states["M_inf_val"], states["M_sup_val"])
        help_hmin = (
            "**h,min = √( M / (α_b · b · μ) )**\n\n"
            f"M = {_fr(M_max_val, 1)} kN·m\n\n"
            f"α_b = {_fr(states['alpha_b'], 2)}\n\n"
            f"b = {_fr(b * 10, 0)} mm\n\n"
            f"μ = {states['mu_val']}\n\n"
            f"→ h,min = {_fr(hmin_calc, 1)} cm"
        )
        open_bloc_left_right("Vérification de la hauteur", right_h, states["etat_h"], pct=pct_h)
        if units_len == "mm":
            st.markdown(
                f"**h,min** = {hmin_calc*10:.0f} mm  \n"
                f"h,min + distance axe lit 1 (inf.) = {(hmin_calc + dist_l1_inf)*10:.0f} mm ≤ h = {h*10:.0f} mm",
                help=help_hmin,
            )
        else:
            st.markdown(
                f"**h,min** = {hmin_calc:.1f} cm  \n"
                f"h,min + distance axe lit 1 (inf.) = {hmin_calc + dist_l1_inf:.1f} cm ≤ h = {h:.1f} cm",
                help=help_hmin,
            )
        close_bloc()

        # ---- Armatures inférieures / supérieures ----
        _render_face_armatures(beam_id, sec_id, "inf", states, dim_locked, units_as)
        _render_face_armatures(beam_id, sec_id, "sup", states, dim_locked, units_as)

        # ---- Tranchant + étriers ----
        tau_1, tau_2, tau_4 = states["tau_1"], states["tau_2"], states["tau_4"]

        def _shear_need_text(tau):
            if tau <= tau_1:
                return "Pas besoin d’étriers", "ok", "τ_adm_I", tau_1
            if tau <= tau_2:
                return "Besoin d’étriers", "ok", "τ_adm_II", tau_2
            if tau <= tau_4:
                return "Besoin de barres inclinées et d’étriers", "warn", "τ_adm_IV", tau_4
            return "Pas acceptable", "nok", "τ_adm_IV", tau_4

        def _bloc_pas(V_kn: float, pas_key_base: str, titre_tau: str, titre_pas: str, etat_pas_state: str):
            tau = V_kn * 1e3 / (0.75 * b * h * 100)
            besoin, etat_tau, nom_lim, tau_lim = _shear_need_text(tau)

            pct_tau = (tau / tau_lim * 100.0) if tau_lim > 0 else None
            open_bloc_left_right(titre_tau, f"τ={tau:.2f} ≤ {nom_lim}={tau_lim:.2f}", etat_tau, pct=pct_tau)
            st.markdown(f"τ = {tau:.2f} N/mm² ≤ {nom_lim} = {tau_lim:.2f} N/mm² → {besoin}")
            close_bloc()

            # Conclusion AVANT la saisie : c'est l'information principale.
            pas = float(st.session_state.get(KS(pas_key_base, beam_id, sec_id), 30.0) or 30.0)
            Ast_e = _shear_lines_total_Ast_mm2(beam_id, sec_id, reduced=False)
            d_sh = max(states["d_utile_shear"], 0.1)
            pas_th = Ast_e * fyd * (d_sh * 10.0) / (V_kn * 1e3) / 10.0  # cm
            s_max = min(0.75 * d_sh, 30.0)
            pas_lim = min(pas_th, s_max)

            help_pas = (
                "**s,th = Aₛₜ · fyd · d / V**\n\n"
                f"Aₛₜ = {_fr(Ast_e, 1)} mm²\n\n"
                f"fyd = {_fr(fyd, 0)} N/mm²\n\n"
                f"d = {_fr(d_sh, 1)} cm\n\n"
                f"V = {_fr(V_kn, 1)} kN\n\n"
                f"→ s,th = {_fr(pas_th, 1)} cm"
            )

            right_et = f"pas={pas:.1f} ≤ min({pas_th:.1f},{s_max:.1f})={pas_lim:.1f} cm"
            pct_pas = (pas / pas_lim * 100.0) if pas_lim > 0 else None
            open_bloc_left_right(titre_pas, right_et, etat_pas_state, pct=pct_pas)
            a1, a2 = st.columns(2)
            with a1:
                st.markdown(f"**Pas théorique = {pas_th:.1f} cm**", help=help_pas)
            with a2:
                st.markdown(f"**Pas maximal = {s_max:.1f} cm**", help="**s,max = min( 0,75 · d ; 30 cm )**")
            st.caption(_shear_lines_summary(beam_id, sec_id, reduced=False))
            close_bloc()

            # Saisie des armatures d'effort tranchant (après la conclusion)
            _render_shear_lines_ui(beam_id, sec_id, disabled=dim_locked)

        if V_val > 0:
            _bloc_pas(V_val, "shear_pas", "Vérification de l'effort tranchant", "Détermination des étriers", states["etat_pas"])


# ============================================================
#  UI : INFOS PROJET / PARAMÈTRES AVANCÉS
# ============================================================
def _pdf_filename() -> str:
    """
    Nom automatique du rapport : AAA_NDC Partie#Indice_Date.pdf
    AAA = 3 premières lettres (alphanumériques) du nom du projet.
    Ex : REA_NDC Poutres du portique#B_21-02-2026.pdf
    """
    nom = str(st.session_state.get("nom_projet", "") or "")
    aaa = re.sub(r"[^A-Za-z0-9]", "", nom).upper()[:3] or "PRJ"
    partie = str(st.session_state.get("partie", "") or "").strip()
    indice = str(st.session_state.get("indice", "0") or "0").strip() or "0"
    date = str(st.session_state.get("date", "") or datetime.today().strftime("%d/%m/%Y")).strip().replace("/", "-")
    core = f"NDC {partie}".strip()
    name = f"{aaa}_{core}#{indice}_{date}.pdf"
    return re.sub(r'[\\/:*?"<>|]+', "-", name)


def _toggle_param_avances():
    st.session_state["show_param_avances"] = not bool(st.session_state.get("show_param_avances", False))


def _toggle_infos_projet():
    st.session_state["chk_infos_projet"] = not bool(st.session_state.get("chk_infos_projet", False))


def render_infos_projet():
    st.session_state.setdefault("chk_infos_projet", False)
    st.session_state.setdefault("nom_projet", "")
    st.session_state.setdefault("partie", "")
    st.session_state.setdefault("date", datetime.today().strftime("%d/%m/%Y"))
    st.session_state.setdefault("indice", "0")

    shown = bool(st.session_state.get("chk_infos_projet", False))
    cT, cBtn = st.columns([6, 0.6], vertical_alignment="center")
    with cT:
        st.markdown("### Informations sur le projet")
    with cBtn:
        st.button(
            "➖" if shown else "➕",
            key="btn_toggle_infos_projet",
            help="Masquer les informations du projet" if shown else "Ajouter les informations du projet",
            use_container_width=True,
            on_click=_toggle_infos_projet,
        )

    if bool(st.session_state.get("chk_infos_projet", False)):
        with st.container(border=True):
            st.text_input("Nom du projet", placeholder="Nom du projet", key="nom_projet", label_visibility="collapsed")
            st.text_input("Partie", placeholder="Partie", key="partie", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Date", placeholder="Date (jj/mm/aaaa)", key="date", label_visibility="collapsed")
            with c2:
                st.text_input("Indice", placeholder="Indice", key="indice", label_visibility="collapsed")


def render_parametres_avances():
    """
    Paramètres avancés en tableau compact, 3 colonnes thématiques :
      1. Affichage        : unités de longueur / d'armature
      2. Coeff. matériaux : γs (fyd = fyk / γs)
      3. Jeux d'armatures : jeu premier lit / jeu entre lits
    """
    _ensure_global_defaults()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Affichage**")
        st.selectbox("Unité de longueur", ["cm", "mm"], key="units_len")
        st.selectbox("Unité d'armature", ["mm²", "cm²"], key="units_as")

    with c2:
        st.markdown("**Coefficients matériaux**")
        st.number_input(
            "Coefficient acier ELS",
            min_value=1.0,
            max_value=2.0,
            step=0.05,
            format="%.2f",
            key="gamma_s",
            help="fyd = fyk / coefficient — défaut 1,5 (méthode ancienne). "
                 "Ex. acier 500 : 500/1,5 = 333 MPa ; avec 1,15 : 435 MPa.",
        )

    with c3:
        st.markdown("**Jeux d'armatures**")
        st.number_input("Jeu premier lit (cm)", min_value=0.0, step=0.5, key="jeu_enrobage_cm")
        st.number_input("Jeu entre lits (cm)", min_value=0.0, step=0.5, key="jeu_entre_lits_cm")

    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown("**Armatures technologiques**")
        st.selectbox("Ø (mm)", [8, 10, 12, 16], key="techno_d_mm",
                     help="Barres latérales de peau, ajoutées automatiquement sur la coupe "
                          "du PDF quand l'écart vertical entre lits dépasse l'espacement max. "
                          "Dessin uniquement.")
        st.number_input("Espacement vertical max (cm)", min_value=5.0, step=5.0, key="techno_s_max_cm")

    with c5:
        tt1, tt2 = st.columns([3, 0.8], vertical_alignment="center")
        with tt1:
            st.markdown("**Taux d'armature**")
        with tt2:
            st.checkbox("Calcul du taux d'armature", key="taux_arm_enable", label_visibility="collapsed")
        st.number_input("Pourcentage de majoration (%)", min_value=0.0, step=1.0, key="taux_arm_major_pct")
        st.number_input("Retour d'étrier (cm)", min_value=0.0, step=1.0, key="taux_retour_etrier_cm",
                        help="Compté deux fois dans la longueur d'un étrier : "
                             "longueur du rectangle + 2 × retour.")
        st.number_input("Arrondi supérieur (kg/m³)", min_value=1, step=1, key="taux_arrondi_kgm3",
                        help="Ex. : 103 kg/m³ → 105 (arrondi 5) ou 110 (arrondi 10).")

    with c6:
        st.markdown("")


# ============================================================
#  UI : COLONNES GAUCHE / DROITE
# ============================================================
def render_donnees_left(beton_data: dict):
    st.markdown("### Données")
    for b in st.session_state.beams:
        bid = int(b["id"])
        b["nom"] = str(st.session_state.get(f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}")))
        render_caracteristiques_beam(bid)


def render_dimensionnement_right(beton_data: dict):
    for b in st.session_state.beams:
        bid = int(b["id"])
        bnom = str(st.session_state.get(f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}")))
        b["nom"] = bnom

        sec_states = [
            _dimensionnement_compute_states(bid, int(s["id"]), beton_data)["etat_global"]
            for s in b.get("sections", [])
        ]
        beam_state = _status_merge(*sec_states) if sec_states else "ok"
        lock_icon = "🔒 " if bool(st.session_state.get(KB("lock_data", bid), False)) else ""
        beam_label = _status_icon_label(beam_state, f"{lock_icon}{bnom}")

        # expanded=True : libellé dynamique (icône d'état) -> Streamlit
        # réinitialise l'expander à chaque changement de libellé.
        with st.expander(beam_label, expanded=True):
            for s in b.get("sections", []):
                render_dimensionnement_section(bid, int(s["id"]), beton_data)


# ============================================================
#  PAGE
# ============================================================
def show():
    # ---------- Données béton : chargées AVANT l'init ----------
    global BETON_DATA
    try:
        with open("beton_classes.json", "r", encoding="utf-8") as f:
            BETON_DATA = json.load(f)
    except Exception:
        st.error("Impossible de charger beton_classes.json — vérifie que le fichier est présent et valide.")
        st.stop()
    beton_data = BETON_DATA

    _ensure_global_defaults()
    _init_beams_if_needed()

    # FIX PERSISTANCE : épingler toutes les clés persistantes AVANT tout rendu.
    _pin_persistent_state()

    # FIX BUG RECALCUL : synchroniser les saisies décimales FR (*_raw)
    # vers leurs valeurs numériques AVANT tout calcul.
    _sync_float_raw_keys()

    if "retour_accueil_demande" not in st.session_state:
        st.session_state.retour_accueil_demande = False

    if st.session_state.retour_accueil_demande:
        st.session_state.page = "Accueil"
        st.session_state.retour_accueil_demande = False
        st.rerun()

    tH1, tH2, tH3 = st.columns([8, 1.6, 0.55], vertical_alignment="center")
    with tH1:
        st.markdown("## Poutre en béton armé")
    with tH2:
        st.markdown(
            f"<div style='text-align:right;color:#6b7280;font-size:0.9em;'>Version {APP_VERSION}</div>",
            unsafe_allow_html=True,
        )
    with tH3:
        st.button("❔", key="btn_version_hist", help="Historique des versions (à venir)",
                  use_container_width=True)

    btn1, btn2, btn3, btn4, btn5 = st.columns(5)

    with btn1:
        if st.button("🏠 Accueil", use_container_width=True, key="btn_home"):
            st.session_state.retour_accueil_demande = True
            st.rerun()

    with btn2:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="btn_reset"):
            _reset_module()

    with btn3:
        payload = _build_save_payload()
        st.download_button(
            label="💾 Enregistrer",
            data=json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="poutre_ba.json",
            mime="application/json",
            use_container_width=True,
            key="btn_save_dl",
        )

    with btn4:
        if st.button("📂 Ouvrir", use_container_width=True, key="btn_open_toggle"):
            st.session_state["show_open_uploader"] = not st.session_state.get("show_open_uploader", False)

        if st.session_state.get("show_open_uploader", False):
            uploaded = st.file_uploader("Choisir un fichier JSON", type=["json"], label_visibility="collapsed", key="open_uploader")
            if uploaded is not None:
                try:
                    data = json.load(uploaded)
                    if not isinstance(data, dict):
                        raise ValueError("Structure JSON inattendue")
                    _load_from_payload(data)
                    st.session_state["show_open_uploader"] = False
                    st.rerun()
                except Exception:
                    st.error("Fichier invalide ou corrompu — chargement annulé.")

    with btn5:
        if st.button("📄 Générer PDF", use_container_width=True, key="btn_pdf"):
            from modules.export_pdf import generer_rapport_pdf

            infos = {
                "nom_projet": st.session_state.get("nom_projet", ""),
                "partie": st.session_state.get("partie", ""),
                "date": st.session_state.get("date", datetime.today().strftime("%d/%m/%Y")),
                "indice": st.session_state.get("indice", "0"),
            }

            try:
                fichier_pdf = generer_rapport_pdf(
                    beams=st.session_state.beams,
                    values=dict(st.session_state),
                    beton_data=beton_data,
                    infos=infos,
                )
                with open(fichier_pdf, "rb") as f:
                    st.session_state["pdf_bytes"] = f.read()
                st.success("✅ Note de calcul générée")
            except Exception as e:
                st.session_state.pop("pdf_bytes", None)
                st.error(f"Erreur lors de la génération du PDF : {e}")

        if st.session_state.get("pdf_bytes"):
            st.download_button(
                label="⬇️ Télécharger le rapport PDF",
                data=st.session_state["pdf_bytes"],
                file_name=_pdf_filename(),
                mime="application/pdf",
                use_container_width=True,
                key="btn_pdf_dl",
            )

    input_col_gauche, result_col_droite = st.columns([2, 3])

    with input_col_gauche:
        render_infos_projet()
        render_donnees_left(beton_data)

    with result_col_droite:
        st.session_state.setdefault("show_param_avances", False)

        cH1, cH2 = st.columns([18, 1.3], vertical_alignment="center")
        with cH1:
            st.markdown("### Dimensionnement")
        with cH2:
            st.button(
                "⚙️",
                key="btn_toggle_param_avances",
                help="Paramètres avancés",
                use_container_width=True,
                on_click=_toggle_param_avances,
            )

        if bool(st.session_state.get("show_param_avances", False)):
            with st.container(border=True):
                render_parametres_avances()

        render_dimensionnement_right(beton_data)
