# -*- coding: utf-8 -*-
"""Prédalle en béton armé — l'ENTRÉE : les bornes et les valeurs par défaut de l'écran Streamlit ``modules/predalle.py``
v1.1 (la référence de parité), et le contrôle d'une prédalle.

Une prédalle de l'écran est une dalle de l'écran ``modules/dalle.py`` (dont elle est la « copie adaptée ») plus UNE
donnée : l'épaisseur ``h_pre`` de la peau préfabriquée — et, depuis le lot 5 (14/11/2026), la COUTURE de chaque section
(``section["couture"]`` : les barres posées au joint de deux prédalles, beton/predalle/joint.py ; ``None`` =
automatique) ; depuis le lot 6, le PAS d'arrondi de leur longueur (``pas_couture`` : 1, 5 ou 10 cm ; 5 par défaut). Le contrôle de tout le reste (largeur, épaisseur, béton, nuance, sections, couches…) est celui du moteur
des dalles (``beton/dalle/entree.py``), employé tel quel ; seuls les DÉFAUTS diffèrent (« Prédalle 1 », h = 22 cm au
lieu de 20).

Règle de ``normaliser`` (celle du moteur des dalles) : une clé ABSENTE (ou ``null``) prend la valeur par défaut de
l'écran ; une valeur HORS DU DOMAINE de l'écran lève ``ValueError`` avec le chemin du champ. Jamais de substitution
silencieuse : l'écran, lui, ramène ``h_pre`` dans [3 ; min(15 ; h − 1)] sans le dire (``_ensure_defaults_for_dalle``,
audit I-5) ; le moteur REFUSE une peau hors de ces bornes.
"""
import copy

from beton.dalle import entree as ED

from . import joint as J

# ---- ce que la prédalle ajoute à la dalle (modules/predalle.py) ----
H_PRE_MIN = 3.0           # « Prédalle (cm) » : min_value
H_PRE_MAX = 15.0          # « Prédalle (cm) » : max_value, borné en plus à h − 1 (au moins 1 cm coulé en place)
COULE_MIN = 1.0           # cm : _h_pre_max(h) = max(H_PRE_MIN, min(H_PRE_MAX, h − 1))

# ---- les listes et bornes communes : celles du moteur des dalles (les mêmes dans les deux écrans) ----
DIAMETRES = ED.DIAMETRES
MAX_COUCHES = ED.MAX_COUCHES
NUANCES = ED.NUANCES
DIRECTIONS = ED.DIRECTIONS
FACES = ED.FACES
TYPES = ED.TYPES
LIBELLES_TYPES = ED.LIBELLES_TYPES
MOMENTS = ED.MOMENTS
B_MIN, B_MAX = ED.B_MIN, ED.B_MAX
H_MIN, H_MAX = ED.H_MIN, ED.H_MAX
ENROBAGE_MAX = ED.ENROBAGE_MAX
GAMMA_S_MIN, GAMMA_S_MAX = ED.GAMMA_S_MIN, ED.GAMMA_S_MAX
ESP_MIN, ESP_MAX = ED.ESP_MIN, ED.ESP_MAX
N_MIN, N_MAX = ED.N_MIN, ED.N_MAX
BETON_DEFAUT = ED.BETON_DEFAUT

# ---- les valeurs par défaut de l'écran prédalle (_init_dalles_if_needed, _ensure_defaults_for_dalle) ----
DEFAUTS_PREDALLE = dict(ED.DEFAUTS_DALLE, nom="Prédalle 1", h=22, h_pre=6.0, pas_couture=J.PAS_LONGUEUR)
CLES_PREDALLE = ED.CLES_DALLE | {"h_pre", "pas_couture"}
# les clés propres à la prédalle, retirées avant le moteur des dalles (qui les refuserait)
CLES_PROPRES = ("h_pre", "pas_couture")

classes_beton = ED.classes_beton
nouvelle_couche = ED.nouvelle_couche
nouvelle_face = ED.nouvelle_face
nouvelle_section = ED.nouvelle_section
nom_section_suivant = ED.nom_section_suivant
libelle_section = ED.libelle_section


def h_pre_max(h: float) -> float:
    """L'épaisseur de peau la plus forte admise pour une prédalle de h cm — _h_pre_max : au moins 1 cm coulé en place,
    15 cm au plus, jamais moins que la borne basse (3 cm)."""
    return max(H_PRE_MIN, min(H_PRE_MAX, float(h) - COULE_MIN))


def nouvelle_predalle() -> dict:
    """Une prédalle aux valeurs par défaut de l'écran : « Prédalle 1 », bande de 100 cm, 22 cm dont 6 cm de peau
    préfabriquée, C30/37, B500, enrobage 3 cm, direction principale Y, γs 1,5, jeu de pose 1 cm, arrondi 0,5 cm, la
    longueur des barres de couture arrondie au 5 cm, une section « A »."""
    p = copy.deepcopy(DEFAUTS_PREDALLE)
    # (la section « A » et sa couture automatique — None, beton/predalle/joint.py : la forme normalisée)
    p["sections"] = [dict(nouvelle_section("A"), couture=None)]
    return p


def normaliser(predalle: dict) -> dict:
    """Une copie COMPLÈTE et CONTRÔLÉE de la prédalle : la peau ``h_pre`` contrôlée ici (nombre, 3 cm ≤ h_pre ≤
    min(15 ; h − 1)), tout le reste par le moteur des dalles avec les défauts de l'écran prédalle.
    ``normaliser(normaliser(p)) == normaliser(p)``."""
    if not isinstance(predalle, dict):
        raise ValueError(f"prédalle : dict attendu (reçu {predalle!r})")
    inconnues = sorted(str(k) for k in predalle if k not in CLES_PREDALLE)
    if inconnues:
        raise ValueError(f"prédalle : clé(s) inconnue(s) {', '.join(inconnues)} — clés admises : {', '.join(sorted(CLES_PREDALLE))}")
    dalle = {k: v for k, v in predalle.items() if k not in CLES_PROPRES}
    for k in ("nom", "h"):                       # les défauts propres à l'écran prédalle
        if dalle.get(k) is None:
            dalle[k] = DEFAUTS_PREDALLE[k]
    # la COUTURE de chaque section (lot 5) : retirée avant le moteur des dalles, qui ne la connaît pas, contrôlée ici
    coutures = []
    if isinstance(dalle.get("sections"), (list, tuple)):
        secs = []
        for i, s in enumerate(dalle["sections"]):
            if isinstance(s, dict) and "couture" in s:
                coutures.append(J.normaliser_couture(s["couture"], f"sections[{i}].couture"))
                s = {k: v for k, v in s.items() if k != "couture"}
            else:
                coutures.append(None)
            secs.append(s)
        dalle["sections"] = secs
    out = ED.normaliser(dalle)
    for i, s in enumerate(out["sections"]):
        s["couture"] = coutures[i] if i < len(coutures) else None
    hp = predalle.get("h_pre")
    hp = DEFAUTS_PREDALLE["h_pre"] if hp is None else hp
    if isinstance(hp, bool) or not isinstance(hp, (int, float)) or hp != hp or hp in (float("inf"), float("-inf")):
        raise ValueError(f"h_pre : nombre attendu (reçu {hp!r})")
    hp = float(hp)
    maxi = h_pre_max(out["h"])
    if not H_PRE_MIN <= hp <= maxi:
        raise ValueError(f"h_pre : la prédalle fait de {H_PRE_MIN:g} à {maxi:g} cm pour h = {out['h']} cm "
                         f"(au moins {COULE_MIN:g} cm coulé en place, {H_PRE_MAX:g} cm au plus) — reçu {hp:g}")
    out["h_pre"] = hp
    pc = predalle.get("pas_couture")
    out["pas_couture"] = J.lire_pas(DEFAUTS_PREDALLE["pas_couture"] if pc is None else pc)
    return out
