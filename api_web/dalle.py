# -*- coding: utf-8 -*-
"""Le module « Dalles béton armé » du site — dalles COULÉES EN PLACE et PRÉDALLES dans un seul écran
(docs/site/DALLES_FUSION.md § 2) : l'état d'une note (une ou plusieurs dalles, chacune de son TYPE, avec ses sections,
et pour chaque section quatre nappes — inférieure et supérieure, directions X et Y — faites de couches de treillis, de
barres Ø/e ou de « n barres »), les champs de saisie, le résultat section par section, la grande fenêtre de calcul, le
sélecteur de la nappe de base et les dessins — rendus en fragments HTML.

L'écran est l'établi COMPACT, celui de la poutre (api_web/etabli_compact.py, api_web/plaque_compact.py) : à gauche les
rubriques de la dalle ouverte (le sélecteur des dalles dès qu'il y en a deux), au centre les vues sous leur en-tête
technique, à droite un bloc par section. Les couches des données et des dessins portent la même clé.

DEUX MOTEURS, INCHANGÉS, et le site choisit lequel appeler selon ``d["type"]`` :
- « coulee » : ``beton/dalle`` (extrait de l'écran Streamlit modules/dalle.py v2.2 — tests/test_dalle_moteur.py) ;
- « predalle » : ``beton/predalle`` (modules/predalle.py v1.1 — tests/test_predalle_moteur.py) ; la partie basse de la
  dalle (la peau, ``h_pre``) est préfabriquée, la base inférieure principale y est posée en fond de moule.
Ce qui est propre au type prédalle (ses textes, sa pose, ses avertissements, son périmètre, ses formules d'axe mises en
mots) vient de ``api_web/predalle.py``, réemployé comme AIDE (``AP``), jamais recopié ; ce qui est propre aux positions
des couches vient de ``api_web/dalle_positions.py`` (``POS`` : l'axe automatique du site, la copie envoyée au moteur) et
ce qui est propre aux NAPPES de ``api_web/dalle_nappes.py`` (``NAP``). Ce fichier MET EN FORME : il ne calcule rien.

L'ÉTAT : ``{"version": 3, "dalles": [dalle…], "actif": i}``. Chaque dalle est celle de son moteur plus deux clés du
site (CLES_SITE), retirées avant tout appel au moteur : « verrou » (lecture seule), « type » (« coulee » |
« predalle »). Les NAPPES sont TOUJOURS posées l'une sur l'autre (décision de l'ingénieur du 09/11/2026 :
api_web/dalle_positions.py) ; la clé « empilement » des notes d'avant (l'option retirée) est relue sans effet
(CLES_ANCIENNES), et une note d'avant dont les chiffres en changent le dit une fois, à l'ouverture (CLE_AVIS, AVIS).
``normaliser`` relit aussi ``{"dalles": […]}`` sans type (→ coulee) et ``{"predalles": […]}`` (l'ancien module : →
predalle) ; une note qui porte les deux clés est abîmée : refusée, jamais remplacée par une note neuve.

Ce que le site fait de plus que les écrans Streamlit, sans changer un chiffre :
- une saisie que l'écran ramènerait en silence à une autre valeur (moment négatif lu 0, espacement hors bornes, peau
  hors de ses bornes…) est REFUSÉE avec un message, et la dalle reste intacte ;
- le sélecteur de la nappe de base (treillis de la base, barres courantes) trié par section au mètre, la retenue (✓) et
  la plus petite qui tient (★) marquées ;
- les nappes posées l'une sur l'autre (la direction secondaire, base et renforts, sur la principale), que les moteurs
  posent au même niveau ;
- les avertissements des moteurs (nappes X et Y d'un axe saisi au même niveau, enrobage nul lu 3 cm, d utile ≤ 0, base
  qui sort de la peau, couches croisées sur la prédalle) ;
- la reprise d'un fichier de l'écran Streamlit — « dalle_ba.json » OU celui de l'écran prédalle, le type déduit du
  fichier — avec ce que l'écran y remplaçait en silence ;
- un verdict qui, dans une dalle non conforme, nomme une vérification NON conforme (gouvernante()) ;
- des bornes contre l'absurde, que l'écran n'a pas (SOLLICITATION_MAX, EN.H_MAX pour les longueurs saisies, AP.NOM_MAX).
"""
import copy
import sys
from typing import Dict, List, Optional, Sequence

from beton import treillis as TR
from beton.dalle import entree as EN
from beton.dalle import moteur as MO
from beton.predalle import entree as EN_P
from beton.predalle import moteur as MO_P

from . import commun as C
from . import dalle_nappes as NAP
from . import dalle_positions as POS
from . import dessin_dalle as DD
from . import dessin_predalle as DP
from . import etabli_compact as E
from . import plaque_compact as PC
from . import predalle as AP          # l'AIDE du type prédalle : ses textes, sa pose, ses avertissements (réemployés)

MODULE = "dalle"
PAGE = "dalle"
# l'état fusionné (docs/site/DALLES_FUSION.md § 2.1) ; 3 depuis le 09/11/2026 : les nappes toujours posées l'une sur
# l'autre, plus de clé « empilement » — une note d'une version plus petite est une note D'AVANT (voir CLE_AVIS) ; 4 depuis
# le 13/11/2026 (DAL5b, § 8.6) : un treillis soudé posé tel qu'il est (ses fils en contact, les renforts principaux sur
# le treillis) — l'état ne change pas de forme, mais une note 3 dont les chiffres en changent reçoit l'avis « treillis »
VERSION = 4

# les TYPES de dalle : la clé de l'état et son libellé (les segments de la rubrique « 1 — Caractéristiques »)
# (libellés courts, demande de l'ingénieur du 09/11/2026 : CSP = coulé sur place ; l'aide du champ le dit en entier)
# (les libellés : « Dalle pleine » et « Prédalle », demande du propriétaire du 13/11/2026 — c'étaient « Dalle CSP » et
#  « Prédalle + dalle CSP », les libellés courts du 09/11/2026 ; les clés de l'état ne changent pas)
TYPES_DALLE = (("coulee", "Dalle pleine"), ("predalle", "Prédalle"))
TYPES_TXT = dict(TYPES_DALLE)
# les clés que le SITE ajoute à une dalle et que les moteurs ne lisent pas (api_web/dalle_positions.CLES_SITE les
# retire de la copie envoyée au moteur : tests/test_api_web_dalles_fusion.py lie les deux copies)
CLES_SITE = ("verrou", "type")
# les ESPACEMENTS des barres que le sélecteur de la couche de base propose (13/11/2026, demande du propriétaire —
# docs/site/DALLES_FUSION.md § 8.4) : un réglage de la NOTE (une clé de l'état, à côté de « dalles »), choisi parmi ceux
# du moteur (MO.CANDIDATS_ESP, mm) ; tous par défaut ; jamais aucun. Il ne touche aucun calcul : il trie ce que le
# sélecteur propose (la plus petite qui tient ★ parmi les espacements admis) ; une barre se saisit toujours à la main.
ESPACEMENTS = tuple(MO.CANDIDATS_ESP)


def espacements_relus(x) -> List[int]:
    """Les espacements admis d'une note relue : absents (une note d'avant), tous ; sinon une liste non vide de valeurs du
    moteur (mm), rendue triée — une liste abîmée est refusée avec un message (jamais remplacée en silence)."""
    if x is None:
        return list(ESPACEMENTS)
    if (not isinstance(x, list) or not x or any(isinstance(v, bool) or not isinstance(v, int) or v not in ESPACEMENTS for v in x)):
        raise ValueError(f"espacements admis : une liste non vide parmi {', '.join(map(str, ESPACEMENTS))} mm (reçu {x!r})")
    return sorted(set(x))
# les clés qu'une version d'avant du site écrivait sur chaque dalle et qui ne pilotent plus rien : relues sans effet,
# jamais une raison de refuser la note — « empilement », l'option « nappes empilées » retirée le 09/11/2026 (décision de
# l'ingénieur : les nappes se posent toujours l'une sur l'autre ; vraie, elle disait seulement que la base secondaire
# l'était déjà)
CLES_ANCIENNES = ("empilement",)
# L'AVIS d'une note dont les chiffres changent parce que ses nappes sont désormais posées l'une sur l'autre : dit UNE
# fois, à l'ouverture. ``normaliser`` (une note d'avant) ou la reprise d'un fichier Streamlit le posent sous CLE_AVIS ;
# ``resultat`` le montre en tête du résultat (C.message, un avis comme ceux du moteur, pas une erreur) et le RETIRE de
# l'état : la note rendue, enregistrée par l'écran, est dans la forme d'aujourd'hui et ne le redit pas.
CLE_AVIS = "avis_ouverture"
AVIS = {
    "note": ("Note d'avant le 09/11/2026 : les nappes sont désormais posées l'une sur l'autre (la direction secondaire, "
             "base et renforts, sur la principale) ; A<sub>s,req</sub> peut différer de l'ancienne note."),
    "treillis": ("Note d'avant le 13/11/2026 : un treillis soudé est désormais posé tel qu'il est — ses deux directions de "
                 "fils soudées l'une sur l'autre (dans la peau d'une prédalle), les renforts de la direction principale "
                 "posés sur le treillis ; A<sub>s,req</sub> peut différer de l'ancienne note."),
}
# Les NOMS trop longs d'une note relue (un .stab, une note d'une version d'avant) : un nom de dalle
# ou de section de plus de AP.NOM_MAX caractères faisait REFUSER toute la note (la borne du site, ``_bornes_site``) — la
# note ne s'ouvrait plus. Il est désormais COUPÉ à la limite, et l'avis le dit une fois, à l'ouverture (comme CLE_AVIS :
# ``normaliser`` pose la liste sous CLE_NOMS, ``resultat`` la dit et la retire).
# La SAISIE garde son refus clair (AP.nom_saisi) : un nom ne s'allonge jamais au-delà à l'écran.
CLE_NOMS = "noms_coupes"

# des requêtes de plus pour le parcours (api_web/parcours.py) : les champs que le parcours automatique n'essaie pas —
# pour les deux types (une note de prédalle, prête, pour les chemins propres à la peau)
_NOTE_P = {"predalles": [EN_P.nouvelle_predalle()], "actif": 0}
PARCOURS = [
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_y", "couches", "0", "dist"], "valeur": "5"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_y", "yG"], "valeur": "4,2"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "sup_x", "couches", "0", "dist"], "valeur": ""},
    {"action": "modifier", "chemin": ["sections", "0", "Mx_inf"], "valeur": "-20"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_x", "couches", "0", "esp"], "valeur": "600"},
    {"action": "modifier", "chemin": ["h_pre"], "valeur": "8"},
    {"action": "selecteur", "role": "sup_x-0", "familles": ["treillis"]},
    {"action": "choisir", "role": "inf_y-0", "nom": "Ø12/150"},
    {"action": "modifier", "etat": _NOTE_P, "chemin": ["sections", "0", "faces", "inf_x", "couches", "0", "dist"], "valeur": "8"},
    {"action": "modifier", "etat": _NOTE_P, "chemin": ["h_pre"], "valeur": "21"},
    {"action": "modifier", "etat": _NOTE_P, "chemin": ["h"], "valeur": "6"},
    {"action": "modifier", "etat": _NOTE_P, "chemin": ["h_pre"], "valeur": "3"},
    {"action": "modifier", "etat": _NOTE_P, "chemin": ["type"], "valeur": "coulee"},
    {"action": "selecteur", "etat": _NOTE_P, "role": "inf_y-0"},
    {"action": "choisir", "etat": _NOTE_P, "role": "inf_y-0", "nom": "Ø12/150"},
    {"action": "ajouter", "etat": _NOTE_P},
]

TYPES = PC.TYPES
FACES_TXT = PC.FACES_TXT
MOTIFS = {"hauteur_insuffisante": "hauteur insuffisante", "d_utile_negatif": "d utile ≤ 0",
          "acier_insuffisant": "acier insuffisant", "acier_excessif": "acier excessif (A<sub>s</sub> &gt; A<sub>s,max</sub>)",
          "cisaillement_excessif": "τ &gt; τ<sub>adm,I</sub> (pas d'étriers dans une dalle)"}
LIB_FAMILLES = {"treillis": "Treillis", "barres": "Barres"}


class _Refus(ValueError):
    """Une saisie refusée : son message est montré tel quel."""


# ------------------------------------------------------------------ l'état de la note : des dalles, de deux types
def est_predalle(d: Dict) -> bool:
    return d.get("type") == "predalle"


def _EN(d: Dict):
    """Le module d'ENTRÉE du moteur du type (bornes, défauts, normalisation)."""
    return EN_P if est_predalle(d) else EN


def _MO(d: Dict):
    """Le MOTEUR du type : beton/predalle pour une prédalle, beton/dalle sinon."""
    return MO_P if est_predalle(d) else MO


def _M(d: Dict):
    """Le module dont api_web/plaque_compact.py lit les formules mises en mots : l'aide prédalle (ses formules d'axe,
    sa pose sur la ligne du tableau) pour une prédalle, ce module-ci sinon."""
    return AP if est_predalle(d) else _MOI


def _sans_site(d: Dict) -> Dict:
    """La dalle SANS les clés du site (ni celles d'une version d'avant, CLES_ANCIENNES) — pour la re-normaliser (jamais
    pour le moteur : POS.pour_moteur)."""
    return {k: v for k, v in d.items() if k not in CLES_SITE and k not in CLES_ANCIENNES}


def _site(d: Dict, verrou: bool = False, type_: str = "coulee") -> Dict:
    """Une dalle du site : la dalle du moteur de son type, normalisée, et ses clés propres au site — le verrou (lecture
    seule, comme le 🔒 de l'écran) et le type."""
    if type_ not in TYPES_TXT:
        raise ValueError(f"type : « coulee » ou « predalle » attendu (reçu {type_!r})")
    out = (EN_P if type_ == "predalle" else EN).normaliser(d)
    out["verrou"] = bool(verrou)
    out["type"] = type_
    return out


def _resite(d: Dict, **modifs) -> Dict:
    """La dalle du site ``d`` re-normalisée avec ``modifs`` (γs, jeu…), ses clés du site gardées."""
    return _site(dict(_sans_site(d), **modifs), d["verrou"], d["type"])


def nouvel_etat() -> Dict:
    """Une note neuve : une dalle coulée en place aux défauts de l'écran (ses nappes posées l'une sur l'autre, comme
    toutes : api_web/dalle_positions.py)."""
    return {"version": VERSION, "dalles": [_site(EN.nouvelle_dalle())], "actif": 0, "espacements": list(ESPACEMENTS)}


def chiffres_changes(d: Dict, deja_empilee: bool = False) -> bool:
    """Les chiffres d'une dalle relue (du site, normalisée) changent-ils avec ses nappes posées l'une sur l'autre ? Oui
    dès qu'une couche est posée par le site sur la nappe principale (``POS.empilee`` : base ou renfort secondaire
    automatique) — sauf, pour une dalle dont l'option d'avant était cochée (``deja_empilee``), une BASE secondaire,
    que l'option posait déjà ainsi, sur la base principale : seul un renfort secondaire y change (S3, 09/11/2026), ou
    une base secondaire qu'un renfort principal plus haut porte désormais (S5 : ``POS.appui`` n'est plus la base). Un
    axe saisi ne change pas."""
    return any(POS.empilee(d, s, f, j) and (j > 0 or not deja_empilee or POS.appui(d, s, f.split("_")[0])["couche"] != 0)
               for s in d["sections"] for f in EN.FACES for j in range(len(s["faces"][f]["couches"])))


def chiffres_changes_treillis(d: Dict) -> bool:
    """Les chiffres d'une dalle relue (du site, normalisée) changent-ils avec le treillis posé tel qu'il est (DAL5b,
    VERSION 4) ? Oui dès qu'une couche AUTOMATIQUE a un autre axe que par la règle d'avant (``POS.axe_auto(...,
    treillis_reel=False)`` : le treillis lu comme des barres) — un renfort principal d'un treillis unique, des fils
    secondaires qu'il soulevait, les fils secondaires d'une prédalle. Un axe saisi ne change pas."""
    return any(c.get("dist") is None and POS.axe_auto(d, s, f, j) != POS.axe_auto(d, s, f, j, treillis_reel=False)
               for s in d["sections"] for f in EN.FACES for j, c in enumerate(s["faces"][f]["couches"]))


def _bool_relu(x, quoi: str, rang: int) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"dalle n° {rang + 1} : {quoi} vrai ou faux attendu")
    return x


def _couper_noms(d: Dict) -> List[str]:
    """Coupe à AP.NOM_MAX caractères le nom d'une dalle relue et ceux de ses sections ; rend ce qui a été coupé, en mots
    (HTML échappé) — une liste vide si rien ne l'est."""
    coupes = []
    if isinstance(d.get("nom"), str) and len(d["nom"]) > AP.NOM_MAX:
        n = len(d["nom"])
        d["nom"] = d["nom"][:AP.NOM_MAX].rstrip()
        coupes.append(f"la dalle « {C.esc(d['nom'])}… » ({n} caractères)")
    for s in d.get("sections", []):
        if isinstance(s.get("nom"), str) and len(s["nom"]) > AP.NOM_MAX:
            n = len(s["nom"])
            s["nom"] = s["nom"][:AP.NOM_MAX].rstrip()
            coupes.append(f"la section « {C.esc(s['nom'])}… » de « {C.esc(d.get('nom') or 'Dalle')} » ({n} caractères)")
    return coupes


def avis_noms(coupes: Sequence[str]) -> str:
    """L'avis des noms coupés à la relecture (CLE_NOMS), ou « » s'il n'y en a pas."""
    if not coupes:
        return ""
    return (f"Nom de plus de {AP.NOM_MAX} caractères, coupé à {AP.NOM_MAX} à l'ouverture (la limite du site ; une saisie plus "
            f"longue est refusée) : " + " ; ".join(coupes) + ".")


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran relu par le moteur de chaque dalle ; une dalle illisible (fichier abîmé) lève une erreur
    claire. Trois formes sont relues (§ 2.1) : ``{"dalles": […]}`` d'aujourd'hui (une dalle sans « type » est coulée en
    place), ``{"predalles": […]}`` de l'ancien module Prédalles (chaque prédalle devient une dalle de type « predalle »,
    ``actif`` conservé, rien n'est perdu), et une note qui porte les DEUX clés est abîmée : refusée. Une note sans dalle
    est une note neuve ; une liste qui n'en est pas une est une note abîmée (refusée, jamais remplacée en silence par
    une note neuve). Les nappes sont posées l'une sur l'autre quelle que soit la note : la clé « empilement » d'avant
    (CLES_ANCIENNES) est relue sans effet, quelle que soit sa valeur. Une note D'AVANT (version < 3) dont les chiffres en
    changent (``chiffres_changes``) reçoit l'avis d'ouverture « note » (CLE_AVIS, que ``resultat`` dit et retire) ; une note
    3 dont les chiffres changent avec le treillis posé tel qu'il est (``chiffres_changes_treillis``, DAL5b), l'avis
    « treillis ». Un nom
    de plus de AP.NOM_MAX caractères est coupé à la limite (``_couper_noms``) et l'avis le dit (CLE_NOMS) : la note s'ouvre."""
    if not isinstance(etat, dict):
        return nouvel_etat()
    if "dalles" in etat and "predalles" in etat:
        raise ValueError("note illisible (abîmée) : elle porte à la fois « dalles » et « predalles » — rien n'est remplacé")
    cle = "predalles" if "predalles" in etat else "dalles"
    liste = etat.get(cle)
    if liste is None or liste == []:
        return nouvel_etat()
    if not isinstance(liste, list):
        raise ValueError(f"note illisible : « {cle} » doit être une liste (reçu {type(liste).__name__})")
    version = etat.get("version")
    lue = version if isinstance(version, int) and not isinstance(version, bool) else 0
    d_avant = lue < 3                     # d'avant les nappes posées l'une sur l'autre (09/11/2026) : l'avis « note »
    d_avant_treillis = lue < VERSION      # d'avant le treillis posé tel qu'il est (13/11/2026) : l'avis « treillis »
    out = {"version": VERSION, "dalles": []}
    avis = False
    avis_treillis = False
    coupes = []
    for i, d in enumerate(liste):
        if not isinstance(d, dict):
            raise ValueError(f"dalle n° {i + 1} illisible")
        v = _bool_relu(d.get("verrou", False), "verrou", i)
        t = d.get("type", "predalle" if cle == "predalles" else "coulee")
        if cle == "predalles" and t != "predalle":
            raise ValueError(f"prédalle n° {i + 1} : type « {t} » dans une note de prédalles (note abîmée)")
        if t not in TYPES_TXT:
            raise ValueError(f"dalle n° {i + 1} : type « coulee » ou « predalle » attendu (reçu {t!r})")
        try:
            dd = _site(_sans_site(d), v, t)
            coupes += _couper_noms(dd)
            _bornes_site(dd)
            for s in dd["sections"]:          # un nom vide (possible dans un .stab, pas à l'écran) : la première lettre libre
                if not s["nom"].strip():
                    s["nom"] = EN.nom_section_suivant(dd)
            out["dalles"].append(dd)
        except ValueError as e:
            raise ValueError(f"dalle « {d.get('nom', i + 1)} » : {e}") from None
        avis = avis or (d_avant and chiffres_changes(dd, d.get("empilement") is True))
        avis_treillis = avis_treillis or (d_avant_treillis and chiffres_changes_treillis(dd))
    a = etat.get("actif", 0)
    a = a if isinstance(a, int) and not isinstance(a, bool) else 0
    out["actif"] = min(max(0, a), len(out["dalles"]) - 1)
    out["espacements"] = espacements_relus(etat.get("espacements"))
    if avis:
        out[CLE_AVIS] = "note"              # (il dit déjà que A_s,req peut différer : il couvre le treillis aussi)
    elif avis_treillis:
        out[CLE_AVIS] = "treillis"
    if coupes:
        out[CLE_NOMS] = coupes
    return out


def _actif(etat: Dict) -> Dict:
    return etat["dalles"][etat["actif"]]


def _nom_libre(etat: Dict, base: str = "Dalle") -> str:
    pris = {d["nom"] for d in etat["dalles"]}
    k = len(etat["dalles"]) + 1
    while f"{base} {k}" in pris:
        k += 1
    return f"{base} {k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les dalles de la note : en ouvrir une, en ajouter, en copier une (``index`` ; sans : l'ouverte), en supprimer une
    (jamais la dernière). Une dalle ajoutée prend le TYPE de la dalle ouverte, et le γs et le jeu de la note (paramètres
    communs, comme dans l'écran Streamlit)."""
    ds = etat["dalles"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(ds) - 1)
    elif action == "ajouter":
        ouverte = _actif(etat)
        d = EN_P.nouvelle_predalle() if est_predalle(ouverte) else EN.nouvelle_dalle()
        d["nom"] = _nom_libre(etat, "Prédalle" if est_predalle(ouverte) else "Dalle")
        d["gamma_s"], d["jeu_premier_lit"] = ds[0]["gamma_s"], ds[0]["jeu_premier_lit"]
        ds.append(_site(d, False, ouverte["type"]))
        etat["actif"] = len(ds) - 1
    elif action == "dupliquer":
        # la dalle désignée (le menu « ⋯ » de l'arborescence ; sans rang : l'ouverte), copiée juste après elle
        i = etat["actif"] if index is None else min(max(0, int(index)), len(ds) - 1)
        d = copy.deepcopy(ds[i])
        d["nom"] = f"{d.get('nom') or 'Dalle'} (copie)"
        ds.insert(i + 1, d)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(ds) > 1:
        i = min(max(0, int(index or 0)), len(ds) - 1)
        del ds[i]
        if etat["actif"] > i or etat["actif"] >= len(ds):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
# Des bornes PROPRES AU SITE (les écrans Streamlit n'en ont pas) : elles ne refusent que l'absurde, que l'écran calculait
# en chiffres démesurés (taux de 10¹⁵⁰ %, ou « inf ») :
# - une longueur (jeu premier lit, axe d'une couche, centre de gravité) au plus égale à l'épaisseur maximale d'une
#   dalle, EN.H_MAX ;
# - une sollicitation au plus égale à SOLLICITATION_MAX : la plus forte bande du domaine de l'écran (5 m × 100 cm,
#   C50/60) porte moins de 20 000 kN·m et 4 000 kN ;
# - un nom de dalle ou de section d'au plus AP.NOM_MAX caractères (il titre la note, les coupes, la fenêtre de calcul).
SOLLICITATION_MAX = 1e5
NOMBRES = {  # clé → (libellé, unité, mini, maxi, mini exclu, entier, maxi propre au site)
    "b": ("Largeur de bande", "cm", EN.B_MIN, EN.B_MAX, False, True, False),
    "h": ("Épaisseur", "cm", EN.H_MIN, EN.H_MAX, False, True, False),
    "enrobage": ("Enrobage", "cm", 0.0, EN.ENROBAGE_MAX, False, False, False),
    "gamma_s": ("Coefficient acier γs", "", EN.GAMMA_S_MIN, EN.GAMMA_S_MAX, False, False, False),
    "jeu_premier_lit": ("Jeu premier lit", "cm", 0.0, EN.H_MAX, False, False, True),
    "esp": ("Espacement", "mm", EN.ESP_MIN, EN.ESP_MAX, False, True, False),
    "n": ("Nombre de barres", "", EN.N_MIN, EN.N_MAX, False, True, False),
    "dist": ("Axe de la couche", "cm", 0.0, EN.H_MAX, True, False, True),
    "yG": ("Centre de gravité", "cm", 0.0, EN.H_MAX, True, False, True),
}
SOLLICITATIONS = {"Mx_inf": "M<sub>x,inf</sub>", "Mx_sup": "M<sub>x,sup</sub>", "My_inf": "M<sub>y,inf</sub>",
                  "My_sup": "M<sub>y,sup</sub>", "V": "V"}
NOTE_ENTIERE = ("gamma_s", "jeu_premier_lit")


def _nombre(cle: str, valeur):
    nom, u, mini, maxi, exclu, entier, site = NOMBRES[cle]
    x = C.lire_nombre(valeur)
    if x is None:
        raise _Refus(f"{nom} : nombre illisible « {valeur} »")
    unite = f" {u}" if u else ""
    if (x <= mini) if exclu else (x < mini):
        borne = (f"entre {C.nb(mini)} et {C.nb(maxi)}{unite}" if maxi is not None and not site
                 else (f"plus de {C.nb(mini)}{unite}" if exclu else f"au moins {C.nb(mini)}{unite}"))
        raise _Refus(f"{nom} : {borne} (bornes de l'écran Streamlit), pas {AP.saisi(x)}{unite}")
    if maxi is not None and x > maxi:
        if site:
            raise _Refus(f"{nom} : au plus {C.nb(maxi)}{unite}, l'épaisseur maximale d'une dalle (au-delà, la saisie n'a "
                         f"pas de sens), pas {AP.saisi(x)}{unite}")
        raise _Refus(f"{nom} : entre {C.nb(mini)} et {C.nb(maxi)}{unite} (bornes de l'écran Streamlit), pas {AP.saisi(x)}{unite}")
    if entier:
        if x != int(x):
            raise _Refus(f"{nom} : un nombre entier{unite}, pas {AP.saisi(x)}")
        return int(x)
    return x


def _bornes_site(d: Dict) -> None:
    """Les bornes propres au site (voir SOLLICITATION_MAX) sur une dalle relue (fichier .stab, fichier Streamlit) :
    ValueError claire."""
    if d["jeu_premier_lit"] > EN.H_MAX:
        raise ValueError(f"jeu premier lit de {AP.saisi(d['jeu_premier_lit'])} cm : au plus {EN.H_MAX} cm")
    if len(d["nom"]) > AP.NOM_MAX:
        raise ValueError(f"nom de {len(d['nom'])} caractères : {AP.NOM_MAX} au plus")
    for s in d["sections"]:
        if len(s["nom"]) > AP.NOM_MAX:
            raise ValueError(f"section : nom de {len(s['nom'])} caractères, {AP.NOM_MAX} au plus")
        for k in EN.MOMENTS + ("V",):
            if s[k] > SOLLICITATION_MAX:
                raise ValueError(f"section « {s['nom']} » : {k} au-delà de {C.fr(SOLLICITATION_MAX, 0)} (valeur absurde)")
        for f in EN.FACES:
            face = s["faces"][f]
            if any(x is not None and x > EN.H_MAX for x in [face["yG"]] + [c["dist"] for c in face["couches"]]):
                raise ValueError(f"section « {s['nom']} », nappe {f} : axe ou centre de gravité au-delà de {EN.H_MAX} cm")


def _auto_ou_impose(cle: str, valeur, actuel, auto: float):
    """Un axe ou un centre de gravité : vide → automatique ; la valeur automatique affichée, réécrite → reste
    automatique (comme le champ de l'écran Streamlit) ; un autre nombre > 0 → imposé."""
    if valeur is None or (isinstance(valeur, str) and valeur.strip().lower() in ("", "auto")):
        return None
    x = _nombre(cle, valeur)
    if actuel is None and abs(x - auto) <= 0.5e-3 + 1e-12:
        return None
    return x


def _changer_type(d: Dict, valeur) -> None:
    """Le TYPE de la dalle (§ 2.2) : coulee → predalle ajoute la peau (le défaut de l'écran prédalle, bornée à
    h_pre_max(h)) ; predalle → coulee la retire ; sections, faces, couches, positions imposées et noms sont conservés."""
    t = valeur.strip().lower() if isinstance(valeur, str) else valeur
    if t not in TYPES_TXT:
        raise _Refus(f"Type de dalle : « coulee » ou « predalle », pas « {valeur} »")
    if t == d["type"]:
        return
    if t == "predalle":
        d["h_pre"] = min(max(EN_P.DEFAUTS_PREDALLE["h_pre"], EN_P.H_PRE_MIN), EN_P.h_pre_max(d["h"]))
    else:
        d.pop("h_pre", None)
    d["type"] = t


def _ecrire(etat: Dict, d: Dict, chemin: List, valeur) -> None:
    """Écrit une saisie dans la dalle ``d`` (une copie) ; _Refus (ou AP.Refus) si la valeur sort du domaine de l'écran.
    Les chemins « sections/i/nappes/… » vont à api_web/dalle_nappes.ecrire ; les positions (axe automatique) à
    api_web/dalle_positions."""
    k = chemin[0]
    predalle = est_predalle(d)
    if len(chemin) == 1:
        if k == "nom":
            d["nom"] = AP.nom_saisi(valeur, "de la dalle", True)
        elif k == "type":
            _changer_type(d, valeur)
        elif k == "beton":
            if valeur not in MO.classes_beton():
                raise _Refus(f"Béton : classe « {valeur} » inconnue")
            d["beton"] = valeur
        elif k == "fyk":
            x = C.lire_nombre(valeur)
            if x not in EN.NUANCES:
                raise _Refus(f"Acier : B400 ou B500, pas « {valeur} »")
            d["fyk"] = int(x)
        elif k == "dir_principale":
            if str(valeur).upper() not in ("X", "Y"):
                raise _Refus(f"Direction principale : X ou Y, pas « {valeur} »")
            d["dir_principale"] = str(valeur).upper()
        elif k == "h":
            # une prédalle : l'épaisseur totale doit encore contenir la peau (le refus de l'ancien module)
            d["h"] = AP.epaisseur_totale(d, valeur) if predalle else _nombre("h", valeur)
        elif k == "h_pre":
            if not predalle:
                raise _Refus("Épaisseur de la prédalle : une dalle coulée en place n'a pas de prédalle — choisir "
                             f"« {TYPES_TXT['predalle']} » d'abord")
            d["h_pre"] = AP.peau(d, valeur)
        elif k in ("b", "enrobage"):
            d[k] = _nombre(k, valeur)
        else:
            raise _Refus(f"champ inconnu : {k}")
        return
    if k != "sections":
        raise _Refus(f"champ inconnu : {'/'.join(map(str, chemin))}")
    secs = d["sections"]
    if chemin[1] == "+":
        secs.append(EN.nouvelle_section(EN.nom_section_suivant(d)))
        return
    i = chemin[1]
    s = secs[i]
    if chemin[2] == "nappes":
        try:
            NAP.ecrire(d, chemin, valeur)
        except ValueError as e:
            raise _Refus(str(e)) from None
        return
    if chemin[2] == "-":
        if len(secs) <= 1:
            raise _Refus("Il faut au moins une section")
        del secs[i]
        return
    if chemin[2] == "copie":
        c = copy.deepcopy(s)
        c["nom"] = EN.nom_section_suivant(d)
        secs.append(c)
        return
    if chemin[2] == "nom":
        s["nom"] = AP.nom_saisi(valeur, "de section", False)
        return
    if chemin[2] in SOLLICITATIONS:
        x = C.lire_nombre(valeur)
        if x is None:
            raise _Refus(f"Nombre illisible : « {valeur} »")
        if x < 0:
            # (13/11/2026 : les colonnes s'appellent M⁺ et M⁻ — le signe est celui de la colonne, la valeur se saisit
            #  sans lui ; un négatif est refusé, jamais ramené à 0 ni retourné en silence)
            if chemin[2] == "V":
                raise _Refus(f"L'effort tranchant se saisit sans signe : {AP.saisi(-x)}, pas {AP.saisi(x)}.")
            raise _Refus("Sans signe : la colonne dit déjà la face tendue (M⁺ : la face inférieure, M⁻ : la face "
                         f"supérieure) — saisir {AP.saisi(-x)}, pas {AP.saisi(x)}.")
        if x > SOLLICITATION_MAX:
            u = "kN" if chemin[2] == "V" else "kN·m"
            raise _Refus(f"Au plus {C.fr(SOLLICITATION_MAX, 0)} {u}, pas {AP.saisi(x)} {u} : la plus forte bande de dalle "
                         "(5 m de large, 100 cm d'épaisseur) porte moins de 20 000 kN·m et 4 000 kN.")
        s[chemin[2]] = x
        return
    f = chemin[3]
    face = s["faces"][f]
    if chemin[4] == "yG":
        face["yG"] = _auto_ou_impose("yG", valeur, face["yG"], POS.yG_calcule(d, s, f))
        return
    couches = face["couches"]
    if chemin[5] == "+":
        if len(couches) >= EN.MAX_COUCHES:
            raise _Refus(f"Au plus {EN.MAX_COUCHES} couches par face (la base et trois renforts)")
        couches.append(EN.nouvelle_couche(base=False))
        return
    j = chemin[5]
    c = couches[j]
    champ = chemin[6]
    if champ == "-":
        if j == 0:
            raise _Refus("La couche de base ne se retire pas ; changer son type ou sa nappe")
        del couches[j]
    elif champ == "type":
        if valeur not in dict(TYPES):
            raise _Refus(f"Type de couche inconnu : « {valeur} »")
        c["type"] = valeur
    elif champ == "treillis":
        if not TR.est_canonique(valeur):
            raise _Refus(f"Treillis : désignation Øl/Øt/el/et attendue, pas « {valeur} »")
        c["treillis"] = valeur
    elif champ == "d":
        x = C.lire_nombre(valeur)
        if x not in EN.DIAMETRES:
            raise _Refus(f"Ø{valeur} n'est pas proposé ({' '.join(map(str, EN.DIAMETRES))})")
        c["d"] = int(x)
    elif champ in ("esp", "n"):
        c[champ] = _nombre(champ, valeur)
    elif champ == "dist":
        c["dist"] = _auto_ou_impose("dist", valeur, c["dist"], POS.axe_auto(d, s, f, j))
    elif champ == "deplacer":
        # (DAL5c : les flèches ▲▼ de l'axe — « 1 » monte la couche, « -1 » la descend, d'un pas de POS.PAS_AXE ; le
        #  refus dit ce qu'elle traverserait ; revenue sur son axe automatique, elle le redevient et suit à nouveau)
        sens = {"1": 1, "+1": 1, "-1": -1}.get(str(valeur).strip())
        if sens is None:
            raise _Refus(f"Déplacer une couche : « 1 » (monter) ou « -1 » (descendre), pas « {valeur} »")
        try:
            e = POS.deplacer(d, i, f, j, sens)
        except ValueError as x:
            raise _Refus(str(x)) from None
        c["dist"] = None if abs(e - POS.axe_auto(d, s, f, j)) < 1e-9 else e
    else:
        raise _Refus(f"champ inconnu : {champ}")


def _lire_bool(valeur):
    """Une case à cocher telle que l'écran l'envoie (vrai/faux, « true »/« false », « 1 »/« 0 ») ; None sinon."""
    return {True: True, False: False, "true": True, "false": False, "1": True, "0": False}.get(
        valeur.strip().lower() if isinstance(valeur, str) else valeur)


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la dalle ouverte. ``chemin`` : les clés (« sections », 0, « faces », « inf_y », « couches »,
    0, « type ») ; les boutons d'ajout, de copie et de retrait passent par le même chemin, terminé par « + », « copie »
    ou « - ». Une valeur hors du domaine de l'écran est refusée avec un message, et la dalle reste intacte. « verrou »,
    « type », « h_pre » sont ceux de la dalle ouverte ; γs et le jeu ceux de la NOTE (toutes ses dalles)."""
    chemin = [int(c) if isinstance(c, str) and c.isdigit() else c for c in chemin]
    a = etat["actif"]
    d = etat["dalles"][a]
    if chemin == ["verrou"]:
        lu = _lire_bool(valeur)
        if lu is None:
            return f"Verrou : coché ou décoché, pas « {valeur} »"
        d["verrou"] = lu
        return None
    if d["verrou"]:
        return "Dalle verrouillée : le cadenas, à droite de son nom, la déverrouille"
    if chemin == ["espacements"]:
        # une puce des options : son espacement entre dans la liste ou en sort (jamais le dernier)
        x = C.lire_nombre(valeur)
        if x is None or x != int(x) or int(x) not in ESPACEMENTS:
            return f"Espacement : l'un de {', '.join(C.nb(e / 10) for e in ESPACEMENTS)} cm, pas « {valeur} »"
        liste = etat.setdefault("espacements", list(ESPACEMENTS))
        if int(x) in liste:
            if len(liste) == 1:
                return "Au moins un espacement : sans lui, le sélecteur n'aurait plus de barres à proposer"
            liste.remove(int(x))
        else:
            liste.append(int(x))
            liste.sort()
        return None
    if chemin and chemin[0] in NOTE_ENTIERE and len(chemin) == 1:
        # γs et le jeu premier lit sont des paramètres de la NOTE (l'écran Streamlit les partage entre ses dalles)
        try:
            x = _nombre(chemin[0], valeur)
        except _Refus as e:
            return str(e)
        for k, dd in enumerate(etat["dalles"]):
            etat["dalles"][k] = _resite(dd, **{chemin[0]: x})
        return None
    travail = copy.deepcopy(d)
    try:
        _ecrire(etat, travail, chemin, valeur)
        etat["dalles"][a] = _site(_sans_site(travail), d["verrou"], travail["type"])
    except (_Refus, AP.Refus) as e:
        return str(e)
    except ValueError as e:
        return f"Saisie refusée : {e}"
    except (KeyError, IndexError, TypeError, AttributeError):
        return f"Saisie refusée : champ introuvable « {'/'.join(map(str, chemin))} »"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur de la nappe de base : « inf_y-0 » (nappe inférieure Y de la section 0) reçoit « Treillis 8/8/150/150 »
    ou « Ø12/150 » ; ses renforts, son axe imposé et son centre de gravité imposé sont gardés. Sur une nappe qui était
    un TREILLIS UNIQUE (§ 3), un treillis choisi pose aussi sa TRANSPOSÉE sur l'autre direction : la nappe reste un
    treillis unique (des barres choisies la laissent « incohérente », et l'écran le dit)."""
    face, i = role.rsplit("-", 1)
    d = _actif(etat)
    if d["verrou"]:
        raise ValueError("dalle verrouillée")
    cand = MO.lire_candidat(nom)
    travail = copy.deepcopy(d)
    s = travail["sections"][int(i)]
    s["faces"][face]["couches"][0].update(cand)
    cote, dk = face.split("_")
    if cand["type"] == "treillis" and NAP.mode(d["sections"][int(i)], cote, d["dir_principale"])["mode"] in ("treillis", "treillis_renforts"):
        autre = s["faces"][f"{cote}_{'y' if dk == 'x' else 'x'}"]["couches"][0]
        autre.update({"type": "treillis", "treillis": NAP.transposee(cand["treillis"])})
    etat["dalles"][etat["actif"]] = _site(_sans_site(travail), False, d["type"])


# ------------------------------------------------------------------ les données : la colonne de gauche
# la légende des dessins (dernière ligne de l'en-tête de chaque vue : ce qu'on y lit) — celle de la prédalle : AP.LEGENDES
LEGENDES = {"c": "barres coupées en points, barres parallèles en traits", "p": "vue de dessus · barres X horizontales, Y verticales"}
AIDE_AXE = ("Du parement à l'axe des barres. Vide : l'axe automatique — direction principale : enrobage + ½ Ø arrondi au "
            "0,5 cm supérieur + jeu premier lit ; direction secondaire (sa base et ses renforts), posée sur la principale, sur "
            "la plus haute des couches principales qu'elle croise (la base, ou un renfort plus épais) : son axe + ½ Ø_p + ½ Ø. "
            "Un treillis soudé : ses fils de répartition soudés sur les porteurs, les renforts de la direction principale posés "
            "sur le treillis. Une valeur saisie l'impose.")
AIDE_TYPE = ("Dalle pleine : toute l'épaisseur est coulée sur place (moteur beton/dalle). Prédalle : la partie basse h_p est "
             "préfabriquée, la base inférieure de la direction principale y est posée en fond de moule — un treillis soudé y est "
             "en entier, ses deux directions de fils —, les autres couches inférieures sur la prédalle (moteur beton/predalle). "
             "Changer de type garde les sections, les armatures et les positions saisies ; vers la prédalle, h_p prend 6 cm "
             "(borné à l'épaisseur).")
AIDE_RENFORT = ("Un renfort : treillis, barres Ø/e ou n barres ; 3 Ø12 par défaut, posées dans le plan de la base — sur un "
                "treillis soudé, un renfort de la direction principale est posé sur le treillis")


def _legendes(d: Dict) -> Dict[str, str]:
    return AP.LEGENDES if est_predalle(d) else LEGENDES


def _dims(d: Dict) -> str:
    """« 100 × 20 cm · C30/37 · B500 » — une prédalle dit sa peau : « 100 × 22 cm, prédalle 6 cm · C30/37 · B500 »
    (l'en-tête des résultats, le sous-titre des vues), comme l'ancien module Prédalles."""
    peau = f", prédalle {C.nb(d['h_pre'])} cm" if est_predalle(d) else ""
    return f"{C.nb(d['b'])} × {C.nb(d['h'])} cm{peau} · {C.esc(d['beton'])} · B{d['fyk']}"


def _lecture(label: str, valeur: str, sym: str = "") -> str:
    return C.ligne(label, f'<span class="dl-ro">{valeur}</span>', sym)


def _formulaire_verrouille(d: Dict, r: Dict) -> str:
    """Une dalle verrouillée : ses données en lecture seule (son type aussi) ; le cadenas de la ligne de la dalle la
    déverrouille (13/11/2026 : ce n'est plus une case de la rubrique)."""
    peau = (_lecture("prédalle + coulé en place", f"{C.nb(d['h_pre'])} + {C.nb(d['h'] - d['h_pre'])} cm", "h<sub>p</sub>")
            if est_predalle(d) else "")
    s1 = (C.aide("La dalle est verrouillée : ses données se lisent, elles ne se modifient pas. Le cadenas, à droite de son "
                 "nom, la déverrouille.")
          + _lecture("type", C.esc(TYPES_TXT[d["type"]]))
          + _lecture("béton, acier", f"{C.esc(d['beton'])} · B{d['fyk']}")
          + _lecture("", f"{C.nb(d['b'])} cm", "b")
          + _lecture("", f"{C.nb(d['h'])} cm", "h") + peau
          + _lecture("", f"{C.nb(d['enrobage'])} cm", "c") + _lecture("principale", d["dir_principale"]))
    s2 = ""
    for s, rs in zip(d["sections"], r["sections"]):
        mom = " · ".join(f"{SOLLICITATIONS[k]} {C.nb(s[k])}" for k in EN.MOMENTS)
        s2 += f'<div class="grp">{C.esc(rs["titre"])}</div>' + _lecture("moments (kN·m)", mom) + _lecture("effort tranchant", f"{C.nb(s['V'])} kN", "V")
        for f in MO.ordre_faces(rs["principale"]):
            s2 += _lecture(PC.titre_face(f).lower() + (" · principale" if f.endswith(rs["principale"]) else ""),
                           C.esc(rs["faces"][f]["libelle"]))
    return (C.section(1, f"Caractéristiques {C.icon('lock')}", "verrouillée", s1, True, "dl")
            + C.section(2, "Sections et armatures", f"{len(d['sections'])} section(s)", s2, False, "sl"))


def _options(d: Dict, espacements: Sequence[int] = ESPACEMENTS) -> str:
    """La rubrique « 4 — Options » : la méthode (une ligne), puis γs et le jeu du premier lit, communs à toute la note,
    et les ESPACEMENTS admis des barres proposées (des puces, une par espacement du moteur, allumée quand il est admis :
    un appui l'ajoute ou le retire — 13/11/2026, § 8.4).
    (La reprise d'une note Streamlit est retirée le 13/11/2026 : le bureau n'a aucune ancienne note à reprendre ; l'option
    « nappes empilées » n'existe plus depuis le 09/11/2026 : les nappes sont toujours posées l'une sur l'autre.)"""
    jeu = ("Écart ajouté entre l'enrobage et le premier lit des nappes SUPÉRIEURES (les nappes inférieures sont posées "
           "dans et sur la prédalle)." if est_predalle(d) else
           "Écart ajouté entre l'enrobage et le premier lit : il éloigne du parement l'axe automatique de la direction "
           "principale ; la secondaire, posée dessus, le suit.")
    return (f'<div class="row full et-meth"><label><span>Méthode</span></label>'
            f'<b title="{C.esc(_MO(d).REFERENCE)}">Méthode du bureau · ELS · z = 0,9 d</b></div>'
            + C.ligne2(("γ<sub>s</sub> acier", C.num(["gamma_s"], d["gamma_s"], "", mini=EN.GAMMA_S_MIN, maxi=EN.GAMMA_S_MAX),
                        "Coefficient partiel de l'acier : f_yd = f_yk / γs (1,5 : méthode du bureau). Commun à toutes les "
                        "dalles de la note."),
                       ("Jeu 1er lit", C.num(["jeu_premier_lit"], d["jeu_premier_lit"], "cm", mini=0),
                        f"{jeu} Commun à toutes les dalles de la note."))
            + _puces_espacements(espacements))


def _puces_espacements(espacements: Sequence[int]) -> str:
    """Les espacements admis des barres que le sélecteur propose : une puce par espacement du moteur (en cm), allumée
    quand il est admis ; ``data-bascule`` : un appui sur une puce allumée l'éteint (js/etabli.js l'envoie quand même)."""
    puces = "".join(f'<button type="button" class="fchip{" on" if e in espacements else ""}" data-chemin="espacements" '
                    f'data-valeur="{e}" data-bascule="1" aria-pressed="{"true" if e in espacements else "false"}">{C.nb(e / 10)}</button>'
                    for e in ESPACEMENTS)
    aide = ("Les espacements des barres que le sélecteur de la couche de base propose (la plus petite qui tient ★ est "
            "choisie parmi eux). Une barre se saisit toujours à la main, à n'importe quel espacement. Commun à toute la note.")
    return (f'<div class="row full et-esp"><label><span>Espacements</span><span class="aide" title="{C.esc(aide)}">{C.icon("info")}</span>'
            f'</label><div class="et-puces">{puces}<span class="u">cm</span></div></div>')


def _caracteristiques(d: Dict) -> str:
    """La rubrique « 1 — Caractéristiques » (§ 2.2, revue le 13/11/2026 — § 8.2) : le type en deux segments SANS libellé
    (« Dalle pleine » · « Prédalle » : le choix se lit, son explication au survol), béton et acier, puis les dimensions
    en SYMBOLES — b (et son ⓘ : la bande de calcul de 1 m), h, c —, la direction principale ; une prédalle ajoute h_p
    (bornes H_PRE_MIN..h_pre_max(h)) et le COMPLÉMENT coulé en place h − h_p, en lecture seule (recalculé à chaque saisie,
    jamais stocké). Les mots « bande », « épaisseur », « enrobage » ne se répètent plus à côté du symbole : le survol du
    symbole les dit. Le verrou n'est plus une case de cette rubrique : c'est le cadenas de la ligne de la dalle."""
    predalle = est_predalle(d)

    def sym(lettre: str, mot: str) -> str:
        return f'<span class="et-sym" title="{C.esc(mot)}">{lettre}</span>'
    aide_c = ("Du parement au nu des armatures (fond de moule de la prédalle, dessus de dalle). 0 est lu 3 cm." if predalle
              else "Du parement au nu des armatures. 0 est lu 3 cm.")
    aide_p = ("La base inférieure de la direction principale est posée DANS la prédalle (fond de moule) ; la base secondaire et "
              "tous les renforts inférieurs SUR la prédalle. Elle ordonne aussi les vérifications et les coupes." if predalle
              else "La direction principale ordonne les vérifications, les coupes et la note ; aucune formule ne la lit.")
    peau = ""
    if predalle:
        maxi = EN_P.h_pre_max(d["h"])
        peau = C.ligne2((sym("h<sub>p</sub>", "Épaisseur de la prédalle"), C.num(["h_pre"], d["h_pre"], "cm", mini=EN_P.H_PRE_MIN, maxi=maxi),
                         f"Épaisseur de la prédalle préfabriquée : de {C.nb(EN_P.H_PRE_MIN)} à {C.nb(maxi)} cm ici (au moins "
                         f"{C.nb(EN_P.COULE_MIN)} cm coulé en place, {C.nb(EN_P.H_PRE_MAX)} cm au plus). Elle fixe la position des "
                         "couches inférieures."),
                        (sym("h − h<sub>p</sub>", "Épaisseur coulée en place"), f'<span class="dl-ro">{C.nb(d["h"] - d["h_pre"])} cm</span>',
                         "Épaisseur du béton coulé en place au chantier, sur la prédalle : h − h_p, recalculée à chaque saisie."))
    return ('<div class="et-piece et-plat" data-cible="beton">'
            + f'<div class="row full et-type" title="{C.esc(AIDE_TYPE)}">{C.segments(["type"], d["type"], TYPES_DALLE)}</div>'
            + C.ligne2(("Béton", C.choix(["beton"], d["beton"], [(k, k) for k in MO.classes_beton()]), ""),
                       ("Acier", C.choix(["fyk"], d["fyk"], [(n, f"B{n}") for n in EN.NUANCES]), ""))
            + C.ligne2((sym("b", "Largeur de la bande de calcul"), C.num(["b"], d["b"], "cm", entier=True, mini=EN.B_MIN, maxi=EN.B_MAX),
                        "Bande de calcul : 100 cm = 1 m de dalle. Les moments et l'effort tranchant saisis sont ceux de la bande."),
                       (sym("h", "Épaisseur totale (prédalle et béton coulé en place)" if predalle else "Épaisseur de la dalle"),
                        C.num(["h"], d["h"], "cm", entier=True, mini=EN.H_MIN, maxi=EN.H_MAX), ""))
            + peau
            + C.ligne2((sym("c", "Enrobage"), C.num(["enrobage"], d["enrobage"], "cm", mini=0, maxi=EN.ENROBAGE_MAX), aide_c),
                       ("Principale", C.segments(["dir_principale"], d["dir_principale"], [("Y", "Y"), ("X", "X")]), aide_p))
            + "</div>")


def _aide_axe(d: Dict):
    """Ce que dalle_nappes.html passe à plaque_compact.couche pour chaque couche : un texte d'aide (dalle coulée), ou,
    pour une prédalle, la fonction qui donne la POSE de la couche (son pictogramme, sa règle : api_web/predalle)."""
    if not est_predalle(d):
        return AIDE_AXE
    return lambda f, j, c, g: {"details": AP.pose_couche(c, g), "aide_axe": AP.POSES_AIDE[g["pose"]], "dit_auto": False}


def _aide_renfort(d: Dict):
    if not est_predalle(d):
        return AIDE_RENFORT
    return lambda f: f"Un renfort : treillis, barres Ø/e ou n barres ; 3 Ø12 par défaut, posées {'sur la prédalle' if f.startswith('inf') else 'dans le plan de la base'}"


def formulaire(etat: Dict, r: Optional[Dict] = None) -> str:
    """Les quatre rubriques de la dalle ouverte, aux places de celles de la poutre : 1 Caractéristiques (le type et les
    dimensions), 2 Sections et sollicitations, 3 Armatures (de la section courante, par nappe : api_web/dalle_nappes),
    4 Options. ``r`` : son calcul (``_calculer``), refait s'il manque."""
    d = _actif(etat)
    r = r or _calculer(d)
    if d["verrou"]:
        return _formulaire_verrouille(d, r)
    EN_ = _EN(d)
    contenus, resumes = [], []
    for i, (s, rs) in enumerate(zip(d["sections"], r["sections"])):
        verifs = {v["cle"]: v for v in rs["verifications"]}
        contenus.append(NAP.html(EN_, i, s, rs, verifs, rs["principale"], _aide_axe(d), _aide_renfort(d), False))
        resumes.append(PC.resume_armatures(rs))
    s3, r3 = PC.rubrique_armatures([s["nom"] for s in d["sections"]], contenus, resumes)
    # le résumé de la rubrique 1 dit le TYPE
    resume1 = (f"Prédalle h<sub>p</sub> = {C.nb(d['h_pre'])} cm · {C.esc(d['beton'])} · B{d['fyk']}" if est_predalle(d)
               else f"Dalle pleine · {C.esc(d['beton'])} · B{d['fyk']} · c = {C.nb(d['enrobage'])} cm")
    return PC.formulaire("dl", "Caractéristiques", resume1, _caracteristiques(d), PC.cartes(d["sections"]), len(d["sections"]),
                         s3, r3, f"γ<sub>s</sub> {C.nb(d['gamma_s'])} · jeu {C.nb(d['jeu_premier_lit'])} cm",
                         _options(d, etat.get("espacements", ESPACEMENTS)))


def arbre(etat: Dict, r: Optional[Dict] = None, niveaux: Optional[List[str]] = None) -> str:
    """La colonne de gauche COMPACTE (api_web/etabli_compact.arbre_compact — docs/site/DALLES_FUSION.md § 1) : la ligne
    de la dalle ouverte (son nom, qu'un clic modifie ; la liste des dalles « nom · b × h cm », la peau d'une prédalle
    dite, un cadenas si elle est verrouillée, et « Ajouter une dalle » ; à droite le cadenas, copier, supprimer), posée
    par l'écran dans l'en-tête du panneau, puis les rubriques de la dalle ouverte. ``niveaux`` (un par dalle, ceux du
    résultat : PC.niveau_element) : l'état de chaque dalle, dit par la couleur de son nom (§ 8.2)."""
    elements = [(d.get("nom") or f"Dalle {i + 1}",
                 f"· {C.nb(d['b'])} × {C.nb(d['h'])} cm" + (f", prédalle {C.nb(d['h_pre'])} cm" if est_predalle(d) else ""), d["verrou"])
                for i, d in enumerate(etat["dalles"])]
    return E.arbre(E.DALLE, elements, etat["actif"], formulaire(etat, r), "dalle", compact=True, cadenas=True, niveaux=niveaux)


# ------------------------------------------------------------------ le résultat
def _niv(v: Dict) -> str:
    """La couleur d'une vérification : le taux du moteur, et rouge dès qu'il la déclare non conforme (acier excessif,
    d utile ≤ 0…) même sous 100 %."""
    return "r" if v.get("etat") == MO.NOK else C.niveau(v["taux"])


def _n(x: Optional[float], unite: str = "") -> str:
    """Un RÉSULTAT écrit avec la précision d'usage : mm² entiers, cm au millimètre, contraintes au centième."""
    if x is None:
        return "—"
    nd = {"mm²": 0, "mm²/m": 0, "cm": 1, "N/mm²": 2, "kN": 1, "kN·m": 1, "mm": 0}.get(unite)
    if nd is None:
        a = abs(x)
        nd = 0 if a >= 1000 else (1 if a >= 10 else 2)
    return C.fr(x, nd)


def _v(x: Optional[float]) -> str:
    """Une DONNÉE d'une formule écrite sans la dénaturer (12,96 reste 12,96 ; 0,1709 reste 0,1709) ; seules les
    grandeurs calculées longues sont arrondies."""
    if x is None:
        return "—"
    a = abs(x)
    if a >= 100:
        return C.nb(round(x, 1), 1)
    if a >= 10:
        return C.nb(round(x, 2), 2)
    return C.nb(round(x, 4), 4)


def nom_ligne(v: Dict) -> str:
    if v["cle"] == "hauteur":
        return "Hauteur"
    if v["cle"] == "tau":
        return "Effort tranchant"
    return f"{FACES_TXT[v['face']].capitalize()} {v['direction'].upper()}"


def nom_minuscule(v: Dict) -> str:
    """Le nom dans une phrase : « inférieure Y », « hauteur »."""
    n = nom_ligne(v)
    return n[0].lower() + n[1:]


def _piece(v: Dict, s: Dict) -> str:
    g = s["grandeurs"]
    if v["cle"] == "hauteur":
        return f"h = {C.nb(g['h'])} cm"
    if v["cle"] == "tau":
        return f"V = {C.nb(s['V'])} kN"
    geo = s["faces"][v["cle"]]
    cs = geo["couches"]
    if len(cs) == 1:
        return C.esc(cs[0]["libelle"])
    # avec des renforts, la base s'écrit par sa seule désignation (« 10/10/100/100 » se lit treillis) : la ligne tient
    if len(cs) == 2:
        return C.esc(f"{cs[0]['valeur']} + {cs[1]['valeur']}")
    return C.esc(f"{cs[0]['valeur']} + {len(cs) - 1} renforts")


def _egal(a: str, b: str) -> str:
    """« a = b » sans retour à la ligne (espaces insécables) : une comparaison se coupe à son signe, jamais au milieu
    d'un membre (« 0,56 » d'un côté, « N/mm² » de l'autre)."""
    return f"{a}{C.NBSP}={C.NBSP}{b}"


def _comparaison(v: Dict, s: Dict):
    """(valeur, limite) de la vérification, en HTML : « 707 mm² » face à « A_s = 785 mm² »."""
    g = s["grandeurs"]
    N = C.NBSP
    if v["cle"] == "hauteur":
        # (DAL8 : la part calculée + l'enrobage, « h_min = 12,6 + 3,5 = 16,1 cm » — plaque_compact.valeur_hauteur)
        return PC.valeur_hauteur(sys.modules[__name__], v), _egal("h", f"{C.nb(g['h'])}{N}cm")
    if v["cle"] == "tau":
        return (_egal("τ", f"{_n(g['tau'], 'N/mm²')}{N}N/mm²"),
                _egal("τ<sub>adm,I</sub>", f"{_n(g['tau_adm'], 'N/mm²')}{N}N/mm²"))
    return f"{_n(v['valeur'], 'mm²')}{N}mm²", _egal("A<sub>s</sub>", f"{_n(v['limite'], 'mm²')}{N}mm²")


def _motifs(v: Dict) -> str:
    return ", ".join(MOTIFS.get(m, C.esc(m)) for m in v.get("motifs", []))


def _sous_limite_non_conforme(v: Dict) -> bool:
    """Une vérification que le moteur déclare non conforme alors que son taux tient (acier excessif, d utile ≤ 0)."""
    return v["etat"] == MO.NOK and v["taux"] <= 1


def _cle_gouv(v: Dict):
    return (v["etat"] == MO.NOK, v["taux"])


def gouvernante(r: Dict):
    """(indice de section, vérification) que le verdict nomme : la plus chargée (celle du moteur, r["gouvernant"]) ;
    dans une dalle NON CONFORME, la plus chargée de celles que le moteur déclare non conformes — acier excessif ou
    d utile ≤ 0 le sont sous 100 %, et le verdict ne doit pas nommer une vérification conforme. On CHOISIT parmi les
    vérifications du moteur ; rien n'est recalculé. À égalité, la première dans l'ordre de l'écran (comme le moteur)."""
    return max(((i, v) for i, s in enumerate(r["sections"]) for v in s["verifications"]), key=lambda x: _cle_gouv(x[1]))


def gouvernante_section(s: Dict) -> Dict:
    """La vérification qui gouverne une section, selon la même règle que gouvernante()."""
    return max(s["verifications"], key=_cle_gouv)


def _verdict(v: Dict, s: Dict) -> str:
    """La comparaison du verdict : « valeur ≤ limite » ; pour une vérification non conforme sous 100 %, la comparaison
    qui la rend non conforme (A_s > A_s,max) ; pour une face dont le d utile est ≤ 0, quel que soit son taux, « d ≤ 0 »."""
    g, N = s["grandeurs"], C.NBSP
    if _sous_limite_non_conforme(v) and "acier_excessif" in v["motifs"]:
        return (f"{_egal('A<sub>s</sub>', _n(v['limite'], 'mm²') + N + 'mm²')} &gt; "
                f"{_egal('A<sub>s,max</sub>', _n(v['As_max'], 'mm²') + N + 'mm²')} · acier excessif")
    if "d_utile_negatif" in v["motifs"]:          # (quel que soit le taux : la raison, et non un taux sans objet)
        # (court : la ligne du résultat tient sur une ligne au PC — « position des couches incompatible avec
        #  l'épaisseur » y était coupée, 13/11/2026 ; la fenêtre de calcul et la note PDF gardent la phrase longue)
        return f"{_egal('d', _cm(g['d_' + v['cle']]) + N + 'cm')} ≤ 0 · nappe hors de l'épaisseur"
    val, lim = _comparaison(v, s)
    motif = f" · {_motifs(v)}" if v["motifs"] and v["taux"] <= 1 else ""
    return f"{val} {'≤' if v['taux'] <= 1 else '&gt;'} {lim}{motif}"


def _cm(x: float) -> str:
    """Une longueur CALCULÉE en cm (y_G, d, axe) au plus au 1/100 : 3,5 ; 4,86 ; 3,8 (et non 3,7951)."""
    return C.nb(round(x, 2), 2)


def _formule(f: Dict) -> Dict:
    """Une formule du moteur → « symbole = expression = valeurs substituées = résultat »."""
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    u = f["unite"]
    r = _n(f["resultat"], u) + (f" {u}" if u else "")
    k = f["cle"]
    if k == "hu_min":
        parts = (["h_u,min", "√(M·10⁶ / (α_b·b·μ)) / 10",
                  f"√({_v(t['M'])} × 10⁶ / ({_v(t['alpha_b'])} × {_v(t['b'])} × {_v(t['mu'])})) / 10"] if t["M"] > 0
                 else ["h_u,min", "aucun moment appliqué"])
    elif k == "h_min":
        parts = ["h_min", "h_u,min + y_G", f"{_n(t['hu_min'], 'cm')} + {_cm(t['yG'])}"]
    elif k == "d":
        parts = ["d", "h − y_G", f"{_v(t['h'])} − {_cm(t['yG'])}"]
    elif k == "yG":
        cs = list(t.values())
        somme_ae = " + ".join(_n(a, "mm²") + " × " + _cm(e) for a, e in cs)
        somme_a = " + ".join(_n(a, "mm²") for a, _ in cs)
        parts = ["y_G", "Σ A_s·e / Σ A_s", f"({somme_ae}) / ({somme_a})"]
    elif k == "As":
        parts = ["A_s", "A_s/m · b / 100", f"{_n(t['As_pm'], 'mm²')} × {_v(t['b'])} / 100"]
    elif k == "As_req":
        parts = (["A_s,req", "M·10⁶ / (f_yd·0,9·d)", f"{_v(t['M'])} × 10⁶ / ({_v(t['fyd'])} × 0,9 × {_v(t['d'] * 10)})"] if t["M"] > 0
                 else ["A_s,req", "aucun moment sur cette face"])
    elif k == "As_min_ec":
        parts = ["A_s,min,EC", "0,26·f_ctm / f_yk·b·h", f"0,26 × {_v(t['fctm'])} / {_v(t['fyk'])} × {_v(t['b'])} × {_v(t['h'])}"]
    elif k == "As_min_plancher":
        parts = ["A_s,min,pl", "0,0013·b·h", f"0,0013 × {_v(t['b'])} × {_v(t['h'])}"]
    elif k == "As_min_opposee":
        parts = ["A_s,min,opp", "0,25·A_s,req,opp", f"0,25 × {_n(t['As_req_opp'], 'mm²')}"]
    elif k == "As_min":
        parts = ["A_s,min", "max(A_s,min,EC ; A_s,min,pl ; A_s,min,opp)",
                 f"max({_n(t['As_min_ec'], 'mm²')} ; {_n(t['As_min_plancher'], 'mm²')} ; {_n(t['As_min_opposee'], 'mm²')})"]
    elif k == "besoin":
        parts = ["A_s,nécessaire", "max(A_s,req ; A_s,min)", f"max({_n(t['As_req'], 'mm²')} ; {_n(t['As_min'], 'mm²')})"]
    elif k == "As_max":
        parts = ["A_s,max", "0,04·b·h", f"0,04 × {_v(t['b'])} × {_v(t['h'])}"]
    elif k == "tau":
        parts = ["τ", "V·10³ / (0,75·b·h)", f"{_v(t['V'])} × 10³ / (0,75 × {_v(t['b'])} × {_v(t['h'])})"]
    elif k == "tau_lim":
        parts = ["τ_adm,I", "0,016·f_ck,cube / 1,05", f"{C.nb(t['k'])} × {_v(t['fck_cube'])} / {C.nb(t['gamma'])}"]
    else:                                                         # pragma: no cover — une formule nouvelle du moteur
        parts = [k, " · ".join(f"{s} = {x}" for s, x in t.items())]
    return {"parts": parts, "res": r, "niveau": ""}


def _taux_fx(v: Dict, s: Dict) -> Dict:
    """La ligne du taux, en pastille de couleur : η = valeur / limite."""
    g = s["grandeurs"]
    if v["cle"] == "hauteur":
        parts = ["η", "h_min / h", f"{_n(g['h_min'], 'cm')} / {C.nb(g['h'])}"]
    elif v["cle"] == "tau":
        parts = ["η", "τ / τ_adm,I", f"{_n(g['tau'], 'N/mm²')} / {_n(g['tau_adm'], 'N/mm²')}"]
    else:
        parts = ["η", "A_s,nécessaire / A_s", f"{_n(v['valeur'], 'mm²')} / {_n(v['limite'], 'mm²')}"]
    return {"parts": parts, "res": f"{C.pc(C.pct(v['taux']))} %", "niveau": _niv(v)}


def _ligne_tableau(v: Dict, s: Dict, gov: bool) -> Dict:
    val, lim = _comparaison(v, s)
    k = v["cle"]
    if k == "hauteur":
        f = "h<sub>min</sub> = h<sub>u,min</sub> + y<sub>G</sub> ≤ h"
        sous = f"{C.nb(s['grandeurs']['b'])} × {C.nb(s['grandeurs']['h'])} cm"
        if v.get("face"):
            cote, dk = v["face"].split("_")
            sous += f" · famille {cote}. {dk.upper()}"
    elif k == "tau":
        f, sous = "τ = V / (0,75·b·h) ≤ τ<sub>adm,I</sub>", f"V = {C.nb(s['V'])} kN · sans étriers"
    else:
        f = "max(A<sub>s,req</sub> ; A<sub>s,min</sub>) ≤ A<sub>s</sub> ≤ A<sub>s,max</sub>"
        sous = _piece(v, s) + (" · principale" if v.get("principale") else "")
    if v.get("motifs"):
        sous += " · " + _motifs(v)
    return {"nom": C.esc(nom_ligne(v)), "sous": sous, "formule": f, "valeur": val, "limite": lim, "taux": C.pct(v["taux"]),
            "niveau": _niv(v), "gov": gov}


def _calculer(d: Dict) -> Dict:
    """Le calcul d'une dalle : le MOTEUR DE SON TYPE (beton/dalle ou beton/predalle, inchangés) sur la copie que
    api_web/dalle_positions lui prépare (sans les clés du site, les positions du site écrites là où elles diffèrent),
    puis le résultat relu avec la notion d'automatique DU SITE (``POS.annoter`` : ce que l'écran et les dessins
    lisent)."""
    return POS.annoter(_MO(d).verifier(POS.pour_moteur(d)), d)


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : sa dalle la plus chargée — une dalle non conforme d'abord (la règle de gouvernante()) ;
    « Données incomplètes » dès qu'une section n'a aucun effort (comme la poutre)."""
    gouv = [gouvernante(r) for r in resultats]
    k = max(range(len(resultats)), key=lambda j: _cle_gouv(gouv[j][1]))
    r, (i_s, v) = resultats[k], gouv[k]
    nivs = [PC.niveau_element(_MO(q), _M(q), x) for q, x in zip(etat["dalles"], resultats)]
    niv = next((n for n in ("r", "n") if n in nivs), C.niveau(v["taux"]))
    out = {"taux": C.pct(v["taux"]) if niv != "n" else None, "niveau": niv, "statut": C.STATUTS[niv][1],
           "gouvernant": f"{etat['dalles'][k].get('nom') or 'Dalle'} · {r['sections'][i_s]['titre']} · {nom_minuscule(v)}"}
    if niv in C.STATUTS_COURTS:
        out["court"] = C.STATUTS_COURTS[niv]   # la pastille du téléphone
    return out


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les dalles de la note sont l'ARBORESCENCE des données (``arbre``), avec leurs rubriques."""
    return ""


def _avertissements_de(d: Dict, r: Dict, i: Optional[int] = None) -> str:
    """Les avertissements selon le type : ceux de la dalle coulée (ci-dessous), ou ceux de la prédalle (api_web/predalle :
    base qui sort de la peau, couches croisées sur la prédalle, bases X et Y trop proches). Des bases SUPÉRIEURES
    confondues d'une prédalle sont dites comme celles de la dalle coulée : le site y pose la direction secondaire sur la
    principale (api_web/dalle_positions), seul un axe SAISI les y met — la phrase de l'aide prédalle (« comme dans l'écran
    Streamlit… Saisir son axe ») ne vaut plus que pour ses faces INFÉRIEURES, posées par la règle de la prédalle
    (relecture du 09/11/2026)."""
    if not est_predalle(d):
        return _avertissements(r, i)

    def sup(a):
        return a["code"] == "nappes_confondues" and a.get("face") == "sup"
    return (AP.avertissements(dict(r, avertissements=[a for a in r["avertissements"] if not sup(a)]), i)
            + _avertissements(dict(r, avertissements=[a for a in r["avertissements"] if sup(a)]), i))


def _avertissements(r: Dict, i: Optional[int] = None) -> str:
    """Les avertissements du moteur des dalles : ceux de la section ``i`` (posés dans son bloc), ou, sans ``i``, ceux de
    la dalle entière ; les nappes X et Y d'une même face calculées trop près l'une de l'autre sont dites en une phrase —
    le site posant la secondaire sur la principale, seul un axe SAISI les y met : la phrase dit comment le rendre."""
    out = ""
    confondues: List[Dict] = []
    for a in r["avertissements"]:
        if a.get("section") != i:
            continue
        if a["code"] == "nappes_confondues":
            confondues.append(a)
        elif a["code"] == "enrobage_nul":
            out += C.message("Enrobage nul : comme l'écran Streamlit, le calcul pose les couches avec un enrobage de 3 cm.")
        elif a["code"] != "d_utile_negatif":      # (d ≤ 0 : la ligne de la face le dit, en rouge — PC.comparaison)
            out += C.message(C.esc(a["message"]))
    if confondues:
        s = r["sections"][i]
        faces = " et ".join("inférieures" if a["face"] == "inf" else "supérieures" for a in confondues)
        axes = " ; ".join(f"{'inf.' if a['face'] == 'inf' else 'sup.'} {C.fr(a['e_x'], 1)} et {C.fr(a['e_y'], 1)} cm" for a in confondues)
        mini = max(a["ecart_mini"] for a in confondues)
        niveau = ("au même niveau" if all(abs(a["e_x"] - a["e_y"]) < 1e-9 for a in confondues)
                  else "à des niveaux trop proches pour se croiser")
        out += C.message(f"{C.esc(s['titre'])} : les nappes X et Y {faces} sont calculées {niveau} ({axes}) — un axe saisi "
                         f"les y pose. Croisées, l'une passe sur l'autre (au moins {C.fr(mini, 1)} cm d'axe à axe) : son d est "
                         "plus petit. Rendre l'axe saisi à l'automatique (↺) pose la direction secondaire sur la principale.")
    return out


# Ce que le calcul d'une dalle COULÉE EN PLACE vérifie et ne vérifie pas (petits chantiers, 10/11/2026 : la dalle CSP
# le dit comme la prédalle — même composant, même place ; le moteur beton/dalle : hauteur, les quatre nappes avec leurs
# minimum et maximum, τ ≤ τ_adm,I sans étriers ; ni flèche ni fissuration). En entier dans chaque fenêtre de calcul ;
# sous le résultat, la seule phrase « Non vérifiées » (NON_VERIFIEES_CSP) : ce qui est vérifié, les lignes du résultat
# le montrent déjà (le propriétaire, 13/11/2026 : « supprimer les messages explicatifs redondants »).
NON_VERIFIEES_CSP = "Non vérifiées : la flèche et la fissuration."
PERIMETRE_CSP = ("Vérifié : la section en flexion (hauteur, quatre nappes : A<sub>s,req</sub>, minimum et maximum) et "
                 "τ ≤ τ<sub>adm,I</sub> (une dalle sans étriers). " + NON_VERIFIEES_CSP)


def texte_perimetre(d: Dict) -> str:
    """Ce que le calcul de la dalle vérifie et ne vérifie PAS, en entier (les fenêtres de calcul) : celui de la prédalle
    (api_web/predalle.PERIMETRE), celui de la dalle coulée en place (PERIMETRE_CSP) — une source pour les deux."""
    return AP.PERIMETRE if est_predalle(d) else PERIMETRE_CSP


def non_verifies(d: Dict) -> str:
    """Ce que le calcul ne vérifie PAS, en une phrase : sous le résultat, et dans l'en-tête de la dalle de la note PDF
    (api_web/note_dalle.py) — la fin de ``texte_perimetre``, la même source."""
    return AP.NON_VERIFIES if est_predalle(d) else NON_VERIFIEES_CSP


def _hypotheses(d: Dict, m: Dict) -> List[str]:
    return [f"Béton <b>{C.esc(m['beton'])}</b> · f<sub>ck</sub> = {C.nb(m['fck'])} · f<sub>ck,cube</sub> = {C.nb(m['fck_cube'])} N/mm²",
            f"f<sub>ctm</sub> = 0,30·f<sub>ck</sub><sup>2/3</sup> = {C.fr(m['fctm'], 2)} N/mm²",
            f"Acier <b>B{int(m['fyk'])}</b> · f<sub>yd</sub> = {int(m['fyk'])} / {C.nb(m['gamma_s'])} = {C.fr(m['fyd'], 1)} N/mm²",
            f"α<sub>b</sub> = {C.nb(m['alpha_b'])} · μ = {C.nb(m['mu'], 4)}",
            f"Bande b × h = {C.nb(d['b'])} × {C.nb(d['h'])} cm · enrobage {C.nb(d['enrobage'])} cm · jeu {C.nb(d['jeu_premier_lit'])} cm"]


def _efforts_txt(s: Dict) -> str:
    m = [f"{SOLLICITATIONS[k]} = {C.nb(s[k])}" for k in EN.MOMENTS if s[k]]
    txt = (" · ".join(m) + " kN·m") if m else "aucun moment"
    return txt + (f" · V = {C.nb(s['V'])} kN" if s["V"] else "")


def resultat(etat: Dict) -> Dict:
    """Le résultat de la dalle ouverte, par le moteur de son type : la tête (le nom, modifiable là — § 1 —, et les
    dimensions, la peau dite pour une prédalle), un bloc par section (les formules mises en mots par le module du type :
    ce module-ci, ou l'aide prédalle), les avertissements, le périmètre d'une prédalle, les vues (dessin_dalle ou
    dessin_predalle), et le résumé de la note. L'avis d'ouverture (CLE_AVIS : une note d'avant, un fichier Streamlit dont
    les chiffres changent avec les nappes posées l'une sur l'autre) est dit en tête du résultat, une fois : il est RETIRÉ
    de ``etat`` (la note rendue ne le porte pas) ; de même l'avis des noms coupés à la relecture (CLE_NOMS). Les avis
    partent aussi dans « avis » (la liste) : js/etabli.js les dit dans le message de la page là où le résultat ne se voit
    pas à l'ouverture (le téléphone s'ouvre sur l'onglet Données)."""
    avis = [a for a in (AVIS.get(etat.pop(CLE_AVIS, None), ""), avis_noms(etat.pop(CLE_NOMS, None) or [])) if a]
    d = _actif(etat)
    r = _calculer(d)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["dalles"])]
    nom = d.get("nom") or "Dalle"
    predalle = est_predalle(d)
    M = _M(d)
    hyps = AP.hypotheses(d, r["materiaux"]) if predalle else _hypotheses(d, r["materiaux"])
    gouv = gouvernante(r)
    perimetre = C.aide(texte_perimetre(d))
    blocs, volets = "", ""
    for i, s in enumerate(r["sections"]):
        h, v = PC.bloc(M, i, s, nom, hyps, gouv, pied_calcul=perimetre, pied=_avertissements_de(d, r, i))
        blocs += h
        volets += v
    dims = _dims(d)
    pied = (f'<div class="pfoot"><div class="mass"><span>Méthode du bureau · moteur {"prédalle" if predalle else "dalle"}</span>'
            f'<b>v{C.esc(_MO(d).REFERENCE.split(" v")[-1])}</b></div><div class="legend">{C.legende()}</div></div>')
    # l'état de la NOTE (toutes ses dalles) dans l'en-tête du panneau : utile dès deux dalles ; seule, chaque section dit le sien
    # (l'état de CHAQUE dalle colore aussi son nom, dans la ligne et dans la liste des dalles : ``arbre``)
    nivs = [PC.niveau_element(_MO(q), _M(q), x) for q, x in zip(etat["dalles"], resultats)]
    statut = C.statut(next((n for n in ("r", "n") if n in nivs), max(nivs, key=lambda n: {"p": 0, "f": 1}.get(n, 0)))) if len(resultats) > 1 else ""
    element = r["predalle"] if predalle else r["dalle"]
    return {"resultat": f'<div class="pbody">{E.tete_resultats(C.esc(nom), dims)}{"".join(C.message(a) for a in avis)}'
                        f'<div class="et-blocs">{blocs}</div>{_avertissements_de(d, r)}{C.aide(non_verifies(d))}</div>{pied}',
            **({"avis": avis} if avis else {}), "volets": volets, "vues": PC.vues(r["sections"], element, DP if predalle else DD, "pd" if predalle else "dl", dims, _legendes(d)),
            "phrase": "", "statut_html": statut,
            "resume": resume_note(etat, resultats), "donnees": arbre(etat, r, nivs), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ la note PDF
def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None) -> Optional[Dict]:
    """« note_pdf » : la note de calcul PDF de TOUTE la note (chaque dalle, chaque section, les deux types), faite par le
    générateur validé de l'écran Streamlit sur les axes du site (api_web/note_dalle.py) ; ``infos`` : le nom, le
    chantier, l'indice et la date de la note (la page de garde). Les avis d'ouverture (CLE_AVIS, CLE_NOMS) n'y entrent pas."""
    if quoi != "note_pdf":
        return None
    from . import note_dalle as ND
    etat = {k: v for k, v in etat.items() if k not in (CLE_AVIS, CLE_NOMS)}
    return ND.fichier(etat, infos)


# ------------------------------------------------------------------ le sélecteur de la nappe de base
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les nappes de base possibles d'une face (treillis de la base, barres Ø8 à Ø25 de 100 à 250 mm), triées par section
    au mètre, calculées par le moteur du type aux niveaux du site (api_web/dalle_positions.candidats_base) : la retenue
    (✓), la plus petite qui tient (★) ; une prédalle dit l'axe de chaque candidat, recalculé par sa règle de pose."""
    face, i = role.rsplit("-", 1)
    i = int(i)
    d = _actif(etat)
    predalle = est_predalle(d)
    fams = tuple(f for f in (familles or MO.FAMILLES) if f in MO.FAMILLES) or MO.FAMILLES
    esp = etat.get("espacements", list(ESPACEMENTS))
    cands = POS.candidats_base(d, i, face, fams, espacements=esp)
    i_o = next((k for k, c in enumerate(cands) if c["optimal"]), 0)
    i_c = next((k for k, c in enumerate(cands) if c["retenue"]), i_o)
    base = d["sections"][i]["faces"][face]["couches"][0]
    pose_k = MO_P.pose(face, 0, d["dir_principale"])
    if predalle and pose_k == "sur_predalle" and POS.treillis(d, d["sections"][i], "inf"):
        pose_k = "predalle"                # DAL5b : les fils secondaires d'un treillis unique, soudés, sont dans la peau
    pose = AP.POSES_TXT[pose_k] if predalle else ""

    def prop(c):
        axe = ""
        if predalle and "e" in c:
            axe = f"axe {C.fr(c['e'], 1)} cm" + ("" if base["dist"] is not None else f" ({pose})")
        return AP.description_candidat(c, axe)
    rows = [{"nom": c["nom"], "col": C.fr(c["As_pm"], 0), "taux": C.pct(min(c["taux"], 9.99)),
             "niveau": "r" if c["etat"] == MO.NOK else C.niveau(c["taux"]), "cur": c["retenue"], "opt": c["optimal"],
             "prop": prop(c)} for c in cands]
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{f}">{LIB_FAMILLES[f]}</button>'
                      for f in MO.FAMILLES)
    s = d["sections"][i]
    renforts = len(s["faces"][face]["couches"]) - 1
    note = f" · {renforts} renfort(s) gardé(s)" if renforts else ""
    if base["type"] == "n_barres":
        note += f" · la base actuelle ({base['n']} Ø{base['d']}) n'est pas dans la liste"
    elif base["type"] == "barres" and base["esp"] not in esp:
        note += f" · la base actuelle (Ø{base['d']}/{base['esp']}) n'est pas à un espacement admis"
    if list(esp) != list(ESPACEMENTS):
        note += f" · espacements admis : {' · '.join(C.nb(e / 10) for e in esp)} cm"
    dk = face.split("_")[1]
    princ = " (principale)" if dk == d["dir_principale"].lower() else ""
    ou = f", {pose}" if predalle else ""
    return C.selecteur(role, f"Nappe {FACES_TXT[face.split('_')[0]]} {dk.upper()} · {C.esc(EN.libelle_section(s['nom']))}",
                       f"couche de base{princ}{ou} · triée par section au mètre · un clic la retient{note}", filtres, rows,
                       max(0, min(i_o, i_c) - 4), "mm²/m", etoile="la plus petite section qui tient")


_MOI = sys.modules[__name__]   # ce module, que api_web/plaque_compact.py lit (ses formules mises en mots)
