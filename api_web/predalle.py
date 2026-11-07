# -*- coding: utf-8 -*-
"""Le module « Prédalles » du site : l'état d'une note (une ou plusieurs prédalles, chacune avec ses sections, et pour
chaque section quatre nappes — inférieure et supérieure, directions X et Y — faites de couches de treillis, de barres
Ø/e ou de « n barres »), les champs de saisie, le résultat section par section, la grande fenêtre de calcul, le
sélecteur de la nappe de base et les dessins — rendus en fragments HTML.

Une prédalle est une dalle dont la partie basse (la PEAU, h_pre) est préfabriquée : la base inférieure de la direction
principale y est posée en fond de moule, les autres couches inférieures SUR la peau, le reste est coulé en place. Le
calcul est celui du moteur ``beton/predalle`` (extrait de l'écran Streamlit modules/predalle.py v1.1, dont il reproduit
les résultats au bit près — tests/test_predalle_moteur.py). Ce fichier MET EN FORME : il ne calcule rien.

Ce que le site fait de plus que l'écran Streamlit, sans changer un chiffre :
- une saisie que l'écran ramènerait en silence à une autre valeur (peau hors de ses bornes, moment négatif lu 0,
  espacement hors bornes…) est REFUSÉE avec un message, et la prédalle reste intacte ;
- le sélecteur de la nappe de base (treillis de la base, barres courantes) trié par section au mètre, l'axe recalculé
  par la règle de la prédalle, la retenue (✓) et la plus petite qui tient (★) marquées ;
- les avertissements du moteur (nappes croisées calculées trop près, base qui sort de la peau, enrobage nul lu
  3 cm, d utile ≤ 0) ;
- la reprise d'un fichier de l'écran Streamlit (« predalle_ba.json ») avec ce que l'écran y remplaçait en silence ;
- un verdict qui, dans une prédalle non conforme, nomme une vérification NON conforme (gouvernante()) : acier excessif
  et d utile ≤ 0 le sont sous 100 % ;
- des bornes contre l'absurde, que l'écran n'a pas (SOLLICITATION_MAX, EN.H_MAX pour les longueurs saisies, NOM_MAX).
"""
import copy
import json
from typing import Dict, List, Optional, Sequence

from beton import treillis as TR
from beton.predalle import entree as EN
from beton.predalle import fichier as FI
from beton.predalle import moteur as MO

from . import commun as C
from . import dessin_predalle as DP

MODULE = "predalle"
PAGE = "predalle"

# des requêtes de plus pour le parcours (api_web/parcours.py) : les champs que le parcours automatique n'essaie pas
_FICHIER = {"version": "predalle-1.0", "predalles": [{"id": 1, "nom": "Prédalle 1", "sections": [{"id": 1, "nom": "A"}]}],
            "values": {"pre1_sec1_Mx_inf": 28.0, "pre1_sec1_My_inf_raw": "-12,0", "pre1_sec1_V": 55.0, "pre1_h": 8,
                       "pre1_h_pre": 12.0, "pre1_sec1_ncouches_inf_y": 2, "pre1_sec1_arm_type_inf_y_c2": "Barres", "gamma_s": 1.5}}
PARCOURS = [
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_x", "couches", "0", "dist"], "valeur": "8"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_y", "yG"], "valeur": "4,2"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "sup_x", "couches", "0", "dist"], "valeur": ""},
    {"action": "modifier", "chemin": ["sections", "0", "Mx_inf"], "valeur": "-20"},
    {"action": "modifier", "chemin": ["h_pre"], "valeur": "21"},
    {"action": "modifier", "chemin": ["h"], "valeur": "6"},
    {"action": "modifier", "chemin": ["h_pre"], "valeur": "3"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_x", "couches", "0", "esp"], "valeur": "600"},
    {"action": "modifier", "chemin": ["fichier_streamlit"], "valeur": {"nom": "predalle_ba.json", "texte": json.dumps(_FICHIER)}},
    {"action": "modifier", "chemin": ["fichier_streamlit"], "valeur": {"nom": "abime.json", "texte": "{pas du json"}},
    {"action": "selecteur", "role": "sup_x-0", "familles": ["treillis"]},
    {"action": "choisir", "role": "inf_y-0", "nom": "Ø12/150"},
]

TYPES = (("treillis", "Treillis"), ("barres", "Barres"), ("n_barres", "n barres"))   # les libellés de l'écran
FACES_TXT = {"inf": "inférieure", "sup": "supérieure"}
POSES_TXT = {"predalle": "dans la prédalle", "sur_predalle": "sur la prédalle", "superieure": "depuis le dessus"}
POSES_AIDE = {"predalle": "Axe automatique : la base principale est posée en fond de moule, DANS la prédalle — enrobage + ½ Ø "
                          "arrondi au 0,5 cm (sans jeu premier lit).",
              "sur_predalle": "Axe automatique : couche posée SUR la prédalle, au chantier — épaisseur de la prédalle + ½ Ø "
                              "arrondi au 0,5 cm.",
              "superieure": "Axe automatique, depuis le dessus : enrobage + ½ Ø arrondi au 0,5 cm + jeu premier lit."}
MOTIFS = {"hauteur_insuffisante": "hauteur insuffisante", "d_utile_negatif": "d utile ≤ 0",
          "acier_insuffisant": "acier insuffisant", "acier_excessif": "acier excessif (A<sub>s</sub> &gt; A<sub>s,max</sub>)",
          "cisaillement_excessif": "τ &gt; τ<sub>adm,I</sub> (pas d'étriers dans une dalle)"}
LIB_FAMILLES = {"treillis": "Treillis", "barres": "Barres"}


class _Refus(ValueError):
    """Une saisie refusée : son message est montré tel quel."""


# ------------------------------------------------------------------ l'état de la note : des prédalles
def _site(p: Dict, verrou: bool = False) -> Dict:
    """Une prédalle du site : la prédalle du moteur, normalisée, et son verrou (lecture seule, comme le 🔒 de l'écran)."""
    out = EN.normaliser(p)
    out["verrou"] = bool(verrou)
    return out


def _moteur(p: Dict) -> Dict:
    """La prédalle telle que le moteur la lit (sans le verrou, propre au site)."""
    return {k: v for k, v in p.items() if k != "verrou"}


def nouvel_etat() -> Dict:
    return {"version": 1, "predalles": [_site(EN.nouvelle_predalle())], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran relu par le moteur ; une prédalle illisible (fichier abîmé) lève une erreur claire. Une
    note sans prédalle est une note neuve ; « predalles » qui n'est pas une liste est une note abîmée (refusée, jamais
    remplacée en silence par une note neuve)."""
    ps = etat.get("predalles") if isinstance(etat, dict) else None
    if ps is None or ps == []:
        return nouvel_etat()
    if not isinstance(ps, list):
        raise ValueError(f"note illisible : « predalles » doit être une liste (reçu {type(ps).__name__})")
    out = {"version": 1, "predalles": []}
    for i, p in enumerate(ps):
        if not isinstance(p, dict):
            raise ValueError(f"prédalle n° {i + 1} illisible")
        v = p.get("verrou", False)
        if not isinstance(v, bool):
            raise ValueError(f"prédalle n° {i + 1} : verrou vrai ou faux attendu")
        try:
            q = _site(_moteur(p), v)
            _bornes_site(q)
            for s in q["sections"]:           # un nom vide (possible dans un .stab, pas à l'écran) : la première lettre libre
                if not s["nom"].strip():
                    s["nom"] = EN.nom_section_suivant(q)
            out["predalles"].append(q)
        except ValueError as e:
            raise ValueError(f"prédalle « {p.get('nom', i + 1)} » : {e}") from None
    a = etat.get("actif", 0)
    a = a if isinstance(a, int) and not isinstance(a, bool) else 0
    out["actif"] = min(max(0, a), len(out["predalles"]) - 1)
    return out


def _actif(etat: Dict) -> Dict:
    return etat["predalles"][etat["actif"]]


def _nom_libre(etat: Dict) -> str:
    pris = {p["nom"] for p in etat["predalles"]}
    k = len(etat["predalles"]) + 1
    while f"Prédalle {k}" in pris:
        k += 1
    return f"Prédalle {k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les prédalles de la note : en ouvrir une, en ajouter, copier l'ouverte, en supprimer une (jamais la dernière).
    Une prédalle ajoutée prend le γs et le jeu de la note (paramètres communs, comme dans l'écran Streamlit)."""
    ps = etat["predalles"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(ps) - 1)
    elif action == "ajouter":
        p = EN.nouvelle_predalle()
        p["nom"] = _nom_libre(etat)
        p["gamma_s"], p["jeu_premier_lit"] = ps[0]["gamma_s"], ps[0]["jeu_premier_lit"]
        ps.append(_site(p))
        etat["actif"] = len(ps) - 1
    elif action == "dupliquer":
        p = copy.deepcopy(ps[etat["actif"]])
        p["nom"] = f"{p.get('nom') or 'Prédalle'} (copie)"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "supprimer" and len(ps) > 1:
        i = min(max(0, int(index or 0)), len(ps) - 1)
        del ps[i]
        if etat["actif"] > i or etat["actif"] >= len(ps):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
# Bornes PROPRES AU SITE (l'écran Streamlit n'en a pas) : elles ne refusent que l'absurde, que l'écran calculait en
# chiffres démesurés (taux de 10³⁰⁰ %, verdict illisible) :
# - une longueur (jeu premier lit, axe d'une couche, centre de gravité) au plus égale à l'épaisseur maximale d'une
#   prédalle, EN.H_MAX ;
# - une sollicitation au plus égale à SOLLICITATION_MAX : la plus forte bande du domaine de l'écran (5 m × 100 cm,
#   C50/60) porte moins de 20 000 kN·m et 4 000 kN ;
# - un nom de prédalle ou de section d'au plus NOM_MAX caractères (il titre la note, les coupes, la fenêtre de calcul).
SOLLICITATION_MAX = 1e5
NOM_MAX = 60
NOMBRES = {  # clé → (libellé, unité, mini, maxi, mini exclu, entier, maxi propre au site)
    "b": ("Largeur de bande", "cm", EN.B_MIN, EN.B_MAX, False, True, False),
    "h": ("Épaisseur totale", "cm", EN.H_MIN, EN.H_MAX, False, True, False),
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


def _saisi(x: float) -> str:
    """Un nombre SAISI, rappelé dans un message de refus : exact s'il est raisonnable, en puissance de dix sinon (un
    1e300 tapé ne s'écrit pas sur trois cents chiffres)."""
    if abs(x) < 1e7:
        ent, _, dec = C.nb(x, 6).partition(",")
        signe, ent = ("-", ent[1:]) if ent.startswith("-") else ("", ent)
        return signe + f"{int(ent):,}".replace(",", C.NBSP) + ("," + dec if dec else "")
    m, e = f"{x:.2e}".split("e")
    exposant = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m.replace('.', ',')} × 10{exposant}"


def _nombre(cle: str, valeur):
    nom, u, mini, maxi, exclu, entier, site = NOMBRES[cle]
    x = C.lire_nombre(valeur)
    if x is None:
        raise _Refus(f"{nom} : nombre illisible « {valeur} »")
    unite = f" {u}" if u else ""
    if (x <= mini) if exclu else (x < mini):
        borne = (f"entre {C.nb(mini)} et {C.nb(maxi)}{unite}" if maxi is not None and not site
                 else (f"plus de {C.nb(mini)}{unite}" if exclu else f"au moins {C.nb(mini)}{unite}"))
        raise _Refus(f"{nom} : {borne} (bornes de l'écran Streamlit), pas {_saisi(x)}{unite}")
    if maxi is not None and x > maxi:
        if site:
            raise _Refus(f"{nom} : au plus {C.nb(maxi)}{unite}, l'épaisseur maximale d'une prédalle (au-delà, la saisie "
                         f"n'a pas de sens), pas {_saisi(x)}{unite}")
        raise _Refus(f"{nom} : entre {C.nb(mini)} et {C.nb(maxi)}{unite} (bornes de l'écran Streamlit), pas {_saisi(x)}{unite}")
    if entier:
        if x != int(x):
            raise _Refus(f"{nom} : un nombre entier{unite}, pas {_saisi(x)}")
        return int(x)
    return x


def _nom(valeur, quoi: str, vide_admis: bool) -> str:
    """Un nom saisi (prédalle, section) : NOM_MAX caractères au plus ; vide refusé pour une section."""
    t = str(valeur)
    if not vide_admis and not t.strip():
        raise _Refus(f"Nom {quoi} vide")
    if len(t) > NOM_MAX:
        raise _Refus(f"Nom {quoi} : {NOM_MAX} caractères au plus ({len(t)} saisis) — il titre la note, les coupes et la "
                     "fenêtre de calcul")
    return t


def _bornes_site(p: Dict) -> None:
    """Les bornes propres au site (voir SOLLICITATION_MAX) sur une prédalle relue (fichier .stab, fichier Streamlit) :
    ValueError claire."""
    if p["jeu_premier_lit"] > EN.H_MAX:
        raise ValueError(f"jeu premier lit de {_saisi(p['jeu_premier_lit'])} cm : au plus {EN.H_MAX} cm")
    if len(p["nom"]) > NOM_MAX:
        raise ValueError(f"nom de {len(p['nom'])} caractères : {NOM_MAX} au plus")
    for s in p["sections"]:
        if len(s["nom"]) > NOM_MAX:
            raise ValueError(f"section : nom de {len(s['nom'])} caractères, {NOM_MAX} au plus")
        for k in EN.MOMENTS + ("V",):
            if s[k] > SOLLICITATION_MAX:
                raise ValueError(f"section « {s['nom']} » : {k} au-delà de {C.fr(SOLLICITATION_MAX, 0)} (valeur absurde)")
        for f in EN.FACES:
            face = s["faces"][f]
            if any(x is not None and x > EN.H_MAX for x in [face["yG"]] + [c["dist"] for c in face["couches"]]):
                raise ValueError(f"section « {s['nom']} », nappe {f} : axe ou centre de gravité au-delà de {EN.H_MAX} cm")


def _peau(p: Dict, valeur) -> float:
    """L'épaisseur de la prédalle : de 3 cm à min(15 ; h − 1) cm. L'écran Streamlit ramenait une valeur hors de ces
    bornes sans le dire ; le site la refuse."""
    x = C.lire_nombre(valeur)
    if x is None:
        raise _Refus(f"Épaisseur de la prédalle : nombre illisible « {valeur} »")
    maxi = EN.h_pre_max(p["h"])
    if not EN.H_PRE_MIN <= x <= maxi:
        raise _Refus(f"Épaisseur de la prédalle : entre {C.nb(EN.H_PRE_MIN)} et {C.nb(maxi)} cm pour une épaisseur totale de "
                     f"{C.nb(p['h'])} cm (au moins {C.nb(EN.COULE_MIN)} cm coulé en place, {C.nb(EN.H_PRE_MAX)} cm au plus), "
                     f"pas {_saisi(x)} cm")
    return x


def _axe_auto(p: Dict, c: Dict, j: int, f: str) -> float:
    return MO.distance_auto(c, j, f, p["dir_principale"], p["enrobage"], p["jeu_premier_lit"], p["h_pre"])


def _yG_calcule(p: Dict, f: str, face: Dict) -> float:
    axes = [c["dist"] if c["dist"] is not None else _axe_auto(p, c, j, f) for j, c in enumerate(face["couches"])]
    return FI.yG_calcule(face["couches"], axes, p["b"])


def _auto_ou_impose(cle: str, valeur, actuel, auto: float):
    """Un axe ou un centre de gravité : vide → automatique ; la valeur automatique affichée, réécrite → reste
    automatique (comme le champ de l'écran Streamlit) ; un autre nombre > 0 → imposé."""
    if valeur is None or (isinstance(valeur, str) and valeur.strip().lower() in ("", "auto")):
        return None
    x = _nombre(cle, valeur)
    if actuel is None and abs(x - auto) <= 0.5e-3 + 1e-12:
        return None
    return x


def _ecrire(p: Dict, chemin: List, valeur) -> None:
    """Écrit une saisie dans la prédalle ``p`` (une copie) ; _Refus si la valeur sort du domaine de l'écran."""
    k = chemin[0]
    if len(chemin) == 1:
        if k == "nom":
            p["nom"] = _nom(valeur, "de la prédalle", True)
        elif k == "beton":
            if valeur not in MO.classes_beton():
                raise _Refus(f"Béton : classe « {valeur} » inconnue")
            p["beton"] = valeur
        elif k == "fyk":
            x = C.lire_nombre(valeur)
            if x not in EN.NUANCES:
                raise _Refus(f"Acier : B400 ou B500, pas « {valeur} »")
            p["fyk"] = int(x)
        elif k == "dir_principale":
            if str(valeur).upper() not in ("X", "Y"):
                raise _Refus(f"Direction principale : X ou Y, pas « {valeur} »")
            p["dir_principale"] = str(valeur).upper()
        elif k == "h":
            h = _nombre("h", valeur)
            if p["h_pre"] > EN.h_pre_max(h):
                raise _Refus(f"Épaisseur totale : la prédalle de {C.nb(p['h_pre'])} cm demande au moins "
                             f"{C.nb(p['h_pre'] + EN.COULE_MIN)} cm (un centimètre coulé en place) ; réduire d'abord la prédalle. "
                             "L'écran Streamlit ramenait la prédalle sans le dire.")
            p["h"] = h
        elif k == "h_pre":
            p["h_pre"] = _peau(p, valeur)
        elif k in ("b", "enrobage"):
            p[k] = _nombre(k, valeur)
        else:
            raise _Refus(f"champ inconnu : {k}")
        return
    if k != "sections":
        raise _Refus(f"champ inconnu : {'/'.join(map(str, chemin))}")
    secs = p["sections"]
    if chemin[1] == "+":
        secs.append(EN.nouvelle_section(EN.nom_section_suivant(p)))
        return
    i = chemin[1]
    s = secs[i]
    if chemin[2] == "-":
        if len(secs) <= 1:
            raise _Refus("Il faut au moins une section")
        del secs[i]
        return
    if chemin[2] == "copie":
        c = copy.deepcopy(s)
        c["nom"] = EN.nom_section_suivant(p)
        secs.append(c)
        return
    if chemin[2] == "nom":
        s["nom"] = _nom(valeur, "de section", False)
        return
    if chemin[2] in SOLLICITATIONS:
        x = C.lire_nombre(valeur)
        if x is None:
            raise _Refus(f"Nombre illisible : « {valeur} »")
        if x < 0:
            raise _Refus("Les moments et l'effort tranchant se saisissent positifs : la face tendue se choisit par la "
                         "colonne (inf. tend le bas, sup. tend le haut). L'écran Streamlit ramenait un négatif à 0 sans le dire.")
        if x > SOLLICITATION_MAX:
            u = "kN" if chemin[2] == "V" else "kN·m"
            raise _Refus(f"Au plus {C.fr(SOLLICITATION_MAX, 0)} {u}, pas {_saisi(x)} {u} : la plus forte bande de prédalle "
                         "(5 m de large, 100 cm d'épaisseur) porte moins de 20 000 kN·m et 4 000 kN.")
        s[chemin[2]] = x
        return
    f = chemin[3]
    face = s["faces"][f]
    if chemin[4] == "yG":
        face["yG"] = _auto_ou_impose("yG", valeur, face["yG"], _yG_calcule(p, f, face))
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
        c["dist"] = _auto_ou_impose("dist", valeur, c["dist"], _axe_auto(p, c, j, f))
    else:
        raise _Refus(f"champ inconnu : {champ}")


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la prédalle ouverte. ``chemin`` : les clés (« sections », 0, « faces », « inf_y »,
    « couches », 0, « type ») ; les boutons d'ajout, de copie et de retrait passent par le même chemin, terminé par
    « + », « copie » ou « - ». Une valeur hors du domaine de l'écran est refusée avec un message, et la prédalle reste
    intacte."""
    chemin = [int(c) if isinstance(c, str) and c.isdigit() else c for c in chemin]
    if chemin == ["fichier_streamlit"]:
        return _ouvrir_streamlit(etat, valeur)
    a = etat["actif"]
    p = etat["predalles"][a]
    if chemin == ["verrou"]:
        lu = {True: True, False: False, "true": True, "false": False, "1": True, "0": False}.get(
            valeur.strip().lower() if isinstance(valeur, str) else valeur)
        if lu is None:
            return f"Verrou : coché ou décoché, pas « {valeur} »"
        p["verrou"] = lu
        return None
    if p["verrou"]:
        return "Prédalle verrouillée : décocher « Verrouiller la prédalle » pour la modifier"
    if chemin and chemin[0] in NOTE_ENTIERE and len(chemin) == 1:
        # γs et le jeu premier lit sont des paramètres de la NOTE (l'écran Streamlit les partage entre ses prédalles)
        try:
            x = _nombre(chemin[0], valeur)
        except _Refus as e:
            return str(e)
        for k, q in enumerate(etat["predalles"]):
            etat["predalles"][k] = _site(dict(_moteur(q), **{chemin[0]: x}), q["verrou"])
        return None
    travail = copy.deepcopy(p)
    try:
        _ecrire(travail, chemin, valeur)
        etat["predalles"][a] = _site(_moteur(travail), p["verrou"])
    except _Refus as e:
        return str(e)
    except ValueError as e:
        return f"Saisie refusée : {e}"
    except (KeyError, IndexError, TypeError, AttributeError):
        return f"Saisie refusée : champ introuvable « {'/'.join(map(str, chemin))} »"
    return None


def _ouvrir_streamlit(etat: Dict, valeur) -> Optional[str]:
    """Un fichier de l'écran Streamlit (« 💾 Enregistrer » : predalle_ba.json) remplace les prédalles de la note ; ce
    que l'écran remplaçait en silence (peau ramenée, moment négatif lu 0, type inconnu…) est dit."""
    if not isinstance(valeur, dict) or not isinstance(valeur.get("texte"), str):
        return "Fichier illisible"
    nom = str(valeur.get("nom") or "fichier")
    try:
        lu = FI.lire(json.loads(valeur["texte"]))
    except json.JSONDecodeError:
        return f"« {nom} » n'est pas un fichier JSON lisible"
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        return f"« {nom} » refusé : {e}"
    try:
        ps = [_site(p) for p in lu["predalles"]]
        for p in ps:
            _bornes_site(p)
    except ValueError as e:
        return f"« {nom} » refusé : {e}"
    etat["predalles"] = ps
    etat["actif"] = 0
    if lu["remarques"]:
        r = lu["remarques"]
        return (f"« {nom} » repris ({len(lu['predalles'])} prédalle(s)). L'écran Streamlit y remplaçait en silence : "
                + " ; ".join(r[:4]) + (f" ; et {len(r) - 4} autre(s)" if len(r) > 4 else ""))
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur de la nappe de base : « inf_y-0 » (nappe inférieure Y de la section 0) reçoit « Treillis 8/8/150/150 »
    ou « Ø12/150 » ; ses renforts, son axe imposé et son centre de gravité imposé sont gardés."""
    face, i = role.rsplit("-", 1)
    p = _actif(etat)
    if p["verrou"]:
        raise ValueError("prédalle verrouillée")
    cand = MO.lire_candidat(nom)
    travail = copy.deepcopy(p)
    travail["sections"][int(i)]["faces"][face]["couches"][0].update(cand)
    etat["predalles"][etat["actif"]] = _site(_moteur(travail), False)


# ------------------------------------------------------------------ le cadre Données
def _attr(chemin) -> str:
    return C._attr_chemin(chemin)


def _bouton(chemin, texte: str, icone: str = "plus", titre: str = "") -> str:
    t = f' title="{C.esc(titre)}"' if titre else ""
    return f'<button type="button" class="addrow" data-chemin="{_attr(chemin)}" data-valeur="1"{t}>{C.icon(icone)}{texte}</button>'


def _retirer(chemin, titre: str) -> str:
    return (f'<button type="button" class="moins" data-chemin="{_attr(chemin)}" data-valeur="1" title="{C.esc(titre)}">'
            f'{C.icon("x")}</button>')


def _titre_face(f: str, principale: str) -> str:
    cote, dk = f.split("_")
    return f"{FACES_TXT[cote].capitalize()} {dk.upper()}" + (" · principale" if dk == principale else "")


REGLES_TXT = {"predalle": "c + ½ Ø", "sur_predalle": "h<sub>p</sub> + ½ Ø", "superieure": "c + ½ Ø + j"}
# Le pictogramme de la POSE d'une couche : une coupe en miniature (la peau hachurée en bas, ou le dessus de la dalle),
# et la barre à sa place — dans la peau, posée sur la peau, ou sous le dessus. Dessiné à la main : aucun identifiant.
_PEAU = '<rect class="peau" x="0.5" y="{y}" width="17" height="{h}" rx="1"/><path class="hach" d="{d}"/>'
PICTOS_POSE = {
    "predalle": (_PEAU.format(y=4.5, h=7, d="M2 11.5 9 4.5M7 11.5 14 4.5M12 11.5 17 6.5") + '<circle class="barre" cx="9" cy="8" r="2.3"/>'),
    "sur_predalle": (_PEAU.format(y=7.5, h=4, d="M2 11.5 6 7.5M7 11.5 11 7.5M12 11.5 16 7.5") + '<circle class="barre" cx="9" cy="4.9" r="2.3"/>'),
    "superieure": '<path class="dessus" d="M0.5 1.2H17.5"/><circle class="barre" cx="9" cy="4.6" r="2.3"/>',
}


def _picto_pose(pose: str) -> str:
    return f'<svg class="pd-m pd-{pose}" viewBox="0 0 18 12" aria-hidden="true">{PICTOS_POSE[pose]}</svg>'


def _couche(i: int, f: str, j: int, c: Dict, geo_c: Dict) -> str:
    """Une couche sur trois rangées : le type et la nappe ; le détail (fils, espacement ou nombre) et l'axe ; puis OÙ
    la règle de la prédalle la pose (dans la prédalle, sur la prédalle, depuis le dessus) — ou l'axe saisi."""
    base = ["sections", i, "faces", f, "couches", j]
    numero = f'<span class="dl-n" title="{"Couche de base" if j == 0 else f"Renfort {j}"}">{j + 1 if j == 0 else f"R{j}"}</span>'
    if c["type"] == "treillis":
        opts = TR.liste_choix()
        if c["treillis"] not in opts:
            opts = opts + [c["treillis"]]          # un ancien fichier : la désignation reste utilisable
        nappe = C.choix(base + ["treillis"], c["treillis"], [(x, x) for x in opts])
        t = geo_c["treillis"]
        detail = (f'<span class="dl-info" title="Fils porteurs Ø{t[0]} tous les {t[2]} mm ; fils de répartition Ø{t[1]} tous '
                  f'les {t[3]} mm (non comptés, comme dans l\'écran Streamlit)">Ø{t[0]} / {t[2]} mm</span>')
    else:
        nappe = C.choix(base + ["d"], c["d"], [(x, f"Ø{x}") for x in EN.DIAMETRES])
        detail = (C.num(base + ["esp"], c["esp"], "mm", entier=True, mini=EN.ESP_MIN, maxi=EN.ESP_MAX) if c["type"] == "barres"
                  else C.num(base + ["n"], c["n"], "barres", entier=True, mini=EN.N_MIN, maxi=EN.N_MAX))
    retirer = _retirer(base + ["-"], "Retirer ce renfort") if j > 0 else "<span></span>"
    pose = geo_c["pose"]
    marque = _picto_pose(pose)
    if c["dist"] is None:
        raz = "<span></span>"
        ligne = f'{marque}<span title="{C.esc(POSES_AIDE[pose])}">axe auto · {POSES_TXT[pose]} ({REGLES_TXT[pose]})</span>'
    else:
        raz = (f'<button type="button" class="dl-raz pd-raz" data-chemin="{_attr(base + ["dist"])}" data-valeur="" '
               f'title="Revenir à l\'axe automatique ({C.nb(geo_c["e_auto"])} cm, {POSES_TXT[pose]})">↺</button>')
        ligne = f'{marque}<span>axe saisi · l\'automatique serait {C.nb(geo_c["e_auto"])} cm, {POSES_TXT[pose]}</span>'
    axe = C.num(base + ["dist"], geo_c["e"], "cm")
    return (f'<div class="pd-couche">{numero}{C.choix(base + ["type"], c["type"], TYPES)}{nappe}{retirer}'
            f'<span></span>{detail}{axe}{raz}<span></span><div class="pd-pose">{ligne}</div></div>')


def _face(i: int, f: str, face: Dict, geo: Dict, principale: str) -> str:
    out = (f'<div class="grp dl-face">{_titre_face(f, principale)}'
           f'<span>{C.fr(geo["As_pm"], 0)} mm²/m</span></div>')
    out += "".join(_couche(i, f, j, c, geo["couches"][j]) for j, c in enumerate(face["couches"]))
    if len(face["couches"]) < EN.MAX_COUCHES:
        pose = "sur la prédalle" if f.startswith("inf") else "dans le plan de la base"
        out += _bouton(["sections", i, "faces", f, "couches", "+"], "Ajouter un renfort",
                       titre=f"Treillis, barres Ø/e ou n barres ; 3 Ø12 par défaut, posées {pose}")
    yG_champ = C.num(["sections", i, "faces", f, "yG"], geo["yG"], "cm")
    if face["yG"] is None:
        return out + C.ligne("centre de gravité", yG_champ, "y<sub>G</sub>",
                             "Calculé : Σ As·e / Σ As (du parement au centre de gravité des couches) ; une valeur saisie le remplace.")
    return (out + C.ligne("centre de gravité", yG_champ, "y<sub>G</sub>", "Valeur saisie : elle remplace Σ As·e / Σ As.")
            + C.aide(f'y<sub>G</sub> imposé (calculé : {C.fr(geo["yG_calcule"], 2)} cm) · '
                     f'<button type="button" class="dl-raz" data-chemin="{_attr(["sections", i, "faces", f, "yG"])}" data-valeur="">'
                     f'revenir au calcul</button>'))


def _resume_face(geo: Dict) -> str:
    return geo["couches"][0]["valeur"] + (f" +{len(geo['couches']) - 1}" if len(geo["couches"]) > 1 else "")


def _lecture(label: str, valeur: str, sym: str = "") -> str:
    return C.ligne(label, f'<span class="dl-ro">{valeur}</span>', sym)


def _titre_predalle(p: Dict) -> str:
    return p["nom"] if p["nom"].lower().startswith(("prédalle", "predalle")) else f"Prédalle {p['nom']}"


def _formulaire_verrouille(p: Dict, r: Dict) -> str:
    """Une prédalle verrouillée : ses données en lecture seule, et la case pour la déverrouiller."""
    s1 = (C.case(["verrou"], True, "Verrouiller la prédalle (lecture seule)")
          + C.aide("La prédalle est verrouillée : ses données se lisent, elles ne se modifient pas.")
          + _lecture("béton, acier", f"{C.esc(p['beton'])} · B{p['fyk']}")
          + _lecture("largeur de bande", f"{C.nb(p['b'])} cm", "b") + _lecture("épaisseur totale", f"{C.nb(p['h'])} cm", "h")
          + _lecture("prédalle · coulé en place", f"{C.nb(p['h_pre'])} + {C.nb(p['h'] - p['h_pre'])} cm", "h<sub>p</sub>")
          + _lecture("enrobage", f"{C.nb(p['enrobage'])} cm", "c") + _lecture("direction principale", p["dir_principale"]))
    s2 = ""
    for s, rs in zip(p["sections"], r["sections"]):
        mom = " · ".join(f"{SOLLICITATIONS[k]} {C.nb(s[k])}" for k in EN.MOMENTS)
        s2 += f'<div class="grp">{C.esc(rs["titre"])}</div>' + _lecture("moments (kN·m)", mom) + _lecture("effort tranchant", f"{C.nb(s['V'])} kN", "V")
        for f in MO.ordre_faces(rs["principale"]):
            s2 += _lecture(_titre_face(f, rs["principale"]).lower(), C.esc(rs["faces"][f]["libelle"]))
    return (C.section(1, f"{C.esc(_titre_predalle(p))} {C.icon('lock')}", "verrouillée", s1, True, "pd")
            + C.section(2, "Sections et armatures", f"{len(p['sections'])} section(s)", s2, True, "sl"))


def formulaire(etat: Dict, r: Dict) -> str:
    p = _actif(etat)
    if p["verrou"]:
        return _formulaire_verrouille(p, r)
    L = C.ligne
    pr = p["dir_principale"]
    coule = p["h"] - p["h_pre"]
    s1 = (L("Nom", C.texte(["nom"], p["nom"], "ex. Plancher haut R+1"), full=True)
          + L("Béton", C.choix(["beton"], p["beton"], [(k, k) for k in MO.classes_beton()]), full=True)
          + L("Acier", C.segments(["fyk"], p["fyk"], [(n, f"B{n}") for n in EN.NUANCES]), full=True)
          + L("largeur de bande", C.num(["b"], p["b"], "cm", entier=True, mini=EN.B_MIN, maxi=EN.B_MAX), "b",
              "Largeur de la bande étudiée : 100 cm = un mètre courant. Moments et effort tranchant sont ceux de la bande.")
          + '<div class="grp">Épaisseurs</div>'
          + L("épaisseur totale", C.num(["h"], p["h"], "cm", entier=True, mini=EN.H_MIN, maxi=EN.H_MAX), "h",
              "Prédalle et béton coulé en place ensemble : c'est h qui entre dans les formules.")
          + L("prédalle", C.num(["h_pre"], p["h_pre"], "cm", mini=EN.H_PRE_MIN, maxi=EN.h_pre_max(p["h"])), "h<sub>p</sub>",
              f"Épaisseur de la prédalle préfabriquée : de {C.nb(EN.H_PRE_MIN)} à {C.nb(EN.h_pre_max(p['h']))} cm ici : au moins 1 cm coulé en place, 15 cm au plus. "
              "Elle fixe la position des couches inférieures.")
          + _lecture("coulé en place", f"{C.nb(coule)} cm", "h − h<sub>p</sub>")
          + L("enrobage", C.num(["enrobage"], p["enrobage"], "cm", mini=0, maxi=EN.ENROBAGE_MAX), "c",
              "Du parement au nu des armatures (fond de moule de la prédalle, dessus de dalle). 0 est lu 3 cm, comme dans "
              "l'écran Streamlit.")
          + L("Principale", C.segments(["dir_principale"], pr, [("Y", "Direction Y"), ("X", "Direction X")]), full=True)
          + C.aide("La base inférieure de la direction principale est posée DANS la prédalle (fond de moule) ; la base "
                   "secondaire et tous les renforts inférieurs SUR la prédalle. Elle ordonne aussi les vérifications et les coupes.")
          + C.case(["verrou"], False, "Verrouiller la prédalle (lecture seule)"))
    cartes = ""
    for i, (s, rs) in enumerate(zip(p["sections"], r["sections"])):
        t_s = gouvernante_section(rs)["taux"]
        niv = "r" if rs["statut"] == MO.NOK else C.niveau(t_s)
        pct = f'<span class="pct {niv}">{C.pc(C.pct(t_s))}{C.NBSP}%</span>'
        retirer = _retirer(["sections", i, "-"], "Retirer la section") if len(p["sections"]) > 1 else ""
        ch = ["sections", i]
        indice = s["nom"] if len(s["nom"]) <= 3 else str(i + 1)
        cartes += (f'<div class="secard dl-sec"><div class="sh2"><span class="indice">{C.esc(indice)}</span>'
                   f'<span class="dl-st">{C.esc(rs["titre"])}</span>{pct}{retirer}</div>'
                   + L("Nom", C.texte(ch + ["nom"], s["nom"], "ex. A, travée"), full=True)
                   + '<div class="dl-mom"><span></span><span class="dl-h">M<sub>inf</sub> · bas tendu</span>'
                     '<span class="dl-h">M<sub>sup</sub> · haut tendu</span>'
                   + f'<span class="dl-d">X</span>{C.num(ch + ["Mx_inf"], s["Mx_inf"], "kN·m")}{C.num(ch + ["Mx_sup"], s["Mx_sup"], "kN·m")}'
                   + f'<span class="dl-d">Y</span>{C.num(ch + ["My_inf"], s["My_inf"], "kN·m")}{C.num(ch + ["My_sup"], s["My_sup"], "kN·m")}'
                   + f'<span class="dl-d">V</span>{C.num(ch + ["V"], s["V"], "kN")}<span class="dl-vh">tranchant max.</span></div>'
                   + _bouton(ch + ["copie"], "Copier la section", "copy", "Une nouvelle section, avec ces sollicitations et ces armatures")
                   + "</div>")
    s2 = (f'<div class="secards">{cartes}</div>' + _bouton(["sections", "+"], "Ajouter une section")
          + C.aide("M<sub>x</sub> : moments portés par les barres X, M<sub>y</sub> par les barres Y, sur la bande de "
                   "largeur b ; inf. tend le bas (travée), sup. tend le haut (appui). Valeurs de service, en positif."))
    blocs = ""
    for i, (s, rs) in enumerate(zip(p["sections"], r["sections"])):
        corps = C.aide("Par couche : le type, le treillis ou le Ø, l'espacement ou le nombre de barres, puis l'axe (cm, du "
                       "parement à l'axe) — automatique selon la pose (dans ou sur la prédalle, ou depuis le dessus) tant que "
                       "rien n'est saisi.")
        for f in MO.ordre_faces(rs["principale"]):
            corps += _face(i, f, s["faces"][f], rs["faces"][f], rs["principale"])
        p_ = rs["principale"]
        complet = (f"nappe inférieure {p_.upper()} (principale) : {rs['faces']['inf_' + p_]['libelle']} · "
                   f"{C.fr(rs['faces']['inf_' + p_]['As_pm'], 0)} mm²/m")
        resume = f'<span title="{C.esc(complet)}">{p_.upper()} {C.esc(_resume_face(rs["faces"]["inf_" + p_]))}</span>'
        # « Armatures · section A » ; une section nommée : « Armatures · Travée » (le titre court laisse voir le résumé)
        nom_court = rs["titre"][0].lower() + rs["titre"][1:] if len(s["nom"]) <= 3 or s["nom"].lower().startswith("section") else s["nom"]
        blocs += C.section(3 + i, f"Armatures · {C.esc(nom_court)}", resume, corps, i == 0, f"arm{i}")
    s_meth = (L("coefficient acier", C.num(["gamma_s"], p["gamma_s"], "", mini=EN.GAMMA_S_MIN, maxi=EN.GAMMA_S_MAX), "γ<sub>s</sub>",
                "fyd = fyk / γs — 1,5 par défaut (méthode ancienne) ; 1,15 donne 435 N/mm² en B500. Commun à toutes les prédalles de la note.")
              + L("jeu premier lit", C.num(["jeu_premier_lit"], p["jeu_premier_lit"], "cm", mini=0), "j",
                  "Ajouté à l'axe automatique des couches SUPÉRIEURES seulement (la base principale est en fond de moule, les "
                  "autres couches inférieures posées sur la prédalle). Commun à toutes les prédalles de la note.")
              + C.aide("γ<sub>s</sub> et le jeu sont communs à toutes les prédalles de la note, comme dans l'écran Streamlit.")
              + '<div class="grp">Reprendre une note Streamlit</div>'
              + C.aide("Le fichier « predalle_ba.json » de l'écran Streamlit (bouton Enregistrer) : ses prédalles remplacent "
                       "celles de cette note ; ce que l'écran y remplaçait en silence est signalé.")
              + C.fichier(["fichier_streamlit"], "Ouvrir un fichier Streamlit", ".json"))
    n = 3 + len(p["sections"])
    # le résumé d'une section repliée : court (il partage la ligne avec le titre), le détail au survol
    complet = (f"bande {C.nb(p['b'])} × {C.nb(p['h'])} cm dont prédalle {C.nb(p['h_pre'])} cm · {p['beton']} · B{p['fyk']} · "
               f"principale {pr}")
    resume1 = f'<span title="{C.esc(complet)}">{C.nb(p["h"])} ({C.nb(p["h_pre"])}) cm · {C.esc(p["beton"])}</span>'
    return (C.section(1, C.esc(_titre_predalle(p)), resume1, s1, True, "pd")
            + C.section(2, "Sections et sollicitations", f"{len(p['sections'])} section(s)", s2, True, "so")
            + blocs
            + C.section(n, "Méthode", f"γ<sub>s</sub> {C.nb(p['gamma_s'])} · jeu {C.nb(p['jeu_premier_lit'])} cm", s_meth, False, "me"))


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
    n = nom_ligne(v)
    return n[0].lower() + n[1:]


def _piece(v: Dict, s: Dict) -> str:
    g = s["grandeurs"]
    if v["cle"] == "hauteur":
        return f"h = {C.nb(g['h'])} cm"
    if v["cle"] == "tau":
        return f"V = {C.nb(s['V'])} kN"
    cs = s["faces"][v["cle"]]["couches"]
    if len(cs) == 1:
        return C.esc(cs[0]["libelle"])
    if len(cs) == 2:
        return C.esc(f"{cs[0]['valeur']} + {cs[1]['valeur']}")
    return C.esc(f"{cs[0]['valeur']} + {len(cs) - 1} renforts")


def _egal(a: str, b: str) -> str:
    """« a = b » sans retour à la ligne (espaces insécables) : une comparaison se coupe à son signe, jamais au milieu
    d'un membre (« 0,56 » d'un côté, « N/mm² » de l'autre)."""
    return f"{a}{C.NBSP}={C.NBSP}{b}"


def _comparaison(v: Dict, s: Dict):
    """(valeur, limite) de la vérification, en HTML : « 541 mm² » face à « A_s = 785 mm² »."""
    g, N = s["grandeurs"], C.NBSP
    if v["cle"] == "hauteur":
        return _egal("h<sub>min</sub>", f"{_n(g['h_min'], 'cm')}{N}cm"), _egal("h", f"{C.nb(g['h'])}{N}cm")
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
    dans une prédalle NON CONFORME, la plus chargée de celles que le moteur déclare non conformes — acier excessif ou
    d utile ≤ 0 le sont sous 100 %, et le verdict ne doit pas nommer une vérification conforme. On CHOISIT parmi les
    vérifications du moteur ; rien n'est recalculé. À égalité, la première dans l'ordre de l'écran (comme le moteur)."""
    return max(((i, v) for i, s in enumerate(r["sections"]) for v in s["verifications"]), key=lambda x: _cle_gouv(x[1]))


def gouvernante_section(s: Dict) -> Dict:
    """La vérification qui gouverne une section, selon la même règle que gouvernante()."""
    return max(s["verifications"], key=_cle_gouv)


def _cm(x: float) -> str:
    """Une longueur CALCULÉE en cm (y_G, d, axe) au plus au 1/100 : 3,5 ; 4,86 ; 3,8 (et non 3,7951)."""
    return C.nb(round(x, 2), 2)


def _verdict(v: Dict, s: Dict) -> str:
    """La comparaison du verdict : « valeur ≤ limite » ; pour une vérification non conforme sous 100 %, la comparaison
    qui la rend non conforme (A_s > A_s,max ; d ≤ 0)."""
    g, N = s["grandeurs"], C.NBSP
    if _sous_limite_non_conforme(v) and "acier_excessif" in v["motifs"]:
        return (f"{_egal('A<sub>s</sub>', _n(v['limite'], 'mm²') + N + 'mm²')} &gt; "
                f"{_egal('A<sub>s,max</sub>', _n(v['As_max'], 'mm²') + N + 'mm²')} · acier excessif")
    if _sous_limite_non_conforme(v) and "d_utile_negatif" in v["motifs"]:
        return (f"{_egal('d', _cm(g['d_' + v['cle']]) + N + 'cm')} ≤ 0 · position des couches incompatible avec "
                "l'épaisseur")
    val, lim = _comparaison(v, s)
    motif = f" · {_motifs(v)}" if v["motifs"] and v["taux"] <= 1 else ""
    return f"{val} {'≤' if v['taux'] <= 1 else '&gt;'} {lim}{motif}"


def _formule(f: Dict) -> Dict:
    """Une formule du moteur → « symbole = expression = valeurs substituées = résultat »."""
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    u = f["unite"]
    r = _n(f["resultat"], u) + (f" {u}" if u else "")
    k = f["cle"]
    if k == "e":
        s_ = f"e_{f['couche'] + 1}"
        if "saisi" in t:
            parts = [s_, "axe saisi"]
            r += " · saisi"
        elif f["pose"] == "sur_predalle":
            parts = [s_, "h_p + ⌈Ø/2⌉", f"{_v(t['h_pre'])} + {_v(t['demi_d'])}"]
            r += " · sur la prédalle"
        elif f["pose"] == "predalle":
            parts = [s_, "c + ⌈Ø/2⌉", f"{_v(t['c'])} + {_v(t['demi_d'])}"]
            r += " · dans la prédalle"
        else:
            parts = [s_, "c + ⌈Ø/2⌉ + j", f"{_v(t['c'])} + {_v(t['demi_d'])} + {_v(t['jeu'])}"]
    elif k == "hu_min":
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
        c0 = s["faces"][k]["couches"][0]
        sous = _piece(v, s) + (" · principale" if v.get("principale") else "") + f" · d = {_n(s['grandeurs'][f'd_{k}'], 'cm')} cm"
        if k.startswith("inf"):
            sous += f" · base {POSES_TXT[c0['pose']]}" if c0["auto"] else ""
    if v.get("motifs"):
        sous += " · " + _motifs(v)
    return {"nom": C.esc(nom_ligne(v)), "sous": sous, "formule": f, "valeur": val, "limite": lim, "taux": C.pct(v["taux"]),
            "niveau": _niv(v), "gov": gov}


def _calculer(p: Dict) -> Dict:
    return MO.verifier(_moteur(p))


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : sa prédalle la plus chargée — une prédalle non conforme d'abord (la règle de
    gouvernante())."""
    gouv = [gouvernante(r) for r in resultats]
    k = max(range(len(resultats)), key=lambda j: _cle_gouv(gouv[j][1]))
    r, (i_s, v) = resultats[k], gouv[k]
    niv = "r" if r["statut"] == MO.NOK else C.niveau(v["taux"])
    return {"taux": C.pct(v["taux"]), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['predalles'][k].get('nom') or 'Prédalle'} · {r['sections'][i_s]['titre']} · {nom_minuscule(v)}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    items = ""
    for i, (p, r) in enumerate(zip(etat["predalles"], resultats)):
        t = gouvernante(r)[1]["taux"]
        niv = "r" if r["statut"] == MO.NOK else C.niveau(t)
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["predalles"]) > 1 else ""
        verrou = C.icon("lock") if p["verrou"] else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{verrou}'
                  f'{C.esc(p.get("nom") or f"Prédalle {i + 1}")} '
                  f'<span>· {C.nb(p["h"])} ({C.nb(p["h_pre"])}) cm · {C.esc(p["beton"])}</span></span>'
                  f'<span class="pct {niv}">{C.pc(C.pct(t))}{C.NBSP}%</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["predalles"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier la prédalle ouverte">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une prédalle</div></div>')


def _optimaux(p: Dict, r: Dict) -> Dict:
    """La plus petite nappe de base qui tient, pour chaque face de chaque section (l'étoile des lignes du résultat)."""
    out = {}
    for i, s in enumerate(r["sections"]):
        for f in EN.FACES:
            cands = MO.candidats_base(_moteur(p), i, f)
            o = next((c for c in cands if c["optimal"]), None)
            if o is not None and not o["retenue"]:
                out[(i, f)] = o
    return out


def _avertissements(r: Dict) -> str:
    """Les avertissements du moteur ; les bases X et Y d'une même face calculées trop près l'une de l'autre sont dites en une
    phrase par section."""
    out = ""
    confondues: Dict[int, List[Dict]] = {}
    for a in r["avertissements"]:
        if a["code"] == "nappes_confondues":
            confondues.setdefault(a["section"], []).append(a)
        elif a["code"] == "enrobage_nul":
            out += C.message("Enrobage nul : comme l'écran Streamlit, le calcul pose les couches avec un enrobage de 3 cm.")
        else:
            out += C.message(C.esc(a["message"]), grave=a["code"] in ("d_utile_negatif", "base_hors_predalle"))
    for i, xs in confondues.items():
        s = r["sections"][i]
        faces = " et ".join("inférieures" if a["face"] == "inf" else "supérieures" for a in xs)
        axes = " ; ".join(f"{'inf.' if a['face'] == 'inf' else 'sup.'} {C.fr(a['e_x'], 1)} et {C.fr(a['e_y'], 1)} cm" for a in xs)
        mini = max(a["ecart_mini"] for a in xs)
        meme = all(abs(a["e_x"] - a["e_y"]) < 1e-9 for a in xs)
        niveau = "au même niveau" if meme else "à des niveaux trop proches pour se croiser"
        out += C.message(f"{C.esc(s['titre'])} : les bases X et Y {faces} sont calculées {niveau} ({axes}), comme dans "
                         f"l'écran Streamlit. Croisées, l'une passe sur l'autre (il faut au moins {C.fr(mini, 1)} cm d'axe à "
                         f"axe) : son d est plus petit que celui calculé. Saisir son axe pour en tenir compte.")
    return out


# Ce que le module VÉRIFIE, et ce qu'il ne vérifie pas (l'écran Streamlit non plus) : dit sous chaque résultat et dans
# chaque fenêtre de calcul, pour qu'un « conforme » ne se lise pas plus large qu'il n'est.
PERIMETRE = ("Vérifié : la section finale (prédalle et béton coulé en place ensemble), en flexion (hauteur, quatre nappes) "
             "et τ ≤ τ<sub>adm,I</sub>. Non vérifiés, comme dans l'écran Streamlit : la phase provisoire (prédalle seule, "
             "étaiement, levage), le cisaillement à l'interface prédalle / béton coulé en place, la flèche et la fissuration.")


def _hypotheses(p: Dict, m: Dict) -> List[str]:
    return [f"Béton <b>{C.esc(m['beton'])}</b> · f<sub>ck</sub> = {C.nb(m['fck'])} · f<sub>ck,cube</sub> = {C.nb(m['fck_cube'])} N/mm²",
            f"f<sub>ctm</sub> = 0,30·f<sub>ck</sub><sup>2/3</sup> = {C.fr(m['fctm'], 2)} N/mm²",
            f"Acier <b>B{int(m['fyk'])}</b> · f<sub>yd</sub> = {int(m['fyk'])} / {C.nb(m['gamma_s'])} = {C.fr(m['fyd'], 1)} N/mm²",
            f"α<sub>b</sub> = {C.nb(m['alpha_b'])} · μ = {C.nb(m['mu'], 4)}",
            f"Bande b × h = {C.nb(p['b'])} × {C.nb(p['h'])} cm, dont prédalle <b>{C.nb(p['h_pre'])} cm</b> et "
            f"{C.nb(p['h'] - p['h_pre'])} cm coulés en place",
            f"Enrobage {C.nb(p['enrobage'])} cm · jeu premier lit {C.nb(p['jeu_premier_lit'])} cm (nappes supérieures)"]


def _efforts_txt(s: Dict) -> str:
    m = [f"{SOLLICITATIONS[k]} = {C.nb(s[k])}" for k in EN.MOMENTS if s[k]]
    txt = (" · ".join(m) + " kN·m") if m else "aucun moment"
    return txt + (f" · V = {C.nb(s['V'])} kN" if s["V"] else "")


def resultat(etat: Dict) -> Dict:
    p = _actif(etat)
    r = _calculer(p)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["predalles"])]
    i_g, v_g = gouvernante(r)
    t = v_g["taux"]
    niv = "r" if r["statut"] == MO.NOK else C.niveau(t)
    s_g = r["sections"][i_g]
    verdict = (f'<div class="verdict">{C.chip(C.pct(t), niv, xl=True)}<div class="gov"><b>{C.esc(s_g["titre"])} · '
               f'{C.esc(nom_minuscule(v_g))}</b>{_verdict(v_g, s_g)}</div></div>')
    opt = {} if p["verrou"] else _optimaux(p, r)
    rows = ""
    for i, s in enumerate(r["sections"]):
        t_s = gouvernante_section(s)["taux"]
        n_s = "r" if s["statut"] == MO.NOK else C.niveau(t_s)
        rows += (f'<div class="elgrp">{C.esc(s["titre"])} · principale {s["principale"].upper()}'
                 f'<span class="pct {n_s}">{C.pc(C.pct(t_s))}{C.NBSP}%</span></div>')
        for v in s["verifications"]:
            face = v["cle"] in EN.FACES
            o = opt.get((i, v["cle"])) if face else None
            leger = f"Plus petite nappe de base qui tient : {o['nom']} · {C.pc(C.pct(o['taux']))} %" if o else ""
            piece = _piece(v, s)
            if _sous_limite_non_conforme(v):          # non conforme sous 100 % : la ligne dit pourquoi
                piece += " · " + ("acier excessif" if "acier_excessif" in v["motifs"] else "d utile ≤ 0")
            rows += C.ligne_res(nom_ligne(v), piece, C.pct(v["taux"]), _niv(v), calcul=f"s{i}",
                                pk=f"{v['cle']}-{i}" if face and not p["verrou"] else "",
                                gov=(i == i_g and v is v_g), leger=leger)
    pied = (f'<div class="pfoot"><div class="mass"><span>Méthode du bureau · moteur prédalle</span>'
            f'<b>v{C.esc(MO.REFERENCE.split(" v")[-1])}</b></div><div class="legend">{C.legende()}</div></div>')
    volets = ""
    hyps = _hypotheses(p, r["materiaux"])
    for i, s in enumerate(r["sections"]):
        v_s = gouvernante_section(s)
        lignes = [_ligne_tableau(v, s, v is v_s) for v in s["verifications"]]
        detail = ""
        for v in s["verifications"]:
            titre = v["titre"]
            if v["cle"] == "hauteur" and v.get("face"):
                cote, dk = v["face"].split("_")
                titre += f" · la famille {cote}. {dk.upper()} gouverne (la plus exigeante)"
            detail += (f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div>'
                       f'{C.lignes_fx([_formule(f) for f in v["formules"]] + [_taux_fx(v, s)])}</div>')
        volets += C.fenetre_calcul(f"s{i}", f"{C.esc(p['nom'])} · {C.esc(s['titre'])}",
                                   f"{_efforts_txt(s)} · méthode du bureau", hyps, lignes, detail, pied=C.aide(PERIMETRE))
    vues = []
    for i, s in enumerate(r["sections"]):
        nom = s["nom"] if not s["nom"].lower().startswith("section") else s["titre"]
        vues.append({"cle": f"c{i}", "titre": f"Coupes · {s['titre']}", "court": f"Coupe {nom}"[:24],
                     "svg": DP.svg_coupe(f"pd_c{i}", s, r["predalle"]),
                     "legende": f"prédalle hachurée · barres coupées en points · principale {s['principale'].upper()}"})
        vues.append({"cle": f"p{i}", "titre": f"Plan des nappes · {s['titre']}", "court": f"Plan {nom}"[:24],
                     "svg": DP.svg_plan(f"pd_p{i}", s, r["predalle"]),
                     "legende": "vue de dessus · barres X horizontales, Y verticales"})
    phrase = (f"{C.nb(p['b'])} × {C.nb(p['h'])} cm, dont prédalle {C.nb(p['h_pre'])} cm · {C.esc(p['beton'])} · B{p['fyk']} · "
              f"principale {p['dir_principale']}")
    return {"resultat": f'<div class="pbody">{verdict}<div class="els">{rows}</div>{_avertissements(r)}{C.aide(PERIMETRE)}</div>{pied}',
            "volets": volets, "vues": vues, "phrase": phrase, "statut_html": C.statut(niv),
            "resume": resume_note(etat, resultats), "donnees": formulaire(etat, r), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de la nappe de base
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les nappes de base possibles d'une face (treillis de la base, barres Ø8 à Ø25 de 100 à 250 mm), triées par section
    au mètre, calculées par le moteur avec l'axe que la règle de la prédalle leur donne : la retenue (✓), la plus petite
    qui tient (★)."""
    face, i = role.rsplit("-", 1)
    i = int(i)
    p = _actif(etat)
    fams = tuple(f for f in (familles or MO.FAMILLES) if f in MO.FAMILLES) or MO.FAMILLES
    cands = MO.candidats_base(_moteur(p), i, face, fams)
    i_o = next((k for k, c in enumerate(cands) if c["optimal"]), 0)
    i_c = next((k for k, c in enumerate(cands) if c["retenue"]), i_o)
    base = p["sections"][i]["faces"][face]["couches"][0]
    pose = POSES_TXT[MO.pose(face, 0, p["dir_principale"])]

    def prop(c):
        if c["type"] == "treillis":
            t = TR.fils(c["treillis"])
            q = f"treillis soudé · porteurs Ø{t[0]} tous les {t[2]} mm · répartition Ø{t[1]} tous les {t[3]} mm"
        else:
            q = f"barres Ø{c['d']} tous les {c['esp']} mm"
        axe = f"axe {C.fr(c['e'], 1)} cm" + ("" if base["dist"] is not None else f" ({pose})")
        return (f"{q} · {C.fr(c['As_pm'], 0)} mm²/m · {axe} · face : A<sub>s</sub> = {C.fr(c['As'], 0)} mm², "
                f"y<sub>G</sub> = {C.fr(c['yG'], 2)} cm")
    rows = [{"nom": c["nom"], "col": C.fr(c["As_pm"], 0), "taux": C.pct(min(c["taux"], 9.99)),
             "niveau": "r" if c["etat"] == MO.NOK else C.niveau(c["taux"]), "cur": c["retenue"], "opt": c["optimal"],
             "prop": prop(c)} for c in cands]
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{f}">{LIB_FAMILLES[f]}</button>'
                      for f in MO.FAMILLES)
    s = p["sections"][i]
    renforts = len(s["faces"][face]["couches"]) - 1
    note = f" · {renforts} renfort(s) gardé(s)" if renforts else ""
    if base["type"] == "n_barres":
        note += f" · la base actuelle ({base['n']} Ø{base['d']}) n'est pas dans la liste"
    dk = face.split("_")[1]
    princ = " (principale)" if dk == p["dir_principale"].lower() else ""
    return C.selecteur(role, f"Nappe {FACES_TXT[face.split('_')[0]]} {dk.upper()} · {C.esc(EN.libelle_section(s['nom']))}",
                       f"couche de base{princ}, {pose} · triée par section au mètre · un clic la retient{note}", filtres, rows,
                       max(0, min(i_o, i_c) - 4), "mm²/m", etoile="la plus petite section qui tient")
