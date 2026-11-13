# -*- coding: utf-8 -*-
"""Le module « Prédalles » du site : l'état d'une note (une ou plusieurs prédalles, chacune avec ses sections, et pour
chaque section quatre nappes — inférieure et supérieure, directions X et Y — faites de couches de treillis, de barres
Ø/e ou de « n barres »), les champs de saisie, le résultat section par section, la grande fenêtre de calcul, le
sélecteur de la nappe de base et les dessins — rendus en fragments HTML.

L'écran est l'établi COMPACT, celui de la poutre et des dalles (api_web/etabli_compact.py, api_web/plaque_compact.py) :
à gauche l'arborescence de la note (chaque prédalle, et sous la prédalle ouverte ses quatre rubriques), au centre les vues
sous leur en-tête technique, à droite un bloc par section. Les couches des données et des dessins portent la même clé.

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
- un verdict qui, dans une prédalle non conforme, nomme une vérification NON conforme (gouvernante()) : acier excessif
  et d utile ≤ 0 le sont sous 100 % ;
- des bornes contre l'absurde, que l'écran n'a pas (SOLLICITATION_MAX, EN.H_MAX pour les longueurs saisies, NOM_MAX).
"""
import copy
import sys
from typing import Dict, List, Optional, Sequence

from beton import treillis as TR
from beton.predalle import entree as EN
from beton.predalle import fichier as FI
from beton.predalle import moteur as MO

from . import commun as C
from . import dessin_predalle as DP
from . import etabli_compact as E
from . import plaque_compact as PC

MODULE = "predalle"
PAGE = "predalle"

# des requêtes de plus pour le parcours (api_web/parcours.py) : les champs que le parcours automatique n'essaie pas
PARCOURS = [
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_x", "couches", "0", "dist"], "valeur": "8"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_y", "yG"], "valeur": "4,2"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "sup_x", "couches", "0", "dist"], "valeur": ""},
    {"action": "modifier", "chemin": ["sections", "0", "Mx_inf"], "valeur": "-20"},
    {"action": "modifier", "chemin": ["h_pre"], "valeur": "21"},
    {"action": "modifier", "chemin": ["h"], "valeur": "6"},
    {"action": "modifier", "chemin": ["h_pre"], "valeur": "3"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_x", "couches", "0", "esp"], "valeur": "600"},
    {"action": "selecteur", "role": "sup_x-0", "familles": ["treillis"]},
    {"action": "choisir", "role": "inf_y-0", "nom": "Ø12/150"},
]

TYPES = PC.TYPES
FACES_TXT = PC.FACES_TXT
POSES_TXT = {"predalle": "dans la prédalle", "sur_predalle": "sur la prédalle", "superieure": "depuis le dessus"}
POSES_AIDE = {"predalle": "Axe automatique : la base principale est posée en fond de moule, DANS la prédalle — enrobage + ½ Ø "
                          "arrondi au 0,5 cm (sans jeu premier lit).",
              "sur_predalle": "Axe automatique : couche posée SUR la prédalle, au chantier — épaisseur de la prédalle + ½ Ø "
                              "arrondi au 0,5 cm.",
              "superieure": "Axe automatique, depuis le dessus : enrobage + ½ Ø arrondi au 0,5 cm + jeu premier lit."}
# la légende des dessins (dernière ligne de l'en-tête de chaque vue : ce qu'on y lit)
LEGENDES = {"c": "prédalle hachurée · barres coupées en points, barres parallèles en traits",
            "p": "vue de dessus · barres X horizontales, Y verticales"}
MOTIFS = {"hauteur_insuffisante": "hauteur insuffisante", "d_utile_negatif": "d utile ≤ 0",
          "acier_insuffisant": "acier insuffisant", "acier_excessif": "acier excessif (A<sub>s</sub> &gt; A<sub>s,max</sub>)",
          "cisaillement_excessif": "τ &gt; τ<sub>adm,I</sub> (pas d'étriers dans une dalle)"}
LIB_FAMILLES = {"treillis": "Treillis", "barres": "Barres"}


class Refus(ValueError):
    """Une saisie refusée : son message est montré tel quel. (Le module fusionné api_web/dalle.py, qui réemploie ce
    fichier comme AIDE du type prédalle, l'attrape sous ce nom.)"""


_Refus = Refus


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
    """Les prédalles de la note : en ouvrir une, en ajouter, en copier une (``index`` ; sans : l'ouverte), en supprimer
    une (jamais la dernière). Une prédalle ajoutée prend le γs et le jeu de la note (paramètres communs, comme dans
    l'écran Streamlit)."""
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
        # la prédalle désignée (le menu « ⋯ » de l'arborescence ; sans rang : l'ouverte), copiée juste après elle
        i = etat["actif"] if index is None else min(max(0, int(index)), len(ps) - 1)
        p = copy.deepcopy(ps[i])
        p["nom"] = f"{p.get('nom') or 'Prédalle'} (copie)"
        ps.insert(i + 1, p)
        etat["actif"] = i + 1
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


def saisi(x: float) -> str:
    """Un nombre SAISI, rappelé dans un message de refus : exact s'il est raisonnable, en puissance de dix sinon (un
    1e300 tapé ne s'écrit pas sur trois cents chiffres)."""
    if abs(x) < 1e7:
        ent, _, dec = C.nb(x, 6).partition(",")
        signe, ent = ("-", ent[1:]) if ent.startswith("-") else ("", ent)
        return signe + f"{int(ent):,}".replace(",", C.NBSP) + ("," + dec if dec else "")
    m, e = f"{x:.2e}".split("e")
    exposant = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m.replace('.', ',')} × 10{exposant}"


_saisi = saisi


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


def nom_saisi(valeur, quoi: str, vide_admis: bool) -> str:
    """Un nom saisi (prédalle, dalle, section) : NOM_MAX caractères au plus ; vide refusé pour une section."""
    t = str(valeur)
    if not vide_admis and not t.strip():
        raise _Refus(f"Nom {quoi} vide")
    if len(t) > NOM_MAX:
        raise _Refus(f"Nom {quoi} : {NOM_MAX} caractères au plus ({len(t)} saisis) — il titre la note, les coupes et la "
                     "fenêtre de calcul")
    return t


_nom = nom_saisi


def bornes_site(p: Dict) -> None:
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


_bornes_site = bornes_site


def peau(p: Dict, valeur) -> float:
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


_peau = peau


def epaisseur_totale(p: Dict, valeur) -> int:
    """L'épaisseur totale d'une prédalle : dans les bornes de l'écran, et telle qu'elle contienne encore la peau (au
    moins COULE_MIN coulé en place) — sinon refusée : l'écran ramenait la peau sans le dire."""
    h = _nombre("h", valeur)
    if p["h_pre"] > EN.h_pre_max(h):
        raise _Refus(f"Épaisseur totale : la prédalle de {C.nb(p['h_pre'])} cm demande au moins "
                     f"{C.nb(p['h_pre'] + EN.COULE_MIN)} cm (un centimètre coulé en place) ; réduire d'abord la prédalle. "
                     "L'écran Streamlit ramenait la prédalle sans le dire.")
    return h


def axe_auto(p: Dict, c: Dict, j: int, f: str) -> float:
    return MO.distance_auto(c, j, f, p["dir_principale"], p["enrobage"], p["jeu_premier_lit"], p["h_pre"])


def yG_calcule(p: Dict, f: str, face: Dict) -> float:
    axes = [c["dist"] if c["dist"] is not None else axe_auto(p, c, j, f) for j, c in enumerate(face["couches"])]
    return FI.yG_calcule(face["couches"], axes, p["b"])


_axe_auto, _yG_calcule = axe_auto, yG_calcule


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
            p["h"] = epaisseur_totale(p, valeur)
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


# ------------------------------------------------------------------ les données : l'arborescence (colonne de gauche)
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


def sur_principale(x: Dict) -> str:
    """Sur quoi repose une couche (ou sa formule « e ») que le site a empilée, dans les mots de la prédalle : « la base
    principale » (comme avant S5), sinon ce que ``dalle_positions.annoter`` en dit (``sur`` : « le renfort Y », « le
    renfort R2 Y ») — S5 : un renfort principal plus haut que la base porte la nappe secondaire."""
    return "la base principale" if not x.get("appui") else (x.get("sur") or "la nappe principale")


def pose_couche(c: Dict, geo_c: Dict) -> str:
    """OÙ la règle de la prédalle pose une couche (dans la prédalle, sur la prédalle, depuis le dessus), et l'axe
    automatique qu'elle en tire — ou, l'axe saisi, ce que l'automatique serait : la ligne du détail de la couche. Une
    couche secondaire EMPILÉE par le site (module fusionné, § 4.1 : faces supérieures ; sa base et, depuis le 09/11/2026,
    ses renforts — Ø_s est alors celui du renfort) le dit, et sur quoi elle repose : la base principale, ou le renfort
    principal plus haut qu'elle (S5 : e_p, Ø_p sont alors ceux du renfort)."""
    pose = geo_c["pose"]
    if geo_c.get("empilee"):
        # (une couche supérieure secondaire que le site a posée sur la principale : « depuis le dessus (c + ½ Ø + j) » mentirait)
        txt = (f"axe auto · posée sur {sur_principale(geo_c)} (e<sub>p</sub> + ½ Ø<sub>p</sub> + ½ Ø<sub>s</sub>)")
    elif c["dist"] is None:
        txt = f"axe auto · {POSES_TXT[pose]} ({REGLES_TXT[pose]})"
    else:
        txt = f"axe saisi · l'automatique serait {C.nb(geo_c['e_auto'])} cm, {POSES_TXT[pose]}"
    return f'<div class="pd-pose" title="{C.esc(POSES_AIDE[pose])}">{_picto_pose(pose)}<span>{txt}</span></div>'


_pose = pose_couche


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
            s2 += _lecture(PC.titre_face(f).lower() + (" · principale" if f.endswith(rs["principale"]) else ""),
                           C.esc(rs["faces"][f]["libelle"]))
    return (C.section(1, f"Prédalle {C.icon('lock')}", "verrouillée", s1, True, "pd")
            + C.section(2, "Sections et armatures", f"{len(p['sections'])} section(s)", s2, False, "sl"))


def _options(p: Dict) -> str:
    """La rubrique « Options » : la méthode (une ligne), γs et le jeu du premier lit, communs à la note (la reprise d'une
    note Streamlit est retirée le 13/11/2026 : le bureau n'a aucune ancienne note à reprendre)."""
    return (f'<div class="row full et-meth"><label><span>Méthode</span></label>'
            f'<b title="{C.esc(MO.REFERENCE)}">Méthode du bureau · ELS · z = 0,9 d</b></div>'
            + C.ligne2(("γ<sub>s</sub> acier", C.num(["gamma_s"], p["gamma_s"], "", mini=EN.GAMMA_S_MIN, maxi=EN.GAMMA_S_MAX),
                        "Coefficient partiel de l'acier : f_yd = f_yk / γs (1,5 : méthode du bureau). Commun à toutes les "
                        "prédalles de la note."),
                       ("Jeu 1er lit", C.num(["jeu_premier_lit"], p["jeu_premier_lit"], "cm", mini=0),
                        "Écart ajouté entre l'enrobage et le premier lit des nappes SUPÉRIEURES (les nappes inférieures sont "
                        "posées dans et sur la prédalle). Commun à toutes les prédalles de la note.")))


def formulaire(etat: Dict, r: Optional[Dict] = None) -> str:
    """Les quatre rubriques de la prédalle ouverte, aux places de celles de la poutre et des dalles : 1 Prédalle,
    2 Sections et sollicitations, 3 Armatures (de la section courante), 4 Options. ``r`` : son calcul, refait s'il
    manque."""
    p = _actif(etat)
    r = r or _calculer(p)
    if p["verrou"]:
        return _formulaire_verrouille(p, r)
    L = C.ligne
    # condensée (demande de l'ingénieur, 11/2026) : le nom se modifie dans l'arborescence ; deux champs par ligne, les
    # explications dans l'infobulle ⓘ de leur champ
    s1 = ('<div class="et-piece et-plat" data-cible="beton">'
          + C.ligne2(("Béton", C.choix(["beton"], p["beton"], [(k, k) for k in MO.classes_beton()]), ""),
                     ("Acier", C.choix(["fyk"], p["fyk"], [(n, f"B{n}") for n in EN.NUANCES]), ""))
          + C.ligne2(("b bande", C.num(["b"], p["b"], "cm", entier=True, mini=EN.B_MIN, maxi=EN.B_MAX),
                      "Largeur de la bande étudiée : 100 cm = un mètre courant. Moments et effort tranchant sont ceux de la bande."),
                     ("h totale", C.num(["h"], p["h"], "cm", entier=True, mini=EN.H_MIN, maxi=EN.H_MAX),
                      "Épaisseur totale : prédalle et béton coulé en place ensemble ; c'est h qui entre dans les formules."))
          + C.ligne2(("h<sub>p</sub> prédalle", C.num(["h_pre"], p["h_pre"], "cm", mini=EN.H_PRE_MIN, maxi=EN.h_pre_max(p["h"])),
                      f"Épaisseur de la prédalle préfabriquée : de {C.nb(EN.H_PRE_MIN)} à {C.nb(EN.h_pre_max(p['h']))} cm ici (au "
                      "moins 1 cm coulé en place, 15 cm au plus). Elle fixe la position des couches inférieures."),
                     ("c enrobage", C.num(["enrobage"], p["enrobage"], "cm", mini=0, maxi=EN.ENROBAGE_MAX),
                      "Du parement au nu des armatures (fond de moule de la prédalle, dessus de dalle). 0 est lu 3 cm, comme "
                      "dans l'écran Streamlit."))
          + C.ligne2(("Principale", C.segments(["dir_principale"], p["dir_principale"], [("Y", "Y"), ("X", "X")]),
                      "La base inférieure de la direction principale est posée DANS la prédalle (fond de moule) ; la base "
                      "secondaire et tous les renforts inférieurs SUR la prédalle. Elle ordonne aussi les vérifications et les "
                      "coupes."),
                     ("coulé h − h<sub>p</sub>", f'<span class="dl-ro">{C.nb(p["h"] - p["h_pre"])} cm</span>',
                      "Épaisseur du béton coulé en place au chantier, sur la prédalle."))
          + "</div>" + C.case(["verrou"], False, "Verrouiller la prédalle (lecture seule)"))
    contenus, resumes = [], []
    for i, (s, rs) in enumerate(zip(p["sections"], r["sections"])):
        verifs = {v["cle"]: v for v in rs["verifications"]}
        corps = ""
        for f in MO.ordre_faces(rs["principale"]):
            couches = "".join(PC.couche(EN, i, f, j, c, g, details=_pose(c, g), aide_axe=POSES_AIDE[g["pose"]], dit_auto=False)
                              for j, (c, g) in enumerate(zip(s["faces"][f]["couches"], rs["faces"][f]["couches"])))
            pose = "sur la prédalle" if f.startswith("inf") else "dans le plan de la base"
            corps += PC.nappe(EN, i, f, s["faces"][f], rs["faces"][f], verifs[f], rs["principale"], couches,
                              f"Un renfort : treillis, barres Ø/e ou n barres ; 3 Ø12 par défaut, posées {pose}")
        contenus.append(corps)
        resumes.append(PC.resume_armatures(rs))
    s3, r3 = PC.rubrique_armatures([s["nom"] for s in p["sections"]], contenus, resumes)
    resume1 = f"{C.esc(p['beton'])} · B{p['fyk']} · h<sub>p</sub> = {C.nb(p['h_pre'])} cm"
    return PC.formulaire("pd", "Prédalle", resume1, s1, PC.cartes(p["sections"]), len(p["sections"]), s3, r3,
                         f"γ<sub>s</sub> {C.nb(p['gamma_s'])} · jeu {C.nb(p['jeu_premier_lit'])} cm", _options(p))


def arbre(etat: Dict, r: Optional[Dict] = None) -> str:
    """La colonne de gauche : l'ARBORESCENCE de la note (api_web/etabli_compact.py, la même que la poutre et les
    dalles). Chaque prédalle, « nom · b × h cm » (un cadenas si elle est verrouillée) ; la prédalle ouverte montre ses
    quatre rubriques. Copier, supprimer, renommer : dans le menu « ⋯ » de la prédalle."""
    elements = [(q.get("nom") or f"Prédalle {i + 1}", f"· {C.nb(q['b'])} × {C.nb(q['h'])} cm", q["verrou"])
                for i, q in enumerate(etat["predalles"])]
    return E.arbre(E.PREDALLE, elements, etat["actif"], formulaire(etat, r), "predalle")


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
    qui la rend non conforme (A_s > A_s,max) ; pour une face dont le d utile est ≤ 0, quel que soit son taux, « d ≤ 0 »."""
    g, N = s["grandeurs"], C.NBSP
    if _sous_limite_non_conforme(v) and "acier_excessif" in v["motifs"]:
        return (f"{_egal('A<sub>s</sub>', _n(v['limite'], 'mm²') + N + 'mm²')} &gt; "
                f"{_egal('A<sub>s,max</sub>', _n(v['As_max'], 'mm²') + N + 'mm²')} · acier excessif")
    if "d_utile_negatif" in v["motifs"]:          # (quel que soit le taux : la raison, et non un taux sans objet)
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
        if f.get("empilee") and "e_p" in t:
            # une couche secondaire posée par le SITE sur la nappe principale (module fusionné, § 4.1 ; les termes e_p,
            # demi_p, demi_s sont posés par dalle_positions.annoter, e_p et demi_p ceux de la couche qui porte : la base,
            # ou un renfort principal plus haut, S5) : le moteur a reçu l'axe imposé, la formule dit d'où il vient
            parts = [s_, "e_p + ⌈Ø_p/2⌉ + ⌈Ø_s/2⌉", f"{_v(t['e_p'])} + {_v(t['demi_p'])} + {_v(t['demi_s'])}"]
            r += f" · posée sur {sur_principale(f)}"
        elif "saisi" in t:
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
    gouvernante()) ; « Données incomplètes » dès qu'une section n'a aucun effort (comme la poutre et les dalles)."""
    gouv = [gouvernante(r) for r in resultats]
    k = max(range(len(resultats)), key=lambda j: _cle_gouv(gouv[j][1]))
    r, (i_s, v) = resultats[k], gouv[k]
    nivs = [PC.niveau_element(MO, _MOI, x) for x in resultats]
    niv = next((n for n in ("r", "n") if n in nivs), C.niveau(v["taux"]))
    out = {"taux": C.pct(v["taux"]) if niv != "n" else None, "niveau": niv, "statut": C.STATUTS[niv][1],
           "gouvernant": f"{etat['predalles'][k].get('nom') or 'Prédalle'} · {r['sections'][i_s]['titre']} · {nom_minuscule(v)}"}
    if niv in C.STATUTS_COURTS:
        out["court"] = C.STATUTS_COURTS[niv]   # la pastille du téléphone
    return out


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les prédalles de la note sont l'ARBORESCENCE des données (``arbre``), avec leurs rubriques."""
    return ""


def avertissements(r: Dict, i: Optional[int] = None) -> str:
    """Les avertissements du moteur : ceux de la section ``i`` (posés dans son bloc), ou, sans ``i``, ceux de la
    prédalle entière ; les bases X et Y d'une même face calculées trop près l'une de l'autre sont dites en une phrase."""
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
            out += C.message(C.esc(a["message"]), grave=a["code"] == "base_hors_predalle")
    if confondues:
        s = r["sections"][i]
        faces = " et ".join("inférieures" if a["face"] == "inf" else "supérieures" for a in confondues)
        axes = " ; ".join(f"{'inf.' if a['face'] == 'inf' else 'sup.'} {C.fr(a['e_x'], 1)} et {C.fr(a['e_y'], 1)} cm" for a in confondues)
        mini = max(a["ecart_mini"] for a in confondues)
        meme = all(abs(a["e_x"] - a["e_y"]) < 1e-9 for a in confondues)
        niveau = "au même niveau" if meme else "à des niveaux trop proches pour se croiser"
        out += C.message(f"{C.esc(s['titre'])} : les bases X et Y {faces} sont calculées {niveau} ({axes}), comme dans "
                         f"l'écran Streamlit. Croisées, l'une passe sur l'autre (il faut au moins {C.fr(mini, 1)} cm d'axe à "
                         f"axe) : son d est plus petit que celui calculé. Saisir son axe pour en tenir compte.")
    return out


_avertissements = avertissements

# Ce que le module VÉRIFIE, et ce qu'il ne vérifie pas : en entier dans chaque fenêtre de calcul, pour qu'un
# « conforme » ne se lise pas plus large qu'il n'est ; sous le résultat, la seule phrase « Non vérifiés » (NON_VERIFIES) —
# ce qui est vérifié, les lignes du résultat le montrent déjà (le propriétaire, 13/11/2026).
NON_VERIFIES = ("Non vérifiés : la phase provisoire (prédalle seule, étaiement, levage), le cisaillement à l'interface "
                "prédalle / béton coulé en place, la flèche et la fissuration.")
PERIMETRE = ("Vérifié : la section finale (prédalle et béton coulé en place ensemble), en flexion (hauteur, quatre nappes) "
             "et τ ≤ τ<sub>adm,I</sub>. " + NON_VERIFIES)


def hypotheses(p: Dict, m: Dict) -> List[str]:
    return [f"Béton <b>{C.esc(m['beton'])}</b> · f<sub>ck</sub> = {C.nb(m['fck'])} · f<sub>ck,cube</sub> = {C.nb(m['fck_cube'])} N/mm²",
            f"f<sub>ctm</sub> = 0,30·f<sub>ck</sub><sup>2/3</sup> = {C.fr(m['fctm'], 2)} N/mm²",
            f"Acier <b>B{int(m['fyk'])}</b> · f<sub>yd</sub> = {int(m['fyk'])} / {C.nb(m['gamma_s'])} = {C.fr(m['fyd'], 1)} N/mm²",
            f"α<sub>b</sub> = {C.nb(m['alpha_b'])} · μ = {C.nb(m['mu'], 4)}",
            f"Bande b × h = {C.nb(p['b'])} × {C.nb(p['h'])} cm, dont prédalle <b>{C.nb(p['h_pre'])} cm</b> et "
            f"{C.nb(p['h'] - p['h_pre'])} cm coulés en place",
            f"Enrobage {C.nb(p['enrobage'])} cm · jeu premier lit {C.nb(p['jeu_premier_lit'])} cm (nappes supérieures)"]


_hypotheses = hypotheses


def _efforts_txt(s: Dict) -> str:
    m = [f"{SOLLICITATIONS[k]} = {C.nb(s[k])}" for k in EN.MOMENTS if s[k]]
    txt = (" · ".join(m) + " kN·m") if m else "aucun moment"
    return txt + (f" · V = {C.nb(s['V'])} kN" if s["V"] else "")


def resultat(etat: Dict) -> Dict:
    p = _actif(etat)
    r = _calculer(p)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["predalles"])]
    nom = p.get("nom") or "Prédalle"
    hyps = _hypotheses(p, r["materiaux"])
    gouv = gouvernante(r)
    blocs, volets = "", ""
    for i, s in enumerate(r["sections"]):
        h, v = PC.bloc(_MOI, i, s, nom, hyps, gouv, pied_calcul=C.aide(PERIMETRE), pied=_avertissements(r, i))
        blocs += h
        volets += v
    dims = f"{C.nb(p['b'])} × {C.nb(p['h'])} cm, prédalle {C.nb(p['h_pre'])} cm · {C.esc(p['beton'])} · B{p['fyk']}"
    pied = (f'<div class="pfoot"><div class="mass"><span>Méthode du bureau · moteur prédalle</span>'
            f'<b>v{C.esc(MO.REFERENCE.split(" v")[-1])}</b></div><div class="legend">{C.legende()}</div></div>')
    # l'état de la NOTE (toutes ses prédalles) dans l'en-tête du panneau : utile dès deux prédalles
    nivs = [PC.niveau_element(MO, _MOI, x) for x in resultats]
    statut = C.statut(next((n for n in ("r", "n") if n in nivs), max(nivs, key=lambda n: {"p": 0, "f": 1}.get(n, 0)))) if len(resultats) > 1 else ""
    return {"resultat": f'<div class="pbody">{E.tete_resultats(C.esc(nom), dims)}<div class="et-blocs">{blocs}</div>'
                        f'{_avertissements(r)}{C.aide(NON_VERIFIES)}</div>{pied}',
            "volets": volets, "vues": PC.vues(r["sections"], r["predalle"], DP, "pd", dims, LEGENDES), "phrase": "", "statut_html": statut,
            "resume": resume_note(etat, resultats), "donnees": arbre(etat, r), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de la nappe de base
def description_candidat(c: Dict, axe: str = "") -> str:
    """La ligne de détail d'un candidat du sélecteur : ses fils ou ses barres, sa section au mètre, son axe (``axe``,
    un texte — vide quand le module ne le dit pas), la section et le centre de gravité de la face qu'il donne."""
    if c["type"] == "treillis":
        t = TR.fils(c["treillis"])
        q = f"treillis soudé · porteurs Ø{t[0]} tous les {t[2]} mm · répartition Ø{t[1]} tous les {t[3]} mm"
    else:
        q = f"barres Ø{c['d']} tous les {c['esp']} mm"
    milieu = f" · {axe}" if axe else ""
    return (f"{q} · {C.fr(c['As_pm'], 0)} mm²/m{milieu} · face : A<sub>s</sub> = {C.fr(c['As'], 0)} mm², "
            f"y<sub>G</sub> = {C.fr(c['yG'], 2)} cm")


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
        return description_candidat(c, f"axe {C.fr(c['e'], 1)} cm" + ("" if base["dist"] is not None else f" ({pose})"))
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


_MOI = sys.modules[__name__]   # ce module, que api_web/plaque_compact.py lit (ses formules mises en mots)
