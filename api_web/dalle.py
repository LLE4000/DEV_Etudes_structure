# -*- coding: utf-8 -*-
"""Le module « Dalles béton armé » du site : l'état d'une note (une ou plusieurs dalles, chacune avec ses sections, et
pour chaque section quatre nappes — inférieure et supérieure, directions X et Y — faites de couches de treillis, de
barres Ø/e ou de « n barres »), les champs de saisie, le résultat section par section, la grande fenêtre de calcul, le
sélecteur de la nappe de base et les dessins — rendus en fragments HTML.

L'écran est l'établi COMPACT, celui de la poutre (api_web/etabli_compact.py, api_web/plaque_compact.py) : à gauche
l'arborescence de la note (chaque dalle, et sous la dalle ouverte ses quatre rubriques), au centre les vues sous leur
en-tête technique, à droite un bloc par section. Les couches des données et des dessins portent la même clé.

Le calcul est celui du moteur ``beton/dalle`` (extrait de l'écran Streamlit modules/dalle.py v2.2, dont il reproduit
les résultats au bit près — tests/test_dalle_moteur.py). Ce fichier MET EN FORME : il ne calcule rien.

Ce que le site fait de plus que l'écran Streamlit, sans changer un chiffre :
- une saisie que l'écran ramènerait en silence à une autre valeur (moment négatif lu 0, espacement hors bornes…) est
  REFUSÉE avec un message, et la dalle reste intacte ;
- le sélecteur de la nappe de base (treillis de la base, barres courantes) trié par section au mètre, la retenue (✓) et
  la plus petite qui tient (★) marquées ;
- les avertissements du moteur (nappes X et Y calculées au même niveau, enrobage nul lu 3 cm, d utile ≤ 0) ;
- la reprise d'un fichier de l'écran Streamlit (« dalle_ba.json ») avec ce que l'écran y remplaçait en silence ;
- un verdict qui, dans une dalle non conforme, nomme une vérification NON conforme (gouvernante()) ;
- deux bornes contre l'absurde, que l'écran n'a pas (SOLLICITATION_MAX, EN.H_MAX pour les longueurs saisies).
"""
import copy
import json
import sys
from typing import Dict, List, Optional, Sequence

from beton import treillis as TR
from beton.dalle import entree as EN
from beton.dalle import fichier as FI
from beton.dalle import moteur as MO

from . import commun as C
from . import dessin_dalle as DD
from . import etabli_compact as E
from . import plaque_compact as PC

MODULE = "dalle"
PAGE = "dalle"

# des requêtes de plus pour le parcours (api_web/parcours.py) : les champs que le parcours automatique n'essaie pas
_V1 = {"dalles": [{"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "A"}]}],
       "values": {"dal1_sec1_M_inf": 28.0, "dal1_sec1_M_sup_raw": "-12,0", "dal1_sec1_V": 55.0, "dal1_sec1_ncouches_inf": 2,
                  "dal1_sec1_arm_type_inf_c2": "Barres", "dal1_sec1_ø_barres_inf_c2": 12, "gamma_s": 1.5}}
PARCOURS = [
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_y", "couches", "0", "dist"], "valeur": "5"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_y", "yG"], "valeur": "4,2"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "sup_x", "couches", "0", "dist"], "valeur": ""},
    {"action": "modifier", "chemin": ["sections", "0", "Mx_inf"], "valeur": "-20"},
    {"action": "modifier", "chemin": ["sections", "0", "faces", "inf_x", "couches", "0", "esp"], "valeur": "600"},
    {"action": "modifier", "chemin": ["fichier_streamlit"], "valeur": {"nom": "dalle_ba.json", "texte": json.dumps(_V1)}},
    {"action": "modifier", "chemin": ["fichier_streamlit"], "valeur": {"nom": "abime.json", "texte": "{pas du json"}},
    {"action": "selecteur", "role": "sup_x-0", "familles": ["treillis"]},
    {"action": "choisir", "role": "inf_y-0", "nom": "Ø12/150"},
]

TYPES = PC.TYPES
FACES_TXT = PC.FACES_TXT
MOTIFS = {"hauteur_insuffisante": "hauteur insuffisante", "d_utile_negatif": "d utile ≤ 0",
          "acier_insuffisant": "acier insuffisant", "acier_excessif": "acier excessif (A<sub>s</sub> &gt; A<sub>s,max</sub>)",
          "cisaillement_excessif": "τ &gt; τ<sub>adm,I</sub> (pas d'étriers dans une dalle)"}
LIB_FAMILLES = {"treillis": "Treillis", "barres": "Barres"}


class _Refus(ValueError):
    """Une saisie refusée : son message est montré tel quel."""


# ------------------------------------------------------------------ l'état de la note : des dalles
def _site(d: Dict, verrou: bool = False) -> Dict:
    """Une dalle du site : la dalle du moteur, normalisée, et son verrou (lecture seule, comme le 🔒 de l'écran)."""
    out = EN.normaliser(d)
    out["verrou"] = bool(verrou)
    return out


def _moteur(d: Dict) -> Dict:
    """La dalle telle que le moteur la lit (sans le verrou, propre au site)."""
    return {k: v for k, v in d.items() if k != "verrou"}


def nouvel_etat() -> Dict:
    return {"version": 1, "dalles": [_site(EN.nouvelle_dalle())], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran relu par le moteur ; une dalle illisible (fichier abîmé) lève une erreur claire. Une note
    sans dalle est une note neuve ; « dalles » qui n'est pas une liste est une note abîmée (refusée, jamais remplacée
    en silence par une note neuve)."""
    dalles = etat.get("dalles") if isinstance(etat, dict) else None
    if dalles is None or dalles == []:
        return nouvel_etat()
    if not isinstance(dalles, list):
        raise ValueError(f"note illisible : « dalles » doit être une liste (reçu {type(dalles).__name__})")
    out = {"version": 1, "dalles": []}
    for i, d in enumerate(dalles):
        if not isinstance(d, dict):
            raise ValueError(f"dalle n° {i + 1} illisible")
        v = d.get("verrou", False)
        if not isinstance(v, bool):
            raise ValueError(f"dalle n° {i + 1} : verrou vrai ou faux attendu")
        try:
            dd = _site(_moteur(d), v)
            _bornes_site(dd)
            for s in dd["sections"]:          # un nom vide (possible dans un .stab, pas à l'écran) : la première lettre libre
                if not s["nom"].strip():
                    s["nom"] = EN.nom_section_suivant(dd)
            out["dalles"].append(dd)
        except ValueError as e:
            raise ValueError(f"dalle « {d.get('nom', i + 1)} » : {e}") from None
    a = etat.get("actif", 0)
    a = a if isinstance(a, int) and not isinstance(a, bool) else 0
    out["actif"] = min(max(0, a), len(out["dalles"]) - 1)
    return out


def _actif(etat: Dict) -> Dict:
    return etat["dalles"][etat["actif"]]


def _nom_libre(etat: Dict) -> str:
    pris = {d["nom"] for d in etat["dalles"]}
    k = len(etat["dalles"]) + 1
    while f"Dalle {k}" in pris:
        k += 1
    return f"Dalle {k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les dalles de la note : en ouvrir une, en ajouter, en copier une (``index`` ; sans : l'ouverte), en supprimer une
    (jamais la dernière). Une dalle ajoutée prend le γs et le jeu de la note (paramètres communs, comme dans l'écran
    Streamlit)."""
    ds = etat["dalles"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(ds) - 1)
    elif action == "ajouter":
        d = EN.nouvelle_dalle()
        d["nom"] = _nom_libre(etat)
        d["gamma_s"], d["jeu_premier_lit"] = ds[0]["gamma_s"], ds[0]["jeu_premier_lit"]
        ds.append(_site(d))
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
# Deux bornes PROPRES AU SITE (l'écran Streamlit n'en a pas) : elles ne refusent que l'absurde, que l'écran calculait
# en chiffres démesurés (taux de 10¹⁵⁰ %, ou « inf ») :
# - une longueur (jeu premier lit, axe d'une couche, centre de gravité) au plus égale à l'épaisseur maximale d'une
#   dalle, EN.H_MAX ;
# - une sollicitation au plus égale à SOLLICITATION_MAX : la plus forte bande du domaine de l'écran (5 m × 100 cm,
#   C50/60) porte moins de 20 000 kN·m et 4 000 kN.
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
        raise _Refus(f"{nom} : {borne} (bornes de l'écran Streamlit), pas {C.nb(x, 6)}{unite}")
    if maxi is not None and x > maxi:
        if site:
            raise _Refus(f"{nom} : au plus {C.nb(maxi)}{unite}, l'épaisseur maximale d'une dalle (au-delà, la saisie n'a "
                         f"pas de sens), pas {C.fr(x, 1)}{unite}")
        raise _Refus(f"{nom} : entre {C.nb(mini)} et {C.nb(maxi)}{unite} (bornes de l'écran Streamlit), pas {C.nb(x, 6)}{unite}")
    if entier:
        if x != int(x):
            raise _Refus(f"{nom} : un nombre entier{unite}, pas {C.nb(x, 6)}")
        return int(x)
    return x


def _bornes_site(d: Dict) -> None:
    """Les bornes propres au site (voir SOLLICITATION_MAX) sur une dalle relue (fichier .stab) : ValueError claire."""
    if d["jeu_premier_lit"] > EN.H_MAX:
        raise ValueError(f"jeu premier lit de {C.fr(d['jeu_premier_lit'], 1)} cm : au plus {EN.H_MAX} cm")
    for s in d["sections"]:
        for k in EN.MOMENTS + ("V",):
            if s[k] > SOLLICITATION_MAX:
                raise ValueError(f"section « {s['nom']} » : {k} au-delà de {C.fr(SOLLICITATION_MAX, 0)} (valeur absurde)")
        for f in EN.FACES:
            face = s["faces"][f]
            if any(x is not None and x > EN.H_MAX for x in [face["yG"]] + [c["dist"] for c in face["couches"]]):
                raise ValueError(f"section « {s['nom']} », nappe {f} : axe ou centre de gravité au-delà de {EN.H_MAX} cm")


def _axe_auto(d: Dict, c: Dict) -> float:
    return MO.distance_auto(c, d["enrobage"], d["jeu_premier_lit"])


def _yG_calcule(d: Dict, face: Dict) -> float:
    return MO._face(face["couches"], None, d["b"], d["enrobage"], d["jeu_premier_lit"])["yG_calcule"]


def _auto_ou_impose(cle: str, valeur, actuel, auto: float):
    """Un axe ou un centre de gravité : vide → automatique ; la valeur automatique affichée, réécrite → reste
    automatique (comme le champ de l'écran Streamlit) ; un autre nombre > 0 → imposé."""
    if valeur is None or (isinstance(valeur, str) and valeur.strip().lower() in ("", "auto")):
        return None
    x = _nombre(cle, valeur)
    if actuel is None and abs(x - auto) <= 0.5e-3 + 1e-12:
        return None
    return x


def _ecrire(etat: Dict, d: Dict, chemin: List, valeur) -> None:
    """Écrit une saisie dans la dalle ``d`` (une copie) ; _Refus si la valeur sort du domaine de l'écran."""
    k = chemin[0]
    if len(chemin) == 1:
        if k == "nom":
            d["nom"] = str(valeur)
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
        elif k in ("b", "h", "enrobage"):
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
        if not str(valeur).strip():
            raise _Refus("Nom de section vide")
        s["nom"] = str(valeur)
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
            raise _Refus(f"Au plus {C.fr(SOLLICITATION_MAX, 0)} {u}, pas {C.fr(x, 0)} {u} : la plus forte bande de dalle "
                         "(5 m de large, 100 cm d'épaisseur) porte moins de 20 000 kN·m et 4 000 kN.")
        s[chemin[2]] = x
        return
    face = s["faces"][chemin[3]]
    if chemin[4] == "yG":
        face["yG"] = _auto_ou_impose("yG", valeur, face["yG"], _yG_calcule(d, face))
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
        c["dist"] = _auto_ou_impose("dist", valeur, c["dist"], _axe_auto(d, c))
    else:
        raise _Refus(f"champ inconnu : {champ}")


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la dalle ouverte. ``chemin`` : les clés (« sections », 0, « faces », « inf_y », « couches »,
    0, « type ») ; les boutons d'ajout, de copie et de retrait passent par le même chemin, terminé par « + », « copie »
    ou « - ». Une valeur hors du domaine de l'écran est refusée avec un message, et la dalle reste intacte."""
    chemin = [int(c) if isinstance(c, str) and c.isdigit() else c for c in chemin]
    if chemin == ["fichier_streamlit"]:
        return _ouvrir_streamlit(etat, valeur)
    a = etat["actif"]
    d = etat["dalles"][a]
    if chemin == ["verrou"]:
        lu = {True: True, False: False, "true": True, "false": False, "1": True, "0": False}.get(
            valeur.strip().lower() if isinstance(valeur, str) else valeur)
        if lu is None:
            return f"Verrou : coché ou décoché, pas « {valeur} »"
        d["verrou"] = lu
        return None
    if d["verrou"]:
        return "Dalle verrouillée : décocher « Verrouiller la dalle » pour la modifier"
    if chemin and chemin[0] in NOTE_ENTIERE and len(chemin) == 1:
        # γs et le jeu premier lit sont des paramètres de la NOTE (l'écran Streamlit les partage entre ses dalles)
        try:
            x = _nombre(chemin[0], valeur)
        except _Refus as e:
            return str(e)
        for k, dd in enumerate(etat["dalles"]):
            etat["dalles"][k] = _site(dict(_moteur(dd), **{chemin[0]: x}), dd["verrou"])
        return None
    travail = copy.deepcopy(d)
    try:
        _ecrire(etat, travail, chemin, valeur)
        etat["dalles"][a] = _site(_moteur(travail), d["verrou"])
    except _Refus as e:
        return str(e)
    except ValueError as e:
        return f"Saisie refusée : {e}"
    except (KeyError, IndexError, TypeError, AttributeError):
        return f"Saisie refusée : champ introuvable « {'/'.join(map(str, chemin))} »"
    return None


def _ouvrir_streamlit(etat: Dict, valeur) -> Optional[str]:
    """Un fichier de l'écran Streamlit (« 💾 Enregistrer » : dalle_ba.json) remplace les dalles de la note ; ce que
    l'écran remplaçait en silence (moment négatif lu 0, type inconnu…) est dit."""
    if not isinstance(valeur, dict) or not isinstance(valeur.get("texte"), str):
        return "Fichier illisible"
    nom = str(valeur.get("nom") or "fichier")
    try:
        lu = FI.lire(json.loads(valeur["texte"]))
    except json.JSONDecodeError:
        return f"« {nom} » n'est pas un fichier JSON lisible"
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        return f"« {nom} » refusé : {e}"
    etat["dalles"] = [_site(d) for d in lu["dalles"]]
    etat["actif"] = 0
    if lu["remarques"]:
        r = lu["remarques"]
        return (f"« {nom} » repris ({len(lu['dalles'])} dalle(s)). L'écran Streamlit y remplaçait en silence : "
                + " ; ".join(r[:4]) + (f" ; et {len(r) - 4} autre(s)" if len(r) > 4 else ""))
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur de la nappe de base : « inf_y-0 » (nappe inférieure Y de la section 0) reçoit « Treillis 8/8/150/150 »
    ou « Ø12/150 » ; ses renforts, son axe imposé et son centre de gravité imposé sont gardés."""
    face, i = role.rsplit("-", 1)
    d = _actif(etat)
    if d["verrou"]:
        raise ValueError("dalle verrouillée")
    cand = MO.lire_candidat(nom)
    travail = copy.deepcopy(d)
    travail["sections"][int(i)]["faces"][face]["couches"][0].update(cand)
    etat["dalles"][etat["actif"]] = _site(_moteur(travail), False)


# ------------------------------------------------------------------ les données : l'arborescence (colonne de gauche)
# la légende des dessins (dernière ligne de l'en-tête de chaque vue : ce qu'on y lit)
LEGENDES = {"c": "barres coupées en points, barres parallèles en traits", "p": "vue de dessus · barres X horizontales, Y verticales"}
AIDE_AXE = ("Du parement à l'axe des barres. Vide : l'axe automatique — enrobage + ½ Ø arrondi au 0,5 cm supérieur + jeu "
            "premier lit ; une valeur saisie l'impose.")


def _lecture(label: str, valeur: str, sym: str = "") -> str:
    return C.ligne(label, f'<span class="dl-ro">{valeur}</span>', sym)


def _formulaire_verrouille(d: Dict, r: Dict) -> str:
    """Une dalle verrouillée : ses données en lecture seule, et la case pour la déverrouiller."""
    s1 = (C.case(["verrou"], True, "Verrouiller la dalle (lecture seule)")
          + C.aide("La dalle est verrouillée : ses données se lisent, elles ne se modifient pas.")
          + _lecture("béton, acier", f"{C.esc(d['beton'])} · B{d['fyk']}")
          + _lecture("largeur de bande", f"{C.nb(d['b'])} cm", "b") + _lecture("épaisseur", f"{C.nb(d['h'])} cm", "h")
          + _lecture("enrobage", f"{C.nb(d['enrobage'])} cm", "c") + _lecture("direction principale", d["dir_principale"]))
    s2 = ""
    for s, rs in zip(d["sections"], r["sections"]):
        mom = " · ".join(f"{SOLLICITATIONS[k]} {C.nb(s[k])}" for k in EN.MOMENTS)
        s2 += f'<div class="grp">{C.esc(rs["titre"])}</div>' + _lecture("moments (kN·m)", mom) + _lecture("effort tranchant", f"{C.nb(s['V'])} kN", "V")
        for f in MO.ordre_faces(rs["principale"]):
            s2 += _lecture(PC.titre_face(f).lower() + (" · principale" if f.endswith(rs["principale"]) else ""),
                           C.esc(rs["faces"][f]["libelle"]))
    return (C.section(1, f"Dalle {C.icon('lock')}", "verrouillée", s1, True, "dl")
            + C.section(2, "Sections et armatures", f"{len(d['sections'])} section(s)", s2, False, "sl"))


def _options(d: Dict) -> str:
    L = C.ligne
    return (f'<div class="row full et-meth"><label><span>Méthode</span></label><b>Méthode du bureau</b><span class="hint">'
            f'{C.esc(MO.REFERENCE)} · ELS, z = 0,9·d</span></div>'
            + L("γ<sub>s</sub> acier", C.num(["gamma_s"], d["gamma_s"], "", mini=EN.GAMMA_S_MIN, maxi=EN.GAMMA_S_MAX), "",
                "f_yd = f_yk / γ_s — 1,5 par défaut (méthode ancienne) ; 1,15 donne 435 N/mm² en B500. Commun à toutes les "
                "dalles de la note.")
            + L("Jeu premier lit", C.num(["jeu_premier_lit"], d["jeu_premier_lit"], "cm", mini=0), "",
                "Ajouté à l'axe automatique de chaque couche. Commun à toutes les dalles de la note.")
            + C.aide("γ<sub>s</sub> et le jeu sont communs à toutes les dalles de la note, comme dans l'écran Streamlit.")
            + '<div class="et-gh"><span class="t">Reprendre une note Streamlit</span></div>'
            + C.aide("Le fichier « dalle_ba.json » de l'écran Streamlit (bouton Enregistrer) : ses dalles remplacent celles "
                     "de cette note ; ce que l'écran y remplaçait en silence est signalé.")
            + C.fichier(["fichier_streamlit"], "Ouvrir un fichier Streamlit", ".json"))


def formulaire(etat: Dict, r: Optional[Dict] = None) -> str:
    """Les quatre rubriques de la dalle ouverte, aux places de celles de la poutre : 1 Dalle, 2 Sections et
    sollicitations, 3 Armatures (de la section courante), 4 Options. ``r`` : son calcul (``MO.verifier``), refait s'il
    manque."""
    d = _actif(etat)
    r = r or _calculer(d)
    if d["verrou"]:
        return _formulaire_verrouille(d, r)
    L = C.ligne
    s1 = ('<div class="et-piece et-plat" data-cible="beton">'
          + L("Nom", C.texte(["nom"], d["nom"], "ex. Plancher haut"), full=True)
          + L("Béton", C.choix(["beton"], d["beton"], [(k, k) for k in MO.classes_beton()]), full=True)
          + L("Acier", C.segments(["fyk"], d["fyk"], [(n, f"B{n}") for n in EN.NUANCES]), full=True)
          + L("b bande", C.num(["b"], d["b"], "cm", entier=True, mini=EN.B_MIN, maxi=EN.B_MAX), "",
              "Largeur de la bande étudiée : 100 cm = un mètre courant. Moments et effort tranchant sont ceux de la bande.")
          + L("h épaisseur", C.num(["h"], d["h"], "cm", entier=True, mini=EN.H_MIN, maxi=EN.H_MAX))
          + L("c enrobage", C.num(["enrobage"], d["enrobage"], "cm", mini=0, maxi=EN.ENROBAGE_MAX), "",
              "Du parement au nu des armatures. 0 est lu 3 cm, comme dans l'écran Streamlit.")
          + L("Principale", C.segments(["dir_principale"], d["dir_principale"], [("Y", "Direction Y"), ("X", "Direction X")]), full=True)
          + C.aide("La direction principale ordonne les vérifications, les coupes et la note ; aucune formule ne la lit.")
          + "</div>" + C.case(["verrou"], False, "Verrouiller la dalle (lecture seule)"))
    contenus, resumes = [], []
    for i, (s, rs) in enumerate(zip(d["sections"], r["sections"])):
        verifs = {v["cle"]: v for v in rs["verifications"]}
        corps = ""
        for f in MO.ordre_faces(rs["principale"]):
            couches = "".join(PC.couche(EN, i, f, j, c, rs["faces"][f]["couches"][j], aide_axe=AIDE_AXE)
                              for j, c in enumerate(s["faces"][f]["couches"]))
            corps += PC.nappe(EN, i, f, s["faces"][f], rs["faces"][f], verifs[f], rs["principale"], couches,
                              "Un renfort : treillis, barres Ø/e ou n barres ; 3 Ø12 par défaut, posées dans le plan de la base")
        contenus.append(corps)
        resumes.append(PC.resume_armatures(rs))
    s3, r3 = PC.rubrique_armatures([s["nom"] for s in d["sections"]], contenus, resumes)
    return PC.formulaire("dl", "Dalle", f"{C.esc(d['beton'])} · B{d['fyk']} · c = {C.nb(d['enrobage'])} cm", s1,
                         PC.cartes(d["sections"]), len(d["sections"]), s3, r3,
                         f"γ<sub>s</sub> {C.nb(d['gamma_s'])} · jeu {C.nb(d['jeu_premier_lit'])} cm", _options(d))


def arbre(etat: Dict, r: Optional[Dict] = None) -> str:
    """La colonne de gauche : l'ARBORESCENCE de la note (api_web/etabli_compact.py, la même que la poutre). Chaque
    dalle, « nom · b × h cm » (un cadenas si elle est verrouillée), sans taux ni pastille ; la dalle ouverte montre ses
    quatre rubriques. Copier, supprimer, renommer : dans le menu « ⋯ » de la dalle."""
    elements = [(d.get("nom") or f"Dalle {i + 1}", f"· {C.nb(d['b'])} × {C.nb(d['h'])} cm", d["verrou"])
                for i, d in enumerate(etat["dalles"])]
    return E.arbre(E.DALLE, elements, etat["actif"], formulaire(etat, r), "dalle")


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
    dans une dalle NON CONFORME, la plus chargée de celles que le moteur déclare non conformes — acier excessif ou
    d utile ≤ 0 le sont sous 100 %, et le verdict ne doit pas nommer une vérification conforme. On CHOISIT parmi les
    vérifications du moteur ; rien n'est recalculé. À égalité, la première dans l'ordre de l'écran (comme le moteur)."""
    return max(((i, v) for i, s in enumerate(r["sections"]) for v in s["verifications"]), key=lambda x: _cle_gouv(x[1]))


def gouvernante_section(s: Dict) -> Dict:
    """La vérification qui gouverne une section, selon la même règle que gouvernante()."""
    return max(s["verifications"], key=_cle_gouv)


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
    return MO.verifier(_moteur(d))


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : sa dalle la plus chargée — une dalle non conforme d'abord (la règle de gouvernante()) ;
    « Données incomplètes » dès qu'une section n'a aucun effort (comme la poutre)."""
    gouv = [gouvernante(r) for r in resultats]
    k = max(range(len(resultats)), key=lambda j: _cle_gouv(gouv[j][1]))
    r, (i_s, v) = resultats[k], gouv[k]
    nivs = [PC.niveau_element(MO, _MOI, x) for x in resultats]
    niv = next((n for n in ("r", "n") if n in nivs), C.niveau(v["taux"]))
    out = {"taux": C.pct(v["taux"]) if niv != "n" else None, "niveau": niv, "statut": C.STATUTS[niv][1],
           "gouvernant": f"{etat['dalles'][k].get('nom') or 'Dalle'} · {r['sections'][i_s]['titre']} · {nom_minuscule(v)}"}
    if niv in C.STATUTS_COURTS:
        out["court"] = C.STATUTS_COURTS[niv]   # la pastille du téléphone
    return out


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les dalles de la note sont l'ARBORESCENCE des données (``arbre``), avec leurs rubriques."""
    return ""


def _optimaux(d: Dict, r: Dict) -> Dict:
    """La plus petite nappe de base qui tient, pour chaque face de chaque section (l'étoile des lignes du résultat)."""
    out = {}
    for i, s in enumerate(r["sections"]):
        for f in EN.FACES:
            cands = MO.candidats_base(_moteur(d), i, f)
            o = next((c for c in cands if c["optimal"]), None)
            if o is not None and not o["retenue"]:
                out[(i, f)] = o
    return out


def _avertissements(r: Dict, i: Optional[int] = None) -> str:
    """Les avertissements du moteur : ceux de la section ``i`` (posés dans son bloc), ou, sans ``i``, ceux de la dalle
    entière ; les nappes X et Y d'une même face calculées au même niveau sont dites en une phrase."""
    out = ""
    confondues: List[Dict] = []
    for a in r["avertissements"]:
        if a.get("section") != i:
            continue
        if a["code"] == "nappes_confondues":
            confondues.append(a)
        elif a["code"] == "enrobage_nul":
            out += C.message("Enrobage nul : comme l'écran Streamlit, le calcul pose les couches avec un enrobage de 3 cm.")
        else:
            out += C.message(C.esc(a["message"]), grave=a["code"] == "d_utile_negatif")
    if confondues:
        s = r["sections"][i]
        faces = " et ".join("inférieures" if a["face"] == "inf" else "supérieures" for a in confondues)
        axes = " ; ".join(f"{'inf.' if a['face'] == 'inf' else 'sup.'} {C.fr(a['e_x'], 1)} et {C.fr(a['e_y'], 1)} cm" for a in confondues)
        mini = max(a["ecart_mini"] for a in confondues)
        out += C.message(f"{C.esc(s['titre'])} : les nappes X et Y {faces} sont calculées au même niveau ({axes}), comme dans "
                         f"l'écran Streamlit. Croisées, l'une passe sur l'autre (au moins {C.fr(mini, 1)} cm plus loin du "
                         f"parement) : son d est plus petit. Saisir son axe pour en tenir compte.")
    return out


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
    d = _actif(etat)
    r = _calculer(d)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["dalles"])]
    nom = d.get("nom") or "Dalle"
    hyps = _hypotheses(d, r["materiaux"])
    opt = {} if d["verrou"] else _optimaux(d, r)
    gouv = gouvernante(r)
    blocs, volets = "", ""
    for i, s in enumerate(r["sections"]):
        h, v = PC.bloc(_MOI, i, s, nom, hyps, gouv, opt, d["verrou"], pied=_avertissements(r, i))
        blocs += h
        volets += v
    dims = f"{C.nb(d['b'])} × {C.nb(d['h'])} cm · {C.esc(d['beton'])} · B{d['fyk']}"
    pied = (f'<div class="pfoot"><div class="mass"><span>Méthode du bureau · moteur dalle</span>'
            f'<b>v{C.esc(MO.REFERENCE.split(" v")[-1])}</b></div><div class="legend">{C.legende()}</div></div>')
    # l'état de la NOTE (toutes ses dalles) dans l'en-tête du panneau : utile dès deux dalles ; seule, chaque section dit le sien
    nivs = [PC.niveau_element(MO, _MOI, x) for x in resultats]
    statut = C.statut(next((n for n in ("r", "n") if n in nivs), max(nivs, key=lambda n: {"p": 0, "f": 1}.get(n, 0)))) if len(resultats) > 1 else ""
    return {"resultat": f'<div class="pbody">{E.tete_resultats(C.esc(nom), dims)}<div class="et-blocs">{blocs}</div>'
                        f'{_avertissements(r)}</div>{pied}',
            "volets": volets, "vues": PC.vues(r["sections"], r["dalle"], DD, "dl", dims, LEGENDES), "phrase": "", "statut_html": statut,
            "resume": resume_note(etat, resultats), "donnees": arbre(etat, r), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de la nappe de base
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les nappes de base possibles d'une face (treillis de la base, barres Ø8 à Ø25 de 100 à 250 mm), triées par section
    au mètre, calculées par le moteur : la retenue (✓), la plus petite qui tient (★)."""
    face, i = role.rsplit("-", 1)
    i = int(i)
    d = _actif(etat)
    fams = tuple(f for f in (familles or MO.FAMILLES) if f in MO.FAMILLES) or MO.FAMILLES
    cands = MO.candidats_base(_moteur(d), i, face, fams)
    i_o = next((k for k, c in enumerate(cands) if c["optimal"]), 0)
    i_c = next((k for k, c in enumerate(cands) if c["retenue"]), i_o)
    base = d["sections"][i]["faces"][face]["couches"][0]

    def prop(c):
        if c["type"] == "treillis":
            t = TR.fils(c["treillis"])
            q = f"treillis soudé · porteurs Ø{t[0]} tous les {t[2]} mm · répartition Ø{t[1]} tous les {t[3]} mm"
        else:
            q = f"barres Ø{c['d']} tous les {c['esp']} mm"
        return (f"{q} · {C.fr(c['As_pm'], 0)} mm²/m · face : A<sub>s</sub> = {C.fr(c['As'], 0)} mm², "
                f"y<sub>G</sub> = {C.fr(c['yG'], 2)} cm")
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
    dk = face.split("_")[1]
    princ = " (principale)" if dk == d["dir_principale"].lower() else ""
    return C.selecteur(role, f"Nappe {FACES_TXT[face.split('_')[0]]} {dk.upper()} · {C.esc(EN.libelle_section(s['nom']))}",
                       f"couche de base{princ} · triée par section au mètre · un clic la retient{note}", filtres, rows,
                       max(0, min(i_o, i_c) - 4), "mm²/m", etoile="la plus petite section qui tient")


_MOI = sys.modules[__name__]   # ce module, que api_web/plaque_compact.py lit (ses formules mises en mots)
