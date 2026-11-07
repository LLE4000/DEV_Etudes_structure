# -*- coding: utf-8 -*-
"""Le module « Taux d'armature » du site : l'état d'une note (une ou plusieurs poutres), les champs de saisie, le
quantitatif (volume de béton, masse d'acier majorée, taux en kg/m³), la fenêtre de calcul, le sélecteur de diamètre et
les dessins (coupe, élévation, tableau récapitulatif) — rendus en fragments HTML.

Le calcul est celui du moteur ``beton/taux`` (extrait de l'écran Streamlit modules/taux_armature.py, dont il reproduit
les chiffres au bit près — tests/test_taux_moteur.py) ; les effets de bord de la saisie sont ceux de
``beton/taux/saisie.py``. Ce fichier MET EN FORME : il ne calcule rien.

Le module est un QUANTITATIF, sans vérification : le statut de la note est « Consultation » ; les barres du résultat
disent la PART de chaque famille dans la masse d'acier (gris : ce n'est pas un taux de travail).
"""
import copy
import math
import re
from typing import Dict, List, Optional, Sequence

from beton.taux import entree as EN
from beton.taux import moteur as MO
from beton.taux import saisie as SA

from . import commun as C
from . import dessin_taux as DT

MODULE = "taux"
PAGE = "taux-armature"

# une poutre complète, pour que le parcours de l'écran passe aussi par le sélecteur, les mètres, les talons
_EXEMPLE = {"nom": "P1", "forme": "talon_gauche", "L": 6.0, "b": 30.0, "h": 60.0, "enrobage": 3.0,
            "talon_gauche": {"b": 25.0, "h": 20.0},
            "familles": [{"role": "principal", "position": "ame_inf", "n": 3, "d": 20},
                         {"role": "renfort", "position": "ame_sup", "n": 2, "d": 12, "de": 0.0, "a": 30.0},
                         {"role": "renfort", "position": "talon_g_inf", "n": 3, "d": 16},
                         {"role": "lateral", "position": "lat_2", "n": 2, "d": 10, "retours": 2, "h_retour": 24.0}],
            "types_etriers": [{"nom": "E1", "d": 8.0}],
            "zones": [{"nom": "Z1", "de": 0.0, "a": 25.0, "s": 10.0, "type": "E1"},
                      {"nom": "Z2", "de": 25.0, "a": 75.0, "s": 20.0, "type": "E1"}]}
_ETAT_EXEMPLE = {"version": 1, "poutres": [_EXEMPLE], "actif": 0}
PARCOURS = [
    {"action": "calculer", "etat": _ETAT_EXEMPLE},
    {"action": "selecteur", "etat": _ETAT_EXEMPLE, "role": "d-0"},
    {"action": "choisir", "etat": _ETAT_EXEMPLE, "role": "d-0", "nom": "3Ø25"},
    {"action": "modifier", "etat": _ETAT_EXEMPLE, "chemin": ["longueurs"], "valeur": "metre"},
    {"action": "modifier", "etat": dict(_ETAT_EXEMPLE, poutres=[dict(_EXEMPLE, longueurs="metre")]), "chemin": ["familles", "1", "a_m"], "valeur": "2,5"},
    {"action": "modifier", "etat": dict(_ETAT_EXEMPLE, poutres=[dict(_EXEMPLE, longueurs="metre")]), "chemin": ["L"], "valeur": "4"},
    {"action": "modifier", "etat": _ETAT_EXEMPLE, "chemin": ["forme"], "valeur": "rectangulaire"},
    {"action": "modifier", "etat": _ETAT_EXEMPLE, "chemin": ["forme"], "valeur": "deux_talons"},
    {"action": "modifier", "etat": _ETAT_EXEMPLE, "chemin": ["types_etriers", "0", "-"], "valeur": "1"},
    {"action": "modifier", "etat": _ETAT_EXEMPLE, "chemin": ["types_etriers", "0", "nom"], "valeur": "Cadre"},
    {"action": "modifier", "etat": _ETAT_EXEMPLE, "chemin": ["familles", "+"], "valeur": "talon"},
    {"action": "modifier", "etat": _ETAT_EXEMPLE, "chemin": ["familles", "3", "retours"], "valeur": "0"},
    {"action": "ajouter", "etat": _ETAT_EXEMPLE},
]

LONGUEURS = {"pourcent": "% de L", "metre": "mètres"}
SORTES = {"principal": "un lit principal", "renfort": "un renfort local", "lateral": "des barres latérales", "talon": "un renfort de talon"}
ROLES_COURTS = {"principal": "lit principal", "renfort": "renfort", "lateral": "latérales"}
GROUPES_ROLES = (("principal", "Lits principaux"), ("renfort", "Renforts (y compris talons)"), ("lateral", "Barres latérales"))
DIAMETRES_ETRIERS = (6, 8, 10, 12, 14, 16)


# ------------------------------------------------------------------ l'état de la note : des poutres
def nouvel_etat() -> Dict:
    """La note neuve : une poutre, celle de l'écran Streamlit au premier affichage (30 × 60 cm, 6 m, aucune armature)."""
    p = EN.nouvelle_poutre()
    p["nom"] = "P1"
    return {"version": 1, "poutres": [p], "actif": 0}


def _relire(p, i: int) -> Dict:
    """Une poutre relue par le moteur ; une famille que le module ne sait pas calculer (talon d'une section « deux
    talons » : l'écran Streamlit s'arrête) est refusée avec le message du module."""
    q = EN.normaliser(p)
    try:
        MO.calculer(q)
    except ValueError as e:
        raise ValueError(f"poutre {q.get('nom') or i + 1} : {e} (l'écran Streamlit s'arrête sur cette erreur)") from None
    return q


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; une poutre illisible lève une erreur claire."""
    poutres = etat.get("poutres") if isinstance(etat, dict) else None
    if not poutres:
        return nouvel_etat()
    if not isinstance(poutres, list):
        raise ValueError("poutres : une liste est attendue")
    out = {"version": 1, "poutres": [_relire(p, i) for i, p in enumerate(poutres)]}
    actif = etat.get("actif", 0) or 0
    if isinstance(actif, str) and actif.strip().lstrip("-").isdigit():
        actif = int(actif)
    if isinstance(actif, bool) or not isinstance(actif, (int, float)) or not math.isfinite(actif):
        raise ValueError(f"actif : le rang de la poutre ouverte est un nombre, reçu {actif!r}")
    out["actif"] = min(max(0, int(actif)), len(out["poutres"]) - 1)
    return out


def _poutre(etat: Dict) -> Dict:
    return etat["poutres"][etat["actif"]]


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les poutres de la note : en ouvrir une, en ajouter, copier l'ouverte, en supprimer une (jamais la dernière)."""
    ps = etat["poutres"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(ps) - 1)
    elif action == "ajouter":
        p = EN.nouvelle_poutre()
        p["nom"] = f"P{len(ps) + 1}"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "dupliquer":
        p = copy.deepcopy(ps[etat["actif"]])
        p["nom"] = f"{p.get('nom') or 'Poutre'} (copie)"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "supprimer" and len(ps) > 1:
        i = min(max(0, int(index or 0)), len(ps) - 1)
        del ps[i]
        if etat["actif"] > i or etat["actif"] >= len(ps):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
TEXTES = {"nom", "forme", "longueurs", "acier", "position", "type"}
ENTIERS = {"n", "d", "retours"}


def _aller(d, chemin: Sequence):
    for c in chemin:
        d = d[c]
    return d


def _affiche(p: Dict, chemin: List) -> Optional[float]:
    """Le nombre que le champ ``chemin`` AFFICHE (un champ en mètres montre L × % / 100)."""
    k = chemin[-1]
    if k in ("de_m", "a_m"):
        f = _aller(p, chemin[:-1])
        return p["L"] * f[k[:-2]] / 100.0
    v = _aller(p, chemin[:-1]).get(k) if isinstance(_aller(p, chemin[:-1]), dict) else None
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _codes(p: Dict) -> List[str]:
    return MO.codes(p["familles"])


def _appliquer(p: Dict, chemin: List, valeur) -> Optional[str]:
    """Une saisie écrite dans la poutre, avec les effets de bord de l'écran Streamlit (beton/taux/saisie.py) ; rend le
    message à montrer quand l'écran aurait changé autre chose en silence."""
    k = chemin[-1]
    if k == "+":
        liste = chemin[0]
        if liste == "familles":
            if valeur not in SORTES:
                raise ValueError(f"famille : sorte {valeur!r} inconnue")
            if valeur == "talon" and p["forme"] == "deux_talons":
                raise ValueError("« deux talons » : le calcul de l'écran Streamlit ne connaît pas les talons de cette section "
                                 "(il s'arrête sur « Talon gauche non défini. ») — renfort de talon impossible")
            SA.ajouter_famille(p, valeur)
        elif liste == "types_etriers":
            SA.ajouter_type(p)
        elif liste == "zones":
            SA.ajouter_etrier(p)
        else:
            raise KeyError(liste)
        return None
    if k == "-":
        liste, i = chemin[0], int(chemin[1])
        if liste == "types_etriers":
            nom = p["types_etriers"][i]["nom"]
            touchees = SA.supprimer_type(p, i)
            if touchees and p["types_etriers"]:
                return (f"Type {nom} supprimé : {', '.join(touchees)} passe{'nt' if len(touchees) > 1 else ''} sur "
                        f"{p['types_etriers'][0]['nom']} (le premier type restant, comme l'écran Streamlit)")
            if touchees:
                return f"Type {nom} supprimé : {', '.join(touchees)} n'{'ont' if len(touchees) > 1 else 'a'} plus de type et ne compte{'nt' if len(touchees) > 1 else ''} plus"
            return None
        if liste not in ("familles", "zones"):
            raise KeyError(liste)
        del p[liste][i]
        return None
    if k in TEXTES:
        v = str(valeur if valeur is not None else "")
        if chemin == ["forme"]:
            avant = _codes(p)
            if v == "deux_talons":
                sur = [avant[i] for i, f in enumerate(p["familles"]) if f["position"].startswith("talon")]
                if sur:
                    raise ValueError(f"section à deux talons impossible : {', '.join(sur)} {'sont posées' if len(sur) > 1 else 'est posée'} sur un "
                                     "talon, que le calcul de l'écran Streamlit ne connaît pas pour cette forme (il s'arrête sur "
                                     "« Talon gauche non défini. ») — déplacer d'abord ces barres")
            deplacees = SA.changer_forme(p, v)
            if deplacees:
                noms = ", ".join(avant[i] for i in deplacees)
                un = len(deplacees) == 1
                return (f"{noms} : {'son' if un else 'leur'} talon n'existe plus, position ramenée sur « {EN.POSITIONS['ame_inf'][1]} » "
                        f"(la première du menu, comme l'écran Streamlit)")
            return None
        if chemin[0] == "types_etriers" and k == "nom":
            SA.renommer_type(p, int(chemin[1]), v)
            return None
        _aller(p, chemin[:-1])[k] = v
        return None
    # un nombre : une saisie identique à ce que le champ affiche ne change rien (le champ arrondit à 3 décimales)
    x = C.lire_nombre(valeur)
    if x is None:
        raise ValueError(f"nombre illisible : « {valeur} »")
    entier = chemin[0] == "familles" and k in ENTIERS
    actuel = _affiche(p, chemin)
    if actuel is not None and (int(round(x)) == actuel if entier else C.nb(x) == C.nb(actuel)):
        return None
    if k in ("de_m", "a_m"):
        SA.longueur_en_metres(p, _aller(p, chemin[:-1]), k[:-2], x)
        return None
    if chemin == ["L"]:
        hors = []
        if p["longueurs"] == "metre":
            cds = _codes(p)
            hors = [cds[i] for i, f in enumerate(p["familles"]) if any(p["L"] * f[c] / 100.0 > x for c in ("de", "a"))]
        SA.changer_portee(p, x)
        if hors:
            return (f"{', '.join(hors)} : une longueur en mètres dépasse la nouvelle portée ; ces barres gardent leur "
                    "pourcentage de L (comme l'écran Streamlit)")
        return None
    if k == "retours":
        if x != int(x):
            raise ValueError(f"retours : {C.nb(x)} — 0, 1 ou 2")
        SA.changer_retours(_aller(p, chemin[:-1]), int(x))
        return None
    d = _aller(p, chemin[:-1])
    if k not in d:
        raise KeyError(k)
    if entier:
        if x != int(x):
            raise ValueError(f"{k} : nombre entier attendu, reçu {C.nb(x)}")
        d[k] = int(x)
    else:
        d[k] = x
    return None


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la poutre ouverte. ``chemin`` : les clés (« familles », 0, « n ») ; les boutons d'ajout et
    de retrait passent par le même chemin, terminé par « + » (la valeur dit quoi ajouter) ou « - ». Le moteur relit la
    poutre : une valeur hors du domaine de l'écran est REFUSÉE avec un message, et la poutre reste intacte."""
    p = _poutre(etat)
    avant = copy.deepcopy(p)
    chemin = [int(c) if isinstance(c, str) and c.isdigit() else c for c in chemin]
    try:
        msg = _appliquer(p, chemin, valeur)
        etat["poutres"][etat["actif"]] = _relire(p, etat["actif"])
        return msg
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        etat["poutres"][etat["actif"]] = avant
        texte = str(e) if isinstance(e, ValueError) else f"champ inconnu ({e})"
        return f"Saisie refusée : {texte}"


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur de diamètre : « d-2 » (la famille 2) reçoit le Ø de la ligne choisie (« 3Ø20 »)."""
    i = _index_role(etat, role)
    m = re.search(r"Ø(\d+)$", str(nom))
    if not m:
        raise ValueError(f"diamètre illisible : « {nom} »")
    p = _poutre(etat)
    p["familles"][i]["d"] = int(m.group(1))
    etat["poutres"][etat["actif"]] = _relire(p, etat["actif"])


def _index_role(etat: Dict, role: str) -> int:
    m = re.fullmatch(r"d-(\d+)", str(role))
    if not m or int(m.group(1)) >= len(_poutre(etat)["familles"]):
        raise ValueError(f"sélecteur inconnu : « {role} »")
    return int(m.group(1))


# ------------------------------------------------------------------ le cadre Données
def _attr(chemin: Sequence) -> str:
    return C._attr_chemin(chemin)


def _moins(chemin: List, titre: str) -> str:
    return f'<button type="button" class="moins" data-chemin="{_attr(chemin + ["-"])}" data-valeur="1" title="{C.esc(titre)}">{C.icon("x")}</button>'


def _ajout(chemin: List, valeur: str, texte: str) -> str:
    return f'<button type="button" class="addrow" data-chemin="{_attr(chemin)}" data-valeur="{C.esc(valeur)}">{C.icon("plus")}{texte}</button>'


def _pastille(couleur: str, groupe: str) -> str:
    return f'<i class="tx-pastille" style="background:{couleur};box-shadow:inset 0 0 0 1.5px {DT.BORDS[groupe]}"></i>'


def _carte_famille(p: Dict, i: int, rf: Optional[Dict]) -> str:
    f = p["familles"][i]
    metre = p["longueurs"] == "metre"
    opts = [o for o in EN.options_positions(p["forme"], f["role"]) if not (p["forme"] == "deux_talons" and o.startswith("talon"))]
    if f["position"] not in opts:                                # une position que le menu n'offre plus (garde-fou)
        opts = [f["position"]] + opts
    code = rf["code"] if rf else ""
    masse = f'<span class="muted">{C.fr(rf["masse_majoree"], 1)} kg</span>' if rf else ""
    tete = (f'<div class="sh2">{_pastille(rf["couleur"], rf["groupe"]) if rf else ""}<span class="indice">{C.esc(code)}</span>'
            f'{C.esc(EN.ROLES[f["role"]][1])}<span class="tx-sp"></span>{masse}{_moins(["familles", i], "Retirer cette famille")}</div>')
    pos = C.ligne("Position", C.choix(["familles", i, "position"], f["position"], [(o, EN.POSITIONS[o][1]) for o in opts]), full=True)
    trio1 = ('<div class="trio">'
             f'<label><span>n barres</span>{C.num(["familles", i, "n"], f["n"], "", entier=True, mini=1, maxi=20)}</label>'
             f'<label><span>Ø</span>{C.choix(["familles", i, "d"], f["d"], [(d, f"Ø{d}") for d in EN.DIAMETRES])}</label>'
             f'<label><span>retours</span>{C.choix(["familles", i, "retours"], f["retours"], [(r, "aucun" if r == 0 else str(r)) for r in EN.RETOURS])}</label></div>')
    if metre:
        de = C.num(["familles", i, "de_m"], p["L"] * f["de"] / 100.0, "m", mini=0, maxi=p["L"])
        a = C.num(["familles", i, "a_m"], p["L"] * f["a"] / 100.0, "m", mini=0, maxi=p["L"])
    else:
        de = C.num(["familles", i, "de"], f["de"], "%", mini=0, maxi=100)
        a = C.num(["familles", i, "a"], f["a"], "%", mini=0, maxi=100)
    trio2 = ('<div class="trio">'
             f'<label><span>h retour</span>{C.num(["familles", i, "h_retour"], f["h_retour"], "cm", desactive=not f["retours"], mini=0, maxi=1000)}</label>'
             f'<label><span>de</span>{de}</label><label><span>à</span>{a}</label></div>')
    return f'<div class="secard tx-carte">{tete}{pos}{trio1}{trio2}</div>'


def formulaire(etat: Dict, res: Optional[Dict] = None) -> str:
    """Le cadre Données : les sections repliables, chaque champ lié à un chemin de la poutre ouverte ; ``res`` (le calcul
    de cette poutre) donne les codes, couleurs et bilans affichés dans les cartes."""
    p = _poutre(etat)
    res = res if res is not None else MO.calculer(p)
    L = C.ligne
    forme = p["forme"]
    lib = EN.FORMES[forme]
    # 1 · la poutre
    s1 = (L("Nom", C.texte(["nom"], p.get("nom"), "ex. P1"), full=True)
          + L("Section", C.choix(["forme"], forme, list(EN.FORMES.items())), full=True)
          + L("portée", C.num(["L"], p["L"], "m", mini=0.5, maxi=60), "L")
          + L("largeur", C.num(["b"], p["b"], "cm", mini=5, maxi=200), "b")
          + L("hauteur", C.num(["h"], p["h"], "cm", mini=10, maxi=300), "h")
          + L("enrobage", C.num(["enrobage"], p["enrobage"], "cm", mini=1, maxi=10), "c",
              "L'axe des barres est à c + Ø/2 du parement ; les étriers sont comptés sur le contour de l'âme à c."))
    for cote, titre, ok in (("talon_gauche", "Talon gauche", EN.a_talon_gauche(forme)), ("talon_droit", "Talon droit", EN.a_talon_droit(forme))):
        if ok:
            t = p[cote]
            s1 += (f'<div class="grp">{titre}</div>'
                   + L("largeur", C.num([cote, "b"], t["b"], "cm", maxi=300), "b<sub>t</sub>", "Mesurée depuis le nu de l'âme, vers l'extérieur.")
                   + L("hauteur", C.num([cote, "h"], t["h"], "cm", maxi=300), "h<sub>t</sub>", "Depuis la fibre inférieure de la poutre."))
    if forme == "deux_talons":
        s1 += C.message("Deux talons : le calcul de l'écran Streamlit ne les compte pas — volume, poids et dessin sont ceux de "
                        "l'âme seule (défaut du module, reproduit ici).", grave=True)
    # 2 · les armatures longitudinales
    rf = {f["index"]: f for f in res["familles"]} if res else {}
    s2 = L("Longueurs", C.segments(["longueurs"], p["longueurs"], list(LONGUEURS.items())), full=True,
           aide="« De » et « À » se saisissent en % de la portée ou en mètres ; le calcul garde le pourcentage.")
    for role, titre in GROUPES_ROLES:
        idx = [i for i, f in enumerate(p["familles"]) if f["role"] == role]
        if idx:
            s2 += f'<div class="grp">{titre}</div><div class="secards">' + "".join(_carte_famille(p, i, rf.get(i)) for i in idx) + "</div>"
    if not p["familles"]:
        s2 += C.aide("Aucune barre longitudinale : en ajouter ci-dessous.")
    s2 += '<div class="tx-ajouts">' + _ajout(["familles", "+"], "principal", "Lit principal") + _ajout(["familles", "+"], "renfort", "Renfort local")
    s2 += _ajout(["familles", "+"], "lateral", "Barres latérales")
    if EN.renfort_talon_possible(forme) and forme != "deux_talons":
        s2 += _ajout(["familles", "+"], "talon", "Renfort de talon")
    s2 += "</div>"
    # 3 · les étriers
    s3 = '<div class="grp">Types d\'étriers</div>'
    for i, t in enumerate(p["types_etriers"]):
        opts = [(d, f"Ø{d}") for d in DIAMETRES_ETRIERS]
        if t["d"] not in DIAMETRES_ETRIERS:
            opts = [(t["d"], f"Ø{C.nb(t['d'])}")] + opts
        s3 += (f'<div class="lit tx-type"><span>Type</span>{C.texte(["types_etriers", i, "nom"], t["nom"], "E1")}'
               f'{C.choix(["types_etriers", i, "d"], t["d"], opts)}{_moins(["types_etriers", i], "Retirer ce type")}</div>')
    if not p["types_etriers"]:
        s3 += C.aide("Aucun type : « Zone d'étriers » en crée un (E1, Ø8), comme le bouton « + Étrier » de l'écran Streamlit.")
    s3 += '<div class="grp">Zones</div>'
    noms = [t["nom"] for t in p["types_etriers"]]
    re_ = {e["index"]: e for e in res["etriers"]} if res else {}
    ign = {z["index"]: z for z in res["zones_ignorees"]} if res else {}
    cartes = ""
    for i, z in enumerate(p["zones"]):
        typ = (C.choix(["zones", i, "type"], z["type"], [(n, n) for n in noms]) if noms
               else '<div class="field locked tx-sans">sans type</div>')
        e = re_.get(i)
        bilan = (f'{e["n"]} × Ø{C.nb(e["d"])} · {C.fr(e["masse_majoree"], 1)} kg' if e
                 else ("non comptée : sans type" if ign.get(i, {}).get("motif") == "sans_type" else "non comptée : « À » ≤ « De »"))
        cartes += (f'<div class="secard tx-carte"><div class="lit tx-zone"><span>Zone</span>{C.texte(["zones", i, "nom"], z["nom"], "Z1")}{typ}'
                   f'{_moins(["zones", i], "Retirer cette zone")}</div>'
                   '<div class="trio">'
                   f'<label><span>de</span>{C.num(["zones", i, "de"], z["de"], "%", mini=0, maxi=100)}</label>'
                   f'<label><span>à</span>{C.num(["zones", i, "a"], z["a"], "%", mini=0, maxi=100)}</label>'
                   f'<label><span>espacement s</span>{C.num(["zones", i, "s"], z["s"], "cm", mini=2, maxi=100)}</label></div>'
                   f'<div class="tx-bilan{"" if e else " ko"}">{bilan}</div></div>')
    if cartes:
        s3 += f'<div class="secards">{cartes}</div>'
    else:
        s3 += C.aide("Aucune zone d'étriers.")
    if p["zones"] and not noms:
        s3 += C.message("Aucun type d'étrier : les zones ne sont pas comptées. Ajouter un type.", grave=True)
    s3 += ('<div class="tx-ajouts">' + _ajout(["zones", "+"], "1", "Zone d'étriers") + _ajout(["types_etriers", "+"], "1", "Type d'étrier") + "</div>"
           + C.aide("Les zones se saisissent en % de la portée, comme dans l'écran Streamlit."))
    # 4 · le quantitatif
    s4 = (L("Acier", C.segments(["acier"], p["acier"], [(a, a) for a in EN.ACIERS]), full=True)
          + C.aide("La qualité d'acier n'entre dans aucun calcul (comme l'écran Streamlit) : elle est écrite sur la note.")
          + L("majoration", C.num(["majoration"], p["majoration"], "%", mini=0, maxi=50), "",
              "Ajoutée à toute la masse d'acier (chutes, recouvrements, façonnage) ; 5 % par défaut, comme l'écran."))
    nf, nz = len(p["familles"]), len(p["zones"])
    resume2 = " · ".join(f["libelle"] for f in res["familles"]) if res and res["familles"] else "aucune"
    resume3 = f"{nz} zone{'s' if nz > 1 else ''} · {len(noms)} type{'s' if len(noms) > 1 else ''}" if nz or noms else "aucun"
    return (C.section(1, f"Poutre {C.esc(p.get('nom') or '')}", f"{C.nb(p['b'])} × {C.nb(p['h'])} · L {C.nb(p['L'])} m · {C.esc(lib)}", s1, True, "pt")
            + C.section(2, "Armatures longitudinales", f"{nf} famille{'s' if nf > 1 else ''} · {C.esc(resume2)}", s2, True, "ar")
            + C.section(3, "Étriers", resume3, s3, True, "et")
            + C.section(4, "Quantitatif", f"{p['acier']} · majoration {C.nb(p['majoration'])} %", s4, False, "qt"))


# ------------------------------------------------------------------ le résultat
def _v(x) -> str:
    """Une DONNÉE d'une formule écrite sans la dénaturer (6 reste 6, 52,5 reste 52,5, 0,394584 garde ses chiffres),
    les milliers séparés par une espace fine (1 800)."""
    a = abs(x)
    s = C.nb(round(x, 2), 2) if a >= 100 else (C.nb(round(x, 3), 3) if a >= 1 else C.nb(round(x, 6), 6))
    ent, virg, dec = s.partition(",")
    if len(ent.lstrip("-")) > 3:
        ent = f"{int(ent):,}".replace(",", C.NBSP)
    return ent + virg + dec


def _net(x: float, nd: int) -> str:
    """Un résultat sans zéro inutile après la virgule (1 800 cm², 52,5 cm²)."""
    s = C.fr(x, nd)
    return s.rstrip("0").rstrip(",") if "," in s else s


def _quotient(q: float) -> str:
    """L/s tel que le module le tronque : écrit en entier quand il « tombe juste » à l'écran mais pas en machine."""
    if abs(q - round(q)) < 1e-9 and q != round(q):
        return repr(q).replace(".", ",")
    return _v(q)


def _fx_famille(f: Dict, maj: float, Vc: float, L: float) -> List[Dict]:
    out = []
    if f["L_active"] > 0:
        out.append({"parts": ["L_active", "L · (À − De) / 100", f"{_v(L)} × ({_v(f['a'])} − {_v(f['de'])}) / 100"], "res": f"{C.fr(f['L_active'], 3)} m"})
    else:
        out.append({"parts": ["L_active", "0 (À ≤ De)"], "res": "0 m"})
    if f["position"] == "lat_2":
        out.append({"parts": ["n_eff", "2 · n (les deux côtés)", f"2 × {f['n']}"], "res": f"{f['n_eff']} barres"})
    if f["L_active"] <= 0:
        out.append({"parts": ["L_tot", "0 (longueur active nulle)"], "res": "0 m"})
    elif f["retours"]:
        out.append({"parts": ["L_tot", "n_eff · L_active + r · h_r / 100 · n_eff",
                              f"{f['n_eff']} × {_v(f['L_active'])} + {f['retours']} × {_v(f['h_retour'])} / 100 × {f['n_eff']}"],
                    "res": f"{C.fr(f['L_tot'], 3)} m"})
    else:
        out.append({"parts": ["L_tot", "n_eff · L_active", f"{f['n_eff']} × {_v(f['L_active'])}"], "res": f"{C.fr(f['L_tot'], 3)} m"})
    out.append({"parts": ["m_lin", "π · (Ø / 2 000)² · 7 850", f"π × ({f['d']} / 2 000)² × 7 850"], "res": f"{C.fr(f['m_lin'], 3)} kg/m"})
    out.append({"parts": ["M", "L_tot · m_lin · (1 + maj / 100)", f"{_v(f['L_tot'])} × {_v(f['m_lin'])} × (1 + {_v(maj)} / 100)"],
                "res": f"{C.fr(f['masse_majoree'], 2)} kg"})
    out.append({"parts": ["Apport", "M / V_c", f"{_v(f['masse_majoree'])} / {_v(Vc)}"], "res": f"{C.fr(f['apport'], 1)} kg/m³"})
    return out


def _fx_zone(e: Dict, g: Dict, maj: float, Vc: float, L: float) -> List[Dict]:
    return [
        {"parts": ["L_zone", "L · (À − De) / 100", f"{_v(L)} × ({_v(e['a'])} − {_v(e['de'])}) / 100"], "res": f"{C.fr(e['L_zone'], 3)} m"},
        {"parts": ["n", "⌊L_zone / s⌋ + 1", f"⌊{_v(e['L_zone'])} / {_v(e['s_m'])}⌋ + 1 = ⌊{_quotient(e['quotient'])}⌋ + 1"], "res": f"{e['n']} étriers"},
        {"parts": ["L_étrier", "(2 · ((b − 2c) + (h − 2c)) · 10 + 2 · 8 · Ø) / 1 000",
                   f"(2 × (({_v(g['b'])} − 2 × {_v(g['c'])}) + ({_v(g['h'])} − 2 × {_v(g['c'])})) × 10 + 2 × 8 × {_v(e['d'])}) / 1 000"],
         "res": f"{C.fr(e['L_un'], 3)} m"},
        {"parts": ["L_tot", "n · L_étrier", f"{e['n']} × {_v(e['L_un'])}"], "res": f"{C.fr(e['L_tot'], 3)} m"},
        {"parts": ["m_lin", "π · (Ø / 2 000)² · 7 850", f"π × ({_v(e['d'])} / 2 000)² × 7 850"], "res": f"{C.fr(e['m_lin'], 3)} kg/m"},
        {"parts": ["M", "L_tot · m_lin · (1 + maj / 100)", f"{_v(e['L_tot'])} × {_v(e['m_lin'])} × (1 + {_v(maj)} / 100)"],
         "res": f"{C.fr(e['masse_majoree'], 2)} kg"},
        {"parts": ["Apport", "M / V_c", f"{_v(e['masse_majoree'])} / {_v(Vc)}"], "res": f"{C.fr(e['apport'], 1)} kg/m³"},
    ]


def _fx_poutre(r: Dict) -> List[Dict]:
    g, q, p = r["geometrie"], r["quantites"], r["poutre"]
    maj = p["majoration"]
    expr, vals = ["b · h"], [f"{_v(g['b'])} × {_v(g['h'])}"]
    if g["bg"] > 0:
        expr.append("b_t,g · h_t,g")
        vals.append(f"{_v(g['bg'])} × {_v(g['hg'])}")
    if g["bd"] > 0:
        expr.append("b_t,d · h_t,d")
        vals.append(f"{_v(g['bd'])} × {_v(g['hd'])}")
    return [
        {"parts": ["A_c", " + ".join(expr), " + ".join(vals)], "res": f"{_net(q['aire_cm2'], 2)} cm²"},
        {"parts": ["V_c", "A_c · L / 10 000", f"{_v(q['aire_cm2'])} × {_v(g['L'])} / 10 000"], "res": f"{C.fr(q['Vc'], 3)} m³"},
        {"parts": ["M_c", "V_c · 2 500", f"{_v(q['Vc'])} × 2 500"], "res": f"{C.fr(q['masse_beton'], 0)} kg"},
        {"parts": ["V_s", "V_long + V_étriers", f"{_v(q['V_longitudinales'])} + {_v(q['V_etriers'])}"], "res": f"{C.nb(round(q['V_acier'], 6), 6)} m³"},
        {"parts": ["M_s", "V_s · 7 850", f"{_v(q['V_acier'])} × 7 850"], "res": f"{C.fr(q['masse_acier'], 2)} kg"},
        {"parts": ["M_s,maj", "M_s · (1 + maj / 100)", f"{_v(q['masse_acier'])} × (1 + {_v(maj)} / 100)"], "res": f"{C.fr(q['masse_acier_majoree'], 2)} kg"},
        {"parts": ["Taux", "M_s,maj / V_c", f"{_v(q['masse_acier_majoree'])} / {_v(q['Vc'])}"], "res": f"{C.fr(q['kg_m3_majore'], 1)} kg/m³"},
        {"parts": ["Taux sans majoration", "M_s / V_c", f"{_v(q['masse_acier'])} / {_v(q['Vc'])}"], "res": f"{C.fr(q['kg_m3'], 1)} kg/m³"},
    ]


def _part(x: float) -> int:
    """La part (0 à 1) écrite en % entier, sans jamais montrer 0 % pour une part non nulle ni 100 % pour une part
    incomplète."""
    v = math.floor(x * 100 + 0.5)
    if x > 0 and v == 0:
        return 1 if x >= 0.005 else 0
    if x < 1 and v == 100:
        return 99
    return int(v)


def _lignes_tableau(r: Dict, actif: str) -> List[Dict]:
    """Les lignes de la fenêtre de calcul : chaque famille, chaque zone, puis le total (masse, apport au taux, part)."""
    q = r["quantites"]
    out = []
    for f in r["familles"]:
        out.append({"nom": C.esc(f["code"]), "sous": C.esc(f"{f['libelle_role']} · {f['libelle']}"),
                    "formule": "L<sub>tot</sub> · m<sub>lin</sub> · (1 + maj)", "valeur": f"{C.fr(f['masse_majoree'], 2)} kg",
                    "limite": f"{C.fr(f['apport'], 1)} kg/m³", "taux": _part(f["part"]), "niveau": "c", "gov": actif == f"f{f['index']}"})
    for e in r["etriers"]:
        out.append({"nom": C.esc(e["zone"]), "sous": C.esc(f"{e['n']} étriers {e['type']} Ø{C.nb(e['d'])} · e = {C.nb(e['s'])} cm"),
                    "formule": "n · L<sub>étrier</sub> · m<sub>lin</sub> · (1 + maj)", "valeur": f"{C.fr(e['masse_majoree'], 2)} kg",
                    "limite": f"{C.fr(e['apport'], 1)} kg/m³", "taux": _part(e["part"]), "niveau": "c", "gov": actif == f"z{e['index']}"})
    out.append({"nom": "Total", "sous": f"majoration {C.nb(q['majoration'])} % comprise", "formule": "M<sub>s,maj</sub> / V<sub>c</sub>",
                "valeur": f"{C.fr(q['masse_acier_majoree'], 2)} kg", "limite": f"{C.fr(q['kg_m3_majore'], 1)} kg/m³",
                "taux": 100 if q["V_acier"] > 0 else 0, "niveau": "c", "gov": actif == "q"})
    return out


TETES = (('<th>Vérification</th>', '<th>Armature</th>'), ('<th class="n">Valeur</th>', '<th class="n">Masse</th>'),
         ('<th class="n">Limite</th>', '<th class="n">Apport au taux</th>'), ('<th class="n">Taux</th>', '<th class="n">Part</th>'))


def _fenetre(ident: str, titre: str, sous: str, hyps: List[str], lignes: List[Dict], detail: str, pied: str = "") -> str:
    """La fenêtre de calcul commune (C.fenetre_calcul), ses en-têtes de colonnes dits pour un QUANTITATIF : armature,
    masse, apport au taux, part de l'acier (la brique écrit « vérification, valeur, limite, taux »)."""
    h = C.fenetre_calcul(ident, titre, sous, hyps, lignes, detail, pied)
    for a, b in TETES:
        h = h.replace(a, b, 1)
    return h


def _groupe(titre: str, lignes: List[Dict]) -> str:
    return f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div>{C.lignes_fx([dict(li, niveau="") for li in lignes])}</div>'


def _hypotheses(r: Dict) -> List[str]:
    p, g = r["poutre"], r["geometrie"]
    sec = f"b × h = {C.nb(g['b'])} × {C.nb(g['h'])} cm"
    if g["bg"] > 0:
        sec += f" · talon gauche {C.nb(g['bg'])} × {C.nb(g['hg'])}"
    if g["bd"] > 0:
        sec += f" · talon droit {C.nb(g['bd'])} × {C.nb(g['hd'])}"
    return [f"Acier ρ<sub>s</sub> = 7 850 kg/m³ · <b>{C.esc(p['acier'])}</b> (sans effet sur le calcul)", "Béton ρ<sub>c</sub> = 2 500 kg/m³",
            f"Majoration <b>{C.nb(p['majoration'])} %</b>", f"L = {C.nb(g['L'])} m · {sec} · c = {C.nb(g['c'])} cm"]


def _volets(r: Dict) -> str:
    p, g, q = r["poutre"], r["geometrie"], r["quantites"]
    maj, Vc, L = p["majoration"], q["Vc"], g["L"]
    hyps = _hypotheses(r)
    nom = C.esc(p.get("nom") or "Poutre")
    detail_q = _groupe("Béton et taux de la poutre", _fx_poutre(r))
    detail_q += "".join(_groupe(f"{f['code']} · {f['libelle_role']} {f['libelle']}", _fx_famille(f, maj, Vc, L)) for f in r["familles"])
    detail_q += "".join(_groupe(f"Zone {e['zone']} · étriers {e['type']}", _fx_zone(e, g, maj, Vc, L)) for e in r["etriers"])
    out = _fenetre("q", f"{nom} · quantitatif", f"L = {C.nb(L)} m · majoration {C.nb(maj)} % · le calcul de l'écran Streamlit",
                   hyps, _lignes_tableau(r, "q"), detail_q)
    for f in r["familles"]:
        detail = _groupe(f"{f['code']} · {f['libelle_role']} {f['libelle']} · {f['libelle_position']}", _fx_famille(f, maj, Vc, L))
        detail += _groupe("Béton et taux de la poutre", _fx_poutre(r))
        out += _fenetre(f"f{f['index']}", f"{nom} · {C.esc(f['code'])} · {C.esc(f['libelle_role'])} {C.esc(f['libelle'])}",
                        f"{C.esc(f['libelle_position'])} · de {C.nb(f['de'])} à {C.nb(f['a'])} % de L", hyps, _lignes_tableau(r, f"f{f['index']}"), detail)
    for e in r["etriers"]:
        detail = _groupe(f"Zone {e['zone']} · étriers {e['type']} Ø{C.nb(e['d'])}", _fx_zone(e, g, maj, Vc, L))
        detail += _groupe("Béton et taux de la poutre", _fx_poutre(r))
        pied = ""
        if any(a["code"] == "troncature_etriers" and f"Zone {e['zone']} :" in a["message"] for a in r["avertissements"]):
            pied = ('<div class="ck-autres">L / s tombe juste à l\'écran mais pas en machine : la troncature du module compte un étrier '
                    'de moins (défaut reproduit).</div>')
        out += _fenetre(f"z{e['index']}", f"{nom} · zone {C.esc(e['zone'])} · étriers {C.esc(e['type'])}",
                        f"de {C.nb(e['de'])} à {C.nb(e['a'])} % de L · e = {C.nb(e['s'])} cm", hyps, _lignes_tableau(r, f"z{e['index']}"), detail, pied)
    return out


def _resultat_html(etat: Dict, r: Dict, synth: Dict) -> str:
    p, q = r["poutre"], r["quantites"]
    t = q["kg_m3_majore"]
    # deux lignes (la masse, puis le béton) ; une espace insécable lie chaque nombre à son unité : au téléphone, la
    # ligne se coupait entre « 1,380 » et « m³ »
    nb = C.NBSP
    verdict = (f'<div class="verdict tx-verdict" data-calcul="q" title="Voir le calcul"><span class="chip xl c tx-taux">{C.fr(t, 1)}<small>kg/m³</small></span>'
               f'<div class="gov" title="Masse d\'acier majorée de {C.nb(p["majoration"])} % (chutes, recouvrements, façonnage)">'
               f'<b>Taux d\'armature</b><span class="tx-l">{C.fr(q["masse_acier_majoree"], 1)}{nb}kg d\'acier (+{C.nb(p["majoration"])}{nb}%)</span>'
               f'<span class="tx-l">pour {C.fr(q["Vc"], 3)}{nb}m³ de béton</span></div></div>')
    chiffres = ('<div class="tx-synth">'
                f'<div><span>Volume de béton</span><b>{C.fr(q["Vc"], 3)} m³</b></div>'
                f'<div><span>Masse d\'acier, majorée</span><b>{C.fr(q["masse_acier_majoree"], 1)} kg</b></div>'
                f'<div title="Masse volumique du béton : 2 500 kg/m³"><span>Poids du béton</span><b>{C.fr(q["masse_beton"], 0)} kg</b></div>'
                f'<div title="Masse d\'acier sans majoration : {C.fr(q["masse_acier"], 1)} kg"><span>Taux sans majoration</span>'
                f'<b>{C.fr(q["kg_m3"], 1)} kg/m³</b></div></div>')
    rows = ""
    if r["familles"]:
        m = sum(f["masse_majoree"] for f in r["familles"])
        rows += f'<div class="elgrp">Armatures longitudinales<span class="pct c">{C.fr(m, 1)} kg</span></div>'
        for f in r["familles"]:
            rows += C.ligne_res(f"{f['code']} · {ROLES_COURTS[f['role']]}", f"{C.esc(f['libelle'])} · {C.fr(f['masse_majoree'], 1)} kg",
                                _part(f["part"]), "c", calcul=f"f{f['index']}", pk=f"d-{f['index']}")
    if r["etriers"]:
        m = sum(e["masse_majoree"] for e in r["etriers"])
        rows += f'<div class="elgrp">Étriers<span class="pct c">{C.fr(m, 1)} kg</span></div>'
        for e in r["etriers"]:
            rows += C.ligne_res(f"{e['zone']} · étriers {e['type']}", f"{e['n']} × Ø{C.nb(e['d'])} · {C.fr(e['masse_majoree'], 1)} kg",
                                _part(e["part"]), "c", calcul=f"z{e['index']}")
    if rows:
        rows += '<div class="tx-note-barre">La barre : la part de chaque armature dans la masse d\'acier.</div>'
    messages = "".join(C.message(C.esc(a["message"]), a["grave"]) for a in r["avertissements"])
    leg = "".join(f'<span class="tx-lg"><i style="background:{c};box-shadow:inset 0 0 0 1.5px {b}"></i>{t}</span>'
                  for c, b, t in (("#e66565", DT.BORDS["bas"], "lits du bas"), ("#6f8fe6", DT.BORDS["haut"], "lits du haut"),
                                  ("#666666", DT.BORDS["cote"], "latérales"), (DT.ETRIER, DT.ETRIER, "étriers")))
    note = ""
    if synth["n"] > 1:
        note = (f'<div class="mass" title="{C.fr(synth["Vc"], 3)} m³ de béton · {C.fr(synth["masse_acier_majoree"], 1)} kg d\'acier majoré">'
                f'<span>Note entière · {synth["n"]} poutres</span><b>{C.fr(synth["kg_m3_majore"], 1)}{C.NBSP}kg/m³</b></div>')
    pied = (f'<div class="pfoot">{note}<div class="mass" title="Un quantitatif : le module ne vérifie rien"><span>Acier de la poutre · {C.esc(p["acier"])}</span>'
            f'<b>{C.fr(q["masse_acier_majoree"], 1)}{C.NBSP}kg</b></div><div class="legend tx-leg">{leg}</div></div>')
    return f'<div class="pbody">{verdict}{chiffres}<div class="els">{rows}</div>{messages}</div>{pied}'


def _tableau(r: Dict) -> str:
    """Le tableau récapitulatif de l'écran : mêmes colonnes, mêmes précisions (2 décimales pour les longueurs, 3 pour les
    kg/m, 2 pour les kg) ; le Ø d'un étrier est écrit tel quel (l'écran le tronque à l'entier)."""
    q = r["quantites"]
    lignes = [{"couleur": f["couleur"], "bord": DT.BORDS[f["groupe"]], "code": f["code"], "position": f["libelle_position"], "n": f["n_eff"],
               "d": f["d"], "L_active": C.fr(f["L_active"], 2), "L_tot": C.fr(f["L_tot"], 2), "m_lin": C.fr(f["m_lin"], 3),
               "masse": C.fr(f["masse_majoree"], 2)} for f in r["familles"]]
    lignes += [{"couleur": DT.ETRIER, "bord": DT.ETRIER, "code": e["type"], "position": f"Zone {e['zone']}", "n": e["n"], "d": C.nb(e["d"]),
                "L_active": C.fr(e["L_un"], 2), "L_tot": C.fr(e["L_tot"], 2), "m_lin": C.fr(e["m_lin"], 3), "masse": C.fr(e["masse_majoree"], 2)}
               for e in r["etriers"]]
    total = {"libelle": f"majoration {C.nb(q['majoration'])} % comprise", "L_tot": "", "masse": C.fr(q["masse_acier_majoree"], 2),
             "pied": ("Comme l'écran Streamlit : « kg tot » inclut la majoration ; pour les étriers, « n barres » est le nombre "
                      "d'étriers et « L active » la longueur d'un étrier.")}
    return DT.tableau(lignes, total)


def _tableau_note(etat: Dict, resultats: List[Dict], synth: Dict) -> str:
    rows = ""
    for i, (p, r) in enumerate(zip(etat["poutres"], resultats)):
        if "quantites" not in r:
            continue
        q = r["quantites"]
        rows += (f'<tr{" class=\"on\"" if i == etat["actif"] else ""}><th>{C.esc(p.get("nom") or f"Poutre {i + 1}")}</th>'
                 f'<td class="po">{C.nb(p["b"])} × {C.nb(p["h"])} · {C.esc(EN.FORMES[p["forme"]])}</td><td>{C.nb(p["L"])}</td>'
                 f'<td>{C.fr(q["Vc"], 3)}</td><td>{C.fr(q["masse_acier_majoree"], 1)}</td><td class="kg">{C.fr(q["kg_m3_majore"], 1)}</td></tr>')
    pied = (f'<tr><th>Note</th><td class="po">{synth["n"]} poutres</td><td></td><td>{C.fr(synth["Vc"], 3)}</td>'
            f'<td>{C.fr(synth["masse_acier_majoree"], 1)}</td><td class="kg">{C.fr(synth["kg_m3_majore"], 1)}</td></tr>')
    return ('<div class="tx-vue"><div class="tx-defile"><table class="tx-tab tx-note"><thead><tr><th>Poutre</th><th class="po">Section</th>'
            '<th>L<small>m</small></th><th>V<sub>c</sub><small>m³</small></th><th>Acier<small>kg</small></th><th>Taux<small>kg/m³</small></th>'
            f'</tr></thead><tbody>{rows}</tbody><tfoot>{pied}</tfoot></table></div>'
            '<div class="tx-pied">Le taux de la note : la masse d\'acier majorée de toutes les poutres divisée par leur volume de béton '
            '(ajout du site : l\'écran Streamlit n\'a qu\'une poutre).</div></div>')


def _calculer(p: Dict) -> Dict:
    try:
        return MO.calculer(p)
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : une consultation (pas de verdict) ; le taux de la note et sa poutre la plus armée."""
    ok = [(r["quantites"]["kg_m3_majore"], i) for i, r in enumerate(resultats) if "quantites" in r]
    if not ok:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    synth = MO.synthese_note([resultats[i] for _, i in ok])
    t, i = max(ok)
    nom = etat["poutres"][i].get("nom") or f"Poutre {i + 1}"
    if len(ok) == 1:
        gouv = f"{nom} · {C.fr(t, 1)} kg/m³"
    else:
        gouv = f"note : {C.fr(synth['kg_m3_majore'], 1)} kg/m³ · la plus armée : {nom} ({C.fr(t, 1)} kg/m³)"
    return {"taux": None, "niveau": "i", "statut": C.STATUTS["i"][1], "gouvernant": gouv}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    items = ""
    for i, (p, r) in enumerate(zip(etat["poutres"], resultats)):
        t = r.get("quantites", {}).get("kg_m3_majore")
        txt = f"{C.fr(t, 1)}{C.NBSP}kg/m³" if t is not None else "—"
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["poutres"]) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(p.get("nom") or f"Poutre {i + 1}")} '
                  f'<span>· {C.nb(p["b"])} × {C.nb(p["h"])} · {C.nb(p["L"])} m</span></span><span class="pct c">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["poutres"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier la poutre ouverte">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une poutre</div></div>')


def resultat(etat: Dict) -> Dict:
    p = _poutre(etat)
    r = MO.calculer(p)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["poutres"])]
    synth = MO.synthese_note([x for x in resultats if "quantites" in x])
    g = r["geometrie"]
    lib = EN.FORMES[p["forme"]]
    vues = [{"cle": "coupe", "titre": "Coupe", "court": "Coupe", "svg": DT.coupe("tx_coupe", r),
             "legende": f"{C.nb(g['b'])} × {C.nb(g['h'])} cm · {lib}"},
            {"cle": "elevation", "titre": "Élévation", "court": "Élévation", "svg": DT.elevation("tx_elevation", r),
             "legende": "longueurs à l'échelle" + (f" · hauteurs × {C.nb(DT.amplification(g['L'], g['h']))}" if DT.amplification(g["L"], g["h"]) > 1 else "")},
            {"cle": "tableau", "titre": "Tableau récapitulatif", "court": "Tableau", "svg": _tableau(r), "legende": "les armatures de la poutre"}]
    if len(etat["poutres"]) > 1:
        vues.append({"cle": "note", "titre": "Note entière", "court": "Note", "svg": _tableau_note(etat, resultats, synth),
                     "legende": f"{synth['n']} poutres"})
    phrase = (f"{C.nb(p['b'])} × {C.nb(p['h'])} cm · L = {C.nb(p['L'])} m · c = {C.nb(p['enrobage'])} cm · {C.esc(lib)}")
    return {"resultat": _resultat_html(etat, r, synth), "volets": _volets(r), "vues": vues, "phrase": phrase,
            "statut_html": C.statut("i"), "resume": resume_note(etat, resultats), "donnees": formulaire(etat, r),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de diamètre
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les dix Ø de l'écran pour une famille, triés par diamètre (donc par masse) : pour chacun, le taux d'armature de
    la poutre et la part de la famille dans l'acier, recalculés par le moteur. Un quantitatif n'a pas d'« optimal » :
    aucune ligne n'est étoilée."""
    i = _index_role(etat, role)
    p = _poutre(etat)
    r = MO.calculer(p)
    f = r["familles"][i]
    cands = MO.candidats_diametre(p, i)
    i_c = next((k for k, c in enumerate(cands) if c["retenu"]), 0)
    rows = [{"nom": c["nom"], "col": C.fr(c["kg_m3_majore"], 1), "taux": _part(c["part"]), "niveau": "c", "cur": c["retenu"], "opt": False,
             "prop": (f"{c['nom']} · A<sub>s</sub> = {C.fr(c['As'], 0)} mm² · {C.fr(c['m_lin'], 3)} kg/m · {C.fr(c['masse_majoree'], 1)} kg "
                      f"(majorés) · taux de la poutre {C.fr(c['kg_m3_majore'], 1)} kg/m³")} for c in cands]
    unite = "taux de la poutre (kg/m³) · part de la famille dans l'acier"
    h = C.selecteur(role, f"{C.esc(f['code'])} · {C.esc(f['libelle_role'])} · diamètre",
                    f"{C.esc(f['libelle_position'])} · un clic retient le{C.NBSP}Ø", "", rows, max(0, i_c - 4),
                    unite, etoile="pas d'optimal : quantitatif sans vérification")
    # la brique écrit « <unité> · taux » au pied : ici la pastille est une PART, pas un taux de travail
    return h.replace(f"<span>{unite} · taux</span>", f"<span>{unite}</span>", 1)
