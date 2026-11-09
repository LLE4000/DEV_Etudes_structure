# -*- coding: utf-8 -*-
"""Le module « Cornières d'appui » du site, refondu le 10/11/2026 sur la méthode du classeur de l'ingénieur
(``docs/acier/CORNIERE_DECISIONS.md`` § 4 et § 6 ; le moteur : ``acier/corniere``, ``docs/acier/CORNIERE_MOTEUR.md``),
sur l'ÉTABLI COMPACT — le modèle de la poutre (api_web/poutre.py, api_web/etabli_compact.py) :

- à gauche, la ligne de la cornière ouverte (son nom, la liste des cornières de la note, copier, supprimer) puis ses
  quatre rubriques : 1 Cornière (catalogue ou saisie libre, la désignation tapée, la fiche du catalogue, l'aile
  horizontale avec les longueurs réelles, la configuration A/B en pictogrammes, la nuance, le coefficient), 2 Charge
  (R, L, X ± dx, a, R_ELU), 3 Ancrage (Y ± dy, n, d), 4 Hypothèses (secondaire : le mode — inactif en B, avec la
  raison —, le jeu d'ancrage du classeur, le pivot y_c, la ligne d'action u_V) ; données condensées à deux champs par
  ligne, les explications dans les ⓘ ;
- au centre, les quatre VUES de ``acier/corniere/schemas.py`` posées par ``api_web/dessin_corniere.py`` (coupe,
  moments, contraintes, élévation), sous leur en-tête technique ; le CONTRAT DE SÉLECTION : chaque pièce du dessin
  (``data-sel``) a sa ligne dans les données sous la même clé (``data-cible`` : X, dx, Y, dy, R, L, a, n, d, profil,
  yc, uV, jeu_ancrage), un clic sur l'une montre l'autre (js/compact.js) ;
- à droite, le VERDICT (taux max, section gouvernante et sa combinaison de jeux, configuration et mode), trois blocs
  THÉMATIQUES — « Sections » (S1 à S4, les maxima du congé, la section nette en B : deux lignes par vérification,
  colorées, chacune avec la combinaison de jeux qui la gouverne ; en mode historique les maxima sont en information,
  hors verdict), « Ancrages » (N et V par ancrage, ELS et ELU, les positions ; pression diamétrale, arrachement,
  compression du mur en A ou pression sur le support en B : des lignes d'information, grises, hors verdict, comme le
  pied de poteau), « Comparaison » (le taux de l'autre mode et la garde, en A seulement) —, les hypothèses du moteur,
  les avertissements titrés (le grave en tête), la masse ; cinq FENÊTRES DE CALCUL (sections, ancrages, comparaison,
  géométrie et matériau, efforts) dont chaque ligne « symbole = expression = valeurs = résultat » s'écrit à partir des
  seuls termes du moteur (le test la réévalue), la cellule du classeur entre parenthèses en mode historique ;
- le SÉLECTEUR de la ligne « Cornière » : les 233 cornières du catalogue triées par masse, chacune calculée avec les
  charges de la cornière ouverte — ✓ la retenue, ★ la plus légère qui tient ET dont la charge et l'ancrage sont dans
  les ailes, grise hors domaine (sa raison au survol), rouge quand elle ne tient pas.

Ce fichier MET EN FORME : il ne calcule rien (R3) ; il CHOISIT parmi ce que le moteur a nommé et COMPOSE les phrases.
Toute saisie passe par ``moteur.normaliser`` : hors domaine, elle est REFUSÉE avec le message du moteur et la cornière
reste intacte. Une note du module d'avant (version 1, modèle lame : clés V, e, s, gamma_ELU, bord, sig_lim_*) se
rouvre : le nom et la cornière sont gardés, le reste prend les défauts (``entree.depuis_v1``), et l'AVIS le dit une
fois à l'ouverture (CLE_AVIS, le mécanisme des anciennes dalles) ; l'avertissement ``note_v1`` du moteur reste tant que
la charge ou l'ancrage n'ont pas été ressaisis (la clé ``v1`` de la cornière retombe alors).
"""
import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

from acier.corniere import entree as EN
from acier.corniere import moteur as MO

from . import commun as C
from . import dessin_corniere as DC
from . import etabli_compact as E

MODULE = "corniere"
PAGE = "corniere"
VERSION = 2                     # l'état de la note : {"version": 2, "cornieres": [...], "actif": 0} (1 : le modèle lame)
# le nom des cornières neuves : « C1 », « C2 »… — court, comme « V1 » de la poutre (« Cornière 1 » ne tenait pas entier dans la
# ligne de l'élément à la tablette portrait, 77 px à côté de six boutons de 44)
NOM_DEFAUT = "C1"
CORNIERE = E.Genre("Cornières", "la cornière", "cette cornière", "une cornière")
# L'AVIS d'une note du module d'avant (modèle lame), dit UNE fois à l'ouverture : ``normaliser`` le pose sous CLE_AVIS
# quand une cornière arrive dans la forme de la version 1 (entree.est_v1) ; ``resultat`` le montre en tête du résultat
# et le RETIRE de l'état — la note rendue, enregistrée par l'écran, est dans la forme d'aujourd'hui et ne le redit pas.
# (Le texte est celui des décisions § 6, source unique : entree.AVIS_V1 ; le moteur le redit en avertissement tant que
# la cornière porte « v1 ».)
CLE_AVIS = "avis_ouverture"
AVIS = {"v1": EN.AVIS_V1[0].upper() + EN.AVIS_V1[1:] + "."}

# les champs de la saisie : nombres, « vide = auto », choix ; une saisie de la charge ou de l'ancrage (RESSAISIE) fait
# retomber le drapeau « v1 » d'une cornière relue du modèle lame (ses charges et son ancrage sont alors ressaisis)
NUMERIQUES = ("A", "B", "t", "r", "fy", "coef", "R", "L", "X", "dx", "Y", "dy", "n", "d", "jeu_ancrage", "yc")
AUTO = EN.OPTIONNELS                                      # a (vide = L), R_ELU (vide = 1,5·R), uV (vide = e/2)
CHOIX = tuple(EN.CHOIX)                                   # aile_horizontale, configuration, mode
RESSAISIE = ("R", "L", "X", "dx", "a", "R_ELU", "Y", "dy", "n", "d")
ORIGINES = (("catalogue", "Catalogue"), ("saisie", "Saisie libre"))
# les nuances du site, par limite élastique croissante : (f_y, « S355 ») — la source unique est entree.NUANCES
NUANCES = tuple((int(fy) if float(fy).is_integer() else fy, nom) for nom, fy in sorted(EN.NUANCES.items(), key=lambda kv: kv[1]))

# une note du module d'avant (version 1, modèle lame), telle qu'un fichier .stab d'avant le 10/11/2026 la porte : le
# parcours la rejoue (la relecture, l'avis), tests/test_api_web_corniere.py la contrôle
NOTE_V1 = {"version": 1, "actif": 0,
           "cornieres": [{"nom": "Cornière de rive", "standard": True, "profil": "L 100x100x10", "charge": "lineaire", "V": 5.0,
                          "L": 1.0, "e": 2.3, "gamma_ELU": 1.35, "s": 60.0, "bord": 5.0, "sig_lim_ELS": 156.7, "tau_lim_ELS": 0.8,
                          "sig_lim_ELU": 235.0}]}


def _note(**champs) -> Dict:
    """Une note neuve dont la cornière reçoit ``champs`` (pour le parcours : l'ancrage descendu à Y = 65 avant une
    inégale dont l'aile verticale n'a que 100 mm)."""
    c = MO.nouvelle_corniere(NOM_DEFAUT)
    c.update(champs)
    return {"version": VERSION, "cornieres": [c], "actif": 0}


# les requêtes de plus pour le parcours (api_web/parcours.py) et le relevé du paquet : les deux configurations, les deux
# modes, une inégale dans les deux ordres (150 puis 100 à l'horizontale), le sélecteur, des saisies refusées (la charge
# dans l'angle, une cornière trop petite pour la charge, l'ancrage hors de l'aile), la largeur d'appui, R_ELU saisie,
# sans jeu, la saisie libre, la note v1
PARCOURS = [
    {"action": "modifier", "etat": _note(Y=65.0), "chemin": ["profil"], "valeur": "L 150×100×12"},
    {"action": "modifier", "etat": _note(aile_horizontale="B"), "chemin": ["profil"], "valeur": "100 x 150 x 12"},
    {"action": "modifier", "chemin": ["profil"], "valeur": "L 150×100×12"},
    {"action": "modifier", "chemin": ["aile_horizontale"], "valeur": "B"},
    {"action": "modifier", "chemin": ["configuration"], "valeur": "B"},
    {"action": "modifier", "chemin": ["mode"], "valeur": EN.EQUILIBRE},
    {"action": "modifier", "chemin": ["X"], "valeur": "10"},
    {"action": "modifier", "chemin": ["profil"], "valeur": "L 45×45×4,5"},
    {"action": "modifier", "chemin": ["a"], "valeur": "100"},
    {"action": "modifier", "chemin": ["R_ELU"], "valeur": "68,9"},
    {"action": "modifier", "chemin": ["dx"], "valeur": "0"},
    {"action": "modifier", "chemin": ["dy"], "valeur": "0"},
    {"action": "modifier", "chemin": ["standard"], "valeur": "saisie"},
    {"action": "choisir", "role": "corniere", "nom": "L 150×150×15"},
    {"action": "selecteur", "role": "corniere", "familles": ["LI"]},
    {"action": "calculer", "etat": NOTE_V1},
]


# ------------------------------------------------------------------ l'état de la note : des cornières
def nouvel_etat() -> Dict:
    """Une note neuve : une cornière aux défauts des décisions § 2 (L 120×120×15, configuration A, méthode historique,
    S355, coef 1,5, R = 35 kN sur 300 mm, X = 75 ± 20, Y = 85 ± 10, 3 Ø16)."""
    return {"version": VERSION, "cornieres": [MO.nouvelle_corniere(NOM_DEFAUT)], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; une cornière illisible lève une erreur claire
    (le routeur la rend, l'écran l'affiche). Une note sans cornière est une note neuve ; une liste qui n'en est pas une
    est une note abîmée (refusée). Une cornière du module d'AVANT (version 1, modèle lame : ``entree.est_v1``) est
    relue par le moteur — nom et section gardés, le reste aux défauts posés dans ses ailes — et l'avis d'ouverture est
    posé (CLE_AVIS, que ``resultat`` dit et retire)."""
    cornieres = etat.get("cornieres") if isinstance(etat, dict) else None
    if not cornieres:
        return nouvel_etat()
    if not isinstance(cornieres, list):
        raise ValueError("note abîmée : « cornieres » doit être une liste")
    out = {"version": VERSION, "cornieres": []}
    avis = False
    for i, c in enumerate(cornieres):
        if not isinstance(c, dict):
            raise ValueError(f"cornière n° {i + 1} illisible : un objet est attendu")
        avis = avis or EN.est_v1(c)
        try:
            out["cornieres"].append(MO.normaliser(c))
        except ValueError as e:
            raise ValueError(f"cornière « {c.get('nom', i + 1)} » : {e}") from None
    out["actif"] = min(max(0, _indice(etat.get("actif"))), len(out["cornieres"]) - 1)
    if avis:
        out[CLE_AVIS] = "v1"
    return out


def _indice(x) -> int:
    """L'indice de la cornière ouverte, lu sans faire tomber la note : un nombre entier fini, sinon la première."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return 0
    return int(x)


def _corniere(etat: Dict) -> Dict:
    return etat["cornieres"][etat["actif"]]


def _nom_libre(cs: List[Dict]) -> str:
    pris = {c.get("nom") for c in cs}
    k = len(cs) + 1
    while f"C{k}" in pris:
        k += 1
    return f"C{k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les cornières de la note : en ouvrir une, en ajouter, en copier une (``index`` ; sans : l'ouverte — la copie
    s'insère juste après), en supprimer une (jamais la dernière)."""
    cs = etat["cornieres"]
    if action == "activer":
        etat["actif"] = min(max(0, _indice(index)), len(cs) - 1)
    elif action == "ajouter":
        c = MO.nouvelle_corniere(_nom_libre(cs))
        cs.append(c)
        etat["actif"] = len(cs) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else min(max(0, _indice(index)), len(cs) - 1)
        c = copy.deepcopy(cs[i])
        c["nom"] = f"{c.get('nom') or 'Cornière'} (copie)"[:EN.NOM_MAX]
        cs.insert(i + 1, c)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(cs) > 1:
        i = min(max(0, _indice(index)), len(cs) - 1)
        del cs[i]
        if etat["actif"] > i or etat["actif"] >= len(cs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la cornière ouverte ; ``chemin`` : une seule clé (les champs sont à plat). Le moteur relit
    la cornière : une valeur hors de son domaine est REFUSÉE avec son message (jamais ramenée en silence), et la
    cornière reste intacte. ``standard`` : « catalogue » ou « saisie » (la saisie libre part des dimensions de la
    cornière du catalogue affichée) ; ``profil`` : une désignation tapée, relue au catalogue ; a, R_ELU, uV : « vide =
    automatique » (commun.lire_auto). Une saisie de la charge ou de l'ancrage fait retomber le drapeau « v1 » d'une
    cornière relue du modèle lame."""
    c = _corniere(etat)
    avant = copy.deepcopy(c)
    if len(chemin) != 1:
        return f"Champ inconnu : « {'/'.join(str(x) for x in chemin)} »"
    k = chemin[0]
    try:
        if k == "nom":
            nom = str(valeur if valeur is not None else "").strip()
            if len(nom) > EN.NOM_MAX:
                return f"Nom trop long : {len(nom)} caractères (au plus {EN.NOM_MAX})"
            c["nom"] = nom
        elif k == "standard":
            v = {"catalogue": True, "saisie": False, True: True, False: False, "True": True, "False": False}.get(valeur)
            if v is None:
                return f"Origine de la cornière : « {valeur} » (catalogue ou saisie libre)"
            if not v and c["standard"]:
                sec = MO.section(c)                      # la saisie libre part de la cornière affichée
                c.update(A=sec["A"], B=sec["B"], t=sec["t"], r=sec["r1"])
            c["standard"] = v
        elif k == "profil":
            nom = EN.nom_catalogue(valeur)
            if nom is None:
                return f"Cornière absente du catalogue : « {valeur} » (ex. L 80×80×8, 100×65×8)"
            c["profil"] = nom
            c["standard"] = True
        elif k in CHOIX:
            c[k] = valeur
        elif k in AUTO:
            try:
                c[k] = C.lire_auto(valeur)
            except ValueError as e:
                return str(e)
        elif k in NUMERIQUES:
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            c[k] = int(x) if k == "n" and float(x).is_integer() else x
        else:
            return f"Champ inconnu : « {k} »"
        if k in RESSAISIE and c.get("v1"):
            c["v1"] = False
        etat["cornieres"][etat["actif"]] = MO.normaliser(c)
    except (ValueError, KeyError, TypeError) as e:
        etat["cornieres"][etat["actif"]] = avant
        return f"Saisie refusée : {e}"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le clic dans le sélecteur : la cornière du catalogue est retenue (la cornière passe en « catalogue »)."""
    c = _corniere(etat)
    n = EN.nom_catalogue(nom)
    if n is None:
        raise ValueError(f"cornière absente du catalogue : « {nom} »")
    c["standard"] = True
    c["profil"] = n
    etat["cornieres"][etat["actif"]] = MO.normaliser(c)


# ------------------------------------------------------------------ les nombres
def _nd(x: float, nd: int, sig: int = 2) -> int:
    """Assez de décimales pour garder ``sig`` chiffres significatifs aux petites valeurs (0,0345 kN·m, pas « 0,03 »)."""
    a = abs(x)
    while a and nd < 6 and a < 10 ** (sig - 1 - nd):
        nd += 1
    return nd


DECIMALES = {"N/mm²": 1, "N/mm": 1, "kN": 2, "kN·m": 3, "mm³": 0, "mm²": 0, "mm": 1, "": 4}


def _n(x: Optional[float], unite: str = "") -> str:
    """Un RÉSULTAT à la précision d'usage : contraintes au dixième, efforts au centième, moments au millième (comme le
    classeur), sections entières, mm au dixième, coefficients à quatre décimales — et toujours deux chiffres
    significatifs."""
    if x is None:
        return "—"
    nd = DECIMALES.get(unite, 2)
    return C.fr(x, _nd(x, nd) if nd else 0)


def _nu(x: Optional[float], unite: str) -> str:
    """Un résultat suivi de son unité (insécable)."""
    return C.qte(x, unite, _nd(x, DECIMALES.get(unite, 2)) if x is not None and DECIMALES.get(unite, 2) else 0) if unite else _n(x, unite)


def _v(x: Optional[float]) -> str:
    """Une VALEUR substituée dans une formule, écrite sans la dénaturer : 75 reste 75, 1,5 reste 1,5, 42,76 reste 42,76 ;
    une grandeur calculée longue (236,666…) est arrondie à la précision d'usage ; un nombre négatif est mis entre
    parenthèses (il suit un signe)."""
    if x is None:
        return "—"
    a = abs(x)
    s = C.nb(round(x, 1), 1) if a >= 100 else (C.nb(round(x, 2), 2) if a >= 10 else C.nb(round(x, 4), 4))
    return f"({s})" if x < 0 else s


def _pct(t: Optional[float]) -> str:
    return f"{C.pc(C.pct(t, 1))}{C.NBSP}%" if t is not None else "—"


def _niv(v: Dict) -> str:
    """La couleur d'une vérification : rouge dès que le moteur la déclare non conforme, sinon le taux."""
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


def _niv_res(res: Dict) -> str:
    return "r" if res["statut"] == MO.NOK else C.niveau(res["taux_max"])


# ------------------------------------------------------------------ les formules du moteur, mises en mots
# l'EXPRESSION de chaque formule du moteur (la table des clés de la docstring de acier/corniere/moteur.py) : écrite une
# fois ici, avec les noms que le moteur donne à ses termes ; les valeurs substituées sont ses termes, le résultat est le
# sien — rien n'est recalculé (tests/test_api_web_corniere.py réévalue chaque ligne affichée)
def _fx(f: Dict) -> Dict:
    """Une formule du moteur → « symbole (cellule) = expression = valeurs substituées = résultat »."""
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    noms = list(t)
    k, s = f["cle"], f["symbole"]

    def v(nom):
        return _v(t[nom])

    if k == "sigma_adm":
        parts = [s, "f_y / coef", f"{v('f_y')} / {v('coef')}"]
    elif k == "tau_adm":
        parts = [s, "σ_adm / √3", f"{v('σ_adm')} / √3"]
    elif k in ("Xp", "Yp"):
        parts = [s, f"{noms[0]} + {noms[1]}", f"{v(noms[0])} + {v(noms[1])}"]
    elif k == "Xe":
        parts = [s, f"{noms[0]} − r − e", f"{v(noms[0])} − {v('r')} − {v('e')}"]
    elif k == "Xer":
        parts = [s, f"{noms[0]} − e", f"{v(noms[0])} − {v('e')}"]
    elif k == "Ye":
        parts = [s, "Y − e − r", f"{v('Y')} − {v('e')} − {v('r')}"]
    elif k == "Yer":
        parts = [s, "Y − e", f"{v('Y')} − {v('e')}"]
    elif k == "YH":
        parts = [s, "Y − e/2", f"{v('Y')} − {v('e')}/2"]
    elif k == "M_bras_j":
        parts = [s, f"R·({noms[1]} + Xj) / 1000", f"{v('R')} × ({v(noms[1])} + {v('Xj')}) / 1000"]
    elif k == "Mmax_j":
        parts = [s, "R·(X + Xj) / 1000", f"{v('R')} × ({v('X')} + {v('Xj')}) / 1000"]
    elif k == "rho":
        parts = [s, f"({noms[0]} + Yj) / (YH + Yj)", f"({v(noms[0])} + {v('Yj')}) / ({v('YH')} + {v('Yj')})"]
    elif k == "M_rho":
        parts = [s, f"{noms[0]}·Mmax", f"{v(noms[0])} × {v('Mmax')}"]
    elif k == "H_hist":
        parts = [s, "Mmax·1000 / (YH + Yj)", f"{v('Mmax')} × 1000 / ({v('YH')} + {v('Yj')})"]
    elif k == "M_hist_v":
        parts = [s, "H_s·(Y′ − s) / 1000", f"{v('H_s')} × ({v('Y′')} − {v('s')}) / 1000"]
    elif k == "M_bras":
        parts = [s, f"R·{noms[1]} / 1000", f"{v('R')} × {v(noms[1])} / 1000"]
    elif k == "M_lin":
        parts = [s, "R·(X′ − s) / 1000", f"{v('R')} × ({v('X′')} − {v('s')}) / 1000"]
    elif k == "bras_A":
        parts = [s, "Y′ − y_c", f"{v('Y′')} − {v('y_c')}"]
    elif k == "H_eq":
        parts = [s, "R·(X′ − u_V) / bras", f"{v('R')} × ({v('X′')} − {v('u_V')}) / {v('bras')}"]
    elif k == "M_eq_A":
        parts = [s, "(H·(Y′ − s) − R·(e/2 − u_V)) / 1000",
                 f"({v('H')} × ({v('Y′')} − {v('s')}) − {v('R')} × ({v('e')}/2 − {v('u_V')})) / 1000"]
    elif k == "y_a":
        parts = [s, "haut − Y′", f"{v('haut')} − {v('Y′')}"]
    elif k == "bras_B":
        parts = [s, "y_a − c", f"{v('y_a')} − {v('c')}"]
    elif k == "M_angle_B":
        parts = [s, "R·(X′ − e/2) / 1000", f"{v('R')} × ({v('X′')} − {v('e')}/2) / 1000"]
    elif k == "M_bas_B":
        parts = [s, "H·(s_c − s) / 1000", f"{v('H')} × ({v('s_c')} − {v('s')}) / 1000"]
    elif k == "t_conge":
        parts = [s, "e + r − √(r² − (e + r − s)²)", f"{v('e')} + {v('r')} − √({v('r')}² − ({v('e')} + {v('r')} − {v('s')})²)"]
    elif k == "W":
        parts = [s, "b·t² / 6", f"{v('b')} × {v('t')}² / 6"]
    elif k == "A":
        parts = [s, "b·t", f"{v('b')} × {v('t')}"]
    elif k == "sigma_M":
        parts = [s, "M·10⁶ / W", f"{v('M')} × 10⁶ / {v('W')}"]
    elif k == "sigma_MN":
        parts = [s, "|M|·10⁶ / W + |N|·10³ / A", f"|{v('M')}| × 10⁶ / {v('W')} + |{v('N')}| × 10³ / {v('A')}"]
    elif k == "tau_RLt":
        parts = [s, "R·10³ / (b·t)", f"{v('R')} × 10³ / ({v('b')} × {v('t')})"]
    elif k == "tau_VA":
        parts = [s, "|V|·10³ / A", f"|{v('V')}| × 10³ / {v('A')}"]
    elif k == "sigma_eq":
        parts = [s, "√(σ² + 3·τ²)", f"√({v('σ')}² + 3 × {v('τ')}²)"]
    elif k == "taux":
        parts = [s + (f" ({f['cellule']})" if f.get("cellule") else ""), f"{noms[0]} / {noms[1]}", f"{v(noms[0])} / {v(noms[1])}"]
        return {"parts": parts, "res": _pct(f["resultat"]), "niveau": C.niveau(f["resultat"])}
    elif k == "N_ELS_hist":
        parts = [s, "Mmax·1000 / ((Y − jeu)·n)", f"{v('Mmax')} × 1000 / (({v('Y')} − {v('jeu')}) × {v('n')})"]
    elif k == "N_ELS_eq":
        parts = [s, "H / n", f"{v('H')} / {v('n')}"]
    elif k == "V_ELS":
        parts = [s, "R / n", f"{v('R')} / {v('n')}"]
    elif k == "rapport_ELU":
        parts = [s, "R_ELU / R", f"{v('R_ELU')} / {v('R')}"]
    elif k == "ELU":
        parts = [s, f"{noms[0]}·ρ_ELU", f"{v(noms[0])} × {v('ρ_ELU')}"]
    elif k == "R_ELU_auto":
        parts = [s, "1,5·R", f"1,5 × {v('R')}"]
    elif k == "p_diam":
        parts = [s, "V_ELS·10³ / (d·e)", f"{v('V_ELS')} × 10³ / ({v('d')} × {v('e')})"]
    elif k == "arrachement":
        parts = [s, "V_ELS·10³ / (2·(haut − Y)·e)", f"{v('V_ELS')} × 10³ / (2 × ({v('haut')} − {v('Y')}) × {v('e')})"]
    elif k == "L_eff":
        parts = [s, "min(L ; a + 2·Xe′)", f"min({v('L')} ; {v('a')} + 2 × {v('Xe′')})"]
    elif k == "d0":
        parts = [s, "d + 2", f"{v('d')} + 2"]
    elif k == "b_net":
        parts = [s, "L − n·d0", f"{v('L')} − {v('n')} × {v('d0')}"]
    elif k == "pression_support":
        parts = [s, "C·10³ / L", f"{v('C')} × 10³ / {v('L')}"]
    else:                                                        # pragma: no cover — une formule nouvelle du moteur
        parts = [s, " · ".join(f"{n} = {v(n)}" for n in noms)]
    if f.get("cellule"):
        parts[0] = f"{parts[0]} ({f['cellule']})"
    return {"parts": parts, "res": _nu(f["resultat"], f["unite"]), "niveau": ""}


# les formules qui mènent au MOMENT d'une coupe (la fenêtre « efforts ») ; les autres sont celles de la coupe
MOMENT = frozenset(("Xp", "Yp", "Xe", "Xer", "Ye", "Yer", "YH", "M_bras_j", "Mmax_j", "rho", "M_rho", "H_hist", "M_hist_v", "M_bras",
                    "M_lin", "bras_A", "H_eq", "M_eq_A", "y_a", "bras_B", "M_angle_B", "M_bas_B"))


def _groupe(titre: str, formules: List[Dict], sous: str = "") -> str:
    """Un groupe du détail d'une fenêtre : son titre (texte), sa combinaison (texte), ses lignes."""
    if not formules:
        return ""
    st = f'<span class="ck-comb">{C.esc(sous)}</span>' if sous else ""
    return f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}{st}</div>{C.lignes_fx([_fx(f) for f in formules])}</div>'


def _bloc_calcul(res: Dict, cle: str) -> Optional[Dict]:
    return next((b for b in res["calcul"] if b["cle"] == cle), None)


# ------------------------------------------------------------------ les mots du résultat
# (la 1ʳᵉ ligne d'un résultat ne passe jamais à la ligne, dans 316 px : des noms courts ; le lieu — pied du congé, nu de
# l'autre aile, s et t — se lit en 2ᵉ ligne et dans les fenêtres)
COURTS = {"S1": "Section 1", "S2": "Section 2", "S3": "Section 3", "S4": "Section 4", "max_h": "Max. du congé", "max_v": "Max. du congé",
          "nette": "Section nette"}
# le nom COURT d'un mode dans la 1ʳᵉ ligne d'un résultat (bloc Comparaison : « Mode historique (P8) ») — « Mode méthode
# historique (classeur P8) » y était coupé à l'ellipse dans les 316 px de la colonne (mesuré le 11/11/2026, relecture de
# l'écran : 241 px demandés pour 185 au PC) ; le libellé entier (entree.LIBELLES_MODE) reste partout ailleurs
MODES_COURTS = {EN.HISTORIQUE: "historique (P8)", EN.EQUILIBRE: "équilibre"}
LIEUX = {"S1": "pied du congé", "S2": "nu de l'aile verticale", "S3": "pied du congé", "S4": "nu de l'aile horizontale",
         "max_h": "le long du congé", "max_v": "le long du congé", "nette": "au droit du trou"}
AILES = {MO.HORIZONTALE: "aile horizontale", MO.VERTICALE: "aile verticale"}
# le début de chaque avertissement du moteur, en gras ; ce qui ne commence pas par son titre le reçoit devant
TITRES_AVERTISSEMENTS = {"chevilles_non_verifiees": "Chevilles non vérifiées", "pic_conge": "Pic de contrainte au congé non vérifié",
                         "effet_levier": "Effet de levier non couvert", "pression_support": "Pression du bout libre sur le support à vérifier",
                         "equilibre_plus_severe": "Équilibre P1", "mode_impose_B": "Configuration B : mode équilibre imposé",
                         "R_ELU_auto": "R_ELU automatique", "note_v1": "Note d'avant le 10/11/2026, modèle lame", "fy_epaisseur": "Épaisseur"}
GRAVES = (MO.ROUGE,)
INFOS = {"pression_diametrale": "Pression diamétrale", "arrachement": "Arrachement", "compression_mur": "Compression du mur",
         "pression_support": "Pression du support", "autre_mode": "Autre mode", "L_eff": "Largeur efficace"}


def _mode_txt(res: Dict) -> str:
    return EN.LIBELLES_MODE[res["mode"]]


def _cfg_txt(res: Dict) -> str:
    return f"configuration {res['configuration']} · {EN.LIBELLES_CONFIGURATION[res['configuration']]}"


def _comb(cb: Optional[Dict]) -> str:
    """La combinaison de jeux qui gouverne une ligne, en texte échappé (« charge à X + 20, ancrage à Y − 10 »)."""
    return C.esc(cb["libelle"]) if cb else ""


def _lieu(v: Dict) -> str:
    """Où est la coupe : « aile verticale · pied du congé · s = 28 · t = 15 mm » (s depuis le dos de l'autre aile)."""
    return f"{AILES[v['aile']]} · {LIEUX.get(v['cle'], '')} · s = {C.nb(round(v['s'], 2), 2)} · t = {C.nb(round(v['t'], 2), 2)} mm"


def _coupe_txt(v: Dict) -> str:
    """« pied du congé · s = 28 · t = 15 mm » : le lieu de la coupe sans son aile (la ligne 1 la nomme déjà)."""
    return f"{LIEUX.get(v['cle'], '')} · s = {C.nb(round(v['s'], 2), 2)} · t = {C.nb(round(v['t'], 2), 2)} mm"


def _comparaison_coupe(v: Dict, lieu: bool = True) -> str:
    """La ligne 2 d'une vérification : le lieu de la coupe, σ_éq face à σ_adm, la combinaison qui gouverne."""
    le = "≤" if v["ok"] else "&gt;"
    return ((f"{C.esc(_coupe_txt(v))} · " if lieu else "")
            + f"σ<sub>éq</sub> = {_n(v['sigma_eq'], 'N/mm²')} {le} σ<sub>adm</sub> = {_n(v['sigma_adm'], 'N/mm²')}{C.NBSP}N/mm²"
            f" · {_comb(v['combinaison'])}")


# ------------------------------------------------------------------ le cadre Données : les quatre rubriques
def _duo(*cellules: Tuple[Optional[str], str, str, str]) -> str:
    """Deux champs sur une ligne (les données condensées, commun.ligne2), chacun une PIÈCE à part entière : ``(cible,
    libellé HTML, champ, aide texte)`` — la cible est la clé de sa pièce sur le dessin (data-cible = data-sel), None
    pour un champ que le dessin ne montre pas."""
    out = ""
    for cible, label, champ, aide in cellules:
        a = f'<span class="aide" title="{C.esc(aide)}">{C.icon("info")}</span>' if aide else ""
        dc = f' data-cible="{C.esc(cible)}"' if cible else ""
        out += f'<div class="et-piece"{dc}><label><span>{label}</span>{a}</label>{champ}</div>'
    return f'<div class="row2 et-cibles n{len(cellules)}">{out}</div>'


def _picto_configuration(cfg: str) -> str:
    """Le pictogramme d'une configuration : le mur à gauche, la cornière en L, la charge qui descend sur l'aile
    horizontale — en bas (A) ou en haut (B)."""
    if cfg == "A":
        corps, fleche = "M5 2.5V11.5H23", '<path d="M17 3v5" stroke="currentColor" stroke-width="1.3" fill="none"/><path d="M15 7.5l2 3 2-3z" fill="currentColor"/>'
    else:
        corps, fleche = "M5 12V4.5H23", '<path d="M17 0.5v2" stroke="currentColor" stroke-width="1.3" fill="none"/><path d="M15 1.5l2 2.5 2-2.5z" fill="currentColor"/>'
    return (f'<svg class="et-pic" viewBox="0 0 26 14" aria-hidden="true"><path d="M1.5 1V13" stroke="currentColor" stroke-width="1.1" '
            f'fill="none" opacity=".45"/><path d="{corps}" stroke="currentColor" stroke-width="2.4" fill="none" stroke-linejoin="round"/>{fleche}</svg>')


def _configuration(c: Dict) -> str:
    """La configuration A/B en segments, un pictogramme et un mot chacun (la poutre fait de même pour la répartition)."""
    bt = "".join(f'<button type="button" class="{"on" if k == c["configuration"] else ""}" data-chemin="{C._attr_chemin(["configuration"])}" '
                 f'data-sorte="choix" data-valeur="{k}" title="{C.esc(aide)}">{_picto_configuration(k)}<span>{t}</span></button>'
                 for k, t, aide in (("A", "A · en bas", "Aile horizontale en bas (le classeur) : l'aile verticale monte le long du mur, "
                                     "l'ancrage est au-dessus de l'angle, le talon s'appuie sur le mur."),
                                    ("B", "B · en haut", "Aile horizontale en haut : l'aile verticale descend, l'ancrage est au-dessous de "
                                     "l'angle, le bout libre s'appuie sur le mur ; toujours en mode équilibre.")))
    return f'<div class="seg full et-rep">{bt}</div>'


def _fiche(sec: Dict) -> str:
    """La fiche de la cornière du catalogue : ailes, épaisseur, congés, masse, aire — ce que la coupe ne cote pas."""
    lignes = [("Ailes A × B", f"{C.nb(sec['A'])} × {C.nb(sec['B'])}{C.NBSP}mm"), ("Épaisseur t", f"{C.nb(sec['t'])}{C.NBSP}mm"),
              ("Congés r<sub>1</sub> / r<sub>2</sub>", f"{C.nb(sec['r1'])} / {C.nb(sec['r2'])}{C.NBSP}mm"),
              ("Masse", C.qte(sec["masse"], "kg/m", 2)), ("Aire", C.qte(sec["aire"], "cm²", 2))]
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in lignes)
    fam = EN.LIBELLES_FAMILLE.get(sec.get("famille"), "")
    return (f'<div class="secard"><div class="sh2">{C.esc(sec["designation"])}<span class="muted">{C.esc(fam)} · {C.esc(sec["norme"] or "")}</span></div>'
            f'<dl class="facts">{dl}</dl></div>')


def _rubrique_corniere(c: Dict, sec: Dict) -> str:
    L = C.ligne
    s = L("Origine", C.segments(["standard"], "catalogue" if c["standard"] else "saisie", ORIGINES), full=True)
    if c["standard"]:
        s += (L("Profil", C.texte(["profil"], sec["designation"], "ex. L 80×80×8"), "", "La désignation tapée (« 80x80x8 », "
                "« 100 × 65 × 8 », les ailes dans un ordre ou dans l'autre) est relue au catalogue EN 10056-1 ; la liste des 233 "
                "cornières, triée par masse et calculée avec vos charges, s'ouvre depuis la ligne « Cornière » du résultat.", full=True)
              + _fiche(sec))
    else:
        s += (C.ligne2(("A", C.num(["A"], c["A"], "mm"), "La première aile (A au catalogue)."),
                       ("B", C.num(["B"], c["B"], "mm"), "La seconde aile."))
              + C.ligne2(("t", C.num(["t"], c["t"], "mm"), "L'épaisseur des ailes : e = t dans le calcul."),
                         ("r congé", C.num(["r"], c["r"], "mm"), "Le congé intérieur r₁ (0 : angle vif, le côté sûr) ; l'épaisseur dans le "
                          "congé vaut e + r − √(r² − (e + r − s)²)."))
              + C.aide(f"Masse et aire calculées sans les arrondis de bout d'aile : {C.qte(sec['masse'], 'kg/m', 2)} · {C.qte(sec['aire'], 'cm²', 2)}."))
    ailes = [("A", f"A = {C.nb(sec['A'])} mm"), ("B", f"B = {C.nb(sec['B'])} mm")]
    s += L("Aile", C.segments(["aile_horizontale"], c["aile_horizontale"], ailes), "",
           "L'aile HORIZONTALE : laquelle des deux ailes porte la charge (l'autre est fixée au mur) ; A est la première aile du nom. "
           "Les ailes égales ne changent rien, les ailes inégales (LI) changent tout.", full=True)
    piece = f'<div class="et-piece et-plat" data-cible="profil">{s}</div>'
    cfg = (f'<div class="row full"><label><span>Pose</span><span class="aide" title="La pose de la cornière (sa configuration). A : l\'aile horizontale en bas, celle du classeur '
           f'(l\'aile verticale monte, l\'ancrage est au-dessus de l\'angle). B : l\'aile horizontale en haut (l\'aile verticale descend, '
           f'l\'ancrage est au-dessous, le bout libre s\'appuie sur le mur) — les formules du classeur ne s\'y appliquent pas : B se calcule '
           f'toujours en équilibre. Y se compte depuis la face extérieure de l\'aile horizontale dans les deux cas : la même cote pour le même '
           f'trou quand on retourne la cornière.">{C.icon("info")}</span></label>{_configuration(c)}</div>')
    mat = C.ligne2(("Nuance", C.choix(["fy"], int(c["fy"]) if float(c["fy"]).is_integer() else c["fy"], NUANCES),
                    "La nuance de l'acier (f_y pour t ≤ 40 mm, EN 1993-1-1 Tableau 3.1) : σ_adm = f_y / coef."),
                   ("coef", C.num(["coef"], c["coef"], ""),
                    "Le coefficient global sur f_y de la méthode aux contraintes admissibles (1,5 dans le classeur) : σ_adm = f_y/coef, "
                    "τ_adm = σ_adm/√3, sur les efforts ELS non majorés — ce n'est pas une justification EC3 aux états limites."))
    return piece + cfg + mat


def _rubrique_charge(c: Dict, res: Dict) -> str:
    anc, g = res["ancrages"], res["geometrie"]
    return (_duo(("R", "R", C.num(["R"], c["R"], "kN"), "La réaction d'appui (ELS, non majorée) posée sur l'aile horizontale, sur la longueur "
                  "reprise L. Les n ancrages se partagent R en cisaillement (V = R/n)."),
                 ("L", "L reprise", C.num(["L"], c["L"], "mm"), "La longueur de cornière qui reprend R : la largeur de toutes les sections "
                  "vérifiées, et la longueur sur laquelle les n ancrages se répartissent (entraxe L/n, le premier à L/(2n))."))
            + _duo(("X", "X charge", C.num(["X"], c["X"], "mm"), "La distance du nu du mur (le dos de l'aile verticale) à la ligne d'action "
                    "de R. Elle doit porter sur la partie droite de l'aile horizontale : e + r < X − dx, et X + dx ≤ aile."),
                   ("dx", "± dx jeu", C.num(["dx"], c["dx"], "mm"), "Le jeu de position de la charge, dans les DEUX sens : le moteur calcule "
                    "la cornière pour la charge à X − dx, X et X + dx (trois positions) et chaque grandeur vérifiée garde la plus défavorable — "
                    "l'ENVELOPPE, avec la combinaison qui la donne. 0 : le classeur exactement (Xj = 0 dans la note validée)."))
            + _duo(("a", "a appui", C.num(["a"], c["a"], "mm", auto=f"auto · L = {C.nb(c['L'])}"), "La largeur d'appui réelle de la charge ; "
                    "vide = L. Si a + 2·Xe < L, l'aile horizontale ne travaille que sur L_eff = min(L ; a + 2·Xe) (diffusion à 45° jusqu'au "
                    f"pied du congé, section 1) et σ y est majorée de L/L_eff. Aujourd'hui L_eff = {C.nb(round(g['L_eff'], 1), 1)} mm."),
                   (None, "R<sub>ELU</sub>", C.num(["R_ELU"], c["R_ELU"], "kN", auto=f"auto · 1,5·R = {C.nb(round(anc['R_ELU'], 3), 3)}"),
                    "La réaction ELU, pour les efforts d'ancrage à donner à Hilti PROFIS (N_ELU, V_ELU = N_ELS, V_ELS × R_ELU/R) ; vide : "
                    "1,5·R, affiché « auto » — à remplacer par la combinaison réelle 1,35·G + 1,5·Q.")))


def _rubrique_ancrage(c: Dict, res: Dict) -> str:
    anc = res["ancrages"]
    return (_duo(("Y", "Y ancrage", C.num(["Y"], c["Y"], "mm"), "La distance de la FACE EXTÉRIEURE de l'aile horizontale (le dessous en A, "
                  "le dessus en B) à l'axe de l'ancrage : la même cote pour le même trou quand on retourne la cornière. Le trou doit rester "
                  "dans la partie droite de l'aile verticale : Y − dy − d0/2 ≥ e + r et Y + dy + d0/2 ≤ aile."),
                 ("dy", "± dy tolér.", C.num(["dy"], c["dy"], "mm"), "La tolérance de position de l'ancrage (perçage), dans les DEUX sens : "
                  "l'ancrage est calculé à Y − dy, Y et Y + dy et chaque grandeur garde la plus défavorable des trois — en A l'ancrage haut "
                  "charge les sections 3 et 4 et l'ancrage bas la traction ; l'enveloppe le trouve seule. 0 : le classeur exactement."))
            + _duo(("n", "n ancrages", C.num(["n"], c["n"], "", entier=True, mini=1), "Le nombre d'ancrages sur L, à l'entraxe L/n, le premier "
                    f"à L/(2n) du bord : z = {' · '.join(C.nb(z) for z in anc['positions_z'])} mm. Ils se partagent R et la traction H."),
                   ("d", "Ø d", C.num(["d"], c["d"], "mm"), f"Le diamètre de l'ancrage ; trou de passage d0 = d + 2 = {C.nb(anc['d0'])} mm "
                    "(section nette en B, pression diamétrale, position du trou dans l'aile).")))


def _rubrique_hypotheses(c: Dict, res: Dict) -> str:
    g, cfg_B = res["geometrie"], res["configuration"] == "B"
    modes = [(EN.HISTORIQUE, "Historique · classeur P8"), (EN.EQUILIBRE, "Équilibre")]
    aide_mode = ("Historique : les formules de la feuille P8 du classeur mot pour mot (pivot e/2, τ = R/(L·t) dans les deux ailes, traction "
                 "d'ancrage sur le bras Y − jeu), chaque grandeur avec sa cellule. Équilibre : un seul corps libre, pivot y_c et ligne "
                 "d'action u_V nommés, N et V exacts dans chaque aile, le maximum cherché le long du congé. Le résultat montre en information "
                 "le taux de l'autre mode.")
    s = C.ligne("Mode", C.segments(["mode"], c["mode"], modes, desactive=cfg_B), "", aide_mode, full=True)
    if cfg_B:
        s += C.aide("<b>Configuration B : toujours l'équilibre.</b> Les formules du classeur ne s'appliquent pas à la cornière retournée "
                    "(elles donneraient −18 % sur M3) ; le mode demandé est gardé pour le jour où la cornière repasse en A.")
    s += _duo(("jeu_ancrage", "jeu C76", C.num(["jeu_ancrage"], c["jeu_ancrage"], "mm"), "Mode historique : le « C76 » du classeur (10 mm), "
               f"retranché à Y pour le bras de la traction d'ancrage : N_ELS = Mmax/((Y − jeu)·n) ; aujourd'hui Y − jeu = {C.nb(c['Y'] - c['jeu_ancrage'])} mm."),
              ("yc", "y<sub>c</sub> pivot", C.num(["yc"], c["yc"], "mm"), "Mode équilibre : la position de la résultante de compression sur le "
               "mur — en A depuis la face extérieure de l'aile horizontale (0 = l'arête, ce que trouve le maillage sur un support raide ; e/2 "
               "redonne l'équilibre P1, proche du classeur) ; en B depuis le bout libre (0 = le bout libre)."))
    s += _duo(("uV", "u<sub>V</sub>", C.num(["uV"], c["uV"], "mm", auto=f"auto · e/2 = {C.nb(g['e'] / 2)}"), "Mode équilibre : la ligne "
               "d'action du cisaillement d'ancrage depuis le nu du mur ; vide = e/2, l'axe de l'aile verticale. Entre 0 et e."))
    return s


def _resume_hyp(c: Dict, res: Dict) -> str:
    mode = "équilibre" if res["mode"] == EN.EQUILIBRE else "historique"
    if res["configuration"] == "B":
        return f"équilibre (imposé en B) · y<sub>c</sub> {C.nb(c['yc'])}"
    return f"{mode} · jeu {C.nb(c['jeu_ancrage'])}" if res["mode"] == EN.HISTORIQUE else f"{mode} · y<sub>c</sub> {C.nb(c['yc'])} · u<sub>V</sub> {C.nb(res['geometrie']['uV'])}"


def formulaire(etat: Dict, res: Optional[Dict] = None) -> str:
    """Les quatre rubriques de la cornière ouverte : 1 Cornière, 2 Charge, 3 Ancrage, 4 Hypothèses (secondaire). ``res`` :
    le calcul de la cornière ouverte (``MO.verifier``), refait s'il manque."""
    c = _corniere(etat)
    res = res or MO.verifier(c)
    sec = res["section"]
    r1 = f"{C.esc(sec['designation'])} · {C.esc(res['materiau']['nuance'] or C.nb(c['fy']))} · {res['configuration']}"
    r2 = f"R {C.nb(c['R'])} kN · L {C.nb(c['L'])} · X {C.nb(c['X'])} ± {C.nb(c['dx'])}"
    r3 = f"Y {C.nb(c['Y'])} ± {C.nb(c['dy'])} · {c['n']} Ø{C.nb(c['d'])}"
    return (C.section(1, "Cornière", r1, _rubrique_corniere(c, sec), True, "co")
            + C.section(2, "Charge", r2, _rubrique_charge(c, res), True, "ch")
            + C.section(3, "Ancrage", r3, _rubrique_ancrage(c, res), False, "an")
            + C.section(4, "Hypothèses", _resume_hyp(c, res), _rubrique_hypotheses(c, res), False, "hy", secondaire=True))


def _dims(c: Dict, res: Dict) -> str:
    """« · L 120×120×15 · S355 · A » : les dimensions d'une cornière dans la liste de la note (HTML)."""
    sec = res.get("section") or {}
    des = sec.get("designation") or (f"{C.nb(c['A'])} × {C.nb(c['B'])} × {C.nb(c['t'])}" if not c.get("standard") else c.get("profil", ""))
    return f"· {C.esc(des)} · {res.get('configuration', c.get('configuration', ''))}"


def arbre(etat: Dict, res: Optional[Dict] = None, resultats: Optional[List[Dict]] = None) -> str:
    """La colonne de gauche COMPACTE (api_web/etabli_compact.arbre_compact, la même que la poutre et la dalle) : la ligne de
    la cornière ouverte (son nom, qu'un clic modifie ; la liste des cornières « nom · désignation · configuration », sans
    taux ; copier, supprimer), posée par l'écran dans l'en-tête du panneau, puis ses quatre rubriques."""
    resultats = resultats or [_calculer(x) for x in etat["cornieres"]]
    elements = [(c.get("nom") or f"C{i + 1}", _dims(c, r), False) for i, (c, r) in enumerate(zip(etat["cornieres"], resultats))]
    return E.arbre(CORNIERE, elements, etat["actif"], formulaire(etat, res), "corniere", compact=True)


# ------------------------------------------------------------------ le résultat
def _calculer(c: Dict) -> Dict:
    try:
        return MO.verifier(c)
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE (l'accueil, la liste des notes) : sa cornière la plus chargée."""
    vals = [(r["taux_max"], i) for i, r in enumerate(resultats) if "taux_max" in r]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    r = resultats[i]
    niv = _niv_res(r)
    g = r["gouvernant"]
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['cornieres'][i].get('nom') or 'Cornière'} · {COURTS.get(g['cle'], g['nom'])} · {AILES[g['aile']]}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les cornières de la note sont la LIGNE de l'élément et les rubriques de la colonne de gauche (``arbre``)."""
    return ""


def _avertissement(a: Dict) -> str:
    """Un avertissement du moteur, son titre en gras (« Chevilles non vérifiées : … »), tout le texte échappé ; un
    message qui commence par son titre (à la capitale près) ne le répète pas."""
    msg, titre = C.esc(a["message"]), C.esc(TITRES_AVERTISSEMENTS.get(a["code"], ""))
    if titre and msg[:len(titre)].lower() == titre.lower():
        return f"<b>{titre}</b>{msg[len(titre):]}"
    return f"<b>{titre}</b> — {msg}" if titre else msg


def _verdict(res: Dict) -> str:
    g = res["gouvernant"]
    vg = next(v for v in res["verifications"] if v["cle"] == g["cle"])
    niv = _niv_res(res)
    return (f'<div class="verdict" data-calcul="s0" title="Voir le calcul">{C.chip(C.pct(res["taux_max"]), niv, xl=True)}<div class="gov">'
            f'<b>{C.esc(COURTS.get(g["cle"], g["nom"]))} · {C.esc(AILES[g["aile"]])} · {C.esc(LIEUX.get(g["cle"], ""))}</b>{_comparaison_coupe(vg, lieu=False)}'
            f'<br>{C.esc(_cfg_txt(res))} · {C.esc(_mode_txt(res))}</div></div>')


def _bloc_sections(res: Dict) -> Tuple[str, str]:
    """Le bloc « Sections » : la géométrie et les efforts (deux lignes qui ouvrent leurs fenêtres), puis S1 à S4, les
    maxima du congé et la section nette — deux lignes par vérification, la combinaison de jeux qui la gouverne."""
    g, ef, mat = res["geometrie"], res["efforts"], res["materiau"]
    hist = res["mode"] == EN.HISTORIQUE
    # la ligne « Cornière » : sa pièce ouvre le SÉLECTEUR (data-pk), la ligne la fenêtre « géométrie et matériau »
    rows = C.ligne_verif("Cornière", f"e = {C.nb(g['e'])} · r = {C.nb(g['r'])} · {C.nb(g['larg'])} × {C.nb(g['haut'])} mm · "
                         f"σ<sub>adm</sub> = {_n(mat['sigma_adm'], 'N/mm²')} · τ<sub>adm</sub> = {_n(mat['tau_adm'], 'N/mm²')} N/mm²",
                         None, "c", calcul="geometrie", pk="corniere", piece=C.esc(res["section"]["designation"]),
                         etat_txt=f"{len(res['combinaisons'])} comb.")
    ef_txt = f"M<sub>max</sub> = {_n(ef['Mmax'], 'kN·m')} kN·m · H = {_n(ef['H'], 'kN')} kN ({_comb(ef['combinaison_H'])})"
    if ef["M_angle"] is not None:
        ef_txt += f" · M<sub>angle</sub> = {_n(ef['M_angle'], 'kN·m')} kN·m"
    rows += C.ligne_verif("Efforts", ef_txt, None, "c", calcul="efforts", piece=f"R = {C.nb(res['corniere']['R'])}{C.NBSP}kN",
                          etat_txt="enveloppe")
    if res["corniere"]["a"] is not None:
        # la largeur d'appui : L_eff en information (du côté sûr, l'aile horizontale ne travaille que sur L_eff)
        inf = next(x for x in res["informations"] if x["cle"] == "L_eff")
        rows += C.ligne_verif(INFOS["L_eff"], f"L<sub>eff</sub> = {C.esc(inf['texte'])} · {_comb(inf['combinaison'])}", None, "c", calcul="geometrie",
                              piece=f"a = {C.nb(res['corniere']['a'])}{C.NBSP}mm", etat_txt=f"{C.nb(round(g['L_eff'], 1), 1)} mm")
    gov = res["gouvernant"]["cle"]
    for v in res["verifications"]:
        nom, piece = COURTS.get(v["cle"], v["nom"]), AILES[v["aile"]]
        if v["verdict"]:
            rows += C.ligne_verif(nom, _comparaison_coupe(v), C.pct(v["taux"]), _niv(v), calcul="s0", piece=piece, gov=v["cle"] == gov)
        else:
            # information, hors verdict (le classeur n'a pas les maxima du congé) : le taux dans la phrase, pas dans une pastille
            cmp_ = (f"s = {C.nb(round(v['s'], 2), 2)} · t = {C.nb(round(v['t'], 2), 2)} mm · σ<sub>éq</sub> = {_n(v['sigma_eq'], 'N/mm²')} N/mm², soit "
                    f"{_pct(v['taux'])} de σ<sub>adm</sub> ({'+' if v['ecart'] >= 0 else ''}{C.pc(C.pct(v['ecart'], 1))} % sur {v['reference']}, "
                    f"{C.nb(round(v['recul'], 2), 2)} mm avant le pied du congé) · {_comb(v['combinaison'])}")
            rows += C.ligne_verif(nom, cmp_, None, "c", calcul="s0", piece=piece, etat_txt="hors verdict")
    cls, etat_txt = ("ok", "Conforme") if res["statut"] == MO.OK else ("nok", "Non conforme")
    etat_txt += f" · {_pct(res['taux_max'])}"
    efforts = f"{len(res['verifications'])} vérifications · {C.esc(_mode_txt(res))}" + (" · maxima du congé en information" if hist else "")
    return E.bloc(0, "Sections", cls, etat_txt, efforts, rows, theme=True), ""


def _formule_ancrage(res: Dict, cle: str, symbole: Optional[str] = None) -> Optional[Dict]:
    """Une formule du bloc « ancrages » du moteur, par sa clé (et son symbole quand la clé est partagée : ELU)."""
    b = _bloc_calcul(res, "ancrages")
    return next((f for f in (b["formules"] if b else []) if f["cle"] == cle and (symbole is None or f["symbole"] == symbole)), None)


def _expression(f: Optional[Dict]) -> str:
    """« expression = valeurs » d'une formule du moteur (la ligne 2 d'un effort d'ancrage), en HTML."""
    if not f:
        return ""
    p = _fx(f)["parts"]
    return " = ".join(C.symb_math(x) for x in p[1:])


def _bloc_ancrages(res: Dict) -> str:
    """Le bloc « Ancrages » : N et V par ancrage (ELS, ELU), les positions ; puis les lignes d'INFORMATION, grises, hors
    verdict (pression diamétrale, arrachement, compression du mur ou pression sur le support)."""
    anc, c = res["ancrages"], res["corniere"]
    hist = res["mode"] == EN.HISTORIQUE
    comb = _comb(anc["combinaison_N"])
    rows = C.ligne_verif("Traction N · ELS", f"{_expression(_formule_ancrage(res, 'N_ELS_hist' if hist else 'N_ELS_eq'))} · {comb}", None, "c",
                         calcul="s1", piece="par ancrage", etat_txt=f"{_n(anc['N_ELS'], 'kN')} kN")
    rows += C.ligne_verif("Cisaillement V · ELS", f"{_expression(_formule_ancrage(res, 'V_ELS'))} · R = {C.nb(c['R'])} kN sur {c['n']} ancrage(s)",
                          None, "c", calcul="s1", piece="par ancrage", etat_txt=f"{_n(anc['V_ELS'], 'kN')} kN")
    elu = (f"R<sub>ELU</sub> = {C.nb(round(anc['R_ELU'], 3), 3)} kN{' (auto · 1,5·R)' if anc['R_ELU_auto'] else ' (saisie)'}"
           + (f" · ρ<sub>ELU</sub> = {_n(anc['rapport_ELU'], '')}" if anc["rapport_ELU"] is not None else ""))
    rows += C.ligne_verif("Traction N · ELU", f"{_expression(_formule_ancrage(res, 'ELU', 'N_ELU'))} · {elu} · pour Hilti PROFIS", None, "c",
                          calcul="s1", piece="par ancrage", etat_txt=f"{_n(anc['N_ELU'], 'kN')} kN")
    rows += C.ligne_verif("Cisaillement V · ELU", f"{_expression(_formule_ancrage(res, 'ELU', 'V_ELU'))} · pour Hilti PROFIS", None, "c",
                          calcul="s1", piece="par ancrage", etat_txt=f"{_n(anc['V_ELU'], 'kN')} kN")
    rows += C.ligne_verif("Positions", f"z = {' · '.join(C.nb(z) for z in anc['positions_z'])} mm sur L = {C.nb(c['L'])} (le premier à L/(2n)) · "
                          f"trou Ø{C.nb(anc['d0'])} · bras de la traction {C.nb(round(anc['bras'], 2), 2)} mm", None, "c", calcul="s1",
                          piece=f"{c['n']} × Ø{C.nb(c['d'])}", etat_txt=f"entraxe {C.nb(c['L'] / c['n'])}")
    for inf in res["informations"]:
        if inf["cle"] in ("autre_mode", "L_eff"):
            continue
        val = f"{_n(inf['valeur'], inf['unite'])} {inf['unite']}".strip()
        texte = C.esc(inf["texte"]) + (f" · {_comb(inf['combinaison'])}" if inf["combinaison"] else "")
        rows += C.ligne_verif(INFOS.get(inf["cle"], inf["nom"]), texte, None, "c", calcul="s1", piece="hors verdict", etat_txt=val)
    efforts = f"H = {_n(anc['H'], 'kN')} kN ({comb}) · R = {C.nb(c['R'])} kN · {C.esc(EN.LIBELLES_MODE[anc['methode']])}"
    # l'état de l'en-tête est COURT : « efforts par ancrage · chevilles non vérifiées » (254 px) réduisait le titre
    # « Ancrages » à 6 px au PC, 0 à la tablette, 21 au téléphone — le bloc n'avait plus de nom (mesuré le 11/11/2026)
    return E.bloc(1, "Ancrages", "inc", "chevilles non vérifiées", efforts, rows, theme=True)


def _bloc_comparaison(res: Dict) -> str:
    """Le bloc « Comparaison » (configuration A) : le taux maximal de l'autre mode, en information ; en mode historique,
    la GARDE (l'équilibre P1, rouge quand il est plus sévère sur l'aile verticale)."""
    cp, gd = res["comparaison"], res["garde"]
    if cp is None and gd is None:
        return ""
    rows = ""
    if cp is not None:
        rows += C.ligne_verif(f"Mode {MODES_COURTS[cp['mode']]}", f"gouvernant : {C.esc(COURTS.get(cp['gouvernant']['cle'], cp['gouvernant']['nom']))} · "
                              f"le même calcul dans l'autre mode, pour voir l'écart", None, "c", calcul="s2", piece="information", etat_txt=_pct(cp["taux_max"]))
    if gd is not None:
        niv = "r" if gd["depasse"] else "c"
        rows += C.ligne_verif("Garde · équilibre P1", f"pivot y<sub>c</sub> = {C.nb(gd['yc'])}, u<sub>V</sub> = {C.nb(gd['uV'])} mm : {_pct(gd['taux_equilibre_P1'])} "
                              f"{'&gt;' if gd['depasse'] else '≤'} {_pct(gd['taux_historique'])} (méthode historique, aile verticale)"
                              + (" — la méthode historique n'est plus du côté sûr" if gd["depasse"] else ""), None, niv, calcul="s2",
                              etat_txt="plus sévère" if gd["depasse"] else "du côté sûr")
    cls = "nok" if gd is not None and gd["depasse"] else "inc"
    # le titre est COURT (« Comparaison », le mot des décisions § 4) : avec « garde dépassée » à côté, « Comparaison des
    # modes » perdait sa fin à l'ellipse au PC et à la tablette (173 px de texte pour 166 et 154, mesuré le 11/11/2026)
    return E.bloc(2, "Comparaison", cls, "information" if cls == "inc" else "garde dépassée",
                  f"{C.esc(_mode_txt(res))} face à l'autre mode · configuration A", rows, theme=True)


def _hypotheses_html(res: Dict) -> str:
    return '<div class="et-hyps">' + "".join(C.aide(C.esc(h)) for h in res["hypotheses"]) + "</div>"


def _messages(res: Dict, sans: Sequence[str] = ()) -> str:
    """Les avertissements du moteur, titrés, le grave en tête (rouge, puis attention, puis information) ; ``sans`` : les
    codes qu'on ne redit pas (l'avis d'ouverture d'une note v1 vient d'être dit, en tête du résultat)."""
    av = sorted((a for a in res["avertissements"] if a["code"] not in sans), key=lambda a: (a["niveau"] not in GRAVES, a["niveau"] != MO.ATTENTION))
    return "".join(C.message(_avertissement(a), a["niveau"] in GRAVES) for a in av)


# ------------------------------------------------------------------ les fenêtres de calcul
def _ligne_tableau(v: Dict, gov: bool) -> Dict:
    f = "σ<sub>éq</sub> = √(σ² + 3τ²) ≤ σ<sub>adm</sub>"
    li = {"nom": C.esc(COURTS.get(v["cle"], v["nom"])), "sous": C.esc(f"{_lieu(v)} · {v['combinaison']['libelle']}"), "formule": f,
          "valeur": f"σ<sub>éq</sub> = {_n(v['sigma_eq'], 'N/mm²')} N/mm²", "limite": f"σ<sub>adm</sub> = {_n(v['sigma_adm'], 'N/mm²')} N/mm²",
          "taux": C.pct(v["taux"]), "niveau": _niv(v), "gov": gov}
    if not v["verdict"]:
        li["sous"] += " · information, hors verdict"
        li["niveau"] = "c"
    return li


def _hyps(res: Dict) -> List[str]:
    c, m = res["corniere"], res["materiau"]
    return [f"Cornière <b>{C.esc(res['section']['designation'])}</b> · e = {C.nb(res['geometrie']['e'])} · r = {C.nb(res['geometrie']['r'])} mm · "
            f"aile horizontale {C.nb(res['geometrie']['larg'])}, verticale {C.nb(res['geometrie']['haut'])} mm",
            f"{C.esc(_cfg_txt(res))} · <b>{C.esc(_mode_txt(res))}</b>",
            f"{C.esc(m['nuance'] or '')} · σ<sub>adm</sub> = f<sub>y</sub>/{C.nb(c['coef'])} = {_n(m['sigma_adm'], 'N/mm²')} · τ<sub>adm</sub> = σ<sub>adm</sub>/√3 = "
            f"{_n(m['tau_adm'], 'N/mm²')} N/mm²",
            f"R = {C.nb(c['R'])} kN (ELS) sur L = {C.nb(c['L'])} mm · X = {C.nb(c['X'])} ± {C.nb(c['dx'])} · Y = {C.nb(c['Y'])} ± {C.nb(c['dy'])} mm · "
            f"{c['n']} Ø{C.nb(c['d'])} · {len(res['combinaisons'])} combinaison(s)"]


def _volets(res: Dict, nom: str) -> str:
    hist = res["mode"] == EN.HISTORIQUE
    hyps = _hyps(res)
    gov = res["gouvernant"]["cle"]
    sous = f"{C.esc(_cfg_txt(res))} · {C.esc(_mode_txt(res))}" + (" · la cellule du classeur entre parenthèses" if hist else "")
    # 1 · les sections : toutes les vérifications, puis le détail de chaque coupe (matériau d'abord)
    lignes = [_ligne_tableau(v, v["cle"] == gov) for v in res["verifications"]]
    detail = _groupe("Matériau · contraintes admissibles", _bloc_calcul(res, "materiau")["formules"])
    for v in res["verifications"]:
        b = _bloc_calcul(res, v["cle"])
        detail += _groupe(f"{COURTS.get(v['cle'], v['nom'])} · {_lieu(v)}", b["formules"], b["combinaison"]["libelle"] if b["combinaison"] else "")
    pied = ('<div class="ck-autres">Les maxima du congé sont calculés et montrés (vue Contraintes) mais restent en INFORMATION en mode '
            'historique : le classeur ne les a pas, et son cas de contrôle rend 88,86 % exactement.</div>' if hist else
            '<div class="ck-autres">En mode équilibre, le maximum le long de chaque aile (dans le congé, où l\'épaisseur t(s) décroît) entre '
            'dans le verdict avec les sections repères.</div>')
    out = C.fenetre_calcul("s0", f"{nom} · sections", sous, hyps, lignes, detail, pied)
    # 2 · les ancrages : les efforts par ancrage et les informations
    anc = res["ancrages"]
    comb = anc["combinaison_N"]["libelle"]
    la = [{"nom": "N<sub>ELS</sub>", "sous": C.esc(f"traction par ancrage · {comb}"), "formule": "H / n", "valeur": f"{_n(anc['N_ELS'], 'kN')} kN",
           "limite": "—", "taux": None, "niveau": "c"},
          {"nom": "V<sub>ELS</sub>", "sous": "cisaillement par ancrage", "formule": "R / n", "valeur": f"{_n(anc['V_ELS'], 'kN')} kN", "limite": "—",
           "taux": None, "niveau": "c"},
          {"nom": "N<sub>ELU</sub>", "sous": C.esc(f"R_ELU = {C.nb(round(anc['R_ELU'], 3), 3)} kN{' (auto)' if anc['R_ELU_auto'] else ''}"), "formule": "N<sub>ELS</sub> · R<sub>ELU</sub>/R",
           "valeur": f"{_n(anc['N_ELU'], 'kN')} kN", "limite": "—", "taux": None, "niveau": "c"},
          {"nom": "V<sub>ELU</sub>", "sous": "", "formule": "V<sub>ELS</sub> · R<sub>ELU</sub>/R", "valeur": f"{_n(anc['V_ELU'], 'kN')} kN", "limite": "—",
           "taux": None, "niveau": "c"}]
    for inf in res["informations"]:
        if inf["cle"] in ("autre_mode",):
            continue
        la.append({"nom": C.esc(INFOS.get(inf["cle"], inf["nom"])), "sous": C.esc(inf["texte"]), "formule": "information, hors verdict",
                   "valeur": f"{_n(inf['valeur'], inf['unite'])} {inf['unite']}".strip(), "limite": "—", "taux": None, "niveau": "c"})
    ba = _bloc_calcul(res, "ancrages")
    out += C.fenetre_calcul("s1", f"{nom} · ancrages", sous, hyps, la, _groupe("Efforts par ancrage", ba["formules"], comb),
                            '<div class="ck-autres">Chevilles non vérifiées : N et V par ancrage, ELS et ELU séparés, à donner à Hilti PROFIS ; '
                            'pression diamétrale et arrachement sans limite, comme le classeur.</div>',
                            colonnes=("Effort", "Formule", "Valeur par ancrage", "Limite", "Taux"))
    # 3 · la comparaison des modes (A)
    cp, gd = res["comparaison"], res["garde"]
    if cp is not None or gd is not None:
        lc = []
        if cp is not None:
            lc.append({"nom": C.esc(f"Mode {EN.LIBELLES_MODE[cp['mode']]}"), "sous": C.esc(COURTS.get(cp["gouvernant"]["cle"], cp["gouvernant"]["nom"])),
                       "formule": "le même calcul dans l'autre mode", "valeur": f"taux max {_pct(cp['taux_max'])}", "limite": "information",
                       "taux": None, "niveau": "c"})
        if gd is not None:
            lc.append({"nom": "Garde · équilibre P1", "sous": C.esc(f"y_c = {C.nb(gd['yc'])}, u_V = {C.nb(gd['uV'])} mm, aile verticale"),
                       "formule": "taux P1 ≤ taux historique", "valeur": f"{_pct(gd['taux_equilibre_P1'])}", "limite": f"{_pct(gd['taux_historique'])}",
                       "taux": None, "niveau": "r" if gd["depasse"] else "c", "etat_txt": "plus sévère" if gd["depasse"] else "du côté sûr"})
        texte = ('<div class="ck-autres">Dans chaque mode, le résultat montre en information le taux maximal de l\'autre mode (sur le cas de '
                 'contrôle du classeur : historique 88,86 %, équilibre P0 76,91 %). En mode historique, le moteur calcule aussi l\'équilibre P1 '
                 '(pivot e/2, u_V = e/2) : s\'il donne un taux plus grand sur l\'aile verticale (ancrage bas), la méthode historique n\'est plus '
                 'du côté sûr et un avertissement rouge le dit.</div>')
        out += C.fenetre_calcul("s2", f"{nom} · comparaison des modes", sous, hyps, lc, "", texte,
                                colonnes=("Mode", "Ce qu'on compare", "Valeur", "Référence", "Taux"))
    # 4 · la géométrie et le matériau
    g, sec = res["geometrie"], res["section"]
    src = f"{'catalogue ' + C.esc(sec['nom']) if sec['source'] == 'catalogue' else 'saisie libre'}"
    lg = [{"nom": "Aile horizontale", "sous": C.esc(f"aile {sec['aile_horizontale']} : porte la charge"), "formule": "larg", "valeur": f"{C.nb(g['larg'])} mm", "limite": src, "taux": None, "niveau": "c"},
          {"nom": "Aile verticale", "sous": "fixée au mur", "formule": "haut", "valeur": f"{C.nb(g['haut'])} mm", "limite": src, "taux": None, "niveau": "c"},
          {"nom": "Épaisseur", "sous": "", "formule": "e = t", "valeur": f"{C.nb(g['e'])} mm", "limite": src, "taux": None, "niveau": "c"},
          {"nom": "Congé", "sous": "r₁ · bloc d'angle sous s = e, congé de e à e + r", "formule": "r", "valeur": f"{C.nb(g['r'])} mm", "limite": src, "taux": None, "niveau": "c"}]
    for s_ in g["sections"]:
        lg.append({"nom": C.esc(COURTS.get(s_["cle"], s_["nom"])), "sous": C.esc(AILES[s_["aile"]]), "formule": "s depuis le dos de l'autre aile · t",
                   "valeur": f"s = {C.nb(s_['s'])} · t = {C.nb(s_['t'])} mm", "limite": "repère", "taux": None, "niveau": "c"})
    if res["corniere"]["a"] is not None:
        lg.append({"nom": "Largeur efficace", "sous": C.esc(f"a = {C.nb(res['corniere']['a'])} mm · {g['combinaison_L_eff']['libelle']}"), "formule": "min(L ; a + 2·Xe′)",
                   "valeur": f"L<sub>eff</sub> = {C.nb(round(g['L_eff'], 1), 1)} mm", "limite": f"σ × {C.nb(round(g['facteur_L_eff'], 3), 3)}", "taux": None, "niveau": "c"})
    dg = _groupe("Matériau · contraintes admissibles", _bloc_calcul(res, "materiau")["formules"])
    ble = _bloc_calcul(res, "largeur_efficace")
    if ble:
        dg += _groupe("Largeur efficace de l'aile horizontale", ble["formules"], ble["combinaison"]["libelle"] if ble["combinaison"] else "")
    combos = "".join(f"<li>{C.esc(cb['libelle'])} — X′ = {C.nb(cb['X'])}, Y′ = {C.nb(cb['Y'])} mm (charge {C.esc(cb['position_charge'])}, ancrage "
                     f"{C.esc(cb['position_ancrage'])})</li>" for cb in res["combinaisons"])
    out += C.fenetre_calcul("geometrie", f"{nom} · géométrie et matériau", sous, hyps, lg, dg,
                            f'<div class="ck-autres"><b>Les {len(res["combinaisons"])} combinaison(s) de jeux</b> (la nominale d\'abord ; chaque grandeur '
                            f'vérifiée garde la plus défavorable) :<ul>{combos}</ul></div>',
                            colonnes=("Grandeur", "Définition", "Valeur", "Source", "Taux"))
    # 5 · les efforts : les moments de chaque coupe et la traction d'ancrage, enveloppe
    ef = res["efforts"]
    le = [{"nom": "R", "sous": "réaction ELS", "formule": "saisie", "valeur": f"{C.nb(ef['R'])} kN", "limite": "—", "taux": None, "niveau": "c"},
          {"nom": "R<sub>ELU</sub>", "sous": "auto · 1,5·R" if ef["R_ELU_auto"] else "saisie", "formule": "1,5·R" if ef["R_ELU_auto"] else "saisie",
           "valeur": f"{C.nb(round(ef['R_ELU'], 3), 3)} kN", "limite": "—", "taux": None, "niveau": "c"},
          {"nom": "M<sub>max</sub>", "sous": "au nu du mur, enveloppe", "formule": "R·(X + dx)", "valeur": f"{_n(ef['Mmax'], 'kN·m')} kN·m", "limite": "—", "taux": None, "niveau": "c"},
          {"nom": "H", "sous": C.esc(f"traction totale des ancrages = compression du mur · {ef['combinaison_H']['libelle']}"), "formule": "enveloppe",
           "valeur": f"{_n(ef['H'], 'kN')} kN", "limite": f"bras {C.nb(round(ef['bras_H'], 2), 2)} mm", "taux": None, "niveau": "c"}]
    if ef["H_sections"] is not None:
        le.append({"nom": "H<sub>s</sub>", "sous": "le H implicite des sections du classeur", "formule": "M<sub>max</sub>/(YH + Yj)",
                   "valeur": f"{_n(ef['H_sections'], 'kN')} kN", "limite": "—", "taux": None, "niveau": "c"})
    if ef["M_angle"] is not None:
        le.append({"nom": "M<sub>angle</sub>", "sous": "B : constant de l'ancrage à l'angle", "formule": "R·(X′ − e/2)", "valeur": f"{_n(ef['M_angle'], 'kN·m')} kN·m",
                   "limite": "—", "taux": None, "niveau": "c"})
    de = ""
    for v in res["verifications"]:
        b = _bloc_calcul(res, v["cle"])
        de += _groupe(f"{COURTS.get(v['cle'], v['nom'])} · {_lieu(v)}", [f for f in b["formules"] if f["cle"] in MOMENT],
                      b["combinaison"]["libelle"] if b["combinaison"] else "")
    de += _groupe("Traction d'ancrage", [f for f in ba["formules"] if f["cle"] in MOMENT or f["cle"] in ("N_ELS_hist", "N_ELS_eq")], comb)
    out += C.fenetre_calcul("efforts", f"{nom} · efforts", sous, hyps, le, de,
                            '<div class="ck-autres">Chaque moment est l\'ENVELOPPE des combinaisons de jeux (la plus défavorable, nommée) ; '
                            'M > 0 : la face côté congé est tendue (tout positif en A, tout négatif en B).</div>',
                            colonnes=("Effort", "Origine", "Formule", "Valeur", "Taux"))
    return out


# ------------------------------------------------------------------ les vues
def _entete(res: Dict, titre: str, legende: str) -> str:
    c, ef = res["corniere"], res["efforts"]
    bande = [("R", "", ef["R"], "kN"), ("M", "max", ef["Mmax"], "kN·m"), ("H", "", ef["H"], "kN")]
    lignes = [f"{_cfg_txt(res)} · {_mode_txt(res)}",
              f"R = {C.nb(c['R'])} kN sur L = {C.nb(c['L'])} mm · X = {C.nb(c['X'])} ± {C.nb(c['dx'])} · Y = {C.nb(c['Y'])} ± {C.nb(c['dy'])} mm · "
              f"{c['n']} Ø{C.nb(c['d'])}"]
    sous = f"{C.esc(res['section']['designation'])} · {C.esc(res['materiau']['nuance'] or '')} · coef {C.nb(c['coef'])}"
    return E.entete(titre, sous, lignes, bande, legende)


def _vues(res: Dict, ident: str) -> List[Dict]:
    out = []
    for v in DC.vues(res, ident):
        v["entete"] = _entete(res, v["titre"].split(" · ")[0].upper(), v["legende"])
        out.append(v)
    return out


def resultat(etat: Dict) -> Dict:
    """Le résultat de la cornière ouverte : la tête (le nom, les dimensions), l'avis d'ouverture (CLE_AVIS, dit une fois et
    retiré de l'état ; dans « avis » aussi, pour le téléphone), le verdict, les trois blocs, les hypothèses, les
    avertissements, la masse ; les fenêtres de calcul, les vues, la colonne de gauche, le résumé de la note."""
    avis = [AVIS[k] for k in (etat.pop(CLE_AVIS, None),) if k in AVIS]
    c = _corniere(etat)
    r = MO.verifier(c)
    resultats = [r if i == etat["actif"] else _calculer(x) for i, x in enumerate(etat["cornieres"])]
    sec = r["section"]
    nom = C.esc(c.get("nom") or "Cornière")
    dims = f"{C.esc(sec['designation'])} · {C.esc(r['materiau']['nuance'] or '')} · {C.esc(_cfg_txt(r))} · {C.esc(_mode_txt(r))}"
    tete = E.tete_resultats(nom, dims)
    bs, _ = _bloc_sections(r)
    blocs = bs + _bloc_ancrages(r) + _bloc_comparaison(r)
    masse = f"{C.qte(sec['masse'], 'kg/m', 2)} ({'catalogue' if sec['masse_source'] == 'catalogue' else 'calculée'})"
    pied = (f'<div class="pfoot"><div class="mass"><span>Masse de la cornière</span><b>{masse}</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    nivs = [_niv_res(x) if "taux_max" in x else "x" for x in resultats]
    statut = C.statut(next((n for n in ("x", "r") if n in nivs), max(nivs, key=lambda n: {"p": 0, "f": 1}.get(n, 0)))) if len(resultats) > 1 else ""
    corps = (f'<div class="pbody">{tete}{"".join(C.message(a) for a in avis)}{_verdict(r)}<div class="et-blocs">{blocs}</div>'
             f'{_messages(r, sans=("note_v1",) if avis else ())}{_hypotheses_html(r)}</div>{pied}')
    return {"resultat": corps, **({"avis": avis} if avis else {}), "volets": _volets(r, nom), "vues": _vues(r, f"cor{etat['actif']}"),
            "phrase": "", "statut_html": statut, "resume": resume_note(etat, resultats), "donnees": arbre(etat, r, resultats),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ la note PDF
def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None) -> Optional[Dict]:
    """« note_pdf » : la note de calcul PDF de TOUTE la note (chaque cornière, deux pages chacune), faite par
    api_web/note_corniere.py (le lot « note ») ; ``infos`` : le nom, le chantier, l'indice et la date de la note. Tant
    que ce fichier n'est pas dans l'arbre, la demande est REFUSÉE avec un message clair (jamais une exception) ; l'avis
    d'ouverture (CLE_AVIS) n'entre pas dans la note."""
    if quoi != "note_pdf":
        return None
    try:
        from . import note_corniere as NC
    except ImportError:
        raise ValueError("La note de calcul PDF des cornières n'est pas encore disponible sur le site (api_web/note_corniere.py absent).") from None
    etat = {k: v for k, v in etat.items() if k != CLE_AVIS}
    return NC.fichier(etat, infos)


# ------------------------------------------------------------------ le sélecteur
def selecteur(etat: Dict, role: str, familles: Optional[Sequence[str]] = None, froid: bool = True) -> str:
    """Les 233 cornières du catalogue TRIÉES PAR MASSE, chacune calculée avec les charges, l'ancrage et les hypothèses de
    la cornière ouverte (``moteur.candidats``) : ✓ la retenue, ★ la plus légère qui tient ET dont la charge et l'ancrage
    tombent dans les ailes ; une cornière hors domaine est grise, « exclue » (sa raison au survol, pas de taux), une qui
    ne tient pas est rouge ; les familles cochées filtrent."""
    c = _corniere(etat)
    fams = tuple(f for f in (familles or EN.FAMILLES) if f in EN.FAMILLES) or EN.FAMILLES
    cands = MO.candidats(c, fams)
    i_o = next((k for k, x in enumerate(cands) if x["optimal"]), 0)
    i_c = next((k for k, x in enumerate(cands) if x["retenue"]), i_o)

    def prop(x):
        t = (f"{EN.LIBELLES_FAMILLE[x['famille']]} · EN 10056-1 · A × B × t = {C.nb(x['A'])} × {C.nb(x['B'])} × {C.nb(x['t'])} mm · "
             f"r<sub>1</sub> = {C.nb(x['r1'])} mm · aire {C.fr(x['aire'], 2)} cm²")
        if not x["domaine"]:
            t += f" · <b>hors domaine</b> : {C.esc(x['raison'])}"
        elif not x["ok"]:
            t += f" · <b>ne tient pas</b> ({_pct(x['taux'])})"
        return t

    def niveau(x):
        return "c" if not x["domaine"] else ("r" if not x["ok"] else C.niveau(x["taux"]))
    # hors domaine : pas de taux (la cornière n'est pas calculée) — une pastille grise « exclue » à sa place (52 px : « hors
    # domaine » n'y tient pas ; la raison est au survol), la ligne
    # inactive (le moteur refuserait de la retenir : son refus se lit si on la clique quand même)
    rows = [{"nom": x["designation"], "col": C.fr(x["masse"], 2), "taux": C.pct(min(x["taux"], 9.99)) if x["taux"] is not None else None,
             "niveau": niveau(x), "cur": x["retenue"], "opt": x["optimal"], "prop": prop(x), "etat": "" if x["domaine"] else "exclue",
             "inactif": not x["domaine"]}
            for x in cands]
    hors = sum(not x["domaine"] for x in cands)
    sous = (f"{len(cands)} cornières · par masse · calculées avec vos charges"
            + (f" · {hors} exclue(s), grisées : la charge ou l'ancrage hors de leurs ailes (la raison au survol)" if hors else ""))
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{f}">{EN.LIBELLES_FAMILLE[f]}</button>'
                      for f in EN.FAMILLES)
    return C.selecteur(role, "Cornière · profilé", sous, filtres, rows, max(0, min(i_o, i_c) - 4), "kg/m",
                       etoile="la plus légère qui tient, charge et ancrage dans ses ailes")
