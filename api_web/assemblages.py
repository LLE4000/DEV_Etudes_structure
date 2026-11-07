# -*- coding: utf-8 -*-
"""Le module « Assemblages métalliques » du site : une note = un PROJET d'assemblages (la page projet de Streamlit,
modules/assemblages.py et acier/projet.py) ; chaque élément de la note est un assemblage, de l'une des sept
configurations du registre (acier/registre.py) :

- poutre–poutre : plat d'âme soudé, doubles cornières, platine d'about (api_web/assemblages_poutre.py) ;
- poutre–poteau : les trois mêmes attaches sur l'aile ou dans l'âme d'un poteau (api_web/assemblages_poteau.py) ;
- pied de poteau sur platine béton, avec son calcul « platine à la main » (api_web/assemblages_pied.py).

Ce fichier est l'AIGUILLAGE et la CARTE COMMUNE : l'état de la note, la liste des assemblages (ajouter, copier,
supprimer), le menu « Configuration » en tête du cadre Données, puis la mise en forme commune à toutes les
configurations — le cadre Données (sections numérotées), le résultat (verdict, une ligne par pièce), les fenêtres de
calcul (tableau des vérifications et détail des formules), les sélecteurs (boulon, profilés, cornière, solutions du
prédimensionnement) et les vues de la toile (dessins, prédimensionnement, note texte, méthode, validation).

Chaque configuration est DÉCRITE par sa famille (``Config`` ci-dessous) : son moteur (``entrees``, ``moteur``,
``synthese``, ``formules``, ``notation``, ``texte``, ``methode``, ``benchmark``, ``schemas`` du dépôt — jamais
``interface.py``, ``ecran_*.py``, ``note.py``, ``carte.py`` ni ``ui_commun.py``, qui chargent streamlit ou
reportlab), ses libellés courts, ses sections, ses vues. Rien n'est recalculé ici : les taux, les formules, les
alertes, les dessins viennent du moteur ; ce fichier les MET EN FORME.

L'état d'un assemblage reprend celui de l'écran Streamlit : ``values`` porte les clés de ``entrees.defaults()``,
normalisées comme les widgets les rendent (``carte.valeur_widget`` : un nombre est un flottant ou None, un choix est
l'une de ses options, un texte est une chaîne) — c'est exactement ce que l'écran Streamlit passe au moteur
(tests/test_api_web_assemblages.py le prouve en faisant tourner l'écran Streamlit lui-même).
"""
import copy
import html
import math
import re
from typing import Callable, Dict, List, Optional, Sequence

from acier import projet as PJ
from acier import registre as RG
from acier.bibliotheques import DB, PERSO, find

from . import commun as C

MODULE = "assemblages"
PAGE = "assemblages"

VISSERIE_DEFAUT = "1 rondelle + 1 écrou"
NIVEAUX_COTATION = (("0", "Simple"), ("1", "Édition"), ("2", "Toutes"))   # « Vue simple », « Complète » : rognés à 1024 px
# les vues de la toile : leur nom court (les onglets du haut de la toile)
UNITES = {"kNm": "kN·m", "MPa": "N/mm²", "-": ""}
OUI_NON = ["Oui", "Non"]
# les nombres qui comptent (rangées, files de tiges) : entiers ≥ 1 — Streamlit les arrondit en silence
# (moteur : n1 = max(1, Math.round(n1))), le site refuse une saisie qui ne l'est pas
COMPTES = {"n1_u", "n1S_u", "n1P_u", "n_ax", "n_ay"}
# … et BORNÉS : l'écran Streamlit n'a pas de maximum, et le moteur pose un boulon par case — n₁ = 10⁹ rangées (une
# faute de frappe, un fichier abîmé) épuise la mémoire (mesuré : 14 Go, processus tué) et figerait l'onglet du
# navigateur. 50 rangées (une platine de 3 m au pas de 60 mm) et 20 × 20 tiges (40 × 40 : 6,6 s par calcul hors
# navigateur, huit calculs par ouverture du sélecteur des tiges) couvrent tous les assemblages réels.
COMPTES_MAXI = {"n1_u": 50, "n1S_u": 50, "n1P_u": 50, "n_ax": 20, "n_ay": 20}
# les grandeurs qui peuvent être négatives (signe = sens) ; les autres dimensions sont ≥ 0
SIGNEES = {"V_Ed", "N_Ed", "M_Ed", "H_Ed", "Mx_Ed", "My_Ed", "Vx_Ed", "Vy_Ed", "T_Ed", "xN", "yN", "d_top"}
# les coefficients et résistances strictement positifs (une division par zéro donnerait des taux absurdes)
POSITIVES = re.compile(r"^(g_M\w*|g_c|g_Mc|fy_u|fu_u|bw_u|eta_c|k_e1|k_p1|k_e2|k_rot|a_cc)$")
# les dimensions saisies d'un profil, d'une cornière, d'un acier personnalisés, reprises de la pièce qu'on quitte
PERSO_PROFIL = {"prof_P": "P", "prof_S": "S"}
PERSO_ACIER = (("fy_u", "fy"), ("fu_u", "fu"), ("bw_u", "bw"))


# ===================================================================== la description d'une configuration
class Config:
    """Ce qu'une famille confie à la carte commune pour une configuration (voir les fichiers assemblages_*.py).

    ``E`` entrees, ``calculer(u)`` → R, ``S`` synthese, ``N`` notation, ``T`` texte, ``methode`` (texte Markdown),
    ``bench`` (module benchmark ou None), ``vues(R, opt)`` → [(clé, titre, court, Dessin)], ``options_dessin(R,
    niveau, hl)``, ``sections(u, R)`` → [(clé, titre, résumé, [items], ouvert)], ``libelles`` {clé: (symbole,
    libellé)}, ``formules(R, c)`` → (aide, [lignes de C.lignes_fx]), ``parametres(R)`` → [[libellé, valeur]],
    ``non_reverifies`` (texte de l'écran Streamlit), ``piece(R, u, groupe)`` → (texte, rôle du sélecteur),
    ``predim`` {apply, label, nom} ou None, ``tableaux_boulons(R)`` → [(titre, liste d'EffortBoulon)],
    ``dimensions`` (valeurs provisoires des dimensions vides), ``segments`` (clés en segments), ``affichage``
    (options courtes), ``famille`` (clé de la famille du registre), ``blocs(R, groupe)`` (HTML de plus dans la
    fenêtre d'une pièce : la matrice aile A · aile B des cornières). ``predim`` porte aussi ``detail(p)`` (le
    tableau « Détail de la solution retenue par défaut » de l'onglet Prédim), ``regle`` (sa légende) et
    ``geo(r)`` (les pinces d'une solution) ; ses ``colonnes`` peuvent être une fonction du résultat.

    Facultatifs (famille poutre–poteau) : ``noms_aciers`` {« P »: « poteau », …} (les aciers des hypothèses),
    ``sans_objet(R)`` (ce que l'écran ajoute au « sans objet » de son repli Périmètre), ``regle_33`` (la règle du
    Tableau 3.3 que l'écran écrit sous ses pinces), ``decimales_bench(ligne)`` → (décimales de la référence, de
    l'outil) de l'onglet Benchmark.

    Facultatifs (famille pied de poteau, deux modes et des tables libres) : ``visserie_defaut`` (la visserie d'un
    assemblage neuf de l'écran Streamlit, ``entrees.VISSERIE_DEFAUT``), ``exemples()`` → [(id, libellé, saisies
    brutes)] (« Charger ces données dans l'outil » : les cas de chaque mode), ``etat`` {clé: normaliser} (un état
    d'écran propre à l'assemblage, hors saisies du moteur : la grille à poser), ``actions`` {tête: f(cfg, el, reste,
    valeur)} (les chemins de cet état), ``sur_saisie`` {clé: f(cfg, el, valeur)} (une saisie qui fait plus que
    s'écrire : la bascule de mode reprend la disposition), ``aides`` (dict ou f(u) : l'aide « ? » de l'écran
    Streamlit d'un champ), ``hypotheses(R, u)`` (pastilles de plus), ``normes`` {groupe: norme}, ``blocs_tete(R,
    groupe)`` (HTML en tête du détail d'une fenêtre), ``lignes_resultat(R, u)`` (lignes de plus dans le résultat),
    ``complement_verdict(R)``, ``legende_33`` (le pied de la fenêtre Géométrie), ``natures`` [(code, sens)] et
    ``titre_non_verifies`` (la vue Méthode), ``validation(cfg)`` (la vue Validation, à la place de la générique),
    ``nom_menu`` (le nom de l'attache dans le menu Configuration quand son nom court redit sa famille), ``refus(u)``
    → [(chemin, message)] (les lignes de ses tables qu'une note relue porte et que la saisie refuserait),
    ``legende_manuel`` (la légende et la colonne de contrôle des calculs manuels de son onglet Benchmark)."""

    def __init__(self, **k):
        self.__dict__.update(k)
        a = RG.trouver(self.cle)
        if not a:
            raise KeyError(f"configuration absente du registre : {self.cle}")
        self.famille, self.titre, self.court = a[0].id, a[1].titre, a[1].court or a[1].titre
        self.titre_famille = a[0].titre
        for nom, defaut in (("affichage", {}), ("segments", frozenset()), ("libelles", {}), ("dimensions", {}),
                            ("predim", None), ("bench", None), ("non_reverifies", ()), ("tableaux_boulons", None),
                            ("selecteurs", {}), ("entiers", frozenset()), ("listes", {}), ("est_manuel", None),
                            ("blocs", None), ("visserie_defaut", VISSERIE_DEFAUT), ("exemples", None), ("etat", {}),
                            ("actions", {}), ("sur_saisie", {}), ("aides", None), ("hypotheses", None), ("normes", {}),
                            ("blocs_tete", None), ("lignes_resultat", None), ("complement_verdict", None),
                            ("legende_33", None), ("natures", None), ("titre_non_verifies", None), ("validation", None),
                            ("nom_menu", None), ("refus", None)):
            if not hasattr(self, nom):
                setattr(self, nom, defaut)


CONFIGS: Dict[str, Config] = {}


def enregistrer(cfg: Config) -> None:
    CONFIGS[cfg.cle] = cfg


def ordre() -> List[str]:
    """Les configurations dans l'ordre du registre (familles, puis assemblages)."""
    return [a.id for f in RG.FAMILLES for a in f.assemblages if a.id in CONFIGS]


def _cfg(el: Dict) -> Config:
    return CONFIGS[el["type"]]


def pour(cfg: Config, R) -> Config:
    """La description qui rend le résultat ``R`` : celle de la configuration, ou de sa variante (le pied de poteau
    a deux moteurs, la vérification complète et le calcul à la main, avec chacun sa synthèse, son texte, ses vues)."""
    v = getattr(cfg, "variante", None)
    return v(R) if (v and R is not None) else cfg


# ===================================================================== les valeurs d'un assemblage
def valeur_widget(cfg: Config, k: str, v):
    """La valeur telle que l'écran Streamlit la tient (``carte.valeur_widget``, recopiée faute de pouvoir importer
    carte.py qui charge streamlit — test : tests/test_api_web_assemblages.py la compare à l'original)."""
    f = cfg.E.CHAMPS.get(k)
    if f is None:
        return "" if v is None else str(v)
    if f["t"] == "n":
        if v is None or v == "":
            return None
        try:
            return float(str(v).strip().replace(",", "."))
        except (TypeError, ValueError):
            return None
    if f["t"] == "s":
        s = str(v) if v is not None else f["o"][0]
        if s not in f["o"]:
            try:
                s = str(int(float(s)))
            except (TypeError, ValueError):
                pass
        return s if s in f["o"] else f["o"][0]
    return "" if v is None else str(v)


def valeurs_normales(cfg: Config, src) -> Dict:
    """Un jeu de saisies (fichier, état reçu) relu comme l'écran Streamlit le relit : fusion sur les défauts
    (``entrees.charger_json``), puis chaque clé normalisée comme son widget ; les listes (calcul de la platine à la
    main) passent par la famille."""
    if not isinstance(src, dict):
        raise ValueError("saisies d'assemblage illisibles")
    u = cfg.E.charger_json(src)
    out = {}
    for k in cfg.E.defaults():
        if k in cfg.listes:
            out[k] = cfg.listes[k]["normaliser"](u.get(k))
        else:
            out[k] = valeur_widget(cfg, k, u.get(k))
    return out


def dimensions_vides(cfg: Config, u: Dict) -> List[str]:
    """Les dimensions visibles d'une pièce personnalisée restées vides ou ≤ 0 (``carte.dimensions_vides``) : la
    vérification est suspendue, le dessin garde des dimensions provisoires."""
    from acier.js import N
    return [k for k in cfg.dimensions if cfg.E.champ_visible(u, k) and not (N(u.get(k)) or 0) > 0]


def a_calculer(cfg: Config, u: Dict) -> Dict:
    """Les saisies passées au moteur : celles de l'écran, les dimensions vides remplacées par leurs valeurs
    provisoires (``carte.provisoire``) — interface.py fait de même avant ``compute``."""
    vides = dimensions_vides(cfg, u)
    return dict(u, **{k: cfg.dimensions[k] for k in vides}) if vides else u


def valeurs_defaut(cfg: Config) -> Dict:
    """Les saisies d'un assemblage NEUF : ``entrees.defaults()`` mis en forme des widgets — « Créer » de la page
    projet (``projet_ui.creer``) et 🔄 de l'écran. Ce n'est pas la lecture d'un fichier vide : ``charger_json`` lit un
    fichier sans « mode » comme un fichier d'avant le calcul à la main (pied de poteau : vérification complète), alors
    qu'un pied de poteau neuf s'ouvre sur le calcul à la main."""
    return valeurs_normales(cfg, cfg.E.defaults())


def trop_grand(cfg: Config, u: Dict) -> Optional[str]:
    """Le motif d'un calcul qu'on NE LANCE PAS : un nombre de rangées ou de tiges au-delà de ``COMPTES_MAXI`` (une
    note abîmée, l'écran les refuse à la saisie), ou une table du calcul à la main plus longue que ``LIGNES_MAXI``."""
    from acier.js import N
    for k, m in COMPTES_MAXI.items():
        x = N(u.get(k)) if k in cfg.E.CHAMPS else None
        if x is not None and x > m:
            return f"{_nom_champ(cfg, k)} : {C.nb(x)} — {m} au plus"
    for k in cfg.listes:
        if isinstance(u.get(k), list) and len(u[k]) > LIGNES_MAXI:
            return f"table « {k} » : {len(u[k])} lignes — {LIGNES_MAXI} au plus"
    return None


LIGNES_MAXI = 400          # boulons, éléments ou raidisseurs d'une table du calcul à la main (400 boulons : 0,05 s)


def moteur(cfg: Config, u: Dict):
    """Le calcul du moteur sur des saisies de l'écran (dimensions vides → provisoires), jamais sur un nombre de
    rangées ou de tiges hors bornes (``trop_grand`` : ValueError)."""
    motif = trop_grand(cfg, u)
    if motif:
        raise ValueError(motif)
    return cfg.calculer(a_calculer(cfg, u))


def calculer(cfg: Config, el: Dict):
    """(R, vides, erreur) d'un assemblage."""
    u = el["values"]
    vides = dimensions_vides(cfg, u)
    motif = trop_grand(cfg, u)
    if motif:                                     # le calcul n'est pas lancé (il épuiserait la mémoire de l'onglet)
        return None, vides, motif
    try:
        return moteur(cfg, u), vides, None
    except Exception as e:  # noqa: BLE001 — l'écran Streamlit affiche « Données incomplètes » ; le site aussi
        return None, vides, f"{type(e).__name__} : {e}"


def saisies_refusees(cfg: Config, u: Dict) -> List:
    """Les saisies d'un assemblage RELU (fichier .stab, note du navigateur) que l'écran du site refuserait si on les
    tapait (``_lire_saisie``) : V<sub>Ed</sub> vide, γ ≤ 0, rangées 2,4, dimension négative… L'écran Streamlit les
    calcule EN SILENCE (V vide = 0 : plat d'âme « vérifié » à 60 % au lieu de 74 % ; γ<sub>M2</sub> = 0 : boulons à
    0 %) — non conservatif. Le site les garde dans les champs, à corriger, et suspend le verdict. [(clé, message)]"""
    pred = bool(getattr(cfg.E, "mode_predim", lambda u: False)(u))
    pil = set(getattr(cfg.E, "PILOTEES_PAR_PREDIM", ())) if pred else set()
    out = []
    for k, f in cfg.E.CHAMPS.items():
        if f["t"] != "n" or k in cfg.listes or k in pil or k in cfg.sur_saisie or not cfg.E.champ_visible(u, k):
            continue
        _x, msg = _lire_saisie(cfg, u, k, u.get(k))
        if msg:
            out.append((k, msg))
    if cfg.refus:                                 # les tables d'une configuration (pied : éléments soudés)
        out += cfg.refus(u)
    return out


# ===================================================================== l'état de la note
def _etat_ecran(cfg: Config, el: Optional[Dict] = None) -> Dict:
    """L'état d'écran propre à la configuration (``Config.etat``), relu : une valeur abîmée reprend son défaut."""
    out = {}
    for k, normal in cfg.etat.items():
        try:
            out[k] = normal((el or {}).get(k))
        except (ValueError, TypeError, KeyError):
            out[k] = normal(None)
    return out


def nouvel_assemblage(cle: str, repere: str = "A1") -> Dict:
    cfg = CONFIGS[cle]
    v = valeurs_defaut(cfg)
    v["id_rep"] = repere
    return dict({"type": cle, "values": v, "visserie": cfg.visserie_defaut, "cotation": 1, "texte_complet": False},
                **_etat_ecran(cfg))


def nouvel_etat() -> Dict:
    return {"version": 1, "assemblages": [nouvel_assemblage(ordre()[0])], "actif": 0}


def _assemblage_relu(el) -> Dict:
    if not isinstance(el, dict) or el.get("type") not in CONFIGS:
        raise ValueError(f"assemblage illisible ou de configuration inconnue : {str(el)[:80]}")
    cfg = CONFIGS[el["type"]]
    v = el.get("values")
    if v is None:
        v = {}
    niv = el.get("cotation", 1)
    niv = int(niv) if isinstance(niv, (int, float)) and not isinstance(niv, bool) and 0 <= niv <= 2 else 1
    vis = el.get("visserie")
    return dict({"type": cfg.cle, "values": valeurs_normales(cfg, v), "visserie": str(vis) if vis is not None else cfg.visserie_defaut,
                 "cotation": niv, "texte_complet": bool(el.get("texte_complet", False))}, **_etat_ecran(cfg, el))


def normaliser(etat) -> Dict:
    """L'état reçu (navigateur, fichier .stab) relu : chaque assemblage relu comme l'écran Streamlit relit un fichier
    projet ; un assemblage illisible ou d'une configuration inconnue lève une erreur claire."""
    lst = etat.get("assemblages") if isinstance(etat, dict) else None
    if not lst:
        return nouvel_etat()
    if not isinstance(lst, list):
        raise ValueError("la liste des assemblages est illisible")
    out = {"version": 1, "assemblages": [_assemblage_relu(e) for e in lst]}
    try:
        a = int(etat.get("actif", 0) or 0)
    except (TypeError, ValueError):
        a = 0
    out["actif"] = min(max(0, a), len(out["assemblages"]) - 1)
    return out


def _actif(etat: Dict) -> Dict:
    return etat["assemblages"][etat["actif"]]


def _repere(el: Dict, i: int) -> str:
    return str(el["values"].get("id_rep") or "").strip() or f"A{i + 1}"


def _repere_libre(etat: Dict) -> str:
    """Le premier repère « A1, A2… » libre (acier/projet.repere_libre)."""
    projet = {"assemblages": [{"values": e["values"]} for e in etat["assemblages"]]}
    return PJ.repere_libre(projet)


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les assemblages de la note : en ouvrir un, en ajouter un (même configuration que l'ouvert, valeurs par
    défaut, premier repère libre — « Nouvel assemblage » de Streamlit), copier l'ouvert (« Dupliquer » : repère
    « … (copie) »), en supprimer un (jamais le dernier)."""
    lst = etat["assemblages"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(lst) - 1)
    elif action == "ajouter":
        lst.append(nouvel_assemblage(_actif(etat)["type"], _repere_libre(etat)))
        etat["actif"] = len(lst) - 1
    elif action == "dupliquer":
        e = copy.deepcopy(_actif(etat))
        e["values"]["id_rep"] = f"{_repere(e, etat['actif'])} (copie)"
        lst.insert(etat["actif"] + 1, e)
        etat["actif"] += 1
    elif action == "supprimer" and len(lst) > 1:
        i = min(max(0, int(index or 0)), len(lst) - 1)
        del lst[i]
        if etat["actif"] > i or etat["actif"] >= len(lst):
            etat["actif"] = max(0, etat["actif"] - 1)


# ===================================================================== la saisie
# Les saisies qui gardent leur SENS d'une attache poutre–poutre ou poutre–poteau à l'autre : profilés, grugeage,
# boulons (sauf leur disposition, propre à chaque pièce), efforts, coefficients, prédimensionnement, cartouche. La
# géométrie de la pièce (h_p, b_p, t_p, z, a, rangées, entraxes, pinces) ne suit PAS : la largeur d'un plat d'âme
# (100 mm) donnerait à une platine d'about une pince e2 de 5 mm.
GARDEES = frozenset({
    "id_projet", "id_rep", "id_red", "id_date", "mode_calc",
    "prof_P", "nu_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u", "prof_S", "nu_S", "hS_u", "bS_u", "twS_u", "tfS_u", "rS_u",
    "d_top", "d_nt", "d_nb", "l_n", "r_n", "lt_ok",
    "boulon_u", "classe", "trou", "d0_u", "filet", "cat", "mu_s", "k_s", "k_ser",
    "V_Ed", "N_Ed", "M_Ed", "H_Ed", "nat_N",
    "g_M0", "g_M1", "g_M2", "g_M2n", "g_M3", "g_M3s", "g_Mu", "opt_blf", "expo", "fy_u", "fu_u", "bw_u",
    "eta_c", "k_e1", "k_p1", "k_e2", "pd_dmin", "pd_dmax",
    # poutre–poteau : le cas d'appui et le jeu de montage de la poutre entre les ailes gardent leur sens d'une attache
    # à l'autre (la nature de N_Ed suit N_Ed : une traction de chaînage reste une traction de chaînage)
    "appui", "j_lat"})
# la pièce porteuse : une poutre principale (poutre–poutre) n'est pas un poteau (poutre–poteau)
PORTEUSE = frozenset({"prof_P", "nu_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"})
# D'une famille à l'autre, la poutre portée et son grugeage ne suivent pas non plus : la pièce neuve (plat, cornières,
# platine) a la géométrie par défaut de SA famille, dimensionnée pour SA poutre par défaut (HEA 300 grugé sous une
# poutre, IPE 400 non grugé sur un poteau). Gardés, ils entraient en conflit avec elle : un HEA 300 sous un plat posé à
# z_p = 40 mm (t_f + r = 41 mm), ou un grugeage de 50 mm sur un plat à 40 mm, donnaient « Impossible » à l'ouverture.
PORTEE = frozenset({"prof_S", "nu_S", "hS_u", "bS_u", "twS_u", "tfS_u", "rS_u"})
GRUGEAGE = frozenset({"d_top", "d_nt", "d_nb", "l_n"})


def gardees(ancien: Config, cfg: Config) -> frozenset:
    """Les saisies que l'assemblage garde quand il change de configuration : dans la même famille, GARDEES ; entre
    poutre–poutre et poutre–poteau, GARDEES sans les pièces (porteuse, portée et son grugeage : l'assemblage repart
    des défauts de la configuration, comme « Créer » de Streamlit, avec ses boulons, ses efforts, ses coefficients et
    son cartouche) ; avec le pied de poteau (N_Ed y est une COMPRESSION positive, une traction ailleurs), le cartouche
    seulement."""
    if ancien.famille == cfg.famille:
        return GARDEES
    if "pied_poteau" in (ancien.famille, cfg.famille):
        return frozenset({"id_projet", "id_rep", "id_red", "id_date"})
    return GARDEES - PORTEUSE - PORTEE - GRUGEAGE


def _changer_configuration(etat: Dict, cle) -> Optional[str]:
    """Le menu « Configuration » : l'assemblage ouvert prend une autre configuration, aux valeurs par défaut ; il
    garde son repère, son cartouche, sa visserie et les saisies qui gardent leur sens (``gardees``)."""
    if cle not in CONFIGS:
        return f"Configuration inconnue : « {cle} »"
    el = _actif(etat)
    if cle == el["type"]:
        return None
    ancien, cfg = _cfg(el), CONFIGS[cle]
    v = valeurs_defaut(cfg)
    for k in gardees(ancien, cfg):
        if k not in v or k not in el["values"] or k in cfg.listes:
            continue
        f = cfg.E.CHAMPS.get(k)
        x = el["values"][k]
        if f is None or (f["t"] == "s" and x not in f["o"]):
            continue
        v[k] = valeur_widget(cfg, k, x)
    for k in ancien.etat:
        el.pop(k, None)
    el.update({"type": cle, "values": v}, **_etat_ecran(cfg))
    if el.get("visserie") == ancien.visserie_defaut:     # la visserie par défaut suit (celle de l'écran de l'attache)
        el["visserie"] = cfg.visserie_defaut
    return None


def _reprendre_profil(cfg: Config, u: Dict, k: str, ancien) -> None:
    """Un profilé (ou une cornière) passé en « Personnalisé » : ses dimensions, si elles sont vides, reprennent
    celles de la pièce qu'on quitte (Streamlit les laisse vides et suspend la vérification ; le site part d'une
    pièce réelle, qu'on modifie)."""
    if k in PERSO_PROFIL:
        X = PERSO_PROFIL[k]
        cles = [(f"{d}{X}_u", d) for d in ("h", "b", "tw", "tf", "r")]
        liste = "profils"
    elif k == "corn_u":
        cles, liste = [("k1_u", "a1"), ("k2_u", "a2"), ("kt_u", "t"), ("kr_u", "r")], "cornieres"
    else:
        return
    try:
        src = find(DB[liste], ancien)
    except (ValueError, KeyError, TypeError):
        return
    for cle, champ in cles:
        if cle in u and not (u.get(cle) or 0) > 0:
            u[cle] = float(src[champ])


def _reprendre_acier(cfg: Config, u: Dict, ancien) -> None:
    """Un acier passé en « Personnalisé » : f_y, f_u et β_w vides reprennent ceux de la nuance qu'on quitte (vides,
    le moteur prendrait f_y = 0)."""
    try:
        src = find(DB["aciers"], ancien)
    except (ValueError, KeyError, TypeError):
        return
    for cle, champ in PERSO_ACIER:
        if cle in u and not (u.get(cle) or 0) > 0:
            u[cle] = float(src[champ])


def _lire_saisie(cfg: Config, u: Dict, k: str, valeur):
    """(valeur normalisée, message) d'une saisie dans le champ ``k``."""
    f = cfg.E.CHAMPS[k]
    if f["t"] == "s" and list(f["o"]) == OUI_NON and isinstance(valeur, bool):
        return ("Oui" if valeur else "Non"), None
    if f["t"] == "s":
        s = str(valeur)
        if s not in f["o"]:
            return None, f"Option inconnue pour {_nom_champ(cfg, k)} : « {valeur} »"
        return s, None
    if f["t"] == "x":
        return ("" if valeur is None else str(valeur)), None
    if valeur is None or (isinstance(valeur, str) and not valeur.strip()):
        if k in cfg.dimensions or k in ("d0_u",) or k.endswith(("P_u", "S_u")) or k in ("fy_u", "fu_u", "bw_u"):
            return None, None
        return None, f"{_nom_champ(cfg, k)} : une valeur est attendue"
    x = C.lire_nombre(valeur)
    if x is None:
        return None, f"Nombre illisible : « {valeur} »"
    if k in COMPTES or k in cfg.entiers:
        if x < 1 or abs(x - round(x)) > 1e-9:
            return None, (f"{_nom_champ(cfg, k)} : un nombre entier ≥ 1 est attendu (« {valeur} » ; l'écran Streamlit "
                          f"l'arrondirait en silence)")
        if x > COMPTES_MAXI.get(k, x):
            return None, f"{_nom_champ(cfg, k)} : {COMPTES_MAXI[k]} au plus (« {valeur} »)"
        return float(round(x)), None
    if POSITIVES.match(k) and x <= 0:
        return None, f"{_nom_champ(cfg, k)} : une valeur strictement positive est attendue"
    if x < 0 and k not in SIGNEES:
        return None, f"{_nom_champ(cfg, k)} : une valeur positive ou nulle est attendue (« {valeur} »)"
    return x, None


def _nom_champ(cfg: Config, k: str) -> str:
    lab = cfg.libelles.get(k)
    if lab:
        return re.sub(r"<[^>]+>", "", (lab[0] + " " + lab[1]).strip())
    f = cfg.E.CHAMPS.get(k)
    return f["l"] if f else k


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Une saisie dans l'assemblage ouvert. ``chemin`` : [« type »] (le menu Configuration), [« v », clé] (une donnée
    du moteur), [« v », liste, …] (les tables du calcul à la main), [« visserie »], [« cotation »], [« texte_complet
    »], [« d0_impose »], [« exemple »], [« reinitialiser »]. Rend un message si la saisie est refusée — l'assemblage
    reste intact."""
    el = _actif(etat)
    cfg = _cfg(el)
    chemin = [int(c) if isinstance(c, str) and c.lstrip("-").isdigit() and c != "-" else c for c in chemin]
    tete = chemin[0] if chemin else None
    if chemin == ["type"]:
        return _changer_configuration(etat, valeur)
    if chemin == ["visserie"]:
        el["visserie"] = "" if valeur is None else str(valeur)
        return None
    if chemin == ["cotation"]:
        x = C.lire_nombre(valeur)
        if x is None or x not in (0, 1, 2):
            return f"Niveau de cotation inconnu : « {valeur} »"
        el["cotation"] = int(x)
        return None
    if chemin == ["texte_complet"]:
        el["texte_complet"] = bool(valeur)
        return None
    if chemin == ["d0_impose"]:
        return _d0_impose(cfg, el["values"], bool(valeur))
    if chemin == ["exemple"]:
        return _charger_exemple(cfg, el, valeur)
    if chemin == ["reinitialiser"]:
        return _reinitialiser(cfg, el)
    if tete in cfg.actions:                       # un état d'écran de la configuration (pied : la grille à poser)
        return cfg.actions[tete](cfg, el, chemin[1:], valeur)
    if tete != "v" or len(chemin) < 2:
        return f"Champ inconnu : {'/'.join(str(c) for c in chemin)}"
    u = el["values"]
    k = chemin[1]
    if k in cfg.listes:
        avant = copy.deepcopy(u[k])
        try:
            msg = cfg.listes[k]["modifier"](u, chemin[2:], valeur)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            msg = f"Saisie refusée : {e}"
        if msg:
            u[k] = avant
        return msg
    if k not in cfg.E.CHAMPS or len(chemin) != 2:
        return f"Champ inconnu : {'/'.join(str(c) for c in chemin)}"
    x, msg = _lire_saisie(cfg, u, k, valeur)
    if msg:
        return msg
    if k in cfg.sur_saisie:                       # pied : la bascule de mode reprend la disposition
        return cfg.sur_saisie[k](cfg, el, x)
    ancien = u.get(k)
    u[k] = x
    if x == PERSO and ancien != PERSO:
        if k in PERSO_PROFIL or k == "corn_u":
            _reprendre_profil(cfg, u, k, ancien)
        elif k.startswith("nu_"):
            _reprendre_acier(cfg, u, ancien)
    return None


def _d0_impose(cfg: Config, u: Dict, oui: bool) -> Optional[str]:
    """La case « trou imposé » : décochée, d0 est vide (trou normal du boulon, Tab. 3.3) ; cochée, d0 part du trou
    normal du boulon retenu, à modifier (un d0 déjà saisi est gardé)."""
    if "d0_u" not in u:
        return None
    if not oui:
        u["d0_u"] = None
        return None
    if not (u.get("d0_u") or 0) > 0:
        try:
            u["d0_u"] = float(find(DB["boulons"], u.get("boulon_u"))["d0"])
        except (ValueError, KeyError, TypeError):
            u["d0_u"] = None
    return None


def _reinitialiser(cfg: Config, el: Dict) -> None:
    """« 🔄 Réinitialiser » de l'écran Streamlit (``interface._reinitialiser`` puis ``projet_ui.entree_reinitialisee``) :
    les valeurs par défaut de la configuration et la visserie par défaut ; le repère est gardé."""
    rep = el["values"].get("id_rep")
    el["values"] = valeurs_defaut(cfg)
    el["values"]["id_rep"] = rep
    el["visserie"] = cfg.visserie_defaut
    el.update(_etat_ecran(cfg))


def exemples(cfg: Config) -> List:
    """Les exemples à charger : [(id, libellé, saisies brutes)] — ceux de la configuration (``Config.exemples``), ou
    les exemples publiés de son benchmark (les défauts, puis les données de l'exemple)."""
    if cfg.exemples:
        return list(cfg.exemples())
    out = []
    for x in (getattr(cfg.bench, "BENCH", None) or []):
        d = cfg.E.defaults()
        d.update(x["inp"])
        out.append((x["id"], f"{x['id']} · {x['titre'][:40]}", d))
    return out


def _charger_exemple(cfg: Config, el: Dict, valeur) -> Optional[str]:
    """« Charger ces données dans l'outil » (onglet Benchmark de Streamlit) : les saisies de l'exemple (``exemples``)
    ; le repère est gardé."""
    if valeur in ("", None):
        return None
    ex = {i: d for i, _lab, d in exemples(cfg)}
    if valeur not in ex:
        return f"Exemple inconnu : « {valeur} »"
    d = copy.deepcopy(ex[valeur])
    rep = el["values"].get("id_rep")
    el["values"] = valeurs_normales(cfg, d)
    el["values"]["id_rep"] = rep
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """La ligne choisie dans un sélecteur : un boulon, un profilé, une cornière, une solution du prédimensionnement."""
    el = _actif(etat)
    cfg = _cfg(el)
    u = el["values"]
    sel = _selecteurs(cfg).get(role)
    if sel is None:
        raise ValueError(f"sélecteur inconnu : {role}")
    sel["choisir"](cfg, u, nom)


# ===================================================================== le texte : symboles et formules
_STOP = {"max", "min", "avec", "et", "si", "sinon", "ou", "sur", "sous", "par", "pour", "les", "des", "une", "un", "le",
         "la", "du", "de", "en", "au", "aux", "dans", "net", "brut", "Oui", "Non", "mm", "cm", "kN", "kNm", "MPa", "kg",
         "sqrt", "abs", "cos", "sin", "tan", "ln", "exp", "lim", "file", "files", "plat", "flexion", "limite",
         "domaine", "Valide", "valeur", "selon", "entre", "via", "tout", "tous", "deux", "soit", "est", "sans", "non",
         "cas", "pas", "and", "EN", "Tab", "MSB", "SCI", "ECCS", "ANB", "ELU", "ELS", "base", "bord", "haut", "bas",
         "trou", "rangs", "rang", "vers", "face", "aile", "ailes", "kN·m", "pb", "py",
         # des mots français des formules poutre–poteau et pied de poteau que la règle prenait pour des symboles
         # (« aucun bord libre » s'écrivait a_ucun bord l_ibre)
         "aucun", "aucune", "libre", "paroi", "mode", "plan", "plans", "bande", "zone", "pince", "ligne", "due", "son",
         "ne", "qu", "exact", "effet", "plus", "gauge", "tige", "tiges", "seul", "coin", "pry", "out", "objet", "bout",
         "se", "sa", "raidie", "donc",
         # pied de poteau : les profilés nommés (HEA 300 s'écrivait H_EA 300) et « au nu de la paroi »
         "HEA", "HEB", "HEM", "HEAA", "IPE", "RHS", "SHS", "CHS", "UPN", "UPE", "nu", "plat", "rond"}
_JETON = re.compile(r"(?<![\w.,])([A-Za-zσητγαμ])([A-Za-z0-9]+(?:,[A-Za-z0-9]+)*)(?![\w(])")


def indices(s: str) -> str:
    """La notation des assemblages (« Fv,Rd », « γM2 », « fub », « tp ») écrite pour C.symb_math (« F_v,Rd »,
    « γ_M2 », « f_ub », « t_p ») : la première lettre d'un symbole est sa base, le reste son indice. Les mots
    français, les unités et les fonctions restent tels quels."""
    def rep(m):
        base, reste = m.group(1), m.group(2)
        mot = base + reste
        if mot in _STOP or mot.lower() in _STOP:
            return mot
        if len(reste) > 4 and not re.search(r"[,0-9A-Z]", reste):
            return mot
        return base + "_" + reste
    s = re.sub(r"(?<![\w.,])([A-Z]),(Ed|Rd)(?!\w)", r"\1\2", s or "")      # « F,Ed » du moteur : F_Ed
    # un exposant (« Fz² ») n'empêche pas l'indice : il est mis de côté le temps de la règle (² et ³ sont des \w)
    s = s.replace("²", "\x01").replace("³", "\x02")
    return _JETON.sub(rep, s).replace("\x01", "²").replace("\x02", "³")


def math_html(s: str) -> str:
    """Un symbole ou une expression du moteur, écrit en HTML avec ses indices (texte échappé)."""
    return C.symb_math(indices(s))


def lignes_texte(R, c, textes: Callable) -> (str, List[Dict]):
    """Le détail d'une vérification dont le moteur compose les formules en TEXTE (``formules.textes`` de la
    famille : formule générale, valeurs introduites, « η = Ed / Rd = … ») → (la formule générale en clair, les
    lignes « symbole = valeur » et la ligne du taux)."""
    t = textes(R, c)
    formule, vals, eta = (t[0] if t else ""), (t[1] if len(t) > 2 else ""), (t[-1] if t else "")
    lignes = []
    for morceau in re.split(r"\s+;\s+|\s+→\s+", vals or ""):
        morceau = morceau.strip().rstrip(".")
        if not morceau:
            continue
        if " = " in morceau:
            g, d = morceau.split(" = ", 1)
            lignes.append({"parts": [indices(g.strip())], "res": d.strip(), "niveau": ""})
        elif " : " in morceau:
            g, d = morceau.split(" : ", 1)
            lignes.append({"parts": [g.strip()], "res": d.strip(), "niveau": ""})
        else:
            lignes.append({"parts": [indices(morceau)], "res": "", "niveau": ""})
    m = re.match(r"η = (.+?) = ([^=]+?)\s*(?:≤|>) 100 %$", eta or "")
    if m:
        lignes.append({"parts": ["η", m.group(1)], "res": f"{C.pc(taux_c(c, 1))} %", "niveau": niveau_c(c)})
    else:
        lignes.append({"parts": ["η"], "res": f"{C.pc(taux_c(c, 1))} %", "niveau": niveau_c(c)})
    return formule, [li for li in lignes if li["res"] != "" or len(li["parts"]) > 1]


def lignes_sub(R, c, F) -> (str, List[Dict]):
    """Le détail d'une vérification dont le moteur compose des SUBSTITUTIONS (``formules.lignes`` : « Fv,Rd =
    αv·fub·A/γM2 = 0,6 × 800 × 245/1,25/10³ = 94,08 kN ») → les lignes de C.lignes_fx, telles quelles."""
    L = F.lignes(R, c)
    if not L:
        return lignes_texte(R, c, F.textes)
    out = []
    for ligne in L:
        for s in ligne:
            parts, res = F.morceaux(s)
            parts = [indices(p) for p in parts]
            if s.nom == "η":
                out.append({"parts": parts, "res": f"{C.pc(taux_c(c, 1))} %", "niveau": niveau_c(c)})
            else:
                out.append({"parts": parts, "res": res, "niveau": ""})
    return c.formula, out


def taux_estime(eta, nd: int = 1):
    """(taux affiché, niveau) d'un taux ESTIMÉ du prédimensionnement : plafonné à 999 %, rouge au-delà de 100 % ; un
    taux indéterminé (NaN : une combinaison dégénérée, que l'écran Streamlit écrit « — ») est rouge, à 999 %, et dit
    « indéterminé » là où la place le permet."""
    if not isinstance(eta, (int, float)) or not math.isfinite(eta):
        return C.pct(9.99, nd), "r"
    return C.pct(min(eta, 9.99), nd), ("r" if eta > 1 else C.niveau(eta))


def taux(eta, nd: int = 0):
    """Un taux du moteur à afficher (C.pct), plafonné à 999 % : un taux infini ou indéterminé (pied de poteau :
    aucune épaisseur ne suffit, η = ∞ — l'écran Streamlit écrit « > 999 % ») reste rouge au lieu d'arrêter l'écran."""
    if not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta > 9.99:
        return C.pct(9.99, nd)
    return C.pct(eta, nd)


def eta_vu(eta, ok: bool = True):
    """Le taux à MONTRER : celui du moteur, sauf qu'un taux que le moteur déclare vérifié ne passe pas 100 %. Le
    moteur admet η ≤ 1 + 10⁻⁹ (``c.ok``, l'arrondi des flottants : une gorge égale à la gorge minimale, un effort
    pile à la résistance) et l'écran Streamlit écrit alors « 100,0 % · VÉRIFIÉ » ; sans cette borne, le site écrivait
    « 101 % » en rouge, « Non conforme », à côté de « E<sub>d</sub> ≤ R<sub>d</sub »."""
    if ok and isinstance(eta, (int, float)) and math.isfinite(eta) and eta > 1:
        return 1.0
    return eta


def taux_c(c, nd: int = 1):
    """Le taux affiché d'une vérification (``eta_vu`` puis ``taux``)."""
    return taux(eta_vu(c.eta, c.ok), nd)


def eta_R(R):
    """Le taux affiché d'un assemblage : son η max, borné à 100 % quand sa vérification gouvernante est vérifiée."""
    g = R.get("gov")
    return eta_vu(R.eta_max, g.ok if g is not None else bool(R.verified))


def niveau_c(c) -> str:
    """La couleur d'une vérification : son taux, rouge dès que le moteur la déclare non vérifiée."""
    return "r" if not c.ok else C.niveau(eta_vu(c.eta))


def niveau_R(R) -> str:
    """La couleur d'un assemblage : son taux maximal, rouge dès que le moteur ne le déclare pas vérifié (géométrie,
    pinces, alerte bloquante) même sous 100 %."""
    return "r" if not R.verified else C.niveau(eta_R(R))


def markdown(md: str) -> str:
    """Le texte de l'onglet Méthode (Markdown simple du moteur : titres ####, listes, gras, `code`, paragraphes) en
    HTML échappé."""
    out, para, liste = [], [], []

    def ligne(t):
        t = html.escape(t, quote=False)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", t)
        return t

    def vider():
        if para:
            out.append("<p>" + ligne(" ".join(para)) + "</p>")
            para.clear()
        if liste:
            out.append("<ul>" + "".join("<li>" + ligne(x) + "</li>" for x in liste) + "</ul>")
            liste.clear()

    for brut in (md or "").splitlines():
        t = brut.rstrip()
        if not t.strip():
            vider()
            continue
        if t.lstrip().startswith("#"):
            vider()
            out.append("<h4>" + ligne(t.lstrip("# ").strip()) + "</h4>")
        elif re.match(r"^\s*[-*] ", t):
            if para:
                vider()
            liste.append(re.sub(r"^\s*[-*] ", "", t))
        elif liste and brut.startswith("  "):
            liste[-1] += " " + t.strip()
        else:
            if liste:
                vider()
            para.append(t.strip())
    vider()
    return "".join(out)


def fr_js(x, nd: int = 1) -> str:
    """Un nombre de l'onglet Benchmark écrit comme l'écran Streamlit l'écrit (``acier.formats.F`` : l'arrondi de
    ``toFixed``, 31,25 → « 31,3 » là où l'arrondi de Python dit « 31,2 »), avec l'espace fine des milliers du site."""
    from acier.formats import F
    s = F(x, nd)
    ent, virg, dec = s.partition(",")
    signe = "-" if ent.startswith("-") else ""
    ent = ent.lstrip("-")
    if ent.isdigit() and len(ent) > 3:
        ent = f"{int(ent):,}".replace(",", "\u202f")
    return signe + ent + virg + dec


def tableau(entetes: Sequence[str], lignes: Sequence[Sequence[str]], droite: Sequence[int] = (), classe: str = "") -> str:
    """Un tableau de la charte (classe vtab) ; les cellules sont déjà du HTML."""
    th = "".join(f'<th class="{"n" if i in droite else ""}">{h}</th>' for i, h in enumerate(entetes))
    tr = "".join("<tr>" + "".join(f'<td class="{"n" if i in droite else ""}">{x}</td>' for i, x in enumerate(li)) + "</tr>"
                 for li in lignes)
    return f'<table class="vtab {classe}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


# ===================================================================== le cadre Données
def _unite(f: Dict) -> str:
    un = f.get("un") or ""
    return UNITES.get(un, un)


def _aide(f: Dict, cfg: Optional[Config] = None, u: Optional[Dict] = None) -> str:
    """L'aide d'un champ : le libellé long et l'aide de l'écran Streamlit (entrees.GROUPES, source unique) — ou,
    quand la configuration les porte (``Config.aides`` : les « ? » de la carte Streamlit), celle-là."""
    t = f.get("l") or ""
    aides = (cfg.aides(u or {}) if callable(cfg.aides) else cfg.aides) if cfg is not None and cfg.aides else {}
    h = aides.get(f["k"]) or f.get("h")
    if h:
        t += " — " + h
    return t


def champ(cfg: Config, u: Dict, k: str, verrou: bool = False, plein: Optional[bool] = None, libelle=None) -> str:
    """Une ligne de saisie pour la clé ``k`` du moteur : symbole et libellé courts, aide = libellé long de l'écran
    Streamlit, le champ selon le type (nombre, choix, texte), verrouillé si le prédimensionnement le pilote.
    ``libelle`` (symbole, libellé) : celui du mode (pied de poteau : « Tige » ou « Boulon »)."""
    f = cfg.E.CHAMPS[k]
    sym, lab = libelle or cfg.libelles.get(k, ("", f["l"]))
    v = u.get(k)
    if f["t"] == "s":
        aff = cfg.affichage.get(k, {})
        opts = [(o, aff.get(o, o)) for o in f["o"]]
        if list(f["o"]) == OUI_NON:
            return C.case(["v", k], v == "Oui", (sym + " " if sym else "") + lab)
        if k in cfg.segments:
            return C.ligne(lab, C.segments(["v", k], v, opts), sym, _aide(f, cfg, u), full=True)
        large = plein if plein is not None else max(len(t) for _, t in opts) > 15
        return C.ligne(lab, C.choix(["v", k], v, opts, desactive=verrou), sym, _aide(f, cfg, u), full=large)
    if f["t"] == "x":
        return C.ligne(lab, C.texte(["v", k], v, ""), sym, _aide(f, cfg, u), full=True)
    entier = k in COMPTES or k in cfg.entiers
    return C.ligne(lab, C.num(["v", k], v, _unite(f), dec=3, entier=entier, desactive=verrou, mini=1 if entier else None), sym, _aide(f, cfg, u))


def champs(cfg: Config, u: Dict, cles: Sequence[str]) -> str:
    """Les champs visibles (``entrees.champ_visible``, les conditions de l'écran Streamlit) parmi ``cles`` ; les
    champs pilotés par le prédimensionnement sont verrouillés (◆ dans Streamlit)."""
    pred = bool(getattr(cfg.E, "mode_predim", lambda u: False)(u))
    pil = set(getattr(cfg.E, "PILOTEES_PAR_PREDIM", ()))
    out = ""
    for k in cles:
        if isinstance(k, tuple) and k[0] == "lib":        # un champ sous le libellé de son mode : ("lib", clé, (sym, lab))
            _g, cle, lab = k
            if cle in cfg.E.CHAMPS and cfg.E.champ_visible(u, cle):
                out += champ(cfg, u, cle, verrou=pred and cle in pil, libelle=lab)
            continue
        if isinstance(k, tuple):
            genre, x = k
            if genre == "grp":
                out += f'<div class="grp">{x}</div>'
            elif genre == "aide":
                out += C.aide(x)
            elif genre == "html":
                out += x
            continue
        if k in cfg.E.CHAMPS and cfg.E.champ_visible(u, k):
            out += champ(cfg, u, k, verrou=pred and k in pil)
    return out


def _menu_configuration(el: Dict) -> str:
    """Le menu « Configuration » : les sept attaches du registre, sur toute la largeur du cadre (« Poutre – Poutre ·
    Platine d'about » ne tient pas à côté d'un libellé)."""
    opts = []
    for f in RG.FAMILLES:
        for a in f.assemblages:
            if a.id in CONFIGS:
                opts.append((a.id, f"{f.titre} · {CONFIGS[a.id].nom_menu or a.court or a.titre}"))
    aide = C.esc("Les sept attaches du registre des assemblages (EN 1993-1-8) : poutre–poutre, poutre–poteau (aile ou "
                 "âme), pied de poteau sur béton. Les saisies communes (profilés, boulons, efforts) suivent dans la même famille.")
    return (f'<div class="grp">Configuration <span class="aide" title="{aide}">{C.icon("info")}</span></div>'
            f'<div class="asm-config">{C.choix(["type"], el["type"], opts)}</div>')


def formulaire(etat: Dict, R=None) -> str:
    """Le cadre Données : la configuration et le repère en tête, puis les sections de la configuration."""
    el = _actif(etat)
    cfg = _cfg(el)
    u = el["values"]
    out = ""
    for n, (cle, titre, resume, items, ouvert) in enumerate(cfg.sections(u, R, el), start=1):
        corps = champs(cfg, u, items)
        if n == 1:
            corps = _menu_configuration(el) + corps
        if corps:
            out += C.section(n, titre, resume, corps, ouvert, cle)
    return out


def section_cartouche(cfg: Config, u: Dict, el: Dict, avec_visserie: bool = True, libelle_visserie: str = "Visserie",
                      aide_visserie: str = "") -> List:
    """Les items de la section « Cartouche et affichage » (commune) : rédacteur, visserie, projet, date, niveau de
    cotation des dessins, texte complet de la note, exemple publié à charger."""
    items = ["id_red"]
    if avec_visserie:
        items.append(("html", C.ligne(libelle_visserie, C.texte(["visserie"], el.get("visserie"), cfg.visserie_defaut), "",
                                      aide_visserie or "Rondelles et écrous par boulon, portés au cartouche du plan. Ex. : "
                                      "« 1 rondelle + 2 écrous (contre-écrou) ».", full=True)))
    items += ["id_projet", "id_date", ("grp", "Affichage")]
    items.append(("html", C.ligne("Cotes", C.segments(["cotation"], str(el.get("cotation", 1)), NIVEAUX_COTATION), "",
                                  "Niveau de cotation des dessins : vue simple, édition (toutes les cotes saisies), "
                                  "cotations complètes (grandeurs calculées comprises).", full=True)))
    items.append(("html", C.case(["texte_complet"], bool(el.get("texte_complet")), "Note texte : toutes les vérifications, "
                                 "avec leurs formules")))
    ex = exemples(cfg)
    if ex:
        opts = [("", "—")] + [(i, lab) for i, lab, _d in ex]
        items += [("grp", "Exemples publiés" if not cfg.exemples else "Exemples et cas de validation"),
                  ("html", C.ligne("Charger", C.choix(["exemple"], "", opts), "",
                                   "Remplace les données par celles d'un exemple publié (onglet Benchmark de "
                                   "Streamlit) ; le repère est gardé.", full=True))]
    items.append(("html", f'<button type="button" class="addrow" data-chemin="reinitialiser" data-valeur="1" title="Le repère est gardé">{C.icon("history")}'
                          'Rétablir les valeurs par défaut</button>'))
    return items


def sections_generiques(cfg: Config, u: Dict, R, el: Dict, ouvertes=("Boulons", "Efforts")) -> List:
    """Les sections du cadre Données tirées des GROUPES de l'écran Streamlit (entrees.GROUPES : même ordre, mêmes
    titres, mêmes conditions d'affichage) : l'identification et le mode en tête (section « Assemblage »), le
    cartouche à la fin. Sert aux configurations qui n'ont pas (encore) de disposition propre."""
    tete = ["id_rep"]
    corps, fin = [], []
    for g in cfg.E.GROUPES:
        cles = [f["k"] for f in g["f"]]
        titre = g["t"].split(" – ")[0].split(" (")[0]
        if g["t"] == "Identification":
            continue
        if g["t"].startswith("Mode"):
            tete += cles
            continue
        if g["t"].startswith("Règles du prédimensionnement"):
            fin = [("grp", "Prédimensionnement")] + cles
            continue
        if "d0_u" in cles:
            i = cles.index("d0_u")
            cles = cles[:i] + d0_items(u) + cles[i + 1:]
        ouvert = any(titre.startswith(o) for o in ouvertes)
        corps.append((_slug(titre)[:12], titre, "", cles, ouvert))
    out = [("as", "Assemblage", C.esc(cfg.court), tete + fin, True)] + corps
    out.append(("ca", "Cartouche et affichage", C.esc(u.get("id_red") or "rédacteur à saisir"), section_cartouche(cfg, u, el), False))
    return out


def d0_items(u: Dict) -> List:
    """La case du trou imposé, puis d0 s'il est imposé (Streamlit : un champ vide = trou normal)."""
    if "d0_u" not in u:
        return []
    oui = (u.get("d0_u") or 0) > 0
    out = [("html", C.case(["d0_impose"], oui, "Trou de diamètre d<sub>0</sub> imposé (sinon trou normal, Tab. 3.3)"))]
    if oui:
        out.append("d0_u")
    return out


# ===================================================================== le résultat
def information(cfg: Config, R, c) -> bool:
    """Une vérification affichée EN INFORMATION, hors verdict (pied de poteau : béton d'ancrage EN 1992-4 quand le
    périmètre est l'EN 1993-1-8 seule, niveau 2, pression maximale — ``synthese.info``) : elle ne colore ni sa pièce
    ni le verdict, comme dans l'écran Streamlit."""
    f = getattr(cfg.S, "info", None)
    return bool(f and f(R, c.key))


def _groupes(cfg: Config, R) -> List:
    """Les pièces vérifiées (synthese.ELEMENTS) : (groupe, nom, η, ok, pire, [vérifications actives], information).
    Le taux d'une pièce est celui que synthese.taux_par_element nomme ; une pièce qui n'a que des vérifications
    d'information est une ligne d'information (grise), hors verdict."""
    out = []
    par = cfg.S.taux_par_element(R)
    for grp, _court in cfg.S.ELEMENTS:
        on = [c for c in R.checks if c.grp == grp and c.active]
        if not on:
            continue
        utiles = [c for c in on if not information(cfg, R, c)]
        if not utiles:
            pire = max(on, key=lambda c: c.eta)
            out.append((grp, grp, pire.eta, True, pire, on, True))
            continue
        t = next((x for x in par if x[3] in utiles), None)
        pire = t[3] if t else max(utiles, key=lambda c: c.eta)
        out.append((grp, grp, pire.eta, all(c.ok for c in utiles), pire, on, False))
    return out


def _slug(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "g"


def _court(cfg: Config, c) -> str:
    return cfg.S.court(c)


def _comparaison(cfg: Config, c) -> str:
    ed, rd, un = cfg.S.ed_rd(c)
    u = f" {un}" if un else ""
    return f"{ed}{u} {'≤' if c.ok else '&gt;'} {rd}{u}"


def premiere_formule(f) -> str:
    """La formule principale d'une vérification : son texte jusqu'au premier « ; » ou « — » HORS parenthèses (« tp ≥
    max(tT ; tS ; tc ; tb) — … » garde sa parenthèse entière)."""
    f = f or ""
    prof = 0
    for i, ch in enumerate(f):
        prof += (ch == "(") - (ch == ")")
        if prof <= 0 and (f.startswith(" ; ", i) or f.startswith(" — ", i)):
            return f[:i]
    return f


def _ligne_vtab(cfg: Config, c, gov: bool, R=None) -> Dict:
    ed, rd, un = cfg.S.ed_rd(c)
    u = f" {un}" if un else ""
    form = premiere_formule(c.formula)
    sous = C.esc(c.nat) + " · " + C.esc(cfg.N.ref_courte(c.ref))
    info = R is not None and information(cfg, R, c)
    if info:
        sous += " · information, hors verdict"
    return {"nom": C.esc(_court(cfg, c)), "sous": sous, "formule": math_html(form), "valeur": f"{ed}{u}", "limite": f"{rd}{u}",
            "taux": taux(c.eta, 1) if info else taux_c(c, 1), "niveau": "c" if info else niveau_c(c), "gov": gov}


def _hypotheses(cfg: Config, R, u: Dict) -> List[str]:
    """Les hypothèses en pastilles : aciers, boulon, coefficients partiels."""
    h = []
    aciers = []
    noms = getattr(cfg, "noms_aciers", None) or {}           # poutre–poteau : « P » est le poteau, « S » la poutre portée
    for X, nom in (("P", "principale"), ("S", "secondaire"), ("L", getattr(cfg, "nom_plat", "plat")), ("C", "cornières"),
                   ("c", "poteau"), ("p", "platine")):
        nom = noms.get(X, nom)
        fy, fu = R.get("fy_" + X), R.get("fu_" + X)
        if fy:
            aciers.append(f"{nom} f<sub>y</sub> = {C.nb(fy)} · f<sub>u</sub> = {C.nb(fu)}" if fu else f"{nom} f<sub>y</sub> = {C.nb(fy)}")
    if aciers:
        h.append("Aciers · " + " ; ".join(aciers) + " N/mm²")
    if R.get("boulon"):
        cl = u.get("classe") or ""
        d0 = R.get("d_0")
        h.append(f"Boulons <b>{C.esc(R.boulon)} {C.esc(cl)}</b>" + (f" · d<sub>0</sub> = {C.nb(d0)} mm" if d0 else "")
                 + (f" · f<sub>ub</sub> = {C.nb(R.f_ub)} N/mm²" if R.get("f_ub") else ""))
    gs = [(k, v) for k, v in u.items() if k.startswith("g_M") and v is not None and cfg.E.champ_visible(u, k)]
    if gs:
        h.append(" · ".join(f"γ<sub>{C.esc(k[2:].replace('n', ',net').replace('s', ',ser'))}</sub> = {C.nb(v)}" for k, v in gs))
    if cfg.hypotheses:                            # pied de poteau : les tiges et le béton
        h += cfg.hypotheses(R, u)
    return h


def volets(cfg: Config, R, u: Dict, el: Dict) -> str:
    """Les fenêtres de calcul : une par pièce (tableau des vérifications, détail des formules), la géométrie
    (Tableau 3.3 et paramètres retenus), le prédimensionnement."""
    hyps = _hypotheses(cfg, R, u)
    out = ""
    sans_objet = [_court(cfg, c) for c in R.checks if not c.active]
    for grp, nom, eta, ok, pire, on, info in _groupes(cfg, R):
        lignes = [_ligne_vtab(cfg, c, c is pire and not info, R) for c in on]
        detail = (cfg.blocs_tete(R, grp) or "") if cfg.blocs_tete else ""
        for c in on:
            aide, lf = cfg.formules(R, c)
            detail += (f'<div class="ck-g"><div class="ck-t">{C.esc(_court(cfg, c))} · {C.esc(c.nat)}</div>'
                       + (C.aide(math_html(aide)) if aide else "") + C.lignes_fx(lf) + "</div>")
        for titre, tab in (cfg.tableaux_boulons(R) if cfg.tableaux_boulons and grp == "Boulons" else []):
            if tab:
                detail += (f'<div class="ck-g asm-large"><div class="ck-t">Répartition élastique par boulon · {C.esc(titre)}</div>'
                           + tableau(["Rangée / file", "x (mm)", "y (mm)", "F<sub>x</sub> (kN)", "F<sub>z</sub> (kN)", "F (kN)"],
                                     [[f"{b.i} / {b.j}", C.fr(b.x, 1), C.fr(b.y, 1), C.fr(b.fx, 2), C.fr(b.fz, 2), f"<b>{C.fr(b.f, 2)}</b>"]
                                      for b in tab], droite=(1, 2, 3, 4, 5), classe="asm-tab") + "</div>")
        if cfg.blocs:
            detail += cfg.blocs(R, grp) or ""
        pied = (f'<div class="ck-autres">{C.esc(cfg.N.LEGENDE_REFERENCES)}'
                + (f"<br>Sans objet dans cette configuration : {C.esc(' ; '.join(sans_objet))}." if sans_objet else "")
                + "</div>")
        sous = f"{C.esc(cfg.court)} · {C.esc(_court(cfg, pire))} · {C.esc(cfg.normes.get(grp, 'EN 1993-1-8'))}"
        out += C.fenetre_calcul(_slug(grp), C.esc(nom), sous, hyps, lignes, detail, pied)
    out += _volet_geometrie(cfg, R, hyps)
    if cfg.predim and R.get("pd") is not None:
        out += _volet_predim(cfg, R, u, hyps)
    return out


def _taux_distance(x) -> float:
    """Le rapport d'une pince à sa limite (e,min / e, ou e / e,max), une PROJECTION pour la jauge et la couleur de la
    ligne du Tableau 3.3 (l'écran Streamlit n'y montre que ✓ / ✗) ; le verdict reste celui du moteur (``x.ok``)."""
    t = 0.0
    if x.min and x.val:
        t = max(t, x.min / x.val)
    elif x.min and not x.val:
        t = 9.99
    if x.max:
        t = max(t, x.val / x.max)
    # une pince que le moteur déclare conforme (sa tolérance : val ≥ min − 0,001 mm) ne passe pas 100 % : p₁ = 48,4 mm
    # pour 2,2·d₀ = 48,400000000000006 mm s'écrivait « 101 % » rouge, quand le Tableau 3.3 la dit conforme
    return min(t, 1.0) if x.ok else min(t, 9.99)


def _volet_geometrie(cfg: Config, R, hyps) -> str:
    dist = list(R.get("dist") or [])
    lignes = []
    for x in dist:
        t = _taux_distance(x)
        niv = "r" if not x.ok else C.niveau(t)
        lim = f"min {C.fr(x.min, 1)}" + (f" · max {C.fr(x.max, 0)}" if x.max is not None else "")
        regle = (x.rule or "").split(";")[0].strip()
        lignes.append({"nom": C.esc(x.lab), "sous": C.esc(x.rule), "formule": math_html(regle) if regle else "",
                       "valeur": f"{C.fr(x.val, 1)} mm", "limite": f"{lim} mm", "taux": C.pct(t, 1), "niveau": niv, "gov": not x.ok})
    params = cfg.parametres(R) if cfg.parametres else []
    detail = ""
    if params:
        detail = ('<div class="ck-g asm-large"><div class="ck-t">Paramètres retenus</div>'
                  + tableau(["Paramètre", "Valeur"], [[C.esc(a), f"<b>{C.esc(b)}</b>"] for a, b in params], droite=(1,), classe="asm-tab asm-param")
                  + "</div>")
    regle = getattr(cfg, "regle_33", None) or ("e ≥ 1,2·d<sub>0</sub> ; p<sub>1</sub> ≥ 2,2·d<sub>0</sub> ; "
                                                 "p<sub>2</sub> ≥ 2,4·d<sub>0</sub>")
    pied = (f'<div class="ck-autres">{regle} ; '
            'p ≤ min(14t ; 200 mm) ; e ≤ 4t + 40 mm si exposé (EN 1993-1-8 Tableau 3.3).</div>') if dist else ""
    if cfg.legende_33:                            # pied de poteau : la règle que l'écran écrit sous son Tableau 3.3
        pied = f'<div class="ck-autres">{cfg.legende_33}</div>'
    verdict = "conformes" if R.get("dist_ok", True) else "NON conformes"
    return C.fenetre_calcul("geometrie", "Géométrie · pinces et entraxes", f"Tableau 3.3 : {verdict} · paramètres retenus",
                            hyps, lignes, detail, pied)


def _volet_predim(cfg: Config, R, u: Dict, hyps) -> str:
    p = R.pd
    lignes = []
    for r in _solutions(cfg, R):
        lignes.append({"nom": C.esc(cfg.predim["nom"](r)), "sous": C.esc(f"{r.b['n']} · {int(r.n)} rangées"
                                                                         + (f" · {r.get('var')}" if r.get("var") else "")),
                       "formule": math_html(geo_solution(cfg, r)),
                       "valeur": "proposée" if r is p.pick else "", "limite": f"cible {C.pc(C.pct(C.lire_nombre(u.get('eta_c')) or 0.9, 0))} %",
                       "taux": taux_estime(r.eta)[0], "niveau": taux_estime(r.eta)[1], "gov": r is p.pick})
    detail = ('<div class="ck-g asm-large"><div class="ck-t">Détail de la solution retenue par défaut</div>'
              + tableau(["", ""], [[C.esc(a), f"<b>{C.esc(b)}</b>"] for a, b in detail_predim(cfg, p)], droite=(1,), classe="asm-tab")
              + "</div>")
    if p.get("reqs"):
        detail += ('<div class="ck-g asm-large"><div class="ck-t">Épaisseur nécessaire par vérification</div>'
                   + tableau(["Vérification", "Épaisseur"], [[C.esc(q[0]), f"{C.fr(q[1], 2)} mm"] for q in p.reqs], droite=(1,),
                             classe="asm-tab") + "</div>")
    pied = (f'<div class="ck-autres">{C.esc(regle_predim(cfg))} La ligne « Proposition » du résultat ouvre les solutions ; '
            'en choisir une la recopie dans les données et repasse en vérification.</div>')
    return C.fenetre_calcul("predim", "Prédimensionnement", C.esc(p.msg or ""), hyps, lignes[:12], detail, pied)


def detail_predim(cfg: Config, p) -> List:
    """Le tableau « Détail de la solution retenue par défaut » de l'onglet Prédim (lignes [libellé, valeur], textes
    de l'écran Streamlit — copies nommées de chaque famille) ; à défaut, le boulon, les rangées, les pinces et l'état."""
    f = cfg.predim.get("detail")
    if f:
        return [[str(a), str(b)] for a, b in f(p)]
    return [["Boulon proposé", str(p.boulon)], ["Rangées n1", C.nb(p.n) if p.get("n") is not None else "—"],
            ["e1 / p1 / e2", f"{C.nb(p.e1)} / {C.nb(p.p1)} / {C.nb(p.e2)} mm"], ["État", str(p.msg)]]


def regle_predim(cfg: Config) -> str:
    return cfg.predim.get("regle") or ("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus "
                                       "petit diamètre. Seule la vérification détaillée fait foi.")


def _solutions(cfg: Config, R) -> List:
    """Les solutions du prédimensionnement offertes (onglet Prédim de Streamlit : taux ≤ 1, la cible d'abord, puis
    le score) — au plus neuf."""
    p = R.pd
    rows = [r for r in p.rows if r.geom and r.prop]
    sols = sorted([r for r in rows if r.eta <= 1], key=lambda r: (0 if r.ok else 1, r.score if r.ok else r.eta))[:9]
    return sols


def _bilan(cfg: Config, el: Dict) -> Dict:
    """Le calcul d'un assemblage et son résumé (liste, note)."""
    refus = saisies_refusees(cfg, el["values"])
    R, vides, err = calculer(cfg, el)
    if err:
        return {"R": None, "taux": None, "niveau": "x", "statut": "Calcul impossible", "erreur": err, "vides": vides,
                "refus": refus}
    if vides or refus:
        return {"R": R, "taux": None, "niveau": "c", "statut": "À compléter" if vides else "À corriger", "erreur": None,
                "vides": vides, "refus": refus}
    niv = niveau_R(R)
    return {"R": R, "taux": eta_R(R), "niveau": niv, "statut": C.STATUTS[niv][1], "erreur": None, "vides": [], "refus": []}


def resume_note(etat: Dict, bilans: List[Dict]) -> Dict:
    """Le résumé de la NOTE : son assemblage le plus chargé ; un assemblage non vérifié sous 100 % (géométrie)
    passe devant un assemblage vérifié."""
    vals = [((b["niveau"] == "r", b["taux"]), i) for i, b in enumerate(bilans) if b["taux"] is not None]
    if not vals:
        b0 = bilans[etat["actif"]]
        return {"taux": None, "niveau": b0["niveau"], "statut": b0["statut"], "gouvernant": ""}
    _, i = max(vals)
    b = bilans[i]
    el = etat["assemblages"][i]
    cfg = pour(_cfg(el), b["R"])
    g = b["R"].gov
    gouv = f"{_repere(el, i)} · {cfg.court}" + (f" · {g.grp} · {_court(cfg, g)}" if g else "")
    return {"taux": taux(b["taux"]), "niveau": b["niveau"], "statut": b["statut"], "gouvernant": gouv}


def elements(etat: Dict, bilans: List[Dict]) -> str:
    """« Dans cette note » : chaque assemblage (repère, configuration) et son taux ; un clic l'ouvre."""
    items = ""
    n = len(etat["assemblages"])
    for i, (el, b) in enumerate(zip(etat["assemblages"], bilans)):
        t = b["taux"]
        txt = f"{C.pc(taux(t))}{C.NBSP}%" if t is not None else ("—" if b["niveau"] == "x" else "à compléter")
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if n > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(_repere(el, i))} '
                  f'<span>· {C.esc(_cfg(el).court)}</span></span><span class="pct {b["niveau"]}">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {n}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier l\'assemblage ouvert">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter un assemblage</div></div>')


def _messages(cfg: Config, R) -> str:
    out = ""
    note = getattr(cfg, "note_information", None)
    if note and note(R):
        out += C.message(C.esc(note(R)))
    for a in R.alerts:
        titre, det = cfg.N.ligne_alerte(a, R)
        out += C.message(f"<b>{C.esc(titre)}</b>" + (f" — {C.esc(det)}" if det else ""), grave=a.block)
    if R.get("reserve"):
        out += C.message("Vérifié sous réserve : voir les remarques ci-dessus et la fenêtre de la pièce concernée.")
    return out


def _minuscule(s: str) -> str:
    """Un nom de vérification après « Pièce · » : sa première lettre en minuscule, le reste tel quel (« Ailes B – gorge
    minimale » → « ailes B – … » ; un ``lower()`` écrivait « ailes b », « groupe s », « tab. 3.3 »), un sigle intact."""
    if len(s) > 1 and s[1].isupper():
        return s
    return s[:1].lower() + s[1:]


def _verdict(cfg: Config, R, niv: str) -> str:
    g = R.gov
    if g is None:
        return f'<div class="verdict">{C.chip(taux(eta_R(R)), niv, xl=True)}<div class="gov"><b>{C.esc(R.statut)}</b></div></div>'
    motif = ""
    if not R.geo_ok:
        motif = " · géométrie non valide"
    elif not R.dist_ok:
        motif = " · pinces ou entraxes non conformes"
    elif R.verified and (R.get("reserve") or R.alerts):
        motif = " · sous réserve des alertes"
    suite = cfg.complement_verdict(R) if cfg.complement_verdict else ""
    return (f'<div class="verdict">{C.chip(taux(eta_R(R)), niv, xl=True)}<div class="gov"><b>{C.esc(g.grp)} · {C.esc(_minuscule(_court(cfg, g)))}</b>'
            f'{_comparaison(cfg, g)}{suite}{C.esc(motif)}</div></div>')


def _rangee_geometrie(R) -> str:
    """La ligne « Géométrie » du résultat : le Tableau 3.3 (pinces et entraxes), conforme ou non selon le moteur."""
    dist = list(R.get("dist") or [])
    t = max([_taux_distance(x) for x in dist] or [0.0])
    niv = "r" if not (R.get("dist_ok", True) and R.get("geo_ok", True)) else C.niveau(t)
    n = len(dist)
    txt = (f"{n} pinces et entraxes" if n > 1 else "1 distance au bord") if dist else "géométrie"
    return C.ligne_res("Géométrie", f"Tab. 3.3 · {txt}", C.pct(t), niv, calcul="geometrie")


def resultat(etat: Dict) -> Dict:
    el = _actif(etat)
    cfg = _cfg(el)
    u = el["values"]
    bilans = [_bilan(_cfg(e), e) for e in etat["assemblages"]]
    b = bilans[etat["actif"]]
    R = b["R"]
    pied = (f'<div class="pfoot"><div class="mass"><span>{C.esc(cfg.titre_famille)} · {C.esc(cfg.nom_menu or cfg.court)}</span>'
            f'<b>EN 1993-1-8</b></div><div class="legend">{C.legende()}</div></div>')
    if R is None:
        verdict = (f'<div class="verdict"><div class="gov"><b>Calcul impossible</b>Données incomplètes : {C.esc(b["erreur"])}</div></div>')
        return {"resultat": f'<div class="pbody">{verdict}</div>{pied}', "volets": "", "vues": [], "phrase": C.esc(cfg.titre),
                "statut_html": C.statut("x"), "resume": resume_note(etat, bilans), "donnees": formulaire(etat, None),
                "elements": elements(etat, bilans)}
    if b["vides"] or b["refus"]:
        corps = ""
        if b["vides"]:
            noms = ", ".join(_nom_champ(cfg, k) for k in b["vides"])
            corps = (f'<div class="verdict"><div class="gov"><b>À compléter</b>Dimensions à saisir : {C.esc(noms)}. Vérification '
                     f'suspendue ; le dessin garde des dimensions provisoires.</div></div>')
        if b["refus"]:
            if not corps:
                corps = ('<div class="verdict"><div class="gov"><b>À corriger</b>Des données relues de la note ne sont pas '
                         'admises (l\'écran Streamlit les calculerait en silence, V<sub>Ed</sub> vide pris égal à 0, '
                         'γ nul…). Vérification suspendue jusqu\'à leur correction dans le cadre Données.</div></div>')
            corps += "".join(C.message(C.esc(msg), grave=True) for _k, msg in b["refus"])
        html_res, vol = f'<div class="pbody">{corps}</div>{pied}', ""
        niv = "c"
    else:
        cfg = pour(cfg, R)
        niv = niveau_R(R)
        rows = ""
        gouv = R.gov
        # lignes de plus (pied de poteau : les quatre critères de l'épaisseur requise, sous la ligne Platine) — un
        # texte va à la fin, un dict {groupe: html} sous la ligne de ce groupe
        extra = cfg.lignes_resultat(R, u) if cfg.lignes_resultat else ""
        for grp, nom, eta, ok, pire, on, info in _groupes(cfg, R):
            texte, pk = cfg.piece(R, u, grp)
            if info:
                texte = (texte + " · " if texte else "") + "information"
            vu = eta if info else eta_vu(eta, ok)
            rows += C.ligne_res(nom, texte, taux(vu), "c" if info else ("r" if not ok else C.niveau(vu)), calcul=_slug(grp),
                                pk=pk or "", gov=bool(gouv is not None and gouv in on and not info))
            if isinstance(extra, dict):
                rows += extra.pop(grp, "")
        rows += "".join(extra.values()) if isinstance(extra, dict) else extra
        rows += _rangee_geometrie(R)
        if cfg.predim and R.get("pd") is not None and R.pd.get("pick") is not None:
            pick = R.pd.pick
            rows += f'<div class="elgrp">{"Mode prédimensionnement" if R.pred else "Prédimensionnement"}</div>'
            rows += C.ligne_res("Proposition", C.esc(cfg.predim["nom"](pick)), *taux_estime(pick.eta, 0),
                                calcul="predim", pk="solution")
        msg = ""
        if R.pred:
            msg += C.message(f"<b>Mode prédimensionnement</b> — {C.esc(cfg.predim['label'](R.pd.pick) if R.pd.get('pick') is not None else '')}. "
                             f"{C.esc(R.pd.msg)}")
        msg += _messages(cfg, R)
        html_res = f'<div class="pbody">{_verdict(cfg, R, niv)}<div class="els">{rows}</div>{msg}</div>{pied}'
        vol = volets(cfg, R, u, el)
    cfg = pour(cfg, R)
    return {"resultat": html_res, "volets": vol, "vues": vues(cfg, R, el), "phrase": phrase(cfg, u),
            "statut_html": C.statut(niv), "resume": resume_note(etat, bilans), "donnees": formulaire(etat, R),
            "elements": elements(etat, bilans)}


def phrase(cfg: Config, u: Dict) -> str:
    """La phrase sous le titre du premier dessin : la configuration, les pièces et les efforts (acier/projet.description,
    la ligne de la liste du projet Streamlit)."""
    d = PJ.description(u)
    return C.esc(cfg.court + (" · " + d if d else ""))


# ===================================================================== les vues de la toile
def _hl(R) -> Dict:
    """Les cotes et pièces en cause dans les alertes, en rouge sur les dessins (Streamlit : celles de l'alerte
    sélectionnée ; le site les montre toutes)."""
    dims, elems = set(), set()
    for a in R.alerts:
        dims |= set(a.dims)
        if a.block:
            elems |= set(a.elems)
    return {"dims": dims, "elems": elems} if (dims or elems) else None


def vues(cfg: Config, R, el: Dict) -> List[Dict]:
    """Les dessins du moteur (``schemas`` de la configuration), puis le prédimensionnement, la note texte, la méthode
    et la validation (les onglets de l'écran Streamlit), en tableaux."""
    niveau = int(el.get("cotation", 1))
    out = []
    for cle, titre, court, dessin, legende in cfg.vues(R, niveau, _hl(R)):
        out.append({"cle": cle, "titre": titre, "court": court, "svg": dessin.svg(identifiant=f"asm_{cle}"), "legende": legende})
    if cfg.predim and R.get("pd") is not None:
        out.append({"cle": "predim", "titre": "Prédimensionnement", "court": "Prédim", "svg": vue_predim(cfg, R, el["values"]),
                    "legende": "toutes les combinaisons géométriquement possibles"})
    out.append({"cle": "note", "titre": "Note texte", "court": "Note", "svg": vue_note(cfg, R, el),
                "legende": "à copier dans un courriel ou un rapport"})
    out.append({"cle": "methode", "titre": "Méthode", "court": "Méthode", "svg": vue_methode(cfg, R),
                "legende": "le modèle mécanique, ses sources et ses limites"})
    if cfg.validation:                            # pied de poteau : l'onglet Benchmark de chaque mode
        out.append({"cle": "validation", "titre": "Validation", "court": "Validation", "svg": cfg.validation(cfg),
                    "legende": "calculs manuels indépendants rejoués par le moteur"})
    elif cfg.bench:
        out.append({"cle": "validation", "titre": "Validation", "court": "Validation", "svg": vue_validation(cfg),
                    "legende": "exemples publiés rejoués par le moteur"})
    return out


def vue_predim(cfg: Config, R, u: Dict) -> str:
    """L'onglet Prédim de Streamlit : la solution retenue par défaut, l'épaisseur nécessaire par vérification, puis
    toutes les combinaisons géométriquement possibles (taux estimés)."""
    p = R.pd
    rows = [r for r in p.rows if r.geom]
    cols = cfg.predim.get("colonnes", [])
    if callable(cols):                       # des colonnes qui dépendent du résultat (platine sur poteau : le cas d'appui)
        cols = cols(R)
    entetes = ["Boulon", "n<sub>1</sub>"] + [h for h, _ in cols] + ["η estimé", ""]
    lignes = []
    for r in rows:
        ch = C.chip(*taux_estime(r.eta)) if math.isfinite(r.eta) else "—"
        mark = "★ retenue" if r is p.pick else ("" if r.ok else "au-delà de la cible")
        lignes.append([f"<b>{C.esc(r.b['n'])}</b>", C.nb(r.n)] + [C.esc(f(r)) for _, f in cols] + [ch, C.esc(mark)])
    head = (f'<div class="asm-kpi"><div><span>Proposition</span><b>{C.esc(cfg.predim["label"](p.pick)) if p.get("pick") is not None else "aucune"}</b></div>'
            f'<div><span>Taux estimé</span><b>{C.pc(taux_estime(p.eta or 0)[0])} %</b></div><div><span>État</span><b>{C.esc(p.msg)}</b></div></div>')
    det = ("<h4>Détail de la solution retenue par défaut</h4>"
           + tableau(["", ""], [[C.esc(a), f"<b>{C.esc(b)}</b>"] for a, b in detail_predim(cfg, p)], droite=(1,), classe="asm-tab asm-cle"))
    reqs = ""
    if p.get("reqs"):
        reqs = ("<h4>Épaisseur nécessaire par vérification</h4>"
                + tableau(["Vérification", "Épaisseur"], [[C.esc(q[0]), f"{C.fr(q[1], 2)} mm"] for q in p.reqs], droite=(1,),
                          classe="asm-tab asm-cle"))
    corps = (head + det + reqs + "<h4>Toutes les combinaisons géométriquement possibles</h4>"
             + tableau(entetes, lignes, droite=tuple(range(1, len(entetes) - 1)), classe="asm-tab")
             + f'<p class="asm-note">{C.esc(regle_predim(cfg))} Pour recopier une solution dans les données : la ligne '
               '« Proposition » du résultat.</p>')
    return f'<div class="asm-doc">{corps}</div>'


def vue_note(cfg: Config, R, el: Dict) -> str:
    """L'onglet Note de Streamlit : l'export texte (essentiel, ou toutes les vérifications avec leurs formules)."""
    t = cfg.T.construire_texte(R, bool(el.get("texte_complet")))
    return f'<div class="asm-doc"><pre class="asm-txt">{C.esc(t)}</pre></div>'


def non_reverifies(cfg: Config, R) -> tuple:
    """Ce que l'écran Streamlit déclare « non revérifié » (son repli Périmètre) : une liste fixe, ou une fonction du
    résultat (poutre–poteau : les limites dépendent de l'appui, aile ou âme)."""
    nr = cfg.non_reverifies
    return tuple(nr(R) if callable(nr) else nr or ())


def vue_methode(cfg: Config, R) -> str:
    """L'onglet Méthode de Streamlit, et le périmètre de la vérification (ce qui n'est pas revérifié, ce qui est
    sans objet dans cette configuration)."""
    sans = [_court(cfg, c) for c in R.checks if not c.active]
    # ce que l'écran Streamlit ajoute au « sans objet » (poutre–poteau sur l'aile : pourquoi l'aile n'est pas vérifiée)
    sans += list(getattr(cfg, "sans_objet", None) and cfg.sans_objet(R) or ())
    per = ""
    nr = non_reverifies(cfg, R)
    if nr or sans:
        per = "<h4>Périmètre de la vérification</h4><ul>"
        if nr:
            per += f"<li><b>{C.esc(cfg.titre_non_verifies or 'Non revérifiés ici')}</b> : " + C.esc(" ; ".join(nr)) + ".</li>"
        if sans:
            per += "<li><b>Sans objet dans cette configuration</b> : " + C.esc(" ; ".join(sans)) + ".</li>"
        per += "</ul>"
    nat = cfg.natures or (("EC", "formule de l'Eurocode"), ("COMP", "modèle complémentaire reconnu"),
                          ("INT", "interprétation de l'outil"))
    natures = ("<h4>Natures des vérifications</h4><ul>" + "".join(f"<li><b>{C.esc(n)}</b> : {C.esc(t)}</li>" for n, t in nat)
               + f'</ul><p class="asm-note">{C.esc(cfg.N.LEGENDE_REFERENCES)}</p>')
    return f'<div class="asm-doc">{markdown(cfg.methode)}{per}{natures}</div>'


# la légende des calculs manuels indépendants de l'onglet Benchmark (ecran_resultats.onglet_benchmark : copie nommée,
# test : tests/test_api_web_assemblages.py la relit dans le source de chaque écran) et le nom de sa colonne de contrôle
LEGENDE_MANUEL = ("Colonne « manuel » : calcul indépendant hors moteur ; colonne « contrôle » : même valeur recomposée à "
                  "la main (arrondie).", "Contrôle")


def vue_validation(cfg: Config) -> str:
    """L'onglet Benchmark de Streamlit : les exemples publiés et les calculs manuels, rejoués par le moteur."""
    B = cfg.bench.run_bench()
    S = getattr(cfg.bench, "SRC", {}) or {}
    url = f" Document : {C.esc(S['url'])}." if S.get("url") else ""      # écrit, jamais chargé (aucune ressource externe)
    out = (f'<div class="asm-kpi"><div><span>Benchmark du module</span><b>{C.esc(B["statut"])}</b></div>'
           f'<div><span>Source</span><b>{C.esc(S.get("titre", ""))}</b></div></div>'
           f'<p class="asm-note">{C.esc(S.get("org", ""))}, {C.esc(S.get("annee", ""))}. Norme : {C.esc(S.get("norme", ""))}.{url} '
           f'Ces tableaux sont recalculés par le moteur à chaque ouverture ; ils ne dépendent pas des données saisies. '
           f'Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; &gt; 3 % à vérifier. Aucune formule n\'a été ajustée pour coller à la source. '
           f'Pour rejouer un exemple : section « Cartouche et affichage », « Exemples publiés ».</p>')
    for b in B["bench"]:
        lignes = []
        for r in b["rows"]:
            lab = r["lab"] + ((" — " + r["exp"]) if (r["exp"] and abs(r["ec"]) > 0.003) else "")
            # les décimales de l'onglet Benchmark de la configuration (platine sur poteau : 3 pour un rapport sans unité)
            nd, nv = cfg.decimales_bench(r) if getattr(cfg, "decimales_bench", None) else ((1, 2) if r["ref"] < 10 else (0, 1))
            un = "" if r["unit"] in ("-", "") else " " + C.esc(r["unit"])        # un rapport n'a pas d'unité
            lignes.append([C.esc(lab), f'<span class="nw">{C.esc(r["page"])}</span>', f'{fr_js(r["ref"], nd)}{un}',
                           fr_js(r["val"], nv),
                           f'{fr_js(r["ec"] * 100, 2)} %', C.esc(r["st"])])
        out += (f"<h5>{C.esc(b['id'])} · {C.esc(b['titre'])}</h5><p class=\"asm-note\">Données : {C.esc(b['data'])}</p>"
                + tableau(["Vérification", "Page", "Référence", "Outil", "Écart", ""], lignes, droite=(2, 3, 4), classe="asm-tab"))
    if getattr(cfg.bench, "NOCOVER", None):
        out += "<h4>Vérifications sans exemple publié consulté</h4><ul>" + "".join(
            f"<li><b>{C.esc(x[0])}</b> — {C.esc(x[1])}</li>" for x in cfg.bench.NOCOVER) + "</ul>"
    for c in B.get("valid", []):
        lignes = []
        for r in c["rows"]:
            main = r.get("hand", r.get("main", 0.0)) or 0.0
            ctrl = r.get("xls", r.get("ref", main))
            n = 4 if abs(main) < 10 else (2 if abs(main) < 1000 else 0)
            un = "" if r.get("unit", "-") == "-" else " " + r["unit"]
            st = r.get("st") or ("OK" if r.get("ok", True) else "À VÉRIFIER")
            lignes.append([C.esc(r["lab"]), C.esc(fr_js(main, n) + un), fr_js(ctrl, n), fr_js(r.get("val", 0.0), n),
                           f'{fr_js(r.get("ec", 0.0) * 100, 3)} %', C.esc(st)])
        legende, colonne = getattr(cfg, "legende_manuel", None) or LEGENDE_MANUEL
        out += (f"<h5>{C.esc(c['id'])} · calcul manuel indépendant</h5><p class=\"asm-note\">{C.esc(c['titre'] + '. ' + legende)}</p>"
                + tableau(["Grandeur", "Manuel", C.esc(colonne), "Outil", "Écart", ""], lignes, droite=(1, 2, 3, 4), classe="asm-tab"))
    return f'<div class="asm-doc">{out}</div>'


# ===================================================================== les sélecteurs
def _avec(cfg: Config, u: Dict, **k):
    """Le calcul de l'assemblage avec quelques saisies changées (une ligne de sélecteur)."""
    v = dict(u)
    v.update(k)
    try:
        return moteur(cfg, v)
    except Exception:  # noqa: BLE001 — une ligne impossible (profilé trop petit…) se montre en rouge, sans arrêter la liste
        return None


def _ligne_sel(nom: str, col: str, R, cur: bool, prop: str) -> Dict:
    if R is None:
        return {"nom": nom, "col": col, "taux": 999, "niveau": "r", "cur": cur, "opt": False, "prop": prop + " · calcul impossible", "_ok": False}
    return {"nom": nom, "col": col, "taux": C.pct(min(eta_R(R), 9.99)), "niveau": niveau_R(R), "cur": cur, "opt": False,
            "prop": prop + f" · {C.esc(R.statut.lower())}", "_ok": bool(R.verified)}


def _marquer_optimal(rows: List[Dict]) -> List[Dict]:
    for r in rows:
        if r["_ok"]:
            r["opt"] = True
            break
    return rows


def _debut(rows: List[Dict]) -> int:
    i_o = next((i for i, r in enumerate(rows) if r["opt"]), 0)
    i_c = next((i for i, r in enumerate(rows) if r["cur"]), i_o)
    return max(0, min(i_o, i_c) - 4)


def sel_boulon(cfg: Config, u: Dict, familles=None) -> str:
    rows = []
    for b in DB["boulons"]:
        R = _avec(cfg, u, boulon_u=b["n"])
        rows.append(_ligne_sel(b["n"], C.fr(b["As"], 0), R, b["n"] == u.get("boulon_u"),
                               f"{b['n']} · d<sub>0</sub> = {C.nb(b['d0'])} mm · A<sub>s</sub> = {C.fr(b['As'], 1)} mm²"))
    _marquer_optimal(rows)
    return C.selecteur("boulon", f"Boulons · classe {C.esc(u.get('classe') or '')}", "triés par diamètre · un clic retient le boulon",
                       "", rows, _debut(rows), "A<sub>s</sub> mm²", etoile="le plus petit qui tient")


def _fam(nom: str) -> str:
    return (nom or "").split()[0]


FAMILLES_PROFILS = sorted({_fam(p["n"]) for p in DB["profils"]}, key=lambda f: ["IPE", "HEA", "HEB", "HEM"].index(f) if f in ("IPE", "HEA", "HEB", "HEM") else 9)


def sel_profil(cle: str, titre: str):
    """Le sélecteur d'un profilé du catalogue des assemblages (acier/bibliotheques) : par famille (puces), trié par
    hauteur ; le taux de l'assemblage avec chaque profilé."""
    def rendre(cfg: Config, u: Dict, familles=None) -> str:
        cur = u.get(cle)
        fams = [f for f in (familles or []) if f in FAMILLES_PROFILS] or [_fam(cur) if _fam(cur) in FAMILLES_PROFILS else FAMILLES_PROFILS[0]]
        rows = []
        for p in sorted((p for p in DB["profils"] if _fam(p["n"]) in fams), key=lambda p: (FAMILLES_PROFILS.index(_fam(p["n"])), p["h"], p["b"])):
            R = _avec(cfg, u, **{cle: p["n"]})
            rows.append(_ligne_sel(p["n"], C.fr(p["h"], 0), R, p["n"] == cur,
                                   f"{p['n']} · h {C.nb(p['h'])} · b {C.nb(p['b'])} · t<sub>w</sub> {C.nb(p['tw'])} · t<sub>f</sub> {C.nb(p['tf'])} mm"))
        _marquer_optimal(rows)
        filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{f}">{f}</button>' for f in FAMILLES_PROFILS)
        return C.selecteur(cle, titre, "triés par hauteur · un clic retient le profilé", filtres, rows, _debut(rows), "h mm",
                           etoile="le plus petit qui tient")
    return rendre


def _choisir_liste(cle: str, liste: str):
    def choisir(cfg: Config, u: Dict, nom: str) -> None:
        find(DB[liste], nom)                      # un nom inconnu lève ValueError
        u[cle] = valeur_widget(cfg, cle, nom)
    return choisir


def _selecteurs(cfg: Config) -> Dict:
    s = {"boulon": {"rendre": sel_boulon, "choisir": _choisir_liste("boulon_u", "boulons")}} if "boulon_u" in cfg.E.CHAMPS else {}
    titres = getattr(cfg, "titres_profils", {})
    for cle, titre in (("prof_S", "Poutre secondaire · profilé"), ("prof_P", "Poutre principale · profilé")):
        if cle in cfg.E.CHAMPS:
            s[cle] = {"rendre": sel_profil(cle, titres.get(cle, titre)), "choisir": _choisir_liste(cle, "profils")}
    if cfg.predim:
        s["solution"] = {"rendre": sel_solution, "choisir": choisir_solution}
    s.update(cfg.selecteurs)
    return s


def _nom_solution(cfg: Config, R, r) -> str:
    noms = [cfg.predim["nom"](x) for x in R.pd.rows if x.geom and x.prop]
    n = cfg.predim["nom"](r)
    return n if noms.count(n) <= 1 else f"{n} ({r.get('var') or r.get('i')})"


def sel_solution(cfg: Config, u: Dict, familles=None) -> str:
    R = moteur(cfg, u)
    p = R.pd
    rows = []
    for r in [x for x in p.rows if x.geom and x.prop]:
        applique = valeurs_normales(cfg, cfg.predim["apply"](u, r))
        cur = all(applique.get(k) == u.get(k) for k in applique if k != "mode_calc") and not R.pred
        t, niv = taux_estime(r.eta, 0)
        rows.append({"nom": _nom_solution(cfg, R, r), "col": C.nb(cfg.predim["col"](r)), "taux": t,
                     "niveau": niv, "cur": cur, "opt": r is p.pick,
                     "prop": C.esc(cfg.predim["label"](r) + " · " + geo_solution(cfg, r) + " · taux estimé"
                                   + ("" if r.ok else " au-dessus du taux cible")
                                   + ("" if math.isfinite(r.eta) else " (indéterminé)"))})
    return C.selecteur("solution", "Solutions du prédimensionnement", "taux estimé · un clic recopie la solution et repasse en vérification",
                       "", rows, _debut([dict(r, _ok=r["opt"]) for r in rows]), cfg.predim["unite_col"],
                       etoile="proposée par défaut (le moins de rangées)")


def geo_solution(cfg: Config, r) -> str:
    """Les pinces et entraxes d'une solution, comme sur sa carte de l'onglet Prédim de Streamlit."""
    f = cfg.predim.get("geo")
    return f(r) if f else f"e1 {C.nb(r.e1)} / p1 {C.nb(r.p1)} / e2 {C.nb(r.e2)} mm"


def choisir_solution(cfg: Config, u: Dict, nom: str) -> None:
    """Une solution du prédimensionnement recopiée dans les saisies (``moteur.apply_solution`` : boulons, rangées,
    pinces, pièce ; repasse en VÉRIFICATION), normalisée comme l'écran Streamlit la réécrit."""
    R = moteur(cfg, u)
    for r in [x for x in R.pd.rows if x.geom and x.prop]:
        if _nom_solution(cfg, R, r) == nom:
            u.update(valeurs_normales(cfg, cfg.predim["apply"](u, r)))
            return
    raise ValueError(f"solution inconnue : {nom}")


def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    el = _actif(etat)
    cfg = _cfg(el)
    s = _selecteurs(cfg).get(role)
    if s is None:
        raise ValueError(f"sélecteur inconnu : {role}")
    return s["rendre"](cfg, el["values"], familles)


# ===================================================================== les familles
# chargées en DERNIER : elles décrivent leurs configurations avec les briques ci-dessus (enregistrer, champs…)
from . import assemblages_poutre  # noqa: E402,F401
from . import assemblages_poteau  # noqa: E402,F401
from . import assemblages_pied  # noqa: E402,F401
