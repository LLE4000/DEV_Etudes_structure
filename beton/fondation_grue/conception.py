# -*- coding: utf-8 -*-
"""Fondations de grues — les OPTIONS DE CONCEPTION : ce que le plan d'armatures demande et que le calcul ne fixe pas.
Bibliothèque standard seulement.

Ce sont les retours des barres, la longueur des zones d'étriers serrés, les barres latérales et leur recouvrement, les
coupes, les cages de renfort au droit des pieux, les barres scellées des plaques (les perfos), les niveaux, les textes
du plan et la mise en page des feuilles. Elles ne servent qu'au DESSIN : le fichier « rbar.ouvrage » que RBAR dessine
(``beton/fondation_grue/ouvrage.py``). Le calcul ne les lit jamais ; aucune vérification n'en dépend.

LES VALEURS PAR DÉFAUT sont celles du chantier Greenwood (décision du propriétaire du 12/11/2026) : les valeurs que la
conversation RBAR a relevées sur le plan de référence « ARMATURE FONDATION GRUE TOUR GT1 » (plan GWW VAL COF 001,
indice A du 22-04-25) pour écrire son fichier d'exemple (dépôt RBAR : ``tests/Rbar.Core.Tests/Fixtures/
ouvrage-exemple.json``, version 0.4.27). Elles sont PROVISOIRES : la valeur que le bureau veut donner à chacune est une
question posée au propriétaire (``docs/beton/FONDATIONS_GRUE_RBAR.md``, « Les questions sur les options de
conception »). Ses réponses iront ici (les défauts), dans le classeur, ou dans le code des règles.

Elles vivent DANS la fondation (``f["conception"]``) : une note exporte les mêmes plans sur n'importe quel poste. Des
réglages communs au bureau (« Paramètres ») ne viendront qu'avec les comptes du bureau ; des réglages gardés dans un seul
navigateur feraient sortir deux plans différents de la même note.

    defauts()            les options neuves (une copie)
    normaliser(c)        copie contrôlée ; une clé absente prend sa valeur par défaut ; ValueError (message à montrer)
    mise_en_page(f)      la ligne de la table « mise en page par châssis » retenue, et pourquoi
    retour(c, piece, h_mm, enrobage_mm, etrier_mm)   la longueur hors tout du retour d'une barre longitudinale (mm)
"""
import copy

# ------------------------------------------------------------------ la mise en page des deux feuilles, par châssis
# Décision du propriétaire du 12/11/2026, transmise par la conversation RBAR : plus de recherche d'échelle, la taille du
# châssis fixe le format et les échelles des deux feuilles. Valeurs de départ : celles du PDF d'essai du 12/11 (A0).
# Échelles en 1/N : plan (plan de coffrage et ses coupes), details (détails de coffrage), familles (élévations et vues
# en plan des armatures), coupes (coupes des familles), croisements (détails des croisements), implantation (feuille
# d'armatures). Pas de 1/30.
ECHELLES = ("plan", "details", "familles", "coupes", "croisements", "implantation")
MISES_EN_PAGE = (
    # châssis (cm), format, plan, details, familles, coupes, croisements, implantation
    (450.0, "A0", 20, 10, 20, 20, 5, 50),
    (460.0, "A0", 20, 10, 20, 20, 5, 50),
    (600.0, "A0", 20, 10, 20, 10, 10, 50),
    (800.0, "A0", 25, 10, 25, 10, 10, 50),
    (1000.0, "A0", 50, 20, 20, 10, 10, 100),
)
AUTO, RBAR_CHOISIT = "auto", "rbar"


def _cle_ligne(chassis_cm: float) -> str:
    return f"{chassis_cm:g}"


CHOIX_MISE_EN_PAGE = ((AUTO, "Selon le châssis"),) + tuple(
    (_cle_ligne(L[0]), f"Châssis {L[0] / 100:.2f} m".replace(".", ",") + f" · {L[1]} · plan 1/{L[2]}") for L in MISES_EN_PAGE) + (
    (RBAR_CHOISIT, "RBAR choisit"),)

# ------------------------------------------------------------------ les options
RETOUR_FIXE, RETOUR_ARRET = "fixe", "arret"
CHOIX_RETOUR = ((RETOUR_FIXE, "Longueur fixe"), (RETOUR_ARRET, "Arrêté sous la face opposée"))
COUPES_SUPPORT = (("zone_milieu", "Zone serrée et milieu"), ("milieu", "Au milieu"), ("aucune", "Aucune"))
COUPES_SEMELLE = (("pieu", "Au droit d'un pieu"), ("milieu", "Au milieu"), ("aucune", "Aucune"))
TEXTE_MAX = 80
POSE_MAX = 600                 # les remarques de pose : quelques lignes courtes (une ligne du fichier = une ligne du plan)
POSE_LIGNES_MAX = 8

_DEFAUTS = {
    "mise_en_page": AUTO,
    "support": {"retour": RETOUR_FIXE, "retour_mm": 500.0, "arret_mm": 100.0, "retour_lat_mm": 500.0,
                "zone_serree_mm": 2000.0, "laterales": 2, "recouvrement_lat": True, "coupes": "zone_milieu"},
    "semelle": {"retour": RETOUR_FIXE, "retour_mm": 600.0, "arret_mm": 100.0, "retour_lat_mm": 800.0,
                "laterales": 2, "recouvrement_lat": True, "coupes": "pieu",
                "cage_long_mm": 700.0, "cage_larg_mm": 700.0, "cage_cadres_long": 5, "cage_cadres_travers": 5,
                "cage_jambe_mm": 650.0},
    "perfos": {"longueur_mm": 600.0, "scelle_mm": 400.0, "remarque": "scellement chimique, en seconde phase"},
    # les niveaux de Greenwood (dessus des poutres −0,50, recépage −1,20, relatifs au ±0,00) : des données du CHANTIER
    # plus que des règles du bureau — le contrôle d'export demande de les vérifier tant qu'ils valent ceux de Greenwood
    "niveaux": {"dessus_m": -0.5, "recepage_m": -1.2},
    "textes": {"plan": "Implantation", "support": "Poutres transversales", "semelle": "Poutres longitudinales",
               "chassis": "Châssis de la grue",
               "pose": ("1. Poser les barres inférieures dans les poutres longitudinales\n"
                        "2. Poser la cage d'armatures des poutres transversales\n"
                        "3. Poser les barres supérieures des poutres longitudinales")},
}

# Les règles FIXES du dessin (pas des options : rien ne les fait varier aujourd'hui), elles aussi tirées de Greenwood.
# - les barres longitudinales (inférieures, supérieures, latérales) ont un retour à chaque about ;
# - les étriers de la poutre support sont des CADRES qui se chevauchent (2 pour 4 brins, 3 pour 6) ; ceux de la semelle,
#   DEUX CADRES pour 4 brins (des cadres qui se chevauchent au-delà) ; 2 brins : un étrier pleine largeur ;
# - jamais de barres de montage ;
# - les poutres sont vues d'en haut (trait plein : « retombée ») ;
# - les perfos d'une plaque : trois barres scellées dans la poutre support, en triangle, du côté de l'about le plus
#   proche du pied (abscisses comptées DEPUIS LE PIED vers l'about ; écart depuis l'axe, + à gauche) : deux à 420 mm du
#   pied, à ± 150 mm de l'axe, une à 110 mm du pied, sur l'axe — le motif du plan de référence (3 perfos par plaque).
MOTIF_PERFOS = ((420.0, 150.0), (420.0, -150.0), (110.0, 0.0))
PERFOS_PAR_MOTIF = 3


def defauts() -> dict:
    """Les options neuves : les valeurs de Greenwood (une copie)."""
    return copy.deepcopy(_DEFAUTS)


# ------------------------------------------------------------------ normalisation
def _exiger(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _nombre(nom: str, x, mini: float, maxi: float) -> float:
    _exiger(isinstance(x, (int, float)) and not isinstance(x, bool), f"{nom} : nombre attendu")
    x = float(x)
    _exiger(x == x and mini <= x <= maxi, f"{nom} : {x:g} hors du domaine ({mini:g} à {maxi:g})")
    return x


def _entier(nom: str, x, mini: int, maxi: int) -> int:
    x = _nombre(nom, x, mini, maxi)
    _exiger(float(x).is_integer(), f"{nom} : nombre entier attendu")
    return int(x)


def _choix(nom: str, x, choix) -> str:
    cles = [k for k, _ in choix]
    _exiger(x in cles, f"{nom} : « {x} » inconnu ({', '.join(cles)})")
    return x


def _texte(nom: str, x, maxi: int = TEXTE_MAX) -> str:
    _exiger(isinstance(x, str), f"{nom} : texte attendu")
    t = x.strip()
    _exiger(len(t) <= maxi, f"{nom} : {len(t)} caractères (au plus {maxi})")
    return t


def _bool(nom: str, x) -> bool:
    _exiger(isinstance(x, bool), f"{nom} : vrai ou faux attendu")
    return x


def _niveau(nom: str, x):
    if x is None:
        return None
    return _nombre(nom, x, -1000.0, 1000.0)


def _bloc(nom: str, src, defaut: dict) -> dict:
    """Le bloc ``src`` complété par son défaut ; une clé inconnue est refusée (une note abîmée ne passe pas en silence)."""
    if src is None:
        return copy.deepcopy(defaut)
    _exiger(isinstance(src, dict), f"options de conception, {nom} : données illisibles")
    inconnues = sorted(set(src) - set(defaut))
    _exiger(not inconnues, f"options de conception, {nom} : champ inconnu {', '.join(inconnues)}")
    out = copy.deepcopy(defaut)
    out.update(src)
    return out


def _piece(nom: str, src: dict, coupes) -> dict:
    out = {"retour": _choix(f"{nom}, retour", src["retour"], CHOIX_RETOUR),
           "retour_mm": _nombre(f"{nom}, retour des barres", src["retour_mm"], 0, 5000),
           "arret_mm": _nombre(f"{nom}, arrêt sous la face opposée", src["arret_mm"], 0, 1000),
           "retour_lat_mm": _nombre(f"{nom}, retour des barres latérales", src["retour_lat_mm"], 0, 5000),
           "laterales": _entier(f"{nom}, rangées de barres latérales par face", src["laterales"], 0, 10),
           "recouvrement_lat": _bool(f"{nom}, recouvrement des barres latérales au milieu", src["recouvrement_lat"]),
           "coupes": _choix(f"{nom}, coupes", src["coupes"], coupes)}
    return out


def normaliser(c) -> dict:
    """Copie contrôlée des options de conception ; une clé absente prend sa valeur par défaut (une note d'avant les
    options les reçoit toutes) ; une valeur hors du domaine lève ValueError avec un message à montrer tel quel."""
    d = _DEFAUTS
    if c is None:
        return defauts()
    _exiger(isinstance(c, dict), "options de conception : données illisibles")
    inconnues = sorted(set(c) - set(d))
    _exiger(not inconnues, f"options de conception : champ inconnu {', '.join(inconnues)}")
    out = {"mise_en_page": _choix("mise en page des plans", c.get("mise_en_page", d["mise_en_page"]), CHOIX_MISE_EN_PAGE)}
    su = _bloc("poutre support", c.get("support"), d["support"])
    out["support"] = _piece("poutre support", su, COUPES_SUPPORT)
    out["support"]["zone_serree_mm"] = _nombre("poutre support, zone d'étriers serrés", su["zone_serree_mm"], 0, 50000)
    sm = _bloc("semelle", c.get("semelle"), d["semelle"])
    out["semelle"] = _piece("semelle", sm, COUPES_SEMELLE)
    for cle, nom, maxi in (("cage_long_mm", "cage de renfort, longueur", 5000), ("cage_larg_mm", "cage de renfort, largeur", 5000),
                           ("cage_jambe_mm", "cage de renfort, jambe des cadres", 5000)):
        out["semelle"][cle] = _nombre(nom, sm[cle], 0, maxi)
    for cle, nom in (("cage_cadres_long", "cage de renfort, cadres dans le sens de la semelle"),
                     ("cage_cadres_travers", "cage de renfort, cadres en travers")):
        out["semelle"][cle] = _entier(nom, sm[cle], 0, 50)
    pe = _bloc("barres scellées", c.get("perfos"), d["perfos"])
    out["perfos"] = {"longueur_mm": _nombre("barres scellées, longueur", pe["longueur_mm"], 0, 5000),
                     "scelle_mm": _nombre("barres scellées, longueur scellée", pe["scelle_mm"], 0, 5000),
                     "remarque": _texte("barres scellées, remarque", pe["remarque"])}
    _exiger(out["perfos"]["scelle_mm"] <= out["perfos"]["longueur_mm"],
            "barres scellées : la longueur scellée dépasse la longueur de la barre")
    ni = _bloc("niveaux", c.get("niveaux"), d["niveaux"])
    out["niveaux"] = {"dessus_m": _niveau("niveau du dessus des poutres", ni["dessus_m"]),
                      "recepage_m": _niveau("niveau de recépage des pieux", ni["recepage_m"])}
    tx = _bloc("textes du plan", c.get("textes"), d["textes"])
    out["textes"] = {k: _texte(f"texte du plan « {k} »", tx[k]) for k in ("plan", "support", "semelle", "chassis")}
    pose = _texte("remarques de pose", tx["pose"], POSE_MAX)
    lignes = [l.strip() for l in pose.split("\n") if l.strip()]
    _exiger(len(lignes) <= POSE_LIGNES_MAX, f"remarques de pose : {len(lignes)} lignes (au plus {POSE_LIGNES_MAX})")
    _exiger(all(len(l) <= TEXTE_MAX for l in lignes), f"remarques de pose : une ligne dépasse {TEXTE_MAX} caractères")
    out["textes"]["pose"] = "\n".join(lignes)
    return out


# ------------------------------------------------------------------ ce que les options donnent
def mise_en_page(f: dict):
    """(ligne, pourquoi) : la ligne de la table retenue — un tuple de ``MISES_EN_PAGE`` — ou None (RBAR choisit), et
    la phrase qui le dit. « Selon le châssis » : la ligne de ce châssis, sinon celle du châssis immédiatement plus
    grand (le plan d'un châssis plus grand tient sur sa feuille) ; au-delà du plus grand, rien n'est imposé."""
    choix = f["conception"]["mise_en_page"]
    Lc = float(f["chassis"])
    if choix == RBAR_CHOISIT:
        return None, "mise en page laissée à RBAR (option de conception)"
    if choix != AUTO:
        ligne = next(L for L in MISES_EN_PAGE if _cle_ligne(L[0]) == choix)
        return ligne, f"ligne du châssis {_m(ligne[0])} imposée (option de conception)"
    ligne = next((L for L in MISES_EN_PAGE if L[0] >= Lc - 1e-9), None)
    if ligne is None:
        return None, (f"châssis {_m(Lc)} : au-delà de la table de mise en page (jusqu'à {_m(MISES_EN_PAGE[-1][0])}) — "
                      "RBAR choisit le format et les échelles")
    if abs(ligne[0] - Lc) < 1e-9:
        return ligne, f"ligne du châssis {_m(Lc)}"
    return ligne, f"châssis {_m(Lc)} absent de la table : ligne du châssis immédiatement plus grand ({_m(ligne[0])})"


def _m(cm: float) -> str:
    return f"{cm / 100:.2f} m".replace(".", ",")


def feuilles(ligne) -> dict:
    """La clé « feuilles » du fichier pour une ligne de la table."""
    return {"format": ligne[1], "echelles": dict(zip(ECHELLES, ligne[2:]))}


NOMS_PIECES = {"support": "Poutre support", "semelle": "Semelle"}


def _cm(mm: float) -> str:
    return f"{mm / 10:.1f}".replace(".", ",").removesuffix(",0") + " cm"


def retour(c: dict, piece: str, h_mm: float, enrobage_mm: float, etrier_mm: float) -> float:
    """La longueur HORS TOUT du retour d'une barre longitudinale inférieure ou supérieure (mm) ; 0 : barres droites.

    La barre est posée DANS les étriers : sa face extérieure est à « enrobage + Ø étrier » de sa face — l'enrobage se
    mesure à l'extérieur des étriers (l'encadré « Matériaux » du plan le dit, et le calcul de la hauteur utile du moteur
    fait de même : d = h − c − Ø étrier − Ø/2). « Longueur fixe » : la longueur de l'option. « Arrêté sous la face
    opposée » : le retour monte (ou descend) jusqu'à ``arret_mm`` de la face opposée, soit
    h − enrobage − Ø étrier − arrêt. Dans les deux cas, il doit tenir dans la cage des étriers : au plus
    h − 2 × (enrobage + Ø étrier). ValueError (message à montrer) sinon."""
    p = c[piece]
    nom, cage = NOMS_PIECES[piece], h_mm - 2 * (enrobage_mm + etrier_mm)
    if p["retour"] == RETOUR_FIXE:
        r = p["retour_mm"]
        _exiger(r <= cage + 1e-9, f"{nom} : un retour de {_cm(r)} ne tient pas dans la hauteur — au plus {_cm(cage)} "
                                  f"(h {_cm(h_mm)} moins deux fois l'enrobage et l'étrier ; option de conception)")
        return r
    a = p["arret_mm"]
    r = h_mm - enrobage_mm - etrier_mm - a
    _exiger(r > 0, f"{nom} : le retour arrêté à {_cm(a)} de la face opposée n'a pas de longueur (h {_cm(h_mm)} ; "
                   "option de conception)")
    _exiger(r <= cage + 1e-9, f"{nom} : un retour arrêté à {_cm(a)} de la face opposée entame l'enrobage — l'arrêt doit "
                              f"valoir au moins {_cm(enrobage_mm + etrier_mm)} (enrobage et étrier ; option de conception)")
    return r
