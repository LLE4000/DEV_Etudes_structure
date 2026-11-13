# -*- coding: utf-8 -*-
"""La POSITION des armatures — UNE règle pour la poutre, la dalle pleine et la prédalle (décision du propriétaire du
13/11/2026 ; docs/site/SUITE_DU_CHANTIER.md, UNI0, et docs/site/CHARTE_ERGONOMIE.md E15). Bibliothèque standard seule.

Chaque barre a DEUX positions, mesurées depuis le parement de sa face (cm) :

  · la position RÉELLE — là où la barre est posée, ce que les DESSINS montrent : au contact de l'enrobage,
    c + Ø/2 (une poutre : c + Ø étrier + Ø/2, la barre dans l'angle du cadre) ; une couche posée sur une autre :
    le dessus de celle-ci + le jeu entre couches + Ø/2. Les Ø réels, aucun arrondi ;

  · l'axe de CALCUL — la position réelle plus le JEU DE POSE (la barre peut finir plus loin du parement que dessiné,
    jamais plus près), arrondi UNE fois, à la fin, vers l'intérieur (plus loin du parement : d diminue, A_s,req
    augmente), au pas choisi (« façon A », tout depuis le béton) :

        e = ⌈réel + jeu⌉_pas

    Exemples (c = 3 cm, jeu 1 cm, pas 0,5 cm) : un Ø12 sur l'enrobage, 3 + 0,6 + 1 (jeu) = 4,6 → 5,0 cm ; un Ø10 posé
    sur ce Ø12, 3 + 1,2 + 0,5 + 1 (jeu) = 5,7 → 6,0 cm (et non 5,0 + 0,6 + 0,5 = 6,1 → 6,5 : l'arrondi compté deux
    fois) ; une poutre, étrier Ø8 et Ø16 : 3 + 0,8 + 0,8 + 1 (jeu) = 5,6 → 6,0 cm.

Les valeurs par défaut (décision du 13/11/2026) : jeu de pose 1 cm, arrondi 0,5 cm, 0 entre deux couches. Un axe
SAISI à l'écran est une position réelle : le calcul lui ajoute le jeu et l'arrondit de même.

API
---
    arrondi_haut(x, pas) -> float          x arrondi au pas SUPÉRIEUR (pas 0 : x) — aux erreurs des flottants près
    axe_calcul(reel, jeu, pas) -> float    ⌈reel + jeu⌉_pas
    sur_parement(c, d_mm, d_etrier_mm=0)   la position réelle d'une barre au contact de l'enrobage (et du cadre)
    sur_couche(dessus, d_mm, jeu_couches)  la position réelle d'une barre posée sur une autre couche
    dessus(reel, d_mm)                     le dessus d'une couche (son bord côté intérieur), position réelle + Ø/2
    detail(termes, jeu, pas) -> dict       le calcul d'un axe, terme à terme, pour l'écran et la note
    lire_arrondi(x) -> float               un pas d'arrondi relu (ValueError hors de ARRONDIS)
    jeu_note_d_avant(jeu)                  le jeu de pose d'une note d'AVANT la règle, relu : JEU_POSE au moins
"""
import math

JEU_POSE = 1.0                # cm — le jeu de pose par défaut
ARRONDI = 0.5                 # cm — le pas d'arrondi par défaut
ARRONDIS = (0.0, 0.5, 1.0)    # cm — les pas offerts (0 : aucun arrondi)
JEU_COUCHES = 0.0             # cm — le jeu entre deux couches qui se croisent (dalle) : au contact
_DECIMALES = 6                # x / pas est arrondi à 1e-6 avant ⌈ ⌉ : 6,0 / 0,5 calculé 12,000000000000002 rend 6,0


def arrondi_haut(x: float, pas: float) -> float:
    """``x`` arrondi au multiple SUPÉRIEUR de ``pas`` (cm) ; ``pas`` nul : ``x`` inchangé. Le quotient est arrondi à
    1e-6 avant la partie entière supérieure : une somme de flottants qui tombe juste sur le pas (3 + 1 + 1,2 + 0,8)
    ne passe pas au pas suivant."""
    x, pas = float(x), float(pas or 0.0)
    if pas <= 0:
        return x
    return math.ceil(round(x / pas, _DECIMALES)) * pas


def axe_calcul(reel: float, jeu: float = JEU_POSE, pas: float = ARRONDI) -> float:
    """L'axe de CALCUL (cm) d'une barre à la position réelle ``reel`` : ⌈reel + jeu⌉ au pas ``pas``."""
    return arrondi_haut(float(reel) + float(jeu or 0.0), pas)


def sur_parement(c: float, d_mm: float, d_etrier_mm: float = 0.0) -> float:
    """La position RÉELLE (cm) d'une barre de Ø ``d_mm`` au contact de l'enrobage ``c`` (cm) — d'une poutre, au contact
    du cadre de Ø ``d_etrier_mm`` : c + Ø_étrier + Ø/2."""
    return float(c) + float(d_etrier_mm or 0.0) / 10.0 + float(d_mm) / 20.0


def dessus(reel: float, d_mm: float) -> float:
    """Le DESSUS d'une couche (cm, depuis le parement) : son bord côté intérieur, position réelle + Ø/2."""
    return float(reel) + float(d_mm) / 20.0


def sur_couche(dessus_cm: float, d_mm: float, jeu_couches: float = JEU_COUCHES) -> float:
    """La position RÉELLE (cm) d'une barre de Ø ``d_mm`` posée sur une couche dont le dessus est à ``dessus_cm`` :
    dessus + jeu entre couches + Ø/2."""
    return float(dessus_cm) + float(jeu_couches or 0.0) + float(d_mm) / 20.0


def detail(termes, jeu: float = JEU_POSE, pas: float = ARRONDI) -> dict:
    """Le calcul d'un axe terme à terme — ce que l'écran écrit « 3 + 0,6 + 1 (jeu) = 4,6 → 5,0 cm ». ``termes`` : la
    position réelle en morceaux, ``[(symbole, valeur cm), …]`` (« c », « Ø/2 »…). Rend ``{"termes": [{"symbole",
    "valeur"}…], "reel": Σ, "jeu", "brut": reel + jeu, "pas", "axe": ⌈brut⌉_pas}``."""
    tt = [{"symbole": s, "valeur": float(v)} for s, v in termes]
    reel = sum(t["valeur"] for t in tt)
    jeu = float(jeu or 0.0)
    return {"termes": tt, "reel": reel, "jeu": jeu, "brut": reel + jeu, "pas": float(pas or 0.0),
            "axe": axe_calcul(reel, jeu, pas)}


def _txt_cm(a: float) -> str:
    return f"{a:g}".replace(".", ",")


def lire_arrondi(x) -> float:
    """Un pas d'arrondi relu (cm) : l'un de ARRONDIS ; ValueError sinon (jamais remplacé en silence)."""
    permis = ", ".join(_txt_cm(a) for a in ARRONDIS[:-1]) + f" ou {_txt_cm(ARRONDIS[-1])} cm"
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"arrondi : {permis} attendu (reçu {x!r})")
    for a in ARRONDIS:
        if abs(float(x) - a) < 1e-9:
            return a
    raise ValueError(f"arrondi de {_txt_cm(float(x))} cm : {permis} attendu")


def jeu_note_d_avant(jeu):
    """Le jeu de pose d'une note d'AVANT la règle du 13/11/2026 (le « jeu sous le premier lit », 0 par défaut), relu :
    porté au jeu du bureau (JEU_POSE) s'il est plus petit ou absent ; un jeu plus grand est gardé (un choix). Une valeur
    qui n'est pas un nombre est rendue telle quelle : la relecture de la note la refuse avec son message. La poutre, la
    dalle pleine, la prédalle et un fichier de l'écran Streamlit relisent leur jeu par elle."""
    if jeu is None:
        return JEU_POSE
    if isinstance(jeu, bool) or not isinstance(jeu, (int, float)):
        return jeu
    return max(float(jeu), JEU_POSE)
