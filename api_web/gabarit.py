# -*- coding: utf-8 -*-
"""Le GABARIT DU BUREAU (lot 6, 14/11/2026 ; docs/site/MODELES_DIMENSIONNEMENT.md) : les valeurs de départ des OPTIONS
d'un module, fixées par un administrateur sur la page « Paramètres » — « on prend la longueur de recouvrement et on
l'arrondit tout le temps au 5 cm près ou à 10 cm près » ; chaque note peut ensuite changer une valeur au cas par cas
(« pour cette dalle-là, pas de jeu »).

Ce que ce fichier fait, et seulement cela :
- ``appliquer(mod, etat, gabarit)`` : une note NEUVE reçoit les valeurs du gabarit, chacune écrite par le module LUI-MÊME
  (sa fonction ``ecrire_gabarit``, sinon ``modifier`` — le contrôle de la rubrique Options : une valeur que le moteur
  n'admet plus est refusée, la valeur d'usine reste, le refus est dit) ;
- ``page(mod, q)`` : ce que la page « Paramètres » montre d'un module — sa rubrique Options rendue par le module pour le
  gabarit (le même composant que dans la note), les valeurs, celles d'usine, et une saisie éventuelle.

Ce qu'il ne fait PAS : il ne calcule rien et ne recopie aucune valeur. Le moteur reste la source unique du calcul (R3) :
un gabarit choisit des DONNÉES (un jeu, un pas d'arrondi), que le moteur vérifie comme toutes les autres. Les valeurs
d'usine sont celles de ``nouvel_etat()`` du module, LUES ici, jamais écrites. Une note existante ne bouge jamais quand le
gabarit change : il ne s'applique qu'à la création (l'action « nouveau » du routeur).

Un module qui a un gabarit déclare :
    GABARIT : tuple          les clés des options, dans l'ordre où elles s'écrivent (« taux/actif » : un chemin) ;
    lire_gabarit(etat) -> dict          les valeurs de ces options dans la note ;
    options_gabarit(etat) -> str        la rubrique Options telle que la page la montre ;
    ecrire_gabarit(etat, cle, valeur)   (facultatif) une valeur écrite autrement que par ``modifier`` ; le message du refus.

Les refus parlent français et nomment le champ ; jamais une exception vers l'écran.
"""
from typing import Dict, List, Optional


def a_un_gabarit(mod) -> bool:
    return all(hasattr(mod, x) for x in ("GABARIT", "lire_gabarit", "options_gabarit"))


def _ecrire(mod, etat: Dict, cle: str, valeur) -> Optional[str]:
    f = getattr(mod, "ecrire_gabarit", None)
    if callable(f):
        return f(etat, cle, valeur)
    return mod.modifier(etat, cle.split("/"), valeur)


def appliquer(mod, etat: Dict, gabarit) -> List[str]:
    """Écrit dans la note neuve ``etat`` chaque valeur du ``gabarit`` que le module connaît, dans l'ordre du module ; rend
    les refus (une clé inconnue — d'un module d'une autre version — ou une valeur que le moteur refuse : la valeur
    d'usine reste). Un gabarit qui n'est pas un dictionnaire est ignoré, et le refus le dit."""
    if not gabarit:
        return []
    if not isinstance(gabarit, dict):
        return ["Gabarit du bureau illisible : les valeurs d'usine sont gardées"]
    refus = []
    for cle in mod.GABARIT:
        if cle in gabarit:
            try:
                msg = _ecrire(mod, etat, cle, gabarit[cle])
            except (TypeError, ValueError, KeyError) as e:      # (une valeur de forme inattendue : refusée, dite)
                msg = str(e) or type(e).__name__
            if msg:
                refus.append(f"{msg} — gabarit du bureau : la valeur d'usine est gardée")
    inconnues = sorted(str(k) for k in gabarit if k not in mod.GABARIT)
    if inconnues:
        refus.append(f"Gabarit du bureau : option(s) inconnue(s) de ce module ({', '.join(inconnues)}), ignorée(s)")
    return refus


def ecarts(valeurs: Dict, usine: Dict) -> Dict:
    """Les valeurs qui diffèrent de celles d'usine : ce que le gabarit GARDE (une valeur égale à l'usine suit l'usine,
    même quand elle change)."""
    return {k: v for k, v in valeurs.items() if k in usine and v != usine[k]}


def page(mod, q: Dict) -> Dict:
    """L'action « gabarit » (la page « Paramètres ») : {gabarit, chemin?, valeur?} → la rubrique Options du module rendue
    pour ce gabarit, ses valeurs, celles d'usine, les écarts (ce que la page enregistre), le message d'une saisie
    refusée et les refus du gabarit reçu. Une saisie hors des options du gabarit est refusée."""
    if not a_un_gabarit(mod):
        return {"ok": False, "erreur": "Ce module n'a pas de gabarit du bureau."}
    etat = mod.nouvel_etat()
    usine = mod.lire_gabarit(etat)
    refus = appliquer(mod, etat, q.get("gabarit"))
    message = None
    chemin = q.get("chemin")
    if chemin is not None:
        ch = [str(c) for c in chemin] if isinstance(chemin, (list, tuple)) else []
        if "/".join(ch) not in mod.GABARIT:
            message = "Cette valeur n'est pas une option du gabarit du bureau"
        else:
            # (la saisie de la page : celle de la rubrique Options, par le module — une puce d'espacement bascule, etc.)
            message = mod.modifier(etat, ch, q.get("valeur"))
    valeurs = mod.lire_gabarit(etat)
    return {"ok": True, "html": mod.options_gabarit(etat), "valeurs": valeurs, "usine": usine,
            "ecarts": ecarts(valeurs, usine), "message": message, "refus": refus}
