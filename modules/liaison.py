# -*- coding: utf-8 -*-
"""Les widgets à clé d'un écran sont des VUES de l'état (une seule source) :
quand l'état change par ailleurs — changement de cas, de configuration,
fichier ouvert, proposition appliquée — chaque widget doit relire sa valeur.

Supprimer la clé de ``st.session_state`` ne suffit PAS : un widget à clé garde
son identité d'un rendu à l'autre (``key_as_main_identity`` de Streamlit) et
le navigateur garde sa dernière valeur tant qu'on ne la lui RÉASSIGNE pas.
Défaut trouvé par les captures du 04/11/2026 : après C3 → C6 la gorge affichait
5 alors que le calcul prenait 4 ; sur « Charges sur plancher », après « ＋ Cas »,
la portée affichait encore celle du cas précédent (le harnais AppTest ne le voit
pas : il n'a pas de navigateur). La règle : on ÉCRIT la valeur de l'état dans
``st.session_state[clé]`` pour chaque widget enregistré (``registre`` : clé →
chemin dans le cas, conversion, options admises), et on supprime les clés sans
chemin dans le nouvel état (leur widget ne sera pas rendu).

Depuis le corps du script, après que des widgets ont été instanciés, Streamlit
refuse d'écrire ces clés : ``demander`` pose un drapeau et ``au_rendu`` fait le
travail au début du rendu suivant ; depuis un callback (rien n'est encore
instancié), ``synchroniser`` s'appelle directement.
"""
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import streamlit as st

Registre = Dict[str, Tuple[Sequence, Optional[Callable], Optional[Sequence]]]


def enregistrer(registre: Registre, cle: str, chemin: Sequence, conv: Optional[Callable] = None,
                options: Optional[Sequence] = None) -> str:
    """Déclare le widget ``cle`` comme vue du ``chemin`` du cas actif."""
    registre[cle] = (tuple(chemin), conv, list(options) if options is not None else None)
    return cle


def _lire(cas: Dict, chemin: Sequence):
    d = cas
    for c in chemin[:-1]:
        d = d[c]
    return d[chemin[-1]]


def synchroniser(prefixe: str, registre: Registre, cas: Dict) -> int:
    """Chaque clé ``prefixe + "w_"…`` présente prend la valeur de l'état ; les
    clés sans chemin (widget d'une autre configuration, tableau versionné) sont
    supprimées. Rend le nombre de clés réécrites ; incrémente la génération
    ``prefixe + "gen"`` (pour les widgets qu'on recrée plutôt que réassigner)."""
    n = 0
    for k in [k for k in list(st.session_state) if isinstance(k, str) and k.startswith(prefixe + "w_")]:
        reg = registre.get(k)
        if reg is None:
            del st.session_state[k]
            continue
        chemin, conv, options = reg
        try:
            v = _lire(cas, chemin)
        except (KeyError, TypeError, IndexError):
            del st.session_state[k]
            continue
        if conv is not None:
            v = conv(v)
        if options is not None and v not in options:
            del st.session_state[k]
            continue
        st.session_state[k] = v
        n += 1
    st.session_state[prefixe + "gen"] = int(st.session_state.get(prefixe + "gen", 0)) + 1
    return n


def generation(prefixe: str) -> int:
    return int(st.session_state.get(prefixe + "gen", 0))


def demander(prefixe: str) -> None:
    """Depuis le corps du script : la synchronisation se fera au rendu suivant."""
    st.session_state[prefixe + "sync"] = True


def au_rendu(prefixe: str, registre: Registre, cas: Dict) -> bool:
    """À appeler en tête du rendu, avant tout widget : honore une demande en attente."""
    if st.session_state.pop(prefixe + "sync", False):
        synchroniser(prefixe, registre, cas)
        return True
    return False
