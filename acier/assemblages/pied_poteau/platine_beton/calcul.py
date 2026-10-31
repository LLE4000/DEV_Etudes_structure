# -*- coding: utf-8 -*-
"""Le calcul de l'assemblage selon son MODE : calcul manuel de la platine
(``manuel.moteur``) ou vérification complète (``moteur``). C'est ce point
d'entrée que l'écran, le projet et le rapport appellent ; les deux moteurs
restent testables séparément."""
from .entrees import est_complet
from . import moteur as _complet
from .manuel import moteur as _manuel


def compute(u):
    """``R`` du mode de ``u`` ; ``R.mode`` vaut « manuel » ou « complet »."""
    if est_complet(u or {}):
        R = _complet.compute(u)
        R.mode = "complet"
        return R
    return _manuel.compute(u)


def est_manuel_R(R):
    return getattr(R, "mode", "complet") == "manuel"
