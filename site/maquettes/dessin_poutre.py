# -*- coding: utf-8 -*-
"""La coupe d'une poutre béton armé : le dessin vit dans api_web/dessin_poutre.py (le site l'emploie) ; les maquettes
le reprennent d'ici, sans copie."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from api_web.dessin_poutre import *  # noqa: E402,F401,F403
from api_web.dessin_poutre import section, svg  # noqa: E402,F401
