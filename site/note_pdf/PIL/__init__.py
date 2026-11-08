# -*- coding: utf-8 -*-
"""Pillow FACTICE du paquet de la note PDF du site (site/web/construire.py, paquet_note) : reportlab l'importe au
chargement (reportlab.lib.utils), mais la note de calcul n'a aucune image matricielle. Dans le navigateur, ce paquet
remplace Pillow ; ailleurs (CPython, Streamlit), le vrai Pillow reste celui qu'on charge."""
__version__ = "0"
