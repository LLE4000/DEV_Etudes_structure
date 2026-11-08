# -*- coding: utf-8 -*-
"""Pillow FACTICE (voir __init__.py) : toute utilisation réelle d'une image lève une erreur claire."""


class Image:  # le type que reportlab nomme
    pass


def open(*a, **k):  # noqa: A001 — le nom de Pillow
    raise RuntimeError("images non prises en charge dans la note de calcul du site")
