# -*- coding: utf-8 -*-
"""(provisoire) La rubrique « Plaques » : en cours d'écriture."""
from api_web import biblio_autres as A


def rendre(params, hote="page"):
    return {"titre": "Plaques rectangulaires", "barre": "", "tableau": A.refus("En cours d'écriture", "…"), "bande": "", "pourquoi": "", "sources": "",
            "params": {}, "message": None}
