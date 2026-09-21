# -*- coding: utf-8 -*-
"""Structures de résultat communes à la famille Acier.

Les noms de champs sont ceux du moteur de référence (clés du JSON de
parité) ; les docstrings donnent leur sens en français.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Verification:
    """Une vérification : sollicitation ``Ed``, résistance ``Rd``, taux ``eta``.

    - ``key``      : clé courte (ex. ``bv_S``) ;
    - ``grp``      : groupe d'affichage (Boulons, Cornières, …) ;
    - ``label``    : libellé complet ;
    - ``unit``     : unité (kN, kNm, mm, N/mm ; « - » pour un taux) ;
    - ``ref``      : référence normative ou bibliographique ;
    - ``nat``      : nature — ``EC`` (Eurocode direct), ``EC + COMP``,
                     ``COMP`` (modèle complémentaire), ``INT`` (interprétation) ;
    - ``active``   : False = « sans objet » dans cette configuration ;
    - ``formula``  : formule générale ; ``vals`` : valeurs introduites et
                     résultats intermédiaires ;
    - ``ess``      : vérification « essentielle » (export texte court, note) ;
    - ``eta``      : Ed/Rd (1e9 si Rd = 0 et Ed > 0) ; None si inactive ;
    - ``ok``       : eta ≤ 1 + 1e-9 ; None si inactive."""
    key: str
    grp: str
    label: str
    Ed: float
    Rd: float
    unit: str
    ref: str
    nat: str
    active: bool
    formula: str
    vals: str = ""
    ess: bool = False
    eta: Optional[float] = None
    ok: Optional[bool] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Alerte:
    """Une alerte de configuration.

    - ``id``     : identifiant stable (ex. ``h_dispo``, ``dist_S_p1``) ;
    - ``msg``    : message ; ``block`` : bloquante (True) ou informative ;
    - ``fields`` : clés d'entrée en cause ; ``dims`` : cotes du dessin en
                   cause ; ``elems`` : éléments du dessin en cause ;
    - ``why``    : explication chiffrée de la relation contrôlée ;
    - ``covers`` : lignes du Tableau 3.3 que cette alerte couvre déjà."""
    id: str
    msg: str
    block: bool
    fields: list = field(default_factory=list)
    dims: list = field(default_factory=list)
    elems: list = field(default_factory=list)
    why: str = ""
    covers: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Pince:
    """Une ligne du Tableau 3.3 (EN 1993-1-8) : pince ou entraxe.

    ``val`` valeur, ``min`` = ``kmin``·d0, ``max`` (None = pas de maximum),
    ``ok`` conformité, ``rule`` règle en clair, ``alert`` identifiant de
    l'alerte qui la localise sur le dessin."""
    id: str
    lab: str
    val: float
    min: float
    max: Optional[float]
    kmin: float
    fields: list
    dims: list
    rule: str
    ok: bool
    alert: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class EffortBoulon:
    """Effort sur un boulon (répartition élastique, inertie polaire) :
    rangée ``i``, file ``j``, position ``x``, ``y`` (mm), composantes ``fx``,
    ``fz`` et résultante ``f`` (kN)."""
    i: int
    j: int
    x: float
    y: float
    fx: float
    fz: float
    f: float

    def to_dict(self):
        return asdict(self)
