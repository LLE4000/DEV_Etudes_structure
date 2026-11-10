# -*- coding: utf-8 -*-
"""Les vues de la cornière d'appui POSÉES dans la toile de l'établi : une couche mince sur ``acier.corniere.schemas``
(les quatre scènes partagées avec la note — coupe, moments, contraintes, élévation), qui ne fait que les AJUSTER à la
toile et les rendre au format des modules du site (``docs/site/MODULES_SITE.md`` § 3).

Chaque vue est reconstruite (``schemas.ajuster``) jusqu'à ce que son texte se lise à ``TEXTE_PX`` pixels dans la toile
``TOILE`` du format PC (mesurée par site/web/verifier_site.js) : une petite cornière (L 40×40×4) ne disparaît pas sous
ses cotes, une grande n'a pas un texte minuscule. Une vue qu'on ne sait pas dessiner (proportions absurdes : une
tranche de plusieurs mètres face à une aile de quelques centimètres) est remplacée par le cadre « Dessin impossible à
cette échelle » : jamais d'erreur, jamais de texte illisible — le calcul, lui, est rendu normalement.

Rien n'est calculé ici : les grandeurs viennent du moteur (``acier.corniere.moteur.verifier``), les dessins de
``schemas``. La légende de chaque vue (ce qu'elle montre, et le mode) vient aussi de ``schemas.legende`` : une seule
source pour l'écran et la note.
"""
import re
from typing import Dict, List, Tuple

from acier.corniere import schemas as SC

TOILE = (626.0, 682.0)          # la toile de l'établi au format PC (px), mesurée par le contrôle navigateur
TEXTE_PX = 13.0                 # la taille du texte des dessins à l'écran (px)
TEXTE_MIN_PX = 6.0              # sous cette taille, le dessin n'est plus lisible : on le dit au lieu de le montrer

VUES = SC.VUES                  # (clé, titre, titre court, à l'échelle)


def _ajuster(construire, fs0: float):
    """La vue reconstruite jusqu'à ce que son texte se lise à TEXTE_PX dans la TOILE."""
    return SC.ajuster(construire, TOILE, TEXTE_PX, fs0)


def _fs0(res: Dict, cle: str) -> float:
    """La taille de texte de départ (mm de la scène), proportionnelle à la pièce : les quatre vues dessinent la
    cornière à l'échelle (l'élévation compte la tranche, coupée au-delà de 2,5 fois la hauteur)."""
    g = res["geometrie"]
    if cle == "elevation":
        return (min(g["L"], SC.COUPURE_TRANCHE * g["haut"]) + 1.5 * (g["larg"] + g["haut"])) / 44.0
    return (g["larg"] + g["haut"]) / 34.0


def vues(res: Dict, ident: str = "co") -> List[Dict]:
    """Les vues d'un résultat du moteur : [{cle, titre, court, svg, legende, echelle}] (le SVG autonome, style embarqué ;
    ``echelle`` False pour les diagrammes, hors échelle). Une vue qu'on ne sait pas dessiner est remplacée par le cadre
    qui le dit."""
    out = []
    for cle, titre, court, echelle in VUES:
        fn = SC.FONCTIONS[cle]
        try:
            d = _ajuster(lambda fs: fn(res, fs), _fs0(res, cle))
            svg = SC.svg(d, f"{ident}_{cle}")
            if not TEXTE_MIN_PX <= taille_texte(svg) < 1e6:
                raise ValueError("texte illisible")
        except (ArithmeticError, ValueError, ZeroDivisionError):
            svg = SC.secours(f"{ident}_{cle}", titre)
        out.append({"cle": cle, "titre": titre, "court": court, "svg": svg, "legende": SC.legende(res, cle), "echelle": echelle})
    return out


def taille_texte(svg: str, fs: float = None) -> float:
    """La taille à l'écran (px) d'un texte de ``fs`` unités (par défaut : la plus fréquente du dessin) dans la TOILE —
    pour les tests de lisibilité."""
    vb = svg.split('viewBox="', 1)[1].split('"', 1)[0].split()
    w, h = float(vb[2]), float(vb[3])
    if fs is None:
        tailles = re.findall(r'font-size="([0-9.]+)"', svg) or re.findall(r"font-size:([0-9.]+)", svg)
        fs = float(max(set(tailles), key=tailles.count)) if tailles else 0.0
    return fs * min(TOILE[0] / w, TOILE[1] / h)


def aire_dessin(svg: str) -> Tuple[float, float]:
    """La boîte de vue d'un SVG (largeur, hauteur) — pour les tests de proportion."""
    vb = svg.split('viewBox="', 1)[1].split('"', 1)[0].split()
    return float(vb[2]), float(vb[3])
