# -*- coding: utf-8 -*-
"""Textes du plat d'âme pour l'écran et la note : références courtes et
lignes d'alerte. Le moteur écrit directement en notation finale (hp, tp,
zp, e2,b…) — pas de couche de correspondance ; ``ref_courte`` est celle de
la famille (module doubles cornières)."""
import re

from acier.formats import F
from ..doubles_cornieres.notation import ref_courte  # noqa: F401 — réutilisée telle quelle

LEGENDE_REFERENCES = ("Références : EC3 = EN 1993-1-8 sauf indication (§, Tab.) ; EC3-1-1 = EN 1993-1-1 ; "
                      "MSB P5 ; P358 = SCI P358 ; ECCS = ECCS n°126 ; BS 5950-1 : courbe de déversement "
                      "des plats longs (Annexe B.2.1 / Table 17).")

TITRES_ALERTES = {
    "e1bot": "Impossible — pince basse du groupe de boulons",
    "e2p": "Impossible — plat trop court (pince e2 au bord libre)",
    "zc_top": "Impossible — dessus du plat dans le grugeage ou le congé",
    "h_dispo": "Impossible — hauteur disponible insuffisante",
    "pl_web": "Impossible — plat hors de la partie droite de l'âme porteuse",
    "dnt_min": "Grugeage supérieur insuffisant",
    "ln_min": "Longueur de grugeage insuffisante",
    "gap_neg": "Hors domaine — la poutre secondaire descend sous la principale",
    "dnb_min": "Grugeage inférieur nécessaire ou insuffisant",
    "one_bolt": "Impossible — un seul boulon",
    "d0_manq": "Trou surdimensionné : saisir d0",
    "tp_duct": "tp > 0,5·d : ductilité non assurée (recommandation)",
    "hp_rec": "hp < 0,6·h : maintien en torsion non assuré (recommandation)",
    "long_nr": "Plat long et poutre non maintenue au déversement",
    "a_rec": "Gorge sous la pleine résistance recommandée",
    "excen": "Attache d'un seul côté : flexion hors plan de l'âme porteuse",
    "MEd": "MEd ≠ 0 : traité comme moment parasite",
    "NEd": "NEd de traction : flexion hors plan de l'âme porteuse non vérifiée",
    "cat_cl": "Catégorie B ou C : boulons 8.8 ou 10.9 requis (§3.4.1)",
}


def ec(txt):
    """Identité (les textes du moteur sont déjà en notation finale)."""
    return txt or ""


def titre_alerte(a):
    if a.id.startswith("dist_"):
        return "Tab. 3.3 — " + a.msg.split(" : ", 1)[1].rstrip(".") if " : " in a.msg else a.msg
    return TITRES_ALERTES.get(a.id, a.msg)


def _premiere_phrase(txt):
    m = re.match(r"(.+?\.)(?:\s|$)", txt)
    return (m.group(1) if m else txt).rstrip(".")


def ligne_alerte(a, R):
    """``(titre, détail)`` — le détail est la première phrase chiffrée, ou la
    ligne du Tableau 3.3 (valeur, limite)."""
    if a.id.startswith("dist_"):
        for x in R.dist:
            if x.alert == a.id:
                if x.val < x.min - 0.001:
                    det = F(x.val, 1) + " mm < " + F(x.kmin, 1) + "·d0 = " + F(x.min, 1) + " mm"
                else:
                    det = F(x.val, 1) + " mm > " + F(x.max, 0) + " mm"
                return titre_alerte(a), det
    return titre_alerte(a), _premiere_phrase(a.why) if a.why else ""
