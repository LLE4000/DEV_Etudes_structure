# -*- coding: utf-8 -*-
"""Textes de la platine d'about sur poteau pour l'écran et la note :
références courtes et lignes d'alerte. Le moteur écrit directement en
notation finale (hp, tp, bp, zp, p3, e1, twP, tfP…) — pas de couche de
correspondance ; ``ref_courte`` est celle de la famille (module doubles
cornières)."""
import re

from acier.formats import F
from ...poutre_poutre.doubles_cornieres.notation import ref_courte as _ref_courte_famille

# Commentaires internes que le moteur porte dans ses références (« — même
# modèle que … ») : retirés de la VUE (écran, note, export), le moteur ne
# change pas.
_COMMENTAIRES_INTERNES = re.compile(r"\s+—\s+même modèle que[^;]*")


def ref_courte(ref):
    """La référence courte de la famille, sans les commentaires internes du moteur."""
    return _ref_courte_famille(_COMMENTAIRES_INTERNES.sub("", ref or ""))


# La ligne « Notations » du pied de la note ET le paragraphe « Notations » de
# l'onglet Méthode : une source (charte du 01/11/2026, point 4 — les cotes ne
# portent plus que leur symbole). Elle tient sur la ligne de la note
# (≤ 798 pt à 5,6 pt, contrôlé par la suite d'écran).
NOTATIONS = ("Notations : EN 1993-1-8 Tab. 3.3 (e1, p1, e2) ; p3 : entraxe des deux files ; hp, bp, tp : "
             "platine ; zp : position de la platine ; c, dc : grugeage ; rc : recoupe des semelles ; "
             "j : jeu latéral ; ℓ : largeur droite entre congés ; e : pince au bout d'aile ; "
             "h, b, tw, tf : poteau ; a : gorge (cordon double) ; e_col : axe du poteau → attache.")

LEGENDE_REFERENCES = ("Références : EC3 = EN 1993-1-8 sauf indication (§, Tab.) ; EC3-1-1 = EN 1993-1-1 ; "
                      "MSB P5 = Steel Buildings in Europe, Part 5 (§2 : platine d'about partielle) ; "
                      "P358 = SCI P358 ; ECCS = ECCS n°126.")

TITRES_ALERTES = {
    "e1bot": "Impossible — pince basse du groupe de boulons",
    "e2p": "Impossible — platine trop étroite (pince e2)",
    "p3_cordon": "Impossible — la rondelle chevauche le cordon (p3 trop faible)",
    "p3_congeP": "Impossible — l'écrou chevauche le congé du poteau (p3 trop faible)",
    "p3_libreP": "Impossible — l'écrou sort de la partie droite de l'âme (p3 trop grand)",
    "bp_libreP": "Impossible — la platine ne tient pas entre les congés de l'âme",
    "bp_aile": "La platine dépasse le bout d'aile du poteau (recommandation)",
    "zc_top": "Impossible — dessus de la platine dans le grugeage ou le congé",
    "h_dispo": "Impossible — hauteur disponible insuffisante",
    "aile_poutre": "Impossible — la semelle ne passe pas entre les ailes du poteau",
    "d0_manq": "Trou surdimensionné : saisir d0",
    "tp_duct": "Platine trop épaisse : ductilité non assurée (recommandation)",
    "tp_min": "Platine < 8 mm : déconseillée (recommandation)",
    "hp_rec": "hp < 0,6·h : maintien en torsion non assuré (recommandation)",
    "a_rec": "Gorge sous la pleine résistance recommandée",
    "ame_hp": "Appui dans l'âme : flexion hors plan de l'âme non vérifiée",
    "NEd": "NEd de traction à l'ELU : les deux tronçons en T sont vérifiés",
    "N_tying": "NEd en tying : force de chaînage vérifiée seule (γMu)",
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
