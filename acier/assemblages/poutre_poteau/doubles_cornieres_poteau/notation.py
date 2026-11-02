# -*- coding: utf-8 -*-
"""Textes des doubles cornières sur poteau pour l'écran et la note :
références courtes et lignes d'alerte. Le moteur écrit directement en
notation finale (hc, zc, tc, e2,b, twP, tfP, p3…) — pas de couche de
correspondance ; ``ref_courte`` est celle de la famille (module doubles
cornières poutre–poutre)."""
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
NOTATIONS = ("Notations : EN 1993-1-8 Tab. 3.3 (e1, p1, e2, p2) ; hc, zc, tc : cornières ; gA, p3 : "
             "groupe P ; e2,b : pince à l'about ; c, dc : grugeage ; gh : jeu à l'about ; "
             "rc : recoupe des semelles ; j : jeu latéral ; ℓ : largeur droite entre congés ; "
             "h, b, tw, tf : poteau ; ℓh : retour des cordons ; a : gorge ; z : excentricité ; "
             "e_col : axe du poteau → attache.")

LEGENDE_REFERENCES = ("Références : EC3 = EN 1993-1-8 sauf indication (§, Tab.) ; EC3-1-1 = EN 1993-1-1 ; "
                      "MSB P5 ; P358 = SCI P358 ; ECCS = ECCS n°126.")

TITRES_ALERTES = {
    "S_e1bot": "Impossible — pince basse du groupe S",
    "P_e1bot": "Impossible — pince basse du groupe P",
    "S_e2": "Impossible — aile B trop courte (pince e2 du groupe S)",
    "P_e2": "Impossible — aile A trop courte (pince e2 du groupe P)",
    "zc_top": "Impossible — dessus des cornières dans le grugeage ou le congé",
    "h_dispo": "Impossible — hauteur disponible insuffisante",
    "fillet": "Impossible — trou de boulon dans le congé de la cornière",
    "p3_congeP": "Impossible — l'écrou chevauche le congé du poteau (gauge p3)",
    "p3_libreP": "Impossible — l'écrou dépasse la partie droite de l'âme du poteau",
    "bA_congeP": "Impossible — l'aile A dépasse la partie droite de l'âme du poteau",
    "aile_poutre": "Impossible — la semelle ne passe pas entre les ailes du poteau",
    "one_bolt": "Impossible — un seul boulon côté poutre portée",
    "d0_manq": "Trou surdimensionné : saisir d0",
    "ame_hp": "Appui dans l'âme : flexion hors plan de l'âme non vérifiée",
    "n2P_T": "Deux files par cornière : tronçon en T de l'aile sur la file p3",
    "MEd": "MEd ≠ 0 : traité comme moment parasite",
    "NEd": "NEd de traction : flexion hors plan du poteau non vérifiée (cas b)",
    "N_tying": "NEd en tying : force de chaînage vérifiée seule (γMu)",
    "HEd": "HEd > 10 % de VEd : flexion hors plan des cornières non couverte",
    "soude2": "Soudée des deux côtés : capacité de rotation à justifier (§5.2)",
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
