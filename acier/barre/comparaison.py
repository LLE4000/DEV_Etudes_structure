# -*- coding: utf-8 -*-
"""LA COMPARAISON des deux méthodes d'une barre, sujet par sujet (brief v12 § 9 : la synthèse de la note « les deux »).

Sept sujets, ceux du § 9 : flambement autour de l'axe faible et de l'axe fort, flexion composée, déversement,
cisaillement, âme sous charges, raidisseurs. Chaque identifiant de vérification que le moteur rend est rangé dans UN
sujet (``SUJETS``) : un identifiant que rien ne range est une erreur que les tests voient
(tests/test_barre_comparaison.py), jamais une ligne qui disparaît en silence.

Choix déclarés :
- les résistances de SECTION de l'Eurocode 3 (E-N, E-My, E-Mz, E-MV, E-NM) sont rangées sous « Flexion composée »,
  avec l'interaction (6.61), (6.62) : le § 9 n'a pas de ligne « section » ;
- le cisaillement de la méthode ELS est sa contrainte équivalente (H-EQ, le contrôle hérité du classeur, brief § 6.1) :
  son taux est montré, il combine σ et τ ;
- l'âme (EN 1993-1-5) vaut pour les deux méthodes : elle est « commun » — son voilement par cisaillement sous
  « Cisaillement », ses forces concentrées sous « Âme sous charges », ses raidisseurs sous « Raidisseurs » ;
- une méthode qui REFUSE la barre (E-SECTION de l'Eurocode 3, le refus du moteur) n'a pas de ligne à elle : elle est
  « non couvert » sur chaque ligne.

Une ligne : {"sujet", "historique", "ec3", "commun", "ecart"} ; une case (ou None quand la méthode n'a rien sur ce
sujet) : {"taux" (le plus fort taux JUGÉ — ok, nok, hors domaine —, None sinon), "statut" (le plus défavorable des
vérifications du sujet : resultat.pire_statut), "gouvernant" (l'identifiant de ce taux ; sans taux jugé, la première
vérification non jugée)} ; "ecart" : taux Eurocode 3 − taux ELS, en points de pourcentage, None si l'un manque.

Le moteur NOMME ; l'écran et la note composent (api_web/profil_resultat.py, api_web/note_profil.py).
"""
import re
from typing import Dict, List, Optional, Sequence

from . import resultat as RE

# (sujet, vérifications de la méthode ELS, de l'Eurocode 3, communes — l'âme, EN 1993-1-5 : identifiants exacts ou
# motifs des lieux, « A-L00-F » …)
SUJETS = (
    ("Flambement, axe faible", ("H-N-z",), ("E-Nb-z", "E-Nb-T"), ()),
    ("Flambement, axe fort", ("H-N-y",), ("E-Nb-y",), ()),
    ("Flexion composée", ("H-SIGMA",), ("E-N", "E-My", "E-Mz", "E-MV", "E-NM", "E-661", "E-662"), ()),
    ("Déversement", ("H-DEV",), ("E-Mb",), ()),
    ("Cisaillement", ("H-EQ",), ("E-Vz", "E-Vy"), ("A-Vb", "A-Vb-y", "A-VbM")),
    ("Âme sous charges", (), (), ("A-AME", "A-GR", "A-S", r"A-L\d+-F", r"A-L\d+-B", r"A-L\d+-FM", r"A-L\d+-EL")),
    ("Raidisseurs", (), (), (r"A-L\d+-R", r"A-L\d+-Rb")),
)
# le refus d'une méthode : pas de ligne, « non couvert » partout
REFUS = ("E-SECTION",)
COTES = ("historique", "ec3", "commun")


def _range(ident: str, motifs: Sequence[str]) -> bool:
    return any(re.fullmatch(m, ident) for m in motifs)


def sujet_de(ident: str) -> Optional[str]:
    """Le sujet d'un identifiant de vérification ; None s'il n'est rangé nulle part (ni refus)."""
    for nom, *cotes in SUJETS:
        if any(_range(ident, motifs) for motifs in cotes):
            return nom
    return None


def ranges() -> List[str]:
    """Tous les identifiants et motifs rangés (le refus compris) : ce que le garde-fou des tests compare."""
    return [m for _, *cotes in SUJETS for motifs in cotes for m in motifs] + list(REFUS)


def _refuse(r: Optional[Dict]) -> bool:
    """La méthode refuse-t-elle la barre (le refus du moteur, ou la section que l'Eurocode 3 ne couvre pas) ?"""
    return bool(r) and (bool(r.get("refus")) or any(v["id"] in REFUS for v in r["verifications"]))


def case(r: Optional[Dict], motifs: Sequence[str]) -> Optional[Dict]:
    """La case d'une méthode pour un sujet : son plus fort taux jugé, le pire statut, la vérification qui gouverne."""
    if not r or not motifs:
        return None
    if _refuse(r):
        return {"taux": None, "statut": RE.NON_COUVERT, "gouvernant": next(v["id"] for v in r["verifications"]
                                                                          if v["id"] in REFUS or v["statut"] == RE.NON_COUVERT)}
    vs = [v for v in r["verifications"] if _range(v["id"], motifs)]
    if not vs:
        return None
    juges = [v for v in vs if v["statut"] in RE.JUGES and v["taux"] is not None]
    g = max(juges, key=lambda v: v["taux"]) if juges else next(v for v in vs if v["statut"] not in RE.JUGES or v["taux"] is None)
    return {"taux": g["taux"] if juges else None, "statut": RE.pire_statut(v["statut"] for v in vs), "gouvernant": g["id"]}


def comparer(methodes: Dict, ame: Optional[Dict]) -> List[Dict]:
    """Les deux méthodes côte à côte, sujet par sujet (forme : la docstring du fichier). Vide quand une méthode manque."""
    if not all(methodes.get(k) for k in ("historique", "ec3")):
        return []
    lignes = []
    for nom, h, e, a in SUJETS:
        li = {"sujet": nom, "historique": case(methodes["historique"], h), "ec3": case(methodes["ec3"], e), "commun": case(ame, a)}
        if all(li[k] is None for k in COTES):
            continue
        th, te = ((li[k] or {}).get("taux") for k in ("historique", "ec3"))
        li["ecart"] = None if th is None or te is None else 100.0 * (te - th)
        lignes.append(li)
    return lignes
