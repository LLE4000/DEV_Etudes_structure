# -*- coding: utf-8 -*-
"""La barre comprimée : ses données, leurs défauts et leur relecture.

    {"nom": "P1", "profil": "HEA 200", "nuance": "S235", "N": 500.0 (kN, compression > 0), "L": 4.0 (m),
     "axes": {"y": {"appuis": "AA", "k": 1.0, "Lcr": 4.0, "maintiens": 0}, "z": {…}},
     "gamma_M0": 1.0, "gamma_M1": 1.0}

Par axe, la longueur de flambement vient des APPUIS : un cas type (k connu), un coefficient saisi (L_cr = k·L) ou une
longueur saisie (L_cr). Entre deux articulations, des MAINTIENS intermédiaires équidistants découpent la barre :
L_cr = L / (n + 1). Les coefficients des cas types sont les valeurs théoriques d'Euler (barre idéale) ; un coefficient
de pratique (0,85 pour encastrée–articulée…) se saisit.
"""
import copy
import math
from typing import Dict, Optional

from . import normes as NO
from . import section as SE

# (clé, libellé, k) ; k = None : saisi
APPUIS = [("AA", "Articulée – articulée · k = 1", 1.0), ("EA", "Encastrée – articulée · k = 0,7", 0.7),
          ("EE", "Encastrée – encastrée · k = 0,5", 0.5), ("EG", "Encastrée – guidée · k = 1", 1.0),
          ("EL", "Encastrée – libre · k = 2", 2.0), ("k", "Coefficient k saisi", None), ("L", "Longueur L_cr saisie", None)]
K_APPUIS = {k: v for k, _, v in APPUIS}
LIB_APPUIS = {k: t for k, t, _ in APPUIS}
MAINTIENS_MAX = 9
AXE_DEFAUT = {"appuis": "AA", "k": 1.0, "Lcr": 4.0, "maintiens": 0}
# le domaine des saisies : au-delà, ce n'est plus une barre (et le calcul déborderait : 1e200 m donne N_cr = 0,
# 1e-300 m un N_cr infini). Les bornes sont larges : N_pl du plus gros profilé du catalogue ≈ 60 000 kN.
N_MAX = 1.0e6                            # kN
L_MIN, L_MAX = 0.01, 100.0               # m
K_MIN, K_MAX = 0.01, 10.0
LCR_MIN, LCR_MAX = 0.001, 1000.0         # m


def nouvelle_barre(nom: str = "P1") -> Dict:
    return {"nom": nom, "profil": "HEA 200", "nuance": "S235", "N": 500.0, "L": 4.0,
            "axes": {"y": dict(AXE_DEFAUT), "z": dict(AXE_DEFAUT)}, "gamma_M0": NO.GAMMA_M0, "gamma_M1": NO.GAMMA_M1}


def _nombre(x, nom: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise ValueError(f"{nom} : nombre attendu, reçu {x!r}")
    return float(x)


def _fr(s: str) -> str:
    """Un message à la française : 1,000,000 → 1 000 000 ; 0.01 → 0,01."""
    return s.replace(",", " ").replace(".", ",")


def verifier_valeur(cle: str, x: float) -> Optional[str]:
    """Le domaine d'une saisie ; le message de refus, ou None. Une valeur hors domaine n'est jamais ramenée en silence
    à une autre."""
    if cle == "N" and x < 0:
        return "N_Ed est l'effort de COMPRESSION, positif (une traction ne flambe pas)"
    if cle == "N" and x > N_MAX:
        return _fr(f"N_Ed au plus {N_MAX:,.0f} kN")
    if cle == "L" and x <= 0:
        return "la longueur L doit être positive"
    if cle == "L" and not (L_MIN <= x <= L_MAX):
        return _fr(f"la longueur L doit être comprise entre {L_MIN:g} et {L_MAX:g} m")
    if cle == "k" and x <= 0:
        return "le coefficient k doit être positif"
    if cle == "k" and not (K_MIN <= x <= K_MAX):
        return _fr(f"le coefficient k doit être compris entre {K_MIN:g} et {K_MAX:g}")
    if cle == "Lcr" and x <= 0:
        return "la longueur de flambement L_cr doit être positive"
    if cle == "Lcr" and not (LCR_MIN <= x <= LCR_MAX):
        return _fr(f"la longueur de flambement L_cr doit être comprise entre {LCR_MIN:g} et {LCR_MAX:g} m")
    if cle == "maintiens" and not (0 <= x <= MAINTIENS_MAX):
        return f"de 0 à {MAINTIENS_MAX} maintiens intermédiaires"
    if cle in ("gamma_M0", "gamma_M1") and not (NO.GAMMA_MIN <= x <= NO.GAMMA_MAX):
        return f"γ entre {NO.GAMMA_MIN:g} et {NO.GAMMA_MAX:g} (ANB belge : 1,0)".replace(".", ",")
    return None


def normaliser(b: Dict) -> Dict:
    """Une barre relue (fichier .stab, navigateur) : défauts pour les clés absentes, ValueError claire sinon."""
    if not isinstance(b, dict):
        raise ValueError("barre illisible")
    out = nouvelle_barre()
    nom = b.get("nom", out["nom"])
    out["nom"] = "" if nom is None else str(nom)
    prof = str(b.get("profil", out["profil"]))
    SE.profil(prof)                               # ValueError si inconnu
    out["profil"] = prof
    nu = str(b.get("nuance", out["nuance"]))
    if nu not in NO.NUANCES:
        raise ValueError(f"nuance inconnue : {nu}")
    out["nuance"] = nu
    for cle in ("N", "L", "gamma_M0", "gamma_M1"):
        x = _nombre(b.get(cle, out[cle]), cle)
        msg = verifier_valeur(cle, x)
        if msg:
            raise ValueError(msg)
        out[cle] = x
    axes = b.get("axes") if isinstance(b.get("axes"), dict) else {}
    for a in ("y", "z"):
        src = axes.get(a) if isinstance(axes.get(a), dict) else {}
        ax = dict(AXE_DEFAUT)
        ap = str(src.get("appuis", ax["appuis"]))
        if ap not in K_APPUIS:
            raise ValueError(f"appuis inconnus : {ap}")
        ax["appuis"] = ap
        for cle in ("k", "Lcr"):
            x = _nombre(src.get(cle, ax[cle]), cle)
            msg = verifier_valeur(cle, x)
            if msg:
                raise ValueError(msg)
            ax[cle] = x
        m = _nombre(src.get("maintiens", 0), "maintiens")
        if m != int(m) or verifier_valeur("maintiens", m):
            raise ValueError(verifier_valeur("maintiens", m) or "nombre entier de maintiens attendu")
        ax["maintiens"] = int(m)
        out["axes"][a] = ax
    return out


def coefficient(ax: Dict, L: float) -> float:
    """Le coefficient de longueur de flambement k = L_cr / L de l'axe."""
    ap = ax["appuis"]
    if ap == "AA":
        return 1.0 / (ax["maintiens"] + 1)
    if ap == "k":
        return ax["k"]
    if ap == "L":
        return ax["Lcr"] / L
    return K_APPUIS[ap]


def longueur_flambement(ax: Dict, L: float) -> float:
    """L_cr (m) de l'axe : L/(n + 1) entre articulations, k·L d'un cas type ou saisi, ou la longueur saisie."""
    ap = ax["appuis"]
    if ap == "AA":
        return L / (ax["maintiens"] + 1)
    if ap == "L":
        return ax["Lcr"]
    return (ax["k"] if ap == "k" else K_APPUIS[ap]) * L


def copie(b: Dict) -> Dict:
    return copy.deepcopy(b)
