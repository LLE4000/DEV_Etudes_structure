# -*- coding: utf-8 -*-
"""EUROCODE 3 — la BARRE vérifiée : sa section (ec3_section.py), puis son instabilité — flambement
(ec3_flambement.py), déversement (ec3_deversement.py, mcr.py), flexion composée (ec3_interaction.py).
NBN EN 1993-1-1 et son annexe belge ; efforts de CALCUL (ELU). docs/acier/BARRES_ACIER.md § 5.

La barre (N, mm), en plus de ce que lit ec3_section.lire :

    "axes": {"y": {"L", "u" (1), "courbe" (None : celle du tableau 6.2)}, "z": {…}}     flambement : L_cr = u·L
    "torsion": {"L", "u"}                       flambement par torsion — à défaut, la longueur de l'axe faible
    "flexion": {"y": {"diagramme", "psi"}, "z": {…}}
        le diagramme du moment de chaque axe : « constant » (à défaut — du côté sûr), « lineaire » (ψ = M₂/M₁, entre
        −1 et 1), « reparti » (charge répartie sur deux appuis), « ponctuel » (charge à mi-portée), « general » (avec
        "delta", la flèche maximale, et "L" : la formule générale du tableau A.2)
    "deversement": {"actif" (True), "Lc" (None : la longueur de l'axe faible ; 0 : semelle comprimée tenue en continu),
                    "niveau" (« centre », « haut », « bas » : où la charge est appliquée) ou "zg" (mm),
                    "Mcr" (N·mm : un moment critique donné), "kc", "troncon" (un tronçon décrit en entier : la barre
                    chargée — voir ec3_deversement.troncon_impose pour sa forme)}

Les paramètres nationaux « à confirmer » qui ont servi sont rendus dans ``grandeurs["a_confirmer"]`` : la note les
imprime.
"""
import math
from typing import Dict

from acier.flambement import normes as NO

from . import ec3_deversement as DV
from . import ec3_flambement as FL
from . import ec3_interaction as IN
from . import ec3_section as ES
from . import mcr as MC
from . import parametres as PA
from .resultat import Journal, Refus

METHODE, NORME = ES.METHODE, ES.NORME
NIVEAUX = {"centre": 0.0, "haut": 1.0, "bas": -1.0}


def _nombre(v, nom: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise Refus(f"« {nom} » doit être un nombre : {v!r}")
    return float(v)


def lire(barre: Dict) -> Dict:
    """Les données de la barre, complétées et contrôlées (une COPIE)."""
    d = ES.lire(barre)
    s = d["section"]
    axes = barre.get("axes") or {}
    d["axes"] = {}
    for a in ("y", "z"):
        ax = dict(axes.get(a) or {})
        L, u = _nombre(ax.get("L", 0.0), f"axes.{a}.L"), _nombre(ax.get("u", 1.0), f"axes.{a}.u")
        if L < 0 or u <= 0:
            raise Refus(f"axe {a} : la longueur L doit être positive ou nulle et le coefficient u positif")
        courbe = ax.get("courbe")
        if courbe is not None and courbe not in NO.ALPHA:
            raise Refus(f"axe {a} : une courbe parmi a0, a, b, c, d — reçu {courbe!r}")
        d["axes"][a] = {"L": L, "u": u, "Lcr": u * L, "courbe": courbe}
    t = dict(barre.get("torsion") or {})
    LT, uT = _nombre(t.get("L", d["axes"]["z"]["L"]), "torsion.L"), _nombre(t.get("u", d["axes"]["z"]["u"]), "torsion.u")
    if LT < 0 or uT <= 0:
        raise Refus("torsion : la longueur L doit être positive ou nulle et le coefficient u positif")
    d["torsion"] = {"L": LT, "u": uT, "Lcr": uT * LT}
    flexion = barre.get("flexion") or {}
    d["flexion"] = {}
    for a in ("y", "z"):
        f = dict(flexion.get(a) or {})
        diag = f.get("diagramme", DV.CONSTANT)
        if diag not in DV.DIAGRAMMES:
            raise Refus(f"flexion.{a} : diagramme parmi {', '.join(DV.DIAGRAMMES)} — reçu {diag!r}")
        psi = _nombre(f.get("psi", 1.0), f"flexion.{a}.psi")
        if not -1.0 <= psi <= 1.0:
            raise Refus(f"flexion.{a} : ψ = M₂/M₁ doit être entre −1 et 1 (|M₂| ≤ |M₁|) : {psi!r}")
        delta, Lf = f.get("delta"), f.get("L")
        d["flexion"][a] = {"diagramme": diag, "psi": psi, "delta": None if delta is None else _nombre(delta, f"flexion.{a}.delta"),
                           "L": None if Lf is None else _nombre(Lf, f"flexion.{a}.L")}
    dv = dict(barre.get("deversement") or {})
    Lc = dv.get("Lc")
    Lc = d["axes"]["z"]["L"] if Lc is None else _nombre(Lc, "deversement.Lc")
    if Lc < 0:
        raise Refus("la distance L_c entre sections tenues doit être positive ou nulle")
    niveau = dv.get("niveau", "centre")
    if niveau not in NIVEAUX:
        raise Refus(f"deversement.niveau : centre, haut ou bas — reçu {niveau!r}")
    zg = dv.get("zg")
    zg = NIVEAUX[niveau] * (s.get("h") or 0.0) / 2.0 if zg is None else _nombre(zg, "deversement.zg")
    Mcr, kc, tr = dv.get("Mcr"), dv.get("kc"), dv.get("troncon")
    if Mcr is not None and not _nombre(Mcr, "deversement.Mcr") > 0:
        raise Refus("le moment critique donné doit être positif")
    if kc is not None and not 0.0 < _nombre(kc, "deversement.kc") <= 1.0:
        raise Refus("k_c doit être entre 0 et 1")
    if tr is not None:
        if not isinstance(tr, dict) or not callable(tr.get("moment")) or not _nombre(tr.get("L", 0.0), "troncon.L") > 0:
            raise Refus("deversement.troncon : une longueur L positive et une fonction « moment »")
        tr = {"appuis": (MC.FOURCHE, MC.FOURCHE), "maintiens": [], **tr}
        tr["appuis"] = tuple(tr["appuis"])
        if len(tr["appuis"]) != 2 or any(x not in MC.APPUIS for x in tr["appuis"]):
            raise Refus(f"deversement.troncon : deux bouts parmi {', '.join(MC.APPUIS)}")
        Lc = tr["L"]
    d["deversement"] = {"actif": bool(dv.get("actif", True)), "Lc": Lc, "niveau": niveau, "zg": zg, "Mcr": Mcr, "kc": kc, "troncon": tr}
    return d


def verifier(barre: Dict) -> Dict:
    """La barre vérifiée selon l'Eurocode 3 : la section (§ 6.2) puis la barre (§ 6.3). Rend la forme de
    acier/barre/resultat.py ; vérifications E-N, E-My, E-Mz, E-Vz, E-Vy, E-Vb, E-MV, E-NM (section), E-Nb-y, E-Nb-z,
    E-Nb-T (flambement), E-Mb (déversement), E-661, E-662 (flexion composée avec instabilité)."""
    d = lire(barre)
    J = Journal(METHODE, NORME)
    ES.ajouter(J, d)
    FL.ajouter(J, d)
    DV.ajouter(J, d)
    IN.ajouter(J, d)
    G = J.G
    cles = set(G.get("deversement", {}).get("a_confirmer", [])) | set(G.get("a_confirmer_interaction", []))
    G["a_confirmer"] = PA.a_confirmer(cles) if cles else []
    return J.rendre()
