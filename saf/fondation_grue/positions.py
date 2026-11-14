# -*- coding: utf-8 -*-
"""Fondation de grue — les POSITIONS DES CHARGES : la grue tourne, ses quatre réactions aussi.

Un jeu R = (R1, R2, R3, R4) est donné pour les pieds P1 à P4 (sens horaire). Toutes les positions physiques :
- rotation de k × 90° : le pied i reçoit R[(i + k) mod 4] ;
- miroir : le pied i reçoit R[(k − i) mod 4] ;
- châssis carré : k = 0 à 3 en rotation et en miroir (8 candidats) ; rectangulaire : rotations 0 et 180°, miroirs
  selon les deux axes (4 candidats) ;
- les doublons sont retirés. Les autres permutations ne correspondent à aucune position de grue.

Les cas sont nommés « j.x » (jeu j, lettre x dans l'ordre rotations puis miroirs) et rangés comme le fichier du bureau :
lettre par lettre, puis jeu par jeu — 1.a, 2.a, 3.a, 1.b, 2.b, 3.b, … LC2 est le premier, LC3 le deuxième…
"""
from typing import Dict, List, Optional, Sequence, Tuple

LETTRES = "abcdefgh"


def positions(R: Sequence[float], chassis_carre: bool = True) -> List[Tuple[float, ...]]:
    """Toutes les positions physiques des 4 réactions (rotations puis miroirs), doublons retirés."""
    ks_rot = (0, 1, 2, 3) if chassis_carre else (0, 2)
    ks_mir = (0, 1, 2, 3) if chassis_carre else (1, 3)
    candidats = [tuple(R[(i + k) % 4] for i in range(4)) for k in ks_rot]
    candidats += [tuple(R[(k - i) % 4] for i in range(4)) for k in ks_mir]
    out, vus = [], set()
    for r in candidats:
        if r not in vus:
            vus.add(r)
            out.append(r)
    return out


def _nombre(x: float) -> str:
    """Une réaction dans une description : sans zéro inutile, la virgule décimale (350 ; 135,5)."""
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return (s or "0").replace(".", ",")


def cas_de_charge(jeux: Sequence[Sequence[float]], chassis_carre: bool = True, numeros: Optional[Sequence[int]] = None) -> List[Dict]:
    """Les cas de charge des réactions, dans l'ordre du fichier : [{nom « j.x », jeu (1… : son rang parmi ``jeux``),
    lettre, R (P1 à P4), lc (« LC2 »…), description}]. Un jeu dont les quatre réactions sont nulles ne donne aucun cas.
    ``numeros`` : le numéro affiché de chaque jeu (son rang dans la liste saisie, quand un jeu incomplet est sauté)."""
    par_jeu = [positions([float(v) for v in R], chassis_carre) if any(float(v) for v in R) else [] for R in jeux]
    out = []
    for rang in range(max((len(p) for p in par_jeu), default=0)):
        for j, pos in enumerate(par_jeu, 1):
            if rang < len(pos):
                r = pos[rang]
                nom = f"{numeros[j - 1] if numeros else j}.{LETTRES[rang]}"
                out.append({"nom": nom, "jeu": j, "lettre": LETTRES[rang], "R": r, "lc": f"LC{len(out) + 2}",
                            "description": f"Réactions CAS {nom} ({'/'.join(_nombre(v) for v in r)})"})
    return out
