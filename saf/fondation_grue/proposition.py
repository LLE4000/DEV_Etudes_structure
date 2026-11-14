# -*- coding: utf-8 -*-
"""Fondation de grue — « PROPOSER » : pour chaque épaisseur, la plus petite dalle dont la contrainte de contact tient.

- Rectangle : L_x et L_y gagnent le même d de chaque côté ; contour : il est décalé de d vers l'extérieur. La grue garde
  ses cotes par rapport au centre ; les appuis et les plots choisis sont gardés.
- d est un multiple de 5 cm, cherché par dichotomie (σ max diminue quand d augmente).
- σ max ne varie pas toujours dans le même sens avec l'épaisseur (le poids propre augmente) : chaque épaisseur est
  essayée, sans dichotomie sur h.
- Critère : σ max ≤ limite pour TOUS les k cochés « SCIA ». Maille de 0,25 m au plus (une maille de 0,50 m peut
  sous-estimer de 4 % quand la plaque fait 0,50 m).

    pour_epaisseur(el, h) -> {"h", "d" (None : aucune dalle ne tient jusqu'à ``D_MAX``), "Lx", "Ly" (rectangle),
                              "sommets" (contour), "sigma_max" (kPa), "volume" (m³ : dalle + plots)}
    tableau(el)           -> [pour_epaisseur de chaque épaisseur testée]
    meilleure(lignes)     -> le rang de la ligne au plus petit volume, parmi celles qui tiennent (None sinon)
"""
import copy
from typing import Dict, List, Optional

from . import projet as PR
from . import verification as V

PAS, D_MAX, MAILLE = 0.05, 6.0, 0.25


def pour_epaisseur(el: Dict, h: float, ks: Optional[List[float]] = None, rapport_c2=None) -> Dict:
    """La plus petite dalle d'épaisseur ``h`` qui tient (voir l'en-tête du fichier). ``ks`` : les k à tenir (par défaut
    ceux cochés « SCIA ») ; ``rapport_c2`` : (C2x/k, C2y/k), par défaut ceux des options."""
    e = copy.deepcopy(el)
    e["dalle"]["h"] = h
    e["options"]["maille"] = min(e["options"]["maille"], MAILLE)
    ks = ks if ks is not None else PR.k_scia(e)
    rapport_c2 = rapport_c2 if rapport_c2 is not None else (e["options"]["c2x"], e["options"]["c2y"])
    limite = e["limite"] * V.KG_CM2

    def modele_de(d: float) -> Optional[Dict]:
        a = PR.agrandi(e, d)
        if a is None:
            return None
        m = PR.modele(a)
        ctl = m["controles"]
        if not all(ctl["plaques_dedans"]) or (m["plots"] and not all(ctl["plots_dedans"])):
            return None                                           # la grue sort de la dalle : σ infinie, on agrandit
        return m

    d, sigma = V.proposer_un(modele_de, ks, limite, rapport_c2, PAS, D_MAX)
    out = {"h": h, "d": d, "sigma_max": sigma, "Lx": None, "Ly": None, "sommets": None, "volume": None}
    if d is not None:
        a = PR.agrandi(e, d)
        out["volume"] = V.volume(PR.modele(a))
        if a["dalle"]["forme"] == "rectangle":
            out["Lx"], out["Ly"] = round(a["dalle"]["Lx"], 3), round(a["dalle"]["Ly"], 3)
        else:
            out["sommets"] = [[round(x, 3), round(y, 3)] for x, y in a["dalle"]["sommets"]]
    return out


def tableau(el: Dict, ks: Optional[List[float]] = None, rapport_c2=None) -> List[Dict]:
    return [pour_epaisseur(el, h, ks, rapport_c2) for h in PR.epaisseurs(el)]


def meilleure(lignes: List[Dict]) -> Optional[int]:
    """Le rang de la ligne au plus petit volume de béton, parmi celles qui tiennent."""
    ok = [i for i, li in enumerate(lignes) if li and li.get("d") is not None]
    return min(ok, key=lambda i: lignes[i]["volume"]) if ok else None
