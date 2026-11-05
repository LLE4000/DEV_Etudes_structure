# -*- coding: utf-8 -*-
"""Les charges d'un garde-corps et leur nature.

- la charge LINÉAIRE HORIZONTALE q_h (kN/m) sur la lisse haute, à la hauteur de la main
  courante (au plus 1,20 m) : EN 1991-1-1 §6.4, Tableau 6.12, par catégorie d'usage — les
  valeurs ci-dessous sont les valeurs RECOMMANDÉES de l'EN ; l'annexe nationale belge (NBN EN
  1991-1-1 ANB) peut en fixer d'autres : la page le dit, la note imprime la valeur retenue et
  son origine, et une valeur libre est toujours possible ;
- ou directement la charge PONCTUELLE P (kN) qui revient à chaque montant (mode « P ») ;
- une charge VERTICALE q_v (kN/m) sur la main courante (usage du bureau) ;
- une charge ponctuelle horizontale Q (kN) sur la lisse haute, en option ;
- les charges du REMPLISSAGE : une pression q_r (kN/m²) et une charge ponctuelle Q_r (kN) —
  deux cas séparés, le plus défavorable gouverne (NBN B 03-004 : Q_kh,2 = 0,5 kN sur le
  remplissage).

NATURE : la méthode ELS du bureau prend les charges de service telles quelles (contraintes
admissibles) ; la méthode Eurocode les multiplie par γ_Q pour les résistances, et garde les
valeurs de service pour les flèches.
"""
from typing import Dict, List, Optional, Tuple

# (clé, libellé, q_k recommandé EN 1991-1-1 Tab. 6.12 en kN/m, plage de la note du tableau)
CATEGORIES: List[Tuple[str, str, Optional[float], str]] = [
    ("A", "A · habitation", 0.5, "0,2 à 1,0"),
    ("B", "B · bureaux", 0.5, "0,2 à 1,0"),
    ("C1", "C1 · lieux de réunion avec tables (écoles, cafés)", 0.5, "0,2 à 1,0"),
    ("C2-C4", "C2 à C4 · sièges fixes, sans obstacle, activités physiques", 1.0, "0,8 à 1,0"),
    ("D", "D · commerces", 1.0, "0,8 à 1,0"),
    ("C5", "C5 · foules (salles de concert, tribunes, quais)", 3.0, "3,0 à 5,0"),
    ("E", "E · stockage, industrie", 2.0, "0,8 à 2,0"),
    ("perso", "valeur libre", None, ""),
]
CATEGORIE_DEFAUT = "A"
QV_DEFAUT = 0.5        # kN/m, charge verticale de service sur la main courante (usage du bureau)
QR_DEFAUT = 0.5        # kN/m², pression de service sur le remplissage (usage du bureau)
QR_PONCT_DEFAUT = 0.5  # kN, charge ponctuelle sur le remplissage (NBN B 03-004, Q_kh,2)
H_MAX_APPLICATION = 1200.0   # mm : la charge linéaire s'applique au plus à 1,20 m (EN 1991-1-1 §6.4(1))
SOURCE_QH = "EN 1991-1-1 §6.4 Tab. 6.12 (valeur recommandée — l'ANB belge peut en fixer une autre)"
SOURCE_QV = "charge verticale de service sur la main courante, usage du bureau"
SOURCE_QR = "pression de service sur le remplissage, usage du bureau"
SOURCE_QR_PONCT = "charge ponctuelle sur le remplissage, NBN B 03-004 (Q_kh,2)"


def categorie(cle: str) -> Tuple[str, str, Optional[float], str]:
    for c in CATEGORIES:
        if c[0] == cle:
            return c
    return CATEGORIES[-1]


def qh_de_categorie(cle: str) -> Optional[float]:
    return categorie(cle)[2]


def charges_defaut() -> Dict:
    return {"categorie": CATEGORIE_DEFAUT, "qh": qh_de_categorie(CATEGORIE_DEFAUT), "mode": "q", "P": 1.0,
            "qv": QV_DEFAUT, "Q": 0.0, "qr": QR_DEFAUT, "Qr": QR_PONCT_DEFAUT}


def normaliser(ch: Dict) -> Dict:
    """Complète une saisie de charges : la catégorie fixe q_h sauf en valeur libre."""
    d = charges_defaut()
    d.update({k: v for k, v in (ch or {}).items() if v is not None})
    cat = d.get("categorie") if d.get("categorie") in {c[0] for c in CATEGORIES} else "perso"
    d["categorie"] = cat
    q_cat = qh_de_categorie(cat)
    if q_cat is not None:
        d["qh"] = q_cat
    d["qh"] = max(0.0, float(d.get("qh") or 0.0))
    d["mode"] = "P" if d.get("mode") == "P" else "q"
    for k in ("P", "qv", "Q", "qr", "Qr"):
        d[k] = max(0.0, float(d.get(k) or 0.0))
    return d


def libelle_categorie(cle: str) -> str:
    return categorie(cle)[1]


def origine_qh(ch: Dict) -> str:
    """L'origine imprimable de la charge horizontale retenue."""
    if ch.get("mode") == "P":
        return "charge ponctuelle par montant saisie directement"
    if ch.get("categorie") == "perso":
        return "valeur libre saisie par le bureau"
    return f"catégorie {libelle_categorie(ch['categorie'])} — {SOURCE_QH}"


def phrase(ch: Dict) -> str:
    """Une ligne qui dit les charges : « q_h = 0,5 kN/m (catégorie A …), q_v = 0,5 kN/m, … »."""
    def f(v, nd=2):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".").replace(".", ",")
    if ch["mode"] == "P":
        s = f"P = {f(ch['P'])} kN par montant (saisie directe)"
    else:
        s = f"q_h = {f(ch['qh'])} kN/m ({origine_qh(ch)})"
    s += f" · q_v = {f(ch['qv'])} kN/m"
    if ch.get("Q", 0) > 0:
        s += f" · Q = {f(ch['Q'])} kN sur la lisse"
    s += f" · remplissage q_r = {f(ch['qr'])} kN/m², Q_r = {f(ch['Qr'])} kN"
    return s
