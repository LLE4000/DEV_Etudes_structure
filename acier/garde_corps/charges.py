# -*- coding: utf-8 -*-
"""Les QUATRE charges d'un garde-corps, nommées, et leur nature.

1. q_h — charge linéaire HORIZONTALE sur la main courante, à la hauteur H (au plus 1,20 m) :
   EN 1991-1-1 §6.4, Tableau 6.12, par catégorie d'usage (NBN B 03-004 : Q_kh,1). Les valeurs
   ci-dessous sont les valeurs RECOMMANDÉES de l'EN ; l'annexe nationale belge peut en fixer
   d'autres : valeur libre toujours possible, la note imprime la valeur retenue et son origine.
   Ou, directement, la charge ponctuelle P qui revient à chaque montant (mode « P »).
2. q_v — charge linéaire VERTICALE sur la main courante (usage du bureau ; NBN B 03-004 Q_kv).
3. Q_r — charge PONCTUELLE horizontale sur le remplissage (NBN B 03-004 : Q_kh,2 = 0,5 kN).
4. q_r — PRESSION horizontale répartie sur le remplissage (kN/m², usage du bureau).

CAS DE CHARGE : par défaut les quatre charges sont des cas DISTINCTS — chacune agit seule
(pratique des normes de garde-corps : les charges de service ne se cumulent pas). L'option
``cumul`` fait agir les charges 1 et 2 ensemble (côté sûr). Les charges 3 et 4 restent toujours
des cas séparés. En option, une charge ponctuelle horizontale Q sur la main courante.

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
    ("ind", "plateforme industrielle (machines)", 0.3, "0,3"),
    ("perso", "valeur libre", None, ""),
]
# L'origine de q_h hors Tableau 6.12 : garde-corps des plateformes et passerelles de machines (EN ISO 14122-3 : 300 N/m
# horizontaux en tête de main courante — valeur à confirmer sur l'édition en vigueur ; sa flèche admissible se saisit en mm)
SOURCES_CAT = {"ind": "EN ISO 14122-3, 0,3 kN/m à confirmer sur l'édition en vigueur"}
CATEGORIE_DEFAUT = "A"
QV_DEFAUT = 0.5        # kN/m
QR_DEFAUT = 0.5        # kN/m²
QR_PONCT_DEFAUT = 0.5  # kN (NBN B 03-004, Q_kh,2)
H_MAX_APPLICATION = 1200.0   # mm (EN 1991-1-1 §6.4(1))
SOURCE_QH = "EN 1991-1-1 §6.4 Tab. 6.12 (valeur recommandée — l'ANB belge peut en fixer une autre)"
SOURCE_QV = "charge verticale de service sur la main courante, usage du bureau"
SOURCE_QR = "pression de service sur le remplissage, usage du bureau"
SOURCE_QR_PONCT = "charge ponctuelle sur le remplissage, NBN B 03-004 (Q_kh,2)"

# Les quatre charges, dans l'ordre de la note : numéro, clé, symbole, nom, unité, norme
CHARGES: List[Dict] = [
    {"n": 1, "cle": "qh", "symbole": "q_h", "nom": "charge linéaire horizontale sur la main courante", "unite": "kN/m",
     "norme": "EN 1991-1-1 §6.4, Tab. 6.12 (NBN B 03-004 : Q_kh,1)"},
    {"n": 2, "cle": "qv", "symbole": "q_v", "nom": "charge linéaire verticale sur la main courante", "unite": "kN/m",
     "norme": "usage du bureau (NBN B 03-004 : Q_kv, à confirmer)"},
    {"n": 3, "cle": "Qr", "symbole": "Q_r", "nom": "charge ponctuelle horizontale sur le remplissage", "unite": "kN",
     "norme": "NBN B 03-004 : Q_kh,2"},
    {"n": 4, "cle": "qr", "symbole": "q_r", "nom": "pression horizontale répartie sur le remplissage", "unite": "kN/m²",
     "norme": "usage du bureau"},
]
CAS_LIBELLES = {"1": "cas 1 · q_h seule", "2": "cas 2 · q_v seule", "3": "cas 3 · Q_r seule", "4": "cas 4 · q_r seule", "1+2": "cas 1 + 2 · q_h et q_v cumulées"}


def categorie(cle: str) -> Tuple[str, str, Optional[float], str]:
    for c in CATEGORIES:
        if c[0] == cle:
            return c
    return CATEGORIES[-1]


def qh_de_categorie(cle: str) -> Optional[float]:
    return categorie(cle)[2]


def charges_defaut() -> Dict:
    return {"categorie": CATEGORIE_DEFAUT, "qh": qh_de_categorie(CATEGORIE_DEFAUT), "mode": "q", "P": 1.0,
            "qv": QV_DEFAUT, "Q": 0.0, "qr": QR_DEFAUT, "Qr": QR_PONCT_DEFAUT, "cumul": False}


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
    d["cumul"] = bool(d.get("cumul", False))
    for k in ("P", "qv", "Q", "qr", "Qr"):
        d[k] = max(0.0, float(d.get(k) or 0.0))
    return d


def libelle_categorie(cle: str) -> str:
    return categorie(cle)[1]


def origine_qh(ch: Dict) -> str:
    if ch.get("mode") == "P":
        return "charge ponctuelle par montant saisie directement"
    if ch.get("categorie") == "perso":
        return "valeur libre saisie par le bureau"
    if ch.get("categorie") in SOURCES_CAT:
        return f"{libelle_categorie(ch['categorie'])} — {SOURCES_CAT[ch['categorie']]}"
    return f"catégorie {libelle_categorie(ch['categorie'])} — {SOURCE_QH}"


def valeur(ch: Dict, cle: str) -> float:
    return float(ch.get(cle) or 0.0)


def tableau(ch: Dict) -> List[Dict]:
    """Les quatre charges avec leur valeur et leur origine, pour les hypothèses de la note et l'écran."""
    out = []
    for c in CHARGES:
        d = dict(c)
        if c["cle"] == "qh" and ch.get("mode") == "P":
            d.update({"symbole": "P", "nom": "charge ponctuelle horizontale en tête de chaque montant", "unite": "kN", "valeur": ch["P"],
                      "norme": "saisie directe"})
        else:
            d["valeur"] = valeur(ch, c["cle"])
            if c["cle"] == "qh":
                d["norme"] = f"{libelle_categorie(ch['categorie'])} — " + ("valeur libre" if ch["categorie"] == "perso" else
                                                                           SOURCES_CAT.get(ch["categorie"], "Tab. 6.12 EN 1991-1-1, valeur recommandée"))
        out.append(d)
    return out


def phrase_cumul(ch: Dict) -> str:
    if ch.get("cumul"):
        return "charges 1 et 2 cumulées (côté sûr) ; charges 3 et 4 en cas séparés"
    return "quatre cas de charge distincts : chaque charge agit seule, sans cumul"


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
        s += f" · Q = {f(ch['Q'])} kN sur la main courante"
    s += f" · Q_r = {f(ch['Qr'])} kN · q_r = {f(ch['qr'])} kN/m² · {phrase_cumul(ch)}"
    return s
