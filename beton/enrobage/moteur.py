# -*- coding: utf-8 -*-
"""Enrobage du béton — le CALCUL, extrait de l'écran Streamlit ``modules/enrobage.py`` (VERSION 2.0), qui reste la
référence : ``a_feu``, ``c_durabilite`` et ``c_feu`` sont ses fonctions, mot pour mot ; ``calculer`` enchaîne ce que
sa fonction ``show()`` fait de leurs résultats (le maximum, le gouvernant, les deux encadrés d'information) et NOMME
chaque terme avec sa règle. Bibliothèque standard seule (le site le fait tourner dans le navigateur).

    from beton.enrobage import moteur as M
    res = M.calculer(M.nouvel_element())
    res["c_nom"]          # 30 : l'enrobage nominal retenu (mm)

Entrée : un élément au format de ``beton/enrobage/entree.py`` (normalisé ici). Sortie de ``calculer`` (JSON strict,
nombres non arrondis — l'écran formate ; les types sont ceux du module : entiers pour la durabilité, flottants pour
le feu) :

- ``element`` : l'élément normalisé ;
- ``exposition`` : la classe et ses quatre textes (titre, mécanisme, exemples, quand la choisir) ;
- ``durabilite`` : ``classe_structurale``, ``classe_exposition``, ``classe_structurale_proposee``, ``c_min_dur``,
  ``c_min_b``, ``plancher`` (10 mm), ``delta_dur_gamma`` / ``delta_dur_st`` / ``delta_dur_add`` (0 : laissés à 0 par
  l'écran), ``c_min``, ``terme_c_min`` (``"c_min_b"``, ``"c_min_dur"`` ou ``"plancher"`` : le premier terme qui
  atteint le maximum, dans l'ordre de l'écran), ``delta_dev``, ``c_nom`` ;
- ``feu`` : ``None`` si la résistance au feu n'est pas vérifiée, sinon ``r``, ``tableau`` (« 5.5 »…), ``a``, ``note``
  (la caption de l'écran), ``b_min`` (poutre, poteau), ``variantes`` ([[b_min, a], …]), ``demi_barre`` (Ø/2),
  ``diam_etrier``, ``c_min`` (borné à 0), ``c_brut`` (a − Ø/2 − Ø_étrier avant la borne), ``delta_dev``, ``c_nom``,
  ``couvert`` (c_min ≤ 0 : « feu non dimensionnant ») ;
- ``c_nom_dur``, ``c_nom_feu`` (0.0 si le feu n'est pas vérifié, comme l'écran), ``c_nom`` = max(c_nom_dur ;
  c_nom_feu), ``gouvernant`` (``"durabilite"`` si c_nom_dur ≥ c_nom_feu, sinon ``"feu"``), ``proches`` (feu vérifié et
  |c_nom_dur − c_nom_feu| ≤ 2 mm : l'encadré « Durabilité et feu proches ») ;
- ``termes`` : chaque grandeur avec son symbole, sa valeur, son unité et sa RÈGLE (EN 1992-1-1 §4.4.1, EN 1992-1-2 §5,
  conventions de l'écran) ;
- ``avertissements`` : [{code, message, grave}] — les deux encadrés de l'écran (``feu_proche``, ``feu_couvert``) et
  les garde-fous du site (voir ``avertissements``).

``candidats_feu(element)`` : les variantes du tableau feu (poutre, poteau) recalculées une à une — le sélecteur.
"""
from . import donnees as D
from . import entree as EN

REFERENCE = "modules/enrobage.py (VERSION 2.0)"
nouvel_element = EN.nouvel_element
normaliser = EN.normaliser
appliquer = EN.appliquer
variantes = EN.variantes

# les tableaux de l'EN 1992-1-2 dont viennent les distances à l'axe (les captions de l'écran les citent)
TABLEAUX_FEU = {"Dalle": "5.8", "Voile": "5.4", "Poteau": "5.2a", "Poutre": "5.5"}
SECTIONS_FEU = {"Dalle": "§5.7", "Voile": "§5.4", "Poteau": "§5.3", "Poutre": "§5.6"}


# ============================================================ les fonctions de l'écran, mot pour mot
def a_feu(type_element: str, r_level: str, b_min_choice=None):
    """
    Retourne (a_mm, note) : distance à l'axe requise selon l'élément,
    la durée R et éventuellement la largeur mini choisie (poutre/poteau).
    """
    if type_element == "Dalle":
        return D.FEU_DALLE.get(r_level, 0), "Dalle pleine portant sur 1 direction (tab. 5.8)."
    if type_element == "Voile":
        return D.FEU_VOILE.get(r_level, 0), "Voile porteur exposé sur 1 face, μ_fi≈0,7 (tab. 5.4)."
    if type_element == "Poteau":
        variants = D.FEU_POTEAU.get(r_level, [])
        if not variants:
            return 0, ""
        if b_min_choice is None:
            b_min_choice = variants[0]
        b, a = b_min_choice
        return a, f"Poteau exposé sur >1 face, μ_fi≈0,7, b_min={b} mm (tab. 5.2a)."
    if type_element == "Poutre":
        variants = D.FEU_POUTRE.get(r_level, [])
        if not variants:
            return 0, ""
        if b_min_choice is None:
            b_min_choice = variants[0]
        b, a = b_min_choice
        return a, f"Poutre sur appuis simples, b_min={b} mm (tab. 5.5)."
    return 0, ""


def c_durabilite(classe_struct: str, classe_expo: str, diam_barre: float, delta_dev: float):
    """Retourne (c_min_dur, c_min_b, c_min, c_nom)."""
    idx = D.EXPO_CLASSES.index(classe_expo)
    c_min_dur = D.C_MIN_DUR[classe_struct][idx]
    c_min_b = diam_barre
    c_min = max(c_min_b, c_min_dur, 10)
    return c_min_dur, c_min_b, c_min, c_min + delta_dev


def c_feu(a_mm: float, diam_barre: float, diam_etrier: float, delta_dev: float):
    """
    c_feu,min = a − Ø_barre/2 − Ø_étrier  (enrobage au nu de l'étrier).
    Retourne (c_feu_min, c_feu_nom). Borné à 0.
    """
    c_min = max(a_mm - diam_barre / 2.0 - diam_etrier, 0.0)
    return c_min, c_min + delta_dev


# ============================================================ le calcul d'un élément (l'enchaînement de show())
def _variante(el: dict):
    """Le tuple (b_min, a) que le menu « Largeur minimale » de l'écran rendrait (None pour une dalle, un voile)."""
    return next((v for v in EN.variantes(el["type"], el["r"]) if v[0] == el["b_min"]), None)


def calculer(element: dict) -> dict:
    """L'enrobage d'un élément : durabilité, feu (s'il est vérifié), valeur retenue — comme l'écran."""
    el = EN.normaliser(element)
    expo, struct, dmax, de, dd = el["classe_expo"], el["classe_struct"], el["diam_barre"], el["diam_etrier"], el["delta_dev"]
    # --- A. Durabilité
    cmd, cmb, cmin_d, cnom_d = c_durabilite(struct, expo, dmax, dd)
    terme = "c_min_b" if cmin_d == cmb else ("c_min_dur" if cmin_d == cmd else "plancher")
    dur = {"classe_structurale": struct, "classe_exposition": expo, "classe_structurale_proposee": D.CLASSE_STRUCTURALE_DEFAULT[expo],
           "c_min_dur": cmd, "c_min_b": cmb, "plancher": 10, "delta_dur_gamma": 0, "delta_dur_st": 0, "delta_dur_add": 0,
           "c_min": cmin_d, "terme_c_min": terme, "delta_dev": dd, "c_nom": cnom_d}
    # --- B. Feu
    cnom_f = 0.0
    feu = None
    if el["feu"] and el["r"]:
        choix = _variante(el)
        a_val, note = a_feu(el["type"], el["r"], choix)
        cmin_f, cnom_f = c_feu(a_val, dmax, de, dd)
        feu = {"r": el["r"], "tableau": TABLEAUX_FEU[el["type"]], "section": SECTIONS_FEU[el["type"]], "a": a_val, "note": note,
               "b_min": choix[0] if choix else None, "variantes": [list(v) for v in EN.variantes(el["type"], el["r"])],
               "demi_barre": dmax / 2.0, "diam_etrier": de, "c_brut": a_val - dmax / 2.0 - de, "c_min": cmin_f,
               "delta_dev": dd, "c_nom": cnom_f, "couvert": cmin_f <= 0}
    # --- C. Retenu
    c_final = max(cnom_d, cnom_f)
    gouv = "durabilite" if cnom_d >= cnom_f else "feu"
    proches = bool(el["feu"] and abs(cnom_d - cnom_f) <= 2)
    res = {"reference": REFERENCE, "element": el,
           "exposition": dict(zip(("classe", "titre", "description", "exemples", "quand"), (expo,) + D.EXPOSITION_INFO[expo])),
           "durabilite": dur, "feu": feu, "c_nom_dur": cnom_d, "c_nom_feu": cnom_f, "c_nom": c_final, "gouvernant": gouv,
           "proches": proches}
    res["termes"] = termes(res)
    res["avertissements"] = avertissements(res)
    return res


# ============================================================ les termes et leurs règles (le moteur NOMME)
REGLES = {
    "c_nom_dur": "EN 1992-1-1 §4.4.1.1(2), éq. (4.1) : c_nom = c_min + Δc_dev",
    "c_min": "EN 1992-1-1 §4.4.1.2(2), éq. (4.2) : c_min = max(c_min,b ; c_min,dur + Δc_dur,γ − Δc_dur,st − Δc_dur,add ; 10 mm)",
    "c_min_b": "EN 1992-1-1 §4.4.1.2(3), tab. 4.2 : adhérence, barre individuelle — c_min,b = Ø de la barre principale",
    "c_min_dur": "EN 1992-1-1 §4.4.1.2(5), tab. 4.4N : durabilité selon la classe structurale (tab. 4.3N) et l'exposition",
    "plancher": "EN 1992-1-1 éq. (4.2) : 10 mm au moins",
    "delta_dur": "EN 1992-1-1 §4.4.1.2(6) à (8) : Δc_dur,γ, Δc_dur,st, Δc_dur,add — laissés à 0 par l'écran (valeurs recommandées)",
    "delta_dev": "EN 1992-1-1 §4.4.1.3(1)P : 10 mm recommandé ; (3) 10 ≥ Δc_dev ≥ 5 mm avec contrôle de la qualité, jusqu'à 0 avec mesure précise",
    "a": "EN 1992-1-2 {section}, tab. {tableau} : distance à l'axe des armatures principales (méthode tabulée)",
    "c_feu_min": "Convention de l'écran : c_feu,min = a − Ø/2 − Ø étrier (au nu de l'étrier), borné à 0",
    "c_nom_feu": "Convention de l'écran : c_nom,feu = c_feu,min + Δc_dev",
    "c_nom": "Écran : c_nom = max(c_nom,dur ; c_nom,feu)",
}


def termes(res: dict) -> list:
    """Chaque grandeur du calcul, dans l'ordre de l'écran, avec son symbole, son unité et sa règle."""
    d, f = res["durabilite"], res["feu"]

    def t(cle, symbole, nom, valeur, regle_cle=None, unite="mm"):
        return {"cle": cle, "symbole": symbole, "nom": nom, "valeur": valeur, "unite": unite, "regle": REGLES[regle_cle or cle]}
    out = [t("c_min_b", "c_min,b", "adhérence (= Ø barre)", d["c_min_b"]),
           t("c_min_dur", "c_min,dur", f"durabilité ({d['classe_structurale']} / {d['classe_exposition']})", d["c_min_dur"]),
           t("plancher", "c_min ≥ 10 mm", "minimum absolu", d["plancher"]),
           t("delta_dur", "Δc_dur", "Δc_dur,γ − Δc_dur,st − Δc_dur,add", d["delta_dur_gamma"] - d["delta_dur_st"] - d["delta_dur_add"]),
           t("c_min", "c_min", "enrobage minimal", d["c_min"]),
           t("delta_dev", "Δc_dev", "tolérance d'exécution", d["delta_dev"]),
           t("c_nom_dur", "c_nom,dur", "enrobage nominal · durabilité", d["c_nom"])]
    if f is not None:
        out += [{**t("a", "a", f"distance à l'axe ({f['r']}, tab. {f['tableau']})", f["a"]),
                 "regle": REGLES["a"].format(section=f["section"], tableau=f["tableau"])},
                t("c_feu_min", "c_feu,min", "a − Ø/2 − Ø_étrier", f["c_min"]),
                t("c_nom_feu", "c_nom,feu", "enrobage nominal · feu", f["c_nom"])]
    out.append(t("c_nom", "c_nom", "enrobage nominal retenu", res["c_nom"]))
    return out


# ============================================================ les avertissements
def avertissements(res: dict) -> list:
    """Les deux encadrés de l'écran, puis les GARDE-FOUS du site (ils n'ajoutent aucun calcul : ils disent ce que
    l'écran tait — docs/beton/ENROBAGE_MOTEUR.md § 6)."""
    el, f, d = res["element"], res["feu"], res["durabilite"]
    out = []

    def a(code, message, grave=False):
        out.append({"code": code, "message": message, "grave": grave})
    # --- les encadrés de l'écran Streamlit
    if f is not None and f["couvert"]:
        a("feu_couvert", "Distance à l'axe couverte par la durabilité : feu non dimensionnant.")
    if res["proches"]:
        a("feu_proche", "Durabilité et feu proches (écart ≤ 2 mm) : vérifier le cas réel.")
    # --- les garde-fous du site
    if f is not None and el["type"] == "Poteau":
        a("tableau_poteau", "Poteau : les distances a de l'écran ne sont pas celles de la colonne μ_fi = 0,7 du tableau 5.2a "
                            "annoncée (elles se rapprochent de la colonne μ_fi = 0,2) — à vérifier dans la norme avant usage.", True)
    if f is not None and el["type"] == "Voile" and el["r"] in ("R180", "R240"):
        a("tableau_voile", f"Voile {el['r']} : a = {f['a']} mm repris de l'écran ; le tableau 5.4 (μ_fi = 0,7, une face) "
                           "semble demander 5 mm de plus — à vérifier dans la norme avant usage.", True)
    if f is not None and el["type"] in ("Dalle", "Voile") and el["diam_etrier"] > 0:
        a("etrier_decompte", f"{el['type']} : le Ø étrier de {el['diam_etrier']} mm est décompté de la distance à l'axe ; "
                             "mettre 0 s'il n'y a ni étrier ni barre extérieure aux armatures principales.")
    if D.CLASSES_STRUCTURALES.index(d["classe_structurale"]) < D.CLASSES_STRUCTURALES.index("S4"):
        a("classe_reduite", f"Classe structurale {d['classe_structurale']}, sous S4 (référence 50 ans) : la réduction se "
                            "justifie par le tableau 4.3N (classe de résistance, élément plan, contrôle de la qualité).")
    if el["position"] == "Extérieur" and d["classe_exposition"] in ("X0", "XC1"):
        a("position_exposition", f"Élément extérieur classé {d['classe_exposition']} : la position n'entre pas dans le calcul ; "
                                 "vérifier la classe (extérieur abrité : XC3 ; exposé à la pluie : XC4).")
    if d["classe_exposition"] == "XC2/XC3":
        a("contre_sol", "Si le béton est coulé contre le sol (fondation, XC2) : l'EN 1992-1-1 §4.4.1.3(4) demande c_nom ≥ 40 mm "
                        "sur sol préparé (béton de propreté) et ≥ 75 mm directement contre le sol — non vérifié ici.")
    return out


# ============================================================ le sélecteur de la largeur (feu)
def candidats_feu(element: dict) -> list:
    """Les variantes (b_min, a) du tableau feu de l'élément (poutre, poteau ; [] sinon), chacune recalculée comme si
    elle était retenue : ``b_min``, ``a``, ``c_min_feu``, ``c_nom_feu``, ``c_nom`` (retenu), ``gouvernant``,
    ``retenue`` (la largeur de l'élément), ``optimale`` (la plus petite largeur qui donne le plus petit c_nom retenu :
    au-delà, élargir l'élément ne réduit plus l'enrobage)."""
    el = EN.normaliser(element)
    out = []
    for b, a in EN.variantes(el["type"], el["r"]):
        r = calculer({**el, "feu": True, "b_min": b})
        out.append({"b_min": b, "a": a, "c_min_feu": r["feu"]["c_min"], "c_nom_feu": r["c_nom_feu"], "c_nom": r["c_nom"],
                    "gouvernant": r["gouvernant"], "retenue": b == el["b_min"], "optimale": False})
    if out:
        mini = min(c["c_nom"] for c in out)
        next(c for c in out if c["c_nom"] == mini)["optimale"] = True
    return out
