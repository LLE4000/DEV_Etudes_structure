# -*- coding: utf-8 -*-
"""Fondations de grues — les FICHES DE CALCUL des classeurs du bureau, transcrites MOT POUR MOT.

Chaque ligne ``r.pose(clé, cellule, formule)`` est UNE cellule du classeur : même formule, mêmes opérations dans le
même ordre, mêmes unités (cm, kN, kN·m, mm, N/mm², T), aucun arrondi (les seuls arrondis sont dans la base de données,
``donnees.py``). Rien n'est simplifié ni « amélioré » ici : les écarts trouvés à la relecture
(``docs/beton/FONDATIONS_GRUE_ETUDE.md`` § 5) sont SIGNALÉS par ``ecarts.py``, jamais corrigés dans ce fichier.

Trois fiches, donc trois fonctions — chacune reçoit la fondation NORMALISÉE (``entree.normaliser``) et rend un
``Releve`` : un dict clé → valeur, et ``.cellules`` clé → cellule d'origine (source unique de la correspondance
classeur → module, lue par les tests de fidélité et par la note).

    cadre4(f)       classeur « VARIANTE … 4 pieux »          : 4 pieux, 4 poutres en cadre
    diagonales8(f)  classeur « VARIANTE … 8 pieux »          : 4 poutres primaires sur 2 pieux + 4 secondaires
    semelles8(f)    classeurs « BASE » et « GWW » (Greenwood) : 2 poutres support sur 4 semelles de 2 pieux
                    (ou une paroi de pieux sécants d'un côté) — avec ses grands tableaux et sa feuille de la semelle

Les états sont ceux du classeur, en toutes lettres : « OK », « LIMITE », « PAS OK », « A VERIFIER ».
"""
import math

from . import donnees as D

OK, LIMITE, PAS_OK, A_VERIFIER = "OK", "LIMITE", "PAS OK", "A VERIFIER"
BETON_SEUL, ETRIERS = "Le béton résiste seul", "Prévoir des étriers"
FICHE = "Fiche de calcul"

# --- les grands tableaux de BASE (feuilles « R+V+M support chassis_CAS A.a » … « B.b ») ---
SOUS_CAS = ("A.a", "A.b", "B.a", "B.b")
FEUILLE_SOUS_CAS = {k: f"R+V+M support chassis_CAS {k}" for k in SOUS_CAS}
FEUILLE_MAX = {"A": "R+V+M support chassis_CAS A", "B": "R+V+M support chassis_CAS B"}
FEUILLE_SEMELLE = "M semelle sur pieux_CAS A"
# les 8 orientations d'un cas : (pied côté semelle, pied côté pieux sécants) — indices dans A, B, C, D
ORIENTATIONS = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3))
NOMS_ORIENTATION = ("A/B", "B/A", "B/C", "C/B", "C/D", "D/C", "D/A", "A/D")
# « RB respectif » : la colonne de l'AUTRE poutre support sous le même cas (ligne 29 : =G28, =F28, =I28, =H28, …)
RESPECTIVE = (5, 4, 7, 6, 1, 0, 3, 2)
PAS_X = 0.05                 # m : pas des abscisses des grands tableaux (A36:A246)
N_X = 211                    # 0 … 10,5 m
N_T = 101                    # feuille de la semelle : t = 0 … 1 par 0,01 (A17:A117)
MARGE_LIMITE = 0.95          # « LIMITE » : à 5 % près (hauteur, armatures)
MARGE_PAS = 1.05             # « LIMITE » : pas choisi jusqu'à 5 % au-dessus du pas calculé
MARGES_PIEU = (1.1, 1.05)    # pieu : OK si sollicitation × 1,10 < capacité, LIMITE si × 1,05 < capacité


class Releve(dict):
    """Les valeurs d'une fiche (clé → valeur) et la cellule de chacune (``cellules`` : clé → « C34 » ou
    « feuille!C34 » hors de la fiche de calcul)."""

    def __init__(self):
        super().__init__()
        self.cellules = {}

    def pose(self, cle: str, cellule: str, valeur):
        self[cle] = valeur
        self.cellules[cle] = cellule
        return valeur


# ------------------------------------------------------------------ petites formules partagées, écrites comme le classeur
def _signe(calcule: float, choisi: float) -> str:
    """=SI(OU(C<G;C=G);"≤";"≥")"""
    return "≤" if (calcule < choisi or calcule == choisi) else "≥"


def _racine(x: float):
    """RACINE d'Excel : #NOMBRE! (ici None) pour un nombre négatif."""
    return math.sqrt(x) if x >= 0 else None


def _sup(a, b) -> bool:
    """a > b, faux si l'un des deux n'a pas pu être calculé (#NOMBRE! dans le classeur)."""
    return a is not None and b is not None and a > b


def _etat2(condition: bool) -> str:
    return OK if condition else PAS_OK


def _etat3(choisi, calcule, dernier: str = PAS_OK) -> str:
    """=SI(G>C;"OK";SI(G>0,95*C;"LIMITE";"PAS OK"))"""
    if calcule is None:
        return dernier
    return OK if choisi > calcule else (LIMITE if choisi > MARGE_LIMITE * calcule else dernier)


def _etat3_pas(choisi, calcule) -> str:
    """=SI(G<C;"OK";SI(G<1,05*C;"LIMITE";"PAS OK"))"""
    return OK if choisi < calcule else (LIMITE if choisi < MARGE_PAS * calcule else PAS_OK)


def _texte_tau(tau: float) -> str:
    """=SI(C<0,56;"Le béton résiste seul";"Prévoir des étriers") — 0,56 écrit en dur dans le classeur."""
    return BETON_SEUL if tau < D.TAU_SEUIL_EN_DUR else ETRIERS


def _nombre_fr(x: float) -> str:
    """Un nombre comme Excel l'écrit dans un texte (« 0,56 », « 1,12 ») : 15 chiffres significatifs, virgule."""
    return format(x, ".15g").replace(".", ",")


def _zone_tau(tau: float, rwk: float):
    """Les deux cellules de zone de la fiche BASE : le texte long (G142) et le texte court (G163)."""
    z1, z2, z3 = 0.016 * rwk, 0.032 * rwk, 0.048 * rwk
    if tau < z1:
        return f"Zone 1 = τ < {_nombre_fr(z1)} N/mm²", "ZONE 1"
    if tau < z2:
        return f"Zone 2 = {_nombre_fr(z1)} N/mm² < τ < {_nombre_fr(z2)} N/mm²", "ZONE 2"
    if tau < z3:
        return f"Zone 3 = {_nombre_fr(z2)} N/mm² < τ << {_nombre_fr(z3)} N/mm²", "ZONE 3"
    return f"> Zone 3 =  {_nombre_fr(z3)} N/mm²", "> ZONE 3"


def _materiaux(r: Releve, f: dict, cellules) -> dict:
    m = D.materiaux(f["beton"], f["acier"])
    for cle, cel in zip(("Rwk", "sigma_b", "mu", "sigma_adm"), cellules):
        r.pose(f"materiaux.{cle}", cel, m[cle])
    return m


def _renfort_pieu(r: Releve, P: float, b: float, Dp: float, d: float, m: dict, cel: dict) -> None:
    """« Renforcement au droit du pieu », commun aux trois fiches jusqu'aux moments : la poutre (largeur b, cm) déborde
    du pieu (diamètre Dp, cm) ; p1 = P/b vers le bas sur toute la largeur, p2 = réaction du pieu sur Dp (CAS A) ou sur
    Dp/2 (CAS B) ; moment au droit de l'axe. ``cel`` : les cellules de chaque grandeur."""
    r.pose("renfort.P", cel["P"], P)
    p1 = r.pose("renfort.p1", cel["p1"], P / (b / 100))
    p2a = r.pose("renfort.p2a", cel["p2a"], P / (Dp / 100))
    p2b = r.pose("renfort.p2b", cel["p2b"], P / (Dp / 2 / 100))
    MA = r.pose("renfort.M_A", cel["M_A"], p1 * (b / 2 / 100) ** 2 / 2 - p2a * (Dp / 2 / 100) ** 2 / 2)
    MB = r.pose("renfort.M_B", cel["M_B"], p1 * (b / 2 / 100) ** 2 / 2 - p2b * (Dp / 4 / 100) ** 2 / 2)
    hA = r.pose("renfort.h_calc_A", cel["h_A"], _racine(MA * 10 ** 6 / (m["mu"] * Dp * 10 * m["sigma_b"])))
    hB = r.pose("renfort.h_calc_B", cel["h_B"], _racine(MB * 10 ** 6 / (m["mu"] * Dp * 10 * m["sigma_b"])))
    r.pose("renfort.d", cel["d"], d)
    r.pose("renfort.omega_sup_A", cel["om_A"], MA * 10 ** 6 / (m["sigma_adm"] * 0.9 * d))
    r.pose("renfort.omega_sup_B", cel["om_B"], MB * 10 ** 6 / (m["sigma_adm"] * 0.9 * d))
    if hA is not None:
        r.pose("renfort.signe_h_A", cel["s_h_A"], _signe(hA, d))
    if hB is not None:
        r.pose("renfort.signe_h_B", cel["s_h_B"], _signe(hB, d))


def _horizontal_variantes(r: Releve, f: dict, cel: dict) -> None:
    """« Effort horizontal » des deux variantes : =MAX(Hhs/3;Hs/3+M/2/L/100), L en cm — la formule du classeur telle
    qu'elle est écrite (écart E1 : le terme du moment y est divisé par 100·L(cm) ; voir ``ecarts.py``)."""
    ch, L = f["charges"], f["chassis"]
    Hhs = r.pose("horizontal.H_hs", cel["H_hs"], ch["H_hs"])
    Hs = r.pose("horizontal.H_s", cel["H_s"], ch["H_s"])
    Ms = r.pose("horizontal.M_s", cel["M_s"], ch["M_s"])
    H = r.pose("horizontal.H_pied", cel["H"], max(Hhs / 3, Hs / 3 + Ms / 2 / L / 100))
    om = r.pose("horizontal.omega_perfos", cel["om"], (H * 10 ** 3) / D.TAUX_PERFOS)
    ch_ = r.pose("horizontal.omega_perfos_choisie", cel["om_ch"], D.aire(f["perfos"]["n"], f["perfos"]["d"]))
    r.pose("horizontal.signe", cel["signe"], _signe(om, ch_))
    r.pose("horizontal.perfos_par_plaque", cel["plaque"], f["perfos"]["n"] * 3)
    r.pose("horizontal.d_perfos", cel["d"], f["perfos"]["d"])
    r.pose("etat.perfos", cel["etat"], _etat2(ch_ > om))


# ================================================================== VAR4 — 4 pieux, 4 poutres en cadre
def cadre4(f: dict) -> Releve:
    r = Releve()
    m = _materiaux(r, f, ("G13", "G14", "G15", "G16"))
    C10, C11, C12, C13 = f["chassis"], f["poutre"]["b"], f["poutre"]["h"], f["pieux"]["D"]
    C24, C29 = f["charges"]["P"], f["e"]
    po = f["poutre"]
    C21 = r.pose("pp.poutre", "C21", 25 * C11 / 100 * C12 / 100)

    # --- CAS A : pieu rentré de e → R max, M min ; CAS B : pieu sorti de e → M max
    C34 = r.pose("poutre.cas_A.RA", "C34", (C24 * C10 / 100 + C21 * (C10 / 100) ** 2 / 2) / ((C10 - C29) / 100))
    C35 = r.pose("poutre.cas_A.RB", "C35",
                 (C21 * ((C10 - C29) / 100) ** 2 / 2 - C24 * C29 / 100 - C21 * (C29 / 100) ** 2 / 2) / ((C10 - C29) / 100))
    C37 = r.pose("poutre.cas_A.M_min", "C37", -C24 * C29 / 100 - C21 * (C29 / 100) ** 2 / 2)
    C38 = r.pose("poutre.cas_A.V_max", "C38", C24 + C21 * C29 / 100)
    C39 = r.pose("poutre.cas_A.V_travee", "C39", max(abs(C24 + C21 * 0.8 - C34), abs(C35 - C21 * 0.8)))
    G34 = r.pose("poutre.cas_B.RA", "G34", (C24 * C10 / 100 + C21 * ((C10 + C29) / 100) ** 2 / 2) / ((C10 + C29) / 100))
    G35 = r.pose("poutre.cas_B.RB", "G35", (C24 * C29 / 100 + C21 * ((C10 + C29) / 100) ** 2 / 2) / ((C10 + C29) / 100))
    G37 = r.pose("poutre.cas_B.M_max", "G37", G34 * C29 / 100 - C21 * (C29 / 100) ** 2 / 2)
    G38 = r.pose("poutre.cas_B.V_max", "G38", G34)
    G39 = r.pose("poutre.cas_B.V_travee", "G39", max(abs(G34 - 0.8 * C21 - C24), abs(G35 - 0.8 * C21)))
    C43 = r.pose("poutre.dim.R_max", "C43", max(C34, G34))
    C44 = r.pose("poutre.dim.M_max", "C44", G37)
    C45 = r.pose("poutre.dim.M_min", "C45", C37)
    C46 = r.pose("poutre.dim.V_max", "C46", max(C38, G38))
    C47 = r.pose("poutre.dim.V_travee", "C47", max(C39, G39))

    # --- dimensionnement de la poutre
    G15, G14, G16 = m["mu"], m["sigma_b"], m["sigma_adm"]
    C53 = r.pose("section.poutre.h_calc", "C53", math.sqrt(max(abs(C44), abs(C45)) * 10 ** 6 / (G15 * C11 * 10 * G14)))
    G53 = r.pose("section.poutre.d", "G53", C12 * 10 - 50)
    r.pose("section.poutre.signe_hauteur", "E53", _signe(C53, G53))
    C55 = r.pose("section.poutre.omega_inf", "C55", C44 * 10 ** 6 / (G16 * 0.9 * (G53)))
    G55 = r.pose("section.poutre.omega_inf_choisie", "G55", D.aire(po["inf"]["n"], po["inf"]["d"]))
    r.pose("section.poutre.signe_inf", "E55", _signe(C55, G55))
    C59 = r.pose("section.poutre.omega_sup", "C59", abs(C45) * 10 ** 6 / (G16 * 0.9 * (G53 - 50)))
    G59 = r.pose("section.poutre.omega_sup_choisie", "G59", D.aire(po["sup"]["n"], po["sup"]["d"]))
    r.pose("section.poutre.signe_sup", "E59", _signe(C59, G59))
    G64 = po["part_V"]
    C63 = r.pose("section.poutre.tau", "C63", (C46 * G64 / 100) * 10 ** 3 / (0.75 * C11 * 10 * G53))
    r.pose("section.poutre.tau_texte", "F63", _texte_tau(C63))
    ea, et = po["etriers_appui"], po["etriers_travee"]
    C66 = r.pose("section.poutre.pas_appui", "C66",
                 ea["brins"] * math.pi * (ea["d"] / 2) ** 2 * G16 * G53 * 0.9 / (C46 * 10 ** 3))
    r.pose("section.poutre.signe_pas_appui", "E66", _signe(C66, ea["pas"]))
    C70 = r.pose("section.poutre.pas_travee", "C70",
                 et["brins"] * math.pi * (et["d"] / 2) ** 2 * G16 * G53 * 0.9 / (C47 * 10 ** 3))
    r.pose("section.poutre.signe_pas_travee", "E70", _signe(C70, et["pas"]))
    # le récapitulatif de la fiche (copies des choix) et les états
    r.pose("recap.poutre.d", "D75", G53)
    r.pose("etat.poutre.hauteur", "G75", _etat2(G53 > C53))
    r.pose("recap.poutre.inf_n", "D77", po["inf"]["n"])
    r.pose("recap.poutre.inf_d", "F77", po["inf"]["d"])
    r.pose("etat.poutre.inf", "G77", _etat2(G55 > C55))
    r.pose("recap.poutre.sup_n", "D78", po["sup"]["n"])
    r.pose("recap.poutre.sup_d", "F78", po["sup"]["d"])
    r.pose("etat.poutre.sup", "G78", _etat2(G59 > C59))
    r.pose("recap.poutre.etriers_appui_brins", "C79", ea["brins"])
    r.pose("recap.poutre.etriers_appui_d", "D79", ea["d"])
    r.pose("recap.poutre.etriers_appui_pas", "F79", ea["pas"])
    r.pose("etat.poutre.etriers_appui", "G79", _etat2(ea["pas"] < C66))
    r.pose("recap.poutre.etriers_travee_brins", "C80", et["brins"])
    r.pose("recap.poutre.etriers_travee_d", "D80", et["d"])
    r.pose("recap.poutre.etriers_travee_pas", "F80", et["pas"])
    r.pose("etat.poutre.etriers_travee", "G80", _etat2(et["pas"] < C70))

    # --- capacité portante des pieux : R max + poids propre de la deuxième poutre
    G84 = f["pieux"]["capacite"]
    C84 = r.pose("pieux.charge", "C84", C43 + C10 / 100 / 2 * C21)
    r.pose("pieux.signe", "E84", _signe(C84, G84))
    r.pose("pieux.capacite_T", "D88", G84 / 10)
    r.pose("etat.pieux", "G88", _etat2(G84 > C84))

    # --- renforcement au droit du pieu
    r.pose("renfort.D", "C91", C13)
    G105 = C12 * 10 - 75
    _renfort_pieu(r, C46, C11, C13, G105, m,
                  {"P": "C94", "p1": "C95", "p2a": "C96", "p2b": "C97", "M_A": "C101", "M_B": "C102", "h_A": "C105",
                   "h_B": "C106", "d": "G105", "om_A": "C108", "om_B": "C109", "s_h_A": "E105", "s_h_B": "E106"})
    G108 = r.pose("renfort.omega_brins_horizontaux", "G108",
                  math.pi * (ea["d"] / 2) ** 2 * ea["brins"] / 2 * (C13 / 2 * 10 / ea["pas"]))
    r.pose("renfort.signe_sup_A", "E108", _signe(r["renfort.omega_sup_A"], G108))
    r.pose("renfort.signe_sup_B", "E109", _signe(r["renfort.omega_sup_B"], G108))
    C112 = r.pose("renfort.tau", "C112", (C46) * 10 ** 3 / (0.75 * C11 * 10 * G105))
    r.pose("renfort.tau_texte", "F112", _texte_tau(C112))
    C114 = r.pose("renfort.omega_etriers", "C114", C46 * 10 ** 3 / G16)
    G114 = r.pose("renfort.omega_etriers_choisie", "G114", D.aire(f["renfort"]["brins"], f["renfort"]["d"]))
    r.pose("renfort.signe_etriers", "E114", _signe(C114, G114))
    r.pose("recap.renfort.d", "D120", G105)
    r.pose("etat.renfort.hauteur", "G120", _etat2(_sup(G105, r["renfort.h_calc_A"]) and _sup(G105, r["renfort.h_calc_B"])))
    e_sup = r.pose("etat.renfort.sup", "G122",
                   _etat2(G108 > r["renfort.omega_sup_A"] and G108 > r["renfort.omega_sup_B"]))
    r.pose("renfort.remarque_sup", "F122",
           "Vérifier si les étriers complémentaires au droit des pieux permettent de reprendre l'effort"
           if e_sup == PAS_OK else None)
    r.pose("recap.renfort.brins", "D123", f["renfort"]["brins"])
    r.pose("recap.renfort.d_brins", "F123", f["renfort"]["d"])
    r.pose("etat.renfort.etriers", "G123", _etat2(G114 > C114))

    # --- effort horizontal
    _horizontal_variantes(r, f, {"H_hs": "C128", "H_s": "G128", "M_s": "G129", "H": "C130", "om": "C133",
                                 "om_ch": "G133", "signe": "E133", "plaque": "D139", "d": "F139", "etat": "G139"})
    return r


# ================================================================== VAR8 — 8 pieux, primaires sur 2 pieux + secondaires
def diagonales8(f: dict) -> Releve:
    r = Releve()
    m = _materiaux(r, f, ("G13", "G14", "G15", "G16"))
    G15, G14, G16 = m["mu"], m["sigma_b"], m["sigma_adm"]
    se, pr = f["secondaire"], f["primaire"]
    C10, C11, C12, C13, C14, C15 = f["chassis"], se["b"], se["h"], pr["b"], pr["h"], pr["L"]
    C16, C17 = f["pieux"]["entraxe"], f["pieux"]["D"]
    C26, C31, C49 = f["charges"]["P"], f["e_secondaire"], f["e_primaire"]
    C22 = r.pose("pp.secondaire", "C22", 25 * C11 / 100 * C12 / 100)
    C23 = r.pose("pp.primaire", "C23", 25 * C13 / 100 * C14 / 100)

    # --- poutre secondaire : les formules de la poutre de la variante 4 pieux
    C36 = r.pose("secondaire.cas_A.RA", "C36", (C26 * C10 / 100 + C22 * (C10 / 100) ** 2 / 2) / ((C10 - C31) / 100))
    r.pose("secondaire.cas_A.RB", "C37",
           (C22 * ((C10 - C31) / 100) ** 2 / 2 - C26 * C31 / 100 - C22 * (C31 / 100) ** 2 / 2) / ((C10 - C31) / 100))
    C39 = r.pose("secondaire.cas_A.M_min", "C39", -C26 * C31 / 100 - C22 * (C31 / 100) ** 2 / 2)
    C40 = r.pose("secondaire.cas_A.V_max", "C40", C26 + C22 * C31 / 100)
    G36 = r.pose("secondaire.cas_B.RA", "G36", (C26 * C10 / 100 + C22 * ((C10 + C31) / 100) ** 2 / 2) / ((C10 + C31) / 100))
    r.pose("secondaire.cas_B.RB", "G37", (C26 * C31 / 100 + C22 * ((C10 + C31) / 100) ** 2 / 2) / ((C10 + C31) / 100))
    G39 = r.pose("secondaire.cas_B.M_max", "G39", G36 * C31 / 100 - C22 * (C31 / 100) ** 2 / 2)
    G40 = r.pose("secondaire.cas_B.V_max", "G40", G36)
    C43 = r.pose("secondaire.dim.R_max", "C43", max(C36, G36))
    C44 = r.pose("secondaire.dim.M_max", "C44", G39)
    C45 = r.pose("secondaire.dim.M_min", "C45", C39)
    C46 = r.pose("secondaire.dim.V_max", "C46", max(C40, G40))

    # --- poutre primaire : charge = R max secondaire + poids propre de la deuxième poutre secondaire
    C50 = r.pose("primaire.charge", "C50", C43 + C22 * C10 / 100 / 2)
    C55 = r.pose("primaire.cas_A.RA", "C55",
                 C50 * ((C16 - C49) / 2 + C49 / 2) / 100 / ((C16 - C49) / 100) + C23 * (C15 / 100) / 2)
    r.pose("primaire.cas_A.RB", "C56",
           C50 * ((C16 - C49) / 2 - C49 / 2) / 100 / ((C16 - C49) / 100) + C23 * (C15 / 100) / 2)
    r.pose("primaire.cas_A.M_max", "C58",
           C55 * ((C16 - C49) / 2 - C49 / 2) / 100 - C23 * ((C15 / 2 - C49 / 2) / 100) ** 2 / 2)
    C59 = r.pose("primaire.cas_A.V_max", "C59", C55 - C23 * (C15 - (C16 - C49)) / 2 / 100)
    G55 = r.pose("primaire.cas_B.RA", "G55", C50 / 2 + C23 * (C15 / 100) / 2)
    r.pose("primaire.cas_B.RB", "G56", C50 / 2 + C23 * (C15 / 100) / 2)
    G58 = r.pose("primaire.cas_B.M_max", "G58", G55 * (C16 / 2 + C49 / 2) / 100 - C23 * ((C15 / 2) / 100) ** 2 / 2)
    G59 = r.pose("primaire.cas_B.V_max", "G59", G55 - C23 * (C15 - C16 - C49) / 2 / 100)
    r.pose("primaire.dim.R_max", "C62", max(C55, G55))
    C63 = r.pose("primaire.dim.M_max", "C63", G58)
    C64 = r.pose("primaire.dim.V_max", "C64", max(C59, G59))

    # --- dimensionnement de la poutre secondaire
    C68 = r.pose("section.secondaire.h_calc", "C68", math.sqrt(max(abs(C44), abs(C45)) * 10 ** 6 / (G15 * C11 * 10 * G14)))
    G68 = r.pose("section.secondaire.d", "G68", C12 * 10 - 50)
    r.pose("section.secondaire.signe_hauteur", "E68", _signe(C68, G68))
    C70 = r.pose("section.secondaire.omega_inf", "C70", C44 * 10 ** 6 / (G16 * 0.9 * (G68)))
    G70 = r.pose("section.secondaire.omega_inf_choisie", "G70", D.aire(se["inf"]["n"], se["inf"]["d"]))
    r.pose("section.secondaire.signe_inf", "E70", _signe(C70, G70))
    C74 = r.pose("section.secondaire.omega_sup", "C74", abs(C45) * 10 ** 6 / (G16 * 0.9 * (G68 - 50)))
    G74 = r.pose("section.secondaire.omega_sup_choisie", "G74", D.aire(se["sup"]["n"], se["sup"]["d"]))
    r.pose("section.secondaire.signe_sup", "E74", _signe(C74, G74))
    G79 = se["part_V"]
    C78 = r.pose("section.secondaire.tau", "C78", (C46 * G79 / 100) * 10 ** 3 / (0.75 * C11 * 10 * G68))
    r.pose("section.secondaire.tau_texte", "F78", _texte_tau(C78))
    es = se["etriers"]
    C81 = r.pose("section.secondaire.pas", "C81",
                 es["brins"] * math.pi * (es["d"] / 2) ** 2 * G16 * G68 * 0.9 / (C46 * 10 ** 3 * G79 / 100))
    r.pose("section.secondaire.signe_pas", "E81", _signe(C81, es["pas"]))
    r.pose("recap.secondaire.d", "D86", G68)
    r.pose("etat.secondaire.hauteur", "G86", _etat2(G68 > C68))
    r.pose("recap.secondaire.inf_n", "D88", se["inf"]["n"])
    r.pose("recap.secondaire.inf_d", "F88", se["inf"]["d"])
    r.pose("etat.secondaire.inf", "G88", _etat2(G70 > C70))
    r.pose("recap.secondaire.sup_n", "D89", se["sup"]["n"])
    r.pose("recap.secondaire.sup_d", "F89", se["sup"]["d"])
    r.pose("etat.secondaire.sup", "G89", _etat2(G74 > C74))
    r.pose("recap.secondaire.etriers_brins", "C90", es["brins"])
    r.pose("recap.secondaire.etriers_d", "D90", es["d"])
    r.pose("recap.secondaire.etriers_pas", "F90", es["pas"])
    r.pose("etat.secondaire.etriers", "G90", _etat2(es["pas"] < C81))

    # --- dimensionnement de la poutre primaire
    C93 = r.pose("section.primaire.h_calc", "C93", math.sqrt(C63 * 10 ** 6 / (G15 * C13 * 10 * G14)))
    G93 = r.pose("section.primaire.d", "G93", C14 * 10 - 75)
    r.pose("section.primaire.signe_hauteur", "E93", _signe(C93, G93))
    C95 = r.pose("section.primaire.omega_inf", "C95", C63 * 10 ** 6 / (G16 * 0.9 * G93))
    G95 = r.pose("section.primaire.omega_inf_choisie", "G95", D.aire(pr["inf"]["n"], pr["inf"]["d"]))
    r.pose("section.primaire.signe_inf", "E95", _signe(C95, G95))
    C99 = r.pose("section.primaire.omega_sup", "C99", 0.4 * C95)
    G99 = r.pose("section.primaire.omega_sup_choisie", "G99", D.aire(pr["sup"]["n"], pr["sup"]["d"]))
    r.pose("section.primaire.signe_sup", "E99", _signe(C99, G99))
    C103 = r.pose("section.primaire.tau", "C103", (C64) * 10 ** 3 / (0.75 * C13 * 10 * G93))
    r.pose("section.primaire.tau_texte", "F103", _texte_tau(C103))
    ep = pr["etriers"]
    C105 = r.pose("section.primaire.pas", "C105",
                  ep["brins"] * math.pi * (ep["d"] / 2) ** 2 * G16 * G93 * 0.9 / (C64 * 10 ** 3))
    r.pose("section.primaire.signe_pas", "E105", _signe(C105, ep["pas"]))
    r.pose("recap.primaire.d", "D110", G93)
    r.pose("etat.primaire.hauteur", "G110", _etat2(G93 > C93))
    r.pose("recap.primaire.inf_n", "D112", pr["inf"]["n"])
    r.pose("recap.primaire.inf_d", "F112", pr["inf"]["d"])
    r.pose("etat.primaire.inf", "G112", _etat2(G95 > C95))
    r.pose("recap.primaire.sup_n", "D113", pr["sup"]["n"])
    r.pose("recap.primaire.sup_d", "F113", pr["sup"]["d"])
    r.pose("etat.primaire.sup", "G113", _etat2(G99 > C99))
    r.pose("recap.primaire.etriers_brins", "C114", ep["brins"])
    r.pose("recap.primaire.etriers_d", "D114", ep["d"])
    r.pose("recap.primaire.etriers_pas", "F114", ep["pas"])
    r.pose("etat.primaire.etriers", "G114", _etat2(ep["pas"] < C105))

    # --- renforcement au droit du pieu (sous la poutre primaire)
    r.pose("renfort.D", "C116", C17)
    G130 = C14 * 10 - 75
    _renfort_pieu(r, C64, C13, C17, G130, m,
                  {"P": "C119", "p1": "C120", "p2a": "C121", "p2b": "C122", "M_A": "C126", "M_B": "C127", "h_A": "C130",
                   "h_B": "C131", "d": "G130", "om_A": "C133", "om_B": "C134", "s_h_A": "E130", "s_h_B": "E131"})
    G133 = r.pose("renfort.omega_brins_horizontaux", "G133",
                  math.pi * (ep["d"] / 2) ** 2 * ep["brins"] / 2 * (C17 / 2 * 10 / ep["pas"]))
    r.pose("renfort.signe_sup_A", "E133", _signe(r["renfort.omega_sup_A"], G133))
    r.pose("renfort.signe_sup_B", "E134", _signe(r["renfort.omega_sup_B"], G133))
    C137 = r.pose("renfort.tau", "C137", (C64) * 10 ** 3 / (0.75 * C13 * 10 * G130))
    r.pose("renfort.tau_texte", "F137", _texte_tau(C137))
    C139 = r.pose("renfort.omega_etriers", "C139", C64 * 10 ** 3 / G16)
    G139 = r.pose("renfort.omega_etriers_choisie", "G139", D.aire(f["renfort"]["brins"], f["renfort"]["d"]))
    r.pose("renfort.signe_etriers", "E139", _signe(C139, G139))
    r.pose("recap.renfort.d", "D145", G130)
    r.pose("etat.renfort.hauteur", "G145", _etat2(_sup(G130, r["renfort.h_calc_A"]) and _sup(G130, r["renfort.h_calc_B"])))
    e_sup = r.pose("etat.renfort.sup", "G147",
                   _etat2(G133 > r["renfort.omega_sup_A"] and G133 > r["renfort.omega_sup_B"]))
    r.pose("renfort.remarque_sup", "F147",
           "Vérifier si les étriers complémentaires au droit des pieux permettent de reprendre l'effort"
           if e_sup == PAS_OK else None)
    r.pose("recap.renfort.brins", "D148", f["renfort"]["brins"])
    r.pose("recap.renfort.d_brins", "F148", f["renfort"]["d"])
    r.pose("etat.renfort.etriers", "G148", _etat2(G139 > C139))

    # --- effort horizontal
    _horizontal_variantes(r, f, {"H_hs": "C153", "H_s": "G153", "M_s": "G154", "H": "C155", "om": "C158",
                                 "om_ch": "G158", "signe": "E158", "plaque": "D164", "d": "F164", "etat": "G164"})
    return r


# ================================================================== BASE / GWW — 2 poutres support sur 4 semelles
def geometrie_sous_cas(f: dict) -> dict:
    """a' et b' (m) de chaque sous-cas de tolérance — cellules B6 et B7 des quatre grands tableaux."""
    a, b, e = f["dist_a"], f["dist_b"], f["e_support"]
    return {"A.a": ((a + e) / 100, (b) / 100),          # distance a + excentricité ; distance b
            "A.b": ((a) / 100, (b + e) / 100),          # distance a ; distance b + excentricité
            "B.a": ((a + e) / 100, (b - e) / 100),      # châssis décalé de e
            "B.b": ((a - e) / 100, (b + e) / 100)}


def colonnes_du_tableau(f: dict):
    """Les colonnes d'un grand tableau, dans l'ordre de la feuille : un cas de la grue × 8 orientations.
    Rend [(numéro du cas, orientation, P1 côté semelle, P2 côté pieux sécants, H par platine)]."""
    ch = f["charges"]
    cols = []
    for i, cas in enumerate(ch["cas"]):
        H4 = (ch["H_hs"] if cas["etat"] == "hors" else ch["H_s"]) / 4
        pieds = (cas["A"], cas["B"], cas["C"], cas["D"])
        for o, (p1, p2) in enumerate(ORIENTATIONS):
            cols.append((i + 1, o, pieds[p1], pieds[p2], H4))
    return cols


def _lettres(j: int) -> str:
    """La colonne Excel d'une colonne de tableau (0 → B, 47 → AW)."""
    n, s = j + 2, ""
    while n:
        n, reste = divmod(n - 1, 26)
        s = chr(65 + reste) + s
    return s


def _premier(valeurs, cible):
    """EQUIV(cible ; plage ; 0) : l'indice de la première valeur exactement égale, None si aucune (#N/A)."""
    for j, v in enumerate(valeurs):
        if v == cible:
            return j
    return None


def tableau_support(f: dict, sous_cas: str, pp: float) -> dict:
    """UN grand tableau (« R+V+M support chassis_CAS A.a »…) : la poutre support, portée L = Lc + a' + b', posée sur ses
    deux appuis d'extrémité, les deux pieds de la grue en a' et a' + Lc. Rend les données, les 48 colonnes et les
    maximums avec leur cas, comme la feuille."""
    B5 = f["chassis"] / 100
    B6, B7 = geometrie_sous_cas(f)[sous_cas]
    B8 = (B5 + B6 + B7)
    B9 = pp
    suffixe = sous_cas[-1]                                   # « a » ou « b » : CAS 1.a, CAS 1.b
    hb = (f["support"]["h"] / 100) / (f["support"]["b"] / 100)
    cols = []
    for (cas, o, P1, P2, H4) in colonnes_du_tableau(f):
        RA = (P1 * (B5 + B7) + P2 * B7) / B8 + B9 * B8 / 2
        RB = (P2 * (B5 + B6) + P1 * B6) / B8 + B9 * B8 / 2
        Mhor = max(((B7 + B5) + B7) * B6 * H4 / B8, ((B6 + B5) + B6) * B7 * H4 / B8) * hb * 100 / 125
        M = []
        for i in range(N_X):
            x = i / 20
            if x > B8:
                M.append(0)
            elif x > (B6 + B5):
                M.append(RA * x - B9 * x ** 2 / 2 - P1 * (x - B6) - P2 * (x - B6 - B5))
            elif x > B6:
                M.append(RA * x - B9 * x ** 2 / 2 - P1 * (x - B6))
            else:
                M.append(RA * x - B9 * x ** 2 / 2)
        Mmax = max(M)
        cols.append({"cas": cas, "orientation": NOMS_ORIENTATION[o],
                     "libelle": f"(CAS {cas}.{suffixe}-{NOMS_ORIENTATION[o]})",
                     "P1": P1, "P2": P2, "H4": H4, "RA": RA, "RB": RB,
                     "V_max": max(RA, RB), "V_central": max(RA - B9 * B6 - P1, RB - B9 * B7 - P2),
                     "M_hor_eq": Mhor, "M_max": Mmax, "M_max_eq": Mhor + Mmax, "M": M})
    for j, c in enumerate(cols):                             # ligne 29 : le RB de l'autre poutre, même cas
        groupe = (j // 8) * 8
        c["RB_resp"] = cols[groupe + RESPECTIVE[j % 8]]["RB"]
        c["symetrique"] = c["RB"] == c["RB_resp"]
    lib = [c["libelle"] for c in cols]

    def maxi(cle):
        v = max(c[cle] for c in cols)
        j = _premier([c[cle] for c in cols], v)
        return {"valeur": v, "colonne": j, "libelle": lib[j]}

    RB_max = maxi("RB")
    j_resp = _premier([c["RB_resp"] for c in cols], cols[RB_max["colonne"]]["RB_resp"])
    sym = [c["RB"] for c in cols if c["symetrique"]]
    RB_symm = max(sym) if sym else 0                          # MAX(SI(…="SYMM.";…)) : 0 sans colonne symétrique
    j_symm = _premier([c["RB"] for c in cols], RB_symm)
    return {"sous_cas": sous_cas, "feuille": FEUILLE_SOUS_CAS[sous_cas],
            "geometrie": {"Lc": B5, "a": B6, "b": B7, "L": B8, "pp": B9},
            "colonnes": cols,
            "max": {"RA": maxi("RA"), "RB": RB_max,
                    "RB_resp": {"valeur": cols[RB_max["colonne"]]["RB_resp"], "colonne": j_resp, "libelle": lib[j_resp]},
                    "RB_symm": {"valeur": RB_symm, "colonne": j_symm, "libelle": None if j_symm is None else lib[j_symm]},
                    "V_max": maxi("V_max"), "V_central": maxi("V_central"),
                    "M_max": maxi("M_max"), "M_max_eq": maxi("M_max_eq")}}


# les lignes des feuilles : sous-cas (B12:B19), maximums CAS A / CAS B (B5:B12), fiche (lignes 79 à 87)
GRANDEURS_SUPPORT = ("RA", "RB", "RB_resp", "RB_symm", "V_max", "V_central", "M_max", "M_max_eq")
NOMS_GRANDEUR = {"RA": "RA,max", "RB": "RB,max", "RB_resp": "RB,resp. max", "RB_symm": "RB,symm", "V_max": "V,max",
                 "V_central": "V,central", "M_max": "M,max", "M_max_eq": "M,max,eq"}
LIGNE_FICHE = {"RA": 79, "RB": 80, "RB_resp": 81, "RB_symm": 82, "M_max": 84, "M_max_eq": 85, "V_max": 86, "V_central": 87}


def _maximum_de_deux(g: str, t1: dict, t2: dict) -> dict:
    """Une ligne des feuilles « …_CAS A » / « …_CAS B » : le maximum des deux sous-cas et son libellé (le premier
    des deux à égalité, comme EQUIV)."""
    v1, v2 = t1["max"][g]["valeur"], t2["max"][g]["valeur"]
    if g == "RB_resp":                                       # =INDEX(B19:C23;5;EQUIV(RB,max;B20:C20;0))
        r1, r2 = t1["max"]["RB"]["valeur"], t2["max"]["RB"]["valeur"]
        t = t1 if r1 == max(r1, r2) else t2
        v = t["max"]["RB_resp"]["valeur"]
        src = t1 if v1 == v else t2                          # libellé : =INDEX(B22:C23;1;EQUIV(B7;B23:C23;0))
    else:
        v = max(v1, v2)
        src = t1 if v1 == v else t2
    lib = src["max"][g]["libelle"]
    return {"valeur": v, "sous_cas": src["sous_cas"], "libelle": lib,
            "texte": None if lib is None else f"{NOMS_GRANDEUR[g]} {lib}"}


def semelles8(f: dict) -> Releve:
    r = Releve()
    m = _materiaux(r, f, ("G12", "G13", "G14", "G15"))
    G12, G13, G14, G15 = m["Rwk"], m["sigma_b"], m["mu"], m["sigma_adm"]
    su, sm, pi_ = f["support"], f["semelle"], f["pieux"]
    C15 = f["enrobage"]
    C20, C21 = su["h"], su["b"]
    C24, C25, C26 = sm["L"], sm["h"], sm["b"]
    C29, C30, C31 = pi_["entraxe"], pi_["D"], pi_["capacite"]
    C34, C35, C37 = f["chassis"], f["dist_a"], f["dist_b"]
    r.pose("geometrie.longueur_support", "C19", C34 + C35 + C37)
    C43 = r.pose("pp.support", "C43", 25 * C21 / 100 * C20 / 100)
    C44 = r.pose("pp.semelle", "C44", 25 * C26 / 100 * C25 / 100)

    # --- poutre support châssis : les quatre grands tableaux, puis CAS A = max(A.a ; A.b), CAS B = max(B.a ; B.b)
    tableaux = {k: tableau_support(f, k, C43) for k in SOUS_CAS}
    r.tableaux = tableaux
    for k, t in tableaux.items():
        feuille = t["feuille"]
        for cle, cel in (("Lc", "B5"), ("a", "B6"), ("b", "B7"), ("L", "B8"), ("pp", "B9")):
            r.pose(f"support.sous_cas.{k}.{cle}", f"{feuille}!{cel}", t["geometrie"][cle])
        for i, g in enumerate(GRANDEURS_SUPPORT):
            r.pose(f"support.sous_cas.{k}.{g}", f"{feuille}!B{12 + i}", t["max"][g]["valeur"])
            lib = t["max"][g]["libelle"]
            r.pose(f"support.sous_cas.{k}.{g}.cas", f"{feuille}!D{12 + i}",
                   None if lib is None else f"{NOMS_GRANDEUR[g]} {lib}")
    cas = {}
    for lettre, (k1, k2) in (("A", ("A.a", "A.b")), ("B", ("B.a", "B.b"))):
        cas[lettre] = {g: _maximum_de_deux(g, tableaux[k1], tableaux[k2]) for g in GRANDEURS_SUPPORT}
        col_fiche = "C" if lettre == "A" else "G"
        for i, g in enumerate(GRANDEURS_SUPPORT):
            r.pose(f"support.cas_{lettre}.{g}", f"{col_fiche}{LIGNE_FICHE[g]}", cas[lettre][g]["valeur"])
            r.pose(f"support.cas_{lettre}.{g}.cas", f"{'B' if lettre == 'A' else 'F'}{LIGNE_FICHE[g]}",
                   cas[lettre][g]["texte"])
            r.pose(f"support.cas_{lettre}.{g}.feuille", f"{FEUILLE_MAX[lettre]}!B{5 + i}", cas[lettre][g]["valeur"])
            r.pose(f"support.cas_{lettre}.{g}.feuille_cas", f"{FEUILLE_MAX[lettre]}!D{5 + i}", cas[lettre][g]["texte"])
    r.cas_support = cas
    A, B = cas["A"], cas["B"]
    C90 = r.pose("support.dim.RA_max", "C90", max(A["RA"]["valeur"], B["RA"]["valeur"]))
    C91 = r.pose("support.dim.RB_max", "C91", max(A["RB"]["valeur"], B["RB"]["valeur"]))
    # =INDEX(B80:G81;2;EQUIV(C91;B80:G80;0)) : le RB respectif du CAS (A d'abord) où RB,max est atteint
    r.pose("support.dim.RB_resp", "C92", A["RB_resp"]["valeur"] if A["RB"]["valeur"] == C91 else B["RB_resp"]["valeur"])
    r.pose("support.dim.RB_symm", "C93", max(A["RB_symm"]["valeur"], B["RB_symm"]["valeur"]))
    C94 = r.pose("support.dim.M_max", "C94", max(A["M_max"]["valeur"], B["M_max"]["valeur"]))
    C95 = r.pose("support.dim.M_max_eq", "C95", max(A["M_max_eq"]["valeur"], B["M_max_eq"]["valeur"]))
    C96 = r.pose("support.dim.V_max", "C96", max(A["V_max"]["valeur"], B["V_max"]["valeur"]))
    C97 = r.pose("support.dim.V_central", "C97", max(A["V_central"]["valeur"], B["V_central"]["valeur"]))

    # --- semelle sur pieux : charge = RA,max de la poutre support, répartie sur c
    C101, C102 = f["e_semelle"], f["c_appui"]
    C104 = r.pose("semelle.charge", "C104", C90)
    C112 = r.pose("semelle.cas_A.RA", "C112", C104 * (C29 / 2) / 100 / ((C29 - C101) / 100) + C44 * (C24 / 100) / 2)
    r.pose("semelle.cas_A.RB", "C113", C104 * (C29 / 2 - C101) / 100 / ((C29 - C101) / 100) + C44 * (C24 / 100) / 2)
    # la feuille « M semelle sur pieux_CAS A » : M(t) sur la zone chargée, t = 0 … 1 par 0,01
    sB5 = r.pose("semelle.cas_A.feuille.a", f"{FEUILLE_SEMELLE}!B5", C24 / 100 / 2 - C101 / 100 / 2 - C102 / 100 / 2)
    sB6 = r.pose("semelle.cas_A.feuille.b", f"{FEUILLE_SEMELLE}!B6", C29 / 2 / 100 - C101 / 100 - C102 / 100 / 2)
    sB7 = r.pose("semelle.cas_A.feuille.c", f"{FEUILLE_SEMELLE}!B7", C102 / 100)
    sB8 = r.pose("semelle.cas_A.feuille.pp", f"{FEUILLE_SEMELLE}!B8", C44)
    sB9 = r.pose("semelle.cas_A.feuille.P", f"{FEUILLE_SEMELLE}!B9", C104)
    sB10 = r.pose("semelle.cas_A.feuille.RA", f"{FEUILLE_SEMELLE}!B10", C112)
    Mt = [(sB10 * (sB6 + (i / 100) * sB7) - sB8 * (sB5 + (i / 100) * sB7) ** 2 / 2 - sB9 * (i / 100) ** 2 * sB7 / 2)
          for i in range(N_T)]
    r.M_semelle = Mt
    r.pose("semelle.cas_A.feuille.M_max", f"{FEUILLE_SEMELLE}!B16", max(Mt))
    r.pose("semelle.cas_A.feuille.M_max_resultat", f"{FEUILLE_SEMELLE}!B13", max(Mt))
    r.pose("semelle.cas_A.M_max", "C115", max(Mt))
    C116 = r.pose("semelle.cas_A.V_max", "C116", C112 - C44 * (C24 - (C29 - C101)) / 2 / 100)
    G112 = r.pose("semelle.cas_B.RA", "G112", C104 / 2 + C44 * (C24 / 100) / 2)
    r.pose("semelle.cas_B.RB", "G113", C104 / 2 + C44 * (C24 / 100) / 2)
    G115 = r.pose("semelle.cas_B.M_max", "G115",
                  G112 * (C29 / 2 + C101 / 2) / 100 - C44 * ((C24 / 2) / 100) ** 2 / 2 - C104 * C102 / 100 / 8)
    G116 = r.pose("semelle.cas_B.V_max", "G116", G112 - C44 * (C24 - C29 - C101) / 2 / 100)
    r.pose("semelle.dim.R_max", "C119", max(C112, G112))
    C120 = r.pose("semelle.dim.M_max", "C120", G115)
    C121 = r.pose("semelle.dim.V_max", "C121", max(C116, G116))

    # --- dimensionnement de la poutre support châssis
    inf, sup = su["inf"], su["sup"]
    ee, ec = su["etriers_extremites"], su["etriers_centre"]
    G129, G130, G131, G132 = inf[0]["n"], inf[0]["d"], inf[1]["n"], inf[1]["d"]
    G136, G137, G138, G139 = sup[0]["n"], sup[0]["d"], sup[1]["n"], sup[1]["d"]
    G143, G145, G146, G147 = su["part_V"], ee["pas"], ee["d"], ee["brins"]
    G152, G153, G154 = ec["pas"], ec["d"], ec["brins"]
    C126 = r.pose("section.support.h_calc", "C126", math.sqrt(C94 * 10 ** 6 / (G14 * C21 * 10 * G13)))
    G126 = r.pose("section.support.d", "G126",
                  C20 * 10 - C15 - max(G146, G153) - max(G130 if G129 > 0 else 0, G132 if G131 > 0 else 0) / 2)
    r.pose("section.support.signe_hauteur", "E126", _signe(C126, G126))
    C128 = r.pose("section.support.omega_inf", "C128", C95 * 10 ** 6 / (G15 * 0.9 * (G126)))
    G128 = r.pose("section.support.omega_inf_choisie", "G128", math.pi * (G129 * (G130 / 2) ** 2 + G131 * (G132 / 2) ** 2))
    r.pose("section.support.signe_inf", "E128", _signe(C128, G128))
    C135 = r.pose("section.support.omega_sup", "C135", 0.25 * C128)
    G135 = r.pose("section.support.omega_sup_choisie", "G135", math.pi * (G136 * (G137 / 2) ** 2 + G138 * (G139 / 2) ** 2))
    r.pose("section.support.signe_sup", "E135", _signe(C135, G135))
    C142 = r.pose("section.support.tau", "C142", (C96 * G143 / 100) * 10 ** 3 / (0.75 * C21 * 10 * G126))
    long_, court = _zone_tau(C142, G12)
    r.pose("section.support.tau_zone", "G142", long_)
    G149 = r.pose("section.support.hauteur_etrier_extremites", "G149", C20 * 10 - C15 * 2 - G146)
    C145 = r.pose("section.support.pas_extremites", "C145",
                  G147 * math.pi * (G146 / 2) ** 2 * G15 * (G149) / (C96 * 10 ** 3 * G143 / 100))
    r.pose("section.support.signe_pas_extremites", "E145", _signe(C145, G145))
    C152 = r.pose("section.support.pas_centre", "C152",
                  G154 * math.pi * (G153 / 2) ** 2 * G15 * (G126 - C15) / (C97 * 10 ** 3 * G143 / 100))
    r.pose("section.support.signe_pas_centre", "E152", _signe(C152, G152))
    r.pose("section.support.hauteur_etrier_centre", "G156", C20 * 10 - C15 * 2 - G153)
    r.pose("recap.support.d", "D161", G126)
    r.pose("etat.support.hauteur", "G161", _etat3(G126, C126))
    r.pose("etat.support.tau_zone", "G163", court)
    r.pose("recap.support.inf_1", "D165", f"{G129}Ф{G130}")
    r.pose("recap.support.inf_2", "F165", f"{G131}Ф{G132}")
    r.pose("etat.support.inf", "G165", _etat3(G128, C128))
    r.pose("recap.support.sup_1", "D166", f"{G136}Ф{G137}")
    r.pose("recap.support.sup_2", "F166", f"{G138}Ф{G139}")
    r.pose("etat.support.sup", "G166", _etat3(G135, C135))
    r.pose("recap.support.etriers_extremites_brins", "C167", G147)
    r.pose("recap.support.etriers_extremites_d", "D167", G146)
    r.pose("recap.support.etriers_extremites_pas", "F167", G145)
    r.pose("etat.support.etriers_extremites", "G167", _etat3_pas(G145, C145))
    r.pose("recap.support.etriers_centre_brins", "C168", G154)
    r.pose("recap.support.etriers_centre_d", "D168", G153)
    r.pose("recap.support.etriers_centre_pas", "F168", G152)
    r.pose("etat.support.etriers_centre", "G168", _etat3_pas(G152, C152))

    # --- dimensionnement de la semelle sur pieux
    sinf, ssup, es = sm["inf"], sm["sup"], sm["etriers"]
    G175, G176, G177, G178 = sinf[0]["n"], sinf[0]["d"], sinf[1]["n"], sinf[1]["d"]
    G181, G182, G183, G184 = ssup[0]["n"], ssup[0]["d"], ssup[1]["n"], ssup[1]["d"]
    G189, G190, G191 = es["pas"], es["d"], es["brins"]
    C172 = r.pose("section.semelle.h_calc", "C172", math.sqrt(C120 * 10 ** 6 / (G14 * C26 * 10 * G13)))
    G172 = r.pose("section.semelle.d", "G172",
                  C25 * 10 - C15 - G190 - max(G176 if G175 > 0 else 0, G178 if G177 > 0 else 0) / 2)
    r.pose("section.semelle.signe_hauteur", "E172", _signe(C172, G172))
    C174 = r.pose("section.semelle.omega_inf", "C174", C120 * 10 ** 6 / (G15 * 0.9 * G172))
    G174 = r.pose("section.semelle.omega_inf_choisie", "G174", math.pi * (G175 * (G176 / 2) ** 2 + G177 * (G178 / 2) ** 2))
    r.pose("section.semelle.signe_inf", "E174", _signe(C174, G174))
    C180 = r.pose("section.semelle.omega_sup", "C180", 0.25 * C174)
    G180 = r.pose("section.semelle.omega_sup_choisie", "G180", math.pi * (G181 * (G182 / 2) ** 2 + G183 * (G184 / 2) ** 2))
    r.pose("section.semelle.signe_sup", "E180", _signe(C180, G180))
    C187 = r.pose("section.semelle.tau", "C187", (C121) * 10 ** 3 / (0.75 * C26 * 10 * G172))
    long_, court = _zone_tau(C187, G12)
    r.pose("section.semelle.tau_zone", "G187", long_)
    C189 = r.pose("section.semelle.pas", "C189",
                  G191 * math.pi * (G190 / 2) ** 2 * G15 * (G172 - C15) / (C121 * 10 ** 3))
    r.pose("section.semelle.signe_pas", "E189", _signe(C189, G189))
    r.pose("section.semelle.hauteur_etrier", "G193", C25 * 10 - C15 * 2 - G190)
    r.pose("recap.semelle.d", "D197", G172)
    r.pose("etat.semelle.hauteur", "G197", _etat3(G172, C172))
    r.pose("etat.semelle.tau_zone", "G199", court)
    r.pose("recap.semelle.inf_1", "D201", f"{G175}Ф{G176}")
    r.pose("recap.semelle.inf_2", "F201", f"{G177}Ф{G178}")
    r.pose("etat.semelle.inf", "G201", _etat3(G174, C174))
    r.pose("recap.semelle.sup_1", "D202", f"{G181}Ф{G182}")
    r.pose("recap.semelle.sup_2", "F202", f"{G183}Ф{G184}")
    r.pose("etat.semelle.sup", "G202", _etat3(G180, C180))
    r.pose("recap.semelle.etriers_brins", "C203", G191)
    r.pose("recap.semelle.etriers_d", "D203", G190)
    r.pose("recap.semelle.etriers_pas", "F203", G189)
    r.pose("etat.semelle.etriers", "G203", _etat3_pas(G189, C189))

    # --- pieu et renforcement au droit du pieu
    r.pose("renfort.D", "C208", C30)
    G224 = G172
    _renfort_pieu(r, C121, C26, C30, G224, m,
                  {"P": "C209", "p1": "C212", "p2a": "C213", "p2b": "C214", "M_A": "C218", "M_B": "C219", "h_A": "C224",
                   "h_B": "C225", "d": "G224", "om_A": "C227", "om_B": "C228", "s_h_A": "E224", "s_h_B": "E225"})
    C222 = r.pose("pieux.sollicitation_T", "C222", C121 / 10)
    G222 = r.pose("pieux.capacite_T", "G222", C31)
    r.pose("pieux.signe", "E222", _signe(C222, G222))
    G227 = r.pose("renfort.omega_brins_horizontaux", "G227", math.pi * (G190 / 2) ** 2 * (C30 * 10 / G189))
    r.pose("renfort.signe_sup_A", "E227", _signe(r["renfort.omega_sup_A"], G227))
    r.pose("renfort.signe_sup_B", "E228", _signe(r["renfort.omega_sup_B"], G227))
    C231 = r.pose("renfort.tau", "C231", (C121) * 10 ** 3 / (0.75 * C26 * 10 * G224))
    r.pose("renfort.tau_texte", "F231", _texte_tau(C231))
    C233 = r.pose("renfort.omega_etriers", "C233", C121 * 10 ** 3 / G15)
    re = f["renfort"]
    G234, G235, G236, G237 = re["brins"], re["d"], re["brins_base"], G190
    r.pose("renfort.d_brins_base", "G237", G237)
    G233 = r.pose("renfort.omega_etriers_choisie", "G233",
                  math.pi * (G236 * (G237 / 2) ** 2) + math.pi * (G234 * (G235 / 2) ** 2))
    r.pose("renfort.signe_etriers", "E233", _signe(C233, G233))
    r.pose("recap.pieux.capacite", "D241", G222)
    r.pose("etat.pieux", "G241",
           OK if C222 * MARGES_PIEU[0] < G222 else (LIMITE if C222 * MARGES_PIEU[1] < G222 else PAS_OK))
    r.pose("recap.renfort.d", "D243", G224)
    hA, hB = r["renfort.h_calc_A"], r["renfort.h_calc_B"]
    r.pose("etat.renfort.hauteur", "G243",
           OK if (_sup(G224, hA) and _sup(G224, hB)) else
           (LIMITE if (hA is not None and hB is not None and G224 > 0.95 * hA and G224 > 0.95 * hB) else PAS_OK))
    oA, oB = r["renfort.omega_sup_A"], r["renfort.omega_sup_B"]
    r.pose("etat.renfort.sup", "G245",
           OK if (G227 > oA and G227 > oB) else (LIMITE if (G227 > 0.95 * oA and G227 > 0.95 * oB) else A_VERIFIER))
    r.pose("recap.renfort.compl", "D246", f"{G234}Ф{G235}")
    r.pose("recap.renfort.base", "F246", f"{G236}Ф{G237}")
    r.pose("etat.renfort.etriers", "G246", _etat3(G233, C233))

    # --- effort horizontal : le moment sur 4 pieds avec, pour bras de levier, la diagonale du châssis
    ch = f["charges"]
    C252 = r.pose("horizontal.H_hs", "C252", ch["H_hs"])
    G252 = r.pose("horizontal.H_s", "G252", ch["H_s"])
    G253 = r.pose("horizontal.M_s", "G253", ch["M_s"])
    C254 = r.pose("horizontal.H_pied", "C254",
                  max(C252 / 3, G252 / 3 + G253 / (2 * C34 / math.sin(math.radians(45)) / 100)))
    C257 = r.pose("horizontal.omega_perfos", "C257", (C254 * 10 ** 3) / D.TAUX_PERFOS)
    G257 = r.pose("horizontal.omega_perfos_choisie", "G257", D.aire(f["perfos"]["n"], f["perfos"]["d"]))
    r.pose("horizontal.signe", "E257", _signe(C257, G257))
    r.pose("horizontal.perfos_par_plaque", "D263", f["perfos"]["n"] * 3)
    r.pose("horizontal.d_perfos", "F263", f["perfos"]["d"])
    r.pose("etat.perfos", "G263", _etat3(G257, C257))
    return r


FICHES = {"cadre4": cadre4, "diagonales8": diagonales8, "semelles8": semelles8}
