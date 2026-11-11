# -*- coding: utf-8 -*-
"""Fondations de grues — les ÉCARTS trouvés en relisant les classeurs (``docs/beton/FONDATIONS_GRUE_ETUDE.md`` § 5).

Le module reproduit les classeurs TELS QU'ILS SONT (``classeur.py``). Ce fichier ne corrige rien : pour la fondation
calculée, il dit où une formule du classeur s'écarte de ce que la statique ou sa propre logique donnent, avec les deux
chiffres, pour que le bureau tranche. Chaque écart :

    {"code": "E1", "gravite": "non_conservatif" | "conservatif" | "incoherence" | "manque" | "limite",
     "bloc": le groupe de résultats concerné (« horizontal », « pieux », « poutre »…),
     "cellule": la cellule du classeur, "titre": une ligne, "message": la phrase complète,
     "classeur": la valeur du classeur, "redemontre": la valeur redémontrée, "unite": …,
     "change_verdict": True si la valeur redémontrée ferait changer l'état de la vérification}

Un écart n'apparaît que s'il a un EFFET sur la fondation calculée (ou s'il manque une vérification). Bibliothèque
standard seulement.
"""
import math

from . import classeur as K
from . import donnees as D
from . import entree as E
from . import statique as S

NON_CONSERVATIF, CONSERVATIF, INCOHERENCE, MANQUE, LIMITE = "non_conservatif", "conservatif", "incoherence", "manque", "limite"
LIBELLES_GRAVITE = {NON_CONSERVATIF: "non conservatif", CONSERVATIF: "conservatif", INCOHERENCE: "incohérence",
                    MANQUE: "vérification absente", LIMITE: "limite du classeur"}
SEUIL = 1e-9                     # en deçà, deux valeurs sont tenues pour égales (bruit du calcul flottant)
X_MAX_TABLEAU = (K.N_X - 1) * K.PAS_X     # 10,5 m : fin des grands tableaux de BASE


def _n(x: float, nd: int = 1) -> str:
    """Un nombre à la française pour un message (virgule, sans zéros inutiles au-delà de ``nd``)."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def _ecart(code, gravite, bloc, cellule, titre, message, classeur=None, redemontre=None, unite="", change=False):
    return {"code": code, "gravite": gravite, "bloc": bloc, "cellule": cellule, "titre": titre, "message": message,
            "classeur": classeur, "redemontre": redemontre, "unite": unite, "change_verdict": bool(change)}


def _differe(a: float, b: float) -> bool:
    return abs(a - b) > SEUIL * max(abs(a), abs(b), 1.0)


# ------------------------------------------------------------------ écarts communs aux deux variantes
def _horizontal_variantes(f: dict, r: K.Releve, cellule: str) -> list:
    """E1 — =MAX(Hhs/3;Hs/3+M/2/L/100) avec L en cm : le terme du moment est divisé par 100·L(cm) au lieu de L(m)."""
    ch, L = f["charges"], f["chassis"] / 100
    H = r["horizontal.H_pied"]
    diagonale = max(ch["H_hs"] / 3, ch["H_s"] / 3 + ch["M_s"] / (2 * L / math.sin(math.radians(45))))
    cote = max(ch["H_hs"] / 3, ch["H_s"] / 3 + ch["M_s"] / (2 * L))
    if not (_differe(H, diagonale) or _differe(H, cote)):
        return []
    choisi = r["horizontal.omega_perfos_choisie"]
    pire = max(diagonale, cote)
    return [_ecart(
        "E1", NON_CONSERVATIF, "horizontal", cellule, "Effort horizontal par pied : le moment ne compte pas",
        f"La formule du classeur divise le moment par 100 × L (en cm) au lieu de L (en m) : son terme vaut "
        f"{_n(ch['M_s'] / 2 / f['chassis'] / 100, 4)} kN. Classeur : {_n(H)} kN. Avec la formule du classeur BASE "
        f"(bras = diagonale du châssis) : {_n(diagonale)} kN ; avec M / (2·L) : {_n(cote)} kN, soit "
        f"{_n(pire * 1000 / D.TAUX_PERFOS, 0)} mm² de perfos au lieu de {_n(r['horizontal.omega_perfos'], 0)} "
        f"(choisi : {_n(choisi, 0)} mm²).",
        H, {"diagonale": diagonale, "cote": cote}, "kN", change=pire * 1000 / D.TAUX_PERFOS >= choisi)]


def _seuil_tau(f: dict, r: K.Releve, cles_cellules) -> list:
    """E2 — le seuil « 0,56 » est écrit en dur : il vaut 0,016·R'wk en C30/37 seulement."""
    seuil = 0.016 * D.RWK[f["beton"]]
    if not _differe(seuil, D.TAU_SEUIL_EN_DUR):
        return []
    out = []
    for cle, cellule, bloc in cles_cellules:
        tau = r[cle]
        change = (tau < D.TAU_SEUIL_EN_DUR) != (tau < seuil)
        out.append(_ecart(
            "E2", NON_CONSERVATIF if seuil < D.TAU_SEUIL_EN_DUR else CONSERVATIF, bloc, cellule,
            "Seuil de cisaillement écrit en dur",
            f"Le classeur compare τ = {_n(tau, 2)} N/mm² à 0,56 (0,016 × 35, le béton C30/37). Pour le {f['beton']}, "
            f"0,016 × R'wk = {_n(seuil, 2)} N/mm²" + (" : la conclusion change." if change else "."),
            D.TAU_SEUIL_EN_DUR, seuil, "N/mm²", change))
    return out


def _moment_sous_charge(r: K.Releve, poutres: dict, cle_M: str, cellule: str, bloc: str, cle_omega: str,
                        cle_choisi: str, d: float, sigma: float) -> list:
    """E16 — CAS B : le classeur prend le moment SOUS LA CHARGE (RA·e − pp·e²/2) ; si l'effort tranchant ne change pas
    de signe sous la charge, le maximum est plus loin dans la travée."""
    exact = poutres["B"].extremes()["M_max"]
    M = r[cle_M]
    if exact["valeur"] <= M * (1 + 1e-6):
        return []
    omega = exact["valeur"] * 10 ** 6 / (sigma * 0.9 * d)
    return [_ecart(
        "E16", NON_CONSERVATIF, bloc, cellule, "Moment maximal pris sous la charge",
        f"En CAS B le classeur prend le moment sous le pied de la grue : {_n(M)} kN·m. L'effort tranchant y reste "
        f"positif (le poids propre continue de charger la travée) : le maximum est {_n(exact['valeur'])} kN·m, "
        f"{_n(exact['x'] - poutres['B'].xa, 2)} m après l'appui (+{_n(100 * (exact['valeur'] / M - 1))} %), soit "
        f"Ω inf = {_n(omega, 0)} mm² au lieu de {_n(r[cle_omega], 0)} (choisi : {_n(r[cle_choisi], 0)} mm²).",
        M, exact["valeur"], "kN·m", change=omega >= r[cle_choisi])]


def _soulevement(r: K.Releve, cle: str, cellule: str, bloc: str, nom: str) -> list:
    """E5 — une réaction d'appui négative (soulèvement) que le classeur ne signale pas."""
    RB = r[cle]
    if RB >= 0:
        return []
    return [_ecart("E5", LIMITE, bloc, cellule, "Appui soulevé",
                   f"{nom} : RB = {_n(RB)} kN en CAS A. L'appui opposé à la charge se soulève ; le classeur ne le "
                   f"signale pas et ne vérifie pas que la poutre y est tenue.", RB, None, "kN")]


# ------------------------------------------------------------------ par schéma
def _cadre4(f: dict, r: K.Releve) -> list:
    out = _horizontal_variantes(f, r, "C130")
    out += _seuil_tau(f, r, (("section.poutre.tau", "F63", "poutre"), ("renfort.tau", "F112", "renfort")))
    L, e = f["chassis"] / 100, f["e"] / 100
    poutres = S.poutre_un_pied(L, e, f["charges"]["P"], r["pp.poutre"])
    out += _moment_sous_charge(r, poutres, "poutre.cas_B.M_max", "G37", "poutre", "section.poutre.omega_inf",
                               "section.poutre.omega_inf_choisie", r["section.poutre.d"], r["materiaux.sigma_adm"])
    out += _soulevement(r, "poutre.cas_A.RB", "C35", "poutre", "Poutre")
    part = f["poutre"]["part_V"]
    if part != 100:
        out.append(_ecart(
            "E6", INCOHERENCE, "poutre", "C63 / C66", "τ et étriers ne parlent pas du même effort",
            f"τ est calculé sur {_n(part, 0)} % de V max ({_n(r['poutre.dim.V_max'] * part / 100)} kN), les étriers à "
            f"l'appui sur 100 % ({_n(r['poutre.dim.V_max'])} kN). Dans la variante 8 pieux, la même part s'applique "
            f"aux deux.", part, 100.0, "%"))
    return out


def _diagonales8(f: dict, r: K.Releve) -> list:
    out = _horizontal_variantes(f, r, "C155")
    out += _seuil_tau(f, r, (("section.secondaire.tau", "F78", "secondaire"), ("section.primaire.tau", "F103", "primaire"),
                             ("renfort.tau", "F137", "renfort")))
    L, e = f["chassis"] / 100, f["e_secondaire"] / 100
    poutres = S.poutre_un_pied(L, e, f["charges"]["P"], r["pp.secondaire"])
    out += _moment_sous_charge(r, poutres, "secondaire.cas_B.M_max", "G39", "secondaire", "section.secondaire.omega_inf",
                               "section.secondaire.omega_inf_choisie", r["section.secondaire.d"], r["materiaux.sigma_adm"])
    out += _soulevement(r, "secondaire.cas_A.RB", "C37", "secondaire", "Poutre secondaire")
    out.append(_ecart(
        "E4", MANQUE, "pieux", "C62", "Capacité des pieux non vérifiée",
        f"Le classeur calcule la réaction maximale d'un pieu (R max de la poutre primaire = "
        f"{_n(r['primaire.dim.R_max'])} kN, soit {_n(r['primaire.dim.R_max'] / 10)} T) mais ne la compare à aucune "
        f"capacité portante.", r["primaire.dim.R_max"], None, "kN"))
    return out


def _semelles8(f: dict, r: K.Releve) -> list:
    out = _seuil_tau(f, r, (("renfort.tau", "F231", "renfort"),))
    # E3 — le pieu vérifié avec V max de la semelle au lieu de sa réaction R max
    V, R, cap = r["semelle.dim.V_max"], r["semelle.dim.R_max"], f["pieux"]["capacite"]
    if _differe(V, R):
        def etat(t):
            return K.OK if t * K.MARGES_PIEU[0] < cap else (K.LIMITE if t * K.MARGES_PIEU[1] < cap else K.PAS_OK)
        e_r = etat(R / 10)
        out.append(_ecart(
            "E3", NON_CONSERVATIF, "pieux", "C222", "Pieu vérifié sur l'effort tranchant, pas sur sa réaction",
            f"Le classeur compare à la capacité V max de la semelle / 10 = {_n(V / 10)} T (effort tranchant au nu "
            f"intérieur du pieu). La réaction du pieu, R max = {_n(R)} kN soit {_n(R / 10)} T, est calculée (C119) et "
            f"n'est pas utilisée. Avec R max : état « {e_r} » (capacité {_n(cap, 0)} T).",
            V / 10, R / 10, "T", change=e_r != r["etat.pieux"]))
    # E7 — bras des étriers : d − enrobage au lieu de la hauteur de l'étrier calculée à côté
    for cle, h_cle, cellule, bloc, nom, pas_cle in (
            ("section.support.pas_centre", "section.support.hauteur_etrier_centre", "C152", "support",
             "étriers de la partie centrale", "support/etriers_centre/pas"),
            ("section.semelle.pas", "section.semelle.hauteur_etrier", "C189", "semelle", "étriers de la semelle",
             "semelle/etriers/pas")):
        d = r["section.support.d" if bloc == "support" else "section.semelle.d"]
        z, h = d - f["enrobage"], r[h_cle]
        if _differe(z, h):
            pas = r[cle] * h / z
            out.append(_ecart(
                "E7", CONSERVATIF if h > z else NON_CONSERVATIF, bloc, cellule, "Bras de levier des étriers",
                f"Le pas des {nom} est calculé avec d − enrobage = {_n(z, 0)} mm ; la « hauteur de calcul des "
                f"étriers » affichée à côté ({_n(h, 0)} mm) n'entre pas dans la formule, alors que les étriers des "
                f"extrémités l'utilisent. Avec elle : {_n(pas, 0)} mm au lieu de {_n(r[cle], 0)} mm "
                f"(choisi : {_n(E.lire(f, pas_cle), 0)} mm).", r[cle], pas, "mm"))
    # E15 — le « Check : pas choisi ≤ hauteur étrier » n'est qu'un texte
    pas_c, h_c = f["support"]["etriers_centre"]["pas"], r["section.support.hauteur_etrier_centre"]
    if pas_c > h_c:
        out.append(_ecart(
            "E15", LIMITE, "support", "G152", "Pas des étriers centraux supérieur à leur hauteur",
            f"Pas choisi en partie centrale : {_n(pas_c, 0)} mm, hauteur de l'étrier : {_n(h_c, 0)} mm. Le classeur "
            f"l'écrit en rouge (« Check : pas choisi ≤ hauteur étrier ? ») sans changer l'état de la vérification.",
            pas_c, h_c, "mm"))
    # E9 — hauteur calculée sur M max alors que Ω inf est calculée sur M max éq
    M, Meq = r["support.dim.M_max"], r["support.dim.M_max_eq"]
    if _differe(M, Meq):
        h_eq = math.sqrt(Meq * 10 ** 6 / (r["materiaux.mu"] * f["support"]["b"] * 10 * r["materiaux.sigma_b"]))
        out.append(_ecart(
            "E9", NON_CONSERVATIF, "support", "C126", "Hauteur calculée sans le moment horizontal",
            f"La hauteur calculée de la poutre support utilise M max = {_n(M)} kN·m alors que Ω inf utilise "
            f"M max éq = {_n(Meq)} kN·m (avec l'effet de H). Avec M max éq : {_n(h_eq, 0)} mm au lieu de "
            f"{_n(r['section.support.h_calc'], 0)} mm (hauteur utile : {_n(r['section.support.d'], 0)} mm).",
            r["section.support.h_calc"], h_eq, "mm", change=h_eq >= r["section.support.d"]))
    # E8 — maximum échantillonné (5 cm ; 1 % de la zone chargée), tableau arrêté à 10,5 m
    Lc, a, b, e = f["chassis"] / 100, f["dist_a"] / 100, f["dist_b"] / 100, f["e_support"] / 100
    exact = 0.0
    for k, t in r.tableaux.items():
        for col in t["colonnes"]:
            exact = max(exact, S.poutre_support(Lc, a, b, e, k, col["P1"], col["P2"], r["pp.support"]).extremes()["M_max"]["valeur"])
    if exact > M * (1 + 1e-4):
        out.append(_ecart(
            "E8", NON_CONSERVATIF, "support", "C94", "Moment maximal lu tous les 5 cm",
            f"Le classeur prend le plus grand M(x) d'un tableau au pas de 5 cm : {_n(M, 2)} kN·m. Le maximum exact est "
            f"{_n(exact, 2)} kN·m (+{_n(100 * (exact / M - 1), 2)} %) : le pied de la grue ne tombe pas sur un point du "
            f"tableau.", M, exact, "kN·m"))
    L_max = max(t["geometrie"]["L"] for t in r.tableaux.values())
    if L_max > X_MAX_TABLEAU:
        out.append(_ecart(
            "E8", NON_CONSERVATIF, "support", "A246", "Poutre plus longue que le tableau du classeur",
            f"La portée atteint {_n(L_max, 2)} m ; les tableaux du classeur s'arrêtent à x = {_n(X_MAX_TABLEAU)} m : "
            f"le moment au-delà n'est pas lu.", X_MAX_TABLEAU, L_max, "m"))
    semelle = S.poutre_deux_pieux(f["semelle"]["L"] / 100, f["pieux"]["entraxe"] / 100, f["e_semelle"] / 100,
                                  r["semelle.charge"], r["pp.semelle"], f["c_appui"] / 100)
    Ma = semelle["A"].extremes()["M_max"]["valeur"]
    if Ma > r["semelle.cas_A.M_max"] * (1 + 1e-4):
        out.append(_ecart(
            "E8", NON_CONSERVATIF, "semelle", "C115", "Moment de la semelle lu en 101 points",
            f"CAS A de la semelle : {_n(r['semelle.cas_A.M_max'], 2)} kN·m au tableau, {_n(Ma, 2)} kN·m exact.",
            r["semelle.cas_A.M_max"], Ma, "kN·m"))
    # E10 — la semelle n'est calculée qu'avec RA,max
    RA, RB = r["support.dim.RA_max"], r["support.dim.RB_max"]
    if f["cote_b"] == E.COTE_SEMELLES and RB > RA * (1 + 1e-9):
        out.append(_ecart(
            "E10", NON_CONSERVATIF, "semelle", "C104", "Semelle du côté b plus chargée",
            f"La semelle est calculée avec RA max = {_n(RA)} kN. Avec quatre semelles, celle du côté b reçoit "
            f"RB max = {_n(RB)} kN (+{_n(100 * (RB / RA - 1))} %), que le classeur réserve à la paroi de pieux sécants.",
            RA, RB, "kN"))
    return out


_PAR_TYPE = {E.CADRE4: _cadre4, E.DIAGONALES8: _diagonales8, E.SEMELLES8: _semelles8}


def signaler(f: dict, r: K.Releve) -> list:
    """Les écarts qui touchent la fondation NORMALISÉE ``f``, dont ``r`` est le relevé (``classeur.FICHES``)."""
    return _PAR_TYPE[f["type"]](f, r)
