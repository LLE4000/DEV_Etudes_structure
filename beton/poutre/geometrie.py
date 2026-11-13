# -*- coding: utf-8 -*-
"""Poutre en béton armé — la GÉOMÉTRIE des aciers : position de chaque lit (distance parement → axe),
centre de gravité y_G d'une face, aires des barres, section des armatures d'effort tranchant.

Les aires et la section des armatures d'effort tranchant reprennent l'expression du module ``modules/poutre.py`` (la
fonction d'origine est nommée) ; la POSITION des lits suit la règle du 13/11/2026, commune à la poutre, à la dalle et à
la prédalle (``beton/position.py`` : la position réelle pour le dessin, l'axe de calcul avec le jeu de pose et
l'arrondi). Unités : Ø en mm, distances en cm, aires en mm².
"""
import math

from beton import position as P

from .entree import BRINS

ENROBAGE_SI_NUL = 3.0    # cm — le module lit l'enrobage « float(enrobage or 3.0) » (_auto_dist_lit)
D_ETRIER_SI_AUCUN = 8.0  # mm — _stirrup_diam_mm sans aucune ligne (inatteignable : une ligne au moins)


def aire_barre(d_mm: float) -> float:
    """Aire d'une barre (mm²) : π·(Ø/2)² — _bar_area_mm2."""
    return math.pi * (d_mm / 2.0) ** 2


def arrondi_demi_cm(x_cm: float) -> float:
    """Arrondi au 0,5 cm SUPÉRIEUR — _round_up_to_half_cm."""
    return math.ceil(float(x_cm) * 2.0) / 2.0


def enrobage_de_calcul(enrobage: float) -> float:
    """L'enrobage (cm) qui place le lit 1 : la saisie, sauf un enrobage NUL, que le module lit 3,0 cm
    (« or 3.0 » dans _auto_dist_lit, alors que son écran accepte 0). Défaut du module reproduit,
    signalé par un avertissement de ``moteur.verifier``."""
    return float(enrobage or ENROBAGE_SI_NUL)


def diametre_etrier(etriers: list) -> float:
    """Ø (mm) des armatures d'effort tranchant qui place le lit 1 : le plus grand des lignes
    (cas défavorable) — _stirrup_diam_mm."""
    diams = [float(e["d"]) for e in etriers]
    return max(diams) if diams else D_ETRIER_SI_AUCUN


def positions_lits(lits: list, enrobage_calc: float, d_etrier: float, jeu_entre_lits: float) -> list:
    """La position RÉELLE de chaque lit (cm, parement → axe) — ce que les dessins montrent (beton/position.py) :

        lit 1 : c + Ø_étrier + Ø₁/2                 (la barre dans l'angle du cadre)
        lit i : dessus du lit i−1 + jeu entre lits + Øᵢ/2 = pos(i−1) + Ø(i−1)/2 + jeu entre lits + Øᵢ/2

    les Ø réels, sans arrondi (Ø_étrier = le plus grand des étriers et épingles). Une position IMPOSÉE (clé « pos » du
    lit, site) remplace la règle : lit 1, la distance parement → axe ; lit i > 1, l'entraxe vertical depuis le lit
    précédent (pos(i) = pos(i−1) + « pos »)."""
    out = []
    for i, lit in enumerate(lits):
        pos = lit.get("pos")
        if i == 0:
            r = float(pos) if pos is not None else P.sur_parement(enrobage_calc, lit["d"], d_etrier)
        elif pos is not None:
            r = out[-1] + float(pos)
        else:
            r = P.sur_couche(P.dessus(out[-1], lits[i - 1]["d"]), lit["d"], jeu_entre_lits)
        out.append(r)
    return out


def distances_lits(lits: list, enrobage_calc: float, d_etrier: float, jeu_premier_lit: float,
                   jeu_entre_lits: float, arrondi: float = P.ARRONDI) -> list:
    """L'axe de CALCUL de chaque lit (cm, parement → axe) : ⌈position réelle + jeu de pose⌉ au pas d'arrondi
    (beton/position.py, règle du 13/11/2026 — « jeu_premier_lit » est le jeu de pose). C'était, dans le module
    Streamlit : lit 1, c + ⌈Ø_étrier/10 + Ø₁/20⌉ + jeu ; lit i, dist(i−1) + ⌈Ø(i−1)/20⌉ + jeu entre lits + ⌈Øᵢ/20⌉."""
    return [P.axe_calcul(r, jeu_premier_lit, arrondi) for r in positions_lits(lits, enrobage_calc, d_etrier, jeu_entre_lits)]


def details_lits(lits: list, enrobage_calc: float, d_etrier: float, jeu_premier_lit: float, jeu_entre_lits: float,
                 arrondi: float = P.ARRONDI) -> list:
    """Le calcul de l'axe de chaque lit, terme à terme (``P.detail``) — « c + Ø_étr + Ø/2 + jeu = 3 + 0,8 + 0,8 + 1 =
    5,6 → 6,0 cm » ; un lit au-dessus d'un autre reprend les termes de celui-ci, son demi-Ø devenu son Ø entier (le
    dessus du lit), puis le jeu entre lits et son demi-Ø ; une position imposée : « posé », ou « lit i−1 » + « e_v »."""
    reels = positions_lits(lits, enrobage_calc, d_etrier, jeu_entre_lits)
    termes = []
    for i, lit in enumerate(lits):
        pos = lit.get("pos")
        if i == 0:
            tt = ([("posé", float(pos))] if pos is not None
                  else [("c", float(enrobage_calc)), ("Ø_étr", float(d_etrier) / 10.0), ("Ø/2", lit["d"] / 20.0)])
        elif pos is not None:
            tt = [(f"lit {i}", reels[i - 1]), ("e_v", float(pos))]
        else:
            avant = termes[-1]
            dessous = (avant[:-1] + [(f"Ø_{i}", lits[i - 1]["d"] / 10.0)] if avant[-1][0] == "Ø/2"
                       else avant + [(f"Ø_{i}/2", lits[i - 1]["d"] / 20.0)])
            tt = dessous + [("jeu lits", float(jeu_entre_lits)), ("Ø/2", lit["d"] / 20.0)]
        termes.append(tt)
    return [P.detail(tt, jeu_premier_lit, arrondi) for tt in termes]


# ============================================================ le détail horizontal (site) : dessin, note, dispositions
# Les ABSCISSES des barres et l'emprise des cadres, en cm depuis la face GAUCHE de la poutre. La convention est celle de
# la coupe de la note PDF (ndc_pdf/section.py) : le nu extérieur de l'étrier à l'enrobage, l'axe d'une barre au contact
# du nu intérieur de l'étrier (enrobage + Ø étrier + Ø/2). Rien de ceci n'entre dans le calcul de flexion ou d'effort
# tranchant : ces grandeurs servent au dessin, à la note et au contrôle des dispositions constructives.
def abscisses_lit(lit: dict, b: float, enrobage_calc: float, d_etrier: float, ref=None) -> list:
    """Les abscisses (cm) des n barres d'un lit. ``ref`` : les abscisses du lit 1 de la face (répartition « aligne »).

    uniforme : réparties entre les nus intérieurs de l'étrier (n = 1 : au milieu) ;
    aligne   : sur les barres du lit 1, les angles d'abord, puis réparties (au-delà de n₁ barres : uniforme) ;
    entraxe  : un groupe centré, entraxe ``sh`` (cm) — sans ``sh`` : uniforme."""
    n, r = int(lit["n"]), lit["d"] / 20.0
    x0 = enrobage_calc + d_etrier / 10.0 + r
    x1 = b - x0
    rep = lit.get("rep", "uniforme")
    if rep == "entraxe" and lit.get("sh"):
        sh = float(lit["sh"])
        return [b / 2.0 + (k - (n - 1) / 2.0) * sh for k in range(n)]
    if rep == "aligne" and ref and 1 <= n < len(ref):
        if n == 1:
            return [ref[len(ref) // 2]] if len(ref) % 2 else [b / 2.0]
        m = len(ref) - 1
        return [ref[round(k * m / (n - 1))] for k in range(n)]
    if n == 1:
        return [b / 2.0]
    return [x0 + k * (x1 - x0) / (n - 1) for k in range(n)]


def abscisses_face(lits: list, b: float, enrobage_calc: float, d_etrier: float) -> list:
    """Les abscisses de chaque lit d'une face (le lit 1 sert de référence à la répartition « aligne »)."""
    out = []
    for i, lit in enumerate(lits):
        out.append(abscisses_lit(lit, b, enrobage_calc, d_etrier, out[0] if i else None))
    return out


def laterales(section: dict, poutre: dict, h: float, dist_inf1: float, dist_sup1: float) -> dict:
    """Les armatures LATÉRALES (de peau) : par face latérale, n barres réparties entre les lits 1 des deux faces.

    Automatique (pas de « n » imposé) : la règle des armatures de peau de la note PDF Streamlit (_peau_bars) —
    écart vertical d = h − dist₁,inf − dist₁,sup ; si d > s_max : n = ⌈d / s_max⌉ − 1 par face, sinon aucune.
    Rend {"n", "d" (mm), "ys" (cm depuis le bas), "auto" (bool), "s_max", "ecart"}."""
    imp = section.get("laterales") or {}
    d_mm = int(imp.get("d") or poutre.get("peau_d") or 10)
    s_max = float(poutre.get("peau_smax") or 30.0)
    ecart = h - dist_inf1 - dist_sup1
    if imp.get("n") is not None:
        n, auto = int(imp["n"]), False
    else:
        auto = True
        n = max(0, int(math.ceil(ecart / s_max)) - 1) if (s_max > 0 and ecart > s_max) else 0
    pas = ecart / (n + 1) if n else 0.0
    return {"n": n, "d": d_mm, "ys": [dist_inf1 + k * pas for k in range(1, n + 1)], "auto": auto, "s_max": s_max,
            "ecart": ecart}


def emprise_etrier(etrier: dict, xs_inf1: list, d_lit1: float, b: float, enrobage_calc: float) -> dict:
    """L'emprise horizontale (cm) d'une ligne d'effort tranchant, au nu EXTÉRIEUR : un étrier entoure les barres « de »
    à « a » du lit 1 inférieur (absentes : toute la largeur, nu extérieur à l'enrobage) ; une épingle longe la barre
    « de » (absente : la barre du milieu). Rend {"x0", "x1", "de", "a", "complet", "brins"}."""
    n1 = len(xs_inf1)
    de = etrier.get("de")
    a = etrier.get("a")
    borne = lambda v, dflt: max(1, min(n1, int(v))) if v is not None else dflt  # noqa: E731
    de_, a_ = borne(de, 1), borne(a, n1)
    if etrier["type"] == "epingle":
        k = borne(de, (n1 + 1) // 2)
        x = xs_inf1[k - 1]
        return {"x0": x, "x1": x, "de": k, "a": k, "complet": False, "brins": 1}
    if de_ > a_:
        de_, a_ = a_, de_
    complet = de_ == 1 and a_ == n1
    if complet:
        return {"x0": enrobage_calc, "x1": b - enrobage_calc, "de": 1, "a": n1, "complet": True, "brins": 2}
    r = d_lit1 / 20.0
    de_mm = etrier["d"] / 10.0
    return {"x0": xs_inf1[de_ - 1] - r - de_mm, "x1": xs_inf1[a_ - 1] + r + de_mm, "de": de_, "a": a_, "complet": False,
            "brins": 2}


def face(lits: list, distances: list, yG_impose=None):
    """Une face : (As total en mm², y_G calculé en cm, y_G retenu en cm) — _layers_geometry.

    y_G calculé = Σ(Asᵢ·eᵢ) / ΣAsᵢ, eᵢ = distance du lit i ; y_G retenu = l'imposé s'il est donné
    (champ « CDG (cm) » de l'écran, _ycdg_manual), sinon le calculé."""
    As_tot = 0.0
    somme_As_e = 0.0
    for lit, e in zip(lits, distances):
        As_i = lit["n"] * aire_barre(lit["d"])
        As_tot += As_i
        somme_As_e += As_i * e
    yG_calcule = (somme_As_e / As_tot) if As_tot > 0 else distances[0]
    return As_tot, yG_calcule, (yG_calcule if yG_impose is None else yG_impose)


def aire_etriers(etriers: list) -> float:
    """A_st (mm²) = Σ brins × π·(Ø/2)², une ligne = un étrier (2 brins) ou une épingle (1 brin) —
    _shear_lines_total_Ast_mm2."""
    Ast = 0.0
    for e in etriers:
        Ast += BRINS[e["type"]] * aire_barre(float(e["d"]))
    return Ast


def libelle_lits(lits: list) -> str:
    """« 2Ø16 + 2Ø20 » : la désignation que l'écran affiche (_layers_geometry)."""
    return " + ".join(f"{lit['n']}Ø{lit['d']}" for lit in lits)


def resume_etriers(etriers: list) -> str:
    """« Étrier Ø8 + 2 épingles Ø8 » : la désignation du bandeau des étriers — _shear_lines_summary."""
    ordre, compte = [], {}
    for e in etriers:
        cle = ("Épingle" if BRINS[e["type"]] == 1 else "Étrier", int(float(e["d"])))
        if cle not in compte:
            compte[cle] = 0
            ordre.append(cle)
        compte[cle] += 1
    return " + ".join(f"{nom} Ø{d}" if compte[(nom, d)] == 1 else f"{compte[(nom, d)]} {nom.lower()}s Ø{d}"
                      for nom, d in ordre)
