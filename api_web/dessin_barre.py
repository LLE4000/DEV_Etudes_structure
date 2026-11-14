# -*- coding: utf-8 -*-
"""Les dessins d'une BARRE CHARGÉE du module « Choix de profilé » : ses cas de charge (une bande par cas), la poutre —
ses appuis, ses maintiens, ses portées —, puis les enveloppes du moment et de l'effort tranchant ; et les TAUX de
toutes les vérifications, méthode par méthode.

Rien n'est calculé ici : les enveloppes viennent du moteur (``resultat["statique"]``), les taux de ses vérifications ;
ce fichier les POSE. Du SVG en unités d'écran, aux couleurs des dessins du module (api_web/dessin_profil). La scène de la
barre a la largeur qu'on lui donne (640 à l'écran) : la note PDF la demande plus étroite, pour que ses textes restent
lisibles dans une colonne de 10 cm — le même dessin, le même code (ndc_pdf/svg.py le peint).
"""
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

from acier.barre import resultat as RE

from . import commun as C
from . import dessin_profil as DP
from . import profil_mots as M

W = 640.0                        # la largeur de la scène à l'écran
ML, MR = 78.0, 30.0              # à gauche : le nom d'un cas et sa nature, le nom d'un diagramme
H_CAS = 44.0                     # la hauteur de la bande d'un cas de charge
H_DIAG = 84.0
CHARGE, CHARGE_FOND, DIAG_M, DIAG_V = "#9C3341", "#F6E3E6", "#2B4BCF", "#0F766E"
FS = 11.0


def _t(x, y, s, fs=FS, ancre="middle", couleur=DP.ENCRE, gras=False) -> str:
    return DP._txt(x, y, s, fs, ancre=ancre, couleur=couleur, gras=gras)


def _trait(d: str, couleur: str, lw: float, extra: str = "") -> str:
    return f'<path d="{d}" fill="none" stroke="{couleur}" stroke-width="{DP._f(lw)}" stroke-linecap="round" stroke-linejoin="round"{extra}/>'


def _fleche(x: float, y1: float, y2: float, couleur: str, lw: float = 1.2) -> str:
    """Une flèche verticale de y1 vers y2 (la pointe en y2)."""
    s = 1.0 if y2 > y1 else -1.0
    return (_trait(f"M{x:.1f} {y1:.1f}V{y2 - s:.1f}", couleur, lw)
            + f'<path d="M{x:.1f} {y2:.1f}l-3.2 {-s * 6.5:.1f}h6.4z" fill="{couleur}"/>')


def _nb(x: float, nd: int = 1) -> str:
    return C.fr(x, nd).replace("-", M.MOINS)


class Cadre:
    """La scène d'une barre : sa largeur, ses marges, l'échelle des abscisses (px par mètre)."""

    def __init__(self, largeur: float, Lt: float):
        self.w, self.Lt = float(largeur), float(Lt)
        self.ech = (self.w - ML - MR) / self.Lt

    def X(self, x: float) -> float:
        return ML + x * self.ech

    def ancre(self, x: float, marge: float) -> str:
        """Où accrocher un texte posé à l'abscisse écran ``x`` pour qu'il reste dans la scène."""
        return "start" if x < ML + marge else ("end" if x > self.w - MR - marge else "middle")


def _nature(c: Dict) -> str:
    """« permanente », « exploitation A » : la nature d'un cas, sans majuscule au premier mot."""
    t = M.NATURE_COURTE[c["nature"]]
    return t[:1].lower() + t[1:]


# ------------------------------------------------------------------ les cas de charge
def _largeur(texte: str, fs: float) -> float:
    """La largeur d'un texte gras, estimée (un dessin n'a pas la métrique de la police) : 0,58 du corps par caractère."""
    return 0.58 * fs * len(texte)


def _bande_cas(c: Dict, y: float, K: Cadre, vmax: float) -> str:
    """La bande d'un cas : son nom à gauche, ses charges sur une ligne de base ; ``y`` : la ligne de base. Les valeurs
    des charges ponctuelles et des couples sont placées d'abord ; celle d'une charge répartie se met au milieu, sinon à
    un bout de la charge, là où elle ne recouvre aucune valeur déjà écrite. Les charges réparties sont peintes dessous,
    les flèches des charges ponctuelles et toutes les valeurs dessus."""
    X, Lt, W = K.X, K.Lt, K.w
    out = [_t(8, y - 16, c["nom"], 11.5, "start", gras=True), _t(8, y - 3, _nature(c), 9.0, "start", DP.MUET),
           _trait(f"M{X(0):.1f} {y:.1f}H{X(Lt):.1f}", DP.GRIS, 1.0)]
    fonds: List[str] = []                                         # les charges réparties (dessous)
    dessus: List[str] = []                                        # les charges ponctuelles, les couples, toutes les valeurs
    pris: List[Tuple[float, float]] = []                          # l'étendue (x1, x2) de chaque valeur déjà écrite
    for ch in c["charges"]:
        t, v = ch["type"], ch["v"]
        if t in ("q", "tri"):
            continue
        x = X(ch["a"])
        bas = v >= 0
        if t == "P":
            dessus.append(_fleche(x, y - 30, y - 0.6, CHARGE, 1.6) if bas else _fleche(x, y - 0.6, y - 30, CHARGE, 1.6))
            txt, dx, dy, marge = f"{M.n(abs(v), 2)} kN", 5.0, 22.0, 60.0
        else:
            sens = 1 if bas else 0
            dessus.append(f'<path d="M{x - 9:.1f} {y - 4:.1f}A10 10 0 1 {sens} {x + 9:.1f} {y - 4:.1f}" fill="none" stroke="{CHARGE}" stroke-width="1.4"/>')
            dessus.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{CHARGE}"/>')
            txt, dx, dy, marge = f"{M.n(abs(v), 2)} kN·m", 13.0, 18.0, 70.0
        ancre = "start" if x < W - MR - marge else "end"
        w = _largeur(txt, 10.0)
        x1, x2 = (x + dx, x + dx + w) if ancre == "start" else (x - dx - w, x - dx)
        if any(o["type"] in ("q", "tri") and X(o["a"]) < x2 and X(o["b"]) > x1 for o in c["charges"]):
            dy = 32.0                                             # au-dessus d'une charge répartie : la valeur passe par-dessus son bord
        dessus.append(_t(x + (dx if ancre == "start" else -dx), y - dy, txt, 10.0, ancre, CHARGE, True))
        pris.append((x1, x2))
    for ch in c["charges"]:
        t, v = ch["type"], ch["v"]
        if t not in ("q", "tri"):
            continue
        bas = v >= 0
        xa, xb = X(ch["a"]), X(ch["b"])
        h = 9.0 + 17.0 * abs(v) / vmax
        ha, hb = (h, h) if t == "q" else ((0.0, h) if ch.get("sens", "croissante") == "croissante" else (h, 0.0))
        s = -1.0 if bas else 1.0                                  # une charge vers le bas se dessine AU-DESSUS de la ligne
        fonds.append(f'<path d="M{xa:.1f} {y:.1f}L{xa:.1f} {y + s * ha:.1f}L{xb:.1f} {y + s * hb:.1f}L{xb:.1f} {y:.1f}Z" fill="{CHARGE_FOND}" '
                     f'stroke="{CHARGE}" stroke-width="0.9"/>')
        nb_ = max(2, int((xb - xa) / 30.0) + 1)
        for j in range(nb_):
            xj = xa + (xb - xa) * j / (nb_ - 1)
            hj = ha + (hb - ha) * j / (nb_ - 1)
            if hj > 8:
                fonds.append(_fleche(xj, y + s * hj, y + s * 0.6, CHARGE, 0.9))
        txt = f"{M.n(abs(v), 2)} kN/m"
        w = _largeur(txt, 10.0)
        milieu = (xa + xb) / 2
        xm = next((x_ for x_ in (milieu, xa + w / 2 + 3, xb - w / 2 - 3)
                   if all(x_ + w / 2 + 4 < a or x_ - w / 2 - 4 > b for a, b in pris)), milieu)
        dessus.append(_t(xm, y + s * h - 4 if bas else y + s * h + 12, txt, 10.0, couleur=CHARGE, gras=True))
        pris.append((xm - w / 2, xm + w / 2))
    if c.get("N"):
        dessus.append(_t(W - 4, y + 12, f"N = {M.n(c['N'], 1)} kN", 9.5, "end", CHARGE))
    return "".join(out + fonds + dessus)


# ------------------------------------------------------------------ la poutre
def _appuis(el: Dict) -> List[Tuple[float, str]]:
    """Les appuis du modèle : (abscisse, « articule » | « encastre » | « libre »)."""
    L = sum(el["travees"])
    mo = el["modele"]
    if mo == "console":
        return [(0.0, "encastre"), (L, "libre")]
    if mo == "enc_app":
        return [(0.0, "encastre"), (L, "articule")]
    if mo == "bi_enc":
        return [(0.0, "encastre"), (L, "encastre")]
    xs, x = [0.0], 0.0
    for t in el["travees"]:
        x += t
        xs.append(x)
    return [(x, "articule") for x in xs]


def _poutre(el: Dict, yb: float, K: Cadre) -> Tuple[str, float]:
    """La poutre, ses appuis, ses maintiens et ses cotes : (SVG, ordonnée du bas)."""
    X, Lt = K.X, K.Lt
    out = [_trait(f"M{X(0):.1f} {yb:.1f}H{X(Lt):.1f}", DP.ENCRE, 3.4)]
    for k, (x0, genre) in enumerate(_appuis(el)):
        x = X(x0)
        if genre == "encastre":
            g = x0 <= 0
            out.append(_trait(f"M{x:.1f} {yb - 18:.1f}V{yb + 18:.1f}", DP.ENCRE, 2.0))
            out += [_trait(f"M{x:.1f} {yb - 16 + 8 * j:.1f}l{-8 if g else 8} 7", DP.COTE, 0.9) for j in range(5)]
        elif genre == "articule":
            out.append(f'<path d="M{x:.1f} {yb + 2.5:.1f}l-8 13h16z" fill="#fff" stroke="{DP.ENCRE}" stroke-width="1.3" stroke-linejoin="round"/>')
            out.append(_trait(f"M{x - 12:.1f} {yb + 18.5:.1f}h24", DP.ENCRE, 1.3))
        if genre != "libre":
            out.append(_t(x, yb + 32, chr(ord("A") + k), 10.5, couleur=DP.MUET, gras=True))
    for mt in el["maintiens"]:
        x = X(mt["x"])
        haut = f'<path d="M{x:.1f} {yb - 3:.1f}l-4.5 -8h9z" fill="{DP.ACCENT}"/>'
        bas = f'<path d="M{x:.1f} {yb + 3:.1f}l-4.5 8h9z" fill="{DP.ACCENT}"/>'
        out.append({"complet": haut + bas, "haut": haut, "bas": bas}[mt["genre"]])
    yc = yb + 48
    x0 = 0.0
    for i, L in enumerate(el["travees"]):
        xa, xb = X(x0), X(x0 + L)
        out.append(_trait(f"M{xa:.1f} {yc:.1f}H{xb:.1f}M{xa:.1f} {yc - 4:.1f}v8M{xb:.1f} {yc - 4:.1f}v8", DP.COTE, 0.8))
        nom = "L" if len(el["travees"]) == 1 else f"L_{i + 1}"
        out.append(_t((xa + xb) / 2, yc - 5, f"{nom} = {C.fr(L, 2)} m", 10.5, couleur=DP.COTE))
        x0 += L
    return "".join(out), yc + 8


# ------------------------------------------------------------------ les enveloppes
def _diagramme(titre: str, unite: str, xs: Sequence[float], hauts: Sequence[float], bas: Sequence[float], y_haut: float, positif_bas: bool,
               couleur: str, K: Cadre) -> str:
    """Une enveloppe : l'axe, l'aire entre les deux courbes (max et min sur les combinaisons), ses deux extrêmes cotés."""
    X, Lt = K.X, K.Lt
    lo, hi = min(min(bas), 0.0), max(max(hauts), 0.0)
    k = H_DIAG / ((hi - lo) or 1.0)
    Y = (lambda v: y_haut + (v - lo) * k) if positif_bas else (lambda v: y_haut + (hi - v) * k)       # noqa: E731
    y0 = Y(0.0)
    out = [_t(8, y_haut + H_DIAG / 2 - 2, titre, 11.5, "start", couleur, True), _t(8, y_haut + H_DIAG / 2 + 11, unite, 9.0, "start", DP.MUET)]
    # l'aire va de l'axe à la courbe la plus éloignée : au-dessus de zéro sous la courbe des maximums, en dessous sous celle des minimums
    for vals, borne in ((hauts, max), (bas, min)):
        aire = (f"M{X(xs[0]):.1f} {y0:.1f} L" + " L".join(f"{X(x):.1f} {Y(borne(v, 0.0)):.1f}" for x, v in zip(xs, vals))
                + f" L{X(xs[-1]):.1f} {y0:.1f}Z")
        out.append(f'<path d="{aire}" fill="{couleur}" fill-opacity="0.13"/>')
    for vals in (hauts, bas):
        out.append(_trait("M" + " L".join(f"{X(x):.1f} {Y(v):.1f}" for x, v in zip(xs, vals)), couleur, 1.5))
    out.append(_trait(f"M{X(0):.1f} {y0:.1f}H{X(Lt):.1f}", DP.ENCRE, 0.9))
    for vals, choisir in ((hauts, max), (bas, min)):
        v = choisir(vals)
        if abs(v) < 1e-9 * max(abs(lo), abs(hi), 1e-300):
            continue
        x = X(xs[list(vals).index(v)])
        y = Y(v)
        dessus = y < y0
        ancre = K.ancre(x, 50)
        dx = 5 if ancre == "start" else (-5 if ancre == "end" else 0)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{couleur}"/>')
        out.append(_t(x + dx, y - 6 if dessus else y + 14, _nb(v, 1), 10.5, ancre, couleur, True))
    return "".join(out)


def enveloppes(r: Dict) -> Optional[Tuple[str, Dict]]:
    """La méthode dont on dessine les enveloppes (l'Eurocode quand il est rendu : les efforts de calcul) et ses
    enveloppes ; None sans statique."""
    env = (r.get("statique") or {}).get("enveloppes") or {}
    for k in ("ec3", "historique"):
        if k in env:
            return k, env[k]
    return None


def barre(ident: str, el: Dict, r: Dict, largeur: float = W, legende: bool = True) -> str:
    """Le dessin de la barre chargée : les cas de charge, la poutre, puis les enveloppes de M et de V. ``largeur`` : celle
    de la scène (plus étroite, les textes y sont plus grands) ; ``legende`` : la ligne qui dit ce qu'est un maintien."""
    K = Cadre(largeur, sum(el["travees"]))
    vmax = max((abs(ch["v"]) for c in el["cas"] for ch in c["charges"] if ch["type"] in ("q", "tri")), default=1.0) or 1.0
    corps, y = [], 12.0
    for c in el["cas"]:
        y += H_CAS
        corps.append(_bande_cas(c, y, K, vmax))
    yb = y + 40.0
    p, y = _poutre(el, yb, K)
    corps.append(p)
    env = enveloppes(r)
    if env:
        k, e = env
        quoi = "ELU" if k == "ec3" else "service"
        xs = e["x"]
        yM = y + 26
        corps.append(_diagramme(f"M ({quoi})", "kN·m", xs, e["M_max"], e["M_min"], yM, True, DIAG_M, K))
        yV = yM + H_DIAG + 34
        corps.append(_diagramme(f"V ({quoi})", "kN", xs, e["V_max"], e["V_min"], yV, False, DIAG_V, K))
        y = yV + H_DIAG + 14
    if el["maintiens"] and legende:
        # (sous la cote du minimum de V, qui s'écrit 14 sous le bas du diagramme — au milieu de la barre sur une poutre
        #  continue : la légende passait dessous, « −81,4 » se lisait sur elle ; trouvé à l'œil, 14/11/2026)
        y += 18
        corps.append(f'<path d="M{ML:.1f} {y - 3:.1f}l-4.5 -8h9z" fill="{DP.ACCENT}"/>')
        corps.append(_t(ML + 9, y - 4, "maintien latéral : au-dessus de la barre, la semelle haute ; en dessous, la basse ; les deux : complet", 9.5,
                        "start", DP.MUET))
        y += 6
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {K.w:.0f} {y + 6:.0f}" width="100%" role="img" '
            f'aria-label="La barre : ses cas de charge, ses appuis et ses maintiens, les enveloppes du moment et de l\'effort tranchant">'
            f'{"".join(corps)}</svg>')


# ------------------------------------------------------------------ les taux, méthode par méthode
def taux(ident: str, methodes: Sequence[Tuple[str, Dict]]) -> str:
    """Toutes les vérifications, une barre par vérification, la limite de 100 % tracée ; ``methodes`` : (titre, résultat)."""
    x0, x1 = 214.0, W - 62.0
    plein = 1.25                                                   # la barre la plus longue vaut 125 %
    larg = x1 - x0
    X = lambda t: x0 + larg * min(t, plein) / plein               # noqa: E731
    corps, y = [], 8.0
    for titre, x in methodes:
        y += 20
        corps.append(_t(8, y, titre, 12.0, "start", gras=True))
        corps.append(_t(W - 8, y, x["norme"], 9.5, "end", DP.MUET))
        y += 6
        for v in x["verifications"]:
            y += 20
            gov = v["id"] == x.get("gouvernant")
            corps.append(_t(x0 - 8, y + 4, v["titre"], 10.5, "end", DP.ENCRE if gov else DP.COTE, gov))
            corps.append(f'<rect x="{x0:.1f}" y="{y - 5:.1f}" width="{larg:.1f}" height="11" rx="3" fill="{DP.GRILLE}"/>')
            if v["taux"] is not None:
                niv = "r" if v["statut"] == RE.NOK else C.niveau(v["taux"])
                fond, encre = DP.NIVEAU[niv]
                corps.append(f'<rect x="{x0:.1f}" y="{y - 5:.1f}" width="{max(2.0, X(v["taux"]) - x0):.1f}" height="11" rx="3" fill="{fond}"/>')
                p = C.pct(v["taux"])
                txt = f"> {C.PLAFOND_TAUX} %" if p > C.PLAFOND_TAUX else f"{C.pc(p)} %"
                corps.append(_t(x1 + 6, y + 4, txt, 10.5, "start", encre, True))
            else:
                corps.append(_t(x1 + 6, y + 4, M.MOT_STATUT[v["statut"]].lower(), 9.5, "start", DP.MUET))
        y += 14
    haut = 30.0
    corps.insert(0, _trait(f"M{X(1.0):.1f} {haut:.1f}V{y - 6:.1f}", DP.ENCRE, 1.0, ' stroke-dasharray="4 3"'))
    corps.append(_t(X(1.0), y + 6, "100 %", 9.5, couleur=DP.MUET))
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {y + 14:.0f}" width="100%" role="img" '
            f'aria-label="Le taux de chaque vérification, méthode par méthode">{"".join(corps)}</svg>')
