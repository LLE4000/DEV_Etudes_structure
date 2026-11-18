# -*- coding: utf-8 -*-
"""Les dessins d'une BARRE CHARGÉE du module « Choix de profilé » : ses cas de charge (une bande par cas), la poutre —
ses appuis, ses maintiens, ses raidisseurs, ses portées —, puis les enveloppes du moment et de l'effort tranchant ; et
les TAUX de toutes les vérifications, méthode par méthode.

Rien n'est calculé ici : les enveloppes viennent du moteur (``resultat["statique"]``), les taux de ses vérifications ;
ce fichier les POSE. Du SVG en unités d'écran, aux couleurs des dessins du module (api_web/dessin_profil). La scène de la
barre a la largeur qu'on lui donne (640 à l'écran) : la note PDF la demande plus étroite, pour que ses textes restent
lisibles dans une colonne de 10 cm — le même dessin, le même code (ndc_pdf/svg.py le peint).
"""
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

from acier.barre import resultat as RE
from acier.barre import statique as ST

from . import commun as C
from . import dessin_profil as DP
from . import profil_mots as M

W = 640.0                        # la largeur de la scène à l'écran
ML, MR = 78.0, 30.0              # à gauche : le nom d'un cas et sa nature, le nom d'un diagramme
H_CAS = 44.0                     # la hauteur de la bande d'un cas de charge
H_DIAG = 84.0
CHARGE, CHARGE_FOND, DIAG_M, DIAG_V = "#9C3341", "#F6E3E6", "#2B4BCF", "#0F766E"
RAIDISSEUR = "#B45309"           # un raidisseur transversal : un plat debout, à cheval sur la poutre
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


# ------------------------------------------------------------------ (lot C, C4) les repères : un SYMBOLE dessiné et des mots
def _signe_niveau(niv: str, x: float, y: float, couleur: str = DP.ENCRE) -> str:
    """Le signe du niveau d'une charge, centré en (x, y) — ▲ semelle haute, ● centre, ▼ semelle basse : une FORME, lisible
    en noir et blanc (brief § 15.5), dessinée plutôt qu'écrite (une police sans ces glyphes ne la perd pas)."""
    if niv == "haut":
        return f'<path d="M{x:.1f} {y - 3.6:.1f}l3.6 6.4h-7.2z" fill="{couleur}"/>'
    if niv == "bas":
        return f'<path d="M{x:.1f} {y + 3.6:.1f}l3.6 -6.4h-7.2z" fill="{couleur}"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{couleur}"/>'


def _niveaux_meles(el: Dict) -> bool:
    """Les charges de la barre sont-elles à des niveaux différents ? (Le signe n'est dessiné qu'alors : sinon le niveau de
    la barre, dit une fois dans les données, suffit.)"""
    return len({ch.get("niveau") or el["niveau"] for c in el["cas"] for ch in c["charges"]}) > 1


def _losange(x: float, y: float) -> str:
    """Le repère d'un point vérifié : un losange CREUX sur l'axe de la barre."""
    return f'<path d="M{x:.1f} {y - 4.5:.1f}l4.5 4.5l-4.5 4.5l-4.5 -4.5z" fill="#fff" stroke="{DP.ENCRE}" stroke-width="1.2"/>'


def _maintien(x: float, yb: float, genre: str, neglige: bool) -> str:
    """Un maintien latéral : un triangle au-dessus (semelle haute), en dessous (semelle basse), les deux (complet) ; CREUX
    quand il est négligé dans M_cr (lot C, C1 : il tient la barre, le moment critique ne le compte pas)."""
    peint = f'fill="#fff" stroke="{DP.ACCENT}" stroke-width="1.2"' if neglige else f'fill="{DP.ACCENT}"'
    haut = f'<path d="M{x:.1f} {yb - 3:.1f}l-4.5 -8h9z" {peint}/>'
    bas = f'<path d="M{x:.1f} {yb + 3:.1f}l-4.5 8h9z" {peint}/>'
    return {"complet": haut + bas, "haut": haut, "bas": bas}[genre]


# ------------------------------------------------------------------ les cas de charge
def _largeur(texte: str, fs: float) -> float:
    """La largeur d'un texte gras, estimée (un dessin n'a pas la métrique de la police) : 0,58 du corps par caractère."""
    return 0.58 * fs * len(texte)


SIGNE = 11.0                     # (lot C, C4) la place du signe de niveau derrière la valeur d'une charge


def _N_troncons(c: Dict, y: float, K: Cadre) -> List[str]:
    """(lot C, C4) L'effort normal d'un cas PAR TRONÇON (``N_troncons`` : [{"x2", "N"}…], le premier part du bout gauche) :
    sous la ligne du cas, un trait de cote par tronçon et sa valeur au milieu — « N = 50 kN », « 50 » si le tronçon est
    trop court pour le mot, rien s'il l'est même pour le nombre (sa valeur se lit dans la saisie)."""
    out, x1 = [], 0.0
    for t in c["N_troncons"]:
        x2 = min(float(t["x2"]), K.Lt)
        a, b = K.X(x1), K.X(x2)
        out.append(_trait(f"M{a + 1:.1f} {y + 5:.1f}V{y + 9:.1f}H{b - 1:.1f}V{y + 5:.1f}", CHARGE, 0.8))
        for txt in (f"N = {M.n(t['N'], 1)} kN", M.n(t["N"], 1)):
            if _largeur(txt, 9.0) + 4 <= b - a:
                out.append(_t((a + b) / 2, y + 18, txt, 9.0, couleur=CHARGE))
                break
        x1 = x2
    return out


def _bande_cas(c: Dict, y: float, K: Cadre, vmax: float, niv_defaut: Optional[str] = None) -> str:
    """La bande d'un cas : son nom à gauche, ses charges sur une ligne de base ; ``y`` : la ligne de base. Les valeurs
    des charges ponctuelles et des couples sont placées d'abord ; celle d'une charge répartie se met au milieu, sinon à
    un bout de la charge, là où elle ne recouvre aucune valeur déjà écrite. Les charges réparties sont peintes dessous,
    les flèches des charges ponctuelles et toutes les valeurs dessus. (lot C, C4) ``niv_defaut`` : le niveau de la barre,
    quand ses charges sont à des niveaux MÊLÉS — chaque valeur porte alors le signe du sien (▲ ● ▼) ; l'effort normal se
    dessine par tronçon quand il en a."""
    X, Lt, W = K.X, K.Lt, K.w
    signe = SIGNE if niv_defaut else 0.0
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
        s_ = signe if t == "P" else 0.0                           # (un couple n'a pas de niveau : il ne charge pas l'âme)
        x1, x2 = (x + dx, x + dx + w + s_) if ancre == "start" else (x - dx - w - s_, x - dx)
        if any(o["type"] in ("q", "tri") and X(o["a"]) < x2 and X(o["b"]) > x1 for o in c["charges"]):
            dy = 32.0                                             # au-dessus d'une charge répartie : la valeur passe par-dessus son bord
        dessus.append(_t(x + (dx if ancre == "start" else -dx - s_), y - dy, txt, 10.0, ancre, CHARGE, True))
        if s_:
            dessus.append(_signe_niveau(ch.get("niveau") or niv_defaut, x2 - 4.0 if ancre == "start" else x - dx - 4.0, y - dy - 3.5, CHARGE))
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
        w = _largeur(txt, 10.0) + signe                           # (le signe de niveau, derrière la valeur)
        milieu = (xa + xb) / 2
        xm = next((x_ for x_ in (milieu, xa + w / 2 + 3, xb - w / 2 - 3)
                   if all(x_ + w / 2 + 4 < a or x_ - w / 2 - 4 > b for a, b in pris)), milieu)
        yt = y + s * h - 4 if bas else y + s * h + 12
        dessus.append(_t(xm - signe / 2, yt, txt, 10.0, couleur=CHARGE, gras=True))
        if signe:
            dessus.append(_signe_niveau(ch.get("niveau") or niv_defaut, xm + w / 2 - 4.0, yt - 3.5, CHARGE))
        pris.append((xm - w / 2, xm + w / 2))
    if c.get("N_troncons"):
        dessus += _N_troncons(c, y, K)
    elif c.get("N"):
        dessus.append(_t(W - 4, y + 12, f"N = {M.n(c['N'], 1)} kN", 9.5, "end", CHARGE))
    return "".join(out + fonds + dessus)


# ------------------------------------------------------------------ la poutre
N_TRONCONS = 16.0               # (lot C, C4) la place de plus sous un cas dont l'effort normal est donné par tronçon


def _appuis(el: Dict) -> List[Tuple[float, str]]:
    """Les appuis du modèle : (abscisse, « articule » | « encastre » | « libre »). (lot C, C4) Leurs abscisses sont celles
    du moteur (``statique.appuis_x`` : depuis le bout gauche, débord compris) et la barre a sa longueur entière
    (``statique.longueur`` : les débords en plus des portées)."""
    L = ST.longueur(el)
    xs = ST.appuis_x(el)
    mo = el["modele"]
    if mo == "console":
        return [(0.0, "encastre"), (L, "libre")]
    if mo == "enc_app":
        return [(xs[0], "encastre"), (xs[-1], "articule")]
    if mo == "bi_enc":
        return [(xs[0], "encastre"), (xs[-1], "encastre")]
    return [(x, "articule") for x in xs]


def _plat(x: float, y: float) -> str:
    """Le signe d'un raidisseur : un plat debout, centré en (x, y)."""
    return f'<rect x="{x - 1.9:.1f}" y="{y - 8:.1f}" width="3.8" height="16" rx="0.8" fill="{RAIDISSEUR}" stroke="#fff" stroke-width="0.7"/>'


def _poutre(el: Dict, yb: float, K: Cadre, raidisseurs: Sequence[float] = ()) -> Tuple[str, float]:
    """La poutre, ses appuis, ses maintiens, ses raidisseurs (leurs abscisses) et ses cotes : (SVG, ordonnée du bas)."""
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
        out.append(_maintien(X(mt["x"]), yb, mt["genre"], bool(mt.get("neglige"))))
    out += [_plat(X(x), yb) for x in raidisseurs]
    # (lot C, C4) les points vérifiés : un losange creux sur l'axe, et sa lettre au-dessus (p1, p2…) ; deux lettres trop
    # proches l'une de l'autre se superposent en hauteur, jamais l'une sur l'autre
    dernier = -1e9
    hauteur = 0
    for k, xp in enumerate(el.get("points") or []):
        x = X(xp)
        hauteur = hauteur + 1 if x - dernier < 16 else 0
        dernier = x
        out.append(_losange(x, yb))
        out.append(_t(x, yb - 16 - 10 * hauteur, f"p{k + 1}", 9.0, couleur=DP.ENCRE, gras=True))
    # (lot C, C4) la position de la barre, quand elle n'est pas horizontale : dans la marge, sur la ligne de la poutre
    if el.get("position") and el["position"] != "horizontale":
        pos = "verticale" if el["position"] == "verticale" else f"inclinée {M.n(el.get('angle') or 0.0, 1)}°"
        out.append(_t(8, yb + 4, pos, 9.5, "start", DP.MUET))
    yc = yb + 48
    g, d = (el.get("debords") or (0.0, 0.0))[:2]
    x0 = g
    for i, L in enumerate(el["travees"]):
        xa, xb = X(x0), X(x0 + L)
        out.append(_trait(f"M{xa:.1f} {yc:.1f}H{xb:.1f}M{xa:.1f} {yc - 4:.1f}v8M{xb:.1f} {yc - 4:.1f}v8", DP.COTE, 0.8))
        nom = "L" if len(el["travees"]) == 1 else f"L_{i + 1}"
        out.append(_t((xa + xb) / 2, yc - 5, f"{nom} = {C.fr(L, 2)} m", 10.5, couleur=DP.COTE))
        x0 += L
    # (lot C, C4) les débords : leur cote prolonge celle des portées, leur valeur dessous (la place au-dessus est aux portées)
    bas = yc + 8
    for a_, b_ in ((0.0, g), (g + sum(el["travees"]), g + sum(el["travees"]) + d)):
        if b_ - a_ <= 1e-9:
            continue
        xa, xb = X(a_), X(b_)
        out.append(_trait(f"M{xa:.1f} {yc:.1f}H{xb:.1f}M{xa:.1f} {yc - 4:.1f}v8M{xb:.1f} {yc - 4:.1f}v8", DP.COTE, 0.8))
        xm = (xa + xb) / 2
        out.append(_t(xm, yc + 13, f"débord {C.fr(b_ - a_, 2)} m", 9.5, K.ancre(xm, 40), DP.COTE))
        bas = yc + 18
    return "".join(out), bas


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
    # (chaque trait part de l'AXE et y retombe : l'effort tranchant d'une barre redescend à zéro à ses extrémités — un
    # trait arrêté en l'air à V(0) et V(L) se lisait comme une barre qui continue ; demande du propriétaire, 14/11/2026)
    for vals in (hauts, bas):
        out.append(_trait(f"M{X(xs[0]):.1f} {y0:.1f} L" + " L".join(f"{X(x):.1f} {Y(v):.1f}" for x, v in zip(xs, vals))
                          + f" L{X(xs[-1]):.1f} {y0:.1f}", couleur, 1.5))
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
    K = Cadre(largeur, ST.longueur(el))                  # (lot C, C4 : débords compris)
    vmax = max((abs(ch["v"]) for c in el["cas"] for ch in c["charges"] if ch["type"] in ("q", "tri")), default=1.0) or 1.0
    meles = _niveaux_meles(el)
    corps, y = [], 12.0
    for c in el["cas"]:
        y += H_CAS
        corps.append(_bande_cas(c, y, K, vmax, el["niveau"] if meles else None))
        if c.get("N_troncons"):
            y += N_TRONCONS
    yb = y + 40.0
    raidis = [L["x"] for L in M.raidis(r.get("ame") or {})]
    p, y = _poutre(el, yb, K, raidis)
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
    if raidis and legende:
        y += 16 if el["maintiens"] else 18
        corps.append(_plat(ML, y - 7))
        corps.append(_t(ML + 9, y - 4, "raidisseur transversal : une paire de plats, de part et d'autre de l'âme", 9.5, "start", DP.MUET))
        y += 6
    # (lot C, C4) les autres repères, chacun par son SIGNE et ses mots (brief § 15.5 : lisible en noir et blanc)
    if legende and any(mt.get("neglige") for mt in el["maintiens"]):
        y += 16
        corps.append(_maintien(ML, y - 4, "haut", True))
        corps.append(_t(ML + 9, y - 4, "maintien creux : présent, mais négligé dans M_cr", 9.5, "start", DP.MUET))
        y += 6
    if legende and meles:
        y += 16
        xg = ML
        for niv in ("haut", "centre", "bas"):
            corps.append(_signe_niveau(niv, xg, y - 7.5, CHARGE))
            mot = M.NOM_NIVEAU[niv]
            corps.append(_t(xg + 7, y - 4, mot, 9.5, "start", DP.MUET))
            xg += 7 + _largeur(mot, 9.5) * 0.9 + 16
        y += 6
    if legende and el.get("points"):
        y += 16
        corps.append(_losange(ML, y - 7.5))
        corps.append(_t(ML + 9, y - 4, "p1, p2… : points vérifiés en plus des sections critiques", 9.5, "start", DP.MUET))
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
        if x["norme"] not in titre:                                  # (« Méthode ELS (NBN B 51-001) » la porte déjà)
            corps.append(_t(W - 8, y, x["norme"], 9.5, "end", DP.MUET))
        y += 6
        for v in x["verifications"]:
            y += 20
            gov = v["id"] == x.get("gouvernant")
            corps.append(_t(x0 - 8, y + 4, v["titre"], 10.5, "end", DP.ENCRE if gov else DP.COTE, gov))
            corps.append(f'<rect x="{x0:.1f}" y="{y - 5:.1f}" width="{larg:.1f}" height="11" rx="3" fill="{DP.GRILLE}"/>')
            if v["taux"] is not None:
                niv = M.niveau(v)                                    # (hors domaine : « v », jamais la couleur « conforme »)
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
