# -*- coding: utf-8 -*-
"""Les DIAGRAMMES ENVELOPPES des bandes d'intégration de la note du « Générateur SAF — Fondation de grue » (lot 5, cahier des
charges du 15/11/2026, § 10.5, page Efforts) : M(s) avec M+ et M−, et en trait interrompu ce que reprend le treillis en bas
et en haut (m_Rd × largeur de la bande) ; V(s) avec son signe (V max au-dessus de l'axe, V min en dessous) et les abscisses
où V se lit (à d des zones chargées). Les valeurs les plus fortes sont écrites sur le diagramme.

Le dessin est fait EN POINTS : la note le pose à sa largeur exacte (une colonne), son texte fait donc 7,5 pt imprimé (§ 10.3 :
7 pt au moins). Ce fichier ne calcule rien : il dessine l'enveloppe des efforts (``saf.fondation_grue.efforts.enveloppe``)."""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from saf.fondation_grue import efforts as EF

from . import commun as C

POLICE = "Carlito, Calibri, sans-serif"
ENCRE, GRIS, GRILLE = "#15181f", "#6e7480", "#dee0e7"
M_PLUS, M_MOINS = "#1f5fa8", "#b45309"
TREILLIS = "#0f766e"
TRANCHANT, LECTURE = "#374151", "#9a3412"
CHARGE, PLOT = "#f1e7d6", "#e5e7eb"
TXT = 7.5                       # pt : le plus petit texte du diagramme
N = C.NBSP


def _t(x: float, y: float, txt: str, taille: float = TXT, couleur: str = ENCRE, ancre: str = "start", gras: bool = False) -> str:
    g = ' font-weight="600"' if gras else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{taille:.1f}" font-family="{POLICE}" text-anchor="{ancre}" fill="{couleur}"{g}>'
            f'{C.esc(txt)}</text>')


def _pas(etendue: float, n: int = 5) -> float:
    """Un pas « rond » (1, 2, 5 × 10ⁿ) pour environ ``n`` graduations."""
    brut = max(etendue, 1e-9) / n
    p = 10 ** np.floor(np.log10(brut))
    for f in (1, 2, 5, 10):
        if brut <= f * p + 1e-12:
            return float(f * p)
    return float(10 * p)


def _troncons(masque: np.ndarray) -> List[Tuple[int, int]]:
    out, i, n = [], 0, len(masque)
    while i < n:
        if masque[i]:
            j = i
            while j + 1 < n and masque[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


class _Cadre:
    """Le repère d'un diagramme : s (m) en abscisse, la grandeur en ordonnée (vers le haut)."""

    def __init__(self, w: float, h: float, L: float, ymin: float, ymax: float, gauche: float = 40.0, haut: float = 16.0, bas: float = 22.0):
        self.w, self.h, self.L = w, h, L
        self.x0, self.x1, self.y0, self.y1 = gauche, w - 8.0, haut, h - bas
        self.ymin, self.ymax = ymin, ymax

    def X(self, s: float) -> float:
        return self.x0 + (self.x1 - self.x0) * s / self.L

    def Y(self, v: float) -> float:
        return self.y1 - (self.y1 - self.y0) * (v - self.ymin) / (self.ymax - self.ymin)


def _axes(c: _Cadre, unite: str) -> str:
    """La grille, l'axe des s (m) et l'axe des valeurs, gradués."""
    out = []
    ps = 1.0 if c.L <= 10 else 2.0
    s = 0.0
    while s <= c.L + 1e-9:
        x = c.X(s)
        out.append(f'<line x1="{x:.1f}" y1="{c.y0:.1f}" x2="{x:.1f}" y2="{c.y1:.1f}" stroke="{GRILLE}" stroke-width="0.4"/>')
        out.append(_t(x, c.y1 + 9.5, C.nb(s), TXT, GRIS, "middle"))
        s += ps
    out.append(_t(c.x1, c.y1 + 19.0, "s (m)", TXT, GRIS, "end"))
    pv = _pas(c.ymax - c.ymin, 4)
    v = np.ceil(c.ymin / pv) * pv
    while v <= c.ymax + 1e-9:
        y = c.Y(v)
        out.append(f'<line x1="{c.x0:.1f}" y1="{y:.1f}" x2="{c.x1:.1f}" y2="{y:.1f}" stroke="{GRILLE}" stroke-width="0.4"/>')
        out.append(_t(c.x0 - 3, y + 2.6, C.fr(v + 0.0, 0), TXT, GRIS, "end"))
        v += pv
    y0 = c.Y(0.0)
    out.append(f'<line x1="{c.x0:.1f}" y1="{y0:.1f}" x2="{c.x1:.1f}" y2="{y0:.1f}" stroke="{ENCRE}" stroke-width="0.7"/>')
    out.append(_t(c.x0 - 3, c.y0 - 5, unite, TXT, GRIS, "end"))
    return "".join(out)


def _courbe(c: _Cadre, s: np.ndarray, v: np.ndarray, couleur: str, plein: bool = True) -> str:
    pts = " ".join(f"{c.X(a):.1f},{c.Y(b):.1f}" for a, b in zip(s, v))
    out = ""
    if plein:
        poly = f"{c.X(s[0]):.1f},{c.Y(0):.1f} {pts} {c.X(s[-1]):.1f},{c.Y(0):.1f}"
        out += f'<polygon points="{poly}" fill="{couleur}" fill-opacity="0.12" stroke="none"/>'
    return out + f'<polyline points="{pts}" fill="none" stroke="{couleur}" stroke-width="1.1" stroke-linejoin="round"/>'


def _zones(c: _Cadre, b: Dict) -> str:
    """Les zones chargées (sous les pieds, avec leur nom) et l'axe dans un plot coulé en place."""
    out = []
    s = b["s"]
    for i0, i1 in _troncons(np.asarray(b["dans_plot"], bool)):
        out.append(f'<rect x="{c.X(s[i0]):.1f}" y="{c.y0:.1f}" width="{c.X(s[i1]) - c.X(s[i0]):.1f}" height="{c.y1 - c.y0:.1f}" fill="{PLOT}"/>')
    pieds = EF.PIEDS.get(b["nom"], ())
    for k, (s1, s2) in enumerate(b["charge"]):
        a, z = max(0.0, s1), min(b["L"], s2)
        out.append(f'<rect x="{c.X(a):.1f}" y="{c.y1 - 3:.1f}" width="{max(c.X(z) - c.X(a), 0.8):.1f}" height="3" fill="{M_MOINS}" fill-opacity="0.55"/>')
        if k < len(pieds):
            out.append(_t(c.X((a + z) / 2), c.y0 - 5, f"P{pieds[k] + 1}", TXT, GRIS, "middle"))
    return "".join(out)


def _etiquette(c: _Cadre, s: float, v: float, txt: str, couleur: str, dessus: bool) -> str:
    """Une valeur écrite près de son point, dans le cadre."""
    x, y = c.X(s), c.Y(v)
    ancre = "start" if x < (c.x0 + c.x1) / 2 else "end"
    dx = 4 if ancre == "start" else -4
    yy = y - 4 if dessus else y + 9
    yy = min(max(yy, c.y0 + 8), c.y1 - 3)
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8" fill="{couleur}"/>'
            f'<text x="{x + dx:.1f}" y="{yy:.1f}" font-size="{TXT:.1f}" font-family="{POLICE}" font-weight="600" text-anchor="{ancre}" '
            f'fill="#ffffff" stroke="#ffffff" stroke-width="2.2" stroke-linejoin="round">{C.esc(txt)}</text>' + _t(x + dx, yy, txt, TXT, couleur, ancre, True))


def moments(b: Dict, mrd_inf: Optional[float], mrd_sup: Optional[float], ech_inf: float = 1.0, ech_sup: float = 1.0, w: float = 380.0,
            h: float = 150.0) -> str:
    """M(s) de la bande ``b`` (kN·m sur la largeur) : M+ et M− en enveloppe (× l'échelle SCIA de la face, § 8.2), le treillis du
    bas (+ m_Rd × b) et du haut (− m_Rd × b) en trait interrompu — sans m_Rd (hors du moteur Dalles), pas de trait —, M+ et M−
    les plus forts écrits (hors des plots coulés)."""
    s = np.asarray(b["s"], float)
    lb = b["largeur"]
    mp, mn = np.asarray(b["Mmax"], float) * ech_inf, np.asarray(b["Mmin"], float) * ech_sup
    traits = [] if mrd_inf is None or mrd_sup is None else [(mrd_inf * lb, "treillis bas"), (-mrd_sup * lb, "treillis haut")]
    tb, th = (traits[0][0], traits[1][0]) if traits else (0.0, 0.0)
    ymax = max(float(mp.max()), tb, 0.0) * 1.18 or 1.0
    ymin = min(float(mn.min()), th, 0.0) * 1.18 or -1.0
    c = _Cadre(w, h, b["L"], ymin, ymax)
    out = [_zones(c, b), _axes(c, "kN·m"), _courbe(c, s, mp, M_PLUS), _courbe(c, s, mn, M_MOINS)]
    hors = ~np.asarray(b["dans_plot"], bool)
    ip = int(np.argmax(np.where(hors, mp, -np.inf))) if hors.any() else None
    im = int(np.argmin(np.where(hors, mn, np.inf))) if hors.any() else None
    for v, mot in traits:
        y = c.Y(v)
        out.append(f'<line x1="{c.x0:.1f}" y1="{y:.1f}" x2="{c.x1:.1f}" y2="{y:.1f}" stroke="{TREILLIS}" stroke-width="0.9" stroke-dasharray="4 2.5"/>')
        # (le texte du treillis du côté OPPOSÉ à la valeur écrite de sa face : à droite tous les deux, ils se recouvraient
        #  — revue du 15/11/2026)
        i_ = ip if v > 0 else im
        gauche = i_ is not None and s[i_] > b["L"] / 2
        out.append(_t(c.x0 + 2 if gauche else c.x1 - 2, y - 2.5 if v > 0 else y + 8.5, f"{mot} {C.fr(v, 1)}", TXT, TREILLIS,
                      "start" if gauche else "end"))
    if hors.any():
        out.append(_etiquette(c, s[ip], mp[ip], f"M+ {C.fr(mp[ip], 1)} ({C.fr(mp[ip] / lb, 1)}/m)", M_PLUS, True))
        out.append(_etiquette(c, s[im], mn[im], f"M− {C.fr(mn[im], 1)} ({C.fr(mn[im] / lb, 1)}/m)", M_MOINS, False))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">{"".join(out)}</svg>'


COUVERT = "#0f766e"             # ce que reprennent les armatures (le treillis, puis le treillis + le renfort)


def _resistance(L: float, mrd: float, renforts: Sequence[Dict]) -> List[Tuple[float, float]]:
    """Les points (s, kN·m/m) de ce que reprennent les armatures d'une face le long de la bande : le treillis (``mrd``),
    et sur chaque renfort le treillis + le renfort (``mRd_pose``), atteint à un ancrage de chacune de ses extrémités — en
    rampe sur l'ancrage (la barre ne reprend tout qu'ancrée) ; au bord de la dalle, pas de rampe (``bord0``, ``bord1``)."""
    pts = [(0.0, mrd)]
    for r in renforts:
        a, z, an, T = r["s0"], r["s1"], max(r["ancrage"], 1e-9), r["mRd_pose"]

        def val(x, a=a, z=z, an=an, T=T, r=r):
            k = min(1.0, 1.0 if r["bord0"] else (x - a) / an, 1.0 if r["bord1"] else (z - x) / an)
            return mrd + (T - mrd) * max(k, 0.0)
        mid = (a + z) / 2
        xs = sorted({a, z, min(a + an, mid), max(z - an, mid)})
        pts.append((a, mrd))
        pts += [(x, val(x)) for x in xs]
        pts.append((z, mrd))
    pts.append((L, mrd))
    return pts


def couverture(b: Dict, cov: Dict, ech_inf: float = 1.0, ech_sup: float = 1.0, w: float = 380.0, h: float = 160.0,
               barres: Optional[Dict[str, str]] = None) -> str:
    """La COUVERTURE de M(s) de la bande ``b`` (lot 6) : M+ et M− en enveloppe (× l'échelle SCIA de la face), et ce que ses
    armatures reprennent (``saf.fondation_grue.sections.couverture``, × la largeur) — le treillis, et sur chaque renfort le
    treillis + le renfort, en rampe sur l'ancrage. Là où la courbe des moments passe au-delà, les armatures ne la couvrent
    pas. ``barres`` : {repère: « 4 Ø12 »}, écrit sur chaque renfort."""
    s = np.asarray(b["s"], float)
    lb = b["largeur"]
    mp, mn = np.asarray(b["Mmax"], float) * ech_inf, np.asarray(b["Mmin"], float) * ech_sup
    lignes = {face: [(x, v * lb * (1.0 if face == "inf" else -1.0)) for x, v in _resistance(b["L"], cov[face]["mRd"], cov[face]["renforts"])]
              for face in ("inf", "sup")}
    ymax = max(float(mp.max()), max(v for _, v in lignes["inf"]), 0.0) * 1.22 or 1.0
    ymin = min(float(mn.min()), min(v for _, v in lignes["sup"]), 0.0) * 1.22 or -1.0
    c = _Cadre(w, h, b["L"], ymin, ymax)
    out = [_zones(c, b), _axes(c, "kN·m"), _courbe(c, s, mp, M_PLUS), _courbe(c, s, mn, M_MOINS)]
    for face in ("inf", "sup"):
        pts = " ".join(f"{c.X(x):.1f},{c.Y(v):.1f}" for x, v in lignes[face])
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COUVERT}" stroke-width="1.0" stroke-dasharray="4 2.5" stroke-linejoin="round"/>')
        haut = face == "inf"
        mrd = cov[face]["mRd"] * lb * (1.0 if haut else -1.0)
        # (le treillis écrit au bord gauche, les renforts sur leur palier : rien ne se recouvre)
        out.append(_t(c.x0 + 2, c.Y(mrd) - 2.5 if haut else c.Y(mrd) + 8.5, f"treillis {C.fr(mrd, 1)}", TXT, COUVERT, "start"))
        for r in cov[face]["renforts"]:
            T = r["mRd_pose"] * lb * (1.0 if haut else -1.0)
            x = c.X((r["s0"] + r["s1"]) / 2)
            mot = f"{r['rep']}" + (f" {barres[r['rep']]}" if barres and r["rep"] in barres else "") + f" : {C.fr(T, 1)}"
            y = c.Y(T) - 2.5 if haut else c.Y(T) + 8.5
            out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{TXT:.1f}" font-family="{POLICE}" text-anchor="middle" fill="#ffffff" '
                       f'stroke="#ffffff" stroke-width="2.2" stroke-linejoin="round">{C.esc(mot)}</text>' + _t(x, y, mot, TXT, COUVERT, "middle"))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">{"".join(out)}</svg>'


def legende_couverture(w: float = 380.0, plot: bool = False) -> str:
    """La légende du diagramme de couverture : une ligne ; une seconde avec des plots coulés en place (``plot``) — dans leur
    zone grise, M n'est pas comparé au treillis : le plot est hors du principe, son ferraillage est à prévoir."""
    items = [(M_PLUS, "M+ (bas tendu)", ""), (M_MOINS, "M− (haut tendu)", ""),
             (COUVERT, "ce que reprennent le treillis et les renforts (× b)", "4 2.5")]
    out, x = [], 2.0
    for coul, mot, tirets in items:
        da = f' stroke-dasharray="{tirets}"' if tirets else ""
        out.append(f'<line x1="{x:.1f}" y1="8" x2="{x + 14:.1f}" y2="8" stroke="{coul}" stroke-width="1.4"{da}/>')
        out.append(_t(x + 17, 10.6, mot, TXT, ENCRE))
        x += 22 + len(mot) * 3.6
    h = 16
    if plot:
        out.append(f'<rect x="2" y="20" width="14" height="8" fill="{PLOT}"/>')
        out.append(_t(19, 27.2, "plot coulé en place : hors du principe, son ferraillage est à prévoir", TXT, ENCRE))
        h = 31
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h}" width="{w:.0f}" height="{h}">{"".join(out)}</svg>'


def tranchant(b: Dict, w: float = 380.0, h: float = 130.0, d_v: Optional[float] = None) -> str:
    """V(s) de la bande ``b`` (kN sur la largeur) AVEC SON SIGNE, comme M(s) : l'enveloppe V max au-dessus de l'axe, V min en
    dessous (V a le signe de dM/ds : positif avant un pied, négatif après) ; les abscisses où V se lit (à d des zones
    chargées : bords des tronçons « ok_v ») ; « V à d » écrit sur la courbe où il est lu (le plus grand de |V max| et
    |V min| à ces abscisses)."""
    s = np.asarray(b["s"], float)
    vp, vn = np.asarray(b["Vmax"], float), np.asarray(b["Vmin"], float)
    ok = np.asarray(b["ok_v"], bool)
    ymax = max(float(vp.max()), 0.0, 1.0) * 1.2
    ymin = min(float(vn.min()), 0.0, -1.0) * 1.2
    c = _Cadre(w, h, b["L"], ymin, ymax)
    out = [_zones(c, b), _axes(c, "kN"), _courbe(c, s, vp, TRANCHANT), _courbe(c, s, vn, TRANCHANT)]
    for i0, i1 in _troncons(ok):
        for i in (i0, i1):
            if 0 < i < len(s) - 1:
                x = c.X(s[i])
                out.append(f'<line x1="{x:.1f}" y1="{c.y0:.1f}" x2="{x:.1f}" y2="{c.y1:.1f}" stroke="{LECTURE}" stroke-width="0.8" stroke-dasharray="2 2"/>')
                out.append(_t(x, c.y0 + 7.5, "d", TXT, LECTURE, "middle"))
    if b.get("V_d") is not None and ok.any():
        va = np.maximum(vp, -vn)
        iv = int(np.argmax(np.where(ok, va, -np.inf)))
        haut = vp[iv] >= -vn[iv]                          # (la courbe où la valeur est lue : V max, ou V min)
        out.append(_etiquette(c, s[iv], vp[iv] if haut else vn[iv], f"V à d {C.fr(va[iv], 1)} ({C.fr(va[iv] / b['largeur'], 1)}/m)",
                              LECTURE, haut))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">{"".join(out)}</svg>'


def legende(w: float = 380.0) -> str:
    """La légende commune des diagrammes (une ligne)."""
    items = [(M_PLUS, "M+ (bas tendu)", ""), (M_MOINS, "M− (haut tendu)", ""), (TREILLIS, "ce que reprend le treillis (m_Rd × b)", "4 2.5"),
             (LECTURE, "abscisses de V à d", "2 2")]
    out, x = [], 2.0
    for coul, mot, tirets in items:
        da = f' stroke-dasharray="{tirets}"' if tirets else ""
        out.append(f'<line x1="{x:.1f}" y1="8" x2="{x + 14:.1f}" y2="8" stroke="{coul}" stroke-width="1.4"{da}/>')
        out.append(_t(x + 17, 10.6, mot, TXT, ENCRE))
        x += 22 + len(mot) * 3.6
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} 16" width="{w:.0f}" height="16">{"".join(out)}</svg>'
