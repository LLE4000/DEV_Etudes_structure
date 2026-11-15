# -*- coding: utf-8 -*-
"""Les DIAGRAMMES ENVELOPPES des bandes d'intégration de la note du « Générateur SAF — Fondation de grue » (lot 5, cahier des
charges du 15/11/2026, § 10.5, page Efforts) : M(s) avec M+ et M−, et en trait interrompu ce que reprend le treillis en bas
et en haut (m_Rd × largeur de la bande) ; V(s) avec les abscisses où V se lit (à d des zones chargées). Les valeurs les plus
fortes sont écrites sur le diagramme.

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
            f'<text x="{x + dx:.1f}" y="{yy:.1f}" font-size="{TXT:.1f}" font-family="{POLICE}" text-anchor="{ancre}" fill="none" '
            f'stroke="#ffffff" stroke-width="2.2" stroke-linejoin="round">{C.esc(txt)}</text>' + _t(x + dx, yy, txt, TXT, couleur, ancre, True))


def moments(b: Dict, mrd_inf: float, mrd_sup: float, ech_inf: float = 1.0, ech_sup: float = 1.0, w: float = 380.0, h: float = 150.0) -> str:
    """M(s) de la bande ``b`` (kN·m sur la largeur) : M+ et M− en enveloppe (× l'échelle SCIA de la face, § 8.2), le treillis du
    bas (+ m_Rd × b) et du haut (− m_Rd × b) en trait interrompu, M+ et M− les plus forts écrits (hors des plots coulés)."""
    s = np.asarray(b["s"], float)
    lb = b["largeur"]
    mp, mn = np.asarray(b["Mmax"], float) * ech_inf, np.asarray(b["Mmin"], float) * ech_sup
    tb, th = mrd_inf * lb, -mrd_sup * lb
    ymax = max(float(mp.max()), tb, 0.0) * 1.18 or 1.0
    ymin = min(float(mn.min()), th, 0.0) * 1.18 or -1.0
    c = _Cadre(w, h, b["L"], ymin, ymax)
    out = [_zones(c, b), _axes(c, "kN·m"), _courbe(c, s, mp, M_PLUS), _courbe(c, s, mn, M_MOINS)]
    for v, mot in ((tb, "treillis bas"), (th, "treillis haut")):
        y = c.Y(v)
        out.append(f'<line x1="{c.x0:.1f}" y1="{y:.1f}" x2="{c.x1:.1f}" y2="{y:.1f}" stroke="{TREILLIS}" stroke-width="0.9" stroke-dasharray="4 2.5"/>')
        out.append(_t(c.x1 - 2, y - 2.5 if v > 0 else y + 8.5, f"{mot} {C.fr(v, 1)}", TXT, TREILLIS, "end"))
    hors = ~np.asarray(b["dans_plot"], bool)
    if hors.any():
        ip = int(np.argmax(np.where(hors, mp, -np.inf)))
        im = int(np.argmin(np.where(hors, mn, np.inf)))
        out.append(_etiquette(c, s[ip], mp[ip], f"M+ {C.fr(mp[ip], 1)} ({C.fr(mp[ip] / lb, 1)}/m)", M_PLUS, True))
        out.append(_etiquette(c, s[im], mn[im], f"M− {C.fr(mn[im], 1)} ({C.fr(mn[im] / lb, 1)}/m)", M_MOINS, False))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">{"".join(out)}</svg>'


def tranchant(b: Dict, w: float = 380.0, h: float = 110.0, d_v: Optional[float] = None) -> str:
    """V(s) de la bande ``b`` (|V|, kN sur la largeur), les abscisses où V se lit (à d des zones chargées : bords des
    tronçons « ok_v » de l'enveloppe), V à d le plus fort écrit."""
    s = np.asarray(b["s"], float)
    v = np.asarray(b["Vabs"], float)
    ok = np.asarray(b["ok_v"], bool)
    ymax = max(float(v.max()), 1.0) * 1.2
    c = _Cadre(w, h, b["L"], 0.0, ymax)
    out = [_zones(c, b), _axes(c, "kN"), _courbe(c, s, v, TRANCHANT)]
    for i0, i1 in _troncons(ok):
        for i in (i0, i1):
            if 0 < i < len(s) - 1:
                x = c.X(s[i])
                out.append(f'<line x1="{x:.1f}" y1="{c.y0:.1f}" x2="{x:.1f}" y2="{c.y1:.1f}" stroke="{LECTURE}" stroke-width="0.8" stroke-dasharray="2 2"/>')
                out.append(_t(x, c.y0 + 7.5, "d", TXT, LECTURE, "middle"))
    if b.get("V_d") is not None and ok.any():
        iv = int(np.argmax(np.where(ok, v, -np.inf)))
        out.append(_etiquette(c, s[iv], v[iv], f"V à d {C.fr(v[iv], 1)} ({C.fr(v[iv] / b['largeur'], 1)}/m)", LECTURE, True))
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
