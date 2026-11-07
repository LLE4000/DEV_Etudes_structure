# -*- coding: utf-8 -*-
"""Les dessins du module « Rigidité du sol » : les graphiques de l'écran Streamlit (modules/rigidite_sol.py, tracés par
matplotlib) refaits en SVG — axes, graduations, quadrillage, légendes — et les coupes à l'échelle (semelle sur son
profil, sondages côte à côte, plat en béton).

Rien n'est calculé ici : chaque série vient du moteur (sol/rigidite.py : donnees_diffusion, k_fonction_B, donnees_cpt,
theories, abaque, plat) ; ce fichier choisit les échelles et dessine. Les couleurs sont celles de l'écran Streamlit
(Δσ rouge, σ'v0 ardoise, critère vert, nappe bleue, k violet, qc bleu, fs ambre, Ic sarcelle) ; les traits, les cotes
et les textes suivent les dessins du garde-corps et de la poutre (encre #15181F, cotes grises, police système).

Tout texte qui vient de la note (nom de sondage, type de sol) est échappé ici.
"""
import math
import re
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

ENCRE, GRILLE, AXE, DOUX = "#15181F", "#E8EBF0", "#7F8C9A", "#5B6576"
ROUGE, ARDOISE, VERT, AMBRE, BLEU, VIOLET, BLEU_QC, SARCELLE = ("#b91c1c", "#334155", "#059669", "#b45309", "#2563eb", "#7c3aed",
                                                                 "#1d4ed8", "#0f766e")
BETON, MORTIER, PLAT = "#E6EAEF", "#D9CBB0", "#9AA6B5"
POLICE = "system-ui,Arial,sans-serif"


# ------------------------------------------------------------------ petits outils
def _f(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def nombre(x: float, nd: int = 2) -> str:
    """Un nombre à la française pour une graduation ou une étiquette : virgule, espace fine des milliers, zéros inutiles
    retirés."""
    s = f"{x:,.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(",", "\u202f").replace(".", ",")


def _m(t: str) -> str:
    """Un texte FIXE du dessin (jamais un nom venu de la note) avec ses indices écrits en indice : « q_net » → q puis
    « net » abaissé et réduit, « σ'v0 » → σ' puis « v0 » en indice. Le décalage vertical est rendu par des tspan
    imbriqués (dy), lus par tous les navigateurs."""
    morceaux = re.split(r"(_[A-Za-z0-9]+)", t.replace("σ'v0", "σ'_v0"))
    out, ouverts = "", 0
    for m in morceaux:
        if m.startswith("_") and len(m) > 1:
            out += f'<tspan font-size="76%" dy="3.5">{m[1:]}</tspan><tspan dy="-3.5">'
            ouverts += 1
        else:
            out += m
    return out + "</tspan>" * ouverts


def _paragraphe(x, y, texte: str, car: int = 100, taille: float = 11.5, couleur=None, ancre="start") -> Tuple[str, int]:
    """Une légende de plusieurs lignes (le texte coupé aux mots, ``car`` caractères au plus par ligne) ; rend le SVG et le
    nombre de lignes. Le texte est échappé ici."""
    lignes, cur = [], ""
    for mot in texte.split():
        if cur and len(cur) + 1 + len(mot) > car:
            lignes.append(cur)
            cur = mot
        else:
            cur = f"{cur} {mot}" if cur else mot
    if cur:
        lignes.append(cur)
    pas = taille + 5
    svg = "".join(_txt(x, y + k * pas, escape(li), taille, ancre, couleur or DOUX) for k, li in enumerate(lignes))
    return svg, len(lignes)


def _txt(x, y, t, taille=13.0, ancre="middle", couleur=ENCRE, gras=False, extra="") -> str:
    poids = ' font-weight="600"' if gras else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{extra}>{t}</text>')


def _ligne(x1, y1, x2, y2, couleur=ENCRE, ep=1.0, tirets="", extra="") -> str:
    d = f' stroke-dasharray="{tirets}"' if tirets else ""
    return f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(ep)}"{d}{extra}/>'


def _poly(points: Sequence[Tuple[float, float]], couleur, ep=1.5, tirets="", fond="none", opacite=1.0) -> str:
    if not points:
        return ""
    p = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
    d = f' stroke-dasharray="{tirets}"' if tirets else ""
    o = f' fill-opacity="{_f(opacite)}"' if fond != "none" and opacite != 1.0 else ""
    trait = f'stroke="{couleur}" stroke-width="{_f(ep)}"' if couleur != "none" else 'stroke="none"'
    return f'<polyline points="{p}" fill="{fond}"{o} {trait}{d} stroke-linejoin="round" stroke-linecap="round"/>'


def _pas(etendue: float, n: int) -> float:
    brut = etendue / max(n, 1)
    p = 10 ** math.floor(math.log10(brut))
    for m in (1, 2, 2.5, 5, 10):
        if brut <= m * p * (1 + 1e-9):
            return m * p
    return 10 * p


def graduations(lo: float, hi: float, n: int = 5) -> List[float]:
    """Des graduations « rondes » (1, 2, 2,5 ou 5 × 10ⁿ) qui couvrent [lo ; hi]."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        hi = lo + 1.0
    s = _pas(hi - lo, n)
    a = math.floor(lo / s + 1e-9)
    b = math.ceil(hi / s - 1e-9)
    return [round(k * s, 10) for k in range(a, b + 1)]


def _nd(pas: float) -> int:
    """Le nombre de décimales d'une graduation de pas donné."""
    for nd in range(0, 5):
        if abs(round(pas * 10 ** nd) - pas * 10 ** nd) < 1e-6:
            return nd
    return 4


class Cadre:
    """Un graphique : un rectangle (x0, y0, l, h) en unités du dessin, deux échelles, l'axe vertical éventuellement
    retourné (la profondeur croît vers le bas), l'axe horizontal éventuellement logarithmique."""

    def __init__(self, x0, y0, l, h, xlo, xhi, ylo, yhi, y_bas=False, x_log=False):
        self.x0, self.y0, self.l, self.h = x0, y0, l, h
        self.xlo, self.xhi, self.ylo, self.yhi = xlo, xhi, ylo, yhi
        self.y_bas, self.x_log = y_bas, x_log

    def X(self, v: float) -> float:
        if self.x_log:
            a, b = math.log10(self.xlo), math.log10(self.xhi)
            return self.x0 + (math.log10(max(v, self.xlo)) - a) / (b - a) * self.l
        return self.x0 + (v - self.xlo) / (self.xhi - self.xlo) * self.l

    def Y(self, v: float) -> float:
        t = (v - self.ylo) / (self.yhi - self.ylo)
        return self.y0 + (t if self.y_bas else 1 - t) * self.h

    def clip(self, ident: str) -> str:
        return (f'<clipPath id="{ident}"><rect x="{_f(self.x0)}" y="{_f(self.y0)}" width="{_f(self.l)}" height="{_f(self.h)}"/>'
                f'</clipPath>')

    def axes(self, gx: Optional[List[float]], gy: Optional[List[float]], titre_x="", titre_y="", nd_x=None, nd_y=None,
             etiquettes_y=True, taille=12.0) -> str:
        """Le quadrillage, le cadre, les graduations chiffrées, les titres des axes."""
        out = [f'<rect x="{_f(self.x0)}" y="{_f(self.y0)}" width="{_f(self.l)}" height="{_f(self.h)}" fill="#fff"/>']
        if gx:
            ndx = nd_x if nd_x is not None else _nd(gx[1] - gx[0] if len(gx) > 1 else 1)
            for v in gx:
                x = self.X(v)
                if self.x0 - .5 <= x <= self.x0 + self.l + .5:
                    out.append(_ligne(x, self.y0, x, self.y0 + self.h, GRILLE, 1))
                    out.append(_ligne(x, self.y0 + self.h, x, self.y0 + self.h + 5, AXE, 1))
                    out.append(_txt(x, self.y0 + self.h + 19, nombre(v, ndx), taille, couleur=DOUX))
        if gy:
            ndy = nd_y if nd_y is not None else _nd(gy[1] - gy[0] if len(gy) > 1 else 1)
            for v in gy:
                y = self.Y(v)
                if self.y0 - .5 <= y <= self.y0 + self.h + .5:
                    out.append(_ligne(self.x0, y, self.x0 + self.l, y, GRILLE, 1))
                    out.append(_ligne(self.x0 - 5, y, self.x0, y, AXE, 1))
                    if etiquettes_y:
                        out.append(_txt(self.x0 - 8, y + 4, nombre(v, ndy), taille, "end", DOUX))
        out.append(f'<rect x="{_f(self.x0)}" y="{_f(self.y0)}" width="{_f(self.l)}" height="{_f(self.h)}" fill="none" '
                   f'stroke="{AXE}" stroke-width="1"/>')
        if titre_x:
            out.append(_txt(self.x0 + self.l / 2, self.y0 + self.h + 40, titre_x, taille + 0.5, couleur=ENCRE))
        if titre_y:
            xt, yt = self.x0 - 48, self.y0 + self.h / 2
            out.append(_txt(xt, yt, titre_y, taille + 0.5, couleur=ENCRE, extra=f' transform="rotate(-90 {_f(xt)} {_f(yt)})"'))
        return "".join(out)


def _legende(x, y, entrees: List[Tuple[str, str]], largeur=None, taille=12.0, ancre_droite=True) -> str:
    """Une légende encadrée : [(échantillon SVG à dessiner dans 26 × 12, texte)] ; (x, y) est le coin haut droit
    (ou gauche)."""
    larg = largeur or max(120.0, max(len(t) for _, t in entrees) * taille * 0.53 + 46)
    h = 10 + len(entrees) * (taille + 7)
    x0 = x - larg if ancre_droite else x
    out = [f'<rect x="{_f(x0)}" y="{_f(y)}" width="{_f(larg)}" height="{_f(h)}" rx="6" fill="#fff" fill-opacity=".94" '
           f'stroke="{GRILLE}" stroke-width="1"/>']
    for i, (ech, t) in enumerate(entrees):
        yy = y + 8 + i * (taille + 7) + taille / 2
        out.append(f'<g transform="translate({_f(x0 + 10)} {_f(yy - 6)})">{ech}</g>')
        out.append(_txt(x0 + 44, yy + taille * 0.35, t, taille, "start"))
    return "".join(out)


def _legende_plate(x0, y, entrees: List[Tuple[str, str]], colonnes: int = 2, lcol: float = 300.0, taille: float = 11.5) -> str:
    """Une légende SANS cadre, sous le graphique, en colonnes : elle ne cache aucune courbe."""
    out = []
    for i, (ech, t) in enumerate(entrees):
        x, yy = x0 + (i % colonnes) * lcol, y + (i // colonnes) * (taille + 7)
        out.append(f'<g transform="translate({_f(x)} {_f(yy - 6)})">{ech}</g>' + _txt(x + 34, yy + taille * 0.35, t, taille, "start"))
    return "".join(out)


def _ech_trait(couleur, ep=2.0, tirets="") -> str:
    return _ligne(0, 6, 26, 6, couleur, ep, tirets)


def _ech_aire(couleur, opacite=.18, bord="") -> str:
    b = f' stroke="{bord}" stroke-width="1.4"' if bord else ""
    return f'<rect x="0" y="0" width="26" height="12" rx="2" fill="{couleur}" fill-opacity="{_f(opacite)}"{b}/>'


def _svg(ident: str, l: float, h: float, corps: str, label: str, defs: str = "") -> str:
    d = f"<defs>{defs}</defs>" if defs else ""
    return (f'<svg id="{escape(ident)}" class="sol-graphe" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(l)} {_f(h)}" '
            f'width="100%" role="img" aria-label="{escape(label)}">{d}{corps}</svg>')


def _boite(x, y, texte, couleur, fond, taille=12.0, ancre="end") -> str:
    """Une étiquette encadrée (le texte posé sur un fond clair), comme les annotations de l'écran Streamlit."""
    larg = len(texte) * taille * 0.54 + 12
    x0 = x - larg if ancre == "end" else (x - larg / 2 if ancre == "middle" else x)
    return (f'<rect x="{_f(x0)}" y="{_f(y - taille - 5)}" width="{_f(larg)}" height="{_f(taille + 8)}" rx="4" fill="{fond}" '
            f'stroke="{couleur}" stroke-width=".8"/>' + _txt(x0 + larg / 2, y - 1, escape(texte), taille, couleur=couleur, gras=True))


def _coupe_nom(nom: str, n: int = 24) -> str:
    nom = nom or "—"
    return nom if len(nom) <= n else nom[:n - 1].rstrip() + "…"


def _nappe(x, y, couleur=BLEU, taille=11.0) -> str:
    """Le symbole de la nappe : un triangle pointe en bas posé sur le niveau, et le mot « nappe »."""
    return (f'<path d="M{_f(x - 6)} {_f(y - 10)} L{_f(x + 6)} {_f(y - 10)} L{_f(x)} {_f(y)} Z" fill="none" stroke="{couleur}" '
            f'stroke-width="1.3"/>' + _ligne(x - 4, y + 3, x + 4, y + 3, couleur, 1.1) + _ligne(x - 2, y + 6, x + 2, y + 6, couleur, 1.1)
            + _txt(x + 10, y - 2, "nappe", taille, "start", couleur))


# ------------------------------------------------------------------ 1 · la diffusion de la contrainte (modes 1 et 2)
def _encre_sur(couleur: str) -> str:
    """L'encre lisible sur un fond : blanche sur une teinte sombre (le bas de la palette YlOrBr), noire sinon."""
    r, g, b = (int(couleur[k:k + 2], 16) / 255 for k in (1, 3, 5))
    return "#fff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else ENCRE


def diffusion(ident: str, d: Dict, titre: str, note: str = "") -> str:
    """Le graphique de l'écran : à gauche le profil (couches sous l'assise, teinte selon log M), à droite Δσ, σ'v0,
    le critère d'arrêt et la profondeur d'influence ; ``d`` = rigidite.donnees_diffusion. ``note`` : la légende de
    l'écran Streamlit sous le graphique (texte, échappé ici)."""
    L, H = 700.0, 680.0
    y0, hh = 50.0, 512.0
    ytop, ybot = d["z_haut"], d["z_max"]
    gy = [v for v in graduations(ytop if ytop < 0 else 0.0, ybot, 8) if ytop - 1e-9 <= v <= ybot + 1e-9]
    prof = Cadre(62, y0, 150, hh, 0, 1, ytop, ybot, y_bas=True)
    xs = [0.0] + d["delta_sigma"] + d["sigma_v0"] + d["critere_sigma"]
    gx = graduations(0.0, max(xs) * 1.04 if max(xs) > 0 else 1.0, 5)
    cad = Cadre(232, y0, 446, hh, 0.0, gx[-1], ytop, ybot, y_bas=True)
    out = [_txt(prof.x0 + prof.l / 2, 32, "Profil", 13.5, gras=True), _txt(cad.x0 + cad.l / 2, 32, escape(titre), 13.5, gras=True)]
    out.append(prof.axes(None, gy, "", "Profondeur sous l'assise [m]"))
    for b in d["bandes"]:
        ya, yb = prof.Y(b["z0"]), prof.Y(b["z1"])
        out.append(f'<rect x="{_f(prof.x0)}" y="{_f(ya)}" width="{_f(prof.l)}" height="{_f(yb - ya)}" fill="{b["couleur"]}" '
                   f'stroke="#000" stroke-width=".7"/>')
        if b["etiquette"]:
            # l'étiquette au milieu de la part SOUS l'assise (la ligne d'assise ne la barre pas)
            za = max(b["z0"], 0.0) if b["z1"] > 0 else b["z0"]
            ym = (prof.Y(za) + yb) / 2
            encre = _encre_sur(b["couleur"])
            m = f"M = {nombre(b['M'], 0)} MPa" if b["M"] else "⚠ M manquant"
            out.append(_txt(prof.x0 + prof.l / 2, ym - 3, escape(_coupe_nom(b["nom"], 21)), 11.5, couleur=encre, gras=True))
            out.append(_txt(prof.x0 + prof.l / 2, ym + 12, m, 11, couleur=encre if b["M"] else AMBRE))
    out.append(_ligne(prof.x0, prof.Y(0), prof.x0 + prof.l, prof.Y(0), "#111", 2.6))
    if d["D"] > 0 and prof.Y(0) - prof.Y(-d["D"]) > 22:
        ya = (prof.Y(-d["D"]) + prof.Y(0)) / 2
        out.append(f'<rect x="{_f(prof.x0 + prof.l / 2 - 26)}" y="{_f(ya - 10)}" width="52" height="18" rx="3" fill="#fff" fill-opacity=".85"/>'
                   + _txt(prof.x0 + prof.l / 2, ya + 4, "fouille", 11.5, couleur=DOUX))
    out.append(_txt(prof.x0 + prof.l - 4, prof.Y(0) - 5, "assise", 11.5, "end", gras=True))
    out.append(cad.axes(gx, gy, "Contrainte [kPa]", "", etiquettes_y=False))
    clip = f"{ident}_clip"
    corps = []
    if d["z"]:
        zs = d["z"]
        aire = [(cad.X(0), cad.Y(zs[0]))] + [(cad.X(x), cad.Y(z)) for x, z in zip(d["delta_sigma"], zs)] + [(cad.X(0), cad.Y(zs[-1]))]
        corps.append(_poly(aire, "none", fond=ROUGE, opacite=.16))
        corps.append(_poly([(cad.X(x), cad.Y(z)) for x, z in zip(d["delta_sigma"], zs)], ROUGE, 2.4))
        corps.append(_poly([(cad.X(x), cad.Y(z)) for x, z in zip(d["sigma_v0"], zs)], ARDOISE, 1.5, "8 5"))
        corps.append(_poly([(cad.X(x), cad.Y(z)) for x, z in zip(d["critere_sigma"], zs)], VERT, 1.7, "2 4"))
    zi = d["z_influence"]
    tronque = d["tronque"]
    coul, fond = (AMBRE, "#fffbeb") if tronque else (VERT, "#ecfdf5")
    corps.append(_ligne(cad.x0, cad.Y(zi), cad.x0 + cad.l, cad.Y(zi), coul, 1.9, "8 5" if tronque else ""))
    out.append(f'<g clip-path="url(#{clip})">{"".join(corps)}</g>')
    out.append(_ligne(cad.x0, cad.Y(0), cad.x0 + cad.l, cad.Y(0), "#111", 2.6))
    if d["nappe"] is not None:
        yn = prof.Y(d["nappe"])
        out.append(_ligne(prof.x0, yn, prof.x0 + prof.l, yn, BLEU, 1.6, "7 4") + _ligne(cad.x0, yn, cad.x0 + cad.l, yn, BLEU, 1.2, "7 4"))
        out.append(f'<rect x="{_f(prof.x0 + 4)}" y="{_f(yn - 16)}" width="58" height="15" rx="3" fill="#fff" fill-opacity=".85"/>'
                   + _nappe(prof.x0 + 14, yn - 2))
    lib = (f"profil interrompu à {nombre(zi, 2)} m — k surestimé" if tronque else f"profondeur d'influence  {nombre(zi, 2)} m")
    yl = cad.Y(zi) - 7
    if yl < cad.y0 + 24:
        yl = cad.Y(zi) + 25
    out.append(_boite(cad.x0 + cad.l - 8, yl, lib, coul, fond, 12))
    crit = _m(f"critère d'arrêt : {nombre(d['critere'] * 100, 0)} % · σ'v0")
    leg = [(_ech_aire(ROUGE, .16), "Δσ mobilisée (comprime le sol)"), (_ech_trait(ROUGE, 2.4), "Δσ apportée par la fondation"),
           (_ech_trait(ARDOISE, 1.5, "8 5"), _m("σ'v0 en place")), (_ech_trait(VERT, 1.7, "2 4"), crit)]
    if d["nappe"] is not None:
        leg.append((_ech_trait(BLEU, 1.3, "7 4"), "nappe"))
    y_leg = cad.y0 + cad.h + 66
    out.append(_legende_plate(prof.x0 + 10, y_leg, leg, 2, 312, 11.5))
    if note:
        y_note = y_leg + ((len(leg) + 1) // 2) * 18.5 + 14
        texte, n = _paragraphe(prof.x0 - 40, y_note, note, 108, 11.5)
        out.append(texte)
        H = max(H, y_note + (n - 1) * 16.5 + 16)
    return _svg(ident, L, H, "".join(out), f"Diffusion de la contrainte · {titre}", cad.clip(clip))


# ------------------------------------------------------------------ 2 · k selon la largeur de la fondation
def k_largeur(ident: str, d: Dict, k_projet: Optional[Tuple[float, float]] = None, note: str = "") -> str:
    """_fig_k_vs_B : k au centre pour 18 largeurs (bornes basse et haute, l'encadrement entre les deux), la largeur du
    projet en tirets ; ``d`` = rigidite.k_fonction_B."""
    L, H = 700.0, 540.0
    Bs, kb, kh = d["B"], d["k_bas"], d["k_haut"]
    ymax = max(kb + kh + [1e-6])
    gx = graduations(0.0, max(Bs) * 1.02, 7)
    gy = graduations(0.0, ymax * 1.08, 6)
    cad = Cadre(76, 50, 598, 410, gx[0], gx[-1], 0.0, gy[-1])
    out = [_txt(cad.x0 + cad.l / 2, 28, "k n'est pas une propriété du sol : il dépend de la fondation", 13.5, gras=True),
           cad.axes(gx, gy, "Largeur de fondation B [m]", "k [MN/m³]")]
    clip = f"{ident}_clip"
    haut = [(cad.X(b), cad.Y(k)) for b, k in zip(Bs, kh)]
    bas = [(cad.X(b), cad.Y(k)) for b, k in zip(Bs, kb)]
    corps = [_poly(haut + bas[::-1] + [haut[0]], "none", fond=VIOLET, opacite=.16), _poly(bas, VIOLET, 2.0), _poly(haut, VIOLET, 2.0, "8 5")]
    xb = cad.X(d["B_projet"])
    corps.append(_ligne(xb, cad.y0, xb, cad.y0 + cad.h, "#94a3b8", 1.3, "6 4"))
    out.append(f'<g clip-path="url(#{clip})">{"".join(corps)}</g>')
    out.append(_txt(xb + 6, cad.y0 + 18, f"B = {nombre(d['B_projet'], 2)} m (projet)", 12, "start", DOUX))
    for b, k in zip(Bs, kb):
        out.append(f'<circle cx="{_f(cad.X(b))}" cy="{_f(cad.Y(k))}" r="2.6" fill="{VIOLET}"/>')
    if k_projet:
        for k in k_projet:
            out.append(f'<circle cx="{_f(xb)}" cy="{_f(cad.Y(k))}" r="5" fill="#fff" stroke="{VIOLET}" stroke-width="2"/>')
    leg = [(_ech_aire(VIOLET, .16), "encadrement"), (_ech_trait(VIOLET, 2.0), "borne basse des modules"),
           (_ech_trait(VIOLET, 2.0, "8 5"), "borne haute des modules")]
    out.append(_legende(cad.x0 + cad.l - 10, cad.y0 + 10, leg, 212, 11.5))
    if note:
        texte, n = _paragraphe(cad.x0 - 50, cad.y0 + cad.h + 68, note, 106, 11.5)
        out.append(texte)
        H = max(H, cad.y0 + cad.h + 68 + (n - 1) * 16.5 + 16)
    return _svg(ident, L, H, "".join(out), "k en fonction de la largeur de fondation", cad.clip(clip))


# ------------------------------------------------------------------ 3 · les courbes CPT et la classification
def cpt(ident: str, d: Dict, titre: str = "", decoupage: Optional[List[Dict]] = None) -> str:
    """_fig_cpt : qc, fs (s'il y en a) et Ic par couche en fonction de la profondeur, les bornes des zones de Robertson
    en pointillé, la nappe ; ``d`` = rigidite.donnees_cpt. ``decoupage`` (aperçu d'un import) : les limites des couches
    découpées, tracées en traits fins sur les trois panneaux."""
    a_fs = bool(d["fs"])
    n = 3 if a_fs else 2
    L, H = 700.0, 700.0
    zmax = max(d["z_max"], 0.5)
    gy = [v for v in graduations(0.0, zmax, 8) if v <= zmax + 1e-9]
    x0, larg_tot, ecart = 64.0, 620.0, 22.0
    lp = (larg_tot - ecart * (n - 1)) / n
    y0, hh = 48.0, 562.0
    out, defs = [], []
    if titre:
        out.append(_txt(L / 2, 24, escape(titre), 13.5, gras=True))
    panneaux = []
    qmax = max(d["qc"] + [1e-6])
    gq = graduations(0.0, qmax * 1.02, 4)
    panneaux.append(("qc [MPa]", Cadre(x0, y0, lp, hh, 0.0, gq[-1], 0.0, zmax, y_bas=True), gq, [(d["qc"], d["z"], BLEU_QC)]))
    if a_fs:
        fmax = max(d["fs"] + [1e-6])
        gf = graduations(0.0, fmax * 1.02, 4)
        panneaux.append(("fs [kPa]", Cadre(x0 + lp + ecart, y0, lp, hh, 0.0, gf[-1], 0.0, zmax, y_bas=True), gf, [(d["fs"], d["z_fs"], AMBRE)]))
    ci = Cadre(x0 + (n - 1) * (lp + ecart), y0, lp, hh, 1.0, 4.0, 0.0, zmax, y_bas=True)
    panneaux.append(("Ic (Robertson)", ci, [1.0, 2.0, 3.0, 4.0], []))
    for k, (titre_x, cad, gx, series) in enumerate(panneaux):
        clip = f"{ident}_c{k}"
        defs.append(cad.clip(clip))
        out.append(cad.axes(gx, gy, titre_x, "Profondeur [m]" if k == 0 else "", etiquettes_y=(k == 0), taille=11.5))
        corps = []
        for c in decoupage or []:
            corps.append(_ligne(cad.x0, cad.Y(c["z1"]), cad.x0 + cad.l, cad.Y(c["z1"]), "#64748b", .9, "3 3"))
        for xs, zs, coul in series:
            corps.append(_poly([(cad.X(x), cad.Y(z)) for x, z in zip(xs, zs)], coul, 1.2))
        if cad is ci:
            for b, _ in d["bornes_sbt"]:
                corps.append(_ligne(cad.X(b), cad.y0, cad.X(b), cad.y0 + cad.h, "#94a3b8", 1, "2 3"))
            bornes = [1.0] + [b for b, _ in d["bornes_sbt"]] + [4.0]
            for j in range(len(bornes) - 1):
                xm = (cad.X(bornes[j]) + cad.X(min(bornes[j + 1], 4.0))) / 2
                corps.append(_txt(xm, cad.y0 + 14, str(7 - j), 10.5, couleur="#94a3b8", gras=True))
            for ic, z0, z1 in d["ic"]:
                corps.append(_ligne(cad.X(ic), cad.Y(z0), cad.X(ic), cad.Y(z1), SARCELLE, 4.4, extra=' stroke-linecap="butt"'))
        if d["nappe"] is not None and d["nappe"] <= zmax:
            yn = cad.Y(d["nappe"])
            corps.append(_ligne(cad.x0, yn, cad.x0 + cad.l, yn, BLEU, 1.3, "7 4"))
        out.append(f'<g clip-path="url(#{clip})">{"".join(corps)}</g>')
    if d["nappe"] is not None and d["nappe"] <= zmax:
        yn = panneaux[0][1].Y(d["nappe"])
        out.append(f'<rect x="{_f(x0 + 3)}" y="{_f(yn - 16)}" width="58" height="15" rx="3" fill="#fff" fill-opacity=".85"/>'
                   + _nappe(x0 + 12, yn - 2))
    out.append(_txt(L / 2, H - 30, "zones de Robertson (Ic) : 7 sable graveleux · 6 sable · 5 sable limoneux · 4 limon argileux", 11, couleur=DOUX))
    out.append(_txt(L / 2, H - 14, "3 argile · 2 sol organique" + (" · tirets gris : limites des couches découpées" if decoupage else ""),
                    11, couleur=DOUX))
    if d.get("erreur"):
        out.append(_txt(ci.x0 + ci.l / 2, ci.y0 + ci.h / 2, "Ic non tracé :", 12, couleur=AMBRE, gras=True)
                   + _txt(ci.x0 + ci.l / 2, ci.y0 + ci.h / 2 + 16, "découpage impossible", 12, couleur=AMBRE))
    return _svg(ident, L, H, "".join(out), "Courbes CPT et classification", "".join(defs))


# ------------------------------------------------------------------ 4 · la coupe : la semelle sur son profil, à l'échelle
_YLORBR = ("#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506")


def _hachures(ident: str) -> str:
    return (f'<pattern id="{ident}_beton" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="10" height="10" fill="{BETON}"/><line x1="0" y1="0" x2="0" y2="10" stroke="#9AA6B5" stroke-width="1"/></pattern>')


def _cote_v(x, y1, y2, texte, taille=12.5, gauche=True) -> str:
    t = 5.0
    xt = x - 8 if gauche else x + 15
    return (_ligne(x, y1, x, y2, AXE, 1) + _ligne(x - t, y1 + t, x + t, y1 - t, ENCRE, 1.2) + _ligne(x - t, y2 + t, x + t, y2 - t, ENCRE, 1.2)
            + _txt(xt, (y1 + y2) / 2, texte, taille, gras=True, extra=f' transform="rotate(-90 {_f(xt)} {_f((y1 + y2) / 2)})"'))


def _cote_h(x1, x2, y, texte, taille=12.5) -> str:
    t = 5.0
    return (_ligne(x1, y, x2, y, AXE, 1) + _ligne(x1 - t, y + t, x1 + t, y - t, ENCRE, 1.2) + _ligne(x2 - t, y + t, x2 + t, y - t, ENCRE, 1.2)
            + _txt((x1 + x2) / 2, y - 7, texte, taille, gras=True))


def _teinte(M: float, lo: float, hi: float) -> str:
    """La teinte d'une couche selon log M, sur la palette YlOrBr du graphique de diffusion mais plus DOUCE (0,08 à
    0,50) : sur une coupe, la couche se lit sans écraser les cotes et les textes."""
    t = ((math.log10(max(M, .1)) - math.log10(max(lo, .1))) / ((math.log10(max(hi, .1)) - math.log10(max(lo, .1))) or 1)
         if M > 0 and hi > lo else 0.0)
    x = (0.08 + 0.42 * min(max(t, 0.0), 1.0)) * (len(_YLORBR) - 1)
    i = min(int(x), len(_YLORBR) - 2)
    a, b, u = _YLORBR[i], _YLORBR[i + 1], x - i
    return "#" + "".join(f"{round(int(a[k:k + 2], 16) * (1 - u) + int(b[k:k + 2], 16) * u):02x}" for k in (1, 3, 5))


def coupe(ident: str, c: Dict) -> str:
    """La semelle (largeur B, assise D) posée sur le profil du sondage, À L'ÉCHELLE (une même échelle dans les deux
    sens) : les couches depuis le terrain naturel avec leur module, la nappe, la profondeur d'influence sous l'assise
    et la pression q. ``c`` : B, L, D, q, q_net, nappe (ou None), couches [{h, nom, M, M_haut}], z_infl (ou None),
    tronque, nom."""
    B, D = c["B"], c["D"]
    H_prof = sum(x["h"] for x in c["couches"]) or 1.0
    zi = c.get("z_infl")
    # la profondeur dessinée : ce qui compte sous la semelle (la zone d'influence et un peu au-delà), pas tout le profil
    utile = max(D + 1.5 * zi, D + 2.5 * B, D + 3.0) if zi else max(D + 3.0 * B, D + 4.0)
    z_bas = min(H_prof, utile) if H_prof > 0 else utile
    z_bas = max(z_bas, D + (zi or 0.0) * 1.1, D + 0.5)
    coupe_profil = H_prof > z_bas + 1e-9
    larg_m = max(3.0 * B, 0.75 * z_bas, 4.0)
    LW, HW = 700.0, 700.0
    marge_g, marge_h, marge_b = 64.0, 112.0, 30.0
    zone_l, zone_h = 400.0, HW - marge_h - marge_b
    ech = min(zone_l / larg_m, zone_h / z_bas)
    xc = marge_g + zone_l / 2
    X = lambda x: xc + x * ech  # noqa: E731 — l'axe de la semelle en x = 0
    Y = lambda z: marge_h + z * ech  # noqa: E731
    xg, xd = X(-zone_l / ech / 2), X(zone_l / ech / 2)
    out = []
    Ms = [x["M"] for x in c["couches"] if x.get("M")]
    lo, hi = (min(Ms), max(Ms)) if Ms else (1.0, 100.0)
    cum = 0.0
    for x in c["couches"]:
        if x["h"] <= 0:
            continue
        z0, z1 = cum, cum + x["h"]
        cum = z1
        if z0 >= z_bas - 1e-9:
            break
        y1 = Y(min(z1, z_bas))
        M = x.get("M") or 0.0
        out.append(f'<rect x="{_f(xg)}" y="{_f(Y(z0))}" width="{_f(xd - xg)}" height="{_f(y1 - Y(z0))}" fill="{_teinte(M, lo, hi)}" '
                   f'stroke="#000" stroke-width=".6"/>')
        ym = (Y(z0) + y1) / 2
        if y1 - Y(z0) >= 15:
            deux = y1 - Y(z0) >= 32
            mtxt = (f"M = {nombre(M, 0)}" + (f" – {nombre(x['M_haut'], 0)}" if x.get("M_haut") and x["M_haut"] > M else "") + " MPa") if M else "M manquant"
            out.append(_txt(xd + 12, ym - (3 if deux else -4), escape(_coupe_nom(x["nom"], 29)), 12, "start", gras=True))
            if deux:
                out.append(_txt(xd + 12, ym + 12, mtxt, 11.5, "start", ENCRE if M else AMBRE))
        if z1 <= z_bas + 1e-9:
            out.append(_txt(xg - 7, Y(z1) + 4, nombre(z1, 2), 11, "end", DOUX))
    out.append(_txt(xg - 7, Y(0) + 4, "0", 11, "end", DOUX))
    if coupe_profil:
        yb_ = Y(z_bas)
        pas = (xd - xg) / 24
        zig = " ".join(f"{_f(xg + k * pas)},{_f(yb_ + (4 if k % 2 else -4))}" for k in range(25))
        out.append(f'<rect x="{_f(xg - 1)}" y="{_f(yb_ - 4)}" width="{_f(xd - xg + 2)}" height="12" fill="#fff"/>'
                   f'<polyline points="{zig}" fill="none" stroke="{ENCRE}" stroke-width="1.2"/>'
                   + _txt((xg + xd) / 2, yb_ + 22, f"profil saisi jusqu'à {nombre(H_prof, 2)} m sous le TN (coupé ici)", 11.5, couleur=DOUX))
    # le terrain naturel, la fouille et la semelle
    ep = min(max(0.25 * B, 0.3), 0.8)
    fx0, fx1 = X(-B / 2), X(B / 2)
    jeu = 0.12 * B * ech
    if D > 0:
        out.append(f'<rect x="{_f(fx0 - jeu)}" y="{_f(Y(0))}" width="{_f(fx1 - fx0 + 2 * jeu)}" height="{_f(D * ech)}" fill="#fff"/>')
        ytop = Y(max(D - ep, 0.0))
    else:
        ytop = Y(-ep)
    out.append(f'<rect x="{_f(fx0)}" y="{_f(ytop)}" width="{_f(fx1 - fx0)}" height="{_f(Y(D) - ytop)}" fill="url(#{ident}_beton)" '
               f'stroke="{ENCRE}" stroke-width="2"/>')
    if D > 0 and D - ep > 0:
        lf = min(0.3 * B, 0.4) * ech
        out.append(f'<rect x="{_f((fx0 + fx1) / 2 - lf / 2)}" y="{_f(Y(0) - 10)}" width="{_f(lf)}" height="{_f(ytop - Y(0) + 10)}" '
                   f'fill="url(#{ident}_beton)" stroke="{ENCRE}" stroke-width="1.6"/>')
    out.append(_ligne(xg, Y(0), fx0 - (jeu if D > 0 else 0), Y(0), ENCRE, 2.2) + _ligne(fx1 + (jeu if D > 0 else 0), Y(0), xd, Y(0), ENCRE, 2.2))
    if D > 0:
        out.append(_ligne(fx0 - jeu, Y(0), fx0 - jeu, Y(D), ENCRE, 1.2, "4 3") + _ligne(fx1 + jeu, Y(0), fx1 + jeu, Y(D), ENCRE, 1.2, "4 3"))
    out.append(_txt(xg + 4, Y(0) - 7, "TN", 12, "start", gras=True))
    # la pression
    ya = (Y(0) - 10 if D > 0 and D - ep > 0 else ytop) - 6
    for k in range(7):
        x = fx0 + (fx1 - fx0) * (k + .5) / 7
        out.append(_ligne(x, ya - 28, x, ya - 4, ROUGE, 1.6) + f'<path d="M{_f(x - 4)} {_f(ya - 10)} L{_f(x + 4)} {_f(ya - 10)} L{_f(x)} {_f(ya - 2)} Z" fill="{ROUGE}"/>')
    out.append(_ligne(fx0, ya - 28, fx1, ya - 28, ROUGE, 1.6))
    qt = _m(f"q = {nombre(c['q'], 0)} kPa" + (f" · q_net = {nombre(c['q_net'], 1)} kPa" if c.get("q_net") is not None else ""))
    out.append(_txt((fx0 + fx1) / 2, ya - 36, qt, 12.5, couleur=ROUGE, gras=True))
    if c.get("nappe") is not None and c["nappe"] <= z_bas:
        yn = Y(c["nappe"])
        out.append(_ligne(xg, yn, xd, yn, BLEU, 1.5, "8 4") + f'<rect x="{_f(xg + 3)}" y="{_f(yn - 16)}" width="58" height="15" rx="3" fill="#fff" '
                   f'fill-opacity=".85"/>' + _nappe(xg + 12, yn - 2))
    if zi is not None and zi > 0:
        yi = Y(D + zi)
        coul = AMBRE if c.get("tronque") else VERT
        out.append(_ligne(xg, yi, xd, yi, coul, 1.8, "9 5"))
        lib = "profil interrompu — k surestimé" if c.get("tronque") else "profondeur d'influence"
        out.append(f'<rect x="{_f(xg + 4)}" y="{_f(yi - 20)}" width="{_f(len(lib) * 6.4 + 8)}" height="16" rx="3" fill="#fff" '
                   f'fill-opacity=".85"/>' + _txt(xg + 8, yi - 7, lib, 12, "start", coul, gras=True))
        out.append(_cote_v(fx1 + 22, Y(D), yi, f"z = {nombre(zi, 2)} m", 12.5, gauche=False))
    yb = Y(D) + 24
    out.append(_ligne(fx0, Y(D) + 4, fx0, yb + 6, AXE, .8, "4 3") + _ligne(fx1, Y(D) + 4, fx1, yb + 6, AXE, .8, "4 3")
               + f'<rect x="{_f((fx0 + fx1) / 2 - 50)}" y="{_f(yb - 22)}" width="100" height="18" fill="#fff" fill-opacity=".85" rx="3"/>'
               + _cote_h(fx0, fx1, yb, f"B = {nombre(B, 2)} m"))
    if D > 0:
        out.append(_cote_v(fx0 - jeu - 18, Y(0), Y(D), f"D = {nombre(D, 2)}", 12.5))
    out.append(_txt(LW / 2, 26, escape(f"Coupe · {c.get('nom', '')}") + f" · semelle {nombre(B, 2)} × {nombre(c['L'], 2)} m", 13.5, gras=True))
    out.append(_txt(LW / 2, 44, "profondeurs en m sous le terrain naturel · même échelle dans les deux sens", 11, couleur=DOUX))
    return _svg(ident, LW, HW, "".join(out), "Coupe de la semelle sur le profil", _hachures(ident))


# ------------------------------------------------------------------ 5 · les sondages côte à côte
def sondages(ident: str, cols: List[Dict], D: float, nappe: Optional[float]) -> str:
    """Les sondages de la fondation côte à côte, à la même échelle des profondeurs : les couches (teinte selon log M,
    commune à tous), l'assise, la nappe, et sous chaque colonne son k au centre. ``cols`` : nom, couches [{h, nom, M}],
    k (couple bas/haut) ou None (écarté), z_infl."""
    n = max(len(cols), 1)
    Hmax = max([sum(x["h"] for x in c["couches"]) for c in cols] + [D + 1.0])
    LW, HW = 700.0, 660.0
    y0, hh = 70.0, 500.0
    Y = lambda z: y0 + z / Hmax * hh  # noqa: E731
    ecart = 512.0 / n                        # les colonnes de 66 à 578 ; à droite, la marge des repères (assise, nappe)
    lc = min(140.0, ecart * 0.62)
    Ms = [x["M"] for c in cols for x in c["couches"] if x.get("M")]
    lo, hi = (min(Ms), max(Ms)) if Ms else (1.0, 100.0)
    out = [_txt(LW / 2, 26, "Les sondages de la fondation · même échelle des profondeurs", 13.5, gras=True)]
    for v in [v for v in graduations(0.0, Hmax, 8) if v <= Hmax + 1e-9]:
        out.append(_ligne(58, Y(v), LW - 14, Y(v), GRILLE, 1) + _txt(52, Y(v) + 4, nombre(v, 1), 11, "end", DOUX))
    out.append(_txt(18, y0 + hh / 2, "Profondeur sous le TN [m]", 12, extra=f' transform="rotate(-90 18 {_f(y0 + hh / 2)})"'))
    # les repères dans la marge de droite : l'assise au-dessus de son trait ; la nappe au-dessus du sien, ou dessous quand
    # les deux traits sont trop proches pour deux étiquettes
    proches = nappe is not None and abs(Y(nappe) - Y(D)) < 18
    y_ass = Y(D) + 15 if proches and nappe <= D else Y(D) - 6
    out.append(_ligne(58, Y(D), LW - 14, Y(D), "#111", 1.8, "10 4") + _txt(LW - 16, y_ass, f"assise D = {nombre(D, 2)} m", 11.5, "end", gras=True))
    if nappe is not None and nappe <= Hmax:
        out.append(_ligne(58, Y(nappe), LW - 14, Y(nappe), BLEU, 1.4, "7 4") + _nappe(LW - 66, Y(nappe)))
    for i, c in enumerate(cols):
        xm = 66 + ecart * (i + .5)
        x0 = xm - lc / 2
        cum = 0.0
        for x in c["couches"]:
            if x["h"] <= 0:
                continue
            z0, z1 = cum, cum + x["h"]
            cum = z1
            M = x.get("M") or 0.0
            coul = _teinte(M, lo, hi)
            out.append(f'<rect x="{_f(x0)}" y="{_f(Y(z0))}" width="{_f(lc)}" height="{_f(Y(z1) - Y(z0))}" fill="{coul}" stroke="#000" '
                       f'stroke-width=".6"/>')
            if Y(z1) - Y(z0) >= 28:
                encre = _encre_sur(coul)
                out.append(_txt(xm, (Y(z0) + Y(z1)) / 2 - 2, escape(_coupe_nom(x["nom"], max(8, int(lc / 6.6)))), 10.5, couleur=encre, gras=True)
                           + _txt(xm, (Y(z0) + Y(z1)) / 2 + 11, f"M = {nombre(M, 0)} MPa" if M else "M manquant", 10.5,
                                  couleur=encre if M else AMBRE))
        out.append(_txt(xm, y0 - 12, escape(_coupe_nom(c["nom"], 18)), 12.5, gras=True))
        if c.get("k"):
            kb, kh = c["k"]
            kt = f"{nombre(kb, 2)}" + (f" à {nombre(kh, 2)}" if abs(kh - kb) > 1e-9 else "")
            out.append(_txt(xm, y0 + hh + 24, f"k = {kt}", 12.5, gras=True, couleur=VIOLET) + _txt(xm, y0 + hh + 40, "MN/m³ (centre)", 11, couleur=DOUX))
            if c.get("z_infl"):
                yi = Y(D + c["z_infl"])
                if yi <= y0 + hh:
                    out.append(_ligne(x0 - 8, yi, x0 + lc + 8, yi, VERT, 1.8, "6 4"))
        else:
            out.append(_txt(xm, y0 + hh + 24, "écarté : aucun M", 12, couleur=AMBRE, gras=True))
        # l'assise et la nappe repérées sur le bord de la colonne
        out.append(_ligne(x0 - 7, Y(D), x0, Y(D), "#111", 2.2) + _ligne(x0 + lc, Y(D), x0 + lc + 7, Y(D), "#111", 2.2))
    out.append(_txt(LW / 2, HW - 12, "tirets verts : profondeur d'influence sous l'assise (centre, borne basse des modules)", 11, couleur=DOUX))
    return _svg(ident, LW, HW, "".join(out), "Sondages côte à côte")


# ------------------------------------------------------------------ 6 · mode 3 : k = q / w
def droite_qw(ident: str, x: Dict) -> str:
    """La loi de Winkler q = k·w : la droite par l'origine et le couple (w, q) saisi."""
    LW, HW = 700.0, 540.0
    w, q, k = x["w"], x["q"], x["k_kNm3"]
    gx = graduations(0.0, w * 1.6 if w > 0 else 1.0, 6)
    gy = graduations(0.0, max(q * 1.6, 1.0), 6)
    cad = Cadre(80, 50, 592, 410, 0.0, gx[-1], 0.0, gy[-1])
    out = [_txt(cad.x0 + cad.l / 2, 28, "Loi de Winkler : q = k · w", 13.5, gras=True), cad.axes(gx, gy, "Tassement w [mm]", "Pression q [kPa]")]
    clip = f"{ident}_clip"
    corps = []
    if k > 0:
        corps.append(_poly([(cad.X(0), cad.Y(0)), (cad.X(gx[-1]), cad.Y(k * gx[-1] / 1000.0))], VIOLET, 2.4))
    out.append(f'<g clip-path="url(#{clip})">{"".join(corps)}</g>')
    out.append(_ligne(cad.X(w), cad.Y(0), cad.X(w), cad.Y(q), DOUX, 1, "5 4") + _ligne(cad.X(0), cad.Y(q), cad.X(w), cad.Y(q), DOUX, 1, "5 4"))
    out.append(f'<circle cx="{_f(cad.X(w))}" cy="{_f(cad.Y(q))}" r="6" fill="#fff" stroke="{ROUGE}" stroke-width="2.4"/>')
    out.append(_txt(cad.X(w) + 12, cad.Y(q) + 20, f"w = {nombre(w, 2)} mm ; q = {nombre(q, 1)} kPa", 12.5, "start", ROUGE, True))
    out.append(_boite(cad.x0 + cad.l - 10, cad.y0 + 30, f"k = {nombre(k / 1000.0, 2)} MN/m³", VIOLET, "#f5f3ff", 13))
    return _svg(ident, LW, HW, "".join(out), "Loi de Winkler q = k·w", cad.clip(clip))


# ------------------------------------------------------------------ 7 · mode 4 : les théories comparées
def theories(ident: str, t: Dict) -> str:
    """Les cinq valeurs de k du mode 4 en barres horizontales, la valeur retenue (tassement calculé) mise en avant."""
    LW, HW = 700.0, 500.0
    meth = t["methodes"]
    kmax = max([m["k"] for m in meth] + [1e-6])
    gx = graduations(0.0, kmax * 1.15, 5)
    cad = Cadre(250, 70, 420, 340, 0.0, gx[-1], 0, len(meth))
    out = [_txt(LW / 2, 26, "Cinq façons d'estimer k pour le même sol", 13.5, gras=True),
           _txt(LW / 2, 44, f"sol homogène E = {nombre(t['E'], 1)} MPa · semelle {nombre(t['B'], 2)} × {nombre(t['L'], 2)} m · ν = {nombre(t['nu'], 2)}",
                11.5, couleur=DOUX),
           cad.axes(gx, None, "k [MN/m³]", "")]
    hb = cad.h / len(meth)
    for i, m in enumerate(meth):
        y = cad.y0 + i * hb + hb * 0.22
        ret = m["cle"] == "tassement"
        biais = m["cle"] == "ancienne"
        coul = VIOLET if ret else ("#FCD34D" if biais else "#A5B4C8")
        out.append(f'<rect x="{_f(cad.X(0))}" y="{_f(y)}" width="{_f(max(cad.X(m["k"]) - cad.X(0), 1))}" height="{_f(hb * 0.56)}" rx="3" '
                   f'fill="{coul}"/>')
        out.append(_txt(cad.X(m["k"]) + 7, y + hb * 0.28 + 5, nombre(m["k"], 2), 12.5, "start", VIOLET if ret else ENCRE, True))
        out.append(_txt(cad.x0 - 10, y + hb * 0.28 - 2, escape(m["nom"]), 12, "end", ENCRE, ret))
        out.append(_txt(cad.x0 - 10, y + hb * 0.28 + 13, escape(m["domaine"]), 10.5, "end", AMBRE if biais else DOUX))
    out.append(_txt(LW / 2, HW - 14, f"retenue : tassement calculé · w = {nombre(t['tassement']['w_mm'], 1)} mm · profondeur d'influence "
                                     f"{nombre(t['tassement']['z_influence'], 2)} m", 11.5, couleur=DOUX))
    return _svg(ident, LW, HW, "".join(out), "Comparaison des théories")


# ------------------------------------------------------------------ 8 · mode 5 : l'abaque des sols
def abaque(ident: str, a: Dict) -> str:
    """Les sols de la base : k de plaque 0,30 m (clair) et k corrigé pour la largeur B (foncé), en barres de plage sur
    une échelle logarithmique."""
    lignes = a["lignes"]
    n = len(lignes)
    LW = 700.0
    hl = 23.0
    HW = 84 + n * hl + 92
    vals = [v for x in lignes for v in (x["k_bas"], x["kp_max"]) if v > 0]
    lo = 10 ** math.floor(math.log10(min(vals))) if vals else 0.1
    hi = 10 ** math.ceil(math.log10(max(vals))) if vals else 1000.0
    cad = Cadre(232, 60, 444, n * hl, lo, hi, 0, n, x_log=True)
    gx = [10 ** e for e in range(int(round(math.log10(lo))), int(round(math.log10(hi))) + 1)]
    out = [_txt(LW / 2, 24, f"k de plaque 0,30 m et k corrigé pour B = {nombre(a['B'], 2)} m (Terzaghi)", 13.5, gras=True),
           _txt(LW / 2, 42, "échelle logarithmique · k en MN/m³", 11, couleur=DOUX), cad.axes(gx, None, "k [MN/m³]", "", nd_x=3)]
    cat = None
    for i, x in enumerate(lignes):
        y = cad.y0 + i * hl
        if x["categorie"] != cat:
            cat = x["categorie"]
            if i:
                out.append(_ligne(8, y, cad.x0 + cad.l, y, "#CBD2DC", 1))
        out.append(_txt(cad.x0 - 9, y + hl * 0.68, escape(_coupe_nom(x["type"], 33)), 11.5, "end"))
        out.append(f'<rect x="{_f(cad.X(x["kp_min"]))}" y="{_f(y + 4)}" width="{_f(max(cad.X(x["kp_max"]) - cad.X(x["kp_min"]), 2))}" '
                   f'height="{_f(hl - 8)}" rx="3" fill="#DDD6FE"/>')
        out.append(f'<rect x="{_f(cad.X(x["k_bas"]))}" y="{_f(y + 7.5)}" width="{_f(max(cad.X(x["k_haut"]) - cad.X(x["k_bas"]), 2))}" '
                   f'height="{_f(hl - 15)}" rx="2" fill="{VIOLET}"/>')
    leg = [(_ech_aire("#DDD6FE", 1), "k de plaque 0,30 m (tables)"), (_ech_aire(VIOLET, 1), f"k pour B = {nombre(a['B'], 2)} m")]
    out.append(_legende(cad.x0 + cad.l, cad.y0 + cad.h + 44, leg, 212, 11.5))
    return _svg(ident, LW, HW, "".join(out), "Abaque des sols")


# ------------------------------------------------------------------ 9 · mode 6 : le plat en béton, à l'échelle
def plat(ident: str, p: Dict) -> str:
    """La coupe du contact d'un plat sur le béton, à l'échelle (mm) : le plat B, le lit de mortier t_g s'il y en a, la
    hauteur de béton mobilisée h_c = α·min(B, L) hachurée ; à droite la vue en plan B × L."""
    B, L, hc = p["B"], p["L"], p["hc_m"] * 1000.0
    tg = p["tg"] if p["mortier"] else 0.0
    tp = max(min(B, L) * 0.08, 8.0)
    larg = max(B * 1.5, 120.0)
    haut = tp + tg + max(hc * 1.3, 0.4 * B, 40.0)
    LW, HW = 700.0, 600.0
    ech = min(380.0 / larg, 380.0 / haut)
    x0 = 60 + (380 - larg * ech) / 2
    y0 = 130.0 + (380 - haut * ech) / 3
    X = lambda x: x0 + (x + larg / 2) * ech  # noqa: E731
    Y = lambda z: y0 + z * ech  # noqa: E731
    out = [_txt(LW / 2, 26, _m("Raideur du contact : compression d'une hauteur de béton h_c"), 13.5, gras=True),
           _txt(LW / 2, 44, "coupe et vue en plan à l'échelle · cotes en mm", 11, couleur=DOUX)]
    zb = tp + tg
    out.append(f'<rect x="{_f(X(-larg / 2))}" y="{_f(Y(zb))}" width="{_f(larg * ech)}" height="{_f((haut - zb) * ech)}" fill="{BETON}" '
               f'stroke="{ENCRE}" stroke-width="1.6"/>')
    out.append(f'<rect x="{_f(X(-B / 2))}" y="{_f(Y(zb))}" width="{_f(B * ech)}" height="{_f(min(hc, haut - zb) * ech)}" fill="url(#{ident}_beton)" '
               f'stroke="{ROUGE}" stroke-width="1.6" stroke-dasharray="6 4"/>')
    if tg > 0:
        out.append(f'<rect x="{_f(X(-B / 2))}" y="{_f(Y(tp))}" width="{_f(B * ech)}" height="{_f(tg * ech)}" fill="{MORTIER}" stroke="{ENCRE}" '
                   f'stroke-width="1.2"/>')
    out.append(f'<rect x="{_f(X(-B / 2))}" y="{_f(Y(0))}" width="{_f(B * ech)}" height="{_f(tp * ech)}" fill="{PLAT}" stroke="{ENCRE}" '
               f'stroke-width="1.8"/>')
    for k in range(7):
        x = X(-B / 2 + B * (k + .5) / 7)
        out.append(_ligne(x, Y(0) - 32, x, Y(0) - 6, ROUGE, 1.6) + f'<path d="M{_f(x - 4)} {_f(Y(0) - 12)} L{_f(x + 4)} {_f(Y(0) - 12)} '
                                                                    f'L{_f(x)} {_f(Y(0) - 3)} Z" fill="{ROUGE}"/>')
    out.append(_cote_h(X(-B / 2), X(B / 2), Y(0) - 44, f"B = {nombre(B, 0)}"))
    out.append(_cote_v(X(B / 2) + 26, Y(zb), Y(zb + min(hc, haut - zb)), _m(f"h_c = {nombre(hc, 1)}"), 12.5, gauche=False))
    if tg > 0:
        out.append(_txt(X(-B / 2) - 8, Y(tp + tg / 2) + 4, f"mortier {nombre(tg, 0)}", 12, "end", ENCRE, True))
    out.append(_txt(X(-B / 2) - 8, Y(tp / 2) + 4, "plat", 12, "end", ENCRE, True))
    out.append(_txt(X(0), Y(haut) + 22, _m(f"béton E_c = {nombre(p['E'], 1)} GPa · h_c = α·min(B ; L) = {nombre(p['alpha'], 2)} × "
                                            f"{nombre(min(B, L), 0)} mm"), 12, couleur=ENCRE))
    if tg > 0:
        out.append(_txt(X(0), Y(haut) + 40, _m(f"mortier E_g = {nombre(p['Eg'], 1)} GPa · t_g = {nombre(tg, 0)} mm"), 12, couleur=ENCRE))
    pl = 170.0
    e2 = pl / max(B, L)
    px, py = 500 + (pl - B * e2) / 2, 150 + (pl - L * e2) / 2
    out.append(_txt(500 + pl / 2, 120, "Vue en plan", 12.5, gras=True))
    out.append(f'<rect x="{_f(px)}" y="{_f(py)}" width="{_f(B * e2)}" height="{_f(L * e2)}" fill="{PLAT}" stroke="{ENCRE}" stroke-width="1.8"/>')
    out.append(_cote_h(px, px + B * e2, py + L * e2 + 24, f"B = {nombre(B, 0)}") + _cote_v(px - 20, py, py + L * e2, f"L = {nombre(L, 0)}"))
    return _svg(ident, LW, HW, "".join(out), "Plat en béton", _hachures(ident))


# ------------------------------------------------------------------ les tableaux posés dans la toile
def tableau(classe: str, titre: str, entetes: List[str], lignes: List[List[str]], pied: str = "", alignes: str = "") -> str:
    """Un tableau HTML de la toile (vue « Tableau ») : en-têtes et cellules déjà mis en forme (HTML) ; ``alignes`` : une
    lettre par colonne, « g » à gauche, « d » à droite (par défaut : la première à gauche, les autres à droite)."""
    al = alignes or ("g" + "d" * (len(entetes) - 1))
    th = "".join(f'<th class="{"g" if a == "g" else "d"}">{h}</th>' for h, a in zip(entetes, al))
    tr = "".join("<tr>" + "".join(f'<td class="{"g" if a == "g" else "d"}">{c}</td>' for c, a in zip(li, al)) + "</tr>" for li in lignes)
    t = f'<div class="sol-t">{titre}</div>' if titre else ""
    p = f'<div class="sol-pied">{pied}</div>' if pied else ""
    return f'<table class="sol-tab {classe}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'.join([t, p])


def vue_tableaux(*blocs: str) -> str:
    """La vue d'un ou de plusieurs tableaux (un bloc HTML, pas un dessin) ; elle défile dans la feuille."""
    return f'<div class="sol-vue"><div class="sol-defile">{"".join(blocs)}</div></div>'
