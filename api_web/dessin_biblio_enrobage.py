# -*- coding: utf-8 -*-
"""Le DÉTAIL D'ENROBAGE de la Bibliothèque Béton armé (rubrique « Enrobages », onglet « Calcul ») dessiné à l'échelle
(mm) à partir du résultat de ``beton/bibliotheque/enrobage.calculer`` — rien n'est recalculé ici : chaque cote est une
grandeur que le moteur nomme (c_nom, c_min, Δc_dev, a_réel, a et a_sd exigés).

- poutre, poteau : un DÉTAIL D'ANGLE de la coupe (deux parements, l'étrier qui contourne la barre d'angle) ; pour une
  poutre à un seul lit, la distance a_sd exigée des barres d'angle est repérée sur la face latérale ;
- dalle, voile : une COUPE de la face exposée, les barres principales en rangée derrière la nappe extérieure s'il y en
  a une (Ø_ext, 0 par défaut : aucun étrier n'est retiré de a).

La bande bleue va de c_min à c_nom,dur = c_min + Δc_dev (la tolérance d'exécution) ; quand le feu ou le coulage contre
le sol gouverne, c_nom dépasse c_nom,dur et le béton au-delà de la bande est l'enrobage supplémentaire. Les exigences
du feu sont en rouge. L'espacement des barres dessinées est une convention de dessin (pas une donnée du calcul) et n'est
pas coté. La Toile et le style sont ceux de ``api_web/dessin_enrobage.py`` (l'ancien module), repris sans copie.
"""
from html import escape

from .dessin_enrobage import TOLERANCE, Toile, _boite, _f, _indice, _mm, _renvois
from .dessin_poutre import ACIER, BETON, COTE, EFFORT, ENCRE

TITRES = {"poutre": "Détail d'angle de la poutre", "poteau": "Détail d'angle du poteau", "dalle": "Coupe de la dalle",
          "voile": "Coupe horizontale du voile"}


def _grandeurs(res: dict) -> tuple:
    """(Ø barre, Ø extérieur, c_nom, Δc_dev, c_min, a exigé, a_sd exigé, durée) — lus dans le résultat du moteur."""
    p, d, f = res["entree"], res["durabilite"], res["feu"]
    a = a_sd = None
    r = ""
    if f is not None and f.get("a") is not None:
        a = float(f["a"])
        a_sd = float(f["a_sd"]) if (f.get("a_sd") is not None and f.get("angle")) else None
        r = f["duree"]
    return (float(p["phi"]), float(res["phi_ext"]), float(res["c_nom"]), float(d["delta_dev"]), float(d["c_min"]), a, a_sd, r)


def _angle(ident: str, res: dict) -> str:
    """Poutre, poteau : le détail d'angle de la coupe (deux parements, l'étrier qui contourne la barre d'angle)."""
    d, e, c, dd, c_min, a_ex, a_sd, duree = _grandeurs(res)
    pas = d + max(d, 20.0)                                       # convention de dessin : l'entraxe des barres dessinées
    xc0 = c + e + d / 2                                          # l'axe de la barre d'angle : a_réel
    W = max(120.0, xc0 + pas + d / 2 + max(40.0, d))
    H = W
    u = 1.2 * H / 23
    T = Toile(u)
    g = T.g
    sw = u * 0.085
    tir = f"{_f(u * 0.5)} {_f(u * 0.3)}"
    fin = f"{_f(u * 0.3)} {_f(u * 0.25)}"
    c_dur = c_min + dd
    # ---- le béton, la bande de tolérance Δc_dev (de c_min à c_nom,dur), la ligne de c_min en tirets, les parements
    g.append(f'<rect x="0" y="0" width="{_f(W)}" height="{_f(H)}" fill="{BETON}"/>')
    if dd > 0:
        g.append(f'<path d="M{_f(c_min)} 0 V{_f(H - c_min)} H{_f(W)} V{_f(H - c_dur)} H{_f(c_dur)} V0 Z" fill="{TOLERANCE}" fill-opacity=".14"/>')
    g.append(f'<path d="M{_f(c_min)} 0 V{_f(H - c_min)} H{_f(W)}" fill="none" stroke="{TOLERANCE}" stroke-width="{_f(sw * 1.1)}" '
             f'stroke-dasharray="{tir}"/>')
    g.append(f'<path d="M0 0 V{_f(H)} H{_f(W)}" fill="none" stroke="{ENCRE}" stroke-width="{_f(sw * 2.6)}" stroke-linejoin="miter"/>')
    T.rupture(0, 0, W, 0)
    T.rupture(W, 0, W, H)
    # ---- l'étrier (il contourne la barre d'angle), les barres
    yc = H - xc0
    if e > 0:
        r = d / 2 + e / 2
        g.append(f'<path d="M{_f(c + e / 2)} 0 V{_f(yc)} A{_f(r)} {_f(r)} 0 0 0 {_f(xc0)} {_f(yc + r)} H{_f(W)}" fill="none" '
                 f'stroke="{ACIER}" stroke-width="{_f(e)}" stroke-linecap="butt"/>')
    xs = [xc0, xc0 + pas]
    for x in xs:
        g.append(f'<circle cx="{_f(x)}" cy="{_f(yc)}" r="{_f(d / 2)}" fill="{ENCRE}"/>')
    # ---- sous le parement : c_nom, puis a_réel (à l'axe de la barre) ; la distance exigée par le feu en rouge
    t_c = _indice("c", "nom", f" = {_mm(c)}", u)
    t_a = (_indice("a", "réel", f" = {_mm(c)} + {_mm(e)} + {_mm(d / 2)} = {_mm(xc0)}", u) if e > 0
           else _indice("a", "réel", f" = {_mm(c)} + {_mm(d / 2)} = {_mm(xc0)}", u))
    y1, y2 = H + 2.0 * u, H + 4.3 * u
    T.ligne(0, H + u * 0.4, 0, y2 + u * 0.5)
    T.ligne(c, H - c, c, y1 + u * 0.5, COTE, 0.8, fin)
    T.ligne(xc0, yc, xc0, y2 + u * 0.5, COTE, 0.8, fin)
    c_dedans = T.tient(0, c, t_c)
    T.cote_h(0, c, y1, t_c, dehors="gauche")
    T.cote_h(0, xc0, y2, t_a)
    # la même cote c_nom sur le parement latéral
    xv = -2.0 * u
    T.ligne(-u * 0.4, H - c, xv - u * 0.5, H - c, COTE, 0.8, fin)
    T.ligne(-u * 0.4, H, xv - u * 0.5, H)
    T.cote_v(xv, H - c, H, t_c)
    gauche_txt = 0.0
    if a_ex is not None:
        t_feu = f"a ≥ {_mm(a_ex)} · {escape(duree)}"
        g.append(f'<path d="M{_f(a_ex)} {_f(y2)} l{_f(-u * 0.38)} {_f(u * 0.75)} h{_f(u * 0.76)} Z" fill="{EFFORT}"/>')
        T.txt(max(a_ex, T.largeur(t_feu, 0.86) / 2 - 3.2 * u), y2 + u * 1.85, t_feu, 0.86, couleur=EFFORT, gras=True)
    if a_sd is not None:
        # poutre à un lit : les barres d'angle, depuis la face latérale — repère sur une cote verticale jusqu'à l'axe
        xv2 = -4.4 * u
        T.ligne(xv - u * 0.5, yc, xv2 - u * 0.5, yc, COTE, 0.8, fin)
        T.ligne(xv - u * 0.5, H, xv2 - u * 0.5, H)
        T.cote_v(xv2, yc, H, None)
        g.append(f'<path d="M{_f(xv2)} {_f(H - a_sd)} l{_f(-u * 0.75)} {_f(-u * 0.38)} v{_f(u * 0.76)} Z" fill="{EFFORT}"/>')
        t_sd = _indice("a", "sd", f" ≥ {_mm(a_sd)}", u)
        T.txt(xv2 - u * 0.95, H - a_sd + u * 0.33, t_sd, 0.86, "end", couleur=EFFORT, gras=True)
        gauche_txt = 4.4 * u + 0.95 * u + T.largeur(t_sd, 0.86) + 0.6 * u
    # ---- les renvois, en marge droite
    renvois = [((xs[-1] + d / 2 * 0.7, yc - d / 2 * 0.7), f"barre Ø{_mm(d)}", ENCRE)]
    if e > 0:
        renvois.append(((W - 0.5 * (W - xs[-1] - d / 2), H - c - e / 2), f"étrier Ø{_mm(e)}", ACIER))
    if dd > 0:
        renvois.append(((W - u * 1.2, H - c_min - dd / 2), _indice("Δc", "dev", f" = {_mm(dd)}", u), TOLERANCE))
    renvois.append(((W - u * 2.4, H - c_min), _indice("c", "min", f" = {_mm(c_min)}", u), TOLERANCE))
    haut = min(-1.2 * u, _renvois(T, renvois, W + 1.8 * u, H + 0.6 * u) - 1.2 * u)
    # ---- la boîte
    gauche = 3.6 * u if c_dedans else max(3.6 * u, T.largeur(t_c, 0.92) + 1.8 * u)
    gauche = max(gauche, gauche_txt)
    droite = max(T.largeur(t, 0.95) for _, t, _ in renvois) + 3.4 * u
    bas = 7.6 * u if a_ex is not None else 5.6 * u
    return _boite(ident, T, -gauche, haut, W + droite, H + bas, TITRES[res["entree"]["element"]], c)


def _face(ident: str, res: dict) -> str:
    """Dalle, voile : la coupe de la face exposée, les barres principales en rangée derrière la nappe extérieure."""
    d, e, c, dd, c_min, a_ex, _, duree = _grandeurs(res)
    pas = d + max(d, 20.0)                                       # convention de dessin : l'entraxe des barres dessinées
    xc0 = c + e + d / 2                                          # a_réel (parement → axe des barres)
    x0 = max(30.0, 1.2 * pas)
    W = x0 + 2 * pas + d / 2 + max(30.0, d)
    H = max(0.65 * W, c + e + d + max(50.0, 1.5 * d))
    u = max(W, 1.3 * H) / 13.5
    T = Toile(u)
    g = T.g
    tir = f"{_f(u * 0.5)} {_f(u * 0.3)}"
    fin = f"{_f(u * 0.3)} {_f(u * 0.25)}"
    # ---- le béton : le parement exposé en trait fort en bas, coupé à gauche et à droite, continu en haut
    g.append(f'<rect x="0" y="0" width="{_f(W)}" height="{_f(H)}" fill="{BETON}"/>')
    if dd > 0:
        g.append(f'<rect x="0" y="{_f(H - c_min - dd)}" width="{_f(W)}" height="{_f(dd)}" fill="{TOLERANCE}" fill-opacity=".14"/>')
    T.ligne(0, H - c_min, W, H - c_min, TOLERANCE, 1.3, tir)
    g.append(f'<line x1="0" y1="{_f(H)}" x2="{_f(W)}" y2="{_f(H)}" stroke="{ENCRE}" stroke-width="{_f(u * 0.085 * 2.6)}"/>')
    T.rupture(0, H, 0, 0)
    T.rupture(W, 0, W, H)
    T.ligne(0, 0, W, 0, ENCRE, 0.6, f"{_f(u * 0.9)} {_f(u * 0.5)}")
    # ---- la nappe extérieure (s'il y en a une), puis les barres principales
    yc = H - xc0
    if e > 0:
        g.append(f'<rect x="0" y="{_f(H - c - e)}" width="{_f(W)}" height="{_f(e)}" fill="{ACIER}"/>')
    xs = [x0, x0 + pas, x0 + 2 * pas]
    for x in xs:
        g.append(f'<circle cx="{_f(x)}" cy="{_f(yc)}" r="{_f(d / 2)}" fill="{ENCRE}"/>')
    # ---- les cotes verticales à gauche (c_nom, a_réel), la distance exigée au feu en rouge ; leurs textes à
    # l'horizontale, en marge, reliés par un trait (une cote de 20 mm ne porte pas un texte lisible)
    x1, x2 = -1.3 * u, -2.6 * u
    T.ligne(-u * 0.4, H, x2 - u * 0.5, H)
    T.ligne(0, H - c, x1 - u * 0.5, H - c, COTE, 0.8, fin)
    T.ligne(xs[0], yc, x2 - u * 0.5, yc, COTE, 0.8, fin)
    T.cote_v(x1, H - c, H, None)
    T.cote_v(x2, yc, H, None)
    gauches = [(x1, H - c / 2, _indice("c", "nom", f" = {_mm(c)}", u), ENCRE, False),
               (x2, H - xc0 / 2, _indice("a", "réel", f" = {_mm(xc0)}", u), ENCRE, False)]
    if a_ex is not None:
        g.append(f'<path d="M{_f(x2)} {_f(H - a_ex)} l{_f(-u * 0.75)} {_f(-u * 0.38)} v{_f(u * 0.76)} Z" fill="{EFFORT}"/>')
        gauches.append((x2 - u * 0.75, H - a_ex, f"a ≥ {_mm(a_ex)} · {escape(duree)}", EFFORT, True))
    gauches.sort(key=lambda r: r[1])
    xt = x2 - 1.5 * u
    ys = []
    for _, y, _, _, _ in gauches:
        ys.append(y if not ys else max(y, ys[-1] + 1.45 * u))
    deb = min(0.0, H + 0.3 * u - ys[-1])
    ys = [y + deb for y in ys]
    for (xa, ya, t, coul, gras), y in zip(gauches, ys):
        T.ligne(xa, ya, xt + u * 0.3, y, coul, 0.9)
        T.txt(xt, y + u * 0.33, t, 0.9, "end", couleur=coul, gras=gras)
    # ---- sous le parement : son nom, puis la distance à l'axe décomposée
    T.txt(W / 2, H + 1.25 * u, "parement exposé", 0.8, couleur=COTE, italique=True)
    chiffres = f"{_mm(c)} + {_mm(e)} + {_mm(d / 2)} = {_mm(xc0)}" if e > 0 else f"{_mm(c)} + {_mm(d / 2)} = {_mm(xc0)}"
    t_a_long = (_indice("a", "réel", " = c", u) + _indice("", "nom", " + Ø", u) + (_indice("", "ext", " + Ø/2", u) if e > 0 else "/2")
                + f" = {chiffres}")
    T.txt(W / 2, H + 2.75 * u, t_a_long, 0.9)
    # ---- les renvois, en marge droite
    renvois = [((xs[-1] + d / 2 * 0.7, yc - d / 2 * 0.7), f"barre Ø{_mm(d)}", ENCRE)]
    if e > 0:
        renvois.append(((W - 0.3 * (W - xs[-1]), H - c - e / 2), f"nappe extérieure Ø{_mm(e)}", ACIER))
    if dd > 0:
        renvois.append(((W - u * 1.0, H - c_min - dd / 2), _indice("Δc", "dev", f" = {_mm(dd)}", u), TOLERANCE))
    renvois.append(((W - u * 2.0, H - c_min), _indice("c", "min", f" = {_mm(c_min)}", u), TOLERANCE))
    haut = min(-0.9 * u, _renvois(T, renvois, W + 1.6 * u, H + 0.6 * u) - 1.1 * u, ys[0] - 1.1 * u)
    # ---- la boîte
    gauche = max(T.largeur(t, 0.9) for _, _, t, _, _ in gauches) - xt + 0.8 * u
    droite = max(T.largeur(t, 0.95) for _, t, _ in renvois) + 3.0 * u
    larg_a = T.largeur(t_a_long, 0.9) / 2 - W / 2
    return _boite(ident, T, -max(gauche, larg_a + u), haut, W + max(droite, larg_a + u), H + 3.5 * u,
                  TITRES[res["entree"]["element"]], c)


def detail(ident: str, res: dict) -> str:
    """Le dessin de l'enrobage d'un élément (le résultat de ``beton/bibliotheque/enrobage.calculer``) : le détail
    d'angle d'une poutre ou d'un poteau, la coupe de la face exposée d'une dalle ou d'un voile."""
    return _angle(ident, res) if res["entree"]["element"] in ("poutre", "poteau") else _face(ident, res)
