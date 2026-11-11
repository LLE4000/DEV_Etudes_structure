# -*- coding: utf-8 -*-
"""Fondations de grues — les DESSINS, construits sur la seule sortie de ``moteur.verifier`` : rien n'est calculé ici.

Chaque dessin est une SCÈNE : une liste de primitives (traits, polygones, cercles, textes) dans un cadre, que ``svg``
écrit pour l'écran et que la note peindra telle quelle. Deux sortes de scènes :

    plan(res)                 la VUE EN PLAN, en mm et à l'échelle : pieux, semelles, poutres, axes, pieds de la grue et
                              châssis, cotes, réactions maximales — redessinée à chaque saisie ;
    diagrammes(res, cle)      pour UNE poutre ou UNE semelle : son schéma (la poutre sur deux appuis et ses positions de
                              tolérance), puis M et V — une courbe fine par position, l'enveloppe en trait fort, les
                              extremums cotés. Hors échelle.

    vues(res)  ->  [(clé, titre, titre court, à l'échelle, scène, légende)]
    svg(scene, ident, titre) -> str

Conventions des diagrammes : M positif (fibre inférieure tendue) dessiné VERS LE BAS, du côté de la fibre tendue ;
V positif vers le haut. Bibliothèque standard seulement.
"""
import html
import math

from . import entree as E
from . import geometrie as G

ENCRE, GRIS, GRIS_CLAIR, BETON, PIEU = "#0D1525", "#8B95A7", "#CBD5E1", "#EEF1F5", "#D7DEE8"
ROUGE, BLEU = "#B91C1C", "#2563EB"
COULEURS = {"A": "#2563EB", "B": "#D97706", "A.a": "#2563EB", "A.b": "#0891B2", "B.a": "#D97706", "B.b": "#9333EA"}
MOINS = "−"


def _fr(x: float, nd: int = 1) -> str:
    """Un nombre du dessin : virgule décimale, vrai signe moins, jamais « −0 »."""
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",").replace("-", MOINS)


class Scene:
    """Une scène : un cadre (x, y, largeur, hauteur) et des primitives, dans l'ordre où elles se peignent."""

    def __init__(self):
        self.items = []
        self.cadre = (0.0, 0.0, 100.0, 100.0)

    def ligne(self, x1, y1, x2, y2, trait=ENCRE, ep=1.0, tirets=None):
        self.items.append(("ligne", x1, y1, x2, y2, trait, ep, tirets))

    def poly(self, pts, trait=ENCRE, ep=1.0, rempli=None, ferme=True, tirets=None, opacite=None):
        self.items.append(("poly", [tuple(p) for p in pts], trait, ep, rempli, ferme, tirets, opacite))

    def cercle(self, x, y, r, trait=ENCRE, ep=1.0, rempli=None):
        self.items.append(("cercle", x, y, r, trait, ep, rempli))

    def texte(self, x, y, s, taille, ancre="middle", couleur=ENCRE, gras=False, angle=0.0):
        self.items.append(("texte", x, y, str(s), taille, ancre, couleur, gras, angle))


def svg(scene: Scene, ident: str, titre: str) -> str:
    """La scène en SVG autonome (aucune feuille de style, aucune ressource) ; tout texte est échappé."""
    x, y, w, h = scene.cadre
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:.1f} {y:.1f} {w:.1f} {h:.1f}" role="img" '
           f'aria-label="{html.escape(titre, quote=True)}" id="{html.escape(ident, quote=True)}" '
           f'font-family="Inter,system-ui,sans-serif">']
    for it in scene.items:
        k = it[0]
        if k == "ligne":
            _, x1, y1, x2, y2, trait, ep, tirets = it
            d = f' stroke-dasharray="{tirets}"' if tirets else ""
            out.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{trait}" stroke-width="{ep:.2f}"{d}/>')
        elif k == "poly":
            _, pts, trait, ep, rempli, ferme, tirets, opacite = it
            d = f' stroke-dasharray="{tirets}"' if tirets else ""
            o = f' fill-opacity="{opacite}"' if opacite is not None else ""
            p = " ".join(f"{a:.2f},{b:.2f}" for a, b in pts)
            out.append(f'<{"polygon" if ferme else "polyline"} points="{p}" fill="{rempli or "none"}"{o} stroke="{trait or "none"}" '
                       f'stroke-width="{ep:.2f}" stroke-linejoin="round"{d}/>')
        elif k == "cercle":
            _, cx, cy, r, trait, ep, rempli = it
            out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{rempli or "none"}" stroke="{trait}" stroke-width="{ep:.2f}"/>')
        else:
            _, tx, ty, s, taille, ancre, couleur, gras, angle = it
            r = f' transform="rotate({angle:.1f} {tx:.2f} {ty:.2f})"' if angle else ""
            g = ' font-weight="650"' if gras else ""
            out.append(f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="{taille:.2f}" text-anchor="{ancre}" fill="{couleur}"{g}{r}>'
                       f'{html.escape(s)}</text>')
    out.append("</svg>")
    return "".join(out)


# ================================================================== la vue en plan (mm, à l'échelle)
def _cote(sc: Scene, a, b, texte, fs, ep):
    """Une cote : le trait, ses deux arrêts à 45°, le texte au-dessus du milieu (ou à gauche d'une cote verticale)."""
    (x1, y1), (x2, y2) = a, b
    sc.ligne(x1, y1, x2, y2, GRIS, ep)
    t = fs * 0.35
    for (x, y) in (a, b):
        sc.ligne(x - t, y + t, x + t, y - t, ENCRE, ep * 1.4)
    if abs(y2 - y1) < 1e-6 * (abs(x2 - x1) + 1):
        sc.texte((x1 + x2) / 2, y1 - fs * 0.45, texte, fs, "middle", ENCRE)
    elif abs(x2 - x1) < 1e-6 * (abs(y2 - y1) + 1):
        sc.texte(x1 - fs * 0.45, (y1 + y2) / 2, texte, fs, "middle", ENCRE, angle=-90.0)
    else:                                                       # une cote oblique : le texte le long du trait
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if ang > 90 or ang < -90:
            ang += 180
        sc.texte((x1 + x2) / 2 + fs * 0.45 * math.sin(math.radians(ang)), (y1 + y2) / 2 - fs * 0.45 * math.cos(math.radians(ang)),
                 texte, fs, "middle", ENCRE, angle=ang)


def plan(res: dict) -> Scene:
    """La vue en plan : la géométrie de ``geometrie.plan`` (cm) dessinée en mm, y vers le haut."""
    f, v = res["fondation"], res["valeurs"]
    g = G.plan(f)
    K = 10.0                                                    # cm → mm

    def P(x, y):
        return (x * K, -y * K)

    x0, y0, x1, y1 = g["etendue"]
    larg = max(x1 - x0, y1 - y0) * K
    fs, ep = larg / 52, larg / 900
    sc = Scene()
    for p in g["pieces"]:                                       # les semelles sous les poutres : dessinées d'abord
        if p["cle"] in ("semelle", "primaire"):
            sc.poly([P(*q) for q in p["contour"]], ENCRE, ep * 1.6, BETON)
    for p in g["pieces"]:
        if p["cle"] not in ("semelle", "primaire"):
            sc.poly([P(*q) for q in p["contour"]], ENCRE, ep * 1.6, "#F7F8FA")
    for p in g["pieces"]:
        (ax, ay), (bx, by) = p["axe"]
        sc.ligne(*P(ax, ay), *P(bx, by), GRIS, ep, f"{fs * 1.2:.0f} {fs * 0.4:.0f} {fs * 0.2:.0f} {fs * 0.4:.0f}")
    if g["paroi"]:
        w = g["paroi"]
        y, pas = w["y0"], 0.8 * w["D"]
        while y <= w["y1"] + 1e-9:
            sc.cercle(*P(w["x"], y), w["D"] / 2 * K, ENCRE, ep, PIEU)
            y += pas
        sc.texte(P(w["x"], 0)[0] + w["D"] * K, P(0, w["y1"])[1] + fs, "paroi de pieux sécants", fs, "start", ENCRE)
    for p in g["pieux"]:
        cx, cy = P(p["x"], p["y"])
        r = p["D"] / 2 * K
        sc.cercle(cx, cy, r, ENCRE, ep * 1.4, PIEU)
        sc.ligne(cx - r * 1.25, cy, cx + r * 1.25, cy, GRIS, ep)
        sc.ligne(cx, cy - r * 1.25, cx, cy + r * 1.25, GRIS, ep)
    h = g["chassis"] / 2
    sc.poly([P(-h, -h), P(h, -h), P(h, h), P(-h, h)], BLEU, ep * 1.2, None, True, f"{fs * 0.6:.0f} {fs * 0.4:.0f}")
    sc.texte(0, 0, "châssis de la grue", fs, "middle", BLEU)
    c = g["cote_pied"] / 2
    for p in g["pieds"]:
        sc.poly([P(p["x"] - c, p["y"] - c), P(p["x"] + c, p["y"] - c), P(p["x"] + c, p["y"] + c), P(p["x"] - c, p["y"] + c)],
                ROUGE, ep * 1.8, "#FEE2E2")
        cx, cy = P(p["x"], p["y"])
        sc.texte(cx, cy + fs * 0.36, p["nom"], fs, "middle", ROUGE, True)
    for k in g["cotes"]:
        _cote(sc, P(*k["a"]), P(*k["b"]), k["texte"], fs, ep)
    # les noms des pièces et les réactions maximales
    lignes = []
    if f["type"] == E.SEMELLES8:
        su, sm = f["support"], f["semelle"]
        xa = -h - f["dist_a"]
        sc.texte(0, P(0, h)[1] - su["b"] / 2 * K - fs * 0.5, f"poutre support {su['b']:g} × {su['h']:g}", fs, "middle", ENCRE, True)
        sc.texte(P(xa, 0)[0], P(0, 0)[1], f"semelle {sm['L']:g} × {sm['b']:g} × {sm['h']:g}", fs, "middle", ENCRE, True, angle=-90.0)
        lignes = [f"RA max = {_fr(v['support.dim.RA_max'], 0)} kN sur une semelle · RB max = {_fr(v['support.dim.RB_max'], 0)} kN du côté b",
                  f"pieu Ø{f['pieux']['D']:g} : {_fr(v['pieux.sollicitation_T'])} T pour {_fr(f['pieux']['capacite'], 0)} T · "
                  f"tolérances e = {f['e_support']:g} cm (poutre), {f['e_semelle']:g} cm (semelle)"]
    elif f["type"] == E.CADRE4:
        po = f["poutre"]
        sc.texte(0, P(0, h)[1] - po["b"] / 2 * K - fs * 0.5, f"poutre {po['b']:g} × {po['h']:g}", fs, "middle", ENCRE, True)
        lignes = [f"charge sur un pieu Ø{f['pieux']['D']:g} : {_fr(v['pieux.charge'], 0)} kN pour {_fr(f['pieux']['capacite'], 0)} kN · "
                  f"tolérance e = {f['e']:g} cm"]
    else:
        se, pr = f["secondaire"], f["primaire"]
        sc.texte(0, P(0, h)[1] + se["b"] / 2 * K + fs * 1.2, f"poutre secondaire {se['b']:g} × {se['h']:g}", fs, "middle", ENCRE, True)
        sc.texte(0, P(0, -h)[1] - se["b"] / 2 * K - fs * 0.5, f"poutres primaires {pr['b']:g} × {pr['h']:g}, L = {pr['L']:g}, à 45°",
                 fs, "middle", ENCRE, True)
        lignes = [f"réaction maximale d'un pieu Ø{f['pieux']['D']:g} : {_fr(v['primaire.dim.R_max'], 0)} kN · "
                  f"tolérances e = {f['e_secondaire']:g} cm (secondaire), {f['e_primaire']:g} cm (primaire)"]
    marge = fs * 2.2
    bas = -y0 * K + marge
    par_ligne = int((x1 - x0) * K / (fs * 0.9 * 0.5))           # caractères qui tiennent dans la largeur du plan
    textes = []
    for i, s in enumerate(lignes):
        mots, courant = s.split(" "), ""
        for mot in mots:
            if courant and len(courant) + 1 + len(mot) > par_ligne:
                textes.append((courant, i < len(lignes)))
                courant = mot
            else:
                courant = (courant + " " + mot).strip()
        textes.append((courant, i < len(lignes)))
    for i, (s, fort) in enumerate(textes):
        sc.texte(x0 * K, bas + i * fs * 1.5, s, fs * 0.9, "start", ENCRE if fort else GRIS)
    n = len(textes)
    sc.cadre = (x0 * K - marge, -y1 * K - marge, (x1 - x0) * K + 2 * marge, (y1 - y0) * K + 2 * marge + n * fs * 1.5)
    return sc


# ================================================================== les diagrammes d'un élément (hors échelle)
LARGEUR, MG, MD = 660.0, 62.0, 26.0
RANGEE, H_M, H_V, ECART = 27.0, 160.0, 124.0, 42.0
FS = 13.0


def _couper(texte: str, n: int) -> list:
    """Le texte coupé aux espaces en lignes de ``n`` caractères au plus."""
    lignes, courant = [], ""
    for mot in texte.split(" "):
        if courant and len(courant) + 1 + len(mot) > n:
            lignes.append(courant)
            courant = mot
        else:
            courant = (courant + " " + mot).strip()
    return lignes + [courant] if courant else lignes


def _series(p: dict):
    """Les séries à tracer pour une position : l'enveloppe de ses colonnes (poutre support), sinon sa courbe."""
    if "enveloppe" in p:
        en = p["enveloppe"]
        return en["x"], [en["M_max"]], [en["V_max"], en["V_min"]]
    c = p["courbe"]
    return c["x"], [c["M"]], [c["V"]]


def _pas_abscisses(etendue: float) -> float:
    """Le pas des graduations de l'axe des abscisses (m) : 0,5 m, 1 m ou 2 m selon la longueur."""
    return 0.5 if etendue <= 3.2 else 1.0 if etendue <= 9.0 else 2.0


def diagrammes(res: dict, cle: str) -> Scene:
    """Le schéma et les diagrammes M et V d'un élément de ``res["elements"]``.

    Le schéma : UNE RANGÉE PAR POSITION de tolérance — la poutre, ses deux appuis là où cette position les met, ses
    charges ; un trait fin vertical marque la position NOMINALE des appuis, pour que le décalage se voie. Les
    diagrammes : une courbe fine par position, l'ENVELOPPE en trait fort (le plus grand moment positif et le plus grand
    moment négatif à chaque abscisse), ses extremums cotés ; un tireté repère les valeurs que le classeur retient."""
    el = next(e for e in res["elements"] if e["cle"] == cle)
    pos, env, dim = el["positions"], el["enveloppe"], el.get("dimensionnant", {})
    xmin, xmax = min(p["x0"] for p in pos), max(p["x1"] for p in pos)
    kx = (LARGEUR - MG - MD) / (xmax - xmin)

    def X(x):
        return MG + (x - xmin) * kx

    sc = Scene()
    # --- le schéma : une rangée par position (le nom de la pièce et sa tolérance sont dans l'en-tête de la vue)
    y1 = 40.0
    y_fin = y1 + (len(pos) - 1) * RANGEE
    na, nb = el["repere"]["appuis_nominaux"]
    for xn in (na, nb):                                         # la position nominale des appuis
        sc.ligne(X(xn), y1 - 8, X(xn), y_fin + 15, GRIS, 0.8, "3 3")
    noms = [ch["nom"] for ch in pos[0]["charges"]]
    nominaux = el["repere"].get("pieds_nominaux") or [ch["x"] for ch in pos[0]["charges"]]
    for k, xn in enumerate(nominaux):
        sc.texte(X(xn), y1 - 26, noms[k] if k < len(noms) else "", FS - 1.5, "middle", ENCRE)
    for i, p in enumerate(pos):
        coul, y = COULEURS.get(p["cle"], ENCRE), y1 + i * RANGEE
        sc.texte(MG - 12, y + 4, p["titre"].replace("CAS ", ""), FS - 1, "end", coul, True)
        sc.ligne(X(p["x0"]), y, X(p["x1"]), y, ENCRE, 2.4)
        for xa in p["appuis"]:
            sc.poly([(X(xa), y + 1.6), (X(xa) - 5.5, y + 11), (X(xa) + 5.5, y + 11)], coul, 1.0, coul, True, None, 0.85)
        for ch in p["charges"]:
            x = X(ch["x"])
            if ch.get("c"):                                     # la charge arrive par une platine de largeur c
                d = ch["c"] * kx / 2
                sc.poly([(x - d, y - 5), (x + d, y - 5), (x + d, y - 1.4), (x - d, y - 1.4)], coul, 0.8, coul, True, None, 0.35)
            sc.ligne(x, y - 19, x, y - 3, coul, 1.6)
            sc.poly([(x, y - 1.6), (x - 3.4, y - 9), (x + 3.4, y - 9)], coul, 0.5, coul)
    y_app = y_fin + 25
    sc.texte(X(na), y_app, "appui A", FS - 1.5, "middle", GRIS)
    sc.texte(X(nb), y_app, "appui B", FS - 1.5, "middle", GRIS)

    pas = _pas_abscisses(xmax - xmin)
    graduations = [k * pas for k in range(math.ceil(xmin / pas - 1e-9), math.floor(xmax / pas + 1e-9) + 1)]

    def diagramme(y_haut, hauteur, titre, unite, series_par_pos, env_haut, env_bas, positif_bas, reperes):
        haut = [max(v, 0.0) for v in env_haut]                  # l'enveloppe se lit par rapport à l'axe
        bas = [min(v, 0.0) for v in env_bas]
        vmax, vmin = max(haut), min(bas)
        etendue = (vmax - vmin) or 1.0
        if vmin > -0.005 * etendue:
            bas = None
        ky = (hauteur - 34) / etendue
        y_zero = y_haut + 17 + ((vmax if not positif_bas else -vmin) * ky)

        def Y(v):
            return y_zero + (v * ky if positif_bas else -v * ky)

        sc.texte(4, y_haut - 7, titre, FS, "start", ENCRE, True)
        for g in graduations:                                   # le quadrillage des abscisses, très léger
            sc.ligne(X(g), y_haut + 6, X(g), y_haut + hauteur - 6, GRIS_CLAIR, 0.6)
        enveloppes = [haut] + ([bas] if bas is not None else [])
        for serie in enveloppes:                                # une aire légère entre l'enveloppe et l'axe
            pts = [(X(env["x"][0]), y_zero)] + [(X(x), Y(v)) for x, v in zip(env["x"], serie)] + [(X(env["x"][-1]), y_zero)]
            sc.poly(pts, None, 0.0, ENCRE, True, None, 0.06)
        sc.ligne(X(xmin), y_zero, X(xmax), y_zero, ENCRE, 1.0)
        for p, (xs, series) in zip(pos, series_par_pos):
            for serie in series:
                sc.poly([(X(x), Y(v)) for x, v in zip(xs, serie)], COULEURS.get(p["cle"], GRIS), 1.3, None, False)
        for serie in enveloppes:
            sc.poly([(X(x), Y(v)) for x, v in zip(env["x"], serie)], ENCRE, 2.4, None, False)
        extremes, pris = [], []
        for serie, plus_grand in zip(enveloppes, (True, False)):
            i = (max if plus_grand else min)(range(len(serie)), key=lambda k: serie[k])
            if abs(serie[i]) >= 0.02 * etendue:                 # un extremum négligeable ne se cote pas (il reste tracé)
                extremes.append((serie[i], env["x"][i]))
        for val, x in extremes:
            sc.cercle(X(x), Y(val), 3.2, ENCRE, 1.0, "#FFFFFF")
            dessous = (val > 0) == positif_bas
            ancre = "start" if X(x) < LARGEUR * 0.5 else "end"
            texte = f"{_fr(val)} {unite} à x = {_fr(x, 2)} m"
            larg, xt, yt = len(texte) * FS * 0.56, X(x) + (7 if ancre == "start" else -7), Y(val) + (FS + 4 if dessous else -7)
            pris.append((xt if ancre == "start" else xt - larg, xt + larg if ancre == "start" else xt, yt))
            sc.texte(xt, yt, texte, FS, ancre, ENCRE, True)
        for nom, val in reperes:                                # ce que le classeur retient : un tireté, du côté libre
            if val is None or abs(val) < 0.02 * etendue or not vmin - 0.15 * etendue <= val <= vmax + 0.15 * etendue:
                continue
            texte = f"classeur : {nom} = {_fr(val)}"
            larg, yt = len(texte) * (FS - 1.5) * 0.56, Y(val) + (FS + 1 if (val > 0) == positif_bas else -5)

            def libre(x0, x1):
                return all(x1 < a - 6 or x0 > b + 6 or abs(yt - y) > FS + 1 for a, b, y in pris)

            a_droite = libre(X(xmax) - 4 - larg, X(xmax) - 4) or not libre(X(xmin) + 4, X(xmin) + 4 + larg)
            sc.ligne(X(xmin), Y(val), X(xmax), Y(val), GRIS, 0.9, "5 4")
            sc.texte(X(xmax) - 4 if a_droite else X(xmin) + 4, yt, texte, FS - 1.5, "end" if a_droite else "start", GRIS)

    ym = y_app + 32
    donnees = [_series(p) for p in pos]
    diagramme(ym, H_M, "M (kN·m) — tracé du côté de la fibre tendue", "kN·m", [(xs, m) for xs, m, _ in donnees],
              env["M_max"], env["M_min"], True, [("M max", dim.get("M_max")), ("M min", dim.get("M_min"))])
    yv = ym + H_M + ECART
    v_dim = dim.get("V_max")
    if v_dim is not None and -min(env["V_min"]) > max(env["V_max"]):
        v_dim = -v_dim                                          # le repère du tranchant du côté où l'enveloppe est la plus forte
    diagramme(yv, H_V, "V (kN)", "kN", [(xs, v) for xs, _, v in donnees], env["V_max"], env["V_min"], False,
              [("V max", v_dim)])
    # --- l'axe des abscisses
    ya = yv + H_V + 10
    sc.ligne(X(xmin), ya, X(xmax), ya, GRIS, 0.8)
    for g in graduations:
        sc.ligne(X(g), ya, X(g), ya + 4, GRIS, 0.8)
        sc.texte(X(g), ya + 4 + FS - 1.5, _fr(g, 1 if pas < 1 else 0), FS - 1.5, "middle", GRIS)
    sc.texte(LARGEUR - 2, ya + 4 + FS - 1.5, "x (m)", FS - 1.5, "end", GRIS)
    # --- la légende : une entrée par position, puis l'enveloppe
    y = ya + 40
    par_ligne = int((LARGEUR - MG - 40) / ((FS - 1.5) * 0.5))
    for p in pos:
        sc.ligne(MG - 30, y - 4, MG - 6, y - 4, COULEURS.get(p["cle"], GRIS), 2.0)
        lignes = _couper(f"{p['titre']} — {p['texte']}", par_ligne)
        if p.get("colonne"):                                    # la colonne du classeur qui gouverne : si elle tient sur la ligne
            suite = "gouverne : " + p["colonne"].strip("()")
            if len(lignes[-1]) + 3 + len(suite) <= par_ligne:
                lignes[-1] += " · " + suite
        for ligne in lignes:
            sc.texte(MG, y, ligne, FS - 1.5, "start", ENCRE)
            y += 15.5
        y += 2.5
    sc.ligne(MG - 30, y - 4, MG - 6, y - 4, ENCRE, 2.6)
    sc.texte(MG, y, "enveloppe de toutes les positions" + (" et de tous les cas de la grue" if "enveloppe" in pos[0] else ""),
             FS - 1.5, "start", ENCRE, True)
    sc.cadre = (0.0, 0.0, LARGEUR, y + 12)
    return sc


def vues(res: dict) -> list:
    """Les vues d'une fondation : le plan, puis les diagrammes de chaque poutre ou semelle."""
    f = res["fondation"]
    conventions = G.plan(f)["conventions"]
    out = [("plan", "Vue en plan", "Plan", True, plan(res),
            "Plan de principe, cotes en cm : ni plan de coffrage, ni plan d'armatures. " + " ; ".join(c[:1].upper() + c[1:] for c in conventions) + ".")]
    for el in res["elements"]:
        out.append((f"mv_{el['cle']}", f"{el['nom']} · M et V", el["nom"].replace("Poutre ", "P. ").replace(" sur pieux", ""), False,
                    diagrammes(res, el["cle"]),
                    "Une rangée et une courbe par position de tolérance ; l'enveloppe en trait fort ; en tireté, les valeurs que le classeur "
                    "retient. M est tracé du côté de la fibre tendue (positif vers le bas), abscisses en m depuis l'appui A."))
    return out
