# -*- coding: utf-8 -*-
"""Tests des SCHÉMAS de la cornière d'appui (acier/corniere/schemas.py : les quatre vues partagées par l'écran et la
note ; api_web/dessin_corniere.py : la couche mince qui les pose dans la toile) — décisions
docs/acier/CORNIERE_DECISIONS.md § 4 et § 9.1 à 9.3 (révision du 10/11 au soir). Rien n'est recalculé : chaque
contrôle compare le DESSIN à la sortie du moteur (acier.corniere.moteur.verifier), et chaque garantie rougit si on la
retire (§ 12, mutations).

  1. VUES : quatre vues (coupe, moments, contraintes, élévation) au format des modules du site (cle, titre, court, svg,
     legende, echelle) ; SVG bien formés (analyseur XML) ; aucune valeur indéfinie ; aucun indice « _ » brut dans les
     textes ; texte à 13 px dans la toile de l'établi (les quatre vues sont à l'échelle de la cornière) ; légende qui
     dit la configuration, le mode et l'ÉTAT LIMITE ; vingt cas (A historique, A équilibre, A avec jeux, B à Y = 85 et
     Y = 45, L150×100×12 dans les deux orientations, L 40×40×4, L 200×200×20, a = 100, A et B sur poutre, poutre
     haute (coupures), traversant à deux rondelles, chimique sans rondelle, ELU à 60 kN, L = 900 (coupures), B chargée
     à 144 %, A à 102 %, entraxe saisi).
  2. COUPE À L'ÉCHELLE (unités SVG = mm) : le contour de la cornière mesure larg × haut, l'aile horizontale est en bas
     (A : y ≤ 0) ou en haut (B : y ≥ 0) ; l'axe de l'ancrage à Y de la face extérieure, les positions Y ± dy ; la
     charge à X du nu du mur, les positions X ± dx ; la résultante de compression au pivot (A : Y′ − bras = y_c ou
     jeu_ancrage) ou au contact du bout libre (B : haut − c) ; la ligne u_V en équilibre.
  3. SECTIONS 1 à 4 exactement : S1 en u = e + r et S2 en u = e à travers l'aile horizontale, S3 en y = e + r et S4 en
     y = e à travers l'aile verticale, depuis la bonne face selon A/B ; les quatre pastilles numérotées.
  4. BRAS DU MODE : historique → les cotes Xe, Xe+r, Ye, Ye+r, YH avec les valeurs du classeur ; équilibre A → X − uV
     et Y − yc ; B → ya − c ; les cotes X, Y, larg, haut, e, r toujours ; les valeurs R, H, C écrites.
     SUPPORT ET ANCRAGE (§ 9.2, 9.3) : le voile sans limite (rompu) ou la poutre entre ses deux arêtes (y = −haut
     poutre, +bas poutre), cotées depuis l'axe d'ancrage le plus proche (« bord haut », « bord bas » = bords_poutre),
     coupée (trois morceaux, lignes de rupture) quand elle dépasse 1,5 × haut ; la tige jusqu'à −profondeur et la cote
     « hef » (« ép. » pour le traversant) ; les rondelles (0, 1, 2) aux dimensions de la table ISO, puis l'écrou ; le
     rendu du type : douille (mécanique), scellement hachuré (chimique), écrou et rondelle derrière un support
     d'épaisseur « profondeur » à face arrière droite (traversant) ; la tige coupée quand la cheville est profonde.
  5. DATA-SEL : chaque pièce qu'une donnée pilote porte la clé de sa donnée (X, dx, Y, dy, R, d, profil, yc ou
     jeu_ancrage, uV en équilibre ; L, a, n, d, X, Y, profil sur l'élévation) ; aucune clé hors de CLES_SEL ; rien sans
     ``interactif``.
  6. MOMENTS SUR LA CORNIÈRE : la cornière larg × haut ; l'axe M = 0 de chaque aile tracé et nommé ; le diagramme de
     chaque aile PERPENDICULAIRE à l'aile, depuis son axe, hors de l'angle (horizontale : vers le bas en A, vers le
     haut en B — le signe de B porté de l'autre côté ; verticale : côté support), à une ÉCHELLE COMMUNE (le plus grand
     moment entre 0,6 et 0,8 × haut) ; autant de courbes que le moteur (extrêmes fines, nominale moyenne), l'enveloppe
     en gras avec autant de points que le moteur, REMPLIE jusqu'au bout du prolongement (un polygone fermé sur l'axe par
     aile, un sommet à Mmax au nu du support, au pivot pour l'aile verticale) ; les courbes de l'aile verticale quittent
     l'axe à Y − dy, Y, Y + dy (trois zéros) ; M1 à M4 écrits avec la valeur de l'enveloppe (grandeur, 3 décimales),
     Mmax en bas ET à gauche (historique), « M angle » en B ; l'ancrage (axe, bande ± dy) ; R.
  7. CONTRAINTES, MAILLAGE : (len(s) − 1) × 12 mailles par aile (96 × 12), posées sur la vraie géométrie DÉFORMÉE —
     les nœuds de la face extérieure aux abscisses s du moteur, ceux de la face du congé à t(s), chacun déplacé du
     déplacement RÉEL du moteur (déformée élastique de la combinaison gouvernante) amplifié pour que le plus grand
     fasse PART_DEFORMEE × haut, le bloc d'angle translaté d'un bloc — et 12 × 12 dans le coin (le champ « coin » du
     moteur) ; CHAQUE maille porte la couleur du plus grand des taux de ses quatre nœuds selon l'échelle IMPOSÉE,
     recopiée ici depuis les décisions (bleu #2563EB, cyan #06B6D4 à 25 %, vert #22C55E à 50 %, jaune #FACC15 à 75 %,
     orange #F97316 à 100 %, → #C2410C à 105 %, rouge #DC2626 au-delà) ; la maille la plus chargée porte la couleur du
     taux max ; le ◉ de chaque aile à maxima[aile]["s"] sur la face ζ du maillage déformé, avec le taux de la
     vérification ; plus de bloc d'angle masqué ; le contour d'ORIGINE en tirets (« hd ») et le contour déformé
     par-dessus ; les pastilles 1 à 4 HORS de la pièce, reliées à la face déformée par un pointillé « sl » qui s'arrête
     au contour (rien ne traverse le dessin) ; l'axe de l'ancrage en repères courts hors de la pièce ; la barre de
     couleur de 0 au MAXIMUM du maillage (sa valeur écrite ; 50, 100 et 105 % seulement s'ils sont sous le maximum et à
     plus de ECART_GRADUATION × max de toute graduation gardée), graduée des deux côtés (N/mm² à gauche, % à droite) ;
     les trois lignes « σ = … », « coin : sections 2 et 4 prolongées · pas un calcul par éléments finis » et
     « déformée élastique · combinaison · u max = … mm · × amplification ».
  8. ÉLÉVATION : n ancrages aux positions z_i = e₁ + i·s du moteur à la hauteur Y, la chaîne de cotes e₁ · s · … · e₁,
     L, haut, larg ; N et V PAR ANCRAGE dans l'état limite saisi (« N ELS », « N ELU ») ; la charge de largeur a et le
     trapèze à 45° dont la base à la section 1 mesure L_eff ; sans a : « sur L » ; COUPURES sur L = 900 (la tranche
     dessinée plus courte, la cote L et la chaîne gardent leurs vraies valeurs).
  9. VUE IMPOSSIBLE : une tranche de 10 m face à une L 40 → cadre « Dessin impossible à cette échelle » pour
     l'élévation seule ; R = 0 et une saisie libre sans congé se dessinent ; sans diagrammes, moments et contraintes
     refusent avec un message clair.
 10. PEINTRES : les quatre vues se peignent en PDF par le peintre SVG de la note (ndc_pdf.svg.peindre_svg) et par le
     peintre de scène du kit (rapport.peindre), et leurs textes s'y retrouvent.
 11. PURETÉ : schemas.py n'importe ni Streamlit, ni numpy, ni le moteur, ni api_web, et ne calcule rien ; le dégradé
     aux bornes imposées.
 12. MUTATIONS : le module rechargé depuis sa source, modifié d'UN trait (section 1 déplacée, B dessiné comme A,
     charge décalée, ancrages décalés sur l'élévation, data-sel oublié, diagramme porté dans l'angle, une courbe de
     moment oubliée, enveloppe non remplie, enveloppe perdue, dégradé inversé, coin non maillé, barre sans les N/mm²,
     maille colorée par le plus petit taux, maximum sur la mauvaise face, déformée non dessinée, barre jusqu'à 105 %
     quel que soit le maillage, repère de section à travers le dessin, profondeur non dessinée, cote L prise sur la
     longueur dessinée, section 3 depuis la mauvaise face en B, ancrage compté depuis l'angle), est rejeté par les
     contrôles de ce fichier ; un témoin inoffensif ne l'est pas.

    /usr/bin/python3.12 tests/test_corniere_schemas.py
"""
import io
import math
import os
import re
import sys
import types
import xml.etree.ElementTree as ET

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from acier.assemblages.poutre_poutre.doubles_cornieres.rapport import peindre  # noqa: E402
from acier.corniere import moteur as M  # noqa: E402
from acier.corniere import schemas as SC  # noqa: E402
from api_web import dessin_corniere as DC  # noqa: E402
from ndc_pdf.svg import peindre_svg  # noqa: E402

OK, KO = [], []
TOILE = (626.0, 682.0)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


def pres(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def fr(x, nd):
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def f0(x):
    """0 ou 1 décimale, comme les cotes du kit."""
    return fr(x, 1 if abs(x - round(x)) > 0.05 else 0)


def kn(x):
    """Un effort comme le dessin l'écrit : deux décimales sans zéro inutile, « kN »."""
    t = fr(x, 2)
    if "," in t:
        t = t.rstrip("0").rstrip(",")
    return t + " kN"


# ------------------------------------------------------------------ les cas
def cas(**kw):
    return dict(M.nouvelle_corniere(), **{"R": 42.76, **kw})


CAS = {
    "A historique (cas de contrôle)": cas(dx=0.0, dy=0.0, R_ELU=68.9),
    "A historique avec jeux": cas(),
    "A équilibre": cas(mode="equilibre"),
    "B Y = 85": cas(configuration="B"),
    "B Y = 45": cas(configuration="B", Y=45.0, dy=5.0),
    "L150×100×12, 150 horizontal": cas(profil="L 150x100x12", aile_horizontale="A", Y=65.0),
    "L150×100×12, 100 horizontal": cas(profil="L 150x100x12", aile_horizontale="B", Y=115.0),
    "L 40×40×4": dict(M.nouvelle_corniere(), profil="L 40x40x4", R=2.0, X=25.0, dx=5.0, Y=25.0, dy=3.0, d=8.0, L=120.0, n=2, profondeur=60.0),
    "L 200×200×20": dict(M.nouvelle_corniere(), profil="L 200x200x20", R=120.0, X=120.0, Y=140.0, d=24.0, L=400.0),
    "A équilibre, a = 100": cas(mode="equilibre", a=100.0),
    "A sur poutre": cas(support="poutre", support_haut=160.0, support_bas=120.0),
    "B sur poutre": cas(configuration="B", support="poutre"),
    "poutre haute (coupures)": cas(support="poutre", support_haut=420.0, support_bas=300.0),
    "traversant, 2 rondelles": cas(type_ancrage="traversant", rondelles=2, profondeur=140.0),
    "chimique sans rondelle": cas(type_ancrage="chimique", rondelles=0),
    "ELU R = 60": cas(etat_limite="ELU", R=60.0),
    "L = 900 (coupures)": cas(L=900.0),
    "B chargée (144 %)": cas(configuration="B", dx=20.0),
    "A à 102 %": cas(R=49.1, dx=0.0, dy=0.0),
    "entraxe saisi": cas(n=2, entraxe=150.0),
}
RES = {nom: M.verifier(c) for nom, c in CAS.items()}


# ------------------------------------------------------------------ lecture des SVG
def racine(svg):
    return ET.fromstring(svg)


def tag(el):
    return el.tag.split("}")[-1]


def classes(el):
    return (el.get("class") or "").split()


def points_d(d):
    """Les points d'un attribut ``d`` du kit (M, L, H, V), sous-chemins à plat."""
    pts, x, y = [], 0.0, 0.0
    for cmd, args in re.findall(r"([MLHV])([^MLHV]*)", d):
        nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", args)]
        if cmd in "ML":
            for i in range(0, len(nums) - 1, 2):
                x, y = nums[i], nums[i + 1]
                pts.append((x, y))
        elif cmd == "H":
            for v in nums:
                x = v
                pts.append((x, y))
        else:
            for v in nums:
                y = v
                pts.append((x, y))
    return pts


def sous_chemins(d):
    """Les sous-chemins (un par « M ») d'un attribut ``d``."""
    return [points_d("M" + part) for part in d.split("M") if part.strip()]


def points_poly(el):
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", el.get("points") or "")]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def tous(root, nom=None, cls=None):
    out = []
    for el in root.iter():
        if nom and tag(el) != nom:
            continue
        if cls and cls not in classes(el):
            continue
        out.append(el)
    return out


def groupe_sel(root, cle):
    return [g for g in root.iter() if tag(g) == "g" and g.get("data-sel") == cle]


def sels(svg):
    return sorted(set(re.findall(r'data-sel="([^"]+)"', svg)))


def textes(root):
    return [("".join(t.itertext())).strip() for t in tous(root, "text")]


def dim(root, ident):
    """Le texte de la cote ``data-dim=ident`` (None si absente)."""
    for g in root.iter():
        if tag(g) == "g" and g.get("data-dim") == ident:
            return "".join(g.itertext()).strip()
    return None


def tailles_px(svg):
    """La plus petite et la plus grande taille de texte du SVG, en pixels dans la toile de l'établi."""
    vb = svg.split('viewBox="', 1)[1].split('"', 1)[0].split()
    k = min(TOILE[0] / float(vb[2]), TOILE[1] / float(vb[3]))
    fs = [float(x) for x in re.findall(r'font-size="([0-9.]+)"', svg)]
    return min(fs) * k, max(fs) * k


def cote(root, ident):
    """Les deux bouts de la ligne de cote ``data-dim=ident`` (None si absente) — ce que la cote MESURE sur le dessin."""
    for g in root.iter():
        if tag(g) == "g" and "dm" in classes(g) and any(tag(x) == "g" and x.get("data-dim") == ident for x in g.iter()):
            for p in tous(g, "path", "dln"):
                q = sous_chemins(p.get("d"))[0]
                return q[0], q[-1]
    return None


def translation(el):
    m = re.search(r"translate\(([-\d.e]+) ([-\d.e]+)\)", el.get("transform") or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def rect_box(el):
    x, y, w, h = (float(el.get(k)) for k in ("x", "y", "width", "height"))
    return x, y, x + w, y + h


# ------------------------------------------------------------------ le dégradé imposé, recopié des décisions (§ 9.1)
def _lerp(c0, c1, t):
    t = min(1.0, max(0.0, t))
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02X%02X%02X" % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


PALETTE = ((0.0, "#2563EB"), (0.25, "#06B6D4"), (0.5, "#22C55E"), (0.75, "#FACC15"), (1.0, "#F97316"))
ROUGE = "#DC2626"


def couleur_attendue(t):
    """0 → 100 % : bleu #2563EB, cyan #06B6D4 (25 %), vert #22C55E (50 %), jaune #FACC15 (75 %), orange #F97316 (100 %),
    par morceaux ; 100 → 105 % : #F97316 → #C2410C ; au-delà rouge #DC2626 — la règle des décisions (§ 9.1 révisé le 10/11
    au soir sur la remarque de l'ingénieur), écrite ici pour ne pas faire confiance au module."""
    if t <= 1.0:
        t = max(0.0, t)
        for (t0, c0), (t1, c1) in zip(PALETTE, PALETTE[1:]):
            if t <= t1:
                return _lerp(c0, c1, (t - t0) / (t1 - t0))
    if t <= 1.05:
        return _lerp("#F97316", "#C2410C", (t - 1.0) / 0.05)
    return ROUGE


def est_froid(c):
    """Une couleur du bas de l'échelle (bleu, cyan, vert : taux faible) — le bleu l'emporte sur le rouge."""
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return b > r


def est_depasse(c):
    """Une couleur de la bande 100 → 105 % (de l'orange #F97316 au rouge-orangé #C2410C), le rouge franc exclu."""
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return c != ROUGE and 0xC2 <= r <= 0xF9 and 0x41 <= g <= 0x73 and b <= 0x16


# ------------------------------------------------------------------ les contrôles réutilisés par les mutations
def fautes_coupe(mod, res):
    """Ce que la coupe de ``mod`` ne respecte pas face au résultat ``res`` : échelle, orientation, ancrage, charge,
    sections, pivot ou contact, cotes du mode, support, profondeur, rondelles, type d'ancrage."""
    g, cfg, mode = res["geometrie"], res["configuration"], res["mode"]
    e, r, larg, haut, X, dx, Y, dy = g["e"], g["r"], g["larg"], g["haut"], g["X"], g["dx"], g["Y"], g["dy"]
    d, prof, typ, nr, rond = g["d"], g["profondeur"], g["type_ancrage"], g["rondelles"], g["rondelle"]
    k = -1.0 if cfg == "A" else 1.0
    f = []
    root = racine(mod.svg(mod.coupe(res, fs=5.0), "t"))
    prof_grp = groupe_sel(root, "profil")
    poly = [p for grp in prof_grp for p in tous(grp, "polygon", "pp")]
    if not poly:
        return ["contour de la cornière absent (data-sel profil)"]
    pts = points_poly(poly[0])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if not (pres(max(xs) - min(xs), larg) and pres(max(ys) - min(ys), haut)):
        f.append(f"contour {max(xs) - min(xs):.2f} × {max(ys) - min(ys):.2f} au lieu de {larg} × {haut}")
    if not (pres(min(xs), 0.0) and pres(min(ys) if cfg == "B" else max(ys), 0.0)):
        f.append(f"origine du contour : x {min(xs):.2f}, face extérieure {min(ys) if cfg == 'B' else max(ys):.2f}")
    if (cfg == "A" and max(ys) > 1e-6) or (cfg == "B" and min(ys) < -1e-6):
        f.append("l'aile horizontale n'est pas du bon côté (A : en bas, B : en haut)")
    # l'ancrage : axe à Y de la face extérieure, positions Y ± dy
    ax = [p for grp in groupe_sel(root, "Y") for p in tous(grp, "path", "ax")]
    ya = {round(p[1], 6) for a in ax for p in points_d(a.get("d"))}
    if ya != {round(k * Y, 6)}:
        f.append(f"axe de l'ancrage à y = {sorted(ya)} au lieu de {k * Y}")
    if dy > 0:
        yd = {round(p[1], 6) for grp in groupe_sel(root, "dy") for h in tous(grp, "path", "hd") for p in points_d(h.get("d"))}
        if yd != {round(k * (Y - dy), 6), round(k * (Y + dy), 6)}:
            f.append(f"positions Y ± dy à {sorted(yd)} au lieu de {k * (Y - dy)}, {k * (Y + dy)}")
    # la charge : fût à X, positions X ± dx
    fx = {round(p[0], 6) for grp in groupe_sel(root, "X") for z in tous(grp, "path", "fz") for p in points_d(z.get("d"))}
    if fx != {round(X, 6)}:
        f.append(f"charge à x = {sorted(fx)} au lieu de {X}")
    if dx > 0:
        xd = {round(p[0], 6) for grp in groupe_sel(root, "dx") for z in tous(grp, "path", "fzd") for p in points_d(z.get("d"))}
        if xd != {round(X - dx, 6), round(X + dx, 6)}:
            f.append(f"positions X ± dx à {sorted(xd)} au lieu de {X - dx}, {X + dx}")
    # la résultante de compression : pivot (A) ou contact (B)
    anc = res["ancrages"]
    y_C = anc["combinaison_N"]["Y"] - anc["bras"] if cfg == "A" else anc["combinaison_N"]["Y"] + anc["bras"]
    piv = tous(root, "circle", "fcp")
    if not piv or not (pres(float(piv[0].get("cx")), 0.0) and pres(float(piv[0].get("cy")), k * y_C)):
        f.append(f"résultante de compression à {[(p.get('cx'), p.get('cy')) for p in piv]} au lieu de (0, {k * y_C})")
    if cfg == "B" and not pres(y_C, haut - g["yc"]):
        f.append("en B le contact n'est pas au bout libre")
    # les sections : S1 en u = e + r, S2 en u = e, S3 en y = e + r, S4 en y = e ; les pastilles 1 à 4
    sc = [points_d(p.get("d")) for p in tous(root, "path", "sc")]
    vert = sorted({round(s[0][0], 6) for s in sc if abs(s[0][0] - s[1][0]) < 1e-9})
    hori = sorted({round(s[0][1], 6) for s in sc if abs(s[0][1] - s[1][1]) < 1e-9})
    if vert != sorted({round(e, 6), round(e + r, 6)}):
        f.append(f"sections 1 et 2 en u = {vert} au lieu de {sorted({e, e + r})}")
    if hori != sorted({round(k * e, 6), round(k * (e + r), 6)}):
        f.append(f"sections 3 et 4 en y = {hori} au lieu de {sorted({k * e, k * (e + r)})}")
    for s in sc:
        if abs(s[0][0] - s[1][0]) < 1e-9:                      # S1 ou S2 : traverse l'aile horizontale (y de 0 à ±e, ±(e+r))
            if not (min(k * p[1] for p in s) <= 0.0 + 1e-9 and max(k * p[1] for p in s) >= e - 1e-9):
                f.append("une section 1/2 ne traverse pas l'aile horizontale")
        elif not (min(p[0] for p in s) <= 0.0 + 1e-9 and max(p[0] for p in s) >= e - 1e-9):
            f.append("une section 3/4 ne traverse pas l'aile verticale")
    nums = sorted(int("".join(t.itertext()).strip()) for grp in tous(root, "g", "scg") for t in tous(grp, "text"))
    if nums != [1, 2, 3, 4]:
        f.append(f"pastilles {nums}")
    # les cotes du mode
    attend = {"X": X, "Y": Y, "larg": larg, "haut": haut, "e": e}
    if mode == "historique":
        attend.update({"Xe": X - r - e, "Xer": X - e, "Ye": Y - e - r, "Yer": Y - e, "YH": Y - e / 2.0})
    elif cfg == "A":
        attend.update({"bras": Y - g["yc"], "bras_x": X - g["uV"]})
    else:
        attend.update({"bras": haut - g["yc"] - Y, "bras_x": X - g["uV"]})
    if g["support"] == "poutre":
        attend.update({"bord_haut": g["bords_poutre"]["haut"], "bord_bas": g["bords_poutre"]["bas"]})
        # ce que les deux cotes mesurent : depuis l'axe d'ancrage le plus proche du bord (Y ± dy), pas l'axe nominal
        for ident, y_anc in (("bord_haut", k * (Y - k * dy)), ("bord_bas", k * (Y + k * dy))):
            c = cote(root, ident)
            if c is None or not any(pres(p[1], y_anc) for p in c):
                f.append(f"la cote {ident} ne part pas de l'axe d'ancrage le plus proche du bord (y = {y_anc})")
    for ident, val in attend.items():
        t = dim(root, ident)
        if t is None or not re.search(r"(^|[ =])" + re.escape(f0(val)) + "$", t):
            f.append(f"cote {ident} : « {t} » au lieu de {val}")
    tx = " ".join(textes(root))
    if r > 0 and f"r = {f0(r)}" not in tx:
        f.append("le congé r n'est pas coté")
    # --- le support : voile sans limite ou poutre entre ses deux arêtes ; coupée quand elle est haute
    be = [points_poly(p) for p in tous(root, "polygon", "be")]
    if not be:
        f.append("support absent")
    else:
        y_min = min(p[1] for q in be for p in q)
        y_max = max(p[1] for q in be for p in q)
        x_min = min(p[0] for q in be for p in q)
        if g["support"] == "poutre":
            hs, bs = g["support_haut"], g["support_bas"]
            coupe_h, coupe_b = hs > 1.5 * haut, bs > 1.5 * haut
            if len(be) != 1 + int(coupe_h) + int(coupe_b):
                f.append(f"poutre en {len(be)} morceau(x) pour {1 + int(coupe_h) + int(coupe_b)} attendu(s)")
            if not coupe_h and not pres(y_min, -hs):
                f.append(f"arête haute de la poutre à y = {y_min:.2f} au lieu de {-hs}")
            if not coupe_b and not pres(y_max, bs):
                f.append(f"arête basse de la poutre à y = {y_max:.2f} au lieu de {bs}")
            if coupe_h and not (-hs < y_min < -haut * 1.5):
                f.append(f"poutre haute non coupée (y_min {y_min:.1f})")
            if "poutre" not in tx:
                f.append("le support ne dit pas « poutre »")
        else:
            if not (y_min < min(0.0, k * haut) - 1e-6 and y_max > max(0.0, k * haut) + 1e-6):
                f.append("le voile ne dépasse pas la cornière des deux côtés")
            if "voile" not in tx:
                f.append("le support ne dit pas « voile »")
        if typ == "traversant":
            if not pres(x_min, -prof):
                f.append(f"support traversé : face arrière à x = {x_min:.2f} au lieu de {-prof}")
            if sum(1 for q in be for p in q if pres(p[0], -prof)) != 2 * len(be):
                f.append("la face arrière du support traversé n'est pas droite")
        elif x_min > -prof + 1e-6 and prof <= 1.2 * haut:
            f.append(f"le support ({x_min:.1f}) n'englobe pas la cheville (−{prof})")
    # --- la tige jusqu'à −profondeur (coupée au-delà de 1,2 × haut), la cote de la profondeur
    tg = [points_poly(p) for grp in groupe_sel(root, "d") for p in tous(grp, "polygon", "tg")]
    if not tg:
        f.append("tige absente")
    else:
        x_tige = min(p[0] for q in tg for p in q)
        if prof <= 1.2 * haut:
            attendu = -prof if typ != "traversant" else -prof - rond["normale"][1] - 1.1 * d
            if len(tg) != 1 or not pres(x_tige, attendu):
                f.append(f"tige de {x_tige:.2f} ({len(tg)} morceau(x)) au lieu de {attendu}")
        elif len(tg) != 2 or x_tige <= -prof or x_tige > -0.45 * haut:
            f.append(f"cheville profonde : tige en {len(tg)} morceau(x), bout à {x_tige:.1f} (coupure attendue)")
    libelle = "hef" if typ != "traversant" else "ép."
    if f"{libelle} = {f0(prof)}" not in tx:
        f.append(f"la profondeur « {libelle} = {f0(prof)} » n'est pas cotée")
    # --- les rondelles et l'écrou sur l'aile : nr rondelles aux dimensions de la table, puis l'écrou de 0,8 d
    devant = sorted((rect_box(p) for p in tous(root, "rect", "bo") if float(p.get("x")) >= e - 1e-6), key=lambda b: b[0])
    attendus = ([rond["large"]] if nr == 2 else []) + ([rond["normale"]] if nr >= 1 else [])
    if len(devant) != len(attendus) + 1:
        f.append(f"{len(devant)} pièces sur l'aile pour {nr} rondelle(s) + l'écrou")
    else:
        x = e
        for (x1, y1, x2, y2), (D_, ep_) in zip(devant, attendus):
            if not (pres(x1, x) and pres(x2 - x1, ep_) and pres(y2 - y1, D_)):
                f.append(f"rondelle {x2 - x1:.1f} × {y2 - y1:.1f} en x = {x1:.1f} au lieu de {ep_} × {D_} en {x:.1f}")
            x += ep_
        x1, y1, x2, y2 = devant[-1]
        if not (pres(x1, x) and pres(x2 - x1, 0.8 * d) and pres(y2 - y1, 1.7 * d)):
            f.append(f"écrou {x2 - x1:.1f} × {y2 - y1:.1f} en x = {x1:.1f}")
    # --- le type d'ancrage : douille, scellement, ou pile arrière
    derriere = [rect_box(p) for p in tous(root, "rect", "bo") if float(p.get("x")) < 0.0]
    scel = tous(root, "polygon", "scel")
    # le cône de la douille : un polygone « tg » à quatre sommets de hauteurs toutes différentes (la douille est à plat, hors data-sel)
    cones = [points_poly(p) for p in tous(root, "polygon", "tg") if len(points_poly(p)) == 4
             and len({round(pt[1], 6) for pt in points_poly(p)}) == 4]
    if typ == "mecanique":
        if len(derriere) != 1 or not cones or scel:
            f.append(f"cheville mécanique : {len(derriere)} manchon(s), {len(cones)} cône(s), {len(scel)} scellement(s)")
    elif typ == "chimique":
        if not scel or derriere or cones:
            f.append(f"cheville chimique : {len(scel)} scellement(s), {len(derriere)} pièce(s) arrière, {len(cones)} cône(s)")
        elif not any(pres(min(p[0] for p in points_poly(q)), -prof) for q in scel) and prof <= 1.2 * haut:
            f.append("le scellement ne va pas jusqu'au fond du forage")
    else:
        if len(derriere) != 2 or scel or cones:
            f.append(f"boulon traversant : {len(derriere)} pièce(s) arrière, {len(scel)} scellement(s), {len(cones)} cône(s)")
        elif not any(pres(b[2], -prof) for b in derriere):
            f.append("la rondelle arrière n'est pas contre la face arrière du support")
    return f


def fautes_elevation(mod, res):
    g, anc, cfg = res["geometrie"], res["ancrages"], res["configuration"]
    e, r, larg, haut, L, n, Y = g["e"], g["r"], g["larg"], g["haut"], g["L"], g["n"], g["Y"]
    k = -1.0 if cfg == "A" else 1.0
    f = []
    root = racine(mod.svg(mod.elevation(res, fs=8.0), "t"))
    ps = [points_poly(p) for grp in groupe_sel(root, "profil") for p in tous(grp, "polygon", "ps")]
    coupee = len(ps) > 2
    bo = [c for grp in groupe_sel(root, "n") for c in tous(grp, "circle", "bo")]
    zs = sorted(round(float(c.get("cx")), 6) for c in bo)
    attendu = [round(z, 6) for z in anc["positions_z"]]
    if any(not pres(float(c.get("cy")), k * Y) for c in bo) or len(zs) != n:
        f.append(f"{len(zs)} ancrages, y = {[c.get('cy') for c in bo]} (attendu {n} à {k * Y})")
    if not coupee and zs != attendu:
        f.append(f"ancrages en z = {zs} (attendu {attendu})")
    if not any(pres(float(c.get("r")), g["d"] / 2) for c in bo):
        f.append("le rond de la tige n'a pas le diamètre d")
    for ident, val in (("L", L), ("haut", haut), ("larg", larg), ("Y", Y), ("X", g["X"])):
        t = dim(root, ident)
        if t is None or not t.endswith(f0(val)):
            f.append(f"cote {ident} : « {t} »")
    tx = " ".join(textes(root))
    # la chaîne e₁ · s · … · e₁
    for val in {anc["distance_bout"], anc["entraxe"]} if n > 1 else {anc["distance_bout"]}:
        if f0(val) not in textes(root):
            f.append(f"la chaîne des cotes n'écrit pas {f0(val)}")
    # la face de l'aile verticale : L × haut, du bon côté ; l'aile rabattue : L × (larg − e) de l'autre côté
    boites = [bbox(q) for q in ps]
    hauteurs = sorted(round(b[3] - b[1], 6) for b in boites)
    if coupee:
        if L <= 2.5 * haut or len(ps) % 2 or not all(b[2] - b[0] < L for b in boites):
            f.append(f"tranche coupée : {len(ps)} face(s), largeurs {[round(b[2] - b[0]) for b in boites]} pour L = {L} et haut = {haut}")
        if dim(root, "L") != f"L = {f0(L)}":
            f.append("la cote L ne garde pas sa vraie valeur")
    else:
        if len(ps) != 2 or sorted(round(b[2] - b[0], 6) for b in boites) != [round(L, 6)] * 2:
            f.append(f"faces {[(round(b[2] - b[0], 3), round(b[3] - b[1], 3)) for b in boites]}")
    if hauteurs[:1] + hauteurs[-1:] != sorted([round(larg - e, 6), round(haut, 6)]):
        f.append(f"hauteurs des faces {hauteurs}")
    face = [b for b in boites if pres(b[3] - b[1], haut)]
    if face and not pres(face[0][1], min(0.0, k * haut)):
        f.append("la face de l'aile verticale n'est pas du bon côté de l'aile horizontale")
    # N et V par ancrage, dans l'état limite saisi
    et = anc["etat_limite"]
    if f"N {et} = {kn(anc['N'])}" not in tx or f"V {et} = {kn(anc['V'])}" not in tx:
        f.append(f"N {et} / V {et} par ancrage absents ({kn(anc['N'])}, {kn(anc['V'])})")
    if g["a"] is not None:
        ae = [p for grp in groupe_sel(root, "a") for p in tous(grp, "polygon", "ae")]
        if not ae:
            f.append("pas de trapèze de diffusion alors que a est donnée")
        else:
            pts = points_poly(ae[0])
            ys = sorted({round(p[1], 6) for p in pts})
            base = [p[0] for p in pts if round(p[1], 6) == (max(ys) if cfg == "A" else min(ys))]
            if not pres(max(base) - min(base), g["L_eff"]):
                f.append(f"base du trapèze {max(base) - min(base):.2f} au lieu de L_eff = {g['L_eff']}")
        t = dim(root, "Leff")
        if t is None or not t.endswith(f0(g["L_eff"])):
            f.append(f"cote L_eff : « {t} »")
    else:
        if f"R = {kn(res['efforts']['R'])} sur L" not in tx:
            f.append("sans a, la charge ne dit pas « R = … sur L »")
    return f


def fautes_moments(mod, res):
    """Les moments sur la cornière : échelle commune, côté, courbes, enveloppe remplie, valeurs écrites, ancrage."""
    g, cfg = res["geometrie"], res["configuration"]
    e, r, larg, haut, Y, dy, X, dx = g["e"], g["r"], g["larg"], g["haut"], g["Y"], g["dy"], g["X"], g["dx"]
    k = -1.0 if cfg == "A" else 1.0
    dgs, verifs, eff = res["diagrammes"], {v["cle"]: v for v in res["verifications"]}, res["efforts"]
    f = []
    root = racine(mod.svg(mod.moments(res, fs=5.0), "t"))
    ps = tous(root, "polygon", "ps")
    if len(ps) != 1 or not (pres(bbox(points_poly(ps[0]))[2], larg) and pres(bbox(points_poly(ps[0]))[3] - bbox(points_poly(ps[0]))[1], haut)):
        return [f"la cornière n'est pas dessinée à l'échelle ({len(ps)} polygone(s))"]
    tx = textes(root)
    # les axes M = 0
    m0 = [points_d(p.get("d")) for p in tous(root, "path", "m0")]
    pts_m0 = [p for q in m0 for p in q]
    if tx.count("M = 0") != 2 or not any(pres(p[1], k * e / 2) for p in pts_m0) or not any(pres(p[0], e / 2) for p in pts_m0):
        f.append("l'axe M = 0 de chaque aile n'est pas tracé et nommé")
    # les courbes et l'enveloppe : autant que le moteur, autant de points
    n_courbes = sum(len(dgs[a]["courbes"]) for a in ("horizontale", "verticale"))
    cvf, cvm, env = tous(root, "path", "cvf"), tous(root, "path", "cvm"), tous(root, "path", "env")
    if len(cvf) + len(cvm) != n_courbes or len(cvm) != 2:
        f.append(f"{len(cvf)} courbes fines + {len(cvm)} moyennes pour {n_courbes} courbes du moteur")
    envs = [points_d(p.get("d")) for p in env]
    if len(envs) != 2:
        return f + [f"{len(envs)} enveloppes"]
    y_ax, x_ax = k * e / 2, e / 2
    echelles = []
    for aile in ("horizontale", "verticale"):
        pts = dgs[aile]["points"]
        if aile == "horizontale":
            cand = [q for q in envs if all(pres(a[0], p["s"]) for a, p in zip(q, pts)) and len(q) == len(pts)]
        else:
            cand = [q for q in envs if all(pres(a[1], k * p["s"]) for a, p in zip(q, pts)) and len(q) == len(pts)]
        if len(cand) != 1:
            f.append(f"{aile} : enveloppe introuvable aux abscisses du moteur ({len(pts)} points)")
            continue
        q = cand[0]
        for a, p in zip(q, pts):
            if abs(p["M"]) < 1e-9:
                if not pres(a[1] if aile == "horizontale" else a[0], y_ax if aile == "horizontale" else x_ax):
                    f.append(f"{aile} : un moment nul n'est pas sur l'axe")
                continue
            # porté hors de l'angle : l'aile horizontale vers la face extérieure (bas en A, haut en B), la verticale côté support
            ecart = (a[1] - y_ax) * (-k) if aile == "horizontale" else (x_ax - a[0])
            if ecart <= 0:
                f.append(f"{aile} : le diagramme est porté dans l'angle")
                break
            echelles.append(ecart / abs(p["M"]))
    if echelles and (max(echelles) - min(echelles)) > 1e-6 * max(echelles):
        f.append(f"l'échelle des moments n'est pas commune ({min(echelles):.4f} à {max(echelles):.4f} mm/kN·m)")
    if echelles:
        plus_grand = abs(eff["Mmax"]) * max(echelles)
        if not 0.6 * haut - 1e-6 <= plus_grand <= 0.8 * haut + 1e-6:
            f.append(f"le plus grand moment occupe {plus_grand / haut:.2f} × haut")
    # le remplissage : un polygone par aile, fermé sur l'axe, avec les points de l'enveloppe ET le bout du prolongement
    # (jusqu'à Mmax au nu du support pour l'aile horizontale ; au pivot pour la verticale en historique et en B)
    prolongee = {"horizontale": True, "verticale": res["mode"] == "historique" or cfg == "B"}
    envf = [points_poly(p) for p in tous(root, "polygon", "envf")]
    if len(envf) != 2 or sorted(len(q) for q in envf) != sorted(len(dgs[a]["points"]) + (3 if prolongee[a] else 2) for a in ("horizontale", "verticale")):
        f.append(f"remplissage de l'enveloppe : {len(envf)} polygone(s), {[len(q) for q in envf]} points")
    else:
        for q in envf:
            if not ((pres(q[0][1], y_ax) and pres(q[-1][1], y_ax)) or (pres(q[0][0], x_ax) and pres(q[-1][0], x_ax))):
                f.append("un remplissage ne se referme pas sur l'axe M = 0")
        if echelles:
            ech_ = max(echelles)
            # l'aile horizontale : un sommet du remplissage au nu du support (x = 0), à Mmax de l'axe
            if not any(pres(p[0], 0.0) and pres(abs(p[1] - y_ax), abs(eff["Mmax"]) * ech_, 1e-6) for q in envf for p in q):
                f.append("le remplissage de l'aile horizontale ne va pas jusqu'à Mmax au nu du support")
            if prolongee["verticale"]:
                Mv = abs(eff["Mmax"]) if res["mode"] == "historique" else abs(eff["M_angle"])
                if not any(pres(p[1], k * e / 2) and pres(x_ax - p[0], Mv * ech_, 1e-6) for q in envf for p in q):
                    f.append("le remplissage de l'aile verticale ne va pas jusqu'au bout de son prolongement (pivot)")
    # les trois zéros de l'aile verticale (A, dy > 0) : chaque courbe quitte l'axe à SON ancrage Y′ = Y − dy, Y, Y + dy
    if cfg == "A" and dy > 0:
        zeros = set()
        for p in tous(root, "path", "cvf") + tous(root, "path", "cvm"):
            pts_c = points_d(p.get("d"))
            if all(pres(a[1], k * q["s"]) for a, q in zip(pts_c, dgs["verticale"]["points"])) and len(pts_c) == len(dgs["verticale"]["points"]):
                sur_axe = [a[1] for a in pts_c if pres(a[0], x_ax)]
                if sur_axe:
                    zeros.add(round(min(abs(y_) for y_ in sur_axe), 6))
        if zeros != {round(Y - dy, 6), round(Y, 6), round(Y + dy, 6)}:
            f.append(f"les zéros des courbes de l'aile verticale sont à {sorted(zeros)} et non à Y ± dy")
    # les valeurs : M1 à M4 (grandeur de l'enveloppe), Mmax (en bas, et à gauche en historique), M angle en B
    joint = " ".join(tx)
    for i, cle in enumerate(("S1", "S2", "S3", "S4")):
        if not re.search(rf"M{i + 1} = (M angle = )?{re.escape(fr(abs(verifs[cle]['M']), 3))}", joint):
            f.append(f"la valeur M{i + 1} = {fr(abs(verifs[cle]['M']), 3)} n'est pas écrite")
    n_mmax = tx.count(f"Mmax = {fr(abs(eff['Mmax']), 3)}")
    if n_mmax != (2 if res["mode"] == "historique" else 1):
        f.append(f"Mmax écrit {n_mmax} fois (attendu : en bas, et à gauche en historique)")
    if cfg == "B" and f"M angle = {fr(abs(eff['M_angle']), 3)}" not in joint:
        f.append("en B, M angle n'est pas écrit")
    nums = sorted(int("".join(t.itertext()).strip()) for grp in tous(root, "g", "scg") for t in tous(grp, "text"))
    if nums != [1, 2, 3, 4]:
        f.append(f"pastilles {nums}")
    # l'ancrage : axe à Y, bande ± dy ; la charge à X
    ax = {round(p[1], 6) for a in tous(root, "path", "ax") for p in points_d(a.get("d"))}
    if ax != {round(k * Y, 6)}:
        f.append(f"axe de l'ancrage à {sorted(ax)} au lieu de {k * Y}")
    tol = tous(root, "rect", "tol")
    if dy > 0 and (len(tol) != 1 or sorted((float(tol[0].get("y")), float(tol[0].get("y")) + float(tol[0].get("height")))) != sorted((k * (Y - dy), k * (Y + dy)))):
        f.append("la bande ± dy manque ou n'est pas à Y ± dy")
    fz = {round(p[0], 6) for a in tous(root, "path", "fz") for p in points_d(a.get("d"))}
    if fz != {round(X, 6)} or (dx > 0 and len(tous(root, "path", "fzd")) != 2):
        f.append(f"la charge n'est pas à X ({sorted(fz)}) ou ses positions ± dx manquent")
    return f


def fautes_contraintes(mod, res):
    """Le maillage : nombre et position des mailles, couleur de chacune, maximum, bloc d'angle, barre, mention."""
    g, cfg = res["geometrie"], res["configuration"]
    e, r, larg, haut, Y = g["e"], g["r"], g["larg"], g["haut"], g["Y"]
    k = -1.0 if cfg == "A" else 1.0
    mz = res["maillage"]
    f = []
    root = racine(mod.svg(mod.contraintes(res, fs=5.0), "t"))
    mailles = tous(root, "polygon", "mz")
    Zc, TXc = mz["coin"]["zeta"], mz["coin"]["taux"]
    attendu_n = sum((len(mz[a]["s"]) - 1) * (len(mz[a]["zeta"]) - 1) for a in ("horizontale", "verticale")) + (len(Zc) - 1) ** 2
    if len(mailles) != attendu_n:
        return [f"{len(mailles)} mailles au lieu de {attendu_n} (deux ailes et le coin)"]
    # la DÉFORMÉE : les déplacements du moteur amplifiés pour que le plus grand fasse PART_DEFORMEE × haut (la règle de la
    # scène, réécrite ici) ; l'aile horizontale suit (0, k), l'aile verticale (1, 0), le bloc translate de u_bloc
    df = mz["deformee"]
    amp = mod.PART_DEFORMEE * haut / df["u_max"] if df["u_max"] > 1e-12 else 0.0
    dx_bloc = amp * df["u_bloc"]

    def interp(xs, ys, x):
        if x <= xs[0]:
            return ys[0] if abs(x - xs[0]) < 1e-9 else 0.0
        for i in range(1, len(xs)):
            if x <= xs[i]:
                a, b = xs[i - 1], xs[i]
                return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - a) / (b - a)
        return ys[-1]

    def ih(x):
        return amp * interp(df["horizontale"]["s"], df["horizontale"]["u"], x)

    def iv(x):
        return amp * interp(df["verticale"]["s"], df["verticale"]["u"], x)

    def deplace(x, y):
        ya = abs(y)
        if x >= e and (x - e) >= (ya - e):
            return (x + dx_bloc, y + k * ih(x))
        if ya >= e:
            return (x + iv(ya), y)
        return (x + dx_bloc, y)

    # les mailles attendues : leurs quatre nœuds sur la vraie géométrie DÉFORMÉE, leur couleur par le plus grand taux
    attendues = {}
    for aile in ("horizontale", "verticale"):
        ch = mz[aile]
        S_, T_, Z_, TX, U_ = ch["s"], ch["t"], ch["zeta"], ch["taux"], df[aile]["u"]

        def noeud(i, j):
            u = (Z_[j] + 1.0) / 2.0 * T_[i]
            if aile == "horizontale":
                return (round(S_[i] + dx_bloc, 4), round(k * u + k * amp * U_[i], 4))
            return (round(u + amp * U_[i], 4), round(k * S_[i], 4))

        for i in range(len(S_) - 1):
            for j in range(len(Z_) - 1):
                cle = frozenset([noeud(i, j), noeud(i + 1, j), noeud(i + 1, j + 1), noeud(i, j + 1)])
                attendues[cle] = couleur_attendue(max(TX[i][j], TX[i + 1][j], TX[i][j + 1], TX[i + 1][j + 1]))
    # le coin (bloc d'angle e × e), maillé lui aussi : i suit x depuis le nu du support, j suit y depuis la face extérieure ;
    # il translate avec le pied de l'aile verticale
    for i in range(len(Zc) - 1):
        for j in range(len(Zc) - 1):
            xs = [round((Zc[i] + 1.0) / 2.0 * e + dx_bloc, 4), round((Zc[i + 1] + 1.0) / 2.0 * e + dx_bloc, 4)]
            ys = [round(k * (Zc[j] + 1.0) / 2.0 * e, 4), round(k * (Zc[j + 1] + 1.0) / 2.0 * e, 4)]
            cle = frozenset([(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])])
            attendues[cle] = couleur_attendue(max(TXc[i][j], TXc[i + 1][j], TXc[i][j + 1], TXc[i + 1][j + 1]))
    fautes_couleur, introuvables = 0, 0
    for m in mailles:
        cle = frozenset((round(p[0], 4), round(p[1], 4)) for p in points_poly(m))
        if cle not in attendues:
            introuvables += 1
        elif (m.get("fill") or "").upper() != attendues[cle]:
            fautes_couleur += 1
    if introuvables:
        f.append(f"{introuvables} maille(s) hors de la géométrie du moteur")
    if fautes_couleur:
        f.append(f"{fautes_couleur} maille(s) d'une autre couleur que le plus grand taux de leurs nœuds")
    # la maille la plus chargée porte la couleur du taux max du maillage
    t_max = max(mz[a]["max"]["taux"] for a in ("horizontale", "verticale"))
    fills = {(m.get("fill") or "").upper() for m in mailles}
    if couleur_attendue(t_max) not in fills:
        f.append(f"la couleur du taux max ({couleur_attendue(t_max)}) n'est portée par aucune maille")
    # le maximum ◉ de chaque aile : à s de la vérification, sur la face ζ du maillage DÉFORMÉ ; son étiquette
    gv = [(float(c.get("cx")), float(c.get("cy"))) for c in tous(root, "circle", "gv")]
    tx = " ".join(textes(root))
    for aile in ("horizontale", "verticale"):
        mx, z = res["maxima"][aile], mz[aile]["max"]["zeta"]
        u = (z + 1.0) / 2.0 * mx["t"]
        att = (mx["s"] + dx_bloc, k * u + k * ih(mx["s"])) if aile == "horizontale" else (u + iv(mx["s"]), k * mx["s"])
        if not any(pres(a, att[0]) and pres(b, att[1]) for a, b in gv):
            f.append(f"{aile} : le ◉ n'est pas en {att} ({gv})")
        if f"{fr(100 * mx['taux'], 1)} % · s = {fr(mx['s'], 1)}" not in tx:
            f.append(f"{aile} : l'étiquette « {fr(100 * mx['taux'], 1)} % · s = {fr(mx['s'], 1)} » manque")
    # plus de bloc d'angle masqué (fond gris, hachures) : le coin est maillé ; le contour d'origine en tirets (hd), le
    # contour DÉFORMÉ par-dessus (ctr : chaque sommet déplacé comme la scène le fait, le plus grand déplacement =
    # PART_DEFORMEE × haut), les pastilles hors du dessin reliées par un pointillé léger (sl) qui s'arrête à la face
    # extérieure déformée (plus aucun trait de section « sc » à travers la pièce), l'axe de l'ancrage en repères courts
    # hors de la pièce, la barre et les mentions
    if tous(root, "polygon", "bg") or tous(root, "path", "ht"):
        f.append("le bloc d'angle est encore masqué (fond ou hachures) au lieu d'être maillé")
    hd_poly = [points_poly(q) for q in tous(root, "polygon", "hd")]
    if not any(bbox(q) == (0.0, min(0.0, k * haut), larg, max(0.0, k * haut)) for q in hd_poly):
        f.append("le contour d'origine (tirets) manque")
    ctr = tous(root, "polygon", "ctr")
    if not ctr:
        f.append("le contour déformé n'est pas tracé par-dessus")
    else:
        orig = next((q for q in hd_poly if bbox(q) == (0.0, min(0.0, k * haut), larg, max(0.0, k * haut))), None)
        pc = points_poly(ctr[0])
        if orig is None or len(pc) != len(orig):
            f.append("le contour déformé n'a pas les sommets du contour d'origine")
        else:
            attendu_c = [deplace(x, y) for x, y in orig]
            if any(abs(a[0] - b[0]) > 1e-3 or abs(a[1] - b[1]) > 1e-3 for a, b in zip(pc, attendu_c)):
                f.append("le contour déformé n'est pas celui de la déformée du moteur")
            # l'amplitude : chaque COMPOSANTE d'un déplacement dessiné est bornée par PART_DEFORMEE × haut (u_max est un maximum
            # de composantes : u_h, u_v, u_bloc), et le contour bouge visiblement (au moins le tiers de cette borne)
            comp = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(pc, orig)]
            if df["u_max"] > 1e-12 and (max(comp) > mod.PART_DEFORMEE * haut * (1 + 1e-6) or max(comp) < mod.PART_DEFORMEE * haut / 3.0):
                f.append(f"le plus grand déplacement dessiné fait {max(comp):.3f} (borne {mod.PART_DEFORMEE * haut}, au moins le tiers attendu)")
    if tous(root, "path", "sc"):
        f.append("des traits de section traversent encore le dessin")
    # les pastilles : un groupe « scg » translaté (transform), le cercle « scb » dedans ; son centre dans la scène = translation + (cx, cy)
    centres, rayons = {}, {}
    for grp in tous(root, "g", "scg"):
        num = int("".join("".join(t.itertext()) for t in tous(grp, "text")).strip())
        tr = re.search(r"translate\(\s*([-+0-9.eE]+)[ ,]+([-+0-9.eE]+)\s*\)", grp.get("transform") or "")
        tx_, ty_ = (float(tr.group(1)), float(tr.group(2))) if tr else (0.0, 0.0)
        c_ = tous(grp, "circle")[0]
        centres[num] = (tx_ + float(c_.get("cx")), ty_ + float(c_.get("cy")))
        rayons[num] = float(c_.get("r"))
    if sorted(centres) != [1, 2, 3, 4]:
        f.append(f"pastilles {sorted(centres)}")
    else:
        faces = {1: deplace(e + r, 0.0), 2: deplace(e, 0.0), 3: deplace(0.0, k * (e + r)), 4: deplace(0.0, k * e)}
        sl = [points_d(q.get("d")) for q in tous(root, "path", "sl")]
        for num, face in faces.items():
            # le pointillé part de la face extérieure DÉFORMÉE et finit dans la pastille (à moins d'un rayon de son centre)
            ok = any(len(seg) == 2 and ((pres(seg[0][0], face[0], 1e-3) and pres(seg[0][1], face[1], 1e-3) and math.hypot(seg[1][0] - centres[num][0], seg[1][1] - centres[num][1]) <= rayons[num])
                                        or (pres(seg[1][0], face[0], 1e-3) and pres(seg[1][1], face[1], 1e-3) and math.hypot(seg[0][0] - centres[num][0], seg[0][1] - centres[num][1]) <= rayons[num]))
                     for q in sl for seg in [q])
            if not ok:
                f.append(f"le renvoi pointillé de la section {num} ne va pas de la face extérieure déformée à la pastille")
            # la pastille est hors de la pièce, cercle compris : sous la face extérieure (1, 2) ou côté support (3, 4)
            if (num in (1, 2) and (centres[num][1] - face[1]) * (-k) < rayons[num]) or (num in (3, 4) and face[0] - centres[num][0] < rayons[num]):
                f.append(f"la pastille {num} n'est pas hors du dessin")
    ax = [points_d(a.get("d")) for a in tous(root, "path", "ax")]
    ax_y = {round(p[1], 6) for q in ax for p in q}
    if ax_y != {round(k * Y, 6)} or any(0.0 + 1e-9 < p[0] < e - 1e-9 for q in ax for p in q):
        f.append(f"axe de l'ancrage à {sorted(ax_y)} ou tracé à travers la pièce")
    # la barre de couleur : de 0 au MAXIMUM du maillage, graduée des deux côtés — la règle des graduations réécrite ici :
    # 0 et le maximum toujours ; 50, 100 et 105 % quand ils sont sous le maximum et à plus de ECART_GRADUATION de toute
    # graduation gardée
    barre = tous(root, "rect", "mz")
    s_lim = res["materiau"]["sigma_adm"]
    t_haut = max(mz["horizontale"]["max"]["taux"], mz["verticale"]["max"]["taux"], max(max(l) for l in TXc), 0.05)
    grads = [(0.0, "0"), (t_haut, f"{fr(100 * t_haut, 1)} %")]
    for t_, txt in ((0.5, "50"), (1.0, "100 %"), (1.05, "105 %")):
        if t_ < t_haut and all(abs(t_ - g) >= mod.ECART_GRADUATION * t_haut for g, _ in grads):
            grads.append((t_, txt))
    droite = [txt for _, txt in grads]
    gauche = [fr(t_ * s_lim, 1) for t_, _ in grads if t_ > 0]
    absents = [txt for txt in ("50", "100 %", "105 %") if txt not in droite]
    T = textes(root)
    if len(barre) < 20 or not all(t in T for t in droite) or any(t in T for t in absents):
        f.append(f"barre de couleur : {len(barre)} degrés, graduations attendues {droite} (absentes : {absents}), lues {[t for t in T if '%' in t or t in ('0', '50')]}")
    if not all(t in T for t in gauche) or "σéq N/mm²" not in T:
        f.append(f"barre de couleur : les N/mm² à gauche manquent ({gauche}, « σéq N/mm² »)")
    if "pas un calcul par éléments finis" not in tx or "coin : sections 2 et 4 prolongées" not in tx:
        f.append("la mention « coin : sections 2 et 4 prolongées · pas un calcul par éléments finis » manque")
    if f"déformée élastique · {df['combinaison']['libelle']} · u max = {fr(df['u_max'], 2)} mm · × {fr(amp, 0)}" not in tx:
        f.append("la mention de la déformée (combinaison, u max, amplification) manque")
    return f


def fautes_sel(mod, res):
    f = []
    attendu_coupe = {"X", "Y", "R", "d", "profil"} | ({"dx"} if res["geometrie"]["dx"] > 0 else set()) | ({"dy"} if res["geometrie"]["dy"] > 0 else set())
    attendu_coupe |= {"jeu_ancrage"} if res["mode"] == "historique" else {"yc", "uV"}
    s_coupe = set(sels(mod.svg(mod.coupe(res), "t")))
    if s_coupe != attendu_coupe:
        f.append(f"coupe : data-sel {sorted(s_coupe)} au lieu de {sorted(attendu_coupe)}")
    s_elev = set(sels(mod.svg(mod.elevation(res), "t")))
    if s_elev != {"L", "a", "n", "d", "X", "Y", "profil"}:
        f.append(f"élévation : data-sel {sorted(s_elev)}")
    if any(k not in mod.CLES_SEL for k in s_coupe | s_elev):
        f.append("une clé hors de CLES_SEL")
    if sels(mod.svg(mod.coupe(res, interactif=False), "t")) or sels(mod.svg(mod.elevation(res, interactif=False), "t")):
        f.append("data-sel posé alors que interactif=False")
    return f


# ================================================================== 1. les vues
print("=== 1. Les vues : format, SVG, texte lisible, légendes ===")
for nom, r in RES.items():
    vs = DC.vues(r, "t")
    fautes = []
    if [v["cle"] for v in vs] != ["coupe", "moments", "contraintes", "elevation"]:
        fautes.append([v["cle"] for v in vs])
    for v in vs:
        if set(v) != {"cle", "titre", "court", "svg", "legende", "echelle"}:
            fautes.append(f"{v['cle']} : clés {sorted(v)}")
        if v["echelle"] is not True:
            fautes.append(f"{v['cle']} : echelle {v['echelle']}")
        try:
            root = ET.fromstring(v["svg"])
        except ET.ParseError as ex:
            fautes.append(f"{v['cle']} : SVG mal formé ({ex})")
            continue
        if root.get("id") != f"t_{v['cle']}" or not root.get("viewBox") or root.get("role") != "img":
            fautes.append(f"{v['cle']} : id/viewBox/role")
        if re.search(r'="[^"]*\b(nan|inf|NaN|Infinity)\b', v["svg"]):
            fautes.append(f"{v['cle']} : valeur indéfinie")
        corps = re.sub(r"<style>.*?</style>", "", v["svg"], flags=re.S)
        if re.search(r">[^<]*\w_\w[^<]*<", corps):
            fautes.append(f"{v['cle']} : indice « _ » brut : {re.findall(r'>[^<]*\\w_\\w[^<]*<', corps)[:2]}")
        if "impossible" in v["svg"]:
            fautes.append(f"{v['cle']} : secours")
        px_min, px_max = tailles_px(v["svg"])
        if not 12.5 <= px_max <= 13.5 or px_min < 10.0:
            fautes.append(f"{v['cle']} : texte de {px_min:.1f} à {px_max:.1f} px")
        leg = v["legende"]
        if f"configuration {r['configuration']}" not in leg or r["materiau"]["etat_limite"] not in leg or (
                v["cle"] != "elevation" and ("historique" if r["mode"] == "historique" else "équilibre") not in leg):
            fautes.append(f"{v['cle']} : légende « {leg[:60]} »")
    chk(f"{nom} : quatre vues au format du site, SVG bien formés, sans valeur indéfinie ni « _ », texte à 13 px, légende avec "
        f"configuration, mode et état limite", not fautes, fautes[:3])

r0 = RES["A historique (cas de contrôle)"]
chk("les titres courts sont Coupe, Moments, Contraintes, Élévation ; les titres disent « sur la cornière » et « von Mises, maillage » ; "
    "les quatre vues sont à l'échelle (echelle True)",
    [v["court"] for v in DC.vues(r0)] == ["Coupe", "Moments", "Contraintes", "Élévation"]
    and [v["echelle"] for v in DC.vues(r0)] == [True, True, True, True]
    and "sur la cornière" in SC.VUES[1][1] and "von Mises, maillage" in SC.VUES[2][1])
chk("la légende de la coupe nomme les bras du classeur en historique, X − uV et Y − yc en équilibre, ya − c en B, l'état limite et R",
    "Xe, Xe+r, Ye, Ye+r, YH" in SC.legende(r0, "coupe") and "X − uV et Y − yc" in SC.legende(RES["A équilibre"], "coupe")
    and "ya − c" in SC.legende(RES["B Y = 85"], "coupe") and "ELS : R = 42,76 kN" in SC.legende(r0, "coupe")
    and "ELU : R = 60 kN" in SC.legende(RES["ELU R = 60"], "coupe") and "cheville mécanique Ø16, profondeur 100" in SC.legende(r0, "coupe")
    and "boulon traversant" in SC.legende(RES["traversant, 2 rondelles"], "coupe") and "poutre en béton" in SC.legende(RES["A sur poutre"], "coupe"))
chk("la légende des contraintes donne la limite (σadm 236,7 en ELS ; fy/γM0 355 en ELU) et dit « pas un calcul par éléments finis » ; "
    "celle des moments dit le côté et le signe (B : M < 0, M angle) ; celle de l'élévation l'entraxe, e₁, L eff et les coupures",
    "σadm = 236,7" in SC.legende(r0, "contraintes") and "fy/γM0 = 355,0" in SC.legende(RES["ELU R = 60"], "contraintes")
    and "pas un calcul par éléments finis" in SC.legende(r0, "contraintes")
    and "hors de l'angle" in SC.legende(r0, "moments") and "M angle" in SC.legende(RES["B Y = 85"], "moments")
    and "entraxe 100, e₁ = 50" in SC.legende(r0, "elevation") and "L eff = 174" in SC.legende(RES["A équilibre, a = 100"], "elevation")
    and "entraxe 150, e₁ = 75" in SC.legende(RES["entraxe saisi"], "elevation")
    and "coupée" in SC.legende(RES["L = 900 (coupures)"], "elevation") and "coupée" not in SC.legende(r0, "elevation"))
d_pc = SC.ajuster(lambda fs: SC.coupe(r0, fs), TOILE, 13.0, 7.0)
d_petit = SC.ajuster(lambda fs: SC.coupe(r0, fs), (300.0, 300.0), 9.0, 7.0)
chk("ajuster : une vue se reconstruit jusqu'à 13 px dans 626 × 682 ; dans une toile trop petite pour la cible, la taille s'arrête là où "
    "le dessin ne gagne plus rien (texte sous la cible, jamais au-dessus)",
    pres(SC.taille_texte(d_pc, TOILE), 13.0, 0.03) and d_petit.fs != d_pc.fs
    and SC.taille_texte(d_petit, (300.0, 300.0)) <= 9.0 + 0.3 and d_petit.fs < 15.0)
chk("ajuster garde la meilleure passe : la coupe de B (Y = 85) se lit à 13 px à 3 % près (elle restait à 14 px)",
    pres(SC.taille_texte(SC.ajuster(lambda fs: SC.coupe(RES["B Y = 85"], fs), TOILE, 13.0, 7.0), TOILE), 13.0, 0.03))

# ================================================================== 2 à 4. la coupe
print("\n=== 2–4. La coupe : échelle, orientation, ancrage, charge, pivot, sections, bras du mode, support, profondeur, rondelles ===")
for nom, r in RES.items():
    f = fautes_coupe(SC, r)
    chk(f"{nom} : coupe à l'échelle, ancrage à Y ± dy, charge à X ± dx, résultante au pivot ou au contact, sections 1–4, cotes du mode, "
        f"support, tige à −profondeur et cote, rondelles, type d'ancrage", not f, f[:3])
rb = RES["B Y = 85"]
root_b = racine(SC.svg(SC.coupe(rb, fs=5.0), "t"))
root_a = racine(SC.svg(SC.coupe(r0, fs=5.0), "t"))
chk("B : l'ancrage est SOUS l'aile horizontale (y = +Y), le contact au bout libre (y = +haut), les sections 3 et 4 à y = +e+r, +e ; "
    "en A tout est au-dessus (y négatifs)",
    {round(p[1], 6) for grp in groupe_sel(root_b, "Y") for a in tous(grp, "path", "ax") for p in points_d(a.get("d"))} == {85.0}
    and pres(float(tous(root_b, "circle", "fcp")[0].get("cy")), 120.0)
    and {round(p[1], 6) for grp in groupe_sel(root_a, "Y") for a in tous(grp, "path", "ax") for p in points_d(a.get("d"))} == {-85.0}
    and pres(float(tous(root_a, "circle", "fcp")[0].get("cy")), -10.0))
chk("A historique : la résultante C est au pivot des ancrages (jeu_ancrage = 10 mm du dessous) et l'étiquette dit « C = H »",
    pres(float(tous(root_a, "circle", "fcp")[0].get("cy")), -10.0) and any(t.startswith("C = H = ") for t in textes(root_a)))
root_e = racine(SC.svg(SC.coupe(RES["A équilibre"], fs=5.0), "t"))
chk("A équilibre : la résultante C est au pivot y_c (l'arête, 0), la ligne u_V est tracée à x = uV = e/2 et nommée",
    pres(float(tous(root_e, "circle", "fcp")[0].get("cy")), 0.0)
    and {round(p[0], 6) for grp in groupe_sel(root_e, "uV") for a in tous(grp, "path", "ael") for p in points_d(a.get("d"))} == {7.5}
    and "uV" in textes(root_e))
tx_a = " ".join(textes(root_a))
chk("les valeurs R, H, C sont écrites avec leurs unités (cas de contrôle : R = 42,76 kN, H = 42,76 kN), la profondeur « hef = 100 »",
    "R = 42,76 kN" in tx_a and "H = 42,76 kN" in tx_a and "C = H = 42,76 kN" in tx_a and "Ø16 · n = 3" in tx_a and "hef = 100" in tx_a)
chk("les cotes du classeur portent les valeurs du cas de contrôle : Xe = 47, Xe+r = 60, Ye = 57, Ye+r = 70, YH = 77,5",
    [dim(root_a, k) for k in ("Xe", "Xer", "Ye", "Yer", "YH")] == ["Xe = 47", "Xe+r = 60", "Ye = 57", "Ye+r = 70", "YH = 77,5"])
chk("B : la cote du bras est « ya − c » (35 pour Y = 85) et il n'y a pas de bras du classeur",
    dim(root_b, "bras") in ("ya − c 35", "ya − c = 35") and dim(root_b, "Ye") is None and dim(root_b, "YH") is None)
root_p = racine(SC.svg(SC.coupe(RES["L 40×40×4"], fs=2.3), "t"))
chk("L 40×40×4 (congé étroit) : les quatre pastilles sont là et ne se chevauchent pas",
    len(tous(root_p, "g", "scg")) == 4 and all(
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 >= 2 * 0.78 * 2.3 - 1e-6
        for i, a in enumerate([translation(g) for g in tous(root_p, "g", "scg")])
        for b in [translation(g) for g in tous(root_p, "g", "scg")][i + 1:]))
chk("le voile est hachuré, nommé, rompu en haut et en bas (il continue) ; la poutre a deux arêtes droites",
    tous(root_a, "polygon", "be") and "voile" in textes(root_a) and len(points_poly(tous(root_a, "polygon", "be")[0])) > 8
    and "poutre" in textes(racine(SC.svg(SC.coupe(RES["A sur poutre"], fs=5.0), "t"))))
root_pt = racine(SC.svg(SC.coupe(RES["A sur poutre"], fs=5.0), "t"))
chk("A sur poutre (160 au-dessus, 120 au-dessous) : les arêtes à y = −160 et +120, « bord haut = 70 » (160 − 90) et « bord bas = 200 » (120 + 80), "
    "la valeur du moteur (bords_poutre)",
    dim(root_pt, "bord_haut") == "bord haut = 70" and dim(root_pt, "bord_bas") == "bord bas = 200"
    and RES["A sur poutre"]["geometrie"]["bords_poutre"] == {"haut": 70.0, "bas": 200.0}
    and pres(min(p[1] for p in points_poly(tous(root_pt, "polygon", "be")[0])), -160.0)
    and pres(max(p[1] for p in points_poly(tous(root_pt, "polygon", "be")[0])), 120.0))
root_ph = racine(SC.svg(SC.coupe(RES["poutre haute (coupures)"], fs=5.0), "t"))
be_ph = [bbox(points_poly(p)) for p in tous(root_ph, "polygon", "be")]
chk("poutre haute (420 / 300) : trois morceaux (deux coupures), la hauteur dessinée plus petite que 720, les cotes « bord haut = 330 » et "
    "« bord bas = 380 » vraies",
    len(be_ph) == 3 and max(b[3] for b in be_ph) - min(b[1] for b in be_ph) < 720.0 - 100.0
    and dim(root_ph, "bord_haut") == "bord haut = 330" and dim(root_ph, "bord_bas") == "bord bas = 380")
root_tr = racine(SC.svg(SC.coupe(RES["traversant, 2 rondelles"], fs=5.0), "t"))
rond_tr = RES["traversant, 2 rondelles"]["geometrie"]["rondelle"]
chk("traversant à deux rondelles : le voile fait exactement 140 d'épaisseur (face arrière droite, sans rupture), la rondelle large Ø50 × 3 "
    "contre l'aile puis la Ø30 × 3 et l'écrou, écrou et rondelle derrière ; la cote dit « ép. = 140 »",
    rond_tr["large"] == [50.0, 3.0] and rond_tr["normale"] == [30.0, 3.0] and "ép. = 140" in textes(root_tr)
    and pres(min(p[0] for q in tous(root_tr, "polygon", "be") for p in points_poly(q)), -140.0)
    and len([p for p in tous(root_tr, "rect", "bo") if float(p.get("x")) < 0]) == 2 and not fautes_coupe(SC, RES["traversant, 2 rondelles"]))
root_ch = racine(SC.svg(SC.coupe(RES["chimique sans rondelle"], fs=5.0), "t"))
chk("chimique sans rondelle : le scellement hachuré du nu du voile au fond (x = −100), aucune rondelle, l'écrou contre l'aile",
    tous(root_ch, "polygon", "scel") and pres(min(p[0] for q in tous(root_ch, "polygon", "scel") for p in points_poly(q)), -100.0)
    and [float(p.get("x")) for p in tous(root_ch, "rect", "bo") if float(p.get("x")) >= 15 - 1e-6] == [15.0])
chk("L 40×40×4 à 60 de profondeur (> 1,2 × haut) : la tige est coupée en deux morceaux par des lignes de rupture, « hef = 60 » coté",
    len([p for grp in groupe_sel(root_p, "d") for p in tous(grp, "polygon", "tg")]) == 2 and "hef = 60" in textes(root_p))

# ================================================================== 5. data-sel
print("\n=== 5. Le contrat de sélection (data-sel) ===")
for nom, r in RES.items():
    f = fautes_sel(SC, r)
    chk(f"{nom} : les pièces portent la clé de leur donnée, rien de plus, rien sans interactif", not f, f)
chk("CLES_SEL énumère les clés admises, toutes des champs de la cornière",
    set(SC.CLES_SEL) <= set(M.nouvelle_corniere()) and {"X", "dx", "Y", "dy", "R", "L", "a", "n", "d", "profil", "yc", "uV"} <= set(SC.CLES_SEL))
svg_a = SC.svg(SC.coupe(r0), "t")
chk("les pièces sélectionnables sont des groupes <g class=\"pc\" data-sel=…> (la classe que l'établi compact écoute)",
    re.search(r'<g class="pc" data-sel="X">', svg_a) is not None and 'class="pc" data-sel="profil"' in svg_a)

# ================================================================== 6. les moments sur la cornière
print("\n=== 6. Les moments sur la cornière ===")
for nom, r in RES.items():
    f = fautes_moments(SC, r)
    chk(f"{nom} : cornière à l'échelle, axes M = 0, diagrammes perpendiculaires hors de l'angle à une échelle commune, courbes et enveloppe "
        f"remplie, M1–M4, Mmax, ancrage, charge", not f, f[:3])
root_m = racine(SC.svg(SC.moments(RES["A historique avec jeux"], fs=5.0), "t"))
chk("avec dx = 10 : deux courbes fines (X ± 10) et une moyenne (X) par aile, l'enveloppe en gras en plus, deux remplissages",
    len(tous(root_m, "path", "cvf")) == 4 and len(tous(root_m, "path", "cvm")) == 2 and len(tous(root_m, "path", "env")) == 2
    and len(tous(root_m, "polygon", "envf")) == 2)
root_m0 = racine(SC.svg(SC.moments(r0, fs=5.0), "t"))
chk("sans jeu : une seule courbe (nominale) par aile sous l'enveloppe, pas de bande de tolérance",
    len(tous(root_m0, "path", "cvf")) == 0 and len(tous(root_m0, "path", "cvm")) == 2 and not tous(root_m0, "rect", "tol"))
tx_m0 = " ".join(textes(root_m0))
chk("cas de contrôle : M1 = 2,010, M2 = 2,566, M3 = 2,359, M4 = 2,897 et Mmax = 3,207 écrits (les nombres du classeur)",
    all(t in tx_m0 for t in ("M1 = 2,010", "M2 = 2,566", "M3 = 2,359", "M4 = 2,897", "Mmax = 3,207")))


def envs_par_aile(root, res):
    """Les points de l'enveloppe dessinée de chaque aile (reconnue à ses abscisses)."""
    k = -1.0 if res["configuration"] == "A" else 1.0
    out = {}
    for q in (points_d(p.get("d")) for p in tous(root, "path", "env")):
        for aile in ("horizontale", "verticale"):
            pts = res["diagrammes"][aile]["points"]
            if len(q) == len(pts) and all(pres(a[0] if aile == "horizontale" else a[1], p["s"] if aile == "horizontale" else k * p["s"]) for a, p in zip(q, pts)):
                out[aile] = q
    return out


e_a, e_b = envs_par_aile(root_m0, r0), envs_par_aile(racine(SC.svg(SC.moments(rb, fs=5.0), "t")), rb)
chk("A : le diagramme de l'aile horizontale est porté vers le BAS (y > −e/2, sous l'aile), celui de l'aile verticale côté support (x < e/2) ; "
    "B : l'aile horizontale vers le HAUT (y < +e/2) — le signe de B porté de l'autre côté —, la verticale toujours côté support",
    all(p[1] >= -7.5 - 1e-9 for p in e_a["horizontale"]) and max(p[1] for p in e_a["horizontale"]) > 50
    and all(p[0] <= 7.5 + 1e-9 for p in e_a["verticale"]) and min(p[0] for p in e_a["verticale"]) < -50
    and all(p[1] <= 7.5 + 1e-9 for p in e_b["horizontale"]) and min(p[1] for p in e_b["horizontale"]) < -50
    and all(p[0] <= 7.5 + 1e-9 for p in e_b["verticale"]))
chk("l'échelle est commune : Mmax = 3,207 kN·m occupe 0,7 × haut = 84 mm (le point de l'enveloppe horizontale en s = 0 serait à y = 76,5) ; "
    "M4 = 2,897 → 75,9 mm côté support",
    pres(max(p[1] for p in e_a["horizontale"]) + 7.5, 2.56560 / 3.207 * 84.0, 1e-3)
    and pres(7.5 - min(p[0] for p in e_a["verticale"]), 2.89665 / 3.207 * 84.0, 1e-3))
tx_mb = " ".join(textes(racine(SC.svg(SC.moments(rb, fs=5.0), "t"))))
chk("B : M3 = M4 = 3,314 (moment constant de l'ancrage à l'angle), écrit « M4 = M angle = 3,314 » et « M3 = 3,314 », Mmax = 3,635",
    "M4 = M angle = 3,314" in tx_mb and "M3 = 3,314" in tx_mb and "Mmax = 3,635" in tx_mb
    and pres(abs(rb["efforts"]["M_angle"]), 42.76 * (85.0 - 7.5) / 1000.0, 1e-9))
chk("la vue des moments ne porte aucun paragraphe : seuls les repères (M = 0, R, Y, M1 à M4, Mmax) sont écrits",
    all(len(t) <= 24 for t in textes(root_m0)) and len(textes(root_m0)) <= 16)

# ================================================================== 7. les contraintes : le maillage
print("\n=== 7. Les contraintes : le maillage de von Mises ===")
for nom, r in RES.items():
    f = fautes_contraintes(SC, r)
    chk(f"{nom} : 96 × 12 mailles par aile sur la vraie géométrie DÉFORMÉE (déplacements du moteur × amplification), 12 × 12 dans le coin "
        f"translaté, chacune de la couleur du plus grand taux de ses nœuds, le ◉ de chaque aile déplacé, contour d'origine en tirets, pastilles "
        f"hors de la pièce reliées par un pointillé, barre arrêtée au maximum, trois mentions", not f, f[:3])
root_c = racine(SC.svg(SC.contraintes(r0, fs=5.0), "t"))
fills_c = {(m.get("fill") or "").upper() for m in tous(root_c, "polygon", "mz")}
chk("cas de contrôle (88,9 % en S3) : aucune maille de la bande 100–105 % ni rouge, des mailles froides (bleu, cyan, vert) aux bouts des ailes, "
    "la plus chargée porte la couleur du taux max du maillage, le ◉ de l'aile verticale dit « 90,2 % · s = 26,3 » (le maximum du congé, 1,66 mm "
    "avant le pied) et celui de l'horizontale « 77,4 % · s = 26,0 »",
    not any(est_depasse(c) for c in fills_c) and ROUGE not in fills_c and any(est_froid(c) for c in fills_c)
    and couleur_attendue(r0["maillage"]["verticale"]["max"]["taux"]) in fills_c
    and "90,2 % · s = 26,3" in " ".join(textes(root_c)) and "77,4 % · s = 26,0" in " ".join(textes(root_c)))
r102 = RES["A à 102 %"]
fills_102 = {(m.get("fill") or "").upper() for m in tous(racine(SC.svg(SC.contraintes(r102, fs=5.0), "t")), "polygon", "mz")}
chk("A à 102 % (R = 49,1 kN) : le maillage culmine entre 100 et 105 % — des mailles de la bande orange → rouge-orangé, aucune rouge franche, "
    "des froides",
    1.0 < r102["maillage"]["verticale"]["max"]["taux"] < 1.05 and any(est_depasse(c) for c in fills_102)
    and ROUGE not in fills_102 and any(est_froid(c) for c in fills_102))
r144 = RES["B chargée (144 %)"]
fills_144 = {(m.get("fill") or "").upper() for m in tous(racine(SC.svg(SC.contraintes(r144, fs=5.0), "t")), "polygon", "mz")}
chk("B chargée (144,5 % au pied du congé) : des mailles rouges #DC2626, des mailles de la bande 100–105 % et des froides",
    r144["maillage"]["verticale"]["max"]["taux"] > 1.05 and ROUGE in fills_144 and any(est_depasse(c) for c in fills_144)
    and any(est_froid(c) for c in fills_144))
chk("l'échelle imposée aux bornes : 0 % → bleu #2563EB, 25 % → cyan #06B6D4, 50 % → vert #22C55E, 75 % → jaune #FACC15, 100 % → orange #F97316, "
    "105 % → #C2410C, au-delà rouge #DC2626 ; par morceaux entre les bornes (la règle réécrite ici)",
    SC.couleur_taux(0.0) == "#2563EB" and SC.couleur_taux(0.25) == "#06B6D4" and SC.couleur_taux(0.5) == "#22C55E" and SC.couleur_taux(0.75) == "#FACC15"
    and SC.couleur_taux(1.0) == "#F97316" and SC.couleur_taux(1.05) == "#C2410C" and SC.couleur_taux(1.2) == "#DC2626" and SC.couleur_taux(2.0) == "#DC2626"
    and SC.couleur_taux(-0.1) == "#2563EB"
    and all(SC.couleur_taux(t) == couleur_attendue(t) for t in (0.0, 0.1, 0.3, 0.5, 0.62, 0.77, 0.99, 1.0, 1.02, 1.049, 1.05, 1.3)))
chk("les mailles portent leur couleur en attribut fill (lisible sans feuille de style, par le peintre PDF et en thème sombre)",
    all(re.fullmatch(r"#[0-9A-F]{6}", m.get("fill") or "") for m in tous(root_c, "polygon", "mz")))
root_elu = racine(SC.svg(SC.contraintes(RES["ELU R = 60"], fs=5.0), "t"))
rects_c = [(b.get("fill") or "").upper() for b in tous(root_c, "rect", "mz")]
chk("la barre de couleur s'ARRÊTE au maximum du maillage (§ 9.9) : sur le cas de contrôle, 24 degrés de 0 à 90,2 %, du bleu au jaune, ni bande "
    "100–105 % ni rouge puisque le maillage n'y va pas ; graduée en N/mm² à GAUCHE (0, 118,3 à 50 %, 213,4 au maximum — la limite 236,7 n'est "
    "pas atteinte) et en % à droite (0, 50, 90,2 %, sans 100 ni 105 %) ; son titre nomme la limite (σadm ; fy/γM0 à l'ELU : 177,5 et 345,7 à "
    "97,4 %, 355,0 hors de l'échelle)",
    len(rects_c) == 24 and any(est_froid(c) for c in rects_c) and not any(est_depasse(c) for c in rects_c) and ROUGE not in rects_c
    and "taux σéq / σadm" in textes(root_c) and "σéq N/mm²" in textes(root_c)
    and all(t in textes(root_c) for t in ("0", "118,3", "213,4", "50", "90,2 %")) and not any(t in textes(root_c) for t in ("100 %", "105 %", "236,7"))
    and "taux σéq / fy/γM0" in textes(root_elu) and all(t in textes(root_elu) for t in ("177,5", "345,7", "97,4 %")) and "355,0" not in textes(root_elu))
root_144 = racine(SC.svg(SC.contraintes(r144, fs=5.0), "t"))
rects_144 = [(b.get("fill") or "").upper() for b in tous(root_144, "rect", "mz")]
chk("B chargée : la barre monte à 144,5 % (342,1 N/mm²), rouge en haut, froide en bas ; la limite 236,7 écrite face au repère 100 % ; 105 % tu, "
    "trop près de 100 % (ECART_GRADUATION × max = 10 points)",
    ROUGE in rects_144 and any(est_froid(c) for c in rects_144)
    and all(t in textes(root_144) for t in ("342,1", "144,5 %", "236,7", "100 %", "118,3", "50")) and "105 %" not in textes(root_144))
rects_102 = [(b.get("fill") or "").upper() for b in tous(racine(SC.svg(SC.contraintes(r102, fs=5.0), "t")), "rect", "mz")]
chk("A à 102 % : la barre monte à 103,5 % et finit dans la bande orange → rouge-orangé ; 100 % tu (à 3,5 points du maximum)",
    any(est_depasse(c) for c in rects_102) and ROUGE not in rects_102
    and "103,5 %" in textes(racine(SC.svg(SC.contraintes(r102, fs=5.0), "t"))) and "100 %" not in textes(racine(SC.svg(SC.contraintes(r102, fs=5.0), "t"))))
amp0 = SC.PART_DEFORMEE * r0["geometrie"]["haut"] / r0["maillage"]["deformee"]["u_max"]
dxb0 = amp0 * r0["maillage"]["deformee"]["u_bloc"]
chk("le coin est maillé : 144 mailles (12 × 12) de la couleur du champ « coin » du moteur dans le carré e × e TRANSLATÉ avec le bloc d'angle "
    "(dx = amplification × u_bloc), plus de hachures ni de fond gris, la mention le dit",
    not tous(root_c, "path", "ht") and not tous(root_c, "polygon", "bg") and abs(dxb0) > 0.1
    and sum(1 for m in tous(root_c, "polygon", "mz") if all(dxb0 - 1e-3 <= p[0] <= dxb0 + 15.0 + 1e-3 and -15.0 - 1e-3 <= p[1] <= 1e-3 for p in points_poly(m))
            and bbox(points_poly(m))[2] - bbox(points_poly(m))[0] < 2.0) == 144
    and sum(1 for m in tous(root_c, "polygon", "mz") if all(-1e-3 <= p[0] <= 15.0 + 1e-3 and -15.0 - 1e-3 <= p[1] <= 1e-3 for p in points_poly(m))
            and bbox(points_poly(m))[2] - bbox(points_poly(m))[0] < 2.0) < 144
    and "coin : sections 2 et 4 prolongées · pas un calcul par éléments finis" in textes(root_c))

# ================================================================== 8. l'élévation
print("\n=== 8. L'élévation : ancrages, chaîne, L, L_eff, N et V, coupures ===")
for nom, r in RES.items():
    f = fautes_elevation(SC, r)
    chk(f"{nom} : n ancrages aux positions z_i du moteur à la hauteur Y, chaîne e₁ · s, cotes L, haut, larg, X, Y, N et V par ancrage, charge "
        f"et diffusion", not f, f[:3])
root_el = racine(SC.svg(SC.elevation(RES["A équilibre, a = 100"]), "t"))
chk("a = 100 : la cote a, le trapèze à 45° (base L_eff = 174 à la section 1), la cote L eff = 174, la position X′ qui gouverne en tirets",
    "a = 100" in textes(root_el) and dim(root_el, "Leff") == "Leff = 174" and tous(root_el, "path", "fzd") and tous(root_el, "polygon", "ae"))
tx_el = " ".join(textes(racine(SC.svg(SC.elevation(r0), "t"))))
chk("cas de contrôle : la chaîne 50 · 100 · 100 · 50, « 3 Ø16 », N ELS et V ELS par ancrage 14,25 kN, « R = 42,76 kN sur L »",
    all(t in tx_el for t in ("50", "100", "3 Ø16", "N ELS = 14,25 kN", "V ELS = 14,25 kN", "R = 42,76 kN sur L")))
tx_elu = " ".join(textes(racine(SC.svg(SC.elevation(RES["ELU R = 60"]), "t"))))
r_elu = RES["ELU R = 60"]
chk("ELU : les étiquettes disent « N ELU » et « V ELU » avec les efforts par ancrage de l'état limite saisi (H/n, R/n)",
    f"N ELU = {kn(r_elu['ancrages']['N'])}" in tx_elu and f"V ELU = {kn(r_elu['ancrages']['V'])}" in tx_elu and "ELS" not in tx_elu
    and pres(r_elu["ancrages"]["V"], 20.0))
root_en = racine(SC.svg(SC.elevation(RES["entraxe saisi"]), "t"))
chk("entraxe saisi (n = 2, s = 150 sur L = 300) : ancrages à z = 75 et 225, chaîne 75 · 150 · 75",
    sorted(round(float(c.get("cx")), 6) for grp in groupe_sel(root_en, "n") for c in tous(grp, "circle", "bo")) == [75.0, 225.0]
    and "75" in textes(root_en) and "150" in textes(root_en))
root_g = racine(SC.svg(SC.elevation(RES["L 200×200×20"]), "t"))
chk("L 200 sur 400 mm : entraxes 133,3 et bords 66,7 écrits", "133,3" in textes(root_g) and "66,7" in textes(root_g))
r900 = RES["L = 900 (coupures)"]
root_900 = racine(SC.svg(SC.elevation(r900, fs=12.0), "t"))
bo_900 = sorted(float(c.get("cx")) for grp in groupe_sel(root_900, "n") for c in tous(grp, "circle", "bo"))
ps_900 = [bbox(points_poly(p)) for grp in groupe_sel(root_900, "profil") for p in tous(grp, "polygon", "ps")]
chk("L = 900 (3 ancrages à 300) : la tranche est COUPÉE entre les ancrages — six faces (trois morceaux par vue), les ancrages dessinés à "
    "moins de 300 d'écart, la longueur dessinée sous 700, la cote « L = 900 » et la chaîne 150 · 300 vraies",
    len(ps_900) == 6 and all(bo_900[i + 1] - bo_900[i] < 300.0 - 1e-6 for i in range(2)) and bo_900[0] == 150.0
    and max(b[2] for b in ps_900) < 700.0 and dim(root_900, "L") == "L = 900" and "300" in textes(root_900) and "150" in textes(root_900))
chk("la tranche n'est pas coupée tant que L ≤ 2,5 × haut : L = 300 et L 40 sur 120 (le gain serait dérisoire) gardent une face entière",
    len([p for grp in groupe_sel(racine(SC.svg(SC.elevation(r0), "t")), "profil") for p in tous(grp, "polygon", "ps")]) == 2
    and len([p for grp in groupe_sel(racine(SC.svg(SC.elevation(RES["L 40×40×4"]), "t")), "profil") for p in tous(grp, "polygon", "ps")]) == 2)
chk("l'élévation ne porte presque plus de texte : un titre court (« aile horizontale rabattue »), pas d'explication",
    "aile horizontale rabattue" in textes(root_el) and all(len(t) <= 60 for t in textes(root_el)))

# ================================================================== 9. la vue impossible
print("\n=== 9. Le dessin qu'on ne sait pas faire ===")
r_long = M.verifier(dict(M.nouvelle_corniere(), profil="L 40x40x4", R=2.0, X=25.0, dx=5.0, Y=25.0, dy=3.0, d=8.0, L=10000.0, n=20))
vs = DC.vues(r_long, "t")
chk("une tranche de 10 m face à une L 40 : l'élévation seule est remplacée par « Dessin impossible à cette échelle », les trois autres "
    "vues sont dessinées", [("impossible" in v["svg"]) for v in vs] == [False, False, False, True] and vs[3]["svg"].startswith("<svg")
    and ET.fromstring(vs[3]["svg"]) is not None and "le calcul reste valable" in vs[3]["svg"])
try:
    SC.ajuster(lambda fs: SC.elevation(r_long, fs), TOILE, 13.0, 5.0)
    leve = False
except ValueError as ex:
    leve = "impossible" in str(ex)
chk("ajustée à la toile, schemas.elevation lève ValueError (« dessin impossible ») sur ce cas, et la couche api_web l'attrape", leve)
r_zero = M.verifier(dict(M.nouvelle_corniere(), R=0.0))
r_vif = M.verifier(dict(M.nouvelle_corniere(), standard=False, A=100.0, B=60.0, t=8.0, r=0.0, X=40.0, dx=5.0, Y=45.0, dy=5.0, d=10.0))
chk("R = 0 et une saisie libre sans congé (r = 0, angles vifs) se dessinent (quatre vues chacune)",
    all("impossible" not in v["svg"] for v in DC.vues(r_zero) + DC.vues(r_vif)) and not fautes_coupe(SC, r_vif)
    and not fautes_contraintes(SC, r_vif) and not fautes_moments(SC, r_vif))
r_sans = M.calculer(M.nouvelle_corniere(), diagrammes=False)
refus = []
for fn, mot in ((SC.moments, "diagrammes"), (SC.contraintes, "maillage")):
    try:
        fn(r_sans)
        refus.append(False)
    except ValueError as ex:
        refus.append(mot in str(ex))
chk("sans diagrammes (calculer(…, diagrammes=False)), moments et contraintes refusent avec un message clair", refus == [True, True])

# ================================================================== 10. les peintres
print("\n=== 10. Les peintres de la note ===")


class _D:
    def __init__(self, c):
        self.c = c


for nom in ("A historique avec jeux", "B Y = 85", "B chargée (144 %)"):
    r = RES[nom]
    try:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=landscape(A4))
        for i, v in enumerate(DC.vues(r)):
            peindre_svg(c, v["svg"], 20 + (i % 2) * 400, 40 + (i // 2) * 260, 380, 240)
        c.showPage()
        for i, (cle, _, _, _) in enumerate(SC.VUES):
            peindre(_D(c), SC.FONCTIONS[cle](r, interactif=False), 20 + (i % 2) * 400, 40 + (i // 2) * 260, 380, 240)
        c.showPage()
        c.save()
        doc = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
        t1, t2 = doc[0].get_text(), doc[1].get_text()
        attendus = ("R = 42,76 kN", "Mmax", "larg = 120", "u max", "pas un calcul par éléments finis", "hef = 100")
        chk(f"{nom} : les quatre vues se peignent par le peintre SVG (ndc_pdf) et par le peintre de scène (kit), et leurs textes s'y lisent",
            all(t in t1 for t in attendus) and all(t in t2 for t in attendus), [t for t in attendus if t not in t1 or t not in t2])
    except Exception as ex:  # noqa: BLE001 — un peintre qui plante est l'échec à montrer
        chk(f"{nom} : les quatre vues se peignent", False, f"{type(ex).__name__}: {ex}"[:300])

# ================================================================== 11. pureté
print("\n=== 11. Pureté ===")
src = open("acier/corniere/schemas.py", encoding="utf-8").read()
chk("schemas.py n'importe ni Streamlit, ni numpy, ni le moteur, ni api_web ; il ne calcule rien (pas de verifier/calculer)",
    not re.search(r"^\s*(import|from)\s+(streamlit|numpy|api_web)", src, re.M) and "from . import moteur" not in src
    and "from .moteur" not in src and "verifier(" not in src and "calculer(" not in src.replace("calculer(…, diagrammes=False)", ""))
chk("schemas.py ne recalcule ni moment, ni contrainte, ni taux : aucune division par W ou A, aucune racine carrée hors du contour",
    "sqrt" not in src.replace("math.sqrt", "") and "/ W" not in src and "/ A" not in src and "sigma_eq\"] /" not in src)
src_dc = open("api_web/dessin_corniere.py", encoding="utf-8").read()
chk("dessin_corniere.py est une couche mince : il n'importe que schemas, pose TOILE/TEXTE_PX et ne dessine rien lui-même",
    "from acier.corniere import schemas" in src_dc and "_poly(" not in src_dc and "_rect(" not in src_dc and "TOILE = (626.0, 682.0)" in src_dc)
chk("les légendes et titres viennent de schemas (une seule source pour l'écran et la note)",
    "SC.legende(" in src_dc and "VUES = SC.VUES" in src_dc)

# ================================================================== 12. mutations
print("\n=== 12. Mutations : un trait faux est rejeté ===")
PAQUET = sys.modules["acier.corniere"]


def charger_mutant(avant, apres):
    """schemas.py rechargé depuis sa source avec UN remplacement (qui doit y figurer une fois) ; rend le module mutant
    sans toucher au module installé."""
    chemin = os.path.join(RACINE, "acier", "corniere", "schemas.py")
    s = open(chemin, encoding="utf-8").read()
    if s.count(avant) != 1:
        raise AssertionError(f"mutation périmée : « {avant[:60]} » ({s.count(avant)} fois)")
    m = types.ModuleType("acier.corniere.schemas_mutant")
    m.__package__, m.__file__ = "acier.corniere", chemin
    exec(compile(s.replace(avant, apres), chemin, "exec"), m.__dict__)
    return m


CAS_MUTATIONS = ("A historique avec jeux", "B Y = 85", "A sur poutre", "L = 900 (coupures)", "B chargée (144 %)")


def rejets(mod):
    """Les reproches des contrôles de ce fichier au module ``mod`` (coupe A, B et sur poutre, élévation coupée, moments,
    maillage, data-sel)."""
    out = []
    for nom in CAS_MUTATIONS:
        r = RES[nom]
        for fn in (fautes_coupe, fautes_elevation, fautes_moments, fautes_contraintes, fautes_sel):
            try:
                out += [f"{nom} · {fn.__name__} : {x}" for x in fn(mod, r)]
            except Exception as ex:  # noqa: BLE001 — une exception est aussi un rejet
                out.append(f"{nom} · {fn.__name__} : {type(ex).__name__} {ex}")
    return out


MUTATIONS = [
    ("section 1 déplacée de 1 mm (coupe)", "secs = [_path([\"sc\"], [[(e + r, yv(e)), (e + r, yv(-1.4 * fs))]]),",
     "secs = [_path([\"sc\"], [[(e + r + 1.0, yv(e)), (e + r + 1.0, yv(-1.4 * fs))]]),"),
    ("B dessiné comme A (coupe)", "k = -1.0 if A else 1.0\n\n    def yv(y: float) -> float:\n        \"\"\"y de la scène d'une abscisse",
     "k = -1.0\n\n    def yv(y: float) -> float:\n        \"\"\"y de la scène d'une abscisse"),
    ("charge décalée de 5 mm", "_path([\"fz\"], [[(X, y_t), (X, y_face - 0.6 * fs)]])", "_path([\"fz\"], [[(X + 5.0, y_t), (X + 5.0, y_face - 0.6 * fs)]])"),
    ("ancrages de l'élévation décalés", "ancr += [_circle([\"bw\"], fz(z), ya, 0.95 * d), _circle([\"bo\"], fz(z), ya, d / 2)]",
     "ancr += [_circle([\"bw\"], fz(z) + 1.0, ya, 0.95 * d), _circle([\"bo\"], fz(z) + 1.0, ya, d / 2)]"),
    ("data-sel du profil oublié", "_hach([\"ht\"], pts, max(1.2, 0.22 * fs))], sel(\"profil\"))", "_hach([\"ht\"], pts, max(1.2, 0.22 * fs))], None)"),
    ("diagramme de l'aile horizontale porté dans l'angle", "return (sv, y_ax - k * abs(M) * ech)", "return (sv, y_ax + k * abs(M) * ech)"),
    ("échelle des moments propre à chaque aile", "return (x_ax - abs(M) * ech, yv(sv))", "return (x_ax - abs(M) * ech * 0.9, yv(sv))"),
    ("une courbe de moment oubliée", "for c in sorted(dg[\"courbes\"], key=lambda c: abs(c[\"jeu\"]), reverse=True):",
     "for c in sorted(dg[\"courbes\"], key=lambda c: abs(c[\"jeu\"]), reverse=True)[1:]:"),
    ("enveloppe non remplie", "s.append(_poly([\"envf\"], debut + [pt(p[\"s\"], p[\"M\"]) for p in pts]", "s.append(_poly([\"envx\"], debut + [pt(p[\"s\"], p[\"M\"]) for p in pts]"),
    ("remplissage arrêté au nu de l'angle (pas jusqu'à Mmax)",
     "debut = [pt(prolong[aile][-1][0], 0.0), pt(*prolong[aile][-1])] if aile in prolong else [pt(pts[0][\"s\"], 0.0)]", "debut = [pt(pts[0][\"s\"], 0.0)]"),
    ("Mmax non écrit à gauche (aile verticale)", "        if VERTICALE in prolong:\n            sv_, Mv_ = prolong[VERTICALE][-1]",
     "        if False and VERTICALE in prolong:\n            sv_, Mv_ = prolong[VERTICALE][-1]"),
    ("enveloppe perdue (tracée comme la nominale)", "s.append(_path([\"env\"], [[pt(p[\"s\"], p[\"M\"]) for p in pts]]))",
     "s.append(_path([\"cvm\"], [[pt(p[\"s\"], p[\"M\"]) for p in pts]]))"),
    ("échelle de couleurs inversée", "return _lerp(c0, c1, (t - t0) / (t1 - t0))", "return _lerp(c1, c0, (t - t0) / (t1 - t0))"),
    ("coin non maillé", "    for i in range(len(Zc) - 1):\n        for j in range(len(Zc) - 1):\n            t = max(TXc[i][j]",
     "    for i in range(0):\n        for j in range(len(Zc) - 1):\n            t = max(TXc[i][j]"),
    ("barre de couleur sans les N/mm² à gauche", "\"0\" if t_ == 0 else _num(t_ * s_lim, 1), 0.85 * fs, \"end\"))", "\"\", 0.85 * fs, \"end\"))"),
    ("déformée non dessinée", "amp = PART_DEFORMEE * haut / df[\"u_max\"] if df[\"u_max\"] > 1e-12 else 0.0", "amp = 0.0"),
    ("barre de couleur jusqu'à 105 % quel que soit le maillage",
     "t_haut = max(mz[HORIZONTALE][\"max\"][\"taux\"], mz[VERTICALE][\"max\"][\"taux\"], max(max(l) for l in TXc), 0.05)", "t_haut = 1.2"),
    ("repère de section à travers le dessin", "s.append(_path([\"sl\"], [[(px, py), (px, py - k * d_)]]))",
     "s.append(_path([\"sl\"], [[(px, py + k * e), (px, py - k * d_)]]))"),
    ("maille colorée par le plus petit taux de ses nœuds", "t = max(TX[i][j], TX[i + 1][j], TX[i][j + 1], TX[i + 1][j + 1])",
     "t = min(TX[i][j], TX[i + 1][j], TX[i][j + 1], TX[i + 1][j + 1])"),
    ("maximum posé sur la mauvaise face", "u = (z + 1.0) / 2.0 * mx[\"t\"]", "u = (1.0 - z) / 2.0 * mx[\"t\"]"),
    ("profondeur non dessinée", "    _cote_libre(S.g, fs, (x_fond, ya), (0.0, ya), pos_p, txt_p)\n", "\n"),
    ("tige dessinée sans aller au fond", "x_tige = -prof if not traversant else -prof - ep_n - 1.1 * d", "x_tige = -0.6 * prof if not traversant else -prof - ep_n - 1.1 * d"),
    ("cote L prise sur la longueur dessinée", "sym=\"L\", val=L, lvl=1), sel(\"L\"))", "sym=\"L\", val=Ld, lvl=1), sel(\"L\"))"),
    ("section 3 posée depuis la mauvaise face en B", "_path([\"sc\"], [[(-0.6 * fs, yv(e + r)), (e + 1.0 * fs, yv(e + r))]])",
     "_path([\"sc\"], [[(-0.6 * fs, -yv(e + r)), (e + 1.0 * fs, -yv(e + r))]])"),
    ("ancrage dessiné à Y depuis l'angle (et non la face extérieure)", "    ya = yv(Y)\n    pile: List[Dict] = []", "    ya = yv(Y + e)\n    pile: List[Dict] = []"),
    ("arête de la poutre posée depuis l'ancrage nominal", "a=yv(Y - k * dy), b=Cy.f(-hs)", "a=yv(Y), b=Cy.f(-hs)"),
]
for nom, avant, apres in MUTATIONS:
    try:
        rj = rejets(charger_mutant(avant, apres))
        chk(f"mutation « {nom} » rejetée", bool(rj), "acceptée")
    except AssertionError as ex:
        chk(f"mutation « {nom} » rejetée", False, str(ex))
try:
    temoin = rejets(charger_mutant('nom = "voile"', 'nom = "voile "'))
    chk("témoin : un libellé à peine changé (« voile » → « voile  ») n'est pas rejeté", not temoin, temoin[:3])
except AssertionError as ex:
    chk("témoin", False, str(ex))
chk("le module installé est intact après les mutations", not rejets(SC) and SC.coupe.__module__ == "acier.corniere.schemas")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
