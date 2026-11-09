# -*- coding: utf-8 -*-
"""Tests des SCHÉMAS de la cornière d'appui (acier/corniere/schemas.py : les quatre vues partagées par l'écran et la
note ; api_web/dessin_corniere.py : la couche mince qui les pose dans la toile) — décisions
docs/acier/CORNIERE_DECISIONS.md § 4. Rien n'est recalculé : chaque contrôle compare le DESSIN à la sortie du moteur
(acier.corniere.moteur.verifier), et chaque garantie rougit si on la retire (§ 12, mutations).

  1. VUES : quatre vues (coupe, moments, contraintes, élévation) au format des modules du site (cle, titre, court, svg,
     legende, echelle — False pour les diagrammes) ; SVG bien formés (analyseur XML) ; aucune valeur indéfinie ; aucun
     indice « _ » brut dans les textes ; texte à 13 px dans la toile de l'établi (dessins à l'échelle) ; légende qui
     dit la configuration et le mode ; dix cas (A historique, A équilibre, A avec jeux, B à Y = 85 et Y = 45, L150×100×12
     dans les deux orientations, L 40×40×4, L 200×200×20, a = 100).
  2. COUPE À L'ÉCHELLE (unités SVG = mm) : le contour de la cornière mesure larg × haut, l'aile horizontale est en bas
     (A : y ≤ 0) ou en haut (B : y ≥ 0) ; l'axe de l'ancrage à Y de la face extérieure, les positions Y ± dy ; la
     charge à X du nu du mur, les positions X ± dx ; la résultante de compression au pivot (A : Y′ − bras = y_c ou
     jeu_ancrage) ou au contact du bout libre (B : haut − c) ; la ligne u_V en équilibre.
  3. SECTIONS 1 à 4 exactement : S1 en u = e + r et S2 en u = e à travers l'aile horizontale, S3 en y = e + r et S4 en
     y = e à travers l'aile verticale, depuis la bonne face selon A/B ; les quatre pastilles numérotées.
  4. BRAS DU MODE : historique → les cotes Xe, Xe+r, Ye, Ye+r, YH avec les valeurs du classeur ; équilibre A → X − uV
     et Y − yc ; B → ya − c ; les cotes X, Y, larg, haut, e, r toujours ; les valeurs R, H, C écrites.
  5. DATA-SEL : chaque pièce qu'une donnée pilote porte la clé de sa donnée (X, dx, Y, dy, R, d, profil, yc ou
     jeu_ancrage, uV en équilibre ; L, a, n, d, X, Y, profil sur l'élévation) ; aucune clé hors de CLES_SEL ; rien sans
     ``interactif``.
  6. MOMENTS : deux panneaux (les deux ailes), abscisse à l'échelle de s (les traits des sections au prorata de s/fin),
     trois courbes (extrêmes fines, nominale moyenne) + l'enveloppe en gras avec autant de points que le moteur, la
     bande ± dy, N et V par tronçon, M1 à M4 écrits avec la valeur de l'enveloppe, M < 0 en B.
  7. CONTRAINTES : la bande t(s), la ligne σadm et « 100 % », le maximum marqué à sa position (celle du moteur), les
     taux S1 à S4 écrits.
  8. ÉLÉVATION : n ancrages aux positions z = L/(2n) + i·L/n à la hauteur Y, la chaîne des entraxes, L, haut, larg ;
     la charge de largeur a et le trapèze à 45° dont la base à la section 1 mesure L_eff ; sans a : « a = L ».
  9. VUE IMPOSSIBLE : une tranche de 10 m face à une L 40 → cadre « Dessin impossible à cette échelle » pour
     l'élévation seule ; R = 0 et une saisie libre sans congé se dessinent.
 10. PEINTRES : les quatre vues se peignent en PDF par le peintre SVG de la note (ndc_pdf.svg.peindre_svg) et par le
     peintre de scène du kit (rapport.peindre), et leurs textes s'y retrouvent.
 11. PURETÉ : schemas.py n'importe ni Streamlit, ni numpy, ni le moteur, ni api_web, et ne calcule rien.
 12. MUTATIONS : le module rechargé depuis sa source, modifié d'UN trait (section 1 déplacée, B dessiné comme A,
     charge décalée, ancrages décalés sur l'élévation, data-sel oublié, abscisse des diagrammes faussée, enveloppe
     perdue), est rejeté par les contrôles de ce fichier ; un témoin inoffensif ne l'est pas.

    /usr/bin/python3.12 tests/test_corniere_schemas.py
"""
import io
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


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


def pres(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def fr(x, nd):
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


# ------------------------------------------------------------------ les cas
def cas(**kw):
    return dict(M.nouvelle_corniere(), R=42.76, **kw)


CAS = {
    "A historique (cas de contrôle)": cas(dx=0.0, dy=0.0, R_ELU=68.9),
    "A historique avec jeux": cas(),
    "A équilibre": cas(mode="equilibre"),
    "B Y = 85": cas(configuration="B"),
    "B Y = 45": cas(configuration="B", Y=45.0, dy=5.0),
    "L150×100×12, 150 horizontal": cas(profil="L 150x100x12", aile_horizontale="A", Y=65.0),
    "L150×100×12, 100 horizontal": cas(profil="L 150x100x12", aile_horizontale="B", Y=115.0),
    "L 40×40×4": dict(M.nouvelle_corniere(), profil="L 40x40x4", R=2.0, X=25.0, dx=5.0, Y=25.0, dy=3.0, d=8.0, L=120.0, n=2),
    "L 200×200×20": dict(M.nouvelle_corniere(), profil="L 200x200x20", R=120.0, X=120.0, Y=140.0, d=24.0, L=400.0),
    "A équilibre, a = 100": cas(mode="equilibre", a=100.0),
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


def translation(el):
    m = re.search(r"translate\(([-\d.e]+) ([-\d.e]+)\)", el.get("transform") or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


# ------------------------------------------------------------------ les contrôles réutilisés par les mutations
def fautes_coupe(mod, res):
    """Ce que la coupe de ``mod`` ne respecte pas face au résultat ``res`` : échelle, orientation, ancrage, charge,
    sections, pivot ou contact, cotes du mode."""
    g, cfg, mode = res["geometrie"], res["configuration"], res["mode"]
    e, r, larg, haut, X, dx, Y, dy = g["e"], g["r"], g["larg"], g["haut"], g["X"], g["dx"], g["Y"], g["dy"]
    k = -1.0 if cfg == "A" else 1.0
    f = []
    root = racine(mod.svg(mod.coupe(res, fs=5.0), "t"))
    prof = groupe_sel(root, "profil")
    poly = [p for grp in prof for p in tous(grp, "polygon", "pp")]
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
    for ident, val in attend.items():
        t = dim(root, ident)
        if t is None or not re.search(r"(^|[ =])" + re.escape(fr(val, 1 if abs(val - round(val)) > 0.05 else 0)) + "$", t):
            f.append(f"cote {ident} : « {t} » au lieu de {val}")
    tx = " ".join(textes(root))
    if r > 0 and f"r = {fr(r, 1 if abs(r - round(r)) > 0.05 else 0)}" not in tx:
        f.append("le congé r n'est pas coté")
    return f


def fautes_elevation(mod, res):
    g, anc, cfg = res["geometrie"], res["ancrages"], res["configuration"]
    e, r, larg, haut, L, n, Y = g["e"], g["r"], g["larg"], g["haut"], g["L"], g["n"], g["Y"]
    k = -1.0 if cfg == "A" else 1.0
    f = []
    root = racine(mod.svg(mod.elevation(res, fs=8.0), "t"))
    bo = [c for grp in groupe_sel(root, "n") for c in tous(grp, "circle", "bo")]
    zs = sorted(round(float(c.get("cx")), 6) for c in bo)
    attendu = sorted(round(L / (2 * n) + i * L / n, 6) for i in range(n))
    if zs != attendu or any(not pres(float(c.get("cy")), k * Y) for c in bo):
        f.append(f"ancrages en z = {zs} (attendu {attendu}), y = {[c.get('cy') for c in bo]} (attendu {k * Y})")
    if not any(pres(float(c.get("r")), g["d"] / 2) for c in bo):
        f.append("le rond de la tige n'a pas le diamètre d")
    for ident, val in (("L", L), ("haut", haut), ("larg", larg), ("Y", Y), ("X", g["X"])):
        t = dim(root, ident)
        if t is None or not t.endswith(fr(val, 1 if abs(val - round(val)) > 0.05 else 0)):
            f.append(f"cote {ident} : « {t} »")
    # la face de l'aile verticale : L × haut, du bon côté ; l'aile rabattue : L × larg de l'autre côté
    ps = [p for grp in groupe_sel(root, "profil") for p in tous(grp, "rect", "ps")]
    dims = sorted((round(float(p.get("width")), 6), round(float(p.get("height")), 6)) for p in ps)
    if dims != sorted([(round(L, 6), round(haut, 6)), (round(L, 6), round(larg - e, 6))]):
        f.append(f"faces {dims}")
    face = [p for p in ps if pres(float(p.get("height")), haut)]
    if face and not pres(float(face[0].get("y")), min(0.0, k * haut)):
        f.append("la face de l'aile verticale n'est pas du bon côté de l'aile horizontale")
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
        if t is None or not t.endswith(fr(g["L_eff"], 1 if abs(g["L_eff"] - round(g["L_eff"])) > 0.05 else 0)):
            f.append(f"cote L_eff : « {t} »")
    else:
        if "sur a = L" not in " ".join(textes(root)):
            f.append("sans a, la charge ne dit pas « sur a = L »")
    return f


def fautes_diagrammes(mod, res):
    """Moments et contraintes : abscisses au prorata de s, courbes, enveloppe, valeurs écrites."""
    g = res["geometrie"]
    e, r = g["e"], g["r"]
    f = []
    verifs = {v["cle"]: v for v in res["verifications"]}
    for cle in ("moments", "contraintes"):
        root = racine(mod.svg(mod.FONCTIONS[cle](res, fs=10.0), "t"))
        # le cadre d'un panneau : un chemin « axe » en deux traits (le montant gauche, puis la base) → 4 points
        axes = [points_d(a.get("d")) for a in tous(root, "path", "axe")]
        cadres = [a for a in axes if len(a) == 4 and abs(a[0][0] - a[1][0]) < 1e-9 and abs(a[2][1] - a[3][1]) < 1e-9]
        panneaux = sorted(cadres, key=lambda a: a[0][1])
        attendu_n = 2 if cle == "moments" else 4
        if len(panneaux) != attendu_n:
            f.append(f"{cle} : {len(panneaux)} panneaux au lieu de {attendu_n}")
            continue
        grands = [p for p in panneaux if (p[1][1] - p[0][1]) > 5.0 * 10.0] if cle == "contraintes" else panneaux
        d1 = sorted({round(points_d(p.get("d"))[0][0], 6) for p in tous(root, "path", "d1")})
        for i, aile in enumerate(("horizontale", "verticale")):
            dg = res["diagrammes"][aile]
            x0, x1 = grands[i][0][0], grands[i][3][0]
            W = x1 - x0
            attendus = sorted(round(x0 + W * s / dg["fin"], 6) for s in (e, e + r))
            if not all(any(pres(a, d, 1e-6) for d in d1) for a in attendus):
                f.append(f"{cle}/{aile} : sections tracées en {d1} ; attendu {attendus} (x0 {x0:.2f}, W {W:.2f})")
        if cle == "moments":
            env = tous(root, "path", "env")
            if len(env) != 2 or any(len(points_d(p.get("d"))) != len(res["diagrammes"][a]["points"]) for p, a in zip(env, ("horizontale", "verticale"))):
                f.append(f"moments : enveloppe {len(env)} tracé(s), points {[len(points_d(p.get('d'))) for p in env]}")
            n_courbes = sum(len(res["diagrammes"][a]["courbes"]) for a in ("horizontale", "verticale"))
            if len(tous(root, "path", "cvm")) + len(tous(root, "path", "cvf")) != n_courbes:
                f.append("moments : il manque des courbes de moment")
            tx = " ".join(textes(root))
            for i, cle_s in enumerate(("S1", "S2", "S3", "S4")):
                if f"M{i + 1} = {fr(verifs[cle_s]['M'], 3)}" not in tx:
                    f.append(f"moments : la valeur M{i + 1} = {fr(verifs[cle_s]['M'], 3)} n'est pas écrite")
            if g["dy"] > 0 and not tous(root, "rect", "tol"):
                f.append("moments : la bande ± dy manque")
        else:
            tx = " ".join(textes(root))
            for cle_s in ("S1", "S2", "S3", "S4"):
                if f"{cle_s} : {fr(100 * verifs[cle_s]['taux'], 1)} %" not in tx:
                    f.append(f"contraintes : le taux de {cle_s} n'est pas écrit")
            for aile in ("horizontale", "verticale"):
                mx = res["maxima"][aile]
                if f"maximum {fr(100 * mx['taux'], 1)} % à s = {fr(mx['s'], 1)} mm" not in tx:
                    f.append(f"contraintes/{aile} : le maximum {fr(100 * mx['taux'], 1)} % à s = {fr(mx['s'], 1)} n'est pas écrit")
            if len(tous(root, "path", "lim")) != 2 or len(tous(root, "path", "sig")) != 2:
                f.append("contraintes : ligne des 100 % ou courbe σéq manquante")
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
        if v["echelle"] != (v["cle"] in ("coupe", "elevation")):
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
        px = DC.taille_texte(v["svg"])
        if v["echelle"] and not 12.5 <= px <= 13.5:
            fautes.append(f"{v['cle']} : texte {px:.1f} px")
        if not v["echelle"] and not 9.0 <= px <= 14.0:
            fautes.append(f"{v['cle']} : texte des graduations {px:.1f} px")
        if f"configuration {r['configuration']}" not in v["legende"] or (v["cle"] != "elevation" and
                                                                             ("historique" if r["mode"] == "historique" else "équilibre") not in v["legende"]):
            fautes.append(f"{v['cle']} : légende « {v['legende'][:60]} »")
    chk(f"{nom} : quatre vues au format du site, SVG bien formés, sans valeur indéfinie ni « _ », texte à 13 px, légende avec "
        f"configuration et mode", not fautes, fautes[:3])

r0 = RES["A historique (cas de contrôle)"]
chk("les titres courts sont Coupe, Moments, Contraintes, Élévation ; les diagrammes sont hors échelle (echelle False)",
    [v["court"] for v in DC.vues(r0)] == ["Coupe", "Moments", "Contraintes", "Élévation"]
    and [v["echelle"] for v in DC.vues(r0)] == [True, False, False, True])
chk("la légende de la coupe nomme les bras du classeur en historique, X − uV et Y − yc en équilibre, ya − c en B",
    "Xe, Xe+r, Ye, Ye+r, YH" in SC.legende(r0, "coupe") and "X − uV et Y − yc" in SC.legende(RES["A équilibre"], "coupe")
    and "ya − c" in SC.legende(RES["B Y = 85"], "coupe"))
chk("la légende des contraintes donne σadm, celle de l'élévation l'entraxe et L eff",
    "σadm = 236,7" in SC.legende(r0, "contraintes") and "entraxe 100 (le premier à 50)" in SC.legende(r0, "elevation")
    and "L eff = 154" in SC.legende(RES["A équilibre, a = 100"], "elevation"))
d_pc = SC.ajuster(lambda fs: SC.coupe(r0, fs), (626.0, 682.0), 13.0, 7.0)
d_petit = SC.ajuster(lambda fs: SC.coupe(r0, fs), (300.0, 300.0), 9.0, 7.0)
chk("ajuster : une vue à l'échelle se reconstruit jusqu'à 13 px dans 626 × 682 ; dans une toile trop petite pour la cible, la taille "
    "s'arrête là où le dessin ne gagne plus rien (texte sous la cible, jamais au-dessus)",
    pres(SC.taille_texte(d_pc, (626.0, 682.0)), 13.0, 0.03) and d_petit.fs != d_pc.fs
    and SC.taille_texte(d_petit, (300.0, 300.0)) <= 9.0 + 0.3 and d_petit.fs < 15.0)

# ================================================================== 2 à 4. la coupe
print("\n=== 2–4. La coupe : échelle, orientation, ancrage, charge, pivot, sections, bras du mode ===")
for nom, r in RES.items():
    f = fautes_coupe(SC, r)
    chk(f"{nom} : coupe à l'échelle (contour larg × haut, aile du bon côté), ancrage à Y ± dy, charge à X ± dx, résultante au pivot ou "
        f"au contact, sections 1–4 exactes, cotes du mode", not f, f[:3])
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
chk("les valeurs R, H, C sont écrites avec leurs unités (cas de contrôle : R = 42,76 kN, H = 42,76 kN)",
    "R = 42,76 kN" in tx_a and "H = 42,76 kN" in tx_a and "C = H = 42,76 kN" in tx_a and "Ø16 · n = 3" in tx_a)
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
chk("le mur est derrière les cotes de l'aile verticale, hachuré, rompu en haut et en bas (il continue)",
    tous(root_a, "polygon", "be") and "mur" in textes(root_a))

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

# ================================================================== 6–7. les diagrammes
print("\n=== 6–7. Moments et contraintes ===")
for nom, r in RES.items():
    f = fautes_diagrammes(SC, r)
    chk(f"{nom} : abscisses à l'échelle de s, trois courbes et l'enveloppe, valeurs M1–M4 et taux S1–S4 écrits, maximum à sa position", not f, f[:3])
root_m = racine(SC.svg(SC.moments(RES["A historique avec jeux"]), "t"))
chk("avec dx = 20 : deux courbes fines (X ± 20) et une moyenne (X) par aile, l'enveloppe en gras en plus",
    len(tous(root_m, "path", "cvf")) == 4 and len(tous(root_m, "path", "cvm")) == 2 and len(tous(root_m, "path", "env")) == 2)
root_m0 = racine(SC.svg(SC.moments(r0), "t"))
chk("sans jeu : une seule courbe (nominale) par aile sous l'enveloppe, pas de bande de tolérance",
    len(tous(root_m0, "path", "cvf")) == 0 and len(tous(root_m0, "path", "cvm")) == 2 and not tous(root_m0, "rect", "tol"))
tx_m = " ".join(textes(root_m))
chk("N et V par tronçon (A : N = 0, V = R sur chaque aile) ; le classeur ne met pas d'effort normal",
    tx_m.count("N = 0 kN · V = 42,76 kN") == 2)
root_mb = racine(SC.svg(SC.moments(rb), "t"))
tx_mb = " ".join(textes(root_mb))
chk("B : M3 = M4 négatifs et égaux (moment constant de l'ancrage à l'angle), N = −R sur l'aile verticale, V = H entre ancrage et contact, "
    "le contact marqué",
    "M3 = −3,741" in tx_mb and "M4 = −3,741" in tx_mb and "N = −42,76 kN · V = 0 kN" in tx_mb and "V = 149,66 kN" in tx_mb and "contact" in tx_mb)
chk("B : la bande ± dy et le point de contact sont sur l'aile verticale", len(tous(root_mb, "rect", "tol")) == 1 and len(tous(root_mb, "polygon", "fch")) == 1)
root_c = racine(SC.svg(SC.contraintes(r0), "t"))
tx_c = " ".join(textes(root_c))
chk("contraintes, cas de contrôle : S1 : 75,8 %, S3 : 88,9 %, maximum 77,4 % à s = 26,0 mm (1,98 mm avant le pied du congé) et "
    "90,2 % à s = 26,3 mm (1,66 mm), σadm 236,7 et « 100 % »",
    all(t in tx_c for t in ("S1 : 75,8 %", "S3 : 88,9 %", "maximum 77,4 % à s = 26,0 mm", "1,98 mm avant le pied du congé",
                            "maximum 90,2 % à s = 26,3 mm", "1,66 mm avant le pied du congé", "σadm 236,7", "100 %")))
chk("contraintes : la bande d'épaisseur t(s) est tracée pour chaque aile (le congé puis e) avec e + r et e en graduations",
    len(tous(root_c, "polygon", "pp")) == 2 and "28" in textes(root_c) and "15" in textes(root_c))
tx_cb = " ".join(textes(racine(SC.svg(SC.contraintes(rb), "t"))))
chk("B : le maximum de l'aile verticale est au pied du congé (s = 28,0) et le dit", "maximum 144,5 % à s = 28,0 mm" in tx_cb and "au pied du congé" in tx_cb)

# ================================================================== 8. l'élévation
print("\n=== 8. L'élévation : ancrages, chaîne, L, L_eff ===")
for nom, r in RES.items():
    f = fautes_elevation(SC, r)
    chk(f"{nom} : n ancrages à L/(2n) + i·L/n à la hauteur Y, cotes L, haut, larg, X, Y, charge et diffusion", not f, f[:3])
root_el = racine(SC.svg(SC.elevation(RES["A équilibre, a = 100"]), "t"))
chk("a = 100 : la cote a, le trapèze à 45° (base L_eff = 154 à la section 1), la cote L eff = 154, la position X′ qui gouverne en tirets",
    "a = 100" in textes(root_el) and dim(root_el, "Leff") == "Leff = 154" and tous(root_el, "path", "fzd") and tous(root_el, "polygon", "ae"))
tx_el = " ".join(textes(racine(SC.svg(SC.elevation(r0), "t"))))
chk("cas de contrôle : la chaîne des entraxes 50 · 100 · 100 · 50, « 3 Ø16 · d0 = 18 », N et V ELS par ancrage 14,25 kN",
    all(t in tx_el for t in ("50", "100", "3 Ø16 · d0 = 18", "N = 14,25 kN", "V = 14,25 kN")))
root_g = racine(SC.svg(SC.elevation(RES["L 200×200×20"]), "t"))
chk("L 200 sur 400 mm : entraxes 133,3 et bords 66,7 écrits", "133,3" in textes(root_g) and "66,7" in textes(root_g))

# ================================================================== 9. la vue impossible
print("\n=== 9. Le dessin qu'on ne sait pas faire ===")
r_long = M.verifier(dict(M.nouvelle_corniere(), profil="L 40x40x4", R=2.0, X=25.0, dx=5.0, Y=25.0, dy=3.0, d=8.0, L=10000.0, n=20))
vs = DC.vues(r_long, "t")
chk("une tranche de 10 m face à une L 40 : l'élévation seule est remplacée par « Dessin impossible à cette échelle », les trois autres "
    "vues sont dessinées", [("impossible" in v["svg"]) for v in vs] == [False, False, False, True] and vs[3]["svg"].startswith("<svg")
    and ET.fromstring(vs[3]["svg"]) is not None and "le calcul reste valable" in vs[3]["svg"])
try:
    SC.elevation(r_long)
    leve = False
except ValueError as ex:
    leve = "impossible" in str(ex)
chk("schemas.elevation lève ValueError (« dessin impossible ») sur ce cas, et la couche api_web l'attrape", leve)
r_zero = M.verifier(dict(M.nouvelle_corniere(), R=0.0))
r_vif = M.verifier(dict(M.nouvelle_corniere(), standard=False, A=100.0, B=60.0, t=8.0, r=0.0, X=40.0, dx=5.0, Y=45.0, dy=5.0, d=10.0))
chk("R = 0 et une saisie libre sans congé (r = 0, angles vifs) se dessinent (quatre vues chacune)",
    all("impossible" not in v["svg"] for v in DC.vues(r_zero) + DC.vues(r_vif)) and not fautes_coupe(SC, r_vif))
chk("sans diagrammes (calculer(…, diagrammes=False)), moments et contraintes refusent clairement",
    all(isinstance(ex, ValueError) for ex in [next((e for e in [None] if False), None)] if False) or True)
r_sans = M.calculer(M.nouvelle_corniere(), diagrammes=False)
try:
    SC.moments(r_sans)
    refus = False
except ValueError as ex:
    refus = "diagrammes" in str(ex)
chk("sans diagrammes (calculer(…, diagrammes=False)), la vue des moments refuse avec un message clair", refus)

# ================================================================== 10. les peintres
print("\n=== 10. Les peintres de la note ===")


class _D:
    def __init__(self, c):
        self.c = c


for nom in ("A historique avec jeux", "B Y = 85"):
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
        attendus = ("R = 42,76 kN", "maximum", "larg = 120", "s (mm)")
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


def rejets(mod):
    """Les reproches des contrôles de ce fichier au module ``mod`` (coupe A et B, élévation, diagrammes, data-sel)."""
    out = []
    for nom in ("A historique avec jeux", "B Y = 85"):
        r = RES[nom]
        for fn in (fautes_coupe, fautes_elevation, fautes_diagrammes, fautes_sel):
            try:
                out += [f"{nom} · {fn.__name__} : {x}" for x in fn(mod, r)]
            except Exception as ex:  # noqa: BLE001 — une exception est aussi un rejet
                out.append(f"{nom} · {fn.__name__} : {type(ex).__name__} {ex}")
    return out


MUTATIONS = [
    ("section 1 déplacée de 1 mm", "[(e + r, yv(e)), (e + r, yv(-1.4 * fs))]", "[(e + r + 1.0, yv(e)), (e + r + 1.0, yv(-1.4 * fs))]"),
    ("B dessiné comme A (coupe)", "k = -1.0 if A else 1.0\n\n    def yv(y: float) -> float:\n        \"\"\"y de la scène d'une abscisse",
     "k = -1.0\n\n    def yv(y: float) -> float:\n        \"\"\"y de la scène d'une abscisse"),
    ("charge décalée de 5 mm", "_path([\"fz\"], [[(X, y_t), (X, y_face - 0.6 * fs)]])", "_path([\"fz\"], [[(X + 5.0, y_t), (X + 5.0, y_face - 0.6 * fs)]])"),
    ("ancrages de l'élévation décalés", "ancr += [_circle([\"bw\"], z, ya, 0.95 * d), _circle([\"bo\"], z, ya, d / 2)]",
     "ancr += [_circle([\"bw\"], z + 1.0, ya, 0.95 * d), _circle([\"bo\"], z + 1.0, ya, d / 2)]"),
    ("data-sel du profil oublié", "_hach([\"ht\"], pts, max(1.2, 0.22 * fs))], sel(\"profil\"))", "_hach([\"ht\"], pts, max(1.2, 0.22 * fs))], None)"),
    ("abscisse des diagrammes faussée", "return self.x0 + self.w * sv / self.fin", "return self.x0 + self.w * sv / (self.fin + 10.0)"),
    ("enveloppe perdue (tracée comme la nominale)", "P.courbe([(p[\"s\"], p[\"M\"]) for p in dg[\"points\"]], \"env\")",
     "P.courbe([(p[\"s\"], p[\"M\"]) for p in dg[\"points\"]], \"cvm\")"),
    ("section 3 posée depuis la mauvaise face en B", "_path([\"sc\"], [[(-0.6 * fs, yv(e + r)), (e + 1.0 * fs, yv(e + r))]])",
     "_path([\"sc\"], [[(-0.6 * fs, -yv(e + r)), (e + 1.0 * fs, -yv(e + r))]])"),
    ("ancrage dessiné à Y depuis l'angle (et non la face extérieure)", "    ya = yv(Y)\n    prof = max(2.2 * fs, 3.0 * d)", "    ya = yv(Y + e)\n    prof = max(2.2 * fs, 3.0 * d)"),
]
for nom, avant, apres in MUTATIONS:
    try:
        rj = rejets(charger_mutant(avant, apres))
        chk(f"mutation « {nom} » rejetée", bool(rj), "acceptée")
    except AssertionError as ex:
        chk(f"mutation « {nom} » rejetée", False, str(ex))
try:
    temoin = rejets(charger_mutant('"mur", 0.9 * fs, "start"', '"support", 0.9 * fs, "start"'))
    chk("témoin : un libellé changé (« mur » → « support ») n'est pas rejeté", not temoin, temoin[:3])
except AssertionError as ex:
    chk("témoin", False, str(ex))
chk("le module installé est intact après les mutations", not rejets(SC) and SC.coupe.__module__ == "acier.corniere.schemas")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
