# -*- coding: utf-8 -*-
"""Tests des schémas du garde-corps (acier/garde_corps/schemas.py) : six vues en dessin
technique sur le kit des assemblages, rien n'est recalculé, et chaque garantie rougit si on la
retire :

  1. VUES : élévation, coupe A-A, plan, platine en plan, coupe sur la fixation, section pour le
     cas par défaut ; sans platine en plan quand le montant est soudé sur un plat, ni platine ni
     coupe sur la fixation quand il est scellé ; « Élévation développée » pour un L ; aucune vue
     sur un refus de géométrie ; SVG autonome avec viewBox.
  2. ÉLÉVATION : la dalle hachurée, un montant par montant à son abscisse, la main courante et
     la lisse basse sur toute la longueur, les barreaux centrés par travée à leur entraxe et à
     une largeur lisible, les platines l_p × t_p et leurs chevilles (écrou visible, tige cachée),
     la chaîne des entraxes sur UNE rangée, L, H, h_bas, e_b, les noms M1 …, le gouvernant ◉
     avec son taux, le repère A-A, les quatre charges nommées et la phrase des cas ; les murs
     d'extrémité et l'angle d'un L ; le type lisses, le type panneau (un panneau par travée), le
     montant scellé (mortier), le mode P ; au mur : consoles C1 …, platines sur le mur, chevilles.
  3. COUPE A-A : dalle, platine b_p × t_p, deux chevilles à ±e_x/2, le montant vu de côté à sa
     profondeur h_p, la main courante et la lisse basse COUPÉES (contours du catalogue, trou,
     hachures de la paroi), les cordons, q_h, q_v, Q_r, H, b_p, les noms des pièces ; au mur : le
     mur, la platine verticale, la console, e, q_h qui tire.
  4. PLAN : la main courante en bande (un tronçon droit, deux en L, trois en U), un montant par
     position, les charges perpendiculaires, le repère A-A, le mur et les consoles au mur.
  5. PLATINE EN PLAN : la platine, la section du montant avec sa bande de cordon, les trous à
     ±e_x/2 et ±e_y/2 au diamètre d_0 avec leurs axes, e_x, e_y, b_p, l_p et les chaînes des
     bords (niveau 2 seulement) ; boulons M12 (d_0 13) ; au mur : e_x vertical et la charge q_v.
  6. COUPE SUR LA FIXATION : platine et parois du montant hachurées, deux tiges coupées à
     ±e_x/2 dans le béton (acier pour des boulons), cordons en ocre, cotes h_p, b_p, e_x, t_p, a,
     bras m et z (niveau 2), efforts V et M du cas gouvernant nommé ; soudé sur un plat : pas de
     tige ; au mur : tiges horizontales, V vertical.
  7. SECTION : contour du catalogue, hachures de la paroi seulement (jamais dans le trou), axes
     y et z, cotes h × b et t ; plat plein hachuré.
  8. PEINTRE : chaque vue se peint en PDF par le peintre du kit sans erreur, et ses textes s'y
     retrouvent.
  9. PURETÉ : aucun import de Streamlit ni du moteur dans schemas.py.

Lancement : python3 tests/test_garde_corps_schemas.py (depuis la racine).
"""
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402

from acier.garde_corps import bibliotheque as BI  # noqa: E402
from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import fixation as FX  # noqa: E402
from acier.garde_corps import schemas as SC  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres.rapport import peindre  # noqa: E402
from ndc_pdf.kit import Doc  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def cas(type_="barreaudage", **kw):
    c = ET.nouveau_cas(None, type_)
    for k, v in kw.items():
        c[k] = {**c[k], **v} if isinstance(v, dict) else v
    return c


def prims(d, t=None, cls=None):
    """Les primitives du corps (à plat), filtrées par type et par classe."""
    out = []

    def rec(lst):
        for p in lst:
            if p["t"] == "g":
                rec(p["enfants"])
            else:
                out.append(p)
    rec(d.corps)
    return [p for p in out if (t is None or p["t"] == t) and (cls is None or cls in p["cls"])]


def textes(d):
    out = []

    def rec(lst):
        for p in lst:
            if p["t"] == "text":
                out.append(str(p["txt"]))
            elif p["t"] == "g":
                rec(p["enfants"])
    rec(d.corps)
    rec(d.cotes)
    return out


def vue(res, cle, lvl=2):
    return {k: d for k, _, d in SC.vues(res, SC.options_schemas(lvl=lvl))}[cle]


def rects(d, cls, w=None, h=None, tol=1e-6):
    return [r for r in prims(d, "rect", cls) if (w is None or abs(r["w"] - w) < tol) and (h is None or abs(r["h"] - h) < tol)]


def dedans(poly, p):
    """Point dans un polygone (rayon vers +x)."""
    x, y = p
    n = len(poly)
    ok = False
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            ok = not ok
    return ok


# ================================================================
print("\n=== 1. Vues ===")
res = ET.verifier(cas())
vs = SC.vues(res)
chk("six vues : élévation, coupe A-A, plan, platine en plan, coupe sur la fixation, section",
    [k for k, _, _ in vs] == ["elevation", "coupe", "plan", "platine", "pied", "section"], str([k for k, _, _ in vs]))
chk("titres : « Élévation », « Coupe A-A », « Vue en plan », « Platine en plan », « Coupe sur la fixation », « Section »",
    [t for _, t, _ in vs] == ["Élévation", "Coupe A-A", "Vue en plan", "Platine en plan", "Coupe sur la fixation", "Section"])
svg = SC.svg(vs[0][2], "t0")
chk("SVG autonome : viewBox, style embarqué, identifiant", svg.startswith("<svg") and 'viewBox="' in svg and "<style>" in svg and 'id="t0"' in svg)
chk("montant scellé : ni platine en plan ni coupe sur la fixation",
    [k for k, _, _ in SC.vues(ET.verifier(cas(fixation={"type": "scellee"})))] == ["elevation", "coupe", "plan", "section"])
chk("montant soudé sur un plat : pas de platine en plan, la coupe sur la fixation reste",
    [k for k, _, _ in SC.vues(ET.verifier(cas(fixation={"type": "soudee"})))] == ["elevation", "coupe", "plan", "pied", "section"])
rL = ET.verifier(cas(geometrie={"forme": "L", "fin": "mur"}))
chk("en L : « Élévation développée »", SC.vues(rL)[0][1] == "Élévation développée")
g2 = cas()
g2["geometrie"]["troncons"][0] = {"L": 1000, "s_max": 1000, "entraxes": []}
g2["geometrie"]["debut"] = g2["geometrie"]["fin"] = "mur"
chk("refus de géométrie : aucune vue", SC.vues(ET.verifier(g2)) == [])

# ================================================================
print("\n=== 2. Élévation ===")
e = vue(res, "elevation")
pp = prims(e, "rect", "pp")
xs_m = sorted(round(r["x"] + r["w"] / 2, 1) for r in pp if r["h"] > 500)
chk("cinq montants à x = 0, 900, 1800, 2700, 3600 (rectangles de la hauteur du montant)", xs_m == [0.0, 900.0, 1800.0, 2700.0, 3600.0], str(xs_m))
chk("la dalle : un polygone béton hachuré, rompu, sous toute la longueur", len(prims(e, "poly", "be")) == 1 and min(x for x, _ in prims(e, "poly", "be")[0]["pts"]) < 0
    and max(x for x, _ in prims(e, "poly", "be")[0]["pts"]) > 3600 and len(prims(e, "poly", "be")[0]["pts"]) > 8 and prims(e, "path", "ht"))
longs = [r for r in prims(e, "rect", "ps") if abs(r["w"] - 3600) < 1e-6]
chk("la main courante sur toute la longueur, axe à −(H + t_p) = −1110 ; la lisse basse à −(t_p + h_bas) = −110",
    any(abs(r["y"] + r["h"] / 2 + 1110) < 1e-6 for r in longs) and any(abs(r["y"] + r["h"] / 2 + 110) < 1e-6 for r in longs), str([r["y"] + r["h"] / 2 for r in longs]))
barreaux = sorted((r for r in prims(e, "rect", "ps") if r["w"] < 80 and r["h"] > 500), key=lambda r: r["x"])
chk("les barreaux : 7 par travée (28), à l'entraxe de 110, centrés dans la travée, d'une largeur lisible (≥ 0,22·fs)", len(barreaux) == 28
    and abs((barreaux[1]["x"] - barreaux[0]["x"]) - 110) < 1e-6 and abs((barreaux[0]["x"] + barreaux[0]["w"] / 2) - (900 - 6 * 110) / 2) < 1e-6
    and barreaux[0]["w"] >= 0.22 * e.fs - 1e-6, f"{len(barreaux)} ; x0 = {barreaux[0]['x'] + barreaux[0]['w'] / 2 if barreaux else None}")
chk("une platine l_p × t_p = 150 × 10 sous chaque montant, deux chevilles par montant (écrou sur la platine, tige cachée dans la dalle)",
    len(rects(e, "pp", 150, 10)) == 5 and len(prims(e, "rect", "bo")) == 10 and len(prims(e, "path", "hd")) == 10)
chaines = [p for p in e.cotes if p["t"] == "g" and any(q["t"] == "path" and "dln" in q["cls"] and len(q["lignes"]) >= 6 for q in p["enfants"])]
chk("la chaîne des entraxes : UNE ligne de cote continue avec ses repères (une rangée)", len(chaines) == 1)
txt = textes(e)
chk("les noms M1 … M5, le gouvernant ◉ M2 avec son taux, H = 1100, L = 3600, h_bas et e_b (niveau 2)",
    all(any(t == n for t in txt) for n in ("M1", "M3", "M4", "M5")) and any("◉ M2" in t and "%" in t for t in txt)
    and any("H = 1100" in t for t in txt) and any("L = 3600" in t for t in txt) and any("h_bas" in t and "100" in t for t in txt) and any("e_b" in t and "110" in t for t in txt))
chk("niveau 1 (note) : ni h_bas ni e_b", not any("h_bas" in t or "e_b" in t for t in textes(vue(res, "elevation", 1))))
chk("le repère de la coupe A-A : deux lettres A, dans la travée à droite du montant gouvernant M2 (x = 900), décalé du ⊙ qui s'y trouve",
    [t for t in txt if t == "A"] == ["A", "A"] and any(p["t"] == "path" and "ax" in p["cls"] and 1000 < p["lignes"][0][0][0] < 1350 for p in prims(e, "path", "ax")))
chk("les quatre charges nommées dans le bloc, et la phrase des cas distincts",
    any("q_h = 0,50 kN/m" in t and "q_v = 0,50 kN/m" in t and "Q_r = 0,50 kN" in t and "q_r = 0,50 kN/m²" in t for t in txt)
    and any("quatre cas de charge distincts" in t for t in txt))
chk("⊙ pour q_h (5) et Q_r (1), flèches ↓ pour q_v (5), étiquettes q_h, q_v, Q_r", len(prims(e, "circle", "gv")) >= 7 and len(prims(e, "poly", "fzh")) >= 5
    and all(any(t == n for t in txt) for n in ("q_h", "q_v", "Q_r")))
chk("titre « Élévation »", any(t == "Élévation" for t in txt))
eL = vue(rL, "elevation")
chk("en L avec fin au mur : un trait d'axe à l'angle, un mur hachuré à droite, « angle » écrit",
    any(p["t"] == "path" and abs(p["lignes"][0][0][0] - 3600) < 1e-6 for p in prims(eL, "path", "ax"))
    and any(p["t"] == "poly" and max(x for x, _ in p["pts"]) > 6000 for p in prims(eL, "poly", "pp")) and any(t == "angle" for t in textes(eL)))
rLi = ET.verifier(cas("lisses"))
eLi = vue(rLi, "elevation")
chk("type lisses : la main courante et deux lisses intermédiaires sur toute la longueur, e_l coté, pas de barreau",
    len([r for r in prims(eLi, "rect", "ps") if abs(r["w"] - 3600) < 1e-6]) == 3 and any("e_l" in t for t in textes(eLi))
    and not [r for r in prims(eLi, "rect", "ps") if r["w"] < 80 and r["h"] > 500])
ePa = vue(ET.verifier(cas("panneau")), "elevation")
chk("type panneau : un panneau par travée (4), lisse basse présente", len(prims(ePa, "poly", "bg")) == 4
    and any(abs(r["w"] - 3600) < 1e-6 and abs(r["y"] + r["h"] / 2 + 110) < 1e-6 for r in prims(ePa, "rect", "ps")))
eSc = vue(ET.verifier(cas(fixation={"type": "scellee"})), "elevation")
chk("montant scellé : un scellement au mortier par montant, pas de platine", len(prims(eSc, "rect", "gr")) == 5 and not rects(eSc, "pp", 150, 10))
eP = vue(ET.verifier(cas(charges={"mode": "P", "P": 1.0})), "elevation")
chk("mode P : ⊙ en tête de chaque montant et « P = 1,00 kN par montant »", len(prims(eP, "circle", "gv")) >= 6 and any("P = 1,00 kN par montant" in t for t in textes(eP)))
rM = ET.verifier(cas("murale"))
eM = vue(rM, "elevation")
chk("au mur : consoles C1 … C5, le gouvernant ◉ C…, platines 150 × 150 sur le mur, 20 chevilles vues de face, la face du mur en fond, pas de barreau",
    any(t == "C1" for t in textes(eM)) and any("◉ C" in t for t in textes(eM)) and len(rects(eM, "pp", 150, 150)) == 5
    and len(prims(eM, "circle", "bo")) + len(prims(eM, "circle", "hd")) == 20 and prims(eM, "rect", "bg")
    and not [r for r in prims(eM, "rect", "ps") if r["w"] < 80 and r["h"] > 500])

# ================================================================
print("\n=== 3. Coupe A-A ===")
c = vue(res, "coupe")
chk("la dalle hachurée, la platine b_p × t_p = 150 × 10, deux chevilles à ±e_x/2 = ±50 (écrous), tiges cachées",
    prims(c, "poly", "be") and rects(c, "pp", 150, 10) and sorted(round(r["x"] + r["w"] / 2) for r in prims(c, "rect", "bo")) == [-50, 50] and len(prims(c, "path", "hd")) == 2)
post = [r for r in prims(c, "rect", "pp") if r["h"] > 900]
chk("le montant vu de côté : profondeur h_p = 50, de la platine (y = −10) au dessous de la main courante (y = −1110 + 25)",
    len(post) == 1 and abs(post[0]["w"] - 50) < 1e-6 and abs(post[0]["y"] + post[0]["h"] + 10) < 1e-6 and abs(post[0]["y"] + 1085) < 1e-6)
coupes = [p for p in prims(c, "poly", "pp") if len(p["pts"]) > 100]
trous = [p for p in prims(c, "poly", "bo") if len(p["pts"]) > 100]
chk("la main courante (axe à −1110) et la lisse basse (axe à −110) COUPÉES : contour du catalogue, trou, hachures",
    len(coupes) == 2 and len(trous) == 2 and sorted(round(sum(y for _, y in p["pts"]) / len(p["pts"])) for p in coupes) == [-1110, -110]
    and sum(len(p["lignes"]) for p in prims(c, "path", "ht")) >= 20)
chk("deux cordons au pied du montant", len(prims(c, "poly", "wb")) == 2)
tc = textes(c)
chk("q_h, q_v, Q_r étiquetés ; H = 1100 ; b_p et e_x (niveau 2) ; h_p 50 ; les trois pièces nommées ; titre « Coupe A-A »",
    all(any(t == n for t in tc) for n in ("q_h", "q_v", "Q_r")) and any("H = 1100" in t for t in tc) and any("b_p" in t and "150" in t for t in tc)
    and any("e_x" in t and "100" in t for t in tc) and any("h_p" in t and "50" in t for t in tc)
    and any("RHS 50x30x3 · main courante" in t for t in tc) and any("RHS 50x30x3 · montant" in t for t in tc) and any("lisse basse" in t for t in tc)
    and any(t == "Coupe A-A" for t in tc))
chk("niveau 1 : pas de e_x dans la coupe", not any("e_x" in t for t in textes(vue(res, "coupe", 1))))
cLi = vue(rLi, "coupe")
chk("type lisses : la main courante et les deux lisses coupées (trois sections)", len([p for p in prims(cLi, "poly", "pp") if len(p["pts"]) > 100]) == 3)
cM = vue(rM, "coupe")
chk("au mur : le mur hachuré, la platine verticale t_p × b_p = 10 × 150, deux chevilles horizontales à ±e_x/2, la console (h = 50) du mur à la main courante, e = 100",
    prims(cM, "poly", "be") and rects(cM, "pp", 10, 150) and sorted(round(r["y"] + r["h"] / 2 - 50) for r in prims(cM, "rect", "bo")) == [-50, 50]
    and any(abs(r["h"] - 50) < 1e-6 and abs(r["w"] - 105) < 1e-6 for r in prims(cM, "rect", "pp")) and any("e = 100" in t for t in textes(cM)))
chk("au mur : q_v sur la main courante et q_h qui tire la console hors du mur", any(t == "q_v" for t in textes(cM)) and any("tire la console hors du mur" in t for t in textes(cM)))

# ================================================================
print("\n=== 4. Plan ===")
p = vue(res, "plan")
bandes = [q for q in prims(p, "poly", "ps") if len(q["pts"]) == 4]
chk("droit : une bande de main courante, cinq montants, L₁ = 3600, deux lettres A", len(bandes) == 1 and len([q for q in prims(p, "poly", "pp") if len(q["pts"]) == 4]) == 5
    and any("L" in t and "3600" in t for t in textes(p)) and [t for t in textes(p) if t == "A"] == ["A", "A"])
pL = vue(rL, "plan")
postes_L = [q for q in prims(pL, "poly", "pp") if len(q["pts"]) == 4 and max(x for x, _ in q["pts"]) - min(x for x, _ in q["pts"]) < 100]   # sans le mur du bout
chk("en L (fin au mur) : deux bandes, L₂ coté à droite, un montant à l'angle (3600, 0), le dernier M7 à (3600, −1600), aucun au bout (mur)",
    len([q for q in prims(pL, "poly", "ps") if len(q["pts"]) == 4]) == 2 and any("L" in t and "2400" in t for t in textes(pL))
    and any(abs(sum(x for x, _ in q["pts"]) / 4 - 3600) < 1e-6 and abs(sum(y for _, y in q["pts"]) / 4) < 1e-6 for q in postes_L)
    and any(abs(sum(x for x, _ in q["pts"]) / 4 - 3600) < 1e-6 and abs(sum(y for _, y in q["pts"]) / 4 + 1600) < 1e-6 for q in postes_L)
    and not any(abs(sum(y for _, y in q["pts"]) / 4 + 2400) < 1e-6 for q in postes_L) and len(postes_L) == 7)
rU = ET.verifier(cas(geometrie={"forme": "U"}))
pU = vue(rU, "plan")
chk("en U : trois bandes, des flèches de charge sur chacune", len([q for q in prims(pU, "poly", "ps") if len(q["pts"]) == 4]) == 3 and len(prims(pU, "poly", "fzh")) >= 9)
pM = vue(rM, "plan")
murs = [q for q in prims(pM, "poly", "pp") if len(q["pts"]) == 4 and min(y for _, y in q["pts"]) >= 100 - 1e-6]
consoles = [q for q in prims(pM, "poly", "pp") if len(q["pts"]) == 4 and abs(min(y for _, y in q["pts"])) < 1e-6 and abs(max(y for _, y in q["pts"]) - 100) < 1e-6]
chk("au mur : le mur hachuré du côté extérieur (à e = 100 de l'axe), cinq consoles de l'axe de la main courante au mur, dessinées SOUS la bande de la main courante",
    len(murs) == 1 and len(consoles) == 5 and sum(len(q["lignes"]) for q in prims(pM, "path", "ht")) >= 2
    and pM.corps.index(consoles[0]) < pM.corps.index([q for q in prims(pM, "poly", "ps") if len(q["pts"]) == 4][0]))
rC = ET.verifier(cas(montant={"section": {"nom": "CHS 48.3x3.2"}}))
chk("montants ronds : des cercles en plan", len(prims(vue(rC, "plan"), "circle", "pp")) == 5)

# ================================================================
print("\n=== 5. Platine en plan ===")
pl = vue(res, "platine")
trous = prims(pl, "circle", "bo")
chk("la platine 150 × 150, quatre trous d_0 = 14 à (±50, ±50) avec leurs axes", rects(pl, "pp", 150, 150)
    and sorted((round(t["cx"]), round(t["cy"])) for t in trous) == [(-50, -50), (-50, 50), (50, -50), (50, 50)] and all(abs(t["r"] - 7) < 1e-6 for t in trous)
    and len(prims(pl, "path", "ax")) == 4)
chk("la section du montant à l'échelle (contour, trou) avec sa bande de cordon en ocre, hachures de la paroi",
    len([q for q in prims(pl, "poly", "ps") if len(q["pts"]) > 100]) == 1 and len([q for q in prims(pl, "poly", "bo") if len(q["pts"]) > 100]) == 1
    and len(prims(pl, "poly", "wb")) == 1 and len(prims(pl, "poly", "wb")[0]["pts"]) > 100 and prims(pl, "path", "ht"))
bande = prims(pl, "poly", "wb")[0]["pts"]
chk("la bande du cordon déborde la section de a·√2 = 5,7 (sur h_p = 50 : ±30,7)", abs(max(x for x, _ in bande) - (25 + 4 * 2 ** 0.5)) < 0.3)
tp_ = textes(pl)
chk("cotes e_x = 100, e_y = 100, b_p = 150, l_p = 150 ; chaînes des bords 25 | 100 | 25 (niveau 2) ; titre avec les chevilles ; la charge q_h",
    all(any(t == n for t in tp_) for n in ("e_x = 100", "e_y = 100", "b_p = 150", "l_p = 150")) and tp_.count("25") == 4 and tp_.count("100") == 2
    and any("Platine 150 × 150 × 10 · 4 cheville M12" in t for t in tp_) and any(t == "q_h" for t in tp_))
chk("niveau 1 : pas de chaîne des bords", "25" not in textes(vue(res, "platine", 1)))
rB = ET.verifier(cas(fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}}))
plB = vue(rB, "platine")
chk("boulons M12 : trous d_0 = 13, titre « 4 M12 classe 8.8 »", all(abs(t["r"] - 6.5) < 1e-6 for t in prims(plB, "circle", "bo")) and any("4 M12 classe 8.8" in t for t in textes(plB)))
plM = vue(rM, "platine")
fz = [q for q in prims(plM, "path", "fz")]
chk("au mur : la charge q_v verticale (flèche verticale), titre « au mur »", any(t == "q_v" for t in textes(plM)) and fz
    and abs(fz[0]["lignes"][0][0][0] - fz[0]["lignes"][0][1][0]) < 1e-6 and any("au mur" in t for t in textes(plM)))

# ================================================================
print("\n=== 6. Coupe sur la fixation ===")
d = vue(res, "pied")
tiges = prims(d, "rect", "tg")
chk("deux tiges coupées à ±e_x/2 = ±50, dans le béton (chevilles)", len(tiges) == 2 and sorted(round(r["x"] + r["w"] / 2) for r in tiges) == [-50, 50] and prims(d, "poly", "be"))
parois = [q for q in prims(d, "poly", "pp") if abs(max(x for x, _ in q["pts"]) - min(x for x, _ in q["pts"]) - 3) < 1e-6]
chk("la platine coupée (150 de large) et les deux parois du tube (t = 3) hachurées, deux cordons en ocre",
    any(abs(max(x for x, _ in q["pts"]) - min(x for x, _ in q["pts"]) - 150) < 1e-6 for q in prims(d, "poly", "pp")) and len(parois) == 2
    and len(prims(d, "poly", "wb")) == 2 and sum(len(q["lignes"]) for q in prims(d, "path", "ht")) >= 10)
td = textes(d)
chk("cotes h_p 50, b_p = 150, e_x, t_p 10, a 4, bras m 20,5 et z 75 (niveau 2) ; V et M ; le cas gouvernant nommé avec ses efforts",
    any("h_p" in t and "50" in t for t in td) and any(t == "b_p = 150" for t in td) and any("e_x" in t for t in td) and any("t_p" in t and "10" in t for t in td)
    and any(t == "a 4" for t in td) and any(t == "m 20,5" for t in td) and any(t == "z 75" for t in td) and any(t == "V" for t in td) and any(t == "M" for t in td)
    and any("cas 1 · q_h seule : V = 0,45 kN" in t and "M = 0,495 kN·m" in t for t in td))
chk("niveau 1 : ni m ni z", not any(t.startswith("m ") or t.startswith("z ") for t in textes(vue(res, "pied", 1))))
dK = vue(ET.verifier(cas(charges={"cumul": True})), "pied")
chk("cumul : N (compression) dessiné et le cas 1 + 2 nommé", any(t == "N" for t in textes(dK)) and any("cas 1 + 2" in t and "compression" in t for t in textes(dK)))
dS = vue(ET.verifier(cas(fixation={"type": "soudee"})), "detail") if False else vue(ET.verifier(cas(fixation={"type": "soudee"})), "pied")
chk("soudé sur un plat : pas de tige, support acier, cordons présents, titre « soudé sur plat »", not prims(dS, "rect", "tg") and not prims(dS, "poly", "be")
    and len(prims(dS, "poly", "wb")) == 2 and any("soudé sur plat" in t for t in textes(dS)))
chk("boulons : support acier (pas de teinte béton), tiges présentes", not prims(vue(rB, "pied"), "poly", "be") and len(prims(vue(rB, "pied"), "rect", "tg")) == 2)
dM = vue(rM, "pied")
tg = prims(dM, "rect", "tg")
chk("au mur : deux tiges horizontales à ±e_x/2, console coupée hachurée, V vertical, cas 2 nommé, titre « au mur »",
    len(tg) == 2 and sorted(round(r["y"] + r["h"] / 2) for r in tg) == [-50, 50] and all(r["w"] > r["h"] for r in tg)
    and any("cas 2 · q_v seule : V = 0,45 kN" in t for t in textes(dM)) and any("au mur" in t for t in textes(dM)))

# ================================================================
print("\n=== 7. Section ===")
s = vue(res, "section")
chk("RHS 50x30x3 : contour extérieur, trou, axes y et z, cotes b = 30, h = 50, t 3, nom et masse",
    len([q for q in prims(s, "poly", "pp") if len(q["pts"]) > 100]) == 1 and len([q for q in prims(s, "poly", "bo") if len(q["pts"]) > 100]) == 1
    and all(any(t == n for t in textes(s)) for n in ("y", "z", "b 30", "h = 50", "t 3")) and any("RHS 50x30x3 · 3,41 kg/m" in t for t in textes(s)))
trou = [(float(x), -float(y)) for x, y in BI.contours(res["section_montant"])[1]]
hach = [l for q in prims(s, "path", "ht") for l in q["lignes"]]
chk("aucune hachure ne traverse le trou : le milieu de chaque segment est dans la paroi, et il y en a au moins 12",
    hach and not any(dedans(trou, ((l[0][0] + l[1][0]) / 2, (l[0][1] + l[1][1]) / 2)) for l in hach) and len(hach) >= 12)
sP = vue(ET.verifier(cas(montant={"section": {"nom": "Plat 50x10"}})), "section")
chk("plat 50 × 10 : plein hachuré, cotes h 50 et b 10", any("h" in t and "50" in t for t in textes(sP)) and any("b" in t and "10" in t for t in textes(sP)) and prims(sP, "path", "ht")
    and not prims(sP, "poly", "bo"))
chk("au mur : la charge de la section est q_v", any(t == "q_v" for t in textes(vue(rM, "section"))))

# ================================================================
print("\n=== 8. Peintre ===")
for nom, r in (("défaut", res), ("L mur", rL), ("au mur", rM), ("U", rU), ("lisses", rLi)):
    chemin = os.path.join("/tmp", f"gc_schemas_{abs(hash(nom)) % 1000}.pdf")
    doc = Doc(chemin, landscape(A4))
    doc.new_page(landscape(A4))
    try:
        y = 420
        for k, titre, dv in SC.vues(r, SC.options_schemas(lvl=1)):
            peindre(doc, dv, 30, y - 180, 380, 180)
            y -= 190
            if y < 100:
                doc.new_page(landscape(A4))
                y = 420
        doc.save()
        texte = " ".join(pg.get_text() for pg in pymupdf.open(chemin))
        chk(f"{nom} : toutes les vues se peignent, et « M1 » ou « C1 », « Coupe A-A » et « Platine » sont dans le PDF",
            ("M1" in texte or "C1" in texte) and "Coupe A-A" in texte and "Platine" in texte)
    except Exception as ex:  # noqa: BLE001
        chk(f"{nom} : toutes les vues se peignent", False, f"{type(ex).__name__}: {ex}"[:200])

# ================================================================
print("\n=== 9. Pureté ===")
src = open("acier/garde_corps/schemas.py", encoding="utf-8").read()
chk("schemas.py n'importe ni Streamlit ni etat ni membres ni fixation, et ne vérifie rien", not re.search(r"^\s*(import|from)\s+streamlit", src, re.M)
    and not re.search(r"from \. import (etat|membres|fixation)", src) and "verifier(" not in src)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
