# -*- coding: utf-8 -*-
"""Poutre en béton armé — les DISPOSITIONS CONSTRUCTIVES d'une section (site, 11/2026) : ce que l'écran Streamlit ne
vérifiait pas, et que le ferraillage détaillé (lits positionnés, cadres multiples) rend nécessaire.

Contrôlé, pour chaque section, à partir de la géométrie réelle des barres (``geometrie.abscisses_face``, positions des
lits) — les mêmes positions que le dessin et la note :

- l'espacement LIBRE entre barres voisines d'un même lit, et entre barres de deux lits successifs (la plus petite
  distance nu à nu, que les lits soient alignés ou décalés) ;
- les barres des deux faces, qui ne doivent pas se toucher ;
- les barres d'un lit, qui doivent tenir entre les montants de l'étrier ;
- les armatures latérales, qui ne doivent toucher aucune barre ;
- l'enrobage des barres longitudinales (enrobage + Ø étrier ≥ Ø, condition d'adhérence c_min,b = Ø).

Le minimum de l'espacement libre est celui de l'EC2 §8.2(2), valeurs recommandées (k₁ = 1, k₂ = 5 mm) :

    s_min = max(k₁·Ø ; d_g + k₂ ; 20 mm)        Ø : le plus gros des deux barres voisines ; d_g : le plus gros granulat

ÉTATS : un chevauchement (distance nu à nu ≤ 0) ou une barre hors du cadre est NON CONFORME (« nok ») — la disposition
est physiquement impossible ; un espacement positif mais inférieur à s_min est « À VÉRIFIER » (« warn ») : la méthode du
bureau place le lit 2 avec un jeu de 1 cm par défaut, ce qui donne, pour deux lits Ø16, 14 mm libres — en dessous du
minimum de l'EC2 (25 mm avec d_g = 20 mm). Ce rapport le signale sans changer le calcul (docs/beton/POUTRE_FERRAILLAGE.md).

Unités : cm pour les positions, mm pour les espacements rendus. Aucune grandeur de flexion ni d'effort tranchant n'est
touchée : les vérifications de la méthode du bureau restent celles du module, à l'identique.
"""
import math

from . import geometrie as G

OK, WARN, NOK = "ok", "warn", "nok"
K1 = 1.0            # s_min ≥ k₁·Ø
K2 = 5.0            # mm : s_min ≥ d_g + k₂
S_ABS = 20.0        # mm : s_min ≥ 20 mm
EPS = 1e-6


def s_min(d1: float, d2: float, dg: float) -> float:
    """L'espacement libre minimal (mm) entre deux barres de Ø d1 et d2 (mm) : max(k₁·Ø ; d_g + k₂ ; 20 mm)."""
    return max(K1 * max(d1, d2), dg + K2, S_ABS)


def _libre(x1, y1, d1, x2, y2, d2) -> float:
    """La distance nu à nu (mm) de deux barres d'axes (x, y) en cm et de Ø d1, d2 en mm."""
    return math.hypot(x2 - x1, y2 - y1) * 10.0 - (d1 + d2) / 2.0


def _etat(libre: float, mini: float) -> str:
    return NOK if libre <= EPS else (WARN if libre < mini - EPS else OK)


def controler(p: dict, s: dict, enrobage_calc: float, d_etrier: float, distances: dict) -> dict:
    """Les dispositions d'une section NORMALISÉE ; ``distances`` = {"inf": [...], "sup": [...]} (cm, parement → axe).

    Rend {"etat", "s_min_ref" (mm, pour Ø = 20), "dg", "items": [{"cle", "face", "lit", "texte", "libre", "mini", "etat"}],
          "gouvernant": l'item le plus défavorable, "abscisses": {"inf": [[x…]…], "sup": …}, "laterales": {...},
          "cadres": [{"x0", "x1", "de", "a", "complet", "brins", "type", "d"}]}"""
    b, h = float(p["b"]), float(p["h"])
    dg = float(p.get("dg") or 20.0)
    xs = {f: G.abscisses_face(s[f], b, enrobage_calc, d_etrier) for f in ("inf", "sup")}
    # y depuis le BAS : la face inférieure à sa distance, la supérieure à h − distance
    ys = {"inf": list(distances["inf"]), "sup": [h - e for e in distances["sup"]]}
    items = []

    def ajoute(cle, face, lit, texte, libre, mini, etat=None):
        items.append({"cle": cle, "face": face, "lit": lit, "texte": texte, "libre": libre, "mini": mini,
                      "etat": etat or _etat(libre, mini)})

    nu_g = enrobage_calc + d_etrier / 10.0   # nu intérieur de l'étrier (cm)
    for face in ("inf", "sup"):
        nom_face = "inférieur" if face == "inf" else "supérieur"
        for i, lit in enumerate(s[face]):
            d = float(lit["d"])
            xi = xs[face][i]
            # 1. dans le lit : barres voisines
            if len(xi) > 1:
                pas = min(xi[k + 1] - xi[k] for k in range(len(xi) - 1))
                ajoute("lit", face, i, f"Lit {nom_face} L{i + 1} : espacement entre barres", pas * 10.0 - d, s_min(d, d, dg))
            # 2. le lit tient-il entre les montants de l'étrier ?
            deborde = max(nu_g - (min(xi) - d / 20.0), (max(xi) + d / 20.0) - (b - nu_g))
            if deborde > EPS:
                ajoute("cadre", face, i, f"Lit {nom_face} L{i + 1} : barres hors de l'étrier", -deborde * 10.0, 0.0, NOK)
            # 3. avec le lit précédent : la plus petite distance nu à nu
            if i:
                lp = s[face][i - 1]
                dp = float(lp["d"])
                libre = min(_libre(x1, ys[face][i - 1], dp, x2, ys[face][i], d) for x1 in xs[face][i - 1] for x2 in xi)
                ajoute("entre_lits", face, i, f"Lits {nom_face}s L{i} – L{i + 1} : espacement libre", libre, s_min(d, dp, dg))
    # 4. les deux faces ne se touchent pas
    libre_faces = min(_libre(x1, ys["inf"][i], float(s["inf"][i]["d"]), x2, ys["sup"][j], float(s["sup"][j]["d"]))
                      for i in range(len(s["inf"])) for j in range(len(s["sup"]))
                      for x1 in xs["inf"][i] for x2 in xs["sup"][j])
    d_max = max(float(lit["d"]) for f in ("inf", "sup") for lit in s[f])
    ajoute("faces", None, None, "Faces inférieure et supérieure : espacement libre", libre_faces, s_min(d_max, d_max, dg))
    # 5. les armatures latérales ne touchent aucune barre (contrôle de collision seulement : barres de peau)
    lat = G.laterales(s, p, h, distances["inf"][0], distances["sup"][0])
    if lat["n"]:
        dl = float(lat["d"])
        xg = nu_g + dl / 20.0
        libre_lat = min(_libre(xl, yl, dl, x, ys[f][i], float(s[f][i]["d"]))
                        for xl in (xg, b - xg) for yl in lat["ys"]
                        for f in ("inf", "sup") for i in range(len(s[f])) for x in xs[f][i])
        ajoute("laterales", None, None, "Armatures latérales : espacement libre", libre_lat, 0.0,
               NOK if libre_lat <= EPS else OK)
    # 6. l'enrobage des barres longitudinales : enrobage + Ø étrier ≥ Ø (adhérence, c_min,b = Ø)
    c_long = (enrobage_calc + d_etrier / 10.0) * 10.0
    d_long = max(float(lit["d"]) for f in ("inf", "sup") for lit in s[f])
    ajoute("enrobage", None, None, "Enrobage des barres longitudinales", c_long, d_long,
           OK if c_long >= d_long - EPS else WARN)
    # les cadres : l'emprise de chaque ligne (dessin, note)
    cadres = []
    d1 = float(s["inf"][0]["d"])
    for e in s["etriers"]:
        c = G.emprise_etrier(e, xs["inf"][0], d1, b, enrobage_calc)
        c.update({"type": e["type"], "d": e["d"]})
        cadres.append(c)
    # 7. un cadre FERMÉ tient une barre longitudinale dans chacun de ses angles : en bas, ce sont les barres « de » et
    #    « a » du lit 1 inférieur (par construction) ; en haut, il faut une barre du lit 1 supérieur au contact de chaque
    #    montant (à 5 mm près). Un cadre intérieur sous une face supérieure de deux barres n'en a pas : barres de montage
    #    à prévoir (« À VÉRIFIER »).
    ds = float(s["sup"][0]["d"])
    rs = ds / 20.0
    for k, (e, c) in enumerate(zip(s["etriers"], cadres)):
        if e["type"] == "epingle":
            continue
        de_cm = float(e["d"]) / 10.0
        attendus = (c["x0"] + de_cm + rs, c["x1"] - de_cm - rs)
        manque = [x for x in attendus if not any(abs(xb - x) <= rs + 0.5 for xb in xs["sup"][0])]
        if manque:
            items.append({"cle": "angles_cadre", "face": None, "lit": None,
                          "texte": f"Cadre C{k + 1} : angle{'s' if len(manque) > 1 else ''} supérieur{'s' if len(manque) > 1 else ''} "
                                   f"sans barre (barres de montage à prévoir)", "libre": None, "mini": None, "etat": WARN})
    # une position hors des barres du lit 1 : signalée (le dessin la ramène dans [1 ; n₁], comme le module)
    n1 = int(s["inf"][0]["n"])
    for k, e in enumerate(s["etriers"]):
        for cle in ("de", "a"):
            if e.get(cle) is not None and int(e[cle]) > n1:
                items.append({"cle": "position_cadre", "face": None, "lit": None,
                              "texte": f"Ligne {k + 1} : barre {int(e[cle])} inexistante (le lit 1 inférieur en compte {n1})",
                              "libre": None, "mini": None, "etat": WARN})
    rang = {OK: 0, WARN: 1, NOK: 2}
    etat = max((it["etat"] for it in items), key=lambda e: rang[e], default=OK)
    pire = None
    for it in items:
        if it["libre"] is None:
            continue
        marge = it["libre"] - (it["mini"] or 0.0)
        if pire is None or (rang[it["etat"]], -marge) > (rang[pire["etat"]], -(pire["libre"] - (pire["mini"] or 0.0))):
            pire = it
    return {"etat": etat, "dg": dg, "s_min_ref": s_min(20.0, 20.0, dg), "items": items, "gouvernant": pire,
            "abscisses": xs, "laterales": lat, "cadres": cadres}
