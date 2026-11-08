# -*- coding: utf-8 -*-
"""Poutre en béton armé — les DISPOSITIONS CONSTRUCTIVES d'une section (site, 11/2026) : ce que l'écran Streamlit ne
vérifiait pas, et que le ferraillage détaillé (lits positionnés, cadres multiples) rend nécessaire.

Contrôlé, pour chaque section, à partir de la géométrie réelle des barres (``geometrie.abscisses_face``, positions des
lits) — les mêmes positions que le dessin et la note :

- l'espacement LIBRE entre barres voisines d'un même lit, et entre barres de deux lits successifs (la plus petite
  distance nu à nu, que les lits soient alignés ou décalés) ;
- les barres de deux lits successifs SUPERPOSÉES (EC2 8.2(3) : « les barres de chaque lit sont placées verticalement
  au-dessus les unes des autres ») ;
- les barres des deux faces, qui ne doivent pas se toucher ;
- les barres d'un lit, qui doivent tenir entre les montants de l'étrier ;
- les armatures latérales, qui ne doivent toucher aucune barre ;
- l'enrobage des barres longitudinales (enrobage + Ø étrier ≥ Ø, condition d'adhérence c_min,b = Ø, EC2 4.4.1.2) ;
- chaque cadre fermé tient une barre longitudinale dans ses angles supérieurs, chaque épingle une barre haute à son
  droit ; une position de cadre hors des barres.

Le minimum de l'espacement libre est celui de l'EC2 §8.2(2), valeurs recommandées (k₁ = 1, k₂ = 5 mm) :

    s_min = max(k₁·Ø ; d_g + k₂ ; 20 mm)        Ø : le plus gros des deux barres voisines ; d_g : le plus gros granulat

ÉTATS (décision du bureau, 11/2026 — docs/beton/POUTRE_FERRAILLAGE.md §2) :
- NON CONFORME (« nok ») : une règle de l'EC2 n'est pas tenue — un espacement libre sous s_min (même positif : 10 mm
  pour 25 mm reste une non-conformité tant que la géométrie n'est pas corrigée), un chevauchement, une barre hors du
  cadre, un enrobage sous Ø. Une non-conformité ne se « valide » pas : seule une correction la lève.
- À EXAMINER (« warn ») : une disposition que l'ingénieur doit regarder et peut JUSTIFIER (barres de montage dans un
  angle de cadre, lits non superposés, cadre posé sur une barre inexistante d'un ancien fichier). Une justification
  enregistrée (clé « justifs » de la section) la rend « examinée » — jamais « conforme » : l'état effectif compte comme
  tenu, l'observation et son texte restent écrits (écran et note).
- « impossible » (marque des items nok) : la disposition ne se POSE pas (barres qui se chevauchent ou sortent du cadre)
  — le sélecteur d'armatures ne la propose pas ; un espacement sous s_min, lui, se pose (il est « serré »).

Chaque item porte un IDENTIFIANT stable (« lit:inf:0 », « angles_cadre:1 »…) et une EMPREINTE (ce qui a été examiné :
les valeurs arrondies, les angles en cause) : une justification ne vaut que pour l'empreinte qu'elle a vue — la
disposition change, la justification devient « à revoir ».

Unités : cm pour les positions, mm pour les espacements rendus. Aucune grandeur de flexion ni d'effort tranchant n'est
touchée : les vérifications de la méthode du bureau restent celles du module, à l'identique.
"""
import math

from . import geometrie as G

OK, WARN, NOK = "ok", "warn", "nok"
JUST = "just"       # état EFFECTIF d'une observation « à examiner » justifiée par l'ingénieur
K1 = 1.0            # s_min ≥ k₁·Ø
K2 = 5.0            # mm : s_min ≥ d_g + k₂
S_ABS = 20.0        # mm : s_min ≥ 20 mm
EPS = 1e-6
RANG = {OK: 0, JUST: 0, WARN: 1, NOK: 2}


def s_min(d1: float, d2: float, dg: float) -> float:
    """L'espacement libre minimal (mm) entre deux barres de Ø d1 et d2 (mm) : max(k₁·Ø ; d_g + k₂ ; 20 mm)."""
    return max(K1 * max(d1, d2), dg + K2, S_ABS)


def _libre(x1, y1, d1, x2, y2, d2) -> float:
    """La distance nu à nu (mm) de deux barres d'axes (x, y) en cm et de Ø d1, d2 en mm."""
    return math.hypot(x2 - x1, y2 - y1) * 10.0 - (d1 + d2) / 2.0


def _etat(libre: float, mini: float) -> str:
    return OK if libre >= mini - EPS else NOK


def _empreinte(libre, mini) -> str:
    return f"{libre:.1f}/{mini:.1f}"


def controler(p: dict, s: dict, enrobage_calc: float, d_etrier: float, distances: dict) -> dict:
    """Les dispositions d'une section NORMALISÉE ; ``distances`` = {"inf": [...], "sup": [...]} (cm, parement → axe).

    Rend {"etat" (effectif : les observations justifiées comptent comme tenues), "dg", "s_min_ref" (mm, pour Ø = 20),
          "items": [{"id", "cle", "face", "lit", "texte", "libre", "mini", "etat", "etat_eff", "impossible", "empreinte",
                     ["justification" | "justification_perimee"]}],
          "compte": {"ok", "warn", "nok", "just"}, "gouvernant": l'item le plus défavorable,
          "abscisses": {"inf": [[x…]…], "sup": …}, "laterales": {...},
          "cadres": [{"x0", "x1", "de", "a", "complet", "brins", "type", "d"}]}"""
    b, h = float(p["b"]), float(p["h"])
    dg = float(p.get("dg") or 20.0)
    xs = {f: G.abscisses_face(s[f], b, enrobage_calc, d_etrier) for f in ("inf", "sup")}
    # y depuis le BAS : la face inférieure à sa distance, la supérieure à h − distance
    ys = {"inf": list(distances["inf"]), "sup": [h - e for e in distances["sup"]]}
    items = []

    def ajoute(ident, cle, face, lit, texte, libre, mini, etat=None, impossible=False, empreinte=None):
        items.append({"id": ident, "cle": cle, "face": face, "lit": lit, "texte": texte, "libre": libre, "mini": mini,
                      "etat": etat or _etat(libre, mini), "impossible": impossible,
                      "empreinte": empreinte if empreinte is not None else _empreinte(libre, mini or 0.0)})

    nu_g = enrobage_calc + d_etrier / 10.0   # nu intérieur de l'étrier (cm)
    for face in ("inf", "sup"):
        nom_face = "inférieur" if face == "inf" else "supérieur"
        for i, lit in enumerate(s[face]):
            d = float(lit["d"])
            xi = xs[face][i]
            # 1. dans le lit : barres voisines
            if len(xi) > 1:
                pas = min(xi[k + 1] - xi[k] for k in range(len(xi) - 1))
                libre = pas * 10.0 - d
                ajoute(f"lit:{face}:{i}", "lit", face, i, f"Lit {nom_face} L{i + 1} : espacement entre barres", libre,
                       s_min(d, d, dg), impossible=libre <= EPS)
            # 2. le lit tient-il entre les montants de l'étrier ?
            deborde = max(nu_g - (min(xi) - d / 20.0), (max(xi) + d / 20.0) - (b - nu_g))
            if deborde > EPS:
                ajoute(f"cadre:{face}:{i}", "cadre", face, i, f"Lit {nom_face} L{i + 1} : barres hors de l'étrier",
                       -deborde * 10.0, 0.0, NOK, impossible=True)
            if i:
                lp = s[face][i - 1]
                dp = float(lp["d"])
                # 3. avec le lit précédent : la plus petite distance nu à nu
                libre = min(_libre(x1, ys[face][i - 1], dp, x2, ys[face][i], d) for x1 in xs[face][i - 1] for x2 in xi)
                ajoute(f"entre_lits:{face}:{i}", "entre_lits", face, i, f"Lits {nom_face}s L{i} – L{i + 1} : espacement libre",
                       libre, s_min(d, dp, dg), impossible=libre <= EPS)
                # 4. superposés (EC2 8.2(3)) : chaque barre du lit au-dessus d'une barre du lit précédent (à ½ Ø près)
                tol = max(0.5, max(d, dp) / 20.0)
                ecarts = [min(abs(x2 - x1) for x1 in xs[face][i - 1]) for x2 in xi]
                seules = sum(1 for e in ecarts if e > tol + EPS)
                if seules:
                    ajoute(f"superposes:{face}:{i}", "superposes", face, i,
                           f"Lits {nom_face}s L{i} – L{i + 1} : {seules} barre{'s' if seules > 1 else ''} non superposée{'s' if seules > 1 else ''} (EC2 8.2(3))",
                           None, None, WARN, empreinte=" ".join(f"{x:.1f}" for x in xi) + "|" + " ".join(f"{x:.1f}" for x in xs[face][i - 1]))
    # 5. les deux faces ne se touchent pas
    libre_faces = min(_libre(x1, ys["inf"][i], float(s["inf"][i]["d"]), x2, ys["sup"][j], float(s["sup"][j]["d"]))
                      for i in range(len(s["inf"])) for j in range(len(s["sup"]))
                      for x1 in xs["inf"][i] for x2 in xs["sup"][j])
    d_max = max(float(lit["d"]) for f in ("inf", "sup") for lit in s[f])
    ajoute("faces", "faces", None, None, "Faces inférieure et supérieure : espacement libre", libre_faces,
           s_min(d_max, d_max, dg), impossible=libre_faces <= EPS)
    # 6. les armatures latérales ne touchent aucune barre (contrôle de collision seulement : barres de peau)
    lat = G.laterales(s, p, h, distances["inf"][0], distances["sup"][0])
    if lat["n"]:
        dl = float(lat["d"])
        xg = nu_g + dl / 20.0
        libre_lat = min(_libre(xl, yl, dl, x, ys[f][i], float(s[f][i]["d"]))
                        for xl in (xg, b - xg) for yl in lat["ys"]
                        for f in ("inf", "sup") for i in range(len(s[f])) for x in xs[f][i])
        ajoute("laterales", "laterales", None, None, "Armatures latérales : espacement libre", libre_lat, 0.0,
               NOK if libre_lat <= EPS else OK, impossible=libre_lat <= EPS)
    # 7. l'enrobage des barres longitudinales : enrobage + Ø étrier ≥ Ø (adhérence, c_min,b = Ø ; EC2 4.4.1.2)
    c_long = (enrobage_calc + d_etrier / 10.0) * 10.0
    d_long = max(float(lit["d"]) for f in ("inf", "sup") for lit in s[f])
    ajoute("enrobage", "enrobage", None, None, "Enrobage des barres longitudinales", c_long, d_long,
           OK if c_long >= d_long - EPS else NOK)
    # les cadres : l'emprise de chaque ligne (dessin, note)
    cadres = []
    d1 = float(s["inf"][0]["d"])
    for e in s["etriers"]:
        c = G.emprise_etrier(e, xs["inf"][0], d1, b, enrobage_calc)
        c.update({"type": e["type"], "d": e["d"]})
        cadres.append(c)
    # 8. un cadre FERMÉ tient une barre longitudinale dans chacun de ses angles : en bas, ce sont les barres « de » et
    #    « a » du lit 1 inférieur (par construction) ; en haut, il faut une barre du lit 1 supérieur au contact de chaque
    #    montant (à 5 mm près). Un cadre intérieur sous une face supérieure de deux barres n'en a pas : barres de montage
    #    à prévoir (« À EXAMINER »).
    ds = float(s["sup"][0]["d"])
    rs = ds / 20.0
    for k, (e, c) in enumerate(zip(s["etriers"], cadres)):
        if e["type"] == "epingle":
            # une épingle s'accroche à une barre basse (la sienne) ET à une barre haute à son droit
            if not any(abs(xb - c["x0"]) <= rs + 0.5 for xb in xs["sup"][0]):
                ajoute(f"epingle:{k}", "epingle", None, None,
                       f"Épingle C{k + 1} : sans barre haute à son droit (barre de montage à prévoir)", None, None, WARN,
                       empreinte=f"{c['x0']:.1f}|" + " ".join(f"{x:.1f}" for x in xs["sup"][0]))
            continue
        de_cm = float(e["d"]) / 10.0
        attendus = (("gauche", c["x0"] + de_cm + rs), ("droit", c["x1"] - de_cm - rs))
        manque = [(cote, x) for cote, x in attendus if not any(abs(xb - x) <= rs + 0.5 for xb in xs["sup"][0])]
        if manque:
            quoi = " et ".join(cote for cote, _ in manque)
            ajoute(f"angles_cadre:{k}", "angles_cadre", None, None,
                   f"Cadre C{k + 1} : angle{'s' if len(manque) > 1 else ''} supérieur{'s' if len(manque) > 1 else ''} {quoi} "
                   f"sans barre (barres de montage à prévoir)", None, None, WARN,
                   empreinte=f"{c['x0']:.1f}-{c['x1']:.1f}|" + " ".join(f"{x:.1f}" for _, x in manque))
    # 9. une position hors des barres du lit 1 : signalée (le dessin la ramène dans [1 ; n₁], comme le module)
    n1 = int(s["inf"][0]["n"])
    for k, e in enumerate(s["etriers"]):
        for cle in ("de", "a"):
            if e.get(cle) is not None and int(e[cle]) > n1:
                ajoute(f"position_cadre:{k}:{cle}", "position_cadre", None, None,
                       f"Ligne {k + 1} : barre {int(e[cle])} inexistante (le lit 1 inférieur en compte {n1})", None, None,
                       WARN, empreinte=f"{int(e[cle])}/{n1}")
    # les justifications de l'ingénieur : une observation « à examiner » seulement, et pour l'empreinte qu'il a vue
    justifs = s.get("justifs") or {}
    for it in items:
        it["etat_eff"] = it["etat"]
        j = justifs.get(it["id"])
        if j is None or it["etat"] != WARN:
            continue
        if j.get("empreinte") == it["empreinte"]:
            it["justification"] = j.get("texte", "")
            it["etat_eff"] = JUST
        else:
            it["justification_perimee"] = j.get("texte", "")
    etat = max((it["etat_eff"] for it in items), key=lambda e: RANG[e], default=OK)
    etat = OK if etat == JUST else etat
    compte = {k: sum(1 for it in items if it["etat_eff"] == k) for k in (OK, WARN, NOK, JUST)}
    pire = None
    for it in items:
        if RANG[it["etat_eff"]] == 0 and it["libre"] is None:
            continue
        marge = (it["libre"] - (it["mini"] or 0.0)) if it["libre"] is not None else -1e9
        cle = (RANG[it["etat_eff"]], -marge)
        if pire is None or cle > pire[0]:
            pire = (cle, it)
    return {"etat": etat, "dg": dg, "s_min_ref": s_min(20.0, 20.0, dg), "items": items, "compte": compte,
            "gouvernant": pire[1] if pire else None, "abscisses": xs, "laterales": lat, "cadres": cadres}
