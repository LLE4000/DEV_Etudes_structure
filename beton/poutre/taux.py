# -*- coding: utf-8 -*-
"""Poutre en béton armé — le TAUX D'ARMATURE SIMPLIFIÉ : la masse d'acier au MÈTRE COURANT de poutre, section par
section, et le taux de la poutre entière quand ses sections ont une longueur.

C'est l'option « Taux d'armature » de l'écran Streamlit (``modules/poutre.py`` v2.39+ : ``_taux_armature_section``,
``_section_poids_vol``, ``_beam_totals``), extraite ici et branchée sur la GÉOMÉTRIE que le moteur du site calcule déjà
pour le dessin (``moteur._geometrie`` : les lits, l'emprise de chaque cadre, les armatures latérales). Sur une poutre que
les deux écrans savent décrire (lits répartis, cadres, épingles sur une barre, armatures de peau automatiques), les deux
calculs donnent les mêmes nombres — ``tests/test_poutre_moteur.py`` rejoue le module et le prouve.

Ce que le calcul compte, au mètre courant :

    barres des lits      n barres de 1 m                                   masse = n · m_l(Ø)
    cadre                100 / pas cours par mètre ; un cadre mesure       l = 2·(w + h − 2c) + 2·retour
                         (w : sa largeur hors tout — b − 2c quand il entoure toute la largeur, sinon l'emprise des
                         barres entourées, étrier compris)
    épingle              100 / pas par mètre ; une épingle mesure          l = (h − 2c) + 2·retour
    latérales (peau)     2·n barres de 1 m (n par face)
    masse linéique       m_l(Ø) = ρ_acier · π·Ø²/4                          ρ_acier = 7 850 kg/m³
    taux brut            masse · (1 + majoration) / (b · h · 1 m)           kg/m³
    taux                 le taux brut arrondi au multiple SUPÉRIEUR de « arrondi »

    poutre               Σ (masse majorée_i · l_i) / Σ (b·h · l_i), sur les sections qui ont une longueur l_i

Ce qu'il ne compte PAS — c'est un ratio d'avant-métré, pas un bordereau : les ancrages, les recouvrements, les arrêts de
barres, les abouts, les crochets des barres longitudinales. La majoration les couvre forfaitairement ; le module « Taux
d'armature » du site les détaille (``beton/taux``).

Unités : Ø en mm, longueurs de section en cm (b, h, c, retour, pas), longueurs de barre et de poutre en m, masses en kg.
"""
import math

RHO_ACIER = 7850.0   # kg/m³ — masse volumique de l'acier des armatures (modules/poutre.py RHO_ACIER ; EN 10080 : 7,85 kg/dm³)

# les réglages de l'option (« Paramètres avancés » de l'écran Streamlit), et leurs bornes
DEFAUTS = {"majoration": 5.0,   # % — chutes, ligatures, recouvrements : « taux_arm_major_pct »
           "retour": 10.0,      # cm — retour d'un étrier ou d'une épingle, compté DEUX fois : « taux_retour_etrier_cm »
           "arrondi": 5}        # kg/m³ — le taux s'arrondit au multiple supérieur : « taux_arrondi_kgm3 »
BORNES = {"majoration": (0.0, 50.0), "retour": (0.0, 50.0), "arrondi": (1, 50)}
LONGUEUR_MAX = 200.0            # m — la longueur d'une section (une faute de frappe en cm se voit)
EPS = 1e-9                      # tolérance des arrondis supérieurs (100,000 000 1 ne monte pas à 105)


def masse_lineique(d_mm: float) -> float:
    """Masse d'un mètre de barre (kg/m) : ρ·π·Ø²/4 — ``_masse_lin_kg_m``."""
    return RHO_ACIER * math.pi * (d_mm / 1000.0) ** 2 / 4.0


def arrondi_superieur(x: float, pas: float) -> float:
    """x arrondi au multiple supérieur de ``pas`` (0 reste 0) — ``math.ceil(ta / arrondi) * arrondi`` du module, à EPS
    près pour qu'un multiple exact ne monte pas d'un cran sur une poussière de flottant."""
    if x <= 0:
        return 0.0
    return math.ceil(x / pas - EPS) * pas


def reglages(taux) -> dict:
    """Les réglages complets de l'option : ceux de la poutre (clé « taux », facultative), sinon les défauts."""
    r = dict(DEFAUTS)
    for k in DEFAUTS:
        if taux and taux.get(k) is not None:
            r[k] = taux[k]
    return r


def _ligne(cle, position, famille, n, d, l_barre, formule=None) -> dict:
    """Une ligne du tableau : ``n`` barres (ou cours par mètre) de Ø ``d``, de ``l_barre`` m chacune."""
    ml = masse_lineique(d)
    l_tot = n * l_barre
    out = {"cle": cle, "position": position, "famille": famille, "n": n, "d": d, "aire": math.pi * (d / 2.0) ** 2,
           "masse_lineique": ml, "l_barre": l_barre, "l_totale": l_tot, "masse": l_tot * ml}
    if formule:
        out["formule"] = formule
    return out


def section(geo: dict, taux=None) -> dict:
    """Le taux d'armature d'UNE section, au mètre courant. ``geo`` : la « geometrie » de la section calculée par le
    moteur (``moteur.verifier(...)["sections"][i]["geometrie"]``) ; ``taux`` : les réglages (clé « taux » de la poutre).

    Rend {"lignes": [{"cle", "position", "famille", "n", "d", "aire", "masse_lineique", "l_barre", "l_totale", "masse",
    "part"[, "formule"]}], "masse" (kg/m, sans majoration), "majoration" (%), "masse_majoree" (kg/m), "volume" (m³/m),
    "taux_brut" (kg/m³, majoré, non arrondi), "arrondi", "taux" (kg/m³), "retour", "pas", "cours" (par mètre)}."""
    r = reglages(taux)
    b, h, c = float(geo["b"]), float(geo["h"]), float(geo["enrobage_calcul"])
    retour = float(r["retour"])
    lignes = []
    # ---- les barres longitudinales : 1 m par barre
    for face, nom in (("inf", "inf."), ("sup", "sup.")):
        for j, lit in enumerate(geo[face]["lits"], start=1):
            lignes.append(_ligne(f"{face}-{j}", f"Lit {j} {nom}", face, int(lit["n"]), float(lit["d"]), 1.0))
    # ---- les cadres et les épingles : 100 / pas cours par mètre
    pas = float(geo["etriers"]["pas"])
    cours = 100.0 / pas if pas > 0 else 0.0
    haut = h - 2.0 * c
    for k, cd in enumerate(geo["etriers"]["cadres"], start=1):
        d = float(cd["d"])
        if cd["type"] == "epingle":
            l_cm = haut + 2.0 * retour
            f = {"forme": "epingle", "hauteur": haut, "retour": retour, "l_cm": l_cm}
            nom = f"Épingle C{k}"
        else:
            w = float(cd["x1"]) - float(cd["x0"])
            l_cm = 2.0 * (w + haut) + 2.0 * retour
            f = {"forme": "cadre", "largeur": w, "hauteur": haut, "retour": retour, "l_cm": l_cm, "complet": bool(cd["complet"])}
            nom = f"Cadre C{k}"
        lignes.append(_ligne(f"cadre-{k}", nom, "etriers", cours, d, l_cm / 100.0, f))
    # ---- les armatures latérales (de peau) : n par face, sur les deux faces
    lat = geo.get("laterales") or {}
    if lat.get("n"):
        lignes.append(_ligne("lat", "Latérales", "laterales", 2 * int(lat["n"]), float(lat["d"]), 1.0))
    masse = sum(x["masse"] for x in lignes)
    for x in lignes:
        x["part"] = x["masse"] / masse if masse > 0 else 0.0
    volume = (b / 100.0) * (h / 100.0) * 1.0                       # m³ par mètre courant
    masse_maj = masse * (1.0 + float(r["majoration"]) / 100.0)
    brut = masse_maj / volume if volume > 0 else 0.0
    return {"lignes": lignes, "masse": masse, "majoration": float(r["majoration"]), "masse_majoree": masse_maj,
            "volume": volume, "taux_brut": brut, "arrondi": r["arrondi"], "taux": arrondi_superieur(brut, r["arrondi"]),
            "retour": retour, "pas": pas, "cours": cours}


def poutre(sections: list, longueurs: list, taux=None):
    """Le taux d'armature de la POUTRE : Σ(masse majorée_i · l_i) / Σ(volume_i · l_i) sur les sections qui ont une
    longueur (m). ``sections`` : les résultats de ``section`` ; ``longueurs`` : la longueur de chaque section, ou None.

    Rend None quand aucune section n'a de longueur ; sinon {"longueur" (m), "masse" (kg, majorée), "volume" (m³),
    "taux_brut", "taux" (arrondi au multiple supérieur), "comptees" (le nombre de sections comptées), "sans_longueur"
    (les rangs des sections laissées de côté)} — ``_beam_totals``."""
    r = reglages(taux)
    l_tot = masse = volume = 0.0
    comptees, sans = 0, []
    for i, (s, lg) in enumerate(zip(sections, longueurs)):
        if lg is None or float(lg) <= 0:
            sans.append(i)
            continue
        lg = float(lg)
        comptees += 1
        l_tot += lg
        masse += s["masse_majoree"] * lg
        volume += s["volume"] * lg
    if not comptees or volume <= 0:
        return None
    brut = masse / volume
    return {"longueur": l_tot, "masse": masse, "volume": volume, "taux_brut": brut,
            "taux": arrondi_superieur(brut, r["arrondi"]), "comptees": comptees, "sans_longueur": sans}


def calculer(p: dict, sections_calculees: list):
    """Le taux d'armature d'une poutre NORMALISÉE dont l'option est allumée (``p["taux"]["actif"]``) ; None sinon.
    ``sections_calculees`` : ``moteur.verifier(p)["sections"]`` (leur « geometrie » est lue).

    Rend {"reglages", "sections": [<section> + {"nom", "longueur"}], "poutre": <poutre> | None}."""
    t = p.get("taux")
    if not t or not t.get("actif"):
        return None
    out = []
    for s, sc in zip(p["sections"], sections_calculees):
        x = section(sc["geometrie"], t)
        x["nom"] = s["nom"]
        x["longueur"] = s.get("longueur")
        out.append(x)
    return {"reglages": reglages(t), "sections": out, "poutre": poutre(out, [x["longueur"] for x in out], t)}
