# -*- coding: utf-8 -*-
"""La SAISIE du « Générateur SAF — Fondation de grue » (api_web/saf.py) : la colonne de gauche, en rubriques — Projet,
Dalle, Sol, Grue, Réactions, Bandes, Limite — et un seul panneau « Options », replié.

Peu de champs, des libellés courts, les unités dans les champs, aucune phrase d'explication (cahier des charges § 1) :
ce qui demande un mot se lit au survol de ⓘ. Un champ vide reste vide. Ce fichier ne calcule rien : il lit le projet,
le châssis (``saf.fondation_grue.chassis.lire``) et la synthèse du calcul (``saf_resultat.synthese``).
"""
import json
from typing import Dict, List, Optional

from saf.fondation_grue import chassis as CH
from saf.fondation_grue import projet as PR
from saf.fondation_grue import verification as V

from . import commun as C
from . import dessin_saf as DS
from . import etabli_compact as E

SAISIE_TXT = "Saisie"            # le châssis hors bibliothèque, dans le sélecteur
# (« Préfabriqués » ne tenait pas dans son bouton à la largeur de la colonne — coupé, vu au harnais ; le mot entier au survol)
PLOTS_TXT = (("aucun", "Aucun"), ("prefabrique", "Préfab."), ("coule", "Coulés"))
PLOTS_LONGS = {"aucun": "aucun plot", "prefabrique": "plots préfabriqués", "coule": "plots coulés en place"}
ORIGINES = {"mesure": ("≈ ", "valeur mesurée sur le document du constructeur"), "calcule": ("calc. ", "valeur calculée")}


def phrase(texte: str) -> str:
    """Un refus du moteur, écrit comme une phrase : la majuscule, sans point double."""
    t = str(texte or "").strip().rstrip(".")
    return (t[:1].upper() + t[1:]) if t else "Saisie refusée"


def dimensions(el: Dict) -> str:
    """« 8,00 × 8,00 × 0,50 m » ; un contour : « contour 6 sommets · 0,50 m » (HTML)."""
    d = el["dalle"]
    if d["forme"] == "rectangle":
        return f"{C.fr(d['Lx'], 2)} × {C.fr(d['Ly'], 2)} × {C.fr(d['h'], 2)}{C.NBSP}m"
    return f"contour {len(d['sommets'])} sommets · {C.fr(d['h'], 2)}{C.NBSP}m"


def _x(chemin: List, titre: str) -> str:
    """Le petit bouton qui retire une ligne (un k, un jeu, un sommet)."""
    return (f'<button type="button" class="sf-x" data-chemin="{C._attr_chemin(chemin)}" data-valeur="1" title="{C.esc(titre)}" '
            f'aria-label="{C.esc(titre)}">{C.icon("x")}</button>')


def _origine(platine: Optional[Dict], champ: str, texte: str) -> str:
    """Une valeur de la bibliothèque avec son origine, comme dans la bibliothèque : « ≈ » en italique (mesurée),
    « calc. » (calculée)."""
    o = ORIGINES.get(((platine or {}).get("origines") or {}).get(champ))
    if not o:
        return texte
    return f'<i class="sf-orig" title="{C.esc(o[1])}">{o[0]}{texte}</i>' if o[0] == "≈ " else f'<span class="sf-orig" title="{C.esc(o[1])}">{o[0]}{texte}</span>'


def surface_appui(ch: Dict) -> str:
    """Ce qu'est la surface d'appui retenue, en une ligne (HTML) : sa forme, ses dimensions avec leur origine, son aire ;
    « saisie » quand elle l'est."""
    p = ch["platine"]
    if ch["appuis"] == "rectangle":
        return f'{C.esc(ch["dims"])} · {C.fr(ch["aire"], 4)}{C.NBSP}m² · <b class="sf-saisi">saisie</b>'
    if p["forme"] == "rectangle":
        dims = f'{_origine(p, "a", C.fr(p["a"], 2))} × {_origine(p, "b", C.fr(p["b"], 2))}{C.NBSP}m' + (" à 45°" if ch["angle"] == 45 else "")
    elif p["forme"] == "cercle":
        dims = f'Ø {_origine(p, "diametre", C.fr(p["diametre"], 2))}{C.NBSP}m' + (
            " en 48 côtés" if ch["forme"] == "cercle" else f' → carré {_origine(p, "cote_carre_meme_aire", C.fr(p["cote_carre"], 3))}{C.NBSP}m')
    else:
        dims = f'octogone, {_origine(p, "cote_sur_plats", C.fr(p["cote_sur_plats"], 2))}{C.NBSP}m sur plats'
    exc = f' · excentré de {_origine(p, "excentrement_radial", C.fr(p["excentrement"] * 1000, 0))}{C.NBSP}mm' if p["excentrement"] else ""
    return f'{dims}{exc} · {C.fr(ch["aire"], 4)}{C.NBSP}m²'


def _projet(el: Dict) -> str:
    p = el["projet"]
    L = C.ligne
    return (L("Projet", C.texte(["projet", "projet"], p["projet"]), full=True)
            + L("Grue", C.texte(["projet", "grue"], p["grue"]), full=True)
            + C.ligne2(("Création", C.texte(["projet", "creation"], p["creation"], "jj/mm/aaaa"), ""),
                       ("Mise à jour", C.texte(["projet", "mise_a_jour"], p["mise_a_jour"], "jj/mm/aaaa"), ""))
            + L("Indice", C.texte(["projet", "indice"], p["indice"])))


def _dalle(el: Dict) -> str:
    d = el["dalle"]
    out = C.ligne("Forme", C.segments(["dalle", "forme"], d["forme"], [("rectangle", "Rectangle"), ("contour", "Contour")]), full=True)
    if d["forme"] == "rectangle":
        out += C.ligne2(("L<sub>x</sub>", C.num(["dalle", "Lx"], d["Lx"], "m", mini=1, maxi=60), ""),
                        ("L<sub>y</sub>", C.num(["dalle", "Ly"], d["Ly"], "m", mini=1, maxi=60), ""))
    else:
        seul = len(d["sommets"]) <= 3
        lignes = "".join(f'<div class="sf-som"><span class="n">{i + 1}</span>{C.num(["dalle", "sommets", i, "0"], p[0], "m")}'
                         f'{C.num(["dalle", "sommets", i, "1"], p[1], "m")}{"" if seul else _x(["dalle", "sommets", i, "-"], f"Retirer le sommet {i + 1}")}</div>'
                         for i, p in enumerate(d["sommets"]))
        out += (f'<div class="sf-tab"><div class="sf-som sf-som-h"><span class="n"></span><span>x</span><span>y</span></div>{lignes}</div>'
                + E.plus(["dalle", "sommets", "+"], "1", "Sommet"))
    out += C.ligne2(("Épaisseur", C.num(["dalle", "h"], d["h"], "m", mini=0.1, maxi=3), ""),
                    ("Béton", C.choix(["dalle", "beton"], d["beton"], [(b, b) for b in PR.BETONS]), ""))
    return out


def _sol(el: Dict) -> str:
    ks = el["sol"]["k"]
    seul = len(ks) <= 1
    lignes = "".join(f'<div class="sf-k">{C.num(["sol", "k", i, "k"], x["k"], "MN/m³")}{C.case(["sol", "k", i, "scia"], x["scia"], "SCIA")}'
                     f'{"" if seul else _x(["sol", "k", i, "-"], "Retirer cette valeur de k")}</div>' for i, x in enumerate(ks))
    reprendre = C.esc(json.dumps({"module": "sol", "chemin": "sol/reprise", "titre": "Reprendre une note de raideur du sol",
                                                "vide": "Aucune note « Rigidité du sol » dans ce navigateur."}, ensure_ascii=False))
    out = (f'<div class="sf-tab">{lignes}</div><div class="sf-pied">{E.plus(["sol", "k", "+"], "1", "k") if len(ks) < PR.K_MAX_VALEURS else ""}'
           f'<button type="button" class="btn sm" data-reprendre="{reprendre}" title="Reprendre les valeurs de k d\'une note « Rigidité du sol » ; '
           f'sa page de résultats est jointe à la note en annexe">{C.icon("download")}Reprendre</button></div>')
    r = el["sol"]["reprise"]
    if r:
        page = r.get("page") or {}
        out += (f'<div class="sf-repris"><span>Repris de <b>{C.esc(r.get("note") or "sans nom")}</b>'
                + (f' · {C.esc(page.get("fondation") or "")}' if page.get("fondation") else "") + '</span>'
                + _x(["sol", "reprise", "-"], "Détacher cette note (les valeurs de k restent)") + '</div>')
    return out


def _grue(el: Dict, m: Dict, carnet: List[Dict]) -> str:
    g, ch = el["grue"], m["chassis"]
    L, L2 = C.ligne, C.ligne2
    nom = (ch["nom"] if ch["constructeur"] and ch["constructeur"].lower() in ch["nom"].lower() else f"{ch['constructeur']} {ch['nom']}".strip()) \
        if ch["id"] else SAISIE_TXT
    out = L("Châssis", f'<button type="button" class="prof sf-chassis-b" data-pk="chassis" title="Choisir le châssis dans la bibliothèque">'
                       f'<span>{C.esc(nom)}</span>{C.icon("chevron-down")}</button>', full=True)
    if ch["note_fondation"]:
        out += f'<div class="sf-note" title="Note de la bibliothèque sur la partie centrale du châssis">{C.esc(ch["note_fondation"])}</div>'
    if ch["id"]:
        out += f'<div class="sf-lu"><span>Entraxes</span><b>{C.fr(ch["a"], 2)} × {C.fr(ch["b"], 2)}{C.NBSP}m</b></div>'
    else:
        out += L2(("Entraxe a", C.num(["grue", "a"], g["a"], "m", mini=1, maxi=20), ""), ("b", C.num(["grue", "b"], g["b"], "m", mini=1, maxi=20), ""))
    # les appuis : réels (la bibliothèque) ou rectangle
    reels_possible = bool(ch["id"]) and not ch["impose"]
    pastille = (f'<span class="sf-pastille" title="La bibliothèque ne donne pas la platine de ce châssis : les appuis sont un rectangle, dont les '
                f'dimensions sont saisies.">{C.icon("triangle-alert")}platine non indiquée</span>' if ch["impose"] else "")
    out += L("Appuis", C.segments(["grue", "appuis"], ch["appuis"], [("reels", "Réels"), ("rectangle", "Rectangle")], desactive=not reels_possible)
             + pastille, full=True)
    if ch["appuis"] == "rectangle":
        angle = ch["angle"]
        out += (L2(("a", C.num(["grue", "rect", "a"], g["rect"]["a"], "m", mini=0.05, maxi=3), "À 0° : a selon x, b selon y. À 45° : a en travers du bras, b le long du bras."),
                   ("b", C.num(["grue", "rect", "b"], g["rect"]["b"], "m", mini=0.05, maxi=3), ""))
                + L("Orientation", C.segments(["grue", "rect", "angle"], str(angle), [("0", "0°"), ("45", "45°")]), full=True))
    out += f'<div class="sf-lu"><span>Surface d\'appui</span><b>{surface_appui(ch)}</b></div>'
    # les plots
    p = g["plots"]
    out += L("Plots", C.segments(["grue", "plots", "type"], p["type"], [(k, t, PLOTS_LONGS[k].capitalize()) for k, t in PLOTS_TXT]), full=True)
    if p["type"] != "aucun":
        out += C.ligne2(("a", C.num(["grue", "plots", "a"], p["a"], "m", mini=0.2, maxi=10), ""), ("b", C.num(["grue", "plots", "b"], p["b"], "m", mini=0.2, maxi=10), ""),
                        ("h", C.num(["grue", "plots", "h"], p["h"], "m", mini=0.05, maxi=2), "Hauteur du plot."))
    # la position, l'angle, la distance mini
    out += L2(("e<sub>x</sub>", C.num(["grue", "ex"], g["ex"], "m"), "Position du centre du châssis par rapport aux axes de la dalle. Sur le plan, toucher un "
                                                                      "bord de la dalle le prend pour repère des cotes."),
              ("e<sub>y</sub>", C.num(["grue", "ey"], g["ey"], "m"), ""))
    cb = PR.cotes_au_bord(el, m)
    if cb is not None:
        out += (f'<div class="sf-repris"><span>Cotes au <b>bord {cb["bord"] + 1}</b> de la dalle</span>'
                + _x(["grue", "repere"], "Revenir aux axes de la dalle") + '</div>'
                + L2(("au bord", C.num(["grue", "cote", "perp"], round(cb["perp"], 3), "m"), "Distance du centre du châssis au bord pris pour repère."),
                     ("le long", C.num(["grue", "cote", "long"], round(cb["long"], 3), "m"), "Du début du bord au pied de la perpendiculaire.")))
    out += (L2(("Angle β", C.num(["grue", "beta"], g["beta"], "°", mini=-360, maxi=360), ""),
               ("Bord mini", C.num(["grue", "dmin"], g["dmin"], "m", mini=0, maxi=5), "Distance minimale entre une surface d'appui et le bord de la dalle.")))
    # les grues enregistrées
    lignes = "".join(f'<div class="sf-grue"><button type="button" class="sf-grue-b" data-chemin="carnet/prendre" data-valeur="{i}" '
                     f'title="Reprendre cette grue : son châssis, ses appuis, ses plots et ses réactions">{C.esc(x["nom"])}</button>'
                     f'{_x(["carnet", i, "-"], "Retirer cette grue des grues enregistrées")}</div>' for i, x in enumerate(carnet))
    out += (f'<div class="grp">Grues enregistrées</div>{lignes}'
            f'<button type="button" class="et-add" data-chemin="carnet/garder" data-valeur="1" title="Enregistrer le châssis, les appuis, les plots et '
            f'les réactions de cette grue, pour un autre projet">{C.icon("plus")}Enregistrer cette grue</button>')
    return out


def _reactions(el: Dict) -> str:
    jeux = el["reactions"]
    lignes = ""
    for i, j in enumerate(jeux):
        champs = "".join(C.num(["reactions", i, "P", p], j["P"][p], "", auto=f"P{p + 1}") for p in range(4))
        lignes += (f'<div class="sf-jeu"><span class="n">{i + 1}</span>{C.texte(["reactions", i, "nom"], j["nom"], "nom")}'
                   f'{_x(["reactions", i, "-"], f"Retirer le jeu {i + 1}")}<div class="sf-p">{champs}</div></div>')
    entetes = '<div class="sf-jeu sf-jeu-h"><span class="n"></span><div class="sf-p">' + "".join(f"<span>P{p + 1}</span>" for p in range(4)) + "</div></div>"
    return (f'<div class="sf-num-l">{DS.numerotation()}<span>Réactions en kN,<br>grue non tournée</span></div><div class="sf-tab">{entetes}{lignes}</div>'
            + (E.plus(["reactions", "+"], "1", "Jeu") if len(jeux) < PR.JEUX_MAX else ""))


def _options(el: Dict) -> str:
    o = el["options"]
    L, L2 = C.ligne, C.ligne2
    return (L("Maille du calcul", C.num(["options", "maille"], o["maille"], "m", mini=PR.MAILLES[0], maxi=PR.MAILLES[1]))
            + L("Anneau D2", C.segments(["options", "anneau"], f"{o['anneau']:.2f}", [("0.01", "1 cm"), ("0.02", "2 cm"), ("0.03", "3 cm")]), full=True,
                aide="L'anneau qui entoure la dalle dans le SAF (avec l'ouverture O1). Il n'est pas dans la vérification rapide.")
            + L2(("C1x / k", C.num(["options", "c1x"], o["c1x"], "", mini=0, maxi=1), "Frottement du sol dans le SAF : C1x = k × cette valeur."),
                 ("C1y / k", C.num(["options", "c1y"], o["c1y"], "", mini=0, maxi=1), ""))
            + L2(("C2x / k", C.num(["options", "c2x"], o["c2x"], "", mini=0, maxi=1), "Pasternak, dans le SAF et dans la vérification rapide : C2x = k × cette valeur (MN/m)."),
                 ("C2y / k", C.num(["options", "c2y"], o["c2y"], "", mini=0, maxi=1), ""))
            + L("Platine ronde", C.segments(["options", "rond"], o["rond"], [("carre", "Carré de même aire"), ("48", "48 côtés")]), full=True)
            + C.case(["options", "aretes"], o["aretes"], "Arêtes internes des bandes")
            + C.case(["options", "nonlinear"], o["nonlinear"], "Combinaisons Nonlinear (SAF 2.2.0)")
            + '<div class="grp">Épaisseurs testées par « Proposer »</div>'
            + L2(("de", C.num(["options", "h_min"], o["h_min"], "m", mini=0.1, maxi=3), ""), ("à", C.num(["options", "h_max"], o["h_max"], "m", mini=0.1, maxi=3), ""),
                 ("pas", C.num(["options", "h_pas"], o["h_pas"], "m", mini=0.05, maxi=1), "")))


def formulaire(el: Dict, calc: Dict, s: Dict, carnet: List[Dict]) -> str:
    """Les rubriques de la fondation ouverte. ``s`` : la synthèse du calcul (``saf_resultat.synthese``)."""
    m = calc["modele"]
    ch, g, p = m["chassis"], el["grue"], el["projet"]
    ks = el["sol"]["k"]
    n_scia = sum(1 for x in ks if x["scia"])
    r_sol = " · ".join(C.nb(x["k"]) for x in ks) + f" · {n_scia} SCIA"
    r_grue = " · ".join(t for t in ((ch["nom"] or f"saisie {C.fr(ch['a'], 2)} × {C.fr(ch['b'], 2)}"),
                                    PLOTS_LONGS[g["plots"]["type"]] if g["plots"]["type"] != "aucun" else "",
                                    f"β {C.nb(round(g['beta'], 2), 2)}°" if abs(g["beta"]) > 1e-9 else "") if t)
    n_jeux = len(m["jeux"])
    r_reac = f"{n_jeux} jeu{'x' if n_jeux > 1 else ''} · {len(m['cas'])} cas" if n_jeux else "à saisir"
    limite_kpa = el["limite"] * V.KG_CM2
    o = el["options"]
    s7 = (C.ligne("Contrainte admise", C.num(["limite"], el["limite"], "kg/cm²", mini=0.05, maxi=5))
          + f'<div class="sf-lu"><span>soit</span><b>{C.fr(limite_kpa, 1)}{C.NBSP}kPa</b></div>')
    return (C.section(1, "Projet", C.esc(" · ".join(t for t in (p["projet"], p["grue"]) if t)), _projet(el), False, "projet")
            + C.section(2, "Dalle", f"{dimensions(el)} · {C.esc(el['dalle']['beton'])}", _dalle(el), True, "dalle")
            + C.section(3, "Sol", f"k {r_sol}", _sol(el), False, "sol")
            + C.section(4, "Grue", C.esc(r_grue), _grue(el, m, carnet), False, "grue")
            + C.section(5, "Réactions", r_reac, _reactions(el), False, "reactions")
            + C.section(6, "Bandes", f"{C.fr(el['bandes']['largeur'], 2)}{C.NBSP}m",
                        C.ligne("Largeur", C.segments(["bandes", "largeur"], f"{el['bandes']['largeur']:.1f}", [("1.0", "1,00 m"), ("0.5", "0,50 m")]), full=True),
                        False, "bandes")
            + C.section(7, "Limite", f"{C.nb(el['limite'])}{C.NBSP}kg/cm² · {C.fr(limite_kpa, 1)}{C.NBSP}kPa", s7, False, "limite")
            + C.section(8, "Options", f"maille {C.fr(o['maille'], 2)}{C.NBSP}m · anneau {C.nb(o['anneau'] * 100)}{C.NBSP}cm", _options(el), False, "options",
                        secondaire=True))
