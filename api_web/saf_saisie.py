# -*- coding: utf-8 -*-
"""La SAISIE du « Générateur SAF — Fondation de grue » (api_web/saf.py) : la colonne de gauche, en QUATRE blocs comme le
module « Poutres béton armé » (lot 5, § 4) — 1 Dalle et sol, 2 Grue et charges, 3 Ferraillage, 4 Options —, chacun une
suite de LIGNES (un libellé court à gauche, les champs à droite, plusieurs champs sur une ligne quand ils vont ensemble),
SANS sous-titre. Le nom du projet et l'indice sont ceux de la barre du haut, les dates sont automatiques ; le nom de la
grue est dans le bloc 2.

L'unité des contraintes de sol (Options) : kg/cm² par défaut, kPa ou MPa — la même PARTOUT (le champ de la limite, les
résultats, la carte, la note) ; le calcul reste en kPa. ``unite``, ``sol`` et ``sol_nb`` l'écrivent.

Peu de champs, des libellés courts, les unités dans les champs, aucune phrase d'explication : ce qui demande un mot se
lit au survol de ⓘ. Un champ vide reste vide. Ce fichier ne calcule rien : il lit le projet, le châssis
(``saf.fondation_grue.chassis.lire``) et la synthèse du calcul (``saf_resultat.synthese``).
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
# l'unité des contraintes de sol (§ 2.4) : (symbole, kPa par unité, décimales) — 2 en kg/cm² (1,12), 1 en kPa (110,3), 3 en
# MPa (0,110) ; 1 kg/cm² = 98,0665 kPa, 1 MPa = 1 000 kPa
UNITES_SOL = {"kgcm2": ("kg/cm²", V.KG_CM2, 2), "kpa": ("kPa", 1.0, 1), "mpa": ("MPa", 1000.0, 3)}
UNITES_TXT = (("kgcm2", "kg/cm²"), ("kpa", "kPa"), ("mpa", "MPa"))


def unite(el: Dict):
    """(symbole, kPa par unité, décimales) de l'unité des contraintes de sol choisie."""
    return UNITES_SOL[el["options"]["unite"]]


def sol_nb(el: Dict, kpa: float) -> str:
    """Une contrainte de sol (en kPa) écrite dans l'unité choisie, sans l'unité : « 1,12 »."""
    sym, div, nd = unite(el)
    return C.fr(kpa / div, nd)


def sol(el: Dict, kpa: float) -> str:
    """Une contrainte de sol (en kPa) écrite dans l'unité choisie : « 1,12 kg/cm² »."""
    return f"{sol_nb(el, kpa)}{C.NBSP}{unite(el)[0]}"


def limite_kpa(el: Dict) -> float:
    """La limite de contrainte de contact en kPa (gardée en kg/cm² dans le projet)."""
    return el["limite"] * V.KG_CM2


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


def _limite(el: Dict) -> str:
    """La limite de contrainte de contact, dans l'unité choisie (§ 2.4) — gardée en kg/cm² dans le projet."""
    sym, div, nd = unite(el)
    return C.ligne("Limite", C.num(["limite"], round(limite_kpa(el) / div, nd + 3), sym, mini=0.0), full=True,
                   aide="La contrainte de contact admise (σ). Le calcul est en kPa ; l'unité se choisit dans Options.")


def _dalle(el: Dict) -> str:
    """Les lignes Forme, Dimensions, Béton du bloc 1 (§ 4)."""
    d = el["dalle"]
    out = C.ligne("Forme", C.segments(["dalle", "forme"], d["forme"], [("rectangle", "Rectangle"), ("contour", "Contour")]), full=True)
    if d["forme"] == "rectangle":
        out += C.ligne2(("L<sub>x</sub>", C.num(["dalle", "Lx"], d["Lx"], "m", mini=1, maxi=60), ""),
                        ("L<sub>y</sub>", C.num(["dalle", "Ly"], d["Ly"], "m", mini=1, maxi=60), ""),
                        ("h", C.num(["dalle", "h"], d["h"], "m", mini=0.1, maxi=3), "Épaisseur de la dalle."))
    else:
        seul = len(d["sommets"]) <= 3
        lignes = "".join(f'<div class="sf-som"><span class="n">{i + 1}</span>{C.num(["dalle", "sommets", i, "0"], p[0], "m")}'
                         f'{C.num(["dalle", "sommets", i, "1"], p[1], "m")}{"" if seul else _x(["dalle", "sommets", i, "-"], f"Retirer le sommet {i + 1}")}</div>'
                         for i, p in enumerate(d["sommets"]))
        out += (f'<div class="sf-tab"><div class="sf-som sf-som-h"><span class="n"></span><span>x</span><span>y</span></div>{lignes}</div>'
                + E.plus(["dalle", "sommets", "+"], "1", "Sommet")
                + C.ligne("h", C.num(["dalle", "h"], d["h"], "m", mini=0.1, maxi=3), full=True, aide="Épaisseur de la dalle."))
    out += C.ligne("Béton", C.choix(["dalle", "beton"], d["beton"], [(b, b) for b in PR.BETONS]), full=True)
    return out


def _sol(el: Dict) -> str:
    """La ligne Sol du bloc 1 (§ 4) : les valeurs de k en pastilles, chacune avec sa case « SCIA » ; « Reprendre »."""
    ks = el["sol"]["k"]
    seul = len(ks) <= 1
    pastilles = "".join(f'<div class="sf-kp{" on" if x["scia"] else ""}">{C.num(["sol", "k", i, "k"], x["k"], "")}'
                        f'<label class="sf-kc" title="Coché : ce k entre dans l\'enveloppe ; le SAF porte le premier coché"><input type="checkbox" '
                        f'data-chemin="sol/k/{i}/scia" data-sorte="case"{" checked" if x["scia"] else ""}><span>SCIA</span></label>'
                        f'{"" if seul else _x(["sol", "k", i, "-"], "Retirer cette valeur de k")}</div>' for i, x in enumerate(ks))
    plus = (f'<button type="button" class="sf-kplus" data-chemin="sol/k/+" data-valeur="1" title="Une valeur de k de plus">{C.icon("plus")}</button>'
            if len(ks) < PR.K_MAX_VALEURS else "")
    reprendre = C.esc(json.dumps({"module": "sol", "chemin": "sol/reprise", "titre": "Reprendre une note de raideur du sol",
                                  "vide": "Aucune note « Rigidité du sol » dans ce navigateur."}, ensure_ascii=False))
    out = (f'<div class="row full sf-sol"><label><span>Sol</span><span class="aide" title="k en MN/m³ ; « SCIA » : le k entre dans l\'enveloppe. '
           f'« Reprendre » : les k d\'une note Rigidité du sol (sa page est jointe à la note).">{C.icon("info")}</span></label>'
           f'<div class="sf-kps">{pastilles}{plus}<button type="button" class="btn sm sf-repr" data-reprendre="{reprendre}" '
           f'title="Reprendre les valeurs de k d\'une note « Rigidité du sol »">{C.icon("download")}Reprendre</button></div></div>')
    r = el["sol"]["reprise"]
    if r:
        page = r.get("page") or {}
        out += (f'<div class="sf-repris"><span>Repris de <b>{C.esc(r.get("note") or "sans nom")}</b>'
                + (f' · {C.esc(page.get("fondation") or "")}' if page.get("fondation") else "") + '</span>'
                + _x(["sol", "reprise", "-"], "Détacher cette note (les valeurs de k restent)") + '</div>')
    return out


def _grue(el: Dict, m: Dict, carnet: List[Dict]) -> str:
    """Les lignes Grue, Châssis, Appuis, Position, Plots du bloc 2 (§ 4)."""
    g, ch = el["grue"], m["chassis"]
    L, L2 = C.ligne, C.ligne2
    nom = (ch["nom"] if ch["constructeur"] and ch["constructeur"].lower() in ch["nom"].lower() else f"{ch['constructeur']} {ch['nom']}".strip()) \
        if ch["id"] else SAISIE_TXT
    garder = (f'<button type="button" class="btn sm sf-garder" data-chemin="carnet/garder" data-valeur="1" title="Enregistrer le châssis, les appuis, '
              f'les plots et les réactions de cette grue, pour un autre projet">{C.icon("plus")}Enregistrer</button>')
    out = f'<div class="row full sf-l-grue"><label><span>Grue</span></label><div class="sf-grue-l">{C.texte(["projet", "grue"], el["projet"]["grue"], "nom")}{garder}</div></div>'
    if carnet:
        puces = "".join(f'<span class="sf-grue"><button type="button" class="sf-grue-b" data-chemin="carnet/prendre" data-valeur="{i}" '
                        f'title="Reprendre cette grue : son châssis, ses appuis, ses plots et ses réactions">{C.esc(x["nom"])}</button>'
                        f'{_x(["carnet", i, "-"], "Retirer cette grue des grues enregistrées")}</span>' for i, x in enumerate(carnet))
        out += f'<div class="row full"><label><span>Enregistrées</span></label><div class="sf-grues">{puces}</div></div>'
    out += L("Châssis", f'<button type="button" class="prof sf-chassis-b" data-pk="chassis" title="Choisir le châssis dans la bibliothèque">'
                       f'<span>{C.esc(nom)}</span>{C.icon("chevron-down")}</button>', full=True,
             aide=ch["note_fondation"] or "")
    if ch["id"]:
        out += f'<div class="sf-lu"><span>a × b</span><b>{C.fr(ch["a"], 2)} × {C.fr(ch["b"], 2)}{C.NBSP}m</b></div>'
    else:
        out += L2(("a", C.num(["grue", "a"], g["a"], "m", mini=1, maxi=20), "Entraxe des pieds en x."), ("b", C.num(["grue", "b"], g["b"], "m", mini=1, maxi=20), "Entraxe en y."))
    reels_possible = bool(ch["id"]) and not ch["impose"]
    pastille = (f'<span class="sf-pastille" title="La bibliothèque ne donne pas la platine de ce châssis : les appuis sont un rectangle, dont les '
                f'dimensions sont saisies.">{C.icon("triangle-alert")}platine non indiquée</span>' if ch["impose"] else "")
    out += L("Appuis", C.segments(["grue", "appuis"], ch["appuis"], [("reels", "Réels"), ("rectangle", "Rectangle")], desactive=not reels_possible)
             + pastille, full=True)
    if ch["appuis"] == "rectangle":
        angle = ch["angle"]
        out += L2(("a", C.num(["grue", "rect", "a"], g["rect"]["a"], "m", mini=0.05, maxi=3), "À 0° : a selon x, b selon y. À 45° : a en travers du bras, b le long du bras."),
                  ("b", C.num(["grue", "rect", "b"], g["rect"]["b"], "m", mini=0.05, maxi=3), ""),
                  ("∠", C.segments(["grue", "rect", "angle"], str(angle), [("0", "0°"), ("45", "45°")]), "Orientation de la platine."))
    out += f'<div class="sf-lu"><span>Surface d\'appui</span><b>{surface_appui(ch)}</b></div>'
    # la position, l'angle
    out += L2(("e<sub>x</sub>", C.num(["grue", "ex"], g["ex"], "m"), "Position du centre du châssis par rapport aux axes de la dalle. Sur le plan, toucher un "
                                                                      "bord de la dalle le prend pour repère des cotes."),
              ("e<sub>y</sub>", C.num(["grue", "ey"], g["ey"], "m"), ""),
              ("β", C.num(["grue", "beta"], g["beta"], "°", mini=-360, maxi=360), "Angle du châssis."))
    cb = PR.cotes_au_bord(el, m)
    if cb is not None:
        out += (f'<div class="sf-repris"><span>Cotes au <b>bord {cb["bord"] + 1}</b> de la dalle</span>'
                + _x(["grue", "repere"], "Revenir aux axes de la dalle") + '</div>'
                + L2(("au bord", C.num(["grue", "cote", "perp"], round(cb["perp"], 3), "m"), "Distance du centre du châssis au bord pris pour repère."),
                     ("le long", C.num(["grue", "cote", "long"], round(cb["long"], 3), "m"), "Du début du bord au pied de la perpendiculaire.")))
    # les plots
    p = g["plots"]
    out += L("Plots", C.segments(["grue", "plots", "type"], p["type"], [(k, t, PLOTS_LONGS[k].capitalize()) for k, t in PLOTS_TXT]), full=True,
             aide="La réaction arrive sur la dalle par un carré de diffusion à 45° depuis une plaque de 0,50 × 0,50 (0,50 + 2 × hauteur, "
                  "sans dépasser le plot). Coulés en place : la dalle est plus épaisse sous le plot.")
    if p["type"] != "aucun":
        out += C.ligne2(("a", C.num(["grue", "plots", "a"], p["a"], "m", mini=0.2, maxi=10), ""), ("b", C.num(["grue", "plots", "b"], p["b"], "m", mini=0.2, maxi=10), ""),
                        ("h", C.num(["grue", "plots", "h"], p["h"], "m", mini=0.05, maxi=2), "Hauteur du plot."))
    return out


def _reactions(el: Dict) -> str:
    """La ligne Réactions du bloc 2 (§ 4) : un petit tableau, une ligne par jeu (nom, P1 à P4 en kN), « + jeu » ; un
    tableau copié dans Excel se colle dans un champ (data-coller : une ligne par jeu, à partir de celui-ci)."""
    jeux = el["reactions"]
    lignes = ""
    for i, j in enumerate(jeux):
        champs = "".join(C.num(["reactions", i, "P", p], j["P"][p], "kN", auto=f"P{p + 1}").replace(
            "<input ", f'<input data-coller="reactions/coller/{i}/{p}" ', 1) for p in range(4))
        lignes += (f'<div class="sf-jeu"><span class="n">{i + 1}</span>{C.texte(["reactions", i, "nom"], j["nom"], "nom")}'
                   f'{_x(["reactions", i, "-"], f"Retirer le jeu {i + 1}")}<div class="sf-p">{champs}</div></div>')
    entetes = '<div class="sf-jeu sf-jeu-h"><span class="n"></span><div class="sf-p">' + "".join(f"<span>P{p + 1}</span>" for p in range(4)) + "</div></div>"
    plus = E.plus(["reactions", "+"], "1", "jeu") if len(jeux) < PR.JEUX_MAX else ""
    return (f'<div class="row full sf-l-reac"><label><span>Réactions</span><span class="aide" title="En kN, grue non tournée ; les charges tournent '
            f'dans toutes les positions. Un tableau copié dans Excel (P1 à P4, un jeu par ligne) se colle dans un champ.">{C.icon("info")}</span></label>'
            f'{DS.numerotation()}</div><div class="sf-tab">{entetes}{lignes}</div>{plus}')


def _treillis(el: Dict, face: str) -> str:
    """La ligne Inférieur ou Supérieur du bloc 3 (§ 4) : Ø · e, les mêmes en x et en y ; « x ≠ y » ouvre une deuxième
    ligne ; en haut, le bouton rapide « e 10 ». Les listes sont celles du module Dalles."""
    t = el["armatures"][face]
    nom = "Inférieur" if face == "inf" else "Supérieur"
    d_ = [(d, f"Ø{d}") for d in PR.DIAMETRES_TREILLIS]
    e_ = [(e, f"e {nb_cm(e)}") for e in PR.ESPACEMENTS_TREILLIS]

    def champs(s):
        return (C.choix(["armatures", face, s, "d"], t[s]["d"], d_) + C.choix(["armatures", face, s, "e"], t[s]["e"], e_))
    rapide = (f'<button type="button" class="btn sm sf-e10" data-chemin="armatures/{face}/e10" data-valeur="1" title="Ø{t["x"]["d"]} tous les 10 cm, '
              f'en x et en y">e 10</button>') if face == "sup" else ""
    xy = C.case(["armatures", face, "xy"], t["xy"], "x ≠ y")
    out = (f'<div class="row full sf-l-tr"><label><span>{nom}</span></label><div class="sf-tr">{champs("x")}{rapide}{xy}</div></div>')
    if t["xy"]:
        out += f'<div class="row full sf-l-tr"><label><span>{nom} y</span></label><div class="sf-tr">{champs("y")}</div></div>'
    return out


def nb_cm(e_mm: int) -> str:
    return C.nb(e_mm / 10)


def _ferraillage(el: Dict, m: Dict, sc: Optional[Dict], ef: Optional[Dict], fer: Optional[Dict]) -> str:
    """Le bloc 3 (§ 4) : l'enrobage, les treillis inférieur et supérieur, la largeur des bandes, le Ø des renforts, le
    tableau des efforts SCIA et son aide."""
    from . import saf_armatures as SA
    a = el["armatures"]
    L = C.ligne
    rf = a["renfort"]
    out = (L("Enrobage", C.num(["armatures", "enrobage"], round(a["enrobage"] * 100, 2), "cm", mini=1, maxi=15), full=True)
           + _treillis(el, "inf") + _treillis(el, "sup")
           + L("Bandes", C.segments(["bandes", "largeur"], f"{el['bandes']['largeur']:.1f}", [("0.5", "0,50 m"), ("1.0", "1,00 m")]), full=True,
               aide="La largeur des bandes sous les pieds (BX1, BX2, BY1, BY2), dans le module et dans SCIA.")
           + L("Renforts", C.segments(["armatures", "renfort"], str(rf), [("auto", "Auto", "Ø automatique : 12, 16 ou 20"), ("12", "Ø12"), ("16", "Ø16"), ("20", "Ø20")]), full=True,
               aide="Automatique : le plus léger de Ø12, Ø16 et Ø20 (à poids égal, le plus gros)."))
    out += (f'<div class="row full sf-l-scia"><label><span>Efforts SCIA</span><span class="aide" title="Après le calcul SCIA : la valeur du module est en gris, '
            f'on tape la valeur SCIA par-dessus. SCIA donne le niveau, le module la forme.">{C.icon("info")}</span></label></div>'
            f'<div class="sf-scia-t">{SA.tableau_scia(el, sc, ef, unite(el))}</div>{SA.aide_scia(el, m, ef, fer)}')
    return out


def _options(el: Dict) -> str:
    """Le bloc 4 (§ 4) : Unité, Maille, Anneau, Bord, Sol, Calcul, Ancrage, Proposer — une ligne chacune."""
    o, g, a = el["options"], el["grue"], el["armatures"]
    L, L2 = C.ligne, C.ligne2
    return (L("Unité", C.segments(["options", "unite"], o["unite"], list(UNITES_TXT)), full=True,
              aide="L'unité des contraintes de sol, partout : carte, résultats, tableau par k, note. Le calcul reste en kPa.")
            + L("Maille", C.num(["options", "maille"], o["maille"], "m", mini=PR.MAILLES[0], maxi=PR.MAILLES[1]), full=True)
            + L("Anneau", C.segments(["options", "anneau"], f"{o['anneau']:.2f}", [("0.01", "1 cm"), ("0.02", "2 cm"), ("0.03", "3 cm")]), full=True,
                aide="L'anneau D2 qui entoure la dalle dans le SAF (avec l'ouverture O1).")
            + L("Bord", C.num(["grue", "dmin"], g["dmin"], "m", mini=0, maxi=5), full=True, aide="Distance minimale entre une surface d'appui et le bord de la dalle.")
            + L2(("C<sub>1x</sub>", C.num(["options", "c1x"], o["c1x"], "k", mini=0, maxi=1), "Sol dans le SAF : C1x = k × cette valeur (k/10 par défaut)."),
                 ("C<sub>1y</sub>", C.num(["options", "c1y"], o["c1y"], "k", mini=0, maxi=1), ""),
                 ("C<sub>2x</sub>", C.num(["options", "c2x"], o["c2x"], "k", mini=0, maxi=1), "Pasternak, dans le SAF et dans le calcul : C2x = k × cette valeur (MN/m)."),
                 ("C<sub>2y</sub>", C.num(["options", "c2y"], o["c2y"], "k", mini=0, maxi=1), ""))
            + f'<div class="row full sf-l-calc"><label><span>Calcul</span></label><div class="sf-cases">'
            + C.case(["options", "rond"], o["rond"] == "48", "platine ronde en 48 côtés")
            + C.case(["options", "aretes"], o["aretes"], "arêtes internes") + C.case(["options", "nonlinear"], o["nonlinear"], "combinaisons Nonlinear")
            + '</div></div>'
            + L("Ancrage", C.num(["armatures", "ancrage"], a["ancrage"], "Ø", mini=10, maxi=120), full=True,
                aide="Le moteur Dalles n'a pas de règle d'ancrage : la longueur d'ancrage des renforts vaut n × Ø (50 Ø par défaut).")
            + L2(("Proposer", C.num(["options", "h_min"], o["h_min"], "m", mini=0.1, maxi=3), "Les épaisseurs testées par « Proposer » : de, à, pas."),
                 ("à", C.num(["options", "h_max"], o["h_max"], "m", mini=0.1, maxi=3), ""),
                 ("pas", C.num(["options", "h_pas"], o["h_pas"], "m", mini=0.05, maxi=1), "")))


def formulaire(el: Dict, calc: Dict, s: Dict, carnet: List[Dict], sc: Optional[Dict] = None, ef: Optional[Dict] = None,
               fer: Optional[Dict] = None) -> str:
    """Les quatre blocs de la fondation ouverte (§ 4) — sans sous-titre ; un résumé court à droite de chaque titre. ``s`` :
    la synthèse du calcul (``saf_resultat.synthese``) ; ``sc``, ``ef``, ``fer`` : le tableau SCIA, les efforts, le principe
    d'armatures (les valeurs grises et l'aide du bloc 3)."""
    m = calc["modele"]
    ch = m["chassis"]
    ks = el["sol"]["k"]
    n_jeux = len(m["jeux"])
    o, a = el["options"], el["armatures"]
    scia = [C.nb(x["k"]) for x in ks if x["scia"]]
    r1 = f"{dimensions(el)} · {C.esc(el['dalle']['beton'])} · k {' · '.join(scia) if scia else 'aucun SCIA'} · {sol(el, limite_kpa(el))}"
    appuis = ch["dims"].replace(" m", "") if ch["appuis"] == "rectangle" else "réels"
    plots = {"aucun": "sans plot", "prefabrique": "plots préfab.", "coule": "plots coulés"}[el["grue"]["plots"]["type"]]
    r2 = (f"châssis {C.fr(ch['a'], 2)} × {C.fr(ch['b'], 2)} · appuis {C.esc(appuis)} · "
          + (f"{n_jeux} jeu{'x' if n_jeux > 1 else ''}" if n_jeux else "réactions à saisir") + f" · {plots}")
    def tr(face):
        t = a[face]
        return f"Ø{t['x']['d']}{C.NBSP}e{nb_cm(t['x']['e'])}" + ("" if t["x"] == t["y"] else "…")
    etat_scia = "SCIA saisi" if sc and sc["complet"] else ("SCIA en cours" if sc and sc["saisi"] else "SCIA à saisir")
    r3 = f"inf {tr('inf')} · sup {tr('sup')} · bandes {C.fr(el['bandes']['largeur'], 2)}{C.NBSP}m · {etat_scia}"
    r4 = f"{unite(el)[0]} · maille {C.fr(o['maille'], 2)}{C.NBSP}m · anneau {C.nb(o['anneau'] * 100)}{C.NBSP}cm"
    return (C.section(1, "Dalle et sol", r1, _dalle(el) + _sol(el) + _limite(el), True, "dalle")
            + C.section(2, "Grue et charges", r2, _grue(el, m, carnet) + _reactions(el), False, "grue")
            + C.section(3, "Ferraillage", r3, _ferraillage(el, m, sc, ef, fer), False, "armatures")
            + C.section(4, "Options", r4, _options(el), False, "options", secondaire=True))
