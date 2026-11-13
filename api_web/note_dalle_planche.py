# -*- coding: utf-8 -*-
"""LE CONTENU d'une planche de la note PDF des dalles (lot DAL6 — demande du propriétaire du 13/11/2026, point 8 : « je
veux voir clairement le calcul détaillé de la hauteur minimale », « regrouper ces données dans un en-tête compact et
homogène, sans les répéter inutilement à chaque calcul », « les valeurs d'armatures dans la note doivent être des
résultats », « supprimer simplement le bloc Observations en bas à gauche »).

La note ne calcule rien : chaque chiffre est une grandeur que le moteur du site NOMME — la section ``rs`` de
``api_web/dalle._calculer``, ses ``grandeurs`` et les termes de ses formules (``verifications[…]["formules"]``) ; ce
fichier CHOISIT, COMPOSE et MET EN FORME, aux arrondis de la note (``ndc_pdf.data.fn``). La planche du générateur
Streamlit dit les mêmes chiffres : ``api_web/note_dalle.ecarts`` les compare avant chaque note.

- ``entete`` : l'EN-TÊTE COMPACT, une fois sous le titre de chaque planche (``ndc_pdf.styles.Style.entete``) — la dalle
  (son type, b, h, h_p, c, le jeu, la direction principale), le béton (f_ck, f_ctm, α_b), l'acier (f_yd = f_yk / γ_s,
  μ), et la phrase « Non vérifié(e)s : … » de l'écran (``api_web/dalle.non_verifies``, une source) ;
- ``sollicitations`` : les efforts de la section en tableau — une ligne par direction, M⁺ · inf. tendue, M⁻ · sup.
  tendue, puis V : la saisie de l'écran (``api_web/plaque_compact.cartes``) ;
- ``verifications`` : 1 la HAUTEUR MINIMALE détaillée — le moment dimensionnant et sa famille, h_u,min en symboles, en
  valeurs et son résultat, l'enrobage mécanique y_G (l'enrobage c, le demi-diamètre, le jeu ou la couche porteuse,
  quand cette décomposition redonne EXACTEMENT l'axe du moteur ; la formule des couches sinon), puis h_min ≤ h ; 2 les
  sections d'acier minimale et maximale COMMUNES aux quatre nappes, calculées une fois (f_ctm, b et h ne se répètent
  plus dans chaque nappe) ; 3 à 6 les nappes dans l'ordre de l'écran (``MO.ordre_faces``) : le moment, d, A_s,req,
  A_s,min, puis l'armature POSÉE comme un résultat (A_s, et le taux dans le verdict) ; 7 l'effort tranchant (τ et
  τ_adm,I en formules) ; enfin, seulement s'il y en a, les AVERTISSEMENTS de l'écran (« ! ») — le bloc
  « Observations » du panneau de gauche n'existe plus.

Les textes riches (« $f_{ck}$ ») sont composés par ``ndc_pdf.riche`` ; les formules par ``ndc_pdf.mathx``.
"""
from typing import Dict, List, Optional, Tuple

from beton.dalle import moteur as MO

from . import commun as C
from . import dalle_positions as POS

FACES = {"inf": "inférieure", "sup": "supérieure"}
FACES_PLURIEL = {"inf": "inférieures", "sup": "supérieures"}
PAREMENTS = {"inf": "inférieur", "sup": "supérieur"}
TOLERANCE_AXE = 1e-9                  # cm : une décomposition de l'axe ne s'écrit que si elle le redonne exactement
TITRE_AVERTISSEMENTS = "Avertissements"
NUM_AVERTISSEMENTS = "!"
TITRE_HAUTEUR = "Hauteur minimale"
TITRE_MINMAX = "Sections d'acier minimale et maximale"
TITRE_TRANCHANT = "Effort tranchant"
PREMIERE_NAPPE = 3            # le numéro de la première nappe : 1 la hauteur, 2 les sections minimale et maximale
# la largeur d'une colonne de calcul (pt) en deux et en trois colonnes : ndc_pdf.styles.Style.planche (842 pt de page
# paysage, marges de 28, panneau de 292 + 24, 20 entre deux colonnes) — une ligne qui n'y tient pas se replie (tests/test_note_dalle.py les lie)
LARGEUR_COLONNE = {2: 225.0, 3: 143.0}


def _nd():
    """Le module de mise en forme de la note (ndc_pdf.data : fn, sci, unités insécables) — chargé à la demande : le
    paquet de la note, pas le moteur du site."""
    from ndc_pdf import data as ND
    return ND


def _termes(v: Dict, cle: str) -> Optional[Dict]:
    """Les termes {symbole: valeur} de la formule ``cle`` d'une vérification du moteur, plus son résultat (« = ») ; None
    si le moteur ne la nomme pas."""
    for f in v.get("formules", []):
        if f["cle"] == cle:
            out = {t["symbole"]: t["valeur"] for t in f["termes"]}
            out["="] = f["resultat"]
            return out
    return None


def _par_cle(rs: Dict) -> Dict[str, Dict]:
    return {v["cle"]: v for v in rs["verifications"]}


def _taux(v: Dict) -> str:
    """Le taux d'une vérification, écrit comme à l'écran (« 90 % » ; jamais hors de la bande de sa couleur)."""
    return f"{C.pc(C.pct(v['taux']))}{C.NBSP}%"


def _cm(x: float) -> str:
    return _nd().fn(x, 1)


def _mm2(x: float) -> str:
    return _nd().fn(x, 0)


# ------------------------------------------------------------------ l'en-tête compact de la planche
def type_txt(d: Dict) -> str:
    """Le type de la dalle, comme l'identification du module à l'écran : « Dalle pleine » ou « Prédalle »."""
    return "Prédalle" if POS.type_dalle(d) == "predalle" else "Dalle pleine"


def entete(d: Dict, rs: Dict) -> Dict:
    """L'en-tête compact d'une planche (``ndc_pdf.styles.Style.entete``) : trois cases — la dalle, le béton, l'acier —
    et la phrase de ce qui n'est pas vérifié. Les données saisies telles quelles (``C.nb``), les grandeurs calculées aux
    arrondis de la note."""
    from . import dalle as DL
    ND = _nd()
    g = rs["grandeurs"]
    dalle = f"$b$ = {C.nb(g['b'])} cm · $h$ = {C.nb(g['h'])} cm"
    if POS.type_dalle(d) == "predalle":
        dalle += f" (prédalle $h_{{p}}$ = {C.nb(g['h_pre'])} cm)"
    dalle += f" · $c$ = {C.nb(g['enrobage_calcul'])} cm"
    if not g["enrobage"]:
        dalle += " (0 saisi)"
    if g["jeu_premier_lit"]:
        dalle += f" · jeu {C.nb(g['jeu_premier_lit'])} cm"
    dalle += f" · principale {str(g['principale']).upper()}"
    beton = (f"$f_{{ck}}$ = {ND.fn(g['fck'], 0)} · $f_{{ctm}}$ = {ND.fn(g['fctm'], 2)} · "
             rf"$\alpha_{{b}}$ = {ND.fn(g['alpha_b'], 2)} N/mm²")
    acier = (f"$f_{{yd}}$ = {int(g['fyk'])} / {C.nb(g['gamma_s'])} = {ND.fn(g['fyd'], 1)} N/mm² · "
             rf"$\mu$ = {ND.fn(g['mu'], 4)}")
    return {"cases": [(type_txt(d), dalle), (f"Béton {g['beton']}", beton), (f"Acier B{int(g['fyk'])}", acier)],
            "note": DL.non_verifies(d)}


# ------------------------------------------------------------------ les sollicitations (le panneau de gauche)
TITRE_SOLLICITATIONS = "Sollicitations · ELS"
ENTETES_SOLLICITATIONS = ("", "$M^{+}$ · inf. tendue", "$M^{-}$ · sup. tendue")


def sollicitations(rs: Dict) -> Dict:
    """Les efforts de la section, en tableau comme à la saisie : {"titre", "entetes", "lignes": [(direction, M⁺, M⁻)],
    "V"} — les nombres aux arrondis de la note (kN·m et kN au dixième)."""
    ND = _nd()
    return {"titre": TITRE_SOLLICITATIONS, "entetes": ENTETES_SOLLICITATIONS,
            "lignes": [(f"Direction {dk.upper()}", ND.fn(rs[f"M{dk}_inf"], 1), ND.fn(rs[f"M{dk}_sup"], 1)) for dk in ("x", "y")],
            "V": ND.fn(rs["V"], 1)}


# ------------------------------------------------------------------ 1. la hauteur minimale, détaillée
def decomposition(d: Dict, s: Dict, rs: Dict, f: str, j: int) -> Optional[Tuple[str, str]]:
    """L'axe automatique de la couche ``j`` de la face ``f``, décomposé — (expression mini-LaTeX, texte riche) — ou
    None : un axe saisi, ou une règle qu'aucune décomposition ci-dessous ne redonne EXACTEMENT (à ``TOLERANCE_AXE``
    près : la note n'écrit jamais une somme fausse). Une couche que le site pose sur une autre (la nappe secondaire, un
    renfort sur le treillis) : l'axe de la couche porteuse + son demi-Ø + le sien (``POS.support``) ; une couche posée par
    la règle du moteur (beton/dalle, beton/predalle : ``distance_auto``) : c + Ø/2 + jeu (la dalle, une face
    supérieure), c + Ø/2 (la base inférieure d'une prédalle, sans jeu), h_p + Ø/2 (posée sur la prédalle) — le demi-Ø
    arrondi au 0,5 cm supérieur, comme le moteur."""
    ND = _nd()
    c = rs["faces"][f]["couches"][j]
    if not c.get("auto"):
        return None
    g = rs["grandeurs"]
    demi = POS._demi(s["faces"][f]["couches"][j])
    phi = f"Ø {C.nb(c['d'])}"
    arrondi = "" if abs(demi - float(c["d"]) / 20.0) < 1e-9 else ", arrondi au 0,5 cm supérieur"
    if c.get("empilee"):
        sup = POS.support(d, s, f, j)
        if abs(sup["e"] + sup["demi"] + demi - c["e"]) < TOLERANCE_AXE:
            expr = (rf"e_{{appui}} + \frac{{\phi_{{appui}}}}{{2}} + \frac{{\phi}}{{2}} = "
                    rf"{ND.fn(sup['e'], 1)} + {C.nb(sup['demi'])} + {C.nb(demi)}")
            return expr, (f"La couche est posée sur {POS.nom_support(d, s, f, j)} : l'axe de celle-ci, son demi-diamètre, "
                          f"puis celui de la couche ({phi}{arrondi}).")
        return None
    c_, jeu = g["enrobage_calcul"], g["jeu_premier_lit"]
    regles = []
    if jeu:
        regles.append((c_ + demi + jeu, rf"c + \frac{{\phi}}{{2}} + jeu = {C.nb(c_)} + {C.nb(demi)} + {C.nb(jeu)}",
                       f"L'enrobage $c$, le demi-diamètre de la couche ({phi}{arrondi}), puis le jeu du premier lit."))
    regles.append((c_ + demi, rf"c + \frac{{\phi}}{{2}} = {C.nb(c_)} + {C.nb(demi)}",
                   f"L'enrobage $c$ puis le demi-diamètre de la couche ({phi}{arrondi})."))
    if g.get("h_pre"):
        regles.append((g["h_pre"] + demi, rf"h_{{p}} + \frac{{\phi}}{{2}} = {C.nb(g['h_pre'])} + {C.nb(demi)}",
                       f"La couche est posée sur la prédalle : son épaisseur $h_{{p}}$ puis le demi-diamètre de la couche "
                       f"({phi}{arrondi})."))
    for valeur, expr, texte in regles:
        if abs(valeur - c["e"]) < TOLERANCE_AXE:
            return expr, texte
    return None


COUCHES_EN_FORMULE = 3     # au-delà, les couches d'un y_G se listent en texte (la substitution ne tiendrait pas en colonne)
CORPS_UNE_LIGNE = 7.0      # pt : une formule qui tient dans sa colonne à ce corps s'écrit sur une ligne


def tient_formule(expr: str, n_cols: int = 2, corps: float = CORPS_UNE_LIGNE) -> bool:
    """La formule tient-elle dans une colonne de la planche à ``n_cols`` colonnes, au corps ``corps`` (``ndc_pdf.mathx``,
    le style des formules du kit) ?"""
    from ndc_pdf import mathx as MX
    from ndc_pdf import styles as STY
    from ndc_pdf.fonts import register_all
    register_all()
    return MX.measure(MX.parse(expr), STY.Style().ms(), corps).w <= LARGEUR_COLONNE[n_cols]


def yG_couches(v: Dict) -> List[Tuple]:
    """Les lignes de la formule de y_G d'une face à plusieurs couches (les termes du moteur : A_s,i en mm²/m, e_i en cm) :
    jusqu'à COUCHES_EN_FORMULE couches, la formule en symboles, sa substitution et son résultat — sur une ligne si elle
    tient dans la colonne (``tient_formule``), sur deux sinon ; au-delà, les couches en une phrase repliée."""
    ND = _nd()
    t = _termes(v, "yG")
    couches = [x for k, x in t.items() if k != "="]
    res = rf"\res{{{_cm(t['='])} \u{{cm}}}}"
    symb = r"y_{G} = \frac{\sum A_{s,i} \cdot e_{i}}{\sum A_{s,i}}"
    if len(couches) <= COUCHES_EN_FORMULE:
        num = " + ".join(rf"{ND.fn(a, 0)} \cdot {ND.fn(e, 1)}" for a, e in couches)
        den = " + ".join(ND.fn(a, 0) for a, _ in couches)
        subst = rf"\frac{{{num}}}{{{den}}} = {res}"
        if tient_formule(f"{symb} = {subst}"):
            return [("f", "", f"{symb} = {subst}")]
        return [("f", "", symb), ("f", "", f"= {subst}")]
    liste = " ; ".join(f"{ND.fn(a, 0)} à {ND.fn(e, 1)} cm" for a, e in couches)
    return [("f", "", symb), ("r", f"Les {len(couches)} couches ($A_{{s,i}}$ en mm²/m, à leur axe $e_{{i}}$) : {liste}."),
            ("f", "", f"= {res}")]


def _yG_lignes(d: Dict, s: Dict, rs: Dict, v: Dict, libelle: Optional[str], ref: Optional[int] = None) -> List[Tuple]:
    """Les lignes de y_G d'une face : imposé (saisi) ; plusieurs couches : la formule des couches dans sa nappe
    (``libelle`` None), sa valeur et le renvoi à la nappe ``ref`` dans la hauteur ; une seule couche : rien dans une nappe
    (d = h − y_G dit l'axe, la coupe le cote), son axe décomposé dans la hauteur (``decomposition``), à défaut sa valeur
    et pourquoi."""
    g = rs["grandeurs"]
    f = v["cle"]
    cote, dk = f.split("_")
    yG = g[f"yG_{f}"]
    du = f"Du parement {PAREMENTS[cote]} au centre de gravité des armatures {cote}. {dk.upper()}"
    if g.get(f"yG_impose_{f}"):
        return [("v", libelle or "Centre de gravité", "y_{G}", _cm(yG), "cm"), ("r", f"{du} : imposé (saisi).")]
    if _termes(v, "yG") is not None:               # plusieurs couches : Σ A·e / Σ A, les termes du moteur
        if libelle is None:
            return yG_couches(v)
        return [("v", libelle, "y_{G}", _cm(yG), "cm"),
                ("r", f"{du} : le centre de gravité de ses couches, calculé en {ref}." if ref else f"{du}.")]
    if libelle is None:
        return []
    dec = decomposition(d, s, rs, f, 0)
    if dec is None:
        quoi = "axe saisi" if not rs["faces"][f]["couches"][0].get("auto") else "l'axe de la couche"
        return [("v", libelle, "y_{G}", _cm(yG), "cm"), ("r", f"{du} : {quoi}.")]
    expr, texte = dec
    return [("f", libelle, rf"y_{{G}} = {expr} = \res{{{_cm(yG)} \u{{cm}}}}"), ("r", f"{du}. {texte}")]


def verif_hauteur(d: Dict, s: Dict, rs: Dict, num: int = 1) -> Dict:
    """La hauteur minimale, détaillée (point 8) : le moment dimensionnant et sa face (la plus exigeante des faces
    sollicitées, que le moteur nomme), h_u,min en symboles PUIS en valeurs, son résultat, l'enrobage mécanique y_G
    décomposé, h_min = h_u,min + y_G comparée à h, le verdict et son taux."""
    ND = _nd()
    g = rs["grandeurs"]
    v = _par_cle(rs)["hauteur"]
    tu, th = _termes(v, "hu_min"), _termes(v, "h_min")
    f = v.get("face") or g.get("face_hauteur")
    items: List[Tuple] = []
    if tu is not None and tu["M"] > 0 and f:
        cote, dk = f.split("_")
        items += [
            ("v", "Moment dimensionnant", f"M_{{{dk},{cote}}}", ND.fn(tu["M"], 1), "kNm"),
            ("r", f"Face {FACES[cote]} {dk.upper()} : la plus exigeante des faces sollicitées ($h_{{u,min}} + y_{{G}}$ le plus "
                  "grand)."),
            ("f", "Hauteur utile minimale", r"h_{u,min} = \sqrt{\frac{M}{\alpha_{b} \cdot b \cdot \mu}}"),
            ("f", "", rf"= \sqrt{{\frac{{{ND.sci(tu['M'] * 1e6)}}}{{{ND.fn(tu['alpha_b'], 2)} \cdot {ND.fn(tu['b'], 0)} \cdot "
                      rf"{ND.fn(tu['mu'], 4)}}}}} = \res{{{_cm(tu['='])} \u{{cm}}}}"),
            ("r", f"$M$ en N·mm, $b$ en mm (la bande) ; $\\alpha_{{b}}$ = {ND.fn(tu['alpha_b'], 2)} N/mm² et "
                  f"$\\mu$ = {ND.fn(tu['mu'], 4)} : le béton et l'acier de l'en-tête (valeurs du bureau)."),
        ]
        items += _yG_lignes(d, s, rs, _par_cle(rs)[f], "Enrobage mécanique", ref=PREMIERE_NAPPE + MO.ordre_faces(rs["principale"]).index(f))
    else:
        items.append(("r", "Aucun moment appliqué : pas de hauteur utile minimale ($h_{u,min}$ = 0)."))
    ok = v["etat"] == MO.OK
    h_min, h = th["="], g["h"]
    items += [
        ("f", "Hauteur minimale",
         rf"h_{{min}} = h_{{u,min}} + y_{{G}} = {_cm(th['hu_min'])} + {_cm(th['yG'])} = \res{{{_cm(h_min)} \u{{cm}}}}"
         rf" {r'\le' if ok else '>'} h = {C.nb(h)} \u{{cm}}"),
        ("k", 0),
    ]
    txt = f"Épaisseur {C.nb(h)} cm {'≥' if ok else '<'} hauteur minimale {_cm(h_min)} cm — taux {_taux(v)}"
    return dict(num=num, titre=TITRE_HAUTEUR, items=items,
                verdicts=[dict(etat="ok" if ok else "ko", texte=ND._unites_insecables(txt))])


# ------------------------------------------------------------------ 2. les sections minimale et maximale communes
def verif_minmax(rs: Dict, num: int = 2) -> Dict:
    """Les sections d'acier minimale et maximale COMMUNES aux quatre nappes, une fois (point 8 : f_ctm, b et h ne se
    répètent plus dans chaque nappe) — les termes des formules du moteur (identiques sur les quatre faces)."""
    ND = _nd()
    g = rs["grandeurs"]
    v = _par_cle(rs)[MO.ordre_faces(rs["principale"])[0]]
    te, tp, tx = _termes(v, "As_min_ec"), _termes(v, "As_min_plancher"), _termes(v, "As_max")
    b, h = ND.fn(te["b"], 0), ND.fn(te["h"], 0)
    items = [
        ("f", "Minimum du béton tendu",
         rf"A_{{s,min,EC}} = {C.nb(te['k'], 4)} \cdot \frac{{f_{{ctm}}}}{{f_{{yk}}}} \cdot b \cdot h = {C.nb(te['k'], 4)} \cdot "
         rf"\frac{{{ND.fn(te['fctm'], 2)}}}{{{ND.fn(te['fyk'], 0)}}} \cdot {b} \cdot {h} = \res{{{_mm2(te['='])} \u{{mm}}^{{2}}}}"),
        ("f", "Minimum de plancher",
         rf"A_{{s,min,pl}} = {C.nb(tp['k'], 4)} \cdot b \cdot h = {C.nb(tp['k'], 4)} \cdot {b} \cdot {h}"
         rf" = \res{{{_mm2(tp['='])} \u{{mm}}^{{2}}}}"),
        ("f", "Minimum commun aux quatre nappes",
         rf"A_{{s,min,0}} = \t{{max}}(A_{{s,min,EC}} ; A_{{s,min,pl}}) = \res{{{_mm2(g['As_min_base'])} \u{{mm}}^{{2}}}}"),
        ("f", "Maximum",
         rf"A_{{s,max}} = {C.nb(tx['k'], 4)} \cdot b \cdot h = {C.nb(tx['k'], 4)} \cdot {b} \cdot {h}"
         rf" = \res{{{_mm2(tx['='])} \u{{mm}}^{{2}}}}"),
        ("r", "$b$ et $h$ en mm. Le minimum de chaque nappe y ajoute le quart de l'acier requis sur la face opposée."),
    ]
    return dict(num=num, titre=TITRE_MINMAX, items=items, verdicts=[])


# ------------------------------------------------------------------ 3 à 6. les nappes
def titre_nappe(v: Dict) -> str:
    cote, dk = v["cle"].split("_")
    return f"Armatures {FACES_PLURIEL[cote]} {dk.upper()} ({'principale' if v.get('principale') else 'secondaire'})"


def verdict_nappe(v: Dict, rs: Dict) -> Dict:
    """Le verdict d'une nappe : la comparaison de l'écran et son taux ; pour une nappe que le moteur déclare non
    conforme pour une raison que « A_s ≥ A_s,néc » ne dit pas, cette raison — acier excessif (A_s couvre A_s,néc), ou
    d ≤ 0 QUEL QUE SOIT le taux (``api_web/dalle._verdict``)."""
    ND = _nd()
    g = rs["grandeurs"]
    f = v["cle"]
    cote, dk = f.split("_")
    face = f"{FACES[cote]} {dk.upper()}"
    ok = v["etat"] == MO.OK
    if not ok and "d_utile_negatif" in v["motifs"]:
        txt = (f"Hauteur utile {face} : d = {_cm(g['d_' + f])} cm ≤ 0 — position des couches incompatible avec "
               "l'épaisseur")
    elif not ok and "acier_excessif" in v["motifs"] and v["taux"] <= 1:
        txt = (f"Section d'armature {face} : {_mm2(v['limite'])} mm² > section d'armature maximale : "
               f"{_mm2(v['As_max'])} mm² (acier excessif)")
    else:
        bande = "" if abs(g["b"] - 100.0) < TOLERANCE_AXE else f" sur la bande de {C.nb(g['b'])} cm"   # A_s = A_s/m · b
        txt = (f"{_mm2(g['As_' + f])} mm² posés{bande} {'≥' if ok else '<'} {_mm2(g['besoin_' + f])} mm² requis — "
               f"taux {_taux(v)}")
    return dict(etat="ok" if ok else "ko", texte=ND._unites_insecables(txt))


CORPS_MIN_POSEES = 6.4        # le corps sous lequel la ligne en gras des armatures posées serait illisible (pt)


def _tient(texte: str, n_cols: int) -> bool:
    """La ligne en gras (``t``) tient-elle dans une colonne de la planche à ``n_cols`` colonnes, sans descendre sous
    CORPS_MIN_POSEES ? (la police et le corps de la ligne : ceux du kit)."""
    from ndc_pdf import styles as STY
    from ndc_pdf.fonts import register_all, string_width
    register_all()                                 # (les polices du kit : une fois, comme Doc)
    return string_width(texte, STY.Style.f_v, CORPS_MIN_POSEES) <= LARGEUR_COLONNE[n_cols]


def lignes_posees(libelle: str, as_m: str) -> List[Tuple]:
    """L'armature POSÉE d'une nappe, un résultat (point 8) : « Armatures posées : Treillis 10/10/100/100 — 785 mm²/m »
    sur une ligne en gras ; une désignation trop longue pour la colonne (des renforts) se replie : la section en gras, la
    désignation dessous. Le 3ᵉ élément : la forme d'une planche en trois colonnes (``note_dalle._trois_colonnes``)."""
    ligne = f"Armatures posées : {libelle} — {as_m} mm²/m"
    replie = [("t", f"Armatures posées : {as_m} mm²/m"), ("r", libelle)]
    if not _tient(ligne, 2):
        return replie
    return [("t", ligne, ligne if _tient(ligne, 3) else replie)]


def verif_nappe(d: Dict, s: Dict, rs: Dict, v: Dict, num: int) -> Dict:
    """Une nappe (face × direction) : le moment, y_G (plusieurs couches, ou imposé), d, A_s,req, A_s,min (le minimum
    commun ou le quart de la face opposée), puis l'armature POSÉE, un résultat : sa désignation et A_s sur la bande."""
    ND = _nd()
    g = rs["grandeurs"]
    f = v["cle"]
    cote, dk = f.split("_")
    # (sans libellé au-dessus des formules : leur symbole le dit, la planche tient en deux colonnes — DAL6)
    items: List[Tuple] = list(_yG_lignes(d, s, rs, v, None))
    td = _termes(v, "d")
    items.append(("f", "", rf"d = h - y_{{G}} = {C.nb(td['h'])} - {_cm(td['yG'])} = \res{{{_cm(td['='])} \u{{cm}}}}"))
    tr = _termes(v, "As_req")
    if tr["M"] > 0:
        items.append(("f", "",
                      rf"A_{{s,req}} = \frac{{M}}{{f_{{yd}} \cdot {C.nb(tr['k'])} \cdot d}} = \frac{{{ND.sci(tr['M'] * 1e6)}}}"
                      rf"{{{ND.fn(tr['fyd'], 1)} \cdot {C.nb(tr['k'])} \cdot {ND.fn(tr['d'] * 10, 0)}}}"
                      rf" = \res{{{_mm2(tr['='])} \u{{mm}}^{{2}}}}"))
    else:
        items.append(("f", "", r"A_{s,req} = \res{0 \u{mm}^{2}} \q \t{(aucun moment sur cette face)}"))
    tm, to = _termes(v, "As_min"), _termes(v, "As_min_opposee")
    opp = v["opposee"].split("_")[0]
    items.append(("f", "",
                  rf"A_{{s,min}} = \t{{max}}(A_{{s,min,0}} ; {C.nb(to['k'])} \cdot A_{{s,req,{opp}}}) = "
                  rf"\t{{max}}({_mm2(g['As_min_base'])} ; {_mm2(to['='])}) = \res{{{_mm2(tm['='])} \u{{mm}}^{{2}}}}",
                  # la forme courte d'une planche en trois colonnes (``note_dalle._trois_colonnes``)
                  rf"A_{{s,min}} = \t{{max}}({_mm2(g['As_min_base'])} ; {_mm2(to['='])}) = \res{{{_mm2(tm['='])} \u{{mm}}^{{2}}}}"))
    items += lignes_posees(v["libelle"], _mm2(_termes(v, "As")["As_pm"])) + [("k", 0)]
    return dict(num=num, titre=titre_nappe(v), items=items, verdicts=[verdict_nappe(v, rs)], cle=f)


# ------------------------------------------------------------------ 7. l'effort tranchant
def verif_tranchant(rs: Dict, num: int) -> Dict:
    """τ = V / (0,75·b·h) contre τ_adm,I (une dalle sans étriers), les deux en formules."""
    ND = _nd()
    v = _par_cle(rs)["tau"]
    tt, tl = _termes(v, "tau"), _termes(v, "tau_lim")
    ok = v["etat"] == MO.OK
    items = [
        ("v", "Effort tranchant", "V", ND.fn(tt["V"], 1), "kN"),
        ("f", "Contrainte tangentielle",
         rf"\tau = \frac{{V}}{{{C.nb(tt['k'])} \cdot b \cdot h}} = \frac{{{ND.sci(tt['V'] * 1e3)}}}{{{C.nb(tt['k'])} \cdot "
         rf"{ND.fn(tt['b'], 0)} \cdot {ND.fn(tt['h'], 0)}}} = \res{{{ND.fn(tt['='], 2)} \u{{N/mm}}^{{2}}}}"),
        ("f", "Contrainte admissible d'une dalle sans étriers",
         rf"\tau_{{adm,I}} = \frac{{{C.nb(tl['k'], 4)} \cdot f_{{ck,cube}}}}{{{C.nb(tl['gamma'])}}} = "
         rf"\frac{{{C.nb(tl['k'], 4)} \cdot {C.nb(tl['fck_cube'])}}}{{{C.nb(tl['gamma'])}}} = \res{{{ND.fn(tl['='], 2)} "
         rf"\u{{N/mm}}^{{2}}}}"),
        ("k", 0),
    ]
    txt = (f"Contrainte tangentielle {ND.fn(tt['='], 2)} N/mm² {'≤' if ok else '>'} admissible {ND.fn(tl['='], 2)} N/mm² — "
           f"taux {_taux(v)}")
    return dict(num=num, titre=TITRE_TRANCHANT, items=items,
                verdicts=[dict(etat="ok" if ok else "ko", texte=ND._unites_insecables(txt))])


# ------------------------------------------------------------------ la planche
def verif_avertissements(textes: List[str]) -> Dict:
    """Les avertissements de l'écran (rares : un axe saisi qui met deux nappes au même niveau, une base qui sort de la
    peau d'une prédalle…), en fin de planche, seulement s'il y en a — un paragraphe chacun, sans verdict (la couleur de
    l'état de la planche reste celle de l'écran)."""
    return dict(num=NUM_AVERTISSEMENTS, titre=TITRE_AVERTISSEMENTS, items=[("p", t) for t in textes], verdicts=[])


def verifications(d: Dict, i: int, rs: Dict, avertissements: List[str]) -> List[Dict]:
    """Les vérifications de la section ``i`` de la dalle ``d`` (``rs`` : sa section calculée par le moteur du site) :
    hauteur, sections minimale et maximale, les quatre nappes dans l'ordre de l'écran, l'effort tranchant s'il y en a
    un, les avertissements s'il y en a."""
    s = d["sections"][i]
    par_cle = _par_cle(rs)
    out = [verif_hauteur(d, s, rs, 1), verif_minmax(rs, 2)]
    for k, f in enumerate(MO.ordre_faces(rs["principale"])):
        out.append(verif_nappe(d, s, rs, par_cle[f], PREMIERE_NAPPE + k))
    if rs["V"] > 0 and "tau" in par_cle and _termes(par_cle["tau"], "tau") is not None:
        out.append(verif_tranchant(rs, len(out) + 1))
    if avertissements:
        out.append(verif_avertissements(avertissements))
    return out
