r"""
data.py — contenu de la note.

Deux rôles, clairement séparés :

1. `construire_doc()` et `construire_sections()` — LE branchement sur le
   moteur de calcul. Elles reçoivent des données NEUTRES (dictionnaires de
   nombres et de chaînes produits par `modules/export_pdf._compute_section`,
   fidèle à poutre.py) et composent DOC / SECTIONS. Aucune valeur n'est
   recalculée ici : le moteur calcule, ce fichier transcrit. Les nombres
   sont formatés avec les mêmes conventions que l'export d'origine
   (virgule décimale, mêmes décimales, notation a·10^n identique à
   sci_tokens).

2. Le jeu d'essai figé (DOC / SECTIONS ci-dessous) — strictement celui de
   l'export Streamlit de référence. Il alimente `build.py` (recette :
   11 PDF) et sert de référence d'identité contre NOTE_DE_CALCUL.pdf.
   Ne pas le modifier : c'est un étalon, pas un exemple.

Aucune donnée n'est ajoutée : pas de nom de projet, pas de rédacteur, pas de
référence de norme. Les champs PROJET et PARTIE restent vides comme dans
l'export d'origine. Les libellés sont ceux de la note.
"""
import math


# ============================================================
#  FORMAT NOMBRES — mêmes conventions que l'export d'origine
# ============================================================
def fn(x, nd=2):
    """Virgule décimale française, nd décimales (copie de export_pdf.fn)."""
    try:
        return f"{float(x):.{nd}f}".replace(".", ",")
    except Exception:
        return str(x)


def sci(value):
    r"""Notation a·10^n en mini-LaTeX — mêmes paliers que sci_tokens :
    n = 6 dès 10^6, n = 3 dès 10^3, sinon pas de puissance."""
    v = float(value)
    if v == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(v))))
    n = 6 if exp >= 6 else (3 if exp >= 3 else 0)
    mant = v / (10 ** n)
    ms = f"{round(mant):d}" if abs(mant - round(mant)) < 1e-9 else f"{mant:.1f}".replace(".", ",")
    if n == 0:
        return ms
    return rf"{ms} \cdot 10^{{{n}}}"


def _fmt_pas(p):
    """Pas d'étrier : entier si rond, sinon une décimale (légende d'origine)."""
    p = float(p)
    return f"{p:.0f}" if abs(p - round(p)) < 0.05 else f"{p:.1f}".replace(".", ",")


def fnt(x):
    """TRONQUÉ vers le bas à une décimale (28,9 et non 29,0 pour 28,95) :
    une valeur affichée doit être recopiable comme pas choisi sans faire
    basculer le verdict. Affichage uniquement — les comparaisons du
    moteur restent sur la valeur exacte."""
    return fn(math.floor(float(x) * 10.0) / 10.0, 1)


# espace insécable entre un nombre et son unité : la conclusion ne coupe
# jamais « 73,2 / cm » en deux lignes
NBSP = " "


def _unites_insecables(texte):
    """Colle chaque nombre à son unité dans les textes de conclusion."""
    for u in ("N/mm²", "kg/m³", "mm²", "kNm", "cm", "kN"):
        texte = texte.replace(f" {u}", f"{NBSP}{u}")
    return texte


# ============================================================
#  CONSTRUCTION DEPUIS LE MOTEUR
# ============================================================
# `resultats` : liste de dicts NEUTRES, un par section calculée :
#   poutre, section : noms affichés
#   R               : dict de _compute_section (fidèle à poutre.py)
#   stirrups        : lignes étriers/épingles [{d, brins}] (dessin)
#   ta_global       : taux d'armature global de la poutre (kg/m³) ou None
ETAT_LABELS = {"ok": "Vérifié", "warn": "À surveiller", "nok": "Non vérifié"}
_ETAT_NDC = {"ok": "ok", "warn": "att", "nok": "ko"}


def construire_doc(infos, date_defaut=""):
    """DOC depuis les infos projet de l'application. Champs absents = vides."""
    infos = infos or {}
    return dict(
        bureau="Bureau d'Études Valens",
        date=str(infos.get("date") or date_defaut),
        indice=str(infos.get("indice", "0") or "0"),
        projet=str(infos.get("nom_projet") or ""),
        partie=str(infos.get("partie") or ""),
        titre="Note de calcul",
    )


def _coupe_depuis_R(R, stirrups, pas, peau=None):
    """Payload de la coupe : millimètres, valeurs réelles du moteur.
    Multi-lits : liste complète avec la position d'axe RÉELLE (e, mm).
    Étriers/épingles : tous les groupes, avec leur position (de barre →
    à barre, sur le lit 1 inférieur). Armatures de peau : positions
    calculées par le moteur (_peau_bars)."""
    b_mm = R["b"] * 10.0
    h_mm = R["h"] * 10.0
    stirrups = stirrups or [{"d": 8, "brins": 2}]
    d_cadre = max(float(g.get("d", 8) or 8) for g in stirrups)

    def _lits(geo):
        lits, labs = [], []
        for lit in geo["lits"]:
            lits.append(dict(n=int(lit["n"]), dia=int(lit["d"]), e=lit["e"] * 10.0))
            labs.append((f"Lit {lit['i']} : {lit['n']} Ø{lit['d']}",
                         f"{fn(lit['As'], 0)} mm²"))
        return lits, labs

    lits_inf, labs_inf = _lits(R["geo_inf"])
    lits_sup, labs_sup = _lits(R["geo_sup"])

    # tous les groupes, avec leur emplacement (indices de barres du lit 1
    # inférieur ; None = toute la largeur)
    cadres = [dict(dia=float(g.get("d", 8) or 8), brins=int(g.get("brins", 2)),
                   de=g.get("from"), a=g.get("to")) for g in stirrups]

    # légende : une ligne par groupe, mêmes formes que la légende d'origine
    # (pas d'étrier sur les fermés, positions portées par le dessin)
    def _lab(g):
        if int(g.get("brins", 2)) == 1:
            return f"Épingle : Ø{int(g['d'])}"
        base = f"Étrier : Ø{int(g['d'])}"
        return base + (f" — {_fmt_pas(pas)} cm" if pas else "")

    # groupes identiques regroupés : « 2× Étrier : Ø8 — 20 cm »
    labs_cadre = []
    for lab in [_lab(g) for g in stirrups]:
        for k, (l0, n0) in enumerate(labs_cadre):
            if l0 == lab:
                labs_cadre[k] = (l0, n0 + 1)
                break
        else:
            labs_cadre.append((lab, 1))
    labs_cadre = [(f"{n}× {l}" if n > 1 else l) for l, n in labs_cadre]
    lab_cadre = f"Étrier : Ø{int(d_cadre)}"
    lab_cadre2 = f"{_fmt_pas(pas)} cm" if pas else ""
    if len(stirrups) == 1 and int(stirrups[0].get("brins", 2)) == 1:
        lab_cadre = f"Épingle : Ø{int(stirrups[0]['d'])}"

    out = dict(
        b=b_mm, h=h_mm,
        enrobage=R["enrob_beton"] * 10.0,
        cadre_dia=d_cadre,
        d=R["di"] * 10.0,
        lit_inf=dict(n=lits_inf[0]["n"], dia=lits_inf[0]["dia"]),
        lit_sup=dict(n=lits_sup[0]["n"], dia=lits_sup[0]["dia"]),
        lits_inf=lits_inf, lits_sup=lits_sup,
        labs_inf=labs_inf, labs_sup=labs_sup,
        cadres=cadres,
        b_label=f"b = {fn(R['b'], 0)} cm", h_label=f"h = {fn(R['h'], 0)} cm",
        c_label=f"c = {fn(R['enrob_beton'], 1)} cm",
        d_label=f"d = {fn(R['di'], 1)} cm",
        # d₁ de la face inférieure (h = d + d₁), cohérent avec la cote d
        d1_label=f"d₁ = {fn(R['geo_inf']['e_cdg'], 1)} cm",
        lab_sup=labs_sup[0][0], lab_sup2=labs_sup[0][1],
        lab_inf=labs_inf[0][0], lab_inf2=labs_inf[0][1],
        lab_cadre=lab_cadre, lab_cadre2=lab_cadre2,
    )
    if len(stirrups) > 1:
        out["labs_cadre"] = labs_cadre
    if peau and peau.get("n"):
        out["peau"] = dict(
            dia=float(peau["d"]),
            ys=[y * 10.0 for y in peau["ys"]],
            label=f"Armature de peau : 2×{int(peau['n'])} Ø{int(peau['d'])}")
    return out


def _blocs_depuis_R(R, ta_global=None):
    """DIMENSIONS / MATÉRIAUX / SOLLICITATIONS — libellés de la note."""
    dims = [("Largeur", "b", fn(R["b"], 0), "cm"),
            ("Hauteur", "h", fn(R["h"], 0), "cm"),
            # « c » : même symbole que la cote de la coupe (c = enrobage)
            ("Enrobage béton", "c", fn(R["enrob_beton"], 1), "cm")]
    mats = [("Béton", None, R["beton"], ""),
            (None, "f_{ck}", fn(R["fck"], 0), "N/mm²"),
            ("Acier", None, f"B{int(R['fyk'])}", ""),
            ("Coefficient acier ELS", None, fn(R["gamma_s"], 2), ""),
            ("Contrainte de calcul acier", "f_{yd}", fn(R["fyd"], 0), "N/mm²")]
    soll = [(None, "M_{inf}", fn(R["M_inf"], 1), "kNm")]
    if R.get("has_Msup"):
        soll.append((None, "M_{sup}", fn(R["M_sup"], 1), "kNm"))
    soll.append((None, "V", fn(R["V"], 1), "kN"))
    blocs = [("DIMENSIONS", dims), ("MATÉRIAUX", mats), ("SOLLICITATIONS", soll)]
    if ta_global is not None:
        # T.A. global de la poutre — uniquement si l'option PDF est activée
        blocs.append(("TAUX D'ARMATURE",
                      [("T.A. de la poutre", None, f"{ta_global:.0f}", "kg/m³")]))
    return blocs


def _verif_hauteur(R):
    ok = R["etat_h"] == "ok"
    txt = (f"Hauteur de la poutre : {fn(R['h'], 0)} cm "
           f"{'≥' if ok else '<'} hauteur minimale de la poutre : "
           f"{fn(R['h_min_poutre'], 1)} cm")
    return dict(
        num=1, titre="Vérification de la hauteur",
        items=[
            ("f", "Hauteur utile minimale",
             rf"h_{{u,min}} = \sqrt{{\frac{{{sci(R['M_max'] * 1e6)}}}"
             rf"{{{fn(R['alpha_b'], 2)} \cdot {fn(R['b'] * 10, 0)} \cdot {fn(R['mu'], 4)}}}}}"
             rf" = \res{{{fn(R['hmin'], 1)} \u{{cm}}}}"),
            # d₁ = distance du parement au c.d.g. des aciers de la face
            # dimensionnante (enrobage mécanique : h = d + d₁)
            ("f", "Hauteur minimale de la poutre",
             rf"h_{{u,min}} + d_{{1}} = {fn(R['hmin'], 1)} + {fn(R['e_cdg_gov'], 1)}"
             rf" = \res{{{fn(R['h_min_poutre'], 1)} \u{{cm}}}}"),
            ("v", "Hauteur minimale de la poutre", "h_{min}", fn(R["h_min_poutre"], 1), "cm"),
            ("v", "Hauteur de la poutre", "h", fn(R["h"], 0), "cm"),
            ("k", 0),
        ],
        verdicts=[dict(etat="ok" if ok else "ko", texte=_unites_insecables(txt))],
    )


def _verif_armatures(R, which, num):
    if which == "inf":
        titre, M = "Armatures inférieures", R["M_inf"]
        Ar, geo, d, et = R["As_req_inf"], R["geo_inf"], R["di"], R["etat_inf"]
        As_min, as_req_opp, face_opp = R["As_min_inf"], R["As_req_sup"], "sup"
        m_lab, m_sym, face_txt = "Moment inférieur", "M_{inf}", "inférieure"
    else:
        titre, M = "Armatures supérieures", R["M_sup"]
        Ar, geo, d, et = R["As_req_sup"], R["geo_sup"], R["ds"], R["etat_sup"]
        As_min, as_req_opp, face_opp = R["As_min_sup"], R["As_req_inf"], "inf"
        m_lab, m_sym, face_txt = "Moment supérieur", "M_{sup}", "supérieure"

    b_mm, h_mm = R["b"] * 10.0, R["h"] * 10.0
    nl = geo["nl"]
    du_label = "Hauteur utile" + (f" (c.d.g. de {nl} lits)" if nl > 1 else "")
    prend = " + ".join(f"{lit['n']} Ø{lit['d']}" for lit in geo["lits"])
    quart = 0.25 * as_req_opp

    ok = et == "ok"
    besoin = max(Ar, As_min)
    txt = (f"Section d'armature {face_txt} : {fn(geo['As'], 0)} mm² "
           f"{'≥' if ok else '<'} section d'armature requise : {fn(besoin, 0)} mm²")

    return dict(
        num=num, titre=titre,
        items=[
            ("v", m_lab, m_sym, fn(M, 1), "kNm"),
            ("f", du_label,
             rf"d_{{u}} = {fn(R['h'], 0)} - {fn(geo['e_cdg'], 1)}"
             rf" = \res{{{fn(d, 1)} \u{{cm}}}}"),
            ("f", "Acier requis",
             rf"A_{{s,req}} = \frac{{{sci(M * 1e6)}}}"
             rf"{{{fn(R['fyd'], 1)} \cdot 0,9 \cdot {fn(d * 10, 0)}}}"
             rf" = \res{{{fn(Ar, 0)} \u{{mm}}^{{2}}}}"),
            ("f", "Section d'acier min",
             rf"A_{{s,min}} = \max{{\frac{{0,26 \cdot {fn(R['fctm'], 1)}}}{{{int(R['fyk'])}}}"
             rf" \cdot {fn(b_mm, 0)} \cdot {fn(h_mm, 0)} = {fn(R['As_min_ec'], 0)} ; "
             rf"0,0013 \cdot {fn(b_mm, 0)} \cdot {fn(h_mm, 0)} = {fn(R['As_min_plancher'], 0)} ; "
             rf"0,25 \cdot A_{{s,req,{face_opp}}} = 0,25 \cdot {fn(as_req_opp, 0)}"
             rf" = {fn(quart, 0)}}} = \res{{{fn(As_min, 0)} \u{{mm}}^{{2}}}}"),
            ("f", "Section d'acier max",
             rf"A_{{s,max}} = 0,04 \cdot {fn(b_mm, 0)} \cdot {fn(h_mm, 0)}"
             rf" = \res{{{fn(R['As_max'], 0)} \u{{mm}}^{{2}}}}"),
            ("v", "Acier requis", "A_{s,req}", fn(Ar, 0), "mm²"),
            ("v", "Acier minimal", "A_{s,min}", fn(As_min, 0), "mm²"),
            ("t", f"On prend {prend} ({fn(geo['As'], 0)} mm²)"
                  + (f" · {nl} lits" if nl > 1 else "")),
            ("k", 0),
        ],
        verdicts=[dict(etat="ok" if ok else "ko", texte=_unites_insecables(txt))],
    )


def _verif_tranchant(Sh, R, num):
    b_mm, h_mm = R["b"] * 10.0, R["h"] * 10.0
    okt = Sh["tau"] <= Sh["tau_lim"]
    okp = Sh["pas"] <= Sh["pas_lim"]
    # τ : le moteur fournit TROIS états (ok / warn « barres inclinées » / nok)
    # -> ok / att / ko. Aucun seuil inventé ici : c'est celui de shear_need.
    etat_tau = "ko" if not okt else ("att" if Sh["etat_tau"] == "warn" else "ok")
    txt_tau = (f"Contrainte tangentielle : {fn(Sh['tau'], 2)} N/mm² "
               f"{'≤' if okt else '>'} contrainte tangentielle admissible : "
               f"{fn(Sh['tau_lim'], 2)} N/mm²")
    # les pas AFFICHÉS sont tronqués vers le bas (fnt) : recopiables sans
    # faire basculer le verdict, qui compare les valeurs exactes
    txt_pas = (f"Pas des armatures d'effort tranchant : {fn(Sh['pas'], 1)} cm "
               f"{'≤' if okp else '>'} pas maximal : {fnt(Sh['pas_lim'])} cm")

    if Sh["Ast"] > 0 and Sh["V"] > 0:
        f_sth = (rf"s_{{th}} = \frac{{{fn(Sh['Ast'], 1)} \cdot {fn(R['fyd'], 1)}"
                 rf" \cdot {fn(R['dsh'] * 10, 0)}}}{{{sci(Sh['V'] * 1e3)}}}"
                 rf" = \res{{{fnt(Sh['pas_th'])} \u{{cm}}}}")
    else:
        f_sth = r"s_{th} = —"

    return dict(
        num=num, titre="Effort tranchant — étriers",
        items=[
            ("f", "Contrainte tangentielle",
             rf"\tau = \frac{{{sci(Sh['V'] * 1e3)}}}"
             rf"{{0,75 \cdot {fn(b_mm, 0)} \cdot {fn(h_mm, 0)}}}"
             rf" = \res{{{fn(Sh['tau'], 2)} \u{{N/mm}}^{{2}}}}"),
            ("v", "Contrainte admissible", r"\tau_{adm}", fn(Sh["tau_lim"], 2), "N/mm²"),
            ("k", 0),
            ("s", "Étriers"),
            ("t", f"On prend {Sh['summary']}"),
            ("v", "Section", "A_{sw}", fn(Sh["Ast"], 1), "mm²"),
            ("f", "Pas théorique", f_sth),
            ("f", "Pas maximal",
             rf"s_{{max}} = \min{{0,75 \cdot d ; 30}} = \res{{{fnt(Sh['s_max'])} \u{{cm}}}}"),
            ("f", "Pas admissible",
             rf"s_{{adm}} = \min{{{fnt(Sh['pas_th'])} ; {fnt(Sh['s_max'])}}}"
             rf" = \res{{{fnt(Sh['pas_lim'])} \u{{cm}}}}"),
            ("v", "Pas retenu", "s", fn(Sh["pas"], 1), "cm"),
            ("k", 1),
        ],
        verdicts=[dict(etat=etat_tau, texte=_unites_insecables(txt_tau)),
                  dict(etat="ok" if okp else "ko", texte=_unites_insecables(txt_pas))],
    )


def construire_sections(resultats):
    """Liste SECTIONS (une planche par section) depuis les résultats réels.

    Chaque entrée de `resultats` : dict(poutre, section, R, stirrups,
    ta_global). Les formules sont transcrites depuis R sans aucun recalcul —
    mêmes facteurs, mêmes valeurs intermédiaires, mêmes arrondis que
    l'export d'origine."""
    sections = []
    for res in resultats:
        R = res["R"]
        Sh = R.get("shear")
        verifs = [_verif_hauteur(R),
                  _verif_armatures(R, "inf", 2),
                  _verif_armatures(R, "sup", 3)]
        if Sh:
            verifs.append(_verif_tranchant(Sh, R, 4))
        sections.append(dict(
            poutre=res["poutre"], section=res["section"],
            beton=R["beton"], acier=f"B{int(R['fyk'])}",
            etat=ETAT_LABELS.get(R["etat_global"], "Non vérifié"),
            coupe=_coupe_depuis_R(R, res.get("stirrups"),
                                  (Sh or {}).get("pas"), peau=res.get("peau")),
            blocs=_blocs_depuis_R(R, res.get("ta_global")),
            verifs=verifs,
        ))
    return sections


# ============================================================
#  JEU D'ESSAI FIGÉ — référence d'identité (NOTE_DE_CALCUL.pdf)
# ============================================================
DOC = dict(
    bureau="Bureau d'Études Valens",
    date="24/08/2026",
    indice=0,
    projet="",            # vide dans l'export d'origine
    partie="",
    titre="Note de calcul",
)

POUTRE = "Poutre 1"
SECTION = "Section A"
BETON = "C30/37"
ACIER = "B500"
ETAT = "Non vérifié"

COUPE = dict(
    b=200, h=400, enrobage=30, cadre_dia=10, d=340,
    lit_inf=dict(n=2, dia=16), lit_sup=dict(n=2, dia=16),
    b_label="b = 20 cm", h_label="h = 40 cm", c_label="c = 3,0 cm",
    d_label="d = 34 cm",
    lab_sup="Lit 1 : 2 Ø16", lab_sup2="402 mm²",
    lab_inf="Lit 1 : 2 Ø16", lab_inf2="402 mm²",
    lab_cadre="Étrier : Ø10", lab_cadre2="30 cm",
)

# (libellé, symbole, valeur, unité)
DIMENSIONS = [
    ("Largeur", "b", "20", "cm"),
    ("Hauteur", "h", "40", "cm"),
    ("Enrobage béton", None, "3,0", "cm"),
]
MATERIAUX = [
    ("Béton", None, "C30/37", ""),
    (None, "f_{ck}", "30", "N/mm²"),
    ("Acier", None, "B500", ""),
    ("Coefficient acier ELS", None, "1,50", ""),
    ("Contrainte de calcul acier", "f_{yd}", "333", "N/mm²"),
]
SOLLICITATIONS = [
    (None, "M_{inf}", "200,0", "kNm"),
    (None, "M_{sup}", "140,0", "kNm"),
    (None, "V", "230,0", "kN"),
]

BLOCS = [("DIMENSIONS", DIMENSIONS),
         ("MATÉRIAUX", MATERIAUX),
         ("SOLLICITATIONS", SOLLICITATIONS)]

# items : ("f", libellé, formule) | ("v", libellé, symbole, valeur, unité)
#         ("t", texte) | ("s", sous-titre) | ("k", verdict)
VERIFS = [
    dict(
        num=1, titre="Vérification de la hauteur",
        items=[
            ("f", "Hauteur utile minimale",
             r"h_{u,min} = \sqrt{\frac{200 \cdot 10^{6}}"
             r"{12,96 \cdot 200 \cdot 0,1709}} = \res{67,2 \u{cm}}"),
            ("f", "Hauteur minimale de la poutre",
             r"h_{u,min} + CDG_{arm} = 67,2 + 6,0 = \res{73,2 \u{cm}}"),
            ("v", "Hauteur minimale de la poutre", "h_{min}", "73,2", "cm"),
            ("v", "Hauteur de la poutre", "h", "40", "cm"),
            ("k", 0),
        ],
        verdicts=[dict(
            etat="ko",
            texte="Hauteur de la poutre : 40 cm < hauteur minimale de la poutre : 73,2 cm",
            gauche="40 cm", op="<", droite="73,2 cm", ratio=73.2 / 40.0,
            court="Hauteur")],
    ),
    dict(
        num=2, titre="Armatures inférieures",
        items=[
            ("v", "Moment inférieur", "M_{inf}", "200,0", "kNm"),
            ("f", "Hauteur utile", r"d_{u} = 40 - 6,0 = \res{34,0 \u{cm}}"),
            ("f", "Acier requis",
             r"A_{s,req} = \frac{200 \cdot 10^{6}}{333,3 \cdot 0,9 \cdot 340}"
             r" = \res{1961 \u{mm}^{2}}"),
            ("f", "Section d'acier min",
             r"A_{s,min} = \max{\frac{0,26 \cdot 2,9}{500} \cdot 200 \cdot 400 = 120 ; "
             r"0,0013 \cdot 200 \cdot 400 = 104 ; "
             r"0,25 \cdot A_{s,req,sup} = 0,25 \cdot 1373 = 343} = \res{343 \u{mm}^{2}}"),
            ("f", "Section d'acier max",
             r"A_{s,max} = 0,04 \cdot 200 \cdot 400 = \res{3200 \u{mm}^{2}}"),
            ("v", "Acier requis", "A_{s,req}", "1961", "mm²"),
            ("v", "Acier minimal", "A_{s,min}", "343", "mm²"),
            ("t", "On prend 2 Ø16 (402 mm²)"),
            ("k", 0),
        ],
        verdicts=[dict(
            etat="ko",
            texte="Section d'armature inférieure : 402 mm² < section d'armature requise : 1961 mm²",
            gauche="402 mm²", op="<", droite="1961 mm²", ratio=1961 / 402.0,
            court="Armatures inf.")],
    ),
    dict(
        num=3, titre="Armatures supérieures",
        items=[
            ("v", "Moment supérieur", "M_{sup}", "140,0", "kNm"),
            ("f", "Hauteur utile", r"d_{u} = 40 - 6,0 = \res{34,0 \u{cm}}"),
            ("f", "Acier requis",
             r"A_{s,req} = \frac{140 \cdot 10^{6}}{333,3 \cdot 0,9 \cdot 340}"
             r" = \res{1373 \u{mm}^{2}}"),
            ("f", "Section d'acier min",
             r"A_{s,min} = \max{\frac{0,26 \cdot 2,9}{500} \cdot 200 \cdot 400 = 120 ; "
             r"0,0013 \cdot 200 \cdot 400 = 104 ; "
             r"0,25 \cdot A_{s,req,inf} = 0,25 \cdot 1961 = 490} = \res{490 \u{mm}^{2}}"),
            ("f", "Section d'acier max",
             r"A_{s,max} = 0,04 \cdot 200 \cdot 400 = \res{3200 \u{mm}^{2}}"),
            ("v", "Acier requis", "A_{s,req}", "1373", "mm²"),
            ("v", "Acier minimal", "A_{s,min}", "490", "mm²"),
            ("t", "On prend 2 Ø16 (402 mm²)"),
            ("k", 0),
        ],
        verdicts=[dict(
            etat="ko",
            texte="Section d'armature supérieure : 402 mm² < section d'armature requise : 1373 mm²",
            gauche="402 mm²", op="<", droite="1373 mm²", ratio=1373 / 402.0,
            court="Armatures sup.")],
    ),
    dict(
        num=4, titre="Effort tranchant — étriers",
        items=[
            ("f", "Contrainte tangentielle",
             r"\tau = \frac{230 \cdot 10^{3}}{0,75 \cdot 200 \cdot 400}"
             r" = \res{3,83 \u{N/mm}^{2}}"),
            ("v", "Contrainte admissible", r"\tau_{adm}", "2,26", "N/mm²"),
            ("k", 0),
            ("s", "Étriers"),
            ("t", "On prend Étrier Ø10"),
            ("v", "Section", "A_{sw}", "157,1", "mm²"),
            ("f", "Pas théorique",
             r"s_{th} = \frac{157,1 \cdot 333,3 \cdot 340}{230 \cdot 10^{3}}"
             r" = \res{7,7 \u{cm}}"),
            ("f", "Pas maximal",
             r"s_{max} = \min{0,75 \cdot d ; 30} = \res{25,5 \u{cm}}"),
            ("f", "Pas admissible",
             r"s_{adm} = \min{7,7 ; 25,5} = \res{7,7 \u{cm}}"),
            ("v", "Pas retenu", "s", "30,0", "cm"),
            ("k", 1),
        ],
        verdicts=[
            dict(etat="ko",
                 texte="Contrainte tangentielle : 3,83 N/mm² > contrainte tangentielle "
                       "admissible : 2,26 N/mm²",
                 gauche="3,83 N/mm²", op=">", droite="2,26 N/mm²",
                 ratio=3.83 / 2.26, court="Cisaillement"),
            dict(etat="ko",
                 texte="Pas des armatures d'effort tranchant : 30,0 cm > pas maximal : 7,7 cm",
                 gauche="30,0 cm", op=">", droite="7,7 cm",
                 ratio=30.0 / 7.7, court="Pas des étriers"),
        ],
    ),
]


# Une planche paysage par section. Le rapport complet = page de garde
# portrait + une planche par entrée de SECTIONS.
SECTION_A = dict(
    poutre=POUTRE, section=SECTION, beton=BETON, acier=ACIER, etat=ETAT,
    coupe=COUPE, blocs=BLOCS, verifs=VERIFS,
)

SECTIONS = [SECTION_A]


def all_verdicts():
    out = []
    for v in VERIFS:
        for k in v["verdicts"]:
            out.append((v, k))
    return out
