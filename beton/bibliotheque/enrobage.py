# -*- coding: utf-8 -*-
"""L'ENROBAGE NOMINAL aux valeurs de la NORME — EN 1992-1-1:2004 §4.4.1 (durabilité, classe structurale) et
EN 1992-1-2:2004 §5 (méthode tabulée, résistance au feu), les libellés belges de la NBN B 15-001 tels que la NIT 260
(CSTC, 2017) les met en regard des classes de l'EN 206. C'est le moteur de la rubrique « Enrobages et feu » de la
bibliothèque ; l'ancien module « Enrobage » du site garde ``beton/enrobage`` (la copie de l'écran Streamlit, avec ses
écarts D1 à D11 de docs/beton/ENROBAGE_MOTEUR.md) jusqu'à son retrait.

Durabilité (chaque étape est un TERME nommé de ``calculer``) ::

    c_min,b    = Ø (barre isolée), + 5 mm si le plus gros granulat dépasse 32 mm        §4.4.1.2(3), tableau 4.2
    S          = S4 (durée d'utilisation 50 ans) + modifications du tableau 4.3N       §4.4.1.2(5) : 100 ans + 2 ;
                 classe de résistance − 1 ; élément assimilable à une dalle − 1 ;
                 maîtrise de la qualité − 1 ; bornée à [S1 ; S6]                        (= NIT 260, tableau 21)
    c_min,dur  = tableau 4.4N [S ; classe d'exposition]                                 (= NIT 260, tableau 20)
    c_min      = max(c_min,b ; c_min,dur + Δc_dur,γ − Δc_dur,st − Δc_dur,add ; 10 mm)  §4.4.1.2(2), éq. (4.2),
                 les Δc_dur valant 0 (valeurs recommandées)
    c_nom,dur  = c_min + Δc_dev                                                         §4.4.1.1(2), éq. (4.1) ;
                 Δc_dev = 10 mm, 5 mm (assurance qualité avec mesure de l'enrobage) ou 0 (préfabrication contrôlée)
    contre le sol : c_nom ≥ 40 mm sur béton de propreté, ≥ 75 mm directement contre le sol  §4.4.1.3(4)

Feu (méthode tabulée) : la distance à l'axe ``a`` des armatures principales, lue dans les tableaux de l'EN 1992-1-2
pour la durée R ; l'enrobage nominal qu'elle impose à l'armature la plus extérieure est c_nom,feu = a − Ø/2 − Ø_étrier
(``a`` est une valeur nominale : §5.2(5) ; aucune tolérance ne s'y ajoute). Barres d'angle d'une poutre à un seul lit :
a_sd = a + 10 mm (note du tableau 5.5), appliqué à la face latérale quand b < la plus grande largeur du tableau.
Tableaux portés : poutres sur appuis simples (tableau 5.5), dalles portant dans une direction (tableau 5.8). Les
tableaux des poteaux (5.2a) et des voiles (5.4) ne sont PAS transcrits : les valeurs de l'écran Streamlit s'en écartent
(D1, D2 : non conservatives), et cette session n'a pas le texte sous les yeux — ``FEU_A_TRANSCRIRE`` le dit, l'écran
aussi. Hors du moteur : poutres continues (tableau 5.6), dalles continues ou bidirectionnelles, méthode A des poteaux.
"""

SOURCE_DUR = "EN 1992-1-1:2004 §4.4.1 (tableaux 4.2, 4.3N, 4.4N) ; NBN EN 1992-1-1 ANB ; NBN B 15-001 (classes EI, EE, ES) via NIT 260 tableaux 20-21"
SOURCE_FEU = "EN 1992-1-2:2004 §5 (méthode tabulée : tableau 5.5 poutres, tableau 5.8 dalles)"

# ---------------------------------------------------------------------------------------------- classes d'exposition
# (classe EN 206, colonne du tableau 4.4N, libellé belge NBN B 15-001 tel que la NIT 260 le met en regard (tableau 20),
#  seuil de classe de résistance du tableau 4.3N (= NIT 260 tableau 21), description)
EXPOSITIONS = (
    ("X0", "X0", "—", "C30/37", "Aucun risque de corrosion : béton non armé, ou béton armé en ambiance très sèche"),
    ("XC1", "XC1", "EI", "C30/37", "Carbonatation : sec ou humide en permanence (intérieur sec, béton immergé)"),
    ("XC2", "XC2/XC3", "EE1, EE2", "C35/45", "Carbonatation : humide, rarement sec (fondations, contact durable avec l'eau)"),
    ("XC3", "XC2/XC3", "EE1, EE2", "C35/45", "Carbonatation : humidité modérée (extérieur abrité de la pluie)"),
    ("XC4", "XC4", "EE3", "C40/50", "Carbonatation : alternativement humide et sec (extérieur exposé à la pluie)"),
    ("XD1", "XD1/XS1", "ES2", "C40/50", "Chlorures hors eau de mer : humidité modérée"),
    ("XD2", "XD2/XS2", "ES1, ES3", "C40/50", "Chlorures hors eau de mer : humide, rarement sec (piscines, eaux industrielles)"),
    ("XD3", "XD3/XS3", "EE4, ES4", "C45/55", "Chlorures hors eau de mer : alternativement humide et sec (sels de déverglaçage, parkings)"),
    ("XS1", "XD1/XS1", "ES2", "C40/50", "Chlorures de l'eau de mer : air marin sans contact direct"),
    ("XS2", "XD2/XS2", "ES1, ES3", "C45/55", "Chlorures de l'eau de mer : immersion permanente"),
    ("XS3", "XD3/XS3", "EE4, ES4", "C45/55", "Chlorures de l'eau de mer : zones de marnage, embruns"),
)
EXPO_CLES = tuple(e[0] for e in EXPOSITIONS)
COLONNES_4_4N = ("X0", "XC1", "XC2/XC3", "XC4", "XD1/XS1", "XD2/XS2", "XD3/XS3")
LIBELLES_BELGES = {"X0": "—", "XC1": "EI", "XC2/XC3": "EE1, EE2", "XC4": "EE3", "XD1/XS1": "ES2", "XD2/XS2": "ES1, ES3", "XD3/XS3": "EE4, ES4"}
NOTE_XA = "XA (attaque chimique) et XF (gel-dégel) : prendre la plus grande valeur exigée pour les autres classes (NIT 260, tableau 20, note 1) — hors du moteur."

# ---------------------------------------------------------------------------------------------- tableau 4.4N (mm)
C_MIN_DUR = {
    "S1": (10, 10, 10, 15, 20, 25, 30),
    "S2": (10, 10, 15, 20, 25, 30, 35),
    "S3": (10, 10, 20, 25, 30, 35, 40),
    "S4": (10, 15, 25, 30, 35, 40, 45),
    "S5": (15, 20, 30, 35, 40, 45, 50),
    "S6": (20, 25, 35, 40, 45, 50, 55),
}
CLASSES_STRUCTURALES = tuple(C_MIN_DUR.keys())
S_REFERENCE = "S4"                   # §4.4.1.2(5) : durée d'utilisation de projet de 50 ans
DUREES = (50, 100)
C_MIN_PLANCHER = 10                  # mm, éq. (4.2)
DELTA_DEV = (
    (10, "10 mm : valeur recommandée (§4.4.1.3(1)P)"),
    (5, "5 mm : fabrication soumise à un système d'assurance qualité incluant la mesure de l'enrobage (§4.4.1.3(3))"),
    (0, "0 mm : éléments préfabriqués, mesure très précise et rejet des éléments non conformes (§4.4.1.3(3))"),
)
CONTRE_SOL = (("non", "non", 0), ("prepare", "sur béton de propreté (k1)", 40), ("direct", "directement contre le sol (k2)", 75))
DG_MAJORATION = 32                   # mm : au-delà de ce granulat, c_min,b + 5 mm (tableau 4.2)
ELEMENTS = (("poutre", "Poutre"), ("poteau", "Poteau"), ("dalle", "Dalle"), ("voile", "Voile"))

# ---------------------------------------------------------------------------------------------- feu (EN 1992-1-2)
R_DUREES = ("R30", "R60", "R90", "R120", "R180", "R240")
# poutres sur appuis simples, tableau 5.5 : R → [(b_min, a)…] (mm) ; au-delà de la plus grande largeur, a_sd n'est plus majoré
FEU_POUTRE = {
    "R30": ((80, 25), (120, 20), (160, 15), (200, 15)),
    "R60": ((120, 40), (160, 35), (200, 30), (300, 25)),
    "R90": ((150, 55), (200, 45), (300, 40), (400, 35)),
    "R120": ((200, 65), (240, 60), (300, 55), (500, 50)),
    "R180": ((240, 80), (300, 70), (400, 65), (600, 60)),
    "R240": ((280, 90), (350, 80), (500, 75), (700, 70)),
}
A_SD_ANGLE = 10                      # mm : a_sd = a + 10 pour les barres d'angle d'une poutre à un seul lit (tableau 5.5, note)
# dalles pleines portant dans une direction, tableau 5.8 : R → (h_s min, a) (mm)
FEU_DALLE = {"R30": (60, 10), "R60": (80, 20), "R90": (100, 30), "R120": (120, 40), "R180": (150, 55), "R240": (175, 65)}
FEU_A_TRANSCRIRE = {"poteau": "tableau 5.2a (méthode A, μ_fi = 0,7)", "voile": "tableau 5.4 (voiles porteurs, μ_fi = 0,7)"}

DEFAUTS = {"element": "poutre", "exposition": "XC1", "classe": "C30/37", "duree": 50, "dalle": None, "qualite": False,
           "phi": 20, "phi_etrier": 8, "dg": 20, "delta_dev": 10, "contre_sol": "non", "classe_structurale": None,
           "feu": None}


def _entier(x, nom, mini, maxi):
    if isinstance(x, bool):
        raise ValueError(f"{nom} : un nombre est attendu")
    try:
        v = int(round(float(str(x).replace(",", "."))))
    except ValueError:
        raise ValueError(f"{nom} : nombre illisible « {x} »")
    if not mini <= v <= maxi:
        raise ValueError(f"{nom} : entre {mini} et {maxi}")
    return v


def _bool(x) -> bool:
    return x.lower() in ("1", "true", "oui", "on") if isinstance(x, str) else bool(x)


def _fck(classe: str) -> int:
    from beton.bibliotheque.ancrages import fck_de
    return int(fck_de(classe))


def normaliser(params: dict) -> dict:
    p = {**DEFAUTS, **(params or {})}
    el = p["element"]
    if el not in dict(ELEMENTS):
        raise ValueError(f"élément : « {el} » (poutre, poteau, dalle ou voile)")
    if p["exposition"] not in EXPO_CLES:
        raise ValueError(f"classe d'exposition : « {p['exposition']} » n'est pas une classe de l'EN 206 portée ({', '.join(EXPO_CLES)})")
    out = {"element": el, "exposition": p["exposition"], "classe": str(p["classe"]),
           "duree": _entier(p["duree"], "durée d'utilisation", 50, 100), "qualite": _bool(p["qualite"]),
           "phi": _entier(p["phi"], "Ø de la barre", 6, 60), "phi_etrier": _entier(p["phi_etrier"], "Ø de l'étrier", 0, 40),
           "dg": _entier(p["dg"], "granulat", 4, 63), "delta_dev": _entier(p["delta_dev"], "Δc_dev", 0, 10),
           "contre_sol": p["contre_sol"], "classe_structurale": p["classe_structurale"], "feu": None}
    if out["duree"] not in DUREES:
        raise ValueError("durée d'utilisation : 50 ou 100 ans")
    if out["delta_dev"] not in tuple(d for d, _ in DELTA_DEV):
        raise ValueError("Δc_dev : 10, 5 ou 0 mm")
    if out["contre_sol"] not in tuple(k for k, _, _ in CONTRE_SOL):
        raise ValueError("coulé contre le sol : non, prepare ou direct")
    if out["classe_structurale"] not in (None, "") and out["classe_structurale"] not in CLASSES_STRUCTURALES:
        raise ValueError("classe structurale imposée : S1 à S6")
    if out["classe_structurale"] == "":
        out["classe_structurale"] = None
    # « élément assimilable à une dalle » : par défaut les dalles le sont (faces attaquées dans une direction), les autres non
    out["dalle"] = (el == "dalle") if p["dalle"] in (None, "") else _bool(p["dalle"])
    _fck(out["classe"])
    f = p["feu"]
    if f:
        r = f.get("r") if isinstance(f, dict) else None
        if r not in R_DUREES:
            raise ValueError(f"résistance au feu : « {r} » ({', '.join(R_DUREES)})")
        b = f.get("b_min") if isinstance(f, dict) else None
        out["feu"] = {"r": r, "b_min": None if b in (None, "") else _entier(b, "largeur b", 50, 3000)}
    return out


# ---------------------------------------------------------------------------------------------- durabilité
def colonne(exposition: str) -> str:
    return dict((e[0], e[1]) for e in EXPOSITIONS)[exposition]


def classe_structurale(p: dict) -> dict:
    """S4 et les modifications du tableau 4.3N : [{code, libellé, delta}], la classe résultante (bornée), ou la classe
    imposée si elle l'est."""
    fck = _fck(p["classe"])
    seuil = dict((e[0], e[3]) for e in EXPOSITIONS)[p["exposition"]]
    modifs = []
    if p["duree"] == 100:
        modifs.append({"code": "duree", "libelle": "durée d'utilisation de projet de 100 ans", "delta": +2})
    if fck >= _fck(seuil):
        modifs.append({"code": "resistance", "libelle": f"classe de résistance ≥ {seuil} pour {p['exposition']}", "delta": -1})
    if p["dalle"]:
        modifs.append({"code": "dalle", "libelle": "élément assimilable à une dalle (faces attaquées dans une seule direction)", "delta": -1})
    if p["qualite"]:
        modifs.append({"code": "qualite", "libelle": "maîtrise particulière de la qualité de production du béton", "delta": -1})
    # S4 + 2 − 3 au plus : les modifications du tableau 4.3N vont de S1 à S6, la borne de la norme ne joue jamais
    calculee = f"S{4 + sum(m['delta'] for m in modifs)}"
    imposee = p.get("classe_structurale")
    return {"reference": S_REFERENCE, "modifications": modifs, "calculee": calculee, "retenue": imposee or calculee, "imposee": bool(imposee)}


def c_min_dur(classe_struct: str, exposition: str) -> int:
    return C_MIN_DUR[classe_struct][COLONNES_4_4N.index(colonne(exposition))]


def durabilite(p: dict) -> dict:
    s = classe_structurale(p)
    cmin_b = p["phi"] + (5 if p["dg"] > DG_MAJORATION else 0)
    cmin_dur = c_min_dur(s["retenue"], p["exposition"])
    cmin = max(cmin_b, cmin_dur, C_MIN_PLANCHER)
    terme = "c_min,b" if cmin == cmin_b else ("c_min,dur" if cmin == cmin_dur else "plancher")
    cnom = cmin + p["delta_dev"]
    k_sol = dict((k, v) for k, _, v in CONTRE_SOL)[p["contre_sol"]]
    return {"classe_structurale": s, "c_min_b": cmin_b, "majoration_granulat": p["dg"] > DG_MAJORATION, "c_min_dur": cmin_dur,
            "plancher": C_MIN_PLANCHER, "c_min": cmin, "terme_c_min": terme, "delta_dev": p["delta_dev"], "c_nom": cnom,
            "contre_sol": k_sol, "c_nom_sol": max(cnom, k_sol)}


# ---------------------------------------------------------------------------------------------- feu
def variantes_feu(element: str, r: str) -> list:
    """Les couples (b_min, a) du tableau de l'élément pour la durée R ; [] pour un élément non transcrit."""
    if element == "poutre":
        return [{"b_min": b, "a": a} for b, a in FEU_POUTRE[r]]
    if element == "dalle":
        hs, a = FEU_DALLE[r]
        return [{"h_min": hs, "a": a}]
    return []


def feu(p: dict) -> dict:
    """La distance à l'axe et l'enrobage nominal qu'elle impose ; ``transcrit`` False pour poteau et voile."""
    f = p["feu"]
    if not f:
        return None
    el, r = p["element"], f["r"]
    if el in FEU_A_TRANSCRIRE:
        return {"r": r, "transcrit": False, "tableau": FEU_A_TRANSCRIRE[el], "a": None, "c_nom": None, "variantes": []}
    vs = variantes_feu(el, r)
    if el == "poutre":
        b = f["b_min"]
        choisie = next((v for v in vs if b is not None and v["b_min"] <= b), None) if b is not None else None
        if b is not None and choisie is None:
            # plus étroite que la plus petite largeur du tableau : la ligne la plus étroite, et un avertissement
            choisie, trop_etroite = vs[0], True
        else:
            trop_etroite = False
        if choisie is None:
            choisie = vs[0]
        # la plus grande largeur qui ne dépasse pas b : la ligne du tableau applicable (a décroît avec b)
        if b is not None and not trop_etroite:
            choisie = max((v for v in vs if v["b_min"] <= b), key=lambda v: v["b_min"])
        a = choisie["a"]
        angle = b is None or b < vs[-1]["b_min"]
        a_sd = a + A_SD_ANGLE if angle else a
        c_bas = a - p["phi"] / 2.0 - p["phi_etrier"]
        c_cote = a_sd - p["phi"] / 2.0 - p["phi_etrier"]
        return {"r": r, "transcrit": True, "tableau": "5.5", "b": b, "ligne": choisie, "variantes": vs, "trop_etroite": trop_etroite,
                "a": a, "a_sd": a_sd, "angle": angle, "c_bas": max(0.0, c_bas), "c_cote": max(0.0, c_cote),
                "c_nom": max(0.0, c_bas, c_cote)}
    hs, a = FEU_DALLE[r]
    c = a - p["phi"] / 2.0 - p["phi_etrier"]
    return {"r": r, "transcrit": True, "tableau": "5.8", "h_min": hs, "a": a, "variantes": vs, "c_nom": max(0.0, c)}


# ---------------------------------------------------------------------------------------------- tout
def calculer(params: dict) -> dict:
    p = normaliser(params)
    d = durabilite(p)
    f = feu(p)
    c_dur = d["c_nom_sol"]
    c_feu = f["c_nom"] if f and f.get("transcrit") else None
    c_nom = max(c_dur, c_feu) if c_feu is not None else c_dur
    gouv = "feu" if c_feu is not None and c_feu > c_dur else ("sol" if d["contre_sol"] > d["c_nom"] else "durabilite")
    res = {"entree": p, "durabilite": d, "feu": f, "c_nom_dur": c_dur, "c_nom_feu": c_feu, "c_nom": c_nom, "gouvernant": gouv}
    res["termes"] = termes(res)
    res["avertissements"] = avertissements(res)
    return res


def termes(res: dict) -> list:
    p, d, f = res["entree"], res["durabilite"], res["feu"]
    s = d["classe_structurale"]
    out = []

    def t(cle, symbole, nom, valeur, unite, regle):
        out.append({"cle": cle, "symbole": symbole, "nom": nom, "valeur": valeur, "unite": unite, "regle": regle})

    t("cmin_b", "c_min,b", "enrobage minimal vis-à-vis de l'adhérence", d["c_min_b"], "mm",
      "Ø de la barre isolée" + (" + 5 mm (granulat > 32 mm)" if d["majoration_granulat"] else "") + " — §4.4.1.2(3), tableau 4.2")
    mods = " ; ".join(f"{m['libelle']} ({m['delta']:+d})" for m in s["modifications"]) or "aucune modification"
    t("S", "S", "classe structurale", s["retenue"], "", (f"imposée ({s['calculee']} calculée)" if s["imposee"] else f"S4 (50 ans) ; {mods}") + " — §4.4.1.2(5), tableau 4.3N (NIT 260, tableau 21)")
    t("cmin_dur", "c_min,dur", "enrobage minimal vis-à-vis de la durabilité", d["c_min_dur"], "mm",
      f"tableau 4.4N [{s['retenue']} ; {colonne(p['exposition'])} ({LIBELLES_BELGES[colonne(p['exposition'])]})] — §4.4.1.2(5) (NIT 260, tableau 20)")
    t("cmin", "c_min", "enrobage minimal", d["c_min"], "mm", "max(c_min,b ; c_min,dur ; 10 mm) — §4.4.1.2(2), éq. (4.2), Δc_dur = 0")
    t("ddev", "Δc_dev", "tolérance d'exécution", d["delta_dev"], "mm", dict(DELTA_DEV)[d["delta_dev"]])
    t("cnom_dur", "c_nom,dur", "enrobage nominal, durabilité", d["c_nom"], "mm", "c_min + Δc_dev — §4.4.1.1(2), éq. (4.1)")
    if d["contre_sol"]:
        t("sol", "k", "coulé contre le sol", d["contre_sol"], "mm", "c_nom ≥ k1 = 40 mm sur béton de propreté, ≥ k2 = 75 mm contre le sol — §4.4.1.3(4)")
    if f and f.get("transcrit"):
        if p["element"] == "poutre":
            lg = f["ligne"]
            t("a", "a", "distance à l'axe des armatures principales", f["a"], "mm", f"tableau 5.5 ({f['r']}, b_min = {lg['b_min']} mm) — EN 1992-1-2 §5.6.2")
            if f["angle"]:
                t("asd", "a_sd", "barres d'angle (un seul lit)", f["a_sd"], "mm", "a + 10 mm — tableau 5.5, note")
            t("cfeu", "c_nom,feu", "enrobage nominal imposé par le feu", f["c_nom"], "mm", "max(a − Ø/2 − Ø_étrier ; a_sd − Ø/2 − Ø_étrier)")
        else:
            t("a", "a", "distance à l'axe des armatures", f["a"], "mm", f"tableau 5.8 ({f['r']}, h_s ≥ {f['h_min']} mm) — EN 1992-1-2 §5.7.3")
            t("cfeu", "c_nom,feu", "enrobage nominal imposé par le feu", f["c_nom"], "mm", "a − Ø/2 − Ø_étrier")
    t("cnom", "c_nom", "ENROBAGE NOMINAL RETENU", res["c_nom"], "mm", "max(c_nom,dur ; k_sol ; c_nom,feu) — "
      + {"feu": "le feu gouverne", "sol": "le coulage contre le sol gouverne", "durabilite": "la durabilité gouverne"}[res["gouvernant"]])
    return out


def avertissements(res: dict) -> list:
    p, d, f = res["entree"], res["durabilite"], res["feu"]
    out = []
    if f and not f.get("transcrit"):
        out.append({"code": "feu_a_transcrire", "message": f"Résistance au feu : le {f['tableau']} n'est pas transcrit dans la bibliothèque — à lire dans l'EN 1992-1-2."})
    if f and f.get("trop_etroite"):
        out.append({"code": "poutre_etroite", "message": "La poutre est plus étroite que la plus petite largeur du tableau 5.5 : la durée R n'est pas atteinte par la méthode tabulée."})
    if p["exposition"] == "X0":
        out.append({"code": "x0", "message": "X0 : béton non armé ou ambiance très sèche ; rare pour du béton armé."})
    return out


# ---------------------------------------------------------------------------------------------- les tableaux de l'écran
def tableau_c_min_dur() -> dict:
    return {"lignes": list(CLASSES_STRUCTURALES), "colonnes": list(COLONNES_4_4N), "belges": [LIBELLES_BELGES[c] for c in COLONNES_4_4N],
            "cellules": {s: list(v) for s, v in C_MIN_DUR.items()}, "source": SOURCE_DUR}


def tableau_feu(element: str) -> dict:
    if element == "poutre":
        return {"element": element, "tableau": "5.5", "durees": list(R_DUREES), "lignes": {r: [list(c) for c in FEU_POUTRE[r]] for r in R_DUREES},
                "source": SOURCE_FEU, "transcrit": True}
    if element == "dalle":
        return {"element": element, "tableau": "5.8", "durees": list(R_DUREES), "lignes": {r: list(FEU_DALLE[r]) for r in R_DUREES},
                "source": SOURCE_FEU, "transcrit": True}
    return {"element": element, "tableau": FEU_A_TRANSCRIRE[element], "durees": list(R_DUREES), "lignes": {}, "source": SOURCE_FEU, "transcrit": False}
