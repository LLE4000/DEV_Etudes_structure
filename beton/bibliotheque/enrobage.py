# -*- coding: utf-8 -*-
"""L'ENROBAGE NOMINAL aux valeurs de la NORME — EN 1992-1-1:2004 §4.4.1 (durabilité, classe structurale) et
EN 1992-1-2:2004 §5 (méthode tabulée, résistance au feu), les libellés belges de la NBN B 15-001 tels que la NIT 260
(CSTC, 2017) les met en regard des classes de l'EN 206. C'est le moteur de la rubrique « Enrobages » de la
bibliothèque (ses deux onglets, « Calcul » et « Tableaux ») ; l'ancien module « Enrobage » du site garde
``beton/enrobage`` (la copie de l'écran Streamlit, avec ses écarts de docs/beton/ENROBAGE_MOTEUR.md — depuis le
10/11/2026 sans D1, D2 ni D6 : ses tableaux du feu sont lus dans ``feu.py``) jusqu'à son retrait.

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

Feu (méthode tabulée de l'EN 1992-1-2:2004 section 5) : la distance à l'axe ``a`` des armatures principales et la
dimension minimale de l'élément sont lues dans ``beton/bibliotheque/feu.py`` — la SEULE source des tableaux du feu
(poteaux 5.2a et formule (5.7), voiles 5.3 et 5.4, poutres 5.5 et 5.6, dalles 5.8 et 5.9). L'enrobage nominal que le feu
impose à l'armature la plus extérieure est ::

    c_nom,feu = a − Ø/2 − Ø_ext        Ø_ext : le Ø de l'étrier pour une poutre ou un poteau (8 mm par défaut) ;
                                       pour une DALLE ou un VOILE, aucun étrier : le Ø de la nappe extérieure, s'il y
                                       en a une entre la barre et le parement (0 par défaut)

``a`` est une valeur NOMINALE (EN 1992-1-2 §5.2) : aucune tolérance Δc_dev ne s'y ajoute. Barres d'angle d'une poutre à
un seul lit : a_sd = a + 10 mm depuis la face latérale ; un seul enrobage étant prescrit pour les deux faces, c'est
a_sd qui le fixe. La distance à l'axe OBTENUE avec l'enrobage retenu est rendue elle aussi : a_réel = c_nom + Ø_ext +
Ø/2, à comparer à la distance exigée (et, pour un poteau lu par la formule (5.7), la durée R que cette distance donne).
Une dimension sous le minimum du tableau, ou une case vide (poteau R 240 à μ_fi = 0,7) : « durée non atteinte par la
méthode tabulée », dit par un avertissement grave — jamais une valeur inventée.
"""

from beton.bibliotheque import feu as FEU

SOURCE_DUR = "EN 1992-1-1:2004 §4.4.1 (tableaux 4.2, 4.3N, 4.4N) ; NBN EN 1992-1-1 ANB ; NBN B 15-001 (classes EI, EE, ES) via NIT 260 tableaux 20-21"
SOURCE_FEU = f"{FEU.NORME} {FEU.SECTION} — a est une valeur nominale, rien n'est interpolé"

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
# les éléments dont les armatures principales sont entourées d'un ÉTRIER (un cadre) : il se retire de la distance à
# l'axe. Une dalle ou un voile n'en a pas : seule une nappe extérieure, s'il y en a une, se retire (Ø_ext, 0 par défaut)
ELEMENTS_A_ETRIER = ("poutre", "poteau")

# ---------------------------------------------------------------------------------------------- feu (EN 1992-1-2)
R_DUREES = FEU.R_DUREES              # les tableaux eux-mêmes sont dans beton/bibliotheque/feu.py, et seulement là

DEFAUTS = {"element": "poutre", "exposition": "XC1", "classe": "C30/37", "duree": 50, "dalle": None, "qualite": False,
           "phi": 20, "phi_etrier": 8, "phi_ext": 0, "dg": 20, "delta_dev": 10, "contre_sol": "non", "classe_structurale": None,
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


def _n(x) -> str:
    """Un nombre dans une expression écrite (« 50 − 10 − 8 ») : entier s'il l'est, sinon au dixième, virgule décimale."""
    v = round(float(x), 1)
    return str(int(v)) if v == int(v) else f"{v:.1f}".replace(".", ",")


def normaliser(params: dict) -> dict:
    """Les paramètres relus et contrôlés (les absents prennent leur défaut ; hors domaine : ``ValueError``, le champ
    nommé). ``feu`` : None, ou {"r": "R30" … "R240", …les paramètres de l'élément pour ``feu.exigence``} — poutre :
    ``appuis``, ``b`` ; dalle : ``portee``, ``h`` ; poteau : ``methode``, ``exposition``, ``b``, ``mu_fi``, ``h``,
    ``l0_fi``, ``barres`` ; voile : ``porteur``, ``mu_fi``, ``faces``, ``e``."""
    p = {**DEFAUTS, **(params or {})}
    el = p["element"]
    if el not in dict(ELEMENTS):
        raise ValueError(f"élément : « {el} » (poutre, poteau, dalle ou voile)")
    if p["exposition"] not in EXPO_CLES:
        raise ValueError(f"classe d'exposition : « {p['exposition']} » n'est pas une classe de l'EN 206 portée ({', '.join(EXPO_CLES)})")
    out = {"element": el, "exposition": p["exposition"], "classe": str(p["classe"]),
           "duree": _entier(p["duree"], "durée d'utilisation", 50, 100), "qualite": _bool(p["qualite"]),
           "phi": _entier(p["phi"], "Ø de la barre", 6, 60), "phi_etrier": _entier(p["phi_etrier"], "Ø de l'étrier", 0, 40),
           "phi_ext": _entier(p["phi_ext"], "Ø de la nappe extérieure", 0, 40),
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
        if not isinstance(f, dict):
            raise ValueError("résistance au feu : la durée et les paramètres de l'élément sont attendus ({r, …})")
        r = f.get("r")
        if r not in R_DUREES:
            raise ValueError(f"résistance au feu : « {r} » ({', '.join(R_DUREES)})")
        reste = {k: v for k, v in f.items() if k != "r" and v is not None and v != ""}
        FEU.exigence(el, r, **reste)          # un paramètre inconnu ou hors domaine est refusé ici, le champ nommé
        out["feu"] = {"r": r, **reste}
    return out


def phi_exterieur(p: dict) -> int:
    """Le Ø de l'armature entre la barre principale et le parement : l'étrier d'une poutre ou d'un poteau ; pour une
    dalle ou un voile, la nappe extérieure s'il y en a une (0 par défaut) — jamais un étrier."""
    return p["phi_etrier"] if p["element"] in ELEMENTS_A_ETRIER else p["phi_ext"]


def symbole_exterieur(p: dict) -> str:
    return "Ø_étrier" if p["element"] in ELEMENTS_A_ETRIER else "Ø_ext"


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
def feu(p: dict) -> dict:
    """Ce que le feu demande (``feu.exigence`` : la combinaison retenue, a, a_sd, la dimension minimale, « atteinte »
    ou non, les notes, les conditions, la source) et l'enrobage nominal qu'il impose à l'armature la plus extérieure :
    ``c_bas`` = a − Ø/2 − Ø_ext, ``c_cote`` = a_sd − Ø/2 − Ø_ext (poutre : les barres d'angle), ``c_nom`` le plus grand
    des deux — ``a`` est une valeur nominale, Δc_dev ne s'y ajoute PAS. None quand aucune distance à l'axe n'est
    exigée (voile non porteur) ou que le tableau n'a pas de valeur (la durée est alors non atteinte)."""
    f = p["feu"]
    if not f:
        return None
    ex = FEU.exigence(p["element"], f["r"], **{k: v for k, v in f.items() if k != "r"})
    phi_ext = phi_exterieur(p)
    a, a_sd = ex["a"], ex["a_sd"]
    c_bas = c_cote = c = None
    brut = None
    if a is not None:
        c_bas = a - p["phi"] / 2.0 - phi_ext
        c_cote = None if a_sd is None else a_sd - p["phi"] / 2.0 - phi_ext
        brut = c_bas if c_cote is None else max(c_bas, c_cote)
        c = max(0.0, brut)
    return {**ex, "phi_ext": phi_ext, "c_bas": c_bas, "c_cote": c_cote, "c_brut": brut, "c_nom": c}


def _obtenu(res: dict) -> None:
    """La distance à l'axe OBTENUE avec l'enrobage retenu, face à la distance exigée ; pour un poteau lu par la formule
    (5.7), la durée R que cette distance donne."""
    p, f = res["entree"], res["feu"]
    exige = None if f["a"] is None else max(f["a"], f["a_sd"] or 0)
    f["a_reel"] = res["a_reel"]
    f["a_exige"] = exige
    f["symbole_exige"] = "a_sd" if (f["a_sd"] is not None and f["angle"]) else "a"
    f["marge"] = None if exige is None else res["a_reel"] - exige
    f["duree_obtenue"] = None
    if f["methode"] == "formule":
        pa = f["parametres"]
        if res["a_reel"] >= FEU.A_FORMULE[0]:
            f["duree_obtenue"] = FEU.poteau_duree(res["a_reel"], pa["b"], pa["h"], pa["l0_fi"], pa["mu_fi"], pa["barres"])
    del p


# ---------------------------------------------------------------------------------------------- tout
def calculer(params: dict) -> dict:
    p = normaliser(params)
    d = durabilite(p)
    f = feu(p)
    c_dur = d["c_nom_sol"]
    c_feu = f["c_nom"] if f else None
    c_nom = max(c_dur, c_feu) if c_feu is not None else c_dur
    gouv = "feu" if c_feu is not None and c_feu > c_dur else ("sol" if d["contre_sol"] > d["c_nom"] else "durabilite")
    phi_ext = phi_exterieur(p)
    res = {"entree": p, "durabilite": d, "feu": f, "c_nom_dur": c_dur, "c_nom_feu": c_feu, "c_nom": c_nom, "gouvernant": gouv,
           "phi_ext": phi_ext, "a_reel": c_nom + phi_ext + p["phi"] / 2.0}
    if f:
        _obtenu(res)
    res["termes"] = termes(res)
    res["avertissements"] = avertissements(res)
    return res


GOUVERNANTS = {"feu": "le feu gouverne", "sol": "le coulage contre le sol gouverne", "durabilite": "la durabilité gouverne"}
ETATS = {True: "atteinte", False: "non atteinte", None: "à vérifier"}


def termes(res: dict) -> list:
    """Chaque grandeur du calcul, dans l'ordre où on l'écrit à la main : ``groupe`` (« durabilite », « feu »,
    « retenu »), ``symbole``, ``nom``, ``valeur``, ``unite``, ``regle`` (la référence), et pour la feuille de calcul
    ``formule`` (l'expression en symboles) et ``valeurs`` (la même, valeurs substituées — recalculable telle quelle) ;
    ``brut`` : la valeur avant arrondi quand le terme est arrondi ; ``comparaison`` : {symbole, valeur, ok} quand le
    terme se compare à une autre grandeur (dimension minimale face à la dimension réelle, distance obtenue face à la
    distance exigée)."""
    p, d, f = res["entree"], res["durabilite"], res["feu"]
    s = d["classe_structurale"]
    out = []

    def t(cle, groupe, symbole, nom, valeur, unite, regle, formule="", valeurs="", **plus):
        out.append({"cle": cle, "groupe": groupe, "symbole": symbole, "nom": nom, "valeur": valeur, "unite": unite, "regle": regle,
                    "formule": formule, "valeurs": valeurs, **plus})

    # ---- durabilité
    t("cmin_b", "durabilite", "c_min,b", "enrobage minimal vis-à-vis de l'adhérence", d["c_min_b"], "mm",
      "Ø de la barre isolée" + (" + 5 mm (granulat > 32 mm)" if d["majoration_granulat"] else "") + " — §4.4.1.2(3), tableau 4.2",
      "Ø + 5" if d["majoration_granulat"] else "Ø", f"{p['phi']} + 5" if d["majoration_granulat"] else "")
    mods = " ; ".join(f"{m['libelle']} ({m['delta']:+d})" for m in s["modifications"]) or "aucune modification"
    t("S", "durabilite", "S", "classe structurale", s["retenue"], "",
      (f"imposée ({s['calculee']} calculée)" if s["imposee"] else f"S4 (50 ans) ; {mods}") + " — §4.4.1.2(5), tableau 4.3N (NIT 260, tableau 21)",
      "classe imposée" if s["imposee"] else "S4 + modifications du tableau 4.3N",
      "" if s["imposee"] else "S4" + "".join(f" {'+' if m['delta'] > 0 else '−'} {abs(m['delta'])}" for m in s["modifications"]))
    t("cmin_dur", "durabilite", "c_min,dur", "enrobage minimal vis-à-vis de la durabilité", d["c_min_dur"], "mm",
      f"tableau 4.4N [{s['retenue']} ; {colonne(p['exposition'])} ({LIBELLES_BELGES[colonne(p['exposition'])]})] — §4.4.1.2(5) (NIT 260, tableau 20)",
      f"tableau 4.4N [{s['retenue']} ; {colonne(p['exposition'])}]")
    t("cmin", "durabilite", "c_min", "enrobage minimal", d["c_min"], "mm", "max(c_min,b ; c_min,dur ; 10 mm) — §4.4.1.2(2), éq. (4.2), Δc_dur = 0",
      "max(c_min,b ; c_min,dur ; 10)", f"max({d['c_min_b']} ; {d['c_min_dur']} ; {d['plancher']})")
    t("ddev", "durabilite", "Δc_dev", "tolérance d'exécution", d["delta_dev"], "mm", dict(DELTA_DEV)[d["delta_dev"]])
    t("cnom_dur", "durabilite", "c_nom,dur", "enrobage nominal, durabilité", d["c_nom"], "mm", "c_min + Δc_dev — §4.4.1.1(2), éq. (4.1)",
      "c_min + Δc_dev", f"{d['c_min']} + {d['delta_dev']}")
    if d["contre_sol"]:
        t("sol", "durabilite", "k", "coulé contre le sol", d["contre_sol"], "mm",
          "c_nom ≥ k1 = 40 mm sur béton de propreté, ≥ k2 = 75 mm contre le sol — §4.4.1.3(4)",
          "c_nom ≥ k_1 (béton de propreté)" if d["contre_sol"] == 40 else "c_nom ≥ k_2 (contre le sol)")
    # ---- feu
    if f:
        _termes_feu(res, t)
    # ---- retenu
    parts = [("c_nom,dur", d["c_nom"])] + ([("k", d["contre_sol"])] if d["contre_sol"] else []) + ([("c_nom,feu", res["c_nom_feu"])] if res["c_nom_feu"] is not None else [])
    seul = len(parts) == 1
    t("cnom", "retenu", "c_nom", "ENROBAGE NOMINAL RETENU", res["c_nom"], "mm",
      "max(c_nom,dur ; k_sol ; c_nom,feu) — " + GOUVERNANTS[res["gouvernant"]],
      parts[0][0] if seul else "max(" + " ; ".join(n for n, _ in parts) + ")",
      "" if seul else "max(" + " ; ".join(_n(v) for _, v in parts) + ")")
    sym_ext = symbole_exterieur(p)
    ext = res["phi_ext"]
    cmp_ = None
    if f and f["a_exige"] is not None:
        cmp_ = {"symbole": f["symbole_exige"], "valeur": f["a_exige"], "sens": "≥", "ok": f["marge"] >= -1e-9}
    t("a_reel", "retenu", "a_réel", "distance à l'axe obtenue avec l'enrobage retenu", res["a_reel"], "mm",
      "du parement à l'axe de la barre principale : c_nom" + (f" + {sym_ext}" if ext else "") + " + Ø/2"
      + (f" — à comparer à la distance exigée {f['symbole_exige']} = {_n(f['a_exige'])} mm" if cmp_ else ""),
      "c_nom" + (f" + {sym_ext}" if ext else "") + " + Ø/2",
      f"{_n(res['c_nom'])}" + (f" + {ext}" if ext else "") + f" + {_n(p['phi'] / 2.0)}", comparaison=cmp_)
    if f and f.get("duree_obtenue"):
        do = f["duree_obtenue"]
        t("R_obtenue", "retenu", "R", "durée de résistance obtenue avec l'enrobage retenu", do["R"], "min",
          f"formule (5.7) avec a = {_n(do['a_calcul'])} mm — EN 1992-1-2 §5.3.2(4)" + (f" : classe {FEU.libelle_duree(do['classe'])}" if do["classe"] else " : sous R 30"),
          "120·[(R_ηfi + R_a + R_l + R_b + R_n)/120]^1,8",
          f"120·[({_n(do['R_eta'])} + {_n(do['R_a'])} + {_n(do['R_l'])} + {_n(do['R_b'])} + {_n(do['R_n'])})/120]^1,8",
          comparaison={"symbole": "R", "valeur": f["minutes"], "sens": "≥", "ok": do["R"] >= f["minutes"] - 1e-9})
    return out


def _termes_feu(res: dict, t) -> None:
    p, f = res["entree"], res["feu"]
    sym_ext = symbole_exterieur(p)
    ext = res["phi_ext"]
    num, duree, ref = f["tableau"], f["duree"], f"{FEU.NORME} {f['clause']}"
    sd = f["symbole_dimension"]
    if f["methode"] == "formule":
        fo = f["formule"]
        barres = "4 barres" if fo["barres"] == "4" else "plus de 4 barres"
        t("R_eta", "feu", "R_ηfi", "terme du taux de chargement", fo["R_eta"], "",
          "avec α_cc = 0,85 (annexe belge de l'EN 1992-1-1), (0,85/α_cc) = 1 : le terme en ω se simplifie, R_ηfi = 83·(1 − μ_fi) — " + ref,
          "83·[1,00 − μ_fi·(1 + ω)/((0,85/α_cc) + ω)]", f"83·(1 − {_n2(fo['mu_calcul'])})")
        t("R_l", "feu", "R_l", "terme de la longueur efficace au feu", fo["R_l"], "", "l_0,fi en mètres, de 2 à 6 m — " + ref,
          "9,60·(5 − l_0,fi)", f"9,60·(5 − {_n2(fo['l0_calcul'])})")
        t("b_prime", "feu", "b′", "largeur équivalente du poteau", fo["b_prime"], "mm", "2·A_c/(b + h) ; le diamètre d'un poteau rond ; de 200 à 450 mm — " + ref,
          "2·b·h/(b + h)", f"2·{_n(fo['b'])}·{_n(fo['h'])}/({_n(fo['b'])} + {_n(fo['h'])})")
        t("R_b", "feu", "R_b", "terme de la largeur", fo["R_b"], "", ref, "0,09·b′", f"0,09·{_n(fo['b_prime'])}")
        t("R_n", "feu", "R_n", "terme du nombre de barres", fo["R_n"], "", f"0 pour 4 barres (aux angles), 12 pour plus de 4 — ici {barres} — " + ref)
        t("R_exige", "feu", "ΣR", f"somme exigée pour {duree}", fo["somme_exigee"], "", "l'équation (5.7) inversée — " + ref,
          "120·(R/120)^(1/1,8)", f"120·({fo['minutes']}/120)^(1/1,8)")
        t("a", "feu", "a", "distance à l'axe exigée", fo["a"], "mm",
          f"valeur calculée {_n(fo['a_brut'])} mm, arrondie au millimètre supérieur, jamais sous 25 mm ; au-delà de 80 mm la formule ne vaut plus — " + ref,
          "30 + (ΣR − R_ηfi − R_l − R_b − R_n)/1,60",
          f"30 + ({_n(fo['somme_exigee'])} − {_n(fo['R_eta'])} − {_n(fo['R_l'])} − {_n(fo['R_b'])} − {_n(fo['R_n'])})/1,60",
          brut=fo["a_brut"], etat=ETATS[f["atteinte"]])
        t("comparaison", "feu", "5.2a", "tableau 5.2a de l'EN, pour comparaison", f["comparaison"]["texte"], "mm" if f["comparaison"]["dimension_min"] else "",
          f["comparaison"]["phrase"] + " — établi avec α_cc = 1,0 : la lecture par défaut, la plus sévère")
    else:
        if f["retenue"] is None:
            t("combinaison", "feu", f"{sd}/a", "combinaison du tableau", "aucune valeur", "", f"{f['raison']} — {ref}", f"tableau {num} [{duree}]", etat=ETATS[False])
            return
        choix = (f"la plus favorable que {sd} = {_n(f['dimension'])} mm autorise" if f["atteinte"] else
                 (f"la première ({f['nom_dimension']} non donnée)" if f["atteinte"] is None else f"la première : {f['raison']}"))
        seule = len(f["combinaisons"]) == 1
        t("combinaison", "feu", f"{sd}_min/a" if f["a"] is not None and sd != "h_s" else (f"{sd}/a" if f["a"] is not None else f"{sd}_min"),
          "combinaison retenue (dimension minimale / distance à l'axe)" if f["a"] is not None else "épaisseur minimale",
          f["combinaison"], "mm",
          f"tableau {num}, {duree}, {f['colonne_titre']}" + ("" if seule else f" : {choix}")
          + (" — ** au moins 8 barres" if f["huit_barres"] else "") + f" — {ref}",
          f"tableau {num} [{duree}]")
        sym_min = sd if sd == "h_s" else f"{sd}_min"
        reel = "h" if sd == "h_s" else sd
        t("dimension", "feu", sym_min, f"{f['nom_dimension']} minimale exigée", f["dimension_min"], "mm",
          f"tableau {num}, {duree} — {ref}", etat=ETATS[f["atteinte"]],
          comparaison={"symbole": reel, "valeur": f["dimension"], "sens": "≤", "ok": f["atteinte"]})
        if f["a"] is None:
            return
        t("a", "feu", "a", "distance à l'axe exigée", f["a"], "mm",
          f"du parement exposé à l'axe des armatures principales, valeur nominale — tableau {num}"
          + (" ; * l'enrobage de l'EN 1992-1-1 gouverne normalement" if f["etoile"] else "") + f" — {ref}")
        if f["a_sd"] is not None:
            if f["angle"]:
                t("asd", "feu", "a_sd", "barres d'angle, depuis la face latérale (un seul lit)", f["a_sd"], "mm",
                  f"a + 10 mm jusqu'à b = {f['largeur_a_sd']} mm compris ({FEU.COLONNE_A_SD[num]}) — EN 1992-1-2 §5.6.1(8)", "a + 10", f"{f['a']} + {FEU.A_SD_ANGLE}")
            else:
                t("asd", "feu", "a_sd", "barres d'angle, depuis la face latérale (un seul lit)", f["a_sd"], "mm",
                  f"aucune majoration : b dépasse {f['largeur_a_sd']} mm ({FEU.COLONNE_A_SD[num]}) — EN 1992-1-2 §5.6.1(8)", "a")
    if f["c_nom"] is None:
        return
    pilote = "a_sd" if (f["a_sd"] is not None and f["angle"]) else "a"
    exige = f["a_exige"]
    expr = f"{pilote} − Ø/2" + (f" − {sym_ext}" if ext else "")
    vals = f"{_n(exige)} − {_n(p['phi'] / 2.0)}" + (f" − {ext}" if ext else "")
    borne = f["c_brut"] < 0
    t("cfeu", "feu", "c_nom,feu", "enrobage nominal imposé par le feu", f["c_nom"], "mm",
      "à l'armature la plus extérieure ; a est nominale : Δc_dev ne s'y ajoute pas — EN 1992-1-2 §5.2"
      + ("" if ext or p["element"] in ELEMENTS_A_ETRIER else " ; dalle ou voile : aucun étrier à retirer"),
      f"max({expr} ; 0)" if borne else expr, f"max({vals} ; 0)" if borne else vals)


def _n2(x) -> str:
    """Un coefficient saisi dans une expression (μ_fi = 0,35 ; l_0,fi = 2,75) : au centième, sans zéro inutile."""
    return f"{float(x):.2f}".rstrip("0").rstrip(".").replace(".", ",")


def avertissements(res: dict) -> list:
    """Ce qui doit se lire à côté de la valeur : [{code, message, grave}]. ``grave`` : la durée de résistance au feu
    n'est PAS atteinte par la méthode tabulée."""
    p, f = res["entree"], res["feu"]
    out = []

    def a(code, message, grave=False):
        out.append({"code": code, "message": message, "grave": grave})

    if f:
        quoi = "la formule (5.7)" if f["methode"] == "formule" else "la méthode tabulée"
        if f["atteinte"] is False:
            a("feu_non_atteint", f"Résistance au feu {f['duree']} : durée non atteinte par {quoi} — {f['raison']}.", True)
        elif f["atteinte"] is None:
            a("feu_dimension", f"Résistance au feu {f['duree']} : {f['nom_dimension']} non donnée — la première combinaison du tableau "
                               f"{f['tableau']} est retenue ; {f['symbole_dimension']} ≥ {f['dimension_min']} mm reste à vérifier.")
        if f["huit_barres"]:
            a("huit_barres", f"Tableau 5.2a, combinaison {f['combinaison']} : au moins 8 barres longitudinales.")
        if f["a"] is None and f["retenue"] is not None:
            a("feu_sans_a", f"Tableau {f['tableau']} ({f['titre'][0].lower()}{f['titre'][1:]}) : l'épaisseur seule est exigée, aucune distance à l'axe.")
        if res["a_reel"] >= FEU.A_PEAU:
            a("armature_peau", f"Distance à l'axe de {_n(res['a_reel'])} mm, soit {FEU.A_PEAU} mm ou plus : armature de peau à prévoir "
                               "contre l'éclatement du béton (EN 1992-1-2 §4.5.2).")
        for m in f["mentions"]:
            a("feu_mention", m)
        if f["conditions"]:
            a("feu_conditions", ("Conditions d'emploi de la formule (5.7) : " if f["methode"] == "formule" else f"Conditions d'emploi du tableau {f['tableau']} : ")
              + " ; ".join(f["conditions"]) + ".")
    if p["exposition"] == "X0":
        a("x0", "X0 : béton non armé ou ambiance très sèche ; rare pour du béton armé.")
    return out


# ---------------------------------------------------------------------------------------------- les tableaux de l'écran
def tableau_c_min_dur() -> dict:
    return {"lignes": list(CLASSES_STRUCTURALES), "colonnes": list(COLONNES_4_4N), "belges": [LIBELLES_BELGES[c] for c in COLONNES_4_4N],
            "cellules": {s: list(v) for s, v in C_MIN_DUR.items()}, "source": SOURCE_DUR}


def tableau_feu(element: str, **parametres) -> dict:
    """Le tableau du feu de l'élément et de sa variante (``feu.tableau`` : la source unique)."""
    return FEU.tableau(element, **parametres)
