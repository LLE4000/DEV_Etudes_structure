# -*- coding: utf-8 -*-
"""Rigidité du sol — ce que l'ÉCRAN Streamlit ``modules/rigidite_sol.py`` v5.0 calcule, sans Streamlit.

L'écran tient son état dans ``st.session_state`` (clés ``rs_*``, ``snd<id>_*``) et calcule dans ses fonctions
d'affichage (``_resultats_profil``, ``_resultats_theories``…). Ce fichier en est l'EXTRACTION : mêmes expressions, dans
le même ordre d'opérations, sur un état écrit en dictionnaires. tests/test_sol_moteur.py rejoue l'écran avec un faux
« st » et compare chaque nombre affiché et chaque série tracée, au bit près.

Une FONDATION (un élément de la note du site ; dans Streamlit, tout l'état du module) :

    {"nom": "F1", "mode": "cpt" | "profil" | "rapide" | "theories" | "abaque" | "plat",
     # modes 1 et 2 (« Fondation », « Options de calcul ») — m, kPa, %
     "B": 2.0, "L": 2.0, "D": 1.0, "q": 150.0, "nappe_active": false, "nappe": 3.0,
     "critere": 20, "q_net": true, "nu": 0.30, "position": "centre",
     "sondages": [{"id": 1, "nom": "CPT01", "source": "", "nappe": null,
                   "points": [[z_m, qc_MPa, fs_kPa | null], …],                      # mesures brutes (après import)
                   "couches": [{"h": 2.0, "type": "Limon (loess)", "qc": 2.0, "rf": 2.0, "M": 22.0, "gamma": 18.0,
                                "M_haut": null, "Ic": null, "sbt": null}, …],       # depuis le TERRAIN NATUREL
                   "import": null | {…},  # un essai lu, en attente de « Remplir le tableau » (voir lire_essai)
                   "colle": ""}],         # le tableau collé (site), tel quel
     # mode 3 · mode 4 · mode 5 · mode 6
     "rapide": {"q": 150.0, "w": 20.0},
     "theories": {"E": 25.0, "B": 2.0, "L": 2.0, "nu": 0.30},
     "abaque": {"B": 2.0, "w_ref": 20.0},
     "plat": {"B": 200.0, "L": 200.0, "alpha": 0.5, "E": 30.0, "mortier": false, "tg": 20.0, "Eg": 20.0}}

Unités : longueurs en m (plat en béton : mm), pressions en kPa, modules en MPa (béton et mortier : GPa), qc en MPa, fs
en kPa, Rf en %, γ en kN/m³, k en MN/m³ (et kN/m³). Les résultats sont rendus NON ARRONDIS : l'affichage arrondit.
"""
import copy
import math

from . import base as SB
from . import imports as SI
from . import theorie as ST

VERSION = "v5.0"
REFERENCE = "modules/rigidite_sol.py v5.0 (sol_theorie v1.0, sol_base v2.0, sol_import v1.0)"

# les six modes de l'écran (« Que veux-tu faire ? »), dans son ordre et avec ses libellés
MODES = (("cpt", "1. Sondage CPT (import de fichier)"), ("profil", "2. Profil de sol saisi à la main"),
         ("rapide", "3. Vérification rapide k = q / w"), ("theories", "4. Comparer les théories"),
         ("abaque", "5. Abaque des sols"), ("plat", "6. Raideur d'un plat en béton"))
LIBELLES_MODES = dict(MODES)
CRITERES = (10, 15, 20, 25, 30)                       # select_slider « Critère de profondeur d'influence » (%)
SEUILS_IC = tuple(round(0.10 + 0.05 * i, 2) for i in range(11))   # slider « Finesse du découpage » 0,10 → 0,60
POSITIONS = ST.POSITIONS
TYPES = tuple(SB.soil_types_list())                   # « — » puis la base des sols
SANS_TYPE = ("—", "Personnalisé")

# le profil de départ d'un sondage (_add_sounding) : volontairement PROFOND, un profil court tronquerait le tassement
COUCHES_DEFAUT = ((2.0, "Limon (loess)", 2.0, 2.0, 22.0, 18.0),
                  (4.0, "Sable moyennement compact", 8.5, 0.5, 37.0, 19.0),
                  (14.0, "Argile ferme / raide", 3.5, 2.5, 37.0, 19.0))
# les défauts de l'écran (_init_state et la valeur initiale de chaque widget)
FONDATION_DEFAUT = {"B": 2.0, "L": 2.0, "D": 1.0, "q": 150.0, "nappe_active": False, "nappe": 3.0, "critere": 20,
                    "q_net": True, "nu": 0.30, "position": "centre"}
RAPIDE_DEFAUT = {"q": 150.0, "w": 20.0}
THEORIES_DEFAUT = {"E": 25.0, "B": 2.0, "L": 2.0, "nu": 0.30}
ABAQUE_DEFAUT = {"B": 2.0, "w_ref": 20.0}
PLAT_DEFAUT = {"B": 200.0, "L": 200.0, "alpha": 0.5, "E": 30.0, "mortier": False, "tg": 20.0, "Eg": 20.0}
IMPORT_DEFAUT = {"seuil": 0.25, "h_min": 0.50}

# Les BORNES des champs : le min_value / max_value de chaque widget de l'écran. Streamlit ramène une saisie hors bornes
# à la borne sans le dire ; le site la REFUSE avec un message (docs/sol/RIGIDITE_SOL_MOTEUR.md).
BORNES = {
    "B": (0.2, None), "L": (0.2, None), "D": (0.0, None), "q": (1.0, None), "nappe": (0.0, None), "nu": (0.0, 0.49),
    "couche.h": (0.0, None), "couche.qc": (0.0, None), "couche.rf": (0.0, None), "couche.M": (0.0, None),
    "couche.gamma": (8.0, 26.0),
    "import.nappe": (0.0, None), "import.seuil": (0.10, 0.60), "import.h_min": (0.20, 5.0),
    "rapide.q": (0.0, None), "rapide.w": (0.01, None),
    "theories.E": (0.5, None), "theories.B": (0.2, None), "theories.L": (0.2, None), "theories.nu": (0.0, 0.49),
    "abaque.B": (0.3, None), "abaque.w_ref": (1.0, 100.0),
    "plat.B": (20.0, None), "plat.L": (20.0, None), "plat.alpha": (0.05, None), "plat.E": (5.0, None),
    "plat.tg": (1.0, None), "plat.Eg": (1.0, None),
}
# Au-delà des bornes de l'écran, un PLAFOND DE VRAISEMBLANCE (site seulement) : aucune donnée de ce module n'approche
# 10⁶ dans son unité (q de 1 GPa, M de 1 000 GPa, B de 1 000 km…). Streamlit n'a pas de borne haute sur la plupart des
# champs : une saisie comme 1e308 y donne « inf » ou une exception ; le site la refuse avec un message.
VRAISEMBLANCE = 1.0e6
NOMS_CHAMPS = {
    "B": "Largeur B", "L": "Longueur L", "D": "Assise D", "q": "Pression q", "nappe": "Nappe", "nu": "ν (Poisson)",
    "couche.h": "Épaisseur h", "couche.qc": "qc", "couche.rf": "Rf", "couche.M": "Module M", "couche.gamma": "γ",
    "import.nappe": "Nappe", "import.seuil": "Finesse du découpage", "import.h_min": "Épaisseur minimale",
    "rapide.q": "q", "rapide.w": "w", "theories.E": "E", "theories.B": "B", "theories.L": "L", "theories.nu": "ν",
    "abaque.B": "Largeur de fondation B", "abaque.w_ref": "Tassement de référence",
    "plat.B": "B", "plat.L": "L", "plat.alpha": "α", "plat.E": "E béton", "plat.tg": "Épaisseur du mortier",
    "plat.Eg": "E mortier",
}


# ------------------------------------------------------------------ l'état : couches, sondages, fondation
def nouvelle_couche(h=1.0, type_sol="—", qc=0.0, rf=0.0, M=0.0, gamma=19.0) -> dict:
    """Une couche telle que _init_layer la pose (les clés d'import M_haut, Ic, sbt à None)."""
    return {"h": float(h), "type": type_sol, "qc": float(qc), "rf": float(rf), "M": float(M), "gamma": float(gamma),
            "M_haut": None, "Ic": None, "sbt": None}


def nouveau_sondage(ident: int = 1, nom=None, premiere_couche: bool = True) -> dict:
    """Un sondage tel que _add_sounding le crée : « CPT<id> » et le profil de départ profond (20 m)."""
    s = {"id": int(ident), "nom": nom or f"CPT{int(ident):02d}", "source": "", "nappe": None, "points": [], "couches": [],
         "import": None, "colle": ""}
    if premiere_couche:
        s["couches"] = [nouvelle_couche(h=h, type_sol=t, qc=qc, rf=rf, M=M, gamma=g) for h, t, qc, rf, M, g in COUCHES_DEFAUT]
    return s


def nouvelle_fondation(nom: str = "F1") -> dict:
    """La note neuve de l'écran : mode 1 (sondage CPT), fondation 2 × 2 m à 1 m sous le TN, q = 150 kPa, CPT01."""
    f = {"nom": nom, "mode": "cpt"}
    f.update(copy.deepcopy(FONDATION_DEFAUT))
    f["sondages"] = [nouveau_sondage(1, "CPT01")]
    f["rapide"], f["theories"] = dict(RAPIDE_DEFAUT), dict(THEORIES_DEFAUT)
    f["abaque"], f["plat"] = dict(ABAQUE_DEFAUT), dict(PLAT_DEFAUT)
    return f


def id_sondage_suivant(f: dict) -> int:
    """_new_sounding_id : le plus grand identifiant + 1 (un identifiant n'est jamais réutilisé tant qu'il existe)."""
    ids = [int(s["id"]) for s in f["sondages"]]
    return (max(ids) + 1) if ids else 1


def ajouter_sondage(f: dict) -> dict:
    s = nouveau_sondage(id_sondage_suivant(f))
    f["sondages"].append(s)
    return s


def copier_sondage(f: dict, i: int) -> dict:
    """_copy_sounding : « <nom> (copie) », mêmes couches (toutes leurs clés), même nappe, mêmes mesures, même source."""
    src = f["sondages"][i]
    s = nouveau_sondage(id_sondage_suivant(f), f"{src['nom']} (copie)", premiere_couche=False)
    s["couches"] = copy.deepcopy(src["couches"])
    for cle in ("nappe", "points", "source"):
        s[cle] = copy.deepcopy(src[cle])
    f["sondages"].append(s)
    return s


def supprimer_sondage(f: dict, i: int) -> bool:
    """_delete_sounding : jamais le dernier."""
    if len(f["sondages"]) <= 1:
        return False
    del f["sondages"][i]
    return True


def ajouter_couche(s: dict) -> None:
    """_add_layer : une couche par défaut (h = 1 m, type « — », M = 0) EN BAS du profil."""
    s["couches"].append(nouvelle_couche())


def supprimer_couche(s: dict, j: int) -> bool:
    """_delete_layer : jamais la dernière."""
    if len(s["couches"]) <= 1:
        return False
    del s["couches"][j]
    return True


def appliquer_type(c: dict, nouveau: str) -> None:
    """Le préremplissage de _render_layer_row quand le TYPE d'une couche change : qc, Rf (sols où le CPT a un sens) et
    M au milieu des plages de la base, SEULEMENT si qc et M sont encore nuls ; γ du type, TOUJOURS."""
    if nouveau == c["type"]:
        return
    if nouveau not in SANS_TYPE and float(c.get("qc", 0) or 0) <= 0 and float(c.get("M", 0) or 0) <= 0:
        d = SB.SOIL_DB.get(nouveau, {})
        if d.get("cpt_ok"):
            q = SB.soil_default_qc(nouveau)
            if q is not None:
                c["qc"] = q
            r = SB.soil_default_Rf(nouveau)
            if r is not None:
                c["rf"] = r
        m = SB.soil_default_M(nouveau)
        if m is not None:
            c["M"] = m
    c["gamma"] = SB.soil_gamma(nouveau, 19.0)
    c["type"] = nouveau


def lire_couche(c: dict) -> dict:
    """_get_layer : les valeurs nulles lues comme l'écran les lit (h, qc, Rf, M → 0 ; γ → 19)."""
    g = c.get("gamma", 19.0)
    return {"h": float(c.get("h", 0.0) or 0.0), "type": c.get("type", "—"), "qc": float(c.get("qc", 0.0) or 0.0),
            "rf": float(c.get("rf", 0.0) or 0.0), "M": float(c.get("M", 0.0) or 0.0), "gamma": float(g or 19.0)}


def infos_couche(c: dict) -> dict:
    """Ce que la légende de chaque ligne de couche dit (_render_layer_row) : prise en compte, Ic et classe CPT, M
    encadré par l'import, ou M suggéré d'après qc (M = α·qc converti en module oedométrique, ν = 0,30), ou roche."""
    lv = lire_couche(c)
    out = {"prise_en_compte": lv["h"] > 0 and lv["M"] > 0, "Ic": c.get("Ic"), "sbt": c.get("sbt"), "M_encadre": None,
           "M_suggere": None, "roche": False}
    Mh = c.get("M_haut")
    if Mh and lv["M"] and Mh > lv["M"]:
        out["M_encadre"] = (lv["M"], Mh)
    elif lv["type"] not in SANS_TYPE:
        sug = SB.suggest_M_from_qc(lv["qc"], lv["type"])
        if sug:
            out["M_suggere"] = sug
        elif SB.is_rock(lv["type"]):
            out["roche"] = True
    return out


def profil_encadre(s: dict):
    """_profil_encadre : (couches_bas, couches_haut) au format de theorie.tassement, depuis le terrain naturel ; une
    couche d'épaisseur nulle est écartée ; M nul → None (couche SANS module) ; M haut = M_haut de l'import, sinon M."""
    bas, haut = [], []
    for c in s["couches"]:
        lv = lire_couche(c)
        if lv["h"] <= 0:
            continue
        Mb = lv["M"] or None
        Mh = c.get("M_haut") or Mb
        base = {"h": lv["h"], "gamma": lv["gamma"], "nom": lv["type"]}
        bas.append(dict(base, M=Mb))
        haut.append(dict(base, M=Mh))
    return bas, haut


# ------------------------------------------------------------------ l'import d'un essai
def lire_essai(nom_fichier: str, texte: str, nappe_defaut: float) -> dict:
    """Un fichier d'essai (GEF, CSV, TXT, ASC) lu comme l'écran le lit (SI.importer sur ses octets), prêt pour
    l'aperçu : nom, source, points, avertissements, et les trois réglages du découpage à leurs valeurs initiales —
    nappe du fichier, sinon celle de la fondation (« rs_nappe », même si la nappe n'y est pas cochée), finesse 0,25,
    épaisseur minimale 0,50 m. ValueError (message de l'analyseur) si le fichier est illisible."""
    s = SI.importer(str(texte).encode("utf-8"), str(nom_fichier))
    return _attente(s, nappe_defaut, "fichier")


def lire_tableau_colle(texte: str, nom: str, nappe_defaut: float) -> dict:
    """Un tableau collé (site seulement : l'écran n'a pas de zone de collage). Le MÊME analyseur que le CSV
    (SI.parse_csv) : un en-tête reconnu (« Profondeur ; qc ; fs ») ; sans en-tête reconnu, les colonnes sont lues dans
    l'ordre profondeur, qc (MPa), fs (kPa) — trois colonnes au moins — puis profondeur, qc s'il n'y en a que deux."""
    octets = str(texte).encode("utf-8")
    try:
        s = SI.parse_csv(octets, nom)
    except ValueError as e:
        err = e
        s = None
        for mapping in ({"z": 0, "qc": 1, "fs": 2}, {"z": 0, "qc": 1}):
            try:
                s = SI.parse_csv(octets, nom, mapping=mapping)
                break
            except ValueError as e2:
                err = e2
        if s is None:
            raise err
        s["source"] = s["source"].replace("CSV", "Tableau collé")
    else:
        s["source"] = s["source"].replace("CSV", "Tableau collé")
    return _attente(s, nappe_defaut, "colle")


def _attente(s: dict, nappe_defaut: float, origine: str) -> dict:
    return {"nom": s["nom"], "source": s["source"], "points": [[z, q, f] for z, q, f in s["points"]],
            "avertissements": list(s.get("avertissements", [])),
            "nappe": float(s.get("nappe_m") or nappe_defaut), "seuil": IMPORT_DEFAUT["seuil"],
            "h_min": IMPORT_DEFAUT["h_min"], "origine": origine}


def decoupage_impossible(h_min: float) -> str:
    return (f"Le découpage automatique échoue sur ce sondage : ses dernières mesures, sur moins de {h_min:g} m".replace(".", ",")
            + ", n'ont pas d'indice Ic (frottement nul ou absent, ou qc inférieur à la contrainte en place). L'écran Streamlit "
              "s'arrête sur une erreur au même endroit (sol_theorie.profil_depuis_cpt). Réduire l'épaisseur minimale, ou retirer "
              "ces dernières lignes du fichier.")


def decouper(points, nappe, seuil=0.25, h_min=0.50, nu=0.30) -> list:
    """profil_depuis_cpt, tel quel ; là où il PLANTE (TypeError : un dernier groupe de mesures sans Ic, plus mince que
    h_min, comparé à son voisin — défaut de sol_theorie v1.0, l'écran affiche alors une exception), une ValueError qui
    le dit. Le calcul n'est pas corrigé : il n'y a pas de résultat, ici comme dans l'écran."""
    try:
        return ST.profil_depuis_cpt(points, nappe_m=nappe, seuil_ic=seuil, h_min=h_min, nu=nu)
    except TypeError as e:
        raise ValueError(decoupage_impossible(h_min)) from e


def apercu_import(imp: dict, nu: float) -> dict:
    """_apercu_et_valider : le découpage automatique en couches (profil_depuis_cpt avec la nappe, la finesse et
    l'épaisseur minimale de l'aperçu, ν de la fondation), les lignes du tableau d'aperçu (arrondies comme l'écran), la
    présence de fs. ``couches`` vide : « Aucune couche exploitable n'a pu être formée »."""
    pts = [tuple(p) for p in imp["points"]]
    couches = decouper(pts, imp["nappe"], imp["seuil"], imp["h_min"], float(nu))
    lignes = [{"de": round(c["z0"], 2), "a": round(c["z1"], 2), "h": round(c["h"], 2),
               "Ic": None if c["Ic"] is None else round(c["Ic"], 2), "type": c["sbt"], "qc": round(c["qc"] or 0, 2),
               "M_bas": None if c["M_bas"] is None else round(c["M_bas"], 1),
               "M_haut": None if c["M_haut"] is None else round(c["M_haut"], 1)} for c in couches]
    return {"couches": couches, "lignes": lignes, "a_fs": any(p[2] is not None for p in pts),
            "z0": pts[0][0] if pts else None, "z1": pts[-1][0] if pts else None, "n": len(pts)}


def couches_depuis_cpt(couches: list) -> list:
    """_remplir_depuis_couches : chaque couche découpée devient une ligne du tableau — type de la base reconnu d'après
    la classe de Robertson (match_soil_type), Rf = fs/qc, M = M bas ARRONDI au dixième (M haut gardé tel quel), h et qc
    au centième, γ au dixième. Aucune couche : une couche par défaut."""
    out = []
    for c in couches:
        libelle = c.get("sbt") or c.get("type") or "—"
        type_db = SB.match_soil_type(libelle)
        M = c.get("M_bas") or c.get("M") or 0.0
        rf = 0.0
        if c.get("fs") and c.get("qc"):
            rf = c["fs"] / (c["qc"] * 1000.0) * 100.0
        x = nouvelle_couche(h=round(c.get("h", 0.0), 2), type_sol=type_db, qc=round(c.get("qc", 0.0) or 0.0, 2),
                            rf=round(rf, 2), M=round(M or 0.0, 1), gamma=round(c.get("gamma", 19.0), 1))
        x["M_haut"], x["Ic"], x["sbt"] = c.get("M_haut"), c.get("Ic"), c.get("sbt")
        out.append(x)
    if not out:
        out.append(nouvelle_couche())
    return out


def valider_import(f: dict, i: int) -> None:
    """« ✅ Remplir le tableau » : les couches du sondage remplacées par le découpage, les mesures gardées, la nappe de
    l'aperçu reportée sur le sondage ET sur la fondation (nappe cochée), le nom du sondage pris dans le fichier."""
    s = f["sondages"][i]
    imp = s["import"]
    ap = apercu_import(imp, f["nu"])
    if not ap["couches"]:
        raise ValueError("Aucune couche exploitable n'a pu être formée.")
    s["couches"] = couches_depuis_cpt(ap["couches"])
    s["nappe"] = float(imp["nappe"])
    f["nappe_active"], f["nappe"] = True, float(imp["nappe"])
    if imp["source"]:
        s["source"] = imp["source"]
    s["points"] = [list(p) for p in imp["points"]]
    s["nom"] = imp["nom"]
    s["import"], s["colle"] = None, ""


def csv_mesures(nom: str, points: list) -> str:
    """« ⬇️ CSV des mesures » : SI.vers_csv."""
    return SI.vers_csv({"nom": nom, "points": [tuple(p) for p in points]})


def csv_couches(nom: str, couches: list) -> str:
    """« ⬇️ CSV des couches » : SI.couches_vers_csv sur le découpage de l'aperçu. Une couche SANS module (pas d'indice
    Ic : frottement nul ou absent) fait planter l'écran à cet endroit (TypeError : « {None:.1f} », défaut de
    sol_import.couches_vers_csv v1.0) ; ici une ValueError qui le dit, et pas d'export."""
    if any(c.get("M_bas") is None or c.get("M_haut") is None for c in couches):
        raise ValueError(EXPORT_COUCHES_IMPOSSIBLE)
    return SI.couches_vers_csv(nom, couches)


EXPORT_COUCHES_IMPOSSIBLE = ("Export des couches impossible : une couche au moins n'a pas de module (pas d'indice Ic, faute de "
                             "frottement fs). L'écran Streamlit plante au même endroit (sol_import.couches_vers_csv).")


# ------------------------------------------------------------------ modes 1 et 2 : le calcul du profil
def parametres(f: dict) -> dict:
    """Les paramètres du calcul (_resultats_profil) ; B et L échangés si L < B (« B est le petit côté »)."""
    B, L = float(f["B"]), float(f["L"])
    echange = L < B
    if echange:
        B, L = L, B
    return {"B": B, "L": L, "q": float(f["q"]), "D": float(f["D"]),
            "nappe": float(f["nappe"]) if f["nappe_active"] else None, "crit": float(f["critere"]) / 100.0,
            "critere_pct": f["critere"], "q_net": bool(f["q_net"]), "echange": echange}


def calculer_sondage(s: dict, p: dict):
    """Un sondage aux quatre positions, bornes basse et haute des modules : None s'il est ÉCARTÉ (aucune couche ne
    porte de module M). Les zones gardent le détail de la borne basse (« detail », celui de l'écran) et, en plus, celui
    de la borne haute."""
    bas, haut = profil_encadre(s)
    if not bas or all(not c.get("M") for c in bas):
        return None
    zones = {}
    for pos in ST.POSITIONS:
        rb = ST.tassement(bas, p["B"], p["L"], p["q"], D=p["D"], nappe_m=p["nappe"], critere=p["crit"], position=pos,
                          q_net=p["q_net"])
        rh = ST.tassement(haut, p["B"], p["L"], p["q"], D=p["D"], nappe_m=p["nappe"], critere=p["crit"], position=pos,
                          q_net=p["q_net"])
        zones[pos] = {"k_bas_MNm3": min(rb["k_MNm3"], rh["k_MNm3"]), "k_haut_MNm3": max(rb["k_MNm3"], rh["k_MNm3"]),
                      "w_bas_mm": min(rb["w_mm"], rh["w_mm"]), "w_haut_mm": max(rb["w_mm"], rh["w_mm"]),
                      "detail": rb, "detail_haut": rh}
    c = zones["centre"]
    rb_c = c["detail"]
    rb_c["critere_pct"] = p["critere_pct"]
    return {"nom": s["nom"], "zones": zones, "k_bas": c["k_bas_MNm3"], "k_haut": c["k_haut_MNm3"],
            "w_bas": c["w_bas_mm"], "w_haut": c["w_haut_mm"], "z_infl": rb_c["z_influence"], "detail": rb_c,
            "couches": [dict(cc, M_haut=hh.get("M")) for cc, hh in zip(bas, haut)], "q_net": rb_c["q_net_kPa"],
            "convergence": rb_c["convergence"], "bas": bas, "haut": haut}


def resultats_profil(f: dict) -> dict:
    """Tout ce que _resultats_profil affiche, en données : résultats par sondage (dans l'ordre des sondages, les écartés
    en moins), écartés, incalculables, panneau SCIA à la position de référence, enveloppe, profils trop courts,
    zonage, état de chaque sondage, et les tranches affichées (une sur ``pas``)."""
    p = parametres(f)
    resultats, ecartes = [], []
    for i, s in enumerate(f["sondages"]):
        r = calculer_sondage(s, p)
        if r is None:
            ecartes.append(s["nom"])
            continue
        r["index"] = i
        resultats.append(r)
    out = {"parametres": p, "resultats": resultats, "ecartes": ecartes, "incalculables": [], "scia": [], "enveloppe": None,
           "courts": None, "zonage": [], "position": f["position"]}
    if not resultats:
        return out
    out["incalculables"] = [r["nom"] for r in resultats if "incomplet" in r["convergence"]]
    pos = f["position"]
    out["scia"] = [{"nom": r["nom"], "k_bas": r["zones"][pos]["k_bas_MNm3"], "k_haut": r["zones"][pos]["k_haut_MNm3"],
                    "w_bas": r["zones"][pos]["w_bas_mm"], "w_haut": r["zones"][pos]["w_haut_mm"]} for r in resultats]
    ks = [r["zones"][pos]["k_bas_MNm3"] for r in resultats] + [r["zones"][pos]["k_haut_MNm3"] for r in resultats]
    ks = [v for v in ks if v > 0]
    n_inc = len(out["incalculables"])
    lib_env = ("Enveloppe tous sondages" if not (ecartes or n_inc)
               else f"Enveloppe des {len(resultats) - n_inc} sondages calculés sur {len(resultats) + len(ecartes)}")
    if ks and max(ks) - min(ks) > 1e-6:
        out["enveloppe"] = {"libelle": lib_env, "unique": False, "k_min": min(ks), "k_max": max(ks)}
    elif ks:
        out["enveloppe"] = {"libelle": lib_env, "unique": True, "k_min": ks[0], "k_max": ks[0]}
    courts = [r for r in resultats if "épuisé" in r["convergence"] or "plafond" in r["convergence"]]
    if courts:
        # « profondeur_conseillee » est le conseil de l'écran (2,5·B) ; il ne dit pas d'où il se compte. Le profil se
        # saisit depuis le TN et la zone d'influence commence à l'assise : le site précise D + 2,5·B sous le TN.
        out["courts"] = {"noms": [r["nom"] for r in courts], "manque": max(r["z_infl"] for r in courts),
                         "B": p["B"], "profondeur_conseillee": 2.5 * p["B"], "profondeur_conseillee_tn": p["D"] + 2.5 * p["B"]}
    for pz, lab in ST.POSITIONS.items():
        vals = [r["zones"][pz] for r in resultats]
        out["zonage"].append({"position": pz, "libelle": lab, "k_bas": min(v["k_bas_MNm3"] for v in vals),
                              "k_haut": max(v["k_haut_MNm3"] for v in vals)})
    for r in resultats:
        r["etat"] = "ok" if r["k_bas"] > 0 else "nok"
        r["tranches_affichees"], r["pas_tranches"] = tranches_affichees(r["detail"].get("tranches") or [])
        r["tronque"] = "épuisé" in str(r["convergence"]) or "plafond" in str(r["convergence"])
    return out


def tranches_affichees(tr: list):
    """Le tableau du détail : une tranche sur ``pas`` = max(1, n // 12)."""
    pas = max(1, len(tr) // 12)
    return tr[::pas], pas


def k_fonction_B(bas: list, haut: list, B: float, L: float, q: float, D: float, nappe, critere: float) -> dict:
    """_fig_k_vs_B : k au CENTRE pour 18 largeurs b = max(0,5 ; B·(0,25 + 0,15·i)), même élancement L/B, tranches de
    0,10 m — et la pression NETTE toujours (l'option « pression nette » de l'écran n'y est pas transmise)."""
    ratio = (L / B) if B > 0 else 1.0
    Bs, kb, kh = [], [], []
    for i in range(18):
        b = max(0.5, B * (0.25 + i * 0.15))
        Bs.append(b)
        rb = ST.tassement(bas, b, b * ratio, q, D=D, nappe_m=nappe, critere=critere, dz=0.10)
        rh = ST.tassement(haut, b, b * ratio, q, D=D, nappe_m=nappe, critere=critere, dz=0.10)
        kb.append(rb["k_MNm3"])
        kh.append(rh["k_MNm3"])
    return {"B": Bs, "k_bas": kb, "k_haut": kh, "B_projet": B}


# couleurs « YlOrBr » de matplotlib (ColorBrewer, neuf teintes interpolées linéairement) : le profil du graphique
_YLORBR = ("#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506")


def couleur_ylorbr(x: float) -> str:
    """La teinte de la palette YlOrBr à la fraction x ∈ [0, 1] (interpolation linéaire entre les neuf teintes)."""
    x = min(max(x, 0.0), 1.0) * (len(_YLORBR) - 1)
    i = min(int(x), len(_YLORBR) - 2)
    t = x - i
    a, b = _YLORBR[i], _YLORBR[i + 1]
    rgb = [round(int(a[k:k + 2], 16) * (1 - t) + int(b[k:k + 2], 16) * t) for k in (1, 3, 5)]
    return "#" + "".join(f"{v:02x}" for v in rgb)


def donnees_diffusion(r: dict, D: float, nappe) -> dict:
    """_fig_diffusion : ce que le graphique de diffusion trace, en données — la profondeur montrée (z_max), les bandes
    de couches (sous l'assise, coupées à z_max, teinte selon log M), la nappe, les courbes Δσ, σ'v0 et critère·σ'v0,
    la profondeur d'influence et si le profil a été INTERROMPU (« épuisé » ou « plafond » : k surestimé)."""
    res = r["detail"]
    tr = res.get("tranches") or []
    z_max = max(res.get("z_influence", 1.0) * 1.35, 1.0)
    Ms = [c.get("M") for c in r["couches"] if c.get("M")]
    Mlo, Mhi = (min(Ms), max(Ms)) if Ms else (1.0, 100.0)
    bandes, cum = [], 0.0
    for c in r["couches"]:
        h = c.get("h", 0.0)
        if h <= 0:
            continue
        z0, z1 = cum, cum + h
        cum = z1
        z0r, z1r = z0 - D, z1 - D
        if z1r <= 0 or z0r >= z_max:
            continue
        z0r, z1r = max(z0r, -D), min(z1r, z_max)
        M = c.get("M") or 0.0
        if M > 0 and Mhi > Mlo:
            t = (math.log10(max(M, .1)) - math.log10(max(Mlo, .1))) / \
                (math.log10(max(Mhi, .1)) - math.log10(max(Mlo, .1)) or 1)
        else:
            t = 0.0
        frac = 0.18 + 0.6 * min(max(t, 0), 1)
        bandes.append({"z0": z0r, "z1": z1r, "nom": c.get("nom") or "—", "M": M, "M_haut": c.get("M_haut"),
                       "fraction": frac, "couleur": couleur_ylorbr(frac), "etiquette": z1r - z0r > z_max * 0.05})
    zn = None
    if nappe is not None and -D <= nappe - D <= z_max:
        zn = nappe - D
    crit = res.get("critere_pct", 20) / 100.0
    tronque = "épuisé" in str(res.get("convergence", "")) or "plafond" in str(res.get("convergence", ""))
    return {"z_max": z_max, "z_haut": -D if D > 0 else -0.05, "D": D, "bandes": bandes, "nappe": zn,
            "z": [t["z_sous_assise"] for t in tr], "delta_sigma": [t["delta_sigma"] for t in tr],
            "sigma_v0": [t["sigma_v0_eff"] for t in tr], "critere": crit, "critere_sigma": [t["sigma_v0_eff"] * crit for t in tr],
            "z_influence": res.get("z_influence", 0.0), "tronque": tronque, "convergence": res.get("convergence", "")}


def donnees_cpt(points: list, nappe) -> dict:
    """_fig_cpt : qc et fs en fonction de la profondeur, et Ic par couche — découpage profil_depuis_cpt(points,
    nappe) avec ses réglages PAR DÉFAUT (finesse 0,25, épaisseur 0,50 m, ν 0,30), pas ceux de l'aperçu d'import. Là où
    ce découpage plante dans l'écran (voir decouper), ``erreur`` le dit et Ic n'est pas tracé."""
    pts = [tuple(p) for p in points]
    erreur = None
    try:
        couches = decouper(pts, nappe)
    except ValueError as e:                                  # l'écran plante ici ; le site trace qc et fs, sans Ic
        couches, erreur = [], str(e)
    return {"erreur": erreur, "z": [p[0] for p in pts], "qc": [p[1] for p in pts],
            "z_fs": [p[0] for p in pts if p[2] is not None], "fs": [p[2] for p in pts if p[2] is not None],
            "ic": [(c["Ic"], c["z0"], c["z1"]) for c in couches if c["Ic"] is not None],
            "bornes_sbt": [(b, lab) for b, _, lab in ST.SBT_ZONES[:-1]], "nappe": nappe,
            "z_max": max(p[0] for p in pts) if pts else 0.0, "couches": couches}


# ------------------------------------------------------------------ modes 3 à 6
def calcul_rapide(q: float, w: float) -> dict:
    """Mode 3 : k = q / w (w en mm)."""
    k = q / (w / 1000.0) if w > 0 else 0.0
    return {"q": q, "w": w, "k_kNm3": k, "k_MNm3": k / 1000, "etat": "ok" if k > 0 else "nok"}


THEORIES = (("tassement", "Tassement calculé (retenue)", "profil quelconque, fondation réelle"),
            ("rigide", "Élastique — semelle rigide", "massif homogène semi-infini"),
            ("souple", "Élastique — souple, centre", "massif homogène semi-infini"),
            ("vesic", "Vesić (1961)", "sol homogène, ordre de grandeur"),
            ("ancienne", "Ancienne formule 1/k = Σh/E (2B)", "⚠ contrainte supposée uniforme — biaisée"))


def theories(E: float, B: float, L: float, nu: float) -> dict:
    """Mode 4 (_resultats_theories) : sol homogène de 60 m, M = M(E, ν), q = 150 kPa, assise au TN, critère 20 %,
    pression BRUTE ; puis les formules élastiques, Vesić et l'ancienne formule E/(2B). B et L ne sont PAS échangés
    (l'écran ne le fait que dans les modes 1 et 2)."""
    M = ST.module_oedometrique(E, nu)
    couches = [{"h": 60.0, "gamma": 19.0, "M": M, "nom": "sol homogène"}]
    q = 150.0
    r = ST.tassement(couches, B, L, q, D=0.0, critere=0.20, q_net=False)
    k = {"tassement": r["k_MNm3"], "rigide": ST.k_elastique(E, B, nu, Is=0.88), "souple": ST.k_elastique(E, B, nu, Is=1.12),
         "vesic": ST.k_vesic(E, B, nu), "ancienne": E / (2 * B)}
    return {"E": E, "B": B, "L": L, "nu": nu, "M": M, "q": q, "tassement": r,
            "methodes": [{"cle": c, "nom": n, "k": k[c], "domaine": d} for c, n, d in THEORIES],
            "k_retenu": r["k_MNm3"], "etat": "ok" if r["k_MNm3"] > 0 else "nok"}


def abaque(B: float, w_ref: float) -> dict:
    """Mode 5 (_resultats_abaque) : chaque sol de la base (hors « Personnalisé » et sans k de plaque) — k de plaque
    0,30 m, k corrigé de Terzaghi pour la largeur B (SANS correction de forme : L n'est pas transmis), et la pression
    qui produit le tassement de référence."""
    lignes = []
    for nom, d in SB.SOIL_DB.items():
        if nom == "Personnalisé" or d.get("kp_min") is None:
            continue
        kb = ST.k_terzaghi_taille(d["kp_min"], B, nature=d["nature"])
        kh = ST.k_terzaghi_taille(d["kp_max"], B, nature=d["nature"])
        lignes.append({"categorie": d["category"], "type": nom, "gamma": d["gamma"], "M_min": d["M_min"],
                       "M_max": d["M_max"], "kp_min": d["kp_min"], "kp_max": d["kp_max"], "nature": d["nature"],
                       "k_bas": kb, "k_haut": kh, "q_bas": kb * 1000 * w_ref / 1000, "q_haut": kh * 1000 * w_ref / 1000})
    return {"B": B, "w_ref": w_ref, "lignes": lignes}


def plat(B: float, L: float, alpha: float, E: float, mortier: bool, tg: float, Eg: float) -> dict:
    """Mode 6 (_resultats_plat) : contact assimilé à une compression 1D, k_c = E/h_c avec h_c = α·min(B, L) ; lit de
    mortier en série : 1/k_eq = 1/k_c + 1/k_g, k_g = E_g/t_g (B, L, t_g en mm ; E en GPa ; k en kN/m³)."""
    hc = alpha * min(B, L) / 1000.0
    kc = (E * 1e6) / hc if hc > 0 else 0.0
    keq = kc
    kg = 0.0
    tg_m = None
    if mortier:
        tg_m = float(tg) / 1000.0
        kg = (float(Eg) * 1e6) / tg_m if tg_m > 0 else 0.0
        if kc > 0 and kg > 0:
            keq = 1.0 / (1.0 / kc + 1.0 / kg)
    return {"B": B, "L": L, "alpha": alpha, "E": E, "mortier": bool(mortier), "tg": tg, "Eg": Eg, "hc_m": hc, "kc": kc,
            "kg": kg, "keq": keq, "tg_m": tg_m, "etat": "ok" if keq > 0 else "nok"}


# ------------------------------------------------------------------ le fichier « 💾 Enregistrer » de l'écran
_LIGNE_DEFAUT = {"h": 1.0, "type": "—", "qc": 0.0, "rf": 0.0, "M": 0.0, "gamma": 19.0}   # setdefault de _render_layer_row


def depuis_payload(payload: dict, nom: str = "F1") -> dict:
    """Une fondation lue dans le fichier JSON de l'écran (raideur_sol.json, _payload) : les valeurs scalaires de la
    session (rs_*, snd<id>_*), l'ordre des couches de chaque sondage. Ce que l'écran n'y enregistre PAS est perdu à la
    relecture, ici comme dans l'écran : les mesures brutes d'un CPT importé (liste, non scalaire). Un sondage sans
    couche reçoit une couche par défaut (l'écran n'en afficherait aucune et n'offrirait plus de bouton pour en ajouter).
    ValueError si le fichier n'est pas celui de l'écran ou porte une valeur hors de ses bornes."""
    if not isinstance(payload, dict) or not isinstance(payload.get("values"), dict) or "soundings" not in payload:
        raise ValueError("ce fichier n'est pas un enregistrement de l'écran « Raideur élastique des sols »")
    v = payload["values"]
    modes = {lib: k for k, lib in MODES}
    f = {"nom": nom, "mode": modes.get(v.get("rs_mode"), "cpt")}
    for k, cle in (("B", "rs_B"), ("L", "rs_L"), ("D", "rs_D"), ("q", "rs_q"), ("nappe", "rs_nappe"), ("nu", "rs_nu"),
                   ("nappe_active", "rs_nappe_active"), ("critere", "rs_critere"), ("q_net", "rs_q_net"),
                   ("position", "rs_pos_scia")):
        f[k] = v.get(cle, FONDATION_DEFAUT[k])
    f["sondages"] = []
    ordres = payload.get("orders") or {}
    sondages = payload.get("soundings") or []
    if not isinstance(ordres, dict) or not isinstance(sondages, list):
        raise ValueError("liste des sondages ou ordre des couches illisible")
    for s in sondages:
        # le fichier de l'écran : {"id": entier, "nom": texte} ; un autre contenu est refusé avec un message lisible
        ident = s.get("id") if isinstance(s, dict) else None
        lisible = (not isinstance(ident, bool)) and (isinstance(ident, int) or (isinstance(ident, float) and ident.is_integer())
                                                      or (isinstance(ident, str) and ident.strip().isdigit()))
        if not lisible:
            raise ValueError(f"sondage illisible dans le fichier : {s!r}")
        sid = int(ident)
        lids = ordres.get(str(sid), [])
        if not isinstance(lids, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in lids):
            raise ValueError(f"ordre des couches du sondage {sid} illisible : {lids!r}")
        couches = []
        for lid in lids:
            c = {k: v.get(f"snd{sid}_layer_{lid}_{k}", d) for k, d in _LIGNE_DEFAUT.items()}
            for k in ("M_haut", "Ic", "sbt"):
                c[k] = v.get(f"snd{sid}_layer_{lid}_{k}")
            couches.append(c)
        f["sondages"].append({"id": sid, "nom": v.get(f"snd{sid}_nom", s.get("nom")), "source": v.get(f"snd{sid}_source") or "",
                              "nappe": v.get(f"snd{sid}_nappe"), "points": [], "import": None, "colle": "",
                              "couches": couches or [nouvelle_couche()]})
    if not f["sondages"]:
        f["sondages"] = [nouveau_sondage(1, "CPT01")]
    f["rapide"] = {"q": v.get("rs3_q", RAPIDE_DEFAUT["q"]), "w": v.get("rs3_w", RAPIDE_DEFAUT["w"])}
    f["theories"] = {k: v.get(f"rs4_{k}", d) for k, d in THEORIES_DEFAUT.items()}
    f["abaque"] = {"B": v.get("rs5_B", ABAQUE_DEFAUT["B"]), "w_ref": v.get("rs_w_ref", ABAQUE_DEFAUT["w_ref"])}
    f["plat"] = {"B": v.get("rs6_B", 200.0), "L": v.get("rs6_L", 200.0), "alpha": v.get("rs6_a", 0.5), "E": v.get("rs6_E", 30.0),
                 "mortier": bool(v.get("rs6_grout", False)), "tg": v.get("rs6_tg", 20.0), "Eg": v.get("rs6_Eg", 20.0)}
    return normaliser(f)


# ------------------------------------------------------------------ la relecture d'une fondation
def _fr(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def lire_nombre(valeur, cle: str, nom: str = "") -> float:
    """Un nombre d'un champ, dans les bornes de l'écran ; ValueError (message lisible) sinon."""
    nom = nom or NOMS_CHAMPS.get(cle, cle)
    if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
        raise ValueError(f"{nom} : nombre attendu, pas « {valeur} »")
    v = float(valeur)
    if not math.isfinite(v):
        raise ValueError(f"{nom} : nombre attendu, pas « {valeur} »")
    if abs(v) > VRAISEMBLANCE:
        raise ValueError(f"{nom} = {v:g} refusé : hors de toute vraisemblance (au plus 1 000 000 dans l'unité du champ)"
                         .replace(".", ","))
    lo, hi = BORNES.get(cle, (None, None))
    if lo is not None and v < lo:
        raise ValueError(f"{nom} = {_fr(v)} refusé : au moins {_fr(lo)} (borne de l'écran Streamlit, qui ramène la saisie à "
                         "la borne sans le dire)")
    if hi is not None and v > hi:
        raise ValueError(f"{nom} = {_fr(v)} refusé : au plus {_fr(hi)} (borne de l'écran Streamlit, qui ramène la saisie à "
                         "la borne sans le dire)")
    return v


def _bool(v, nom: str) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"{nom} : vrai ou faux attendu ({v!r})")
    return v


def _texte(v, nom: str, vide_ok: bool = True) -> str:
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise ValueError(f"{nom} : texte attendu ({v!r})")
    return v


def _optionnel(v, nom: str):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or abs(v) > VRAISEMBLANCE:
        raise ValueError(f"{nom} : nombre attendu ({v!r})")
    return float(v)


def _points(pts, nom: str) -> list:
    if pts is None:
        return []
    if not isinstance(pts, list):
        raise ValueError(f"{nom} : liste de mesures attendue")
    out = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) != 3:
            raise ValueError(f"{nom} : une mesure est [profondeur, qc, fs]")
        z, q, fs = p
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > VRAISEMBLANCE
               for x in (z, q)) or \
                (fs is not None and (isinstance(fs, bool) or not isinstance(fs, (int, float)) or not math.isfinite(fs)
                                     or abs(fs) > VRAISEMBLANCE)):
            raise ValueError(f"{nom} : mesure illisible {p!r}")
        out.append([float(z), float(q), None if fs is None else float(fs)])
    return out


def normaliser_couche(c) -> dict:
    if not isinstance(c, dict):
        raise ValueError("une couche doit être un objet")
    t = c.get("type", "—")
    if t not in TYPES:
        raise ValueError(f"type de sol inconnu : « {t} »")
    out = {"type": t}
    defaut = nouvelle_couche()
    for k in ("h", "qc", "rf", "M", "gamma"):
        out[k] = lire_nombre(c.get(k, defaut[k]) if c.get(k) is not None else defaut[k], "couche." + k)
    out["M_haut"] = _optionnel(c.get("M_haut"), "M haut")
    out["Ic"] = _optionnel(c.get("Ic"), "Ic")
    sbt = c.get("sbt")
    out["sbt"] = None if sbt is None else _texte(sbt, "classe CPT")
    return out


def normaliser_import(imp):
    if imp is None:
        return None
    if not isinstance(imp, dict):
        raise ValueError("import en attente illisible")
    out = {"nom": _texte(imp.get("nom", "CPT"), "nom de l'essai") or "CPT", "source": _texte(imp.get("source", ""), "source"),
           "points": _points(imp.get("points"), "mesures de l'essai"),
           "avertissements": [_texte(a, "avertissement") for a in (imp.get("avertissements") or [])],
           "nappe": lire_nombre(imp.get("nappe", FONDATION_DEFAUT["nappe"]), "import.nappe"),
           "seuil": lire_nombre(imp.get("seuil", IMPORT_DEFAUT["seuil"]), "import.seuil"),
           "h_min": lire_nombre(imp.get("h_min", IMPORT_DEFAUT["h_min"]), "import.h_min"),
           "origine": imp.get("origine") if imp.get("origine") in ("fichier", "colle") else "fichier"}
    if len(out["points"]) < 1:
        raise ValueError("essai en attente sans mesure")
    return out


def normaliser_sondage(s, i: int = 0) -> dict:
    if not isinstance(s, dict):
        raise ValueError("un sondage doit être un objet")
    ident = s.get("id", i + 1)
    if isinstance(ident, bool) or not isinstance(ident, (int, float)) or int(ident) != ident or ident < 1:
        raise ValueError(f"identifiant de sondage illisible : {ident!r}")
    couches = s.get("couches")
    if not isinstance(couches, list) or not couches:
        raise ValueError(f"sondage « {s.get('nom')} » : au moins une couche")
    return {"id": int(ident), "nom": _texte(s.get("nom", f"CPT{int(ident):02d}"), "nom du sondage"),
            "source": _texte(s.get("source", ""), "source"), "nappe": _optionnel(s.get("nappe"), "nappe du sondage"),
            "points": _points(s.get("points"), "mesures"), "couches": [normaliser_couche(c) for c in couches],
            "import": normaliser_import(s.get("import")), "colle": _texte(s.get("colle", ""), "tableau collé")}


def _bloc(d, defaut: dict, prefixe: str) -> dict:
    d = d if isinstance(d, dict) else {}
    out = {}
    for k, v0 in defaut.items():
        v = d.get(k, v0)
        if v is None:
            v = v0
        out[k] = _bool(v, k) if isinstance(v0, bool) else lire_nombre(v, f"{prefixe}.{k}")
    return out


def normaliser(f) -> dict:
    """Une fondation complétée (clés absentes : défauts de l'écran) et CONTRÔLÉE : ValueError (message lisible) pour un
    nombre hors des bornes de l'écran, un type de sol inconnu, un mode inconnu, un sondage sans couche."""
    if not isinstance(f, dict):
        raise ValueError("une fondation doit être un objet")
    mode = f.get("mode", "cpt")
    if mode not in LIBELLES_MODES:
        raise ValueError(f"mode inconnu : « {mode} »")
    out = {"nom": _texte(f.get("nom", "F1"), "nom"), "mode": mode}
    for k in ("B", "L", "D", "q", "nappe", "nu"):
        v = f.get(k, FONDATION_DEFAUT[k])
        out[k] = lire_nombre(FONDATION_DEFAUT[k] if v is None else v, k)
    out["nappe_active"] = _bool(f.get("nappe_active", False), "nappe")
    out["q_net"] = _bool(f.get("q_net", True), "pression nette")
    crit = f.get("critere", 20)
    if crit not in CRITERES or isinstance(crit, bool):
        raise ValueError(f"critère de profondeur d'influence : {crit!r} (10, 15, 20, 25 ou 30 %)")
    out["critere"] = int(crit)
    pos = f.get("position", "centre")
    if pos not in ST.POSITIONS:
        raise ValueError(f"position de référence inconnue : « {pos} »")
    out["position"] = pos
    sondages = f.get("sondages")
    if sondages is None:
        sondages = [nouveau_sondage(1, "CPT01")]
    if not isinstance(sondages, list) or not sondages:
        raise ValueError("au moins un sondage")
    out["sondages"] = [normaliser_sondage(s, i) for i, s in enumerate(sondages)]
    ids = [s["id"] for s in out["sondages"]]
    if len(set(ids)) != len(ids):
        raise ValueError("deux sondages portent le même identifiant")
    out["rapide"] = _bloc(f.get("rapide"), RAPIDE_DEFAUT, "rapide")
    out["theories"] = _bloc(f.get("theories"), THEORIES_DEFAUT, "theories")
    out["abaque"] = _bloc(f.get("abaque"), ABAQUE_DEFAUT, "abaque")
    out["plat"] = _bloc(f.get("plat"), PLAT_DEFAUT, "plat")
    return out
