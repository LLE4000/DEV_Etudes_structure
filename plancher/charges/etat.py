# -*- coding: utf-8 -*-
"""État de l'écran « Charges sur plancher » : un dictionnaire sérialisable
(cas multiples, fiches ajoutées), ses défauts, sa construction en objets
du moteur, le contexte des dessins et l'application des messages du plan.
Module PUR (pas de Streamlit) : l'écran ne fait que le tenir en session.

Structure :
  etat = {"version": 1, "actif": 0, "cas": [cas…],
          "fiches": {"engins": […], "repartiteurs": […], "hourdis": […]}}
  cas  = {"nom", "plancher": {"type", "sens", "L", "q_presentes", "hourdis": {…}, "dalle": {…}, "predalle": {…}},
          "charges": [{"type": "engin"|"ponctuel"|"lineaire"|"surfacique", "ref", …}],
          "repartiteurs": {ref: {"type": "plateau", "nom", "dx", "dy"} | {"type": "filiere", "id"}},
          "filieres": [{"id", "nom", "longueur", "orientation", "x", "y"}],
          "params": {…}, "selection": clé | None, "vue": "plan", "options": {…}}
Positions en MÈTRES dans le repère MOTEUR (x le long de la portée depuis A,
y en travers depuis le bord haut) ; dimensions des plateaux en mm.
"""
import copy
import json
from typing import Dict, List, Optional, Tuple

from . import bibliotheque as BIB
from . import modele as M
from . import repartiteurs as RP
from .parametres import Parametres

VERSION = 1


# ---------------------------------------------------------------- défauts
def defaut_plancher() -> Dict:
    return {
        "type": "hourdis", "sens": 1, "L": 7.2, "q_presentes": 1.0,
        "hourdis": {"type_cat": "Hourdis générique 200 · à renseigner", "b_element": 1.2, "n_elements": 8, "h": 0.2,
                    "g": 2.9, "E": 36000.0, "I": None, "etat": "clavetes", "M_Rd": 0.0, "V_Rd": 0.0, "nature": "Rd",
                    "M_Rd_chape": None, "V_Rd_chape": None, "M_cr": None, "p_loc_adm": None,
                    "chape": {"oui": False, "epaisseur": 0.05, "fck_t": 20.0, "collaborante": True, "fck_exigee": 25.0},
                    "parts_annexe_C": None, "source": ""},
        "dalle": {"h": 0.2, "d": 0.17, "beton": "C30/37", "fck_t": None, "As_inf": 524.0, "fyk": 500.0, "gamma_s": 1.15,
                  "armature_transversale_ok": True, "h1": 0.0, "largeur": 9.6},
        "predalle": {"h": 0.2, "d": 0.17, "beton": "C30/37", "fck_t": None, "As_inf": 524.0, "fyk": 500.0, "gamma_s": 1.15,
                     "armature_transversale_ok": True, "h1": 0.0, "largeur": 9.6, "b_predalle": 2.4, "joints_armes": True,
                     "chape_durcie": True, "portee_etayee": None},
    }


def defaut_params() -> Dict:
    return {"gamma_G": 1.35, "gamma_Q": 1.5, "part_tranchant_element": 1.0, "k_flexion_heft240": None, "joint_adm_kN_m": None,
            "joint_longueur_m": None, "gamma_c": 1.5, "alpha_cc": 1.0, "coef_vrd_max": 0.4}


def nouveau_cas(nom: str) -> Dict:
    return {"nom": nom, "plancher": defaut_plancher(), "charges": [], "repartiteurs": {}, "filieres": [],
            "params": defaut_params(), "selection": None, "vue": "plan",
            "options": {"cotes": True, "aires": True, "taux": True},
            "identification": {"projet": "", "partie": "", "indice": "0"}}


def nouvel_etat() -> Dict:
    return {"version": VERSION, "actif": 0, "cas": [nouveau_cas("C1")],
            "fiches": {"engins": [], "repartiteurs": [], "hourdis": []}}


def cas_actif(etat: Dict) -> Dict:
    etat["actif"] = max(0, min(etat.get("actif", 0), len(etat["cas"]) - 1))
    return etat["cas"][etat["actif"]]


def ref_libre(cas: Dict, prefixe: str) -> str:
    pris = {c["ref"] for c in cas["charges"]} | {f["id"] for f in cas.get("filieres", [])}
    n = 1
    while f"{prefixe}{n}" in pris:
        n += 1
    return f"{prefixe}{n}"


# ---------------------------------------------------------------- bibliothèques fusionnées
def gabarits(etat: Dict) -> Dict[str, Dict]:
    """Fiches d'engins livrées + ajoutées, sous forme de dicts JSON."""
    out = {}
    for d in BIB._lire("engins.json") + list(etat["fiches"].get("engins", [])):
        out[d["nom"]] = d
    return out


def repartiteurs_disponibles(etat: Dict) -> Dict[str, Dict]:
    """Plateaux et filières : fiches livrées (plateaux.json), profilés du
    catalogue (filières) et fiches ajoutées (plateaux ou filières libres)."""
    out = {}
    for d in BIB._lire("plateaux.json"):
        out[d["nom"]] = dict(d, type="plateau", origine="livré")
    for nom, p in BIB.profils().items():
        out[nom] = {"nom": nom, "type": "filiere", "origine": "catalogue", "section": {"profil": nom}, "materiau": "S235",
                    "E": 210000.0, "f": None, "masse": p.get("masse", 0.0), "source": "profiles_test.json"}
    for d in etat["fiches"].get("repartiteurs", []):
        out[d["nom"]] = dict(d, origine="fiche")
    return out


def types_hourdis(etat: Dict) -> Dict[str, Dict]:
    out = {}
    for d in BIB._lire("hourdis.json") + list(etat["fiches"].get("hourdis", [])):
        out[d["nom"]] = d
    return out


CHAMPS_FICHE_HOURDIS = ("b_element", "h", "g", "E", "I", "M_Rd", "V_Rd", "nature", "M_Rd_chape", "V_Rd_chape", "M_cr",
                        "p_loc_adm", "source")


def appliquer_type_hourdis(cas: Dict, fiche: Dict) -> None:
    """Copie les valeurs d'une fiche de hourdis dans le cas (modifiables ensuite)."""
    h = cas["plancher"]["hourdis"]
    h["type_cat"] = fiche["nom"]
    for k in CHAMPS_FICHE_HOURDIS:
        if k in fiche:
            h[k] = fiche[k]


def ajouter_fiche_hourdis(etat: Dict, cas: Dict, nom: str) -> Dict:
    """P-1 : les valeurs saisies à l'étape Plancher deviennent une fiche du
    catalogue de ce fichier (un catalogue transcrit sert de défaut, le bureau
    peut toujours indiquer les siennes). Un nom déjà pris est remplacé."""
    h = cas["plancher"]["hourdis"]
    fiche = {"nom": nom, "generique": False}
    for k in CHAMPS_FICHE_HOURDIS:
        fiche[k] = h.get(k)
    etat["fiches"]["hourdis"] = [f for f in etat["fiches"].get("hourdis", []) if f["nom"] != nom] + [fiche]
    h["type_cat"] = nom
    return fiche


# ---------------------------------------------------------------- construction des objets du moteur
def _section_filiere(d: Dict) -> Tuple[float, float, float]:
    """(EI kN·m², W_el cm³, b_semelle m) d'une fiche de filière : profilé du
    catalogue, ou rectangle b × h (mm) posé sur champ (I = b·h³/12, W = b·h²/6)."""
    sec = d.get("section", {})
    E = float(d.get("E") or 210000.0)
    if sec.get("profil"):
        p = BIB.profils()[sec["profil"]]
        return E * 1e3 * p["Iy"] * 1e-8, float(p["Wel"]), p["b"] / 1e3
    b, h = float(sec["b"]), float(sec["h"])
    I = b * h ** 3 / 12.0                       # mm⁴
    return E * 1e3 * I * 1e-12, b * h * h / 6.0 / 1e3, b / 1e3


def _f_filiere(d: Dict) -> float:
    if d.get("f"):
        return float(d["f"])
    if d.get("materiau") in BIB.aciers():
        return float(BIB.aciers()[d["materiau"]]["fy"])
    raise ValueError(f"Filière « {d['nom']} » : résistance f absente (fy/γM0 ou f_m,d), à renseigner dans la fiche.")


def construire(etat: Dict, cas: Optional[Dict] = None):
    """Objets du moteur pour un cas : (plancher, charges, repartiteurs, params, index).
    ``index`` : {"engins": {ref: Engin}, "empreintes": {ref: Empreinte}, "plateaux": {ref: Plateau}, "filieres": {id: Filiere}}."""
    cas = cas or cas_actif(etat)
    pl = cas["plancher"]
    t = pl["type"]
    if t == "hourdis":
        h = pl["hourdis"]
        chape = None
        if h.get("chape", {}).get("oui"):
            c = h["chape"]
            chape = M.Chape(float(c["epaisseur"]), float(c["fck_t"]), bool(c.get("collaborante", True)),
                            c.get("fck_exigee"))
        plancher = M.Hourdis(nom=h.get("type_cat", "Hourdis"), L=float(pl["L"]), q_presentes=float(pl["q_presentes"]),
                             b_element=float(h["b_element"]), n_elements=int(h["n_elements"]), h=float(h["h"]),
                             g=float(h["g"]), E=float(h.get("E") or 36000.0), I=h.get("I"), etat=h.get("etat", "clavetes"),
                             M_Rd=float(h.get("M_Rd") or 0.0), V_Rd=float(h.get("V_Rd") or 0.0), nature=h.get("nature", "Rd"),
                             M_Rd_chape=h.get("M_Rd_chape"), V_Rd_chape=h.get("V_Rd_chape"), M_cr=h.get("M_cr"),
                             p_loc_adm=h.get("p_loc_adm"), chape=chape,
                             parts_annexe_C={int(k): float(v) for k, v in (h.get("parts_annexe_C") or {}).items()} or None,
                             source=h.get("source", ""))
    elif t == "dalle":
        d = pl["dalle"]
        plancher = M.DalleBA(nom="Dalle", L=float(pl["L"]), q_presentes=float(pl["q_presentes"]), h=float(d["h"]), d=float(d["d"]),
                             beton=d["beton"], fck_t=d.get("fck_t"), As_inf=float(d["As_inf"]), fyk=float(d["fyk"]),
                             gamma_s=float(d["gamma_s"]), armature_transversale_ok=bool(d["armature_transversale_ok"]),
                             h1=float(d["h1"]), largeur=float(d["largeur"]))
    else:
        d = pl["predalle"]
        plancher = M.Predalle(nom="Prédalle", L=float(pl["L"]), q_presentes=float(pl["q_presentes"]), h=float(d["h"]), d=float(d["d"]),
                              beton=d["beton"], fck_t=d.get("fck_t"), As_inf=float(d["As_inf"]), fyk=float(d["fyk"]),
                              gamma_s=float(d["gamma_s"]), armature_transversale_ok=bool(d["armature_transversale_ok"]),
                              h1=float(d["h1"]), largeur=float(d["largeur"]), b_predalle=float(d["b_predalle"]),
                              joints_armes=bool(d["joints_armes"]), chape_durcie=bool(d["chape_durcie"]),
                              portee_etayee=d.get("portee_etayee"))
    gabs = gabarits(etat)
    charges, index = [], {"engins": {}, "empreintes": {}, "plateaux": {}, "filieres": {}}
    for c in cas["charges"]:
        if c["type"] == "engin":
            g = copy.deepcopy(gabs[c["gabarit"]])
            for ap in g["appuis"]:
                ap["P"] = float((c.get("P") or {}).get(ap["nom"], ap.get("P", 0.0)) or 0.0)
            gab = BIB.gabarit_depuis(g)
            en = M.Engin(c["ref"], gab, float(c["x"]), float(c["y"]), int(c.get("angle", 0)))
            charges.append(en)
            index["engins"][c["ref"]] = en
            for e in en.empreintes():
                index["empreintes"][e.ref] = e
        elif c["type"] == "ponctuel":
            e = M.Empreinte(c["ref"], float(c["x"]), float(c["y"]), float(c["a"]), float(c["b"]), float(c["P"]), float(c.get("phi", 1.0)))
            charges.append(e)
            index["empreintes"][e.ref] = e
        elif c["type"] == "lineaire":
            charges.append(M.ChargeLineaire(c["ref"], float(c["x"]), float(c["y"]), float(c["longueur"]), float(c["largeur"]),
                                            float(c["p"]), c.get("orientation", "y"), float(c.get("phi", 1.0))))
        elif c["type"] == "surfacique":
            charges.append(M.ChargeSurfacique(c["ref"], float(c["x0"]), float(c["y0"]), float(c["lx"]), float(c["ly"]), float(c["q"])))
    dispo = repartiteurs_disponibles(etat)
    for f in cas.get("filieres", []):
        d = dispo.get(f["nom"])
        if d is None:
            continue
        EI, W, b = _section_filiere(d)
        fil = M.Filiere(f"{f['id']} · {f['nom']}", EI, W, _f_filiere(d), float(f["longueur"]), float(f["x"]), float(f["y"]),
                        f.get("orientation", "y"), b, [r for r, rp in cas["repartiteurs"].items() if rp.get("id") == f["id"]],
                        1.0, float(d.get("masse", 0.0)), d.get("source", ""))
        index["filieres"][f["id"]] = fil
    repartiteurs = {}
    for ref, rp in cas["repartiteurs"].items():
        if rp["type"] == "plateau":
            d = dispo.get(rp["nom"])
            if d is None or d.get("type") != "plateau":
                continue
            p = M.Plateau(d["nom"], d.get("f"), float(d["t"]), float(d["A"]), float(d["Bp"]), bool(d.get("rigide_declare", False)),
                          d.get("p_adm_declare"), float(rp.get("dx", 0.0)), float(rp.get("dy", 0.0)), d.get("source", ""),
                          float(d.get("masse", 0.0)))
            repartiteurs[ref] = p
            index["plateaux"][ref] = p
        elif rp["type"] == "filiere" and rp.get("id") in index["filieres"]:
            repartiteurs[ref] = index["filieres"][rp["id"]]
    pr = cas.get("params", {})
    params = Parametres(gamma_G=float(pr.get("gamma_G", 1.35)), gamma_Q=float(pr.get("gamma_Q", 1.5)),
                        part_tranchant_element=float(pr.get("part_tranchant_element", 1.0)),
                        k_flexion_heft240=pr.get("k_flexion_heft240"), joint_adm_kN_m=pr.get("joint_adm_kN_m"),
                        joint_longueur_m=pr.get("joint_longueur_m"), gamma_c=float(pr.get("gamma_c", 1.5)),
                        alpha_cc=float(pr.get("alpha_cc", 1.0)), coef_vrd_max=pr.get("coef_vrd_max", 0.4))
    return plancher, charges, repartiteurs, params, index


# ---------------------------------------------------------------- contexte des dessins
def contexte_dessin(etat: Dict, cas: Dict, res: Dict, plancher, index: Dict) -> Dict:
    pl = cas["plancher"]
    gabs = gabarits(etat)
    dispo = repartiteurs_disponibles(etat)
    taux = {e["nom"]: e for e in res.get("elements", [])}
    elements = [{"nom": e.nom, "y0": e.y0, "y1": e.y1, "taux": taux.get(e.nom, {}).get("taux"),
                 "gouverne": taux.get(e.nom, {}).get("gouverne")} for e in plancher.elements()]
    valeurs = {"L": round(plancher.L * 1e3), "sens": pl.get("sens", 1)}
    groupes = {}
    engins, libelles = [], {}
    patches = {p["ref"]: p for p in res.get("patches", [])}
    for c in cas["charges"]:
        if c["type"] != "engin":
            continue
        en = index["engins"][c["ref"]]
        emp = []
        for e in en.empreintes():
            emp.append({"ref": e.ref, "nom": e.ref.split(".")[-1], "x": e.x, "y": e.y, "a": e.a, "b": e.b, "P": e.P, "phi": e.phi})
        g = copy.deepcopy(gabs[c["gabarit"]])
        engins.append({"ref": c["ref"], "gabarit": g, "x": c["x"], "y": c["y"], "angle": int(c.get("angle", 0)), "empreintes": emp,
                       "rayon_etiquette": max(abs(a["dy"]) for a in g["appuis"]) * 1e3 + 900 if g["appuis"] else 1200})
        libelles[c["ref"]] = f"{c['ref']} · {g['nom'].split(' · ')[0]}"
        sx, sy = _vers_ecran(pl, c["x"], c["y"])
        valeurs[f"engin:{c['ref']}.sx"], valeurs[f"engin:{c['ref']}.sy"] = round(sx), round(sy)
        valeurs[f"engin:{c['ref']}.angle"] = str(int(c.get("angle", 0)))
        valeurs[f"engin:{c['ref']}.action"] = "—"
        champs = [{"key": f"engin:{c['ref']}.angle", "lab": "Angle (°)", "type": "s", "options": ["0", "90", "180", "270"]}]
        for ap in g["appuis"]:
            k = f"engin:{c['ref']}.P.{ap['nom']}"
            valeurs[k] = float((c.get("P") or {}).get(ap["nom"], ap.get("P", 0.0)) or 0.0)
            champs.append({"key": k, "lab": f"{ap['nom']} (kN)", "type": "n", "step": 1})
        champs.append({"key": f"engin:{c['ref']}.action", "lab": "Action", "type": "s", "options": ["—", "Dupliquer", "Supprimer"]})
        groupes[f"engin:{c['ref']}"] = {"titre": f"{c['ref']} · {g['nom']}", "champs": champs,
                                        "note": "Charges par appui : tableau de stabilité du constructeur, pour l'opération prévue. Glisser l'engin pour le déplacer."}
    charges = []
    for c in cas["charges"]:
        if c["type"] == "engin":
            continue
        d = dict(c)
        charges.append(d)
        k = f"charge:{c['ref']}"
        if c["type"] == "surfacique":
            sx, sy = _vers_ecran(pl, c["x0"], c["y0"])
            valeurs[k + ".sx"], valeurs[k + ".sy"] = round(sx), round(sy)
            valeurs[k + ".q"] = c["q"]; valeurs[k + ".lx"] = round(c["lx"] * 1e3); valeurs[k + ".ly"] = round(c["ly"] * 1e3)
            champs = [{"key": k + ".q", "lab": "q (kN/m²)", "type": "n", "step": 0.5},
                      {"key": k + ".lx", "lab": "longueur (mm)", "type": "n", "step": 100},
                      {"key": k + ".ly", "lab": "largeur (mm)", "type": "n", "step": 100}]
        elif c["type"] == "lineaire":
            sx, sy = _vers_ecran(pl, c["x"], c["y"])
            valeurs[k + ".sx"], valeurs[k + ".sy"] = round(sx), round(sy)
            valeurs[k + ".p"] = c["p"]; valeurs[k + ".longueur"] = round(c["longueur"] * 1e3); valeurs[k + ".orientation"] = c.get("orientation", "y")
            champs = [{"key": k + ".p", "lab": "p (kN/m)", "type": "n", "step": 1},
                      {"key": k + ".longueur", "lab": "longueur (mm)", "type": "n", "step": 100},
                      {"key": k + ".orientation", "lab": "Orientation", "type": "s", "options": ["x", "y"],
                       "affichage": {"x": "dans le sens porteur", "y": "en travers"}}]
        else:
            sx, sy = _vers_ecran(pl, c["x"], c["y"])
            valeurs[k + ".sx"], valeurs[k + ".sy"] = round(sx), round(sy)
            valeurs[k + ".P"] = c["P"]; valeurs[k + ".a"] = round(c["a"] * 1e3); valeurs[k + ".b"] = round(c["b"] * 1e3)
            champs = [{"key": k + ".P", "lab": "P (kN)", "type": "n", "step": 1},
                      {"key": k + ".a", "lab": "semelle a (mm)", "type": "n", "step": 10},
                      {"key": k + ".b", "lab": "semelle b (mm)", "type": "n", "step": 10}]
        valeurs[k + ".action"] = "—"
        champs.append({"key": k + ".action", "lab": "Action", "type": "s", "options": ["—", "Dupliquer", "Supprimer"]})
        groupes[k] = {"titre": f"Charge {c['ref']}", "champs": champs}
    plateaux = []
    noms_plateaux = [n for n, d in dispo.items() if d.get("type") == "plateau"]
    noms_filieres = [n for n, d in dispo.items() if d.get("type") == "filiere"]
    for ref, e in index["empreintes"].items():
        rp = cas["repartiteurs"].get(ref)
        k = f"rep:{ref}"
        typ = rp["type"] if rp else "aucun"
        valeurs[k + ".type"] = typ
        valeurs[k + ".element"] = (rp.get("nom") if rp and rp["type"] == "plateau" else (noms_plateaux[0] if noms_plateaux else ""))
        valeurs[k + ".idem"] = "non"
        if rp and rp["type"] == "plateau" and ref in index["plateaux"]:
            p = index["plateaux"][ref]
            pt = patches.get(ref, {})
            plateaux.append({"ref": ref, "nom": p.nom, "x": e.x + p.dx, "y": e.y + p.dy, "A": p.A, "Bp": p.Bp,
                             "a_eff": (pt.get("a", 0) * 1e3 if pt else None), "b_eff": (pt.get("b", 0) * 1e3 if pt else None),
                             "complet": pt.get("complet"), "pression": pt.get("pression")})
            sx, sy = _vers_ecran(pl, e.x + p.dx, e.y + p.dy)
            valeurs[k + ".sx"], valeurs[k + ".sy"] = round(sx), round(sy)
        groupes[k] = {"titre": f"Répartiteur sous {ref}", "champs": [
            {"key": k + ".type", "lab": "Type", "type": "s", "options": ["aucun", "plateau", "filiere"],
             "affichage": {"aucun": "Aucun", "plateau": "Plateau", "filiere": "Filière"}},
            {"key": k + ".element", "lab": "Plateau", "type": "s", "options": noms_plateaux},
            {"key": k + ".idem", "lab": "Idem tous les appuis de l'engin", "type": "s", "options": ["non", "oui"]}],
            "note": "Un plateau se pose centré sous l'appui ; glisser le plateau pour le décaler. Une filière se règle dans l'étape 3."}
    # un étançon n'a qu'un symbole cliquable : sa fenêtre porte aussi le répartiteur de sa semelle
    for c in cas["charges"]:
        if c["type"] == "ponctuel" and f"rep:{c['ref']}" in groupes:
            g = groupes[f"charge:{c['ref']}"]
            g["champs"][-1:-1] = groupes[f"rep:{c['ref']}"]["champs"][:2]
    filieres = []
    for f in cas.get("filieres", []):
        fil = index["filieres"].get(f["id"])
        if not fil:
            continue
        filieres.append({"id": f["id"], "nom": f["nom"], "x": f["x"], "y": f["y"], "longueur": f["longueur"],
                         "orientation": f.get("orientation", "y"), "b": fil.b_semelle})
        k = f"fil:{f['id']}"
        sx, sy = _vers_ecran(pl, f["x"], f["y"])
        valeurs[k + ".sx"], valeurs[k + ".sy"] = round(sx), round(sy)
        valeurs[k + ".element"] = f["nom"]; valeurs[k + ".longueur"] = round(f["longueur"] * 1e3)
        valeurs[k + ".orientation"] = f.get("orientation", "y"); valeurs[k + ".action"] = "—"
        groupes[k] = {"titre": f"Filière {f['id']}", "champs": [
            {"key": k + ".element", "lab": "Élément", "type": "s", "options": noms_filieres},
            {"key": k + ".longueur", "lab": "Longueur (mm)", "type": "n", "step": 100},
            {"key": k + ".orientation", "lab": "Orientation", "type": "s", "options": ["y", "x"],
             "affichage": {"y": "en travers des éléments", "x": "dans le sens porteur"}},
            {"key": k + ".action", "lab": "Action", "type": "s", "options": ["—", "Supprimer"]}]}
    if pl["type"] == "hourdis":
        cote = f"{pl['hourdis']['n_elements']} × {pl['hourdis']['b_element']:.2f} = {plancher.B:.2f} m".replace(".", ",")
        valeurs["n_elements"] = int(pl["hourdis"]["n_elements"])
    else:
        cote = f"largeur {plancher.B:.2f} m".replace(".", ",")
    return {"L": plancher.L, "B": plancher.B, "sens": pl.get("sens", 1), "type": pl["type"], "elements": elements,
            "joints": plancher.joints() if pl["type"] == "predalle" else [], "engins": engins, "charges": charges,
            "plateaux": plateaux, "filieres": filieres, "selection": cas.get("selection"), "options": cas.get("options", {}),
            "seuils": (0.85, 1.0), "libelles": libelles, "groupes": groupes, "valeurs": valeurs, "cote_largeur": cote}


def contexte_coupe(etat: Dict, cas: Dict, res: Dict, plancher, index: Dict) -> Dict:
    """Contexte de la coupe (``dessin.coupe``) : l'appui sélectionné, sinon
    le premier appui. Même source pour l'écran et la note."""
    sel = cas.get("selection") or ""
    ref = sel.split(":", 1)[1] if sel.startswith("rep:") else None
    emps = index["empreintes"]
    if ref not in emps:
        ref = next(iter(emps), None)
    pl = cas["plancher"]
    patches = {p["ref"]: p for p in res.get("patches", [])}
    e = emps.get(ref)
    el = plancher.element_de(e.y) if e else None
    els = plancher.elements()
    i = el.index if el else 0
    voisins = (els[i - 1].nom if i > 0 else "", els[i].nom if els else "", els[i + 1].nom if i + 1 < len(els) else "")
    taux = {x["nom"]: x["taux"] for x in res.get("elements", [])}
    pt = patches.get(ref, {})
    plat = index["plateaux"].get(ref)
    chape = 0.0
    if pl["type"] == "hourdis" and pl["hourdis"].get("chape", {}).get("oui"):
        chape = float(pl["hourdis"]["chape"]["epaisseur"])
    return {"type": "hourdis" if pl["type"] == "hourdis" else "dalle", "h": float(pl[pl["type"]]["h"]), "chape": chape,
            "b_element": els[0].largeur if els else 1.2, "voisins": voisins, "nom_element": voisins[1],
            "plateau": {"A": plat.A, "t": plat.t} if plat else None, "semelle": e.a if e else 0.15,
            "P": (e.P * e.phi) if e else 0.0, "a_eff": pt.get("a") if pt else None, "taux": taux.get(voisins[1], 0.0) or 0.0,
            "joints_predalle": pl["type"] == "predalle"}


def _vers_ecran(pl: Dict, x: float, y: float) -> Tuple[float, float]:
    """Moteur (m) → écran (mm absolus du SVG) selon le sens d'affichage."""
    from .dessin import OX, OY
    if pl.get("sens", 1) == 2:
        return OX + y * 1e3, OY + x * 1e3
    return OX + x * 1e3, OY + y * 1e3


def _vers_moteur(pl: Dict, sx: float, sy: float) -> Tuple[float, float]:
    from .dessin import OX, OY
    if pl.get("sens", 1) == 2:
        return (sy - OY) / 1e3, (sx - OX) / 1e3
    return (sx - OX) / 1e3, (sy - OY) / 1e3


# ---------------------------------------------------------------- application des messages du plan
def appliquer_changements(etat: Dict, cas: Dict, changes: Dict, index: Dict) -> List[str]:
    """Applique les clés renvoyées par le composant au cas ; rend les textes
    des changements (toast). Positions reçues en mm ÉCRAN."""
    pl = cas["plancher"]
    faits = []
    pos: Dict[str, Dict[str, float]] = {}
    for k, v in changes.items():
        if k == "L":
            pl["L"] = max(0.5, float(v) / 1e3)
            faits.append(f"L = {pl['L']:.2f} m")
        elif k == "sens":
            pl["sens"] = 2 if int(v) >= 2 else 1
            faits.append("sens porteur : " + ("appuis en haut et en bas" if pl["sens"] == 2 else "appuis à gauche et à droite"))
        elif k == "n_elements":
            pl["hourdis"]["n_elements"] = max(1, int(v))
            faits.append(f"{pl['hourdis']['n_elements']} hourdis")
        elif ":" in k:
            obj, reste = k.split(":", 1)
            # La référence d'un appui porte un point (« E1.P2 ») : pour « rep », le champ est
            # après le DERNIER point ; pour les autres objets (« E1 », « T1 », « F1 »), après le
            # PREMIER — le champ lui-même peut en porter un (« P.P1 », charge par appui).
            # Défaut trouvé par le harnais le 02/11/2026 : « engin:E1.P.P1 » était lu
            # ref « E1.P », champ « P1 », et la fenêtre de l'engin n'écrivait rien.
            ref, _, champ = reste.rpartition(".") if obj == "rep" else reste.partition(".")
            if champ in ("sx", "sy"):
                pos.setdefault(k.rsplit(".", 1)[0], {})[champ] = float(v)
            else:
                faits.extend(_champ(etat, cas, obj, ref, champ, v, index))
    for cle, d in pos.items():
        obj, ref = cle.split(":", 1)
        faits.extend(_deplacer(etat, cas, obj, ref, d, index))
    return faits


def _deplacer(etat, cas, obj, ref, d, index) -> List[str]:
    pl = cas["plancher"]
    if obj == "engin" or obj == "charge":
        c = next((c for c in cas["charges"] if c["ref"] == ref), None)
        if not c:
            return []
        if c["type"] == "surfacique":
            sx, sy = _vers_ecran(pl, c["x0"], c["y0"])
        else:
            sx, sy = _vers_ecran(pl, c["x"], c["y"])
        sx, sy = d.get("sx", sx), d.get("sy", sy)
        x, y = _vers_moteur(pl, sx, sy)
        if c["type"] == "surfacique":
            c["x0"], c["y0"] = round(x, 3), round(y, 3)
        else:
            c["x"], c["y"] = round(x, 3), round(y, 3)
        cas["selection"] = f"{obj}:{ref}"
        return [f"{ref} déplacé en x = {x:.2f} m, y = {y:.2f} m"]
    if obj == "rep":
        e = index["empreintes"].get(ref)
        rp = cas["repartiteurs"].get(ref)
        if not e or not rp or rp["type"] != "plateau":
            return []
        sx, sy = _vers_ecran(pl, e.x + rp.get("dx", 0.0), e.y + rp.get("dy", 0.0))
        sx, sy = d.get("sx", sx), d.get("sy", sy)
        x, y = _vers_moteur(pl, sx, sy)
        rp["dx"], rp["dy"] = round(x - e.x, 3), round(y - e.y, 3)
        cas["selection"] = f"rep:{ref}"
        return [f"plateau sous {ref} décalé de {rp['dx'] * 1e3:.0f} / {rp['dy'] * 1e3:.0f} mm"]
    if obj == "fil":
        f = next((f for f in cas.get("filieres", []) if f["id"] == ref), None)
        if not f:
            return []
        sx, sy = _vers_ecran(pl, f["x"], f["y"])
        sx, sy = d.get("sx", sx), d.get("sy", sy)
        x, y = _vers_moteur(pl, sx, sy)
        f["x"], f["y"] = round(x, 3), round(y, 3)
        cas["selection"] = f"fil:{ref}"
        return [f"filière {ref} déplacée en x = {x:.2f} m, y = {y:.2f} m"]
    return []


def _champ(etat, cas, obj, ref, champ, v, index) -> List[str]:
    if obj == "engin":
        c = next((c for c in cas["charges"] if c["ref"] == ref), None)
        if not c:
            return []
        if champ == "angle":
            c["angle"] = int(float(v)) % 360
            cas["selection"] = f"engin:{ref}"
            return [f"{ref} tourné à {c['angle']}°"]
        if champ.startswith("P."):
            c.setdefault("P", {})[champ[2:]] = float(v)
            return [f"{ref}.{champ[2:]} = {float(v):g} kN"]
        if champ == "action":
            return _action_charge(cas, c, v)
    if obj == "charge":
        c = next((c for c in cas["charges"] if c["ref"] == ref), None)
        if not c:
            return []
        if champ == "action":
            return _action_charge(cas, c, v)
        if champ in ("a", "b", "lx", "ly", "longueur"):
            c[champ] = float(v) / 1e3
        elif champ in ("P", "q", "p", "orientation"):
            c[champ] = v if champ == "orientation" else float(v)
        cas["selection"] = f"charge:{ref}"
        return [f"{ref} : {champ} modifié"]
    if obj == "rep":
        return regler_repartiteur(etat, cas, ref, champ, v)
    if obj == "fil":
        f = next((f for f in cas.get("filieres", []) if f["id"] == ref), None)
        if not f:
            return []
        if champ == "action" and v == "Supprimer":
            supprimer_filiere(cas, ref)
            return [f"filière {ref} supprimée"]
        if champ == "longueur":
            f["longueur"] = max(0.1, float(v) / 1e3)
        elif champ in ("element", "orientation"):
            f["nom" if champ == "element" else "orientation"] = v
        cas["selection"] = f"fil:{ref}"
        return [f"filière {ref} : {champ} modifié"]
    return []


def _action_charge(cas, c, v) -> List[str]:
    if v == "Supprimer":
        cas["charges"] = [x for x in cas["charges"] if x["ref"] != c["ref"]]
        for ref in [r for r in cas["repartiteurs"] if r == c["ref"] or r.startswith(c["ref"] + ".")]:
            cas["repartiteurs"].pop(ref, None)
        cas["selection"] = None
        return [f"{c['ref']} supprimé"]
    if v == "Dupliquer":
        d = copy.deepcopy(c)
        d["ref"] = ref_libre(cas, c["ref"].rstrip("0123456789") or "E")
        if "x" in d:
            d["x"] = round(min(d["x"] + 1.0, cas["plancher"]["L"]), 3)
        cas["charges"].append(d)
        for ref, rp in list(cas["repartiteurs"].items()):
            if ref.startswith(c["ref"] + ".") and rp["type"] == "plateau":
                cas["repartiteurs"][d["ref"] + ref[len(c["ref"]):]] = copy.deepcopy(rp)
        cas["selection"] = f"{'engin' if c['type'] == 'engin' else 'charge'}:{d['ref']}"
        return [f"{c['ref']} dupliqué en {d['ref']}"]
    return []


def regler_repartiteur(etat, cas, ref, champ, v, idem_refs=None) -> List[str]:
    """Règle le répartiteur d'un appui (type, élément, idem) depuis le plan ou l'étape 3."""
    rp = cas["repartiteurs"].get(ref)
    dispo = repartiteurs_disponibles(etat)
    refs = [ref]
    if champ == "idem" and v == "oui":
        engin = ref.split(".")[0]
        refs = [r for r in _refs_empreintes(etat, cas) if r.split(".")[0] == engin]
        champ, v = "type", (rp["type"] if rp else "aucun")
        src = copy.deepcopy(rp)
        for r in refs:
            if src:
                cas["repartiteurs"][r] = copy.deepcopy(src)
            else:
                cas["repartiteurs"].pop(r, None)
        return [f"répartiteur de {ref} appliqué à {len(refs)} appuis"]
    if champ == "type":
        if v == "aucun":
            cas["repartiteurs"].pop(ref, None)
            cas["selection"] = f"rep:{ref}"
            return [f"{ref} : sans répartiteur"]
        if v == "plateau":
            noms = [n for n, d in dispo.items() if d.get("type") == "plateau"]
            nom = rp.get("nom") if rp and rp["type"] == "plateau" else (noms[0] if noms else None)
            if nom is None:
                return []
            cas["repartiteurs"][ref] = {"type": "plateau", "nom": nom, "dx": 0.0, "dy": 0.0}
            cas["selection"] = f"rep:{ref}"
            return [f"{ref} : plateau {nom}"]
        if v == "filiere":
            fid = ajouter_filiere(etat, cas, [ref])
            cas["selection"] = f"fil:{fid}"
            return [f"{ref} : filière {fid}"]
    if champ == "element":
        if rp and rp["type"] == "plateau" and v in dispo and dispo[v].get("type") == "plateau":
            rp["nom"] = v
            cas["selection"] = f"rep:{ref}"
            return [f"{ref} : plateau {v}"]
    return []


def _refs_empreintes(etat, cas) -> List[str]:
    _, _, _, _, index = construire(etat, cas)
    return list(index["empreintes"].keys())


def ajouter_filiere(etat, cas, refs: List[str], nom: Optional[str] = None, longueur: Optional[float] = None,
                    orientation: str = "y") -> str:
    """Crée une filière en travers sous les appuis ``refs`` (même x), centrée
    sur eux, longue de trois éléments par défaut ; les appuis y sont liés."""
    _, _, _, _, index = construire(etat, cas)
    emps = [index["empreintes"][r] for r in refs if r in index["empreintes"]]
    if not emps:
        return ""
    dispo = repartiteurs_disponibles(etat)
    noms = [n for n, d in dispo.items() if d.get("type") == "filiere"]
    nom = nom or ("HEB 140" if "HEB 140" in noms else noms[0])
    plancher, _, _, _, _ = construire(etat, cas)
    larg = plancher.elements()[0].largeur
    longueur = longueur or 3 * larg
    x = sum(e.x for e in emps) / len(emps)
    y = sum(e.y for e in emps) / len(emps)
    if orientation == "y":
        y = min(max(y, longueur / 2), plancher.B - longueur / 2)
    else:
        x = min(max(x, longueur / 2), plancher.L - longueur / 2)
    fid = ref_libre(cas, "F")
    cas.setdefault("filieres", []).append({"id": fid, "nom": nom, "longueur": round(longueur, 3), "orientation": orientation,
                                           "x": round(x, 3), "y": round(y, 3)})
    for r in refs:
        cas["repartiteurs"][r] = {"type": "filiere", "id": fid}
    return fid


def supprimer_filiere(cas, fid: str) -> None:
    cas["filieres"] = [f for f in cas.get("filieres", []) if f["id"] != fid]
    for r in [r for r, rp in cas["repartiteurs"].items() if rp.get("id") == fid]:
        cas["repartiteurs"].pop(r, None)


# ---------------------------------------------------------------- ajouts de charges
def ajouter_engin(etat, cas, nom_gabarit: str, x: Optional[float] = None, y: Optional[float] = None) -> str:
    g = gabarits(etat)[nom_gabarit]
    ref = ref_libre(cas, "E")
    plancher, _, _, _, _ = construire(etat, cas)
    c = {"type": "engin", "ref": ref, "gabarit": nom_gabarit, "x": round(x if x is not None else plancher.L / 2, 3),
         "y": round(y if y is not None else plancher.B / 2, 3), "angle": 0,
         "P": {a["nom"]: float(a.get("P", 0.0) or 0.0) for a in g["appuis"]}}
    cas["charges"].append(c)
    cas["selection"] = f"engin:{ref}"
    return ref


def ajouter_charge(etat, cas, typ: str) -> str:
    plancher, _, _, _, _ = construire(etat, cas)
    if typ == "ponctuel":
        ref = ref_libre(cas, "T")
        cas["charges"].append({"type": "ponctuel", "ref": ref, "x": round(plancher.L / 2, 3), "y": round(plancher.B / 2, 3),
                               "a": 0.15, "b": 0.15, "P": 30.0, "phi": 1.0})
    elif typ == "lineaire":
        ref = ref_libre(cas, "F")
        cas["charges"].append({"type": "lineaire", "ref": ref, "x": round(plancher.L / 2, 3), "y": round(plancher.B / 2, 3),
                               "longueur": 2.4, "largeur": 0.15, "orientation": "y", "p": 10.0, "phi": 1.0})
    else:
        ref = ref_libre(cas, "S")
        cas["charges"].append({"type": "surfacique", "ref": ref, "x0": round(plancher.L / 2 - 1.2, 3), "y0": round(plancher.B / 2 - 0.6, 3),
                               "lx": 2.4, "ly": 1.2, "q": 15.0})
    cas["selection"] = f"charge:{ref}"
    return ref


# ---------------------------------------------------------------- fichier
def payload(etat: Dict) -> Dict:
    return {"module": "charges_plancher", "version": VERSION, "etat": copy.deepcopy(etat)}


def charger(data: Dict) -> Dict:
    if data.get("module") != "charges_plancher" or "etat" not in data:
        raise ValueError("Ce fichier n'est pas un cas « Charges sur plancher ».")
    e = data["etat"]
    base = nouvel_etat()
    base.update({k: v for k, v in e.items() if k in ("actif", "cas", "fiches")})
    base["fiches"] = {**{"engins": [], "repartiteurs": [], "hourdis": []}, **(e.get("fiches") or {})}
    if not base["cas"]:
        base["cas"] = [nouveau_cas("C1")]
    for cas in base["cas"]:
        ref = nouveau_cas(cas.get("nom", "C"))
        for k, v in ref.items():
            cas.setdefault(k, v)
        for k, v in ref["plancher"].items():
            cas["plancher"].setdefault(k, v)
        for k, v in ref["params"].items():
            cas["params"].setdefault(k, v)
    return base


def texte_json(etat: Dict) -> str:
    return json.dumps(payload(etat), indent=1, ensure_ascii=False)
