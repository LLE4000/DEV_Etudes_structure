# -*- coding: utf-8 -*-
"""Les OPÉRATIONS que l'écran d'origine (modules/charges_plancher.py) faisait lui-même, extraites sans écran.

L'écran d'origine ne calcule presque rien : l'état, la construction des objets, la vérification et les dessins vivent
déjà dans ce paquet (etat.py, verification.py, dessin.py). Restaient dans la vue quelques opérations qui décident d'un
résultat ou d'une donnée — on les EXTRAIT ici, mot pour mot, pour que le site « Stab Méthodes » (api_web/plancher.py)
fasse exactement ce que fait l'écran d'origine :

- ``proposer_filiere`` / ``appliquer_proposition`` : « Proposer une filière sous les appuis en défaut » (``_proposer``
  et le bouton « Appliquer la proposition » de ``_carte_resultat``) ;
- ``fiche_engin`` : la fiche d'engin du formulaire « Nouvelle fiche d'engin » (``_dialogue_bibliotheque``) ;
- ``fiche_repartiteur`` : la fiche du formulaire « Nouvel élément de répartition » (``_nouvel_element``) ;
- ``lire_parts_annexe_c`` / ``texte_parts_annexe_c`` : le champ « Parts annexe C » (``_parts_annexe_c``) ;
- ``resume_plancher``, ``barres``, ``non_verifies``, ``total_charges``, ``appuis_engin``, ``balayage`` : les résumés
  et les tableaux que l'écran compose (``_resume_plancher``, ``_carte_resultat``, ``_etape_charges``, ``_balayage``).

La PARITÉ est prouvée par tests/test_charges_plancher_parite.py : l'écran d'origine est exécuté avec un faux module
d'écran qui enregistre au lieu d'afficher, sur de nombreux cas, et chaque sortie est comparée à celle d'ici au bit
près ; des mutations de ce fichier font rougir le test. Les défauts de l'écran d'origine sont REPRODUITS ici (on
extrait, on ne réécrit pas) et écrits dans docs/plancher/CHARGES_PLANCHER_SITE.md ; c'est au site de refuser une
saisie que l'écran d'origine ramenait en silence à une valeur fausse.

Module PUR : bibliothèque standard et ce paquet (numpy par le moteur).
"""
from typing import Dict, List, Optional, Sequence, Tuple

from . import bibliotheque as BIB
from . import etat as ET
from . import verification as V

# les libellés de l'écran d'origine (mêmes textes)
TYPES = {"hourdis": "Hourdis", "dalle": "Dalle BA", "predalle": "Prédalle"}
ETATS_H = {"poses": "Posés", "clavetes": "Clavetés", "chapes": "Chapés"}
FAMILLES = [("flexion", "Flexion"), ("tranchant", "Tranchant"), ("local", "Local"), ("plateau", "Plateaux"),
            ("filiere", "Filières"), ("joint", "Joints")]
RECTANGLE = "— rectangle b × h —"
MATERIAUX = ["S235", "S355", "Bois C24", "Azobé", "OSB", "Autre"]
FY_MATERIAU = {"S235": 235.0, "S355": 355.0}          # la table de l'écran d'origine, telle quelle
TYPES_APPUIS = {"pieds": "stabilisateurs", "roues": "roues", "chenilles": "chenilles + stabilisateurs"}
# les appuis par défaut du formulaire « Nouvelle fiche d'engin » (le tableau de départ de l'écran d'origine)
APPUIS_FICHE_DEFAUT = [
    {"nom": "P1", "dx": 2.3, "dy": -2.3, "a": 0.30, "b": 0.30, "P": 0.0, "phi": 1.0},
    {"nom": "P2", "dx": 2.3, "dy": 2.3, "a": 0.30, "b": 0.30, "P": 0.0, "phi": 1.0},
    {"nom": "P3", "dx": -2.3, "dy": -2.3, "a": 0.30, "b": 0.30, "P": 0.0, "phi": 1.0},
    {"nom": "P4", "dx": -2.3, "dy": 2.3, "a": 0.30, "b": 0.30, "P": 0.0, "phi": 1.0}]
FICHE_ENGIN_DEFAUT = {"longueur": 5.0, "largeur": 2.4, "fleche": 3.0}
FICHE_REP_DEFAUT = {"type": "plateau", "nom": "", "materiau": "S235", "f": 0.0, "E": 210000.0, "masse": 0.0, "source": "",
                    "A": 600.0, "B": 600.0, "t": 20.0, "profil": RECTANGLE}


def fmt(x, nd=2) -> str:
    """Le format des nombres de l'écran d'origine (``_fmt``) : espace des milliers, virgule décimale."""
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


# ---------------------------------------------------------------- résumés et barres
def resume_plancher(cas: Dict) -> str:
    """La ligne de résumé de l'étape Plancher (``_resume_plancher``)."""
    pl = cas["plancher"]
    if pl["type"] == "hourdis":
        h = pl["hourdis"]
        ch = f" + couche de compression {h['chape']['epaisseur'] * 1e3:.0f} mm" if h.get("chape", {}).get("oui") else ""
        typ = (h.get("type_cat") or "").split(" · ")[0] or "Hourdis"
        if not typ.lower().startswith("hourdis"):
            typ = "Hourdis " + typ
        return (f"{typ}{ch} · L {fmt(pl['L'])} m · {h['n_elements']} × {fmt(h['b_element'])} m · "
                f"{ETATS_H.get(h.get('etat'), '')}")
    d = pl[pl["type"]]
    return f"{TYPES[pl['type']]} · L {fmt(pl['L'])} m · largeur {fmt(d['largeur'])} m · h {d['h'] * 1e3:.0f} mm · {d['beton']}"


def barres(res: Dict) -> List[Tuple[str, str, float]]:
    """Les barres de la carte Résultat : par famille, le plus grand taux CHIFFRÉ (statuts ok, attention, nok)."""
    out = []
    for fam, lab in FAMILLES:
        ts = [v["taux"] for v in res.get("verifications", []) if v["famille"] == fam and v["taux"] is not None
              and v["statut"] in ("ok", "attention", "nok")]
        if not ts:
            continue
        out.append((fam, lab, max(ts)))
    return out


def non_verifies(res: Dict) -> List[Dict]:
    """Les lignes « non vérifiées » (paramètre à transcrire ou donnée absente) que la carte Résultat compte."""
    return [v for v in res.get("verifications", []) if v["statut"] == "non_verifie"]


def total_charges(cas: Dict) -> float:
    """Le total de l'en-tête de l'étape Charges, TEL QUE l'écran d'origine le calcule : la somme des charges par appui
    des engins et des charges des étançons. Les charges linéaires (p, kN/m) et surfaciques (q, kN/m²) n'ont pas de
    clé « P » et n'y entrent pas (défaut de l'écran d'origine : l'en-tête dit « kN au total »)."""
    return sum(sum((c.get("P") or {}).values()) if c["type"] == "engin" else c.get("P", 0.0) for c in cas["charges"])


def appuis_engin(cas: Dict, res: Dict, index: Dict, ref_engin: str) -> List[Dict]:
    """Le tableau « Appuis de E1 » de l'étape Charges : par appui, la charge P·φ, le répartiteur, l'élément qui le
    porte (bande [y0, y1[ du résultat) et son taux."""
    lignes = []
    for e in index["engins"][ref_engin].empreintes():
        el = next((x for x in res.get("elements", []) if x["y0"] <= e.y < x["y1"]), None)
        rp = cas["repartiteurs"].get(e.ref)
        rtxt = "—" if not rp else (rp["nom"] if rp["type"] == "plateau" else f"filière {rp['id']}")
        lignes.append({"ref": e.ref, "appui": e.ref.split(".")[-1], "charge": e.P * e.phi, "repartiteur": rtxt,
                       "element": el["nom"] if el else None, "taux": el["taux"] if el else None})
    return lignes


# ---------------------------------------------------------------- balayage
def balayage(etat: Dict, cas: Dict, res: Dict, ref: str) -> Optional[Dict]:
    """La vue Balayage (``_balayage``) : l'enveloppe du taux quand l'engin ``ref`` parcourt la portée, sa position
    actuelle et les positions admissibles ; None si le calcul est refusé (l'écran montre alors le message)."""
    plancher, charges, reps, params, _ = ET.construire(etat, cas)
    if res["statut"] == "refus":
        return None
    b = V.balayer(plancher, charges, reps, ref, params)
    x_act = next(c["x"] for c in cas["charges"] if c["ref"] == ref)
    return {"balayage": b, "x_actuel": x_act, "L": plancher.L}


def legende_balayage(b: Dict) -> str:
    """La légende sous le bandeau du balayage (textes de l'écran d'origine)."""
    if b["x_admissibles"]:
        xa = b["x_admissibles"]
        return (f"Positions admissibles du centre de l'engin : de {fmt(min(xa))} à {fmt(max(xa))} m par pas de 10 cm "
                f"({len(xa)} positions sur {len(b['points'])}).")
    return "Aucune position admissible le long de la portée avec les charges et répartiteurs actuels."


# ---------------------------------------------------------------- parts de l'annexe C
def lire_parts_annexe_c(s: str) -> Optional[Dict[int, float]]:
    """Le champ « Parts annexe C » lu comme l'écran d'origine le lit : couples « k:part » séparés par « , » ou « ; ».
    Un morceau sans « : » est IGNORÉ (« 0:0,5 » devient {0: 0.0}, la virgule décimale coupant le couple : défaut de
    l'écran d'origine, que le site refuse) ; deux « : » ou un nombre illisible lèvent ValueError."""
    s = s.strip()
    parts = {}
    for morceau in s.replace(";", ",").split(","):
        if ":" in morceau:
            k, v = morceau.split(":")
            parts[int(k.strip())] = float(v.strip().replace(",", "."))
    return parts or None


def morceaux_ignores(s: str) -> List[str]:
    """Les morceaux non vides que ``lire_parts_annexe_c`` ignore en silence (sans « : »)."""
    return [m.strip() for m in s.strip().replace(";", ",").split(",") if m.strip() and ":" not in m]


def texte_parts_annexe_c(parts: Optional[Dict]) -> str:
    """Le texte affiché dans le champ (« 0:0.5, 1:0.25 »)."""
    return ", ".join(f"{k}:{v}" for k, v in (parts or {}).items())


# ---------------------------------------------------------------- fiches ajoutées
def fiche_engin(nom: str, typ: str, longueur: float, largeur: float, fleche: float, appuis: Sequence[Dict],
                source: str) -> Dict:
    """La fiche d'engin du formulaire « Nouvelle fiche d'engin » ; un nom vide lève ValueError (« Un nom est
    nécessaire. »)."""
    if not nom.strip():
        raise ValueError("Un nom est nécessaire.")
    return {"nom": nom.strip(), "type_appuis": typ, "generique": False, "source": source,
            "silhouette": {"longueur": float(longueur), "largeur": float(largeur), "fleche": float(fleche)},
            "appuis": [{"nom": str(r["nom"]), "dx": float(r["dx"]), "dy": float(r["dy"]), "a": float(r["a"]), "b": float(r["b"]),
                        "P": float(r["P"]), "phi": float(r["phi"])} for r in appuis]}


def fiche_repartiteur(typ: str, nom: str, materiau: str, f: float, E: float, masse: float, source: str, A: float, B: float,
                      t: float, profil: str) -> Dict:
    """La fiche du formulaire « Nouvel élément de répartition » : f = f saisi, sinon fy de la table S235/S355, sinon
    None (à renseigner) ; plateau : t, A, Bp ; filière : un profilé du catalogue ou un rectangle b × h sur champ."""
    if not nom.strip():
        raise ValueError("Un nom est nécessaire.")
    fy = FY_MATERIAU.get(materiau)
    fiche = {"nom": nom.strip(), "type": typ, "materiau": materiau, "f": (f or fy or None), "E": float(E), "masse": float(masse),
             "source": source}
    if typ == "plateau":
        fiche.update({"t": float(t), "A": float(A), "Bp": float(B)})
    else:
        fiche["section"] = {"profil": profil} if profil != RECTANGLE else {"b": float(A), "h": float(B)}
    return fiche


# ---------------------------------------------------------------- proposition d'une filière
def groupes_en_defaut(cas: Dict, res: Dict, plancher, index: Dict) -> List[List[str]]:
    """Les appuis sous lesquels proposer une filière : ceux d'un élément en défaut (taux > 1) sans répartiteur, ou
    sur un plateau — un groupe par appui (la condition de l'écran d'origine, telle quelle)."""
    defauts = {e["nom"] for e in res["elements"] if e["taux"] > 1.0}
    groupes = []
    for ref, e in index["empreintes"].items():
        el = plancher.element_de(e.y)
        if el and el.nom in defauts and ref not in cas["repartiteurs"] or (el and el.nom in defauts and cas["repartiteurs"].get(ref, {}).get("type") == "plateau"):
            groupes.append([ref])
    return groupes


def proposer_filiere(etat: Dict, cas: Dict, res: Dict, plancher, index: Dict) -> Dict:
    """« Proposer une filière sous les appuis en défaut » : une filière en travers par appui en défaut, le premier
    profilé du catalogue (par masse croissante, longueur de 2 à 5 éléments) qui ramène TOUT le système sous 100 %
    (verification.dimensionner_filiere, P-14). Rend la proposition telle que l'écran d'origine la garde."""
    _, charges, reps, params, _ = ET.construire(etat, cas)
    groupes = groupes_en_defaut(cas, res, plancher, index)
    if not groupes:
        return {"cas": cas["nom"], "trouve": False, "essais": []}
    cands = [(n, 210000e3 * BIB.profils()[n]["Iy"] * 1e-8, BIB.profils()[n]["Wel"]) for n in BIB.candidats_filieres(3.6)]
    d = V.dimensionner_filiere(plancher, charges, reps, groupes, cands,
                               lambda n, l, x, y, refs: BIB.filiere_profil(n, l, x, y, refs=refs), params)
    if not d["trouve"]:
        return {"cas": cas["nom"], "trouve": False, "essais": d["essais"]}
    fils, rep = [], {}
    for fil, refs in zip(d["filieres"], groupes):
        fid = ET.ref_libre(cas, "F") + f"p{len(fils)}"
        nom = fil.nom.split(" · ")[0]
        fils.append({"id": fid, "nom": nom, "longueur": round(fil.longueur, 3), "orientation": "y", "x": round(fil.x, 3), "y": round(fil.y, 3)})
        for r in refs:
            rep[r] = {"type": "filiere", "id": fid}
    return {"cas": cas["nom"], "trouve": True, "filieres": fils, "repartiteurs": rep,
            "taux_max": d["taux_max"], "gouverne": d["gouverne"], "taux_montes": d["taux_montes"],
            "texte": f"{fils[0]['nom']} × {fmt(fils[0]['longueur'])} m sous " + ", ".join(g[0] for g in groupes)}


def phrase_proposition(prop: Dict) -> str:
    """Le texte de la proposition (textes de l'écran d'origine)."""
    if prop["trouve"]:
        montes = ", ".join(f"{m['nom']} {m['avant'] * 100:.0f} → {m['apres'] * 100:.0f} %" for m in prop["taux_montes"]) or "aucun"
        return (f"Proposition : {prop['texte']} → pire taux du système {prop['taux_max'] * 100:.0f} % "
                f"({prop['gouverne']['objet']}). Taux montés chez les voisins : {montes}.")
    return f"Aucune filière du catalogue ne ramène tout le système sous 100 % ({len(prop['essais'])} essais)."


def appliquer_proposition(cas: Dict, prop: Dict) -> None:
    """« Appliquer la proposition » : les filières proposées rejoignent le cas, leurs appuis y sont liés."""
    for f in prop["filieres"]:
        cas.setdefault("filieres", []).append(f)
    for r, rp in prop["repartiteurs"].items():
        cas["repartiteurs"][r] = rp
