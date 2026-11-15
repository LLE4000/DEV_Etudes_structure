# -*- coding: utf-8 -*-
"""La LECTURE de la base des prix (cahier § 5). La base est le fichier de départ, importé une fois, et ce que le module
y ajoute ensuite (nos corrections, nos réglages, l'historique, les retours des chantiers) :

    {"version", "loueurs": [{id, nom, role, contact, catalogues, questions}],
     "cas": [{nom, code_calcul, loueur, type, logique}],          # un CATALOGUE = un type de grue chez un loueur
     "champs_grues": {logique: [{cle, libelle, unite, categorie, fourni_par, essentiel, groupe}]},
     "grues": [{id, cas, modele, <champs>, annee_tarif, source_tarif, annee_par_champ, source_par_champ, …}],
     "postes": [{code, poste, unite, prix_par_cas: {cas: valeur}, annee_par_cas, source, fourni_par, auto_par_defaut}],
     "postes_par_logique": {logique: [code]}, "conditions_loueurs": {loueur: [{n, texte, effet}]},
     "fondations_tableau": [{modele, dalle_sur_sol: {avec_demolition, sans_demolition}, …}],
     "parametres": [{code, libelle, valeur, unite, annee}], "rex_fondations_historique": […],
     # ajouts du module
     "reglages": {…}, "historique": […], "fondations_corrections": {…}, "retours_fondations": […]}

Chaque prix se lit avec son ORIGINE (§ 5.2) : sa valeur, l'année du tarif, la source, la période de validité si le loueur
l'a donnée, et notre CORRECTION éventuelle (la valeur du loueur reste dessous). L'année et la source d'un champ de grue
sont ``annee_par_champ``/``source_par_champ`` s'ils existent pour ce champ, sinon ``annee_tarif``/``source_tarif`` ; celles
d'un poste, ``annee_par_cas`` et ``source``."""
import datetime as dt
from typing import Any, Dict, List, Optional

from . import valeurs as V
from .periode import lire_date

LOGIQUES = ("T", "R", "MS", "MA")
NOMS_LOGIQUES = {"T": "Grue tour", "R": "Grue à montage rapide", "MS": "Grue mobile, grutier à part",
                 "MA": "Grue mobile, grutier compris"}
CLES_OBLIGATOIRES = ("cas", "grues", "postes")


class BaseIllisible(ValueError):
    """Le fichier n'est pas une base des prix."""


def verifier(base: Any) -> Dict:
    """La base reçue, contrôlée : un dictionnaire avec ses catalogues, ses grues et ses postes."""
    if not isinstance(base, dict):
        raise BaseIllisible("ce fichier n'est pas une base des prix")
    manque = [k for k in CLES_OBLIGATOIRES if not isinstance(base.get(k), list)]
    if manque:
        raise BaseIllisible("il manque dans ce fichier : " + ", ".join(manque))
    for c in base["cas"]:
        if not isinstance(c, dict) or c.get("code_calcul") not in LOGIQUES or not c.get("nom"):
            raise BaseIllisible("un catalogue du fichier n'a pas de logique de calcul connue")
    return base


# ------------------------------------------------------------------ catalogues, loueurs
def catalogues(base: Dict) -> List[Dict]:
    return [c for c in base.get("cas") or [] if isinstance(c, dict)]


def catalogue(base: Dict, nom: str) -> Optional[Dict]:
    return next((c for c in catalogues(base) if c.get("nom") == nom), None)


def logique(base: Dict, nom: str) -> Optional[str]:
    c = catalogue(base, nom)
    return c.get("code_calcul") if c else None


def loueurs(base: Dict) -> List[Dict]:
    return [l for l in base.get("loueurs") or [] if isinstance(l, dict)]


def loueur(base: Dict, nom: str) -> Optional[Dict]:
    return next((l for l in loueurs(base) if l.get("nom") == nom), None)


def catalogues_du_loueur(base: Dict, nom: str) -> List[Dict]:
    return [c for c in catalogues(base) if c.get("loueur") == nom]


def conditions(base: Dict, nom_loueur: str) -> List[Dict]:
    return [c for c in (base.get("conditions_loueurs") or {}).get(nom_loueur) or [] if isinstance(c, dict)]


# ------------------------------------------------------------------ grues et champs
def champs(base: Dict, lg: str) -> List[Dict]:
    return [c for c in (base.get("champs_grues") or {}).get(lg) or [] if isinstance(c, dict)]


def champ(base: Dict, lg: str, cle: str) -> Optional[Dict]:
    return next((c for c in champs(base, lg) if c.get("cle") == cle), None)


def champs_prix(base: Dict, lg: str) -> List[Dict]:
    return [c for c in champs(base, lg) if c.get("categorie") == "prix"]


def grues(base: Dict) -> List[Dict]:
    return [g for g in base.get("grues") or [] if isinstance(g, dict)]


def grue(base: Dict, gid: str) -> Optional[Dict]:
    return next((g for g in grues(base) if g.get("id") == gid), None)


def retiree(g: Dict) -> bool:
    return str(g.get("retiree") or "").strip().lower() in ("oui", "true", "1", "x")


def grues_du(base: Dict, cas: str, retirees: bool = False) -> List[Dict]:
    """Les grues d'un catalogue, sans les grues retirées (sauf ``retirees``)."""
    return [g for g in grues(base) if g.get("cas") == cas and (retirees or not retiree(g))]


def _origine(valeur, annee, source, validite, correction, a_verifier, ref) -> Dict:
    p = {"valeur": valeur, "annee": annee, "source": source, "validite": validite or None, "corrige": False,
         "a_verifier": bool(a_verifier), "ref": ref}
    if isinstance(correction, dict) and "valeur" in correction:
        d = lire_date(correction.get("date"))
        p.update(valeur=correction.get("valeur"), valeur_loueur=valeur, annee_loueur=annee, corrige=True,
                 correction=correction, annee=d.year if d else annee, source="correction interne")
    return p


def prix_grue(g: Dict, cle: str) -> Dict:
    """Un champ d'une grue, avec son origine (et notre correction, si elle existe)."""
    apc, spc = g.get("annee_par_champ") or {}, g.get("source_par_champ") or {}
    annee = apc[cle] if cle in apc else g.get("annee_tarif")
    source = spc[cle] if cle in spc else g.get("source_tarif")
    validite = (g.get("validite_par_champ") or {}).get(cle) or g.get("validite")
    return _origine(g.get(cle), annee, source, validite, (g.get("corrections") or {}).get(cle),
                    cle in (g.get("a_verifier") or ()), ("grue", g.get("id"), cle))


# ------------------------------------------------------------------ postes
def postes(base: Dict) -> List[Dict]:
    return [p for p in base.get("postes") or [] if isinstance(p, dict)]


def poste(base: Dict, code: str) -> Optional[Dict]:
    return next((p for p in postes(base) if p.get("code") == code), None)


def postes_de(base: Dict, lg: str) -> List[Dict]:
    """Les postes d'une logique, dans l'ordre de la base."""
    codes = list((base.get("postes_par_logique") or {}).get(lg) or [])
    out = []
    for c in codes:
        p = poste(base, c)
        if p:
            out.append(p)
    return out


def prix_poste(p: Dict, cas: str) -> Dict:
    """Le prix d'un poste dans un catalogue, avec son origine."""
    apc = p.get("annee_par_cas") or {}
    validite = (p.get("validite_par_cas") or {}).get(cas)
    source = (p.get("source_par_cas") or {}).get(cas) or p.get("source")
    return _origine((p.get("prix_par_cas") or {}).get(cas), apc.get(cas), source, validite,
                    (p.get("corrections") or {}).get(cas), cas in (p.get("a_verifier") or ()), ("poste", p.get("code"), cas))


# ------------------------------------------------------------------ paramètres et réglages
def parametre(base: Dict, code: str) -> Optional[Dict]:
    return next((p for p in base.get("parametres") or [] if isinstance(p, dict) and p.get("code") == code), None)


def _val(base, code, defaut):
    p = parametre(base, code)
    v = p.get("valeur") if p else None
    return v if V.est_nombre(v) else defaut


def reglages(base: Dict) -> Dict:
    """Les réglages (§ 4.2) : les paramètres de la base, puis ce que l'utilisateur a changé (``base["reglages"]``)."""
    kwh = {}
    p = parametre(base, "prix_kwh")
    if p and V.est_nombre(p.get("valeur")):
        kwh[str(p.get("annee") or "")] = p["valeur"]
    r = {"jours_par_mois": _val(base, "jours_ouvrables_mois_location", 20),
         "jours_par_mois_grutier": _val(base, "jours_ouvrables_mois_grutier", 20),
         "semaines_par_mois": _val(base, "semaines_par_mois", 4),
         "heures_normales": 8, "jours_grutier": "20 par mois", "conges": {}, "prix_kwh": kwh, "taux_mo": {},
         "taux_actualisation": None, "seuil_ecart": 15, "delai_relance": 15, "mois_rappel": 11,
         "coef_fondation_poutres": _val(base, "coef_fondation_poutres", 1.0),
         "coef_fondation_pieux": _val(base, "coef_fondation_pieux", 1.3),
         "arrondi_fondation_pieux": _val(base, "arrondi_fondation_pieux", 500),
         "jours_par_mois_loueur": {}}
    for k, v in (base.get("reglages") or {}).items():
        if k in ("prix_kwh", "taux_mo", "conges", "jours_par_mois_loueur") and isinstance(v, dict):
            r[k] = {**r[k], **v}
        else:
            r[k] = v
    return r


def par_annee(table: Dict, annee: Optional[int] = None) -> Optional[Dict]:
    """Dans une table {année: valeur} (prix du kWh, taux de main-d'œuvre) : la valeur de l'année demandée, sinon la plus
    récente ; {"valeur", "annee"} ou ``None``."""
    vus = []
    for k, v in (table or {}).items():
        if not V.est_nombre(v):
            continue
        try:
            a = int(k)
        except (TypeError, ValueError):
            a = None
        vus.append((a, v))
    if not vus:
        return None
    if annee is not None:
        for a, v in vus:
            if a == annee:
                return {"valeur": v, "annee": a}
    a, v = max(vus, key=lambda x: (x[0] is not None, x[0] or 0))
    return {"valeur": v, "annee": a}


def prix_kwh(base: Dict) -> Optional[Dict]:
    """Le dernier prix du kWh des réglages, avec son année (une origine de prix)."""
    r = par_annee(reglages(base).get("prix_kwh"))
    if not r:
        return None
    return {"valeur": r["valeur"], "annee": r["annee"], "source": "réglages", "validite": None, "corrige": False,
            "a_verifier": False, "ref": ("reglage", "prix_kwh", r["annee"])}


def jours_par_mois(base: Dict, nom_loueur: Optional[str]) -> float:
    """Les jours facturés par mois : le réglage du loueur s'il en a un, sinon le réglage général (20)."""
    r = reglages(base)
    v = (r.get("jours_par_mois_loueur") or {}).get(nom_loueur or "")
    return float(v) if V.est_nombre(v) and v > 0 else float(r.get("jours_par_mois") or 20)
