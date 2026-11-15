# -*- coding: utf-8 -*-
"""RELIRE le fichier « Mise à jour des prix » renvoyé par un loueur (cahier § 7.2) — sans rien appliquer : le résultat est
la liste à CONTRÔLER avant l'import (``import_loueur.appliquer``).

1. la feuille très cachée ``_META`` : le format, le loueur, l'année, la feuille de chaque catalogue ;
2. dans chaque feuille visible, une ligne dont la colonne A vaut « #cle » donne les clés des colonnes ; chaque ligne
   suivante qui porte un identifiant en colonne A se lit avec ces clés (une clé « _… » est de l'affichage : pas relue) —
   jamais par position ;
3. chaque valeur se compare à la base : un CHANGEMENT (avec son écart en %), une case VIDÉE, une valeur NON LUE (le texte
   du loueur, à saisir à la main), un prix CONFIRMÉ ; plus les nouvelles grues, les grues retirées, les autres
   prestations, les réponses aux questions et les remarques.

Un loueur ne peut modifier que SES prix : une clé d'un autre fournisseur, ou un champ qu'il ne fournit pas, est REFUSÉE
et listée. La valeur comparée est celle du LOUEUR (notre correction reste à part : ``base.prix_grue``)."""
import datetime as dt
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from . import base as B
from . import valeurs as V
from . import xlsx as X
from .fichier_loueur import FORMAT, LOUEUR_DU_CATALOGUE, UNITES_NUMERIQUES

LISTES = ("changements", "cases_videes", "valeurs_non_lues", "prix_confirmes", "nouvelles_grues", "grues_retirees",
          "autres_prestations", "reponses", "remarques", "refus")
CLES_DATES = ("retour_avant", "date", "valable_du", "valable_au")


class FichierRefuse(ValueError):
    """Ce fichier n'est pas une « Mise à jour des prix » lisible pour cette base (le message dit pourquoi)."""


# ------------------------------------------------------------------ les valeurs
def _entier_si_possible(x):
    return int(x) if isinstance(x, float) and x.is_integer() else x


def lire_prix(v) -> Tuple[Any, Optional[str]]:
    """(valeur, texte non lu). Un nombre, 0 (compris), « non disponible », « sur demande », None (vide) ; un texte que
    l'on ne sait pas lire (« ± 30 ») revient en second : il est montré tel quel, à saisir à la main."""
    if v is None:
        return None, None
    if isinstance(v, bool):
        return None, str(v)
    if V.est_nombre(v):
        return v, None
    n, x = V.lire_mot(v)
    if n == V.TEXTE:
        return None, str(v).strip()
    return _entier_si_possible(x), None


def lire_texte(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (dt.date, dt.datetime)):
        return (v.date() if isinstance(v, dt.datetime) else v).isoformat()
    if V.est_nombre(v):
        return str(int(v)) if float(v).is_integer() else str(v)
    s = str(v).strip()
    return s or None


def lire_date(v) -> Optional[str]:
    """Une date de la feuille « À LIRE » (un numéro de série d'Excel, ou un texte jj/mm/aaaa) → « aaaa-mm-jj »."""
    if V.est_nombre(v) and 20000 < v < 80000:
        return X.de_serie(v).isoformat()
    t = lire_texte(v)
    if not t:
        return None
    from .periode import lire_date as ld
    d = ld(t)
    return d.isoformat() if d else t


def lire_oui(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = (lire_texte(v) or "").lower()
    return True if s == "oui" else False if s == "non" else None


def lire_entier(v) -> Optional[int]:
    x, _ = lire_prix(v)
    return int(x) if V.est_nombre(x) else None


def egal(a, b) -> bool:
    if V.est_nombre(a) and V.est_nombre(b):
        return abs(a - b) < 1e-9
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b


# ------------------------------------------------------------------ la lecture
def _meta(feuilles: List[Dict]) -> Dict:
    m = next((f for f in feuilles if f["nom"] == "_META"), None)
    if m is None:
        raise FichierRefuse("Ce fichier n'est pas un fichier « Mise à jour des prix » (pas de feuille _META)")
    out = {}
    for (l, c), v in m["cellules"].items():
        if c == 1:
            out[str(v)] = m["cellules"].get((l, 2))
    if out.get("format") != FORMAT:
        raise FichierRefuse("Ce fichier n'est pas un fichier « Mise à jour des prix »")
    return out


def lire(base: Dict, octets: bytes, nom_fichier: str = "") -> Dict:
    """La liste à contrôler (même forme que la lecture de référence du cahier) : {loueur, loueur_id, annee, fichier,
    empreinte, infos, changements, cases_videes, valeurs_non_lues, prix_confirmes, nouvelles_grues, grues_retirees,
    autres_prestations, reponses, remarques, refus, resume}."""
    try:
        cl = X.lire(octets)
    except X.ClasseurIllisible as e:
        raise FichierRefuse(str(e))
    meta = _meta(cl["feuilles"])
    lo = next((l for l in B.loueurs(base) if str(l.get("id")) == str(meta.get("loueur_id"))), None)
    if lo is None:
        raise FichierRefuse(f"Ce fichier est celui d'un loueur que la base ne connaît pas ({meta.get('loueur')})")
    try:
        annee = int(meta.get("annee"))
    except (TypeError, ValueError):
        raise FichierRefuse("L'année du fichier est illisible")
    nom = lo.get("nom")
    feuille_cas = {k.split(":", 1)[1]: v for k, v in meta.items() if k.startswith("feuille:")}
    code_de = {c["nom"]: c.get("code_calcul") for c in B.catalogues(base)}
    par_logique = base.get("postes_par_logique") or {}
    cats = list(lo.get("catalogues") or [])
    a_des_grues = {c: bool(B.grues_du(base, c, retirees=True)) for c in code_de}
    res = {"loueur": nom, "loueur_id": lo.get("id"), "annee": annee, "fichier": (nom_fichier or "").split("/")[-1],
           "empreinte": hashlib.sha256(octets).hexdigest()[:16], "genere_le": meta.get("genere_le"), "infos": {}}
    for k in LISTES:
        res[k] = []

    def note(objet: Dict, cle: str, avant, apres, cellule: str, est_prix: bool = True):
        if apres is None and avant is not None:
            res["cases_videes"].append({**objet, "cle": cle, "avant": avant, "cellule": cellule})
            return
        if egal(avant, apres):
            if apres is not None and est_prix:
                res["prix_confirmes"].append({**objet, "cle": cle, "valeur": apres})
            return
        ch = {**objet, "cle": cle, "avant": avant, "apres": apres, "cellule": cellule}
        if V.est_nombre(avant) and V.est_nombre(apres) and avant:
            ch["ecart_pct"] = round(100 * (apres - avant) / avant, 1)
        res["changements"].append(ch)

    for f in cl["feuilles"]:
        if f["etat"] != "visible":
            continue
        lignes: Dict[int, Dict[int, Any]] = {}
        for (l, c), v in f["cellules"].items():
            lignes.setdefault(l, {})[c] = v
        cles = None
        for l in sorted(lignes):
            cases = lignes[l]
            a = cases.get(1)
            if a == "#cle":
                cles = {c: str(v) for c, v in cases.items() if c >= 2 and v not in (None, "")}
                continue
            if a is None or not cles:
                continue
            a = str(a).strip()
            ligne = {k: cases.get(c) for c, k in cles.items()}
            case_de = {k: f"{f['nom']}!{X.ref(l, c)}" for c, k in cles.items()}
            if a.startswith("L:"):
                k = a[2:]
                v = ligne.get("valeur")
                res["infos"][k] = lire_oui(v) if k == "inchanges_valables" else (lire_date(v) if k in CLES_DATES else lire_texte(v))
            elif a.startswith("Q-") or a == "REMARQUE":
                t = lire_texte(ligne.get("reponse"))
                if t:
                    res["reponses"].append({"id": a, "reponse": t})
            elif a.startswith("AUTRE-"):
                vals = {k: ligne.get(k) for k in ("libelle", "type", "unite", "prix", "remarque")}
                if any(lire_texte(v) for v in vals.values()):
                    prix, brut = lire_prix(vals["prix"])
                    res["autres_prestations"].append({"id": a, "libelle": lire_texte(vals["libelle"]), "type": lire_texte(vals["type"]),
                                                      "unite": lire_texte(vals["unite"]), "prix": prix, "prix_non_lu": brut,
                                                      "remarque": lire_texte(vals["remarque"])})
            elif a.startswith("P:"):
                _poste(base, res, note, a, ligne, case_de, lo, cats, code_de, par_logique, a_des_grues)
            elif a == "NOUVEAU":
                cas = feuille_cas.get(f["nom"])
                if not cas or cas not in cats:
                    if any(lire_texte(v) for k, v in ligne.items() if not k.startswith("_")):
                        res["refus"].append({"id": a, "raison": "nouvelle grue dans une feuille inconnue", "cellule": f"{f['nom']}!{l}"})
                    continue
                vals = {}
                for k, v in ligne.items():
                    if k.startswith("_") or v is None or not str(v).strip():
                        continue
                    ch = B.champ(base, code_de.get(cas) or "", k) or {}
                    if ch.get("categorie") == "prix" or ch.get("unite") in UNITES_NUMERIQUES:
                        x, brut = lire_prix(v)
                    elif ch.get("categorie") == "parc" and k != "retiree":
                        x, brut = lire_entier(v), None
                    else:
                        x, brut = lire_texte(v), None
                    vals[k] = x if brut is None else brut
                if vals:
                    res["nouvelles_grues"].append({"cas": cas, "valeurs": vals, "ligne": f"{f['nom']}!{l}"})
            else:
                g = B.grue(base, a)
                if g is None:
                    res["refus"].append({"id": a, "raison": "identifiant inconnu"})
                    continue
                _grue(base, res, note, g, ligne, case_de, lo, code_de)
    res["resume"] = {k: len(res[k]) for k in LISTES}
    return res


def _grue(base, res, note, g, ligne, case_de, lo, code_de):
    nom, a = lo.get("nom"), g.get("id")
    lg = code_de.get(g.get("cas")) or ""
    for k, v in ligne.items():
        if k.startswith("_"):
            continue
        ch = B.champ(base, lg, k)
        if ch and ch.get("categorie") == "identité":
            continue
        if k == "remarque" or (ch and ch.get("categorie") == "texte"):
            if lire_texte(v):
                res["remarques"].append({"id": a, "modele": g.get("modele"), "texte": lire_texte(v)})
            continue
        if not ch or not ((ch.get("fourni_par") == LOUEUR_DU_CATALOGUE and g.get("cas") in (lo.get("catalogues") or []))
                          or ch.get("fourni_par") == nom):
            if v is not None:
                res["refus"].append({"id": a, "cle": k, "raison": "champ non modifiable par ce loueur"})
            continue
        if k == "retiree":
            if lire_oui(v) and not B.retiree(g):
                res["grues_retirees"].append({"id": a, "modele": g.get("modele")})
            continue
        objet = {"type": "grue", "id": a, "modele": g.get("modele")}
        if ch.get("categorie") == "prix":
            apres, brut = lire_prix(v)
            if brut:
                res["valeurs_non_lues"].append({**objet, "cle": k, "texte": brut, "cellule": case_de[k]})
                continue
        elif ch.get("categorie") == "parc":
            apres = lire_entier(v)
        elif V.est_nombre(g.get(k)) or ch.get("unite") in UNITES_NUMERIQUES:
            apres, brut = lire_prix(v)
            if brut:
                apres = brut
        else:
            apres = lire_texte(v)
        note(objet, k, g.get(k), apres, case_de[k], est_prix=ch.get("categorie") == "prix")


def _poste(base, res, note, a, ligne, case_de, lo, cats, code_de, par_logique, a_des_grues):
    nom = lo.get("nom")
    p = B.poste(base, a[2:])
    if not p:
        res["refus"].append({"id": a, "raison": "poste inconnu"})
        return
    ppc = p.get("prix_par_cas") or {}
    for k, v in ligne.items():
        if k.startswith("_"):
            continue
        objet = {"type": "poste", "id": a, "poste": p.get("poste")}
        if k == "remarque":
            if lire_texte(v):
                res["remarques"].append({"id": a, "texte": lire_texte(v)})
            continue
        if k == "unite":
            if p.get("unite_a_confirmer"):
                note(objet, "unite", p.get("unite"), lire_texte(v), case_de[k], est_prix=False)
            continue
        if not k.startswith("prix:"):
            if v is not None:
                res["refus"].append({"id": a, "cle": k, "raison": "colonne inconnue"})
            continue
        cas = k.split(":", 1)[1]
        if cas == "*":       # un poste que ce loueur fournit pour les grues d'un autre loueur
            if p.get("fourni_par") != nom:
                if lire_texte(v) not in (None, "—"):
                    res["refus"].append({"id": a, "cle": k, "raison": "prix d'un autre fournisseur"})
                continue
            apres, brut = lire_prix(v)
            if brut:
                res["valeurs_non_lues"].append({**objet, "cle": k, "texte": brut, "cellule": case_de[k]})
                continue
            for c_ in ppc:
                note({**objet, "cas": c_}, "prix", ppc.get(c_), apres, case_de[k])
            continue
        s_applique = cas in ppc or (cas in code_de and not a_des_grues.get(cas) and p.get("code") in (par_logique.get(code_de.get(cas)) or []))
        if lire_texte(v) == "—" or (not s_applique and v is None):
            continue
        ok = s_applique and ((p.get("fourni_par") == LOUEUR_DU_CATALOGUE and cas in cats) or p.get("fourni_par") == nom)
        if not ok:
            if lire_texte(v) not in (None, "—"):
                res["refus"].append({"id": a, "cle": k, "raison": "prix d'un autre fournisseur"})
            continue
        apres, brut = lire_prix(v)
        if brut:
            res["valeurs_non_lues"].append({**objet, "cas": cas, "cle": k, "texte": brut, "cellule": case_de[k]})
            continue
        note({**objet, "cas": cas}, "prix", ppc.get(cas), apres, case_de[k])
