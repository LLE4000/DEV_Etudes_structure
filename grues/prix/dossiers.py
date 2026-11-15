# -*- coding: utf-8 -*-
"""Le DOSSIER d'un loueur (cahier § 4, § 9) : les prix dont il est responsable, leur écusson, le pourcentage à jour, les prix
manquants, les questions ouvertes — et l'écusson de l'électricité (le prix du kWh des réglages).

Les prix d'un loueur : dans chacun de ses catalogues, chaque champ de prix des grues (non retirées) que « le loueur du
catalogue » fournit, et chaque poste qu'il fournit pour ce catalogue ; plus les postes qu'il fournit pour les grues d'un
AUTRE loueur (le châssis hors standard). Un dossier sans aucun prix connu (rien reçu) est « Pas à jour » ; entre l'envoi
du fichier et l'import de sa réponse, « Demande envoyée »."""
import datetime as dt
from typing import Dict, List, Tuple

from . import base as B
from . import ecussons as E
from . import valeurs as V

FOURNI_LOUEUR = "loueur du catalogue"


def prix_du_loueur(base: Dict, nom: str) -> List[Tuple[Tuple, Dict, bool]]:
    """[(ref, prix, essentiel)] : les cases de prix dont le loueur ``nom`` est responsable."""
    out = []
    for cat in B.catalogues_du_loueur(base, nom):
        cas, lg = cat["nom"], cat["code_calcul"]
        champs = [c for c in B.champs_prix(base, lg) if c.get("fourni_par") == FOURNI_LOUEUR]
        for g in B.grues_du(base, cas):
            for c in champs:
                out.append((("grue", g.get("id"), c["cle"]), B.prix_grue(g, c["cle"]), bool(c.get("essentiel"))))
        for p in B.postes(base):
            if p.get("fourni_par") == FOURNI_LOUEUR and cas in (p.get("prix_par_cas") or {}) and V.nature((p.get("prix_par_cas") or {}).get(cas)) != V.TEXTE:
                out.append((("poste", p.get("code"), cas), B.prix_poste(p, cas), False))
    for p in B.postes(base):
        if p.get("fourni_par") == nom:
            for cas in (p.get("prix_par_cas") or {}):
                out.append((("poste", p.get("code"), cas), B.prix_poste(p, cas), True))
    return out


def dossier(base: Dict, nom: str, aujourdhui: dt.date) -> Dict:
    """{nom, ecusson, pct_a_jour, n_prix, manquants, essentiels_manquants, compte, annees, questions_ouvertes, demande}."""
    prix = prix_du_loueur(base, nom)
    ecs, connus, annees = [], 0, set()
    manquants = essentiels = 0
    for ref, p, ess in prix:
        n = V.nature(p.get("valeur"))
        if n == V.INCONNU:
            manquants += 1
            essentiels += ess
            continue
        if n == V.NON_DISPONIBLE:
            continue
        connus += 1
        e = E.pour_ensemble(E.ecusson(p, aujourdhui), p, aujourdhui)
        ecs.append(e)
        if p.get("annee"):
            annees.add(p["annee"])
    cpt = E.compter(ecs)
    d = (base.get("dossiers") or {}).get(nom) or {}
    l = B.loueur(base, nom) or {}
    ouvertes = sum(1 for q in l.get("questions") or [] if isinstance(q, dict) and not (q.get("reponse") or "").strip() and not q.get("traitee"))
    ec = E.agreger(ecs, vide=E.PAS_A_JOUR)
    if d.get("demande") and not d["demande"].get("reponse_importee"):
        ec = E.DEMANDE
    return {"nom": nom, "ecusson": ec, "pct_a_jour": round(100.0 * cpt[E.A_JOUR] / connus) if connus else 0, "n_prix": len(prix),
            "connus": connus, "manquants": manquants, "essentiels_manquants": essentiels, "compte": cpt,
            "annees": sorted(a for a in annees if V.est_nombre(a)), "questions_ouvertes": ouvertes, "demande": d.get("demande")}


def electricite(base: Dict, aujourdhui: dt.date) -> Dict:
    """L'écusson du prix du kWh (le dernier des réglages)."""
    p = B.prix_kwh(base)
    return {"prix": p, "ecusson": E.ecusson(p, aujourdhui) if p else E.MANQUANT}


def a_faire(base: Dict, aujourdhui: dt.date) -> List[Dict]:
    """La liste « À faire » (§ 1.4) que le moteur sait déjà dire : retours de chantier à valider, questions aux loueurs
    sans réponse, dossiers pas à jour (à partir du mois du rappel annuel). → [{texte, page, cible}]."""
    from . import fondations as F
    out = []
    att = [r for r in F.retours(base) if r["statut"] == F.A_VERIFIER]
    if att:
        out.append({"texte": f"{len(att)} retour{'s' if len(att) > 1 else ''} de chantier à valider", "page": "fondations", "cible": None})
    mois = B.reglages(base).get("mois_rappel") or 11
    for l in B.loueurs(base):
        d = dossier(base, l.get("nom"), aujourdhui)
        if d["questions_ouvertes"]:
            out.append({"texte": f"{l.get('nom')} : {d['questions_ouvertes']} question{'s' if d['questions_ouvertes'] > 1 else ''} sans réponse",
                        "page": "base", "cible": l.get("nom")})
        if aujourdhui.month >= int(mois) and d["ecusson"] == E.PAS_A_JOUR:
            out.append({"texte": f"{l.get('nom')} : fichier des prix {aujourdhui.year + 1} à préparer", "page": "base", "cible": l.get("nom")})
    return out
