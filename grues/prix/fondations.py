# -*- coding: utf-8 -*-
"""Le prix d'une FONDATION de grue (cahier § 6 « Fondation », § 8.4) — pour une grue, un type de fondation (dalle sur sol,
poutres sur sol, sur pieux) et un choix de démolition (sans, avec) :

1. notre CORRECTION de la case (page Fondations) ;
2. la MOYENNE des retours VALIDÉS de nos chantiers : même grue, même type, même choix de démolition (seuls les retours
   validés comptent ; un retour importé reste « À vérifier » jusqu'au clic « Valider ») ;
3. l'ANCIEN TABLEAU : dalle sur sol ; poutres = dalle × 1,0 ; pieux = dalle × 1,3, arrondi au multiple de 500 € (réglages) ;
4. sinon, rien : le prix est MANQUANT.

(Le prix forcé dans un calcul passe avant tout : c'est ``calcul`` qui le décide.)

Un retour garde son ANNÉE : si le taux d'actualisation est rempli, les prix des années passées sont ramenés à l'année en
cours (la case le dit, « actualisé ») ; année inconnue : pas d'actualisation, et c'est signalé.

Écusson d'une case : « À jour » si le retour validé le plus récent est de l'année en cours ou de la précédente ; « Pas à
jour » s'il est plus ancien ; « À vérifier » si l'année est inconnue (anciens retours, tableau) ou s'il n'y a que des
retours à vérifier ; « Corrigé » pour notre correction ; « Manquant » sans prix.

Les ANCIENS RETOURS (``rex_fondations_historique``) se lisent « À vérifier », année inconnue, avec leur choix de
démolition ; les aberrants sont écartés, ceux sans prix ne comptent pas. Leur identifiant est leur rang (H01, H02…)."""
import datetime as dt
import math
from typing import Dict, List, Optional

from . import base as B
from . import ecussons as E
from . import valeurs as V
from .periode import lire_date

TYPES = ("Dalle sur sol", "Poutres sur sol", "Sur pieux")
CLE_TABLEAU = {"Dalle sur sol": "dalle_sur_sol", "Poutres sur sol": "poutres_sur_sol_calcule", "Sur pieux": "sur_pieux_calcule"}

A_VERIFIER, VALIDE, ECARTE, SANS_PRIX = "a_verifier", "valide", "ecarte", "sans_prix"
LIBELLES_STATUT = {A_VERIFIER: "À vérifier", VALIDE: "Validé", ECARTE: "Écarté", SANS_PRIX: "Sans prix"}


def type_fondation(t) -> Optional[str]:
    """Le type de fondation, écrit comme dans la base (« dalle », « pieux »… sont reconnus)."""
    s = str(t or "").strip().lower()
    for k in TYPES:
        if s == k.lower():
            return k
    if "pieu" in s:
        return "Sur pieux"
    if "poutre" in s:
        return "Poutres sur sol"
    if "dalle" in s:
        return "Dalle sur sol"
    return None


def statut(s) -> str:
    t = str(s or "").strip().lower()
    if t.startswith("valid"):
        return VALIDE
    if t.startswith("écart") or t.startswith("ecart") or t.startswith("aberrant"):
        return ECARTE
    if t.startswith("sans prix"):
        return SANS_PRIX
    return A_VERIFIER


def cle_case(modele: str, type_: str, demolition: bool) -> str:
    return f"{modele}|{type_}|{'avec' if demolition else 'sans'}"


# ------------------------------------------------------------------ l'ancien tableau
def _arrondi_multiple(x: float, pas: float) -> float:
    """ARRONDI.AU.MULTIPLE d'Excel : le multiple de ``pas`` le plus proche (moitié vers le haut)."""
    if not pas:
        return x
    return math.floor(x / pas + 0.5) * pas


def tableau(base: Dict, modele: str, type_: str, demolition: bool) -> Optional[float]:
    """Le prix de l'ancien tableau : dalle sur sol lue ; poutres et pieux calculés depuis la dalle (réglages)."""
    ligne = next((f for f in base.get("fondations_tableau") or [] if isinstance(f, dict) and f.get("modele") == modele), None)
    if not ligne:
        return None
    dalle = V.montant((ligne.get("dalle_sur_sol") or {}).get("avec_demolition" if demolition else "sans_demolition"))
    if dalle is None:
        return None
    r = B.reglages(base)
    if type_ == "Dalle sur sol":
        return dalle
    if type_ == "Poutres sur sol":
        return dalle * float(r.get("coef_fondation_poutres") or 1.0)
    if type_ == "Sur pieux":
        return _arrondi_multiple(dalle * float(r.get("coef_fondation_pieux") or 1.3), float(r.get("arrondi_fondation_pieux") or 0))
    return None


# ------------------------------------------------------------------ les retours des chantiers
def _taux_mo(reglages: Dict, annee) -> Optional[float]:
    t = B.par_annee(reglages.get("taux_mo"), int(annee) if V.est_nombre(annee) else None)
    return t["valeur"] if t else None


def cout_retour(r: Dict, reglages: Dict) -> Dict:
    """Le coût d'un retour (§ 8.4) : total des factures + main-d'œuvre de notre équipe (hommes × jours × 8 h × taux de
    l'année). Sans taux : les factures seules, « main-d'œuvre non chiffrée ». → {cout, mo, mo_non_chiffree}."""
    total = V.montant(r.get("total"))
    hj = 0.0
    for v in (r.get("main_oeuvre") or {}).values():
        try:
            h, j = float(v[0] or 0), float(v[1] or 0)
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        hj += h * j
    taux = _taux_mo(reglages, r.get("annee")) if hj else None
    mo = hj * 8 * taux if hj and taux is not None else 0.0
    return {"cout": None if total is None else total + mo, "mo": mo, "mo_non_chiffree": bool(hj) and taux is None}


def retours(base: Dict) -> List[Dict]:
    """Tous les retours, sous une forme commune : {id, origine, grue, type, chantier, repere, annee, statut, sans, avec,
    mo_non_chiffree, demolition_comprise, motif}. ``sans`` / ``avec`` : le coût sans ou avec démolition (``None`` si ce
    retour ne le dit pas)."""
    out = []
    reg = B.reglages(base)
    for i, r in enumerate(base.get("rex_fondations_historique") or []):
        if not isinstance(r, dict):
            continue
        st = statut(r.get("statut"))
        prix = V.montant(r.get("prix_reel"))
        if prix is None and st != ECARTE:
            st = SANS_PRIX
        comprise = bool(r.get("demolition_comprise"))
        out.append({"id": f"H{i + 1:02d}", "origine": "historique", "grue": r.get("grue"), "type": type_fondation(r.get("type")),
                    "chantier": r.get("chantier") or r.get("fondation_realisee"), "repere": None, "annee": r.get("annee"),
                    "statut": st, "sans": None if comprise else prix, "avec": prix if comprise else None,
                    "demolition_comprise": comprise, "mo_non_chiffree": False, "motif": r.get("motif"), "brut": r})
    # les retours reçus des chantiers : réalisation et démolition, rapprochées par (chantier, repère)
    neufs = [r for r in base.get("retours_fondations") or [] if isinstance(r, dict)]
    demol = {(r.get("chantier"), r.get("repere")): r for r in neufs if r.get("sorte") == "demolition" and statut(r.get("statut")) == VALIDE}
    for r in neufs:
        if r.get("sorte") == "demolition":
            continue
        c = cout_retour(r, reg)
        comprise = bool(r.get("demolition_comprise"))
        avec = None
        if c["cout"] is not None:
            if comprise:
                avec = c["cout"]
            else:
                d = demol.get((r.get("chantier"), r.get("repere")))
                cd = cout_retour(d, reg)["cout"] if d else None
                avec = c["cout"] + cd if cd is not None else None
        out.append({"id": r.get("id"), "origine": "chantier", "grue": r.get("grue"), "type": type_fondation(r.get("type")),
                    "chantier": r.get("chantier"), "repere": r.get("repere"), "annee": r.get("annee"),
                    "statut": statut(r.get("statut")) if c["cout"] is not None or statut(r.get("statut")) == ECARTE else SANS_PRIX,
                    "sans": None if comprise else c["cout"], "avec": avec, "demolition_comprise": comprise,
                    "mo_non_chiffree": c["mo_non_chiffree"], "motif": r.get("motif"), "brut": r})
    return out


def actualiser(prix: float, annee, reglages: Dict, aujourdhui: dt.date) -> Dict:
    """Le prix ramené à l'année en cours (taux d'actualisation des réglages, en % par an) : {prix, actualise, inconnue}."""
    t = reglages.get("taux_actualisation")
    try:
        a = int(annee)
    except (TypeError, ValueError):
        return {"prix": prix, "actualise": False, "inconnue": True}
    if not V.est_nombre(t) or not t or a >= aujourdhui.year:
        return {"prix": prix, "actualise": False, "inconnue": False}
    return {"prix": prix * (1 + t / 100.0) ** (aujourdhui.year - a), "actualise": True, "inconnue": False}


def _texte_annees(annees: List[int]) -> str:
    connues = sorted(a for a in annees if a is not None)
    if not connues:
        return "année inconnue"
    if len(set(connues)) == 1:
        return str(connues[0])
    return f"{connues[0]} à {connues[-1]}"


def case(base: Dict, modele: str, type_: str, demolition: bool, aujourdhui: dt.date) -> Dict:
    """La case de la grille (et la ligne fondation du calcul) : {valeur, origine, source, ecusson, chantiers, n, moyenne,
    min, max, actualise, annee_inconnue, a_verifier (retours en attente), correction}."""
    type_ = type_fondation(type_) or type_
    reg = B.reglages(base)
    cote = "avec" if demolition else "sans"
    tous = [r for r in retours(base) if r["grue"] == modele and r["type"] == type_ and r[cote] is not None]
    valides = [r for r in tous if r["statut"] == VALIDE]
    attente = [r for r in tous if r["statut"] == A_VERIFIER]
    out = {"modele": modele, "type": type_, "demolition": demolition, "valeur": None, "origine": None, "source": "",
           "ecusson": E.MANQUANT, "chantiers": [], "n": 0, "moyenne": None, "min": None, "max": None, "actualise": False,
           "annee_inconnue": False, "a_verifier": len(attente), "correction": None, "tableau": tableau(base, modele, type_, demolition)}
    chantiers = []
    for r in valides + attente:
        a = actualiser(r[cote], r["annee"], reg, aujourdhui)
        chantiers.append({"id": r["id"], "chantier": r["chantier"], "repere": r["repere"], "annee": r["annee"], "cout": r[cote],
                          "cout_actualise": a["prix"], "actualise": a["actualise"], "statut": r["statut"],
                          "mo_non_chiffree": r["mo_non_chiffree"], "origine": r["origine"]})
    out["chantiers"] = chantiers
    corr = (base.get("fondations_corrections") or {}).get(cle_case(modele, type_, demolition))
    if isinstance(corr, dict) and V.montant(corr.get("valeur")) is not None:
        d = lire_date(corr.get("date"))
        out.update(valeur=V.montant(corr["valeur"]), origine="correction", source="corrigé", ecusson=E.CORRIGE,
                   correction=corr, annee=d.year if d else None)
        return out
    if valides:
        vals = [c["cout_actualise"] for c in chantiers if c["statut"] == VALIDE]
        annees = [r["annee"] for r in valides]
        n = len(vals)
        moy = sum(vals) / n
        recente = max((int(a) for a in annees if V.est_nombre(a)), default=None)
        if any(not V.est_nombre(a) for a in annees) and recente is None:
            ec = E.A_VERIFIER
        elif recente is None:
            ec = E.A_VERIFIER
        else:
            ec = E.A_JOUR if recente >= aujourdhui.year - 1 else E.PAS_A_JOUR
        out.update(valeur=moy, origine="retours", n=n, moyenne=moy, min=min(vals), max=max(vals), ecusson=ec,
                   actualise=any(c["actualise"] for c in chantiers if c["statut"] == VALIDE),
                   annee_inconnue=any(not V.est_nombre(a) for a in annees),
                   source=(f"{n} chantier · {_texte_annees(annees)}" if n == 1 else f"moyenne de {n} chantiers · {_texte_annees(annees)}"))
        return out
    if out["tableau"] is not None:
        out.update(valeur=out["tableau"], origine="tableau", source="ancien tableau", ecusson=E.A_VERIFIER)
        return out
    return out


def grues_avec_prix(base: Dict) -> List[str]:
    """Les modèles qui ont un prix de fondation (tableau, retour ou correction) — la grille ne montre qu'eux par défaut."""
    vus = []
    for f in base.get("fondations_tableau") or []:
        if isinstance(f, dict) and f.get("modele") not in vus:
            vus.append(f.get("modele"))
    for r in retours(base):
        if r["grue"] and r["grue"] not in vus and r["statut"] in (VALIDE, A_VERIFIER):
            vus.append(r["grue"])
    for k in (base.get("fondations_corrections") or {}):
        m = k.split("|")[0]
        if m not in vus:
            vus.append(m)
    return vus
