# -*- coding: utf-8 -*-
"""Les MODIFICATIONS de la base des prix (cahier § 4.1, § 4.2, § 8.4) — chacune écrite dans l'HISTORIQUE (date, objet,
champ, avant, après, origine, motif, initiales) :

- corriger un prix (une case de grue ou de poste) : la case passe en « Corrigé », la valeur du loueur reste dessous ;
  revenir au prix du loueur ; marquer une case « à vérifier » ;
- retirer une grue (elle disparaît des listes, sans être effacée : les anciens calculs la gardent) ;
- valider ou écarter un retour de chantier (avec un motif) ; corriger le prix d'une case de la grille des fondations,
  revenir au calcul automatique ;
- changer un réglage.

Une valeur saisie se lit comme celle d'un loueur (``valeurs.lire_mot``) : un nombre, « compris », « non disponible »,
« sur demande » ; un texte qui n'est pas un prix est REFUSÉ (ValueError, avec le message à montrer)."""
import datetime as dt
from typing import Any, Dict, Optional, Tuple

from . import base as B
from . import fondations as F
from . import valeurs as V

HISTORIQUE_MAX = 2000
REGLAGES_NOMBRES = {"jours_par_mois": (1, 31), "jours_par_mois_grutier": (1, 31), "semaines_par_mois": (1, 5),
                    "heures_normales": (1, 24), "taux_actualisation": (0, 50), "seuil_ecart": (0, 100),
                    "delai_relance": (1, 365), "mois_rappel": (1, 12), "coef_fondation_poutres": (0.1, 10),
                    "coef_fondation_pieux": (0.1, 10), "arrondi_fondation_pieux": (0, 100000)}
REGLAGES_CHOIX = {"jours_grutier": ("20 par mois", "calendrier réel")}


def _date(date) -> str:
    return (date or dt.date.today()).isoformat() if not isinstance(date, str) else date


def historiser(base: Dict, objet: str, champ: str, avant: Any, apres: Any, origine: str = "saisie", motif: str = "",
               initiales: str = "", date=None) -> None:
    h = base.setdefault("historique", [])
    h.append({"date": _date(date), "objet": objet, "champ": champ, "avant": avant, "apres": apres, "origine": origine,
              "motif": (motif or "").strip()[:200], "initiales": (initiales or "").strip()[:6]})
    if len(h) > HISTORIQUE_MAX:
        del h[: len(h) - HISTORIQUE_MAX]


def lire_valeur(texte: Any) -> Any:
    """Une valeur saisie pour une case de prix, telle que la base la garde ; un texte qui n'est pas un prix est refusé."""
    n, v = V.lire_mot(texte)
    if n == V.TEXTE:
        raise ValueError(f"« {str(texte)[:40]} » n'est pas un prix : un nombre, « compris », « non disponible » ou « sur demande »")
    return v


def _cible(base: Dict, ref: Tuple) -> Tuple[Dict, str, str]:
    """(l'objet qui porte la case, la clé de sa correction, son nom pour l'historique)."""
    sorte, ident, cle = ref
    if sorte == "grue":
        g = B.grue(base, ident)
        if not g:
            raise ValueError("grue inconnue")
        if not B.champ(base, B.logique(base, g.get("cas")) or "", cle) and cle not in g:
            raise ValueError("champ inconnu")
        return g, cle, f"{g.get('modele')} ({g.get('cas')})"
    if sorte == "poste":
        p = B.poste(base, ident)
        if not p:
            raise ValueError("poste inconnu")
        if not B.catalogue(base, cle):
            raise ValueError("catalogue inconnu")
        return p, cle, f"{p.get('poste')} ({cle})"
    raise ValueError("case inconnue")


def _prix(base: Dict, ref: Tuple) -> Dict:
    o, cle, _ = _cible(base, ref)
    return B.prix_grue(o, cle) if ref[0] == "grue" else B.prix_poste(o, cle)


def corriger(base: Dict, ref: Tuple, texte: Any, motif: str = "", initiales: str = "", date=None) -> str:
    """Notre correction d'une case : la valeur du loueur reste dessous (« Revenir au prix du loueur »)."""
    v = lire_valeur(texte)
    o, cle, nom = _cible(base, ref)
    avant = _prix(base, ref)["valeur"]
    o.setdefault("corrections", {})[cle] = {"valeur": v, "motif": (motif or "").strip()[:200], "date": _date(date),
                                            "initiales": (initiales or "").strip()[:6]}
    historiser(base, nom, ref[2] if ref[0] == "grue" else ref[1], avant, v, "correction interne", motif, initiales, date)
    return f"Prix corrigé : {nom}"


def revenir(base: Dict, ref: Tuple, initiales: str = "", date=None) -> str:
    o, cle, nom = _cible(base, ref)
    c = (o.get("corrections") or {}).pop(cle, None)
    if c is None:
        return "Ce prix n'était pas corrigé"
    historiser(base, nom, ref[2] if ref[0] == "grue" else ref[1], c.get("valeur"), _prix(base, ref)["valeur"],
               "retour au prix du loueur", "", initiales, date)
    return f"Prix du loueur repris : {nom}"


def marquer(base: Dict, ref: Tuple, oui: bool, initiales: str = "", date=None) -> str:
    o, cle, nom = _cible(base, ref)
    lst = [x for x in o.get("a_verifier") or [] if x != cle]
    if oui:
        lst.append(cle)
    o["a_verifier"] = lst
    historiser(base, nom, ref[2] if ref[0] == "grue" else ref[1], None, "à vérifier" if oui else "vérifié", "saisie", "", initiales, date)
    return "Marqué à vérifier" if oui else "Plus à vérifier"


def retirer_grue(base: Dict, gid: str, oui: bool, initiales: str = "", date=None) -> str:
    g = B.grue(base, gid)
    if not g:
        raise ValueError("grue inconnue")
    g["retiree"] = "Oui" if oui else None
    historiser(base, f"{g.get('modele')} ({g.get('cas')})", "retiree", None, "Oui" if oui else "", "saisie", "", initiales, date)
    return f"{g.get('modele')} : {'retirée des listes' if oui else 'remise dans les listes'}"


# ------------------------------------------------------------------ fondations
def statut_retour(base: Dict, rid: str, statut: str, motif: str = "", initiales: str = "", date=None) -> str:
    """Valider, écarter (avec un motif) ou remettre « à vérifier » un retour de chantier (ancien : H01…, ou reçu)."""
    if statut not in (F.VALIDE, F.ECARTE, F.A_VERIFIER):
        raise ValueError("statut inconnu")
    r = next((x for x in F.retours(base) if x["id"] == rid), None)
    if r is None:
        raise ValueError("retour inconnu")
    if r["statut"] == F.SANS_PRIX and statut == F.VALIDE:
        raise ValueError("ce retour n'a pas de prix : il ne peut pas compter")
    brut = r["brut"]
    avant = brut.get("statut")
    brut["statut"] = {F.VALIDE: "validé", F.ECARTE: "écarté", F.A_VERIFIER: "à valider"}[statut]
    if motif or statut == F.ECARTE:
        brut["motif"] = (motif or "").strip()[:200]
    historiser(base, f"retour {rid} · {r['grue']} · {r['type']}", "statut", avant, brut["statut"], "fondations", motif, initiales, date)
    return {F.VALIDE: "Retour validé : il compte dans le prix", F.ECARTE: "Retour écarté", F.A_VERIFIER: "Retour remis à vérifier"}[statut]


def corriger_fondation(base: Dict, modele: str, type_: str, demolition: bool, texte: Any, motif: str = "", initiales: str = "",
                       date=None) -> str:
    t = F.type_fondation(type_)
    if not t:
        raise ValueError("type de fondation inconnu")
    k = F.cle_case(modele, t, demolition)
    tab = base.setdefault("fondations_corrections", {})
    if texte is None or (isinstance(texte, str) and not texte.strip()):
        c = tab.pop(k, None)
        if c is not None:
            historiser(base, f"fondation {modele} · {t} · {'avec' if demolition else 'sans'} démolition", "correction", c.get("valeur"),
                       None, "fondations", "retour au calcul automatique", initiales, date)
        return "Retour au calcul automatique"
    v = V.montant(lire_valeur(texte))
    if v is None:
        raise ValueError("une fondation se corrige par un montant")
    avant = (tab.get(k) or {}).get("valeur")
    tab[k] = {"valeur": v, "motif": (motif or "").strip()[:200], "date": _date(date), "initiales": (initiales or "").strip()[:6]}
    historiser(base, f"fondation {modele} · {t} · {'avec' if demolition else 'sans'} démolition", "correction", avant, v, "fondations",
               motif, initiales, date)
    return "Prix de la fondation corrigé"


# ------------------------------------------------------------------ réglages
def regler(base: Dict, cle: str, valeur: Any, annee: Optional[int] = None, initiales: str = "", date=None) -> str:
    """Un réglage (§ 4.2). Les tables par année (prix du kWh, taux de main-d'œuvre) prennent ``annee``."""
    r = base.setdefault("reglages", {})
    avant = B.reglages(base).get(cle)
    if cle in ("prix_kwh", "taux_mo"):
        if annee is None:
            raise ValueError("année manquante")
        t = dict(r.get(cle) or {})
        x = V.lire_nombre(valeur) if isinstance(valeur, str) else valeur
        if x is None or (isinstance(valeur, str) and not valeur.strip()):
            t.pop(str(annee), None)
        elif not V.est_nombre(x) or x < 0:
            raise ValueError("un nombre positif")
        else:
            t[str(annee)] = float(x)
        # (le prix du kWh des paramètres de départ reste lisible : la table des réglages le complète)
        r[cle] = t
    elif cle in REGLAGES_CHOIX:
        if valeur not in REGLAGES_CHOIX[cle]:
            raise ValueError("choix inconnu")
        r[cle] = valeur
    elif cle in REGLAGES_NOMBRES:
        x = V.lire_nombre(valeur) if isinstance(valeur, str) else valeur
        if (x is None or (isinstance(valeur, str) and not valeur.strip())) and cle == "taux_actualisation":
            r[cle] = None
        else:
            a, b = REGLAGES_NOMBRES[cle]
            if not V.est_nombre(x) or not a <= x <= b:
                raise ValueError(f"entre {a} et {b}")
            r[cle] = float(x)
    elif cle == "conges":
        if annee is None or not isinstance(valeur, list):
            raise ValueError("congés illisibles")
        t = dict(r.get("conges") or {})
        t[str(annee)] = [list(p)[:2] for p in valeur if isinstance(p, (list, tuple)) and len(p) >= 2]
        r["conges"] = t
    else:
        raise ValueError("réglage inconnu")
    historiser(base, "réglages", cle + (f" {annee}" if annee is not None else ""), avant if not isinstance(avant, dict) else None,
               valeur, "réglages", "", initiales, date)
    return "Réglage enregistré"
