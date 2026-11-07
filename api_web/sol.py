# -*- coding: utf-8 -*-
"""Le module « Rigidité du sol » du site : une note de FONDATIONS (chacune avec sa méthode : sondage CPT, profil saisi,
k = q / w, théories comparées, abaque des sols, plat en béton), les champs de saisie, l'import d'un essai (fichier GEF,
CSV, TXT ou tableau collé), le résultat (k encadré et zoné à encoder dans SCIA), les fenêtres de calcul et les dessins —
rendus en fragments HTML.

Le calcul est celui du moteur ``sol`` (sol/rigidite.py, extrait de l'écran Streamlit modules/rigidite_sol.py v5.0 dont
il reproduit chaque nombre — tests/test_sol_moteur.py). Ce fichier MET EN FORME : il ne calcule rien, sinon la
substitution des valeurs dans les formules affichées (contrôlée par tests/test_api_web_sol.py).

Une note : ``{"version": 1, "fondations": [fondation, …], "actif": 0}`` ; une fondation est celle du moteur (voir la
docstring de sol/rigidite.py), plus ``trace`` : l'indice du sondage dessiné (le menu « Sondage » des graphiques de
l'écran Streamlit).
"""
import copy
import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

from sol import rigidite as R

from . import commun as C
from . import dessin_sol as DS

MODULE = "sol"
PAGE = "rigidite-sol"

SB, ST = R.SB, R.ST
# les six modes de l'écran (« Que veux-tu faire ? »), libellés COURTS : le menu tient dans sa colonne sans être coupé
METHODES = (("cpt", "Sondage CPT importé"), ("profil", "Profil saisi à la main"), ("rapide", "k = q / w (tassement connu)"),
            ("theories", "Comparer les théories"), ("abaque", "Abaque des sols"), ("plat", "Plat en béton"))
COURTS = {"cpt": "Sondage CPT", "profil": "Profil saisi", "rapide": "k = q / w", "theories": "Théories", "abaque": "Abaque des sols",
          "plat": "Plat en béton"}
AIDES_MODES = {
    "cpt": "Importer un essai (GEF, CSV, TXT, ou un tableau collé) ou saisir les couches : k est calculé par le tassement, "
           "encadré (les corrélations CPT divergent d'un facteur 2 à 4) et zoné, prêt à encoder dans SCIA.",
    "profil": "Saisir les couches depuis le terrain naturel, avec leur module OEDOMÉTRIQUE M. Choisir un type de sol préremplit "
              "les valeurs.",
    "rapide": "Si un tassement a déjà été calculé par le géotechnicien : k = q / w.",
    "theories": "Sol homogène. Sert à situer un ordre de grandeur, jamais à produire la valeur retenue.",
    "abaque": "Les coefficients tabulés sont des valeurs mesurées à la PLAQUE de 0,30 m ; la correction de taille de Terzaghi "
              "est appliquée pour la largeur choisie.",
    "plat": "Contact assimilé à une compression 1D : k = E / h<sub>c</sub>.",
}
MEMO = ("<b>Winkler</b> : q = k·w, donc k = q / w ; q en kPa, w en m, k en kN/m³ (1 MN/m³ = 1 000 kN/m³).<br>"
        "<b>k n'est pas une propriété du sol.</b> Sur un même terrain, il varie d'un facteur 5 à 10 entre une semelle isolée et un "
        "grand radier : il se calcule POUR une fondation donnée — c'est pourquoi la géométrie est demandée.<br>"
        "<b>Méthode</b> : Δσ(z) = q<sub>net</sub>·I(z) (Boussinesq/Newmark), puis w = Σ Δσ<sub>i</sub>·h<sub>i</sub> / M<sub>i</sub>, "
        "puis k = q<sub>net</sub> / w.<br>"
        "<b>M est le module OEDOMÉTRIQUE</b>, pas le module de Young : M = E(1 − ν) / [(1 + ν)(1 − 2ν)] ≈ 1,35·E pour ν = 0,30.<br>"
        "<b>Profondeur d'influence</b> : elle se calcule (Δσ ≤ 20 %·σ'<sub>v0</sub>), elle ne se décrète pas à 2B. Sous un radier de "
        "20 m, 2B = 40 m n'a aucun sens.<br>"
        "<b>Un k trop faible n'est pas « du côté de la sécurité »</b> : il étale la charge, augmente les tassements calculés mais "
        "diminue souvent les moments de pointe sous poteaux. D'où l'encadrement systématique, à passer dans les deux bornes.<br>"
        "<b>Rocher</b> : le CPT est en refus de pointe — caractériser par RQD/pressiomètre, jamais par une corrélation sur qc.")
PIED = ("Valeurs de pré-dimensionnement. Les modules déduits d'un CPT sont des corrélations : elles divergent d'un facteur 2 à 4 "
        "selon les auteurs, d'où l'encadrement systématique. Se référer au rapport géotechnique et à l'EN 1997 pour le "
        "dimensionnement final.")
LEGENDE_DIFFUSION = ("À gauche le profil, à droite les contraintes. L'aire rouge est ce qui comprime réellement le sol : elle décroît "
                     "avec la profondeur au lieu de rester égale à q. Le calcul s'arrête où la courbe rouge croise la courbe verte — "
                     "c'est la profondeur d'influence, elle n'est pas imposée.")
LEGENDE_KB = ("À sol identique, k varie fortement avec la taille de la fondation. C'est pourquoi une valeur de k n'a de sens qu'avec "
              "la géométrie qui l'accompagne.")
SANS_FS = ("Pas de frottement latéral fs dans ce fichier : la classification automatique du sol (Robertson) a besoin de qc et de fs. "
           "Les couches seront créées d'après qc seul et le type de sol restera à choisir.")
NOTE_DIFFUSION = ("Δσ(z) = q<sub>net</sub>·I(z), avec I(z) = 4·I<sub>coin</sub>(B/2 ; L/2 ; z) au centre (Boussinesq intégré par "
                  "Newmark), sommé par tranches de 5 cm sous l'assise.")
TABLE_VIDE = ('<table class="vtab"><thead><tr><th>Vérification</th><th class="fo">Formule</th><th class="n">Valeur</th>'
              '<th class="n">Limite</th><th class="n">Taux</th></tr></thead><tbody></tbody></table>')


# ------------------------------------------------------------------ l'état de la note : des fondations
def _fondation(nom: str) -> Dict:
    f = R.nouvelle_fondation(nom)
    f["trace"] = 0
    return f


def nouvel_etat() -> Dict:
    return {"version": 1, "fondations": [_fondation("F1")], "actif": 0}


def _relire(f) -> Dict:
    """Une fondation relue par le moteur (bornes de l'écran, types connus…) ; ``trace`` ramené à un sondage existant."""
    if not isinstance(f, dict):
        raise ValueError("une fondation doit être un objet")
    t = f.get("trace", 0)
    g = R.normaliser({k: v for k, v in f.items() if k != "trace"})
    t = int(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else 0
    g["trace"] = min(max(0, t), len(g["sondages"]) - 1)
    return g


def normaliser(etat: Dict) -> Dict:
    """L'état reçu (navigateur, fichier .stab) relu par le moteur ; une fondation illisible lève une erreur claire."""
    fs = etat.get("fondations") if isinstance(etat, dict) else None
    if not fs:
        return nouvel_etat()
    if not isinstance(fs, list):
        raise ValueError("note abîmée : la liste des fondations est illisible")
    out = {"version": 1, "fondations": [_relire(f) for f in fs]}
    a = etat.get("actif", 0)
    a = int(a) if isinstance(a, (int, float)) and not isinstance(a, bool) else 0
    out["actif"] = min(max(0, a), len(out["fondations"]) - 1)
    return out


def _f(etat: Dict) -> Dict:
    return etat["fondations"][etat["actif"]]


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les fondations de la note : en ouvrir une, en ajouter, copier l'ouverte, en supprimer une (jamais la dernière)."""
    fs = etat["fondations"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(fs) - 1)
    elif action == "ajouter":
        fs.append(_fondation(f"F{len(fs) + 1}"))
        etat["actif"] = len(fs) - 1
    elif action == "dupliquer":
        g = copy.deepcopy(fs[etat["actif"]])
        g["nom"] = f"{g.get('nom') or 'Fondation'} (copie)"
        fs.append(g)
        etat["actif"] = len(fs) - 1
    elif action == "supprimer" and len(fs) > 1:
        i = min(max(0, int(index or 0)), len(fs) - 1)
        del fs[i]
        if etat["actif"] > i or etat["actif"] >= len(fs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
NOMBRES = {"B", "L", "D", "q", "nappe", "nu"}
CASES = {"nappe_active", "q_net"}
BLOCS = {"rapide", "theories", "abaque", "plat"}
CHAMPS_COUCHE = {"h", "qc", "rf", "M", "gamma"}
REGLAGES_IMPORT = {"nappe", "seuil", "h_min"}


def _nombre(valeur) -> float:
    x = C.lire_nombre(valeur)
    if x is None:
        raise ValueError(f"nombre illisible : « {valeur} »")
    return x


def _entier(valeur) -> int:
    x = _nombre(valeur)
    if x != int(x):
        raise ValueError(f"nombre entier attendu : « {valeur} »")
    return int(x)


def _case(valeur) -> bool:
    if isinstance(valeur, bool):
        return valeur
    if valeur in ("true", "1", 1):
        return True
    if valeur in ("false", "0", 0, "", None):
        return False
    raise ValueError(f"case : vrai ou faux attendu ({valeur!r})")


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la fondation ouverte, avec les effets de bord de l'écran Streamlit (le type d'une couche
    préremplit qc, Rf, M et γ ; « Remplir le tableau » reporte la nappe sur la fondation…). Le moteur relit la fondation :
    une valeur hors des bornes de l'écran est REFUSÉE avec un message (Streamlit la ramenait à la borne sans le dire) et la
    fondation reste intacte. Les boutons passent par le même chemin (« sondages/+ », « sondages/0/copie »,
    « sondages/0/import/valider »…)."""
    i = etat["actif"]
    f = etat["fondations"][i]
    avant = copy.deepcopy(f)
    ch = [int(c) if isinstance(c, str) and c.isdigit() else c for c in chemin]
    try:
        msg = _ecrire(f, ch, valeur)
        if msg:
            etat["fondations"][i] = avant
            return msg
        etat["fondations"][i] = _relire(f)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        etat["fondations"][i] = avant
        return f"Saisie refusée : {e}"
    return None


def _ecrire(f: Dict, ch: List, valeur) -> Optional[str]:
    k0 = ch[0]
    if ch == ["nom"]:
        f["nom"] = str(valeur or "")
    elif ch == ["mode"]:
        if valeur not in R.LIBELLES_MODES:
            return f"Méthode inconnue : « {valeur} »"
        f["mode"] = valeur
    elif ch == ["critere"]:
        f["critere"] = _entier(valeur)
    elif ch == ["position"]:
        f["position"] = str(valeur)
    elif ch == ["trace"]:
        t = _entier(valeur)
        if not 0 <= t < len(f["sondages"]):
            return "Sondage inconnu"
        f["trace"] = t
    elif len(ch) == 1 and k0 in NOMBRES:
        f[k0] = _nombre(valeur)
    elif len(ch) == 1 and k0 in CASES:
        f[k0] = _case(valeur)
    elif len(ch) == 2 and k0 in BLOCS and ch[1] in f[k0]:
        f[k0][ch[1]] = _case(valeur) if isinstance(f[k0][ch[1]], bool) else _nombre(valeur)
    elif ch == ["streamlit"]:
        return _reprendre_streamlit(f, valeur)
    elif k0 == "sondages":
        return _ecrire_sondage(f, ch[1:], valeur)
    else:
        return f"Champ inconnu : {'/'.join(map(str, ch))}"
    return None


def _reprendre_streamlit(f: Dict, valeur) -> Optional[str]:
    """« 📂 Ouvrir » de l'écran Streamlit : son fichier raideur_sol.json remplace la fondation ouverte (son nom gardé)."""
    if not isinstance(valeur, dict) or "texte" not in valeur:
        return "Fichier illisible"
    try:
        payload = json.loads(valeur["texte"])
    except (ValueError, TypeError) as e:
        return f"Fichier refusé : ce n'est pas un fichier JSON ({e})"
    try:
        g = R.depuis_payload(payload, f.get("nom") or "F1")
    except (ValueError, KeyError, TypeError) as e:
        return f"Fichier refusé : {e}"
    f.clear()
    f.update(g)
    f["trace"] = 0
    return None


def _ecrire_sondage(f: Dict, ch: List, valeur) -> Optional[str]:
    if ch == ["+"]:
        R.ajouter_sondage(f)
        return None
    i = ch[0]
    s = f["sondages"][i]
    reste = ch[1:]
    if reste == ["-"]:
        if not R.supprimer_sondage(f, i):
            return "Il faut au moins un sondage"
        if f["trace"] > i or f["trace"] >= len(f["sondages"]):
            f["trace"] = max(0, f["trace"] - 1)
    elif reste == ["copie"]:
        R.copier_sondage(f, i)
    elif reste == ["nom"]:
        s["nom"] = str(valeur or "")
    elif reste == ["fichier"]:
        if not isinstance(valeur, dict) or "texte" not in valeur:
            return "Fichier illisible"
        try:
            s["import"] = R.lire_essai(str(valeur.get("nom") or "essai"), str(valeur["texte"]), f["nappe"])
        except ValueError as e:
            return f"Lecture impossible : {e}"
        s["colle"] = ""
    elif reste == ["colle"]:
        texte = str(valeur or "")
        if texte == s["colle"]:
            return None
        s["colle"] = texte
        en_attente = s["import"] is not None and s["import"].get("origine") == "colle"
        if not texte.strip():
            if en_attente:
                s["import"] = None
            return None
        try:
            s["import"] = R.lire_tableau_colle(texte, s["nom"] or "Tableau", f["nappe"])
        except ValueError:
            if en_attente:
                s["import"] = None                       # l'erreur s'affiche sous la zone, le texte est gardé
    elif reste and reste[0] == "import":
        imp = s["import"]
        if imp is None:
            return "Aucun essai en attente"
        if reste[1:] == ["valider"]:
            try:
                R.valider_import(f, i)
            except ValueError as e:
                return f"Import impossible : {e}"
            f["trace"] = i
        elif reste[1:] == ["annuler"]:
            if imp.get("origine") == "colle":
                s["colle"] = ""
            s["import"] = None
        elif len(reste) == 2 and reste[1] in REGLAGES_IMPORT:
            imp[reste[1]] = _nombre(valeur)
        else:
            return "Réglage d'import inconnu"
    elif reste and reste[0] == "couches":
        if reste[1:] == ["+"]:
            R.ajouter_couche(s)
            return None
        j = reste[1]
        c = s["couches"][j]
        if reste[2:] == ["-"]:
            if not R.supprimer_couche(s, j):
                return "Il faut au moins une couche"
        elif reste[2:] == ["type"]:
            if valeur not in R.TYPES:
                return f"Type de sol inconnu : « {valeur} »"
            R.appliquer_type(c, valeur)
        elif len(reste) == 3 and reste[2] in CHAMPS_COUCHE:
            c[reste[2]] = _nombre(valeur)
        else:
            return "Champ de couche inconnu"
    else:
        return "Champ de sondage inconnu"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Pas de sélecteur dans ce module (aucun choix dans un catalogue à dimensionner) : sans effet."""
    return None


def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Pas de sélecteur dans ce module : une liste vide, au format de la brique commune."""
    return C.selecteur(role, "Sans objet", "ce module n'a pas de choix dans un catalogue", "", [], 0, "")


# ------------------------------------------------------------------ nombres
def _k(x: float) -> str:
    """k en MN/m³, au centième (comme l'écran Streamlit)."""
    return C.fr(x, 2)


def _kn(x: float) -> str:
    """k en kN/m³ (valeur en MN/m³ × 1 000), entier."""
    return C.fr(x * 1000.0, 0)


def _plage(a: float, b: float, nd: int = 2) -> str:
    sa, sb = C.fr(a, nd), C.fr(b, nd)
    return sa if sa == sb else f"{sa} à {sb}"


def _plage_courte(a: float, b: float, nd: int = 2) -> str:
    sa, sb = C.fr(a, nd), C.fr(b, nd)
    return sa if sa == sb else f"{sa}–{sb}"


def _texte_fr(t: str) -> str:
    """Le motif d'arrêt du moteur (texte de sol_theorie, écrit à l'anglaise : « 2.00 m », « 20% ») à la française :
    « 2,00 m », « 20 % ». NON échappé : pour les formules (C.lignes_fx échappe)."""
    t = re.sub(r"(\d)\.(\d)", r"\1,\2", str(t))
    return re.sub(r"(\d)%", "\\1\u202f%", t)


def _conv(t: str) -> str:
    """_texte_fr, échappé : pour le HTML posé tel quel."""
    return C.esc(_texte_fr(t))


def _v(x: float, nd: int = 2) -> str:
    """Une donnée dans une formule : sans zéros inutiles (2 reste 2, 0,25 reste 0,25)."""
    return DS.nombre(x, nd)


# ------------------------------------------------------------------ les briques propres à un module de consultation
def _ligne_k(nom: str, piece: str, gauche: float, droite: float, texte: str, calcul: str = "", gov: bool = False, niv: str = "") -> str:
    """Une ligne du résultat SANS taux (k n'est pas vérifié, il est calculé) : le nom et la pièce, puis la plage de k sur
    une barre commune à la note (``gauche`` et ``droite`` en fractions de la barre) et sa valeur. Même structure et mêmes
    classes que C.ligne_res, qui ne sait écrire qu'un taux."""
    ouvre = f' data-calcul="{C.esc(calcul)}" title="Voir le calcul"' if calcul else ""
    g = min(max(gauche, 0.0), 1.0) * 100
    w = max(min(max(droite, 0.0), 1.0) * 100 - g, 1.4) if droite > 0 else 0.0
    bar = f'<div class="bar sol-bar"><b class="sol-k{" " + niv if niv else ""}" style="margin-left:{g:.1f}%;width:{w:.1f}%"></b></div>'
    return (f'<div class="el{" gov" if gov else ""}"{ouvre}><div class="l1"><b>{C.esc(nom)}</b><span class="prof"><span>{piece}</span></span></div>'
            f'<div class="l2">{bar}<span class="pct sol-kv{" " + niv if niv else ""}">{texte}</span></div></div>')


def _verdict(valeur: str, titre: str, texte: str, niv: str = "", unite: str = "MN/m³", lettre: str = "k") -> str:
    """L'en-tête du résultat : la valeur de k en grand (pas un taux), ce qu'elle est, d'où elle vient."""
    u = f'<small class="u">{unite}</small>' if unite else ""
    lt = f"<small>{lettre}</small>" if lettre else ""
    return (f'<div class="verdict sol-v"><span class="chip xl sol-k {niv}">{lt}{valeur}{u}</span>'
            f'<div class="gov"><b>{titre}</b>{texte}</div></div>')


def _fenetre(ident: str, titre: str, sous: str, hyps: List[str], table: str, detail: str, pied: str = "") -> str:
    """La fenêtre de calcul commune (C.fenetre_calcul : hypothèses, tableau, détail des formules), son tableau remplacé
    par celui du module : des grandeurs calculées, pas des vérifications à un taux."""
    h = C.fenetre_calcul(ident, titre, sous, hyps, [], detail, pied)
    if TABLE_VIDE not in h:                                     # pragma: no cover — la brique commune a changé
        raise RuntimeError("fenetre_calcul : le tableau vide attendu n'y est plus")
    return h.replace(TABLE_VIDE, table)


def _vtab(entetes: List[Tuple[str, bool]], lignes: List[List[str]], gov: int = -1) -> str:
    th = "".join(f'<th class="{"n" if n else ""}">{t}</th>' for t, n in entetes)
    tr = "".join(f'<tr class="{"gov" if r == gov else ""}">' + "".join(
        f'<td class="{("n v" if k == len(li) - 1 else "n") if entetes[k][1] else ("nm" if k == 0 else "fo")}">{c}</td>'
        for k, c in enumerate(li)) + "</tr>" for r, li in enumerate(lignes))
    return f'<table class="vtab sol-vtab"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def _groupe(titre: str, lignes: List[Dict], note: str = "") -> str:
    n = f'<div class="sol-note">{note}</div>' if note else ""
    return f'<div class="ck-g"><div class="ck-t">{titre}</div>{n}{C.lignes_fx(lignes)}</div>'


def _fx(parts: List[str], res: str) -> Dict:
    return {"parts": parts, "res": res, "niveau": ""}


def _lien_csv(nom_fichier: str, texte: str, libelle: str) -> str:
    """Un export CSV : un lien de téléchargement (le contenu en URL data:, encodé) — le bouton « ⬇️ CSV » de l'écran."""
    return (f'<a class="btn sm sol-dl" download="{C.esc(nom_fichier)}" href="data:text/csv;charset=utf-8,{quote(texte, safe="")}">'
            f'{C.icon("download")}{C.esc(libelle)}</a>')


# ------------------------------------------------------------------ les calculs de la note (le moteur)
def calculer(f: Dict) -> Dict:
    """Le moteur sur une fondation, selon sa méthode ; rend ``{"mode", "res" | "x", "niveau", "court", "k"}`` —
    ``k`` : (bas, haut) du résumé, ou None."""
    m = f["mode"]
    if m in ("cpt", "profil"):
        res = R.resultats_profil(f)
        env = res["enveloppe"]
        epuise = any("épuisé" in r["convergence"] for r in res["resultats"])
        if not res["resultats"] or res["ecartes"] or res["incalculables"] or epuise:
            niv = "c"                                            # à compléter (un profil épuisé : k surestimé)
        elif env is None:
            niv = "x"
        else:
            niv = "i"
        return {"mode": m, "res": res, "niveau": niv, "k": (env["k_min"], env["k_max"]) if env else None}
    if m == "rapide":
        x = R.calcul_rapide(f["rapide"]["q"], f["rapide"]["w"])
        return {"mode": m, "x": x, "niveau": "i" if x["etat"] == "ok" else "x", "k": (x["k_MNm3"], x["k_MNm3"]) if x["k_MNm3"] > 0 else None}
    if m == "theories":
        x = R.theories(**f["theories"])
        return {"mode": m, "x": x, "niveau": "i" if x["etat"] == "ok" else "x", "k": (x["k_retenu"], x["k_retenu"]) if x["k_retenu"] > 0 else None}
    if m == "abaque":
        x = R.abaque(**f["abaque"])
        return {"mode": m, "x": x, "niveau": "i", "k": None}
    x = R.plat(**f["plat"])
    return {"mode": m, "x": x, "niveau": "i" if x["etat"] == "ok" else "x", "k": (x["keq"] / 1000.0, x["keq"] / 1000.0) if x["keq"] > 0 else None}


def _resume_k(f: Dict, c: Dict) -> str:
    if c["mode"] == "abaque":
        return f"abaque · B = {C.nb(f['abaque']['B'])} m"
    if not c["k"]:
        return "k non calculé"
    kb, kh = c["k"]
    nd = 0 if c["mode"] == "plat" else 2
    return f"k = {_plage(kb, kh, nd)} MN/m³"


def resume_note(etat: Dict, calculs: List[Dict]) -> Dict:
    """Le résumé de la NOTE : une consultation (pas de taux) ; « à compléter » ou « calcul impossible » dès qu'une
    fondation l'est, la fondation nommée."""
    ordre = {"x": 0, "c": 1, "i": 2}
    i = min(range(len(calculs)), key=lambda j: (ordre.get(calculs[j]["niveau"], 2), j != etat["actif"]))
    f, c = etat["fondations"][i], calculs[i]
    pos = ""
    if c["mode"] in ("cpt", "profil") and c["k"]:
        pos = f" ({ST.POSITIONS[f['position']].lower()})"
    return {"taux": None, "niveau": c["niveau"], "statut": C.STATUTS[c["niveau"]][1],
            "gouvernant": f"{f.get('nom') or 'Fondation'} · {_resume_k(f, c)}{pos}"}


def elements(etat: Dict, calculs: List[Dict]) -> str:
    items = ""
    for i, (f, c) in enumerate(zip(etat["fondations"], calculs)):
        if c["k"]:
            kb, kh = c["k"]
            nd = 0 if c["mode"] == "plat" else 1
            txt = _plage_courte(kb, kh, nd)
        else:
            txt = "—" if c["mode"] != "abaque" else "abaque"
        niv = {"c": "c", "x": "r"}.get(c["niveau"], "")
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["fondations"]) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(f.get("nom") or f"Fondation {i + 1}")} '
                  f'<span>· {C.esc(COURTS[f["mode"]])}</span></span><span class="pct {niv}" title="k en MN/m³">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["fondations"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier la fondation ouverte">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une fondation</div></div>')


# ------------------------------------------------------------------ le cadre Données
def _bornes(cle: str) -> Dict:
    lo, hi = R.BORNES.get(cle, (None, None))
    return {"mini": lo, "maxi": hi}


def _n(chemin: List, valeur, unite: str, cle: str, dec: int = 3) -> str:
    return C.num(chemin, valeur, unite, dec=dec, **_bornes(cle))


def _section_fondation(f: Dict) -> str:
    L = C.ligne
    corps = (L("Nom", C.texte(["nom"], f.get("nom"), "ex. F1 · semelle S3"), full=True)
             + L("Méthode", C.choix(["mode"], f["mode"], METHODES), full=True)
             + C.aide(AIDES_MODES[f["mode"]])
             + f'<div class="sol-reprise">{C.fichier(["streamlit"], "Reprendre un fichier de l’écran Streamlit", ".json")}'
               f'<span class="hint">raideur_sol.json, le fichier « Enregistrer » de Streamlit : il remplace cette fondation.</span></div>')
    return C.section(1, f"Fondation {C.esc(f.get('nom') or '')}", C.esc(COURTS[f["mode"]]), corps, True, "fo")


def _section_semelle(f: Dict, p: Dict) -> str:
    L = C.ligne
    corps = (C.aide("k dépend de la fondation : sans elle, le calcul n'a pas de sens.")
             + L("largeur", _n(["B"], f["B"], "m", "B"), "B", "Petit côté de la zone chargée (B et L sont échangés si L < B).")
             + L("longueur", _n(["L"], f["L"], "m", "L"), "L")
             + L("assise sous le TN", _n(["D"], f["D"], "m", "D"), "D",
                 "Profondeur du dessous de fondation sous le terrain naturel. Les couches au-dessus sont automatiquement écartées "
                 "du calcul.")
             + L("pression ELS", _n(["q"], f["q"], "kPa", "q"), "q", "Pression de contact à l'ELS sous la fondation.")
             + C.case(["nappe_active"], f["nappe_active"], "Nappe"))
    if f["nappe_active"]:
        corps += L("nappe sous le TN", _n(["nappe"], f["nappe"], "m", "nappe"), "z<sub>w</sub>")
    if p["echange"]:
        corps += C.aide("B est le petit côté par convention : B et L ont été échangés pour le calcul.")
    # la position de référence SCIA : dans l'écran Streamlit, elle se choisit DANS le panneau des résultats (toujours
    # visible) ; le cadre Résultat du site n'a pas de champ, elle vient donc ici, ouverte, et non dans les options repliées
    corps += L("k SCIA", C.choix(["position"], f["position"], list(ST.POSITIONS.items())), "",
               "La position de référence des valeurs à encoder dans SCIA (paramètres de sol C1z) : centre, milieu d'un côté ou angle. "
               "Le zonage de la dalle donne les quatre.", full=True)
    resume = (f"{C.nb(f['B'])} × {C.nb(f['L'])} m · D {C.nb(f['D'])} m · q {C.nb(f['q'])} kPa"
              + (f" · nappe {C.nb(f['nappe'])} m" if f["nappe_active"] else ""))
    return C.section(2, "Semelle et charge", resume, corps, True, "se")


def _carte_sondage(f: Dict, i: int, s: Dict, r: Optional[Dict]) -> str:
    """La carte d'un sondage dans la liste : nom, couches, profondeur, son k au centre."""
    H = sum(c["h"] for c in s["couches"])
    if r is None:
        etat_, txt = "c", "écarté : aucun M"
    elif r["k_bas"] > 0:
        etat_, txt = "", f"{_plage_courte(r['k_bas'], r['k_haut'])} MN/m³"
    else:
        etat_, txt = "r", "k non calculé"
    trace = ' <span class="sol-trace" title="Sondage dessiné">dessiné</span>' if i == f["trace"] and len(f["sondages"]) > 1 else ""
    return (f'<div class="secard sol-carte{" on" if i == f["trace"] else ""}"><div class="sh2"><span class="indice">{i + 1}</span>'
            f'<span class="nm">{C.esc(s["nom"] or f"Sondage {i + 1}")}</span>{trace}<span class="pct {etat_}">{txt}</span></div>'
            f'<div class="hint">{len(s["couches"])} couche{"s" if len(s["couches"]) > 1 else ""} · {C.fr(H, 2)} m depuis le TN'
            + (f" · {len(s['points'])} mesures CPT" if s["points"] else "") + (" · essai en attente" if s["import"] else "") + "</div></div>")


def _section_sondages(f: Dict, res: Dict) -> str:
    par_index = {r["index"]: r for r in res["resultats"]}
    cartes = "".join(_carte_sondage(f, i, s, par_index.get(i)) for i, s in enumerate(f["sondages"]))
    corps = (f'<div class="secards">{cartes}</div>'
             f'<button type="button" class="addrow" data-chemin="sondages/+" data-valeur="1">{C.icon("plus")}Ajouter un sondage</button>')
    if len(f["sondages"]) > 1:
        corps += C.ligne("Dessiné", C.choix(["trace"], f["trace"], [(i, s["nom"] or f"Sondage {i + 1}") for i, s in enumerate(f["sondages"])]),
                         full=True)
    if f["mode"] == "profil":
        corps += C.aide("Saisir les couches depuis le terrain naturel, avec leur module OEDOMÉTRIQUE M. Choisir un type de sol "
                        "préremplit les valeurs.")
    else:
        corps += C.aide("Plusieurs sondages sous une même fondation : le résultat les enveloppe et zone la dalle.")
    titre = "Profil" if f["mode"] == "profil" else "Sondages"
    return C.section(3, titre, f"{len(f['sondages'])} sondage{'s' if len(f['sondages']) > 1 else ''}", corps, True, "so")


def _legende_couche(c: Dict) -> str:
    inf = R.infos_couche(c)
    out = [f'<span class="ok">{C.icon("check")}prise en compte</span>' if inf["prise_en_compte"]
           else f'<span class="att">{C.icon("triangle-alert")}ignorée (h ou M manquant)</span>']
    if inf["Ic"] is not None:
        out.append(f"I<sub>c</sub> = {C.fr(inf['Ic'], 2)}")
    if inf["sbt"]:
        out.append(f"CPT : {C.esc(inf['sbt'])}")
    if inf["M_encadre"]:
        a, b = inf["M_encadre"]
        out.append(f"M encadré : {C.fr(a, 0)} – {C.fr(b, 0)} MPa")
    elif inf["M_suggere"]:
        out.append(f"M suggéré (qc) : {C.fr(inf['M_suggere'], 0)} MPa")
    elif inf["roche"]:
        out.append("roche : refus de pointe probable, qc sans objet")
    return f'<div class="sol-leg">{" · ".join(out)}</div>'


def _couche(i: int, j: int, c: Dict, z0: float, n: int) -> str:
    base = ["sondages", i, "couches", j]
    retirer = (f'<button type="button" class="moins" data-chemin="{C._attr_chemin(base + ["-"])}" data-valeur="1" title="Retirer la couche">'
               f'{C.icon("x")}</button>') if n > 1 else ""
    desc = (SB.SOIL_DB.get(c["type"]) or {}).get("desc", "")
    aide = f'<span class="aide" title="{C.esc(desc)}">{C.icon("info")}</span>' if desc else ""
    return (f'<div class="secard sol-couche"><div class="sh2"><span class="indice">{j + 1}</span>'
            f'<span class="sol-z">{C.fr(z0, 2)} → {C.fr(z0 + c["h"], 2)} m</span>{aide}{retirer}</div>'
            f'{C.choix(base + ["type"], c["type"], [(t, t) for t in R.TYPES])}'
            f'<div class="trio"><label><span>épaisseur h</span>{_n(base + ["h"], c["h"], "m", "couche.h")}</label>'
            f'<label><span>module M</span>{_n(base + ["M"], c["M"], "MPa", "couche.M")}</label>'
            f'<label><span>poids γ</span>{_n(base + ["gamma"], c["gamma"], "kN/m³", "couche.gamma")}</label></div>'
            f'<div class="trio"><label><span>pointe q<sub>c</sub></span>{_n(base + ["qc"], c["qc"], "MPa", "couche.qc")}</label>'
            f'<label><span>frottement R<sub>f</sub></span>{_n(base + ["rf"], c["rf"], "%", "couche.rf")}</label>'
            f'<label class="sol-vide"></label></div>{_legende_couche(c)}</div>')


def _bloc_import(f: Dict, i: int, s: Dict) -> str:
    """L'import d'un essai : un fichier, ou un tableau collé ; puis l'aperçu (réglages du découpage, couches découpées)
    et « Remplir le tableau » — l'expander « 📥 Importer un sondage » de l'écran."""
    base = ["sondages", i]
    out = ('<div class="grp">Importer un essai</div>'
           + C.aide("Les valeurs sont lues directement dans le fichier, jamais estimées. Par ordre de fidélité : fichier GEF (format "
                    "d'échange belge/néerlandais, pas réel de 2 cm) → CSV → tableau collé. Un PDF à courbes vectorielles se relit dans "
                    "l'écran Streamlit ; un scan ou une photo ne contient aucune donnée lisible.")
           + f'<div class="sol-import">{C.fichier(base + ["fichier"], "Fichier GEF, CSV, TXT", ".gef,.csv,.txt,.asc")}</div>'
           + C.zone_texte(base + ["colle"], s["colle"], 3, "ou coller un tableau : profondeur [m] ; qc [MPa] ; fs [kPa]"))
    imp = s["import"]
    if s["colle"].strip() and (imp is None or imp.get("origine") != "colle"):
        try:
            R.lire_tableau_colle(s["colle"], s["nom"] or "Tableau", f["nappe"])
        except ValueError as e:
            out += C.message(f"Tableau collé illisible : {C.esc(e)}", True)
    if imp is None:
        return out
    pts = imp["points"]
    out += (f'<div class="secard sol-attente"><div class="sh2">{C.icon("file-text")}<span><b>{C.esc(imp["nom"])}</b> — {C.esc(imp["source"])} · '
            f'{len(pts)} points de {C.fr(pts[0][0], 2)} à {C.fr(pts[-1][0], 2)} m</span></div>')
    out += "".join(C.message(C.esc(a)) for a in imp["avertissements"])
    if not any(p[2] is not None for p in pts):
        out += C.message(SANS_FS)
    b = base + ["import"]
    out += (C.ligne("nappe sous le TN", _n(b + ["nappe"], imp["nappe"], "m", "import.nappe"), "z<sub>w</sub>")
            + C.ligne("Finesse", C.choix(b + ["seuil"], imp["seuil"], [(x, C.fr(x, 2)) for x in R.SEUILS_IC]), full=True,
                      aide="Écart d'indice Ic au-delà duquel on ouvre une nouvelle couche. Plus petit = plus de couches.")
            + C.ligne("épaisseur mini", _n(b + ["h_min"], imp["h_min"], "m", "import.h_min"), "h<sub>min</sub>",
                      "Épaisseur minimale d'une couche découpée : une couche plus mince est fusionnée avec sa voisine la plus proche en Ic."))
    try:
        ap = R.apercu_import(imp, f["nu"])
        erreur = None
    except ValueError as e:
        ap, erreur = None, str(e)
    if erreur:
        out += C.message(C.esc(erreur), True)
    elif not ap["couches"]:
        out += C.message("Aucune couche exploitable n'a pu être formée.", True)
    else:
        lignes = "".join(f'<div class="sol-dec"><span>{C.fr(x["de"], 2)} – {C.fr(x["a"], 2)} m</span>'
                         f'<span class="m">{"M " + _plage(x["M_bas"], x["M_haut"], 1) + " MPa" if x["M_bas"] is not None else "sans module"}</span>'
                         f'<b>{C.esc(x["type"] or "—")}</b></div>'
                         for x in ap["lignes"])
        out += (f'<div class="sol-decs">{lignes}</div>'
                + C.aide("Le type de sol est déduit de qc et fs par l'indice de comportement Ic (Robertson) — aucune saisie, aucune "
                         "estimation. Le module M est encadré car les corrélations CPT divergent d'un facteur 2 à 4. Le détail est dans "
                         "la vue « Import » de la toile."))
    actions = ""
    if ap and ap["couches"] and not erreur:
        actions += (f'<button type="button" class="btn sm primary" data-chemin="{C._attr_chemin(b + ["valider"])}" data-valeur="1">'
                    f'{C.icon("check")}Remplir le tableau</button>')
    actions += (f'<button type="button" class="btn sm" data-chemin="{C._attr_chemin(b + ["annuler"])}" data-valeur="1">'
                f'{C.icon("x")}Abandonner</button>')
    liens = _lien_csv(f"{imp['nom']}_mesures.csv", R.csv_mesures(imp["nom"], pts), "CSV des mesures")
    if ap and ap["couches"] and not erreur:
        try:
            liens += _lien_csv(f"{imp['nom']}_couches.csv", R.csv_couches(imp["nom"], ap["couches"]), "CSV des couches")
        except ValueError as e:
            liens += f'<span class="hint">{C.esc(e)}</span>'
    out += f'<div class="sol-actions">{actions}</div><div class="sol-actions">{liens}</div></div>'
    return out


def _section_sondage(n: int, f: Dict, i: int, s: Dict, r: Optional[Dict]) -> str:
    base = ["sondages", i]
    corps = C.ligne("Nom", C.texte(base + ["nom"], s["nom"], "ex. CPT01"), full=True)
    if s["source"]:
        corps += C.aide(f"Source : {C.esc(s['source'])}")
    corps += _bloc_import(f, i, s)
    corps += ('<div class="grp">Couches, depuis le terrain naturel</div>'
              + C.aide("Saisir tout le profil depuis la surface : le niveau d'assise est appliqué par le calcul, il ne faut pas "
                       "retirer les couches supérieures à la main.")
              + C.aide("<b>M</b> est le module OEDOMÉTRIQUE (déformation latérale empêchée) : c'est lui qui entre dans le tassement, "
                       "pas le module de Young E — M = E(1 − ν) / ((1 + ν)(1 − 2ν)), soit ≈ 1,35·E pour ν = 0,30. <b>γ</b> sert au "
                       "poids des terres σ'<sub>v0</sub>, donc à la profondeur d'influence."))
    z = 0.0
    cartes = ""
    for j, c in enumerate(s["couches"]):
        cartes += _couche(i, j, c, z, len(s["couches"]))
        z += c["h"]
    corps += (f'<div class="secards">{cartes}</div>'
              f'<button type="button" class="addrow" data-chemin="{C._attr_chemin(base + ["couches", "+"])}" data-valeur="1">'
              f'{C.icon("plus")}Ajouter une couche</button>')
    pied = (f'<button type="button" class="addrow" data-chemin="{C._attr_chemin(base + ["copie"])}" data-valeur="1">'
            f'{C.icon("copy")}Copier ce sondage</button>')
    if s["points"]:
        pied += _lien_csv(f"{s['nom']}_mesures.csv", R.csv_mesures(s["nom"], s["points"]), "Exporter les mesures brutes (CSV)")
    if len(f["sondages"]) > 1:
        pied += (f'<button type="button" class="btn sm sol-suppr" data-chemin="{C._attr_chemin(base + ["-"])}" data-valeur="1">'
                 f'{C.icon("x")}Supprimer ce sondage</button>')
    corps += f'<div class="sol-actions">{pied}</div>'
    if r is None:
        resume = "écarté : aucun module M"
    elif r["k_bas"] > 0:
        resume = f"k {_plage_courte(r['k_bas'], r['k_haut'])} MN/m³"
    else:
        resume = "k non calculé"
    titre = f"Sondage {C.esc(s['nom'] or str(i + 1))}"
    return C.section(n, titre, f"{len(s['couches'])} couche{'s' if len(s['couches']) > 1 else ''} · {resume}", corps,
                     i == f["trace"] or s["import"] is not None, f"s{s['id']}")


def _section_options(n: int, f: Dict) -> str:
    L = C.ligne
    corps = (L("Critère", C.choix(["critere"], f["critere"], [(v, f"Δσ ≤ {v} % · σ'v0") for v in R.CRITERES]), full=True,
               aide="Critère de profondeur d'influence : 20 % est l'usage courant ; 10 % est plus strict et descend plus profond "
                    "(k plus faible).")
             + L("coefficient de Poisson", _n(["nu"], f["nu"], "", "nu"), "ν",
                 "Sert à convertir le module de Young en module oedométrique dans le découpage d'un essai CPT.")
             + C.case(["q_net"], f["q_net"], "Utiliser la pression NETTE (q − poids des terres excavées)")
             + C.aide("Usage normal en fondation : le sol a déjà supporté le poids des terres retirées."))
    resume = f"Δσ ≤ {f['critere']} % · ν {C.nb(f['nu'])} · pression {'nette' if f['q_net'] else 'brute'}"
    return C.section(n, "Options de calcul", resume, corps, False, "op")


def _section_memo(n: int) -> str:
    return C.section(n, "Fiche mémo", "ce que k est, et ce qu'il n'est pas", f'<div class="sol-memo">{MEMO}</div>', False, "me")


def _section_mode(f: Dict) -> str:
    """La section 2 des modes 3 à 6 : leurs champs (ceux de l'écran Streamlit)."""
    m, L = f["mode"], C.ligne
    if m == "rapide":
        x = f["rapide"]
        corps = (L("pression", _n(["rapide", "q"], x["q"], "kPa", "rapide.q"), "q")
                 + L("tassement", _n(["rapide", "w"], x["w"], "mm", "rapide.w"), "w")
                 + C.aide("Valable si w provient d'un vrai calcul de tassement, pour cette fondation et cette charge."))
        return C.section(2, "k à partir d'un couple (q, w)", f"q {C.nb(x['q'])} kPa · w {C.nb(x['w'])} mm", corps, True, "m3")
    if m == "theories":
        x = f["theories"]
        corps = (L("module de Young", _n(["theories", "E"], x["E"], "MPa", "theories.E"), "E")
                 + L("largeur", _n(["theories", "B"], x["B"], "m", "theories.B"), "B")
                 + L("longueur", _n(["theories", "L"], x["L"], "m", "theories.L"), "L")
                 + L("coefficient de Poisson", _n(["theories", "nu"], x["nu"], "", "theories.nu"), "ν")
                 + C.aide("Sol homogène de 60 m, q = 150 kPa (pression brute), assise au terrain naturel, critère 20 %."))
        return C.section(2, "Comparaison des théories", f"E {C.nb(x['E'])} MPa · {C.nb(x['B'])} × {C.nb(x['L'])} m · ν {C.nb(x['nu'])}",
                         corps, True, "m4")
    if m == "abaque":
        x = f["abaque"]
        corps = (L("largeur de fondation", _n(["abaque", "B"], x["B"], "m", "abaque.B"), "B")
                 + L("tassement de référence", _n(["abaque", "w_ref"], x["w_ref"], "mm", "abaque.w_ref"), "w<sub>réf</sub>",
                     "La dernière colonne de l'abaque donne la pression qui produit ce tassement."))
        return C.section(2, "Abaque des sols", f"B {C.nb(x['B'])} m · w<sub>réf</sub> {C.nb(x['w_ref'])} mm", corps, True, "m5")
    x = f["plat"]
    corps = (L("largeur", _n(["plat", "B"], x["B"], "mm", "plat.B"), "B")
             + L("longueur", _n(["plat", "L"], x["L"], "mm", "plat.L"), "L")
             + L("hauteur mobilisée", _n(["plat", "alpha"], x["alpha"], "", "plat.alpha"), "α", "h_c = α·min(B, L)")
             + L("module du béton", _n(["plat", "E"], x["E"], "GPa", "plat.E"), "E<sub>c</sub>")
             + C.case(["plat", "mortier"], x["mortier"], "Lit de mortier"))
    if x["mortier"]:
        corps += (L("épaisseur du mortier", _n(["plat", "tg"], x["tg"], "mm", "plat.tg"), "t<sub>g</sub>")
                  + L("module du mortier", _n(["plat", "Eg"], x["Eg"], "GPa", "plat.Eg"), "E<sub>g</sub>"))
    return C.section(2, "Raideur d'un plat en béton", f"{C.nb(x['B'])} × {C.nb(x['L'])} mm · α {C.nb(x['alpha'])}"
                     + (" · mortier" if x["mortier"] else ""), corps, True, "m6")


def formulaire(f: Dict, c: Dict) -> str:
    if f["mode"] in ("cpt", "profil"):
        res = c["res"]
        par_index = {r["index"]: r for r in res["resultats"]}
        out = _section_fondation(f) + _section_semelle(f, res["parametres"]) + _section_sondages(f, res)
        for i, s in enumerate(f["sondages"]):
            out += _section_sondage(4 + i, f, i, s, par_index.get(i))
        n = 4 + len(f["sondages"])
        return out + _section_options(n, f) + _section_memo(n + 1)
    return _section_fondation(f) + _section_mode(f) + _section_memo(3)


# ------------------------------------------------------------------ modes 1 et 2 : le résultat
def _sigma_d(couches: List[Dict], D: float) -> Tuple[List[str], float]:
    """Les termes γ·h de σ_v0 à l'assise, dans l'ordre de contraintes_en_place (au-delà du profil : le γ de la dernière
    couche) — pour ÉCRIRE la substitution ; la valeur reste celle du moteur."""
    termes, reste, s = [], D, 0.0
    for c in couches:
        if reste <= 0:
            break
        h = min(c["h"], reste)
        if h <= 0:
            continue
        termes.append(f"{_v(c['gamma'], 2)} × {_v(h, 3)}")
        s += c["gamma"] * h
        reste -= h
    if reste > 0 and couches:
        termes.append(f"{_v(couches[-1]['gamma'], 2)} × {_v(reste, 3)}")
        s += couches[-1]["gamma"] * reste
    return termes, s


def _groupes_tassement(det: Dict, titre: str, p: Dict) -> str:
    if not det["tranches"]:
        return _groupe(titre, [_fx(["k", "q_net / w"], f"non calculé : {_texte_fr(det['convergence'])}")])
    crit = p["crit"]
    lignes = []
    t = det["tranches"][-1]
    if det["convergence"].startswith("Δσ"):
        lignes.append(_fx(["z_infl", f"première profondeur au-delà de B où Δσ ≤ {_v(crit, 2)}·σ'v0",
                           f"Δσ = {C.fr(t['delta_sigma'], 2)} ≤ {_v(crit, 2)} × {C.fr(t['sigma_v0_eff'], 2)} = "
                           f"{C.fr(crit * t['sigma_v0_eff'], 2)} kPa"], f"{C.fr(det['z_influence'], 2)} m"))
    else:
        lignes.append(_fx(["z_infl", f"arrêt : {_texte_fr(det['convergence'])}"], f"{C.fr(det['z_influence'], 2)} m"))
    if det.get("h_sans_module", 0) > 1e-9:
        lignes.append(_fx(["w", "Σ Δσ_i·h_i / M_i"], f"non calculé : {C.fr(det['h_sans_module'], 2)} m sans module"))
        return _groupe(titre, lignes, NOTE_DIFFUSION)
    lignes.append(_fx(["w", "Σ Δσ_i·h_i / M_i", f"somme sur {len(det['tranches'])} tranches"], f"{C.fr(det['w_mm'], 2)} mm"))
    if det["w_m"] > 1e-12:
        lignes.append(_fx(["k", "q_net / w", f"{C.fr(det['q_net_kPa'], 1)} / {C.fr(det['w_m'], 5)}"],
                          f"{_kn(det['k_MNm3'])} kN/m³ = {_k(det['k_MNm3'])} MN/m³"))
    return _groupe(titre, lignes, NOTE_DIFFUSION)


def _fenetre_sondage(f: Dict, r: Dict, p: Dict) -> str:
    det, deth = r["detail"], r["zones"]["centre"]["detail_haut"]
    hyps = [f"Semelle <b>{C.nb(p['B'])} × {C.nb(p['L'])} m</b>" + (" (B et L échangés)" if p["echange"] else "") + f" · assise D = {C.nb(p['D'])} m",
            f"q = {C.nb(p['q'])} kPa · pression {'nette' if p['q_net'] else 'brute'}",
            f"Nappe à {C.nb(p['nappe'])} m" if p["nappe"] is not None else "Sans nappe",
            f"Arrêt : Δσ ≤ {p['critere_pct']} % · σ'<sub>v0</sub>, au-delà de B, plafond 6·B"]
    lignes = []
    for pos, lab in ST.POSITIONS.items():
        z = r["zones"][pos]
        lignes.append([C.esc(lab), "k = q<sub>net</sub> / w", f"{_plage(z['w_bas_mm'], z['w_haut_mm'], 1)} mm",
                       f"{_plage(z['k_bas_MNm3'] * 1000, z['k_haut_MNm3'] * 1000, 0)}", f"{_plage(z['k_bas_MNm3'], z['k_haut_MNm3'])}"])
    table = _vtab([("Position", False), ("Formule", False), ("w", True), ("k [kN/m³]", True), ("k [MN/m³]", True)], lignes,
                  gov=list(ST.POSITIONS).index(f["position"]))
    termes, _ = _sigma_d(r["bas"], p["D"])
    if "sigma_v0_assise" not in det:                           # le moteur s'est arrêté avant (pression nette nulle…)
        g1 = [_fx(["q_net", "max(q − σ_v0(D) ; 0)" if p["q_net"] else "q"], f"{C.fr(det['q_net_kPa'], 2)} kPa — {_texte_fr(det['convergence'])}")]
    else:
        g1 = [_fx(["σ_v0(D)", "Σ γ·h au-dessus de l'assise", " + ".join(termes) or "0"], f"{C.fr(det['sigma_v0_assise'], 2)} kPa")]
    if "sigma_v0_assise" not in det:
        pass
    elif p["q_net"]:
        g1.append(_fx(["q_net", "max(q − σ_v0(D) ; 0)", f"{_v(p['q'])} − {C.fr(det['sigma_v0_assise'], 2)}"], f"{C.fr(det['q_net_kPa'], 2)} kPa"))
    else:
        g1.append(_fx(["q_net", "q (pression brute)"], f"{C.fr(det['q_net_kPa'], 2)} kPa"))
    detail = _groupe("Pression sous la semelle", g1) + _groupes_tassement(det, "Tassement au centre · borne basse des modules", p)
    if any(c.get("M_haut") and c["M_haut"] != c.get("M") for c in r["couches"]):
        detail += _groupes_tassement(deth, "Tassement au centre · borne haute des modules (M haut de l'import)", p)
    detail += _groupe("Encadrement", [_fx(["k_centre", "[min ; max] des deux bornes"], f"{_plage(r['k_bas'], r['k_haut'])} MN/m³"),
                                      _fx(["w_centre", "[min ; max] des deux bornes"], f"{_plage(r['w_bas'], r['w_haut'], 1)} mm")])
    tr, pas = r["tranches_affichees"], r["pas_tranches"]
    n = len(det.get("tranches") or [])
    pied = f'<div class="sol-note">Arrêt de l\'intégration : {_conv(r["convergence"])}.</div>'
    if tr:
        rows = [[C.fr(t["z_sous_assise"], 2), C.fr(t["delta_sigma"], 1), C.fr(t["sigma_v0_eff"], 1),
                 "—" if not t["ratio"] else C.fr(t["ratio"], 2), C.fr(t["M"], 1), C.fr(t["dw_mm"], 3), C.esc(t["couche"] or "—")] for t in tr]
        pied += (f'<div class="ck-sub">Les tranches (centre, borne basse des modules)</div>'
                 + DS.tableau("sol-tr", "", ["z sous assise [m]", "Δσ [kPa]", "σ'<sub>v0</sub> [kPa]", "Δσ/σ'<sub>v0</sub>", "M [MPa]",
                                             "dw [mm]", "couche"], rows, f"Une ligne sur {pas} — {n} tranches au total.", "ddddddg"))
    if det.get("h_sans_module", 0) > 0:
        pied += C.message(f"{C.fr(det['h_sans_module'], 2)} m de profil sans module M n'ont pas été comptés : compléter ces couches.", True)
    sous = f"k au centre = {_plage(r['k_bas'], r['k_haut'])} MN/m³ · Boussinesq/Newmark, module oedométrique"
    return _fenetre(f"s{r['index']}", f"{C.esc(f.get('nom') or 'Fondation')} · sondage {C.esc(r['nom'])}", sous, hyps, table, detail, pied)


def _fenetre_zonage(f: Dict, res: Dict) -> str:
    p = res["parametres"]
    noms = [r["nom"] for r in res["resultats"]]
    lignes = []
    for z in res["zonage"]:
        vals = [_plage(r["zones"][z["position"]]["k_bas_MNm3"], r["zones"][z["position"]]["k_haut_MNm3"]) for r in res["resultats"]]
        lignes.append([C.esc(z["libelle"]), " · ".join(f"{C.esc(nm)} {v}" for nm, v in zip(noms, vals)), f"{_plage(z['k_bas'], z['k_haut'])}"])
    table = _vtab([("Zone", False), ("Sondages [MN/m³]", False), ("k [MN/m³]", True)], lignes)
    detail = _groupe("Zonage", [_fx(["k_zone", "[min des k bas ; max des k haut] sur les sondages calculés"], "par position")])
    detail += _groupe("Pourquoi zoner", [_fx(["centre", "tasse plus que les bords"], "k plus faible au centre")])
    pied = ('<div class="sol-note">Le centre tasse plus que les bords, donc k y est plus faible. Encoder un k unique concentre '
            "artificiellement les réactions en rive : c'est le défaut connu du modèle de Winkler, pas de ce calcul.</div>")
    hyps = [f"Semelle <b>{C.nb(p['B'])} × {C.nb(p['L'])} m</b> · q = {C.nb(p['q'])} kPa", f"{len(noms)} sondage(s) calculé(s)"]
    return _fenetre("zon", f"{C.esc(f.get('nom') or 'Fondation')} · zonage de la dalle", "sans module itératif sol-structure, la façon "
                    "d'approcher la raideur variable", hyps, table, detail, pied)


def _resultat_profil(f: Dict, c: Dict) -> Dict:
    res = c["res"]
    p = res["parametres"]
    pos = f["position"]
    lab = ST.POSITIONS[pos]
    # deux familles de messages : les GRAVES (sondage écarté, k non calculable, profil trop court — k surestimé, un écart du
    # mauvais côté) se lisent sous le verdict, avant les lignes, sans défiler ; les autres viennent après les lignes
    graves, msgs = [], []
    if p["echange"]:
        msgs.append(C.message("B est le petit côté par convention : B et L ont été échangés."))
    rs = res["resultats"]
    if not rs:
        verdict = _verdict("—", "À compléter", "Renseigner au moins une couche avec une épaisseur et un module M, ou importer un sondage.",
                           "c", "")
        corps = verdict + "".join(msgs)
        return {"html": corps, "volets": "", "niveau": c["niveau"]}
    if res["ecartes"]:
        graves.append(C.message(f"{C.esc(', '.join(res['ecartes']))} n'entre pas dans les résultats : aucune couche ne porte de module M. "
                              f"Les valeurs ci-dessous ne couvrent que {len(rs)} sondage(s) sur {len(rs) + len(res['ecartes'])}.", True))
    for r in rs:
        if r["nom"] in res["incalculables"]:
            graves.append(C.message(f"{C.esc(r['nom'])} : k non calculable. {_conv(r['convergence'])}. Une couche sans module ne peut pas être "
                                  "traitée comme incompressible — ce serait l'hypothèse la plus favorable possible, et elle surestime k. "
                                  "Compléter la colonne M de cette couche.", True))
    env = res["enveloppe"]
    if env and not env["unique"]:
        verdict = _verdict(_plage_courte(env["k_min"], env["k_max"]), f"{C.esc(env['libelle'])} · {C.esc(lab.lower())}",
                           f"k = {_plage(env['k_min'], env['k_max'])} MN/m³ = {_plage(env['k_min'] * 1000, env['k_max'] * 1000, 0)} kN/m³ · "
                           "à encoder dans SCIA (C1z)")
        msgs.append(C.message("La dalle doit être vérifiée aux DEUX bornes : un k faible augmente les tassements, un k élevé augmente les "
                              "moments de pointe."))
    elif env:
        verdict = _verdict(_k(env["k_min"]), f"{C.esc(env['libelle'])} · {C.esc(lab.lower())}",
                           f"k = {_k(env['k_min'])} MN/m³ = {_kn(env['k_min'])} kN/m³ · valeur unique, à encoder dans SCIA (C1z)")
        msgs.append(C.message("Valeur unique : les modules ont été saisis directement, il n'y a donc pas d'incertitude de corrélation à "
                              "reporter. Un module issu d'un CPT serait encadré : les corrélations divergent d'un facteur 2 à 4. Faire varier M à la "
                              "main pour mesurer la sensibilité de la dalle."))
    else:
        conv = rs[0]["convergence"]
        verdict = _verdict("—", "Calcul impossible", f"k nul : {_conv(conv)}", "c", "")
    if res["courts"]:
        co = res["courts"]
        graves.append(C.message(f"Profil trop court — k surestimé. Pour {C.esc(', '.join(co['noms']))}, l'intégration s'est arrêtée à "
                              f"{C.fr(co['manque'], 2)} m sous l'assise parce que les couches saisies s'arrêtent là, et non parce que la "
                              "contrainte était devenue négligeable. Le tassement est donc tronqué et le k affiché est trop élevé — c'est "
                              "un écart du mauvais côté pour les moments de la dalle. Prolonger le profil : sous une fondation de "
                              f"{C.fr(co['B'], 1)} m, il faut typiquement descendre à {C.fr(co['profondeur_conseillee'], 0)} m.", True))
        if p["D"] > 0:
            graves.append(C.message(f"Précision du site : ces {C.fr(co['profondeur_conseillee'], 0)} m (2,5·B) se comptent sous l'assise ; le "
                                  f"profil, saisi depuis le terrain naturel, doit donc descendre à au moins D + 2,5·B = "
                                  f"{C.fr(co['profondeur_conseillee_tn'], 1)} m sous le TN."))
        plaf = [r["nom"] for r in rs if "plafond" in r["convergence"]]
        if plaf:
            graves.append(C.message(f"Précision du site : pour {C.esc(', '.join(plaf))}, l'arrêt vient du plafond de calcul 6·B, pas de la fin "
                                  "du profil saisi ; le message de l'écran Streamlit, repris ci-dessus, le dit mal (défaut écrit dans "
                                  "docs/sol/RIGIDITE_SOL_MOTEUR.md)."))
    for r in rs:
        h0 = r["detail"].get("h_sans_module", 0)
        if h0 > 0 and r["nom"] not in res["incalculables"]:
            msgs.append(C.message(f"{C.esc(r['nom'])} : {C.fr(h0, 2)} m de profil sans module M n'ont pas été comptés : compléter ces couches."))
    # les lignes : SCIA puis zonage, sur une même échelle de k
    kmax = max([x["k_haut"] for x in res["scia"]] + [z["k_haut"] for z in res["zonage"]] + [1e-9])
    rows = f'<div class="elgrp">À encoder dans SCIA · {C.esc(lab)}</div>'
    for x, r in zip(res["scia"], rs):
        inc = r["nom"] in res["incalculables"]
        if inc or x["k_haut"] <= 0:
            rows += _ligne_k(x["nom"], "k non calculable" if inc else _conv(r["convergence"]), 0, 0, "—", f"s{r['index']}", niv="r")
            continue
        piece = f"w = {_plage(x['w_bas'], x['w_haut'], 1)} mm"
        rows += _ligne_k(x["nom"], piece, x["k_bas"] / kmax, x["k_haut"] / kmax, f"{_plage_courte(x['k_bas'], x['k_haut'])} MN/m³",
                         f"s{r['index']}", gov=(len(rs) > 1 and x["k_bas"] == min(y["k_bas"] for y in res["scia"])))
    rows += '<div class="elgrp">Zonage de la dalle</div>'
    for z in res["zonage"]:
        rows += _ligne_k(z["libelle"], f"{_plage(z['k_bas'] * 1000, z['k_haut'] * 1000, 0)} kN/m³", z["k_bas"] / kmax, z["k_haut"] / kmax,
                         f"{_plage_courte(z['k_bas'], z['k_haut'])} MN/m³", "zon", gov=z["position"] == pos)
    msgs.append(C.message("Le centre tasse plus que les bords, donc k y est plus faible. Encoder un k unique concentre artificiellement les "
                          "réactions en rive : c'est le défaut connu du modèle de Winkler, pas de ce calcul."))
    volets = "".join(_fenetre_sondage(f, r, p) for r in rs) + _fenetre_zonage(f, res)
    return {"html": verdict + "".join(graves) + f'<div class="els">{rows}</div>' + "".join(msgs), "volets": volets, "niveau": c["niveau"]}


def _vues_profil(f: Dict, c: Dict) -> List[Dict]:
    res = c["res"]
    p = res["parametres"]
    rs = res["resultats"]
    vues = []
    # l'essai en attente d'import (le premier) : ses courbes et son découpage, EN TÊTE (c'est ce qu'on est en train de faire)
    for i, s in enumerate(f["sondages"]):
        imp = s["import"]
        if imp is None:
            continue
        try:
            ap = R.apercu_import(imp, f["nu"])
        except ValueError:
            ap = None
        d = R.donnees_cpt(imp["points"], imp["nappe"])
        vues.append({"cle": "import", "titre": f"Import · {imp['nom']}", "court": "Import",
                     "svg": DS.cpt(f"sol_imp_{i}", d, f"{imp['nom']} — essai lu, en attente (découpage de l'aperçu en tirets)",
                                   ap["couches"] if ap else None),
                     "legende": f"{len(imp['points'])} points · {imp['source']}"})
        if ap and ap["lignes"]:
            rows = [[C.fr(x["de"], 2), C.fr(x["a"], 2), C.fr(x["h"], 2), "—" if x["Ic"] is None else C.fr(x["Ic"], 2), C.esc(x["type"] or "—"),
                     C.fr(x["qc"], 2), "—" if x["M_bas"] is None else C.fr(x["M_bas"], 1), "—" if x["M_haut"] is None else C.fr(x["M_haut"], 1)]
                    for x in ap["lignes"]]
            vues.append({"cle": "import_tab", "titre": f"Découpage · {imp['nom']}", "court": "Découpage",
                         "svg": DS.vue_tableaux(DS.tableau("sol-dec-t", f"Découpage automatique · {C.esc(imp['nom'])}",
                                                           ["de [m]", "à [m]", "h [m]", "I<sub>c</sub>", "Type reconnu", "q<sub>c</sub> moy [MPa]",
                                                            "M bas [MPa]", "M haut [MPa]"], rows,
                                                           "Le type de sol est déduit de qc et fs par l'indice de comportement Ic (Robertson). "
                                                           "Le module M est encadré car les corrélations CPT divergent d'un facteur 2 à 4.",
                                                           "ddddgddd")),
                         "legende": f"{len(ap['lignes'])} couches découpées"})
        break
    s_t = f["sondages"][f["trace"]]
    r_t = next((r for r in rs if r["index"] == f["trace"]), rs[0] if rs else None)
    if r_t is not None:
        d = R.donnees_diffusion(r_t, p["D"], p["nappe"])
        vues.append({"cle": "diffusion", "titre": f"Diffusion · {r_t['nom']}", "court": "Diffusion",
                     "svg": DS.diffusion("sol_dif", d, f"{r_t['nom']} — {ST.POSITIONS['centre'].lower()}", LEGENDE_DIFFUSION),
                     "legende": "Δσ décroît avec la profondeur ; le calcul s'arrête où Δσ croise le critère"})
    couches = [{"h": c_["h"], "nom": c_["type"], "M": c_["M"], "M_haut": c_.get("M_haut")} for c_ in s_t["couches"] if c_["h"] > 0]
    r_s = r_t if (r_t is not None and r_t["index"] == f["trace"]) else None
    vues.append({"cle": "coupe", "titre": f"Coupe · {s_t['nom']}", "court": "Coupe",
                 "svg": DS.coupe("sol_coupe", {"B": p["B"], "L": p["L"], "D": p["D"], "q": p["q"],
                                               "q_net": r_s["q_net"] if r_s else None, "nappe": p["nappe"], "couches": couches,
                                               "z_infl": r_s["z_infl"] if r_s else None, "tronque": r_s["tronque"] if r_s else False,
                                               "nom": s_t["nom"]}),
                 "legende": "à l'échelle · profondeurs sous le terrain naturel"})
    if r_t is not None:
        kb = R.k_fonction_B(r_t["bas"], r_t["haut"], p["B"], p["L"], p["q"], p["D"], p["nappe"], p["crit"])
        vues.append({"cle": "kb", "titre": "k selon la largeur de fondation", "court": "k(B)",
                     "svg": DS.k_largeur("sol_kb", kb, note=LEGENDE_KB), "legende": "à sol identique, k varie avec la taille de la fondation"})
    if r_t is not None:
        pts = f["sondages"][r_t["index"]]["points"]
        if pts:
            vues.append({"cle": "cpt", "titre": f"Courbes CPT · {r_t['nom']}", "court": "CPT",
                         "svg": DS.cpt("sol_cpt", R.donnees_cpt(pts, p["nappe"]), f"{r_t['nom']} — courbes CPT et classification"),
                         "legende": "qc, fs et Ic (Robertson) ; découpage par défaut (finesse 0,25, épaisseur 0,50 m)"})
    if len(f["sondages"]) > 1:
        par_index = {r["index"]: r for r in rs}
        cols = [{"nom": s["nom"], "couches": [{"h": x["h"], "nom": x["type"], "M": x["M"]} for x in s["couches"] if x["h"] > 0],
                 "k": (par_index[i]["k_bas"], par_index[i]["k_haut"]) if i in par_index and par_index[i]["k_haut"] > 0 else None,
                 "z_infl": par_index[i]["z_infl"] if i in par_index else None} for i, s in enumerate(f["sondages"])]
        vues.append({"cle": "sondages", "titre": "Les sondages", "court": "Sondages", "svg": DS.sondages("sol_snd", cols, p["D"], p["nappe"]),
                     "legende": "même échelle des profondeurs"})
    if rs:
        lignes = [[C.esc(x["nom"]), _k(x["k_bas"]), _k(x["k_haut"]), _kn(x["k_bas"]), _kn(x["k_haut"]),
                   f"{C.fr(x['w_bas'], 0)} – {C.fr(x['w_haut'], 0)}"] for x in res["scia"]]
        zon = [[C.esc(z["libelle"]), _k(z["k_bas"]), _k(z["k_haut"])] for z in res["zonage"]]
        env = res["enveloppe"]
        pied_env = (f"{C.esc(env['libelle'])} : k = {_plage(env['k_min'], env['k_max'])} MN/m³." if env else "")
        vues.append({"cle": "scia", "titre": f"À encoder dans SCIA · {ST.POSITIONS[f['position']]}", "court": "Tableau",
                     "svg": DS.vue_tableaux(
                         DS.tableau("sol-scia", f"À encoder dans SCIA (paramètres de sol C1z) · {C.esc(ST.POSITIONS[f['position']].lower())}",
                                    ["Sondage", "k bas [MN/m³]", "k haut [MN/m³]", "k bas [kN/m³]", "k haut [kN/m³]", "w [mm]"], lignes, pied_env),
                         DS.tableau("sol-zon", "Zonage de la dalle", ["Zone", "k bas [MN/m³]", "k haut [MN/m³]"], zon,
                                    "Le centre tasse plus que les bords, donc k y est plus faible.")),
                     "legende": "les valeurs de l'écran Streamlit, arrondies comme lui"})
    return vues


# ------------------------------------------------------------------ modes 3 à 6 : le résultat
def _resultat_rapide(f: Dict, c: Dict) -> Tuple[str, str, List[Dict], str]:
    x = c["x"]
    q, w, k = x["q"], x["w"], x["k_kNm3"]
    if x["etat"] == "ok":
        verdict = _verdict(_k(x["k_MNm3"]), "Coefficient de réaction", f"k = q / w = {_v(q)} / {_v(w / 1000.0, 5)} = {C.fr(k, 0)} kN/m³")
    else:
        verdict = _verdict("—", "Calcul impossible", "k nul : la pression q est nulle.", "c", "")
    rows = (_ligne_k("k", f"{C.fr(k, 0)} kN/m³", 0, 1 if k > 0 else 0, f"{_k(x['k_MNm3'])} MN/m³", "rap", True)
            + _ligne_k("Tassement", f"q = {C.nb(q)} kPa", 0, 0, f"{C.nb(w)} mm", "rap"))
    html = verdict + f'<div class="els">{rows}</div>' + C.message("Valable si w provient d'un vrai calcul de tassement, pour cette "
                                                                   "fondation et cette charge.")
    detail = _groupe("Loi de Winkler", [_fx(["k", "q / w", f"{_v(q)} / {_v(w / 1000.0, 5)}"], f"{C.fr(k, 0)} kN/m³ = {_k(x['k_MNm3'])} MN/m³")])
    table = _vtab([("Grandeur", False), ("Formule", False), ("Valeur", True)],
                  [["k", "q / w", f"{C.fr(k, 0)} kN/m³"], ["k", "k / 1 000", f"{_k(x['k_MNm3'])} MN/m³"]], gov=1)
    volet = _fenetre("rap", f"{C.esc(f.get('nom') or 'Fondation')} · k = q / w", "un tassement déjà calculé", [f"q = {C.nb(q)} kPa",
                     f"w = {C.nb(w)} mm"], table, detail)
    vues = [{"cle": "winkler", "titre": "Loi de Winkler", "court": "q = k·w", "svg": DS.droite_qw("sol_qw", x), "legende": "q = k·w"}]
    return html, volet, vues, f"q = {C.nb(q)} kPa · w = {C.nb(w)} mm"


def _resultat_theories(f: Dict, c: Dict) -> Tuple[str, str, List[Dict], str]:
    t = c["x"]
    E, B, L, nu, M = t["E"], t["B"], t["L"], t["nu"], t["M"]
    tas = t["tassement"]
    if t["etat"] == "ok":
        verdict = _verdict(_k(t["k_retenu"]), "Valeur retenue · tassement calculé",
                           f"Tassement {C.fr(tas['w_mm'], 1)} mm, profondeur d'influence {C.fr(tas['z_influence'], 2)} m.")
    else:
        verdict = _verdict("—", "Calcul impossible", _conv(tas["convergence"]), "c", "")
    kmax = max([m["k"] for m in t["methodes"]] + [1e-9])
    pieces = {"tassement": "retenue", "ancienne": "biaisée"}
    rows = "".join(_ligne_k(m["nom"].replace(" (retenue)", ""), pieces.get(m["cle"], ""), 0, m["k"] / kmax, f"{_k(m['k'])} MN/m³", "th",
                            gov=m["cle"] == "tassement", niv="att" if m["cle"] == "ancienne" else "") for m in t["methodes"])
    html = (verdict + f'<div class="els">{rows}</div>'
            + C.message(f"Sol homogène E = {C.fr(E, 0)} MPa (soit M = {C.fr(M, 0)} MPa pour ν = {C.fr(nu, 2)}), semelle {C.fr(B, 1)}×{C.fr(L, 1)} m, "
                        f"q = {C.fr(t['q'], 0)} kPa. La dernière ligne est la formule de la version 4 : elle donne exactement E/(2B), donc un "
                        "résultat piloté par la règle des 2B et non par le sol — d'où le facteur 2 à 2,5 d'écart."))
    a = 1 - nu * nu
    g = [_fx(["M", "E·(1 − ν) / ((1 + ν)·(1 − 2ν))", f"{_v(E)} × (1 − {_v(nu)}) / ((1 + {_v(nu)}) × (1 − 2 × {_v(nu)}))"], f"{C.fr(M, 2)} MPa")]
    g2 = [_fx(["w", "Σ Δσ_i·h_i / M", f"sol homogène de 60 m, q = 150 kPa brute, {len(tas['tranches'])} tranches"], f"{C.fr(tas['w_mm'], 2)} mm")]
    if tas["w_m"] > 1e-12:
        g2.append(_fx(["k", "q / w", f"150 / {C.fr(tas['w_m'], 5)}"], f"{_k(tas['k_MNm3'])} MN/m³"))
    g3 = [_fx(["k_rigide", "E / (B·(1 − ν²)·0,88)", f"{_v(E)} / ({_v(B)} × {_v(a, 4)} × 0,88)"], f"{_k(t['methodes'][1]['k'])} MN/m³"),
          _fx(["k_souple", "E / (B·(1 − ν²)·1,12)", f"{_v(E)} / ({_v(B)} × {_v(a, 4)} × 1,12)"], f"{_k(t['methodes'][2]['k'])} MN/m³"),
          _fx(["k_Vesić", "E / (B·(1 − ν²))", f"{_v(E)} / ({_v(B)} × {_v(a, 4)})"], f"{_k(t['methodes'][3]['k'])} MN/m³"),
          _fx(["k_v4", "E / (2B)", f"{_v(E)} / (2 × {_v(B)})"], f"{_k(t['methodes'][4]['k'])} MN/m³")]
    detail = (_groupe("Module oedométrique", g) + _groupe("Tassement calculé (retenu)", g2)
              + _groupe("Formules élastiques (massif homogène semi-infini)", g3))
    table = _vtab([("Méthode", False), ("Domaine", False), ("k [MN/m³]", True)],
                  [[C.esc(m["nom"]), C.esc(m["domaine"]), _k(m["k"])] for m in t["methodes"]], gov=0)
    volet = _fenetre("th", f"{C.esc(f.get('nom') or 'Fondation')} · théories comparées", "un ordre de grandeur, jamais la valeur retenue",
                     [f"E = {C.nb(E)} MPa", f"ν = {C.nb(nu)}", f"B × L = {C.nb(B)} × {C.nb(L)} m", "q = 150 kPa (brute)",
                      "critère 20 %, assise au TN"], table, detail)
    vues = [{"cle": "theories", "titre": "Théories comparées", "court": "Comparaison", "svg": DS.theories("sol_th", t),
             "legende": "k en MN/m³"}]
    d = R.donnees_diffusion({"detail": dict(tas, critere_pct=20), "couches": [{"h": 60.0, "gamma": 19.0, "M": M, "nom": "sol homogène"}]},
                            0.0, None)
    vues.append({"cle": "diffusion", "titre": "Diffusion · sol homogène", "court": "Diffusion",
                 "svg": DS.diffusion("sol_th_dif", d, "sol homogène — centre", LEGENDE_DIFFUSION),
                 "legende": "Δσ décroît avec la profondeur ; le calcul s'arrête où Δσ croise le critère"})
    return html, volet, vues, f"E = {C.nb(E)} MPa · {C.nb(B)} × {C.nb(L)} m · ν = {C.nb(nu)}"


def _resultat_abaque(f: Dict, c: Dict) -> Tuple[str, str, List[Dict], str]:
    a = c["x"]
    B, w = a["B"], a["w_ref"]
    vals = [v for x in a["lignes"] for v in (x["k_bas"], x["k_haut"]) if v > 0]
    lo, hi = math.log10(min(vals)), math.log10(max(vals))
    frac = lambda v: (math.log10(max(v, 10 ** lo)) - lo) / ((hi - lo) or 1)  # noqa: E731 — l'échelle logarithmique des barres
    verdict = _verdict(C.nb(B), "Abaque des sols · correction de Terzaghi",
                       f"{len(a['lignes'])} sols de la base : k de plaque 0,30 m corrigé pour la largeur B ; q pour w<sub>réf</sub> = {C.nb(w)} mm",
                       "", "m", "B")
    rows, cat = "", None
    for x in a["lignes"]:
        if x["categorie"] != cat:
            cat = x["categorie"]
            rows += f'<div class="elgrp">{C.esc(cat)}</div>'
        rows += _ligne_k(x["type"], f"plaque {DS.nombre(x['kp_min'], 0)} – {DS.nombre(x['kp_max'], 0)}", frac(x["k_bas"]), frac(x["k_haut"]),
                         f"{_plage_courte(x['k_bas'], x['k_haut'], 1)} MN/m³", "ab")
    html = (verdict + C.message("Les deux colonnes de k ne sont pas interchangeables. Celle de la plaque est la valeur mesurée sur une "
                                "plaque de 0,30 m (tables de Terzaghi 1955) ; l'autre lui applique la correction de taille pour la largeur "
                                "choisie. Utiliser la valeur de plaque pour une fondation réelle surestime k d'un facteur 3 à 30. La version "
                                "4 de ce module ne faisait pas cette correction.")
            + f'<div class="els">{rows}</div>'
            + C.message("Un abaque reste un ordre de grandeur : la méthode « Sondage CPT » calcule la valeur pour le sol et la fondation."))
    fs, fa = ((B + 0.30) / (2.0 * B)) ** 2, 0.30 / B
    detail = (_groupe("Correction de taille (Terzaghi 1955)",
                      [_fx(["k_sable", "k_plaque·((B + 0,30) / (2B))²", f"k_plaque × (({_v(B)} + 0,30) / (2 × {_v(B)}))²"], f"k_plaque × {C.fr(fs, 4)}"),
                       _fx(["k_argile", "k_plaque·0,30 / B", f"k_plaque × 0,30 / {_v(B)}"], f"k_plaque × {C.fr(fa, 4)}")])
              + _groupe("Pression pour le tassement de référence", [_fx(["q", "k·w_réf", f"k × 1 000 × {_v(w)} / 1 000"], "kPa")]))
    lignes = [[C.esc(x["type"]), C.esc(x["nature"]), f"{DS.nombre(x['kp_min'], 0)} – {DS.nombre(x['kp_max'], 0)}", f"{C.fr(x['k_bas'], 1)} – {C.fr(x['k_haut'], 1)}",
               f"{C.fr(x['q_bas'], 0)} – {C.fr(x['q_haut'], 0)}"] for x in a["lignes"]]
    table = _vtab([("Type de sol", False), ("Nature", False), ("k plaque", True), (f"k pour B = {C.fr(B, 1)} m", True), ("q pour w réf [kPa]", True)], lignes)
    volet = _fenetre("ab", f"{C.esc(f.get('nom') or 'Fondation')} · abaque des sols", "correction de taille sans correction de forme (L "
                     "n'est pas transmis, comme dans l'écran)", [f"B = {C.nb(B)} m", f"w<sub>réf</sub> = {C.nb(w)} mm"], table, detail)
    rows_t = [[C.esc(x["categorie"]), C.esc(x["type"]), C.fr(x["gamma"], 1), f"{C.fr(x['M_min'], 0)} – {C.fr(x['M_max'], 0)}" if x["M_min"] else "—",
               f"{DS.nombre(x['kp_min'], 0)} – {DS.nombre(x['kp_max'], 0)}", f"{C.fr(x['k_bas'], 1)} – {C.fr(x['k_haut'], 1)}",
               f"{C.fr(x['q_bas'], 0)} – {C.fr(x['q_haut'], 0)}"] for x in a["lignes"]]
    vues = [{"cle": "abaque", "titre": "Abaque des sols", "court": "Graphique", "svg": DS.abaque("sol_ab", a), "legende": "échelle logarithmique"},
            {"cle": "abaque_tab", "titre": "Abaque des sols · tableau", "court": "Tableau",
             "svg": DS.vue_tableaux(DS.tableau("sol-ab", f"Abaque des sols · B = {C.esc(C.nb(B))} m · w<sub>réf</sub> = {C.esc(C.nb(w))} mm",
                                               ["Catégorie", "Type de sol", "γ [kN/m³]", "M [MPa]", "k plaque 0,30 m", f"k pour B = {C.fr(B, 1)} m",
                                                "q pour w réf [kPa]"], rows_t,
                                               "k en MN/m³. Les tables de plaque surestiment k pour une fondation réelle : utiliser la colonne corrigée.",
                                               "ggddddd")),
             "legende": "les colonnes de l'écran Streamlit"}]
    return html, volet, vues, f"B = {C.nb(B)} m · w<sub>réf</sub> = {C.nb(w)} mm"


def _resultat_plat(f: Dict, c: Dict) -> Tuple[str, str, List[Dict], str]:
    p = c["x"]
    keq = p["keq"]
    if p["etat"] == "ok":
        verdict = _verdict(C.fr(keq / 1000.0, 0), "Raideur du contact", f"k = {C.fr(keq / 1000.0, 0)} MN/m³ = {C.fr(keq, 0)} kN/m³"
                           + (" · plat sur lit de mortier" if p["mortier"] else ""))
    else:
        verdict = _verdict("—", "Calcul impossible", "k nul", "c", "")
    kmax = max(p["kc"], p["kg"], keq, 1e-9)
    rows = _ligne_k("Béton", f"h<sub>c</sub> = {C.fr(p['hc_m'] * 1000, 1)} mm", 0, p["kc"] / kmax, f"{C.fr(p['kc'] / 1000.0, 0)} MN/m³", "pl")
    if p["mortier"]:
        rows += _ligne_k("Mortier", f"t<sub>g</sub> = {C.nb(p['tg'])} mm", 0, p["kg"] / kmax, f"{C.fr(p['kg'] / 1000.0, 0)} MN/m³", "pl")
    rows += _ligne_k("Contact", "en série" if p["mortier"] else "béton seul", 0, keq / kmax, f"{C.fr(keq / 1000.0, 0)} MN/m³", "pl", True)
    html = verdict + f'<div class="els">{rows}</div>'
    g = [_fx(["h_c", "α·min(B ; L)", f"{_v(p['alpha'])} × min({_v(p['B'])} ; {_v(p['L'])})"], f"{C.fr(p['hc_m'] * 1000, 1)} mm"),
         _fx(["k_c", "E_c / h_c", f"{_v(p['E'])} × 10⁶ / {_v(p['hc_m'], 5)}"], f"{C.fr(p['kc'] / 1000.0, 0)} MN/m³")]
    if p["mortier"]:
        g.append(_fx(["k_g", "E_g / t_g", f"{_v(p['Eg'])} × 10⁶ / {_v(p['tg_m'], 5)}"], f"{C.fr(p['kg'] / 1000.0, 0)} MN/m³"))
        g.append(_fx(["k_eq", "1 / (1/k_c + 1/k_g)", f"1 / (1/{C.fr(p['kc'] / 1000.0, 0)} + 1/{C.fr(p['kg'] / 1000.0, 0)})"],
                     f"{C.fr(keq / 1000.0, 0)} MN/m³"))
    detail = _groupe("Compression 1D du contact", g)
    lignes = [["k<sub>c</sub>", "E<sub>c</sub> / h<sub>c</sub>", f"{C.fr(p['kc'] / 1000.0, 0)}"]]
    if p["mortier"]:
        lignes += [["k<sub>g</sub>", "E<sub>g</sub> / t<sub>g</sub>", f"{C.fr(p['kg'] / 1000.0, 0)}"],
                   ["k<sub>eq</sub>", "1/k<sub>eq</sub> = 1/k<sub>c</sub> + 1/k<sub>g</sub>", f"{C.fr(keq / 1000.0, 0)}"]]
    table = _vtab([("Raideur", False), ("Formule", False), ("k [MN/m³]", True)], lignes, gov=len(lignes) - 1)
    hyps = [f"Plat {C.nb(p['B'])} × {C.nb(p['L'])} mm", f"α = {C.nb(p['alpha'])}", f"E<sub>c</sub> = {C.nb(p['E'])} GPa"]
    if p["mortier"]:
        hyps.append(f"mortier t<sub>g</sub> = {C.nb(p['tg'])} mm · E<sub>g</sub> = {C.nb(p['Eg'])} GPa")
    volet = _fenetre("pl", f"{C.esc(f.get('nom') or 'Fondation')} · plat en béton", "contact assimilé à une compression 1D : k = E / h_c",
                     hyps, table, detail)
    vues = [{"cle": "plat", "titre": "Plat en béton", "court": "Coupe", "svg": DS.plat("sol_plat", p), "legende": "à l'échelle · mm"}]
    return html, volet, vues, f"plat {C.nb(p['B'])} × {C.nb(p['L'])} mm · α = {C.nb(p['alpha'])} · E<sub>c</sub> = {C.nb(p['E'])} GPa"


# ------------------------------------------------------------------ resultat
def resultat(etat: Dict) -> Dict:
    f = _f(etat)
    calculs = [calculer(g) for g in etat["fondations"]]
    c = calculs[etat["actif"]]
    if f["mode"] in ("cpt", "profil"):
        r = _resultat_profil(f, c)
        html, volets, vues = r["html"], r["volets"], _vues_profil(f, c)
        p = c["res"]["parametres"]
        phrase = (f"semelle {C.nb(p['B'])} × {C.nb(p['L'])} m à D = {C.nb(p['D'])} m · q = {C.nb(p['q'])} kPa"
                  + (f" · nappe {C.nb(p['nappe'])} m" if p["nappe"] is not None else "")
                  + f" · {C.esc(f['sondages'][f['trace']]['nom'])}")
    else:
        fn = {"rapide": _resultat_rapide, "theories": _resultat_theories, "abaque": _resultat_abaque, "plat": _resultat_plat}[f["mode"]]
        html, volets, vues, phrase = fn(f, c)
    # le pied, comme celui de la poutre : la méthode et le moteur, la version de l'écran Streamlit (l'avertissement de
    # pré-dimensionnement est le dernier message du résultat, une seule fois)
    meth = "Boussinesq · Newmark" if f["mode"] in ("cpt", "profil", "theories") else C.esc(COURTS[f["mode"]])
    pied = f'<div class="pfoot"><div class="mass"><span>{meth} · moteur rigidite_sol</span><b>{C.esc(R.VERSION)}</b></div></div>'
    return {"resultat": f'<div class="pbody">{html}{C.message(PIED)}</div>{pied}', "volets": volets, "vues": vues, "phrase": phrase,
            "statut_html": C.statut(c["niveau"]), "resume": resume_note(etat, calculs), "donnees": formulaire(f, c),
            "elements": elements(etat, calculs)}


# ------------------------------------------------------------------ le parcours propre (api_web/parcours.py)
def _essai_demo() -> str:
    """Un essai GEF de démonstration (limon puis sable dense, nappe à 2,4 m) : le parcours l'importe."""
    lignes = []
    for k in range(1, 121):
        z = k * 0.05
        qc = 1.6 + 0.2 * (k % 3) if z < 2.5 else (9.0 + 0.6 * (k % 4) if z < 4.5 else 3.0 + 0.2 * (k % 2))
        fs = qc * (32.0 if z < 2.5 else (6.0 if z < 4.5 else 40.0))
        lignes.append(f"{z:.3f};{qc:.3f};{fs / 1000:.5f}!")
    return ("#GEFID= 1,1,0\n#COLUMN= 3\n#COLUMNINFO= 1, m, Sondeerlengte, 1\n#COLUMNINFO= 2, MPa, Conusweerstand, 2\n"
            "#COLUMNINFO= 3, MPa, Wrijvingsweerstand, 3\n#COLUMNSEPARATOR= ;\n#RECORDSEPARATOR= !\n#TESTID= CPT-DEMO\n"
            "#MEASUREMENTVAR= 14, 2.40, m, grondwaterstand\n#EOH=\n" + "\n".join(lignes) + "\n")


ESSAI_DEMO = _essai_demo()


def _etat_avec_import() -> Dict:
    e = nouvel_etat()
    modifier(e, ["sondages", "0", "fichier"], {"nom": "CPT-DEMO.gef", "texte": ESSAI_DEMO})
    return e


def _etat_deux_sondages() -> Dict:
    e = nouvel_etat()
    modifier(e, ["sondages", "+"], 1)
    modifier(e, ["sondages", "1", "couches", "2", "M"], 0)
    return e


PARCOURS = [
    {"action": "modifier", "chemin": ["sondages", "0", "fichier"], "valeur": {"nom": "CPT-DEMO.gef", "texte": ESSAI_DEMO}},
    {"action": "modifier", "chemin": ["sondages", "0", "fichier"], "valeur": {"nom": "faux.gef", "texte": "pas un essai"}},
    {"action": "modifier", "chemin": ["sondages", "0", "colle"], "valeur": "0,5;2,1;30\n1,0;2,4;35\n1,5;3,0;40\n2,0;5,2;41\n2,5;6,0;44\n3,0;8,0;50\n"},
    {"action": "modifier", "chemin": ["sondages", "0", "colle"], "valeur": "rien de lisible"},
    {"action": "modifier", "chemin": ["sondages", "0", "import", "valider"], "valeur": 1, "etat": _etat_avec_import()},
    {"action": "modifier", "chemin": ["sondages", "0", "import", "annuler"], "valeur": 1, "etat": _etat_avec_import()},
    {"action": "modifier", "chemin": ["sondages", "0", "import", "seuil"], "valeur": "0.1", "etat": _etat_avec_import()},
    {"action": "modifier", "chemin": ["sondages", "0", "import", "h_min"], "valeur": "2", "etat": _etat_avec_import()},
    {"action": "modifier", "chemin": ["trace"], "valeur": "1", "etat": _etat_deux_sondages()},
    {"action": "modifier", "chemin": ["sondages", "1", "-"], "valeur": 1, "etat": _etat_deux_sondages()},
    {"action": "modifier", "chemin": ["streamlit"], "valeur": {"nom": "raideur_sol.json", "texte": json.dumps(
        {"version": "v5.0", "soundings": [{"id": 1, "nom": "CPT01"}], "orders": {"1": [1]},
         "values": {"rs_mode": "2. Profil de sol saisi à la main", "rs_B": 3.0, "snd1_nom": "CPT01", "snd1_layer_1_h": 12.0,
                    "snd1_layer_1_type": "Sable dense", "snd1_layer_1_M": 60.0, "snd1_layer_1_gamma": 20.0}})}},
    {"action": "modifier", "chemin": ["B"], "valeur": "0,1"},
]
