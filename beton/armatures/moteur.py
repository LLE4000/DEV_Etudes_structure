# -*- coding: utf-8 -*-
"""Tableau des armatures — le MOTEUR de consultation, sans Streamlit : bibliothèque standard seule (math), Python 3.12
et 3.13 (donc Pyodide). Ses nombres sont ceux de ``beton/armatures/tableau.py``, c'est-à-dire ceux de l'image que
l'écran Streamlit ``modules/tableau_armatures.py`` affiche (preuve : tests/test_armatures_moteur.py, qui relit l'image
dans ses pixels). Méthode et défauts relevés : docs/beton/ARMATURES_MOTEUR.md.

Ce que le moteur ajoute à l'image — c'est le même calcul, mis au service d'une question :
- un BESOIN (une section requise, en mm² pour des barres, en mm²/m pour une nappe) marque les cases du tableau qui
  suffisent et propose la disposition LA PLUS ÉCONOMIQUE : la plus petite section qui suffit, parmi les cases du
  tableau dans les critères (Ø mini/maxi, nombre de barres ou espacement mini/maxi) ; à section égale, le moins de
  barres, puis le plus petit Ø ;
- une DISPOSITION retenue (Ø et nombre de barres, ou Ø et espacement) donne sa section, sa masse et son entraxe, avec
  le détail des formules. Un nombre de barres au-delà de 10, ou un espacement libre, sortent du tableau : même
  formule, signalés comme tels.

API
---
    nouveau_besoin(nom="Repère 1") -> dict        un besoin aux valeurs par défaut
    normaliser(besoin) -> dict                    copie complétée et contrôlée (ValueError claire sinon)
    disposition(besoin) -> dict                   la disposition retenue : section, masse, entraxe, formules
    candidats(besoin, avec_retenue=False) -> list les cases du tableau dans les critères, triées (sélecteur)
    optimale(besoin) -> dict | None               la plus économique qui suffit (None sans section requise)
    calculer(besoin) -> dict                      tout : disposition, optimale, tableau marqué, avertissements
    ESPACEMENTS                                   les espacements du tableau (clé, e en cm, barres/m, libellé)

ENTRÉE — un besoin (dict JSON ; une clé absente prend sa valeur par défaut, une clé inconnue ou une valeur hors du
domaine lève ValueError) :
    {"nom": "Repère 1",                 le nom du repère (120 caractères au plus)
     "mode": "barres" | "nappe",        des barres (section en mm²) ou une nappe (section par mètre, mm²/m)
     "As_req": 400.0,                   la section requise (0 à 1 000 000 ; 0 : pas de recherche)
     "d": 12,                           le Ø retenu, parmi les diamètres du tableau (mm)
     "n": 4,                            mode barres : le nombre de barres retenu (1 à 200)
     "espacement": "e15",               mode nappe : une clé d'ESPACEMENTS (« n4 » = 4 barres/m = 25 cm, « e15 » =
                                        15 cm…) ou « libre »
     "e_libre": 15.0,                   l'espacement libre (cm, 0 < e ≤ 100), lu si espacement = « libre »
     "d_min": 6, "d_max": 40,           les critères de recherche : Ø du tableau
     "n_min": 1, "n_max": 10,           mode barres : nombre de barres (1 à 10, l'étendue du tableau)
     "e_min": 10.0, "e_max": 40.0}      mode nappe : espacement (cm, > 0)

SORTIE de calculer (JSON-sérialisable, nombres non arrondis ; « …_arrondi » : la valeur que le tableau écrit)
    {"reference", "besoin", "disposition": <disposition>, "optimale": <disposition> | None,
     "suffisantes": nombre de cases du tableau qui suffisent dans les critères,
     "tableau": {"lignes": [{"d", "aire", "masse", "masse_txt", "cases": [<case>, ...]}], "colonnes": [...]},
     "avertissements": [{"code", "message"}, ...]}
    <disposition> = {"mode", "d", "n" (barres) | "espacement" (nappe), "e" (cm), "jour" (e − Ø, mm), "barres_par_metre", "nom",
                     "aire", "As", "As_arrondi", "unite", "masse_barre", "masse", "unite_masse",
                     "dans_tableau", "colonne" ("n4", "e15"… ou None), "dans_criteres",
                     "taux" (As_req / As), "suffit", "formules": [<formule>, ...]}
    <formule> = {"cle", "symbole", "expression", "termes": [{"symbole", "valeur", "unite"}], "resultat", "unite"}
    <case> = {"colonne", "valeur", "arrondi", "utile" (la colonne sert au mode), "dans_criteres", "suffit",
              "optimale", "retenue"}
"""
import copy
import math
from typing import Dict, List, Optional

from . import tableau as T

REFERENCE = T.REFERENCE
MODES = {"barres": "Barres", "nappe": "Nappe par mètre"}
UNITES = {"barres": "mm²", "nappe": "mm²/m"}
LIBRE = "libre"
N_MAX_SAISIE = 200          # barres d'une disposition saisie (le tableau s'arrête à 10)
E_LIBRE_MAX = 100.0         # cm : une barre par mètre au moins (la colonne « 1 » du tableau, e = 100 cm)
NOM_MAX = 120               # caractères du nom d'un repère (au-delà, il défigure le résultat et la liste)
AS_REQ_MAX = 1e6            # mm² ou mm²/m : au-delà, la saisie est absurde (le tableau s'arrête à 10Ø40)


def _nombre_txt(x: float) -> str:
    """Un nombre d'un message, à la française : « 1 250,5 » (espace fine des milliers, virgule, deux décimales au plus)."""
    s = f"{x:,.2f}".rstrip("0").rstrip(".")
    return s.replace(",", "\u202f").replace(".", ",")


def section_txt(As: float, req: float) -> str:
    """Une section écrite face à une section requise : au mm² comme le tableau, SAUF quand l'arrondi tromperait la
    comparaison — 1Ø40 = 1 256,64 mm² face à 1 257 requis s'écrit « 1 256,6 » (et non « 1 257 < 1 257 »)."""
    for nd in (0, 1, 2, 3):
        if ((T.arrondi(As) if nd == 0 else round(As, nd)) >= req) == (As >= req):
            break
    if nd == 0:
        return _nombre_txt(T.arrondi(As))
    return f"{As:,.{nd}f}".replace(",", " ").replace(".", ",")


def _libelle_e(e: float) -> str:
    """Un espacement écrit court : « 15 », « 16,7 », « 12,5 » (cm)."""
    s = f"{e:.1f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def libelle_espacement(e: float) -> str:
    """Un espacement à écrire (cm) : celui d'une colonne du tableau comme son en-tête (« 16,7 » pour 100/6), un autre
    au centième près (« 16,75 », « 8,33 ») — une décimale le déformerait (16,75 s'écrirait 16,8)."""
    if any(e == x["e"] for x in ESPACEMENTS):
        return _libelle_e(e)
    s = f"{e:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


# les espacements du tableau, triés : les colonnes « n barres par mètre » (e = 100 / n) et les « espacements fixes »
ESPACEMENTS: List[Dict] = sorted(
    [{"cle": f"n{n}", "e": T.espacement(n), "barres_par_metre": n, "libelle": _libelle_e(T.espacement(n))} for n in T.NOMBRES]
    + [{"cle": f"e{e}", "e": float(e), "barres_par_metre": None, "libelle": _libelle_e(e)} for e in T.ESPACEMENTS_FIXES],
    key=lambda x: x["e"])
_ESP = {x["cle"]: x for x in ESPACEMENTS}
COLONNES = [f"n{n}" for n in T.NOMBRES] + [f"e{e}" for e in T.ESPACEMENTS_FIXES]

DEFAUT = {"nom": "Repère 1", "mode": "barres", "As_req": 400.0, "d": 12, "n": 4, "espacement": "e15", "e_libre": 15.0,
          "d_min": T.DIAMETRES[0], "d_max": T.DIAMETRES[-1], "n_min": 1, "n_max": T.NOMBRES[-1], "e_min": 10.0, "e_max": 40.0}


def nouveau_besoin(nom: str = "Repère 1") -> Dict:
    b = copy.deepcopy(DEFAUT)
    b["nom"] = nom
    return b


# ------------------------------------------------------------------ la relecture d'un besoin
def _nombre(x, nom: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{nom} : un nombre est attendu (reçu {x!r})")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{nom} : nombre infini ou indéfini")
    return x


def _entier(x, nom: str) -> int:
    v = _nombre(x, nom)
    if v != int(v):
        raise ValueError(f"{nom} : un nombre entier est attendu (reçu {x!r})")
    return int(v)


def _diametre(x, nom: str) -> int:
    d = _entier(x, nom)
    if d not in T.DIAMETRES:
        raise ValueError(f"{nom} : Ø{d} n'est pas un diamètre du tableau ({', '.join(map(str, T.DIAMETRES))})")
    return d


def normaliser(besoin: Dict) -> Dict:
    """Une copie complétée (défauts) et contrôlée d'un besoin ; ValueError claire si une valeur sort du domaine."""
    if not isinstance(besoin, dict):
        raise ValueError("un besoin d'armatures est un objet JSON")
    inconnues = sorted(set(besoin) - set(DEFAUT))
    if inconnues:
        raise ValueError(f"clé(s) inconnue(s) : {', '.join(inconnues)}")
    b = copy.deepcopy(DEFAUT)
    b.update({k: v for k, v in besoin.items() if v is not None})
    out = {"nom": b["nom"] if isinstance(b["nom"], str) else str(b["nom"])}
    if len(out["nom"]) > NOM_MAX:
        raise ValueError(f"le nom du repère va jusqu'à {NOM_MAX} caractères (reçu {len(out['nom'])})")
    if b["mode"] not in MODES:
        raise ValueError(f"mode inconnu : {b['mode']!r} (barres ou nappe)")
    out["mode"] = b["mode"]
    out["As_req"] = _nombre(b["As_req"], "Section requise") + 0.0          # + 0.0 : « -0 » devient 0
    if out["As_req"] < 0:
        raise ValueError("la section requise ne peut pas être négative")
    if out["As_req"] > AS_REQ_MAX:
        raise ValueError(f"la section requise va au plus à {_nombre_txt(AS_REQ_MAX)} {UNITES[out['mode']]} (le plus grand nombre du tableau : {_nombre_txt(T.arrondi(T.section(T.NOMBRES[-1], T.DIAMETRES[-1])))})")
    out["d"] = _diametre(b["d"], "Ø")
    out["n"] = _entier(b["n"], "Nombre de barres")
    if not 1 <= out["n"] <= N_MAX_SAISIE:
        raise ValueError(f"le nombre de barres va de 1 à {N_MAX_SAISIE}")
    if b["espacement"] != LIBRE and b["espacement"] not in _ESP:
        raise ValueError(f"espacement inconnu : {b['espacement']!r}")
    out["espacement"] = b["espacement"]
    out["e_libre"] = _nombre(b["e_libre"], "Espacement")
    if not 0 < out["e_libre"] <= E_LIBRE_MAX:
        raise ValueError(f"l'espacement doit être compris entre 0 (exclu) et {E_LIBRE_MAX:g} cm (une barre par mètre au moins)")
    out["d_min"], out["d_max"] = _diametre(b["d_min"], "Ø mini"), _diametre(b["d_max"], "Ø maxi")
    if out["d_min"] > out["d_max"]:
        raise ValueError(f"Ø mini ({out['d_min']}) plus grand que Ø maxi ({out['d_max']})")
    out["n_min"], out["n_max"] = _entier(b["n_min"], "Nombre de barres mini"), _entier(b["n_max"], "Nombre de barres maxi")
    if not (T.NOMBRES[0] <= out["n_min"] <= out["n_max"] <= T.NOMBRES[-1]):
        raise ValueError(f"le nombre de barres de la recherche va de {T.NOMBRES[0]} à {T.NOMBRES[-1]} (le tableau), mini ≤ maxi")
    out["e_min"], out["e_max"] = _nombre(b["e_min"], "Espacement mini"), _nombre(b["e_max"], "Espacement maxi")
    if not 0 < out["e_min"] <= out["e_max"]:
        raise ValueError("les espacements de la recherche sont positifs, mini ≤ maxi")
    return out


# ------------------------------------------------------------------ une disposition
def _formule(cle: str, symbole: str, expression: str, termes: List, resultat: float, unite: str) -> Dict:
    return {"cle": cle, "symbole": symbole, "expression": expression,
            "termes": [{"symbole": s, "valeur": v, "unite": u} for s, v, u in termes], "resultat": resultat, "unite": unite}


def _colonne(mode: str, d: int, n: Optional[int], cle_e: Optional[str], e: Optional[float]) -> Optional[str]:
    """La colonne du tableau qui porte cette disposition (« n4 », « e15 »), ou None hors du tableau."""
    if mode == "barres":
        return f"n{n}" if n in T.NOMBRES else None
    if cle_e in _ESP:
        return cle_e
    for x in ESPACEMENTS:                      # un espacement libre égal, au bit près, à celui d'une colonne
        if e == x["e"]:
            return x["cle"]
    return None


def _dispo(b: Dict, mode: str, d: int, n: Optional[int] = None, cle_e: Optional[str] = None) -> Dict:
    """Le calcul d'une disposition : Ø ``d`` et ``n`` barres (mode barres), ou Ø ``d`` à l'espacement ``cle_e`` (une clé
    d'ESPACEMENTS, ou « libre » : b["e_libre"])."""
    a = T.aire(d)
    m = T.masse_lineique(d)
    f = [_formule("a", "a", "π·Ø²/4", [("Ø", d, "mm")], a, "mm²")]
    if mode == "barres":
        As = T.section(n, d)
        e = T.espacement(n)
        bpm = float(n)
        M = n * m
        nom = f"{n}Ø{d}"
        f += [_formule("As", "A_s", "n·a", [("n", n, ""), ("a", a, "mm²")], As, "mm²"),
              _formule("m", "m", "ρ·a·10⁻⁶", [("ρ", T.MASSE_VOLUMIQUE, "kg/m³"), ("a", a, "mm²")], m, "kg/m"),
              _formule("M", "M", "n·m", [("n", n, ""), ("m", m, "kg/m")], M, "kg/m"),
              _formule("e", "e", "100 / n", [("n", n, "")], e, "cm")]
        col = _colonne(mode, d, n, None, None)
        dans = col is not None
        crit = b["d_min"] <= d <= b["d_max"] and b["n_min"] <= n <= b["n_max"]
    else:
        if cle_e == LIBRE:
            e, k = b["e_libre"], None
        else:
            e, k = _ESP[cle_e]["e"], _ESP[cle_e]["barres_par_metre"]
        if k is not None:                     # une colonne « n barres par mètre » : la section est n·a, comme l'image
            As = T.section(k, d)
            bpm = float(k)
            f += [_formule("e", "e", "100 / n", [("n", k, "")], e, "cm"),
                  _formule("As", "A_s", "n·a", [("n", k, ""), ("a", a, "mm²")], As, "mm²/m")]
        else:
            As = T.section_par_metre(d, e)
            bpm = T.METRE / e
            f += [_formule("n", "n", "100 / e", [("e", e, "cm")], bpm, "barres/m"),
                  _formule("As", "A_s", "a·100 / e", [("a", a, "mm²"), ("e", e, "cm")], As, "mm²/m")]
        M = bpm * m
        f += [_formule("m", "m", "ρ·a·10⁻⁶", [("ρ", T.MASSE_VOLUMIQUE, "kg/m³"), ("a", a, "mm²")], m, "kg/m"),
              _formule("M", "M", "n·m", [("n", bpm, "barres/m"), ("m", m, "kg/m")], M, "kg/m²")]
        nom = f"Ø{d} · {libelle_espacement(e)} cm"
        col = _colonne(mode, d, None, cle_e, e)
        dans = col is not None
        crit = b["d_min"] <= d <= b["d_max"] and b["e_min"] <= e <= b["e_max"]
    taux = b["As_req"] / As
    if b["As_req"] > 0:
        f.append(_formule("taux", "η", "A_s,req / A_s", [("A_s,req", b["As_req"], UNITES[mode]), ("A_s", As, UNITES[mode])], taux, ""))
    out = {"mode": mode, "d": d, "e": e, "jour": e * 10 - d, "barres_par_metre": bpm, "nom": nom, "aire": a, "As": As, "As_arrondi": T.arrondi(As),
           "unite": UNITES[mode], "masse_barre": m, "masse": M, "unite_masse": "kg/m" if mode == "barres" else "kg/m²",
           "dans_tableau": dans, "colonne": col, "dans_criteres": crit, "taux": taux, "suffit": As >= b["As_req"], "formules": f}
    if mode == "barres":
        out["n"] = n
    else:
        out["espacement"] = cle_e
    return out


def disposition(besoin: Dict) -> Dict:
    """La disposition RETENUE du besoin (normalisé)."""
    b = besoin
    if b["mode"] == "barres":
        return _dispo(b, "barres", b["d"], n=b["n"])
    return _dispo(b, "nappe", b["d"], cle_e=b["espacement"])


def _ordre(x: Dict):
    """Le tri du sélecteur et de la recherche : la section, puis le moins de barres, puis le plus petit Ø. La section
    est comparée au millionième de mm² : deux dispositions de même section (2Ø16 et 8Ø8 : n·Ø² = 512 ; Ø16 à 40 cm et
    Ø8 à 10 cm) ne sont pas départagées par le dernier bit d'un flottant, mais par le nombre de barres."""
    return (round(x["As"], 6), x["barres_par_metre"], x["d"])


def _retenue(b: Dict, x: Dict) -> bool:
    if b["mode"] == "barres":
        return x["d"] == b["d"] and x.get("n") == b["n"]
    return x["d"] == b["d"] and x.get("espacement") == b["espacement"]


def candidats(besoin: Dict, avec_retenue: bool = False) -> List[Dict]:
    """Les cases du tableau qui servent au mode, dans les critères de recherche, triées par section (puis le moins de
    barres, puis le plus petit Ø), chacune marquée « retenue » et « optimale ». ``avec_retenue`` : la disposition
    retenue y est ajoutée si elle n'y est pas (hors critères, ou hors du tableau)."""
    b = besoin
    out = []
    for d in T.DIAMETRES:
        if not b["d_min"] <= d <= b["d_max"]:
            continue
        if b["mode"] == "barres":
            out += [_dispo(b, "barres", d, n=n) for n in T.NOMBRES if b["n_min"] <= n <= b["n_max"]]
        else:
            out += [_dispo(b, "nappe", d, cle_e=x["cle"]) for x in ESPACEMENTS if b["e_min"] <= x["e"] <= b["e_max"]]
    out.sort(key=_ordre)
    opt = next((x for x in out if x["suffit"]), None) if b["As_req"] > 0 else None
    for x in out:
        x["retenue"] = _retenue(b, x)
        x["optimale"] = x is opt
    if avec_retenue and not any(x["retenue"] for x in out):
        r = disposition(b)
        r["retenue"], r["optimale"] = True, False
        out.append(r)
        out.sort(key=_ordre)
    return out


def optimale(besoin: Dict) -> Optional[Dict]:
    """La disposition la plus économique qui suffit : la plus petite section ≥ A_s,req parmi les cases du tableau dans
    les critères (à section égale : le moins de barres, puis le plus petit Ø) ; None sans section requise, ou si
    aucune ne suffit."""
    return next((x for x in candidats(besoin) if x["optimale"]), None)


# ------------------------------------------------------------------ le tableau marqué et le calcul complet
def _tableau_marque(b: Dict, retenue: Dict, opt: Optional[Dict]) -> Dict:
    """Le tableau de l'image, chaque case marquée pour ce besoin : utile au mode, dans les critères, suffit, optimale,
    retenue."""
    tab = T.tableau()
    lignes = []
    for li in tab["lignes"]:
        d = li["d"]
        cases = []
        for j, col in enumerate(COLONNES):
            if col.startswith("n"):
                n = int(col[1:])
                val, arr = li["barres"][n - 1], li["barres_arrondi"][n - 1]
                e = T.espacement(n)
                crit_n = b["n_min"] <= n <= b["n_max"] if b["mode"] == "barres" else b["e_min"] <= e <= b["e_max"]
                utile = True
            else:
                e = float(col[1:])
                val, arr = li["fixes"][j - len(T.NOMBRES)], li["fixes_arrondi"][j - len(T.NOMBRES)]
                crit_n = b["e_min"] <= e <= b["e_max"]
                utile = b["mode"] == "nappe"
            crit = utile and crit_n and b["d_min"] <= d <= b["d_max"]
            suffit = utile and b["As_req"] > 0 and val >= b["As_req"]
            cases.append({"colonne": col, "valeur": val, "arrondi": arr, "utile": utile, "dans_criteres": crit, "suffit": suffit,
                          "optimale": bool(opt and opt["d"] == d and opt["colonne"] == col),
                          "retenue": bool(retenue["dans_tableau"] and retenue["d"] == d and retenue["colonne"] == col)})
        lignes.append({"d": d, "aire": li["aire"], "masse": li["masse"], "masse_txt": li["masse_txt"], "cases": cases})
    return {"colonnes": list(COLONNES), "espacements_n": tab["espacements_n"], "espacements_n_txt": tab["espacements_n_txt"],
            "espacements_fixes": tab["espacements_fixes"], "masse_volumique": tab["masse_volumique"], "lignes": lignes}


def calculer(besoin: Dict) -> Dict:
    """Le besoin (normalisé ici) : sa disposition retenue, la plus économique, le tableau marqué, les avertissements."""
    b = normaliser(besoin)
    r = disposition(b)
    cands = candidats(b)
    opt = next((x for x in cands if x["optimale"]), None)
    av = []
    u = UNITES[b["mode"]]
    if b["As_req"] > 0 and not r["suffit"]:
        av.append({"code": "insuffisante", "message": f"La disposition retenue {r['nom']} ne suffit pas\u00a0: {section_txt(r['As'], b['As_req'])} {u} "
                                                      f"< {_nombre_txt(b['As_req'])} {u} requis."})
    if not r["dans_tableau"]:
        av.append({"code": "hors_tableau", "message": f"{r['nom']} n'est pas une case du tableau du bureau\u00a0: même formule, "
                                                      "valeur nouvelle sur le site — à valider par le bureau."})
    if b["mode"] == "nappe" and r["jour"] <= 0:
        av.append({"code": "chevauchement", "message": f"Espacement de {libelle_espacement(r['e'])} cm pour des Ø{r['d']}\u00a0: les barres se "
                                                       "touchent ou se chevauchent (jour e − Ø ≤ 0)."})
    if b["As_req"] > 0 and opt is None:
        av.append({"code": "aucune", "message": "Aucune disposition du tableau ne suffit dans les critères de recherche\u00a0: "
                                                "élargir les Ø, le nombre de barres ou les espacements."})
    return {"reference": REFERENCE, "besoin": b, "disposition": r, "optimale": opt,
            "suffisantes": sum(1 for x in cands if x["suffit"]) if b["As_req"] > 0 else 0,
            "tableau": _tableau_marque(b, r, opt), "avertissements": av}
