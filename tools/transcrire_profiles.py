# -*- coding: utf-8 -*-
"""Transcription du catalogue des profilés — HORS chemin de production.

Produit ``acier/donnees/profiles.json`` (la source unique du catalogue) et
la MESURE datée ``docs/validation/profiles_transcription_<date>.md``.

Ce que le script lit (des DONNÉES seulement, aucun code étranger n'est
exécuté — les tables Python sont lues par ``ast.literal_eval``) :

- dimensions nominales des profilés laminés (h, b, t_w, t_f, r, r1, r2 ;
  mm), faits normatifs (EN 10365, EN 10056-1, DIN 1025/1026) relevés dans
  des paquets publiés : structuralcodes (fib, Apache-2.0) pour IPE, HE A/B/M,
  IPN, UPN, UPE, HD, HP, L et LI ; steelsnakes (GPL-2.0) pour HE AA, HL,
  IPE A / O / 750 et les cornières complémentaires. Seules les DIMENSIONS
  sont reprises ; aucune propriété calculée de ces paquets n'est livrée ;
- listes de tailles des profils creux : EN 10210 (eurocodepy, LGPL-3,
  et steelsnakes UK) et EN 10219 (steelsnakes UK, gamme Blue Book) ;
- valeurs PUBLIÉES (A, I, W, i, I_t, I_w, masse) des mêmes paquets et de
  steel-sections (Apache-2.0), pour le CONTRÔLE CROISÉ du rapport — elles
  ne sont pas livrées non plus.

Toutes les propriétés livrées sont CALCULÉES par ``acier/profiles``
(intégration du contour, Saint-Venant, Vlasov, EN 1993-1-1 §6.2.6,
EN 10210-2 annexe B). scipy, s'il est présent, sert de solveur direct pour
Saint-Venant (plus rapide que le gradient conjugué numpy du moteur, même
résultat à 10⁻⁹) ; il n'entre jamais dans le chemin de production.

Lancement : python3 tools/transcrire_profiles.py <dossier des paquets dépaquetés>
            (sous-dossiers x_sc, x_ss, x_euro, x_steel)
"""
import ast
import csv
import json
import os
import re
import sys
import time
from datetime import date

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
os.chdir(RACINE)

import numpy as np  # noqa: E402

from acier.profiles import familles as F  # noqa: E402

SORTIE = os.path.join("acier", "donnees", "profiles.json")
RAPPORTS = os.path.join("docs", "validation")
VERSION = 1


# ------------------------------------------------------------------ lecture des sources
def table_python(chemin):
    """Le dictionnaire ``parameters = {...}`` d'un fichier Python, lu comme
    un littéral (ast.literal_eval), jamais importé."""
    txt = open(chemin, encoding="utf-8").read()
    i = txt.index("parameters = {")
    j = i + len("parameters = ")
    depth = 0
    for k in range(j, len(txt)):
        if txt[k] == "{":
            depth += 1
        elif txt[k] == "}":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(txt[j:k + 1])
    raise ValueError(chemin)


def nom_sc(cle):
    """'IPE80' → 'IPE 80', 'HEA100' → 'HEA 100', 'L20x20x3' → 'L 20x20x3', 'HD260x54.1' → 'HD 260x54.1'."""
    m = re.match(r"([A-Z]+)(\d.*)", cle)
    return f"{m.group(1)} {m.group(2)}"


def g(x):
    return f"{float(x):g}"


def nom_ss(cle):
    """Clés steelsnakes EU → nom du catalogue : 'HE-100-AA' → 'HEAA 100',
    'HE-200-A' → 'HEA 200', 'IPE-80-A' → 'IPE A 80', 'IPE-750x147' →
    'IPE 750x147', 'HL-920x344' → 'HL 920x344', 'HL-1000-A' → 'HL 1000 A',
    'UPE-200' → 'UPE 200'."""
    p = cle.split("-")
    fam, reste = p[0], p[1:]
    if fam == "HE" and len(reste) == 2:
        return f"HE{reste[1]} {reste[0]}"
    if fam == "IPE" and len(reste) == 2:
        return f"IPE {reste[1]} {reste[0]}"
    if fam == "HL" and len(reste) == 2:
        return f"HL {reste[0]} {reste[1]}"
    return f"{fam} {'-'.join(reste)}"


def dims_laminés(dossier):
    """{famille: {nom: dims}} des profilés laminés."""
    sc = os.path.join(dossier, "x_sc", "structuralcodes", "geometry", "profiles")
    ss = os.path.join(dossier, "x_ss", "steelsnakes", "EU", "data")
    out = {k: {} for k in ("IPE", "IPE A", "IPE O", "IPN", "HEA", "HEB", "HEM", "HEAA", "HL", "HD", "HP", "UPE", "UPN", "L", "LI")}
    cles5 = ("h", "b", "tw", "tf", "r")
    for cle, d in table_python(os.path.join(sc, "_ipe.py")).items():
        out["IPE"][nom_sc(cle)] = {k: d[k] for k in cles5}
    for cle, d in table_python(os.path.join(sc, "_he.py")).items():
        fam = cle[:3]
        out[fam][nom_sc(cle)] = {k: d[k] for k in cles5}
    for cle, d in table_python(os.path.join(sc, "_ipn.py")).items():
        out["IPN"][nom_sc(cle)] = {k: d[k] for k in ("h", "b", "tw", "tf", "r1", "r2")}
    for cle, d in table_python(os.path.join(sc, "_upn.py")).items():
        out["UPN"][nom_sc(cle)] = {k: d[k] for k in ("h", "b", "tw", "tf", "r1", "r2")}
    for cle, d in table_python(os.path.join(sc, "_upe.py")).items():
        out["UPE"][nom_sc(cle)] = {k: d[k] for k in cles5}
    for cle, d in table_python(os.path.join(sc, "_hd.py")).items():
        out["HD"][nom_sc(cle)] = {k: d[k] for k in cles5}
    for cle, d in table_python(os.path.join(sc, "_hp.py")).items():
        out["HP"][nom_sc(cle)] = {k: d[k] for k in cles5}
    for cle, d in table_python(os.path.join(sc, "_l.py")).items():
        out["L"][nom_sc(cle)] = {"h": d["h"], "b": d["b"], "t": d["t"], "r1": d["r1"], "r2": d["r2"]}
    for cle, d in table_python(os.path.join(sc, "_li.py")).items():
        out["LI"][nom_sc(cle)] = {"h": d["h"], "b": d["b"], "t": d["t"], "r1": d["r1"], "r2": d["r2"]}
    # compléments : HE AA, HL, IPE A / O / 750, cornières manquantes (dimensions seulement)
    he = json.load(open(os.path.join(ss, "HE.json"), encoding="utf-8"))
    for cle, d in he.items():
        if cle.endswith("-AA"):
            out["HEAA"][nom_ss(cle)] = {k: float(d[k]) for k in cles5}
    for cle, d in json.load(open(os.path.join(ss, "HL.json"), encoding="utf-8")).items():
        dims = {k: float(d[k]) for k in cles5}
        if dims["h"] >= 1090.0 and dims["r"] == 30.0:
            # HL 1100 : la source donne r = 30 (valeur des HE 1000), mais sa masse publiée
            # (343 kg/m pour HL 1100 A) n'est retrouvée qu'avec r = 20 : 2·400·31 + 1028·18 +
            # (4 − π)·20² = 43 647 mm² → 342,6 kg/m, contre 346 avec r = 30. Le rayon est donc
            # DÉDUIT de la masse publiée, et la fiche le dit (``note``).
            dims["r"] = 20.0
            dims["_note"] = "r = 20 mm déduit de la masse publiée (la source portait 30, incompatible avec 343 kg/m)"
        out["HL"][nom_ss(cle)] = dims
    for cle, d in json.load(open(os.path.join(ss, "IPE.json"), encoding="utf-8")).items():
        p = cle.split("-")
        if len(p) == 2 and "x" in p[1]:
            out["IPE"][nom_ss(cle)] = {k: float(d[k]) for k in cles5}
        elif len(p) == 3 and p[2] == "A":
            out["IPE A"][nom_ss(cle)] = {k: float(d[k]) for k in cles5}
        elif len(p) == 3 and p[2] == "O":
            out["IPE O"][nom_ss(cle)] = {k: float(d[k]) for k in cles5}
    for fichier, fam in (("L_EQUAL.json", "L"), ("L_UNEQUAL.json", "LI")):
        for cle, d in json.load(open(os.path.join(ss, fichier), encoding="utf-8")).items():
            hh, bb, tt = cle.split("x")
            nom = f"L {g(hh)}x{g(bb)}x{g(tt)}"
            if nom not in out[fam]:
                # taille hors de la liste EN 10056-1 de structuralcodes : gamme étendue du
                # lamineur, gardée mais hors « séries courantes »
                out[fam][nom] = {"h": float(hh), "b": float(bb), "t": float(tt), "r1": float(d["r_1"]), "r2": float(d["r_2"]),
                                 "_courant": False}
    return out


def dims_creux(dossier):
    """{famille: {nom: dims}} des profils creux : EN 10210 (eurocodepy ∪ Blue
    Book finis à chaud), EN 10219 (Blue Book formés à froid)."""
    eu = os.path.join(dossier, "x_euro", "eurocodepy", "data")
    uk = os.path.join(dossier, "x_ss", "steelsnakes", "UK", "data")
    out = {k: {} for k in ("CHS", "SHS", "RHS", "CHS-F", "SHS-F", "RHS-F")}
    for p in json.load(open(os.path.join(eu, "chs_profiles_euro.json"), encoding="utf-8")):
        d, t = round(p["h"] * 10, 2), round(p["tw"] * 10, 2)
        out["CHS"][f"CHS {g(d)}x{g(t)}"] = {"d": d, "t": t}
    for fichier, fam in (("shs_profiles_euro.json", "SHS"), ("rhs_profiles_euro.json", "RHS")):
        for p in json.load(open(os.path.join(eu, fichier), encoding="utf-8")):
            h, b, t = round(p["h"] * 10, 2), round(p["b"] * 10, 2), round(p["tw"] * 10, 2)
            out[fam][f"{fam} {g(h)}x{g(b)}x{g(t)}"] = {"h": h, "b": b, "t": t}
    for fichier, fam, suffixe in (("HFCHS.json", "CHS", ""), ("CFCHS.json", "CHS-F", " (froid)")):
        for cle, p in json.load(open(os.path.join(uk, fichier), encoding="utf-8")).items():
            d, t = float(p["d"]), float(p["t"])
            out[fam].setdefault(f"CHS {g(d)}x{g(t)}{suffixe}", {"d": d, "t": t})
    for fichier, fam, suffixe in (("HFSHS.json", "SHS", ""), ("HFRHS.json", "RHS", ""),
                                  ("CFSHS.json", "SHS-F", " (froid)"), ("CFRHS.json", "RHS-F", " (froid)")):
        for cle, p in json.load(open(os.path.join(uk, fichier), encoding="utf-8")).items():
            hh, bb, tt = cle.split("x")
            h, b, t = float(hh), float(bb), float(tt)
            base = "SHS" if fam.startswith("SHS") else "RHS"
            out[fam].setdefault(f"{base} {g(h)}x{g(b)}x{g(t)}{suffixe}", {"h": h, "b": b, "t": t})
    return out


def references(dossier):
    """Valeurs publiées pour le contrôle croisé : {nom: {source: {prop: valeur}}}
    en unités de catalogue (cm², cm⁴, cm³, cm⁶, kg/m)."""
    ss = os.path.join(dossier, "x_ss", "steelsnakes", "EU", "data")
    uk = os.path.join(dossier, "x_ss", "steelsnakes", "UK", "data")
    eu = os.path.join(dossier, "x_euro", "eurocodepy", "data")
    st = os.path.join(dossier, "x_steel", "steel_sections", "data")
    refs = {}

    def ajouter(nom, source, vals):
        vals = {k: float(v) for k, v in vals.items() if v is not None and v != ""}
        refs.setdefault(nom, {})[source] = vals

    champs_ss = {"A": "A", "I_yy": "Iy", "I_zz": "Iz", "W_el_yy": "Wel_y", "W_el_zz": "Wel_z", "W_pl_yy": "Wpl_y",
                 "W_pl_zz": "Wpl_z", "i_yy": "iy", "i_zz": "iz", "I_t": "It", "mass_per_metre": "masse"}
    for fichier in ("HE.json", "IPE.json", "HL.json", "HD.json", "HP.json", "UPE.json", "UPN.json"):
        for cle, d in json.load(open(os.path.join(ss, fichier), encoding="utf-8")).items():
            vals = {v: d.get(k) for k, v in champs_ss.items()}
            if d.get("I_w") is not None:
                vals["Iw"] = float(d["I_w"]) * 1e6                 # dm⁶ → cm⁶
            ajouter(nom_ss(cle), "steelsnakes (ArcelorMittal)", vals)
    for fichier in ("L_EQUAL.json", "L_UNEQUAL.json"):
        for cle, d in json.load(open(os.path.join(ss, fichier), encoding="utf-8")).items():
            hh, bb, tt = cle.split("x")
            vals = {"A": d.get("A"), "Iy": d.get("I_yy"), "Iz": d.get("I_zz"), "Iu": d.get("I_uu"), "Iv": d.get("I_vv"),
                    "Wel_y": d.get("W_el_yy"), "It": d.get("I_t"), "masse": d.get("mass_per_metre")}
            ajouter(f"L {g(hh)}x{g(bb)}x{g(tt)}", "steelsnakes (ArcelorMittal)", vals)
    for fichier, fam, suffixe in (("HFCHS.json", "CHS", ""), ("CFCHS.json", "CHS", " (froid)")):
        for cle, d in json.load(open(os.path.join(uk, fichier), encoding="utf-8")).items():
            ajouter(f"CHS {g(d['d'])}x{g(d['t'])}{suffixe}", "steelsnakes (Blue Book)",
                    {"A": d["A"], "Iy": d["I"], "Wel_y": d["W_el"], "Wpl_y": d["W_pl"], "It": d["I_t"], "Wt": d["W_t"], "masse": d["mass_per_metre"]})
    for fichier, base, suffixe in (("HFSHS.json", "SHS", ""), ("CFSHS.json", "SHS", " (froid)")):
        for cle, d in json.load(open(os.path.join(uk, fichier), encoding="utf-8")).items():
            hh, bb, tt = cle.split("x")
            ajouter(f"{base} {g(hh)}x{g(bb)}x{g(tt)}{suffixe}", "steelsnakes (Blue Book)",
                    {"A": d["A"], "Iy": d["I"], "Wel_y": d["W_el"], "Wpl_y": d["W_pl"], "It": d["I_t"], "Wt": d["W_t"], "masse": d["mass_per_metre"]})
    for fichier, base, suffixe in (("HFRHS.json", "RHS", ""), ("CFRHS.json", "RHS", " (froid)")):
        for cle, d in json.load(open(os.path.join(uk, fichier), encoding="utf-8")).items():
            hh, bb, tt = cle.split("x")
            ajouter(f"{base} {g(hh)}x{g(bb)}x{g(tt)}{suffixe}", "steelsnakes (Blue Book)",
                    {"A": d["A"], "Iy": d["I_yy"], "Iz": d["I_zz"], "Wel_y": d["W_el_yy"], "Wpl_y": d["W_pl_yy"],
                     "It": d["I_t"], "Wt": d["W_t"], "masse": d["mass_per_metre"]})
    champs_eu = {"A": "A", "Av_z": "Avz", "Iy": "Iy", "iy": "iy", "Wel_y": "Wel_y", "Wpl_y": "Wpl_y", "Iz": "Iz",
                 "Wel_z": "Wel_z", "Wpl_z": "Wpl_z", "IT": "It", "WT": "Wt", "Iw": "Iw", "m": "masse"}
    for fichier, fam in (("chs_profiles_euro.json", "CHS"), ("shs_profiles_euro.json", "SHS"), ("rhs_profiles_euro.json", "RHS")):
        for p in json.load(open(os.path.join(eu, fichier), encoding="utf-8")):
            if fam == "CHS":
                nom = f"CHS {g(round(p['h'] * 10, 2))}x{g(round(p['tw'] * 10, 2))}"
            else:
                nom = f"{fam} {g(round(p['h'] * 10, 2))}x{g(round(p['b'] * 10, 2))}x{g(round(p['tw'] * 10, 2))}"
            ajouter(nom, "eurocodepy", {v: p.get(k) for k, v in champs_eu.items()})
    for p in json.load(open(os.path.join(eu, "i_profiles_euro.json"), encoding="utf-8")):
        ajouter(nom_sc(p["Section"]), "eurocodepy", {v: p.get(k) for k, v in champs_eu.items()})
    for fichier in ("ipe.csv", "hea.csv", "heb.csv", "hem.csv", "upn.csv"):
        with open(os.path.join(st, fichier), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ajouter(row["section"], "steel-sections (structuralcodes)",
                        {"A": row.get("area_cm2"), "Iy": row.get("iy_cm4"), "Iz": row.get("iz_cm4"), "masse": row.get("mass_kg_m")})
    return refs


# ------------------------------------------------------------------ solveur direct (option)
def solveur_scipy():
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
    except ImportError:
        return None

    def solveur(masque, diag, cE, cW, cN, cS):
        """Assemble le stencil de Shortley-Weller du moteur (mêmes coefficients)
        en matrice creuse et résout directement."""
        ny, nz = masque.shape
        n = int(masque.sum())
        idx = -np.ones(masque.shape, dtype=np.int64)
        idx[masque] = np.arange(n)
        rows = [np.arange(n)]
        cols = [np.arange(n)]
        vals = [diag[masque]]
        # voisin Est (i+1) et Ouest (i−1)
        a = masque[:-1, :] & masque[1:, :]
        i1, i2 = idx[:-1, :][a], idx[1:, :][a]
        rows += [i1, i2]
        cols += [i2, i1]
        vals += [-cE[:-1, :][a], -cW[1:, :][a]]
        # voisin Nord (j+1) et Sud (j−1)
        a = masque[:, :-1] & masque[:, 1:]
        i1, i2 = idx[:, :-1][a], idx[:, 1:][a]
        rows += [i1, i2]
        cols += [i2, i1]
        vals += [-cN[:, :-1][a], -cS[:, 1:][a]]
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        phi = spl.spsolve(A, 2.0 * np.ones(n))
        out = np.zeros(masque.shape)
        out[masque] = phi
        return out
    return solveur


# ------------------------------------------------------------------ construction
ORDRE = ["IPE", "IPE A", "IPE O", "IPN", "HEA", "HEB", "HEM", "HEAA", "HL", "HD", "HP", "UPE", "UPN", "L", "LI",
         "CHS", "SHS", "RHS", "CHS-F", "SHS-F", "RHS-F"]
PROPS = ["A", "masse", "perimetre", "Iy", "Wel_y", "Wpl_y", "iy", "Avz", "Iz", "Wel_z", "Wpl_z", "iz", "Avy",
         "It", "Wt", "Iw", "Iu", "Iv", "Wel_u", "Wel_v", "iu", "iv", "alpha_deg", "Iyz", "yc", "zc", "y_s", "z_s",
         "cf_tf", "cw_tw", "d_ame", "z_sup", "z_inf", "y_droite", "y_gauche"]


def arrondi(v, n=6):
    if v is None:
        return None
    v = float(v)
    if v == 0.0:
        return 0.0
    return float(f"{v:.{n}g}")


def construire(dossier, solveur=None, journal=print):
    laminés = dims_laminés(dossier)
    creux = dims_creux(dossier)
    tout = {**laminés, **creux}
    profils = []
    t0 = time.time()
    for fam in ORDRE:
        entrees = tout[fam]
        cles_tri = lambda kv: (kv[1].get("h", kv[1].get("d", 0.0)), kv[1].get("b", 0.0), kv[1].get("tf", kv[1].get("t", 0.0)))  # noqa: E731
        for nom, dims in sorted(entrees.items(), key=cles_tri):
            note = dims.get("_note")
            courant = bool(dims.get("_courant", F.FAMILLES[fam]["courant"]))
            dims = {k: v for k, v in dims.items() if not k.startswith("_")}
            p = F.calculer(fam, dims, torsion_exacte=True, solveur=solveur)
            rec = {"nom": nom, "famille": fam, "courant": courant,
                   "dims": {k: arrondi(v) for k, v in dims.items()}}
            if note:
                rec["note"] = note
            for k in PROPS:
                if k in p:
                    rec[k] = arrondi(p[k])
            rec["methode_It"] = p.get("methode_It")
            profils.append(rec)
        journal(f"  {fam:6s} {len(entrees):4d} profilés  ({time.time() - t0:.0f} s)")
    return profils


def controler(profils, refs):
    """Écarts relatifs des propriétés calculées aux valeurs publiées, par
    famille, propriété et source : (n, écart max, écart moyen, pires)."""
    out = {}
    for rec in profils:
        r = refs.get(rec["nom"])
        if not r:
            continue
        for source, vals in r.items():
            for k, v in vals.items():
                c = rec.get(k)
                if c is None or v == 0:
                    continue
                e = abs(c / v - 1.0)
                cle = (rec["famille"], k, source)
                d = out.setdefault(cle, {"n": 0, "max": 0.0, "somme": 0.0, "pires": []})
                d["n"] += 1
                d["somme"] += e
                d["max"] = max(d["max"], e)
                d["pires"].append((e, rec["nom"], c, v))
    for d in out.values():
        d["pires"].sort(reverse=True)
        d["pires"] = d["pires"][:3]
    return out


TOLERANCES = {"A": 0.01, "masse": 0.01, "Iy": 0.015, "Iz": 0.02, "Wel_y": 0.015, "Wel_z": 0.02, "Wpl_y": 0.015,
              "Wpl_z": 0.02, "iy": 0.01, "iz": 0.015, "Avz": 0.02, "Iu": 0.02, "Iv": 0.03, "It": 0.08, "Wt": 0.05, "Iw": 0.10}
# Ce que chaque dépassement VEUT DIRE, écrit après lecture des pires cas (04/11/2026) ; une clé
# ("*", propriété) vaut pour toutes les familles.
LECTURES = {
    ("*", "Wel_z"): "la source publie W_el,z arrondi à l'unité pour les petits profilés (IPE 80 : « 4 » pour 3,69) ; "
                    "l'écart est l'arrondi de la source, pas le calcul.",
    ("*", "It"): "les catalogues emploient la formule empirique I_t = 2/3·b·t_f³ + 1/3·(h − 2t_f)·t_w³ + 2·α·D⁴ − 0,42·t_f⁴, "
                 "dont le terme de congé 2·α·D⁴ est un ajustement ; pour des semelles minces à grand congé (HE AA, "
                 "HD légers : r/t_f ≈ 2,5) il surestime I_t de 8 à 10 % par rapport à la solution exacte de Saint-Venant, "
                 "qui est retenue (convergée à 0,1 % entre deux pas de grille ; sécuritaire pour le déversement).",
    ("HL", "A"): "la source donnait r = 30 mm aux HL 1100 ; sa propre masse publiée n'est retrouvée qu'avec r = 20 mm : "
                 "le rayon a été corrigé à 20 et la fiche le dit — ce dépassement est celui de la source.",
    ("UPE", "Wpl_z"): "W_pl,z par l'axe d'aires égales (dans l'âme pour les grands UPE) : UPE 400 → âme 400 × 13,5 = 54 cm² sur "
                      "91,9, axe à 11,5 mm du dos, 2·(18 × 101,5)·52,75 + 400·(11,5²/2 + 2²/2) = 220 cm³, calcul à la main ; "
                      "la valeur publiée (191) ne correspond pas à cette définition et n'est pas reprise.",
    ("UPN", "Wpl_y"): "semelles coniques : la source arrondit et modélise l'inclinaison autrement ; 2 % au pire.",
    ("UPN", "iy"): "arrondi de la source (12,1 pour 11,98).",
    ("UPN", "Iw"): "I_w de Vlasov sur la ligne moyenne à épaisseur variable ; les catalogues emploient un modèle voisin ; "
                   "l'écart restant est celui des deux modèles de parois minces d'une aile conique.",
    ("L", "A"): "tailles hors EN 10056-1 (300 × 300 × 26 à 35) : la source arrondit l'aire à 3 chiffres ; écart ≤ 1,1 %.",
    ("L", "masse"): "L 110 × 110 × 14 : la source publie A = 29,0 cm² (= calcul) mais une masse de 23,3 kg/m, "
                    "incohérente avec son aire (22,8) ; l'erreur est dans la source.",
    ("L", "Iy"): "L 200 × 200 × 16 : la source publie 2 430 ; le calcul à la main (ailes 16 × 200 et 184 × 16, "
                 "centre de gravité à 55,9 mm, congés) donne 2 342 ; chiffres inversés dans la source (2 340).",
    ("L", "Iz"): "même cas que I_y (cornière égale).",
    ("L", "Wel_y"): "suit I_y et l'arrondi de la source sur les très grandes cornières.",
    ("SHS", "Avz"): "eurocodepy n'emploie pas A·h/(b + h) de l'EN 1993-1-1 §6.2.6(3) f) ; la formule de la norme est retenue.",
    ("RHS", "Avz"): "idem SHS.",
}


def rapport(profils, ctrl, duree, jour):
    fam_n = {}
    for p in profils:
        fam_n[p["famille"]] = fam_n.get(p["famille"], 0) + 1
    L = [f"# Transcription du catalogue des profilés — {jour}", "",
         "*MESURE datée (ne se réécrit pas). Produite par `tools/transcrire_profiles.py` : dimensions nominales "
         "relevées dans des paquets publiés, toutes les propriétés CALCULÉES par `acier/profiles` (intégration du "
         "contour, Saint-Venant par différences finies, Vlasov, EN 1993-1-1 §6.2.6, EN 10210-2 annexe B), puis "
         "comparées aux valeurs publiées par trois sources indépendantes. Rien n'est recopié de mémoire.*", "",
         f"**{len(profils)} profilés dans {len(fam_n)} familles, calculés en {duree:.0f} s.**", "",
         "| Famille | Nombre | Norme des dimensions | Source des dimensions |", "|---|---|---|---|"]
    src = {"IPE": "structuralcodes (IPE 80–600) + steelsnakes (IPE 750)", "IPE A": "steelsnakes", "IPE O": "steelsnakes",
           "IPN": "structuralcodes", "HEA": "structuralcodes", "HEB": "structuralcodes", "HEM": "structuralcodes",
           "HEAA": "steelsnakes", "HL": "steelsnakes", "HD": "structuralcodes", "HP": "structuralcodes",
           "UPE": "structuralcodes", "UPN": "structuralcodes", "L": "structuralcodes + steelsnakes (tailles manquantes)",
           "LI": "structuralcodes + steelsnakes (tailles manquantes)", "CHS": "eurocodepy ∪ Blue Book (finis à chaud)",
           "SHS": "eurocodepy ∪ Blue Book", "RHS": "eurocodepy ∪ Blue Book", "CHS-F": "Blue Book (formés à froid)",
           "SHS-F": "Blue Book", "RHS-F": "Blue Book"}
    for fam in ORDRE:
        if fam in fam_n:
            L.append(f"| {fam} | {fam_n[fam]} | {F.FAMILLES[fam]['norme']} | {src.get(fam, '')} |")
    L += ["", "## Contrôle croisé : propriétés calculées contre valeurs publiées", "",
          "Écart relatif |calculé / publié − 1| ; tolérance de la famille × propriété entre parenthèses. "
          "Un dépassement est listé avec ses trois pires cas ; il ne corrige JAMAIS une valeur à la main (RV2) : "
          "il questionne une dimension ou une convention, et la réponse est écrite ici.", "",
          "| Famille | Propriété | Source | n | écart max | écart moyen | tolérance | verdict |", "|---|---|---|---|---|---|---|---|"]
    depassements = []
    for (fam, k, source), d in sorted(ctrl.items(), key=lambda kv: (ORDRE.index(kv[0][0]), PROPS.index(kv[0][1]) if kv[0][1] in PROPS else 99, kv[0][2])):
        tol = TOLERANCES.get(k, 0.05)
        verdict = "OK" if d["max"] <= tol else "DÉPASSEMENT"
        if verdict != "OK":
            depassements.append((fam, k, source, d))
        L.append(f"| {fam} | {k} | {source} | {d['n']} | {d['max'] * 100:.2f} % | {d['somme'] / d['n'] * 100:.2f} % | {tol * 100:.0f} % | {verdict} |")
    L += ["", "## Dépassements et leur lecture", ""]
    if not depassements:
        L.append("Aucun.")
    for fam, k, source, d in depassements:
        pires = " ; ".join(f"{nom} : {c:.4g} calculé / {v:.4g} publié ({e * 100:.1f} %)" for e, nom, c, v in d["pires"])
        lecture = LECTURES.get((fam, k)) or LECTURES.get(("*", k)) or ""
        L.append(f"- **{fam} · {k}** ({source}) : {pires}." + (f" **Lecture :** {lecture}" if lecture else ""))
    L += ["", "## Méthodes et conventions", "",
          "- I/H à ailes parallèles : contour exact avec quatre congés r ; IPN : semelles à 14 %, t_f à b/4 de l'axe, "
          "congé r1, arrondi r2 ; UPN : semelles à 8 % (5 % au-delà de 300), t_f à b/2 du bout d'aile (h ≤ 300) ou au "
          "milieu de la partie inclinée (h > 300) — conventions retenues parce qu'elles retrouvent les aires publiées ;",
          "- cornières : congé r1 au talon, arrondis r2 aux bouts ; axes principaux u-v calculés ; I_w non défini ;",
          "- creux : coins r_ext = 1,5·t / r_int = t (EN 10210), 2·t à 3·t (EN 10219) ; I_t et W_t par l'annexe B "
          "de l'EN 10210-2 ; tubes ronds exacts ;",
          "- I_t des sections ouvertes : Saint-Venant par différences finies sur le contour exact, pas = épaisseur "
          "minimale / 20 (erreur de discrétisation ≈ +0,5 %, mesurée sur le rectangle contre la série exacte) ; les "
          "catalogues emploient une formule approchée avec terme de congé — l'écart observé est la différence des deux "
          "méthodes, pas une erreur de transcription ;",
          "- I_w : I_z·(h − t_f)²/4 pour les I/H (forme des catalogues), Vlasov sur la ligne moyenne pour les U ;",
          "- A_v : EN 1993-1-1 §6.2.6(3), sans le plancher η·h_w·t_w ;",
          "- masse : A × 7 850 kg/m³ ; surface : périmètre extérieur.", ""]
    return "\n".join(L) + "\n"


def main(dossier):
    jour = date.today().strftime("%Y-%m-%d")
    t0 = time.time()
    solveur = solveur_scipy()
    print("solveur :", "scipy (direct)" if solveur else "gradient conjugué numpy")
    profils = construire(dossier, solveur=solveur)
    duree = time.time() - t0
    refs = references(dossier)
    ctrl = controler(profils, refs)
    meta = {"version": VERSION, "date": jour, "nombre": len(profils),
            "unites": {"dims": "mm", "A": "cm²", "Avz": "cm²", "Avy": "cm²", "I": "cm⁴", "W": "cm³", "i": "cm",
                       "It": "cm⁴", "Wt": "cm³", "Iw": "cm⁶", "masse": "kg/m", "perimetre": "m²/m", "alpha_deg": "°",
                       "yc, zc, y_s, z_s, d_ame, z_sup, z_inf, y_droite, y_gauche": "mm"},
            "conventions": "EN 1993-1-1 §1.7 : y-y axe fort (horizontal), z-z axe faible ; W_el minimal ; "
                           "yc, zc depuis l'origine du contour (dos de l'âme pour les U, talon pour les L) ; "
                           "y_s, z_s = centre de cisaillement depuis le centre de gravité",
            "sources_dimensions": [
                "structuralcodes 0.7.3 (fib International, Apache-2.0) : IPE, HE A/B/M, IPN, UPN, UPE, HD, HP, L, LI — EN 10365, EN 10056-1, DIN 1025/1026",
                "steelsnakes 0.0.1a11 (GPL-2.0, dimensions nominales seulement) : HE AA, HL, IPE A, IPE O, IPE 750, cornières complémentaires ; gammes Blue Book EN 10210 / EN 10219 des creux",
                "eurocodepy 2027.0.12 (LGPL-3, listes de tailles seulement) : CHS, SHS, RHS EN 10210"],
            "controle": f"docs/validation/profiles_transcription_{jour}.md (valeurs publiées de steelsnakes, eurocodepy, steel-sections)",
            "calcul": "acier/profiles/familles.calculer — intégration du contour, Saint-Venant, Vlasov, EN 1993-1-1 §6.2.6, EN 10210-2 annexe B"}
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with open(SORTIE, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "profils": profils}, fh, ensure_ascii=False, separators=(",", ":"))
    os.makedirs(RAPPORTS, exist_ok=True)
    chemin = os.path.join(RAPPORTS, f"profiles_transcription_{jour}.md")
    with open(chemin, "w", encoding="utf-8") as fh:
        fh.write(rapport(profils, ctrl, duree, jour))
    n_dep = sum(1 for (fam, k, s), d in ctrl.items() if d["max"] > TOLERANCES.get(k, 0.05))
    print(f"{len(profils)} profilés → {SORTIE} ({os.path.getsize(SORTIE) // 1024} ko) ; rapport {chemin} ; "
          f"{len(ctrl)} contrôles, {n_dep} dépassement(s) ; {duree:.0f} s")
    return profils, ctrl


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
