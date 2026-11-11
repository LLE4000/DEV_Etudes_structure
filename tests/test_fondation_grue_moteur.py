# -*- coding: utf-8 -*-
"""Fondations de grues (beton/fondation_grue) : la FIDÉLITÉ aux quatre classeurs du bureau, cellule par cellule, et
la redémonstration de leurs formules. Référence : ``tests/ref_fondation_grue.json`` (relevé des classeurs BASE, GWW,
VAR4, VAR8 par ``tools/transcrire_fondation_grue.py``) ; étude : ``docs/beton/FONDATIONS_GRUE_ETUDE.md``.

  1. BASE DE DONNÉES : σ'b, μ (15 valeurs), σ adm des quatre classeurs, à l'égalité stricte (mêmes arrondis qu'Excel).
  2. SAISIES : les quatre exemples du moteur portent exactement les saisies des classeurs ; aucune cellule de saisie
     numérique d'une fiche n'est oubliée par le module.
  3. FICHES : chaque cellule CALCULÉE des quatre « Fiche de calcul » est retrouvée à 1e-9 (valeurs, signes ≤ / ≥,
     états OK / LIMITE / PAS OK, textes de zone, cas gouvernants) — ou figure dans la liste, courte et motivée, des
     libellés écrits comme des formules. Rien d'autre n'est laissé de côté.
  4. TABLEAUX de BASE et GWW : les 48 colonnes des quatre grands tableaux (lignes 23 à 35), deux colonnes entières
     de M(x) (211 lignes), les maximums et leurs cas, les feuilles CAS A / CAS B, la feuille de la semelle (101 lignes).
  5. STATIQUE : la poutre sur deux appuis (statique.py), qui ne sait rien des classeurs, redonne leurs réactions,
     leurs moments et leurs efforts tranchants — sur les exemples et sur des fondations tirées au sort ; les seuls
     écarts sont ceux que l'étude nomme (E8 : maximum lu tous les 5 cm ; E16 : moment pris sous la charge).
  6. FORMULES : chaque formule imprimée (note, fenêtres de calcul) se retrouve à 1e-9 à partir de ses seuls termes,
     et à l'arrondi d'affichage près avec les nombres tels qu'ils sont imprimés (kit des notes) ; et son ÉCRITURE
     (``ecriture.py``) : la chaîne IMPRIMÉE, relue caractère par caractère, vaut le gabarit ; chaque symbole y est écrit
     tel quel ; ni caractère invisible, ni tiret à la place du signe moins, ni parenthèses autour d'un symbole seul
     (11/11/2026 : un « × 10³ » s'imprimait « × 100 », « H_pied » s'écrivait « H_πed » — le résultat était juste,
     l'écriture fausse, et le contrôle d'alors ne relisait que le gabarit).
  7. ÉCARTS : ceux de l'étude, chiffrés, sur les quatre exemples ; ils apparaissent et disparaissent avec leur cause.
  8. MOTEUR : états (OK → ok, LIMITE → warn, PAS OK → nok, zones de τ), taux, vérification gouvernante, contrôles de
     géométrie de la fiche BASE, diagrammes (enveloppe = maximum des positions), armatures = saisies (l'export ne
     recalcule rien), JSON strict, entrée intacte, normalisation idempotente.
  9. DOMAINE : les refus et leurs messages.
 10. PURETÉ : bibliothèque standard seule (sous-processus où numpy, openpyxl, streamlit… sont refusés).
 11. MUTATIONS : le moteur rechargé depuis ses sources avec UNE formule changée est rejeté ; un témoin ne l'est pas.

    /usr/bin/python3.12 tests/test_fondation_grue_moteur.py
"""
import copy
import json
import math
import os
import random
import re
import subprocess
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import beton.fondation_grue as PAQUET  # noqa: E402
from beton.fondation_grue import classeur, donnees, ecarts, ecriture, entree, formules, moteur, statique  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres import formules as KIT  # noqa: E402

NOMS_MODULES = ("donnees", "entree", "statique", "classeur", "ecarts", "formules", "moteur")
with open(os.path.join(RACINE, "tests", "ref_fondation_grue.json"), encoding="utf-8") as _f:
    REF = json.load(_f)["classeurs"]
FICHE = "Fiche de calcul"
CLASSEURS = ("VAR4", "VAR8", "BASE", "GWW")
# les « formules » des fiches qui sont des LIBELLÉS commençant par « = » (rien à calculer)
LIBELLES = {"VAR4": {"B85"}, "VAR8": {"F50"},
            "BASE": {"B36", "B38", "B72", "F103", "F104", "F90", "F91", "F92", "F93", "F95", "F96", "F97"}}
LIBELLES["GWW"] = LIBELLES["BASE"]
LIGNES_TABLEAU = {24: "P1", 25: "P2", 26: "H4", 27: "RA", 28: "RB", 29: "RB_resp", 31: "V_max", 32: "V_central",
                  33: "M_max_eq", 34: "M_hor_eq", 35: "M_max"}

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


def proche(a, b, tol=1e-9):
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b or (a is None and b in (None, ""))
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def cellule(nom, adresse):
    """La valeur relevée d'une cellule « C34 » (fiche) ou « feuille!C34 »."""
    feuille, _, c = adresse.rpartition("!")
    src = REF[nom]["feuilles"][feuille or FICHE]
    return src["formules"].get(c, src["saisies"].get(c, "<absente>"))


def variantes(mods, n, graine=20261111):
    """Des fondations tirées au sort autour des quatre exemples (géométrie, charges, tolérances), dans le domaine."""
    rnd = random.Random(graine)
    E = mods["entree"]
    out = []
    while len(out) < n:
        f = E.exemple(rnd.choice(CLASSEURS))

        def fois(chemin, a, b):
            E.poser(f, chemin, round(E.lire(f, chemin) * rnd.uniform(a, b), 1))

        fois("chassis", 0.7, 1.4)
        if f["type"] == E.CADRE4:
            for c in ("poutre/b", "poutre/h", "pieux/D", "charges/P", "charges/H_s", "charges/M_s", "e"):
                fois(c, 0.6, 1.5)
        elif f["type"] == E.DIAGONALES8:
            for c in ("secondaire/b", "secondaire/h", "primaire/b", "primaire/h", "pieux/D", "charges/P", "e_secondaire",
                      "e_primaire", "charges/M_s"):
                fois(c, 0.7, 1.4)
        else:
            for c in ("support/b", "support/h", "semelle/b", "semelle/h", "dist_a", "dist_b", "e_support", "e_semelle",
                      "c_appui", "charges/H_s", "charges/M_s"):
                fois(c, 0.75, 1.3)
            for cas in f["charges"]["cas"]:
                for p in E.PIEDS:
                    cas[p] = round(cas[p] * rnd.uniform(0.5, 1.5), 0)
            f["beton"] = rnd.choice(mods["donnees"].BETONS)
        try:
            out.append(E.normaliser(f))
        except ValueError:
            continue
    return out


# ================================================================== les contrôles, rejouables sur un moteur mutant
def controles(mods) -> list:
    """Ce que ce fichier reproche aux modules ``mods`` (noms → modules) : liste vide = tout est retrouvé."""
    D, E, S, K, X, F, M = (mods[n] for n in NOMS_MODULES)
    r = []

    # 1. base de données
    for nom in CLASSEURS:
        f = REF[nom]["feuilles"]["Base de données"]["formules"]
        for i, b in enumerate(D.BETONS):
            col = "BCDEF"[i]
            if f[f"{col}7"] != D.sigma_b(b):
                r.append(f"{nom} σ'b {b}")
            for j, a in enumerate(D.ACIERS):
                if f[f"{col}{11 + j}"] != D.mu(b, a):
                    r.append(f"{nom} μ {b} {a}")
        for j, a in enumerate(D.ACIERS):
            if f[f"B{16 + j}"] != D.sigma_adm(a):
                r.append(f"{nom} σadm {a}")

    for nom in CLASSEURS:
        f = E.exemple(nom)
        t = f["type"]
        fiche = REF[nom]["feuilles"][FICHE]
        # 2. saisies
        for chemin, cel in E.SAISIES[t].items():
            v, w = E.lire(f, chemin), fiche["saisies"].get(cel)
            egal = (v == str(w).replace(".", ",") or v == w) if isinstance(v, str) else (w is not None and float(w) == float(v))
            if not egal:
                r.append(f"{nom} saisie {chemin} ({cel}) : {v!r} ≠ {w!r}")
        # 3. fiche et feuilles annexes : chaque cellule posée par le relevé
        try:
            rel = K.FICHES[t](E.normaliser(f))
        except Exception as err:  # noqa: BLE001 — un mutant peut casser le calcul : c'est un rejet
            r.append(f"{nom} : le relevé échoue ({type(err).__name__} : {err})")
            continue
        for cle, cel in rel.cellules.items():
            if not proche(rel[cle], cellule(nom, cel)):
                r.append(f"{nom} {cel} ({cle}) : {rel[cle]!r} ≠ {cellule(nom, cel)!r}")
        # 4. grands tableaux
        if t == E.SEMELLES8:
            for k, tab in rel.tableaux.items():
                for j, col in enumerate(tab["colonnes"]):
                    L = K._lettres(j)
                    if cellule(nom, f"{tab['feuille']}!{L}23") != col["libelle"]:
                        r.append(f"{nom} {k} {L}23 libellé")
                    if (cellule(nom, f"{tab['feuille']}!{L}30") == "SYMM.") != col["symetrique"]:
                        r.append(f"{nom} {k} {L}30 symétrie")
                    for lig, cle in LIGNES_TABLEAU.items():
                        if not proche(col[cle], cellule(nom, f"{tab['feuille']}!{L}{lig}")):
                            r.append(f"{nom} {k} {L}{lig} ({cle})")
                    if L in ("B", "AW"):
                        for i in range(K.N_X):
                            if not proche(col["M"][i], cellule(nom, f"{tab['feuille']}!{L}{36 + i}")):
                                r.append(f"{nom} {k} {L}{36 + i} M(x)")
                                break
            for i in range(K.N_T):
                if not proche(rel.M_semelle[i], cellule(nom, f"{K.FEUILLE_SEMELLE}!B{17 + i}")):
                    r.append(f"{nom} semelle B{17 + i} M(t)")
                    break

    # 5. statique indépendante = classeur, sur les exemples et des tirages
    for f in [E.normaliser(E.exemple(n)) for n in CLASSEURS] + variantes(mods, 60):
        t = f["type"]
        try:
            rel = K.FICHES[t](f)
        except Exception as err:  # noqa: BLE001
            r.append(f"relevé d'une variante : {type(err).__name__} : {err}")
            continue

        def un_pied(prefixe, e_cm, pp, travee):
            L, e, P = f["chassis"] / 100, e_cm / 100, f["charges"]["P"]
            p = S.poutre_un_pied(L, e, P, pp)
            vus = [("A.RA", p["A"].RA, rel[f"{prefixe}.cas_A.RA"]), ("A.RB", p["A"].RB, rel[f"{prefixe}.cas_A.RB"]),
                   ("A.Mmin", p["A"].M(e), rel[f"{prefixe}.cas_A.M_min"]),
                   ("A.V", -p["A"].V(e, gauche=True), rel[f"{prefixe}.cas_A.V_max"]),
                   ("B.RA", p["B"].RA, rel[f"{prefixe}.cas_B.RA"]), ("B.RB", p["B"].RB, rel[f"{prefixe}.cas_B.RB"]),
                   ("B.M", p["B"].M(0.0), rel[f"{prefixe}.cas_B.M_max"]), ("B.V", p["B"].V(-e), rel[f"{prefixe}.cas_B.V_max"])]
            if travee and L > 1.7 and e < 0.8:
                vus += [("A.Vt", max(abs(p["A"].V(0.8)), abs(p["A"].V(L - 0.8))), rel[f"{prefixe}.cas_A.V_travee"]),
                        ("B.Vt", max(abs(p["B"].V(-e + 0.8)), abs(p["B"].V(L - 0.8))), rel[f"{prefixe}.cas_B.V_travee"])]
            if e > 0 and p["A"].extremes()["M_min"]["valeur"] > rel[f"{prefixe}.cas_A.M_min"] + 1e-6:
                r.append(f"{prefixe} : M min exact au-dessus de celui du classeur")
            if p["B"].extremes()["M_max"]["valeur"] < rel[f"{prefixe}.cas_B.M_max"] - 1e-6:
                r.append(f"{prefixe} : M max exact sous celui du classeur")
            return [(prefixe + " " + n, a, b) for n, a, b in vus]

        def deux_pieux(prefixe, Lp, e_cm, P, pp, c):
            p = S.poutre_deux_pieux(Lp / 100, f["pieux"]["entraxe"] / 100, e_cm / 100, P, pp, c / 100)
            vus = [("A.RA", p["A"].RA, rel[f"{prefixe}.cas_A.RA"]), ("A.RB", p["A"].RB, rel[f"{prefixe}.cas_A.RB"]),
                   ("A.V", p["A"].V(p["A"].xa), rel[f"{prefixe}.cas_A.V_max"]),
                   ("B.RA", p["B"].RA, rel[f"{prefixe}.cas_B.RA"]),
                   ("B.M", p["B"].M(0.0), rel[f"{prefixe}.cas_B.M_max"]), ("B.V", p["B"].V(p["B"].xa), rel[f"{prefixe}.cas_B.V_max"])]
            exact = p["A"].extremes()["M_max"]["valeur"]
            if not (rel[f"{prefixe}.cas_A.M_max"] <= exact * (1 + 1e-9) and rel[f"{prefixe}.cas_A.M_max"] >= exact * (1 - 2e-3)):
                r.append(f"{prefixe} CAS A : M max classeur {rel[f'{prefixe}.cas_A.M_max']} loin de l'exact {exact}")
            return [(prefixe + " " + n, a, b) for n, a, b in vus]

        if t == E.CADRE4:
            paires = un_pied("poutre", f["e"], rel["pp.poutre"], True)
        elif t == E.DIAGONALES8:
            paires = un_pied("secondaire", f["e_secondaire"], rel["pp.secondaire"], False)
            paires += deux_pieux("primaire", f["primaire"]["L"], f["e_primaire"], rel["primaire.charge"], rel["pp.primaire"], 0.0)
        else:
            paires = deux_pieux("semelle", f["semelle"]["L"], f["e_semelle"], rel["semelle.charge"], rel["pp.semelle"],
                                f["c_appui"])
            Lc, a, b, e = f["chassis"] / 100, f["dist_a"] / 100, f["dist_b"] / 100, f["e_support"] / 100
            for k, tab in rel.tableaux.items():
                for col in tab["colonnes"][::5]:
                    q = S.poutre_support(Lc, a, b, e, k, col["P1"], col["P2"], rel["pp.support"])
                    x1, x2 = q.ponctuelles[0][0], q.ponctuelles[1][0]
                    paires += [(f"support {k} RA", q.RA, col["RA"]), (f"support {k} RB", q.RB, col["RB"]),
                               (f"support {k} Vc", max(q.V(x1), -q.V(x2, gauche=True)), col["V_central"]),
                               (f"support {k} V", max(q.V(q.xa), -q.V(q.xb, gauche=True)), col["V_max"])]
                    exact = q.extremes()["M_max"]["valeur"]
                    if not (col["M_max"] <= exact * (1 + 1e-9) and col["M_max"] >= exact - max(0.012 * exact, 0.6)):
                        r.append(f"support {k} : M max du tableau {col['M_max']} loin de l'exact {exact}")
                    if abs(q.RA + q.RB - col["P1"] - col["P2"] - rel["pp.support"] * tab["geometrie"]["L"]) > 1e-6:
                        r.append(f"support {k} : équilibre vertical")
        for n, a_, b_ in paires:
            if not proche(a_, b_, 1e-8):
                r.append(f"statique ≠ classeur ({t}) {n} : {a_} ≠ {b_}")

        # 6. chaque formule imprimée se retrouve à partir de ses termes
        try:
            res = M.verifier(f)
        except Exception as err:  # noqa: BLE001
            r.append(f"verifier d'une variante : {type(err).__name__} : {err}")
            continue
        for bloc in res["blocs"]:
            fs = (list(bloc["sollicitations"]) + [x for v in bloc["verifications"] for x in v["formules"]]
                  + [i["formule"] for i in bloc["informations"] if i.get("formule")])
            for fx in fs:
                env = {t_[0]: t_[2] for t_ in fx["termes"]}
                env.update(sqrt=math.sqrt, min=min, max=max, abs=abs, pi=math.pi)
                try:
                    v = eval(fx["gabarit"], {"__builtins__": {}}, env)  # noqa: S307 — gabarits internes
                except Exception as err:  # noqa: BLE001
                    r.append(f"formule {bloc['cle']} {fx['nom']} : {err}")
                    continue
                if fx["resultat"] is None or not proche(v, fx["resultat"]):
                    r.append(f"formule {bloc['cle']} « {fx['nom']} » : {v} ≠ {fx['resultat']}")
                if fx["cellule"] and fx["cellule"] in res["cellules"].values():
                    cle = next(c for c, cel in res["cellules"].items() if cel == fx["cellule"])
                    if not proche(res["valeurs"][cle], fx["resultat"]):
                        r.append(f"formule {fx['nom']} : son résultat n'est pas la cellule {fx['cellule']}")
        # 8. l'enveloppe couvre chaque position, les armatures sont les saisies
        for el in res["elements"]:
            mx = max(p["extremes"]["M_max"]["valeur"] for p in el["positions"])
            if el["extremes"]["M_max"]["valeur"] < mx - max(1e-6, 2e-3 * abs(mx)):
                r.append(f"enveloppe de {el['cle']} sous le maximum d'une position")
        for a_ in res["armatures"]:
            if a_["genre"] == "etriers" and a_["saisie"]:
                et = E.lire(f, a_["saisie"])
                if (a_["brins"], a_["d"], a_["pas"]) != (et["brins"], et["d"], et["pas"]):
                    r.append(f"armature {a_['poste']} ≠ saisie")
            if a_["genre"] == "barres":
                lit = E.lire(f, a_["saisie"])
                lits = lit if isinstance(lit, list) else [lit]
                if a_["groupes"] != [{"n": g["n"], "d": g["d"]} for g in lits if g["n"] > 0]:
                    r.append(f"armature {a_['poste']} ≠ saisie")

    # 7. écarts de l'étude, chiffrés
    def codes(nom, **modifs):
        f = E.exemple(nom)
        for chemin, v in modifs.items():
            E.poser(f, chemin.replace("__", "/"), v)
        return {e["code"]: e for e in M.verifier(f)["ecarts"]}

    try:
        e4 = codes("VAR4")
        if not (set(e4) == {"E1", "E16", "E6"} and proche(e4["E1"]["redemontre"]["diagonale"], 27.0468, 1e-4)
                and proche(e4["E1"]["redemontre"]["cote"], 33.5556, 1e-4) and proche(e4["E1"]["classeur"], 23.0)
                and proche(e4["E16"]["redemontre"], 76.3475, 1e-5) and proche(e4["E16"]["classeur"], 72.6033, 1e-5)):
            r.append(f"écarts VAR4 : {sorted(e4)}")
        e8 = codes("VAR8")
        if not (set(e8) == {"E1", "E4", "E5"} and proche(e8["E1"]["redemontre"]["cote"], 45.9167, 1e-4)
                and proche(e8["E5"]["classeur"], -6.41026, 1e-5) and proche(e8["E4"]["classeur"], 516.8487, 1e-6)):
            r.append(f"écarts VAR8 : {sorted(e8)}")
        eb, eg = codes("BASE"), codes("GWW")
        if not (set(eb) == {"E3", "E7", "E9", "E15"} and proche(eb["E3"]["redemontre"], 61.4827, 1e-5)
                and proche(eb["E3"]["classeur"], 60.2171, 1e-5) and set(eg) == {"E3", "E7", "E9"}):
            r.append(f"écarts BASE / GWW : {sorted(eb)} / {sorted(eg)}")
        if "E2" in e4 or "E2" not in codes("VAR4", beton="C25/30") or "E2" not in codes("BASE", beton="C35/45"):
            r.append("écart E2 (seuil 0,56) : n'apparaît pas avec le béton")
        if "E6" in codes("VAR4", poutre__part_V=100.0) or "E15" in codes("BASE", support__etriers_centre__pas=600.0):
            r.append("écarts E6 / E15 : ne disparaissent pas avec leur cause")
        if "E10" not in codes("BASE", dist_a=60.0) or "E10" in codes("BASE", dist_a=60.0, cote_b="paroi"):
            r.append("écart E10 (semelle du côté b) : quatre semelles et a ≠ b seulement")
        if "E8" not in codes("BASE", dist_a=47.0) or "E8" in eb:
            r.append("écart E8 (maximum lu tous les 5 cm) : quand le pied n'est pas sur le pas du tableau")
        if "E16" in e8 or "E5" in e4:
            r.append("écarts E16 / E5 : signalés sans raison")
    except Exception as err:  # noqa: BLE001
        r.append(f"écarts : {type(err).__name__} : {err}")

    # 8. états, taux, gouvernant, contrôles de géométrie
    try:
        base, gww, v4 = M.verifier(E.exemple("BASE")), M.verifier(E.exemple("GWW")), M.verifier(E.exemple("VAR4"))
        etats = {v["cle"]: (v["etat_classeur"], v["etat"]) for v in base["verifications"]}
        if etats["pieux"] != ("LIMITE", "warn") or etats["support.inf"] != ("OK", "ok"):
            r.append(f"états BASE : {etats['pieux']}, {etats['support.inf']}")
        if base["statut"] != "warn" or v4["statut"] != "ok" or gww["statut"] != "warn":
            r.append(f"statuts : {base['statut']}, {v4['statut']}, {gww['statut']}")
        if base["gouvernant"]["cle"] != "support.etriers_centre" or not proche(base["taux_max"], 750 / 753.170028, 1e-8):
            r.append(f"gouvernant BASE : {base['gouvernant']}")
        if [c["cle"] for c in base["controles"] if not c["ok"]] != ["h_semelle", "entraxe", "pas_centre"] or any(
                not c["ok"] for c in gww["controles"]):
            r.append("contrôles de géométrie de la fiche BASE")
        f = E.exemple("VAR4")
        f["perfos"]["d"] = 6
        nok = M.verifier(f)
        if nok["statut"] != "nok" or next(v for v in nok["verifications"] if v["cle"] == "perfos")["etat"] != "nok":
            r.append("PAS OK → nok (perfos Ø6)")
        f = E.exemple("BASE")
        f["support"]["b"] = 60.0
        if next(i for i in M.verifier(f)["blocs"][0]["informations"] if i.get("zone"))["etat"] != "nok":
            r.append("τ au-delà de la zone 3 → nok")
        # la bande « LIMITE » : à 5 % sous le besoin (armatures), à 5 % au-dessus du pas calculé (étriers)
        def etat_de(cle, **modifs):
            f_ = E.exemple("BASE")
            for chemin, v in modifs.items():
                E.poser(f_, chemin.replace("__", "/"), v)
            v_ = next(x for x in M.verifier(f_)["verifications"] if x["cle"] == cle)
            return v_["etat_classeur"], v_["etat"]

        limites = (etat_de("support.inf", support__inf__0__n=10, support__inf__0__d=16),      # 2 011 mm² pour 2 081
                   etat_de("support.inf", support__inf__0__n=6, support__inf__0__d=20),       # 1 885 mm² : sous 95 %
                   etat_de("support.etriers_extremites", support__etriers_extremites__pas=155.0),   # 150,4 calculé
                   etat_de("support.etriers_extremites", support__etriers_extremites__pas=160.0))
        if limites != (("LIMITE", "warn"), ("PAS OK", "nok"), ("LIMITE", "warn"), ("PAS OK", "nok")):
            r.append(f"bande LIMITE (5 %) : {limites}")
        zone = next(i for i in gww["blocs"][0]["informations"] if i.get("zone"))
        if (zone["zone"], zone["etat"]) != ("ZONE 3", "warn"):
            r.append(f"τ zone 3 → warn : {zone['zone']}, {zone['etat']}")
    except Exception as err:  # noqa: BLE001
        r.append(f"moteur : {type(err).__name__} : {err}")

    # 9. domaine
    def refuse(nom, mot, **modifs):
        f = E.exemple(nom)
        for chemin, v in modifs.items():
            E.poser(f, chemin.replace("__", "/"), v)
        try:
            M.verifier(f)
        except ValueError as err:
            return mot in str(err)
        return False

    for nom, mot, modifs in (
            ("VAR4", "excentricité", {"e": 450.0}), ("VAR4", "béton", {"beton": "C12/15"}),
            ("VAR4", "Ø parmi", {"perfos__d": 13}), ("VAR4", "> 0", {"chassis": 0.0}),
            ("VAR8", "moitié de l'entredistance", {"e_primaire": 80.0}), ("VAR8", "longueur doit couvrir", {"primaire__L": 150.0}),
            ("BASE", "sortirait de la portée", {"e_support": 30.0}), ("BASE", "s/2 − e − c/2", {"c_appui": 100.0}),
            ("BASE", "au moins une barre", {"support__inf__0__n": 0}), ("BASE", "hauteur utile", {"semelle__h": 5.0}),
            ("BASE", "paroi", {"cote_b": "mur"}), ("BASE", "entier", {"support__etriers_centre__brins": 3.5})):
        if not refuse(nom, mot, **modifs):
            r.append(f"refus manquant : {nom} {modifs} (« {mot} »)")
    try:
        f = E.exemple("BASE")
        f["inconnu"] = 1
        E.normaliser(f)
        r.append("champ inconnu admis")
    except ValueError:
        pass
    return r


# ================================================================== le moteur intact
print("=== 1 à 9. Fidélité aux classeurs, statique, formules, écarts, états, domaine ===")
MODS = {n: getattr(PAQUET, n) for n in NOMS_MODULES}
REPROCHES = controles(MODS)
chk("les quatre classeurs sont retrouvés, la statique les redémontre, les formules se relisent, les écarts sont ceux de l'étude",
    not REPROCHES, REPROCHES[:8])

print("\n=== 3. Couverture : aucune cellule calculée d'une fiche n'est laissée de côté ===")
total = 0
for nom in CLASSEURS:
    f = entree.normaliser(entree.exemple(nom))
    rel = classeur.FICHES[f["type"]](f)
    vues = {c for c in rel.cellules.values() if "!" not in c}
    fiche = REF[nom]["feuilles"][FICHE]
    manquent = sorted(set(fiche["formules"]) - vues - LIBELLES[nom])
    total += len(vues)
    chk(f"{nom} : les {len(fiche['formules'])} formules de la fiche sont reprises ({len(vues)} cellules comparées) ou sont "
        f"des libellés ({len(LIBELLES[nom])})", not manquent, manquent)
    chk(f"{nom} : les libellés écartés sont bien des textes", all(isinstance(fiche["formules"][c], str) for c in LIBELLES[nom]))
    numeriques = {c for c, v in fiche["saisies"].items() if isinstance(v, (int, float)) and c[0] in "CG"}
    oubliees = sorted(numeriques - set(entree.SAISIES[f["type"]].values()))
    chk(f"{nom} : toute saisie numérique de la fiche (colonnes C et G) est un champ du module "
        f"({len(entree.SAISIES[f['type']])} champs)", not oubliees, oubliees)
    for feuille, src in REF[nom]["feuilles"].items():
        if feuille in (FICHE, "Base de données"):
            continue
        posees = {c.rpartition("!")[2] for c in rel.cellules.values() if c.startswith(feuille + "!")}
        if feuille.startswith("R+V+M") and feuille[-2] == ".":
            attendues = {c for c in src["formules"] if c[0] in "BD" and c[1:].isdigit() and 5 <= int(c[1:]) <= 19}
        elif feuille == classeur.FEUILLE_SEMELLE:
            attendues = {c for c in src["formules"] if int(c[1:]) <= 16}
        else:
            attendues = {c for c in src["formules"] if c[0] in "BD" and int(c[1:]) <= 12}
        chk(f"{nom} « {feuille} » : données et résultats repris ({len(attendues)} cellules)", attendues <= posees,
            sorted(attendues - posees)[:10])
chk("au total, 541 cellules calculées des quatre fiches sont comparées (93 + 118 + 165 + 165)", total == 541, total)

print("\n=== 6. Les formules telles qu'elles sont IMPRIMÉES (ecriture.py, kit des notes) ===")


def formules_de(res):
    """Toutes les formules qu'un résultat fait imprimer : sollicitations, vérifications, cisaillement."""
    for bloc in res["blocs"]:
        yield from bloc["sollicitations"]
        for v in bloc["verifications"]:
            yield from v["formules"]
        for i in bloc["informations"]:
            if i.get("formule"):
                yield i["formule"]


# les quatre classeurs, et des fondations tirées au sort autour d'eux (autres nombres, mêmes écritures)
RESULTATS_ECRITURE = [moteur.verifier(f) for f in [entree.exemple(n) for n in CLASSEURS] + variantes(MODS, 24, graine=7)]
# ce qui doit se LIRE, relu à la main sur les classeurs : (classeur, début du nom, cellule) → la formule entière ou sa fin
ATTENDUES = (
    ("BASE", "C145", "/(V_max·part de V (%)/100) = 6 × π × (12/2)² × 333,33 × 658/(989,56 × 10³ × 100/100) = 150 mm"),
    ("BASE", "C142", "τ = V_max·part de V (%)/100/(0,75·b·d) = 989,56 × 100/100 × 10³/(0,75 × 1200 × 688) = 1,6 N/mm²"),
    ("BASE", "C257", "Ω_perfos = H_pied/150 = 49,439 × 10³/150 = 330 mm²"),
    ("BASE", "C189", "·(d − enrobage)/V_max = 4 × π × (12/2)² × 333,33 × (688 − 40)/(602,17 × 10³) = 162 mm"),
    ("VAR4", "C37", "M_min (CAS A) = −P·e − pp·e²/2 = −474 × 0,15 − 11,25 × 0,15²/2 = −71,2 kN·m"),
    ("GWW", "C94", "= R_A·x − pp·x²/2 − P_1·(x − a') = 1032,4 × 1,15 − 24,5 × 1,15²/2 − 1004 × (1,15 − 0,45) = 468,3 kN·m"),
    ("GWW", "C128", "Ω_inf = M_max,éq/(σ_adm·0,9·d) = 470,5 × 10⁶/(333,33 × 0,9 × 638) = 2458 mm²"),
)
_SYMBOLE_SEUL = re.compile(r"/\(([^()\s·×+−/*-]+)\)")
_PI_COLLE = re.compile(r"[^\W\d_]π|π[^\W\d_]")


def fautes_ecriture(W) -> list:
    """Ce que l'écriture ``W`` (le module ``ecriture``, ou un mutant) imprime de faux ; liste vide = rien."""
    r, n = [], 0
    for res in RESULTATS_ECRITURE:
        for fx in formules_de(res):
            n += 1
            parts, resultat = W.morceaux(fx)
            if not parts or parts[0] != fx["nom"] or not resultat or any(not p.strip() for p in parts):
                r.append(f"morceaux vides ou sans nom : {fx['nom']} → {parts} = {resultat}")
            if not fx["gabarit"]:
                continue
            lit, num = W.litteral(fx), W.numerique(fx)
            ici = f"{fx['nom']} = {lit} = {num}"
            # 1. la chaîne IMPRIMÉE, relue, vaut le gabarit évalué avec les mêmes nombres imprimés
            try:
                relu, attendu = W.relire(num), KIT.evaluer(W.sub(fx))
                if not math.isclose(relu, attendu, rel_tol=1e-9, abs_tol=1e-9):
                    r.append(f"la chaîne imprimée ne vaut pas son gabarit ({relu:.6g} ≠ {attendu:.6g}) : {ici}")
            except Exception as err:  # noqa: BLE001
                r.append(f"chaîne imprimée illisible ({type(err).__name__}: {err}) : {ici}")
            # 2. … et redonne le résultat du classeur, à l'arrondi d'affichage près
            ok, v = KIT.verifier(W.sub(fx), tol=0.02)
            if not ok:
                r.append(f"les nombres imprimés ne redonnent pas le résultat ({v} ≠ {fx['resultat']}) : {ici}")
            # 3. chaque symbole est écrit tel quel dans la forme littérale
            ids = set(KIT._IDENT.findall(fx["gabarit"]))
            for ident, symbole, _ in fx["termes"]:
                if ident in ids and symbole not in lit:
                    r.append(f"symbole abîmé : « {symbole} » introuvable dans « {lit} » ({fx['nom']})")
            # 4. la typographie : rien d'invisible, π seul, le vrai signe moins, la virgule décimale, pas de parenthèses
            #    autour d'un symbole seul
            for p in parts[1:] + [lit, num]:
                if "\u2063" in p or _PI_COLLE.search(p) or "-" in p or _SYMBOLE_SEUL.search(p) or re.search(r"\d\.\d", p):
                    r.append(f"typographie : « {p} » ({fx['nom']})".replace("\u2063", "<invisible>"))
                    break
            # 5. l'écriture des gabarits : la virgule d'un argument est suivie d'un espace (la virgule décimale ne l'est pas)
            if re.search(r",(?! )", fx["gabarit"]):
                r.append(f"gabarit : une virgule d'argument sans espace ({fx['gabarit']})")
    if n < 1000:
        r.append(f"seulement {n} formules relues")
    # 6. ce qui doit se lire, à la lettre
    for nom, cellule_, attendu in ATTENDUES:
        ecrites = [" = ".join(W.morceaux(fx)[0] + [W.morceaux(fx)[1]]) for fx in formules_de(RESULTATS_ECRITURE[CLASSEURS.index(nom)])
                   if fx.get("cellule") == cellule_]
        if not ecrites or not any(e.endswith(attendu) for e in ecrites):
            r.append(f"écriture attendue ({nom} {cellule_}) : « {attendu} » — écrit : {ecrites[:1]}")
    return r


FAUTES = fautes_ecriture(ecriture)
N_FORMULES = sum(1 for res in RESULTATS_ECRITURE for _ in formules_de(res))
chk(f"les {N_FORMULES} formules de {len(RESULTATS_ECRITURE)} fondations (les quatre classeurs et 24 variantes) : la chaîne IMPRIMÉE, relue, "
    "vaut son gabarit ; les nombres imprimés redonnent le résultat à 2 % près ; chaque symbole est écrit tel quel ; ni caractère "
    "invisible, ni « π » collé à une lettre, ni tiret pour un moins, ni point décimal, ni « /(symbole) »", not FAUTES, FAUTES[:4])
chk("relire : « 989,6 × 10³ × 100/100 » vaut 989 600, « √(4 × 10⁶/(2 × 0,5)) » vaut 2 000, « max(|−3|, 2,5) » vaut 3",
    (ecriture.relire("989,6 × 10³ × 100/100"), ecriture.relire("√(4 × 10⁶/(2 × 0,5))"), ecriture.relire("max(|−3|, 2,5)")) == (989600.0, 2000.0, 3.0))
for nom_, cellule_, attendu_ in ATTENDUES:
    chk(f"{nom_}, cellule {cellule_} : « {attendu_} »", not any(x.startswith(f"écriture attendue ({nom_} {cellule_})") for x in FAUTES),
        [x for x in FAUTES if cellule_ in x][:1])
chk("un terme s'imprime à CINQ chiffres significatifs : 1032,41 → 1032,4 ; 989,56 ; 333,33 ; 0,1708 ; 12 ; 0,45 (le calcul refait à la "
    "main avec les nombres imprimés retombe sur le résultat imprimé)",
    [KIT.terme(("", v, ecriture.decimales(v))) for v in (1032.4108, 989.5615, 333.33, 0.1708, 12.0, 0.45, 468.2718, 0.0)]
    == ["1032,4", "989,56", "333,33", "0,1708", "12", "0,45", "468,27", "0"])
res4 = moteur.verifier(entree.exemple("VAR4"))
h = next(x for x in res4["blocs"][3]["sollicitations"] if x["nom"] == "H_pied")
texte = " = ".join(ecriture.morceaux(h)[0] + [ecriture.morceaux(h)[1]])
chk("l'effort horizontal des variantes s'imprime comme le classeur l'écrit (M/2/L/100, L en cm)",
    "200/2/450/100" in texte and texte.endswith("23 kN"), texte)

print("\n=== 8. Sortie : JSON strict, entrée intacte, normalisation idempotente ===")
for nom in CLASSEURS:
    f0 = entree.exemple(nom)
    f1 = copy.deepcopy(f0)
    res = moteur.verifier(f1)
    chk(f"{nom} : la fondation reçue n'est pas modifiée", f1 == f0)
    try:
        js = json.dumps(res, allow_nan=False)
        chk(f"{nom} : la sortie est du JSON strict ({len(js) // 1024} Ko)", json.loads(js)["statut"] == res["statut"])
    except ValueError as err:
        chk(f"{nom} : la sortie est du JSON strict", False, err)
    n1 = entree.normaliser(f0)
    chk(f"{nom} : normaliser(normaliser(f)) = normaliser(f) ; verifier la rend dans « fondation »",
        entree.normaliser(n1) == n1 and res["fondation"] == n1)
for t in entree.TYPES:
    f = entree.nouvelle_fondation(t)
    chk(f"une fondation neuve « {t} » se calcule et ne porte ni projet ni grue",
        moteur.verifier(f)["statut"] in ("ok", "warn") and f["projet"] == "" and f["grue"] == "")
chk("« C45/50 » (le libellé du classeur) est relu comme C40/50",
    entree.normaliser(dict(entree.exemple("VAR4"), beton="C45/50"))["beton"] == "C40/50")

print("\n=== 10. Pureté : bibliothèque standard seule ===")
CODE = r"""
import sys, json
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split('.')[0] in ('numpy', 'openpyxl', 'streamlit', 'pandas', 'scipy', 'matplotlib', 'reportlab', 'PIL', 'fitz'):
            raise ImportError('refusé : ' + nom)
sys.meta_path.insert(0, Refus())
sys.path.insert(0, %r)
from beton.fondation_grue import moteur, entree
print(json.dumps({n: [moteur.verifier(entree.exemple(n))['taux_max'], moteur.verifier(entree.exemple(n))['statut']] for n in ('VAR4', 'VAR8', 'BASE', 'GWW')}))
""" % RACINE
attendu = {n: [moteur.verifier(entree.exemple(n))["taux_max"], moteur.verifier(entree.exemple(n))["statut"]] for n in CLASSEURS}
for exe in ("/usr/bin/python3.12", "/usr/bin/python3.13"):
    if not os.path.exists(exe):
        continue
    p = subprocess.run([exe, "-c", CODE], capture_output=True, text=True, timeout=120)
    chk(f"{os.path.basename(exe)} : le moteur se charge et calcule sans numpy, openpyxl, streamlit… — mêmes résultats",
        p.returncode == 0 and json.loads(p.stdout or "{}") == attendu, (p.stderr or p.stdout)[-400:])

print("\n=== 11. Mutations ===")


def lire_source(n):
    with open(os.path.join(RACINE, "beton", "fondation_grue", n + ".py"), encoding="utf-8") as f_:
        return f_.read()


def charger_mutant(mutations):
    """Les modules rechargés depuis leurs SOURCES après [(module, avant, après)] ; « avant » doit y figurer UNE fois."""
    sauve = {n: sys.modules[f"beton.fondation_grue.{n}"] for n in NOMS_MODULES}
    mods = {}
    try:
        for n in NOMS_MODULES:
            src = lire_source(n)
            for module, avant, apres in mutations:
                if module == n:
                    if src.count(avant) != 1:
                        raise AssertionError(f"mutation périmée : « {avant} » ({src.count(avant)} fois dans {n}.py)")
                    src = src.replace(avant, apres)
            m = types.ModuleType(f"beton.fondation_grue.{n}")
            m.__package__, m.__file__ = "beton.fondation_grue", os.path.join(RACINE, "beton", "fondation_grue", n + ".py")
            sys.modules[m.__name__] = m
            setattr(PAQUET, n, m)
            exec(compile(src, m.__file__, "exec"), m.__dict__)
            mods[n] = m
    finally:
        sys.modules.update({f"beton.fondation_grue.{n}": sauve[n] for n in NOMS_MODULES})
        for n in NOMS_MODULES:
            setattr(PAQUET, n, sauve[n])
    return mods


MUTATIONS = (
    ("σ'b arrondi à 3 chiffres", [("donnees", "RWK[nom_beton(beton)] / DIVISEUR_BETON, 2)", "RWK[nom_beton(beton)] / DIVISEUR_BETON, 3)")]),
    ("arrondi « au pair » de Python au lieu de celui d'Excel", [("donnees", "rounding=ROUND_HALF_UP", "rounding='ROUND_DOWN'")]),
    ("μ sans le terme 1 − k/3", [("donnees", "* (1 - 1 / 3 * (15 * sb / (ss + (15 * sb)))) / 2, 4)", "/ 2, 4)")]),
    ("VAR4 CAS A : RA sans le poids propre", [("classeur", '(C24 * C10 / 100 + C21 * (C10 / 100) ** 2 / 2) / ((C10 - C29) / 100))\n    C35', '(C24 * C10 / 100) / ((C10 - C29) / 100))\n    C35')]),
    ("VAR4 : hauteur utile h − 40", [("classeur", 'r.pose("section.poutre.d", "G53", C12 * 10 - 50)', 'r.pose("section.poutre.d", "G53", C12 * 10 - 40)')]),
    ("VAR4 : Ω sup sur d et non d − 50", [("classeur", '"C59", abs(C45) * 10 ** 6 / (G16 * 0.9 * (G53 - 50)))', '"C59", abs(C45) * 10 ** 6 / (G16 * 0.9 * (G53)))')]),
    ("variantes : l'effort horizontal « corrigé » (M/2/L en m)", [("classeur", "Hs / 3 + Ms / 2 / L / 100))", "Hs / 3 + Ms / 2 / (L / 100)))")]),
    ("VAR8 primaire : 25 % au lieu de 40 % en haut", [("classeur", '"C99", 0.4 * C95)', '"C99", 0.25 * C95)')]),
    ("VAR8 secondaire : étriers sur 100 % de V", [("classeur", "G16 * G68 * 0.9 / (C46 * 10 ** 3 * G79 / 100))", "G16 * G68 * 0.9 / (C46 * 10 ** 3))")]),
    ("BASE : RB respectif pris dans la même colonne", [("classeur", 'c["RB_resp"] = cols[groupe + RESPECTIVE[j % 8]]["RB"]', 'c["RB_resp"] = cols[j]["RB"]')]),
    ("BASE : M(x) lu tous les 10 cm", [("classeur", "x = i / 20\n            if x > B8:", "x = (i // 2) / 10\n            if x > B8:")]),
    ("BASE : M hor éq sans le rapport h/b", [("classeur", "((B6 + B5) + B6) * B7 * H4 / B8) * hb * 100 / 125", "((B6 + B5) + B6) * B7 * H4 / B8) * 100 / 125")]),
    ("BASE : sous-cas B.b avec a + e", [("classeur", '"B.b": ((a - e) / 100, (b + e) / 100)}', '"B.b": ((a + e) / 100, (b + e) / 100)}')]),
    ("BASE : cas hors service chargés avec H en service", [("classeur", '(ch["H_hs"] if cas["etat"] == "hors" else ch["H_s"]) / 4', 'ch["H_s"] / 4')]),
    ("BASE : semelle CAS B sans le terme P·c/8", [("classeur", "- C44 * ((C24 / 2) / 100) ** 2 / 2 - C104 * C102 / 100 / 8)", "- C44 * ((C24 / 2) / 100) ** 2 / 2)")]),
    ("BASE : pieu vérifié sur R max (l'écart E3 « corrigé » en silence)", [("classeur", 'r.pose("pieux.sollicitation_T", "C222", C121 / 10)', 'r.pose("pieux.sollicitation_T", "C222", r["semelle.dim.R_max"] / 10)')]),
    ("BASE : étriers centraux avec la hauteur d'étrier (E7 « corrigé »)", [("classeur", "G15 * (G126 - C15) / (C97 * 10 ** 3 * G143 / 100))", "G15 * (G149) / (C97 * 10 ** 3 * G143 / 100))")]),
    ("BASE : LIMITE à 10 %", [("classeur", "MARGE_LIMITE = 0.95", "MARGE_LIMITE = 0.90")]),
    ("BASE : marge du pieu 1,00", [("classeur", "MARGES_PIEU = (1.1, 1.05)", "MARGES_PIEU = (1.0, 1.0)")]),
    ("BASE : zone 3 à 0,064·R'wk", [("classeur", "z1, z2, z3 = 0.016 * rwk, 0.032 * rwk, 0.048 * rwk", "z1, z2, z3 = 0.016 * rwk, 0.032 * rwk, 0.064 * rwk")]),
    ("BASE : effort horizontal avec le côté et non la diagonale", [("classeur", "G253 / (2 * C34 / math.sin(math.radians(45)) / 100)))", "G253 / (2 * C34 / 100)))")]),
    ("renfort au pieu : p2b sur le diamètre entier", [("classeur", 'cel["p2b"], P / (Dp / 2 / 100))', 'cel["p2b"], P / (Dp / 100))')]),
    ("statique : réaction RB sans les charges réparties", [("statique", "self.RB = moment_a / (xb - xa)", "self.RB = sum(p * (x - xa) for x, p in self.ponctuelles) / (xb - xa)")]),
    ("statique : M sans le poids propre", [("statique", "m -= q * (max(x - a, 0.0) ** 2 - max(x - b, 0.0) ** 2) / 2", "m -= 0.0")]),
    ("statique : CAS A de la semelle sans décalage de la charge", [("statique", "pa, ra = charge(-e / 2)", "pa, ra = charge(0.0)")]),
    ("statique : sous-cas B.a avec le châssis non décalé", [("statique", "[(a + e, P1), (a + e + Lc, P2)]", "[(a, P1), (a + Lc, P2)]")]),
    ("écart E1 tu", [("ecarts", "if not (_differe(H, diagonale) or _differe(H, cote)):", "if True:")]),
    ("écart E3 tu", [("ecarts", "if _differe(V, R):", "if False:")]),
    ("écart E16 tu", [("ecarts", "if exact[\"valeur\"] <= M * (1 + 1e-6):", "if True:")]),
    ("écart E2 signalé en C30/37", [("ecarts", "if not _differe(seuil, D.TAU_SEUIL_EN_DUR):\n        return []", "if False:\n        return []")]),
    ("LIMITE tenu pour conforme", [("moteur", "ETATS = {K.OK: OK, K.LIMITE: WARN,", "ETATS = {K.OK: OK, K.LIMITE: OK,")]),
    ("zone 3 de τ tenue pour verte", [("moteur", '"ZONE 3": WARN,', '"ZONE 3": OK,')]),
    ("taux inversé", [("formules", "taux = valeur / limite", "taux = limite / valeur")]),
    ("formule imprimée de Ω avec 0,85·d", [("formules", '"M*1e6/(sa*0.9*d)"', '"M*1e6/(sa*0.85*d)"')]),
    ("armature exportée : le Ø des étriers remplacé", [("moteur", '"genre": "etriers", "brins": et["brins"], "d": et["d"]', '"genre": "etriers", "brins": et["brins"], "d": 10')]),
    ("contrôle de géométrie : entraxe ≥ 2 Ø", [("moteur", 'not pi_["entraxe"] < pi_["D"] * 3', 'not pi_["entraxe"] < pi_["D"] * 2')]),
    ("refus levé : e ≥ a admis", [("entree", '_exiger(out["dist_a"] - out["e_support"] >= 0 and out["dist_b"] - out["e_support"] >= 0,', '_exiger(True or out["dist_a"] - out["e_support"] >= 0,')]),
)
TEMOIN = [("classeur", "# --- CAS A : pieu rentré de e", "# --- CAS A (témoin) : pieu rentré de e")]
survivants, comptes = [], []
for nom, muts in MUTATIONS:
    try:
        rej = controles(charger_mutant(muts))
    except AssertionError as err:
        survivants.append(f"{nom} : {err}")
        continue
    except Exception as err:  # noqa: BLE001 — un mutant qui ne se charge plus est rejeté
        rej = [f"{type(err).__name__}"]
    comptes.append(len(rej))
    if not rej:
        survivants.append(nom)
chk(f"{len(MUTATIONS)} mutations (une formule ou une règle chacune : base de données, fiches VAR4, VAR8, BASE, tableaux, "
    f"statique, écarts, états, formules imprimées, armatures, domaine) : TOUTES rejetées "
    f"(de {min(comptes or [0])} à {max(comptes or [0])} reproches)", not survivants, survivants)
chk("la mutation témoin (un commentaire) n'est rejetée par aucun contrôle", not controles(charger_mutant(TEMOIN)))
chk("après les mutations, le moteur chargé est intact", not controles({n: getattr(PAQUET, n) for n in NOMS_MODULES}))

print("\n=== 11 bis. Mutations de l'écriture (le kit des substitutions, ecriture.py) ===")


def ecriture_mutante(mut_kit=(), mut_ecriture=()):
    """Le module ``ecriture`` rechargé depuis ses sources avec [(avant, après)], sur un KIT rechargé de même : rien
    n'est changé dans les modules chargés."""
    def charger(nom, fichier, paquet, mutations):
        with open(fichier, encoding="utf-8") as f_:
            src = f_.read()
        for avant, apres in mutations:
            if src.count(avant) != 1:
                raise AssertionError(f"mutation périmée : « {avant} » ({src.count(avant)} fois dans {os.path.basename(fichier)})")
            src = src.replace(avant, apres)
        m = types.ModuleType(nom)
        m.__package__, m.__file__ = paquet, fichier
        sys.modules[nom] = m                      # (une dataclass cherche son module dans sys.modules)
        try:
            exec(compile(src, fichier, "exec"), m.__dict__)
        finally:
            del sys.modules[nom]
        return m

    kit = charger("kit_mutant", KIT.__file__, KIT.__package__, mut_kit)
    w = charger("ecriture_mutante", ecriture.__file__, "beton.fondation_grue", mut_ecriture)
    w.KIT = kit
    return w


MUTATIONS_ECRITURE = (
    ("kit : le jeton du facteur 1000 porte une lettre (un terme « k » prend sa place : « × 10³ » imprimé « × 100 »)",
     [('_KILO = "\\u2063\\u2063"', '_KILO = "\\u2063k\\u2063"')], []),
    ("kit : « pi » remplacé partout (« H_pied » imprimé « H_πed »)", [('expr = _PI.sub("π", expr)', 'expr = expr.replace("pi", "π")')], []),
    ("kit : le moins en tête d'expression reste un tiret", [('    expr = _MOINS_UNAIRE.sub(lambda m: m.group(1) + "−", expr)\n', "")], []),
    ("kit : la virgule décimale n'est plus posée (« 0.45 » imprimé)", [('expr = re.sub(r"(?<![A-Za-z])(\\d+)\\.(\\d+)", r"\\1,\\2", expr)', "expr = expr")], []),
    ("ecriture : les parenthèses autour d'un symbole seul restent (« /(V_max) »)", [], [('return _PARENTHESES_INUTILES.sub(r"/\\1", expression)', "return expression")]),
    ("ecriture : relire lit « 10³ » comme 100", [], [('t.replace("10³", "1000")', 't.replace("10³", "100")')]),
    ("ecriture : quatre chiffres significatifs (« 1032 × 1,15 − … = 468,3 » ne se refait plus à la main)", [],
     [("return max(0, min(5, 4 - int(math.floor(math.log10(abs(v))))))", "return max(0, min(4, 3 - int(math.floor(math.log10(abs(v))))))")]),
    ("kit : les décimales données pour un terme sont ignorées", [("    return fmt(v[1], v[2] if len(v) > 2 else None)", "    return fmt(v[1])")], []),
)
survivants_e = []
for nom, mk, me in MUTATIONS_ECRITURE:
    try:
        rej = fautes_ecriture(ecriture_mutante(mk, me))
    except AssertionError as err:
        survivants_e.append(f"{nom} : {err}")
        continue
    except Exception as err:  # noqa: BLE001
        rej = [type(err).__name__]
    if not rej:
        survivants_e.append(nom)
chk(f"{len(MUTATIONS_ECRITURE)} mutations de l'écriture (le jeton du facteur 1000, la constante π, le signe moins, la virgule décimale, les "
    "parenthèses, la relecture, les chiffres significatifs) : TOUTES rejetées", not survivants_e, survivants_e)
chk("le kit et l'écriture rechargés SANS mutation n'impriment rien de faux (le témoin)", not fautes_ecriture(ecriture_mutante()))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)
