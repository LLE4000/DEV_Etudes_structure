# -*- coding: utf-8 -*-
"""Barres en acier — L'ÉCRITURE DES FORMULES (acier/barre/ecriture.py) : « symbole = formule = valeurs = résultat », la
même dans la fenêtre de calcul de l'écran et dans la note PDF. docs/acier/BARRES_ACIER.md § 9.

Ce que la suite prouve :
  1. LES NOMBRES — un entier en entier, cinq chiffres significatifs sinon, le vrai signe moins, la virgule.
  2. DES FORMULES ÉCRITES À LA MAIN — la chaîne exacte de formules connues (flambement, charge critique en kN, unités
     × 10³ et / 10⁵, valeur négative, symbole composé, exposants, coefficient partiel, taux d'une vérification) ; la
     relecture, jugée elle-même.
  3. TOUS LES GABARITS — chacun, sur un bloc de valeurs quelconques : la formule en valeurs ne garde aucun identifiant,
     la formule en symboles aucun facteur d'unités ni opérateur de programme, et la relecture redonne le résultat.
  4. LE MOTEUR — toutes les formules de barres réelles (efforts imposés sur sept profilés, barres chargées) : le
     calcul refait à la main avec les nombres IMPRIMÉS redonne le résultat IMPRIMÉ.
  5. MUTATIONS — chaque altération plausible du texte de l'écriture fait rougir un contrôle.
  6. PURETÉ.

    /usr/bin/python3.12 tests/test_barre_ecriture.py
"""
import importlib.util
import math
import os
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.barre import ecriture as EC  # noqa: E402
from acier.barre import entree as EN  # noqa: E402
from acier.barre import formules as FO  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.barre.resultat import formule, terme  # noqa: E402

OK, KO = [], []
CHEMIN = os.path.join(RACINE, "acier", "barre", "ecriture.py")
# ce qu'une formule EN VALEURS peut contenir : chiffres, virgule, opérateurs, parenthèses, racine, π, exposants
VALEURS = re.compile(r"^[0-9 ,()+−×/√π^⁰¹²³⁴⁵⁶⁷⁸⁹]+$")
PROGRAMME = re.compile(r"\*|sqrt|1e\d|\bpi\b|(?<![A-Za-z_,])\d+\.\d")          # ce qu'une formule en symboles ne doit plus porter


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def bloc(gabarit, symbole, termes, resultat, unite=""):
    """Un bloc tel que le moteur le rend : ``termes`` = [(symbole, valeur)]."""
    return formule("x", symbole, [terme(s, v) for s, v in termes], resultat, unite, gabarit)


def ligne(e, b, symboles=None):
    parts, res = e.morceaux(b, symboles)
    return " = ".join(parts + [res])


def quelconque(gabarit):
    """Un bloc de valeurs quelconques pour ce gabarit, que son expression sait évaluer (plusieurs jeux essayés : une
    racine veut un argument positif), son résultat étant celui du gabarit."""
    noms, _, _ = FO.GABARITS[gabarit]
    for depart, pas in ((1.37, 0.211), (0.83, 0.047), (2.6, 0.9), (0.31, 0.013), (-0.4, 0.3)):
        b = formule("x", "X", [terme(f"s_{n}", depart + pas * i) for i, n in enumerate(noms)], 0.0, "u", gabarit)
        try:
            v = FO.evaluer(b)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if isinstance(v, float) and math.isfinite(v) and abs(v) > 1e-6:
            b["resultat"] = v
            return b
    return None


def barres():
    """Des barres réelles, pour toutes les formules que le moteur écrit."""
    out = []
    b = EN.nouvelle_barre("B1")
    out.append(b)
    b = EN.nouvelle_barre("R2")
    b.update({"poids_propre": False, "cas": [
        {"nom": "G", "nature": "permanent", "N": 80.0, "charges": [{"type": "q", "v": 2.42, "a": 0.0, "b": 6.0}, {"type": "P", "v": 10.0, "a": 3.0}]},
        {"nom": "Q", "nature": "exploitation_A", "charges": [{"type": "P", "v": 15.0, "a": 3.0}]}]})
    out.append(b)
    b = EN.nouvelle_barre("C")
    b.update({"profil": "HEA 200", "modele": "continue", "travees": [5.0, 4.0], "maintiens": [{"x": 2.5, "genre": "haut"}]})
    out.append(b)
    for prof, nu in (("HEB 200", "S235"), ("IPE 600", "S355"), ("HEA 300", "S355"), ("RHS 300x100x8", "S355"), ("CHS 219.1x6.3", "S235"),
                     ("HEA 1000", "S460"), ("IPE 300", "S235")):
        for N, My, Mz, Vz in ((300.0, 40.0, 10.0, 60.0), (900.0, 0.0, 0.0, 0.0), (-200.0, 80.0, 0.0, 30.0), (0.0, 150.0, 20.0, 200.0),
                              (2500.0, 300.0, 30.0, 400.0), (50.0, 0.0, 12.0, 0.0)):
            for forme in ("constant", "lineaire", "reparti", "ponctuel"):
                b = EN.nouvelle_barre("E")
                b.update({"mode": "efforts", "profil": prof, "nuance": nu, "flambement": {"y": {"L": 6.0, "u": 1.0}, "z": {"L": 3.0, "u": 1.0}},
                          "diagramme": {"y": {"forme": forme, "psi": -0.5}, "z": {"forme": forme, "psi": 0.3}}})
                b["efforts"]["elu"].update({"N": N, "My": My, "Mz": Mz, "Vz": Vz, "Vy": 10.0})
                b["efforts"]["service"].update({"N": N / 1.4, "My": My / 1.4, "Mz": Mz / 1.4, "Vz": Vz / 1.4})
                out.append(b)
    return out


BLOCS = []                        # toutes les formules du moteur sur ces barres : (profilé, méthode, bloc)
SUITES = []                       # … et, par vérification, ses formules dans l'ordre : (profilé, méthode, identifiant, blocs)


def blocs_du_moteur():
    if not BLOCS:
        for b in barres():
            r = MO.verifier(b)
            for m, x in r["methodes"].items():
                BLOCS.extend((b["profil"], m, f) for f in x["calcul"])
                par = {f["cle"]: f for f in x["calcul"]}
                SUITES.extend((b["profil"], m, v["id"], [par[k] for k in v["formules"]]) for v in x["verifications"])
    return BLOCS


def redites(suites):
    """Dans une même vérification, un terme qui porte le symbole d'une ligne écrite plus haut doit en avoir la valeur :
    [(profilé, méthode, vérification, clé, symbole, valeur de la ligne, valeur du terme)] pour chaque écart."""
    faux = []
    for prof, m, id_, blocs in suites:
        vus = {}
        for f in blocs:
            for t in f["termes"]:
                v = vus.get(t["symbole"])
                if v is not None and abs(v - t["valeur"]) > 1e-9 * max(1.0, abs(v)):
                    faux.append((prof, m, id_, f["cle"], t["symbole"], v, t["valeur"]))
            vus[f["symbole"]] = f["resultat"]
    return faux


# ------------------------------------------------------------------ les contrôles (rejoués par les mutations)
def controles(e=None):
    e = e or EC
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    # ---- 1. les nombres
    n = e.nombre
    lus = [n(235.0), n(1.0), n(0.34), n(0.627803), n(1.118234), n(348.2617), n(68.704), n(5381.4), n(210000.0), n(-12.5), n(0.0), n(-0.0),
           n(0.000234567), n(None), n(float("nan")), n(12345.678)]
    c("un nombre imprimé : un entier en entier (235, 210000), sinon cinq chiffres significatifs sans zéro inutile (0,6278 ; 1,1182 ; "
      "348,26 ; 68,704 ; 5381,4 ; 12346), le vrai signe moins, « — » s'il manque",
      lus == ["235", "1", "0,34", "0,6278", "1,1182", "348,26", "68,704", "5381,4", "210000", "−12,5", "0", "0", "0,00023457", "—", "—", "12346"], lus)
    c("les décimales d'un résultat sont celles qu'on lui donne, sans zéro inutile : 569,23 à une décimale → 569,2 ; 0,5 à trois → 0,5 ; "
      "1264,96 à zéro → 1265", [n(569.23, 1), n(0.5, 3), n(1264.96, 0)] == ["569,2", "0,5", "1265"], [n(569.23, 1), n(0.5, 3), n(1264.96, 0)])

    # ---- 2. des formules écrites à la main
    b1 = bloc("e_N_b_Rd", "N_b,z,Rd", [("χ_z", 0.31022), ("A", 7808.0), ("f_y", 235.0), ("γ_M1", 1.0)], 0.31022 * 7808.0 * 235.0 / 1e3, "kN")
    c("flambement : « N_b,z,Rd = χ_z·A·f_y/γ_M1 = 0,31022 × 7808 × 235 / 1,0 / 10³ = 569,2 kN » — le facteur d'unités est ôté de la "
      "formule en symboles et imprimé « / 10³ » dans celle en valeurs ; le coefficient partiel garde sa décimale",
      ligne(e, b1) == "N_b,z,Rd = χ_z·A·f_y/γ_M1 = 0,31022 × 7808 × 235 / 1,0 / 10³ = 569,2 kN", ligne(e, b1))
    b2 = bloc("e_N_cr", "N_cr,y", [("E", 210000.0), ("I_y", 8356.0), ("L_cr,y", 6.0)], math.pi ** 2 * 210000.0 * 8356.0 / 36.0 / 1e5, "kN")
    c("charge critique : « N_cr,y = π²·E·I_y/L_cr,y² = π² × 210000 × 8356 / 6² / 10⁵ = 4811 kN » (π, le carré, / 10⁵)",
      ligne(e, b2) == "N_cr,y = π²·E·I_y/L_cr,y² = π² × 210000 × 8356 / 6² / 10⁵ = 4811 kN", ligne(e, b2))
    b3 = bloc("e_alpha", "α", [("N_Ed", -200.0), ("c", 134.0), ("t_w", 9.0), ("f_y", 235.0)], 0.5 - 200.0 * 1e3 / (2 * 134.0 * 9.0 * 235.0))
    c("une valeur négative se met entre parenthèses : « α = 0,5 + N_Ed/(2·c·t_w·f_y) = 0,5 + (−200) × 10³ / (2 × 134 × 9 × 235) = 0,147 »",
      ligne(e, b3) == "α = 0,5 + N_Ed/(2·c·t_w·f_y) = 0,5 + (−200) × 10³ / (2 × 134 × 9 × 235) = 0,147", ligne(e, b3))
    b4 = bloc("e_lambda_p", "λ̄_p", [("c/t", 42.833), ("ε", 0.81362), ("k_σ", 4.0)], 42.833 / (28.4 * 0.81362 * 2.0))
    c("un symbole composé prend des parenthèses devant une barre de fraction : « λ̄_p = (c/t)/(28,4·ε·√(k_σ)) = 42,833 / (28,4 × 0,81362 × √4) "
      "= 0,927 »", ligne(e, b4) == "λ̄_p = (c/t)/(28,4·ε·√(k_σ)) = 42,833 / (28,4 × 0,81362 × √4) = 0,927", ligne(e, b4))
    b5 = bloc("e_mu", "μ_y", [("N_Ed/N_cr,y", 0.022448), ("χ_y", 0.92028)], (1 - 0.022448) / (1 - 0.92028 * 0.022448))
    c("… et n'en prend pas là où la priorité des opérations ne le demande pas : « μ_y = (1 − N_Ed/N_cr,y)/(1 − χ_y·N_Ed/N_cr,y) = … = 0,998 »",
      ligne(e, b5) == "μ_y = (1 − N_Ed/N_cr,y)/(1 − χ_y·N_Ed/N_cr,y) = (1 − 0,022448) / (1 − 0,92028 × 0,022448) = 0,998", ligne(e, b5))
    b6 = bloc("e_deviee", "η", [("M_y,Ed", 40.0), ("M_N,y,Rd", 142.86), ("α", 2.0), ("M_z,Ed", 10.0), ("M_N,z,Rd", 71.867), ("β", 1.19)],
              (40.0 / 142.86) ** 2 + (10.0 / 71.867) ** 1.19)
    c("exposants : le carré en exposant, les autres en « ^ » ; un taux en pourcentage, à une décimale : « η = (M_y,Ed/M_N,y,Rd)^α + "
      "(M_z,Ed/M_N,z,Rd)^β = (40 / 142,86)² + (10 / 71,867)^1,19 = 17,4 % »",
      ligne(e, b6) == "η = (M_y,Ed/M_N,y,Rd)^α + (M_z,Ed/M_N,z,Rd)^β = (40 / 142,86)² + (10 / 71,867)^1,19 = 17,4 %", ligne(e, b6))
    b7 = bloc("e_lambda_0_lim", "λ̄_0,lim", [("C₁", 1.0), ("N_Ed/N_cr,z", 0.31069), ("N_Ed/N_cr,T", 0.07723)],
              0.2 * ((1 - 0.31069) * (1 - 0.07723)) ** 0.25)
    c("racine d'un nombre, puissance un quart : « λ̄_0,lim = 0,2·√(C₁)·((1 − N_Ed/N_cr,z)·(1 − N_Ed/N_cr,T))^0,25 = 0,2 × √1 × "
      "((1 − 0,31069) × (1 − 0,07723))^0,25 = 0,179 »",
      ligne(e, b7) == "λ̄_0,lim = 0,2·√(C₁)·((1 − N_Ed/N_cr,z)·(1 − N_Ed/N_cr,T))^0,25 = 0,2 × √1 × ((1 − 0,31069) × (1 − 0,07723))^0,25 = 0,179",
      ligne(e, b7))
    b8 = bloc("e_W_eff", "W_eff,z", [("I_eff", 1052.0), ("v", 50.0), ("e", 0.94225)], 1052.0 * 10 / 50.94225, "cm³")
    c("un facteur dix s'imprime « × 10 » : « W_eff,z = I_eff/(v + e) = 1052 × 10 / (50 + 0,94225) = 206,5 cm³ »",
      ligne(e, b8) == "W_eff,z = I_eff/(v + e) = 1052 × 10 / (50 + 0,94225) = 206,5 cm³", ligne(e, b8))
    b9 = bloc("P_cr", "P_cr,y", [("E", 21000.0), ("I_y", 8356.5), ("u", 1.0), ("L", 600.0)], math.pi ** 2 * 21000.0 * 8356.5 / 600.0 ** 2, "kN")
    c("méthode historique, les unités des classeurs (E en kN/cm², cm⁴, cm) : « P_cr,y = π²·E·I_y/(u·L)² = π² × 21000 × 8356,5 / (1 × 600)² "
      "= 4811 kN » — aucun facteur", ligne(e, b9) == "P_cr,y = π²·E·I_y/(u·L)² = π² × 21000 × 8356,5 / (1 × 600)² = 4811 kN", ligne(e, b9))
    b10 = bloc("taux", "η", [("valeur", 68.702), ("limite", 76.74)], 68.702 / 76.74)
    c("le taux d'une vérification s'écrit avec les deux membres de son critère : « η = M_y,Ed/M_b,Rd = 68,702 / 76,74 = 89,5 % »",
      ligne(e, b10, ["M_y,Ed", "M_b,Rd"]) == "η = M_y,Ed/M_b,Rd = 68,702 / 76,74 = 89,5 %", ligne(e, b10, ["M_y,Ed", "M_b,Rd"]))
    b11 = bloc("e_eps", "ε", [("f_y", 235.0)], 1.0)
    c("une constante de la norme reste dans la formule : « ε = √(235/f_y) = √(235 / 235) = 1 » ; les membres sont rendus séparés "
      "(symbole, formule, valeurs) puis le résultat", ligne(e, b11) == "ε = √(235/f_y) = √(235 / 235) = 1"
      and e.morceaux(b11) == (["ε", "√(235/f_y)", "√(235 / 235)"], "1"), ligne(e, b11))
    # la relecture : le gabarit sur les nombres IMPRIMÉS, pas sur les valeurs exactes
    b12 = bloc("e_n", "n", [("N_Ed", 1.000004), ("N_pl,Rd", 3.0)], 1.000004 / 3.0)
    c("la relecture évalue le gabarit sur les nombres IMPRIMÉS : N_Ed = 1,000004 s'imprime « 1 », la relecture rend 1/3 et non 0,333335",
      e.ecrire(b12)["numerique"] == "1 / 3" and e.relire(b12) == 1.0 / 3.0 and e.relecture_juste(b12), (e.ecrire(b12)["numerique"], e.relire(b12)))
    juste = bloc("e_N_Rd", "N_c,Rd", [("A", 10000.0), ("f_y", 355.0), ("γ_M0", 1.0)], 3551.0, "kN")
    faux = bloc("e_N_Rd", "N_c,Rd", [("A", 10000.0), ("f_y", 355.0), ("γ_M0", 1.0)], 3557.1, "kN")
    c("la relecture est jugée au dernier chiffre imprimé ou à 5·10⁻⁴ près : 10000 × 355 / 10³ = 3550 passe pour un résultat imprimé 3551 "
      "(0,03 %), pas pour 3557,1 (0,2 %)", e.relecture_juste(juste) and not e.relecture_juste(faux))
    try:
        e.ecrire(bloc("e_eps", "ε", [("f_y", 235.0), ("x", 1.0)], 1.0))
        refus = False
    except ValueError as ex:
        refus = "e_eps" in str(ex)
    c("un bloc dont les termes ne sont pas ceux de son gabarit est refusé, le gabarit nommé", refus)

    # ---- 3. tous les gabarits, sur des valeurs quelconques
    sans, restes, programme, relus = [], [], [], []
    for g in FO.GABARITS:
        b = quelconque(g)
        if b is None:
            sans.append(g)
            continue
        ec = e.ecrire(b)
        if not VALEURS.match(ec["numerique"]):
            restes.append((g, ec["numerique"]))
        if PROGRAMME.search(ec["litteral"]) or any(f"s_{n_}" not in ec["litteral"] for n_ in FO.GABARITS[g][0]):
            programme.append((g, ec["litteral"]))
        if not e.relecture_juste(b):
            relus.append((g, e.relire(b), b["resultat"]))
    c(f"les {len(FO.GABARITS)} gabarits, chacun sur des valeurs quelconques : la formule en VALEURS ne garde ni identifiant ni opérateur de "
      "programme", not sans and not restes, (sans, restes[:3]))
    c("… la formule en SYMBOLES porte chacun de ses symboles, sans facteur d'unités, sans « * », « sqrt », « pi » ni point décimal",
      not programme, programme[:3])
    c("… et le gabarit évalué sur les nombres IMPRIMÉS redonne le résultat (au dernier chiffre imprimé, ou à 5·10⁻⁴ près)", not relus, relus[:3])
    c("les facteurs d'unités des gabarits s'écrivent 1eN, jamais 1000 ni 100000 (c'est ce qui les distingue d'un coefficient de la norme)",
      not [g for g, (_, x, _) in FO.GABARITS.items() if re.search(r"[*/]10{3,}(?![0-9.])", x)])
    return C


def charger(mutation):
    src = open(CHEMIN, encoding="utf-8").read()
    if src.count(mutation[0]) != 1:
        return None
    spec = importlib.util.spec_from_file_location("acier.barre.ecriture_mute", CHEMIN)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "acier.barre"
    exec(compile(src.replace(mutation[0], mutation[1]), CHEMIN, "exec"), mod.__dict__)
    return mod


MUTATIONS = [
    ("trois chiffres significatifs au lieu de cinq", "CHIFFRES = 5 ", "CHIFFRES = 3 "),
    ("le facteur d'unités gardé dans la formule en symboles", 'litteral = _habiller(_IDENT.sub(lit, _UNITE.sub("", expression)), False)',
     "litteral = _habiller(_IDENT.sub(lit, expression), False)"),
    ("le facteur d'unités toujours imprimé 10³", '("" if m.group(2) == "1" else _EXPOSANTS[m.group(2)])', '_EXPOSANTS["3"]'),
    ("le facteur d'unités oublié dans la formule en valeurs", 'brut = _UNITE.sub(lambda m: f" {_JETON}{m.group(1)}{m.group(2)}{_JETON}", expression)',
     'brut = _UNITE.sub("", expression)'),
    ("une division d'unités imprimée comme un produit", 'lambda m: m.group(1) + "10"', 'lambda m: " × 10"'),
    ("une valeur négative sans parenthèses", '        return f"({s})" if s.startswith(MOINS) else s\n', "        return s\n"),
    ("un symbole composé sans parenthèses", '    if any(c in s for c in ("+", MOINS, " - ")) or ("/" in s and (avant == "/" or apres[:1] == "/" or apres.startswith("**"))):',
     "    if False:"),
    ("des parenthèses à tout symbole composé", '("/" in s and (avant == "/" or apres[:1] == "/" or apres.startswith("**")))', '"/" in s'),
    ("le coefficient partiel sans sa décimale", '            return nombre(v) + ",0"\n', "            return nombre(v)\n"),
    ("le pourcentage sans × 100", 'res = nombre(bloc["resultat"] * 100.0, 1) + " %"', 'res = nombre(bloc["resultat"], 1) + " %"'),
    ("l'unité du résultat oubliée", 'res = nombre(bloc["resultat"], nd) + (" " + bloc["unite"] if bloc["unite"] else "")', 'res = nombre(bloc["resultat"], nd)'),
    ("le résultat à cinq chiffres au lieu des décimales de son gabarit", 'res = nombre(bloc["resultat"], nd) + (" " + bloc["unite"] if bloc["unite"] else "")',
     'res = nombre(bloc["resultat"]) + (" " + bloc["unite"] if bloc["unite"] else "")'),
    ("le point décimal d'une constante gardé", '    expression = _DECIMALE.sub(r"\\1,\\2", expression)\n', ""),
    ("π non écrit", '        if k == "pi":\n            return "π"\n        if k not in par_sym:', '        if k not in par_sym:'),
    ("le carré écrit « ^2 »", 'return _EXPOSANTS[e] if e in ("2", "3", "4", "5") else "^" + e', 'return "^" + e'),
    ("le signe moins du clavier", 'MOINS = "−"', 'MOINS = "-"'),
    ("les symboles donnés (le critère d'un taux) ignorés", 'syms = list(symboles) if symboles is not None else [t["symbole"] for t in bloc["termes"]]',
     'syms = [t["symbole"] for t in bloc["termes"]]'),
    ("un bloc mal formé accepté", "    if len(noms) != len(bloc[\"termes\"]):", "    if False:"),
    ("la relecture sur les valeurs exactes (elle ne jugerait plus l'impression)", 'env = {n: _lu(nombre(t["valeur"])) for n, t in zip(noms, bloc["termes"])}',
     'env = {n: t["valeur"] for n, t in zip(noms, bloc["termes"])}'),
    ("la tolérance de relecture portée à 5 %", "RELECTURE = 5e-4 ", "RELECTURE = 5e-2 "),
]


def main():
    print("== 1 à 3. les nombres, des formules écrites à la main, tous les gabarits")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 4. les formules du moteur")
    blocs = blocs_du_moteur()
    faux, restes, vides = [], [], []
    vus = set()
    for prof, m, f in blocs:
        vus.add(f["gabarit"])
        ec = EC.ecrire(f)
        if not EC.relecture_juste(f):
            faux.append((prof, m, f["cle"], EC.relire(f), f["resultat"]))
        if not VALEURS.match(ec["numerique"]) or PROGRAMME.search(ec["litteral"]):
            restes.append((prof, m, f["cle"], ec["litteral"], ec["numerique"]))
        if not ec["symbole"] or not ec["resultat"] or ec["resultat"].startswith("—"):
            vides.append((prof, m, f["cle"]))
    chk(f"{len(blocs)} formules du moteur ({len(vus)} gabarits ; efforts imposés sur sept profilés, quatre diagrammes ; trois barres chargées) : "
        "refaite à la main avec ses nombres IMPRIMÉS, chacune redonne son résultat IMPRIMÉ", len(blocs) > 8000 and len(vus) >= 88 and not faux, faux[:3])
    chk("… aucune ne garde un identifiant, un opérateur de programme ou un résultat vide", not restes and not vides, (restes[:2], vides[:2]))
    faux = redites(SUITES)
    termes = sum(len(f["termes"]) for _p, _m, _i, blocs in SUITES for f in blocs)
    chk(f"dans une vérification, une grandeur écrite deux fois garde sa valeur : un terme qui porte le symbole d'une ligne écrite plus haut en a "
        f"le résultat ({len(SUITES)} vérifications, {termes} termes) — le lecteur retrouve chaque nombre", len(SUITES) > 1500 and not faux, faux[:3])
    piege = [("X", "ec3", "E-Mb", [bloc("e_chi_LT_mod", "χ_LT,mod", [("χ_LT", 0.5), ("f", 0.9)], 0.5 / 0.9),
                                   bloc("e_M_b_Rd", "M_b,Rd", [("χ_LT,mod", 0.6), ("W_pl,y", 628.4), ("f_y", 235.0), ("γ_M1", 1.0)], 88.6, "kN·m")])]
    chk("… et ce contrôle voit l'écart : une ligne « χ_LT,mod = 0,556 » suivie d'un terme « χ_LT,mod » à 0,6 est relevée",
        [x[3:5] for x in redites(piege)] == [("x", "χ_LT,mod")], redites(piege))
    b = EN.nouvelle_barre("R2")
    b.update({"poids_propre": False, "cas": [
        {"nom": "G", "nature": "permanent", "N": 80.0, "charges": [{"type": "q", "v": 2.42, "a": 0.0, "b": 6.0}, {"type": "P", "v": 10.0, "a": 3.0}]},
        {"nom": "Q", "nature": "exploitation_A", "charges": [{"type": "P", "v": 15.0, "a": 3.0}]}]})
    r = MO.verifier(b)["methodes"]["ec3"]
    par = {f["cle"]: f for f in r["calcul"]}
    mb = ligne(EC, par["E-Mb/M_b_Rd"])
    chk("R2 (IPE 300, 6 m) : la résistance au déversement s'écrit « M_b,Rd = χ_LT·W_pl,y·f_y/γ_M1 = … × 628,4 × 235 / 1,0 / 10³ = … kN·m »",
        mb.startswith("M_b,Rd = χ_LT·W_pl,y·f_y/γ_M1 = 0,") and "× 235 / 1,0 / 10³ = " in mb and mb.endswith(" kN·m"), mb)

    print("== 5. mutations")
    for nom, ancien, nouveau in MUTATIONS:
        mod = charger((ancien, nouveau))
        if mod is None:
            chk(f"mutation « {nom} » : le texte à muter est dans l'écriture, une seule fois", False, "texte absent ou répété")
            continue
        try:
            rouges = [n_ for n_, ok, _ in controles(mod) if not ok]
        except Exception as ex:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
            rouges = [f"exception {type(ex).__name__}: {ex}"]
        chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")

    print("== 6. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab', 'PIL', 'fitz', 'numpy'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from acier.barre import ecriture as E\n"
            "from acier.barre.resultat import formule, terme\n"
            "print(E.morceaux(formule('x', 'ε', [terme('f_y', 355.0)], (235 / 355.0) ** 0.5, '', 'e_eps')))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("l'écriture se charge et écrit sans numpy, streamlit, pandas, matplotlib, scipy, reportlab (refusés) : bibliothèque standard seule",
        p.returncode == 0 and "√(235 / 355)" in p.stdout, (p.stdout.strip(), p.stderr[-300:]))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
