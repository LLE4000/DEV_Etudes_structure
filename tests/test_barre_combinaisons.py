# -*- coding: utf-8 -*-
"""Barres en acier — les COMBINAISONS de l'Eurocode (acier/barre/combinaisons.py) et la TABLE des paramètres
(acier/barre/parametres.py). La spécification : brief v12 § 5, § 7.2, § 9 (hypothèses imprimées), § 13.1, § 13.2 ;
docs/acier/BARRES_ACIER.md § 5.8. Lot C, bloc C2.

Les contrôles :
  1. LA TABLE — ψ₁ de la neige = 0 ; γ, ψ, γ_M, E et G « pointé » avec leur source (la clause, jamais son texte) ; ξ,
     ξ en géotechnique, ψ₀ de courte durée ; un seul paramètre « à confirmer » (n_pl) ; les DÉCISIONS BUREAU posées
     par C0 (n° 1, 2, 3, 4, 9, 10) : statut, source, point du § 13.2, défaut inchangé.
  2. LES NATURES — exécution et machines (ψ₀ saisi, demandé seulement quand il sert) ; la limite compte les NATURES.
  3. TEST 23 du § 11.5 — (6.10a) et (6.10b) : ξ = 0,85, G_inf = 1,00, géotechnique ξ = 1,00 ; l'enveloppe du moteur
     retient la plus défavorable des deux.
  4. TEST 24 du § 11.5 — la règle belge de courte durée, cas par cas.
  5. γ_G PAR SOURCE — une continue de deux travées, G₁ sur la travée 1, G₂ sur la travée 2 : la combinaison
     1,35·G₁ + G₂ existe et donne le plus grand moment de la travée 1 (valeur du formulaire) ; deux cas de même
     répartition font une source ; le budget (refus « au plus k cas permanents »).
  6. LES NOMS ET LES HYPOTHÈSES — un nom court ; une hypothèse = une ligne de 120 caractères au plus.
  7. MUTATIONS — chaque règle altérée fait rougir un contrôle ; le module muté est remis, ceux qui l'emploient
     rechargés.

    /usr/bin/python3.12 -B tests/test_barre_combinaisons.py
"""
import importlib
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import acier.barre as PAQUET  # noqa: E402
from acier.barre import moteur as _moteur  # noqa: E402,F401 — chargé une fois : une mutation le recharge

OK, KO = [], []
DOSSIER = os.path.join(RACINE, "acier", "barre")


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def m(nom):
    """Le module du paquet tel qu'il est MAINTENANT (une mutation le remplace)."""
    return sys.modules[f"acier.barre.{nom}"]


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def q(v, a, b):
    return {"type": "q", "v": v, "a": a, "b": b}


def leve(f, classe=None):
    try:
        f()
    except Exception as e:  # noqa: BLE001
        return classe is None or isinstance(e, classe)
    return False


def facteurs(combs, cas, **cond):
    """{nom du cas: facteur} de LA combinaison dont les champs valent ``cond`` (None si aucune ou plusieurs)."""
    t = [c for c in combs if all(c.get(k) == v for k, v in cond.items())]
    return {x["nom"]: f for x, f in zip(cas, t[0]["facteurs"])} if len(t) == 1 else None


def avec(combs, cas, **fs):
    """Les combinaisons où chaque cas nommé a ce facteur (à 10⁻⁹)."""
    idx = {c["nom"]: i for i, c in enumerate(cas)}
    return [c for c in combs if all(proche(c["facteurs"][idx[n]], v) for n, v in fs.items())]


G = {"nom": "G", "nature": "permanent"}
QA = {"nom": "Q", "nature": "exploitation_A"}
S = {"nom": "S", "nature": "neige"}
W = {"nom": "W", "nature": "vent"}
T = {"nom": "T", "nature": "temperature"}


def barre(**k):
    b = m("entree").nouvelle_barre("B")
    b.update(k)
    return b


def verif(r, id_):
    return next((v for v in r["verifications"] if v["id"] == id_), None)


# ------------------------------------------------------------------ les contrôles (rejoués par les mutations)
def controles():
    PA, CO, ST, MO = m("parametres"), m("combinaisons"), m("statique"), m("moteur")
    P = PA.jeu()
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    # ---- 1. la table
    T_ = PA.PARAMETRES
    c("test 24 : neige ψ₁ = 0 (tableau A1.1 ANB : 0,5 / 0 / 0) ; vent 0,6 / 0,2 / 0 ; température 0,6 / 0,5 / 0",
      PA.valeur("psi_neige") == (0.5, 0.0, 0.0) and PA.valeur("psi_vent") == (0.6, 0.2, 0.0) and PA.valeur("psi_temperature") == (0.6, 0.5, 0.0),
      (PA.valeur("psi_neige"), PA.valeur("psi_vent")))
    # chaque VALEUR contre le brief : § 5 (γ, ξ, ψ du tableau A1.1 ANB), § 7.1 (E, G), § 7.2 (γ_M) — un ψ faux ne se voit
    # dans aucun autre contrôle (le statut et la source ne disent pas la valeur)
    brief = {"gamma_G_sup": 1.35, "gamma_G_inf": 1.00, "gamma_Q": 1.50, "xi": 0.85, "xi_geotechnique": 1.00, "psi0_courte_duree": 0.3,
             "psi_A": (0.7, 0.5, 0.3), "psi_B": (0.7, 0.5, 0.3), "psi_C": (0.7, 0.7, 0.6), "psi_D": (0.7, 0.7, 0.6), "psi_E": (1.0, 0.9, 0.8),
             "psi_F": (0.7, 0.7, 0.6), "psi_G": (0.7, 0.5, 0.3), "psi_H": (0.0, 0.0, 0.0), "psi_neige": (0.5, 0.0, 0.0),
             "psi_vent": (0.6, 0.2, 0.0), "psi_temperature": (0.6, 0.5, 0.0),
             "gamma_M0": 1.00, "gamma_M1": 1.00, "gamma_M2": 1.25, "E": 210000.0, "G": 81000.0}
    ecarts = {k: (PA.valeur(k), v) for k, v in brief.items() if PA.valeur(k) != v or P[k] != v}
    c("la table contre le brief, valeur par valeur : γ_G,sup, γ_G,inf, γ_Q, ξ, ψ des catégories A à H, neige, vent, température "
      "(§ 5) ; E, G (§ 7.1) ; γ_M0, γ_M1, γ_M2 (§ 7.2)", not ecarts, ecarts)
    sources = {"NBN EN 1990 ANB:2021, tableau A1.2(B), CC2": ("gamma_G_sup", "gamma_G_inf", "gamma_Q"),
               "NBN EN 1990 ANB:2021, tableau A1.1": tuple(f"psi_{k}" for k in "ABCDEFGH") + ("psi_neige", "psi_vent", "psi_temperature"),
               "NBN EN 1993-1-1 ANB:2018, 6.1(1)B": ("gamma_M0", "gamma_M1", "gamma_M2"), "EN 1993-1-1 § 3.2.6": ("E", "G")}
    faux = [(k, T_[k]["statut"], T_[k]["source"]) for src, cles in sources.items() for k in cles
            if T_[k]["statut"] != PA.POINTE or src not in T_[k]["source"]]
    c("γ_G, γ_Q (A1.2(B), CC2), ψ (A1.1), γ_M (ANB:2018 6.1(1)B), E et G (3.2.6) : « pointé », la source dit la clause", not faux, faux)
    c("ξ = 0,85 ; ξ = 1,00 en géotechnique ; ψ₀ de courte durée = 0,3 (note 3 du tableau A1.1 ANB) : dans la table, pointés",
      (P["xi"], P["xi_geotechnique"], P["psi0_courte_duree"]) == (0.85, 1.0, 0.3) and T_["psi0_courte_duree"]["source"].endswith("note 3")
      and all(T_[k]["statut"] == PA.POINTE for k in ("xi", "xi_geotechnique", "psi0_courte_duree")), (P["xi"], P["xi_geotechnique"]))
    c("un seul paramètre « à confirmer » : n_pl (γ_M0 ≠ γ_M1) ; plus aucun « à pointer »",
      [x["cle"] for x in PA.a_confirmer()] == ["npl_gamma"] and not [k for k, p in T_.items() if p["statut"] == PA.A_POINTER],
      [x["cle"] for x in PA.a_confirmer()])
    textes = [p["source"] for p in T_.values()]
    c("les sources disent la clause, jamais son texte : 200 caractères au plus, chacune une référence (NBN, EN, ANB)",
      all(len(t) <= 200 and re.search(r"\b(EN|ANB)\b", t) for t in textes), [t for t in textes if len(t) > 200][:2])
    dec = PA.decisions()
    attendu = [("eta1_ame", 1, "elastique", "plastique"), ("contact_inconnu", 2, "non", "oui"), ("avy_lamine", 3, "2btf", "A_moins_hwtw"),
               ("methode_deversement_lamines", 4, "6.3.2.3", "6.3.2.2"), ("console_NM", 9, "non_couvert", None), ("prs_soudee_equivalente", 10, False, True)]
    c("les DÉCISIONS BUREAU de C0 (n° 1, 2, 3, 4, 9, 10) : statut « decision », le défaut et la variante du brief § 13.2, la source "
      "nomme son numéro (un défaut changé ici rougit)",
      [(d["cle"], d["point"], d["valeur"], d["variante"]) for d in dec] == attendu
      and all(f"DÉCISION BUREAU n° {d['point']}" in d["source"] for d in dec), [(d["cle"], d["point"], d["valeur"]) for d in dec])

    # ---- 2. les natures
    c("natures neuves : « execution » et « machines », ψ₀ SAISI (aucune entrée ψ dans la table) ; pas de « tassements » ; "
      "fluides et matières stockés en catégorie E",
      CO.NATURES["execution"][1] == CO.SAISI and CO.NATURES["machines"][1] == CO.SAISI and "tassements" not in CO.NATURES
      and "fluides" in CO.NATURES["exploitation_E"][0] and not any(k in T_ for k in ("psi_execution", "psi_machines")), list(CO.NATURES))
    c("l'aide du ψ₀ : exécution 1,0 / – / 0,2 du tableau A1.1 ANB (note 2 : au cas par cas) ; machines au cas par cas (note 1) ; "
      "une ligne chacune",
      "1,0" in CO.AIDE_PSI0["execution"] and "0,2" in CO.AIDE_PSI0["execution"] and "note 2" in CO.AIDE_PSI0["execution"]
      and "note 1" in CO.AIDE_PSI0["machines"] and all(len(t) <= 120 and "\n" not in t for t in CO.AIDE_PSI0.values()), CO.AIDE_PSI0)
    E0 = {"nom": "E", "nature": "execution"}
    seule = CO.elu([G, E0], P)
    c("exécution seule (6.10), ψ₀ non saisi : il ne sert pas — 1,35·G + 1,5·E (γ_Q = 1,5), rien n'est demandé",
      [(x["nom"], x["facteurs"]) for x in seule] == [("1,35·G", [1.35, 0.0]), ("1,35·G + 1,5·E", [1.35, 1.5]), ("G + 1,5·E", [1.0, 1.5])],
      [x["nom"] for x in seule])
    c("exécution qui ACCOMPAGNE une autre nature, ψ₀ non saisi : bloqué, valeur à fournir (AFournir, un refus qui nomme le cas) ; "
      "(6.10a) : toujours (toutes les variables y sont à ψ₀)",
      leve(lambda: CO.elu([G, E0, QA], P), CO.AFournir) and leve(lambda: CO.elu([G, E0], P, {"expression": "6.10ab"}), CO.AFournir)
      and _message(lambda: CO.elu([G, E0, QA], P)) == "ψ₀ à fournir (au cas par cas, tableau A1.1 ANB) : « E »"
      and issubclass(CO.AFournir, m("resultat").Refus), _message(lambda: CO.elu([G, E0, QA], P)))
    E1 = dict(E0, psi0=1.0)
    M6 = {"nom": "M", "nature": "machines", "psi0": 0.6}
    cas = [G, E1, QA, M6]
    co = CO.elu(cas, P)
    c("ψ₀ saisi : exécution 1,0 → 1,5·E derrière Q ; machines 0,6 → 0,9·M ; en tête, 1,5 ; Q derrière l'exécution : 1,05 (tableau)",
      facteurs(co, cas, dominante="exploitation_A", nom="1,35·G + 1,5·E + 1,5·Q + 0,9·M") is not None
      and facteurs(co, cas, dominante="execution", nom="1,35·G + 1,5·E + 1,05·Q + 0,9·M") is not None
      and facteurs(co, cas, dominante="machines", nom="1,35·G + 1,5·E + 1,05·Q + 1,5·M") is not None, sorted(x["nom"] for x in co)[:6])
    Qs = dict(QA, psi0=0.2)
    c("un ψ₀ saisi sur une autre nature ne sert pas : la catégorie A garde 0,7 du tableau",
      avec(CO.elu([G, Qs, S], P), [G, Qs, S], G=1.35, Q=1.05, S=1.5) != [], "")
    cinq = [G] + [dict(QA, nom=f"Q{i}") for i in range(3)] + [dict(S, nom=f"S{i}") for i in range(2)]
    n5 = len(CO.elu(cinq, P))
    c(f"la limite compte les NATURES : cinq cas de deux natures acceptés ({n5} combinaisons) ; cinq natures refusées (« au plus 4 natures »)",
      n5 == 105 and _message(lambda: CO.elu([G, QA, dict(QA, nom="B", nature="exploitation_B"), S, W, T], P))
      == "au plus 4 natures d'actions variables (la barre en a 5)", n5)

    # ---- 3. test 23 : (6.10a) et (6.10b)
    ab = {"expression": "6.10ab"}
    co = CO.elu([G, QA], P, ab)
    c("test 23 : (6.10a) 1,35·G + 1,5·ψ₀·Q ; (6.10b) ξ·1,35·G (1,1475) + 1,5·Q ; G_inf = 1,00 dans les deux ; versées ensemble",
      [(x["nom"], x["expression"], [round(f, 12) for f in x["facteurs"]]) for x in co]
      == [("1,35·G (6.10a)", "6.10a", [1.35, 0.0]), ("1,35·G + 1,05·Q (6.10a)", "6.10a", [1.35, 1.05]), ("G + 1,05·Q (6.10a)", "6.10a", [1.0, 1.05]),
          ("1,15·G + 1,5·Q (6.10b)", "6.10b", [1.1475, 1.5]), ("G + 1,5·Q (6.10b)", "6.10b", [1.0, 1.5])], [(x["nom"], x["facteurs"]) for x in co])
    co = CO.elu([G, QA], P, dict(ab, geotechnique=True))
    c("test 23 : une application géotechnique impose ξ = 1,00 — (6.10b) 1,35·G + 1,5·Q ; (6.10) ignore ξ",
      avec(co, [G, QA], G=1.35, Q=1.5) and all(x["expression"] == "6.10b" for x in avec(co, [G, QA], G=1.35, Q=1.5))
      and CO.elu([G, QA], P, {"geotechnique": True}) == CO.elu([G, QA], P), [x["nom"] for x in co])
    L = 6.0
    for g, qq, gagne, M_ab, M_610 in ((20.0, 2.0, "(6.10a)", (1.35 * 20 + 1.05 * 2), 1.35 * 20 + 1.5 * 2),
                                      (2.0, 20.0, "(6.10b)", (0.85 * 1.35 * 2 + 1.5 * 20), 1.35 * 2 + 1.5 * 20)):
        cas_b = [{"nom": "G", "nature": "permanent", "charges": [q(g, 0.0, L)]}, {"nom": "Q", "nature": "exploitation_A", "charges": [q(qq, 0.0, L)]}]
        r = MO.verifier(barre(profil="IPE 400", travees=[L], poids_propre=False, cas=cas_b, combinaisons=ab))
        r0 = MO.verifier(barre(profil="IPE 400", travees=[L], poids_propre=False, cas=cas_b))
        v, v0 = verif(r["methodes"]["ec3"], "E-My"), verif(r0["methodes"]["ec3"], "E-My")
        c(f"test 23, moteur : G = {g:g}, Q = {qq:g} kN/m sur 6 m — l'enveloppe garde la plus DÉFAVORABLE, {gagne} : M_Ed = "
          f"{M_ab * L * L / 8:.2f} kN·m (≤ (6.10) : {M_610 * L * L / 8:.2f})",
          v and v["combinaison"].endswith(gagne) and proche(v["valeur"], M_ab * L * L / 8, 1e-6) and proche(v0["valeur"], M_610 * L * L / 8, 1e-6),
          (v and (v["combinaison"], v["valeur"]), v0 and v0["valeur"]))

    # ---- 4. test 24 : la règle belge de courte durée
    co = CO.elu([G, W, S], P)
    c("test 24 : vent dominant + neige → ψ₀,neige = 0,3 (1,5·W + 0,45·S) ; neige dominante + vent → ψ₀,vent = 0,3 (0,45·W + 1,5·S)",
      avec(co, [G, W, S], G=1.35, W=1.5, S=0.45) and avec(co, [G, W, S], G=1.35, W=0.45, S=1.5)
      and not avec(co, [G, W, S], W=1.5, S=0.75) and not avec(co, [G, W, S], W=0.9, S=1.5), [x["nom"] for x in co])
    co = CO.elu([G, W, S], P, {"expression": "6.10ab"})
    c("test 24 : (6.10b) aussi — vent en tête, neige à 0,3",
      any(x["expression"] == "6.10b" for x in avec(co, [G, W, S], G=1.1475, W=1.5, S=0.45)), [x["nom"] for x in co])
    co = CO.elu([G, W, S], P, {"regle_courte_duree": False})
    c("test 24 : règle désactivée → ψ₀ du tableau, 0,5 et 0,6 (0,75·S et 0,9·W)",
      avec(co, [G, W, S], W=1.5, S=0.75) and avec(co, [G, W, S], W=0.9, S=1.5) and not any(x["courte_duree"] for x in co), [x["nom"] for x in co])
    co = CO.elu([G, QA, S], P)
    Qc = dict(QA, courte_duree=True)
    cc = CO.elu([G, Qc, S], P)
    c("test 24 : charge d'exploitation dominante + neige → ψ₀,neige = 0,5 ; 0,3 si elle est déclarée de courte durée",
      avec(co, [G, QA, S], G=1.35, Q=1.5, S=0.75) and not avec(co, [G, QA, S], Q=1.5, S=0.45)
      and avec(cc, [G, Qc, S], G=1.35, Q=1.5, S=0.45) and not avec(cc, [G, Qc, S], Q=1.5, S=0.75), [x["nom"] for x in cc])
    c("test 24 : charge d'exploitation d'ACCOMPAGNEMENT → ψ₀ du tableau (0,7 en catégorie A), jamais 0,3 — déclarée de courte durée ou non",
      avec(co, [G, QA, S], Q=1.05, S=1.5) and avec(cc, [G, Qc, S], Q=1.05, S=1.5) and not avec(cc, [G, Qc, S], Q=0.45), "")
    Q2 = [dict(Qc, nom="Q1"), dict(QA, nom="Q2")]
    cd = CO.elu([G] + Q2 + [S], P)
    c("une nature est de courte durée si TOUS ses cas présents le sont (Q1 déclaré, Q2 non : avec les deux, neige à 0,5)",
      avec(cd, [G] + Q2 + [S], Q1=1.5, Q2=1.5, S=0.75) and avec(cd, [G] + Q2 + [S], Q1=1.5, Q2=0.0, S=0.45), "")
    co = CO.elu([G, W, S], P, {"expression": "6.10ab"})
    c("test 24 : (6.10a) vent d'accompagnement principal (ψ₀ = 0,6) + neige → ψ₀,neige = 0,3 ; la neige principale → vent à 0,3",
      any(x["principale"] == "vent" and x["expression"] == "6.10a" for x in avec(co, [G, W, S], G=1.35, W=0.9, S=0.45))
      and any(x["principale"] == "neige" for x in avec(co, [G, W, S], G=1.35, W=0.45, S=0.75)), [x["nom"] for x in co if x["expression"] == "6.10a"])
    cas = [G, QA, S, W]
    co = CO.elu(cas, P)
    lim = avec(co, cas, G=1.35, Q=1.5, S=0.75, W=0.9)
    hyp = CO.hypotheses(None, cas)
    lignes_lim = [h for h in hyp if h.startswith("Limite")]
    c("test 24 : charge d'exploitation de longue durée dominante + neige + vent → 0,5 et 0,6, avec le message de limite (une ligne)",
      len(lim) == 1 and lim[0]["limite"] and not avec(co, cas, Q=1.5, S=0.45) and len(lignes_lim) == 1 and len(lignes_lim[0]) <= 120
      and not any(x["limite"] for x in CO.elu([G, QA, S], P)), (lim, lignes_lim))

    # ---- 5. γ_G par source
    L = 5.0
    g1, g2 = 10.0, 10.0
    deux = [{"nom": "G1", "nature": "permanent", "charges": [q(g1, 0.0, L)]}, {"nom": "G2", "nature": "permanent", "charges": [q(g2, L, 2 * L)]}]
    b2 = m("entree").normaliser(barre(profil="HEA 200", modele="continue", travees=[L, L], poids_propre=False, cas=deux))
    co = CO.elu(b2["cas"], P)
    w1, w2 = 1.35 * g1, 1.0 * g2
    RA = w1 * L / 2 - (w1 + w2) * L / 16                         # formulaire : deux travées égales, M_B = −(w₁ + w₂)·L²/16
    M_form = RA ** 2 / (2 * w1)
    sols = ST.resoudre(b2, b2["cas"])
    xs = [RA / w1] + [L * i / 200 for i in range(201)]
    par = {x["nom"]: max(ST.Combinaison(x["nom"], sols, x["facteurs"], [0.0, 0.0]).M(t) for t in xs) for x in co}
    c(f"continue de deux travées, G₁ sur la travée 1, G₂ sur la travée 2 : « 1,35·G1 + G2 » existe et donne le plus grand moment de "
      f"la travée 1, {M_form:.3f} kN·m (formulaire) — plus que 1,35·(G₁ + G₂) ({max(par.get('1,35·G', 0.0), 0.0):.3f})",
      "1,35·G1 + G2" in par and proche(par["1,35·G1 + G2"], M_form, 1e-9) and max(par, key=par.get) == "1,35·G1 + G2"
      and par["1,35·G1 + G2"] > par["1,35·G"] + 1.0, par)
    r = MO.verifier(barre(profil="HEA 200", modele="continue", travees=[L, L], poids_propre=False, cas=deux))
    env = r["statique"]["enveloppes"]["ec3"]
    M1 = max(Mx for x, Mx in zip(env["x"], env["M_max"]) if x < L)
    Mtous = max(ST.Combinaison("", sols, [1.35, 1.35], [0.0, 0.0]).M(x) for x in env["x"] if x < L)
    c("… et l'enveloppe ELU du moteur le garde (travée 1 : plus que la seule 1,35·(G₁ + G₂))", M1 > Mtous + 1.0, (M1, Mtous))
    pp = [{"nom": "Poids propre", "nature": "permanent", "N": 0.0, "charges": [q(0.4, 0.0, 6.0)]},
          {"nom": "G", "nature": "permanent", "N": 0.0, "charges": [dict(q(5.0, 0.0, 6.0), niveau=None)]},
          {"nom": "Q", "nature": "exploitation_A", "charges": [q(5.0, 0.0, 6.0)]}]
    c("deux cas de MÊME répartition (le poids propre et un G réparti sur toute la barre) : une seule source — trois combinaisons, "
      "« 1,35·G + 1,5·Q »",
      CO.sources(pp) == [[0, 1]] and [x["nom"] for x in CO.elu(pp, P)] == ["1,35·G", "1,35·G + 1,5·Q", "G + 1,5·Q"], [x["nom"] for x in CO.elu(pp, P)])
    haut = [pp[0], dict(pp[1], charges=[dict(q(5.0, 0.0, 6.0), niveau="haut")]), pp[2]]
    c("… pas à un autre niveau d'application (la hauteur de charge change M_cr) : deux sources", CO.sources(haut) == [[0], [1]], CO.sources(haut))
    # deux G de MÊME forme et de SIGNES contraires : réunis sous un seul γ, ils donneraient 1,35·(10 − 4) = 8,1 kN/m au lieu
    # de 1,35·10 − 1,00·4 = 9,5 (non conservatif)
    opp = [{"nom": "G1", "nature": "permanent", "charges": [q(10.0, 0.0, 6.0)]}, {"nom": "G2", "nature": "permanent", "charges": [q(-4.0, 0.0, 6.0)]}]
    try:
        r_opp = verif(MO.verifier(barre(profil="IPE 400", travees=[6.0], poids_propre=False, cas=opp))["methodes"]["ec3"], "E-My")
    except Exception as e:  # noqa: BLE001
        r_opp = {"combinaison": repr(e), "valeur": None}
    c("G₁ = +10 et G₂ = −4 kN/m sur la même portée : deux sources (rapport négatif) — « 1,35·G1 + G2 » existe et le moteur la "
      "garde : M_Ed = 9,5 × 6²/8 = 42,750 kN·m (et non 8,1 × 6²/8 = 36,450)",
      CO.sources(opp) == [[0], [1]] and facteurs(CO.elu(opp, P), opp, nom="1,35·G1 + G2") == {"G1": 1.35, "G2": 1.0}
      and r_opp is not None and r_opp["combinaison"] == "1,35·G1 + G2" and proche(r_opp["valeur"], 9.5 * 36 / 8, 1e-6), (CO.sources(opp), r_opp and r_opp["valeur"]))
    tri = [{"nom": "G1", "nature": "permanent", "charges": [{"type": "tri", "v": 6.0, "a": 0.0, "b": 6.0, "sens": "croissante"}]},
           {"nom": "G2", "nature": "permanent", "charges": [{"type": "tri", "v": 6.0, "a": 0.0, "b": 6.0, "sens": "decroissante"}]}]
    c("deux charges triangulaires de SENS contraires sur la même portée : deux sources (le sens est dans la répartition)",
      CO.sources(tri) == [[0], [1]], CO.sources(tri))
    nn = [{"nom": "G1", "nature": "permanent", "N": 0.0, "charges": [q(5.0, 0.0, 6.0)]}, {"nom": "G2", "nature": "permanent", "N": 50.0, "charges": [q(5.0, 0.0, 6.0)]},
          {"nom": "G3", "nature": "permanent", "N": 100.0, "charges": [q(10.0, 0.0, 6.0)]}]
    c("mêmes charges, l'un avec un effort normal : deux sources ; charges ET effort normal dans le même rapport (× 2) : une seule",
      CO.sources(nn) == [[0], [1, 2]], CO.sources(nn))
    var4 = [{"nom": "Q", "nature": "exploitation_B", "charges": [q(6.0, 0.0, 25.0)]}, {"nom": "S", "nature": "neige", "charges": [q(2.0, 0.0, 25.0)]},
            {"nom": "W", "nature": "vent", "charges": [q(1.5, 0.0, 25.0)]}, {"nom": "T", "nature": "temperature", "charges": [q(0.5, 0.0, 25.0)]}]
    perms = [{"nom": "Poids propre", "nature": "permanent", "charges": [q(0.4, 0.0, 25.0)]}] + \
            [{"nom": f"G{i}", "nature": "permanent", "charges": [q(1.0 + i, 5.0 * i, 5.0 * i + 5.0)]} for i in range(4)]
    n = [CO.controler(perms[:k] + var4) for k in (1, 2, 3)]
    c(f"le nombre de combinaisons, 4 natures, (6.10) : 1, 2, 3 sources → {n} (× 2^(n − 1), moins G seul à 1,00)",
      n == [65, 131, 263], n)
    msg = _message(lambda: CO.elu(perms[:4] + var4, P))
    c("au-delà du budget (512) : REFUS « au plus 3 cas permanents » — une ligne ; jamais une combinaison abandonnée en silence",
      msg == "527 combinaisons, 512 au plus : au plus 3 cas permanents de répartitions différentes" and len(msg) <= 120, msg)
    var6 = [dict(var4[0], nom="Q1"), dict(var4[0], nom="Q2", charges=[q(1.0, 5.0, 10.0)]), dict(var4[1], nom="S1"),
            dict(var4[1], nom="S2", charges=[q(1.0, 0.0, 12.0)]), var4[2], var4[3]]
    n6 = (CO.controler(perms[:1] + var6), CO.controler(perms[:1] + var6, {"expression": "6.10ab"}))
    c(f"une seule source est toujours admise : six cas variables de quatre natures → {n6[0]} combinaisons en (6.10), {n6[1]} en (6.10a)/(6.10b)",
      n6 == (321, 537), n6)

    # ---- 6. les noms et les hypothèses
    mix = CO.elu(deux, P)
    c("noms courts : « 1,35·G » quand les permanents ont le même γ ; chacun sinon (« 1,35·G1 + G2 », « G1 + 1,35·G2 ») ; ELS « G + Q »",
      [x["nom"] for x in mix] == ["1,35·G", "1,35·G1 + G2", "G1 + 1,35·G2"]
      and [x["nom"] for x in CO.service(deux + [QA])] == ["G", "G + Q"] and mix[1]["gamma_G"] == {"G1": "sup", "G2": "inf"},
      [x["nom"] for x in mix])
    c("un cas variable nommé « G » : les permanents s'écrivent chacun (pas deux « G »)",
      [x["nom"] for x in CO.service(deux + [dict(QA, nom="G")])] == ["G1 + G2", "G1 + G2 + G"], "")
    hyp = CO.hypotheses({"expression": "6.10ab"}, [G, dict(QA, courte_duree=True), S, W] + deux)
    longues = [h for h in hyp if len(h) > 120 or "\n" in h]
    c("hypothèses : une ligne chacune, 120 caractères au plus — expression et CC2, γ, ξ, ψ₀ employés, règle appliquée, actions "
      "déclarées, limite, γ_G par source (À VÉRIFIER), ce que « G » nomme",
      not longues and hyp[0].startswith("Combinaisons ELU : (6.10a) et (6.10b)") and "CC2" in hyp[0]
      and any("γ_G,sup = 1,35 ; γ_G,inf = 1,00 ; γ_Q = 1,5" in h for h in hyp) and any("ξ = 0,85 ; ξ·γ_G,sup = 1,1475" in h for h in hyp)
      and any(h.startswith("ψ₀ (tableau A1.1 ANB) : exploitation A 0,7 ; neige 0,5 ; vent 0,6") for h in hyp)
      and any(h.startswith("Règle de courte durée appliquée") for h in hyp) and "Déclarées de courte durée : « Q »." in hyp
      and any(h.startswith("γ_G,sup ou γ_G,inf par source permanente (À VÉRIFIER) : « G » ; « G1 » ; « G2 »") for h in hyp)
      and "G dans un nom, les cas permanents au même γ_G : G + G1 + G2." in hyp, longues or hyp)
    hd = CO.hypotheses({"regle_courte_duree": False, "geotechnique": True}, [G, S, W])
    c("hypothèses : (6.10) ; règle désactivée dite ; ξ absent en (6.10) ; sans neige ni vent, rien sur la règle",
      hd[0].startswith("Combinaisons ELU : (6.10),") and "Règle de courte durée désactivée : ψ₀ du tableau A1.1 ANB partout." in hd
      and not any("ξ" in h for h in hd) and not any("courte durée" in h for h in CO.hypotheses(None, [G, QA])), hd)
    hg = CO.hypotheses({"expression": "6.10ab", "geotechnique": True}, [G, E0, QA])
    c("hypothèses : géotechnique ξ = 1,00 ; un ψ₀ qui manque est dit « à fournir » et le calcul « bloqué »",
      "(6.10b) : ξ = 1,00 (application géotechnique) ; ξ·γ_G,sup = 1,35 ; γ_G,inf = 1,00." in hg
      and any("« E » (exécution) à fournir" in h for h in hg) and hg[-1].startswith("Combinaisons bloquées : ψ₀ à fournir"), hg)
    h_sans = CO.hypotheses(None, [G, QA, S])
    c("hypothèses : la règle « sans effet » dit la vraie raison — la neige est bien en tête (1,35·G + 1,05·Q + 1,5·S), mais ni neige "
      "ni vent ne suit une action de courte durée",
      "Règle de courte durée (tableau A1.1 ANB, note 3) sans effet ici : ni neige ni vent derrière une action de courte durée." in h_sans
      and not any("courte durée en tête" in h for h in h_sans) and avec(CO.elu([G, QA, S], P), [G, QA, S], G=1.35, Q=1.05, S=1.5) != [], h_sans)
    h_seule = CO.hypotheses(None, [G, E0])
    c("hypothèses : un ψ₀ d'exécution non saisi qui NE SERT PAS (l'exécution seule, en (6.10)) n'est pas « à fournir » — « non saisi, "
      "sans emploi ici », et le calcul aboutit",
      "ψ₀ (tableau A1.1 ANB) : « E » (exécution) non saisi, sans emploi ici." in h_seule and not any("à fournir" in h or "bloquées" in h for h in h_seule)
      and len(CO.elu([G, E0], P)) == 3, h_seule)
    doc = CO.__doc__
    c("la méthode ELS : G + ΣQ à 100 %, « décision du bureau du 16/11/2026 (13.2 point 5) » — plus « à confirmer : Q4 »",
      "décision du bureau du 16/11/2026" in doc and "13.2 point 5" in doc and "Q4" not in doc and len(CO.HYPOTHESE_SERVICE) <= 120, "")
    return C


def _message(f):
    try:
        f()
    except Exception as e:  # noqa: BLE001
        return str(e)
    return None


# ------------------------------------------------------------------ 7. les mutations
def _ordre():
    """Les modules du paquet et ceux de chacun qu'ils importent."""
    dep = {}
    for f in sorted(os.listdir(DOSSIER)):
        if f.endswith(".py") and f != "__init__.py":
            src = open(os.path.join(DOSSIER, f), encoding="utf-8").read()
            dep[f[:-3]] = set(re.findall(r"^\s*from \. import (\w+)", src, re.M)) | set(re.findall(r"^\s*from \.(\w+) import", src, re.M))
    return dep


DEP = _ordre()


def _dependants(cible):
    """Les modules qui emploient ``cible`` (directement ou non), chacun après ce qu'il importe."""
    touches, change = {cible}, True
    while change:
        change = False
        for k, v in DEP.items():
            if k not in touches and v & touches:
                touches.add(k)
                change = True
    touches.discard(cible)
    ordre, faits = [], {cible}
    while touches - set(ordre):
        for k in sorted(touches - set(ordre)):
            if DEP[k] & touches <= set(ordre) | faits:
                ordre.append(k)
    return ordre


def _installer(nom, mod):
    sys.modules[f"acier.barre.{nom}"] = mod
    setattr(PAQUET, nom, mod)
    for k in _dependants(nom):
        if f"acier.barre.{k}" in sys.modules:
            importlib.reload(sys.modules[f"acier.barre.{k}"])


def muter(fichier, ancien, nouveau):
    """Les contrôles qui rougissent quand ``ancien`` devient ``nouveau`` dans ``fichier`` ; None si le texte n'y est pas
    une seule fois. Le module d'origine est remis, ceux qui l'emploient rechargés, quoi qu'il arrive."""
    chemin = os.path.join(DOSSIER, fichier + ".py")
    src = open(chemin, encoding="utf-8").read()
    if src.count(ancien) != 1:
        return None
    orig = sys.modules[f"acier.barre.{fichier}"]
    mod = types.ModuleType(f"acier.barre.{fichier}")
    mod.__file__, mod.__package__ = chemin, "acier.barre"
    try:
        exec(compile(src.replace(ancien, nouveau), chemin, "exec"), mod.__dict__)
        _installer(fichier, mod)
        rouges = [n for n, ok, _ in controles() if not ok]
    except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
        rouges = [f"exception {type(e).__name__}: {e}"]
    finally:
        _installer(fichier, orig)
    return rouges


MUTATIONS = [
    # la table
    ("parametres", "ψ₁ de la neige remis à 0,2 (la valeur de l'EN)", '"psi_neige": {"valeur": (0.5, 0.0, 0.0)', '"psi_neige": {"valeur": (0.5, 0.2, 0.0)'),
    ("parametres", "ξ = 1 (6.10b sans réduction)", '"xi": {"valeur": 0.85', '"xi": {"valeur": 1.0'),
    ("parametres", "ψ₀ de courte durée 0,5", '"psi0_courte_duree": {"valeur": 0.3', '"psi0_courte_duree": {"valeur": 0.5'),
    ("parametres", "le vent redevient « à confirmer »",
     '"psi_vent": {"valeur": (0.6, 0.2, 0.0), "nom": "ψ, vent", "source": "NBN EN 1990 ANB:2021, tableau A1.1", "statut": POINTE}',
     '"psi_vent": {"valeur": (0.6, 0.2, 0.0), "nom": "ψ, vent", "source": "NBN EN 1990 ANB:2021, tableau A1.1", "statut": A_CONFIRMER}'),
    ("parametres", "n_pl « pointé » (plus aucun « à confirmer »)", '"nom": "γ_M de n_pl quand γ_M0 ≠ γ_M1", "statut": A_CONFIRMER',
     '"nom": "γ_M de n_pl quand γ_M0 ≠ γ_M1", "statut": POINTE'),
    ("parametres", "γ_M1 « à pointer »", '"gamma_M1": {"valeur": 1.00, "nom": "γ_M1", "source": "NBN EN 1993-1-1 ANB:2018, 6.1(1)B (bâtiments)", "statut": POINTE}',
     '"gamma_M1": {"valeur": 1.00, "nom": "γ_M1", "source": "NBN EN 1993-1-1 ANB:2018, 6.1(1)B (bâtiments)", "statut": A_POINTER}'),
    ("parametres", "ψ₀ de la catégorie E à 0,7 (au lieu de 1,0 ; non conservatif)", '"psi_E": {"valeur": (1.0, 0.9, 0.8)', '"psi_E": {"valeur": (0.7, 0.9, 0.8)'),
    ("parametres", "ψ₁ de la catégorie C à 0,5 (au lieu de 0,7)", '"psi_C": {"valeur": (0.7, 0.7, 0.6)', '"psi_C": {"valeur": (0.7, 0.5, 0.6)'),
    ("parametres", "ψ₂ de la catégorie H à 0,3 (au lieu de 0)", '"psi_H": {"valeur": (0.0, 0.0, 0.0)', '"psi_H": {"valeur": (0.0, 0.0, 0.3)'),
    ("parametres", "γ_M2 = 1,00 (au lieu de 1,25 ; non conservatif)", '"gamma_M2": {"valeur": 1.25', '"gamma_M2": {"valeur": 1.00'),
    ("parametres", "G sans arrondi (80 769 au lieu de 81 000)", '"G": {"valeur": 81000.0', '"G": {"valeur": 80769.0'),
    ("parametres", "γ_G,inf = 0,9 (la valeur d'une autre annexe)", '"gamma_G_inf": {"valeur": 1.00', '"gamma_G_inf": {"valeur": 0.90'),
    ("parametres", "un défaut de décision changé (η₁ plastique)", '"eta1_ame": {"valeur": "elastique"', '"eta1_ame": {"valeur": "plastique"'),
    # les natures et ψ₀
    ("combinaisons", "l'accompagnement reçoit γ_Q sans ψ₀", "                        f[i] = P[\"gamma_Q\"] * p\n", "                        f[i] = P[\"gamma_Q\"]\n"),
    ("combinaisons", "ψ₁ au lieu de ψ₀", 'return float(c["psi0"]) if cle == SAISI else P[cle][0]', 'return float(c["psi0"]) if cle == SAISI else P[cle][1]'),
    ("combinaisons", "deux cas de même nature : un seul en tête", 'if n == tete and expr != "6.10a":', 'if i == s[0] and expr != "6.10a":'),
    ("combinaisons", "un ψ₀ saisi demandé même quand il ne sert pas", 'and (o["expression"] == "6.10ab" or len(natures) > 1)]', "]"),
    ("combinaisons", "un ψ₀ saisi qui manque n'est jamais demandé", "    if manque:\n        raise AFournir(manque)", "    if False:\n        raise AFournir(manque)"),
    ("combinaisons", "la limite compte les CAS", 'n = len({cas[i]["nature"] for i in var})', "n = len(var)"),
    # (6.10a) et (6.10b)
    ("combinaisons", "(6.10b) sans ξ", 'g_sup = P["gamma_G_sup"] * (xi if expr == "6.10b" else 1.0)', 'g_sup = P["gamma_G_sup"]'),
    ("combinaisons", "géotechnique ignorée (ξ = 0,85)", '    ab = o["expression"] == "6.10ab"\n    xi = P["xi_geotechnique"] if o["geotechnique"] else P["xi"]',
     '    ab = o["expression"] == "6.10ab"\n    xi = P["xi"]'),
    ("combinaisons", "G_inf réduit par ξ dans (6.10b)", '(g_sup if sup else P["gamma_G_inf"])', '(g_sup if sup else P["gamma_G_inf"] * (xi if expr == "6.10b" else 1.0))'),
    ("combinaisons", "(6.10b) oubliée", '(("6.10a", "6.10b") if ab else ("6.10",))', '(("6.10a",) if ab else ("6.10",))'),
    ("combinaisons", "(6.10a) avec l'action en tête à γ_Q", 'if n == tete and expr != "6.10a":', "if n == tete:"),
    # la règle de courte durée
    ("combinaisons", "neige ↔ vent pas d'office", "    return nature in COURTES or all(", "    return all("),
    ("combinaisons", "la déclaration ignorée", 'return nature in COURTES or all(cas[i].get("courte_duree") is True for i in s if cas[i]["nature"] == nature)',
     "return nature in COURTES"),
    ("combinaisons", "un seul cas déclaré suffit (au lieu de tous)", 'or all(cas[i].get("courte_duree") is True', 'or any(cas[i].get("courte_duree") is True'),
    ("combinaisons", "0,3 aussi pour une charge d'exploitation", "if regle and n != tete and n in COURTES and tete_courte:", "if regle and n != tete and tete_courte:"),
    ("combinaisons", "l'option « règle désactivée » ignorée", "if regle and n != tete and n in COURTES and tete_courte:", "if n != tete and n in COURTES and tete_courte:"),
    ("combinaisons", "le message de limite jamais posé", "limite = bool(regle and tete is not None", "limite = bool(False and tete is not None"),
    # γ_G par source
    ("combinaisons", "un seul γ pour tous les cas permanents", 'gam = {i: (g_sup if sup else P["gamma_G_inf"]) for src, sup in zip(srcs, jeu) for i in src}',
     'gam = {i: (g_sup if jeu[0] else P["gamma_G_inf"]) for src, sup in zip(srcs, jeu) for i in src}'),
    ("combinaisons", "G favorable (1,00) oublié", "jeux = list(product((True, False), repeat=len(srcs)))", "jeux = [tuple(True for _ in srcs)]"),
    ("combinaisons", "les cas de même répartition jamais réunis", "            if _proportionnelles(_repartition(cas[src[0]]), r):", "            if False:"),
    ("combinaisons", "le niveau de la charge ignoré par la répartition", 'ch.get("niveau"), ch.get("introduction")', 'None, ch.get("introduction")'),
    ("combinaisons", "deux G de signes contraires réunis (k ≠ 0 au lieu de k > 0 ; non conservatif)", "return k > 0 and all(", "return k != 0 and all("),
    ("combinaisons", "le sens d'une charge triangulaire hors de la répartition",
     'None if bout is None else round(float(bout), 9), ch.get("sens"),', 'None if bout is None else round(float(bout), 9), None,'),
    ("combinaisons", "l'effort normal hors de la répartition", '    ajouter(("N",), c.get("N") or 0.0)\n', "\n"),
    ("combinaisons", "pas de budget", "plafond = max(COMBINAISONS_MAX, n_une)", "plafond = 10 ** 9"),
    # les noms et les hypothèses
    ("combinaisons", "les permanents au même γ écrits un par un", '        termes = [(f[perm[0]], "G")]', '        termes = [(f[i], cas[i]["nom"]) for i in perm]'),
    ("combinaisons", "les suffixes (6.10a) / (6.10b) perdus", 'SUFFIXES = {"6.10": "", "6.10a": " (6.10a)", "6.10b": " (6.10b)"}', 'SUFFIXES = {"6.10": "", "6.10a": "", "6.10b": ""}'),
    ("combinaisons", "l'hypothèse γ_G par source oubliée", "    if len(srcs) > 1:\n        out += _lignes(", "    if False:\n        out += _lignes("),
    ("combinaisons", "la règle « sans effet » redit l'ancienne raison, fausse (« aucune action de courte durée en tête »)",
     "(tableau A1.1 ANB, note 3) sans effet ici : ni neige ni vent derrière une action de courte durée.",
     "(tableau A1.1 ANB, note 3) : sans effet ici, aucune action de courte durée en tête."),
    ("combinaisons", "un ψ₀ non saisi toujours dit « à fournir », même quand il ne sert pas",
     '"à fournir" if cas[i]["nom"] in manque else "non saisi, sans emploi ici"', '"à fournir"'),
]


def main():
    print("== 1 à 6. la table, les natures, (6.10a)/(6.10b), la règle de courte durée, γ_G par source, les noms et les hypothèses")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 7. mutations")
    for fichier, nom, ancien, nouveau in MUTATIONS:
        rouges = muter(fichier, ancien, nouveau)
        if rouges is None:
            chk(f"mutation « {nom} » : le texte à muter est dans {fichier}.py, une seule fois", False, "texte absent ou répété")
        else:
            chk(f"mutation « {nom} » ({fichier}.py) : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    chk("après les mutations, le paquet est rendu : les contrôles repassent tous", all(ok for _, ok, _ in controles()))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
