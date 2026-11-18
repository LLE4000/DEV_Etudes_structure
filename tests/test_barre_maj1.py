# -*- coding: utf-8 -*-
"""Barres en acier — LA MISE À JOUR 1 (brief du 18/11/2026), sur son cas de contrôle « PRS-C4 » : PRS 300×30 / 1420×25 /
500×50, S235, deux appuis, L = 28,00 m, poids propre + Q = 20 kN/m sur la semelle haute, 4 maintiens de la semelle haute
à 5,60 / 11,20 / 16,80 / 22,40 m.

  Étape 1 (§ 1.5, tolérance 0,1 %) :
    1. les quatre réglages des DÉCISIONS BUREAU n° 11 (âme de classe 3) et n° 12 (M_cr retenu) : W_y, M_cr, λ̄_LT, χ_LT,
       M_b,Rd, taux — « élastique + solveur » redonne exactement le résultat d'avant ;
    2. W_pl,eff : ε_w, b, z_p, le trou, la flexion (49,0 %) ; chaque formule imprimée redonne le chiffre du moteur ;
    3. les deux lignes du M_cr, à l'écran et dans la note ;
    4. la même barre sans maintien : la formule prend le diagramme « réparti », z_g pondéré, et ne dépasse pas le solveur
       de plus de 2 % ;
    5. le garde-fou : une formule forcée au-dessus du solveur rend le solveur et sa ligne ;
    6. la méthode affichée : « ELS » sans la carte Eurocode 3, et inversement ; plus le bloc des grosses pastilles.

    /usr/bin/python3.12 -B tests/test_barre_maj1.py
"""
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.barre import ec3_deversement as DV  # noqa: E402
from acier.barre import formules as FO  # noqa: E402
from api_web import profil as P  # noqa: E402

OK, KO = [], []
MAINTIENS = (5.6, 11.2, 16.8, 22.4)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK  " if cond else "KO  ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def proche(a, b, tol=1e-3):
    return a is not None and abs(a - b) <= tol * abs(b)


def barre(maintiens=MAINTIENS, params=None):
    b = P.barre_neuve("B1")
    b.update(profil="PRS", prs={"b1": 300.0, "t1": 30.0, "hw": 1420.0, "tw": 25.0, "b2": 500.0, "t2": 50.0, "a": 10.0, "bords": "oxycoupes",
                                "nom": None},
             nuance="S235", travees=[28.0], poids_propre=True, niveau="haut",
             maintiens=[{"x": x, "genre": "haut", "neglige": False} for x in maintiens])
    b["cas"] = [dict(b["cas"][1], nom="Q", charges=[{"type": "q", "v": 20.0, "a": 0.0, "b": 28.0, "niveau": None}])]
    if params:
        b["parametres"] = dict(params)
    return b


def ec3(b, impression="deux"):
    etat = P.normaliser(P.etat_de([b], impression=impression))
    return P.calculer(etat["elements"][0])["methodes"]["ec3"]


def verif(e, ident):
    return next((v for v in e["verifications"] if v["id"] == ident), None)


def blanc(t):
    return re.sub(r"[\s  ]+", " ", t)


def main():
    print("== 1. les quatre réglages (§ 1.5)")
    TABLE = (({}, 34736.0, 8704.5, 0.9263, 0.5062, 3780.1, 96.9),
             ({"mcr_retenu": "solveur"}, 34736.0, 10716.7, 0.8348, 0.5587, 4172.2, 87.8),
             ({"ame_classe_3": "elastique"}, 24367.7, 8704.5, 0.7758, 0.5947, 3115.6, 117.6),
             ({"ame_classe_3": "elastique", "mcr_retenu": "solveur"}, 24367.7, 10716.7, 0.6992, 0.6437, 3372.2, 108.6))
    for prm, W, Mcr, lb, chi, MbRd, taux in TABLE:
        e = ec3(barre(params=prm))
        D, v = e["grandeurs"]["deversement"], verif(e, "E-Mb")
        nom = " + ".join(f"{k} = {x}" for k, x in prm.items()) or "défauts (classe 2 efficace + formule)"
        chk(f"{nom} : W_y {W}, M_cr {Mcr}, λ̄_LT {lb}, χ_LT {chi}, M_b,Rd {MbRd}, taux {taux} %",
            proche(D["W_y"] / 1e3, W) and proche(D["M_cr"] / 1e6, Mcr) and proche(D["lambda_LT"], lb) and proche(D["chi_LT"], chi)
            and proche(D["M_b_Rd"] / 1e6, MbRd) and proche(v["taux"] * 100, taux) and proche(v["valeur"], 3662.7),
            (D["W_y"] / 1e3, D["M_cr"] / 1e6, D["lambda_LT"], D["chi_LT"], D["M_b_Rd"] / 1e6, v["taux"]))

    print("== 2. W_pl,eff (§ 5.5.2(11), § 6.2.2.4)")
    e = ec3(barre())
    G, v = e["grandeurs"], verif(e, "E-My")
    k = G["classe_2_efficace"] or {}
    chk("ε_w = 1,02198 ; b = 510,99 mm ; z_p = 1 038,02 mm ; trou de 16,04 mm",
        proche(k.get("eps_w"), 1.02198, 1e-5) and proche(k.get("b"), 510.99, 1e-5) and proche(k.get("z_p"), 1038.02, 1e-5)
        and proche(k.get("trou"), 16.04, 1e-3), k)
    chk("flexion : 3 662,7 / 7 468,2 = 49,0 % ; classe affichée « 3 → classe 2 efficace »",
        proche(v["valeur"], 3662.7) and proche(v["limite"], 7468.2) and proche(v["taux"] * 100, 49.0)
        and G["classe_affichee"] == "3 → classe 2 efficace" and any("classe 3 → classe 2 efficace" in h for h in e["hypotheses"]),
        (v["valeur"], v["limite"], G.get("classe_affichee")))
    ecarts = [(b["cle"], b["resultat"], FO.evaluer(b)) for b in e["calcul"] if abs(FO.evaluer(b) - b["resultat"]) > 1e-9 * max(1.0, abs(b["resultat"]))]
    chk("chaque formule imprimée (W_pl,eff, z_p, b, M_cr de la formule…) redonne le chiffre du moteur",
        not ecarts and {"E-My/W_pl_eff", "E-My/z_p_eff2", "E-My/b_eff2", "E-Mb/M_cr_E1"} <= {b["cle"] for b in e["calcul"]}, ecarts)
    hors = ec3(barre(params={"ame_classe_3": "elastique"}))
    chk("variante « élastique » : W_el, classe 3, aucune formule de W_pl,eff",
        hors["grandeurs"]["classe_2_efficace"] is None and hors["grandeurs"]["classe_affichee"] == "3"
        and not any("eff2" in b["cle"] for b in hors["calcul"]), hors["grandeurs"].get("classe_affichee"))

    print("== 3. les deux lignes du M_cr")
    m = [blanc(t) for t in verif(e, "E-Mb")["messages"]]
    chk("« M_cr retenu : formule, tronçon 11,20 à 16,80 m, moment constant = 8 704,5 kN·m » puis « Solveur (poutre entière) : "
        "10 716,7 kN·m (+23 %) »",
        m[:2] == ["M_cr retenu : formule, tronçon 11,20 à 16,80 m, moment constant = 8 704,5 kN·m.",
                  "Solveur (poutre entière) : 10 716,7 kN·m (+23 %)."], m[:3])
    s = [blanc(t) for t in verif(ec3(barre(params={"mcr_retenu": "solveur"})), "E-Mb")["messages"]]
    chk("réglage « solveur » : « M_cr retenu : solveur… » puis la formule et son écart",
        s[:2] == ["M_cr retenu : solveur (poutre entière) = 10 716,7 kN·m.",
                  "Formule, tronçon 11,20 à 16,80 m, moment constant : 8 704,5 kN·m (−19 %)."], s[:3])

    print("== 4. sans maintien")
    e0 = ec3(barre(maintiens=()))
    D0 = e0["grandeurs"]["deversement"]
    F = D0["mcr"]["formule"]
    t0 = F["troncons"][0] if F and F["troncons"] else {}
    chk("la formule : diagramme « réparti », z_g pondéré (entre le centre de gravité et la semelle haute), retenue",
        D0["mcr"]["retenu"] == "annexe_E" and t0.get("diagramme") == "reparti" and t0.get("C1") == 1.12 and t0.get("z_g") is not None, t0)
    chk("elle ne dépasse pas le solveur de plus de 2 %", t0 and t0["M_cr"] <= 1.02 * D0["mcr"]["solveur"], (t0.get("M_cr"), D0["mcr"]["solveur"]))
    cles0 = [b["cle"] for b in e0["calcul"]]
    chk("la formule retenue n'a qu'UN bloc « M_cr » de l'annexe E (l'ancien contrôle (E.1) du PRS ne le réécrit pas)",
        cles0.count("E-Mb/M_cr_E1") == 1, cles0)

    print("== 5. le garde-fou")
    garde = DV.GARDE_FORMULE
    DV.GARDE_FORMULE = 0.5                                    # (forcé : la formule passe « au-dessus » du solveur)
    try:
        P._RESULTATS.clear()                                  # (le calcul de l'écran est gardé en mémoire : le refaire)
        ef = ec3(barre())
    finally:
        DV.GARDE_FORMULE = garde
        P._RESULTATS.clear()
    Df, mf = ef["grandeurs"]["deversement"], [blanc(t) for t in verif(ef, "E-Mb")["messages"]]
    chk("une formule au-dessus du solveur : le solveur est retenu, avec sa ligne",
        Df["mcr"]["retenu"] == "solveur" and proche(Df["M_cr"] / 1e6, 10716.7) and mf[0] == "Formule au-dessus du solveur : solveur retenu.", mf[:2])

    print("== 6. la méthode affichée (§ 1.4)")
    for imp, avec, sans in (("historique", "ELS", "Eurocode 3"), ("ec3", "Eurocode 3", "ELS"), ("deux", "ELS", None)):
        etat = P.normaliser(P.etat_de([barre()], impression=imp))
        html = P.resultat(etat)["resultat"]
        titres = re.findall(r'class="et-bloc[^"]*".*?<h3[^>]*>(.*?)</h3>', html)
        chk(f"méthode affichée « {imp} » : la carte {avec}" + (f", jamais celle {'de l' if sans.startswith('E') else 'des'} {sans}" if sans else
                                                                ", les deux cartes") + " ; plus de bloc des grosses pastilles",
            avec in html and (sans is None or sans not in html) and (sans is not None or "Eurocode 3" in html) and "pf-verdicts" not in html,
            titres[:4])
    etat = P.normaliser(P.etat_de([barre()], impression="ec3"))
    chk("méthode affichée « Eurocode 3 » : le taux de la barre est celui de l'Eurocode 3 (97 %), pas celui de l'ELS",
        str(P.resultat(etat)["resume"]["taux"]) == "97", P.resultat(etat)["resume"])

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
