# -*- coding: utf-8 -*-
"""Barres en acier — la QUALIFICATION EXTÉRIEURE de la stabilité Eurocode 3 (brief § 11.4) : les benchmarks publiés et
les recoupements avec les logiciels du bureau. docs/validation/barres_cas_ltbeam_scia.md, docs/acier/BARRES_ACIER.md § 5.7.

Critère (brief, règle 7) : écart ≤ 1 % : conforme ; 1 à 3 % : à expliquer (l'explication est écrite avec le cas) ;
plus de 3 % : à corriger ou à justifier. Une formule n'est jamais ajustée pour coller à un benchmark.

  1. ECCS N°119, « Rules for Member Stability in EN 1993-1-1 », repris dans l'aide de SCIA Engineer (« Benchmarks 17 à
     22 », section Steel code checks – EC3 – Benchmarks). Les pages ne sont pas joignables d'ici (domaines hors de la
     liste du réseau, 16/11/2026) : le benchmark 21 est comparé aux valeurs que le brief en donne ; les autres sont
     BLOQUÉS — données à fournir (aucune donnée n'est inventée). Un cas bloqué est imprimé « BLOQUÉ », il ne fait pas
     échouer la suite ; il est dans la liste « À VÉRIFIER » du rapport du lot B. La recherche (16/11/2026) ne rend, du benchmark 21,
     que ses résistances et ses élancements (contrôlés ci-dessous) et une remarque : la référence prend le cas général du déversement
     en méthode 1 et la formule corrigée de C_zz (AC:2009) — lue dans un extrait, pas sur la page : À VÉRIFIER.
  2. SCIA Engineer 26, annexe belge (S1, S2, S3, reçus le 16/11/2026) et LTBeamN (L17a) : les valeurs relevées.

    /usr/bin/python3.12 tests/test_barre_benchmarks.py
"""
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.barre import ec3 as EC  # noqa: E402
from acier.barre import ec3_deversement as DV  # noqa: E402
from acier.barre import entree as EN  # noqa: E402
from acier.barre import mcr as MC  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.barre import section as SE  # noqa: E402

OK, KO, BLOQUES = [], [], []
E_ = 210000.0


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info and not cond else ""))


def bloque(nom, il_faut):
    BLOQUES.append((nom, il_faut))
    print(f"  BLOQUÉ {nom} — données à fournir : {il_faut}")


def n(x, d=3):
    """Un nombre écrit à la française."""
    return f"{x:.{d}f}".replace(".", ",")


def ecart(x, ref):
    return x / ref - 1.0


def verif(r, id_):
    return next((v for v in r["verifications"] if v["id"] == id_), None)


def barre(nom, nuance, L, **k):
    return EC.verifier({"section": SE.du_catalogue(nom), "nuance": nuance, "axes": {"y": {"L": L}, "z": {"L": L}}, **k})


IL_FAUT = ("la page de l'aide de SCIA Engineer (capture ou PDF) : section, nuance, longueurs (flambement, déversement), appuis, charges "
           "et leur niveau, diagrammes de M_y et M_z, γ_M ; les résultats de la référence (χ_y, χ_z, χ_LT, M_cr, k_yy à k_zz, taux de (6.61) "
           "et (6.62)) et les remarques de SCIA sur les erreurs connues de la référence")


def main():
    print("== 1. ECCS N°119 — benchmarks 17 à 22 de l'aide de SCIA Engineer")
    print(f"    (pour chaque benchmark bloqué, il faut : {IL_FAUT})")
    for n_, quoi in ((17, "exemple 1"), (18, "exemple 2"), (19, "exemple 3"), (20, "exemple 4"), (22, "ossature de bâtiment")):
        bloque(f"benchmark {n_} (ECCS N°119, {quoi})", "la page")
    # benchmark 21 = exemple 5 : IPE 500 S235, 3,75 m, N_Ed = 500 kN, M_y,Ed = 200 kN·m, M_z,Ed = 25 kN·m triangulaire (données
    # relevées au lot 3 par la recherche ; la page elle-même n'est pas lisible) — valeurs de référence : brief § 11.4
    ref = {"mu_y": 1.00, "w_y": 1.138, "mu_z": 0.937, "w_z": 1.5, "C_mz0": 0.771, "M_cr0": 899.0, "lambda_0": 0.757, "a_LT": 0.998,
           "C_my0": 0.999, "b_LT": 0.043, "c_LT": 0.468, "d_LT": 0.347, "e_LT": 0.719, "C_yy": 0.981}
    # (mise à jour 1 : le harnais suit) les lectures de χ_LT ci-dessous sont faites au λ̄_LT du SOLVEUR (M_cr 1 227,2 kN·m) : la
    # variante de la décision n° 12 ; avec la formule de l'annexe E ANB (le défaut), M_cr = 1 215,2 et le module lit 0,809
    r = barre("IPE 500", "S235", 3750.0, NEd=500e3, MyEd=200e6, MzEd=25e6, parametres={"mcr_retenu": "solveur"},
              flexion={"y": {"diagramme": "ponctuel"}, "z": {"diagramme": "lineaire", "psi": 0.0}})
    A, D = r["grandeurs"]["annexe_A"], r["grandeurs"]["deversement"]
    site = dict(A, M_cr0=D["M_cr0"] / 1e6, lambda_0=D["lambda_0"])
    e = {k: ecart(site[k], x) for k, x in ref.items()}
    print("    benchmark 21 — " + " ; ".join(f"{k} {n(site[k], 4)} / {n(x, 3)} ({'+' if e[k] >= 0 else '−'}{n(abs(100 * e[k]), 2)} %)" for k, x in ref.items()))
    independants = ("mu_y", "w_y", "mu_z", "w_z", "C_mz0", "M_cr0", "lambda_0", "a_LT")
    chk("benchmark 21 (ECCS N°119, exemple 5) — les huit valeurs qui ne dépendent pas du diagramme de M_y : écart ≤ 0,5 % (conforme)",
        all(abs(e[k]) <= 0.005 for k in independants), {k: e[k] for k in independants})
    chk("benchmark 21 — C_my,0 0,999 ; b_LT 0,043 ; C_yy 0,981 conformes (≤ 1,2 %, l'arrondi de b_LT à deux chiffres) ; c_LT, d_LT, e_LT à "
        "moins de 3 % (aujourd'hui +2 % : à expliquer — tous ∝ 1/χ_LT, ils impliquent χ_LT ≈ 0,826 dans la référence, le module prend 0,811 "
        "(§ 6.3.2.3, courbe c, k_c = 0,86, f) ; 0,826 n'est retrouvé par aucune lecture essayée — contrôle suivant). Diagramme de M_y "
        "DÉDUIT de C_my,0 (charge concentrée à mi-portée) : À VÉRIFIER sur la page (Q8)",
        abs(e["C_my0"]) <= 0.001 and abs(e["b_LT"]) <= 0.012 and abs(e["C_yy"]) <= 0.001
        and all(abs(e[k]) <= 0.03 for k in ("c_LT", "d_LT", "e_LT")), {k: e[k] for k in ("C_my0", "b_LT", "C_yy", "c_LT", "d_LT", "e_LT")})
    # les lectures essayées pour le χ_LT ≈ 0,826 de la référence, au même λ̄_LT (celles que BARRES_ACIER.md § 5.7 cite)
    lb = D["lambda_LT"]
    lectures = {"§ 6.3.2.2 courbe b": (DV.chi_general(lb, 0.34), 0.812),
                "§ 6.3.2.3 courbe b, f (k_c 0,94)": (DV.chi_lamine(lb, 0.34, 0.2, 1.0, 0.94)["chi"], 0.836),
                "§ 6.3.2.3 courbe b, f (k_c 0,86)": (DV.chi_lamine(lb, 0.34, 0.2, 1.0, 0.86)["chi"], 0.870),
                "(0,4 ; 0,75) courbe c (k_c 1)": (DV.chi_lamine(lb, 0.49, 0.4, 0.75, 1.0)["chi"], 0.857),
                "(0,4 ; 0,75) courbe c, f (k_c 0,86)": (DV.chi_lamine(lb, 0.49, 0.4, 0.75, 0.86)["chi"], 0.919),
                "le module : § 6.3.2.3 courbe c, f (k_c 0,86)": (D["chi_LT"], 0.811)}
    chk("benchmark 21 — les lectures essayées du χ_LT de la référence (≈ 0,826) : § 6.3.2.2 courbe b 0,812 ; § 6.3.2.3 courbe b avec f "
        "0,836 (k_c 0,94) et 0,870 (0,86) ; (0,4 ; 0,75) courbe c 0,857 (k_c 1) et 0,919 (0,86) ; le module 0,811 — aucune à 0,5 % de 0,826",
        all(round(x, 3) == y for x, y in lectures.values()) and all(abs(ecart(x, 0.826)) > 0.005 for x, _ in lectures.values()),
        {k: round(x, 4) for k, (x, _) in lectures.items()})
    # les résultats de la référence relevés par la recherche (le texte de la page n'est pas lisible d'ici, 16/11/2026) : ils fixent la
    # section, la nuance, la longueur et la charge au centre de cisaillement (z_g = 0)
    F = r["grandeurs"]["flambement"]
    lus = {"N_c,Rd (kN)": (verif(r, "E-N")["limite"], 2715.0, 0.0005), "M_c,y,Rd (kN·m)": (verif(r, "E-My")["limite"], 516.0, 0.001),
           "M_c,z,Rd (kN·m)": (verif(r, "E-Mz")["limite"], 78.9, 0.001), "λ̄_y": (F["y"]["lambda_bar"], 0.195, 0.003),
           "λ̄_z": (F["z"]["lambda_bar"], 0.927, 0.001), "χ_z": (F["z"]["chi"], 0.644, 0.001)}
    chk("benchmark 21 — les résistances et les élancements de la référence : N_c,Rd 2 715 kN ; M_c,y,Rd 516 et M_c,z,Rd 78,9 kN·m ; λ̄_y 0,195 ; "
        "λ̄_z 0,927 ; χ_z 0,644 ; courbes a et b — conformes (≤ 0,3 %, l'arrondi de la référence)",
        all(abs(ecart(x, y)) <= t for x, y, t in lus.values()) and (F["y"]["courbe"], F["z"]["courbe"]) == ("a", "b"),
        {k: (round(x, 4), y) for k, (x, y, t) in lus.items()})
    bloque("benchmark 21 — taux de (6.61) et (6.62) de la référence (le module : "
           f"{n(verif(r, 'E-661')['taux'], 3)} et {n(verif(r, 'E-662')['taux'], 3)})", "les taux de la référence et ses χ_LT, M_cr ; le diagramme de M_y")

    print("== 2. Recoupement logiciel — SCIA Engineer 26 (annexe belge) et LTBeamN, valeurs reçues le 16/11/2026")
    # S1 : IPE 300 S235, 6 m, G 2,42 kN/m + 10 kN, Q 15 kN à mi-portée, charges sur la semelle haute (diagramme hors tableau 6.6)
    b = EN.nouvelle_barre("S1")
    b.update({"profil": "IPE 300", "nuance": "S235", "modele": "appuis", "travees": [6.0], "poids_propre": False, "niveau": "haut",
              "methodes": ["ec3"], "cas": [{"nom": "G", "nature": "permanent", "N": 0.0,
                                            "charges": [{"type": "q", "v": 2.42, "a": 0.0, "b": 6.0}, {"type": "P", "v": 10.0, "a": 3.0}]},
                                           {"nom": "Q", "nature": "exploitation_A", "N": 0.0, "charges": [{"type": "P", "v": 15.0, "a": 3.0}]}]})
    r1 = MO.verifier(b)["methodes"]["ec3"]
    D1 = r1["grandeurs"]["deversement"]
    t1 = verif(r1, "E-Mb")["taux"]
    chk(f"SCIA S1 — M_c,Rd {n(verif(r1, 'E-My')['limite'], 2)} (SCIA 147,65) ; déversement, semelle haute : M_cr {n(D1['M_cr'] / 1e6, 2)} kN·m "
        f"(SCIA 84,93 par C₁ 1,35 et C₂ 0,63 : −1,7 %, formule approchée) ; taux {n(t1, 3)} avec k_c = 1, f = 1 (hors tableau 6.6 ; SCIA 1,08 avec "
        "son k_c tiré de C₁) : écart 1,7 %, expliqué", round(t1, 3) == 1.099 and D1["kc"] == 1.0 and round(D1["M_cr"] / 1e6, 2) == 86.36
        and abs(ecart(verif(r1, "E-My")["limite"], 147.65)) < 0.001, (t1, D1["kc"], D1["M_cr"]))
    # S2 : HEB 300 S235, 5 m, N 1 400 kN, M_y 70 kN·m en tête (ψ = 0) — la borne de C_mi,0 retirée au lot B
    r2 = barre("HEB 300", "S235", 5000.0, NEd=1400e3, MyEd=70e6, VzEd=14e3, flexion={"y": {"diagramme": "lineaire", "psi": 0.0}})
    t61, t62 = verif(r2, "E-661")["taux"], verif(r2, "E-662")["taux"]
    chk(f"SCIA S2 — N_b,y,Rd {n(verif(r2, 'E-Nb-y')['limite'], 1)} et N_b,z,Rd {n(verif(r2, 'E-Nb-z')['limite'], 1)} kN (SCIA 3 105,47 ; 2 457,24) ; "
        f"(6.61) {n(t61, 3)} et (6.62) {n(t62, 3)} (SCIA 0,60 ; 0,65 : conforme au chiffre de SCIA ; avant le lot B, avec la borne : 0,631 ; 0,667)",
        abs(ecart(verif(r2, "E-Nb-y")["limite"], 3105.47)) < 0.001 and abs(ecart(verif(r2, "E-Nb-z")["limite"], 2457.24)) < 0.001
        and round(t61, 2) == 0.60 and round(t62, 2) == 0.65, (t61, t62))
    # S3 : IPE 360 S355, deux travées de 6 m, G 6 kN/m, Q1 et Q2 10 kN/m (catégorie B), charges sur la semelle haute
    b = EN.nouvelle_barre("S3")
    b.update({"profil": "IPE 360", "nuance": "S355", "modele": "continue", "travees": [6.0, 6.0], "poids_propre": False, "niveau": "haut",
              "methodes": ["ec3"], "cas": [{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 6.0, "a": 0.0, "b": 12.0}]},
                                           {"nom": "Q1", "nature": "exploitation_B", "N": 0.0, "charges": [{"type": "q", "v": 10.0, "a": 0.0, "b": 6.0}]},
                                           {"nom": "Q2", "nature": "exploitation_B", "N": 0.0, "charges": [{"type": "q", "v": 10.0, "a": 6.0, "b": 12.0}]}]})
    r3 = MO.verifier(b)["methodes"]["ec3"]
    D3 = r3["grandeurs"]["deversement"]
    chk(f"SCIA S3 — flexion sur l'appui {n(verif(r3, 'E-My')['taux'], 3)} et cisaillement {n(verif(r3, 'E-Vz')['taux'], 3)} (SCIA 0,29 ; 0,12) ; "
        f"déversement de la travée 1 (1,35·G + 1,5·Q1) : M_cr {n(D3['M_cr'] / 1e6, 2)} (SCIA 134,84 : −0,25 %) ; taux {n(verif(r3, 'E-Mb')['taux'], 3)} "
        "(SCIA 0,72, moment lu à ses nœuds : −1 %, expliqué)",
        round(verif(r3, "E-My")["taux"], 2) == 0.29 and round(verif(r3, "E-Vz")["taux"], 2) == 0.12 and abs(ecart(D3["M_cr"] / 1e6, 134.84)) < 0.01
        and round(verif(r3, "E-Mb")["taux"], 3) == 0.727, (D3["M_cr"], verif(r3, "E-Mb")["taux"]))
    # L17a : IPE 360, 6 m, q 23,1 kN/m sur la semelle haute (+180 mm) et −103,95 kN·m au bout droit ; LTBeamN : μ_cr 2,127, 221,1 kN·m
    L, q, MB = 6000.0, 23.1, -103.95e6

    def M17(x):
        return q * x * (L - x) / 2.0 + MB * x / L

    Mcr = MC.moment_critique(L, E_ * 1043e4, E_ * 313.6e9, 80769.0 * 37.32e4, M17, reparties=[(0.0, L, q, 180.0)], M_max=103.95e6)["M_cr"]
    chk(f"LTBeamN L17a — solveur du site avec les propriétés des fabricants (I_t 37,32 ; I_w 313,6·10³) et G = 80 769 : {n(Mcr / 1e6, 2)} kN·m "
        "contre 221,1 (conforme, ≤ 1 %) — la charge sur la semelle haute reste déstabilisante au droit de l'appui",
        abs(ecart(Mcr / 1e6, 221.1)) <= 0.01, Mcr / 1e6)

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s), {len(BLOQUES)} bloqué(s) — données à fournir")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
