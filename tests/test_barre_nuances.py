# -*- coding: utf-8 -*-
"""Barres en acier — la SECTION du lot C, bloc C1b : les nuances du module (S420 compris), la norme produit (N/NL ou
M/ML), f_u et son origine, les U et les cornières (compression simple de la méthode ELS, en efforts imposés), et la
variante de la DÉCISION BUREAU n° 3 (A_v,y d'un I ou H laminé). Brief v12 § 3, 4.1, 7.1, 7.5, 13.2 point 3.

  1. NUANCES ET NORME PRODUIT (acier/barre/nuances.py) : S420 N/NL, plat de 30 mm → 400 N/mm² ; à 70 mm, M/ML 380 et
     N/NL 370 ; S460 sans norme produit inchangé (tableau 3.4 ANB) ; S460 M/ML au tableau 3.5 ANB ; « M » sans effet sur
     S235 à S355 ; le défaut N/NL au plus M/ML à toute épaisseur du domaine ; les listes des autres modules inchangées ;
     les refus du domaine mot pour mot ceux du module Flambement (copie nommée, liée ici).
  2. LE RÉSULTAT EUROCODE 3 rend la norme produit qui sert, f_u, son origine et son tableau ; la norme produit est dite
     une fois, avec f_y. La méthode ELS : f_y nominale, S420 « hors domaine » comme S460.
  3. CORNIÈRES ET U : la cornière L 60 × 60 × 8 de la doc (p. 8 : λ̄ 1,820, φ_c 0,236, P_adm 33,4 kN), courbe c ;
     chaque axe d'une cornière avec la plus grande longueur saisie ; l'Eurocode 3 « non couvert » en une ligne ; un moment
     sur un U : non couvert ; la barre chargée refusée.
  4. DÉCISION BUREAU n° 3 : A_v,y d'un I ou H laminé = 2·b·t_f (défaut de la table) ; la variante A − h_w·t_w et son
     V_pl,y,Rd rendues pour information ; le défaut changé dans la table change V_pl,y,Rd ; I soudé, IPN, PRS : aucune.
  5. S420 AILLEURS QUE DANS LA SECTION : un raidisseur d'une barre en S420 (reprise : moteur.verifier levait « nuance
     inconnue : S420 », raidisseurs.materiau lisant la liste du module Flambement) ; sa norme produit transmise au plat ;
     un PRS en S420 (reprise : prs.section_prs le refusait, « S235, S275, S355 ou S460 »).
  6. MUTATIONS : chaque altération plausible du texte (nuances.py, ec3_section.py, entree.py, historique.py,
     raidisseurs.py, la table des paramètres) ou des lignes de la table des aciers fait rougir au moins un contrôle.

    /usr/bin/python3.12 -B tests/test_barre_nuances.py
"""
import importlib
import math
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import acier.barre as PAQUET  # noqa: E402
from acier import materiaux as MA  # noqa: E402
from acier.barre import ec3 as _ec3  # noqa: E402,F401 — chargés une fois : la mutation les recharge
from acier.barre import moteur as _moteur  # noqa: E402,F401
from acier.barre import nuances as _nuances  # noqa: E402,F401
from acier.barre import section as SE  # noqa: E402
from acier.flambement import normes as NO  # noqa: E402

OK, KO = [], []
DOSSIER = os.path.join(RACINE, "acier", "barre")
R3 = math.sqrt(3.0)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def m(nom):
    """Le module du paquet tel qu'il est MAINTENANT (une mutation le remplace) : jamais une référence gardée."""
    return sys.modules[f"acier.barre.{nom}"]


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def refus(f, *a, **k):
    """Le message du Refus (ou de l'erreur de domaine) levé par ``f`` ; None s'il n'en lève pas."""
    try:
        f(*a, **k)
    except (m("resultat").Refus, NO.HorsDomaine) as e:
        return str(e)
    return None


def verif(r, id_):
    return next((v for v in r["verifications"] if v["id"] == id_), None)


def essai(f, *a, **k):
    """(le résultat, None) ; ou (None, « Type : message ») si ``f`` lève — un contrôle rougit au lieu de tout arrêter."""
    try:
        return f(*a, **k), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__} : {e}"


def efforts(profil, nuance="S235", N=0.0, My=0.0, Ly=2.0, Lz=2.0, **autres):
    """Une barre en efforts imposés (kN, kN·m, m) : les mêmes efforts de service et de calcul (×1,5)."""
    e = {"N": N, "My": My}
    return dict({"profil": profil, "nuance": nuance, "mode": "efforts", "efforts": {"service": e, "elu": {k: 1.5 * v for k, v in e.items()}},
                 "flambement": {"y": {"L": Ly}, "z": {"L": Lz}}}, **autres)


# ------------------------------------------------------------------ 1 à 5. les contrôles (la même liste sert aux mutations)
def controles():
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    NU, ES, EC, HI, MO = m("nuances"), m("ec3_section"), m("ec3"), m("historique"), m("moteur")

    # ---- 1. nuances et norme produit
    c("les nuances du module Barres : S235, S275, S355, S420, S460 ; celles des autres modules ne bougent pas (table, Flambement, "
      "cornière, Catalogue : S235, S275, S355, S460)",
      NU.NUANCES == ("S235", "S275", "S355", "S420", "S460") and MA.NUANCES == ("S235", "S275", "S355", "S460")
      and list(NO.NUANCES) == list(MA.NUANCES) and list(sys.modules["acier.corniere.entree"].NUANCES) == list(MA.NUANCES)
      and sys.modules["api_web.flambement"].NUANCES == list(MA.NUANCES), (NU.NUANCES, MA.NUANCES))
    c("S420 N/NL (défaut), plat de 30 mm → 400 N/mm² (tableau 3.4 ANB, EN 10025-3:2004)",
      NU.fy("S420", 30.0) == 400.0 and NU.source("S420", 30.0) == "tableau 3.4 ANB (EN 10025-3:2004, S420N), 16 < t ≤ 40 mm",
      (NU.fy("S420", 30.0), NU.source("S420", 30.0)))
    c("S420 de 70 mm : M/ML 380 N/mm² (tableau 3.5 ANB, EN 10025-4:2004), N/NL 370",
      NU.fy("S420", 70.0, "lamine", "M") == 380.0 and NU.fy("S420", 70.0) == 370.0
      and NU.tableau("S420", "lamine", "M") == "tableau 3.5 ANB", (NU.fy("S420", 70.0, "lamine", "M"), NU.fy("S420", 70.0)))
    tt = (5.0, 16.0, 16.5, 30.0, 40.0, 41.0, 63.0, 64.0, 80.0)
    c("S460 sans norme produit : inchangé — la ligne N/NL du tableau 3.4 ANB, comme le module Flambement",
      all(NU.fy("S460", t) == MA.fy("S460", t) == NO.limite_elastique("S460", t) for t in tt)
      and NU.source("S460", 30.0).startswith("tableau 3.4 ANB (EN 10025-3:2004, S460N)"), NU.source("S460", 30.0))
    c("S460 M/ML : tableau 3.5 ANB (EN 10025-4:2004, S460M) ; f_u de 70 mm 510 (N/NL : 540)",
      NU.source("S460", 30.0, "lamine", "M").startswith("tableau 3.5 ANB (EN 10025-4:2004, S460M)")
      and NU.fu("S460", 70.0, "lamine", "M") == 510.0 and NU.fu("S460", 70.0) == 540.0, NU.source("S460", 30.0, "lamine", "M"))
    g355 = ES.verifier({"section": SE.du_catalogue("HEB 300"), "nuance": "S355", "norme_produit": "M", "NEd": 1000e3})["grandeurs"]
    c("« M » donné avec S235, S275 ou S355 : sans effet (EN 10025-2) — norme effective vide, mêmes f_y ; le résultat rend la norme qui SERT",
      all(NU.norme_effective(n, "M") is None and NU.fy(n, t, "lamine", "M") == NU.fy(n, t) for n in ("S235", "S275", "S355") for t in tt)
      and NU.norme_effective("S420", "M") == "M" and NU.norme_effective("S460", None) is None
      and g355["norme_produit"] is None and g355["fy"] == 345.0 and "S355JR" in g355["fy_source"], {k: g355.get(k) for k in ("norme_produit", "fy", "fy_source")})
    c("refus : une nuance inconnue (les cinq nuances dites), une norme produit inconnue, S420 formé à froid N/NL (valeur à fournir)",
      refus(NU.fy, "S690", 10.0) == "nuance inconnue : 'S690' (S235, S275, S355, S420 ou S460)"
      and (refus(NU.fy, "S420", 10.0, "lamine", "N") or "").startswith("norme produit")
      and refus(NU.fy, "S420", 10.0, "creux_froid") == "S420 creux formé à froid : aucune ligne dans les tableaux de l'ANB (valeur à fournir)",
      (refus(NU.fy, "S690", 10.0), refus(NU.fy, "S420", 10.0, "creux_froid")))
    # la borne du domaine : la copie nommée des refus du module Flambement, mot pour mot (R2 : un test les lie)
    ecarts = []
    for n in NO.NUANCES:
        for p in ("lamine", "creux_chaud", "creux_froid"):
            for t in (8.0, 16.0, 30.0, 40.0, 40.5, 66.0, 80.0, 80.04, 81.0, 89.9):
                try:
                    a = ("v", NO.limite_elastique(n, t, produit=p))
                except NO.HorsDomaine as e:
                    a = ("r", str(e))
                try:
                    b = ("v", NU.fy(n, t, p))
                except m("resultat").Refus as e:
                    b = ("r", str(e))
                if a != b:
                    ecarts.append((n, p, t, a, b))
    c("les quatre nuances communes : nuances.fy rend la valeur ou le refus du module Flambement, mot pour mot (80 mm, 40 mm à froid, 65 mm)",
      not ecarts, ecarts[:3])
    nm = [(n, p, t) for n, p, tmax in (("S420", "lamine", 80.0), ("S460", "lamine", 80.0), ("S460", "creux_froid", 40.0))
          for t in [x / 4.0 for x in range(1, int(4 * tmax) + 1)] if not NU.fy(n, t, p) <= NU.fy(n, t, p, "M")]
    c("le défaut N/NL est le côté sûr pour f_y : au plus M/ML à toute épaisseur du domaine (S420, S460 ; laminés et formés à froid)", not nm, nm[:3])

    # ---- 2. le résultat de l'Eurocode 3
    r = ES.verifier({"section": SE.du_catalogue("HEB 300"), "nuance": "S420", "NEd": 1000e3})
    g = r["grandeurs"]
    hyp = " ".join(r["hypotheses"])
    c("HEB 300 S420 : f_y = 400 (t_f 19 mm, tableau 3.4 ANB, S420N) ; f_u = 520 et son origine (tableau 3.4 ANB, t ≤ 100 mm) rendues ; "
      "norme produit N/NL (vide)",
      g["fy"] == 400.0 and g["fy_source"] == "tableau 3.4 ANB (EN 10025-3:2004, S420N), 16 < t ≤ 40 mm" and g["norme_produit"] is None
      and g["fu"] == 520.0 and g["fu_source"] == "tableau 3.4 ANB (EN 10025-3:2004, S420N), t ≤ 100 mm" and g["fu_ref"] == "tableau 3.4 ANB",
      {k: g.get(k) for k in ("fy", "fy_source", "norme_produit", "fu", "fu_source", "fu_ref")})
    c("la norme produit est dite une fois, avec f_y (« EN 10025-3 » une seule fois dans les hypothèses)", hyp.count("EN 10025-3") == 1, hyp)
    hl = SE.du_catalogue("HL 920x725")
    rn = ES.verifier({"section": hl, "nuance": "S420", "NEd": 1000e3})["grandeurs"]
    rm = ES.verifier({"section": hl, "nuance": "S420", "norme_produit": "M", "NEd": 1000e3})["grandeurs"]
    c("HL 920x725 (t_f 68,1 mm) S420 : N/NL f_y 370, f_u 520 ; M/ML f_y 380, f_u 480 (tableau 3.5 ANB, 63 < t ≤ 80 mm), norme « M » rendue ; "
      "semelles 370 → 380, âme (38,1 mm) 400 : chaque paroi dans la ligne de sa norme produit",
      (rn["fy"], rn["fu"], rn["norme_produit"]) == (370.0, 520.0, None) and (rm["fy"], rm["fu"], rm["norme_produit"]) == (380.0, 480.0, "M")
      and (rn["fy_parois"]["semelle"], rm["fy_parois"]["semelle"], rm["fy_parois"]["ame"]) == (370.0, 380.0, 400.0)
      and rm["fy_source"] == "tableau 3.5 ANB (EN 10025-4:2004, S420M), 63 < t ≤ 80 mm" and rm["fu_ref"] == "tableau 3.5 ANB",
      ({k: rn.get(k) for k in ("fy", "fu", "norme_produit")}, {k: rm.get(k) for k in ("fy", "fu", "norme_produit", "fy_source")}))
    r460 = EC.verifier({"section": SE.du_catalogue("HEB 300"), "nuance": "S460", "norme_produit": "M", "NEd": 1000e3,
                        "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}}})["grandeurs"]
    c("la barre (ec3.verifier) rend aussi la norme produit et f_u : HEB 300 S460 M/ML, f_y 440, f_u 540 (tableau 3.5 ANB)",
      (r460.get("fy"), r460.get("fu"), r460.get("norme_produit"), r460.get("fu_ref")) == (440.0, 540.0, "M", "tableau 3.5 ANB"),
      {k: r460.get(k) for k in ("fy", "fu", "norme_produit", "fu_ref")})
    c("refus de la section : une nuance hors du module (les cinq dites), une norme produit inconnue",
      (refus(ES.verifier, {"section": hl, "nuance": "S690", "NEd": 1e3}) or "").startswith("nuance inconnue : 'S690' (S235, S275, S355, S420 ou S460")
      and (refus(ES.verifier, {"section": hl, "nuance": "S420", "norme_produit": "N", "NEd": 1e3}) or "").startswith("norme produit"))
    rs = MO.verifier(efforts("IPE 300", "S420", N=100.0))
    hs, es = rs["methodes"]["historique"], rs["methodes"]["ec3"]
    c("IPE 300 S420 : méthode ELS à la f_y NOMINALE 420, « hors domaine » (nuance hors AE235 à AE355, comme S460) ; Eurocode 3 à 420 "
      "(t_f 10,7 mm, tableau 3.4 ANB)",
      hs["grandeurs"]["fy"] == 420.0 and hs["statut"] == m("resultat").HORS_DOMAINE and any(a["code"] == "nuance" for a in hs["alertes"])
      and es["grandeurs"]["fy"] == 420.0 and es["grandeurs"]["fy_ref"] == "tableau 3.4 ANB", (hs["statut"], hs["grandeurs"]["fy"], es["grandeurs"].get("fy")))

    # ---- 3. cornières et U
    corniere = SE.saisie("L", "lamine", nom="L 60 × 60 × 8", A=903.0, Iy=903 * 22.9 ** 2, Iz=903 * 11.7 ** 2, iy=22.9, iz=11.7)
    gd = HI.verifier({"section": corniere, "fy": 235.0, "P": 30e3, "axes": {"y": {"L": 2000.0}, "z": {"L": 2000.0}}})["grandeurs"]["z"]
    c("doc p. 8, cornière L 60 × 60 × 8 comprimée (2 m, i_v 11,7 mm) : courbe c, λ̄ = 1,820, φ_c = 0,236, P_adm = 33,4 kN",
      gd["courbe"] == "c" and round(gd["lambda_bar"], 3) == 1.820 and round(gd["phi"], 3) == 0.236 and round(gd["P_adm"] / 1e3, 1) == 33.4,
      (gd["courbe"], gd["lambda_bar"], gd["phi"], gd["P_adm"]))
    rl = MO.verifier(efforts("L 60x60x8", N=20.0, Ly=2.0, Lz=1.0))
    hl_, el_ = rl["methodes"]["historique"], rl["methodes"]["ec3"]
    gz, gy = hl_["grandeurs"]["z"], hl_["grandeurs"]["y"]
    c("L 60x60x8 du catalogue en efforts imposés, L_y = 2 m, L_z = 1 m : courbe c sur les deux axes, chacun à la plus grande longueur "
      "(2 m) — v-v, le plus petit rayon, gouverne ; l'hypothèse dite une fois",
      (gy["courbe"], gz["courbe"]) == ("c", "c") and gy["Lf"] == gz["Lf"] == 2000.0 and gz["nom"] == "v-v"
      and hl_["gouvernant"] == "H-N-z" and hl_["hypotheses"].count(HI.MESSAGE_CORNIERE) == 1, (gy["Lf"], gz["Lf"], hl_["gouvernant"]))
    c("cornière : Eurocode 3 non couvert, en une ligne « Eurocode 3 : non couvert pour une cornière (v1). »",
      el_["statut"] == m("resultat").NON_COUVERT and [v["id"] for v in el_["verifications"]] == ["E-SECTION"]
      and [a["message"] for a in el_["alertes"]] == ["Eurocode 3 : non couvert pour une cornière (v1)."], el_["alertes"])
    ru = MO.verifier(efforts("UPN 200", N=100.0, My=5.0, Ly=3.0, Lz=3.0))
    hu, eu = ru["methodes"]["historique"], ru["methodes"]["ec3"]
    c("un moment sur un U (UPN 200) : la flexion de la méthode ELS non couverte, l'Eurocode 3 non couvert en une ligne",
      hu["statut"] == m("resultat").NON_COUVERT and verif(hu, "H-SIGMA")["statut"] == m("resultat").NON_COUVERT
      and eu["statut"] == m("resultat").NON_COUVERT and [a["message"] for a in eu["alertes"]] == ["Eurocode 3 : non couvert pour un U (v1)."],
      (hu["statut"], eu["alertes"]))
    msg = m("entree").MSG_EFFORTS_SEULS
    c("une cornière ou un U en barre chargée : refusé en une ligne (efforts imposés seulement) ; un I y reste admis",
      refus(m("entree").normaliser, {"profil": "L 60x60x8", "nuance": "S235"}) == msg
      and refus(m("entree").normaliser, {"profil": "UPE 200", "nuance": "S235"}) == msg
      and refus(m("entree").normaliser, {"profil": "IPE 300", "nuance": "S420"}) is None and len(msg) <= 120, msg)

    # ---- 4. la DÉCISION BUREAU n° 3
    heb = SE.du_catalogue("HEB 200")
    geo = heb["A"] - (heb["h"] - 2.0 * heb["tf"]) * heb["tw"]
    r = ES.verifier({"section": heb, "nuance": "S235", "VyEd": 100e3})
    g, va = r["grandeurs"], r["grandeurs"].get("A_v_variante") or {}
    c("HEB 200 laminé, défaut de la table : A_v,y = 2·b·t_f = 6 000 mm², V_pl,y,Rd = 814,1 kN ; la variante A − h_w·t_w = 6 278 mm², "
      "V_pl,y,Rd = 851,8 kN, rendue pour information",
      proche(g["A_v"]["y"], 6000.0) and round(g["V_pl_y"] / 1e3, 1) == 814.1 and round(verif(r, "E-Vy")["limite"], 1) == 814.1
      and va.get("regle_y") == ES.AVY_LAMINE_VARIANTE and proche(va.get("A_v_y"), geo) and proche(va.get("V_pl_y_Rd"), geo * 235.0 / R3)
      and round(va["V_pl_y_Rd"] / 1e3, 1) == 851.8 and va.get("pour_information") is True,
      (g["A_v"]["y"], g.get("V_pl_y"), va))
    # (reprise) la variante a SON texte : non retenue, pour information — jamais celui de la règle prise (« choisie dans la table »)
    mv = [a["message"] for a in r["alertes"] if a["code"] == "Avy"]
    c("la variante dite « non retenue, pour information » : « Pour information (décision du bureau n° 3, non retenue) : "
      "A_v,y = A − h_w·t_w. » — un autre texte que celui de la règle prise (2·b·t_f, le défaut)",
      va.get("message") == "Pour information (décision du bureau n° 3, non retenue) : A_v,y = A − h_w·t_w."
      and va["message"] not in ES.MSG_AVY.values() and mv == [ES.MSG_AVY[ES.AVY_LAMINE]] and len(va["message"]) <= 120, (va.get("message"), mv))
    r2 = ES.verifier({"section": heb, "nuance": "S235", "VyEd": 100e3, "parametres": {"avy_lamine": "A_moins_hwtw"}})
    v2 = r2["grandeurs"].get("A_v_variante") or {}
    c("la variante choisie dans la table : V_pl,y,Rd = 851,8 kN décide, 2·b·t_f (814,1 kN) devient la variante rendue",
      round(verif(r2, "E-Vy")["limite"], 1) == 851.8 and v2.get("regle_y") == ES.AVY_LAMINE and round(v2["V_pl_y_Rd"] / 1e3, 1) == 814.1
      and v2.get("message") == "Pour information (décision du bureau n° 3, non retenue) : A_v,y = 2·b·t_f."
      and [a["message"] for a in r2["alertes"] if a["code"] == "Avy"] == [ES.MSG_AVY[ES.AVY_LAMINE_VARIANTE]], v2)
    soude = SE.saisie("I", "soude_oxycoupe", h=400.0, b=200.0, tw=8.0, tf=12.0, A=200.0 * 24 + 376.0 * 8, Iy=2.3e8, Iz=1.6e7, iy=171.0, iz=45.0,
                      Wel_y=1.15e6, Wel_z=1.6e5, Wpl_y=1.3e6, Wpl_z=2.4e5)
    sans = [n for n, s_ in (("IPN 200", SE.du_catalogue("IPN 200")), ("I soudé", soude),
                            ("PRS", m("prs").section_prs(300, 20, 560, 10, 300, 20, a=5, nuance="S355", bords="oxycoupes")))
            if "A_v_variante" in ES.verifier({"section": s_, "nuance": "S355", "VyEd": 50e3})["grandeurs"]]
    c("un IPN (la plus petite des deux), un I soudé et un PRS (la règle de la norme) : aucune variante", not sans, sans)

    # ---- 5. S420 ailleurs que dans la section : le raidisseur (le cas du test 18 du brief, en S420)
    RA, PA = m("raidisseurs"), m("parametres")
    b18 = m("entree").nouvelle_barre("B")
    b18.update(profil="IPE 300", nuance="S420", travees=[2.0], poids_propre=False, appuis={"mode": "poses", "ss": 100.0, "c": 0.0},
               maintiens=[{"x": 1.0, "genre": "complet"}],
               cas=[{"nom": "Q", "nature": "exploitation_A", "charges": [{"type": "P", "v": 300.0, "a": 1.0, "ss": 100.0}]}])
    r18, err = essai(MO.verifier, b18)
    L18 = next((L for L in (r18 or {}).get("ame", {}).get("lieux", []) if L["nom"] == "Charge à 1,00 m"), {})
    vr = verif(r18["ame"], "A-L01-R") if r18 else None
    c("IPE 300 S420, P = 300 kN à 1 m (s_s 100) : la barre se vérifie (plus de « nuance inconnue : S420 ») ; le raidisseur est "
      "dimensionné, 2 plats 70 × 10 mm, écrasement 0,974",
      err is None and L18.get("decision") == "raidisseur" and (L18.get("raidisseur") or {}).get("designation") == "2 plats 70 × 10 mm"
      and vr is not None and round(vr["taux"], 3) == 0.974, err or (L18.get("decision"), L18.get("raidisseur"), vr and vr["taux"]))
    Pm = {k: PA.valeur(k) for k in ("E", "gamma_M0", "gamma_M1")}
    ipe = SE.du_catalogue("IPE 300")
    m20, m20m = RA.materiau(420.0, 20.0, "S420"), RA.materiau(420.0, 20.0, "S460", None, "M")
    c("le plat d'un raidisseur dans la liste du module : S420 de 20 mm → 400 (tableau 3.4 ANB) ; S460 M/ML de 20 mm → 440",
      m20["fy_plat"] == 400.0 and m20["fy_calcul"] == 400.0 and m20m["fy_plat"] == 440.0, (m20, m20m))
    c("la norme produit de la barre va jusqu'au plat (dimensionner → plat → materiau) : une norme inconnue y est refusée",
      (refus(RA.dimensionner, ipe, 420.0, Pm, 300e3, nuance="S420", norme_produit="X") or "").startswith("norme produit")
      and RA.dimensionner(ipe, 420.0, Pm, 300e3, nuance="S420", norme_produit="M")["fy_plat"] == 420.0)

    # ---- 5 bis. un PRS en S420 (la liste du module Barres, plus celle du Flambement)
    PR = m("prs")
    plats = {"b1": 300, "t1": 20, "hw": 560, "tw": 10, "b2": 300, "t2": 20, "a": 5, "bords": "oxycoupes"}
    nb = m("entree").nouvelle_barre("B")
    nb.update(nuance="S420", prs=dict(plats))
    nn, err = essai(m("entree").normaliser, nb)
    sp, err2 = essai(PR.section_prs, **plats, nuance="S420")
    gp = ES.verifier({"section": sp, "nuance": "S420", "NEd": 1000e3})["grandeurs"] if sp else {}
    c("PRS 600 × 300 × 10 × 20 en S420 : admis par l'entrée et par le PRS ; f_y 400 (semelles de 20 mm, tableau 3.4 ANB), âme 420",
      err is None and err2 is None and nn["profil"] == "PRS 600x300x10x20" and nn["nuance"] == "S420" and sp["nuance"] == "S420"
      and gp.get("fy") == 400.0 and gp.get("fy_source") == "tableau 3.4 ANB (EN 10025-3:2004, S420N), 16 < t ≤ 40 mm"
      and (gp.get("fy_parois") or {}).get("ame") == 420.0, err or err2 or {k: gp.get(k) for k in ("fy", "fy_source", "fy_parois")})
    c("PRS : une nuance hors du module refusée en une ligne qui dit les cinq nuances",
      refus(PR.section_prs, **plats, nuance="S690") == "PRS : nuance inconnue 'S690' (S235, S275, S355, S420 ou S460)",
      refus(PR.section_prs, **plats, nuance="S690"))
    rp, err = essai(MO.verifier, efforts(None, "S420", N=500.0, My=100.0, Ly=4.0, Lz=4.0, prs=dict(plats)))
    c("PRS S420 en efforts imposés : Eurocode 3 à f_y 400, méthode ELS « hors domaine » (nuance hors AE235 à AE355)",
      err is None and rp["methodes"]["ec3"]["grandeurs"]["fy"] == 400.0 and rp["methodes"]["historique"]["statut"] == m("resultat").HORS_DOMAINE,
      err or (rp["methodes"]["ec3"]["grandeurs"].get("fy"), rp["methodes"]["historique"]["statut"]))
    return C


# ------------------------------------------------------------------ 6. les mutations
def _ordre():
    """Les modules du paquet et ceux de chacun qu'ils importent (« from . import x », « from .x import »)."""
    dep = {}
    for f in sorted(os.listdir(DOSSIER)):
        if f.endswith(".py") and f != "__init__.py":
            src = open(os.path.join(DOSSIER, f), encoding="utf-8").read()
            dep[f[:-3]] = set(re.findall(r"^\s*from \. import (\w+)", src, re.M)) | set(re.findall(r"^\s*from \.(\w+) import", src, re.M))
    return dep


DEP = _ordre()


def _dependants(cible):
    """Les modules qui emploient ``cible`` (directement ou non), dans un ordre où chacun vient après ce qu'il importe."""
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
    """Les contrôles qui rougissent quand le texte ``ancien`` du module ``fichier`` du paquet devient ``nouveau`` ; None si
    le texte n'y est pas une seule fois. Le module d'origine est remis, et ceux qui l'emploient rechargés, quoi qu'il arrive."""
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
    ("nuances", "S420 absent de la liste du module", 'NUANCES = ("S235", "S275", "S355", "S420", "S460")', 'NUANCES = ("S235", "S275", "S355", "S460")'),
    ("nuances", "le défaut lu en M/ML (N/NL et M/ML échangés)", 'return nuance + (norme_effective(nuance, norme_produit) or "")',
     'return nuance + ("" if norme_effective(nuance, norme_produit) else "M" if nuance in AVEC_NORME else "")'),
    ("nuances", "« M » appliqué à S235, S275, S355", "return norme_produit if nuance in AVEC_NORME else None", "return norme_produit"),
    ("nuances", "la borne de 80 mm du module oubliée", "    if t > NO.T_MAX_TABLEAU:\n", "    if False:\n"),
    ("ec3_section", "la norme produit de la barre ignorée", 'NU.norme_effective(d["nuance"], barre.get("norme_produit"))', 'NU.norme_effective(d["nuance"], None)'),
    ("ec3_section", "la norme produit oubliée pour les parois (ε de N/NL sous une f_y de M/ML)", "        return _fy(nuance_, s_, t, norme_produit)\n",
     "        return _fy(nuance_, s_, t, None)\n"),
    ("ec3_section", "S420 refusé par la section (la liste du Flambement)", 'if d["nuance"] not in NU.NUANCES:', 'if d["nuance"] not in NO.NUANCES:'),
    ("ec3_section", "f_u lu dans la colonne de f_y", '{"fu": NU.fu(nuance, t, produit(s), norme_produit)', '{"fu": NU.fy(nuance, t, produit(s), norme_produit)'),
    ("ec3_section", "f_u et son origine non rendus", "        d.update(resistance_traction(n, s, norme))\n", "        d.update(fu=None, fu_source=None, fu_ref=None)\n"),
    ("ec3_section", "la norme produit rendue telle que saisie", '        d["norme_produit"] = norme\n', '        d["norme_produit"] = barre.get("norme_produit")\n'),
    ("ec3_section", "cornière : le message d'avant (deux idées, un renvoi)", 'MSG_FORME = "Eurocode 3 : non couvert pour {quoi} (v1)."',
     'MSG_FORME = "Cornière, U ou té : hors du module (compression simple : module Flambement)."'),
    ("ec3_section", "décision n° 3 : la variante non rendue", '            if a == "y" and av.get("variante_y"):', "            if False:"),
    ("ec3_section", "décision n° 3 : la variante = la règle prise",
     '{"y": A - hw * s["tw"], "regle_y": AVY_LAMINE_VARIANTE} if regle == AVY_LAMINE', '{"y": avy, "regle_y": AVY_LAMINE_VARIANTE} if regle == AVY_LAMINE'),
    ("ec3_section", "décision n° 3 : un IPN rend une variante", "            avy, regle = min(avy, A - hw * s[\"tw\"]), AVY_MIN\n            variante = None\n",
     "            avy, regle = min(avy, A - hw * s[\"tw\"]), AVY_MIN\n"),
    ("ec3_section", "reprise — décision n° 3 : la variante dite par le texte de la règle prise (« choisie dans la table »)",
     '"message": MSG_AVY_VARIANTE[var["regle_y"]]', '"message": MSG_AVY[var["regle_y"]]'),
    ("ec3_section", "reprise — décision n° 3 : les deux textes de la variante échangés", '"message": MSG_AVY_VARIANTE[var["regle_y"]]',
     '"message": MSG_AVY_VARIANTE[AVY_LAMINE if var["regle_y"] == AVY_LAMINE_VARIANTE else AVY_LAMINE_VARIANTE]'),
    ("ec3_section", "décision n° 3 : un I soudé rend une variante", '        variante = None if soude else ({"y"', '        variante = ({"y"'),
    ("parametres", "décision n° 3 : le défaut de la table changé (A − h_w·t_w)", '"avy_lamine": {"valeur": "2btf"', '"avy_lamine": {"valeur": "A_moins_hwtw"'),
    ("entree", "S420 refusé par l'entrée", 'if b["nuance"] not in NU.NUANCES:', 'if b["nuance"] not in ("S235", "S275", "S355", "S460"):'),
    ("entree", "une cornière ou un U admis en barre chargée", 'elif b["mode"] == "chargee" and SE.FAMILLES.get(', "elif False and SE.FAMILLES.get("),
    ("historique", "cornière : chaque axe garde sa longueur saisie (v-v à 1 m)", '    corniere = s0["forme"] == "L" and d["P"] > 0\n', "    corniere = False\n"),
    ("prs", "reprise — PRS : la liste du module Flambement (S420 refusé)", "    if nuance not in NU.NUANCES:",
     '    if nuance not in ("S235", "S275", "S355", "S460"):'),
    ("prs", "reprise — PRS : le refus ne dit que quatre nuances", "({', '.join(NU.NUANCES[:-1])} ou {NU.NUANCES[-1]})", "(S235, S275, S355 ou S460)"),
    ("raidisseurs", "reprise — le plat lu dans la liste du module Flambement (S420 inconnu)", 'NU.fy(nuance, ts, "lamine", norme_produit)',
     'NO.limite_elastique(nuance, ts, produit="lamine")'),
    ("raidisseurs", "reprise — plat ne transmet pas la norme produit", "    m = materiau(fy, ts, nuance, fy_ame, norme_produit)\n",
     "    m = materiau(fy, ts, nuance, fy_ame)\n"),
    ("raidisseurs", "reprise — dimensionner ne transmet pas la norme produit", "        p = plat(s, fy, P, ts, dispo, nuance, fy_ame, norme_produit)\n",
     "        p = plat(s, fy, P, ts, dispo, nuance, fy_ame)\n"),
]


def _ligne(nuance, produit, valeur):
    """Une mutation de la TABLE des aciers (acier/materiaux.py) : une ligne de ``LIGNES`` remplacée, en place."""
    def appliquer():
        MA.LIGNES[nuance][produit] = valeur
    return appliquer


MUTATIONS_TABLE = [
    ("table — S420 N/NL lu au tableau 3.5 (M/ML)", _ligne("S420", "lamine", ("3.5", "S420M"))),
    ("table — S460 M/ML lu au tableau 3.4 (N/NL)", _ligne("S460M", "lamine", ("3.4", "S460N"))),
    ("table — S460 sans norme produit lu au tableau 3.5", _ligne("S460", "lamine", ("3.5", "S460M"))),
]


def muter_table(appliquer):
    sauf = {k: dict(v) for k, v in MA.LIGNES.items()}
    try:
        appliquer()
        rouges = [n for n, ok, _ in controles() if not ok]
    except Exception as e:  # noqa: BLE001
        rouges = [f"exception {type(e).__name__}: {e}"]
    finally:
        MA.LIGNES.clear()
        MA.LIGNES.update(sauf)
    return rouges


def main():
    import acier.corniere.entree  # noqa: F401 — les listes des autres modules, lues par les contrôles
    import api_web.flambement  # noqa: F401
    print("== 1 à 5. nuances et norme produit, le résultat, cornières et U, décision n° 3, S420 du raidisseur")
    for nom, ok, info in controles():
        chk(nom, ok, info)
    print("== 6. mutations")
    for fichier, nom, ancien, nouveau in MUTATIONS:
        rouges = muter(fichier, ancien, nouveau)
        if rouges is None:
            chk(f"mutation « {nom} » : le texte à muter est dans {fichier}.py, une seule fois", False, "texte absent ou répété")
        else:
            chk(f"mutation « {nom} » ({fichier}.py) : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    for nom, appliquer in MUTATIONS_TABLE:
        rouges = muter_table(appliquer)
        chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    chk("après les mutations, le paquet et la table sont rendus : les contrôles repassent tous", all(ok for _, ok, _ in controles()))
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
