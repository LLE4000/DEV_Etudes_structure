# -*- coding: utf-8 -*-
"""Les formules de la note du garde-corps (acier/garde_corps/formules.py) : chaque
substitution imprimée, ré-évaluée avec ses nombres tels qu'imprimés, redonne le résultat
du moteur — montant, lisse, barreau, platine, console (système mur), en ELS et en
Eurocode, avec un angle, en mode P, avec boulons et avec chevilles ; les critères en
typographie mathématique se composent ; la soudure reprend les formules du module Soudures.

Lancement : python3 tests/test_garde_corps_formules.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import fixation as FX  # noqa: E402
from acier.garde_corps import formules as FO  # noqa: E402
from ndc_pdf import mathx as M  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def cas(**kw):
    c = ET.nouveau_cas()
    for k, v in kw.items():
        c[k] = {**c[k], **v} if isinstance(v, dict) else v
    return c


print("=== 1. Chaque substitution redonne le résultat du moteur ===")
CAS = [("défaut (ELS, chevilles)", cas()),
       ("Eurocode, boulons M12", cas(methode={"mode": "eurocode"}, fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}})),
       ("en L, RHS 60×40, Q sur la lisse", cas(geometrie={"forme": "L"}, montant={"section": {"nom": "RHS 60x40x3"}}, charges={"Q": 0.3})),
       ("en L, Q en « max »", cas(geometrie={"forme": "L"}, charges={"Q": 0.3}, methode={"comb": "max"})),
       ("mode P direct", cas(charges={"mode": "P", "P": 1.0})),
       ("plat 50×10, flèche en mm", cas(montant={"section": {"nom": "Plat 50x10"}}, methode={"fleche_mode": "mm", "fleche_val": 20.0})),
       ("lisses intermédiaires", cas(remplissage={"type": "lisses"})),
       ("barreaux : la pression gouverne", cas(remplissage={"type": "barreaux", "section": {"nom": "Plat 80x8"}, "entraxe": 300.0, "L": 1000.0}, charges={"qr": 3.0, "Qr": 0.2})),
       ("système mur (console, traction)", cas(geometrie={"systeme": "mur"})),
       ("système mur, Eurocode, boulons", cas(geometrie={"systeme": "mur"}, methode={"mode": "eurocode"}, fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}}))]
n_sub = 0
for nom, c in CAS:
    res = ET.verifier(c)
    t = FO.toutes(res)
    blocs = {k: v for k, v in t.items() if k != "soudure"}
    tous = [s for subs in blocs.values() for s in subs]
    mauvais = [FO.rendre(s) for s in tous if not FO.verifier(s)]
    n_sub += len(tous)
    chk(f"{nom} : {len(tous)} substitutions ré-évaluées", tous and not mauvais, " ; ".join(mauvais)[:300])
    sou = t["soudure"]
    lignes = sou["groupe"] + sou["point"] + sou["critere"]
    mauvais_s = [FO.rendre(s) for s in lignes if not FO.verifier(s)]
    chk(f"{nom} : la soudure a ses formules ({len(lignes)}) et elles se ré-évaluent", lignes and not mauvais_s, " ; ".join(mauvais_s)[:200])
chk("au moins 250 substitutions en tout", n_sub >= 250, str(n_sub))

print("\n=== 2. Contenu et typographie ===")
res = ET.verifier(cas())
t = FO.toutes(res)
txt = "\n".join(FO.rendre(s) for s in t["montant"])
chk("le montant : P₁ = q_h·s_trib, M₁ = P₁·H, σ = M/W_el + N/A, f = P₁·H³/(3·E·I_y), η_f, η = max", all(k in txt for k in ("P₁ = q_h·s_trib", "M₁ = P₁·H", "σ = M/W_el + N/A", "H³/(3·E·I_y)", "η_f", "η = max(η_σ, η_τ, η_f)")))
chk("les exposants s'écrivent ² ³ ⁴ et ^1,5 (jamais « ** »)", "**" not in txt and "⁴" in "\n".join(FO.rendre(s) for s in t["lisse"]) and "^1,5" in "\n".join(FO.rendre(s) for s in t["platine"]))
txt_p = "\n".join(FO.rendre(s) for s in t["platine"])
chk("la platine : z, Z, F_t, F_v, N_Rd/γ_F, η_inter, m, b_eff, M_p, σ_p, t_req", all(k in txt_p for k in ("z = e_x/2 + h_p/2", "Z = max(0, (M − N·h_p/2)/z)", "F_t = Z/n", "N_Rd/γ_F", "η_inter", "m = e_x/2 − h_p/2 − 0,8·a·√2", "b_eff", "M_p = Z·m", "σ_p = 6·M_p/(b_eff·t_p²)", "t_req")))
rW = ET.verifier(cas(geometrie={"systeme": "mur"}))
tW = FO.toutes(rW)
chk("système mur : console R_v, R_h, M = R_v·e ; platine tirée : enveloppe des deux modèles", "M = R_v·e" in "\n".join(FO.rendre(s) for s in tW["console"])
    and "M/e_x + N_t/2" in "\n".join(FO.rendre(s) for s in tW["platine"]) and not tW["montant"])
for cle, expr in FO.CRITERES_MATH.items():
    try:
        node = M.parse(expr)
        chk(f"critère « {cle} » composé en typographie mathématique", node is not None)
    except Exception as e:  # noqa: BLE001
        chk(f"critère « {cle} » composé en typographie mathématique", False, str(e)[:200])
chk("la clé de méthode suit le mode", FO.cle_methode(res) == "els" and FO.cle_methode(ET.verifier(cas(methode={"mode": "eurocode"}))) == "eurocode")
rE = ET.verifier(cas(methode={"mode": "eurocode"}))
txtE = "\n".join(FO.rendre(s) for s in FO.toutes(rE)["montant"])
chk("en Eurocode, σ porte γ_Q et la limite f_y/γ_M0 est écrite", "γ_Q·M/W_el" in txtE and "σ_lim = fy/γM0" in txtE)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
