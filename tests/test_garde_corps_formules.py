# -*- coding: utf-8 -*-
"""Les formules de la note du garde-corps (acier/garde_corps/formules.py) : chaque bloc
(montant, main courante, barreaux ou lisses, console, platine) nomme le cas de charge qui le
gouverne et les autres cas avec leur taux ; chaque substitution imprimée, ré-évaluée avec ses
nombres tels qu'imprimés, redonne le résultat du moteur — cas distincts et cumul, deux appuis
et continue, ELS et Eurocode, angle, mode P, boulons et chevilles, types du menu ; les
critères en typographie mathématique se composent ; la soudure reprend les formules du
module Soudures.

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
from ndc_pdf import riche as RI  # noqa: E402

OK, KO = [], []
BLOCS = ("montant", "console", "lisse", "barre", "platine")


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def cas(type_="barreaudage", **kw):
    c = ET.nouveau_cas(None, type_)
    for k, v in kw.items():
        c[k] = {**c[k], **v} if isinstance(v, dict) else v
    return c


print("=== 1. Chaque substitution redonne le résultat du moteur ===")
CAS = [("défaut (ELS, chevilles, cas distincts)", cas()),
       ("cumul q_h + q_v", cas(charges={"cumul": True})),
       ("main courante continue (4 travées)", cas(lisse={"appuis": "continue"})),
       ("continue et cumul, Q sur la main courante", cas(lisse={"appuis": "continue"}, charges={"cumul": True, "Q": 0.3})),
       ("Eurocode, boulons M12", cas(methode={"mode": "eurocode"}, fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}})),
       ("en L, RHS 60×40, Q sur la main courante", cas(geometrie={"forme": "L"}, montant={"section": {"nom": "RHS 60x40x3"}}, charges={"Q": 0.3})),
       ("en L, Q en « max »", cas(geometrie={"forme": "L"}, charges={"Q": 0.3}, methode={"comb": "max"})),
       ("mode P direct", cas(charges={"mode": "P", "P": 1.0})),
       ("plat 50×10, flèche en mm", cas(montant={"section": {"nom": "Plat 50x10"}}, methode={"fleche_mode": "mm", "fleche_val": 20.0})),
       ("type lisses", cas("lisses")),
       ("barreaux : la pression gouverne", cas(remplissage={"section": {"nom": "Plat 80x8"}, "entraxe": 300.0}, charges={"qr": 3.0, "Qr": 0.2})),
       ("murale (console sous q_v, tirée sous q_h)", cas("murale")),
       ("murale cumulée, Eurocode, boulons", cas("murale", charges={"cumul": True}, methode={"mode": "eurocode"}, fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}})),
       ("platine latérale (nez de dalle)", cas(fixation={"type": "laterale"})),
       ("platine latérale en L, plat, boulons, cumul", cas(geometrie={"forme": "L"}, montant={"section": {"nom": "Plat 60x12"}}, charges={"cumul": True},
                                                          fixation={"type": "laterale", "platine": {"bp": 200.0, "lp": 160.0, "tp": 10.0, "d_h": 60.0},
                                                                    "ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon", "ex": 140.0, "ey": 90.0}})),
       ("sans montant (loggia de mur à mur)", cas(geometrie={"debut": "mur", "fin": "mur", "troncons": [{"L": 2500, "s_max": 3000, "entraxes": []}, {"L": 2400, "s_max": 1000, "entraxes": []}, {"L": 3600, "s_max": 1000, "entraxes": []}]}))]
n_sub = 0
for nom, c in CAS:
    res = ET.verifier(c)
    t = FO.toutes(res)
    tous = [s for k in BLOCS for s in t[k]["subs"]]
    mauvais = [FO.rendre(s) for s in tous if not FO.verifier(s) or "⁣" in FO.rendre(s)]
    n_sub += len(tous)
    chk(f"{nom} : {len(tous)} substitutions ré-évaluées", tous and not mauvais, " ; ".join(mauvais)[:300])
    sou = t["soudure"]
    lignes = sou["groupe"] + sou["point"] + sou["critere"]
    mauvais_s = [FO.rendre(s) for s in lignes if not FO.verifier(s)]
    if "sans montant" in nom:
        chk(f"{nom} : sans montant, pas de soudure — aucune formule de soudure ({len(lignes)})", not lignes, " ; ".join(FO.rendre(s) for s in lignes)[:200])
    else:
        chk(f"{nom} : la soudure a ses formules ({len(lignes)}) et elles se ré-évaluent", lignes and not mauvais_s, " ; ".join(mauvais_s)[:200])
chk("au moins 300 substitutions en tout", n_sub >= 300, str(n_sub))

print("\n=== 2. Contenu et typographie ===")
res = ET.verifier(cas())
t = FO.toutes(res)
txt = "\n".join(FO.rendre(s) for s in t["montant"]["subs"])
chk("le montant : bloc « cas 1 · q_h seule », l'autre cas cité avec son taux, P₁ = q_h·s_trib, M₁ = P₁·H, σ = M₁/W_el,y, f = P₁·H³/(3·E·I_y), η = max",
    t["montant"]["cas"] == "cas 1 · q_h seule" and "cas 2 · q_v seule" in t["montant"]["autres"] and "%" in t["montant"]["autres"]
    and all(k in txt for k in ("P₁ = q_h·s_trib", "M₁ = P₁·H", "σ = M₁/W_el,y", "H³/(3·E·I_y)", "η = max(η_τ, η_eq, η_f)")) and "N =" not in txt)
txt_l = "\n".join(FO.rendre(s) for s in t["lisse"]["subs"])
chk("la main courante : cas 1 (q_h seule), M_h = q_h·L²/8, σ = M_h/W_el,z, f = 5·q_h·L⁴/(384·E·I_z) ; exposants ² ⁴ (jamais « ** »)",
    t["lisse"]["cas"].startswith("cas 1") and "M_h = q_h·L²/8" in txt_l and "σ = M_h/W_el,z" in txt_l and "L⁴/(384·E·I_z)" in txt_l and "**" not in txt_l)
tc = FO.toutes(ET.verifier(cas(lisse={"appuis": "continue"})))
chk("main courante continue : M_h = k_M·q_h·L² avec k_M = 0,107, V = k_V·q_h·L, f = k_f·q_h·L⁴/(E·I_z)", "M_h = k_M·q_h·L² = 0,107" in "\n".join(FO.rendre(s) for s in tc["lisse"]["subs"])
    and "V = k_V·q_h·L" in "\n".join(FO.rendre(s) for s in tc["lisse"]["subs"]) and "f = k_f·q_h·L⁴/(E·I_z)" in "\n".join(FO.rendre(s) for s in tc["lisse"]["subs"]))
tk = FO.toutes(ET.verifier(cas(charges={"cumul": True})))
chk("cumul : bloc « cas 1 + 2 », σ = M₁/W_el,y + N/A au montant, σ = M_v/W_el,y + M_h/W_el,z à la main courante",
    tk["montant"]["cas"].startswith("cas 1 + 2") and "σ = M₁/W_el,y + N/A" in "\n".join(FO.rendre(s) for s in tk["montant"]["subs"])
    and "σ = M_v/W_el,y + M_h/W_el,z" in "\n".join(FO.rendre(s) for s in tk["lisse"]["subs"]))
txt_b = "\n".join(FO.rendre(s) for s in t["barre"]["subs"])
chk("les barreaux : cas 3 · Q_r seule, M = Q_r·L/4 avec L = 950 déduite, f = Q_r·L³/(48·E·I)", t["barre"]["cas"] == "cas 3 · Q_r seule" and "M = Q_r·L/4 = 0,5 × 950/4" in txt_b and "L³/(48·E·I)" in txt_b)
txt_p = "\n".join(FO.rendre(s) for s in t["platine"]["subs"])
chk("la platine (cas 1) : z, Z = M/z avec N = 0, F_t, F_v, N_Rd/γ_F, η_inter en ^1,5, m, b_eff, M_p, σ_p, t_req",
    t["platine"]["cas"] == "cas 1 · q_h seule" and all(k in txt_p for k in ("z = e_x/2 + h_p/2", "Z = max(0, (M − N·h_p/2)/z)", "F_t = Z/n", "N_Rd/γ_F", "^1,5", "m = e_x/2 − h_p/2 − 0,8·a·√2", "b_eff", "M_p = Z·m", "σ_p = 6·M_p/(b_eff·t_p²)", "t_req")))
rW = ET.verifier(cas("murale"))
tW = FO.toutes(rW)
chk("murale : console « cas 2 · q_v seule », R_v, M = R_v·e, σ = M/W_el ; l'autre cas (q_h, tirée) cité ; pas de bloc montant",
    tW["console"]["cas"] == "cas 2 · q_v seule" and "M = R_v·e" in "\n".join(FO.rendre(s) for s in tW["console"]["subs"]) and "cas 1 · q_h seule" in tW["console"]["autres"]
    and not tW["montant"]["subs"])
tWK = FO.toutes(ET.verifier(cas("murale", charges={"cumul": True})))
chk("murale cumulée : platine tirée, enveloppe des deux modèles (M/e_x + N_t/2)", "M/e_x + N_t/2" in "\n".join(FO.rendre(s) for s in tWK["platine"]["subs"]))
for cle, expr in FO.CRITERES_MATH.items():
    try:
        node = M.parse(expr)
        chk(f"critère « {cle} » composé en typographie mathématique", node is not None)
    except Exception as e:  # noqa: BLE001
        chk(f"critère « {cle} » composé en typographie mathématique", False, str(e)[:200])
chk("la clé de méthode suit le mode", FO.cle_methode(res) == "els" and FO.cle_methode(ET.verifier(cas(methode={"mode": "eurocode"}))) == "eurocode")
rE = ET.verifier(cas(methode={"mode": "eurocode"}))
txtE = "\n".join(FO.rendre(s) for s in FO.toutes(rE)["montant"]["subs"])
chk("en Eurocode, σ porte γ_Q et la limite f_y/γ_M0 est écrite", "γ_Q·M₁/W_el,y" in txtE and "σ_lim = f_y/γ_M0" in txtE and "τ_lim = f_y/(√3·γ_M0)" in txtE)

print("\n=== 3. Les groupes, dans l'ordre de lecture ===")
GROUPES = [FO.CHARGES, FO.SOLLICITATIONS, FO.CONTRAINTES, FO.FLECHE, FO.TAUX]
for cle in ("montant", "lisse", "barre"):
    libs = [lib for lib, _ in t[cle]["groupes"]]
    chk(f"{cle} : les groupes se suivent — charges, sollicitations, contraintes, flèche, taux (ceux qui existent), et « subs » les met à plat",
        libs == [g for g in GROUPES if g in libs] and libs[-3:] == [FO.CONTRAINTES, FO.FLECHE, FO.TAUX]
        and [s.nom for s in t[cle]["subs"]] == [s.nom for _, subs in t[cle]["groupes"] for s in subs], str(libs))
noms_m = [s.nom for s in t["montant"]["subs"]]
chk("montant : P₁ puis M₁, V, puis σ, τ, η_τ, σ_eq, η_eq (chaque taux suit sa contrainte), puis f, f_adm, η_f, puis η",
    noms_m == ["P₁", "M₁", "V", "σ", "τ", "η_τ", "σ_eq", "η_eq", "f", "f_adm", "η_f", "η"], str(noms_m))
chk("un taux reprend le nombre IMPRIMÉ de sa contrainte : η_eq = σ_eq/σ_adm = 91,3/156,7 (et non 91,34/156,67), η_f = f/f_adm = 7,01/11",
    "η_eq = σ_eq/σ_adm = 91,3/156,7 = 58,3 %" in txt and "η_f = f/f_adm = 7,01/11 = 63,7 %" in txt)
chk("les limites de l'ELS sont DÉRIVÉES de la nuance (σ_adm = f_y/1,5 = 156,7, τ_adm = σ_adm/√3 = 90,5 — écrites une fois dans les hypothèses, pas dans chaque bloc) ; "
    "chaque bloc porte la contrainte équivalente σ_eq = √(σ² + 3·τ²) et son taux η_eq, et η = max(η_τ, η_eq, η_f) sans η_σ (règle du bureau du 05/11/2026 : plus de 160 ni de 90)",
    "σ_eq = √(σ² + 3·τ²) = √(91,2² + 3 × 1,66²) = 91,3 N/mm²" in txt and "η_eq = σ_eq/σ_adm = 91,3/156,7 = 58,3 %" in txt and "η_τ = τ/τ_adm = 1,66/90,5 = 1,8 %" in txt
    and "σ_adm = f_y" not in txt and "/160" not in txt and "/90 " not in txt and "η = max(η_τ, η_eq, η_f)" in txt and "η_σ" not in txt)
chk("en ELS les limites s'appellent σ_adm et τ_adm ; en Eurocode σ_lim et τ_lim (σ_eq/σ_lim, et la ligne σ_lim = f_y/γ_M0)",
    "τ/τ_adm" in txt and "σ_eq/σ_adm" in txt and "σ_eq/σ_lim" in txtE and "σ_lim = f_y/γ_M0" in txtE)
libs_p = [lib for lib, _ in t["platine"]["groupes"]]
chk("platine : trois groupes — ancrages (efforts), ancrages (résistances et taux), platine en flexion", libs_p == [FO.ANCRAGES_EFFORTS, FO.ANCRAGES_RESISTANCES, FO.PLATINE_FLEXION], str(libs_p))
chk("les autres cas sont aussi donnés en liste (libellé, taux)", t["montant"]["cas_taux"] and t["montant"]["cas_taux"][0][0] == "cas 2 · q_v seule" and 0 < t["montant"]["cas_taux"][0][1] < 0.1)
tL = FO.toutes(ET.verifier(cas(geometrie={"forme": "L"})))
chk("montant d'angle : P₂ et M₂ sans mention entre parenthèses (le titre du bloc dit l'angle), V₁ et V₂ nommés, τ sur la direction qui gouverne (V₂/A_v,y), f₁, f₂ et f_rés",
    [s.nom for s in tL["montant"]["subs"]][:6] == ["P₁", "P₂", "M₁", "M₂", "V₁", "V₂"] and "τ = V₂/A_v,y = " in "\n".join(FO.rendre(s) for s in tL["montant"]["subs"])
    and [s.nom for s in tL["montant"]["subs"] if s.nom.startswith("f")] == ["f₁", "f₂", "f_rés", "f_adm"]
    and "P₁ = q_h·s_trib,1 = " in "\n".join(FO.rendre(s) for s in tL["montant"]["subs"]), str([s.nom for s in tL["montant"]["subs"]][:8]))
tLat = FO.toutes(ET.verifier(cas(fixation={"type": "laterale"})))
txtLat = "\n".join(FO.rendre(s) for s in tLat["platine"]["subs"])
chk("platine latérale : z = e_x = 100 mm, Z = M/z + P/2, F_t = Z/n, F_v = V/2n, m = e_y/2 − b/2 − 0,8·a·√2, b_eff = min(b_p, …), t_p = 10 mm face à t_req",
    "z = e_x = 100 mm" in txtLat and "Z = M/z + P/2 = " in txtLat and "F_t = Z/n" in txtLat and "F_v = V/2n" in txtLat and "m = e_y/2 − b/2 − 0,8·a·√2 = 100/2 − 30/2 − 4,525" in txtLat
    and "b_eff = min(b_p, n·min(2·π·m, 4·m + 1,25·e))" in txtLat and "t_p = 10 mm" in txtLat and "t_req" in txtLat
    and [lib for lib, _ in tLat["platine"]["groupes"]] == [FO.ANCRAGES_EFFORTS, FO.ANCRAGES_RESISTANCES, FO.PLATINE_FLEXION])
tLatL = FO.toutes(ET.verifier(cas(geometrie={"forme": "L"}, fixation={"type": "laterale"})))
chk("montant d'angle en latéral : F_v = V/2n + F_v,tors (la torsion des ancrages est écrite)", "F_v = V/2n + F_v,tors" in "\n".join(FO.rendre(s) for s in tLatL["platine"]["subs"]))
chk("chaque morceau d'une substitution se compose en typographie mathématique (riche.tex puis mathx.parse)",
    all(M.parse(RI.tex(p)) is not None for b in BLOCS for s in t[b]["subs"] for p in FO.morceaux(s)[0] + [FO.morceaux(s)[1]]))


# ================================================================
print("\n=== 5. Relecture 3.1 : chaque bloc écrit V, τ et sa flèche ; lisse intermédiaire sur l'axe faible ===")
tM = FO.toutes(ET.verifier(cas("murale")))
txtC = "\n".join(FO.rendre(s) for s in tM["console"]["subs"])
chk("console au mur : R_v, M = R_v·e, V, σ, τ = V/A_v, σ_eq, f = R_v·e³/(3·E·I), f_adm, η — et chaque substitution se ré-évalue",
    [s.nom for s in tM["console"]["subs"]] == ["R_v", "M", "V", "σ", "τ", "η_τ", "σ_eq", "η_eq", "f", "f_adm", "η_f", "η"] and "f = R_v·e³/(3·E·I) = " in txtC
    and all(FO.verifier(s) for s in tM["console"]["subs"]))
tB = FO.toutes(ET.verifier(cas()))
chk("barreau : M, V, σ = M/W_el, τ = V/A_v, σ_eq, f — V et τ écrits (ils ne l'étaient pas)",
    [s.nom for s in tB["barre"]["subs"]][:5] == ["M", "V", "σ", "τ", "η_τ"] and "τ = V/A_v = 0,25 × 10³/240 = 1,04 N/mm²" in "\n".join(FO.rendre(s) for s in tB["barre"]["subs"]))
tLi = FO.toutes(ET.verifier(cas("lisses", geometrie={"forme": "L"}, montant={"section": {"nom": "RHS 60x40x3"}}, lisse={"appuis": "continue"})))
txtLi = "\n".join(FO.rendre(s) for s in tLi["barre"]["subs"])
chk("lisse intermédiaire : σ = M/W_el,z, τ = V/A_v,y, f sur I_z (axe faible, h vertical)", "σ = M/W_el,z = " in txtLi and "τ = V/A_v,y = " in txtLi and "48·E·I_z" in txtLi)
chk("la limite de la platine porte le nom de la méthode : σ_adm à l'ELS, σ_lim en Eurocode",
    "η_p = σ_p/σ_adm = " in "\n".join(FO.rendre(s) for s in tB["platine"]["subs"]) and "η_p = σ_p/σ_lim = " in "\n".join(FO.rendre(s) for s in FO.toutes(ET.verifier(cas(methode={"mode": "eurocode"})))["platine"]["subs"]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
