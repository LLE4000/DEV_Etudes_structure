# -*- coding: utf-8 -*-
"""c09 — Le module actuel du site (acier/corniere) face au cas de contrôle du classeur.

Le moteur du site est IMPORTÉ en lecture seule depuis /home/user/integ (sys.dont_write_bytecode : aucun fichier
compilé n'est écrit dans ce dossier). On l'appelle avec ses propres fonctions (calculer), sans rien modifier.
"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import commun as K  # noqa: E402

sys.path.insert(0, "/home/user/integ")
from acier.corniere import moteur as M  # noqa: E402
from acier.corniere import entree as EN  # noqa: E402

RACINE = os.path.dirname(ICI)
p = K.CAS
xl = K.classeur(**p)
fy = p["fy"]
res = {"defauts_site": {k: v[0] for k, v in EN.CHAMPS.items()}}
print("Défauts du site :", {k: v[0] for k, v in EN.CHAMPS.items()})

# le site : charge ponctuelle V (kN) répartie sur L (m), excentricité e (cm), modèle « lame » : seul t compte.
# trois façons honnêtes de remplir e pour le cas de contrôle
choix = [("e = X = 7,5 cm (moment au nu du mur)", p["X"] / 10),
         ("e = Xe = 4,7 cm (bras de la section 1)", xl["C19_Xe"] / 10),
         ("e = X·Ye/YH = 5,516 cm (équivalent section 3)", p["X"] * xl["C56_rapport3"] / 10)]
lim_s355 = (fy / 1.5, fy / 1.5 / math.sqrt(3), fy / 1.5 * 1.35)   # σ_adm, τ_adm, et σ_lim,ELU « cohérente » γ 1,35
for nom, e_cm in choix:
    for lnom, (sl, tl, su) in (("limites par défaut du site", (155.0, 0.8, 186.0)),
                               ("limites S355 / 1,5", lim_s355)):
        g = M.calculer(p["e"], p["R"], "ponctuelle", p["b_cm"] / 100, e_cm, 1.35, 30.0, sl, tl, su)
        print(f"  {nom:46s} [{lnom:26s}] σ_ELS = {g['sigma_ELS']:7.2f} MPa, τ_ELS = {g['tau_ELS']:5.2f} ; "
              f"U_σ = {g['U_sigma'] * 100:6.1f} %, U_τ = {g['U_tau'] * 100:7.1f} %, U_ELU = {g['U_ELU'] * 100:6.1f} %")
        res.setdefault("cas", []).append(dict(choix=nom, limites=lnom, sigma=g["sigma_ELS"], tau=g["tau_ELS"],
                                              U_sigma=g["U_sigma"], U_tau=g["U_tau"], U_ELU=g["U_ELU"]))
print(f"  classeur : S1 σ = {xl['S1']['sigma']:.2f}, S3 σ = {xl['S3']['sigma']:.2f} MPa, taux max {xl['D3_taux_max'] * 100:.2f} %")

# le défaut « longitudinal » : s se simplifie
print("\nVérification « longitudinale » du site, cas de contrôle (e = 7,5 cm), entraxe s variable :")
for s_cm in (10.0, 30.0, 100.0, 150.0, 1000.0):
    g = M.calculer(p["e"], p["R"], "ponctuelle", p["b_cm"] / 100, p["X"] / 10, 1.35, s_cm, 155.0, 0.8, 186.0)
    print(f"  s = {s_cm:6.0f} cm : σ_long,ELU = {g['sigma_long_ELU']:.6f} MPa ; σ_ELU transversal = {g['sigma_ELU']:.6f} ; "
          f"U_long = {g['U_long_ELU']:.6f} ; U_ELU = {g['U_ELU']:.6f} ; identiques : {g['U_long_ELU'] == g['U_ELU']}")
    res.setdefault("longitudinal", []).append(dict(s=s_cm, U_long=g["U_long_ELU"], U_ELU=g["U_ELU"]))
# ce que ce taux aurait dû contenir : flexion entre fixations, ex. q·s²/8 sur la cornière L120x120x15
with open("/home/user/integ/acier/donnees/profiles.json", encoding="utf-8") as f:
    prof = {pp["nom"]: pp for pp in json.load(f)["profils"]}
Wy = prof["L 120x120x15"]["Wel_y"] * 1e3
q = p["R"] / (p["b_cm"] / 100)                     # kN/m
for s_cm in (30.0, 100.0, 150.0):
    Ml = q * (s_cm / 100) ** 2 / 8
    print(f"  q·s²/8 avec q = {q:.1f} kN/m, s = {s_cm:.0f} cm : {Ml:.3f} kN·m → σ = {Ml * 1e6 / Wy:.1f} MPa (W_el,y = "
          f"{Wy / 1e3:.2f} cm³) — jamais calculé par le site")
    res.setdefault("manquant", []).append(dict(s=s_cm, M=Ml, sigma=Ml * 1e6 / Wy))
json.dump(res, open(os.path.join(RACINE, "sorties", "c09_site.json"), "w"), ensure_ascii=False, indent=1,
          default=float)

# écart relatif maximal U_long / U_ELU sur un balayage de s
ecarts = []
for s_cm in [10 + 7.3 * i for i in range(200)]:
    g = M.calculer(p["e"], p["R"], "ponctuelle", p["b_cm"] / 100, p["X"] / 10, 1.35, s_cm, 155.0, 0.8, 186.0)
    ecarts.append(abs(g["U_long_ELU"] / g["U_ELU"] - 1))
print(f"\nBalayage de 200 entraxes (10 à 1 463 cm) : écart relatif max U_long/U_ELU − 1 = {max(ecarts):.1e} "
      f"(égalité au bit près dans {sum(1 for x in ecarts if x == 0)} cas sur 200)")
res["ecart_long_max"] = max(ecarts)
# la flexion entre fixations, cas PAR DÉFAUT du site : L 100x100x10, V = 5 kN/m, γ = 1,35
W100 = prof["L 100x100x10"]["Wel_y"] * 1e3
for s_cm in (30.0, 100.0, 150.0):
    Ml = 5.0 * (s_cm / 100) ** 2 / 8
    print(f"  défaut du site (L 100x100x10, 5 kN/m) : s = {s_cm:.0f} cm → q·s²/8 = {Ml:.3f} kN·m, σ_ELS = "
          f"{Ml * 1e6 / W100:.1f} MPa (W_el,y = {W100 / 1e3:.2f} cm³) ; le taux « longitudinal » du site ne bouge pas")
    res.setdefault("manquant_defaut", []).append(dict(s=s_cm, M=Ml, sigma=Ml * 1e6 / W100))
json.dump(res, open(os.path.join(RACINE, "sorties", "c09_site.json"), "w"), ensure_ascii=False, indent=1,
          default=float)
