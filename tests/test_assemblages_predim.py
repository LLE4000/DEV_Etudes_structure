# -*- coding: utf-8 -*-
"""Le PRÉDIMENSIONNEMENT garde les données de base — les six attaches (25/11/2026).

Le propriétaire : « quand j'ai un prédimensionnement ou une optimisation, il faut que ça me garde mes données de base :
les efforts, le profilé, le niveau auquel ils sont. Après, c'est la plaque, le nombre de boulons qui doit changer. »

Ce que cette suite verrouille, pour chaque attache et sur plusieurs jeux de données (défaut, autres profilés, autres
efforts, niveaux décalés, grugeage, appui sur l'âme) :
  1. le moteur, en mode prédimensionnement, calcule AVEC les efforts, les profilés, les aciers et les niveaux saisis ;
  2. recopier une solution — CHACUNE des solutions géométriquement possibles — ne change QUE les champs pilotés
     (``entrees.PILOTEES_PAR_PREDIM`` : boulons, rangées, pinces, pièce d'attache, gorge), le mode, et les dimensions
     d'une cornière hors bibliothèque ; par le moteur (``apply_solution``) comme par l'écran du site
     (``api_web.assemblages.choisir_solution``) ;
  3. les champs pilotés ne contiennent AUCUNE donnée de base (effort, profilé, acier, niveau, grugeage, appui) ;
  4. l'écran le dit (ligne d'aide du mode, ligne « Proposition », sous-titre de la fenêtre des solutions).

Chaque garantie rougit si on la retire : une ``apply_solution`` qui toucherait à VEd, à un profilé ou à Δz fait rougir 2.

Lancement : python3 tests/test_assemblages_predim.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from api_web import assemblages as A  # noqa: E402
import api_web.assemblages_poutre as PP  # noqa: E402,F401  (enregistre les configurations)
import api_web.assemblages_poteau as PT  # noqa: E402,F401

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


# les DONNÉES DE BASE : ce que le prédimensionnement ne doit jamais toucher (les clés absentes d'une attache sont ignorées)
BASE = ("V_Ed", "N_Ed", "H_Ed", "M_Ed", "nat_N",                                          # efforts
        "prof_P", "prof_S", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u", "hS_u", "bS_u", "twS_u", "tfS_u", "rS_u",   # profilés
        "nu_P", "nu_S", "nu_pl", "nu_C", "fy_u", "fu_u", "bw_u",                          # aciers
        "d_top", "z_C",                                                                   # niveaux
        "d_nt", "d_nb", "l_n", "r_n", "lt_ok", "g_h",                                     # grugeage, jeu
        "appui", "rec_S", "j_lat",                                                        # appui sur le poteau
        "classe", "cat", "trou", "filet", "fix_P", "fix_S")                               # choix de l'ingénieur
CORNIERE_PERSO = {"k1_u", "k2_u", "kt_u", "kr_u"}       # une cornière proposée hors bibliothèque s'écrit en dimensions

JEUX = {
    "plat_ame": [dict(), dict(prof_P="IPE 500", prof_S="IPE 300", V_Ed=180, N_Ed=20, d_top=30, d_nt=0, l_n=0, z_C=45),
                 dict(prof_P="HEB 300", prof_S="HEA 240", nu_S="S235", V_Ed=90, M_Ed=2, d_nt=40, l_n=140, d_nb=30)],
    "doubles_cornieres": [dict(), dict(prof_P="IPE 500", prof_S="IPE 300", V_Ed=160, d_top=25, d_nt=0, l_n=0, z_C=45),
                          dict(prof_P="HEB 300", prof_S="HEA 240", nu_S="S235", V_Ed=80, N_Ed=30, H_Ed=5, d_nt=40, l_n=140)],
    "platine_about": [dict(), dict(prof_P="IPE 600", prof_S="IPE 400", V_Ed=260, d_top=40, d_nt=0, l_n=0, z_C=50),
                      dict(prof_P="HEA 800", prof_S="IPE 600", nu_pl="S235", V_Ed=450, N_Ed=40, d_nt=45, l_n=160)],
    "plat_ame_poteau": [dict(), dict(prof_P="HEB 240", prof_S="IPE 300", V_Ed=110, N_Ed=60, z_C=45),
                        dict(appui="Âme du poteau", prof_P="HEA 300", prof_S="IPE 330", nu_S="S235", V_Ed=95, rec_S=0)],
    "doubles_cornieres_poteau": [dict(), dict(prof_P="HEB 240", prof_S="IPE 300", V_Ed=110, N_Ed=40, z_C=45),
                                 dict(appui="Âme du poteau", prof_P="HEB 340", prof_S="IPE 330", V_Ed=95, M_Ed=2)],
    "platine_about_poteau": [dict(), dict(prof_P="HEB 240", prof_S="IPE 300", V_Ed=110, N_Ed=50, z_C=45),
                             dict(appui="Âme du poteau", prof_P="HEB 340", prof_S="IPE 330", nu_pl="S235", V_Ed=95)],
}
LU = {"V_Ed": lambda R: R.u.V_Ed, "prof_P": lambda R: R.u.prof_P, "prof_S": lambda R: R.u.prof_S,
      "d_top": lambda R: R.u.get("d_top"), "z_C": lambda R: R.u.z_C, "d_nt": lambda R: R.u.d_nt}

total = 0
for cle, jeux in JEUX.items():
    cfg = A.CONFIGS[cle]
    pilotees = set(cfg.E.PILOTEES_PAR_PREDIM)
    print(f"\n=== {cle} ===")
    chk("les champs pilotés ne contiennent aucune donnée de base (effort, profilé, acier, niveau, grugeage, appui)",
        not (pilotees & set(BASE)), str(sorted(pilotees & set(BASE))))
    for i, modifs in enumerate(jeux):
        el = A.nouvel_assemblage(cle)
        u0 = el["values"]
        inconnues = [k for k in modifs if k not in u0]
        u0.update(modifs)
        u0["mode_calc"] = cfg.E.MODE_PREDIM
        R, _vides, err = A.calculer(cfg, el)
        if R is None or inconnues:
            chk(f"jeu {i + 1} : le calcul aboutit", False, f"{err} {inconnues}")
            continue
        base = [k for k in BASE if k in u0]
        avant = A.valeurs_normales(cfg, u0)
        # 1. le moteur calcule avec les données saisies (profilés lus dans la bibliothèque, efforts tels quels)
        from acier.bibliotheques import DB, find
        hS = find(DB["profils"], u0["prof_S"])["h"]; hP = find(DB["profils"], u0["prof_P"])["h"]
        chk(f"jeu {i + 1} : en mode prédimensionnement, le moteur calcule avec VEd = {u0['V_Ed']} kN, {u0['prof_P']} et "
            f"{u0['prof_S']}, leurs niveaux et leur grugeage",
            R.pred and R.u.V_Ed == abs(float(u0["V_Ed"])) and R.h_S == hS and R.h_P == hP
            and all(float(R.u[k]) == float(u0[k]) for k in ("z_C", "d_nt", "d_nb", "l_n", "g_h") if k in u0)
            and ("d_top" not in u0 or float(R.u.d_top) == float(u0["d_top"])),
            f"{R.u.V_Ed} {R.h_S} {R.h_P}")
        lignes = [r for r in R.pd.rows if r.geom and r.prop]
        fautes, touchees = [], set()
        for r in lignes:
            # par le moteur …
            u_m = A.valeurs_normales(cfg, cfg.predim["apply"](dict(u0), r))
            # … et par l'écran du site
            u_s = dict(u0)
            A.choisir_solution(cfg, u_s, A._nom_solution(cfg, R, r))
            u_s = A.valeurs_normales(cfg, u_s)
            for nom, u2 in (("moteur", u_m), ("site", u_s)):
                chg = {k for k in set(avant) | set(u2) if avant.get(k) != u2.get(k)}
                touchees |= chg
                interdits = chg - pilotees - {"mode_calc"} - CORNIERE_PERSO
                if interdits or any(avant[k] != u2[k] for k in base):
                    fautes.append((nom, r.b["n"], r.n, sorted(interdits)))
                if u2.get("mode_calc") == cfg.E.MODE_PREDIM:
                    fautes.append((nom, "reste en prédimensionnement"))
            if u_m != u_s:
                fautes.append(("moteur ≠ site", r.b["n"], r.n))
        total += len(lignes)
        chk(f"jeu {i + 1} : {len(lignes)} solutions recopiées (moteur ET site) — seuls les boulons, la pièce d'attache et "
            f"sa gorge changent ; les {len(base)} données de base restent telles quelles",
            len(lignes) >= 3 and not fautes and touchees - {"mode_calc"} and touchees <= pilotees | {"mode_calc"} | CORNIERE_PERSO,
            f"{len(lignes)} lignes ; {fautes[:3]} ; touchées {sorted(touchees)}")
    # 4. l'écran le dit
    el = A.nouvel_assemblage(cle); el["values"]["mode_calc"] = cfg.E.MODE_PREDIM
    etat = A.nouvel_etat(); etat["assemblages"] = [el]; etat["actif"] = 0
    html = A.formulaire(etat)
    chk("l'écran du mode prédimensionnement dit ce qui est gardé et ce qui est proposé",
        "Vos efforts, vos profilés, leur acier, leurs niveaux" in html and "sont gardés" in html and "sont proposés" in html
        and "Vos efforts" not in A.formulaire(dict(etat, assemblages=[A.nouvel_assemblage(cle)])))

print("\n=== La fenêtre des solutions et la ligne « Proposition » ===")
chk("le sous-titre de la fenêtre des solutions et la ligne « Proposition » disent : efforts, profilés et niveaux gardés",
    "efforts, profilés et niveaux gardés" in A.SOUS_TITRE_SOLUTIONS and "efforts, profilés et niveaux gardés" in A.PROPOSITION_GARDE)
chk(f"au moins 200 solutions recopiées sur les six attaches ({total})", total >= 200, str(total))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)
