# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : le FERRAILLAGE (saf/fondation_grue/ferraillage.py), méthode du bureau, ELS,
avec le moteur du module « Poutres béton armé ». Cahier des charges du lot 4 (15/11/2026), § 5 et § 10 : R1 à R8.

Ce que la suite prouve :
  R1  la couverture : en CHAQUE élément hors des zones de renfort (et des plots coulés), l'acier en place (treillis +
      renforts locaux) ≥ l'acier demandé par le moteur poutre — rejoué ici par un appel DIRECT du moteur sur tous les
      éléments, pour chaque face et chaque direction (la linéarité qu'emploie le module est donc vérifiée partout) ;
  R2  chaque bande, chaque face : treillis + renforts ≥ l'acier demandé (moteur, bande et prolongement) ; renfort nul si
      le treillis suffit ; les barres réparties sur toute la zone de renfort ;
  R3  le treillis automatique est le plus léger des quatre (calculés un par un) ;
  R4  un treillis imposé : les renforts se recalculent, R1 et R2 restent vrais ;
  R5  les U de fermeture (53 par bord, dalle type, Ø12 e = 15, c = 5 cm) ;
  R6  le bordereau (poids = Σ n × L × masse au mètre ; béton = aire × h + plots coulés) ;
  R7  les bandes de la dalle type (0,50 m : M+ = 71,6 kNm, 143,2 kNm/m ; M− = −63,3 kNm ; l'acier du moteur pour une
      poutre 0,50 × 0,50 ; V à d au d du moteur) ;
  R8  le plot coulé : la cage complète, les brins verticaux « à dimensionner » ;
  et : l'effort tranchant (C verticaux quand le moteur les demande), l'arrachement, le poinçonnement non vérifié,
  le recontrôle des éléments par le moteur, la pureté (numpy seul), les MUTATIONS.

    /usr/bin/python3.12 tests/test_saf_grue_ferraillage.py
    /usr/bin/python3.12 tests/test_saf_grue_ferraillage.py --rapide [--racine dossier]
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ARGS = sys.argv[1:]
RACINE_MOTEUR = ARGS[ARGS.index("--racine") + 1] if "--racine" in ARGS else RACINE
RAPIDE = "--rapide" in ARGS
os.chdir(RACINE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, RACINE_MOTEUR)
sys.path.insert(1, ICI)

import numpy as np  # noqa: E402

from beton.poutre import moteur as MO  # noqa: E402
from saf.fondation_grue import efforts as EF  # noqa: E402
from saf.fondation_grue import ferraillage as FE  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402
from saf.fondation_grue import verification as V  # noqa: E402

OK, KO = [], []
JEUX = [(350, 440, 440, 350), (150, 150, 550, 550), (135, 100, 800, 135)]
COULE = {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}
PREF = {"type": "prefabrique", "a": 2.5, "b": 2.5, "h": 0.35}
MASSES = {10: 0.617, 12: 0.888, 16: 1.578, 20: 2.466}         # § 6.3 : π·Ø²/4 × 7 850 kg/m³


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info != "" and not cond else ""))


def projet(plots=None, largeur=0.5, h=0.5, L=8.0, jeux=JEUX, treillis=("auto", "auto")):
    """La dalle type du cahier (8 × 8 × 0,50 m, châssis 6 × 6 centré, appuis 0,5 × 0,5, 3 jeux) aux réglages par défaut du
    lot 4 (maille 0,20 m, bandes de 0,50 m, k = 2 ; 5 ; 10 cochés « SCIA », C2 = k/10, enrobage 5 cm)."""
    el = PR.nouveau()
    el["reactions"] = [{"nom": "", "P": list(j)} for j in jeux]
    el["dalle"].update(Lx=L, Ly=L, h=h)
    el["bandes"]["largeur"] = largeur
    el["armatures"]["treillis_inf"], el["armatures"]["treillis_sup"] = treillis
    if plots:
        el["grue"]["plots"] = dict(plots)
    return PR.lire(el)


def ferrailler(el):
    m = PR.modele(el)
    rs = [r for r in V.par_k(m, [x["k"] for x in el["sol"]["k"]], (el["options"]["c2x"], el["options"]["c2y"]))
          if r["k"] in set(PR.k_scia(el))]
    # (le d de l'enveloppe ne sert qu'à remplir « V_d » : le ferraillage relit V à d au d du MOTEUR — 0,30 m ici, pour que
    # les deux ne se confondent pas)
    ef = EF.enveloppe(m, rs, 0.30, el["options"]["torsion"])
    return m, ef, FE.calculer(el, m, ef)


def lit(c_cm, d, dessous=0.0):
    """La règle des nappes (§ 5.1), écrite ICI indépendamment du module : position réelle = c + Ø de dessous + Ø/2."""
    return {"n": 1, "d": int(d), "pos": round(c_cm + dessous / 10.0 + d / 20.0, 6)}


def moteur(el, b_cm, h_cm, secs):
    o = el["options"]
    return MO.verifier({"b": b_cm, "h": h_cm, "beton": el["dalle"]["beton"], "gamma_s": o["gamma_s"],
                        "enrobage": el["armatures"]["enrobage"] * 100, "jeu_premier_lit": o["jeu"], "arrondi": o["arrondi"],
                        "jeu_entre_lits": 0.0, "sections": secs})["sections"]


def sec(Mi, Ms, V_, li, ls, etriers=None):
    return {"M_inf": max(Mi, 0.0), "M_sup": max(Ms, 0.0), "V": V_, "inf": [li], "sup": [ls],
            "etriers": etriers or [{"type": "epingle", "d": 6}]}


def couverture(nom, el, ef, fer):
    """R1 : chaque élément contrôlé, chaque face, chaque direction — le moteur poutre appelé directement."""
    c_cm, h_cm = el["armatures"]["enrobage"] * 100, el["dalle"]["h"] * 100
    pire, n_el, ecart = 1e9, 0, 0.0
    for cote in ("inf", "sup"):
        f = fer["faces"][cote]
        for sens in ("x", "y"):
            cv = f["couverture"][sens]
            idx = np.flatnonzero(cv["controle"])
            Mi = np.maximum(ef["champ"][f"inf_{sens}"], 0.0)
            Ms = np.maximum(-ef["champ"][f"sup_{sens}"], 0.0)
            opp = lit(c_cm, 20, 20 if sens == "y" else 0)
            secs = []
            for i in idx:
                lf = lit(c_cm, cv["d_couche"][i], cv["dessous"])
                secs.append(sec(Mi[i], Ms[i], 0.0, *((lf, opp) if cote == "inf" else (opp, lf))))
            besoin = np.array([r["grandeurs"][f"besoin_{cote}"] for r in moteur(el, 100.0, h_cm, secs)])
            ecart = max(ecart, float(np.abs(besoin - cv["besoin"][idx]).max()))
            pire = min(pire, float((cv["As_place"][idx] - besoin).min()))
            n_el += len(idx)
    chk(f"R1 {nom} : {n_el} contrôles (éléments × faces × directions, hors zones de renfort et plots coulés) — l'acier en place "
        f"≥ l'acier demandé par le moteur poutre (plus petite marge {pire:+.1f} mm²/m) ; le module rend l'acier du moteur à "
        f"{ecart:.1e} mm²/m près", pire >= -1e-6 and ecart < 1e-6 and n_el > 1000, (pire, ecart, n_el))


def bandes(nom, el, ef, fer):
    """R2 : chaque bande, chaque face — rejoué par le moteur ; renfort nul si le treillis suffit ; barres sur la zone."""
    c_cm, h_cm = el["armatures"]["enrobage"] * 100, el["dalle"]["h"] * 100
    hp_cm = (fer["hp"] or 0.0) * 100
    marge, ecarts, regles = 1e9, 0.0, True
    for cote in ("inf", "sup"):
        for nb, b in fer["faces"][cote]["bandes"].items():
            rb = next(x for x in ef["bandes"] if x["nom"] == nb)
            sens = "x" if nb.startswith("BX") else "y"
            dessous = fer["faces"][cote]["dessous_y"] if sens == "y" else 0.0
            # (la face y repose sur la plus grosse barre en x de SA face — celle que le module a retenue)
            opp = lit(c_cm, 20, 20 if sens == "y" else 0)
            lf = lit(c_cm, b["retenu"]["d_couche"], dessous if sens == "y" else 0.0)
            li, ls = (lf, opp) if cote == "inf" else (opp, lf)
            bcm = rb["largeur"] * 100
            Mp, Mm = max(rb["M_plus"], 0.0), max(-rb["M_moins"], 0.0)
            if fer["coule"] and cote == "inf":
                dans = np.zeros(len(rb["s"]), bool)
                for s1, s2 in rb["charge"]:
                    dans |= (rb["s"] >= s1 - 1e-9) & (rb["s"] <= s2 + 1e-9)
                Md, Mpl = float(np.maximum(rb["Mmax"][~dans], 0).max()), float(np.maximum(rb["Mmax"][dans], 0).max())
                bes = max(moteur(el, bcm, h_cm, [sec(Md, Mm, 0, li, ls)])[0]["grandeurs"]["besoin_inf"],
                          moteur(el, bcm, h_cm + hp_cm, [sec(Mpl, Mm, 0, li, ls)])[0]["grandeurs"]["besoin_inf"]) * 100 / bcm
            else:
                bes = moteur(el, bcm, h_cm, [sec(Mp, Mm, 0, li, ls)])[0]["grandeurs"][f"besoin_{cote}"] * 100 / bcm
            prol = moteur(el, 100.0, h_cm, [sec(max(rb["prol_inf"], 0), max(-rb["prol_sup"], 0), 0, li, ls)])[0]["grandeurs"][f"besoin_{cote}"]
            nec = max(bes, prol)
            ecarts = max(ecarts, abs(nec - b["retenu"]["necessaire_pm"]))
            r = b["renfort"]
            place = b["As_t_pm"] + (r["n"] * math.pi * r["d"] ** 2 / 4 / b["W"] if r else 0.0)
            marge = min(marge, place - nec)
            W = rb["zone"][1] - rb["zone"][0]
            if r:
                regles = regles and r["n"] == max(1, math.ceil(W * 100 / r["s"] - 1e-6)) and abs(r["L"] - (rb["L"] - 2 * el["armatures"]["enrobage"])) < 1e-9
                regles = regles and b["base"]["necessaire_pm"] > b["As_t_pm"] + 1e-9
            else:
                regles = regles and b["base"]["necessaire_pm"] <= b["As_t_pm"] + 1e-9
    chk(f"R2 {nom} : chaque bande, chaque face — treillis + renforts ≥ l'acier demandé par le moteur (bande, prolongement jusqu'au "
        f"bord ; plus petite marge {marge:+.1f} mm²/m) ; renfort nul seulement si le treillis suffit ; barres réparties sur TOUTE "
        f"la largeur de la zone, longueur de la bande moins deux enrobages", marge >= -1e-6 and ecarts < 1e-6 and regles, (marge, ecarts, regles))


# ------------------------------------------------------------------ la relecture du lot 4 (15/11/2026) : dix défauts trouvés
def zones_independantes(ef):
    """Les zones de renfort de chaque direction, recalculées ICI depuis la géométrie des bandes (axe, largeur de zone,
    longueur) : une BX ne pose que des barres en x, une BY qu'en y."""
    ch = ef["champ"]
    z = {"x": np.zeros(len(ch["x"]), bool), "y": np.zeros(len(ch["x"]), bool)}
    for rb in ef["bandes"]:
        u = np.asarray(rb["u"], float)
        nrm = np.array([-u[1], u[0]])
        S = (ch["x"] - rb["p0"][0]) * u[0] + (ch["y"] - rb["p0"][1]) * u[1]
        T = (ch["x"] - rb["p0"][0]) * nrm[0] + (ch["y"] - rb["p0"][1]) * nrm[1]
        z["x" if rb["nom"].startswith("BX") else "y"] |= ((S >= -1e-6) & (S <= rb["s"][-1] + 1e-6) & (T >= rb["zone"][0] - 1e-6)
                                                          & (T <= rb["zone"][1] + 1e-6))
    return z


def revue_lot4():
    """La relecture adverse du lot 4 : chaque défaut confirmé, figé ici (chaque contrôle rougit sans sa correction)."""
    print("== relecture du lot 4 : les défauts confirmés, figés")
    # 1. (non conservatif) dans la zone d'une bande, la direction PERPENDICULAIRE est contrôlée élément par élément
    for nom, kw in (("dalle type", {}), ("bandes de 1,00 m", {"largeur": 1.0})):
        el, (m, ef, fer) = None, (None, None, None)
        el = projet(**kw)
        m, ef, fer = ferrailler(el)
        z = zones_independantes(ef)
        ch = ef["champ"]
        manque, transverses = 0, 0
        for cote in ("inf", "sup"):
            for sens in ("x", "y"):
                cv = fer["faces"][cote]["couverture"][sens]
                doit = ~ch["plot"] & ~z[sens]
                manque += int(np.sum(doit & ~cv["controle"]))
                autre = "y" if sens == "x" else "x"
                transverses += int(np.sum(cv["controle"] & z[autre]))
        chk(f"relecture 1 · {nom} : chaque élément hors plot et hors de la zone d'une bande de SA direction est contrôlé un à un "
            f"(dans la zone d'une BX, les barres en y le sont aussi : {transverses} contrôles transversaux) — aucun oublié",
            manque == 0 and transverses > 0, (manque, transverses))
        if not kw:
            # la zone courante (la hauteur) : en x hors des zones des BX, en y hors des zones des BY
            cour = {k: ~ch["plot"] & ~z[k] for k in ("x", "y")}
            att = {f"inf_{k}": float(ch[f"inf_{k}"][cour[k]].max()) for k in ("x", "y")}
            chk("relecture 1 · la zone courante se lit direction par direction (les moments transversaux des zones de renfort y entrent) : "
                f"inf_x = {att['inf_x']:.1f}, inf_y = {att['inf_y']:.1f} kNm/m",
                all(abs(ef["courante"][k] - v) < 1e-9 for k, v in att.items()), (att, {k: ef["courante"][k] for k in att}))
            i = int(np.argmin((ch["x"] - 1.3) ** 2 + (ch["y"] - 6.9) ** 2))
            cv = fer["faces"]["inf"]["couverture"]["y"]
            chk("relecture 1 · l'élément (1,3 ; 6,9), face inf., direction y — dans la zone de BX1, hors de BY1 — est contrôlé et couvert "
                f"({cv['besoin'][i]:.1f} ≤ {cv['As_place'][i]:.1f} mm²/m)",
                bool(cv["controle"][i]) and cv["besoin"][i] <= cv["As_place"][i] + 1e-6 and cv["besoin"][i] > 1400, (cv["besoin"][i], cv["As_place"][i]))
    # 2. (non conservatif) les C vont partout où |V(s)| demande des armatures, jusqu'au bord de la zone chargée
    el = projet(h=0.35)
    m, ef, fer = ferrailler(el)
    t = fer["tranchant"]["BX1"]
    rb = next(b for b in ef["bandes"] if b["nom"] == "BX1")
    li, ls = fer["faces"]["inf"]["bandes"]["BX1"]["retenu"]["lit"], fer["faces"]["sup"]["bandes"]["BX1"]["retenu"]["lit"]
    bt = [r["grandeurs"]["besoin_tau"] for r in moteur(el, rb["largeur"] * 100, 35.0, [sec(0, 0, float(v), li, ls) for v in rb["Vabs"]])]
    hors = np.all([(rb["s"] <= s1 + 1e-9) | (rb["s"] >= s2 - 1e-9) for s1, s2 in rb["charge"]], axis=0)
    demande = [float(s_) for s_, b_, h_ in zip(rb["s"], bt, hors) if h_ and b_ not in (None, "pas_d_etriers")]
    zs = (t["C"] or {}).get("zones", [])
    couvert = all(any(z0 - 1e-9 <= s_ <= z1 + 1e-9 for z0, z1 in zs) for s_ in demande)
    bords = all(any(abs(z1 - s1) < 1e-9 or abs(z0 - s2) < 1e-9 for z0, z1 in zs) for s1, s2 in rb["charge"])
    chk(f"relecture 2 · h = 0,35 m, BX1 : des C partout où |V(s)| demande des armatures hors de la zone chargée ({len(demande)} abscisses), "
        f"jusqu'au BORD de la zone chargée — entre ce bord et d, |V| est le plus fort ({t['C'] and t['C']['n']} C)",
        bool(demande) and couvert and bords, (zs, demande[:6]))
    # 3. le treillis automatique : le plus léger parmi ceux qui CONVIENNENT
    el = projet(h=0.35, jeux=[tuple(2 * v for v in j) for j in JEUX])
    fer = ferrailler(el)[2]
    fs = fer["faces"]["sup"]
    valides = [c for c in fs["candidats"] if c["ok"]]
    leger_faux = [c for c in fs["candidats"] if not c["ok"] and c["poids"] < min(v["poids"] for v in valides)] if valides else []
    chk("relecture 3 · h = 0,35 m, réactions × 2, face sup. automatique : un candidat plus léger ne convient pas (ses renforts impossibles "
        f"« pèsent » 0) — le module retient le plus léger de ceux qui conviennent ({fs['code']})",
        bool(leger_faux) and fs["ok"] and fs["code"] == min(valides, key=lambda c: c["poids"])["code"],
        ([(c["code"], round(c["poids"]), c["ok"]) for c in fs["candidats"]], fs["code"]))
    # 4. τ en bande IV : le moteur demande des barres inclinées — la bande est « à vérifier », jamais conforme
    jeux4 = [list(j) for j in JEUX]
    jeux4[2][2] = 1900
    fer = ferrailler(projet(jeux=[tuple(j) for j in jeux4]))[2]
    t = fer["tranchant"]["BX1"]
    chk("relecture 4 · P3 = 1 900 kN : τ en bande IV (barres inclinées et étriers demandés), le module ne pose que des C — « à vérifier » : "
        "une ligne « Barres inclinées » et les barres inclinées dans « Non vérifiés »",
        t["besoin_tau"] == "barres_inclinees_et_etriers" and fer["a_verifier"] and any(x[0] == "Barres inclinées" for x in fer["lignes"])
        and any("barres inclinées" in x for x in fer["non_verifies"]), (t["besoin_tau"], fer["a_verifier"], fer["non_verifies"]))
    # 5. V à d illisible (plots coulés plus proches que 2d) : « non vérifié », jamais V = 0
    el = PR.nouveau()
    el["reactions"] = [{"nom": "", "P": list(j)} for j in JEUX]
    el["grue"]["a"] = el["grue"]["b"] = 4.0
    el["grue"]["plots"] = {"type": "coule", "a": 3.2, "b": 3.2, "h": 0.35}
    el = PR.lire(el)
    m, ef, fer = ferrailler(el)
    rb = ef["bandes"][0]
    chk("relecture 5 · châssis 4 × 4, plots coulés 3,20 m : aucune abscisse à d des zones chargées — V à d ne se lit pas (None, pas 0) ; "
        "l'effort tranchant de chaque bande est « non vérifié » et nommé dans « Non vérifiés »",
        EF.v_a_d(rb, 0.43) is None and all(t["V"] is None and t["etat"] == "non_verifie" for t in fer["tranchant"].values())
        and fer["a_verifier"] and any("V à d ne se lit pas" in x for x in fer["non_verifies"]),
        ({n: (t["V"], t["etat"]) for n, t in fer["tranchant"].items()}, fer["non_verifies"]))


# ------------------------------------------------------------------ les contrôles RAPIDES (rejoués par les mutations)
def rapides():
    t = time.time()
    el0 = projet()
    m0, ef0, f0 = ferrailler(el0)
    print(f"== R1, R2 — dalle type, défauts du lot 4 (ferraillage en {time.time() - t:.2f} s, efforts compris)")
    couverture("dalle type (treillis automatiques)", el0, ef0, f0)
    bandes("dalle type", el0, ef0, f0)
    loc = [r for cote in ("inf", "sup") for r in f0["faces"][cote]["locaux"]]
    ext = True
    ch = ef0["champ"]
    for r in loc:
        p = r["elements"]
        lr = el0["options"]["recouvrement"] * r["d"] / 1000.0
        if r["sens"] == "x":
            g0, g1 = ch["x0"] + ch["I"][p].min() * ch["a"], ch["x0"] + (ch["I"][p].max() + 1) * ch["a"]
            ext = ext and abs(r["rect"][0] - max(g0 - lr, 0.05)) < 1e-9 and abs(r["rect"][2] - min(g1 + lr, 7.95)) < 1e-9
        else:
            g0, g1 = ch["y0"] + ch["J"][p].min() * ch["b"], ch["y0"] + (ch["J"][p].max() + 1) * ch["b"]
            ext = ext and abs(r["rect"][1] - max(g0 - lr, 0.05)) < 1e-9 and abs(r["rect"][3] - min(g1 + lr, 7.95)) < 1e-9
    chk(f"renforts locaux (§ 5.4) : {len(loc)} rectangles parallèles aux axes, allongés d'un recouvrement (50 Ø) de chaque côté dans le "
        "sens des barres, sans sortir de la dalle (à c du bord), aucun chevauchement dans une même face et direction",
        len(loc) > 0 and ext and _sans_chevauchement(loc), [(r["sens"], r["rect"]) for r in loc][:4])

    print("== R3 — le treillis automatique, le plus léger des quatre")
    poids = {}
    for code in FE.TREILLIS_CHOIX:
        _, _, fx = ferrailler(projet(treillis=(code, code)))
        poids[code] = (fx["faces"]["inf"]["poids"], fx["faces"]["sup"]["poids"])
    choix = tuple(min(FE.TREILLIS_CHOIX, key=lambda c: (round(poids[c][k], 6), FE.TREILLIS_CHOIX.index(c))) for k in (0, 1))
    chk(f"R3 dalle type : chaque face prend le treillis du poids total le plus faible (treillis + renforts locaux + renforts de bandes), "
        f"chacun des 4 calculé séparément — inf. {choix[0]}, sup. {choix[1]} ; poids (kg) "
        + " ; ".join(f"{c} : {poids[c][0]:.0f} / {poids[c][1]:.0f}" for c in FE.TREILLIS_CHOIX),
        (f0["faces"]["inf"]["code"], f0["faces"]["sup"]["code"]) == choix and f0["faces"]["inf"]["auto"]
        and [c["code"] for c in f0["faces"]["inf"]["candidats"]] == list(FE.TREILLIS_CHOIX)
        and all(abs(c["poids"] - poids[c["code"]][0]) < 1e-6 for c in f0["faces"]["inf"]["candidats"]), (choix, poids))

    print("== R4 — un treillis imposé")
    el4 = projet(treillis=("10-15", "10-15"))
    _, ef4, f4 = ferrailler(el4)
    chk("R4 treillis imposés Ø10 e = 15 cm en bas et en haut : ils sont gardés (non « automatique »), les renforts se recalculent "
        "(plus de renforts qu'avec le choix automatique)",
        (f4["faces"]["inf"]["code"], f4["faces"]["sup"]["code"]) == ("10-15", "10-15") and not f4["faces"]["inf"]["auto"]
        and f4["faces"]["sup"]["poids_R"] + f4["faces"]["sup"]["poids_L"] >= f0["faces"]["sup"]["poids_R"] + f0["faces"]["sup"]["poids_L"] - 1e-9,
        (f4["faces"]["sup"]["poids_R"], f4["faces"]["sup"]["poids_L"]))
    couverture("treillis imposés Ø10 e = 15", el4, ef4, f4)
    bandes("treillis imposés Ø10 e = 15", el4, ef4, f4)

    print("== R5 — les U de fermeture")
    el5 = projet(treillis=("12-15", "12-15"))
    _, _, f5 = ferrailler(el5)
    u = f5["U"]
    chk("R5 dalle type, treillis Ø12 e = 15 cm, c = 5 cm : U Ø12 e = 15, partie entière de (8,00 − 0,10) ÷ 0,15 + 1 = 53 U par bord "
        "(212 en tout) ; âme h − 2c = 0,40 m, branches = 50 Ø = 0,60 m, longueur 1,60 m",
        (u["d"], u["e"]) == (12, 15.0) and [b["n"] for b in u["bords"]] == [53] * 4 and u["n"] == 212
        and abs(u["ame"] - 0.40) < 1e-12 and abs(u["branche"] - 0.60) < 1e-12 and abs(u["L"] - 1.60) < 1e-12, (u["d"], u["e"], [b["n"] for b in u["bords"]]))
    fort = max((f0["faces"]["inf"], f0["faces"]["sup"]), key=lambda f: (f["As_pm"], f["d"]))
    chk(f"R5 treillis automatiques : les U prennent le Ø et l'espacement du treillis le plus FORT des deux faces ({fort['code']})",
        (f0["U"]["d"], f0["U"]["e"]) == (fort["d"], fort["e"]), (f0["U"]["d"], f0["U"]["e"], fort["code"]))

    print("== R6 — le bordereau")
    bd = f0["bordereau"]
    somme = sum(ln["n"] * ln["L"] * MASSES[ln["d"]] for ln in bd["lignes"])
    chk(f"R6 dalle type : poids total = Σ nombre × longueur × poids au mètre = {bd['total']:.1f} kg (masses du cahier : Ø10 = 0,617 ; "
        f"Ø12 = 0,888 ; Ø16 = 1,578 ; Ø20 = 2,466 kg/m, à 0,1 % près) ; béton = 8 × 8 × 0,50 = 32,0 m³ ; ratio {bd['ratio']:.1f} kg/m³ ; "
        "totaux par Ø ; repères T1 à T4, R, L, U",
        abs(somme - bd["total"]) / bd["total"] < 1e-3 and abs(bd["beton"] - 32.0) < 1e-9 and abs(bd["ratio"] - bd["total"] / 32.0) < 1e-9
        and abs(sum(bd["par_d"].values()) - bd["total"]) < 1e-6 and all(abs(FE.masse(d) - v) < 5e-4 for d, v in MASSES.items())
        and {ln["rep"] for ln in bd["lignes"] if ln["famille"] == "T"} == {"T1", "T2", "T3", "T4"} and any(ln["rep"] == "U1" for ln in bd["lignes"])
        and all(abs(ln["L"] - sum(ln["cotes"])) < 1e-9 for ln in bd["lignes"]), (somme, bd["total"], bd["beton"]))
    for cote in ("inf", "sup"):
        f = f0["faces"][cote]
        lignes = [ln for ln in bd["lignes"] if ln.get("cote") == cote and ln["famille"] in "TRL"]
        chk(f"R6 face {cote}. : le poids qui a choisi le treillis (treillis + renforts de bandes + renforts locaux, {f['poids']:.1f} kg) = "
            "celui des lignes T, R et L du bordereau (au 5 cm près des longueurs)",
            abs(sum(ln["poids"] for ln in lignes) - f["poids"]) <= 0.01 * f["poids"], (sum(ln["poids"] for ln in lignes), f["poids"]))

    print("== R7 — les bandes de la dalle type")
    b7 = f0["faces"]["inf"]["bandes"]["BX1"]
    bsup = f0["faces"]["sup"]["bandes"]["BX1"]
    g7 = b7["retenu"]["moteur"]["grandeurs"]
    rb = ef0["bandes"][0]
    t7 = f0["tranchant"]["BX1"]
    chk(f"R7 bandes de 0,50 m (défaut) : BX1 M+ = 71,6 kNm, soit 143,2 kNm/m ; M− = −63,3 kNm ; le moteur poutre appelé pour une "
        f"poutre 0,50 × 0,50 m (b = {g7['b']:g} cm, h = {g7['h']:g} cm) donne l'acier demandé {g7['besoin_inf']:.1f} mm² ; V à d au d du "
        f"moteur (d = {t7['d_v']:.3f} m) : {t7['V']:.1f} kN — avec d = 0,44 m : 94,0 kN",
        round(rb["M_plus"], 1) == 71.6 and round(rb["M_plus"] / 0.5, 1) == 143.2 and round(rb["M_moins"], 1) == -63.3
        and (g7["b"], g7["h"]) == (50.0, 50.0) and abs(b7["retenu"]["besoin_pm"] - g7["besoin_inf"] * 2) < 1e-9
        and round(b7["retenu"]["M"], 1) == 71.6 and round(bsup["retenu"]["M"], 1) == 63.3
        and abs(t7["V"] - EF.v_a_d(rb, t7["d_v"])) < 1e-12 and round(EF.v_a_d(rb, 0.44), 1) == 94.0
        and abs(t7["d_v"] * 100 - t7["moteur"]["grandeurs"]["d_cisaillement"]) < 1e-9, (rb["M_plus"], g7["b"], t7["d_v"], t7["V"]))
    _, ef1, f1 = ferrailler(projet(largeur=1.0))
    chk("R7 bandes de 1,00 m : 127,1 et −126,5 kNm, poutre 1,00 × 0,50 m",
        round(f1["faces"]["inf"]["bandes"]["BX1"]["retenu"]["M"], 1) == 127.1 and round(f1["faces"]["sup"]["bandes"]["BX1"]["retenu"]["M"], 1) == 126.5
        and f1["faces"]["inf"]["bandes"]["BX1"]["retenu"]["moteur"]["grandeurs"]["b"] == 100.0, None)
    bandes("bandes de 1,00 m (BX1, BY1 prolongées jusqu'au bord : zone de 1,50 m)", projet(largeur=1.0), ef1, f1)

    print("== R8 — le plot coulé en place")
    el8 = projet(plots=COULE)
    m8, ef8, f8 = ferrailler(el8)
    fi = f8["faces"]["inf"]
    n_cote = int(math.floor((2.0 - 0.10) / (fi["e"] / 100) + 1e-9)) + 1
    cg = f8["cages"]
    chk(f"R8 plots coulés 2,00 × 2,00 × 0,35 : 4 cages — C verticaux sur les 4 faces (Ø et espacement du treillis inférieur, "
        f"{fi['code']} : 2 × {n_cote} par direction), montant h + hp − 2c = 0,75 m, branche haute = côté − 2c = 1,90 m, branche "
        "basse = un recouvrement ; 2 U horizontaux en tête (1,90 m) ; brins verticaux sous le pied « à dimensionner »",
        len(cg) == 4 and all(len(x["C"]) == 2 and [c["n"] for c in x["C"]] == [2 * n_cote] * 2 for x in cg)
        and all(abs(c["montant"] - 0.75) < 1e-9 and abs(c["haute"] - 1.90) < 1e-9 and abs(c["basse"] - 50 * fi["d"] / 1000) < 1e-9
                for x in cg for c in x["C"])
        and all(len(x["U"]) == 2 and all(abs(uu["ame"] - 1.9) < 1e-9 and abs(uu["branche"] - 1.9) < 1e-9 for uu in x["U"]) for x in cg)
        and all(x["d"] == fi["d"] for x in cg) and all("à dimensionner" in x["brins"] for x in cg)
        and ("Brins verticaux sous le pied", "à dimensionner (poinçonnement non vérifié)", "inc") in f8["lignes"]
        and "les brins verticaux sous le pied" in f8["non_verifies"], [(x["C"][0]["n"], x["C"][0]["montant"]) for x in cg])
    couverture("plots coulés en place (éléments sous les plots exclus : leurs moments vont dans la carte PLOTS)", el8, ef8, f8)
    bandes("plots coulés (section h + plot au droit d'un plot, en face inférieure)", el8, ef8, f8)
    bd8 = f8["bordereau"]
    chk("R6 plots coulés : béton = 32,0 + 4 × 2,00 × 2,00 × 0,35 = 37,6 m³ ; les C et les U des cages au bordereau ; plots préfabriqués : "
        "béton de la dalle seule (dalle 10 × 10 × 0,50 = 50,0 m³), pas de cage",
        abs(bd8["beton"] - 37.6) < 1e-9 and any(ln["famille"] == "C" and ln["forme"] == "C" for ln in bd8["lignes"])
        and any(ln["famille"] == "C" and ln["forme"] == "U" for ln in bd8["lignes"])
        and (lambda fp: fp["cages"] == [] and abs(fp["bordereau"]["beton"] - 50.0) < 1e-9)(_pref()), bd8["beton"])

    print("== effort tranchant, arrachement, poinçonnement")
    el9 = projet(h=0.35)
    _, ef9, f9 = ferrailler(el9)
    t9 = f9["tranchant"]["BX1"]
    C9 = t9["C"]
    rb9 = ef9["bandes"][0]
    rr = moteur(el9, 50.0, 35.0, [sec(0, 0, t9["V"], f9["faces"]["inf"]["bandes"]["BX1"]["retenu"]["lit"],
                                      f9["faces"]["sup"]["bandes"]["BX1"]["retenu"]["lit"], [{"type": "epingle", "d": C9["d"]}] * C9["branches"])])[0]
    chk(f"dalle de 0,35 m : le moteur demande des armatures d'effort tranchant (τ en bande II) → C verticaux Ø{C9['d']} à "
        f"{C9['s']:g} cm (le pas du moteur), {C9['branches']} branches par rangée, {C9['rangs']} rangées sur les zones où |V| dépasse ce que "
        "la dalle reprend sans armatures ; au bordereau (repère V)",
        t9["besoin_tau"] == "etriers" and C9 is not None and C9["s"] == rr["grandeurs"]["s_lim_tronque"]
        and C9["rangs"] == sum(int(math.floor((z1 - z0) * 100 / C9["s"] + 1e-9)) + 1 for z0, z1 in C9["zones"])
        and C9["n"] == C9["rangs"] * C9["branches"] and any(ln["famille"] == "V" for ln in f9["bordereau"]["lignes"])
        and len(C9["zones"]) == 4, (t9["besoin_tau"], C9))
    chk("dalle type : pas d'armature d'effort tranchant (bande I, « pas d'étriers »), taux τ/τ_adm affiché",
        all(t["besoin_tau"] == "pas_d_etriers" and t["C"] is None and 0 < t["taux"] < 1 for t in f0["tranchant"].values()), None)
    try:
        projet(jeux=[(135, 100, 800, -20)])
        refus = None
    except ValueError as e:
        refus = str(e)
    el_a = projet(jeux=[(135, 100, 800, 20)])
    m_a = PR.modele(el_a)
    rs_a = [V.calculer(m_a, k, k / 10, k / 10) for k in (2.0, 5.0, 10.0)]
    el_a["reactions"][0]["P"][3] = -20.0                      # (un projet relu d'ailleurs : la saisie, elle, refuse)
    fa = FE.calculer(el_a, m_a, EF.enveloppe(m_a, rs_a, 0.44))
    chk("une réaction négative : la saisie la refuse (« une réaction négative est un arrachement : non traité ») ; le ferraillage d'un "
        "projet qui en porterait une écrit « Arrachement : non traité » en rouge et n'est pas conforme ; le poinçonnement est « non "
        "vérifié » (le logiciel n'a pas de vérification selon la méthode du bureau)",
        refus == "une réaction négative est un arrachement : non traité"
        and fa["arrachement"] and ("Arrachement", "une réaction négative", "nok") in fa["lignes"] and not fa["ok"]
        and "l'arrachement" in fa["non_verifies"] and f0["lignes"][0][0] == "Poinçonnement" and "le poinçonnement" in f0["non_verifies"]
        and f0["ok"], fa["lignes"])
    besoin = f0["faces"]["inf"]["couverture"]["x"]["besoin"].copy()
    i = int(np.flatnonzero(f0["faces"]["inf"]["couverture"]["x"]["controle"])[0])
    besoin[i] *= 1.01
    ctx = FE._Contexte(el0, m0, ef0)
    Mi, Ms = ctx.moments("x")
    try:
        FE._recontroler(ctx, "inf", besoin, {i: FE._couche(5.0, "x", f0["faces"]["inf"]["d"], 0)}, FE._opposee(5.0, "x"), Mi, Ms, np.array([i]))
        leve = False
    except RuntimeError:
        leve = True
    chk("le recontrôle par le moteur : un acier par élément qui s'écarte du moteur poutre (de 1 %) lève une erreur au lieu d'être rendu", leve)
    # le choix d'un renfort (§ 5.3) : le plus LÉGER qui couvre ; les espacements dans les intervalles du treillis
    ch_ = FE.choisir_renfort({12: 500.0, 16: 500.0, 20: 500.0}, 12, 15.0)
    chk("le renfort le plus léger qui couvre 500 mm²/m sur un treillis Ø12 e = 15 : Ø20 tous les 60 cm (524 mm²/m, 4,11 kg/m²) — et non "
        "Ø12 e = 15 (5,92 kg/m²), Ø16 e = 30 (5,26) ni Ø20 e = 45 (5,48) ; deux renforts par intervalle seulement si l'espacement libre du "
        "moteur poutre le permet (Ø12 dans un treillis e = 15 : oui ; Ø20 dans un treillis e = 10 : non, 17 mm < 25 mm)",
        ch_ == (20, 60.0) and (7.5, 0) in FE.options_espacement(12, 10, 15.0) and all(sp != 5.0 for sp, _ in FE.options_espacement(20, 12, 10.0))
        and FE.choisir_renfort({12: 5000.0, 16: 5000.0, 20: 5000.0}, 12, 15.0) is None, ch_)
    # la règle du moteur dans les éléments, terme « 0,25·A_s,req de la face opposée » compris : des moments où chaque terme gouverne
    Mi = np.array([0.0, 40.0, 180.0, 0.0, 600.0])
    Ms = np.array([600.0, 0.0, 20.0, 0.0, 0.0])
    for cote in ("inf", "sup"):
        for sens in ("x", "y"):
            lf, lo = lit(5.0, 12, 12 if sens == "y" else 0), lit(5.0, 20, 20 if sens == "y" else 0)
            b_ = FE.besoin_elements(ctx, cote, sens, lf, lo, Mi, Ms)
            ref = [r["grandeurs"] for r in moteur(el0, 100.0, 50.0, [sec(a_, b2, 0, *((lf, lo) if cote == "inf" else (lo, lf))) for a_, b2 in zip(Mi, Ms)])]
            chk(f"l'acier par élément = celui du moteur poutre, face {cote}., direction {sens} — A_s,req, A_s,min de base et 0,25·A_s,req "
                "opposé gouvernant tour à tour",
                np.allclose(b_, [g[f"besoin_{cote}"] for g in ref], rtol=1e-12, atol=1e-9)
                and {g[f"As_min_terme_{cote}"] for g in ref} >= {"opposee", "ec"}, (list(b_), [g[f"besoin_{cote}"] for g in ref]))

    revue_lot4()


def _pref():
    return ferrailler(projet(plots=PREF, L=10.0))[2]


def _sans_chevauchement(loc):
    for i, a in enumerate(loc):
        for b in loc[i + 1:]:
            if a["face"] == b["face"] and a["sens"] == b["sens"]:
                r1, r2 = a["rect"], b["rect"]
                if r1[0] < r2[2] - 1e-9 and r2[0] < r1[2] - 1e-9 and r1[1] < r2[3] - 1e-9 and r2[1] < r1[3] - 1e-9:
                    return False
    return True


# ------------------------------------------------------------------ mutations (arbre miroir, contrôles rapides)
MUTATIONS = [
    ("ferraillage.py", "    return np.maximum(np.maximum(kf * Mf, base), MO.K_AS_MIN_OPPOSEE * ko * Mo)", "    return np.maximum(kf * Mf, base)",
     "l'acier minimal de la face opposée oublié dans les éléments"),
    ("ferraillage.py", '    return _lit(c_cm, d, dessous_y if sens == "y" else 0.0)', "    return _lit(c_cm, d, 0.0)",
     "les barres en y au niveau des barres en x (d trop grand)"),
    ("ferraillage.py", '        out["necessaire_pm"] = max(out["besoin_pm"], out["prol_pm"])', '        out["necessaire_pm"] = out["besoin_pm"]',
     "le prolongement de la zone jusqu'au bord ignoré"),
    ("ferraillage.py", '                       besoin_pm=max(r["grandeurs"]["besoin_inf"], rp["grandeurs"]["besoin_inf"]) * 100.0 / b_cm)',
     '                       besoin_pm=r["grandeurs"]["besoin_inf"] * 100.0 / b_cm)', "la section h + plot ignorée au droit d'un plot coulé"),
    ("ferraillage.py", "            cle = (round(masse(d) * 100.0 / s, 9), -d)", "            cle = (-round(masse(d) * 100.0 / s, 9), -d)",
     "le renfort le plus LOURD"),
    ("ferraillage.py", '        bords.append({"de": tuple(p), "a": tuple(q), "L": L, "n": int(math.floor((L - 2 * ctx.c) / e + 1e-9)) + 1})',
     '        bords.append({"de": tuple(p), "a": tuple(q), "L": L, "n": int(math.floor((L - 2 * ctx.c) / e + 1e-9))})', "un U de moins par bord"),
    ("ferraillage.py", '    fort = max((fi, fs), key=lambda f: (f["As_pm"], f["d"]))', '    fort = min((fi, fs), key=lambda f: (f["As_pm"], f["d"]))',
     "les U du treillis le plus faible"),
    ("ferraillage.py", "    montant = ctx.h + ctx.hp - 2 * c", "    montant = ctx.h - 2 * c", "le montant des C sans la hauteur du plot"),
    ("ferraillage.py", '            "Lt": n * L, "poids": n * L * masse(d), "libelle": libelle, **plus}', '            "Lt": n * L, "poids": n * masse(d), "libelle": libelle, **plus}',
     "le poids sans la longueur"),
    ("ferraillage.py", '(sum(G.aire(q) for q in pl["polys"]) * pl["hauteur"] if pl and pl["type"] == "coule" else 0.0)', '0.0',
     "le béton des plots coulés oublié"),
    ("ferraillage.py", "            f = min(valides, key=lambda x: (round(x[\"poids\"], 6), TREILLIS_CHOIX.index(x[\"code\"])))",
     "            f = max(valides, key=lambda x: (round(x[\"poids\"], 6), TREILLIS_CHOIX.index(x[\"code\"])))", "le treillis le plus lourd"),
    ("ferraillage.py", '            valides = [x for x in cands if x["ok"]] or cands', "            valides = cands",
     "le treillis automatique retient un candidat qui ne convient pas (ses renforts impossibles pèsent 0)"),
    ("ferraillage.py", '        self.controle = {"x": ~ch["zone_x"] & ~ch["plot"], "y": ~ch["zone_y"] & ~ch["plot"]}',
     '        self.controle = {"x": ~ch["zone_renfort"] & ~ch["plot"], "y": ~ch["zone_renfort"] & ~ch["plot"]}',
     "la direction perpendiculaire non contrôlée dans les zones de renfort"),
    ("efforts.py", '            zone_sens["x" if g["bande"]["nom"].startswith("BX") else "y"] |= dz',
     '            zone_sens["x"] |= dz\n            zone_sens["y"] |= dz', "les zones de renfort réunies pour les deux directions"),
    ("efforts.py", '                for zone_, plot_ in ((cour[k] & ~dans_plot, False), (cour[k] & dans_plot, True)):',
     '                for zone_, plot_ in ((cour["x"] & cour["y"] & ~dans_plot, False), (cour["x"] & cour["y"] & dans_plot, True)):',
     "la zone courante lue hors de TOUTES les zones (les moments transversaux oubliés)"),
    ("ferraillage.py", '    arm = np.array([bt not in (None, "pas_d_etriers") for bt in besoins]) & hors',
     '    arm = np.array([bt not in (None, "pas_d_etriers") for bt in besoins]) & EF.lecture_v(rb["s"], rb["charge"], d_v)',
     "les C arrêtés à d du bord de la zone chargée"),
    ("ferraillage.py", '    inclinees = [n for n, t in tr.items() if t["besoin_tau"] == "barres_inclinees_et_etriers"]', "    inclinees = []",
     "τ en bande IV dit conforme (les barres inclinées oubliées)"),
    ("efforts.py", '    return float(bande["Vabs"][ok].max()) if ok.any() else None', '    return float(bande["Vabs"][ok].max()) if ok.any() else 0.0',
     "V à d illisible pris pour 0"),
    ("ferraillage.py", "    rouge = np.flatnonzero(ctx.controle[sens] & (besoins[d_t] > As_t + EPS))", "    rouge = np.flatnonzero(ctx.controle[sens] & False)",
     "aucun renfort local"),
    ("ferraillage.py", '            if sens == "x":\n                X0, X1 = X0 - lr, X1 + lr', '            if False:\n                X0, X1 = X0 - lr, X1 + lr',
     "les renforts locaux en x sans recouvrement"),
    ("ferraillage.py", '    W = rb["zone"][1] - rb["zone"][0]\n', '    W = b\n', "le renfort de bande sur la seule bande (pas sur toute la zone)"),
    ("ferraillage.py", "    V = EF.v_a_d(rb, d_v)", '    V = rb["V_d"]', "V à d lu avec un d qui n'est pas celui du moteur"),
    ("ferraillage.py", '            raise RuntimeError(f"ferraillage : l\'acier demandé par élément s\'écarte du moteur poutre ({besoin[i]} ≠ {v})")',
     "            pass", "le recontrôle qui ne dit rien"),
    ("ferraillage.py", '    arr = any(v is not None and v < 0 for j in el["reactions"] for v in j["P"])', "    arr = False", "l'arrachement ignoré"),
    ("ferraillage.py", '    rangs = sum(int(math.floor((z1 - z0) * 100.0 / s_lim + 1e-9)) + 1 for z0, z1 in zones)',
     '    rangs = int(math.floor(sum(z1 - z0 for z0, z1 in zones) * 100.0 / s_lim + 1e-9)) + 1', "les rangées de C comptées sur une seule zone"),
]
TEMOIN = ("ferraillage.py", "RETOUR_C = 10 ", "RETOUR_C = 10 ")


def muter(fichier, ancien, nouveau):
    with tempfile.TemporaryDirectory() as tmp:
        for paquet in ("saf", "grues"):
            shutil.copytree(os.path.join(RACINE, paquet), os.path.join(tmp, paquet), ignore=shutil.ignore_patterns("__pycache__"))
        os.symlink(os.path.join(RACINE, "beton"), os.path.join(tmp, "beton"))
        os.symlink(os.path.join(RACINE, "beton_classes.json"), os.path.join(tmp, "beton_classes.json"))
        cible = os.path.join(tmp, "saf", "fondation_grue", fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rapide", "--racine", tmp], capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    rapides()
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    print("== pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('scipy', 'streamlit', 'pandas', 'matplotlib', 'reportlab', 'PIL', 'openpyxl'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from saf.fondation_grue import projet as PR, verification as V, efforts as EF, ferraillage as FE\n"
            "el = PR.nouveau(); el['reactions'] = [{'nom': '', 'P': [135, 100, 800, 135]}]; el = PR.lire(el); m = PR.modele(el)\n"
            "rs = [V.calculer(m, k, k / 10, k / 10) for k in (2.0, 5.0, 10.0)]\n"
            "print(FE.calculer(el, m, EF.enveloppe(m, rs, 0.44))['ok'])\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le ferraillage se charge et calcule avec numpy seul (scipy, streamlit, pandas, reportlab, openpyxl refusés)",
        p.returncode == 0 and p.stdout.strip() == "True", (p.stdout.strip(), p.stderr[-300:]))
    print("== mutations")
    trouve, code_ = muter(*TEMOIN)
    chk("témoin : le moteur recopié sans rien changer passe les contrôles rapides", trouve and code_ == 0, code_)
    for fichier, ancien, nouveau, nom in MUTATIONS:
        trouve, code_ = muter(fichier, ancien, nouveau)
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code_ == 1,
            "texte absent ou répété" if not trouve else f"code {code_} : AUCUN contrôle ne rougit")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
