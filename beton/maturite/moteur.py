# -*- coding: utf-8 -*-
"""Maturité du béton — le MOTEUR, sans Streamlit ni matplotlib ni réseau : bibliothèque standard et numpy (présent
dans Pyodide). Extrait de l'écran Streamlit ``modules/age_beton.py`` v3.2, qui reste la RÉFÉRENCE :
``tests/test_maturite_moteur.py`` rejoue ``show()`` avec un faux « st » et prouve, cas par cas, que ce moteur rend les
mêmes nombres (au bit près), les mêmes états et les mêmes courbes. Méthode, provenance et preuve :
``docs/beton/MATURITE_MOTEUR.md``.

API
---
    nouvel_element(nom) -> dict            un coulage aux valeurs de l'écran au premier affichage
    normaliser(element) -> dict            copie complétée et contrôlée (ValueError hors du domaine de l'écran)
    calculer(element) -> dict              tout ce que l'écran affiche (et ce que le site ajoute, nommé comme tel)
    candidats_classes(element) -> list     les classes de béton pour la cible ③ (le sélecteur du site)

ENTRÉE — un coulage (JSON). Unités : °C, jours, MPa, kJ/mol.
------
    {"nom": "Coulage 1",
     "classe": "C30/37",                 « Classe béton » (C20/25 … C50/60)
     "ciment": "N",                      « Classe ciment » : R (s = 0,20), N (0,25), S (0,38)
     "date_coulage": null,               « Date de coulage » (AAAA-MM-JJ) : date les seuils
     "temperature": {"mode": "constante" | "releve" | "meteo", « Manuel » | « Météo » : relevé saisi, ou météo du lieu
                     "T": 20.0,          « T moyenne (°C) » (−5 à 60), mode constante
                     "releve": "",       le relevé journalier (beton/maturite/releve.py : format) — collé, lu, ou écrit
                                         par la météo récupérée (et corrigé à la main si l'on veut)
                     "fichier": "", "lieu": "",               le nom du fichier lu, le lieu (« Lieu du chantier »)
                     "debut": null, "fin": null,              « Coulage », « Fin (décoffrage) » de la météo (AAAA-MM-JJ ;
                                                              null : les dates de l'écran au premier affichage)
                     "meteo": null | {...}},                  la récupération GARDÉE (beton/maturite/meteo.py : lieu,
                                                              dates, jour et heure, réponses d'Open-Meteo réduites)
     "avance": false, "Ea_kJ": 33.256,   « ⚙️ » et « Ea [kJ/mol] » (20 à 60) ; sans ⚙️, E_a = 4000·R
     "age": 28,                          « Âge (jours réels) » : entier, 1 à 28 (1 à n jours du relevé)
     "seuil": {"mode": "pct" | "mpa", "pct": 75, "mpa": 20.0, "afficher": false},                       ②
     "optim": {"mpa": 30.0, "delai": 14.0, "afficher": false},                                           ③
     "comparateur": {"classes": [...], "ciment": "N", "afficher": false},                                 ④
     "essai": {"mpa": 0.0, "age": 7.0, "T": 20.0, "afficher": false}}                                     ⑤

SORTIE de calculer — JSON strict (sans NaN), nombres non arrondis (le site formate)
------
    {"reference", "entree",
     "materiau": {"classe", "fck28", "fcm", "ciment", "ciment_libelle", "ciment_detail", "s"},
     "temperature": {"mode", "source": "manuelle" | "releve", "lieu", "T" (T de calcul : la constante ou la moyenne
                     du relevé), "Ea", "Ea_kJ", "Ea_R", "T_abs", "facteur" (jours équivalents par jour réel à T),
                     "t28_reel" (28/facteur), "t3_reel" (3/facteur, site), "releve": <synthèse> | None,
                     "defav": {...} | None, "meteo": <meteo.synthese> | None (mode meteo, météo récupérée)},
     "age": <①>, "seuil": <②>, "optim": <③>, "comparateur": <④>, "essai": <⑤>,
     "courbes": {"t", "ref", "defav", "comparateur", "opt", "seuil", "lab", "point", "t28", "titre"},   la figure
     "tableau": {...},                    les valeurs jour par jour (le site : vue « Tableau ») ; « horaire » : chaque
                                          ligne porte T_min_h et T_max_h (le détail horaire de la météo est chargé)
     "avertissements": [{"code", "message"}, ...]}

  <①> age = {"t", "age_max", "t_e", "fck", "pct", "hors_domaine" (t_e ≤ 3 j), "etat": "ok" | "warn",
             "beta" (None si t_e ≥ 28), "defav": {"t_e", "fck", "pct"} | None, "formules": [...]}
  <②> seuil = {"mode", "pct", "mpa", "cible", "t" (None : non atteignable), "date", "etat": "ok" | "nok", "t_e",
               "hors_domaine", "afficher", "controle": {"t_e", "t"} (forme fermée, site), "taux_age" (f_cible / f_ck(t) de ①,
               site ; None si f_ck(t) ≤ 0), "atteint_age", "formules",
               "releve" (site, avec un relevé : {"jour" (1er jour du relevé où f_ck ≥ cible), "date", "jour_annonce" (⌈t⌉),
               "fck_jour_annonce", "optimiste" (② annonce le seuil avant le relevé : NON CONSERVATIF)} | None)}
  <③> optim = {"mpa", "delai", "t_e", "classe", "fck28", "fck", "actuelle", "etat", "taux_retenue",
               "classes": [{"classe", "fck28", "fck", "taux" (None si f_ck ≤ 0), "ok"}], "afficher", "formules"}
  <④> comparateur = {"classes", "ciment", "s", "afficher", "courbes": [...]}
  <⑤> essai = {"mpa", "age", "T", "actif" (mpa > 0), "t_e", "fck28_est", "classe", "fck_classe", "ecart_classe",
               "ecart", "verdict": "coherent" | "dessus" | "sous", "etat", "hors_domaine", "afficher", "formules",
               "fck28_est_moyenne" (site : la mesure lue comme une moyenne, docs/beton/MATURITE_MOTEUR.md §7)}

  formules = [{"cle", "termes": [{"symbole", "valeur", "unite"}, ...], "resultat", "unite"}, ...] : chaque résultat se
  retrouve à partir de ses seuls termes (le test le prouve).
"""
import math

import numpy as np

from . import entree as EN
from . import meteo as MT
from . import releve as RL
from .ec2 import (CIMENT_DETAIL, CIMENTS, CLASSES, EA_EC2, HORIZON_PREVISION_J, R_GAZ, T_REF_K, age_equivalent,
                  age_equivalent_point, beta_cc, classe_estimee_depuis_mesure, classe_min_pour_delai, facteur_maturite,
                  fck_age_equivalent, fck_cible_pourcentage, fck_classe, temps_reel_pour_cible)
from .entree import nouvel_element, normaliser  # noqa: F401 — l'API

REFERENCE = "modules/age_beton.py v3.2"
T_COURBE = (1, 40, 500)                     # np.linspace(1, 40, 500) : l'axe de la figure du module
COULEURS = {"ref": "#2e6fb0", "defav": "#b91c1c", "seuil": "#b45309", "opt": "#7c3aed", "mes": "#1f8a70"}
COULEURS_CMP = ["#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad", "#16a085", "#d35400"]
AGES_TABLEAU = (1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 28)
SEUIL_DOMAINE = 3.0                         # t_e ≤ 3 j : « hors domaine EC2 » (§3.1.2(5))
ECART_COHERENT = 8.0                        # ⑤ : |écart| ≤ 8 % → cohérent


def _f(cle, termes, resultat, unite=""):
    return {"cle": cle, "termes": [{"symbole": s, "valeur": v, "unite": u} for s, v, u in termes], "resultat": resultat, "unite": unite}


def _fr(x: float, nd: int = 1) -> str:
    """Un nombre d'un message, à la française."""
    return f"{x:.{nd}f}".replace(".", ",")


def _num(x):
    """Un nombre JSON : float, et None pour NaN (un jour sans donnée)."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


# ------------------------------------------------------------------ la température
def _temperature(e: dict, avert: list) -> dict:
    """Le mode Manuel, le relevé ou la météo du lieu : T de calcul, profil, variante défavorable — ``_saisie_temperature``
    du module. La météo récupérée est un relevé comme un autre (la note garde ses températures, le calcul se fait hors
    ligne) ; ``meteo`` dit d'où il vient (``meteo.synthese`` : lieu trouvé, dates, récupération, relevé modifié depuis)."""
    t = e["temperature"]
    out = {"mode": t["mode"], "source": "manuelle", "lieu": t["lieu"], "fichier": t["fichier"], "releve": None, "defav": None,
           "profil": None, "jours": None, "meteo": None}
    if t["mode"] == "constante":
        out["T"] = t["T"]
        return out
    if t["mode"] == "meteo" and t["meteo"]:
        out["meteo"] = MT.synthese(t["meteo"], t["releve"], t["lieu"], t["debut"], t["fin"])
    rel = RL.lire(t["releve"])
    syn = RL.synthese(rel) if rel["temps"] else None
    if syn is None:
        # le module, sans météo récupérée (ou sans aucune température exploitable) : T = 20 °C, « manuelle »
        out["T"] = 20.0
        if t["mode"] == "meteo" and not rel["temps"] and t["meteo"]:
            # la météo a été récupérée, puis ses températures effacées à la main : le dire (« pas encore récupérée »
            # contredisait la date de récupération écrite au-dessus)
            avert.append({"code": "meteo_effacee", "message": "Températures de la météo effacées : T = 20 °C par défaut. « Récupérer à "
                          "nouveau la météo » remet celles du service (ou saisissez-les, une ligne par jour)."})
        elif t["mode"] == "meteo" and not rel["temps"]:
            avert.append({"code": "meteo_absente", "message": "Météo pas encore récupérée : T = 20 °C par défaut. Renseignez le lieu et "
                          "les dates, puis « Récupérer la météo » (ou saisissez une température constante, un relevé)."})
        else:
            avert.append({"code": "releve_vide" if not rel["temps"] else "releve_inexploitable",
                          "message": "Aucun relevé : T = 20 °C par défaut. Collez les températures journalières (ou lisez un fichier)."
                          if not rel["temps"] else "Aucune température exploitable sur la période : T = 20 °C par défaut."})
        return out
    out.update(T=syn["T_moy"], source="releve", profil=syn["profil"], defav=syn["defav"])
    out["releve"] = {k: syn[k] for k in ("T_rempl", "n_nan", "n_extra", "n_prev", "n_mes", "T_min", "T_max", "t_prol", "n_jours",
                                         "t_e_releve", "t_e_releve_defav")}
    out["jours"] = [{"jour": i + 1, "date": (rel["dates"] or [None] * len(rel["temps"]))[i], "T_brut": _num(tb), "T": tp,
                     "source": "prolonge" if rel["extra"][i] else ("prevision" if rel["prev"][i] else "mesure")}
                    for i, (tb, tp) in enumerate(zip(rel["temps"], syn["profil"]))]
    if syn["n_extra"]:
        avert.append({"code": "prolongation", "message": f"{syn['n_extra']} jour(s) prolongé(s) au-delà des prévisions (T ≈ "
                      f"{_fr(syn['t_prol'])} °C) : à recouper à l'approche du décoffrage."})
    if syn["n_nan"]:
        avert.append({"code": "jours_sans_donnee", "message": f"{syn['n_nan']} jour(s) sans donnée, remplacé(s) par la moyenne "
                      f"({_fr(syn['T_rempl'])} °C)."})
    return out


# ------------------------------------------------------------------ le calcul
def calculer(element: dict) -> dict:
    """Tout ce que l'écran du module affiche pour un coulage (``show()``), dans l'ordre de l'écran."""
    e = EN.normaliser(element)
    avert = []
    beton_label = e["classe"]
    fck28 = fck_classe(beton_label)
    ciment = EN.CIMENT_LIBELLES[e["ciment"]]
    s_ref = CIMENTS[ciment]
    tp = _temperature(e, avert)
    T, profil_temps, defav = tp["T"], tp["profil"], tp["defav"]
    Ea = EA_EC2
    if e["avance"]:
        Ea = e["Ea_kJ"] * 1000.0
    T_abs = float(T) + 273.15
    facteur = float(age_equivalent(1.0, T, Ea))
    t28_real = 28.0 / facteur if facteur > 0 else None
    # t3_reel : l'âge réel où t_e = 3 j à la température de la courbe (le site grise le domaine t_e ≤ 3 j)
    tp.update(Ea=Ea, Ea_kJ=Ea / 1000.0, Ea_R=Ea / R_GAZ, T_abs=T_abs, facteur=facteur, t28_reel=t28_real,
              t3_reel=SEUIL_DOMAINE / facteur if facteur > 0 else None)

    # ---- ① résistance à un âge donné
    age_max = len(profil_temps) if profil_temps else 28
    t_sel = e["age"]
    t_e_sel = age_equivalent_point(t_sel, T, profil_temps, Ea)
    fck_val = float(fck_age_equivalent(fck28, s_ref, t_e_sel))
    pct_val = fck_val / fck28 * 100.0 if fck28 else 0.0
    hors = t_e_sel <= SEUIL_DOMAINE
    beta = float(beta_cc(t_e_sel, s_ref)) if t_e_sel < 28.0 else None
    fx = []
    if profil_temps:
        n = min(int(round(t_sel)), len(profil_temps))
        fx.append(_f("t_e_releve", [("n", n, "j"), ("E_a/R", Ea / R_GAZ, "K")], t_e_sel, "j"))
    else:
        fx += [_f("T_abs", [("T", float(T), "°C")], T_abs, "K"),
               _f("facteur", [("E_a/R", Ea / R_GAZ, "K"), ("T_abs", T_abs, "K"), ("T_ref", T_REF_K, "K")], facteur_maturite(T, Ea), ""),
               _f("t_e", [("t", float(t_sel), "j"), ("facteur", facteur, "")], t_e_sel, "j")]
    if beta is not None:
        fx += [_f("beta", [("s", s_ref, ""), ("t_e", t_e_sel, "j")], beta, ""),
               _f("fcm", [("fck", float(fck28), "MPa")], fck28 + 8.0, "MPa"),
               _f("fck_t", [("beta", beta, ""), ("fcm", fck28 + 8.0, "MPa")], fck_val, "MPa")]
    else:
        fx.append(_f("fck_t_28", [("fck", float(fck28), "MPa")], fck_val, "MPa"))
    fx.append(_f("pct", [("fck_t", fck_val, "MPa"), ("fck", float(fck28), "MPa")], pct_val, "%"))
    age = {"t": t_sel, "age_max": int(max(age_max, 2)), "t_e": t_e_sel, "fck": fck_val, "pct": pct_val, "hors_domaine": hors,
           "etat": "warn" if hors else "ok", "beta": beta, "defav": None, "formules": fx}
    if hors:
        avert.append({"code": "hors_domaine_age", "message": f"t_e ≤ 3 j à {t_sel} j : hors du domaine de l'EC2 (§3.1.2(5)) — "
                      "la résistance au jeune âge se détermine par essais."})
    if defav is not None:
        t_e_def = age_equivalent_point(t_sel, defav["T_moy"], defav["profil"], Ea)
        fck_def = float(fck_age_equivalent(fck28, s_ref, t_e_def))
        age["defav"] = {"t_e": t_e_def, "fck": fck_def, "pct": fck_def / fck28 * 100.0, "T_moy": defav["T_moy"], "marge": defav["marge"],
                        "n_prev": defav["n_prev"], "n_extra": defav["n_extra"]}
        avert.append({"code": "defavorable", "message": f"Variante défavorable (jours prévus − {defav['marge']:.0f} °C) : la météo "
                      "prévue n'est pas acquise ; recalculer au jour J."})

    # ---- ② recherche d'un seuil de résistance (à T CONSTANTE : la moyenne en mode relevé, comme le module)
    sl = e["seuil"]
    if sl["mode"] == "pct":
        target_s = fck_cible_pourcentage(fck28, sl["pct"])
        fx_s = [_f("cible_pct", [("p", float(sl["pct"]), "%"), ("fck", float(fck28), "MPa")], target_s, "MPa")]
    else:
        target_s = sl["mpa"]
        fx_s = []
    t_seuil = temps_reel_pour_cible(fck28, s_ref, target_s, T, Ea)
    seuil = {"mode": sl["mode"], "pct": sl["pct"], "mpa": sl["mpa"], "cible": target_s, "cible_pct": target_s / fck28 * 100,
             "t": t_seuil, "date": EN.date_seuil(e["date_coulage"], t_seuil), "etat": "nok" if t_seuil is None else "ok",
             "afficher": sl["afficher"], "t_e": None, "hors_domaine": False, "controle": None, "formules": fx_s,
             # le site : la cible comparée à la résistance à l'âge choisi ① (f_cible / f_ck(t) ≤ 1 : le seuil est atteint
             # à cet âge) — une lecture des deux résultats du module, sans calcul nouveau ; None si f_ck(t) ≤ 0
             "taux_age": target_s / fck_val if fck_val > 0 else None, "atteint_age": fck_val >= target_s}
    if t_seuil is not None:
        t_e_s = float(age_equivalent(t_seuil, T, Ea))
        seuil["t_e"] = t_e_s
        seuil["hors_domaine"] = t_e_s <= SEUIL_DOMAINE
        f_s = float(fck_age_equivalent(fck28, s_ref, t_e_s))
        seuil["fck_t"] = f_s
        # le contrôle du site : la forme fermée de f_ck(t_e) = cible (t_e < 28 j), t = t_e / facteur
        if target_s < fck28:
            r = (target_s + 8.0) / (fck28 + 8.0)
            te_c = 28.0 / (1.0 - math.log(r) / s_ref) ** 2
            seuil["controle"] = {"t_e": te_c, "t": te_c / facteur}
            fx_s += [_f("t_e_cible", [("cible", target_s, "MPa"), ("fcm", fck28 + 8.0, "MPa"), ("s", s_ref, "")], te_c, "j"),
                     _f("t_cible", [("t_e", te_c, "j"), ("facteur", facteur, "")], te_c / facteur, "j")]
        fx_s.append(_f("t_seuil", [("cible", target_s, "MPa"), ("T", float(T), "°C")], t_seuil, "j"))
        fx_s.append(_f("fck_seuil", [("t_e", t_e_s, "j")], f_s, "MPa"))
        if seuil["hors_domaine"]:
            avert.append({"code": "seuil_hors_domaine", "message": "Seuil atteint avec t_e ≤ 3 j : hors du domaine de l'EC2 "
                          "(§3.1.2(5)) — à confirmer par essais. (Avertissement ajouté par le site ; l'écran Streamlit ne le donne pas.)"})
    else:
        avert.append({"code": "seuil_non_atteignable", "message": "Seuil non atteignable : la cible dépasse f_ck, ou n'est pas "
                      "atteinte en 90 jours à cette température."})

    # ---- ③ optimiser la classe (délai réduit), à T constante
    op = e["optim"]
    cible_opt, delai_opt = op["mpa"], op["delai"]
    res_opt = classe_min_pour_delai(cible_opt, s_ref, T, delai_opt, Ea)
    t_e_o = float(age_equivalent(delai_opt, T, Ea))
    classes_o = []
    for label in CLASSES:                               # la boucle de classe_min_pour_delai, chaque classe nommée
        fc = fck_classe(label)
        f_att = float(fck_age_equivalent(fc, s_ref, t_e_o))
        # taux = cible / f_ck(délai) (le site) ; None quand la formule rend une résistance nulle ou négative (t_e très
        # petit : β_cc·f_cm < 8) — la sortie reste du JSON strict, sans infini
        classes_o.append({"classe": label, "fck28": fc, "fck": f_att, "taux": cible_opt / f_att if f_att > 0 else None,
                          "ok": f_att + 1e-6 >= cible_opt})
    retenue = next(c for c in classes_o if c["classe"] == beton_label)
    optim = {"mpa": cible_opt, "delai": delai_opt, "t_e": t_e_o, "classe": None, "fck28": None, "fck": None, "actuelle": False,
             "etat": "nok", "afficher": op["afficher"], "classes": classes_o, "fck_retenue": retenue["fck"], "ok_retenue": retenue["ok"],
             "taux_retenue": retenue["taux"],
             "formules": [_f("t_e_delai", [("delai", delai_opt, "j"), ("facteur", facteur, "")], t_e_o, "j")]}
    if res_opt is not None:
        lbl_o, fck28_o, fck_att = res_opt
        sur = fck_classe(lbl_o) <= fck28
        optim.update(classe=lbl_o, fck28=fck28_o, fck=fck_att, actuelle=sur, etat="info" if sur else "ok")
    else:
        avert.append({"code": "aucune_classe", "message": f"Aucune classe de l'écran (jusqu'à {CLASSES[-1]}) n'atteint "
                      f"{_fr(cible_opt)} MPa en {_fr(delai_opt)} j."})
    if t_e_o < 28.0:
        b_o = float(beta_cc(t_e_o, s_ref))
        optim["beta"] = b_o
        optim["formules"].append(_f("beta", [("s", s_ref, ""), ("t_e", t_e_o, "j")], b_o, ""))

    # ---- ④ comparateur d'équivalence
    cp = e["comparateur"]
    ciment_cmp = EN.CIMENT_LIBELLES[cp["ciment"]]
    s_cmp = CIMENTS[ciment_cmp]
    comparateur = {"classes": list(cp["classes"]), "ciment": cp["ciment"], "ciment_libelle": ciment_cmp, "s": s_cmp, "afficher": cp["afficher"]}

    # ---- ⑤ analyse d'un essai mesuré
    es = e["essai"]
    essai = {"mpa": es["mpa"], "age": es["age"], "T": es["T"], "actif": es["mpa"] > 0, "afficher": es["afficher"], "formules": []}
    if es["mpa"] > 0:
        est = classe_estimee_depuis_mesure(es["mpa"], es["age"], s_ref, es["T"], Ea)
        ec = (est["fck28_est"] - fck28) / fck28 * 100.0
        verdict = "coherent" if abs(ec) <= ECART_COHERENT else ("dessus" if ec > ECART_COHERENT else "sous")
        essai.update(t_e=est["t_e"], fck28_est=est["fck28_est"], classe=est["classe"], fck_classe=est["fck_classe"],
                     ecart_classe=est["ecart_pct"], ecart=ec, verdict=verdict, etat="warn" if verdict == "sous" else "ok",
                     hors_domaine=est["t_e"] <= SEUIL_DOMAINE)
        f_e = facteur_maturite(es["T"], Ea)
        essai["formules"] = [_f("facteur", [("E_a/R", Ea / R_GAZ, "K"), ("T_abs", float(es["T"]) + 273.15, "K"), ("T_ref", T_REF_K, "K")], f_e, ""),
                             _f("t_e", [("t", es["age"], "j"), ("facteur", f_e, "")], est["t_e"], "j")]
        if est["t_e"] < 28.0:
            b_e = float(beta_cc(est["t_e"], s_ref))
            essai["beta"] = b_e
            essai["formules"] += [_f("beta", [("s", s_ref, ""), ("t_e", est["t_e"], "j")], b_e, ""),
                                  _f("fck28_est", [("f_mes", es["mpa"], "MPa"), ("beta", b_e, "")], est["fck28_est"], "MPa")]
        else:
            essai["formules"].append(_f("fck28_est_28", [("f_mes", es["mpa"], "MPa")], est["fck28_est"], "MPa"))
        essai["formules"] += [_f("ecart_classe", [("fck28_est", est["fck28_est"], "MPa"), ("fck", float(est["fck_classe"]), "MPa")], est["ecart_pct"], "%"),
                              _f("ecart", [("fck28_est", est["fck28_est"], "MPa"), ("fck", float(fck28), "MPa")], ec, "%")]
        # le site, pour information : la mesure lue comme une résistance MOYENNE (f_cm(t) = β_cc·f_cm, EC2 éq. 3.1), et non
        # comme f_ck(t) ainsi que le fait le module : f_ck,28 = f_mes/β_cc − 8 (f_mes − 8 si t_e ≥ 28 j) — 8/β_cc MPa de moins
        essai["fck28_est_moyenne"] = (es["mpa"] / essai["beta"] if est["t_e"] < 28.0 else es["mpa"]) - 8.0
        if essai["hors_domaine"]:
            avert.append({"code": "essai_hors_domaine", "message": "Essai : t_e ≤ 3 j, hors du domaine de l'EC2."})
    if tp["source"] == "releve":
        avert.append({"code": "releve_t_moyenne", "message": f"Comme l'écran Streamlit : la courbe, le seuil ② et la classe ③ sont "
                      f"calculés à la température MOYENNE du relevé ({_fr(T)} °C) ; seule la résistance à l'âge choisi ① suit le "
                      "relevé jour par jour."})

    out = {"reference": REFERENCE, "entree": e,
           "materiau": {"classe": beton_label, "fck28": fck28, "fcm": fck28 + 8.0, "ciment": e["ciment"], "ciment_libelle": ciment,
                        "ciment_detail": CIMENT_DETAIL[ciment], "s": s_ref},
           "temperature": {k: v for k, v in tp.items() if k not in ("profil", "jours")} | {"profil": profil_temps},
           "age": age, "seuil": seuil, "optim": optim, "comparateur": comparateur, "essai": essai, "avertissements": avert}
    if out["temperature"]["defav"]:
        out["temperature"]["defav"] = dict(out["temperature"]["defav"])
    out["courbes"] = _courbes(out, beton_label, ciment, fck28, s_ref, T, Ea, defav, t28_real, t_sel, fck_val)
    out["tableau"] = _tableau(out, fck28, s_ref, T, Ea, profil_temps, defav, tp.get("jours"))
    _extremes_horaires(out["tableau"], tp.get("meteo"))
    seuil["releve"] = _seuil_releve(seuil, out["tableau"]) if profil_temps else None
    if seuil["releve"] and seuil["releve"]["optimiste"]:
        sr = seuil["releve"]
        quand = (f"il n'est atteint qu'au jour {sr['jour']}" if sr["jour"] is not None
                 else f"il n'est pas atteint dans les {len(out['tableau']['lignes'])} jours du relevé")
        avert.append({"code": "seuil_optimiste", "message": f"Seuil ② NON CONSERVATIF avec ce relevé : à la température moyenne "
                      f"({_fr(T)} °C), l'écran l'annonce après {_fr(t_seuil)} j ; jour par jour (comme ①), f_ck au jour {sr['jour_annonce']} "
                      f"= {_fr(sr['fck_jour_annonce'], 2)} MPa < {_fr(seuil['cible'])} MPa : {quand}. Se fier au relevé jour par jour. "
                      "(Avertissement ajouté par le site.)"})
    return out


def _seuil_releve(seuil: dict, tableau: dict):
    """Le seuil ② LU dans le relevé jour par jour (le site, sans calcul nouveau : les valeurs du tableau, celles de ①) :
    le premier jour du relevé où f_ck ≥ cible, et f_ck au jour que ② annonce (⌈t_seuil⌉). ``optimiste`` : ② (à la
    température MOYENNE, comme l'écran) annonce le seuil plus tôt que le relevé — c'est le cas quand les jours froids
    viennent d'abord (docs/beton/MATURITE_MOTEUR.md, défaut non conservatif n° 2). None si le seuil n'est pas atteint."""
    lignes = tableau["lignes"]
    cible = seuil["cible"]
    if seuil["t"] is None:
        return None
    premier = next((li for li in lignes if li["fck"] >= cible), None)
    j_annonce = math.ceil(seuil["t"] - 1e-9)
    if j_annonce > len(lignes):                                  # le relevé est trop court pour en juger
        return {"jour": premier["age"] if premier else None, "date": premier.get("date") if premier else None,
                "jour_annonce": None, "fck_jour_annonce": None, "optimiste": False}
    li_a = lignes[j_annonce - 1]
    return {"jour": premier["age"] if premier else None, "date": premier.get("date") if premier else None,
            "jour_annonce": j_annonce, "fck_jour_annonce": li_a["fck"], "optimiste": li_a["fck"] < cible}


# ------------------------------------------------------------------ la figure (les données de build_figure)
def _courbes(out, beton_label, ciment, fck28, s_ref, T, Ea, defav, t28_real, t_sel, fck_val) -> dict:
    """Les arguments que ``show()`` passe à ``build_figure`` : abscisses, courbes, repères — recalculés à l'identique."""
    t_real = np.linspace(*T_COURBE)
    y_ref = fck_age_equivalent(fck28, s_ref, age_equivalent(t_real, T, Ea))
    c = {"t": [float(x) for x in t_real],
         "ref": {"label": f"{beton_label} — {ciment.split(' — ')[0]}", "y": [float(x) for x in y_ref]},
         "defav": None, "comparateur": None, "opt": None, "seuil": None, "lab": None,
         "point": [t_sel, fck_val], "t28": t28_real, "titre": f"fck(t) — {beton_label} — {T:.0f} °C"}
    cp = out["comparateur"]
    if cp["afficher"] and cp["classes"]:
        c["comparateur"] = []
        for cl in cp["classes"]:
            if cl == beton_label:
                continue                                 # pas de doublons avec la référence
            yc = fck_age_equivalent(fck_classe(cl), cp["s"], age_equivalent(t_real, T, Ea))
            c["comparateur"].append({"label": cl, "y": [float(x) for x in yc]})
    sl = out["seuil"]
    if sl["afficher"] and sl["t"] is not None:
        c["seuil"] = {"fck": sl["cible"], "t": sl["t"]}
    op = out["optim"]
    if op["afficher"] and op["classe"] is not None:
        y_o = fck_age_equivalent(op["fck28"], s_ref, age_equivalent(t_real, T, Ea))
        c["opt"] = {"label": f"{op['classe']} (optim.)", "y": [float(x) for x in y_o], "t": op["delai"], "fck": op["fck"]}
    es = out["essai"]
    if es["afficher"] and es["actif"]:
        c["lab"] = {"fck": es["mpa"], "age": es["age"]}
    if defav is not None:
        y_def = fck_age_equivalent(fck28, s_ref, age_equivalent(t_real, defav["T_moy"], Ea))
        c["defav"] = {"label": f"défavorable (prévisions − {defav['marge']:.0f} °C)", "y": [float(x) for x in y_def]}
    return c


# ------------------------------------------------------------------ le tableau des valeurs (le site)
def _tableau(out, fck28, s_ref, T, Ea, profil, defav, jours) -> dict:
    """Les valeurs que la figure trace, lues en clair (une ajoute du site, sans calcul nouveau : les mêmes fonctions
    que ①) — à T constante : les âges usuels et l'âge choisi ; avec un relevé : chaque jour du relevé, jour par jour
    (comme ①) et à la température moyenne (comme la courbe)."""
    cp = out["comparateur"]
    autres = [cl for cl in cp["classes"] if cl != out["materiau"]["classe"]] if cp["afficher"] else []
    lignes = []
    if profil:
        for j in jours:
            d = j["jour"]
            t_e = age_equivalent_point(d, T, profil, Ea)
            li = {"age": d, "date": j["date"], "T_brut": j["T_brut"], "T": j["T"], "source": j["source"], "t_e": t_e,
                  "fck": float(fck_age_equivalent(fck28, s_ref, t_e)),
                  "fck_moyenne": float(fck_age_equivalent(fck28, s_ref, age_equivalent(d, T, Ea)))}
            if defav is not None:
                t_d = age_equivalent_point(d, defav["T_moy"], defav["profil"], Ea)
                li.update(T_defav=defav["profil"][d - 1], t_e_defav=t_d, fck_defav=float(fck_age_equivalent(fck28, s_ref, t_d)))
            li["classes"] = {cl: float(fck_age_equivalent(fck_classe(cl), cp["s"], age_equivalent(d, T, Ea))) for cl in autres}
            lignes.append(li)
    else:
        ages = sorted(set(AGES_TABLEAU) | {out["age"]["t"]})
        for d in ages:
            t_e = age_equivalent_point(d, T, None, Ea)
            lignes.append({"age": d, "t_e": t_e, "fck": float(fck_age_equivalent(fck28, s_ref, t_e)),
                           "classes": {cl: float(fck_age_equivalent(fck_classe(cl), cp["s"], age_equivalent(d, T, Ea))) for cl in autres}})
    for li in lignes:
        li["pct"] = li["fck"] / fck28 * 100.0
    return {"releve": bool(profil), "defav": defav is not None, "classes": autres, "lignes": lignes}


def _extremes_horaires(tableau: dict, meteo) -> None:
    """Le détail horaire de la météo, quand il est chargé : T min h et T max h de chaque jour du relevé (``None`` pour un
    jour sans valeur horaire) — les colonnes du tableau « Données météo » de l'écran."""
    tableau["horaire"] = bool(meteo and meteo["horaire"] is not None and tableau["releve"])
    if tableau["horaire"]:
        for li in tableau["lignes"]:
            li["T_min_h"], li["T_max_h"] = meteo["horaire"].get(li.get("date"), (None, None))


# ------------------------------------------------------------------ le sélecteur de classe (③)
def candidats_classes(element: dict) -> list:
    """Les sept classes pour la cible de l'optimisation ③ (cible en MPa au délai), dans l'ordre de résistance : f_ck au
    délai, taux = cible / f_ck(délai), « ok » au critère du module (f + 1e-6 ≥ cible), la retenue (la classe du coulage)
    et l'optimale (la première qui tient : ``classe_min_pour_delai``)."""
    r = calculer(element)
    op = r["optim"]
    return [dict(c, retenue=c["classe"] == r["materiau"]["classe"], optimal=c["classe"] == op["classe"]) for c in op["classes"]]


__all__ = ["nouvel_element", "normaliser", "calculer", "candidats_classes", "CLASSES", "HORIZON_PREVISION_J"]
