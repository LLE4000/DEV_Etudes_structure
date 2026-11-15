# -*- coding: utf-8 -*-
"""Prix des grues — le MOTEUR (grues/prix) sur la BASE DE TEST (tests/donnees/prix_grues_base_test.json).

La base de test est la base de départ RENDUE ANONYME : prix fictifs (tous multipliés par un même facteur, gardé hors du
dépôt), loueurs renommés « Loueur A, B, C », textes libres neutres, chantiers numérotés. Les calculs sont LINÉAIRES dans
les prix : les tests T1 à T4 du cahier (§ 13.1) s'y retrouvent à l'euro près, multipliés par le même facteur (les attendus
de la base de test). La vraie base n'est jamais dans le dépôt.

Ce que la suite prouve :
  1. T1 à T4 (§ 13.1) — total arrondi, matériel, main-d'œuvre, chaque ligne, les prix manquants (T2 : télécommande ;
     T3 : carburant), l'alerte de variante HSC (T1).
  2. DURÉES (§ 6) — 01/09/2025 → 01/05/2026 = 8,0 mois ; la fraction du mois entamé ; le calendrier réel (170 jours sur la
     période de T1, 65 de juin à août 2026 : fériés belges déduits) et son effet sur le grutier.
  3. VALEURS (§ 5.3, § 7.2) — « 9 000 € », « 1.234,56 », compris, non disponible, sur demande, « ± 30 » non lu ; un prix
     inconnu n'est JAMAIS 0 € (la ligne est manquante, le calcul « incomplet »).
  4. CHOIX AUTO (§ 6.1) — la conduite (cabine comprise → cabine), les accessoires qui suivent leur location, les heures
     sup. au-delà de 8 h, le grutier absent avec opérateur, le châssis roulant (rails, pose, transport), le ballast du
     week-end, le déplacement du personnel au-delà d'un jour, le minimum facturé (MA), la contribution sans la fondation.
  5. FONDATIONS (§ 6, § 8.4, § 13.5.4) — le tableau (À vérifier) ; poutres = dalle × 1,0 ; pieux = dalle × 1,3 arrondi
     (le tableau de la base le confirme ligne par ligne) ; seuls les retours VALIDÉS comptent ; l'ancien retour validé
     remplace le tableau et T1 passe au prix attendu ; notre correction passe avant ; l'actualisation.
  6. ÉCUSSONS (§ 9) — année passée, année en cours, inconnue, période de validité, sur demande, corrigé.
  7. CONFIDENTIALITÉ — la base de test ne contient aucun nom de loueur ni de personne (contrôlé par empreintes : les noms
     ne sont écrits nulle part dans le dépôt).
  8. PURETÉ (bibliothèque standard seule) et MUTATIONS (copie du moteur, contrôles rapides).

    /usr/bin/python3.12 tests/test_prix_grues_moteur.py
    /usr/bin/python3.12 tests/test_prix_grues_moteur.py --rapide [--racine dossier]
"""
import copy
import datetime as dt
import hashlib
import json
import os
import re
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
sys.path.insert(0, RACINE_MOTEUR)

from grues.prix import base as B  # noqa: E402
from grues.prix import calcul as K  # noqa: E402
from grues.prix import ecussons as E  # noqa: E402
from grues.prix import fondations as F  # noqa: E402
from grues.prix import periode as P  # noqa: E402
from grues.prix import valeurs as V  # noqa: E402

BASE_TEST = os.path.join(ICI, "donnees", "prix_grues_base_test.json")
AUJ = dt.date(2026, 11, 14)
OK, KO = [], []
# les empreintes (sha256, 16 premiers caractères) des noms qui ne doivent JAMAIS être dans la base de test : les trois
# loueurs réels, les personnes, les chantiers et les entreprises cités dans la base de départ — les noms eux-mêmes ne
# sont écrits nulle part
EMPREINTES_INTERDITES = {
    "13f03b8b24b92a23", "1597d34d6709348e", "41f5e317af341ec9", "446acc36bce9a33c", "6056acc6ac81444c", "637d992f210633bc",
    "6ce25227a2accca3", "8f7958107bb9a7c6", "c558e9a266675984", "e1b4e88e1eae11ed", "e23c3d7ff76f6e62",
}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info != "" and not cond else ""))


def charger():
    return json.load(open(BASE_TEST, encoding="utf-8"))


def presque(a, b, tol=0.01):
    return a is not None and b is not None and abs(a - b) <= tol


def saisie_test(base, t):
    """La grue d'un test d'acceptation (``tests_acceptation``) : ses postes retenus « Oui », tous les autres « Non »."""
    g = next(g for g in base["grues"] if g["cas"] == t["cas"] and g["modele"] == t["modele"])
    s = K.nouvelle_grue("GT1", t["cas"], g["id"])
    if t.get("debut"):
        s["periode"] = {"mode": "dates", "debut": t["debut"], "fin": t["fin"]}
    elif t.get("duree_mois"):
        s["periode"] = {"mode": "mois", "mois": t["duree_mois"]}
    else:
        s["periode"] = {"mode": "jours", "jours": t["duree_jours"]}
    s["heures_jour"] = t["reglages"]["heures_grutier_par_jour"]
    s["hsc_m"] = t.get("hsc_m")
    if t.get("conduite"):
        s["conduite"] = t["conduite"]
    if t.get("fondation"):
        s["fondation"] = {"type": t["fondation"]["type"], "demolition": t["fondation"]["demolition"]}
    s["quantites"] = dict(t.get("quantites") or {})
    s["forces"] = {k: {"prix": v, "justification": "test d'acceptation"} for k, v in (t.get("prix_forces") or {}).items()}
    r = K.calculer(base, s, AUJ)
    s["choix"] = {ln["code"]: ("oui" if ln["code"] in t["postes_retenus"] else "non") for ln in r["lignes"]}
    return s


def ligne(r, code):
    return next((ln for ln in r["lignes"] if ln["code"] == code), None)


def grue_de(base, cas_lettre, modele):
    return next(g for g in base["grues"] if g["cas"].endswith(cas_lettre) and g["modele"] == modele)


# ------------------------------------------------------------------ 1. T1 à T4
def tests_acceptation(base):
    print("== 1. T1 à T4 (§ 13.1)")
    for t in base["tests_acceptation"]:
        s = saisie_test(base, t)
        r = K.calculer(base, s, AUJ)
        a = t["attendu"]
        chk(f"{t['id']} : total {r['total_arrondi']} € = {a['total_arrondi']} € (matériel {r['materiel']:.2f}, main-d'œuvre "
            f"{r['main_oeuvre']:.2f}), {len(r['manquants'])} prix manquant(s)",
            r["total_arrondi"] == a["total_arrondi"] and presque(r["materiel"], a["materiel"]) and presque(r["main_oeuvre"], a["main_oeuvre"])
            and len(r["manquants"]) == a["prix_manquants"], (r["total_arrondi"], r["materiel"], r["main_oeuvre"], r["manquants"]))
        faux = [(c, ligne(r, c) and ligne(r, c)["total"], v) for c, v in a["lignes"].items() if not presque((ligne(r, c) or {}).get("total"), v)]
        chk(f"{t['id']} : chaque ligne attendue ({len(a['lignes'])}) à 0,01 € près", not faux, faux)
        chk(f"{t['id']} : le calcul est {'complet' if not a['prix_manquants'] else 'INCOMPLET'}", r["complet"] == (a["prix_manquants"] == 0))
        if t["id"].startswith("T1"):
            chk("T1 : l'alerte de variante HSC (« HSC 25 m ≤ 39 m : la variante max. 39 m suffit ») et l'autre variante proposée",
                any("variante max. 39 m suffit" in x["texte"] and x.get("autre") for x in r["attention"]), r["attention"])
            chk("T1 : « 2 prix forcés » dans les points d'attention", any(x["texte"] == "2 prix forcés" for x in r["attention"]))
            chk("T1 : la fondation vient de l'ancien tableau (écusson À vérifier)", ligne(r, "FOND")["ecusson"] == E.A_VERIFIER
                and ligne(r, "FOND")["fondation"]["origine"] == "tableau")
            chk("T1 : la location est d'une année passée (2025) : « Pas à jour » ; le calcul aussi, avec sa raison",
                ligne(r, "LOC")["ecusson"] == E.PAS_A_JOUR and r["ecusson"] == E.PAS_A_JOUR and "pas à jour" in r["ecusson_raison"])
        if t["id"].startswith("T2"):
            chk("T2 : le prix manquant est la télécommande (TEL), comptée vide (jamais 0 €)",
                [m["code"] for m in r["manquants"]] == ["TEL"] and ligne(r, "TEL")["total"] is None and r["ecusson"] == E.MANQUANT)
        if t["id"].startswith("T3"):
            chk("T3 : le prix manquant est le carburant (CARB)", [m["code"] for m in r["manquants"]] == ["CARB"])


# ------------------------------------------------------------------ 2. durées
def durees(base):
    print("== 2. durées (§ 6)")
    d = P.lire_date
    chk("01/09/2025 → 01/05/2026 = 8,0 mois", P.mois_entre(d("2025-09-01"), d("2026-05-01")) == 8.0)
    chk("01/09/2025 → 16/10/2025 = 1 mois + 15/31", presque(P.mois_entre(d("2025-09-01"), d("2025-10-16")), 1 + 15 / 31, 1e-12))
    chk("31/01 + 1 mois = le dernier jour de février", P.plus_mois(d("2026-01-31"), 1) == d("2026-02-28"))
    chk("Pâques 2026 = 5 avril ; Pentecôte (lundi) 25 mai ; Ascension 14 mai",
        P.paques(2026) == d("2026-04-05") and d("2026-05-25") in P.feries_belges(2026) and d("2026-05-14") in P.feries_belges(2026))
    chk("calendrier réel de T1 (01/09/2025 → 01/05/2026) : 170 jours ouvrables, fériés déduits",
        P.jours_ouvrables(d("2025-09-01"), d("2026-05-01")) == 170, P.jours_ouvrables(d("2025-09-01"), d("2026-05-01")))
    chk("de juin à août 2026 : 65 jours ouvrables", P.jours_ouvrables(d("2026-06-01"), d("2026-09-01")) == 65)
    # (le cahier dit « environ 50 » : 15 jours ouvrables de congés, dont le 21 juillet déjà férié → 65 − 14 = 51)
    chk("de juin à août 2026 avec trois semaines de congés du bâtiment (20/07 → 07/08) : 51 jours (le 21 juillet, férié, "
        "n'est pas compté deux fois)",
        P.jours_ouvrables(d("2026-06-01"), d("2026-09-01"), [["2026-07-20", "2026-08-07"]]) == 51,
        P.jours_ouvrables(d("2026-06-01"), d("2026-09-01"), [["2026-07-20", "2026-08-07"]]))
    t1 = next(t for t in base["tests_acceptation"] if t["id"].startswith("T1"))
    b2 = copy.deepcopy(base)
    b2.setdefault("reglages", {})["jours_grutier"] = "calendrier réel"
    r = K.calculer(b2, saisie_test(b2, t1), AUJ)
    gr = ligne(r, "GRUTIER")
    taux = V.montant(B.prix_poste(B.poste(base, "GRUTIER"), t1["cas"])["valeur"])
    chk("réglage « calendrier réel » : le grutier de T1 compte 170 jours (8 h × 170 × taux), la location garde 160 jours",
        presque(gr["total"], 8 * 170 * taux) and ligne(r, "LOC")["facturee"] == 160, (gr["facturee"], gr["total"]))
    s = K.nouvelle_grue("GT1", t1["cas"], grue_de(base, "A", t1["modele"])["id"])
    s["periode"] = {"mode": "jours", "jours": 50}
    r = K.calculer(base, s, AUJ)
    chk("une durée en jours : 50 jours facturés = 2,5 mois", r["periode"]["jours"] == 50 and presque(r["periode"]["mois"], 2.5, 1e-12))


# ------------------------------------------------------------------ 3. valeurs
def valeurs():
    print("== 3. valeurs (§ 5.3, § 7.2)")
    cas = [("9 000 €", (V.NOMBRE, 9000.0)), ("1.234,56", (V.NOMBRE, 1234.56)), ("27,5", (V.NOMBRE, 27.5)), ("6T", (V.NOMBRE, 6.0)),
           ("compris", (V.COMPRIS, 0)), ("Inclus", (V.COMPRIS, 0)), ("nd", (V.NON_DISPONIBLE, "non disponible")),
           ("-", (V.NON_DISPONIBLE, "non disponible")), ("sur devis", (V.SUR_DEMANDE, "sur demande")),
           ("op aanvraag", (V.SUR_DEMANDE, "sur demande")), ("± 30", (V.TEXTE, "± 30")), ("", (V.INCONNU, None)), (0, (V.COMPRIS, 0))]
    faux = [(t, V.lire_mot(t), a) for t, a in cas if V.lire_mot(t) != a]
    chk("les mots d'une case de prix : nombres, compris, non disponible, sur demande, valeur non lue", not faux, faux)
    chk("montant : 0 = compris (0 €) ; vide, « non disponible », « sur demande » = PAS de montant",
        V.montant(0) == 0.0 and V.montant(None) is None and V.montant("non disponible") is None and V.montant("sur demande") is None)


# ------------------------------------------------------------------ 4. choix auto et règles des postes
def choix_auto(base):
    print("== 4. choix Auto et règles des postes (§ 6, § 6.1)")
    g4 = grue_de(base, "A", "Liebherr 125 EC-B 6 Litronic (> 39m HSC)")
    cas_t = g4["cas"]
    s = K.nouvelle_grue("GT1", cas_t, g4["id"])
    s["periode"] = {"mode": "dates", "debut": "2025-09-01", "fin": "2026-05-01"}
    r = K.calculer(base, s, AUJ)
    chk("grue tour, cabine comprise : conduite en cabine par défaut (CAB Oui, TEL Non, TELEC_MONT Non)",
        ligne(r, "CAB")["retenu"] and not ligne(r, "TEL")["retenu"] and not ligne(r, "TELEC_MONT")["retenu"])
    s2 = copy.deepcopy(s)
    s2["conduite"] = "telecommande"
    r2 = K.calculer(base, s2, AUJ)
    chk("conduite à la télécommande du loueur (payante) : TEL Oui, CAB Non, son montage et son démontage la suivent",
        ligne(r2, "TEL")["retenu"] and not ligne(r2, "CAB")["retenu"] and ligne(r2, "TELEC_MONT")["retenu"] and ligne(r2, "TELEC_DEM")["retenu"])
    s3 = copy.deepcopy(s)
    s3["choix"] = {"ANTICOL_LOC": "non"}
    r3 = K.calculer(base, s3, AUJ)
    chk("l'anti-collision retirée : son montage et son démontage (Auto) la suivent", not ligne(r3, "ANTICOL_MONT")["retenu"]
        and not ligne(r3, "ANTICOL_DEM")["retenu"] and ligne(r, "ANTICOL_MONT")["retenu"])
    s4 = copy.deepcopy(s)
    s4["heures_jour"] = 10
    r4 = K.calculer(base, s4, AUJ)
    hs = ligne(r4, "HS")
    t_hs = V.montant(B.prix_poste(B.poste(base, "HS"), cas_t)["valeur"])
    chk("10 h par jour : grutier 8 h × 160 j au taux normal, heures sup. 2 h × 160 j au taux des heures sup. (HS Oui)",
        hs["retenu"] and presque(hs["total"], 2 * 160 * t_hs) and ligne(r4, "GRUTIER")["quantite"] == 8 and not ligne(r, "HS")["retenu"])
    chk("sans montage ni démontage au tarif : deux prix manquants (jamais 0 €), le calcul est INCOMPLET",
        {m["code"] for m in r["manquants"]} == {"MONT", "DEMONT"} and not r["complet"] and r["ecusson"] == E.MANQUANT
        and ligne(r, "MONT")["total"] is None, r["manquants"])
    total_sans = r["total"]
    s5 = copy.deepcopy(s)
    s5["forces"] = {"MONT": {"prix": 1000, "justification": ""}}
    r5 = K.calculer(base, s5, AUJ)
    chk("un prix forcé sans justification : il compte, et c'est signalé", presque(r5["total"], total_sans + 1000)
        and any("sans justification" in a["texte"] for a in r5["attention"]))
    # châssis roulant
    s6 = copy.deepcopy(s)
    s6.update(chassis="roulant", voie_m=70, butees=2)
    r6 = K.calculer(base, s6, AUJ)
    rail = ligne(r6, "RAIL")
    p_rail = V.montant(B.prix_poste(B.poste(base, "RAIL"), cas_t)["valeur"])
    chk("châssis roulant, 70 m de voie : rails au mètre × jours, pose et démontage 3 jours (30 m/j, arrondi au-dessus), 2 "
        "transports aller et "
        "retour (36 m), train de roulement Non, point d'attention",
        rail["retenu"] and presque(rail["total"], p_rail * 70 * 160) and ligne(r6, "RAIL_POSE")["quantite"] == 3
        and ligne(r6, "RAIL_DEPOSE")["quantite"] == 3 and ligne(r6, "RAIL_TRANSPORT")["quantite"] == 2 and ligne(r6, "RAIL_EVAC")["quantite"] == 2
        and not ligne(r6, "RIJWERK")["retenu"] and any("Châssis roulant" in a["texte"] for a in r6["attention"]))
    chk("châssis roulant : la location prend le champ « châssis roulant » de la grue (vide ici : manquant) et le châssis "
        "« non disponible » pour ce modèle est manquant lui aussi",
        ligne(r6, "LOC")["manquant"] and ligne(r6, "CHROUL")["manquant"] and ligne(r6, "CHROUL")["manquant_raison"] == "non disponible pour ce modèle")
    # variante HSC dans l'autre sens
    g3 = grue_de(base, "A", "Liebherr 125 EC-B 6 Litronic (max. 39m HSC)")
    v = K.variante_hsc(base, g3, 45)
    chk("variante « max. 39 m » et HSC 45 m : prendre la variante > 39 m (proposée)", v and "prendre la variante > 39 m" in v["texte"] and v["autre"] == g4["id"], v)
    chk("variante « max. 39 m » et HSC 30 m : rien à dire", K.variante_hsc(base, g3, 30) is None)
    # électricité
    g16 = grue_de(base, "A", "Liebherr 53K")
    s7 = K.nouvelle_grue("GR1", g16["cas"], g16["id"])
    s7["periode"] = {"mode": "mois", "mois": 3}
    r7 = K.calculer(base, s7, AUJ)
    chk("montage rapide sans consommation connue : électricité au forfait par jour × jours facturés ; télécommande par "
        "défaut (pas la cabine) ; talkies Non", ligne(r7, "ELEC")["unite"] == "€/J" and presque(ligne(r7, "ELEC")["total"], g16["electricite_jour"] * 60)
        and ligne(r7, "TEL")["retenu"] and not ligne(r7, "CAB")["retenu"] and not ligne(r7, "TW_CH")["retenu"])
    s8 = copy.deepcopy(s)
    s8["electricite"] = {"prix_kwh": 0.5, "inf_32a": True}
    r8 = K.calculer(base, s8, AUJ)
    chk("grue tour, consommation connue : kWh/mois × prix du kWh du CHANTIER × mois ; raccordement < 32 A : Greenbox 80 kVA Oui "
        "et point d'attention", presque(ligne(r8, "ELEC")["total"], 1300 * 0.5 * 8) and ligne(r8, "GB80")["retenu"]
        and any("Greenbox nécessaire" in a["texte"] for a in r8["attention"]))
    # grue mobile avec opérateur
    g22 = grue_de(base, "A", "LTM 1100 (100T)")
    s9 = K.nouvelle_grue("GM1", g22["cas"], g22["id"])
    s9.update(periode={"mode": "mois", "mois": 6}, operateur=True)
    r9 = K.calculer(base, s9, AUJ)
    chk("grue mobile avec opérateur : location avec opérateur (vide ici : manquant), pas de ligne grutier ; fondation Non par défaut",
        ligne(r9, "LOC")["manquant"] and not ligne(r9, "GRUTIER")["retenu"] and not ligne(r9, "FOND")["retenu"])
    # logique MA
    g32 = grue_de(base, "B", "LTM - 1120")
    s10 = K.nouvelle_grue("GM2", g32["cas"], g32["id"])
    s10.update(periode={"mode": "jours", "jours": 1}, heures_jour=6)
    r10 = K.calculer(base, s10, AUJ)
    chk("MA, 1 jour à 6 h sur une grue de 120 t (minimum 8 h) : 8 h facturées ; pas de déplacement du personnel",
        ligne(r10, "LOC")["quantite"] == 8 and not ligne(r10, "DEPL_GRUTIER")["retenu"])
    petite = next(g for g in base["grues"] if g["cas"] == g32["cas"] and g.get("minimum_heures") == 5)
    s11 = K.nouvelle_grue("GM3", petite["cas"], petite["id"])
    s11.update(periode={"mode": "jours", "jours": 1}, heures_jour=4)
    s12 = copy.deepcopy(s11)
    s12["periode"] = {"mode": "jours", "jours": 3}
    chk("MA : petite grue (minimum 5 h), 1 jour à 4 h → 5 h ; 3 jours à 4 h → 8 h par jour",
        ligne(K.calculer(base, s11, AUJ), "LOC")["quantite"] == 5 and ligne(K.calculer(base, s12, AUJ), "LOC")["quantite"] == 8)
    s13 = copy.deepcopy(s10)
    s13.update(periode={"mode": "jours", "jours": 2}, heures_jour=10, quantites={"DEPL": 2, "SUP_SAM": 4, "RIGGER": 1},
               ma={"ballast_sam": True, "heures_personnel": {"RIGGER": 6}}, forces={"FOND": {"prix": 1000, "justification": "x"}})
    r13 = K.calculer(base, s13, AUJ)
    bal = g32["transport_ballast_ff"]
    coef = V.montant(B.prix_poste(B.poste(base, "BALLAST_SAM"), g32["cas"])["valeur"])
    p_rig = V.montant(B.prix_poste(B.poste(base, "RIGGER"), g32["cas"])["valeur"])
    assiette = sum(ln["total"] for ln in r13["lignes"] if ln["retenu"] and ln["code"] not in ("FOND", "CONTRIB_ENV"))
    ce = ligne(r13, "CONTRIB_ENV")
    chk("MA, 2 jours à 10 h : heures sup. 2 h × 2 j, samedi 4 h, ballast le samedi = coef × transport, rigger max(6 ; 8) h × 2 j, "
        "déplacement du personnel 2 trajets ; la contribution porte sur toutes ces lignes, PAS sur la fondation",
        ligne(r13, "SUP_HS")["retenu"] and ligne(r13, "SUP_HS")["quantite"] == 2 and ligne(r13, "SUP_SAM")["retenu"]
        and presque(ligne(r13, "BALLAST_SAM")["total"], coef * bal) and presque(ligne(r13, "RIGGER")["total"], p_rig * 8 * 2)
        and ligne(r13, "DEPL_GRUTIER")["retenu"] and presque(ce["total"], ce["prix"] / 100 * assiette)
        and presque(ce["facturee"], assiette), (ce, assiette))
    chk("MA : la main-d'œuvre regroupe le déplacement du personnel et les suppléments horaires (SUP_*)",
        presque(r13["main_oeuvre"], sum(ligne(r13, c)["total"] for c in ("DEPL_GRUTIER", "SUP_HS", "SUP_SAM"))))


# ------------------------------------------------------------------ 5. fondations
def fondations(base):
    print("== 5. fondations (§ 6, § 8.4, § 13.5.4)")
    m4 = "Liebherr 125 EC-B 6 Litronic (> 39m HSC)"
    k = F.case(base, m4, "Dalle sur sol", True, AUJ)
    ligne_t = next(f for f in base["fondations_tableau"] if f["modele"] == m4)
    chk("125 EC-B (> 39 m) · dalle sur sol · avec démolition : le prix du TABLEAU, « À vérifier », 1 retour en attente",
        k["origine"] == "tableau" and k["valeur"] == ligne_t["dalle_sur_sol"]["avec_demolition"] and k["ecusson"] == E.A_VERIFIER
        and k["a_verifier"] == 1, k)
    faux = []
    for f in base["fondations_tableau"]:
        for dem, cle in ((True, "avec_demolition"), (False, "sans_demolition")):
            pp, pi = F.tableau(base, f["modele"], "Poutres sur sol", dem), F.tableau(base, f["modele"], "Sur pieux", dem)
            egal = lambda x, y: (x is None and y is None) or presque(x, y)  # noqa: E731
            if not egal(pp, f["poutres_sur_sol_calcule"][cle]) or not egal(pi, f["sur_pieux_calcule"][cle]):
                faux.append((f["modele"], cle, pp, pi))
    chk(f"poutres = dalle × 1,0 et pieux = dalle × 1,3 arrondi au multiple (réglages) : les {len(base['fondations_tableau'])} lignes "
        "du tableau de la base retrouvées", not faux, faux)
    rex = [r for r in F.retours(base) if r["grue"] == m4 and r["type"] == "Dalle sur sol" and r["avec"] is not None]
    chk("un seul ancien retour pour cette case (H02), « À vérifier », année inconnue", len(rex) == 1 and rex[0]["id"] == "H02"
        and rex[0]["statut"] == F.A_VERIFIER, rex)
    b2 = copy.deepcopy(base)
    b2["rex_fondations_historique"][1]["statut"] = "validé"
    k2 = F.case(b2, m4, "Dalle sur sol", True, AUJ)
    prix = base["rex_fondations_historique"][1]["prix_reel"]
    chk("le retour validé remplace le tableau : sa valeur, « 1 chantier · année inconnue », À vérifier",
        k2["origine"] == "retours" and k2["valeur"] == prix and k2["source"] == "1 chantier · année inconnue" and k2["ecusson"] == E.A_VERIFIER, k2)
    t1 = next(t for t in base["tests_acceptation"] if t["id"].startswith("T1"))
    r1 = K.calculer(b2, saisie_test(b2, t1), AUJ)
    attendu = round((t1["attendu"]["materiel"] + t1["attendu"]["main_oeuvre"]) - ligne_t["dalle_sur_sol"]["avec_demolition"] + prix)
    chk(f"§ 13.5.4 : T1 avec ce retour validé passe à {attendu} € (le prix attendu de la base de départ, au facteur près)",
        r1["total_arrondi"] == attendu, (r1["total_arrondi"], attendu))
    k3 = F.case(b2, m4, "Dalle sur sol", False, AUJ)
    chk("le retour « avec démolition » ne compte pas pour la case « sans démolition »", k3["origine"] == "tableau")
    b3 = copy.deepcopy(b2)
    b3["fondations_corrections"] = {F.cle_case(m4, "Dalle sur sol", True): {"valeur": 20000, "motif": "essai", "date": "2026-11-14"}}
    k4 = F.case(b3, m4, "Dalle sur sol", True, AUJ)
    chk("notre correction passe avant les retours : « Corrigé »", k4["valeur"] == 20000 and k4["ecusson"] == E.CORRIGE)
    b4 = copy.deepcopy(base)
    b4["rex_fondations_historique"][1].update(statut="validé", annee=2024)
    b4["reglages"] = {"taux_actualisation": 3}
    k5 = F.case(b4, m4, "Dalle sur sol", True, AUJ)
    chk("un retour validé de 2024, actualisé à 3 %/an : × 1,03² ; « Pas à jour » (plus ancien que l'année précédente)",
        presque(k5["valeur"], prix * 1.03 ** 2) and k5["actualise"] and k5["ecusson"] == E.PAS_A_JOUR, k5)
    b4["rex_fondations_historique"][1]["annee"] = 2025
    chk("… de 2025 : « À jour » (l'année précédente)", F.case(b4, m4, "Dalle sur sol", True, AUJ)["ecusson"] == E.A_JOUR)
    exclus = [r for r in F.retours(base) if r["statut"] == F.ECARTE]
    sans = [r for r in F.retours(base) if r["statut"] == F.SANS_PRIX]
    chk(f"anciens retours : {len(F.retours(base))} lus, les aberrants écartés ({len(exclus)}), ceux sans prix ne comptent pas ({len(sans)})",
        len(F.retours(base)) == 32 and len(exclus) == 2 and len(sans) == 4)


# ------------------------------------------------------------------ 6. écussons
def ecussons():
    print("== 6. écussons (§ 9)")
    p = lambda **k: {"valeur": 100, "annee": None, "validite": None, "corrige": False, "a_verifier": False, **k}  # noqa: E731
    cas = [(p(annee=2025), E.PAS_A_JOUR), (p(annee=2026), E.A_JOUR), (p(annee=2027), E.A_JOUR), (p(), E.A_VERIFIER),
           (p(annee=2024, validite={"du": "2024-01-01", "au": "2026-12-31"}), E.A_JOUR),
           (p(annee=2026, validite={"du": "2026-01-01", "au": "2026-06-30"}), E.PAS_A_JOUR),
           (p(valeur="sur demande", annee=2026), E.A_VERIFIER), (p(annee=2026, a_verifier=True), E.A_VERIFIER),
           (p(valeur=None, annee=2026), E.MANQUANT), (p(annee=2020, corrige=True), E.CORRIGE), (p(valeur=0, annee=2025), E.PAS_A_JOUR)]
    faux = [(x, E.ecusson(x, AUJ), a) for x, a in cas if E.ecusson(x, AUJ) != a]
    chk("l'écusson d'un prix au 14/11/2026 : année, validité, sur demande, à vérifier, vide, corrigé", not faux, faux)
    chk("un ensemble : Pas à jour l'emporte, puis À vérifier ; un loueur sans aucun prix : Pas à jour",
        E.agreger([E.A_JOUR, E.A_VERIFIER]) == E.A_VERIFIER and E.agreger([E.A_VERIFIER, E.PAS_A_JOUR]) == E.PAS_A_JOUR
        and E.agreger([E.A_JOUR]) == E.A_JOUR and E.agreger([]) == E.PAS_A_JOUR)


# ------------------------------------------------------------------ 7. confidentialité
def confidentialite():
    print("== 7. confidentialité de la base de test")
    texte = open(BASE_TEST, encoding="utf-8").read()
    mots = {m.lower() for m in re.findall(r"[A-Za-zÀ-ÿ]{4,}", texte)}
    paires = {" ".join(x) for x in zip(re.findall(r"[A-Za-zÀ-ÿ]+", texte), re.findall(r"[A-Za-zÀ-ÿ]+", texte)[1:])}
    vus = [m for m in mots | {p.lower() for p in paires} if hashlib.sha256(m.encode()).hexdigest()[:16] in EMPREINTES_INTERDITES]
    chk("aucun nom de loueur réel dans la base de test (contrôle par empreintes)", not vus, len(vus))
    d = json.loads(texte)
    chk("les loueurs s'appellent Loueur A, B, C ; aucun contact, aucune condition, aucune correspondance de leurs listes",
        [l["nom"] for l in d["loueurs"]] == ["Loueur A", "Loueur B", "Loueur C"] and all(not l["contact"] for l in d["loueurs"])
        and not d["conditions_loueurs"] and not d["correspondances"])
    chk("aucun montant écrit dans un texte (« 60 €/h », « 50 € par trajet ») : seulement dans les champs de prix",
        not re.search(r"\d\s*€", texte))


# ------------------------------------------------------------------ rapides, complets, mutations
def rapides():
    base = charger()
    B.verifier(base)
    tests_acceptation(base)
    durees(base)
    valeurs()
    choix_auto(base)
    fondations(base)
    ecussons()


MUTATIONS = [
    ("periode.py", "    return n + (fin - a).days / (b - a).days", "    return float(n)", "la fraction du mois entamé oubliée"),
    ("periode.py", "            dt.date(annee, 11, 11), dt.date(annee, 12, 25)}", "            dt.date(annee, 12, 25)}", "le 11 novembre oublié"),
    ("valeurs.py", "    if n == COMPRIS:\n        return 0.0\n    return None", "    return 0.0", "un prix inconnu compté 0 €"),
    ("calcul.py", "            hs = max(0.0, c.h - c.hn)", "            hs = c.h", "les heures sup. sur toutes les heures"),
    ("calcul.py", "            assiette = sum(x[\"total\"] or 0.0 for x in lignes if x[\"retenu\"] and x is not ln and x[\"code\"] != \"FOND\"",
     "            assiette = sum(x[\"total\"] or 0.0 for x in lignes if x[\"retenu\"] and x is not ln", "la contribution sur la fondation"),
    ("calcul.py", "    if c.jours > 1:\n        m = c.hn", "    if False:\n        m = c.hn", "MA : 8 h par jour sur plusieurs jours oublié"),
    ("calcul.py", "        if ln[\"code\"] in SUIT:\n            _evaluer(ln, c, retenus)", "        if ln[\"code\"] in SUIT:\n            _evaluer(ln, c, {})",
     "le montage d'un accessoire qui ne suit plus sa location"),
    ("calcul.py", "        return (not c.operateur), (", "        return True, (", "le grutier compté avec l'opérateur"),
    ("calcul.py", "            n = math.ceil(c.voie / PAS_POSE_RAILS_M) if c.voie else 0", "            n = c.voie / PAS_POSE_RAILS_M if c.voie else 0",
     "la pose des rails sans arrondi au-dessus"),
    ("fondations.py", "        return _arrondi_multiple(dalle * float(r.get(\"coef_fondation_pieux\") or 1.3), float(r.get(\"arrondi_fondation_pieux\") or 0))",
     "        return dalle * float(r.get(\"coef_fondation_pieux\") or 1.3)", "les pieux sans arrondi"),
    ("fondations.py", "    valides = [r for r in tous if r[\"statut\"] == VALIDE]", "    valides = [r for r in tous if r[\"statut\"] in (VALIDE, A_VERIFIER)]",
     "un retour à vérifier qui compte"),
    ("fondations.py", "    if isinstance(corr, dict) and V.montant(corr.get(\"valeur\")) is not None:", "    if False:", "notre correction ignorée"),
    ("ecussons.py", "    return A_JOUR if a >= aujourdhui.year else PAS_A_JOUR", "    return A_JOUR", "une année passée « À jour »"),
]
TEMOIN = ("calcul.py", "CHOIX = (", "CHOIX = (")


def muter(fichier, ancien, nouveau):
    with tempfile.TemporaryDirectory() as tmp:
        for nom in os.listdir(RACINE):
            if nom in (".git", ".temoin_tests", "__pycache__", "node_modules", "grues"):
                continue
            os.symlink(os.path.join(RACINE, nom), os.path.join(tmp, nom))
        shutil.copytree(os.path.join(RACINE, "grues"), os.path.join(tmp, "grues"), ignore=shutil.ignore_patterns("__pycache__"))
        cible = os.path.join(tmp, "grues", "prix", fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rapide", "--racine", tmp], capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    t0 = time.time()
    rapides()
    print(f"   (contrôles rapides : {time.time() - t0:.1f} s)")
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    confidentialite()
    print("== 8. pureté")
    code = ("import sys, json, datetime\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('numpy', 'scipy', 'pandas', 'openpyxl', 'streamlit', 'matplotlib', 'reportlab', 'PIL'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from grues.prix import calcul as K\n"
            f"b = json.load(open({BASE_TEST!r}))\n"
            "g = [g for g in b['grues'] if g['id'] == 'G004'][0]\n"
            "s = K.nouvelle_grue('GT1', g['cas'], g['id']); s['periode'] = {'mode': 'mois', 'mois': 8}\n"
            "print(len(K.calculer(b, s, datetime.date(2026, 11, 14))['lignes']) > 20)\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le moteur se charge et calcule avec la bibliothèque standard seule (numpy, openpyxl… refusés)",
        p.returncode == 0 and p.stdout.strip() == "True", (p.stdout.strip(), p.stderr[-300:]))
    print("== 9. mutations")
    trouve, code = muter(*TEMOIN)
    chk("témoin : le moteur recopié sans rien changer passe les contrôles rapides", trouve and code == 0, code)
    for fichier, ancien, nouveau, nom in MUTATIONS:
        trouve, code = muter(fichier, ancien, nouveau)
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
