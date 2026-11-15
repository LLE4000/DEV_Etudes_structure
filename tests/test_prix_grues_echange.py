# -*- coding: utf-8 -*-
"""Prix des grues — l'ÉCHANGE ANNUEL avec les loueurs (cahier § 7.1, § 7.2, § 13.2) : le fichier Excel « Mise à jour des
prix » d'un loueur, sa relecture, l'import contrôlé et son annulation — sur la BASE DE TEST (prix fictifs, loueurs A/B/C).

Ce que la suite prouve :
  1. LE CLASSEUR (grues/prix/xlsx.py, bibliothèque standard) — relu par openpyxl, un lecteur INDÉPENDANT de l'écrivain :
     feuilles et ordre, « _META » très cachée, colonne A et ligne « #cle » cachées, copies cachées (+40 colonnes),
     protection sans mot de passe, cases à remplir déverrouillées, listes de validation, volets figés ; les octets sont
     reproductibles ; les textes difficiles (<&>", blancs, retours à la ligne) et les nombres reviennent tels quels.
  2. LE CONTENU (§ 7.1) — chaque grue non retirée du catalogue (G001…), ses valeurs telles que la base les garde, 0 écrit
     « compris », un prix essentiel vide compté « à compléter » ; la valeur ENVOYÉE est celle du loueur (jamais notre
     correction) ; nos remarques internes ne partent jamais.
  3. LA CONFIDENTIALITÉ (§ 13.2.3) — le fichier d'un loueur ne porte aucune grue ni aucun prix d'un autre loueur, aucune
     fondation, aucun forfait électricité, aucun prix du kWh, aucun poste interne : chaque nombre du fichier est un nombre
     de CE loueur (ou un compteur du fichier).
  4. L'ALLER-RETOUR À VIDE (§ 13.2.1) — chaque fichier relu sans rien changer : 0 changement, prix confirmés 186 / 109 / 0.
  5. LE FICHIER REMPLI (§ 13.2.2) — tests/donnees/prix_grues_loueur_a_rempli.xlsx (la demande du Loueur A modifiée et
     enregistrée par openpyxl, comme le ferait un loueur, sur les cases de la simulation du cahier) : 9 changements,
     1 case vidée, 1 valeur non lue (« ± 30 »), 181 confirmés, 1 nouvelle grue, 1 retirée, 1 autre prestation, 1 réponse,
     1 remarque, 0 refus ; « 9 000 € » lu 9000 ; les dates et « prix non modifiés valables ».
  6. L'IMPORT (§ 7.2) — appliqué avec les choix par défaut : l'année, la période et la source du fichier ; les confirmés
     datés ; la case vidée NON appliquée (sauf cochée) ; la valeur non lue appliquée si on la saisit ; notre correction
     gardée (« à revoir ») ; « prix non modifiés valables » = Non : les confirmés gardent leur année ; une nouvelle grue au
     numéro suivant ; un même fichier refusé la deuxième fois ; « Annuler le dernier import » remet la base à l'identique ;
     le calcul T1 prend le nouveau taux du grutier ; le dossier « Demande envoyée » jusqu'à l'import de la réponse.
  7. LES REFUS — un prix dans une case d'un autre fournisseur, un identifiant inconnu, un fichier qui n'est pas un
     classeur, un classeur sans « _META », un loueur inconnu.
  8. PURETÉ (bibliothèque standard seule) et MUTATIONS (copie du moteur, contrôles rapides).

    /usr/bin/python3.12 tests/test_prix_grues_echange.py [--rapide] [--racine dossier]
"""
import copy
import datetime as dt
import io
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

import openpyxl  # noqa: E402

from grues.prix import base as B  # noqa: E402
from grues.prix import calcul as K  # noqa: E402
from grues.prix import dossiers as D  # noqa: E402
from grues.prix import ecussons as E  # noqa: E402
from grues.prix import edition as ED  # noqa: E402
from grues.prix import fichier_loueur as FL  # noqa: E402
from grues.prix import import_loueur as IL  # noqa: E402
from grues.prix import lecture_loueur as LL  # noqa: E402
from grues.prix import xlsx as X  # noqa: E402

BASE_TEST = os.path.join(ICI, "donnees", "prix_grues_base_test.json")
REMPLI = os.path.join(ICI, "donnees", "prix_grues_loueur_a_rempli.xlsx")
AUJ = dt.date(2026, 11, 15)
RECU = dt.date(2026, 12, 12)
OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info != "" and not cond else ""))


def charger():
    return json.load(open(BASE_TEST, encoding="utf-8"))


def fichier(base, nom, annee=2027, par=None, retour=None):
    return FL.preparer(base, nom, annee, par, retour, AUJ)


def classeur(octets):
    return openpyxl.load_workbook(io.BytesIO(octets))


def lignes_a_id(ws):
    """{identifiant (colonne A) : {clé : valeur}} d'une feuille, lu par les lignes « #cle » (comme le site)."""
    out, cles = {}, None
    for r in range(1, ws.max_row + 1):
        a = ws.cell(r, 1).value
        if a == "#cle":
            cles = {c: ws.cell(r, c).value for c in range(2, ws.max_column + 1) if ws.cell(r, c).value}
            continue
        if a and cles:
            out[str(a) if a != "NOUVEAU" else f"NOUVEAU@{r}"] = {k: ws.cell(r, c).value for c, k in cles.items()}
    return out


# ------------------------------------------------------------------ 1. le classeur
def le_classeur():
    print("== 1. le classeur (relu par openpyxl)")
    base = charger()
    nom, o = fichier(base, "Loueur A", par="J. T.", retour="2026-12-15")
    chk("le nom du fichier est celui du cahier (DEMANDE_PRIX_{ANNÉE}_{LOUEUR}.xlsx)", nom == "DEMANDE_PRIX_2027_LOUEUR_A.xlsx", nom)
    chk("les octets sont reproductibles (même base, même jour → mêmes octets)", fichier(base, "Loueur A", par="J. T.", retour="2026-12-15")[1] == o)
    wb = classeur(o)
    noms = [ws.title for ws in wb.worksheets]
    chk("les feuilles du Loueur A, dans l'ordre du modèle", noms == ["À LIRE", "GRUES TOUR", "MONTAGE RAPIDE", "GRUES MOBILES", "ACCESSOIRES",
                                                                       "GRUTIERS", "AUTRES", "QUESTIONS", "_META"], noms)
    meta = {wb["_META"].cell(r, 1).value: wb["_META"].cell(r, 2).value for r in range(1, wb["_META"].max_row + 1)}
    chk("« _META » est très cachée et dit le format, le loueur, l'année, la feuille de chaque catalogue",
        wb["_META"].sheet_state == "veryHidden" and meta.get("format") == "demande_prix_grues" and meta.get("loueur_id") == "A"
        and meta.get("annee") == 2027 and meta.get("feuille:GRUES TOUR") == "Grue tour - Loueur A", meta)
    ws = wb["GRUES TOUR"]
    caches = {i for k, d in ws.column_dimensions.items() if d.hidden for i in range(d.min, d.max + 1)}
    ncol = max(c for c in range(2, 40) if ws.cell(6, c).value)
    chk("colonne A (identifiants) et ligne 6 (« #cle ») cachées ; les copies des valeurs (+40 colonnes) cachées",
        1 in caches and ws.row_dimensions[6].hidden and all(c + 40 in caches for c in range(2, ncol + 1)), (sorted(caches)[:5], ncol))
    chk("chaque feuille de tableau est protégée, sans mot de passe ; les volets figés en C7",
        all(wb[t].protection.sheet and not wb[t].protection.password for t in noms if t != "_META") and ws.freeze_panes == "C7")
    g1 = next(r for r in range(7, ws.max_row + 1) if ws.cell(r, 1).value == "G001")
    c_loc = next(c for c in range(2, ncol + 1) if ws.cell(6, c).value == "location_jour_chassis_standard")
    chk("une case de prix est déverrouillée, une case d'information verrouillée (le modèle)",
        ws.cell(g1, c_loc).protection.locked is False and ws.cell(g1, 2).protection.locked is True)
    dv = [d for d in ws.data_validations.dataValidation]
    chk("les cases de prix portent la liste de validation qui AVERTIT sans bloquer (« compris », « non disponible »…)",
        any(d.type == "custom" and "non disponible" in (d.formula1 or "") and d.errorStyle == "warning" for d in dv)
        and any(d.type == "list" and "Oui,Non" in (d.formula1 or "") for d in dv), len(dv))
    cf = ws.conditional_formatting
    chk("une case modifiée devient verte (format conditionnel : la case ≠ sa copie cachée)",
        any("<>" in (r.formula[0] if r.formula else "") for rng in cf for r in rng.rules))
    al = wb["À LIRE"]
    chk("« À LIRE » : « Demandé par » et « Retour souhaité avant le » remplis, le tableau d'état en formules",
        al["D5"].value == "J. T." and getattr(al["D6"].value, "date", lambda: None)() == dt.date(2026, 12, 15)
        and str(al["G41"].value or "").startswith("=SUMPRODUCT") and "COUNTBLANK" in str(al["H41"].value or ""),
        (al["D5"].value, al["D6"].value, al["G41"].value))
    # l'écrivain et le lecteur du site sur des textes et des nombres difficiles
    c = X.Classeur("essai")
    f = c.feuille("Essai")
    pieges = ["<b>&\"x\"</b>", "  blancs  ", "ligne 1\nligne 2", "é€ç ±", 1234567, 0.1 + 0.2, -3, True]
    for i, v in enumerate(pieges):
        f.ecrire(1, i + 1, v)
    f.ecrire(2, 1, dt.date(2027, 1, 1), {"format": "DD/MM/YYYY"})
    o2 = c.octets()
    ws2 = classeur(o2)["Essai"]
    lu = X.lire(o2)["feuilles"][0]["cellules"]
    chk("les textes difficiles et les nombres reviennent tels quels (openpyxl et le lecteur du site)",
        [ws2.cell(1, i + 1).value for i in range(len(pieges))] == pieges and [lu[(1, i + 1)] for i in range(len(pieges))] == pieges,
        [ws2.cell(1, i + 1).value for i in range(len(pieges))])
    chk("une date s'écrit en date Excel (relue en date par openpyxl, en numéro de série par le site)",
        ws2.cell(2, 1).value.date() == dt.date(2027, 1, 1) and X.de_serie(lu[(2, 1)]) == dt.date(2027, 1, 1))
    try:
        X.lire(b"pas un classeur")
        chk("un fichier qui n'est pas un classeur est refusé", False)
    except X.ClasseurIllisible:
        chk("un fichier qui n'est pas un classeur est refusé", True)


# ------------------------------------------------------------------ 2. le contenu, 3. la confidentialité
def nombres(o) -> set:
    """Tous les nombres VALEURS du fichier (hors formules), arrondis."""
    out = set()
    for f in X.lire(o)["feuilles"]:
        for v in f["cellules"].values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.add(round(float(v), 6))
    return out


def textes(o) -> str:
    return " ".join(str(v) for f in X.lire(o)["feuilles"] for v in f["cellules"].values() if isinstance(v, str))


def le_contenu():
    print("== 2. le contenu · 3. la confidentialité")
    base = charger()
    g = B.grue(base, "G001")
    g.setdefault("corrections", {})["location_jour_chassis_standard"] = {"valeur": 999.0, "motif": "devis", "date": "2026-11-01"}
    g["remarques_loueur"] = "REMARQUE INTERNE À NE PAS ENVOYER"
    _, o = fichier(base, "Loueur A")
    wb = classeur(o)
    gt = lignes_a_id(wb["GRUES TOUR"])
    attendus = [x["id"] for x in B.grues_du(base, "Grue tour - Loueur A")]
    chk("GRUES TOUR : une ligne par grue NON retirée du catalogue, dans l'ordre, puis 5 lignes « NOUVEAU »",
        [k for k in gt if not k.startswith("NOUVEAU")] == attendus and sum(k.startswith("NOUVEAU") for k in gt) == 5, list(gt)[:6])
    chk("la valeur ENVOYÉE est celle du loueur, pas notre correction (132, pas 999)",
        gt["G001"]["location_jour_chassis_standard"] == 132.0 and "999" not in str(gt["G001"]), gt["G001"]["location_jour_chassis_standard"])
    chk("nos remarques internes ne partent jamais", "REMARQUE INTERNE" not in textes(o))
    zero = next(((gid, k) for gid, row in gt.items() for k, v in row.items() if v == "compris"), None)
    chk("0 € s'écrit « compris »", zero is not None, zero)
    al = wb["À LIRE"]
    manque = str(al["H41"].value or "")
    chk("les prix essentiels vides de GRUES TOUR sont comptés « à compléter » (une case par prix manquant)",
        manque.count("COUNTBLANK") == sum(1 for gid in attendus for c in B.champs_prix(base, "T")
                                          if c.get("essentiel") and B.grue(base, gid).get(c["cle"]) is None), manque[:120])
    # 3. la confidentialité, pour chaque loueur
    for lo in B.loueurs(base):
        nom = lo["nom"]
        _, o = fichier(charger(), nom)
        b = charger()
        mes_cats = set(lo["catalogues"])
        autres_grues = {x["id"] for x in B.grues(b) if x["cas"] not in mes_cats}
        ids = {str(v) for f in X.lire(o)["feuilles"] for (l, c), v in f["cellules"].items() if c == 1}
        internes = {p["code"] for p in B.postes(b) if p.get("fourni_par") == "interne"}
        # les nombres de CE loueur : ses grues, ses postes (de ses catalogues ou fournis par lui) ; et les compteurs du fichier
        siens = set()
        for x in B.grues(b):
            if x["cas"] in mes_cats:
                siens |= {round(float(v), 6) for v in x.values() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for p in B.postes(b):
            for cas, v in (p.get("prix_par_cas") or {}).items():
                if isinstance(v, (int, float)) and (cas in mes_cats or p.get("fourni_par") == nom):
                    siens.add(round(float(v), 6))
        compteurs = set(range(0, 100)) | {2027.0}
        etrangers = {n for n in nombres(o) if n not in siens and n not in compteurs and not (40000 < n < 60000)}
        kwh = (B.prix_kwh(b) or {}).get("valeur")
        chk(f"{nom} : aucune grue d'un autre loueur, aucun poste interne (fondations, renforcement), aucun forfait électricité",
            not (ids & autres_grues) and not any(f"P:{c}" in ids for c in internes) and "electricite_jour" not in textes(o),
            (sorted(ids & autres_grues)[:3], [c for c in internes if f"P:{c}" in ids]))
        chk(f"{nom} : chaque nombre du fichier est un nombre de CE loueur (ou un compteur) — ni prix d'un autre loueur, ni kWh, "
            f"ni fondation", not etrangers and (kwh is None or round(kwh, 6) not in nombres(o) - siens), sorted(etrangers)[:6])


# ------------------------------------------------------------------ 4. l'aller-retour à vide, 5. le fichier rempli
def aller_retour():
    print("== 4. aller-retour à vide · 5. le fichier rempli")
    base = charger()
    confirmes = {}
    for lo in B.loueurs(base):
        _, o = fichier(base, lo["nom"])
        r = LL.lire(base, o, "x.xlsx")
        confirmes[lo["nom"]] = r["resume"]["prix_confirmes"]
        autres = {k: v for k, v in r["resume"].items() if k != "prix_confirmes"}
        chk(f"{lo['nom']} : relu sans rien changer → 0 changement, 0 refus, rien d'autre", not any(autres.values()), autres)
    chk("prix confirmés : Loueur A 186, Loueur B 109, Loueur C 0 (§ 13.2.1)", confirmes == {"Loueur A": 186, "Loueur B": 109, "Loueur C": 0}, confirmes)
    r = LL.lire(base, open(REMPLI, "rb").read(), "prix_grues_loueur_a_rempli.xlsx")
    attendu = {"changements": 9, "cases_videes": 1, "valeurs_non_lues": 1, "prix_confirmes": 181, "nouvelles_grues": 1, "grues_retirees": 1,
               "autres_prestations": 1, "reponses": 1, "remarques": 1, "refus": 0}
    chk("le fichier rempli : 9 changements, 1 case vidée, 1 non lue, 181 confirmés, 1 nouvelle grue, 1 retirée, 1 autre "
        "prestation, 1 réponse, 1 remarque, 0 refus (§ 13.2.2)", r["resume"] == attendu, r["resume"])
    ch = {(c["id"], c.get("cas"), c["cle"]): c for c in r["changements"]}
    g1 = ch.get(("G001", None, "location_jour_chassis_standard"))
    chk("G001 : 132 → 115, écart −12,9 % (surligné : une baisse)", g1 and g1["avant"] == 132.0 and g1["apres"] == 115 and g1["ecart_pct"] == -12.9
        and IL.a_surligner(g1, 15), g1)
    chk("« 9 000 € » est lu 9000 ; un prix nouveau (avant vide) n'a pas d'écart",
        (ch.get(("G004", None, "demontage_transport_ff")) or {}).get("apres") == 9000
        and "ecart_pct" not in ch[("G004", None, "demontage_transport_ff")])
    chk("« ± 30 » est une valeur NON LUE, montrée avec le texte du loueur", r["valeurs_non_lues"][0]["texte"] == "± 30"
        and r["valeurs_non_lues"][0]["id"] == "G002", r["valeurs_non_lues"])
    chk("la case vidée : G003 anti-intrusion", r["cases_videes"][0]["id"] == "G003" and r["cases_videes"][0]["cle"] == "anti_intrusion_jour")
    ng = r["nouvelles_grues"][0]
    chk("la nouvelle grue : Liebherr 172 EC-B 8, 210 €/J, 8 000 / 7 000 €, 2 disponibles",
        ng["valeurs"] == {"modele": "Liebherr 172 EC-B 8 Litronic", "location_jour_chassis_standard": 210, "transport_montage_ff": 8000,
                          "demontage_transport_ff": 7000, "nb_disponibles": 2} and ng["cas"] == "Grue tour - Loueur A", ng)
    chk("la grue retirée G013, la location de portique 150 €/S, la réponse à Q-A-01, la remarque sur P:VINCOTTE",
        r["grues_retirees"][0]["id"] == "G013" and r["autres_prestations"][0]["prix"] == 150 and r["autres_prestations"][0]["unite"] == "€/S"
        and r["reponses"][0]["id"] == "Q-A-01" and r["remarques"][0]["id"] == "P:VINCOTTE")
    chk("les informations du loueur : rempli par, date, période de validité, « prix non modifiés valables » = Oui",
        r["infos"] == {"demande_par": None, "retour_avant": None, "rempli_par": "Test", "mail": None, "telephone": None, "date": "2026-12-10",
                       "valable_du": "2027-01-01", "valable_au": "2027-12-31", "inchanges_valables": True}, r["infos"])


# ------------------------------------------------------------------ 6. l'import
def importer():
    print("== 6. l'import contrôlé, l'annulation")
    octets = open(REMPLI, "rb").read()
    base = charger()
    avant = copy.deepcopy(base)
    ED.corriger(base, ("poste", "GRUTIER", "Grue tour - Loueur A"), "60", "devis", "TT", AUJ)
    IL.marquer_demande(base, "Loueur A", 2027, "TT", "2026-12-15", "TT", AUJ)
    chk("le dossier passe en « Demande envoyée » quand le fichier est parti", D.dossier(base, "Loueur A", AUJ)["ecusson"] == E.DEMANDE)
    t1 = base["tests_acceptation"][0]
    r = LL.lire(base, octets, "rempli.xlsx")
    n = IL.appliquer(base, r, {}, {}, RECU, "TT", RECU)
    g = B.grue(base, "G001")
    chk("avec les choix par défaut, les 9 changements s'appliquent (pas la case vidée ni la valeur non lue)", n["appliques"] == 9, n)
    chk("un prix appliqué prend l'année, la période et la source du fichier",
        g["location_jour_chassis_standard"] == 115 and g["annee_par_champ"]["location_jour_chassis_standard"] == 2027
        and g["source_par_champ"]["location_jour_chassis_standard"] == "Fichier Loueur A 2027 reçu le 12/12/2026"
        and g["validite_par_champ"]["location_jour_chassis_standard"] == {"du": "2027-01-01", "au": "2027-12-31"})
    p_g3 = B.prix_grue(B.grue(base, "G003"), "anti_intrusion_jour")
    chk("la case vidée n'est PAS appliquée (G003 garde son prix)", p_g3["valeur"] is not None and p_g3["valeur"] == B.grue(avant, "G003")["anti_intrusion_jour"])
    chk("« prix non modifiés valables » = Oui : les 181 confirmés prennent l'année 2027", n["confirmes_dates"] == 181
        and B.prix_grue(B.grue(base, "G001"), "telecommande_loueur_prix")["annee"] == 2027)
    pg = B.prix_poste(B.poste(base, "GRUTIER"), "Grue tour - Loueur A")
    chk("notre correction n'est pas écrasée : elle reste (60), la valeur du loueur passe dessous (56,5), « à revoir »",
        pg["corrige"] and pg["valeur"] == 60.0 and pg["valeur_loueur"] == 56.5 and pg["correction"].get("a_revoir")
        and ("poste", "GRUTIER", "Grue tour - Loueur A") in n["a_revoir"], pg)
    neuve = B.grue(base, n["neuves"][0])
    ids = sorted(int(x["id"][1:]) for x in B.grues(avant) if re.match(r"^G\d+$", x["id"]))
    chk("la nouvelle grue prend le numéro suivant, son année et sa source", neuve["id"] == f"G{ids[-1] + 1:03d}" and neuve["annee_tarif"] == 2027
        and neuve["modele"] == "Liebherr 172 EC-B 8 Litronic", neuve["id"])
    chk("la grue « plus proposée » est retirée des listes (pas effacée)", B.retiree(B.grue(base, "G013")) and B.grue(base, "G013") is not None)
    chk("la réponse va à la question du loueur, la remarque à son dossier (à traiter)",
        next(q for q in B.loueur(base, "Loueur A")["questions"] if q["id"] == "Q-A-01")["reponse"] == "20 jours, inchangé"
        and base["dossiers"]["Loueur A"]["messages"][0]["texte"].startswith("Facturé en direct"))
    chk("le dossier quitte « Demande envoyée » ; le dernier fichier est noté", D.dossier(base, "Loueur A", RECU)["ecusson"] != E.DEMANDE
        and base["dossiers"]["Loueur A"]["dernier_fichier"]["changements"] == 9)
    chk("tout va dans l'historique (origine « fichier Loueur A 2027 »)",
        sum(1 for h in base["historique"] if h["origine"] == "fichier Loueur A 2027") >= 12)
    try:
        IL.appliquer(base, r, {}, {}, RECU, "TT", RECU)
        chk("le même fichier ne s'importe pas deux fois", False)
    except ValueError as e:
        chk("le même fichier ne s'importe pas deux fois", "déjà été importé" in str(e), str(e))
    # T1 sur la base importée : le taux du grutier du loueur passe dessous, la correction (60) reste celle du calcul
    s = K.nouvelle_grue("GT1", t1["cas"], next(x["id"] for x in B.grues(base) if x["cas"] == t1["cas"] and x["modele"] == t1["modele"]))
    s["periode"].update(mode="dates", debut=t1["debut"], fin=t1["fin"])
    s["hsc_m"] = t1["hsc_m"]
    rk = K.calculer(base, s, RECU)
    ln = next(x for x in rk["lignes"] if x["code"] == "GRUTIER")
    chk("le calcul prend le prix CORRIGÉ du grutier (60 €/h), pas celui du fichier", ln["prix"] == 60.0, ln["prix"])
    msg = IL.annuler_dernier(base, "TT", RECU)
    ref = _apres_correction()
    chk("« Annuler le dernier import » remet les grues, les postes, les questions et les catalogues à l'identique",
        all(base[k] == ref[k] for k in ("grues", "postes", "postes_par_logique", "loueurs")), msg)
    chk("… et le dossier revient en « Demande envoyée »", D.dossier(base, "Loueur A", RECU)["ecusson"] == E.DEMANDE)
    try:
        IL.annuler_dernier(base)
        chk("un deuxième « Annuler » n'a plus rien à annuler", False)
    except ValueError:
        chk("un deuxième « Annuler » n'a plus rien à annuler", True)
    # les choix de l'écran : la case vidée cochée, la non lue saisie, les prix non modifiés NON valables
    base = charger()
    r = LL.lire(base, octets, "rempli.xlsx")
    r["infos"]["inchanges_valables"] = False
    choix = {IL.cle_item("cases_videes", r["cases_videes"][0]): True, IL.cle_item("changements", r["changements"][0]): False}
    n = IL.appliquer(base, r, choix, {IL.cle_item("valeurs_non_lues", r["valeurs_non_lues"][0]): "32,5"}, RECU, "", RECU)
    chk("cochée, la case vidée s'applique ; saisie, la valeur non lue aussi ; décoché, un changement ne s'applique pas",
        B.grue(base, "G003")["anti_intrusion_jour"] is None and B.grue(base, "G002")["cabine_jour"] == 32.5
        and B.grue(base, "G001")["location_jour_chassis_standard"] == 132.0 and n["appliques"] == 10, n["appliques"])
    chk("« prix non modifiés valables » = Non : les confirmés gardent leur année", n["confirmes_dates"] == 0
        and B.prix_grue(B.grue(base, "G001"), "telecommande_loueur_prix")["annee"] == 2025)
    base = charger()
    r = LL.lire(base, octets, "rempli.xlsx")
    n = IL.appliquer(base, r, {IL.cle_item("autres_prestations", r["autres_prestations"][0]): True}, {}, RECU, "", RECU)
    p = B.poste(base, n["postes"][0])
    chk("une autre prestation devient un poste du catalogue (non retenu par défaut), sur les grues tour du loueur",
        p["poste"] == "Location portique" and p["prix_par_cas"] == {"Grue tour - Loueur A": 150} and p["auto_par_defaut"] == "Non"
        and p["code"] in base["postes_par_logique"]["T"], p)


def _apres_correction():
    """La base de test, corrigée et « demande envoyée » comme au début de ``importer`` : l'état qu'« Annuler » doit rendre."""
    b = charger()
    ED.corriger(b, ("poste", "GRUTIER", "Grue tour - Loueur A"), "60", "devis", "TT", AUJ)
    return b


# ------------------------------------------------------------------ 7. les refus
def refus():
    print("== 7. les refus")
    base = charger()
    _, o = fichier(base, "Loueur A")
    wb = classeur(o)
    ws = wb["ACCESSOIRES"]
    kr = next(r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "#cle")
    tiret = next(((r, c) for r in range(kr + 1, ws.max_row + 1) for c in range(4, 8) if ws.cell(r, c).value == "—"), None)
    ws.cell(*tiret).value = 12
    ws.cell(ws.max_row + 1, 1).value = "G999"
    tampon = io.BytesIO()
    wb.save(tampon)
    r = LL.lire(base, tampon.getvalue(), "x.xlsx")
    raisons = sorted(x["raison"] for x in r["refus"])
    chk("un prix tapé dans une case « — » (un autre fournisseur) et un identifiant inconnu sont REFUSÉS et listés",
        raisons == ["identifiant inconnu", "prix d'un autre fournisseur"], r["refus"])
    for nom, octets_, mot in (("un fichier qui n'est pas un classeur", b"PK\x03\x04 abime", "classeur"),
                              ("un classeur sans « _META »", _classeur_simple(), "_META")):
        try:
            LL.lire(base, octets_, "x.xlsx")
            chk(f"{nom} est refusé avec un message clair", False)
        except LL.FichierRefuse as e:
            chk(f"{nom} est refusé avec un message clair", mot in str(e) or "Mise à jour des prix" in str(e), str(e))
    b2 = charger()
    b2["loueurs"] = [l for l in b2["loueurs"] if l["nom"] != "Loueur A"]
    try:
        LL.lire(b2, o, "x.xlsx")
        chk("le fichier d'un loueur que la base ne connaît pas est refusé", False)
    except LL.FichierRefuse as e:
        chk("le fichier d'un loueur que la base ne connaît pas est refusé", "ne connaît pas" in str(e), str(e))
    try:
        FL.preparer(base, "Loueur Z", 2027)
        chk("préparer le fichier d'un loueur inconnu est refusé", False)
    except ValueError:
        chk("préparer le fichier d'un loueur inconnu est refusé", True)
    chk("l'année par défaut : l'année suivante à partir de novembre, sinon l'année en cours (§ 7.1)",
        FL.annee_par_defaut(dt.date(2026, 11, 1)) == 2027 and FL.annee_par_defaut(dt.date(2026, 10, 31)) == 2026)


def _classeur_simple():
    c = X.Classeur()
    c.feuille("Feuil1").ecrire(1, 1, "rien")
    return c.octets()


# ------------------------------------------------------------------ 8. pureté, mutations
def purete():
    print("== 8. pureté")
    code = ("import sys\nclass Refus:\n    def find_spec(self, nom, *a, **k):\n        if nom.split('.')[0] in ('openpyxl', 'numpy', 'scipy', 'pandas'):\n"
            "            raise ImportError(nom)\nsys.meta_path.insert(0, Refus())\nsys.path.insert(0, sys.argv[1])\n"
            "from grues.prix import fichier_loueur, lecture_loueur, import_loueur, xlsx\nprint('ok')\n")
    p = subprocess.run([sys.executable, "-I", "-c", code, RACINE_MOTEUR], capture_output=True, text=True, timeout=120)
    chk("l'échange se charge SANS openpyxl, numpy, scipy ni pandas (le navigateur n'a que la bibliothèque standard)",
        p.stdout.strip() == "ok", p.stderr[-300:])


def rapides():
    le_classeur()
    le_contenu()
    aller_retour()
    importer()
    refus()


MUTATIONS = [
    ("grues/prix/fichier_loueur.py", '''                elif k == "remarques_loueur":
                    v = None        # nos remarques internes ne sont jamais envoyées''',
     '''                elif k == "remarques_loueur":
                    v = g.get(k)''', "le fichier qui envoie nos remarques internes"),
    ("grues/prix/fichier_loueur.py", '''                    v = affiche(g.get(k), cat_)''', '''                    v = affiche(B.prix_grue(g, k)["valeur"], cat_)''',
     "le fichier qui envoie notre correction au lieu du prix du loueur"),
    ("grues/prix/lecture_loueur.py", '''            if p.get("fourni_par") != nom:
                if lire_texte(v) not in (None, "—"):''', '''            if False:
                if lire_texte(v) not in (None, "—"):''', "la lecture qui accepte le prix d'un autre fournisseur (prix:*)"),
    ("grues/prix/lecture_loueur.py", '''        if not ok:
            if lire_texte(v) not in (None, "—"):''', '''        if False:
            if lire_texte(v) not in (None, "—"):''', "la lecture qui accepte un prix hors de ses catalogues"),
    ("grues/prix/import_loueur.py", '''"cases_videes": False,''', '''"cases_videes": True,''', "les cases vidées appliquées par défaut"),
    ("grues/prix/import_loueur.py", '''    if cle in (g.get("corrections") or {}):
        g["corrections"][cle]["a_revoir"] = True''', '''    if cle in (g.get("corrections") or {}):
        g["corrections"].pop(cle)''', "l'import qui efface notre correction"),
    ("grues/prix/import_loueur.py", '''    if cas in (p.get("corrections") or {}):
        p["corrections"][cas]["a_revoir"] = True''', '''    if cas in (p.get("corrections") or {}):
        p["corrections"].pop(cas)''', "l'import qui efface notre correction (poste)"),
    ("grues/prix/import_loueur.py", '''    if (lecture.get("infos") or {}).get("inchanges_valables") is True:''',
     '''    if True:''', "les confirmés datés même si le loueur ne le dit pas"),
    ("grues/prix/import_loueur.py", '''    d_ = deja_importe(base, lecture)
    if d_:''', '''    d_ = None
    if d_:''', "le même fichier importé deux fois"),
    ("grues/prix/import_loueur.py", '''                      for g in gs if g.get("id") not in av["neuves"]]''', '''                      for g in gs]''',
     "« Annuler » qui garde les grues ajoutées"),
    ("grues/prix/xlsx.py", '''            cols += f'<col min="{c}" max="{c}" width="{w}" customWidth="1"{" hidden=" + chr(34) + "1" + chr(34) if c in self.colonnes_cachees else ""}/>\'''',
     '''            cols += f'<col min="{c}" max="{c}" width="{w}" customWidth="1"/>\'''', "les colonnes cachées qui ne le sont plus"),
    ("grues/prix/xlsx.py", '''            prot = '<protection locked="0"/>' if libre else ""''', '''            prot = ""''', "les cases à remplir verrouillées"),
    ("grues/prix/lecture_loueur.py", '''    if apres is None and avant is not None:
            res["cases_videes"]''', '''    if False:
            res["cases_videes"]''', "la case vidée prise pour un changement"),
]


def muter(fichier_, ancien, nouveau):
    copie = tempfile.mkdtemp(prefix="pg_echange_mut_")
    for d in ("grues",):
        shutil.copytree(os.path.join(RACINE_MOTEUR, d), os.path.join(copie, d), ignore=shutil.ignore_patterns("__pycache__"))
    p = os.path.join(copie, fichier_)
    s = open(p, encoding="utf-8").read()
    assert s.count(ancien) == 1, (fichier_, ancien[:60], s.count(ancien))
    open(p, "w", encoding="utf-8").write(s.replace(ancien, nouveau))
    return copie


def mutations():
    print("== 9. mutations (copie du moteur, contrôles rapides)")
    temoin = muter(MUTATIONS[0][0], MUTATIONS[0][1], MUTATIONS[0][1])       # une copie à l'identique
    t = subprocess.run([sys.executable, __file__, "--rapide", "--racine", temoin], capture_output=True, text=True, timeout=600)
    shutil.rmtree(temoin, ignore_errors=True)
    chk("témoin : le moteur recopié sans rien changer passe les contrôles rapides", t.returncode == 0, t.stdout[-600:])
    for fichier_, ancien, nouveau, nom in MUTATIONS:
        copie = muter(fichier_, ancien, nouveau)
        p = subprocess.run([sys.executable, __file__, "--rapide", "--racine", copie], capture_output=True, text=True, timeout=600)
        chk(f"mutation « {nom} » ({os.path.basename(fichier_)}) : un contrôle rougit", p.returncode != 0, p.stdout[-300:])
        shutil.rmtree(copie, ignore_errors=True)


if __name__ == "__main__":
    t0 = time.time()
    rapides()
    if not RAPIDE:
        purete()
        mutations()
    print(f"\nDurée : {time.time() - t0:.1f} s")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:200])
    sys.exit(1 if KO else 0)
