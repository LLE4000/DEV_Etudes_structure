# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Poutres bois » sur le site (bois/poutre/note.py, api_web/note_bois.py — l'écran y arrive par
api_web/bois.telecharger), 24/11/2026, vérifiée en Python ordinaire : l'état de la note du site est construit par les
MÊMES fonctions que l'écran (api_web/bois : nouvel_etat, modifier, gerer_elements), {"version": 2, "poutres": [...]}.

- TOUTE la note en un PDF valide : la page de garde du site (projet, partie, date, indice ; le sommaire : une ligne par
  poutre — son nom, sa section, sa qualité, sa portée, le type de bois, sa norme, son η max, son état) puis UNE planche
  par poutre (deux quand elle se poursuit), dans l'ordre de la liste, sur dix états : la poutre neuve, une solive en
  charges surfaciques, un chêne D30 maintenu aux appuis qui ne tient pas, un lamellé-collé sur mesure entretoisé (classe
  de service 2, effet système, appui non vérifié, contre-flèche, L/400 et L/300), un azobé D70, G seule (Q = 0), une
  poutre à la limite, une note à deux poutres, la note à trois poutres du relevé du paquet (construire.NOTES_RELEVE), une
  note d'avant (version 1) par le routeur ;
- PARITÉ avec le moteur de l'écran (``moteur.verifier``), poutre par poutre, par pdftotext : le titre de la planche et la
  ligne du sommaire, l'ÉTAT (chip) et l'η max, le taux de CHAQUE vérification et ce qu'elle compare (valeur, limite), le
  RÉSULTAT de chaque formule imprimée (les substitutions) et de chaque valeur lue (k_mod, k_h, k_crit, k_c,90, k_def), le
  panneau (section, portée, appui, maintien, service, limites, essence, classe, qualité visuelle, résistances, modules,
  masse volumique, charges, nature, durée, ψ₂, coefficients), le tableau des combinaisons (une ligne par combinaison) ;
  les HYPOTHÈSES propres à la poutre SEULEMENT — jamais la prose générale du moteur : un budget de caractères, et des
  phrases interdites —, ce qui n'est pas couvert, la référence ;
- l'IDENTIFICATION : la note du site (chantier et nom PIÉGÉS — accents, guillemets, « & », tiret long —, indice « C »,
  date) dans la garde et l'en-tête de chaque page, jamais « indice 0 » ni « indice A » ; le nom du fichier ;
- la PAGE DE SUITE : une planche trop longue se poursuit (« (suite) »), la garde compte ses pages, rien ne déborde ;
- les REFUS : une poutre hors domaine (son nom et le message du moteur), une note sans poutre ;
- COMME DANS LE NAVIGATEUR (la sonde commune, tests/note_cas_outils.py) : sans fontTools ni accélérateurs C, avec le Pillow
  factice et les seules polices du paquet, la note se fait, son texte est celui de CPython, elle n'ajoute pas numpy, et
  les fichiers du dépôt qu'elle charge en plus de l'écran sont ceux que ``note_bois.FICHIERS`` déclare — le paquet de la
  note les porte ;
- MUTATIONS (chacune doit faire rougir un contrôle) : une poutre oubliée, une donnée non transmise au moteur (l'appui),
  le taux d'un verdict pris sur l'η max, l'η max pris sur la première vérification, l'indice de la note perdu, un
  résultat de formule non imprimé, la qualité visuelle non imprimée, la prose générale des hypothèses imprimée.

    /usr/bin/python3.12 tests/test_note_bois.py
"""
import base64
import copy
import io
import json
import os
import re
import sys
import time
import zipfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

import note_cas_outils as O  # noqa: E402
from acier.garde_corps import membres as ME  # noqa: E402
from api_web import bois as B  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import note_bois as NB  # noqa: E402
from api_web import routeur as R  # noqa: E402
from bois.poutre import moteur as MO  # noqa: E402
from bois.poutre import note as NT  # noqa: E402

sys.path.insert(0, os.path.join(RACINE, "site", "web"))
try:
    import construire as K  # noqa: E402
finally:
    sys.path.pop(0)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def note(*poutres):
    """Une note du site construite COMME L'ÉCRAN la construit : chaque poutre est une suite de saisies (chemin, valeur)
    passées à ``bois.modifier`` (refusée = le test s'arrête) ; la première poutre est celle de la note neuve."""
    etat = B.nouvel_etat()
    for i, saisies in enumerate(poutres):
        if i:
            B.gerer_elements(etat, "ajouter")
        for ch, v in saisies:
            refus = B.modifier(etat, ch, v)
            assert refus is None, (ch, v, refus)
    etat["actif"] = 0
    etat.pop(B.CLE_AVIS, None)
    return etat


SOLIVE = ((["nom"], "Solive S1"), (["charges", "mode"], "surfacique"), (["charges", "g"], "1,8"), (["charges", "q"], "2,5"), (["charges", "entraxe"], "0,5"))
CHENE = ((["nom"], "Linteau chêne"), (["essence"], "chene"), (["L"], "3"), (["charges", "G"], "4"), (["charges", "Q"], "6"), (["charges", "nature"], "C"),
         (["charges", "poids_propre"], False), (["appui", "a"], "120"), (["maintien", "type"], "appuis"))
GL = ((["nom"], "Poutre GL"), (["classe"], "GL28c"), (["standard"], "mesure"), (["b"], "140"), (["h"], "450"), (["L"], "6"), (["charges", "G"], "3"),
      (["charges", "Q"], "5"), (["charges", "nature"], "B"), (["maintien", "type"], "entretoises"), (["maintien", "entraxe"], "2,5"), (["service"], "2"),
      (["systeme"], True), (["appui", "verifier"], False), (["contre_fleche"], "10"), (["lim_inst"], "400"), (["lim_fin"], "300"))
AZOBE = ((["nom"], "Azobé"), (["essence"], "azobe"), (["classe"], "D70"), (["standard"], "mesure"), (["b"], "100"), (["h"], "120"), (["L"], "2"),
         (["charges", "G"], "5"), (["charges", "Q"], "5"), (["maintien", "type"], "appuis"))
G_SEULE = ((["nom"], "Panne G seule"), (["charges", "Q"], "0"))
LIMITE = ((["nom"], "À la limite"), (["L"], "4,85"))
_ns = {}
exec(K.NOTES_RELEVE["bois"], _ns)  # noqa: S102 — la recette du relevé du paquet de la note (construire.NOTES_RELEVE)
TROIS = _ns["etat"]

ETATS = {
    "la poutre neuve (C24 épicéa 75 × 225, L = 4,00 m)": note(()),
    "une solive en charges surfaciques (g, q, entraxe)": note(SOLIVE),
    "un chêne D30 maintenu aux appuis, poids propre non compté, qui ne tient pas": note(CHENE),
    "un GL28c sur mesure, entretoisé, service 2, effet système, appui non vérifié, contre-flèche, L/400 et L/300": note(GL),
    "un azobé D70 sur mesure 100 × 120, maintenu aux appuis": note(AZOBE),
    "G seule (Q = 0 : une seule combinaison)": note(G_SEULE),
    "une poutre à la limite (95 à 100 %)": note(LIMITE),
    "deux poutres (la solive, l'azobé)": note(SOLIVE, AZOBE),
    "trois poutres : la note riche du relevé du paquet (solive, chêne, GL)": TROIS,
}
NOM = "Plancher « Est » & l'atelier \"Nord\""
CHANTIER = "Résidence d'Été & Cie — « Lot 2 »"
INFOS = {"nom": NOM, "chantier": CHANTIER, "indice": "C", "date": "03/04/2026"}
INTERDITS = ("indice 0", "indice A", "INDICE 0", "INDICE A")
# la prose GÉNÉRALE du moteur (ses hypothèses communes à toute poutre), que la planche ne répète pas : la planche la DIT
# par ses blocs (les deux combinaisons dans le tableau, V_d à l'appui, les parts de flèche, l'essence au panneau)
PROSE_INTERDITE = ("sur deux appuis simples", "deux combinaisons", "pris à l'axe de l'appui", "en combinaison caractéristique",
                   "est écrite, pas calculée")
# les formules du calcul que la planche N'IMPRIME PAS (le verdict porte le taux ; A et I ne servent qu'à W et aux flèches ;
# k_mod est dans le tableau ; les parts de la flèche de q et les limites sont dans le verdict ; ψ₂ est au panneau)
NON_IMPRIMEES = {"taux", "A", "I", "kmod", "w_q_m", "w_q_v", "lim_inst", "psi2", "lim_fin"}
# le budget du bloc Hypothèses (caractères sans blancs : les hypothèses propres, le non couvert, la référence) : le plus
# grand observé sur les états (768) + ≈ 15 % ; la mutation « prose générale imprimée » (≈ 700 caractères de plus) le dépasse
SEUIL_HYPOTHESES = 880


def norm(t):
    return O.norm(t)


def plat(t):
    return re.sub(r"\s+", "", norm(t))


def plat_sym(t):
    """Sans blanc ni « _ », « $ », « { », « } » : un symbole indicé (k_c,90, $σ_{m,d}$) sort de pdftotext en « kc,90 »."""
    return plat(t.replace("_", "").replace("$", "").replace("{", "").replace("}", ""))


def pages_brutes(octets):
    return O.pages(octets, brut=True)


def pages_planche(tx, titre, depuis):
    """(indice de la planche d'une poutre — son titre « nom — section », et pas « (suite) » —, nombre de pages qu'elle
    occupe), cherchés à partir de la page ``depuis``."""
    for i in range(depuis, len(tx)):
        n = norm(tx[i])
        if titre in n and f"{titre} (suite)" not in n:
            k = 1
            while i + k < len(tx) and f"{titre} (suite)" in norm(tx[i + k]):
                k += 1
            return i, k
    return None, 0


def blocs_imprimes(r):
    """Les blocs du détail du calcul dont la planche imprime les formules : les charges, la combinaison qui gouverne
    l'ELU, la section, le déversement, la flexion, le cisaillement et l'appui dans LEUR combinaison, les flèches."""
    par = {v["cle"]: v for v in r["verifications"]}
    out = ["charges", f"combinaison_{r['combinaison_elu']}", "section", "deversement", f"flexion_{par['flexion']['combinaison']}",
           f"cisaillement_{par['cisaillement']['combinaison']}"]
    if "appui" in par:
        out += ["appui", f"appui_{par['appui']['combinaison']}"]
    return out + ["fleche_inst", "fleche_fin"]


def attendus_panneau(r):
    """Ce que le panneau de la planche doit porter, d'après le MOTEUR (pas d'après le générateur)."""
    p, geo, mat, ch, co, ap, fl = r["poutre"], r["geometrie"], r["materiau"], r["charges"], r["coefficients"], r["appui"], r["fleches"]
    out = [geo["designation"], NT._fixe(geo["L"], 2), NT._fixe(ap["a"], 0), str(co["service"]), f"L/{fl['n_inst']} · L/{fl['n_fin']}",
           {"continu": "continu", "appuis": "aux appuis", "entretoises": "entretoises"}[p["maintien"]["type"]],
           mat["classe"], NT._fixe(mat["fm_k"], 1), NT._fixe(mat["fv_k"], 1), NT._fixe(mat["fc90_k"], 1), NT._fixe(mat["E0_mean"], 0),
           NT._fixe(mat["E0_05"], 0), NT._fixe(mat["G_mean"], 0), NT._fixe(mat["rho_mean"], 0), co["libelle_nature"], co["libelle_duree"],
           NT._fixe(co["psi2"], 2), NT._fixe(co["kmod_permanent"], 2), NT._fixe(co["kmod_variable"], 2), NT._fixe(co["kdef"], 2),
           NT._fixe(co["gamma_M"], 2), f"{NT._fixe(co['gamma_G'], 2)} · {NT._fixe(co['gamma_Q'], 2)}", NT._fmt(ch["poids_propre"], "kN/m")]
    out.append(mat["libelle_essence"] if mat["essence_dite"] else {"massif": "bois résineux", "feuillu": "bois feuillu", "lamelle": "lamellé-collé"}[mat["type"]])
    if mat["qualites"]:
        out.append(mat["qualites"][0])
    if ch["mode"] == "surfacique":
        out += [NT._fmt(ch["g_surf"], "kN/m²"), NT._fmt(ch["q_surf"], "kN/m²"), NT._fixe(ch["entraxe"], 2)]
    else:
        out += [NT._fmt(ch["G"], "kN/m"), NT._fmt(ch["Q"], "kN/m")]
    if not ch["poids_propre_compte"]:
        out.append("(non compté)")
    if co["systeme"]:
        out.append(NT._fixe(co["ksys"], 2))
    return out


def bloc_hypotheses(pl):
    """Le bloc Hypothèses de la planche, sans blancs (de son titre à la fin de la planche — c'est le dernier bloc)."""
    t = plat(pl)
    i = t.rfind("Hypothèses")
    return t[i:] if i >= 0 else t


def ecarts_parite(e, octets):
    """Les écarts entre la note et le moteur de l'écran, poutre par poutre (voir l'en-tête)."""
    tx = pages_brutes(octets)
    garde = norm(tx[0])
    out, debuts, attendu_pages, depuis = [], [], 1, 1
    for p in e["poutres"]:
        r = MO.verifier(p)
        sec = NT.section(p)
        nom = p.get("nom") or "Poutre"
        titre = f"{sec['poutre']} — {sec['section']}"
        i, k = pages_planche(tx, titre, depuis)
        if i is None:
            out.append(f"« {nom} » : aucune planche à son titre « {titre} »")
            continue
        depuis = i + k
        debuts.append(i)
        attendu_pages += k
        pl = norm(" ".join(tx[i:i + k]))
        s = plat_sym(pl)
        lignes = [plat_sym(l) for l in "\n".join(tx[i:i + k]).splitlines()]      # chaque formule commence sa ligne : « W = … »
        # la garde : la ligne du sommaire (le titre, le type et la norme avec l'η max, l'état)
        eta = f"η max {NT._pct(r['taux_max'])}"
        etat_txt = {ME.OK: "VÉRIFIÉ", ME.ATTENTION: "À LA LIMITE", ME.NOK: "NON VÉRIFIÉ"}[ME.statut(r["taux_max"]) if r["statut"] == MO.OK else ME.NOK]
        sous = norm(f"{mat_txt(r)} / {r['materiau']['norme']} · {eta}")
        if not (titre in garde and sous in garde and etat_txt in garde):
            out.append(f"« {nom} » garde : ligne du sommaire « {titre} · {sous} · {etat_txt} » incomplète")
        # l'en-tête de la planche : l'η max (le η du verdict de l'écran), le chip de l'état
        if not (O.present(eta, pl) and etat_txt in pl):
            out.append(f"« {nom} » : « {eta} » ou « {etat_txt} » absent de l'en-tête")
        # chaque vérification : son taux, ce qu'elle compare (valeur et limite)
        for v in r["verifications"]:
            u = v["unite"]
            for x in (NT._pct(v["taux"]), NT._qte(v["valeur"], u), NT._qte(v["limite"], u)):
                if not O.present(x, pl):
                    out.append(f"« {nom} » : {v['cle']} — « {x} » absent")
        # chaque formule imprimée : son résultat (la substitution), ou la valeur lue
        par_bloc = {b["cle"]: b for b in r["calcul"]}
        for cle in blocs_imprimes(r):
            for f in par_bloc[cle]["formules"]:
                if f["cle"] in NON_IMPRIMEES or (f["cle"] == "q" and r["charges"]["mode"] != "surfacique") \
                        or (f["cle"] == "w_net_fin" and r["fleches"]["w_c"] == 0):
                    continue
                x = NT._fmt(f["resultat"], f["unite"]) if f["gabarit"] is not None else NT._qte(f["resultat"], f["unite"])
                if not O.present(x, pl):
                    out.append(f"« {nom} » : formule {cle}/{f['cle']} — résultat « {x} » absent")
                if not any(l.startswith(plat_sym(f["symbole"]) + "=") for l in lignes):   # la formule elle-même, en tête de sa ligne
                    out.append(f"« {nom} » : formule {cle}/{f['cle']} — la ligne « {f['symbole']} = … » manque")
        # le panneau : ses valeurs d'après le moteur
        for x in attendus_panneau(r):
            if norm(x) not in pl:
                out.append(f"« {nom} » panneau : « {x} » absent")
        # le tableau des combinaisons : une ligne par combinaison du moteur (une seule quand q = 0)
        for c_ in r["combinaisons"]:
            for x in (c_["nom"], NT._fmt(c_["p_d"], "kN/m"), NT._fmt(c_["M_d"], "kN·m"), NT._fmt(c_["V_d"], "kN")):
                if not O.present(x, pl) and x not in pl:
                    out.append(f"« {nom} » combinaisons : « {x} » absent")
        if (len(r["combinaisons"]) == 1) != (r["charges"]["q"] == 0):
            out.append(f"« {nom} » : {len(r['combinaisons'])} combinaison(s) pour q = {r['charges']['q']}")
        # les hypothèses : celles de la poutre, jamais la prose générale ; le non couvert ; la référence ; le budget
        for h in r["hypotheses"]:
            tete = h.split(" : ")[0][:28]
            if h.startswith(NT.HYPOTHESES_GENERALES):
                continue
            if plat_sym(tete) not in s:
                out.append(f"« {nom} » hypothèses : « {h[:50]} » absente")
        for x in PROSE_INTERDITE:
            if plat(x) in plat(pl):
                out.append(f"« {nom} » hypothèses : la prose générale « {x} » est imprimée")
        if "Non couvert :" not in pl or plat(r["non_couvert"][0]) not in plat(pl) or plat_sym(r["reference"][:30]) not in s:
            out.append(f"« {nom} » : le non couvert ou la référence manque")
        n_hyp = len(bloc_hypotheses(pl))
        if n_hyp > SEUIL_HYPOTHESES:
            out.append(f"« {nom} » : hypothèses trop longues ({n_hyp} caractères, budget {SEUIL_HYPOTHESES})")
    if len(tx) != attendu_pages:
        out.append(f"{len(tx)} pages pour {attendu_pages} attendues (garde + planches)")
    annonces = [int(x) for x in re.findall(r"page (\d+)", tx[0])]
    if annonces != [i + 1 for i in debuts]:
        out.append(f"sommaire : pages annoncées {annonces}, pages réelles {[i + 1 for i in debuts]}")
    return out


def mat_txt(r):
    return r["materiau"]["libelle_type"]


def ecarts_identification(octets, infos=INFOS):
    tx = pages_brutes(octets)
    out = []
    tout = norm(" ".join(tx))
    out += [f"« {x} » imprimé" for x in INTERDITS if x in tout]
    g = plat(tx[0])
    for lab, val in (("PROJET", infos["chantier"]), ("PARTIE", infos["nom"]), ("DATE", infos["date"]), ("INDICE", infos["indice"])):
        if plat(lab + val) not in g:
            out.append(f"garde : {lab} {val} absent")
    for i, t in enumerate(tx[1:]):
        if f"{infos['date']} · indice {infos['indice']}" not in norm(t):
            out.append(f"page {i + 2} : « {infos['date']} · indice {infos['indice']} » absent de l'en-tête")
    return out


# ---------------------------------------------------------------- 1. la note de chaque état, la parité
NOTES, DUREES, MESURES = {}, {}, {}
for nom_e, e in ETATS.items():
    t0 = time.time()
    f = NB.fichier(copy.deepcopy(e), INFOS)
    DUREES[nom_e] = time.time() - t0
    octets = base64.b64decode(f["base64"])
    NOTES[nom_e] = octets
    n = len(e["poutres"])
    tx = pages_brutes(octets)
    chk(f"{nom_e} : un PDF ({len(tx)} pages, {len(octets) // 1000} ko, {DUREES[nom_e]:.2f} s), nommé d'après la note, qui passe le contrôle du routeur",
        octets[:5] == b"%PDF-" and len(tx) >= 1 + n and f["nom"] == f"Note de calcul - {C.nom_fichier(NOM, '')}.pdf" and f["type"] == "application/pdf"
        and C.refus_fichier(f) is None, (octets[:8], len(tx), f["nom"]))
    ec = ecarts_parite(e, octets)
    chk(f"{nom_e} : parité avec le moteur — sommaire, η max et état, taux et comparaisons, résultats des formules, panneau, combinaisons, hypothèses "
        "propres (sans prose générale), non couvert, référence, pages", not ec, ec[:5])
    ec = ecarts_identification(octets)
    chk(f"{nom_e} : l'identification de la note du site dans la garde et l'en-tête de chaque page", not ec, ec[:4])
    for p in e["poutres"]:
        sec = NT.section(p)
        i, k = pages_planche(tx, f"{sec['poutre']} — {sec['section']}", 1)
        if i is not None:
            MESURES[(nom_e, i)] = (len(bloc_hypotheses(norm(" ".join(tx[i:i + k])))), k)
statuts = {ME.statut(MO.verifier(p)["taux_max"]) for e in ETATS.values() for p in e["poutres"]}
chk("les états couvrent les trois états (vérifié, à la limite, non vérifié), les trois types de bois, les deux modes de charge, une note à deux et "
    "une à trois poutres",
    statuts == {ME.OK, ME.ATTENTION, ME.NOK} and {MO.verifier(p)["materiau"]["type"] for e in ETATS.values() for p in e["poutres"]} == {"massif", "feuillu", "lamelle"}
    and {p["charges"]["mode"] for e in ETATS.values() for p in e["poutres"]} == {"lineaire", "surfacique"}
    and sorted(len(e["poutres"]) for e in ETATS.values())[-2:] == [2, 3], statuts)
print(f"--   durée de génération : {min(DUREES.values()):.2f} à {max(DUREES.values()):.2f} s par note (CPython, {len(ETATS)} notes)")
print(f"--   bloc Hypothèses (caractères sans blancs) : {min(m[0] for m in MESURES.values())} à {max(m[0] for m in MESURES.values())} (budget {SEUIL_HYPOTHESES})")
chk("aucune planche ne passe en page de suite sur les états d'essai (une page par poutre)", all(m[1] == 1 for m in MESURES.values()),
    [k for k, m in MESURES.items() if m[1] != 1])

# le chêne : k_c,90 = 1 (feuillu), k_h sans majoration, le verdict rouge ; le GL : l'appui « non vérifiée », w_net,fin, k_sys
t_ch = norm(" ".join(pages_brutes(NOTES["un chêne D30 maintenu aux appuis, poids propre non compté, qui ne tient pas"])))
r_ch = MO.verifier(ETATS["un chêne D30 maintenu aux appuis, poids propre non compté, qui ne tient pas"]["poutres"][0])
chk("le chêne D30 : « NON VÉRIFIÉ », la qualité « D30 chêne » au titre, k_c,90 = 1,000 (feuillu), le poids propre « (non compté) », la nature C",
    r_ch["statut"] == MO.NOK and "NON VÉRIFIÉ" in t_ch and "D30 chêne" in t_ch and plat_sym("k_c,90 = 1,000") in plat_sym(t_ch)
    and "(non compté)" in t_ch and "C · lieux de réunion" in t_ch, t_ch[:300])
t_gl = norm(" ".join(pages_brutes(NOTES["un GL28c sur mesure, entretoisé, service 2, effet système, appui non vérifié, contre-flèche, L/400 et L/300"])))
r_gl = MO.verifier(ETATS["un GL28c sur mesure, entretoisé, service 2, effet système, appui non vérifié, contre-flèche, L/400 et L/300"]["poutres"][0])
chk("le GL28c : « lamellé-collé résineux », l'appui « Non vérifiée », w_net,fin imprimé avec la contre-flèche, k_sys au panneau, « L/400 · L/300 »",
    "lamellé-collé résineux" in t_gl and "Non vérifiée" in t_gl and plat_sym("w_net,fin") in plat_sym(t_gl)
    and O.present(NT._fmt(r_gl["fleches"]["w_net_fin"], "mm"), t_gl) and "Effet système" in t_gl and "L/400 · L/300" in t_gl, t_gl[:300])
t_g = norm(" ".join(pages_brutes(NOTES["G seule (Q = 0 : une seule combinaison)"])))
chk("G seule : le tableau n'a qu'une ligne, « G + Q » n'est pas imprimé", "G + Q" not in t_g and "G seule" in t_g)

# ---------------------------------------------------------------- 2. le routeur : la note riche, une note d'avant (version 1)
r_tr = json.loads(R.traiter(json.dumps({"module": "bois", "action": "telecharger", "etat": copy.deepcopy(TROIS), "quoi": "note_pdf", "infos": INFOS})))
chk("le routeur rend la note riche du relevé (trois poutres) : ok, un PDF au même texte que la note faite ici",
    r_tr.get("ok") is True and O.pages(base64.b64decode(r_tr["fichier"]["base64"])) == O.pages(NOTES["trois poutres : la note riche du relevé du paquet (solive, chêne, GL)"]),
    r_tr.get("erreur"))
r_v1 = json.loads(R.traiter(json.dumps({"module": "bois", "action": "telecharger", "etat": copy.deepcopy(B.NOTE_V1), "quoi": "note_pdf", "infos": INFOS})))
t_v1 = norm(" ".join(pages_brutes(base64.b64decode(r_v1["fichier"]["base64"])))) if r_v1.get("ok") else ""
chk("une note d'avant (version 1) passe par le routeur : des poutres neuves qui gardent leurs noms, une planche chacune",
    r_v1.get("ok") is True and all(p["nom"] in t_v1 for p in B.NOTE_V1["poutres"])
    and len(pages_brutes(base64.b64decode(r_v1["fichier"]["base64"]))) == 1 + len(B.NOTE_V1["poutres"]), r_v1.get("erreur"))

# ---------------------------------------------------------------- 3. l'identification, le nom du fichier, la correspondance
e2 = ETATS["deux poutres (la solive, l'azobé)"]
chk("infos_doc : la correspondance de la poutre (projet = chantier, partie = nom, indice, date)",
    NB.infos_doc(e2, INFOS) == {"nom_projet": CHANTIER, "partie": NOM, "indice": "C", "date": "03/04/2026"})
chk("sans chantier : projet = le nom de la note, partie = « Poutres bois » ; sans infos : le nom de la poutre seule, indice A",
    NB.infos_doc(e2, {"nom": "Terrasse"}) == {"nom_projet": "Terrasse", "partie": NB.TITRE, "indice": "A"}
    and NB.infos_doc(ETATS["une solive en charges surfaciques (g, q, entraxe)"], {}) == {"nom_projet": "Solive S1", "partie": NB.TITRE, "indice": "A"}
    and NB.infos_doc(e2, {}) == {"nom_projet": NB.TITRE, "partie": NB.TITRE, "indice": "A"})
f_sans = NB.fichier(copy.deepcopy(ETATS["la poutre neuve (C24 épicéa 75 × 225, L = 4,00 m)"]), {})
t_sans = norm(" ".join(pages_brutes(base64.b64decode(f_sans["base64"]))))
chk("sans infos : le fichier porte le nom de la poutre, la garde dit « indice A » et la date du jour",
    f_sans["nom"] == "Note de calcul - P1.pdf" and "indice A" in t_sans and time.strftime("%d/%m/%Y") in t_sans, (f_sans["nom"], t_sans[:200]))
f_dang = NB.fichier(copy.deepcopy(e2), {"nom": "Note <b>/x"})
chk("un nom de note dangereux donne un nom de fichier sûr (celui de commun.nom_fichier)",
    f_dang["nom"] == C.nom_fichier("Note de calcul - Note <b>/x", "pdf") and "<" not in f_dang["nom"] and "/" not in f_dang["nom"]
    and C.refus_fichier(f_dang) is None, f_dang["nom"])

# ---------------------------------------------------------------- 4. la page de suite, les refus
vrai_verifs = NT._verifs


def verifs_bavards(r):
    """Soixante lignes de plus dans la première vérification : les colonnes débordent, la planche se poursuit (le bloc
    Hypothèses, lui, reste ce qu'il est : son budget ne bouge pas)."""
    out = vrai_verifs(r)
    out[0]["items"] += [("pr", f"Ligne de remplissage {i + 1} pour faire déborder les colonnes de la planche.") for i in range(60)]
    return out


NT._verifs = verifs_bavards
try:
    octets = NB.pdf(copy.deepcopy(ETATS["la poutre neuve (C24 épicéa 75 × 225, L = 4,00 m)"]), INFOS)
finally:
    NT._verifs = vrai_verifs
tx = pages_brutes(octets)
sec_a = NT.section(ETATS["la poutre neuve (C24 épicéa 75 × 225, L = 4,00 m)"]["poutres"][0])
titre_a = f"{sec_a['poutre']} — {sec_a['section']} (suite)"
chk("une planche trop longue (soixante lignes de plus) se POURSUIT : 1 + 2 pages, le titre « … (suite) » en page 3 et pas en page 2, le sommaire "
    "l'annonce en page 2, rien ne déborde, la parité tient",
    len(tx) == 3 and titre_a in norm(tx[2]) and titre_a not in norm(tx[1]) and [int(x) for x in re.findall(r"page (\d+)", tx[0])] == [2]
    and ecarts_parite(ETATS["la poutre neuve (C24 épicéa 75 × 225, L = 4,00 m)"], octets) == [], (len(tx), [t[:80] for t in tx[1:]]))
try:
    NB.fichier({"version": 2, "poutres": [dict(MO.nouvelle_poutre("Trop longue"), L=50.0)], "actif": 0}, INFOS)
    r_ = "aucune erreur"
except ValueError as err:
    r_ = str(err)
chk("une poutre hors domaine est refusée avec son nom et le message du moteur, jamais une note fausse", "« Trop longue »" in r_ and "portée" in r_.lower(), r_)
try:
    NB.fichier({"version": 2, "poutres": [], "actif": 0}, INFOS)
    r_ = "aucune erreur"
except ValueError as err:
    r_ = str(err)
chk("une note sans poutre est refusée", "aucune poutre" in r_, r_)
chk("generer_note accepte aussi une simple liste de poutres", NT.generer_note([MO.nouvelle_poutre()], {"indice": "B"})[:5] == b"%PDF-")
chk("la note refuse un autre téléchargement que « note_pdf »", B.telecharger(copy.deepcopy(e2), "autre", INFOS) is None)

# ---------------------------------------------------------------- 5. les mutations
e_a = ETATS["la poutre neuve (C24 épicéa 75 × 225, L = 4,00 m)"]
e_ch = ETATS["un chêne D30 maintenu aux appuis, poids propre non compté, qui ne tient pas"]
e_gl = ETATS["un GL28c sur mesure, entretoisé, service 2, effet système, appui non vérifié, contre-flèche, L/400 et L/300"]
vrai_sections = NT.sections
NT.sections = lambda e: vrai_sections(e)[1:]
try:
    mut = NB.pdf(copy.deepcopy(e2), INFOS)
finally:
    NT.sections = vrai_sections
chk("mutation « une poutre oubliée » : la parité rougit", ecarts_parite(e2, mut) != [])

vrai_section = NT.section


def section_sans_appui(p):
    """L'appui transmis au moteur n'est pas celui de la saisie (120 → 100 mm)."""
    return vrai_section(dict(p, appui=dict(p["appui"], a=100.0)))


NT.section = section_sans_appui
try:
    mut = NB.pdf(copy.deepcopy(e_ch), INFOS)
finally:
    NT.section = vrai_section
chk("mutation « une donnée non transmise au moteur » (la longueur d'appui) : la parité rougit", ecarts_parite(e_ch, mut) != [])

vrai_verdict = NT._verdict


def verdict_eta(r, v):
    k = vrai_verdict(r, v)
    k["cale"] = NT._pct(r["taux_max"])
    return k


NT._verdict = verdict_eta
try:
    mut = NB.pdf(copy.deepcopy(e_ch), INFOS)
finally:
    NT._verdict = vrai_verdict
ec = ecarts_parite(e_ch, mut)
chk("mutation « le taux de chaque verdict pris sur l'η max » : la parité rougit sur les taux des vérifications", ec and all("absent" in x for x in ec), ec[:3])


def section_eta_premiere(p):
    """L'η max de l'en-tête et du sommaire pris sur la PREMIÈRE vérification (la flexion) : sur le GL, la flèche gouverne."""
    sec = vrai_section(p)
    r = MO.verifier(p)
    assert r["gouvernant"]["cle"] != r["verifications"][0]["cle"], "mutation périmée : la flexion gouverne"
    sec["acier"] = sec["acier"].replace(NT._pct(r["taux_max"]), NT._pct(r["verifications"][0]["taux"]))
    return sec


NT.section = section_eta_premiere
try:
    mut = NB.pdf(copy.deepcopy(e_gl), INFOS)
finally:
    NT.section = vrai_section
ec = ecarts_parite(e_gl, mut)
chk("mutation « l'η max pris sur la première vérification » : la parité rougit, sur le sommaire et sur l'en-tête", ec and all("η max" in x for x in ec), ec[:3])

vrai_infos = NB.infos_doc
NB.infos_doc = lambda etat_, infos: {k: v for k, v in vrai_infos(etat_, infos).items() if k != "indice"}
try:
    mut = NB.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NB.infos_doc = vrai_infos
chk("mutation « l'indice de la note perdu » : l'identification rougit (« indice 0 » imprimé)", ecarts_identification(mut) != [])

vrai_items = NT._items
NT._items = lambda r, cles, sauf=(): vrai_items(r, cles, sauf=tuple(sauf) + ("W",))
try:
    mut = NB.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NT._items = vrai_items
ec = ecarts_parite(e_a, mut)
chk("mutation « un résultat de formule non imprimé » (W) : la parité rougit sur cette formule", ec and all("section/W" in x for x in ec), ec[:3])

vrai_blocs = NT._blocs
NT._blocs = lambda r: [(t, [row for row in rows if row[0] != "Qualité visuelle"]) for t, rows in vrai_blocs(r)]
try:
    mut = NB.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NT._blocs = vrai_blocs
ec = ecarts_parite(e_a, mut)
chk("mutation « la qualité visuelle non imprimée » : la parité rougit sur le panneau", ec and all("panneau" in x and "NF B 52-001-1" in x for x in ec), ec[:3])

vrai_generales = NT.HYPOTHESES_GENERALES
NT.HYPOTHESES_GENERALES = ("rien de général",)
try:
    mut = NB.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NT.HYPOTHESES_GENERALES = vrai_generales
ec = ecarts_parite(e_a, mut)
chk("mutation « la prose générale des hypothèses imprimée » : la parité rougit — les phrases interdites, et le budget du bloc",
    ec and any("prose générale" in x for x in ec) and any("trop longues" in x for x in ec) and all("hypothèses" in x for x in ec), ec[:4])

# ---------------------------------------------------------------- 6. comme dans le navigateur
e_nav = [e_a, e2, TROIS, ETATS["G seule (Q = 0 : une seule combinaison)"]]
infos_nav = {"nom": "relevé", "chantier": "C", "indice": "B", "date": "01/01/2026"}
s = O.sonde("bois", e_nav, infos_nav, liste="poutres")
chk("navigateur : sans fontTools ni accélérateurs C, avec le Pillow factice et les seules polices du paquet, la note se fait (dont la note à trois poutres)",
    "erreur" not in s and s["fonttools"] is False and s["rl_accel"] is False and s["pil"].startswith(O.STUB), s.get("erreur") or {k: v for k, v in s.items() if k != "sorties"})
if "erreur" not in s:
    cpy = [O.pages(NB.pdf(copy.deepcopy(x), infos_nav)) for x in e_nav]
    chk("navigateur : le texte de chaque page est celui de la note faite en CPython", [O.pages(o) for o in s["sorties"]] == cpy)
    pol = sorted({p for o in s["sorties"] for p in O.polices(o)})
    noms_paquet = {f[:-4] for f in K.NOTE_POLICES}
    chk(f"navigateur : les polices de la note sont celles du paquet (et Helvetica, police standard non embarquée) : {pol}",
        set(pol) <= noms_paquet | {"Helvetica"} and set(s["polices"]) <= set(K.NOTE_POLICES), (pol, s["polices"]))
    chk("navigateur : la note ne charge pas numpy (le moteur bois est pur, la note aussi)", s["numpy"] is False and s["numpy_ecran"] is False, (s["numpy"], s["numpy_ecran"]))
    with open(os.path.join(RACINE, "site", "web", "moteur", "manifeste.json"), encoding="utf-8") as fh:
        moteur = set(json.load(fh)["fichiers"])
    deja = {f for f in s["note"] if f.startswith(("ndc_pdf/", "site/note_pdf/")) or f.split("/")[0] in K.PAQUETS_ENTIERS}
    hors = sorted(set(s["note"]) - moteur - deja)
    print(f"--   fichiers du dépôt chargés par la note, hors écran, paquet du moteur et paquet de la poutre : {hors}")
    chk("navigateur : la note charge le générateur (bois/poutre/note.py) en plus de l'écran, et rien que le paquet du moteur ne porte déjà — "
        "note_bois.FICHIERS ne déclare rien, et c'est exact",
        "bois/poutre/note.py" in s["note"] and "bois/poutre/note.py" in moteur and hors == [] and NB.FICHIERS == (),
        (hors, [f for f in s["note"] if not f.startswith(("ndc_pdf/", "site/note_pdf/"))]))
if "bois" in K.NOTE_PDF:
    octets_zip, desc = K.paquet_note(modules=["bois"])
    noms = set(zipfile.ZipFile(io.BytesIO(octets_zip)).namelist())
    chk("paquet de la note (note.zip, relevé sur la note riche du bois) : aucun fichier de bois/ n'y entre (le paquet du moteur les a), la note ne "
        "charge pas numpy", not any(n.startswith("bois/") for n in noms) and "bois" not in desc.get("avec_numpy", ()),
        ([n for n in noms if n.startswith("bois/")], desc.get("avec_numpy")))
else:
    print("--   paquet de la note : « bois » n'est pas dans construire.NOTE_PDF — contrôle en attente de l'intégration")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
