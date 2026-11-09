# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Garde-corps » sur le site (api_web/garde_corps.telecharger → api_web/note_cas.py), vérifiée
en Python ordinaire.

- TOUTE la note en un PDF : deux pages par garde-corps, dans l'ordre de la liste ; le texte de chaque page (pdftotext)
  est celui de la note de ce garde-corps faite SEULE par le générateur validé (acier/garde_corps/note.py), et la taille
  retenue (normale, serrée) est la sienne — ce qui lie les deux copies nommées de note_cas (la boucle des variantes, le
  corps de ``_essai``) ;
- PARITÉ avec l'écran du site, cas par cas : le calcul que la note refait (``Calcul``) rend les taux, statuts,
  gouvernants et profils du moteur de l'écran ; le verdict, chaque taux de la colonne Résultat, chaque profil et
  chaque résultat des fenêtres de calcul (les formules de la note, efforts compris) s'écrivent dans les pages du cas.
  Seul écart, MESURÉ et rapporté (pas corrigé : la note est le générateur validé) : en bord de bande, l'écran arrondit
  le taux SANS quitter sa bande de couleur (94,57 % → « 94 % », 100,009 % → « 101 % »), la note à l'entier ordinaire
  (« 95 % », « 100 % » — sous « NON CONFORME ») ;
- l'identification : la note du site (nom, chantier, indice, date) arrive dans l'en-tête et le cartouche de chaque
  garde-corps, avec la correspondance de la note de la poutre (pas de champ d'identification à l'écran du garde-corps :
  celle d'un fichier Streamlit ne s'imprime pas) ;
- le routeur la rend en base 64, nommée d'après la note (nom sûr) ; un autre fichier est refusé ; une note qui déborde
  est refusée, jamais tronquée ;
- MUTATIONS (chacune doit faire rougir un contrôle) : un garde-corps oublié, des infos mal transmises, la page 2 sans
  l'échelle de la page 1, la taille toujours « normale » ;
- comme dans le navigateur (sonde calquée sur celle du paquet de la note) : sans fontTools ni accélérateurs C, avec le
  Pillow factice et les seules polices du paquet, la note se fait et son texte est celui de CPython ; ses polices sont
  celles du paquet ; numpy est déjà chargé par l'écran du module ; les fichiers du dépôt qu'elle charge en plus sont
  ceux que ``garde_corps.NOTE.fichiers`` déclare.

    python3 tests/test_note_garde_corps.py
"""
import base64
import copy
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

import note_cas_outils as O  # noqa: E402
from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import note as NT  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import garde_corps as G  # noqa: E402
from api_web import note_cas as NC  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def faire(*saisies, base=None):
    """Une note construite par les saisies de l'écran (sur le garde-corps OUVERT) ; ``"+"`` ajoute un garde-corps."""
    e = copy.deepcopy(base) if base else G.nouvel_etat()
    for ch, v in saisies:
        if ch == "+":
            G.gerer_elements(e, "ajouter")
            continue
        m = G.modifier(e, ch, v)
        assert m is None, (ch, v, m)
    return G.normaliser(e)


ETATS = {
    "neuve (un garde-corps)": faire(),
    "trois garde-corps (dont un NON CONFORME, H = 1600, en taille serrée)": faire(
        (["nom"], "G1 Terrasse"), ("+", None), (["nom"], "G2 Lisses"), (["type"], "lisses"), (["geometrie", "H"], 1100),
        ("+", None), (["nom"], "G3 Haut"), (["geometrie", "H"], 1600)),
    "fixation latérale, Eurocode (taille serrée)": faire((["nom"], "Nez de dalle"), (["fixation", "type"], "laterale"),
                                                          (["methode", "mode"], "eurocode")),
    "consoles au mur, puis main courante de mur à mur": faire(
        (["type"], "murale"), ("+", None), (["type"], "main_courante"), (["geometrie", "debut"], "mur"), (["geometrie", "fin"], "mur")),
    "panneau en L, foules (C5) : NON CONFORME": faire((["type"], "panneau"), (["geometrie", "forme"], "L"), (["charges", "categorie"], "C5")),
    "bords de bande : 94,57 % et 100,009 %": faire((["geometrie", "H"], 1340), ("+", None), (["geometrie", "H"], 1378)),
}
INFOS = {"nom": "Terrasse", "chantier": "Résidence Les Tilleuls", "indice": "B", "date": "01/02/2026"}
STATUT_ECRAN = {ET.OK: "p", ET.ATTENTION: "f", ET.NOK: "r"}
DECIMALE = lambda t: re.sub(r"(\d)\.(\d)", r"\1,\2", t)  # noqa: E731

# 1. la note de toute la note : un PDF, deux pages par garde-corps, chaque page = celle du garde-corps seul
NOTES = {}
for nom, e in ETATS.items():
    f, octets, ks = O.note_et_tailles(G, e, INFOS)
    NOTES[nom] = octets
    n = len(e["cas"])
    chk(f"{nom} : un PDF de {2 * n} pages ({n} garde-corps), nommé d'après la note",
        octets[:5] == b"%PDF-" and len(O.pages(octets)) == 2 * n and f["nom"] == "Note de calcul - Terrasse.pdf"
        and f["type"] == "application/pdf", (octets[:8], len(O.pages(octets)), f["nom"]))
    ecarts = O.controle_identite(G, e, INFOS, octets, ks)
    chk(f"{nom} : chaque page est celle du garde-corps fait seul par le générateur, dans sa taille", not ecarts, ecarts[:3])
serrees = sum(1 for e in ETATS.values() for c in e["cas"] if (NT.generer_note(e, c, {}), NT.derniere_variante)[1] == "serrée")
chk(f"les états d'essai comptent des garde-corps en taille serrée ({serrees}) : la copie de la boucle des variantes est jouée",
    serrees >= 3, serrees)

# 2. la parité avec l'écran du site
bords, n_cas, n_fx, n_lignes = [], 0, 0, 0
ecarts = []
for nom, e in ETATS.items():
    tx = O.pages(NOTES[nom])
    for i, c in enumerate(e["cas"]):
        n_cas += 1
        ei = copy.deepcopy(e)
        ei["actif"] = i
        r = G.resultat(ei)
        site = ET.verifier(ei["cas"][i])
        note = NT.Calcul(ei, ei["cas"][i]).res
        # le calcul de la note = le moteur de l'écran
        cle = lambda x: (x["statut"], x["taux_max"], (x.get("gouvernant") or {}).get("libelle"),  # noqa: E731
                         [(el["cle"], el["libelle"], el["taux"], el["statut"]) for el in x.get("elements", [])])
        if cle(site) != cle(note):
            ecarts.append((nom, i, "calcul", cle(site)[:3], cle(note)[:3]))
        p1, p12 = O.norm(tx[2 * i]), O.norm(tx[2 * i] + " " + tx[2 * i + 1])
        # le verdict : la grande pastille, le statut, le gouvernant
        v = O.verdict(r["resultat"])
        t = site["taux_max"]
        if v is None or v[0] != STATUT_ECRAN.get(site["statut"]) or f"η max {O.pct_note(t)}" not in p1 or NT.STATUTS[site["statut"]][0] not in p1:
            ecarts.append((nom, i, "verdict", v, O.pct_note(t), NT.STATUTS[site["statut"]][0]))
        elif v[1] != O.pct_note(t):
            bords.append((nom, c.get("nom"), "η max", v[1], O.pct_note(t)))
        gov = site["gouvernant"]["libelle"].split(" · ")[0]
        if f"gouverne : {gov}" not in p1 or f"<b>{C.esc(gov)}</b>" not in r["resultat"]:
            ecarts.append((nom, i, "gouvernant", gov))
        # chaque ligne de la colonne Résultat : son taux, son profil
        lignes = O.lignes_resultat(r["resultat"])
        if len(lignes) != len(site["elements"]):
            ecarts.append((nom, i, "lignes", len(lignes), len(site["elements"])))
        for (nom_l, piece, niv, txt), el in zip(lignes, site["elements"]):
            n_lignes += 1
            if not O.present(O.pct_note(el["taux"]), p1) or niv != C.niveau(el["taux"]):
                ecarts.append((nom, i, nom_l, txt, O.pct_note(el["taux"])))
            elif txt != O.pct_note(el["taux"]):
                bords.append((nom, c.get("nom"), nom_l, txt, O.pct_note(el["taux"])))
            # le profil (la note écrit le nom du catalogue, « 2.5 », et le libellé du moteur, « 2,5 » : même profil)
            if el["cle"] in ET.ROLES and DECIMALE(piece) not in DECIMALE(p12):
                ecarts.append((nom, i, nom_l, "profil", piece))
        # chaque résultat des fenêtres de calcul (les formules de la note, les efforts en pied)
        for fx, res in O.resultats_fenetres(r["volets"]):
            n_fx += 1
            if not O.present(res, p12):
                ecarts.append((nom, i, "fenêtre", res, fx[:80]))
chk(f"parité : {n_cas} garde-corps, {n_lignes} lignes du résultat, {n_fx} résultats des fenêtres de calcul — la note écrit les "
    "taux, statuts, gouvernants, profils, efforts et formules de l'écran", not ecarts and n_cas >= 9 and n_fx >= 400, ecarts[:4])
chk("le seul écart mesuré est l'arrondi en bord de bande (l'écran tient la bande, la note arrondit à l'entier)",
    sorted({(b[3], b[4]) for b in bords}) == [("101 %", "100 %"), ("94 %", "95 %")], bords)
for b in sorted(set((x[3], x[4]) for x in bords)):
    print(f"--   écart d'arrondi rapporté : écran « {b[0]} », note « {b[1]} »")

# 3. l'identification
e = ETATS["trois garde-corps (dont un NON CONFORME, H = 1600, en taille serrée)"]
attendu = lambda i, c: ["Note de calcul · 01/02/2026 · indice B", f"Résidence Les Tilleuls · Terrasse · cas {c['nom']}"]  # noqa: E731
ecarts = O.controle_infos(NOTES["trois garde-corps (dont un NON CONFORME, H = 1600, en taille serrée)"], e, attendu)
chk("le chantier, le nom, l'indice et la date de la note du site sont dans l'en-tête de chaque garde-corps", not ecarts, ecarts)
_, sans_chantier, _ = O.note_et_tailles(G, e, {"nom": "Terrasse", "indice": "C", "date": "03/03/2026"})
ecarts = O.controle_infos(sans_chantier, e, lambda i, c: ["indice C", "03/03/2026", f"Terrasse · Garde-corps · cas {c['nom']}"])
chk("sans chantier : projet = le nom de la note, partie = « Garde-corps » (la correspondance de la poutre)", not ecarts, ecarts)
ident = copy.deepcopy(ETATS["neuve (un garde-corps)"])
ident["identification"] = {"projet": "Projet du fichier", "partie": "Partie cachée", "indice": "Z"}
f, octets, _ = O.note_et_tailles(G, ident, {})
t = O.norm(" ".join(O.pages(octets)))
chk("sans infos : le nom du garde-corps nomme la note ; l'identification d'un fichier, que l'écran ne montre pas, ne s'imprime pas",
    f["nom"] == "Note de calcul - Garde-corps 1.pdf" and "Garde-corps 1 · Garde-corps · cas Garde-corps 1" in t and "indice A" in t
    and "Projet du fichier" not in t and "Partie cachée" not in t, (f["nom"], t[:300]))
chk("infos_du_cas : la correspondance écrite (projet = chantier, partie = nom, indice, date)",
    NC.infos_du_cas(G.NOTE, e, e["cas"][0], INFOS) == {"nom_projet": "Résidence Les Tilleuls", "partie": "Terrasse", "indice": "B", "date": "01/02/2026"})

# 4. le routeur, les refus
r = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "telecharger", "etat": e, "quoi": "note_pdf",
                                     "infos": {"nom": "Note <b>", "chantier": "C"}})))
chk("le routeur rend la note en base 64 sous un nom de fichier sûr",
    r.get("ok") and r["fichier"]["nom"] == "Note de calcul - Note -b.pdf" and base64.b64decode(r["fichier"]["base64"])[:5] == b"%PDF-",
    r.get("erreur") or r["fichier"]["nom"])
r = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "telecharger", "etat": e, "quoi": "autre"})))
chk("un autre fichier que la note est refusé avec un message", not r["ok"] and "inconnu" in r["erreur"], r)
vrai = NT._essai
NT._essai = lambda calc, doc_meta, k, chemin: vrai(calc, doc_meta, k, chemin) + ["page 1, colonne 22 : 7 pt"]
try:
    r1 = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "telecharger", "etat": ETATS["neuve (un garde-corps)"], "quoi": "note_pdf"})))
    r3 = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "telecharger", "etat": e, "quoi": "note_pdf"})))
finally:
    NT._essai = vrai
chk("une note qui déborde encore en taille serrée est refusée, jamais tronquée (un garde-corps, puis trois)",
    not r1["ok"] and not r3["ok"] and all("ne tient pas" in x["erreur"] and "7 pt" in x["erreur"] for x in (r1, r3))
    and "« G1 Terrasse »" in r3["erreur"], (r1.get("erreur"), r3.get("erreur")))

vrai_pages = NC.pages
NC.pages = lambda *a: vrai_pages(*a) + ["page 2, colonne 300 : 2 pt"]
try:
    r3b = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "telecharger", "etat": e, "quoi": "note_pdf"})))
finally:
    NC.pages = vrai_pages
chk("… et le débordement relevé en posant les pages d'un cas dans la note entière (second garde-fou) l'est aussi",
    not r3b["ok"] and "ne tient pas" in r3b["erreur"] and "2 pt" in r3b["erreur"], r3b.get("erreur"))

# 5. les mutations : chacune doit faire rougir un contrôle
e = ETATS["trois garde-corps (dont un NON CONFORME, H = 1600, en taille serrée)"]
vrai_pages = NC.pages
compte = []


def sans_dernier(NT_, gen, d, calc, doc_meta, k):
    compte.append(1)
    return [] if len(compte) == 3 else vrai_pages(NT_, gen, d, calc, doc_meta, k)


NC.pages = sans_dernier
try:
    mut = NC.pdf(G.NOTE, copy.deepcopy(e), INFOS)
finally:
    NC.pages = vrai_pages
chk("mutation « un garde-corps oublié » : le contrôle d'identité rougit", O.controle_identite(G, e, INFOS, mut) != [])
vrai_infos = NC.infos_du_cas
NC.infos_du_cas = lambda gen, etat, cas, infos: {}
try:
    mut = NC.pdf(G.NOTE, copy.deepcopy(e), INFOS)
finally:
    NC.infos_du_cas = vrai_infos
chk("mutation « infos mal transmises » : le contrôle de l'identification rougit", O.controle_infos(mut, e, attendu) != [])
vrai_note = G.NOTE
G.NOTE = G.NOTE._replace(suite=lambda n1: (None,))
try:
    mut = NC.pdf(G.NOTE, copy.deepcopy(e), INFOS)
finally:
    G.NOTE = vrai_note
chk("mutation « page 2 sans l'échelle de la page 1 » : le contrôle d'identité rougit", O.controle_identite(G, e, INFOS, mut) != [])
vrai_var = NC.variante
for k, nom_k in NT.VARIANTES:
    NC.variante = lambda NT_, calc, doc_meta, brouillon, k=k: (k, [])
    try:
        _, mut, ks = O.note_et_tailles(G, e, INFOS)
        rouge = O.controle_identite(G, e, INFOS, mut, ks) != []
    except AssertionError as err:      # la taille normale fait déborder G3 : la note est refusée — rouge aussi
        rouge, ks = "ne tient pas" in str(err), str(err)[:80]
    finally:
        NC.variante = vrai_var
    chk(f"mutation « toujours la taille {nom_k} » : un contrôle rougit (tailles, ou refus du débordement)", rouge, ks)

# 6. comme dans le navigateur
e_nav = [ETATS["neuve (un garde-corps)"], ETATS["trois garde-corps (dont un NON CONFORME, H = 1600, en taille serrée)"],
         ETATS["fixation latérale, Eurocode (taille serrée)"], ETATS["consoles au mur, puis main courante de mur à mur"]]
infos_nav = {"nom": "relevé", "chantier": "C", "indice": "B", "date": "01/01/2026"}
s = O.sonde("garde_corps", e_nav, infos_nav)
chk("navigateur : sans fontTools ni accélérateurs C, avec le Pillow factice et les polices du paquet, la note se fait",
    "erreur" not in s and s["fonttools"] is False and s["rl_accel"] is False and s["pil"].startswith(O.STUB), s.get("erreur") or s)
if "erreur" not in s:
    cpy = [O.pages(base64.b64decode(G.telecharger(copy.deepcopy(x), "note_pdf", infos_nav)["base64"])) for x in e_nav]
    chk("navigateur : le texte de chaque page est celui de la note faite en CPython", [O.pages(o) for o in s["sorties"]] == cpy)
    pol = sorted({p for o in s["sorties"] for p in O.polices(o)})
    noms_paquet = {f[:-4] for f in s["polices_paquet"]}
    chk(f"navigateur : les polices de la note sont celles du paquet (et Helvetica, police standard non embarquée) : {pol}",
        set(pol) <= noms_paquet | {"Helvetica"} and set(s["polices"]) <= set(s["polices_paquet"]), (pol, s["polices"]))
    chk("navigateur : numpy est déjà chargé par l'écran du module (avec_numpy) — la note ne l'ajoute pas", s["numpy_ecran"] is True)
    rel = O.releve_paquet(G, s)
    print(f"--   fichiers du dépôt chargés par la note, hors paquet du moteur et paquet de la poutre : {rel['hors']}")
    chk("navigateur : ces fichiers sont ceux que garde_corps.NOTE.fichiers déclare (le paquet de la note les reçoit)",
        set(rel["hors"]) <= set(rel["declares"]) and not rel["non_charges"], rel)
manque = O.paquet_integre(G)
if manque is None:
    print("--   paquet de la note : « garde_corps » n'est pas encore dans construire.NOTE_PDF — contrôle en attente de l'intégration")
else:
    chk("paquet de la note (note.zip) : il porte les fichiers que la note du garde-corps déclare", not manque, manque)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
