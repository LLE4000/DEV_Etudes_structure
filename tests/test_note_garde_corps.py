# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Garde-corps » sur le site (api_web/garde_corps.telecharger → api_web/note_cas.py), vérifiée
en Python ordinaire.

- TOUTE la note en un PDF : deux pages par garde-corps (trois avec sa page de suite), dans l'ordre de la liste ; le
  texte de chaque page (pdftotext) est celui de la note de ce garde-corps faite SEULE par le générateur validé
  (acier/garde_corps/note.py), et la taille retenue (normale, serrée) est la sienne — le site lui demande sa mise en
  page (``mise_en_page``) et lui fait poser ses pages (``poser``), sans copie ;
- un garde-corps qui DÉBORDAIT de ses deux pages (en U avec « Cumuler qh et qv » : refusé avant le 09/11/2026) sort
  complet, en trois pages ; une note de plusieurs garde-corps qui en compte n'est pas refusée (la grille complète :
  tests/test_note_garde_corps_grille.py) ;
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
- MUTATIONS (chacune doit faire rougir un contrôle) : un garde-corps oublié, des infos mal transmises, la page de suite
  oubliée dans la note entière, la taille toujours « normale » ou toujours « serrée » ;
- comme dans le navigateur (sonde calquée sur celle du paquet de la note), PAGE DE SUITE COMPRISE : sans fontTools ni
  accélérateurs C, avec le Pillow factice et les seules polices du paquet, la note se fait et son texte est celui de
  CPython ; ses polices sont celles du paquet ; numpy est déjà chargé par l'écran du module ; les fichiers du dépôt
  qu'elle charge en plus sont ceux que ``garde_corps.NOTE.fichiers`` déclare ; et la note « riche » du relevé du paquet
  (construire.NOTES_RELEVE) passe elle aussi par la page de suite (relecture du 09/11/2026).

    python3 tests/test_note_garde_corps.py
"""
import base64
import copy
import json
import os
import re
import sys
import tempfile

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
    f["nom"] == "Note de calcul - G1.pdf" and "G1 · Garde-corps · cas G1" in t and "indice A" in t
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

# le débordement du GÉNÉRATEUR lui-même (relecture du 09/11/2026) : un garde-corps en U avec « Cumuler qh et qv »
# (l'option côté sûr) ne tenait pas dans ses deux pages, même serrées (28 pt en page 1, 73 pt en page 2) — la note du
# site était REFUSÉE (86 des 240 états d'une grille). Depuis la page de suite du générateur (acier/garde_corps/note.py,
# preuve complète : tests/test_note_garde_corps_grille.py), elle sort COMPLÈTE : trois pages, rien qui déborde, le
# récapitulatif et la bibliothèque en page 3. L'écrit suit (DEFAUTS_STREAMLIT.md, MODULES_SITE.md) : ce contrôle le lie.
u_cumul = faire((["geometrie", "forme"], "U"), (["charges", "cumul"], True))
ru = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "telecharger", "etat": u_cumul, "quoi": "note_pdf",
                                      "infos": INFOS})))
with open(os.path.join(RACINE, "docs", "site", "DEFAUTS_STREAMLIT.md"), encoding="utf-8") as fh:
    defauts = " ".join(fh.read().split())
with open(os.path.join(RACINE, "docs", "site", "MODULES_SITE.md"), encoding="utf-8") as fh:
    modules_site = " ".join(fh.read().split())
tu =O.pages(base64.b64decode(ru["fichier"]["base64"])) if ru.get("ok") else []
chk("un garde-corps en U avec « Cumuler qh et qv » (refusé avant la page de suite) : la note sort, COMPLÈTE, en trois pages "
    "— la page 3 porte le récapitulatif et la bibliothèque, la page 1 et la page 2 le disent ; l'écrit suit "
    "(DEFAUTS_STREAMLIT.md, MODULES_SITE.md)",
    ru.get("ok") and len(tu) == 3 and "SUITE DES PAGES 1 ET 2" in tu[2]
    and all(x in re.sub(r"\s+", "", tu[2]) for x in ("RÉCAPITULATIF", "SECTIONDEL'ÉLÉMENTPRINCIPALETBIBLIOTHÈQUE"))
    and "Suite p. 3 : récapitulatif." in O.norm(tu[0]) and "Suite p. 3 : section et bibliothèque." in O.norm(tu[1])
    and "page de suite" in defauts and "86 des 240 états d'une grille du garde-corps débordaient" in modules_site, ru.get("erreur") or [t[:200] for t in tu])

# plusieurs garde-corps, dont deux qui débordaient : la note entière n'est pas refusée, chaque garde-corps y a ses pages
# (trois avec la page de suite), chacune celle du garde-corps fait seul par le générateur
deborde = faire((["nom"], "U cumul"), (["geometrie", "forme"], "U"), (["charges", "cumul"], True), ("+", None), (["nom"], "Droit"),
                ("+", None), (["nom"], "L soudé"), (["geometrie", "forme"], "L"), (["fixation", "type"], "soudee"))
f_d, o_d, ks_d = O.note_et_tailles(G, deborde, INFOS)
ecarts = O.controle_identite(G, deborde, INFOS, o_d, ks_d)
chk(f"trois garde-corps dont deux qui débordaient (U avec cumul, L à montant soudé) : une note de {len(O.pages(o_d))} pages (3 + 2 + 3), "
    "chaque page celle du garde-corps fait seul, page de suite comprise", not ecarts and len(O.pages(o_d)) == 8, ecarts[:3])
ecarts = O.controle_infos(o_d, deborde, lambda i, c: ["Note de calcul · 01/02/2026 · indice B", f"Résidence Les Tilleuls · Terrasse · cas {c['nom']}"],
                          nb_pages=[3, 2, 3])
chk("… et l'identification de la note du site dans les pages de chacun", not ecarts, ecarts)

# 5. les mutations : chacune doit faire rougir un contrôle
e = ETATS["trois garde-corps (dont un NON CONFORME, H = 1600, en taille serrée)"]
vrai_pages = NC.pages
compte = []


def sans_dernier(NT_, gen, d, calc, doc_meta, k, reports=()):
    compte.append(1)
    return [] if len(compte) == 3 else vrai_pages(NT_, gen, d, calc, doc_meta, k, reports)


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
# (la mutation « page 2 sans l'échelle de la page 1 » n'a plus d'objet ici : le garde-corps pose lui-même ses pages
# — NT.poser —, le site n'en garde plus de copie ; « la page de suite oubliée » la remplace)
NC.pages = lambda NT_, gen, d, calc, doc_meta, k, reports=(): vrai_pages(NT_, gen, d, calc, doc_meta, k, ())
try:
    r_sans = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "telecharger", "etat": deborde, "quoi": "note_pdf",
                                              "infos": INFOS})))
finally:
    NC.pages = vrai_pages
chk("mutation « la page de suite oubliée dans la note entière » : la note est refusée (le débordement est relevé), jamais tronquée",
    not r_sans["ok"] and "ne tient pas" in r_sans["erreur"], r_sans.get("erreur"))
vrai_disp = NC.disposition
for k, nom_k in NT.VARIANTES:
    NC.disposition = lambda NT_, calc, doc_meta, brouillon, k=k: (k, (), [])
    try:
        _, mut, ks = O.note_et_tailles(G, e, INFOS)
        rouge = O.controle_identite(G, e, INFOS, mut, ks) != []
    except AssertionError as err:      # la taille normale fait déborder G3 : la note est refusée — rouge aussi
        rouge, ks = "ne tient pas" in str(err), str(err)[:80]
    finally:
        NC.disposition = vrai_disp
    chk(f"mutation « toujours la taille {nom_k} » : un contrôle rougit (tailles, ou refus du débordement)", rouge, ks)

# 6. comme dans le navigateur — page de suite comprise (relecture du 09/11/2026 : la sonde, le relevé du paquet de la
# note et le harnais ne passaient que par des notes de deux pages ; un module refusé dans le navigateur, chargé sur le
# seul chemin de la page de suite, laissait cette suite et la construction au vert)
e_nav = [ETATS["neuve (un garde-corps)"], ETATS["trois garde-corps (dont un NON CONFORME, H = 1600, en taille serrée)"],
         ETATS["fixation latérale, Eurocode (taille serrée)"], ETATS["consoles au mur, puis main courante de mur à mur"],
         deborde]
infos_nav = {"nom": "relevé", "chantier": "C", "indice": "B", "date": "01/01/2026"}
sys.path.insert(0, os.path.join(RACINE, "site", "web"))
try:
    import construire as K  # noqa: E402
finally:
    sys.path.pop(0)
_ns = {}
exec(K.NOTES_RELEVE["garde_corps"], _ns)  # noqa: S102 — le code de construire.NOTES_RELEVE
reports_riche = []
for c in _ns["etat"]["cas"]:
    fd_r, ch_r = tempfile.mkstemp(suffix=".pdf")
    os.close(fd_r)
    try:
        NT.generer_note(_ns["etat"], c, {}, chemin=ch_r)
    finally:
        os.remove(ch_r)
    reports_riche.append((c["nom"], NT.derniers_reports, list(NT.derniers_avertissements)))
chk("la note « riche » du relevé du paquet de la note (construire.NOTES_RELEVE, rejouée paquets déballés par "
    "test_site_web.py et dans le vrai navigateur par le harnais « note_pdf ») passe par la page de suite, sans débordement",
    any(r for _n, r, _w in reports_riche) and not any(w for _n, _r, w in reports_riche), reports_riche)
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

# 7. lot B (brief v12 § 19.1 point 8) : le champ H se vide — 1 100 mm, dit à l'écran ET dans la note ; 0 mm est refusé
def ctl_h_vide_nulle() -> list:
    """Les écarts : champ H vidé sur l'écran du site (accepté, le moteur prend 1 100 mm), message à l'écran, ligne
    « Hauteur » de la note avec « (champ vide : valeur par défaut) » ; H = 0 : refus nommé, « Calcul impossible »."""
    ecarts = []
    ev = G.nouvel_etat()
    if G.modifier(ev, ["geometrie", "H"], "") is not None or ev["cas"][0]["geometrie"]["H"] is not None:
        return ["le champ H vidé est refusé ou n'est pas vide dans l'état"]
    html_ = G.resultat(G.normaliser(ev))["resultat"]
    if "Hauteur non saisie : H = 1\u202f100 mm par défaut" not in html_:
        ecarts.append("message de l'écran absent")
    t = O.norm(" ".join(O.pages(O.note_et_tailles(G, G.normaliser(ev), INFOS)[1])))
    if "1100 mm (champ vide : valeur par défaut)" not in t:
        ecarts.append("note : « (champ vide : valeur par défaut) » absent")
    e0 = G.nouvel_etat()
    if G.modifier(e0, ["geometrie", "H"], "0") is not None:
        return ecarts + ["H = 0 refusé à la saisie (le moteur doit le dire)"]
    r0 = G.resultat(G.normaliser(e0))["resultat"]
    if "Calcul impossible" not in r0 or "hauteur nulle (H = 0 mm)" not in r0:
        ecarts.append("H = 0 : pas de refus nommé")
    return ecarts


ec = ctl_h_vide_nulle()
chk("lot B · champ H vide : 1 100 mm, « Hauteur non saisie : H = 1 100 mm par défaut » à l'écran, « (champ vide : valeur par "
    "défaut) » dans la note ; H = 0 mm : « Calcul impossible », refus nommé", not ec, ec)
vrai_va = G.VIDE_ADMIS
G.VIDE_ADMIS = set()
try:
    chk("mutation « champ H vide refusé » (l'écran d'avant : « Nombre illisible ») : le contrôle rougit", ctl_h_vide_nulle() != [])
finally:
    G.VIDE_ADMIS = vrai_va
vrai_hd = NT.Calcul.h_defaut
NT.Calcul.h_defaut = lambda self: ""
try:
    chk("mutation « la note ne dit pas la valeur par défaut » : le contrôle rougit", ctl_h_vide_nulle() != [])
finally:
    NT.Calcul.h_defaut = vrai_hd


def ctl_h_vidable_page() -> list:
    """Le champ H se vide DANS LA PAGE, pas seulement par l'API : un champ « nombre » est refusé vide par la page
    (site/web/js/commun.js, valeurSaisie : « un nombre est attendu ») et reprend sa valeur en sortie (viderAuFocus) —
    il doit partir en texte, « vide = défaut » (C.num(..., auto=…) : data-vide, texte indicatif « 1 100 par défaut »).
    Champ vidé : il s'affiche vide, le texte indicatif en clair, et l'en-tête de la rubrique dit « H 1100 (défaut) »."""
    ecarts = []
    motif = re.compile(r'<input data-chemin="geometrie/H"[^>]*>')
    e = G.nouvel_etat()
    champ = (motif.findall(G.formulaire(e)) or [""])[0]
    if 'data-sorte="texte"' not in champ or 'data-vide="auto"' not in champ or 'placeholder="1 100 par défaut"' not in champ:
        ecarts.append(f"champ H non vidable dans la page : {champ}")
    G.modifier(e, ["geometrie", "H"], "")
    f = G.formulaire(G.normaliser(e))
    champ = (motif.findall(f) or [""])[0]
    if 'value=""' not in champ:
        ecarts.append(f"champ H vidé non affiché vide : {champ}")
    if "H 1100 (défaut)" not in f:
        ecarts.append("en-tête de la rubrique sans la hauteur par défaut")
    e2 = G.nouvel_etat()
    if G.modifier(e2, ["geometrie", "H"], "douze") != "Nombre illisible : « douze »" or e2 != G.nouvel_etat():
        ecarts.append("un texte illisible n'est pas refusé")
    return ecarts


ec = ctl_h_vidable_page()
chk("lot B · champ H vidable DANS LA PAGE (texte « vide = défaut », data-vide, « 1 100 par défaut ») ; vidé : affiché vide, "
    "en-tête « H 1100 (défaut) » ; « douze » refusé", not ec, ec)
vrai_hv = G.H_VIDE
G.H_VIDE = ""
try:
    chk("mutation « champ H en nombre » (la page refusait le champ vide) : le contrôle rougit", ctl_h_vidable_page() != [])
finally:
    G.H_VIDE = vrai_hv
vrai_ht = G._h_tete
G._h_tete = C.nb
try:
    chk("mutation « en-tête “Garde-corps · H” sans valeur » : le contrôle rougit", ctl_h_vidable_page() != [])
finally:
    G._h_tete = vrai_ht

# 8. lot B (contre-vérification 2) : le champ H SE VIDE QUAND ON CLIQUE DEDANS, comme tout champ numérique (règle du
# propriétaire du 13/11, site/web/js/commun.js : viderAuFocus), sans perdre « vide = défaut » : quitté sans rien taper,
# il reprend sa valeur (rien n'est envoyé) ; EFFACÉ (Retour arrière / Suppr dans le champ vidé, ou ce qu'on a tapé), il
# part vide et le moteur prend 1 100 mm. Le mécanisme de la page est JOUÉ par node sur les champs que le module rend
# (un DOM minimal : les écouteurs de la racine, closest() sur les sélecteurs de commun.js), puis ce qui part passe par
# le module (G.modifier).
import shutil  # noqa: E402
import subprocess  # noqa: E402

JS_COMMUN = os.path.join(RACINE, "site", "web", "js", "commun.js")
ESSAI_VIDER = r"""
const c = await import(process.argv[1]);
const champs = JSON.parse(process.argv[2]);
// un champ : ses attributs (data-* → dataset), value / defaultValue / placeholder, closest() sur les sélecteurs simples
// de commun.js (balise, [attr], [attr="v"], :not([attr]), :not(:disabled), alternatives séparées par des virgules)
function champ(a) {
  const ds = {};
  for (const [k, v] of Object.entries(a)) if (k.startsWith("data-")) ds[k.slice(5).replace(/-(\w)/g, (_, x) => x.toUpperCase())] = v;
  const el = { tagName: "INPUT", dataset: ds, value: a.value || "", defaultValue: a.value || "", placeholder: a.placeholder || "",
    disabled: "disabled" in a, readOnly: "readonly" in a };
  const attr = (n) => (n.startsWith("data-") ? el.dataset[n.slice(5).replace(/-(\w)/g, (_, x) => x.toUpperCase())]
    : n === "readonly" ? (el.readOnly ? "" : undefined) : n === "disabled" ? (el.disabled ? "" : undefined) : undefined);
  const un = (s) => {
    s = s.trim();
    if (!/^input\b/.test(s)) return false;
    if (/:not\(:disabled\)/.test(s) && el.disabled) return false;
    for (const m of s.matchAll(/:not\(\[([\w-]+)\]\)/g)) if (attr(m[1]) !== undefined) return false;
    const reste = s.replace(/:not\([^)]*\)/g, "");
    for (const m of reste.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)) {
      const v = attr(m[1]);
      if (v === undefined || (m[2] !== undefined && v !== m[2])) return false;
    }
    return true;
  };
  el.closest = (sel) => (sel.split(",").some(un) ? el : null);
  return el;
}
const ecoute = [];
const racine = { addEventListener: (type, f, o = {}) => ecoute.push({ type, f, capture: !!o.capture }) };
c.viderAuFocus(racine, {});
const emettre = (type, el, x = {}) => {
  const ev = { type, target: el, ...x };
  for (const l of ecoute.filter((l) => l.type === type && l.capture)) l.f(ev);
  for (const l of ecoute.filter((l) => l.type === type && !l.capture)) l.f(ev);
};
// ce que l'établi envoie en quittant le champ (js/etabli.js : la valeur, si elle diffère de defaultValue) ; null : rien
const quitter = (el) => { emettre("focusout", el); return el.value !== el.defaultValue ? el.value : null; };
const taper = (el, t) => { el.value = t; emettre("input", el); };
const vu = (el) => ({ v: el.value, ph: el.placeholder });
const out = {};
let h = champ(champs.H);
emettre("focusin", h); out.clic = vu(h);
out.sans_rien = { envoye: quitter(h), ...vu(h) };
h = champ(champs.H); emettre("focusin", h); taper(h, "1300"); out.tape = { envoye: quitter(h) };
h = champ(champs.H); emettre("focusin", h); emettre("keydown", h, { key: "Backspace" }); out.efface = vu(h);
out.efface.envoye = quitter(h); out.efface.apres = vu(h);
h = champ(champs.H); emettre("focusin", h); emettre("keydown", h, { key: "Delete" }); out.suppr = { envoye: quitter(h) };
h = champ(champs.H); emettre("focusin", h); taper(h, "5"); taper(h, ""); out.retape = { envoye: quitter(h) };
h = champ(champs.H); emettre("focusin", h); emettre("keydown", h, { key: "Backspace" }); taper(h, "1250"); out.reprise = { envoye: quitter(h) };
const p = champ(champs.pente);
emettre("focusin", p); out.pente_clic = vu(p); emettre("keydown", p, { key: "Backspace" }); out.pente = { envoye: quitter(p), ...vu(p) };
const t = champ(champs.texte);
emettre("focusin", t); out.texte = vu(t);
const hv = champ(champs.H_vide);
emettre("focusin", hv); out.h_vide = vu(hv);
console.log(JSON.stringify(out));
"""


def _attributs(balise: str) -> dict:
    a = dict(re.findall(r'([\w-]+)="([^"]*)"', balise))
    for nu in ("disabled", "readonly"):
        if re.search(rf"\s{nu}(?=[\s>])", balise):
            a[nu] = ""
    return a


def juger_vider_au_clic(js=JS_COMMUN) -> list:
    """Les écarts du champ H joué dans la page (node) : vidé au clic, sa valeur en gris ; quitté sans rien taper, il la
    reprend et rien ne part ; « 1300 » tapé part ; effacé (Retour arrière, Suppr, ou ce qu'on a tapé) il part vide et
    le module prend 1 100 mm ; la pente (un nombre) garde la règle des nombres ; un texte libre n'est pas vidé ; H déjà
    vide n'est pas touché."""
    e = G.nouvel_etat()
    f = G.formulaire(e)

    def balise(ch):
        return (re.findall(rf'<input data-chemin="{re.escape(ch)}"[^>]*>', f) or [""])[0]
    texte = _attributs(balise("geometrie/troncons/0/entraxes"))
    texte["value"] = "1000 ; 1000"
    ev = G.nouvel_etat()
    G.modifier(ev, ["geometrie", "H"], "")
    champs = {"H": _attributs(balise("geometrie/H")), "pente": _attributs(balise("geometrie/pente")), "texte": texte,
              "H_vide": _attributs((re.findall(r'<input data-chemin="geometrie/H"[^>]*>', G.formulaire(G.normaliser(ev)))
                                    or [""])[0])}
    try:
        p = subprocess.run(["node", "--input-type=module", "-e", ESSAI_VIDER, "--", "file://" + js, json.dumps(champs)],
                           capture_output=True, text=True, timeout=60)
        r = json.loads([x for x in p.stdout.split("\n") if x.strip()][-1])
    except Exception as ex:  # noqa: BLE001
        return [f"node : {ex} {locals().get('p') and p.stderr[-300:]}"]
    ecarts = []
    if r["clic"] != {"v": "", "ph": "1\u202f100"}:
        ecarts.append(f"clic dans H : pas vidé, valeur en gris ({r['clic']})")
    if r["sans_rien"] != {"envoye": None, "v": "1\u202f100", "ph": "1\u202f100 par défaut"}:
        ecarts.append(f"H quitté sans rien taper : {r['sans_rien']}")
    if r["tape"]["envoye"] != "1300" or r["reprise"]["envoye"] != "1250":
        ecarts.append(f"H tapé : {r['tape']} / {r['reprise']}")
    if r["efface"] != {"v": "", "ph": "1\u202f100 par défaut", "envoye": "", "apres": {"v": "", "ph": "1\u202f100 par défaut"}}:
        ecarts.append(f"H effacé (Retour arrière) : {r['efface']}")
    if r["suppr"]["envoye"] != "" or r["retape"]["envoye"] != "":
        ecarts.append(f"H effacé (Suppr / tapé puis effacé) : {r['suppr']} / {r['retape']}")
    if r["pente_clic"] != {"v": "", "ph": "0"} or r["pente"] != {"envoye": None, "v": "0", "ph": ""}:
        ecarts.append(f"pente (un nombre) : {r['pente_clic']} / {r['pente']}")
    if r["texte"]["v"] != "1000 ; 1000":
        ecarts.append(f"un texte libre (entraxes) vidé au clic : {r['texte']}")
    if r["h_vide"] != {"v": "", "ph": "1\u202f100 par défaut"}:
        ecarts.append(f"H déjà vide touché au clic : {r['h_vide']}")
    # ce qui part passe par le module : vidé → 1 100 mm par défaut ; tapé → la valeur
    ef = G.nouvel_etat()
    if r["efface"]["envoye"] != "" or G.modifier(ef, ["geometrie", "H"], r["efface"]["envoye"]) is not None \
            or ef["cas"][0]["geometrie"]["H"] is not None \
            or "Hauteur non saisie : H = 1 100 mm par défaut" not in G.resultat(G.normaliser(ef))["resultat"]:
        ecarts.append("H effacé dans la page : le module ne prend pas 1 100 mm par défaut")
    return ecarts


def _commun_mute(avant: str, apres: str) -> str:
    d = tempfile.mkdtemp(prefix="gc_commun_")
    s = open(JS_COMMUN, encoding="utf-8").read()
    assert s.count(avant) == 1, avant
    open(os.path.join(d, "commun.js"), "w", encoding="utf-8").write(s.replace(avant, apres))
    return os.path.join(d, "commun.js")


# … et le sélecteur élargi ne prend AUCUN autre texte : data-vide n'est posé que par commun.num (un champ NUMÉRIQUE
# « vide = automatique » ; test_api_web_modules : chacun accepte le vide et refuse un texte), aucun script de la page ne
# le pose. Les champs dont le comportement change sont donc tous des nombres « vide = automatique » (liste au rapport).
_API = os.path.join(RACINE, "api_web")
_JS = os.path.join(RACINE, "site", "web", "js")
poseurs = sorted(os.path.relpath(os.path.join(d, f), RACINE) for d in (_API, _JS) for f in os.listdir(d)
                 if f.endswith((".py", ".js")) and f != "commun.js" and "data-vide" in open(os.path.join(d, f), encoding="utf-8").read())
chk("data-vide n'est posé que par commun.num (champ numérique « vide = automatique ») : viderAuFocus ne vide aucun autre "
    "texte", poseurs == ["api_web/commun.py"] and open(os.path.join(_API, "commun.py"), encoding="utf-8").read().count(
        "data-vide") == 1, poseurs)
if shutil.which("node"):
    ec = juger_vider_au_clic()
    chk("lot B · champ H vidé AU CLIC (« 1 100 » en gris) ; quitté sans rien taper : il reprend sa valeur, rien ne part ; "
        "« 1300 » part ; EFFACÉ (Retour arrière, Suppr, tapé puis effacé) : il part vide → 1 100 mm par défaut ; la pente "
        "garde la règle des nombres ; un texte libre n'est pas vidé ; H déjà vide n'est pas touché", not ec, ec)
    for nom, avant, apres in [
            ("H hors des champs vidés au clic (le sélecteur d'avant)", ", '\n  + 'input[data-sorte=\"texte\"][data-vide]:not([readonly]):not(:disabled)'", "'"),
            ("tout texte vidé au clic (data-vide oublié)", "input[data-sorte=\"texte\"][data-vide]", "input[data-sorte=\"texte\"]"),
            ("le vide voulu repris en sortie (data-vider ignoré)",
             "if (el.value.trim() === \"\" && el.dataset.vider === undefined) el.value", "if (el.value.trim() === \"\") el.value"),
            ("Retour arrière / Suppr sans effet", "(ev.key === \"Backspace\" || ev.key === \"Delete\") && ", "false && "),
            # la voie des claviers de téléphone (un clavier virtuel n'envoie pas toujours Retour arrière dans un champ vide) :
            # taper puis effacer passe par l'événement input
            ("« taper puis effacer » sans effet (la voie des claviers de téléphone)",
             "if (el.value === \"\") videVoulu(el);", "if (false) videVoulu(el);")]:
        chk(f"mutation « {nom} » : le contrôle rougit", juger_vider_au_clic(_commun_mute(avant, apres)) != [])
else:
    print("--   node absent : le champ H joué dans la page n'est pas contrôlé ici (le parcours du navigateur le fait)")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
