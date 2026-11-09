# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Soudures » sur le site (api_web/soudures.telecharger → api_web/note_cas.py), vérifiée en
Python ordinaire.

- TOUTE la note en un PDF : deux pages par cas, dans l'ordre de la liste ; le texte de chaque page (pdftotext) est
  celui de la note de ce cas faite SEULE par le générateur validé (acier/soudures/note.py), et la taille retenue est
  la sienne — ce qui lie les deux copies nommées de note_cas ;
- PARITÉ avec l'écran du site, cas par cas : le calcul que la note refait (``Calcul``) rend le statut, le taux, le
  point gouvernant et le taux de chaque cordon du moteur de l'écran ; le verdict, le taux de chaque cordon, la gorge,
  le profilé soudé, les efforts saisis, les efforts réduits au point de référence (même valeur, une décimale de moins
  dans la note) et les formules du point gouvernant (fenêtre du cordon gouvernant) s'écrivent dans les pages du cas.
  Seul écart de VERDICT, mesuré et rapporté : entre 85 et 95 %, l'écran dit « Conforme » (le code couleur du bureau, écart
  voulu de l'écran : docs/acier/SOUDURES_SITE.md), la note « CONFORME · ATTENTION » (le seuil du moteur des soudures,
  ``etat.SEUIL_ATTENTION`` = 0,85, celui de l'écran Streamlit). Ce que l'écran montre EN PLUS, par construction de la note validée (« sans la comparaison des méthodes ») : les
  quatre lectures indicatives (EC3 simplifiée, ELS bureau, k Eurocode) et les formules au point le plus chargé des
  cordons qui ne gouvernent pas ;
- l'identification : TOUJOURS celle de la note du site (nom, chantier, indice, date), avec la correspondance de la note
  de la poutre ; celle que porte l'état (un fichier Streamlit) ne s'imprime pas, et l'écran ne la montre plus
  (09/11/2026 : avant, le champ « Indice » de l'écran, « 0 » par défaut, passait devant — « indice 0 » sous une note B) ;
- le routeur la rend en base 64, nommée d'après la note ; un autre fichier est refusé ; une note qui déborde est
  refusée, jamais tronquée ;
- MUTATIONS (chacune doit faire rougir un contrôle) : un cas oublié, des infos mal transmises, la page 2 sans l'échelle
  ni les points de la page 1, la taille imposée ;
- comme dans le navigateur : la note se fait sans fontTools ni accélérateurs C, avec le Pillow factice et les polices
  du paquet ; son texte est celui de CPython ; les fichiers du dépôt qu'elle charge en plus de l'écran sont ceux que
  ``soudures.NOTE.fichiers`` déclare.

    python3 tests/test_note_soudures.py
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
from acier.soudures import configurations as CF  # noqa: E402
from acier.soudures import criteres as K  # noqa: E402
from acier.soudures import etat as ET  # noqa: E402
from acier.soudures import note as NT  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import note_cas as NC  # noqa: E402
from api_web import routeur as R  # noqa: E402
from api_web import soudures as S  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def faire(*saisies, base=None):
    """Une note construite par les saisies de l'écran (sur le cas OUVERT) ; ``"+"`` ajoute un cas."""
    e = copy.deepcopy(base) if base else S.normaliser(S.nouvel_etat())
    for ch, v in saisies:
        if ch == "+":
            S.gerer_elements(e, "ajouter")
            continue
        m = S.modifier(e, ch, v)
        assert m is None, (ch, v, m)
    return S.normaliser(e)


ETATS = {
    "neuve (console C3, Eurocode)": faire(),
    "trois cas : IPE 300 sur platine (ELS), cordons libres C8, plats C1 NON CONFORMES": faire(
        (["nom"], "Poteau P1"), (["config"], "C5"), (["params", "profil"], "IPE 300"), (["methode", "mode"], K.ELS),
        (["efforts", "My"], 40), (["efforts", "Vz"], 80), ("+", None), (["nom"], "Gousset"), (["config"], "C8"), (["efforts", "Vy"], 30),
        ("+", None), (["nom"], "Éclisse"), (["config"], "C1"), (["efforts", "N"], 400)),
    "tube C6 en pénétration partielle, identification saisie": faire(
        (["config"], "C6"), (["params", "jonction"], "partielle"), (["efforts", "N"], 100),
        (["identification", "projet"], "Hall B"), (["identification", "partie"], "Portiques"), (["identification", "indice"], "C")),
    "C2 pleine pénétration (NON CONFORME), puis C7 bout à bout et C4 plat": faire(
        (["config"], "C2"), (["params", "jonction"], "pleine"), (["efforts", "N"], 50), ("+", None), (["config"], "C7"),
        (["efforts", "N"], 200), ("+", None), (["config"], "C4"), (["efforts", "V"], 50)),
}
INFOS = {"nom": "Assemblages soudés", "chantier": "Usine Nord", "indice": "B", "date": "01/02/2026"}
STATUT_ECRAN = {ET.OK: "p", ET.ATTENTION: "f", ET.NOK: "r"}


def nombre(t: str) -> float:
    return float(t.replace("−", "-").replace(" ", "").replace(" ", "").replace(" ", "").replace(",", "."))


# 1. la note de toute la note : un PDF, deux pages par cas, chaque page = celle du cas seul
NOTES = {}
for nom, e in ETATS.items():
    f, octets, ks = O.note_et_tailles(S, e, INFOS)
    NOTES[nom] = octets
    n = len(e["cas"])
    chk(f"{nom} : un PDF de {2 * n} pages ({n} cas), nommé d'après la note",
        octets[:5] == b"%PDF-" and len(O.pages(octets)) == 2 * n and f["nom"] == "Note de calcul - Assemblages soudés.pdf",
        (octets[:8], len(O.pages(octets)), f["nom"]))
    ecarts = O.controle_identite(S, e, INFOS, octets, ks)
    chk(f"{nom} : chaque page est celle du cas fait seul par le générateur, dans sa taille", not ecarts, ecarts[:3])
nok = [(nom, c["nom"]) for nom, e in ETATS.items() for c in e["cas"] if ET.verifier(c)["statut"] == ET.NOK]
chk(f"les états d'essai comptent des cas NON CONFORMES ({len(nok)}) et huit configurations différentes",
    len(nok) >= 2 and len({c["config"] for e in ETATS.values() for c in e["cas"]}) >= 7, nok)

# 2. la parité avec l'écran du site
ecarts, statuts, comparaisons, n_cas, n_fx, n_ef = [], [], 0, 0, 0, 0
for nom, e in ETATS.items():
    tx, brut = O.pages(NOTES[nom]), O.pages(NOTES[nom], brut=True)
    for i, c in enumerate(e["cas"]):
        n_cas += 1
        ei = copy.deepcopy(e)
        ei["actif"] = i
        r = S.resultat(ei)
        site = ET.verifier(ei["cas"][i])
        note = NT.Calcul(ei, ei["cas"][i]).res
        cle = lambda x: (x["statut"], x["taux_max"], (x.get("gouvernant") or {}).get("cordon"),  # noqa: E731
                         (x.get("gouvernant") or {}).get("position"), [(ch["nom"], ch["taux"]) for ch in x.get("chaines", [])])
        if cle(site) != cle(note):
            ecarts.append((nom, i, "calcul", cle(site)[:4], cle(note)[:4]))
        p1, p12 = O.norm(tx[2 * i]), O.norm(tx[2 * i] + " " + tx[2 * i + 1])
        # le verdict, le statut, le point gouvernant
        v = O.verdict(r["resultat"])
        g = site["gouvernant"]
        ch_de = S._chaine_de(site)
        if (v is None or v[0] != C.niveau(site["taux_max"]) or v[1] != O.pct_note(site["taux_max"])
                or f"η max {O.pct_note(site['taux_max'])}" not in p1 or NT.STATUTS[site["statut"]][0] not in p1
                or f"gouverne : {ch_de.get(g['cordon'], g['cordon'])}, {g['position']}" not in p1):
            ecarts.append((nom, i, "verdict", v, O.pct_note(site["taux_max"]), g))
        elif v[0] != STATUT_ECRAN[site["statut"]]:
            # le code couleur du bureau (écran : attention dès 95 %) contre le seuil du moteur des soudures (la note :
            # « CONFORME · ATTENTION » dès 85 %, etat.SEUIL_ATTENTION) — écart voulu de l'écran, docs/acier/SOUDURES_SITE.md
            statuts.append((nom, c.get("nom"), site["taux_max"], v[0], NT.STATUTS[site["statut"]][0]))
        # chaque cordon : son taux, sa gorge ; puis le profilé soudé ; les quatre lectures ne sont pas dans la note
        lignes = O.lignes_resultat(r["resultat"])
        par_nom = {S._maj(ch["nom"]): ch for ch in site["chaines"]}
        for nom_l, piece, niv, txt in lignes:
            ch = par_nom.get(nom_l)
            if ch is not None:
                # la ligne du cordon dans le tableau CORDONS de la note (texte brut : un nom coupé y reste d'un tenant) :
                # « semelle sup. int. droite 5,0 56 432 88 % gauche » — sa gorge (t en pleine pénétration : « cordon (pleine
                # pén.) 10,0 ») et son taux sont ceux de l'écran
                gorge = re.search(r"= (\d+(?:,\d+)?) mm", piece)
                m = re.search(re.escape(ch["nom"]) + r"(?: \(pleine pén\.\))? (\d+,\d) [\d ,]*? (\d+ %)", O.norm(brut[2 * i]))
                if (txt != O.pct_note(ch["taux"]) or not m or m.group(2) != txt
                        or (gorge and abs(nombre(m.group(1)) - nombre(gorge.group(1))) > 1e-9)):
                    ecarts.append((nom, i, nom_l, piece, txt, m and m.groups()))
            elif nom_l == "Sur la platine":
                if piece not in p12:
                    ecarts.append((nom, i, "profilé", piece))
            elif "à titre indicatif" in piece or "méthode choisie" in piece:
                comparaisons += 1
            else:
                ecarts.append((nom, i, "ligne inconnue", nom_l, piece))
        # les efforts SAISIS (champs de l'écran) = la ligne « Efforts — » de la note
        cfg = ei["cas"][i]["config"]
        champs = dict(re.findall(r'<input data-chemin="efforts/(\w+)"[^>]*value="([^"]*)"', r["donnees"]))
        b1 = O.norm(brut[2 * i])
        ligne = re.search(r"Efforts — (.*?) — efforts de", b1)
        for k, lib, unite, _ in CF.EFFORTS[cfg]:
            n_ef += 1
            sym = lib.split(" (")[0]
            m = re.search(re.escape(sym) + r" = (-?[\d ]+,\d) " + re.escape(unite), ligne.group(1) if ligne else "")
            if not m or abs(nombre(m.group(1)) - nombre(champs.get(k) or "0")) > 0.05 + 1e-9:
                ecarts.append((nom, i, "effort saisi", k, champs.get(k), m and m.group(1)))
        # les efforts RÉDUITS au point de référence : la fenêtre du cordon (deux décimales) = la note (une décimale)
        ef = re.search(r"V_x (\S+) kN, V_y (\S+) kN, N (\S+) kN, M_x (\S+), M_y (\S+), M_z (\S+) kN·m", b1)
        gi = next(k for k, ch in enumerate(site["chaines"]) if ch["nom"] == ch_de.get(g["cordon"]))
        fen = r["volets"].split(f'id="ck-c{gi}"')[1].split('<div class="flyout')[0]
        vals = dict((fx.split("|")[1].strip(), res) for fx, res in O.resultats_fenetres(fen))
        for j, sym in enumerate(("Vx", "Vy", "N", "Mx", "My", "Mz")):
            n_ef += 1
            if not ef or sym not in vals or abs(nombre(vals[sym].split()[0]) - nombre(ef.group(j + 1))) > (0.05 if j < 3 else 0.005) + 1e-9:
                ecarts.append((nom, i, "effort réduit", sym, vals.get(sym), ef and ef.group(j + 1)))
        # les formules du point gouvernant (la fenêtre du cordon gouvernant, après les efforts et le groupe)
        point = fen.split("Au point le plus chargé")[1]
        for fx, res in O.resultats_fenetres(point):
            n_fx += 1
            if not O.present(res, p12):
                ecarts.append((nom, i, "formule", res, fx[:80]))
chk(f"parité : {n_cas} cas, {n_ef} efforts (saisis et réduits), {n_fx} résultats des formules du point gouvernant — la note écrit "
    "le verdict, les taux des cordons, les gorges, le profilé, les efforts et les formules de l'écran", not ecarts and n_cas >= 8 and n_fx >= 60,
    ecarts[:4])
chk("le seul écart de verdict mesuré : entre 85 et 95 %, l'écran dit « Conforme » (code du bureau), la note « CONFORME · ATTENTION » "
    "(seuil du moteur des soudures)", len(statuts) >= 1 and all(ET.SEUIL_ATTENTION < x[2] < 0.95 and x[3] == "p" and x[4] == "CONFORME · ATTENTION"
                                                               for x in statuts), statuts)
for x in statuts:
    print(f"--   écart de verdict rapporté : {x[1]} à {x[2] * 100:.2f} % — écran « Conforme », note « {x[4]} »")
chk(f"ce que l'écran montre en plus, par construction de la note : les lectures indicatives des autres méthodes ({comparaisons})",
    comparaisons >= 3 * n_cas - 3, comparaisons)

# 3. l'identification
e = ETATS["trois cas : IPE 300 sur platine (ELS), cordons libres C8, plats C1 NON CONFORMES"]
# (09/11/2026) l'indice — et le projet, la partie, la date — ont UNE source : la note du site. Avant, le champ « Indice »
# de l'écran (« 0 » par défaut) passait devant : le PDF disait « indice 0 » sous une note « B » (relecture du 09/11).
attendu = lambda i, c: ["Note de calcul · 01/02/2026 · indice B", f"Usine Nord · Assemblages soudés · cas {c['nom']}"]  # noqa: E731
ecarts = O.controle_infos(NOTES["trois cas : IPE 300 sur platine (ELS), cordons libres C8, plats C1 NON CONFORMES"], e, attendu)
chk("le chantier, le nom, l'indice et la date de la note du site sont dans l'en-tête de chaque cas (l'identification de "
    "l'état, « 0 » par défaut, n'y est pas)", not ecarts and "indice 0" not in O.norm(" ".join(O.pages(NOTES[
        "trois cas : IPE 300 sur platine (ELS), cordons libres C8, plats C1 NON CONFORMES"]))), ecarts)
e6 = ETATS["tube C6 en pénétration partielle, identification saisie"]
t6 = O.norm(" ".join(O.pages(NOTES["tube C6 en pénétration partielle, identification saisie"])))
ecarts = O.controle_infos(NOTES["tube C6 en pénétration partielle, identification saisie"], e6, attendu)
chk("l'identification que porte l'état (un fichier Streamlit : « Hall B », « Portiques », indice C) ne s'imprime pas : "
    "celle de la note du site", not ecarts and "Hall B" not in t6 and "Portiques" not in t6 and "indice C" not in t6, ecarts)
chk("infos_du_cas : la note du site seule, avec la correspondance de la poutre",
    NC.infos_du_cas(S.NOTE, e6, e6["cas"][0], INFOS) == {"nom_projet": "Usine Nord", "partie": "Assemblages soudés", "indice": "B", "date": "01/02/2026"}
    and NC.infos_du_cas(S.NOTE, e, e["cas"][0], {"nom": "N"}) == {"nom_projet": "N", "partie": "Soudures", "indice": "A"},
    (NC.infos_du_cas(S.NOTE, e6, e6["cas"][0], INFOS), NC.infos_du_cas(S.NOTE, e, e["cas"][0], {"nom": "N"})))
donnees = S.resultat(copy.deepcopy(e6))["donnees"]
with open(os.path.join(RACINE, "docs", "site", "MODULES_SITE.md"), encoding="utf-8") as fh:
    modules_site = " ".join(fh.read().split())
chk("l'écran ne montre plus de champ d'identification (un seul composant, les trois formats) : il dit d'où vient celle "
    "de la note PDF ; l'écrit suit (MODULES_SITE.md)",
    'data-chemin="identification/' not in donnees and NC.AIDE_IDENTIFICATION in donnees
    and "L'INDICE a une seule source" in modules_site, donnees.count("identification"))

# 4. le routeur, les refus
r = json.loads(R.traiter(json.dumps({"module": "soudures", "action": "telecharger", "etat": e, "quoi": "note_pdf",
                                     "infos": {"nom": "Soudures : P1/P2", "chantier": "C"}})))
chk("le routeur rend la note en base 64 sous un nom de fichier sûr",
    r.get("ok") and r["fichier"]["nom"] == "Note de calcul - Soudures - P1-P2.pdf" and base64.b64decode(r["fichier"]["base64"])[:5] == b"%PDF-",
    r.get("erreur") or r["fichier"]["nom"])
r = json.loads(R.traiter(json.dumps({"module": "soudures", "action": "telecharger", "etat": e, "quoi": "autre"})))
chk("un autre fichier que la note est refusé avec un message", not r["ok"] and "inconnu" in r["erreur"], r)
vrai = NT._essai
NT._essai = lambda calc, doc_meta, k, chemin: vrai(calc, doc_meta, k, chemin) + ["page 2, colonne 22 : 4 pt"]
try:
    r1 = json.loads(R.traiter(json.dumps({"module": "soudures", "action": "telecharger", "etat": ETATS["neuve (console C3, Eurocode)"], "quoi": "note_pdf"})))
    r3 = json.loads(R.traiter(json.dumps({"module": "soudures", "action": "telecharger", "etat": e, "quoi": "note_pdf"})))
finally:
    NT._essai = vrai
chk("une note qui déborde encore en taille serrée est refusée, jamais tronquée (un cas, puis trois)",
    not r1["ok"] and not r3["ok"] and all("ne tient pas" in x["erreur"] and "4 pt" in x["erreur"] for x in (r1, r3))
    and "« Poteau P1 »" in r3["erreur"], (r1.get("erreur"), r3.get("erreur")))

vrai_pages = NC.pages
NC.pages = lambda *a: vrai_pages(*a) + ["page 2, colonne 300 : 2 pt"]
try:
    r3b = json.loads(R.traiter(json.dumps({"module": "soudures", "action": "telecharger", "etat": e, "quoi": "note_pdf"})))
finally:
    NC.pages = vrai_pages
chk("… et le débordement relevé en posant les pages d'un cas dans la note entière (second garde-fou) l'est aussi",
    not r3b["ok"] and "ne tient pas" in r3b["erreur"] and "2 pt" in r3b["erreur"], r3b.get("erreur"))

# 5. les mutations
vrai_pages = NC.pages
compte = []


def sans_dernier(NT_, gen, d, calc, doc_meta, k, reports=()):
    compte.append(1)
    return [] if len(compte) == 3 else vrai_pages(NT_, gen, d, calc, doc_meta, k, reports)


NC.pages = sans_dernier
try:
    mut = NC.pdf(S.NOTE, copy.deepcopy(e), INFOS)
finally:
    NC.pages = vrai_pages
chk("mutation « un cas oublié » : le contrôle d'identité rougit", O.controle_identite(S, e, INFOS, mut) != [])
vrai_infos = NC.infos_du_cas
NC.infos_du_cas = lambda gen, etat, cas, infos: {"indice": (infos or {}).get("indice")}
try:
    mut = NC.pdf(S.NOTE, copy.deepcopy(e), INFOS)
finally:
    NC.infos_du_cas = vrai_infos
chk("mutation « infos mal transmises » : le contrôle de l'identification rougit", O.controle_infos(mut, e, attendu) != [])
vrai_note = S.NOTE
S.NOTE = S.NOTE._replace(suite=lambda n1: (n1.echelle,))
try:
    mut = NC.pdf(S.NOTE, copy.deepcopy(e), INFOS)
finally:
    S.NOTE = vrai_note
chk("mutation « page 2 sans les points montrés en page 1 » : le contrôle d'identité rougit", O.controle_identite(S, e, INFOS, mut) != [])
vrai_var = NC.variante
for k, nom_k in NT.VARIANTES:
    NC.variante = lambda NT_, calc, doc_meta, brouillon, k=k: (k, [])
    try:
        _, mut, ks = O.note_et_tailles(S, e, INFOS)
        rouge = O.controle_identite(S, e, INFOS, mut, ks) != []
    except AssertionError as err:
        rouge, ks = "ne tient pas" in str(err), str(err)[:80]
    finally:
        NC.variante = vrai_var
    if k == NT.VARIANTES[0][0]:
        # les cas d'essai tiennent tous en taille normale : l'imposer ne change rien — la copie n'y est pas mise à l'épreuve
        chk(f"taille imposée « {nom_k} » : les cas d'essai la prennent déjà (rien ne rougit, attendu)", not rouge, ks)
    else:
        chk(f"mutation « toujours la taille {nom_k} » : le contrôle des tailles rougit", rouge, ks)

# 6. comme dans le navigateur
e_nav = list(ETATS.values())
infos_nav = {"nom": "relevé", "chantier": "C", "indice": "B", "date": "01/01/2026"}
s = O.sonde("soudures", e_nav, infos_nav)
chk("navigateur : sans fontTools ni accélérateurs C, avec le Pillow factice et les polices du paquet, la note se fait",
    "erreur" not in s and s["fonttools"] is False and s["rl_accel"] is False and s["pil"].startswith(O.STUB), s.get("erreur") or s)
if "erreur" not in s:
    cpy = [O.pages(base64.b64decode(S.telecharger(copy.deepcopy(x), "note_pdf", infos_nav)["base64"])) for x in e_nav]
    chk("navigateur : le texte de chaque page est celui de la note faite en CPython", [O.pages(o) for o in s["sorties"]] == cpy)
    pol = sorted({p for o in s["sorties"] for p in O.polices(o)})
    noms_paquet = {f[:-4] for f in s["polices_paquet"]}
    chk(f"navigateur : les polices de la note sont celles du paquet (et Helvetica, police standard non embarquée) : {pol}",
        set(pol) <= noms_paquet | {"Helvetica"} and set(s["polices"]) <= set(s["polices_paquet"]), (pol, s["polices"]))
    chk("navigateur : numpy est déjà chargé par l'écran du module (avec_numpy) — la note ne l'ajoute pas", s["numpy_ecran"] is True)
    rel = O.releve_paquet(S, s)
    print(f"--   fichiers du dépôt chargés par la note, hors paquet du moteur et paquet de la poutre : {rel['hors']}")
    chk("navigateur : ces fichiers sont ceux que soudures.NOTE.fichiers déclare (le paquet de la note les reçoit)",
        set(rel["hors"]) <= set(rel["declares"]) and not rel["non_charges"], rel)
manque = O.paquet_integre(S)
if manque is None:
    print("--   paquet de la note : « soudures » n'est pas encore dans construire.NOTE_PDF — contrôle en attente de l'intégration")
else:
    chk("paquet de la note (note.zip) : il porte les fichiers que la note des soudures déclare", not manque, manque)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
