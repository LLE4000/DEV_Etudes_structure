# -*- coding: utf-8 -*-
"""L'IDENTIFICATION imprimée par la note PDF des SIX modules du site qui la font (poutre, dalle, garde-corps, soudures,
charges sur plancher, assemblages) : TOUJOURS celle de la note du site (R2 : une source unique) — son nom, son
chantier, son indice, sa date (``infos`` que la page envoie, js/etabli.js).

Défaut corrigé (09/11/2026) : les notes des soudures et du plancher imprimaient « indice 0 » sous une note du site
d'indice B — le champ « Indice » de leur écran (« 0 » par défaut) passait devant celui de la note. Les champs
« Identification de la note » ont quitté l'écran du site ; l'identification que porte l'état (un fichier Streamlit)
ne s'imprime plus.

Pour chaque module, sa note « riche » (site/web/construire.NOTES_RELEVE, celle du relevé du paquet de la note) est
demandée au ROUTEUR, comme le fait la page, avec une identification PIÉGÉE — accents, guillemets droits et français,
apostrophe, tiret long, « & » — et l'indice « C » ; l'identification de l'état (soudures, plancher) est piégée elle
aussi (« Projet du fichier », indice « Z »). pdftotext doit retrouver, EXACTEMENT :

- poutre, dalle, assemblages : la page de garde — PROJET = le chantier, PARTIE = le nom, DATE, INDICE C ; puis l'en-tête
  de chaque planche ou note (« 03/04/2026 · indice C ») ; pour les assemblages, l'en-tête de chaque note d'assemblage
  porte « chantier · nom · repère » ;
- garde-corps, soudures, plancher : la première page de chaque cas — « Note de calcul · 03/04/2026 · indice C » et
  « chantier · nom · cas … » — puis le cartouche (« indice C », la date) ;
- nulle part « indice 0 », « indice A », « indice Z », ni l'identification de l'état.

MUTATIONS (chacune doit faire rougir le module visé) : l'ancienne traduction des modules à cas (la saisie de l'écran
d'abord), une poutre, une dalle, des assemblages qui impriment « 0 » au lieu de l'indice de la note.

    python3 tests/test_note_identification.py
"""
import base64
import copy
import json
import os
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "site", "web"))

import construire as K  # noqa: E402
from api_web import note_assemblages as NA  # noqa: E402
from api_web import note_cas as NC  # noqa: E402
from api_web import note_dalle as ND  # noqa: E402
from api_web import note_poutre as NP  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


NOM = "Façade « Est » & l'atelier \"Nord\""
CHANTIER = "Résidence d'Été & Cie — « Lot 2 »"
INFOS = {"nom": NOM, "chantier": CHANTIER, "indice": "C", "date": "03/04/2026"}
# l'identification que porte l'état d'un module à cas (un fichier de l'écran Streamlit) : elle ne doit plus s'imprimer
PIEGE_ETAT = {"projet": "Projet du fichier", "partie": "Partie cachée", "indice": "Z"}
INTERDITS = ("indice 0", "indice A", "indice Z", "INDICE 0", "INDICE A", "INDICE Z", "Projet du fichier", "Partie cachée")


def etats():
    """La note riche de chaque module (construire.NOTES_RELEVE), l'identification de l'état piégée."""
    out = {}
    for k in K.NOTE_PDF:
        ns = {}
        exec(K.NOTES_RELEVE[k], ns)
        e = ns["etat"]
        if k == "soudures":
            e["identification"] = dict(PIEGE_ETAT)
        if k == "plancher":
            for c in e["cas"]:
                c["identification"] = dict(PIEGE_ETAT)
        out[k] = e
    return out


def note(module: str, etat: dict):
    r = json.loads(R.traiter(json.dumps({"module": module, "action": "telecharger", "etat": copy.deepcopy(etat),
                                         "quoi": "note_pdf", "infos": INFOS})))
    if not r.get("ok"):
        raise AssertionError(f"{module} : {r.get('erreur')}")
    return base64.b64decode(r["fichier"]["base64"])


def pages(octets: bytes):
    """Le texte de chaque page (pdftotext -raw : une ligne du dessin reste d'un tenant)."""
    p = subprocess.run(["pdftotext", "-raw", "-", "-"], input=octets, capture_output=True, check=True)
    return p.stdout.decode("utf-8").split("\f")[:-1]


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace(" ", " ").replace(" ", " ").replace(" ", " "))


def plat(t: str) -> str:
    """Sans aucun blanc : les titres de la garde sont espacés lettre à lettre (« P R O J E T »)."""
    return re.sub(r"\s+", "", t)


def premieres(tx, debut: str):
    """Les indices des pages qui commencent un élément (leur en-tête porte « Note de calcul · date · indice »)."""
    return [i for i, t in enumerate(tx) if debut in norm(t)]


def ecarts(module: str, octets: bytes, etat: dict) -> list:
    tx = pages(octets)
    out = []
    tout = norm(" ".join(tx))
    out += [f"« {x} » imprimé" for x in INTERDITS if x in tout]
    tete = "Note de calcul · 03/04/2026 · indice C"
    if module in ("poutre", "dalle", "assemblages"):
        g = plat(tx[0])
        for lab, val in (("PROJET", CHANTIER), ("PARTIE", NOM), ("DATE", "03/04/2026"), ("INDICE", "C")):
            if plat(lab + val) not in g:
                out.append(f"garde : {lab} {val} absent")
        suivantes = tx[1:] if module != "assemblages" else [t for t in tx[1:] if "Bureau" in t]
        for i, t in enumerate(suivantes):
            if "03/04/2026 · indice C" not in norm(t):
                out.append(f"page {i + 2} : « 03/04/2026 · indice C » absent de l'en-tête")
        if module == "assemblages":
            debuts = premieres(tx, tete)
            if len(debuts) != len(etat["assemblages"]):
                out.append(f"{len(debuts)} notes d'assemblage pour {len(etat['assemblages'])} assemblages")
            for n, i in enumerate(debuts):
                if f"{CHANTIER} · {NOM} · " not in norm(tx[i]) or "indice C" not in norm(tx[i + 1]):
                    out.append(f"assemblage {n + 1} : « chantier · nom · repère » ou le cartouche absent")
    else:
        debuts = premieres(tx, tete)
        if len(debuts) != len(etat["cas"]):
            out.append(f"{len(debuts)} premières pages pour {len(etat['cas'])} cas")
        for n, i in enumerate(debuts):
            if f"{CHANTIER} · {NOM} · cas " not in norm(tx[i]):
                out.append(f"cas {n + 1} : « chantier · nom · cas » absent de la première page")
            if "indice C" not in norm(tx[i + 1]) or "03/04/2026" not in tx[i + 1]:
                out.append(f"cas {n + 1} : le cartouche ne dit pas « indice C » et la date")
    return out


ETATS = etats()
NOTES = {}
for mod, e in ETATS.items():
    NOTES[mod] = note(mod, e)
    ec = ecarts(mod, NOTES[mod], e)
    chk(f"{mod} : le nom, le chantier (accents, guillemets, « & »), l'indice C et la date de la note du site, exactement aux "
        f"endroits prévus ({len(pages(NOTES[mod]))} pages) ; rien d'autre", not ec, ec[:4])
chk("les six modules qui font leur note sur le site sont contrôlés", set(NOTES) == set(K.NOTE_PDF) and len(NOTES) == 6, sorted(NOTES))

# ---------------------------------------------------------------- MUTATIONS : chacune doit faire rougir son module
vrai_infos = NC.infos_du_cas


def ancienne_traduction(gen, etat, cas, infos):
    """La traduction d'avant le 09/11/2026 : l'identification saisie à l'écran (l'état) passait devant la note."""
    out = vrai_infos(gen, etat, cas, infos)
    saisi = (cas.get("identification") or etat.get("identification") or {})
    for cle, k in (("nom_projet", "projet"), ("partie", "partie"), ("indice", "indice")):
        if str(saisi.get(k) or "").strip():
            out[cle] = str(saisi[k]).strip()
    return out


NC.infos_du_cas = ancienne_traduction
try:
    for mod in ("soudures", "plancher"):
        chk(f"mutation « la saisie de l'écran d'abord » : {mod} rougit", ecarts(mod, note(mod, ETATS[mod]), ETATS[mod]) != [])
finally:
    NC.infos_du_cas = vrai_infos

for mod, M, nom_f in (("poutre", NP, "pdf"), ("dalle", ND, "pdf")):
    vrai = getattr(M, nom_f)
    setattr(M, nom_f, lambda etat, doc=None, _v=vrai: _v(etat, {**(doc or {}), "indice": "0"}))
    try:
        chk(f"mutation « indice 0 » : {mod} rougit", ecarts(mod, note(mod, ETATS[mod]), ETATS[mod]) != [])
    finally:
        setattr(M, nom_f, vrai)
vrai_p = NA.infos_projet
NA.infos_projet = lambda infos: {**vrai_p(infos), "indice": "0"}
try:
    chk("mutation « indice 0 » : assemblages rougit", ecarts("assemblages", note("assemblages", ETATS["assemblages"]), ETATS["assemblages"]) != [])
finally:
    NA.infos_projet = vrai_p
vrai_g = NC.infos_du_cas
NC.infos_du_cas = lambda gen, etat, cas, infos: {**vrai_g(gen, etat, cas, infos), "indice": "A"}
try:
    chk("mutation « indice A » (l'indice de la note ignoré) : garde-corps rougit",
        ecarts("garde_corps", note("garde_corps", ETATS["garde_corps"]), ETATS["garde_corps"]) != [])
finally:
    NC.infos_du_cas = vrai_g

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
