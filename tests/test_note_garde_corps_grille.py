# -*- coding: utf-8 -*-
"""La GRILLE de 240 états du garde-corps et sa note PDF (acier/garde_corps/note.py, le générateur validé, celui de
l'écran Streamlit ET du site).

La grille : 5 types × 3 formes × « Cumuler qh et qv » non/oui × 4 fixations × 2 méthodes, le reste par défaut (les
saisies de l'écran du site, api_web/garde_corps.modifier). C'est celle de la relecture du 09/11/2026 : 86 de ses 240
états débordaient de leurs deux pages même en taille serrée (docs/site/DEFAUTS_STREAMLIT.md).

LA RÉFÉRENCE FIGÉE (``tests/ref_note_garde_corps_grille.json``) : pour chaque état, la note du générateur tel qu'il
était le 09/11/2026 (commit e4c173c), son nombre de pages, l'empreinte SHA-256 du texte de chaque page (pdftotext
-layout) et les débordements qu'il relevait. Elle s'écrit UNE fois, avec le générateur d'avant :

    python3 tests/test_note_garde_corps_grille.py --geler

Ce que la suite prouve :

- (a) les états qui TENAIENT dans leurs deux pages donnent EXACTEMENT la même note qu'avant : même nombre de pages, le
  texte de chaque page identique (empreinte) ;

    python3 tests/test_note_garde_corps_grille.py
"""
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier.garde_corps import note as NT  # noqa: E402
from api_web import garde_corps as G  # noqa: E402

REF = os.path.join(RACINE, "tests", "ref_note_garde_corps_grille.json")
TYPES = ("barreaudage", "lisses", "panneau", "main_courante", "murale")
FORMES = ("droite", "L", "U")
CUMUL = (False, True)
FIXATIONS = ("platine", "laterale", "soudee", "scellee")
METHODES = ("els", "eurocode")
# une identification fixe (la date du jour changerait le texte de la note chaque jour)
INFOS = {"nom_projet": "Grille", "partie": "Garde-corps", "indice": "A", "date": "09/11/2026"}

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def grille():
    """Les 240 états : (clé lisible, état de la note du site)."""
    for t, f, cu, fx, m in itertools.product(TYPES, FORMES, CUMUL, FIXATIONS, METHODES):
        e = G.nouvel_etat()
        for ch, v in ((["type"], t), (["geometrie", "forme"], f), (["charges", "cumul"], cu), (["fixation", "type"], fx),
                      (["methode", "mode"], m)):
            msg = G.modifier(e, ch, v)
            assert msg is None, (t, f, cu, fx, m, ch, msg)
        yield f"{t} · {f} · {'cumul' if cu else 'séparées'} · {fx} · {m}", G.normaliser(e)


def textes(octets: bytes):
    """Le texte de chaque page (pdftotext -layout)."""
    p = subprocess.run(["pdftotext", "-layout", "-", "-"], input=octets, capture_output=True, check=True)
    return p.stdout.decode("utf-8").split("\f")[:-1]


def empreinte(octets: bytes) -> dict:
    tx = textes(octets)
    return {"pages": len(tx), "sha": [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in tx]}


def geler():
    if os.path.exists(REF):
        print(f"{os.path.relpath(REF, RACINE)} existe : la référence ne se réécrit pas (la retirer d'abord, en le disant).")
        return 1
    out = {}
    for cle, e in grille():
        octets = NT.generer_note(e, e["cas"][0], INFOS)
        out[cle] = {**empreinte(octets), "variante": NT.derniere_variante, "debordements": list(NT.derniers_avertissements)}
    with open(REF, "w", encoding="utf-8") as f:
        json.dump({"_": "Note du garde-corps, générateur du 09/11/2026 (commit e4c173c) : pages, empreinte SHA-256 du texte "
                        "de chaque page (pdftotext -layout), variante retenue, débordements relevés — tests/test_note_garde_corps_grille.py",
                   "infos": INFOS, "etats": out}, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{len(out)} états figés, {sum(1 for v in out.values() if v['debordements'])} débordaient")
    return 0


def main():
    with open(REF, encoding="utf-8") as f:
        ref = json.load(f)
    chk("la référence figée porte les 240 états de la grille, avec l'identification fixe de la suite",
        len(ref["etats"]) == 240 and ref["infos"] == INFOS, (len(ref["etats"]), ref["infos"]))
    tenaient = {k for k, v in ref["etats"].items() if not v["debordements"]}
    print(f"--   référence : {len(tenaient)} états tenaient dans leurs deux pages, {240 - len(tenaient)} débordaient")
    t0 = time.time()
    ecarts_a = []
    for cle, e in grille():
        octets = NT.generer_note(e, e["cas"][0], INFOS)
        emp = empreinte(octets)
        r = ref["etats"].get(cle)
        if r is None:
            ecarts_a.append((cle, "absent de la référence"))
            continue
        if cle in tenaient and (emp["pages"] != r["pages"] or emp["sha"] != r["sha"]):
            ecarts_a.append((cle, emp["pages"], r["pages"], [i + 1 for i, (a, b) in enumerate(zip(emp["sha"], r["sha"])) if a != b]))
    print(f"--   {240} notes faites en {time.time() - t0:.0f} s")
    chk(f"(a) les {len(tenaient)} états qui tenaient donnent EXACTEMENT la note d'avant (pages, texte de chaque page)",
        not ecarts_a and len(tenaient) == 154, ecarts_a[:4])

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(geler() if "--geler" in sys.argv else main())
