#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Le PARCOURS COMPLET du navigateur (site/web/verifier_site.js) joué EN MORCEAUX, plusieurs à la fois — le même parcours,
sans un contrôle de moins, en moins de temps (demande du propriétaire, 13/11/2026 : « si on sait faire plein de trucs en
même temps … sans se tromper »).

    python3 outils/parcours_parallele.py http://127.0.0.1:8770 <dossier>          # tout le parcours, 3 à la fois
    python3 outils/parcours_parallele.py http://127.0.0.1:8770 <dossier> -j 2     # 2 à la fois
    python3 outils/parcours_parallele.py … --reference <journal d'un passage complet>   # compare les contrôles

Chaque MORCEAU est un filtre du parcours (« garde », « poutre », « formats »…), joué par son propre processus et son
propre navigateur, exactement comme on le joue à la main (``node site/web/verifier_site.js <base> <captures> <filtre>``).
Les morceaux couvrent TOUT le parcours : la liste MORCEAUX est liée par un test aux sections du fichier (chaque garde
« if (… FILTRE === 'x') » a son morceau), et « demarrage » (verifier_demarrage.js) en est un.

Ce qui garantit qu'on ne perd rien :
- chaque morceau rend son code de sortie ; un seul rouge (ou un « non mesuré », code 3) fait rougir le tout ;
- le COMPTE : la somme des contrôles est écrite, morceau par morceau ;
- la COUVERTURE (``--reference``) : chaque contrôle d'un passage complet de référence (son NOM, sans le détail mesuré)
  doit se retrouver dans l'union des morceaux ; un nom absent est dit, et le passage rougit ;
- l'empreinte de construction : chaque morceau la vérifie déjà (le parcours refuse de conclure si le site a été
  reconstruit sous lui).

Les durées du dernier passage (``.temoin_tests/parcours.json``, jamais versionné) ordonnent les morceaux, les plus longs
d'abord.
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIF = os.path.join(RACINE, "site", "web", "verifier_site.js")
TEMOIN = os.path.join(RACINE, ".temoin_tests", "parcours.json")
# les morceaux : l'union est le passage complet (test : tests/test_parcours_parallele.py)
MORCEAUX = ("formats", "modules", "garde", "poutre", "telecharger", "exporter", "meteo", "sonde", "enregistrement",
            "ouverture", "pilote", "corniere", "plaques", "piege", "bibliotheque", "note_pdf", "bureau", "demarrage")
LIGNE = re.compile(r"^(OK|KO)\s{2}(.*?)(?:\s{2}.*)?$")
NOMBRE = re.compile(r"\d+(?:[.,]\d+)*")


def nom_controle(ligne: str):
    """(« OK » | « KO », le NOM du contrôle) d'une ligne du journal — le détail mesuré, après deux espaces, est retiré."""
    m = LIGNE.match(ligne.rstrip("\n"))
    return (m.group(1), m.group(2).strip()) if m else None


def cle_controle(nom: str) -> str:
    """Le nom d'un contrôle sans ses MESURES : certains noms disent une durée ou un compte (« 18 pages, 3.8 s ») qui varie
    d'un passage à l'autre — la comparaison les neutralise (« # pages, # s »)."""
    return NOMBRE.sub("#", nom)


def lire_journal(texte: str):
    """[(etat, nom)] des contrôles d'un journal."""
    return [x for x in (nom_controle(l) for l in texte.splitlines()) if x]


def jouer(base: str, dossier: str, morceau: str) -> dict:
    cap = os.path.join(dossier, f"cap_{morceau.replace(':', '_')}")
    os.makedirs(cap, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHON", sys.executable)
    t0 = time.time()
    p = subprocess.run(["node", VERIF, base, cap, morceau], cwd=RACINE, env=env, capture_output=True, text=True)
    duree = time.time() - t0
    journal = os.path.join(dossier, f"{morceau.replace(':', '_')}.log")
    with open(journal, "w", encoding="utf-8") as fh:
        fh.write(p.stdout)
        if p.stderr:
            fh.write("\n---- stderr\n" + p.stderr)
    ctl = lire_journal(p.stdout)
    return {"morceau": morceau, "code": p.returncode, "duree": duree, "ok": sum(1 for e, _ in ctl if e == "OK"),
            "ko": [n for e, n in ctl if e == "KO"], "noms": sorted({n for _, n in ctl}), "journal": journal}


def principal(argv) -> int:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("base")
    a.add_argument("dossier")
    a.add_argument("-j", type=int, default=3)
    a.add_argument("--reference", help="le journal d'un passage complet : chacun de ses contrôles doit se retrouver")
    a.add_argument("morceaux", nargs="*", help="(défaut : tous)")
    x = a.parse_args(argv)
    os.makedirs(x.dossier, exist_ok=True)
    morceaux = list(x.morceaux or MORCEAUX)
    try:
        with open(TEMOIN, encoding="utf-8") as fh:
            durees = json.load(fh)
    except (OSError, ValueError):
        durees = {}
    morceaux.sort(key=lambda m: -durees.get(m, 1e9))
    t0 = time.time()
    res = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, x.j)) as ex:
        futs = {ex.submit(jouer, x.base, x.dossier, m): m for m in morceaux}
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            res.append(r)
            etat = "VERT " if r["code"] == 0 else ("NON MESURÉ" if r["code"] == 3 else "ROUGE")
            print(f"{etat:6} {r['morceau']:16} {r['ok']:5} OK  {len(r['ko']):3} KO  {r['duree']:7.1f} s", flush=True)
            for n in r["ko"][:10]:
                print(f"         KO  {n[:160]}", flush=True)
    total = time.time() - t0
    durees.update({r["morceau"]: round(r["duree"], 1) for r in res})
    os.makedirs(os.path.dirname(TEMOIN), exist_ok=True)
    with open(TEMOIN, "w", encoding="utf-8") as fh:
        json.dump(durees, fh, ensure_ascii=False, indent=0)
    ok = sum(r["ok"] for r in res)
    ko = sum(len(r["ko"]) for r in res)
    rouges = [r["morceau"] for r in res if r["code"] != 0]
    manquants = []
    if x.reference:
        with open(x.reference, encoding="utf-8") as fh:
            ref = {cle_controle(n) for _, n in lire_journal(fh.read())}
        vus = {cle_controle(n) for r in res for n in r["noms"]}
        manquants = sorted(ref - vus)
        print(f"\ncouverture : {len(ref) - len(manquants)}/{len(ref)} contrôles du passage de référence retrouvés")
        for n in manquants[:40]:
            print(f"   absent : {n[:160]}")
    somme = sum(r["duree"] for r in res)
    print(f"\nTOTAL : {ok} OK, {ko} KO — {len(res) - len(rouges)}/{len(res)} morceaux verts — {total:.0f} s "
          f"(joués à la suite : {somme:.0f} s ; {x.j} à la fois)")
    return 0 if not rouges and not manquants else 1


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
