#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lance les suites de tests du dépôt (des scripts ``tests/test_*.py`` : chacune imprime son compte-rendu et rend un
code de sortie), EN PARALLÈLE, et ne rejoue que celles dont une DÉPENDANCE a changé depuis leur dernier vert.

    python3 lancer_tests.py                 # les suites à rejouer (dépendance changée, jamais vue, non suivie)
    python3 lancer_tests.py assemblages     # celles dont le nom contient « assemblages »
    python3 lancer_tests.py --tout          # toutes, sans regarder le témoin
    python3 lancer_tests.py -j 1            # une à la fois (défaut : un processus par cœur)
    python3 lancer_tests.py --pourquoi      # dit, pour chaque suite, pourquoi elle est rejouée ou non

Le TÉMOIN (dossier ``.temoin_tests/``, jamais versionné) garde, pour chaque suite VERTE, l'empreinte de ce qu'elle a
lu pendant qu'elle tournait : modules du dépôt importés, fichiers ouverts en lecture, dossiers listés, sous-processus
(le suivi est ``outils/suivi_tests/sitecustomize.py``, posé sur PYTHONPATH le temps de la suite). À la prochaine
demande, une suite dont AUCUNE de ces empreintes n'a bougé est dite INCHANGÉE et n'est pas rejouée : son vert
d'alors vaut toujours. Tout doute rejoue : une suite jamais vue, une suite rouge, un Python différent, un
sous-processus que le suivi n'a pas pu lire (« python -I », node, pdftotext…), une dépendance disparue.
La couverture ne baisse pas : ce qui a changé est rejoué, ce qui n'a pas changé a déjà été prouvé sur ces octets-là.

Code de sortie : 0 si toutes les suites jouées sont vertes, 1 sinon.
"""
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

VERSION_TEMOIN = 1
TEMOIN = ".temoin_tests"
SUIVI = os.path.join("outils", "suivi_tests")


# ------------------------------------------------------------------ les arguments
def lire_arguments(argv):
    """(filtre, tout, pourquoi, n_parallele, racine)."""
    filtre, tout, pourquoi, n, racine = "", False, False, None, None
    it = iter(argv)
    for a in it:
        if a == "--tout":
            tout = True
        elif a == "--pourquoi":
            pourquoi = True
        elif a == "-j":
            n = int(next(it))
        elif a.startswith("-j"):
            n = int(a[2:])
        elif a == "--racine":
            racine = next(it)
        elif a.startswith("-"):
            sys.exit(f"argument inconnu : {a}")
        else:
            filtre = a
    return filtre, tout, pourquoi, n or max(1, os.cpu_count() or 1), racine


# ------------------------------------------------------------------ les empreintes
def empreinte_fichier(chemin):
    try:
        with open(chemin, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None                      # disparu : une dépendance qui manque est un changement


def listing(chemin):
    """Les noms d'un dossier, sans le bruit du cache Python (créé ou effacé au gré des exécutions)."""
    try:
        return sorted(n for n in os.listdir(chemin) if n != "__pycache__" and not n.endswith(".pyc"))
    except OSError:
        return None


def chemin_temoin(racine, suite):
    return os.path.join(racine, TEMOIN, os.path.basename(suite) + ".json")


def lire_temoin(racine, suite):
    try:
        with open(chemin_temoin(racine, suite), encoding="utf-8") as fh:
            t = json.load(fh)
        return t if t.get("version") == VERSION_TEMOIN else None
    except (OSError, ValueError):
        return None


def raison_de_rejouer(racine, suite, temoin):
    """None si la suite est inchangée ; sinon pourquoi on la rejoue."""
    if temoin is None:
        return "jamais vue verte"
    if temoin.get("python") != sys.version.split()[0]:
        return f"Python {temoin.get('python')} → {sys.version.split()[0]}"
    if not temoin.get("suivie", False):
        return "non suivie : " + (temoin.get("raison") or "sous-processus")
    for rel, h in temoin["fichiers"].items():
        if empreinte_fichier(os.path.join(racine, rel)) != h:
            return f"{rel} a changé"
    for rel, noms in temoin["dossiers"].items():
        if listing(os.path.join(racine, rel if rel != "." else "")) != noms:
            return f"le dossier {rel}/ a changé"
    return None


# ------------------------------------------------------------------ le suivi d'une suite
def _est_python(commande):
    return os.path.basename(commande).lower().startswith("python")


def depouiller_suivi(racine, suite, fichier_suivi):
    """Les empreintes de la suite depuis ce que le suivi a écrit : (suivie, raison, fichiers, dossiers)."""
    lignes = []
    try:
        with open(fichier_suivi, encoding="utf-8") as fh:
            for l in fh:
                l = l.strip()
                if l:
                    try:
                        lignes.append(json.loads(l))
                    except ValueError:
                        pass
    except OSError:
        pass
    principale = [l for l in lignes if l.get("argv") and os.path.normpath(l["argv"][0]) == os.path.normpath(suite)]
    if not principale:
        return False, "la suite n'a pas écrit son suivi", {}, {}
    suivie, raison = True, ""
    fichiers, dossiers = set(), set()
    for l in lignes:
        fichiers.update(l.get("fichiers", []))
        dossiers.update(l.get("dossiers", []))
    for l in lignes:
        python_lances = 0
        for enfant in l.get("enfants", []):
            if not enfant:
                continue
            if not _est_python(enfant[0]):
                suivie, raison = False, f"sous-processus {os.path.basename(enfant[0])}"
                break
            python_lances += 1
        if not suivie:
            break
        # chaque fils Python a dû écrire sa propre ligne (son ppid est ce processus) ; « -I », « -E », « -S » et un
        # environnement refait sans PYTHONPATH l'en empêchent → non suivie
        ecrits = sum(1 for m in lignes if m.get("ppid") == l.get("pid"))
        if ecrits < python_lances:
            suivie, raison = False, f"{python_lances - ecrits} fils Python sans suivi (-I, -E ou environnement refait)"
            break
    fichiers.add(os.path.relpath(suite, racine).replace(os.sep, "/"))
    emp_f = {rel: empreinte_fichier(os.path.join(racine, rel)) for rel in sorted(fichiers)}
    emp_d = {rel: listing(os.path.join(racine, rel if rel != "." else "")) for rel in sorted(dossiers)}
    return suivie, raison, emp_f, emp_d


def jouer(racine, suite):
    """Joue une suite dans un sous-processus suivi ; rend (etat, ok, ko, duree, sortie, erreurs, temoin)."""
    os.makedirs(os.path.join(racine, TEMOIN), exist_ok=True)
    fichier_suivi = os.path.join(racine, TEMOIN, "_suivi_" + os.path.basename(suite) + ".jsonl")
    try:
        os.remove(fichier_suivi)
    except OSError:
        pass
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(racine, SUIVI) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["STAB_SUIVI"] = fichier_suivi
    env["STAB_SUIVI_RACINE"] = racine
    t0 = time.time()
    p = subprocess.run([sys.executable, suite], capture_output=True, text=True, cwd=racine, env=env)
    duree = time.time() - t0
    m = re.findall(r"RÉSULTAT\s*(?:FINAL)?\s*:\s*(\d+) OK, (\d+) échec", p.stdout)
    ok, ko = (int(m[-1][0]), int(m[-1][1])) if m else (0, 1)
    ignore = "ignoré" in p.stdout and not m
    etat = "IGNORÉ" if ignore else ("VERT" if p.returncode == 0 and ko == 0 else "ROUGE")
    temoin = None
    if etat == "VERT":
        suivie, raison, fichiers, dossiers = depouiller_suivi(racine, suite, fichier_suivi)
        temoin = {"version": VERSION_TEMOIN, "python": sys.version.split()[0], "quand": datetime.now().isoformat(timespec="seconds"),
                  "duree": round(duree, 1), "ok": ok, "suivie": suivie, "raison": raison, "fichiers": fichiers, "dossiers": dossiers}
    try:
        os.remove(fichier_suivi)
    except OSError:
        pass
    return etat, ok, ko, duree, p.stdout, p.stderr, temoin


def ecrire_temoin(racine, suite, temoin):
    chemin = chemin_temoin(racine, suite)
    if temoin is None:
        try:
            os.remove(chemin)               # rouge : le vert d'avant ne vaut plus
        except OSError:
            pass
        return
    with open(chemin, "w", encoding="utf-8") as fh:
        json.dump(temoin, fh, ensure_ascii=False, indent=0)


# ------------------------------------------------------------------ le lanceur
def principal(argv):
    filtre, tout, pourquoi, n_par, racine = lire_arguments(argv)
    racine = os.path.abspath(racine or os.path.dirname(os.path.abspath(__file__)))
    suites = sorted(f for f in glob.glob(os.path.join(racine, "tests", "test_*.py")) if filtre in os.path.basename(f))
    suites = [os.path.relpath(f, racine) for f in suites]
    temoins = {f: lire_temoin(racine, f) for f in suites}
    a_jouer, inchangees = [], []
    for f in suites:
        raison = "demandé (--tout)" if tout else raison_de_rejouer(racine, f, temoins[f])
        if raison is None:
            inchangees.append(f)
        else:
            a_jouer.append((f, raison))
    # les plus longues d'abord (meilleur remplissage des cœurs) ; les inconnues en tête
    a_jouer.sort(key=lambda fr: -(temoins[fr[0]] or {}).get("duree", 1e9))
    resume = []
    for f in inchangees:
        t = temoins[f]
        quand = datetime.fromisoformat(t["quand"]).strftime("%d/%m %H:%M")
        print(f"{'INCHANGÉ':7s} {f:45s} {t['ok']:4d} OK   0 échec(s)  {t['duree']:6.1f} s  (vert le {quand}, rien n'a bougé)")
        resume.append((f, t["ok"], 0, "INCHANGÉ", 0.0))
    debut = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_par)) as pool:
        futurs = {pool.submit(jouer, racine, f): (f, raison) for f, raison in a_jouer}
        for fut in concurrent.futures.as_completed(futurs):
            f, raison = futurs[fut]
            etat, ok, ko, duree, sortie, erreurs, temoin = fut.result()
            ecrire_temoin(racine, f, temoin)
            resume.append((f, ok, ko, etat, duree))
            suivi = "" if temoin is None or temoin["suivie"] else "  · non suivie, toujours rejouée"
            print(f"{etat:7s} {f:45s} {ok:4d} OK {ko:3d} échec(s)  {duree:6.1f} s" + (f"  ← {raison}" if pourquoi else "") + suivi, flush=True)
            if etat == "ROUGE":
                print("        " + (sortie.strip().splitlines() or [""])[-1][:160])
                for l in erreurs.strip().splitlines()[-3:]:
                    print("        " + l[:160])
    tot_ok = sum(r[1] for r in resume)
    tot_ko = sum(r[2] for r in resume)
    rouges = [r[0] for r in resume if r[3] == "ROUGE"]
    print(f"\nTOTAL : {tot_ok} OK, {tot_ko} échec(s) — {len(suites) - len(rouges)}/{len(suites)} suites vertes "
          f"({len(a_jouer)} jouée{'s' if len(a_jouer) > 1 else ''} en {time.time() - debut:.0f} s sur {n_par} processus, "
          f"{len(inchangees)} inchangée{'s' if len(inchangees) > 1 else ''})")
    return 1 if rouges else 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
