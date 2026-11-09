# -*- coding: utf-8 -*-
"""Le SUIVI des dépendances d'une suite de tests, chargé par Python au démarrage (``sitecustomize``) quand
``lancer_tests.py`` le met sur ``PYTHONPATH`` et pose ``STAB_SUIVI`` (le fichier où écrire) et ``STAB_SUIVI_RACINE``
(le dépôt). Sans ces variables, ce module ne fait RIEN : il n'a aucun effet sur un Python ordinaire.

Ce qu'il note, puis écrit en UNE ligne JSON à la fin du processus (les fils Python héritent de l'environnement et
écrivent la leur) :
- les MODULES du dépôt chargés (``sys.modules`` à la sortie : imports directs, différés, par chaîne) ;
- les FICHIERS du dépôt ouverts en lecture (événement d'audit ``open`` : json, html, svg, zip, gels, docs…) ;
- les DOSSIERS du dépôt listés (``os.listdir``, ``os.scandir`` — donc ``glob``, ``os.walk``, ``pathlib``) : un
  fichier ajouté ou retiré y change la liste ;
- les SOUS-PROCESSUS lancés (``subprocess.Popen``) : le lanceur vérifie qu'un fils Python a bien écrit sa ligne ; un
  fils qu'il ne peut pas suivre (``python -I``, ``node``, ``pdftotext``…) rend la suite « non suivie », donc toujours
  rejouée.
Ce qui n'est pas noté est traité comme inconnu par le lanceur, jamais comme « sans changement » : un doute rejoue.
Bibliothèque standard seule ; aucune exception ne sort d'ici (un suivi qui casse une suite serait pire que pas de suivi).
"""
import os
import sys

_FICHIER = os.environ.get("STAB_SUIVI")
_RACINE = os.environ.get("STAB_SUIVI_RACINE")

if _FICHIER and _RACINE:
    _lectures = []
    _dossiers = []
    _enfants = []
    _O_ECRITURE = getattr(os, "O_WRONLY", 1) | getattr(os, "O_RDWR", 2) | getattr(os, "O_CREAT", 64) | getattr(os, "O_TRUNC", 512) | getattr(os, "O_APPEND", 1024)

    def _chemin(x):
        if isinstance(x, bytes):
            try:
                return x.decode(sys.getfilesystemencoding(), "surrogateescape")
            except Exception:
                return None
        if isinstance(x, str):
            return x
        try:
            return os.fspath(x) if not isinstance(x, int) else None
        except Exception:
            return None

    def _depuis_importlib():
        """Vrai si l'appel vient du mécanisme d'import (frames « <frozen importlib… ») et non du code de la suite."""
        try:
            f = sys._getframe(2)
            for _ in range(6):
                if f is None:
                    return False
                nom = f.f_code.co_filename
                if "importlib" in nom or nom.startswith("<frozen zipimport"):
                    return True
                f = f.f_back
        except Exception:
            return False
        return False

    def _audit(evenement, args):
        try:
            if evenement == "open":
                chemin, mode, drapeaux = args[0], args[1], args[2]
                if mode is None:
                    ecriture = bool((drapeaux or 0) & _O_ECRITURE)
                else:
                    ecriture = any(k in str(mode) for k in ("w", "a", "x", "+"))
                if not ecriture:
                    c = _chemin(chemin)
                    if c:
                        _lectures.append(c)
            elif evenement in ("os.listdir", "os.scandir"):
                # (le mécanisme d'import liste les dossiers de sys.path pour trouver un module : ce n'est pas une
                # lecture de la suite — ajouter un test ne doit pas rejouer toutes les suites)
                if _depuis_importlib():
                    return
                c = _chemin(args[0]) if args and args[0] is not None else "."
                if c is not None:
                    _dossiers.append(c)
            elif evenement == "subprocess.Popen":
                a = args[1]
                if isinstance(a, (list, tuple)):
                    _enfants.append([str(x) for x in a])
                else:
                    _enfants.append([str(a)])
        except Exception:
            pass

    def _relatif(chemin, cwd):
        """Le chemin rapporté au dépôt, ou None s'il en sort (ou s'il est du bruit : __pycache__, .git…)."""
        try:
            absolu = os.path.realpath(chemin if os.path.isabs(chemin) else os.path.join(cwd, chemin))
        except Exception:
            return None
        racine = os.path.realpath(_RACINE)
        if absolu != racine and not absolu.startswith(racine + os.sep):
            return None
        rel = os.path.relpath(absolu, racine)
        if rel == ".":
            return "."
        parts = rel.split(os.sep)
        if any(p in ("__pycache__", ".git", "node_modules", ".temoin_tests") for p in parts):
            return None
        return rel.replace(os.sep, "/")

    def _source(rel):
        """Un .pyc compilé → son .py (c'est la source qui change)."""
        if rel.endswith(".pyc") and "__pycache__" in rel:
            try:
                import importlib.util
                return _relatif(importlib.util.source_from_cache(os.path.join(_RACINE, rel)), _RACINE)
            except Exception:
                return None
        return rel

    def _ecrire():
        try:
            import json
            cwd = os.getcwd()
            fichiers = set()
            for m in list(sys.modules.values()):
                f = getattr(m, "__file__", None)
                if f:
                    r = _relatif(f, cwd)
                    if r and r != ".":
                        fichiers.add(r)
            for c in _lectures:
                r = _relatif(c, cwd)
                if r and r != ".":
                    r = _source(r)
                    if r:
                        fichiers.add(r)
            dossiers = set()
            for c in _dossiers:
                r = _relatif(c, cwd)
                if r is not None:
                    dossiers.add(r)
            ligne = {"pid": os.getpid(), "ppid": os.getppid(), "argv": list(sys.argv), "python": sys.version.split()[0],
                     "fichiers": sorted(fichiers), "dossiers": sorted(dossiers), "enfants": _enfants}
            texte = json.dumps(ligne, ensure_ascii=False) + "\n"
            with open(_FICHIER, "a", encoding="utf-8") as fh:
                try:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
                fh.write(texte)
        except Exception:
            pass

    try:
        sys.addaudithook(_audit)
        import atexit
        atexit.register(_ecrire)
    except Exception:
        pass
