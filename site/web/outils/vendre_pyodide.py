# -*- coding: utf-8 -*-
"""Le site fait tourner le moteur Python DANS le navigateur, avec Pyodide. Ce script dépose dans
site/web/vendor/pyodide-<version>/ les fichiers dont le site a besoin, pour qu'il ne dépende d'aucun serveur tiers :

- le cœur de Pyodide, tiré du paquet npm OFFICIEL ``pyodide@<version>`` (npm vérifie son intégrité) ;
- la roue numpy de cette version de Pyodide. Elle n'est pas dans le paquet npm officiel : on la prend sur le CDN
  officiel de Pyodide quand il répond, sinon dans un paquet npm qui la republie (``@mayflowergmbh/wasmsh-pyodide``) —
  et, dans les deux cas, on n'accepte le fichier que si son empreinte SHA-256 est EXACTEMENT celle que le fichier
  ``pyodide-lock.json`` officiel inscrit pour numpy. Pyodide refait d'ailleurs ce contrôle à chaque chargement.

Les empreintes de tous les fichiers déposés sont écrites dans ``empreintes.json`` ; tests/test_site_web.py les relit.

    python3 site/web/outils/vendre_pyodide.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

VERSION = "0.29.5"
ICI = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(ICI, "..", "vendor", f"pyodide-{VERSION}")
COEUR = ["pyodide.mjs", "pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
REPUBLIE = ("@mayflowergmbh/wasmsh-pyodide@0.9.0", "package/assets/")


def sha256(chemin: str) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def npm_pack(spec: str, dossier: str) -> str:
    """Télécharge un paquet npm (sans l'installer) et le déballe ; rend le dossier déballé."""
    sortie = subprocess.run(["npm", "pack", spec, "--silent"], cwd=dossier, check=True, capture_output=True, text=True)
    archive = os.path.join(dossier, sortie.stdout.strip().splitlines()[-1])
    cible = os.path.join(dossier, os.path.basename(archive) + ".d")
    with tarfile.open(archive) as t:
        t.extractall(cible, filter="data")
    return cible


def main():
    os.makedirs(DEST, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        officiel = npm_pack(f"pyodide@{VERSION}", tmp)
        for f in COEUR:
            shutil.copy2(os.path.join(officiel, "package", f), os.path.join(DEST, f))
        lock = json.load(open(os.path.join(DEST, "pyodide-lock.json"), encoding="utf-8"))
        numpy = lock["packages"]["numpy"]
        roue, attendu = numpy["file_name"], numpy["sha256"]
        assert not numpy.get("depends"), "numpy dépend d'autres paquets : les déposer aussi"
        chemin = os.path.join(DEST, roue)
        source = f"https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/{roue}"
        try:
            urllib.request.urlretrieve(source, chemin)
        except Exception:
            source = f"npm {REPUBLIE[0]}"
            rep = npm_pack(REPUBLIE[0], tmp)
            shutil.copy2(os.path.join(rep, REPUBLIE[1], roue), chemin)
        obtenu = sha256(chemin)
        if obtenu != attendu:
            os.remove(chemin)
            sys.exit(f"REFUSÉ : {roue} venu de {source} a l'empreinte {obtenu}, le pyodide-lock officiel dit {attendu}")
    empreintes = {f: sha256(os.path.join(DEST, f)) for f in sorted(os.listdir(DEST)) if f not in ("empreintes.json", "PROVENANCE.md")}
    json.dump({"version": VERSION, "numpy": {"fichier": roue, "source": source, "sha256_lock_officiel": attendu},
               "fichiers": empreintes}, open(os.path.join(DEST, "empreintes.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"Pyodide {VERSION} déposé dans {os.path.relpath(DEST)} ; numpy : {roue} ({source}) — empreinte conforme au lock officiel")


if __name__ == "__main__":
    main()
