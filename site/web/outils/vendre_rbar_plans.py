# -*- coding: utf-8 -*-
"""Le moteur de PLANS de RBAR dans le navigateur (WebAssembly) : ce script dépose dans site/web/vendor/rbar-plans-web-<version>/
les fichiers dont le site a besoin pour faire les plans PDF d'un ouvrage (le fichier « rbar.ouvrage »), sans serveur et sans
rien charger d'un tiers.

L'archive vient de la conversation qui développe RBAR (le propriétaire la transmet) : ``rbar-plans-web-<version>.zip``, le
paquet que la CI du dépôt RBAR construit (artefact ``rbar-plans-web-<version>``). Rien n'est accepté sans contrôle :

- l'empreinte SHA-256 de l'ARCHIVE doit être celle que RBAR a annoncée (``ARCHIVE_SHA256`` : copie NOMMÉE de son message
  du 12/11/2026) ;
- chaque fichier déposé doit avoir la taille et l'empreinte que la notice du paquet (``LISEZMOI.md``) inscrit, et, pour ceux
  que le chargeur de .NET vérifie lui-même (``_framework/blazor.boot.json``), l'empreinte qu'il y lit ;
- chaque fichier fait au plus 25 Mio (la limite d'un fichier publié par Cloudflare) ;
- les fichiers ne sont JAMAIS modifiés : le chargeur de .NET refuse un fichier changé.

Ne sont pas déposés : la page d'exemple et son travailleur (``exemple.html``, ``exemple-worker.js``) et la notice — le site a
son propre travailleur (``js/plans-travailleur.js``). Les empreintes déposées sont écrites dans ``empreintes.json`` ;
tests/test_site_web.py les relit.

Une version REMPLACE la précédente, dossier entier : les fichiers de deux versions ne se mélangent pas (``blazor.boot.json``
porte l'empreinte de chacun), et un ancien dossier laissé dans ``vendor/`` serait publié pour rien — tout autre dossier
``rbar-plans-web-*`` est retiré. L'archive range ses fichiers à sa racine (0.4.40) ou sous ``rbar-plans-web/`` (0.4.38).

    python3 site/web/outils/vendre_rbar_plans.py <rbar-plans-web-0.4.43.zip>
"""
import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile

VERSION = "0.4.43"
# l'empreinte de l'archive annoncée par RBAR (« rbar-plans-web-0.4.43.zip.sha256 », joint à « MESSAGE-AU-SITE-cages-emboitees-0443 »,
# 13/11/2026) : copie NOMMÉE
ARCHIVE_SHA256 = "98bcea1190bd1cfd98d0aebfb492cff62cf684568ecff9697b3d608d91ea8550"
RACINE_ARCHIVE = "rbar-plans-web/"  # (0.4.38 ; depuis 0.4.40, les fichiers sont à la racine de l'archive)
NON_DEPOSES = ("LISEZMOI.md", "exemple.html", "exemple-worker.js")
TAILLE_MAX = 25 * 1024 * 1024
ICI = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(ICI, "..", "vendor", f"rbar-plans-web-{VERSION}")


def sha256(octets: bytes) -> str:
    return hashlib.sha256(octets).hexdigest()


def table_de_la_notice(texte: str) -> dict:
    """{chemin: (octets, sha256)} : le tableau « Fichiers » de LISEZMOI.md."""
    out = {}
    for chemin, octets, emp in re.findall(r"^\| ([^|]+?) \| (\d+) \| ([0-9a-f]{64}) \|$", texte, re.M):
        out[chemin.strip()] = (int(octets), emp)
    return out


def integrites_du_chargeur(boot: dict) -> dict:
    """{chemin dans le paquet: sha256 hexadécimal} : les fichiers que le chargeur de .NET vérifie (blazor.boot.json)."""
    out = {}
    for groupe in boot["resources"].values():
        if not isinstance(groupe, dict):
            continue
        for nom, integrite in groupe.items():
            if isinstance(integrite, str) and integrite.startswith("sha256-"):
                out["_framework/" + nom] = base64.b64decode(integrite[len("sha256-"):]).hex()
    return out


def deposer(archive: str) -> dict:
    with open(archive, "rb") as h:
        brut = h.read()
    if sha256(brut) != ARCHIVE_SHA256:
        raise SystemExit(f"archive refusée : empreinte {sha256(brut)}, RBAR a annoncé {ARCHIVE_SHA256}")
    z = zipfile.ZipFile(io.BytesIO(brut))
    fichiers = [i for i in z.infolist() if not i.is_dir()]
    racine = RACINE_ARCHIVE if fichiers and all(i.filename.startswith(RACINE_ARCHIVE) for i in fichiers) else ""
    if any(i.filename.startswith("/") or ".." in i.filename.split("/") for i in fichiers):
        raise SystemExit("archive refusée : un chemin sort du dossier")
    contenu = {i.filename[len(racine):]: z.read(i) for i in fichiers}
    notice = table_de_la_notice(contenu["LISEZMOI.md"].decode("utf-8"))
    chargeur = integrites_du_chargeur(json.loads(contenu["_framework/blazor.boot.json"].decode("utf-8")))
    a_deposer = sorted(f for f in contenu if f not in NON_DEPOSES)
    fautes = []
    if sorted(notice) != sorted(f for f in contenu if f != "LISEZMOI.md"):
        fautes.append(f"la notice et l'archive ne listent pas les mêmes fichiers : {sorted(set(notice) ^ set(contenu) - {'LISEZMOI.md'})}")
    for f in a_deposer:
        o = contenu[f]
        if f not in notice or notice[f] != (len(o), sha256(o)):
            fautes.append(f"{f} : taille ou empreinte différente de la notice")
        if f in chargeur and chargeur[f] != sha256(o):
            fautes.append(f"{f} : empreinte différente de celle que le chargeur de .NET vérifie")
        if len(o) > TAILLE_MAX:
            fautes.append(f"{f} : {len(o)} octets, plus de 25 Mio")
    if sorted(chargeur) != sorted(f for f in chargeur if f in contenu):
        fautes.append("un fichier que le chargeur de .NET attend manque à l'archive")
    if fautes:
        raise SystemExit("archive refusée :\n- " + "\n- ".join(fautes))
    vendor = os.path.dirname(os.path.abspath(DEST))
    for ancien in sorted(os.listdir(vendor)):
        if ancien.startswith("rbar-plans-web-") and os.path.isdir(os.path.join(vendor, ancien)):
            shutil.rmtree(os.path.join(vendor, ancien))
    for f in a_deposer:
        chemin = os.path.join(DEST, *f.split("/"))
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        with open(chemin, "wb") as h:
            h.write(contenu[f])
    emp = {"version": VERSION, "archive_sha256": ARCHIVE_SHA256,
           "fichiers": {f: sha256(contenu[f]) for f in a_deposer},
           "non_deposes": {f: sha256(contenu[f]) for f in NON_DEPOSES if f in contenu}}
    with open(os.path.join(DEST, "empreintes.json"), "w", encoding="utf-8") as h:
        json.dump(emp, h, ensure_ascii=False, indent=1)
        h.write("\n")
    return emp


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    e = deposer(sys.argv[1])
    total = sum(os.path.getsize(os.path.join(DEST, *f.split("/"))) for f in e["fichiers"])
    print(f"{len(e['fichiers'])} fichiers déposés dans {os.path.relpath(DEST)} ({total / 1048576:.2f} Mio)")
