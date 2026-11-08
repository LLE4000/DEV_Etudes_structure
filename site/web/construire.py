# -*- coding: utf-8 -*-
"""Construit le site « Stab Méthodes » (site/web), un site statique qui calcule DANS le navigateur :

- les pages : l'accueil (index.html) et l'établi de chaque module du registre (api_web/registre.py : un fichier
  api_web/<clé>.py qui déclare MODULE est un module ; son établi est <PAGE>.html) ; js/modules.js, la même liste pour
  les scripts de l'écran ;
- la charte : css/charte.css est la COPIE de site/maquettes/charte.css (celle que le bureau a validée) ;
- le PAQUET DU MOTEUR, moteur/moteur.zip : les fichiers Python que le site exécute (api_web, le moteur de chaque
  module, leurs données), relevés en faisant tourner l'API sur des cas variés, bibliothèques lourdes refusées.
  Le paquet est reproductible (dates fixes, ordre trié) : tests/test_site_web.py le reconstruit et le compare.

    python3 site/web/construire.py                                  # le site, dans site/web
    python3 site/web/construire.py --dossier /tmp/x --modules dalle # une copie à part, paquet relevé sur ces modules

Les icônes et les noms des modules viennent du générateur des maquettes (une seule source) ; le cœur Python du
navigateur (Pyodide) est déposé par outils/vendre_pyodide.py.
"""
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
MAQUETTES = os.path.join(RACINE, "site", "maquettes")
# le générateur des maquettes, chargé sous son propre nom (les deux fichiers s'appellent construire.py) : les icônes,
# les familles et les noms des modules ont ainsi UNE source
_spec = importlib.util.spec_from_file_location("maquettes_construire", os.path.join(MAQUETTES, "construire.py"))
MQ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MQ)

PYODIDE = "vendor/pyodide-0.29.5"
DATE_ZIP = (2026, 1, 1, 0, 0, 0)
# les modules du site : le registre (un fichier api_web/<clé>.py qui déclare MODULE) ; les autres tuiles de l'accueil
# disent « bientôt »
sys.path.insert(0, RACINE)
from api_web import registre as REG  # noqa: E402

_FAMILLES = {k for _, _, mods in MQ.FAMILLES for k, _ in mods}
# clé → adresse de l'établi, SANS « .html » (Cloudflare la sert ainsi) ; une clé hors des familles de l'accueil n'a
# ni nom ni icône : elle est écartée (tests/test_site_web.py le signale)
ADRESSES = {k: v for k, v in REG.pages().items() if k in _FAMILLES}
PAGES = {k: v + ".html" for k, v in ADRESSES.items()}
ACCUEIL = "./"
# L'établi COMPACT (liste des éléments et des étapes à gauche, dessin au centre, propriétés et résultat à droite), en
# PILOTE sur ces modules seulement (docs/site/REFONTE_ETABLI.md) : les autres gardent l'établi d'avant. Leur carte de
# l'accueil ouvre le calcul DIRECTEMENT (sans la fenêtre « Nouvelle note ») ; la note s'enregistre à la première saisie.
ETABLI_COMPACT = ("poutre",)
# le relevé du paquet rejoue le PARCOURS de chaque module (api_web/parcours.py) : ce qu'un chemin rare importe doit
# être dans le paquet
MAX_RELEVE = 120
# les paquets de calcul propres au site, empaquetés ENTIERS (un import paresseux n'y manque jamais) ; acier et plancher,
# qui portent aussi les écrans Streamlit, sont relevés fichier par fichier
PAQUETS_ENTIERS = ("api_web", "beton", "bois", "sol", "flambement")
INTERDITS = ("streamlit", "reportlab", "matplotlib", "pandas", "fitz", "scipy", "PIL")


_SONDE = """
import builtins, json, os, sys
RACINE = {racine!r}
sys.path.insert(0, RACINE); os.chdir(RACINE)
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {interdits!r}:
            raise ImportError("refusé dans le navigateur : " + name)
sys.meta_path.insert(0, Refus())
lus = set(); _open = builtins.open
def espion(f, *a, **k):
    try:
        p = os.path.abspath(f)
        if p.startswith(RACINE + os.sep): lus.add(os.path.relpath(p, RACINE))
    except Exception: pass
    return _open(f, *a, **k)
builtins.open = espion
from api_web import parcours as P, routeur as R
echecs = []
for desc, ok, err in P.parcourir({cle!r}, R.traiter, maximum={maximum}):
    if not ok:
        echecs.append({cle!r} + " · " + desc + " : " + str(err))
if echecs:
    sys.stderr.write("\\n".join(echecs[:20])); sys.exit(3)
mods = {{os.path.relpath(m.__file__, RACINE) for m in list(sys.modules.values()) if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(RACINE + os.sep)}}
print(json.dumps({{"fichiers": sorted(mods | lus), "numpy": "numpy" in sys.modules}}))
"""
# un fichier qui importe numpy, même au fond d'une fonction que le parcours n'a pas suivie
_IMPORTE_NUMPY = re.compile(r"^\s*(?:import\s+numpy\b|from\s+numpy\b)", re.M)


def _releve_un(cle: str) -> dict:
    """Le relevé d'UN module, dans son propre sous-processus (bibliothèques lourdes refusées) : les fichiers du dépôt
    que son parcours charge, et s'il charge numpy — un module seul, pour que numpy chargé par un autre ne compte pas."""
    sonde = _SONDE.format(racine=RACINE, interdits=INTERDITS, cle=cle, maximum=MAX_RELEVE)
    p = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True, cwd=RACINE)
    if p.returncode != 0:
        raise SystemExit(f"Le relevé du moteur a échoué ({cle}) :\n" + p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def releve(modules=None) -> dict:
    """Le relevé du paquet du moteur : ``fichiers`` (les fichiers du dépôt que l'API charge, modules Python et données,
    relevés en rejouant le parcours de chaque module là où les bibliothèques lourdes sont refusées : ce qui y manque
    manquerait aussi dans le navigateur) et ``avec_numpy`` (les modules qui chargent numpy : le travailleur ne le charge
    que pour eux — il pèse une à deux secondes au démarrage ; un fichier qui l'importe, même dans une fonction que le
    parcours n'a pas suivie, compte). Un sous-processus par module, quatre à la fois."""
    cles = list(modules or ADRESSES)
    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as ex:
        par_module = dict(zip(cles, ex.map(_releve_un, cles)))
    fichiers = set().union(*(set(r["fichiers"]) for r in par_module.values()))
    avec_numpy = []
    for cle, r in par_module.items():
        sources = [f for f in r["fichiers"] if f.endswith(".py")]
        if r["numpy"] or any(_IMPORTE_NUMPY.search(open(os.path.join(RACINE, f), encoding="utf-8").read()) for f in sources):
            avec_numpy.append(cle)
    # les paquets du site entiers (petits), pour qu'un import paresseux ne manque jamais
    for paquet_ in PAQUETS_ENTIERS:
        if not os.path.isdir(os.path.join(RACINE, paquet_)):
            continue
        for d, _, fs in os.walk(os.path.join(RACINE, paquet_)):
            fichiers |= {os.path.relpath(os.path.join(d, f), RACINE) for f in fs if f.endswith(".py")}
    return {"fichiers": sorted(f for f in fichiers if "__pycache__" not in f and not f.endswith(".pyc")),
            "avec_numpy": sorted(avec_numpy)}


def fichiers_du_moteur(modules=None) -> list:
    """Les fichiers du dépôt que l'API charge (voir ``releve``)."""
    return releve(modules)["fichiers"]


def paquet(fichiers: list) -> bytes:
    """Le zip du moteur, reproductible : mêmes fichiers → mêmes octets."""
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in fichiers:
            info = zipfile.ZipInfo(f.replace(os.sep, "/"), DATE_ZIP)
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(os.path.join(RACINE, f), "rb") as src:
                z.writestr(info, src.read())
    return tampon.getvalue()


# ------------------------------------------------------------------ le paquet de la NOTE PDF (chargé à la demande)
# La note de calcul PDF des modules qui en font une (aujourd'hui : « Poutres béton armé ») est faite dans le navigateur
# par le générateur VALIDÉ de l'écran Streamlit (modules/export_pdf.py, ndc_pdf/) et reportlab, en Python pur. Ce paquet
# pèse trop pour le démarrage : le travailleur ne le charge qu'à la première demande de note (moteur/note.zip, empreinte
# dans le manifeste) et le déballe à côté du moteur. Les fichiers de reportlab sont RELEVÉS en faisant de vraies notes
# dans un sous-processus où fontTools, le Pillow réel et les accélérateurs C sont refusés (comme dans le navigateur).
NOTE_SOURCES = os.path.join(RACINE, "site", "note_pdf")
NOTE_POLICES = ("Carlito-Regular.ttf", "Carlito-Bold.ttf", "Carlito-Italic.ttf", "Poppins-Medium.ttf",
                "DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
_SONDE_NOTE = """
import builtins, json, os, sys
RACINE, STUB = {racine!r}, {stub!r}
sys.path.insert(0, STUB); sys.path.insert(1, RACINE); os.chdir(RACINE)
class Refus:
    def find_spec(self, name, path=None, target=None):
        racine = name.split(".")[0]
        if racine in ("fontTools", "_rl_accel", "rl_accel", "streamlit", "numpy", "matplotlib") or name == "reportlab.lib._rl_accel":
            raise ImportError("refusé dans le navigateur : " + name)
sys.meta_path.insert(0, Refus())
lus = set(); _open = builtins.open
def espion(f, *a, **k):
    try: lus.add(os.path.abspath(f))
    except Exception: pass
    return _open(f, *a, **k)
builtins.open = espion
from api_web import poutre as P, note_poutre as NP
etat = P.nouvel_etat()
for ch, v in ((["sections", 0, "M_inf"], 120), (["sections", 0, "V"], 80), (["sections", 0, "inf", "+"], 1), (["sections", 0, "etriers", "+"], "epingle"),
              (["sections", "+"], 1), (["sections", 1, "M_sup"], 90), (["sections", 0, "laterales", "n"], 1)):
    P.modifier(etat, ch, v)
P.gerer_elements(etat, "dupliquer")
NP.fichier(etat, {{"nom": "relevé"}})
import reportlab
base = os.path.dirname(os.path.dirname(os.path.abspath(reportlab.__file__)))
mods = {{os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}}
rl = sorted(os.path.relpath(f, base) for f in mods | lus if f.startswith(os.path.join(base, "reportlab") + os.sep) and not f.endswith(".pyc"))
print(json.dumps({{"base": base, "reportlab": rl, "version": reportlab.Version}}))
"""


def releve_note() -> dict:
    """Ce que la note PDF charge (sous-processus, comme le navigateur) : {"base", "reportlab": [fichiers], "version"}."""
    sonde = _SONDE_NOTE.format(racine=RACINE, stub=NOTE_SOURCES)
    p = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True, cwd=RACINE,
                       env={**os.environ, "NDC_FONT_DIR": os.path.join(RACINE, "ndc_pdf", "fonts")})
    if p.returncode != 0:
        raise SystemExit("Le relevé de la note PDF a échoué :\n" + p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def paquet_note() -> tuple:
    """(octets du zip, description) : le générateur de la note, reportlab relevé (et sa licence), les polices de la mise
    en page, le Pillow factice. Reproductible : mêmes fichiers → mêmes octets."""
    rel = releve_note()
    entrees = [("modules/export_pdf.py", os.path.join(RACINE, "modules", "export_pdf.py"))]
    for f in sorted(os.listdir(os.path.join(RACINE, "ndc_pdf"))):
        if f.endswith(".py"):
            entrees.append((f"ndc_pdf/{f}", os.path.join(RACINE, "ndc_pdf", f)))
    entrees += [(f"ndc_pdf/fonts/{f}", os.path.join(RACINE, "ndc_pdf", "fonts", f)) for f in NOTE_POLICES]
    entrees += [(f.replace(os.sep, "/"), os.path.join(rel["base"], f)) for f in rel["reportlab"]]
    entrees.append(("reportlab/license.txt", os.path.join(rel["base"], "reportlab", "license.txt")))
    for d, _, fs in os.walk(NOTE_SOURCES):
        for f in sorted(fs):
            if f.endswith((".py", ".txt")) and "__pycache__" not in d:
                src = os.path.join(d, f)
                entrees.append((os.path.relpath(src, NOTE_SOURCES).replace(os.sep, "/"), src))
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for nom, src in sorted(set(entrees)):
            info = zipfile.ZipInfo(nom, DATE_ZIP)
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(src, "rb") as f:
                z.writestr(info, f.read())
    return tampon.getvalue(), {"reportlab": rel["version"], "fichiers": len(set(entrees))}


# ------------------------------------------------------------------ les morceaux des pages
def _module(cle: str):
    for fam, nom_fam, mods in MQ.FAMILLES:
        for k, nom in mods:
            if k == cle:
                return fam, nom
    raise KeyError(cle)


def rail(actif: str) -> str:
    """La barre de cinq icônes ; Projets, Équipe et Paramètres viendront avec les comptes (Supabase)."""
    def it(ic, nom, cle, attr):
        return f'<a class="navitem{" on" if cle == actif else ""}" data-tip="{nom}"{attr}>{MQ.icon(ic)}</a>'
    bientot = ' aria-disabled="true" data-bientot="1"'
    return (f'<nav class="nav"><a class="brand" href="{ACCUEIL}" title="Accueil"><div class="logo">{MQ.LOGO}</div></a>'
            + it("house", "Accueil", "accueil", f' href="{ACCUEIL}"') + it("folder", "Projets · bientôt", "projets", bientot)
            + it("layout-grid", "Modules", "modules", ' data-modules="1"') + it("users", "Équipe · bientôt", "equipe", bientot)
            + it("settings", "Paramètres · bientôt", "parametres", bientot)
            + f'<div style="margin-top:auto;display:flex;flex-direction:column;align-items:center;gap:10px">'
              f'<a class="navitem" data-theme-toggle="1" data-tip="Thème clair ou sombre">{MQ.icon("sun-moon")}</a></div></nav>' + modpanel())


def mnav(actif: str) -> str:
    """Les onglets du bas du téléphone : les cinq entrées de la barre du PC, dans le même ordre (cahier des charges
    § 1 bis) ; Projets, Équipe et Paramètres viendront avec les comptes."""
    def it(ic, nom, cle, attr):
        return f'<a class="{"on" if cle == actif else ""}"{attr}>{MQ.icon(ic)}<span>{nom}</span></a>'
    bientot = ' aria-disabled="true" data-bientot="1" title="Bientôt"'
    return ('<nav class="mnav">' + it("house", "Accueil", "accueil", f' href="{ACCUEIL}"') + it("folder", "Projets", "projets", bientot)
            + it("layout-grid", "Modules", "modules", ' data-modules="1" role="button"') + it("users", "Équipe", "equipe", bientot)
            + it("settings", "Paramètres", "parametres", bientot) + "</nav>")


def modpanel() -> str:
    corps = ""
    for fam, nom_fam, mods in MQ.FAMILLES:
        corps += f'<div class="navlabel">{nom_fam}</div>'
        for k, n in mods:
            if k in PAGES:
                corps += f'<a class="navitem" href="{ADRESSES[k]}"><span class="modic {fam}">{MQ.MOD[k]}</span><span>{MQ.esc(n)}</span></a>'
            else:
                corps += (f'<a class="navitem bientot" aria-disabled="true"><span class="modic {fam}">{MQ.MOD[k]}</span><span>{MQ.esc(n)}</span>'
                          f'<span class="tag draft">bientôt</span></a>')
    return (f'<aside class="modpanel"><div class="mp-h"><div class="h2">Modules</div>'
            f'<a class="btn icon ghost" data-modules="1" title="Fermer">{MQ.icon("x")}</a></div>{corps}</aside>')


def tete(titre: str, css: str = "", css2: str = "") -> str:
    lien = "".join(f'<link rel="stylesheet" href="{c}">\n' for c in (css2, css) if c)
    return ('<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{titre}</title>\n<link rel="icon" href="favicon.svg" type="image/svg+xml">\n'
            '<link rel="stylesheet" href="css/charte.css">\n<link rel="stylesheet" href="css/site.css">\n' + lien
            + '<script src="js/theme.js"></script>\n</head>\n')


def _sous_titre_modules() -> str:
    tous = [k for _, _, mods in MQ.FAMILLES for k, _ in mods]
    n = sum(k in PAGES for k in tous)
    return "Tous les modules du bureau" if n == len(tous) else f"{n} modules sur {len(tous)} · les autres arrivent"


def page_accueil() -> str:
    tuiles = ""
    for fam, nom_fam, mods in MQ.FAMILLES:
        t = ""
        for k, n in mods:
            if k in PAGES:
                ouvre = f'data-module="{k}"' if k in ETABLI_COMPACT else f'data-nouvelle-module="{k}"'
                t += (f'<a class="tile" href="{ADRESSES[k]}" {ouvre}><span class="modic {fam}">{MQ.MOD[k]}</span>'
                      f'<span class="nm">{MQ.esc(n)}</span></a>')
            else:
                t += (f'<div class="tile bientot" title="Ce module n\'est pas encore sur le site (il reste dans Streamlit)">'
                      f'<span class="modic {fam}">{MQ.MOD[k]}</span><span class="nm">{MQ.esc(n)}</span><span class="tag draft">bientôt</span></div>')
        tuiles += f'<div class="famille"><div class="kicker">{nom_fam}</div><div class="tiles">{t}</div></div>'
    options = "".join(f'<button type="button" class="mopt" data-module="{k}"><span class="modic {_module(k)[0]}">{MQ.MOD[k]}</span>{MQ.esc(_module(k)[1])}</button>'
                      for k in PAGES)
    modal = (f'<div class="modal" id="nouvelle" hidden><div class="mcard"><div class="mh"><span class="h2">Nouvelle note</span>'
             f'<button class="btn icon ghost" data-fermer-modal="1" title="Fermer">{MQ.icon("x")}</button></div>'
             f'<div class="mb"><div class="mrow3"><label><span>Chantier</span><input id="nn-chantier" list="nn-chantiers" placeholder="ex. Réalex"></label>'
             f'<label><span>Dossier</span><input id="nn-dossier" placeholder="ex. Étude de la dalle"></label>'
             f'<label><span>Nom de la note</span><input id="nn-nom" placeholder="ex. Poutres R+1"></label></div><datalist id="nn-chantiers"></datalist>'
             f'<div class="mlabel">Module</div><div class="mgrid">{options}</div></div>'
             f'<div class="mf"><button class="btn" data-fermer-modal="1">Annuler</button>'
             f'<button class="btn primary" id="nn-creer" disabled>{MQ.icon("plus")}Créer la note</button></div></div></div>')
    # au téléphone : l'en-tête (le nom du site, le thème) et les cinq entrées de la barre, en bas, dans le même ordre
    mhead = (f'<header class="mhead"><div class="logo">{MQ.LOGO}</div><span class="ttl">Stab Méthodes</span><span class="spacer"></span>'
             f'<button type="button" class="btn icon ghost" data-theme-toggle="1" title="Thème clair ou sombre">{MQ.icon("sun-moon")}</button></header>')
    corps = (f'<body data-page="accueil"><div class="app rail">{rail("accueil")}<main class="main">{mhead}'
             f'<div class="page accueil"><div class="col"><div class="et-memoire" id="et-memoire" role="alert" hidden></div>'
             f'<header class="hello"><div><h1 class="h1">Bonjour</h1><p class="muted" id="date"></p></div><span class="spacer"></span>'
             f'<button class="btn primary big" id="ouvrir-nouvelle">{MQ.icon("plus")}Nouvelle note</button></header>'
             f'<section><div class="sechead"><span class="h2">Reprendre</span><span class="muted" id="nb-notes"></span></div>'
             f'<div class="resume" id="recentes"></div></section>'
             f'<section><div class="sechead"><span class="h2">Modules</span><span class="muted">{_sous_titre_modules()}</span></div>{tuiles}</section>'
             f'</div><aside class="side"><div class="card"><div class="sechead"><span class="h3">Chantiers</span></div><div id="chantiers"></div></div>'
             f'<div class="card note-locale"><div class="h3">Où sont mes notes ?</div><p>Pour l\'instant, chaque note est enregistrée '
             f'<b>dans ce navigateur</b>, sur ce PC. Le bouton <b>.stab</b> d\'une note l\'exporte en fichier ; les comptes du bureau et le partage '
             f'viendront ensuite.</p><label class="btn sm" for="importer">{MQ.icon("upload")}Importer un fichier .stab</label>'
             f'<input type="file" id="importer" accept=".stab,.json" hidden></div></aside></div>{mnav("accueil")}</main></div>{modal}'
             f'<div class="toast" id="toast" role="status" hidden></div>'
             f'<script type="module" src="js/accueil.js"></script></body>\n</html>\n')
    return tete("Stab Méthodes") + corps


def _bouton(icone: str, libelle: str, attrs: str, cls: str = "btn") -> str:
    """Un bouton de la barre du haut : l'icône et son nom ; le nom se replie sous 1 100 px (la tablette), le titre
    reste au survol et pour les lecteurs d'écran."""
    return f'<button type="button" class="{cls}"{attrs}>{MQ.icon(icone)}<span class="lbl">{libelle}</span></button>'


PDF_BIENTOT = "La note PDF arrive bientôt sur le site (elle reste dans Streamlit)"


def enregistrement() -> str:
    """L'indicateur d'enregistrement de la note (.et-enr) : le MÊME composant dans la barre du haut (PC, tablette) et sur
    la 2ᵉ ligne de l'en-tête du téléphone ; js/etabli.js y écrit l'état de la dernière écriture (neuf, enregistré,
    échec). Aucune règle ne le cache, ni son texte (.et-enr-txt) : un échec doit se lire à toutes les largeurs.
    L'ancien indicateur (.saved .txt) était caché par la charte sous 1 360 px — l'échec y était muet.
    Le texte (.et-enr-txt) porte l'ÉTAT (.et-enr-etat, « Non enregistré ») puis, en échec, la RAISON (.et-enr-raison,
    « : stockage plein »), dans la même ligne de texte : la 2ᵉ ligne du téléphone n'a pas la place des deux à côté de
    la pastille de la note (mesuré : « Non enregistré : stockage plein » passait SOUS la pastille « À compléter », à
    390 px) ; elle n'y écrit que l'état, la raison restant dans le message (#et-alerte), dans le title et dans le menu ⋯
    (menu_enregistrement)."""
    return ('<span class="et-enr" data-enr="neuf" role="status">'
            f'<span class="et-enr-ic">{MQ.icon("clock", "i et-i-neuf")}{MQ.icon("circle-check", "i et-i-ok")}'
            f'{MQ.icon("triangle-alert", "i et-i-echec")}</span>'
            '<span class="et-enr-mot"></span><span class="et-enr-txt"><span class="et-enr-etat">Pas encore enregistré</span>'
            '<span class="et-enr-raison"></span></span></span>')


def menu_enregistrement() -> str:
    """L'entrée du menu ⋯ du téléphone qui dit un échec d'enregistrement EN ENTIER (« Non enregistré : stockage plein »)
    et rouvre son message (#et-alerte, avec « Exporter .stab ») : la 2ᵉ ligne de l'en-tête n'y écrit que « Non
    enregistré », et un message fermé ne revient pas de lui-même. Cachée hors échec ; js/etabli.js l'écrit. Une cible de
    46 px, comme les autres entrées du menu (au doigt, toute cible ≥ 44 px)."""
    return (f'<button type="button" class="mi et-mi-enr" data-revoir-alerte="1" hidden>{MQ.icon("triangle-alert")}'
            '<span class="et-mi-enr-txt"></span></button>')


def alerte_enregistrement() -> str:
    """Le message d'un échec d'enregistrement (#et-alerte) : dit une fois, il reste jusqu'à ce qu'on le ferme ou qu'une
    écriture réussisse, et offre « Exporter .stab » (la note n'est alors gardée nulle part ailleurs). Le même aux trois
    formats ; js/etabli.js l'écrit."""
    return (f'<div class="et-alerte" id="et-alerte" role="alert" hidden><span class="et-alerte-ic">{MQ.icon("triangle-alert")}</span>'
            '<span class="et-alerte-txt"></span>'
            f'<button type="button" class="btn sm primary et-alerte-stab" data-action="exporter">{MQ.icon("download")}Exporter .stab</button>'
            f'<button type="button" class="btn icon ghost et-alerte-x" data-fermer-alerte="1" title="Fermer le message" '
            f'aria-label="Fermer le message">{MQ.icon("x")}</button></div>')


def page_module(cle: str, titre: str) -> str:
    """L'établi d'un module : la toile du dessin sur toute la largeur, les données et le résultat en cadres flottants ;
    tout le contenu vient du moteur (le travailleur Pyodide).

    Trois formats, une seule page (les fonctions sont les mêmes, la disposition change, css/site.css) :
    - PC (1 200 px et plus) : la barre du haut, les cadres flottants de la maquette ;
    - tablette (701 à 1 199 px) : la toile en haut sur toute la largeur, les deux cadres dessous (portrait), ou la
      toile à droite sur toute la hauteur et les deux cadres l'un sur l'autre à gauche (paysage) ;
    - téléphone (700 px et moins) : l'en-tête compact (.mtop, les actions secondaires dans un menu), les onglets
      Données · Dessin · Résultat (.mtabs) et la barre du bas (.mbar), le dessin des maquettes."""
    fam, nom = _module(cle)
    top = (f'<header class="top"><div class="crumbs"><a href="{ACCUEIL}" title="Accueil">{MQ.icon("house")}</a>{MQ.icon("chevron-right")}'
           f'<span id="fil-chantier" class="muted"></span><span id="fil-sep">{MQ.icon("chevron-right")}</span>'
           f'<input id="nom-note" class="nomnote" data-nom-note="1" value="" aria-label="Nom de la note" spellcheck="false"></div>'
           f'<span class="tag draft"><span class="dot"></span>Brouillon</span>'
           f'<span class="spacer"></span>{enregistrement()}'
           f'<label class="btn ghost" for="ouvrir" title="Ouvrir un fichier .stab">{MQ.icon("upload")}<span class="lbl">Ouvrir</span></label>'
           + _bouton("download", ".stab", ' data-action="exporter" title="Exporter la note en fichier .stab"')
           + _bouton("printer", "Imprimer", ' data-action="imprimer" title="Imprimer l\'écran"')
           + _bouton("file-down", "Note PDF", f' disabled title="{PDF_BIENTOT}"', "btn primary") + '</header>')
    # le téléphone : l'en-tête compact (retour, nom de la note, taux, menu), son menu, les onglets, la barre du bas
    mtop = (f'<header class="mtop"><a class="btn icon ghost" href="{ACCUEIL}" title="Accueil" aria-label="Accueil">{MQ.icon("arrow-left")}</a>'
            f'<div class="t"><input id="nom-note-m" class="nomnote" data-nom-note="1" value="" aria-label="Nom de la note" spellcheck="false">'
            f'<span class="et-l2">{enregistrement()}<span id="fil-m"></span></span></div><span class="calcule"></span>'
            f'<button type="button" class="mchip" id="chip-m" data-vers-onglet="resultat" title="Voir le résultat"></button>'
            f'<button type="button" class="btn icon ghost" id="menu-m" aria-haspopup="true" aria-expanded="false" aria-controls="mmenu" '
            f'title="Plus d\'actions">{MQ.icon("ellipsis")}</button></header>')
    mmenu = (f'<div class="mmenu" id="mmenu" hidden>{menu_enregistrement()}<label class="mi" for="ouvrir">{MQ.icon("upload")}Ouvrir un fichier .stab</label>'
             f'<button type="button" class="mi" data-action="exporter">{MQ.icon("download")}Exporter la note en .stab</button>'
             f'<button type="button" class="mi" data-action="imprimer">{MQ.icon("printer")}Imprimer</button>'
             f'<button type="button" class="mi" data-action="pdf" aria-disabled="true">{MQ.icon("file-down")}Note PDF<span class="tag draft">bientôt</span></button>'
             f'<button type="button" class="mi" data-theme-toggle="1">{MQ.icon("sun-moon")}Thème clair ou sombre</button>'
             f'<button type="button" class="mi" data-modules="1">{MQ.icon("layout-grid")}Tous les modules</button></div>')
    mtabs = ('<nav class="mtabs" aria-label="Données, dessin, résultat"><div class="seg full">'
             + "".join(f'<button type="button" data-onglet="{k}" aria-pressed="{"true" if k == "donnees" else "false"}"'
                       f'{" class=\"on\"" if k == "donnees" else ""}>{t}</button>'
                       for k, t in (("donnees", "Données"), ("dessin", "Dessin"), ("resultat", "Résultat")))
             + '</div></nav>')
    mbar = (f'<div class="mbar">{_bouton("download", "Exporter .stab", " data-action=\"exporter\"")}'
            f'{_bouton("file-down", "Note PDF", " data-action=\"pdf\" aria-disabled=\"true\"", "btn primary")}</div>')
    gauche = (f'<div class="phead"><span class="h3">Données</span><span class="calcule" id="calcule"></span>'
              f'<button class="pcoll" data-coll="g" title="Replier les données">{MQ.icon("panel-left")}</button></div>'
              f'<div class="pbody" id="donnees"><div class="attente">Chargement…</div></div>')
    droite = (f'<div class="phead"><span class="h3">Résultat</span><span id="statut"></span>'
              f'<button class="pcoll" data-coll="d" title="Replier le résultat">{MQ.icon("panel-right")}</button></div>'
              f'<div id="resultat" class="resultat"><div class="pbody"><div class="attente">Chargement…</div></div></div>')
    # la barre de la toile : les vues, puis le zoom (réduire, ajuster, agrandir) et le plein écran
    zoom = (f'<div class="zoom" id="zoom"><button type="button" data-zoom="moins" title="Réduire le dessin">{MQ.icon("zoom-out")}</button>'
            f'<button type="button" class="zpct" data-zoom="ajuster" title="Ajuster le dessin à la feuille">100 %</button>'
            f'<button type="button" data-zoom="plus" title="Agrandir le dessin">{MQ.icon("zoom-in")}</button>'
            f'<button type="button" class="zplein" data-plein="1" title="Plein écran">{MQ.icon("maximize-2")}{MQ.icon("x")}</button></div>')
    toile = (f'<section class="canvas"><div class="ctool"><div class="seg" id="vues"></div>{zoom}</div>'
             f'<div class="stage" id="stage"></div></section>')
    demarrage = (f'<div class="demarrage" id="demarrage"><div class="carte"><div class="logo">{MQ.LOGO}</div><b>Chargement du moteur de calcul</b>'
                 f'<span id="demarrage-etape">Python démarre dans le navigateur…</span><div class="barre"><i></i></div>'
                 f'<small>La première ouverture prend quelques secondes ; ensuite, chaque calcul est immédiat.</small></div></div>')
    corps = (f'<body data-page="module" data-module="{cle}" data-titre="{MQ.esc(nom)}"><div class="app rail">{rail("modules")}<main class="main">{top}'
             f'{mtop}{mmenu}{mtabs}'
             f'<div class="bench" data-tab="donnees">{toile}<section class="panel left">{gauche}</section>'
             f'<section class="panel right">{droite}</section><div id="volets"></div>{demarrage}</div>{mbar}</main></div>'
             f'<input type="file" id="ouvrir" accept=".stab,.json" hidden>{alerte_enregistrement()}'
             f'<div class="toast" id="toast" role="status" hidden></div>'
             f'<script type="module" src="js/etabli.js"></script></body>\n</html>\n')
    if cle in ETABLI_COMPACT:
        corps = compact(corps)
    css = f"css/modules/{cle}.css"
    return tete(f"{titre} · Stab Méthodes", css if os.path.exists(os.path.join(ICI, css)) else "",
                "css/etabli.css" if cle in ETABLI_COMPACT else "") + corps


def compact(corps: str) -> str:
    """L'établi COMPACT, tiré de l'établi ordinaire (les mêmes fragments du moteur, les mêmes identifiants : #donnees,
    #resultat, #volets ; js/etabli.js, js/compact.js, js/rangement.js) :
    - à gauche, les DONNÉES : les éléments de la note, puis les rubriques du module en accordéon (une ouverte à la fois),
      avec « Annuler » et « Rétablir » dans l'en-tête ;
    - au centre, le DESSIN : un clic sur une pièce l'ouvre dans les données (et une ligne des données la montre sur le
      dessin) ;
    - à droite, les RÉSULTATS seuls, compacts ; un clic ouvre le détail ;
    - les deux colonnes se replient et se REDIMENSIONNENT (une poignée sur leur bord intérieur) ;
    - la barre du haut gagne « Enregistrer comme note… »."""
    gauche_av = ('<div class="phead"><span class="h3">Données</span><span class="calcule" id="calcule"></span>'
                 f'<button class="pcoll" data-coll="g" title="Replier les données">{MQ.icon("panel-left")}</button></div>')
    droite_av = ('<div class="phead"><span class="h3">Résultat</span><span id="statut"></span>'
                 f'<button class="pcoll" data-coll="d" title="Replier le résultat">{MQ.icon("panel-right")}</button></div>')
    gauche = ('<div class="phead"><span class="h3">Données</span><span class="calcule" id="calcule"></span>'
              f'<button type="button" class="et-hist" data-historique="annuler" title="Annuler (Ctrl+Z)" aria-label="Annuler" disabled>{MQ.icon("undo-2")}</button>'
              f'<button type="button" class="et-hist" data-historique="retablir" title="Rétablir (Ctrl+Maj+Z)" aria-label="Rétablir" disabled>{MQ.icon("redo-2")}</button>'
              f'<button class="pcoll" data-coll="g" title="Replier les données">{MQ.icon("panel-left")}</button></div>')
    droite = ('<div class="phead"><span class="h3">Résultats</span><span id="statut"></span>'
              f'<button class="pcoll" data-coll="d" title="Replier les résultats">{MQ.icon("panel-right")}</button></div>')
    assert gauche_av in corps and droite_av in corps, "le gabarit de l'établi a changé : compact() doit suivre"
    corps = corps.replace(gauche_av, gauche).replace(droite_av, droite)
    # les poignées qui redimensionnent les deux colonnes (PC) : à CHEVAL sur le bord intérieur de chaque panneau, posées
    # dans l'établi et non dans le panneau (qui rogne ce qui dépasse : son bord, le trait qu'on vise, n'était pas saisissable)
    for cote, nom in (("left", "g"), ("right", "d")):
        marque = f'<section class="panel {cote}">'
        assert corps.count(marque) == 1
        corps = corps.replace(marque, f'<div class="et-poignee" data-poignee="{nom}" role="separator" aria-orientation="vertical" '
                              f'title="Glisser pour élargir ; double-clic : largeur d\'origine"></div>' + marque, 1)
    corps = corps.replace('<body data-page="module"', '<body data-page="module" data-etabli="compact"', 1)
    # « Enregistrer comme note… » : à côté de l'indicateur d'enregistrement (barre du haut), et dans le menu ⋯ du téléphone
    bouton = (f'<button type="button" class="btn sm et-ranger" data-ranger="1" title="Nommer la note et la ranger dans un chantier">'
              f'{MQ.icon("folder")}<span class="lbl">Enregistrer comme note…</span></button>')
    marque = '<label class="btn ghost" for="ouvrir"'
    assert marque in corps
    corps = corps.replace(marque, bouton + marque, 1)
    # (au téléphone, l'échec d'enregistrement reste la PREMIÈRE entrée du menu : « Enregistrer comme note… » le suit)
    revoir = menu_enregistrement()
    assert corps.count('<div class="mmenu" id="mmenu" hidden>' + revoir) == 1
    corps = corps.replace('<div class="mmenu" id="mmenu" hidden>' + revoir, '<div class="mmenu" id="mmenu" hidden>' + revoir
                          + f'<button type="button" class="mi" data-ranger="1">{MQ.icon("folder")}<span class="et-mi-ranger">Enregistrer comme note…</span></button>', 1)
    # la note PDF : le module la prépare dans le navigateur (js/etabli.js, action « telecharger », quoi = « note_pdf »)
    for avant, apres in ((_bouton("file-down", "Note PDF", f' disabled title="{PDF_BIENTOT}"', "btn primary"),
                          _bouton("file-down", "Note PDF", ' data-action="pdf" title="Préparer la note de calcul PDF"', "btn primary")),
                         ('data-action="pdf" aria-disabled="true">' + MQ.icon("file-down") + 'Note PDF<span class="tag draft">bientôt</span>',
                          'data-action="pdf">' + MQ.icon("file-down") + 'Note PDF'),
                         (_bouton("file-down", "Note PDF", ' data-action="pdf" aria-disabled="true"', "btn primary"),
                          _bouton("file-down", "Note PDF", ' data-action="pdf"', "btn primary"))):
        assert corps.count(avant) == 1, avant
        corps = corps.replace(avant, apres, 1)
    # la fenêtre « Enregistrer comme note » : le nom, le chantier, le dossier (texte, avec les chantiers déjà connus)
    fenetre = ('<div class="et-rangement" id="et-rangement" role="dialog" aria-modal="true" aria-labelledby="et-rg-titre" hidden>'
               '<form class="et-rg-carte" id="et-rg-form"><div class="et-rg-h"><b id="et-rg-titre">Enregistrer comme note</b>'
               f'<button type="button" class="btn icon ghost" data-fermer-rangement="1" aria-label="Fermer">{MQ.icon("x")}</button></div>'
               '<label class="et-rg-l">Nom de la note<input id="et-rg-nom" class="field-in" autocomplete="off" spellcheck="false" maxlength="200"></label>'
               '<label class="et-rg-l">Chantier<input id="et-rg-chantier" list="et-rg-chantiers" class="field-in" autocomplete="off" spellcheck="false" maxlength="200" placeholder="Sans chantier"></label>'
               '<label class="et-rg-l">Dossier<input id="et-rg-dossier" list="et-rg-dossiers" class="field-in" autocomplete="off" spellcheck="false" maxlength="200" placeholder="Sans dossier"></label>'
               '<datalist id="et-rg-chantiers"></datalist><datalist id="et-rg-dossiers"></datalist>'
               '<p class="et-rg-aide">La note est enregistrée dans ce navigateur ; « Exporter .stab » en garde une copie en fichier.</p>'
               '<div class="et-rg-b"><button type="button" class="btn ghost" data-fermer-rangement="1">Annuler</button>'
               '<button type="submit" class="btn primary">Enregistrer</button></div></form></div>')
    corps = corps.replace('<div class="toast" id="toast"', fenetre + '<div class="toast" id="toast"', 1)
    return corps


def favicon() -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#2B4BCF"/>'
            '<g transform="translate(4 4)">' + MQ.LOGO.replace("<svg", "<svg width=\"24\" height=\"24\"") + '</g></svg>')


def ecrire(chemin: str, contenu, binaire: bool = False) -> None:
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "wb" if binaire else "w", **({} if binaire else {"encoding": "utf-8"})) as f:
        f.write(contenu)


FIXES = ("js", "css", "_headers", "favicon.svg", ".assetsignore", "wrangler.jsonc", "LICENCES.md", "LICENCE-icones.txt")


def modules_js() -> str:
    """La liste des modules pour les scripts de l'écran (adresse, nom, famille) : la même que celle des pages."""
    d = {}
    for fam, _, mods in MQ.FAMILLES:
        for k, nom in mods:
            if k in ADRESSES:
                d[k] = {"page": ADRESSES[k], "nom": nom, "famille": fam}
    return ("// Écrit par site/web/construire.py depuis le registre des modules (api_web/registre.py) : ne pas modifier.\n"
            f"export const MODULES = {json.dumps(d, ensure_ascii=False)};\n")


def construire(dossier: str = ICI, modules=None) -> dict:
    """Écrit tout dans ``dossier`` (le site lui-même, ou une copie à part pour vérifier un module pendant que d'autres
    se construisent) ; le paquet du moteur est relevé sur ``modules`` (tous par défaut). Rend le manifeste."""
    dossier = os.path.abspath(dossier)
    if dossier != ICI:
        os.makedirs(dossier, exist_ok=True)
        for nom in FIXES:
            src, dst = os.path.join(ICI, nom), os.path.join(dossier, nom)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif os.path.exists(src):
                shutil.copyfile(src, dst)
        lien = os.path.join(dossier, "vendor")      # le cœur Python (15 Mo) n'est pas recopié : un lien suffit
        if not os.path.exists(lien):
            os.symlink(os.path.join(ICI, "vendor"), lien)
    shutil.copyfile(os.path.join(MAQUETTES, "charte.css"), os.path.join(dossier, "css", "charte.css"))
    ecrire(os.path.join(dossier, "index.html"), page_accueil())
    for cle, page in PAGES.items():
        ecrire(os.path.join(dossier, page), page_module(cle, _module(cle)[1]))
    ecrire(os.path.join(dossier, "js", "modules.js"), modules_js())
    ecrire(os.path.join(dossier, "favicon.svg"), favicon())
    rel = releve(modules)
    z = paquet(rel["fichiers"])
    ecrire(os.path.join(dossier, "moteur", "moteur.zip"), z, binaire=True)
    # « avec_numpy » : les modules dont le travailleur charge numpy dès le démarrage (les autres s'en passent ; un
    # import de numpy imprévu le fait charger à la demande, voir js/travailleur.js)
    # la note PDF : son paquet à part, chargé à la première demande (js/travailleur.js)
    zn, dn = paquet_note()
    ecrire(os.path.join(dossier, "moteur", "note.zip"), zn, binaire=True)
    manifeste = {"pyodide": PYODIDE, "sha256": hashlib.sha256(z).hexdigest(), "fichiers": rel["fichiers"],
                 "avec_numpy": rel["avec_numpy"],
                 "note": {"sha256": hashlib.sha256(zn).hexdigest(), "octets": len(zn), **dn}}
    ecrire(os.path.join(dossier, "moteur", "manifeste.json"), json.dumps(manifeste, ensure_ascii=False, indent=1) + "\n")
    return manifeste


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Construit le site Stab Méthodes")
    ap.add_argument("--dossier", default=ICI, help="où écrire (par défaut : site/web)")
    ap.add_argument("--modules", default="", help="relever le paquet sur ces modules seulement (clés séparées par des virgules)")
    a = ap.parse_args()
    mods = [m for m in a.modules.split(",") if m] or None
    m = construire(a.dossier, mods)
    print(f"{os.path.relpath(os.path.abspath(a.dossier), RACINE)} : {len(PAGES) + 1} pages ({', '.join(PAGES)}), paquet du moteur "
          f"{len(m['fichiers'])} fichiers, {os.path.getsize(os.path.join(a.dossier, 'moteur', 'moteur.zip')) // 1024} ko (sha256 {m['sha256'][:12]})")
