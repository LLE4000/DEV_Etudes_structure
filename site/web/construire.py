# -*- coding: utf-8 -*-
"""Construit le site « Stab Méthodes » (site/web), un site statique qui calcule DANS le navigateur :

- les pages : l'accueil (index.html) et l'établi de chaque module disponible (garde-corps.html, poutre.html) ;
- la charte : css/charte.css est la COPIE de site/maquettes/charte.css (celle que le bureau a validée) ;
- le PAQUET DU MOTEUR, moteur/moteur.zip : les fichiers Python que le site exécute (api_web, le moteur de chaque
  module, leurs données), relevés en faisant tourner l'API sur des cas variés, bibliothèques lourdes refusées.
  Le paquet est reproductible (dates fixes, ordre trié) : tests/test_site_web.py le reconstruit et le compare.

    python3 site/web/construire.py

Les icônes et les noms des modules viennent du générateur des maquettes (une seule source) ; le cœur Python du
navigateur (Pyodide) est déposé par outils/vendre_pyodide.py.
"""
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

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
# les pages des modules disponibles (clé du module → page) ; les autres tuiles de l'accueil disent « bientôt »
PAGES = {"garde_corps": "garde-corps.html", "poutre": "poutre.html"}
# les liens visent l'adresse SANS « .html », celle que Cloudflare Pages sert (il redirige « x.html » vers « x ») ;
# l'accueil est « ./ ». js/notes.js porte les mêmes adresses (tests/test_site_web.py les lie).
ADRESSES = {k: v[: -len(".html")] for k, v in PAGES.items()}
ACCUEIL = "./"
# des requêtes qui passent par tous les chemins de l'API : le paquet doit contenir ce qu'elles chargent
SCENARIOS = r"""
import json
from api_web import routeur as R
def q(**k):
    r = json.loads(R.traiter(json.dumps(k)))
    assert r["ok"], r.get("erreur")
    return r
e = q(module="garde_corps", action="nouveau")["etat"]
q(module="garde_corps", action="selecteur", etat=e, role="montant")
q(module="garde_corps", action="choisir", etat=e, role="montant", nom="RHS 60x40x3")
for chemin, v in ((["type"], "lisses"), (["type"], "panneau"), (["type"], "murale"), (["type"], "main_courante"), (["fixation", "type"], "laterale"),
                  (["fixation", "type"], "soudee"), (["fixation", "ancrage", "type"], "boulon"), (["methode", "mode"], "eurocode"), (["geometrie", "forme"], "U")):
    q(module="garde_corps", action="modifier", etat=e, chemin=chemin, valeur=v)
try:
    import api_web.poutre  # noqa: F401
    p = q(module="poutre", action="nouveau")["etat"]
    q(module="poutre", action="selecteur", etat=p, role="inf-0")
except ImportError:
    pass
"""
INTERDITS = ("streamlit", "reportlab", "matplotlib", "pandas", "fitz", "scipy", "PIL")


def fichiers_du_moteur() -> list:
    """Les fichiers du dépôt que l'API charge (modules Python et données), relevés dans un sous-processus où les
    bibliothèques lourdes sont refusées : ce qui y manque manquerait aussi dans le navigateur."""
    sonde = f"""
import builtins, os, sys
RACINE = {RACINE!r}
sys.path.insert(0, RACINE); os.chdir(RACINE)
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {INTERDITS!r}:
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
{SCENARIOS}
mods = {{os.path.relpath(m.__file__, RACINE) for m in list(sys.modules.values()) if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(RACINE + os.sep)}}
print("\\n".join(sorted(mods | lus)))
"""
    p = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True, cwd=RACINE)
    if p.returncode != 0:
        raise SystemExit("Le relevé du moteur a échoué :\n" + p.stderr[-3000:])
    fichiers = {f for f in p.stdout.split() if f}
    # les paquets du site entiers (petits), pour qu'un import paresseux ne manque jamais
    for paquet in ("api_web", "beton"):
        for d, _, fs in os.walk(os.path.join(RACINE, paquet)):
            fichiers |= {os.path.relpath(os.path.join(d, f), RACINE) for f in fs if f.endswith(".py")}
    return sorted(f for f in fichiers if "__pycache__" not in f and not f.endswith(".pyc"))


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


def tete(titre: str) -> str:
    return ('<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{titre}</title>\n<link rel="icon" href="favicon.svg" type="image/svg+xml">\n'
            '<link rel="stylesheet" href="css/charte.css">\n<link rel="stylesheet" href="css/site.css">\n'
            '<script src="js/theme.js"></script>\n</head>\n')


def page_accueil() -> str:
    tuiles = ""
    for fam, nom_fam, mods in MQ.FAMILLES:
        t = ""
        for k, n in mods:
            if k in PAGES:
                t += (f'<a class="tile" href="{ADRESSES[k]}" data-nouvelle-module="{k}"><span class="modic {fam}">{MQ.MOD[k]}</span>'
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
    corps = (f'<body data-page="accueil"><div class="app rail">{rail("accueil")}<main class="main">'
             f'<div class="page accueil"><div class="col">'
             f'<header class="hello"><div><h1 class="h1">Bonjour</h1><p class="muted" id="date"></p></div><span class="spacer"></span>'
             f'<button class="btn primary big" id="ouvrir-nouvelle">{MQ.icon("plus")}Nouvelle note</button></header>'
             f'<section><div class="sechead"><span class="h2">Reprendre</span><span class="muted" id="nb-notes"></span></div>'
             f'<div class="resume" id="recentes"></div></section>'
             f'<section><div class="sechead"><span class="h2">Modules</span><span class="muted">Garde-corps et poutres béton armé · les autres arrivent</span></div>{tuiles}</section>'
             f'</div><aside class="side"><div class="card"><div class="sechead"><span class="h3">Chantiers</span></div><div id="chantiers"></div></div>'
             f'<div class="card note-locale"><div class="h3">Où sont mes notes ?</div><p>Pour l\'instant, chaque note est enregistrée '
             f'<b>dans ce navigateur</b>, sur ce PC. Le bouton <b>.stab</b> d\'une note l\'exporte en fichier ; les comptes du bureau et le partage '
             f'viendront ensuite.</p><label class="btn sm" for="importer">{MQ.icon("upload")}Importer un fichier .stab</label>'
             f'<input type="file" id="importer" accept=".stab,.json" hidden></div></aside></div></main></div>{modal}'
             f'<script type="module" src="js/accueil.js"></script></body>\n</html>\n')
    return tete("Stab Méthodes") + corps


def page_module(cle: str, titre: str) -> str:
    """L'établi d'un module : la toile du dessin sur toute la largeur, les données et le résultat en cadres flottants ;
    tout le contenu vient du moteur (le travailleur Pyodide)."""
    fam, nom = _module(cle)
    top = (f'<header class="top"><div class="crumbs"><a href="{ACCUEIL}">{MQ.icon("house")}</a>{MQ.icon("chevron-right")}'
           f'<span id="fil-chantier" class="muted"></span><span id="fil-sep">{MQ.icon("chevron-right")}</span>'
           f'<input id="nom-note" class="nomnote" value="" aria-label="Nom de la note" spellcheck="false"></div>'
           f'<span class="tag draft"><span class="dot"></span>Brouillon</span>'
           f'<span class="spacer"></span><span class="saved" id="enregistre">{MQ.icon("circle-check")}<span class="txt">—</span></span>'
           f'<label class="btn ghost" title="Ouvrir un fichier .stab">{MQ.icon("upload")}Ouvrir<input type="file" id="ouvrir" accept=".stab,.json" hidden></label>'
           f'<button class="btn" id="exporter" title="Exporter la note en fichier .stab">{MQ.icon("download")}.stab</button>'
           f'<button class="btn" id="imprimer" title="Imprimer l\'écran">{MQ.icon("printer")}Imprimer</button>'
           f'<button class="btn primary" disabled title="La note PDF arrive bientôt sur le site (elle reste dans Streamlit)">{MQ.icon("file-down")}Note PDF</button></header>')
    gauche = (f'<div class="phead"><span class="h3">Données</span><span class="calcule" id="calcule"></span>'
              f'<button class="pcoll" data-coll="g" title="Replier les données">{MQ.icon("panel-left")}</button></div>'
              f'<div class="pbody" id="donnees"><div class="attente">Chargement…</div></div>')
    droite = (f'<div class="phead"><span class="h3">Résultat</span><span id="statut"></span>'
              f'<button class="pcoll" data-coll="d" title="Replier le résultat">{MQ.icon("panel-right")}</button></div>'
              f'<div id="resultat" class="resultat"><div class="pbody"><div class="attente">Chargement…</div></div></div>')
    toile = (f'<section class="canvas"><div class="ctool"><div class="seg" id="vues"></div></div>'
             f'<div class="stage" id="stage"></div></section>')
    demarrage = (f'<div class="demarrage" id="demarrage"><div class="carte"><div class="logo">{MQ.LOGO}</div><b>Chargement du moteur de calcul</b>'
                 f'<span id="demarrage-etape">Python démarre dans le navigateur…</span><div class="barre"><i></i></div>'
                 f'<small>La première ouverture prend quelques secondes ; ensuite, chaque calcul est immédiat.</small></div></div>')
    corps = (f'<body data-page="module" data-module="{cle}" data-titre="{MQ.esc(nom)}"><div class="app rail">{rail("modules")}<main class="main">{top}'
             f'<div class="bench" data-tab="donnees">{toile}<section class="panel left">{gauche}</section>'
             f'<section class="panel right">{droite}</section><div id="volets"></div>{demarrage}</div></main></div>'
             f'<div class="toast" id="toast" hidden></div>'
             f'<script type="module" src="js/etabli.js"></script></body>\n</html>\n')
    return tete(f"{titre} · Stab Méthodes") + corps


def favicon() -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#2B4BCF"/>'
            '<g transform="translate(4 4)">' + MQ.LOGO.replace("<svg", "<svg width=\"24\" height=\"24\"") + '</g></svg>')


def ecrire(chemin: str, contenu, binaire: bool = False) -> None:
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "wb" if binaire else "w", **({} if binaire else {"encoding": "utf-8"})) as f:
        f.write(contenu)


def construire() -> dict:
    """Écrit tout ; rend le manifeste du paquet."""
    shutil.copyfile(os.path.join(MAQUETTES, "charte.css"), os.path.join(ICI, "css", "charte.css"))
    ecrire(os.path.join(ICI, "index.html"), page_accueil())
    ecrire(os.path.join(ICI, "garde-corps.html"), page_module("garde_corps", "Garde-corps"))
    ecrire(os.path.join(ICI, "poutre.html"), page_module("poutre", "Poutres béton armé"))
    ecrire(os.path.join(ICI, "favicon.svg"), favicon())
    fichiers = fichiers_du_moteur()
    z = paquet(fichiers)
    ecrire(os.path.join(ICI, "moteur", "moteur.zip"), z, binaire=True)
    manifeste = {"pyodide": PYODIDE, "sha256": hashlib.sha256(z).hexdigest(), "fichiers": fichiers}
    ecrire(os.path.join(ICI, "moteur", "manifeste.json"), json.dumps(manifeste, ensure_ascii=False, indent=1) + "\n")
    return manifeste


if __name__ == "__main__":
    m = construire()
    print(f"site/web : 3 pages, paquet du moteur {len(m['fichiers'])} fichiers, "
          f"{os.path.getsize(os.path.join(ICI, 'moteur', 'moteur.zip')) // 1024} ko (sha256 {m['sha256'][:12]})")
