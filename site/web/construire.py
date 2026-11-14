# -*- coding: utf-8 -*-
"""Construit le site « Stab Méthodes » (site/web), un site statique qui calcule DANS le navigateur :

- les pages : l'accueil (index.html) et l'établi de chaque module du registre (api_web/registre.py : un fichier
  api_web/<clé>.py qui déclare MODULE est un module ; son établi est <PAGE>.html) ; js/modules.js, la même liste pour
  les scripts de l'écran ;
- la charte : css/charte.css est la COPIE de site/maquettes/charte.css (celle que le bureau a validée) ;
- le PAQUET DU MOTEUR, moteur/moteur.zip : les fichiers Python que le site exécute (api_web, le moteur de chaque
  module, leurs données), relevés en faisant tourner l'API sur des cas variés, bibliothèques lourdes refusées.
  Le paquet est reproductible (dates fixes, ordre trié) : tests/test_site_web.py le reconstruit et le compare ;
- les APERÇUS, moteur/apercus/<clé>.json : la réponse « nouveau » de chaque établi, calculée à la construction (l'écran
  la peint pendant que le moteur démarre ; tests/test_apercus.py la recalcule) ;
- la VERSION du site : une empreinte des fichiers publiés et des pages, la même dans chaque page (<meta
  name="stab-version">) et dans le manifeste (js/app.js ne passe d'une page à l'autre sans rechargement qu'à version égale).
  Toutes les pages n'ont qu'un script de module, js/app.js.

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
# le moteur de PLANS de RBAR pour le navigateur (outils/vendre_rbar_plans.py ; js/plans-travailleur.js en porte une COPIE)
RBAR_PLANS = "vendor/rbar-plans-web-0.4.43"
DATE_ZIP = (2026, 1, 1, 0, 0, 0)
# les modules du site : le registre (un fichier api_web/<clé>.py qui déclare MODULE) ; les autres tuiles de l'accueil
# disent « bientôt »
sys.path.insert(0, RACINE)
from api_web import registre as REG  # noqa: E402

_FAMILLES = {k for _, _, mods in MQ.FAMILLES for k, _ in mods}
# Les modules ABSORBÉS par un autre (docs/site/DALLES_FUSION.md § 2.4) : ancien module → module qui le sert. Leur page
# est une REDIRECTION vers celle du module qui les sert, qui garde l'adresse (?note=…, ?nom=…) : une ancienne note
# s'ouvre dans le module fusionné. Ils ne sont plus au registre (leur écran est retiré : le routeur ne les sert plus), sont
# masqués de l'accueil (MASQUES, site/maquettes/construire.py), et js/modules.js les porte avec « vers » — la table
# d'alias que js/notes.js lit pour ouvrir leurs notes par la bonne page.
REDIRECTIONS = {"predalle": "dalle"}
# clé → adresse de l'établi, SANS « .html » (Cloudflare la sert ainsi) ; une clé hors des familles de l'accueil n'a
# ni nom ni icône : elle est écartée (tests/test_site_web.py le signale)
ADRESSES = {k: v for k, v in REG.pages().items() if k in _FAMILLES}
# les modules ABSORBÉS (REDIRECTIONS, ci-dessus) n'ont plus de module au registre (l'écran de l'ancien module Prédalles est
# retiré le 13/11/2026, lot 2) : leur adresse reste leur clé, leur page une redirection
ADRESSES.update({k: k for k in REDIRECTIONS if k in _FAMILLES})
PAGES = {k: v + ".html" for k, v in ADRESSES.items()}
ACCUEIL = "./"
# L'établi COMPACT (l'arborescence de la note à gauche, le dessin et son en-tête au centre, un bloc de résultats par
# section à droite), sur ces modules seulement (docs/site/REFONTE_ETABLI.md) : les modules béton armé, leurs fonctions
# aux mêmes places (api_web/etabli_compact.py) ; les autres gardent l'établi d'avant. (Ouvrir un calcul n'écrit rien —
# décision D1 — vaut pour TOUS les modules depuis le 10/11/2026 : chaque carte de l'accueil ouvre le calcul directement,
# la note est provisoire jusqu'à la première modification, « Enregistrer comme note… » est sur toute page d'établi :
# ``page_module``.)
ETABLI_COMPACT = ("poutre", "dalle", "corniere", "bois", "age", "taux", "fondation_grue", "garde_corps", "flambement", "profil", "sol", "soudures", "plancher", "assemblages")
# … et parmi eux, ceux dont la colonne de gauche est COMPACTE (docs/site/DALLES_FUSION.md § 1, revue avec l'ingénieur
# le 09/11/2026) : ni titre « Données », ni en-tête « POUTRES (n) », ni arborescence — l'en-tête du panneau porte la
# LIGNE de l'élément ouvert (son nom qu'un clic modifie, la liste des éléments, copier, supprimer : le moteur la rend
# avec ``etabli_compact.arbre(…, compact=True)``, js/etabli.js la pose dans l'emplacement ``.et-elt``) à côté
# d'« Annuler » et « Rétablir » (tests/test_site_web.py lie ce tuple à ce que le moteur rend).
ARBRE_COMPACT = ("dalle", "poutre", "corniere", "bois", "age", "taux", "fondation_grue", "garde_corps", "flambement", "profil", "sol", "soudures", "plancher", "assemblages")
# … et parmi eux, ceux dont « Annuler », « Rétablir » et « Replier les données » sont en haut à gauche du DESSIN, près du
# titre de la section, au PC et à la tablette (la dalle, demande du propriétaire du 13/11/2026 — docs/site/DALLES_FUSION.md
# § 8.2 ; la poutre au lot 4, la discipline commune E15) : la ligne de l'élément garde la place pour son nom, à droite le
# cadenas, copier, supprimer. Au téléphone, le dessin est un onglet à part : les trois restent dans l'en-tête des données
# (css/etabli.css choisit, selon le format).
OUTILS_AU_DESSIN = ("dalle", "poutre")
# les BIBLIOTHÈQUES du site (docs/site/BIBLIOTHEQUES.md) : une page par famille, hors du registre des modules (une
# consultation n'est pas une note) ; la clé est celle que le routeur reçoit (action « consulter », « bibliotheque ») :
# clé → (module de rendu, titre, famille). La Bibliothèque Acier (BIB5) porte le catalogue des profilés : le module
# « Catalogue des profilés » est masqué de l'accueil (MASQUES), sa page et ses notes restent servies.
BIBLIOTHEQUES = {"beton": ("api_web.biblio_beton", "Béton armé", "beton"),
                 "acier": ("api_web.biblio_acier", "Acier", "acier"),
                 "autres": ("api_web.biblio_autres", "Autres", "autres")}
PAGES_BIBLIOTHEQUES = {k: f"bibliotheque-{k}.html" for k in BIBLIOTHEQUES}
ADRESSES_BIBLIOTHEQUES = {k: f"bibliotheque-{k}" for k in BIBLIOTHEQUES}
# les modules masqués de l'accueil (site/maquettes/construire.py, MASQUES) : servis, plus listés
MASQUES = frozenset(MQ.MASQUES)
# Les pages du BUREAU (docs/site/PROJETS_EQUIPE.md) : les projets et l'équipe. Ni module ni bibliothèque : des écrans que
# js/app.js monte comme l'accueil (body data-page), sans moteur. Leur feuille : css/bureau.css (l'accueil la charge aussi :
# il porte les projets, les pastilles et le fil d'activité).
PAGES_BUREAU = {"projets": "projets.html", "equipe": "equipe.html"}
ADRESSES_BUREAU = {"projets": "projets", "equipe": "equipe"}
FEUILLE_BUREAU = "css/bureau.css"
# les icônes que ces écrans posent eux-mêmes (js/vues_bureau.js les lit dans la page : rien n'est chargé d'ailleurs)
ICONES_BUREAU = ("plus", "x", "check", "pencil-line", "trash-2", "ellipsis", "chevron-right", "chevron-left", "folder", "folder-open",
                 "archive", "search", "user-plus", "users", "file-plus-2", "file-text", "layers", "clock", "mail", "upload",
                 "badge-check", "shield-check", "book-open")


def _bibliotheques_de(fam: str) -> list:
    """Les bibliothèques d'une famille de l'accueil : [(clé, titre, rubriques = [(titre du menu, icône)])] — les rubriques
    sont celles du menu de la bibliothèque (``RUBRIQUES`` de son module de rendu : une source)."""
    out = []
    for k, (chemin, titre, famille) in BIBLIOTHEQUES.items():
        if famille == fam:
            B = importlib.import_module(chemin)
            out.append((k, titre, [(t, ic) for _, t, _, ic in B.RUBRIQUES]))
    return out
# … et parmi eux, ceux dont le site fait la NOTE PDF (le module rend « telecharger note_pdf », docs/site/MODULES_SITE.md
# § 4 quinquies — décision de l'ingénieur du 09/11/2026 : les notes PDF de Streamlit passent sur le site) : leur bouton
# « Note PDF » s'allume sur leur page, compacte ou non, aux trois emplacements (``allumer_note_pdf``) ; les autres le
# gardent grisé, « bientôt ». Chaque clé a sa note « riche » dans ``NOTES_RELEVE`` (le relevé du paquet de la note).
NOTE_PDF = ("poutre", "dalle", "garde_corps", "soudures", "plancher", "assemblages", "corniere", "fondation_grue")
# le relevé du paquet rejoue le PARCOURS de chaque module (api_web/parcours.py) : ce qu'un chemin rare importe doit
# être dans le paquet
MAX_RELEVE = 120
# les paquets de calcul propres au site, empaquetés ENTIERS (un import paresseux n'y manque jamais) ; acier et plancher,
# qui portent aussi les écrans Streamlit, sont relevés fichier par fichier
PAQUETS_ENTIERS = ("api_web", "beton", "bois", "sol", "flambement", "rdm")
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
{parcours}
if echecs:
    sys.stderr.write("\\n".join(echecs[:20])); sys.exit(3)
mods = {{os.path.relpath(m.__file__, RACINE) for m in list(sys.modules.values()) if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(RACINE + os.sep)}}
print(json.dumps({{"fichiers": sorted(mods | lus), "numpy": "numpy" in sys.modules}}))
"""
# le parcours d'un MODULE : celui de api_web/parcours.py
_PARCOURS_MODULE = """
for desc, ok, err in P.parcourir({cle!r}, R.traiter, maximum={maximum}):
    if not ok:
        echecs.append({cle!r} + " · " + desc + " : " + str(err))
"""
# le parcours d'une BIBLIOTHÈQUE (clé « biblio_<k> ») : chaque rubrique de son menu, avec ses défauts — ce que sa page et
# le panneau d'une note demandent (action « consulter »)
_PARCOURS_BIBLIOTHEQUE = """
import importlib
for rub in importlib.import_module(R.BIBLIOTHEQUES[{biblio!r}]).CLES:
    r = json.loads(R.traiter(json.dumps({{"action": "consulter", "bibliotheque": {biblio!r}, "rubrique": rub}})))
    if not r.get("ok"):
        echecs.append({cle!r} + " · " + rub + " : " + str(r.get("erreur")))
"""
# la clé d'une bibliothèque dans le relevé et dans « avec_numpy » du manifeste : celle que le travailleur lit dans une
# requête « consulter » (CONTRAT § 1 : "biblio_" + req.bibliotheque)
PREFIXE_BIBLIOTHEQUE = "biblio_"
# un fichier qui importe numpy, même au fond d'une fonction que le parcours n'a pas suivie
_IMPORTE_NUMPY = re.compile(r"^\s*(?:import\s+numpy\b|from\s+numpy\b)", re.M)


def _releve_un(cle: str) -> dict:
    """Le relevé d'UN module (ou d'une bibliothèque, « biblio_<k> »), dans son propre sous-processus (bibliothèques
    lourdes refusées) : les fichiers du dépôt que son parcours charge, et s'il charge numpy — un module seul, pour que
    numpy chargé par un autre ne compte pas."""
    if cle.startswith(PREFIXE_BIBLIOTHEQUE):
        parcours_ = _PARCOURS_BIBLIOTHEQUE.format(cle=cle, biblio=cle[len(PREFIXE_BIBLIOTHEQUE):])
    else:
        parcours_ = _PARCOURS_MODULE.format(cle=cle, maximum=MAX_RELEVE)
    sonde = _SONDE.format(racine=RACINE, interdits=INTERDITS, parcours=parcours_.strip())
    p = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True, cwd=RACINE)
    if p.returncode != 0:
        raise SystemExit(f"Le relevé du moteur a échoué ({cle}) :\n" + p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def releve(modules=None) -> dict:
    """Le relevé du paquet du moteur : ``fichiers`` (les fichiers du dépôt que l'API charge, modules Python et données,
    relevés en rejouant le parcours de chaque module là où les bibliothèques lourdes sont refusées : ce qui y manque
    manquerait aussi dans le navigateur) et ``avec_numpy`` (les modules qui chargent numpy : le travailleur ne le charge
    que pour eux — il pèse une à deux secondes au démarrage ; un fichier qui l'importe, même dans une fonction que le
    parcours n'a pas suivie, compte). Un sous-processus par module, quatre à la fois. Les BIBLIOTHÈQUES sont relevées
    aussi, toujours (clés « biblio_<k> » : la Bibliothèque Acier charge numpy — le travailleur le charge avant elle)."""
    # (une REDIRECTION n'a pas de module à relever : son écran est retiré, sa page ne charge rien — etablis())
    cles = list(modules or etablis()) + [PREFIXE_BIBLIOTHEQUE + k for k in BIBLIOTHEQUES]
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
            "avec_numpy": sorted(avec_numpy),
            # ce que CHAQUE module charge (la version d'un module en est l'empreinte : ``versions_modules``)
            "par_module": {cle: sorted(f for f in r["fichiers"] if "__pycache__" not in f and not f.endswith(".pyc"))
                           for cle, r in par_module.items()}}


# ------------------------------------------------------------------ la VERSION d'un module (cahier des charges § 4)
# « Vérifié » dit qu'un administrateur a relu la MÉTHODE d'un module ; la pastille doit TOMBER quand le calcul change —
# sinon elle certifie un calcul que personne n'a relu. La version d'un module est donc l'EMPREINTE de ce que son calcul
# charge : les fichiers du moteur que son parcours fait lire (``releve`` : ses paquets de calcul, leurs données, son
# adaptateur api_web et ses dessins), SANS l'habillage commun à tous les modules (HORS_VERSION : y changer une icône ne
# redemande pas dix-sept vérifications). Dans un fichier Python, ni une ligne vide ni une ligne de commentaire ne
# comptent ; tout le reste compte, à l'octet (une donnée, un coefficient, une formule, un texte affiché).
# js/versions.js porte ces empreintes ; js/verifications.js compare celle d'aujourd'hui à celle du jour de la vérification.
HORS_VERSION = frozenset({"api_web/__init__.py", "api_web/routeur.py", "api_web/registre.py", "api_web/parcours.py",
                          "api_web/icones.py", "api_web/commun.py", "api_web/etabli_compact.py", "api_web/fiches.py"})


def _calcul_seul(chemin: str, octets: bytes) -> bytes:
    """Ce qui COMPTE d'un fichier pour la version d'un module : un fichier Python sans ses lignes vides ni ses lignes de
    commentaire (les fins de ligne et les blancs de fin de ligne ne comptent pas non plus) ; tout autre fichier, tel quel."""
    if not chemin.endswith(".py"):
        return octets
    lignes = (l.rstrip() for l in octets.decode("utf-8").replace("\r\n", "\n").split("\n"))
    return "\n".join(l for l in lignes if l.strip() and not l.lstrip().startswith("#")).encode("utf-8")


def _lire_depot(chemin: str) -> bytes:
    with open(os.path.join(RACINE, chemin), "rb") as f:
        return f.read()


def empreinte_module(fichiers, lire=_lire_depot) -> str:
    """L'empreinte (16 caractères hexadécimaux) du calcul d'un module : ses fichiers (hors HORS_VERSION), dans l'ordre de
    leurs noms, chacun réduit à ce qui compte (``_calcul_seul``). ``lire`` : pour les essais."""
    h = hashlib.sha256()
    for f in sorted(set(fichiers) - HORS_VERSION):
        h.update(f.encode("utf-8") + b"\0" + _calcul_seul(f, lire(f)) + b"\0")
    return h.hexdigest()[:16]


def versions_modules(par_module: dict) -> dict:
    """{clé: empreinte} des ÉTABLIS (les modules de ADRESSES, sans les redirections : un module absorbé n'a plus d'écran)."""
    return {k: empreinte_module(par_module[k]) for k in ADRESSES if k not in REDIRECTIONS and k in par_module}


def versions_js(versions: dict) -> str:
    return ("// Écrit par site/web/construire.py (versions_modules) : l'empreinte du CALCUL de chaque module — ne pas modifier.\n"
            "// Une vérification posée sur un module (js/verifications.js) vaut pour CETTE empreinte : le calcul change, elle tombe.\n"
            f"export const VERSIONS = {json.dumps(versions, ensure_ascii=False, sort_keys=True)};\n")


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
# La note de calcul PDF des modules qui en font une (NOTE_PDF) est faite dans le navigateur par le générateur VALIDÉ de
# l'écran Streamlit (modules/export_pdf*.py, acier/**/note.py, plancher/charges/note.py, ndc_pdf/) et reportlab, en
# Python pur. Ce paquet pèse trop pour le démarrage : le travailleur ne le charge qu'à la première demande de note
# (moteur/note.zip, empreinte dans le manifeste) et le déballe À CÔTÉ du moteur (/moteur, js/travailleur.js) — un fichier
# du paquet du moteur n'a donc pas à y être. Ce qu'il porte est RELEVÉ : pour CHAQUE module de NOTE_PDF, une vraie note
# « riche » (NOTES_RELEVE), demandée au routeur comme le fait le navigateur, dans un sous-processus où fontTools, le
# Pillow réel, les accélérateurs C de reportlab et les bibliothèques lourdes sont refusés, et où ne se lisent que les
# polices du paquet (NOTE_POLICES) : les fichiers du dépôt qu'elle charge (hors du paquet du moteur), ceux de reportlab,
# les polices lues, et si elle charge numpy (le travailleur le charge alors AVANT la note : manifeste « note.avec_numpy »).
NOTE_SOURCES = os.path.join(RACINE, "site", "note_pdf")
NOTE_POLICES = ("Carlito-Regular.ttf", "Carlito-Bold.ttf", "Carlito-Italic.ttf", "Poppins-Medium.ttf",
                "DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
# refusés pendant la note (comme dans le navigateur : rien de tout cela n'y est) ; numpy est PERMIS — le travailleur le
# charge à la demande — et son chargement est relevé
NOTE_REFUS = ("fontTools", "_rl_accel", "rl_accel", "streamlit", "matplotlib", "pandas", "scipy", "pymupdf", "fitz")
# La note « riche » de chaque module de NOTE_PDF : du code qui pose ``etat`` par les MÊMES fonctions que l'écran (les
# états proposés par les lots qui ont porté chaque note, docs/site/MODULES_SITE.md § 4 quinquies) ; elle doit passer
# par les chemins rares de sa note (plusieurs éléments, les deux types, les deux modes…), sinon un fichier manquerait
# au navigateur. tests/test_site_web.py rejoue chacune, paquets déballés seuls.
NOTES_RELEVE = {
    "poutre": """
from api_web import poutre as P
etat = P.nouvel_etat()
for ch, v in ((["sections", 0, "M_inf"], 120), (["sections", 0, "V"], 80), (["sections", 0, "inf", "+"], 1), (["sections", 0, "etriers", "+"], "epingle"),
              (["sections", "+"], 1), (["sections", 1, "M_sup"], 90), (["sections", 0, "laterales", "n"], 1)):
    P.modifier(etat, ch, v)
P.gerer_elements(etat, "dupliquer")
""",
    # une dalle coulée (renfort, treillis par nappe, deux sections) puis une prédalle (renfort secondaire)
    "dalle": """
from api_web import dalle as P
etat = P.nouvel_etat()
for ch, v in ((["sections", 0, "My_inf"], 28), (["sections", 0, "Mx_sup"], 12), (["sections", 0, "V"], 60), (["enrobage"], 0),
              (["sections", 0, "faces", "inf_x", "couches", "+"], 1), (["sections", 0, "faces", "inf_x", "couches", 1, "type"], "barres"),
              (["sections", 0, "nappes", "sup", "treillis"], "8/10/100/150"), (["sections", "+"], 1), (["sections", 1, "My_sup"], 30)):
    assert P.modifier(etat, ch, v) is None, ch
P.gerer_elements(etat, "ajouter")
for ch, v in ((["type"], "predalle"), (["sections", 0, "My_inf"], 22), (["sections", 0, "faces", "inf_y", "couches", "+"], 1)):
    assert P.modifier(etat, ch, v) is None, ch
""",
    # une fixation latérale à l'Eurocode, puis un garde-corps mural, puis un garde-corps en U avec « Cumuler qh et qv » :
    # sa note a une PAGE DE SUITE (le récapitulatif de la page 1, la bibliothèque de la page 2 : acier/garde_corps/note.Suite),
    # un chemin que les deux premiers ne prennent pas (relecture du 09/11/2026)
    "garde_corps": """
from api_web import garde_corps as G
etat = G.nouvel_etat()
for ch, v in ((["fixation", "type"], "laterale"), (["methode", "mode"], "eurocode")):
    assert G.modifier(etat, ch, v) is None, ch
G.gerer_elements(etat, "ajouter")
assert G.modifier(etat, ["type"], "murale") is None
G.gerer_elements(etat, "ajouter")
for ch, v in ((["geometrie", "forme"], "U"), (["charges", "cumul"], True)):
    assert G.modifier(etat, ch, v) is None, ch
""",
    # le cas C3 de départ, puis une C5 sur IPE 300 à l'ELS
    "soudures": """
from api_web import soudures as S
etat = S.nouvel_etat()
S.gerer_elements(etat, "ajouter")
for ch, v in ((["config"], "C5"), (["params", "profil"], "IPE 300"), (["methode", "mode"], "els")):
    assert S.modifier(etat, ch, v) is None, ch
""",
    # le hourdis d'exemple, puis la dalle coulée d'exemple (ses fiches réunies)
    "plancher": """
import copy
from api_web import plancher as PL
etat, d = PL.exemple("hourdis"), PL.exemple("dalle")
etat["cas"] += copy.deepcopy(d["cas"])
for k in ("engins", "repartiteurs", "hourdis"):
    etat["fiches"][k] += [f for f in d["fiches"][k] if f not in etat["fiches"][k]]
""",
    # les sept configurations et le pied de poteau dans ses deux modes (tous les générateurs)
    "assemblages": """
from api_web import note_assemblages as NA
etat = NA.etat_releve()
""",
    # deux cornières d'appui (intégration du 10/11/2026, docs/acier/CORNIERE_MOTEUR.md § 12) : la première en configuration
    # A, méthode historique, avec R_ELU saisie et la largeur d'appui a (la ligne L_eff) ; la seconde retournée (B, mode
    # équilibre verrouillé) avec l'ancrage à Y = 45 et sans jeu dy — la section nette, les formules de B, la garde du site
    "corniere": """
from api_web import corniere as K
etat = K.nouvel_etat()
for ch, v in ((["R"], "42,76"), (["R_ELU"], "68,9"), (["a"], "100")):
    assert K.modifier(etat, ch, v) is None, ch
K.gerer_elements(etat, "ajouter")
for ch, v in ((["configuration"], "B"), (["dy"], "0"), (["Y"], "45"), (["support"], "poutre"), (["support_haut"], "160"),
              (["support_bas"], "150"), (["type_ancrage"], "chimique")):
    assert K.modifier(etat, ch, v) is None, ch
K.gerer_elements(etat, "ajouter")
for ch, v in ((["etat_limite"], "ELU"), (["R"], "60"), (["mode"], "equilibre"), (["type_ancrage"], "traversant"), (["entraxe"], "80"),
              (["rondelles"], "2"), (["profondeur"], "200")):
    assert K.modifier(etat, ch, v) is None, ch
""",
    # trois fondations de grue (docs/beton/FONDATIONS_GRUE.md § 6) : le classeur Greenwood avec une paroi de pieux sécants du
    # côté b, un second groupe de barres et un septième cas de la grue ; la variante 4 pieux avec un pieu PAS OK (les
    # bandes rouges, la conclusion « non conforme ») ; la variante 8 pieux — les trois schémas, leurs dessins, leurs tableaux
    "fondation_grue": """
from api_web import fondation_grue as FG
etat = FG.nouvel_etat()
for ch, v in ((["exemple"], "GWW"), (["cote_b"], "paroi"), (["support", "inf", "1", "n"], "2"), (["charges", "cas", "+"], "1"),
              (["charges", "cas", "6", "B"], "900"), (["charges", "cas", "6", "A"], "350")):
    assert FG.modifier(etat, ch, v) is None, ch
FG.gerer_elements(etat, "ajouter")
for ch, v in ((["exemple"], "VAR4"), (["pieux", "capacite"], "500")):
    assert FG.modifier(etat, ch, v) is None, ch
FG.gerer_elements(etat, "ajouter")
assert FG.modifier(etat, ["exemple"], "VAR8") is None
""",
}
_SONDE_NOTE = """
import base64, builtins, json, os, sys
RACINE, STUB, CLE, REFUS = {racine!r}, {stub!r}, {cle!r}, {refus!r}
sys.path.insert(0, STUB); sys.path.insert(1, RACINE); os.chdir(RACINE)
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in REFUS or name == "reportlab.lib._rl_accel":
            raise ImportError("refusé dans le navigateur : " + name)
sys.meta_path.insert(0, Refus())
lus = set(); _open = builtins.open
def espion(f, *a, **k):
    mode = a[0] if a else k.get("mode", "r")
    try:
        if not any(c in str(mode) for c in "wax+"): lus.add(os.path.abspath(f))
    except Exception: pass
    return _open(f, *a, **k)
builtins.open = espion
from api_web import routeur as R
{etat}
r = json.loads(R.traiter(json.dumps({{"module": CLE, "action": "telecharger", "etat": etat, "quoi": "note_pdf",
                                      "infos": {{"nom": "relevé", "chantier": "Essai", "indice": "B"}}}})))
if not r.get("ok"):
    sys.stderr.write(CLE + " : la note est refusée — " + str(r.get("erreur"))); sys.exit(3)
pdf = base64.b64decode(r["fichier"]["base64"])
if pdf[:5] != b"%PDF-":
    sys.stderr.write(CLE + " : la note n'est pas un PDF"); sys.exit(3)
import reportlab
base = os.path.dirname(os.path.dirname(os.path.abspath(reportlab.__file__)))
mods = {{os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}}
rl = sorted(os.path.relpath(f, base) for f in mods | lus if f.startswith(os.path.join(base, "reportlab") + os.sep) and not f.endswith(".pyc"))
depot = sorted(os.path.relpath(f, RACINE) for f in mods | lus
               if f.startswith(RACINE + os.sep) and not f.startswith(STUB + os.sep) and os.path.isfile(f)
               and "__pycache__" not in f and not f.endswith((".pyc", ".ttf", ".otf"))
               and not os.path.relpath(f, RACINE).startswith(os.path.join("outils", "suivi_tests") + os.sep))
polices = sorted(os.path.basename(f) for f in lus if f.lower().endswith((".ttf", ".otf", ".afm", ".pfb")))
print(json.dumps({{"base": base, "reportlab": rl, "version": reportlab.Version, "depot": depot, "polices": polices,
                   "numpy": "numpy" in sys.modules, "octets": len(pdf)}}))
"""


def _releve_note_un(cle: str) -> dict:
    """La note riche d'UN module (NOTES_RELEVE), dans son propre sous-processus, comme dans le navigateur : les seules
    polices du paquet (un dossier à part, NDC_FONT_DIR), NOTE_REFUS refusés, le Pillow factice."""
    import tempfile
    polices = tempfile.mkdtemp(prefix="note_polices_")
    try:
        for f in NOTE_POLICES:
            shutil.copyfile(os.path.join(RACINE, "ndc_pdf", "fonts", f), os.path.join(polices, f))
        sonde = _SONDE_NOTE.format(racine=RACINE, stub=NOTE_SOURCES, cle=cle, refus=NOTE_REFUS, etat=NOTES_RELEVE[cle].strip())
        p = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True, cwd=RACINE,
                           env={**os.environ, "NDC_FONT_DIR": polices})
    finally:
        shutil.rmtree(polices, ignore_errors=True)
    if p.returncode != 0:
        raise SystemExit(f"Le relevé de la note PDF a échoué ({cle}) :\n" + p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def releve_note(modules=None) -> dict:
    """Ce que la note PDF charge, module par module (NOTE_PDF par défaut ; un sous-processus chacun, quatre à la fois) :
    {"base", "version", "reportlab": [fichiers], "depot": [fichiers du dépôt], "polices": [lues], "avec_numpy":
    [modules dont la note charge numpy], "par_module": {clé: {"depot", "octets", …}}}."""
    cles = list(modules or NOTE_PDF)
    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as ex:
        par = dict(zip(cles, ex.map(_releve_note_un, cles)))
    bases = {r["base"] for r in par.values()}
    if len(bases) != 1:
        raise SystemExit(f"Le relevé de la note PDF : reportlab vient de plusieurs dossiers ({sorted(bases)})")
    hors = sorted({p for r in par.values() for p in r["polices"]} - set(NOTE_POLICES))
    if hors:
        raise SystemExit(f"Le relevé de la note PDF : des polices hors du paquet sont lues ({hors})")
    return {"base": bases.pop(), "version": next(iter(par.values()))["version"],
            "reportlab": sorted({f for r in par.values() for f in r["reportlab"]}),
            "depot": sorted({f for r in par.values() for f in r["depot"]}),
            "polices": sorted({p for r in par.values() for p in r["polices"]}),
            "avec_numpy": sorted(k for k, r in par.items() if r["numpy"]),
            "par_module": par}


def _dans_le_moteur(f: str, moteur) -> bool:
    """Un fichier du dépôt que le paquet du MOTEUR porte déjà (le travailleur déballe les deux paquets au même endroit) :
    tout ce qui est sous un paquet empaqueté entier, et les fichiers relevés du moteur quand on les donne."""
    return f.split("/")[0] in PAQUETS_ENTIERS or (moteur is not None and f in moteur)


def paquet_note(moteur=None, modules=None) -> tuple:
    """(octets du zip, description) : les fichiers du dépôt que les notes chargent (relevés, hors de ce que le paquet du
    MOTEUR porte déjà : ``moteur`` = ses fichiers, à défaut les seuls paquets entiers), tout ndc_pdf/, reportlab relevé
    (et sa licence), les polices de la mise en page, le Pillow factice. Reproductible : mêmes fichiers → mêmes octets."""
    rel = releve_note(modules)
    moteur = None if moteur is None else set(moteur)
    entrees = [(f, os.path.join(RACINE, f)) for f in rel["depot"] if not _dans_le_moteur(f, moteur)]
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
    return tampon.getvalue(), {"reportlab": rel["version"], "fichiers": len(set(entrees)), "modules": sorted(rel["par_module"]),
                               "avec_numpy": rel["avec_numpy"]}


# ------------------------------------------------------------------ les morceaux des pages
def _module(cle: str):
    for fam, nom_fam, mods in MQ.FAMILLES:
        for k, nom in mods:
            if k == cle:
                return fam, nom
    raise KeyError(cle)


def marque_verifie(cle: str) -> str:
    """La marque « vérifié » d'un module (cahier des charges § 4), là où le module est nommé : la tuile de l'accueil, le
    volet des modules, « Nouvelle note ». Écrite CACHÉE : js/marques.js la montre si un administrateur a déclaré le module
    vérifié ET que son calcul n'a pas changé depuis (js/verifications.js), et y écrit par qui et quand."""
    return f'<span class="ok" data-verifie="{cle}" hidden>{MQ.icon("badge-check")}</span>'


def legende_verifie(texte: str = "vérifié par un administrateur", deux: str = "par deux") -> str:
    """La légende de la marque, là où on la voit : ARGENTÉE, un administrateur l'a vérifié ; DORÉE, deux administrateurs
    différents (demande du propriétaire, 14/11/2026). Les couleurs sont celles de la marque (css/site.css)."""
    i = MQ.icon("badge-check")
    return (f'<span class="okleg"><span class="okleg-i" data-niveau="argent">{i}</span><span class="okleg-t">{MQ.esc(texte)}</span> '
            f'<span class="okleg-i" data-niveau="or">{i}</span><span class="okleg-t">{MQ.esc(deux)}</span></span>')


def rail(actif: str) -> str:
    """La barre de cinq icônes. Projets et Équipe mènent à leurs pages (11/11/2026 : elles étaient « bientôt ») ;
    Paramètres viendra avec les comptes du bureau."""
    def it(ic, nom, cle, attr):
        return f'<a class="navitem{" on" if cle == actif else ""}" data-tip="{nom}"{attr}>{MQ.icon(ic)}</a>'
    bientot = ' aria-disabled="true" data-bientot="1"'
    return (f'<nav class="nav"><a class="brand" href="{ACCUEIL}" title="Accueil"><div class="logo">{MQ.LOGO}</div></a>'
            + it("house", "Accueil", "accueil", f' href="{ACCUEIL}"') + it("folder", "Projets", "projets", f' href="{ADRESSES_BUREAU["projets"]}"')
            + it("layout-grid", "Modules", "modules", ' data-modules="1"') + it("users", "Équipe", "equipe", f' href="{ADRESSES_BUREAU["equipe"]}"')
            + it("settings", "Paramètres · bientôt", "parametres", bientot)
            # en bas : le thème, puis la PASTILLE de la personne du poste (js/marques.js l'écrit ; elle mène à l'équipe)
            + f'<div style="margin-top:auto;display:flex;flex-direction:column;align-items:center;gap:10px">'
              f'<a class="navitem" data-theme-toggle="1" data-tip="Thème clair ou sombre">{MQ.icon("sun-moon")}</a>'
              f'<a class="navitem rail-moi" href="{ADRESSES_BUREAU["equipe"]}" data-moi-rail="1" data-tip="Mon profil et l\'équipe" aria-label="Mon profil et l\'équipe"></a>'
              f'</div></nav>' + modpanel())


def mnav(actif: str) -> str:
    """Les onglets du bas du téléphone : les cinq entrées de la barre du PC, dans le même ordre (cahier des charges
    § 1 bis) ; Paramètres viendra avec les comptes."""
    def it(ic, nom, cle, attr):
        return f'<a class="{"on" if cle == actif else ""}"{attr}>{MQ.icon(ic)}<span>{nom}</span></a>'
    bientot = ' aria-disabled="true" data-bientot="1" title="Bientôt"'
    return ('<nav class="mnav">' + it("house", "Accueil", "accueil", f' href="{ACCUEIL}"') + it("folder", "Projets", "projets", f' href="{ADRESSES_BUREAU["projets"]}"')
            + it("layout-grid", "Modules", "modules", ' data-modules="1" role="button"') + it("users", "Équipe", "equipe", f' href="{ADRESSES_BUREAU["equipe"]}"')
            + it("settings", "Paramètres", "parametres", bientot) + "</nav>")


def modpanel() -> str:
    """Le volet « Modules » du rail : les BIBLIOTHÈQUES d'abord (un groupe à part), puis les calculs de chaque famille ;
    les modules masqués (MASQUES) n'y sont pas."""
    corps = '<div class="navlabel">Bibliothèques</div>'
    for fam, _nom_fam, _mods in MQ.FAMILLES:
        for k, titre, _rubriques in _bibliotheques_de(fam):
            corps += f'<a class="navitem" href="{ADRESSES_BIBLIOTHEQUES[k]}"><span class="modic {fam}">{MQ.icon("book-open")}</span><span>{MQ.esc(titre)}</span></a>'
    for fam, nom_fam, mods in MQ.FAMILLES:
        corps += f'<div class="navlabel">{nom_fam}</div>'
        for k, n in mods:
            if k in MASQUES:
                continue
            if k in PAGES:
                corps += f'<a class="navitem" href="{ADRESSES[k]}"><span class="modic {fam}">{MQ.MOD[k]}</span><span>{MQ.esc(n)}</span>{marque_verifie(k)}</a>'
            else:
                corps += (f'<a class="navitem bientot" aria-disabled="true"><span class="modic {fam}">{MQ.MOD[k]}</span><span>{MQ.esc(n)}</span>'
                          f'<span class="tag draft">bientôt</span></a>')
    return (f'<aside class="modpanel"><div class="mp-h"><div class="h2">Modules</div>{legende_verifie("vérifié")}'
            f'<a class="btn icon ghost" data-modules="1" title="Fermer">{MQ.icon("x")}</a></div>{corps}</aside>')


# La VERSION du site (S4-D3, C7) : chaque page la porte dans cette balise, la même sur toutes les pages d'un site
# construit ; js/app.js compare celle de la page cible à celle du document courant (une autre → navigation ordinaire).
# Les gabarits l'écrivent VIDE ; ``construire`` calcule la version (``version_site``) sur les fichiers publiés et sur
# les pages ainsi rendues — aucune circularité —, puis la pose (``poser_version``).
META_VERSION = '<meta name="stab-version" content="{}">'
META_VERSION_VIDE = META_VERSION.format("")
# le SEUL script de module de toutes les pages (S4-D1) : l'application monte l'écran que le <body> nomme (data-page) ;
# js/theme.js reste un script classique en tête (le thème avant le premier rendu)
SCRIPT_APP = '<script type="module" src="js/app.js"></script>'
# Le FILET du démarrage (10/11/2026, après la mise en ligne du lot S4 : sur une tablette, l'écran restait sur
# « Chargement… » sans un mot quand js/app.js ne démarrait pas) : un message posé dans CHAQUE page, invisible, qu'une
# animation CSS rend visible au bout de SECOURS_S secondes, et que la feuille EN LIGNE de l'en-tête cache dès que
# l'application a monté un écran (body[data-montage], posé par js/app.js). Rien n'y dépend d'un fichier à charger : il
# parle même quand tous les scripts et toutes les feuilles du site sont refusés (session de connexion expirée, fichiers
# d'une autre version, navigateur qui ne les lit pas). La politique de sécurité admet les styles en ligne
# (style-src 'unsafe-inline') ; aucun script en ligne. Le lien « Recharger la page » est un lien ordinaire vers la page
# même (href vide) : il marche sans script.
SECOURS_S = 12
STYLE_SECOURS = ('<style>.secours-demarrage{position:fixed;left:16px;right:16px;bottom:16px;z-index:2147483000;margin:0;'
                 'padding:14px 16px;border-radius:10px;background:#b42318;color:#fff;font:600 15px/1.45 system-ui,sans-serif;'
                 'box-shadow:0 6px 24px rgba(0,0,0,.3);opacity:0;visibility:hidden;'
                 f'animation:secours-demarrage 0s linear {SECOURS_S}s forwards}}'
                 '.secours-demarrage a{color:#fff;text-decoration:underline;display:block;width:max-content;min-height:44px;'
                 'line-height:44px;margin-top:4px}'
                 '@keyframes secours-demarrage{to{opacity:1;visibility:visible}}'
                 'body[data-montage] .secours-demarrage{display:none}</style>')
SECOURS = ('<p class="secours-demarrage" role="alert">Le site ne démarre pas sur cet appareil. Rechargez la page ; si cela '
           'continue, reconnectez-vous (la session de connexion a peut-être expiré).'
           '<a href="">Recharger la page</a></p>')


def tete(titre: str, *feuilles: str) -> str:
    """L'en-tête d'une page : la version du site (vide ici, posée par ``construire``), la charte, site.css, puis les
    feuilles données, dans cet ordre (une vide est ignorée)."""
    lien = "".join(f'<link rel="stylesheet" href="{c}">\n' for c in feuilles if c)
    return ('<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + META_VERSION_VIDE + '\n'
            f'<title>{titre}</title>\n<link rel="icon" href="favicon.svg" type="image/svg+xml">\n'
            '<link rel="stylesheet" href="css/charte.css">\n<link rel="stylesheet" href="css/site.css">\n' + lien
            + '<script src="js/theme.js"></script>\n' + STYLE_SECOURS + '\n</head>\n')


def _sous_titre_modules() -> str:
    tous = [k for _, _, mods in MQ.FAMILLES for k, _ in mods if k not in MASQUES]
    n = sum(k in PAGES for k in tous)
    return "Tous les modules du bureau" if n == len(tous) else f"{n} modules sur {len(tous)} · les autres arrivent"


def _carte_bibliotheque(fam: str, k: str, titre: str, rubriques: list) -> str:
    """La carte de la BIBLIOTHÈQUE d'une famille, à l'accueil (docs/site/BIBLIOTHEQUES.md § 2.1, révision du 09/11/2026) :
    l'icône d'un livre et « Bibliothèque » / son titre en tête, ses rubriques en liste dessous (les titres de son menu,
    avec leurs icônes). Toute la carte est le lien. Une seule carte, trois formats : css/site.css la pose ÉTROITE et HAUTE
    à gauche des modules de sa famille (PC, tablette), en tête de la liste de sa famille, ses rubriques sur une ligne
    (téléphone)."""
    lis = "".join(f'<li>{MQ.icon(ic)}<span>{MQ.esc(t)}</span></li>' for t, ic in rubriques)
    return (f'<a class="tile biblio" href="{ADRESSES_BIBLIOTHEQUES[k]}" data-bibliotheque="{k}"><div class="bib-c">'
            f'<div class="bib-h"><span class="modic {fam}">{MQ.icon("book-open")}</span>'
            f'<b class="nm"><span class="bib-k">Bibliothèque</span> <span class="bib-t">{MQ.esc(titre)}</span></b></div>'
            f'<ul class="bib-r">{lis}</ul></div></a>')


def page_accueil() -> str:
    """L'accueil : les notes à reprendre, puis les CALCULS par famille ; la BIBLIOTHÈQUE d'une famille est une carte à
    part, ÉTROITE et HAUTE, à GAUCHE du bloc de ses modules, qui s'alignent à sa DROITE dans leur propre grille (demande
    de l'ingénieur du 09/11/2026) — le bloc ``.tiles.fam-b`` : la carte, puis la grille ``.tiles`` des modules ; une
    famille sans bibliothèque garde sa grille ``.tiles`` seule. Les modules masqués ne sont pas listés
    (docs/site/BIBLIOTHEQUES.md § 2.1)."""
    tuiles = ""
    for fam, nom_fam, mods in MQ.FAMILLES:
        cartes = [_carte_bibliotheque(fam, *b) for b in _bibliotheques_de(fam)]
        # le gabarit ne place qu'UNE bibliothèque par famille (la première colonne ; les modules prennent les suivantes)
        if len(cartes) > 1:
            raise ValueError(f"famille « {fam} » : {len(cartes)} bibliothèques, l'accueil n'en place qu'une par famille")
        t = ""
        for k, n in mods:
            if k in MASQUES:
                continue
            if k in PAGES:
                # la carte ouvre le calcul LUI-MÊME, sans fenêtre (décision D1, tous les modules : la note est provisoire
                # tant que rien n'est modifié) ; « Nouvelle note » reste le chemin qui nomme et range d'abord
                t += (f'<a class="tile" href="{ADRESSES[k]}" data-module="{k}"><span class="modic {fam}">{MQ.MOD[k]}</span>'
                      f'<span class="nm">{MQ.esc(n)}</span>{marque_verifie(k)}</a>')
            else:
                t += (f'<div class="tile bientot" title="Ce module n\'est pas encore sur le site (il reste dans Streamlit)">'
                      f'<span class="modic {fam}">{MQ.MOD[k]}</span><span class="nm">{MQ.esc(n)}</span><span class="tag draft">bientôt</span></div>')
        grille = f'<div class="tiles fam-b">{cartes[0]}<div class="tiles">{t}</div></div>' if cartes else f'<div class="tiles">{t}</div>'
        tuiles += f'<div class="famille" data-fam="{fam}"><div class="kicker">{nom_fam}</div>{grille}</div>'
    # (l'avatar de la personne du poste mène à l'équipe ; « Bonjour » prend son prénom dès qu'elle s'est présentée)
    corps = (f'<body data-page="accueil"><div class="app rail">{rail("accueil")}<main class="main">{mhead("Stab Méthodes")}'
             f'<div class="page accueil"><div class="col"><div class="et-memoire" id="et-memoire" role="alert" hidden></div>'
             f'<header class="hello"><a class="bu-moi" id="moi" href="{ADRESSES_BUREAU["equipe"]}" title="Mon profil et l\'équipe"></a>'
             f'<div><h1 class="h1" id="bonjour">Bonjour</h1><p class="muted" id="date"></p></div><span class="spacer"></span>'
             # la RECHERCHE (maquette validée) : un module, un projet, une note — Ctrl K y mène, Entrée ouvre le premier
             f'<div class="bu-cherche"><form class="bigsearch" id="ac-recherche-form" role="search" autocomplete="off">{MQ.icon("search")}'
             f'<input id="ac-recherche" type="search" maxlength="80" placeholder="Rechercher un module, un projet, une note…" '
             f'aria-label="Rechercher un module, un projet, une note" aria-controls="ac-resultats" aria-expanded="false" enterkeyhint="search">'
             f'<span class="kbd" aria-hidden="true">Ctrl K</span></form>'
             f'<div class="bu-resultats" id="ac-resultats" role="listbox" aria-label="Résultats de la recherche" hidden></div></div>'
             f'<button class="btn primary big" id="ouvrir-nouvelle">{MQ.icon("plus")}Nouvelle note</button></header>'
             f'<section><div class="sechead"><span class="h2">Reprendre</span><span class="muted" id="nb-notes"></span>'
             f'<a class="btn ghost sm" href="{ADRESSES_BUREAU["projets"]}">Tout voir{MQ.icon("chevron-right")}</a></div>'
             f'<div class="resume" id="recentes"></div></section>'
             # (la légende dit ce que veut dire la marque d'une tuile ; le sous-titre ne reste que s'il annonce des modules à venir)
             f'<section><div class="sechead"><span class="h2">Modules</span>'
             f'{"" if _sous_titre_modules() == "Tous les modules du bureau" else f'<span class="muted">{_sous_titre_modules()}</span>'}'
             f'{legende_verifie()}</div>{tuiles}</section>'
             f'</div><aside class="side">'
             # les PROJETS (c'était « Chantiers » : la liste seule) : en créer un d'ici — son nom, Entrée —, les ouvrir
             f'<div class="card bu-projets"><div class="sechead"><span class="h3">Projets</span>'
             f'<a class="btn ghost sm" href="{ADRESSES_BUREAU["projets"]}">Tout voir{MQ.icon("chevron-right")}</a></div>'
             f'{champ_creer("ac-projet", "Nouveau projet : le nom du chantier", "plus")}<div id="chantiers"></div></div>'
             # l'ACTIVITÉ : qui a fait quoi (le journal) ; tant que personne ne s'est présenté sur ce poste, on le demande ici
             f'<div class="card bu-activite"><div class="sechead"><span class="h3">Activité</span>'
             f'<a class="btn ghost sm" href="{ADRESSES_BUREAU["equipe"]}">Équipe{MQ.icon("chevron-right")}</a></div>'
             f'<div id="ac-qui"></div><div id="activite"></div></div>'
             f'<div class="card note-locale"><div class="h3">Où sont mes notes ?</div><p>Pour l\'instant, chaque note est enregistrée '
             f'<b>dans ce navigateur</b>, sur ce PC — comme les projets, l\'équipe et le journal. Le bouton <b>.stab</b> d\'une note l\'exporte '
             f'en fichier ; les comptes du bureau et le partage entre les postes viendront ensuite.</p>'
             f'<label class="btn sm" for="importer">{MQ.icon("upload")}Importer un fichier .stab</label>'
             f'<input type="file" id="importer" accept=".stab,.json" hidden></div></aside></div>{mnav("accueil")}</main></div>{fenetre_nouvelle_note()}'
             f'{icones_bureau()}<div class="toast" id="toast" role="status" hidden></div>'
             f'{SECOURS}{SCRIPT_APP}</body>\n</html>\n')
    return tete("Stab Méthodes", FEUILLE_BUREAU) + corps


def mhead(titre: str) -> str:
    """Au téléphone : l'en-tête d'une page du bureau (le nom, le thème) ; les cinq entrées de la barre sont en bas."""
    return (f'<header class="mhead"><div class="logo">{MQ.LOGO}</div><span class="ttl">{MQ.esc(titre)}</span><span class="spacer"></span>'
            f'<button type="button" class="btn icon ghost" data-theme-toggle="1" title="Thème clair ou sombre">{MQ.icon("sun-moon")}</button></header>')


def champ_creer(ident: str, invite: str, icone: str) -> str:
    """Un champ qui CRÉE à la touche Entrée (un projet, un sous-projet) : pas de fenêtre, pas d'étape — le nom, Entrée.
    Le bouton « Créer » ne paraît que quand le champ porte un nom (js/vues_bureau.js)."""
    return (f'<form class="bu-creer" id="{ident}-form" autocomplete="off"><div class="field text">{MQ.icon(icone)}'
            f'<input id="{ident}" maxlength="80" placeholder="{MQ.esc(invite)}" aria-label="{MQ.esc(invite)}" enterkeyhint="done"></div>'
            f'<button type="submit" class="btn sm primary" id="{ident}-creer" hidden>Créer</button></form>')


def fenetre_nouvelle_note() -> str:
    """La fenêtre « Nouvelle note » (l'accueil, la page des projets : js/nouvelle.js) : le projet, le sous-projet, le nom,
    puis le module. Le projet et le sous-projet déjà connus sont proposés ; on peut en taper un nouveau."""
    options = "".join(f'<button type="button" class="mopt" data-module="{k}"><span class="modic {_module(k)[0]}">{MQ.MOD[k]}</span>{MQ.esc(_module(k)[1])}{marque_verifie(k)}</button>'
                      for k in PAGES if k not in MASQUES)
    return (f'<div class="modal" id="nouvelle" hidden><div class="mcard"><div class="mh"><span class="h2">Nouvelle note</span>'
            f'<button class="btn icon ghost" data-fermer-modal="1" title="Fermer">{MQ.icon("x")}</button></div>'
            f'<div class="mb"><div class="mrow3"><label><span>Projet</span><input id="nn-chantier" list="nn-chantiers" maxlength="80" placeholder="le chantier, ex. Réalex"></label>'
            f'<label><span>Sous-projet</span><input id="nn-dossier" list="nn-dossiers" maxlength="80" placeholder="ex. Étude de la dalle"></label>'
            f'<label><span>Nom de la note</span><input id="nn-nom" placeholder="ex. Poutres R+1"></label></div>'
            f'<datalist id="nn-chantiers"></datalist><datalist id="nn-dossiers"></datalist>'
            f'<div class="mlabel">Module</div><div class="mgrid">{options}</div></div>'
            f'<div class="mf"><button class="btn" data-fermer-modal="1">Annuler</button>'
            f'<button class="btn primary" id="nn-creer" disabled>{MQ.icon("plus")}Créer la note</button></div></div></div>')


def icones_bureau() -> str:
    """Les icônes que les écrans du bureau posent eux-mêmes — celles de ICONES_BUREAU et le pictogramme de chaque module —,
    déposées dans la page, inertes (<template>) : js/vues_bureau.js les y lit. Rien n'est chargé d'ailleurs (R5)."""
    return ('<div id="icones" hidden>' + "".join(f'<template data-icone="{n}">{MQ.icon(n)}</template>' for n in ICONES_BUREAU)
            + "".join(f'<template data-icone-module="{k}">{MQ.MOD[k]}</template>' for k in PAGES) + '</div>')


def page_projets() -> str:
    """La page PROJETS (docs/site/PROJETS_EQUIPE.md § 4.2, la maquette site/maquettes/pages/projets.html) : l'ARBRE des
    projets et de leurs sous-projets à gauche — son champ cherche et crée —, les NOTES du projet choisi au centre, son
    ACTIVITÉ à droite. Un gabarit vide : js/ecran_projets.js le remplit depuis les notes et le registre de ce navigateur."""
    corps = (f'<body data-page="projets"><div class="app rail">{rail("projets")}<main class="main">'
             # UNE barre pour les trois formats (R1) : au téléphone, la charte cache « .top » — celle-ci reste (css/bureau.css,
             # « .top.bu-top »), avec le logo et le thème à la place de l'en-tête « .mhead » : le fil d'Ariane, l'import d'un
             # .stab et « Nouvelle note » y sont donc atteignables aussi au téléphone
             f'<header class="top bu-top"><div class="logo bu-tel">{MQ.LOGO}</div><div class="crumbs" id="pj-fil">{MQ.icon("folder")}<b>Projets</b></div>'
             f'<span class="spacer"></span>'
             f'<label class="btn" for="importer" title="Importer un fichier .stab">{MQ.icon("upload")}<span class="lbl">Importer .stab</span></label>'
             f'<input type="file" id="importer" accept=".stab,.json" hidden>'
             f'<button type="button" class="btn primary" id="ouvrir-nouvelle" title="Nouvelle note">{MQ.icon("plus")}<span class="lbl">Nouvelle note</span></button>'
             f'<button type="button" class="btn icon ghost bu-tel" data-theme-toggle="1" title="Thème clair ou sombre">{MQ.icon("sun-moon")}</button></header>'
             f'<div class="proj bu-proj" id="pj" data-vue="arbre">'
             f'<aside class="tree" id="pj-arbre">{champ_creer("pj-champ", "Chercher ou créer un projet", "search")}'
             f'<nav id="pj-noeuds" aria-label="Les projets et leurs sous-projets"></nav></aside>'
             f'<section class="list" id="pj-liste" aria-live="polite"></section>'
             f'<aside class="drawer" id="pj-volet"></aside></div>{mnav("projets")}</main></div>{fenetre_nouvelle_note()}'
             f'{icones_bureau()}<div class="toast" id="toast" role="status" hidden></div>'
             f'{SECOURS}{SCRIPT_APP}</body>\n</html>\n')
    return tete("Projets · Stab Méthodes", FEUILLE_BUREAU) + corps


def page_equipe() -> str:
    """La page ÉQUIPE (docs/site/PROJETS_EQUIPE.md § 4.3, la maquette site/maquettes/pages/equipe.html) : la personne du
    poste (son nom, son e-mail, ses initiales dans leur pastille), les membres, et le journal d'activité entier. Un gabarit
    vide : js/ecran_equipe.js le remplit depuis ce navigateur."""
    corps = (f'<body data-page="equipe"><div class="app rail">{rail("equipe")}<main class="main">'
             f'<header class="top"><div class="crumbs">{MQ.icon("users")}<b>Équipe du bureau</b></div><span class="spacer"></span>'
             f'<span class="muted bu-nb" id="eq-nb"></span></header>{mhead("Équipe")}'
             f'<div class="team bu-team"><section class="tmain">'
             f'<div class="card" id="eq-moi"></div>'
             f'<div class="card flush"><div class="sechead"><div class="h3" id="eq-titre">Membres</div></div><div id="eq-membres"></div>'
             f'<form class="invrow bu-ajout" id="eq-form" autocomplete="off">'
             f'<div class="field text">{MQ.icon("user-plus")}<input id="eq-nom" maxlength="80" placeholder="Prénom et nom" aria-label="Prénom et nom du nouveau membre"></div>'
             f'<div class="field text bu-ini"><input id="eq-ini" maxlength="5" placeholder="Initiales" aria-label="Initiales (1 à 5)" autocapitalize="characters"></div>'
             f'<div class="field text">{MQ.icon("mail")}<input id="eq-email" type="email" maxlength="120" placeholder="adresse e-mail (facultatif)" aria-label="Adresse e-mail"></div>'
             f'<button type="submit" class="btn primary">{MQ.icon("plus")}Ajouter</button></form>'
             f'<p class="hint bu-erreur" id="eq-erreur" role="alert" hidden></p></div>'
             # la VÉRIFICATION DES MODULES (cahier des charges § 4, la maquette) : un administrateur déclare un module vérifié
             f'<div class="card flush" id="eq-verif"><div class="sechead"><div class="h3">Vérification des modules</div>'
             f'<span class="muted bu-nb" id="eq-verif-nb"></span></div><div id="eq-modules"></div></div>'
             f'<div class="card"><div class="h3">Qui peut faire quoi</div><div class="rights">'
             f'<div><span class="ic">{MQ.icon("shield-check")}</span><div><b>Administrateur</b>Nomme les autres administrateurs et déclare un module vérifié. '
             f'Le premier inscrit du poste l\'est ; il y en a toujours un.</div></div>'
             f'<div><span class="ic">{MQ.icon("users")}</span><div><b>Membre</b>Crée ses notes et ses projets ; ses initiales les signent.</div></div>'
             f'<div><span class="ic">{MQ.icon("badge-check")}</span><div><b>Module vérifié</b>Un administrateur a relu sa méthode : on s\'en sert sans la revérifier — '
             f'chaque note reste à relire par son auteur. Si le calcul du module change, la marque tombe : il est à revérifier.</div></div>'
             f'</div></div>'
             f'<div class="card note-locale"><div class="h3">Où est enregistrée l\'équipe ?</div><p>Pour l\'instant, l\'équipe et le journal '
             f'sont enregistrés <b>dans ce navigateur</b>, comme les notes : chaque poste a les siens — les rôles et les modules vérifiés '
             f'aussi. Les initiales d\'une personne voyagent avec ses notes (le fichier <b>.stab</b> les porte). Les invitations par e-mail, '
             f'les mots de passe et le partage entre les postes viendront avec les comptes du bureau.</p></div>'
             f'</section><aside class="bu-journal"><div class="card"><div class="sechead"><div class="h3">Journal d\'activité</div>'
             f'<span id="eq-filtre"></span></div><div id="eq-journal"></div></div></aside></div>{mnav("equipe")}</main></div>'
             f'{icones_bureau()}<div class="toast" id="toast" role="status" hidden></div>'
             f'{SECOURS}{SCRIPT_APP}</body>\n</html>\n')
    return tete("Équipe · Stab Méthodes", FEUILLE_BUREAU) + corps


def _bouton(icone: str, libelle: str, attrs: str, cls: str = "btn") -> str:
    """Un bouton de la barre du haut : l'icône et son nom ; le nom se replie sous 1 100 px (la tablette), le titre
    reste au survol et pour les lecteurs d'écran."""
    return f'<button type="button" class="{cls}"{attrs}>{MQ.icon(icone)}<span class="lbl">{libelle}</span></button>'


PDF_BIENTOT = "La note PDF arrive bientôt sur le site (elle reste dans Streamlit)"


def boutons_note_pdf(actif: bool) -> dict:
    """Le bouton « Note PDF » à ses TROIS emplacements (R1 : un composant, trois formats) — la barre du haut (PC,
    tablette), le menu ⋯ et la barre du bas du téléphone : ``actif`` (le module est dans NOTE_PDF) → ``data-action="pdf"``,
    que js/etabli.js envoie au moteur (« telecharger », quoi = « note_pdf ») ; sinon grisé, « bientôt » (le clic le dit).
    La page le dit aussi : body ``data-note-pdf="1"`` (que js/etabli.js lit). Toute page d'établi, compacte ou non, les
    prend ici (``page_module``)."""
    if actif:
        return {"top": _bouton("file-down", "Note PDF", ' data-action="pdf" title="Préparer la note de calcul PDF"', "btn primary"),
                "menu": f'<button type="button" class="mi" data-action="pdf">{MQ.icon("file-down")}Note PDF</button>',
                "bas": _bouton("file-down", "Note PDF", ' data-action="pdf"', "btn primary")}
    return {"top": _bouton("file-down", "Note PDF", f' disabled title="{PDF_BIENTOT}"', "btn primary"),
            "menu": (f'<button type="button" class="mi" data-action="pdf" aria-disabled="true">{MQ.icon("file-down")}Note PDF'
                     '<span class="tag draft">bientôt</span></button>'),
            "bas": _bouton("file-down", "Note PDF", ' data-action="pdf" aria-disabled="true"', "btn primary")}


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
    formats ; js/etabli.js l'écrit. Il dit aussi qu'une note demandée par l'adresse (``?note=…``) n'est pas dans ce
    navigateur, avec « Importer .stab » (une étiquette du champ #ouvrir) : rien n'est créé à sa place."""
    return (f'<div class="et-alerte" id="et-alerte" role="alert" hidden><span class="et-alerte-ic">{MQ.icon("triangle-alert")}</span>'
            '<span class="et-alerte-txt"></span>'
            f'<button type="button" class="btn sm primary et-alerte-stab" data-action="exporter">{MQ.icon("download")}Exporter .stab</button>'
            f'<label class="btn sm primary et-alerte-importer" for="ouvrir" hidden>{MQ.icon("upload")}Importer .stab</label>'
            f'<button type="button" class="btn icon ghost et-alerte-x" data-fermer-alerte="1" title="Fermer le message" '
            f'aria-label="Fermer le message">{MQ.icon("x")}</button></div>')


def etiquette_note() -> str:
    """L'étiquette de la note dans la barre du haut (PC, tablette), à la place de l'ancienne étiquette figée « Brouillon »
    (décision D9) : js/etabli.js y écrit l'INDICE réel de la note (« Indice B »), ou « Non enregistrée » tant que la note est
    provisoire (rien n'est écrit tant que rien n'est modifié, décision D1). Au téléphone, la même phrase s'écrit en tête
    de la 2ᵉ ligne de l'en-tête (#fil-m) : js/etabli.js la compose une fois (``identite``) pour les deux."""
    return '<span class="tag draft et-ind" id="et-ind"></span>'


def bouton_ranger() -> str:
    """« Enregistrer comme note… » dans la barre du haut (PC, tablette ; son nom se replie en icône sous 1 100 px) : sur
    TOUTE page d'établi (décision D1, étendue à tous les modules le 10/11/2026) — js/rangement.js l'écrit « Classer la
    note… » une fois la note écrite. Au téléphone, l'entrée du menu ⋯ (``entree_ranger``)."""
    return (f'<button type="button" class="btn sm et-ranger" data-ranger="1" title="Nommer la note et la ranger dans un chantier">'
            f'{MQ.icon("folder")}<span class="lbl">Enregistrer comme note…</span></button>')


def entree_ranger() -> str:
    """« Enregistrer comme note… » dans le menu ⋯ du téléphone : le même bouton que la barre du haut (``bouton_ranger``)."""
    return f'<button type="button" class="mi" data-ranger="1">{MQ.icon("folder")}<span class="et-mi-ranger">Enregistrer comme note…</span></button>'


def fenetre_rangement() -> str:
    """La fenêtre « Enregistrer comme note » (js/rangement.js) : le nom, le chantier, le dossier (texte, avec les
    chantiers et dossiers déjà connus). Une seule, aux trois formats."""
    return ('<div class="et-rangement" id="et-rangement" role="dialog" aria-modal="true" aria-labelledby="et-rg-titre" hidden>'
            '<form class="et-rg-carte" id="et-rg-form"><div class="et-rg-h"><b id="et-rg-titre">Enregistrer comme note</b>'
            f'<button type="button" class="btn icon ghost" data-fermer-rangement="1" aria-label="Fermer">{MQ.icon("x")}</button></div>'
            '<label class="et-rg-l">Nom de la note<input id="et-rg-nom" class="field-in" autocomplete="off" spellcheck="false" maxlength="200"></label>'
            # (11/11/2026) le PROJET et le SOUS-PROJET — les mêmes mots, les mêmes listes et la même longueur que la page des
            # projets et « Nouvelle note » (c'était « Chantier » et « Dossier » : les champs des notes gardent ces noms)
            '<label class="et-rg-l">Projet<input id="et-rg-chantier" list="et-rg-chantiers" class="field-in" autocomplete="off" spellcheck="false" maxlength="80" placeholder="Sans projet"></label>'
            '<label class="et-rg-l">Sous-projet<input id="et-rg-dossier" list="et-rg-dossiers" class="field-in" autocomplete="off" spellcheck="false" maxlength="80" placeholder="Sans sous-projet"></label>'
            '<datalist id="et-rg-chantiers"></datalist><datalist id="et-rg-dossiers"></datalist>'
            '<p class="et-rg-aide">La note est enregistrée dans ce navigateur ; « Exporter .stab » en garde une copie en fichier.</p>'
            '<div class="et-rg-b"><button type="button" class="btn ghost" data-fermer-rangement="1">Annuler</button>'
            '<button type="submit" class="btn primary">Enregistrer</button></div></form></div>')


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
    pdf = boutons_note_pdf(cle in NOTE_PDF)
    top = (f'<header class="top"><div class="crumbs"><a href="{ACCUEIL}" title="Accueil">{MQ.icon("house")}</a>{MQ.icon("chevron-right")}'
           f'<span id="fil-chantier" class="muted"></span><span id="fil-sep">{MQ.icon("chevron-right")}</span>'
           f'<input id="nom-note" class="nomnote" data-nom-note="1" value="" aria-label="Nom de la note" spellcheck="false"></div>'
           f'{etiquette_note()}<span class="spacer"></span>{enregistrement()}{bouton_ranger()}'
           f'<label class="btn ghost" for="ouvrir" title="Ouvrir un fichier .stab">{MQ.icon("upload")}<span class="lbl">Ouvrir</span></label>'
           + _bouton("download", ".stab", ' data-action="exporter" title="Exporter la note en fichier .stab"')
           + _bouton("printer", "Imprimer", ' data-action="imprimer" title="Imprimer l\'écran"')
           + pdf["top"] + '</header>')
    # le téléphone : l'en-tête compact (retour, nom de la note, taux, menu), son menu, les onglets, la barre du bas
    mtop = (f'<header class="mtop"><a class="btn icon ghost" href="{ACCUEIL}" title="Accueil" aria-label="Accueil">{MQ.icon("arrow-left")}</a>'
            f'<div class="t"><input id="nom-note-m" class="nomnote" data-nom-note="1" value="" aria-label="Nom de la note" spellcheck="false">'
            f'<span class="et-l2">{enregistrement()}<span id="fil-m"></span></span></div><span class="calcule"></span>'
            f'<button type="button" class="mchip" id="chip-m" data-vers-onglet="resultat" title="Voir le résultat"></button>'
            f'<button type="button" class="btn icon ghost" id="menu-m" aria-haspopup="true" aria-expanded="false" aria-controls="mmenu" '
            f'title="Plus d\'actions">{MQ.icon("ellipsis")}</button></header>')
    # (l'échec d'enregistrement reste la PREMIÈRE entrée du menu : « Enregistrer comme note… » le suit)
    mmenu = (f'<div class="mmenu" id="mmenu" hidden>{menu_enregistrement()}{entree_ranger()}<label class="mi" for="ouvrir">{MQ.icon("upload")}Ouvrir un fichier .stab</label>'
             f'<button type="button" class="mi" data-action="exporter">{MQ.icon("download")}Exporter la note en .stab</button>'
             f'<button type="button" class="mi" data-action="imprimer">{MQ.icon("printer")}Imprimer</button>'
             + pdf["menu"] +
             f'<button type="button" class="mi" data-theme-toggle="1">{MQ.icon("sun-moon")}Thème clair ou sombre</button>'
             f'<button type="button" class="mi" data-modules="1">{MQ.icon("layout-grid")}Tous les modules</button></div>')
    mtabs = ('<nav class="mtabs" aria-label="Données, dessin, résultat"><div class="seg full">'
             + "".join(f'<button type="button" data-onglet="{k}" aria-pressed="{"true" if k == "donnees" else "false"}"'
                       f'{" class=\"on\"" if k == "donnees" else ""}>{t}</button>'
                       for k, t in (("donnees", "Données"), ("dessin", "Dessin"), ("resultat", "Résultat")))
             + '</div></nav>')
    mbar = (f'<div class="mbar">{_bouton("download", "Exporter .stab", " data-action=\"exporter\"")}'
            f'{pdf["bas"]}</div>')
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
    # (plus de voile #demarrage, S4-D7 : l'écran est visible tout de suite ; le bandeau du démarrage est posé par l'écran)
    corps = (f'<body data-page="module" data-module="{cle}" data-titre="{MQ.esc(nom)}"{' data-note-pdf="1"' if cle in NOTE_PDF else ''}><div class="app rail">{rail("modules")}<main class="main">{top}'
             f'{mtop}{mmenu}{mtabs}'
             f'<div class="bench" data-tab="donnees">{toile}<section class="panel left">{gauche}</section>'
             f'<section class="panel right">{droite}</section><div id="volets"></div></div>{mbar}</main></div>'
             f'<input type="file" id="ouvrir" accept=".stab,.json" hidden>{alerte_enregistrement()}{fenetre_rangement()}'
             # la marque « Module vérifié / à vérifier » (cahier des charges § 4) : un gabarit inerte, que js/marques.js pose au
             # pied du cadre Résultat à chaque calcul — au même endroit aux trois formats (R1)
             # — une ligne qu'on LIT (pas un bouton : au doigt, 44 px de plus au pied du résultat lui prenaient une ligne) ;
             # un administrateur déclare le module vérifié sur la page Équipe
             f'<template id="et-verif-gabarit"><span class="et-verif" data-verif-module="{cle}">'
             f'<span class="et-verif-i et-verif-oui">{MQ.icon("badge-check")}</span><span class="et-verif-i et-verif-non">{MQ.icon("shield-check")}</span>'
             f'<span class="et-verif-t"></span></span></template>'
             f'<div class="toast" id="toast" role="status" hidden></div>'
             f'{SECOURS}{SCRIPT_APP}</body>\n</html>\n')
    # la bibliothèque de la famille, dans un panneau de la note (docs/site/BIBLIOTHEQUES.md § 2.4) : pour un module de
    # l'établi compact qui sait la préremplir ; sa feuille de style vient avec
    biblio = _bibliotheque_du_module(cle) if cle in ETABLI_COMPACT else None
    if cle in ETABLI_COMPACT:
        corps = compact(corps, biblio, cle in ARBRE_COMPACT, cle in OUTILS_AU_DESSIN)
    css = f"css/modules/{cle}.css"
    return tete(f"{titre} · Stab Méthodes", "css/etabli.css" if cle in ETABLI_COMPACT else "", "css/bibliotheque.css" if biblio else "",
                css if os.path.exists(os.path.join(ICI, css)) else "") + corps


def _gabarit_bibliotheque(B) -> str:
    """Le gabarit d'une bibliothèque — le menu des rubriques à gauche ; la barre de paramètres, LE TABLEAU, la bande de
    la valeur retenue et les sources à droite ; les identifiants que js/biblio.js cherche — le même dans sa page et dans
    le panneau d'une note (un composant, deux hôtes). (Le voile de démarrage que la page y ajoutait a disparu, S4-D7.)"""
    return (f'<div class="bib"><nav class="bib-menu" id="bib-menu" aria-label="Rubriques">{B.menu()}</nav>'
            f'<section class="bib-corps" id="bib-corps" aria-busy="false"><div class="bib-barre" id="bib-barre"></div>'
            f'<div class="bib-tableau" id="bib-tableau"><div class="attente">Chargement…</div></div>'
            f'<div class="bib-bande" id="bib-bande"></div><div class="bib-sources" id="bib-sources"></div></section></div>')


def _pourquoi_bibliotheque() -> str:
    """Le volet « Pourquoi ? » d'une bibliothèque (js/biblio.js l'ouvre et le remplit)."""
    return ('<div class="bib-pq" id="bib-pourquoi" role="dialog" aria-modal="true" aria-label="Pourquoi cette valeur" hidden>'
            '<div class="bib-pq-c" id="bib-pq-corps"></div></div>')


def _bibliotheque_du_module(cle: str):
    """La bibliothèque qu'un module ouvre dans son panneau (docs/site/BIBLIOTHEQUES.md § 2.4) : celle de sa famille, si
    le module sait la préremplir (``bibliotheque_prefill`` dans api_web/<cle>.py) ; sinon aucune — et pas de bouton."""
    fam = _module(cle)[0]
    cles = [k for k, (_, _, f) in BIBLIOTHEQUES.items() if f == fam]
    if not cles:
        return None
    mod = importlib.import_module(f"api_web.{cle}")
    return cles[0] if callable(getattr(mod, "bibliotheque_prefill", None)) else None


def page_bibliotheque(cle: str) -> str:
    """La page d'une BIBLIOTHÈQUE (docs/site/BIBLIOTHEQUES.md § 2.2) : à gauche le menu de ses rubriques, à droite la
    barre de paramètres, LE TABLEAU sur toute la largeur, la bande de la valeur retenue et les sources ; « Pourquoi ? »
    dans un volet. Sans les trois colonnes de l'établi, sans note : tout le HTML vient de ``consulter`` (le travailleur),
    js/bibliotheque.js le pose et tient l'adresse (?rubrique=…&…). Au téléphone, le menu devient un bandeau et la barre
    du bas est celle de l'accueil."""
    chemin, titre, _fam = BIBLIOTHEQUES[cle]
    B = importlib.import_module(chemin)
    top = (f'<header class="top"><div class="crumbs"><a href="{ACCUEIL}" title="Accueil">{MQ.icon("house")}</a>{MQ.icon("chevron-right")}'
           f'<span>Bibliothèque</span>{MQ.icon("chevron-right")}<b>{MQ.esc(titre)}</b>{MQ.icon("chevron-right")}<span id="bib-titre"></span></div>'
           f'<span class="tag pub">{MQ.icon("book-open")}Bibliothèque</span><span class="spacer"></span>'
           f'<button type="button" class="btn ghost" data-theme-toggle="1" title="Thème clair ou sombre">{MQ.icon("sun-moon")}<span class="lbl">Thème</span></button></header>')
    mhead = (f'<header class="mhead"><a class="btn icon ghost" href="{ACCUEIL}" title="Accueil" aria-label="Accueil">{MQ.icon("arrow-left")}</a>'
             f'<span class="ttl">{MQ.esc(titre)}</span><span class="spacer"></span>'
             f'<button type="button" class="btn icon ghost" data-theme-toggle="1" title="Thème clair ou sombre">{MQ.icon("sun-moon")}</button></header>')
    corps = (f'<body data-page="bibliotheque" data-bibliotheque="{cle}" data-titre="{MQ.esc(titre)}"><div class="app rail">{rail("modules")}<main class="main">{top}{mhead}'
             f'{_gabarit_bibliotheque(B)}'
             f'{mnav("modules")}</main></div>{_pourquoi_bibliotheque()}<div class="toast" id="toast" role="status" hidden></div>'
             f'{SECOURS}{SCRIPT_APP}</body>\n</html>\n')
    # une bibliothèque peut avoir sa feuille de style propre (css/bibliotheque-<clé>.css : les outils de la Bibliothèque Autres)
    propre = f"css/bibliotheque-{cle}.css"
    return tete(f"Bibliothèque {titre} · Stab Méthodes", "css/bibliotheque.css", propre if os.path.exists(os.path.join(ICI, propre)) else "") + corps


def compact(corps: str, bibliotheque=None, arbre_compact: bool = False, outils_au_dessin: bool = False) -> str:
    """L'établi COMPACT, tiré de l'établi ordinaire (les mêmes fragments du moteur, les mêmes identifiants : #donnees,
    #resultat, #volets ; js/etabli.js, js/compact.js, js/rangement.js) :
    - à gauche, les DONNÉES : les éléments de la note, puis les rubriques du module en accordéon (une ouverte à la fois),
      avec « Annuler » et « Rétablir » dans l'en-tête ; ``arbre_compact`` (ARBRE_COMPACT, docs/site/DALLES_FUSION.md
      § 1) : l'en-tête du panneau n'écrit plus le titre « Données » — il porte l'EMPLACEMENT ``.et-elt`` où js/etabli.js
      pose la ligne de l'élément ouvert que le moteur rend (nom, liste, copier, supprimer), Annuler et Rétablir à droite ;
    - au centre, le DESSIN : un clic sur une pièce l'ouvre dans les données (et une ligne des données la montre sur le
      dessin) ;
    - à droite, les RÉSULTATS seuls, compacts ; un clic ouvre le détail ;
    - les deux colonnes se replient et se REDIMENSIONNENT (une poignée sur leur bord intérieur) ;
    - (« Enregistrer comme note… » n'est plus son affaire depuis le 10/11/2026 : ``page_module`` le pose sur toute page
      d'établi, ``bouton_ranger``) ;
    - (la note PDF n'est plus son affaire : ``page_module`` allume le bouton de tout module de NOTE_PDF, compact ou non,
      par ``boutons_note_pdf``) ;
    - ``outils_au_dessin`` (OUTILS_AU_DESSIN) : « Annuler », « Rétablir » et « Replier les données » AUSSI en haut à gauche
      du dessin (``.et-outils``, au début de la barre de la toile) ; la page le dit (body data-outils-dessin) et
      css/etabli.css montre l'un ou l'autre selon le format — le dessin au PC et à la tablette, l'en-tête des données au
      téléphone (les mêmes attributs : js/etabli.js les sert tous, « [data-historique] », « [data-coll] ») ;
    - ``bibliotheque`` (la clé d'une bibliothèque, docs/site/BIBLIOTHEQUES.md § 2.4) : un bouton discret « Bibliothèque »
      dans la barre du haut et dans le menu ⋯ du téléphone, et le PANNEAU #et-bibliotheque (le gabarit de la page de la
      bibliothèque, dans une fenêtre modale large ; js/biblio_note.js) — la page le dit (body data-bibliotheque)."""
    gauche_av = ('<div class="phead"><span class="h3">Données</span><span class="calcule" id="calcule"></span>'
                 f'<button class="pcoll" data-coll="g" title="Replier les données">{MQ.icon("panel-left")}</button></div>')
    droite_av = ('<div class="phead"><span class="h3">Résultat</span><span id="statut"></span>'
                 f'<button class="pcoll" data-coll="d" title="Replier le résultat">{MQ.icon("panel-right")}</button></div>')
    gauche = ('<div class="phead">' + ('<div class="et-elt" role="group" aria-label="L\'élément ouvert"></div>' if arbre_compact else '<span class="h3">Données</span>')
              + '<span class="calcule" id="calcule"></span>'
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
    if outils_au_dessin:
        outils = ('<div class="et-outils" role="group" aria-label="Annuler, rétablir, replier les données">'
                  f'<button type="button" class="et-hist et-replier" data-coll="g" title="Replier ou déplier les données" '
                  f'aria-label="Replier ou déplier les données">{MQ.icon("panel-left")}</button>'
                  f'<button type="button" class="et-hist" data-historique="annuler" title="Annuler (Ctrl+Z)" aria-label="Annuler" disabled>{MQ.icon("undo-2")}</button>'
                  f'<button type="button" class="et-hist" data-historique="retablir" title="Rétablir (Ctrl+Maj+Z)" aria-label="Rétablir" disabled>{MQ.icon("redo-2")}</button></div>')
        barre = '<div class="ctool"><div class="seg" id="vues"></div>'
        assert corps.count(barre) == 1, barre
        corps = corps.replace(barre, '<div class="ctool">' + outils + '<div class="seg" id="vues"></div>', 1)
        corps = corps.replace('<body data-page="module" data-etabli="compact"', '<body data-page="module" data-etabli="compact" data-outils-dessin="1"', 1)
    # « ajuster » le dessin : une icône — « 100 % » laissait croire à une échelle d'impression, le dessin s'ajuste à la
    # feuille ; le pourcentage reste dans le bouton, caché par la feuille de l'établi (js/toile.js l'écrit)
    avant = '<button type="button" class="zpct" data-zoom="ajuster" title="Ajuster le dessin à la feuille">100 %</button>'
    assert corps.count(avant) == 1, avant
    corps = corps.replace(avant, '<button type="button" class="zpct" data-zoom="ajuster" title="Ajuster le dessin à la feuille" '
                          f'aria-label="Ajuster le dessin à la feuille">{MQ.icon("scan")}<span class="v">100 %</span></button>', 1)
    if bibliotheque:
        chemin_b, titre_b, _fam_b = BIBLIOTHEQUES[bibliotheque]
        B = importlib.import_module(chemin_b)
        corps = corps.replace('<body data-page="module" data-etabli="compact"', f'<body data-page="module" data-etabli="compact" data-bibliotheque="{bibliotheque}"', 1)
        # le bouton du ruban, discret, avant « Enregistrer comme note… » ; l'entrée du menu ⋯ du téléphone, après elle
        bouton_b = _bouton("book-open", "Bibliothèque", f' data-action="bibliotheque" title="Consulter la {MQ.esc(titre_b)} sans quitter la note"', "btn ghost et-biblio")
        marque = '<button type="button" class="btn sm et-ranger"'
        assert corps.count(marque) == 1
        corps = corps.replace(marque, bouton_b + marque, 1)
        marque = '<span class="et-mi-ranger">Enregistrer comme note…</span></button>'
        assert corps.count(marque) == 1
        corps = corps.replace(marque, marque + f'<button type="button" class="mi" data-action="bibliotheque">{MQ.icon("book-open")}Bibliothèque</button>', 1)
        panneau = (f'<div class="et-bib" id="et-bibliotheque" role="dialog" aria-modal="true" aria-labelledby="et-bib-titre" hidden><div class="et-bib-c">'
                   f'<div class="et-bib-h"><span class="tag pub">{MQ.icon("book-open")}Bibliothèque</span><b id="et-bib-titre">{MQ.esc(titre_b)}</b>'
                   f'<span class="sep">›</span><span id="bib-titre"></span>'
                   f'<button type="button" class="btn icon ghost" data-fermer-bibliotheque="1" aria-label="Fermer la bibliothèque" title="Fermer (Échap)">{MQ.icon("x")}</button></div>'
                   f'{_gabarit_bibliotheque(B)}{_pourquoi_bibliotheque()}</div></div>')
        corps = corps.replace('<div class="toast" id="toast"', panneau + '<div class="toast" id="toast"', 1)
    return corps


# le court lien visible d'une page de redirection (S4, CONTRAT § 2), par module servi
LIENS_REDIRECTION = {"dalle": "Ouvrir les dalles"}


def page_redirection(cle: str) -> str:
    """La page d'un module ABSORBÉ (REDIRECTIONS : « predalle » → « dalle ») : body ``data-page="redirection"
    data-vers=<adresse du module qui le sert>`` — js/app.js (le seul script de la page) y mène en GARDANT l'adresse
    (``?note=…``, ``?nom=…&chantier=…``), et dans l'application ``naviguer`` résout ``vers`` avant même de lire cette
    page. Aucun voile (S4-D7) : un court lien visible, le repli. Plus de ``<meta http-equiv="refresh">`` : il partirait
    APRÈS un échange de corps (le <head> reste) et mènerait à la page du module SANS la requête — la note perdue. Rien
    d'un tiers (R5)."""
    vers = REDIRECTIONS[cle]
    _fam, nom = _module(cle)
    _fam_v, nom_vers = _module(vers)
    corps = (f'<body data-page="redirection" data-module="{cle}" data-vers="{ADRESSES[vers]}" data-titre="{MQ.esc(nom)}">'
             f'<main class="main"><p class="redirection">Le module « {MQ.esc(nom)} » a rejoint « {MQ.esc(nom_vers)} ». '
             f'<a href="{ADRESSES[vers]}">{MQ.esc(LIENS_REDIRECTION[vers])}</a></p></main>'
             f'{SECOURS}{SCRIPT_APP}</body>\n</html>\n')
    return tete(f"{nom} · Stab Méthodes") + corps


def page(cle: str) -> str:
    """La page d'un module du registre : son établi (``page_module``) — ou sa redirection s'il est absorbé par un autre."""
    return page_redirection(cle) if cle in REDIRECTIONS else page_module(cle, _module(cle)[1])


def favicon() -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#2B4BCF"/>'
            '<g transform="translate(4 4)">' + MQ.LOGO.replace("<svg", "<svg width=\"24\" height=\"24\"") + '</g></svg>')


def ecrire(chemin: str, contenu, binaire: bool = False) -> None:
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "wb" if binaire else "w", **({} if binaire else {"encoding": "utf-8"})) as f:
        f.write(contenu)


FIXES = ("js", "css", "_headers", "favicon.svg", ".assetsignore", "wrangler.jsonc", "LICENCES.md", "LICENCE-icones.txt")


def modules_js() -> str:
    """La liste des modules pour les scripts de l'écran (adresse, nom, famille) : la même que celle des pages — les
    modules masqués y restent (une note existante s'ouvre encore), marqués « masque » ; un module absorbé par un autre
    (REDIRECTIONS) porte « vers » : la clé du module qui sert ses notes (js/notes.js, moduleServi)."""
    d = {}
    for fam, _, mods in MQ.FAMILLES:
        for k, nom in mods:
            if k in ADRESSES:
                d[k] = {"page": ADRESSES[k], "nom": nom, "famille": fam, **({"masque": True} if k in MASQUES else {}),
                        **({"vers": REDIRECTIONS[k]} if k in REDIRECTIONS else {})}
    # les BIBLIOTHÈQUES (le routage de js/app.js, le préchauffage) : clé → adresse de sa page et titre, depuis
    # ADRESSES_BIBLIOTHEQUES (tests/test_site_web.py lie cette adresse à api_web/biblio_*.PAGE, R2)
    b = {k: {"page": ADRESSES_BIBLIOTHEQUES[k], "titre": t} for k, (_c, t, _f) in BIBLIOTHEQUES.items()}
    return ("// Écrit par site/web/construire.py depuis le registre des modules (api_web/registre.py) : ne pas modifier.\n"
            f"export const MODULES = {json.dumps(d, ensure_ascii=False)};\n"
            f"export const BIBLIOTHEQUES = {json.dumps(b, ensure_ascii=False)};\n")


# ------------------------------------------------------------------ les APERÇUS (S4-D8, C12)
# Pour chaque ÉTABLI (une clé de ADRESSES sans redirection : 17), moteur/apercus/<clé>.json est la réponse de
# « nouveau » calculée par CPython à la construction, dans un processus NEUF par module (rien d'un autre module chargé),
# bibliothèques lourdes refusées comme dans le navigateur : la chaîne que rend routeur.traiter, à l'octet. L'écran la
# peint tant que le moteur démarre (cadres seulement). tests/test_apercus.py la recalcule dans un processus neuf.
APERCUS = "apercus"
_SONDE_APERCU = """
import json, os, sys
RACINE = {racine!r}
sys.path.insert(0, RACINE); os.chdir(RACINE)
_sortie, sys.stdout = sys.stdout, sys.stderr      # rien d'autre que la réponse sur la sortie
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {interdits!r}:
            raise ImportError("refusé dans le navigateur : " + name)
sys.meta_path.insert(0, Refus())
from api_web import routeur as R
_sortie.buffer.write(R.traiter(json.dumps({{"module": {cle!r}, "action": "nouveau"}})).encode("utf-8"))
_sortie.flush()
"""


def etablis() -> list:
    """Les clés des ÉTABLIS : les modules du site moins ceux qui ne sont qu'une redirection (REDIRECTIONS)."""
    return [k for k in ADRESSES if k not in REDIRECTIONS]


def verifier_apercu(cle: str, texte: str) -> None:
    """Refuse (SystemExit) un aperçu qui n'est pas une réponse COMPLÈTE du moteur : du JSON, ``ok`` vrai, ``etat`` et
    ``resultat`` présents (C12) — un aperçu incomplet peindrait un écran faux."""
    try:
        r = json.loads(texte)
    except ValueError as e:
        raise SystemExit(f"L'aperçu de « {cle} » n'est pas du JSON : {e}")
    manque = [k for k in ("etat", "resultat") if not isinstance(r, dict) or k not in r]
    if not isinstance(r, dict) or r.get("ok") is not True or manque:
        erreur = r.get("erreur") if isinstance(r, dict) else None
        raise SystemExit(f"L'aperçu de « {cle} » n'est pas une réponse complète (ok vrai, etat, resultat) : "
                         f"ok={r.get('ok') if isinstance(r, dict) else None!r}, manque {manque}, erreur {erreur!r}")


def apercu(cle: str) -> bytes:
    """L'aperçu d'UN établi : la réponse de « nouveau », calculée dans un processus neuf ; refusé s'il est incomplet."""
    sonde = _SONDE_APERCU.format(racine=RACINE, interdits=INTERDITS, cle=cle)
    p = subprocess.run([sys.executable, "-c", sonde], capture_output=True, cwd=RACINE)
    if p.returncode != 0:
        raise SystemExit(f"L'aperçu de « {cle} » a échoué :\n" + p.stderr.decode("utf-8", "replace")[-3000:])
    verifier_apercu(cle, p.stdout.decode("utf-8"))
    return p.stdout


def apercus(dossier: str) -> dict:
    """Écrit moteur/apercus/<clé>.json pour chaque établi (le dossier vidé d'abord : aucun aperçu périmé n'est publié) ;
    rend {clé: sha256 (16 hex) du fichier} — ce que le manifeste porte."""
    cible = os.path.join(dossier, "moteur", APERCUS)
    shutil.rmtree(cible, ignore_errors=True)
    cles = etablis()
    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as ex:
        octets = dict(zip(cles, ex.map(apercu, cles)))
    for cle, o in octets.items():
        ecrire(os.path.join(cible, f"{cle}.json"), o, binaire=True)
    return {cle: hashlib.sha256(o).hexdigest()[:16] for cle, o in sorted(octets.items())}


# ------------------------------------------------------------------ la VERSION du site (S4-D3, C7)
# Empreinte de tout ce qu'une mise en ligne peut changer pour l'écran : les fichiers PUBLIÉS non HTML (sous js/, css/,
# moteur/ — aperçus compris —, et favicon.svg, _headers) et les pages telles que les gabarits les rendent, la version
# VIDE (aucune circularité). Le manifeste (moteur/manifeste.json) n'y entre pas : il porte la version, réécrit en dernier.
# Le cœur Python (vendor/) n'y entre pas : son adresse change avec sa version (« immutable », _headers).
VERSION_DOSSIERS = ("js", "css", "moteur")
VERSION_FICHIERS = ("favicon.svg", "_headers")
VERSION_EXCLUS = ("moteur/manifeste.json",)


def fichiers_publies(dossier: str) -> list:
    """Les fichiers publiés non HTML qui font la version (chemins publiés, « / », triés)."""
    out = []
    for racine_ in VERSION_DOSSIERS:
        for d, ds, fs in os.walk(os.path.join(dossier, racine_)):
            ds[:] = [x for x in ds if x != "__pycache__"]
            for f in fs:
                rel = os.path.relpath(os.path.join(d, f), dossier).replace(os.sep, "/")
                if rel not in VERSION_EXCLUS and not f.endswith(".pyc"):
                    out.append(rel)
    out += [f for f in VERSION_FICHIERS if os.path.isfile(os.path.join(dossier, f))]
    return sorted(out)


def pages_rendues() -> dict:
    """Les pages du site, telles que les gabarits les rendent (version VIDE) : {fichier publié: HTML} — l'accueil, les
    deux pages du bureau (projets, équipe), une par module, une par bibliothèque."""
    pages = {"index.html": page_accueil(), PAGES_BUREAU["projets"]: page_projets(), PAGES_BUREAU["equipe"]: page_equipe()}
    pages.update({fichier: page(cle) for cle, fichier in PAGES.items()})
    pages.update({fichier: page_bibliotheque(cle) for cle, fichier in PAGES_BIBLIOTHEQUES.items()})
    return pages


def version_site(dossier: str, pages=None, publies=None) -> str:
    """La version d'un site construit : sha256 (16 hex) de la liste triée des (chemin publié, sha256 du contenu) des
    fichiers publiés non HTML de ``dossier`` (``publies``, par défaut ``fichiers_publies``) et des pages rendues
    (``pages``, par défaut ``pages_rendues()`` ; leur balise de version VIDE)."""
    pages = pages_rendues() if pages is None else pages
    entrees = []
    for rel in (fichiers_publies(dossier) if publies is None else publies):
        with open(os.path.join(dossier, rel), "rb") as f:
            entrees.append([rel, hashlib.sha256(f.read()).hexdigest()])
    for nom, html in pages.items():
        if html.count(META_VERSION_VIDE) != 1:
            raise ValueError(f"{nom} : la balise de version doit être VIDE et unique pour l'empreinte")
        entrees.append([nom, hashlib.sha256(html.encode("utf-8")).hexdigest()])
    return hashlib.sha256(json.dumps(sorted(entrees), separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def poser_version(html: str, version: str) -> str:
    """La page avec sa version (la balise vide des gabarits, remplie)."""
    if html.count(META_VERSION_VIDE) != 1 or not re.fullmatch(r"[0-9a-f]{16}", version):
        raise ValueError("poser_version : une balise vide, une version de 16 chiffres hexadécimaux")
    return html.replace(META_VERSION_VIDE, META_VERSION.format(version), 1)


def construire(dossier: str = ICI, modules=None) -> dict:
    """Écrit tout dans ``dossier`` (le site lui-même, ou une copie à part pour vérifier un module pendant que d'autres
    se construisent) ; le paquet du moteur est relevé sur ``modules`` (tous par défaut). Rend le manifeste.

    L'ordre compte (C7) : tout ce qui fait la version d'abord — la charte, js/modules.js, la favicon, les paquets, le
    manifeste, les aperçus —, PUIS la version, PUIS les pages (qui la portent), puis le manifeste réécrit avec elle."""
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
    ecrire(os.path.join(dossier, "js", "modules.js"), modules_js())
    ecrire(os.path.join(dossier, "favicon.svg"), favicon())
    rel = releve(modules)
    # la version de chaque module (l'empreinte de son calcul) : js/verifications.js y compare les vérifications posées
    ecrire(os.path.join(dossier, "js", "versions.js"), versions_js(versions_modules(rel["par_module"])))
    z = paquet(rel["fichiers"])
    ecrire(os.path.join(dossier, "moteur", "moteur.zip"), z, binaire=True)
    # « avec_numpy » : les modules dont le travailleur charge numpy dès le démarrage (les autres s'en passent ; un
    # import de numpy imprévu le fait charger à la demande, voir js/travailleur.js)
    # la note PDF : son paquet à part, chargé à la première demande (js/travailleur.js)
    # (sans ce que le paquet du moteur porte déjà : le travailleur les déballe au même endroit)
    zn, dn = paquet_note(rel["fichiers"])
    ecrire(os.path.join(dossier, "moteur", "note.zip"), zn, binaire=True)
    manifeste = {"pyodide": PYODIDE, "sha256": hashlib.sha256(z).hexdigest(), "fichiers": rel["fichiers"],
                 "avec_numpy": rel["avec_numpy"],
                 "note": {"sha256": hashlib.sha256(zn).hexdigest(), "octets": len(zn), **dn}}
    chemin_manifeste = os.path.join(dossier, "moteur", "manifeste.json")
    ecrire(chemin_manifeste, json.dumps(manifeste, ensure_ascii=False, indent=1) + "\n")
    # les aperçus des établis (moteur/apercus/<clé>.json) et leur empreinte (l'écran les lit avec ?v=<empreinte>)
    manifeste["apercus"] = apercus(dossier)
    # la version : tout ce qui la fait est écrit ; les pages la portent ; le manifeste aussi (réécrit en dernier)
    pages = pages_rendues()
    version = version_site(dossier, pages)
    for fichier, html in pages.items():
        ecrire(os.path.join(dossier, fichier), poser_version(html, version))
    manifeste["version"] = version
    ecrire(chemin_manifeste, json.dumps(manifeste, ensure_ascii=False, indent=1) + "\n")
    return manifeste


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Construit le site Stab Méthodes")
    ap.add_argument("--dossier", default=ICI, help="où écrire (par défaut : site/web)")
    ap.add_argument("--modules", default="", help="relever le paquet sur ces modules seulement (clés séparées par des virgules)")
    a = ap.parse_args()
    mods = [m for m in a.modules.split(",") if m] or None
    m = construire(a.dossier, mods)
    print(f"{os.path.relpath(os.path.abspath(a.dossier), RACINE)} : {len(PAGES) + 1 + len(PAGES_BUREAU) + len(PAGES_BIBLIOTHEQUES)} pages ({', '.join(PAGES)} ; "
          f"bureau : {', '.join(PAGES_BUREAU)} ; bibliothèques : {', '.join(PAGES_BIBLIOTHEQUES)}), paquet du moteur "
          f"{len(m['fichiers'])} fichiers, {os.path.getsize(os.path.join(a.dossier, 'moteur', 'moteur.zip')) // 1024} ko (sha256 {m['sha256'][:12]}), "
          f"{len(m['apercus'])} aperçus, version {m['version']}")
