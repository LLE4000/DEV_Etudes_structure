# -*- coding: utf-8 -*-
"""Le site « Stab Méthodes » (site/web) : ce qui doit rester VRAI entre deux constructions.

- le paquet du moteur (moteur/moteur.zip) contient EXACTEMENT les fichiers actuels du dépôt que l'API charge : un
  fichier du moteur modifié sans reconstruire le site, et ce test rougit (le navigateur calculerait avec l'ancien) ;
- le manifeste porte l'empreinte du paquet (le navigateur la vérifie avant de l'exécuter) ;
- le paquet se suffit : déballé seul, sans le dépôt et sans les bibliothèques lourdes, il calcule ;
- les pages et la charte sont celles que le générateur écrit (css/charte.css = la charte des maquettes) ;
- le cœur Python déposé (vendor) a les empreintes relevées, et la roue numpy celle du pyodide-lock officiel ;
- chaque élément que les scripts de l'écran cherchent par son identifiant existe dans sa page ;
- les trois formats : chaque établi porte l'en-tête, les onglets et la barre du téléphone, la barre du zoom et du plein
  écran ; l'accueil, les cinq entrées du bas dans l'ordre de la barre ; le seuil du téléphone (700 px) est le même
  dans la charte, css/site.css et js/etabli.js (trois copies d'un même nombre, liées ici) ;
- le travailleur ne charge numpy que pour les modules que le relevé a vus s'en servir (manifeste « avec_numpy ») ;
- un nom de fichier venu du moteur est nettoyé avant le téléchargement (js/commun.js, joué par node) ;
- la politique de sécurité : tout du site, aucun script en ligne ; la seule ouverture est de DONNÉES — connect-src permet
  EXACTEMENT les trois adresses d'Open-Meteo (la météo du lieu, importée à la demande), copies liées ici de
  beton/maturite/meteo.py et de js/meteo.js ; js/meteo.js (joué par node, fetch simulé) ne construit d'adresse que vers
  elles, n'envoie que les paramètres permis, et change chaque panne en message.

    python3 tests/test_site_web.py
"""
import fnmatch
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(RACINE, "site", "web")
sys.path.insert(0, WEB)
import construire as SW  # noqa: E402  (le générateur du site)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:240]}"))


def sha(chemin: str) -> str:
    with open(chemin, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# 1. le paquet du moteur suit le dépôt
zip_chemin = os.path.join(WEB, "moteur", "moteur.zip")
man = json.load(open(os.path.join(WEB, "moteur", "manifeste.json"), encoding="utf-8"))
RELEVE = SW.releve()
attendus = RELEVE["fichiers"]
with zipfile.ZipFile(zip_chemin) as z:
    noms = sorted(z.namelist())
    differents = [n for n in noms if n in attendus and z.read(n) != open(os.path.join(RACINE, n), "rb").read()]
chk("le paquet contient exactement les fichiers que l'API charge", noms == sorted(attendus),
    (sorted(set(attendus) - set(noms))[:5], sorted(set(noms) - set(attendus))[:5]))
chk("chaque fichier du paquet est identique à celui du dépôt (sinon : python3 site/web/construire.py)", not differents, differents[:5])
chk("le manifeste porte l'empreinte du paquet", man["sha256"] == sha(zip_chemin), (man["sha256"][:12], sha(zip_chemin)[:12]))
chk("le manifeste liste les fichiers du paquet", man["fichiers"] == noms)
# numpy pèse une à deux secondes au démarrage : le travailleur ne le charge d'emblée que pour ces modules-là
# (S4 : « avec_numpy » couvre aussi les BIBLIOTHÈQUES, sous « biblio_<k> » — la clé que le travailleur lit dans une
# requête « consulter » ; mesuré : la Bibliothèque Acier charge numpy, celle du béton non)
_CLES_NUMPY = set(SW.ADRESSES) | {SW.PREFIXE_BIBLIOTHEQUE + b for b in SW.BIBLIOTHEQUES}
chk(f"le manifeste nomme les modules et les bibliothèques (biblio_<k>) qui chargent numpy, comme le relevé ({', '.join(RELEVE['avec_numpy'])})",
    man.get("avec_numpy") == RELEVE["avec_numpy"] and set(RELEVE["avec_numpy"]) < _CLES_NUMPY
    and "biblio_acier" in RELEVE["avec_numpy"] and "biblio_beton" not in RELEVE["avec_numpy"], (man.get("avec_numpy"), RELEVE["avec_numpy"]))
TRAV = open(os.path.join(WEB, "js", "travailleur.js"), encoding="utf-8").read()
chk("le travailleur lit « avec_numpy » et guette un import de numpy imprévu (la requête est alors rejouée)",
    "man.avec_numpy" in TRAV and "GuetNumpy" in TRAV and "sys.meta_path.insert(0, GuetNumpy())" in TRAV and "guet.demande" in TRAV)

# 1 bis. la sonde du démarrage (js/sonde.js) : une MESURE, fermée par défaut. Une seule porte — performance.mark ne
# s'écrit nulle part ailleurs dans js/ — qui ne s'ouvre que sur « sonde=1 » dans l'adresse ; fermée, elle ne marque
# rien et rend la promesse qu'on lui donne telle quelle. Joué par node ; le comportement de l'établi entier, avec et sans
# la sonde, se contrôle dans le vrai navigateur (verifier_site.js, filtre « sonde »).
JS = os.path.join(WEB, "js")
hors_porte = [f for f in sorted(os.listdir(JS)) if f.endswith(".js") and f != "sonde.js"
              and "performance.mark" in open(os.path.join(JS, f), encoding="utf-8").read()]
chk("la sonde du démarrage n'a qu'une porte : performance.mark ne s'écrit que dans js/sonde.js", not hors_porte, hors_porte)
SONDE_URL = json.dumps("file://" + os.path.join(JS, "sonde.js"))
ESSAI_SONDE = """
const p = Promise.resolve(1), lus = [];
for (const [cle, adresse] of [["a", "http://x/poutre?note=n1"], ["b", "http://x/poutre?sonde=0"], ["c", "http://x/poutre?sonde=oui"],
                              ["d", "http://x/poutre?sonde=1&note=n1"]]) {
  globalThis.self = { location: { href: adresse } };
  performance.clearMarks();
  const m = await import(%s + "?" + cle);
  m.marquer("x"); m.marquerUneFois("y"); m.marquerUneFois("y");
  const meme = m.finir("z", p) === p;
  await new Promise((ok) => setTimeout(ok, 0));
  lus.push({ sonde: m.SONDE, meme, marques: performance.getEntriesByType("mark").map((e) => e.name), absolues: m.marquesAbsolues().map((e) => e[0]) });
}
console.log(JSON.stringify(lus));
""" % SONDE_URL
try:
    sortie = subprocess.run(["node", "--input-type=module", "-e", ESSAI_SONDE], capture_output=True, text=True, timeout=60)
    sondes = json.loads(sortie.stdout.strip().split("\n")[-1])
except Exception as e:  # noqa: BLE001
    sondes = [str(e)]
FERMEE = {"sonde": False, "meme": True, "marques": [], "absolues": []}
chk("sonde fermée (adresse sans « sonde=1 », ou « sonde=0 », « sonde=oui ») : aucune marque, la même promesse rendue",
    sondes[:3] == [FERMEE] * 3, sondes[:3])
chk("sonde ouverte (« ?sonde=1 ») : chaque marque une fois, préfixée « stab: », la fin d'une promesse marquée",
    sondes[3:] == [{"sonde": True, "meme": False, "marques": ["stab:x", "stab:y", "stab:z"], "absolues": ["x", "y", "z"]}], sondes[3:])

# 2. le paquet se suffit (déballé seul, sans le dépôt, bibliothèques lourdes refusées)
with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(zip_chemin) as z:
        z.extractall(tmp)
    essai = f"""
import json, sys
sys.path = [p for p in sys.path if not p.startswith({RACINE!r})]
sys.path.insert(0, {tmp!r})
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {SW.INTERDITS!r}:
            raise ImportError("refusé : " + name)
sys.meta_path.insert(0, Refus())
from api_web import routeur as R, registre as REG
import api_web
assert api_web.__file__.startswith({tmp!r}), api_web.__file__
out = {{}}
for m in REG.cles():
    r = json.loads(R.traiter(json.dumps({{"module": m, "action": "nouveau"}})))
    out[m] = (r["ok"], r.get("erreur"), (r.get("resume") or {{}}).get("taux"))
# les BIBLIOTHÈQUES (action « consulter ») : chaque rubrique de chacune, avec ses défauts — ce que leur page chargera
import importlib
for b, chemin in R.BIBLIOTHEQUES.items():
    for rub in importlib.import_module(chemin).CLES:
        r = json.loads(R.traiter(json.dumps({{"action": "consulter", "bibliotheque": b, "rubrique": rub}})))
        out["@" + b + ":" + rub] = (r["ok"] and r.get("rubrique") == rub, r.get("erreur"), None)
print(json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", essai], capture_output=True, text=True, cwd=tmp)
    try:
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        res = {"_": (False, p.stderr[-400:], None)}
chk("déballé seul, le paquet calcule chaque module et rend chaque rubrique de chaque bibliothèque (le paquet porte ce que la "
    "Bibliothèque Acier lit : le catalogue, bibliotheques.json, le moteur du flambement…)", res and all(v[0] for v in res.values()), res)
chk(f"chaque module du registre est dans le paquet ({len(SW.ADRESSES)})", sorted(k for k in res if not k.startswith("@")) == sorted(SW.ADRESSES),
    (list(res), list(SW.ADRESSES)))
chk("chaque rubrique de chaque bibliothèque du site a été rendue par le paquet seul",
    sorted(k for k in res if k.startswith("@")) == sorted(f"@{b}:{r}" for b, (c, _t, _f) in SW.BIBLIOTHEQUES.items()
                                                          for r in importlib.import_module(c).CLES), [k for k in res if k.startswith("@")])

# 2 bis. le paquet de la NOTE PDF (moteur/note.zip : chargé à la première note, déballé À CÔTÉ du moteur, js/travailleur.js)
# — son empreinte au manifeste ; il est celui que le générateur écrit (reproductible ; relevé en faisant la note « riche »
# de CHAQUE module de NOTE_PDF, construire.NOTES_RELEVE) ; il ne répète rien du paquet du moteur ; et, déballé seul AVEC
# le paquet du moteur — sans le dépôt, reportlab et Pillow pris dans les paquets, fontTools, les accélérateurs C et les
# bibliothèques lourdes refusés, les seules polices du paquet —, il fait la note de chaque module, et c'est la MÊME que
# celle du dépôt (texte page à page) : ce que le navigateur fera.
note_chemin = os.path.join(WEB, "moteur", "note.zip")
mn = man.get("note") or {}
chk("le manifeste porte l'empreinte et la taille du paquet de la note",
    os.path.exists(note_chemin) and mn.get("sha256") == sha(note_chemin) and mn.get("octets") == os.path.getsize(note_chemin), mn)
octets_note, desc_note = SW.paquet_note(man["fichiers"])
chk("le paquet de la note est celui que le générateur écrit, relevé sur la note de chaque module de NOTE_PDF "
    "(sinon : python3 site/web/construire.py)",
    octets_note == open(note_chemin, "rb").read() and desc_note["modules"] == sorted(SW.NOTE_PDF) and mn.get("modules") == sorted(SW.NOTE_PDF),
    (desc_note["modules"], mn.get("modules")))
with zipfile.ZipFile(note_chemin) as z_note:
    noms_note = sorted(z_note.namelist())
chk(f"le paquet de la note ne répète rien du paquet du moteur ({len(noms_note)} fichiers ; le travailleur les déballe au même endroit)",
    not set(noms_note) & set(noms), sorted(set(noms_note) & set(noms))[:5])
chk(f"le manifeste nomme les modules dont la note charge numpy ({', '.join(desc_note['avec_numpy']) or 'aucun'}), et le "
    "travailleur charge numpy pour eux AVANT la note",
    mn.get("avec_numpy") == desc_note["avec_numpy"] and "m.avec_numpy.includes(cle)) await chargerNumpy()" in TRAV, mn.get("avec_numpy"))
ESSAI_NOTE = """
import base64, json, os, sys
TMP, CLE = {tmp!r}, {cle!r}
sys.path = [TMP] + [p for p in sys.path if p and not p.startswith({racine!r})]
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {refus!r} or name == "reportlab.lib._rl_accel":
            raise ImportError("refusé : " + name)
sys.meta_path.insert(0, Refus())
from api_web import routeur as R
{etat}
r = json.loads(R.traiter(json.dumps({{"module": CLE, "action": "telecharger", "etat": etat, "quoi": "note_pdf",
                                      "infos": {{"nom": "Essai paquet", "chantier": "Contrôle", "indice": "B", "date": "09/11/2026"}}}})))
import PIL, reportlab, api_web
lus = sorted({{os.path.relpath(m.__file__, TMP) for m in list(sys.modules.values()) if getattr(m, "__file__", None) and m.__file__.startswith(TMP)}})
print(json.dumps({{"ok": r.get("ok"), "erreur": r.get("erreur"), "base64": (r.get("fichier") or {{}}).get("base64"),
                   "depuis_paquets": all(m.__file__.startswith(TMP) for m in (PIL, reportlab, api_web)),
                   "fonttools": "fontTools" in sys.modules}}))
"""


def _texte_pdf(octets: bytes) -> list:
    p_ = subprocess.run(["pdftotext", "-layout", "-", "-"], input=octets, capture_output=True)
    return p_.stdout.decode("utf-8", "replace").split("\f")[:-1]


from api_web import routeur as _R  # noqa: E402
import base64  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    for z_ in (zip_chemin, note_chemin):
        with zipfile.ZipFile(z_) as z:
            z.extractall(tmp)
    env_ = {k_: v_ for k_, v_ in os.environ.items() if k_ != "NDC_FONT_DIR"}
    for k in SW.NOTE_PDF:
        code = ESSAI_NOTE.format(tmp=tmp, cle=k, racine=RACINE, refus=SW.NOTE_REFUS, etat=SW.NOTES_RELEVE[k].strip())
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp, env=env_)
        try:
            o = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            o = {"ok": False, "erreur": p.stderr[-600:]}
        # la même note, faite depuis le dépôt (ce processus)
        _ns = {}
        exec(SW.NOTES_RELEVE[k], _ns)  # noqa: S102 — le code de construire.NOTES_RELEVE
        r_d = json.loads(_R.traiter(json.dumps({"module": k, "action": "telecharger", "etat": _ns["etat"], "quoi": "note_pdf",
                                                "infos": {"nom": "Essai paquet", "chantier": "Contrôle", "indice": "B", "date": "09/11/2026"}})))
        t_p = _texte_pdf(base64.b64decode(o["base64"])) if o.get("ok") and o.get("base64") else []
        t_d = _texte_pdf(base64.b64decode(r_d["fichier"]["base64"])) if r_d.get("ok") else ["(dépôt : " + str(r_d.get("erreur")) + ")"]
        chk(f"paquets du moteur et de la note déballés seuls : la note PDF de « {k} » se fait (reportlab, Pillow factice, polices "
            f"et générateurs pris dans les paquets, fontTools refusé), {len(t_p)} pages, le même texte que celle du dépôt",
            o.get("ok") is True and o.get("depuis_paquets") is True and o.get("fonttools") is False and t_p and t_p == t_d,
            o.get("erreur") or (len(t_p), len(t_d)))

# 3. les pages et la charte
chk("css/charte.css est la charte des maquettes", sha(os.path.join(WEB, "css", "charte.css")) == sha(os.path.join(RACINE, "site", "maquettes", "charte.css")))
# (S4 : chaque page porte la VERSION du site, recalculée ici sur les fichiers publiés et les pages rendues — elle n'est
# pas lue dans une page ; les deux bibliothèques sont comparées aussi)
VERSION = SW.version_site(WEB)
_PAGES_RENDUES = SW.pages_rendues()
# (js/versions.js : l'empreinte du CALCUL de chaque module, tirée du relevé — une vérification posée sur un module vaut
#  pour elle, js/verifications.js ; un fichier du calcul modifié sans reconstruire le site rougit ICI)
_VERSIONS_MODULES = SW.versions_modules(RELEVE["par_module"])
for nom, html in [("js/modules.js", SW.modules_js()), ("js/versions.js", SW.versions_js(_VERSIONS_MODULES))] + [(n, SW.poser_version(h, VERSION)) for n, h in _PAGES_RENDUES.items()]:
    chk(f"{nom} est celle que le générateur écrit", os.path.exists(os.path.join(WEB, nom)) and open(os.path.join(WEB, nom), encoding="utf-8").read() == html)
# la version d'un module suit SON calcul : ses fichiers à lui (hors la plomberie partagée) — chaque module en a, deux modules
# n'ont jamais la même, et le moteur de la poutre n'entre pas dans celle des soudures
_propres = {k: sorted(set(f) - SW.HORS_VERSION) for k, f in RELEVE["par_module"].items()}
chk(f"la version d'un module (js/versions.js) : une par établi ({len(_VERSIONS_MODULES)}), tirée de SES fichiers de calcul — chacun en a hors de la plomberie "
    "partagée, deux modules n'ont pas la même, et un module ne dépend pas du moteur d'un autre (la poutre n'est pas dans les soudures)",
    sorted(_VERSIONS_MODULES) == sorted(k for k in SW.ADRESSES if k not in SW.REDIRECTIONS) and len(set(_VERSIONS_MODULES.values())) == len(_VERSIONS_MODULES)
    and all(_propres.get(k) and f"api_web/{k}.py" in _propres[k] for k in _VERSIONS_MODULES)
    and any(f.startswith("beton/poutre/") for f in _propres["poutre"]) and not any(f.startswith("beton/poutre/") for f in _propres["soudures"])
    and all(_VERSIONS_MODULES[k] == SW.empreinte_module(RELEVE["par_module"][k]) for k in _VERSIONS_MODULES),
    {k: len(v) for k, v in _propres.items()})
# la version (S4-D3, C7) : la MÊME balise sur les 25 pages (21 + la Bibliothèque Autres, puis Projets et Équipe, puis les Fondations de grues,
# 11/11/2026), la valeur recalculée,
# la même dans le manifeste
_versions = {n: re.findall(r'<meta name="stab-version" content="([^"]*)">', open(os.path.join(WEB, n), encoding="utf-8").read()) for n in _PAGES_RENDUES}
chk(f"même stab-version sur les {len(_PAGES_RENDUES)} pages (une balise chacune, dans <head>), égale à la version recalculée et à celle du manifeste",
    len(_PAGES_RENDUES) == 25 and all(v == [VERSION] for v in _versions.values()) and man.get("version") == VERSION
    and all(open(os.path.join(WEB, n), encoding="utf-8").read().index('name="stab-version"') < open(os.path.join(WEB, n), encoding="utf-8").read().index("</head>")
            for n in _PAGES_RENDUES), ({n: v for n, v in _versions.items() if v != [VERSION]}, VERSION, man.get("version")))
# un seul script de module, js/app.js, sur chaque page ; js/theme.js reste le seul script classique, en tête (S4-D1)
_scripts = {n: re.findall(r"<script\b[^>]*>", open(os.path.join(WEB, n), encoding="utf-8").read()) for n in _PAGES_RENDUES}
chk("chaque page : un seul script de module, js/app.js (en fin de <body>), et js/theme.js en tête — aucun autre script",
    all(s == ['<script src="js/theme.js">', '<script type="module" src="js/app.js">'] for s in _scripts.values())
    and all(open(os.path.join(WEB, n), encoding="utf-8").read().endswith('<script type="module" src="js/app.js"></script></body>\n</html>\n') for n in _PAGES_RENDUES),
    {n: s for n, s in _scripts.items() if s != ['<script src="js/theme.js">', '<script type="module" src="js/app.js">']})
# le FILET du démarrage (10/11/2026 : une tablette restait sur « Chargement… » sans un mot quand js/app.js ne démarrait
# pas) : chaque page porte le message, juste avant le script de l'application, et l'en-tête la feuille EN LIGNE qui le
# montre au bout de SECOURS_S secondes et le cache dès que l'application a monté un écran (body[data-montage]) ; il ne
# dépend d'aucun fichier, et la politique de sécurité admet ce style en ligne. Le harnais le VOIT (demarrage:secours).
_secours = {}
for n in _PAGES_RENDUES:
    _h = open(os.path.join(WEB, n), encoding="utf-8").read()
    _tete = _h[:_h.index("</head>")]
    _secours[n] = (_h.count('class="secours-demarrage"') == 1 and _h.endswith(SW.SECOURS + SW.SCRIPT_APP + "</body>\n</html>\n")
                   and _tete.count(SW.STYLE_SECOURS) == 1 and _h.count("<style>") == 1)
chk(f"chaque page ({len(_secours)}) : le message de secours une fois, juste avant js/app.js, et sa feuille en ligne une fois dans <head>",
    len(_secours) == 25 and all(_secours.values()), [n for n, ok in _secours.items() if not ok])
chk(f"le message de secours : caché tant que l'application démarre (body[data-montage] … display:none), visible au bout de {SW.SECOURS_S} s "
    "par une animation sans script (opacity et visibility), fixé dans l'écran, un lien « Recharger la page » vers la page même",
    "body[data-montage] .secours-demarrage{display:none}" in SW.STYLE_SECOURS
    and f"animation:secours-demarrage 0s linear {SW.SECOURS_S}s forwards" in SW.STYLE_SECOURS
    and "@keyframes secours-demarrage{to{opacity:1;visibility:visible}}" in SW.STYLE_SECOURS
    and "opacity:0;visibility:hidden;" in SW.STYLE_SECOURS and "position:fixed;" in SW.STYLE_SECOURS
    and 8 <= SW.SECOURS_S <= 20 and SW.SECOURS.endswith('<a href="">Recharger la page</a></p>')
    and "Le site ne démarre pas sur cet appareil." in SW.SECOURS, (SW.SECOURS_S, SW.STYLE_SECOURS))
_app_js = open(os.path.join(WEB, "js", "app.js"), encoding="utf-8").read()
chk("js/app.js pose body[data-montage] à chaque montage (ce qui cache le message de secours)",
    len(re.findall(r"corps\.dataset\.montage = String\(m\);", _app_js)) == 1)
# le voile #demarrage a disparu des trois gabarits (S4-D7) : l'écran est visible tout de suite
_voile =[n for n in _PAGES_RENDUES if re.search(r'id="demarrage|class="demarrage', open(os.path.join(WEB, n), encoding="utf-8").read())]
chk("aucun id=demarrage dans les pages (ni #demarrage-etape, ni le voile .demarrage) : plus de voile, aux trois gabarits", not _voile, _voile)
# ce que js/app.js monte : le corps dit sa sorte (data-page) et ce qu'il faut pour la monter
_corps_de = {n: dict(re.findall(r'\s([\w-]+)="([^"]*)"', re.search(r"<body[^>]*>", open(os.path.join(WEB, n), encoding="utf-8").read()).group(0)))
             for n in _PAGES_RENDUES}
_sorte = {"index.html": {"data-page": "accueil"}}
_sorte.update({SW.PAGES[k]: ({"data-page": "redirection", "data-module": k, "data-vers": SW.ADRESSES[SW.REDIRECTIONS[k]]} if k in SW.REDIRECTIONS
                             else {"data-page": "module", "data-module": k}) for k in SW.PAGES})
_sorte.update({SW.PAGES_BIBLIOTHEQUES[b]: {"data-page": "bibliotheque", "data-bibliotheque": b, "data-titre": SW.BIBLIOTHEQUES[b][1]} for b in SW.BIBLIOTHEQUES})
# (11/11/2026) les pages du BUREAU — les projets, l'équipe : leur sorte EST leur clé, rien d'autre au corps
_sorte.update({SW.PAGES_BUREAU[k]: {"data-page": k} for k in SW.PAGES_BUREAU})
chk("chaque page dit sa sorte au corps : accueil (data-page=accueil, rien d'autre), projets et équipe (leur clé, rien d'autre), établi (module + clé), "
    "bibliothèque (clé + titre), redirection (clé + data-vers)",
    sorted(_sorte) == sorted(_PAGES_RENDUES) and all(_sorte[n].items() <= _corps_de[n].items() for n in _PAGES_RENDUES)
    and all(_corps_de[n] == _sorte[n] for n in ["index.html", *SW.PAGES_BUREAU.values()]),
    {n: _corps_de[n] for n in _PAGES_RENDUES if not _sorte.get(n, {"?": 1}).items() <= _corps_de[n].items()})
# les ÉTABLIS : les pages des modules, moins celles qui ne sont qu'une redirection (REDIRECTIONS : un module absorbé par
# un autre, docs/site/DALLES_FUSION.md § 2.4) — ce qui suit décrit un établi, et ne s'applique pas à une redirection
ETABLIS = {k: v for k, v in SW.PAGES.items() if k not in SW.REDIRECTIONS}

# 4. le cœur Python déposé
v = os.path.join(WEB, SW.PYODIDE)
emp = json.load(open(os.path.join(v, "empreintes.json"), encoding="utf-8"))
faux = [f for f, h in emp["fichiers"].items() if not os.path.exists(os.path.join(v, f)) or sha(os.path.join(v, f)) != h]
chk(f"les {len(emp['fichiers'])} fichiers de Pyodide ont les empreintes relevées", not faux, faux)
lock = json.load(open(os.path.join(v, "pyodide-lock.json"), encoding="utf-8"))
npy = lock["packages"]["numpy"]
chk("la roue numpy a l'empreinte du pyodide-lock officiel", sha(os.path.join(v, npy["file_name"])) == npy["sha256"])
chk("le travailleur charge la version déposée", f'"../{SW.PYODIDE}/pyodide.mjs"' in open(os.path.join(WEB, "js", "travailleur.js"), encoding="utf-8").read())

# 5. les identifiants que les scripts cherchent existent dans leur page (S4, partie « écrans » : l'établi est un écran
# montable, il ne nomme plus le VOILE retiré — #demarrage, #demarrage-etape — : plus aucune exception à cette règle)
VOILE_RETIRE = ()
for js, page in [(js, ETABLIS[k]) for k in ETABLIS for js in ("etabli.js", "toile.js")] + [("accueil.js", "index.html")]:
    src = open(os.path.join(WEB, "js", js), encoding="utf-8").read()
    html = open(os.path.join(WEB, page), encoding="utf-8").read()
    ids = sorted((set(re.findall(r'\$\("#([\w-]+)', src)) | set(re.findall(r'getElementById\("([\w-]+)"', src))) - set(VOILE_RETIRE))
    absents = [i for i in ids if f'id="{i}"' not in html]
    chk(f"{js} → {page} : les {len(ids)} identifiants cherchés existent", not absents, absents)

# 5 bis (avant). l'établi COMPACT (pilote : construire.ETABLI_COMPACT, « Poutres béton armé ») : ses pages portent la
# liste, les propriétés, le résultat (les identifiants que js/compact.js cherche) ; les AUTRES modules gardent l'établi
# d'avant, sans rien de ce pilote. « Enregistrer comme note… » et sa fenêtre (les identifiants que js/rangement.js
# cherche), l'étiquette de la note (#et-ind : l'indice réel, ou « Non enregistrée ») sont sur TOUTE page d'établi depuis
# le 10/11/2026 (décision D1 étendue : ouvrir un calcul n'écrit rien, pour tous les modules) ; à l'accueil, la carte de
# CHAQUE module ouvre le calcul lui-même (lien direct) — « Nouvelle note » reste le chemin qui nomme d'abord.
_src_compact = open(os.path.join(WEB, "js", "compact.js"), encoding="utf-8").read()
_src_rangement = open(os.path.join(WEB, "js", "rangement.js"), encoding="utf-8").read()
_ids_compact = sorted(set(re.findall(r'getElementById\("([\w-]+)"', _src_compact)))
# js/rangement.js cherche ses éléments par un raccourci $("et-…") = document.getElementById
_ids_rangement = sorted(set(re.findall(r'getElementById\("([\w-]+)"', _src_rangement)) | set(re.findall(r'\$\("(et-[\w-]+)"\)', _src_rangement)))
chk("l'établi compact : les deux modules béton armé qui se ressemblent (poutres, dalles — les prédalles ont rejoint les dalles), les "
    "cornières d'appui (refonte du 10/11/2026), puis les poutres bois, la maturité du béton et le taux d'armature (lot Q, 10/11/2026 : "
    "docs/site/LOT_Q_QUATRE_MODULES.md ; chacun a rejoint la liste AVEC son écran refait, jamais avant : sa page serait cassée), "
    "puis les fondations de grues (11/11/2026, né sur l'établi compact : docs/beton/FONDATIONS_GRUE.md), leurs fonctions aux mêmes places ; "
    "les autres modules attendent l'avis du bureau",
    SW.ETABLI_COMPACT == ("poutre", "dalle", "corniere", "bois", "age", "taux", "fondation_grue"), SW.ETABLI_COMPACT)
# la NOTE PDF (décision de l'ingénieur du 09/11/2026 : les notes PDF de Streamlit passent sur le site, celle de la dalle
# dit les chiffres du site) : les modules qui en ont une, chacun avec son ``telecharger(etat, quoi, infos)`` et sa note
# « riche » pour le relevé du paquet de la note (construire.NOTES_RELEVE)
import inspect  # noqa: E402
import importlib  # noqa: E402
chk("la note PDF se fait dans le site pour la poutre, la dalle, le garde-corps, les soudures, les charges sur plancher, "
    "les assemblages et les cornières d'appui (10/11/2026) ; chacun rend « telecharger(etat, quoi, infos) » et a sa note "
    "riche au relevé du paquet de la note",
    SW.NOTE_PDF == ("poutre", "dalle", "garde_corps", "soudures", "plancher", "assemblages", "corniere")
    and sorted(SW.NOTES_RELEVE) == sorted(SW.NOTE_PDF) and set(SW.NOTE_PDF) <= set(ETABLIS)
    and all("infos" in inspect.signature(getattr(importlib.import_module(f"api_web.{k}"), "telecharger")).parameters for k in SW.NOTE_PDF),
    (SW.NOTE_PDF, sorted(SW.NOTES_RELEVE)))
for k in ETABLIS:
    html = open(os.path.join(WEB, SW.PAGES[k]), encoding="utf-8").read()
    # (toute page) « Enregistrer comme note… » dans la barre du haut et le menu ⋯ du téléphone, sa fenêtre, l'étiquette de
    # la note ; plus de « Brouillon » figé ; « Importer .stab » dans le message (une note demandée absente)
    _manque = [i for i in _ids_rangement + ["et-ind", "et-alerte"] if html.count(f'id="{i}"') != 1]
    chk(f"{SW.PAGES[k]} : « Enregistrer comme note… » (barre du haut, menu ⋯), sa fenêtre ({len(_ids_rangement)} identifiants, une fois "
        "chacun), l'étiquette de la note (#et-ind, sans « Brouillon »), « Importer .stab » dans le message",
        not _manque and html.count('data-ranger="1"') == 2 and "Brouillon" not in html
        and len(re.findall(r'<label class="btn sm primary et-alerte-importer" for="ouvrir" hidden>', html)) == 1
        and html.index('id="et-ind"') < html.index('class="et-enr"') < html.index('class="btn sm et-ranger"'), _manque)
    if k in SW.ETABLI_COMPACT:
        absents = [i for i in _ids_compact if f'id="{i}"' not in html]
        chk(f"{SW.PAGES[k]} (établi compact) : la liste, les propriétés, le résultat ; "
            f"les {len(_ids_compact)} identifiants cherchés existent, une seule fois",
            not absents and 'data-etabli="compact"' in html and 'href="css/etabli.css"' in html
            and all(html.count(f'id="{i}"') == 1 for i in _ids_compact)
            # les deux colonnes se replient (boutons communs .pcoll) et s'élargissent (poignées), l'état s'annule
            and all(html.count(f'data-{a}="{c}"') == 1 for a in ("coll", "poignee") for c in "gd")
            and all(html.count(f'data-historique="{h}"') == 1 for h in ("annuler", "retablir")), (absents, _ids_compact))
    else:
        chk(f"{SW.PAGES[k]} : l'établi d'avant, rien du pilote compact",
            "data-etabli" not in html and "etabli.css" not in html and 'id="liste"' not in html)
    # la note PDF, sur TOUTE page d'établi (compacte ou non : construire.boutons_note_pdf, un composant, trois emplacements) :
    # la page dit si son module la fait (body data-note-pdf, que js/etabli.js lit) ; le bouton « Note PDF » est alors
    # allumé (data-action="pdf") dans la barre du haut, le menu ⋯ et la barre du bas du téléphone — sinon grisé, « bientôt »
    _corps = re.search(r"<body[^>]*>", html).group(0)
    _pdf = re.findall(r'<button([^>]*)>((?:(?!</button>).)*?Note PDF(?:(?!</button>).)*?)</button>', html, re.S)
    _grise = [("disabled" in a_ or 'aria-disabled="true"' in a_) for a_, _ in _pdf]
    chk(f"{SW.PAGES[k]} : la note PDF {'se fait dans le navigateur' if k in SW.NOTE_PDF else 'reste « bientôt » (boutons grisés)'} — "
        "barre du haut, menu du téléphone, barre du bas",
        len(_pdf) == 3 and (('data-note-pdf="1"' in _corps) == (k in SW.NOTE_PDF))
        and all(g_ == (k not in SW.NOTE_PDF) for g_ in _grise)
        and all(('data-action="pdf"' in a_) == (k in SW.NOTE_PDF or i_ > 0) for i_, (a_, _) in enumerate(_pdf))
        and all(("bientôt" in a_ + c_) == (k not in SW.NOTE_PDF) for a_, c_ in _pdf[:2]), (_corps, _pdf))
# 5 ter. la BIBLIOTHÈQUE d'une note (docs/site/BIBLIOTHEQUES.md § 2.4) : le bouton « Bibliothèque » (barre du haut et menu ⋯ du
# téléphone) et le panneau #et-bibliotheque — le gabarit de la page de la bibliothèque, les identifiants que js/biblio.js et
# js/biblio_note.js cherchent, la feuille css/bibliotheque.css — dans la page d'un module qui sait préremplir sa bibliothèque
# (api_web/<cle>.bibliotheque_prefill) ; rien de tout cela dans les autres pages
_src_biblio = open(os.path.join(WEB, "js", "biblio.js"), encoding="utf-8").read() + open(os.path.join(WEB, "js", "biblio_note.js"), encoding="utf-8").read()
_ids_biblio = sorted(set(re.findall(r'\$\("#([\w-]+)', _src_biblio)) | set(re.findall(r'getElementById\("([\w-]+)"', _src_biblio)))
_avec = [k for k in ETABLIS if k in SW.ETABLI_COMPACT and SW._bibliotheque_du_module(k)]
chk("la poutre ouvre la bibliothèque béton armé dans sa note (bibliotheque_prefill) ; les dalles pas encore", _avec == ["poutre"], _avec)
for _k_bib in SW.BIBLIOTHEQUES:
    _page_bib = open(os.path.join(WEB, SW.PAGES_BIBLIOTHEQUES[_k_bib]), encoding="utf-8").read()
    chk(f"{SW.PAGES_BIBLIOTHEQUES[_k_bib]} : les {len(_ids_biblio)} identifiants que js/biblio.js cherche existent, une seule fois (hors ceux du panneau d'une note)",
        all(_page_bib.count(f'id="{i}"') == 1 for i in _ids_biblio if i != "et-bibliotheque"), [i for i in _ids_biblio if _page_bib.count(f'id="{i}"') != 1])
for k in ETABLIS:
    html = open(os.path.join(WEB, SW.PAGES[k]), encoding="utf-8").read()
    _corps = re.search(r"<body[^>]*>", html).group(0)
    if k in _avec:
        absents = [i for i in _ids_biblio if f'id="{i}"' not in html]
        chk(f"{SW.PAGES[k]} : « Bibliothèque » dans la barre du haut et le menu ⋯, le panneau avec ses {len(_ids_biblio)} identifiants (une fois chacun), "
            "le menu des six rubriques, la feuille css/bibliotheque.css, et la page le dit (body data-bibliotheque)",
            not absents and all(html.count(f'id="{i}"') == 1 for i in _ids_biblio) and html.count('data-action="bibliotheque"') == 2
            and 'data-bibliotheque="beton"' in _corps and 'href="css/bibliotheque.css"' in html and len(re.findall(r'class="bib-m(?: on)?" data-rubrique="', html)) == 6
            and html.index('data-action="bibliotheque"') < html.index('data-ranger="1"'), (absents, html.count('data-action="bibliotheque"')))
    else:
        chk(f"{SW.PAGES[k]} : sans bibliothèque de note (ni bouton, ni panneau, ni feuille, ni data-bibliotheque)",
            'data-action="bibliotheque"' not in html and 'id="et-bibliotheque"' not in html and "bibliotheque.css" not in html and "data-bibliotheque" not in _corps)
_tuiles = {k: a for a, k in re.findall(r'<a class="tile" href="[^"]+" data-(module|nouvelle-module)="(\w+)"',
                                       open(os.path.join(WEB, "index.html"), encoding="utf-8").read())}
chk("accueil : la carte de CHAQUE module ouvre le calcul lui-même (lien direct, aucune fenêtre interceptée) ; les modules masqués n'ont pas de tuile",
    _tuiles and all(a == "module" for k, a in _tuiles.items())
    and set(_tuiles) == set(SW.PAGES) - SW.MASQUES
    and "nouvelle-module" not in open(os.path.join(WEB, "js", "accueil.js"), encoding="utf-8").read()
    and "nouvelle-module" not in open(os.path.join(WEB, "index.html"), encoding="utf-8").read(), _tuiles)
# la note PROVISOIRE pour tout module (js/etabli.js) : plus de condition sur l'établi compact ; une note absente le dit ; la
# date n'avance qu'à une vraie modification (notes.ecrire(…, {date}))
_ETABLI_JS = open(os.path.join(WEB, "js", "etabli.js"), encoding="utf-8").read()
chk("etabli.js : la note est provisoire pour TOUT module (sans nom ni chantier demandés), une note demandée absente le dit avec « Importer .stab », "
    "la date n'avance qu'à une action qui modifie et change l'état",
    re.search(r'provisoire = !\["nom", "chantier", "dossier"\]\.some', _ETABLI_JS) is not None and "COMPACT && !" not in _ETABLI_JS
    and 'direAlerte("absente"' in _ETABLI_JS and "enregistrer({ date: provisoire || date })" in _ETABLI_JS
    and 'const DATENT = new Set([...MODIFIENT].filter((a) => a !== "activer"))' in _ETABLI_JS, "")
# la GARDE de l'établi (revue S4, resultats-apercu-dessin-modifiable) : envoyer() refuse une action qui MODIFIE la note
# tant que l'écran n'est pas « pret » (l'aperçu, le démarrage, l'erreur ne montrent pas l'état de la note), AVANT toute
# requête au moteur. Sans effet aujourd'hui (aucun contrôle actif de l'aperçu ne modifie) : elle protège un aperçu futur
# éditable. Le contrôle se juge aussi sur une copie sans la garde (il doit la voir manquer).
_GARDE = 'if (modifie && hote.dataset.ecran !== "pret") return null;'


def _garde_modification(js):
    m = re.search(r"async function envoyer\(action[^)]*\)[^{]*\{(.*?)\n    \}\n", js, re.S)
    if not m:
        return False
    corps = m.group(1)
    g, d = corps.find(_GARDE), corps.find("client.demander(")
    return 0 <= g < d and corps.find("const modifie = MODIFIENT.has(action) || modification;") < g


chk("etabli.js : envoyer() refuse une action qui modifie la note tant que l'écran n'est pas « pret » (avant toute requête au moteur) "
    "— et le contrôle voit la garde manquer sur une copie sans elle",
    _garde_modification(_ETABLI_JS) and not _garde_modification(_ETABLI_JS.replace(_GARDE, "")), "")
# la page d'un module ABSORBÉ (predalle.html : son repli sans script) a un style minimal PARTAGÉ (revue S4,
# ecrans-redirection-sans-style) : la règle .redirection de css/site.css (largeur bornée, centrée), que la page charge
_SITE_CSS_R = open(os.path.join(WEB, "css", "site.css"), encoding="utf-8").read()
_redir = re.search(r"^\.redirection \{([^}]*)\}", _SITE_CSS_R, re.M)
_predalle = open(os.path.join(WEB, "predalle.html"), encoding="utf-8").read()
chk("predalle.html (redirection) : son texte porte la classe .redirection, que css/site.css (chargée par la page) met en forme — largeur bornée, centrée",
    _redir is not None and "max-width" in _redir.group(1) and "margin: 24px auto" in _redir.group(1)
    and '<p class="redirection">' in _predalle and 'href="css/site.css"' in _predalle, _redir.group(0) if _redir else "aucune règle .redirection")
# 4 ter. les BIBLIOTHÈQUES (docs/site/BIBLIOTHEQUES.md § 2.1, révision du 09/11/2026) : la carte d'une bibliothèque est
# ÉTROITE et HAUTE, à GAUCHE du bloc de sa famille, et les modules de la famille s'alignent à sa DROITE dans leur propre
# grille — le bloc .tiles.fam-b : la carte, puis la grille .tiles des modules (css/site.css la pose en sous-grille à partir
# de la deuxième colonne : aucun module sous la carte ; la mesure à l'écran est dans verifier_site.js, filtre
# « bibliotheque ») ; une famille sans bibliothèque garde sa grille .tiles seule. Le volet « Modules » les liste en
# premier ; les modules masqués restent servis (leur page est écrite, js/modules.js les garde, marqués « masque ») mais ne
# sont ni des tuiles, ni des choix de « Nouvelle note », ni des entrées du volet.
import importlib  # noqa: E402
from html.parser import HTMLParser  # noqa: E402


class _Arbre(HTMLParser):
    """L'arbre d'une page (balise, attributs, enfants, texte) : assez pour lire la STRUCTURE de l'accueil."""
    VIDES = {"br", "img", "input", "meta", "link", "hr", "source", "col", "area", "base", "wbr"}

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.racine = {"tag": "#", "a": {}, "e": [], "t": ""}
        self.pile = [self.racine]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        n = {"tag": tag, "a": dict(attrs), "e": [], "t": ""}
        self.pile[-1]["e"].append(n)
        if tag not in self.VIDES:
            self.pile.append(n)

    def handle_startendtag(self, tag, attrs):
        self.pile[-1]["e"].append({"tag": tag, "a": dict(attrs), "e": [], "t": ""})

    def handle_endtag(self, tag):
        for i in range(len(self.pile) - 1, 0, -1):
            if self.pile[i]["tag"] == tag:
                del self.pile[i:]
                break

    def handle_data(self, d):
        self.pile[-1]["e"].append({"tag": "#texte", "a": {}, "e": [], "t": d})


def _cls(n):
    return (n["a"].get("class") or "").split()


def _tous(n, pred):
    for x in n["e"]:
        if pred(x):
            yield x
        yield from _tous(x, pred)


def _texte(n):
    return n["t"] if n["tag"] == "#texte" else "".join(_texte(x) for x in n["e"])


def _el(n):
    """Les enfants ÉLÉMENTS d'un nœud (sans les textes)."""
    return [x for x in n["e"] if x["tag"] != "#texte"]


def _familles_accueil(html):
    """Chaque bloc de famille de l'accueil : {fam, forme, carte, rubriques, modules} — ``forme`` dit comment le bloc est
    bâti : « biblio » (la carte puis la grille des modules), « grille » (la grille seule), sinon la liste des classes."""
    out = []
    for f in _tous(_Arbre(html).racine, lambda x: x["tag"] == "div" and "famille" in _cls(x)):
        enf = _el(f)
        corps = enf[1] if len(enf) == 2 else None
        o = {"fam": f["a"].get("data-fam"), "forme": None, "carte": None, "rubriques": [], "modules": [], "icones": 0}
        tuiles = []
        if corps is not None and _cls(corps) == ["tiles", "fam-b"] and len(_el(corps)) == 2:
            carte, grille = _el(corps)
            if carte["tag"] == "a" and _cls(carte) == ["tile", "biblio"] and _cls(grille) == ["tiles"]:
                o["forme"] = "biblio"
                o["carte"] = (carte["a"].get("data-bibliotheque"), carte["a"].get("href"),
                              " ".join(_texte(next(_tous(carte, lambda x: "nm" in _cls(x)))).split()))
                lis = list(_tous(carte, lambda x: x["tag"] == "li"))
                o["rubriques"] = [_texte(li).strip() for li in lis]
                o["icones"] = sum(any(x["tag"] == "svg" for x in _el(li)) for li in lis)
                tuiles = _el(grille)
        elif corps is not None and _cls(corps) == ["tiles"]:
            o["forme"] = "grille"
            tuiles = _el(corps)
        else:
            o["forme"] = [_cls(x) for x in enf]
        o["modules"] = [" ".join(_texte(next(_tous(t, lambda x: "nm" in _cls(x)))).split()) for t in tuiles if "tile" in _cls(t)]
        o["autres"] = [t["tag"] for t in tuiles if "tile" not in _cls(t) or "biblio" in _cls(t)]
        out.append(o)
    return out


_accueil = open(os.path.join(WEB, "index.html"), encoding="utf-8").read()
_FAMS = _familles_accueil(_accueil)
_fam_biblio = {fam for _c, (_m, _t, fam) in SW.BIBLIOTHEQUES.items()}
_attendu = [(fam, [n for k, n in SW.MQ.listes(mods)]) for fam, _nom, mods in SW.MQ.FAMILLES]
chk("accueil : un bloc par famille, dans l'ordre, chacun avec EXACTEMENT ses modules listés (MASQUES absents), aucune "
    "autre pièce dans la grille des modules",
    [(f["fam"], f["modules"]) for f in _FAMS] == _attendu and not any(f["autres"] for f in _FAMS),
    ([(f["fam"], f["modules"], f["autres"]) for f in _FAMS], _attendu))
chk("accueil : une famille qui a une bibliothèque → la carte à GAUCHE, puis la grille de ses modules à DROITE "
    "(.tiles.fam-b = la carte, puis .tiles) ; une famille sans bibliothèque → sa grille .tiles seule",
    all(f["forme"] == ("biblio" if f["fam"] in _fam_biblio else "grille") for f in _FAMS), [(f["fam"], f["forme"]) for f in _FAMS])
_cartes = [f["carte"] for f in _FAMS if f["carte"]]
chk("accueil : chaque bibliothèque a sa carte, dans le bloc de SA famille, qui vise sa page (href = bibliotheque-<clé>)",
    sorted(k for k, _, _ in _cartes) == sorted(SW.BIBLIOTHEQUES)
    and all(f["carte"][0] in SW.BIBLIOTHEQUES and SW.BIBLIOTHEQUES[f["carte"][0]][2] == f["fam"] for f in _FAMS if f["carte"])
    and all(h == f"bibliotheque-{k}" == SW.ADRESSES_BIBLIOTHEQUES[k] for k, h, _ in _cartes), _cartes)
_rub_beton = [t for _, t, _, _ in importlib.import_module("api_web.biblio_beton").RUBRIQUES]
_fb = next((f for f in _FAMS if f["carte"] and f["carte"][0] == "beton"), {"carte": (None, None, ""), "rubriques": [], "icones": 0})
chk("accueil : la carte dit « Bibliothèque Béton armé » et LISTE ses rubriques, une par ligne avec leur icône, dans l'ordre "
    "du menu de la bibliothèque (RUBRIQUES, une source)",
    _fb["carte"][2] == "Bibliothèque Béton armé" and _fb["rubriques"] == _rub_beton == ["Bétons", "Aciers", "Sections", "Enrobages", "Ancrages", "Hourdis"]
    and _fb["icones"] == len(_rub_beton), (_fb["carte"], _fb["rubriques"], _fb["icones"]))
# BIB5 : la Bibliothèque ACIER, déclarée sur la famille Acier, a pris la forme prévue sans rien recoder (le gabarit LIT
# _bibliotheques_de(fam)) : sa carte dit « Bibliothèque Acier » et liste les rubriques de son menu (RUBRIQUES de
# api_web/biblio_acier.py, une source) ; le « Catalogue des profilés » qu'elle porte est masqué (MASQUES), ses
# modules de calcul sont à sa droite
_rub_acier = [t for _, t, _, _ in importlib.import_module("api_web.biblio_acier").RUBRIQUES]
_fa = next((f for f in _FAMS if f["fam"] == "acier"), {"carte": (None, None, ""), "rubriques": [], "icones": 0, "modules": [], "forme": None})
chk("accueil : la carte « Bibliothèque Acier » (BIB5) est à gauche du bloc Acier, vise sa page, liste ses rubriques (Profilés, Nuances, Boulons) avec "
    "leurs icônes ; le Catalogue des profilés n'est plus une tuile, les cinq calculs Acier sont à sa droite",
    _fa["forme"] == "biblio" and _fa["carte"] == ("acier", "bibliotheque-acier", "Bibliothèque Acier") and _fa["rubriques"] == _rub_acier == ["Profilés", "Nuances", "Boulons"]
    and _fa["icones"] == 3 and "catalogue" in SW.MASQUES and "Catalogue des profilés" not in _fa["modules"]
    and _fa["modules"] == ["Choix de profilé", "Cornières", "Flambement", "Assemblages métalliques", "Soudures"], (_fa["carte"], _fa["rubriques"], _fa["modules"]))
# la Bibliothèque AUTRES (11/11/2026, docs/site/BIBLIOTHEQUE_AUTRES.md) : la troisième famille a sa carte, de la même forme
# (à gauche de ses modules, haute de deux rangées même si la famille n'a qu'une rangée de modules), ses trois outils en
# rubriques avec leurs icônes. (L'essai du gabarit générique se faisait sur « Autres » tant qu'elle n'avait pas de
# bibliothèque : le contrôle SUIT la vraie carte, comme il avait suivi celle de l'Acier à BIB5.)
_fo = next((f for f in _FAMS if f["fam"] == "autres"), {"carte": (None, None, ""), "rubriques": [], "icones": 0, "modules": [], "forme": None})
_rub_autres = [t for _, t, _, _ in importlib.import_module("api_web.biblio_autres").RUBRIQUES]
chk("accueil : la carte « Bibliothèque Autres » est à gauche du bloc Autres, vise sa page, liste ses trois outils (Poutres, Plaques, Sections) avec "
    "leurs icônes ; les modules de la famille sont à sa droite, les mêmes",
    _fo["forme"] == "biblio" and _fo["carte"] == ("autres", "bibliotheque-autres", "Bibliothèque Autres") and _fo["rubriques"] == _rub_autres == ["Poutres", "Plaques", "Sections"]
    and _fo["icones"] == 3 and _fo["modules"] == [n for f, ns in _attendu if f == "autres" for n in ns], (_fo["carte"], _fo["rubriques"], _fo["modules"]))
# deux bibliothèques dans une famille sont refusées à la construction (le gabarit n'en place qu'une : la première colonne)
_sauve = dict(SW.BIBLIOTHEQUES), dict(SW.ADRESSES_BIBLIOTHEQUES)
try:
    SW.BIBLIOTHEQUES["essai2"] = ("api_web.biblio_beton", "Autres bis", "autres")
    SW.ADRESSES_BIBLIOTHEQUES["essai2"] = "bibliotheque-essai2"
    try:
        SW.page_accueil()
        _deux = "acceptées"
    except ValueError as e:
        _deux = str(e)
finally:
    SW.BIBLIOTHEQUES.clear(); SW.BIBLIOTHEQUES.update(_sauve[0])
    SW.ADRESSES_BIBLIOTHEQUES.clear(); SW.ADRESSES_BIBLIOTHEQUES.update(_sauve[1])
chk("accueil : les trois familles ont leur bibliothèque, de la même forme ; deux bibliothèques dans une famille sont refusées à la construction",
    all(f["forme"] == "biblio" for f in _FAMS) and len(_FAMS) == 3 and "n'en place qu'une" in _deux, ([(f["fam"], f["forme"]) for f in _FAMS], _deux))
# la feuille du site porte la règle de la forme (la mesure à l'écran est celle de verifier_site.js) : la grille des
# modules d'un bloc à bibliothèque commence à la DEUXIÈME colonne, en sous-grille (les colonnes de la charte, une
# source) ; la carte ne se mesure pas (la hauteur est celle du bloc des modules) ; plus de carte « large » (span 2)
_site_css = open(os.path.join(WEB, "css", "site.css"), encoding="utf-8").read()
chk("css/site.css : la grille des modules d'un bloc à bibliothèque commence à la 2ᵉ colonne en sous-grille, la carte prend "
    "la hauteur du bloc (container-type size), plus de carte large (span 2)",
    re.search(r"\.accueil \.fam-b > \.tiles \{ grid-column: 2 / -1; grid-template-columns: subgrid; \}", _site_css)
    and re.search(r"\.accueil \.tile\.biblio \{ container: bib / size;", _site_css) and not re.search(r"\.tile\.biblio \{[^}]*span 2", _site_css))
_js_mod = json.loads(re.search(r"MODULES = (\{.*\});", open(os.path.join(WEB, "js", "modules.js"), encoding="utf-8").read()).group(1))
chk("js/modules.js garde les modules masqués (notes existantes) et les marque « masque »",
    all(_js_mod[k].get("masque") is True for k in SW.MASQUES) and not any(_js_mod[k].get("masque") for k in SW.PAGES if k not in SW.MASQUES)
    and set(SW.MASQUES) <= set(SW.PAGES), {k: _js_mod[k] for k in SW.MASQUES})
chk("accueil : « Nouvelle note » ne propose pas les modules masqués, le volet « Modules » commence par les bibliothèques sans les masqués",
    not any(f'data-module="{k}"' in _accueil for k in SW.MASQUES)
    and re.search(r'<aside class="modpanel">.*?<div class="navlabel">Bibliothèques</div><a class="navitem" href="bibliotheque-beton">', _accueil, re.S)
    and re.search(r'<div class="navlabel">Bibliothèques</div>(?:(?!navlabel).)*<a class="navitem" href="bibliotheque-acier">', _accueil, re.S)
    and not any(f'href="{SW.ADRESSES[k]}"' in _accueil.split('<aside class="modpanel">')[1] for k in SW.MASQUES))
chk("les pages des modules masqués sont toujours écrites et servies", all(os.path.exists(os.path.join(WEB, SW.PAGES[k])) for k in SW.MASQUES))

# 4 quater. la FUSION des dalles et des prédalles (docs/site/DALLES_FUSION.md § 1, § 2.4) :
# - « predalle » est absorbé par « dalle » (REDIRECTIONS) : masqué de l'accueil, encore au registre et dans js/modules.js
#   (« masque », « vers » : la table d'alias que js/notes.js lit), sa page est une REDIRECTION vers dalle qui garde
#   l'adresse (js/redirection.js, un script du site — jamais en ligne ; le <meta refresh> et un lien en repli), sans
#   rien d'un tiers ;
# - la colonne de gauche COMPACTE (ARBRE_COMPACT : la dalle ET la poutre, demande de l'ingénieur du 09/11/2026) : l'en-tête
#   du panneau de leur page n'écrit plus « Données » — son premier enfant est l'EMPLACEMENT .et-elt où js/etabli.js pose la
#   LIGNE de l'élément ouvert (.et-elsel : le nom en champ, la liste, copier, supprimer) ; les autres modules gardent
#   « Données » ; et le moteur rend bien cette ligne (et-arbre-c, .et-elsel, le nom en champ) pour ces modules-là, la tête
#   des résultats sans champ : le tuple du générateur, le rendu du moteur et le déplacement du JS sont trois copies,
#   liées ici.
chk("la fusion : « predalle » est absorbé par « dalle » (REDIRECTIONS), masqué de l'accueil, toujours au registre et servi",
    SW.REDIRECTIONS == {"predalle": "dalle"} and "predalle" in SW.MASQUES and "predalle" in SW.PAGES and "dalle" not in SW.MASQUES
    and all(v in SW.ADRESSES and v not in SW.REDIRECTIONS for v in SW.REDIRECTIONS.values()), (SW.REDIRECTIONS, sorted(SW.MASQUES)))
chk("js/modules.js : un module absorbé porte « vers » (le module qui sert ses notes) ; les autres non",
    all(_js_mod[k].get("vers") == v for k, v in SW.REDIRECTIONS.items()) and not any("vers" in _js_mod[k] for k in SW.PAGES if k not in SW.REDIRECTIONS)
    and "moduleServi" in open(os.path.join(WEB, "js", "notes.js"), encoding="utf-8").read()
    and ".vers" in open(os.path.join(WEB, "js", "notes.js"), encoding="utf-8").read(), {k: _js_mod[k] for k in SW.REDIRECTIONS})
chk("js/notes.js, js/accueil.js, js/etabli.js : la page d'une note vient de pageDe (l'alias), plus jamais de MODULES[module].page",
    all("pageDe(" in open(os.path.join(WEB, "js", f), encoding="utf-8").read() for f in ("accueil.js", "etabli.js"))
    and not any(re.search(r"MODULES\[[^\]]+\]\.page", open(os.path.join(WEB, "js", f), encoding="utf-8").read()) for f in ("accueil.js", "etabli.js", "rangement.js")))
# (S4 : la page de redirection a pour seul script de module js/app.js, qui y mène en gardant l'adresse — dans
# l'application, naviguer() résout « vers » avant même de la lire ; plus de voile ni de <meta refresh> — il partirait
# après un échange de corps et perdrait la requête — : un court lien visible, le repli)
for k, vers in SW.REDIRECTIONS.items():
    _red = open(os.path.join(WEB, SW.PAGES[k]), encoding="utf-8").read()
    _corps_red = re.search(r"<body[^>]*>", _red).group(0)
    chk(f"{SW.PAGES[k]} : une redirection vers « {SW.ADRESSES[vers]} » — body data-page=redirection et data-vers, js/app.js, un court lien "
        f"visible « {SW.LIENS_REDIRECTION[vers]} », sans voile ni <meta refresh>, rien d'un tiers, aucun script en ligne, pas d'établi",
        f'data-vers="{SW.ADRESSES[vers]}"' in _corps_red and 'data-page="redirection"' in _corps_red and SW.SCRIPT_APP in _red
        and "redirection.js" not in _red and "http-equiv" not in _red
        and f'<a href="{SW.ADRESSES[vers]}">{SW.LIENS_REDIRECTION[vers]}</a>' in _red and SW.LIENS_REDIRECTION == {"dalle": "Ouvrir les dalles"}
        and "http" not in _red and 'id="donnees"' not in _red and not re.search(r'id="demarrage|class="demarrage', _red)
        and not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", _red), _red[:300])
# (S4, partie « écrans » : js/redirection.js a disparu — la page de redirection n'a plus que js/app.js, qui la mène, ouverte
# directement, à la page qui la sert, et qui, DANS l'application, résout « vers » (js/modules.js) avant de lire la page)
_redir_code = re.sub(r"(^|\s)//[^\n]*", r"\1", open(os.path.join(WEB, "js", "app.js"), encoding="utf-8").read())   # le code seul
chk("js/app.js mène une redirection : ouverte directement, location.replace(body data-vers + la requête + l'ancre) ; dans "
    "l'application, la route « vers » (js/modules.js) est résolue avant de lire la page, l'adresse gardée ; aucune adresse "
    "écrite en dur dans le code ; js/redirection.js n'existe plus",
    "location.replace(corps.dataset.vers + location.search + location.hash)" in _redir_code and 'corps.dataset.page === "redirection"' in _redir_code
    and 'if (r.sorte === "redirection") { u = new URL(r.vers + u.search + u.hash, u); r = route(u.href); }' in _redir_code
    and not any(f'"{k}"' in _redir_code or f'"{v}"' in _redir_code for k, v in SW.REDIRECTIONS.items())
    and not os.path.exists(os.path.join(WEB, "js", "redirection.js")), "")
chk("ARBRE_COMPACT : la colonne de gauche compacte est celle des dalles et des poutres (09/11/2026), des cornières (10/11/2026), des "
    "trois modules du lot Q (poutres bois, maturité, taux d'armature) et des fondations de grues (11/11/2026), un sous-ensemble de l'établi compact",
    set(SW.ARBRE_COMPACT) == {"dalle", "poutre", "corniere", "bois", "age", "taux", "fondation_grue"} and len(SW.ARBRE_COMPACT) == 7
    and set(SW.ARBRE_COMPACT) <= set(SW.ETABLI_COMPACT), SW.ARBRE_COMPACT)
EMPLACEMENT_ELT = '<div class="et-elt" role="group" aria-label="L\'élément ouvert"></div>'
for k in ETABLIS:
    html = open(os.path.join(WEB, SW.PAGES[k]), encoding="utf-8").read()
    gauche = html[html.index('<section class="panel left">'):html.index('<section class="panel right">')]
    tete_g = gauche[:gauche.index('id="donnees"')]
    if k in SW.ARBRE_COMPACT:
        chk(f"{SW.PAGES[k]} (colonne compacte) : le premier enfant de l'en-tête du panneau des données est l'emplacement .et-elt de la "
            "ligne de l'élément (vide : l'écran l'y pose), une seule fois dans la page ; plus de titre « Données » ; Annuler et Rétablir en place",
            gauche.startswith('<section class="panel left"><div class="phead">' + EMPLACEMENT_ELT) and html.count('class="et-elt"') == 1
            and "Données" not in tete_g.replace('title="Replier les données"', "") and 'data-historique="annuler"' in tete_g
            and 'data-historique="retablir"' in tete_g, gauche[:200])
    else:
        chk(f"{SW.PAGES[k]} : l'en-tête du panneau des données porte le titre « Données » ({'établi compact' if k in SW.ETABLI_COMPACT else 'établi d avant'}), "
            "sans emplacement de ligne d'élément",
            gauche.startswith('<section class="panel left"><div class="phead"><span class="h3">Données</span>') and "et-elt" not in html, gauche[:200])
sys.path.insert(0, RACINE)
from api_web import routeur as _RT  # noqa: E402

for k in SW.ETABLI_COMPACT:
    _out = json.loads(_RT.traiter(json.dumps({"module": k, "action": "nouveau"})))
    _compact = k in SW.ARBRE_COMPACT
    chk(f"{k} : le moteur rend {'la colonne compacte (et-arbre-c ; la ligne .et-elsel, le nom en champ) et la tête des résultats sans champ' if _compact else 'l arbre d avant (en-tête de liste, ligne d élément) et la tête des résultats sans champ'}"
        " — comme ARBRE_COMPACT le dit",
        _out.get("ok") and ('class="et-arbre et-arbre-c"' in _out["donnees"]) == _compact and ('class="et-arbre-h"' in _out["donnees"]) == (not _compact)
        and (re.search(r'<div class="et-elsel" data-elements="1"><input data-chemin="nom" data-sorte="texte" value="[^"]*" class="et-elt-nom"',
                       _out["donnees"]) is not None) == _compact
        and '<input data-chemin="nom"' in _out["donnees"] and "data-chemin" not in _out["resultat"].split('<div class="et-blocs">')[0]
        and 'class="et-rp-nom"' not in _out["resultat"], _out.get("erreur"))
# … et l'écran la déplace : js/etabli.js cherche l'emplacement dans l'en-tête du panneau de gauche, y pose la ligne .et-elsel
# que le moteur a rendue dans #donnees (à chaque réponse, un nom en cours de frappe survit), et lui passe ses clics
# (agirElement) ; le nom s'y valide en quittant le champ ou sur Entrée — js/compact.js y ouvre la liste (le menu)
_js_etabli = open(os.path.join(WEB, "js", "etabli.js"), encoding="utf-8").read()
_poser = re.search(r"if \(ELT\) \{\s*const ligne = zone\.querySelector\(\"\.et-elsel\"\);\s*ELT\.replaceChildren\(\.\.\.\(ligne \? \[ligne\] : \[\]\)\);"
                   r"\s*reprendreSaisie\(ELT, saisieElt\);", _js_etabli)
chk("js/etabli.js déplace la ligne .et-elsel de #donnees dans l'emplacement .et-elt de l'en-tête (replaceChildren, à chaque réponse ; "
    "la saisie du nom en cours reprise), lui passe ses clics (agirElement) et valide le nom en quittant le champ",
    'const ELT = $(".panel.left .phead .et-elt");' in _js_etabli and _poser is not None
    and _js_etabli.index("zone.innerHTML = elements + html;") < _poser.start()
    and "const saisieElt = ELT ? saisieEnCours(ELT) : null;" in _js_etabli
    and re.search(r'ELT\.addEventListener\("click", \(ev\) => \{ agirElement\(ev\); \}(, \{ signal \})?\)', _js_etabli) is not None
    and re.search(r'ELT\.addEventListener\("focusout", .*?valider\(el\);', _js_etabli, re.S) is not None
    and 'document.querySelector(".panel.left .phead .et-elt")' in _src_compact and "zonesMenus = [donnees, elt]" in _src_compact)

# 5 bis. les trois formats : ce que le téléphone et la tablette trouvent dans chaque page (la disposition est dans
# css/site.css et se vérifie dans le vrai navigateur : verifier_site.js, filtre « formats »)
for k in ETABLIS:
    html = open(os.path.join(WEB, SW.PAGES[k]), encoding="utf-8").read()
    onglets = re.findall(r'data-onglet="(\w+)"', html)
    manque = [x for x in ('class="mtop"', 'class="mtabs"', 'class="mbar"', 'id="mmenu"', 'id="menu-m"', 'id="nom-note-m"', 'id="chip-m"',
                          'data-zoom="plus"', 'data-zoom="moins"', 'data-zoom="ajuster"', 'data-plein="1"', 'id="ouvrir"') if x not in html]
    chk(f"{SW.PAGES[k]} : l'en-tête, le menu, les onglets Données · Dessin · Résultat, la barre du bas, le zoom et le plein écran",
        not manque and onglets == ["donnees", "dessin", "resultat"], (manque, onglets))
    # les actions du téléphone sont celles du PC (mêmes fonctions, autre disposition)
    actions = lambda frag: sorted(set(re.findall(r'data-action="(\w+)"', frag)))
    haut = html[html.index('<header class="top"'):html.index("</header>")]
    tel = html[html.index('<header class="mtop"'):html.index('<div class="bench"')] + html[html.index('<div class="mbar"'):html.index("</main>")]
    chk(f"{SW.PAGES[k]} : le téléphone offre les actions de la barre du haut (ouvrir, exporter, imprimer)",
        set(actions(haut)) <= set(actions(tel)) and 'for="ouvrir"' in haut and 'for="ouvrir"' in tel, (actions(haut), actions(tel)))
# l'enregistrement de la note : le MÊME indicateur (construire.enregistrement) dans la barre du haut (PC, tablette) et
# sur la 2ᵉ ligne de l'en-tête du téléphone, et le message d'échec qui offre « Exporter .stab », dans chaque établi
IND, ALERTE, REVOIR = SW.enregistrement(), SW.alerte_enregistrement(), SW.menu_enregistrement()
fautes = []
for k in ETABLIS:
    html = open(os.path.join(WEB, SW.PAGES[k]), encoding="utf-8").read()
    haut = html[html.index('<header class="top"'):html.index("</header>")]
    tel = html[html.index('<header class="mtop"'):html.index('<div class="mmenu"')]
    menu = html[html.index('<div class="mmenu"'):html.index('<nav class="mtabs"')]
    # au téléphone, la 2ᵉ ligne n'écrit que l'état d'un échec : le menu ⋯ le dit EN ENTIER, en première entrée
    if not (IND in haut and IND in tel and html.count(IND) == 2 and tel.index(IND) < tel.index('id="fil-m"') and ALERTE in html
            and 'class="saved' not in html and html.count(REVOIR) == 1 and menu.index(REVOIR) < menu.index('class="mi"')):
        fautes.append(SW.PAGES[k])
chk("chaque établi : le même indicateur d'enregistrement dans la barre du haut et sur la 2ᵉ ligne de l'en-tête du téléphone, "
    "le message d'échec avec « Exporter .stab » (plus d'ancien .saved), et l'échec en entier en tête du menu ⋯",
    not fautes and 'data-action="exporter"' in ALERTE and "Exporter .stab" in ALERTE and 'data-enr="neuf"' in IND
    and re.search(r'<span class="et-enr-txt"><span class="et-enr-etat">[^<]+</span><span class="et-enr-raison"></span></span>', IND)
    and "data-revoir-alerte" in REVOIR and " hidden>" in REVOIR, fautes[:3])
chk("l'accueil porte le signal d'une mémoire des notes illisible (#et-memoire)", 'id="et-memoire"' in open(os.path.join(WEB, "index.html"), encoding="utf-8").read())


# aucune règle des feuilles du site ne CACHE l'indicateur (.et-enr), son texte (.et-enr-txt) ni la 2ᵉ ligne du téléphone
# (.et-l2), à aucune largeur — l'impression mise à part (le papier n'a pas d'état d'enregistrement). C'est la règle de la
# charte « .saved .txt { display: none } » sous 1 360 px qui rendait l'échec muet ; la charte ne nomme pas .et-enr.
def _sans_impression(css: str) -> str:
    """La feuille sans ses blocs « @media print { … } » (accolades comptées)."""
    out, i = [], 0
    for m in re.finditer(r"@media\s+print\s*\{", css):
        if m.start() < i:
            continue
        out.append(css[i:m.start()])
        j, prof = m.end(), 1
        while prof and j < len(css):
            prof += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        i = j
    return "".join(out) + css[i:]


PROTEGES = re.compile(r"\.(?:et-enr|et-enr-txt|et-enr-etat|et-l2)(?![\w-])")
# la RAISON d'un échec (.et-enr-raison) ne cède qu'au téléphone (.mtop), où la 2ᵉ ligne tient entre le retour et la
# pastille de la note : la barre du haut (PC, tablette) la dit toujours
RAISON = re.compile(r"\.et-enr-raison(?![\w-])")
raisons = []
CACHE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\.0*)?\s*(?:;|!|$)|content-visibility\s*:\s*hidden", re.I)
caches = []
feuilles = ["site.css", "charte.css"] + [os.path.join("modules", f) for f in sorted(os.listdir(os.path.join(WEB, "css", "modules")))]
for f in feuilles:
    css = _sans_impression(re.sub(r"/\*.*?\*/", "", open(os.path.join(WEB, "css", f), encoding="utf-8").read(), flags=re.S))
    for sel, decl in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if not CACHE.search(decl):
            continue
        for un in sel.split(","):
            sujet = re.split(r"[\s>+~]+", un.strip())[-1]
            if PROTEGES.search(sujet):
                caches.append(f"{f} : {un.strip()}")
            if RAISON.search(sujet) and not re.search(r"\.mtop(?![\w-])", un):
                raisons.append(f"{f} : {un.strip()}")
chk("aucune règle des feuilles du site ne cache l'indicateur d'enregistrement, son texte, son état ou la 2ᵉ ligne du téléphone (hors impression)",
    not caches, caches[:4])
chk("la raison d'un échec d'enregistrement n'est cachée qu'au téléphone (une règle sans .mtop la cacherait dans la barre du haut)",
    not raisons, raisons[:4])
# la pastille la plus large de l'en-tête du téléphone, dans verifier_site.js (filtre « enregistrement ») : une COPIE NOMMÉE
# du statut « calcul impossible » du moteur, le plus long des statuts qu'écrit la pastille quand la note n'a pas de taux
# (niveaux c et x, et les statuts écrits en clair par les modules)
sys.path.insert(0, RACINE)
from api_web import commun as _CW  # noqa: E402
_harnais = open(os.path.join(WEB, "verifier_site.js"), encoding="utf-8").read()
_pastille = re.findall(r"const PASTILLE_LA_PLUS_LARGE = '([^']+)';", _harnais)
_sans_taux = {_CW.STATUTS["c"][1], _CW.STATUTS["x"][1]}
for _f in sorted(os.listdir(os.path.join(RACINE, "api_web"))):
    if _f.endswith(".py"):
        for _ligne in open(os.path.join(RACINE, "api_web", _f), encoding="utf-8"):
            # « "statut": "À compléter" if vides else "À corriger" » sur une ligne sans taux : les deux textes
            _m = re.search(r'"statut":\s*("[^"]*"(?:\s+if\s+[^"]+?\s+else\s+"[^"]*")?)', _ligne)
            if _m and '"taux": None' in _ligne:
                _sans_taux |= set(re.findall(r'"([^"]*)"', _m.group(1)))
chk(f"verifier_site.js : la pastille la plus large essayée est le statut « calcul impossible » du moteur, le plus long des statuts sans taux "
    f"({len(_sans_taux)} statuts)", _pastille == [_CW.STATUTS["x"][1]] and all(len(x) <= len(_pastille[0]) for x in _sans_taux),
    (_pastille, sorted(_sans_taux, key=len)[-3:]))
# les classes posées sur un composant « et- » (l'indicateur, le message, le signal de l'accueil) portent elles aussi le
# préfixe « et- » (spécification de la refonte, B.2.6) : une classe nue (« .et-alerte.info ») peut rencontrer une règle de
# la charte ou du <style> d'un module. Les :not(…) sont retirés : ils ne posent pas de classe sur l'élément.
_css_site = re.sub(r":not\([^)]*\)", "", re.sub(r"/\*.*?\*/", "", open(os.path.join(WEB, "css", "site.css"), encoding="utf-8").read(), flags=re.S))
nues = sorted({c for compose in re.findall(r"[\w-]*(?:\.[\w-]+|\[[^\]]*\]|:{1,2}[\w-]+)+", _css_site) if ".et-" in compose
               for c in re.findall(r"\.([\w-]+)", compose) if not c.startswith("et-")})
chk("les classes posées sur un composant « et- » portent le préfixe « et- » (B.2.6)", not nues, nues)
acc = open(os.path.join(WEB, "index.html"), encoding="utf-8").read()
nav_pc = re.findall(r'data-tip="([^"·]+?)(?: · bientôt)?"', acc[acc.index('<nav class="nav"'):acc.index("</nav>")])
nav_tel = re.findall(r"<span>([^<]+)</span></a>", acc[acc.index('<nav class="mnav"'):])[:5]
chk("accueil : au téléphone, l'en-tête et les cinq entrées de la barre, dans le même ordre", 'class="mhead"' in acc
    and [x.strip() for x in nav_pc[:5]] == nav_tel, (nav_pc, nav_tel))
# le seuil du téléphone : la charte (maquettes), la feuille du site et le script de l'établi disent le même nombre
seuils = {"charte.css": set(re.findall(r"max-width:\s*(\d+)px\)\s*\{\s*body", open(os.path.join(WEB, "css", "charte.css"), encoding="utf-8").read())),
          "site.css": set(re.findall(r"@media \(max-width:\s*(\d+)px\)", open(os.path.join(WEB, "css", "site.css"), encoding="utf-8").read())),
          "etabli.js": set(re.findall(r'matchMedia\("\(max-width:\s*(\d+)px\)"\)', open(os.path.join(WEB, "js", "etabli.js"), encoding="utf-8").read()))}
chk("le seuil du téléphone est le même dans la charte, css/site.css et js/etabli.js (700 px)",
    all(v == {"700"} for v in seuils.values()), seuils)

# 5 ter. un nom de fichier rendu par le moteur (télécharger) ou tiré d'une note : jamais de chemin, jamais « .. »
NOMS = {"../../dossier/essai..json": "essai.json", "C:\\x\\a:b?.json": "a_b_.json", "...": "fichier", "note<script>.json": "note_script_.json",
        " .cache": "cache", "a/../../b": "b",
        # les caractères INVISIBLES : l'inversion du sens d'écriture (U+202E) ferait lire « facture[U+202E]nosj.exe » comme un
        # « .exe » ; les marques de sens, les séparateurs de ligne, l'espace sans largeur
        "facture\u202enosj.exe.json": "facture_nosj.exe.json", "a\u200fb\u2028c\u200bd.json": "a_b_c_d.json", "\u2066x\u2069.csv": "_x_.csv"}
essai = ("const m = await import(" + json.dumps("file://" + os.path.join(WEB, "js", "commun.js")) + ");"
         "console.log(JSON.stringify(" + json.dumps(list(NOMS)) + ".map((n) => m.nomDeFichier(n)).concat([m.nomDeFichier('a'.repeat(300) + '.json')])));")
try:
    sortie = subprocess.run(["node", "--input-type=module", "-e", essai], capture_output=True, text=True, timeout=60)
    lus = json.loads(sortie.stdout.strip().split("\n")[-1])
except Exception as e:  # noqa: BLE001
    lus = [str(e)]
chk("un nom de fichier venu du moteur est nettoyé (ni chemin, ni « .. », ni caractère réservé ni invisible ; l'extension survit)",
    lus[:-1] == list(NOMS.values()) and len(lus[-1]) == 150 and lus[-1].endswith(".json"), lus)

# 5 quater. le taux d'une note (en-tête du téléphone, carte de l'accueil) s'écrit comme les pastilles du moteur : la
# même fonction écrit les deux (js/commun.js, texteTaux), et elle écrit ce qu'écrit commun._taux_txt — « > 999 % » au
# delà du plafond, la valeur exacte au survol. Le plafond est une COPIE (deux langages) : liée ici.
sys.path.insert(0, RACINE)
from api_web import commun as CW  # noqa: E402
import html as _html  # noqa: E402

COMMUN_JS = open(os.path.join(WEB, "js", "commun.js"), encoding="utf-8").read()
plafond_js = re.findall(r"export const PLAFOND_TAUX = (\d+);", COMMUN_JS)
chk(f"le plafond des taux est le même à l'écran (js/commun.js) et dans le moteur (commun.PLAFOND_TAUX = {CW.PLAFOND_TAUX})",
    plafond_js == [str(CW.PLAFOND_TAUX)], plafond_js)
TAUX = [0, 64, 63.7, -3, 99.5, 100.2, 999, 999.4, 1000, 1040.8, 12345678, 101, 5.5]
essai = ("const m = await import(" + json.dumps("file://" + os.path.join(WEB, "js", "commun.js")) + ");"
         "console.log(JSON.stringify(" + json.dumps(TAUX) + ".map((t) => m.texteTaux(t)).concat([m.texteTaux(null), m.texteTaux(undefined), m.texteTaux('abc')])));")
try:
    sortie = subprocess.run(["node", "--input-type=module", "-e", essai], capture_output=True, text=True, timeout=60)
    ecrits = json.loads(sortie.stdout.strip().split("\n")[-1])
except Exception as e:  # noqa: BLE001
    ecrits = [str(e)] * (len(TAUX) + 3)


def _attendu(t):
    txt, titre = CW._taux_txt(t)
    return {"texte": _html.unescape(txt), "titre": _html.unescape(titre.split('"')[1]) if titre else ""}


ecarts = [(t, e, _attendu(t)) for t, e in zip(TAUX, ecrits) if e != _attendu(t)]
chk(f"l'écran écrit le taux comme la pastille du moteur ({len(TAUX)} taux, dont « > 999 % » et sa valeur exacte au survol)",
    not ecarts and len(ecrits) == len(TAUX) + 3, ecarts[:3] or ecrits[-3:])
chk("un taux absent ou illisible ne s'écrit pas (ni « null % », ni « NaN % »)", ecrits[-3:] == [{"texte": "", "titre": ""}] * 3, ecrits[-3:])
ETABLI_JS = open(os.path.join(WEB, "js", "etabli.js"), encoding="utf-8").read()
ACCUEIL_JS = open(os.path.join(WEB, "js", "accueil.js"), encoding="utf-8").read()
chk("l'en-tête du téléphone et la carte de l'accueil écrivent le taux par texteTaux (jamais String(taux) à la main)",
    "texteTaux(res.taux)" in ETABLI_JS and "texteTaux(r.taux)" in ACCUEIL_JS
    and not re.search(r"String\((res|r)\.taux\)", ETABLI_JS + ACCUEIL_JS))

# 5 quinquies. la mémoire des notes ABÎMÉE n'est jamais écrasée en silence (js/notes.js, joué par node, localStorage
# simulé). Une mémoire illisible n'est pas une mémoire vide : son texte brut est copié une fois à part
# (« stab.notes.v1.illisible.<heure> ») AVANT toute écriture, et si la copie échoue, rien n'est écrit et l'écriture est
# refusée (ecrire rend false, raison « illisible »). Défaut mesuré avant la correction : 30 notes, le JSON tronqué d'un
# octet, une écriture → 1 note, le texte d'origine perdu.
ESSAI_NOTES = r"""
const m = await import(%s);
class Memoire {
  constructor(refuser = () => null, depart = {}) { this.m = new Map(Object.entries(depart)); this.refuser = refuser; }
  get length() { return this.m.size; }
  key(i) { return [...this.m.keys()][i] ?? null; }
  getItem(k) { return this.m.has(k) ? this.m.get(k) : null; }
  setItem(k, v) { const e = this.refuser(k); if (e) throw e; this.m.set(k, String(v)); }
  removeItem(k) { this.m.delete(k); }
}
const CLE = "stab.notes.v1", t = 1762500000000;
const trente = {};
for (let i = 0; i < 30; i++) {
  const id = `n${String(i).padStart(2, "0")}`;
  trente[id] = { id, module: "poutre", nom: `Poutre P${i + 1}`, chantier: "Réalex", dossier: "Fondations", indice: "A", cree: t + i, modifie: t + i,
                 etat: { poutres: [{ nom: `P${i + 1}`, b: 25, h: 50 }] }, resume: { taux: 40 + i, niveau: "p" }, vignette: "" };
}
const entier = JSON.stringify(trente), abime = entier.slice(0, -1);
const plein = () => new DOMException("plein", "QuotaExceededError");
const out = { longueur: abime.length };
// chaque scène dans son propre localStorage simulé ; une exception y est notée, les autres scènes se jouent quand même
const jouer = (nom, memoire, f) => { globalThis.localStorage = memoire; try { out[nom] = f(memoire); } catch (e) { out[nom] = { erreur: String(e) }; } };
const copiesDe = (mem) => [...mem.m.keys()].filter((k) => k.startsWith("stab.notes.v1.illisible."));
// LE DÉFAUT MESURÉ PAR LE JURY, avec la seule API d'avant (nouvelle, ecrire) : 30 notes, le JSON tronqué d'un octet, une
// écriture → le texte d'origine doit survivre, sous sa clé ou dans une copie
jouer("jury", new Memoire(() => null, { [CLE]: abime }), (mem) => {
  const ok = m.ecrire(m.nouvelle({ module: "poutre", nom: "Nouvelle" }));
  return { ok, survit: [...mem.m.values()].includes(abime), cles: mem.m.size };
});
// A. la même mémoire abîmée, vue de près : lue « illisible » ; la copie, une fois, datée ; la mémoire neuve ensuite
jouer("copie", new Memoire(() => null, { [CLE]: abime }), (mem) => {
  const avant = { etat: m.memoire().etat, lisibles: Object.keys(m.toutes()).length, recentes: m.recentes().length };
  const ok = m.ecrire(m.nouvelle({ module: "poutre", nom: "Nouvelle" })), b = m.bilan();
  const ok2 = m.ecrire(m.nouvelle({ module: "poutre", nom: "Deuxième" }));
  return { avant, ok, ok2, b, copies: copiesDe(mem).length, liste: m.copies().map((c) => [c.taille, c.quand !== null]),
           texte: m.texteCopie(b.copie) === abime, apres: { etat: m.memoire().etat, notes: Object.keys(m.toutes()).length } };
});
// B. mémoire abîmée, la copie échoue : rien n'est écrit, l'écriture est refusée et dit pourquoi
jouer("sans_copie", new Memoire((k) => (k.startsWith("stab.notes.v1.illisible.") ? plein() : null), { [CLE]: abime }), (mem) => {
  const ok = m.ecrire(m.nouvelle({ module: "poutre" })), b = m.bilan ? m.bilan() : null, sup = m.supprimer("n00");
  return { ok, b, sup, intacte: mem.getItem(CLE) === abime, cles: [...mem.m.keys()] };
});
// C. autre chose qu'un objet de notes (un tableau) : illisible aussi
jouer("tableau", new Memoire(() => null, { [CLE]: "[1,2]" }), (mem) => {
  const etat = m.memoire ? m.memoire().etat : null, ok = m.ecrire(m.nouvelle({ module: "poutre" }));
  return { etat, ok, survit: [...mem.m.values()].includes("[1,2]") };
});
// D. mémoire absente puis lue : une écriture ordinaire, sans copie ; les 30 notes lisibles restent
jouer("absente", new Memoire(), (mem) => {
  const etat = m.memoire().etat, ok = m.ecrire(m.nouvelle({ module: "poutre" }));
  return { etat, ok, b: m.bilan(), cles: [...mem.m.keys()] };
});
jouer("lue", new Memoire(() => null, { [CLE]: entier }), (mem) => {
  const ok = m.ecrire(m.nouvelle({ module: "poutre" }));
  return { etat: m.memoire().etat, ok, notes: Object.keys(m.toutes()).length, copies: copiesDe(mem).length };
});
// E. stockage plein : refus « plein » ; l'import d'un .stab refuse d'ouvrir une note qu'il n'a pas pu enregistrer
jouer("plein", new Memoire(() => plein(), { [CLE]: entier }), (mem) => ({ ok: m.ecrire(m.nouvelle({ module: "poutre" })), b: m.bilan ? m.bilan() : null,
  intacte: mem.getItem(CLE) === entier }));
// F. rouvrir sans modifier : ecrire(note, {date: false}) réécrit la note (ses caches) SANS avancer sa date ; une écriture
// ordinaire (une modification) l'avance
jouer("date", new Memoire(), (mem) => {
  const n = m.nouvelle({ module: "poutre" });
  n.modifie = t;
  const ok = m.ecrire(n, { date: false }), garde = JSON.parse(mem.getItem(CLE))[n.id].modifie;
  const ok2 = m.ecrire(n), apres = JSON.parse(mem.getItem(CLE))[n.id].modifie;
  return { ok, ok2, garde: garde === t, avance: apres > t };
});
globalThis.localStorage = new Memoire(() => plein(), { [CLE]: entier });
const fichier = { text: async () => JSON.stringify({ format: "stab", version: 1, note: { module: "poutre", nom: "Importée", etat: { poutres: [] } } }) };
try { await m.importer(fichier); out.importer = "ouverte"; } catch (e) { out.importer = e.message; }
console.log(JSON.stringify(out));
""" % json.dumps("file://" + os.path.join(WEB, "js", "notes.js"))
try:
    sortie = subprocess.run(["node", "--input-type=module", "-e", ESSAI_NOTES], capture_output=True, text=True, timeout=60)
    JN = json.loads(sortie.stdout.strip().split("\n")[-1])
except Exception as e:  # noqa: BLE001
    JN = {"_": str(e), "stderr": locals().get("sortie") and sortie.stderr[-400:]}
J = JN.get("jury", {})
chk("notes.js : le défaut du jury — 30 notes, le JSON tronqué d'un octet, une écriture : le texte d'origine SURVIT (sous sa clé ou dans une copie)",
    J.get("ok") is True and J.get("survit") is True, (J, JN.get("stderr")))
A = JN.get("copie", {})
chk("notes.js : une mémoire abîmée (30 notes, le JSON tronqué d'un octet) se lit « illisible », sans note lisible — jamais « vide »",
    A.get("avant") == {"etat": "illisible", "lisibles": 0, "recentes": 0}, (A.get("avant"), JN.get("stderr")))
chk("notes.js : écrire sur une mémoire abîmée copie d'abord son texte brut à part (stab.notes.v1.illisible.<heure>) — le texte d'origine survit, "
    "à l'octet ; la note s'écrit ensuite dans une mémoire neuve",
    A.get("ok") is True and A.get("texte") is True and (A.get("b") or {}).get("copie", "").startswith("stab.notes.v1.illisible.")
    and A.get("apres") == {"etat": "lue", "notes": 2}, A)
chk("notes.js : la copie se fait UNE fois (deux écritures, une copie), datée et mesurée (copies())",
    A.get("ok2") is True and A.get("copies") == 1 and A.get("liste") == [[JN.get("longueur"), True]] and JN["longueur"] > 1000, (A.get("liste"), JN.get("longueur")))
B = JN.get("sans_copie", {})
chk("notes.js : si la copie échoue, RIEN n'est écrit — ecrire et supprimer rendent false, raison « illisible » ; la mémoire abîmée reste intacte",
    B.get("ok") is False and B.get("sup") is False and (B.get("b") or {}).get("raison") == "illisible" and B.get("intacte") is True
    and B.get("cles") == ["stab.notes.v1"], B)
chk("notes.js : autre chose qu'un objet de notes (un tableau JSON) est illisible aussi : copié avant d'écrire",
    JN.get("tableau") == {"etat": "illisible", "ok": True, "survit": True}, JN.get("tableau"))
# (11/11/2026 : la note créée s'inscrit aussi au JOURNAL d'activité, sous sa propre clé — aucune copie « .illisible. »)
chk("notes.js : une mémoire absente s'écrit sans copie (les notes, et la ligne du journal sous sa clé) ; une mémoire lue garde ses 30 notes et n'est pas copiée",
    (JN.get("absente") or {}).get("etat") == "absente" and JN["absente"].get("ok") is True
    and sorted(JN["absente"].get("cles") or []) == ["stab.journal.v1", "stab.notes.v1"]
    and JN.get("lue") == {"etat": "lue", "ok": True, "notes": 31, "copies": 0}, (JN.get("absente"), JN.get("lue")))
chk("notes.js : écrire sans dater (rouvrir une note sans la modifier) garde sa date ; une modification l'avance",
    JN.get("date") == {"ok": True, "ok2": True, "garde": True, "avance": True}, JN.get("date"))
chk("notes.js : stockage plein → ecrire rend false, raison « plein », la mémoire intacte ; un .stab importé qui ne s'enregistre pas ne s'ouvre "
    "pas (l'établi l'aurait remplacé par une note neuve, sans un mot)",
    (JN.get("plein") or {}).get("ok") is False and (JN["plein"].get("b") or {}).get("raison") == "plein" and JN["plein"].get("intacte") is True
    and "pas pu être enregistrée" in str(JN.get("importer")) and "plein" in str(JN.get("importer")), (JN.get("plein"), JN.get("importer")))
NOTES_JS = open(os.path.join(WEB, "js", "notes.js"), encoding="utf-8").read()
# (11/11/2026 : trois gestes passent la porte — ecrire, supprimer, et reclasser, qui renomme un projet sur toutes ses
# notes ; chacun garde le verdict de changer() pour inscrire le geste au journal SEULEMENT s'il est écrit)
chk("notes.js : toute écriture de la mémoire passe par changer() (une seule porte : ecrire, supprimer, reclasser)",
    len(re.findall(r"localStorage\.setItem\(CLE", NOTES_JS)) == 1 and len(re.findall(r"\bconst ok = changer\(", NOTES_JS)) == 3
    and len(re.findall(r"(?<!function )\bchanger\(", NOTES_JS)) == 3 and len(re.findall(r"localStorage\.setItem\(", NOTES_JS)) == 2,
    (re.findall(r"localStorage\.setItem\([^)]*\)", NOTES_JS), len(re.findall(r"(?<!function )\bchanger\(", NOTES_JS))))

# 6. la politique de sécurité (_headers) et ce qui la rend possible : aucun script écrit dans une page, aucun
# gestionnaire « on…= » dans les pages ni dans les fragments que Python rend (la politique les bloquerait : un bouton
# muet en ligne, et un nom piégé qui s'exécuterait si la politique manquait)
sys.path.insert(0, os.path.join(WEB, "outils"))
import serveur_local as SL  # noqa: E402

from beton.maturite import meteo as MTO  # noqa: E402  (les adresses d'Open-Meteo : la source)

HOTES_METEO = [re.match(r"(https://[^/]+)/", u).group(1) for u in MTO.ADRESSES.values()]
for url in ("/index.html", "/poutre.html", "/maturite-beton.html", "/js/travailleur.js", f"/{SW.PYODIDE}/pyodide.asm.wasm", "/moteur/moteur.zip"):
    csp = SL.entetes(url).get("Content-Security-Policy", "")
    d = {x.split()[0]: x.split()[1:] for x in (y.strip() for y in csp.split(";")) if x}
    # connect-src : le site lui-même, et EXACTEMENT les trois adresses d'Open-Meteo (la météo du lieu, des DONNÉES
    # importées à la demande) — rien d'autre ; aucun script, aucune feuille, aucune image n'en vient
    chk(f"{url} : politique de sécurité stricte (tout du site, aucun script en ligne, WebAssembly seul permis ; connect-src : le site et les "
        "trois adresses d'Open-Meteo, rien d'autre)",
        d.get("default-src") == ["'self'"] and d.get("script-src") == ["'self'", "'wasm-unsafe-eval'"] and d.get("object-src") == ["'none'"]
        and d.get("frame-ancestors") == ["'none'"] and "'unsafe-eval'" not in csp and d.get("connect-src") == ["'self'"] + HOTES_METEO
        and not any("open-meteo" in " ".join(v) for k, v in d.items() if k != "connect-src"), csp)
# le filet du démarrage est une feuille EN LIGNE (§ 3) : la politique de sécurité des pages doit l'admettre, sinon le
# message ne s'afficherait jamais (et un script en ligne, lui, reste refusé)
_csp_page = {x.split()[0]: x.split()[1:] for x in (y.strip() for y in SL.entetes("/poutre.html").get("Content-Security-Policy", "").split(";")) if x}
chk("la politique de sécurité des pages admet les styles en ligne (le filet du démarrage), pas les scripts en ligne",
    _csp_page.get("style-src") == ["'self'", "'unsafe-inline'"] and "'unsafe-inline'" not in _csp_page.get("script-src", []), _csp_page)
# les moteurs de recherche : ni indexation ni liens suivis, sur TOUTE adresse (petits chantiers, 10/11/2026) — l'en-tête
# X-Robots-Tag de _headers, que outils/serveur_local.py applique comme Cloudflare
_robots = {u: SL.entetes(u).get("X-Robots-Tag") for u in ("/", "/index.html", "/poutre", "/garde-corps.html", "/bibliotheque-beton", "/js/etabli.js",
                                                           "/css/site.css", "/moteur/moteur.zip", f"/{SW.PYODIDE}/pyodide.asm.wasm", "/favicon.svg")}
chk("X-Robots-Tag: noindex, nofollow sur toute adresse du site (pages, scripts, feuilles, moteur, cœur Python) — une fois chacune",
    all(v == "noindex, nofollow" for v in _robots.values()), _robots)
# les trois adresses : une SOURCE (le moteur), deux copies NOMMÉES (l'écran, la politique de sécurité), liées ici
METEO_JS = open(os.path.join(WEB, "js", "meteo.js"), encoding="utf-8").read()
copie_js = dict(re.findall(r'^\s+(geocodage|archive|prevision): "(https://[^"]+)",$', METEO_JS, re.M))
chk("les adresses d'Open-Meteo sont les mêmes dans le moteur (meteo.ADRESSES), l'écran (js/meteo.js) et la politique de sécurité (_headers)",
    copie_js == MTO.ADRESSES and len(HOTES_METEO) == 3 and len(set(HOTES_METEO)) == 3, (copie_js, MTO.ADRESSES))
chk("seul js/meteo.js appelle l'extérieur : aucun autre script de l'écran ne nomme un hôte ni n'appelle fetch vers un tiers",
    all("open-meteo" not in open(os.path.join(WEB, "js", f), encoding="utf-8").read() for f in os.listdir(os.path.join(WEB, "js")) if f != "meteo.js")
    and 'from "./meteo.js"' in open(os.path.join(WEB, "js", "etabli.js"), encoding="utf-8").read())
# js/meteo.js joué par node, fetch simulé : ce qui part (adresses, paramètres), ce qui revient (réponses brutes), les pannes
ESSAI_METEO = r"""
const m = await import(%s);
const plan = %s;
const journal = [];
const geo = {results: [{name: "Bruxelles", admin1: "Bruxelles-Capitale", country: "Belgique", latitude: 50.85045, longitude: 4.34878}]};
const jour = (u) => ({daily: {time: [new URL(u).searchParams.get("start_date")], temperature_2m_mean: [7.5]}});
// de VRAIES réponses (Response de node, comme celles de fetch) : le corps se lit comme dans le navigateur
const reponse = (status, corps, entetes = {}) => new Response(typeof corps === "string" || corps instanceof ReadableStream ? corps : JSON.stringify(corps),
  {status, headers: {"content-type": "application/json", ...entetes}});
// une réponse ÉNORME (50 Mo, par morceaux de 64 ko, sans taille annoncée) : on compte ce qui en est tiré
let tire = 0;
const enorme = () => new ReadableStream({pull(c) { if (tire >= 50 * 1024 * 1024) return c.close(); const b = new Uint8Array(65536).fill(32); tire += b.byteLength; c.enqueue(b); }});
const service = (mode) => async (u, init) => {
  journal.push(u);
  if (mode === "reseau") throw new TypeError("Failed to fetch");
  if (mode === "lent") return new Promise((ok, ko) => init.signal.addEventListener("abort", () => ko(Object.assign(new Error("abort"), {name: "AbortError"}))));
  if (mode === "429" || (mode === "429-puis" && journal.filter((x) => x === u).length === 1)) return reponse(429, {reason: "Too many"});
  if (mode === "500") return reponse(500, {error: true, reason: "Panne <b>simulée</b>"});
  if (mode === "illisible") return reponse(200, "pas du JSON {");
  if (mode === "enorme") return reponse(200, enorme());
  if (mode === "annonce") return reponse(200, enorme(), {"content-length": String(50 * 1024 * 1024)});
  if (u.startsWith(m.ADRESSES.geocodage)) return reponse(200, mode === "inconnu" ? {generationtime_ms: 0.4} : geo);
  return reponse(200, jour(u));
};
const out = {};
for (const mode of ["bon", "inconnu", "reseau", "lent", "429", "429-puis", "500", "illisible", "enorme", "annonce"]) {
  tire = 0;
  journal.length = 0;
  const attentes = [];
  const r = await m.recupererMeteo(plan, {fetch: service(mode), attendre: async (ms) => attentes.push(ms), delai: 50});
  out[mode] = {r, journal: [...journal], attentes, tire};
}
out.erreur = await m.recupererMeteo({erreur: "Période entièrement au-delà"}, {fetch: service("bon")});
out.malicieux = await m.recupererMeteo({...plan, appels: [{source: "ailleurs", parametres: {}}]}, {fetch: service("bon")});
journal.length = 0;
out.parametre = await m.recupererMeteo({...plan, appels: [{source: "archive", parametres: {start_date: "2026-11-01", callback: "x"}}]}, {fetch: service("bon")});
out.parametre_journal = [...journal];
journal.length = 0;
out.position = await m.recupererMeteo({...plan, lieu: undefined, geocodage: undefined, position: {latitude: 50.5, longitude: 4.9}}, {fetch: service("bon")});
out.position_journal = [...journal];
out.adresse = m.adresse("geocodage", {name: "Saint-Gilles & Co/1 «é»", count: 1, language: "fr"});
console.log(JSON.stringify(out));
"""
PLAN_M = MTO.plan("Bruxelles, Belgique", "2026-10-24", "2026-11-07", "2026-11-07")
try:
    sortie = subprocess.run(["node", "--input-type=module", "-e", ESSAI_METEO % (json.dumps("file://" + os.path.join(WEB, "js", "meteo.js")), json.dumps(PLAN_M))],
                            capture_output=True, text=True, timeout=60)
    JM = json.loads(sortie.stdout.strip().split("\n")[-1])
except Exception as e:  # noqa: BLE001
    JM = {"_": str(e), "stderr": locals().get("sortie") and sortie.stderr[-400:]}
bon = JM.get("bon", {})
att_urls = [MTO.ADRESSES["geocodage"] + "?name=Bruxelles%2C+Belgique&count=1&language=fr"] + [
    MTO.ADRESSES[a["source"]] + "?" + "&".join(f"{k}={str(v).replace(':', '%3A')}" for k, v in MTO.parametres(a, {"latitude": 50.85045, "longitude": 4.34878}).items())
    for a in PLAN_M["appels"]]
chk("js/meteo.js : le géocodage puis chaque appel du plan, vers les adresses d'Open-Meteo, paramètres dans l'ordre de l'écran (latitude, longitude, dates, fuseau, variable), encodés",
    bon.get("journal") == att_urls, (bon.get("journal"), att_urls, JM.get("stderr")))
chk("js/meteo.js : les réponses BRUTES partent au module avec le plan et le lieu ({lieu, plan, geocodage, reponses})",
    bon.get("r", {}).get("ok") and bon["r"]["valeur"]["plan"] == PLAN_M and bon["r"]["valeur"]["lieu"] == "Bruxelles, Belgique"
    and bon["r"]["valeur"]["geocodage"]["results"][0]["name"] == "Bruxelles" and len(bon["r"]["valeur"]["reponses"]) == 2, bon.get("r"))
chk("js/meteo.js : un lieu introuvable part au module sans réponses (c'est le module qui le dit), sans autre appel",
    JM.get("inconnu", {}).get("r", {}).get("valeur", {}).get("reponses") == [] and len(JM["inconnu"]["journal"]) == 1, JM.get("inconnu"))
MSG = {"reseau": "Pas de réseau", "lent": "ne répond pas", "429": "saturé", "500": "Service météo en panne (erreur 500 — Panne <b>simulée</b>)",
       "illisible": "Réponse illisible"}
chk("js/meteo.js : pas de réseau, service muet (délai), saturé (429 deux fois), en panne (500 et sa raison), réponse illisible → un message en "
    "français, jamais une exception ; le calcul hors ligne est rappelé",
    all(not JM.get(k, {}).get("r", {}).get("ok", True) and v in JM[k]["r"]["erreur"] and "hors ligne" in JM[k]["r"]["erreur"] for k, v in MSG.items()),
    {k: JM.get(k, {}).get("r") for k in MSG})
chk("js/meteo.js : un « trop de demandes » (429) est réessayé une fois, 2 s après (comme l'écran) — chaque appel du plan, puis il passe ; deux 429 de suite arrêtent tout",
    JM.get("429-puis", {}).get("r", {}).get("ok") and JM["429-puis"]["attentes"] == [2000] * 3 and len(JM["429-puis"]["journal"]) == 6
    and JM.get("429", {}).get("attentes") == [2000] and len(JM["429"]["journal"]) == 2, {k: JM.get(k, {}).get("attentes") for k in ("429", "429-puis")})
chk("js/meteo.js : un plan impossible (dates hors de portée) rend son message, sans appel ; une source inconnue ou un paramètre non permis sont refusés",
    JM.get("erreur") == {"ok": False, "erreur": "Période entièrement au-delà"} and not JM.get("malicieux", {}).get("ok", True)
    and not JM.get("parametre", {}).get("ok", True) and "callback" in JM["parametre"]["erreur"]
    and all("callback" not in u for u in JM.get("parametre_journal", ["callback"])), (JM.get("malicieux"), JM.get("parametre")))
# (relecture adverse : ce contrôle ne regardait que « ok » — un écran qui géocodait quand même le passait ; il lit
# maintenant le journal des appels : aucun géocodage, chaque appel du plan à la position donnée)
chk("js/meteo.js : un plan qui donne la position (le détail horaire) ne géocode pas — chaque appel part à la position donnée ; les "
    "paramètres sont encodés",
    JM.get("position", {}).get("ok") and len(JM.get("position_journal", [])) == len(PLAN_M["appels"])
    and not any("geocoding" in u for u in JM["position_journal"]) and all("latitude=50.5&longitude=4.9&" in u for u in JM["position_journal"])
    and JM.get("adresse") == MTO.ADRESSES["geocodage"] + "?name=Saint-Gilles+%26+Co%2F1+%C2%AB%C3%A9%C2%BB&count=1&language=fr",
    (JM.get("position"), JM.get("position_journal"), JM.get("adresse")))
# une réponse énorme (relecture adverse : elle était lue tout entière, puis envoyée au travailleur) : la lecture s'arrête à
# 5 Mo, rien ne part au moteur, le message le dit ; annoncée trop grosse, elle n'est même pas lue
ENO = JM.get("enorme", {})
chk("js/meteo.js : une réponse énorme (50 Mo) n'est lue que jusqu'à 5 Mo, puis refusée en message ; annoncée trop grosse, elle n'est pas lue du tout",
    ENO.get("r", {}).get("ok") is False and "trop volumineuse" in ENO["r"].get("erreur", "") and "hors ligne" in ENO["r"]["erreur"]
    and 5 * 1024 * 1024 < ENO.get("tire", 0) <= 5 * 1024 * 1024 + 4 * 65536 and len(ENO.get("journal", [])) == 1
    and JM.get("annonce", {}).get("r", {}).get("ok") is False and "trop volumineuse" in JM["annonce"]["r"].get("erreur", "")
    and JM["annonce"].get("tire", 99e9) <= 4 * 65536,
    {k: (JM.get(k, {}).get("r"), JM.get(k, {}).get("tire")) for k in ("enorme", "annonce")})
chk("le cœur Python reste en cache, le paquet du moteur est relu à chaque ouverture",
    "immutable" in SL.entetes(f"/{SW.PYODIDE}/pyodide.asm.wasm").get("Cache-Control", "")
    and SL.entetes("/moteur/manifeste.json").get("Cache-Control") == "no-cache")
# les scripts et les feuilles sont REVALIDÉS (C7) : jamais un ancien script après une mise en ligne — une seule règle
# chacun (un en-tête joint « no-cache, … » serait le signe d'une règle en double)
_cache = {u: SL.entetes(u).get("Cache-Control") for u in ("/js/app.js", "/js/etabli.js", "/js/modules.js", "/css/site.css", "/css/modules/dalle.css",
                                                            "/moteur/apercus/poutre.json")}
chk("/js/* et /css/* (et les aperçus, sous /moteur/*) : Cache-Control: no-cache — une revalidation à chaque ouverture ; outils/serveur_local.py "
    "applique _headers", all(v == "no-cache" for v in _cache.values()), _cache)
EN_LIGNE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>|\son[a-z]+\s*=", re.I)
fautes = [p for p in ["index.html"] + list(SW.PAGES.values()) if EN_LIGNE.search(open(os.path.join(WEB, p), encoding="utf-8").read())]
chk("aucune page ne porte de script en ligne ni de gestionnaire « on…= »", not fautes, fautes)
sys.path.insert(0, RACINE)
from api_web import routeur as RT  # noqa: E402

fautes = []
for mod in SW.ADRESSES:
    out = json.loads(RT.traiter(json.dumps({"module": mod, "action": "nouveau"})))
    for k in ("resultat", "volets", "donnees", "elements", "statut_html"):
        if EN_LIGNE.search(out[k]):
            fautes.append((mod, k, EN_LIGNE.search(out[k]).group(0)))
    fautes += [(mod, "vue", v["cle"]) for v in out["vues"] if EN_LIGNE.search(v["svg"])]
chk("les fragments que Python rend n'ont ni script ni gestionnaire « on…= »", not fautes, fautes[:3])

# 7. les adresses sans « .html » (Cloudflare Pages redirige « x.html » vers « x » ; la note ouverte n'a pas à survivre
# à une redirection) : les pages générées et js/notes.js visent les mêmes adresses — deux copies, liées ici
modules_js = open(os.path.join(WEB, "js", "modules.js"), encoding="utf-8").read()
# (S4 : js/modules.js exporte aussi BIBLIOTHEQUES : la table MODULES est lue sur SA ligne — le contrôle suit)
pages_js = {k: v["page"] for k, v in json.loads(re.search(r"^export const MODULES = (\{.*\});$", modules_js, re.M).group(1)).items()}
chk("js/modules.js vise les adresses du générateur (sans « .html »), et notes.js le lit",
    pages_js == SW.ADRESSES and 'from "./modules.js"' in open(os.path.join(WEB, "js", "notes.js"), encoding="utf-8").read(), (pages_js, SW.ADRESSES))
# … et les BIBLIOTHÈQUES (le routage de js/app.js) : l'adresse de construire.ADRESSES_BIBLIOTHEQUES est celle que
# chaque module de rendu déclare (api_web/biblio_*.PAGE) — deux copies, liées ici (R2) — et leur titre, celui du site
_bib_js = json.loads(re.search(r"^export const BIBLIOTHEQUES = (\{.*\});$", modules_js, re.M).group(1))
chk("js/modules.js exporte BIBLIOTHEQUES (clé → page, titre) ; l'adresse est celle que api_web/biblio_*.py déclare (PAGE)",
    _bib_js == {b: {"page": SW.ADRESSES_BIBLIOTHEQUES[b], "titre": t} for b, (_c, t, _f) in SW.BIBLIOTHEQUES.items()}
    and all(importlib.import_module(c).PAGE == SW.ADRESSES_BIBLIOTHEQUES[b] for b, (c, _t, _f) in SW.BIBLIOTHEQUES.items())
    and sorted(_bib_js) == sorted(SW.BIBLIOTHEQUES) == ["acier", "autres", "beton"], _bib_js)       # (+ « autres », 11/11/2026)
liens = [(p, h) for p in ["index.html"] + list(SW.PAGES.values())
         for h in re.findall(r'<a [^>]*href="([^"]+)"', open(os.path.join(WEB, p), encoding="utf-8").read()) if h.endswith(".html")]
chk("aucun lien des pages ne vise une adresse en « .html »", not liens, liens[:4])

# 8. la mise en ligne par la méthode « Workers » de Cloudflare : le projet s'appelle stab-methodes, son dossier est
# site/web, servi tel quel (adresses sans « .html ») ; les outils de développement ne sont pas publiés
wr = json.loads(re.sub(r"^\s*//.*$", "", open(os.path.join(WEB, "wrangler.jsonc"), encoding="utf-8").read(), flags=re.M))
chk("wrangler.jsonc : le projet stab-methodes publie ce dossier, adresses sans « .html »",
    wr.get("name") == "stab-methodes" and wr.get("assets", {}).get("directory") == "."
    and wr["assets"].get("html_handling") == "auto-trailing-slash" and "main" not in wr, wr)
ignore = {l.strip() for l in open(os.path.join(WEB, ".assetsignore"), encoding="utf-8") if l.strip() and not l.startswith("#")}
chk("les outils de développement et la configuration ne sont pas publiés (.assetsignore)",
    {"wrangler.jsonc", "README.md", "construire.py", "verifier_site.js", "outils", "__pycache__", ".wrangler"} <= ignore
    and not ignore & {"index.html", "js", "css", "vendor", "moteur", "_headers"}, sorted(ignore))
# (revue S4, deploiement-07) la liste fixe laissait passer verifier_demarrage.js, harnais posé à la racine du site : une
# RÈGLE, qui rougira aussi sur le prochain. Un script .js à la racine du dossier publié qu'AUCUNE page ne charge est un
# outil (un harnais, une sonde) : exclu de la mise en ligne ; un script qu'une page charge ne l'est pas (la page casserait).
# .assetsignore se lit comme Cloudflare le lit (motifs à la gitignore) : un nom ou un motif « * » sans « / » vaut pour le
# nom du fichier, quel que soit le dossier.


def _exclu(nom, motifs):
    return any(fnmatch.fnmatchcase(nom, m.strip("/")) for m in motifs if "/" not in m.strip("/"))


_pages_publiees = {f: open(os.path.join(WEB, f), encoding="utf-8").read() for f in os.listdir(WEB) if f.endswith(".html")}


def _pages_qui_chargent(js):
    motif = re.compile(r'(?:src|href)="(?:\./|/)?' + re.escape(js) + r'(?:[?#][^"]*)?"')
    return sorted(p for p, t in _pages_publiees.items() if motif.search(t))


_js_racine = sorted(f for f in os.listdir(WEB) if f.endswith(".js") and os.path.isfile(os.path.join(WEB, f)))
_publies_sans_page = [f for f in _js_racine if not _pages_qui_chargent(f) and not _exclu(f, ignore)]
_charges_exclus = [f for f in _js_racine if _pages_qui_chargent(f) and _exclu(f, ignore)]
chk(f"chaque script .js de la racine du site qu'aucune page ne charge est exclu de la mise en ligne, et aucun script chargé "
    f"ne l'est ({', '.join(_js_racine)})", "verifier_site.js" in _js_racine and len(_pages_publiees) >= 21
    and not _publies_sans_page and not _charges_exclus, {"publiés sans être chargés": _publies_sans_page, "chargés mais exclus": _charges_exclus})

# 9. chaque module du registre a sa place dans les familles de l'accueil (nom, icône) : sinon il serait écarté du site
from api_web import registre as REG2  # noqa: E402
hors = [k for k in REG2.cles() if k not in SW.ADRESSES]
chk("chaque module du registre est une clé des familles de l'accueil", not hors, hors)

# 10. le harnais (verifier_site.js, contrôle « l'état verrouillé des briques ») charge api_web.commun avec le python3 du
# SYSTÈME (3.11 ici), pas avec celui du site : un antislash dans l'expression d'une f-string y est une erreur de syntaxe
# (Python < 3.12). Le harnais complet du 10/11/2026 l'a trouvé dans le sélecteur (ligne inactive) ; ce contrôle le dit en
# une seconde au lieu d'une heure. Un python3 absent ou déjà en 3.12 rend le contrôle sans objet (dit tel quel).
try:
    _py3 = subprocess.run(["python3", "-c", "import sys; sys.path.insert(0, sys.argv[1]); from api_web import commun; "
                           "print('.'.join(map(str, sys.version_info[:2])))", RACINE], capture_output=True, text=True, timeout=120)
    chk("api_web/commun.py se charge avec le python3 du système (celui du harnais, contrôle des briques)",
        _py3.returncode == 0, (_py3.stderr or _py3.stdout).strip()[-300:] or f"python3 {_py3.stdout.strip()}")
except FileNotFoundError:
    chk("api_web/commun.py se charge avec le python3 du système (celui du harnais, contrôle des briques)", True, "pas de python3 : sans objet")

# 11. le passage COMPLET du harnais (verifier_site.js sans filtre) joue AUSSI verifier_demarrage.js, à la fin, et son
# verdict compte (revue S4, preuves-demarrage-hors-complet : le filtre de la promesse centrale du lot — un moteur, pas de
# rechargement — n'était joué que si on le demandait) ; un contrôle qui a REFUSÉ de conclure (« temps : non mesurable »,
# la charge de la machine) n'est jamais un succès silencieux : code 3, nommé au bilan. Les fonctions de la décision sont
# PURES et exportées : jouées ici par node, playwright remplacé par un bouchon (le parcours ne part pas : require.main
# n'est pas le harnais) ; l'appel, lui, se lit dans le texte de principal() — là où il doit être : à la fin.
_VS = open(os.path.join(WEB, "verifier_site.js"), encoding="utf-8").read()
ESSAI_PASSAGE = r"""
const Module = require('module');
const charger = Module._load;
Module._load = function (q, ...r) { return q === 'playwright' ? { chromium: null, devices: {} } : charger.call(this, q, ...r); };
const V = require(process.argv[1]);
const T = 'KO  demarrage · PC · temps : non mesurable  charge sur 1 min 3.40 ≥ 3.00 (0,75 × 4 cœurs) pendant 60 s';
const A = 'KO  demarrage · PC · aperçu : visible au plus 300 ms après DOMContentLoaded  {"ms":420}';
console.log(JSON.stringify({
  sous: ['', 'demarrage', 'demarrage:temps,apercu', 'poutre', 'formats:poutre', 'demarrages', 'module:demarrage'].map(V.sousFiltresDemarrage),
  bilans: [[0, 'OK  x\ndemarrage : 0 KO'], [1, T + '\ndemarrage : 1 KO'], [1, T + '\n' + A], [2, ''], [1, ''], [0, A]].map(([s, t]) => V.bilanDemarrage(s, t).code),
  nonMesures: V.bilanDemarrage(1, T).nonMesures,
  codes: [[0, { code: 0 }], [0, { code: 1 }], [0, { code: 3 }], [2, { code: 0 }], [2, { code: 3 }]].map(([k, d]) => V.codeSortie(k, d)),
}));
"""
_sortie_passage = None
try:
    _sortie_passage = subprocess.run(["node", "-e", ESSAI_PASSAGE, os.path.join(WEB, "verifier_site.js")], capture_output=True, text=True, timeout=60)
    _passage = json.loads(_sortie_passage.stdout.strip().split("\n")[-1])
except Exception as e:  # noqa: BLE001
    _passage = {"erreur": str(e), "stderr": _sortie_passage.stderr[-300:] if _sortie_passage else ""}
chk("verifier_site.js : sans filtre, le passage joue TOUS les sous-filtres de demarrage ; « demarrage[:…] » les siens ; aucun autre filtre ne le joue",
    _passage.get("sous") == ["", "", "temps,apercu", None, None, None, None], _passage)
chk("verifier_site.js : un « non mesurable » seul rend le passage NON CONCLUANT (code 3, nommé) — jamais un succès ; un vrai KO, ou un passage "
    "qui a planté, l'emporte (code 1)", _passage.get("bilans") == [0, 3, 1, 1, 1, 1] and _passage.get("codes") == [0, 1, 3, 1, 1]
    and len(_passage.get("nonMesures") or []) == 1 and "temps : non mesurable" in _passage["nonMesures"][0], _passage)
_principal = _VS[_VS.index("const principal = async () => {"):_VS.index("if (require.main === module && !DELEGUE) principal();")]
_fin = _principal[_principal.rindex("await b.close();"):]
chk("verifier_site.js : le passage complet joue verifier_demarrage.js À LA FIN de principal() (un navigateur à la fois), et son verdict "
    "entre dans le code de sortie",
    "const DEMARRAGE = path.join(__dirname, 'verifier_demarrage.js');" in _VS and "spawn(process.execPath, [DEMARRAGE, BASE, OUT, sous]" in _VS
    and re.search(r"if \(!FILTRE\) \{[^}]*?demarrage = await jouerDemarrage\(sousFiltresDemarrage\(FILTRE\)\);", _fin, re.S) is not None
    and _fin.count("process.exit(") == 1 and "process.exit(codeSortie(ko, demarrage));" in _fin, _fin[:400])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
