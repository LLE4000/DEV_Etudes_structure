# -*- coding: utf-8 -*-
"""Le site « Stab Méthodes » (site/web) : ce qui doit rester VRAI entre deux constructions.

- le paquet du moteur (moteur/moteur.zip) contient EXACTEMENT les fichiers actuels du dépôt que l'API charge : un
  fichier du moteur modifié sans reconstruire le site, et ce test rougit (le navigateur calculerait avec l'ancien) ;
- le manifeste porte l'empreinte du paquet (le navigateur la vérifie avant de l'exécuter) ;
- le paquet se suffit : déballé seul, sans le dépôt et sans les bibliothèques lourdes, il calcule ;
- les pages et la charte sont celles que le générateur écrit (css/charte.css = la charte des maquettes) ;
- le cœur Python déposé (vendor) a les empreintes relevées, et la roue numpy celle du pyodide-lock officiel ;
- chaque élément que les scripts de l'écran cherchent par son identifiant existe dans sa page.

    python3 tests/test_site_web.py
"""
import hashlib
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
attendus = SW.fichiers_du_moteur()
with zipfile.ZipFile(zip_chemin) as z:
    noms = sorted(z.namelist())
    differents = [n for n in noms if n in attendus and z.read(n) != open(os.path.join(RACINE, n), "rb").read()]
chk("le paquet contient exactement les fichiers que l'API charge", noms == sorted(attendus),
    (sorted(set(attendus) - set(noms))[:5], sorted(set(noms) - set(attendus))[:5]))
chk("chaque fichier du paquet est identique à celui du dépôt (sinon : python3 site/web/construire.py)", not differents, differents[:5])
chk("le manifeste porte l'empreinte du paquet", man["sha256"] == sha(zip_chemin), (man["sha256"][:12], sha(zip_chemin)[:12]))
chk("le manifeste liste les fichiers du paquet", man["fichiers"] == noms)

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
from api_web import routeur as R
import api_web
assert api_web.__file__.startswith({tmp!r}), api_web.__file__
out = {{}}
for m in sorted(R.MODULES):
    r = json.loads(R.traiter(json.dumps({{"module": m, "action": "nouveau"}})))
    out[m] = (r["ok"], r.get("erreur"), (r.get("resume") or {{}}).get("taux"))
print(json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", essai], capture_output=True, text=True, cwd=tmp)
    try:
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        res = {"_": (False, p.stderr[-400:], None)}
chk("déballé seul, le paquet calcule chaque module", res and all(v[0] for v in res.values()), res)
chk("le module garde-corps est dans le paquet", "garde_corps" in res, list(res))

# 3. les pages et la charte
chk("css/charte.css est la charte des maquettes", sha(os.path.join(WEB, "css", "charte.css")) == sha(os.path.join(RACINE, "site", "maquettes", "charte.css")))
for nom, html in (("index.html", SW.page_accueil()), ("garde-corps.html", SW.page_module("garde_corps", "Garde-corps")),
                  ("poutre.html", SW.page_module("poutre", "Poutres béton armé"))):
    chk(f"{nom} est celle que le générateur écrit", open(os.path.join(WEB, nom), encoding="utf-8").read() == html)

# 4. le cœur Python déposé
v = os.path.join(WEB, SW.PYODIDE)
emp = json.load(open(os.path.join(v, "empreintes.json"), encoding="utf-8"))
faux = [f for f, h in emp["fichiers"].items() if not os.path.exists(os.path.join(v, f)) or sha(os.path.join(v, f)) != h]
chk(f"les {len(emp['fichiers'])} fichiers de Pyodide ont les empreintes relevées", not faux, faux)
lock = json.load(open(os.path.join(v, "pyodide-lock.json"), encoding="utf-8"))
npy = lock["packages"]["numpy"]
chk("la roue numpy a l'empreinte du pyodide-lock officiel", sha(os.path.join(v, npy["file_name"])) == npy["sha256"])
chk("le travailleur charge la version déposée", f'"../{SW.PYODIDE}/pyodide.mjs"' in open(os.path.join(WEB, "js", "travailleur.js"), encoding="utf-8").read())

# 5. les identifiants que les scripts cherchent existent dans leur page
for js, page in (("etabli.js", "garde-corps.html"), ("etabli.js", "poutre.html"), ("accueil.js", "index.html")):
    src = open(os.path.join(WEB, "js", js), encoding="utf-8").read()
    html = open(os.path.join(WEB, page), encoding="utf-8").read()
    ids = sorted(set(re.findall(r'\$\("#([\w-]+)', src)) | set(re.findall(r'getElementById\("([\w-]+)"', src)))
    absents = [i for i in ids if f'id="{i}"' not in html]
    chk(f"{js} → {page} : les {len(ids)} identifiants cherchés existent", not absents, absents)

# 6. la politique de sécurité (_headers) et ce qui la rend possible : aucun script écrit dans une page, aucun
# gestionnaire « on…= » dans les pages ni dans les fragments que Python rend (la politique les bloquerait : un bouton
# muet en ligne, et un nom piégé qui s'exécuterait si la politique manquait)
sys.path.insert(0, os.path.join(WEB, "outils"))
import serveur_local as SL  # noqa: E402

for url in ("/index.html", "/poutre.html", "/js/travailleur.js", f"/{SW.PYODIDE}/pyodide.asm.wasm", "/moteur/moteur.zip"):
    csp = SL.entetes(url).get("Content-Security-Policy", "")
    d = {x.split()[0]: x.split()[1:] for x in (y.strip() for y in csp.split(";")) if x}
    chk(f"{url} : politique de sécurité stricte (tout du site, aucun script en ligne, WebAssembly seul permis)",
        d.get("default-src") == ["'self'"] and d.get("script-src") == ["'self'", "'wasm-unsafe-eval'"] and d.get("object-src") == ["'none'"]
        and d.get("frame-ancestors") == ["'none'"] and "'unsafe-eval'" not in csp and d.get("connect-src") == ["'self'"], csp)
chk("le cœur Python reste en cache, le paquet du moteur est relu à chaque ouverture",
    "immutable" in SL.entetes(f"/{SW.PYODIDE}/pyodide.asm.wasm").get("Cache-Control", "")
    and SL.entetes("/moteur/manifeste.json").get("Cache-Control") == "no-cache")
EN_LIGNE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>|\son[a-z]+\s*=", re.I)
fautes = [p for p in ("index.html", "garde-corps.html", "poutre.html") if EN_LIGNE.search(open(os.path.join(WEB, p), encoding="utf-8").read())]
chk("aucune page ne porte de script en ligne ni de gestionnaire « on…= »", not fautes, fautes)
sys.path.insert(0, RACINE)
from api_web import routeur as RT  # noqa: E402

fautes = []
for mod in ("garde_corps", "poutre"):
    out = json.loads(RT.traiter(json.dumps({"module": mod, "action": "nouveau"})))
    for k in ("resultat", "volets", "donnees", "elements", "statut_html"):
        if EN_LIGNE.search(out[k]):
            fautes.append((mod, k, EN_LIGNE.search(out[k]).group(0)))
    fautes += [(mod, "vue", v["cle"]) for v in out["vues"] if EN_LIGNE.search(v["svg"])]
chk("les fragments que Python rend n'ont ni script ni gestionnaire « on…= »", not fautes, fautes[:3])

# 7. les adresses sans « .html » (Cloudflare Pages redirige « x.html » vers « x » ; la note ouverte n'a pas à survivre
# à une redirection) : les pages générées et js/notes.js visent les mêmes adresses — deux copies, liées ici
notes_js = open(os.path.join(WEB, "js", "notes.js"), encoding="utf-8").read()
pages_js = dict(re.findall(r'(\w+): \{ page: "([^"]+)"', notes_js))
chk("js/notes.js vise les adresses du générateur (sans « .html »)", pages_js == SW.ADRESSES, (pages_js, SW.ADRESSES))
liens = [(p, h) for p in ("index.html", "garde-corps.html", "poutre.html")
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

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
