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
- un nom de fichier venu du moteur est nettoyé avant le téléchargement (js/commun.js, joué par node).

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
chk(f"le manifeste nomme les modules qui chargent numpy, comme le relevé ({', '.join(RELEVE['avec_numpy'])})",
    man.get("avec_numpy") == RELEVE["avec_numpy"] and set(RELEVE["avec_numpy"]) < set(SW.ADRESSES), (man.get("avec_numpy"), RELEVE["avec_numpy"]))
TRAV = open(os.path.join(WEB, "js", "travailleur.js"), encoding="utf-8").read()
chk("le travailleur lit « avec_numpy » et guette un import de numpy imprévu (la requête est alors rejouée)",
    "man.avec_numpy" in TRAV and "GuetNumpy" in TRAV and "sys.meta_path.insert(0, GuetNumpy())" in TRAV and "guet.demande" in TRAV)

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
print(json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", essai], capture_output=True, text=True, cwd=tmp)
    try:
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        res = {"_": (False, p.stderr[-400:], None)}
chk("déballé seul, le paquet calcule chaque module", res and all(v[0] for v in res.values()), res)
chk(f"chaque module du registre est dans le paquet ({len(SW.ADRESSES)})", sorted(res) == sorted(SW.ADRESSES), (list(res), list(SW.ADRESSES)))

# 3. les pages et la charte
chk("css/charte.css est la charte des maquettes", sha(os.path.join(WEB, "css", "charte.css")) == sha(os.path.join(RACINE, "site", "maquettes", "charte.css")))
for nom, html in [("index.html", SW.page_accueil()), ("js/modules.js", SW.modules_js())] + [(SW.PAGES[k], SW.page_module(k, SW._module(k)[1])) for k in SW.PAGES]:
    chk(f"{nom} est celle que le générateur écrit", os.path.exists(os.path.join(WEB, nom)) and open(os.path.join(WEB, nom), encoding="utf-8").read() == html)

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
for js, page in [(js, SW.PAGES[k]) for k in SW.PAGES for js in ("etabli.js", "toile.js")] + [("accueil.js", "index.html")]:
    src = open(os.path.join(WEB, "js", js), encoding="utf-8").read()
    html = open(os.path.join(WEB, page), encoding="utf-8").read()
    ids = sorted(set(re.findall(r'\$\("#([\w-]+)', src)) | set(re.findall(r'getElementById\("([\w-]+)"', src)))
    absents = [i for i in ids if f'id="{i}"' not in html]
    chk(f"{js} → {page} : les {len(ids)} identifiants cherchés existent", not absents, absents)

# 5 bis. les trois formats : ce que le téléphone et la tablette trouvent dans chaque page (la disposition est dans
# css/site.css et se vérifie dans le vrai navigateur : verifier_site.js, filtre « formats »)
for k in SW.PAGES:
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
pages_js = {k: v["page"] for k, v in json.loads(modules_js[modules_js.index("{"):modules_js.rindex("}") + 1]).items()}
chk("js/modules.js vise les adresses du générateur (sans « .html »), et notes.js le lit",
    pages_js == SW.ADRESSES and 'from "./modules.js"' in open(os.path.join(WEB, "js", "notes.js"), encoding="utf-8").read(), (pages_js, SW.ADRESSES))
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

# 9. chaque module du registre a sa place dans les familles de l'accueil (nom, icône) : sinon il serait écarté du site
from api_web import registre as REG2  # noqa: E402
hors = [k for k in REG2.cles() if k not in SW.ADRESSES]
chk("chaque module du registre est une clé des familles de l'accueil", not hors, hors)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
