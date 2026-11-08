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

# 5 bis (avant). l'établi COMPACT (pilote : construire.ETABLI_COMPACT, « Poutres béton armé ») : ses pages portent la
# liste, les propriétés, le résultat et la fenêtre « Enregistrer comme note » (les identifiants que js/compact.js et
# js/rangement.js cherchent) ; les AUTRES modules gardent l'établi d'avant, sans rien de ce pilote. À l'accueil, la tuile
# d'un module compact ouvre le calcul lui-même (lien direct) ; les autres ouvrent la fenêtre « Nouvelle note ».
_src_compact = open(os.path.join(WEB, "js", "compact.js"), encoding="utf-8").read()
_src_rangement = open(os.path.join(WEB, "js", "rangement.js"), encoding="utf-8").read()
# js/rangement.js cherche ses éléments par un raccourci $("et-…") = document.getElementById
_ids_compact = sorted(set(re.findall(r'getElementById\("([\w-]+)"', _src_compact + _src_rangement)) | set(re.findall(r'\$\("(et-[\w-]+)"\)', _src_rangement)))
chk("le pilote compact est la seule poutre (les autres modules attendent l'avis du bureau)", SW.ETABLI_COMPACT == ("poutre",), SW.ETABLI_COMPACT)
for k in SW.PAGES:
    html = open(os.path.join(WEB, SW.PAGES[k]), encoding="utf-8").read()
    if k in SW.ETABLI_COMPACT:
        absents = [i for i in _ids_compact if f'id="{i}"' not in html]
        chk(f"{SW.PAGES[k]} (établi compact) : la liste, les propriétés, le résultat, « Enregistrer comme note » ; "
            f"les {len(_ids_compact)} identifiants cherchés existent, une seule fois",
            not absents and 'data-etabli="compact"' in html and 'href="css/etabli.css"' in html
            and all(html.count(f'id="{i}"') == 1 for i in _ids_compact) and html.count('data-ranger="1"') == 2
            # les deux colonnes se replient (boutons communs .pcoll) et s'élargissent (poignées), l'état s'annule
            and all(html.count(f'data-{a}="{c}"') == 1 for a in ("coll", "poignee") for c in "gd")
            and all(html.count(f'data-historique="{h}"') == 1 for h in ("annuler", "retablir")), (absents, _ids_compact))
    else:
        chk(f"{SW.PAGES[k]} : l'établi d'avant, rien du pilote compact",
            "data-etabli" not in html and "etabli.css" not in html and "data-ranger" not in html and 'id="liste"' not in html)
_tuiles = {k: a for a, k in re.findall(r'<a class="tile" href="[^"]+" data-(module|nouvelle-module)="(\w+)"',
                                       open(os.path.join(WEB, "index.html"), encoding="utf-8").read())}
chk("accueil : la tuile d'un module compact ouvre le calcul (lien direct), les autres « Nouvelle note »",
    _tuiles and all(a == ("module" if k in SW.ETABLI_COMPACT else "nouvelle-module") for k, a in _tuiles.items())
    and set(_tuiles) == set(SW.PAGES)
    and ".tile[data-nouvelle-module]" in open(os.path.join(WEB, "js", "accueil.js"), encoding="utf-8").read(), _tuiles)

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
# l'enregistrement de la note : le MÊME indicateur (construire.enregistrement) dans la barre du haut (PC, tablette) et
# sur la 2ᵉ ligne de l'en-tête du téléphone, et le message d'échec qui offre « Exporter .stab », dans chaque établi
IND, ALERTE, REVOIR = SW.enregistrement(), SW.alerte_enregistrement(), SW.menu_enregistrement()
fautes = []
for k in SW.PAGES:
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
chk("notes.js : une mémoire absente s'écrit sans copie ; une mémoire lue garde ses 30 notes et n'est pas copiée",
    (JN.get("absente") or {}).get("etat") == "absente" and JN["absente"].get("ok") is True and JN["absente"].get("cles") == ["stab.notes.v1"]
    and JN.get("lue") == {"etat": "lue", "ok": True, "notes": 31, "copies": 0}, (JN.get("absente"), JN.get("lue")))
chk("notes.js : stockage plein → ecrire rend false, raison « plein », la mémoire intacte ; un .stab importé qui ne s'enregistre pas ne s'ouvre "
    "pas (l'établi l'aurait remplacé par une note neuve, sans un mot)",
    (JN.get("plein") or {}).get("ok") is False and (JN["plein"].get("b") or {}).get("raison") == "plein" and JN["plein"].get("intacte") is True
    and "pas pu être enregistrée" in str(JN.get("importer")) and "plein" in str(JN.get("importer")), (JN.get("plein"), JN.get("importer")))
NOTES_JS = open(os.path.join(WEB, "js", "notes.js"), encoding="utf-8").read()
chk("notes.js : toute écriture de la mémoire passe par changer() (une seule porte : ecrire et supprimer)",
    len(re.findall(r"localStorage\.setItem\(CLE", NOTES_JS)) == 1 and "return changer(" in NOTES_JS
    and len(re.findall(r"return changer\(", NOTES_JS)) == 2, re.findall(r"localStorage\.setItem\([^)]*\)", NOTES_JS))

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
