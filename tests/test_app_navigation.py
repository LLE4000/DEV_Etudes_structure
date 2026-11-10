# -*- coding: utf-8 -*-
"""L'APPLICATION et les ÉCRANS MONTABLES (lot S4 : docs/site/REFONTE_ETABLI.md, révision du 10/11/2026 au soir, S4-D1 à
S4-D7, C1 à C12 ; contrat de réalisation § 3). Essais node des fonctions PURES de js/app.js et des écrans importés sans
document ; contrôles statiques du code des écrans. Ce que seul le vrai navigateur dit (aucune navigation de document, un
seul travailleur, la fermeture, les feuilles actives, les écouteurs vivants…) est le filtre « demarrage » du harnais
(site/web/verifier_demarrage.js).

- route(url) : chaque page du registre (js/modules.js) mène à son module, chaque bibliothèque à la sienne, « predalle »
  est une REDIRECTION vers « dalle », l'accueil a deux adresses, le reste n'est pas une route ;
- lienInterceptable(ev, a) : seul un clic ordinaire sur un lien du site vers une route connue est pris — pas un clic
  déjà pris (le ✕ « Supprimer » d'une carte Reprendre, une entrée « bientôt »), ni un bouton autre que le principal, ni
  une touche, ni un target, ni un download, ni blob:/data:/mailto:, ni une autre origine, ni un simple changement d'ancre ;
- operationsFeuilles(actuelles, cibles) : les feuilles de la cible, dans SON ordre, à leur place, les autres désactivées
  — sur des cas écrits, un tirage, et chaque paire de pages réelles du site construit ;
- etabli.js, accueil.js, bibliotheque.js, app.js s'importent SANS document, sans lever, sans y toucher, sans travailleur
  (aucun effet au chargement : les écrans sont montés par l'application) ;
- le chemin Safari du démarrage en fond : sans requestIdleCallback, un délai de 1,5 s ; sans navigator.connection, le
  moteur ne démarre en fond que si ce navigateur l'a déjà démarré (drapeau stab.moteur.vu = la version de Pyodide) ;
- la météo annulée (l'écran qui se ferme) : ni le message du délai, ni un appel de plus ;
- un message dit pendant la fermeture d'un écran est retenu, puis rendu à l'écran suivant (commun.retenirToasts) ;
- contrôles statiques : history., location.(href|search|…) =, location.assign/replace(, document.location,
  document.title = n'apparaissent dans js/ que dans app.js (location.reload() seulement pour « Recharger », dans
  commun.js) ; cablerRail n'est appelé que par app.js ; chaque écouteur posé sur document, window ou une requête média
  par un écran porte le signal du montage ; chaque ResizeObserver se déconnecte au signal ; plus de voile #demarrage
  nommé dans js/ ni dans les feuilles.
- chaque contrôle a sa MUTATION : la même vérification, sur une copie de js/ abîmée exprès, doit rougir.

    python3 tests/test_app_navigation.py
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(RACINE, "site", "web")
JS = os.path.join(WEB, "js")
CSS = os.path.join(WEB, "css")

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def node(script, *args, timeout=60):
    """Le JSON de la dernière ligne de sortie d'un script node (module), ou {"erreur": …}."""
    try:
        p = subprocess.run(["node", "--input-type=module", "-e", script, "--", *args], capture_output=True, text=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return {"erreur": repr(e)}
    lignes = [x for x in p.stdout.strip().split("\n") if x.strip()]
    try:
        return json.loads(lignes[-1])
    except Exception:  # noqa: BLE001
        return {"erreur": (p.stderr or p.stdout)[-800:]}


def copie_mutee(mutations):
    """Une copie de js/ (et de rien d'autre) où chaque (fichier, avant, après) est appliqué UNE fois ; rend le dossier."""
    d = tempfile.mkdtemp(prefix="stab_app_")
    cible = os.path.join(d, "js")
    shutil.copytree(JS, cible)
    for f, avant, apres in mutations:
        p = os.path.join(cible, f)
        s = open(p, encoding="utf-8").read()
        assert s.count(avant) == 1, (f, avant)
        open(p, "w", encoding="utf-8").write(s.replace(avant, apres))
    return cible


def url_js(d):
    return "file://" + d.rstrip("/") + "/"


MUTATIONS = []   # (nom, fonction qui rend vrai si le contrôle a bien rougi)

# ------------------------------------------------------------------ 1. route(url)
ESSAI_ROUTE = r"""
globalThis.self = { location: { href: "http://x.test/" } };
const base = process.argv[1];
const app = await import(base + "app.js");
const mods = (await import(base + "modules.js"));
const R = "http://x.test/";
const cas = {};
for (const u of ["./", "", "index.html", "index", "poutre", "poutre.html", "poutre?note=n1", "predalle?note=n1#x", "bibliotheque-beton",
                 "bibliotheque-acier?rubrique=profils", "garde-corps", "inconnu", "js/app.js", "moteur/manifeste.json",
                 "http://autre.test/poutre", "sous/poutre"]) cas[u] = app.route(u, R);
const pages = {};
for (const [k, m] of Object.entries(mods.MODULES)) pages[k] = app.route(m.page, R);
const biblios = {};
for (const [k, b] of Object.entries(mods.BIBLIOTHEQUES)) biblios[k] = app.route(b.page, R);
console.log(JSON.stringify({ cas, pages, biblios, modules: mods.MODULES }));
"""


def juger_route(d):
    r = node(ESSAI_ROUTE, url_js(d))
    if "erreur" in r:
        return False, r
    c, mods = r["cas"], r["modules"]
    fautes = []
    for u in ("./", "", "index.html", "index"):
        if c[u]["sorte"] != "accueil":
            fautes.append(u)
    if c["poutre"] != {"sorte": "module", "cle": "poutre", "page": "poutre", "vers": None}:
        fautes.append("poutre")
    if c["poutre.html"]["cle"] != "poutre" or c["poutre?note=n1"]["cle"] != "poutre":
        fautes.append("poutre.html / ?note")
    if c["predalle?note=n1#x"] != {"sorte": "redirection", "cle": "predalle", "page": "predalle", "vers": "dalle"}:
        fautes.append(("predalle", c["predalle?note=n1#x"]))
    if c["bibliotheque-beton"] != {"sorte": "bibliotheque", "cle": "beton", "page": "bibliotheque-beton", "vers": None}:
        fautes.append("bibliotheque-beton")
    if c["bibliotheque-acier?rubrique=profils"]["cle"] != "acier" or c["garde-corps"]["cle"] != "garde_corps":
        fautes.append("acier / garde-corps")
    for u in ("inconnu", "js/app.js", "moteur/manifeste.json", "http://autre.test/poutre", "sous/poutre"):
        if c[u]["sorte"] is not None:
            fautes.append(u)
    for k, m in mods.items():
        attendu = ("redirection" if m.get("vers") else "module")
        if r["pages"][k]["sorte"] != attendu or r["pages"][k]["cle"] != k or (m.get("vers") and r["pages"][k]["vers"] != mods[m["vers"]]["page"]):
            fautes.append(("page", k, r["pages"][k]))
    for k, x in r["biblios"].items():
        if x["sorte"] != "bibliotheque" or x["cle"] != k:
            fautes.append(("biblio", k))
    return not fautes and len(r["pages"]) == len(mods) >= 18, fautes or len(r["pages"])


ok, info = juger_route(JS)
chk(f"route : les {info if ok else '?'} pages du registre mènent à leur module (predalle : une redirection vers dalle), les deux "
    "bibliothèques à la leur, l'accueil à « ./ », « index », « index.html » ; une autre origine, un sous-dossier, une adresse "
    "inconnue ne sont pas des routes", ok, info)
MUTATIONS.append(("route : la redirection oubliée (predalle traité comme un établi)", lambda: not juger_route(copie_mutee([(
    "app.js", 'PAGES.set(m.page, m.vers ? { sorte: "redirection", cle: k, vers: MODULES[m.vers].page } : { sorte: "module", cle: k, vers: null });',
    'PAGES.set(m.page, { sorte: "module", cle: k, vers: null });')]))[0]))
MUTATIONS.append(("route : une autre origine acceptée", lambda: not juger_route(copie_mutee([(
    "app.js", "  if (u.origin !== r.origin || !u.pathname.startsWith(r.pathname)) return rien;", "  if (!u.pathname.startsWith(r.pathname)) return rien;")]))[0]))

# ------------------------------------------------------------------ 1 bis. adresseCanonique(url)
# (intégration du lot S4, le harnais a TROUVÉ : en PASSAGE=1, un lien vers « poutre.html » rechargeait le document — la
# lecture rendait la redirection 308 de Cloudflare (« .html » → sans), prise pour une session à refaire. L'application
# va désormais tout droit à l'adresse canonique, celle que Cloudflare sert.)
ESSAI_CANON = r"""
globalThis.self = { location: { href: "http://x.test/" } };
const app = await import(process.argv[1] + "app.js");
const R = "http://x.test/";
const cas = {};
for (const u of ["poutre.html", "poutre.html?note=n1#s2", "index.html", "index.html?x=1", "poutre", "poutre?note=n1", "./", "",
                 "js/app.js", "moteur/manifeste.json", "sous/page.html", "http://autre.test/poutre.html", "dalle.htm"]) cas[u] = app.adresseCanonique(u, R);
console.log(JSON.stringify(cas));
"""
CANON = {"poutre.html": "http://x.test/poutre", "poutre.html?note=n1#s2": "http://x.test/poutre?note=n1#s2",
         "index.html": "http://x.test/", "index.html?x=1": "http://x.test/?x=1", "poutre": "http://x.test/poutre",
         "poutre?note=n1": "http://x.test/poutre?note=n1", "./": "http://x.test/", "": "http://x.test/",
         "js/app.js": "http://x.test/js/app.js", "moteur/manifeste.json": "http://x.test/moteur/manifeste.json",
         "sous/page.html": "http://x.test/sous/page.html", "http://autre.test/poutre.html": "http://autre.test/poutre.html",
         "dalle.htm": "http://x.test/dalle.htm"}


def juger_canon(d):
    r = node(ESSAI_CANON, url_js(d))
    if "erreur" in r:
        return False, r
    fautes = {k: (r.get(k), v) for k, v in CANON.items() if r.get(k) != v}
    return not fautes, fautes


ok, info = juger_canon(JS)
chk("adresseCanonique : « poutre.html » → « poutre » (requête et ancre gardées), « index.html » → la racine ; une adresse "
    "déjà canonique, un fichier d'un sous-dossier, une autre origine restent tels quels", ok, info)
MUTATIONS.append(("adresseCanonique : « .html » gardé (le lien repart en navigation ordinaire)", lambda: not juger_canon(copie_mutee([(
    "app.js", '  u.pathname = r.pathname + (seg === "index.html" ? "" : seg.slice(0, -5));\n', "")]))[0]))
MUTATIONS.append(("adresseCanonique : un sous-dossier canonisé aussi", lambda: not juger_canon(copie_mutee([(
    "app.js", '  if (seg.includes("/") || !seg.endsWith(".html")) return u.href;', '  if (!seg.endsWith(".html")) return u.href;')]))[0]))
# (et naviguer, prechauffer passent par elle : sans quoi la fonction existerait sans servir)
_app = open(os.path.join(JS, "app.js"), encoding="utf-8").read()
chk("naviguer et prechauffer lisent l'adresse canonique (adresseCanonique) avant toute route ou lecture",
    _app.count("try { u = new URL(adresseCanonique(new URL(url, location.href).href)); } catch (e) { return; }") == 2)

# ------------------------------------------------------------------ 2. lienInterceptable(ev, a)
ESSAI_LIEN = r"""
globalThis.self = { location: { href: "http://x.test/" } };
const app = await import(process.argv[1] + "app.js");
const R = "http://x.test/", ICI = "http://x.test/";
// un faux lien : ses attributs ; ``dans`` : les sélecteurs de ses ancêtres (closest)
const lien = (attrs, dans = []) => ({
  getAttribute: (n) => (n in attrs ? attrs[n] : null), hasAttribute: (n) => n in attrs,
  closest: (s) => (s.split(",").some((x) => dans.includes(x.trim())) ? {} : null) });
const cible = (dans = []) => ({ closest: (s) => (s.split(",").some((x) => dans.includes(x.trim())) ? {} : null) });
const ev = (o = {}) => ({ defaultPrevented: false, button: 0, ctrlKey: false, metaKey: false, shiftKey: false, altKey: false, target: cible(), ...o });
const L = (e, a, ici = ICI) => app.lienInterceptable(e, a, ici, R);
const res = {
  ordinaire: L(ev(), lien({ href: "poutre" })),
  avecNote: L(ev(), lien({ href: "poutre?note=n1" })),
  absolu: L(ev(), lien({ href: "http://x.test/dalle" })),
  self: L(ev(), lien({ href: "poutre", target: "_self" })),
  accueil: L(ev(), lien({ href: "./" }), "http://x.test/poutre"),
  ancreAilleurs: L(ev(), lien({ href: "poutre#x" })),
  predalle: L(ev(), lien({ href: "predalle?note=n1" })),
  deja: L(ev({ defaultPrevented: true }), lien({ href: "poutre" })),
  milieu: L(ev({ button: 1 }), lien({ href: "poutre" })),
  droit: L(ev({ button: 2 }), lien({ href: "poutre" })),
  ctrl: L(ev({ ctrlKey: true }), lien({ href: "poutre" })),
  meta: L(ev({ metaKey: true }), lien({ href: "poutre" })),
  shift: L(ev({ shiftKey: true }), lien({ href: "poutre" })),
  alt: L(ev({ altKey: true }), lien({ href: "poutre" })),
  blank: L(ev(), lien({ href: "poutre", target: "_blank" })),
  download: L(ev(), lien({ href: "poutre", download: "" })),
  supprimer: L(ev({ target: cible(["[data-supprimer]"]) }), lien({ href: "poutre?note=n1" })),
  bientot: L(ev(), lien({ href: "poutre", "data-bientot": "1" }, ["[data-bientot]"])),
  blob: L(ev(), lien({ href: "blob:http://x.test/1b2c-3d" })),
  data: L(ev(), lien({ href: "data:text/plain,abc" })),
  mailto: L(ev(), lien({ href: "mailto:bureau@x.test" })),
  autre: L(ev(), lien({ href: "http://autre.test/poutre" })),
  inconnue: L(ev(), lien({ href: "inconnu" })),
  ancre: L(ev(), lien({ href: "#haut" })),
  ancreMeme: L(ev(), lien({ href: "poutre?note=n1#x" }), "http://x.test/poutre?note=n1"),
  sansHref: L(ev(), lien({})),
};
console.log(JSON.stringify(res));
"""
REFUS_LIEN = ("deja", "milieu", "droit", "ctrl", "meta", "shift", "alt", "blank", "download", "supprimer", "bientot", "blob", "data",
              "mailto", "autre", "inconnue", "ancre", "ancreMeme", "sansHref")


def juger_lien(d):
    r = node(ESSAI_LIEN, url_js(d))
    if "erreur" in r:
        return False, r
    pris = {"ordinaire": "http://x.test/poutre", "avecNote": "http://x.test/poutre?note=n1", "absolu": "http://x.test/dalle",
            "self": "http://x.test/poutre", "accueil": "http://x.test/", "ancreAilleurs": "http://x.test/poutre#x",
            "predalle": "http://x.test/predalle?note=n1"}
    fautes = [k for k, v in pris.items() if r.get(k) != v] + [k for k in REFUS_LIEN if r.get(k) is not None]
    return not fautes, (fautes, r)


ok, info = juger_lien(JS)
chk(f"lienInterceptable : 7 clics ordinaires pris (route connue, même origine, ancre vers une autre page comprise) ; {len(REFUS_LIEN)} "
    "laissés au navigateur (déjà pris, bouton du milieu ou droit, ctrl, meta, shift, alt, target=_blank, download, ✕ « Supprimer » "
    "d'une carte, « bientôt », blob:, data:, mailto:, autre origine, route inconnue, ancre seule, sans href)", ok, info)
for nom, avant, apres in [
    ("le clic déjà pris (defaultPrevented) intercepté quand même", "  if (!ev || ev.defaultPrevented) return null;", "  if (!ev) return null;"),
    ("le ✕ « Supprimer » d'une carte ouvrirait la note", 'const pris = "[data-supprimer], [data-bientot]";', 'const pris = "[data-rien]";'),
    ("les touches (ctrl, meta…) oubliées", " || ev.ctrlKey || ev.metaKey || ev.shiftKey || ev.altKey", ""),
    ("le download oublié", '  if (a.hasAttribute("download")) return null;\n', ""),
    ("le simple changement d'ancre intercepté", "  if (u.hash && u.pathname === i.pathname && u.search === i.search) return null;\n", ""),
]:
    MUTATIONS.append((f"lienInterceptable : {nom}", (lambda a=avant, b=apres: not juger_lien(copie_mutee([("app.js", a, b)]))[0])))

# ------------------------------------------------------------------ 3. operationsFeuilles(actuelles, cibles)
_LIENS = {}
for f in sorted(os.listdir(WEB)):
    if f.endswith(".html"):
        _LIENS[f] = re.findall(r'<link rel="stylesheet" href="([^"]+)"', open(os.path.join(WEB, f), encoding="utf-8").read())
ESSAI_FEUILLES = r"""
globalThis.self = { location: { href: "http://x.test/" } };
const app = await import(process.argv[1] + "app.js");
const entree = JSON.parse(process.argv[2]);
const sortie = entree.map(([a, c]) => app.operationsFeuilles(a, c));
console.log(JSON.stringify(sortie));
"""


def cas_feuilles():
    c, s, e, b, m = "charte", "site", "etabli", "bibliotheque", "modules/dalle"
    ecrits = [([c, s], [c, s, e, b]), ([c, s, e, b], [c, s, b]), ([c, s, e, m], [c, s, e, b]), (["b", "a"], ["a", "b"]),
              (["x"], ["a", "x"]), ([], [c, s]), ([c, s, e, b, m], [c, s, e, m]), ([m, c, s], [c, s, e, m])]
    alea = random.Random(20261110)
    tires = []
    for _ in range(200):
        univers = [f"f{i}" for i in range(7)]
        alea.shuffle(univers)
        actuelles = alea.sample(univers, alea.randint(0, 6))
        cibles = alea.sample(univers, alea.randint(0, 6))
        tires.append((actuelles, cibles))
    # chaque paire de pages du site construit : la feuille de l'une, puis l'autre (l'ordre des opérations successives
    # n'est pas rejoué : chaque paire part des feuilles de sa première page)
    reels = [(_LIENS[a], _LIENS[z]) for a in _LIENS for z in _LIENS]
    return ecrits, tires, reels


def juger_feuilles(d):
    ecrits, tires, reels = cas_feuilles()
    tout = ecrits + tires + reels
    r = node(ESSAI_FEUILLES, url_js(d), json.dumps(tout))
    if isinstance(r, dict):
        return False, r
    fautes = []
    for (actuelles, cibles), o in zip(tout, r):
        ordre = o["ordre"]
        actives = [h for h in ordre if h in cibles]
        if actives != cibles or set(actuelles) - set(ordre) or o["activer"] != cibles or o["desactiver"] != [h for h in ordre if h not in cibles]:
            fautes.append((actuelles, cibles, o))
        # rejouer les opérations sur la liste d'origine redonne l'ordre annoncé
        lst = list(actuelles)
        for op in o["operations"]:
            if op["op"] == "deplacer":
                lst.remove(op["href"])
            pos = lst.index(op["apres"]) + 1 if op["apres"] is not None else 0
            lst.insert(pos, op["href"])
        if lst != ordre:
            fautes.append(("rejeu", actuelles, cibles, lst, ordre))
    ajout = r[0]["operations"]
    attendu0 = [{"op": "ajouter", "href": "etabli", "apres": "site"}, {"op": "ajouter", "href": "bibliotheque", "apres": "etabli"}]
    if ajout != attendu0 or r[2]["ordre"] != ["charte", "site", "etabli", "bibliotheque", "modules/dalle"] or r[3]["operations"] != [{"op": "deplacer", "href": "b", "apres": "a"}]:
        fautes.append(("cas écrits", ajout, r[2]["ordre"], r[3]["operations"]))
    return not fautes, (len(ecrits), len(tires), len(reels), fautes[:3])


ok, info = juger_feuilles(JS)
chk(f"operationsFeuilles : les feuilles ACTIVES sont celles de la cible, dans SON ordre (une nouvelle à sa place, une mal placée "
    f"déplacée), les autres désactivées et gardées — {info[0] if ok else '?'} cas écrits, {info[1] if ok else '?'} tirés, et les "
    f"{info[2] if ok else '?'} paires de pages du site construit ; le rejeu des opérations redonne l'ordre annoncé", ok, info)
MUTATIONS.append(("operationsFeuilles : une nouvelle feuille ajoutée en FIN (ordre de la cascade faux)", lambda: not juger_feuilles(copie_mutee([(
    "app.js", "      ordre.splice(j + 1, 0, h);\n      operations.push({ op: \"ajouter\", href: h, apres: prec });",
    "      ordre.push(h);\n      operations.push({ op: \"ajouter\", href: h, apres: ordre[ordre.length - 2] ?? null });")]))[0]))
MUTATIONS.append(("operationsFeuilles : une feuille mal placée n'est pas déplacée", lambda: not juger_feuilles(copie_mutee([(
    "app.js", "    } else if (i < j) {", "    } else if (false) {")]))[0]))

# ------------------------------------------------------------------ 4. les écrans s'importent sans document, sans effet
ESSAI_IMPORT = r"""
globalThis.self = { location: { href: "http://x.test/poutre?sonde=0" } };
const base = process.argv[1];
const touches = [];
let travailleurs = 0;
Object.defineProperty(globalThis, "document", { configurable: true, get() { touches.push("document"); return undefined; } });
for (const n of ["matchMedia", "requestAnimationFrame", "localStorage", "sessionStorage", "history"]) {
  Object.defineProperty(globalThis, n, { configurable: true, get() { touches.push(n); return undefined; } });
}
globalThis.Worker = class { constructor() { travailleurs++; } };
const res = {};
for (const [f, e] of [["etabli.js", "ouvrirEtabli"], ["accueil.js", "ouvrirAccueil"], ["bibliotheque.js", "ouvrirBibliotheque"]]) {
  try { const m = await import(base + f); res[f] = { ok: typeof m[e] === "function" && m[e].length === 2, exports: Object.keys(m) }; }
  catch (x) { res[f] = { ok: false, erreur: String(x && x.message || x) }; }
}
const avantApp = touches.slice();
delete globalThis.document;   // app.js regarde s'il y a un document (et n'y touche pas s'il n'y en a pas)
try { const m = await import(base + "app.js"); res["app.js"] = { ok: ["naviguer", "prechauffer", "route", "lienInterceptable", "operationsFeuilles"].every((k) => typeof m[k] === "function") }; }
catch (x) { res["app.js"] = { ok: false, erreur: String(x && x.message || x) }; }
console.log(JSON.stringify({ res, touches: avantApp, touchesApp: touches.slice(avantApp.length), travailleurs }));
"""


def juger_import(d):
    r = node(ESSAI_IMPORT, url_js(d))
    if "erreur" in r:
        return False, r
    ok = all(v["ok"] for v in r["res"].values()) and not r["touches"] and not r["touchesApp"] and r["travailleurs"] == 0 and len(r["res"]) == 4
    return ok, r


ok, info = juger_import(JS)
chk("etabli.js, accueil.js, bibliotheque.js s'importent dans node SANS document : ni erreur, ni lecture de document, matchMedia, "
    "requestAnimationFrame, localStorage, sessionStorage ou history, ni travailleur ; chacun exporte son ouvrirX(hote, ctx) — et "
    "app.js s'importe sans rien démarrer", ok, info)
MUTATIONS.append(("import : l'accueil câble le rail à son chargement (l'effet d'avant)", lambda: not juger_import(copie_mutee([(
    "accueil.js", 'const $ = (s) => document.querySelector(s);\n', 'const $ = (s) => document.querySelector(s);\ndocument.addEventListener("click", () => {});\n')]))[0]))
MUTATIONS.append(("import : l'établi lit le corps à son chargement", lambda: not juger_import(copie_mutee([(
    "etabli.js", "const HIST_MAX = 50;\n", "const HIST_MAX = 50;\nconst MODULE_DU_CORPS = document.body.dataset.module;\n")]))[0]))
MUTATIONS.append(("import : app.js démarre un moteur à son chargement", lambda: not juger_import(copie_mutee([(
    "app.js", "const nouvelleCle = () => `${DOC}.${++nCles}`;\n", "const nouvelleCle = () => `${DOC}.${++nCles}`;\nmoteurPartage();\n")]))[0]))

# ------------------------------------------------------------------ 5. le chemin Safari du démarrage en fond (C6)
ESSAI_SAFARI = r"""
globalThis.self = { location: { href: "http://x.test/" } };
const app = await import(process.argv[1] + "app.js");
const appels = [];
const sansIdle = { setTimeout: (f, ms) => { appels.push(["setTimeout", ms]); return 1; } };
const avecIdle = { setTimeout: (f, ms) => { appels.push(["setTimeout", ms]); return 1; }, requestIdleCallback: (f, o) => { appels.push(["idle", o && o.timeout]); return 2; } };
app.planifierFond(() => {}, sansIdle);
const a1 = appels.splice(0);
app.planifierFond(() => {}, avecIdle);
const a2 = appels.splice(0);
// le vrai globalThis de node : ni requestIdleCallback ni navigator.connection (comme Safari)
const st = globalThis.setTimeout;
globalThis.setTimeout = (f, ms) => { appels.push(["setTimeout global", ms]); return 0; };
app.planifierFond(() => {});
globalThis.setTimeout = st;
const a3 = appels.splice(0);
const P = app.demarrageEnFondPermis, V = app.versionPyodide({ pyodide: "vendor/pyodide-0.29.5" });
const cas = {
  cache: P({ visible: false, connexion: { effectiveType: "4g" } }),
  quatreG: P({ visible: true, connexion: { effectiveType: "4g", saveData: false } }),
  economie: P({ visible: true, connexion: { effectiveType: "4g", saveData: true } }),
  troisG: P({ visible: true, connexion: { effectiveType: "3g" } }),
  deuxG: P({ visible: true, connexion: { effectiveType: "2g" } }),
  lent: P({ visible: true, connexion: { effectiveType: "slow-2g" } }),
  safariVu: P({ visible: true, connexion: null, vu: V, pyodide: V }),
  safariJamais: P({ visible: true, connexion: null, vu: null, pyodide: V }),
  safariAutre: P({ visible: true, connexion: null, vu: "0.28.0", pyodide: V }),
  safariCache: P({ visible: false, connexion: null, vu: V, pyodide: V }),
  node: { connexion: typeof navigator === "undefined" ? "sans navigator" : String(navigator.connection), idle: typeof requestIdleCallback },
};
console.log(JSON.stringify({ a1, a2, a3, cas, V, VU: app.VU }));
"""


def juger_safari(d):
    r = node(ESSAI_SAFARI, url_js(d))
    if "erreur" in r:
        return False, r
    c = r["cas"]
    ok = (r["a1"] == [["setTimeout", 1500]] and r["a2"] == [["idle", 2000]] and r["a3"] == [["setTimeout global", 1500]]
          and c["quatreG"] is True and c["safariVu"] is True and not any(c[k] for k in ("cache", "economie", "troisG", "deuxG", "lent",
                                                                                         "safariJamais", "safariAutre", "safariCache"))
          and r["V"] == "0.29.5" and r["VU"] == "stab.moteur.vu" and c["node"]["connexion"] == "undefined" and c["node"]["idle"] == "undefined")
    return ok, r


ok, info = juger_safari(JS)
chk("démarrage en fond, chemin Safari : sans requestIdleCallback, setTimeout de 1,5 s (dans node aussi, qui n'a ni "
    "requestIdleCallback ni navigator.connection) ; avec, requestIdleCallback(…, 2 s) ; permis seulement page visible, sans "
    "économie de données ni 2g/3g ; sans navigator.connection, seulement si stab.moteur.vu vaut la version de Pyodide (0.29.5)", ok, info)
MUTATIONS.append(("Safari : plus de repli setTimeout quand requestIdleCallback manque", lambda: not juger_safari(copie_mutee([(
    "app.js", "  return g.setTimeout(f, 1500);", "  return null;")]))[0]))
MUTATIONS.append(("Safari : le moteur démarre en fond sans savoir s'il est en cache", lambda: not juger_safari(copie_mutee([(
    "app.js", "  return !!pyodide && vu === pyodide;", "  return true;")]))[0]))
MUTATIONS.append(("le 3g n'arrête plus le démarrage en fond", lambda: not juger_safari(copie_mutee([(
    "app.js", "/^(slow-2g|2g|3g)$/", "/^(slow-2g|2g)$/")]))[0]))

# ------------------------------------------------------------------ 6. la météo annulée (C1 : l'écran se ferme)
ESSAI_METEO = r"""
const m = await import(process.argv[1] + "meteo.js");
const plan = { lieu: "Liège", geocodage: { name: "Liège", count: 1, language: "fr" },
  appels: [{ source: "archive", parametres: { start_date: "2026-11-01", end_date: "2026-11-05", timezone: "Europe/Brussels", daily: "temperature_2m_mean" } }] };
const ctl = new AbortController();
const vus = [];
const faux = (u, init) => new Promise((ok, ko) => {
  vus.push(u);
  if (vus.length === 1) setTimeout(() => ctl.abort(), 5);   // l'écran se ferme pendant le géocodage
  init.signal.addEventListener("abort", () => { const e = new Error("abandon"); e.name = "AbortError"; ko(e); });
});
const r = await m.recupererMeteo(plan, { fetch: faux, signal: ctl.signal, delai: 20000 });
// annulé AVANT de commencer : rien ne part
const ctl2 = new AbortController(); ctl2.abort();
const vus2 = [];
const r2 = await m.recupererMeteo(plan, { fetch: (u) => { vus2.push(u); return Promise.reject(new TypeError("réseau")); }, signal: ctl2.signal });
console.log(JSON.stringify({ r, appels: vus.length, r2, appels2: vus2.length, delai: m.MESSAGES.delai }));
"""


def juger_meteo(d):
    r = node(ESSAI_METEO, url_js(d))
    if "erreur" in r:
        return False, r
    ok = (r["r"].get("ok") is False and r["r"].get("annule") is True and r["r"].get("erreur") != r["delai"] and r["appels"] == 1
          and r["r2"].get("annule") is True and r["appels2"] == 0)
    return ok, r


ok, info = juger_meteo(JS)
chk("météo annulée pendant le géocodage (le signal de l'écran qui se ferme) : { ok: false, annule: true }, jamais le message du "
    "délai, aucun appel de plus ; annulée avant de commencer : aucun appel", ok, info)
MUTATIONS.append(("météo : l'annulation redevient « le service ne répond pas (20 s) »", lambda: not juger_meteo(copie_mutee([(
    "meteo.js", "      if (o.signal && o.signal.aborted) return ANNULE;\n      return { ok: false, erreur: e && e.name === \"AbortError\" ? MESSAGES.delai : MESSAGES.reseau };",
    "      return { ok: false, erreur: e && e.name === \"AbortError\" ? MESSAGES.delai : MESSAGES.reseau };")]))[0]))

# ------------------------------------------------------------------ 7. un message dit pendant la fermeture passe à l'écran suivant
ESSAI_TOASTS = r"""
const c = await import(process.argv[1] + "commun.js");
c.retenirToasts();
let leve = null;
try { c.toast("Valeur refusée : « x »", true); c.toast("Météo récupérée"); } catch (e) { leve = String(e.message); }
const rendus = c.lacherToasts();
const encore = c.lacherToasts();
console.log(JSON.stringify({ leve, rendus, encore }));
"""


def juger_toasts(d):
    r = node(ESSAI_TOASTS, url_js(d))
    ok = "erreur" not in r and r["leve"] is None and r["rendus"] == [["Valeur refusée : « x »", True], ["Météo récupérée", False]] and r["encore"] == []
    return ok, r


ok, info = juger_toasts(JS)
chk("pendant la fermeture d'un écran, ses messages sont RETENUS (pas posés dans un corps qui va disparaître), puis rendus une fois "
    "à l'application, qui les dit dans l'écran suivant", ok, info)
MUTATIONS.append(("les messages de la fermeture ne sont plus retenus", lambda: not juger_toasts(copie_mutee([(
    "commun.js", "  if (retenus) { retenus.push([texte, !!erreur]); return; }\n", "")]))[0]))

# ------------------------------------------------------------------ 7 bis. le moteur : la modification gardée d'un écran fermé ne retient pas l'écran suivant
# (correction d'intégration de js/moteur.js, lot S4 · A3 : la requête en vol d'un client FERMÉ ne bloque plus celles des autres
# clients — l'écran suivant n'attend pas la réponse d'une saisie de l'écran quitté, C1 ; un client garde sa file une à une)
ESSAI_MOTEUR = r"""
globalThis.self = { location: { href: "http://x.test/" } };
const postes = [];
let w = null;
globalThis.Worker = class { constructor() { w = this; } postMessage(m) { postes.push(m); } set onmessage(f) { this._f = f; } get onmessage() { return this._f; } };
const M = await import(process.argv[1] + "moteur.js");
const m = new M.Moteur();
w._f({ data: { type: "pret" } });
const calme = async () => { for (let i = 0; i < 4; i++) await new Promise((ok) => setTimeout(ok, 0)); };
const repondre = (id, x) => w._f({ data: { type: "reponse", id, json: JSON.stringify(x) } });
const A = m.client(), B = m.client();
const vus = [];
const pa1 = A.demander(() => ({ module: "poutre", action: "modifier", n: 1 }), { modifie: true }).then((r) => vus.push(["A1", r.n]));
await calme();
const pa2 = A.demander(() => ({ module: "poutre", action: "modifier", n: 2 }), { modifie: true }).then((r) => vus.push(["A2", r.n]));
A.fermer();
const pb = B.demander(() => ({ module: "assemblages", action: "nouveau" })).then((r) => vus.push(["B", r.n]));
await calme();
const avant = postes.map((p) => JSON.parse(p.requete || "{}").module + ":" + (JSON.parse(p.requete || "{}").n ?? "-"));
// B répond avant A1 (la réponse de A1 tarde)
repondre(postes[1].id, { ok: true, n: "b" });
await calme();
const apresB = vus.slice();
repondre(postes[0].id, { ok: true, n: 1 });
await calme();
const postesFin = postes.map((p) => JSON.parse(p.requete || "{}").module + ":" + (JSON.parse(p.requete || "{}").n ?? "-"));
repondre(postes[2].id, { ok: true, n: 2 });
await calme();
// un client OUVERT qui a une requête en vol retient les autres (une à la fois)
const C = m.client(), D = m.client();
const n0 = postes.length;
C.demander(() => ({ module: "dalle", action: "nouveau" }));
D.demander(() => ({ module: "taux", action: "nouveau" }));
await calme();
const ouvert = postes.length - n0;
console.log(JSON.stringify({ avant, apresB, postesFin, vus, ouvert }));
"""


def juger_moteur(d):
    r = node(ESSAI_MOTEUR, url_js(d))
    if "erreur" in r:
        return False, r
    ok = (r["avant"] == ["poutre:1", "assemblages:-"] and r["apresB"] == [["B", "b"]] and r["postesFin"] == ["poutre:1", "assemblages:-", "poutre:2"]
          and r["vus"] == [["B", "b"], ["A1", 1], ["A2", 2]] and r["ouvert"] == 1)
    return ok, r


ok, info = juger_moteur(JS)
chk("moteur : la modification en vol d'un écran FERMÉ ne retient pas la requête de l'écran suivant (partie, et servie, avant sa "
    "réponse) ; la seconde modification gardée du même écran attend la première ; entre clients ouverts, une requête à la fois", ok, info)
MUTATIONS.append(("moteur : toute requête en vol retient toutes les autres (l'écran suivant attend la saisie de l'écran quitté)", lambda: not juger_moteur(copie_mutee([(
    "moteur.js", "    for (const v of this.envols.values()) if (!v.client.ferme || v.client === r.client) return true;",
    "    for (const v of this.envols.values()) return true;")]))[0]))
MUTATIONS.append(("moteur : le client fermé ne garde plus sa propre file une à une", lambda: not juger_moteur(copie_mutee([(
    "moteur.js", "    for (const v of this.envols.values()) if (!v.client.ferme || v.client === r.client) return true;",
    "    for (const v of this.envols.values()) if (!v.client.ferme) return true;")]))[0]))

# ------------------------------------------------------------------ 8. contrôles statiques du code des écrans


def code_seul(src):
    """Le code sans ses commentaires « // … » (en début de ligne ou après un blanc : « https:// » reste) ni « /* … */ »."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(re.sub(r"(^|\s)//.*$", r"\1", ligne) for ligne in src.split("\n"))


INTERDIT = re.compile(r"\bhistory\s*\.|\blocation\s*=(?!=)|\blocation\s*\.\s*(href|search|pathname|hash)\s*=(?!=)|\blocation\s*\.\s*(assign|replace)\s*\("
                      r"|\bdocument\s*\.\s*location\b|\bdocument\s*\.\s*title\s*=(?!=)")
ECRANS = ("etabli.js", "accueil.js", "bibliotheque.js", "compact.js", "toile.js", "rangement.js", "biblio.js", "biblio_note.js",
          "meteo.js", "notes.js", "commun.js", "modules.js", "moteur.js")


def juger_statique(d):
    fautes = {}
    sources = {f: code_seul(open(os.path.join(d, f), encoding="utf-8").read()) for f in sorted(os.listdir(d)) if f.endswith(".js")}
    for f, s in sources.items():
        if f == "app.js":
            continue
        trouve = [m.group(0) for m in INTERDIT.finditer(s)]
        if trouve:
            fautes[f] = trouve
    # aucune lecture de location dans les écrans (ils reçoivent ctx.params) : seul location.reload() de « Recharger »
    lectures = {f: re.findall(r"\blocation\b[^\n]{0,30}", sources[f]) for f in ECRANS if f in sources}
    rechargements = {f: len(re.findall(r"\blocation\.reload\(\)", s)) for f, s in sources.items()}
    for f, l in lectures.items():
        autres = [x for x in l if not x.startswith("location.reload()")]
        if autres:
            fautes[f"{f} (lecture)"] = autres
    if {f: n for f, n in rechargements.items() if n} != {"commun.js": 1}:
        fautes["location.reload()"] = rechargements
    # le rail est câblé UNE fois, par l'application
    rail = {f: len(re.findall(r"(?<!function )\bcablerRail\s*\(\s*\)", s)) for f, s in sources.items()}
    if {f: n for f, n in rail.items() if n} != {"app.js": 1}:
        fautes["cablerRail()"] = rail
    # theme.js (script classique, en tête) ne touche à rien de tout cela
    if "theme.js" in sources and re.search(r"\blocation\b|\bhistory\b", sources["theme.js"]):
        fautes["theme.js"] = "location/history"
    # contrôle positif : app.js est bien la porte
    a = sources.get("app.js", "")
    if not all(x in a for x in ("history.pushState(", "history.replaceState(", "document.title = ", "location.assign(", "location.replace(",
                                 'history.scrollRestoration = "manual"')):
        fautes["app.js"] = "la porte (pushState, replaceState, title, assign, replace, scrollRestoration) n'y est plus"
    return not fautes, fautes


ok, info = juger_statique(JS)
chk("history., location.(href|search|pathname|hash) =, location =, location.assign/replace(, document.location, document.title = "
    "n'apparaissent dans js/ que dans app.js ; les écrans ne lisent pas location (ctx.params) ; location.reload() une seule fois, "
    "pour « Recharger » (commun.js) ; cablerRail() appelé une seule fois, par app.js ; theme.js n'en a rien", ok, info)
for nom, f, avant, apres in [
    ("la bibliothèque réécrit l'adresse elle-même", "bibliotheque.js", '      ctx.adresse(`?${u}`);', '      history.replaceState(null, "", `?${u}`);'),
    ("l'accueil navigue par location.href", "accueil.js", "    ctx.naviguer(`${N.pageDe(module)}${ou}`);", "    location.href = `${N.pageDe(module)}${ou}`;"),
    ("l'établi écrit le titre", "etabli.js", "      ctx.titre(`${note.nom || N.MODULES[MODULE].nom} · Stab Méthodes`);",
     "      document.title = `${note.nom || N.MODULES[MODULE].nom} · Stab Méthodes`;"),
    ("l'établi relit location.search", "etabli.js", "    const q = ctx.params;", "    const q = new URLSearchParams(location.search);"),
    ("un écran recâble le rail", "accueil.js", 'import { esc, toast, texteTaux, telecharger } from "./commun.js";',
     'import { esc, toast, texteTaux, telecharger, cablerRail } from "./commun.js";\nexport const _r = () => cablerRail();'),
]:
    MUTATIONS.append((f"statique : {nom}", (lambda f=f, a=avant, b=apres: not juger_statique(copie_mutee([(f, a, b)]))[0])))


def appels(src, motif):
    """Les appels « motif(…) » de ``src``, chacun avec ses arguments (parenthèses appariées, chaînes simples sautées)."""
    out = []
    for m in re.finditer(motif, src):
        i, prof, chaine = m.end(), 1, None
        while i < len(src) and prof:
            ch = src[i]
            if chaine:
                if ch == "\\":
                    i += 1
                elif ch == chaine:
                    chaine = None
            elif ch in "\"'`":
                chaine = ch
            elif ch == "(":
                prof += 1
            elif ch == ")":
                prof -= 1
            i += 1
        out.append(src[m.start():i])
    return out


SIGNALES = ("etabli.js", "accueil.js", "bibliotheque.js", "compact.js", "toile.js", "rangement.js", "biblio.js", "biblio_note.js")


def juger_signal(d):
    fautes = {}
    nb = 0
    for f in SIGNALES:
        s = code_seul(open(os.path.join(d, f), encoding="utf-8").read())
        # les écouteurs posés sur le document, la fenêtre, une requête média (« pc », « telephone ») et… tous les autres
        for a in appels(s, r"\b(document|window|pc|telephone|[\w$.\[\]\"()]+)\.addEventListener\s*\("):
            if a.startswith("signal.addEventListener") or ".signal.addEventListener" in a.split("(")[0]:
                continue
            nb += 1
            if not re.search(r"\bsignal\b", a):
                fautes.setdefault(f, []).append(a[:90])
        # chaque ResizeObserver se déconnecte à l'abandon du signal
        for m in re.finditer(r"const (\w+) = new ResizeObserver", s):
            if not re.search(rf'signal\.addEventListener\("abort", \(\) => {m.group(1)}\.disconnect\(\)', s):
                fautes.setdefault(f, []).append(f"ResizeObserver {m.group(1)} jamais déconnecté")
        if re.search(r"new ResizeObserver\([^)]*\)\.observe", s):
            fautes.setdefault(f, []).append("ResizeObserver anonyme (jamais déconnecté)")
        if re.search(r"\bsetInterval\s*\(", s):
            fautes.setdefault(f, []).append("setInterval")
    return not fautes and nb > 60, (nb, fautes)


ok, info = juger_signal(JS)
chk(f"chaque écouteur posé par un écran ({info[0]} dans {len(SIGNALES)} fichiers : document, window, requêtes média, éléments) porte "
    "le signal du montage ; chaque ResizeObserver se déconnecte à son abandon ; aucun setInterval dans un écran (le chronomètre du "
    "bandeau, dans commun.js, s'arrête au signal)", ok, info)
MUTATIONS.append(("signal : un écouteur du document sans signal dans compact.js (il s'accumulerait à chaque montage)", lambda: not juger_signal(copie_mutee([(
    "compact.js", 'document.addEventListener("click", (ev) => { if (!ev.target.closest(".et-mn")) fermerMenus(); }, { signal });',
    'document.addEventListener("click", (ev) => { if (!ev.target.closest(".et-mn")) fermerMenus(); });')]))[0]))
MUTATIONS.append(("signal : l'observateur de la toile n'est plus déconnecté", lambda: not juger_signal(copie_mutee([(
    "toile.js", '  if (signal) signal.addEventListener("abort", () => taille.disconnect(), { once: true });\n', "")]))[0]))

# le chronomètre du bandeau : la seule minuterie à répétition, arrêtée au signal et au retrait
_COMMUN = code_seul(open(os.path.join(JS, "commun.js"), encoding="utf-8").read())
chk("commun.js : le chronomètre du bandeau .et-demarre (setInterval) s'arrête à l'abandon du signal du montage et au retrait du bandeau",
    len(re.findall(r"\bsetInterval\s*\(", _COMMUN)) == 1 and 'signal.addEventListener("abort", arreter' in _COMMUN
    and re.search(r"retirer\(\) \{ arreter\(\);", _COMMUN) is not None, "")

# plus de voile : ni js/ ni les feuilles ne nomment #demarrage ou .demarrage
_voile = {}
for dossier, ext in ((JS, ".js"), (CSS, ".css"), (os.path.join(CSS, "modules"), ".css")):
    for f in sorted(os.listdir(dossier)):
        if f.endswith(ext):
            s = code_seul(open(os.path.join(dossier, f), encoding="utf-8").read()) if ext == ".js" else re.sub(r"/\*.*?\*/", "", open(os.path.join(dossier, f), encoding="utf-8").read(), flags=re.S)
            if re.search(r"#demarrage|\.demarrage\b|[\"']demarrage-etape|getElementById\(\s*[\"']demarrage", s):
                _voile[f] = True
chk("plus de voile : aucun script ni aucune feuille ne nomme #demarrage, .demarrage ou #demarrage-etape (S4-D7)", not _voile, _voile)
chk("js/redirection.js a disparu (app.js mène une page de redirection ouverte directement à la page qui la sert)",
    not os.path.exists(os.path.join(JS, "redirection.js")) and "corps.dataset.vers + location.search + location.hash" in open(os.path.join(JS, "app.js"), encoding="utf-8").read())

# ------------------------------------------------------------------ 9. les mutations
print()
for nom, rougit in MUTATIONS:
    try:
        r = rougit()
    except Exception as e:  # noqa: BLE001
        r, nom = False, f"{nom} (essai en erreur : {e!r})"
    chk(f"mutation qui doit rougir : {nom}", r)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
