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
- le BANC (revue S4 : etat-04, 05, 06, 08, deploiement-02 à 05, le refus anonyme ; le CONTRAT du ctx documenté en tête de
  js/app.js) : app.js tel quel dans un faux document (historique avec index et go, adresse, réseau immédiat / différé /
  suspendu / en panne, feuilles retenues ou échouées, stockage de session, document rechargé), les écrans et le moteur
  remplacés par des doublures qui disent ce qu'app.js leur fait — Retour puis Avance pendant une lecture, double clic,
  Retour avant la réponse d'une modification gardée, correction d'adresse après un rechargement, bfcache, version
  changée hors ligne, copie des pages lues pour la vie du document, Retour hors ligne recalé, délai de 10 s et
  indicateur après 300 ms, titre de l'écran quitté sur ses messages, ctx.enCours / apresEnregistrements /
  rendreDefilement / etatEntree / garderEtatEntree, annonce de la clé au moteur ;
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

# ------------------------------------------------------------------ 9. la navigation dans un faux document (le BANC)
# js/app.js tel quel, dans un faux document : l'historique (entrées, index, popstate, go), l'adresse, le réseau (lecture
# immédiate, différée, suspendue, en panne ; navigator.onLine), les feuilles de <head> (chargées, ou retenues), le corps
# échangé, le message (#toast), le stockage de session. Les ÉCRANS (etabli.js, accueil.js, bibliotheque.js) et le MOTEUR
# (moteur.js) sont des DOUBLURES qui disent ce qu'on leur fait : ce qu'app.js leur donne (ctx), quand il les monte et les
# ferme — c'est le contrat entre l'application et les écrans qui se prouve ici (revue S4 : etat-04, etat-05, etat-06,
# etat-08, deploiement-02 à 05, le refus anonyme ; le contrat du lot « écrans »). commun.js, sonde.js, compact.js,
# modules.js sont les vrais.
DOUBLURES = {
    "etabli.js": "// doublure du banc (tests/test_app_navigation.py)\nexport function ouvrirEtabli(hote, ctx) { return globalThis.__banc.ouvrir(\"module\", hote, ctx); }\n",
    "accueil.js": "// doublure du banc\nexport function ouvrirAccueil(hote, ctx) { return globalThis.__banc.ouvrir(\"accueil\", hote, ctx); }\n",
    "bibliotheque.js": "// doublure du banc\nexport function ouvrirBibliotheque(hote, ctx) { return globalThis.__banc.ouvrir(\"bibliotheque\", hote, ctx); }\n",
    "moteur.js": ("// doublure du banc\nexport class EcranFerme extends Error {}\n"
                  "export function moteurPartage() { return globalThis.__banc.moteur; }\n"
                  "export function moteurExiste() { return !!globalThis.__banc.moteurExiste; }\n"),
}


def copie_banc(mutations=()):
    """Une copie de js/ (``mutations`` appliquées, comme copie_mutee) où les écrans et le moteur sont des doublures."""
    d = copie_mutee(list(mutations))
    for f, src in DOUBLURES.items():
        open(os.path.join(d, f), "w", encoding="utf-8").write(src)
    return d


BANC = r"""
const base = process.argv[1], SCEN = process.argv[2];
const RACINE = new URL("../", base).href;
const V = "v-banc";
const st0 = globalThis.setTimeout.bind(globalThis);
const calme = () => new Promise((ok) => st0(ok, 20));
const sommeil = (ms) => new Promise((ok) => st0(ok, ms));
const R = globalThis.__banc = { montes: [], surFermer: null, moteur: null, moteurExiste: false, toasts: [], ordinaires: [], scrolls: [],
  delais: [], lectures: [], ariaBusy: [], mode: {}, lacher: {}, panne: false, versionManifeste: V, decalage: 0, accelerer: false,
  feuillesLentes: new Set(), actif: null, rejets: [] };
// comme un navigateur : un rejet que personne ne traite est consigné, il n'arrête pas tout
process.on("unhandledRejection", (e) => { R.rejets.push(String((e && e.message) || e)); });
const fauxMoteur = ({ etat = "pret", annoncer = true } = {}) => {
  const m = { etat, pret: Promise.resolve(), prechauffes: [], annonces: [], prechauffer(k) { m.prechauffes.push(k); },
    client() { return { demander: async () => ({ ok: true }), fermer: async () => {} }; } };
  if (annoncer) m.annoncer = (k) => m.annonces.push(k);
  return m;
};
R.moteur = fauxMoteur();
R.ouvrir = (sorte, hote, ctx) => {
  const e = { n: R.montes.length + 1, sorte, page: hote.dataset.page, module: ctx.module, bibliotheque: ctx.bibliotheque,
    note: ctx.params.get("note"), ctx, ferme: false, adresse: location.href.slice(RACINE.length) };
  R.montes.push(e);
  return { fermer: () => { e.ferme = true; return (R.surFermer && R.surFermer(e)) || { note: null, enregistree: Promise.resolve({ refus: [] }) }; } };
};
// ---- le temps : les délais de 5 s et plus accélérés (×100) quand R.accelerer ; Date.now décalable
globalThis.setTimeout = (f, ms = 0, ...a) => { R.delais.push(ms); return st0(f, R.accelerer && ms >= 5000 ? ms / 100 : ms, ...a); };
const now0 = Date.now; Date.now = () => now0() + R.decalage;
// ---- les pages du faux site : leur texte nomme la page et sa version, le faux DOMParser les relit
const BASE_CSS = ["css/charte.css", "css/site.css"];
const MOD = (k, t) => ({ page: "module", titre: `${t} · Stab Méthodes`, attrs: { "data-module": k }, feuilles: [...BASE_CSS, "css/etabli.css"] });
const PAGES = { "": { page: "accueil", titre: "Stab Méthodes", feuilles: BASE_CSS }, poutre: MOD("poutre", "Poutre BA"), dalle: MOD("dalle", "Dalle"),
  assemblages: MOD("assemblages", "Assemblages"), soudures: MOD("soudures", "Soudures"), corniere: MOD("corniere", "Cornières"),
  flambement: MOD("flambement", "Flambement"),
  "bibliotheque-beton": { page: "bibliotheque", titre: "Béton armé · Stab Méthodes", attrs: { "data-bibliotheque": "beton", "data-titre": "Béton armé" },
    feuilles: [...BASE_CSS, "css/bibliotheque.css"] } };
const texteDe = (nom, v = V) => `<!doctype html><html><head><meta name="stab-version" content="${v}"></head><body><!--banc:${nom}--></body></html>`;
const nomDe = (u) => String(u && u.href ? u.href : u).slice(RACINE.length).replace(/[?#].*$/, "");
// ---- les éléments
const kebab = (k) => "data-" + k.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
const camel = (n) => n.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
function el(tag = "div", attrs = {}) {
  const at = new Map(Object.entries(attrs)), classes = new Set();
  const e = {
    tagName: tag.toUpperCase(), style: {}, hidden: false, parentNode: null, enfants: [], textContent: "",
    get childNodes() { return e.enfants.slice(); },
    classList: { add: (...c) => c.forEach((x) => classes.add(x)), remove: (...c) => c.forEach((x) => classes.delete(x)), contains: (c) => classes.has(c),
      toggle: (c, f) => { const on = f === undefined ? !classes.has(c) : !!f; if (on) classes.add(c); else classes.delete(c); return on; } },
    get className() { return [...classes].join(" "); }, set className(v) { classes.clear(); String(v).split(/\s+/).filter(Boolean).forEach((x) => classes.add(x)); },
    get attributes() { return [...at].map(([name, value]) => ({ name, value })); },
    setAttribute(n, v) { at.set(n, String(v)); if (e.surAttribut) e.surAttribut(n, String(v)); }, getAttribute(n) { return at.has(n) ? at.get(n) : null; },
    removeAttribute(n) { at.delete(n); }, hasAttribute(n) { return at.has(n); },
    addEventListener() {}, removeEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; },
    append(...xs) { for (const x of xs) { if (x.parentNode) x.remove(); x.parentNode = e; e.enfants.push(x); } },
    replaceChildren(...xs) { for (const x of e.enfants) x.parentNode = null; e.enfants = []; e.append(...xs); },
    remove() { const p = e.parentNode; if (p) { p.enfants = p.enfants.filter((x) => x !== e); e.parentNode = null; } },
    after() {}, before() {}, focus() {}, blur() {}, closest() { return null; }, contains() { return false; }, checkVisibility() { return false; },
  };
  e.dataset = new Proxy({}, {
    get: (_, k) => (typeof k === "string" && at.has(kebab(k)) ? at.get(kebab(k)) : undefined),
    set: (_, k, v) => { at.set(kebab(k), String(v)); return true; },
    deleteProperty: (_, k) => { at.delete(kebab(k)); return true; },
    ownKeys: () => [...at.keys()].filter((n) => n.startsWith("data-")).map(camel),
    getOwnPropertyDescriptor: () => ({ enumerable: true, configurable: true }),
  });
  return e;
}
const toastEl = el("div");
Object.defineProperty(toastEl, "hidden", { get: () => true, set: (v) => { if (!v) R.toasts.push([toastEl.textContent, toastEl.classList.contains("erreur")]); } });
// ---- les feuilles de <head> : une feuille posée se charge au tour suivant, sauf si elle est « lente » (R.feuillesLentes)
const tete = [];
function lien() {
  const l = el("link"), ec = {};
  l.rel = "stylesheet"; l.sheet = null; l.media = ""; l.disabled = false;
  l.addEventListener = (t, f) => { (ec[t] = ec[t] || []).push(f); };
  l.charger = () => { l.sheet = { disabled: false }; const fs = ec.load || []; ec.load = []; fs.forEach((f) => f()); };
  l.echouer = () => { const fs = ec.error || []; ec.error = []; fs.forEach((f) => f()); };
  Object.defineProperty(l, "href", { get: () => l._href || "", set: (v) => { l._href = new URL(v, RACINE).href; if (!R.feuillesLentes.has(l._href.slice(RACINE.length))) st0(() => l.charger(), 0); } });
  l.after = (x) => { const i = tete.indexOf(x); if (i >= 0) tete.splice(i, 1); tete.splice(tete.indexOf(l) + 1, 0, x); };
  l.before = (x) => { const i = tete.indexOf(x); if (i >= 0) tete.splice(i, 1); tete.splice(tete.indexOf(l), 0, x); };
  l.remove = () => { const i = tete.indexOf(l); if (i >= 0) tete.splice(i, 1); };
  l.removeAttribute = (n) => { if (n === "media") l.media = ""; };
  return l;
}
// ---- l'historique, l'adresse, les écouteurs de la fenêtre et du document
const H = { entrees: [], i: 0 };
let W = {}, D = {};
const ecouter = (table) => (t, f, o) => {
  if (o && o.signal) { if (o.signal.aborted) return; o.signal.addEventListener("abort", () => { table[t] = (table[t] || []).filter((x) => x !== f); }); }
  (table[t] = table[t] || []).push(f);
};
globalThis.addEventListener = (t, f, o) => ecouter(W)(t, f, o);
const emettre = (t, ev = {}) => { for (const f of (W[t] || []).slice()) f(ev); };
const emettreDoc = (t, ev = {}) => { for (const f of (D[t] || []).slice()) f(ev); };
globalThis.history = {
  scrollRestoration: "auto",
  get state() { return H.entrees[H.i].state; },
  replaceState(s, _, u) { H.entrees[H.i] = { url: u ? new URL(u, H.entrees[H.i].url).href : H.entrees[H.i].url, state: structuredClone(s) }; },
  pushState(s, _, u) { H.entrees.splice(H.i + 1); H.entrees.push({ url: new URL(u, H.entrees[H.i].url).href, state: structuredClone(s) }); H.i++; },
  go(d) { st0(() => { const j = H.i + d; if (j < 0 || j >= H.entrees.length) return; H.i = j; emettre("popstate", {}); }, 0); },
};
globalThis.location = { get href() { return H.entrees[H.i].url; }, get search() { return new URL(this.href).search; }, get hash() { return new URL(this.href).hash; },
  get pathname() { return new URL(this.href).pathname; }, assign(u) { R.ordinaires.push(["assign", String(u).slice(RACINE.length)]); },
  replace(u) { R.ordinaires.push(["replace", String(u).slice(RACINE.length)]); }, reload() {} };
const traverser = (d) => { H.i += d; emettre("popstate", {}); };
const allerEntree = (j) => { H.i = j; emettre("popstate", {}); };
// ---- le réseau
const nav = { onLine: true };
Object.defineProperty(globalThis, "navigator", { value: nav, configurable: true });
globalThis.fetch = (u, o = {}) => {
  const nom = nomDe(u);
  R.lectures.push(nom);
  if (R.panne || nav.onLine === false) return Promise.reject(new TypeError("Failed to fetch"));
  if (nom === "moteur/manifeste.json") return Promise.resolve({ ok: true, type: "basic", status: 200, json: async () => ({ version: R.versionManifeste, pyodide: "vendor/pyodide-0.29.5" }) });
  const rep = { type: "basic", ok: !!PAGES[nom], status: PAGES[nom] ? 200 : 404, text: async () => texteDe(nom) };
  const mode = R.mode[nom] || "ok";
  if (mode === "ok") return Promise.resolve(rep);
  return new Promise((ok, ko) => {
    if (o.signal) o.signal.addEventListener("abort", () => { const e = new Error("abandon"); e.name = "AbortError"; ko(e); });
    if (mode === "attente") R.lacher[nom] = () => ok(rep);
  });
};
globalThis.DOMParser = class { parseFromString(t) {
  const nom = (/<!--banc:([^>]*?)-->/.exec(t) || [])[1], v = (/name="stab-version" content="([^"]*)"/.exec(t) || [])[1], p = PAGES[nom];
  if (!p) return { title: "", querySelector: () => null, querySelectorAll: () => [], body: el("body") };
  return { title: p.titre, querySelector: (s) => (s.includes("stab-version") && v ? { content: v } : null),
    querySelectorAll: (s) => (s.includes("stylesheet") ? p.feuilles.map((h) => ({ getAttribute: (a) => (a === "href" ? h : null) })) : []),
    body: el("body", { "data-page": p.page, ...(p.attrs || {}) }) };
} };
// ---- le reste du navigateur
globalThis.self = { location: { href: RACINE } };
globalThis.scrollY = 0;
globalThis.scrollTo = (x, y) => { R.scrolls.push(y); globalThis.scrollY = y; };
globalThis.requestAnimationFrame = (f) => st0(f, 0);
globalThis.performance.getEntriesByType = () => [];
const magasin = () => { const m = new Map(); return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k),
  key: (i) => [...m.keys()][i] ?? null, get length() { return m.size; }, _tout: () => Object.fromEntries(m) }; };
globalThis.localStorage = magasin();
globalThis.sessionStorage = magasin();
globalThis.alert = () => {}; globalThis.confirm = () => true;
// ---- un document : sa page, ses feuilles, son corps ; app.js importé (un module NEUF à chaque document)
let A = null, nDoc = 0;
async function ouvrirDocument(chemin, { capture = true, garder = false } = {}) {
  const nom = chemin.replace(/[?#].*$/, ""), p = PAGES[nom];
  if (!garder) { H.entrees = [{ url: RACINE + chemin, state: null }]; H.i = 0; }
  tete.length = 0;
  for (const h of p.feuilles) { const l = lien(); l._href = new URL(h, RACINE).href; l.sheet = { disabled: false }; tete.push(l); }
  const body = el("body", { "data-page": p.page, ...(p.attrs || {}) });
  body.surAttribut = (n, v) => { if (n === "aria-busy") R.ariaBusy.push(v); };
  const html = el("html");
  Object.defineProperty(html, "outerHTML", { get: () => (capture ? texteDe(nom).replace(/^<!doctype html>/, "") : undefined) });
  W = {}; D = {}; R.montes = [];
  globalThis.document = {
    body, documentElement: html, readyState: "complete", visibilityState: "visible", title: p.titre,
    head: { querySelector: () => tete[0] || null, append: (x) => tete.push(x) },
    get activeElement() { return R.actif || body; },
    createElement: (t) => (t === "link" ? lien() : el(t)), adoptNode: (x) => x,
    getElementById: (id) => (id === "toast" ? toastEl : null),
    querySelector: (s) => (s.includes("stab-version") ? { content: V } : null),
    querySelectorAll: (s) => (s.includes("stylesheet") ? tete.slice() : []),
    addEventListener: (t, f, o) => ecouter(D)(t, f, o),
  };
  nDoc++;
  A = await import(base + "app.js" + (nDoc > 1 ? `?doc=${nDoc}` : ""));
  await calme();
  return A;
}
const vue = () => ({ page: document.body.dataset.page, montes: R.montes.length, adresse: location.href.slice(RACINE.length),
  cle: history.state && history.state.cle, i: H.i, n: H.entrees.length, busy: document.body.getAttribute("aria-busy"),
  indicateur: document.body.enfants.some((x) => /\bapp-attente\b/.test(x.className || "")) });
const dernier = () => R.montes[R.montes.length - 1] || {};
const borne = (p, ms = 1500) => Promise.race([p, sommeil(ms)]);

const S = {};
// etat-04 : Retour puis Avance pendant la lecture de la page du Retour
S.retour_avance = async () => {
  await ouvrirDocument("");
  await A.naviguer(RACINE + "bibliotheque-beton"); await calme();
  const cle1 = history.state.cle;
  R.mode[""] = "attente";
  traverser(-1); await calme();
  const pendant = vue();
  traverser(+1); await calme();
  R.lacher[""](); await calme(); await calme();
  return { pendant, fin: vue(), cle1, ordinaires: R.ordinaires };
};
// etat-08 : double clic — la seconde navigation attend la feuille que la première a posée
S.double_clic = async () => {
  await ouvrirDocument("");
  R.feuillesLentes.add("css/bibliotheque.css");
  A.naviguer(RACINE + "bibliotheque-beton"); await calme();
  A.naviguer(RACINE + "bibliotheque-beton"); await calme();
  const avant = vue();
  const l = tete.find((x) => x.href.endsWith("css/bibliotheque.css"));
  if (l) l.charger();
  await calme(); await calme();
  return { avant, fin: vue(), pages: R.montes.map((e) => e.page) };
};
// C9 : une feuille posée dont le chargement a ÉCHOUÉ (sans contenu, plus en chargement) n'est pas prise pour chargée :
// une neuve la remplace, attendue
S.feuille_echouee = async () => {
  await ouvrirDocument("");
  R.feuillesLentes.add("css/bibliotheque.css");
  const p = A.naviguer(RACINE + "bibliotheque-beton"); await calme();
  const l1 = tete.find((x) => x.href.endsWith("css/bibliotheque.css"));
  if (l1) l1.echouer();
  await borne(p); await calme();
  const apresEchec = { ...vue(), ordinaires: R.ordinaires.slice() };
  A.naviguer(RACINE + "bibliotheque-beton"); await calme();
  const avant = vue();
  const l2 = tete.find((x) => x.href.endsWith("css/bibliotheque.css"));
  const neuve = !!l2 && l2 !== l1;
  if (l2) l2.charger();
  await calme(); await calme();
  return { apresEchec, avant, neuve, fin: vue(), feuilles: tete.filter((x) => x.href.endsWith("css/bibliotheque.css")).length };
};
// etat-05 (a) : Retour AVANT la réponse de la modification gardée d'une note provisoire
S.retour_avant_reponse = async () => {
  await ouvrirDocument("poutre");
  let fin = null;
  R.surFermer = (e) => (e.n === 1 ? { note: "P", enregistree: new Promise((ok) => { fin = ok; }) } : null);
  await A.naviguer(RACINE + "assemblages"); await calme();
  traverser(-1); await calme();
  const pendant = vue();
  R.montes[0].ctx.adresse("?note=P");          // la réponse arrive : l'écran fermé écrit la note P et donne son adresse (C3)
  fin({ refus: [] }); await calme(); await calme();
  return { pendant, fin: vue(), dernier: { page: dernier().page, module: dernier().module, note: dernier().note }, ordinaires: R.ordinaires };
};
// etat-05 (b) : la correction d'adresse survit au rechargement de l'onglet (sessionStorage), appliquée au démarrage
S.correction_rechargement = async () => {
  await ouvrirDocument("poutre");
  const k0 = history.state.cle;
  let fin = null;
  R.surFermer = (e) => (e.n === 1 ? { note: "P", enregistree: new Promise((ok) => { fin = ok; }) } : null);
  await A.naviguer(RACINE + "assemblages"); await calme();
  R.montes[0].ctx.adresse("?note=P");          // la réponse arrive APRÈS le départ : l'entrée de la poutre n'est plus la courante
  fin({ refus: [] }); await calme();
  const stocke = JSON.stringify(sessionStorage._tout());
  // F5 sur assemblages, puis Retour : le navigateur charge l'entrée de la poutre dans un document NEUF, son état rendu
  H.i = 0;
  await ouvrirDocument("poutre", { garder: true });
  return { k0, stocke, fin: vue(), note: dernier().note, ordinaires: R.ordinaires };
};
// etat-06 : retour par le bfcache (pageshow persisted) : l'écran remonté depuis le stockage à jour, le moteur vivant
S.bfcache = async () => {
  await ouvrirDocument("poutre?note=X");
  const cle = history.state.cle;
  emettre("pageshow", { persisted: false }); await calme();
  const sans = vue();
  emettre("pageshow", { persisted: true }); await calme(); await calme();
  return { cle, sans, fin: vue(), premierFerme: R.montes[0].ferme, note: dernier().note, ordinaires: R.ordinaires };
};
// deploiement-02 : version changée ET réseau absent : on reste (ou la copie de la page) — jamais de navigation ordinaire
S.version_horsligne = async () => {
  await ouvrirDocument("");
  await A.naviguer(RACINE + "poutre"); await calme();
  R.versionManifeste = "autre";
  emettreDoc("visibilitychange"); await calme();
  const message = R.toasts.map((t) => t[0]);
  nav.onLine = false; R.toasts.length = 0;
  await borne(A.naviguer(RACINE + "dalle")); await calme();
  const jamais = { ...vue(), toasts: R.toasts.map((t) => t[0]), ordinaires: R.ordinaires.slice() };
  await borne(A.naviguer(RACINE)); await calme();
  const deja = { ...vue(), ordinaires: R.ordinaires.slice() };
  traverser(-1); await calme(); await calme();
  const retour = { ...vue(), ordinaires: R.ordinaires.slice() };
  nav.onLine = true; R.panne = true; R.toasts.length = 0;
  await borne(A.naviguer(RACINE + "soudures")); await calme();
  const menteur = { ...vue(), toasts: R.toasts.map((t) => t[0]), ordinaires: R.ordinaires.slice() };
  R.panne = false;
  await borne(A.naviguer(RACINE + "soudures")); await calme();
  return { message, jamais, deja, retour, menteur, enLigne: R.ordinaires.slice(), PAS_DE_RESEAU: A.PAS_DE_RESEAU };
};
// deploiement-03 : un écran déjà ouvert se rouvre hors ligne (la page gardée pour la vie du document) ; le message juste
S.memoire = async () => {
  await ouvrirDocument("");
  await A.naviguer(RACINE + "poutre"); await calme();
  await A.naviguer(RACINE); await calme();
  R.decalage = 31000; nav.onLine = false; R.toasts.length = 0;
  await borne(A.naviguer(RACINE + "poutre")); await calme();
  const deja = { ...vue(), module: dernier().module, toasts: R.toasts.map((t) => t[0]) };
  await borne(A.naviguer(RACINE + "soudures")); await calme();
  const jamais = { ...vue(), toasts: R.toasts.map((t) => t[0]), ordinaires: R.ordinaires.slice() };
  return { deja, jamais, PAS_DE_RESEAU: A.PAS_DE_RESEAU };
};
// deploiement-03 (bis) : la page de CE document (lue par le navigateur, pas par app.js) se rouvre aussi hors ligne
S.memoire_initiale = async () => {
  await ouvrirDocument("poutre");
  await A.naviguer(RACINE + "dalle"); await calme();
  nav.onLine = false; R.toasts.length = 0;
  traverser(-1); await calme(); await calme();
  return { fin: vue(), module: dernier().module, toasts: R.toasts.map((t) => t[0]), ordinaires: R.ordinaires };
};
// deploiement-04 : Retour hors ligne vers une entrée qui ne peut pas s'ouvrir : l'historique revient sur l'écran affiché
S.retour_horsligne = async () => {
  await ouvrirDocument("", { capture: false });
  await A.naviguer(RACINE + "poutre"); await calme();
  await A.naviguer(RACINE + "dalle"); await calme();
  const k2 = history.state.cle;
  nav.onLine = false; R.toasts.length = 0;
  traverser(-2); await calme(); await calme(); await calme();
  const fin = { ...vue(), toasts: R.toasts.map((t) => t[0]), entrees: H.entrees.map((e) => e.url.slice(RACINE.length)) };
  // une entrée sans rang (écrite par une version d'avant) : l'entrée d'arrivée est réécrite à l'adresse et à la clé de l'écran affiché
  H.entrees[0].state = { cle: "ancien.1", defile: 0 };
  allerEntree(0); await calme(); await calme(); await calme();
  const sansRang = vue();
  return { k2, fin, sansRang, ordinaires: R.ordinaires, PAS_DE_RESEAU: A.PAS_DE_RESEAU };
};
// deploiement-05 : un délai (10 s, accéléré ici) et un indicateur visible après 300 ms, retiré quand la navigation finit
S.lent = async () => {
  await ouvrirDocument("");
  R.mode.poutre = "attente";
  const p = A.naviguer(RACINE + "poutre");
  await sommeil(150); const a150 = vue();
  await sommeil(300); const a450 = vue();
  if (R.lacher.poutre) R.lacher.poutre();
  await borne(p); await calme();
  const fin1 = vue();
  const n0 = R.ariaBusy.length;
  await A.naviguer(RACINE + "dalle"); await sommeil(400);
  const rapide = { ...vue(), poses: R.ariaBusy.slice(n0) };
  R.mode.soudures = "attente"; R.mode.assemblages = "attente";
  const q1 = A.naviguer(RACINE + "soudures"); await sommeil(400);
  const q2 = A.naviguer(RACINE + "assemblages"); await calme();
  const remplacee = vue();
  if (R.lacher.soudures) R.lacher.soudures();      // la remplacée finit la PREMIÈRE : l'indicateur reste à la dernière
  await borne(q1); await calme();
  const tardive = vue();
  if (R.lacher.assemblages) R.lacher.assemblages();
  await borne(q2); await calme();
  const fin3 = vue();
  R.mode.corniere = "attente";
  A.naviguer(RACINE + "corniere"); await sommeil(400);
  const avantAnnulee = vue();
  emettre("popstate", {}); await calme();
  const annulee = vue();
  R.accelerer = true; R.mode.flambement = "suspendu"; R.toasts.length = 0;
  await borne(A.naviguer(RACINE + "flambement")); await calme();
  const suspendue = { ...vue(), toasts: R.toasts.map((t) => t[0]), ordinaires: R.ordinaires.slice(), delai: R.delais.includes(10000) };
  return { a150, a450, fin1, rapide, remplacee, fin3, tardive, avantAnnulee, annulee, suspendue, RESEAU_MUET: A.RESEAU_MUET || null };
};
// le refus anonyme : un message d'ERREUR dit pendant la fermeture porte le titre de l'écran quitté ; un refus rendu par
// l'écran (res.refus) passe tel quel (l'établi le préfixe lui-même)
S.refus = async () => {
  await ouvrirDocument("poutre");
  document.title = "Note P1 · Stab Méthodes";
  const C = await import(base + "commun.js");
  R.surFermer = (e) => (e.n === 1 ? { note: "P", enregistree: Promise.resolve({ refus: ["« Note P1 » (Poutre BA) : Le moteur a refusé : essai"] }) } : null);
  R.actif = { blur() { C.toast("Valeur refusée : « -5 »", true); C.toast("Météo récupérée"); } };
  await A.naviguer(RACINE + "dalle"); R.actif = null; await calme(); await calme();
  return { toasts: R.toasts };
};
// le contrat (1) : ctx.enCours(id), ctx.apresEnregistrements()
S.contrat_enregistrements = async () => {
  await ouvrirDocument("poutre");
  let d1, d2, d3;
  R.surFermer = (e) => (e.n === 1 ? { note: "P", enregistree: new Promise((ok) => { d1 = ok; }) }
    : e.n === 2 ? { note: "Q", enregistree: new Promise((ok, ko) => { d2 = ko; }) }
    : e.n === 3 ? { note: "S", enregistree: new Promise((ok) => { d3 = ok; }) } : null);
  await A.naviguer(RACINE + "dalle"); await calme();
  await A.naviguer(RACINE + "assemblages"); await calme();
  const ctx = R.montes[2].ctx;
  if (typeof ctx.enCours !== "function" || typeof ctx.apresEnregistrements !== "function") return { absent: true };
  const e1 = { P: ctx.enCours("P"), Q: ctx.enCours("Q"), X: ctx.enCours("X") };
  let fini = false;
  ctx.apresEnregistrements().then(() => { fini = true; }, () => { fini = "rejetée"; });
  await A.naviguer(RACINE + "soudures"); await calme();
  d1({ refus: [] }); await calme();
  const f1 = fini, e2 = { P: ctx.enCours("P"), Q: ctx.enCours("Q") };
  d2(new Error("panne")); await calme();
  const f2 = fini;
  let s = false;
  R.montes[3].ctx.apresEnregistrements().then(() => { s = true; }); await calme();
  const s1 = s;
  d3({ refus: [] }); await calme();
  let rien = false;
  R.montes[3].ctx.apresEnregistrements().then(() => { rien = true; }); await calme();
  return { e1, f1, e2, f2, s1, s2: s, rien, rejets: R.rejets };
};
// le contrat (2) : ctx.rendreDefilement() — une fois, puis oublié ; sinon l'application à 1,5 s ; l'accueil, à l'image suivante
S.contrat_defilement = async () => {
  await ouvrirDocument("");
  globalThis.scrollY = 180;
  await A.naviguer(RACINE + "poutre"); await calme();
  globalThis.scrollY = 420;
  await A.naviguer(RACINE + "dalle"); await calme();
  const cd = R.montes[2].ctx;
  if (typeof cd.rendreDefilement !== "function") return { absent: true };
  const aller = { scrolls: R.scrolls.slice(), rendu: cd.rendreDefilement() };
  globalThis.scrollY = 250; R.scrolls.length = 0;
  traverser(-1); await calme();
  const c = R.montes[3].ctx;
  const s1 = R.scrolls.slice();
  const r1 = c.rendreDefilement(); const s2 = R.scrolls.slice();
  const r2 = c.rendreDefilement(); const s3 = R.scrolls.slice();
  await sommeil(1700); const s4 = R.scrolls.slice();
  R.scrolls.length = 0;
  traverser(+1); await calme();
  const t1 = R.scrolls.slice();
  await sommeil(1700); const t2 = R.scrolls.slice();
  R.scrolls.length = 0;
  traverser(-2); await calme();
  return { aller, s1, r1, s2, r2, s3, s4, t1, t2, accueil: R.scrolls.slice() };
};
// le contrat (3) : ctx.etatEntree, ctx.garderEtatEntree(obj) — seulement sur l'entrée de CET écran, s'il l'occupe encore
S.contrat_etat_entree = async () => {
  await ouvrirDocument("poutre");
  const c1 = R.montes[0].ctx;
  if (typeof c1.garderEtatEntree !== "function") return { absent: true };
  const cle0 = history.state.cle, lu1 = c1.etatEntree;
  const ok1 = c1.garderEtatEntree({ onglet: "resultat" }), ok1b = c1.garderEtatEntree({ zoom: 2 });
  const st1 = history.state;
  await A.naviguer(RACINE + "dalle"); await calme();
  const ok2 = c1.garderEtatEntree({ onglet: "dessin" }), st2 = history.state;
  traverser(-1); await calme();
  const c3 = R.montes[2].ctx, lu3 = c3.etatEntree;
  const ok3 = c1.garderEtatEntree({ onglet: "x" }), st3 = history.state;
  emettre("pageshow", { persisted: true }); await calme(); await calme();
  const ok4 = c3.garderEtatEntree({ onglet: "y" }), st4 = history.state;
  const lu4 = R.montes[3] ? R.montes[3].ctx.etatEntree : null;
  return { cle0, lu1, ok1, ok1b, st1, ok2, st2, lu3, ok3, st3, ok4, st4, lu4 };
};
// le contrat (4) : moteurApp().annoncer?.(k) au montage d'un écran de module ou de bibliothèque, moteur pas prêt
S.contrat_annonce = async () => {
  R.moteur = fauxMoteur({ etat: "demarrage" });
  await ouvrirDocument("poutre");
  await A.naviguer(RACINE + "bibliotheque-beton"); await calme();
  await A.naviguer(RACINE + "predalle?note=n1"); await calme();
  await A.naviguer(RACINE); await calme();
  R.moteur.etat = "pret";
  await A.naviguer(RACINE + "soudures"); await calme();
  const annonces = R.moteur.annonces.slice();
  R.moteur = fauxMoteur({ etat: "demarrage", annoncer: false });
  await A.naviguer(RACINE + "assemblages"); await calme();
  return { annonces, fin: vue(), modules: R.montes.map((e) => e.module || e.bibliotheque || e.sorte) };
};

let res;
try { res = await S[SCEN](); } catch (e) { res = { erreur: String(e && e.stack || e).slice(0, 600) }; }
console.log(JSON.stringify(res));
process.exit(0);
"""


def banc(scenario, d):
    return node(BANC, url_js(d), scenario, timeout=60)


def egal(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def juger_banc(scenario, d):
    """(vrai si le scénario du banc tient, ce qui a été vu)."""
    r = banc(scenario, d)
    if "erreur" in r:
        return False, r
    J = JUGES[scenario]
    try:
        return bool(J(r)), r
    except Exception as e:  # noqa: BLE001
        return False, {"juge": repr(e), "vu": r}


def _sans_ordinaire(*xs):
    return all(not x for x in xs)


JUGES = {
    "retour_avance": lambda r: (r["fin"]["page"] == "bibliotheque" and r["fin"]["montes"] == 2 and r["fin"]["adresse"] == "bibliotheque-beton"
                                and r["fin"]["cle"] == r["cle1"] and not r["fin"]["busy"] and not r["fin"]["indicateur"] and not r["ordinaires"]),
    "double_clic": lambda r: (r["avant"]["page"] == "accueil" and r["avant"]["montes"] == 1 and r["fin"]["page"] == "bibliotheque"
                              and r["pages"] == ["accueil", "bibliotheque"]),
    "feuille_echouee": lambda r: (r["apresEchec"]["page"] == "accueil" and r["avant"]["page"] == "accueil" and r["avant"]["montes"] == 1 and r["neuve"]
                                  and r["fin"]["page"] == "bibliotheque" and r["feuilles"] == 1),
    "retour_avant_reponse": lambda r: (r["pendant"]["montes"] == 2 and r["fin"]["montes"] == 3 and r["dernier"] == {"page": "module", "module": "poutre", "note": "P"}
                                       and r["fin"]["adresse"] == "poutre?note=P" and not r["ordinaires"]),
    "correction_rechargement": lambda r: (r["k0"] in r["stocke"] and "note=P" in r["stocke"] and r["fin"]["adresse"] == "poutre?note=P"
                                          and r["note"] == "P" and r["fin"]["cle"] == r["k0"] and not r["ordinaires"]),
    "bfcache": lambda r: (r["sans"]["montes"] == 1 and r["fin"]["montes"] == 2 and r["premierFerme"] and r["note"] == "X"
                          and r["fin"]["cle"] == r["cle"] and r["fin"]["adresse"] == "poutre?note=X" and not r["ordinaires"]),
    "version_horsligne": lambda r: (any("mis à jour" in t for t in r["message"])
                                    and r["jamais"]["adresse"] == "poutre" and r["jamais"]["toasts"] == [r["PAS_DE_RESEAU"]]
                                    and r["deja"]["page"] == "accueil" and r["deja"]["adresse"] == ""
                                    and r["retour"]["page"] == "module" and r["retour"]["adresse"] == "poutre"
                                    and r["menteur"]["adresse"] == "poutre" and r["menteur"]["toasts"] == [r["PAS_DE_RESEAU"]]
                                    and _sans_ordinaire(r["jamais"]["ordinaires"], r["deja"]["ordinaires"], r["retour"]["ordinaires"], r["menteur"]["ordinaires"])
                                    and r["enLigne"] == [["assign", "soudures"]]),
    "memoire": lambda r: (r["deja"]["page"] == "module" and r["deja"]["module"] == "poutre" and r["deja"]["adresse"] == "poutre" and not r["deja"]["toasts"]
                          and r["jamais"]["adresse"] == "poutre" and r["jamais"]["toasts"] == [r["PAS_DE_RESEAU"]] and not r["jamais"]["ordinaires"]
                          and "pas encore" not in r["PAS_DE_RESEAU"]),
    "memoire_initiale": lambda r: (r["fin"]["page"] == "module" and r["module"] == "poutre" and r["fin"]["adresse"] == "poutre" and r["fin"]["montes"] == 3
                                   and not r["toasts"] and not r["ordinaires"]),
    "retour_horsligne": lambda r: (r["fin"]["adresse"] == "dalle" and r["fin"]["cle"] == r["k2"] and r["fin"]["i"] == 2 and r["fin"]["montes"] == 3
                                   and r["fin"]["entrees"] == ["", "poutre", "dalle"] and r["fin"]["toasts"] == [r["PAS_DE_RESEAU"]]
                                   and r["sansRang"]["adresse"] == "dalle" and r["sansRang"]["cle"] == r["k2"] and r["sansRang"]["i"] == 0
                                   and r["sansRang"]["montes"] == 3 and not r["ordinaires"]),
    "lent": lambda r: (not r["a150"]["busy"] and not r["a150"]["indicateur"] and r["a450"]["busy"] == "true" and r["a450"]["indicateur"]
                       and r["fin1"]["adresse"] == "poutre" and not r["fin1"]["busy"] and not r["fin1"]["indicateur"]
                       and not r["rapide"]["poses"] and not r["rapide"]["busy"]
                       and r["remplacee"]["busy"] == "true" and r["remplacee"]["indicateur"]
                       and r["tardive"]["adresse"] == "dalle" and r["tardive"]["busy"] == "true" and r["tardive"]["indicateur"]
                       and r["fin3"]["adresse"] == "assemblages" and not r["fin3"]["busy"] and not r["fin3"]["indicateur"]
                       and r["avantAnnulee"]["busy"] == "true" and not r["annulee"]["busy"] and not r["annulee"]["indicateur"]
                       and r["suspendue"]["adresse"] == "assemblages" and r["suspendue"]["delai"] and r["RESEAU_MUET"]
                       and r["suspendue"]["toasts"] == [r["RESEAU_MUET"]] and not r["suspendue"]["ordinaires"] and not r["suspendue"]["busy"]),
    "refus": lambda r: (["« Note P1 » : Valeur refusée : « -5 »", True] in r["toasts"]
                        and ["« Note P1 » (Poutre BA) : Le moteur a refusé : essai", True] in r["toasts"]
                        and not any("Météo" in t[0] for t in r["toasts"]) and len(r["toasts"]) == 2),
    "contrat_enregistrements": lambda r: (r["e1"] == {"P": True, "Q": True, "X": False} and r["f1"] is False and r["e2"] == {"P": False, "Q": True}
                                          and r["f2"] is True and r["s1"] is False and r["s2"] is True and r["rien"] is True and not r["rejets"]),
    "contrat_defilement": lambda r: (r["aller"]["rendu"] is None and 0 in r["aller"]["scrolls"] and 420 not in r["s1"] and r["r1"] == 420
                                     and r["s2"][-1:] == [420] and r["r2"] is None and r["s3"] == r["s2"] and r["s4"] == r["s3"]
                                     and 250 not in r["t1"] and r["t2"][-1:] == [250] and r["accueil"][-1:] == [180]),
    "contrat_etat_entree": lambda r: (r["lu1"] == {} and r["ok1"] and r["ok1b"] and r["st1"]["cle"] == r["cle0"] and r["st1"]["ecran"] == {"onglet": "resultat", "zoom": 2}
                                      and r["ok2"] is False and "ecran" not in r["st2"] and r["lu3"] == {"onglet": "resultat", "zoom": 2}
                                      and r["ok3"] is False and r["st3"]["ecran"] == r["st1"]["ecran"] and r["ok4"] is False
                                      and r["st4"]["ecran"] == r["st1"]["ecran"] and r["lu4"] == r["st1"]["ecran"]),
    "contrat_annonce": lambda r: (r["annonces"] == ["poutre", "biblio_beton", "dalle"] and r["fin"]["adresse"] == "assemblages"
                                  and r["modules"] == ["poutre", "beton", "dalle", "accueil", "soudures", "assemblages"]),
}
NOMS_BANC = {
    "retour_avance": "etat-04 · Retour puis Avance pendant la lecture de la page : l'écran monté reste celui de l'entrée courante (aucun montage de l'écran quitté)",
    "double_clic": "etat-08 · double clic : la seconde navigation attend la feuille que la première a posée (l'écran n'est pas monté sans sa feuille, C9)",
    "feuille_echouee": "C9 · une feuille posée dont le chargement a échoué n'est pas prise pour chargée : une neuve la remplace, attendue avant le montage",
    "retour_avant_reponse": "etat-05 · Retour avant la réponse de la modification gardée : l'entrée attend son enregistrement, prend l'adresse ?note= de la note écrite, et la note se rouvre",
    "correction_rechargement": "etat-05 · la correction d'adresse d'une entrée survit au rechargement (sessionStorage) et s'applique au démarrage du document qui rouvre cette entrée",
    "bfcache": "etat-06 · retour par le bfcache (pageshow persisted) : l'écran est remonté (depuis le stockage à jour), sur la même entrée et la même adresse ; pas sans persisted",
    "version_horsligne": "deploiement-02 · version changée et réseau absent (ou muet) : aucune navigation ordinaire — on reste avec le message, ou la page déjà lue s'ouvre ; réseau revenu : ordinaire",
    "memoire": "deploiement-03 · hors ligne, plus de 30 s après : un écran déjà ouvert se rouvre (la page gardée pour la vie du document) ; un écran jamais ouvert : le message juste",
    "memoire_initiale": "deploiement-03 · hors ligne : la page du document lui-même (lue par le navigateur) se rouvre aussi par Retour",
    "retour_horsligne": "deploiement-04 · Retour hors ligne vers une entrée qui ne s'ouvre pas : l'historique revient sur l'entrée de l'écran affiché (history.go), sans perdre d'entrée ; sans rang, l'entrée d'arrivée est réécrite",
    "lent": "deploiement-05 · une lecture qui traîne : l'indicateur (body[aria-busy], .app-attente) après 300 ms et pas avant, retiré à la fin ou à l'annulation ; 10 s au plus, puis on reste avec le message",
    "refus": "refus anonyme · un message d'erreur dit pendant la fermeture porte le titre de l'écran quitté ; un refus rendu par l'écran passe tel quel (une seule fois préfixé)",
    "contrat_enregistrements": "contrat · ctx.enCours(id) dit l'enregistrement gardé qui court ; ctx.apresEnregistrements() attend ceux en cours à l'appel (pas les suivants), jamais rejetée",
    "contrat_defilement": "contrat · ctx.rendreDefilement() rend le défilement du retour UNE fois ; sinon l'application le rend à 1,5 s ; l'accueil à l'image suivante ; rien à l'aller",
    "contrat_etat_entree": "contrat · ctx.etatEntree (lu au montage) et ctx.garderEtatEntree(obj) : fusionné dans l'entrée de CET écran, refusé quand un autre montage l'occupe",
    "contrat_annonce": "contrat · au montage d'un module ou d'une bibliothèque, moteur pas prêt : moteur.annoncer(clé) (biblio_<k>, le module servi) ; pas prêt sans annoncer() : rien ne lève",
}
_banc = copie_banc()
for _s, _nom in NOMS_BANC.items():
    ok, info = juger_banc(_s, _banc)
    chk(f"banc · {_nom}", ok, info)

# chaque correction retirée (sur une copie d'app.js) fait rougir le scénario qui la prouve
MUTATIONS_BANC = [
    ("etat-04 : l'avance vers l'écran monté n'annule plus la navigation en cours", "retour_avance",
     "      jeton++;\n      finirAttente();\n      appliquerCorrection();\n      return;", "      appliquerCorrection();\n      return;"),
    ("etat-08 : une feuille déjà posée n'est plus attendue (ni remplacée)", "double_clic",
     "    if (chargements.has(h)) attentes.push(chargements.get(h));\n    else if (!l.sheet) {", "    if (false) {"),
    ("C9 : une feuille échouée est prise pour chargée", "feuille_echouee", "    else if (!l.sheet) {", "    else if (false) {"),
    ("etat-05 : le retour n'attend plus l'enregistrement de l'entrée", "retour_avant_reponse", "    if (k && attentes.has(k)) {", "    if (false) {"),
    ("etat-05 : les corrections ne sont plus écrites en sessionStorage", "correction_rechargement",
     "    if (c.size) sessionStorage.setItem(CORRECTIONS, JSON.stringify(Object.fromEntries(c)));", "    if (false) sessionStorage.setItem(CORRECTIONS, '');"),
    ("etat-05 : le démarrage n'applique plus la correction de son entrée", "correction_rechargement", "  appliquerCorrection({ memePage: true });\n", ""),
    ("etat-06 : pageshow persisted ne remonte plus l'écran", "bfcache", "    if (ev && ev.persisted && courant) {", "    if (false) {"),
    ("deploiement-02 : version changée → navigation ordinaire AVANT de savoir si le réseau est là (le code d'avant)", "version_horsligne",
     "  const quitter = versionChangee || !courant;", "  if (versionChangee || !courant) { ordinaire(u, remplace); return; }\n  const quitter = false;"),
    ("deploiement-03 : plus de copie des pages lues (réseau absent : le message)", "memoire", "    const copie = lues.get(u.pathname);", "    const copie = null;"),
    ("deploiement-03 : l'ancien message (« pas encore été chargé »)", "memoire",
     "\"Pas de réseau : cet écran ne peut pas s'ouvrir sans connexion ; l'écran ouvert reste utilisable\"",
     "\"Pas de réseau : cet écran n'a pas encore été chargé sur cet appareil ; les calculs déjà ouverts restent utilisables\""),
    ("deploiement-03 : la page du document n'est plus gardée", "memoire_initiale", "      if (t && versionCourante()) lues.set(", "      if (false) lues.set("),
    ("deploiement-04 : un retour qui reste ne recale plus l'historique", "retour_horsligne", "  if (retour) recaler();", "  if (false) recaler();"),
    ("deploiement-04 : le recalage réécrit l'entrée au lieu de revenir dessus (une entrée perdue)", "retour_horsligne",
     "    try { history.go(e.rang - ra); return; } catch (x) { /* repli ci-dessous */ }", ""),
    ("deploiement-05 : plus d'indicateur après 300 ms", "lent",
     "  attente = { n, montree, minuterie: montree ? 0 : setTimeout(() => montrerAttente(n), ATTENTE_MS) };", "  attente = { n, montree, minuterie: 0 };"),
    ("deploiement-05 : la navigation remplacée retire l'indicateur de la suivante", "lent",
     "  if (!attente || (n !== null && attente.n !== n)) return;", "  if (!attente) return;"),
    ("deploiement-05 : la lecture n'a plus de délai", "lent", "  const minuterie = setTimeout(() => ac.abort(), LECTURE_MS);", "  const minuterie = 0;"),
    ("refus anonyme : les messages de la fermeture sans le titre de l'écran quitté", "refus",
     "toast(quitte ? `« ${quitte} » : ${t}` : t, true);", "toast(t, true);"),
    ("contrat : enCours dit toujours non", "contrat_enregistrements", "    enCours: (id) => enregistrements.has(id),", "    enCours: (id) => false,"),
    ("contrat : apresEnregistrements n'attend rien", "contrat_enregistrements",
     "    apresEnregistrements: () => Promise.all([...enVol]).then(() => undefined, () => undefined),", "    apresEnregistrements: () => Promise.resolve(),"),
    ("contrat : apresEnregistrements rejetée par un enregistrement en échec", "contrat_enregistrements",
     "    const p = Promise.resolve(f.enregistree).catch(() => ({ refus: [] }));", "    const p = Promise.resolve(f.enregistree);\n    p.catch(() => {});"),
    ("contrat : rendreDefilement ne l'oublie plus (rendu deux fois)", "contrat_defilement", "  d.oublier();\n  if (courant !== etat || etat.ferme) return null;",
     "  if (courant !== etat || etat.ferme) return null;"),
    ("contrat : plus de défilement rendu par l'application à 1,5 s", "contrat_defilement",
     "  const minuterie = setTimeout(() => rendreDefilement(etat), DEFILEMENT_MS);", "  const minuterie = 0;"),
    ("contrat : l'accueil attend 1,5 s comme un établi", "contrat_defilement",
     "  if (etat.sorte === \"accueil\") { requestAnimationFrame(() => { if (courant === etat) scrollTo(0, y); }); return; }\n", ""),
    ("contrat : garderEtatEntree sans contrôle du propriétaire", "contrat_etat_entree",
     "  if (!o || typeof o !== \"object\" || proprietaires.get(etat.cle) !== etat.m) return false;", "  if (!o || typeof o !== \"object\") return false;"),
    ("contrat : etatEntree jamais lu", "contrat_etat_entree", "    etatEntree: { ...ecran },", "    etatEntree: {},"),
    ("contrat : la clé n'est plus annoncée au moteur", "contrat_annonce", "    try { if (k) moteur.annoncer(k); } catch (e) { console.error(e); }", ""),
]
for _nom, _s, _avant, _apres in MUTATIONS_BANC:
    MUTATIONS.append((f"banc · {_nom}", (lambda s=_s, a=_avant, b=_apres: not juger_banc(s, copie_banc([("app.js", a, b)]))[0])))

# ------------------------------------------------------------------ 10. les mutations
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
