// L'APPLICATION (lot S4, docs/site/REFONTE_ETABLI.md, révision du 10/11/2026 au soir : S4-D1 à S4-D7 et C1 à C12) : le
// SEUL script de module des pages. Chaque adresse garde son fichier <page>.html (un lien direct, un F5, un partage
// marchent comme avant) ; un clic sur un lien du site ne recharge plus le document : app.js lit la page cible, ferme
// l'écran courant, synchronise les feuilles de style, échange le <body>, pousse l'adresse et monte l'écran de la cible.
// Le moteur Python (js/moteur.js, moteurPartage) reste démarré d'un écran à l'autre.
//
// Les écrans (js/etabli.js, js/accueil.js, js/bibliotheque.js) n'ont aucun effet à leur chargement ; ils rendent
// { fermer } de façon synchrone et ne touchent jamais à l'historique, à l'adresse ni au titre : ils passent par le ctx
// que ce fichier leur donne (adresse, titre, naviguer). C'est ici SEULEMENT que history, location et document.title
// s'écrivent (tests/test_app_navigation.py, contrôle statique).
//
// Tout ce qui n'est pas une route connue, ou dont la lecture dit que la session est à refaire (redirection, 401, 403,
// page sans version) ou que le site a changé (une autre stab-version), part en navigation ORDINAIRE (location.assign,
// ou replace) : rien n'est perdu, la note est dans le navigateur et l'adresse la nomme. Le réseau absent, lui, ne fait
// pas quitter : on reste, avec un message — les calculs déjà ouverts restent utilisables (C8).
import { MODULES, BIBLIOTHEQUES } from "./modules.js";
import { moteurPartage, moteurExiste } from "./moteur.js";
import { cablerRail, toast, retenirToasts, lacherToasts, lireManifeste } from "./commun.js";
import { nouvelleOuverture } from "./sonde.js";
import { SEUIL } from "./compact.js";
import { ouvrirEtabli } from "./etabli.js";
import { ouvrirAccueil } from "./accueil.js";
import { ouvrirBibliotheque } from "./bibliotheque.js";

// le dossier des pages (les pages sont à la racine du site, les scripts dans js/)
export const RACINE = new URL("../", import.meta.url).href;
export const PAS_DE_RESEAU = "Pas de réseau : cet écran n'a pas encore été chargé sur cet appareil ; les calculs déjà ouverts restent utilisables";
export const VU = "stab.moteur.vu";            // localStorage : la version de Pyodide qu'a déjà démarrée ce navigateur
const FRAIS_MS = 30000;                        // une page lue d'avance sert 30 s
const FEUILLES_MS = 3000;                      // les feuilles de la cible : 3 s au plus, sinon navigation ordinaire
// les cibles du préchauffage : tuiles de l'accueil, modules du volet, cartes « Reprendre », modules de « Nouvelle note »
export const CIBLES = "a.tile[data-module], a.tile[data-bibliotheque], .modpanel a.navitem[href], #recentes a.rcard, #nouvelle .mopt";
// le titre principal d'un écran (le focus y va après un échange) : le premier VISIBLE
const TITRES = ["main h1", ".top .crumbs", ".mtop .t", ".mhead .ttl", "main"];

// ------------------------------------------------------------------ les routes (fonction PURE)
// Les pages du site, depuis js/modules.js (le registre) : un module (son établi, ou sa redirection « vers » un autre),
// une bibliothèque, l'accueil.
const PAGES = new Map();
for (const [k, m] of Object.entries(MODULES)) {
  PAGES.set(m.page, m.vers ? { sorte: "redirection", cle: k, vers: MODULES[m.vers].page } : { sorte: "module", cle: k, vers: null });
}
for (const [b, x] of Object.entries(BIBLIOTHEQUES)) PAGES.set(x.page, { sorte: "bibliotheque", cle: b, vers: null });

// {sorte: "accueil"|"module"|"bibliotheque"|"redirection"|null, cle, page, vers} d'une adresse (relative : à la racine)
export function route(url, racine = RACINE) {
  const rien = { sorte: null, cle: null, page: null, vers: null };
  let u, r;
  try { r = new URL(racine); u = new URL(url, r); } catch (e) { return rien; }
  if (u.origin !== r.origin || !u.pathname.startsWith(r.pathname)) return rien;
  let seg = u.pathname.slice(r.pathname.length);
  try { seg = decodeURIComponent(seg); } catch (e) { return rien; }
  if (seg.includes("/")) return rien;
  if (seg.endsWith(".html")) seg = seg.slice(0, -5);
  if (seg === "" || seg === "index") return { sorte: "accueil", cle: null, page: "", vers: null };
  const p = PAGES.get(seg);
  return p ? { sorte: p.sorte, cle: p.cle, page: seg, vers: p.vers } : rien;
}

// ------------------------------------------------------------------ le clic qu'on intercepte (fonction PURE)
// L'adresse (absolue) vers laquelle naviguer DANS l'application, ou null : le navigateur fait ce qu'il fait d'habitude.
// Seulement un clic du bouton principal, sans touche, que personne n'a déjà pris (le ✕ « Supprimer » d'une carte
// Reprendre, une entrée « bientôt »), sur un lien sans target ni download, http(s) de MÊME origine (ni blob:, ni data:, ni
// mailto:), vers une route connue — et pas un simple changement d'ancre.
export function lienInterceptable(ev, a, ici = location.href, racine = RACINE) {
  if (!ev || ev.defaultPrevented) return null;
  if ((ev.button ?? 0) !== 0 || ev.ctrlKey || ev.metaKey || ev.shiftKey || ev.altKey) return null;
  if (!a || typeof a.getAttribute !== "function") return null;
  const href = a.getAttribute("href");
  if (href === null) return null;
  const cible = a.getAttribute("target");
  if (cible && cible !== "_self") return null;
  if (a.hasAttribute("download")) return null;
  const pris = "[data-supprimer], [data-bientot]";
  const t = ev.target && typeof ev.target.closest === "function" ? ev.target : null;
  if ((t && t.closest(pris)) || (typeof a.closest === "function" && a.closest(pris))) return null;
  let u, i;
  try { i = new URL(ici); u = new URL(href, i); } catch (e) { return null; }
  if ((u.protocol !== "http:" && u.protocol !== "https:") || u.origin !== i.origin) return null;
  if (!route(u.href, racine).sorte) return null;
  if (u.hash && u.pathname === i.pathname && u.search === i.search) return null;
  return u.href;
}

// ------------------------------------------------------------------ les feuilles de style (fonction PURE, C9)
// ``actuelles`` : les <link rel=stylesheet> du document, dans l'ordre (adresses absolues, actives ou non) ; ``cibles``
// celles de la page cible. Les feuilles de la cible sont mises À LEUR PLACE, dans son ordre : une nouvelle s'insère juste
// après la cible qui la précède (en tête s'il n'y en a pas) ; une feuille déjà là, mais avant la précédente, est déplacée.
// Rend les opérations dans l'ordre, puis celles à activer (la cible) et à désactiver (les autres), et l'ordre obtenu.
export function operationsFeuilles(actuelles, cibles) {
  const ordre = actuelles.slice();
  const operations = [];
  let prec = null;
  for (const h of cibles) {
    const i = ordre.indexOf(h);
    const j = prec === null ? -1 : ordre.indexOf(prec);
    if (i === -1) {
      ordre.splice(j + 1, 0, h);
      operations.push({ op: "ajouter", href: h, apres: prec });
    } else if (i < j) {
      ordre.splice(i, 1);
      ordre.splice(ordre.indexOf(prec) + 1, 0, h);
      operations.push({ op: "deplacer", href: h, apres: prec });
    }
    prec = h;
  }
  return { operations, activer: cibles.slice(), desactiver: ordre.filter((h) => !cibles.includes(h)), ordre };
}

// ------------------------------------------------------------------ le démarrage du moteur en fond (C6, fonctions PURES)
// Sur l'accueil : page visible, pas d'économie de données ni de 2g/3g déclarées ; là où le navigateur ne dit rien de la
// connexion (Safari), seulement s'il a déjà démarré CE moteur une fois (``vu`` = la version de Pyodide : le cœur est
// alors probablement en cache). Sinon le moteur démarre au premier signe d'intention (le préchauffage).
export function demarrageEnFondPermis({ visible, connexion, vu, pyodide }) {
  if (!visible) return false;
  if (connexion) return !connexion.saveData && !/^(slow-2g|2g|3g)$/.test(String(connexion.effectiveType || ""));
  return !!pyodide && vu === pyodide;
}
// au premier moment de repos (2 s au plus) ; là où requestIdleCallback manque (Safari), 1,5 s
export function planifierFond(f, g = globalThis) {
  if (typeof g.requestIdleCallback === "function") return g.requestIdleCallback(f, { timeout: 2000 });
  return g.setTimeout(f, 1500);
}
// la version de Pyodide d'un manifeste (« vendor/pyodide-0.29.5 » → « 0.29.5 »)
export const versionPyodide = (man) => String((man && man.pyodide) || "").replace(/^.*pyodide-/, "");

// ------------------------------------------------------------------ l'état de l'application
const DOC = Math.random().toString(36).slice(2, 8);   // les clés d'entrée d'historique de CE document
let nCles = 0;
const nouvelleCle = () => `${DOC}.${++nCles}`;
let jeton = 0;                     // la navigation en cours : la dernière gagne
let montages = 0;                  // le jeton de montage (body[data-montage])
let courant = null;                // l'écran monté : { m, sorte, cle, url, abort, ecran, ferme }
let annonce = null;                // la région #annonce (role=status)
let versionChangee = false;        // une mise en ligne vue pendant la session : la prochaine navigation est ordinaire
const corrections = new Map();     // clé d'entrée → l'adresse qu'un écran fermé lui a donnée (C3)
const proprietaires = new Map();   // clé d'entrée → le montage qui l'occupe
const defiles = new Map();         // clé d'entrée → le défilement au départ
const enregistrements = new Map(); // id de note → la promesse de son dernier enregistrement (C1)
const pages = new Map();           // pathname → { t, promesse } (les pages lues, gardées 30 s)
const demandees = new Set();       // les clés déjà préchauffées
const versionCourante = () => (document.querySelector('meta[name="stab-version"]') || {}).content || "";

// LE moteur de l'application, et le drapeau « ce navigateur l'a déjà démarré » posé au premier « pret »
let pretSuivi = null;
function moteurApp() {
  const m = moteurPartage();
  if (m.pret !== pretSuivi) {
    pretSuivi = m.pret;
    m.pret.then(() => lireManifeste()).then((man) => {
      const v = versionPyodide(man);
      if (v) { try { localStorage.setItem(VU, v); } catch (e) { /* stockage bloqué */ } }
    }).catch(() => {});
  }
  return m;
}

// ------------------------------------------------------------------ monter, fermer un écran
const OUVRIR = { accueil: ouvrirAccueil, module: ouvrirEtabli, bibliotheque: ouvrirBibliotheque };

function monter(sorte, url, cle) {
  const m = ++montages;
  const etat = { m, sorte, cle, url, abort: new AbortController(), ecran: null, ferme: false };
  courant = etat;
  proprietaires.set(cle, m);
  const corps = document.body;
  corps.dataset.montage = String(m);
  nouvelleOuverture(m);
  const estCourant = () => courant === etat && !etat.ferme;
  const ctx = {
    jeton: m,
    params: new URL(url).searchParams,
    module: corps.dataset.module,
    bibliotheque: corps.dataset.bibliotheque,
    moteur: sorte === "accueil" ? null : moteurApp(),
    signal: etat.abort.signal,
    adresse: (u, { remplacer = true } = {}) => adresser(etat, u, remplacer),
    titre: (t) => { if (estCourant()) document.title = t; },
    // (jamais pendant le montage lui-même : au tour suivant, si l'écran est toujours le courant)
    naviguer: (u, o) => { if (estCourant()) queueMicrotask(() => { if (estCourant()) naviguer(u, o); }); },
    attendreEnregistrement: (id) => enregistrements.get(id) || Promise.resolve(),
    annoncer: (t) => { if (estCourant()) annoncer(t); },
  };
  etat.ecran = OUVRIR[sorte](corps, ctx);
  if (sorte === "accueil") demarrerEnFond();
  return etat;
}

// fermer l'écran courant (§ 3.1) : (1) le champ actif perd le focus — focusout et change partent, une saisie en cours
// est validée et part au moteur ; (2) fermer() de l'écran (son client garde ses modifications) ; le signal du montage
// est interrompu, les classes de <html> retirées ; (3) on continue aussitôt ; (4) l'enregistrement des modifications
// gardées est suivi par id de note, et ses refus dits dans l'écran suivant.
function fermerCourant() {
  const e = courant;
  if (!e || e.ferme) return;
  retenirToasts();
  try { if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); } catch (x) { /* rien de focalisé */ }
  e.ferme = true;
  let f = null;
  try { f = e.ecran && e.ecran.fermer(); } catch (x) { console.error(x); }
  e.abort.abort();
  document.documentElement.classList.remove("volet-ouvert", "plein-ecran");
  courant = null;
  if (f && f.note && f.enregistree) {
    const p = Promise.resolve(f.enregistree).catch(() => ({ refus: [] }));
    enregistrements.set(f.note, p);
    p.then((res) => {
      if (enregistrements.get(f.note) === p) enregistrements.delete(f.note);
      for (const t of (res && res.refus) || []) toast(t, true);
    });
  }
}

// l'adresse d'un écran (§ 3.3) : sur SA propre entrée d'historique — tout de suite si elle est l'entrée courante, sinon
// au retour sur elle (avant le montage) ; rien si un autre écran l'occupe désormais
function adresser(etat, u, remplacer) {
  let cible;
  try { cible = new URL(u, etat.url).href; } catch (e) { return; }
  if (proprietaires.get(etat.cle) !== etat.m) return;
  const st = history.state;
  if (st && st.cle === etat.cle) {
    try {
      if (remplacer) history.replaceState({ ...st }, "", cible);
      else {
        history.replaceState({ ...st, defile: scrollY }, "");
        const cle = nouvelleCle();
        history.pushState({ cle, defile: 0 }, "", cible);
        proprietaires.set(cle, etat.m);
        etat.cle = cle;
      }
    } catch (e) { return; }
  } else corrections.set(etat.cle, cible);
  etat.url = cible;
}

function annoncer(t) {
  if (!annonce) return;
  const titre = document.title.replace(/\s*·\s*Stab Méthodes$/, "");
  annonce.textContent = t ? `${titre} : ${t}` : titre;
}

function focaliserTitre() {
  const vu = (e) => (typeof e.checkVisibility === "function" ? e.checkVisibility() : !!e.offsetParent);
  for (const s of TITRES) {
    const e = [...document.querySelectorAll(s)].find(vu);
    if (!e) continue;
    if (!e.hasAttribute("tabindex")) e.setAttribute("tabindex", "-1");
    e.dataset.titreEcran = "1";
    try { e.focus({ preventScroll: true }); } catch (x) { /* rien */ }
    return e;
  }
  return null;
}

// ------------------------------------------------------------------ lire une page
function lire(u, priorite = "high") {
  return fetch(u.href, { cache: "no-cache", redirect: "manual", credentials: "same-origin", priority: priorite }).then(async (rep) => {
    // session à refaire (Access : une redirection, 401, 403) ou page absente : la navigation ordinaire le dira
    if (rep.type === "opaqueredirect" || !rep.ok) return { ordinaire: true };
    return { texte: await rep.text() };
  });
}
function garderPage(u, promesse) {
  const k = u.pathname;
  pages.set(k, { t: Date.now(), promesse });
  const oublier = () => { if (pages.get(k) && pages.get(k).promesse === promesse) pages.delete(k); };
  promesse.then((r) => { if (!r.texte) oublier(); }, oublier);
  return promesse;
}
const horsReseau = () => typeof navigator !== "undefined" && navigator.onLine === false;
async function lirePage(u) {
  const c = pages.get(u.pathname);
  if (c && Date.now() - c.t < FRAIS_MS) {
    try { const r = await c.promesse; if (r.texte) return r; } catch (e) { /* relue ci-dessous */ }
  }
  // le navigateur se SAIT hors ligne : on ne tente rien (une lecture vouée à l'échec, et son erreur à la console)
  if (horsReseau()) throw new TypeError("hors ligne");
  return garderPage(u, lire(u));
}
// une lecture qui a échoué : le RÉSEAU manque-t-il (on reste), ou est-ce autre chose (navigation ordinaire) ?
async function horsLigne(e) {
  if (!(e instanceof TypeError)) return false;
  if (horsReseau()) return true;
  try { await fetch(new URL("moteur/manifeste.json", RACINE), { cache: "no-store", redirect: "manual" }); return false; } catch (x) { return true; }
}

// ------------------------------------------------------------------ les feuilles de la cible (§ 3.2)
function preparerFeuilles(doc, u) {
  const sel = 'link[rel~="stylesheet"]';
  const liens = [...document.querySelectorAll(sel)];
  const noeuds = new Map(liens.map((l) => [l.href, l]));
  const medias = new Map();
  const cibles = [...doc.querySelectorAll(sel)].map((l) => {
    const h = new URL(l.getAttribute("href"), u).href;
    medias.set(h, l.getAttribute("media"));
    return h;
  });
  const ops = operationsFeuilles(liens.map((l) => l.href), cibles);
  const attentes = [];
  for (const o of ops.operations) {
    let l = noeuds.get(o.href);
    if (o.op === "ajouter") {
      // elle se télécharge sans s'appliquer
      l = document.createElement("link");
      l.rel = "stylesheet";
      l.media = "not all";
      attentes.push(new Promise((ok) => {
        l.addEventListener("load", () => ok("ok"), { once: true });
        l.addEventListener("error", () => ok("erreur"), { once: true });
      }));
      l.href = o.href;
      noeuds.set(o.href, l);
    }
    const avant = o.apres && noeuds.get(o.apres);
    if (avant) avant.after(l);
    else {
      const premier = document.head.querySelector(sel);
      if (!premier) document.head.append(l); else if (premier !== l) premier.before(l);
    }
  }
  let minuterie = 0;
  const attente = Promise.race([
    Promise.all(attentes).then((x) => (x.includes("erreur") ? "erreur" : "ok")),
    new Promise((ok) => { minuterie = setTimeout(() => ok("delai"), FEUILLES_MS); }),
  ]).finally(() => clearTimeout(minuterie));
  return {
    attente,
    // dans la même tâche que l'échange du corps : la cible active, dans son ordre ; les autres désactivées (gardées : un
    // retour les réactive sans réseau)
    activer() {
      for (const h of ops.activer) {
        const l = noeuds.get(h), m = medias.get(h);
        if (m) l.media = m; else l.removeAttribute("media");
        if (l.sheet) l.sheet.disabled = false; else l.disabled = false;
      }
      for (const h of ops.desactiver) {
        const l = noeuds.get(h);
        if (l) { if (l.sheet) l.sheet.disabled = true; else l.disabled = true; }
      }
    },
  };
}

// le corps de la cible remplace le corps courant : ses attributs, ses enfants (sans ses scripts) ; la région #annonce suit
function echangerCorps(nouveau) {
  const b = document.body;
  for (const a of [...b.attributes]) b.removeAttribute(a.name);
  for (const a of [...nouveau.attributes]) b.setAttribute(a.name, a.value);
  nouveau.querySelectorAll("script").forEach((s) => s.remove());
  b.replaceChildren(...[...nouveau.childNodes].map((x) => document.adoptNode(x)));
  b.append(annonce);
}

function ordinaire(u, remplacer) {
  if (remplacer) location.replace(u.href); else location.assign(u.href);
}

// ------------------------------------------------------------------ naviguer : LA porte (S4-D2)
// ``remplacer`` : l'entrée courante est remplacée (une redirection décidée par un écran, un lien vers l'adresse même) ;
// ``retour`` : un popstate (retour, avance) — l'entrée est déjà la bonne, rien n'est poussé.
export async function naviguer(url, { remplacer = false, retour = false } = {}) {
  const n = ++jeton;
  const yDepart = scrollY;
  let u;
  try { u = new URL(url, location.href); } catch (e) { return; }
  let r = route(u.href);
  // une page de redirection (predalle → dalle) : résolue AVANT d'être lue, l'adresse (requête, ancre) gardée
  if (r.sorte === "redirection") { u = new URL(r.vers + u.search + u.hash, u); r = route(u.href); }
  const remplace = remplacer || retour;
  if (!r.sorte || r.sorte === "redirection" || versionChangee || !courant) { ordinaire(u, remplace); return; }
  let page;
  try { page = await lirePage(u); } catch (e) {
    if (n !== jeton) return;
    if (await horsLigne(e)) { if (n === jeton) toast(PAS_DE_RESEAU, true); return; }
    if (n === jeton) ordinaire(u, remplace);
    return;
  }
  if (n !== jeton) return;
  if (!page.texte) { ordinaire(u, remplace); return; }
  const doc = new DOMParser().parseFromString(page.texte, "text/html");
  const v = doc.querySelector('meta[name="stab-version"]');
  const sorte = doc.body && doc.body.dataset.page;
  // une page sans version n'est pas une page de ce site (une page de connexion rendue en 200) ; une autre version : le
  // site a changé, on le recharge entier (jamais un ancien moteur avec un nouveau balisage)
  if (!v || v.content !== versionCourante() || !OUVRIR[sorte]) { ordinaire(u, remplace); return; }
  const feuilles = preparerFeuilles(doc, u);
  const fin = await feuilles.attente;
  if (n !== jeton) return;
  if (fin !== "ok") {
    if (fin === "erreur" && horsReseau()) { toast(PAS_DE_RESEAU, true); return; }
    ordinaire(u, remplace);
    return;
  }
  // ---- à partir d'ici, UNE tâche : fermer, activer les feuilles, échanger le corps, l'adresse, le titre, monter
  let poussee = false;
  try {
    const avant = courant;
    fermerCourant();
    feuilles.activer();
    echangerCorps(doc.body);
    let cle;
    if (retour) {
      cle = history.state && history.state.cle;
      if (!cle) { cle = nouvelleCle(); history.replaceState({ ...(history.state || {}), cle, defile: 0 }, ""); }
    } else {
      if (avant && history.state && history.state.cle === avant.cle) {
        defiles.set(avant.cle, yDepart);
        history.replaceState({ ...history.state, defile: yDepart }, "");
      }
      cle = nouvelleCle();
      if (remplacer) history.replaceState({ cle, defile: 0 }, "", u.href); else history.pushState({ cle, defile: 0 }, "", u.href);
      poussee = true;
    }
    document.title = doc.title;
    monter(sorte, location.href, cle);
    if (retour) {
      const y = defiles.has(cle) ? defiles.get(cle) : (history.state && history.state.defile) || 0;
      requestAnimationFrame(() => scrollTo(0, y));
    } else scrollTo(0, 0);
    focaliserTitre();
    setTimeout(() => { if (montages === Number(document.body.dataset.montage)) annoncer(""); }, 0);
  } catch (e) {
    console.error(e);
    lacherToasts();
    if (poussee || retour) location.replace(u.href); else location.assign(u.href);
    return;
  }
  for (const [t, erreur] of lacherToasts()) if (erreur) toast(t, true);
}

// ------------------------------------------------------------------ préchauffer (S4-D6, C4)
// Lire la page cible (gardée 30 s) et, au besoin, annoncer sa clé au moteur (une tâche de FOND : js/moteur.js la fait
// passer après toute vraie requête). ``moteur`` faux : la page seule (au doigt, l'appui ne coûte rien au moteur).
export function prechauffer(url, { moteur = true, cle = null } = {}) {
  let u;
  try { u = new URL(url, location.href); } catch (e) { return; }
  let r = route(u.href);
  if (r.sorte === "redirection") { u = new URL(r.vers + u.search, u); r = route(u.href); }
  if (!r.sorte) return;
  const c = pages.get(u.pathname);
  if ((!c || Date.now() - c.t >= FRAIS_MS) && !horsReseau()) garderPage(u, lire(u, "low")).catch(() => {});
  if (!moteur) return;
  const k = cle || (r.sorte === "module" ? r.cle : r.sorte === "bibliotheque" ? `biblio_${r.cle}` : null);
  if (!k || demandees.has(k)) return;
  demandees.add(k);
  moteurApp().prechauffer(k);
}

// la cible de préchauffage sous un élément : { el, url, cle } — la clé : data-module (le module qui le sert), sinon
// biblio_ + data-bibliotheque, sinon celle de la page (route)
function cibleDe(t) {
  if (!t || typeof t.closest !== "function" || t.closest("[data-supprimer], [data-bientot]")) return null;
  const el = t.closest(CIBLES);
  if (!el) return null;
  let cle = null;
  if (el.dataset.module) cle = (MODULES[el.dataset.module] && MODULES[el.dataset.module].vers) || el.dataset.module;
  else if (el.dataset.bibliotheque) cle = `biblio_${el.dataset.bibliotheque}`;
  let url = el.getAttribute("href");
  if (url === null) {
    if (!cle || !MODULES[cle]) return null;
    url = new URL(MODULES[cle].page, RACINE).href;
  }
  return { el, url, cle };
}

// ------------------------------------------------------------------ le démarrage en fond, sur l'accueil (C6)
let fondPrevu = false;
function demarrerEnFond() {
  if (fondPrevu || moteurExiste()) return;
  fondPrevu = true;
  const go = async () => {
    fondPrevu = false;
    if (moteurExiste()) return;
    const connexion = navigator.connection || null;
    let vu = null, pyodide = null;
    if (!connexion) {
      try { vu = localStorage.getItem(VU); } catch (e) { /* stockage bloqué */ }
      try { pyodide = versionPyodide(await lireManifeste()); } catch (e) { /* sans manifeste, pas de démarrage en fond */ }
    }
    if (demarrageEnFondPermis({ visible: document.visibilityState === "visible", connexion, vu, pyodide })) moteurApp();
  };
  const planifier = () => planifierFond(go);
  if (document.readyState === "complete") planifier(); else addEventListener("load", planifier, { once: true });
}

// une mise en ligne pendant la session (C7) : le manifeste relu au retour sur l'onglet ; une autre version → la prochaine
// navigation est ordinaire, et un message discret le dit
function verifierVersion() {
  lireManifeste(true).then((m) => {
    if (m && m.version && m.version !== versionCourante() && !versionChangee) {
      versionChangee = true;
      toast("Le site a été mis à jour : l'écran suivant se chargera en entier.");
    }
  }).catch(() => {});
}

// ------------------------------------------------------------------ le démarrage de l'application
function demarrer() {
  try { history.scrollRestoration = "manual"; } catch (e) { /* navigateur ancien */ }
  cablerRail();
  annonce = document.createElement("div");
  annonce.id = "annonce";
  annonce.className = "sr-only";
  annonce.setAttribute("role", "status");
  annonce.setAttribute("aria-live", "polite");
  document.body.append(annonce);

  // le clic : en BULLE (un écran qui l'a pris l'a dit par preventDefault)
  document.addEventListener("click", (ev) => {
    const c = cibleDe(ev.target);
    const a = ev.target && typeof ev.target.closest === "function" ? ev.target.closest("a[href]") : null;
    const u = a ? lienInterceptable(ev, a) : null;
    if (c && (u || !a)) prechauffer(c.url, { cle: c.cle });
    if (!u) return;
    ev.preventDefault();
    naviguer(u, { remplacer: u === location.href });
  });
  // le préchauffage : le survol d'un pointeur qui n'est pas un doigt ; le focus arrêté 300 ms ; l'appui (au doigt : la
  // page seule ; l'import part au relâcher sans glisser, ou au clic)
  document.addEventListener("pointerover", (ev) => {
    if (ev.pointerType === "touch") return;
    const c = cibleDe(ev.target);
    if (c) prechauffer(c.url, { cle: c.cle });
  });
  let focusMin = 0;
  document.addEventListener("focusin", (ev) => {
    const c = cibleDe(ev.target);
    clearTimeout(focusMin);
    if (!c) return;
    focusMin = setTimeout(() => { if (document.activeElement && c.el.contains(document.activeElement)) prechauffer(c.url, { cle: c.cle }); }, 300);
  });
  let appui = null;
  document.addEventListener("pointerdown", (ev) => {
    const c = cibleDe(ev.target);
    if (!c) { appui = null; return; }
    if (ev.pointerType === "touch") { appui = { id: ev.pointerId, x: ev.clientX, y: ev.clientY, c }; prechauffer(c.url, { moteur: false }); }
    else prechauffer(c.url, { cle: c.cle });
  });
  document.addEventListener("pointerup", (ev) => {
    const a = appui;
    appui = null;
    if (a && ev.pointerId === a.id && Math.hypot(ev.clientX - a.x, ev.clientY - a.y) <= SEUIL.touch) prechauffer(a.c.url, { cle: a.c.cle });
  });
  document.addEventListener("pointercancel", () => { appui = null; });

  // retour, avance : le même chemin, sans pousser ; la correction d'adresse en attente pour cette entrée d'abord (C3)
  addEventListener("popstate", () => {
    const st = history.state;
    if (courant) defiles.set(courant.cle, scrollY);
    if (st && st.cle && courant && st.cle === courant.cle) return;   // (une ancre, la même entrée)
    if (st && st.cle && corrections.has(st.cle)) {
      try { history.replaceState({ ...st }, "", corrections.get(st.cle)); } catch (e) { /* adresse refusée */ }
      corrections.delete(st.cle);
    }
    naviguer(location.href, { retour: true });
  });
  addEventListener("pageshow", verifierVersion);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") verifierVersion(); });

  const corps = document.body;
  // une page de REDIRECTION ouverte directement (predalle) : la page du module qui la sert, l'adresse gardée
  if (corps.dataset.page === "redirection") { location.replace(corps.dataset.vers + location.search + location.hash); return; }
  const st = history.state;
  const cle = st && st.cle ? st.cle : nouvelleCle();
  try { history.replaceState({ ...(st || {}), cle, defile: st && Number.isFinite(st.defile) ? st.defile : 0 }, ""); } catch (e) { /* adresse refusée */ }
  lireManifeste().catch(() => {});
  if (OUVRIR[corps.dataset.page]) monter(corps.dataset.page, location.href, cle);
  // un document rouvert par l'historique (retour vers une page quittée par une navigation ordinaire) : son défilement
  const nav = performance.getEntriesByType && performance.getEntriesByType("navigation")[0];
  if (nav && nav.type === "back_forward" && st && st.defile) requestAnimationFrame(() => scrollTo(0, st.defile));
}

if (typeof document !== "undefined" && document.body) demarrer();
