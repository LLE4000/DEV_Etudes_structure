// L'APPLICATION (lot S4, docs/site/REFONTE_ETABLI.md, révision du 10/11/2026 au soir : S4-D1 à S4-D7 et C1 à C12) : le
// SEUL script de module des pages. Chaque adresse garde son fichier <page>.html (un lien direct, un F5, un partage
// marchent comme avant) ; un clic sur un lien du site ne recharge plus le document : app.js lit la page cible, ferme
// l'écran courant, synchronise les feuilles de style, échange le <body>, pousse l'adresse et monte l'écran de la cible.
// Le moteur Python (js/moteur.js, moteurPartage) reste démarré d'un écran à l'autre.
//
// Les écrans (js/etabli.js, js/accueil.js, js/bibliotheque.js, js/ecran_projets.js, js/ecran_equipe.js, js/ecran_parametres.js) n'ont aucun effet à leur chargement ; ils rendent
// { fermer } de façon synchrone et ne touchent jamais à l'historique, à l'adresse ni au titre : ils passent par le ctx
// que ce fichier leur donne (adresse, titre, naviguer). C'est ici SEULEMENT que history, location et document.title
// s'écrivent (tests/test_app_navigation.py, contrôle statique).
//
// LE CONTRAT ENTRE L'APPLICATION ET LES ÉCRANS : le ctx que monter() donne à ouvrirX(hote, ctx)
//   jeton, params (URLSearchParams de l'adresse), module | bibliotheque, moteur (moteurPartage(), null pour l'accueil),
//   signal (AbortSignal du montage, interrompu à la fermeture) ;
//   adresse(url, {remplacer = true}) : l'adresse de CET écran, sur SA propre entrée d'historique (C3) — tout de suite si
//     elle est l'entrée courante, sinon au retour sur elle (y compris après un rechargement de l'onglet : sessionStorage) ;
//   titre(t), naviguer(url, opts), annoncer(t) : sans effet si l'écran n'est plus le courant ;
//   attendreEnregistrement(id) → Promise : l'enregistrement gardé (C1) de la note id, s'il court ;
//   enCours(id) → vrai si un enregistrement gardé de la note id court encore ;
//   apresEnregistrements() → Promise (jamais rejetée) résolue quand TOUS les enregistrements gardés en cours AU MOMENT
//     DE L'APPEL sont faits (l'accueil s'en sert pour se redessiner) ;
//   rendreDefilement() → rend (scrollTo) le défilement attendu pour cette entrée — un retour ou une avance — UNE fois,
//     puis l'oublie ; rend la valeur, ou null s'il n'y a rien à rendre. L'écran l'appelle après sa première vraie
//     peinture ; s'il ne l'a pas appelé dans les 1,5 s (DEFILEMENT_MS), l'application le rend elle-même (l'accueil : à
//     l'image suivante, comme avant). Un geste de l'utilisateur (molette, doigt, clavier) l'annule : on ne le ramène pas ;
//   etatEntree : l'objet gardé dans l'entrée d'historique de cet écran (lu au montage ; {} par défaut) ;
//   garderEtatEntree(obj) → vrai si obj a été fusionné dans l'état de l'entrée de CET écran (replaceState) : seulement si
//     cette entrée est la courante et que l'écran l'occupe encore (un autre montage sur la même clé l'a prise : refusé).
//   Au montage d'un module ou d'une bibliothèque dont le moteur n'est pas prêt, l'application annonce sa clé au moteur
//   (moteur.annoncer?.(clé) : le module servi, ou biblio_<k>), AVANT que l'écran n'ouvre : le moteur qui démarre prépare
//   ce dont il aura besoin. fermer() rend { note, enregistree } (C1).
//
// Tout ce qui n'est pas une route connue, ou dont la lecture dit que la session est à refaire (redirection, 401, 403,
// page sans version) ou que le site a changé (une autre stab-version, ou une mise en ligne vue pendant la session),
// part en navigation ORDINAIRE (location.assign, ou replace) : rien n'est perdu, la note est dans le navigateur et
// l'adresse la nomme. Le réseau absent ou muet (la lecture n'aboutit pas en 10 s), lui, ne fait JAMAIS quitter (C8) :
// une page déjà lue dans ce document (même version) s'ouvre de sa copie — le moteur porte tous les modules, le calcul
// aboutit réseau coupé (R5) ; sinon on reste, avec un message, et l'historique revient sur l'écran affiché. Une
// lecture qui traîne se voit : après 300 ms, une barre fine en haut de l'écran (.app-attente, site.css) et
// body[aria-busy], retirés quand la navigation finit ou est annulée.
import { MODULES, BIBLIOTHEQUES } from "./modules.js";
import { moteurPartage, moteurExiste } from "./moteur.js";
import { cablerRail, toast, retenirToasts, lacherToasts, lireManifeste } from "./commun.js";
import { nouvelleOuverture } from "./sonde.js";
import { SEUIL } from "./compact.js";
import { ouvrirEtabli } from "./etabli.js";
import { ouvrirAccueil } from "./accueil.js";
import { ouvrirBibliotheque } from "./bibliotheque.js";
import { ouvrirProjets } from "./ecran_projets.js";
import { ouvrirEquipe } from "./ecran_equipe.js";
import { ouvrirParametres } from "./ecran_parametres.js";
import { marquer } from "./marques.js";

// le dossier des pages (les pages sont à la racine du site, les scripts dans js/)
export const RACINE = new URL("../", import.meta.url).href;
// (revue S4, deploiement-03 : l'ancien texte disait « pas encore été chargé sur cet appareil », faux pour un écran déjà
// vu ; un écran ouvert dans ce document se rouvre désormais sans réseau, le message ne vaut que pour les autres)
export const PAS_DE_RESEAU = "Pas de réseau : cet écran ne peut pas s'ouvrir sans connexion ; l'écran ouvert reste utilisable";
export const RESEAU_MUET = "Le réseau ne répond pas : cet écran ne peut pas s'ouvrir pour l'instant ; l'écran ouvert reste utilisable";
export const VU = "stab.moteur.vu";            // localStorage : la version de Pyodide qu'a déjà démarrée ce navigateur
const FRAIS_MS = 30000;                        // une page lue d'avance sert 30 s (ensuite relue ; sa copie reste pour le réseau absent)
const FEUILLES_MS = 3000;                      // les feuilles de la cible : 3 s au plus, sinon navigation ordinaire
export const LECTURE_MS = 10000;               // la lecture d'une page : 10 s au plus, puis traitée comme le réseau absent
const SONDE_MS = 5000;                         // « le réseau est-il là ? » (la lecture du manifeste) : 5 s au plus
export const ATTENTE_MS = 300;                 // l'indicateur de navigation paraît après 300 ms sans réponse
const ENREGISTREMENT_MS = 15000;               // un retour attend l'enregistrement de l'écran fermé sur son entrée : 15 s au plus
export const DEFILEMENT_MS = 1500;             // le défilement d'un retour : rendu par l'écran, sinon par l'application à 1,5 s
const CORRECTIONS = "stab.app.corrections";    // sessionStorage : clé d'entrée → adresse (C3), survit au rechargement de l'onglet
const CORRECTIONS_MAX = 20;
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
// les pages du BUREAU (construire.PAGES_BUREAU : deux copies, liées par tests/test_site_bureau.py) : les projets, l'équipe,
// les paramètres (lot 6 : le gabarit du bureau)
export const PAGES_BUREAU = ["projets", "equipe", "parametres"];
for (const k of PAGES_BUREAU) PAGES.set(k, { sorte: k, cle: null, vers: null });
// les pages du bureau qui DEMANDENT le moteur : les paramètres (il y rend la rubrique Options de chaque module)
export const BUREAU_AVEC_MOTEUR = ["parametres"];
// les écrans qui n'ont PAS besoin du moteur de calcul : ils sont prêts dès qu'ils sont montés
const SANS_MOTEUR = new Set(["accueil", ...PAGES_BUREAU.filter((k) => !BUREAU_AVEC_MOTEUR.includes(k))]);

// {sorte: "accueil"|"projets"|"equipe"|"module"|"bibliotheque"|"redirection"|null, cle, page, vers} d'une adresse
// (relative : à la racine)
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

// L'adresse CANONIQUE d'une page du site (fonction PURE) : sans « .html », l'accueil à la racine — celle que sert
// Cloudflare Pages (une adresse en « .html » y est redirigée, 308, vers celle-ci). Lue telle quelle, une adresse en
// « .html » rendait une redirection, prise pour une session à refaire : le lien partait en navigation ordinaire (le
// document rechargé, le moteur redémarré). Une adresse hors du site, ou illisible, est rendue telle quelle.
export function adresseCanonique(url, racine = RACINE) {
  let u, r;
  try { r = new URL(racine); u = new URL(url, r); } catch (e) { return url; }
  if (u.origin !== r.origin || !u.pathname.startsWith(r.pathname)) return u.href;
  const seg = u.pathname.slice(r.pathname.length);
  if (seg.includes("/") || !seg.endsWith(".html")) return u.href;
  u.pathname = r.pathname + (seg === "index.html" ? "" : seg.slice(0, -5));
  return u.href;
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
let courant = null;                // l'écran monté : { m, sorte, cle, url, rang, abort, ecran, ferme, defile }
let annonce = null;                // la région #annonce (role=status)
let indicateur = null;             // la barre fine d'une navigation qui traîne (.app-attente)
let versionChangee = false;        // une mise en ligne vue pendant la session : la prochaine navigation est ordinaire
let corrections = null;            // clé d'entrée → l'adresse qu'un écran fermé lui a donnée (C3) ; miroir en sessionStorage
const proprietaires = new Map();   // clé d'entrée → le montage qui l'occupe
const defiles = new Map();         // clé d'entrée → le défilement au départ
const enregistrements = new Map(); // id de note → la promesse de son dernier enregistrement (C1)
const enVol = new Set();           // les enregistrements gardés qui courent (ctx.apresEnregistrements)
const attentes = new Map();        // clé d'entrée → l'enregistrement de l'écran fermé qui l'occupait (un retour l'attend)
const pages = new Map();           // pathname → { t, promesse } (les pages lues, gardées 30 s)
const lues = new Map();            // pathname → le texte de la page lue avec succès, même version : pour la vie du document
const chargements = new Map();     // href d'une feuille posée → la promesse de son chargement, tant qu'il court
const demandees = new Set();       // les clés déjà préchauffées
const versionCourante = () => (document.querySelector('meta[name="stab-version"]') || {}).content || "";
// la version écrite dans le texte d'une page (construire.py : <meta name="stab-version" content="…">)
const versionDe = (texte) => { const m = /<meta\s+name="stab-version"\s+content="([^"]*)"/.exec(texte); return m ? m[1] : null; };
const rangDe = (st) => (st && Number.isFinite(st.rang) ? st.rang : null);
// le module SERVI par une clé (predalle → dalle)
const cleServie = (k) => (MODULES[k] && MODULES[k].vers) || k;
// le titre de l'écran affiché, sans le nom du site
const titreEcran = () => String(document.title || "").replace(/\s*·\s*Stab Méthodes$/, "").trim();

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

// ------------------------------------------------------------------ les corrections d'adresse (C3)
// L'adresse qu'un écran FERMÉ donne à son entrée d'historique (une note provisoire écrite après le départ) : notée par
// clé d'entrée, appliquée quand l'entrée redevient la courante — un retour (popstate), ou le démarrage d'un document
// qui la rouvre après un rechargement de l'onglet (d'où le miroir en sessionStorage : la Map seule mourait avec le
// document, revue S4 etat-05).
function lesCorrections() {
  if (!corrections) {
    corrections = new Map();
    try {
      const brut = JSON.parse(sessionStorage.getItem(CORRECTIONS) || "{}");
      for (const [k, v] of Object.entries(brut && typeof brut === "object" ? brut : {})) if (typeof v === "string") corrections.set(k, v);
    } catch (e) { /* stockage bloqué ou illisible : la mémoire seule */ }
  }
  return corrections;
}
function ecrireCorrections() {
  const c = lesCorrections();
  while (c.size > CORRECTIONS_MAX) c.delete(c.keys().next().value);
  try {
    if (c.size) sessionStorage.setItem(CORRECTIONS, JSON.stringify(Object.fromEntries(c)));
    else sessionStorage.removeItem(CORRECTIONS);
  } catch (e) { /* stockage bloqué : la correction vit en mémoire seulement */ }
}
function poserCorrection(cle, url) {
  const c = lesCorrections();
  c.delete(cle);
  c.set(cle, url);
  ecrireCorrections();
}
// la correction en attente pour l'entrée COURANTE, appliquée (replaceState) puis oubliée ; ``memePage`` : seulement si
// elle nomme la page de ce document (au démarrage : le document est déjà celui de l'entrée)
function appliquerCorrection({ memePage = false } = {}) {
  const st = history.state, c = lesCorrections();
  if (!st || !st.cle || !c.has(st.cle)) return;
  const url = c.get(st.cle);
  c.delete(st.cle);
  ecrireCorrections();
  try {
    if (memePage && new URL(adresseCanonique(url)).pathname !== new URL(adresseCanonique(location.href)).pathname) return;
    history.replaceState({ ...st }, "", url);
  } catch (e) { /* adresse refusée */ }
}

// ------------------------------------------------------------------ monter, fermer un écran
const OUVRIR = { accueil: ouvrirAccueil, projets: ouvrirProjets, equipe: ouvrirEquipe, parametres: ouvrirParametres, module: ouvrirEtabli,
  bibliotheque: ouvrirBibliotheque };

// ``defile`` : le défilement à rendre (un retour, une avance), sinon null
function monter(sorte, url, cle, defile = null) {
  const m = ++montages;
  const st = history.state;
  const etat = { m, sorte, cle, url, rang: rangDe(st), abort: new AbortController(), ecran: null, ferme: false, defile: null };
  courant = etat;
  proprietaires.set(cle, m);
  const corps = document.body;
  corps.dataset.montage = String(m);
  nouvelleOuverture(m);
  const estCourant = () => courant === etat && !etat.ferme;
  const moteur = SANS_MOTEUR.has(sorte) ? null : moteurApp();
  // l'intention, AVANT que l'écran n'ouvre : le moteur qui démarre prépare ce dont CE module aura besoin (numpy, en même
  // temps que le reste ; lot « moteur ») — appel facultatif : un moteur sans annoncer() n'en souffre pas
  if (moteur && moteur.etat !== "pret" && typeof moteur.annoncer === "function") {
    const k = sorte === "module" ? cleServie(corps.dataset.module) : corps.dataset.bibliotheque ? `biblio_${corps.dataset.bibliotheque}` : null;
    try { if (k) moteur.annoncer(k); } catch (e) { console.error(e); }
  }
  prevoirDefilement(etat, defile);
  const ecran = st && st.ecran && typeof st.ecran === "object" ? st.ecran : {};
  const ctx = {
    jeton: m,
    params: new URL(url).searchParams,
    module: corps.dataset.module,
    bibliotheque: corps.dataset.bibliotheque,
    moteur,
    signal: etat.abort.signal,
    adresse: (u, { remplacer = true } = {}) => adresser(etat, u, remplacer),
    titre: (t) => { if (estCourant()) document.title = t; },
    // (jamais pendant le montage lui-même : au tour suivant, si l'écran est toujours le courant)
    naviguer: (u, o) => { if (estCourant()) queueMicrotask(() => { if (estCourant()) naviguer(u, o); }); },
    attendreEnregistrement: (id) => enregistrements.get(id) || Promise.resolve(),
    enCours: (id) => enregistrements.has(id),
    apresEnregistrements: () => Promise.all([...enVol]).then(() => undefined, () => undefined),
    rendreDefilement: () => rendreDefilement(etat),
    etatEntree: { ...ecran },
    garderEtatEntree: (o) => garderEtatEntree(etat, o),
    annoncer: (t) => { if (estCourant()) annoncer(t); },
  };
  etat.ecran = OUVRIR[sorte](corps, ctx);
  // la pastille du poste et les modules vérifiés, sur toute page (js/marques.js : gardé, jamais un échec de montage)
  marquer(corps);
  if (sorte === "accueil") demarrerEnFond();
  return etat;
}

// le défilement d'un retour : l'accueil le reçoit à l'image suivante (comme avant) ; un établi ou une bibliothèque le
// demande quand son contenu est là (ctx.rendreDefilement), sinon l'application le rend à DEFILEMENT_MS — sauf si
// l'utilisateur a repris la main entre-temps (revue S4, ecrans-retour-etabli-telephone : à l'image suivante, le contenu
// d'un établi au téléphone n'existe pas encore, et le défilement retombait à 0)
function prevoirDefilement(etat, y) {
  if (y === null || y === undefined) return;
  if (SANS_MOTEUR.has(etat.sorte)) { requestAnimationFrame(() => { if (courant === etat) scrollTo(0, y); }); return; }
  const ac = new AbortController();
  const minuterie = setTimeout(() => rendreDefilement(etat), DEFILEMENT_MS);
  const oublier = () => {
    clearTimeout(minuterie);
    ac.abort();
    if (etat.defile && etat.defile.oublier === oublier) etat.defile = null;
  };
  for (const t of ["wheel", "touchmove", "keydown", "pointerdown"]) addEventListener(t, oublier, { signal: ac.signal, capture: true, passive: true });
  etat.defile = { y, oublier };
}
function rendreDefilement(etat) {
  const d = etat.defile;
  if (!d) return null;
  d.oublier();
  if (courant !== etat || etat.ferme) return null;
  scrollTo(0, d.y);
  return d.y;
}

// l'état que l'écran garde dans SON entrée d'historique (l'onglet du téléphone…) : seulement si l'entrée est la courante
// et qu'il l'occupe encore (un écran remonté sur la même clé — bfcache, retour — l'a prise)
function garderEtatEntree(etat, o) {
  if (!o || typeof o !== "object" || proprietaires.get(etat.cle) !== etat.m) return false;
  const st = history.state;
  if (!st || st.cle !== etat.cle) return false;
  try { history.replaceState({ ...st, ecran: { ...(st.ecran && typeof st.ecran === "object" ? st.ecran : {}), ...o } }, ""); } catch (e) { return false; }
  return true;
}

// fermer l'écran courant (§ 3.1) : (1) le champ actif perd le focus — focusout et change partent, une saisie en cours
// est validée et part au moteur ; (2) fermer() de l'écran (son client garde ses modifications) ; le signal du montage
// est interrompu, les classes de <html> retirées ; (3) on continue aussitôt ; (4) l'enregistrement des modifications
// gardées est suivi par id de note ET par clé d'entrée (un retour sur cette entrée l'attend), et ses refus dits dans
// l'écran suivant (l'écran les a déjà nommés : app.js ne les préfixe pas).
function fermerCourant() {
  const e = courant;
  if (!e || e.ferme) return;
  retenirToasts();
  try { if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); } catch (x) { /* rien de focalisé */ }
  e.ferme = true;
  if (e.defile) e.defile.oublier();
  let f = null;
  try { f = e.ecran && e.ecran.fermer(); } catch (x) { console.error(x); }
  e.abort.abort();
  document.documentElement.classList.remove("volet-ouvert", "plein-ecran");
  courant = null;
  if (f && f.note && f.enregistree) {
    const p = Promise.resolve(f.enregistree).catch(() => ({ refus: [] }));
    const cle = e.cle;
    enregistrements.set(f.note, p);
    enVol.add(p);
    attentes.set(cle, p);
    p.then((res) => {
      enVol.delete(p);
      if (enregistrements.get(f.note) === p) enregistrements.delete(f.note);
      if (attentes.get(cle) === p) attentes.delete(cle);
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
        const rang = rangDe(st) === null ? null : rangDe(st) + 1;
        history.pushState({ cle, defile: 0, rang, ecran: st.ecran }, "", cible);
        proprietaires.set(cle, etat.m);
        etat.cle = cle;
        etat.rang = rang;
      }
    } catch (e) { return; }
  } else poserCorrection(etat.cle, cible);
  etat.url = cible;
}

function annoncer(t) {
  if (!annonce) return;
  const titre = titreEcran();
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

// ------------------------------------------------------------------ l'indicateur d'une navigation qui traîne (deploiement-05)
// Une lecture de page n'allume pas l'indicateur de l'onglet (la navigation ordinaire d'avant, si) : après ATTENTE_MS sans
// réponse, la barre .app-attente (site.css, la même aux trois formats) et body[aria-busy]. Une navigation qui en
// remplace une autre reprend l'indicateur à son compte ; il part quand la DERNIÈRE finit ou est annulée.
let attente = null;   // { n, montree, minuterie }
function commencerAttente(n) {
  const montree = !!(attente && attente.montree);
  if (attente) clearTimeout(attente.minuterie);
  attente = { n, montree, minuterie: montree ? 0 : setTimeout(() => montrerAttente(n), ATTENTE_MS) };
}
function montrerAttente(n) {
  if (!attente || attente.n !== n || n !== jeton || !document.body) return;
  attente.montree = true;
  document.body.setAttribute("aria-busy", "true");
  if (indicateur) document.body.append(indicateur);
}
// ``n`` : seulement si l'indicateur est encore à cette navigation (null : quoi qu'il en soit)
function finirAttente(n = null) {
  if (!attente || (n !== null && attente.n !== n)) return;
  clearTimeout(attente.minuterie);
  attente = null;
  if (document.body) document.body.removeAttribute("aria-busy");
  if (indicateur) indicateur.remove();
}

// ------------------------------------------------------------------ lire une page
// une lecture : LECTURE_MS au plus (AbortController ; l'abandon rejette une AbortError, traitée comme le réseau muet)
function lire(u, priorite = "high") {
  const ac = new AbortController();
  const minuterie = setTimeout(() => ac.abort(), LECTURE_MS);
  return fetch(u.href, { cache: "no-cache", redirect: "manual", credentials: "same-origin", priority: priorite, signal: ac.signal }).then(async (rep) => {
    // session à refaire (Access : une redirection, 401, 403) ou page absente : la navigation ordinaire le dira
    if (rep.type === "opaqueredirect" || !rep.ok) return { ordinaire: true };
    return { texte: await rep.text() };
  }).finally(() => clearTimeout(minuterie));
}
// une page lue (ou en cours de lecture) gardée 30 s ; lue avec succès dans la version de ce document : sa copie reste
// pour la vie du document (le réseau absent l'ouvre de là)
function garderPage(u, promesse) {
  const k = u.pathname;
  pages.set(k, { t: Date.now(), promesse });
  const oublier = () => { if (pages.get(k) && pages.get(k).promesse === promesse) pages.delete(k); };
  promesse.then((r) => {
    if (!r.texte) { oublier(); return; }
    if (versionDe(r.texte) === versionCourante()) lues.set(k, r.texte);
  }, oublier);
  return promesse;
}
const horsReseau = () => typeof navigator !== "undefined" && navigator.onLine === false;
async function lirePage(u) {
  const c = pages.get(u.pathname);
  if (c && Date.now() - c.t < FRAIS_MS) {
    // (une lecture d'avance restée sans réponse LECTURE_MS : le réseau est muet, on ne la recommence pas pour 10 s de plus)
    try { const r = await c.promesse; if (r.texte) return r; } catch (e) { if (e && e.name === "AbortError") throw e; /* sinon relue ci-dessous */ }
  }
  // le navigateur se SAIT hors ligne : on ne tente rien (une lecture vouée à l'échec, et son erreur à la console)
  if (horsReseau()) throw new TypeError("hors ligne");
  return garderPage(u, lire(u));
}
// une lecture qui a échoué : « absent » (le réseau manque : on reste), « muet » (il ne répond pas dans le délai : on
// reste aussi), ou "" (autre chose — une session à refaire, une page absente : la navigation ordinaire le dira)
async function reseauManque(e) {
  if (e && e.name === "AbortError") return "muet";
  if (!(e instanceof TypeError)) return "";
  if (horsReseau()) return "absent";
  const ac = new AbortController(), minuterie = setTimeout(() => ac.abort(), SONDE_MS);
  try {
    await fetch(new URL("moteur/manifeste.json", RACINE), { cache: "no-store", redirect: "manual", signal: ac.signal });
    return "";
  } catch (x) { return x && x.name === "AbortError" ? "muet" : "absent"; } finally { clearTimeout(minuterie); }
}

// ------------------------------------------------------------------ les feuilles de la cible (§ 3.2)
// la promesse du chargement d'une feuille posée (load : « ok », error : « erreur »), gardée tant qu'il court : une
// navigation suivante vers la même cible l'attend aussi (revue S4, etat-08 : au double clic, la seconde navigation
// trouvait la feuille déjà posée, n'attendait rien, et montait l'écran sans elle)
function suivreChargement(l, href) {
  const p = new Promise((ok) => {
    l.addEventListener("load", () => ok("ok"), { once: true });
    l.addEventListener("error", () => ok("erreur"), { once: true });
  });
  chargements.set(href, p);
  p.then(() => { if (chargements.get(href) === p) chargements.delete(href); });
  return p;
}
function nouvelleFeuille(href) {
  // elle se télécharge sans s'appliquer
  const l = document.createElement("link");
  l.rel = "stylesheet";
  l.media = "not all";
  const p = suivreChargement(l, href);
  l.href = href;
  return { l, p };
}
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
  const attentes = [];
  // une feuille de la cible déjà posée : encore en chargement (une navigation précédente vers elle) → on l'attend ; sans
  // contenu et plus en chargement (son chargement a échoué) → une neuve à sa place, attendue
  for (const h of cibles) {
    const l = noeuds.get(h);
    if (!l) continue;
    if (chargements.has(h)) attentes.push(chargements.get(h));
    else if (!l.sheet) {
      const n = nouvelleFeuille(h);
      attentes.push(n.p);
      l.after(n.l);
      l.remove();
      noeuds.set(h, n.l);
    }
  }
  const ops = operationsFeuilles(liens.map((l) => l.href), cibles);
  for (const o of ops.operations) {
    let l = noeuds.get(o.href);
    if (o.op === "ajouter") {
      const n = nouvelleFeuille(o.href);
      l = n.l;
      attentes.push(n.p);
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
    // retour les réactive sans réseau) — une feuille encore en chargement est rendue inerte par son media (« not all »),
    // sans être arrêtée
    activer() {
      for (const h of ops.activer) {
        const l = noeuds.get(h), m = medias.get(h);
        if (m) l.media = m; else l.removeAttribute("media");
        if (l.sheet) l.sheet.disabled = false; else if (l.disabled) l.disabled = false;
      }
      for (const h of ops.desactiver) {
        const l = noeuds.get(h);
        if (l) { if (l.sheet) l.sheet.disabled = true; else l.media = "not all"; }
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

// un retour (ou une avance) qui n'a pas pu ouvrir son écran : le navigateur a déjà changé d'entrée, l'écran affiché est
// resté. L'historique revient sur l'entrée de l'écran affiché (history.go de l'écart des rangs : aucune entrée perdue,
// le Retour se rejoue quand le réseau revient) ; sans rang connu, l'entrée d'arrivée prend l'adresse et la clé de
// l'écran affiché (revue S4, deploiement-04 : l'adresse disait un écran, l'écran en montrait un autre)
function recaler() {
  const e = courant, st = history.state;
  if (!e || !st || st.cle === e.cle) return;
  const ra = rangDe(st);
  if (e.rang !== null && ra !== null && e.rang !== ra) {
    try { history.go(e.rang - ra); return; } catch (x) { /* repli ci-dessous */ }
  }
  try { history.replaceState({ ...st, cle: e.cle, defile: scrollY, rang: e.rang ?? ra }, "", e.url); } catch (x) { /* adresse refusée */ }
}
// on reste (C8) : le message, et l'historique recalé si c'était un retour
function rester(texte, retour) {
  toast(texte, true);
  if (retour) recaler();
}

// un enregistrement attendu, ENREGISTREMENT_MS au plus
function plafonner(p, ms) {
  let minuterie = 0;
  return Promise.race([p, new Promise((ok) => { minuterie = setTimeout(ok, ms); })]).finally(() => clearTimeout(minuterie));
}

// ------------------------------------------------------------------ naviguer : LA porte (S4-D2)
// ``remplacer`` : l'entrée courante est remplacée (une redirection décidée par un écran, un lien vers l'adresse même) ;
// ``retour`` : un popstate (retour, avance) ou un écran remonté sur son entrée (bfcache) — l'entrée est déjà la bonne,
// rien n'est poussé.
export async function naviguer(url, { remplacer = false, retour = false } = {}) {
  const n = ++jeton;
  commencerAttente(n);
  try { await aller(n, url, remplacer, retour); } finally { finirAttente(n); }
}
async function aller(n, url, remplacer, retour) {
  const yDepart = scrollY;
  if (retour) {
    // un retour sur l'entrée d'un écran fermé dont l'enregistrement gardé court encore (C1, C3) : on l'attend — la note
    // provisoire écrite après le départ donne alors son adresse à l'entrée — avant de lire l'adresse (revue S4,
    // etat-05 : la note n'était pas rouverte, un calcul neuf prenait sa place)
    const st = history.state, k = st && st.cle;
    if (k && attentes.has(k)) {
      await plafonner(attentes.get(k), ENREGISTREMENT_MS);
      if (n !== jeton) return;
    }
    appliquerCorrection();
    url = location.href;
  }
  let u;
  try { u = new URL(adresseCanonique(new URL(url, location.href).href)); } catch (e) { return; }
  let r = route(u.href);
  // une page de redirection (predalle → dalle) : résolue AVANT d'être lue, l'adresse (requête, ancre) gardée
  if (r.sorte === "redirection") { u = new URL(r.vers + u.search + u.hash, u); r = route(u.href); }
  const remplace = remplacer || retour;
  if (!r.sorte || r.sorte === "redirection") { ordinaire(u, remplace); return; }
  // une mise en ligne vue pendant la session, ou aucun écran monté : la navigation sera ordinaire — mais seulement si le
  // réseau est là (une lecture qui aboutit, jamais la copie fraîche : elle ne dit rien du réseau) ; réseau absent, on
  // reste (revue S4, deploiement-02 : le document qui calculait hors ligne était déchargé)
  const quitter = versionChangee || !courant;
  let page;
  try {
    if (quitter) { if (horsReseau()) throw new TypeError("hors ligne"); page = await lire(u); } else page = await lirePage(u);
  } catch (e) {
    if (n !== jeton) return;
    const manque = await reseauManque(e);
    if (n !== jeton) return;
    if (!manque) { ordinaire(u, remplace); return; }
    // le réseau manque : la page déjà lue dans ce document (même version : mêmes scripts, même moteur) s'ouvre de sa copie
    const copie = lues.get(u.pathname);
    if (!copie) { rester(manque === "muet" ? RESEAU_MUET : PAS_DE_RESEAU, retour); return; }
    page = { texte: copie, copie: true };
  }
  if (n !== jeton) return;
  if (!page.texte || (quitter && !page.copie)) { ordinaire(u, remplace); return; }
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
    // une feuille manque ou tarde : réseau absent ou muet, on reste ; sinon, navigation ordinaire (C9)
    const manque = horsReseau() ? "absent" : await reseauManque(new TypeError("feuille"));
    if (n !== jeton) return;
    if (manque) { rester(manque === "muet" ? RESEAU_MUET : PAS_DE_RESEAU, retour); return; }
    ordinaire(u, remplace);
    return;
  }
  // ---- à partir d'ici, UNE tâche : fermer, activer les feuilles, échanger le corps, l'adresse, le titre, monter
  let poussee = false;
  // le titre de l'écran quitté, lu AVANT l'échange : un message d'erreur dit pendant sa fermeture (une saisie refusée
  // au blur) le porte, sinon il se lisait comme un message de l'écran suivant (revue S4, refus anonyme)
  const quitte = titreEcran();
  try {
    const avant = courant;
    fermerCourant();
    feuilles.activer();
    echangerCorps(doc.body);
    let cle, defile = null;
    if (retour) {
      cle = history.state && history.state.cle;
      if (!cle) { cle = nouvelleCle(); history.replaceState({ ...(history.state || {}), cle, defile: 0 }, ""); }
      defile = defiles.has(cle) ? defiles.get(cle) : (history.state && history.state.defile) || 0;
    } else {
      const st = history.state;
      if (avant && st && st.cle === avant.cle) {
        defiles.set(avant.cle, yDepart);
        history.replaceState({ ...st, defile: yDepart }, "");
      }
      cle = nouvelleCle();
      const ra = rangDe(st);
      if (remplacer) history.replaceState({ cle, defile: 0, rang: ra }, "", u.href);
      else history.pushState({ cle, defile: 0, rang: ra === null ? null : ra + 1 }, "", u.href);
      poussee = true;
    }
    document.title = doc.title;
    monter(sorte, location.href, cle, defile);
    if (!retour) scrollTo(0, 0);
    focaliserTitre();
    setTimeout(() => { if (montages === Number(document.body.dataset.montage)) annoncer(""); }, 0);
  } catch (e) {
    console.error(e);
    lacherToasts();
    if (poussee || retour) location.replace(u.href); else location.assign(u.href);
    return;
  }
  for (const [t, erreur] of lacherToasts()) if (erreur) toast(quitte ? `« ${quitte} » : ${t}` : t, true);
}

// ------------------------------------------------------------------ préchauffer (S4-D6, C4)
// Lire la page cible (gardée 30 s) et, au besoin, annoncer sa clé au moteur (une tâche de FOND : js/moteur.js la fait
// passer après toute vraie requête). ``moteur`` faux : la page seule (au doigt, l'appui ne coûte rien au moteur).
export function prechauffer(url, { moteur = true, cle = null } = {}) {
  let u;
  try { u = new URL(adresseCanonique(new URL(url, location.href).href)); } catch (e) { return; }
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
  if (el.dataset.module) cle = cleServie(el.dataset.module);
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
// navigation est ordinaire (si le réseau est là), et un message discret le dit
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
  const corps = document.body;
  // la page de CE document, telle que le navigateur l'a lue (avant que l'application ou l'écran n'y touchent) : elle se
  // rouvre hors ligne, comme toute page lue dans ce document (deploiement-03)
  if (OUVRIR[corps.dataset.page]) {
    try {
      const t = document.documentElement && document.documentElement.outerHTML;
      if (t && versionCourante()) lues.set(new URL(adresseCanonique(location.href)).pathname, `<!doctype html>\n${t}`);
    } catch (e) { /* sans copie : hors ligne, cet écran ne se rouvrira pas */ }
  }
  try { history.scrollRestoration = "manual"; } catch (e) { /* navigateur ancien */ }
  cablerRail();
  annonce = document.createElement("div");
  annonce.id = "annonce";
  annonce.className = "sr-only";
  annonce.setAttribute("role", "status");
  annonce.setAttribute("aria-live", "polite");
  document.body.append(annonce);
  indicateur = document.createElement("div");
  indicateur.className = "app-attente";
  indicateur.setAttribute("aria-hidden", "true");

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

  // retour, avance : le même chemin, sans pousser (naviguer attend l'enregistrement de l'entrée, applique sa correction)
  addEventListener("popstate", () => {
    const st = history.state;
    if (courant) defiles.set(courant.cle, scrollY);
    if (st && st.cle && courant && st.cle === courant.cle) {
      // la même entrée que l'écran monté (une avance aussitôt après un retour, le recalage d'un retour qui n'a pas pu
      // s'ouvrir) : la navigation en cours n'a plus lieu d'être — sans quoi elle montait l'écran QUITTÉ sur cette entrée
      // (revue S4, etat-04) ; la correction d'adresse que l'écran a posée entre-temps s'applique
      jeton++;
      finirAttente();
      appliquerCorrection();
      return;
    }
    naviguer(location.href, { retour: true });
  });
  // le retour d'un document gardé par le navigateur (bfcache) : l'écran restauré est celui d'il y a longtemps (une note
  // qu'un autre onglet a pu réécrire depuis) ; le moteur est vivant : l'écran est remonté sur son entrée, depuis le
  // stockage à jour (revue S4, etat-06)
  addEventListener("pageshow", (ev) => {
    verifierVersion();
    if (ev && ev.persisted && courant) {
      defiles.set(courant.cle, scrollY);
      naviguer(location.href, { retour: true });
    }
  });
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") verifierVersion(); });

  // une page de REDIRECTION ouverte directement (predalle) : la page du module qui la sert, l'adresse gardée
  if (corps.dataset.page === "redirection") { location.replace(corps.dataset.vers + location.search + location.hash); return; }
  const st = history.state;
  const cle = st && st.cle ? st.cle : nouvelleCle();
  try {
    history.replaceState({ ...(st || {}), cle, defile: st && Number.isFinite(st.defile) ? st.defile : 0, rang: rangDe(st) ?? 0 }, "");
  } catch (e) { /* adresse refusée */ }
  // C3 : l'adresse qu'un écran d'un document précédent a laissée pour CETTE entrée (un rechargement entre-temps)
  appliquerCorrection({ memePage: true });
  lireManifeste().catch(() => {});
  // un document rouvert par l'historique (retour vers une page quittée par une navigation ordinaire) : son défilement
  const nav = performance.getEntriesByType && performance.getEntriesByType("navigation")[0];
  const defile = nav && nav.type === "back_forward" && st && st.defile ? st.defile : null;
  if (OUVRIR[corps.dataset.page]) monter(corps.dataset.page, location.href, cle, defile);
}

if (typeof document !== "undefined" && document.body) demarrer();
