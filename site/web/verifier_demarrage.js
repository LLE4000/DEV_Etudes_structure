// Le filtre « demarrage » du harnais (lot S4, docs/site/REFONTE_ETABLI.md, révision du 10/11/2026 au soir, S4-D5 à
// S4-D7 et C1 à C14) : UN moteur pour toute la session, des écrans qui se montent et se démontent sans recharger le
// document, l'aperçu tout de suite, le préchauffage à l'approche, la fermeture qui ne perd ni n'écrit rien de travers.
// Chaque contrôle se lit dans le VRAI Chromium, aux formats PC (1 440 × 900) et téléphone (390 × 844), un contexte NEUF
// par format.
//
//   node site/web/verifier_demarrage.js <url> <dossier> [sous-filtre[,sous-filtre…]]
//   (ou : node site/web/verifier_site.js <url> <dossier> demarrage[:sous-filtre], qui délègue ici)
//   DEMARRAGE_FORMATS=pc,tel (les deux par défaut)
//
// Sous-filtres : statique, moteur, prechauffage, doigt, longues, apercu, etablis, rouverte, premiere, reouverture, focus,
// fermeture, historique, supprimer, accueil_frais, refus, erreur, lecture, retour_onglet, js, feuilles, ecouteurs,
// horsligne, lent, version, temps, etalons, secours.
// (rouverte, premiere, reouverture, focus : le lot « preuves » de la revue S4 ; accueil_frais, refus, erreur, lecture,
// retour_onglet : le lot « écrans » ; ceux-ci, quand ils touchent la tablette, la jouent aussi — 820 × 1 180, au doigt —
// quand le passage est au PC et que DEMARRAGE_FORMATS ne la nomme pas : DEMARRAGE_FORMATS=pc,tab,tel la joue comme un
// format à part entière.)
//
// Ce que le harnais pose dans CHAQUE document (addInitScript, avant tout script de la page) :
//   - une sentinelle window.__s (un nombre tiré au sort) : un document rechargé en tire un autre ;
//   - un espion du Worker (sous-classe : le setter onmessage ET addEventListener("message") enveloppés) qui consigne
//     les adresses des travailleurs, chaque message envoyé (requête, préchauffage) et reçu (réponse, sonde…), et sait
//     RETARDER la livraison d'une réponse à la page (window.__retarder = {ms, module}) ou RETENIR « pret » et les
//     réponses (cfg.retenir) — la page voit un moteur lent, le travailleur n'en sait rien ;
//   - un PerformanceObserver « longtask » (buffered) ;
//   - le compte des écouteurs VIVANTS sur window, document et les MediaQueryList (addEventListener /
//     removeEventListener enveloppés, « once » et les signaux d'abandon suivis) et des minuteries setInterval ;
//   - l'instant du clic (écouteur en capture sur window, gardé aussi en sessionStorage d'un document à l'autre) ;
//   - la veille des états de l'écran (body[data-ecran], body[data-montage], et l'ancien voile #demarrage pour pouvoir
//     JOUER ce filtre sur le site d'avant : il doit y rougir, pour la raison annoncée — C14, étape 2) ;
//   - le premier instant où l'APERÇU est visible (data-ecran = apercu, résultat ou dessin peint et vu).
// Les requêtes de navigation du cadre principal sont comptées côté node (page.on("request")) : c'est la preuve qu'aucun
// document n'a été rechargé, indépendante de tout ce que la page dit d'elle-même.
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawnSync } = require('child_process');

const BASE = (process.argv[2] || 'http://127.0.0.1:8770').replace(/\/$/, '');
const OUT = process.argv[3] || '.';
const SOUS = (process.argv[4] || '').split(',').map((x) => x.trim()).filter(Boolean);
const FORMATS = (process.env.DEMARRAGE_FORMATS || 'pc,tel').split(',').map((x) => x.trim()).filter(Boolean);
const RACINE = path.resolve(__dirname, '..', '..');
const PYTHON = process.env.PYTHON || '/usr/bin/python3.12';
const joue = (partie) => !SOUS.length || SOUS.includes(partie);
const attendre = (ms) => new Promise((ok) => setTimeout(ok, ms));
const mediane = (xs) => { const s = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b); if (!s.length) return NaN; const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const court = (x, n = 500) => { const t = typeof x === 'string' ? x : JSON.stringify(x); return t && t.length > n ? t.slice(0, n) + '…' : t; };

let ko = 0;
const verif = (nom, cond, info = '') => { console.log(`${cond ? 'OK' : 'KO'}  ${nom}${info ? '  ' + info : ''}`); if (!cond) ko++; };

const FMT = {
  pc: { nom: 'PC', ctx: { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, isMobile: false, hasTouch: false } },
  tel: { nom: 'téléphone', ctx: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true } },
  tab: { nom: 'tablette', ctx: { viewport: { width: 820, height: 1180 }, deviceScaleFactor: 2, isMobile: false, hasTouch: true } },
};

// ------------------------------------------------------------------------------------------ dans chaque document
const INIT = (cfg) => {
  if (window.__installe) return;
  window.__installe = true;
  cfg = cfg || {};
  const abs = () => performance.timeOrigin + performance.now();
  const origAdd = EventTarget.prototype.addEventListener, origRem = EventTarget.prototype.removeEventListener;
  const oTimeout = window.setTimeout.bind(window);
  window.__abs = abs;
  window.__s = Math.random();

  // ---- les tâches longues du fil principal, dès le début du document
  window.__longues = [];
  try {
    new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__longues.push({ d: Math.round(e.duration), t: Math.round(e.startTime) }); })
      .observe({ type: 'longtask', buffered: true });
  } catch (e) { window.__longues = null; }

  // ---- l'instant du clic : en capture sur window, avant tout écouteur de la page ; gardé d'un document à l'autre
  origAdd.call(window, 'click', () => {
    const t = abs();
    window.__tclic = t; window.__yclic = scrollY;
    try { sessionStorage.setItem('__tclic', String(t)); sessionStorage.setItem('__yclic', String(scrollY)); } catch (e) { /* stockage bloqué */ }
  }, true);

  // ---- les écouteurs VIVANTS sur window, document, MediaQueryList ; les minuteries setInterval
  const compte = { window: 0, document: 0, mql: 0 };
  const genre = (t) => (t === window ? 'window' : t === document ? 'document'
    : (typeof MediaQueryList !== 'undefined' && t instanceof MediaQueryList) ? 'mql' : null);
  const capture = (o) => (typeof o === 'boolean' ? o : !!(o && o.capture));
  const reg = new WeakMap();   // cible → Map("type|capture" → Map(écouteur → {fn}))
  EventTarget.prototype.addEventListener = function (type, f, o) {
    const g = genre(this);
    if (!g || !f) return origAdd.call(this, type, f, o);
    const sig = o && typeof o === 'object' ? o.signal : null;
    if (sig && sig.aborted) return origAdd.call(this, type, f, o);   // le navigateur l'ignore
    let m = reg.get(this); if (!m) reg.set(this, (m = new Map()));
    const k = `${type}|${capture(o)}`;
    let s = m.get(k); if (!s) m.set(k, (s = new Map()));
    if (s.has(f)) return origAdd.call(this, type, s.get(f).fn, o);   // un doublon : ignoré par le navigateur
    const e = { fn: f };
    const retirer = () => { if (s.get(f) === e) { s.delete(f); compte[g]--; } };
    if (o && typeof o === 'object' && o.once) e.fn = function (ev) { retirer(); return typeof f === 'function' ? f.call(this, ev) : f.handleEvent(ev); };
    s.set(f, e); compte[g]++;
    if (sig) origAdd.call(sig, 'abort', retirer, { once: true });
    return origAdd.call(this, type, e.fn, o);
  };
  EventTarget.prototype.removeEventListener = function (type, f, o) {
    const m = reg.get(this), s = m && m.get(`${type}|${capture(o)}`), e = s && s.get(f);
    if (e) { s.delete(f); compte[genre(this)]--; return origRem.call(this, type, e.fn, o); }
    return origRem.call(this, type, f, o);
  };
  if (typeof MediaQueryList !== 'undefined') {
    MediaQueryList.prototype.addListener = function (f) { return this.addEventListener('change', f); };
    MediaQueryList.prototype.removeListener = function (f) { return this.removeEventListener('change', f); };
  }
  const minuteries = new Set(), oSI = window.setInterval, oCI = window.clearInterval, oCT = window.clearTimeout;
  window.setInterval = function (...a) { const id = oSI.apply(window, a); minuteries.add(id); return id; };
  window.clearInterval = function (id) { minuteries.delete(id); return oCI.call(window, id); };
  window.clearTimeout = function (id) { minuteries.delete(id); return oCT.call(window, id); };
  window.__vivants = () => ({ ...compte, minuteries: minuteries.size });

  // ---- l'espion du Worker
  const W = window.Worker;
  window.__W = W;
  window.__travailleurs = []; window.__instances = []; window.__journal = [];
  window.__retardees = []; window.__livrees = [];
  const envois = new WeakMap();   // travailleur → Map(id → {module, action})
  const enAttente = [];
  let relache = cfg.retenir ? performance.timeOrigin + cfg.retenir : 0;
  const vider = () => {
    while (enAttente.length && (!relache || abs() >= relache)) {
      const [w, f, ev] = enAttente.shift();
      try { f.call(w, ev); } catch (e) { oTimeout(() => { throw e; }); }
    }
    if (enAttente.length) oTimeout(vider, 50);
  };
  window.__relacher = () => { relache = 0; vider(); };
  const livrer = (w, f, ev) => {
    const d = (ev && ev.data) || {};
    if (enAttente.length || (relache && abs() < relache && ['pret', 'reponse', 'erreur', 'mort'].includes(d.type))) {
      enAttente.push([w, f, ev]);
      if (enAttente.length === 1) oTimeout(vider, 50);
      return undefined;
    }
    const r = window.__retarder;
    if (r && d.type === 'reponse') {
      const env = (envois.get(w) || new Map()).get(d.id);
      if (!r.module || (env && env.module === r.module)) {
        window.__retardees.push({ id: d.id, module: env && env.module, t: abs(), ms: r.ms });
        oTimeout(() => { window.__livrees.push({ id: d.id, module: env && env.module, t: abs() }); f.call(w, ev); }, r.ms);
        return undefined;
      }
    }
    return f.call(w, ev);
  };
  window.Worker = class extends W {
    constructor(url, o) {
      super(url, o);
      window.__travailleurs.push(String(url));
      window.__instances.push(this);
      envois.set(this, new Map());
      origAdd.call(this, 'message', (ev) => {
        const d = ev.data || {};
        const e = { sens: 'recu', type: d.type, t: abs(), id: d.id };
        if (d.type === 'reponse') {
          e.json = d.json;
          const env = envois.get(this).get(d.id);
          if (env) { e.module = env.module; e.action = env.action; }
        } else if (d.type === 'etape') e.texte = d.texte;
        else e.data = d;
        window.__journal.push(e);
      });
    }
    set onmessage(f) { this.__f = f; super.onmessage = f ? (ev) => livrer(this, f, ev) : null; }
    get onmessage() { return this.__f || null; }
    addEventListener(type, f, o) {
      if (type !== 'message' || !f) return super.addEventListener(type, f, o);
      this.__enveloppes = this.__enveloppes || new Map();
      if (!this.__enveloppes.has(f)) this.__enveloppes.set(f, (ev) => livrer(this, typeof f === 'function' ? f : (x) => f.handleEvent(x), ev));
      return super.addEventListener(type, this.__enveloppes.get(f), o);
    }
    removeEventListener(type, f, o) {
      if (type === 'message' && this.__enveloppes && this.__enveloppes.has(f)) return super.removeEventListener(type, this.__enveloppes.get(f), o);
      return super.removeEventListener(type, f, o);
    }
    postMessage(m, ...r) {
      const e = { sens: 'envoye', t: abs() };
      if (m && typeof m.requete === 'string') {
        e.type = 'requete'; e.id = m.id; e.requete = m.requete; e.note = !!m.note;
        try {
          const q = JSON.parse(m.requete);
          e.module = q.module || (q.bibliotheque ? `biblio_${q.bibliotheque}` : '');
          e.action = q.action;
        } catch (x) { e.module = '?'; }
        envois.get(this).set(m.id, { module: e.module, action: e.action });
      } else if (m && typeof m === 'object') { e.type = m.type; e.cle = m.cle; }
      window.__journal.push(e);
      return super.postMessage(m, ...r);
    }
  };

  // ---- la veille des états de l'écran
  window.__prets = []; window.__ecrans = [];
  let dernier = null, ancienVu = false;
  const veiller = () => {
    const b = document.body;
    if (!b) return;
    const e = b.dataset.ecran, mo = b.dataset.montage;
    const cle = `${e}|${mo}`;
    if (cle !== dernier) { dernier = cle; window.__ecrans.push({ t: abs(), ecran: e ?? null, montage: mo ?? null, chemin: location.pathname }); }
    if (e && /^(pret|erreur)$/.test(e)) {
      if (!window.__prets.some((p) => p.signal === 'ecran' && p.montage === (mo ?? null))) {
        window.__prets.push({ t: abs(), montage: mo ?? null, ecran: e, signal: 'ecran', chemin: location.pathname + location.search });
      }
    } else if (!e && !ancienVu) {
      const d = document.getElementById('demarrage');
      if (d && (d.hidden || d.classList.contains('erreur'))) {
        ancienVu = true;
        window.__prets.push({ t: abs(), montage: null, ecran: d.hidden ? 'pret' : 'erreur', signal: 'voile', chemin: location.pathname + location.search });
      }
    }
  };
  new MutationObserver(veiller).observe(document, { attributes: true, childList: true, subtree: true, attributeFilter: ['data-ecran', 'data-montage', 'hidden', 'class'] });

  // ---- ce que l'œil voit : un voile, le bandeau, l'état des contrôles
  const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
  // Un VOILE cache le contenu : au centre de la part visible de chaque cadre de contenu VU (données, résultat, tuiles,
  // tableau d'une bibliothèque), l'élément du dessus doit appartenir à ce cadre. Un calque posé par-dessus (l'ancien
  // #demarrage : absolute, inset 0, z-index 30) y prend la main. La toile (.canvas), calque pleine taille SOUS les
  // panneaux flottants, n'est pas un voile : elle ne cache aucun cadre (correctif après le premier passage sur le
  // nouveau site, où l'ancien critère « tout élément positionné de 90 % de l'écran » la prenait pour un voile).
  window.__voile = () => {
    const L = innerWidth, H = innerHeight, couvrants = new Set();
    const cadres = ['#donnees', '#resultat', '.tiles', '#bib-tableau'].map((s) => document.querySelector(s)).filter(vu);
    for (const c of cadres) {
      const r = c.getBoundingClientRect();
      const x0 = Math.max(0, r.left), x1 = Math.min(L, r.right), y0 = Math.max(0, r.top), y1 = Math.min(H, r.bottom);
      if (x1 - x0 < 20 || y1 - y0 < 20) continue;
      const h = document.elementFromPoint((x0 + x1) / 2, (y0 + y1) / 2);
      if (!h || c.contains(h) || h.contains(c)) continue;
      let e = h;
      for (; e && e !== document.body; e = e.parentElement) {
        const cs = getComputedStyle(e);
        if (cs.position === 'fixed' || cs.position === 'absolute') break;
      }
      couvrants.add(e && e !== document.body ? (e.id ? `#${e.id}` : String(e.className || e.tagName)) : String(h.className || h.tagName));
    }
    const d = document.getElementById('demarrage');
    return { couvrants: [...couvrants], demarrage: !!d, demarrageVu: vu(d) };
  };
  window.__bandeau = () => {
    const tous = [...document.querySelectorAll('.et-demarre')], b = tous.filter(vu);
    const r = b[0] && b[0].getBoundingClientRect();
    return { n: tous.length, vus: b.length, taille: r ? [Math.round(r.width), Math.round(r.height)] : null,
      dedans: !!r && r.top >= -0.5 && r.left >= -0.5 && r.bottom <= innerHeight + 0.5 && r.right <= innerWidth + 0.5,
      texte: b[0] ? b[0].textContent.replace(/\s+/g, ' ').trim() : '', etat: b[0] ? b[0].dataset.etat || '' : '' };
  };
  // (les croix ✕ des fenêtres de calcul n'en sont pas : fermer, c'est naviguer — revue S4, ecrans-volet-croix-inerte)
  window.__controles = () => {
    const S = ':is(input, select, textarea, button, [data-action]):not([data-fermer])';
    const tous = [...document.querySelectorAll(`#donnees ${S}, #volets ${S}, :is(.top, .mtop, .mbar) [data-action]`)];
    const actifs = tous.filter((e) => !e.closest('[inert]'));
    const don = document.getElementById('donnees');
    return { n: tous.length, actifs: actifs.length, exemples: actifs.slice(0, 6).map((e) => e.dataset.chemin || e.dataset.action || e.id || e.tagName.toLowerCase()),
      donneesInerte: !!(don && don.closest('[inert]')), occupe: !!(don && don.closest('[aria-busy="true"]')) };
  };
  window.__notes = () => { try { return JSON.parse(localStorage.getItem('stab.notes.v1') || '{}'); } catch (e) { return null; } };

  // ---- le premier instant où l'aperçu est VU (data-ecran = apercu, et le résultat ou le dessin peints, à l'écran)
  window.__apercu = null;
  const guetter = () => {
    const b = document.body;
    if (!window.__apercu && b && b.dataset.ecran === 'apercu') {
      // au téléphone, un seul cadre est vu (l'onglet Données par défaut) : les données peintes valent aussi
      const r = document.getElementById('resultat'), d = document.getElementById('donnees'), v = [...document.querySelectorAll('#vues svg, .view svg')].find(vu);
      const cadrePeint = (c) => vu(c) && c.getBoundingClientRect().height > 50 && !c.querySelector('.attente') && c.textContent.trim().length > 0;
      const peint = cadrePeint(r) || cadrePeint(d) || (!!v && v.getBoundingClientRect().height > 50);
      if (peint) window.__apercu = { t: performance.now(), abs: abs(), controles: window.__controles(), bandeau: window.__bandeau(), voile: window.__voile(), notes: window.__notes() };
    }
    if (!window.__apercu && performance.now() < 90000) requestAnimationFrame(guetter);
  };
  origAdd.call(document, 'DOMContentLoaded', () => requestAnimationFrame(guetter));
};

// ------------------------------------------------------------------------------------------ côté node
async function ouvrirContexte(b, fmt, cfg = {}) {
  const c = await b.newContext({ ...FMT[fmt].ctx, acceptDownloads: true, serviceWorkers: 'block' });
  await c.addInitScript(INIT, cfg);
  c.requetes = [];   // toutes les requêtes du contexte (les travailleurs compris)
  c.on('request', (r) => c.requetes.push({ t: Date.now(), url: r.url() }));
  return c;
}
async function nouvellePage(c) {
  const pg = await c.newPage();
  pg.navs = []; pg.travailleurs = 0; pg.erreurs = []; pg.telechargements = 0;
  pg.on('request', (r) => { if (r.isNavigationRequest() && r.frame() === pg.mainFrame()) pg.navs.push({ t: Date.now(), url: r.url() }); });
  pg.on('worker', () => { pg.travailleurs++; });
  pg.on('download', () => { pg.telechargements++; });
  pg.on('pageerror', (e) => pg.erreurs.push(e.message));
  pg.on('console', (m) => { if (m.type() === 'error') pg.erreurs.push(m.text().slice(0, 300)); });
  return pg;
}
// une lecture qui survit à un document qui change sous elle
async function lire(pg, fn, arg) {
  for (let i = 0; ; i++) {
    try { return await pg.evaluate(fn, arg); } catch (e) {
      if (i >= 4 || !/context was destroyed|navigation|Target closed|has been closed/i.test(String(e.message))) throw e;
      await attendre(400);
    }
  }
}
const etatEcran = (pg) => lire(pg, () => ({ montage: document.body ? document.body.dataset.montage ?? null : null, ecran: document.body ? document.body.dataset.ecran ?? null : null,
  s: window.__s, chemin: location.pathname, recherche: location.search, journal: (window.__journal || []).length }));
// l'écran est prêt : un NOUVEL écran (data-montage plus grand, ou un autre document) dont data-ecran vaut pret|erreur —
// ou, sur le site d'avant, le voile #demarrage caché (rouge : erreur) ; l'accueil d'avant n'a aucun signal (il est prêt
// une fois chargé). ``chemin`` : la fin attendue de l'adresse (une redirection passe par un écran intermédiaire).
async function pretEcran(pg, avant, { chemin = null, delai = 120000 } = {}) {
  const r = await pg.waitForFunction(([a, ch]) => {
    const b = document.body;
    if (!b) return false;
    if (ch !== null && !(location.pathname === ch || location.pathname.endsWith(ch))) return false;
    const docNeuf = !a || window.__s !== a.s;
    const e = b.dataset.ecran, mo = b.dataset.montage;
    if (!docNeuf && !(mo != null && (a.montage == null || Number(mo) > Number(a.montage)))) return false;
    if (e) return /^(pret|erreur)$/.test(e) ? `ecran:${e}` : false;
    const d = document.getElementById('demarrage');
    if (d) return d.hidden ? 'voile:pret' : d.classList.contains('erreur') ? 'voile:erreur' : false;
    if (b.dataset.page === 'accueil' && document.readyState !== 'loading') return 'accueil:sans-signal';
    return false;
  }, [avant, chemin], { timeout: delai, polling: 50 });
  const signal = await r.jsonValue();
  try { await pg.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 30000 }); } catch (e) { /* le calcul court encore : l'appelant juge */ }
  return signal;
}
const repere = (pg) => ({ t: Date.now(), navs: pg.navs.length, travailleurs: pg.travailleurs, requetes: pg.context().requetes.length });
const depuis = (pg, m) => ({ navs: pg.navs.length - m.navs, travailleurs: pg.travailleurs - m.travailleurs,
  requetes: pg.context().requetes.slice(m.requetes).map((r) => r.url) });
// un lien du site, posé dans la page et cliqué (le chemin d'un vrai clic : la délégation de app.js l'intercepte)
async function cliquerLien(pg, href) {
  await lire(pg, (h) => {
    let a = document.getElementById('__lien');
    if (!a) { a = document.createElement('a'); a.id = '__lien'; a.textContent = 'lien'; a.style.cssText = 'position:fixed;left:4px;top:4px;z-index:2147483647;padding:12px;background:#ff0;color:#000'; }
    a.setAttribute('href', h);
    document.body.appendChild(a);
  }, href);
  await pg.click('#__lien');
}
async function naviguerApp(pg, href, chemin = null) {
  const avant = await etatEcran(pg);
  await cliquerLien(pg, href);
  return pretEcran(pg, avant, { chemin: chemin ?? `/${href.split('?')[0]}`.replace(/\/\.\/?$|\/$/, '/') });
}
// rejoindre un écran DANS l'application, sauf s'il est déjà l'écran courant
async function allerA(pg, href, chemin = null) {
  const ch = chemin ?? `/${href.split('?')[0]}`.replace(/\/\.\/?$|\/$/, '/');
  const l = await etatEcran(pg);
  if (l.chemin === ch && /^https?:/.test(pg.url())) return 'deja';
  // (une page d'erreur du navigateur — après l'essai hors ligne sur un site qui recharge — n'a pas de lien du site :
  //  on repart d'un chargement direct ; ce n'est jamais le geste mesuré)
  if (!/^https?:/.test(pg.url())) { await pg.goto(`${BASE}/${href === './' ? '' : href}`); return pretEcran(pg, null, { chemin: ch }); }
  return naviguerApp(pg, href, ch);
}
const visible = (pg, s) => pg.locator(s).filter({ visible: true }).first();
async function cliquerAccueil(pg) { await visible(pg, 'a[href="./"]').click(); }
// le volet « Modules » du rail (PC) : ouvert, puis le lien du module
async function cliquerVolet(pg, page) {
  await visible(pg, '.nav a.navitem[data-modules]').click();
  await pg.waitForSelector('.modpanel', { state: 'visible', timeout: 5000 });
  await pg.click(`.modpanel a.navitem[href="${page}"]`);
}
// passer d'un établi à un autre DANS l'application : au PC par le volet, au téléphone par l'accueil (lien Accueil, puis
// la tuile). Rend le signal ; ``avant`` : l'état d'avant le DERNIER clic (celui qui ouvre l'établi)
async function passer(pg, fmt, cle, page) {
  if (fmt === 'pc') {
    const avant = await etatEcran(pg);
    await cliquerVolet(pg, page);
    return { signal: await pretEcran(pg, avant, { chemin: `/${page}` }), avant };
  }
  let avant = await etatEcran(pg);
  await cliquerAccueil(pg);
  await pretEcran(pg, avant, { chemin: '/' });
  avant = await etatEcran(pg);
  await pg.click(`.tile[data-module="${cle}"]`);
  return { signal: await pretEcran(pg, avant, { chemin: `/${page}` }), avant };
}
// le journal de l'espion depuis l'indice n
const journal = (pg, n = 0) => lire(pg, (n) => (window.__journal || []).slice(n).map((e) => ({ ...e, json: undefined })), n);
// sys.modules du travailleur (message « sonde_modules », sous la sonde seulement)
async function sondeModules(pg, delai = 8000) {
  const n = await lire(pg, () => {
    const w = (window.__instances || [])[window.__instances.length - 1];
    if (!w) return -1;
    const n = window.__journal.length;
    w.postMessage({ type: 'sonde_modules' });
    return n;
  });
  if (n < 0) return { erreur: 'aucun travailleur dans ce document' };
  try { await pg.waitForFunction((n) => window.__journal.slice(n).some((e) => e.sens === 'recu' && e.type === 'sonde_modules'), n, { timeout: delai }); } catch (e) {
    return { erreur: `aucune réponse « sonde_modules » en ${delai} ms (le travailleur ne la connaît pas)` };
  }
  return lire(pg, (n) => window.__journal.slice(n).find((e) => e.sens === 'recu' && e.type === 'sonde_modules').data, n);
}
// la RÉFÉRENCE : ce que CPython importe pour les mêmes gestes (préchauffages, puis les requêtes que la page a envoyées)
function referenceCPython(ops) {
  const code = [
    'import sys, json, importlib',
    "sys.path.insert(0, '.')",
    'ops = json.load(sys.stdin)',
    'import api_web.routeur as R',
    'for o in ops:',
    "    if 'prechauffer' in o:",
    "        k = o['prechauffer']",
    '        try:',
    "            if k.startswith('biblio_'): importlib.import_module(R.BIBLIOTHEQUES[k[7:]])",
    "            else: importlib.import_module('api_web.' + k)",
    '        except Exception: pass',
    "    elif 'requete' in o:",
    "        R.traiter(o['requete'])",
    "print(json.dumps({'modules': sorted(m for m in sys.modules if m.startswith('api_web.')), 'numpy': 'numpy' in sys.modules}))",
  ].join('\n');
  const r = spawnSync(PYTHON, ['-c', code], { cwd: RACINE, input: JSON.stringify(ops), encoding: 'utf8', timeout: 300000, maxBuffer: 64 << 20,
    env: { ...process.env, TZ: 'Europe/Paris' } });
  if (r.status !== 0) return { erreur: (r.stderr || String(r.error || '')).slice(-400) };
  try { return JSON.parse(r.stdout.trim().split('\n').pop()); } catch (e) { return { erreur: `sortie illisible : ${r.stdout.slice(-200)}` }; }
}
// sys.modules du navigateur contre la référence : aucune CLÉ du registre (api_web.<clé>, biblio_<k>) en plus
function jugerModules(lu, ref, cles, attendu = null) {
  if (lu.erreur) return { ok: false, info: lu.erreur };
  if (!Array.isArray(lu.modules)) return { ok: false, info: `réponse « sonde_modules » sans liste : ${court(lu, 200)}` };
  if (ref.erreur) return { ok: false, info: `référence CPython : ${ref.erreur}` };
  const r = new Set(ref.modules), enPlus = lu.modules.filter((m) => !r.has(m));
  const clesEnPlus = enPlus.filter((m) => cles.has(m.slice('api_web.'.length)));
  const ok = !clesEnPlus.length && (!attendu || lu.modules.includes(`api_web.${attendu}`)) && (!lu.numpy || ref.numpy);
  return { ok, info: court({ clesEnPlus, aidesEnPlus: enPlus.filter((m) => !clesEnPlus.includes(m)), numpy: lu.numpy, numpyRef: ref.numpy,
    attendu: attendu && lu.modules.includes(`api_web.${attendu}`), n: lu.modules.length }, 600) };
}
// la première réponse du module ``cle`` reçue après l'indice n du journal
async function premiereReponse(pg, n, cle) {
  return lire(pg, ([n, cle]) => { const e = (window.__journal || []).slice(n).find((x) => x.sens === 'recu' && x.type === 'reponse' && x.module === cle); return e ? e.json : null; }, [n, cle]);
}
// deux réponses JSON égales ; exceptions NOMMÉES (REFONTE_ETABLI.md, C12) : soudures numérote ses hachures par un
// compteur de processus (« s<n>_ » normalisé) ; garde_corps : les coordonnées SVG diffèrent au dernier chiffre entre
// Pyodide et CPython (1e-9 relatif). Rend null si égales, sinon le chemin de la première différence.
function difference(a, b, tol, ch = '') {
  if (typeof a !== typeof b) return ch || '/';
  if (a && b && typeof a === 'object') {
    if (Array.isArray(a) !== Array.isArray(b)) return ch || '/';
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) { const d = difference(a[k], b[k], tol, `${ch}/${k}`); if (d !== null) return d; }
    return null;
  }
  if (a === b) return null;
  // deux nombres finis à tol relatif ; un nombre non fini (une suite de chiffres mal découpée) doit être égal à l'octet
  const proche = (x, y) => (Number.isFinite(x) && Number.isFinite(y) ? Math.abs(x - y) <= tol * Math.max(Math.abs(x), Math.abs(y)) : x === y);
  if (tol && typeof a === 'number') return proche(a, b) ? null : ch;
  if (tol && typeof a === 'string') {
    // un NOMBRE du texte : pas collé à un mot, un « # » ou un point (une couleur « #6E7480 » n'est pas 6e7480 ;
    // correctif après le premier passage sur le nouveau site) — le découpage est le même des deux côtés
    const re = /(?<![#\w.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])/gi;
    const na = a.match(re) || [], nb = b.match(re) || [];
    if (a.split(re).join('#') !== b.split(re).join('#') || na.length !== nb.length) return ch;
    return na.every((x, i) => proche(Number(x), Number(nb[i]))) ? null : ch;
  }
  return ch || '/';
}
const EXCEPTIONS = { soudures: 'compteur des hachures (s<n>_) normalisé', garde_corps: 'coordonnées SVG à 1e-9 relatif' };
function comparerApercu(cle, json, apercu) {
  if (json === apercu) return { egal: true, exception: null };
  let a, b;
  try { a = JSON.parse(json); b = JSON.parse(apercu); } catch (e) { return { egal: false, diff: `illisible : ${e.message}` }; }
  let d = difference(a, b, 0);
  if (d === null) return { egal: true, exception: null };
  if (cle === 'soudures') { const n = (x) => JSON.parse(JSON.stringify(x).replace(/\bs\d+_/g, 's#_')); a = n(a); b = n(b); d = difference(a, b, 0); }
  if (cle === 'garde_corps') d = difference(a, b, 1e-9);
  return d === null ? { egal: true, exception: EXCEPTIONS[cle] } : { egal: false, diff: d };
}
const valeurNombre = (s) => Number(String(s).replace(/[\s  ]/g, '').replace(',', '.'));
function lireChemin(etat, chemin) {
  let x = etat;
  for (const p of String(chemin).split('/')) { if (x == null) return undefined; x = x[/^\d+$/.test(p) ? Number(p) : p]; }
  return x;
}
// les pages lues d'avance par app.js vieillissent de 31 s (sa copie fraîche vaut 30 s) : la lecture suivante d'une page
// passe par le réseau — ou, réseau absent, par la copie gardée pour la vie du document (sous-filtres historique, horsligne)
const vieillirPages = (pg) => lire(pg, () => {
  if (!window.__vieillir) { const d0 = Date.now.bind(Date); let dec = 0; Date.now = () => d0() + dec; window.__vieillir = () => { dec += 31000; }; }
  window.__vieillir();
});
// un champ numérique visible de l'établi : { chemin, valeur }, et la valeur à y saisir
const champVisible = (pg) => lire(pg, () => {
  const e = [...document.querySelectorAll('#donnees input[data-sorte="nombre"]:not([disabled])')].find((x) => x.checkVisibility());
  return e ? { chemin: e.dataset.chemin, valeur: e.value } : null;
});
const autreValeur = (champ, k) => Math.round(valeurNombre(champ.valeur) * k * 10) / 10 || 55;
// les messages d'app.js, lus dans le module lui-même (déjà chargé par la page : une seule source)
const messagesApp = (pg) => lire(pg, async () => { const A = await import(new URL('js/app.js', document.baseURI).href); return { PAS_DE_RESEAU: A.PAS_DE_RESEAU, RESEAU_MUET: A.RESEAU_MUET || null }; });
// le message affiché (#toast VU), effacé avant un geste
const messageVu = (pg) => lire(pg, () => { const t = document.getElementById('toast'); return t && t.checkVisibility({ visibilityProperty: true, opacityProperty: true }) ? t.textContent : ''; });
const effacerMessage = (pg) => lire(pg, () => { const t = document.getElementById('toast'); if (t) { t.hidden = true; t.textContent = ''; } });
// une note, en mémoire du navigateur, porte cette valeur à ce chemin (lecture directe, sinon dans son texte)
function noteAvecValeur(notes, module, chemin, valeur) {
  return Object.values(notes || {}).find((n) => {
    if (!n || n.module !== module || !n.etat) return false;
    const x = lireChemin(n.etat, chemin);
    return x !== undefined ? valeurNombre(x) === valeur : JSON.stringify(n.etat).includes(String(valeur));
  }) || null;
}

// ------------------------------------------------------------------------------------------ le passage
(async () => {
  const b = await chromium.launch({ env: { ...process.env, LANG: process.env.LANG || 'C.UTF-8' } });
  let MODULES, CLES, ETABLIS, AVEC_NUMPY;
  {
    const c = await b.newContext();
    const pg = await c.newPage();
    await pg.goto(`${BASE}/`);
    MODULES = await pg.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const man = await (await c.request.get(`${BASE}/moteur/manifeste.json`)).json();
    AVEC_NUMPY = new Set(man.avec_numpy || []);
    await c.close();
  }
  CLES = new Set([...Object.keys(MODULES), ...fs.readdirSync(path.join(RACINE, 'api_web')).filter((f) => /^biblio_[a-z]+\.py$/.test(f)).map((f) => f.slice(0, -3))]);
  ETABLIS = Object.keys(MODULES).filter((k) => !MODULES[k].vers);
  const P = (k) => MODULES[k].page;

  // ---------------------------------------------------------------- statique : plus de voile dans aucune page
  if (joue('statique')) {
    const c = await b.newContext();
    const pages = fs.readdirSync(__dirname).filter((f) => f.endsWith('.html')).map((f) => (f === 'index.html' ? '' : f.slice(0, -5)));
    const avec = [];
    for (const p of pages) { const t = await (await c.request.get(`${BASE}/${p}`)).text(); if (/id="demarrage"/.test(t)) avec.push(p || 'index'); }
    verif(`demarrage · statique : aucune des ${pages.length} pages ne porte le voile #demarrage`, pages.length > 0 && !avec.length, avec.length ? `voile dans : ${avec.join(', ')}` : '');
    await c.close();
  }

  // ---------------------------------------------------------------- secours : le filet du démarrage (10/11/2026)
  // (après la mise en ligne du lot S4, une tablette restait sur « Chargement… » sans un mot quand js/app.js ne démarrait
  // pas) : chaque page porte un message que rien ne montre tant que l'application démarre (body[data-montage] le cache),
  // et qu'une animation CSS rend VISIBLE au bout de 12 s si l'application n'a monté aucun écran — sans aucun script.
  // Aux trois formats : (a) un démarrage normal, le message n'est VU ni à 2 s ni à 13,5 s ; (b) js/app.js refusé,
  // il n'est pas VU à 2 s, il l'est à 13,5 s, dans l'écran, avec « Recharger la page » qui mène à la page même.
  if (joue('secours')) {
    // les six tirages en même temps (trois formats × deux cas) : une seule attente de 13,5 s
    const voir = (pg) => pg.evaluate(() => {
      const s = document.querySelector('.secours-demarrage');
      if (!s) return { present: false };
      const r = s.getBoundingClientRect(), a = s.querySelector('a');
      return { present: true, vu: s.checkVisibility({ visibilityProperty: true, opacityProperty: true }) && r.width > 0 && r.height > 0,
        dedans: r.left >= 0 && r.right <= innerWidth + 0.5 && r.top >= 0 && r.bottom <= innerHeight + 0.5,
        lien: a ? { texte: a.textContent.trim(), href: a.href, ici: location.href } : null, montage: document.body.dataset.montage || null };
    });
    const tirage = async (fmt, refuse) => {
      const c = await b.newContext({ ...FMT[fmt].ctx, serviceWorkers: 'block' });
      try {
        if (refuse) await c.route('**/js/app.js', (route) => route.abort());
        const pg = await c.newPage();
        await pg.goto(`${BASE}/poutre`);
        await pg.waitForTimeout(2000);
        const a2 = await voir(pg);
        await pg.waitForTimeout(11500);
        const a13 = await voir(pg);
        if (refuse && a13.vu) await pg.screenshot({ path: path.join(OUT, `demarrage_secours_${fmt}.png`) });
        return { a2, a13 };
      } finally { await c.close(); }
    };
    const formats = [...new Set([...FORMATS, 'tab'])].filter((f) => FMT[f]);
    const jeux = formats.flatMap((fmt) => [false, true].map((refuse) => ({ fmt, refuse })));
    const res = await Promise.allSettled(jeux.map((j) => tirage(j.fmt, j.refuse)));
    jeux.forEach(({ fmt, refuse }, i) => {
      const N = (x) => `demarrage · ${FMT[fmt].nom} · secours : ${x}`;
      const r = res[i];
      if (r.status !== 'fulfilled') {
        verif(N(refuse ? 'le démarrage refusé va au bout' : 'le démarrage normal va au bout'), false, String(r.reason && r.reason.message).split('\n')[0].slice(0, 300));
        return;
      }
      const { a2, a13 } = r.value;
      if (!refuse) {
        verif(N('un démarrage normal ne le montre jamais (ni à 2 s, ni à 13,5 s, l\'écran monté)'),
          a13.present && !a2.vu && !a13.vu && !!a13.montage, JSON.stringify({ a2s: a2.vu, a13s: a13 }));
      } else {
        verif(N('js/app.js refusé : rien à 2 s, puis le message VU à 13,5 s, dans l\'écran, avec « Recharger la page » vers la page même'),
          !a2.vu && a13.vu && a13.dedans && !a13.montage && !!a13.lien && a13.lien.texte === 'Recharger la page' && a13.lien.href === a13.lien.ici,
          JSON.stringify({ a2s: a2.vu, a13s: a13 }));
      }
    });
  }

  for (const fmt of FORMATS) {
    const F = FMT[fmt].nom;
    const N = (x) => `demarrage · ${F} · ${x}`;
    const essai = async (nom, f) => { try { await f(); } catch (x) { verif(N(`${nom} : le parcours va au bout`), false, String(x.message).split('\n')[0].slice(0, 300)); } };
    const c = await ouvrirContexte(b, fmt);

    // ---------------------------------------------------------------- la page de la session (tout DANS l'application)
    const pg = await nouvellePage(c);
    await pg.goto(`${BASE}/?sonde=1`);
    const S0 = await etatEcran(pg), R0 = repere(pg);

    // ---- aucune tâche longue pendant le démarrage en fond du moteur sur l'accueil (C6 ; le moteur part au repos)
    let pretAccueil = false;
    if (joue('longues')) await essai('longues', async () => {
      try { await pg.waitForFunction(() => (window.__journal || []).some((e) => e.sens === 'recu' && e.type === 'pret'), null, { timeout: 45000 }); pretAccueil = true; } catch (e) { /* jugé plus bas */ }
      const L = await lire(pg, () => window.__longues);
      const max = L ? Math.max(0, ...L.map((x) => x.d)) : null;
      verif(N('accueil : le moteur démarre en fond (« pret » reçu sans aucun geste), sans tâche longue de plus de 200 ms sur la page'),
        pretAccueil && L !== null && max <= 200, `pret ${pretAccueil ? 'reçu' : 'jamais reçu en 45 s'} · tâche la plus longue ${max} ms · ${L ? L.length : '?'} tâche(s) longue(s)`);
    });

    // ---- accueil → tuile poutre → volet assemblages → volet poutre → retour arrière : UN moteur
    let resultatAssemblages = null;
    if (joue('moteur') || joue('modules')) {
      await essai('un seul moteur', async () => {
        let avant = await etatEcran(pg);
        const n0 = avant.journal;
        await pg.click('.tile[data-module="poutre"]');
        await pretEcran(pg, avant, { chemin: `/${P('poutre')}` });
        if (joue('modules')) {
          const lu = await sondeModules(pg);
          const env = (await journal(pg, n0)).filter((e) => e.sens === 'envoye');
          const ops = env.map((e) => (e.type === 'prechauffer' ? { prechauffer: e.cle } : e.type === 'requete' ? { requete: e.requete } : null)).filter(Boolean);
          const j = jugerModules(lu, referenceCPython(ops), CLES, 'poutre');
          verif(N('après la poutre : sys.modules du travailleur sans autre api_web.<clé> que ce que CPython importe pour les mêmes requêtes, et sans numpy'),
            j.ok && !lu.numpy, j.info);
        }
        if (!joue('moteur')) return;
        await passer(pg, fmt, 'assemblages', P('assemblages'));
        resultatAssemblages = await lire(pg, () => (document.getElementById('resultat') || {}).innerHTML || null);
        await passer(pg, fmt, 'poutre', P('poutre'));
        // le retour arrière (au téléphone, l'accueil d'abord)
        avant = await etatEcran(pg);
        await pg.goBack({ timeout: 20000 }).catch(() => null);
        if (fmt === 'tel') { await pretEcran(pg, avant, { chemin: '/' }); avant = await etatEcran(pg); await pg.goBack({ timeout: 20000 }).catch(() => null); }
        await pretEcran(pg, avant, { chemin: `/${P('assemblages')}` });
        const apres = await lire(pg, () => ({ s: window.__s, resultat: (document.getElementById('resultat') || {}).innerHTML || null, travailleurs: window.__travailleurs.length,
          reponses: window.__journal.filter((e) => e.sens === 'recu' && e.type === 'reponse').length, sondes: window.__journal.filter((e) => e.sens === 'recu' && e.type === 'sonde').map((e) => e.data),
          sondesPage: (globalThis.__sondes || []).length }));
        const lp = new Set();
        for (const s of apres.sondes) for (const [k, t] of (s && s.marques) || []) if (k === 'loadPyodide_debut') lp.add(Math.round(t));
        const D = depuis(pg, R0);
        const wasm = D.requetes.filter((u) => /pyodide\.asm\.wasm/.test(u)).length;
        verif(N('un seul moteur : accueil → poutre → assemblages → poutre → retour, sans navigation de document, la sentinelle gardée, UN travailleur, UN démarrage de Python'),
          D.navs === 0 && apres.s === S0.s && pg.travailleurs === 1 && apres.travailleurs === 1 && lp.size === 1,
          court({ navigations: D.navs, sentinelle: apres.s === S0.s ? 'gardée' : 'perdue (document rechargé)', travailleurs_vus: pg.travailleurs, travailleurs_document: apres.travailleurs,
            loadPyodide_debut: lp.size, sondes: apres.sondes.length, wasm_demandes: wasm }));
        verif(N("canari : l'espion a vu des réponses du moteur (et des sondes) dans ce document"), apres.reponses > 0 && apres.sondes.length > 0,
          `${apres.reponses} réponse(s), ${apres.sondes.length} sonde(s), __sondes de la page : ${apres.sondesPage}`);
        verif(N("retour arrière : l'écran d'assemblages revient, prêt, son résultat égal à celui de la première visite"),
          !!resultatAssemblages && apres.resultat === resultatAssemblages && D.navs === 0,
          `${(resultatAssemblages || '').length} / ${(apres.resultat || '').length} caractères · navigations ${D.navs}`);
      });
    }

    // ---- la fermeture : saisie puis clic sur un lien, réponse retardée (1 s, puis 7 s) ; retour vers la note (C1, C2, C3)
    if (joue('fermeture')) {
      await essai('fermeture', async () => {
        await allerA(pg, P('poutre'));
        for (const ms of [1000, 7000]) {
          const champ = await lire(pg, () => {
            const e = [...document.querySelectorAll('#donnees input[data-sorte="nombre"]:not([disabled])')].find((x) => x.checkVisibility());
            return e ? { chemin: e.dataset.chemin, valeur: e.value } : null;
          });
          if (!champ) { verif(N(`fermeture (${ms} ms) : un champ numérique visible à saisir`), false); break; }
          const v = Math.round(valeurNombre(champ.valeur) * (ms === 1000 ? 1.1 : 1.2) * 10) / 10 || 55;
          const sel = `#donnees input[data-chemin="${champ.chemin}"]`;
          const avantRecherche = (await etatEcran(pg)).recherche;
          await pg.fill(sel, String(v).replace('.', ','));
          await lire(pg, (ms) => { window.__retarder = { ms, module: 'poutre' }; }, ms);
          const avant = await etatEcran(pg), R = repere(pg);
          await cliquerLien(pg, P('assemblages'));
          await pretEcran(pg, avant, { chemin: `/${P('assemblages')}`, delai: 120000 });
          const l1 = await lire(pg, () => ({ s: window.__s, chemin: location.pathname, recherche: location.search, resultat: (document.getElementById('resultat') || {}).innerHTML || '',
            titre: document.title, tPret: (window.__prets[window.__prets.length - 1] || {}).t, tClic: window.__tclic ?? Number(sessionStorage.getItem('__tclic')), retardees: window.__retardees.length }));
          // la réponse retardée arrive (ou n'arrivera jamais : document rechargé, travailleur arrêté)
          try { await pg.waitForFunction((n) => window.__livrees.length > 0 && window.__livrees.length >= n, l1.retardees, { timeout: ms + 4000 }); } catch (e) { /* jugé plus bas */ }
          await attendre(600);
          const l2 = await lire(pg, () => ({ s: window.__s, recherche: location.search, resultat: (document.getElementById('resultat') || {}).innerHTML || '', titre: document.title,
            livrees: window.__livrees.length, notes: window.__notes() }));
          await lire(pg, () => { window.__retarder = null; });
          const note = noteAvecValeur(l2.notes, 'poutre', champ.chemin, v);
          const D = depuis(pg, R);
          verif(N(`fermeture (${ms} ms) : l'écran suivant a l'adresse d'assemblages, sans ?note= de la poutre, avant et après la réponse retardée`),
            l1.chemin.endsWith(`/${P('assemblages')}`) && !/note=/.test(l1.recherche) && !/note=/.test(l2.recherche) && D.navs === 0,
            court({ recherche: [l1.recherche, l2.recherche], navigations: D.navs, avant: avantRecherche }));
          verif(N(`fermeture (${ms} ms) : #resultat et le titre d'assemblages inchangés quand la réponse de la poutre arrive`),
            l1.s === l2.s && l1.resultat.length > 0 && l1.resultat === l2.resultat && l1.titre === l2.titre && l2.livrees > 0,
            court({ livrees: l2.livrees, resultat: [l1.resultat.length, l2.resultat.length], memeDocument: l1.s === l2.s }));
          verif(N(`fermeture (${ms} ms) : la note de la poutre est en mémoire avec la valeur saisie (${champ.chemin} = ${v})`), !!note && D.navs === 0,
            court({ note: note ? note.id : null, notes: Object.keys(l2.notes || {}).length, navigations: D.navs }));
          if (ms === 7000) verif(N("fermeture (7 s) : l'écran suivant n'attend pas la réponse retardée"), l1.tPret && l1.tClic && l1.tPret - l1.tClic < 6500 && D.navs === 0,
            `prêt ${Math.round(l1.tPret - l1.tClic)} ms après le clic`);
          if (ms === 1000) {
            // C3 : l'adresse de la note provisoire, écrite APRÈS le départ, corrigée sur SA propre entrée d'historique
            const av = await etatEcran(pg), R3 = repere(pg);
            await pg.goBack({ timeout: 20000 }).catch(() => null);
            await pretEcran(pg, av, { chemin: `/${P('poutre')}` });
            const l3 = await lire(pg, (s) => ({ recherche: location.search, valeur: (document.querySelector(s) || {}).value }), sel);
            verif(N('retour vers la poutre quittée : son adresse porte la note écrite après le départ (?note=), et la valeur saisie'),
              !!note && l3.recherche.includes(`note=${encodeURIComponent(note.id)}`) && valeurNombre(l3.valeur) === v && depuis(pg, R3).navs === 0,
              court({ ...l3, note: note && note.id, navigations: depuis(pg, R3).navs }));
          }
          await allerA(pg, P('poutre'));
        }
      });
    }

    // ---- retour et avance, avec le défilement (aller : 0 ; retour : la position d'avant) (C10)
    if (joue('historique')) {
      await essai('historique', async () => {
        await lire(pg, () => {
          const ns = window.__notes() || {}, t = Date.now();
          for (let i = 0; i < 6; i++) ns[`essai-defile-${i}`] = { id: `essai-defile-${i}`, module: 'poutre', nom: `Défilement ${i + 1}`, chantier: 'Essai', dossier: '', indice: 'A', cree: t + i, modifie: t + i, etat: null, resume: {}, vignette: '' };
          localStorage.setItem('stab.notes.v1', JSON.stringify(ns));
        });
        await allerA(pg, P('poutre'));
        await naviguerApp(pg, './', '/');
        const vp = pg.viewportSize();
        let reduit = false;
        if ((await lire(pg, () => document.documentElement.scrollHeight - innerHeight)) < 100) { reduit = true; await pg.setViewportSize({ width: vp.width, height: 520 }); await attendre(200); }
        await lire(pg, () => { const max = document.documentElement.scrollHeight - innerHeight; scrollTo(0, Math.min(400, Math.floor(max / 2))); });
        await attendre(200);
        const R = repere(pg);
        let avant = await etatEcran(pg);
        await pg.click('.tile[data-module="poutre"]');
        await pretEcran(pg, avant, { chemin: `/${P('poutre')}` });
        // la position au moment du clic (Playwright amène la tuile dans l'écran avant de cliquer)
        const aller = await lire(pg, () => ({ y: scrollY, resultat: (document.getElementById('resultat') || {}).innerHTML || '', yclic: window.__yclic ?? Number(sessionStorage.getItem('__yclic')) }));
        const y = aller.yclic;
        avant = await etatEcran(pg);
        await pg.goBack({ timeout: 20000 }).catch(() => null);
        await pretEcran(pg, avant, { chemin: '/' });
        await attendre(300);
        const retour = await lire(pg, () => scrollY);
        avant = await etatEcran(pg);
        await pg.goForward({ timeout: 20000 }).catch(() => null);
        await pretEcran(pg, avant, { chemin: `/${P('poutre')}` });
        await attendre(300);
        const avance = await lire(pg, () => ({ y: scrollY, s: window.__s, resultat: (document.getElementById('resultat') || {}).innerHTML || '' }));
        const D = depuis(pg, R);
        verif(N(`retour et avance : aller défilé à 0, retour à la position d'avant (${y} px), avance à 0 ; aucun document rechargé ; le résultat de la poutre identique`),
          y > 0 && aller.y === 0 && Math.abs(retour - y) <= 2 && avance.y === 0 && D.navs === 0 && avance.s === S0.s && avance.resultat === aller.resultat && aller.resultat.length > 0,
          court({ defile: y, aller: aller.y, retour, avance: avance.y, navigations: D.navs, sentinelle: avance.s === S0.s, resultat: [aller.resultat.length, avance.resultat.length],
            fenetre: reduit ? `${vp.width}×520 le temps du contrôle (l'accueil ne défile pas à ${vp.width}×${vp.height})` : `${vp.width}×${vp.height}` }));
        if (reduit) await pg.setViewportSize(vp);
      });

      // revue S4, etat-04 : Retour puis Avance AUSSITÔT, pendant que la page du Retour se lit — l'écran de l'entrée
      // QUITTÉE ne doit pas être monté sur l'entrée courante
      await essai('retour puis avance', async () => {
        await allerA(pg, './');
        await naviguerApp(pg, P('poutre'));
        const ici = await etatEcran(pg), R = repere(pg);
        await vieillirPages(pg);
        let lues = 0;
        const estAccueil = (u) => new URL(u).pathname === '/';
        const retenir = async (route) => {
          if (!route.request().isNavigationRequest()) { lues++; await attendre(1500); }
          try { await route.continue(); } catch (e) { /* la page est passée à autre chose */ }
        };
        await pg.route(estAccueil, retenir);
        try {
          await pg.goBack({ timeout: 5000 }).catch(() => null);
          await attendre(150);
          await pg.goForward({ timeout: 5000 }).catch(() => null);
          await attendre(2600);
        } finally { await pg.unroute(estAccueil, retenir); }
        const l = await lire(pg, () => ({ page: document.body.dataset.page, module: document.body.dataset.module, montage: document.body.dataset.montage, chemin: location.pathname,
          s: window.__s, busy: document.body.getAttribute('aria-busy') }));
        const D = depuis(pg, R);
        verif(N("retour puis avance pendant la lecture de la page du retour : l'écran de la poutre reste monté sur son entrée (pas l'accueil sur l'adresse de la poutre), sans document rechargé"),
          lues >= 1 && l.page === 'module' && l.module === 'poutre' && l.chemin === `/${P('poutre')}` && l.montage === ici.montage && l.s === ici.s && D.navs === 0 && !l.busy,
          court({ ...l, lectures: lues, montageAvant: ici.montage, navigations: D.navs }));
      });

      // revue S4, etat-06 : le retour d'un document gardé par le navigateur (bfcache, pageshow persisted) — l'écran
      // restauré est remonté depuis le stockage à jour (un autre onglet a renommé la note) et ne l'écrase pas
      await essai('bfcache', async () => {
        await allerA(pg, './');
        await naviguerApp(pg, P('poutre'));   // (toujours un montage neuf : l'essai ne dépend pas de l'écran laissé par le précédent)
        let champ = await champVisible(pg);
        if (!champ) { verif(N('bfcache : un champ numérique visible à saisir'), false); return; }
        let sel = `#donnees input[data-chemin="${champ.chemin}"]`;
        const n0 = (await etatEcran(pg)).journal;
        await pg.fill(sel, String(autreValeur(champ, 1.1)).replace('.', ','));
        await lire(pg, (s) => document.querySelector(s).blur(), sel);
        await pg.waitForFunction((n) => /note=/.test(location.search) && window.__journal.slice(n).some((e) => e.sens === 'recu' && e.type === 'reponse' && e.module === 'poutre'), n0, { timeout: 30000 });
        await attendre(500);
        const id = new URLSearchParams((await etatEcran(pg)).recherche).get('note');
        await lire(pg, (id) => { const ns = window.__notes(); ns[id].nom = 'Renommée ailleurs'; ns[id].modifie = Date.now() + 1000; localStorage.setItem('stab.notes.v1', JSON.stringify(ns)); }, id);
        const avant = await etatEcran(pg), R = repere(pg);
        await lire(pg, () => { window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true })); });
        const remonte = await pretEcran(pg, avant, { chemin: `/${P('poutre')}`, delai: 20000 }).catch(() => null);   // (jamais remonté : jugé ci-dessous)
        const l = await lire(pg, () => ({ noms: [...document.querySelectorAll('[data-nom-note]')].map((n) => n.value), recherche: location.search, montage: document.body.dataset.montage }));
        // une saisie après le retour : la note enregistrée garde le nom donné ailleurs
        champ = await champVisible(pg);
        sel = `#donnees input[data-chemin="${champ.chemin}"]`;
        const n = (await etatEcran(pg)).journal;
        await pg.fill(sel, String(autreValeur(champ, 1.1)).replace('.', ','));
        await lire(pg, (s) => document.querySelector(s).blur(), sel);
        await pg.waitForFunction((n) => window.__journal.slice(n).some((e) => e.sens === 'recu' && e.type === 'reponse' && e.module === 'poutre'), n, { timeout: 30000 }).catch(() => {});
        await attendre(500);
        const nom = await lire(pg, (id) => ((window.__notes() || {})[id] || {}).nom, id);
        verif(N("retour par le bfcache (pageshow persisted) : l'écran est remonté sur la même adresse depuis le stockage à jour (le nom donné ailleurs), et une saisie ne l'écrase pas"),
          Number(l.montage) > Number(avant.montage) && l.noms.includes('Renommée ailleurs') && l.recherche.includes(`note=${id}`) && nom === 'Renommée ailleurs' && depuis(pg, R).navs === 0,
          court({ ...l, remonte, nomEnregistre: nom, montageAvant: avant.montage, navigations: depuis(pg, R).navs }));
      });

      // revue S4, etat-05 (C3) : Retour AVANT la réponse de la modification gardée d'une note PROVISOIRE — l'entrée
      // attend l'enregistrement, prend l'adresse ?note= de la note écrite après le départ, et la note se rouvre
      await essai('retour avant la réponse', async () => {
        await allerA(pg, './');
        let avant = await etatEcran(pg);
        await pg.click('.tile[data-module="poutre"]');   // un calcul ouvert depuis sa tuile : provisoire, sans ?note=
        await pretEcran(pg, avant, { chemin: `/${P('poutre')}` });
        const champ = await champVisible(pg);
        if (!champ) { verif(N('retour avant la réponse : un champ numérique visible à saisir'), false); return; }
        const v = autreValeur(champ, 1.2), sel = `#donnees input[data-chemin="${champ.chemin}"]`;
        const prov = (await etatEcran(pg)).recherche;
        const deja = Object.keys((await lire(pg, () => window.__notes())) || {});
        await pg.fill(sel, String(v).replace('.', ','));
        await lire(pg, () => { window.__retarder = { ms: 3000, module: 'poutre' }; });
        avant = await etatEcran(pg);
        const R = repere(pg);
        await cliquerLien(pg, P('assemblages'));
        await pretEcran(pg, avant, { chemin: `/${P('assemblages')}` });
        const au = await lire(pg, () => ({ retardees: window.__retardees.length, livrees: window.__livrees.length }));
        avant = await etatEcran(pg);
        await pg.goBack({ timeout: 20000 }).catch(() => null);
        try { await pretEcran(pg, avant, { chemin: `/${P('poutre')}`, delai: 30000 }); } finally { await lire(pg, () => { window.__retarder = null; }); }
        const l = await lire(pg, (s) => ({ recherche: location.search, valeur: (document.querySelector(s) || {}).value, notes: window.__notes(), s: window.__s }), sel);
        // la note que l'adresse nomme : NEUVE (la provisoire, écrite après le départ), et sa valeur est la saisie
        const id = new URLSearchParams(l.recherche).get('note'), note = id && (l.notes || {})[id];
        const enregistree = !!note && !!noteAvecValeur({ [id]: note }, 'poutre', champ.chemin, v);   // (même lecture que les autres essais : le chemin, sinon le texte de l'état)
        verif(N(`retour AVANT la réponse de la modification gardée d'une note provisoire : l'entrée attend son enregistrement, prend ?note= et la note se rouvre avec la valeur saisie (${champ.chemin} = ${v})`),
          !/note=/.test(prov) && au.livrees < au.retardees && !!note && !deja.includes(id) && enregistree && valeurNombre(l.valeur) === v
            && l.s === avant.s && depuis(pg, R).navs === 0,
          court({ provisoire: prov, auRetour: au, recherche: l.recherche, valeur: l.valeur, note: id, neuve: !!id && !deja.includes(id), enregistree, navigations: depuis(pg, R).navs }));
      });

      // revue S4, etat-05 (C3) : la réponse arrive après le départ, puis F5 sur l'écran suivant, puis Retour — la
      // correction d'adresse (sessionStorage) rouvre la note dans le document NEUF (une page à part : son propre onglet)
      await essai('rechargement puis retour', async () => {
        const p2 = await nouvellePage(c);
        try {
          await p2.goto(`${BASE}/${P('poutre')}`);
          await pretEcran(p2, null, { chemin: `/${P('poutre')}` });
          const champ = await champVisible(p2);
          if (!champ) { verif(N('rechargement puis retour : un champ numérique visible à saisir'), false); return; }
          const v = autreValeur(champ, 1.3), sel = `#donnees input[data-chemin="${champ.chemin}"]`;
          const deja = Object.keys((await lire(p2, () => window.__notes())) || {});
          await p2.fill(sel, String(v).replace('.', ','));
          await lire(p2, () => { window.__retarder = { ms: 1500, module: 'poutre' }; });
          let avant = await etatEcran(p2);
          await cliquerLien(p2, P('assemblages'));
          await pretEcran(p2, avant, { chemin: `/${P('assemblages')}` });
          await p2.waitForFunction(() => window.__livrees.length > 0, null, { timeout: 20000 }).catch(() => {});
          await attendre(600);
          const stocke = await lire(p2, () => sessionStorage.getItem('stab.app.corrections'));
          await p2.reload();
          await pretEcran(p2, null, { chemin: `/${P('assemblages')}` });
          avant = await etatEcran(p2);
          await p2.goBack({ timeout: 30000 }).catch(() => null);
          await pretEcran(p2, avant, { chemin: `/${P('poutre')}`, delai: 90000 });
          const l = await lire(p2, (s) => ({ recherche: location.search, valeur: (document.querySelector(s) || {}).value, notes: window.__notes() }), sel);
          const id = new URLSearchParams(l.recherche).get('note'), note = id && (l.notes || {})[id];
          const enregistree = !!note && !!noteAvecValeur({ [id]: note }, 'poutre', champ.chemin, v);   // (même lecture que les autres essais : le chemin, sinon le texte de l'état)
          verif(N(`rechargement entre la réponse et le retour : la correction d'adresse gardée (sessionStorage) rouvre la note provisoire écrite après le départ (${champ.chemin} = ${v})`),
            !!note && !deja.includes(id) && enregistree && valeurNombre(l.valeur) === v,
            court({ recherche: l.recherche, valeur: l.valeur, note: id, neuve: !!id && !deja.includes(id), enregistree, stocke }));
        } finally { await p2.close(); }
      });
    }

    // ---- ✕ « Supprimer » d'une carte Reprendre ne navigue pas (et un clic sur la carte, oui : le témoin)
    if (joue('supprimer')) {
      await essai('supprimer', async () => {
        await lire(pg, () => {
          const ns = window.__notes() || {}, t = Date.now();
          for (const id of ['essai-supp', 'essai-temoin']) ns[id] = { id, module: 'poutre', nom: id === 'essai-supp' ? 'À supprimer' : 'Témoin', chantier: 'Essai', dossier: '', indice: 'A', cree: t, modifie: t, etat: null, resume: {}, vignette: '' };
          localStorage.setItem('stab.notes.v1', JSON.stringify(ns));
        });
        await naviguerApp(pg, P('poutre'));
        await naviguerApp(pg, './', '/');
        pg.once('dialog', (d) => d.accept().catch(() => {}));
        const R = repere(pg), s = (await etatEcran(pg)).s;
        await pg.click('#recentes .rcard [data-supprimer="essai-supp"]');
        await attendre(800);
        const l = await lire(pg, () => ({ chemin: location.pathname, s: window.__s, note: !!(window.__notes() || {})['essai-supp'], carte: !!document.querySelector('[data-supprimer="essai-supp"]') }));
        verif(N('accueil : ✕ « Supprimer » d\'une carte Reprendre supprime la note et ne navigue pas'), l.chemin === '/' && l.s === s && !l.note && !l.carte && depuis(pg, R).navs === 0, court(l));
        const avant = await etatEcran(pg);
        await pg.click('#recentes .rcard[href*="essai-temoin"] .t');
        await pretEcran(pg, avant, { chemin: `/${P('poutre')}` });
        verif(N('témoin : un clic sur la carte elle-même ouvre bien sa note (la mesure voit une navigation)'), (await etatEcran(pg)).recherche.includes('note=essai-temoin'));
      });
    }

    // ================================================================ le lot « écrans » de la revue S4 (10/11/2026)
    // accueil_frais, refus, erreur, lecture, retour_onglet. Les essais qui touchent la tablette la jouent aussi (820 × 1 180,
    // au doigt) quand le passage est au PC et que DEMARRAGE_FORMATS ne la nomme pas.
    const variantes = (f) => (f === 'pc' && !FORMATS.includes('tab') ? ['pc', 'tab'] : [f]);
    // une note de la poutre avec un état qui a un TAUX (une section chargée : M et V) ; son résumé est celui qu'un moteur a
    // écrit autrefois (« 41 % », vert) — le premier calcul de l'établi le récrit
    const ETAT_ESSAI = { poutres: [{ nom: 'P1', b: 25, h: 50, sections: [{ nom: 'S1', M_inf: 80, V: 40 }] }], actif: 0 };
    const noteEssai = (id, nom) => ({ id, module: 'poutre', nom, chantier: 'Essai', dossier: '', indice: 'A', cree: Date.now(), modifie: Date.now(),
      etat: ETAT_ESSAI, resume: { taux: 41, niveau: 'p', statut: 'Conforme' }, vignette: '' });
    const semer = (p, n) => lire(p, (n) => { const ns = window.__notes() || {}; ns[n.id] = n; localStorage.setItem('stab.notes.v1', JSON.stringify(ns)); }, n);
    // le taux écrit à la française, par la fonction de l'écran lui-même (une seule source : commun.texteTaux)
    const texteTauxDe = (p, t) => lire(p, async (t) => (await import(new URL('js/commun.js', document.baseURI).href)).texteTaux(t).texte, t);
    // la carte « Reprendre » d'une note, telle que l'œil la voit
    const carteDe = (p, id) => lire(p, (id) => {
      const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const a = [...document.querySelectorAll('#recentes a.rcard')].find((x) => new URL(x.href).searchParams.get('note') === id);
      const n = (window.__notes() || {})[id];
      const note = n ? { taux: n.resume ? n.resume.taux : null, niveau: n.resume ? n.resume.niveau : null } : null;
      if (!a || !vu(a)) return { carte: false, note };
      const ch = a.querySelector('.chip');
      return { carte: true, chip: ch && vu(ch) ? ch.textContent : null, classes: ch ? [...ch.classList] : [], occupe: a.getAttribute('aria-busy'), note };
    }, id);
    // le moteur rend un REFUS à la prochaine requête nommée (window.__refus = {module, action, erreur}) : l'enveloppe du
    // gestionnaire de messages du travailleur (le setter de l'espion la fait passer par sa livraison, retards compris)
    const installerRefus = (p) => lire(p, () => {
      const w = (window.__instances || [])[window.__instances.length - 1];
      const f = w && w.onmessage;
      if (!f || f.__refus) return !!(f && f.__refus);
      window.__refusFaits = 0;
      const g = function (ev) {
        const d = ev && ev.data, r = window.__refus;
        if (r && d && d.type === 'reponse') {
          const env = window.__journal.find((e) => e.sens === 'envoye' && e.type === 'requete' && e.id === d.id);
          if (env && env.module === r.module && env.action === r.action) {
            window.__refus = null;
            window.__refusFaits++;
            return f.call(this, { data: { ...d, json: JSON.stringify({ ok: false, erreur: r.erreur }) } });
          }
        }
        return f.call(this, ev);
      };
      g.__refus = true;
      w.onmessage = g;
      return true;
    });
    // la barre du haut et le bandeau, tels que l'œil les voit : le bandeau VU (taille, dans l'écran), « Recharger » VU, dans
    // l'écran et pas couvert (elementFromPoint), le débordement de côté, et la part VUE du nom de la note contre la largeur
    // de son texte (la mesure de verifier_site.js, ouverture § 2 bis : le champ a 2 × 7 px de marges et de cadre)
    const MESURE_BARRE = () => {
      const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const W = innerWidth, H = innerHeight;
      const bs = [...document.querySelectorAll('.et-demarre')].filter(vu), b0 = bs[0];
      const rb = b0 ? b0.getBoundingClientRect() : null;
      const rech = b0 ? b0.querySelector('[data-recharger]') : null;
      const rr = rech && vu(rech) ? rech.getBoundingClientRect() : null;
      const hit = rr ? document.elementFromPoint(rr.left + rr.width / 2, rr.top + rr.height / 2) : null;
      let nom = null;
      const cr = document.querySelector('.top .crumbs'), n = document.getElementById('nom-note');
      if (cr && n && vu(n)) {
        const r = n.getBoundingClientRect(), c = cr.getBoundingClientRect(), cs = getComputedStyle(n), t = document.createElement('span');
        t.style.cssText = `position:absolute;white-space:pre;font:${cs.font}`; t.textContent = n.value; document.body.appendChild(t);
        const texte = t.getBoundingClientRect().width; t.remove();
        nom = { valeur: n.value, vu: Math.round(Math.min(r.right, c.right) - Math.max(r.left, c.left)), texte: Math.round(texte) };
      }
      return { n: bs.length, etat: b0 ? b0.dataset.etat || '' : null, texte: b0 ? b0.textContent.replace(/\s+/g, ' ').trim() : '',
        taille: rb ? [Math.round(rb.width), Math.round(rb.height)] : null,
        dedans: !!rb && rb.left >= -0.5 && rb.top >= -0.5 && rb.right <= W + 0.5 && rb.bottom <= H + 0.5,
        recharger: rr ? { droite: Math.round(rr.right), bas: Math.round(rr.bottom), taille: [Math.round(rr.width), Math.round(rr.height)],
          dedans: rr.left >= 0 && rr.top >= 0 && rr.right <= W + 0.5 && rr.bottom <= H + 0.5, touche: !!hit && rech.contains(hit) } : null,
        deborde: document.documentElement.scrollWidth > W, largeur: [document.documentElement.scrollWidth, W], nom, ecran: document.body.dataset.ecran };
    };
    // un contrôle, tel que le doigt le trouve : VU, hors de tout [inert], et le point de son centre est à lui
    const ACTIF = (s) => {
      const e = [...document.querySelectorAll(s)].find((x) => x.checkVisibility({ visibilityProperty: true, opacityProperty: true }));
      if (!e) return { vu: false };
      const r = e.getBoundingClientRect(), h = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return { vu: r.width > 0 && r.height > 0, inerte: !!e.closest('[inert]'), touche: !!h && e.contains(h), taille: [Math.round(r.width), Math.round(r.height)] };
    };
    const toucher = (p, v, s) => (FMT[v].ctx.hasTouch ? p.tap(s) : p.click(s));
    // la réponse retardée est TENUE par l'espion (arrivée à la page, pas encore livrée) : au Retour, la requête ne part
    // qu'à la fermeture de l'écran, et sa réponse arrive après le montage de l'accueil — sans cette attente, l'essai
    // jugeait avant qu'elle n'existe (correctif du premier passage)
    const attendreTenue = (p, n0) => p.waitForFunction((n) => window.__retardees.length > n, n0, { timeout: 15000 }).then(() => true, () => false);

    // ---- l'accueil après une fermeture dont la modification gardée court encore (revue S4, etat-01, etat-03 ; C1, C11)
    if (joue('accueil_frais')) {
      await essai('accueil frais', async () => {
        const id = 'essai-frais';
        // la carte pendant l'enregistrement gardé, puis après : par le lien Accueil (le champ perd le focus à l'appui) et par
        // le Retour (popstate : le focus est perdu DANS la fermeture de l'écran, app.js)
        for (const geste of ['lien', 'retour']) {
          await semer(pg, noteEssai(id, 'Essai frais'));
          await allerA(pg, './');
          await naviguerApp(pg, `${P('poutre')}?note=${id}`, `/${P('poutre')}`);
          await attendre(300);
          // la HAUTEUR de la poutre (le taux gouvernant en dépend ; la largeur, le premier champ, ne le change pas ici)
          const champ = await lire(pg, () => {
            const e = [...document.querySelectorAll('#donnees input[data-sorte="nombre"]:not([disabled])')].find((x) => x.checkVisibility() && /(^|\/)h$/.test(x.dataset.chemin));
            return e ? { chemin: e.dataset.chemin, valeur: e.value } : null;
          });
          if (!champ) { verif(N(`accueil frais (${geste}) : le champ de la hauteur, visible`), false); return; }
          const sel = `#donnees input[data-chemin="${champ.chemin}"]`, v = autreValeur(champ, geste === 'lien' ? 0.6 : 0.7);
          // le taux de la note tel que le premier calcul l'a récrit (l'état d'AVANT la saisie)
          const r0 = await lire(pg, (id) => ((window.__notes() || {})[id] || {}).resume || {}, id);
          const ancien = await texteTauxDe(pg, r0.taux);
          if (!ancien) { verif(N(`accueil frais (${geste}) : la note a un taux avant la saisie`), false, court(r0)); return; }
          await pg.fill(sel, String(v).replace('.', ','));
          const n0 = await lire(pg, () => { window.__retarder = { ms: 3000, module: 'poutre' }; return window.__retardees.length; });
          const avant = await etatEcran(pg);
          if (geste === 'lien') await cliquerAccueil(pg); else await pg.goBack({ timeout: 20000 }).catch(() => null);
          await pretEcran(pg, avant, { chemin: '/' });
          const auMontage = await carteDe(pg, id);
          await attendreTenue(pg, n0);
          const pendant = await carteDe(pg, id);
          const vol = await lire(pg, () => ({ retardees: window.__retardees.length, livrees: window.__livrees.length }));
          try { await pg.waitForFunction((n) => window.__livrees.length >= n, vol.retardees, { timeout: 15000 }); } catch (e) { /* jugé plus bas */ }
          await attendre(700);
          await lire(pg, () => { window.__retarder = null; });
          const apres = await carteDe(pg, id);
          const nouveau = apres.note ? await texteTauxDe(pg, apres.note.taux) : '';
          const neutre = (x) => x.carte && x.chip === 'enregistrement en cours' && x.chip !== ancien && !x.classes.some((k) => ['p', 'f', 'v', 'r'].includes(k));
          verif(N(`accueil (${geste === 'lien' ? 'lien Accueil' : 'Retour'}) : pendant l'enregistrement gardé (au montage, et la réponse tenue), la carte de la note est NEUTRE — « enregistrement en cours », ni l'ancien taux (${ancien}) ni sa pastille`),
            vol.retardees > n0 && vol.livrees < vol.retardees && neutre(auMontage) && neutre(pendant),
            court({ auMontage, pendant, vol, n0 }));
          verif(N(`accueil (${geste === 'lien' ? 'lien Accueil' : 'Retour'}) : la réponse livrée, la carte se redessine — le NOUVEAU taux, sa pastille (le geste change le taux : ${ancien} → ${nouveau})`),
            apres.carte && !!nouveau && nouveau !== ancien && apres.chip === nouveau && apres.classes.includes(apres.note.niveau) && apres.occupe === null,
            court({ apres, nouveau, ancien }));
        }
        // supprimer la note sur l'accueil PENDANT son enregistrement gardé : la suppression attend l'enregistrement (aucun
        // refus « supprimée » : l'écriture a eu lieu avant), et la note ne revient pas
        await semer(pg, noteEssai(id, 'Essai frais'));
        await allerA(pg, './');
        await naviguerApp(pg, `${P('poutre')}?note=${id}`, `/${P('poutre')}`);
        let champ = await champVisible(pg);
        await pg.fill(`#donnees input[data-chemin="${champ.chemin}"]`, String(autreValeur(champ, 0.8)).replace('.', ','));
        let vol = await lire(pg, () => { window.__retarder = { ms: 3000, module: 'poutre' }; return window.__retardees.length; });
        let avant = await etatEcran(pg);
        await cliquerAccueil(pg);
        await pretEcran(pg, avant, { chemin: '/' });
        await attendreTenue(pg, vol);
        await effacerMessage(pg);
        pg.once('dialog', (d) => d.accept().catch(() => {}));
        await pg.click(`#recentes .rcard [data-supprimer="${id}"]`);
        const enVol = await lire(pg, (n) => window.__retardees.length > n && window.__livrees.length < window.__retardees.length, vol);
        try { await pg.waitForFunction(() => window.__livrees.length >= window.__retardees.length, null, { timeout: 15000 }); } catch (e) { /* jugé plus bas */ }
        await attendre(800);
        await lire(pg, () => { window.__retarder = null; });
        let l = await lire(pg, (id) => ({ note: !!(window.__notes() || {})[id], carte: !!document.querySelector(`[data-supprimer="${id}"]`), chemin: location.pathname }), id);
        let t = await messageVu(pg);
        verif(N("accueil : supprimer une note dont la saisie s'enregistre encore — la suppression ATTEND l'enregistrement (aucun refus « supprimée »), la note ne revient pas"),
          enVol && !l.note && !l.carte && l.chemin === '/' && !/supprimée/.test(t), court({ ...l, enVol, message: t }));
        // une note supprimée AILLEURS (un autre onglet) pendant l'enregistrement gardé : la réponse ne la réécrit pas, et le dit
        // en nommant la note et son module
        await semer(pg, noteEssai(id, 'Essai frais'));
        await allerA(pg, './');
        await naviguerApp(pg, `${P('poutre')}?note=${id}`, `/${P('poutre')}`);
        champ = await champVisible(pg);
        await pg.fill(`#donnees input[data-chemin="${champ.chemin}"]`, String(autreValeur(champ, 0.9)).replace('.', ','));
        vol = await lire(pg, () => { window.__retarder = { ms: 3000, module: 'poutre' }; return window.__retardees.length; });
        avant = await etatEcran(pg);
        await cliquerAccueil(pg);
        await pretEcran(pg, avant, { chemin: '/' });
        const tenue = await attendreTenue(pg, vol);
        await effacerMessage(pg);
        await lire(pg, (id) => { const ns = window.__notes() || {}; delete ns[id]; localStorage.setItem('stab.notes.v1', JSON.stringify(ns)); }, id);
        try { await pg.waitForFunction(() => window.__livrees.length >= window.__retardees.length, null, { timeout: 15000 }); } catch (e) { /* jugé plus bas */ }
        await attendre(800);
        await lire(pg, () => { window.__retarder = null; });
        l = await lire(pg, (id) => ({ note: !!(window.__notes() || {})[id], carte: !!document.querySelector(`[data-supprimer="${id}"]`) }), id);
        t = await messageVu(pg);
        verif(N(`une note supprimée ailleurs pendant son enregistrement gardé n'est pas réécrite, et le message le dit en la nommant (« Essai frais » (${MODULES.poutre.nom}))`),
          tenue && !l.note && !l.carte && t.includes(`« Essai frais » (${MODULES.poutre.nom}) : la note a été supprimée`), court({ ...l, tenue, message: t }));
      });
    }

    // ---- une MODIFICATION refusée par le moteur : dans l'écran, et après la fermeture (revue S4, etat-02, refus anonyme)
    if (joue('refus')) {
      await essai('refus', async () => {
        const id = 'essai-refus';
        await semer(pg, noteEssai(id, 'Essai refus'));
        await allerA(pg, './');
        await naviguerApp(pg, `${P('poutre')}?note=${id}`, `/${P('poutre')}`);
        if (!(await installerRefus(pg))) { verif(N("refus : l'enveloppe du travailleur est posée"), false); return; }
        // (1) dans l'écran : le champ revient à la valeur du moteur, la note et le résultat ne changent pas, le message le dit
        const champ = await champVisible(pg);
        const sel = `#donnees input[data-chemin="${champ.chemin}"]`, v = autreValeur(champ, 1.3);
        const lireEcran = () => lire(pg, ([s, id]) => {
          const e = document.querySelector(s), t = document.getElementById('toast');
          return { valeur: e ? e.value : null, defaut: e ? e.defaultValue : null, resultat: (document.getElementById('resultat') || {}).innerHTML || '',
            etat: JSON.stringify(((window.__notes() || {})[id] || {}).etat), message: t && t.checkVisibility({ visibilityProperty: true, opacityProperty: true }) ? t.textContent : '' };
        }, [sel, id]);
        const a0 = await lireEcran();
        await effacerMessage(pg);
        await lire(pg, () => { window.__refus = { module: 'poutre', action: 'modifier', erreur: 'essai de refus' }; });
        await pg.fill(sel, String(v).replace('.', ','));
        await lire(pg, (s) => document.querySelector(s).blur(), sel);
        try { await pg.waitForFunction(() => window.__refusFaits >= 1, null, { timeout: 15000 }); } catch (e) { /* jugé plus bas */ }
        await attendre(500);
        const a1 = await lireEcran();
        verif(N(`refus d'une modification : le champ revient à la valeur du moteur (${champ.valeur}, pas ${v}), la note et le résultat sont inchangés, le message dit que la saisie n'est pas prise`),
          valeurNombre(a1.valeur) === valeurNombre(champ.valeur) && valeurNombre(a1.valeur) !== v && a1.valeur === a1.defaut && a1.etat === a0.etat
            && a1.resultat === a0.resultat && /Le moteur a refusé : essai de refus/.test(a1.message) && /n'est pas prise/.test(a1.message),
          court({ champ: champ.chemin, avant: champ.valeur, tape: v, apres: a1.valeur, defaut: a1.defaut, noteInchangee: a1.etat === a0.etat, resultatInchange: a1.resultat === a0.resultat, message: a1.message }));
        // (2) la réponse arrive APRÈS la fermeture (retardée), refusée : le message est dit dans l'écran suivant, en nommant
        // la note et son module, une seule fois préfixé
        await effacerMessage(pg);
        const c2 = await champVisible(pg);
        const v2 = autreValeur(c2, 1.25);
        await lire(pg, () => { window.__refus = { module: 'poutre', action: 'modifier', erreur: 'Valeur refusée : essai' }; window.__retarder = { ms: 1500, module: 'poutre' }; });
        await pg.fill(`#donnees input[data-chemin="${c2.chemin}"]`, String(v2).replace('.', ','));
        const avant = await etatEcran(pg);
        await cliquerLien(pg, P('assemblages'));
        await pretEcran(pg, avant, { chemin: `/${P('assemblages')}` });
        try { await pg.waitForFunction(() => window.__refusFaits >= 2, null, { timeout: 15000 }); } catch (e) { /* jugé plus bas */ }
        await attendre(500);
        await lire(pg, () => { window.__retarder = null; window.__refus = null; });
        const t = await messageVu(pg);
        const attendu = `« Essai refus » (${MODULES.poutre.nom}) : Le moteur a refusé : Valeur refusée : essai`;
        const etat = await lire(pg, (id) => JSON.stringify(((window.__notes() || {})[id] || {}).etat), id);
        verif(N("refus gardé (la réponse arrive après la fermeture) : dit dans l'écran SUIVANT, VU, en nommant la note et son module (préfixé une seule fois) ; la note n'a pas changé"),
          t.startsWith(attendu) && (t.match(/« /g) || []).length === 1 && etat === a0.etat, court({ message: t, attendu, noteInchangee: etat === a0.etat }));
        // (le refus INJECTÉ est écrit à la console par l'établi, comme tout refus du moteur : ce n'est pas une erreur de la page)
        pg.erreurs = pg.erreurs.filter((x) => !/essai de refus|Valeur refusée : essai/.test(x));
      });
    }

    // ---- le démarrage RATÉ (moteur.zip servi en 500) : le bandeau rouge VU avec « Recharger », ce qui marche sans le moteur
    // reste actif — aux trois formats, à l'établi comme à la bibliothèque ; puis aux largeurs de la tablette et du PC (revue
    // S4, ecrans-erreur-export, preuves-bandeau-erreur, ecrans-menu-telephone-inerte, ecrans-bandeau-barre-tablette)
    if (joue('erreur')) {
      await essai('erreur', async () => {
        const echouer = (cx) => cx.route((u) => /\/moteur\/moteur\.zip$/.test(u.pathname), (r) => r.fulfill({ status: 500, contentType: 'text/plain', body: 'essai' }));
        for (const v of variantes(fmt)) {
          const NV = (x) => `demarrage · ${FMT[v].nom} · ${x}`;
          const c2 = await ouvrirContexte(b, v);
          await echouer(c2);
          const p2 = await nouvellePage(c2);
          try {
            await p2.goto(`${BASE}/`);
            await semer(p2, noteEssai('essai-erreur', 'Essai erreur'));
            await p2.goto(`${BASE}/${P('poutre')}?note=essai-erreur`);
            const sig = await pretEcran(p2, null, { chemin: `/${P('poutre')}`, delai: 60000 });
            const m = await lire(p2, MESURE_BARRE);
            verif(NV("démarrage raté (établi) : le bandeau ROUGE est VU, entier dans l'écran, « n'a pas pu démarrer », avec « Recharger » VU, dans l'écran, sous le doigt ; rien ne déborde de côté"),
              sig === 'ecran:erreur' && m.n === 1 && m.etat === 'erreur' && m.taille && m.taille[0] >= 40 && m.taille[1] >= 10 && m.dedans && /n'a pas pu démarrer/.test(m.texte)
                && !/Recharger la page/.test(m.texte) && !!m.recharger && m.recharger.dedans && m.recharger.touche && !m.deborde,
              court({ sig, ...m }, 700));
            // ce qui marche sans le moteur : « Exporter .stab » (télécharge la note), « Imprimer » ; au téléphone par le menu ⋯
            if (v === 'tel') {
              const mb = await lire(p2, ACTIF, '#menu-m');
              await toucher(p2, v, '#menu-m').catch(() => {});
              await attendre(300);
              const mm = await lire(p2, () => { const e = document.getElementById('mmenu'); return !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true }); });
              verif(NV('démarrage raté (téléphone) : le bouton ⋯ est actif et ouvre le menu'), mb.vu && !mb.inerte && mm, court({ bouton: mb, menu: mm }));
            }
            const ou = v === 'tel' ? '#mmenu' : '.top';
            const exp = await lire(p2, ACTIF, `${ou} [data-action="exporter"]`), imp = await lire(p2, ACTIF, `${ou} [data-action="imprimer"]`);
            const [dl] = await Promise.all([p2.waitForEvent('download', { timeout: 8000 }).catch(() => null), toucher(p2, v, `${ou} [data-action="exporter"]`).catch(() => {})]);
            let fichier = null;
            if (dl) { try { fichier = { nom: dl.suggestedFilename(), id: JSON.parse(fs.readFileSync(await dl.path(), 'utf8')).note.id }; } catch (e) { fichier = { erreur: e.message }; } }
            const restent = await lire(p2, () => ({
              pdf: [...document.querySelectorAll('[data-action="pdf"]')].every((e) => !!e.closest('[inert]')),
              ranger: [...document.querySelectorAll('.top [data-ranger], #mmenu [data-ranger]')].every((e) => !!e.closest('[inert]')),
              champs: [...document.querySelectorAll('#donnees input, #donnees select')].every((e) => !!e.closest('[inert]')),
            }));
            verif(NV('démarrage raté : « Exporter .stab » est VU, actif, sous le doigt, et télécharge LA note (un .stab) ; « Imprimer » est VU et actif ; la note PDF, « ranger » et les champs restent inertes'),
              exp.vu && !exp.inerte && exp.touche && imp.vu && !imp.inerte && imp.touche && !!fichier && /\.stab$/.test(fichier.nom || '') && fichier.id === 'essai-erreur'
                && restent.pdf && restent.ranger && restent.champs,
              court({ exporter: exp, imprimer: imp, fichier, restent }));
            await p2.screenshot({ path: path.join(OUT, `demarrage_erreur_${v}.png`) }).catch(() => {});
            // la bibliothèque : le même bandeau, la même règle
            await p2.goto(`${BASE}/bibliotheque-beton`);
            const sb = await pretEcran(p2, null, { chemin: '/bibliotheque-beton', delai: 60000 });
            const mbib = await lire(p2, MESURE_BARRE);
            verif(NV("démarrage raté (bibliothèque) : le bandeau ROUGE est VU, « n'a pas pu démarrer », « Recharger » VU dans l'écran et sous le doigt ; rien ne déborde"),
              sb === 'ecran:erreur' && mbib.n === 1 && mbib.etat === 'erreur' && mbib.dedans && /n'a pas pu démarrer/.test(mbib.texte) && !!mbib.recharger && mbib.recharger.dedans
                && mbib.recharger.touche && !mbib.deborde, court({ sb, ...mbib }, 600));
            await p2.screenshot({ path: path.join(OUT, `demarrage_erreur_bibliotheque_${v}.png`) }).catch(() => {});
          } finally { await c2.close(); }
        }
        // aux largeurs de la tablette et du PC (portrait, paysage) : la raison passe à la ligne, « Recharger » reste dans
        // l'écran, le nom de la note se lit, rien ne déborde — une note rouverte (son étiquette « Indice A » prend sa place)
        if (fmt === 'pc') {
          const fautes = [];
          for (const [w, h] of [[768, 1024], [820, 1180], [1024, 768], [1180, 820], [1440, 900]]) {
            const c3 = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1, hasTouch: w <= 1180 });
            await c3.addInitScript(INIT, {});
            await echouer(c3);
            const p3 = await c3.newPage();
            try {
              await p3.goto(`${BASE}/`);
              await semer(p3, noteEssai('essai-erreur', 'Essai erreur'));
              await p3.goto(`${BASE}/${P('poutre')}?note=essai-erreur`);
              const sig = await pretEcran(p3, null, { chemin: `/${P('poutre')}`, delai: 60000 });
              const m = await lire(p3, MESURE_BARRE);
              if (sig !== 'ecran:erreur' || !m.recharger || !m.recharger.dedans || !m.recharger.touche || m.deborde || !m.dedans || !m.nom || m.nom.vu < m.nom.texte + 14) {
                fautes.push(`${w}×${h} : ${court({ sig, ...m }, 400)}`);
              }
              if (w === 820) await p3.screenshot({ path: path.join(OUT, 'demarrage_erreur_820.png') }).catch(() => {});
            } catch (e) { fautes.push(`${w}×${h} : ${String(e.message).split('\n')[0]}`); } finally { await c3.close(); }
          }
          verif(N("démarrage raté, à 768, 820, 1 024, 1 180 et 1 440 px : « Recharger » VU dans l'écran et sous le doigt, le bandeau entier, le nom de la note lisible, aucun débordement de côté"),
            !fautes.length, fautes.join(' | ').slice(0, 900));
        }
      });
    }

    // ---- pendant le démarrage (le moteur retenu) : ce qui reste atteignable, ce qui se dit (revue S4, ecrans-apercu-mention,
    // ecrans-volet-croix-inerte, ecrans-menu-telephone-inerte, ecrans-alerte-couvre-onglets, ecrans-bandeau-barre-tablette ; C11)
    if (joue('lecture')) {
      await essai('lecture seule', async () => {
        // la mention de l'aperçu, VUE : en tête du résultat, ou (au téléphone) la suite du bandeau
        const MENTION = () => {
          const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const dedans = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && r.left >= -0.5 && r.right <= innerWidth + 0.5 && r.top >= -0.5 && r.bottom <= innerHeight + 0.5; };
          const t = (s, re) => [...document.querySelectorAll(s)].filter((e) => vu(e) && dedans(e) && re.test(e.textContent)).map((e) => e.textContent.trim());
          return { mention: t('.et-apercu-mention', /aperçu/i), suite: t('.et-demarre .et-demarre-t2', /aperçu/i),
            lecture: t('.et-apercu-mention, .et-demarre .et-demarre-t2', /lecture seule/), titre: (document.querySelector('.et-demarre') || {}).title || '',
            nMention: document.querySelectorAll('.et-apercu-mention').length, nBandeau: document.querySelectorAll('.et-demarre').length, ecran: document.body.dataset.ecran };
        };
        for (const v of variantes(fmt)) {
          const NV = (x) => `demarrage · ${FMT[v].nom} · ${x}`;
          const c2 = await ouvrirContexte(b, v, { retenir: 25000 });
          const p2 = await nouvellePage(c2);
          try {
            // (a) un calcul NEUF : l'aperçu le dit, aux trois formats (au téléphone : à chaque onglet)
            await p2.goto(`${BASE}/${P('poutre')}`, { waitUntil: 'domcontentloaded' });
            await p2.waitForFunction(() => !!window.__apercu, null, { timeout: 15000 }).catch(() => {});
            const vus = [];
            for (const o of v === 'tel' ? ['donnees', 'dessin', 'resultat'] : [null]) {
              if (o) { await p2.tap(`.mtabs [data-onglet="${o}"]`).catch(() => {}); await attendre(250); }
              vus.push({ onglet: o || '—', ...(await lire(p2, MENTION)) });
            }
            verif(NV(`aperçu : il se DIT aperçu, en lecture seule, de façon VUE${v === 'tel' ? ', à chaque onglet' : ' (en tête du résultat)'}`),
              vus.every((x) => x.ecran === 'apercu' && (x.mention.length || x.suite.length) && x.lecture.length && (v === 'tel' || x.mention.length)), court(vus, 700));
            if (v === 'tab') {
              const m = await lire(p2, MESURE_BARRE);
              verif(NV("aperçu : le bandeau cède sa place au NOM de la note (il se lit en entier), rien ne déborde"),
                !!m.nom && m.nom.vu >= m.nom.texte + 14 && m.n === 1 && m.dedans && !m.deborde, court(m, 500));
            }
            // la croix ✕ d'une fenêtre de calcul ouverte pendant l'aperçu la ferme (fermer, c'est naviguer)
            if (v === 'tel') { await p2.tap('.mtabs [data-onglet="resultat"]').catch(() => {}); await attendre(250); }
            const ck = await lire(p2, () => {
              const e = [...document.querySelectorAll('#resultat [data-calcul]')].find((x) => x.checkVisibility());
              if (!e) return null;
              e.setAttribute('data-essai-calcul', '1');
              return e.dataset.calcul;
            });
            let croix = { calcul: ck };
            if (ck) {
              await toucher(p2, v, '[data-essai-calcul]').catch(() => {});
              await attendre(350);
              croix.ouverte = await lire(p2, ACTIF, '.flyout.on [data-fermer]');
              await toucher(p2, v, '.flyout.on [data-fermer]').catch((e) => { croix.erreur = String(e.message).split('\n')[0]; });
              await attendre(350);
              croix.apres = await lire(p2, () => ({ on: [...document.querySelectorAll('.flyout.on')].filter((x) => x.checkVisibility()).length, volet: document.documentElement.classList.contains('volet-ouvert') }));
            }
            verif(NV("aperçu : une fenêtre de calcul s'ouvre depuis le résultat, et sa croix ✕ — VUE, active, sous le doigt — la ferme"),
              !!ck && croix.ouverte && croix.ouverte.vu && !croix.ouverte.inerte && croix.ouverte.touche && croix.apres.on === 0 && !croix.apres.volet, court(croix));
            if (v === 'tel') {
              // le bouton ⋯ : actif ; son menu donne le thème, les modules, « Ouvrir .stab » ; ce qui agit sur la note reste inerte
              await p2.tap('.mtabs [data-onglet="donnees"]').catch(() => {});
              const mb = await lire(p2, ACTIF, '#menu-m');
              const th0 = await lire(p2, () => document.documentElement.dataset.theme || '');
              await p2.tap('#menu-m').catch(() => {});
              await attendre(300);
              const menu = { theme: await lire(p2, ACTIF, '#mmenu [data-theme-toggle]'), modules: await lire(p2, ACTIF, '#mmenu [data-modules]'),
                ouvrir: await lire(p2, ACTIF, '#mmenu label[for="ouvrir"]'), exporterInerte: await lire(p2, () => !!document.querySelector('#mmenu [data-action="exporter"]').closest('[inert]')) };
              await p2.tap('#mmenu [data-theme-toggle]').catch(() => {});
              await attendre(300);
              const th1 = await lire(p2, () => document.documentElement.dataset.theme || '');
              verif(NV("aperçu (téléphone) : le bouton ⋯ est actif ; son menu donne le thème (il bascule), « Tous les modules » et « Ouvrir .stab », VUS et actifs ; « Exporter » reste inerte"),
                mb.vu && !mb.inerte && mb.touche && menu.theme.vu && !menu.theme.inerte && menu.modules.vu && !menu.modules.inerte && menu.ouvrir.vu && !menu.ouvrir.inerte
                  && menu.exporterInerte && th1 !== th0, court({ bouton: mb, menu, theme: [th0, th1] }));
            }
            await p2.screenshot({ path: path.join(OUT, `demarrage_lecture_${v}.png`) }).catch(() => {});
            // la première vraie réponse : la mention s'en va avec le bandeau
            await lire(p2, () => window.__relacher());
            await pretEcran(p2, null, { chemin: `/${P('poutre')}` }).catch(() => {});
            const fin = await lire(p2, MENTION);
            verif(NV("aperçu : la première vraie réponse efface la mention et le bandeau"), fin.ecran === 'pret' && fin.nMention === 0 && fin.nBandeau === 0, court(fin));
          } finally { await c2.close(); }
          // (b) une note ROUVERTE pendant le démarrage : jamais « aperçu » (C11) ; le bandeau dit « lecture seule »
          const c3 = await ouvrirContexte(b, v, { retenir: 25000 });
          const p3 = await nouvellePage(c3);
          try {
            await p3.goto(`${BASE}/`);
            await semer(p3, noteEssai('essai-rouverte', 'Essai rouverte'));
            await p3.goto(`${BASE}/${P('poutre')}?note=essai-rouverte`, { waitUntil: 'domcontentloaded' });
            await p3.waitForFunction(() => !!document.querySelector('#resultat .et-rouverte'), null, { timeout: 15000 }).catch(() => {});
            await attendre(800);
            const vus = [];
            for (const o of v === 'tel' ? ['donnees', 'dessin', 'resultat'] : [null]) {
              if (o) { await p3.tap(`.mtabs [data-onglet="${o}"]`).catch(() => {}); await attendre(250); }
              vus.push({ onglet: o || '—', ...(await lire(p3, MENTION)) });
            }
            const m = await lire(p3, MESURE_BARRE);
            verif(NV("note rouverte pendant le démarrage : rien ne dit « aperçu » (ni la mention, ni le bandeau), le bandeau dit « lecture seule »"),
              vus.every((x) => x.ecran === 'demarrage' && !x.mention.length && !x.suite.length && x.nMention === 0 && /lecture seule$/.test(x.titre) && !/aperçu/.test(x.titre))
                && (v !== 'tel' || vus.every((x) => x.lecture.length)), court(vus, 700));
            if (v === 'tab') verif(NV("note rouverte pendant le démarrage : le nom de la note se lit en entier à côté du bandeau, rien ne déborde"),
              !!m.nom && m.nom.vu >= m.nom.texte + 14 && m.n === 1 && m.dedans && !m.deborde, court(m, 500));
          } finally { await c3.close(); }
          // (c) au téléphone : une note ABSENTE (le message #et-alerte dès le montage) pendant le démarrage — le message ne
          // couvre pas les onglets, que le bandeau agrandit
          if (v === 'tel') {
            const c4 = await ouvrirContexte(b, v, { retenir: 25000 });
            const p4 = await nouvellePage(c4);
            try {
              await p4.goto(`${BASE}/${P('poutre')}?note=essai-inconnue`, { waitUntil: 'domcontentloaded' });
              await p4.waitForFunction(() => { const a = document.getElementById('et-alerte'); return a && a.checkVisibility() && document.querySelector('.mtabs .et-demarre'); }, null, { timeout: 15000 }).catch(() => {});
              await attendre(300);
              const o = await lire(p4, () => {
                const a = document.getElementById('et-alerte');
                return { alerte: !!a && a.checkVisibility({ visibilityProperty: true, opacityProperty: true }), haut: a ? Math.round(a.getBoundingClientRect().top) : null,
                  basOnglets: Math.round(document.querySelector('.mtabs').getBoundingClientRect().bottom),
                  onglets: [...document.querySelectorAll('.mtabs [data-onglet]')].map((t) => {
                    const r = t.getBoundingClientRect(), h = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
                    return { onglet: t.dataset.onglet, touche: !!h && !!h.closest('[data-onglet]') && h.closest('[data-onglet]') === t };
                  }) };
              });
              verif(N("note absente pendant le démarrage (téléphone) : le message #et-alerte est VU et ne couvre aucun onglet (chaque onglet est sous le doigt, le message sous leur bas réel)"),
                o.alerte && o.onglets.length === 3 && o.onglets.every((x) => x.touche) && o.haut >= o.basOnglets, court(o));
              await p4.screenshot({ path: path.join(OUT, 'demarrage_alerte_tel.png') }).catch(() => {});
            } finally { await c4.close(); }
          }
        }
        // (d) aux largeurs de la tablette et du PC : pendant le démarrage, le bandeau ne prend RIEN à la barre du haut (la part
        // vue du nom est la même avec et sans lui) et le nom se lit en entier (calcul neuf et note rouverte ; à 768 px, chaque
        // établi) ; le bandeau entier dans l'écran, rien ne déborde — le moteur ne démarre jamais ici (son travailleur ne se
        // charge pas : l'écran reste au démarrage, sans coût)
        if (fmt === 'pc') {
          const fautes = [];
          const SANS_BANDEAU = () => { document.querySelectorAll('.et-demarre').forEach((x) => { x.dataset.essaiCache = '1'; x.style.display = 'none'; }); };
          const AVEC_BANDEAU = () => { document.querySelectorAll('[data-essai-cache]').forEach((x) => { x.style.display = ''; delete x.dataset.essaiCache; }); };
          let n768 = 0;
          for (const [w, h] of [[768, 1024], [820, 1180], [1024, 768], [1180, 820], [1440, 900]]) {
            const c5 = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1, hasTouch: w <= 1180 });
            await c5.addInitScript(INIT, {});
            await c5.route((u) => /\/js\/travailleur\.js$/.test(u.pathname), () => { /* jamais servi : le moteur reste au démarrage */ });
            const p5 = await c5.newPage();
            try {
              await p5.goto(`${BASE}/`);
              await semer(p5, noteEssai('essai-rouverte', 'Essai rouverte'));
              const cas = [['poutre neuf', P('poutre')], ['poutre rouverte', `${P('poutre')}?note=essai-rouverte`]];
              if (w === 768) for (const k of ETABLIS) if (k !== 'poutre') cas.push([`${k} neuf`, P(k)]);
              for (const [quoi, adr] of cas) {
                await p5.goto(`${BASE}/${adr}`, { waitUntil: 'domcontentloaded' });
                await p5.waitForFunction(() => document.querySelector('.et-demarre'), null, { timeout: 10000 }).catch(() => {});
                await attendre(quoi === 'poutre neuf' ? 900 : 400);
                const m = await lire(p5, MESURE_BARRE);
                await lire(p5, SANS_BANDEAU);
                const sans = await lire(p5, MESURE_BARRE);
                await lire(p5, AVEC_BANDEAU);
                if (w === 768) n768++;
                if (!m.nom || !sans.nom || m.nom.vu !== sans.nom.vu || m.nom.vu < m.nom.texte + 14 || m.n !== 1 || !m.dedans || m.deborde) {
                  fautes.push(`${w}×${h} ${quoi} : ${court({ nom: m.nom, sansBandeau: sans.nom && sans.nom.vu, n: m.n, dedans: m.dedans, deborde: m.deborde, taille: m.taille }, 300)}`);
                }
                if (w === 820 && quoi === 'poutre neuf') await p5.screenshot({ path: path.join(OUT, 'demarrage_barre_820.png') }).catch(() => {});
                if (w === 768 && quoi === 'assemblages neuf') await p5.screenshot({ path: path.join(OUT, 'demarrage_barre_768_assemblages.png') }).catch(() => {});
              }
            } catch (e) { fautes.push(`${w}×${h} : ${String(e.message).split('\n')[0]}`); } finally { await c5.close(); }
          }
          verif(N(`pendant le démarrage, à 768, 820, 1 024, 1 180 et 1 440 px : le bandeau ne prend rien à la barre du haut, le nom de la note s'y lit EN ENTIER (la poutre, neuve et rouverte ; à 768 px, les ${n768} établis), le bandeau dans l'écran, rien ne déborde`),
            !fautes.length && n768 === ETABLIS.length + 1, fautes.join(' | ').slice(0, 900));
        }
      });
    }

    // ---- le retour vers un établi au téléphone : l'onglet et le défilement rendus (revue S4, ecrans-retour-etabli-telephone)
    if (joue('retour_onglet') && fmt === 'tel') {
      await essai('retour onglet', async () => {
        const id = 'essai-onglet';
        await semer(pg, noteEssai(id, 'Essai onglet'));
        await allerA(pg, './');
        await naviguerApp(pg, `${P('poutre')}?note=${id}`, `/${P('poutre')}`);
        await pg.tap('.mtabs [data-onglet="resultat"]');
        await attendre(300);
        const vp = pg.viewportSize();
        let reduit = false;
        if ((await lire(pg, () => document.documentElement.scrollHeight - innerHeight)) < 200) { reduit = true; await pg.setViewportSize({ width: vp.width, height: 520 }); await attendre(250); }
        try {
          const y = await lire(pg, () => { const max = document.documentElement.scrollHeight - innerHeight; scrollTo(0, Math.min(600, Math.floor(max / 2))); return scrollY; });
          await attendre(300);
          await naviguerApp(pg, './', '/');
          const avant = await etatEcran(pg);
          await pg.goBack({ timeout: 20000 }).catch(() => null);
          await pretEcran(pg, avant, { chemin: `/${P('poutre')}` });
          await attendre(250);
          const l = await lire(pg, () => ({ tab: document.querySelector('.bench').dataset.tab, on: (document.querySelector('.mtabs [data-onglet].on') || {}).dataset?.onglet, y: scrollY,
            depuisMontage: Math.round(performance.now() - ((window.__ecrans.filter((e) => e.montage === document.body.dataset.montage)[0] || {}).t - performance.timeOrigin)) }));
          verif(N(`retour vers un établi (téléphone) : l'onglet Résultat et le défilement (${y} px) sont rendus dès la première réponse (avant le repli de l'application à 1,5 s)`),
            y > 0 && l.tab === 'resultat' && l.on === 'resultat' && Math.abs(l.y - y) < 5 && l.depuisMontage < 1500,
            court({ ...l, attendu: y, fenetre: reduit ? `${vp.width}×520` : `${vp.width}×${vp.height}` }));
          // la bibliothèque : son défilement rendu quand son tableau est là
          await naviguerApp(pg, 'bibliotheque-beton?rubrique=sections', '/bibliotheque-beton');
          await attendre(300);
          const yb = await lire(pg, () => { const max = document.documentElement.scrollHeight - innerHeight; scrollTo(0, Math.min(500, Math.floor(max / 2))); return scrollY; });
          await attendre(300);
          await naviguerApp(pg, './', '/');
          const av2 = await etatEcran(pg);
          await pg.goBack({ timeout: 20000 }).catch(() => null);
          await pretEcran(pg, av2, { chemin: '/bibliotheque-beton' });
          await attendre(250);
          const lb = await lire(pg, () => ({ y: scrollY, depuisMontage: Math.round(performance.now() - ((window.__ecrans.filter((e) => e.montage === document.body.dataset.montage)[0] || {}).t - performance.timeOrigin)) }));
          verif(N(`retour vers une bibliothèque (téléphone) : le défilement (${yb} px) est rendu dès son tableau (avant le repli à 1,5 s)`),
            yb > 0 && Math.abs(lb.y - yb) < 5 && lb.depuisMontage < 1500, court({ ...lb, attendu: yb }));
        } finally { if (reduit) await pg.setViewportSize(vp); }
      });
    }

    // ---- les navigations écrites en JavaScript, sans navigation de document (S4-D2)
    if (joue('js')) {
      await essai('navigations JS', async () => {
        const juger = async (nom, R, s, chemin, recherche) => {
          const l = await etatEcran(pg), D = depuis(pg, R);
          verif(N(`${nom} : sans navigation de document, l'écran ${chemin} prêt`), D.navs === 0 && l.s === s && l.chemin.endsWith(chemin) && recherche.test(l.recherche),
            court({ navigations: D.navs, sentinelle: l.s === s, chemin: l.chemin, recherche: l.recherche }));
        };
        // « Nouvelle note » de l'accueil
        await allerA(pg, './');
        await pg.click('#ouvrir-nouvelle');
        await pg.fill('#nn-chantier', 'Essai S4'); await pg.fill('#nn-nom', 'Note S4');
        await pg.click('.mopt[data-module="poutre"]');
        let avant = await etatEcran(pg), R = repere(pg);
        await pg.click('#nn-creer');
        await pretEcran(pg, avant, { chemin: `/${P('poutre')}` });
        await juger('« Nouvelle note »', R, avant.s, `/${P('poutre')}`, /note=/);
        // l'import d'un .stab depuis l'accueil
        await naviguerApp(pg, './', '/');
        const fichier = path.join(OUT, 'demarrage_import.stab');
        fs.writeFileSync(fichier, JSON.stringify({ format: 'stab', version: 1, note: { module: 'garde_corps', nom: 'Import S4', chantier: 'Essai S4', indice: 'A', etat: null } }));
        avant = await etatEcran(pg); R = repere(pg);
        await pg.setInputFiles('#importer', fichier);
        await pretEcran(pg, avant, { chemin: `/${P('garde_corps')}` });
        await juger('import .stab depuis l\'accueil', R, avant.s, `/${P('garde_corps')}`, /note=/);
        const idG = new URLSearchParams((await etatEcran(pg)).recherche).get('note');
        // une note d'un AUTRE module, demandée à la page du plancher : la page de son module, dans l'application
        avant = await etatEcran(pg); R = repere(pg);
        await cliquerLien(pg, `${P('plancher')}?note=${idG}`);
        await pretEcran(pg, avant, { chemin: `/${P('garde_corps')}` });
        await juger('note d\'un autre module (garde-corps demandée au plancher)', R, avant.s, `/${P('garde_corps')}`, new RegExp(`note=${idG}`));
        // un lien vers predalle?note= : la page des dalles, dans l'application
        await lire(pg, () => { const ns = window.__notes() || {}, t = Date.now(); ns['essai-predalle'] = { id: 'essai-predalle', module: 'predalle', nom: 'Prédalle S4', chantier: 'Essai S4', dossier: '', indice: 'A', cree: t, modifie: t, etat: null, resume: {}, vignette: '' }; localStorage.setItem('stab.notes.v1', JSON.stringify(ns)); });
        avant = await etatEcran(pg); R = repere(pg);
        await cliquerLien(pg, `${P('predalle')}?note=essai-predalle`);
        await pretEcran(pg, avant, { chemin: `/${P('dalle')}` });
        await juger('lien vers predalle?note=', R, avant.s, `/${P('dalle')}`, /note=essai-predalle/);
      });
    }

    // ---- feuilles ACTIVES (et leur ordre) et styles calculés égaux à ceux d'un chargement direct (C9)
    if (joue('feuilles')) {
      await essai('feuilles', async () => {
        const relever = (pg) => lire(pg, () => {
          const actives = [...document.styleSheets].filter((s) => !s.disabled && (!s.media.mediaText || s.media.mediaText === 'all' || matchMedia(s.media.mediaText).matches))
            .map((s) => (s.href ? new URL(s.href).pathname : `<style>${(s.ownerNode && s.ownerNode.id) || ''}`));
          const liens = [...document.querySelectorAll('link[rel~="stylesheet"]')].map((l) => new URL(l.href).pathname);
          const PROPS = ['display', 'position', 'color', 'backgroundColor', 'fontFamily', 'fontSize', 'fontWeight', 'lineHeight', 'paddingTop', 'paddingLeft', 'marginTop',
            'borderTopWidth', 'borderTopColor', 'gridTemplateColumns', 'flexDirection', 'gap', 'zIndex', 'overflowY', 'boxShadow', 'borderRadius'];
          const styles = {};
          for (const s of ['html', 'body', '.app', '.nav', '.top', '.mtop', '.mtabs', '.panel.left', '#donnees', '#resultat', '.canvas', '.ctool', '.famille', '.tile', '.h1', 'h1', '.btn', '.et-elt', '#bib-tableau']) {
            const e = document.querySelector(s);
            if (e && !e.closest('.modpanel, #__lien')) { const cs = getComputedStyle(e); styles[s] = PROPS.map((p) => cs[p]).join('|'); }
          }
          return { actives, liens, styles };
        });
        // la souris au même point des deux pages (un :hover ne fausse pas la comparaison)
        const releverA = async (x) => { await x.mouse.move(FMT[fmt].ctx.viewport.width / 2, FMT[fmt].ctx.viewport.height / 2); await attendre(80); return relever(x); };
        const chaine = [['./', '/'], ['bibliotheque-beton', '/bibliotheque-beton'], [P('poutre'), `/${P('poutre')}`], [P('dalle'), `/${P('dalle')}`], [P('poutre'), `/${P('poutre')}`]];
        const vus = [];
        const R = repere(pg), s = (await etatEcran(pg)).s;
        await allerA(pg, './');
        vus.push(await releverA(pg));
        let avant = await etatEcran(pg);
        await pg.click('.tile[data-bibliotheque="beton"]');
        await pretEcran(pg, avant, { chemin: '/bibliotheque-beton' });
        vus.push(await releverA(pg));
        for (const [href, ch] of chaine.slice(2)) { await naviguerApp(pg, href, ch); vus.push(await releverA(pg)); }
        const D = depuis(pg, R), sFin = (await etatEcran(pg)).s;
        const direct = await nouvellePage(c);
        const fautes = [];
        for (let i = 0; i < chaine.length; i++) {
          await direct.goto(`${BASE}/${chaine[i][0] === './' ? '' : chaine[i][0]}`);
          await pretEcran(direct, null, { chemin: chaine[i][1] });
          const d = await releverA(direct), a = vus[i];
          // (les <style> de la page — écrits par ses scripts — se comptent à part : la règle porte sur les <link>)
          const liensActifs = (x) => x.actives.filter((h) => !h.startsWith('<style>')), styles = (x) => x.actives.length - liensActifs(x).length;
          if (liensActifs(a).join() !== d.liens.join()) fautes.push(`${chaine[i][1]} : actives ${liensActifs(a).join(',')} ≠ liens de la page ${d.liens.join(',')}`);
          if (liensActifs(d).join() !== d.liens.join()) fautes.push(`${chaine[i][1]} (direct) : actives ${liensActifs(d).join(',')}`);
          if (styles(a) !== styles(d)) fautes.push(`${chaine[i][1]} : ${styles(a)} <style> actif(s) contre ${styles(d)} en direct`);
          const diff = Object.keys(d.styles).filter((k) => a.styles[k] !== d.styles[k]);
          if (diff.length) fautes.push(`${chaine[i][1]} : styles calculés différents (${diff.join(', ')})`);
        }
        await direct.close();
        verif(N('feuilles : accueil → bibliothèque béton → poutre → dalle → poutre, les feuilles ACTIVES dans l\'ordre de la page cible et les styles calculés d\'un chargement direct, sans document rechargé'),
          !fautes.length && D.navs === 0 && sFin === s, court({ navigations: D.navs, sentinelle: sFin === s, fautes }, 900));
      });
    }

    // ---- écouteurs vivants revenus à leur compte après trois cycles ; le thème bascule UNE fois ; UN téléchargement
    if (joue('ecouteurs')) {
      await essai('écouteurs', async () => {
        await allerA(pg, './');
        const s = (await etatEcran(pg)).s, R = repere(pg);
        const depart = await lire(pg, () => window.__vivants());
        let surPoutre = null;
        const ecarts = [];
        for (let i = 0; i < 3; i++) {
          let avant = await etatEcran(pg);
          await pg.click('.tile[data-module="poutre"]');
          await pretEcran(pg, avant, { chemin: `/${P('poutre')}` });
          const vp = await lire(pg, () => window.__vivants());
          if (!surPoutre) surPoutre = vp; else if (JSON.stringify(vp) !== JSON.stringify(surPoutre)) ecarts.push(`poutre ${i + 1} : ${JSON.stringify(vp)} ≠ ${JSON.stringify(surPoutre)}`);
          avant = await etatEcran(pg);
          await cliquerAccueil(pg);
          await pretEcran(pg, avant, { chemin: '/' });
          const va = await lire(pg, () => window.__vivants());
          if (JSON.stringify(va) !== JSON.stringify(depart)) ecarts.push(`accueil ${i + 1} : ${JSON.stringify(va)} ≠ ${JSON.stringify(depart)}`);
        }
        const D = depuis(pg, R), sFin = (await etatEcran(pg)).s;
        const doc = D.navs === 0 && sFin === s;
        verif(N('écouteurs : après trois cycles accueil → poutre → accueil, les écouteurs vivants (window, document, MediaQueryList) et les minuteries reviennent à leur compte de départ, sans document rechargé'),
          doc && !ecarts.length, court({ depart, surPoutre, navigations: D.navs, sentinelle: sFin === s, ecarts }, 700));
        // le thème : UN changement par clic (un rail câblé deux fois le basculerait deux fois)
        const th = await lire(pg, () => new Promise((ok) => {
          const vus = [], mo = new MutationObserver((l) => l.forEach(() => vus.push(document.documentElement.dataset.theme || '')));
          mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
          const avant = document.documentElement.dataset.theme || '';
          const t = [...document.querySelectorAll('[data-theme-toggle]')].find((e) => e.checkVisibility());
          if (!t) { ok({ erreur: 'aucun bouton de thème visible' }); return; }
          t.click();
          setTimeout(() => { mo.disconnect(); ok({ avant, vus, apres: document.documentElement.dataset.theme || '' }); }, 400);
        }));
        verif(N('thème : un clic le bascule UNE fois'), doc && !th.erreur && th.vus.length === 1 && th.apres !== th.avant, court({ ...th, document: doc }));
        // « Exporter .stab » : UN téléchargement
        const avant = await etatEcran(pg);
        await pg.click('.tile[data-module="poutre"]');
        await pretEcran(pg, avant, { chemin: `/${P('poutre')}` });
        const n0 = pg.telechargements;
        let bouton = visible(pg, '[data-action="exporter"]');
        if (!(await bouton.count())) { await visible(pg, '#menu-m').click(); await attendre(200); bouton = visible(pg, '[data-action="exporter"]'); }
        await bouton.click();
        await attendre(2000);
        const D2 = depuis(pg, R), s2 = (await etatEcran(pg)).s;
        verif(N('« Exporter .stab » produit UN téléchargement'), pg.telechargements - n0 === 1 && D2.navs === 0 && s2 === s,
          `${pg.telechargements - n0} téléchargement(s) · navigations ${D2.navs} · sentinelle ${s2 === s ? 'gardée' : 'perdue'}`);
      });
    }

    // ---- hors ligne : un écran jamais ouvert ne s'ouvre pas, on RESTE, avec le message juste (C8) ; un écran déjà ouvert
    //      dans ce document se rouvre de la copie de sa page et calcule (R5 ; revue S4, deploiement-03) ; un Retour hors
    //      ligne laisse l'adresse et l'écran d'accord (deploiement-04)
    if (joue('horsligne')) {
      await essai('hors ligne', async () => {
        const M = await messagesApp(pg);
        await allerA(pg, './');
        await naviguerApp(pg, P('poutre'));
        const avant = await etatEcran(pg), R = repere(pg);
        await vieillirPages(pg);   // plus de 30 s : la copie fraîche des lectures d'avance ne compte plus
        await effacerMessage(pg);
        await c.setOffline(true);
        try {
          await cliquerLien(pg, P('soudures')).catch(() => {});
          await attendre(2500);
          const l = await lire(pg, () => ({ chemin: location.pathname, s: window.__s, message: (document.getElementById('toast') || {}).textContent || '' })).catch((e) => ({ erreur: String(e.message).slice(0, 120) }));
          verif(N('hors ligne : le lien vers un écran jamais ouvert laisse l\'écran en place, avec le message exact (« ne peut pas s\'ouvrir sans connexion », pas « pas encore chargé »)'),
            !l.erreur && l.chemin === avant.chemin && l.s === avant.s && l.message === M.PAS_DE_RESEAU && !/pas encore/.test(l.message) && depuis(pg, R).navs === 0,
            court({ ...l, attendu: M.PAS_DE_RESEAU, navigations: depuis(pg, R).navs }));
          let av = await etatEcran(pg);
          await cliquerLien(pg, './');
          const s1 = await pretEcran(pg, av, { chemin: '/', delai: 20000 }).catch((e) => `rien : ${String(e.message).split('\n')[0]}`);
          av = await etatEcran(pg);
          await cliquerLien(pg, P('poutre'));
          const s2 = await pretEcran(pg, av, { chemin: `/${P('poutre')}`, delai: 30000 }).catch((e) => `rien : ${String(e.message).split('\n')[0]}`);
          const l2 = await lire(pg, () => ({ chemin: location.pathname, s: window.__s, ecran: document.body.dataset.ecran, resultat: ((document.getElementById('resultat') || {}).innerHTML || '').length }));
          verif(N('hors ligne : un écran déjà ouvert dans ce document (l\'accueil, puis la poutre) se rouvre de la copie de sa page, et la poutre calcule (le moteur porte tous les modules)'),
            s1 === 'ecran:pret' && s2 === 'ecran:pret' && l2.chemin === `/${P('poutre')}` && l2.ecran === 'pret' && l2.resultat > 0 && l2.s === avant.s && depuis(pg, R).navs === 0,
            court({ accueil: s1, poutre: s2, ...l2, navigations: depuis(pg, R).navs }));
          av = await etatEcran(pg);
          await pg.goBack({ timeout: 10000 }).catch(() => null);
          const s3 = await pretEcran(pg, av, { chemin: '/', delai: 20000 }).catch((e) => `rien : ${String(e.message).split('\n')[0]}`);
          const l3 = await lire(pg, () => ({ chemin: location.pathname, page: document.body.dataset.page, s: window.__s }));
          verif(N('hors ligne, Retour : l\'écran de l\'entrée d\'arrivée (l\'accueil, déjà ouvert) s\'ouvre ; l\'adresse et l\'écran disent le même écran'),
            s3 === 'ecran:pret' && l3.chemin === '/' && l3.page === 'accueil' && l3.s === avant.s && depuis(pg, R).navs === 0, court({ signal: s3, ...l3, navigations: depuis(pg, R).navs }));
        } finally { await c.setOffline(false); }
      });
      // la page du document lui-même (lue par le navigateur, pas par app.js) se rouvre aussi hors ligne (une page à part)
      await essai('hors ligne · page du document', async () => {
        const p2 = await nouvellePage(c);
        try {
          await p2.goto(`${BASE}/${P('dalle')}`);
          await pretEcran(p2, null, { chemin: `/${P('dalle')}` });
          await naviguerApp(p2, './', '/');
          const av = await etatEcran(p2), R = repere(p2);
          await c.setOffline(true);
          try {
            await p2.goBack({ timeout: 10000 }).catch(() => null);
            const sig = await pretEcran(p2, av, { chemin: `/${P('dalle')}`, delai: 30000 }).catch((e) => `rien : ${String(e.message).split('\n')[0]}`);
            const l = await lire(p2, () => ({ module: document.body.dataset.module, s: window.__s, chemin: location.pathname }));
            verif(N('hors ligne, Retour vers la page du document lui-même (la dalle, chargée par le navigateur) : elle se rouvre de sa copie, et calcule'),
              sig === 'ecran:pret' && l.module === 'dalle' && l.s === av.s && depuis(p2, R).navs === 0, court({ signal: sig, ...l, navigations: depuis(p2, R).navs }));
          } finally { await c.setOffline(false); }
        } finally { await p2.close(); }
      });
    }

    // ---- une lecture de page qui traîne : la barre fine (aux trois formats : le même composant de site.css) et
    //      body[aria-busy] après 300 ms ; sans réponse, 10 s au plus, puis on RESTE avec le message (deploiement-05)
    if (joue('lent')) {
      await essai('lent', async () => {
        const M = await messagesApp(pg);
        await allerA(pg, P('poutre'));
        const cible1 = `/${P('sol')}`, est1 = (u) => new URL(u).pathname === cible1;
        const lente = async (route) => {
          if (!route.request().isNavigationRequest()) await attendre(2000);
          try { await route.continue(); } catch (e) { /* la page est passée à autre chose */ }
        };
        await pg.route(est1, lente);
        try {
          const avant = await etatEcran(pg), R = repere(pg);
          await cliquerLien(pg, P('sol'));
          await attendre(1000);
          const pendant = await lire(pg, () => {
            const e = document.querySelector('.app-attente'), r = e && e.getBoundingClientRect();
            return { vu: !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true }), boite: r ? [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)] : null,
              busy: document.body.getAttribute('aria-busy'), largeur: innerWidth, chemin: location.pathname };
          });
          await pretEcran(pg, avant, { chemin: cible1 });
          const apres = await lire(pg, () => ({ dom: !!document.querySelector('.app-attente'), busy: document.body.getAttribute('aria-busy') }));
          verif(N('lecture lente (2 s) : la barre .app-attente est VUE en haut de l\'écran, pleine largeur, body[aria-busy] posé, avant l\'échange ; tous deux retirés une fois l\'écran monté'),
            pendant.vu && !!pendant.boite && pendant.boite[0] === 0 && pendant.boite[1] === 0 && pendant.boite[2] >= pendant.largeur - 1 && pendant.boite[3] >= 2
              && pendant.busy === 'true' && pendant.chemin === avant.chemin && !apres.dom && !apres.busy && depuis(pg, R).navs === 0,
            court({ pendant, apres, navigations: depuis(pg, R).navs }));
        } finally { await pg.unroute(est1, lente); }
        const cible2 = `/${P('bois')}`, est2 = (u) => new URL(u).pathname === cible2;
        const muette = async (route) => {
          if (!route.request().isNavigationRequest()) await attendre(15000);
          try { await route.continue(); } catch (e) { /* abandonnée par la page */ }
        };
        await pg.route(est2, muette);
        try {
          await effacerMessage(pg);
          const ici = await etatEcran(pg), R = repere(pg), t0 = Date.now();
          await cliquerLien(pg, P('bois'));
          let t = null;
          try { await pg.waitForFunction((m) => !!m && (document.getElementById('toast') || {}).textContent === m, M.RESEAU_MUET, { timeout: 13000, polling: 100 }); t = Date.now() - t0; } catch (e) { /* jugé ci-dessous */ }
          const l = await lire(pg, () => ({ chemin: location.pathname, s: window.__s, dom: !!document.querySelector('.app-attente'), busy: document.body.getAttribute('aria-busy') }));
          verif(N('lecture sans réponse : après 10 s (AbortController), on RESTE sur l\'écran avec le message « Le réseau ne répond pas », sans navigation de document, la barre retirée'),
            t !== null && t > 9000 && t < 12000 && l.chemin === ici.chemin && l.s === ici.s && !l.dom && !l.busy && depuis(pg, R).navs === 0,
            court({ apres_ms: t, attendu: M.RESEAU_MUET, ...l, navigations: depuis(pg, R).navs }));
        } finally { await pg.unroute(est2, muette); }
      });
    }

    // ---- version changée : la page cible porte une autre stab-version → exactement UNE navigation de document (S4-D3) ;
    //      une mise en ligne vue ET le réseau absent : on reste, jamais de navigation ordinaire (C8 ; deploiement-02)
    if (joue('version')) {
      await essai('version · hors ligne', async () => {
        const M = await messagesApp(pg);
        await allerA(pg, './');
        await naviguerApp(pg, P('poutre'));
        const estManifeste = (u) => new URL(u).pathname === '/moteur/manifeste.json';
        const autre = async (route) => {
          try { const r = await route.fetch(); const j = await r.json(); j.version = 'ffffffffffffffff'; await route.fulfill({ response: r, json: j }); } catch (e) { try { await route.abort(); } catch (x) { /* déjà fini */ } }
        };
        await pg.route(estManifeste, autre);
        const R = repere(pg), s = (await etatEcran(pg)).s;
        let l1 = null, l2 = null, mise = false;
        try {
          await lire(pg, () => document.dispatchEvent(new Event('visibilitychange')));
          mise = await pg.waitForFunction(() => /mis à jour/.test((document.getElementById('toast') || {}).textContent || ''), null, { timeout: 10000 }).then(() => true, () => false);
          await c.setOffline(true);
          try {
            await effacerMessage(pg);
            await cliquerLien(pg, P('soudures')).catch(() => {});
            await attendre(2500);
            l1 = await lire(pg, () => ({ chemin: location.pathname, s: window.__s, message: (document.getElementById('toast') || {}).textContent || '' })).catch((e) => ({ erreur: String(e.message).slice(0, 120) }));
            const av = await etatEcran(pg).catch(() => null);
            await cliquerLien(pg, './').catch(() => {});
            await pretEcran(pg, av, { chemin: '/', delai: 20000 }).catch(() => null);
            l2 = await lire(pg, () => ({ chemin: location.pathname, s: window.__s, page: document.body.dataset.page })).catch((e) => ({ erreur: String(e.message).slice(0, 120) }));
          } finally { await c.setOffline(false); }
        } finally { await pg.unroute(estManifeste, autre); }
        const D = depuis(pg, R);
        verif(N('version changée ET hors ligne : aucune navigation de document — un écran jamais ouvert : on reste, avec le message ; un écran déjà ouvert (l\'accueil) : sa copie'),
          mise && D.navs === 0 && l1 && l1.s === s && l1.chemin === `/${P('poutre')}` && l1.message === M.PAS_DE_RESEAU && l2 && l2.s === s && l2.page === 'accueil',
          court({ miseEnLigneVue: mise, l1, l2, navigations: D.navs }));
        // le réseau revenu : la navigation suivante est ORDINAIRE (une mise en ligne a été vue pendant la session)
        const av = await etatEcran(pg).catch(() => null), R2 = repere(pg);
        await cliquerLien(pg, P('poutre')).catch(() => {});
        await pretEcran(pg, av, { chemin: `/${P('poutre')}` }).catch(() => null);
        verif(N('version changée, réseau revenu : la navigation suivante est ordinaire (exactement une navigation de document)'), depuis(pg, R2).navs === 1,
          `navigations ${depuis(pg, R2).navs}`);
      });
      await essai('version', async () => {
        await allerA(pg, P('poutre')).catch(() => null);
        const meta = await lire(pg, () => (document.querySelector('meta[name="stab-version"]') || {}).content || null);
        const lectures = [];
        const cible = `/${P('flambement')}`;
        const servir = async (route) => {
          lectures.push(route.request().isNavigationRequest() ? 'navigation' : 'lecture');
          const r = await route.fetch();
          const corps = (await r.text()).replace(/(<meta name="stab-version" content=")[^"]*/, '$1ffffffffffffffff');
          await route.fulfill({ response: r, body: corps });
        };
        await pg.route((u) => new URL(u).pathname === cible, servir);
        const avant = await etatEcran(pg), R = repere(pg);
        await cliquerLien(pg, P('flambement'));
        await pretEcran(pg, avant, { chemin: cible });
        await attendre(500);
        await pg.unroute((u) => new URL(u).pathname === cible, servir);
        const D = depuis(pg, R);
        verif(N('version changée : la page cible lue (une autre stab-version), puis EXACTEMENT une navigation de document'),
          !!meta && lectures.includes('lecture') && D.navs === 1, court({ meta, lectures, navigations: D.navs }));
      });
    }
    verif(N('session : aucune erreur JavaScript'), !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 400));
    await pg.screenshot({ path: path.join(OUT, `demarrage_session_${fmt}.png`) }).catch(() => {});
    await pg.close();

    // ---------------------------------------------------------------- le préchauffage au survol (pointeur fin)
    if (joue('prechauffage') && fmt === 'pc') {
      await essai('préchauffage', async () => {
        const p2 = await nouvellePage(c);
        await p2.goto(`${BASE}/?sonde=1`);
        await p2.waitForFunction(() => (window.__journal || []).some((e) => e.sens === 'recu' && e.type === 'pret'), null, { timeout: 45000 }).catch(() => {});
        const cle = 'assemblages';
        const n = (await etatEcran(p2)).journal;
        await p2.hover(`.tile[data-module="${cle}"]`);
        await p2.waitForFunction((n) => window.__journal.slice(n).some((e) => e.sens === 'envoye' && e.type === 'prechauffer'), n, { timeout: 5000 }).catch(() => {});
        await p2.waitForFunction((n) => window.__journal.slice(n).some((e) => e.sens === 'recu' && e.type === 'prechauffe'), n, { timeout: 30000 }).catch(() => {});
        const env = (await journal(p2, 0)).filter((e) => e.sens === 'envoye');
        const pre = env.filter((e) => e.type === 'prechauffer').map((e) => e.cle);
        const lu = await sondeModules(p2);
        const ops = env.map((e) => (e.type === 'prechauffer' ? { prechauffer: e.cle } : e.type === 'requete' ? { requete: e.requete } : null)).filter(Boolean);
        const j = jugerModules(lu, referenceCPython(ops.length ? ops : [{ prechauffer: cle }]), CLES, cle);
        verif(N(`préchauffage : le survol de la tuile « ${cle} » envoie « prechauffer » (cette clé seule) et n'importe QUE ce module`),
          pre.length >= 1 && pre.every((k) => k === cle) && j.ok && (!lu.numpy || AVEC_NUMPY.has(cle)), `prechauffer : ${JSON.stringify(pre)} · modules : ${j.info}`);
        await p2.close();
      });
    }

    // ---------------------------------------------------------------- au doigt : faire défiler l'accueil n'importe rien
    if (joue('doigt') && fmt === 'tel') {
      await essai('doigt', async () => {
        const p2 = await nouvellePage(c);
        await p2.goto(`${BASE}/?sonde=1`);
        await p2.waitForFunction(() => (window.__journal || []).some((e) => e.sens === 'recu' && e.type === 'pret'), null, { timeout: 45000 }).catch(() => {});
        const cdp = await c.newCDPSession(p2);
        const y0 = await lire(p2, () => scrollY);
        const tuiles = ['assemblages', 'soudures', 'flambement'];
        for (const k of tuiles) {
          const bt = await p2.locator(`.tile[data-module="${k}"]`).boundingBox();
          if (!bt) continue;
          const x = bt.x + bt.width / 2, y = Math.min(Math.max(bt.y + bt.height / 2, 120), 700);
          await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
          for (let i = 1; i <= 8; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: y - i * 12 }] }); await attendre(16); }
          await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
          await attendre(400);
        }
        await attendre(1500);
        const l = await lire(p2, () => ({ y: scrollY, chemin: location.pathname, pre: window.__journal.filter((e) => e.sens === 'envoye' && e.type === 'prechauffer').map((e) => e.cle) }));
        const lu = await sondeModules(p2);
        const j = jugerModules(lu, referenceCPython([]), CLES);
        verif(N('au doigt : trois défilements commencés sur des tuiles n\'importent rien (aucun « prechauffer », sys.modules du travailleur sans clé du registre) et ne naviguent pas'),
          l.y !== y0 && l.chemin === '/' && !l.pre.length && j.ok, court({ defile: [y0, l.y], prechauffer: l.pre, modules: j.info }, 600));
        await p2.close();
      });
    }

    // ---------------------------------------------------------------- l'aperçu, en lien direct (S4-D7, C10, C12)
    if (joue('apercu')) {
      await essai('aperçu', async () => {
        const c2 = await ouvrirContexte(b, fmt, { retenir: 15000 });   // le moteur « lent » : 15 s avant « pret » et toute réponse
        const p2 = await nouvellePage(c2);
        await p2.goto(`${BASE}/${P('poutre')}`, { waitUntil: 'domcontentloaded' });
        // le voile, et le bandeau, pendant le démarrage
        await attendre(300);
        const v0 = await lire(p2, () => ({ voile: window.__voile(), bandeau: window.__bandeau() }));
        verif(N('démarrage en lien direct : aucun voile (ni #demarrage, ni calque plein écran sous les cinq points)'),
          !v0.voile.demarrage && !v0.voile.couvrants.length, court(v0.voile));
        try { await p2.waitForFunction(() => !!window.__apercu, null, { timeout: 10000 }); } catch (e) { /* jugé plus bas */ }
        const a = await lire(p2, () => {
          const nav = performance.getEntriesByType('navigation')[0];
          return { apercu: window.__apercu, dcl: nav ? nav.domContentLoadedEventEnd : null, ecrans: window.__ecrans.map((e) => e.ecran),
            pret: (window.__journal.find((e) => e.sens === 'recu' && e.type === 'pret') || {}).t || null };
        });
        const dt = a.apercu && a.dcl != null ? Math.round(a.apercu.t - a.dcl) : null;
        verif(N('aperçu : visible au plus 300 ms après DOMContentLoaded (data-ecran = apercu), avant le « pret » du travailleur'),
          !!a.apercu && dt <= 300 && (!a.pret || a.apercu.abs < a.pret), court({ ms: dt, ecrans: [...new Set(a.ecrans)], avantPret: a.apercu && a.pret ? a.apercu.abs < a.pret : null }));
        const A = a.apercu || {};
        verif(N('aperçu : les CONTRÔLES des données, des volets et des barres d\'actions sont inertes'),
          !!A.controles && A.controles.n > 0 && A.controles.actifs === 0, court(A.controles || 'aucun aperçu'));
        verif(N('aperçu : le panneau des données n\'est pas inerte (lisible, défilable) et porte aria-busy'),
          !!A.controles && !A.controles.donneesInerte && A.controles.occupe, court(A.controles || 'aucun aperçu'));
        verif(N('aperçu : rien n\'est écrit dans la mémoire des notes (aucune note, ou des notes à l\'état nul)'),
          !!A.notes && Object.values(A.notes).every((n) => n.etat == null), court(A.notes === undefined ? 'aucun aperçu' : A.notes));
        // le bandeau, VU (checkVisibility et taille lue) ; au téléphone, à chaque onglet
        const bandeaux = [];
        const onglets = fmt === 'tel' ? await p2.$$eval('.mtabs [data-onglet]', (l) => l.map((x) => x.dataset.onglet)) : [null];
        for (const o of onglets) {
          if (o) { await p2.click(`.mtabs [data-onglet="${o}"]`).catch(() => {}); await attendre(200); }
          bandeaux.push({ onglet: o || 'PC', ...(await lire(p2, () => window.__bandeau())) });
        }
        verif(N(`aperçu : le bandeau .et-demarre « Le moteur de calcul démarre » se voit, entier, dans l'écran${fmt === 'tel' ? ', à chaque onglet' : ''}`),
          bandeaux.every((x) => x.vus === 1 && x.taille && x.taille[0] > 40 && x.taille[1] > 10 && x.dedans && /Le moteur de calcul démarre/.test(x.texte)), court(bandeaux, 700));
        if (fmt === 'tel') await p2.click('.mtabs [data-onglet="donnees"]').catch(() => {});
        // le panneau défile pendant l'aperçu (à la molette, sur lui)
        const sc = await lire(p2, () => {
          const d = document.getElementById('donnees');
          for (let e = d; e && e !== document.body; e = e.parentElement) {
            const cs = getComputedStyle(e);
            if (/(auto|scroll)/.test(cs.overflowY) && e.scrollHeight > e.clientHeight + 4) { const r = e.getBoundingClientRect(); e.setAttribute('data-essai-defile', '1'); return { x: r.left + r.width / 2, y: r.top + Math.min(r.height / 2, 200), avant: e.scrollTop }; }
          }
          return null;
        });
        if (sc) {
          await p2.mouse.move(sc.x, sc.y); await p2.mouse.wheel(0, 300); await attendre(400);
          const apres = await lire(p2, () => (document.querySelector('[data-essai-defile]') || {}).scrollTop);
          verif(N('aperçu : le panneau des données défile'), apres > sc.avant, `${sc.avant} → ${apres}`);
        } else verif(N('aperçu : le panneau des données défile'), !!a.apercu, a.apercu ? 'rien à faire défiler à ce format (contenu court)' : 'aucun aperçu');
        await p2.screenshot({ path: path.join(OUT, `demarrage_apercu_${fmt}.png`) }).catch(() => {});
        await lire(p2, () => window.__relacher());
        await pretEcran(p2, null, { chemin: `/${P('poutre')}` }).catch(() => {});
        await c2.close();
      });
    }

    // ---------------------------------------------------------------- la 1re réponse de chaque établi = son aperçu (S4-D10 (4))
    if (joue('etablis')) {
      await essai('établis', async () => {
        const p2 = await nouvellePage(c);
        await p2.goto(`${BASE}/`);
        const R = repere(p2), s = (await etatEcran(p2)).s;
        const fautes = [], exceptions = {};
        for (const k of ETABLIS) {
          const ra = await c.request.get(`${BASE}/moteur/apercus/${k}.json`);
          const apercu = ra.ok() ? await ra.text() : null;
          const r404 = apercu === null;
          const n = (await etatEcran(p2)).journal;
          let rep = null;
          try {
            await naviguerApp(p2, P(k));
            rep = await premiereReponse(p2, (await lire(p2, () => window.__s)) === s ? n : 0, k);
          } catch (e) { fautes.push(`${k} : ${String(e.message).split('\n')[0].slice(0, 100)}`); continue; }
          if (r404) { fautes.push(`${k} : aperçu absent (moteur/apercus/${k}.json)`); continue; }
          if (!rep) { fautes.push(`${k} : aucune réponse vue`); continue; }
          const cmp = comparerApercu(k, rep, apercu);
          if (!cmp.egal) fautes.push(`${k} : différent en ${cmp.diff}`);
          else if (cmp.exception) exceptions[k] = cmp.exception;
          const autre = cmp.exception && !EXCEPTIONS[k];
          if (autre) fautes.push(`${k} : exception non nommée`);
        }
        const D = depuis(p2, R), sFin = (await etatEcran(p2)).s;
        verif(N(`les ${ETABLIS.length} établis ouverts l'un après l'autre DANS l'application (sans document rechargé) : la première réponse de chacun égale son aperçu`),
          !fautes.length && D.navs === 0 && sFin === s, court({ navigations: D.navs, sentinelle: sFin === s, fautes, exceptions }, 1200));
        await p2.close();
      });
    }

    // ================================================================ les preuves de la revue S4 (lot « preuves »)
    // Sous-filtres rouverte, premiere, reouverture, focus : chacun dans SA page (le moteur démarré dans ce document), aux
    // formats du passage. Ils prouvent le comportement ACTUEL ; chacun a été joué contre une mutation du code qui doit le
    // faire rougir (rapport du lot, 10/11/2026) — la mutation est nommée dans le commentaire de l'essai.
    // la valeur d'un champ dans l'état d'une note : le chemin tel quel, sinon dans l'élément ACTIF (un établi compact :
    // { poutres: [...], actif } — le champ « h » de l'écran est poutres[actif].h)
    const valeurEtat = (etat, chemin) => {
      const x = lireChemin(etat, chemin);
      if (x !== undefined || !etat || typeof etat !== 'object' || !Number.isInteger(etat.actif)) return x;
      for (const l of Object.values(etat)) {
        if (Array.isArray(l) && l[etat.actif] && typeof l[etat.actif] === 'object') { const y = lireChemin(l[etat.actif], chemin); if (y !== undefined) return y; }
      }
      return undefined;
    };
    const vaut = (etat, chemin, v) => { const x = valeurEtat(etat, chemin); return x !== undefined && x !== null && valeurNombre(x) === v; };
    // une saisie VALIDÉE (le champ quitté) et sa réponse : la note s'écrit, l'adresse prend ?note= ; rend l'id de la note
    const saisirValider = async (p, sel, v) => {
      const n = (await etatEcran(p)).journal;
      await p.fill(sel, String(v).replace('.', ','));
      await lire(p, (s) => document.querySelector(s).blur(), sel);
      await p.waitForFunction((n) => /note=/.test(location.search) && window.__journal.slice(n).some((e) => e.sens === 'recu' && e.type === 'reponse' && e.module === 'poutre'), n, { timeout: 30000 });
      await attendre(400);
      return new URLSearchParams((await etatEcran(p)).recherche).get('note');
    };
    // toutes les réponses retardées livrées, puis le temps de l'écriture
    const apresLivraisons = async (p) => {
      await p.waitForFunction(() => window.__livrees.length >= window.__retardees.length, null, { timeout: 20000 }).catch(() => {});
      await attendre(600);
    };
    // l'état du démarrage, LU : le bandeau (dans le DOM, et vu), les contrôles inertes, aria-busy des panneaux, l'annonce
    const etatDemarrage = (p) => lire(p, () => {
      const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const bandeaux = [...document.querySelectorAll('.et-demarre')], inertes = [...document.querySelectorAll('[inert]')];
      return { ecran: document.body.dataset.ecran || null, bandeaux: bandeaux.length, bandeauxVus: bandeaux.filter(vu).length,
        inertes: inertes.length, exemplesInertes: inertes.slice(0, 4).map((e) => e.dataset.chemin || e.dataset.action || e.id || e.tagName.toLowerCase()),
        controles: window.__controles(),
        occupes: [...document.querySelectorAll('#donnees, #resultat, #stage, #bib-tableau')].filter((e) => e.getAttribute('aria-busy') === 'true').map((e) => e.id),
        annonce: (document.getElementById('annonce') || {}).textContent || '' };
    });
    // la première réponse appliquée (S4-D7, C10) : plus de bandeau, aucun contrôle inerte (un établi : tous ses contrôles
    // actifs), aucun panneau occupé, « prêt » annoncé, data-ecran = pret ; et son TÉMOIN, pendant le démarrage
    const premiereFaite = (l, etabli) => l.ecran === 'pret' && l.bandeaux === 0 && l.bandeauxVus === 0 && l.inertes === 0 && !l.occupes.length
      && /: prêt$/.test(l.annonce) && (!etabli || (l.controles.n > 0 && l.controles.actifs === l.controles.n));
    const demarrageVu = (l) => /^(demarrage|apercu)$/.test(l.ecran || '') && l.bandeauxVus >= 1 && l.inertes > 0 && l.occupes.length > 0 && !/prêt/.test(l.annonce);
    const sansControles = (l) => ({ ...l, controles: l.controles && { n: l.controles.n, actifs: l.controles.actifs } });

    // ---- C11 : une note ROUVERTE en lien direct pendant que le moteur démarre (revue S4, preuves-c11-sans-preuve) —
    //      jamais l'aperçu (les valeurs par DÉFAUT) ni une pastille de niveau : son dernier résumé, NEUTRE (« dernier
    //      enregistrement · … · recalcul en cours »), les données en squelette ; la première réponse venue, SES valeurs.
    //      Mutation qui rougit : dans etabli.js, « if (note.etat) montrerRouverte(); else if (…) apercu(); » réduit à
    //      l'aperçu seul.
    if (joue('rouverte')) {
      await essai('note rouverte', async () => {
        // une note poutre VRAIE, calculée par CPython (le routeur) : l'état du défaut modifié (h = 50, M_inf = 80), avec
        // SON résumé (un taux, un niveau) et SA vignette — un état que l'aperçu (le défaut) ne peut pas imiter
        const code = [
          'import sys, json', "sys.path.insert(0, '.')", 'import api_web.routeur as R',
          "r = json.loads(R.traiter(json.dumps({'module': 'poutre', 'action': 'nouveau'})))",
          "for ch, v in ((['h'], '50'), (['sections', '0', 'M_inf'], '80')):",
          "    r = json.loads(R.traiter(json.dumps({'module': 'poutre', 'action': 'modifier', 'etat': r['etat'], 'chemin': ch, 'valeur': v})))",
          "v = next((x['svg'] for x in r.get('vues') or [] if str(x.get('svg') or '').lstrip().startswith('<svg')), '')",
          "print(json.dumps({'ok': r.get('ok'), 'etat': r.get('etat'), 'resume': r.get('resume'), 'vignette': v}))",
        ].join('\n');
        const py = spawnSync(PYTHON, ['-c', code], { cwd: RACINE, encoding: 'utf8', timeout: 120000, maxBuffer: 64 << 20, env: { ...process.env, TZ: 'Europe/Paris' } });
        let sem = null;
        try { sem = JSON.parse(py.stdout.trim().split('\n').pop()); } catch (e) { sem = { erreur: (py.stderr || py.stdout || String(py.error || '')).slice(-300) }; }
        const hNote = sem && sem.etat ? valeurNombre(valeurEtat(sem.etat, 'h')) : NaN;
        if (!sem || !sem.ok || !sem.resume || !Number.isFinite(sem.resume.taux) || !/^[pfvr]$/.test(sem.resume.niveau || '') || !/^\s*<svg/.test(sem.vignette || '') || hNote !== 50) {
          verif(N('note rouverte : une note poutre calculée par CPython (h = 50), avec un taux, un niveau et une vignette'), false, court(sem && { ...sem, etat: undefined, vignette: (sem.vignette || '').length }));
          return;
        }
        const t = Date.now() - 86400000, id = 'essai-rouverte';
        const note = { id, module: 'poutre', nom: 'Rouverte C11', chantier: 'Essai S4', dossier: '', indice: 'A', cree: t, modifie: t, etat: sem.etat, resume: sem.resume, vignette: sem.vignette };
        const c2 = await ouvrirContexte(b, fmt, { retenir: 30000 });   // le moteur « lent » : 30 s avant « pret » et toute réponse
        try {
          const p2 = await nouvellePage(c2);
          await p2.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
          await lire(p2, ([id, n]) => { const ns = window.__notes() || {}; ns[id] = n; localStorage.setItem('stab.notes.v1', JSON.stringify(ns)); }, [id, note]);
          await p2.goto(`${BASE}/${P('poutre')}?note=${id}`, { waitUntil: 'domcontentloaded' });
          // (un aperçu, quand il est peint, l'est en moins de 300 ms : sous-filtre apercu) — on regarde à 1,5 s
          await attendre(1500);
          const lireR = () => lire(p2, () => {
            const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
            const rv = document.querySelector('#resultat .et-rouverte'), sq = document.querySelector('#donnees .et-squelette');
            const r = rv && rv.getBoundingClientRect(), q = sq && sq.getBoundingClientRect();
            return { ecran: document.body.dataset.ecran || null, apercu: !!window.__apercu,
              pastilles: [...document.querySelectorAll('.chip.p, .chip.f, .chip.v, .chip.r')].filter(vu).map((e) => `${e.id || e.className}:${e.textContent.trim()}`),
              rouverte: vu(rv) && r.width > 40 && r.height > 20, texte: rv ? rv.textContent.replace(/\s+/g, ' ').trim() : '',
              squelette: vu(sq) && q.width > 40 && q.height > 20, champs: document.querySelectorAll('#donnees [data-chemin]').length,
              bandeaux: window.__bandeau().vus };
          });
          const vus = [{ onglet: fmt === 'tel' ? 'donnees' : 'PC', ...(await lireR()) }];
          if (fmt === 'tel') {
            await p2.click('.mtabs [data-onglet="resultat"]').catch(() => {});
            await attendre(250);
            vus.push({ onglet: 'resultat', ...(await lireR()) });
          }
          await p2.screenshot({ path: path.join(OUT, `demarrage_rouverte_${fmt}.png`) }).catch(() => {});
          if (fmt === 'tel') { await p2.click('.mtabs [data-onglet="donnees"]').catch(() => {}); await attendre(250); }
          // le taux tel que l'écran l'écrit (js/commun.js, texteTaux : une seule source)
          const taux = (await lire(p2, async (x) => (await import(new URL('js/commun.js', document.baseURI).href)).texteTaux(x).texte, sem.resume.taux)).replace(/\s+/g, ' ');
          const neutre = vus.every((x) => x.ecran === 'demarrage' && !x.apercu && !x.pastilles.length && x.champs === 0 && x.bandeaux >= 1);
          const resume = vus.find((x) => x.rouverte);
          verif(N(`note rouverte (?note=) en lien direct, le moteur qui démarre : jamais l'aperçu ni une pastille de niveau ; son résumé NEUTRE vu (« ${note.nom} », ${taux}, « recalcul en cours »), les données en squelette vues, aucun champ`),
            neutre && !!resume && resume.texte.includes(note.nom) && !!taux && resume.texte.includes(taux) &&/dernier enregistrement/.test(resume.texte) && /recalcul en cours/.test(resume.texte)
              && vus.some((x) => x.squelette), court(vus, 900));
          // la première réponse venue : les valeurs de LA NOTE (jamais celles du défaut), sur son adresse, l'écran prêt
          await lire(p2, () => window.__relacher());
          const sig = await pretEcran(p2, null, { chemin: `/${P('poutre')}` }).catch((e) => `rien : ${String(e.message).split('\n')[0]}`);
          await attendre(300);
          const fin = await lire(p2, () => { const e = document.querySelector('#donnees input[data-chemin="h"]'); return { h: e ? e.value : null, vu: !!e && e.checkVisibility(), recherche: location.search }; });
          const apres = await etatDemarrage(p2);
          verif(N(`note rouverte, la première réponse venue : SES valeurs (h = ${hNote}, vu), sur son adresse, et l'écran prêt (bandeau retiré, contrôles actifs, aria-busy levé, « prêt » annoncé)`),
            sig === 'ecran:pret' && fin.vu && valeurNombre(fin.h) === hNote && fin.recherche.includes(`note=${id}`) && premiereFaite(apres, true), court({ signal: sig, ...fin, apres: sansControles(apres) }, 700));
          verif(N('note rouverte : aucune erreur JavaScript'), !p2.erreurs.length, p2.erreurs.join(' | ').slice(0, 400));
          await p2.close();
        } finally { await c2.close(); }
      });
    }

    // ---- la PREMIÈRE réponse (revue S4, preuves-premiere-reponse-non-verifiee) : le bandeau s'en va (ni dans le DOM, ni
    //      vu), aucun contrôle ne reste inerte, aria-busy est levé des panneaux, #annonce dit « prêt », data-ecran = pret
    //      — un établi (calcul neuf, après son aperçu) et une bibliothèque, en lien direct, le moteur retenu puis relâché ;
    //      le TÉMOIN : tout cela était bien là pendant le démarrage. Mutations qui rougissent (etabli.js) :
    //      « bandeau.retirer(); » retiré ; « lecture.lever(); » retiré ; « ctx.annoncer("prêt"); » retiré.
    if (joue('premiere')) {
      await essai('première réponse', async () => {
        const c2 = await ouvrirContexte(b, fmt, { retenir: 30000 });
        try {
          for (const [nom, page, etabli] of [['établi (poutre, calcul neuf, après son aperçu)', P('poutre'), true], ['bibliothèque (béton armé)', 'bibliotheque-beton', false]]) {
            const p2 = await nouvellePage(c2);
            await p2.goto(`${BASE}/${page}`, { waitUntil: 'domcontentloaded' });
            if (etabli) await p2.waitForFunction(() => !!window.__apercu, null, { timeout: 10000 }).catch(() => {});
            else await attendre(1000);
            const pendant = await etatDemarrage(p2);
            await lire(p2, () => window.__relacher());
            const sig = await pretEcran(p2, null, { chemin: `/${page}` }).catch((e) => `rien : ${String(e.message).split('\n')[0]}`);
            await attendre(300);
            const apres = await etatDemarrage(p2);
            if (etabli) await p2.screenshot({ path: path.join(OUT, `demarrage_premiere_${fmt}.png`) }).catch(() => {});
            verif(N(`première réponse · ${nom} : le bandeau .et-demarre retiré (ni dans le DOM, ni vu), aucun contrôle inerte, aucun aria-busy sur les panneaux, #annonce dit « prêt », data-ecran = pret (témoin : tout cela pendant le démarrage)`),
              demarrageVu(pendant) && sig === 'ecran:pret' && premiereFaite(apres, etabli), court({ signal: sig, pendant: sansControles(pendant), apres: sansControles(apres) }, 900));
            verif(N(`première réponse · ${nom} : aucune erreur JavaScript`), !p2.erreurs.length, p2.erreurs.join(' | ').slice(0, 400));
            await p2.close();
          }
        } finally { await c2.close(); }
      });
    }

    // ---- C1 « l'établi qui rouvre CETTE note attend son enregistrement » (revue S4, preuves-attente-enregistrement) :
    //      une saisie de la note X part au moteur (réponse retardée de 3 s), on quitte la poutre, et la note se ROUVRE
    //      AVANT la réponse — (a) par le retour arrière, (b) par un lien vers poutre?note=X. Le champ vaut la saisie, la
    //      première requête du nouveau montage la porte, la note stockée la garde après les livraisons.
    //      Mutations : app.js « attendreEnregistrement: (id) => Promise.resolve() » fait rougir (b) ; (a) est AUSSI gardé
    //      par l'attente de l'entrée d'historique dans aller() (« attentes », revue S4 etat-05) : il ne rougit qu'avec
    //      les deux retirées.
    if (joue('reouverture')) {
      await essai('réouverture', async () => {
        const p2 = await nouvellePage(c);
        try {
          await p2.goto(`${BASE}/${P('poutre')}`);
          await pretEcran(p2, null, { chemin: `/${P('poutre')}` });
          const champ = await champVisible(p2);
          if (!champ) { verif(N('réouverture : un champ numérique visible à saisir'), false); return; }
          const sel = `#donnees input[data-chemin="${champ.chemin}"]`;
          const id = await saisirValider(p2, sel, autreValeur(champ, 1.1));
          const gestes = [
            ['retour arrière', 1.2, () => p2.goBack({ timeout: 20000 }).catch(() => null)],
            [`lien vers ${P('poutre')}?note=`, 1.3, () => cliquerLien(p2, `${P('poutre')}?note=${encodeURIComponent(id)}`)],
          ];
          for (const [nom, k, geste] of gestes) {
            const v = autreValeur(champ, k);
            await p2.fill(sel, String(v).replace('.', ','));
            await lire(p2, () => { window.__retarder = { ms: 3000, module: 'poutre' }; });
            let avant = await etatEcran(p2);
            const R = repere(p2);
            await cliquerLien(p2, P('assemblages'));   // (le lien prend le focus : la saisie part au moteur, sa réponse retardée)
            await pretEcran(p2, avant, { chemin: `/${P('assemblages')}` });
            const au = await lire(p2, () => ({ retardees: window.__retardees.length, livrees: window.__livrees.length }));
            avant = await etatEcran(p2);
            await geste();
            let sig;
            try { sig = await pretEcran(p2, avant, { chemin: `/${P('poutre')}`, delai: 30000 }); } catch (e) { sig = `rien : ${String(e.message).split('\n')[0]}`; }
            const l = await lire(p2, ([s, n]) => ({ valeur: (document.querySelector(s) || {}).value, recherche: location.search, s: window.__s,
              premiere: (window.__journal.slice(n).find((e) => e.sens === 'envoye' && e.type === 'requete' && e.module === 'poutre') || {}).requete || null }), [sel, avant.journal]);
            await apresLivraisons(p2);
            await lire(p2, () => { window.__retarder = null; });
            const notes = (await lire(p2, () => window.__notes())) || {};
            let q = null;
            try { q = l.premiere && JSON.parse(l.premiere); } catch (e) { /* illisible : jugé ci-dessous */ }
            const lu = { requete: q ? `${q.action} ${champ.chemin}=${valeurEtat(q.etat, champ.chemin)}` : null, note: valeurEtat((notes[id] || {}).etat, champ.chemin) };
            verif(N(`réouverture (${nom}) AVANT la réponse retardée de la dernière saisie : le champ vaut la saisie (${champ.chemin} = ${v}), la première requête du nouveau montage la porte, la note stockée la garde après les livraisons`),
              sig === 'ecran:pret' && au.livrees < au.retardees && valeurNombre(l.valeur) === v && !!q && q.action === 'calculer' && vaut(q.etat, champ.chemin, v)
                && vaut((notes[id] || {}).etat, champ.chemin, v) && l.recherche.includes(`note=${encodeURIComponent(id)}`) && l.s === avant.s && depuis(p2, R).navs === 0,
              court({ signal: sig, auDepart: au, valeur: l.valeur, ...lu, recherche: l.recherche, navigations: depuis(p2, R).navs }));
          }
          verif(N('réouverture : aucune erreur JavaScript'), !p2.erreurs.length, p2.erreurs.join(' | ').slice(0, 400));
        } finally { await lire(p2, () => { window.__retarder = null; }).catch(() => {}); await p2.close(); }
      });
    }

    // ---- C1, étape (1) de la fermeture : le champ actif perd le focus, la saisie en cours est validée (revue S4,
    //      preuves-blur-fermeture-non-exerce). Un Retour (popstate) pendant que le champ est ENCORE focalisé et modifié —
    //      aucun lien ne prend le focus avant : la requête « modifier » part avec la valeur, après le geste, et la note la
    //      garde. Mutation qui rougit : app.js, le blur de fermerCourant() retiré.
    if (joue('focus')) {
      await essai('fermeture, champ focalisé', async () => {
        const p2 = await nouvellePage(c);
        try {
          await p2.goto(`${BASE}/`);
          await pretEcran(p2, null, { chemin: '/' });
          await naviguerApp(p2, P('poutre'));
          const champ = await champVisible(p2);
          if (!champ) { verif(N('fermeture, champ focalisé : un champ numérique visible à saisir'), false); return; }
          const sel = `#donnees input[data-chemin="${champ.chemin}"]`;
          const id = await saisirValider(p2, sel, autreValeur(champ, 1.1));
          const v = autreValeur(champ, 1.25);
          await p2.fill(sel, String(v).replace('.', ','));
          await lire(p2, () => { window.__retarder = { ms: 1000, module: 'poutre' }; });
          const avant = await etatEcran(p2), R = repere(p2);
          // au moment du geste : le champ a le focus, porte la saisie, et rien n'est encore parti pour elle
          const geste = await lire(p2, ([s, n]) => ({ focus: document.activeElement === document.querySelector(s), valeur: document.querySelector(s).value, t: window.__abs(),
            modifier: window.__journal.slice(n).filter((e) => e.sens === 'envoye' && e.type === 'requete' && e.action === 'modifier').length }), [sel, avant.journal]);
          await p2.goBack({ timeout: 20000 }).catch(() => null);
          let sig;
          try { sig = await pretEcran(p2, avant, { chemin: '/' }); } catch (e) { sig = `rien : ${String(e.message).split('\n')[0]}`; }
          const parties = (await journal(p2, avant.journal)).filter((e) => e.sens === 'envoye' && e.type === 'requete' && e.module === 'poutre' && e.action === 'modifier' && e.t >= geste.t)
            .map((e) => { try { return JSON.parse(e.requete); } catch (x) { return {}; } });
          const portee = parties.find((q) => (q.chemin || []).join('/') === champ.chemin && valeurNombre(q.valeur) === v);
          await apresLivraisons(p2);
          await lire(p2, () => { window.__retarder = null; });
          const notes = (await lire(p2, () => window.__notes())) || {};
          verif(N(`fermeture par un Retour (popstate) avec le champ ENCORE focalisé et modifié : la requête « modifier » part avec la valeur saisie (${champ.chemin} = ${v}) après le geste, et la note la garde`),
            /^(ecran:pret|accueil:sans-signal)$/.test(sig) && geste.focus && valeurNombre(geste.valeur) === v && geste.modifier === 0 && !!portee
              && vaut((notes[id] || {}).etat, champ.chemin, v) && depuis(p2, R).navs === 0,
            court({ signal: sig, geste, parties: parties.map((q) => `${(q.chemin || []).join('/')}=${q.valeur}`), note: valeurEtat((notes[id] || {}).etat, champ.chemin), navigations: depuis(p2, R).navs }));
          verif(N('fermeture, champ focalisé : aucune erreur JavaScript'), !p2.erreurs.length, p2.erreurs.join(' | ').slice(0, 400));
        } finally { await lire(p2, () => { window.__retarder = null; }).catch(() => {}); await p2.close(); }
      });
    }

    // ---------------------------------------------------------------- les temps : médiane de 3 tirages entrelacés (C14)
    if (joue('temps') || joue('etalons')) {
      await essai('temps', async () => {
        const seuil = 0.75 * os.cpus().length;
        let charge = os.loadavg()[0];
        const t0 = Date.now();
        while (charge >= seuil && Date.now() - t0 < 60000) { await attendre(2000); charge = os.loadavg()[0]; }
        if (charge >= seuil) { verif(N('temps : non mesurable'), false, `charge sur 1 min ${charge.toFixed(2)} ≥ ${seuil.toFixed(2)} (0,75 × ${os.cpus().length} cœurs) pendant 60 s`); return; }
        const p3 = await nouvellePage(c);
        const froid = async (cle) => {
          await p3.goto(`${BASE}/${P(cle)}`);
          await pretEcran(p3, null, { chemin: `/${P(cle)}` });
          return lire(p3, () => { const x = window.__prets[window.__prets.length - 1]; return x ? x.t - performance.timeOrigin : NaN; });
        };
        const passage = async (cle) => {
          const R = repere(p3);
          const { avant } = await passer(p3, fmt, cle, P(cle));
          const t = await lire(p3, (a) => {
            const tc = window.__tclic ?? Number(sessionStorage.getItem('__tclic'));
            const x = [...window.__prets].reverse().find((p) => window.__s !== a.s || (p.montage != null && Number(p.montage) > Number(a.montage)));
            return x ? x.t - tc : NaN;
          }, avant);
          const D = depuis(p3, R);
          const portes = [];
          if (D.navs) portes.push(`${D.navs} navigation(s) de document`);
          if (D.travailleurs) portes.push(`${D.travailleurs} travailleur(s) créé(s)`);
          const lourd = D.requetes.filter((u) => /moteur\.zip|pyodide\.asm\.wasm/.test(u)).map((u) => u.replace(/^.*\//, '').replace(/\?.*$/, ''));
          if (lourd.length) portes.push(`téléchargé : ${lourd.join(', ')}`);
          return { t, portes };
        };
        const T = [];
        for (let i = 0; i < 3; i++) {
          const fP = await froid('poutre');
          const pA = await passage('assemblages');
          const pP = await passage('poutre');
          const fA = await froid('assemblages');
          T.push({ fP, pA: pA.t, pP: pP.t, fA, portes: [...pA.portes, ...pP.portes] });
        }
        const m = { fP: mediane(T.map((x) => x.fP)), pA: mediane(T.map((x) => x.pA)), pP: mediane(T.map((x) => x.pP)), fA: mediane(T.map((x) => x.fA)) };
        const rA = m.pA / m.fA, rP = m.pP / m.fP, portes = [...new Set(T.flatMap((x) => x.portes))];
        const brut = T.map((x) => `[froid poutre ${Math.round(x.fP)} · → assemblages ${Math.round(x.pA)} · → poutre ${Math.round(x.pP)} · froid assemblages ${Math.round(x.fA)}]`).join(' ');
        console.log(`     ${N('temps')} : charge ${charge.toFixed(2)} · ${brut}`);
        if (joue('temps')) {
          verif(N('temps : poutre → assemblages → poutre, le passage dans l\'application ≤ 0,5 × le lien direct (médianes de 3 tirages entrelacés), portes tenues'),
            !portes.length && rA <= 0.5 && rP <= 0.5, `assemblages ${Math.round(m.pA)}/${Math.round(m.fA)} = ${rA.toFixed(2)} · poutre ${Math.round(m.pP)}/${Math.round(m.fP)} = ${rP.toFixed(2)}${portes.length ? ' · portes : ' + portes.join(' ; ') : ''}`);
        }
        if (joue('etalons')) {
          // ÉTALON : un passage FORCÉ par location.assign (un rechargement) doit dépasser 0,5 × le lien direct
          const avant = await etatEcran(p3);
          await lire(p3, () => { const t = window.__abs(); window.__tclic = t; sessionStorage.setItem('__tclic', String(t)); });
          await lire(p3, (pg) => { location.assign(pg); }, P('poutre')).catch(() => {});
          await pretEcran(p3, avant, { chemin: `/${P('poutre')}` });
          const tf = await lire(p3, () => { const x = window.__prets[window.__prets.length - 1]; return x ? x.t - Number(sessionStorage.getItem('__tclic')) : NaN; });
          verif(N('étalon : un passage forcé par location.assign est VU (rapport au lien direct > 0,5)'), tf / m.fP > 0.5, `${Math.round(tf)}/${Math.round(m.fP)} = ${(tf / m.fP).toFixed(2)}`);
        }
        await p3.close();
      });
    }
    if (joue('etalons')) {
      // ÉTALON : une tâche longue de 300 ms injectée sur l'accueil est détectée
      await essai('étalon tâche longue', async () => {
        const p4 = await nouvellePage(c);
        await p4.goto(`${BASE}/`);
        await lire(p4, () => new Promise((ok) => setTimeout(() => { const t = performance.now(); while (performance.now() - t < 300) { /* occupé */ } ok(); }, 50)));
        await attendre(800);
        const L = await lire(p4, () => window.__longues);
        const max = L ? Math.max(0, ...L.map((x) => x.d)) : null;
        verif(N('étalon : une tâche longue de 300 ms injectée sur l\'accueil est détectée'), max >= 290, `la plus longue : ${max} ms`);
        await p4.close();
      });
      // ÉTALON : l'aperçu servi avec 600 ms de retard → le contrôle des 300 ms rougit
      await essai('étalon aperçu', async () => {
        const c5 = await ouvrirContexte(b, fmt);
        const p5 = await nouvellePage(c5);
        let servis = 0;
        await p5.route((u) => /\/moteur\/apercus\//.test(u.pathname), async (r) => { servis++; await attendre(600); await r.continue(); });
        await p5.goto(`${BASE}/${P('poutre')}`);
        await p5.waitForFunction(() => !!window.__apercu, null, { timeout: 10000 }).catch(() => {});
        const a = await lire(p5, () => { const nav = performance.getEntriesByType('navigation')[0]; return { t: window.__apercu && window.__apercu.t, dcl: nav.domContentLoadedEventEnd }; });
        const dt = a.t ? Math.round(a.t - a.dcl) : null;
        verif(N('étalon : l\'aperçu servi avec 600 ms de retard fait rougir le contrôle des 300 ms (le retard est VU)'), servis > 0 && dt !== null && dt > 300,
          `${servis} aperçu(s) servi(s) · vu ${dt === null ? 'jamais' : dt + ' ms'} après DOMContentLoaded`);
        await c5.close();
      });
    }
    await c.close();
  }
  await b.close();
  console.log(`demarrage : ${ko} KO`);
  process.exit(ko ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
