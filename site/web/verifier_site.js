// Parcours du VRAI site, dans Chromium, comme le ferait le bureau : accueil vide, « Nouvelle note », calcul du
// garde-corps (saisie, type, calcul détaillé, choix d'un profilé, deuxième élément, rechargement), puis la poutre.
// Chaque pas est MESURÉ à l'écran : la bonne page, le résultat qui change, aucune erreur JavaScript, aucun débordement,
// aucun nom coupé (mesure exacte du texte), le dessin visible et à sa taille.
//
//   python3 site/web/outils/serveur_local.py 8770 &      (le site avec les en-têtes de _headers, comme Cloudflare)
//   node site/web/verifier_site.js http://127.0.0.1:8770 <dossier_des_captures> [garde|poutre|piege|modules|module:<clé>|formats|formats:<clé>|telecharger|meteo|sonde|enregistrement|ouverture|pilote|envoi|plaques|nappes|corniere|doigt|bibliotheque|bibliotheque_autres|bureau|note_pdf|demarrage[:sous-filtre,…]]
//
// « demarrage[:sous-filtre,…] » délègue à verifier_demarrage.js (le lot S4 : un moteur, pas de rechargement, aperçu,
// fermeture…). SANS filtre — le passage complet —, verifier_demarrage.js est joué AUSSI, à la fin, avec tous ses
// sous-filtres, et son verdict compte dans le code de sortie (revue S4, preuves-demarrage-hors-complet : la promesse
// centrale du lot n'était gardée que si l'on pensait à demander « demarrage »).
// Code de sortie : 0 tout est OK ; 1 un KO (ou un passage qui a planté) ; 3 NON CONCLUANT — aucun défaut vu, mais un
// contrôle a refusé de conclure (le sous-filtre « temps » de demarrage, sur une machine trop chargée : « non mesurable »).
// Un passage non mesuré n'est jamais un succès silencieux : le bilan le nomme, et le code n'est pas 0.
const { chromium, devices } = require('playwright');
const path = require('path');
const BASE = process.argv[2] || 'http://127.0.0.1:8770';
const OUT = process.argv[3] || '.';
const FILTRE = process.argv[4] || '';

// ---------------------------------------------------------------- le filtre « demarrage » (verifier_demarrage.js)
const DEMARRAGE = path.join(__dirname, 'verifier_demarrage.js');
// les sous-filtres de demarrage que joue un filtre : '' (tous), « a,b » (ceux-là) ; null : ce filtre ne le joue pas.
// Le passage complet (filtre vide) les joue TOUS (fonction PURE, prouvée par tests/test_site_web.py).
function sousFiltresDemarrage(filtre) {
  const f = String(filtre || '');
  if (!f) return '';
  const m = /^demarrage(?::(.*))?$/.exec(f);
  return m ? (m[1] || '') : null;
}
// un contrôle qui a REFUSÉ de conclure (il le dit dans son nom : « non mesurable »), pas un défaut constaté
const NON_MESURE = /:\s*non mesurable\b/;
// le verdict de verifier_demarrage.js, lu dans son code de sortie ET ses lignes : { code, ko, nonMesures } — code 0 (tout
// OK), 1 (un KO, ou le passage a planté : code 2 ou autre), 3 (ses seuls KO sont des contrôles non mesurés) (PURE)
function bilanDemarrage(statut, texte) {
  const lignes = String(texte || '').split('\n').filter((l) => /^KO {2}/.test(l)).map((l) => l.slice(4));
  const nonMesures = lignes.filter((l) => NON_MESURE.test(l)), ko = lignes.filter((l) => !NON_MESURE.test(l));
  if (statut === 0 && !lignes.length) return { code: 0, ko, nonMesures };
  if (statut === 1 && !ko.length && nonMesures.length) return { code: 3, ko, nonMesures };
  return { code: 1, ko, nonMesures };
}
// le code de sortie du passage : un KO du parcours ou de demarrage → 1 ; sinon un contrôle non mesuré → 3 ; sinon 0 (PURE)
function codeSortie(koParcours, demarrage = { code: 0 }) {
  if (koParcours || demarrage.code === 1) return 1;
  return demarrage.code === 3 ? 3 : 0;
}
// une ligne de bilan, lisible : ce que demarrage a dit
function direBilanDemarrage(d) {
  if (!d.code) return 'demarrage : tout OK';
  const nm = d.nonMesures.length ? ` · ${d.nonMesures.length} NON MESURÉ(S) — ${d.nonMesures.map((l) => `« ${l.split('  ')[0]} »`).join(', ')} : ce passage n'a PAS mesuré, il ne vaut pas succès` : '';
  return `demarrage : ${d.ko.length} KO${d.code === 1 && !d.ko.length && !d.nonMesures.length ? ' (le passage a planté)' : ''}${nm}`;
}
// jouer verifier_demarrage.js (un processus à part) : ses lignes passent telles quelles à l'écran ET sont lues pour le bilan
function jouerDemarrage(sous) {
  return new Promise((ok) => {
    const enfant = require('child_process').spawn(process.execPath, [DEMARRAGE, BASE, OUT, sous], { stdio: ['ignore', 'pipe', 'inherit'] });
    let texte = '';
    enfant.stdout.setEncoding('utf8');   // (un caractère accentué coupé entre deux morceaux reste entier)
    enfant.stdout.on('data', (x) => { process.stdout.write(x); texte += x; });
    enfant.on('error', (e) => { console.error(e); ok(bilanDemarrage(2, texte)); });
    enfant.on('close', (statut) => ok(bilanDemarrage(statut ?? 2, texte)));
  });
}
const SOUS_DEMARRAGE = sousFiltresDemarrage(FILTRE);
// « demarrage[:sous-filtre] » seul : verifier_demarrage.js, et rien d'autre
const DELEGUE = require.main === module && !!FILTRE && SOUS_DEMARRAGE !== null;
if (DELEGUE) {
  jouerDemarrage(SOUS_DEMARRAGE).then((d) => {
    console.log(`bilan : ${direBilanDemarrage(d)}`);
    process.exit(codeSortie(0, d));
  });
}

// ================================================================ les aides partagées par tous les filtres (lot S4)
// Elles sont définies HORS du parcours et exportées (module.exports, en bas) : un essai les charge sans lancer le
// parcours (require.main n'est pas ce fichier), pour les prouver sur des pages factices — mutations comprises.

// ---------------------------------------------------------------- le signal « écran prêt »
// (docs/site/REFONTE_ETABLI.md, S4-D7 et C1-C14) app.js pose body[data-montage] (le jeton du montage, avant d'ouvrir
// l'écran) ; l'écran pose body[data-ecran] ∈ demarrage | apercu | pret | erreur ; en erreur, le bandeau
// .et-demarre[data-etat="erreur"]. Les 36 lectures de l'ancien voile passent par ces deux aides, et par elles seules.
// Sans body[data-ecran], c'est une erreur TOUT DE SUITE (« signal de démarrage absent ») : l'attente de 180 s, puis
// l'exception hors de tout bilan, c'était le défaut à éviter.
// (Étape 4 de l'ordre de travail, faite à l'intégration du lot S4, le 10/11/2026 : le repli sur l'ancien voile #demarrage
// est retiré — un voile revenu sans body[data-ecran] est un signal ABSENT, plus un écran prêt. « #demarrage absent »
// est un contrôle : test_site_web « aucun id=demarrage », et le filtre demarrage · statique sur les 21 pages servies.)
const SIGNAL_ABSENT = 'signal de démarrage absent';
// le jeton du montage courant (body[data-montage]) : null sur l'ancien site, ou hors d'un écran monté par app.js
const montage = (pg) => pg.evaluate(() => {
  const v = document.body ? document.body.dataset.montage : undefined;
  return v === undefined || v === '' ? null : Number(v);
});
// attend l'écran PRÊT (ou en erreur) ; rend « pret » ou « erreur ».
//   apres : le jeton de montage lu AVANT une navigation dans le même document — l'écran attendu est un montage
//           POSTÉRIEUR (data-montage > apres), jamais l'écran d'avant ni la valeur statique copiée de la page cible ;
//   exigerPret : un écran en erreur lève (l'appelant attendait le voile caché, rien d'autre) ;
//   delai : le plafond de l'attente (180 s par défaut, comme avant).
async function pretEcran(pg, { apres = null, exigerPret = false, delai = 180000 } = {}) {
  const lire = async () => (await pg.waitForFunction((apres) => {
    const b = document.body;
    if (b && b.hasAttribute('data-ecran')) {
      if (apres !== null && apres !== undefined) {
        if (!b.hasAttribute('data-montage')) return 'sans-montage';
        if (!(Number(b.dataset.montage) > apres)) return false;
      }
      const e = b.dataset.ecran;
      return e === 'pret' || e === 'erreur' ? e : false;
    }
    // (le document se lit encore : le signal peut ne pas être né)
    return document.readyState === 'loading' ? false : 'absent';
  }, apres, { timeout: delai })).jsonValue();
  let r = await lire();
  // (un seul second regard, 300 ms plus tard : l'instant d'un échange de body, entre la copie des attributs de la page
  //  cible et le montage de son écran, ne doit pas passer pour un signal absent)
  if (r === 'absent') { await pg.waitForTimeout(300); r = await lire(); }
  if (r === 'absent') throw new Error(SIGNAL_ABSENT);
  if (r === 'sans-montage') throw new Error('jeton de montage absent : body[data-ecran] sans body[data-montage]');
  if (exigerPret && r === 'erreur') throw new Error(`l'écran est en erreur : ${await echecEcran(pg)}`);
  return r;
}
// l'échec du démarrage, LU : '' si l'écran n'a pas échoué, sinon le texte de l'échec (jamais vide). Il EXIGE qu'un signal
// existe : sans body[data-ecran], il lève (« absent » ne vaut pas « pas d'échec » — sans cette règle, un
// contrôle « pas d'erreur » passe toujours, quoi qu'il arrive).
async function echecEcran(pg) {
  const r = await pg.evaluate(() => {
    const b = document.body;
    const net = (t) => String(t || '').replace(/\s+/g, ' ').trim();
    if (b && b.hasAttribute('data-ecran')) {
      const bandeau = document.querySelector('.et-demarre[data-etat="erreur"]');
      if (b.dataset.ecran !== 'erreur' && !bandeau) return { echec: '' };
      return { echec: (bandeau && net(bandeau.textContent))
        || `écran en erreur (data-ecran=${b.dataset.ecran}${bandeau ? '' : ', sans bandeau .et-demarre[data-etat=erreur]'})` };
    }
    return { absent: true };
  });
  if (r.absent) throw new Error(SIGNAL_ABSENT);
  return r.echec;
}

// le bandeau d'ERREUR du démarrage, tel que l'œil le voit (revue S4, preuves-bandeau-erreur : « la page le dit » ne se
// lit pas dans body[data-ecran], qui passait sans aucun bandeau) : le bandeau .et-demarre[data-etat=erreur] VU
// (checkVisibility, au moins 40 × 10 px, dans l'écran), son texte, et son bouton « Recharger » VU, dans l'écran, sous le
// doigt (elementFromPoint)
async function bandeauErreur(pg) {
  return pg.evaluate(() => {
    const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
    const W = innerWidth, H = innerHeight;
    const dans = (r) => r.left >= -0.5 && r.top >= -0.5 && r.right <= W + 0.5 && r.bottom <= H + 0.5;
    const b = [...document.querySelectorAll('.et-demarre[data-etat="erreur"]')].find(vu);
    if (!b) return { vu: false, texte: '', recharger: false, ecran: document.body ? document.body.dataset.ecran : null };
    const r = b.getBoundingClientRect(), x = b.querySelector('[data-recharger]');
    const rx = x && vu(x) ? x.getBoundingClientRect() : null;
    const h = rx ? document.elementFromPoint(rx.left + rx.width / 2, rx.top + rx.height / 2) : null;
    return { vu: r.width >= 40 && r.height >= 10 && dans(r), taille: [Math.round(r.width), Math.round(r.height)], texte: b.textContent.replace(/\s+/g, ' ').trim(),
      recharger: !!rx && dans(rx) && !!h && x.contains(h), ecran: document.body.dataset.ecran };
  });
}
// le bandeau d'erreur dit-il « la page le dit » : VU, « n'a pas pu démarrer », « Recharger » VU
const bandeauDit = (be) => be.vu && be.recharger && /n'a pas pu démarrer/.test(be.texte);

// ---------------------------------------------------------------- le travailleur, vu de la page (le proxy des réponses)
// Une réponse peut être SIMULÉE (window.__simuler), RETOUCHÉE (window.__retoucher) ou RETARDÉE (window.__retarder), pour
// les gestes dont le moteur n'a pas la réponse (télécharger), qu'aucun module ne produit (zéro vue), ou pour une réponse
// TARDIVE (un écran fermé avant qu'elle arrive). ROBUSTE : il enveloppe le setter onmessage ET
// addEventListener('message') (un moteur partagé peut écouter par l'un ou par l'autre — l'ancien proxy, sur le seul
// setter, se taisait en silence, et les contrôles jugeaient alors le vrai moteur). window.__interceptees compte les
// réponses du VRAI travailleur passées par lui (une par message, quel que soit le nombre d'écouteurs) : un contrôle qui
// s'appuie sur le proxy exige d'abord __interceptees > 0, sinon KO « proxy inactif ». Posé par addInitScript, il survit
// aux échanges de page.
//   __retarder(predicat, ms) : chaque message du travailleur (sa donnée, avant retouche) pour lequel predicat est vrai est
//   livré ms plus tard ; rend une fonction qui retire la règle ; __retarder() sans argument les retire toutes.
const proxyTravailleur = () => {
  const W = window.Worker;
  window.__interceptees = 0;
  const regles = [];
  window.__retarder = (predicat, ms) => {
    if (!predicat) { regles.length = 0; return () => {}; }
    const r = { predicat, ms: Number(ms) || 0 };
    regles.push(r);
    return () => { const i = regles.indexOf(r); if (i >= 0) regles.splice(i, 1); };
  };
  // un message → sa livraison (la donnée retouchée, le retard) : calculée UNE fois, partagée par tous les écouteurs
  const vus = new WeakMap();
  const preparer = (ev) => {
    let t = ev && typeof ev === 'object' ? vus.get(ev) : null;
    if (t) return t;
    const d = ev && ev.data;
    let sortie = ev, delai = 0;
    if (d && d.type === 'reponse') {
      window.__interceptees++;
      if (window.__retoucher) sortie = { data: { ...d, json: window.__retoucher(d.json) } };
    }
    for (const r of regles) { try { if (r.predicat(d)) delai = Math.max(delai, r.ms); } catch (e) { /* un prédicat qui lève ne retarde pas */ } }
    t = { sortie, delai };
    if (ev && typeof ev === 'object') vus.set(ev, t);
    return t;
  };
  const appeler = (f, cible, ev) => (typeof f === 'function' ? f.call(cible, ev) : f && typeof f.handleEvent === 'function' ? f.handleEvent(ev) : undefined);
  const livrer = (f, cible, ev) => {
    const { sortie, delai } = preparer(ev);
    if (delai > 0) setTimeout(() => appeler(f, cible, sortie), delai);
    else appeler(f, cible, sortie);
  };
  window.Worker = class extends W {
    constructor(...a) {
      super(...a);
      this.__f = null;
      this.__enveloppes = new Map();
    }
    set onmessage(f) {
      this.__f = f || null;
      super.onmessage = f ? (ev) => livrer(f, this, ev) : null;
    }
    get onmessage() { return this.__f; }
    addEventListener(type, f, o) {
      if (type !== 'message' || !f) return super.addEventListener(type, f, o);
      let e = this.__enveloppes.get(f);
      if (!e) { e = (ev) => livrer(f, this, ev); this.__enveloppes.set(f, e); }
      return super.addEventListener(type, e, o);
    }
    removeEventListener(type, f, o) {
      if (type === 'message' && f && this.__enveloppes.has(f)) return super.removeEventListener(type, this.__enveloppes.get(f), o);
      return super.removeEventListener(type, f, o);
    }
    postMessage(m, ...reste) {
      const rep = window.__simuler && m && typeof m.requete === 'string' ? window.__simuler(JSON.parse(m.requete)) : null;
      if (rep) {
        // une réponse simulée va à TOUS les écouteurs (le setter et addEventListener), sans passer par le compte
        const ev = { data: { type: 'reponse', id: m.id, json: JSON.stringify(rep) } };
        setTimeout(() => { if (this.__f) appeler(this.__f, this, ev); for (const f of this.__enveloppes.keys()) appeler(f, this, ev); }, 30);
        return undefined;
      }
      return super.postMessage(m, ...reste);
    }
  };
};
// le compte du proxy, lu dans la page (0 si le proxy n'y est pas)
const interceptees = (pg) => pg.evaluate(() => window.__interceptees || 0);

// ---------------------------------------------------------------- PASSAGE=1 : ouvrir un écran DANS l'application
// Sans PASSAGE, ouvrir(pg, url) est page.goto(url) : un document neuf. Avec PASSAGE=1 (variable d'environnement, lue à
// chaque appel), il injecte <a href=url> dans le body courant et le CLIQUE : la vraie porte de navigation (après S4,
// app.js l'intercepte et échange la page dans le même document ; sur l'ancien site, le clic recharge le document, et
// c'est normal). Une page qui n'est pas encore sur le site (about:blank) passe d'abord par l'accueil (une navigation
// ordinaire), puis clique — sauf si la cible EST l'accueil. Rend le jeton de montage lu AVANT le clic (null pour un
// document neuf, et sur l'ancien site) : à passer à pretEcran({ apres }).
async function ouvrir(pg, url, { accueil = `${BASE}/` } = {}) {
  if (process.env.PASSAGE !== '1') { await pg.goto(url); return null; }
  const cible = new URL(url, accueil);
  let ici = null;
  try { ici = new URL(pg.url()); } catch (e) { /* pas d'adresse lisible */ }
  if (!ici || ici.origin !== cible.origin) {
    if (new URL(accueil).pathname === cible.pathname || /^\/(index\.html)?$/.test(cible.pathname)) { await pg.goto(cible.href); return null; }
    await pg.goto(accueil);
    ici = new URL(pg.url());
  }
  // (une ancre seule ne navigue pas : un goto, comme sans PASSAGE)
  if (ici.origin === cible.origin && ici.pathname === cible.pathname && ici.search === cible.search && cible.hash) { await pg.goto(cible.href); return null; }
  const m0 = await montage(pg);
  const s0 = await pg.evaluate(() => (window.__passage = Math.random()));
  await pg.evaluate((href) => {
    const a = document.createElement('a');
    a.href = href; a.id = 'passage-harnais'; a.textContent = 'passage';
    a.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483647;display:block;width:32px;height:32px;background:#f0f;color:#000;font:10px sans-serif';
    document.body.appendChild(a);
  }, cible.href);
  await Promise.all([pg.waitForNavigation({ timeout: 60000 }),
    pg.click('#passage-harnais', { timeout: 10000 }).catch(() => pg.$eval('#passage-harnais', (a) => a.click()))]);
  // (intégration du lot S4 : un clic qui RECHARGE le document n'est pas un passage dans l'application ; sans ce garde,
  //  le jeton de montage du document neuf ne dépassait jamais celui d'avant et pretEcran attendait 180 s pour rien)
  if ((await pg.evaluate(() => window.__passage).catch(() => undefined)) !== s0) throw new Error(`PASSAGE : le clic vers ${cible.pathname} a rechargé le document (navigation ordinaire, pas un passage dans l'application)`);
  return m0;
}

const principal = async () => {
  // une locale UTF-8 pour Chromium : sous la locale « C » d'un serveur, il ne sait pas écrire un nom de fichier accentué
  // sur le disque et enregistre « download » (mesuré : « Contrôle - « essai ».json » → « download ») — les postes du bureau
  // sont en UTF-8
  const b = await chromium.launch({ env: { ...process.env, LANG: process.env.LANG || 'C.UTF-8' } });
  let ko = 0;
  const verif = (nom, cond, info = '') => { console.log(`${cond ? 'OK' : 'KO'}  ${nom}${info ? '  ' + info : ''}`); if (!cond) ko++; };
  const ctx = await b.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
  const p = await ctx.newPage();
  const erreurs = [];
  p.on('pageerror', (e) => erreurs.push(e.message));
  p.on('console', (m) => { if (m.type() === 'error') erreurs.push(m.text().slice(0, 300)); });
  const capture = (nom) => p.screenshot({ path: path.join(OUT, `site_${nom}.png`) });
  // ce que l'œil voit : débordement, noms coupés (mesure au texte près), dessin à sa taille
  const mesure = () => p.evaluate(() => {
    const vis = (e) => e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
    const large = (e) => { const r = document.createRange(); r.selectNodeContents(e); return r.getBoundingClientRect().width; };
    const coupes = [...document.querySelectorAll('.prof > span:not(.leger), .el .l1 b, .row > label > span:not(.sym), .ni .t, .pk-row .nm, .tile .nm, .rows .seg.full button')]
      .filter(vis).filter((e) => e.scrollWidth > e.clientWidth + 1 || large(e) > e.getBoundingClientRect().width + 0.01).map((e) => e.textContent.trim());
    const svg = [...document.querySelectorAll('.view.on svg')].find(vis);
    const r = svg ? svg.getBoundingClientRect() : null;
    // une valeur saisie s'écrit à l'encre : la couleur CALCULÉE du champ, comparée à celle de --ink
    const sonde = document.createElement('span'); sonde.style.color = 'var(--ink)'; document.body.appendChild(sonde);
    const encre = getComputedStyle(sonde).color; sonde.remove();
    const gris = [...document.querySelectorAll('.field:not(.locked) input, .field:not(.locked) select')].filter(vis)
      .filter((e) => getComputedStyle(e).color !== encre).map((e) => e.dataset.chemin || e.id);
    // le contenu de la vue ouverte (un dessin, ou un tableau) : sa taille lue à l'écran
    const bloc = [...document.querySelectorAll('.view.on .draw > *')].find(vis);
    const rb = bloc ? bloc.getBoundingClientRect() : null;
    // un champ de TEXTE (un nom) s'écrit sans chiffres tabulaires : le trait d'union, rendu dans la police CALCULÉE du
    // champ, a sa largeur ordinaire (en tabulaire, « Garde-corps 1 » se lisait « Garde - corps 1 »)
    const trait = (cs, fvn) => { const t = document.createElement('span'); t.style.cssText = `position:absolute;white-space:pre;font:${cs.font};font-variant-numeric:${fvn}`;
      t.textContent = '-'; document.body.appendChild(t); const l = t.getBoundingClientRect().width; t.remove(); return l; };
    // (et le nom de l'élément ouvert, un champ de la ligne de l'élément en tête du panneau de gauche : la poutre, la dalle)
    const tabulaires = [...document.querySelectorAll('#donnees .field.text input, #donnees .field.area textarea, .panel.left .phead input.et-elt-nom')].filter(vis)
      .filter((e) => { const cs = getComputedStyle(e); return trait(cs, cs.fontVariantNumeric) > trait(cs, 'normal') + 0.2; }).map((e) => e.dataset.chemin);
    return { deborde: document.documentElement.scrollWidth > innerWidth, coupes, gris, tabulaires, svg: r ? [Math.round(r.width), Math.round(r.height)] : null,
      vue: rb ? [Math.round(rb.width), Math.round(rb.height)] : null };
  });
  // (refonte du 11/11/2026 : les modules passés ce jour-là sur l'établi compact) ce que l'œil LIT : dans CHAQUE rubrique
  // des données, une valeur saisie se lit en entier dans son champ (« 1 100 » ne se lit pas « 1 10 », vu sur la première
  // capture du garde-corps) ; à droite, ce qu'une vérification compare se lit en entier. Les rubriques sont ouvertes
  // l'une après l'autre (l'accordéon n'en montre qu'une), puis la première est rouverte.
  const REFONTE_1111 = ['garde_corps', 'flambement', 'profil', 'sol', 'soudures', 'plancher', 'assemblages'];
  const lisibles = async (pg) => {
    const out = { rubriques: 0, champs: [], cmps: [] };
    const lire = () => pg.evaluate(() => {
      const vis = (e) => e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      // les champs NUMÉRIQUES (une valeur, qui doit se lire entière) : un champ de texte libre — une source, un nom de
      // fiche — peut être plus long que sa case, il défile (trouvé sur le plancher, 12/11/2026 : « Source » du hourdis)
      return [...document.querySelectorAll('#donnees .field input:not([type=checkbox]):not([data-sorte="texte"])')].filter(vis)
        .filter((e) => e.scrollWidth > e.clientWidth + 1).map((e) => `${e.dataset.chemin} « ${e.value} »`);
    });
    const n = await pg.$$eval('#donnees .sec', (e) => e.length);
    for (let i = 0; i < n; i++) {
      const ouverte = await pg.$$eval('#donnees .sec', (e, k) => e[k].classList.contains('open'), i);
      if (!ouverte) { await (await pg.$$('#donnees .sec > .sechd'))[i].click(); await pg.waitForTimeout(160); }
      if (await pg.$$eval('#donnees .sec', (e, k) => e[k].classList.contains('open'), i)) { out.rubriques += 1; out.champs.push(...await lire()); }
      out.champs = [...new Set(out.champs)];
    }
    if (n && !(await pg.$$eval('#donnees .sec', (e) => e[0].classList.contains('open')))) {
      await (await pg.$$('#donnees .sec > .sechd'))[0].click(); await pg.waitForTimeout(160);
    }
    out.cmps = await pg.evaluate(() => [...document.querySelectorAll('#resultat .el.vc .l2 .cmp')]
      .filter((e) => e.checkVisibility({ visibilityProperty: true, opacityProperty: true }) && e.scrollWidth > e.clientWidth + 1)
      .map((e) => e.textContent.trim().slice(0, 50)));
    return out;
  };
  // (colonne de gauche compacte : la poutre, la dalle — construire.ARBRE_COMPACT) la LIGNE de l'élément ouvert, que
  // js/etabli.js pose dans l'en-tête du panneau des données, à côté d'« Annuler » et « Rétablir ». Ce que l'œil voit :
  // chaque pièce (le nom, la liste, copier, supprimer, puis Annuler, Rétablir, replier) visible et à une taille LUE, dans
  // l'en-tête, sur une seule rangée, sans chevauchement, dans cet ordre ; ``doigt`` : chaque cible à tant de px au moins.
  // ``nomEntier`` : le nom se lit en entier dans son champ (à l'appelant d'en juger : un nom long cède, c'est voulu).
  // null : la page n'a pas cette ligne (un établi ordinaire).
  const ligneElement = (pg, doigt = 0) => pg.evaluate((doigt) => {
    const ph = document.querySelector('.panel.left .phead'), l = ph && ph.querySelector(':scope > .et-elt > .et-elsel');
    if (!l) return null;
    const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
    if (!vu(l)) return { fautes: ['la ligne ne se voit pas'] };
    const nommer = (e) => (e.classList.contains('et-elt-nom') ? 'nom' : e.classList.contains('et-elt-liste') ? 'liste' : e.hasAttribute('data-dupliquer-index') ? 'copier'
      : e.hasAttribute('data-supprimer') ? 'supprimer' : e.dataset.historique || (e.dataset.coll ? 'replier' : e.id || e.className));
    const ATTENDUES = ['nom', 'liste', 'copier', 'supprimer', 'annuler', 'retablir'];
    // (la liste : son BOUTON — le menu, caché, n'a pas de place ; « Calcul… » n'en a que pendant un calcul : ignoré)
    const pieces = [...l.children, ...ph.children].filter((e) => !e.classList.contains('et-elt') && vu(e)).map((e) => {
      const r = (e.classList.contains('et-elt-liste') ? e.querySelector('[data-menu]') : e).getBoundingClientRect();
      return { n: nommer(e), r, t: `${Math.round(r.width)}×${Math.round(r.height)}` };
    }).filter((x) => ATTENDUES.includes(x.n) || x.n === 'replier').sort((a, b) => a.r.left - b.r.left);
    const rp = ph.getBoundingClientRect(), cy = (rp.top + rp.bottom) / 2;
    const fautes = ATTENDUES.filter((n) => !pieces.some((x) => x.n === n)).map((n) => `${n} absent ou invisible`);
    const ordre = pieces.map((x) => x.n);
    if (ordre.join() !== [...ATTENDUES, 'replier'].filter((n) => ordre.includes(n)).join()) fautes.push(`dans le désordre : ${ordre.join()}`);
    pieces.forEach((x, i) => {
      const r = x.r;
      if (r.width < 1 || r.height < 1) fautes.push(`${x.n} sans taille (${x.t})`);
      if (r.left < rp.left - 0.5 || r.right > rp.right + 0.5 || r.top < rp.top - 0.5 || r.bottom > rp.bottom + 0.5) fautes.push(`${x.n} sort de l'en-tête`);
      if (Math.abs((r.top + r.bottom) / 2 - cy) > 2) fautes.push(`${x.n} hors de la rangée`);
      if (i && r.left < pieces[i - 1].r.right - 0.5) fautes.push(`${pieces[i - 1].n} / ${x.n} se chevauchent`);
      if (doigt && Math.min(r.width, r.height) < doigt - 0.5) fautes.push(`${x.n} ${x.t} (moins de ${doigt} px au doigt)`);
    });
    if (ph.scrollWidth > ph.clientWidth + 1 || l.scrollWidth > l.clientWidth + 1) fautes.push('la ligne déborde de l\'en-tête');
    if (document.documentElement.scrollWidth > innerWidth) fautes.push('la page déborde');
    const n = l.querySelector('input.et-elt-nom') || l.querySelector('.et-elt-nom.lu > span');
    return { fautes, nomEntier: !!n && n.scrollWidth <= n.clientWidth + 1, pieces: pieces.map((x) => `${x.n} ${x.t}`).join(', '),
      entete: `${Math.round(rp.width)}×${Math.round(rp.height)}` };
  }, doigt);
  // la LISTE des éléments, le menu de cette ligne : l'ouvrir (un clic, ou un appui au doigt), puis ce que l'œil voit —
  // dans l'écran, chaque entrée atteinte (le point touché en son centre est bien elle : rien ne la couvre), les noms
  // (l'ouvert coché, marqué d'une étoile), « Ajouter … », le compte sur le bouton. Le menu reste ouvert : l'appelant le
  // referme (Échap, ou le bouton de nouveau).
  const LISTE_ELT = '.panel.left .phead .et-elt .et-elt-liste';
  const listeElement = async (pg, auDoigt = false) => {
    if (auDoigt) await pg.tap(`${LISTE_ELT} [data-menu]`); else await pg.click(`${LISTE_ELT} [data-menu]`);
    await pg.waitForTimeout(150);
    return pg.evaluate((s) => {
      const m = document.querySelector(s), l = m.querySelector('.et-mn-l'), q = l.getBoundingClientRect(), entrees = [...l.querySelectorAll('button')];
      const atteinte = (e) => { const r = e.getBoundingClientRect(), x = document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2); return !!x && e.contains(x); };
      return { vu: l.checkVisibility({ visibilityProperty: true, opacityProperty: true }), ouvert: m.querySelector('[data-menu]').getAttribute('aria-expanded') === 'true',
        dedans: q.left >= -0.5 && q.top >= -0.5 && q.right <= innerWidth + 0.5 && q.bottom <= innerHeight + 0.5, touche: entrees.length > 1 && entrees.every(atteinte),
        items: [...l.querySelectorAll('[data-activer]')].map((x) => x.querySelector('.nom').textContent + (x.classList.contains('on') && x.getAttribute('aria-checked') === 'true' ? '*' : '')).join(),
        ajouter: ((l.querySelector('[data-ajouter]') || {}).textContent || '').trim(), compte: (m.querySelector('[data-menu] .n') || {}).textContent || '',
        hMin: Math.round(Math.min(...entrees.map((x) => x.getBoundingClientRect().height))), boite: [q.left, q.top, q.right, q.bottom].map(Math.round).join(',') };
    }, LISTE_ELT);
  };
  const verdict = () => p.evaluate(() => document.querySelector('.verdict')?.textContent.replace(/\s+/g, ' ').trim() || '');
  const pret = (o) => pretEcran(p, o);
  const calme = () => p.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });

  // ---------------------------------------------------------------- l'accueil, vide
  await p.goto(`${BASE}/index.html`);
  await p.waitForTimeout(300);
  // la politique de sécurité de _headers est active (outils/serveur_local.py l'applique, comme Cloudflare) : un script
  // écrit dans la page ne s'exécute pas. Tout le parcours qui suit tourne donc SOUS cette politique.
  const nCsp = erreurs.length;
  const inline = await p.evaluate(() => { const s = document.createElement('script'); s.textContent = 'window.__inline = 1'; document.head.appendChild(s); return window.__inline; });
  await p.waitForTimeout(150);
  verif('politique de sécurité active : un script écrit dans la page est refusé', inline === undefined,
    inline === undefined ? '' : 'servir avec site/web/outils/serveur_local.py');
  erreurs.splice(nCsp, erreurs.length - nCsp, ...erreurs.slice(nCsp).filter((x) => !/Content Security Policy/.test(x)));
  verif('accueil vide : il le dit', await p.isVisible('.vide'));
  // les tuiles ouvertes sont exactement les modules du registre (js/modules.js) ; les autres disent « bientôt ». Une tuile
  // ouvre la fenêtre « Nouvelle note » (data-nouvelle-module) ou, pour l'établi compact (construire.ETABLI_COMPACT), le
  // calcul lui-même (data-module : un lien direct)
  // (les modules MASQUÉS — docs/site/BIBLIOTHEQUES.md § 2.1 — restent au registre, servis pour les notes existantes, mais
  //  n'ont plus de tuile ; la bibliothèque de la famille a la sienne, à part, et ouvre sa page)
  const registre = await p.evaluate(async () => Object.entries((await import('./js/modules.js')).MODULES).filter(([, m]) => !m.masque).map(([k]) => k).sort().join());
  const masques = await p.evaluate(async () => Object.entries((await import('./js/modules.js')).MODULES).filter(([, m]) => m.masque).map(([k]) => k).sort().join());
  const ouvertes = (await p.$$eval('.tile[data-nouvelle-module], .tile[data-module]',
    (t) => t.map((x) => x.dataset.nouvelleModule || x.dataset.module))).sort().join();
  const total = await p.$$eval('.tile:not(.biblio)', (t) => t.length);
  verif('accueil : les modules visibles du registre sont ouverts, les autres « bientôt », les masqués absents',
    ouvertes === registre && (await p.$$eval('.tile.bientot', (t) => t.length)) === total - registre.split(',').length, `${ouvertes} / ${registre} · masqués : ${masques}`);
  // (révision du 09/11/2026 : la carte est ÉTROITE et HAUTE, à gauche du bloc de sa famille, les modules dans leur grille à
  //  sa droite — la forme ici, la mesure à chaque format dans le filtre « bibliotheque »)
  const biblios = await p.$$eval('.tile.biblio', (t) => t.map((x) => ({ k: x.dataset.bibliotheque, href: x.getAttribute('href'), nom: x.querySelector('.nm').textContent.replace(/\s+/g, ' ').trim(),
    premiere: x === x.parentElement.firstElementChild && x.parentElement.matches('.famille > .tiles.fam-b'),
    droite: !!x.nextElementSibling && x.nextElementSibling.matches('.tiles') && x.nextElementSibling.querySelectorAll('.tile:not(.biblio)').length > 0 && !x.nextElementSibling.nextElementSibling })));
  // (BIB5 : la Bibliothèque Acier a sa carte aussi, à gauche du bloc Acier ; le Catalogue des profilés qu'elle porte est masqué)
  verif('accueil : les Bibliothèques Béton armé et Acier sont des cartes à part, à gauche du bloc de leur famille, leurs modules dans leur grille à leur droite, et visent leur page',
    biblios.length >= 2 && biblios.every((b) => b.premiere && b.droite && b.href === `bibliotheque-${b.k}`) && biblios.some((b) => b.k === 'beton' && b.nom === 'Bibliothèque Béton armé')
    && biblios.some((b) => b.k === 'acier' && b.nom === 'Bibliothèque Acier') && masques.split(',').includes('catalogue'), JSON.stringify(biblios));
  verif('accueil : le volet « Modules » du rail commence par les bibliothèques (Béton armé, Acier) et ne liste pas les masqués',
    await p.evaluate(async (masques) => { const labels = [...document.querySelectorAll('.modpanel .navlabel')].map((x) => x.textContent); const liens = [...document.querySelectorAll('.modpanel .navitem')].map((x) => x.getAttribute('href') || '');
      const MODS = (await import('./js/modules.js')).MODULES;
      return labels[0] === 'Bibliothèques' && liens.some((h) => h === 'bibliotheque-beton') && liens.some((h) => h === 'bibliotheque-acier')
        && masques.split(',').filter(Boolean).every((k) => !liens.includes(MODS[k].page) && !document.querySelector(`.mopt[data-module="${k}"]`)); }, masques));
  let m = await mesure();
  verif('accueil : rien ne déborde, aucun nom coupé', !m.deborde && !m.coupes.length, m.coupes.join(', '));
  await capture('accueil_vide');

  async function nouvelle(module, chantier, dossier, nom) {
    await p.goto(`${BASE}/index.html`);
    await p.click('#ouvrir-nouvelle');
    let m0 = null;
    verif(`« Nouvelle note » (${module}) : la fenêtre s'ouvre`, await p.isVisible('#nouvelle .mcard'));
    await p.fill('#nn-chantier', chantier); await p.fill('#nn-dossier', dossier); await p.fill('#nn-nom', nom);
    if (await p.$(`.mopt[data-module="${module}"]`)) {
      await p.click(`.mopt[data-module="${module}"]`);
      m0 = await montage(p);
      await Promise.all([p.waitForNavigation(), p.click('#nn-creer')]);
    } else {
      // un module MASQUÉ (js/modules.js « masque ») : plus d'entrée dans « Nouvelle note », mais sa page reste servie et
      // crée la note depuis l'adresse (js/etabli.js)
      const page = await p.evaluate(async (k) => (await import('./js/modules.js')).MODULES[k].page, module);
      await p.goto(`${BASE}/${page}?${new URLSearchParams({ nom, chantier, dossier })}`);
    }
    const t0 = Date.now();
    await pret({ apres: m0 });
    const echec = await echecEcran(p);
    verif(`${module} : le moteur démarre dans le navigateur`, !echec, echec || `${Date.now() - t0} ms`);
    verif(`${module} : le nom de la note et le chantier sont repris`, (await p.inputValue('#nom-note')) === nom && (await p.textContent('#fil-chantier')).includes(chantier));
  }

  if (!FILTRE || FILTRE === 'garde') {
    // ---------------------------------------------------------------- le garde-corps
    await nouvelle('garde_corps', 'Réalex', 'Garde-corps', 'Garde-corps terrasse R+1');
    let v0 = await verdict();
    verif('garde-corps : le résultat du moteur (64 %)', /64\s?%/.test(v0), v0);
    verif('garde-corps : une ligne par élément', (await p.$$eval('#resultat .el', (e) => e.length)) >= 5);
    m = await mesure();
    verif('garde-corps : le dessin est visible et à sa taille', m.svg && m.svg[0] > 400 && m.svg[1] > 300, JSON.stringify(m.svg));
    verif('garde-corps : rien ne déborde, aucun nom coupé', !m.deborde && !m.coupes.length, m.coupes.join(', '));
    verif('garde-corps : chaque valeur saisie s\'écrit à l\'encre', !m.gris.length, m.gris.join(', '));
    verif('garde-corps : un nom s\'écrit sans chiffres tabulaires (« Garde-corps 1 », pas « Garde - corps 1 »)', !m.tabulaires.length, m.tabulaires.join(', '));
    await capture('garde_corps');
    // une saisie
    await p.fill('input[data-chemin="geometrie/H"]', '1300');
    await p.press('input[data-chemin="geometrie/H"]', 'Enter');
    await calme();
    verif('H = 1 300 mm : le résultat suit (89 %)', /89\s?%/.test(await verdict()), await verdict());
    // le calcul détaillé
    // (l'établi compact, 11/11/2026 : la ligne porte aussi le bouton du profilé — on clique sur ce qu'elle compare)
    await p.click('#resultat .el[data-calcul="montant"] .cmp');
    await p.waitForTimeout(200);
    verif('la ligne du montant ouvre son calcul détaillé', await p.isVisible('.flyout.on.calcul .vtab'));
    await capture('garde_corps_calcul');
    await p.keyboard.press('Escape');
    // le sélecteur, et le plus léger qui tient
    await p.click('#resultat .prof[data-pk="montant"]');
    await p.waitForSelector('.flyout.on.picker .pk-row.cur', { timeout: 60000 });
    m = await mesure();
    verif('le sélecteur s\'ouvre sur le profilé retenu, aucun nom coupé', !m.coupes.length, m.coupes.join(', '));
    await capture('garde_corps_profils');
    const opt = await p.$('.flyout.on .pk-row.opt');
    if (opt) {
      const nomOpt = await opt.getAttribute('data-choix');
      await opt.click();
      await calme();
      verif(`choisir l'optimal (${nomOpt}) le retient`, (await p.textContent('#resultat .prof[data-pk="montant"] span')).trim() === nomOpt);
    }
    // le type change les éléments
    await p.selectOption('select[data-chemin="type"]', 'lisses');
    await calme();
    verif('type « lisses » : la ligne des lisses intermédiaires apparaît', /Lisses \(\d+\)/.test(await p.textContent('#resultat')));
    // l'établi compact dit les noms en COURT (« Montant M2 », « Lisses (2) ») : le nom, le profilé et le taux tiennent sur
    // une ligne du cadre de 316 px, sans être coupés (le profilé se coupait en « RHS 50×25×2,… », l'épaisseur cachée)
    m = await mesure();
    verif('type « lisses » : aucun nom coupé dans le résultat (le profilé des lisses se lit en entier)', !m.coupes.length && !m.deborde, m.coupes.join(', '));
    // un deuxième garde-corps (l'établi compact : la liste de la LIGNE de l'élément, dans l'en-tête du panneau des données)
    const nGardeCorps = () => p.$eval('.panel.left .phead .et-elt .et-elsel[data-elements]', (l) => Number(l.dataset.elements));
    await p.click(`${LISTE_ELT} [data-menu]`);
    await p.click(`${LISTE_ELT} .et-mn-l [data-ajouter]`);
    await calme();
    verif('« Ajouter un garde-corps » : deux éléments dans la note', (await nGardeCorps()) === 2);
    // l'enregistrement : la page rechargée retrouve la note
    await p.reload();
    await pret();
    verif('rechargée, la note garde ses deux garde-corps', (await nGardeCorps()) === 2);
    await p.click(`${LISTE_ELT} [data-menu]`);
    await p.click(`${LISTE_ELT} .et-mn-l [data-activer="0"]`);
    await calme();
    verif('… et le premier garde sa hauteur de 1 300 mm', (await p.inputValue('input[data-chemin="geometrie/H"]')).replace(/\s/g, '') === '1300');
    // sombre
    await p.click('.nav [data-theme-toggle]');
    await p.waitForTimeout(200);
    await capture('garde_corps_sombre');
    await p.click('.nav [data-theme-toggle]');
  }

  if (!FILTRE || FILTRE === 'poutre') {
    // ---------------------------------------------------------------- la poutre
    await nouvelle('poutre', 'Réalex', 'Étude de la dalle', 'Poutres R+1');
    // (établi compact, deuxième passe) plus de grand verdict : l'état d'une section est l'en-tête de son BLOC
    // (« Données incomplètes », « Conforme », « Non conforme »…), ses taux sont ceux de ses lignes
    const blocs = () => p.evaluate(() => [...document.querySelectorAll('#resultat .et-bloc:not([hidden])')].map((b) => `${b.querySelector('.et-bloc-h .e').textContent} : `
      + [...b.querySelectorAll('.el .pct')].map((x) => x.textContent.replace(/\s+/g, ' ').trim()).join(' ')).join(' | '));
    let v = await blocs();
    // (établi compact) une poutre neuve n'a aucun effort : « Données incomplètes » ; les cinq vérifications et les
    // dispositions sont listées, le tranchant et les étriers « non effectuée »
    const resNeuf = await p.textContent('#resultat');
    verif('poutre neuve : « Données incomplètes », six lignes, le tranchant « non effectuée »',
      v.startsWith('Données incomplètes') && (await p.$$eval('#resultat .el', (e) => e.length)) === 6
      && resNeuf.includes('non effectuée'), v);
    // les efforts de la section A : la flexion et l'effort tranchant entrent en jeu (rubrique « Sections et
    // sollicitations », les rubriques de la poutre ouverte — colonne de gauche compacte : plus d'arborescence)
    const ouvrir = async (k) => {
      if (!(await p.$eval(`#donnees .et-arbre .sec[data-sec="${k}"]`, (x) => x.classList.contains('open')))) await p.click(`#donnees .et-arbre .sec[data-sec="${k}"] > .sechd`);
      await p.waitForTimeout(100);
    };
    await ouvrir('so');
    for (const [ch, x] of [['sections/0/M_inf', '122'], ['sections/0/V', '60']]) {
      await p.fill(`input[data-chemin="${ch}"]`, x);
      await p.press(`input[data-chemin="${ch}"]`, 'Enter');
      await calme();
    }
    v = await blocs();
    const nEl = await p.$$eval('#resultat .el', (e) => e.length);
    const vSaisi = await p.inputValue('input[data-chemin="sections/0/V"]');
    const resEff = await p.textContent('#resultat');
    verif('M = 122 kN·m et V = 60 kN : cinq vérifications et les dispositions, le tranchant et les étriers calculés',
      nEl === 6 && resEff.includes('Effort tranchant') && !resEff.includes('non effectuée') && !v.startsWith('Données incomplètes')
      && (v.match(/\d+\s?%/g) || []).length === 5, `${nEl} lignes, V affiché ${vSaisi} · ${v}`);
    m = await mesure();
    verif('poutre : la coupe est visible et à sa taille', m.svg && m.svg[0] > 200 && m.svg[1] > 300, JSON.stringify(m.svg));
    verif('poutre : rien ne déborde, aucun nom coupé', !m.deborde && !m.coupes.length, m.coupes.join(', '));
    verif('poutre : chaque valeur saisie s\'écrit à l\'encre (efforts M et V compris)', !m.gris.length, m.gris.join(', '));
    await capture('poutre');
    const c = await p.$('#resultat .el[data-calcul]');
    if (c) {
      await c.click();
      await p.waitForTimeout(200);
      verif('poutre : une ligne ouvre le calcul de sa section', await p.isVisible('.flyout.on.calcul .vtab'));
      await capture('poutre_calcul');
      await p.keyboard.press('Escape');
    }
    const pk = await p.$('#resultat .prof[data-pk]');
    if (pk) {
      await pk.click();
      await p.waitForSelector('.flyout.on.picker .pk-row', { timeout: 60000 });
      verif('poutre : le choix des armatures s\'ouvre', await p.isVisible('.flyout.on.picker'));
      await capture('poutre_armatures');
      await p.keyboard.press('Escape');
    }
    await ouvrir('pt');
    const h = await p.$('input[data-chemin$="/h"], input[data-chemin="h"]');
    if (h) {
      const avant = await blocs();
      await h.fill('60');
      await h.press('Enter');
      await calme();
      verif('poutre : h = 60 cm change le résultat', (await blocs()) !== avant, `${avant} → ${await blocs()}`);
    }
  }

  const avecDessin = new Set();   // les pages des modules qui ont au moins un dessin : leur vignette en est un
  // ---------------------------------------------------------------- chaque module du registre (contrôle commun)
  // Le même contrôle pour TOUS les modules (js/modules.js, écrit depuis le registre) : la note se crée, le moteur
  // démarre, le résultat et le dessin s'affichent à leur taille, rien ne déborde ni n'est coupé, la fenêtre de calcul
  // et le sélecteur s'ouvrent, une saisie recalcule sans erreur et survit au rechargement.
  //   filtre « modules » : tous ; « module:dalle » : celui-là.
  if (!FILTRE || FILTRE === 'modules' || FILTRE.startsWith('module:')) {
    // la liste comme la page la lit : un module JavaScript (toujours en UTF-8), et non le fichier affiché en texte
    await p.goto(`${BASE}/`);
    const MODULES = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    verif('modules : les noms sont lus tels que la page les lit (accents intacts)', Object.values(MODULES).every((x) => !/[ÃÂ]/.test(x.nom)),
      Object.values(MODULES).map((x) => x.nom).join(', '));
    const choisis = FILTRE.startsWith('module:') ? [FILTRE.slice(7)] : Object.keys(MODULES);
    verif(`modules : ${choisis.join(', ')} au registre`, choisis.every((k) => MODULES[k]), choisis.filter((k) => !MODULES[k]).join());
    for (const cle of choisis.filter((k) => MODULES[k])) {
      const nom = MODULES[cle].nom;
      await nouvelle(cle, 'Contrôle', 'Tous les modules', `Essai ${nom}`);
      if (await echecEcran(p)) continue;
      if (MODULES[cle].vers) {
        // un module ABSORBÉ par un autre (js/modules.js « vers », construire.REDIRECTIONS ; docs/site/DALLES_FUSION.md
        // § 2.4) : sa page a REDIRIGÉ vers celle du module qui le sert, en gardant l'adresse — le nom et le chantier
        // demandés y sont repris (nouvelle() vient de le vérifier) ; la suite du contrôle commun se joue sur cette page.
        // (L'adresse lue ici est déjà « ?note=… » : l'établi remplace « ?nom=… » par la note qu'il vient de créer.)
        const u = new URL(p.url()), vers = MODULES[cle].vers;
        verif(`${cle} : la page redirige vers celle de « ${vers} » (${MODULES[vers].page}) en gardant l'adresse (la note créée du nom demandé)`,
          u.pathname.endsWith(`/${MODULES[vers].page}`) && !!u.searchParams.get('note') && (await p.inputValue('#nom-note')) === `Essai ${nom}`
          && (await p.evaluate(() => document.body.dataset.module)) === vers, `${u.pathname}${u.search}`);
      }
      // les vues : un dessin porte l'échelle, un tableau jamais ; le module a-t-il un dessin (pour sa vignette) ?
      const vuesLues = await p.$$eval('#stage .view', (vs) => vs.map((v) => ({ svg: !!v.querySelector('.draw > svg'), echelle: !!v.querySelector('.scale') })));
      if (vuesLues.some((v) => v.svg)) avecDessin.add(MODULES[cle].page);
      verif(`${cle} : « à l'échelle » n'est jamais écrit sur un tableau`, vuesLues.every((v) => v.svg || !v.echelle), JSON.stringify(vuesLues));
      const nEl = await p.$$eval('#resultat .el', (e) => e.length);
      const verdictMod = await p.$('#resultat .verdict');
      verif(`${cle} : le résultat s'affiche`, nEl > 0 || !!verdictMod, `${nEl} ligne(s)`);
      const nVues = await p.$$eval('#vues button', (e) => e.length);
      m = await mesure();
      if (nVues) verif(`${cle} : la vue (dessin ou tableau) est visible et à sa taille`, m.vue && m.vue[0] > 200 && m.vue[1] > 120, JSON.stringify(m.vue));
      verif(`${cle} : rien ne déborde, aucun nom coupé, les valeurs à l'encre, les noms sans chiffres tabulaires`,
        !m.deborde && !m.coupes.length && !m.gris.length && !m.tabulaires.length, [...m.coupes, ...m.gris, ...m.tabulaires].join(', '));
      if (REFONTE_1111.includes(cle) && (await p.evaluate(() => document.body.dataset.etabli)) === 'compact') {
        const lus = await lisibles(p);
        verif(`${cle} : dans chaque rubrique des données, chaque valeur saisie se lit en entier dans son champ ; à droite, chaque comparaison se lit en entier`,
          lus.rubriques > 0 && !lus.champs.length && !lus.cmps.length, JSON.stringify(lus).slice(0, 500));
      }
      await capture(cle);
      const c = await p.$('#resultat [data-calcul]');
      if (c) {
        await c.click();
        await p.waitForTimeout(250);
        verif(`${cle} : une ligne ouvre son calcul détaillé`, await p.isVisible('.flyout.on.calcul'));
        m = await mesure();
        verif(`${cle} : la fenêtre de calcul n'a aucun nom coupé`, !m.coupes.length, m.coupes.join(', '));
        await capture(`${cle}_calcul`);
        await p.keyboard.press('Escape');
      }
      const pk = await p.$('#resultat .prof[data-pk]');
      if (pk) {
        await pk.click();
        await p.waitForSelector('.flyout.on.picker', { timeout: 60000 });
        verif(`${cle} : le sélecteur s'ouvre`, await p.isVisible('.flyout.on.picker .pk-row, .flyout.on.picker .pk-vide'));
        await capture(`${cle}_selecteur`);
        await p.keyboard.press('Escape');
      }
      // une saisie : le premier nombre modifiable du cadre Données, augmenté de 10 %. Sur l'établi compact ce champ peut
      // être dans une rubrique REPLIÉE (la maturité : la rubrique 1 « Béton » n'a que des menus, une date et une heure ;
      // le premier nombre, la température, est dans la rubrique 2) : on l'ouvre d'abord, comme le fait l'ingénieur.
      const champ = await p.$('#donnees input[data-sorte="nombre"]:not([disabled])');
      if (champ) {
        const chemin = await champ.getAttribute('data-chemin');
        const rubrique = await champ.evaluate((e) => { const s = e.closest('#donnees .sec[data-sec]'); return s && !s.classList.contains('open') ? s.dataset.sec : null; });
        if (rubrique) {
          await p.click(`#donnees .sec[data-sec="${rubrique}"] > .sechd`);
          await p.waitForTimeout(150);
          verif(`${cle} : la rubrique repliée qui porte le premier nombre (${chemin}) s'ouvre d'un clic sur son titre, le champ se voit`,
            await champ.evaluate((e) => e.isConnected && e.checkVisibility({ visibilityProperty: true, opacityProperty: true })), rubrique);
        }
        const avant = (await champ.inputValue()).replace(/[\s  ]/g, '').replace(',', '.');
        const x = Number(avant);
        const neuf = Number.isFinite(x) && x !== 0 ? String(Math.round(x * 1.1 * 1000) / 1000).replace('.', ',') : '1';
        await champ.fill(neuf);
        await champ.press('Enter');
        await calme();
        await p.waitForTimeout(150);
        const refus = await p.evaluate(() => { const t = document.getElementById('toast'); return t && !t.hidden && t.classList.contains('erreur') ? t.textContent : ''; });
        const apres = (await p.inputValue(`#donnees input[data-chemin="${chemin}"]`)).replace(/[\s  ]/g, '');
        verif(`${cle} : une saisie (${chemin} = ${neuf}) recalcule sans erreur`, !refus && apres === neuf.replace(/[\s  ]/g, ''), refus || `${avant} → ${apres}`);
        await p.reload();
        await pret();
        await calme();
        const relu = (await p.inputValue(`#donnees input[data-chemin="${chemin}"]`)).replace(/[\s  ]/g, '');
        verif(`${cle} : rechargée, la note garde la saisie`, relu === apres, `${apres} → ${relu}`);
      }
    }
  }

  // ---------------------------------------------------------------- les FORMATS : téléphone et tablettes
  // Chaque module, au téléphone (390 × 844, tactile), en tablette portrait (820 × 1 180), en tablette paysage
  // (1 180 × 820), en petite tablette paysage (1 024 × 768) et au petit portable (1 280 × 800) : chaque cadre atteignable et
  // VISIBLE à sa taille, la toile assez grande, rien qui déborde, aucun nom coupé, la fenêtre de calcul et le sélecteur
  // dans l'écran, le zoom qui grossit le dessin, la légende de chaque vue lisible, le cartouche sans chevauchement.
  // Seuils de la toile (la place offerte au dessin, mesurée à l'écran) : 300 px de large au téléphone (la largeur moins
  // les marges de la feuille : en dessous, un dessin de 1 m de dalle s'écrit en 6 px), 500 × 300 en tablette portrait
  // (la toile y avait 62 px), 600 × 400 en tablette paysage 1 180 (450 px dans la disposition du PC), 500 × 400 à
  // 1 024 × 768, 400 × 400 au petit portable 1 280 × 800 (la disposition du PC, les deux cadres ouverts autour).
  //   filtre « formats » : tous les modules ; « formats:dalle » : celui-là. FORMATS=tel,tablettes,communs,numpy,accueil
  //   (variable d'environnement) n'en joue qu'une partie — pour prouver une correction par mutation sans tout rejouer.
  const PARTIES = (process.env.FORMATS || 'accueil,tel,tablettes,communs,numpy').split(',');
  if (!FILTRE || FILTRE === 'formats' || FILTRE.startsWith('formats:')) {
    await p.goto(`${BASE}/`);
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const choisis = FILTRE.startsWith('formats:') ? [FILTRE.slice(8)] : Object.keys(MODS);
    // le travailleur, vu de la page : le proxy des réponses (proxyTravailleur, en tête du fichier : simuler, retoucher,
    // retarder, et le compte __interceptees)
    // un contrôle qui s'appuie sur le proxy (simuler, retoucher) : KO « proxy inactif » si le proxy n'a vu passer AUCUNE
    // réponse du vrai travailleur (n : le compte lu là où la retouche a joué, ou dans la page courante)
    const verifProxy = async (pg, nom, cond, info, n = null) => {
      const vues = n === null ? await interceptees(pg) : n;
      verif(nom, vues > 0 && cond, vues > 0 ? info : `proxy inactif (aucune réponse du travailleur interceptée) · ${info}`);
    };
    const contexte = async (w, h, tel) => {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel, acceptDownloads: true });
      await c.addInitScript(proxyTravailleur);
      const pg = await c.newPage();
      pg.erreurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') pg.erreurs.push(m.text().slice(0, 300)); });
      pg.requetes = [];
      c.on('request', (r) => pg.requetes.push(r.url()));
      return pg;
    };
    const vis = (pg, s) => pg.evaluate((s) => { const e = document.querySelector(s); return !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true }); }, s);
    // (petits chantiers, 10/11/2026) AU DOIGT, 44 px : chaque bouton VU de l'établi (la barre du haut, l'en-tête du téléphone,
    // les onglets, la barre de la toile, les en-têtes des panneaux, les données, le résultat), mesuré à l'écran — sa plus
    // petite dimension. Les rubriques des données sont ouvertes avant (chaque bouton se voit). Rend les fautes.
    const cibles44 = (pg) => pg.evaluate(() => {
      const vu = (e) => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      document.querySelectorAll('#donnees .sec:not(.open) > .sechd').forEach((h) => h.click());
      const out = [];
      for (const e of document.querySelectorAll('.top button, .top .btn, .mtop button, .mtop .btn, .mtabs button, .ctool button, .phead button, #donnees button, #donnees .btn, #resultat button, #resultat .btn, .mbar .btn')) {
        if (!vu(e)) continue;
        const r = e.getBoundingClientRect();
        if (r.width < 1 || Math.min(r.width, r.height) >= 43.5) continue;
        out.push(`${(e.className || e.tagName).toString().split(' ')[0] || e.tagName} « ${(e.textContent || e.title || '').trim().slice(0, 16)} » ${Math.round(r.width)}×${Math.round(r.height)}`);
      }
      return out;
    });
    const boite = (pg, s) => pg.evaluate((s) => {
      const e = [...document.querySelectorAll(s)].find((x) => x.checkVisibility({ visibilityProperty: true, opacityProperty: true }));
      if (!e) return null;
      const r = e.getBoundingClientRect();
      return { g: Math.round(r.left), h: Math.round(r.top), l: Math.round(r.width), t: Math.round(r.height), d: Math.round(r.right), b: Math.round(r.bottom), W: innerWidth, H: innerHeight };
    }, s);
    const dansEcran = (r) => r && r.g >= -1 && r.h >= -1 && r.d <= r.W + 1 && r.b <= r.H + 1;
    // ce que l'œil voit, comme mesure() : débordement, noms coupés ; et le contenu de la vue ouverte
    const mesurer = (pg) => pg.evaluate(() => {
      const vis = (e) => e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const large = (e) => { const r = document.createRange(); r.selectNodeContents(e); return r.getBoundingClientRect().width; };
      // (et les libellés à deux champs par ligne, .row2 : données condensées, 11/2026)
      const coupes = [...document.querySelectorAll('.prof > span:not(.leger), .el .l1 b, .row > label > span:not(.sym), .row2 > label > span, .ni .t, .pk-row .nm, .tile .nm, .rows .seg.full button, .mtabs button, .mbar .btn, .mmenu .mi')]
        .filter(vis).filter((e) => e.scrollWidth > e.clientWidth + 1 || large(e) > e.getBoundingClientRect().width + 0.01).map((e) => e.textContent.trim().slice(0, 40));
      // la grande pastille du verdict : son texte tient dans sa boîte (une pastille comprimée coupait « η »)
      const xl = [...document.querySelectorAll('.verdict .chip.xl')].filter(vis).filter((e) => large(e) > e.getBoundingClientRect().width + 0.5).map((e) => e.textContent);
      return { deborde: document.documentElement.scrollWidth > innerWidth, coupes, xl };
    });
    // la place offerte au dessin (la zone .draw de la vue ouverte) et le dessin lui-même (son viewBox projeté à l'écran)
    const toile = (pg) => pg.evaluate(() => {
      const d = [...document.querySelectorAll('.view.on .draw')].find((e) => e.checkVisibility());
      if (!d) return null;
      const r = d.getBoundingClientRect();
      const svg = d.querySelector(':scope > svg');
      let dessin = null, texte = null;
      if (svg && svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width) {
        const m = svg.getScreenCTM(), vb = svg.viewBox.baseVal;
        dessin = [Math.round(vb.width * m.a), Math.round(vb.height * m.d)];
        const t = [...svg.querySelectorAll('text')].map((x) => x.getBoundingClientRect().height).filter((x) => x > 0).sort((a, b) => a - b);
        texte = t.length ? Math.round(t[Math.floor(t.length / 2)] * 10) / 10 : null;
      }
      return { l: Math.round(r.width), t: Math.round(r.height), dessin, texte, svg: !!svg };
    });
    // chaque vue : sa légende visible et entière (ni coupée, ni cachée) ; une vue qui est un tableau MONTRE ses cases
    // (comptées à travers chaque parent qui les rogne : un tableau présent dans la page mais dans une zone de 0 px de
    // haut n'est pas vu)
    const legendes = (pg) => pg.evaluate(async () => {
      const out = [];
      const vu = (e, borne) => {
        let r = e.getBoundingClientRect();
        if (r.width < 1 || r.height < 1 || !e.checkVisibility({ visibilityProperty: true, opacityProperty: true })) return false;
        for (let a = e.parentElement; a && a !== borne.parentElement; a = a.parentElement) {
          if (getComputedStyle(a).overflow === 'visible') continue;
          const p = a.getBoundingClientRect();
          r = { left: Math.max(r.left, p.left), top: Math.max(r.top, p.top), right: Math.min(r.right, p.right), bottom: Math.min(r.bottom, p.bottom) };
          if (r.right - r.left < 1 || r.bottom - r.top < 1) return false;
        }
        return r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
      };
      for (const b of [...document.querySelectorAll('#vues [data-vue]')]) {
        b.click();
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        const l = document.querySelector('.view.on .stitle .leg');
        if (l) {
          const ok = l.checkVisibility({ visibilityProperty: true, opacityProperty: true }) && l.getBoundingClientRect().width > 20
            && l.scrollWidth <= l.clientWidth + 1 && l.scrollHeight <= l.clientHeight + 1;
          if (!ok) out.push(`${b.dataset.vue} : « ${l.textContent.slice(0, 40)} »`);
        }
        const d = [...document.querySelectorAll('.view.on .draw')].find((e) => e.checkVisibility());
        if (d && !d.querySelector(':scope > svg')) {
          // un tableau : ses CASES à l'écran (sa légende et son pied sont dans le même bloc, ils ne prouvent rien) ;
          // un bloc sans tableau (le message d'une note vide, un texte) : ses textes à l'écran
          // un bloc qui COMMENCE par du texte (une méthode rédigée, puis des tableaux plus bas) se juge comme de la prose :
          // son premier tableau peut être sous le pli et se lire en défilant (vue « méthode » du plancher)
          const t0 = document.createTreeWalker(d, NodeFilter.SHOW_TEXT);
          let premier = t0.nextNode();
          while (premier && !premier.data.trim()) premier = t0.nextNode();
          const tableau = !!d.querySelector('td') && !!premier && !!premier.parentElement.closest('table');
          const w = document.createTreeWalker(tableau ? d.querySelector('table') : d, NodeFilter.SHOW_TEXT);
          let total = 0, lus = 0;
          for (let n = w.nextNode(); n; n = w.nextNode()) {
            if (tableau && !n.parentElement.closest('td')) continue;
            if (!n.data.trim()) continue;
            total++;
            const rg = document.createRange(); rg.selectNodeContents(n);
            const r = rg.getBoundingClientRect();
            const boite = { getBoundingClientRect: () => r, checkVisibility: () => n.parentElement.checkVisibility({ visibilityProperty: true, opacityProperty: true }), parentElement: n.parentElement };
            if (vu(boite, d)) lus++;
          }
          // six cases d'un tableau ; trois textes d'un bloc de prose (un long texte se lit en défilant)
          if (lus < Math.min(tableau ? 6 : 3, total) || !total) out.push(`${b.dataset.vue} : contenu invisible (${lus} texte(s) vu(s) sur ${total})`);
        }
      }
      document.querySelector('#vues [data-vue]')?.click();
      return out;
    });
    // la barre de la toile : les vues et le zoom ne passent sous aucun cadre (le point touché est bien dans la barre)
    const barreToile = (pg) => pg.evaluate(() => {
      const f = [];
      for (const e of [document.getElementById('vues'), ...document.querySelectorAll('#zoom button')]) {
        if (!e.checkVisibility({ visibilityProperty: true, opacityProperty: true })) continue;
        const r = e.getBoundingClientRect();
        for (const x of [r.left + 4, (r.left + r.right) / 2, r.right - 4]) {
          const t = document.elementFromPoint(x, (r.top + r.bottom) / 2);
          if (!t || !t.closest('.ctool')) f.push(`${e.id || e.dataset.zoom || e.className} couvert par ${t ? (t.className || t.tagName) : 'rien'}`);
        }
      }
      return [...new Set(f)];
    });
    // le cartouche : ses cases ne se chevauchent pas, l'indice et la date se lisent en entier
    const cartouche = (pg) => pg.evaluate(() => {
      const c = [...document.querySelectorAll('.view.on .cart')].find((e) => e.checkVisibility());
      if (!c) return ['pas de cartouche visible'];
      const cases = [...c.children].filter((e) => e.checkVisibility({ visibilityProperty: true }));
      const fautes = [];
      cases.forEach((e, i) => {
        const r = e.getBoundingClientRect();
        if (i && r.left < cases[i - 1].getBoundingClientRect().right - 0.5) fautes.push(`${e.textContent.trim()} chevauche`);
        if (r.right > c.getBoundingClientRect().right + 0.5) fautes.push(`${e.textContent.trim()} sort du cartouche`);
        for (const x of e.children) if ((e.classList.contains('dt') || e.classList.contains('ind') || x.tagName === 'SPAN') && x.scrollWidth > x.clientWidth + 1) fautes.push(`« ${x.textContent} » coupé`);
      });
      // le nom de la note garde une place lisible (un chantier au nom long le réduisait à « P… »)
      const n0 = cases[0] && cases[0].querySelector('b');
      if (n0 && n0.getBoundingClientRect().width < 60) fautes.push(`la note réduite à ${Math.round(n0.getBoundingClientRect().width)} px`);
      return fautes;
    });
    // le zoom : « + » grossit le dessin (le texte du dessin, lu à l'écran), « ajuster » le remet à sa feuille
    const zoom = async (pg) => {
      // sur la première vue qui est un dessin (un tableau ne se zoome pas : il défile)
      const k = await pg.evaluate(() => { const v = [...document.querySelectorAll('#stage .view')].find((x) => x.querySelector('.draw > svg')); return v ? v.dataset.vue : null; });
      if (!k) return { saute: true };
      await pg.click(`#vues [data-vue="${k}"]`);
      const lire = () => pg.evaluate(() => { const t = [...document.querySelectorAll('.view.on .draw > svg text')].find((x) => x.getBoundingClientRect().height > 0); return t ? t.getBoundingClientRect().height : document.querySelector('.view.on .draw > svg').getBoundingClientRect().width; });
      const a = await lire();
      await pg.click('#zoom [data-zoom="plus"]'); await pg.click('#zoom [data-zoom="plus"]');
      await pg.waitForTimeout(100);
      const z = await lire();
      const pct = await pg.textContent('#zoom .zpct');
      await pg.click('#zoom [data-zoom="ajuster"]');
      await pg.waitForTimeout(100);
      const r = await lire();
      await pg.click('#vues [data-vue]');
      return { a, z, r, pct, ok: z >= a * 1.4 && Math.abs(r - a) < 0.5 };
    };
    // la fenêtre de calcul et le sélecteur s'ouvrent et tiennent dans l'écran
    const volets = async (pg, cle, fmt) => {
      const c = await pg.$('#resultat [data-calcul]');
      if (c) {
        // on touche le NOM de la ligne : la pièce qui s'y choisit (son sélecteur) est une cible de 44 px au doigt, et le
        // milieu de la ligne tombe désormais sur elle (petits chantiers, 10/11/2026)
        const nom = await c.$('.l1 > b');
        await (nom || c).click();
        await pg.waitForTimeout(250);
        const r = await boite(pg, '.flyout.on.calcul');
        verif(`${cle} · ${fmt} : la fenêtre de calcul s'ouvre et tient dans l'écran`, dansEcran(r) && r.l > 280 && r.t > 200, JSON.stringify(r));
        const m = await mesurer(pg);
        verif(`${cle} · ${fmt} : la fenêtre de calcul ne déborde pas`, !m.deborde, JSON.stringify(m));
        if (fmt === 'téléphone') await pg.screenshot({ path: path.join(OUT, `site_${cle}_tel_calcul.png`) });
        await pg.click('.flyout.on [data-fermer]');
        await pg.waitForTimeout(150);
        verif(`${cle} · ${fmt} : la fenêtre de calcul se referme`, !(await vis(pg, '.flyout.on')));
      }
      const pk = await pg.$('#resultat .prof[data-pk]');
      if (pk) {
        await pk.click();
        await pg.waitForSelector('.flyout.on.picker', { timeout: 60000 });
        await pg.waitForTimeout(200);
        const r = await boite(pg, '.flyout.on.picker');
        const lignes = await pg.$$eval('.flyout.on.picker .pk-row', (x) => x.filter((e) => e.checkVisibility()).length);
        verif(`${cle} · ${fmt} : le sélecteur s'ouvre et tient dans l'écran`, dansEcran(r) && (lignes > 0 || await vis(pg, '.flyout.on .pk-vide')), JSON.stringify({ ...r, lignes }));
        if (fmt === 'téléphone') await pg.screenshot({ path: path.join(OUT, `site_${cle}_tel_selecteur.png`) });
        await pg.click('.flyout.on [data-fermer]');
        await pg.waitForTimeout(150);
      }
    };
    // une note neuve, par l'accueil (la fenêtre « Nouvelle note » à ce format)
    const neuve = async (pg, cle, fmt) => {
      await pg.goto(`${BASE}/index.html`);
      await pg.click('#ouvrir-nouvelle');
      const mc = await boite(pg, '#nouvelle .mcard');
      verif(`${cle} · ${fmt} : « Nouvelle note » s'ouvre dans l'écran`, dansEcran(mc), JSON.stringify(mc));
      // un chantier au nom long : le fil d'Ariane, l'en-tête du téléphone et le cartouche doivent le tenir (c'est lui qui,
      // en écrasant la case « Note », faisait chevaucher « Note » et « Chantier »)
      const chantier = 'Résidence Les Tilleuls, bâtiment B — phase 2 (logements)', nom = `${MODS[cle].nom} · ${fmt}`;
      await pg.fill('#nn-chantier', chantier); await pg.fill('#nn-nom', nom);
      let m0 = null;
      if (MODS[cle].masque) {
        // un module MASQUÉ (js/modules.js « masque ») n'a plus d'entrée dans « Nouvelle note » : sa page, toujours servie,
        // crée la note depuis l'adresse (js/etabli.js) — le même repli que nouvelle() plus haut
        await pg.goto(`${BASE}/${MODS[cle].page}?${new URLSearchParams({ nom, chantier })}`);
      } else {
        await pg.click(`.mopt[data-module="${cle}"]`);
        m0 = await montage(pg);
        await Promise.all([pg.waitForNavigation(), pg.click('#nn-creer')]);
      }
      await pretEcran(pg, { apres: m0 });
      const ko = await echecEcran(pg);
      verif(`${cle} · ${fmt} : le moteur démarre`, !ko, ko);
      return !ko;
    };

    // « Reprendre » compacte : six notes semées dans ce navigateur, puis le bloc MESURÉ à l'écran. Les grandes cartes de la
    // maquette (vignette de 150 px au-dessus du nom) prenaient ~460 px au PC pour six notes ; le bureau les a voulues
    // « deux fois plus petites et plus élégantes ».
    const reprendre = async (pg) => {
      await pg.evaluate((mods) => {
        const notes = {}, t = Date.now();
        const svg = '<svg viewBox="0 0 100 60" xmlns="http://www.w3.org/2000/svg"><rect x="6" y="22" width="88" height="16" fill="none" stroke="#7a8699"/></svg>';
        mods.slice(0, 6).forEach((m, i) => {
          const id = `reprendre-${i}`;
          notes[id] = { id, module: m, nom: `Note ${i + 1} — un nom assez long pour devoir être coupé`, chantier: 'Résidence Les Tilleuls, bâtiment B', dossier: 'Phase 2',
            indice: '', cree: t - i * 6e4, modifie: t - i * 6e4, etat: null, resume: { taux: 63.7 + 7 * i, niveau: 'ok' }, vignette: svg };
        });
        localStorage.setItem('stab.notes.v1', JSON.stringify(notes));
      }, Object.keys(MODS));
      await pg.reload();
      await pg.waitForTimeout(300);
      return pg.evaluate(() => {
        const vu = (e) => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const cs = [...document.querySelectorAll('#recentes .rcard')];
        const bloc = document.getElementById('recentes').getBoundingClientRect();
        const r = (e) => e.getBoundingClientRect();
        return {
          n: cs.length,
          h: Math.max(...cs.map((c) => Math.round(r(c).height))),
          bloc: Math.round(bloc.height),
          // les cartes ENTIÈREMENT visibles, sur combien de lignes et de colonnes
          lignes: new Set(cs.filter((c) => r(c).right <= innerWidth + 0.5).map((c) => Math.round(r(c).top))).size,
          colonnes: new Set(cs.filter((c) => r(c).right <= innerWidth + 0.5).map((c) => Math.round(r(c).left))).size,
          // la vignette à gauche, moins de la moitié de la carte ; le nom, la pastille et le ✕ dans la carte
          gauche: cs.every((c) => { const t = r(c.querySelector('.thumb')), k = r(c); return t.left - k.left < 2 && t.width < k.width / 2 && vu(c.querySelector('.thumb svg')); }),
          dedans: cs.every((c) => ['.t', '.chip', '.supp'].every((s) => { const e = c.querySelector(s), a = r(e), k = r(c); return vu(e) && a.right <= k.right + 0.5 && a.bottom <= k.bottom + 0.5; })),
          deborde: document.documentElement.scrollWidth > innerWidth,
        };
      });
    };
    if (PARTIES.includes('accueil')) {
      for (const [fmt, w, h, tel, hmax, cols, lignes, blocmax] of [['PC', 1440, 900, false, 76, 3, 2, 165], ['tablette', 820, 1180, false, 76, 2, 3, 250],
        ['téléphone', 390, 844, true, 66, 1, 2, 150]]) {
        const g = await contexte(w, h, tel);
        await g.goto(`${BASE}/index.html`);
        const q = await reprendre(g);
        verif(`accueil · ${fmt} : « Reprendre » compacte — des cartes basses (≤ ${hmax} px), la vignette à gauche, ${cols} par ligne, le bloc ≤ ${blocmax} px`,
          q.n === 6 && q.h <= hmax && q.colonnes === cols && q.lignes === lignes && q.bloc <= blocmax && q.gauche && q.dedans && !q.deborde, JSON.stringify(q));
        await g.screenshot({ path: path.join(OUT, `site_accueil_reprendre_${fmt === 'téléphone' ? 'tel' : fmt.toLowerCase()}.png`) });
        verif(`accueil · ${fmt} : « Reprendre » sans erreur JavaScript`, !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
        await g.context().close();
      }
    }

    // ------------------------------------------------ l'accueil au téléphone
    if (PARTIES.includes('accueil')) {
      const t = await contexte(390, 844, true);
      await t.goto(`${BASE}/index.html`);
      await t.waitForTimeout(300);
      const acc = { mhead: await boite(t, '.mhead'), mnav: await boite(t, '.mnav'), bouton: await boite(t, '#ouvrir-nouvelle'), rail: await vis(t, '.nav') };
      const m = await mesurer(t);
      verif('accueil · téléphone : l\'en-tête en haut, les cinq entrées en bas, « Nouvelle note » à la taille du doigt, rien ne déborde',
        acc.mhead && acc.mhead.h === 0 && acc.mnav && acc.mnav.b === acc.mnav.H && (await t.$$eval('.mnav a', (x) => x.length)) === 5 && acc.bouton && acc.bouton.t >= 44
        && !acc.rail && !m.deborde && !m.coupes.length, JSON.stringify({ ...acc, ...m }));
      await t.screenshot({ path: path.join(OUT, 'site_accueil_tel.png') });
      await t.click('.mnav [data-modules]');
      await t.waitForTimeout(200);
      const mp = await boite(t, '.modpanel');
      verif('accueil · téléphone : « Modules » ouvre le volet des modules dans l\'écran', dansEcran(mp) && mp.g === 0, JSON.stringify(mp));
      await t.context().close();
    }

    for (const cle of choisis.filter((k) => MODS[k])) {
      // ------------------------------------------------ le téléphone
      const t = await contexte(390, 844, true);
      if (PARTIES.includes('tel') && await neuve(t, cle, 'téléphone')) {
        const fixe = async () => ({ mtop: await boite(t, '.mtop'), mtabs: await boite(t, '.mtabs'), mbar: await boite(t, '.mbar') });
        // l'établi compact (construire.ETABLI_COMPACT, finalisé) : trois cadres comme les autres modules — « Données »
        // montre l'élément ouvert ET ses rubriques (le panneau de gauche ; colonne compacte, la poutre et la dalle —
        // docs/site/DALLES_FUSION.md § 1 : la LIGNE de l'élément dans l'en-tête du panneau, les rubriques dessous),
        // « Résultat » le résultat seul. La ligne est exigée dès que la page en a l'emplacement (.et-elt, construire.compact) :
        // un script qui ne l'y poserait pas rougit ici.
        const compactTel = await t.evaluate(() => document.body.dataset.etabli === 'compact');
        const ligneTel = await t.evaluate(() => !!document.querySelector('.panel.left .phead .et-elt'));
        const ONGLETS = compactTel
          ? [['donnees', '.panel.left', ['.canvas', '.panel.right'], ['#donnees .et-arbre', '#donnees .et-arbre .sec.open', ...(ligneTel ? ['.panel.left .phead .et-elt .et-elsel'] : [])]],
            ['dessin', '.canvas', ['.panel.left', '.panel.right'], []],
            ['resultat', '.panel.right', ['.canvas', '.panel.left'], ['#resultat .el']]]
          : [['donnees', '.panel.left', ['.canvas', '.panel.right'], []], ['dessin', '.canvas', ['.panel.left', '.panel.right'], []],
            ['resultat', '.panel.right', ['.canvas', '.panel.left'], []]];
        for (const [onglet, cadre, autres, aussi] of ONGLETS) {
         try {
          await t.click(`.mtabs [data-onglet="${onglet}"]`);
          await t.waitForTimeout(250);
          const r = await boite(t, cadre);
          const caches = [], manquent = [];
          for (const a of autres) if (await vis(t, a)) caches.push(a);
          for (const a of aussi) if (!(await vis(t, a))) manquent.push(a);
          verif(`${cle} · téléphone : l'onglet ${onglet} montre son cadre, sur toute la largeur, et lui seul${aussi.length ? ` (avec ${aussi.join(', ')})` : ''}`,
            r && r.l >= 370 && r.t >= 200 && !caches.length && !manquent.length, JSON.stringify({ r, caches, manquent }));
          const c44 = await cibles44(t);
          await t.waitForTimeout(150);
          verif(`${cle} · téléphone · ${onglet} : chaque bouton vu à 44 px au moins au doigt`, !c44.length, c44.slice(0, 6).join(' · '));
          const m = await mesurer(t);
          verif(`${cle} · téléphone · ${onglet} : rien ne déborde, aucun nom coupé`, !m.deborde && !m.coupes.length && !m.xl.length, JSON.stringify(m));
          await t.screenshot({ path: path.join(OUT, `site_${cle}_tel_${onglet}.png`) });
          // l'en-tête et les onglets restent en haut, la barre en bas, même la page défilée
          await t.evaluate(() => scrollTo(0, 700));
          await t.waitForTimeout(120);
          const f = await fixe();
          verif(`${cle} · téléphone · ${onglet} : l'en-tête et les onglets restent en haut, la barre en bas (page défilée)`,
            f.mtop && f.mtop.h === 0 && f.mtabs && f.mtabs.h >= f.mtop.b - 1 && f.mtabs.h <= f.mtop.b + 1 && f.mbar && f.mbar.b === f.mbar.H, JSON.stringify(f));
          await t.evaluate(() => scrollTo(0, 0));
          if (onglet === 'donnees') {
            // des champs en 16 px au moins (sinon iOS agrandit la page au toucher), des cibles de 36 px au moins (et le nom de
            // l'élément, un champ de l'en-tête des données : colonne compacte)
            const petits = await t.$$eval('#donnees input:not([type=checkbox]):not([type=file]), #donnees select, #donnees textarea, .panel.left .phead input',
              (x) => x.filter((e) => e.checkVisibility()).filter((e) => parseFloat(getComputedStyle(e).fontSize) < 16).map((e) => `${e.dataset.chemin} ${getComputedStyle(e).fontSize}`));
            verif(`${cle} · téléphone : chaque champ s'écrit en 16 px au moins`, !petits.length, petits.slice(0, 4).join(', '));
            const lg = await ligneElement(t, 44);
            if (lg) {
              // (colonne compacte : la poutre, la dalle) la LIGNE de l'élément dans l'en-tête des données — une rangée, chaque
              // bouton à 44 px au doigt (40 avant les petits chantiers du 10/11/2026 : le harnais SUIT) ; sa liste s'ouvre dans
              // l'écran, chaque entrée atteinte ; un appui sur le nom y pose le curseur sans quitter l'onglet, Échap rend
              // l'ancien nom (R1 : renommer, ouvrir, ajouter au téléphone aussi)
              verif(`${cle} · téléphone : la ligne de l'élément tient dans l'en-tête des données (une rangée, rien ne déborde ni ne se chevauche, le nom entier), chaque bouton à 44 px au doigt`,
                !lg.fautes.length && lg.nomEntier, JSON.stringify(lg));
              const mn = await listeElement(t, true);
              await t.screenshot({ path: path.join(OUT, `site_${cle}_tel_ligne_element.png`) });
              await t.tap(`${LISTE_ELT} [data-menu]`);
              await t.waitForTimeout(150);
              const ferme = await t.$eval(`${LISTE_ELT} .et-mn-l`, (x) => x.hidden);
              verif(`${cle} · téléphone : la liste des éléments s'ouvre dans l'écran (l'ouvert coché, « Ajouter … »), chaque entrée atteinte, à 44 px au doigt ; le chevron la referme`,
                mn.vu && mn.ouvert && mn.dedans && mn.touche && /\*$/.test(mn.items) && /^Ajouter /.test(mn.ajouter) && mn.hMin >= 44 && ferme, JSON.stringify({ ...mn, ferme }));
              const NOM = '.panel.left .phead .et-elt input.et-elt-nom';
              const avant = await t.$eval(NOM, (x) => x.value);
              await t.tap(NOM);
              await t.waitForTimeout(200);
              const foc = await t.evaluate((s) => { const a = document.activeElement, r = a.getBoundingClientRect();
                return { champ: a.matches(s), onglet: document.querySelector('.bench').dataset.tab, dansEcran: r.top >= 0 && r.bottom <= innerHeight && r.left >= 0 && r.right <= innerWidth,
                  police: getComputedStyle(a).fontSize }; }, NOM);
              await t.keyboard.type('x');
              await t.keyboard.press('Escape');
              await t.waitForTimeout(200);
              const rendu = await t.$eval(NOM, (x) => ({ valeur: x.value, focus: document.activeElement === x }));
              verif(`${cle} · téléphone : un appui sur le nom y pose le curseur sans quitter l'onglet Données (16 px : pas d'agrandissement) ; Échap rend « ${avant} »`,
                foc.champ && foc.onglet === 'donnees' && foc.dansEcran && parseFloat(foc.police) >= 16 && rendu.valeur === avant && !rendu.focus, JSON.stringify({ foc, rendu }));
            }
            if (await t.$('#et-bibliotheque')) {
              // la bibliothèque de la note au téléphone : l'entrée du menu ⋯ ouvre une feuille plein écran, qui se referme
              await t.click('#menu-m');
              await t.click('#mmenu [data-action="bibliotheque"]');
              await t.waitForFunction(() => { const p = document.getElementById('et-bibliotheque'); return p && !p.hidden && p.querySelector('#bib-corps').getAttribute('aria-busy') === 'false' && p.querySelector('.bib-val'); }, null, { timeout: 60000 });
              await t.waitForTimeout(250);
              const pb = await boite(t, '#et-bibliotheque .et-bib-c');
              const mb = await mesurer(t);
              verif(`${cle} · téléphone : « Bibliothèque » (menu ⋯) ouvre une feuille plein écran, rien ne déborde`, pb && pb.l === pb.W && pb.t >= pb.H - 1 && !mb.deborde, JSON.stringify({ pb, mb }));
              await t.screenshot({ path: path.join(OUT, `site_${cle}_tel_bibliotheque.png`) });
              await t.click('#et-bibliotheque [data-fermer-bibliotheque]');
              await t.waitForTimeout(150);
              verif(`${cle} · téléphone : la bibliothèque se referme`, await t.$eval('#et-bibliotheque', (x) => x.hidden));
            }
          }
          const cibles = await t.evaluate(() => {
            // (et celles de l'établi compact : la ligne de l'élément en tête des données — le nom, la liste, copier, supprimer —,
            // les menus, les pièces, les barres d'un cadre, les blocs et la liste de contrôle des résultats)
            const s = '.mtop button, .mtop a.btn, .mtabs button, .mbar .btn, #vues button, .zoom button, .sechd, .ni, .nl-add, #donnees .seg button, #donnees .field, .case, .addrow, .moins, .elements .ni.on .x, #resultat .el, #resultat .prof[data-pk]'
              + ', .et-elt-nom, .et-elt-b, .et-mn-b, .et-mn-l button, .et-tag, .et-dep, .et-add, .et-ret, .et-prop, .et-barres .b, .et-large, .et-sec-pick, .et-hist, .et-bloc-t, .et-tout, .et-just, .et-obs';
            const tous = [...document.querySelectorAll(s)].filter((e) => e.checkVisibility({ visibilityProperty: true, opacityProperty: true }));
            return tous.filter((e) => { const r = e.getBoundingClientRect(); return Math.min(r.width, r.height) < 35.5; })
              .map((e) => `${e.className || e.tagName} ${Math.round(e.getBoundingClientRect().width)}×${Math.round(e.getBoundingClientRect().height)}`);
          });
          verif(`${cle} · téléphone · ${onglet} : les cibles des briques communes font 36 px au moins`, !cibles.length, cibles.slice(0, 4).join(', '));
          if (onglet === 'dessin') {
            const tl = await toile(t);
            verif(`${cle} · téléphone : la toile offre 300 px de large au moins au dessin`, tl && tl.l >= 300, JSON.stringify(tl));
            const lg = await legendes(t);
            verif(`${cle} · téléphone : la légende de chaque vue se lit en entier, le contenu de chaque tableau se voit`, !lg.length, lg.join(' | '));
            const bt = await barreToile(t);
            verif(`${cle} · téléphone : la barre de la toile (vues, zoom) n'est couverte par rien`, !bt.length, bt.join(', '));
            const z = await zoom(t);
            if (!z.saute) verif(`${cle} · téléphone : « + » grossit le dessin, « ajuster » le remet`, z.ok, JSON.stringify(z));
            // le plein écran : la vue couvre l'écran
            await t.click('#zoom [data-plein]');
            await t.waitForTimeout(250);
            const pe = await boite(t, '.canvas');
            const fe = await boite(t, '.view.on .sheet');
            verif(`${cle} · téléphone : le plein écran donne l'écran entier à la vue`, pe && pe.g === 0 && pe.h === 0 && pe.l === pe.W && pe.t === pe.H && fe && fe.t > pe.H * 0.6,
              JSON.stringify({ pe, fe }));
            await t.screenshot({ path: path.join(OUT, `site_${cle}_tel_plein.png`) });
            await t.click('#zoom [data-plein]');
            await t.waitForTimeout(150);
            verif(`${cle} · téléphone : le plein écran se quitte`, !(await t.evaluate(() => document.querySelector('.bench').classList.contains('plein'))));
          }
          if (onglet === 'resultat') await volets(t, cle, 'téléphone');
         } catch (e) { verif(`${cle} · téléphone · ${onglet} : le parcours va au bout`, false, String(e.message).split('\n')[0]); await t.keyboard.press('Escape'); }
        }
        // le menu des actions : il s'ouvre, tient dans l'écran, et se referme
        await t.click('#menu-m');
        const mm = await boite(t, '#mmenu');
        verif(`${cle} · téléphone : le menu des actions s'ouvre dans l'écran`, dansEcran(mm) && (await t.$$eval('#mmenu .mi', (x) => x.filter((e) => e.checkVisibility()).length)) >= 6, JSON.stringify(mm));
        await t.click('.mtabs [data-onglet="donnees"]');
        verif(`${cle} · téléphone : un geste ailleurs referme le menu`, !(await vis(t, '#mmenu')));
        verif(`${cle} · téléphone : aucune erreur JavaScript`, !t.erreurs.length, t.erreurs.join(' | ').slice(0, 300));
      }
      await t.context().close();

      // ------------------------------------------------ les tablettes (une seule note, la fenêtre change de taille)
      const g = await contexte(820, 1180, false);
      if (PARTIES.includes('tablettes') && await neuve(g, cle, 'tablette portrait')) {
        for (const [fmt, w, h, seuil, nom] of [['tablette portrait', 820, 1180, [500, 300], 'tab_portrait'], ['tablette paysage', 1180, 820, [600, 400], 'tab_paysage'],
          ['petite tablette paysage', 1024, 768, [500, 400], 'tab_paysage_1024'], ['petit portable', 1280, 800, [400, 400], 'portable_1280']]) {
         try {
          await g.setViewportSize({ width: w, height: h });
          await g.waitForTimeout(400);
          const cadres = { donnees: await boite(g, '.panel.left'), resultat: await boite(g, '.panel.right'), toile: await boite(g, '.canvas') };
          verif(`${cle} · ${fmt} : les données, le résultat et la toile sont visibles ensemble, chacun à sa taille`,
            Object.values(cadres).every((r) => r && r.l >= 280 && r.t >= 240 && dansEcran(r)), JSON.stringify(cadres));
          const tl = await toile(g);
          verif(`${cle} · ${fmt} : la toile offre ${seuil[0]} × ${seuil[1]} px au moins au dessin`, tl && tl.l >= seuil[0] && tl.t >= seuil[1], JSON.stringify(tl));
          // la barre du haut : rien ne s'y chevauche (« Brouillon » passait sous le nom de la note)
          const haut = await g.evaluate(() => {
            const els = [...document.querySelectorAll('.top > *, .top .crumbs > *')].filter((e) => !e.classList.contains('crumbs') && !e.classList.contains('spacer')
              && e.checkVisibility({ visibilityProperty: true, opacityProperty: true })).map((e) => ({ n: (e.id || e.className || e.tagName).toString().slice(0, 20), r: e.getBoundingClientRect() }));
            const f = [];
            for (let i = 0; i < els.length; i++) for (let j = i + 1; j < els.length; j++) {
              const a = els[i].r, b = els[j].r;
              if (a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom && b.top < a.bottom) f.push(`${els[i].n} / ${els[j].n}`);
            }
            const top = document.querySelector('.top').getBoundingClientRect();
            els.forEach((e) => { if (e.r.right > top.right + 0.5) f.push(`${e.n} sort de la barre`); });
            // le nom de la note, trop long pour sa place : il finit par « … », jamais coupé au milieu d'une lettre
            const nn = document.getElementById('nom-note');
            if (nn.scrollWidth > nn.clientWidth + 1 && getComputedStyle(nn).textOverflow !== 'ellipsis') f.push('nom de la note coupé net (sans « … »)');
            return f;
          });
          verif(`${cle} · ${fmt} : la barre du haut ne se chevauche pas`, !haut.length, haut.join(', '));
          if (w < 1200) {
            const c44 = await cibles44(g);
            await g.waitForTimeout(150);
            verif(`${cle} · ${fmt} : chaque bouton vu de l'établi à 44 px au moins au doigt`, !c44.length, c44.slice(0, 6).join(' · '));
          }
          const ca = await cartouche(g);
          verif(`${cle} · ${fmt} : le cartouche ne se chevauche pas, l'indice et la date se lisent`, !ca.length, ca.join(', '));
          if (nom === 'portable_1280') {
            // le cadre le plus étroit (304 px) : chaque section des données ouverte, chaque champ mesuré
            await g.evaluate(() => document.querySelectorAll('#donnees .sec:not(.open) > .sechd').forEach((h) => h.click()));
            await g.waitForTimeout(200);
          }
          const m = await mesurer(g);
          verif(`${cle} · ${fmt} : rien ne déborde, aucun nom coupé, la pastille du verdict entière`, !m.deborde && !m.coupes.length && !m.xl.length, JSON.stringify(m));
          const lgT = await ligneElement(g);
          if (lgT) {
            // (colonne compacte : la poutre, la dalle) la LIGNE de l'élément tient dans l'en-tête des données à ce format ; sa
            // liste s'ouvre dans l'écran, chaque entrée atteinte, et Échap la referme
            const mn = await listeElement(g);
            await g.keyboard.press('Escape');
            await g.waitForTimeout(100);
            const ferme = await g.$eval(`${LISTE_ELT} .et-mn-l`, (x) => x.hidden);
            verif(`${cle} · ${fmt} : la ligne de l'élément tient dans l'en-tête des données (une rangée, rien ne déborde ni ne se chevauche, le nom entier) ; sa liste s'ouvre dans l'écran, chaque entrée atteinte ; Échap la referme`,
              !lgT.fautes.length && lgT.nomEntier && mn.vu && mn.dedans && mn.touche && ferme, JSON.stringify({ lgT, mn, ferme }));
          }
          // une comparaison longue à côté de la grande pastille (soudures, pleine pénétration) : elle ne la comprime pas
          const xl = await g.evaluate(() => {
            const gv = document.querySelector('.verdict .gov'), c = document.querySelector('.verdict .chip.xl');
            if (!gv || !c || !c.checkVisibility()) return null;
            const ajout = document.createTextNode(' · σ⊥ = 218,5 N/mm² ≤ 0,9·fu/γM2 = 259,2 N/mm² (pleine pénétration, métal de base)');
            gv.appendChild(ajout);
            const r = document.createRange(); r.selectNodeContents(c);
            const out = { texte: Math.round(r.getBoundingClientRect().width), boite: Math.round(c.getBoundingClientRect().width) };
            ajout.remove();
            return out;
          });
          if (xl) verif(`${cle} · ${fmt} : une comparaison longue ne comprime pas la grande pastille du verdict`, xl.texte <= xl.boite + 0.5, JSON.stringify(xl));
          const lg = await legendes(g);
          verif(`${cle} · ${fmt} : la légende de chaque vue se lit en entier, le contenu de chaque tableau se voit`, !lg.length, lg.join(' | '));
          const bt = await barreToile(g);
          verif(`${cle} · ${fmt} : la barre de la toile (vues, zoom) n'est couverte par rien`, !bt.length, bt.join(', '));
          await g.screenshot({ path: path.join(OUT, `site_${cle}_${nom}.png`) });
          const z = await zoom(g);
          if (!z.saute) verif(`${cle} · ${fmt} : « + » grossit le dessin, « ajuster » le remet`, z.ok, JSON.stringify(z));
          await volets(g, cle, fmt);
         } catch (e) { verif(`${cle} · ${fmt} : le parcours va au bout`, false, String(e.message).split('\n')[0]); await g.keyboard.press('Escape'); }
        }
        // un nom d'élément long : ce sont ses dernières lettres qui cèdent, jamais ce qui le suit — les dimensions de
        // l'intitulé de la liste ; ou (colonne de gauche compacte : la poutre, la dalle) dans la LIGNE de l'élément, en tête
        // des données, la liste, copier, supprimer, Annuler, Rétablir, entiers dans l'en-tête, et dans la tête des
        // résultats, les dimensions sous le nom
        await g.setViewportSize({ width: 820, height: 1180 });
        await g.waitForTimeout(300);
        const LONG = 'Poutre de rive très longue du bâtiment B, file 12 à 18, niveau +3';
        // (colonne compacte : 58 caractères — la dalle borne un nom à 60, AP.NOM_MAX, et refuse au-delà ; un nom refusé ne
        // prouverait rien ici : le contrôle exige qu'il soit pris, et qu'il ne tienne pas dans l'en-tête)
        const LONG_LIGNE = 'Élément de rive très long du bâtiment B, file 12 à 18, N+3';
        const nomLigne = '.panel.left .phead .et-elt input.et-elt-nom';
        if (await g.$(nomLigne)) {
          await g.fill(nomLigne, LONG_LIGNE);
          await g.press(nomLigne, 'Enter');
          await g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
          await g.waitForTimeout(150);
          const lgL = await ligneElement(g);
          const ln = await g.evaluate((s) => {
            // (un texte qui ne tient pas finit par « … » — text-overflow, hors du focus pour un champ —, jamais coupé net)
            const cede = (e) => e.scrollWidth <= e.clientWidth + 1 || getComputedStyle(e).textOverflow === 'ellipsis';
            const n = document.querySelector(s), rp = document.querySelector('#resultat .et-rp'), b = rp.querySelector('b'), d = rp.querySelector('span');
            const rr = rp.getBoundingClientRect(), rb = b.getBoundingClientRect(), rd = d.getBoundingClientRect();
            return { valeur: n.value, champ: cede(n), champTient: n.scrollWidth <= n.clientWidth + 1, titre: b.textContent, titreCede: cede(b) && rb.right <= rr.right + 0.5,
              dims: d.textContent, entieres: d.scrollWidth <= d.clientWidth + 1 && rd.right <= rr.right + 0.5 && rd.left >= rr.left - 0.5 && rd.width > 20 };
          }, nomLigne);
          verif(`${cle} · tablette portrait : un nom long cède (« … ») dans la ligne de l'élément et la tête des résultats ; la liste, copier, supprimer, Annuler, Rétablir restent entiers dans l'en-tête, les dimensions entières`,
            lgL && !lgL.fautes.length && ln.valeur === LONG_LIGNE && !ln.champTient && ln.champ && ln.titre === LONG_LIGNE && ln.titreCede && ln.entieres, JSON.stringify({ lgL, ln }));
        }
        // le champ du nom de l'élément (établi ordinaire) : celui dont la valeur ouvre l'intitulé de l'élément ouvert de la
        // liste, ses dimensions à sa suite
        const champNom = await g.evaluate(() => {
          const t = document.querySelector('.ni.on .t');
          if (!t || ![...t.children].some((e) => e.tagName === 'SPAN' && !e.classList.contains('nom'))) return null;
          const surLaLigne = t.querySelector('input[data-chemin]');
          if (surLaLigne) return surLaLigne.dataset.chemin;
          const intitule = t.textContent.trim();
          const c = [...document.querySelectorAll('#donnees input[data-sorte="texte"]')].filter((e) => e.value.trim() && intitule.startsWith(e.value.trim()))
            .sort((a, b) => b.value.length - a.value.length)[0];
          return c ? c.dataset.chemin : null;
        });
        if (champNom) {
          // un établi en accordéon : la rubrique du champ s'ouvre d'abord, par son en-tête
          const rub = await g.evaluate((c) => {
            const s = document.querySelector(`#donnees input[data-chemin="${c}"]`)?.closest('#donnees .sec[data-sec]');
            return s && !s.classList.contains('open') ? s.dataset.sec : null;
          }, champNom);
          if (rub) { await g.click(`#donnees .sec[data-sec="${rub}"] > .sechd`); await g.waitForTimeout(150); }
          const sNom = `#donnees input[data-chemin="${champNom}"]`;
          await g.fill(sNom, LONG);
          await g.press(sNom, 'Enter');
          await g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
          await g.waitForTimeout(150);
          // les dimensions (le dernier <span> de l'intitulé) entières et dans leur ligne ; le nom, lui, peut céder
          const ni = await g.evaluate(() => {
            const t = document.querySelector('.ni.on .t');
            const d = [...t.children].filter((e) => e.tagName === 'SPAN' && !e.classList.contains('nom')).pop();
            const rt = t.getBoundingClientRect(), rd = d.getBoundingClientRect();
            return { dims: d.textContent, entieres: d.scrollWidth <= d.clientWidth + 1 && rd.right <= rt.right + 0.5 && rd.left >= rt.left - 0.5, largeur: Math.round(rd.width) };
          });
          verif(`${cle} · tablette portrait : un nom long se coupe, les dimensions restent entières`, ni.entieres && ni.largeur > 20, JSON.stringify(ni));
        }
        verif(`${cle} · tablettes : aucune erreur JavaScript`, !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
      }
      await g.context().close();
    }

    // « Ajouter » un élément : le bouton des données ou de la liste — ou (colonne de gauche compacte : la poutre, la dalle)
    // l'entrée « Ajouter … » de la liste des éléments, un menu de la ligne de l'élément en tête des données (l'ouvrir
    // d'abord) ; le nombre d'éléments de la note : la liste, ou cette ligne (data-elements)
    const ajouterElement = async (pg) => {
      if (await pg.$(`${LISTE_ELT} [data-ajouter]`)) {
        await pg.click(`${LISTE_ELT} [data-menu]`);
        await pg.click(`${LISTE_ELT} .et-mn-l [data-ajouter]`);
      } else await pg.click('#donnees [data-ajouter], #liste [data-ajouter]');
    };
    const nElements = (pg) => pg.evaluate(() => { const l = document.querySelector('.panel.left .phead .et-elt .et-elsel[data-elements]');
      return l ? +l.dataset.elements : document.querySelectorAll('.elements .ni').length; });
    // ------------------------------------------------ les gestes communs, une fois (tablette portrait)
    if (PARTIES.includes('communs') && (choisis.includes('soudures') || choisis.includes('poutre'))) {
      const g = await contexte(820, 1180, false);
      // une feuille qui met les en-têtes de tableau en capitales (la charte le fait ; soudures.css le défait pour les
      // siens) : posée AVANT l'affichage, comme une feuille de module
      await g.context().addInitScript(() => document.addEventListener('DOMContentLoaded', () => {
        const st = document.createElement('style');
        st.textContent = '.vtab th { text-transform: uppercase !important; }';
        document.head.appendChild(st);
      }));
      const cle = choisis.includes('soudures') ? 'soudures' : 'poutre';
      if (await neuve(g, cle, 'gestes communs')) {
        // les lettres grecques d'un en-tête en capitales gardent leur casse : l'œil (innerText applique la casse) doit
        // lire η, σ, τ là où le texte porte η, σ, τ
        const grec = await g.evaluate(async () => {
          const out = { vus: 0, fautes: [] };
          for (const b of [...document.querySelectorAll('#vues [data-vue]')]) {
            b.click();
            await new Promise((r) => requestAnimationFrame(r));
            for (const th of document.querySelectorAll('.view.on .vtab th')) {
              if (!/[Ͱ-Ͽ]/.test(th.textContent)) continue;
              out.vus++;
              const lu = th.innerText;
              if (/[ΗΣΤΓ]/.test(lu) && !/[ΗΣΤΓ]/.test(th.textContent)) out.fautes.push(`${th.textContent} → ${lu}`);
            }
          }
          document.querySelector('#vues [data-vue]').click();
          return out;
        });
        if (cle === 'soudures') verif('en-têtes en capitales : η reste η, σ reste σ (jamais « H », « Σ »)', grec.vus > 0 && !grec.fautes.length, JSON.stringify(grec));
        // l'état VERROUILLÉ des briques (C.segments, C.texte, C.case avec desactive=True) se VOIT : les fragments rendus
        // par les briques elles-mêmes (Python), posés dans le cadre Données à côté des mêmes briques actives, et comparés
        // à l'œil (couleurs calculées) — un segment désactivé ressemblait trait pour trait à un segment actif
        const BRIQUES = [
          'import json, sys',
          'sys.path.insert(0, sys.argv[1])',
          'from api_web import commun as C',
          'o = [("a", "Rive"), ("b", "Continue"), ("c", "Encastrée")]',
          'print(json.dumps({k: C.segments(["essai", "s"], "a", o, desactive=v) + C.texte(["essai", "t"], "D1", desactive=v)',
          '                  + C.case(["essai", "c"], True, "Appui libre", desactive=v) for k, v in (("actif", False), ("verrou", True))}))',
        ].join('\n');
        const fr = require('child_process').spawnSync(process.env.PYTHON || 'python3', ['-c', BRIQUES, path.join(__dirname, '..', '..')], { encoding: 'utf8' });
        let briques = null;
        try { briques = JSON.parse(fr.stdout); } catch (e) { /* le contrôle rougit */ }
        const verrou = briques && await g.evaluate((b) => {
          const z = document.querySelector('#donnees');
          z.insertAdjacentHTML('afterbegin', `<div class="rows" id="essai-actif">${b.actif}</div><div class="rows" id="essai-verrou">${b.verrou}</div>`);
          const cs = (s) => getComputedStyle(document.querySelector(s));
          const diff = (s, p) => cs(`#essai-actif ${s}`)[p] !== cs(`#essai-verrou ${s}`)[p];
          const out = {
            segment: diff('.seg button:not(.on)', 'color'), retenu: diff('.seg button.on', 'backgroundColor') || diff('.seg button.on', 'color'),
            bande: diff('.seg', 'backgroundColor'), curseur: cs('#essai-verrou .seg button').cursor, texte: diff('.field', 'backgroundColor'),
            case: diff('.case', 'color'), lisible: cs('#essai-verrou .seg button.on').color !== cs('#essai-verrou .seg').backgroundColor,
          };
          document.getElementById('essai-actif').remove(); document.getElementById('essai-verrou').remove();
          return out;
        }, briques);
        verif('l\'état verrouillé des briques se voit (segments, texte, case), le choix retenu reste lisible',
          verrou && verrou.segment && verrou.retenu && verrou.bande && verrou.curseur === 'not-allowed' && verrou.texte && verrou.case && verrou.lisible,
          JSON.stringify(verrou || String(fr.stderr).slice(0, 200)));
        // télécharger un fichier propre au module : le moteur rend { fichier: { nom, type, texte } } ; ici la réponse
        // est simulée (le routeur de l'autre équipe la rendra) — le nom arrive piégé et repart nettoyé
        await g.evaluate(() => {
          document.querySelector('#donnees').insertAdjacentHTML('afterbegin', '<button type="button" class="btn sm" data-telecharger="essai">Télécharger</button>');
          window.__vu = null;
          window.__simuler = (q) => {
            if (q.action !== 'telecharger') return null;
            window.__vu = q;
            return q.quoi === 'essai' ? { ok: true, fichier: { nom: '../../dossier/essai..json', type: 'application/json', texte: '{"écran": "Streamlit", "a": 1}' } }
              : { ok: false, erreur: 'pas de fichier pour cette note' };
          };
        });
        const [dl] = await Promise.all([g.waitForEvent('download', { timeout: 20000 }), g.click('#donnees [data-telecharger="essai"]')]);
        const contenu = require('fs').readFileSync(await dl.path(), 'utf8');
        const vu = await g.evaluate(() => window.__vu);
        await verifProxy(g, 'télécharger : la requête part au moteur (action, quoi, état) et le fichier arrive sous un nom nettoyé',
          vu && vu.action === 'telecharger' && vu.quoi === 'essai' && vu.etat && dl.suggestedFilename() === 'essai.json' && contenu === '{"écran": "Streamlit", "a": 1}',
          JSON.stringify({ nom: dl.suggestedFilename(), contenu, action: vu && vu.action, quoi: vu && vu.quoi }));
        await g.evaluate(() => { document.querySelector('[data-telecharger="essai"]').dataset.telecharger = 'refuse'; });
        await g.click('#donnees [data-telecharger="refuse"]');
        await g.waitForTimeout(400);
        const tst = await g.evaluate(() => { const t = document.getElementById('toast'); return { vu: t.checkVisibility(), texte: t.textContent, erreur: t.classList.contains('erreur') }; });
        await verifProxy(g, 'télécharger : un refus du moteur s\'affiche en message', tst.vu && tst.erreur && /refusé.*pas de fichier pour cette note/.test(tst.texte), JSON.stringify(tst));
        // un module qui ne rend aucune vue : un texte neutre (ni « impossible », ni erreur)
        await g.evaluate(() => {
          window.__simuler = null;
          window.__retoucher = (json) => { const r = JSON.parse(json); if (r.ok && r.vues) r.vues = []; return JSON.stringify(r); };
        });
        // « ajouter » recalcule toujours (contrat commun) : sa réponse arrive sans vue
        await ajouterElement(g);
        await g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
        await g.waitForTimeout(300);
        const sv = await g.evaluate(() => ({ texte: document.getElementById('stage').textContent.trim(), zoom: document.getElementById('zoom').checkVisibility() }));
        await verifProxy(g, 'aucune vue : un texte neutre, sans barre de zoom', sv.texte === 'Pas de dessin pour cette note.' && !sv.zoom, JSON.stringify(sv));
        await g.evaluate(() => { window.__retoucher = null; });
        // une réponse illisible du moteur (un « Infinity », que JSON ne connaît pas) : un message, et l'établi reste
        // vivant — la requête restait sans réponse, la file du moteur bloquée, « Calcul… » sans fin
        const nAvant = await nElements(g);
        await g.evaluate(() => { window.__retoucher = (json) => { window.__retoucher = null; return json.replace(/"ok": true/, '"ok": true, "x": Infinity'); }; });
        await ajouterElement(g);
        const libre = await g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 10000 }).then(() => true, () => false);
        const ill = await g.evaluate(() => { const t = document.getElementById('toast'); return { vu: t.checkVisibility(), texte: t.textContent, erreur: t.classList.contains('erreur') }; });
        await ajouterElement(g);
        const repart = await g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 30000 }).then(() => true, () => false);
        const nApres = await nElements(g);
        await verifProxy(g, 'une réponse illisible du moteur : un message, et le calcul suivant passe (la file du moteur ne se bloque pas)',
          libre && ill.vu && ill.erreur && /réponse illisible du moteur/.test(ill.texte) && repart && nApres === nAvant + 1, JSON.stringify({ libre, ...ill, repart, nAvant, nApres }));
        // les erreurs de la réponse illisible sont attendues : la console les dit, la page n'a pas levé
        g.erreurs = g.erreurs.filter((x) => !/réponse illisible du moteur/.test(x));
        verif('gestes communs : aucune erreur JavaScript', !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
      }
      await g.context().close();
      // le taux d'une note au-delà de 999 % : l'en-tête du téléphone et la carte de l'accueil écrivent « > 999 % » (la
      // valeur exacte au survol), comme les pastilles du moteur ; un taux de huit chiffres en sortait
      const tt = await contexte(390, 844, true);
      if (await neuve(tt, cle, 'taux énorme')) {
        await tt.evaluate(() => { window.__retoucher = (json) => { const r = JSON.parse(json); if (r.ok && r.resume) r.resume.taux = 12345678; return JSON.stringify(r); }; });
        await ajouterElement(tt);
        await tt.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
        await tt.evaluate(() => { window.__retoucher = null; });
        const vuesTaux = await interceptees(tt);   // (lu dans l'établi, où la retouche a joué : l'accueil, plus bas, est un autre document)
        const ch = await tt.evaluate(() => {
          const c = document.getElementById('chip-m'), r = c.getBoundingClientRect(), m = document.getElementById('menu-m').getBoundingClientRect();
          return { texte: c.textContent, titre: c.title, vu: c.checkVisibility(), l: Math.round(r.width), entier: c.scrollWidth <= c.clientWidth + 1 && r.right <= m.left + 0.5,
            nom: Math.round(document.getElementById('nom-note-m').getBoundingClientRect().width) };
        });
        await verifProxy(tt, 'taux de plus de 999 % · téléphone : l\'en-tête écrit « > 999 % » (la valeur exacte au survol), la pastille entière, le nom lisible',
          ch.vu && ch.texte === '>\u00a0999\u00a0%' && ch.titre === '12\u202f345\u202f678\u00a0%' && ch.entier && ch.l <= 90 && ch.nom >= 150, JSON.stringify(ch));
        await tt.goto(`${BASE}/index.html`);
        await tt.waitForTimeout(300);
        const ca = await tt.evaluate(() => {
          const c = document.querySelector('#recentes .rcard .chip');
          if (!c) return null;
          const r = c.getBoundingClientRect(), m = c.closest('.m').getBoundingClientRect();
          return { texte: c.textContent, titre: c.title, vu: c.checkVisibility(), dedans: r.right <= m.right + 0.5 && c.scrollWidth <= c.clientWidth + 1 };
        });
        await verifProxy(tt, 'taux de plus de 999 % · accueil : la carte de la note écrit « > 999 % », la valeur exacte au survol, la pastille dans sa carte',
          ca && ca.vu && ca.texte === '>\u00a0999\u00a0%' && ca.titre === '12\u202f345\u202f678\u00a0%' && ca.dedans, JSON.stringify(ca), vuesTaux);
        verif('taux de plus de 999 % : aucune erreur JavaScript', !tt.erreurs.length, tt.erreurs.join(' | ').slice(0, 300));
      }
      await tt.context().close();
    }

    // ------------------------------------------------ numpy : chargé pour ceux qui s'en servent, à la demande sinon
    if (PARTIES.includes('numpy')) {
      const man = await (await p.request.get(`${BASE}/moteur/manifeste.json`)).json();
      const sans = Object.keys(MODS).find((k) => !(man.avec_numpy || []).includes(k));
      const avec = (man.avec_numpy || []).find((k) => MODS[k]);
      if (sans) {
        const g = await contexte(1280, 800, false);
        await g.goto(`${BASE}/${MODS[sans].page}`);
        await pretEcran(g);
        const ok = !(await echecEcran(g));
        verif(`numpy : ${sans} démarre sans le charger`, ok && !g.requetes.some((u) => /numpy-.*\.whl/.test(u)), g.requetes.filter((u) => /numpy/.test(u)).join(', '));
        await g.context().close();
      }
      if (avec) {
        // le manifeste « oublie » ce module : numpy doit être chargé à la demande (le guet du travailleur) et la note
        // calculée quand même
        const g = await contexte(1280, 800, false);
        await g.context().route('**/moteur/manifeste.json', async (route) => {
          const r = await route.fetch();
          const j = await r.json();
          j.avec_numpy = [];
          await route.fulfill({ response: r, json: j });
        });
        await g.goto(`${BASE}/${MODS[avec].page}`);
        await pretEcran(g);
        const etat = { ok: !(await echecEcran(g)), resultat: await g.evaluate(() => !!document.querySelector('#resultat .verdict, #resultat .el')) };
        verif(`numpy : ${avec} oublié du manifeste se calcule quand même (numpy chargé à la demande)`,
          etat.ok && etat.resultat && g.requetes.some((u) => /numpy-.*\.whl/.test(u)), JSON.stringify({ ...etat, numpy: g.requetes.filter((u) => /numpy/.test(u)) }));
        await g.context().close();
      }
    }
  }

  // ---------------------------------------------------------------- télécharger, de bout en bout
  // Le protocole entier, sans rien simuler : un bouton RENDU PAR UN MODULE (C.bouton_telecharger, en Python) → le clic
  // → le travailleur → le routeur (action « telecharger », contrôle du fichier par refus_fichier) → le Blob → le fichier
  // que Chromium enregistre, sous la politique de sécurité du site. Aucun module publié ne propose encore de fichier :
  // le navigateur reçoit ici un paquet du moteur où le module des poutres en propose (le code AJOUT, ajouté à
  // api_web/poutre.py à la volée ; le manifeste suit l'empreinte du paquet). Rien de ceci n'est publié.
  //   filtre « telecharger »
  if (!FILTRE || FILTRE === 'telecharger') {
    const fs = require('fs');
    const AJOUT = [
      '# --- ajouté à la volée par site/web/verifier_site.js (télécharger, de bout en bout) : jamais publié',
      'import json as _json_essai',
      '_resultat_publie = resultat',
      '',
      '',
      'def resultat(etat):',
      '    r = _resultat_publie(etat)',
      '    b = "".join(C.bouton_telecharger(q, t) for q, t in (("json", "JSON de la note"), ("nom-piege", "Nom piégé"),',
      '                                                         ("type-html", "Page HTML"), ("inconnu", "Inconnu")))',
      '    r["donnees"] = \'<div class="essai-telecharger">\' + b + "</div>" + r["donnees"]',
      '    r["volets"] = r["volets"].replace(\'<div class="ck-sub">\', C.bouton_telecharger("json", "JSON de la note") + \'<div class="ck-sub">\')',
      '    return r',
      '',
      '',
      'def telecharger(etat, quoi):',
      '    if quoi == "json":',
      '        return {"nom": C.nom_fichier("Poutre P1/2 : RDC « essai »", "json"), "type": "application/json",',
      '                "texte": _json_essai.dumps(etat, ensure_ascii=False)}',
      '    if quoi == "nom-piege":',
      '        return {"nom": "../../poutre.json", "type": "application/json", "texte": "{}"}',
      '    if quoi == "type-html":',
      '        return {"nom": "page.html", "type": "text/html", "texte": "<script>window.__piege = 1</script>"}',
      '    return None',
    ].join('\n');
    // le paquet d'essai : celui du site, où api_web/poutre.py reçoit AJOUT (Python le réécrit ; le texte passe en base64)
    const ZIPPER = [
      'import base64, io, sys, zipfile',
      'ajout = base64.b64decode(sys.argv[1])',
      'src = zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read()))',
      'out = io.BytesIO()',
      'with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:',
      '    for i in src.infolist():',
      '        d = src.read(i)',
      '        z.writestr(i, d + b"\\n\\n" + ajout + b"\\n" if i.filename == "api_web/poutre.py" else d)',
      'sys.stdout.buffer.write(out.getvalue())',
    ].join('\n');
    const zip0 = await (await p.request.get(`${BASE}/moteur/moteur.zip`)).body();
    const fait = require('child_process').spawnSync(process.env.PYTHON || 'python3', ['-I', '-c', ZIPPER, Buffer.from(AJOUT).toString('base64')],
      { input: zip0, maxBuffer: 1 << 27 });
    const zipEssai = fait.stdout;
    const shaEssai = require('crypto').createHash('sha256').update(zipEssai || '').digest('hex');
    verif('télécharger : le paquet d\'essai se prépare (le module des poutres y propose des fichiers)', fait.status === 0 && zipEssai.length > zip0.length / 2,
      String(fait.stderr || fait.error || '').slice(0, 300));
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const prepare = async (w, h, tel) => {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel, acceptDownloads: true });
      await c.route('**/moteur/manifeste.json', async (route) => {
        const r = await route.fetch();
        const j = await r.json();
        j.sha256 = shaEssai;
        await route.fulfill({ response: r, json: j });
      });
      await c.route('**/moteur/moteur.zip*', (route) => route.fulfill({ status: 200, contentType: 'application/zip', body: zipEssai }));
      // toute violation de la politique de sécurité est notée (un fichier qui s'exécuterait, une ressource refusée…)
      await c.addInitScript(() => {
        window.__csp = [];
        document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
      });
      const pg = await c.newPage();
      pg.erreurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') pg.erreurs.push(m.text().slice(0, 300)); });
      return pg;
    };
    const ouvrir = async (pg, cle) => {
      const rep = await pg.goto(`${BASE}/${MODS[cle].page}?chantier=${encodeURIComponent('Contrôle')}&nom=${encodeURIComponent('Essai télécharger')}`);
      await pretEcran(pg);
      return rep;
    };
    const visible = (pg, s) => pg.evaluate((s) => [...document.querySelectorAll(s)].some((e) => e.checkVisibility({ visibilityProperty: true, opacityProperty: true })
      && e.getBoundingClientRect().width > 20 && e.getBoundingClientRect().height > 20), s);
    const message = (pg) => pg.evaluate(() => { const t = document.getElementById('toast'); return { vu: t.checkVisibility(), texte: t.textContent, erreur: t.classList.contains('erreur') }; });
    const etatNote = (pg) => pg.evaluate(() => {
      const id = new URLSearchParams(location.search).get('note');
      return { id, etat: JSON.parse(localStorage.getItem('stab.notes.v1'))[id].etat };
    });
    // un clic qui ne doit RIEN faire télécharger : le refus du routeur arrive en message
    const refuse = async (pg, sel, motif, nom) => {
      const att = pg.waitForEvent('download', { timeout: 3000 }).then((d) => d.suggestedFilename(), () => null);
      await pg.click(sel);
      const fichier = await att;
      const t = await message(pg);
      verif(`télécharger · ${nom} : refusé, aucun fichier enregistré, un message en français`, fichier === null && t.vu && t.erreur && motif.test(t.texte),
        JSON.stringify({ fichier, ...t }));
    };
    const NOM = 'Poutre P1-2 - RDC « essai ».json';

    // ------------------------------------------------ au PC
    const g = await prepare(1440, 900, false);
    const rep = await ouvrir(g, 'poutre');
    verif('télécharger · PC : le bouton rendu par le module (C.bouton_telecharger) est posé et visible', await visible(g, '#donnees [data-telecharger="json"]'));
    const [dl] = await Promise.all([g.waitForEvent('download', { timeout: 30000 }), g.click('#donnees [data-telecharger="json"]')]);
    const contenu = fs.readFileSync(await dl.path(), 'utf8');
    const note = await etatNote(g);
    let relu = null;
    try { relu = JSON.parse(contenu); } catch (e) { /* illisible : le contrôle rougit */ }
    verif('télécharger · PC : le clic → le routeur → le fichier enregistré sous le nom donné par le module, avec l\'état de la note',
      dl.suggestedFilename() === NOM && relu !== null && JSON.stringify(relu) === JSON.stringify(note.etat),
      JSON.stringify({ nom: dl.suggestedFilename(), contenu: contenu.slice(0, 80) }));
    const t0 = await message(g);
    verif('télécharger · PC : un message dit le fichier téléchargé', t0.vu && !t0.erreur && t0.texte === `${NOM} téléchargé`, JSON.stringify(t0));
    await refuse(g, '#donnees [data-telecharger="nom-piege"]', /^Téléchargement refusé : Le fichier « nom-piege » est refusé : le nom du fichier ne peut contenir/, 'un nom de fichier qui remonte les dossiers');
    await refuse(g, '#donnees [data-telecharger="type-html"]', /^Téléchargement refusé : .*type de fichier non permis : « text\/html »/, 'une page HTML');
    await refuse(g, '#donnees [data-telecharger="inconnu"]', /^Téléchargement refusé : Fichier inconnu : « inconnu »/, 'un fichier que le module ne connaît pas');
    // depuis la fenêtre de calcul : le fichier arrive, la fenêtre reste ouverte
    await g.click('#resultat [data-calcul]');
    await g.waitForTimeout(250);
    const [dl2] = await Promise.all([g.waitForEvent('download', { timeout: 30000 }), g.click('.flyout.on.calcul [data-telecharger="json"]')]);
    await g.waitForTimeout(300);
    verif('télécharger · PC : depuis la fenêtre de calcul, le fichier arrive et la fenêtre reste ouverte',
      dl2.suggestedFilename() === NOM && await visible(g, '.flyout.on.calcul'), dl2.suggestedFilename());
    await g.keyboard.press('Escape');
    // la note elle-même (.stab) passe par le même chemin d'enregistrement
    const [dl3] = await Promise.all([g.waitForEvent('download', { timeout: 30000 }), g.click('.top [data-action="exporter"]')]);
    let stab = null;
    try { stab = JSON.parse(fs.readFileSync(await dl3.path(), 'utf8')); } catch (e) { /* illisible : le contrôle rougit */ }
    verif('exporter la note : le fichier « chantier - nom.stab » arrive, la note entière',
      dl3.suggestedFilename() === 'Contrôle - Essai télécharger.stab' && stab && stab.format === 'stab' && stab.note.id === note.id, dl3.suggestedFilename());
    const csp = rep.headers()['content-security-policy'] || '';
    const apres = await g.evaluate(() => ({ csp: window.__csp, piege: window.__piege }));
    verif('télécharger · PC : sous la politique de sécurité du site, aucune violation, rien du fichier exécuté, aucune erreur JavaScript',
      /script-src 'self'/.test(csp) && !apres.csp.length && apres.piege === undefined && !g.erreurs.length,
      JSON.stringify({ csp: csp.slice(0, 40), violations: apres.csp, erreurs: g.erreurs.slice(0, 3) }));
    // un module publié sans fichier à proposer : le VRAI routeur refuse, en français (le flambement : la dalle fait
    // désormais sa note PDF, et le taux d'armature son bordereau CSV — Lot Q, 10/11/2026 —, ils ont un « telecharger »)
    await ouvrir(g, 'flambement');
    await g.evaluate(() => document.querySelector('#donnees').insertAdjacentHTML('afterbegin', '<button type="button" class="btn sm" data-telecharger="json">JSON</button>'));
    await refuse(g, '#donnees [data-telecharger="json"]', /^Téléchargement refusé : Ce module ne propose aucun fichier à télécharger\.$/, 'un module sans fichier (flambement)');
    await g.context().close();

    // ------------------------------------------------ au téléphone (le même bouton, dans l'onglet Données)
    const tl = await prepare(390, 844, true);
    await ouvrir(tl, 'poutre');
    verif('télécharger · téléphone : le bouton du module est posé et visible', await visible(tl, '#donnees [data-telecharger="json"]'));
    const [dl4] = await Promise.all([tl.waitForEvent('download', { timeout: 30000 }), tl.click('#donnees [data-telecharger="json"]')]);
    let relu4 = null;
    try { relu4 = JSON.parse(fs.readFileSync(await dl4.path(), 'utf8')); } catch (e) { /* illisible : le contrôle rougit */ }
    verif('télécharger · téléphone : le fichier arrive sous le même nom, avec l\'état de la note',
      dl4.suggestedFilename() === NOM && relu4 && JSON.stringify(relu4) === JSON.stringify((await etatNote(tl)).etat), dl4.suggestedFilename());
    verif('télécharger · téléphone : aucune violation de la politique de sécurité, aucune erreur JavaScript',
      !(await tl.evaluate(() => window.__csp.length)) && !tl.erreurs.length, tl.erreurs.join(' | ').slice(0, 300));
    await tl.context().close();
  }

  // ---------------------------------------------------------------- la météo du lieu (Open-Meteo), de bout en bout
  // Le vrai écran, le vrai moteur, la vraie politique de sécurité ; seul Open-Meteo est remplacé : chaque demande vers ses
  // trois adresses est INTERCEPTÉE (page.route) et reçoit une réponse au format documenté du service (la machine n'a pas de
  // réseau, et un contrôle n'appelle jamais le vrai service). Qu'une demande arrive à l'interception prouve que la politique
  // de sécurité la laisse partir : bloquée par connect-src, elle échouerait avant (« Pas de réseau »), sans être vue ici.
  // Le parcours : la cure « Météo du lieu », le clic, l'indicateur « Récupération de la météo… », les appels (adresses,
  // paramètres), les températures reçues (badge, relevé, vues), la maturité recalculée, la note rechargée qui garde tout et
  // se recalcule SANS AUCUN appel, le détail horaire, une correction à la main ; puis les refus — lieu introuvable, réseau
  // coupé, service en panne, dates hors de portée — en message, la note intacte ; au PC, au téléphone et à la tablette.
  //   filtre « meteo »
  if (!FILTRE || FILTRE === 'meteo') {
    const OPEN_METEO = /^https:\/\/(geocoding-api|archive-api|api)\.open-meteo\.com\//;
    const CORS = { 'access-control-allow-origin': '*' };
    const LIEUX = {
      bruxelles: { id: 2800866, name: 'Bruxelles', latitude: 50.85045, longitude: 4.34878, elevation: 28.0, feature_code: 'PPLC', country_code: 'BE',
        admin1_id: 2800867, timezone: 'Europe/Brussels', population: 1019022, country_id: 2802361, country: 'Belgique', admin1: 'Bruxelles-Capitale' },
    };
    // la température d'un jour (°C, au dixième) et d'une heure : déterministes, inventées
    const tJour = (iso) => { const k = Math.round(Date.parse(iso + 'T00:00:00Z') / 864e5); return Math.round((5.5 + 7 * Math.sin(k / 11) + 0.3 * (k % 5)) * 10) / 10; };
    const tHeure = (iso, h) => Math.round((tJour(iso) + 3.5 * Math.sin((h - 9) * Math.PI / 12)) * 10) / 10;
    const jours = (d0, d1) => { const out = []; for (let t = Date.parse(d0 + 'T00:00:00Z'); t <= Date.parse(d1 + 'T00:00:00Z'); t += 864e5) out.push(new Date(t).toISOString().slice(0, 10)); return out; };
    const decale = (iso, n) => new Date(Date.parse(iso + 'T00:00:00Z') + n * 864e5).toISOString().slice(0, 10);
    const fr = (iso) => iso.split('-').reverse().join('/');
    // le faux Open-Meteo : chaque demande journalisée, la réponse tirée de ses paramètres ; « mode » : coupe, panne, lent
    const brancher = async (c, journal, mode) => {
      await c.route(OPEN_METEO, async (route) => {
        const u = new URL(route.request().url());
        journal.push(route.request().url());
        if (mode.lent) await new Promise((r) => setTimeout(r, mode.lent));
        if (mode.coupe) return route.abort('internetdisconnected');
        if (mode.panne) return route.fulfill({ status: 500, headers: CORS, contentType: 'application/json', body: JSON.stringify({ error: true, reason: 'Panne simulée du service' }) });
        // une réponse ÉNORME (8 Mo) : la page ne la lit pas (relecture adverse : elle était lue toute, puis envoyée au moteur)
        if (mode.enorme) return route.fulfill({ status: 200, headers: CORS, contentType: 'application/json', body: `{"bourrage":"${'x'.repeat(8 * 1024 * 1024)}"}` });
        const q = Object.fromEntries(u.searchParams);
        let corps;
        if (u.hostname.startsWith('geocoding-api')) {
          const g = LIEUX[(q.name || '').split(',')[0].trim().toLowerCase()];
          corps = g ? { results: [g], generationtime_ms: 0.61 } : { generationtime_ms: 0.43 };
        } else {
          const js = jours(q.start_date, q.end_date);
          // une archive qui commence le LENDEMAIN du jour demandé (relecture adverse : le relevé glissait d'un jour, sans un mot)
          if (mode.decale && u.hostname.startsWith('archive-api')) js.shift();
          corps = { latitude: 50.85, longitude: 4.35, generationtime_ms: 0.2, utc_offset_seconds: 3600, timezone: 'Europe/Brussels', timezone_abbreviation: 'GMT+1', elevation: 28 };
          if (q.hourly) {
            corps.hourly_units = { time: 'iso8601', [q.hourly]: '°C' };
            corps.hourly = { time: js.flatMap((d) => [...Array(24).keys()].map((h) => `${d}T${String(h).padStart(2, '0')}:00`)), [q.hourly]: js.flatMap((d) => [...Array(24).keys()].map((h) => tHeure(d, h))) };
          } else {
            corps.daily_units = { time: 'iso8601', [q.daily]: '°C' };
            corps.daily = { time: js, [q.daily]: js.map(tJour) };
          }
        }
        return route.fulfill({ status: 200, headers: CORS, contentType: 'application/json', body: JSON.stringify(corps) });
      });
    };
    const preparer = async (w, h, tel) => {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel });
      const journal = [], mode = {};
      await brancher(c, journal, mode);
      await c.addInitScript(() => {
        window.__csp = [];
        document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
      });
      const pg = await c.newPage();
      pg.journal = journal; pg.mode = mode; pg.erreurs = []; pg.ailleurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') pg.erreurs.push(m.text().slice(0, 300)); });
      // aucune autre adresse extérieure n'est jamais demandée
      c.on('request', (r) => { const u = r.url(); if (!u.startsWith(BASE) && !OPEN_METEO.test(u) && !/^(data|blob):/.test(u)) pg.ailleurs.push(u); });
      return pg;
    };
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const ouvrir = async (pg) => {
      const rep = await pg.goto(`${BASE}/${MODS.age.page}?chantier=${encodeURIComponent('Contrôle météo')}&nom=${encodeURIComponent('Dalle R+2')}`);
      await pretEcran(pg);
      return rep;
    };
    const tranquille = (pg) => pg.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 90000 });
    // (Lot Q, 10/11/2026 : la page est passée à l'établi compact, qui n'ouvre que la PREMIÈRE rubrique — le harnais suit)
    // la rubrique « Températures » (data-sec « tp ») ouverte avant d'en toucher les champs
    const ouvrirTp = async (pg) => {
      if (!(await pg.$eval('#donnees .sec[data-sec="tp"]', (e) => e.classList.contains('open')))) await pg.click('#donnees .sec[data-sec="tp"] > .sechd');
      await pg.waitForTimeout(100);
    };
    const LIGNE_ELT = '.panel.left .phead .et-elt .et-elsel';
    const activer = async (pg, i) => { await pg.click(`${LIGNE_ELT} .et-elt-liste [data-menu]`); await pg.click(`${LIGNE_ELT} .et-elt-liste .et-mn-l [data-activer="${i}"]`); };
    const saisirDate = async (pg, chemin, v) => { await pg.fill(`#donnees input[data-chemin="${chemin}"]`, v); await pg.press(`#donnees input[data-chemin="${chemin}"]`, 'Enter'); await tranquille(pg); };
    const lire = (pg) => pg.evaluate(() => {
      const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const b = document.querySelector('#donnees [data-meteo-chemin="temperature/meteo"]');
      const rb = b ? b.getBoundingClientRect() : null;
      const t = document.getElementById('toast');
      const ta = document.querySelector('#donnees textarea[data-chemin="temperature/releve"]');
      const hints = [...document.querySelectorAll('#donnees .hint, #donnees .msg')].filter(vis).map((x) => x.textContent.replace(/\s+/g, ' ').trim());
      return { plan: b ? JSON.parse(b.dataset.meteo) : null, bouton: b ? { vu: vis(b), l: Math.round(rb.width), h: Math.round(rb.height), grise: b.disabled, texte: b.textContent.trim() } : null,
        toast: { vu: vis(t), texte: t.textContent, erreur: t.classList.contains('erreur') }, releve: ta ? ta.value : null, releveVu: vis(ta), hints,
        verdict: document.querySelector('#resultat .verdict')?.textContent.replace(/\s+/g, ' ').trim() || '',
        vues: [...document.querySelectorAll('#vues [data-vue]')].map((x) => x.dataset.vue) };
    });
    // ce que l'œil voit : rien ne déborde, aucun libellé coupé (segments de la cure, bouton, badge)
    const regard = (pg) => pg.evaluate(() => {
      const vis = (e) => e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const large = (e) => { const r = document.createRange(); r.selectNodeContents(e); return r.getBoundingClientRect().width; };
      const coupes = [...document.querySelectorAll('#donnees .rows .seg.full button, #donnees .btn.meteo > span, #donnees .row > label > span:not(.sym)')]
        .filter(vis).filter((e) => e.scrollWidth > e.clientWidth + 1 || large(e) > e.getBoundingClientRect().width + 0.5).map((e) => e.textContent.trim());
      return { deborde: document.documentElement.scrollWidth > innerWidth, coupes };
    });
    const attendre = (pg, f, ms = 60000) => pg.waitForFunction(f, null, { timeout: ms }).then(() => true, () => false);
    const ligneReleve = (d, src) => { const t = tJour(d); return `${d};${(Number.isInteger(t) ? t.toFixed(1) : String(t)).replace('.', ',')};${src}`; };

    // ------------------------------------------------ au PC
    const g = await preparer(1440, 900, false);
    const rep = await ouvrir(g);
    verif('météo · PC : la page est servie avec la politique de sécurité qui permet les trois adresses d\'Open-Meteo, et elles seules',
      /connect-src 'self' https:\/\/geocoding-api\.open-meteo\.com https:\/\/archive-api\.open-meteo\.com https:\/\/api\.open-meteo\.com;/.test(rep.headers()['content-security-policy'] || ''),
      (rep.headers()['content-security-policy'] || '').slice(0, 200));
    const v20 = (await lire(g)).verdict;
    await ouvrirTp(g);
    await g.click('#donnees button[data-chemin="temperature/mode"][data-valeur="meteo"]');
    await tranquille(g);
    let L = await lire(g);
    verif('météo · PC : « Météo du lieu » ouvre le lieu, les dates et le bouton « Récupérer la météo », visible, sur la largeur du cadre',
      L.plan && L.bouton && L.bouton.vu && L.bouton.l > 250 && L.bouton.h >= 32 && !L.bouton.grise && L.bouton.texte === 'Récupérer la météo'
      && (await g.inputValue('#donnees input[data-chemin="temperature/lieu"]')) === 'Bruxelles, Belgique', JSON.stringify(L.bouton));
    const auj = L.plan.aujourdhui;
    const debut = decale(auj, -10), fin = decale(auj, 18);
    await saisirDate(g, 'temperature/debut', fr(debut));
    await saisirDate(g, 'temperature/fin', fr(fin));
    L = await lire(g);
    const plan = L.plan;
    verif('météo · PC : le plan du bouton suit les dates saisies (archive puis prévisions, 2 jours prolongés au-delà de l\'horizon)',
      plan.debut === debut && plan.fin === fin && plan.appels.map((a) => a.source).join() === 'archive,prevision' && plan.n_prolonges === 2, JSON.stringify(plan).slice(0, 300));
    let m = await regard(g);
    verif('météo · PC : rien ne déborde, aucun libellé coupé (« Météo du lieu », le bouton, les champs Lieu, Début, Fin)', !m.deborde && !m.coupes.length, m.coupes.join(', '));
    // le clic : l'indicateur pendant la récupération (le faux service répond en 0,8 s)
    g.mode.lent = 800;
    await g.click('#donnees [data-meteo-chemin="temperature/meteo"]');
    const pendant = await attendre(g, () => {
      const b = document.querySelector('#donnees [data-meteo-chemin="temperature/meteo"]');
      const c = [...document.querySelectorAll('.calcule')].find((x) => x.checkVisibility() && x.textContent);
      return b && b.disabled && b.textContent.trim() === 'Récupération de la météo…' && b.checkVisibility() && c && c.textContent === 'Récupération de la météo…';
    }, 5000);
    verif('météo · PC : pendant la récupération, le bouton et l\'indicateur disent « Récupération de la météo… » (le bouton ne répond plus)', pendant);
    await tranquille(g);
    g.mode.lent = 0;
    L = await lire(g);
    const nJ = jours(debut, fin).length, nMes = jours(debut, auj).length, nPrev = jours(decale(auj, 1), plan.fin_dispo).length;
    const att = [`https://geocoding-api.open-meteo.com/v1/search?name=Bruxelles%2C+Belgique&count=1&language=fr`]
      .concat(plan.appels.map((a) => `https://${a.source === 'archive' ? 'archive-api' : 'api'}.open-meteo.com/v1/${a.source === 'archive' ? 'archive' : 'forecast'}?latitude=50.85045&longitude=4.34878`
        + `&start_date=${a.debut}&end_date=${a.fin}&timezone=auto&daily=temperature_2m_mean`));
    verif('météo · PC : le navigateur a fait les appels du plan, vers Open-Meteo seulement (géocodage, archive, prévisions ; paramètres encodés) — la politique de sécurité les laisse passer',
      JSON.stringify(g.journal) === JSON.stringify(att) && !g.ailleurs.length, JSON.stringify({ journal: g.journal, att, ailleurs: g.ailleurs }));
    const badge = `Bruxelles, Bruxelles-Capitale, Belgique — ${nMes} j mesurés · ${nPrev} j de prévision · 2 j prolongés`;
    verif('météo · PC : les températures reçues : le badge (lieu trouvé, jours mesurés / prévus / prolongés), la date de récupération, un message',
      L.hints.some((x) => x === badge) && L.hints.some((x) => x.startsWith(`Open-Meteo · récupérée le ${fr(auj)} à `)) && L.toast.vu && !L.toast.erreur
      && /^Météo récupérée/.test(L.toast.texte), JSON.stringify({ hints: L.hints.slice(0, 6), toast: L.toast }));
    const lignes = (L.releve || '').split('\n');
    verif('météo · PC : le relevé reçu est écrit, jour par jour, modifiable (une ligne par jour : date;T;source)',
      L.releveVu && lignes.length === nJ && lignes[0] === ligneReleve(debut, 'mesuré') && lignes[nMes] === ligneReleve(decale(auj, 1), 'prévision')
      && lignes[nJ - 1].endsWith(';prolongé'), JSON.stringify({ n: lignes.length, nJ, l0: lignes[0], att0: ligneReleve(debut, 'mesuré') }));
    verif('météo · PC : la maturité est recalculée (le verdict n\'est plus celui de 20 °C), les vues « Températures » et « Tableau » sont là',
      L.verdict && L.verdict !== v20 && L.vues.join() === 'courbe,temperatures,tableau', JSON.stringify({ v20, v: L.verdict, vues: L.vues }));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc.png') });
    // les vues : la courbe des températures et le tableau, vus à l'écran
    await g.click('#vues [data-vue="temperatures"]');
    await g.waitForTimeout(250);
    const courbe = await g.evaluate(() => { const s = [...document.querySelectorAll('.view.on .draw > svg')].find((x) => x.checkVisibility()); const r = s && s.getBoundingClientRect();
      // (Lot Q : la vue a l'en-tête technique de l'établi compact, son titre en capitales et sa légende en dernière ligne — le harnais suit)
      return { titre: document.querySelector('.view.on .stitle b, .view.on .et-ent .l1 b')?.textContent, leg: document.querySelector('.view.on .stitle .leg, .view.on .et-ent .lg')?.textContent,
        l: r ? Math.round(r.width) : 0, h: r ? Math.round(r.height) : 0 }; });
    verif('météo · PC : la vue « Météo du lieu » montre la courbe des températures reçues, à sa taille, sa légende dit Open-Meteo',
      (courbe.titre || '').toUpperCase() === 'MÉTÉO DU LIEU' && /Open-Meteo/.test(courbe.leg || '') && courbe.l > 400 && courbe.h > 250, JSON.stringify(courbe));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc_temperatures.png') });
    await g.click('#vues [data-vue="tableau"]');
    await g.waitForTimeout(250);
    const tab = await g.evaluate(() => { const rows = [...document.querySelectorAll('.view.on .age-tab tbody tr')]; const r0 = rows[0];
      return { n: rows.length, vu: !!r0 && r0.checkVisibility(), cases: r0 ? [...r0.children].map((x) => x.textContent) : [] }; });
    // (Lot Q : le tableau suit le profil jusqu'à l'horizon de la courbe, 28 jours réels au moins — le harnais suit)
    verif('météo · PC : le tableau montre une ligne par jour reçu (date, température, source)',
      tab.n >= nJ && tab.vu && tab.cases[1] === fr(debut) && tab.cases[3] === 'mesuré', JSON.stringify(tab));
    // la note rechargée : tout est gardé, et rien n'est redemandé à Open-Meteo (le calcul se refait hors ligne)
    const nAppels = g.journal.length;
    await g.reload();
    await pretEcran(g);
    await tranquille(g);
    await ouvrirTp(g);
    const R = await lire(g);
    verif('météo · PC : rechargée, la note garde les températures, le badge et la maturité — sans AUCUN nouvel appel (calcul hors ligne)',
      R.releve === L.releve && R.hints.includes(badge) && R.verdict === L.verdict && g.journal.length === nAppels, JSON.stringify({ appels: g.journal.length - nAppels, v: R.verdict }));
    // le détail horaire, à la demande : la position, sans géocodage ; T min h / T max h dans le tableau
    await g.click('#donnees [data-meteo-chemin="temperature/meteo_horaire"]');
    await tranquille(g);
    const hor = g.journal.slice(nAppels);
    await g.click('#vues [data-vue="tableau"]');
    await g.waitForTimeout(250);
    const th = await g.evaluate(() => [...document.querySelectorAll('.view.on .age-tab thead th')].map((x) => x.textContent));
    const t0 = await g.evaluate(() => [...document.querySelector('.view.on .age-tab tbody tr').children].map((x) => x.textContent));
    const hs = [...Array(24).keys()].map((h) => tHeure(debut, h));
    const f1 = (x) => (Math.round(x * 10) / 10).toFixed(1).replace('.', ',').replace('-', '−');
    verif('météo · PC : « Charger le détail horaire » : appels horaires à la position trouvée (sans géocodage) ; le tableau ajoute T min h, T max h',
      hor.length === plan.appels.length && hor.every((u) => /latitude=50\.85045&longitude=4\.34878/.test(u) && /hourly=temperature_2m/.test(u) && !/geocoding/.test(u))
      && th.includes('T min h°C') && th.includes('T max h°C') && t0[4] === f1(Math.min(...hs)) && t0[5] === f1(Math.max(...hs)), JSON.stringify({ hor, th, t0 }));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc_tableau.png') });
    // une correction à la main : le calcul la suit, l'écran le dit, la récupération reste
    const corrige = L.releve.replace(/^([^;]+);[^;]+;/, '$1;-4,0;');
    await g.fill('#donnees textarea[data-chemin="temperature/releve"]', corrige);
    await g.evaluate(() => document.activeElement.blur());
    await tranquille(g);
    const C2 = await lire(g);
    verif('météo · PC : une température corrigée à la main recalcule la maturité ; l\'écran dit « corrigées à la main » et garde la récupération',
      C2.releve === corrige && C2.verdict !== R.verdict && C2.hints.some((x) => /corrigées à la main/.test(x)) && C2.hints.includes(badge.replace(`${nMes} j mesurés`, `${nMes} j mesurés`)),
      JSON.stringify({ v: C2.verdict, hints: C2.hints.slice(0, 8) }));
    // les refus : chacun en message, la note intacte
    const refus = async (nom, avant, motif) => {
      const r0 = await lire(g);
      const n0 = g.journal.length;
      await avant();
      await g.click('#donnees [data-meteo-chemin="temperature/meteo"]');
      await tranquille(g);
      await g.waitForTimeout(200);
      const r1 = await lire(g);
      verif(`météo · PC · ${nom} : un message en français, la note intacte (températures et maturité inchangées), jamais une exception`,
        r1.toast.vu && r1.toast.erreur && motif.test(r1.toast.texte) && r1.releve === r0.releve && r1.verdict === r0.verdict && !r1.bouton.grise,
        JSON.stringify({ toast: r1.toast, appels: g.journal.slice(n0) }));
      return g.journal.slice(n0);
    };
    await saisirDate(g, 'temperature/lieu', 'Atlantide');
    const a1 = await refus('lieu introuvable', async () => {}, /^Lieu introuvable : « Atlantide »/);
    verif('météo · PC · lieu introuvable : le géocodage seul est demandé', a1.length === 1 && /geocoding-api.*name=Atlantide/.test(a1[0]), JSON.stringify(a1));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc_introuvable.png') });
    await saisirDate(g, 'temperature/lieu', 'Bruxelles, Belgique');
    await refus('réseau coupé', async () => { g.mode.coupe = true; }, /^Pas de réseau : la météo n'a pas pu être récupérée.*hors ligne/);
    g.mode.coupe = false;
    await refus('service en panne', async () => { g.mode.panne = true; }, /^Service météo en panne \(erreur 500 — Panne simulée du service\)/);
    g.mode.panne = false;
    // (relecture adverse) une réponse dont les jours ne sont pas ceux demandés, une réponse énorme : refusées en message
    await refus('archive décalée d\'un jour', async () => { g.mode.decale = true; }, new RegExp(`^Météo refusée — réponse incomplète du service météo : le ${fr(debut)} manque`));
    g.mode.decale = false;
    const aE = await refus('réponse énorme (8 Mo)', async () => { g.mode.enorme = true; }, /^Réponse trop volumineuse du service météo \(plus de 5 Mo\)/);
    g.mode.enorme = false;
    verif('météo · PC · réponse énorme : refusée dès le géocodage, aucun autre appel', aE.length === 1 && /geocoding-api/.test(aE[0]), JSON.stringify(aE));
    // (relecture adverse) un AUTRE coulage ouvert pendant la récupération : la météo allait dans le coulage ouvert au retour —
    // une copie (même lieu, mêmes dates) perdait ses corrections à la main, et celui du clic restait sans météo
    await g.click(`${LIGNE_ELT} [data-dupliquer-index]`);
    await tranquille(g);
    await activer(g, 0);
    await tranquille(g);
    const c1 = await lire(g);
    g.mode.lent = 2500;
    await g.click('#donnees [data-meteo-chemin="temperature/meteo"]');
    await attendre(g, () => !!document.querySelector('#donnees [data-meteo].attente'), 3000);
    await activer(g, 1);
    await attendre(g, () => !document.querySelector('#donnees [data-meteo].attente'), 30000);
    await tranquille(g);
    await g.waitForTimeout(200);
    g.mode.lent = 0;
    const c2 = await lire(g);
    await activer(g, 0);
    await tranquille(g);
    const c1b = await lire(g);
    verif('météo · PC · un autre coulage ouvert pendant la récupération : rien n\'est importé, un message le dit ; la copie ouverte garde ses corrections à la main, le coulage du clic aussi',
      c2.toast.vu && c2.toast.erreur && /^Un autre coulage a été ouvert pendant la récupération/.test(c2.toast.texte) && c2.releve === c1.releve
      && c1b.releve === c1.releve && c2.hints.some((x) => /corrigées à la main/.test(x)), JSON.stringify({ toast: c2.toast, r2: (c2.releve || '').slice(0, 40), r1: (c1.releve || '').slice(0, 40) }));
    // dates hors de portée : le bouton est grisé, le message est écrit dessous — aucun appel
    const n0 = g.journal.length;
    await saisirDate(g, 'temperature/fin', fr(decale(auj, 30)));
    await saisirDate(g, 'temperature/debut', fr(decale(auj, 25)));
    const H = await lire(g);
    const msgVu = await g.evaluate(() => { const m = document.querySelector('#donnees .meteo-bt + .msg.grave'); return !!m && m.checkVisibility() && m.getBoundingClientRect().height > 10 ? m.textContent : ''; });
    verif('météo · PC · dates hors de portée : le bouton grisé, le message (horizon des prévisions) visible dessous, aucun appel',
      H.bouton.grise && /horizon des prévisions/.test(msgVu) && g.journal.length === n0, JSON.stringify({ b: H.bouton, msgVu }));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc_hors_portee.png') });
    const csp = await g.evaluate(() => window.__csp);
    // les erreurs attendues : la ressource refusée (réseau coupé) et la réponse 500 de la panne simulée
    const err = g.erreurs.filter((x) => !/Failed to load resource: (net::ERR_INTERNET_DISCONNECTED|the server responded with a status of 500)/.test(x));
    verif('météo · PC : aucune violation de la politique de sécurité, aucune autre adresse extérieure, aucune erreur JavaScript',
      !csp.length && !g.ailleurs.length && !err.length, JSON.stringify({ csp, ailleurs: g.ailleurs, err }));
    await g.context().close();

    // ------------------------------------------------ au téléphone et à la tablette : le même geste, dans l'onglet Données
    for (const [fmt, w, h, tel, nom] of [['téléphone', 390, 844, true, 'tel'], ['tablette portrait', 820, 1180, false, 'tab']]) {
      const t = await preparer(w, h, tel);
      await ouvrir(t);
      if (tel) await t.click('.mtabs [data-onglet="donnees"]');
      await ouvrirTp(t);
      await t.click('#donnees button[data-chemin="temperature/mode"][data-valeur="meteo"]');
      await tranquille(t);
      let M = await lire(t);
      const mm = await regard(t);
      verif(`météo · ${fmt} : la cure « Météo du lieu », le bouton visible et à la taille du doigt, rien ne déborde, aucun libellé coupé`,
        M.bouton && M.bouton.vu && M.bouton.h >= (tel ? 40 : 32) && M.bouton.l >= (tel ? 300 : 250) && !mm.deborde && !mm.coupes.length, JSON.stringify({ b: M.bouton, ...mm }));
      await t.locator('#donnees [data-meteo-chemin="temperature/meteo"]').scrollIntoViewIfNeeded();
      await t.click('#donnees [data-meteo-chemin="temperature/meteo"]');
      await tranquille(t);
      M = await lire(t);
      const n = jours(M.plan.debut, M.plan.fin).length;
      verif(`météo · ${fmt} : récupérée — le badge, la date de récupération et le relevé reçu (${n} j) se lisent dans les données`,
        M.hints.includes(`Bruxelles, Bruxelles-Capitale, Belgique — ${n} j mesurés`) && M.hints.some((x) => x.startsWith('Open-Meteo · récupérée le '))
        && M.releveVu && M.releve.split('\n').length === n && t.journal.length === 3, JSON.stringify({ hints: M.hints.slice(0, 5), appels: t.journal.length }));
      const m2 = await regard(t);
      verif(`météo · ${fmt} : après la récupération, rien ne déborde, aucun libellé coupé`, !m2.deborde && !m2.coupes.length, JSON.stringify(m2));
      await t.locator('#donnees .age-mini').scrollIntoViewIfNeeded();
      await t.screenshot({ path: path.join(OUT, `site_meteo_${nom}_donnees.png`) });
      if (tel) await t.click('.mtabs [data-onglet="dessin"]');
      await t.click('#vues [data-vue="temperatures"]');
      await t.waitForTimeout(300);
      const vue = await t.evaluate(() => { const s = [...document.querySelectorAll('.view.on .draw > svg')].find((x) => x.checkVisibility()); const r = s && s.getBoundingClientRect(); return r ? [Math.round(r.width), Math.round(r.height)] : null; });
      verif(`météo · ${fmt} : la vue des températures reçues est visible, à sa taille`, vue && vue[0] >= (tel ? 300 : 500) && vue[1] > 150, JSON.stringify(vue));
      await t.screenshot({ path: path.join(OUT, `site_meteo_${nom}_temperatures.png`) });
      await t.click('#vues [data-vue="tableau"]');
      await t.waitForTimeout(300);
      const tb = await t.evaluate(() => [...document.querySelectorAll('.view.on .age-tab tbody tr')].filter((x) => x.checkVisibility()).length);
      verif(`météo · ${fmt} : le tableau des jours reçus se voit`, tb >= n, `${tb} / ${n}`);
      await t.screenshot({ path: path.join(OUT, `site_meteo_${nom}_tableau.png`) });
      // le refus au téléphone : réseau coupé → un message visible
      if (tel) {
        await t.click('.mtabs [data-onglet="donnees"]');
        t.mode.coupe = true;
        await t.locator('#donnees [data-meteo-chemin="temperature/meteo"]').scrollIntoViewIfNeeded();
        await t.click('#donnees [data-meteo-chemin="temperature/meteo"]');
        await tranquille(t);
        await t.waitForTimeout(200);
        const T = await lire(t);
        const r = await t.evaluate(() => { const x = document.getElementById('toast').getBoundingClientRect(); return x.right <= innerWidth + 0.5 && x.left >= -0.5 && x.bottom <= innerHeight + 0.5; });
        verif('météo · téléphone · réseau coupé : le message « Pas de réseau » se lit, dans l\'écran ; les températures restent',
          T.toast.vu && T.toast.erreur && /^Pas de réseau/.test(T.toast.texte) && r && T.releve === M.releve, JSON.stringify(T.toast));
        await t.screenshot({ path: path.join(OUT, 'site_meteo_tel_pas_de_reseau.png') });
      }
      const err = t.erreurs.filter((x) => !/Failed to load resource: net::ERR_INTERNET_DISCONNECTED/.test(x));
      verif(`météo · ${fmt} : aucune violation de la politique de sécurité, aucune autre adresse, aucune erreur JavaScript`,
        !(await t.evaluate(() => window.__csp.length)) && !t.ailleurs.length && !err.length, JSON.stringify({ ailleurs: t.ailleurs, err }));
      await t.context().close();
    }
  }

  // ---------------------------------------------------------------- la SONDE du démarrage (js/sonde.js)
  // Une MESURE, fermée par défaut. Sans « ?sonde=1 », l'établi fait EXACTEMENT ce qu'il faisait : aucune marque
  // « stab: » sur la page, aucun message de plus entre la page et le travailleur (les types reçus sont lus sur le vrai
  // Worker), l'adresse du travailleur sans « sonde », le même écran (#donnees, #resultat, au caractère près, comparés
  // à ceux de la page sondée). Avec « ?sonde=1 » : les marques de la page et du travailleur, dans l'ordre du démarrage,
  // l'import du module daté à part de son premier calcul, sans erreur. Poutre (sans numpy) et charges sur plancher
  // (numpy chargé d'emblée), au PC et au téléphone.
  //   filtre « sonde ».
  if (!FILTRE || FILTRE === 'sonde') {
    await p.goto(`${BASE}/`);
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const espion = () => {
      window.__messages = []; window.__travailleurs = [];
      const W = window.Worker;
      window.Worker = class extends W {
        constructor(url, o) {
          super(url, o);
          window.__travailleurs.push(String(url));
          this.addEventListener('message', (ev) => window.__messages.push(ev.data && ev.data.type));
        }
      };
    };
    const charger = async (cle, fmt, sonde) => {
      const tel = fmt === 'téléphone';
      const c = await b.newContext({ viewport: tel ? { width: 390, height: 844 } : { width: 1440, height: 900 }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel });
      await c.addInitScript(espion);
      const pg = await c.newPage();
      const err = [];
      pg.on('pageerror', (e) => err.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') err.push(m.text().slice(0, 300)); });
      await pg.goto(`${BASE}/${MODS[cle].page}${sonde ? '?sonde=1' : ''}`);
      await pretEcran(pg);
      if (sonde) await pg.waitForFunction(() => globalThis.__sonde && performance.getEntriesByName('stab:premier_rendu').length, null, { timeout: 60000 });
      else await pg.waitForTimeout(1500);   // le temps qu'un message de trop arrive, s'il devait arriver
      const echec = await echecEcran(pg);
      const lu = await pg.evaluate((echec) => ({
        echec,
        page: performance.getEntriesByType('mark').filter((m) => m.name.startsWith('stab:')).map((m) => [m.name.slice(5), m.startTime]),
        travailleur: globalThis.__sonde ? globalThis.__sonde.marques.map(([k, t]) => [k, t - performance.timeOrigin]) : null,
        details: globalThis.__sonde ? globalThis.__sonde.details : null,
        sonde: globalThis.__sonde !== undefined,
        types: [...new Set(window.__messages)].sort(), adresses: window.__travailleurs,
        donnees: document.getElementById('donnees').innerHTML, resultat: document.getElementById('resultat').innerHTML,
      }), echec);
      await c.close();
      return { ...lu, err };
    };
    for (const cle of ['poutre', 'plancher']) {
      for (const fmt of ['PC', 'téléphone']) {
        const sans = await charger(cle, fmt, false);
        const avec = await charger(cle, fmt, true);
        const nom = `sonde · ${cle} · ${fmt}`;
        verif(`${nom} : sans « ?sonde=1 », le moteur démarre, aucune marque, aucun message de plus (étape, prêt, réponse), le travailleur à son adresse d'avant`,
          !sans.echec && !sans.page.length && !sans.sonde && JSON.stringify(sans.types) === '["etape","pret","reponse"]'
          && sans.adresses.length === 1 && !/sonde/.test(sans.adresses[0]) && !sans.err.length,
          JSON.stringify({ echec: sans.echec, page: sans.page, types: sans.types, adresses: sans.adresses, err: sans.err }).slice(0, 400));
        verif(`${nom} : avec ou sans la sonde, le même écran (#donnees, #resultat au caractère près)`,
          sans.donnees === avec.donnees && sans.resultat === avec.resultat && sans.donnees.length > 100,
          `${sans.donnees.length}/${avec.donnees.length} · ${sans.resultat.length}/${avec.resultat.length}`);
        // les marques, dans l'ordre du démarrage : la page, puis le travailleur, ramenés à la même origine
        const P = Object.fromEntries(avec.page), T = Object.fromEntries((avec.travailleur || []).filter(([k]) => !k.startsWith('wasm_')));
        const ordre = [['page', 'etabli_debut'], ['page', 'moteur_cree'], ['trav', 'travailleur_debut'], ['trav', 'manifeste_debut'], ['trav', 'loadPyodide_debut'],
          ['trav', 'loadPyodide_fin'], ['trav', 'empreinte_debut'], ['trav', 'deballage_fin'], ['trav', 'numpy_attendu'], ['trav', 'routeur_importe'],
          ['trav', 'pret'], ['page', 'pret_recu'], ['page', 'requete1_envoyee'], ['trav', 'module_import_debut'], ['trav', 'module_import_fin'],
          ['trav', 'requete1_debut'], ['trav', 'requete1_traitee'], ['trav', 'reponse1_postee'], ['page', 'reponse1_recue'],
          ['page', 'reponse1_appliquee'], ['page', 'premier_rendu']];
        const temps = ordre.map(([o, k]) => (o === 'page' ? P : T)[k]);
        const manque = ordre.filter((_, i) => temps[i] == null).map(([o, k]) => `${o}.${k}`);
        const desordre = ordre.filter((_, i) => i && temps[i] != null && temps[i - 1] != null && temps[i] < temps[i - 1] - 2).map(([o, k]) => `${o}.${k}`);
        const wasm = (avec.travailleur || []).some(([k]) => k.startsWith('wasm_'));
        // plancher (numpy) : chargé D'EMBLÉE, c'est-à-dire parti pendant le démarrage, en parallèle du paquet — avant la
        // fin de son déballage (l'écran annonce sa clé au montage, js/moteur.js annoncer) ; pas à la première requête,
        // après « pret » (revue S4, preuves-numpy-demblee)
        const numpy = cle === 'plancher' ? T.numpy_debut != null && T.numpy_fin != null && T.numpy_import_fin != null && T.deballage_fin != null
          && T.numpy_debut < T.deballage_fin : T.numpy_debut == null;
        verif(`${nom} : avec « ?sonde=1 », les marques de la page et du travailleur dans l'ordre du démarrage (wasm, numpy ${cle === 'plancher' ? "chargé d'emblée, parti avant la fin du déballage du paquet" : 'absent'})`,
          !avec.echec && !manque.length && !desordre.length && wasm && numpy && !avec.err.length,
          JSON.stringify({ echec: avec.echec, manque, desordre, wasm, numpy, numpy_debut: T.numpy_debut, deballage_fin: T.deballage_fin, pret: T.pret, err: avec.err }).slice(0, 500));
        const D = avec.details || {};
        verif(`${nom} : la sonde sépare l'import du module de son premier calcul, sans erreur, et mesure une requête ordinaire`,
          !D.import_erreur && !D.sonde_erreur && D.modules_apres_module > D.modules_apres_routeur && D.calculer_ms > 0 && D.cles_ms > 0
          && JSON.stringify(avec.types) === '["etape","pret","reponse","sonde"]', JSON.stringify({ ...D, types: avec.types }).slice(0, 400));
      }
    }
  }

  // ---------------------------------------------------------------- l'enregistrement, vu à l'écran
  // Un échec d'enregistrement se LIT à toutes les largeurs (1 440, 1 280, 820 et 390 px) : l'indicateur .et-enr — la
  // barre du haut au PC et à la tablette, la 2ᵉ ligne de l'en-tête au téléphone — passe au rouge, en entier, et y RESTE
  // (au-delà des 7 s d'un message éphémère) ; le message #et-alerte le dit une fois, avec « Exporter .stab », qui
  // exporte bien la note ; fermé, il ne revient pas à chaque saisie ; une écriture qui réussit efface tout. Défaut d'avant :
  // la charte cachait le texte (.saved .txt) sous 1 360 px — à 1 280 et 820 px, une coche grise, muette ; au téléphone,
  // rien tant que le menu restait fermé. Puis la mémoire des notes ABÎMÉE : jamais réécrite sans que son texte soit
  // copié à part ; la copie impossible, rien n'est écrit et l'écran le dit ; l'accueil le signale et rend le texte tel quel.
  // Au téléphone (390 et 360 px, poutre et plancher), la 2ᵉ ligne de l'en-tête tient entre le retour et la pastille de la
  // note : l'échec y écrit « Non enregistré » seul, entier, dans sa colonne, jamais sous la pastille — même la plus large
  // que le moteur écrit (PASTILLE_LA_PLUS_LARGE) ; la raison est dans le title, le message, et le menu ⋯, dont l'entrée
  // rouvre le message. Défaut d'avant (relecture du lot 0 bis) : « Non enregistré : stockage plein » passait SOUS la
  // pastille « À compléter » du plancher à 390 px, et « … mémoire illisible » sous toute pastille à 360 px.
  //   filtre « enregistrement »
  if (!FILTRE || FILTRE === 'enregistrement') {
    const fs = require('fs');
    await p.goto(`${BASE}/`);
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const CLE = 'stab.notes.v1', PREFIXE = 'stab.notes.v1.illisible.';
    // la pastille de la note la plus large que l'en-tête du téléphone puisse porter : un statut sans taux. COPIE NOMMÉE de
    // api_web/commun.py, STATUTS["x"] ; tests/test_site_web.py la lie (elle est le plus long des statuts sans taux).
    const PASTILLE_LA_PLUS_LARGE = 'Calcul impossible';
    // la mémoire d'un bureau, abîmée : 30 notes, le JSON tronqué d'un octet (la mesure du jury)
    const trente = {};
    for (let i = 0; i < 30; i++) {
      const id = `abime${String(i).padStart(2, '0')}`;
      trente[id] = { id, module: 'poutre', nom: `Poutre P${i + 1}`, chantier: 'Réalex', dossier: 'Fondations', indice: 'A', cree: 1762500000000 + i,
        modifie: 1762500000000 + i, etat: null, resume: { taux: 40 + i, niveau: 'p' }, vignette: '' };
    }
    const ABIME = JSON.stringify(trente).slice(0, -1);
    const ouvrirContexte = async (w, h, tel, tactile, init) => {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tactile, acceptDownloads: true });
      if (init) await c.addInitScript(init);
      const pg = await c.newPage();
      pg.erreurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error' && !/enregistrement impossible|copie de la mémoire/.test(m.text())) pg.erreurs.push(m.text().slice(0, 300)); });
      return pg;
    };
    const demarre = async (pg, o) => {
      await pretEcran(pg, o);
      return echecEcran(pg);
    };
    const calmePg = (pg) => pg.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
    // une saisie dans le premier champ numérique VISIBLE des données (une valeur qui change à chaque appel)
    let saisies = 0;
    const saisir = async (pg) => {
      const sel = await pg.evaluate(() => {
        const e = [...document.querySelectorAll('#donnees input[data-sorte="nombre"]:not([disabled])')].find((x) => x.checkVisibility());
        return e ? e.dataset.chemin : null;
      });
      if (!sel) return false;
      const champ = `#donnees input[data-chemin="${sel}"]`;
      const v = Number(String(await pg.inputValue(champ)).replace(/\s/g, '').replace(',', '.')) || 10;
      await pg.fill(champ, String(Math.round(v * (1.01 + 0.01 * (++saisies % 5)) * 100) / 100).replace('.', ','));
      await pg.press(champ, 'Enter');
      await calmePg(pg);
      await pg.waitForTimeout(150);
      return true;
    };
    // ce que l'œil voit de l'indicateur VISIBLE à ce format (il y en a un, et un seul) : son état, son texte LU (innerText :
    // ce que les règles laissent voir, pas ce que le DOM garde), entier (ni caché, ni coupé par un parent, ni couvert),
    // dans sa colonne au téléphone (entre le retour et la pastille), sa couleur comparée à celle de --nok ; le message ; et
    // l'entrée du menu ⋯ qui dit l'échec en entier
    const lireEnr = (pg) => pg.evaluate(() => {
      const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const vis = [...document.querySelectorAll('.et-enr')].filter(vu);
      const e = vis[0];
      const t = e ? e.querySelector('.et-enr-txt') : null;
      const sonde = document.createElement('span'); sonde.style.color = 'var(--nok)'; document.body.appendChild(sonde);
      const rouge = getComputedStyle(sonde).color; sonde.remove();
      let entier = false, couvert = [], boite = null, colonne = true;
      if (t && vu(t)) {
        const rg = document.createRange(); rg.selectNodeContents(t);
        const r = rg.getBoundingClientRect();
        boite = { l: Math.round(r.width), h: Math.round(r.height), g: Math.round(r.left), d: Math.round(r.right), y: Math.round(r.top) };
        // le texte entier dans chaque parent qui rogne, et dans la fenêtre
        entier = r.width > 20 && r.height > 6 && r.left >= -0.5 && r.right <= innerWidth + 0.5 && r.top >= -0.5 && r.bottom <= innerHeight + 0.5
          && t.scrollWidth <= t.clientWidth + 1;
        for (let a = t.parentElement; a && entier; a = a.parentElement) {
          if (getComputedStyle(a).overflow === 'visible') continue;
          const q = a.getBoundingClientRect();
          if (r.left < q.left - 0.5 || r.right > q.right + 0.5 || r.top < q.top - 0.5 || r.bottom > q.bottom + 0.5) entier = false;
        }
        // rien ne le couvre : trois points du texte VU (rogné par ses parents, s'il l'est) appartiennent à l'indicateur
        let g = r.left, d = r.right;
        for (let a = t; a; a = a.parentElement) {
          if (getComputedStyle(a).overflow === 'visible') continue;
          const q = a.getBoundingClientRect();
          g = Math.max(g, q.left); d = Math.min(d, q.right);
        }
        for (const x of [g + 3, (g + d) / 2, d - 3]) {
          const f = document.elementFromPoint(x, (r.top + r.bottom) / 2);
          if (!f || !f.closest('.et-enr')) couvert.push(f ? (f.id || f.className || f.tagName).toString().slice(0, 30) : 'rien');
        }
        // au téléphone, l'indicateur entier reste dans sa colonne (.mtop .t), sans déborder vers la pastille
        const col = e.closest('.mtop .t');
        if (col) {
          const q = col.getBoundingClientRect(), ri = e.getBoundingClientRect();
          colonne = ri.left >= q.left - 0.5 && ri.right <= q.right + 0.5;
        }
      }
      const chip = document.getElementById('chip-m');
      const mi = document.querySelector('[data-revoir-alerte]');
      const rmi = mi && vu(mi) ? mi.getBoundingClientRect() : null;
      const al = document.getElementById('et-alerte');
      const bt = al && al.querySelector('.et-alerte-stab');
      const ra = al && vu(al) ? al.getBoundingClientRect() : null;
      const rb = bt && vu(bt) ? bt.getBoundingClientRect() : null;
      // la couleur de ce qui s'écrit : chaque élément VU qui porte du texte (pas seulement le conteneur — une règle de la
      // charte peut repeindre un <span> intérieur)
      const porteurs = (x) => [x, ...x.querySelectorAll('*')].filter((y) => vu(y) && [...y.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim()));
      const couleurs = (x) => (x ? porteurs(x).map((y) => getComputedStyle(y).color) : []);
      const enRouge = (x) => { const c = couleurs(x); return c.length > 0 && c.every((y) => y === rouge); };
      return { visibles: vis.length, etat: e ? e.dataset.enr : null, texte: t ? t.innerText : null, titre: e ? e.title : '', entier, couvert, colonne, boite,
        rouge: enRouge(t), couleur: [...new Set(couleurs(t))].join(' / '),
        pastille: chip && vu(chip) ? `${chip.textContent} · ${Math.round(chip.getBoundingClientRect().width)} px` : '',
        menu: { cache: !mi || mi.hidden, vu: !!rmi, texte: mi ? mi.innerText : null, h: rmi ? Math.round(rmi.height) : 0,
          rouge: enRouge(mi) },
        alerte: ra ? { texte: al.querySelector('.et-alerte-txt').textContent, info: al.classList.contains('et-info'),
          dedans: ra.left >= -0.5 && ra.right <= innerWidth + 0.5 && ra.top >= -0.5 && ra.bottom <= innerHeight + 0.5,
          bouton: rb ? { texte: bt.textContent.trim(), h: Math.round(rb.height), l: Math.round(rb.width) } : null } : null,
        deborde: document.documentElement.scrollWidth > innerWidth };
    });
    // un établi ouvert sans note (TOUT module depuis le 10/11/2026 — c'était l'établi compact seul) : rien n'est écrit tant
    // qu'on ne modifie rien — la mémoire reste à l'octet, l'indicateur dit « Pas encore enregistré » ; la première saisie
    // écrit (ou échoue)
    const ecrireAlaSaisie = async (pg, nom) => {
      if (!(await pg.evaluate(() => !new URLSearchParams(location.search).get('note')))) return;
      const avant = await lireMemoire(pg), e0 = await lireEnr(pg);
      verif(`${nom} : à l'ouverture, RIEN n'est écrit (ni note, ni copie) et l'indicateur dit « Pas encore enregistré »`,
        avant.principal === ABIME && !Object.keys(avant.copies).length && e0.etat === 'neuf' && e0.texte === 'Pas encore enregistré',
        JSON.stringify({ copies: Object.keys(avant.copies).length, etat: e0.etat, texte: e0.texte }));
      verif(`${nom} : une saisie part au moteur`, await saisir(pg));
    };
    const lireMemoire = (pg) => pg.evaluate((k) => {
      const out = { principal: localStorage.getItem(k), copies: {} };
      for (let i = 0; i < localStorage.length; i++) { const c = localStorage.key(i); if (c.startsWith(k + '.illisible.')) out.copies[c] = localStorage.getItem(c); }
      return out;
    }, CLE);
    const telechargement = async (pg, sel) => {
      const [dl] = await Promise.all([pg.waitForEvent('download', { timeout: 15000 }), pg.click(sel)]);
      return { nom: dl.suggestedFilename(), texte: fs.readFileSync(await dl.path(), 'utf8') };
    };

    // ------------------------------------------------ 1. le stockage plein, aux quatre largeurs, et au téléphone de 360 px
    // (la poutre ; le plancher, dont la pastille « À compléter » est la plus large des pastilles d'ouverture)
    // CE QUI SE LIT : la barre du haut (PC, tablette) dit l'échec EN ENTIER ; la 2ᵉ ligne du téléphone, « Non enregistré »
    for (const [fmt, w, h, tel, tactile, attendre, mod] of [['PC 1440', 1440, 900, false, false, true, 'poutre'],
      ['portable 1280', 1280, 800, false, false, false, 'poutre'], ['tablette 820', 820, 1180, false, true, false, 'poutre'],
      ['téléphone 390', 390, 844, true, true, true, 'poutre'], ['téléphone 390', 390, 844, true, true, false, 'plancher'],
      ['téléphone 360', 360, 780, true, true, false, 'plancher'], ['téléphone 360', 360, 780, true, true, false, 'poutre']]) {
      const nom = `enregistrement · ${mod} · ${fmt}`;
      const lu = tel ? 'Non enregistré' : 'Non enregistré : stockage plein';
      const pg = await ouvrirContexte(w, h, tel, tactile);
      try {
        const ko0 = await demarre(pg, { apres: await ouvrir(pg, `${BASE}/${MODS[mod].page}?nom=${encodeURIComponent(`Essai d'enregistrement · ${fmt}`)}&chantier=R%C3%A9alex`) });
        verif(`${nom} : le moteur démarre`, !ko0, ko0);
        let e = await lireEnr(pg);
        verif(`${nom} : l'indicateur d'enregistrement est visible, un seul, « enregistré » après le premier calcul`,
          e.visibles === 1 && e.etat === 'ok' && e.entier && /\d{2}:\d{2}/.test(e.texte) && !e.alerte, JSON.stringify(e));
        // le stockage devient plein : chaque écriture lève l'exception du navigateur
        await pg.evaluate(() => {
          window.__setItem = Storage.prototype.setItem;
          Storage.prototype.setItem = function () { throw new DOMException('Le quota est dépassé', 'QuotaExceededError'); };
        });
        verif(`${nom} : une saisie part au moteur`, await saisir(pg));
        e = await lireEnr(pg);
        verif(`${nom} : l'ÉCHEC se lit — « ${lu} », visible, entier, non couvert${tel ? ', dans sa colonne' : ''}, en rouge ; la raison au survol`,
          e.visibles === 1 && e.etat === 'echec' && e.texte === lu && e.entier && !e.couvert.length && e.colonne && e.rouge && !e.deborde
          && e.titre === 'Non enregistré : le stockage de ce navigateur est plein', JSON.stringify(e));
        verif(`${nom} : le message le dit, dans l'écran, avec « Exporter .stab »`,
          e.alerte && !e.alerte.info && e.alerte.dedans && /^Non enregistré : le stockage de ce navigateur est plein\./.test(e.alerte.texte)
          && e.alerte.bouton && e.alerte.bouton.texte === 'Exporter .stab' && e.alerte.bouton.h >= (tactile ? 44 : 24), JSON.stringify(e.alerte));
        await pg.screenshot({ path: path.join(OUT, `site_enregistrement_echec_${mod}_${fmt.split(' ')[1]}.png`) });
        if (attendre) {
          // persistant : bien après les 7 s d'un message éphémère
          await pg.waitForTimeout(7600);
          const e2 = await lireEnr(pg);
          verif(`${nom} : 7,6 s plus tard, l'échec et le message sont toujours là`, e2.etat === 'echec' && e2.entier && e2.rouge && e2.alerte, JSON.stringify(e2));
        }
        // « Exporter .stab » : la note entière, rechargeable
        const f = await telechargement(pg, '#et-alerte .et-alerte-stab');
        let stab = null;
        try { stab = JSON.parse(f.texte); } catch (x) { /* dit plus bas */ }
        verif(`${nom} : « Exporter .stab » du message exporte la note (son état de calcul compris)`,
          /\.stab$/.test(f.nom) && stab && stab.format === 'stab' && stab.note && stab.note.module === mod && stab.note.etat && typeof stab.note.etat === 'object',
          `${f.nom} · ${f.texte.replace(/\s+/g, ' ').slice(0, 80)}`);
        // fermé, le message ne revient pas à la saisie suivante ; l'indicateur, lui, reste rouge
        await pg.click('#et-alerte [data-fermer-alerte]');
        await saisir(pg);
        e = await lireEnr(pg);
        verif(`${nom} : le message fermé ne revient pas à chaque saisie ; l'indicateur reste rouge`, !e.alerte && e.etat === 'echec' && e.entier && e.rouge, JSON.stringify(e));
        if (tel) {
          // la raison, au doigt : le menu ⋯ dit l'échec EN ENTIER (une entrée de 44 px au moins, en rouge), et la toucher
          // rouvre le message, avec « Exporter .stab »
          await pg.click('#menu-m');
          await pg.waitForTimeout(150);
          const e1 = await lireEnr(pg);
          verif(`${nom} : le menu ⋯ dit l'échec en entier — « Non enregistré : stockage plein », en rouge, une cible du doigt`,
            e1.menu.vu && e1.menu.texte === 'Non enregistré : stockage plein' && e1.menu.h >= 44 && e1.menu.rouge, JSON.stringify(e1.menu));
          await pg.screenshot({ path: path.join(OUT, `site_enregistrement_echec_${mod}_${fmt.split(' ')[1]}_menu.png`), clip: { x: 0, y: 0, width: w, height: 420 } });
          await pg.click('[data-revoir-alerte]');
          await pg.waitForTimeout(150);
          const e2 = await lireEnr(pg);
          const menuFerme = await pg.evaluate(() => document.getElementById('mmenu').hidden);
          verif(`${nom} : le toucher rouvre le message (la raison, « Exporter .stab ») et referme le menu`,
            menuFerme && e2.alerte && !e2.alerte.info && e2.alerte.dedans && /^Non enregistré : le stockage de ce navigateur est plein\./.test(e2.alerte.texte)
            && e2.alerte.bouton && e2.alerte.bouton.texte === 'Exporter .stab' && e2.alerte.bouton.h >= 44, JSON.stringify(e2.alerte));
          await pg.click('#et-alerte [data-fermer-alerte]');
          // la pastille la plus large que le moteur écrive (un statut sans taux) : l'échec reste entier, dans sa colonne,
          // jamais dessous (la pastille reprend son texte à la réponse suivante)
          await pg.evaluate((t) => { const c = document.getElementById('chip-m'); c.textContent = t; c.hidden = false; }, PASTILLE_LA_PLUS_LARGE);
          const e3 = await lireEnr(pg);
          verif(`${nom} : sous la pastille la plus large (« ${PASTILLE_LA_PLUS_LARGE} »), « Non enregistré » reste entier, dans sa colonne, non couvert`,
            e3.etat === 'echec' && e3.texte === lu && e3.entier && !e3.couvert.length && e3.colonne && e3.rouge, JSON.stringify(e3));
          await pg.screenshot({ path: path.join(OUT, `site_enregistrement_echec_${mod}_${fmt.split(' ')[1]}_pastille.png`), clip: { x: 0, y: 0, width: w, height: 120 } });
          if (w === 360) {
            // plus étroit encore (320 px, hors des formats du bureau) : la place manque, le texte se termine par « … » —
            // dans sa colonne, en rouge, jamais sous la pastille
            await pg.setViewportSize({ width: 320, height: h });
            await pg.waitForTimeout(150);
            const e4 = await lireEnr(pg);
            verif(`${nom} : à 320 px sous la même pastille, « Non enregistré » peut finir par « … » mais reste dans sa colonne, en rouge, non couvert`,
              e4.etat === 'echec' && e4.texte === lu && !e4.couvert.length && e4.colonne && e4.rouge && !e4.deborde, JSON.stringify(e4));
            await pg.screenshot({ path: path.join(OUT, `site_enregistrement_echec_${mod}_320_pastille.png`), clip: { x: 0, y: 0, width: 320, height: 120 } });
            await pg.setViewportSize({ width: w, height: h });
          }
        }
        // le stockage revient : une écriture réussie efface l'échec
        await pg.evaluate(() => { Storage.prototype.setItem = window.__setItem; });
        await saisir(pg);
        e = await lireEnr(pg);
        verif(`${nom} : le stockage revenu, la saisie suivante s'enregistre et l'indicateur le dit (et le menu ⋯ ne dit plus d'échec)`,
          e.etat === 'ok' && e.entier && !e.rouge && !e.alerte && e.menu.cache, JSON.stringify(e));
        verif(`${nom} : aucune erreur JavaScript`, !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
      } catch (x) { verif(`${nom} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      await pg.context().close();
    }

    // ------------------------------------------------ 2. la mémoire des notes abîmée, et sa copie impossible
    // (chaque copie « stab.notes.v1.illisible.* » est refusée par le navigateur ; les autres écritures passent)
    const refuserCopies = () => {
      const o = Storage.prototype.setItem;
      Storage.prototype.setItem = function (k, v) {
        if (String(k).startsWith('stab.notes.v1.illisible.')) throw new DOMException('Le quota est dépassé', 'QuotaExceededError');
        return o.call(this, k, v);
      };
    };
    for (const [fmt, w, h, tel, mod] of [['PC', 1440, 900, false, 'poutre'], ['téléphone', 390, 844, true, 'poutre'], ['téléphone 360', 360, 780, true, 'plancher']]) {
      const nom = `mémoire abîmée, copie impossible · ${mod} · ${fmt}`;
      const lu = tel ? 'Non enregistré' : 'Non enregistré : mémoire illisible';
      const pg = await ouvrirContexte(w, h, tel, tel, refuserCopies);
      try {
        await ouvrir(pg, `${BASE}/`);
        await pg.evaluate(([k, v]) => localStorage.setItem(k, v), [CLE, ABIME]);
        const ko0 = await demarre(pg, { apres: await ouvrir(pg, `${BASE}/${MODS[mod].page}`) });
        verif(`${nom} : le moteur démarre`, !ko0, ko0);
        await ecrireAlaSaisie(pg, nom);
        const e = await lireEnr(pg);
        const mem = await lireMemoire(pg);
        verif(`${nom} : RIEN n'est écrit — la mémoire abîmée reste à l'octet, aucune copie`, mem.principal === ABIME && !Object.keys(mem.copies).length,
          `${(mem.principal || '').length} caractères · ${Object.keys(mem.copies).length} copie(s)`);
        verif(`${nom} : l'écran le dit — « ${lu} », en rouge, entier, non couvert${tel ? ', dans sa colonne' : ''} ; le message « Non enregistré : la mémoire des notes est illisible » avec « Exporter .stab »`,
          e.etat === 'echec' && e.texte === lu && e.entier && !e.couvert.length && e.colonne && e.rouge && e.titre === 'Non enregistré : la mémoire des notes est illisible'
          && e.alerte && e.alerte.texte.startsWith('Non enregistré : la mémoire des notes est illisible.') && e.alerte.bouton && e.alerte.bouton.texte === 'Exporter .stab',
          JSON.stringify(e));
        if (tel) {
          await pg.click('#et-alerte [data-fermer-alerte]');
          await pg.click('#menu-m');
          await pg.waitForTimeout(150);
          const e1 = await lireEnr(pg);
          verif(`${nom} : le menu ⋯ dit l'échec en entier — « Non enregistré : mémoire illisible »`,
            e1.menu.vu && e1.menu.texte === 'Non enregistré : mémoire illisible' && e1.menu.h >= 44 && e1.menu.rouge, JSON.stringify(e1.menu));
          await pg.click('[data-revoir-alerte]');
          await pg.waitForTimeout(150);
          const e2 = await lireEnr(pg);
          verif(`${nom} : le toucher rouvre le message de la mémoire illisible`,
            e2.alerte && e2.alerte.texte.startsWith('Non enregistré : la mémoire des notes est illisible.') && e2.alerte.bouton, JSON.stringify(e2.alerte));
        }
        await pg.screenshot({ path: path.join(OUT, `site_enregistrement_illisible_${mod}_${tel ? 'tel' + w : 'pc'}.png`) });
        // l'accueil : il le signale et rend le texte brut, tel quel
        await ouvrir(pg, `${BASE}/`);
        await pg.waitForTimeout(300);
        const a = await pg.evaluate(() => {
          const z = document.getElementById('et-memoire');
          const r = z.getBoundingClientRect();
          return { vu: z.checkVisibility({ visibilityProperty: true, opacityProperty: true }), texte: z.textContent, h: Math.round(r.height), haut: Math.round(r.top),
            date: document.getElementById('date').textContent, deborde: document.documentElement.scrollWidth > innerWidth };
        });
        verif(`${nom} : l'accueil signale la mémoire illisible (ses notes ne s'affichent pas, rien n'est effacé)`,
          a.vu && /mémoire des notes de ce navigateur est illisible/.test(a.texte) && /illisible/.test(a.date) && !a.deborde && a.haut < 400, JSON.stringify(a));
        const f = await telechargement(pg, '#et-memoire [data-texte-brut]');
        verif(`${nom} : l'accueil rend le texte de la mémoire abîmée, à l'octet`, f.texte === ABIME && /\.txt$/.test(f.nom), `${f.nom} · ${f.texte.length} caractères`);
        await pg.screenshot({ path: path.join(OUT, `site_accueil_memoire_illisible_${tel ? 'tel' + w : 'pc'}.png`) });
        verif(`${nom} : aucune erreur JavaScript`, !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
      } catch (x) { verif(`${nom} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      await pg.context().close();
    }

    // ------------------------------------------------ 3. la mémoire abîmée, copiée : la note s'écrit, le texte survit
    {
      const nom = 'mémoire abîmée, copiée';
      const pg = await ouvrirContexte(1440, 900, false, false);
      try {
        await ouvrir(pg, `${BASE}/`);
        await pg.evaluate(([k, v]) => localStorage.setItem(k, v), [CLE, ABIME]);
        const ko0 = await demarre(pg, { apres: await ouvrir(pg, `${BASE}/${MODS.poutre.page}`) });
        verif(`${nom} : le moteur démarre`, !ko0, ko0);
        await ecrireAlaSaisie(pg, nom);
        const e = await lireEnr(pg);
        const mem = await lireMemoire(pg);
        const copies = Object.entries(mem.copies);
        let principal = null;
        try { principal = JSON.parse(mem.principal); } catch (x) { /* dit plus bas */ }
        verif(`${nom} : le texte d'origine est copié à part, à l'octet, une fois ; la note s'écrit dans une mémoire neuve`,
          copies.length === 1 && copies[0][1] === ABIME && principal && Object.keys(principal).length === 1, `${copies.length} copie(s) · ${mem.principal && mem.principal.slice(0, 60)}`);
        verif(`${nom} : l'écran dit « enregistré », et une fois que la mémoire était illisible et que son texte est gardé`,
          e.etat === 'ok' && e.alerte && e.alerte.info && /était illisible/.test(e.alerte.texte) && !e.alerte.bouton, JSON.stringify(e));
        await saisir(pg);
        const mem2 = await lireMemoire(pg);
        verif(`${nom} : la saisie suivante ne copie rien de plus`, Object.keys(mem2.copies).length === 1, Object.keys(mem2.copies).join(', '));
        await pg.close();
        verif(`${nom} : aucune erreur JavaScript`, !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
        // l'accueil, aux trois formats : la copie est signalée et se télécharge
        for (const [fmt, w, h, tel] of [['PC', 1440, 900, false], ['tablette', 820, 1180, false], ['téléphone', 390, 844, true]]) {
          const c2 = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: fmt !== 'PC', acceptDownloads: true,
            storageState: await pg.context().storageState() });
          const g = await c2.newPage();
          await ouvrir(g, `${BASE}/`);
          await g.waitForTimeout(300);
          const a = await g.evaluate(() => {
            const z = document.getElementById('et-memoire'), bt = z.querySelector('[data-copie]');
            const r = z.getBoundingClientRect(), rb = bt ? bt.getBoundingClientRect() : null;
            return { vu: z.checkVisibility({ visibilityProperty: true, opacityProperty: true }), texte: z.textContent.replace(/\s+/g, ' ').slice(0, 160),
              haut: Math.round(r.top), dedans: r.left >= 0 && r.right <= innerWidth, bouton: rb ? Math.round(rb.height) : 0, notes: document.querySelectorAll('#recentes .rcard').length,
              deborde: document.documentElement.scrollWidth > innerWidth };
          });
          verif(`${nom} · accueil ${fmt} : la copie gardée à part est signalée, dans l'écran, son bouton à la taille du ${tel ? 'doigt' : 'pointeur'}`,
            a.vu && /gardée à part/.test(a.texte) && a.dedans && !a.deborde && a.haut < 400 && a.bouton >= (tel ? 44 : 24) && a.notes === 1, JSON.stringify(a));
          const f = await telechargement(g, '#et-memoire [data-copie]');
          verif(`${nom} · accueil ${fmt} : la copie se télécharge, à l'octet`, f.texte === ABIME, `${f.nom} · ${f.texte.length} caractères`);
          await g.screenshot({ path: path.join(OUT, `site_accueil_memoire_copie_${fmt === 'téléphone' ? 'tel' : fmt.toLowerCase()}.png`) });
          await c2.close();
        }
      } catch (x) { verif(`${nom} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      await pg.context().close();
    }
  }

  // ---------------------------------------------------------------- ouvrir un calcul n'écrit rien (décision D1), TOUS les modules
  // (10/11/2026 ; c'était la poutre et la dalle seules — une relecture extérieure : « dès qu'il ouvrait un module, ça faisait
  // une note »). Ce que l'œil voit et ce que la mémoire garde :
  //  1. aux trois formats, chaque carte de l'accueil est un LIEN vers le calcul, que rien n'intercepte ;
  //  2. au PC, CHAQUE module (les masqués par leur page) : ouvert, rien n'est écrit (moteur prêt + 1,5 s), l'adresse reste
  //     sans ?note=, l'étiquette dit « Non enregistrée », l'indicateur « Pas encore enregistré », et « Enregistrer comme
  //     note… » est là, à sa taille ;
  //  3. « Enregistrer comme note… » aux trois formats (barre du haut ; menu ⋯ au téléphone) : la fenêtre tient dans l'écran,
  //     nomme, range, écrit UNE note ; l'adresse prend ?note= ; l'étiquette dit « Indice A » ;
  //  4. une saisie écrit la note ; rechargée, la même note, la même valeur ;
  //  5. rouvrir une note sans la modifier ne change pas sa date ; l'indice réel (C) dans l'en-tête, aux trois formats ; la
  //     première modification la date ;
  //  6. un lien ?note= vers une note ABSENTE : le message le dit, avec « Importer .stab » (qui ouvre bien le fichier choisi),
  //     et rien n'est créé ; une note d'un AUTRE module s'ouvre par la page de son module.
  //   filtre « ouverture »
  if (!FILTRE || FILTRE === 'ouverture') {
    const fs = require('fs');
    await p.goto(`${BASE}/`);
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const ctxO = async (w, h, tel, tactile = tel) => {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tactile, acceptDownloads: true });
      const pg = await c.newPage();
      pg.erreurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') pg.erreurs.push(m.text().slice(0, 300)); });
      return pg;
    };
    const pretO = async (pg, o) => {
      await pretEcran(pg, o);
      await pg.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
      return echecEcran(pg);
    };
    const notesO = (pg) => pg.evaluate(() => { try { return JSON.parse(localStorage.getItem('stab.notes.v1') || '{}'); } catch (e) { return null; } });
    // l'en-tête VU à ce format : l'étiquette de la barre du haut, la 2ᵉ ligne du téléphone, l'indicateur, le bouton (ou
    // l'entrée du menu ⋯) « Enregistrer comme note… » — chacun visible, à sa taille lue
    const enteteO = (pg) => pg.evaluate(() => {
      const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const taille = (e) => { const r = e.getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height)]; };
      const ind = document.getElementById('et-ind'), enr = [...document.querySelectorAll('.et-enr')].find(vu), bt = document.querySelector('.top [data-ranger]');
      return { adresse: location.search, etiquette: vu(ind) ? ind.textContent : null, tailleEtiquette: vu(ind) ? taille(ind) : null,
        filM: vu(document.getElementById('fil-m')) ? document.getElementById('fil-m').textContent : null,
        enr: enr ? enr.dataset.enr : null, enrTexte: enr ? enr.querySelector('.et-enr-txt').innerText : null,
        bouton: vu(bt) ? { taille: taille(bt), libelle: vu(bt.querySelector('.lbl')) ? bt.querySelector('.lbl').textContent : '' } : null,
        deborde: document.documentElement.scrollWidth > innerWidth };
    });
    // la barre du haut (PC, tablette) : rien ne s'y chevauche, rien n'en sort
    const barreO = (pg) => pg.evaluate(() => {
      const els = [...document.querySelectorAll('.top > *, .top .crumbs > *')].filter((e) => !e.classList.contains('crumbs') && !e.classList.contains('spacer')
        && e.checkVisibility({ visibilityProperty: true, opacityProperty: true })).map((e) => ({ n: (e.id || e.className || e.tagName).toString().slice(0, 20), r: e.getBoundingClientRect() }));
      const f = [], top = document.querySelector('.top').getBoundingClientRect();
      for (let i = 0; i < els.length; i++) for (let j = i + 1; j < els.length; j++) {
        const a = els[i].r, c = els[j].r;
        if (a.left < c.right - 0.5 && c.left < a.right - 0.5 && a.top < c.bottom && c.top < a.bottom) f.push(`${els[i].n} / ${els[j].n}`);
      }
      els.forEach((e) => { if (e.r.right > top.right + 0.5 || e.r.left < top.left - 0.5) f.push(`${e.n} sort de la barre`); });
      return f;
    });
    const FMT_O = [['PC', 1440, 900, false], ['tablette', 820, 1180, false, true], ['téléphone', 390, 844, true]];

    // ------------------------------------------------ 1. les cartes de l'accueil : des liens, et un clic OUVRE le calcul
    // (lot S4, 10/11/2026 : l'application attrape désormais le clic pour échanger la page sans la recharger — l'ancien
    // « rien n'intercepte » est devenu faux par construction. Ce qui doit tenir, et que D1 exigeait : la carte reste un
    // lien vers la page du calcul, et un VRAI clic y mène — sans recharger le document, sans fenêtre « Nouvelle note ».
    // Une carte par format, une autre à chaque format ; les autres se vérifient par leur lien.)
    const OUVERTES = Object.entries(MODS).filter(([, m]) => !m.masque).map(([k]) => k);
    for (const [i, [fmt, w, h, tel, tactile]] of FMT_O.entries()) {
      const pg = await ctxO(w, h, tel, tactile ?? tel);
      const cle = OUVERTES[(i * 5) % OUVERTES.length];
      try {
        const navs = [];
        pg.on('request', (r) => { if (r.isNavigationRequest() && r.frame() === pg.mainFrame()) navs.push(r.url()); });
        await pg.goto(`${BASE}/`);
        await pg.waitForTimeout(300);
        const t = await pg.evaluate((mods) => {
          window.__s = Math.random();   // la sentinelle : un document rechargé la perd
          const tuiles = [...document.querySelectorAll('.tile[data-module]')];
          return { n: tuiles.length, liens: tuiles.every((x) => x.tagName === 'A' && x.getAttribute('href') === mods[x.dataset.module].page),
            attendues: Object.values(mods).filter((m) => !m.masque).length, nouvelleModule: document.querySelectorAll('[data-nouvelle-module]').length, s: window.__s };
        }, MODS);
        verif(`ouverture · accueil ${fmt} : chacune des ${t.attendues} cartes de module est un lien vers son calcul (aucun bouton « Nouvelle note » par module)`,
          t.n === t.attendues && t.liens && !t.nouvelleModule, JSON.stringify(t));
        const n0 = navs.length;
        await pg.click(`.tile[data-module="${cle}"]`);
        await pg.waitForFunction((page) => location.pathname.endsWith(`/${page}`), MODS[cle].page, { timeout: 20000 }).catch(() => {});
        const a = await pg.evaluate(() => ({ chemin: location.pathname, recherche: location.search, s: window.__s,
          fenetre: !!document.getElementById('nouvelle') && !document.getElementById('nouvelle').hidden }));
        verif(`ouverture · accueil ${fmt} : un vrai clic sur la carte « ${cle} » ouvre son calcul SANS recharger la page (aucune requête de document, la sentinelle survit), sans ?note=, sans fenêtre « Nouvelle note »`,
          a.chemin.endsWith(`/${MODS[cle].page}`) && !a.recherche && a.s === t.s && navs.length === n0 && !a.fenetre,
          JSON.stringify({ ...a, attendu: MODS[cle].page, documents: navs.slice(n0) }));
      } catch (x) { verif(`ouverture · accueil ${fmt} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      await pg.context().close();
    }

    // ------------------------------------------------ 2. chaque module, au PC : ouvert depuis sa carte, rien n'est écrit
    {
      const pg = await ctxO(1440, 900, false);
      for (const [cle, mod] of Object.entries(MODS)) {
        const nom = `ouverture · ${cle} · PC`;
        try {
          await pg.goto(`${BASE}/`);
          let m0 = null;
          if (mod.masque) await pg.goto(`${BASE}/${mod.page}`);   // masqué : plus de carte, sa page reste servie
          else { m0 = await montage(pg); await Promise.all([pg.waitForNavigation(), pg.click(`.tile[data-module="${cle}"]`)]); }
          const ko = await pretO(pg, { apres: m0 });
          await pg.waitForTimeout(1500);   // le temps qu'une écriture de trop arrive, si elle devait arriver
          const e = await enteteO(pg), n = await notesO(pg);
          const vers = mod.vers ? MODS[mod.vers].page : mod.page;
          verif(`${nom} : ouvert${mod.masque ? ' (page d\'un module masqué)' : ' depuis sa carte'}, RIEN n'est écrit — aucune note, l'adresse sans ?note=, « Non enregistrée », « Pas encore enregistré », « Enregistrer comme note… » à sa taille`,
            !ko && n && !Object.keys(n).length && !/note=/.test(e.adresse) && new URL(pg.url()).pathname.endsWith(`/${vers}`) && e.etiquette === 'Non enregistrée'
            && e.enr === 'neuf' && e.enrTexte === 'Pas encore enregistré' && e.bouton && e.bouton.taille[1] >= 24 && e.bouton.libelle === 'Enregistrer comme note…' && !e.deborde,
            JSON.stringify({ ko, notes: n && Object.keys(n).length, ...e, url: pg.url() }));
          const f = await barreO(pg);
          verif(`${nom} : la barre du haut ne se chevauche pas (l'étiquette, l'indicateur, « Enregistrer comme note… »)`, !f.length, f.join(', '));
        } catch (x) { verif(`${nom} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      }
      verif('ouverture · PC : aucune erreur JavaScript', !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
      await pg.screenshot({ path: path.join(OUT, 'site_ouverture_pc.png') });
      await pg.context().close();
    }

    // ------------------------------------------------ 2 bis. à la tablette, le NOM de la note se lit en entier
    // (relecture du 10/11/2026 : « Enregistrer comme note… » sur toute page, ses cibles de 44 px et l'étiquette « Non
    // enregistrée » serraient le champ du nom à 110 px — « Assemblage… » — et le cachaient à la poutre en paysage, 1 px vu
    // à 1 180 px). Ce que l'œil voit : la part du champ que la barre ne coupe pas, contre la largeur du texte du nom.
    for (const [fmt, w, h] of [['tablette portrait', 820, 1180], ['tablette paysage', 1180, 820]]) {
      const pg = await ctxO(w, h, false, true);
      const coupes = [];
      for (const [cle, mod] of Object.entries(MODS)) {
        if (mod.vers) continue;
        try {
          await pg.goto(`${BASE}/${mod.page}`);
          const ko = await pretO(pg);
          const m = await pg.evaluate(() => {
            const cr = document.querySelector('.top .crumbs').getBoundingClientRect(), nom = document.getElementById('nom-note'), r = nom.getBoundingClientRect();
            const cs = getComputedStyle(nom), t = document.createElement('span');
            t.style.cssText = `position:absolute;white-space:pre;font:${cs.font}`; t.textContent = nom.value; document.body.appendChild(t);
            const texte = t.getBoundingClientRect().width; t.remove();
            const vu = Math.min(r.right, cr.right) - Math.max(r.left, cr.left);
            return { nom: nom.value, vu: Math.round(vu), texte: Math.round(texte), deborde: document.documentElement.scrollWidth > innerWidth };
          });
          // (le champ a ses marges intérieures et son cadre, 2 × 7 px : le texte tient dans la part vue moins 14 px)
          if (ko || m.deborde || m.vu < m.texte + 14) coupes.push(`${cle} : ${ko || ''}${JSON.stringify(m)}`);
        } catch (x) { coupes.push(`${cle} : ${String(x.message).split('\n')[0]}`); }
      }
      verif(`ouverture · ${fmt} (${w} px, au doigt) : à chaque module ouvert, le nom de la note se lit EN ENTIER dans la barre du haut, rien ne déborde`,
        !coupes.length, coupes.join(' | ').slice(0, 600));
      if (w === 1180) {
        await pg.goto(`${BASE}/${MODS.poutre.page}`);
        await pretO(pg);
        await pg.screenshot({ path: path.join(OUT, 'site_ouverture_nom_poutre_1180.png'), clip: { x: 0, y: 0, width: w, height: 64 } });
      }
      await pg.context().close();
    }

    // ------------------------------------------------ 3 à 5. le garde-corps aux trois formats : ranger, saisir, rouvrir
    const t0 = Date.now() - 3 * 86400000;   // une note de trois jours
    for (const [fmt, w, h, tel, tactile] of FMT_O) {
      const nom = `ouverture · garde-corps · ${fmt}`;
      const pg = await ctxO(w, h, tel, tactile ?? tel);
      try {
        await pg.goto(`${BASE}/`);
        const m0 = await montage(pg);
        await Promise.all([pg.waitForNavigation(), pg.click('.tile[data-module="garde_corps"]')]);
        verif(`${nom} : le moteur démarre`, !(await pretO(pg, { apres: m0 })));
        // « Enregistrer comme note… » : au téléphone, dans le menu ⋯
        if (tel) { await pg.click('#menu-m'); await pg.waitForTimeout(150); }
        const sel = tel ? '#mmenu [data-ranger]' : '.top [data-ranger]';
        const bt = await pg.evaluate((s) => { const e = document.querySelector(s); const r = e.getBoundingClientRect();
          return { vu: e.checkVisibility({ visibilityProperty: true, opacityProperty: true }), l: Math.round(r.width), h: Math.round(r.height), texte: e.innerText.trim() }; }, sel);
        verif(`${nom} : « Enregistrer comme note… » ${tel ? 'dans le menu ⋯' : 'dans la barre du haut'}, visible, à sa taille`,
          bt.vu && bt.h >= (tactile ?? tel ? 44 : 24) && bt.l >= bt.h && (tel || fmt === 'tablette' || bt.texte === 'Enregistrer comme note…') && (!tel || bt.texte === 'Enregistrer comme note…'), JSON.stringify(bt));
        await pg.click(sel);
        await pg.waitForTimeout(150);
        const fen = await pg.evaluate(() => { const c = document.querySelector('#et-rangement .et-rg-carte'), r = c.getBoundingClientRect();
          const champs = [...document.querySelectorAll('#et-rangement input, #et-rangement button')].filter((x) => x.checkVisibility()).map((x) => Math.round(x.getBoundingClientRect().height));
          return { vu: c.checkVisibility({ visibilityProperty: true, opacityProperty: true }), dedans: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight,
            focus: document.activeElement && document.activeElement.id, titre: document.getElementById('et-rg-titre').textContent, hMin: Math.min(...champs) }; });
        verif(`${nom} : la fenêtre « Enregistrer comme note » s'ouvre dans l'écran, le nom choisi, ses cibles à la taille du ${tactile ?? tel ? 'doigt (44 px)' : 'pointeur'}`,
          fen.vu && fen.dedans && fen.focus === 'et-rg-nom' && fen.titre === 'Enregistrer comme note' && fen.hMin >= (tactile ?? tel ? 44 : 28), JSON.stringify(fen));
        await pg.fill('#et-rg-nom', `Garde-corps rangé · ${fmt}`);
        await pg.fill('#et-rg-chantier', 'Essai ouverture');
        await pg.click('#et-rangement button[type="submit"]');
        await pg.waitForTimeout(250);
        let n = await notesO(pg), e = await enteteO(pg);
        const id = new URLSearchParams(e.adresse).get('note');
        verif(`${nom} : « Enregistrer » écrit UNE note, nommée et rangée ; l'adresse prend ?note= ; l'en-tête dit « Indice A »`,
          Object.keys(n).length === 1 && n[id] && n[id].nom === `Garde-corps rangé · ${fmt}` && n[id].chantier === 'Essai ouverture'
          && (tel ? e.filM && e.filM.startsWith('Indice A · Essai ouverture') : e.etiquette === 'Indice A') && e.enr === 'ok',
          JSON.stringify({ notes: Object.keys(n), ...e }));
        if (tel) { await pg.click('.mtabs [data-onglet="donnees"]'); await pg.waitForTimeout(150); }
        await pg.screenshot({ path: path.join(OUT, `site_ouverture_rangee_${tel ? 'tel' : fmt === 'PC' ? 'pc' : 'tablette'}.png`) });
        // une saisie (H = 1 300 mm) : la note change, sa date avance
        const avant = n[id].modifie;
        await pg.fill('input[data-chemin="geometrie/H"]', '1300');
        await pg.press('input[data-chemin="geometrie/H"]', 'Enter');
        await pretO(pg);
        n = await notesO(pg);
        verif(`${nom} : une saisie modifie la note (H = 1 300 mm) et avance sa date`, Object.keys(n).length === 1 && n[id].modifie > avant
          && JSON.stringify(n[id].etat).includes('1300'), JSON.stringify({ avant, apres: n[id].modifie }));
        // rouvrir sans modifier : la note « de trois jours », d'indice C ; sa date ne bouge pas
        await pg.evaluate(([i, t]) => { const ns = JSON.parse(localStorage.getItem('stab.notes.v1')); ns[i].modifie = t; ns[i].indice = 'C'; localStorage.setItem('stab.notes.v1', JSON.stringify(ns)); }, [id, t0]);
        await pg.reload();
        verif(`${nom} : rouverte, le moteur démarre`, !(await pretO(pg)));
        await pg.waitForTimeout(1500);
        n = await notesO(pg); e = await enteteO(pg);
        verif(`${nom} : ROUVRIR la note sans la modifier ne change pas sa date (il y a trois jours) ; la même note, la même hauteur ; l'en-tête dit son indice réel « Indice C »`,
          Object.keys(n).length === 1 && n[id].modifie === t0 && (await pg.inputValue('input[data-chemin="geometrie/H"]')).replace(/\s/g, '') === '1300'
          && (tel ? e.filM && e.filM.startsWith('Indice C · ') : e.etiquette === 'Indice C'), JSON.stringify({ modifie: n[id].modifie, t0, ...e }));
        // ouvrir un autre élément n'est pas une modification ; une saisie, si
        await pg.fill('input[data-chemin="geometrie/H"]', '1250');
        await pg.press('input[data-chemin="geometrie/H"]', 'Enter');
        await pretO(pg);
        n = await notesO(pg);
        verif(`${nom} : la première modification de la note rouverte la date (aujourd'hui)`, n[id].modifie > Date.now() - 600000, JSON.stringify({ modifie: n[id].modifie }));
        await pg.screenshot({ path: path.join(OUT, `site_ouverture_indice_${tel ? 'tel' : fmt === 'PC' ? 'pc' : 'tablette'}.png`) });
        verif(`${nom} : aucune erreur JavaScript`, !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
      } catch (x) { verif(`${nom} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      await pg.context().close();
    }

    // ------------------------------------------------ 6. un lien vers une note absente ; une note d'un autre module
    for (const [fmt, w, h, tel, tactile] of FMT_O) {
      const nom = `ouverture · note absente · ${fmt}`;
      const pg = await ctxO(w, h, tel, tactile ?? tel);
      try {
        await pg.goto(`${BASE}/${MODS.plancher.page}?note=nulle-part`);
        verif(`${nom} : le moteur démarre`, !(await pretO(pg)));
        await pg.waitForTimeout(1500);
        const a = await pg.evaluate(() => {
          const al = document.getElementById('et-alerte'), imp = al.querySelector('.et-alerte-importer'), vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const r = al.getBoundingClientRect(), ri = vu(imp) ? imp.getBoundingClientRect() : null;
          const x = ri && document.elementFromPoint(ri.left + ri.width / 2, ri.top + ri.height / 2);
          return { vu: vu(al), texte: al.querySelector('.et-alerte-txt').textContent, dedans: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight,
            importer: ri ? { l: Math.round(ri.width), h: Math.round(ri.height), atteint: !!x && imp.contains(x), texte: imp.textContent.trim() } : null,
            exporter: vu(al.querySelector('.et-alerte-stab')), adresse: location.search };
        });
        const n = await notesO(pg), e = await enteteO(pg);
        verif(`${nom} : le message dit que la note n'est pas dans ce navigateur, dans l'écran, avec « Importer .stab » atteint (${tactile ?? tel ? '44' : '24'} px) ; RIEN n'est créé`,
          a.vu && /n'est pas dans ce navigateur/.test(a.texte) && a.dedans && a.importer && a.importer.atteint && a.importer.texte === 'Importer .stab'
          && a.importer.h >= (tactile ?? tel ? 44 : 24) && !a.exporter && n && !Object.keys(n).length && e.enr === 'neuf', JSON.stringify({ ...a, notes: n && Object.keys(n).length, enr: e.enr }));
        await pg.screenshot({ path: path.join(OUT, `site_ouverture_absente_${tel ? 'tel' : fmt === 'PC' ? 'pc' : 'tablette'}.png`) });
        // « Importer .stab » ouvre le choix du fichier ; le fichier choisi s'ouvre par la page de SON module
        const fichier = path.join(OUT, 'ouverture_importee.stab');
        fs.writeFileSync(fichier, JSON.stringify({ format: 'stab', version: 1, note: { module: 'garde_corps', nom: 'Note importée', chantier: 'Essai ouverture', indice: 'B', etat: null } }));
        const [choix] = await Promise.all([pg.waitForEvent('filechooser', { timeout: 10000 }), pg.click('#et-alerte .et-alerte-importer')]);
        const m0 = await montage(pg);
        await Promise.all([pg.waitForNavigation(), choix.setFiles(fichier)]);
        verif(`${nom} : « Importer .stab » ouvre le fichier choisi, par la page de son module`, !(await pretO(pg, { apres: m0 })) && new URL(pg.url()).pathname.endsWith(`/${MODS.garde_corps.page}`)
          && (await pg.inputValue('#nom-note')) === 'Note importée', pg.url());
        // une note d'un AUTRE module, demandée à la page du plancher : elle s'ouvre par la page de son module
        const idG = new URLSearchParams(new URL(pg.url()).search).get('note');
        await pg.goto(`${BASE}/${MODS.plancher.page}?note=${idG}`);
        await pretO(pg);
        verif(`${nom} : une note d'un autre module (garde-corps), demandée à la page du plancher, s'ouvre par la page de son module`,
          new URL(pg.url()).pathname.endsWith(`/${MODS.garde_corps.page}`) && new URL(pg.url()).searchParams.get('note') === idG
          && Object.keys(await notesO(pg)).length === 1, pg.url());
        verif(`${nom} : aucune erreur JavaScript`, !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
      } catch (x) { verif(`${nom} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      await pg.context().close();
    }
  }

  // ---------------------------------------------------------------- l'établi compact (pilote « Poutres béton armé »)
  // Trois colonnes : les DONNÉES (colonne compacte, 09/11/2026 : la LIGNE de la poutre ouverte dans l'en-tête du panneau —
  // son nom qu'un clic modifie, la liste des poutres, copier, supprimer — puis ses rubriques en accordéon), le DESSIN, les
  // RÉSULTATS. La carte de l'accueil ouvre le calcul lui-même ; RIEN n'est écrit tant qu'on ne modifie rien ; une seule
  // rubrique ouverte à la fois et chaque champ s'atteint (R1 : rien d'inatteignable) ; un clic sur une pièce du dessin ouvre SA ligne dans les
  // données (un glisser, non), et une ligne choisie éclaire sa pièce ; la sélection survit au calcul ; « Annuler » et
  // « Rétablir » ; chaque vérification (deux lignes) ouvre SA fenêtre de détail ; une note sans effort dit « Données
  // incomplètes », jamais « Conforme » ; la note PDF se fait dans le navigateur ; le .stab garde son nom (pas de .json) ;
  // un enregistrement Streamlit s'ouvre ; « Enregistrer comme note… » nomme, range et écrit ; les colonnes se replient et
  // se redimensionnent.
  //   filtre « pilote »
  if (!FILTRE || FILTRE === 'pilote') {
    const fs = require('fs');
    const c = await b.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
    const g = await c.newPage();
    g.erreurs = [];
    g.on('pageerror', (e) => g.erreurs.push(e.message));
    g.on('console', (m) => { if (m.type() === 'error') g.erreurs.push(m.text().slice(0, 300)); });
    const nNotes = () => g.evaluate(() => { try { return Object.keys(JSON.parse(localStorage.getItem('stab.notes.v1') || '{}')).length; } catch (e) { return -1; } });
    const tranquille = () => g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
    const lireEtat = () => g.evaluate(() => {
      const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      return { url: location.search,
        ouvertes: [...document.querySelectorAll('#donnees .et-arbre .sec[data-sec].open')].map((x) => x.dataset.sec),
        cibles: [...document.querySelectorAll('#donnees .et-cible[data-cible]')].filter(vu).map((x) => x.dataset.cible),
        surlignees: [...new Set([...document.querySelectorAll('#stage .view.on [data-sel].et-sel')].map((x) => x.dataset.sel))],
        enr: (document.querySelector('.top .et-enr') || {}).dataset?.enr, statut: document.getElementById('statut').textContent.trim() };
    });
    // ouvrir une rubrique si elle est fermée (un clic sur l'en-tête d'une rubrique ouverte la refermerait)
    const ouvrirRub = async (k) => {
      if (!(await g.$eval(`#donnees .sec[data-sec="${k}"]`, (x) => x.classList.contains('open')))) await g.click(`#donnees .sec[data-sec="${k}"] > .sechd`);
      await g.waitForTimeout(80);
    };
    // la LIGNE de la poutre ouverte (en-tête du panneau de gauche) : son nom, le nombre de poutres, le compte sur le bouton
    // de la liste, supprimer allumé ou éteint, le titre des résultats ; la liste (les noms, l'ouverte cochée d'une étoile)
    const LIGNE = '.panel.left .phead .et-elt .et-elsel', NOM = `${LIGNE} input.et-elt-nom`;
    const etatLigne = () => g.evaluate((s) => { const l = document.querySelector(s);
      return { nom: l.querySelector('input.et-elt-nom').value, n: l.dataset.elements, compte: (l.querySelector('.et-elt-liste .et-mn-b .n') || {}).textContent || '',
        supprimer: !l.querySelector('[data-supprimer]').disabled, rp: document.querySelector('#resultat .et-rp b').textContent }; }, LIGNE);
    const liste = () => g.$$eval(`${LIGNE} .et-elt-liste .et-mn-l [data-activer]`,
      (x) => x.map((b) => b.querySelector('.nom').textContent + (b.classList.contains('on') && b.getAttribute('aria-checked') === 'true' ? '*' : '')).join());
    try {
      await ouvrir(g, `${BASE}/index.html`);
      let m0 = await montage(g);
      await Promise.all([g.waitForNavigation(), g.click('.tile[data-module="poutre"]')]);
      await pretEcran(g, { apres: m0 });
      let e = await lireEtat();
      verif('pilote : la carte de l\'accueil ouvre le calcul lui-même (sans fenêtre « Nouvelle note »), une poutre',
        /poutre(\.html)?$/.test(new URL(g.url()).pathname) && !e.url && (await g.$eval(LIGNE, (x) => x.dataset.elements)) === '1', `${g.url()} · ${e.url}`);
      verif('pilote : l\'ouvrir pour regarder n\'écrit rien (aucune note), l\'indicateur dit « Pas encore enregistré »',
        (await nNotes()) === 0 && e.enr === 'neuf', JSON.stringify({ notes: await nNotes(), enr: e.enr }));
      // (deuxième passe) l'état d'une section est l'en-tête de son bloc ; la pastille de la NOTE ne s'affiche que dès deux
      // poutres (seule, chaque section dit le sien)
      const etatBloc = await g.$eval('#resultat .et-bloc .et-bloc-h .e', (x) => x.textContent);
      verif('pilote : une note sans effort dit « Données incomplètes », jamais « Conforme »',
        etatBloc === 'Données incomplètes' && !e.statut.includes('Conforme') && (await g.$eval('#resultat .et-bloc', (x) => x.classList.contains('inc'))), `${etatBloc} · ${e.statut}`);
      verif('pilote : une seule rubrique ouverte par défaut, la « Poutre »', e.ouvertes.join() === 'pt', JSON.stringify(e.ouvertes));
      // (colonne de gauche compacte, 09/11/2026) plus d'arborescence : la LIGNE de la poutre ouverte est dans l'EN-TÊTE du
      // panneau, à côté d'« Annuler » et « Rétablir » — son NOM en champ (le seul champ « nom » de la page : ni dans les
      // données, ni dans les résultats, qui le LISENT), la liste des poutres (« V1 · 20 × 40 cm », cochée ; « Ajouter une
      // poutre »), copier, supprimer (éteint : elle est seule), sans taux ni pastille
      const ligne = await g.evaluate(() => {
        const ph = document.querySelector('.panel.left .phead'), l = ph.querySelector(':scope > .et-elt > .et-elsel'), n = l && l.querySelector('input.et-elt-nom');
        const on = l && l.querySelector('.et-elt-liste .et-mn-l .et-elt-i.on');
        return { ordre: [...ph.children].map((x) => (x.classList.contains('et-elt') ? 'elt' : x.dataset.historique || (x.dataset.coll ? 'replier' : x.id || x.className))).join(),
          nom: n ? n.value : null, champ: !!n && n.dataset.chemin === 'nom' && n.dataset.sorte === 'texte', nomsPage: document.querySelectorAll('[data-chemin="nom"]').length,
          liste: on ? `${on.querySelector('.nom').textContent} ${on.querySelector('.dim').textContent}`.replace(/\s+/g, ' ').trim() : null, coche: !!on && on.getAttribute('aria-checked') === 'true',
          ajouter: ((l && l.querySelector('.et-elt-liste .et-mn-l [data-ajouter]')) || {}).textContent?.trim(), compte: !!l && !!l.querySelector('.et-elt-liste .et-mn-b .n'),
          copier: !!l && !!l.querySelector('button.et-elt-b[data-dupliquer-index="0"]'),
          supprimerEteint: !!l && !!l.querySelector('button.et-elt-b.danger[data-supprimer="0"][data-confirmer]:disabled'),
          pastille: !!l && !!l.querySelector('.chip, .pct, .status') };
      });
      verif('pilote : la LIGNE de la poutre ouverte est dans l\'en-tête du panneau (avant Annuler, Rétablir) — le nom « V1 » en champ (le seul de la page), la liste « V1 · 20 × 40 cm » cochée et « Ajouter une poutre », copier, supprimer éteint (seule), sans taux ni pastille',
        ligne.ordre === 'elt,calcule,annuler,retablir,replier' && ligne.nom === 'V1' && ligne.champ && ligne.nomsPage === 1 && ligne.liste === 'V1 · 20 × 40 cm' && ligne.coche
        && ligne.ajouter === 'Ajouter une poutre' && !ligne.compte && ligne.copier && ligne.supprimerEteint && !ligne.pastille, JSON.stringify(ligne));
      // le contrat du DOM : les données commencent par les rubriques de la poutre ; plus rien de l'arborescence, du sélecteur
      // en segments, du nom en champ dans les résultats (leur tête le lit : « V1 », puis les dimensions)
      const contrat = await g.evaluate(() => {
        const don = document.getElementById('donnees'), a = don.firstElementChild, rp = document.querySelector('#resultat .et-rp');
        return { premier: !!a && a.matches('.et-arbre.et-arbre-c[data-poutre="0"]') && !!a.firstElementChild && a.firstElementChild.matches('.sec[data-sec]'),
          anciens: ['.et-pt', '.et-pt-l', '.et-pt-nom', '.et-pt-add', '.et-arbre-h', '.et-elsel .seg', '.et-elsel.seul', '.et-arbre-c.et-seul', '.et-rp-nom', '.et-rp-c',
            '#donnees .et-elsel', '[data-renommer]'].filter((s) => document.querySelector(s)),
          rp: rp ? [...rp.children].map((x) => `${x.tagName}:${x.textContent.trim()}`).join(' | ') : null };
      });
      verif('pilote : les données commencent par les rubriques (ni arborescence, ni sélecteur en segments, ni « Renommer ») ; la tête des résultats lit le nom (plus de champ)',
        contrat.premier && !contrat.anciens.length && /^B:V1 \| SPAN:20 × 40 cm · /.test(contrat.rp || ''), JSON.stringify(contrat));
      const lg0 = await ligneElement(g);
      verif('pilote · PC 1440 : la ligne de la poutre tient dans l\'en-tête (une rangée, rien ne déborde ni ne se chevauche, le nom entier)', lg0 && !lg0.fautes.length && lg0.nomEntier, JSON.stringify(lg0));
      // l'accordéon : un en-tête ouvre SA rubrique sans fermer les autres (plusieurs, si on le veut), et la referme ;
      // chaque champ s'atteint (une pièce choisie montre son détail) — le nom aussi, dans la ligne de la poutre
      const cles = await g.$$eval('#donnees .et-arbre .sec[data-sec]', (x) => x.map((y) => y.dataset.sec));
      const CHAMPS = '#donnees [data-chemin], .panel.left .phead [data-chemin]';
      const champs = await g.$$eval(CHAMPS, (x) => [...new Set(x.map((y) => y.dataset.chemin))]);
      const vus = new Set();
      const accordeon = [];
      const noterVus = async () => (await g.$$eval(CHAMPS, (x) => x.filter((y) => y.checkVisibility()).map((y) => y.dataset.chemin))).forEach((x) => vus.add(x));
      const deja = [];
      for (const k of cles) {
        if (!(await g.$eval(`#donnees .sec[data-sec="${k}"]`, (x) => x.classList.contains('open')))) await g.click(`#donnees .sec[data-sec="${k}"] > .sechd`);
        await g.waitForTimeout(100);
        deja.push(k);
        e = await lireEtat();
        if (e.ouvertes.join() !== deja.join()) accordeon.push(`${k}: ${e.ouvertes.join()}`);
        await noterVus();
        const pieces = await g.$$eval(`#donnees .sec[data-sec="${k}"] .et-piece [data-deplier].et-dep`, (x) => x.length);
        for (let i = 0; i < pieces; i++) {
          await g.click(`#donnees .sec[data-sec="${k}"] .et-piece:has(.et-dep) >> nth=${i} >> .et-dep`);
          await g.waitForTimeout(60);
          await noterVus();
        }
        // les actions rangées dans un menu ⋯ (copier, supprimer une section) s'atteignent en l'ouvrant
        const menus = await g.$$eval(`#donnees .sec[data-sec="${k}"] [data-menu]`, (x) => x.length);
        for (let i = 0; i < menus; i++) {
          await g.click(`#donnees .sec[data-sec="${k}"] [data-menu] >> nth=${i}`);
          await g.waitForTimeout(60);
          await noterVus();
          await g.keyboard.press('Escape');
        }
      }
      verif(`pilote : accordéon — chaque en-tête (${cles.join(', ')}) ouvre SA rubrique et garde les autres ouvertes`, cles.length === 4 && !accordeon.length, accordeon.join(' | '));
      const caches = champs.filter((x) => !vus.has(x));
      verif(`pilote : chacun des ${champs.length} champs des données s'atteint (rubrique ouverte, pièce dépliée ; le nom dans la ligne de la poutre)`, !caches.length && champs.includes('nom'), caches.join(', '));
      await g.screenshot({ path: path.join(OUT, 'site_pilote_rubriques.png') });
      for (const k of cles) await g.click(`#donnees .sec[data-sec="${k}"] > .sechd`);
      await g.waitForTimeout(100);
      verif('pilote : un second clic sur un en-tête referme sa rubrique (les quatre fermées)', !(await lireEtat()).ouvertes.length, JSON.stringify((await lireEtat()).ouvertes));
      // le dessin : un clic sur le béton ouvre « Poutre » ; un GLISSER ne choisit rien ; un clic sur un lit l'ouvre dans « Armatures »
      await g.keyboard.press('Escape');
      const centre = async (sel) => g.$eval(`#stage .view.on ${sel}`, (x) => { const q = x.getBoundingClientRect(); return { x: q.left + q.width / 2, y: q.top + q.height / 2 }; });
      const pt0 = await centre('[data-sel="s0.beton"] rect');
      await g.mouse.click(pt0.x, pt0.y);
      await g.waitForTimeout(150);
      await g.keyboard.press('Escape');
      const pt = await centre('[data-sel="s0.beton"] rect');
      await g.mouse.move(pt.x, pt.y);
      await g.mouse.down();
      await g.mouse.move(pt.x + 60, pt.y + 30, { steps: 6 });
      await g.mouse.up();
      e = await lireEtat();
      verif('pilote : glisser sur le dessin (déplacer la vue) ne choisit rien', !e.surlignees.length && e.ouvertes.join() === 'pt', JSON.stringify(e));
      await g.click('#zoom [data-zoom="ajuster"]');
      await g.waitForTimeout(150);
      const an = await centre('[data-sel="s0.lit-inf-0"] text.an');
      await g.mouse.click(an.x, an.y);
      await g.waitForTimeout(250);
      e = await lireEtat();
      verif('pilote : un clic sur l\'étiquette « L1 » du dessin ouvre « Armatures », éclaire la ligne du lit et ses barres',
        e.ouvertes.join() === 'arm' && e.cibles.join() === 's0.lit-inf-0' && e.surlignees.join() === 's0.lit-inf-0', JSON.stringify(e));
      await g.screenshot({ path: path.join(OUT, 'site_pilote_clic_dessin.png') });
      // une rubrique ouverte À LA MAIN reste ouverte quand le dessin amène ailleurs ; celle que le dessin avait ouverte se replie
      await g.click('#donnees .sec[data-sec="so"] > .sechd');
      const pb = await centre('[data-sel="s0.beton"] rect');
      await g.mouse.click(pb.x, pb.y);
      await g.waitForTimeout(200);
      e = await lireEtat();
      verif('pilote : le béton cliqué ouvre « Poutre », replie « Armatures » (ouverte par le dessin), garde « Sections » (ouverte à la main)',
        e.ouvertes.join() === 'pt,so' && e.surlignees.join() === 's0.beton', JSON.stringify(e));
      await g.click('#donnees .sec[data-sec="so"] > .sechd');
      const an2 = await centre('[data-sel="s0.lit-inf-0"] text.an');
      await g.mouse.click(an2.x, an2.y);
      await g.waitForTimeout(200);
      // une ligne des données éclaire sa pièce sur le dessin (l'étiquette C1 de la ligne la choisit et la déplie)
      await g.click('#donnees [data-cible="s0.cadre-0"] .et-tag');
      await g.waitForTimeout(100);
      e = await lireEtat();
      verif('pilote : choisir le cadre C1 dans les données l\'éclaire sur le dessin, et montre ses poignées (cachées sinon)', e.surlignees.join() === 's0.cadre-0' && e.cibles.join() === 's0.cadre-0'
        && await g.evaluate(() => [...document.querySelectorAll('#stage .view.on .pc.cadre .poignee')].every((x) => x.checkVisibility({ visibilityProperty: true }) === !!x.closest('.et-sel'))
          && !!document.querySelector('#stage .view.on .pc.cadre.et-sel .poignee')), JSON.stringify(e));
      // la première saisie écrit la note ; la sélection survit au calcul
      await g.fill('#donnees input[data-chemin="sections/0/inf/0/n"]', '3');
      await g.press('#donnees input[data-chemin="sections/0/inf/0/n"]', 'Enter');
      await tranquille();
      await g.waitForTimeout(200);
      e = await lireEtat();
      verif('pilote : la PREMIÈRE saisie écrit la note (une seule), l\'adresse la nomme, l\'indicateur dit « Enregistré »',
        (await nNotes()) === 1 && /^\?note=/.test(e.url) && e.enr === 'ok', JSON.stringify({ notes: await nNotes(), url: e.url, enr: e.enr }));
      verif('pilote : la rubrique ouverte et la sélection survivent au calcul', e.ouvertes.join() === 'arm' && e.surlignees.length === 1, JSON.stringify(e));
      // annuler, rétablir (boutons et clavier)
      const nLit = () => g.inputValue('#donnees input[data-chemin="sections/0/inf/0/n"]');
      await g.click('[data-historique="annuler"]');
      await tranquille();
      await g.waitForTimeout(150);
      const apresAnnuler = await nLit();
      await g.click('[data-historique="retablir"]');
      await tranquille();
      await g.waitForTimeout(150);
      const apresRetablir = await nLit();
      await g.click('#stage');
      await g.keyboard.press('Control+z');
      await tranquille();
      await g.waitForTimeout(150);
      const apresCtrlZ = await nLit();
      verif('pilote : « Annuler » rend 2 barres, « Rétablir » 3, Ctrl+Z annule encore', apresAnnuler === '2' && apresRetablir === '3' && apresCtrlZ === '2',
        JSON.stringify({ apresAnnuler, apresRetablir, apresCtrlZ }));
      // des efforts : un taux ; chaque vérification sur deux lignes ouvre SA fenêtre de détail
      await g.click('#donnees .sec[data-sec="so"] > .sechd');
      for (const [ch, v] of [['sections/0/M_inf', '60'], ['sections/0/V', '40']]) {
        await g.fill(`#donnees input[data-chemin="${ch}"]`, v);
        await g.press(`#donnees input[data-chemin="${ch}"]`, 'Enter');
        await tranquille();
      }
      await g.waitForTimeout(200);
      // DEUX lignes : la 1ʳᵉ (nom, pièce, taux) et la 2ᵉ (la comparaison) ne passent jamais à la ligne
      const lignes = await g.$$eval('#resultat .el.vc[data-calcul]', (x) => x.map((y) => {
        const h = (s) => Math.round(y.querySelector(s).getBoundingClientRect().height);
        return { k: y.dataset.calcul, l1: h('.l1'), l2: h('.l2'), n: y.children.length };
      }));
      verif('pilote : avec des efforts, l\'état n\'est plus « Données incomplètes » ; cinq vérifications à taux', !(await g.textContent('#resultat .et-bloc .et-bloc-h')).includes('incomplètes') && lignes.length === 5, JSON.stringify(lignes));
      verif('pilote : chaque vérification tient sur deux lignes (aucune ne passe à la ligne)', lignes.every((l) => l.n === 2 && l.l1 <= 26 && l.l2 <= 20), JSON.stringify(lignes));
      const fen = [];
      for (const l of lignes) {
        await g.click(`#resultat .el.vc[data-calcul="${l.k}"]`);
        await g.waitForTimeout(80);
        if (!(await g.isVisible(`#ck-${l.k}`))) fen.push(l.k);
        await g.keyboard.press('Escape');
      }
      verif('pilote : chaque ligne du résultat ouvre SA fenêtre de détail (formules, critère, référence)', !fen.length && (await g.$$eval('#volets .ck-crit', (x) => x.length)) >= lignes.length, fen.join(', '));
      await g.click('#resultat .el.vc[data-calcul="s0-inf"]');
      await g.screenshot({ path: path.join(OUT, 'site_pilote_detail.png') });
      await g.keyboard.press('Escape');
      // ---- (deuxième passe) l'en-tête technique du dessin ; le zoom discret
      const ent = await g.evaluate(() => { const x = document.querySelector('#stage .view.on .et-ent'); return x ? { t: x.textContent.replace(/\s+/g, ' ').trim(), eff: x.querySelectorAll('.et-eff > span').length } : null; });
      verif('pilote : l\'en-tête du dessin nomme la section, ses matériaux, ses aciers et ses étriers, et la bande des trois efforts',
        ent && /^SECTION A\s*· 20 × 40 cm · C30\/37 · B500\s*Inf\. 2Ø16 · Sup\. 2Ø16\s*1 cadre Ø10 · e = 30 cm/.test(ent.t) && ent.eff === 3, JSON.stringify(ent));
      await g.mouse.move(5, 450);
      await g.waitForTimeout(300);
      const zLoin = await g.$eval('#zoom', (x) => +getComputedStyle(x).opacity);
      const zc = await g.$eval('.canvas .stage', (x) => { const q = x.getBoundingClientRect(); return { x: q.left + q.width / 2, y: q.top + 40 }; });
      await g.mouse.move(zc.x, zc.y);
      await g.waitForTimeout(300);
      const zPres = await g.$eval('#zoom', (x) => +getComputedStyle(x).opacity);
      verif('pilote : le zoom est discret — invisible loin de la toile, visible au survol ; « ajuster » est une icône (aucun « 100 % » affiché)',
        zLoin === 0 && zPres === 1 && !(await g.$eval('#zoom .zpct', (x) => x.innerText.trim())), JSON.stringify({ zLoin, zPres }));
      // ---- une deuxième section : celle qu'on choisit à gauche est celle du dessin et des résultats
      await g.click('#donnees .sec[data-sec="so"] .et-pied .et-add');
      await tranquille();
      await g.waitForTimeout(200);
      const sync = () => g.evaluate(() => ({ vue: document.querySelector('#vues .on')?.dataset.vue,
        blocs: [...document.querySelectorAll('#resultat .et-bloc')].filter((b) => !b.hidden).map((b) => b.dataset.section),
        arm: document.querySelector('#donnees .et-arm.on')?.dataset.section }));
      let sy = await sync();
      verif('pilote : deux sections — le dessin et les résultats montrent la section courante seule', sy.vue === `s${sy.arm}` && sy.blocs.join() === sy.arm, JSON.stringify(sy));
      await ouvrirRub('arm');
      await g.click('#donnees .et-secsel [data-section-courante="1"]');
      await g.waitForTimeout(200);
      sy = await sync();
      verif('pilote : la section B choisie à gauche : le dessin montre B, les résultats le bloc de B seul', sy.vue === 's1' && sy.blocs.join() === '1' && sy.arm === '1', JSON.stringify(sy));
      await g.click('#vues [data-vue="toutes"]');
      await g.waitForTimeout(200);
      sy = await sync();
      verif('pilote : la vue « Toutes » montre les blocs de toutes les sections', sy.vue === 'toutes' && sy.blocs.join() === '0,1', JSON.stringify(sy));
      await g.screenshot({ path: path.join(OUT, 'site_pilote_toutes.png') });
      await g.click('#donnees .et-secsel [data-section-courante="0"]');
      await g.waitForTimeout(200);
      // ---- les barres qu'un cadre entoure : au dessin (sa poignée s'aimante aux barres), dans les données (un clic par bout)
      await g.fill('#donnees input[data-chemin="sections/0/inf/0/n"]', '4');
      await g.press('#donnees input[data-chemin="sections/0/inf/0/n"]', 'Enter');
      await tranquille();
      await g.click('#donnees .et-arm.on .et-pied .et-add[data-valeur="etrier"]');
      await tranquille();
      await g.waitForTimeout(200);
      const etriers = () => g.evaluate(() => { const id = new URLSearchParams(location.search).get('note'); return JSON.parse(localStorage.getItem('stab.notes.v1'))[id].etat.poutres[0].sections[0].etriers; });
      verif('pilote : « + Cadre » pose un 2ᵉ cadre à l\'intérieur du premier (barres 2 à 3 sur 4)', JSON.stringify((await etriers())[1]) === '{"type":"etrier","d":10,"de":2,"a":3}', JSON.stringify(await etriers()));
      const lab = await g.evaluate(() => { const t = [...document.querySelectorAll('#stage .view.on .pc.cadre text.an')].find((x) => x.textContent === 'C2'); const q = t.getBoundingClientRect(); return { x: q.left + q.width / 2, y: q.top + q.height / 2 }; });
      await g.mouse.click(lab.x, lab.y);
      await g.waitForTimeout(200);
      const hA = await g.$eval('#stage .view.on .pc.cadre.et-sel .poignee[data-bout="a"] .o', (x) => { const q = x.getBoundingClientRect(); return { x: q.left + q.width / 2, y: q.top + q.height / 2 }; });
      const b4 = await g.evaluate(() => { const c = [...document.querySelectorAll('#stage .view.on .pc.lit[data-sel="s0.lit-inf-0"] > circle')].pop(); const q = c.getBoundingClientRect(); return { x: q.left + q.width / 2 }; });
      await g.mouse.move(hA.x, hA.y);
      await g.mouse.down();
      await g.mouse.move(hA.x + (b4.x - hA.x) / 2, hA.y, { steps: 4 });
      await g.mouse.move(b4.x, hA.y, { steps: 4 });
      await g.mouse.up();
      await tranquille();
      await g.waitForTimeout(200);
      verif('pilote : la poignée droite de C2, glissée jusqu\'à la barre 4, s\'y aimante (barres 2 à 4)', JSON.stringify((await etriers())[1]) === '{"type":"etrier","d":10,"de":2,"a":4}', JSON.stringify(await etriers()));
      await g.screenshot({ path: path.join(OUT, 'site_pilote_cadre.png') });
      await g.click('#donnees [data-cible="s0.cadre-1"] .et-barres [data-barre="1"]');
      await tranquille();
      await g.waitForTimeout(150);
      const plage1 = JSON.stringify((await etriers())[1]);
      await g.click('#donnees [data-cible="s0.cadre-1"] .et-barres [data-barre="1"]');
      await tranquille();
      await g.waitForTimeout(150);
      verif('pilote : dans les données, la barre 1 élargit C2 à toute la largeur ; cliquée de nouveau (un bout), elle en sort',
        plage1 === '{"type":"etrier","d":10}' && JSON.stringify((await etriers())[1]) === '{"type":"etrier","d":10,"de":2,"a":4}', `${plage1} → ${JSON.stringify((await etriers())[1])}`);
      // ---- la liste de contrôle : une observation « à examiner » se marque examinée ; une non-conformité, jamais
      await g.click('#resultat .et-bloc:not([hidden]) .et-dispo');
      await g.waitForTimeout(150);
      const avantJ = await g.evaluate(() => ({ warn: document.querySelectorAll('#resultat .et-bloc:not([hidden]) .et-obs.v [data-justifier]').length,
        nok: document.querySelectorAll('#resultat .et-bloc:not([hidden]) .et-obs.r').length,
        nokBouton: document.querySelectorAll('#resultat .et-bloc:not([hidden]) .et-obs.r [data-justifier]').length }));
      await g.click('#resultat .et-bloc:not([hidden]) .et-obs.v [data-justifier]');
      await g.fill('#resultat .et-jf textarea', 'Barres de montage Ø8 prévues au plan.');
      await g.click('#resultat [data-just-valider]');
      await tranquille();
      await g.waitForTimeout(200);
      const apresJ = await g.evaluate(() => { const id = new URLSearchParams(location.search).get('note');
        return { justifs: JSON.parse(localStorage.getItem('stab.notes.v1'))[id].etat.poutres[0].sections[0].justifs,
          j: [...document.querySelectorAll('#resultat .et-bloc:not([hidden]) .et-obs.j')].map((x) => x.textContent.replace(/\s+/g, ' ').trim()) }; });
      verif('pilote : une observation « à examiner » se marque examinée, sa justification gardée dans la note ; une non-conformité n\'a pas ce bouton',
        avantJ.warn >= 1 && avantJ.nok >= 1 && avantJ.nokBouton === 0 && apresJ.justifs && Object.values(apresJ.justifs)[0].texte === 'Barres de montage Ø8 prévues au plan.'
        && apresJ.j.length === 1 && apresJ.j[0].includes('Examinée — justifiée'), JSON.stringify({ avantJ, apresJ }));
      await g.screenshot({ path: path.join(OUT, 'site_pilote_liste.png') });
      // ---- le NOM se modifie dans la ligne de la poutre, en tête du panneau : UN clic y pose le curseur ; Entrée l'écrit
      // dans la note (la liste des poutres et la tête des résultats suivent) ; Échap rend l'ancien nom ; un nom vide est
      // refusé (un message, l'ancien nom revient, rien n'est écrit) ; une saisie directe aussi
      const nomNote = () => g.evaluate((s) => { const id = new URLSearchParams(location.search).get('note');
        const l = document.querySelector(s), on = l.querySelector('.et-elt-liste .et-mn-l .et-elt-i.on');
        return { note: JSON.parse(localStorage.getItem('stab.notes.v1'))[id].etat.poutres[0].nom, champ: l.querySelector('input.et-elt-nom').value,
          liste: `${on.querySelector('.nom').textContent} ${on.querySelector('.dim').textContent}`.replace(/\s+/g, ' ').trim(),
          rp: document.querySelector('#resultat .et-rp b').textContent }; }, LIGNE);
      const aFocus = () => g.evaluate((s) => !!document.activeElement && document.activeElement.matches(s), NOM);
      const lireToast = () => g.evaluate(() => { const t = document.getElementById('toast'); return { vu: t.checkVisibility(), texte: t.textContent, erreur: t.classList.contains('erreur') }; });
      await g.click(NOM);
      const rn0 = { focus: await aFocus() };
      await g.keyboard.press('Control+a');
      await g.keyboard.type('V1 bis');
      await g.keyboard.press('Enter');
      await tranquille();
      await g.waitForTimeout(200);
      const rn1 = { ...(await nomNote()), focus: await aFocus() };
      verif('pilote : un clic sur le nom (ligne de la poutre, en-tête du panneau) y pose le curseur ; « V1 bis » + Entrée l\'écrit dans la note — la liste des poutres et la tête des résultats suivent',
        rn0.focus && rn1.note === 'V1 bis' && rn1.champ === 'V1 bis' && rn1.liste === 'V1 bis · 20 × 40 cm' && rn1.rp === 'V1 bis' && !rn1.focus, JSON.stringify({ rn0, rn1 }));
      await g.click(NOM);
      await g.keyboard.press('Control+a');
      await g.keyboard.type('Brouillon');
      await g.keyboard.press('Escape');
      await g.waitForTimeout(250);
      const rnE = { ...(await nomNote()), focus: await aFocus() };
      await g.click(NOM);
      await g.keyboard.press('Control+a');
      await g.keyboard.press('Backspace');
      await g.keyboard.press('Enter');
      await g.waitForTimeout(300);
      const rnV = { ...(await nomNote()), toast: await lireToast() };
      verif('pilote : Échap rend l\'ancien nom (« V1 bis », rien n\'est écrit) ; un nom vide est refusé — « Un nom ne peut pas être vide », l\'ancien nom revient, la note le garde',
        rnE.champ === 'V1 bis' && rnE.note === 'V1 bis' && !rnE.focus && rnV.champ === 'V1 bis' && rnV.note === 'V1 bis' && rnV.rp === 'V1 bis'
        && rnV.toast.vu && rnV.toast.erreur && /Un nom ne peut pas être vide/.test(rnV.toast.texte), JSON.stringify({ rnE, rnV }));
      await g.fill(NOM, 'V1');
      await g.press(NOM, 'Enter');
      await tranquille();
      await g.waitForTimeout(200);
      const rn2 = await nomNote();
      verif('pilote : une saisie directe dans le nom (« V1 ») l\'écrit aussi', rn2.note === 'V1' && rn2.champ === 'V1' && rn2.liste === 'V1 · 20 × 40 cm' && rn2.rp === 'V1', JSON.stringify(rn2));
      // ---- copier, la liste, supprimer : les gestes de la ligne. « Copier » (une icône) insère la copie après la poutre et
      // l'ouvre ; la liste (le chevron, « 2 » dessus) montre les deux, l'ouverte cochée, s'ouvre dans l'écran et ouvre
      // l'autre ; « Supprimer » demande confirmation — refusée, rien ne change ; acceptée, la poutre part — et s'éteint
      // quand une seule reste
      await g.click(`${LIGNE} [data-dupliquer-index="0"]`);
      await tranquille();
      await g.waitForTimeout(200);
      const cp = { ...(await etatLigne()), liste: await liste() };
      const lgCp = await ligneElement(g);
      const mn = await listeElement(g);
      await g.screenshot({ path: path.join(OUT, 'site_poutre_ligne_element.png') });
      await g.click(`${LIGNE} .et-elt-liste .et-mn-l [data-activer="0"]`);
      await tranquille();
      await g.waitForTimeout(200);
      const ouv = { ...(await etatLigne()), liste: await liste(), ferme: await g.$eval(`${LIGNE} .et-elt-liste .et-mn-l`, (x) => x.hidden) };
      verif('pilote : « Copier » insère la copie et l\'ouvre — « V1 (copie) » dans la ligne et en tête des résultats, « 2 » sur le bouton de la liste, supprimer allumé ; la ligne tient toujours dans l\'en-tête',
        cp.n === '2' && cp.compte === '2' && cp.nom === 'V1 (copie)' && cp.rp === 'V1 (copie)' && cp.liste === 'V1,V1 (copie)*' && cp.supprimer && lgCp && !lgCp.fautes.length && lgCp.nomEntier,
        JSON.stringify({ cp, lgCp }));
      verif('pilote : la liste des poutres s\'ouvre dans l\'écran (les deux, l\'ouverte cochée, « Ajouter une poutre », chaque entrée atteinte) ; elle ouvre « V1 » et se referme',
        mn.vu && mn.ouvert && mn.dedans && mn.touche && mn.items === 'V1,V1 (copie)*' && mn.ajouter === 'Ajouter une poutre' && mn.compte === '2'
        && ouv.nom === 'V1' && ouv.rp === 'V1' && ouv.liste === 'V1*,V1 (copie)' && ouv.ferme, JSON.stringify({ mn, ouv }));
      await g.click(`${LISTE_ELT} [data-menu]`);
      await g.click(`${LIGNE} .et-elt-liste .et-mn-l [data-activer="1"]`);
      await tranquille();
      await g.waitForTimeout(200);
      const msgs = [];
      g.once('dialog', (d) => { msgs.push(d.message()); d.dismiss(); });
      await g.click(`${LIGNE} [data-supprimer="1"]`);
      await g.waitForTimeout(400);
      const refuse = { ...(await etatLigne()), liste: await liste() };
      g.once('dialog', (d) => { msgs.push(d.message()); d.accept(); });
      await g.click(`${LIGNE} [data-supprimer="1"]`);
      await tranquille();
      await g.waitForTimeout(200);
      const sup = { ...(await etatLigne()), liste: await liste() };
      verif('pilote : « Supprimer » demande confirmation (« Supprimer la poutre « V1 (copie) » de la note ? ») — refusée, rien ne change ; acceptée, la poutre part : « V1 » reste, seule, supprimer s\'éteint, le compte disparaît',
        msgs.length === 2 && msgs.every((x) => x === 'Supprimer la poutre « V1 (copie) » de la note ?') && refuse.n === '2' && refuse.nom === 'V1 (copie)' && refuse.liste === 'V1,V1 (copie)*'
        && sup.n === '1' && sup.nom === 'V1' && sup.rp === 'V1' && sup.liste === 'V1*' && !sup.supprimer && sup.compte === '', JSON.stringify({ msgs, refuse, sup }));
      // ---- la BIBLIOTHÈQUE de la note (docs/site/BIBLIOTHEQUES.md § 2.4, § 2.5) : le bouton du ruban ouvre un grand panneau,
      // prérempli depuis la poutre ; « Utiliser cette valeur » écrit c (une saisie ordinaire) ; « Inclure dans la note »
      // ajoute une fiche (bloc « Justifications », vue J1) ; Échap referme et rend le focus ; la note PDF et le .stab
      // emportent la fiche ; le menu ⋯ de la fiche la retire
      const bibPret = () => g.waitForFunction(() => { const p = document.getElementById('et-bibliotheque');
        return p && !p.hidden && p.querySelector('#bib-corps').getAttribute('aria-busy') === 'false' && p.querySelector('#bib-tableau .cur') && p.querySelector('.bib-val'); }, null, { timeout: 60000 });
      await g.click('.top [data-action="bibliotheque"]');
      await bibPret();
      await g.waitForTimeout(250);
      const bb = await g.evaluate(() => { const p = document.getElementById('et-bibliotheque'), c = p.querySelector('.et-bib-c').getBoundingClientRect(), u = p.querySelector('[data-utiliser]');
        return { l: Math.round(c.width), t: Math.round(c.height), dedans: c.left >= 0 && c.right <= innerWidth && c.top >= 0 && c.bottom <= innerHeight,
          rubrique: p.querySelector('.bib-m.on').dataset.rubrique, classe: p.querySelector('[data-param="classe"]').value, phi: p.querySelector('[data-param="phi"]').value,
          element: (p.querySelector('.seg[data-param="element"] button.on') || {}).dataset?.valeur, utiliser: u.textContent.trim(), grise: u.disabled, titre: u.title,
          tableau: p.querySelector('#bib-tableau').scrollHeight <= p.querySelector('#bib-tableau').clientHeight + 1 }; });
      verif('pilote : « Bibliothèque » ouvre un grand panneau dans l\'écran, sur « Enrobages », prérempli depuis la poutre (poutre, C30/37, Ø 16), le tableau entier ; « Utiliser cette valeur » nomme c',
        bb.dedans && bb.l >= 1100 && bb.t >= 700 && bb.rubrique === 'enrobages' && bb.classe === 'C30/37' && bb.phi === '16' && bb.element === 'poutre' && bb.tableau
        && !bb.grise && /^Utiliser cette valeur · c = [\d,]+ cm$/.test(bb.utiliser) && /c enrobage/.test(bb.titre), JSON.stringify(bb));
      await g.screenshot({ path: path.join(OUT, 'site_pilote_bibliotheque.png') });
      const cAvant = await g.$eval('#donnees input[data-chemin="enrobage"]', (x) => x.value);
      const cBib = bb.utiliser.replace(/^.*c = /, '').replace(/ cm$/, '');
      await g.click('#et-bibliotheque [data-utiliser]');
      await tranquille();
      await bibPret();
      await g.waitForTimeout(250);
      const cApres = await g.$eval('#donnees input[data-chemin="enrobage"]', (x) => x.value);
      verif('pilote : « Utiliser cette valeur » écrit c_nom (en cm) dans « c enrobage » de la poutre, le panneau reste ouvert et se relit depuis la note',
        cApres === cBib && cApres !== cAvant && !(await g.$eval('#et-bibliotheque', (x) => x.hidden)) && (await g.$eval('#et-bibliotheque [data-utiliser]', (x) => x.textContent.trim())) === bb.utiliser,
        JSON.stringify({ cAvant, cBib, cApres }));
      await g.click('#et-bibliotheque [data-inclure="fiche"]');
      await tranquille();
      await g.waitForTimeout(300);
      await g.keyboard.press('Escape');
      await g.waitForTimeout(150);
      const fi = await g.evaluate(() => { const id = new URLSearchParams(location.search).get('note'); const n = JSON.parse(localStorage.getItem('stab.notes.v1'))[id];
        return { ferme: document.getElementById('et-bibliotheque').hidden, fiches: (n.etat.fiches || []).map((f) => f.titre), bloc: (document.querySelector('#donnees .et-fiches .n') || {}).textContent,
          onglet: !!document.querySelector('#vues [data-vue="fiche-0"]'), focus: !!document.activeElement && document.activeElement.matches('.top [data-action="bibliotheque"]') }; });
      // l'en-tête du bloc, VU (relecture du 10/11/2026 : son style était parti avec l'ancienne arborescence, l'œil lisait
      // « Justifications1 ») : en capitales, à sa hauteur, le compte en pastille détachée du mot
      const tj = await g.evaluate(() => { const h = document.querySelector('#donnees .et-fiches > div:first-child'), n = h && h.querySelector('.n'), t = h && h.querySelector('span');
        if (!h || !n || !t) return null;
        const r = h.getBoundingClientRect(), rn = n.getBoundingClientRect(), rt = t.getBoundingClientRect();
        return { capitales: getComputedStyle(h).textTransform === 'uppercase', h: Math.round(r.height), ecart: Math.round(rn.left - rt.right),
          pastille: getComputedStyle(n).backgroundColor !== 'rgba(0, 0, 0, 0)', vu: h.checkVisibility({ visibilityProperty: true, opacityProperty: true }) }; });
      verif('pilote : l\'en-tête du bloc des justifications se lit « JUSTIFICATIONS 1 » — en capitales, à sa hauteur (24 px et plus), le compte en pastille détachée du mot',
        !!tj && tj.vu && tj.capitales && tj.h >= 24 && tj.ecart >= 4 && tj.pastille, JSON.stringify(tj));
      await g.click('#donnees [data-vue-fiche="fiche-0"]');
      await g.waitForTimeout(250);
      const vf = await g.evaluate(() => { const v = document.querySelector('#stage .view.on'); const f = v && v.querySelector('.et-fiche'); const r = f && f.getBoundingClientRect();
        return { vue: v && v.dataset.vue, titre: f && f.querySelector('.et-fiche-ht b').textContent, visible: !!r && r.width > 400 && r.height > 200 && r.top >= 0,
          lignes: f ? f.querySelectorAll('.et-fiche-t tr').length : 0, extrait: !!(f && f.querySelector('.et-fiche-xt td.cur')) }; });
      verif('pilote : « Inclure dans la note » ajoute une fiche (état, bloc « Justifications 1 », onglet J1) ; Échap referme et rend le focus ; la fiche se lit au centre (lignes, extrait, case)',
        fi.ferme && fi.fiches.length === 1 && /^Enrobage poutre/.test(fi.fiches[0]) && fi.bloc === '1' && fi.onglet && fi.focus
        && vf.vue === 'fiche-0' && vf.titre === fi.fiches[0] && vf.visible && vf.lignes >= 6 && vf.extrait, JSON.stringify({ fi, vf }));
      await g.screenshot({ path: path.join(OUT, 'site_pilote_justification.png') });
      // la note PDF, faite dans le navigateur — avec la page des justifications et la fiche J1
      const [pdf] = await Promise.all([g.waitForEvent('download', { timeout: 120000 }), g.click('.top [data-action="pdf"]')]);
      const fpdf = path.join(OUT, 'site_pilote_note.pdf');
      await pdf.saveAs(fpdf);
      const tete = fs.readFileSync(fpdf).subarray(0, 5).toString();
      const txtPdf = require('child_process').execSync(`pdftotext -layout "${fpdf}" -`, { encoding: 'utf8' });
      verif('pilote : « Note PDF » télécharge la note de calcul (PDF fait dans le navigateur), avec la page « Justifications de la bibliothèque » et la fiche J1',
        /\.pdf$/.test(pdf.suggestedFilename()) && tete === '%PDF-' && fs.statSync(fpdf).size > 30000 && /Justifications de la bibliothèque/.test(txtPdf) && /Enrobage poutre/.test(txtPdf),
        `${pdf.suggestedFilename()} · ${tete} · ${txtPdf.length} caractères`);
      // le .stab : un nom qui finit par .stab (jamais .stab.json), toute la note dedans — la fiche aussi
      const [stab] = await Promise.all([g.waitForEvent('download'), g.click('.top [data-action="exporter"]')]);
      const fstab = path.join(OUT, 'site_pilote.stab');
      await stab.saveAs(fstab);
      const lu = JSON.parse(fs.readFileSync(fstab, 'utf8'));
      verif('pilote : « .stab » enregistre « ….stab » (pas de « .json »), avec toute la note (la fiche comprise)', /\.stab$/.test(stab.suggestedFilename()) && lu.format === 'stab' && lu.note.etat.poutres[0].sections[0].V === 40
        && (lu.note.etat.fiches || []).length === 1, stab.suggestedFilename());
      // le menu ⋯ de la fiche la retire : plus de bloc, plus d'onglet, plus de clé « fiches » dans l'état
      await g.click('#donnees .et-fi[data-fiche="0"] [data-menu]');
      await g.click('#donnees .et-fi[data-fiche="0"] .et-mn-l [data-fiche-retirer="0"]');
      await tranquille();
      await g.waitForTimeout(200);
      verif('pilote : le menu ⋯ de la fiche la retire — plus de bloc « Justifications », plus d\'onglet J1, l\'état sans clé « fiches »',
        !(await g.$('#donnees .et-fiches')) && !(await g.$('#vues [data-vue="fiche-0"]'))
        && (await g.evaluate(() => { const id = new URLSearchParams(location.search).get('note'); return !('fiches' in JSON.parse(localStorage.getItem('stab.notes.v1'))[id].etat); })));
      // « Enregistrer comme note… » : le nom, le chantier, le dossier
      await g.click('.top [data-ranger]');
      verif('pilote : « Classer la note… » ouvre la fenêtre, le nom déjà rempli', (await g.isVisible('#et-rangement')) && (await g.inputValue('#et-rg-nom')) === 'Poutres béton armé');
      await g.fill('#et-rg-nom', 'Poutres R+2');
      await g.fill('#et-rg-chantier', 'Réalex');
      await g.fill('#et-rg-dossier', 'Étude de la dalle');
      await g.click('#et-rg-form [type=submit]');
      await g.waitForTimeout(200);
      const r = await g.evaluate(() => {
        const id = new URLSearchParams(location.search).get('note'), n = JSON.parse(localStorage.getItem('stab.notes.v1'))[id];
        return { nom: n.nom, chantier: n.chantier, dossier: n.dossier, entete: document.getElementById('nom-note').value, fil: document.getElementById('fil-chantier').textContent,
          cartouche: [...document.querySelectorAll('#stage .view.on .cart b')].map((x) => x.textContent).slice(0, 2).join(' | '), ouverte: !document.getElementById('et-rangement').hidden };
      });
      verif('pilote : « Enregistrer comme note » écrit le nom, le chantier et le dossier ; l\'en-tête et le cartouche les disent',
        r.nom === 'Poutres R+2' && r.chantier === 'Réalex' && r.dossier === 'Étude de la dalle' && r.entete === 'Poutres R+2' && r.fil.includes('Réalex')
        && r.cartouche === 'Poutres R+2 | Réalex' && !r.ouverte, JSON.stringify(r));
      // « Ajouter une poutre » (la dernière entrée de la liste) : un deuxième élément, ouvert
      await g.click(`${LISTE_ELT} [data-menu]`);
      await g.click(`${LIGNE} .et-elt-liste .et-mn-l [data-ajouter]`);
      await tranquille();
      await g.waitForTimeout(200);
      const aj = { ...(await etatLigne()), liste: await liste() };
      verif('pilote : « Ajouter une poutre » (dans la liste) ajoute une poutre et l\'ouvre — « V2 » dans la ligne et en tête des résultats, la liste en montre deux, « 2 » sur son bouton',
        aj.n === '2' && aj.nom === 'V2' && aj.rp === 'V2' && aj.liste === 'V1,V2*' && aj.compte === '2' && aj.supprimer, JSON.stringify(aj));
      // les colonnes : repli (la toile gagne la place), poignée (la largeur suit et se garde)
      const l0 = await g.$eval('.canvas .stage', (x) => x.getBoundingClientRect().width);
      await g.click('.panel.left [data-coll="g"]');
      await g.waitForTimeout(400);
      const l1 = await g.$eval('.canvas .stage', (x) => x.getBoundingClientRect().width);
      const bande = await g.$eval('.panel.left', (x) => { const q = x.getBoundingClientRect(); return { l: Math.round(q.width), t: Math.round(q.height) }; });
      verif('pilote : les données se replient en une bande sur toute la hauteur, la toile gagne la place', l1 > l0 + 150 && bande.l <= 50 && bande.t > 600, JSON.stringify({ l0, l1, bande }));
      await g.click('.panel.left [data-coll="g"]');
      await g.waitForTimeout(400);
      const w0 = await g.$eval('.panel.left', (x) => x.getBoundingClientRect().width);
      // on vise le TRAIT (le bord du panneau), comme à la main
      const h = await g.$eval('.panel.left', (x) => { const q = x.getBoundingClientRect(); return { x: q.right - 0.5, y: q.top + 300 }; });
      await g.mouse.move(h.x, h.y);
      await g.mouse.down();
      await g.mouse.move(h.x + 80, h.y, { steps: 8 });
      await g.mouse.up();
      await g.waitForTimeout(200);
      const w1 = await g.$eval('.panel.left', (x) => x.getBoundingClientRect().width);
      await g.reload();
      await pretEcran(g, { exigerPret: true });
      const w2 = await g.$eval('.panel.left', (x) => x.getBoundingClientRect().width);
      verif('pilote : la poignée élargit les données de 80 px, et la largeur se garde au rechargement', Math.abs(w1 - w0 - 80) <= 2 && Math.abs(w2 - w1) <= 1, JSON.stringify({ w0, w1, w2 }));
      await g.dblclick('.bench > .et-poignee[data-poignee="g"]');
      // un enregistrement de l'écran Streamlit (poutre_ba.json) s'ouvre : une note du site, la poutre traduite
      const fst = path.join(OUT, 'poutre_ba.json');
      fs.writeFileSync(fst, JSON.stringify({ version: '2.39', beams: [{ id: 1, nom: 'Poutre 1', sections: [{ id: 1, nom: 'A' }] }],
        values: { b1_b: 25, b1_h: 55, b1_beton: 'C25/30', b1_sec1_M_inf: 95, b1_sec1_V: 50, b1_sec1_nlits_inf: 2, b1_sec1_n_as_inf: 3, 'b1_sec1_ø_as_inf': 16 } }));
      m0 = await montage(g);
      await Promise.all([g.waitForNavigation(), g.setInputFiles('#ouvrir', fst)]);
      await pretEcran(g, { apres: m0, exigerPret: true });
      await tranquille();
      const st = await g.evaluate(() => { const id = new URLSearchParams(location.search).get('note'); const n = JSON.parse(localStorage.getItem('stab.notes.v1'))[id];
        return { nom: n.nom, p: n.etat && n.etat.poutres && n.etat.poutres[0] }; });
      verif('pilote : un enregistrement Streamlit (poutre_ba.json) s\'ouvre en note du site, la poutre traduite',
        st.nom === 'poutre_ba' && st.p && st.p.b === 25 && st.p.beton === 'C25/30' && st.p.sections[0].inf.length === 2, JSON.stringify(st).slice(0, 200));
      // rouverte depuis l'accueil, la note est reprise (aucune nouvelle note ne s'écrit)
      await ouvrir(g, `${BASE}/index.html`);
      const avant = await nNotes();
      m0 = await montage(g);
      await Promise.all([g.waitForNavigation(), g.click('#recentes .rcard >> nth=1')]);
      await pretEcran(g, { apres: m0, exigerPret: true });
      verif('pilote : rouverte depuis « Reprendre », la note est la même (aucune autre ne s\'écrit)',
        (await nNotes()) === avant && (await g.inputValue('#nom-note')) === 'Poutres R+2' && (await g.$eval(LIGNE, (x) => x.dataset.elements)) === '2');
      await g.screenshot({ path: path.join(OUT, 'site_pilote_reprise.png') });
      verif('pilote : aucune erreur JavaScript', !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
    } catch (x) {
      verif('pilote : le parcours va au bout', false, String(x.message).split('\n').slice(0, 3).join(' | '));
      await g.screenshot({ path: path.join(OUT, 'site_pilote_echec.png') }).catch(() => {});
    }
    await c.close();
  }

  // ---------------------------------------------------------------- « Détailler dans Taux d'armature » (poutre → taux)
  // Une poutre dont l'option « Taux d'armature » est allumée porte, sous son nom, le bouton qui l'envoie dans le module
  // détaillé (docs/site/LOT_Q_QUATRE_MODULES.md § 6.2). Éteint — et il dit pourquoi — tant qu'une section n'a pas de
  // longueur : un clic n'y fait rien. Allumé, il crée UNE note « Taux d'armature » (même chantier, nommée d'après la
  // poutre), l'ouvre, et le module détaillé dit d'où vient la poutre et quand ; la note de la poutre n'est pas touchée.
  // Au PC et au téléphone (R1 : le bouton s'atteint aux deux formats, à hauteur de doigt). Captures : site_envoi_*.png.
  //   filtre « envoi » (joué aussi par « pilote »)
  if (!FILTRE || FILTRE === 'pilote' || FILTRE === 'envoi') {
    const POUTRE_ENVOI = { nom: 'V1', b: 30, h: 60, enrobage: 3, taux: { actif: true },
      sections: [{ nom: 'A', M_inf: 120, V: 60, inf: [{ n: 3, d: 20 }], sup: [{ n: 2, d: 12 }], etriers: [{ type: 'etrier', d: 8 }, { type: 'epingle', d: 8 }], pas: 20, longueur: 4 },
        { nom: 'B', M_sup: 95, V: 120, inf: [{ n: 2, d: 20 }], sup: [{ n: 4, d: 16 }], etriers: [{ type: 'etrier', d: 8 }], pas: 10, longueur: 2 }] };
    for (const [format, options, tel] of [['PC 1440', { viewport: { width: 1440, height: 900 } }, false], ['téléphone', { ...devices['iPhone 13'] }, true]]) {
      const c = await b.newContext(options);
      const g = await c.newPage();
      g.erreurs = [];
      g.on('pageerror', (e) => g.erreurs.push(e.message));
      g.on('console', (m) => { if (m.type() === 'error') g.erreurs.push(m.text().slice(0, 300)); });
      const tranquille = () => g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
      const notes = () => g.evaluate(() => { try { return JSON.parse(localStorage.getItem('stab.notes.v1') || '{}'); } catch (e) { return {}; } });
      const semer = (id, poutre) => g.evaluate(([i, p]) => { const k = 'stab.notes.v1', ns = JSON.parse(localStorage.getItem(k) || '{}'), t = Date.now() - 60000;
        ns[i] = { id: i, module: 'poutre', nom: 'Poutres du R+1', chantier: 'Réalex', dossier: '', indice: 'A', cree: t, modifie: t,
          etat: { version: 1, poutres: [p], actif: 0 }, resume: null, vignette: '' };
        localStorage.setItem(k, JSON.stringify(ns)); }, [id, poutre]);
      const bouton = () => g.evaluate(() => { const x = document.querySelector('#resultat [data-transmettre]');
        return x ? { vers: x.dataset.transmettre, eteint: x.disabled, vu: x.checkVisibility({ visibilityProperty: true, opacityProperty: true }),
          h: Math.round(x.getBoundingClientRect().height), texte: x.textContent.replace(/\s+/g, ' ').trim() } : null; });
      const versResultat = async () => { if (tel) { await g.click('.mtabs [data-onglet="resultat"]'); await g.waitForTimeout(200); } };
      try {
        await ouvrir(g, `${BASE}/index.html`);
        const sans = JSON.parse(JSON.stringify(POUTRE_ENVOI));
        delete sans.sections[1].longueur;
        await semer('envoi-sans', sans);
        await semer('envoi-avec', POUTRE_ENVOI);
        // ---- une section sans longueur : éteint, il dit pourquoi, et un clic n'y fait rien
        await pretEcran(g, { apres: await ouvrir(g, `${BASE}/poutre?note=envoi-sans`) });
        await tranquille();
        await versResultat();
        let bt = await bouton();
        verif(`envoi · ${format} : une section sans longueur — le bouton « Détailler dans Taux d'armature » se voit, éteint, et dit pourquoi`,
          !!bt && bt.vers === 'taux' && bt.vu && bt.eteint && /Détailler dans Taux d'armature/.test(bt.texte) && /longueur/.test(bt.texte), JSON.stringify(bt));
        const adresse = g.url();
        await g.$eval('#resultat [data-transmettre]', (x) => { x.click(); x.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })); });
        await g.waitForTimeout(700);
        verif(`envoi · ${format} : éteint, un clic n'y fait rien (même adresse, aucune note créée)`,
          g.url() === adresse && Object.keys(await notes()).length === 2, `${g.url()} · ${Object.keys(await notes()).length} notes`);
        // ---- chaque section a sa longueur : le bouton envoie la poutre
        await pretEcran(g, { apres: await ouvrir(g, `${BASE}/poutre?note=envoi-avec`) });
        await tranquille();
        await versResultat();
        bt = await bouton();
        verif(`envoi · ${format} : chaque section a sa longueur — le bouton est allumé${tel ? ', à hauteur de doigt (44 px)' : ''}`,
          !!bt && bt.vers === 'taux' && bt.vu && !bt.eteint && (!tel || bt.h >= 44), JSON.stringify(bt));
        await g.screenshot({ path: path.join(OUT, `site_envoi_${tel ? 'tel' : 'pc'}_poutre.png`) });
        const avant = await notes();
        const m0 = await montage(g);
        await g.click('#resultat [data-transmettre]');
        await g.waitForFunction(() => /taux-armature/.test(location.pathname), null, { timeout: 60000 });
        await pretEcran(g, { apres: m0 });
        await tranquille();
        const apres = await notes();
        const neuves = Object.values(apres).filter((n) => !avant[n.id]);
        const id = new URLSearchParams(new URL(g.url()).search).get('note');
        verif(`envoi · ${format} : UNE note « Taux d'armature » est créée et ouverte — nommée d'après la poutre, dans le même chantier`,
          neuves.length === 1 && neuves[0].module === 'taux' && neuves[0].id === id && neuves[0].nom === 'V1 · taux d\'armature' && neuves[0].chantier === 'Réalex',
          JSON.stringify(neuves.map((n) => [n.module, n.nom, n.chantier, n.id === id])));
        verif(`envoi · ${format} : la note de la poutre n'est pas touchée (même état, même date)`,
          !!apres['envoi-avec'] && JSON.stringify(apres['envoi-avec'].etat) === JSON.stringify(avant['envoi-avec'].etat) && apres['envoi-avec'].modifie === avant['envoi-avec'].modifie);
        await versResultat();
        const recu = await g.evaluate(() => ({ jour: new Date().toLocaleDateString('fr-BE'),
          res: document.querySelector('#resultat').textContent.replace(/\s+/g, ' '),
          vu: document.querySelector('#resultat').checkVisibility({ visibilityProperty: true, opacityProperty: true }) }));
        verif(`envoi · ${format} : le module détaillé dit d'où vient la poutre et quand (« reçue de la poutre V1, le ${recu.jour} »)`,
          recu.vu && recu.res.includes(`reçue de la poutre V1, le ${recu.jour}`), recu.res.slice(0, 220));
        verif(`envoi · ${format} : la poutre reçue est calculée (30 × 60, 6,00 m, un taux en kg/m³)`,
          /30\s*×\s*60/.test(recu.res) && /6,00/.test(recu.res) && /\d\s*kg\/m³/.test(recu.res), recu.res.slice(0, 220));
        await g.screenshot({ path: path.join(OUT, `site_envoi_${tel ? 'tel' : 'pc'}_taux.png`) });
        verif(`envoi · ${format} : aucune erreur JavaScript`, !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
      } catch (e) { verif(`envoi · ${format} : le parcours va au bout`, false, String(e.message).split('\n')[0]); }
      await c.close();
    }
  }

  // ---------------------------------------------------------------- les poignées d'un cadre, AU DOIGT (tablette)
  // Signalé par l'ingénieur (11/2026) : sur tablette, les poignées se voient mais ne bougent pas — le navigateur prenait
  // l'appui pour faire défiler la page et annulait le glisser (pointercancel). Les événements tactiles sont ceux de
  // Chromium (CDP Input.dispatchTouchEvent), avec leurs gestes par défaut : un glisser à la souris ne le voit pas.
  if (!FILTRE || FILTRE === 'pilote' || FILTRE === 'doigt') {
    const c = await b.newContext({ viewport: { width: 1180, height: 820 }, hasTouch: true, isMobile: true });
    const g = await c.newPage();
    g.erreurs = [];
    g.on('pageerror', (e) => g.erreurs.push(e.message));
    try {
      await pretEcran(g, { apres: await ouvrir(g, `${BASE}/poutre.html`) });
      const tranquille = () => g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
      const cdp = await c.newCDPSession(g);
      const touche = (type, x, y) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: type === 'touchEnd' ? [] : [{ x, y }] });
      if (!(await g.$eval('#donnees .sec[data-sec="arm"]', (x) => x.classList.contains('open')))) await g.click('#donnees .sec[data-sec="arm"] > .sechd');
      await g.fill('#donnees input[data-chemin="sections/0/inf/0/n"]', '4');
      await g.press('#donnees input[data-chemin="sections/0/inf/0/n"]', 'Enter');
      await tranquille();
      await g.waitForTimeout(300);
      const etriers = () => g.evaluate(() => { const id = new URLSearchParams(location.search).get('note'); const n = JSON.parse(localStorage.getItem('stab.notes.v1') || '{}')[id]; return n ? n.etat.poutres[0].sections[0].etriers : null; });
      // choisir le cadre d'un appui sur son trait, puis glisser la poignée « a » (sur la 4e barre) jusqu'à la 3e
      const tr = await g.$eval('#stage .view.on .pc.cadre[data-cadre="0"] path', (x) => { const q = x.getBoundingClientRect(); return { x: q.left + 2, y: q.top + q.height / 2 }; });
      await touche('touchStart', tr.x, tr.y);
      await g.waitForTimeout(60);
      await touche('touchEnd');
      await g.waitForTimeout(250);
      const h = await g.$eval('#stage .view.on .pc.cadre.et-sel .poignee[data-bout="a"] .o', (x) => { const q = x.getBoundingClientRect(); return { x: q.left + q.width / 2, y: q.top + q.height / 2 }; });
      const b3 = await g.evaluate(() => { const cs = [...document.querySelectorAll('#stage .view.on .pc.lit[data-sel$=".lit-inf-0"] > circle')]; cs.sort((u, v) => +u.getAttribute('cx') - +v.getAttribute('cx')); const q = cs[2].getBoundingClientRect(); return { x: q.left + q.width / 2, y: q.top + q.height / 2 }; });
      await g.evaluate(() => { window.__annule = 0; window.addEventListener('pointercancel', () => { window.__annule += 1; }, true); });
      await touche('touchStart', h.x, h.y);
      for (let i = 1; i <= 10; i++) { await touche('touchMove', h.x + (b3.x - h.x) * i / 10, h.y); await g.waitForTimeout(30); }
      await touche('touchEnd');
      await tranquille().catch(() => {});
      await g.waitForTimeout(400);
      const apres = await etriers();
      const annule = await g.evaluate(() => window.__annule);
      verif('doigt : sur tablette, la poignée d\'un cadre se glisse au doigt (sans annulation) jusqu\'à la 3e barre — le cadre entoure 1 → 3',
        annule === 0 && apres && apres[0].de === 1 && apres[0].a === 3, JSON.stringify({ annule, apres }));
      // les explications rangées dans ⓘ (données condensées) : au doigt, pas de survol — un appui ouvre la bulle, un second
      // la ferme ; le LIBELLÉ du champ l'ouvre aussi (l'icône seule est une petite cible) ; une autre ⓘ (deux champs sur une
      // ligne) change le texte ; un appui ailleurs la ferme
      const lireBulle = () => g.evaluate(() => { const x = document.querySelector('.aide-bulle'); const q = x && x.getBoundingClientRect();
        return { vu: !!x && x.checkVisibility({ visibilityProperty: true, opacityProperty: true }), texte: x ? x.textContent : '',
          dedans: !!q && q.left >= 0 && q.top >= 0 && q.right <= innerWidth && q.bottom <= innerHeight }; });
      for (const k of ['pt', 'op']) if (!(await g.$eval(`#donnees .sec[data-sec="${k}"]`, (x) => x.classList.contains('open')))) await g.tap(`#donnees .sec[data-sec="${k}"] > .sechd`);
      const enr = '#donnees .row:has(input[data-chemin="enrobage"]) .aide';
      await g.$eval(enr, (x) => x.scrollIntoView({ block: 'center' }));
      await g.waitForTimeout(150);
      await g.tap(enr);
      const b1 = await lireBulle();
      await g.tap(enr);
      const b2 = await lireBulle();
      await g.tap('#donnees .row:has(input[data-chemin="enrobage"]) > label > span:not(.aide):not(.sym)');
      const b2l = await lireBulle();
      const jeu = '#donnees .row2:has(input[data-chemin="jeu_premier_lit"]) .aide';
      await g.$eval(jeu, (x) => x.scrollIntoView({ block: 'center' }));
      await g.waitForTimeout(150);
      await g.tap(jeu);
      const b3b = await lireBulle();
      // (un appui sur le libellé « Méthode », sans aide, tout près et dans l'écran : rien ne défile, c'est bien l'appui
      // ailleurs qui la ferme)
      await g.tap('#donnees .row.et-meth > label > span');
      const b4 = await lireBulle();
      verif('doigt : une aide ⓘ s\'ouvre d\'un appui (bulle dans l\'écran, son texte), se ferme d\'un second ; le libellé du champ l\'ouvre aussi ; celle d\'un champ à deux par ligne aussi ; un appui ailleurs la ferme',
        b1.vu && b1.dedans && b1.texte.startsWith('Distance du parement') && !b2.vu && b2l.vu && b2l.texte.startsWith('Distance du parement')
        && b3b.vu && b3b.dedans && b3b.texte.startsWith('Ajouté sous le lit 1') && !b4.vu,
        JSON.stringify({ b1, b2, b2l, b3b, b4 }));
      verif('doigt : aucune erreur JavaScript', !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
    } catch (x) {
      verif('doigt : le parcours va au bout', false, String(x.message).split('\n').slice(0, 3).join(' | '));
      await g.screenshot({ path: path.join(OUT, 'site_doigt_echec.png') }).catch(() => {});
    }
    await c.close();
  }

  // ---------------------------------------------------------------- les CORNIÈRES d'appui (refonte du 10/11/2026, établi compact)
  // Le module refondu sur la méthode du classeur de l'ingénieur (docs/acier/CORNIERE_DECISIONS.md § 4 ; api_web/corniere.py) :
  // à gauche les quatre rubriques, au centre les quatre vues (coupe, moments, contraintes, élévation), à droite le verdict
  // et trois blocs THÉMATIQUES (Sections, Ancrages, Comparaison). Ce qu'on contrôle, à l'écran : la configuration
  // A puis B (la coupe change : l'aile horizontale passe en haut, le contact C au bas de la cornière — MESURÉ dans le SVG,
  // par la position des pièces R, Y, yc et profil) ; la saisie dx qui change le taux ; le clic sur une pièce du dessin qui
  // ouvre sa rubrique et éclaire sa ligne ; le sélecteur (233 cornières, ✓, ★, les grises hors domaine) et le choix de
  // l'optimale ; le mode équilibre (le pivot y_c dessiné) ; une inégale tapée dans l'autre ordre ; une saisie refusée
  // (toast, valeur rendue) ; une note v1 (modèle lame) rouverte avec son avis ; puis les trois formats — PC 1440,
  // tablette 1024 × 768, iPhone 13 (390 × 844, tactile) — pour A, B et A en équilibre : rien ne déborde, aucun nom coupé
  // (les TITRES des blocs compris : « Ancrages » se réduisait à 6 px sous un état d'en-tête trop long, et « Mode méthode
  // historique (classeur P8) » était coupé en A équilibre — relecture du 10/11/2026), les cibles à 44 px au doigt, les
  // champs en 16 px au téléphone. Les captures : site_corniere_*.png (A, B et A-equilibre aux trois formats, les quatre
  // vues, le sélecteur, la saisie refusée, la note v1).
  // Révision § 9 (10/11/2026 soir, CORNIERE_DECISIONS.md § 9) : le volet « Détails » de la fiche du catalogue, replié par
  // défaut, s'ouvre, survit à un calcul (js/etabli.js le garde) et se referme ; la bascule ELS → ELU change la limite
  // affichée (236,7 → 355 N/mm²) et masque coef et R_ELU, l'ELS les rend ; le support « poutre » fait apparaître ses deux
  // étendues et les distances aux bords dans le bloc Ancrages, la coupe reste un dessin (son contenu est au lot des
  // dessins), une étendue trop courte est refusée (toast, valeur rendue) ; un entraxe saisi se lit dans les positions
  // (e₁ · s · z) ; la saisie libre montre A, B, t, r directement. Les formats jouent aussi ELU et poutre.
  //   filtre « corniere » ; « module:corniere » le rejoue aussi (après le contrôle commun).
  if (!FILTRE || FILTRE === 'corniere' || FILTRE === 'module:corniere') {
    const c = await b.newContext({ viewport: { width: 1440, height: 900 } });
    const g = await c.newPage();
    g.erreurs = [];
    g.on('pageerror', (e) => g.erreurs.push(e.message));
    g.on('console', (m) => { if (m.type() === 'error') g.erreurs.push(m.text().slice(0, 300)); });
    const pretG = (o) => pretEcran(g, o);
    const tranquille = () => g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
    const ouvrirRub = async (k) => {
      if (!(await g.$eval(`#donnees .sec[data-sec="${k}"]`, (x) => x.classList.contains('open')))) await g.click(`#donnees .sec[data-sec="${k}"] > .sechd`);
      await g.waitForTimeout(80);
    };
    const saisir = async (s, v) => { await g.fill(s, v); await g.press(s, 'Enter'); await tranquille(); await g.waitForTimeout(150); };
    const lireToast = () => g.evaluate(() => { const t = document.getElementById('toast'); return { vu: t && !t.hidden && t.checkVisibility(), texte: t ? t.textContent : '', erreur: !!t && t.classList.contains('erreur') }; });
    const taux = () => g.evaluate(() => { const x = document.querySelector('#resultat .verdict .chip.xl'); return x ? parseFloat(x.textContent.replace(/[^\d,]/g, '').replace(',', '.')) : null; });
    const gouv = () => g.evaluate(() => (document.querySelector('#resultat .verdict .gov b') || {}).textContent || '');
    const blocsVus = () => g.$$eval('#resultat .et-bloc[data-theme]', (bs) => bs.filter((x) => !x.hidden && x.checkVisibility()).map((x) => x.querySelector('.et-bloc-h .n').textContent));
    // la position à l'écran d'une pièce de la coupe (le centre et la taille de sa boîte), null si elle n'y est pas
    const piece = (sel) => g.evaluate((sel) => {
      const e = document.querySelector(`.view.on .draw svg .pc[data-sel="${sel}"]`);
      if (!e) return null;
      const r = e.getBoundingClientRect();
      return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2), w: Math.round(r.width), h: Math.round(r.height) };
    }, sel);
    // un appui sur une pièce du dessin, comme js/compact.js l'écoute (pointerdown sur la toile, pointerup sans bouger)
    const appuyer = async (sel) => {
      const s = `.view.on .draw svg .pc[data-sel="${sel}"]`;
      await g.dispatchEvent(s, 'pointerdown', { pointerId: 7, pointerType: 'mouse', button: 0, clientX: 20, clientY: 20, bubbles: true, composed: true });
      await g.dispatchEvent(s, 'pointerup', { pointerId: 7, pointerType: 'mouse', button: 0, clientX: 20, clientY: 20, bubbles: true, composed: true });
      await g.waitForTimeout(250);
    };
    try {
      await pretG({ apres: await ouvrir(g, `${BASE}/corniere?${new URLSearchParams({ nom: 'Cornière d\'appui', chantier: 'Essai cornières' })}`) });
      await tranquille();
      // A, méthode historique : la cornière neuve (L 120×120×15, 35 kN, 75 ± 10, 85 ± 5 — les jeux du § 9.4) — η entre 75 et 95 %
      // (84,2 % mesuré ; 95,8 % avec les anciens jeux 20/10), en section 3
      const t0 = await taux();
      const blocsA = await blocsVus();
      verif('cornière A historique : le verdict de la cornière neuve (η entre 75 et 95 %, section 3 · aile verticale), trois blocs thématiques visibles (Sections, Ancrages, Comparaison)',
        t0 > 75 && t0 < 95 && /Section 3/.test(await gouv()) && blocsA.join() === 'Sections,Ancrages,Comparaison', JSON.stringify({ t0, gouv: await gouv(), blocsA }));
      const combs = await g.$$eval('#resultat .et-bloc[data-theme="0"] .el .cmp', (e) => e.map((x) => x.textContent).filter((t) => /charge à X/.test(t)).length);
      verif('cornière : chaque vérification du bloc Sections dit la combinaison de jeux qui la gouverne (« charge à X + 10, ancrage à Y + 5 »)', combs >= 6, `${combs} ligne(s)`);
      // la limite affichée sur la ligne « Cornière » du bloc Sections (ELS : σ_adm = 236,7 N/mm²) ; relue après la bascule ELU
      const limite = () => g.$eval('#resultat .et-bloc[data-theme="0"] .el .cmp', (x) => x.textContent);
      verif('cornière ELS : la ligne Cornière dit σadm = 236,7 et τadm = 136,6 N/mm² (f_y/1,5 et σadm/√3)', /σadm = 236,7/.test(await limite()) && /τadm = 136,6/.test(await limite()), await limite());
      const horsVerdict = await g.$$eval('#resultat .et-bloc[data-theme="0"] .et-mot', (e) => e.map((x) => x.textContent).filter((t) => t === 'hors verdict').length);
      verif('cornière A historique : les deux maxima du congé sont en information, « hors verdict » (le classeur ne les a pas)', horsVerdict === 2, `${horsVerdict}`);
      await g.screenshot({ path: path.join(OUT, 'site_corniere_A_pc.png') });
      // les quatre vues, chacune visible à sa taille
      for (const k of ['moments', 'contraintes', 'elevation']) {
        await g.click(`#vues [data-vue="${k}"]`);
        await g.waitForTimeout(250);
        const tl = await g.evaluate(() => { const d = [...document.querySelectorAll('.view.on .draw')].find((e) => e.checkVisibility()); const svg = d && d.querySelector(':scope > svg'); if (!svg) return null; const r = svg.getBoundingClientRect(); return { l: Math.round(r.width), t: Math.round(r.height), textes: [...svg.querySelectorAll('text')].filter((x) => x.getBoundingClientRect().height > 0).length }; });
        verif(`cornière : la vue « ${k} » est un dessin visible à sa taille, avec ses textes`, tl && tl.l > 400 && tl.t > 300 && tl.textes > 10, JSON.stringify(tl));
        await g.screenshot({ path: path.join(OUT, `site_corniere_vue_${k}.png`) });
      }
      await g.click('#vues [data-vue="coupe"]');
      await g.waitForTimeout(200);
      // un appui sur l'ancrage du dessin (pièce Y) : la rubrique Ancrage (fermée) s'ouvre, la ligne Y s'éclaire, la pièce aussi
      const avantY = await g.$eval('#donnees .sec[data-sec="an"]', (x) => x.classList.contains('open'));
      await appuyer('Y');
      const selY = await g.evaluate(() => ({ ouverte: document.querySelector('#donnees .sec[data-sec="an"]').classList.contains('open'),
        cible: !!document.querySelector('#donnees [data-cible="Y"].et-cible'), piece: !!document.querySelector('.view.on .pc[data-sel="Y"].et-sel'),
        vue: !!document.querySelector('#donnees [data-cible="Y"]') && document.querySelector('#donnees [data-cible="Y"]').checkVisibility() }));
      verif('cornière : un appui sur l\'ancrage du dessin ouvre la rubrique Ancrage (fermée avant) et éclaire la ligne Y des données et la pièce du dessin',
        !avantY && selY.ouverte && selY.cible && selY.piece && selY.vue, JSON.stringify({ avantY, selY }));
      await g.screenshot({ path: path.join(OUT, 'site_corniere_selection.png') });
      await g.keyboard.press('Escape');
      await g.waitForTimeout(100);
      // la saisie dx = 0 : la charge à X seule, le taux baisse ; puis dx = 10 (le défaut du § 9.4) le remonte
      await ouvrirRub('ch');
      await saisir('input[data-chemin="dx"]', '0');
      const t1 = await taux();
      await saisir('input[data-chemin="dx"]', '10');
      const t2 = await taux();
      verif('cornière : dx = 0 fait baisser le taux (plus d\'enveloppe sur la position de la charge), dx = 10 le rend', t1 !== null && t1 < t0 - 5 && t2 === t0, `${t0} → ${t1} → ${t2}`);
      // une saisie refusée : X = 10 (la charge dans l'angle) — le toast d'erreur, le champ rend 75
      await g.fill('input[data-chemin="X"]', '10');
      await g.press('input[data-chemin="X"]', 'Enter');
      await tranquille();
      await g.waitForTimeout(250);
      const refus = await lireToast();
      const xApres = await g.inputValue('input[data-chemin="X"]');
      verif('cornière : X = 10 est refusé par le moteur (« dans l\'angle »), le toast le dit, le champ rend 75', refus.vu && refus.erreur && /angle/.test(refus.texte) && xApres === '75', JSON.stringify({ refus, xApres }));
      await g.screenshot({ path: path.join(OUT, 'site_corniere_refus.png') });
      // ---- révision § 9.7 : le volet « Détails » de la fiche du catalogue — replié par défaut, il s'ouvre, survit au calcul qui
      // suit une saisie (le cadre est refait : js/etabli.js garde l'état), se referme
      await ouvrirRub('co');
      const volet = () => g.evaluate(() => { const d = document.querySelector('#donnees details.et-volet[data-volet="fiche"]'); const f = d && d.querySelector('.facts');
        return d ? { present: true, ouvert: d.open, fiche: !!f && f.checkVisibility(), sommaire: (d.querySelector('summary') || {}).textContent || '' } : { present: false }; });
      const v0 = await volet();
      await g.click('#donnees details.et-volet[data-volet="fiche"] > summary');
      await g.waitForTimeout(150);
      const v1o = await volet();
      await saisir('input[data-chemin="dy"]', '6');
      const v2 = await volet();
      await g.click('#donnees details.et-volet[data-volet="fiche"] > summary');
      await g.waitForTimeout(150);
      const v3 = await volet();
      verif('cornière : le volet « Détails » (fiche du catalogue) est replié par défaut, s\'ouvre au clic (la fiche se voit), reste ouvert après une saisie, se referme',
        v0.present && !v0.ouvert && !v0.fiche && /Détails/.test(v0.sommaire) && /26,64/.test(v0.sommaire) && v1o.ouvert && v1o.fiche && v2.ouvert && v2.fiche && !v3.ouvert && !v3.fiche,
        JSON.stringify({ v0, v1o, v2, v3 }));
      await saisir('input[data-chemin="dy"]', '5');
      // ---- § 9.5 : la bascule ELS → ELU — la limite passe de 236,7 à 355 N/mm², coef et R_ELU disparaissent, le taux change ; l'ELS les rend
      const champs = () => g.evaluate(() => ({ coef: document.querySelectorAll('#donnees input[data-chemin="coef"]').length, relu: document.querySelectorAll('#donnees input[data-chemin="R_ELU"]').length,
        libelleR: (document.querySelector('#donnees [data-cible="R"] label span') || {}).textContent || '' }));
      const chELS = await champs();
      await ouvrirRub('hy');
      await g.click('#donnees [data-chemin="etat_limite"][data-valeur="ELU"]');
      await tranquille();
      await g.waitForTimeout(200);
      const limU = await limite(), chELU = await champs(), tU = await taux();
      const resU = await g.evaluate(() => document.getElementById('resultat').textContent);
      verif('cornière ELU : la limite affichée passe de 236,7 à 355 N/mm² (fy/γM0), σadm n\'apparaît plus, le taux baisse sous la même R, coef et R_ELU sont masqués, R se nomme « R (ELU) »',
        /355/.test(limU) && !/236,7/.test(limU) && !/σadm/.test(resU) && /γM0/.test(resU) && tU !== null && tU < t0 - 10 && chELS.coef === 1 && chELS.relu === 1 && chELS.libelleR === 'R (ELS)'
        && chELU.coef === 0 && chELU.relu === 0 && chELU.libelleR === 'R (ELU)', JSON.stringify({ limU, tU, t0, chELS, chELU }));
      const ancU = await g.$$eval('#resultat .et-bloc[data-theme="1"] .el .l1 b', (e) => e.map((x) => x.textContent));
      verif('cornière ELU : le bloc Ancrages n\'a que N et V à l\'ELU (R saisie à l\'ELU), puis les positions', ancU.slice(0, 3).join() === 'Traction N · ELU,Cisaillement V · ELU,Positions' && !ancU.some((t) => /ELS/.test(t)), ancU.join());
      await g.screenshot({ path: path.join(OUT, 'site_corniere_ELU_pc.png') });
      await g.click('#donnees [data-chemin="etat_limite"][data-valeur="ELS"]');
      await tranquille();
      await g.waitForTimeout(200);
      const chRetour = await champs();
      verif('cornière : le retour à l\'ELS rend coef, R_ELU et la limite 236,7', chRetour.coef === 1 && chRetour.relu === 1 && /236,7/.test(await limite()) && (await taux()) === t0, JSON.stringify(chRetour));
      // ---- § 9.3 : le support « poutre » — ses deux étendues apparaissent (150/150), les distances aux bords se lisent dans le bloc
      // Ancrages (60 et 230 mm pour Y = 85 ± 5), la coupe reste un dessin ; une étendue trop courte est refusée (toast, valeur rendue)
      await ouvrirRub('an');
      const avantP = await g.evaluate(() => document.querySelectorAll('#donnees input[data-chemin="support_haut"], #donnees input[data-chemin="support_bas"]').length);
      await g.click('#donnees [data-chemin="support"][data-valeur="poutre"]');
      await tranquille();
      await g.waitForTimeout(250);
      const poutre = await g.evaluate(() => {
        const vis = (e) => e && e.checkVisibility();
        const h = document.querySelector('#donnees input[data-chemin="support_haut"]'), b = document.querySelector('#donnees input[data-chemin="support_bas"]');
        const bords = [...document.querySelectorAll('#resultat .et-bloc[data-theme="1"] .el')].filter((x) => /Poutre · bord/.test(x.textContent)).map((x) => x.textContent.replace(/\s+/g, ' ').trim().slice(0, 90));
        const d = [...document.querySelectorAll('.view.on .draw')].find(vis); const svg = d && d.querySelector(':scope > svg'); const r = svg && svg.getBoundingClientRect();
        return { haut: vis(h) ? h.value : null, bas: vis(b) ? b.value : null, bords, svg: r ? [Math.round(r.width), Math.round(r.height), svg.querySelectorAll('text').length] : null };
      });
      verif('cornière poutre : les deux étendues (150 / 150 mm) apparaissent dans la rubrique Ancrage (absentes pour un voile), les distances aux bords (60 et 230 mm) se lisent dans le bloc Ancrages, la coupe reste un dessin à sa taille',
        avantP === 0 && poutre.haut === '150' && poutre.bas === '150' && poutre.bords.length === 2 && /60,0 mm/.test(poutre.bords[0]) && /230,0 mm/.test(poutre.bords[1])
        && poutre.svg && poutre.svg[0] > 400 && poutre.svg[2] > 10, JSON.stringify(poutre));
      await g.screenshot({ path: path.join(OUT, 'site_corniere_poutre_pc.png') });
      await g.fill('input[data-chemin="support_haut"]', '90');
      await g.press('input[data-chemin="support_haut"]', 'Enter');
      await tranquille();
      await g.waitForTimeout(250);
      const refusP = await lireToast();
      const hautApres = await g.inputValue('input[data-chemin="support_haut"]');
      verif('cornière poutre : 90 mm au-dessus est refusé (« l\'ancrage sort de la poutre par le haut »), le toast le dit, le champ rend 150', refusP.vu && refusP.erreur && /sort de la poutre/.test(refusP.texte) && hautApres === '150', JSON.stringify({ refusP, hautApres }));
      await g.click('#donnees [data-chemin="support"][data-valeur="voile"]');
      await tranquille();
      // ---- § 9.2 : un entraxe saisi se lit dans les positions (e₁ · s · z) ; vidé, il redevient L/n
      const positions = () => g.$$eval('#resultat .et-bloc[data-theme="1"] .el', (e) => (e.find((x) => /^Positions/.test(x.textContent.trim())) || {}).textContent || '');
      await saisir('input[data-chemin="entraxe"]', '80');
      const posS = await positions();
      await saisir('input[data-chemin="entraxe"]', '');
      const posA = await positions();
      verif('cornière : l\'entraxe saisi (80) se lit dans les positions — e₁ = 70 · s = 80 (saisi) · z = 70 · 150 · 230 — et l\'état « entraxe 80 » ; vidé, « s = 100 (L/n) », z = 50 · 150 · 250',
        /e₁ = 70 · s = 80 \(saisi\) · z = 70 · 150 · 230/.test(posS) && /entraxe 80/.test(posS) && /s = 100 \(L\/n\) · z = 50 · 150 · 250/.test(posA), JSON.stringify({ posS: posS.slice(0, 120), posA: posA.slice(0, 120) }));
      // ---- § 9.7 : la saisie libre montre A, B, t, r directement (rien ne les cache, pas de volet)
      await ouvrirRub('co');
      await g.click('#donnees [data-chemin="standard"][data-valeur="saisie"]');
      await tranquille();
      await g.waitForTimeout(200);
      const libre = await g.evaluate(() => ({ champs: ['A', 'B', 't', 'r'].map((k) => { const e = document.querySelector(`#donnees input[data-chemin="${k}"]`); return e && e.checkVisibility() ? e.value : null; }),
        volet: document.querySelectorAll('#donnees details.et-volet').length }));
      verif('cornière saisie libre : A, B, t, r (120, 120, 15, 13 : la cornière affichée) sont des champs visibles, sans volet', libre.champs.join() === '120,120,15,13' && libre.volet === 0, JSON.stringify(libre));
      await g.screenshot({ path: path.join(OUT, 'site_corniere_saisie_libre_pc.png') });
      await g.click('#donnees [data-chemin="standard"][data-valeur="catalogue"]');
      await tranquille();
      // le sélecteur de la ligne « Cornière » : 233 lignes par masse, ✓ la retenue, ★ l'optimale, les grises hors domaine, les rouges
      await g.click('#resultat .prof[data-pk="corniere"]');
      await g.waitForSelector('.flyout.on.picker .pk-row', { timeout: 60000 });
      await g.waitForTimeout(300);
      const pk = await g.evaluate(() => ({ n: document.querySelectorAll('.flyout.on.picker .pk-row').length, cur: (document.querySelector('.flyout.on.picker .pk-row.cur .nm') || {}).textContent,
        opt: (document.querySelector('.flyout.on.picker .pk-row.opt .nm') || {}).textContent, hors: document.querySelectorAll('.flyout.on.picker .pk-row .chip.c').length,
        rouges: document.querySelectorAll('.flyout.on.picker .pk-row .chip.r').length, vu: document.querySelector('.flyout.on.picker .pk-row.cur').checkVisibility() }));
      verif('cornière : le sélecteur ouvre les 233 cornières par masse — ✓ L 120×120×15, ★ une optimale, des grises « hors domaine », des rouges qui ne tiennent pas, la retenue à l\'écran',
        pk.n === 233 && pk.cur === 'L 120×120×15' && !!pk.opt && pk.hors > 50 && pk.rouges > 10 && pk.vu, JSON.stringify(pk));
      await g.screenshot({ path: path.join(OUT, 'site_corniere_selecteur.png') });
      await g.click('.flyout.on.picker .pk-row.opt');
      await tranquille();
      await g.waitForTimeout(200);
      // la désignation des données est désormais un MENU DÉROULANT des 233 cornières (remarque de l'ingénieur du 10/11 au soir) :
      // l'option sélectionnée porte la désignation retenue
      const menuProfil = () => g.$eval('select[data-chemin="profil"]', (s) => ({ n: s.options.length, groupes: s.querySelectorAll('optgroup').length, choisi: s.options[s.selectedIndex].textContent.trim(), valeur: s.value }));
      const mp = await menuProfil();
      verif(`cornière : choisir l'optimale (${pk.opt}) la retient — le menu des données (233 cornières, deux familles) et la ligne Cornière la disent`,
        mp.n === 233 && mp.groupes === 2 && mp.choisi === pk.opt && (await g.textContent('#resultat .prof[data-pk="corniere"] span')).trim() === pk.opt, JSON.stringify(mp));
      await ouvrirRub('co');
      await g.selectOption('select[data-chemin="profil"]', 'L 120x120x15');
      await tranquille();
      await g.waitForTimeout(200);
      // le mode équilibre : le taux change, la coupe dessine le pivot y_c et u_V (plus le jeu du classeur)
      await ouvrirRub('hy');
      await g.click('#donnees [data-chemin="mode"][data-valeur="equilibre"]');
      await tranquille();
      await g.waitForTimeout(200);
      const tEq = await taux();
      const pEq = { yc: await piece('yc'), uV: await piece('uV'), jeu: await piece('jeu_ancrage') };
      verif('cornière A équilibre : le taux change, la coupe montre le pivot y_c et la ligne u_V, plus le jeu du classeur', tEq !== null && tEq !== t0 && pEq.yc && pEq.uV && !pEq.jeu, JSON.stringify({ t0, tEq, pEq }));
      await g.screenshot({ path: path.join(OUT, 'site_corniere_A_equilibre_pc.png') });
      await g.click('#donnees [data-chemin="mode"][data-valeur="historique"]');
      await tranquille();
      // la configuration B : la coupe se retourne — MESURÉ : la charge R passe au-dessus de l'ancrage (en A elle est en dessous),
      // le contact C (pièce yc) est au bas de la cornière ; le mode est verrouillé ; deux blocs, la section nette gouverne
      await ouvrirRub('co');
      const geoA = { R: await piece('R'), Y: await piece('Y'), prof: await piece('profil') };
      await g.click('#donnees [data-chemin="configuration"][data-valeur="B"]');
      await tranquille();
      await g.waitForTimeout(250);
      const geoB = { R: await piece('R'), Y: await piece('Y'), yc: await piece('yc'), prof: await piece('profil') };
      const modeLocked = await g.$eval('#donnees [data-chemin="mode"]', (x) => x.disabled);
      verif('cornière B : à l\'écran, la charge R passe au-dessus de l\'ancrage (A : au-dessous) et le contact C est au bas de la cornière (mesuré dans le SVG) ; le mode est verrouillé',
        geoA.R && geoA.Y && geoA.R.y > geoA.Y.y && geoB.R && geoB.Y && geoB.R.y < geoB.Y.y && geoB.yc && geoB.prof && geoB.yc.y > geoB.Y.y
        && geoB.yc.y > geoB.prof.y + geoB.prof.h * 0.3 && modeLocked, JSON.stringify({ geoA, geoB, modeLocked }));
      const tB = await taux(), gB = await gouv(), blocsB = await blocsVus();
      verif('cornière B : deux blocs (pas de comparaison), la section nette gouverne, non conforme (η > 100 %) sur la cornière neuve', blocsB.join() === 'Sections,Ancrages' && /Section nette/.test(gB) && tB > 100, JSON.stringify({ tB, gB, blocsB }));
      await g.screenshot({ path: path.join(OUT, 'site_corniere_B_pc.png') });
      await g.click('#donnees [data-chemin="configuration"][data-valeur="A"]');
      await tranquille();
      // une inégale choisie dans le menu, après avoir descendu l'ancrage : retenue, les segments de l'aile disent les longueurs
      await ouvrirRub('an');
      await saisir('input[data-chemin="Y"]', '65');
      await ouvrirRub('co');
      await g.selectOption('select[data-chemin="profil"]', 'L 150x100x12');
      await tranquille();
      await g.waitForTimeout(200);
      const des = (await menuProfil()).choisi;
      const ailes = await g.$$eval('#donnees [data-chemin="aile_horizontale"]', (x) => x.map((e) => e.textContent.trim()));
      verif('cornière inégale : « L 150×100×12 » choisie dans le menu (famille Ailes inégales), l\'aile horizontale en segments avec les longueurs réelles (A = 150 mm, B = 100 mm)',
        des === 'L 150×100×12' && ailes.join() === 'A = 150 mm,B = 100 mm', JSON.stringify({ des, ailes }));
      await g.screenshot({ path: path.join(OUT, 'site_corniere_inegale_pc.png') });
      // une note du module d'avant (version 1, modèle lame) semée dans ce navigateur : elle se rouvre, l'avis est dit une fois
      await g.evaluate(() => { const k = 'stab.notes.v1', notes = JSON.parse(localStorage.getItem(k) || '{}'), t = Date.now();
        notes['ancienne-corniere'] = { id: 'ancienne-corniere', module: 'corniere', nom: 'Ancienne note cornières', chantier: 'Réalex', dossier: '', indice: 'A', cree: t, modifie: t,
          etat: { version: 1, actif: 0, cornieres: [{ nom: 'Cornière de rive', standard: true, profil: 'L 100x100x10', charge: 'lineaire', V: 5, L: 1, e: 2.3, gamma_ELU: 1.35, s: 60, bord: 5, sig_lim_ELS: 156.7, tau_lim_ELS: 0.8, sig_lim_ELU: 235 }] },
          resume: null, vignette: '' };
        localStorage.setItem(k, JSON.stringify(notes)); });
      await pretG({ apres: await ouvrir(g, `${BASE}/corniere?note=ancienne-corniere`) });
      await tranquille();
      await g.waitForTimeout(200);
      const v1 = await g.evaluate(() => ({ avis: [...document.querySelectorAll('#resultat .msg')].map((m) => m.textContent).filter((t) => /modèle lame/.test(t)).length,
        nom: (document.querySelector('.panel.left .phead .et-elt input.et-elt-nom') || {}).value,
        profil: ((s) => s && s.options[s.selectedIndex] ? s.options[s.selectedIndex].textContent.trim() : null)(document.querySelector('select[data-chemin="profil"]')) }));
      v1.erreur = !!(await echecEcran(g));
      verif('cornière : une note du module d\'avant (modèle lame) se rouvre — le nom et la cornière gardés, l\'avis « modèle lame » dit une fois dans le résultat',
        !v1.erreur && v1.avis === 1 && v1.nom === 'Cornière de rive' && v1.profil === 'L 100×100×10', JSON.stringify(v1));
      await g.screenshot({ path: path.join(OUT, 'site_corniere_v1.png') });
      // ressaisir la charge fait tomber l'avertissement
      await ouvrirRub('ch');
      await saisir('input[data-chemin="R"]', '20');
      const encore = await g.evaluate(() => [...document.querySelectorAll('#resultat .msg')].map((m) => m.textContent).filter((t) => /modèle lame/.test(t)).length);
      verif('cornière : la charge ressaisie, l\'avertissement « modèle lame » disparaît', encore === 0, `${encore}`);
      verif('cornière : aucune erreur JavaScript', !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
    } catch (e) { verif('cornière : le parcours va au bout', false, String(e.message).split('\n')[0]); }
    await c.close();
    // les trois formats, A, B et A en équilibre : rien ne déborde, aucun nom coupé (les titres des blocs du résultat aussi : un
    // état d'en-tête trop long les réduit à rien), les cibles à 44 px au doigt (tablette, téléphone), les champs en 16 px au
    // téléphone ; une capture de chaque (au téléphone : les trois onglets)
    for (const [fmt, w, h, tel] of [['pc', 1440, 900, false], ['tab1024', 1024, 768, false], ['iphone13', 390, 844, true]]) {
      const cx = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel });
      const pg = await cx.newPage();
      pg.erreurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') pg.erreurs.push(m.text().slice(0, 300)); });
      const pretP = (o) => pretEcran(pg, o);
      const calmeP = () => pg.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
      const etat = () => pg.evaluate((tel) => {
        const vis = (e) => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const large = (e) => { const r = document.createRange(); r.selectNodeContents(e); return r.getBoundingClientRect().width; };
        const coupes = [...document.querySelectorAll('.prof > span:not(.leger), .el .l1 b, #resultat .et-bloc-t .n, .row > label > span:not(.sym), .row2 > label > span, .row2.et-cibles > .et-piece > label > span, .rows .seg.full button, .mtabs button')]
          .filter(vis).filter((e) => e.scrollWidth > e.clientWidth + 1 || large(e) > e.getBoundingClientRect().width + 0.01).map((e) => e.textContent.trim().slice(0, 40));
        const petites = [];
        if (tel || innerWidth < 1200) {
          document.querySelectorAll('#donnees .sec:not(.open) > .sechd').forEach((x) => x.click());
          for (const e of document.querySelectorAll('.top button, .top .btn, .mtop button, .mtabs button, .ctool button, .phead button, #donnees button, #resultat button, #resultat .el, .mbar .btn')) {
            if (!vis(e)) continue;
            const r = e.getBoundingClientRect();
            if (r.width >= 1 && Math.min(r.width, r.height) < 43.5) petites.push(`${(e.className || e.tagName).toString().split(' ')[0]} « ${(e.textContent || e.title || '').trim().slice(0, 16)} » ${Math.round(r.width)}×${Math.round(r.height)}`);
          }
        }
        const petits = tel ? [...document.querySelectorAll('#donnees input:not([type=checkbox]), #donnees select, .panel.left .phead input')].filter(vis)
          .filter((e) => parseFloat(getComputedStyle(e).fontSize) < 16).map((e) => `${e.dataset.chemin} ${getComputedStyle(e).fontSize}`) : [];
        return { deborde: document.documentElement.scrollWidth > innerWidth, coupes, petites, petits };
      }, tel);
      // (révision § 9 : l'ELU — coef et R_ELU masqués, fy/γM0 partout — et la poutre — deux champs de plus, deux informations de plus)
      for (const cfg of ['A', 'B', 'A-equilibre', 'ELU', 'poutre']) {
        try {
          await pretP({ apres: await ouvrir(pg, `${BASE}/corniere?${new URLSearchParams({ nom: `Cornière ${cfg}`, chantier: 'Formats' })}`) });
          await calmeP();
          if (cfg === 'B') {
            if (tel) await pg.click('.mtabs [data-onglet="donnees"]');
            if (!(await pg.$eval('#donnees .sec[data-sec="co"]', (x) => x.classList.contains('open')))) await pg.click('#donnees .sec[data-sec="co"] > .sechd');
            await pg.click('#donnees [data-chemin="configuration"][data-valeur="B"]');
            await calmeP();
            await pg.waitForTimeout(200);
          } else if (cfg === 'ELU') {
            if (tel) await pg.click('.mtabs [data-onglet="donnees"]');
            if (!(await pg.$eval('#donnees .sec[data-sec="hy"]', (x) => x.classList.contains('open')))) await pg.click('#donnees .sec[data-sec="hy"] > .sechd');
            await pg.click('#donnees [data-chemin="etat_limite"][data-valeur="ELU"]');
            await calmeP();
            await pg.waitForTimeout(200);
          } else if (cfg === 'poutre') {
            if (tel) await pg.click('.mtabs [data-onglet="donnees"]');
            if (!(await pg.$eval('#donnees .sec[data-sec="an"]', (x) => x.classList.contains('open')))) await pg.click('#donnees .sec[data-sec="an"] > .sechd');
            await pg.click('#donnees [data-chemin="support"][data-valeur="poutre"]');
            await calmeP();
            await pg.waitForTimeout(200);
          } else if (cfg === 'A-equilibre') {
            // le mode équilibre : le bloc Comparaison nomme alors l'AUTRE mode, le plus long des deux
            if (tel) await pg.click('.mtabs [data-onglet="donnees"]');
            if (!(await pg.$eval('#donnees .sec[data-sec="hy"]', (x) => x.classList.contains('open')))) await pg.click('#donnees .sec[data-sec="hy"] > .sechd');
            await pg.click('#donnees [data-chemin="mode"][data-valeur="equilibre"]');
            await calmeP();
            await pg.waitForTimeout(200);
          }
          const onglets = tel ? ['donnees', 'dessin', 'resultat'] : [''];
          for (const o of onglets) {
            if (o) { await pg.click(`.mtabs [data-onglet="${o}"]`); await pg.waitForTimeout(250); }
            const m = await etat();
            verif(`cornière ${cfg} · ${fmt}${o ? ' · ' + o : ''} : rien ne déborde, aucun nom coupé${tel || w < 1200 ? ', chaque bouton vu à 44 px au doigt' : ''}${tel ? ', chaque champ en 16 px' : ''}`,
              !m.deborde && !m.coupes.length && !m.petites.length && !m.petits.length, JSON.stringify(m).slice(0, 400));
            await pg.screenshot({ path: path.join(OUT, `site_corniere_${cfg}_${fmt}${o ? '_' + o : ''}.png`) });
          }
          if (tel) {
            // la coupe se voit au téléphone : la toile offre 300 px de large au dessin, son texte se lit
            await pg.click('.mtabs [data-onglet="dessin"]');
            await pg.waitForTimeout(250);
            const tl = await pg.evaluate(() => { const d = [...document.querySelectorAll('.view.on .draw')].find((e) => e.checkVisibility()); const svg = d && d.querySelector(':scope > svg'); if (!svg) return null;
              const t = [...svg.querySelectorAll('text')].map((x) => x.getBoundingClientRect().height).filter((x) => x > 0).sort((a, b) => a - b);
              return { l: Math.round(d.getBoundingClientRect().width), texte: t.length ? Math.round(t[Math.floor(t.length / 2)] * 10) / 10 : null }; });
            verif(`cornière ${cfg} · iphone13 : la coupe a 300 px de large au moins et un texte lisible (≥ 6 px)`, tl && tl.l >= 300 && tl.texte >= 6, JSON.stringify(tl));
          }
        } catch (e) { verif(`cornière ${cfg} · ${fmt} : le parcours va au bout`, false, String(e.message).split('\n')[0]); }
      }
      verif(`cornière · ${fmt} : aucune erreur JavaScript`, !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
      await cx.close();
    }
  }

  // ---------------------------------------------------------------- l'établi compact des DALLES, et l'ancien module Prédalles
  // La COLONNE DE GAUCHE COMPACTE (docs/site/DALLES_FUSION.md § 1, revue avec l'ingénieur le 09/11/2026) : ni titre
  // « Données », ni « DALLES (1) », ni arborescence — la LIGNE de la dalle ouverte dans l'EN-TÊTE du panneau, à côté
  // d'« Annuler » et « Rétablir » : son NOM en champ (un clic, et on le renomme ; Entrée valide, Échap rend l'ancien nom,
  // vide refusé — la liste, la tête des résultats, la confirmation de « Supprimer » et la note enregistrée suivent), la
  // LISTE des dalles (un menu : l'ouverte cochée, « Ajouter une dalle » ; le compte dès deux), COPIER, SUPPRIMER
  // (confirmé, éteint quand elle est seule) ; les rubriques tout en haut des données ; le type « Prédalle + dalle CSP »
  // fait apparaître h_p et le complément coulé (§ 2.2).
  // Les fonctions aux places de la poutre (REFONTE_ETABLI.md) : un bloc par section (« Données incomplètes » sans
  // effort), un clic sur une couche du DESSIN qui la déplie dans les données, la section courante que le dessin et les
  // résultats suivent (la vue garde son genre : du plan au plan), « ↺ » qui rend un axe saisi à l'automatique. Au
  // téléphone, la ligne est en tête de l'onglet Données, ses boutons au doigt ; un appui sur le nom y pose le curseur.
  // L'ancien module PRÉDALLES (§ 2.4) : sa page REDIRIGE vers celle des dalles en gardant l'adresse ; une ancienne note en
  // mémoire s'y ouvre en mode prédalle (le moteur des dalles reçoit « dalle » et l'état {predalles} tel quel), devient
  // une note « dalle », rien de perdu ; la carte de l'accueil vise déjà la page des dalles.
  //   filtre « plaques » : les deux ; « module:dalle » et « module:predalle » rejouent le leur (après le contrôle commun).
  const PLAQUES = ['dalle', 'predalle'].filter((k) => !FILTRE || FILTRE === 'plaques' || FILTRE === `module:${k}`);
  for (const cle of PLAQUES) {
    const c = await b.newContext({ viewport: { width: 1440, height: 900 } });
    // ce que la page envoie au moteur (module, action, forme de l'état) : lu au passage, sans rien changer
    await c.addInitScript(() => {
      const W = window.Worker;
      window.Worker = class extends W {
        postMessage(m) {
          try { const q = JSON.parse(m.requete); (window.__envois = window.__envois || []).push({ module: q.module, action: q.action, predalles: !!(q.etat && q.etat.predalles), dalles: !!(q.etat && q.etat.dalles) }); } catch (e) { /* pas une requête */ }
          super.postMessage(m);
        }
      };
    });
    const g = await c.newPage();
    g.erreurs = [];
    g.on('pageerror', (e) => g.erreurs.push(e.message));
    g.on('console', (m) => { if (m.type() === 'error') g.erreurs.push(m.text().slice(0, 300)); });
    const pretG = (o) => pretEcran(g, o);
    const tranquille = () => g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
    const ouvrirRub = async (k) => {
      if (!(await g.$eval(`#donnees .sec[data-sec="${k}"]`, (x) => x.classList.contains('open')))) await g.click(`#donnees .sec[data-sec="${k}"] > .sechd`);
      await g.waitForTimeout(80);
    };
    const saisir = async (s, v) => {
      await g.fill(s, v);
      await g.press(s, 'Enter');
      await tranquille();
      await g.waitForTimeout(150);
    };
    const sync = () => g.evaluate(() => ({ vue: document.querySelector('#vues .on')?.dataset.vue,
      blocs: [...document.querySelectorAll('#resultat .et-bloc')].filter((x) => !x.hidden).map((x) => x.dataset.section),
      arm: document.querySelector('#donnees .et-arm.on')?.dataset.section }));
    // la LIGNE de l'élément (en-tête du panneau de gauche) : la liste des dalles (les noms, l'ouverte cochée d'une étoile),
    // le nom en champ, son état (nombre, compte du bouton, supprimer allumé ?) ; le titre des résultats ; la note en mémoire
    const LIGNE = '.panel.left .phead .et-elt .et-elsel', NOM = `${LIGNE} input.et-elt-nom`;
    const liste = () => g.$$eval(`${LIGNE} .et-elt-liste .et-mn-l [data-activer]`,
      (x) => x.map((b) => b.querySelector('.nom').textContent + (b.classList.contains('on') && b.getAttribute('aria-checked') === 'true' ? '*' : '')).join());
    const nomElt = () => g.$eval(NOM, (x) => x.value);
    const titreRp = () => g.$eval('#resultat .et-rp b', (x) => x.textContent);
    const etatLigne = () => g.evaluate((s) => { const l = document.querySelector(s);
      return { n: l.dataset.elements, compte: (l.querySelector('.et-elt-liste .et-mn-b .n') || {}).textContent || '', supprimer: !l.querySelector('[data-supprimer]').disabled,
        confirmer: l.querySelector('[data-supprimer]').dataset.confirmer }; }, LIGNE);
    const lireToast = () => g.evaluate(() => { const t = document.getElementById('toast'); return { vu: t.checkVisibility(), texte: t.textContent, erreur: t.classList.contains('erreur') }; });
    const noteMemoire = () => g.evaluate(() => { const id = new URLSearchParams(location.search).get('note'); const n = JSON.parse(localStorage.getItem('stab.notes.v1') || '{}')[id];
      return n ? { module: n.module, dalles: n.etat && n.etat.dalles ? n.etat.dalles.map((d) => `${d.nom}:${d.type}${d.h_pre !== undefined ? ':' + d.h_pre : ''}`).join() : JSON.stringify(n.etat).slice(0, 80) } : null; });
    try {
      if (cle === 'predalle') {
        // une ancienne note du module Prédalles, semée dans ce navigateur (son état : celui de l'ancien module, {predalles})
        await ouvrir(g, `${BASE}/index.html`);
        await g.evaluate(() => { const k = 'stab.notes.v1', notes = JSON.parse(localStorage.getItem(k) || '{}'), t = Date.now() + 60000;
          notes['ancienne-predalle'] = { id: 'ancienne-predalle', module: 'predalle', nom: 'Ancienne note prédalles', chantier: 'Réalex', dossier: '', indice: 'A', cree: t, modifie: t,
            etat: { version: 1, predalles: [{ nom: 'Ancienne P1', h_pre: 6 }], actif: 0 }, resume: null, vignette: '' };
          localStorage.setItem(k, JSON.stringify(notes)); });
        await g.reload();
        await g.waitForTimeout(300);
        const carte = await g.$eval('#recentes .rcard', (a) => a.getAttribute('href'));
        verif('plaques predalle : à l\'accueil, la carte d\'une ancienne note « Prédalles » vise la page des dalles (js/notes.js, table des alias « vers » de js/modules.js)',
          carte === 'dalle?note=ancienne-predalle', carte);
        await pretG({ apres: await ouvrir(g, `${BASE}/predalle?note=ancienne-predalle`) });
        await tranquille();
        const u = new URL(g.url());
        const ouverte = await g.evaluate((echec) => {
          const on = document.querySelector('#donnees button.on[data-chemin="type"]'), h = document.querySelector('#donnees input[data-chemin="h_pre"]');
          const avis = document.querySelector('#resultat .pbody > .et-rp + .msg');
          return { avis: avis && avis.checkVisibility({ visibilityProperty: true, opacityProperty: true }) ? avis.textContent : '',
            module: document.body.dataset.module, nomNote: document.getElementById('nom-note').value,
            nom: (document.querySelector('.panel.left .phead .et-elt input.et-elt-nom') || {}).value, titre: (document.querySelector('#resultat .et-rp b') || {}).textContent,
            type: on && on.dataset.valeur, hPre: h && h.value, erreur: echec, envois: (window.__envois || []).slice(0, 2) };
        }, !!(await echecEcran(g)));
        const mem = await noteMemoire();
        verif('plaques predalle : « predalle?note=… » redirige vers « dalle?note=… » (l\'adresse gardée) ; le moteur reçoit « dalle » et l\'état {predalles} tel quel',
          u.pathname.endsWith('/dalle') && u.searchParams.get('note') === 'ancienne-predalle' && ouverte.module === 'dalle' && !ouverte.erreur
          && ouverte.envois.length >= 1 && ouverte.envois[0].module === 'dalle' && ouverte.envois[0].action === 'calculer' && ouverte.envois[0].predalles && !ouverte.envois[0].dalles,
          JSON.stringify({ url: `${u.pathname}${u.search}`, ...ouverte }));
        verif('plaques predalle : l\'ancienne note s\'ouvre en PRÉDALLE (type prédalle, h_p = 6 cm, « Ancienne P1 » dans la ligne de l\'élément et en tête des résultats, son nom de note) et devient une note « dalle » en mémoire — rien de perdu',
          ouverte.nomNote === 'Ancienne note prédalles' && ouverte.nom === 'Ancienne P1' && ouverte.titre === 'Ancienne P1' && ouverte.type === 'predalle' && ouverte.hPre === '6'
          && mem && mem.module === 'dalle' && mem.dalles === 'Ancienne P1:predalle:6', JSON.stringify({ ouverte, mem }));
        // ses nappes supérieures sont désormais posées l'une sur l'autre (décision du 09/11/2026) : l'avis le dit, une fois
        verif('plaques predalle : l\'ancienne note dit, en tête du résultat, « Note d\'avant le 09/11/2026 : les nappes sont désormais posées l\'une sur l\'autre … » (visible)',
          /^Note d'avant le 09\/11\/2026 : les nappes sont désormais posées l'une sur l'autre/.test(ouverte.avis), ouverte.avis);
        await g.screenshot({ path: path.join(OUT, 'site_plaques_predalle_relue.png') });
        // une note NEUVE demandée à l'ancienne adresse (un vieux lien) : le nom et le chantier arrivent sur les dalles,
        // qui créent la note (l'adresse devient « ?note=… », comme pour toute note demandée par « ?nom= »)
        await pretG({ apres: await ouvrir(g, `${BASE}/predalle?nom=Essai%20pr%C3%A9dalle&chantier=R%C3%A9alex`) });
        const u2 = new URL(g.url());
        const neuve = await g.evaluate(() => { const id = new URLSearchParams(location.search).get('note'); const n = JSON.parse(localStorage.getItem('stab.notes.v1') || '{}')[id];
          return { nom: document.getElementById('nom-note').value, fil: document.getElementById('fil-chantier').textContent, module: n && n.module, enregistree: n && n.nom }; });
        verif('plaques predalle : « predalle?nom=…&chantier=… » arrive sur la page des dalles, le nom et le chantier repris, la note créée en « dalle »',
          u2.pathname.endsWith('/dalle') && !!u2.searchParams.get('note') && neuve.nom === 'Essai prédalle' && neuve.fil.includes('Réalex')
          && neuve.module === 'dalle' && neuve.enregistree === 'Essai prédalle', JSON.stringify({ url: `${u2.pathname}${u2.search}`, ...neuve }));
        // (une adresse sans note ni nom : la page des dalles, telle quelle)
        await pretG({ apres: await ouvrir(g, `${BASE}/predalle`) });
        verif('plaques predalle : « predalle » seul mène à « dalle »', /\/dalle$/.test(new URL(g.url()).pathname), g.url());
      } else {
        const nom = 'Dalle 1', dims = '100 × 20 cm';
        await ouvrir(g, `${BASE}/index.html`);
        const m0 = await montage(g);
        await Promise.all([g.waitForNavigation(), g.click(`.tile[data-module="${cle}"]`)]);
        await pretG({ apres: m0 });
        // 1. la colonne de gauche compacte : ni titre, ni arborescence ; la LIGNE de la dalle dans l'en-tête du panneau
        const colonne = await g.evaluate(() => {
          const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const phead = document.querySelector('.panel.left .phead'), don = document.getElementById('donnees'), arbre = don.querySelector('.et-arbre');
          const premier = don.querySelector('.sec[data-sec]'), l = phead.querySelector(':scope > .et-elt > .et-elsel'), n = l && l.querySelector('input.et-elt-nom');
          const rp = document.querySelector('#resultat .et-rp');
          return { titre: !!phead.querySelector('.h3'), hist: vis(phead.querySelector('[data-historique="annuler"]')),
            enTete: !!document.querySelector('#donnees .et-arbre-h'), lignes: document.querySelectorAll('#donnees .et-pt').length,
            compacte: !!arbre && arbre === don.firstElementChild && arbre.matches('.et-arbre-c[data-dalle="0"]'), selecteur: !!document.querySelector('.et-elsel .seg'),
            elselDonnees: !!don.querySelector('.et-elsel'), ligne: !!l && l.dataset.elements === '1' && vis(l.querySelector('.et-elt-liste [data-menu]'))
              && vis(l.querySelector('button.et-elt-b[data-dupliquer-index="0"]')) && !!l.querySelector('button.et-elt-b.danger[data-supprimer="0"]:disabled'),
            nomGauche: don.querySelectorAll('[data-chemin="nom"]').length, nomResultat: !!(rp && rp.querySelector('input, .et-rp-nom')),
            enHaut: !!premier && Math.abs(premier.getBoundingClientRect().top - don.getBoundingClientRect().top) < 2,
            rubs: [...document.querySelectorAll('#donnees .et-arbre .sec[data-sec]')].map((y) => y.dataset.sec + (y.classList.contains('open') ? '*' : '')).join(),
            nom: n ? n.value : null, champ: !!n && n.dataset.chemin === 'nom' && n.dataset.sorte === 'texte' && vis(n),
            titreRp: rp ? (rp.querySelector('b') || {}).textContent : null, dims: rp ? rp.querySelector('span').textContent : null,
            pastille: !!document.querySelector('#donnees .et-arbre .chip, #donnees .et-arbre .pct, #donnees .et-arbre .status, .phead .et-elsel .chip, .phead .et-elsel .pct') };
        });
        verif(`plaques ${cle} : la colonne de gauche compacte — ni « Données », ni « DALLES (1) », ni arborescence, ni champ « nom » dans les données ; la LIGNE de « ${nom} » dans l'en-tête du panneau (la liste, copier ; supprimer éteint), à côté d'Annuler ; les rubriques tout en haut`,
          !colonne.titre && colonne.hist && !colonne.enTete && colonne.lignes === 0 && colonne.compacte && !colonne.selecteur && !colonne.elselDonnees && colonne.ligne
          && colonne.nomGauche === 0 && colonne.enHaut && !colonne.pastille, JSON.stringify(colonne));
        verif(`plaques ${cle} : le NOM « ${nom} » est un champ dans la ligne de l'élément ; la tête des résultats le LIT (un titre, plus un champ), « ${dims} · … » dessous ; quatre rubriques (dl, so, arm, op), la première seule ouverte`,
          colonne.nom === nom && colonne.champ && colonne.titreRp === nom && !colonne.nomResultat && (colonne.dims || '').startsWith(`${dims} · `) && colonne.rubs === 'dl*,so,arm,op',
          JSON.stringify(colonne));
        const lg0 = await ligneElement(g);
        verif(`plaques ${cle} · PC 1440 : la ligne de la dalle tient dans l'en-tête (une rangée, rien ne déborde ni ne se chevauche, le nom entier)`, lg0 && !lg0.fautes.length && lg0.nomEntier, JSON.stringify(lg0));
        await g.screenshot({ path: path.join(OUT, `site_plaques_${cle}_colonne.png`) });
        // 2. la LISTE (une seule dalle : elle, cochée, et « Ajouter une dalle ») ; « Copier » : une seconde dalle, ouverte —
        // la liste en montre deux, « 2 » sur son bouton, supprimer s'allume
        const mn = await listeElement(g);
        await g.keyboard.press('Escape');
        await g.waitForTimeout(100);
        const mnFerme = await g.$eval(`${LIGNE} .et-elt-liste .et-mn-l`, (x) => x.hidden);
        await g.click(`${LIGNE} [data-dupliquer-index="0"]`);
        await tranquille();
        await g.waitForTimeout(200);
        const s1 = { liste: await liste(), nom: await nomElt(), rp: await titreRp(), ...(await etatLigne()) };
        const mn2 = await listeElement(g);
        await g.screenshot({ path: path.join(OUT, `site_${cle}_ligne_element.png`) });
        await g.keyboard.press('Escape');
        await g.waitForTimeout(100);
        verif(`plaques ${cle} : la liste s'ouvre dans l'écran (« ${nom} » cochée, « Ajouter une dalle », pas de compte), Échap la referme ; « Copier » ajoute une seconde dalle et l'ouvre — la liste en montre deux (la copie cochée), « 2 » sur son bouton, supprimer s'allume, la tête des résultats suit`,
          mn.vu && mn.ouvert && mn.dedans && mn.touche && mn.items === `${nom}*` && mn.ajouter === 'Ajouter une dalle' && mn.compte === '' && mnFerme
          && s1.liste === `${nom},${nom} (copie)*` && s1.nom === `${nom} (copie)` && s1.rp === `${nom} (copie)` && s1.n === '2' && s1.compte === '2' && s1.supprimer
          && mn2.vu && mn2.dedans && mn2.touche && mn2.items === `${nom},${nom} (copie)*` && mn2.compte === '2', JSON.stringify({ mn, mnFerme, s1, mn2 }));
        // 3. RENOMMER d'un clic dans la ligne de l'élément : la liste, la tête des résultats, la confirmation de « Supprimer »
        // et la note enregistrée (.stab) suivent ; Échap rend l'ancien nom ; un nom vide est refusé
        await g.click(NOM);
        const foc = await g.evaluate((s) => !!document.activeElement && document.activeElement.matches(s), NOM);
        await g.screenshot({ path: path.join(OUT, `site_plaques_${cle}_selecteur.png`) });
        await g.keyboard.press('Control+a');
        await g.keyboard.type('D2');
        await g.keyboard.press('Enter');
        await tranquille();
        await g.waitForTimeout(150);
        const apresNom = { foc, liste: await liste(), nom: await nomElt(), rp: await titreRp(), confirmer: (await etatLigne()).confirmer, mem: await noteMemoire() };
        verif(`plaques ${cle} : un clic sur le nom (ligne de l'élément) y pose le curseur ; « D2 » + Entrée — la liste, la tête des résultats, la confirmation de « Supprimer » et la note enregistrée (.stab) suivent`,
          apresNom.foc && apresNom.liste === 'Dalle 1,D2*' && apresNom.nom === 'D2' && apresNom.rp === 'D2' && apresNom.confirmer.includes('« D2 »') && apresNom.mem
          && apresNom.mem.module === 'dalle' && apresNom.mem.dalles === 'Dalle 1:coulee,D2:coulee', JSON.stringify(apresNom));
        await g.click(NOM);
        await g.keyboard.press('Control+a');
        await g.keyboard.type('Brouillon');
        await g.keyboard.press('Escape');
        await g.waitForTimeout(250);
        const echap = { nom: await nomElt(), rp: await titreRp(), mem: await noteMemoire() };
        await g.click(NOM);
        await g.keyboard.press('Control+a');
        await g.keyboard.press('Backspace');
        await g.keyboard.press('Enter');
        await g.waitForTimeout(300);
        const vide = { nom: await nomElt(), rp: await titreRp(), mem: await noteMemoire(), toast: await lireToast() };
        verif(`plaques ${cle} : Échap rend l'ancien nom (« D2 », rien n'est écrit) ; un nom vide est refusé — « Un nom ne peut pas être vide », l'ancien nom revient, la note le garde`,
          echap.nom === 'D2' && echap.rp === 'D2' && echap.mem.dalles === 'Dalle 1:coulee,D2:coulee' && vide.nom === 'D2' && vide.rp === 'D2' && vide.mem.dalles === 'Dalle 1:coulee,D2:coulee'
          && vide.toast.vu && vide.toast.erreur && /Un nom ne peut pas être vide/.test(vide.toast.texte), JSON.stringify({ echap, vide }));
        // 4. la liste ouvre l'autre dalle ; « Supprimer » (confirmé) retire l'ouverte — une dalle reste, supprimer s'éteint
        await g.click(`${LISTE_ELT} [data-menu]`);
        await g.click(`${LIGNE} .et-elt-liste .et-mn-l [data-activer="0"]`);
        await tranquille();
        await g.waitForTimeout(200);
        const ouverte0 = { liste: await liste(), nom: await nomElt(), rp: await titreRp() };
        let question = '';
        g.once('dialog', (d) => { question = d.message(); d.accept(); });
        await g.click(`${LIGNE} [data-supprimer="0"]`);
        await tranquille();
        await g.waitForTimeout(200);
        const reste = { ...(await etatLigne()), liste: await liste(), nom: await nomElt(), rp: await titreRp(), mem: await noteMemoire(), question };
        verif(`plaques ${cle} : la liste ouvre l'autre dalle (son nom dans la ligne et en tête des résultats) ; « Supprimer » (confirmé : « Supprimer la dalle « Dalle 1 » de la note ? ») retire l'ouverte — « D2 » reste, seule : supprimer s'éteint, le compte disparaît`,
          ouverte0.liste === 'Dalle 1*,D2' && ouverte0.nom === 'Dalle 1' && ouverte0.rp === 'Dalle 1' && reste.question === 'Supprimer la dalle « Dalle 1 » de la note ?'
          && reste.n === '1' && !reste.supprimer && reste.compte === '' && reste.liste === 'D2*' && reste.nom === 'D2' && reste.rp === 'D2' && reste.mem && reste.mem.dalles === 'D2:coulee',
          JSON.stringify({ ouverte0, reste }));
        // 5. les fonctions aux places de la poutre
        const e0 = await g.$eval('#resultat .et-bloc', (x) => ({ e: x.querySelector('.et-bloc-h .e').textContent, inc: x.classList.contains('inc') }));
        verif(`plaques ${cle} : sans effort, le bloc de la section dit « Données incomplètes » (gris), jamais « Conforme »`, e0.e === 'Données incomplètes' && e0.inc, JSON.stringify(e0));
        // un moment saisi dans la carte de la section : le bloc se colore, la bande de l'en-tête le porte (une fois)
        await ouvrirRub('so');
        await saisir('#donnees input[data-chemin="sections/0/My_inf"]', '20');
        const e1 = await g.evaluate(() => ({ e: document.querySelector('#resultat .et-bloc .et-bloc-h .e').textContent,
          ok: document.querySelector('#resultat .et-bloc').classList.contains('ok'),
          bande: [...document.querySelectorAll('#stage .view.on .et-eff span')].map((x) => x.textContent.replace(/\s+/g, '')) }));
        verif(`plaques ${cle} : M_y,inf = 20 saisi — le bloc dit « Conforme » (vert), la bande de l'en-tête porte M_y,inf 20,0 kN·m`,
          e1.e === 'Conforme' && e1.ok && e1.bande.includes('My,inf20,0kN·m'), JSON.stringify(e1));
        // le DESSIN → les données : un clic sur la couche de base de la nappe inférieure Y
        await g.keyboard.press('Escape');
        const k0 = 's0.inf_y-0';
        const pt = await g.$eval(`#stage .view.on [data-sel="${k0}"]`, (x) => {
          const q = (x.querySelector('circle, line, path, rect') || x).getBoundingClientRect();
          return { x: q.left + q.width / 2, y: q.top + q.height / 2 };
        });
        await g.mouse.click(pt.x, pt.y);
        await g.waitForTimeout(300);
        const vu = await g.evaluate((k) => {
          const v = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const p = document.querySelector(`#donnees [data-cible="${k}"]`);
          return { arm: document.querySelector('#donnees .sec[data-sec="arm"]').classList.contains('open'), cible: !!p && p.classList.contains('et-cible'),
            det: v(p && p.querySelector('.et-det')), sel: [...new Set([...document.querySelectorAll('#stage .view.on [data-sel].et-sel')].map((x) => x.dataset.sel))] };
        }, k0);
        verif(`plaques ${cle} : un clic sur une couche du dessin ouvre « Armatures » et y déplie SA ligne (axe, section au mètre) ; la pièce s'éclaire`,
          vu.arm && vu.cible && vu.det && vu.sel.join() === k0, JSON.stringify(vu));
        await g.screenshot({ path: path.join(OUT, `site_plaques_${cle}_couche.png`) });
        // « ↺ » : l'axe saisi revient à l'automatique d'un clic
        const axe = 'sections/0/faces/inf_y/couches/0/dist';
        await saisir(`#donnees input[data-chemin="${axe}"]`, '5');
        const raz = await g.$(`#donnees .et-raz[data-chemin="${axe}"]`);
        const razVu = !!raz && await raz.evaluate((x) => x.checkVisibility({ visibilityProperty: true, opacityProperty: true }));
        if (raz) { await raz.click(); await tranquille(); await g.waitForTimeout(150); }
        const apres = await g.evaluate((a) => ({ v: document.querySelector(`#donnees input[data-chemin="${a}"]`)?.value,
          ph: document.querySelector(`#donnees input[data-chemin="${a}"]`)?.placeholder, raz: !!document.querySelector(`#donnees .et-raz[data-chemin="${a}"]`) }), axe);
        verif(`plaques ${cle} : un axe saisi (5 cm) montre « ↺ » ; un clic le rend à l'automatique (champ vide, « auto · … »)`,
          razVu && apres.v === '' && /^auto · /.test(apres.ph || '') && !apres.raz, JSON.stringify({ razVu, apres }));
        // la section courante : une seconde section ; du PLAN de A au plan de B, le bloc de B seul à droite
        await ouvrirRub('so');
        await g.click('#donnees .et-add[data-chemin="sections/+"]');
        await tranquille();
        await g.waitForTimeout(200);
        await g.click('#vues [data-vue="p0"]');
        await g.waitForTimeout(150);
        await ouvrirRub('arm');
        await g.click('#donnees .et-secsel [data-section-courante="1"]');
        await g.waitForTimeout(250);
        const sB = await sync();
        await g.click('#donnees .et-secsel [data-section-courante="0"]');
        await g.waitForTimeout(250);
        const sA = await sync();
        verif(`plaques ${cle} : la section B choisie depuis le plan de A : le plan de B, son bloc seul, ses armatures ; puis retour au plan de A`,
          sB.vue === 'p1' && sB.blocs.join() === '1' && sB.arm === '1' && sA.vue === 'p0' && sA.blocs.join() === '0' && sA.arm === '0', JSON.stringify({ sB, sA }));
        await g.click('#vues [data-vue="toutes"]');
        await g.waitForTimeout(200);
        const st = await sync();
        verif(`plaques ${cle} : la vue « Toutes » montre les blocs des deux sections`, st.vue === 'toutes' && st.blocs.join() === '0,1', JSON.stringify(st));
        await g.screenshot({ path: path.join(OUT, `site_plaques_${cle}_toutes.png`) });
        // 6. le TYPE « Prédalle + dalle CSP » (rubrique 1 ; CSP : coulé sur place, libellés courts du 09/11/2026) : h_p
        // apparaît, le complément coulé se lit, l'état le porte
        await ouvrirRub('dl');
        await g.click('#donnees button[data-chemin="type"][data-valeur="predalle"]');
        await tranquille();
        await g.waitForTimeout(200);
        const pre = await g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const h = document.querySelector('#donnees input[data-chemin="h_pre"]'), on = document.querySelector('#donnees button.on[data-chemin="type"]');
          const rub = document.querySelector('#donnees .sec[data-sec="dl"]');
          return { on: on && on.dataset.valeur, h: vis(h), hv: h && h.value, complement: (rub.querySelector('.dl-ro') || {}).textContent || '', coule: /coul/i.test(rub.textContent),
            libelles: [...document.querySelectorAll('#donnees button[data-chemin="type"]')].map((x) => x.textContent.trim()).join(' | ') }; });
        const memPre = await noteMemoire();
        verif(`plaques ${cle} : les deux types s'appellent « Dalle CSP » et « Prédalle + dalle CSP » ; le second fait apparaître h_p (h_pre, 6 cm par défaut) et le complément coulé (h − h_p = 14 cm) ; la note le porte (type prédalle)`,
          pre.libelles === 'Dalle CSP | Prédalle + dalle CSP' && pre.on === 'predalle' && pre.h && pre.hv === '6' && /14\s*cm/.test(pre.complement) && memPre && memPre.dalles === 'D2:predalle:6',
          JSON.stringify({ pre, memPre }));
        await g.screenshot({ path: path.join(OUT, `site_plaques_${cle}_predalle.png`) });
        await g.click('#donnees button[data-chemin="type"][data-valeur="coulee"]');
        await tranquille();
        await g.waitForTimeout(200);
        verif(`plaques ${cle} : retour à « Dalle CSP » : h_p disparaît, la note aussi l'oublie`,
          !(await g.$('#donnees input[data-chemin="h_pre"]')) && ((await noteMemoire()) || {}).dalles === 'D2:coulee', JSON.stringify(await noteMemoire()));
        // 7. au téléphone (R1 : la fonction aux trois formats) : la ligne de l'élément en tête de l'onglet Données, chaque
        // bouton à 44 px au doigt (40 avant le 10/11/2026) ; la liste s'ouvre dans l'écran ; un appui sur le nom y pose le curseur sans quitter
        // l'onglet (16 px : pas d'agrandissement) ; « D3 » + Entrée renomme — la note et la tête des résultats suivent
        // (la même note : le stockage du navigateur du PC est repris)
        const ct = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, storageState: await c.storageState() });
        const t = await ct.newPage();
        t.erreurs = [];
        t.on('pageerror', (e) => t.erreurs.push(e.message));
        await pretEcran(t, { apres: await ouvrir(t, g.url()) });
        await t.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
        const lgTel = await ligneElement(t, 44);
        const mnTel = await listeElement(t, true);
        await t.tap(`${LISTE_ELT} [data-menu]`);
        await t.waitForTimeout(150);
        await t.tap(NOM);
        await t.waitForTimeout(250);
        const tel = await t.evaluate((s) => { const a = document.activeElement, r = a && a.getBoundingClientRect();
          return { onglet: document.querySelector('.bench').dataset.tab, champ: !!a && a.matches(s), vu: !!a && a.checkVisibility() && r.top >= 0 && r.bottom <= innerHeight,
            taille: a ? Math.round(r.height) : 0, police: a ? parseFloat(getComputedStyle(a).fontSize) : 0 }; }, NOM);
        await t.screenshot({ path: path.join(OUT, `site_plaques_${cle}_tel_renommer.png`) });
        await t.keyboard.press('Control+a');
        await t.keyboard.type('D3');
        await t.keyboard.press('Enter');
        await t.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
        await t.waitForTimeout(200);
        const telNom = await t.evaluate((s) => { const id = new URLSearchParams(location.search).get('note'), n = JSON.parse(localStorage.getItem('stab.notes.v1') || '{}')[id];
          return { champ: document.querySelector(s).value, rp: document.querySelector('#resultat .et-rp b').textContent, note: n && n.etat.dalles.map((d) => d.nom).join() }; }, NOM);
        verif(`plaques ${cle} · téléphone : la ligne de l'élément tient dans l'en-tête des données, chaque bouton à 44 px au doigt ; la liste s'ouvre dans l'écran ; un appui sur le nom y pose le curseur sans quitter l'onglet (16 px) ; « D3 » + Entrée renomme (la note, la tête des résultats)`,
          lgTel && !lgTel.fautes.length && mnTel.vu && mnTel.dedans && mnTel.touche && mnTel.items === 'D2*' && tel.onglet === 'donnees' && tel.champ && tel.vu && tel.taille >= 44 && tel.police >= 16
          && telNom.champ === 'D3' && telNom.rp === 'D3' && telNom.note === 'D3', JSON.stringify({ lgTel, mnTel, tel, telNom }));
        verif(`plaques ${cle} · téléphone : aucune erreur JavaScript`, !t.erreurs.length, t.erreurs.join(' | ').slice(0, 300));
        await ct.close();
      }
      verif(`plaques ${cle} : aucune erreur JavaScript`, !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
    } catch (x) {
      verif(`plaques ${cle} : le parcours va au bout`, false, String(x.message).split('\n').slice(0, 3).join(' | '));
      await g.screenshot({ path: path.join(OUT, `site_plaques_${cle}_echec.png`) }).catch(() => {});
    }
    await c.close();
  }

  // ---------------------------------------------------------------- les NAPPES toujours posées l'une sur l'autre
  // Décision de l'ingénieur du 09/11/2026 (docs/site/DALLES_FUSION.md § 4.3, révision NAP) : plus d'option « Nappes
  // empilées » ; une note d'avant dont les chiffres changent le dit UNE fois, à l'ouverture (un avis sous le nom, en tête
  // du résultat : pas une erreur rouge) ; un RENFORT de la direction secondaire se pose sur sa nappe (S3) : la coupe le
  // montre dans le lit de sa base, au-dessus des barres principales. PC 1440, puis téléphone 390 (la même note : l'avis
  // dans l'onglet Résultat d'une autre note d'avant, la coupe dans l'onglet Dessin). Puis S5 : un renfort principal plus
  // épais que sa base porte la nappe secondaire (PC, puis téléphone). Ce que l'œil voit : visibilité et positions lues à
  // l'écran.
  //   filtre « nappes » ; aussi joué par « plaques » et « module:dalle ».
  if (!FILTRE || ['nappes', 'plaques', 'module:dalle'].includes(FILTRE)) {
    const c = await b.newContext({ viewport: { width: 1440, height: 900 } });
    const g = await c.newPage();
    g.erreurs = [];
    g.on('pageerror', (e) => g.erreurs.push(e.message));
    g.on('console', (m) => { if (m.type() === 'error') g.erreurs.push(m.text().slice(0, 300)); });
    const pretN = (pg, o) => pretEcran(pg, o);
    const calmeN = async (pg) => { await pg.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 }); await pg.waitForTimeout(200); };
    // l'avis en tête du résultat (juste après le nom) : vu, son texte, un avis (pas « grave »), sa hauteur à l'écran
    const lireAvis = (pg) => pg.evaluate(() => { const m = document.querySelector('#resultat .pbody > .et-rp + .msg');
      return { vu: !!m && m.checkVisibility({ visibilityProperty: true, opacityProperty: true }), texte: m ? m.textContent : '', grave: !!m && m.classList.contains('grave'),
        h: m ? Math.round(m.getBoundingClientRect().height) : 0 }; });
    const semer = (pg, id, nom) => pg.evaluate(([i, n]) => { const k = 'stab.notes.v1', notes = JSON.parse(localStorage.getItem(k) || '{}'), t = Date.now() + 60000;
      notes[i] = { id: i, module: 'dalle', nom: n, chantier: 'Réalex', dossier: '', indice: 'A', cree: t, modifie: t,
        etat: { version: 2, dalles: [{ nom: 'Ancienne D1', empilement: false }], actif: 0 }, resume: null, vignette: '' };
      localStorage.setItem(k, JSON.stringify(notes)); }, [id, nom]);
    const memoireN = (pg, id) => pg.evaluate((i) => { const n = JSON.parse(localStorage.getItem('stab.notes.v1') || '{}')[i];
      return n ? { version: n.etat.version, cle: n.etat.dalles.some((d) => 'empilement' in d), avis: 'avis_ouverture' in n.etat } : null; }, id);
    const AVIS = /^Note d'avant le 09\/11\/2026 : les nappes sont désormais posées l'une sur l'autre \(la direction secondaire, base et renforts, sur la principale\) ; As,req peut différer de l'ancienne note\.$/;
    try {
      await ouvrir(g, `${BASE}/index.html`);
      await semer(g, 'ancienne-dalle', 'Ancienne note dalle');
      await pretN(g, { apres: await ouvrir(g, `${BASE}/dalle?note=ancienne-dalle`) });
      await calmeN(g);
      const a1 = await lireAvis(g);
      const m1 = await memoireN(g, 'ancienne-dalle');
      const sansCase = await g.evaluate(() => !document.querySelector('[data-chemin="empilement"]') && !/Nappes empilées/.test(document.getElementById('donnees').textContent));
      verif('nappes · PC : une note d\'avant (nappes au même niveau) s\'ouvre avec l\'avis « Note d\'avant le 09/11/2026 … » sous le nom, en tête du résultat — visible, un avis (pas une erreur rouge) ; enregistrée en version 3, sans la clé « empilement » ni l\'avis ; aucune case « Nappes empilées »',
        a1.vu && AVIS.test(a1.texte) && !a1.grave && a1.h >= 16 && m1 && m1.version === 3 && !m1.cle && !m1.avis && sansCase, JSON.stringify({ a1, m1, sansCase }));
      await g.screenshot({ path: path.join(OUT, 'site_nappes_avis_pc.png') });
      verif('nappes · PC : le résultat se voit à l\'ouverture — l\'avis n\'y est pas répété dans le message de la page', !(await g.isVisible('#et-alerte')));
      await g.reload();
      await pretN(g);
      await calmeN(g);
      const a2 = await lireAvis(g);
      verif('nappes · PC : rouverte telle qu\'enregistrée, la note ne redit pas l\'avis (une fois, à l\'ouverture)', !a2.vu && !a2.texte, JSON.stringify(a2));
      // S3 : « + Renfort » en inf. X (la direction secondaire) — la coupe ⊥ Y le dessine dans le lit de la base X, posé sur la base Y
      if (!(await g.$eval('#donnees .sec[data-sec="arm"]', (x) => x.classList.contains('open')))) await g.click('#donnees .sec[data-sec="arm"] > .sechd');
      await g.waitForTimeout(150);
      await g.$eval('#donnees .et-add[data-chemin="sections/0/faces/inf_x/couches/+"]', (x) => x.click());
      await calmeN(g);
      await g.click('#vues [data-vue="c0"]');
      await g.waitForTimeout(250);
      const geo = await g.evaluate(() => {
        const v = document.querySelector('#stage .view.on');
        const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const cy = (sel, tag) => { const e = v.querySelector(`[data-sel="${sel}"] ${tag}`); if (!vis(e)) return null; const r = e.getBoundingClientRect(); return r.top + r.height / 2; };
        const leg = [...v.querySelectorAll('text')].find((t) => /X · 3 Ø12 · axe à 50 · sur la base Y/.test(t.textContent));
        const ph = document.querySelector('#donnees input[data-chemin="sections/0/faces/inf_x/couches/1/dist"]');
        return { y: cy('s0.inf_y-0', 'circle'), baseX: cy('s0.inf_x-0', 'line'), renfort: cy('s0.inf_x-1', 'line'), legende: vis(leg), place: ph && ph.placeholder };
      });
      const rapport = geo.y && geo.baseX && geo.renfort ? (geo.y - geo.renfort) / (geo.y - geo.baseX) : 0;
      verif('nappes · PC : S3 — le renfort 3 Ø12 ajouté en inf. X est dessiné AU-DESSUS des barres Y, dans le lit de la base X (axe 50 mm contre 45 et 35 : écarts à la base Y dans le rapport 15 / 10 = 1,5 ; au niveau de la base Y, comme avant, il serait de 0,5) ; sa légende « axe à 50 · sur la base Y » se lit ; son champ dit « auto · 5 »',
        Math.abs(rapport - 1.5) < 0.12 && geo.renfort < geo.baseX && geo.legende && geo.place === 'auto · 5', JSON.stringify({ ...geo, rapport }));
      await g.screenshot({ path: path.join(OUT, 'site_nappes_s3_pc.png') });
      // au téléphone (R1 : le même composant) — la coupe dans l'onglet Dessin ; l'avis d'une AUTRE note d'avant dans l'onglet Résultat
      const ct = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, storageState: await c.storageState() });
      const t = await ct.newPage();
      t.erreurs = [];
      t.on('pageerror', (e) => t.erreurs.push(e.message));
      await pretN(t, { apres: await ouvrir(t, g.url()) });
      await calmeN(t);
      await t.tap('.mtabs [data-onglet="dessin"]');
      await t.waitForTimeout(250);
      await t.tap('#vues [data-vue="c0"]').catch(() => {});
      await t.waitForTimeout(300);
      const geoT = await t.evaluate(() => {
        const v = document.querySelector('#stage .view.on');
        const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const cy = (sel, tag) => { const e = v && v.querySelector(`[data-sel="${sel}"] ${tag}`); if (!vis(e)) return null; const r = e.getBoundingClientRect(); return r.top + r.height / 2; };
        return { y: cy('s0.inf_y-0', 'circle'), baseX: cy('s0.inf_x-0', 'line'), renfort: cy('s0.inf_x-1', 'line') };
      });
      const rapportT = geoT.y && geoT.baseX && geoT.renfort ? (geoT.y - geoT.renfort) / (geoT.y - geoT.baseX) : 0;
      await t.screenshot({ path: path.join(OUT, 'site_nappes_s3_tel.png') });
      verif('nappes · téléphone : la même coupe, le renfort secondaire dans le lit de la base X, au-dessus des barres Y (rapport 1,5)',
        Math.abs(rapportT - 1.5) < 0.15 && geoT.renfort < geoT.baseX, JSON.stringify({ ...geoT, rapportT }));
      await ouvrir(t, `${BASE}/index.html`);
      await semer(t, 'ancienne-dalle-tel', 'Ancienne note téléphone');
      await pretN(t, { apres: await ouvrir(t, `${BASE}/dalle?note=ancienne-dalle-tel`) });
      await calmeN(t);
      // (petits chantiers, 10/11/2026) le téléphone s'ouvre sur l'onglet Données : l'avis s'y lit DÈS L'OUVERTURE, dans le
      // message de la page (#et-alerte, le composant des trois formats), le même texte qu'en tête du résultat — avant,
      // seulement dans l'onglet Résultat, qu'on n'ouvre pas forcément
      const aO = await t.evaluate(() => {
        const al = document.getElementById('et-alerte'), r = al.getBoundingClientRect();
        const x = document.elementFromPoint(r.left + r.width / 2, r.top + 12);
        return { onglet: document.querySelector('.bench').dataset.tab, vu: al.checkVisibility({ visibilityProperty: true, opacityProperty: true }),
          texte: al.querySelector('.et-alerte-txt').textContent, info: al.classList.contains('et-info'), dessus: !!x && al.contains(x),
          dedans: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight, resultat: document.getElementById('resultat').checkVisibility() };
      });
      await t.screenshot({ path: path.join(OUT, 'site_nappes_avis_tel_ouverture.png') });
      verif('nappes · téléphone : à l\'ouverture (onglet Données, le résultat caché), l\'avis d\'une note d\'avant se lit dans le message de la page — visible, dans l\'écran, au-dessus de tout, un avis (pas un échec)',
        aO.onglet === 'donnees' && !aO.resultat && aO.vu && aO.dedans && aO.dessus && aO.info && AVIS.test(aO.texte), JSON.stringify(aO));
      await t.click('#et-alerte [data-fermer-alerte]');
      // (petits chantiers, 10/11/2026) une note dont le nom dépasse 60 caractères : elle s'ouvre (avant : refusée tout
      // entière), le nom coupé, et le même message le dit dès l'ouverture
      {
        const ctl = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const tl = await ctl.newPage();
        tl.erreurs = [];
        tl.on('pageerror', (e) => tl.erreurs.push(e.message));
        await ouvrir(tl, `${BASE}/index.html`);
        const LONG = 'Dalle du plancher haut du rez-de-chaussée, travée centrale entre les files B et C';
        await tl.evaluate((n) => { const t0 = Date.now(); localStorage.setItem('stab.notes.v1', JSON.stringify({ 'dalle-nom-long': { id: 'dalle-nom-long', module: 'dalle',
          nom: 'Nom long', chantier: 'Réalex', dossier: '', indice: 'A', cree: t0, modifie: t0, etat: { version: 3, dalles: [{ nom: n }], actif: 0 }, resume: null, vignette: '' } })); }, LONG);
        await pretN(tl, { apres: await ouvrir(tl, `${BASE}/dalle?note=dalle-nom-long`) });
        await calmeN(tl);
        const nl = await tl.evaluate((echec) => { const al = document.getElementById('et-alerte');
          return { echec, vu: al.checkVisibility({ visibilityProperty: true, opacityProperty: true }),
            texte: al.querySelector('.et-alerte-txt').textContent, nom: (document.querySelector('.panel.left .phead input.et-elt-nom') || {}).value || '',
            garde: JSON.parse(localStorage.getItem('stab.notes.v1'))['dalle-nom-long'].etat.dalles[0].nom.length }; }, !!(await echecEcran(tl)));
        verif('nappes · téléphone : une note dont la dalle a un nom de 81 caractères S\'OUVRE (avant : refusée tout entière) — le nom coupé à 60, et le message le dit dès l\'ouverture',
          !nl.echec && nl.vu && /coupé à 60/.test(nl.texte) && /\(81 caractères\)/.test(nl.texte) && nl.nom === LONG.slice(0, 60).trimEnd() && nl.garde <= 60, JSON.stringify(nl));
        await tl.screenshot({ path: path.join(OUT, 'site_dalle_nom_long_tel.png') });
        verif('nappes · téléphone : la note au nom long, aucune erreur JavaScript', !tl.erreurs.length, tl.erreurs.join(' | ').slice(0, 300));
        await ctl.close();
      }
      await t.tap('.mtabs [data-onglet="resultat"]');
      await t.waitForTimeout(250);
      const aT = await lireAvis(t);
      await t.screenshot({ path: path.join(OUT, 'site_nappes_avis_tel.png') });
      verif('nappes · téléphone : l\'avis d\'une note d\'avant se lit en tête de l\'onglet Résultat (le même composant qu\'au PC)',
        aT.vu && AVIS.test(aT.texte) && !aT.grave, JSON.stringify(aT));
      verif('nappes · téléphone : aucune erreur JavaScript', !t.erreurs.length, t.erreurs.join(' | ').slice(0, 300));
      await ct.close();
      // S5 (relecture du 09/11/2026) : un RENFORT PRINCIPAL 3 Ø20 en inf. Y, plus épais que la base (le treillis Ø10), posé dans
      // son lit — la nappe X (sa base, son renfort S3) repose sur lui : ses traits passent AU-DESSUS des points du renfort Y, au
      // contact, sans les traverser (avant S5 : la base X à 45 mm, 10 mm DANS le renfort) ; la légende dit « sur renfort Y » et
      // TIENT dans sa colonne (« sur le renfort Y » débordait sur celle des supérieures : vu sur la première capture S5). Ce que
      // l'œil voit : le bas du trait et le haut des points, lus à l'écran (matrice de l'écran du dessin), visibles ; le bout de la
      // plus longue ligne des inférieures et le début de la colonne des supérieures, à l'écran.
      const mesureS5 = () => {
        const v = document.querySelector('#stage .view.on');
        const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const pts = v ? [...v.querySelectorAll('[data-sel="s0.inf_y-1"] circle')].filter(vis) : [];
        const trait = v && v.querySelector('[data-sel="s0.inf_x-0"] line');
        const renfX = v && v.querySelector('[data-sel="s0.inf_x-1"] line');
        if (!pts.length || !vis(trait) || !vis(renfX)) return { vu: false, pts: pts.length };
        const bas = (l) => { const m = l.getScreenCTM(); return m.d * (l.y1.baseVal.value + parseFloat(l.getAttribute('stroke-width')) / 2) + m.f; };
        const m = trait.getScreenCTM();
        const haut = Math.min(...pts.map((c) => { const mc = c.getScreenCTM(); return mc.d * (c.cy.baseVal.value - c.r.baseVal.value) + mc.f; }));
        const leg = [...v.querySelectorAll('text')].filter((t) => /^X · .* · axe à \d+ · sur renfort Y$/.test(t.textContent) && vis(t)).map((t) => t.textContent);
        const champ = document.querySelector('#donnees input[data-chemin="sections/0/faces/inf_x/couches/0/dist"]');
        // les deux colonnes de chaque légende : le début des supérieures, le bout de la plus longue ligne des inférieures (px)
        const textes = [...v.querySelectorAll('text')].filter(vis);
        const supG = Math.min(...textes.filter((t) => t.textContent === 'Supérieures').map((t) => t.getBoundingClientRect().left));
        const infD = Math.max(...textes.filter((t) => /^[XY] · /.test(t.textContent) && t.getBoundingClientRect().left < supG - 1).map((t) => t.getBoundingClientRect().right));
        return { vu: true, pts: pts.length, ecart: (haut - bas(trait)) / m.d, ecartRenfX: (haut - bas(renfX)) / m.d, px_mm: m.d, leg, place: champ && champ.placeholder,
          jeu_legende_px: supG - infD };
      };
      if (!(await g.$eval('#donnees .sec[data-sec="arm"]', (x) => x.classList.contains('open')))) await g.click('#donnees .sec[data-sec="arm"] > .sechd');
      await g.waitForTimeout(150);
      await g.$eval('#donnees .et-add[data-chemin="sections/0/faces/inf_y/couches/+"]', (x) => x.click());
      await calmeN(g);
      await g.selectOption('#donnees select[data-chemin="sections/0/faces/inf_y/couches/1/d"]', '20');
      await calmeN(g);
      await g.click('#vues [data-vue="c0"]');
      await g.waitForTimeout(250);
      const s5 = await g.evaluate(mesureS5);
      await g.screenshot({ path: path.join(OUT, 'site_nappes_s5_pc.png') });
      verif('nappes · PC : S5 — un renfort principal 3 Ø20 en inf. Y : la base X (axe 55 mm) passe AU-DESSUS de ses points, au contact (le bas du trait au haut des points, à 0,3 mm près), le bas de son renfort S3 (3 Ø12, axe 60) 4 mm au-dessus ; avant S5, la base X 10 mm dedans ; les légendes de la base X et de son renfort disent « sur renfort Y », dans les deux coupes, et chaque ligne des inférieures s\'arrête avant la colonne des supérieures ; le champ de la base X dit « auto · 5,5 »',
        s5.vu && Math.abs(s5.ecart) < 0.3 && s5.ecartRenfX > 3.7 && s5.ecartRenfX < 4.3 && s5.leg.length === 4 && new Set(s5.leg).size === 2
        && s5.leg.some((x) => /axe à 55 · sur renfort Y/.test(x)) && s5.leg.some((x) => /axe à 60 · sur renfort Y/.test(x)) && s5.place === 'auto · 5,5'
        && s5.jeu_legende_px > 0, JSON.stringify(s5));
      // au téléphone (R1 : le même dessin, la même note) — l'onglet Dessin
      const ct5 = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, storageState: await c.storageState() });
      const t5 = await ct5.newPage();
      t5.erreurs = [];
      t5.on('pageerror', (e) => t5.erreurs.push(e.message));
      await pretN(t5, { apres: await ouvrir(t5, g.url()) });
      await calmeN(t5);
      await t5.tap('.mtabs [data-onglet="dessin"]');
      await t5.waitForTimeout(250);
      await t5.tap('#vues [data-vue="c0"]').catch(() => {});
      await t5.waitForTimeout(300);
      const s5T = await t5.evaluate(mesureS5);
      await t5.screenshot({ path: path.join(OUT, 'site_nappes_s5_tel.png') });
      verif('nappes · téléphone : S5 — la même coupe, la base X au contact au-dessus du renfort Y Ø20 (à 0,3 mm près), le bas de son renfort 4 mm plus haut, les légendes « sur renfort Y », qui tiennent dans leur colonne',
        s5T.vu && Math.abs(s5T.ecart) < 0.3 && s5T.ecartRenfX > 3.7 && s5T.ecartRenfX < 4.3 && s5T.leg.length === 4 && new Set(s5T.leg).size === 2
        && s5T.jeu_legende_px > 0, JSON.stringify(s5T));
      verif('nappes · téléphone (S5) : aucune erreur JavaScript', !t5.erreurs.length, t5.erreurs.join(' | ').slice(0, 300));
      await ct5.close();
      verif('nappes : aucune erreur JavaScript', !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
    } catch (x) {
      verif('nappes : le parcours va au bout', false, String(x.message).split('\n').slice(0, 3).join(' | '));
      await g.screenshot({ path: path.join(OUT, 'site_nappes_echec.png') }).catch(() => {});
    }
    await c.close();
  }

  // ---------------------------------------------------------------- un fichier .stab piégé
  // Un .stab vient d'ailleurs (un collègue, un courriel) : rien de ce qu'il porte ne doit s'exécuter dans la page. Deux
  // fichiers : l'un que le moteur refuse (son dessin piégé n'est alors jamais remplacé par celui du moteur), l'autre
  // valide mais aux noms piégés (note, chantier, poutre, section).
  if (!FILTRE || FILTRE === 'piege') {
    const piege = (n) => `<img src="x" onerror="window.__piege=${n}">`;
    const stab = (note) => ({ name: 'piege.stab', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ format: 'stab', version: 1, note })) });
    const nAvant = erreurs.length;
    await p.goto(`${BASE}/index.html`);
    await Promise.all([p.waitForNavigation(), p.setInputFiles('#importer', stab({ module: 'poutre', nom: 'Note refusée', chantier: piege(1),
      dossier: piege(2), indice: piege(3), etat: { poutres: 'abîmé' }, vignette: piege(4), resume: { taux: piege(5), niveau: `"><b>${piege(6)}` } }))]);
    await pret();
    {
      const be = await bandeauErreur(p);
      verif('.stab abîmé : le moteur le refuse et la page le dit — le bandeau rouge VU, « n\'a pas pu démarrer », « Recharger » VU sous le doigt (PC)',
        be.ecran === 'erreur' && bandeauDit(be), JSON.stringify(be).slice(0, 400));
      // le même fichier au téléphone : son bandeau est dans les onglets (le sien, VU)
      const ctT = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
      const pT = await ctT.newPage();
      try {
        await pT.goto(`${BASE}/index.html`);
        await Promise.all([pT.waitForNavigation(), pT.setInputFiles('#importer', stab({ module: 'poutre', nom: 'Note refusée', chantier: 'Essai', etat: { poutres: 'abîmé' } }))]);
        await pretEcran(pT);
        const bt = await bandeauErreur(pT);
        verif('.stab abîmé : le moteur le refuse et la page le dit — au téléphone aussi (le bandeau rouge VU, « Recharger » VU sous le doigt)',
          bt.ecran === 'erreur' && bandeauDit(bt), JSON.stringify(bt).slice(0, 400));
        await pT.screenshot({ path: path.join(OUT, 'site_stab_abime_tel.png') }).catch(() => {});
      } catch (x) { verif('.stab abîmé (téléphone) : le parcours va au bout', false, String(x.message).split('\n')[0]); } finally { await ctT.close(); }
    }
    // deux sections : la vue « Toutes les sections » écrit leurs noms dans le dessin
    const e2 = { poutres: [{ nom: piege(7), b: 25, h: 50, sections: [{ nom: piege(8), M_inf: 80, V: 40 }, { nom: piege(11), M_sup: 50, V: 30 }] }], actif: 0 };
    await p.goto(`${BASE}/index.html`);
    await Promise.all([p.waitForNavigation(), p.setInputFiles('#importer', stab({ module: 'poutre', nom: piege(9), chantier: piege(10), etat: e2 }))]);
    await pret();
    await calme();
    const etabli = await p.evaluate(() => ({ piege: window.__piege, img: document.querySelectorAll('img[src="x"]').length, nom: document.getElementById('nom-note').value }));
    verif('.stab aux noms piégés : la note s\'ouvre, les noms s\'affichent comme du texte, rien ne s\'exécute',
      etabli.piege === undefined && etabli.img === 0 && etabli.nom.startsWith('<img'), JSON.stringify(etabli));
    await p.goto(`${BASE}/index.html`);
    await p.waitForTimeout(300);
    const accueil = await p.evaluate(() => ({ piege: window.__piege, img: document.querySelectorAll('img[src="x"]').length }));
    verif('accueil après deux .stab piégés : aucun code du fichier ne s\'exécute (dessin, résumé, chantier, indice)',
      accueil.piege === undefined && accueil.img === 0, JSON.stringify(accueil));
    // les erreurs attendues (le moteur qui refuse le fichier abîmé) ne comptent pas ; une exception non rattrapée, si
    const recues = erreurs.splice(nAvant);
    const autres = recues.filter((x) => !/Traceback|ValueError|refus/.test(x));
    verif('.stab abîmé : seules les erreurs du refus, aucune exception de la page', !autres.length, `${recues.length} message(s) du refus${autres.length ? ' · ' + autres.join(' | ').slice(0, 300) : ''}`);
    // les deux notes piégées ont servi : elles quittent l'accueil (la capture finale montre les vraies notes)
    await p.evaluate(() => {
      const k = 'stab.notes.v1', notes = JSON.parse(localStorage.getItem(k) || '{}');
      for (const [id, x] of Object.entries(notes)) if (/__piege|Note refusée/.test(`${x.nom} ${x.chantier}`)) delete notes[id];
      localStorage.setItem(k, JSON.stringify(notes));
    });
  }

  //   filtre « bibliotheque » : la page Bibliothèque Béton armé (docs/site/BIBLIOTHEQUES.md § 2.2), aux quatre formats.
  //   Chaque rubrique : le tableau se voit D'UN COUP (PC et tablette : son cadre ne défile ni en hauteur ni en largeur),
  //   la case allumée est à l'écran, rien ne déborde ; puis les gestes (PC) : une case cliquée règle les paramètres, un
  //   paramètre réglé allume la case, « Pourquoi ? » s'ouvre et se ferme, l'adresse garde l'état et le restitue.
  //   D'abord la CARTE de la bibliothèque à l'accueil (docs/site/BIBLIOTHEQUES.md § 2.1, révision du 09/11/2026 :
  //   « elle serait beaucoup moins large et plus haute, et les modules seraient à droite ») : au PC et à la tablette, la
  //   carte occupe UNE colonne (pas plus large qu'une tuile), sur TOUTE la hauteur du bloc de ses modules, à GAUCHE de
  //   chacun d'eux, aucun sous elle ; les modules sur les lignes de colonnes des familles sans bibliothèque (un seul
  //   rythme) ; « Bibliothèque », le titre et les six rubriques visibles, dans la carte, aucun coupé. Au téléphone : la
  //   carte pleine largeur, en tête de la liste de sa famille. Thème sombre : la carte reste distincte d'une tuile.
  const FORMATS_BIB = [['pc', 1440, 900, false], ['tablette', 1180, 820, false], ['portrait', 820, 1180, false], ['tel', 390, 844, true]];
  if (!FILTRE || FILTRE === 'bibliotheque') {
    const mesureAccueilBib = (pg) => pg.evaluate(() => {
      const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const R = (e) => e.getBoundingClientRect();
      const plage = (e) => { const g = document.createRange(); g.selectNodeContents(e); return g.getBoundingClientRect(); };
      const familles = [...document.querySelectorAll('.accueil .famille')].map((f) => {
        const tuiles = [...f.querySelectorAll('.tile:not(.biblio)')].filter(vu);
        const rt = tuiles.map(R);
        const o = { fam: f.dataset.fam, n: tuiles.length, gauches: [...new Set(rt.map((t) => Math.round(t.left)))].sort((a, b) => a - b),
          large: Math.round(Math.max(...rt.map((t) => t.width))), haut: Math.round(Math.min(...rt.map((t) => t.top))), hautT: Math.round(Math.max(...rt.map((t) => t.height))) };
        // les PISTES de la grille du bloc (la grille de la charte : ses colonnes, vides comprises) — leurs bords gauches, lus sur
        // la grille elle-même (une famille peut ne pas remplir toutes ses colonnes : « Autres » n'a que trois modules)
        const grille = f.querySelector(':scope > .tiles');
        if (grille) {
          const gs = getComputedStyle(grille), gr = R(grille), ecart = parseFloat(gs.columnGap) || 0;
          let x = gr.left + (parseFloat(gs.borderLeftWidth) || 0) + (parseFloat(gs.paddingLeft) || 0);
          o.pistes = gs.gridTemplateColumns.split(' ').map(parseFloat).filter((l) => l > 0).map((l) => { const g0 = Math.round(x); x += l + ecart; return g0; });
        } else o.pistes = [];
        const carte = f.querySelector('.tile.biblio');
        if (carte) {
          // (sans le bloc .fam-b — l'ancienne forme, la carte DANS la grille des modules — on mesure la grille qui la porte :
          //  les contrôles rougissent sur ce que l'œil voit, sans que la mesure ne tombe)
          const c = R(carte), bloc = R(f.querySelector('.fam-b > .tiles') || carte.parentElement), cadre = R(f.querySelector('.fam-b') || carte.parentElement);
          const textes = [...carte.querySelectorAll('.nm, .bib-r li')];
          o.carte = { vu: vu(carte), g: Math.round(c.left), d: Math.round(c.right), h: Math.round(c.top), b: Math.round(c.bottom), l: Math.round(c.width), t: Math.round(c.height) };
          o.bloc = { h: Math.round(bloc.top), b: Math.round(bloc.bottom), l: Math.round(cadre.width) };
          o.aDroite = rt.every((t) => t.left >= c.right - 0.5);
          o.sous = rt.filter((t) => t.left < c.right - 0.5).length;
          o.audessus = rt.filter((t) => t.top < c.bottom - 0.5).length;
          o.hors = textes.filter((e) => { const r = R(e), k = plage(e); return !vu(e) || r.width < 1 || [r, k].some((q) => q.left < c.left - 0.5 || q.right > c.right + 0.5 || q.top < c.top - 0.5 || q.bottom > c.bottom + 0.5); })
            .map((e) => e.textContent.trim().slice(0, 30));
          o.coupes = textes.filter((e) => e.scrollWidth > e.clientWidth + 1 || plage(e).width > R(e).width + 0.5).map((e) => e.textContent.trim().slice(0, 30));
          o.rubriques = [...carte.querySelectorAll('.bib-r li')].filter(vu).map((e) => e.textContent.trim()).join(',');
          o.titre = carte.querySelector('.nm').textContent.replace(/\s+/g, ' ').trim();
          // (petits chantiers, 10/11/2026) les rangées de ses modules, lues à l'écran ; l'icône au-dessus du titre ou à sa gauche
          o.rangees = new Set(rt.map((t) => Math.round(t.top))).size;
          const ri = R(carte.querySelector('.modic')), rn = R(carte.querySelector('.nm'));
          o.icone = ri.bottom <= rn.top + 0.5 ? 'dessus' : ri.right <= rn.left + 0.5 ? 'gauche' : 'autre';
          const t0 = tuiles[0] ? getComputedStyle(tuiles[0]) : null, cs = getComputedStyle(carte);
          o.distincte = !!t0 && cs.borderTopColor !== t0.borderTopColor && getComputedStyle(carte.querySelector('.modic')).backgroundColor !== getComputedStyle(tuiles[0].querySelector('.modic')).backgroundColor;
        }
        return o;
      });
      return { familles, deborde: document.documentElement.scrollWidth > innerWidth, theme: document.documentElement.dataset.theme || 'clair' };
    });
    // PC et tablette : chaque carte au plus large comme une tuile, sur toute la hauteur du bloc de ses modules, à gauche de chacun
    const etroiteHaute = (avec) => avec.every((f) => f.n >= 1 && f.carte.l <= f.large + 1 && f.carte.l >= 0.8 * f.large && Math.abs(f.carte.h - f.bloc.h) <= 1
      && Math.abs(f.carte.b - f.bloc.b) <= 1 && f.carte.t >= 2 * 52 && f.aDroite && f.sous === 0);
    // un seul rythme : les MÊMES pistes pour le bloc de chaque famille (la grille de la charte) ; les tuiles des familles sans
    // bibliothèque sur ces pistes ; la carte sur la première, ses modules sur les SUIVANTES, et ils les REMPLISSENT (autant de
    // colonnes occupées que le permettent leur nombre et la grille — une pile d'une seule colonne à côté de la carte, des
    // colonnes vides au bout, serait un défaut)
    const rythme = (familles) => {
      const lignes = familles.length ? familles[0].pistes : [], sur = (x, ls) => ls.some((l) => Math.abs(l - x) <= 1);
      return lignes.length >= 2 && familles.every((f) => f.pistes.length === lignes.length && f.pistes.every((x, i) => Math.abs(x - lignes[i]) <= 1))
        && familles.filter((f) => !f.carte).every((f) => f.gauches.every((x) => sur(x, lignes)))
        && familles.filter((f) => f.carte).every((f) => Math.abs(f.carte.g - lignes[0]) <= 1 && f.gauches.every((x) => sur(x, lignes.slice(1)))
          && f.gauches.length === Math.min(f.n, lignes.length - 1));
    };
    for (const [nomF, opts, sombre] of [['PC 1440', { viewport: { width: 1440, height: 900 } }], ['PC 1280', { viewport: { width: 1280, height: 800 } }],
      ['tablette 1180', { viewport: { width: 1180, height: 820 }, hasTouch: true }], ['tablette 1024', { viewport: { width: 1024, height: 768 }, hasTouch: true }],
      ['portrait 820', { viewport: { width: 820, height: 1180 }, hasTouch: true }], ['petite tablette 701', { viewport: { width: 701, height: 1000 }, hasTouch: true }],
      ['téléphone', { ...devices['iPhone 13'] }],
      ['PC 1440 sombre', { viewport: { width: 1440, height: 900 } }, true], ['téléphone sombre', { ...devices['iPhone 13'] }, true]]) {
      const c = await b.newContext({ deviceScaleFactor: 2, ...opts });
      if (sombre) await c.addInitScript(() => { try { localStorage.setItem('stab.theme', 'dark'); } catch (e) { /* le thème du système */ } });
      const g = await c.newPage();
      g.erreurs = [];
      g.on('pageerror', (e) => g.erreurs.push(e.message));
      await g.goto(`${BASE}/index.html`);
      await g.waitForTimeout(300);
      const m = await mesureAccueilBib(g);
      const tel = nomF.startsWith('téléphone');
      const avec = m.familles.filter((f) => f.carte);
      verif(`accueil · ${nomF} : chaque bibliothèque a sa carte, « Bibliothèque Béton armé » (six rubriques) et « Bibliothèque Acier » (trois), toutes visibles dans leur carte, aucune coupée, rien ne déborde${sombre ? ' (thème sombre)' : ''}`,
        avec.length >= 2 && avec.some((f) => f.fam === 'beton' && f.titre === 'Bibliothèque Béton armé' && f.rubriques === 'Bétons,Aciers,Sections,Enrobages,Ancrages,Hourdis')
        && avec.some((f) => f.fam === 'acier' && f.titre === 'Bibliothèque Acier' && f.rubriques === 'Profilés,Nuances,Boulons')
        && avec.every((f) => f.carte.vu && !f.hors.length && !f.coupes.length) && !m.deborde && (!sombre || m.theme === 'dark'), JSON.stringify({ avec, deborde: m.deborde, theme: m.theme }));
      if (!tel) {
        verif(`accueil · ${nomF} : la carte est ÉTROITE et HAUTE — une colonne (≤ une tuile), toute la hauteur du bloc de ses modules, à GAUCHE de chacun, aucun sous elle`,
          etroiteHaute(avec), JSON.stringify(avec.map((f) => ({ fam: f.fam, carte: f.carte, bloc: f.bloc, large: f.large, n: f.n, sous: f.sous }))));
        // (petits chantiers, 10/11/2026) de 701 à 1 360 px comme au PC : les cinq modules du Béton et de l'Acier sur DEUX rangées
        // (ils en prenaient trois, la carte aussi) ; la carte, plus étroite, pose son icône AU-DESSUS du titre ; au PC large, à gauche
        const etroit = (opts.viewport || {}).width <= 1360;
        // (11/11/2026 : « Autres » a sa bibliothèque — le harnais SUIT : la règle des deux rangées vaut pour le Béton et l'Acier ;
        //  « Autres » n'a qu'une rangée de modules, et sa carte garde la hauteur de DEUX rangées — demande de l'ingénieur : « le
        //  rectangle Bibliothèque de même dimension qu'au-dessus », en attendant que la famille se complète)
        const grandes = avec.filter((f) => f.fam !== 'autres');
        verif(`accueil · ${nomF} : les modules du Béton et de l'Acier sur DEUX rangées, la carte à leur hauteur (120 px), son icône ${etroit ? 'AU-DESSUS du' : 'à gauche du'} titre`,
          grandes.length === 2 && grandes.every((f) => f.rangees === 2 && Math.abs(f.carte.t - 120) <= 1 && f.icone === (etroit ? 'dessus' : 'gauche')),
          JSON.stringify(avec.map((f) => ({ fam: f.fam, rangees: f.rangees, carte: f.carte.t, icone: f.icone }))));
        const fo = avec.find((f) => f.fam === 'autres');
        verif(`accueil · ${nomF} : la carte « Bibliothèque Autres » (Poutres, Plaques, Sections) a la hauteur de DEUX rangées (120 px, comme celles du dessus) bien que sa famille n'ait qu'une rangée de modules, restés à la hauteur d'une tuile`,
          !!fo && fo.titre === 'Bibliothèque Autres' && fo.rubriques === 'Poutres,Plaques,Sections' && fo.rangees === 1 && Math.abs(fo.carte.t - 120) <= 1
          && fo.hautT <= Math.max(...grandes.map((f) => f.hautT)) + 1 && fo.icone === (etroit ? 'dessus' : 'gauche'),
          JSON.stringify(fo && { rangees: fo.rangees, carte: fo.carte, hautT: fo.hautT, rubriques: fo.rubriques, icone: fo.icone }));
        verif(`accueil · ${nomF} : un seul rythme de colonnes — la carte sur la première, ses modules sur TOUTES les suivantes, celles des familles sans bibliothèque`,
          rythme(m.familles), JSON.stringify({ pistes: m.familles.map((f) => [f.fam, f.pistes]), avec: avec.map((f) => [f.carte.g, f.gauches]) }));
        if (nomF === 'PC 1440') {
          // BIB5 (« l'Acier aura sa bibliothèque sans rien recoder ») : la carte d'une bibliothèque Acier posée dans la page sous
          // la forme que construire rend (tests/test_site_web.py, bibliothèque d'essai : .tiles.fam-b = la carte, puis la grille
          // des modules) — la forme tient et les deux contrôles ci-dessus restent verts. (Relecture du 09/11/2026 : déduites des
          // tuiles d'une famille SANS bibliothèque, les lignes de colonnes n'étaient plus que trois à 1 440 dès que « Autres »
          // restait seule — trois modules sur quatre colonnes — et ces contrôles rougissaient sur une forme juste.) Et une
          // famille de PEU de modules (« Autres », trois : une seule rangée, plus basse que la carte) garde des tuiles à la
          // hauteur d'une tuile (elles s'étiraient à 120 px).
          // (BIB5 fait : la carte Acier est la VRAIE ; seule « Autres » reçoit une carte d'essai — une famille qui a déjà sa
          //  carte est laissée telle quelle : le harnais SUIT)
          const tuileH = Math.max(...m.familles.map((f) => f.hautT));
          await g.evaluate(() => {
            for (const [k, nom] of [['acier', 'Acier'], ['autres', 'Autres']]) {
              const fam = document.querySelector(`.famille[data-fam="${k}"]`), grille = fam && fam.querySelector(':scope > .tiles');
              if (!grille || fam.querySelector('.tile.biblio')) continue;
              const carte = document.querySelector('.tile.biblio').cloneNode(true);
              carte.dataset.bibliotheque = `essai-${k}`; carte.setAttribute('href', `bibliotheque-essai-${k}`); carte.querySelector('.bib-t').textContent = nom;
              const bloc = document.createElement('div'); bloc.className = 'tiles fam-b'; bloc.append(carte); grille.replaceWith(bloc); bloc.append(grille);
            }
          });
          await g.waitForTimeout(150);
          const ms = await mesureAccueilBib(g), avecS = ms.familles.filter((f) => f.carte);
          // (11/11/2026 : « Autres » a désormais SA carte ; la boucle ci-dessus ne pose plus rien — le contrôle mesure les trois vraies)
          verif(`accueil · ${nomF} : les trois bibliothèques (Béton armé, Acier, Autres — trois modules) prennent la même forme — étroites et hautes, leurs modules à droite à la hauteur d'une tuile, un seul rythme de colonnes`,
            avecS.length === 3 && ['acier', 'autres'].every((k) => avecS.some((f) => f.fam === k)) && etroiteHaute(avecS) && rythme(ms.familles)
            && ms.familles.every((f) => f.hautT <= tuileH + 1),
            JSON.stringify({ tuileH, pistes: ms.familles.map((f) => [f.fam, f.pistes]), avec: avecS.map((f) => [f.fam, f.carte.g, f.carte.l, f.carte.t, f.gauches, f.hautT]) }));
          await g.reload();
          await g.waitForTimeout(300);
        }
      } else {
        verif(`accueil · ${nomF} : la carte pleine largeur, en tête de la liste de sa famille, les modules dessous`,
          avec.every((f) => f.carte.l >= f.bloc.l - 2 && f.audessus === 0 && f.carte.b <= f.haut + 1), JSON.stringify(avec.map((f) => ({ carte: f.carte, bloc: f.bloc, haut: f.haut }))));
      }
      verif(`accueil · ${nomF} : la carte se distingue d'une tuile de calcul (contour et icône de l'accent)`, avec.every((f) => f.distincte), JSON.stringify(avec.map((f) => f.distincte)));
      verif(`accueil · ${nomF} : aucune erreur JavaScript`, !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
      await g.screenshot({ path: path.join(OUT, `site_accueil_biblio_${nomF.replace(/ /g, '_').replace('é', 'e').replace('é', 'e')}.png`), fullPage: tel });
      await c.close();
    }

    const RUBRIQUES_BIB = ['betons', 'aciers', 'sections', 'enrobages', 'ancrages', 'hourdis'];
    const attendreBib = async (pg) => {
      await pretEcran(pg);
      await pg.waitForFunction(() => {
        const c = document.getElementById('bib-corps');
        return c && c.getAttribute('aria-busy') !== 'true' && document.querySelector('#bib-tableau .bib-tab, #bib-tableau .bib-vide');
      }, null, { timeout: 180000 });
    };
    const calmeBib = async (pg) => { await pg.waitForTimeout(60); await attendreBib(pg); };
    // ce que l'œil voit : le tableau entier dans son cadre, le cadre dans l'écran, la case allumée visible, la bande visible
    const mesureBib = (pg) => pg.evaluate(() => {
      const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const cadre = document.getElementById('bib-tableau');
      const r = cadre.getBoundingClientRect();
      const cur = document.querySelector('#bib-tableau .cur');
      const rc = cur ? cur.getBoundingClientRect() : null;
      const bande = document.getElementById('bib-bande').getBoundingClientRect();
      const large = (e) => { const g = document.createRange(); g.selectNodeContents(e); return g.getBoundingClientRect().width; };
      const coupes = [...document.querySelectorAll('.bib-m .bib-mt, .bib-pl, .bib-tab th.nm, .bib-bande-b .btn span')]
        .filter(vis).filter((e) => e.scrollWidth > e.clientWidth + 1 || large(e) > e.getBoundingClientRect().width + 0.01).map((e) => e.textContent.trim().slice(0, 40));
      // le bouton RETENU d'un segment de la barre tient dans son groupe (à la tablette, les 44 px des cibles de l'établi le
      // faisaient dépasser de 14 px d'un groupe de 30 px : il recouvrait le libellé au-dessus — 10/11/2026)
      const segs = [...document.querySelectorAll('#bib-barre .seg')].filter(vis).map((s) => { const q = s.querySelector('button.on');
        if (!q) return null; const a = s.getBoundingClientRect(), z = q.getBoundingClientRect();
        return [q.textContent.trim().slice(0, 24), Math.round(Math.max(0, a.top - z.top) + Math.max(0, z.bottom - a.bottom))]; }).filter((x) => x && x[1] > 0);
      return { segs, deborde: document.documentElement.scrollWidth > innerWidth, defileY: cadre.scrollHeight > cadre.clientHeight + 1, defileX: cadre.scrollWidth > cadre.clientWidth + 1,
        cadre: [Math.round(r.width), Math.round(r.height)], cadreDansEcran: r.bottom <= innerHeight + 1 && r.right <= innerWidth + 1,
        curVisible: !!rc && rc.top >= r.top - 1 && rc.bottom <= r.bottom + 1 && rc.left >= r.left - 1 && rc.right <= r.right + 1 && vis(cur),
        bandeVisible: bande.height > 30 && bande.bottom <= innerHeight + 1, coupes, titre: document.getElementById('bib-titre').textContent,
        valeur: (document.querySelector('.bib-val') || {}).textContent, vide: !!document.querySelector('#bib-tableau .bib-vide') };
    });
    for (const [nomF, w, h, tel] of FORMATS_BIB) {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel });
      const g = await c.newPage();
      g.erreurs = [];
      g.on('pageerror', (e) => g.erreurs.push(e.message));
      g.on('console', (m) => { if (m.type() === 'error') g.erreurs.push(m.text().slice(0, 300)); });
      await g.goto(`${BASE}/bibliotheque-beton`);
      await attendreBib(g);
      for (const r of RUBRIQUES_BIB) {
        await g.click(`.bib-m[data-rubrique="${r}"]`);
        await calmeBib(g);
        const m = await mesureBib(g);
        const url = new URL(g.url());
        verif(`bibliothèque ${nomF} · ${r} : la rubrique s'ouvre, son titre et son adresse (?rubrique=${r})`, m.titre.length > 3 && url.searchParams.get('rubrique') === r, `${m.titre} · ${url.search}`);
        verif(`bibliothèque ${nomF} · ${r} : rien ne déborde, aucun nom coupé`, !m.deborde && !m.coupes.length, m.coupes.join(', '));
        verif(`bibliothèque ${nomF} · ${r} : le bouton retenu de chaque segment de la barre tient dans son groupe`, !m.segs.length, JSON.stringify(m.segs));
        if (!tel) {
          verif(`bibliothèque ${nomF} · ${r} : le tableau se voit d'un coup (son cadre ne défile pas) et tient dans l'écran avec la bande`,
            !m.defileY && !m.defileX && m.cadreDansEcran && m.bandeVisible, JSON.stringify(m));
        } else {
          verif(`bibliothèque ${nomF} · ${r} : le cadre du tableau et la bande sont dans la page`, m.cadre[0] > 200 && m.cadre[1] > 120, JSON.stringify(m.cadre));
        }
        verif(`bibliothèque ${nomF} · ${r} : la case allumée est visible, la valeur retenue est écrite`, m.curVisible && /\d/.test(m.valeur || ''), JSON.stringify({ cur: m.curVisible, v: m.valeur }));
        await g.screenshot({ path: path.join(OUT, `site_bib_${r}_${nomF}.png`) });
      }
      if (nomF === 'pc') {
        // les gestes, sur les ancrages : la case cliquée règle Ø et classe ; Ø réglé allume la case ; « Pourquoi ? »
        await g.click('.bib-m[data-rubrique="ancrages"]');
        await calmeBib(g);
        const avant = await g.$eval('.bib-val', (x) => x.textContent);
        await g.click('#bib-tableau td[data-sel*="\\"phi\\": 25"][data-sel*="C40/50"]');
        await calmeBib(g);
        const apres = await g.evaluate(() => ({ v: document.querySelector('.bib-val').textContent, phi: document.querySelector('.bib-barre select[data-param="phi"]').value,
          classe: document.querySelector('.bib-barre select[data-param="classe"]').value, cur: document.querySelector('#bib-tableau .cur')?.dataset.sel, url: location.search }));
        verif('bibliothèque pc · ancrages : cliquer la case [Ø 25 ; C40/50] règle Ø et classe, la bande et l\'adresse suivent',
          apres.phi === '25' && apres.classe === 'C40/50' && apres.v !== avant && /phi=25/.test(apres.url) && /C40%2F50|C40\/50/.test(apres.url), JSON.stringify(apres));
        await g.selectOption('.bib-barre select[data-param="phi"]', '12');
        await calmeBib(g);
        const cur = await g.$eval('#bib-tableau .cur', (x) => JSON.parse(x.dataset.sel));
        verif('bibliothèque pc · ancrages : régler Ø 12 allume la case [Ø 12 ; C40/50]', cur.phi === 12 && cur.classe === 'C40/50', JSON.stringify(cur));
        await g.click('.bib-barre .seg[data-param="conditions"] button[data-valeur="mediocres"]');
        await calmeBib(g);
        const v2 = await g.$eval('.bib-val', (x) => x.textContent);
        verif('bibliothèque pc · ancrages : « autres conditions » change la valeur (η1 = 0,7)', v2 !== apres.v, `${apres.v} → ${v2}`);
        await g.click('[data-pourquoi]');
        await g.waitForTimeout(120);
        const pq = await g.evaluate(() => { const p = document.getElementById('bib-pourquoi'); return { ouvert: !p.hidden && p.checkVisibility(), lignes: p.querySelectorAll('.bib-pq-t tr').length, sources: p.querySelectorAll('.bib-src').length }; });
        verif('bibliothèque pc · ancrages : « Pourquoi ? » ouvre la justification (termes, sources)', pq.ouvert && pq.lignes >= 12 && pq.sources >= 1, JSON.stringify(pq));
        await g.screenshot({ path: path.join(OUT, 'site_bib_pourquoi_pc.png') });
        await g.keyboard.press('Escape');
        await g.waitForTimeout(80);
        verif('bibliothèque pc · ancrages : Échap ferme la justification', await g.$eval('#bib-pourquoi', (p) => p.hidden));
        // l'adresse restitue l'état
        const adresse = g.url();
        await g.goto(adresse);
        await attendreBib(g);
        const relu = await g.evaluate(() => ({ v: document.querySelector('.bib-val').textContent, cur: document.querySelector('#bib-tableau .cur')?.dataset.sel,
          cond: document.querySelector('.bib-barre .seg[data-param="conditions"] button.on')?.dataset.valeur }));
        verif('bibliothèque pc · ancrages : l\'adresse rouverte redonne le même état (Ø 12, C40/50, autres conditions, même valeur)',
          relu.v === v2 && JSON.parse(relu.cur).phi === 12 && relu.cond === 'mediocres', JSON.stringify(relu));
        // les enrobages : l'exemple de la NIT 260
        await g.goto(`${BASE}/bibliotheque-beton?rubrique=enrobages&onglet=tableaux&element=dalle&exposition=XC1&classe=C30%2F37&phi=12&phi_ext=0`);
        await attendreBib(g);
        const nit = await g.evaluate(() => ({ v: document.querySelector('.bib-val').textContent.replace(/\s/g, ''), cur: document.querySelector('#bib-tableau .cur')?.textContent,
          ligne: document.querySelector('#bib-tableau tr.on th.nm')?.textContent.trim().slice(0, 2) }));
        verif('bibliothèque pc · enrobages : l\'exemple de la NIT 260 (dalle XC1 C30/37 Ø 12) → 22 mm, case [S2 ; XC1] = 10', nit.v === '22mm' && nit.cur === '10' && nit.ligne === 'S2', JSON.stringify(nit));
        await g.click('.bib-barre .seg[data-param="vue"] button[data-valeur="feu"]');
        await calmeBib(g);
        verif('bibliothèque pc · enrobages : la vue « feu » montre le tableau 5.8 de la dalle', (await g.$$eval('#bib-tableau .bib-feu tbody tr', (x) => x.length)) === 6);
        await g.screenshot({ path: path.join(OUT, 'site_bib_enrobages_feu_pc.png') });
      }
      // les enrobages, deux onglets (Lot Q, 10/11/2026) : un POTEAU R 90 à μ_fi = 0,7 (350/53) et un VOILE REI 180, au PC
      // et au téléphone — « Calcul » (le dessin à l'échelle, la feuille, c_nom allumé) puis « Tableaux » (la case
      // allumée), l'adresse suit l'onglet ; ce que l'œil voit : le cadre tient au PC, aucun texte de dessin sous 6 px
      if (nomF === 'pc' || nomF === 'tel') {
        const vueEnr = () => g.evaluate(() => {
          const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const cadre = document.getElementById('bib-tableau'), svg = cadre.querySelector('svg');
          let police = null;
          if (svg && vis(svg)) {
            const k = svg.getBoundingClientRect().width / svg.viewBox.baseVal.width;
            police = Math.min(...[...svg.querySelectorAll('text')].map((t) => parseFloat(getComputedStyle(t).fontSize) * k));
          }
          return { v: (document.querySelector('.bib-val') || {}).textContent?.replace(/\s/g, ''), cur: document.querySelector('#bib-tableau .cur')?.textContent.replace(/\s/g, ''),
            ligne: document.querySelector('#bib-tableau tr.on')?.dataset.terme || null, dessin: svg && vis(svg) ? Math.round(svg.getBoundingClientRect().width) : 0,
            police: police && Math.round(police * 10) / 10, defile: cadre.scrollHeight > cadre.clientHeight + 1, deborde: document.documentElement.scrollWidth > innerWidth,
            onglet: new URL(location.href).searchParams.get('onglet') };
        });
        await g.goto(`${BASE}/bibliotheque-beton?rubrique=enrobages&element=poteau&feu=R90&mu_fi=0.7&b=350`);
        await attendreBib(g);
        const pc1 = await vueEnr();
        verif(`bibliothèque ${nomF} · enrobages · poteau R 90, μ_fi = 0,7, b = 350 : « Calcul » ouvert par défaut, c_nom = 35 mm allumé dans la feuille, le dessin visible (textes ≥ 6 px)${tel ? '' : ', le cadre tient sans défiler'}`,
          pc1.v === '35mm' && pc1.ligne === 'cnom' && pc1.cur === '35mm' && pc1.dessin > 200 && pc1.police >= 6 && !pc1.deborde && (tel || !pc1.defile), JSON.stringify(pc1));
        await g.screenshot({ path: path.join(OUT, `site_bib_enrobages_poteau_calcul_${nomF}.png`), fullPage: tel });
        await g.click('.bib-barre .seg[data-param="onglet"] button[data-valeur="tableaux"]');
        await calmeBib(g);
        const pt1 = await vueEnr();
        verif(`bibliothèque ${nomF} · enrobages · poteau : « Tableaux » allume la case 350/53 du tableau 5.2a, la même valeur, l'adresse suit (onglet=tableaux)`,
          pt1.cur === '53≥350' && pt1.v === '35mm' && pt1.onglet === 'tableaux' && !pt1.deborde, JSON.stringify(pt1));
        await g.screenshot({ path: path.join(OUT, `site_bib_enrobages_poteau_tableaux_${nomF}.png`), fullPage: tel });
        await g.goto(`${BASE}/bibliotheque-beton?rubrique=enrobages&onglet=tableaux&element=voile&feu=R180&e=210&phi=20`);
        await attendreBib(g);
        const vt = await vueEnr();
        verif(`bibliothèque ${nomF} · enrobages · voile REI 180, μ_fi = 0,7, une face, e = 210 : la case 210/50 du tableau 5.4 allumée, c_nom = 50 − 10 = 40 mm (aucun étrier)`,
          vt.cur === '50≥210' && vt.v === '40mm' && !vt.deborde, JSON.stringify(vt));
        await g.click('.bib-barre .seg[data-param="onglet"] button[data-valeur="calcul"]');
        await calmeBib(g);
        const vc = await vueEnr();
        verif(`bibliothèque ${nomF} · enrobages · voile : « Calcul » — la coupe du voile dessinée (textes ≥ 6 px), c_nom allumé, l'adresse suit (onglet=calcul)${tel ? '' : ', le cadre tient'}`,
          vc.ligne === 'cnom' && vc.v === '40mm' && vc.dessin > 200 && vc.police >= 6 && vc.onglet === 'calcul' && (tel || !vc.defile), JSON.stringify(vc));
        await g.screenshot({ path: path.join(OUT, `site_bib_enrobages_voile_calcul_${nomF}.png`), fullPage: tel });
      }
      verif(`bibliothèque ${nomF} : aucune erreur JavaScript`, g.erreurs.length === 0, g.erreurs.join(' | ').slice(0, 300));
      await c.close();
    }

    // ---- la Bibliothèque ACIER (BIB5, api_web/biblio_acier.py), aux mêmes quatre formats. « Nuances » et « Boulons » sont
    // de petits tableaux : ils se voient d'un coup (PC, tablette). « Profilés » est le catalogue entier (des centaines de
    // lignes) : son tableau DÉFILE dans son cadre — mesuré : le cadre et la bande restent dans l'écran, l'en-tête et la
    // colonne des noms restent posés quand on fait défiler. Puis les gestes (PC) : une ligne ouvre le profilé, un en-tête
    // trie, une pastille retire une famille, un filtre I_y ≥ … nomme le plus léger, la fiche montre la section, le menu
    // « Colonnes » reste ouvert d'un clic à l'autre et se referme dehors, l'adresse rouvre le même état ; au téléphone, un
    // doigt ouvre un profilé et sa fiche plein écran. Enfin, une ANCIENNE note du Catalogue des profilés (masqué) s'ouvre
    // encore dans sa page.
    const RUBRIQUES_ACIER = ['profils', 'nuances', 'boulons'];
    const tableauProfils = (pg) => pg.evaluate(() => {
      const c = document.getElementById('bib-tableau'), r = c.getBoundingClientRect();
      const avant = c.scrollTop;
      c.scrollTop = c.scrollHeight;
      const th = c.querySelector('thead th.nm').getBoundingClientRect(), nom = c.querySelector('tbody tr:last-child th.nm');
      const rn = nom ? nom.getBoundingClientRect() : null;
      const o = { defile: c.scrollHeight > c.clientHeight + 1, lignes: c.querySelectorAll('tbody tr').length, descendu: c.scrollTop > 0,
        entete: th.top >= r.top - 1 && th.bottom <= r.bottom + 1, nomPose: !!rn && rn.left >= r.left - 1 && rn.right <= r.right + 1 };
      c.scrollTop = avant;
      return o;
    });
    for (const [nomF, w, h, tel] of FORMATS_BIB) {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel });
      const g = await c.newPage();
      g.erreurs = [];
      g.on('pageerror', (e) => g.erreurs.push(e.message));
      g.on('console', (m) => { if (m.type() === 'error') g.erreurs.push(m.text().slice(0, 300)); });
      await g.goto(`${BASE}/bibliotheque-acier`);
      await attendreBib(g);
      for (const r of RUBRIQUES_ACIER) {
        await g.click(`.bib-m[data-rubrique="${r}"]`);
        await calmeBib(g);
        const m = await mesureBib(g);
        const url = new URL(g.url());
        verif(`bibliothèque acier ${nomF} · ${r} : la rubrique s'ouvre, son titre et son adresse (?rubrique=${r})`, m.titre.length > 3 && url.searchParams.get('rubrique') === r, `${m.titre} · ${url.search}`);
        verif(`bibliothèque acier ${nomF} · ${r} : rien ne déborde, aucun nom coupé`, !m.deborde && !m.coupes.length, m.coupes.join(', '));
        if (r === 'profils') {
          const t = await tableauProfils(g);
          verif(`bibliothèque acier ${nomF} · profils : le tableau des profilés (${t.lignes} lignes) défile DANS son cadre, l'en-tête et les noms restent posés ; ${tel ? 'le cadre est dans la page' : 'le cadre et la bande restent dans l\'écran'}`,
            t.defile && t.descendu && t.lignes >= 100 && t.entete && t.nomPose && (tel ? m.cadre[1] > 120 : (m.cadreDansEcran && m.bandeVisible)), JSON.stringify({ t, m }));
        } else if (!tel) {
          verif(`bibliothèque acier ${nomF} · ${r} : le tableau se voit d'un coup (son cadre ne défile pas) et tient dans l'écran avec la bande`,
            !m.defileY && !m.defileX && m.cadreDansEcran && m.bandeVisible, JSON.stringify(m));
        } else {
          verif(`bibliothèque acier ${nomF} · ${r} : le cadre du tableau et la bande sont dans la page`, m.cadre[0] > 200 && m.cadre[1] > 120, JSON.stringify(m.cadre));
        }
        verif(`bibliothèque acier ${nomF} · ${r} : la case allumée est visible, la valeur retenue est écrite`, m.curVisible && /\d/.test(m.valeur || ''), JSON.stringify({ cur: m.curVisible, v: m.valeur }));
        await g.screenshot({ path: path.join(OUT, `site_bib_acier_${r}_${nomF}.png`) });
      }
      await g.click('.bib-m[data-rubrique="profils"]');
      await calmeBib(g);
      if (nomF === 'pc') {
        // une ligne ouvre son profilé
        await g.click('#bib-tableau tr[data-sel=\'{"profil": "IPE 300"}\'] th.nm');
        await calmeBib(g);
        const l = await g.evaluate(() => ({ v: document.querySelector('.bib-val').textContent, on: document.querySelector('#bib-tableau tr.on th.nm')?.textContent, url: location.search }));
        verif('bibliothèque acier pc · profils : un clic sur la ligne IPE 300 l\'ouvre — la ligne s\'allume, la bande et l\'adresse le nomment', /IPE 300/.test(l.v) && l.on === 'IPE 300' && /profil=IPE(\+|%20)300/.test(l.url), JSON.stringify(l));
        // un en-tête trie ; un second clic renverse
        await g.click('#bib-tableau thead th[data-sel*=\'"tri": "Iy"\']');
        await calmeBib(g);
        const t1 = await g.evaluate(() => ({ on: document.querySelector('#bib-tableau thead th.on')?.dataset.sel, premier: document.querySelector('#bib-tableau tbody th.nm')?.textContent, url: location.search }));
        await g.click('#bib-tableau thead th.on');
        await calmeBib(g);
        const t2 = await g.evaluate(() => ({ premier: document.querySelector('#bib-tableau tbody th.nm')?.textContent, url: location.search }));
        verif('bibliothèque acier pc · profils : un clic sur l\'en-tête I_y trie (colonne marquée, adresse tri=Iy) ; un second renverse l\'ordre',
          /"tri": "Iy"/.test(t1.on || '') && /tri=Iy/.test(t1.url) && /ordre=decroissant/.test(t2.url) && t1.premier !== t2.premier, JSON.stringify({ t1, t2 }));
        // une pastille retire sa famille
        await g.click('.bib-puces button[title="Poutrelles HE A"]');
        await calmeBib(g);
        const f = await g.evaluate(() => ({ pastille: document.querySelector('.bib-puces button[title="Poutrelles HE A"]')?.getAttribute('aria-pressed'), url: decodeURIComponent(location.search),
          hea: [...document.querySelectorAll('#bib-tableau tbody td.fa')].some((x) => x.textContent === 'HEA') }));
        verif('bibliothèque acier pc · profils : la pastille HEA retire la famille (pastille éteinte, plus aucune ligne HEA, l\'adresse suit)', f.pastille === 'false' && !f.hea && !/familles=[^&]*\bHEA\b/.test(f.url), JSON.stringify(f));
        // un filtre : I_y ≥ 8 000 cm⁴ → le plus léger qui le satisfait
        await g.selectOption('.bib-barre select[data-param="f1"]', 'Iy');
        await calmeBib(g);
        await g.fill('.bib-barre input[data-param="min1"]', '8000');
        await g.press('.bib-barre input[data-param="min1"]', 'Enter');
        await calmeBib(g);
        const fl = await g.evaluate(() => ({ leger: document.querySelector('#bib-bande .bib-leger')?.textContent.replace(/\s+/g, ' ').trim(), etoiles: document.querySelectorAll('#bib-tableau .mk').length, url: location.search }));
        verif('bibliothèque acier pc · profils : le filtre I_y ≥ 8 000 cm⁴ nomme le plus léger qui le satisfait (★ dans la bande et sur sa ligne), l\'adresse garde le filtre',
          /plus léger/.test(fl.leger || '') && fl.etoiles === 1 && /f1=Iy/.test(fl.url) && /min1=8000/.test(fl.url), JSON.stringify(fl));
        // la fiche du profilé : la section à l'échelle, ses propriétés
        await g.click('[data-pourquoi]');
        await g.waitForTimeout(150);
        const fi = await g.evaluate(() => { const p = document.getElementById('bib-pourquoi'), s = p.querySelector('svg#bib_sec'), r = s ? s.getBoundingClientRect() : null;
          return { ouvert: !p.hidden && p.checkVisibility(), dessin: r ? [Math.round(r.width), Math.round(r.height)] : null, proprietes: p.querySelectorAll('.bib-fiche td.s').length }; });
        verif('bibliothèque acier pc · profils : « Fiche du profilé » montre la section à l\'échelle (le dessin du catalogue) et toutes ses propriétés',
          fi.ouvert && fi.dessin && fi.dessin[0] > 250 && fi.dessin[1] > 200 && fi.proprietes >= 20, JSON.stringify(fi));
        await g.screenshot({ path: path.join(OUT, 'site_bib_acier_fiche_pc.png') });
        await g.keyboard.press('Escape');
        await g.waitForTimeout(80);
        verif('bibliothèque acier pc · profils : Échap ferme la fiche', await g.$eval('#bib-pourquoi', (p) => p.hidden));
        // « Colonnes » : il reste ouvert d'un clic à l'autre, la colonne s'ajoute, un clic dehors le referme
        await g.click('.bib-plus > summary');
        await g.click('.bib-plus-c button[title^="Wpl,z"]');
        await calmeBib(g);
        const co = await g.evaluate(() => { const d = document.querySelector('.bib-plus'), c = d.querySelector('.bib-plus-c').getBoundingClientRect();
          return { ouvert: d.open, dedans: c.left >= 0 && c.right <= innerWidth + 1 && c.bottom <= innerHeight + 1, colonne: !!document.querySelector('#bib-tableau thead th[data-sel*=\'"tri": "Wpl_z"\']') }; });
        await g.screenshot({ path: path.join(OUT, 'site_bib_acier_colonnes_pc.png') });
        await g.click('#bib-bande .bib-bande-t');
        const ferme = await g.$eval('.bib-plus', (d) => !d.open);
        verif('bibliothèque acier pc · profils : le menu « Colonnes » reste ouvert après un clic (data-garde), dans l\'écran ; W_pl,z s\'ajoute au tableau ; un clic dehors le referme',
          co.ouvert && co.dedans && co.colonne && ferme, JSON.stringify({ co, ferme }));
        // l'adresse rouvre le même état
        const v0 = await g.$eval('.bib-val', (x) => x.textContent);
        await g.goto(g.url());
        await attendreBib(g);
        const relu = await g.evaluate(() => ({ v: document.querySelector('.bib-val').textContent, f1: document.querySelector('.bib-barre select[data-param="f1"]')?.value,
          min1: document.querySelector('.bib-barre input[data-param="min1"]')?.value, tri: document.querySelector('.bib-barre select[data-param="tri"]')?.value }));
        verif('bibliothèque acier pc · profils : l\'adresse rouverte redonne le même état (profilé, filtre, tri)', relu.v === v0 && relu.f1 === 'Iy' && relu.min1 === '8000' && relu.tri === 'Iy', JSON.stringify({ v0, relu }));
      }
      if (tel) {
        await g.click('#bib-tableau tbody tr:nth-child(3) th.nm');
        await calmeBib(g);
        const tv = await g.evaluate(() => ({ v: document.querySelector('.bib-val').textContent, on: document.querySelector('#bib-tableau tr.on th.nm')?.textContent }));
        await g.click('[data-pourquoi]');
        await g.waitForTimeout(150);
        const ft = await g.evaluate(() => { const p = document.querySelector('#bib-pourquoi .bib-pq-c').getBoundingClientRect(), s = document.querySelector('#bib-pourquoi svg#bib_sec')?.getBoundingClientRect();
          return { large: Math.round(p.width), dessin: s ? Math.round(s.width) : 0 }; });
        verif('bibliothèque acier téléphone · profils : un doigt ouvre le profilé de sa ligne (la bande le nomme), sa fiche s\'ouvre plein écran avec la section',
          tv.on && tv.v.replace(/\s/g, '').startsWith(tv.on.replace(/\s/g, '')) && ft.large >= w - 2 && ft.dessin > 200, JSON.stringify({ tv, ft }));
        await g.screenshot({ path: path.join(OUT, 'site_bib_acier_fiche_tel.png') });
        await g.click('#bib-pourquoi [data-fermer-pourquoi]');
      }
      verif(`bibliothèque acier ${nomF} : aucune erreur JavaScript`, g.erreurs.length === 0, g.erreurs.join(' | ').slice(0, 300));
      await c.close();
    }

    // une ANCIENNE note du Catalogue des profilés (module masqué depuis BIB5) s'ouvre encore : « Reprendre » la propose à
    // l'accueil, sa page la rouvre, son moteur la relit (le profilé ouvert, sa recherche)
    {
      const c = await b.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
      const g = await c.newPage();
      g.erreurs = [];
      g.on('pageerror', (e) => g.erreurs.push(e.message));
      await g.goto(`${BASE}/index.html`);
      await g.evaluate(() => {
        const notes = JSON.parse(localStorage.getItem('stab.notes.v1') || '{}');
        notes['ancien-catalogue'] = { id: 'ancien-catalogue', module: 'catalogue', nom: 'Ancienne note du catalogue', chantier: 'Essai BIB5', dossier: '', indice: '',
          cree: Date.now(), modifie: Date.now(), resume: null, vignette: '',
          etat: { version: 1, recherches: [{ nom: 'Poteaux HEB', recherche: 'HEB 3', fiche: 'HEB 300' }], actif: 0 } };
        localStorage.setItem('stab.notes.v1', JSON.stringify(notes));
      });
      await g.reload();
      await g.waitForTimeout(300);
      const lien = await g.$eval('#recentes', (r) => [...r.querySelectorAll('a')].map((a) => a.getAttribute('href')).find((h) => /note=ancien-catalogue/.test(h)) || '');
      await g.goto(`${BASE}/${lien}`);
      await pretEcran(g);
      await g.waitForFunction(() => { const r = document.getElementById('resultat'); return r && !r.querySelector('.attente') && /HEB 300/.test(r.textContent); }, null, { timeout: 60000 }).catch(() => {});
      const o = await g.evaluate((erreur) => ({ erreur, resultat: /HEB 300/.test(document.getElementById('resultat').textContent),
        lignes: document.querySelectorAll('.cat-t tbody tr').length, nom: document.getElementById('nom-note')?.value }), !!(await echecEcran(g)));
      verif('catalogue masqué : une ancienne note du Catalogue des profilés est proposée par « Reprendre » et s\'ouvre encore dans sa page (HEB 300 ouvert, le tableau, son nom)',
        /^catalogue-profiles\?note=ancien-catalogue$/.test(lien) && !o.erreur && o.resultat && o.lignes > 0 && o.nom === 'Ancienne note du catalogue' && !g.erreurs.length,
        JSON.stringify({ lien, o, erreurs: g.erreurs }));
      await g.screenshot({ path: path.join(OUT, 'site_catalogue_ancienne_note.png') });
      await c.close();
    }
  }

  // ---------------------------------------------- la Bibliothèque « Autres » (docs/site/BIBLIOTHEQUE_AUTRES.md) : trois OUTILS
  // Le cadre ne porte pas un tableau mais un outil de RDM : le dessin, les données, les résultats, la feuille de calcul.
  // Aux quatre formats, CE QUE L'ŒIL VOIT : rien ne dépasse de l'écran (le tableau des résultats des Sections portait la
  // page à 482 px pour 390, le 11/11/2026) ; le dessin se voit et ses textes font 10 px au moins À L'ÉCRAN (ils en
  // faisaient 6 à 8 : un plan de 640 unités réduit à 360 px) ; au doigt, toute cible fait 40 px au moins et ce qui se
  // glisse ne fait pas défiler la page ; la bande dit une valeur. Filtre « bibliotheque » ou, seul, « bibliotheque_autres ».
  // Puis les GESTES (PC) : glisser un élément de la
  // section, une charge de la poutre, cliquer un bord de la plaque, un cas REFUSÉ sans valeur, la feuille à l'impression.
  if (!FILTRE || ['bibliotheque', 'bibliotheque_autres'].includes(FILTRE)) {
    const OUTILS = [['poutres', /kN·m/], ['plaques', /mm/], ['sections', /cm⁴/]];
    const attendreAutres = async (pg) => {
      await pretEcran(pg);
      await pg.waitForFunction(() => {
        const c = document.getElementById('bib-corps');
        return c && c.getAttribute('aria-busy') !== 'true' && document.querySelector('#bib-tableau .rdm');
      }, null, { timeout: 240000 });
    };
    const calmeAutres = async (pg) => { await pg.waitForTimeout(80); await attendreAutres(pg); await pg.waitForTimeout(120); };
    const adresse = (pg) => pg.evaluate(() => Object.fromEntries(new URLSearchParams(location.search)));
    const mesureAutres = (pg, large) => pg.evaluate((large) => {
      const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      // ce qui dépasse de l'écran sans qu'un ancêtre qui défile (et tient, lui, dans l'écran) ne l'excuse
      const hors = [];
      for (const e of document.querySelectorAll('#bib-corps *')) {
        if (!vis(e)) continue;
        const r = e.getBoundingClientRect();
        if (!r.width || r.right <= large + 1) continue;
        let q = e.parentElement, excuse = false;
        while (q && q !== document.body) {
          if (/auto|scroll|hidden/.test(getComputedStyle(q).overflowX) && q.getBoundingClientRect().right <= large + 1) { excuse = true; break; }
          q = q.parentElement;
        }
        if (!excuse) hors.push(`${e.tagName.toLowerCase()}.${String(e.getAttribute('class') || '').slice(0, 24)} → ${Math.round(r.right)}`);
      }
      const svg = document.querySelector('#bib-tableau .rdm-svg');
      const rs = svg ? svg.getBoundingClientRect() : { width: 0, height: 0 };
      const k = svg ? rs.width / svg.viewBox.baseVal.width : 0;
      const textes = svg ? [...svg.querySelectorAll('text')].filter(vis).map((t) => [t.textContent.trim().slice(0, 16), +(parseFloat(getComputedStyle(t).fontSize) * k).toFixed(1)]) : [];
      const cibles = [...document.querySelectorAll('#bib-tableau button, #bib-tableau .field')].filter(vis)
        .map((e) => [(e.textContent || e.getAttribute('aria-label') || '').trim().slice(0, 14), Math.round(e.getBoundingClientRect().height)]);
      const prises = [...document.querySelectorAll('#bib-tableau [data-glisser]')].filter(vis).map((e) => getComputedStyle(e).touchAction);
      const res = document.querySelector('#bib-tableau .rdm-res');
      // ce qui dépasse de SA COLONNE (le dessin, les données, la feuille) sans être dans un cadre qui défile seul : à la
      // tablette en paysage, le tableau des résultats des Sections sortait de 132 px de sa colonne — la remarque de droite
      // était coupée et le cadre entier défilait en largeur (vu sur une photo le 11/11/2026 ; « rien ne dépasse de l'écran »
      // l'excusait, le cadre étant lui-même dans l'écran)
      const colonne = [];
      for (const col of document.querySelectorAll('#bib-tableau .rdm > .rdm-g, #bib-tableau .rdm > .rdm-d, #bib-tableau .rdm > .rdm-f')) {
        const rc = col.getBoundingClientRect();
        for (const e of col.querySelectorAll('*')) {
          if (!vis(e)) continue;
          const r = e.getBoundingClientRect();
          if (!r.width || r.right <= rc.right + 1) continue;
          let q = e.parentElement, excuse = false;
          while (q && q !== col) { if (/auto|scroll/.test(getComputedStyle(q).overflowX)) { excuse = true; break; } q = q.parentElement; }
          if (!excuse) colonne.push(`${col.className.split(' ')[0]} › ${e.tagName.toLowerCase()}.${String(e.getAttribute('class') || '').slice(0, 18)} +${Math.round(r.right - rc.right)}`);
        }
      }
      const cadre = document.getElementById('bib-tableau');
      return { page: document.documentElement.scrollWidth, hors: hors.slice(0, 6), colonne: colonne.slice(0, 5), nColonne: colonne.length,
        defileX: cadre.scrollWidth > cadre.clientWidth + 1,
        svg: [Math.round(rs.width), Math.round(rs.height)], svgVisible: vis(svg), nTextes: textes.length,
        petits: textes.filter((t) => t[1] < 10), basses: cibles.filter((c) => c[1] < 40).slice(0, 6), nCibles: cibles.length,
        prises: prises.length, prisesLibres: prises.filter((t) => t !== 'none').length,
        lignes: res && vis(res) ? res.querySelectorAll('tr').length : 0, feuille: document.querySelectorAll('#bib-tableau .rdm-fc tr').length,
        valeur: (document.querySelector('.bib-val') || {}).textContent || '', titre: document.getElementById('bib-titre').textContent,
        menu: [...document.querySelectorAll('.bib-m')].filter(vis).length, allume: (document.querySelector('.bib-m.on') || {}).dataset?.rubrique };
    }, large);
    for (const [nomF, w, h, tel] of FORMATS_BIB) {
      const doigt = nomF !== 'pc';                         // tablettes et téléphone : l'écran se touche (pointer: coarse)
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: doigt });
      const g = await c.newPage();
      g.erreurs = [];
      g.on('pageerror', (e) => g.erreurs.push(e.message));
      g.on('console', (m) => { if (m.type() === 'error') g.erreurs.push(m.text().slice(0, 300)); });
      await g.goto(`${BASE}/bibliotheque-autres`);
      await attendreAutres(g);
      for (const [r, unite] of OUTILS) {
        await g.click(`.bib-m[data-rubrique="${r}"]`);
        await calmeAutres(g);
        const m = await mesureAutres(g, w);
        const url = new URL(g.url());
        verif(`bibliothèque autres ${nomF} · ${r} : l'outil s'ouvre, son titre, son adresse (?rubrique=${r}), les trois entrées du menu et la sienne allumée`,
          m.titre.length > 3 && url.searchParams.get('rubrique') === r && m.menu === 3 && m.allume === r, JSON.stringify({ t: m.titre, u: url.search, menu: m.menu, on: m.allume }));
        verif(`bibliothèque autres ${nomF} · ${r} : rien ne dépasse de l'écran (${w} px)`, m.page <= w + 1 && !m.hors.length, JSON.stringify({ page: m.page, hors: m.hors }));
        verif(`bibliothèque autres ${nomF} · ${r} : rien ne dépasse de sa colonne (dessin, données et résultats, feuille) et le cadre ne défile pas en largeur`,
          !m.nColonne && !m.defileX, JSON.stringify({ n: m.nColonne, colonne: m.colonne, defileX: m.defileX }));
        verif(`bibliothèque autres ${nomF} · ${r} : le dessin se voit (${m.svg.join(' × ')} px) et chacun de ses ${m.nTextes} textes fait 10 px au moins à l'écran`,
          m.svgVisible && m.svg[0] >= 300 && m.svg[1] >= 150 && m.nTextes >= 4 && !m.petits.length, JSON.stringify({ svg: m.svg, petits: m.petits }));
        verif(`bibliothèque autres ${nomF} · ${r} : les résultats (${m.lignes} lignes) et la feuille de calcul (${m.feuille} lignes) sont là ; la bande dit une valeur (${m.valeur.trim()})`,
          m.lignes >= 3 && m.feuille >= 8 && /\d/.test(m.valeur) && unite.test(m.valeur), JSON.stringify({ l: m.lignes, f: m.feuille, v: m.valeur }));
        if (doigt) {
          verif(`bibliothèque autres ${nomF} · ${r} : au doigt, chacune des ${m.nCibles} cibles (boutons, champs) fait 40 px de haut au moins`,
            m.nCibles >= 5 && !m.basses.length, JSON.stringify(m.basses));
          if (r !== 'plaques') verif(`bibliothèque autres ${nomF} · ${r} : ce qui se glisse sur le dessin (${m.prises}) ne fait pas défiler la page sous le doigt (touch-action: none)`,
            m.prises >= 1 && m.prisesLibres === 0, JSON.stringify({ prises: m.prises, libres: m.prisesLibres }));
        }
        await g.screenshot({ path: path.join(OUT, `site_bib_autres_${r}_${nomF}.png`) });
      }
      if (nomF === 'pc') {
        // ---- sections : glisser l'élément 2 (le plat posé sur le HEA 200), un clic choisit, ajouter, retirer, un exemple
        const pasEcran = (sel) => g.evaluate((sel) => JSON.parse(document.querySelector(sel).dataset.glisser).ech * document.querySelector('.rdm-svg').getScreenCTM().a, sel);
        let bb = await g.locator('.rdm-svg g.elt[data-element="2"] > path').boundingBox();
        let ech = await pasEcran('.rdm-svg g.elt[data-element="2"]');
        await g.mouse.move(bb.x + bb.width / 2 + 40, bb.y + bb.height / 2);
        await g.mouse.down();
        await g.mouse.move(bb.x + bb.width / 2 + 40 + 30 * ech, bb.y + bb.height / 2 - 20 * ech, { steps: 8 });
        const pendant = await g.evaluate(() => ({ t: (document.querySelector('.rdm-svg text.glis-t') || {}).textContent || '',
          suit: document.querySelector('.rdm-svg g.elt[data-element="2"]').classList.contains('glisse') }));
        await g.mouse.up();
        await calmeAutres(g);
        let q = await adresse(g);
        verif('bibliothèque autres pc · sections : glisser le plat de (+30 ; +20) mm → y2 = 30, z2 = 121, il devient l\'élément choisi ; pendant le geste il suit et une étiquette dit sa position',
          q.y2 === '30' && q.z2 === '121' && q.sel === '2' && pendant.suit && /30/.test(pendant.t) && /121/.test(pendant.t), JSON.stringify({ q, pendant }));
        bb = await g.locator('.rdm-svg g.elt[data-element="1"] > path').boundingBox();
        await g.mouse.click(bb.x + bb.width / 2, bb.y + bb.height - 6);
        await calmeAutres(g);
        q = await adresse(g);
        verif('bibliothèque autres pc · sections : un clic choisit le profilé sans le déplacer', q.sel === '1' && (q.y1 || '0') === '0' && (q.z1 || '0') === '0', JSON.stringify(q));
        await g.locator('.rdm-elements .rdm-plus button', { hasText: 'Plat' }).first().click();
        await calmeAutres(g);
        q = await adresse(g);
        const pose = q.n === '3' && q.sel === '3';
        await g.locator('.rdm-el.on button[aria-label^="Retirer"]').click();
        await calmeAutres(g);
        q = await adresse(g);
        verif('bibliothèque autres pc · sections : « + Plat » ajoute un troisième élément, choisi ; « Retirer » le retire et l\'adresse n\'en garde rien',
          pose && q.n === '2' && !Object.keys(q).some((k) => /^[a-z]+3(_\d+)?$/.test(k)), JSON.stringify(q));
        await g.locator('.rdm-ex button', { hasText: 'Demi-HEA' }).click();
        await calmeAutres(g);
        q = await adresse(g);
        verif('bibliothèque autres pc · sections : l\'exemple « Demi-HEA » remplace la section (un HEA 300 coupé à mi-hauteur, seul)', q.n === '1' && q.p1 === 'HEA 300' && q.c1 === 'haut' && !('k2' in q), JSON.stringify(q));
        await g.locator('.rdm-el-f input[data-param="y1"]').fill('25');
        await g.keyboard.press('Tab');
        await calmeAutres(g);
        const foyer = await g.evaluate(() => document.activeElement && document.activeElement.dataset.param);
        verif('bibliothèque autres pc · sections : saisir y1 = 25 puis Tab → la valeur est prise et le champ suivant (z1) garde le foyer', (await adresse(g)).y1 === '25' && foyer === 'z1', String(foyer));
        // ---- poutres : glisser la charge ponctuelle, la borne, la ligne « flèche »
        await g.goto(`${BASE}/bibliotheque-autres?rubrique=poutres&n=2&t2=P&a2=2`);
        await attendreAutres(g);
        bb = await g.locator('.rdm-svg g.charge[data-charge="2"] rect.prise').boundingBox();
        ech = await pasEcran('.rdm-svg g.charge[data-charge="2"]');
        await g.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2);
        await g.mouse.down();
        await g.mouse.move(bb.x + bb.width / 2 + 1.5 * ech, bb.y + bb.height / 2 + 4, { steps: 8 });
        await g.mouse.up();
        await calmeAutres(g);
        const a1 = (await adresse(g)).a2;
        bb = await g.locator('.rdm-svg g.charge[data-charge="2"] rect.prise').boundingBox();
        await g.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2);
        await g.mouse.down();
        await g.mouse.move(bb.x + bb.width / 2 + 9 * ech, bb.y + bb.height / 2, { steps: 8 });
        await g.mouse.up();
        await calmeAutres(g);
        verif('bibliothèque autres pc · poutres : glisser la charge ponctuelle de +1,5 m → a2 = 3,5 ; au-delà de l\'appui elle s\'arrête au bout de la poutre (a2 = 6)',
          a1 === '3.5' && (await adresse(g)).a2 === '6', JSON.stringify({ a1, a2: (await adresse(g)).a2 }));
        await g.locator('.rdm-res tr[data-sel]', { hasText: 'flèche' }).click();
        await calmeAutres(g);
        verif('bibliothèque autres pc · poutres : un clic sur la ligne « flèche » la met dans la bande (mm) et dans l\'adresse (g=f)',
          /mm/.test(await g.locator('.bib-val').textContent()) && (await adresse(g)).g === 'f', await g.locator('.bib-val').textContent());
        // ---- plaques : un clic sur un bord, un cas REFUSÉ (sans valeur), un préréglage
        await g.goto(`${BASE}/bibliotheque-autres?rubrique=plaques`);
        await attendreAutres(g);
        await g.locator('.rdm-svg g.bord[data-bord="y0"] rect.b-zone').click({ position: { x: 60, y: 8 } });
        await calmeAutres(g);
        const b1 = (await adresse(g)).by0;
        await g.locator('.rdm-svg g.bord[data-bord="y0"] rect.b-zone').click({ position: { x: 60, y: 8 } });
        await calmeAutres(g);
        await g.locator('.rdm-svg g.bord[data-bord="x0"] rect.b-zone').click({ position: { x: 8, y: 60 } });
        await calmeAutres(g);
        q = await adresse(g);
        const refus = { n: await g.locator('.rdm-refus').count(), vu: await g.locator('.rdm-refus').first().isVisible().catch(() => false), v: (await g.locator('.bib-val').textContent()).trim() };
        verif('bibliothèque autres pc · plaques : un clic sur le bord bas le fait encastré, un second libre ; bas libre + gauche encastré est REFUSÉ à l\'écran, dit, sans aucune valeur (la bande montre « — »)',
          b1 === 'E' && q.by0 === 'L' && q.bx0 === 'E' && refus.n === 1 && refus.vu && refus.v.startsWith('—') && !/\d/.test(refus.v), JSON.stringify({ b1, q, refus }));
        await g.screenshot({ path: path.join(OUT, 'site_bib_autres_plaques_refus_pc.png') });
        await g.locator('.rdm-bords-b .rdm-plus button', { hasText: '4 encastrés' }).click();
        await calmeAutres(g);
        verif('bibliothèque autres pc · plaques : le préréglage « 4 encastrés » rétablit un cas calculé', (await g.locator('.rdm-refus').count()) === 0 && (await adresse(g)).byb === 'E' && /\d/.test(await g.locator('.bib-val').textContent()));
        // ---- l'impression : la feuille seule
        await g.emulateMedia({ media: 'print' });
        const imp = await g.evaluate(() => { const d = (s) => { const e = document.querySelector(s); return e ? getComputedStyle(e).display : 'absent'; };
          return { menu: d('.bib-menu'), barre: d('.bib-barre'), entete: d('.rdm-imp'), feuille: document.querySelectorAll('.rdm-fc tr').length, premier: document.querySelector('.rdm').firstElementChild.className }; });
        await g.emulateMedia({ media: 'screen' });
        verif('bibliothèque autres pc · impression : le menu et la barre disparaissent, l\'en-tête de la feuille apparaît EN TÊTE, la feuille de calcul reste',
          imp.menu === 'none' && imp.barre === 'none' && imp.entete !== 'none' && imp.entete !== 'absent' && /rdm-imp/.test(imp.premier) && imp.feuille >= 8, JSON.stringify(imp));
      }
      verif(`bibliothèque autres ${nomF} : aucune erreur JavaScript`, !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
      await c.close();
    }
  }

  // ---------------------------------------------------------------- la NOTE PDF de chaque module qui la fait (filtre « note_pdf »)
  // Décision de l'ingénieur du 09/11/2026 : les notes PDF de Streamlit passent sur le site (docs/site/MODULES_SITE.md
  // § 4 quinquies). Les modules sont ceux dont la page le DIT (body data-note-pdf="1", écrit par construire.NOTE_PDF :
  // une source) ; chacun s'ouvre sur sa note « riche » (construire.NOTES_RELEVE, la même que le relevé du paquet de la
  // note : plusieurs éléments), semée dans ce navigateur. Au PC (1440), à la tablette (1024) et au téléphone (iPhone 13) :
  // le bouton « Note PDF » est VISIBLE et actif (mesuré), le clic → le moteur du navigateur → un téléchargement
  // (événement download) ; le fichier est un PDF (%PDF-), ses pages (pdfinfo), et son texte (pdftotext) porte le NOM de
  // chaque élément, le taux de chaque élément que la liste de l'écran affiche, et les chiffres clés que le cadre Résultat
  // MONTRE (parité écran ↔ PDF dans le vrai navigateur) ; le message « … téléchargé » se voit ; aucune erreur JavaScript.
  // Le temps de préparation (du clic au téléchargement) est dit.
  if (!FILTRE || FILTRE === 'note_pdf') {
    const fs = require('fs');
    const cp = require('child_process');
    await p.goto(`${BASE}/`);
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    // les modules qui font leur note : ce que leur page dit (construire.NOTE_PDF)
    const avec = [];
    for (const k of Object.keys(MODS)) {
      const html = await (await p.request.get(`${BASE}/${MODS[k].page}`)).text();
      if (/<body[^>]*data-note-pdf="1"/.test(html)) avec.push(k);
    }
    // les notes riches : le code de construire.NOTES_RELEVE, joué par le moteur du dépôt (les mêmes saisies que l'écran)
    const PY = process.env.PYTHON || (fs.existsSync('/usr/bin/python3.12') ? '/usr/bin/python3.12' : 'python3');
    const ETATS = [
      'import json, sys',
      'sys.path.insert(0, sys.argv[1] + "/site/web")',
      'import construire as K',
      'out = {}',
      'for k in K.NOTE_PDF:',
      '    ns = {}',
      '    exec(K.NOTES_RELEVE[k], ns)',
      '    out[k] = ns["etat"]',
      'print(json.dumps({"note_pdf": list(K.NOTE_PDF), "etats": out}))',
    ].join('\n');
    const fe = cp.spawnSync(PY, ['-c', ETATS, path.join(__dirname, '..', '..')], { encoding: 'utf8', cwd: path.join(__dirname, '..', '..'), maxBuffer: 1 << 26 });
    let riches = null;
    try { riches = JSON.parse(fe.stdout.trim().split('\n').pop()); } catch (e) { /* le contrôle rougit */ }
    verif(`note PDF : les pages qui le disent sont celles de construire.NOTE_PDF (${avec.join(', ')}), chacune a sa note riche`,
      riches && avec.length >= 1 && JSON.stringify([...avec].sort()) === JSON.stringify([...riches.note_pdf].sort())
      && avec.every((k) => riches.etats[k]), String(fe.stderr || '').slice(-300) || JSON.stringify(riches && riches.note_pdf));
    const norm = (t) => String(t || '').replace(/[   ]/g, ' ').replace(/\s+/g, ' ').replace(/(\d) (?=\d)/g, '$1');
    // les chiffres clés que le cadre Résultat MONTRE : la hauteur minimale et les sections d'acier (béton armé, établi
    // compact), sinon le taux de tête (η)
    // (le chiffre lu, et la forme sous laquelle la note l'écrit : « 58,5 cm », « = 1194 mm 2 », « 804 mm² », « 14 % »).
    // Le chip du verdict arrondit η à l'unité (commun.pct) ; une note qui imprime η à la décimale (les cornières :
    // « η max 95,8 % ») le porte à la précision du chip — son η max est à moins d'une unité du chip (10/11/2026)
    const CHIFFRES = [
      ['h min', /hmin\s*=\s*(\d+(?:,\d+)?)\s*cm/, (v) => `${v} cm`], ['A s,requis', /As,requis\s*=\s*(\d+)/, (v) => `= ${v} mm`],
      ['A s,posé', /As,posé\s*=\s*(\d+)\s*mm²/, (v) => `${v} mm²`], ['A s,néc', /As,néc\s*=\s*(\d+)/, (v) => `= ${v} mm`],
      ['A s', /As\s*=\s*(\d+)\s*mm²/, (v) => `${v} mm²`], ['η', /η\s*(\d+)\s*%/, (v) => `${v} %`],
    ];
    const FORMATS_NOTE = [
      ['pc', { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 }],
      ['tablette', { viewport: { width: 1024, height: 768 }, deviceScaleFactor: 1, hasTouch: true }],
      ['telephone', { ...devices['iPhone 13'] }],
    ];
    const temps = [];
    for (const [fmt, opts] of FORMATS_NOTE) {
      const c = await b.newContext({ ...opts, acceptDownloads: true, locale: 'fr-BE' });
      const g = await c.newPage();
      g.erreurs = [];
      g.on('pageerror', (e) => g.erreurs.push(e.message));
      g.on('console', (m) => { if (m.type() === 'error') g.erreurs.push(m.text().slice(0, 300)); });
      await ouvrir(g, `${BASE}/`);
      const tel = fmt === 'telephone';
      for (const k of avec) {
        if (!riches || !riches.etats[k]) continue;
        const id = `note-pdf-${k}`;
        await g.evaluate(([id, k, etat]) => {
          let notes = {};
          try { notes = JSON.parse(localStorage.getItem('stab.notes.v1') || '{}') || {}; } catch (e) { notes = {}; }
          notes[id] = { id, module: k, nom: 'Contrôle note PDF', chantier: 'Essai nuit', dossier: '', indice: 'B', cree: Date.now(), modifie: Date.now(),
                        etat, resume: null, vignette: '' };
          localStorage.setItem('stab.notes.v1', JSON.stringify(notes));
        }, [id, k, riches.etats[k]]);
        await pretEcran(g, { apres: await ouvrir(g, `${BASE}/${MODS[k].page}?note=${id}`) });
        await g.waitForFunction(() => { const r = document.getElementById('resultat'); return r && !r.querySelector('.attente') && r.textContent.trim().length > 20; }, null, { timeout: 60000 }).catch(() => {});
        if (tel) { await g.click('.mtabs [data-onglet="resultat"]'); await g.waitForTimeout(300); }
        // ce que l'écran montre : les noms des éléments (la liste de l'établi compact, ou celle « Dans cette note »), le taux
        // de chaque élément de la liste, et le texte VISIBLE du cadre Résultat
        const ecran = await g.evaluate(() => {
          const premier = (e) => [...e.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join('').trim();
          const noms = [...document.querySelectorAll('.et-elt-liste .et-elt-i .nom')].map((e) => e.textContent.trim())
            .concat([...document.querySelectorAll('#donnees .elements .ni .t')].map((e) => (e.querySelector('.nom') ? e.querySelector('.nom').textContent.trim() : premier(e))));
          const pcts = [...document.querySelectorAll('#donnees .elements .ni .pct')].map((e) => e.textContent.trim());
          const r = document.getElementById('resultat');
          const vu = r.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          return { noms, pcts, texte: vu ? r.innerText : '', vu };
        });
        const tEcran = norm(ecran.texte);
        const chiffres = [];
        for (const [nom, re, forme] of CHIFFRES) { const m = tEcran.match(re); if (m) chiffres.push([nom, forme(m[1])]); }
        // le bouton : là où l'œil le trouve — la barre du haut (PC, tablette), la barre du bas (téléphone) ; et au
        // téléphone, l'entrée du menu ⋯ est active elle aussi
        const sel = tel ? '.mbar [data-action="pdf"]' : '.top [data-action="pdf"]';
        const bouton = await g.evaluate((s) => {
          const e = [...document.querySelectorAll(s)].find((x) => x.checkVisibility({ visibilityProperty: true, opacityProperty: true }));
          if (!e) return null;
          const r = e.getBoundingClientRect();
          return { l: Math.round(r.width), h: Math.round(r.height), grise: e.disabled || e.getAttribute('aria-disabled') === 'true',
                   dedans: r.left >= 0 && r.right <= innerWidth + 0.5 && r.top >= 0 && r.bottom <= innerHeight + 0.5 };
        }, sel);
        let menu = null;
        if (tel) {
          await g.click('#menu-m');
          await g.waitForTimeout(200);
          menu = await g.evaluate(() => {
            const e = document.querySelector('#mmenu [data-action="pdf"]');
            const r = e && e.getBoundingClientRect();
            return e ? { vu: e.checkVisibility({ visibilityProperty: true, opacityProperty: true }) && r.width > 40 && r.height > 30,
                         grise: e.getAttribute('aria-disabled') === 'true' || /bientôt/.test(e.textContent) } : null;
          });
          await g.keyboard.press('Escape');
          await g.waitForTimeout(150);
        }
        verif(`note PDF · ${k} · ${fmt} : le bouton « Note PDF » est visible, dans l'écran, à sa taille, actif${tel ? ' (barre du bas, et dans le menu ⋯)' : ''}`,
          bouton && bouton.l >= 30 && bouton.h >= 28 && !bouton.grise && bouton.dedans && (!tel || (menu && menu.vu && !menu.grise)),
          JSON.stringify({ bouton, menu }));
        // le téléchargement, ou le refus du moteur (un message rouge) : le premier des deux — un refus n'attend pas trois minutes
        const t0 = Date.now();
        const attendu = g.waitForEvent('download', { timeout: 180000 }).catch(() => null);
        await g.click(sel);
        const refus = g.waitForFunction(() => { const e = document.getElementById('toast'); return e && e.classList.contains('erreur') && e.checkVisibility(); },
          null, { timeout: 180000 }).then(() => 'refus', () => null);
        const premier = await Promise.race([attendu, refus]);
        const dl = premier && premier !== 'refus' ? premier : null;
        const duree = (Date.now() - t0) / 1000;
        temps.push(`${k}·${fmt} ${duree.toFixed(1)} s`);
        const fpdf = path.join(OUT, `site_note_pdf_${k}_${fmt}.pdf`);
        let tete = '', pages = 0, texte = '';
        if (dl) {
          await dl.saveAs(fpdf);
          tete = fs.readFileSync(fpdf).subarray(0, 5).toString();
          const info = cp.spawnSync('pdfinfo', [fpdf], { encoding: 'utf8' }).stdout || '';
          pages = +((info.match(/Pages:\s+(\d+)/) || [])[1] || 0);
          texte = norm(cp.spawnSync('pdftotext', ['-layout', fpdf, '-'], { encoding: 'utf8', maxBuffer: 1 << 26 }).stdout);
        }
        await g.waitForTimeout(300);
        const t = await g.evaluate(() => { const e = document.getElementById('toast'); return { vu: e.checkVisibility(), texte: e.textContent, erreur: e.classList.contains('erreur') }; });
        verif(`note PDF · ${k} · ${fmt} : le clic télécharge la note (« ${dl ? dl.suggestedFilename() : '—'} », ${pages} pages, ${duree.toFixed(1)} s), un PDF, et le message le dit`,
          dl && /^Note de calcul - .+\.pdf$/.test(dl.suggestedFilename()) && tete === '%PDF-' && pages >= Math.max(2, ecran.noms.length)
          && t.vu && !t.erreur && /téléchargé$/.test(t.texte), JSON.stringify({ nom: dl && dl.suggestedFilename(), tete, pages, toast: t }));
        const absents = ecran.noms.filter((n) => !texte.includes(norm(n)));
        verif(`note PDF · ${k} · ${fmt} : le PDF nomme chacun des ${ecran.noms.length} éléments de la note (${ecran.noms.join(' ; ')})`,
          ecran.noms.length >= 2 && ecran.noms.every((n) => n.length > 0) && !absents.length, JSON.stringify({ absents, noms: ecran.noms }));
        const pctAbsents = ecran.pcts.filter((x) => !texte.includes(norm(x)));
        const etaProche = (v) => [...texte.matchAll(/η\s*(?:max\s*)?(\d+(?:,\d+)?)\s*%/g)].some((m) => Math.abs(parseFloat(m[1].replace(',', '.')) - parseInt(v, 10)) < 1);
        const chAbsents = chiffres.filter(([nom, v]) => !texte.includes(v) && !(nom === 'η' && etaProche(v)));
        verif(`note PDF · ${k} · ${fmt} : parité écran ↔ PDF — ${chiffres.map(([n, v]) => `${n} « ${v} »`).join(', ')}${ecran.pcts.length ? ` ; taux de la liste ${ecran.pcts.join(', ')}` : ''}`,
          ecran.vu && chiffres.length >= 1 && !chAbsents.length && !pctAbsents.length,
          JSON.stringify({ chAbsents, pctAbsents, vu: ecran.vu, ecran: tEcran.slice(0, 200) }));
        // l'identification imprimée est celle de la note du site (lot PDF-fix, 09/11/2026 : les soudures et le plancher
        // écrivaient « indice 0 » sous une note B) — la note semée ici dit indice B, chantier « Essai nuit »
        const indices = [...texte.matchAll(/\bindice\s*:?\s*([A-Z0-9])\b/gi)].map((m) => m[1]);
        verif(`note PDF · ${k} · ${fmt} : le PDF porte l'identification de la note du site — indice B (${indices.length} fois, aucun autre), chantier « Essai nuit »`,
          dl && indices.length >= 1 && indices.every((x) => x === 'B') && texte.includes('Essai nuit'), JSON.stringify({ indices: [...new Set(indices)] }));
        await g.screenshot({ path: path.join(OUT, `site_note_pdf_${k}_${fmt}.png`) });
      }
      verif(`note PDF · ${fmt} : aucune erreur JavaScript`, g.erreurs.length === 0, g.erreurs.join(' | ').slice(0, 400));
      await c.close();
    }
    console.log(`     temps de préparation de la note (du clic au téléchargement, la première de chaque format compte le chargement du paquet de la note) : ${temps.join(' · ')}`);
  }

  // ---------------------------------------------- le BUREAU : les projets, l'équipe, le journal (docs/site/PROJETS_EQUIPE.md)
  // Aux QUATRE formats, dans un navigateur neuf, le parcours entier : se présenter (les initiales proposées, la pastille),
  // créer un projet d'une ligne, y entrer, deux sous-projets à la suite, une note créée DEPUIS le sous-projet, la retrouver
  // dans le projet — VUE (au téléphone, la charte cachait la table des notes : un projet s'ouvrait vide, le 11/11/2026) —,
  // renommer le projet (ses notes suivent, l'adresse aussi), ranger la note ailleurs, un nom piégé écrit comme du texte, un
  // projet qui a des notes qu'on ne supprime pas ; puis l'équipe (ajouter, refuser des initiales prises, quatre lettres dans
  // une pastille allongée, la teinte, « je suis… »), et le retour à l'accueil. Ce que l'œil voit : rien ne dépasse de
  // l'écran, les pastilles sont RONDES et leurs initiales y tiennent, au doigt chaque cible fait 40 px. Filtre « bureau ».
  // Puis les RÔLES et les MODULES VÉRIFIÉS (docs/site/PROJETS_EQUIPE.md § 8) : le premier inscrit est administrateur, un
  // membre ne vérifie pas, un administrateur déclare deux modules vérifiés — la marque paraît sur la page Équipe, l'accueil,
  // le volet, « Nouvelle note », la recherche et au pied du résultat du module ; le calcul de l'un « change » (son empreinte
  // est réécrite dans le registre) : sa marque tombe partout. La recherche de l'accueil (clavier, accents, Ctrl K), la
  // pastille du poste au pied de la barre. Sept mutations du site le font rougir (docs, § 8.5).
  if (!FILTRE || FILTRE === 'bureau') {
    const mesureBureau = (pg, large, zone) => pg.evaluate(({ large, zone }) => {
      const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const hors = [];
      for (const e of document.querySelectorAll('main *')) {
        if (!vis(e)) continue;
        const r = e.getBoundingClientRect();
        if (!r.width || r.right <= large + 1) continue;
        let q = e.parentElement, excuse = false;
        while (q && q !== document.body) {
          if (/auto|scroll|hidden/.test(getComputedStyle(q).overflowX) && q.getBoundingClientRect().right <= large + 1) { excuse = true; break; }
          q = q.parentElement;
        }
        if (!excuse) hors.push(`${e.tagName.toLowerCase()}.${String(e.getAttribute('class') || '').slice(0, 24)} → ${Math.round(r.right)}`);
      }
      // les pastilles : rondes (ou allongées pour quatre ou cinq initiales), leurs initiales dedans, lisibles
      const pastilles = [...document.querySelectorAll('main .avatar')].filter(vis).map((a) => {
        const r = a.getBoundingClientRect(), cs = getComputedStyle(a);
        const g = document.createRange(); g.selectNodeContents(a);
        const t = g.getBoundingClientRect();
        const n = a.textContent.trim().length;
        return { ini: a.textContent.trim(), l: +r.width.toFixed(1), h: +r.height.toFixed(1), rond: parseFloat(cs.borderTopLeftRadius) >= Math.min(r.width, r.height) / 2 - 0.5 || cs.borderTopLeftRadius === '50%',
          dedans: !n || (t.width <= r.width - 2 && t.left >= r.left - 0.5 && t.right <= r.right + 0.5), police: parseFloat(cs.fontSize), n,
          clair: cs.color, fond: cs.backgroundColor };
      });
      // les cibles : ce qui se touche dans la zone (boutons, champs, menus, nœuds de l'arbre, lignes qui mènent quelque part)
      const cibles = [...document.querySelectorAll(zone)].filter(vis).map((e) => {
        const r = e.getBoundingClientRect();
        return [(e.getAttribute('aria-label') || e.getAttribute('title') || e.getAttribute('placeholder') || e.textContent || e.id || e.className).trim().slice(0, 26), Math.round(r.height), Math.round(r.width)];
      });
      return { page: document.documentElement.scrollWidth, hors: hors.slice(0, 6), nPastilles: pastilles.length, auDoigt: matchMedia('(pointer: coarse)').matches,
        pasRondes: pastilles.filter((x) => !x.rond || (x.n <= 3 && Math.abs(x.l - x.h) > 0.6) || (x.n >= 4 && x.l <= x.h)).slice(0, 4),
        debordent: pastilles.filter((x) => !x.dedans).slice(0, 4), petites: pastilles.filter((x) => x.n && x.police < 8).slice(0, 4),
        basses: cibles.filter((c) => c[1] < 40).slice(0, 8), etroites: cibles.filter((c) => c[2] < 40).slice(0, 8), nCibles: cibles.length };
    }, { large, zone });
    const ZONE_PROJETS = '.bu-top .btn, .bu-top .crumbs a, #pj button, #pj select, #pj input, #pj a.node, #pj a.btn, #pj a.bu-retour, #pj a.bu-tout, #pj summary, #pj a.bu-carte';
    const ZONE_EQUIPE = '.bu-team button, .bu-team select, .bu-team input';
    const ZONE_ACCUEIL = '.accueil .side button, .accueil .side input, .accueil .side a.btn, .accueil .side a.site, .accueil .side label.btn, .accueil .hello .bu-moi, .accueil .hello button, .accueil .hello .bigsearch, .accueil .sechead a.btn';
    const pretBureau = async (pg, page) => {
      await pg.waitForFunction((page) => document.body.dataset.page === page && document.body.dataset.ecran === 'pret', page, { timeout: 30000 });
      await pg.waitForTimeout(150);
    };
    const memoire = (pg) => pg.evaluate(() => {
      const lire = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch (e) { return d; } };
      const notes = Object.values(lire('stab.notes.v1', {}));
      return { notes: notes.map((n) => ({ id: n.id, nom: n.nom, chantier: n.chantier, dossier: n.dossier, auteur: n.auteur && n.auteur.ini, par: n.par && n.par.ini, modifie: n.modifie })),
        journal: lire('stab.journal.v1', []).map((e) => [e.k, e.q && e.q.ini, (e.n && e.n.nom) || e.x || e.s || e.p || '']),
        projets: Object.values(lire('stab.projets.v1', {})).map((p) => [p.nom, p.sous, !!p.archive]), equipe: lire('stab.equipe.v1', { membres: {} }) };
    });
    const vuToast = (pg) => pg.evaluate(() => { const t = document.getElementById('toast'); return t && !t.hidden ? { texte: t.textContent, erreur: t.classList.contains('erreur') } : null; });
    const PIEGE = '<img src=x onerror=window.__xss=1>';
    // La photo de la page ENTIÈRE, puis la page rechargée. PIÈGE (mesuré le 11/11/2026) : pour photographier une page plus
    // haute que l'écran, Playwright change un instant les dimensions émulées — et Chromium en perd l'émulation tactile
    // (matchMedia('(pointer: coarse)') devient faux) jusqu'au document suivant. La page Équipe, atteinte SANS rechargement
    // (le menu) juste après une telle photo, était alors mesurée « à la souris » : champs de 36 px, teintes de 26 — un
    // rouge du harnais, pas du site (les deux autres formats tactiles, dont la page tenait dans l'écran, étaient verts).
    // Chaque mesure dit donc si le navigateur EST au doigt (auDoigt), et le contrôle l'exige.
    const photo = async (pg, nom, page) => {
      await pg.screenshot({ path: path.join(OUT, `site_bureau_${nom}.png`), fullPage: true });
      await pg.reload();
      await pretBureau(pg, page);
    };
    for (const [nomF, w, h, tel] of FORMATS_BIB) {
      const doigt = nomF !== 'pc';
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: doigt });
      const g = await c.newPage();
      g.erreurs = [];
      g.on('pageerror', (e) => g.erreurs.push(e.message));
      // (hors de Cloudflare, l'adresse de connexion n'existe pas : le 404 du serveur local n'est pas une erreur du site)
      g.on('console', (m) => { if (m.type() === 'error' && !/\/cdn-cgi\/access\/get-identity$/.test(m.location().url || '')) g.erreurs.push(m.text().slice(0, 300)); });
      g.on('dialog', (d) => d.accept());
      // l'adresse de connexion : le PC la reçoit (comme derrière Cloudflare Access), le téléphone reçoit une page HTML (la
      // page de connexion), les tablettes rien (404) — dans les trois cas, se présenter marche
      if (nomF === 'pc') await c.route('**/cdn-cgi/access/get-identity', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ email: 'louis@bureau.example', name: 'x' }) }));
      if (nomF === 'tel') await c.route('**/cdn-cgi/access/get-identity', (r) => r.fulfill({ status: 200, contentType: 'text/html', body: '<html>connexion</html>' }));
      const menu = tel ? '.mnav' : '.nav';
      const dit = `bureau ${nomF}`;

      // ---- l'accueil d'un poste neuf : personne ne s'est présenté
      await g.goto(`${BASE}/`);
      await pretBureau(g, 'accueil');
      let a = await g.evaluate((menu) => {
        const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const lien = (h) => { const e = document.querySelector(`${menu} a[href="${h}"]`); return !!e && vis(e) && e.getAttribute('aria-disabled') !== 'true' && !e.dataset.bientot; };
        return { projets: lien('projets'), equipe: lien('equipe'), qui: vis(document.querySelector('#ac-qui form[data-qui]')), moi: document.querySelector('#moi .avatar').textContent,
          inconnu: document.querySelector('#moi .avatar').classList.contains('inconnu'), bonjour: document.getElementById('bonjour').textContent,
          vide: document.getElementById('chantiers').textContent, champ: vis(document.getElementById('ac-projet')), creer: vis(document.getElementById('ac-projet-creer')) };
      }, menu);
      verif(`${dit} · accueil d'un poste neuf : le menu mène aux Projets et à l'Équipe (des liens, plus « bientôt ») ; « Qui êtes-vous ? » se voit, la pastille est un « ? », aucun projet — le champ qui en crée un est là`,
        a.projets && a.equipe && a.qui && a.moi === '?' && a.inconnu && a.bonjour === 'Bonjour' && /Aucun projet/.test(a.vide) && a.champ && !a.creer, JSON.stringify(a));
      await photo(g, `${nomF}_accueil_neuf`, 'accueil');

      // ---- se présenter : le nom ; les initiales sont proposées (LLE), l'adresse préremplie quand le site la connaît
      await g.fill('#ac-qui [name="nom"]', 'Louis Lejeune');
      await g.waitForTimeout(nomF === 'pc' ? 400 : 150);
      a = await g.evaluate(() => { const f = document.querySelector('#ac-qui form'); return { ini: f.querySelector('[name="initiales"]').value, email: f.querySelector('[name="email"]').value,
        apercu: f.querySelector('[data-qui-pastille]').textContent }; });
      verif(`${dit} · se présenter : « Louis Lejeune » → les initiales proposées sont LLE, la pastille les montre à mesure ; l'adresse ${nomF === 'pc' ? 'de connexion préremplit le champ' : 'reste vide (le site ne la connaît pas ici)'}`,
        a.ini === 'LLE' && a.apercu === 'LLE' && a.email === (nomF === 'pc' ? 'louis@bureau.example' : ''), JSON.stringify(a));
      await g.click('#ac-qui button[type="submit"]');
      await g.waitForTimeout(250);
      a = await g.evaluate(() => { const p = document.querySelector('#moi .avatar'); const r = p.getBoundingClientRect();
        return { bonjour: document.getElementById('bonjour').textContent, moi: p.textContent, classe: p.className, rond: Math.abs(r.width - r.height) < 0.6 && r.width >= 40,
          qui: document.querySelector('#ac-qui').children.length, fil: [...document.querySelectorAll('#activite .feed')].map((f) => [f.querySelector('.avatar').textContent, f.querySelector('div').textContent]) }; });
      verif(`${dit} · présenté : « Bonjour Louis », la pastille LLE (ronde, dans sa teinte) mène à l'équipe, « Qui êtes-vous ? » a disparu, le journal dit « Vous avez rejoint l'équipe »`,
        a.bonjour === 'Bonjour Louis' && a.moi === 'LLE' && /\bc\d\b/.test(a.classe) && /\bn3\b/.test(a.classe) && a.rond && a.qui === 0
        && a.fil.length === 1 && a.fil[0][0] === 'LLE' && /^Vous avez rejoint l'équipe/.test(a.fil[0][1]), JSON.stringify(a));

      // ---- créer un projet : son nom, Entrée
      await g.fill('#ac-projet', '  Réalex ');
      a = { creer: await g.isVisible('#ac-projet-creer') };
      await g.press('#ac-projet', 'Enter');
      await g.waitForTimeout(200);
      Object.assign(a, await g.evaluate(() => ({ lignes: [...document.querySelectorAll('#chantiers a.site')].map((s) => [s.getAttribute('href'), s.querySelector('b').textContent, s.querySelector('.nb').textContent]),
        champ: document.getElementById('ac-projet').value, fil: document.querySelector('#activite .feed div').textContent })), { toast: await vuToast(g), m: await memoire(g) });
      verif(`${dit} · créer un projet depuis l'accueil : « Réalex », Entrée — il est dans la liste (0 note), le champ est vidé, le message le dit, le journal aussi (« Vous avez créé le projet Réalex »)`,
        a.creer && a.lignes.length === 1 && a.lignes[0][1] === 'Réalex' && a.lignes[0][2] === '0 note' && /^projets\?p=R/.test(a.lignes[0][0]) && a.champ === ''
        && a.toast && /Projet « Réalex » créé/.test(a.toast.texte) && /^Vous avez créé le projet Réalex/.test(a.fil) && JSON.stringify(a.m.projets) === '[["Réalex",[],false]]', JSON.stringify(a));

      // ---- entrer dans le projet (la navigation de l'application, sans rechargement)
      await g.evaluate(() => { window.__memePage = true; });
      await g.click('#chantiers a.site');
      await pretBureau(g, 'projets');
      a = await g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        return { memePage: window.__memePage === true, adresse: decodeURIComponent(location.pathname.split('/').pop() + location.search), vue: document.getElementById('pj').dataset.vue,
          titre: document.getElementById('pj-nom').value, titreVu: vis(document.getElementById('pj-nom')), fil: document.getElementById('pj-fil').textContent.replace(/\s+/g, ' ').trim(), docTitre: document.title,
          noteIci: vis(document.querySelector('[data-note-ici]')), sousChamp: vis(document.querySelector('form[data-creer-sous] input')), nouvelle: vis(document.getElementById('ouvrir-nouvelle')),
          importer: vis(document.querySelector('label[for="importer"]')) }; });
      verif(`${dit} · le projet s'ouvre sans recharger la page : son adresse (projets?p=Réalex), son nom en titre (modifiable), le fil « Projets › Réalex », « + Note », le champ d'un sous-projet ; « Nouvelle note » et l'import sont atteignables`,
        a.memePage && a.adresse === 'projets?p=Réalex' && a.vue === 'liste' && a.titre === 'Réalex' && a.titreVu && a.fil === 'Projets Réalex' && /^Réalex · Projets/.test(a.docTitre)
        && a.noteIci && a.sousChamp && a.nouvelle && a.importer, JSON.stringify(a));

      // ---- deux sous-projets à la suite : le nom, Entrée ; le champ garde la main
      await g.fill('form[data-creer-sous] input', 'Étude de la dalle');
      await g.press('form[data-creer-sous] input', 'Enter');
      await g.waitForTimeout(150);
      const main1 = await g.evaluate(() => document.activeElement && document.activeElement.closest('form[data-creer-sous]') !== null);
      await g.keyboard.type('Charpente');
      await g.keyboard.press('Enter');
      await g.waitForTimeout(150);
      a = await g.evaluate(() => ({ sous: [...document.querySelectorAll('.bu-sous-nom')].map((i) => i.value), noeuds: [...document.querySelectorAll('#pj-noeuds .node.l1')].map((n) => n.textContent) }));
      verif(`${dit} · deux sous-projets créés à la suite (le nom, Entrée ; le champ garde la main pour le suivant) : « Étude de la dalle », « Charpente »`,
        main1 && JSON.stringify(a.sous) === '["Étude de la dalle","Charpente"]' && JSON.stringify((await memoire(g)).projets) === '[["Réalex",["Étude de la dalle","Charpente"],false]]', JSON.stringify({ main1, ...a }));

      // ---- une note créée DEPUIS le sous-projet : le projet et le sous-projet sont déjà écrits
      await g.click('[data-note-ici="Étude de la dalle"]');
      await g.waitForTimeout(200);
      a = await g.evaluate(() => ({ ouverte: !document.getElementById('nouvelle').hidden, projet: document.getElementById('nn-chantier').value, sous: document.getElementById('nn-dossier').value,
        main: document.activeElement && document.activeElement.id, proposes: [...document.querySelectorAll('#nn-dossiers option')].map((o) => o.value) }));
      verif(`${dit} · « + Note » d'un sous-projet ouvre « Nouvelle note » DÉJÀ rangée : projet « Réalex », sous-projet « Étude de la dalle » ; le nom de la note attend ; les sous-projets connus sont proposés`,
        a.ouverte && a.projet === 'Réalex' && a.sous === 'Étude de la dalle' && a.main === 'nn-nom' && JSON.stringify(a.proposes) === '["Étude de la dalle","Charpente"]', JSON.stringify(a));
      await g.fill('#nn-nom', 'Poutres R+1');
      await g.click('#nouvelle .mopt[data-module="poutre"]');
      await g.click('#nn-creer');
      await g.waitForFunction(() => document.body.dataset.page === 'module' && /note=/.test(location.search), null, { timeout: 30000 });
      let m = await memoire(g);
      const idNote = m.notes[0] && m.notes[0].id;
      verif(`${dit} · la note est créée, RANGÉE et SIGNÉE : « Poutres R+1 », Réalex › Étude de la dalle, auteur LLE ; son établi s'ouvre ; le journal dit « a créé la note »`,
        m.notes.length === 1 && m.notes[0].nom === 'Poutres R+1' && m.notes[0].chantier === 'Réalex' && m.notes[0].dossier === 'Étude de la dalle' && m.notes[0].auteur === 'LLE'
        && JSON.stringify(m.journal[0]) === '["note.creee","LLE","Poutres R+1"]', JSON.stringify(m.notes) + JSON.stringify(m.journal.slice(0, 2)));
      // dans l'ÉTABLI (au PC, une fois) : « Classer la note… » parle des mêmes projets — les mots, les listes ; un nom écrit
      // en minuscules rejoint le sous-projet qui existe (« charpente » → « Charpente »), sans en créer un second
      if (nomF === 'pc') {
        await pretEcran(g);
        await g.click('.top [data-ranger]');
        await g.waitForTimeout(150);
        a = await g.evaluate(() => ({ libelles: [...document.querySelectorAll('#et-rangement label.et-rg-l')].map((l) => l.childNodes[0].textContent.trim()),
          projet: document.getElementById('et-rg-chantier').value, sous: document.getElementById('et-rg-dossier').value,
          projets: [...document.querySelectorAll('#et-rg-chantiers option')].map((o) => o.value), sousP: [...document.querySelectorAll('#et-rg-dossiers option')].map((o) => o.value),
          invites: [document.getElementById('et-rg-chantier').placeholder, document.getElementById('et-rg-dossier').placeholder] }));
        await g.fill('#et-rg-dossier', '  charpente ');
        await g.click('#et-rangement button[type="submit"]');
        await g.waitForTimeout(300);
        m = await memoire(g);
        verif(`${dit} · dans l'établi, « Classer la note… » range dans un PROJET et un SOUS-PROJET (les mêmes mots), propose ceux de la page des projets (Réalex ; Étude de la dalle, Charpente — même vide) ; « charpente » rejoint « Charpente » sans doublon`,
          a.libelles.includes('Projet') && a.libelles.includes('Sous-projet') && !a.libelles.includes('Chantier') && a.projet === 'Réalex' && a.sous === 'Étude de la dalle'
          && JSON.stringify(a.projets) === '["Réalex"]' && JSON.stringify(a.sousP) === '["Charpente","Étude de la dalle"]' && JSON.stringify(a.invites) === '["Sans projet","Sans sous-projet"]'
          && m.notes[0].dossier === 'Charpente' && JSON.stringify(m.projets) === '[["Réalex",["Étude de la dalle","Charpente"],false]]' && m.journal[0][0] === 'note.classee', JSON.stringify({ a, n: m.notes[0], p: m.projets, j: m.journal[0] }));
        // (la note retourne où le parcours l'attend)
        await g.click('.top [data-ranger]');
        await g.waitForTimeout(150);
        await g.fill('#et-rg-dossier', 'Étude de la dalle');
        await g.click('#et-rangement button[type="submit"]');
        await g.waitForTimeout(300);
      }

      // ---- de retour dans le projet : la note s'y VOIT (au téléphone, la table restait cachée : un projet « vide »)
      await g.goto(`${BASE}/projets?p=${encodeURIComponent('Réalex')}`);
      await pretBureau(g, 'projets');
      const dateAvant = (await memoire(g)).notes[0].modifie;
      a = await g.evaluate((large) => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const l = document.querySelector('#pj-liste tr[data-note]'); const r = l ? l.getBoundingClientRect() : null;
        const nom = l && l.querySelector('.bu-n b'); const rn = nom ? nom.getBoundingClientRect() : null;
        return { ligne: vis(l), boite: r && [Math.round(r.left), Math.round(r.right), Math.round(r.height)], nom: nom && nom.textContent, nomVu: vis(nom) && rn.width > 30 && rn.right <= large,
          lien: l && l.querySelector('a').getAttribute('href'), pastille: l && [...l.querySelectorAll('.avatar')].filter(vis).map((x) => x.textContent), ranger: vis(l && l.querySelector('.bu-ranger')),
          groupe: l && l.closest('.bu-groupe') && l.closest('.bu-groupe').querySelector('.bu-sous-nom').value, sousTitre: document.querySelector('.bu-sous').textContent,
          volet: [...document.querySelectorAll('#pj-volet .feed')].map((f) => [f.querySelector('.avatar').textContent, f.querySelector('div').textContent]).filter((l) => / créé la note /.test(l[1])),
          faits: [...document.querySelectorAll('#pj-volet .facts dt')].map((d) => d.textContent), voletVu: vis(document.querySelector('#pj-volet .feed')) }; }, w);
      verif(`${dit} · le projet MONTRE sa note (vue à l'écran, dans l'écran) : « Poutres R+1 » sous « Étude de la dalle », la pastille LLE de son auteur, le menu « Ranger… » ; l'activité du projet se voit : « Vous avez créé la note Poutres R+1 »`,
        a.ligne && a.boite && a.boite[0] >= 0 && a.boite[1] <= w && a.boite[2] >= 40 && a.nom === 'Poutres R+1' && a.nomVu && /^poutre\?note=/.test(a.lien)
        && JSON.stringify(a.pastille) === '["LLE"]' && a.ranger && a.groupe === 'Étude de la dalle' && /^1 note · 2 sous-projets/.test(a.sousTitre)
        && a.voletVu && a.volet.length === 1 && a.volet[0][0] === 'LLE' && /^Vous avez créé la note Poutres R\+1 · Étude de la dalle/.test(a.volet[0][1]) && a.faits.includes('Y ont travaillé'), JSON.stringify(a));
      let mes = await mesureBureau(g, w, ZONE_PROJETS);
      verif(`${dit} · la page d'un projet : rien ne dépasse de l'écran (${w} px) ; les pastilles sont rondes, leurs initiales dedans${doigt ? ' ; au doigt, chaque cible fait 40 px au moins' : ''}`,
        mes.page <= w && !mes.hors.length && mes.nPastilles >= 3 && !mes.pasRondes.length && !mes.debordent.length && !mes.petites.length
        && mes.auDoigt === doigt && (!doigt || (!mes.basses.length && !mes.etroites.length && mes.nCibles >= 10)), JSON.stringify(mes));
      await photo(g, `${nomF}_projet`, 'projets');

      // ---- ranger la note dans un autre sous-projet : elle y passe, sans changer de date
      await g.selectOption(`.bu-ranger[data-ranger="${idNote}"]`, 's:Charpente');
      await g.waitForTimeout(200);
      m = await memoire(g);
      a = await g.evaluate(() => { const l = document.querySelector('#pj-liste tr[data-note]'); return l && l.closest('.bu-groupe').querySelector('.bu-sous-nom').value; });
      verif(`${dit} · « Ranger… » déplace la note dans « Charpente » : elle y est à l'écran et dans la mémoire, sa date n'a pas bougé, le journal dit « a rangé la note »`,
        a === 'Charpente' && m.notes[0].dossier === 'Charpente' && m.notes[0].chantier === 'Réalex' && m.notes[0].modifie === dateAvant && m.journal[0][0] === 'note.classee', JSON.stringify({ a, n: m.notes[0], j: m.journal[0] }));

      // ---- renommer le projet : on écrit par-dessus son titre ; ses notes suivent, l'adresse aussi
      await g.fill('#pj-nom', 'Réalex – Lot 2');
      await g.press('#pj-nom', 'Enter');
      await g.waitForTimeout(250);
      m = await memoire(g);
      a = await g.evaluate(() => ({ adresse: new URLSearchParams(location.search).get('p'), titre: document.getElementById('pj-nom').value, fil: document.getElementById('pj-fil').textContent.replace(/\s+/g, ' ').trim(),
        ligne: !!document.querySelector('#pj-liste tr[data-note]'), noeud: (document.querySelector('#pj-noeuds .node.chantier') || {}).textContent || '' }));
      verif(`${dit} · renommer le projet « Réalex – Lot 2 » : sa note suit (son « chantier » est réécrit, sans changer de date), l'adresse et le fil aussi ; le journal dit « a renommé le projet »`,
        a.adresse === 'Réalex – Lot 2' && a.titre === 'Réalex – Lot 2' && a.fil === 'Projets Réalex – Lot 2' && a.ligne && m.notes[0].chantier === 'Réalex – Lot 2' && m.notes[0].modifie === dateAvant
        && m.projets.length === 1 && m.projets[0][0] === 'Réalex – Lot 2' && m.journal[0][0] === 'projet.renomme', JSON.stringify({ a, n: m.notes[0], p: m.projets, j: m.journal[0] }));

      // ---- un projet qui a encore une note ne se supprime pas (on ne perd jamais une note en supprimant un projet)
      await g.click('.bu-menu > summary');
      await g.waitForTimeout(100);
      const menuVu = await g.evaluate((large) => { const e = document.querySelector('.bu-menu-c'); const r = e.getBoundingClientRect(); return e.checkVisibility() && r.left >= 0 && r.right <= large && r.width > 150; }, w);
      await g.click('[data-supprimer-projet]');
      await g.waitForTimeout(150);
      a = { toast: await vuToast(g), projets: (await memoire(g)).projets.length };
      verif(`${dit} · « ⋯ » s'ouvre dans l'écran ; supprimer un projet qui a encore une note est REFUSÉ, et dit pourquoi (« encore 1 note ») : rien n'est perdu`,
        menuVu && a.toast && a.toast.erreur && /encore 1 note/.test(a.toast.texte) && a.projets === 1, JSON.stringify({ menuVu, ...a }));

      // ---- la liste des projets : le champ cherche, puis crée — un nom PIÉGÉ est écrit comme du texte
      await g.goto(`${BASE}/projets`);
      await pretBureau(g, 'projets');
      await g.fill('#pj-champ', 'réal');
      await g.waitForTimeout(120);
      const cherche = await g.evaluate(() => ({ noeuds: [...document.querySelectorAll('#pj-noeuds a.node.chantier')].map((n) => n.querySelector('.bu-nn').textContent), creer: (document.querySelector('.bu-creer-n') || {}).textContent || '' }));
      await g.fill('#pj-champ', PIEGE);
      await g.waitForTimeout(120);
      const propose = await g.evaluate(() => ({ creer: (document.querySelector('.bu-creer-n') || {}).textContent || '', img: document.querySelectorAll('#pj img').length }));
      await g.press('#pj-champ', 'Enter');
      await g.waitForFunction(() => /p=/.test(location.search), null, { timeout: 10000 });
      await pretBureau(g, 'projets');
      a = await g.evaluate(() => ({ xss: window.__xss, img: document.querySelectorAll('main img, #toast img').length, titre: document.getElementById('pj-nom').value, fil: document.getElementById('pj-fil').textContent,
        volet: (document.querySelector('#pj-volet .feed div') || {}).textContent || '', docTitre: document.title }));
      verif(`${dit} · le champ de l'arbre CHERCHE (« réal » → Réalex – Lot 2, et propose de créer « réal ») puis CRÉE à Entrée ; un nom piégé (« <img onerror> ») est écrit comme du TEXTE partout — titre, fil, journal — et rien ne s'exécute`,
        JSON.stringify(cherche.noeuds) === '["Réalex – Lot 2"]' && /Créer « réal »/.test(cherche.creer) && propose.creer.includes(PIEGE) && propose.img === 0
        && a.xss === undefined && a.img === 0 && a.titre === PIEGE && a.fil.includes(PIEGE) && a.volet.includes(PIEGE), JSON.stringify({ cherche, propose, a }));
      // ce projet est vide : il se supprime (après confirmation), on revient à la liste
      await g.click('.bu-menu > summary');
      await g.click('[data-supprimer-projet]');
      await g.waitForFunction(() => !location.search, null, { timeout: 10000 });
      await pretBureau(g, 'projets');
      m = await memoire(g);
      a = await g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        return { vue: document.getElementById('pj').dataset.vue, noeuds: [...document.querySelectorAll('#pj-noeuds a.node')].filter(vis).map((n) => n.querySelector('.bu-nn').textContent),
          cartes: [...document.querySelectorAll('.bu-carte')].filter(vis).length, champ: vis(document.getElementById('pj-champ')), fil: vis(document.querySelector('#pj-volet .feed')) }; });
      verif(`${dit} · un projet VIDE se supprime ; la liste des projets : « Tous les projets », « Réalex – Lot 2 » — ${tel ? 'au téléphone, la liste seule (pas de cartes en double), son champ en tête' : 'l\'arbre à gauche, les cartes au centre'}, l'activité du bureau`,
        m.projets.length === 1 && a.vue === 'arbre' && JSON.stringify(a.noeuds) === '["Tous les projets","Réalex – Lot 2"]' && a.cartes === (tel ? 0 : 1) && a.champ && a.fil
        && m.journal[0][0] === 'projet.supprime', JSON.stringify({ a, p: m.projets, j: m.journal[0] }));
      mes = await mesureBureau(g, w, ZONE_PROJETS);
      verif(`${dit} · la liste des projets : rien ne dépasse de l'écran, pastilles rondes${doigt ? ', cibles de 40 px au moins' : ''}`,
        mes.page <= w && !mes.hors.length && !mes.pasRondes.length && !mes.debordent.length && mes.auDoigt === doigt && (!doigt || (!mes.basses.length && !mes.etroites.length)), JSON.stringify(mes));
      await photo(g, `${nomF}_projets`, 'projets');

      // ---- l'ÉQUIPE : moi, puis mes collègues
      await g.click(`${menu} a[href="equipe"]`);
      await pretBureau(g, 'equipe');
      a = await g.evaluate(() => ({ moi: document.querySelector('#eq-moi .avatar.xl').textContent, nom: document.querySelector('[data-moi="nom"]').value, ini: document.querySelector('[data-moi="initiales"]').value,
        email: document.querySelector('[data-moi="email"]').value, lignes: document.querySelectorAll('#eq-membres tr[data-membre]').length, vous: !!document.querySelector('#eq-membres .tag.pub'),
        titre: document.getElementById('eq-titre').textContent, journal: document.querySelectorAll('#eq-journal .feed').length, jours: document.querySelectorAll('#eq-journal .bu-jour').length }));
      verif(`${dit} · l'équipe (par le menu) : ma fiche (Louis Lejeune, LLE${nomF === 'pc' ? ', mon adresse' : ''}), « Membres · 1 » avec la mention « vous », le journal entier sous « Aujourd'hui »`,
        a.moi === 'LLE' && a.nom === 'Louis Lejeune' && a.ini === 'LLE' && a.email === (nomF === 'pc' ? 'louis@bureau.example' : '') && a.lignes === 1 && a.vous && a.titre === 'Membres · 1'
        && a.journal >= 7 && a.jours === 1, JSON.stringify(a));
      // ajouter Marie : ses initiales sont proposées ; puis quelqu'un qui en a QUATRE ; puis un refus (LLE est pris)
      await g.fill('#eq-nom', 'Marie Dubois');
      const iniMarie = await g.inputValue('#eq-ini');
      await g.fill('#eq-email', 'marie@bureau.example');
      await g.press('#eq-email', 'Enter');
      await g.waitForTimeout(150);
      await g.fill('#eq-nom', 'Anne Van den Broeck');
      const iniAnne = await g.inputValue('#eq-ini');
      await g.press('#eq-nom', 'Enter');
      await g.waitForTimeout(150);
      await g.fill('#eq-nom', 'Lucie Lebon');
      await g.fill('#eq-ini', 'lle');
      await g.press('#eq-ini', 'Enter');
      await g.waitForTimeout(150);
      a = await g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const err = document.getElementById('eq-erreur');
        return { membres: [...document.querySelectorAll('#eq-membres tr[data-membre]')].map((t) => [t.querySelector('.who2 b').childNodes[0].textContent.trim(), t.querySelector('.avatar').textContent, t.querySelector('.who2 > div > span').textContent]),
          erreur: vis(err) && err.textContent, refus: !!document.querySelector('#eq-form .field.refus #eq-ini'), saisie: document.getElementById('eq-ini').value,
          longue: (() => { const p = [...document.querySelectorAll('#eq-membres .avatar')].find((x) => x.textContent === 'AVDB'); const r = p.getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height)]; })(),
          noms: [...document.querySelectorAll('#eq-membres .who2 > div')].map((d) => Math.round(d.getBoundingClientRect().left)) }; });
      verif(`${dit} · ajouter des collègues : Marie Dubois (initiales proposées MDU, son adresse), Anne Van den Broeck (AVDB : quatre lettres, la pastille s'allonge, les noms restent alignés) ; « Lucie Lebon » avec LLE est REFUSÉE — « déjà celles de Louis Lejeune », le champ est marqué`,
        iniMarie === 'MDU' && iniAnne === 'AVDB' && JSON.stringify(a.membres.map((x) => x.slice(0, 2))) === '[["Anne Van den Broeck","AVDB"],["Louis Lejeune","LLE"],["Marie Dubois","MDU"]]'
        && a.membres[2][2] === 'marie@bureau.example' && /pas d'adresse/.test(a.membres[0][2]) && /LLE sont déjà celles de Louis Lejeune/.test(a.erreur || '') && a.refus && a.saisie === 'LLE'
        && a.longue[0] > a.longue[1] && new Set(a.noms).size === 1, JSON.stringify({ iniMarie, iniAnne, ...a }));
      // ma teinte, mes initiales (elles changent pour la suite ; ce qui est déjà signé garde LLE), « je suis… »
      await g.click('#eq-moi [data-teinte="3"]');
      await g.waitForTimeout(120);
      await g.fill('[data-moi="initiales"]', 'll');
      await g.press('[data-moi="initiales"]', 'Enter');
      await g.waitForTimeout(200);
      a = await g.evaluate(() => ({ moi: document.querySelector('#eq-moi .avatar.xl').textContent, classe: document.querySelector('#eq-moi .avatar.xl').className,
        allumee: (document.querySelector('#eq-moi [data-teinte].on') || { dataset: {} }).dataset.teinte, journal: [...new Set([...document.querySelectorAll('#eq-journal .feed > .avatar')].map((x) => x.textContent))] }));
      const sigNote = (await memoire(g)).notes[0];
      verif(`${dit} · ma fiche : la teinte choisie se voit (la 3), mes initiales deviennent LL — et ce que j'ai déjà signé GARDE « LLE » (la note, les lignes du journal : la signature est une copie du moment)`,
        a.moi === 'LL' && /\bc3\b/.test(a.classe) && /\bn2\b/.test(a.classe) && a.allumee === '3' && JSON.stringify(a.journal) === '["LLE"]' && sigNote.auteur === 'LLE', JSON.stringify({ a, sigNote }));
      const idMarie = await g.evaluate(() => [...document.querySelectorAll('#eq-jesuis option')].find((o) => /Marie/.test(o.textContent)).value);
      await g.selectOption('#eq-jesuis', idMarie);
      await g.waitForTimeout(150);
      const marie = await g.evaluate(() => [document.querySelector('#eq-moi .avatar.xl').textContent, document.querySelector('[data-moi="nom"]').value]);
      await g.click(`[data-voir="${idMarie}"]`);
      await g.waitForTimeout(120);
      const filtre = await g.evaluate(() => ({ puce: document.getElementById('eq-filtre').textContent, lignes: document.querySelectorAll('#eq-journal .feed').length, vide: (document.querySelector('#eq-journal .bu-vide') || {}).textContent || '' }));
      const idLouis = await g.evaluate(() => [...document.querySelectorAll('#eq-jesuis option')].find((o) => /Louis/.test(o.textContent)).value);
      await g.selectOption('#eq-jesuis', idLouis);
      await g.waitForTimeout(150);
      await g.click('#eq-filtre button');
      await g.waitForTimeout(120);
      a = await g.evaluate(() => ({ moi: document.querySelector('#eq-moi .avatar.xl').textContent, lignes: document.querySelectorAll('#eq-journal .feed').length, puce: document.getElementById('eq-filtre').textContent }));
      verif(`${dit} · « Ce n'est pas vous ? » passe le poste à Marie (MDU) puis me le rend ; l'horloge d'un membre ne montre QUE son activité (Marie : rien encore), sa puce rend le journal entier`,
        JSON.stringify(marie) === '["MDU","Marie Dubois"]' && /Marie/.test(filtre.puce) && filtre.lignes === 0 && /Marie n'a encore rien fait/.test(filtre.vide) && a.moi === 'LL' && a.lignes >= 9 && a.puce === '', JSON.stringify({ marie, filtre, a }));
      // ---- les RÔLES et les MODULES VÉRIFIÉS (docs/site/PROJETS_EQUIPE.md § 8) : le premier inscrit est administrateur ; lui
      // seul déclare un module vérifié ; la marque vaut pour le calcul du jour et tombe s'il change
      const jour = await g.evaluate(() => new Date().toLocaleDateString('fr-BE', { day: '2-digit', month: '2-digit', year: 'numeric' }));
      const etatEquipe = () => g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        return { roles: [...document.querySelectorAll('#eq-membres tr[data-membre]')].map((t) => [t.querySelector('.avatar').textContent, ((t.querySelector('.bu-mr .role') || {}).textContent || '').trim(), vis(t.querySelector('.bu-mr .role')), vis(t.querySelector('.bu-role-m'))]),
          nb: document.getElementById('eq-verif-nb').textContent,
          modules: [...document.querySelectorAll('#eq-modules tr[data-module-verif]')].map((t) => [t.dataset.moduleVerif, t.dataset.etat, ((t.querySelector('.bu-ve .valid') || {}).textContent || '').trim(), ((t.querySelector('.bu-vqui') || {}).textContent || '').trim(),
            [...t.querySelectorAll('.bu-vg button')].filter(vis).map((x) => x.textContent.trim()).join()]),
          note: (document.querySelector('.bu-verif-note') || {}).textContent || '', menuRole: !!document.querySelector('#eq-membres form[data-edit] [name="role"]'),
          volet: ['poutre', 'dalle', 'soudures'].map((k) => !document.querySelector(`.modpanel [data-verifie="${k}"]`).hidden) }; });
      a = await etatEquipe();
      verif(`${dit} · les rôles : le premier inscrit du poste (moi) est Administrateur, Anne et Marie sont Membres — ${tel ? 'au téléphone, un écusson après le nom (la colonne « Rôle » s\'efface)' : 'une colonne « Rôle », lue'} ; aucun module n'est vérifié : 13 lignes « À vérifier », chacune avec son bouton « Vérifier » (je suis administrateur)`,
        JSON.stringify(a.roles.map((r) => r.slice(0, 2))) === '[["AVDB","Membre"],["LL","Administrateur"],["MDU","Membre"]]' && a.roles.every((r) => r[2] === !tel)
        && JSON.stringify(a.roles.map((r) => r[3])) === JSON.stringify([false, tel, false]) && a.nb === '0 vérifié sur 14' && a.modules.length === 14
        && a.modules.every((m) => m[1] === 'non' && m[2] === 'À vérifier' && m[3] === '' && m[4] === 'Vérifier') && /Vous êtes administrateur/.test(a.note) && !a.volet.includes(true), JSON.stringify(a));
      // Marie (membre) prend le poste : elle LIT les vérifications, elle n'en pose pas ; la fiche d'un collègue n'a pas de menu « Rôle »
      await g.selectOption('#eq-jesuis', idMarie);
      await g.waitForTimeout(150);
      await g.click(`#eq-membres [data-modifier="${idLouis}"]`);
      await g.waitForTimeout(120);
      const enMembre = await etatEquipe();
      await g.click('#eq-membres [data-annuler]');
      await g.selectOption('#eq-jesuis', idLouis);
      await g.waitForTimeout(150);
      verif(`${dit} · un MEMBRE lit les vérifications, il n'en pose pas : aucun bouton « Vérifier », la page dit pourquoi ; la fiche d'un collègue n'a pas de menu « Rôle »`,
        enMembre.modules.length === 14 && enMembre.modules.every((m) => m[4] === '') && /Seul un administrateur déclare un module vérifié/.test(enMembre.note) && !enMembre.menuRole,
        JSON.stringify({ note: enMembre.note, menuRole: enMembre.menuRole, gestes: enMembre.modules.map((m) => m[4]).join('') }));
      // administrateur : je déclare deux modules vérifiés (une confirmation dit ce qu'on affirme ; ce parcours l'accepte)
      await g.click('#eq-modules [data-verifier="poutre"]');
      await g.waitForTimeout(200);
      const toastVerif = await vuToast(g);
      await g.click('#eq-modules [data-verifier="dalle"]');
      await g.waitForTimeout(200);
      a = await etatEquipe();
      let lg = (cle) => a.modules.find((m) => m[0] === cle) || [];
      verif(`${dit} · un ADMINISTRATEUR déclare « Poutres béton armé » et « Dalles béton armé » vérifiés : la ligne dit « Vérifié », par qui (Louis Lejeune) et quand (${jour}), le bouton devient « Retirer », le compte passe à 2 sur 14 ; le volet des modules les marque aussitôt — eux seuls`,
        toastVerif && !toastVerif.erreur && /Poutres béton armé » vérifié/.test(toastVerif.texte) && a.nb === '2 vérifiés sur 14' && lg('poutre')[1] === 'verifie' && lg('poutre')[2] === 'Vérifié'
        && lg('poutre')[3] === `LLLouis Lejeune · ${jour}` && lg('poutre')[4] === 'Retirer' && lg('dalle')[1] === 'verifie' && a.modules.filter((m) => m[1] === 'non').length === 12
        && JSON.stringify(a.volet) === '[true,true,false]', JSON.stringify({ toastVerif, nb: a.nb, poutre: lg('poutre'), dalle: lg('dalle'), volet: a.volet }));
      // le CALCUL d'un module change (l'empreinte du site n'est plus celle de la vérification) : la marque tombe d'elle-même
      await g.evaluate(() => { const r = JSON.parse(localStorage.getItem('stab.verifications.v1')); r.dalle.version = '0123456789abcdef'; localStorage.setItem('stab.verifications.v1', JSON.stringify(r)); });
      await g.reload();
      await pretBureau(g, 'equipe');
      a = await etatEquipe();
      verif(`${dit} · le calcul des dalles a CHANGÉ depuis sa vérification : la ligne dit « À revérifier » (on sait encore qui l'avait vérifié), le bouton « Revérifier », le compte retombe à 1 sur 14, le volet ne marque plus que la poutre — une marque ne certifie jamais un calcul que personne n'a relu`,
        a.nb === '1 vérifié sur 14' && lg('dalle')[1] === 'change' && lg('dalle')[2] === 'À revérifier' && lg('dalle')[3] === `LLLouis Lejeune · ${jour} · le calcul a changé depuis` && lg('dalle')[4] === 'Revérifier'
        && lg('poutre')[1] === 'verifie' && JSON.stringify(a.volet) === '[true,false,false]', JSON.stringify({ nb: a.nb, poutre: lg('poutre'), dalle: lg('dalle'), volet: a.volet }));
      // nommer Marie administrateur (le menu « Rôle » de sa fiche), lui reprendre le rôle, puis vouloir me retirer le mien
      const nommer = async (id, role) => { await g.click(`#eq-membres [data-modifier="${id}"]`); await g.waitForTimeout(120); await g.selectOption('#eq-membres form[data-edit] [name="role"]', role);
        await g.click('#eq-membres form[data-edit] button[type="submit"]'); await g.waitForTimeout(220); return vuToast(g); };
      const toastNomme = await nommer(idMarie, 'admin');
      const deuxAdmins = (await etatEquipe()).roles.map((r) => r[1]).join();
      await nommer(idMarie, 'membre');
      const toastRefus = await nommer(idLouis, 'membre');
      await g.click('#eq-membres [data-annuler]');
      await g.waitForTimeout(120);
      a = await etatEquipe();
      const lignesRoles = (await memoire(g)).journal.slice(0, 2);
      verif(`${dit} · un administrateur en nomme un autre (Marie) puis lui reprend le rôle — le journal dit les deux ; le DERNIER administrateur ne peut pas se retirer le sien : refusé, la raison est dite, rien ne change`,
        toastNomme && !toastNomme.erreur && deuxAdmins === 'Membre,Administrateur,Administrateur' && toastRefus && toastRefus.erreur && /toujours un administrateur/.test(toastRefus.texte)
        && a.roles.map((r) => r[1]).join() === 'Membre,Administrateur,Membre' && JSON.stringify(lignesRoles) === '[["membre.membre","LL","Marie Dubois"],["membre.admin","LL","Marie Dubois"]]',
        JSON.stringify({ toastNomme, toastRefus, deuxAdmins, roles: a.roles, lignesRoles }));
      mes = await mesureBureau(g, w, ZONE_EQUIPE);
      verif(`${dit} · la page Équipe : rien ne dépasse de l'écran (${w} px), les pastilles sont rondes (AVDB allongée), leurs initiales dedans${doigt ? ' ; au doigt, chaque cible fait 40 px au moins' : ''}`,
        mes.page <= w && !mes.hors.length && mes.nPastilles >= 12 && !mes.pasRondes.length && !mes.debordent.length && !mes.petites.length
        && mes.auDoigt === doigt && (!doigt || (!mes.basses.length && !mes.etroites.length && mes.nCibles >= 20)), JSON.stringify(mes));
      await g.screenshot({ path: path.join(OUT, `site_bureau_${nomF}_equipe.png`), fullPage: true });

      // ---- l'accueil, après : mes projets, l'activité avec les pastilles, la note à reprendre signée ; tout tient au rechargement
      await g.goto(`${BASE}/`);
      await pretBureau(g, 'accueil');
      a = await g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        return { bonjour: document.getElementById('bonjour').textContent, moi: document.querySelector('#moi .avatar').textContent,
          projets: [...document.querySelectorAll('#chantiers a.site')].map((s) => [s.querySelector('b').textContent, s.querySelector('.nb').textContent, [...s.querySelectorAll('.avatar')].filter(vis).map((x) => x.textContent).join()]),
          fil: [...document.querySelectorAll('#activite .feed')].filter(vis).map((f) => f.querySelector('.avatar').textContent + ' ' + f.querySelector('div').textContent.slice(0, 44)),
          carte: [...document.querySelectorAll('#recentes .rcard')].map((r) => [r.querySelector('.t').textContent, r.querySelector('.ou').textContent, (r.querySelector('.avatar') || {}).textContent]) }; });
      verif(`${dit} · l'accueil, intégré : « Bonjour Louis » et ma pastille (LL) ; le projet « Réalex – Lot 2 » (1 note, la pastille de qui y a travaillé) ; l'activité — six lignes au plus, chacune avec la pastille de son auteur ; la note à reprendre dit son projet et son auteur`,
        a.bonjour === 'Bonjour Louis' && a.moi === 'LL' && JSON.stringify(a.projets) === '[["Réalex – Lot 2","1 note","LLE"]]' && a.fil.length === 6 && a.fil.every((l) => /^(LLE|LL) Vous avez /.test(l))
        && a.carte.length === 1 && a.carte[0][0] === 'Poutres R+1' && /Réalex – Lot 2 › Charpente/.test(a.carte[0][1]) && a.carte[0][2] === 'LLE', JSON.stringify(a));
      // ---- les détails de l'accueil : la marque des modules vérifiés, « Nouvelle note », le volet, la pastille du poste, la recherche
      a = await g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const vues = [...document.querySelectorAll('.accueil a.tile [data-verifie]')].filter(vis).map((e) => { const r = e.getBoundingClientRect(); return [e.dataset.verifie, e.title, Math.round(r.width), Math.round(r.height)]; });
        const t = document.querySelector('.accueil a.tile[data-module="poutre"]').getBoundingClientRect(), m = document.querySelector('.accueil a.tile[data-module="poutre"] [data-verifie]').getBoundingClientRect();
        const leg = document.querySelector('.accueil .okleg');
        return { vues, n: document.querySelectorAll('.accueil a.tile [data-verifie]').length, dedans: m.left >= t.left && m.right <= t.right && m.top >= t.top && m.bottom <= t.bottom, legende: vis(leg) && leg.textContent.trim() }; });
      verif(`${dit} · l'accueil marque le module VÉRIFIÉ — « Poutres béton armé », et lui seul des 13 : la marque se voit dans sa tuile (14 px au moins) et dit « Vérifié · LL · ${jour} » ; « Dalles », dont le calcul a changé, ne la porte plus ; la légende l'explique`,
        a.n === 14 && a.vues.length === 1 && a.vues[0][0] === 'poutre' && a.vues[0][1] === `Vérifié · LL · ${jour}` && a.vues[0][2] >= 14 && a.vues[0][3] >= 14 && a.dedans && a.legende === 'vérifié par un administrateur', JSON.stringify(a));
      await g.click('#ouvrir-nouvelle');
      await g.waitForTimeout(250);
      const dansNouvelle = await g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        return { ouverte: vis(document.querySelector('#nouvelle .mopt')), vues: [...document.querySelectorAll('#nouvelle .mopt [data-verifie]')].filter(vis).map((e) => e.dataset.verifie), n: document.querySelectorAll('#nouvelle .mopt [data-verifie]').length }; });
      await g.click('#nouvelle [data-fermer-modal]');
      await g.waitForTimeout(200);
      let volet = null, barre = null;
      if (!tel) {
        await g.click('.nav [data-modules]');
        await g.waitForTimeout(300);
        volet = await g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const leg = document.querySelector('.modpanel .okleg');
          return { ouvert: vis(document.querySelector('.modpanel a.navitem')), vues: [...document.querySelectorAll('.modpanel [data-verifie]')].filter(vis).map((e) => [e.dataset.verifie, Math.round(e.getBoundingClientRect().width)]), legende: vis(leg) && leg.textContent.trim() }; });
        await g.click('.modpanel [data-modules]');
        await g.waitForTimeout(250);
        barre = await g.evaluate(() => { const e = document.querySelector('.nav [data-moi-rail]'), p = e.querySelector('.avatar'), n = document.querySelector('.nav').getBoundingClientRect();
          if (!p) return null;
          const r = p.getBoundingClientRect(), l = e.getBoundingClientRect();
          return { vu: p.checkVisibility({ visibilityProperty: true, opacityProperty: true }), ini: p.textContent, rond: Math.abs(r.width - r.height) < 0.6, dans: r.left >= n.left && r.right <= n.right && r.bottom <= innerHeight && r.top >= 0,
            href: e.getAttribute('href'), tip: e.dataset.tip, cible: [Math.round(l.width), Math.round(l.height)] }; });
      }
      verif(`${dit} · la même marque là où le module est nommé : « Nouvelle note » (la poutre seule)${tel ? '' : ', le volet des modules (la poutre seule ; sa légende « vérifié ») ; au pied de la barre, MA pastille (LL, ronde, dans la barre) mène à l\'équipe' + (doigt ? ' — une cible de 40 px' : '')}`,
        dansNouvelle.ouverte && dansNouvelle.n === 14 && JSON.stringify(dansNouvelle.vues) === '["poutre"]'
        && (tel || (volet.ouvert && volet.vues.length === 1 && volet.vues[0][0] === 'poutre' && volet.vues[0][1] >= 14 && volet.legende === 'vérifié'
          && barre && barre.vu && barre.ini === 'LL' && barre.rond && barre.dans && barre.href === 'equipe' && /^Louis Lejeune/.test(barre.tip) && (!doigt || (barre.cible[0] >= 40 && barre.cible[1] >= 40)))),
        JSON.stringify({ dansNouvelle, volet, barre }));
      // la RECHERCHE : un module (vérifié : il le dit), une note, un projet ; le clavier ; rien ne porte ce nom
      const lireRecherche = () => g.evaluate(() => { const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const bo = document.getElementById('ac-resultats'), r = bo.getBoundingClientRect(), ch = document.getElementById('ac-recherche').getBoundingClientRect();
        return { ouvert: vis(bo), boite: [Math.round(r.left), Math.round(r.right), Math.round(r.top - ch.bottom)], champ: [Math.round(ch.width), Math.round(ch.height)],
          lignes: [...bo.querySelectorAll('a.bu-res')].map((x) => [x.querySelector('.bu-res-t b').textContent, x.querySelector('.bu-res-t span').textContent, x.getAttribute('href'), x.classList.contains('on'), vis(x.querySelector('.ok')), Math.round(x.getBoundingClientRect().height)]),
          vide: (bo.querySelector('.bu-vide') || {}).textContent || '', etendu: document.getElementById('ac-recherche').getAttribute('aria-expanded'), saisie: document.getElementById('ac-recherche').value }; });
      await g.fill('#ac-recherche', 'pout');
      await g.waitForTimeout(150);
      a = await lireRecherche();
      await g.press('#ac-recherche', 'ArrowDown');
      const apresFleche = (await lireRecherche()).lignes.map((l) => l[3]);
      await g.press('#ac-recherche', 'Escape');
      const apresEchap = await lireRecherche();
      verif(`${dit} · la recherche de l'accueil, « pout » : le module « Poutres béton armé » d'abord (il dit qu'il est vérifié), « Poutres bois », la note « Poutres R+1 » (son module, son projet) — la liste s'ouvre sous le champ, dans l'écran${doigt ? ', des lignes de 40 px au moins' : ''} ; la flèche descend d'une ligne, Échap vide le champ et referme`,
        a.ouvert && a.etendu === 'true' && a.lignes.length === 3 && JSON.stringify(a.lignes[0].slice(0, 5)) === '["Poutres béton armé","Module · Béton","poutre",true,true]'
        && JSON.stringify(a.lignes[1].slice(0, 5)) === '["Poutres bois","Module · Autres","poutre-bois",false,false]' && a.lignes[2][0] === 'Poutres R+1' && /^Note · Poutres béton armé · Réalex – Lot 2/.test(a.lignes[2][1])
        && /^poutre\?note=/.test(a.lignes[2][2]) && a.boite[0] >= 0 && a.boite[1] <= w && a.boite[2] >= 0 && a.boite[2] <= 12 && (!doigt || (a.lignes.every((l) => l[5] >= 40) && a.champ[1] >= 40))
        && JSON.stringify(apresFleche) === '[false,true,false]' && !apresEchap.ouvert && apresEchap.saisie === '' && apresEchap.etendu === 'false', JSON.stringify({ a, apresFleche, apresEchap }));
      await g.fill('#ac-recherche', 'zzz');
      await g.waitForTimeout(120);
      const rien = await lireRecherche();
      if (nomF === 'pc') { await g.click('.accueil .h2'); await g.keyboard.press('Control+k'); }
      const auClavier = nomF !== 'pc' || await g.evaluate(() => document.activeElement && document.activeElement.id === 'ac-recherche');
      await g.fill('#ac-recherche', 'realex lot');
      await g.waitForTimeout(150);
      a = await lireRecherche();
      await g.press('#ac-recherche', 'Enter');
      // (si la recherche n'a rien trouvé, Entrée ne mène nulle part : le contrôle doit le DIRE, pas attendre une page qui ne viendra pas)
      const arrive = await g.waitForFunction(() => document.body.dataset.page === 'projets' && document.body.dataset.ecran === 'pret', null, { timeout: 8000 })
        .then(() => g.evaluate(() => ({ p: new URLSearchParams(location.search).get('p'), fil: document.getElementById('pj-fil').textContent.replace(/\s+/g, ' ').trim() })), () => ({ p: null, fil: '' }));
      verif(`${dit} · « zzz » : « Rien ne porte ce nom » ; « realex lot » (sans accent ni capitale) trouve le projet « Réalex – Lot 2 », et Entrée y mène${nomF === 'pc' ? ' ; Ctrl K rend la main au champ de recherche' : ''}`,
        rien.ouvert && !rien.lignes.length && /Rien ne porte ce nom/.test(rien.vide) && auClavier && a.lignes.length >= 1 && a.lignes[0][0] === 'Réalex – Lot 2' && /^Projet · 1 note/.test(a.lignes[0][1])
        && arrive.p === 'Réalex – Lot 2' && /Réalex – Lot 2/.test(arrive.fil), JSON.stringify({ rien: rien.vide, auClavier, lignes: a.lignes, arrive }));
      await g.goto(`${BASE}/`);
      await pretBureau(g, 'accueil');
      mes = await mesureBureau(g, w, ZONE_ACCUEIL);
      verif(`${dit} · l'accueil : rien ne dépasse de l'écran, les pastilles sont rondes${doigt ? ', les cibles des projets et de l\'activité font 40 px au moins' : ''}`,
        mes.page <= w && !mes.hors.filter((x) => !/rcard|thumb|body|\.t |\.m |\.ou|chip|supp|avatar/.test(x)).length && mes.nPastilles >= 8 && !mes.pasRondes.length && !mes.debordent.length
        && mes.auDoigt === doigt && (!doigt || (!mes.basses.length && !mes.etroites.length && mes.nCibles >= 6)), JSON.stringify(mes));
      await g.screenshot({ path: path.join(OUT, `site_bureau_${nomF}_accueil.png`), fullPage: true });
      // ---- DANS le module : au pied du cadre Résultat, « Module vérifié · LL · la date » — une ligne qu'on lit, aux quatre formats
      const piedModule = async (adresse) => {
        await g.goto(`${BASE}/${adresse}`);
        await g.waitForSelector('#resultat .pfoot .et-verif', { timeout: 120000, state: 'attached' });
        if (tel) await g.click('.mtabs [data-onglet="resultat"]');
        await g.waitForTimeout(350);
        return g.evaluate(() => { const e = document.querySelector('#resultat .pfoot .et-verif'); e.scrollIntoView({ block: 'end' });
          const r = e.getBoundingClientRect(), t = e.querySelector('.et-verif-t'), tr = t.getBoundingClientRect(), pied = e.closest('.pfoot').getBoundingClientRect();
          const icones = [...e.querySelectorAll('.et-verif-i')].filter((x) => x.checkVisibility({ visibilityProperty: true, opacityProperty: true })).map((x) => x.className.replace('et-verif-i ', ''));
          return { vu: e.checkVisibility({ visibilityProperty: true, opacityProperty: true }), etat: e.dataset.etat, texte: t.textContent, titre: e.title, entier: t.scrollWidth <= t.clientWidth + 1, police: parseFloat(getComputedStyle(t).fontSize),
            dans: r.left >= pied.left - 1 && r.right <= pied.right + 1 && r.right <= innerWidth && tr.width > 40 && r.bottom <= innerHeight + 1, icones, n: document.querySelectorAll('#resultat .et-verif').length,
            boutons: e.querySelectorAll('button, a').length, auDoigt: matchMedia('(pointer: coarse)').matches }; });
      };
      const pv = await piedModule('poutre');
      verif(`${dit} · dans « Poutres béton armé », le pied du cadre Résultat dit « Module vérifié · LL · ${jour} » : une seule ligne, lue en entier (11 px au moins), dans le cadre et dans l'écran, son pictogramme — ni bouton ni lien (au doigt, un bouton du résultat doit faire 44 px)`,
        pv.vu && pv.etat === 'verifie' && pv.texte === `Module vérifié · LL · ${jour}` && /vérifié par Louis Lejeune/.test(pv.titre) && pv.entier && pv.police >= 11 && pv.dans && JSON.stringify(pv.icones) === '["et-verif-oui"]'
        && pv.n === 1 && pv.boutons === 0 && pv.auDoigt === doigt, JSON.stringify(pv));
      await g.screenshot({ path: path.join(OUT, `site_bureau_${nomF}_module_verifie.png`) });
      const pd = await piedModule('dalle'), pn = await piedModule('corniere');
      verif(`${dit} · « Dalles béton armé », dont le calcul a changé depuis sa vérification, dit « Module à revérifier » ; « Cornières », que personne n'a vérifié, « Module à vérifier » — jamais « vérifié »`,
        pd.vu && pd.etat === 'change' && pd.texte === 'Module à revérifier' && /son calcul a changé/.test(pd.titre) && JSON.stringify(pd.icones) === '["et-verif-non"]' && pd.dans && pd.entier
        && pn.vu && pn.etat === 'non' && pn.texte === 'Module à vérifier' && JSON.stringify(pn.icones) === '["et-verif-non"]' && pn.dans && pn.entier, JSON.stringify({ pd, pn }));
      await g.goto(`${BASE}/`);
      await pretBureau(g, 'accueil');
      await g.reload();
      await pretBureau(g, 'accueil');
      a = await g.evaluate(() => [document.getElementById('bonjour').textContent, document.querySelectorAll('#chantiers a.site').length, document.querySelectorAll('#activite .feed').length]);
      verif(`${dit} · tout est enregistré dans ce navigateur : après un rechargement, « Bonjour Louis », le projet, l'activité sont là ; aucune erreur JavaScript de tout le parcours`,
        JSON.stringify(a) === '["Bonjour Louis",1,6]' && !g.erreurs.length, JSON.stringify(a) + ' ' + g.erreurs.join(' | ').slice(0, 300));
      await c.close();
    }
  }

  // ---------------------------------------------------------------- retour à l'accueil
  await p.goto(`${BASE}/index.html`);
  await p.waitForTimeout(300);
  const n = await p.$$eval('#recentes .rcard', (e) => e.length);
  // (les formats, « télécharger », « météo », la sonde, l'enregistrement, le pilote, l'envoi, les plaques et la bibliothèque travaillent dans leurs propres navigateurs : leurs notes ne sont pas ici)
  // (« piege » seul n'en crée aucune : ses deux .stab s'ouvrent sans être écrits — un calcul ouvert n'écrit rien tant
  //  qu'on ne le modifie pas, décision D1 du 10/11/2026 — le contrôle rougissait sur ce filtre seul, avant le lot S4 déjà)
  if (!FILTRE.startsWith('formats') && !['telecharger', 'meteo', 'sonde', 'enregistrement', 'ouverture', 'pilote', 'envoi', 'plaques', 'doigt', 'bibliotheque', 'bibliotheque_autres', 'bureau', 'nappes', 'corniere', 'note_pdf', 'piege'].includes(FILTRE)) verif('accueil : les notes créées sont à reprendre', n >= 1, `${n} note(s)`);
  // une vignette est un dessin : un tableau (vue HTML d'un module de consultation) y serait coupé et illisible
  const vignettes = await p.$$eval('#recentes .rcard', (cs) => cs.map((c) => ({ page: c.getAttribute('href').split('?')[0],
    v: c.querySelector('.thumb').firstElementChild ? c.querySelector('.thumb').firstElementChild.tagName.toLowerCase() : '' })));
  verif('accueil : chaque vignette est un dessin ou reste vide, jamais un tableau', vignettes.every((x) => x.v === '' || x.v === 'svg'), JSON.stringify(vignettes));
  verif('accueil : la note d\'un module qui a un dessin le montre en vignette', vignettes.every((x) => !avecDessin.has(x.page) || x.v === 'svg'), JSON.stringify(vignettes));
  if (!FILTRE || ['garde', 'poutre'].includes(FILTRE)) verif('accueil : le chantier « Réalex » est listé', (await p.textContent('#chantiers')).includes('Réalex'));
  m = await mesure();
  verif('accueil : rien ne déborde, aucun nom coupé', !m.deborde && !m.coupes.length, m.coupes.join(', '));
  await capture('accueil');
  verif('aucune erreur JavaScript', erreurs.length === 0, erreurs.join(' | ').slice(0, 400));
  await b.close();
  // le passage complet joue AUSSI le filtre demarrage, à la fin (son navigateur à lui : celui du parcours est fermé), et
  // son verdict compte — un « non mesurable » compris : il est dit, et le code de sortie n'est pas 0
  let demarrage = { code: 0, ko: [], nonMesures: [] };
  if (!FILTRE) {
    console.log('---- demarrage (verifier_demarrage.js, tous les sous-filtres : le passage complet le joue aussi)');
    demarrage = await jouerDemarrage(sousFiltresDemarrage(FILTRE));
    console.log(`bilan du passage complet : parcours ${ko} KO · ${direBilanDemarrage(demarrage)}`);
  }
  process.exit(codeSortie(ko, demarrage));
};

// le parcours ne part que si ce fichier est lancé (node site/web/verifier_site.js …) ; chargé par require(), il ne rend
// que ses aides
if (require.main === module && !DELEGUE) principal();
module.exports = { pretEcran, echecEcran, bandeauErreur, bandeauDit, montage, ouvrir, proxyTravailleur, interceptees, SIGNAL_ABSENT, sousFiltresDemarrage, bilanDemarrage, codeSortie };
