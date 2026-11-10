// La mesure du démarrage de l'établi, par sa sonde (?sonde=1, js/sonde.js) : le PROTOCOLE de la refonte (lot 0,
// docs/site/REFONTE_ETABLI.md) — les mesures « avant » et « après » se prennent avec ce script, sur le même protocole.
// Profil Chromium sur disque (comme un vrai navigateur : un contexte privé de Playwright ne garde pas le .wasm de
// 8,6 Mo en cache et le retélécharge à chaque visite). Chiffres versés : docs/site/mesures/AAAA-MM-JJ_demarrage_*.md.
//
//   python3 site/web/outils/serveur_local.py 8770 &
//   flock /tmp/verrou_navigateur node site/web/outils/mesurer_demarrage.js http://127.0.0.1:8770 <sortie.json>
// PLAN=quatre   : poutre, assemblages, plancher, age ; par tirage et par format, un profil NEUF sur disque :
//                 froid (1re visite), 2e visite (même module), passage (au module suivant du cycle, même profil)
//                 puis, DANS l'application (lot S4, docs/site/REFONTE_ETABLI.md S4-D10 « Mesure ») :
//                 passage_app : de l'établi « passage » au module du tirage, au PC par le volet « Modules » du rail
//                               (.modpanel a.navitem), au téléphone par l'accueil (lien Accueil, puis la tuile) ;
//                 accueil_app : lien Accueil, le « pret » du moteur observé, puis la tuile du module d'après
//                 (le scénario « passage » reste par goto : un lien direct, comparable à la mesure du 07/11).
//                 Dans un document qui vit toute la session, seules comptent les marques POSTÉRIEURES au clic (la page
//                 et les sondes du travailleur reçues après lui) ; les navigations de document du cadre principal
//                 sont comptées (attendu : 0) et la sentinelle du document est relue.
// PLAN=dix-huit : les 18 modules ; par tirage, un profil neuf par module : 1re visite puis 2e visite (seule gardée)
// REP (3), FORMATS_M (pc,tel pour « quatre », pc pour « dix-huit »)
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const os = require('os');
const BASE = process.argv[2];
const SORTIE = process.argv[3];
const PLAN = process.env.PLAN || 'quatre';
const REP = +(process.env.REP || 3);
const FORMATS = (process.env.FORMATS_M || (PLAN === 'quatre' ? 'pc,tel' : 'pc')).split(',');
const QUATRE = ['poutre', 'assemblages', 'plancher', 'age'];
const FMT = {
  pc: { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 },
  tel: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true,
    userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Mobile Safari/537.36' },
};

// posé dans chaque document : la sentinelle, l'instant du clic (en capture), le « pret » du moteur vu par un espion, et
// l'échec du démarrage lu sur ce que la page montre — le bandeau rouge (lot S4 : body[data-ecran=erreur], .et-demarre) ou
// l'ancien voile #demarrage.erreur ; jamais une lecture de classList sur un élément peut-être absent
const INIT_M = () => {
  window.__s = Math.random();
  window.__pretVu = false;
  addEventListener('click', () => { window.__tclic = performance.timeOrigin + performance.now(); }, true);
  const W = window.Worker;
  window.Worker = class extends W {
    constructor(u, o) { super(u, o); EventTarget.prototype.addEventListener.call(this, 'message', (ev) => { if (ev.data && ev.data.type === 'pret') window.__pretVu = true; }); }
  };
  window.__lireErreur = () => {
    const d = document.getElementById('demarrage');
    if (d && d.classList.contains('erreur')) return (document.getElementById('demarrage-etape') || d).textContent || 'erreur';
    if (document.body && document.body.dataset.ecran === 'erreur') return (document.querySelector('.et-demarre') || document.body).textContent.replace(/\s+/g, ' ').trim().slice(0, 300) || 'erreur';
    return '';
  };
};

async function relever(p) {
  await p.waitForFunction(() => {
    if (window.__lireErreur && window.__lireErreur()) return true;
    return globalThis.__sonde && performance.getEntriesByName('stab:premier_rendu').length;
  }, null, { timeout: 180000 });
  return p.evaluate(() => {
    const nav = performance.getEntriesByType('navigation')[0];
    const o = performance.timeOrigin;
    const fcp = performance.getEntriesByType('paint').find((e) => e.name === 'first-contentful-paint');
    const S = globalThis.__sonde || { marques: [], details: {}, ressources: [] };
    return {
      erreur: window.__lireErreur ? window.__lireErreur() : '',
      nav: { reponse_fin: +nav.responseEnd.toFixed(1), dcl: +nav.domContentLoadedEventEnd.toFixed(1) },
      fcp: fcp ? +fcp.startTime.toFixed(1) : null,
      page: performance.getEntriesByType('mark').filter((m) => m.name.startsWith('stab:')).map((m) => [m.name.slice(5), +m.startTime.toFixed(1)]),
      travailleur: S.marques.map(([k, t]) => [k, +(t - o).toFixed(1)]),
      details: S.details,
      ressources: S.ressources.map((r) => ({ ...r, debut: +(r.debut - o).toFixed(1), fin: +(r.fin - o).toFixed(1), reponse: +(r.reponse - o).toFixed(1) })),
    };
  });
}

// DANS l'application : rien que ce qui SUIT le clic — les marques de la page postérieures à l'instant du clic, les sondes
// du travailleur reçues après lui (globalThis.__sondes : une par ouverture, js/moteur.js), leurs marques postérieures
// au clic. Un document rechargé sans la sonde (le site d'avant : le lien n'a pas « ?sonde=1 ») est DIT, et la mesure
// s'arrête aussitôt : ce n'est pas un passage dans l'application.
async function releverApres(p, ref) {
  const fin = await p.waitForFunction((ref) => {
    if (window.__lireErreur && window.__lireErreur()) return 'erreur';
    if (window.__s !== ref.s && document.readyState === 'complete' && !new URLSearchParams(location.search).has('sonde')) return 'recharge';
    const tc = window.__tclic;
    if (tc == null) return false;
    const rendu = performance.getEntriesByName('stab:premier_rendu').some((m) => m.startTime > tc - performance.timeOrigin);
    const sondes = (globalThis.__sondes || []).length > (window.__s === ref.s ? ref.s0 : 0);
    return rendu && sondes ? 'ok' : false;
  }, ref, { timeout: 90000 }).then((h) => h.jsonValue()).catch(() => 'delai');
  return p.evaluate(([ref, fin]) => {
    const o = performance.timeOrigin, tc = window.__tclic ?? null, rel = tc != null ? tc - o : Infinity;
    const sondes = (globalThis.__sondes || []).slice(window.__s === ref.s ? ref.s0 : 0);
    const S = sondes[sondes.length - 1] || { details: {}, ressources: [] };
    return {
      fin, memeDocument: window.__s === ref.s, erreur: window.__lireErreur ? window.__lireErreur() : '',
      clic: tc != null ? +rel.toFixed(1) : null, nav: { reponse_fin: tc != null ? +rel.toFixed(1) : null, dcl: null }, fcp: null,
      page: performance.getEntriesByType('mark').filter((m) => m.name.startsWith('stab:') && m.startTime > rel).map((m) => [m.name.slice(5), +m.startTime.toFixed(1)]),
      travailleur: sondes.flatMap((x) => x.marques || []).filter(([, t]) => tc != null && t > tc).map(([k, t]) => [k, +(t - o).toFixed(1)]),
      details: S.details || {},
      ressources: (S.ressources || []).filter((r) => tc != null && r.debut > tc).map((r) => ({ ...r, debut: +(r.debut - o).toFixed(1), fin: +(r.fin - o).toFixed(1), reponse: +(r.reponse - o).toFixed(1) })),
    };
  }, [ref, fin]);
}

// la première occurrence d'une marque (« wasm_instantiate » revient pour les bibliothèques de numpy) ; la DERNIÈRE pour
// les marques de la page d'un passage dans l'application (déjà filtrées : postérieures au clic)
const premiere = (liste, k) => { const x = liste.find(([n]) => n === k); return x ? x[1] : null; };
const derniere = (liste, k) => { const x = [...liste].reverse().find(([n]) => n === k); return x ? x[1] : null; };
function phases(r, choisir = premiere) {
  const P = (k) => choisir(r.page, k), T = (k) => premiere(r.travailleur, k), D = r.details || {};
  const dt = (a, b) => (a != null && b != null ? +(b - a).toFixed(1) : null);
  const wDeb = T('wasm_instantiateStreaming_debut') ?? T('wasm_instantiate_debut');
  const wFin = T('wasm_instantiateStreaming_fin') ?? T('wasm_instantiate_fin');
  const res = Object.fromEntries((r.ressources || []).map((x) => [x.nom, x]));
  const transfert = (r.ressources || []).reduce((a, x) => a + (x.transfert || 0), 0);
  return {
    premier_rendu: P('premier_rendu'), fcp: r.fcp, html: r.nav.reponse_fin,
    page_js: dt(r.nav.reponse_fin, P('etabli_debut')),
    travailleur_lance: dt(P('moteur_cree'), T('travailleur_debut')),
    wasm: dt(wDeb, wFin),
    wasm_telechargement: res['pyodide.asm.wasm'] ? dt(res['pyodide.asm.wasm'].debut, res['pyodide.asm.wasm'].fin) : null,
    init_python: dt(wFin, T('loadPyodide_fin')),
    loadPyodide: dt(T('loadPyodide_debut'), T('loadPyodide_fin')),
    numpy: dt(T('numpy_debut'), T('numpy_fin')),
    attente_numpy: dt(T('deballage_fin'), T('numpy_attendu')),
    empreinte: dt(T('empreinte_debut'), T('empreinte_fin')),
    deballage: dt(T('empreinte_fin'), T('deballage_fin')),
    routeur: dt(T('numpy_attendu'), T('routeur_importe')),
    pret: T('pret'),
    attente_requete: dt(T('pret'), T('module_import_debut') ?? T('numpy_import_debut')),
    import_numpy: dt(T('numpy_import_debut'), T('numpy_import_fin')),
    import_module: dt(T('module_import_debut'), T('module_import_fin')),
    premier_calcul: dt(T('requete1_debut'), T('requete1_rejouee') ?? T('requete1_traitee')),
    vers_page: dt(T('reponse1_postee'), P('reponse1_recue')),
    application: dt(P('reponse1_recue'), P('reponse1_appliquee')),
    peinture: dt(P('reponse1_appliquee'), P('premier_rendu')),
    requete_ordinaire: D.calculer_ms, registre_cles: D.cles_ms != null ? +D.cles_ms.toFixed(1) : null,
    octets_reponse: D.reponse1_octets, modules_python: D.modules_apres_requete1, numpy_importe: D.numpy_importe,
    import_erreur: D.import_erreur || null, transfert_travailleur: transfert,
  };
}

(async () => {
  const MODULES = (await import('file://' + path.join(__dirname, '..', 'js', 'modules.js'))).MODULES;
  const tous = Object.keys(MODULES);
  const sortie = [];
  const visite = async (p, rec, m) => {
    const t0 = Date.now();
    await p.goto(`${BASE}/${MODULES[m].page}?sonde=1`);
    const r = await relever(p);
    const x = { ...rec, module: m, mur_ms: Date.now() - t0, erreur: r.erreur, ...phases(r), brut: r };
    sortie.push(x);
    console.log(`${x.rep} ${x.format} ${x.scenario.padEnd(9)} ${m.padEnd(13)} rendu ${x.premier_rendu} · wasm ${x.wasm} · loadPyodide ${x.loadPyodide} · numpy ${x.numpy} · routeur ${x.routeur} · import ${x.import_module} · calcul ${x.premier_calcul} · peinture ${x.peinture}${x.erreur ? ' ERREUR ' + x.erreur : ''}`);
  };
  // un passage DANS l'application : ``geste`` clique ; la mesure part de l'instant du DERNIER clic (le lien ou la tuile)
  const visiteApp = async (p, rec, m, geste) => {
    const ref = await p.evaluate(() => ({ s: window.__s, s0: (globalThis.__sondes || []).length }));
    const n0 = p.__navs, t0 = Date.now();
    let r;
    try { await geste(); r = await releverApres(p, ref); } catch (e) { r = { fin: 'geste', erreur: String(e.message).split('\n')[0].slice(0, 200), page: [], travailleur: [], nav: {}, memeDocument: null }; }
    const ph = phases(r, derniere);
    const pourquoi = r.fin === 'recharge' ? 'navigation de document : la sonde ne suit pas (ce n\'est pas un passage dans l\'application)'
      : r.fin === 'delai' ? 'délai : aucune marque postérieure au clic' : '';
    const x = { ...rec, module: m, mur_ms: Date.now() - t0, navigations: p.__navs - n0, sentinelle: r.memeDocument ? 'gardée' : 'perdue', fin: r.fin,
      erreur: r.erreur || pourquoi, ...ph, rendu_depuis_clic: ph.premier_rendu != null && r.clic != null ? +(ph.premier_rendu - r.clic).toFixed(1) : null, brut: r };
    sortie.push(x);
    console.log(`${x.rep} ${x.format} ${x.scenario.padEnd(11)} ${m.padEnd(13)} rendu ${x.rendu_depuis_clic} ms après le clic · navigations ${x.navigations} · sentinelle ${x.sentinelle} · import ${x.import_module} · calcul ${x.premier_calcul} · peinture ${x.peinture}${x.erreur ? ' ERREUR ' + x.erreur : ''}`);
  };
  const vu = (p, s) => p.locator(s).filter({ visible: true }).first();
  // l'accueil, rejoint par son lien (et prêt : son écran monté, ou chargé sur le site d'avant)
  const versAccueil = async (p) => {
    await vu(p, 'a[href="./"]').click();
    await p.waitForFunction(() => location.pathname === '/' && document.readyState !== 'loading' && !!document.querySelector('.tile')
      && (!document.body.dataset.ecran || document.body.dataset.ecran === 'pret'), null, { timeout: 60000 });
  };
  const gestePassage = (p, fmt, m) => async () => {
    if (fmt === 'pc') {
      await vu(p, '.nav a.navitem[data-modules]').click();
      await p.waitForSelector('.modpanel', { state: 'visible', timeout: 5000 });
      await p.click(`.modpanel a.navitem[href="${MODULES[m].page}"]`);
    } else { await versAccueil(p); await p.click(`.tile[data-module="${m}"]`); }
  };
  const gesteAccueil = (p, m) => async () => {
    await versAccueil(p);
    // le moteur prêt (« pret » vu par l'espion du document) — au plus 30 s : l'accueil d'avant n'en a pas
    await p.waitForFunction(() => window.__pretVu, null, { timeout: 30000 }).catch(() => {});
    await p.click(`.tile[data-module="${m}"]`);
  };
  for (let rep = 0; rep < REP; rep++) {
    for (const fmt of FORMATS) {
      const liste = PLAN === 'quatre' ? QUATRE : tous;
      for (let i = 0; i < liste.length; i++) {
        const mod = liste[i], suivant = liste[(i + 1) % liste.length], troisieme = liste[(i + 2) % liste.length];
        const dossier = fs.mkdtempSync(path.join(os.tmpdir(), 'stab-profil-'));
        const ctx = await chromium.launchPersistentContext(dossier, { ...FMT[fmt], env: { ...process.env, LANG: 'C.UTF-8' } });
        await ctx.addInitScript(INIT_M);
        const p = ctx.pages()[0] || await ctx.newPage();
        p.__navs = 0;
        p.on('request', (r) => { if (r.isNavigationRequest() && r.frame() === p.mainFrame()) p.__navs++; });
        const charge = os.loadavg()[0];
        const rec = (scenario) => ({ rep, plan: PLAN, format: fmt, scenario, charge_1min: +charge.toFixed(2) });
        if (PLAN === 'quatre') {
          await visite(p, rec('froid'), mod);
          await visite(p, rec('2e_visite'), mod);
          await visite(p, rec('passage'), suivant);
          // DANS l'application, depuis l'établi « passage » (ouvert par goto, avec la sonde) : des clés que ce
          // travailleur n'a pas encore vues (une sonde par clé : la première requête de chacune)
          await visiteApp(p, rec('passage_app'), mod, gestePassage(p, fmt, mod));
          await visiteApp(p, rec('accueil_app'), troisieme, gesteAccueil(p, troisieme));
        } else {
          await visite(p, rec('froid'), mod);
          await visite(p, rec('2e_visite'), mod);
        }
        await ctx.close();
        fs.rmSync(dossier, { recursive: true, force: true });
      }
    }
  }
  fs.writeFileSync(SORTIE, JSON.stringify(sortie, null, 1));
})();
