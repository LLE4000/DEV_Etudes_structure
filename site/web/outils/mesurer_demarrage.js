// La mesure du démarrage de l'établi, par sa sonde (?sonde=1, js/sonde.js) : le PROTOCOLE de la refonte (lot 0,
// docs/site/REFONTE_ETABLI.md) — les mesures « avant » et « après » se prennent avec ce script, sur le même protocole.
// Profil Chromium sur disque (comme un vrai navigateur : un contexte privé de Playwright ne garde pas le .wasm de
// 8,6 Mo en cache et le retélécharge à chaque visite). Chiffres versés : docs/site/mesures/AAAA-MM-JJ_demarrage_*.md.
//
//   python3 site/web/outils/serveur_local.py 8770 &
//   flock /tmp/verrou_navigateur node site/web/outils/mesurer_demarrage.js http://127.0.0.1:8770 <sortie.json>
// PLAN=quatre   : poutre, assemblages, plancher, age ; par tirage et par format, un profil NEUF sur disque :
//                 froid (1re visite), 2e visite (même module), passage (au module suivant du cycle, même profil)
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

async function relever(p) {
  await p.waitForFunction(() => {
    const d = document.getElementById('demarrage');
    if (d && d.classList.contains('erreur')) return true;
    return globalThis.__sonde && performance.getEntriesByName('stab:premier_rendu').length;
  }, null, { timeout: 180000 });
  return p.evaluate(() => {
    const nav = performance.getEntriesByType('navigation')[0];
    const o = performance.timeOrigin;
    const fcp = performance.getEntriesByType('paint').find((e) => e.name === 'first-contentful-paint');
    const d = document.getElementById('demarrage');
    const S = globalThis.__sonde || { marques: [], details: {}, ressources: [] };
    return {
      erreur: d.classList.contains('erreur') ? document.getElementById('demarrage-etape').textContent : '',
      nav: { reponse_fin: +nav.responseEnd.toFixed(1), dcl: +nav.domContentLoadedEventEnd.toFixed(1) },
      fcp: fcp ? +fcp.startTime.toFixed(1) : null,
      page: performance.getEntriesByType('mark').filter((m) => m.name.startsWith('stab:')).map((m) => [m.name.slice(5), +m.startTime.toFixed(1)]),
      travailleur: S.marques.map(([k, t]) => [k, +(t - o).toFixed(1)]),
      details: S.details,
      ressources: S.ressources.map((r) => ({ ...r, debut: +(r.debut - o).toFixed(1), fin: +(r.fin - o).toFixed(1), reponse: +(r.reponse - o).toFixed(1) })),
    };
  });
}

// la première occurrence d'une marque (« wasm_instantiate » revient pour les bibliothèques de numpy)
const premiere = (liste, k) => { const x = liste.find(([n]) => n === k); return x ? x[1] : null; };
function phases(r) {
  const P = (k) => premiere(r.page, k), T = (k) => premiere(r.travailleur, k), D = r.details || {};
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
  for (let rep = 0; rep < REP; rep++) {
    for (const fmt of FORMATS) {
      const liste = PLAN === 'quatre' ? QUATRE : tous;
      for (let i = 0; i < liste.length; i++) {
        const mod = liste[i], suivant = liste[(i + 1) % liste.length];
        const dossier = fs.mkdtempSync(path.join(os.tmpdir(), 'stab-profil-'));
        const ctx = await chromium.launchPersistentContext(dossier, { ...FMT[fmt], env: { ...process.env, LANG: 'C.UTF-8' } });
        const p = ctx.pages()[0] || await ctx.newPage();
        const charge = os.loadavg()[0];
        const rec = (scenario) => ({ rep, plan: PLAN, format: fmt, scenario, charge_1min: +charge.toFixed(2) });
        if (PLAN === 'quatre') {
          await visite(p, rec('froid'), mod);
          await visite(p, rec('2e_visite'), mod);
          await visite(p, rec('passage'), suivant);
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
