// Captures et MESURE de l'écran « Garde-corps » aux largeurs PC 1440 / 1280, tablette 1024, téléphone 390 —
// même protocole que tools/captures_soudures.js : on mesure CE QUE L'ŒIL VOIT.
//
//   node tools/captures_garde_corps.js <dossier_sortie> [états] [largeurs] [port]
//   ex. node tools/captures_garde_corps.js scratchpad/gc defaut,L,mur,biblio,nok 1440,1024,390 8597
//
// États (chaque état PART de la page vierge, le garde-corps par défaut) :
//   « defaut »  la page telle qu'elle s'ouvre : ELS, droit 3,60 m à 0,90 m, CHS 48,3 × 3,2, dessins, RÉSULTAT 75 % ;
//   « L »       forme en L avec des entraxes saisis « 1200 900 » sur le premier tronçon : montant d'angle ;
//   « mur »     système « lisse ancrée dans un mur » : consoles ;
//   « biblio »  onglet Sections, bouton « Prendre le plus léger qui tient » : le montant devient CHS 48.3x2.5 — et
//               le sélecteur de section doit AFFICHER cette valeur (liaison état → widget) ;
//   « nok »     catégorie C5 (3 kN/m) : Non conforme.
// Mesures : aucune exception ; aucune erreur JavaScript ; aucun débordement horizontal ; élévation SVG visible
// (≥ 260 px de large, ≥ 120 px de haut) ; carte RÉSULTAT visible avec η ; états spécifiques ci-dessous.
// Fichiers : apres_garde_corps_<état>_<largeur>.png. Code de sortie 1 dès qu'un seuil n'est pas tenu.
const { chromium } = require('playwright');
const OUT = process.argv[2] || '.';
const ETATS = (process.argv[3] || 'defaut,L,mur,biblio,nok').split(',');
const LARGEURS = (process.argv[4] || '1440,1280,1024,390').split(',').map(Number);
const PORT = process.argv[5] || '8597';
const BASE = `http://localhost:${PORT}/?page=Garde-corps`;
let KO = 0;
let sectionApresBiblio = null;

async function settle(page, ms = 700) {
  await page.waitForTimeout(300);
  try { await page.waitForSelector('[data-testid="stStatusWidget"]', { state: 'detached', timeout: 60000 }); } catch (e) {}
  await page.waitForTimeout(ms);
}
async function saisir(page, cle, val) {
  const inp = page.locator(`.st-key-${cle} input`).first();
  await inp.scrollIntoViewIfNeeded(); await inp.fill(String(val)); await inp.press('Enter');
  await settle(page, 2500);
}
async function cliquerSegment(page, cle, texte) {
  const b = page.locator(`.st-key-${cle} button`, { hasText: texte }).first();
  await b.scrollIntoViewIfNeeded(); await b.click();
  await settle(page, 2500);
}
async function onglet(page, texte) {
  const t = page.locator('[data-testid="stTab"]', { hasText: texte }).first();
  await t.scrollIntoViewIfNeeded(); await t.click(); await settle(page, 1200);
}
async function choisir(page, cle, texte) {
  // selectbox de Streamlit 1.66 : une combobox react-aria — on tape le texte, puis on prend l'option
  const inp = page.locator(`.st-key-${cle} input`).first();
  await inp.scrollIntoViewIfNeeded(); await inp.click(); await inp.fill(texte);
  await page.waitForTimeout(400);
  const opt = page.locator('[role="option"]', { hasText: texte }).first();
  if (await opt.count()) { await opt.click(); } else { await inp.press('Enter'); }
  await settle(page, 2500);
}
async function valeurSelect(page, cle) {
  return page.locator(`.st-key-${cle} input`).first().inputValue();
}
async function preparer(page, etat) {
  if (etat === 'L') {
    await cliquerSegment(page, 'gc_w_geometrie_forme', 'en L');
    await saisir(page, 'gc_w_geometrie_troncons_0_entraxes', '1200 900');
  }
  if (etat === 'mur') await cliquerSegment(page, 'gc_w_geometrie_systeme', 'lisse ancrée');
  if (etat === 'biblio') {
    await onglet(page, '3 · Sections');
    const b = page.locator('.st-key-gc_btn_best button').first();
    await b.scrollIntoViewIfNeeded(); await b.click(); await settle(page, 3000);
    await onglet(page, '3 · Sections');
    sectionApresBiblio = await valeurSelect(page, 'gc_w_montant_section');
  }
  if (etat === 'nok') { await onglet(page, '2 · Charges'); await choisir(page, 'gc_w_charges_categorie', 'C5'); }
  await page.evaluate(() => { window.scrollTo(0, 0); for (const e of document.querySelectorAll('*')) { if (e.scrollTop) e.scrollTop = 0; } });
  await settle(page, 800);
}
function verdict(nom, ok, info) {
  if (!ok) KO++;
  console.log(`   ${ok ? 'OK ' : 'KO '} ${nom}${info ? '   [' + info + ']' : ''}`);
}
async function agrandir(page, w) {
  const h = await page.evaluate(() => {
    const m = document.querySelector('[data-testid="stMainBlockContainer"]') || document.body;
    return Math.ceil(m.getBoundingClientRect().height + 80);
  });
  await page.setViewportSize({ width: w, height: Math.min(Math.max(h, 800), 7000) });
  await settle(page, 800);
}

(async () => {
  const browser = await chromium.launch();
  for (const etat of ETATS) {
    for (const w of LARGEURS) {
      const mobile = w <= 640;
      const ctx = await browser.newContext({ viewport: { width: w, height: mobile ? 844 : 1000 },
        deviceScaleFactor: mobile ? 2 : 1, isMobile: mobile, hasTouch: mobile });
      const page = await ctx.newPage();
      const erreurs = [];
      page.on('pageerror', e => erreurs.push(e.message));
      console.log(`=== garde-corps (${etat}) @ ${w} px`);
      try {
        await page.goto(BASE, { waitUntil: 'networkidle' });
        await page.waitForSelector('text=Dimensionnement d\'un garde-corps', { timeout: 90000 }); await settle(page, 3000);
        await preparer(page, etat);
        const m = await page.evaluate(() => {
          const vis = e => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const svgs = [...document.querySelectorAll('svg')].filter(s => s.getAttribute('viewBox') && vis(s));
          const dessin = svgs.find(s => s.closest('.st-key-gc_dessin'));
          const res = document.querySelector('.gc-res');
          const pill = document.querySelector('.gc-pill');
          const eta = document.querySelector('.gc-eta');
          const texte = (document.querySelector('[data-testid="stMainBlockContainer"]') || document.body).innerText;
          return { innerWidth: window.innerWidth, scrollW: document.documentElement.scrollWidth,
                   dessinW: dessin ? Math.round(dessin.getBoundingClientRect().width) : 0,
                   dessinH: dessin ? Math.round(dessin.getBoundingClientRect().height) : 0,
                   nSvg: svgs.length,
                   resH: res && vis(res) ? Math.round(res.getBoundingClientRect().height) : 0,
                   pill: pill ? pill.innerText : '', eta: eta ? eta.innerText : '',
                   angle: /d'angle/.test(texte), console: /Console C/.test(texte), enL: /en L/.test(texte),
                   exc: [...document.querySelectorAll('[data-testid="stException"]')].map(e => e.innerText.slice(0, 200)) };
        });
        verdict('aucune exception Streamlit', m.exc.length === 0, m.exc.join(' | '));
        verdict('aucune erreur JavaScript', erreurs.length === 0, erreurs.join(' | ').slice(0, 200));
        verdict('aucun débordement horizontal', m.scrollW <= m.innerWidth, `${m.scrollW}/${m.innerWidth}`);
        verdict('élévation visible (≥ 260 × 120 px)', m.dessinW >= 260 && m.dessinH >= 120, `${m.dessinW} × ${m.dessinH} px`);
        verdict('au moins trois dessins SVG visibles', m.nSvg >= 3, `${m.nSvg}`);
        verdict('carte RÉSULTAT visible avec η', m.resH >= 80 && /η = \d+ %/.test(m.eta), `${m.resH} px · ${m.eta}`);
        if (etat === 'defaut') verdict('défaut : Conforme, η = 75 %', /Conforme/.test(m.pill) && /75 %/.test(m.eta), `${m.pill} · ${m.eta}`);
        if (etat === 'L') verdict('L : entraxes 1200 + 900, un montant d\'angle, « en L »', m.angle && m.enL, `${m.eta}`);
        if (etat === 'mur') verdict('mur : des consoles C…', m.console, `${m.eta}`);
        if (etat === 'biblio') verdict('biblio : le sélecteur AFFICHE CHS 48.3x2.5 après le bouton (liaison état → widget)', /48\.3x2\.5/.test(sectionApresBiblio || ''), `affiché ${sectionApresBiblio}`);
        if (etat === 'nok') verdict('nok : Non conforme (C5, 3 kN/m)', /Non conforme/.test(m.pill), `${m.pill} · ${m.eta}`);
        await agrandir(page, w);
        await page.screenshot({ path: `${OUT}/apres_garde_corps_${etat}_${w}.png`, fullPage: true });
      } catch (e) {
        verdict('déroulement sans erreur', false, e.message.slice(0, 200));
        try { await page.screenshot({ path: `${OUT}/apres_garde_corps_${etat}_${w}_ERREUR.png`, fullPage: true }); } catch (e2) {}
      }
      await ctx.close();
    }
  }
  await browser.close();
  console.log(KO ? `\n${KO} contrôle(s) en défaut` : '\nTous les contrôles tenus');
  process.exit(KO ? 1 : 0);
})();
