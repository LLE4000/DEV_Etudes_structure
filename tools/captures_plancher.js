// Captures et MESURE de l'écran « Charges sur plancher » aux quatre largeurs (PC 1440 / 1280, tablette
// 1024, téléphone 390) — même protocole que tools/captures_assemblages.js : on mesure CE QUE L'ŒIL VOIT.
//
//   node tools/captures_plancher.js <dossier_sortie> [états] [largeurs] [port]
//   ex. node tools/captures_plancher.js scratchpad/plancher vide,engin,plateau 1440,1280,1024,390 8597
//
// États (chaque état PART de la page vierge et enchaîne les précédents) :
//   « vide »    la page telle qu'elle s'ouvre (hourdis générique, aucune charge : « Données manquantes ») ;
//   « engin »   M_Rd / V_Rd d'ESSAI saisis (60 kN·m, 80 kN : valeurs de test, PAS un catalogue), un
//               télescopique rotatif posé depuis la bibliothèque, charges par appui d'essai 95/95/60/60 kN ;
//   « plateau » un plateau acier 600 × 600 × 20 sous l'appui P2 (étape 3) ;
//   « clic »    clic sur la semelle P2 du plan : la fenêtre de réglages du composant s'ouvre ;
//   « glisser » l'engin est glissé de 60 px vers la droite : sa position x change dans l'étape 2 ;
//   « coupe »   la vue Coupe ; « balayage » la vue Balayage (après « engin »).
// Mesures : aucun débordement horizontal (scrollWidth == innerWidth) ; le dessin cliquable existe et a une
// hauteur ≥ 200 px ; la carte RÉSULTAT porte une pastille ; en état « engin », la pastille n'est plus
// « Données manquantes » (le calcul a abouti) ; en état « plateau », le plan porte un plateau (hachures).
// Fichiers : apres_plancher_<état>_<largeur>.png. Code de sortie 1 dès qu'un seuil n'est pas tenu.
const { chromium } = require('playwright');
const OUT = process.argv[2] || '.';
const ETATS = (process.argv[3] || 'vide,engin,plateau').split(',');
const LARGEURS = (process.argv[4] || '1440,1280,1024,390').split(',').map(Number);
const PORT = process.argv[5] || '8597';
const BASE = `http://localhost:${PORT}/?page=Charges%20sur%20plancher`;
const ORDRE = ['vide', 'engin', 'plateau', 'clic', 'glisser', 'coupe', 'balayage'];
let KO = 0;

async function settle(page, ms = 700) {
  await page.waitForTimeout(300);
  try { await page.waitForSelector('[data-testid="stStatusWidget"]', { state: 'detached', timeout: 40000 }); } catch (e) {}
  await page.waitForTimeout(ms);
}
async function ouvrirEtape(page, prefixe) {
  // active l'onglet d'étape dont le libellé COMMENCE par `prefixe` (les quatre étapes sont des onglets
  // à libellé FIXE : un expander dont le titre change se referme à chaque saisie — défaut vu le 02/11/2026)
  const fait = await page.evaluate(pref => {
    const t = [...document.querySelectorAll('[data-testid="stTabs"] [role="tab"]')].find(x => x.innerText.trim().startsWith(pref)
                                                                      && x.getBoundingClientRect().width > 0);
    if (!t) return false;
    t.click();
    return true;
  }, prefixe);
  await settle(page, 1500);
  return fait;
}
async function saisir(page, cle, val) {
  const inp = page.locator(`.st-key-${cle} input`).first();
  await inp.scrollIntoViewIfNeeded(); await inp.fill(String(val)); await inp.press('Enter');
  await settle(page, 2000);
}
async function choisir(page, cle, option) {
  const ouvrir = page.locator(`.st-key-${cle} [data-testid="stSelectbox"]`).first();
  await ouvrir.scrollIntoViewIfNeeded(); await ouvrir.click(); await page.waitForTimeout(500);
  await page.locator('[role="option"]', { hasText: option }).first().click();
  await settle(page, 2500);
}
async function jsClick(page, key) {
  await page.waitForSelector(`.st-key-${key} button`, { state: 'attached', timeout: 60000 });
  await page.waitForTimeout(200);
  await page.evaluate(k => [...document.querySelectorAll(`.st-key-${k} button`)].find(b => b.getBoundingClientRect().width > 0).click(), key);
  await settle(page, 2000);
}
async function segment(page, texte) {
  // clic sur le bouton d'un segmented_control portant exactement ce texte
  await page.evaluate(t => {
    const b = [...document.querySelectorAll('[data-testid="stButtonGroup"] button, [data-testid="stSegmentedControl"] button')]
      .find(x => x.innerText.trim() === t && x.getBoundingClientRect().width > 0);
    if (b) b.click();
  }, texte);
  await settle(page, 2500);
}
async function preparer(page, etat) {
  const n = ORDRE.indexOf(etat);
  if (n >= 1) {                                   // « engin »
    await ouvrirEtape(page, '1 · Plancher');
    await saisir(page, 'chp_w_plancher_hourdis_M_Rd', '60');   // valeurs d'ESSAI, pas un catalogue
    await saisir(page, 'chp_w_plancher_hourdis_V_Rd', '80');
    await ouvrirEtape(page, '2 · Charges');          // le bouton de la bibliothèque vit dans l'onglet 2
    await jsClick(page, 'chp_btn_lib');
    await page.waitForSelector('[data-testid="stDialog"]', { timeout: 30000 });
    await page.waitForTimeout(800);
    await page.locator('[data-testid="stDialog"] button', { hasText: 'Poser' }).first().click();
    await settle(page, 3000);
    await ouvrirEtape(page, '2 · Charges');
    for (const [p, v] of [['P1', '95'], ['P2', '95'], ['P3', '60'], ['P4', '60']]) await saisir(page, `chp_w_charges_0_P_${p}`, v);
  }
  if (etat === 'plateau') {
    await ouvrirEtape(page, '3 · Répartiteur');
    await choisir(page, 'chp_sel_rep', 'E1.P2');
    await segment(page, 'Plateau');
  }
  if (etat === 'clic') {
    // la semelle elle-même (le groupe inclut l'étiquette : son centre tombe entre les deux)
    await page.frameLocator('.st-key-chp_col_dessin iframe').locator('g[data-group="rep:E1.P2"] > rect').first().click();
    await page.waitForTimeout(1500);
  }
  if (etat === 'glisser') {
    const el = page.frameLocator('.st-key-chp_col_dessin iframe').locator('g[data-group="engin:E1"]').first();
    // sur téléphone, le plan est sous le pli : on l'amène dans la fenêtre AVANT de lire sa boîte
    // (sinon la souris tombe hors fenêtre et rien ne bouge — faux rouge du 02/11/2026)
    await el.scrollIntoViewIfNeeded(); await page.waitForTimeout(400);
    const bb = await el.boundingBox();
    const cx = bb.x + bb.width / 2, cy = bb.y + bb.height / 2;
    // le glissement vaut 0,8 m DU PLAN, pas 60 px : à 1024 et 390 px, 60 px valaient 1,5 m et un
    // stabilisateur sortait du plancher (refus légitime du moteur, faux rouge du harnais, 02/11/2026)
    const sc = await page.evaluate(() => { const fr = document.querySelector('.st-key-chp_col_dessin iframe');
      const svg = fr.contentDocument.querySelector('svg'); return svg.getBoundingClientRect().width / svg.viewBox.baseVal.width; });
    const dx = Math.max(12, Math.round(800 * sc));
    await page.mouse.move(cx, cy); await page.mouse.down();
    await page.mouse.move(cx + dx / 2, cy, { steps: 8 }); await page.mouse.move(cx + dx, cy, { steps: 8 });
    await page.mouse.up();
    await settle(page, 3500);
    await ouvrirEtape(page, '2 · Charges');
  }
  if (etat === 'coupe') await segment(page, 'Coupe');
  if (etat === 'balayage') await segment(page, 'Balayage');
  // on REMONTE avant de mesurer (Streamlit défile dans son conteneur)
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
      const mobile = w <= 640 && process.env.CHP_TACTILE !== '0';   // CHP_TACTILE=0 : téléphone sans émulation tactile (diagnostic)
      const ctx = await browser.newContext({ viewport: { width: w, height: mobile ? 844 : 1000 },
        deviceScaleFactor: mobile ? 2 : 1, isMobile: mobile, hasTouch: mobile });
      const page = await ctx.newPage();
      const erreurs = [];
      page.on('pageerror', e => erreurs.push(e.message));
      console.log(`=== plancher (${etat}) @ ${w} px`);
      try {
        await page.goto(BASE, { waitUntil: 'networkidle' });
        await page.waitForSelector('text=Charges sur plancher', { timeout: 90000 }); await settle(page, 2500);
        await preparer(page, etat);
        const m = await page.evaluate(() => {
          const vis = e => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const fig = document.querySelector('.st-key-chp_col_dessin iframe');
          const pill = document.querySelector('.chp-pill');
          return { innerWidth: window.innerWidth, scrollW: document.documentElement.scrollWidth,
                   figH: fig && vis(fig) ? Math.round(fig.getBoundingClientRect().height) : 0,
                   pill: pill ? pill.innerText.trim() : '',
                   exc: [...document.querySelectorAll('[data-testid="stException"]')].map(e => e.innerText.slice(0, 200)),
                   resume: [...document.querySelectorAll('summary')].filter(vis).map(s => s.innerText.trim().split('\n').pop().trim().slice(0, 60)) };
        });
        verdict('aucune exception Streamlit', m.exc.length === 0, m.exc.join(' | '));
        verdict('aucune erreur JavaScript', erreurs.length === 0, erreurs.join(' | ').slice(0, 200));
        verdict('aucun débordement horizontal', m.scrollW <= m.innerWidth, `${m.scrollW}/${m.innerWidth}`);
        if (etat !== 'coupe' && etat !== 'balayage') verdict('dessin cliquable visible (≥ 200 px)', m.figH >= 200, `${m.figH} px`);
        verdict('carte résultat avec pastille', m.pill.length > 0, m.pill);
        if (ORDRE.indexOf(etat) >= 1 && etat !== 'coupe' && etat !== 'balayage')
          verdict('le calcul a abouti (pastille ≠ données manquantes)', !/manquantes/i.test(m.pill), m.pill);
        if (etat === 'clic') {
          const pop = await page.evaluate(() => {
            const fr = document.querySelector('.st-key-chp_col_dessin iframe');
            if (!fr || !fr.contentDocument) return 'pas de cadre';
            const p = fr.contentDocument.querySelector('.popover');
            if (!p || !p.checkVisibility({ visibilityProperty: true, opacityProperty: true })) return 'fenêtre absente';
            return p.innerText.trim().replace(/\s+/g, ' ').slice(0, 120);
          });
          verdict('clic sur la semelle P2 : fenêtre de réglages ouverte', !/absente|cadre/.test(pop), pop);
        }
        if (etat === 'glisser') {
          const txt = await page.evaluate(() => [...document.querySelectorAll('[data-testid="stMarkdownContainer"]')]
            .map(e => e.innerText).find(t => /^E1 ·/.test(t.trim())) || 'ligne E1 absente');
          verdict('glisser l\'engin : sa position x a changé (≠ 3,60 m)', /x \d,\d\d/.test(txt) && !/x 3,60/.test(txt), txt.slice(0, 90));
        }
        if (etat === 'plateau') {
          const hach = await page.evaluate(() => {
            const fr = document.querySelector('.st-key-chp_col_dessin iframe');
            if (!fr || !fr.contentDocument) return -1;
            return fr.contentDocument.querySelectorAll('rect[fill="url(#chp_hc)"]').length;
          });
          verdict('le plan porte un plateau (hachures)', hach >= 1, `${hach} rect`);
        }
        await agrandir(page, w);
        await page.screenshot({ path: `${OUT}/apres_plancher_${etat}_${w}.png`, fullPage: true });
      } catch (e) {
        verdict('déroulement sans erreur', false, e.message.slice(0, 200));
        try { await page.screenshot({ path: `${OUT}/apres_plancher_${etat}_${w}_ERREUR.png`, fullPage: true }); } catch (e2) {}
      }
      await ctx.close();
    }
  }
  await browser.close();
  console.log(KO ? `\n${KO} contrôle(s) en défaut` : '\nTous les contrôles tenus');
  process.exit(KO ? 1 : 0);
})();
