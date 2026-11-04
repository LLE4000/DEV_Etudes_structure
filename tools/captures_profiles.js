// Captures et MESURE de l'écran « Catalogue des profilés » (page « Tableau profilés ») aux largeurs
// PC 1440 / 1280, tablette 1024, téléphone 390 — même protocole que tools/captures_plancher.js :
// on mesure CE QUE L'ŒIL VOIT.
//
//   node tools/captures_profiles.js <dossier_sortie> [états] [largeurs] [port]
//   ex. node tools/captures_profiles.js scratchpad/profiles defaut,recherche,filtre,fiche,perso 1440,1024,390 8597
//
// États (chaque état PART de la page vierge) :
//   « defaut »    la page telle qu'elle s'ouvre (familles par défaut, tableau, fiche du premier profilé) ;
//   « recherche » « IPE 2 » tapé dans la recherche : le tableau se réduit aux IPE 200 à 270 (et IPE A/O 2xx si cochés) ;
//   « filtre »    filtre numérique Iy ≥ 10 000 cm⁴ : seuls les gros profilés restent ;
//   « fiche »     la fiche d'un HEB 300 choisie dans la liste : dessin et propriétés ;
//   « perso »     un PRS 500 × 200 × 8 / 12 saisi dans « Profilé personnalisé » et ajouté au tableau.
// Mesures : aucun débordement horizontal ; aucune exception ; tableau visible (≥ 200 px) ; fiche avec dessin
// SVG visible ; en « recherche » le compteur de profilés a baissé ; en « filtre » idem ; en « fiche » le titre
// de la fiche est « HEB 300 » ; en « perso » la fiche porte « PRS 500x200x8x12 » et la pastille « personnalisé ».
// Fichiers : apres_profiles_<état>_<largeur>.png. Code de sortie 1 dès qu'un seuil n'est pas tenu.
const { chromium } = require('playwright');
const OUT = process.argv[2] || '.';
const ETATS = (process.argv[3] || 'defaut,recherche,filtre,fiche,perso').split(',');
const LARGEURS = (process.argv[4] || '1440,1280,1024,390').split(',').map(Number);
const PORT = process.argv[5] || '8597';
const BASE = `http://localhost:${PORT}/?page=Tableau%20profil%C3%A9s`;
let KO = 0;

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
async function choisir(page, cle, option) {
  const ouvrir = page.locator(`.st-key-${cle} [data-testid="stSelectbox"]`).first();
  await ouvrir.scrollIntoViewIfNeeded(); await ouvrir.click(); await page.waitForTimeout(500);
  await page.keyboard.type(option); await page.waitForTimeout(500);
  await page.locator('[role="option"]', { hasText: option }).first().click();
  await settle(page, 2500);
}
async function jsClick(page, key) {
  await page.waitForSelector(`.st-key-${key} button`, { state: 'attached', timeout: 60000 });
  await page.evaluate(k => [...document.querySelectorAll(`.st-key-${k} button`)].find(b => b.getBoundingClientRect().width > 0).click(), key);
  await settle(page, 2500);
}
async function ouvrirExpander(page, texte) {
  await page.evaluate(t => {
    const s = [...document.querySelectorAll('summary')].find(x => x.innerText.includes(t));
    if (s && !s.parentElement.open) s.click();
  }, texte);
  await settle(page, 1500);
}
function compteur(page) {
  return page.evaluate(() => {
    const t = [...document.querySelectorAll('[data-testid="stMarkdownContainer"]')].map(e => e.innerText).find(x => /^\d[\d\s]* profilé/.test(x.trim()));
    return t ? parseInt(t.replace(/\s/g, ''), 10) : -1;
  });
}
async function preparer(page, etat) {
  if (etat === 'recherche') await saisir(page, 'prf_recherche', 'IPE 2');
  if (etat === 'filtre') {
    await ouvrirExpander(page, 'Filtres numériques');
    await choisir(page, 'prf_w_f_cle_0', 'Iy (cm⁴)');
    await saisir(page, 'prf_w_f_min_0', '10000');
  }
  if (etat === 'fiche') await choisir(page, 'prf_w_fiche', 'HEB 300');
  if (etat === 'perso') {
    await ouvrirExpander(page, 'Profilé personnalisé');
    await saisir(page, 'prf_perso_PRS_h', '500');
    await jsClick(page, 'prf_perso_btn');
  }
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
      console.log(`=== profiles (${etat}) @ ${w} px`);
      try {
        await page.goto(BASE, { waitUntil: 'networkidle' });
        await page.waitForSelector('text=Catalogue des profilés', { timeout: 90000 }); await settle(page, 3000);
        const n0 = await compteur(page);
        await preparer(page, etat);
        const m = await page.evaluate(() => {
          const vis = e => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const tab = document.querySelector('[data-testid="stDataFrame"]');
          const svg = [...document.querySelectorAll('svg')].find(s => s.getAttribute('viewBox') && s.querySelector('path') && vis(s) && s.getBoundingClientRect().width > 150);
          const fiche = document.querySelector('.prf-fiche');
          return { innerWidth: window.innerWidth, scrollW: document.documentElement.scrollWidth,
                   tabH: tab && vis(tab) ? Math.round(tab.getBoundingClientRect().height) : 0,
                   svgW: svg ? Math.round(svg.getBoundingClientRect().width) : 0,
                   fiche: fiche ? fiche.innerText.replace(/\s+/g, ' ').trim().slice(0, 120) : '',
                   exc: [...document.querySelectorAll('[data-testid="stException"]')].map(e => e.innerText.slice(0, 200)) };
        });
        const n1 = await compteur(page);
        verdict('aucune exception Streamlit', m.exc.length === 0, m.exc.join(' | '));
        verdict('aucune erreur JavaScript', erreurs.length === 0, erreurs.join(' | ').slice(0, 200));
        verdict('aucun débordement horizontal', m.scrollW <= m.innerWidth, `${m.scrollW}/${m.innerWidth}`);
        verdict('tableau visible (≥ 200 px)', m.tabH >= 200, `${m.tabH} px`);
        verdict('fiche avec dessin SVG visible', m.svgW >= 150 && m.fiche.length > 0, `${m.svgW} px · ${m.fiche.slice(0, 40)}`);
        if (etat === 'recherche' || etat === 'filtre') verdict(`${etat} : le compteur a baissé`, n1 > 0 && n1 < n0, `${n0} → ${n1}`);
        if (etat === 'fiche') verdict('fiche : HEB 300', /HEB 300/.test(m.fiche), m.fiche.slice(0, 60));
        if (etat === 'perso') verdict('personnalisé : PRS 500x200x8x12 dans la fiche', /PRS 500x200x8x12/.test(m.fiche) && /personnalisé/.test(m.fiche), m.fiche.slice(0, 80));
        await agrandir(page, w);
        await page.screenshot({ path: `${OUT}/apres_profiles_${etat}_${w}.png`, fullPage: true });
      } catch (e) {
        verdict('déroulement sans erreur', false, e.message.slice(0, 200));
        try { await page.screenshot({ path: `${OUT}/apres_profiles_${etat}_${w}_ERREUR.png`, fullPage: true }); } catch (e2) {}
      }
      await ctx.close();
    }
  }
  await browser.close();
  console.log(KO ? `\n${KO} contrôle(s) en défaut` : '\nTous les contrôles tenus');
  process.exit(KO ? 1 : 0);
})();
