// Captures et MESURE de l'écran « Soudures » aux largeurs PC 1440 / 1280, tablette 1024, téléphone 390 —
// même protocole que tools/captures_profiles.js : on mesure CE QUE L'ŒIL VOIT.
//
//   node tools/captures_soudures.js <dossier_sortie> [états] [largeurs] [port]
//   ex. node tools/captures_soudures.js scratchpad/soudures defaut,els,tube,libre,nok 1440,1024,390 8597
//
// États (chaque état PART de la page vierge, console fléchie C3) :
//   « defaut »  la page telle qu'elle s'ouvre : Eurocode, console, dessin, gorge, carte des taux, RÉSULTAT ;
//   « els »     le switch sur « Méthode ELS (bureau) » : même console, η passe de 61 % à 156 % ;
//   « tube »    configuration C6 (tube sur platine) avec le pourtour par tronçons : 4 tronçons de 50 mm — et le
//               champ Gorge doit AFFICHER 4 (valeur de l'état) après le passage C3 → C6, pas l'ancien 5 ;
//   « libre »   configuration C8 (cordons libres) : tableau des éléments ;
//   « nok »     V = 130 kN sur la console : Non conforme.
// Mesures : aucune exception ; aucune erreur JavaScript ; aucun débordement horizontal ; dessin technique
// SVG visible (≥ 260 px de large, ≥ 150 px de haut) ; encart de gorge visible ; carte RÉSULTAT visible avec
// η ; en « els » la pastille est « Non conforme » et le texte « Efforts de service » visible ; en « tube »
// la carte des taux nomme « tronçon 4 » ; en « libre » un tableau éditable visible ; en « nok » « Non conforme ».
// Fichiers : apres_soudures_<état>_<largeur>.png. Code de sortie 1 dès qu'un seuil n'est pas tenu.
const { chromium } = require('playwright');
const OUT = process.argv[2] || '.';
const ETATS = (process.argv[3] || 'defaut,els,tube,libre,nok').split(',');
const LARGEURS = (process.argv[4] || '1440,1280,1024,390').split(',').map(Number);
const PORT = process.argv[5] || '8597';
const BASE = `http://localhost:${PORT}/?page=Soudures`;
let KO = 0;
let gorgeApresC6 = null;

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
async function cliquerConfig(page, texte) {
  const b = page.locator('.st-key-sdr_w_config label', { hasText: texte }).first();
  await b.scrollIntoViewIfNeeded(); await b.click();
  await settle(page, 3000);
}
async function cocher(page, cle) {
  const c = page.locator(`.st-key-${cle} input[type="checkbox"]`).first();
  await c.scrollIntoViewIfNeeded();
  if (!(await c.isChecked())) { await page.locator(`.st-key-${cle} label`).first().click(); await settle(page, 2500); }
}
async function onglet(page, texte) {
  const t = page.locator('[data-testid="stTab"]', { hasText: texte }).first();
  await t.scrollIntoViewIfNeeded(); await t.click(); await settle(page, 1200);
}
async function preparer(page, etat) {
  if (etat === 'els') await cliquerSegment(page, 'sdr_w_methode_mode', 'Méthode ELS');
  if (etat === 'tube') {
    await cliquerConfig(page, 'C6');
    gorgeApresC6 = await page.locator('.st-key-sdr_w_params_a input').first().inputValue();
    await cocher(page, 'sdr_w_troncons_on');
    await saisir(page, 'sdr_w_params_troncons_n', '4');
    await saisir(page, 'sdr_w_params_troncons_l', '50');
  }
  if (etat === 'libre') await cliquerConfig(page, 'C8');
  if (etat === 'nok') { await onglet(page, '3 · Efforts'); await saisir(page, 'sdr_w_efforts_V', '130'); }
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
      console.log(`=== soudures (${etat}) @ ${w} px`);
      try {
        await page.goto(BASE, { waitUntil: 'networkidle' });
        await page.waitForSelector('text=Vérification des cordons de soudure', { timeout: 90000 }); await settle(page, 3000);
        await preparer(page, etat);
        const m = await page.evaluate(() => {
          const vis = e => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const svgs = [...document.querySelectorAll('svg')].filter(s => s.getAttribute('viewBox') && vis(s));
          const dessin = svgs.find(s => s.closest('.st-key-sdr_dessin'));
          const gorge = svgs.find(s => s.closest('.st-key-sdr_gorge'));
          const res = document.querySelector('.sdr-res');
          const pill = document.querySelector('.sdr-pill');
          const eta = document.querySelector('.sdr-eta');
          const edit = document.querySelector('[data-testid="stDataFrame"], [data-testid="stDataEditor"]');
          const texte = (document.querySelector('[data-testid="stMainBlockContainer"]') || document.body).innerText;
          return { innerWidth: window.innerWidth, scrollW: document.documentElement.scrollWidth,
                   dessinW: dessin ? Math.round(dessin.getBoundingClientRect().width) : 0,
                   dessinH: dessin ? Math.round(dessin.getBoundingClientRect().height) : 0,
                   gorgeW: gorge ? Math.round(gorge.getBoundingClientRect().width) : 0,
                   resH: res && vis(res) ? Math.round(res.getBoundingClientRect().height) : 0,
                   pill: pill ? pill.innerText : '', eta: eta ? eta.innerText : '',
                   edit: edit && vis(edit) ? Math.round(edit.getBoundingClientRect().height) : 0,
                   tronc4: /tronçon 4/.test(texte), service: /Efforts de service/.test(texte),
                   exc: [...document.querySelectorAll('[data-testid="stException"]')].map(e => e.innerText.slice(0, 200)) };
        });
        verdict('aucune exception Streamlit', m.exc.length === 0, m.exc.join(' | '));
        verdict('aucune erreur JavaScript', erreurs.length === 0, erreurs.join(' | ').slice(0, 200));
        verdict('aucun débordement horizontal', m.scrollW <= m.innerWidth, `${m.scrollW}/${m.innerWidth}`);
        verdict('dessin technique visible (≥ 260 × 150 px)', m.dessinW >= 260 && m.dessinH >= 150, `${m.dessinW} × ${m.dessinH} px`);
        verdict('encart de gorge visible (≥ 200 px)', m.gorgeW >= 200, `${m.gorgeW} px`);
        verdict('carte RÉSULTAT visible avec η', m.resH >= 80 && /η = \d+ %/.test(m.eta), `${m.resH} px · ${m.eta}`);
        if (etat === 'defaut') verdict('défaut : Conforme, η = 61 %', /Conforme/.test(m.pill) && /61 %/.test(m.eta), `${m.pill} · ${m.eta}`);
        if (etat === 'els') verdict('ELS : efforts de service, Non conforme, η = 156 %', m.service && /Non conforme/.test(m.pill) && /156 %/.test(m.eta), `${m.pill} · ${m.eta}`);
        if (etat === 'tube') verdict('tube : la carte des taux nomme le tronçon 4', m.tronc4, m.eta);
        if (etat === 'tube') verdict('tube : après C3 → C6 le champ Gorge AFFICHE la valeur de l\'état (4, défaut du tube), pas 5', parseFloat(gorgeApresC6) === 4, `affiché ${gorgeApresC6}`);
        if (etat === 'libre') verdict('libre : tableau des éléments visible', m.edit >= 60, `${m.edit} px`);
        if (etat === 'nok') verdict('nok : Non conforme', /Non conforme/.test(m.pill), `${m.pill} · ${m.eta}`);
        await agrandir(page, w);
        await page.screenshot({ path: `${OUT}/apres_soudures_${etat}_${w}.png`, fullPage: true });
      } catch (e) {
        verdict('déroulement sans erreur', false, e.message.slice(0, 200));
        try { await page.screenshot({ path: `${OUT}/apres_soudures_${etat}_${w}_ERREUR.png`, fullPage: true }); } catch (e2) {}
      }
      await ctx.close();
    }
  }
  await browser.close();
  console.log(KO ? `\n${KO} contrôle(s) en défaut` : '\nTous les contrôles tenus');
  process.exit(KO ? 1 : 0);
})();
