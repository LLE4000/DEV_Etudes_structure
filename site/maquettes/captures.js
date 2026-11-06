// Captures des maquettes « Stab Méthodes » : chaque page aux largeurs PC 1440, tablette 1280 × 800 (la tablette du bureau, en
// paysage) et téléphone 390, à la densité 2 (des images que l'on regarde). Mesures : aucune erreur JavaScript, aucun débordement
// horizontal, le dessin principal visible et à sa taille, aucun nom de profilé optimal coupé par des points de suspension.
//
//   node site/maquettes/captures.js <dossier_sortie>
const { chromium } = require('playwright');
const path = require('path');
const OUT = process.argv[2] || '.';
const PAGES = path.join(__dirname, 'pages');
const PLAN = [
  ['accueil', 'accueil.html', 1440, 900, '', true],
  ['projets', 'projets.html', 1440, 900, ''],
  ['module', 'garde-corps.html', 1440, 900, ''],
  ['module_sombre', 'garde-corps.html', 1440, 900, '?theme=dark'],
  ['module_profils', 'garde-corps.html', 1440, 900, '?profils=1'],
  ['module_coupe', 'garde-corps.html', 1440, 900, '?vue=coupe'],
  ['module_tablette', 'garde-corps.html', 1280, 800, ''],
  ['accueil_tablette', 'accueil.html', 1280, 800, ''],
  ['tel_resultat', 'garde-corps.html', 390, 844, '?tab=resultat'],
  ['tel_dessin', 'garde-corps.html', 390, 844, '?tab=dessin'],
  ['tel_donnees', 'garde-corps.html', 390, 844, '?tab=donnees'],
];
(async () => {
  const browser = await chromium.launch();
  let ko = 0;
  for (const [nom, fichier, w, h, q, pleine] of PLAN) {
    const mobile = w <= 640;
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: mobile, hasTouch: mobile });
    const page = await ctx.newPage();
    const erreurs = [];
    page.on('pageerror', e => erreurs.push(e.message));
    await page.goto('file://' + path.join(PAGES, fichier) + q);
    await page.waitForTimeout(400);
    const m = await page.evaluate(() => {
      const vis = e => e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const svg = [...document.querySelectorAll('.view.on svg, .thumb svg')].find(vis);
      const r = svg ? svg.getBoundingClientRect() : null;
      // un nom coupé se lit à l'œil : largeur du texte plus grande que la boîte qui le montre
      const coupes = [...document.querySelectorAll('.optline b')].filter(vis).filter(b => b.scrollWidth > b.clientWidth + 1).map(b => b.textContent);
      return { scrollW: document.documentElement.scrollWidth, w: innerWidth, svg: r ? `${Math.round(r.width)}×${Math.round(r.height)}` : '—', coupes };
    });
    const ok = erreurs.length === 0 && m.scrollW <= m.w && m.coupes.length === 0;
    if (!ok) ko++;
    console.log(`${ok ? 'OK' : 'KO'}  ${nom.padEnd(18)} ${w}×${h}  débordement ${m.scrollW}/${m.w}  dessin ${m.svg}`
      + `${m.coupes.length ? '  noms coupés : ' + m.coupes.join(', ') : ''}  ${erreurs.join(' | ').slice(0, 160)}`);
    if (mobile || pleine) {
      // une page entière : on agrandit la fenêtre à la hauteur du contenu (la navigation collante couvre alors toute la hauteur)
      const hh = await page.evaluate(() => document.documentElement.scrollHeight);
      await page.setViewportSize({ width: w, height: Math.max(h, hh) });
      await page.waitForTimeout(300);
    }
    await page.screenshot({ path: path.join(OUT, `maquette_${nom}.png`) });
    await ctx.close();
  }
  await browser.close();
  process.exit(ko ? 1 : 0);
})();
