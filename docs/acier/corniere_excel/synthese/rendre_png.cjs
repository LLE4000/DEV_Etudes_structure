// Rend chaque SVG passé en argument en PNG (même nom), Chromium par Playwright, deviceScaleFactor 2.
// Lancer : node rendre_png.cjs figures/fig_a.svg [...]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const navigateur = await chromium.launch();
  const page = await navigateur.newPage({ deviceScaleFactor: 2 });
  for (const f of process.argv.slice(2)) {
    const svg = fs.readFileSync(f, 'utf8');
    const w = parseFloat(/width="([\d.]+)"/.exec(svg)[1]);
    const h = parseFloat(/height="([\d.]+)"/.exec(svg)[1]);
    await page.setViewportSize({ width: Math.ceil(w), height: Math.ceil(h) });
    await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;background:#fff}</style></head><body>${svg}</body></html>`);
    await page.waitForTimeout(150);
    const sortie = f.replace(/\.svg$/, '.png');
    await page.screenshot({ path: sortie, clip: { x: 0, y: 0, width: Math.ceil(w), height: Math.ceil(h) } });
    console.log(path.basename(sortie), Math.ceil(w) * 2, 'x', Math.ceil(h) * 2);
  }
  await navigateur.close();
})();
