// Rend les maquettes HTML en PNG aux largeurs PC (1440) et téléphone (390).
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const dir = process.argv[2];
  const files = process.argv.slice(3);
  const browser = await chromium.launch();
  for (const f of files) {
    const [name, widths] = f.split(':');
    for (const w of (widths || '1440').split(',').map(Number)) {
      const page = await browser.newPage({ viewport: { width: w, height: 900 }, deviceScaleFactor: 2 });
      await page.goto('file://' + path.join(dir, name + '.html'));
      await page.waitForTimeout(300);
      const out = path.join(dir, `${name}_${w}.png`);
      await page.screenshot({ path: out, fullPage: true });
      console.log('écrit', out);
      await page.close();
    }
  }
  await browser.close();
})();
