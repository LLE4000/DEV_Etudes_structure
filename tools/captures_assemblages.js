// Captures « après » et MESURE des écrans d'assemblage aux quatre largeurs (charte ergonomique du
// 31/10/2026, docs/assemblages/ERGONOMIE.md §2) — issu de tel_carte.js + mesure_proto.js.
//
//   node tools/captures_assemblages.js <dossier_sortie> [modules] [largeurs] [port]
//   ex. node tools/captures_assemblages.js scratchpad/ergo/apres plat_ame 1440,1280,1024,390 8597
//
// Mesure CE QUE L'ŒIL VOIT (getBoundingClientRect + checkVisibility), jamais ce que le DOM garde :
//   - scrollWidth == innerWidth (aucun débordement horizontal) ;
//   - cases des lignes d'objet (<p>lg_*) ≥ 120 px ; 4 par rangée ≥ 1181 px, 2 en dessous ;
//   - titres de bloc (<p>bt_*) dans leur rangée ;
//   - barre d'outils (<p>barre) : 1 rangée ≥ 1024, ≤ 3 rangées et ≤ 130 px à 390 ;
//   - bandeau et premier dessin dans les 844 premiers px après le titre à 390 ;
//   - onglets (<p>ui_onglet) sans débordement.
// La fenêtre est agrandie à la hauteur du contenu avant la capture (Streamlit défile dans son
// conteneur : fullPage ne suffit pas). Fichiers : apres_<module>_<largeur>.png. Code de sortie 1
// dès qu'un seuil n'est pas tenu.
const { chromium } = require('playwright');
const OUT = process.argv[2] || '.';
const MODULES = (process.argv[3] || 'plat_ame').split(',');
const LARGEURS = (process.argv[4] || '1440,1280,1024,390').split(',').map(Number);
const PORT = process.argv[5] || '8597';
const BASE = `http://localhost:${PORT}/?page=Assemblages%20m%C3%A9talliques`;
const PREFIXE = { doubles_cornieres: 'asm_', plat_ame: 'fpl_', platine_about: 'pab_', platine_beton: 'ppb_' };
let KO = 0;

async function settle(page, ms = 700) {
  await page.waitForTimeout(300);
  try { await page.waitForSelector('[data-testid="stStatusWidget"]', { state: 'detached', timeout: 40000 }); } catch (e) {}
  await page.waitForTimeout(ms);
}
async function jsClick(page, key) {
  await page.waitForSelector(`.st-key-${key} button[data-testid^="stBaseButton"]`, { state: 'attached', timeout: 60000 });
  await page.waitForTimeout(300);
  await page.evaluate(k => [...document.querySelectorAll(`.st-key-${k} button[data-testid^="stBaseButton"]`)].find(b => b.getBoundingClientRect().width > 0).click(), key);
  await settle(page);
}
function verdict(nom, ok, info) {
  if (!ok) KO++;
  console.log(`   ${ok ? 'OK ' : 'KO '} ${nom}${info ? '   [' + info + ']' : ''}`);
}

(async () => {
  const browser = await chromium.launch();
  for (const mod of MODULES) {
    const p = PREFIXE[mod];
    for (const w of LARGEURS) {
      const mobile = w <= 640;
      const ctx = await browser.newContext({ viewport: { width: w, height: mobile ? 844 : 1000 },
        deviceScaleFactor: mobile ? 2 : 1, isMobile: mobile, hasTouch: mobile });
      const page = await ctx.newPage();
      console.log(`=== ${mod} @ ${w} px`);
      try {
        await page.goto(BASE, { waitUntil: 'networkidle' });
        await page.waitForSelector('text=Premier assemblage du projet', { timeout: 90000 }); await settle(page, 800);
        await jsClick(page, `asm_btn_open_${mod}`);
        await page.waitForTimeout(3000); await settle(page, 2500);
        const m = await page.evaluate((p) => {
          const rect = e => { const r = e.getBoundingClientRect(); return { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height), b: Math.round(r.bottom) }; };
          const vis = e => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          const rows = rs => { const bs = [...new Set(rs.map(r => r.b))].sort((a, b) => a - b); let n = 0, last = -99; for (const b of bs) { if (b - last > 12) { n++; last = b; } } return n; };
          const out = { innerWidth: window.innerWidth, scrollW: document.documentElement.scrollWidth, lignes: [] };
          const CELL = '[data-testid="stSelectbox"], [data-testid="stNumberInput"], [data-testid="stTextInput"]';
          for (const el of document.querySelectorAll(`[class*="st-key-${p}lg_"]`)) {
            if (!vis(el)) continue;
            const key = [...el.classList].find(c => c.startsWith('st-key-'));
            const cells = [...el.querySelectorAll(CELL)].filter(vis).map(rect);
            const t = el.querySelector(`[class*="st-key-${p}bt_"]`);
            const r = rect(el);
            out.lignes.push({ key, n: cells.length, largeurs: cells.map(c => c.w), rangees: rows(cells),
              titre: t && vis(t) ? { dansRangee: rect(t).b <= r.b + 1 && rect(t).y >= r.y - 1, txt: t.innerText.trim().slice(0, 16) } : null,
              debord: cells.some(c => c.x + c.w > window.innerWidth + 1) });
          }
          const bar = document.querySelector(`.st-key-${p}barre`);
          if (bar) {
            const items = [...bar.querySelectorAll('button, [data-baseweb="select"]')].filter(vis).map(rect);
            out.barre = { rangees: rows(items), hauteur: rect(bar).h, items: items.length };
          }
          const seg = document.querySelector(`.st-key-${p}ui_onglet`);
          if (seg) out.onglets = { right: Math.round(seg.getBoundingClientRect().right), rangees: rows([...seg.querySelectorAll('button')].filter(vis).map(rect)) };
          const h2 = document.querySelector('h2');
          const bandeau = [...document.querySelectorAll('[data-testid="stMarkdown"]')].find(e => /VÉRIFIÉ/.test(e.innerText) && vis(e));
          const dessin = document.querySelector(`.st-key-${p}col_dessin iframe, .st-key-${p}col_dessin svg`);
          const y0 = h2 ? rect(h2).y : 0;
          out.premierEcran = { bandeau: bandeau ? rect(bandeau).b - y0 : null, dessin: dessin ? rect(dessin).y - y0 : null };
          out.debordants = [...document.querySelectorAll('[data-testid="stMainBlockContainer"] *')].filter(e => vis(e) && e.getBoundingClientRect().right > window.innerWidth + 1).length;
          return out;
        }, p);
        verdict('aucun débordement horizontal (scrollWidth == innerWidth)', m.scrollW === m.innerWidth && m.debordants === 0, `${m.scrollW} / ${m.innerWidth}, ${m.debordants} débordant(s)`);
        const petites = m.lignes.filter(l => l.largeurs.some(x => x < 120));
        verdict('cases de nombre ≥ 120 px', petites.length === 0, petites.map(l => l.key + ':' + l.largeurs).join(' '));
        const attendu = w >= 1181 ? 4 : 2;
        const mauvaises = m.lignes.filter(l => l.n === 4 && l.rangees !== 4 / attendu);
        verdict(`lignes de 4 : ${attendu} cases par rangée`, mauvaises.length === 0, mauvaises.map(l => l.key + ':' + l.rangees + ' rangées').join(' '));
        const hors = m.lignes.filter(l => l.titre && !l.titre.dansRangee);
        verdict('titres de bloc dans leur rangée', hors.length === 0, hors.map(l => l.key + ':' + l.titre.txt).join(' '));
        if (m.barre) verdict(w >= 1024 ? 'barre d\'outils sur 1 rangée' : 'barre d\'outils ≤ 3 rangées et ≤ 130 px',
          w >= 1024 ? m.barre.rangees === 1 : (m.barre.rangees <= 3 && m.barre.hauteur <= 130), `${m.barre.rangees} rangée(s), ${m.barre.hauteur} px`);
        else verdict('barre d\'outils partagée présente', false, 'pas de .st-key-' + p + 'barre');
        if (m.onglets) verdict('onglets sans débordement', m.onglets.right <= m.innerWidth, `bord droit ${m.onglets.right}, ${m.onglets.rangees} rangée(s)`);
        if (mobile) verdict('bandeau et premier dessin dans les 844 px après le titre', m.premierEcran.bandeau !== null && m.premierEcran.bandeau <= 844 && m.premierEcran.dessin !== null && m.premierEcran.dessin <= 844, JSON.stringify(m.premierEcran));
        console.log('   lignes :', JSON.stringify(m.lignes.map(l => [l.key.replace('st-key-' + p, ''), l.largeurs, l.rangees])));
        const h = await page.evaluate(() => Math.max(...[...document.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main, .stMainBlockContainer')].map(e => e.scrollHeight), document.documentElement.scrollHeight));
        await page.setViewportSize({ width: w, height: Math.min(h + 100, 12000) });
        await page.waitForTimeout(2500);
        await page.screenshot({ path: `${OUT}/apres_${mod}_${w}.png` });
        console.log(`   capture ${OUT}/apres_${mod}_${w}.png (hauteur ${h})`);
      } catch (e) { KO++; console.log('   échec :', e.message.slice(0, 300)); }
      await ctx.close();
    }
  }
  await browser.close();
  console.log(KO ? `\n${KO} seuil(s) non tenu(s)` : '\nTous les seuils sont tenus');
  process.exit(KO ? 1 : 0);
})();
