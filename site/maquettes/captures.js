// Captures des maquettes « Stab Méthodes » : chaque page aux largeurs PC 1440, tablette 1280 × 800 (la tablette du bureau, en
// paysage) et téléphone 390, à la densité 2 (des images que l'on regarde). Mesures : aucune erreur JavaScript, aucun débordement
// horizontal, le dessin principal visible et à sa taille, aucun nom de profilé optimal coupé par des points de suspension, et
// l'élément ATTENDU de chaque vue réellement vu : présent, visible, à l'écran (une feuille qui monte du bas, une liste en cartes),
// et aucun élément propre à l'AUTRE format visible (le choix de famille du téléphone sur le PC, la barre latérale au téléphone).
//
//   node site/maquettes/captures.js <dossier_sortie>
const { chromium } = require('playwright');
const path = require('path');
const OUT = process.argv[2] || '.';
const PAGES = path.join(__dirname, 'pages');
const PROPRES_TEL = '.mfam, .mhead, .mnav, .mtop, .mtabs, .mbar, .mlist, .tile .chev, .drawer .grab, .drawer .dclose, .fscrim';
const PROPRES_PC = '.nav, .top, .famhead, .tablecard, .phead, .zoom';
// [nom, page, largeur, hauteur, adresse, page entière (par défaut : oui au téléphone), élément attendu à l'écran]
const PLAN = [
  ['accueil', 'accueil.html', 1440, 900, '', true, '.resume'],
  ['projets', 'projets.html', 1440, 900, '', false, '.drawer .timeline'],
  ['module', 'garde-corps.html', 1440, 900, ''],
  ['module_sombre', 'garde-corps.html', 1440, 900, '?theme=dark'],
  ['module_profils', 'garde-corps.html', 1440, 900, '?profils=1', false, '.sheetR'],
  ['module_coupe', 'garde-corps.html', 1440, 900, '?vue=coupe'],
  ['module_tablette', 'garde-corps.html', 1280, 800, ''],
  ['accueil_tablette', 'accueil.html', 1280, 800, ''],
  ['tel_accueil', 'accueil.html', 390, 844, '', true, '.mnav'],
  ['tel_projets', 'projets.html', 390, 844, '', true, '.mlist'],
  ['tel_fiche', 'projets.html', 390, 844, '?fiche=1', false, '.drawer .timeline'],
  ['tel_resultat', 'garde-corps.html', 390, 844, '?tab=resultat'],
  ['tel_profils', 'garde-corps.html', 390, 844, '?tab=resultat&profils=1', false, '.sheetR .ptable'],
  ['tel_dessin', 'garde-corps.html', 390, 844, '?tab=dessin'],
  ['tel_donnees', 'garde-corps.html', 390, 844, '?tab=donnees'],
  ['tel_sombre', 'accueil.html', 390, 844, '?theme=dark', false, '.mhead'],
];
(async () => {
  const browser = await chromium.launch();
  let ko = 0;
  for (const [nom, fichier, w, h, q, pleine, attendu] of PLAN) {
    const mobile = w <= 640;
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: mobile, hasTouch: mobile });
    const page = await ctx.newPage();
    const erreurs = [];
    page.on('pageerror', e => erreurs.push(e.message));
    await page.goto('file://' + path.join(PAGES, fichier) + q);
    await page.waitForTimeout(500);
    const m = await page.evaluate(([attendu, autres]) => {
      const vis = e => e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const svg = [...document.querySelectorAll('.view.on svg, .thumb svg')].find(vis);
      const r = svg ? svg.getBoundingClientRect() : null;
      // un nom coupé se lit à l'œil : largeur du texte plus grande que la boîte qui le montre
      const coupes = [...document.querySelectorAll('.optline b')].filter(vis).filter(b => b.scrollWidth > b.clientWidth + 1).map(b => b.textContent);
      // l'élément attendu : visible, d'une taille réelle, et dans la fenêtre (au moins en partie)
      let vu = true;
      if (attendu) {
        const e = document.querySelector(attendu);
        const b = e && e.getBoundingClientRect();
        vu = !!(vis(e) && b.width > 20 && b.height > 20 && b.top < innerHeight && b.bottom > 0 && b.left < innerWidth && b.right > 0);
      }
      // ce qui appartient à l'autre format ne se voit pas ici
      const intrus = [...document.querySelectorAll(autres)].filter(vis).map(e => e.className.baseVal ?? e.className);
      return { scrollW: document.documentElement.scrollWidth, w: innerWidth, svg: r ? `${Math.round(r.width)}×${Math.round(r.height)}` : '—', coupes, vu, intrus };
    }, [attendu || '', mobile ? PROPRES_PC : PROPRES_TEL]);
    const ok = erreurs.length === 0 && m.scrollW <= m.w && m.coupes.length === 0 && m.vu && m.intrus.length === 0;
    if (!ok) ko++;
    console.log(`${ok ? 'OK' : 'KO'}  ${nom.padEnd(18)} ${w}×${h}  débordement ${m.scrollW}/${m.w}  dessin ${m.svg}`
      + `${attendu ? `  ${attendu} ${m.vu ? 'vu' : 'NON VU'}` : ''}`
      + `${m.coupes.length ? '  noms coupés : ' + m.coupes.join(', ') : ''}`
      + `${m.intrus.length ? '  visibles à tort : ' + [...new Set(m.intrus)].join(', ') : ''}  ${erreurs.join(' | ').slice(0, 160)}`);
    if (pleine === undefined ? mobile : pleine) {
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
