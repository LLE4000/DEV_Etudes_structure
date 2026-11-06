// Parcours de l'aperçu cliquable (apercu.py), comme le ferait le bureau : téléphone en clair et en sombre, puis PC. Chaque pas
// est mesuré à l'écran : la bonne vue affichée, aucun débordement, la feuille des profilés ouverte DEPUIS l'onglet Résultat, la
// fiche du rapport (ses indices) montée du bas, le bouton de thème qui bascule le fond, aucune erreur JavaScript.
//
//   python3 site/maquettes/apercu.py <apercu.html> && node site/maquettes/parcours_apercu.js <apercu.html> <dossier_sortie>
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const SRC = process.argv[2], OUT = process.argv[3];
const html = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>'
  + fs.readFileSync(SRC, 'utf8') + '</body></html>';
const fichier = path.join(OUT, 'apercu_enveloppe.html');
fs.writeFileSync(fichier, html);
(async () => {
  const b = await chromium.launch();
  let ko = 0;
  const verif = (nom, cond, info = '') => { console.log(`${cond ? 'OK' : 'KO'}  ${nom} ${info}`); if (!cond) ko++; };
  for (const [fmt, w, h, scheme] of [['tel', 390, 844, 'light'], ['tel', 390, 844, 'dark'], ['pc', 1440, 900, 'light']]) {
    const ctx = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1, isMobile: fmt === 'tel', hasTouch: fmt === 'tel', colorScheme: scheme });
    const p = await ctx.newPage();
    const err = [];
    p.on('pageerror', e => err.push(e.message));
    await p.goto('file://' + fichier);
    await p.waitForTimeout(400);
    const vis = s => p.evaluate(s => { const e = [...document.querySelectorAll(s)].find(x => x.checkVisibility({ visibilityProperty: true, opacityProperty: true })); return !!e; }, s);
    const etat = () => p.evaluate(() => ({ page: [...document.querySelectorAll('.vpage')].filter(v => !v.hidden).map(v => v.dataset.page).join(','), sw: document.documentElement.scrollWidth, bg: getComputedStyle(document.body).backgroundColor }));
    let e = await etat();
    verif(`${fmt}/${scheme} départ sur l'accueil`, e.page === 'accueil' && e.sw <= w, JSON.stringify(e));
    if (scheme === 'dark') verif(`${fmt}/${scheme} fond sombre`, e.bg === 'rgb(10, 14, 22)', e.bg);
    await p.screenshot({ path: path.join(OUT, `apercu_${fmt}_${scheme}_accueil.png`) });
    // le rapport récent ouvre le module
    await p.click('.vpage[data-page="accueil"] .rcard[data-go]');
    await p.waitForTimeout(250);
    e = await etat();
    verif(`${fmt}/${scheme} rapport récent → module`, e.page === 'garde-corps' && e.sw <= w, JSON.stringify(e));
    if (fmt === 'tel') {
      await p.click('.mtabs button[data-tab="resultat"]');
      await p.click('.vpage[data-page="garde-corps"] .el[data-el="montant"] .prof');
      await p.waitForTimeout(250);
      verif(`${fmt}/${scheme} sélecteur de profilés depuis Résultat`, await vis('.flyout.on .pk-row.cur'));
      await p.screenshot({ path: path.join(OUT, `apercu_${fmt}_${scheme}_profils.png`) });
      await p.click('.flyout.on [data-fermer]');
      await p.click('.mtop [data-go="projets"]');
    } else {
      await p.click('.vpage[data-page="garde-corps"] .nav [data-go="projets"]');
    }
    await p.waitForTimeout(250);
    e = await etat();
    verif(`${fmt}/${scheme} retour → projets`, e.page === 'projets' && e.sw <= w, JSON.stringify(e));
    if (fmt === 'tel') {
      const dx = await p.evaluate(() => { const on = document.querySelector('.tree .node.on'); return Math.round(on.getBoundingClientRect().left); });
      verif(`${fmt}/${scheme} dossier ouvert ramené dans la bande`, dx >= 0 && dx < 60, `x=${dx}`);
      await p.click('.mrow[data-fiche]');
      await p.waitForTimeout(400);
      verif(`${fmt}/${scheme} fiche du rapport (indices) montée`, await vis('.drawer .timeline'));
      await p.screenshot({ path: path.join(OUT, `apercu_${fmt}_${scheme}_fiche.png`) });
    }
    await p.click('.vpage[data-page="projets"] .drawer [data-go="garde-corps"]');
    await p.waitForTimeout(250);
    e = await etat();
    verif(`${fmt}/${scheme} « Ouvrir » → module`, e.page === 'garde-corps', JSON.stringify(e));
    if (fmt === 'pc') {
      // « Voir le calcul » ouvre les formules ; les deux cadres se replient et le dessin prend la largeur
      await p.click('.vpage[data-page="garde-corps"] [data-calcul="montant"]');
      await p.waitForTimeout(200);
      verif(`${fmt}/${scheme} « Voir le calcul » ouvre les formules`, await vis('.flyout.on .ck-body .fx'));
      await p.click('.flyout.on [data-fermer]');
      const avant = await p.evaluate(() => Math.round(document.querySelector('.view.on .sheet').getBoundingClientRect().width));
      await p.click('[data-coll="g"]'); await p.click('[data-coll="d"]');
      await p.waitForTimeout(400);
      const apres = await p.evaluate(() => Math.round(document.querySelector('.view.on .sheet').getBoundingClientRect().width));
      verif(`${fmt}/${scheme} cadres repliés : le dessin s'élargit`, apres > avant + 400, `${avant} → ${apres} px`);
      await p.click('.vpage[data-page="garde-corps"] .nav [data-go="equipe"]');
      await p.waitForTimeout(250);
      e = await etat();
      verif(`${fmt}/${scheme} rail → équipe`, e.page === 'equipe' && await vis('.team .invrow'), JSON.stringify(e));
    }
    // le bouton de thème bascule
    const avant = await p.evaluate(() => getComputedStyle(document.body).backgroundColor);
    await p.evaluate(() => window.ALLER('accueil'));
    await p.waitForTimeout(200);
    await p.click(fmt === 'tel' ? '.vpage[data-page="accueil"] .mhead [data-theme-toggle]' : '.vpage[data-page="accueil"] .nav [data-theme-toggle]');
    await p.waitForTimeout(200);
    const apres = await p.evaluate(() => getComputedStyle(document.body).backgroundColor);
    verif(`${fmt}/${scheme} bouton de thème`, avant !== apres, `${avant} → ${apres}`);
    verif(`${fmt}/${scheme} aucune erreur`, err.length === 0, err.join(' | '));
    await ctx.close();
  }
  await b.close();
  process.exit(ko ? 1 : 0);
})();
