// Parcours de l'aperçu cliquable (apercu.py), comme le ferait le bureau : téléphone en clair et en sombre, puis PC. Chaque pas
// est mesuré à l'écran : la bonne vue affichée, aucun débordement, la feuille des profilés ouverte DEPUIS l'onglet Résultat, la
// fiche du rapport (ses indices) montée du bas, la poutre (calcul d'une section, choix des armatures), les paramètres, le
// volet des modules, la « Nouvelle note », le bouton de thème qui bascule le fond, aucune erreur JavaScript.
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
      await p.click('.vpage[data-page="garde-corps"] .prof[data-pk="montant"]');
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
    if (fmt === 'pc') {
      // la barre : Paramètres, puis le volet des modules, qui mène à la poutre
      await p.click('.vpage[data-page="equipe"] .nav [data-go="parametres"]');
      await p.waitForTimeout(250);
      e = await etat();
      verif(`${fmt}/${scheme} rail → paramètres`, e.page === 'parametres' && e.sw <= w && await vis('.pgrid'), JSON.stringify(e));
      await p.click('.vpage[data-page="parametres"] .nav [data-modules]');
      await p.waitForTimeout(250);
      verif(`${fmt}/${scheme} rail → volet des modules`, await vis('.modpanel'));
      await p.screenshot({ path: path.join(OUT, `apercu_${fmt}_${scheme}_modules.png`) });
      await p.click('.vpage[data-page="parametres"] .modpanel [data-go="poutre"]');
      await p.waitForTimeout(250);
      e = await etat();
      verif(`${fmt}/${scheme} volet → poutres béton armé`, e.page === 'poutre' && !(await vis('.modpanel')), JSON.stringify(e));
      // la ligne d'une vérification ouvre le calcul de sa section ; la pièce ouvre le choix des armatures
      await p.click('.vpage[data-page="poutre"] .el[data-calcul="A"]');
      await p.waitForTimeout(200);
      const lignes = await p.evaluate(() => document.querySelectorAll('.flyout.on .vtab tbody tr').length);
      verif(`${fmt}/${scheme} poutre : calcul de la section A (5 vérifications)`, await vis('.flyout.on .vtab') && lignes === 5, `${lignes} lignes`);
      await p.screenshot({ path: path.join(OUT, `apercu_${fmt}_${scheme}_poutre_calcul.png`) });
      await p.click('.flyout.on [data-fermer]');
      await p.click('.vpage[data-page="poutre"] .prof[data-pk="inf-A"]');
      await p.waitForTimeout(200);
      verif(`${fmt}/${scheme} poutre : choix des armatures, la disposition retenue en vue`, await vis('.flyout.on .pk-row.cur'));
      await p.click('.flyout.on [data-fermer]');
      // « Nouvelle note » : chantier, dossier, module, puis l'établi du module choisi
      await p.evaluate(() => window.ALLER('accueil'));
      await p.waitForTimeout(200);
      await p.click('.vpage[data-page="accueil"] [data-nouvelle]');
      await p.waitForTimeout(200);
      verif(`${fmt}/${scheme} « Nouvelle note » ouvre le choix du module`, await vis('.mcard'));
      await p.click('.vpage[data-page="accueil"] .mcard [data-go="poutre"]');
      await p.waitForTimeout(250);
      e = await etat();
      verif(`${fmt}/${scheme} « Créer la note » → poutre, la fenêtre fermée`, e.page === 'poutre' && !(await vis('.mcard')), JSON.stringify(e));
    } else {
      // au téléphone : l'onglet Paramètres, puis la poutre, onglet Résultat, et le calcul d'une section monté du bas
      await p.evaluate(() => window.ALLER('accueil'));
      await p.waitForTimeout(200);
      await p.click('.vpage[data-page="accueil"] .mnav [data-go="parametres"]');
      await p.waitForTimeout(250);
      e = await etat();
      verif(`${fmt}/${scheme} onglet → paramètres`, e.page === 'parametres' && e.sw <= w && await vis('.pgrid'), JSON.stringify(e));
      await p.evaluate(() => window.ALLER('poutre'));
      await p.waitForTimeout(250);
      await p.click('.vpage[data-page="poutre"] .mtabs button[data-tab="resultat"]');
      await p.click('.vpage[data-page="poutre"] .el[data-calcul="B"]');
      await p.waitForTimeout(250);
      e = await etat();
      verif(`${fmt}/${scheme} poutre : calcul de la section B depuis Résultat`, await vis('.flyout.on .vtab') && e.sw <= w, JSON.stringify(e));
      await p.screenshot({ path: path.join(OUT, `apercu_${fmt}_${scheme}_poutre_calcul.png`) });
      await p.click('.flyout.on [data-fermer]');
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
