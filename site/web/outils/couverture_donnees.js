// La couverture des Données (refonte de l'établi, lot 0) : pour chaque module et chaque format, l'ensemble des
// data-chemin de #donnees sur une note neuve, et ceux que l'œil voit à l'ouverture (onglet Données au téléphone, sans
// rien déplier). La référence « avant » : docs/site/mesures/2026-11-07_couverture_avant.json.
//
//   python3 site/web/outils/serveur_local.py 8770 &
//   flock /tmp/verrou_navigateur node site/web/outils/couverture_donnees.js http://127.0.0.1:8770 <sortie.json>
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const BASE = process.argv[2];
const SORTIE = process.argv[3];
const FORMATS = {
  pc: { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 },
  tablette_paysage: { viewport: { width: 1180, height: 820 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
  tablette_portrait: { viewport: { width: 820, height: 1180 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
  telephone: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true },
};

(async () => {
  const MODULES = (await import('file://' + path.join(__dirname, '..', 'js', 'modules.js'))).MODULES;
  const b = await chromium.launch({ env: { ...process.env, LANG: 'C.UTF-8' } });
  const sortie = {};
  for (const [fmt, opt] of Object.entries(FORMATS)) {
    const ctx = await b.newContext(opt);
    const p = await ctx.newPage();
    const erreurs = [];
    p.on('pageerror', (e) => erreurs.push(e.message));
    for (const cle of Object.keys(MODULES).sort()) {
      await p.goto(`${BASE}/${MODULES[cle].page}`);
      await p.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
      await p.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
      if (fmt === 'telephone') await p.click('.mtabs [data-onglet="donnees"]');
      await p.waitForTimeout(150);
      const lu = await p.evaluate(() => {
        const echec = document.getElementById('demarrage').classList.contains('erreur');
        const els = [...document.querySelectorAll('#donnees [data-chemin]')];
        const vu = (e) => { const r = e.getBoundingClientRect(); return e.checkVisibility({ visibilityProperty: true, opacityProperty: true }) && r.width > 0 && r.height > 0; };
        const tri = (xs) => [...new Set(xs)].sort();
        const pm = (c) => /\/[+-]$|^[+-]$/.test(c);   // les boutons « + » (ajouter une ligne) et « − » (la retirer)
        return {
          echec,
          chemins: tri(els.map((e) => e.dataset.chemin)),
          hors_plus_moins: tri(els.map((e) => e.dataset.chemin).filter((c) => !pm(c))),
          plus_moins: tri(els.map((e) => e.dataset.chemin).filter(pm)),
          visibles_a_l_ouverture: tri(els.filter(vu).map((e) => e.dataset.chemin)),
          sections: [...document.querySelectorAll('#donnees .sec[data-sec]')].map((s) => s.dataset.sec),
        };
      });
      sortie[cle] ||= {};
      sortie[cle][fmt] = lu;
      console.log(`${fmt.padEnd(18)} ${cle.padEnd(14)} ${lu.chemins.length} chemins (${lu.hors_plus_moins.length} hors +/−), ${lu.visibles_a_l_ouverture.length} visibles à l'ouverture${lu.echec ? ' ÉCHEC' : ''}`);
    }
    if (erreurs.length) console.log('erreurs', fmt, erreurs.slice(0, 3));
    await ctx.close();
  }
  await b.close();
  fs.writeFileSync(SORTIE, JSON.stringify(sortie, null, 1));
})();
