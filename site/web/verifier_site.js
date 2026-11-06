// Parcours du VRAI site, dans Chromium, comme le ferait le bureau : accueil vide, « Nouvelle note », calcul du
// garde-corps (saisie, type, calcul détaillé, choix d'un profilé, deuxième élément, rechargement), puis la poutre.
// Chaque pas est MESURÉ à l'écran : la bonne page, le résultat qui change, aucune erreur JavaScript, aucun débordement,
// aucun nom coupé (mesure exacte du texte), le dessin visible et à sa taille.
//
//   python3 site/web/outils/serveur_local.py 8770 &      (le site avec les en-têtes de _headers, comme Cloudflare)
//   node site/web/verifier_site.js http://127.0.0.1:8770 <dossier_des_captures> [garde|poutre|piege|modules|module:<clé>]
const { chromium } = require('playwright');
const path = require('path');
const BASE = process.argv[2] || 'http://127.0.0.1:8770';
const OUT = process.argv[3] || '.';
const FILTRE = process.argv[4] || '';

(async () => {
  const b = await chromium.launch();
  let ko = 0;
  const verif = (nom, cond, info = '') => { console.log(`${cond ? 'OK' : 'KO'}  ${nom}${info ? '  ' + info : ''}`); if (!cond) ko++; };
  const ctx = await b.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
  const p = await ctx.newPage();
  const erreurs = [];
  p.on('pageerror', (e) => erreurs.push(e.message));
  p.on('console', (m) => { if (m.type() === 'error') erreurs.push(m.text().slice(0, 300)); });
  const capture = (nom) => p.screenshot({ path: path.join(OUT, `site_${nom}.png`) });
  // ce que l'œil voit : débordement, noms coupés (mesure au texte près), dessin à sa taille
  const mesure = () => p.evaluate(() => {
    const vis = (e) => e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
    const large = (e) => { const r = document.createRange(); r.selectNodeContents(e); return r.getBoundingClientRect().width; };
    const coupes = [...document.querySelectorAll('.prof > span:not(.leger), .el .l1 b, .row > label > span:not(.sym), .ni .t, .pk-row .nm, .tile .nm')]
      .filter(vis).filter((e) => e.scrollWidth > e.clientWidth + 1 || large(e) > e.getBoundingClientRect().width + 0.01).map((e) => e.textContent.trim());
    const svg = [...document.querySelectorAll('.view.on svg')].find(vis);
    const r = svg ? svg.getBoundingClientRect() : null;
    // une valeur saisie s'écrit à l'encre : la couleur CALCULÉE du champ, comparée à celle de --ink
    const sonde = document.createElement('span'); sonde.style.color = 'var(--ink)'; document.body.appendChild(sonde);
    const encre = getComputedStyle(sonde).color; sonde.remove();
    const gris = [...document.querySelectorAll('.field:not(.locked) input, .field:not(.locked) select')].filter(vis)
      .filter((e) => getComputedStyle(e).color !== encre).map((e) => e.dataset.chemin || e.id);
    return { deborde: document.documentElement.scrollWidth > innerWidth, coupes, gris, svg: r ? [Math.round(r.width), Math.round(r.height)] : null };
  });
  const verdict = () => p.evaluate(() => document.querySelector('.verdict')?.textContent.replace(/\s+/g, ' ').trim() || '');
  const pret = () => p.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
  const calme = () => p.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });

  // ---------------------------------------------------------------- l'accueil, vide
  await p.goto(`${BASE}/index.html`);
  await p.waitForTimeout(300);
  // la politique de sécurité de _headers est active (outils/serveur_local.py l'applique, comme Cloudflare) : un script
  // écrit dans la page ne s'exécute pas. Tout le parcours qui suit tourne donc SOUS cette politique.
  const nCsp = erreurs.length;
  const inline = await p.evaluate(() => { const s = document.createElement('script'); s.textContent = 'window.__inline = 1'; document.head.appendChild(s); return window.__inline; });
  await p.waitForTimeout(150);
  verif('politique de sécurité active : un script écrit dans la page est refusé', inline === undefined,
    inline === undefined ? '' : 'servir avec site/web/outils/serveur_local.py');
  erreurs.splice(nCsp, erreurs.length - nCsp, ...erreurs.slice(nCsp).filter((x) => !/Content Security Policy/.test(x)));
  verif('accueil vide : il le dit', await p.isVisible('.vide'));
  // les tuiles ouvertes sont exactement les modules du registre (js/modules.js) ; les autres disent « bientôt »
  const registre = await p.evaluate(async () => Object.keys((await import('./js/modules.js')).MODULES).sort().join());
  const ouvertes = (await p.$$eval('.tile[data-nouvelle-module]', (t) => t.map((x) => x.dataset.nouvelleModule))).sort().join();
  const total = await p.$$eval('.tile', (t) => t.length);
  verif('accueil : les modules du registre sont ouverts, les autres « bientôt »',
    ouvertes === registre && (await p.$$eval('.tile.bientot', (t) => t.length)) === total - registre.split(',').length, `${ouvertes} / ${registre}`);
  let m = await mesure();
  verif('accueil : rien ne déborde, aucun nom coupé', !m.deborde && !m.coupes.length, m.coupes.join(', '));
  await capture('accueil_vide');

  async function nouvelle(module, chantier, dossier, nom) {
    await p.goto(`${BASE}/index.html`);
    await p.click('#ouvrir-nouvelle');
    verif(`« Nouvelle note » (${module}) : la fenêtre s'ouvre`, await p.isVisible('#nouvelle .mcard'));
    await p.fill('#nn-chantier', chantier); await p.fill('#nn-dossier', dossier); await p.fill('#nn-nom', nom);
    await p.click(`.mopt[data-module="${module}"]`);
    await Promise.all([p.waitForNavigation(), p.click('#nn-creer')]);
    const t0 = Date.now();
    await pret();
    const echec = await p.evaluate(() => document.getElementById('demarrage').classList.contains('erreur') ? document.getElementById('demarrage-etape').textContent : '');
    verif(`${module} : le moteur démarre dans le navigateur`, !echec, echec || `${Date.now() - t0} ms`);
    verif(`${module} : le nom de la note et le chantier sont repris`, (await p.inputValue('#nom-note')) === nom && (await p.textContent('#fil-chantier')).includes(chantier));
  }

  if (!FILTRE || FILTRE === 'garde') {
    // ---------------------------------------------------------------- le garde-corps
    await nouvelle('garde_corps', 'Réalex', 'Garde-corps', 'Garde-corps terrasse R+1');
    let v0 = await verdict();
    verif('garde-corps : le résultat du moteur (64 %)', /64\s?%/.test(v0), v0);
    verif('garde-corps : une ligne par élément', (await p.$$eval('#resultat .el', (e) => e.length)) >= 5);
    m = await mesure();
    verif('garde-corps : le dessin est visible et à sa taille', m.svg && m.svg[0] > 400 && m.svg[1] > 300, JSON.stringify(m.svg));
    verif('garde-corps : rien ne déborde, aucun nom coupé', !m.deborde && !m.coupes.length, m.coupes.join(', '));
    verif('garde-corps : chaque valeur saisie s\'écrit à l\'encre', !m.gris.length, m.gris.join(', '));
    await capture('garde_corps');
    // une saisie
    await p.fill('input[data-chemin="geometrie/H"]', '1300');
    await p.press('input[data-chemin="geometrie/H"]', 'Enter');
    await calme();
    verif('H = 1 300 mm : le résultat suit (89 %)', /89\s?%/.test(await verdict()), await verdict());
    // le calcul détaillé
    await p.click('#resultat .el[data-calcul="montant"]');
    await p.waitForTimeout(200);
    verif('la ligne du montant ouvre son calcul détaillé', await p.isVisible('.flyout.on.calcul .vtab'));
    await capture('garde_corps_calcul');
    await p.keyboard.press('Escape');
    // le sélecteur, et le plus léger qui tient
    await p.click('#resultat .prof[data-pk="montant"]');
    await p.waitForSelector('.flyout.on.picker .pk-row.cur', { timeout: 60000 });
    m = await mesure();
    verif('le sélecteur s\'ouvre sur le profilé retenu, aucun nom coupé', !m.coupes.length, m.coupes.join(', '));
    await capture('garde_corps_profils');
    const opt = await p.$('.flyout.on .pk-row.opt');
    if (opt) {
      const nomOpt = await opt.getAttribute('data-choix');
      await opt.click();
      await calme();
      verif(`choisir l'optimal (${nomOpt}) le retient`, (await p.textContent('#resultat .prof[data-pk="montant"] span')).trim() === nomOpt);
    }
    // le type change les éléments
    await p.selectOption('select[data-chemin="type"]', 'lisses');
    await calme();
    verif('type « lisses » : la ligne des lisses intermédiaires apparaît', (await p.textContent('#resultat')).includes('Lisses intermédiaires'));
    // un deuxième garde-corps
    await p.click('[data-ajouter]');
    await calme();
    verif('« Ajouter un garde-corps » : deux éléments dans la note', (await p.$$eval('.elements .ni', (e) => e.length)) === 2);
    // l'enregistrement : la page rechargée retrouve la note
    await p.reload();
    await pret();
    verif('rechargée, la note garde ses deux garde-corps', (await p.$$eval('.elements .ni', (e) => e.length)) === 2);
    await p.click('.elements .ni[data-activer="0"]');
    await calme();
    verif('… et le premier garde sa hauteur de 1 300 mm', (await p.inputValue('input[data-chemin="geometrie/H"]')).replace(/\s/g, '') === '1300');
    // sombre
    await p.click('.nav [data-theme-toggle]');
    await p.waitForTimeout(200);
    await capture('garde_corps_sombre');
    await p.click('.nav [data-theme-toggle]');
  }

  if (!FILTRE || FILTRE === 'poutre') {
    // ---------------------------------------------------------------- la poutre
    await nouvelle('poutre', 'Réalex', 'Étude de la dalle', 'Poutres R+1');
    let v = await verdict();
    verif('poutre neuve : un résultat (sans effort, trois vérifications)', /\d+\s?%/.test(v) && (await p.$$eval('#resultat .el', (e) => e.length)) === 3, v);
    // les efforts de la section A : la flexion et l'effort tranchant entrent en jeu
    for (const [ch, x] of [['sections/0/M_inf', '122'], ['sections/0/V', '60']]) {
      await p.fill(`input[data-chemin="${ch}"]`, x);
      await p.press(`input[data-chemin="${ch}"]`, 'Enter');
      await calme();
    }
    v = await verdict();
    const nEl = await p.$$eval('#resultat .el', (e) => e.length);
    const vSaisi = await p.inputValue('input[data-chemin="sections/0/V"]');
    verif('M = 122 kN·m et V = 60 kN : cinq vérifications, le tranchant et les étriers compris',
      nEl === 5 && (await p.textContent('#resultat')).includes('Effort tranchant'), `${nEl} lignes, V affiché ${vSaisi} · ${v}`);
    m = await mesure();
    verif('poutre : la coupe est visible et à sa taille', m.svg && m.svg[0] > 200 && m.svg[1] > 300, JSON.stringify(m.svg));
    verif('poutre : rien ne déborde, aucun nom coupé', !m.deborde && !m.coupes.length, m.coupes.join(', '));
    verif('poutre : chaque valeur saisie s\'écrit à l\'encre (efforts M et V compris)', !m.gris.length, m.gris.join(', '));
    await capture('poutre');
    const c = await p.$('#resultat .el[data-calcul]');
    if (c) {
      await c.click();
      await p.waitForTimeout(200);
      verif('poutre : une ligne ouvre le calcul de sa section', await p.isVisible('.flyout.on.calcul .vtab'));
      await capture('poutre_calcul');
      await p.keyboard.press('Escape');
    }
    const pk = await p.$('#resultat .prof[data-pk]');
    if (pk) {
      await pk.click();
      await p.waitForSelector('.flyout.on.picker .pk-row', { timeout: 60000 });
      verif('poutre : le choix des armatures s\'ouvre', await p.isVisible('.flyout.on.picker'));
      await capture('poutre_armatures');
      await p.keyboard.press('Escape');
    }
    const h = await p.$('input[data-chemin$="/h"], input[data-chemin="h"]');
    if (h) {
      const avant = await verdict();
      await h.fill('60');
      await h.press('Enter');
      await calme();
      verif('poutre : h = 60 cm change le résultat', (await verdict()) !== avant, `${avant} → ${await verdict()}`);
    }
  }

  // ---------------------------------------------------------------- chaque module du registre (contrôle commun)
  // Le même contrôle pour TOUS les modules (js/modules.js, écrit depuis le registre) : la note se crée, le moteur
  // démarre, le résultat et le dessin s'affichent à leur taille, rien ne déborde ni n'est coupé, la fenêtre de calcul
  // et le sélecteur s'ouvrent, une saisie recalcule sans erreur et survit au rechargement.
  //   filtre « modules » : tous ; « module:dalle » : celui-là.
  if (!FILTRE || FILTRE === 'modules' || FILTRE.startsWith('module:')) {
    await p.goto(`${BASE}/js/modules.js`);
    const src = await p.evaluate(() => document.body.innerText);
    const MODULES = JSON.parse(src.slice(src.indexOf('{'), src.lastIndexOf('}') + 1));
    const choisis = FILTRE.startsWith('module:') ? [FILTRE.slice(7)] : Object.keys(MODULES);
    verif(`modules : ${choisis.join(', ')} au registre`, choisis.every((k) => MODULES[k]), choisis.filter((k) => !MODULES[k]).join());
    for (const cle of choisis.filter((k) => MODULES[k])) {
      const nom = MODULES[cle].nom;
      await nouvelle(cle, 'Contrôle', 'Tous les modules', `Essai ${nom}`);
      if (await p.evaluate(() => document.getElementById('demarrage').classList.contains('erreur'))) continue;
      const nEl = await p.$$eval('#resultat .el', (e) => e.length);
      const verdictMod = await p.$('#resultat .verdict');
      verif(`${cle} : le résultat s'affiche`, nEl > 0 || !!verdictMod, `${nEl} ligne(s)`);
      const nVues = await p.$$eval('#vues button', (e) => e.length);
      m = await mesure();
      if (nVues) verif(`${cle} : le dessin est visible et à sa taille`, m.svg && m.svg[0] > 200 && m.svg[1] > 150, JSON.stringify(m.svg));
      verif(`${cle} : rien ne déborde, aucun nom coupé, les valeurs à l'encre`, !m.deborde && !m.coupes.length && !m.gris.length,
        [...m.coupes, ...m.gris].join(', '));
      await capture(cle);
      const c = await p.$('#resultat [data-calcul]');
      if (c) {
        await c.click();
        await p.waitForTimeout(250);
        verif(`${cle} : une ligne ouvre son calcul détaillé`, await p.isVisible('.flyout.on.calcul'));
        m = await mesure();
        verif(`${cle} : la fenêtre de calcul n'a aucun nom coupé`, !m.coupes.length, m.coupes.join(', '));
        await capture(`${cle}_calcul`);
        await p.keyboard.press('Escape');
      }
      const pk = await p.$('#resultat .prof[data-pk]');
      if (pk) {
        await pk.click();
        await p.waitForSelector('.flyout.on.picker', { timeout: 60000 });
        verif(`${cle} : le sélecteur s'ouvre`, await p.isVisible('.flyout.on.picker .pk-row, .flyout.on.picker .pk-vide'));
        await capture(`${cle}_selecteur`);
        await p.keyboard.press('Escape');
      }
      // une saisie : le premier nombre modifiable du cadre Données, augmenté de 10 %
      const champ = await p.$('#donnees input[data-sorte="nombre"]:not([disabled])');
      if (champ) {
        const chemin = await champ.getAttribute('data-chemin');
        const avant = (await champ.inputValue()).replace(/[\s  ]/g, '').replace(',', '.');
        const x = Number(avant);
        const neuf = Number.isFinite(x) && x !== 0 ? String(Math.round(x * 1.1 * 1000) / 1000).replace('.', ',') : '1';
        await champ.fill(neuf);
        await champ.press('Enter');
        await calme();
        await p.waitForTimeout(150);
        const refus = await p.evaluate(() => { const t = document.getElementById('toast'); return t && !t.hidden && t.classList.contains('erreur') ? t.textContent : ''; });
        const apres = (await p.inputValue(`#donnees input[data-chemin="${chemin}"]`)).replace(/[\s  ]/g, '');
        verif(`${cle} : une saisie (${chemin} = ${neuf}) recalcule sans erreur`, !refus && apres === neuf.replace(/[\s  ]/g, ''), refus || `${avant} → ${apres}`);
        await p.reload();
        await pret();
        await calme();
        const relu = (await p.inputValue(`#donnees input[data-chemin="${chemin}"]`)).replace(/[\s  ]/g, '');
        verif(`${cle} : rechargée, la note garde la saisie`, relu === apres, `${apres} → ${relu}`);
      }
    }
  }

  // ---------------------------------------------------------------- un fichier .stab piégé
  // Un .stab vient d'ailleurs (un collègue, un courriel) : rien de ce qu'il porte ne doit s'exécuter dans la page. Deux
  // fichiers : l'un que le moteur refuse (son dessin piégé n'est alors jamais remplacé par celui du moteur), l'autre
  // valide mais aux noms piégés (note, chantier, poutre, section).
  if (!FILTRE || FILTRE === 'piege') {
    const piege = (n) => `<img src="x" onerror="window.__piege=${n}">`;
    const stab = (note) => ({ name: 'piege.stab', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ format: 'stab', version: 1, note })) });
    const nAvant = erreurs.length;
    await p.goto(`${BASE}/index.html`);
    await Promise.all([p.waitForNavigation(), p.setInputFiles('#importer', stab({ module: 'poutre', nom: 'Note refusée', chantier: piege(1),
      dossier: piege(2), indice: piege(3), etat: { poutres: 'abîmé' }, vignette: piege(4), resume: { taux: piege(5), niveau: `"><b>${piege(6)}` } }))]);
    await pret();
    verif('.stab abîmé : le moteur le refuse et la page le dit', await p.evaluate(() => document.getElementById('demarrage').classList.contains('erreur')));
    // deux sections : la vue « Toutes les sections » écrit leurs noms dans le dessin
    const e2 = { poutres: [{ nom: piege(7), b: 25, h: 50, sections: [{ nom: piege(8), M_inf: 80, V: 40 }, { nom: piege(11), M_sup: 50, V: 30 }] }], actif: 0 };
    await p.goto(`${BASE}/index.html`);
    await Promise.all([p.waitForNavigation(), p.setInputFiles('#importer', stab({ module: 'poutre', nom: piege(9), chantier: piege(10), etat: e2 }))]);
    await pret();
    await calme();
    const etabli = await p.evaluate(() => ({ piege: window.__piege, img: document.querySelectorAll('img[src="x"]').length, nom: document.getElementById('nom-note').value }));
    verif('.stab aux noms piégés : la note s\'ouvre, les noms s\'affichent comme du texte, rien ne s\'exécute',
      etabli.piege === undefined && etabli.img === 0 && etabli.nom.startsWith('<img'), JSON.stringify(etabli));
    await p.goto(`${BASE}/index.html`);
    await p.waitForTimeout(300);
    const accueil = await p.evaluate(() => ({ piege: window.__piege, img: document.querySelectorAll('img[src="x"]').length }));
    verif('accueil après deux .stab piégés : aucun code du fichier ne s\'exécute (dessin, résumé, chantier, indice)',
      accueil.piege === undefined && accueil.img === 0, JSON.stringify(accueil));
    // les erreurs attendues (le moteur qui refuse le fichier abîmé) ne comptent pas ; une exception non rattrapée, si
    const recues = erreurs.splice(nAvant);
    const autres = recues.filter((x) => !/Traceback|ValueError|refus/.test(x));
    verif('.stab abîmé : seules les erreurs du refus, aucune exception de la page', !autres.length, `${recues.length} message(s) du refus${autres.length ? ' · ' + autres.join(' | ').slice(0, 300) : ''}`);
    // les deux notes piégées ont servi : elles quittent l'accueil (la capture finale montre les vraies notes)
    await p.evaluate(() => {
      const k = 'stab.notes.v1', notes = JSON.parse(localStorage.getItem(k) || '{}');
      for (const [id, x] of Object.entries(notes)) if (/__piege|Note refusée/.test(`${x.nom} ${x.chantier}`)) delete notes[id];
      localStorage.setItem(k, JSON.stringify(notes));
    });
  }

  // ---------------------------------------------------------------- retour à l'accueil
  await p.goto(`${BASE}/index.html`);
  await p.waitForTimeout(300);
  const n = await p.$$eval('#recentes .rcard', (e) => e.length);
  verif('accueil : les notes créées sont à reprendre', n >= 1, `${n} note(s)`);
  if (!FILTRE || ['garde', 'poutre'].includes(FILTRE)) verif('accueil : le chantier « Réalex » est listé', (await p.textContent('#chantiers')).includes('Réalex'));
  m = await mesure();
  verif('accueil : rien ne déborde, aucun nom coupé', !m.deborde && !m.coupes.length, m.coupes.join(', '));
  await capture('accueil');
  verif('aucune erreur JavaScript', erreurs.length === 0, erreurs.join(' | ').slice(0, 400));
  await b.close();
  process.exit(ko ? 1 : 0);
})();
