// Parcours du VRAI site, dans Chromium, comme le ferait le bureau : accueil vide, « Nouvelle note », calcul du
// garde-corps (saisie, type, calcul détaillé, choix d'un profilé, deuxième élément, rechargement), puis la poutre.
// Chaque pas est MESURÉ à l'écran : la bonne page, le résultat qui change, aucune erreur JavaScript, aucun débordement,
// aucun nom coupé (mesure exacte du texte), le dessin visible et à sa taille.
//
//   python3 site/web/outils/serveur_local.py 8770 &      (le site avec les en-têtes de _headers, comme Cloudflare)
//   node site/web/verifier_site.js http://127.0.0.1:8770 <dossier_des_captures> [garde|poutre|piege|modules|module:<clé>|formats|formats:<clé>|telecharger|meteo|sonde|enregistrement]
const { chromium } = require('playwright');
const path = require('path');
const BASE = process.argv[2] || 'http://127.0.0.1:8770';
const OUT = process.argv[3] || '.';
const FILTRE = process.argv[4] || '';

(async () => {
  // une locale UTF-8 pour Chromium : sous la locale « C » d'un serveur, il ne sait pas écrire un nom de fichier accentué
  // sur le disque et enregistre « download » (mesuré : « Contrôle - « essai ».json » → « download ») — les postes du bureau
  // sont en UTF-8
  const b = await chromium.launch({ env: { ...process.env, LANG: process.env.LANG || 'C.UTF-8' } });
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
    const coupes = [...document.querySelectorAll('.prof > span:not(.leger), .el .l1 b, .row > label > span:not(.sym), .ni .t, .pk-row .nm, .tile .nm, .rows .seg.full button')]
      .filter(vis).filter((e) => e.scrollWidth > e.clientWidth + 1 || large(e) > e.getBoundingClientRect().width + 0.01).map((e) => e.textContent.trim());
    const svg = [...document.querySelectorAll('.view.on svg')].find(vis);
    const r = svg ? svg.getBoundingClientRect() : null;
    // une valeur saisie s'écrit à l'encre : la couleur CALCULÉE du champ, comparée à celle de --ink
    const sonde = document.createElement('span'); sonde.style.color = 'var(--ink)'; document.body.appendChild(sonde);
    const encre = getComputedStyle(sonde).color; sonde.remove();
    const gris = [...document.querySelectorAll('.field:not(.locked) input, .field:not(.locked) select')].filter(vis)
      .filter((e) => getComputedStyle(e).color !== encre).map((e) => e.dataset.chemin || e.id);
    // le contenu de la vue ouverte (un dessin, ou un tableau) : sa taille lue à l'écran
    const bloc = [...document.querySelectorAll('.view.on .draw > *')].find(vis);
    const rb = bloc ? bloc.getBoundingClientRect() : null;
    // un champ de TEXTE (un nom) s'écrit sans chiffres tabulaires : le trait d'union, rendu dans la police CALCULÉE du
    // champ, a sa largeur ordinaire (en tabulaire, « Garde-corps 1 » se lisait « Garde - corps 1 »)
    const trait = (cs, fvn) => { const t = document.createElement('span'); t.style.cssText = `position:absolute;white-space:pre;font:${cs.font};font-variant-numeric:${fvn}`;
      t.textContent = '-'; document.body.appendChild(t); const l = t.getBoundingClientRect().width; t.remove(); return l; };
    const tabulaires = [...document.querySelectorAll('#donnees .field.text input, #donnees .field.area textarea')].filter(vis)
      .filter((e) => { const cs = getComputedStyle(e); return trait(cs, cs.fontVariantNumeric) > trait(cs, 'normal') + 0.2; }).map((e) => e.dataset.chemin);
    return { deborde: document.documentElement.scrollWidth > innerWidth, coupes, gris, tabulaires, svg: r ? [Math.round(r.width), Math.round(r.height)] : null,
      vue: rb ? [Math.round(rb.width), Math.round(rb.height)] : null };
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
    verif('garde-corps : un nom s\'écrit sans chiffres tabulaires (« Garde-corps 1 », pas « Garde - corps 1 »)', !m.tabulaires.length, m.tabulaires.join(', '));
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
    // « Lisses intermédiaires (2) » et son profilé ne tiennent pas sur une ligne du cadre de 316 px : le profilé passe
    // dessous, entier (il se coupait en « RHS 50×25×2,… », l'épaisseur cachée)
    m = await mesure();
    verif('type « lisses » : aucun nom coupé dans le résultat (le profilé des lisses se lit en entier)', !m.coupes.length && !m.deborde, m.coupes.join(', '));
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

  const avecDessin = new Set();   // les pages des modules qui ont au moins un dessin : leur vignette en est un
  // ---------------------------------------------------------------- chaque module du registre (contrôle commun)
  // Le même contrôle pour TOUS les modules (js/modules.js, écrit depuis le registre) : la note se crée, le moteur
  // démarre, le résultat et le dessin s'affichent à leur taille, rien ne déborde ni n'est coupé, la fenêtre de calcul
  // et le sélecteur s'ouvrent, une saisie recalcule sans erreur et survit au rechargement.
  //   filtre « modules » : tous ; « module:dalle » : celui-là.
  if (!FILTRE || FILTRE === 'modules' || FILTRE.startsWith('module:')) {
    // la liste comme la page la lit : un module JavaScript (toujours en UTF-8), et non le fichier affiché en texte
    await p.goto(`${BASE}/`);
    const MODULES = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    verif('modules : les noms sont lus tels que la page les lit (accents intacts)', Object.values(MODULES).every((x) => !/[ÃÂ]/.test(x.nom)),
      Object.values(MODULES).map((x) => x.nom).join(', '));
    const choisis = FILTRE.startsWith('module:') ? [FILTRE.slice(7)] : Object.keys(MODULES);
    verif(`modules : ${choisis.join(', ')} au registre`, choisis.every((k) => MODULES[k]), choisis.filter((k) => !MODULES[k]).join());
    for (const cle of choisis.filter((k) => MODULES[k])) {
      const nom = MODULES[cle].nom;
      await nouvelle(cle, 'Contrôle', 'Tous les modules', `Essai ${nom}`);
      if (await p.evaluate(() => document.getElementById('demarrage').classList.contains('erreur'))) continue;
      // les vues : un dessin porte l'échelle, un tableau jamais ; le module a-t-il un dessin (pour sa vignette) ?
      const vuesLues = await p.$$eval('#stage .view', (vs) => vs.map((v) => ({ svg: !!v.querySelector('.draw > svg'), echelle: !!v.querySelector('.scale') })));
      if (vuesLues.some((v) => v.svg)) avecDessin.add(MODULES[cle].page);
      verif(`${cle} : « à l'échelle » n'est jamais écrit sur un tableau`, vuesLues.every((v) => v.svg || !v.echelle), JSON.stringify(vuesLues));
      const nEl = await p.$$eval('#resultat .el', (e) => e.length);
      const verdictMod = await p.$('#resultat .verdict');
      verif(`${cle} : le résultat s'affiche`, nEl > 0 || !!verdictMod, `${nEl} ligne(s)`);
      const nVues = await p.$$eval('#vues button', (e) => e.length);
      m = await mesure();
      if (nVues) verif(`${cle} : la vue (dessin ou tableau) est visible et à sa taille`, m.vue && m.vue[0] > 200 && m.vue[1] > 120, JSON.stringify(m.vue));
      verif(`${cle} : rien ne déborde, aucun nom coupé, les valeurs à l'encre, les noms sans chiffres tabulaires`,
        !m.deborde && !m.coupes.length && !m.gris.length && !m.tabulaires.length, [...m.coupes, ...m.gris, ...m.tabulaires].join(', '));
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

  // ---------------------------------------------------------------- les FORMATS : téléphone et tablettes
  // Chaque module, au téléphone (390 × 844, tactile), en tablette portrait (820 × 1 180), en tablette paysage
  // (1 180 × 820), en petite tablette paysage (1 024 × 768) et au petit portable (1 280 × 800) : chaque cadre atteignable et
  // VISIBLE à sa taille, la toile assez grande, rien qui déborde, aucun nom coupé, la fenêtre de calcul et le sélecteur
  // dans l'écran, le zoom qui grossit le dessin, la légende de chaque vue lisible, le cartouche sans chevauchement.
  // Seuils de la toile (la place offerte au dessin, mesurée à l'écran) : 300 px de large au téléphone (la largeur moins
  // les marges de la feuille : en dessous, un dessin de 1 m de dalle s'écrit en 6 px), 500 × 300 en tablette portrait
  // (la toile y avait 62 px), 600 × 400 en tablette paysage 1 180 (450 px dans la disposition du PC), 500 × 400 à
  // 1 024 × 768, 400 × 400 au petit portable 1 280 × 800 (la disposition du PC, les deux cadres ouverts autour).
  //   filtre « formats » : tous les modules ; « formats:dalle » : celui-là. FORMATS=tel,tablettes,communs,numpy,accueil
  //   (variable d'environnement) n'en joue qu'une partie — pour prouver une correction par mutation sans tout rejouer.
  const PARTIES = (process.env.FORMATS || 'accueil,tel,tablettes,communs,numpy').split(',');
  if (!FILTRE || FILTRE === 'formats' || FILTRE.startsWith('formats:')) {
    await p.goto(`${BASE}/`);
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const choisis = FILTRE.startsWith('formats:') ? [FILTRE.slice(8)] : Object.keys(MODS);
    // le travailleur, vu de la page : une réponse peut être SIMULÉE (window.__simuler) ou RETOUCHÉE (window.__retoucher),
    // pour les gestes dont le moteur n'a pas encore la réponse (télécharger) ou qu'aucun module ne produit (zéro vue)
    const proxyTravailleur = () => {
      const W = window.Worker;
      window.Worker = class extends W {
        set onmessage(f) {
          this.__f = f;
          super.onmessage = (ev) => (window.__retoucher && ev.data && ev.data.type === 'reponse'
            ? f({ data: { ...ev.data, json: window.__retoucher(ev.data.json) } }) : f(ev));
        }
        get onmessage() { return this.__f; }
        postMessage(m) {
          const rep = window.__simuler && m && typeof m.requete === 'string' ? window.__simuler(JSON.parse(m.requete)) : null;
          if (rep) { setTimeout(() => this.__f({ data: { type: 'reponse', id: m.id, json: JSON.stringify(rep) } }), 30); return; }
          super.postMessage(m);
        }
      };
    };
    const contexte = async (w, h, tel) => {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel, acceptDownloads: true });
      await c.addInitScript(proxyTravailleur);
      const pg = await c.newPage();
      pg.erreurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') pg.erreurs.push(m.text().slice(0, 300)); });
      pg.requetes = [];
      c.on('request', (r) => pg.requetes.push(r.url()));
      return pg;
    };
    const vis = (pg, s) => pg.evaluate((s) => { const e = document.querySelector(s); return !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true }); }, s);
    const boite = (pg, s) => pg.evaluate((s) => {
      const e = [...document.querySelectorAll(s)].find((x) => x.checkVisibility({ visibilityProperty: true, opacityProperty: true }));
      if (!e) return null;
      const r = e.getBoundingClientRect();
      return { g: Math.round(r.left), h: Math.round(r.top), l: Math.round(r.width), t: Math.round(r.height), d: Math.round(r.right), b: Math.round(r.bottom), W: innerWidth, H: innerHeight };
    }, s);
    const dansEcran = (r) => r && r.g >= -1 && r.h >= -1 && r.d <= r.W + 1 && r.b <= r.H + 1;
    // ce que l'œil voit, comme mesure() : débordement, noms coupés ; et le contenu de la vue ouverte
    const mesurer = (pg) => pg.evaluate(() => {
      const vis = (e) => e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const large = (e) => { const r = document.createRange(); r.selectNodeContents(e); return r.getBoundingClientRect().width; };
      const coupes = [...document.querySelectorAll('.prof > span:not(.leger), .el .l1 b, .row > label > span:not(.sym), .ni .t, .pk-row .nm, .tile .nm, .rows .seg.full button, .mtabs button, .mbar .btn, .mmenu .mi')]
        .filter(vis).filter((e) => e.scrollWidth > e.clientWidth + 1 || large(e) > e.getBoundingClientRect().width + 0.01).map((e) => e.textContent.trim().slice(0, 40));
      // la grande pastille du verdict : son texte tient dans sa boîte (une pastille comprimée coupait « η »)
      const xl = [...document.querySelectorAll('.verdict .chip.xl')].filter(vis).filter((e) => large(e) > e.getBoundingClientRect().width + 0.5).map((e) => e.textContent);
      return { deborde: document.documentElement.scrollWidth > innerWidth, coupes, xl };
    });
    // la place offerte au dessin (la zone .draw de la vue ouverte) et le dessin lui-même (son viewBox projeté à l'écran)
    const toile = (pg) => pg.evaluate(() => {
      const d = [...document.querySelectorAll('.view.on .draw')].find((e) => e.checkVisibility());
      if (!d) return null;
      const r = d.getBoundingClientRect();
      const svg = d.querySelector(':scope > svg');
      let dessin = null, texte = null;
      if (svg && svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width) {
        const m = svg.getScreenCTM(), vb = svg.viewBox.baseVal;
        dessin = [Math.round(vb.width * m.a), Math.round(vb.height * m.d)];
        const t = [...svg.querySelectorAll('text')].map((x) => x.getBoundingClientRect().height).filter((x) => x > 0).sort((a, b) => a - b);
        texte = t.length ? Math.round(t[Math.floor(t.length / 2)] * 10) / 10 : null;
      }
      return { l: Math.round(r.width), t: Math.round(r.height), dessin, texte, svg: !!svg };
    });
    // chaque vue : sa légende visible et entière (ni coupée, ni cachée) ; une vue qui est un tableau MONTRE ses cases
    // (comptées à travers chaque parent qui les rogne : un tableau présent dans la page mais dans une zone de 0 px de
    // haut n'est pas vu)
    const legendes = (pg) => pg.evaluate(async () => {
      const out = [];
      const vu = (e, borne) => {
        let r = e.getBoundingClientRect();
        if (r.width < 1 || r.height < 1 || !e.checkVisibility({ visibilityProperty: true, opacityProperty: true })) return false;
        for (let a = e.parentElement; a && a !== borne.parentElement; a = a.parentElement) {
          if (getComputedStyle(a).overflow === 'visible') continue;
          const p = a.getBoundingClientRect();
          r = { left: Math.max(r.left, p.left), top: Math.max(r.top, p.top), right: Math.min(r.right, p.right), bottom: Math.min(r.bottom, p.bottom) };
          if (r.right - r.left < 1 || r.bottom - r.top < 1) return false;
        }
        return r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
      };
      for (const b of [...document.querySelectorAll('#vues [data-vue]')]) {
        b.click();
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        const l = document.querySelector('.view.on .stitle .leg');
        if (l) {
          const ok = l.checkVisibility({ visibilityProperty: true, opacityProperty: true }) && l.getBoundingClientRect().width > 20
            && l.scrollWidth <= l.clientWidth + 1 && l.scrollHeight <= l.clientHeight + 1;
          if (!ok) out.push(`${b.dataset.vue} : « ${l.textContent.slice(0, 40)} »`);
        }
        const d = [...document.querySelectorAll('.view.on .draw')].find((e) => e.checkVisibility());
        if (d && !d.querySelector(':scope > svg')) {
          // un tableau : ses CASES à l'écran (sa légende et son pied sont dans le même bloc, ils ne prouvent rien) ;
          // un bloc sans tableau (le message d'une note vide, un texte) : ses textes à l'écran
          // un bloc qui COMMENCE par du texte (une méthode rédigée, puis des tableaux plus bas) se juge comme de la prose :
          // son premier tableau peut être sous le pli et se lire en défilant (vue « méthode » du plancher)
          const t0 = document.createTreeWalker(d, NodeFilter.SHOW_TEXT);
          let premier = t0.nextNode();
          while (premier && !premier.data.trim()) premier = t0.nextNode();
          const tableau = !!d.querySelector('td') && !!premier && !!premier.parentElement.closest('table');
          const w = document.createTreeWalker(tableau ? d.querySelector('table') : d, NodeFilter.SHOW_TEXT);
          let total = 0, lus = 0;
          for (let n = w.nextNode(); n; n = w.nextNode()) {
            if (tableau && !n.parentElement.closest('td')) continue;
            if (!n.data.trim()) continue;
            total++;
            const rg = document.createRange(); rg.selectNodeContents(n);
            const r = rg.getBoundingClientRect();
            const boite = { getBoundingClientRect: () => r, checkVisibility: () => n.parentElement.checkVisibility({ visibilityProperty: true, opacityProperty: true }), parentElement: n.parentElement };
            if (vu(boite, d)) lus++;
          }
          // six cases d'un tableau ; trois textes d'un bloc de prose (un long texte se lit en défilant)
          if (lus < Math.min(tableau ? 6 : 3, total) || !total) out.push(`${b.dataset.vue} : contenu invisible (${lus} texte(s) vu(s) sur ${total})`);
        }
      }
      document.querySelector('#vues [data-vue]')?.click();
      return out;
    });
    // la barre de la toile : les vues et le zoom ne passent sous aucun cadre (le point touché est bien dans la barre)
    const barreToile = (pg) => pg.evaluate(() => {
      const f = [];
      for (const e of [document.getElementById('vues'), ...document.querySelectorAll('#zoom button')]) {
        if (!e.checkVisibility({ visibilityProperty: true, opacityProperty: true })) continue;
        const r = e.getBoundingClientRect();
        for (const x of [r.left + 4, (r.left + r.right) / 2, r.right - 4]) {
          const t = document.elementFromPoint(x, (r.top + r.bottom) / 2);
          if (!t || !t.closest('.ctool')) f.push(`${e.id || e.dataset.zoom || e.className} couvert par ${t ? (t.className || t.tagName) : 'rien'}`);
        }
      }
      return [...new Set(f)];
    });
    // le cartouche : ses cases ne se chevauchent pas, l'indice et la date se lisent en entier
    const cartouche = (pg) => pg.evaluate(() => {
      const c = [...document.querySelectorAll('.view.on .cart')].find((e) => e.checkVisibility());
      if (!c) return ['pas de cartouche visible'];
      const cases = [...c.children].filter((e) => e.checkVisibility({ visibilityProperty: true }));
      const fautes = [];
      cases.forEach((e, i) => {
        const r = e.getBoundingClientRect();
        if (i && r.left < cases[i - 1].getBoundingClientRect().right - 0.5) fautes.push(`${e.textContent.trim()} chevauche`);
        if (r.right > c.getBoundingClientRect().right + 0.5) fautes.push(`${e.textContent.trim()} sort du cartouche`);
        for (const x of e.children) if ((e.classList.contains('dt') || e.classList.contains('ind') || x.tagName === 'SPAN') && x.scrollWidth > x.clientWidth + 1) fautes.push(`« ${x.textContent} » coupé`);
      });
      // le nom de la note garde une place lisible (un chantier au nom long le réduisait à « P… »)
      const n0 = cases[0] && cases[0].querySelector('b');
      if (n0 && n0.getBoundingClientRect().width < 60) fautes.push(`la note réduite à ${Math.round(n0.getBoundingClientRect().width)} px`);
      return fautes;
    });
    // le zoom : « + » grossit le dessin (le texte du dessin, lu à l'écran), « ajuster » le remet à sa feuille
    const zoom = async (pg) => {
      // sur la première vue qui est un dessin (un tableau ne se zoome pas : il défile)
      const k = await pg.evaluate(() => { const v = [...document.querySelectorAll('#stage .view')].find((x) => x.querySelector('.draw > svg')); return v ? v.dataset.vue : null; });
      if (!k) return { saute: true };
      await pg.click(`#vues [data-vue="${k}"]`);
      const lire = () => pg.evaluate(() => { const t = [...document.querySelectorAll('.view.on .draw > svg text')].find((x) => x.getBoundingClientRect().height > 0); return t ? t.getBoundingClientRect().height : document.querySelector('.view.on .draw > svg').getBoundingClientRect().width; });
      const a = await lire();
      await pg.click('#zoom [data-zoom="plus"]'); await pg.click('#zoom [data-zoom="plus"]');
      await pg.waitForTimeout(100);
      const z = await lire();
      const pct = await pg.textContent('#zoom .zpct');
      await pg.click('#zoom [data-zoom="ajuster"]');
      await pg.waitForTimeout(100);
      const r = await lire();
      await pg.click('#vues [data-vue]');
      return { a, z, r, pct, ok: z >= a * 1.4 && Math.abs(r - a) < 0.5 };
    };
    // la fenêtre de calcul et le sélecteur s'ouvrent et tiennent dans l'écran
    const volets = async (pg, cle, fmt) => {
      const c = await pg.$('#resultat [data-calcul]');
      if (c) {
        await c.click();
        await pg.waitForTimeout(250);
        const r = await boite(pg, '.flyout.on.calcul');
        verif(`${cle} · ${fmt} : la fenêtre de calcul s'ouvre et tient dans l'écran`, dansEcran(r) && r.l > 280 && r.t > 200, JSON.stringify(r));
        const m = await mesurer(pg);
        verif(`${cle} · ${fmt} : la fenêtre de calcul ne déborde pas`, !m.deborde, JSON.stringify(m));
        if (fmt === 'téléphone') await pg.screenshot({ path: path.join(OUT, `site_${cle}_tel_calcul.png`) });
        await pg.click('.flyout.on [data-fermer]');
        await pg.waitForTimeout(150);
        verif(`${cle} · ${fmt} : la fenêtre de calcul se referme`, !(await vis(pg, '.flyout.on')));
      }
      const pk = await pg.$('#resultat .prof[data-pk]');
      if (pk) {
        await pk.click();
        await pg.waitForSelector('.flyout.on.picker', { timeout: 60000 });
        await pg.waitForTimeout(200);
        const r = await boite(pg, '.flyout.on.picker');
        const lignes = await pg.$$eval('.flyout.on.picker .pk-row', (x) => x.filter((e) => e.checkVisibility()).length);
        verif(`${cle} · ${fmt} : le sélecteur s'ouvre et tient dans l'écran`, dansEcran(r) && (lignes > 0 || await vis(pg, '.flyout.on .pk-vide')), JSON.stringify({ ...r, lignes }));
        if (fmt === 'téléphone') await pg.screenshot({ path: path.join(OUT, `site_${cle}_tel_selecteur.png`) });
        await pg.click('.flyout.on [data-fermer]');
        await pg.waitForTimeout(150);
      }
    };
    // une note neuve, par l'accueil (la fenêtre « Nouvelle note » à ce format)
    const neuve = async (pg, cle, fmt) => {
      await pg.goto(`${BASE}/index.html`);
      await pg.click('#ouvrir-nouvelle');
      const mc = await boite(pg, '#nouvelle .mcard');
      verif(`${cle} · ${fmt} : « Nouvelle note » s'ouvre dans l'écran`, dansEcran(mc), JSON.stringify(mc));
      // un chantier au nom long : le fil d'Ariane, l'en-tête du téléphone et le cartouche doivent le tenir (c'est lui qui,
      // en écrasant la case « Note », faisait chevaucher « Note » et « Chantier »)
      await pg.fill('#nn-chantier', 'Résidence Les Tilleuls, bâtiment B — phase 2 (logements)'); await pg.fill('#nn-nom', `${MODS[cle].nom} · ${fmt}`);
      await pg.click(`.mopt[data-module="${cle}"]`);
      await Promise.all([pg.waitForNavigation(), pg.click('#nn-creer')]);
      await pg.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
      const ko = await pg.evaluate(() => document.getElementById('demarrage').classList.contains('erreur') ? document.getElementById('demarrage-etape').textContent : '');
      verif(`${cle} · ${fmt} : le moteur démarre`, !ko, ko);
      return !ko;
    };

    // « Reprendre » compacte : six notes semées dans ce navigateur, puis le bloc MESURÉ à l'écran. Les grandes cartes de la
    // maquette (vignette de 150 px au-dessus du nom) prenaient ~460 px au PC pour six notes ; le bureau les a voulues
    // « deux fois plus petites et plus élégantes ».
    const reprendre = async (pg) => {
      await pg.evaluate((mods) => {
        const notes = {}, t = Date.now();
        const svg = '<svg viewBox="0 0 100 60" xmlns="http://www.w3.org/2000/svg"><rect x="6" y="22" width="88" height="16" fill="none" stroke="#7a8699"/></svg>';
        mods.slice(0, 6).forEach((m, i) => {
          const id = `reprendre-${i}`;
          notes[id] = { id, module: m, nom: `Note ${i + 1} — un nom assez long pour devoir être coupé`, chantier: 'Résidence Les Tilleuls, bâtiment B', dossier: 'Phase 2',
            indice: '', cree: t - i * 6e4, modifie: t - i * 6e4, etat: null, resume: { taux: 63.7 + 7 * i, niveau: 'ok' }, vignette: svg };
        });
        localStorage.setItem('stab.notes.v1', JSON.stringify(notes));
      }, Object.keys(MODS));
      await pg.reload();
      await pg.waitForTimeout(300);
      return pg.evaluate(() => {
        const vu = (e) => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
        const cs = [...document.querySelectorAll('#recentes .rcard')];
        const bloc = document.getElementById('recentes').getBoundingClientRect();
        const r = (e) => e.getBoundingClientRect();
        return {
          n: cs.length,
          h: Math.max(...cs.map((c) => Math.round(r(c).height))),
          bloc: Math.round(bloc.height),
          // les cartes ENTIÈREMENT visibles, sur combien de lignes et de colonnes
          lignes: new Set(cs.filter((c) => r(c).right <= innerWidth + 0.5).map((c) => Math.round(r(c).top))).size,
          colonnes: new Set(cs.filter((c) => r(c).right <= innerWidth + 0.5).map((c) => Math.round(r(c).left))).size,
          // la vignette à gauche, moins de la moitié de la carte ; le nom, la pastille et le ✕ dans la carte
          gauche: cs.every((c) => { const t = r(c.querySelector('.thumb')), k = r(c); return t.left - k.left < 2 && t.width < k.width / 2 && vu(c.querySelector('.thumb svg')); }),
          dedans: cs.every((c) => ['.t', '.chip', '.supp'].every((s) => { const e = c.querySelector(s), a = r(e), k = r(c); return vu(e) && a.right <= k.right + 0.5 && a.bottom <= k.bottom + 0.5; })),
          deborde: document.documentElement.scrollWidth > innerWidth,
        };
      });
    };
    if (PARTIES.includes('accueil')) {
      for (const [fmt, w, h, tel, hmax, cols, lignes, blocmax] of [['PC', 1440, 900, false, 76, 3, 2, 165], ['tablette', 820, 1180, false, 76, 2, 3, 250],
        ['téléphone', 390, 844, true, 66, 1, 2, 150]]) {
        const g = await contexte(w, h, tel);
        await g.goto(`${BASE}/index.html`);
        const q = await reprendre(g);
        verif(`accueil · ${fmt} : « Reprendre » compacte — des cartes basses (≤ ${hmax} px), la vignette à gauche, ${cols} par ligne, le bloc ≤ ${blocmax} px`,
          q.n === 6 && q.h <= hmax && q.colonnes === cols && q.lignes === lignes && q.bloc <= blocmax && q.gauche && q.dedans && !q.deborde, JSON.stringify(q));
        await g.screenshot({ path: path.join(OUT, `site_accueil_reprendre_${fmt === 'téléphone' ? 'tel' : fmt.toLowerCase()}.png`) });
        verif(`accueil · ${fmt} : « Reprendre » sans erreur JavaScript`, !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
        await g.context().close();
      }
    }

    // ------------------------------------------------ l'accueil au téléphone
    if (PARTIES.includes('accueil')) {
      const t = await contexte(390, 844, true);
      await t.goto(`${BASE}/index.html`);
      await t.waitForTimeout(300);
      const acc = { mhead: await boite(t, '.mhead'), mnav: await boite(t, '.mnav'), bouton: await boite(t, '#ouvrir-nouvelle'), rail: await vis(t, '.nav') };
      const m = await mesurer(t);
      verif('accueil · téléphone : l\'en-tête en haut, les cinq entrées en bas, « Nouvelle note » à la taille du doigt, rien ne déborde',
        acc.mhead && acc.mhead.h === 0 && acc.mnav && acc.mnav.b === acc.mnav.H && (await t.$$eval('.mnav a', (x) => x.length)) === 5 && acc.bouton && acc.bouton.t >= 44
        && !acc.rail && !m.deborde && !m.coupes.length, JSON.stringify({ ...acc, ...m }));
      await t.screenshot({ path: path.join(OUT, 'site_accueil_tel.png') });
      await t.click('.mnav [data-modules]');
      await t.waitForTimeout(200);
      const mp = await boite(t, '.modpanel');
      verif('accueil · téléphone : « Modules » ouvre le volet des modules dans l\'écran', dansEcran(mp) && mp.g === 0, JSON.stringify(mp));
      await t.context().close();
    }

    for (const cle of choisis.filter((k) => MODS[k])) {
      // ------------------------------------------------ le téléphone
      const t = await contexte(390, 844, true);
      if (PARTIES.includes('tel') && await neuve(t, cle, 'téléphone')) {
        const fixe = async () => ({ mtop: await boite(t, '.mtop'), mtabs: await boite(t, '.mtabs'), mbar: await boite(t, '.mbar') });
        for (const [onglet, cadre, autres] of [['donnees', '.panel.left', ['.canvas', '.panel.right']], ['dessin', '.canvas', ['.panel.left', '.panel.right']],
          ['resultat', '.panel.right', ['.canvas', '.panel.left']]]) {
         try {
          await t.click(`.mtabs [data-onglet="${onglet}"]`);
          await t.waitForTimeout(250);
          const r = await boite(t, cadre);
          const caches = [];
          for (const a of autres) if (await vis(t, a)) caches.push(a);
          verif(`${cle} · téléphone : l'onglet ${onglet} montre son cadre, sur toute la largeur, et lui seul`, r && r.l >= 370 && r.t >= 200 && !caches.length,
            JSON.stringify({ r, caches }));
          const m = await mesurer(t);
          verif(`${cle} · téléphone · ${onglet} : rien ne déborde, aucun nom coupé`, !m.deborde && !m.coupes.length && !m.xl.length, JSON.stringify(m));
          await t.screenshot({ path: path.join(OUT, `site_${cle}_tel_${onglet}.png`) });
          // l'en-tête et les onglets restent en haut, la barre en bas, même la page défilée
          await t.evaluate(() => scrollTo(0, 700));
          await t.waitForTimeout(120);
          const f = await fixe();
          verif(`${cle} · téléphone · ${onglet} : l'en-tête et les onglets restent en haut, la barre en bas (page défilée)`,
            f.mtop && f.mtop.h === 0 && f.mtabs && f.mtabs.h >= f.mtop.b - 1 && f.mtabs.h <= f.mtop.b + 1 && f.mbar && f.mbar.b === f.mbar.H, JSON.stringify(f));
          await t.evaluate(() => scrollTo(0, 0));
          if (onglet === 'donnees') {
            // des champs en 16 px au moins (sinon iOS agrandit la page au toucher), des cibles de 36 px au moins
            const petits = await t.$$eval('#donnees input:not([type=checkbox]):not([type=file]), #donnees select, #donnees textarea',
              (x) => x.filter((e) => e.checkVisibility()).filter((e) => parseFloat(getComputedStyle(e).fontSize) < 16).map((e) => `${e.dataset.chemin} ${getComputedStyle(e).fontSize}`));
            verif(`${cle} · téléphone : chaque champ s'écrit en 16 px au moins`, !petits.length, petits.slice(0, 4).join(', '));
          }
          const cibles = await t.evaluate(() => {
            const s = '.mtop button, .mtop a.btn, .mtabs button, .mbar .btn, #vues button, .zoom button, .sechd, .ni, .nl-add, #donnees .seg button, #donnees .field, .case, .addrow, .moins, .elements .ni.on .x, #resultat .el, #resultat .prof[data-pk]';
            const tous = [...document.querySelectorAll(s)].filter((e) => e.checkVisibility({ visibilityProperty: true, opacityProperty: true }));
            return tous.filter((e) => { const r = e.getBoundingClientRect(); return Math.min(r.width, r.height) < 35.5; })
              .map((e) => `${e.className || e.tagName} ${Math.round(e.getBoundingClientRect().width)}×${Math.round(e.getBoundingClientRect().height)}`);
          });
          verif(`${cle} · téléphone · ${onglet} : les cibles des briques communes font 36 px au moins`, !cibles.length, cibles.slice(0, 4).join(', '));
          if (onglet === 'dessin') {
            const tl = await toile(t);
            verif(`${cle} · téléphone : la toile offre 300 px de large au moins au dessin`, tl && tl.l >= 300, JSON.stringify(tl));
            const lg = await legendes(t);
            verif(`${cle} · téléphone : la légende de chaque vue se lit en entier, le contenu de chaque tableau se voit`, !lg.length, lg.join(' | '));
            const bt = await barreToile(t);
            verif(`${cle} · téléphone : la barre de la toile (vues, zoom) n'est couverte par rien`, !bt.length, bt.join(', '));
            const z = await zoom(t);
            if (!z.saute) verif(`${cle} · téléphone : « + » grossit le dessin, « ajuster » le remet`, z.ok, JSON.stringify(z));
            // le plein écran : la vue couvre l'écran
            await t.click('#zoom [data-plein]');
            await t.waitForTimeout(250);
            const pe = await boite(t, '.canvas');
            const fe = await boite(t, '.view.on .sheet');
            verif(`${cle} · téléphone : le plein écran donne l'écran entier à la vue`, pe && pe.g === 0 && pe.h === 0 && pe.l === pe.W && pe.t === pe.H && fe && fe.t > pe.H * 0.6,
              JSON.stringify({ pe, fe }));
            await t.screenshot({ path: path.join(OUT, `site_${cle}_tel_plein.png`) });
            await t.click('#zoom [data-plein]');
            await t.waitForTimeout(150);
            verif(`${cle} · téléphone : le plein écran se quitte`, !(await t.evaluate(() => document.querySelector('.bench').classList.contains('plein'))));
          }
          if (onglet === 'resultat') await volets(t, cle, 'téléphone');
         } catch (e) { verif(`${cle} · téléphone · ${onglet} : le parcours va au bout`, false, String(e.message).split('\n')[0]); await t.keyboard.press('Escape'); }
        }
        // le menu des actions : il s'ouvre, tient dans l'écran, et se referme
        await t.click('#menu-m');
        const mm = await boite(t, '#mmenu');
        verif(`${cle} · téléphone : le menu des actions s'ouvre dans l'écran`, dansEcran(mm) && (await t.$$eval('#mmenu .mi', (x) => x.filter((e) => e.checkVisibility()).length)) >= 6, JSON.stringify(mm));
        await t.click('.mtabs [data-onglet="donnees"]');
        verif(`${cle} · téléphone : un geste ailleurs referme le menu`, !(await vis(t, '#mmenu')));
        verif(`${cle} · téléphone : aucune erreur JavaScript`, !t.erreurs.length, t.erreurs.join(' | ').slice(0, 300));
      }
      await t.context().close();

      // ------------------------------------------------ les tablettes (une seule note, la fenêtre change de taille)
      const g = await contexte(820, 1180, false);
      if (PARTIES.includes('tablettes') && await neuve(g, cle, 'tablette portrait')) {
        for (const [fmt, w, h, seuil, nom] of [['tablette portrait', 820, 1180, [500, 300], 'tab_portrait'], ['tablette paysage', 1180, 820, [600, 400], 'tab_paysage'],
          ['petite tablette paysage', 1024, 768, [500, 400], 'tab_paysage_1024'], ['petit portable', 1280, 800, [400, 400], 'portable_1280']]) {
         try {
          await g.setViewportSize({ width: w, height: h });
          await g.waitForTimeout(400);
          const cadres = { donnees: await boite(g, '.panel.left'), resultat: await boite(g, '.panel.right'), toile: await boite(g, '.canvas') };
          verif(`${cle} · ${fmt} : les données, le résultat et la toile sont visibles ensemble, chacun à sa taille`,
            Object.values(cadres).every((r) => r && r.l >= 280 && r.t >= 240 && dansEcran(r)), JSON.stringify(cadres));
          const tl = await toile(g);
          verif(`${cle} · ${fmt} : la toile offre ${seuil[0]} × ${seuil[1]} px au moins au dessin`, tl && tl.l >= seuil[0] && tl.t >= seuil[1], JSON.stringify(tl));
          // la barre du haut : rien ne s'y chevauche (« Brouillon » passait sous le nom de la note)
          const haut = await g.evaluate(() => {
            const els = [...document.querySelectorAll('.top > *, .top .crumbs > *')].filter((e) => !e.classList.contains('crumbs') && !e.classList.contains('spacer')
              && e.checkVisibility({ visibilityProperty: true, opacityProperty: true })).map((e) => ({ n: (e.id || e.className || e.tagName).toString().slice(0, 20), r: e.getBoundingClientRect() }));
            const f = [];
            for (let i = 0; i < els.length; i++) for (let j = i + 1; j < els.length; j++) {
              const a = els[i].r, b = els[j].r;
              if (a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom && b.top < a.bottom) f.push(`${els[i].n} / ${els[j].n}`);
            }
            const top = document.querySelector('.top').getBoundingClientRect();
            els.forEach((e) => { if (e.r.right > top.right + 0.5) f.push(`${e.n} sort de la barre`); });
            // le nom de la note, trop long pour sa place : il finit par « … », jamais coupé au milieu d'une lettre
            const nn = document.getElementById('nom-note');
            if (nn.scrollWidth > nn.clientWidth + 1 && getComputedStyle(nn).textOverflow !== 'ellipsis') f.push('nom de la note coupé net (sans « … »)');
            return f;
          });
          verif(`${cle} · ${fmt} : la barre du haut ne se chevauche pas`, !haut.length, haut.join(', '));
          const ca = await cartouche(g);
          verif(`${cle} · ${fmt} : le cartouche ne se chevauche pas, l'indice et la date se lisent`, !ca.length, ca.join(', '));
          if (nom === 'portable_1280') {
            // le cadre le plus étroit (304 px) : chaque section des données ouverte, chaque champ mesuré
            await g.evaluate(() => document.querySelectorAll('#donnees .sec:not(.open) > .sechd').forEach((h) => h.click()));
            await g.waitForTimeout(200);
          }
          const m = await mesurer(g);
          verif(`${cle} · ${fmt} : rien ne déborde, aucun nom coupé, la pastille du verdict entière`, !m.deborde && !m.coupes.length && !m.xl.length, JSON.stringify(m));
          // une comparaison longue à côté de la grande pastille (soudures, pleine pénétration) : elle ne la comprime pas
          const xl = await g.evaluate(() => {
            const gv = document.querySelector('.verdict .gov'), c = document.querySelector('.verdict .chip.xl');
            if (!gv || !c || !c.checkVisibility()) return null;
            const ajout = document.createTextNode(' · σ⊥ = 218,5 N/mm² ≤ 0,9·fu/γM2 = 259,2 N/mm² (pleine pénétration, métal de base)');
            gv.appendChild(ajout);
            const r = document.createRange(); r.selectNodeContents(c);
            const out = { texte: Math.round(r.getBoundingClientRect().width), boite: Math.round(c.getBoundingClientRect().width) };
            ajout.remove();
            return out;
          });
          if (xl) verif(`${cle} · ${fmt} : une comparaison longue ne comprime pas la grande pastille du verdict`, xl.texte <= xl.boite + 0.5, JSON.stringify(xl));
          const lg = await legendes(g);
          verif(`${cle} · ${fmt} : la légende de chaque vue se lit en entier, le contenu de chaque tableau se voit`, !lg.length, lg.join(' | '));
          const bt = await barreToile(g);
          verif(`${cle} · ${fmt} : la barre de la toile (vues, zoom) n'est couverte par rien`, !bt.length, bt.join(', '));
          await g.screenshot({ path: path.join(OUT, `site_${cle}_${nom}.png`) });
          const z = await zoom(g);
          if (!z.saute) verif(`${cle} · ${fmt} : « + » grossit le dessin, « ajuster » le remet`, z.ok, JSON.stringify(z));
          await volets(g, cle, fmt);
         } catch (e) { verif(`${cle} · ${fmt} : le parcours va au bout`, false, String(e.message).split('\n')[0]); await g.keyboard.press('Escape'); }
        }
        // un nom d'élément long : ce sont ses dernières lettres qui cèdent, jamais les dimensions qui le suivent
        await g.setViewportSize({ width: 820, height: 1180 });
        await g.waitForTimeout(300);
        // le champ du nom de l'élément : celui dont la valeur ouvre l'intitulé de l'élément ouvert de la liste
        const champNom = await g.evaluate(() => {
          const t = document.querySelector('.ni.on .t');
          if (!t || ![...t.children].some((e) => e.tagName === 'SPAN' && !e.classList.contains('nom'))) return null;
          const intitule = t.textContent.trim();
          const c = [...document.querySelectorAll('#donnees input[data-sorte="texte"]')].filter((e) => e.value.trim() && intitule.startsWith(e.value.trim()))
            .sort((a, b) => b.value.length - a.value.length)[0];
          return c ? c.dataset.chemin : null;
        });
        if (champNom) {
          await g.fill(`#donnees input[data-chemin="${champNom}"]`, 'Poutre de rive très longue du bâtiment B, file 12 à 18, niveau +3');
          await g.press(`#donnees input[data-chemin="${champNom}"]`, 'Enter');
          await g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
          await g.waitForTimeout(150);
          // les dimensions (le dernier <span> de l'intitulé) entières et dans leur ligne ; le nom, lui, peut céder
          const ni = await g.evaluate(() => {
            const t = document.querySelector('.ni.on .t');
            const d = [...t.children].filter((e) => e.tagName === 'SPAN' && !e.classList.contains('nom')).pop();
            const rt = t.getBoundingClientRect(), rd = d.getBoundingClientRect();
            return { dims: d.textContent, entieres: d.scrollWidth <= d.clientWidth + 1 && rd.right <= rt.right + 0.5 && rd.left >= rt.left - 0.5, largeur: Math.round(rd.width) };
          });
          verif(`${cle} · tablette portrait : un nom long se coupe, les dimensions restent entières`, ni.entieres && ni.largeur > 20, JSON.stringify(ni));
        }
        verif(`${cle} · tablettes : aucune erreur JavaScript`, !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
      }
      await g.context().close();
    }

    // ------------------------------------------------ les gestes communs, une fois (tablette portrait)
    if (PARTIES.includes('communs') && (choisis.includes('soudures') || choisis.includes('poutre'))) {
      const g = await contexte(820, 1180, false);
      // une feuille qui met les en-têtes de tableau en capitales (la charte le fait ; soudures.css le défait pour les
      // siens) : posée AVANT l'affichage, comme une feuille de module
      await g.context().addInitScript(() => document.addEventListener('DOMContentLoaded', () => {
        const st = document.createElement('style');
        st.textContent = '.vtab th { text-transform: uppercase !important; }';
        document.head.appendChild(st);
      }));
      const cle = choisis.includes('soudures') ? 'soudures' : 'poutre';
      if (await neuve(g, cle, 'gestes communs')) {
        // les lettres grecques d'un en-tête en capitales gardent leur casse : l'œil (innerText applique la casse) doit
        // lire η, σ, τ là où le texte porte η, σ, τ
        const grec = await g.evaluate(async () => {
          const out = { vus: 0, fautes: [] };
          for (const b of [...document.querySelectorAll('#vues [data-vue]')]) {
            b.click();
            await new Promise((r) => requestAnimationFrame(r));
            for (const th of document.querySelectorAll('.view.on .vtab th')) {
              if (!/[Ͱ-Ͽ]/.test(th.textContent)) continue;
              out.vus++;
              const lu = th.innerText;
              if (/[ΗΣΤΓ]/.test(lu) && !/[ΗΣΤΓ]/.test(th.textContent)) out.fautes.push(`${th.textContent} → ${lu}`);
            }
          }
          document.querySelector('#vues [data-vue]').click();
          return out;
        });
        if (cle === 'soudures') verif('en-têtes en capitales : η reste η, σ reste σ (jamais « H », « Σ »)', grec.vus > 0 && !grec.fautes.length, JSON.stringify(grec));
        // l'état VERROUILLÉ des briques (C.segments, C.texte, C.case avec desactive=True) se VOIT : les fragments rendus
        // par les briques elles-mêmes (Python), posés dans le cadre Données à côté des mêmes briques actives, et comparés
        // à l'œil (couleurs calculées) — un segment désactivé ressemblait trait pour trait à un segment actif
        const BRIQUES = [
          'import json, sys',
          'sys.path.insert(0, sys.argv[1])',
          'from api_web import commun as C',
          'o = [("a", "Rive"), ("b", "Continue"), ("c", "Encastrée")]',
          'print(json.dumps({k: C.segments(["essai", "s"], "a", o, desactive=v) + C.texte(["essai", "t"], "D1", desactive=v)',
          '                  + C.case(["essai", "c"], True, "Appui libre", desactive=v) for k, v in (("actif", False), ("verrou", True))}))',
        ].join('\n');
        const fr = require('child_process').spawnSync(process.env.PYTHON || 'python3', ['-c', BRIQUES, path.join(__dirname, '..', '..')], { encoding: 'utf8' });
        let briques = null;
        try { briques = JSON.parse(fr.stdout); } catch (e) { /* le contrôle rougit */ }
        const verrou = briques && await g.evaluate((b) => {
          const z = document.querySelector('#donnees');
          z.insertAdjacentHTML('afterbegin', `<div class="rows" id="essai-actif">${b.actif}</div><div class="rows" id="essai-verrou">${b.verrou}</div>`);
          const cs = (s) => getComputedStyle(document.querySelector(s));
          const diff = (s, p) => cs(`#essai-actif ${s}`)[p] !== cs(`#essai-verrou ${s}`)[p];
          const out = {
            segment: diff('.seg button:not(.on)', 'color'), retenu: diff('.seg button.on', 'backgroundColor') || diff('.seg button.on', 'color'),
            bande: diff('.seg', 'backgroundColor'), curseur: cs('#essai-verrou .seg button').cursor, texte: diff('.field', 'backgroundColor'),
            case: diff('.case', 'color'), lisible: cs('#essai-verrou .seg button.on').color !== cs('#essai-verrou .seg').backgroundColor,
          };
          document.getElementById('essai-actif').remove(); document.getElementById('essai-verrou').remove();
          return out;
        }, briques);
        verif('l\'état verrouillé des briques se voit (segments, texte, case), le choix retenu reste lisible',
          verrou && verrou.segment && verrou.retenu && verrou.bande && verrou.curseur === 'not-allowed' && verrou.texte && verrou.case && verrou.lisible,
          JSON.stringify(verrou || String(fr.stderr).slice(0, 200)));
        // télécharger un fichier propre au module : le moteur rend { fichier: { nom, type, texte } } ; ici la réponse
        // est simulée (le routeur de l'autre équipe la rendra) — le nom arrive piégé et repart nettoyé
        await g.evaluate(() => {
          document.querySelector('#donnees').insertAdjacentHTML('afterbegin', '<button type="button" class="btn sm" data-telecharger="essai">Télécharger</button>');
          window.__vu = null;
          window.__simuler = (q) => {
            if (q.action !== 'telecharger') return null;
            window.__vu = q;
            return q.quoi === 'essai' ? { ok: true, fichier: { nom: '../../dossier/essai..json', type: 'application/json', texte: '{"écran": "Streamlit", "a": 1}' } }
              : { ok: false, erreur: 'pas de fichier pour cette note' };
          };
        });
        const [dl] = await Promise.all([g.waitForEvent('download', { timeout: 20000 }), g.click('#donnees [data-telecharger="essai"]')]);
        const contenu = require('fs').readFileSync(await dl.path(), 'utf8');
        const vu = await g.evaluate(() => window.__vu);
        verif('télécharger : la requête part au moteur (action, quoi, état) et le fichier arrive sous un nom nettoyé',
          vu && vu.action === 'telecharger' && vu.quoi === 'essai' && vu.etat && dl.suggestedFilename() === 'essai.json' && contenu === '{"écran": "Streamlit", "a": 1}',
          JSON.stringify({ nom: dl.suggestedFilename(), contenu, action: vu && vu.action, quoi: vu && vu.quoi }));
        await g.evaluate(() => { document.querySelector('[data-telecharger="essai"]').dataset.telecharger = 'refuse'; });
        await g.click('#donnees [data-telecharger="refuse"]');
        await g.waitForTimeout(400);
        const tst = await g.evaluate(() => { const t = document.getElementById('toast'); return { vu: t.checkVisibility(), texte: t.textContent, erreur: t.classList.contains('erreur') }; });
        verif('télécharger : un refus du moteur s\'affiche en message', tst.vu && tst.erreur && /refusé.*pas de fichier pour cette note/.test(tst.texte), JSON.stringify(tst));
        // un module qui ne rend aucune vue : un texte neutre (ni « impossible », ni erreur)
        await g.evaluate(() => {
          window.__simuler = null;
          window.__retoucher = (json) => { const r = JSON.parse(json); if (r.ok && r.vues) r.vues = []; return JSON.stringify(r); };
        });
        // « ajouter » recalcule toujours (contrat commun) : sa réponse arrive sans vue
        await g.click('#donnees [data-ajouter]');
        await g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
        await g.waitForTimeout(300);
        const sv = await g.evaluate(() => ({ texte: document.getElementById('stage').textContent.trim(), zoom: document.getElementById('zoom').checkVisibility() }));
        verif('aucune vue : un texte neutre, sans barre de zoom', sv.texte === 'Pas de dessin pour cette note.' && !sv.zoom, JSON.stringify(sv));
        await g.evaluate(() => { window.__retoucher = null; });
        // une réponse illisible du moteur (un « Infinity », que JSON ne connaît pas) : un message, et l'établi reste
        // vivant — la requête restait sans réponse, la file du moteur bloquée, « Calcul… » sans fin
        const nAvant = await g.$$eval('.elements .ni', (e) => e.length);
        await g.evaluate(() => { window.__retoucher = (json) => { window.__retoucher = null; return json.replace(/"ok": true/, '"ok": true, "x": Infinity'); }; });
        await g.click('#donnees [data-ajouter]');
        const libre = await g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 10000 }).then(() => true, () => false);
        const ill = await g.evaluate(() => { const t = document.getElementById('toast'); return { vu: t.checkVisibility(), texte: t.textContent, erreur: t.classList.contains('erreur') }; });
        await g.click('#donnees [data-ajouter]');
        const repart = await g.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 30000 }).then(() => true, () => false);
        const nApres = await g.$$eval('.elements .ni', (e) => e.length);
        verif('une réponse illisible du moteur : un message, et le calcul suivant passe (la file du moteur ne se bloque pas)',
          libre && ill.vu && ill.erreur && /réponse illisible du moteur/.test(ill.texte) && repart && nApres === nAvant + 1, JSON.stringify({ libre, ...ill, repart, nAvant, nApres }));
        // les erreurs de la réponse illisible sont attendues : la console les dit, la page n'a pas levé
        g.erreurs = g.erreurs.filter((x) => !/réponse illisible du moteur/.test(x));
        verif('gestes communs : aucune erreur JavaScript', !g.erreurs.length, g.erreurs.join(' | ').slice(0, 300));
      }
      await g.context().close();
      // le taux d'une note au-delà de 999 % : l'en-tête du téléphone et la carte de l'accueil écrivent « > 999 % » (la
      // valeur exacte au survol), comme les pastilles du moteur ; un taux de huit chiffres en sortait
      const tt = await contexte(390, 844, true);
      if (await neuve(tt, cle, 'taux énorme')) {
        await tt.evaluate(() => { window.__retoucher = (json) => { const r = JSON.parse(json); if (r.ok && r.resume) r.resume.taux = 12345678; return JSON.stringify(r); }; });
        await tt.click('#donnees [data-ajouter]');
        await tt.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
        await tt.evaluate(() => { window.__retoucher = null; });
        const ch = await tt.evaluate(() => {
          const c = document.getElementById('chip-m'), r = c.getBoundingClientRect(), m = document.getElementById('menu-m').getBoundingClientRect();
          return { texte: c.textContent, titre: c.title, vu: c.checkVisibility(), l: Math.round(r.width), entier: c.scrollWidth <= c.clientWidth + 1 && r.right <= m.left + 0.5,
            nom: Math.round(document.getElementById('nom-note-m').getBoundingClientRect().width) };
        });
        verif('taux de plus de 999 % · téléphone : l\'en-tête écrit « > 999 % » (la valeur exacte au survol), la pastille entière, le nom lisible',
          ch.vu && ch.texte === '>\u00a0999\u00a0%' && ch.titre === '12\u202f345\u202f678\u00a0%' && ch.entier && ch.l <= 90 && ch.nom >= 150, JSON.stringify(ch));
        await tt.goto(`${BASE}/index.html`);
        await tt.waitForTimeout(300);
        const ca = await tt.evaluate(() => {
          const c = document.querySelector('#recentes .rcard .chip');
          if (!c) return null;
          const r = c.getBoundingClientRect(), m = c.closest('.m').getBoundingClientRect();
          return { texte: c.textContent, titre: c.title, vu: c.checkVisibility(), dedans: r.right <= m.right + 0.5 && c.scrollWidth <= c.clientWidth + 1 };
        });
        verif('taux de plus de 999 % · accueil : la carte de la note écrit « > 999 % », la valeur exacte au survol, la pastille dans sa carte',
          ca && ca.vu && ca.texte === '>\u00a0999\u00a0%' && ca.titre === '12\u202f345\u202f678\u00a0%' && ca.dedans, JSON.stringify(ca));
        verif('taux de plus de 999 % : aucune erreur JavaScript', !tt.erreurs.length, tt.erreurs.join(' | ').slice(0, 300));
      }
      await tt.context().close();
    }

    // ------------------------------------------------ numpy : chargé pour ceux qui s'en servent, à la demande sinon
    if (PARTIES.includes('numpy')) {
      const man = await (await p.request.get(`${BASE}/moteur/manifeste.json`)).json();
      const sans = Object.keys(MODS).find((k) => !(man.avec_numpy || []).includes(k));
      const avec = (man.avec_numpy || []).find((k) => MODS[k]);
      if (sans) {
        const g = await contexte(1280, 800, false);
        await g.goto(`${BASE}/${MODS[sans].page}`);
        await g.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
        const ok = await g.evaluate(() => document.getElementById('demarrage').hidden);
        verif(`numpy : ${sans} démarre sans le charger`, ok && !g.requetes.some((u) => /numpy-.*\.whl/.test(u)), g.requetes.filter((u) => /numpy/.test(u)).join(', '));
        await g.context().close();
      }
      if (avec) {
        // le manifeste « oublie » ce module : numpy doit être chargé à la demande (le guet du travailleur) et la note
        // calculée quand même
        const g = await contexte(1280, 800, false);
        await g.context().route('**/moteur/manifeste.json', async (route) => {
          const r = await route.fetch();
          const j = await r.json();
          j.avec_numpy = [];
          await route.fulfill({ response: r, json: j });
        });
        await g.goto(`${BASE}/${MODS[avec].page}`);
        await g.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
        const etat = await g.evaluate(() => ({ ok: document.getElementById('demarrage').hidden, resultat: !!document.querySelector('#resultat .verdict, #resultat .el') }));
        verif(`numpy : ${avec} oublié du manifeste se calcule quand même (numpy chargé à la demande)`,
          etat.ok && etat.resultat && g.requetes.some((u) => /numpy-.*\.whl/.test(u)), JSON.stringify({ ...etat, numpy: g.requetes.filter((u) => /numpy/.test(u)) }));
        await g.context().close();
      }
    }
  }

  // ---------------------------------------------------------------- télécharger, de bout en bout
  // Le protocole entier, sans rien simuler : un bouton RENDU PAR UN MODULE (C.bouton_telecharger, en Python) → le clic
  // → le travailleur → le routeur (action « telecharger », contrôle du fichier par refus_fichier) → le Blob → le fichier
  // que Chromium enregistre, sous la politique de sécurité du site. Aucun module publié ne propose encore de fichier :
  // le navigateur reçoit ici un paquet du moteur où le module des poutres en propose (le code AJOUT, ajouté à
  // api_web/poutre.py à la volée ; le manifeste suit l'empreinte du paquet). Rien de ceci n'est publié.
  //   filtre « telecharger »
  if (!FILTRE || FILTRE === 'telecharger') {
    const fs = require('fs');
    const AJOUT = [
      '# --- ajouté à la volée par site/web/verifier_site.js (télécharger, de bout en bout) : jamais publié',
      'import json as _json_essai',
      '_resultat_publie = resultat',
      '',
      '',
      'def resultat(etat):',
      '    r = _resultat_publie(etat)',
      '    b = "".join(C.bouton_telecharger(q, t) for q, t in (("json", "JSON de la note"), ("nom-piege", "Nom piégé"),',
      '                                                         ("type-html", "Page HTML"), ("inconnu", "Inconnu")))',
      '    r["donnees"] = \'<div class="essai-telecharger">\' + b + "</div>" + r["donnees"]',
      '    r["volets"] = r["volets"].replace(\'<div class="ck-sub">\', C.bouton_telecharger("json", "JSON de la note") + \'<div class="ck-sub">\')',
      '    return r',
      '',
      '',
      'def telecharger(etat, quoi):',
      '    if quoi == "json":',
      '        return {"nom": C.nom_fichier("Poutre P1/2 : RDC « essai »", "json"), "type": "application/json",',
      '                "texte": _json_essai.dumps(etat, ensure_ascii=False)}',
      '    if quoi == "nom-piege":',
      '        return {"nom": "../../poutre.json", "type": "application/json", "texte": "{}"}',
      '    if quoi == "type-html":',
      '        return {"nom": "page.html", "type": "text/html", "texte": "<script>window.__piege = 1</script>"}',
      '    return None',
    ].join('\n');
    // le paquet d'essai : celui du site, où api_web/poutre.py reçoit AJOUT (Python le réécrit ; le texte passe en base64)
    const ZIPPER = [
      'import base64, io, sys, zipfile',
      'ajout = base64.b64decode(sys.argv[1])',
      'src = zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read()))',
      'out = io.BytesIO()',
      'with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:',
      '    for i in src.infolist():',
      '        d = src.read(i)',
      '        z.writestr(i, d + b"\\n\\n" + ajout + b"\\n" if i.filename == "api_web/poutre.py" else d)',
      'sys.stdout.buffer.write(out.getvalue())',
    ].join('\n');
    const zip0 = await (await p.request.get(`${BASE}/moteur/moteur.zip`)).body();
    const fait = require('child_process').spawnSync(process.env.PYTHON || 'python3', ['-I', '-c', ZIPPER, Buffer.from(AJOUT).toString('base64')],
      { input: zip0, maxBuffer: 1 << 27 });
    const zipEssai = fait.stdout;
    const shaEssai = require('crypto').createHash('sha256').update(zipEssai || '').digest('hex');
    verif('télécharger : le paquet d\'essai se prépare (le module des poutres y propose des fichiers)', fait.status === 0 && zipEssai.length > zip0.length / 2,
      String(fait.stderr || fait.error || '').slice(0, 300));
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const prepare = async (w, h, tel) => {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel, acceptDownloads: true });
      await c.route('**/moteur/manifeste.json', async (route) => {
        const r = await route.fetch();
        const j = await r.json();
        j.sha256 = shaEssai;
        await route.fulfill({ response: r, json: j });
      });
      await c.route('**/moteur/moteur.zip*', (route) => route.fulfill({ status: 200, contentType: 'application/zip', body: zipEssai }));
      // toute violation de la politique de sécurité est notée (un fichier qui s'exécuterait, une ressource refusée…)
      await c.addInitScript(() => {
        window.__csp = [];
        document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
      });
      const pg = await c.newPage();
      pg.erreurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') pg.erreurs.push(m.text().slice(0, 300)); });
      return pg;
    };
    const ouvrir = async (pg, cle) => {
      const rep = await pg.goto(`${BASE}/${MODS[cle].page}?chantier=${encodeURIComponent('Contrôle')}&nom=${encodeURIComponent('Essai télécharger')}`);
      await pg.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
      return rep;
    };
    const visible = (pg, s) => pg.evaluate((s) => [...document.querySelectorAll(s)].some((e) => e.checkVisibility({ visibilityProperty: true, opacityProperty: true })
      && e.getBoundingClientRect().width > 20 && e.getBoundingClientRect().height > 20), s);
    const message = (pg) => pg.evaluate(() => { const t = document.getElementById('toast'); return { vu: t.checkVisibility(), texte: t.textContent, erreur: t.classList.contains('erreur') }; });
    const etatNote = (pg) => pg.evaluate(() => {
      const id = new URLSearchParams(location.search).get('note');
      return { id, etat: JSON.parse(localStorage.getItem('stab.notes.v1'))[id].etat };
    });
    // un clic qui ne doit RIEN faire télécharger : le refus du routeur arrive en message
    const refuse = async (pg, sel, motif, nom) => {
      const att = pg.waitForEvent('download', { timeout: 3000 }).then((d) => d.suggestedFilename(), () => null);
      await pg.click(sel);
      const fichier = await att;
      const t = await message(pg);
      verif(`télécharger · ${nom} : refusé, aucun fichier enregistré, un message en français`, fichier === null && t.vu && t.erreur && motif.test(t.texte),
        JSON.stringify({ fichier, ...t }));
    };
    const NOM = 'Poutre P1-2 - RDC « essai ».json';

    // ------------------------------------------------ au PC
    const g = await prepare(1440, 900, false);
    const rep = await ouvrir(g, 'poutre');
    verif('télécharger · PC : le bouton rendu par le module (C.bouton_telecharger) est posé et visible', await visible(g, '#donnees [data-telecharger="json"]'));
    const [dl] = await Promise.all([g.waitForEvent('download', { timeout: 30000 }), g.click('#donnees [data-telecharger="json"]')]);
    const contenu = fs.readFileSync(await dl.path(), 'utf8');
    const note = await etatNote(g);
    let relu = null;
    try { relu = JSON.parse(contenu); } catch (e) { /* illisible : le contrôle rougit */ }
    verif('télécharger · PC : le clic → le routeur → le fichier enregistré sous le nom donné par le module, avec l\'état de la note',
      dl.suggestedFilename() === NOM && relu !== null && JSON.stringify(relu) === JSON.stringify(note.etat),
      JSON.stringify({ nom: dl.suggestedFilename(), contenu: contenu.slice(0, 80) }));
    const t0 = await message(g);
    verif('télécharger · PC : un message dit le fichier téléchargé', t0.vu && !t0.erreur && t0.texte === `${NOM} téléchargé`, JSON.stringify(t0));
    await refuse(g, '#donnees [data-telecharger="nom-piege"]', /^Téléchargement refusé : Le fichier « nom-piege » est refusé : le nom du fichier ne peut contenir/, 'un nom de fichier qui remonte les dossiers');
    await refuse(g, '#donnees [data-telecharger="type-html"]', /^Téléchargement refusé : .*type de fichier non permis : « text\/html »/, 'une page HTML');
    await refuse(g, '#donnees [data-telecharger="inconnu"]', /^Téléchargement refusé : Fichier inconnu : « inconnu »/, 'un fichier que le module ne connaît pas');
    // depuis la fenêtre de calcul : le fichier arrive, la fenêtre reste ouverte
    await g.click('#resultat [data-calcul]');
    await g.waitForTimeout(250);
    const [dl2] = await Promise.all([g.waitForEvent('download', { timeout: 30000 }), g.click('.flyout.on.calcul [data-telecharger="json"]')]);
    await g.waitForTimeout(300);
    verif('télécharger · PC : depuis la fenêtre de calcul, le fichier arrive et la fenêtre reste ouverte',
      dl2.suggestedFilename() === NOM && await visible(g, '.flyout.on.calcul'), dl2.suggestedFilename());
    await g.keyboard.press('Escape');
    // la note elle-même (.stab) passe par le même chemin d'enregistrement
    const [dl3] = await Promise.all([g.waitForEvent('download', { timeout: 30000 }), g.click('.top [data-action="exporter"]')]);
    let stab = null;
    try { stab = JSON.parse(fs.readFileSync(await dl3.path(), 'utf8')); } catch (e) { /* illisible : le contrôle rougit */ }
    verif('exporter la note : le fichier « chantier - nom.stab » arrive, la note entière',
      dl3.suggestedFilename() === 'Contrôle - Essai télécharger.stab' && stab && stab.format === 'stab' && stab.note.id === note.id, dl3.suggestedFilename());
    const csp = rep.headers()['content-security-policy'] || '';
    const apres = await g.evaluate(() => ({ csp: window.__csp, piege: window.__piege }));
    verif('télécharger · PC : sous la politique de sécurité du site, aucune violation, rien du fichier exécuté, aucune erreur JavaScript',
      /script-src 'self'/.test(csp) && !apres.csp.length && apres.piege === undefined && !g.erreurs.length,
      JSON.stringify({ csp: csp.slice(0, 40), violations: apres.csp, erreurs: g.erreurs.slice(0, 3) }));
    // un module publié sans fichier à proposer : le VRAI routeur refuse, en français
    await ouvrir(g, 'dalle');
    await g.evaluate(() => document.querySelector('#donnees').insertAdjacentHTML('afterbegin', '<button type="button" class="btn sm" data-telecharger="json">JSON</button>'));
    await refuse(g, '#donnees [data-telecharger="json"]', /^Téléchargement refusé : Ce module ne propose aucun fichier à télécharger\.$/, 'un module sans fichier (dalle)');
    await g.context().close();

    // ------------------------------------------------ au téléphone (le même bouton, dans l'onglet Données)
    const tl = await prepare(390, 844, true);
    await ouvrir(tl, 'poutre');
    verif('télécharger · téléphone : le bouton du module est posé et visible', await visible(tl, '#donnees [data-telecharger="json"]'));
    const [dl4] = await Promise.all([tl.waitForEvent('download', { timeout: 30000 }), tl.click('#donnees [data-telecharger="json"]')]);
    let relu4 = null;
    try { relu4 = JSON.parse(fs.readFileSync(await dl4.path(), 'utf8')); } catch (e) { /* illisible : le contrôle rougit */ }
    verif('télécharger · téléphone : le fichier arrive sous le même nom, avec l\'état de la note',
      dl4.suggestedFilename() === NOM && relu4 && JSON.stringify(relu4) === JSON.stringify((await etatNote(tl)).etat), dl4.suggestedFilename());
    verif('télécharger · téléphone : aucune violation de la politique de sécurité, aucune erreur JavaScript',
      !(await tl.evaluate(() => window.__csp.length)) && !tl.erreurs.length, tl.erreurs.join(' | ').slice(0, 300));
    await tl.context().close();
  }

  // ---------------------------------------------------------------- la météo du lieu (Open-Meteo), de bout en bout
  // Le vrai écran, le vrai moteur, la vraie politique de sécurité ; seul Open-Meteo est remplacé : chaque demande vers ses
  // trois adresses est INTERCEPTÉE (page.route) et reçoit une réponse au format documenté du service (la machine n'a pas de
  // réseau, et un contrôle n'appelle jamais le vrai service). Qu'une demande arrive à l'interception prouve que la politique
  // de sécurité la laisse partir : bloquée par connect-src, elle échouerait avant (« Pas de réseau »), sans être vue ici.
  // Le parcours : la cure « Météo du lieu », le clic, l'indicateur « Récupération de la météo… », les appels (adresses,
  // paramètres), les températures reçues (badge, relevé, vues), la maturité recalculée, la note rechargée qui garde tout et
  // se recalcule SANS AUCUN appel, le détail horaire, une correction à la main ; puis les refus — lieu introuvable, réseau
  // coupé, service en panne, dates hors de portée — en message, la note intacte ; au PC, au téléphone et à la tablette.
  //   filtre « meteo »
  if (!FILTRE || FILTRE === 'meteo') {
    const OPEN_METEO = /^https:\/\/(geocoding-api|archive-api|api)\.open-meteo\.com\//;
    const CORS = { 'access-control-allow-origin': '*' };
    const LIEUX = {
      bruxelles: { id: 2800866, name: 'Bruxelles', latitude: 50.85045, longitude: 4.34878, elevation: 28.0, feature_code: 'PPLC', country_code: 'BE',
        admin1_id: 2800867, timezone: 'Europe/Brussels', population: 1019022, country_id: 2802361, country: 'Belgique', admin1: 'Bruxelles-Capitale' },
    };
    // la température d'un jour (°C, au dixième) et d'une heure : déterministes, inventées
    const tJour = (iso) => { const k = Math.round(Date.parse(iso + 'T00:00:00Z') / 864e5); return Math.round((5.5 + 7 * Math.sin(k / 11) + 0.3 * (k % 5)) * 10) / 10; };
    const tHeure = (iso, h) => Math.round((tJour(iso) + 3.5 * Math.sin((h - 9) * Math.PI / 12)) * 10) / 10;
    const jours = (d0, d1) => { const out = []; for (let t = Date.parse(d0 + 'T00:00:00Z'); t <= Date.parse(d1 + 'T00:00:00Z'); t += 864e5) out.push(new Date(t).toISOString().slice(0, 10)); return out; };
    const decale = (iso, n) => new Date(Date.parse(iso + 'T00:00:00Z') + n * 864e5).toISOString().slice(0, 10);
    const fr = (iso) => iso.split('-').reverse().join('/');
    // le faux Open-Meteo : chaque demande journalisée, la réponse tirée de ses paramètres ; « mode » : coupe, panne, lent
    const brancher = async (c, journal, mode) => {
      await c.route(OPEN_METEO, async (route) => {
        const u = new URL(route.request().url());
        journal.push(route.request().url());
        if (mode.lent) await new Promise((r) => setTimeout(r, mode.lent));
        if (mode.coupe) return route.abort('internetdisconnected');
        if (mode.panne) return route.fulfill({ status: 500, headers: CORS, contentType: 'application/json', body: JSON.stringify({ error: true, reason: 'Panne simulée du service' }) });
        // une réponse ÉNORME (8 Mo) : la page ne la lit pas (relecture adverse : elle était lue toute, puis envoyée au moteur)
        if (mode.enorme) return route.fulfill({ status: 200, headers: CORS, contentType: 'application/json', body: `{"bourrage":"${'x'.repeat(8 * 1024 * 1024)}"}` });
        const q = Object.fromEntries(u.searchParams);
        let corps;
        if (u.hostname.startsWith('geocoding-api')) {
          const g = LIEUX[(q.name || '').split(',')[0].trim().toLowerCase()];
          corps = g ? { results: [g], generationtime_ms: 0.61 } : { generationtime_ms: 0.43 };
        } else {
          const js = jours(q.start_date, q.end_date);
          // une archive qui commence le LENDEMAIN du jour demandé (relecture adverse : le relevé glissait d'un jour, sans un mot)
          if (mode.decale && u.hostname.startsWith('archive-api')) js.shift();
          corps = { latitude: 50.85, longitude: 4.35, generationtime_ms: 0.2, utc_offset_seconds: 3600, timezone: 'Europe/Brussels', timezone_abbreviation: 'GMT+1', elevation: 28 };
          if (q.hourly) {
            corps.hourly_units = { time: 'iso8601', [q.hourly]: '°C' };
            corps.hourly = { time: js.flatMap((d) => [...Array(24).keys()].map((h) => `${d}T${String(h).padStart(2, '0')}:00`)), [q.hourly]: js.flatMap((d) => [...Array(24).keys()].map((h) => tHeure(d, h))) };
          } else {
            corps.daily_units = { time: 'iso8601', [q.daily]: '°C' };
            corps.daily = { time: js, [q.daily]: js.map(tJour) };
          }
        }
        return route.fulfill({ status: 200, headers: CORS, contentType: 'application/json', body: JSON.stringify(corps) });
      });
    };
    const preparer = async (w, h, tel) => {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel });
      const journal = [], mode = {};
      await brancher(c, journal, mode);
      await c.addInitScript(() => {
        window.__csp = [];
        document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
      });
      const pg = await c.newPage();
      pg.journal = journal; pg.mode = mode; pg.erreurs = []; pg.ailleurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') pg.erreurs.push(m.text().slice(0, 300)); });
      // aucune autre adresse extérieure n'est jamais demandée
      c.on('request', (r) => { const u = r.url(); if (!u.startsWith(BASE) && !OPEN_METEO.test(u) && !/^(data|blob):/.test(u)) pg.ailleurs.push(u); });
      return pg;
    };
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const ouvrir = async (pg) => {
      const rep = await pg.goto(`${BASE}/${MODS.age.page}?chantier=${encodeURIComponent('Contrôle météo')}&nom=${encodeURIComponent('Dalle R+2')}`);
      await pg.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
      return rep;
    };
    const tranquille = (pg) => pg.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 90000 });
    const saisirDate = async (pg, chemin, v) => { await pg.fill(`#donnees input[data-chemin="${chemin}"]`, v); await pg.press(`#donnees input[data-chemin="${chemin}"]`, 'Enter'); await tranquille(pg); };
    const lire = (pg) => pg.evaluate(() => {
      const vis = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const b = document.querySelector('#donnees [data-meteo-chemin="temperature/meteo"]');
      const rb = b ? b.getBoundingClientRect() : null;
      const t = document.getElementById('toast');
      const ta = document.querySelector('#donnees textarea[data-chemin="temperature/releve"]');
      const hints = [...document.querySelectorAll('#donnees .hint, #donnees .msg')].filter(vis).map((x) => x.textContent.replace(/\s+/g, ' ').trim());
      return { plan: b ? JSON.parse(b.dataset.meteo) : null, bouton: b ? { vu: vis(b), l: Math.round(rb.width), h: Math.round(rb.height), grise: b.disabled, texte: b.textContent.trim() } : null,
        toast: { vu: vis(t), texte: t.textContent, erreur: t.classList.contains('erreur') }, releve: ta ? ta.value : null, releveVu: vis(ta), hints,
        verdict: document.querySelector('#resultat .verdict')?.textContent.replace(/\s+/g, ' ').trim() || '',
        vues: [...document.querySelectorAll('#vues [data-vue]')].map((x) => x.dataset.vue) };
    });
    // ce que l'œil voit : rien ne déborde, aucun libellé coupé (segments de la cure, bouton, badge)
    const regard = (pg) => pg.evaluate(() => {
      const vis = (e) => e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const large = (e) => { const r = document.createRange(); r.selectNodeContents(e); return r.getBoundingClientRect().width; };
      const coupes = [...document.querySelectorAll('#donnees .rows .seg.full button, #donnees .btn.meteo > span, #donnees .row > label > span:not(.sym)')]
        .filter(vis).filter((e) => e.scrollWidth > e.clientWidth + 1 || large(e) > e.getBoundingClientRect().width + 0.5).map((e) => e.textContent.trim());
      return { deborde: document.documentElement.scrollWidth > innerWidth, coupes };
    });
    const attendre = (pg, f, ms = 60000) => pg.waitForFunction(f, null, { timeout: ms }).then(() => true, () => false);
    const ligneReleve = (d, src) => { const t = tJour(d); return `${d};${(Number.isInteger(t) ? t.toFixed(1) : String(t)).replace('.', ',')};${src}`; };

    // ------------------------------------------------ au PC
    const g = await preparer(1440, 900, false);
    const rep = await ouvrir(g);
    verif('météo · PC : la page est servie avec la politique de sécurité qui permet les trois adresses d\'Open-Meteo, et elles seules',
      /connect-src 'self' https:\/\/geocoding-api\.open-meteo\.com https:\/\/archive-api\.open-meteo\.com https:\/\/api\.open-meteo\.com;/.test(rep.headers()['content-security-policy'] || ''),
      (rep.headers()['content-security-policy'] || '').slice(0, 200));
    const v20 = (await lire(g)).verdict;
    await g.click('#donnees button[data-chemin="temperature/mode"][data-valeur="meteo"]');
    await tranquille(g);
    let L = await lire(g);
    verif('météo · PC : « Météo du lieu » ouvre le lieu, les dates et le bouton « Récupérer la météo », visible, sur la largeur du cadre',
      L.plan && L.bouton && L.bouton.vu && L.bouton.l > 250 && L.bouton.h >= 32 && !L.bouton.grise && L.bouton.texte === 'Récupérer la météo'
      && (await g.inputValue('#donnees input[data-chemin="temperature/lieu"]')) === 'Bruxelles, Belgique', JSON.stringify(L.bouton));
    const auj = L.plan.aujourdhui;
    const debut = decale(auj, -10), fin = decale(auj, 18);
    await saisirDate(g, 'temperature/debut', fr(debut));
    await saisirDate(g, 'temperature/fin', fr(fin));
    L = await lire(g);
    const plan = L.plan;
    verif('météo · PC : le plan du bouton suit les dates saisies (archive puis prévisions, 2 jours prolongés au-delà de l\'horizon)',
      plan.debut === debut && plan.fin === fin && plan.appels.map((a) => a.source).join() === 'archive,prevision' && plan.n_prolonges === 2, JSON.stringify(plan).slice(0, 300));
    let m = await regard(g);
    verif('météo · PC : rien ne déborde, aucun libellé coupé (« Météo du lieu », le bouton, les champs Lieu, Début, Fin)', !m.deborde && !m.coupes.length, m.coupes.join(', '));
    // le clic : l'indicateur pendant la récupération (le faux service répond en 0,8 s)
    g.mode.lent = 800;
    await g.click('#donnees [data-meteo-chemin="temperature/meteo"]');
    const pendant = await attendre(g, () => {
      const b = document.querySelector('#donnees [data-meteo-chemin="temperature/meteo"]');
      const c = [...document.querySelectorAll('.calcule')].find((x) => x.checkVisibility() && x.textContent);
      return b && b.disabled && b.textContent.trim() === 'Récupération de la météo…' && b.checkVisibility() && c && c.textContent === 'Récupération de la météo…';
    }, 5000);
    verif('météo · PC : pendant la récupération, le bouton et l\'indicateur disent « Récupération de la météo… » (le bouton ne répond plus)', pendant);
    await tranquille(g);
    g.mode.lent = 0;
    L = await lire(g);
    const nJ = jours(debut, fin).length, nMes = jours(debut, auj).length, nPrev = jours(decale(auj, 1), plan.fin_dispo).length;
    const att = [`https://geocoding-api.open-meteo.com/v1/search?name=Bruxelles%2C+Belgique&count=1&language=fr`]
      .concat(plan.appels.map((a) => `https://${a.source === 'archive' ? 'archive-api' : 'api'}.open-meteo.com/v1/${a.source === 'archive' ? 'archive' : 'forecast'}?latitude=50.85045&longitude=4.34878`
        + `&start_date=${a.debut}&end_date=${a.fin}&timezone=auto&daily=temperature_2m_mean`));
    verif('météo · PC : le navigateur a fait les appels du plan, vers Open-Meteo seulement (géocodage, archive, prévisions ; paramètres encodés) — la politique de sécurité les laisse passer',
      JSON.stringify(g.journal) === JSON.stringify(att) && !g.ailleurs.length, JSON.stringify({ journal: g.journal, att, ailleurs: g.ailleurs }));
    const badge = `Bruxelles, Bruxelles-Capitale, Belgique — ${nMes} j mesurés · ${nPrev} j de prévision · 2 j prolongés`;
    verif('météo · PC : les températures reçues : le badge (lieu trouvé, jours mesurés / prévus / prolongés), la date de récupération, un message',
      L.hints.some((x) => x === badge) && L.hints.some((x) => x.startsWith(`Open-Meteo · récupérée le ${fr(auj)} à `)) && L.toast.vu && !L.toast.erreur
      && /^Météo récupérée/.test(L.toast.texte), JSON.stringify({ hints: L.hints.slice(0, 6), toast: L.toast }));
    const lignes = (L.releve || '').split('\n');
    verif('météo · PC : le relevé reçu est écrit, jour par jour, modifiable (une ligne par jour : date;T;source)',
      L.releveVu && lignes.length === nJ && lignes[0] === ligneReleve(debut, 'mesuré') && lignes[nMes] === ligneReleve(decale(auj, 1), 'prévision')
      && lignes[nJ - 1].endsWith(';prolongé'), JSON.stringify({ n: lignes.length, nJ, l0: lignes[0], att0: ligneReleve(debut, 'mesuré') }));
    verif('météo · PC : la maturité est recalculée (le verdict n\'est plus celui de 20 °C), les vues « Températures » et « Tableau » sont là',
      L.verdict && L.verdict !== v20 && L.vues.join() === 'courbe,releve,tableau', JSON.stringify({ v20, v: L.verdict, vues: L.vues }));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc.png') });
    // les vues : la courbe des températures et le tableau, vus à l'écran
    await g.click('#vues [data-vue="releve"]');
    await g.waitForTimeout(250);
    const courbe = await g.evaluate(() => { const s = [...document.querySelectorAll('.view.on .draw > svg')].find((x) => x.checkVisibility()); const r = s && s.getBoundingClientRect();
      return { titre: document.querySelector('.view.on .stitle b')?.textContent, leg: document.querySelector('.view.on .stitle .leg')?.textContent, l: r ? Math.round(r.width) : 0, h: r ? Math.round(r.height) : 0 }; });
    verif('météo · PC : la vue « Météo du lieu » montre la courbe des températures reçues, à sa taille, sa légende dit Open-Meteo',
      courbe.titre === 'Météo du lieu' && /Open-Meteo/.test(courbe.leg || '') && courbe.l > 400 && courbe.h > 250, JSON.stringify(courbe));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc_temperatures.png') });
    await g.click('#vues [data-vue="tableau"]');
    await g.waitForTimeout(250);
    const tab = await g.evaluate(() => { const rows = [...document.querySelectorAll('.view.on .age-tab tbody tr')]; const r0 = rows[0];
      return { n: rows.length, vu: !!r0 && r0.checkVisibility(), cases: r0 ? [...r0.children].map((x) => x.textContent) : [] }; });
    verif('météo · PC : le tableau montre une ligne par jour reçu (date, température, source)',
      tab.n === nJ && tab.vu && tab.cases[1] === fr(debut) && tab.cases[3] === 'mesuré', JSON.stringify(tab));
    // la note rechargée : tout est gardé, et rien n'est redemandé à Open-Meteo (le calcul se refait hors ligne)
    const nAppels = g.journal.length;
    await g.reload();
    await g.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
    await tranquille(g);
    const R = await lire(g);
    verif('météo · PC : rechargée, la note garde les températures, le badge et la maturité — sans AUCUN nouvel appel (calcul hors ligne)',
      R.releve === L.releve && R.hints.includes(badge) && R.verdict === L.verdict && g.journal.length === nAppels, JSON.stringify({ appels: g.journal.length - nAppels, v: R.verdict }));
    // le détail horaire, à la demande : la position, sans géocodage ; T min h / T max h dans le tableau
    await g.click('#donnees [data-meteo-chemin="temperature/meteo_horaire"]');
    await tranquille(g);
    const hor = g.journal.slice(nAppels);
    await g.click('#vues [data-vue="tableau"]');
    await g.waitForTimeout(250);
    const th = await g.evaluate(() => [...document.querySelectorAll('.view.on .age-tab thead th')].map((x) => x.textContent));
    const t0 = await g.evaluate(() => [...document.querySelector('.view.on .age-tab tbody tr').children].map((x) => x.textContent));
    const hs = [...Array(24).keys()].map((h) => tHeure(debut, h));
    const f1 = (x) => (Math.round(x * 10) / 10).toFixed(1).replace('.', ',').replace('-', '−');
    verif('météo · PC : « Charger le détail horaire » : appels horaires à la position trouvée (sans géocodage) ; le tableau ajoute T min h, T max h',
      hor.length === plan.appels.length && hor.every((u) => /latitude=50\.85045&longitude=4\.34878/.test(u) && /hourly=temperature_2m/.test(u) && !/geocoding/.test(u))
      && th.includes('T min h°C') && th.includes('T max h°C') && t0[4] === f1(Math.min(...hs)) && t0[5] === f1(Math.max(...hs)), JSON.stringify({ hor, th, t0 }));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc_tableau.png') });
    // une correction à la main : le calcul la suit, l'écran le dit, la récupération reste
    const corrige = L.releve.replace(/^([^;]+);[^;]+;/, '$1;-4,0;');
    await g.fill('#donnees textarea[data-chemin="temperature/releve"]', corrige);
    await g.evaluate(() => document.activeElement.blur());
    await tranquille(g);
    const C2 = await lire(g);
    verif('météo · PC : une température corrigée à la main recalcule la maturité ; l\'écran dit « corrigées à la main » et garde la récupération',
      C2.releve === corrige && C2.verdict !== R.verdict && C2.hints.some((x) => /corrigées à la main/.test(x)) && C2.hints.includes(badge.replace(`${nMes} j mesurés`, `${nMes} j mesurés`)),
      JSON.stringify({ v: C2.verdict, hints: C2.hints.slice(0, 8) }));
    // les refus : chacun en message, la note intacte
    const refus = async (nom, avant, motif) => {
      const r0 = await lire(g);
      const n0 = g.journal.length;
      await avant();
      await g.click('#donnees [data-meteo-chemin="temperature/meteo"]');
      await tranquille(g);
      await g.waitForTimeout(200);
      const r1 = await lire(g);
      verif(`météo · PC · ${nom} : un message en français, la note intacte (températures et maturité inchangées), jamais une exception`,
        r1.toast.vu && r1.toast.erreur && motif.test(r1.toast.texte) && r1.releve === r0.releve && r1.verdict === r0.verdict && !r1.bouton.grise,
        JSON.stringify({ toast: r1.toast, appels: g.journal.slice(n0) }));
      return g.journal.slice(n0);
    };
    await saisirDate(g, 'temperature/lieu', 'Atlantide');
    const a1 = await refus('lieu introuvable', async () => {}, /^Lieu introuvable : « Atlantide »/);
    verif('météo · PC · lieu introuvable : le géocodage seul est demandé', a1.length === 1 && /geocoding-api.*name=Atlantide/.test(a1[0]), JSON.stringify(a1));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc_introuvable.png') });
    await saisirDate(g, 'temperature/lieu', 'Bruxelles, Belgique');
    await refus('réseau coupé', async () => { g.mode.coupe = true; }, /^Pas de réseau : la météo n'a pas pu être récupérée.*hors ligne/);
    g.mode.coupe = false;
    await refus('service en panne', async () => { g.mode.panne = true; }, /^Service météo en panne \(erreur 500 — Panne simulée du service\)/);
    g.mode.panne = false;
    // (relecture adverse) une réponse dont les jours ne sont pas ceux demandés, une réponse énorme : refusées en message
    await refus('archive décalée d\'un jour', async () => { g.mode.decale = true; }, new RegExp(`^Météo refusée — réponse incomplète du service météo : le ${fr(debut)} manque`));
    g.mode.decale = false;
    const aE = await refus('réponse énorme (8 Mo)', async () => { g.mode.enorme = true; }, /^Réponse trop volumineuse du service météo \(plus de 5 Mo\)/);
    g.mode.enorme = false;
    verif('météo · PC · réponse énorme : refusée dès le géocodage, aucun autre appel', aE.length === 1 && /geocoding-api/.test(aE[0]), JSON.stringify(aE));
    // (relecture adverse) un AUTRE coulage ouvert pendant la récupération : la météo allait dans le coulage ouvert au retour —
    // une copie (même lieu, mêmes dates) perdait ses corrections à la main, et celui du clic restait sans météo
    await g.click('#donnees [data-dupliquer]');
    await tranquille(g);
    await g.click('.elements .ni[data-activer="0"]');
    await tranquille(g);
    const c1 = await lire(g);
    g.mode.lent = 2500;
    await g.click('#donnees [data-meteo-chemin="temperature/meteo"]');
    await attendre(g, () => !!document.querySelector('#donnees [data-meteo].attente'), 3000);
    await g.click('.elements .ni[data-activer="1"]');
    await attendre(g, () => !document.querySelector('#donnees [data-meteo].attente'), 30000);
    await tranquille(g);
    await g.waitForTimeout(200);
    g.mode.lent = 0;
    const c2 = await lire(g);
    await g.click('.elements .ni[data-activer="0"]');
    await tranquille(g);
    const c1b = await lire(g);
    verif('météo · PC · un autre coulage ouvert pendant la récupération : rien n\'est importé, un message le dit ; la copie ouverte garde ses corrections à la main, le coulage du clic aussi',
      c2.toast.vu && c2.toast.erreur && /^Un autre coulage a été ouvert pendant la récupération/.test(c2.toast.texte) && c2.releve === c1.releve
      && c1b.releve === c1.releve && c2.hints.some((x) => /corrigées à la main/.test(x)), JSON.stringify({ toast: c2.toast, r2: (c2.releve || '').slice(0, 40), r1: (c1.releve || '').slice(0, 40) }));
    // dates hors de portée : le bouton est grisé, le message est écrit dessous — aucun appel
    const n0 = g.journal.length;
    await saisirDate(g, 'temperature/fin', fr(decale(auj, 30)));
    await saisirDate(g, 'temperature/debut', fr(decale(auj, 25)));
    const H = await lire(g);
    const msgVu = await g.evaluate(() => { const m = document.querySelector('#donnees .meteo-bt + .msg.grave'); return !!m && m.checkVisibility() && m.getBoundingClientRect().height > 10 ? m.textContent : ''; });
    verif('météo · PC · dates hors de portée : le bouton grisé, le message (horizon des prévisions) visible dessous, aucun appel',
      H.bouton.grise && /horizon des prévisions/.test(msgVu) && g.journal.length === n0, JSON.stringify({ b: H.bouton, msgVu }));
    await g.screenshot({ path: path.join(OUT, 'site_meteo_pc_hors_portee.png') });
    const csp = await g.evaluate(() => window.__csp);
    // les erreurs attendues : la ressource refusée (réseau coupé) et la réponse 500 de la panne simulée
    const err = g.erreurs.filter((x) => !/Failed to load resource: (net::ERR_INTERNET_DISCONNECTED|the server responded with a status of 500)/.test(x));
    verif('météo · PC : aucune violation de la politique de sécurité, aucune autre adresse extérieure, aucune erreur JavaScript',
      !csp.length && !g.ailleurs.length && !err.length, JSON.stringify({ csp, ailleurs: g.ailleurs, err }));
    await g.context().close();

    // ------------------------------------------------ au téléphone et à la tablette : le même geste, dans l'onglet Données
    for (const [fmt, w, h, tel, nom] of [['téléphone', 390, 844, true, 'tel'], ['tablette portrait', 820, 1180, false, 'tab']]) {
      const t = await preparer(w, h, tel);
      await ouvrir(t);
      if (tel) await t.click('.mtabs [data-onglet="donnees"]');
      await t.click('#donnees button[data-chemin="temperature/mode"][data-valeur="meteo"]');
      await tranquille(t);
      let M = await lire(t);
      const mm = await regard(t);
      verif(`météo · ${fmt} : la cure « Météo du lieu », le bouton visible et à la taille du doigt, rien ne déborde, aucun libellé coupé`,
        M.bouton && M.bouton.vu && M.bouton.h >= (tel ? 40 : 32) && M.bouton.l >= (tel ? 300 : 250) && !mm.deborde && !mm.coupes.length, JSON.stringify({ b: M.bouton, ...mm }));
      await t.locator('#donnees [data-meteo-chemin="temperature/meteo"]').scrollIntoViewIfNeeded();
      await t.click('#donnees [data-meteo-chemin="temperature/meteo"]');
      await tranquille(t);
      M = await lire(t);
      const n = jours(M.plan.debut, M.plan.fin).length;
      verif(`météo · ${fmt} : récupérée — le badge, la date de récupération et le relevé reçu (${n} j) se lisent dans les données`,
        M.hints.includes(`Bruxelles, Bruxelles-Capitale, Belgique — ${n} j mesurés`) && M.hints.some((x) => x.startsWith('Open-Meteo · récupérée le '))
        && M.releveVu && M.releve.split('\n').length === n && t.journal.length === 3, JSON.stringify({ hints: M.hints.slice(0, 5), appels: t.journal.length }));
      const m2 = await regard(t);
      verif(`météo · ${fmt} : après la récupération, rien ne déborde, aucun libellé coupé`, !m2.deborde && !m2.coupes.length, JSON.stringify(m2));
      await t.locator('#donnees .age-mini').scrollIntoViewIfNeeded();
      await t.screenshot({ path: path.join(OUT, `site_meteo_${nom}_donnees.png`) });
      if (tel) await t.click('.mtabs [data-onglet="dessin"]');
      await t.click('#vues [data-vue="releve"]');
      await t.waitForTimeout(300);
      const vue = await t.evaluate(() => { const s = [...document.querySelectorAll('.view.on .draw > svg')].find((x) => x.checkVisibility()); const r = s && s.getBoundingClientRect(); return r ? [Math.round(r.width), Math.round(r.height)] : null; });
      verif(`météo · ${fmt} : la vue des températures reçues est visible, à sa taille`, vue && vue[0] >= (tel ? 300 : 500) && vue[1] > 150, JSON.stringify(vue));
      await t.screenshot({ path: path.join(OUT, `site_meteo_${nom}_temperatures.png`) });
      await t.click('#vues [data-vue="tableau"]');
      await t.waitForTimeout(300);
      const tb = await t.evaluate(() => [...document.querySelectorAll('.view.on .age-tab tbody tr')].filter((x) => x.checkVisibility()).length);
      verif(`météo · ${fmt} : le tableau des jours reçus se voit`, tb === n, `${tb} / ${n}`);
      await t.screenshot({ path: path.join(OUT, `site_meteo_${nom}_tableau.png`) });
      // le refus au téléphone : réseau coupé → un message visible
      if (tel) {
        await t.click('.mtabs [data-onglet="donnees"]');
        t.mode.coupe = true;
        await t.locator('#donnees [data-meteo-chemin="temperature/meteo"]').scrollIntoViewIfNeeded();
        await t.click('#donnees [data-meteo-chemin="temperature/meteo"]');
        await tranquille(t);
        await t.waitForTimeout(200);
        const T = await lire(t);
        const r = await t.evaluate(() => { const x = document.getElementById('toast').getBoundingClientRect(); return x.right <= innerWidth + 0.5 && x.left >= -0.5 && x.bottom <= innerHeight + 0.5; });
        verif('météo · téléphone · réseau coupé : le message « Pas de réseau » se lit, dans l\'écran ; les températures restent',
          T.toast.vu && T.toast.erreur && /^Pas de réseau/.test(T.toast.texte) && r && T.releve === M.releve, JSON.stringify(T.toast));
        await t.screenshot({ path: path.join(OUT, 'site_meteo_tel_pas_de_reseau.png') });
      }
      const err = t.erreurs.filter((x) => !/Failed to load resource: net::ERR_INTERNET_DISCONNECTED/.test(x));
      verif(`météo · ${fmt} : aucune violation de la politique de sécurité, aucune autre adresse, aucune erreur JavaScript`,
        !(await t.evaluate(() => window.__csp.length)) && !t.ailleurs.length && !err.length, JSON.stringify({ ailleurs: t.ailleurs, err }));
      await t.context().close();
    }
  }

  // ---------------------------------------------------------------- la SONDE du démarrage (js/sonde.js)
  // Une MESURE, fermée par défaut. Sans « ?sonde=1 », l'établi fait EXACTEMENT ce qu'il faisait : aucune marque
  // « stab: » sur la page, aucun message de plus entre la page et le travailleur (les types reçus sont lus sur le vrai
  // Worker), l'adresse du travailleur sans « sonde », le même écran (#donnees, #resultat, au caractère près, comparés
  // à ceux de la page sondée). Avec « ?sonde=1 » : les marques de la page et du travailleur, dans l'ordre du démarrage,
  // l'import du module daté à part de son premier calcul, sans erreur. Poutre (sans numpy) et charges sur plancher
  // (numpy chargé d'emblée), au PC et au téléphone.
  //   filtre « sonde ».
  if (!FILTRE || FILTRE === 'sonde') {
    await p.goto(`${BASE}/`);
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const espion = () => {
      window.__messages = []; window.__travailleurs = [];
      const W = window.Worker;
      window.Worker = class extends W {
        constructor(url, o) {
          super(url, o);
          window.__travailleurs.push(String(url));
          this.addEventListener('message', (ev) => window.__messages.push(ev.data && ev.data.type));
        }
      };
    };
    const charger = async (cle, fmt, sonde) => {
      const tel = fmt === 'téléphone';
      const c = await b.newContext({ viewport: tel ? { width: 390, height: 844 } : { width: 1440, height: 900 }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tel });
      await c.addInitScript(espion);
      const pg = await c.newPage();
      const err = [];
      pg.on('pageerror', (e) => err.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error') err.push(m.text().slice(0, 300)); });
      await pg.goto(`${BASE}/${MODS[cle].page}${sonde ? '?sonde=1' : ''}`);
      await pg.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
      if (sonde) await pg.waitForFunction(() => globalThis.__sonde && performance.getEntriesByName('stab:premier_rendu').length, null, { timeout: 60000 });
      else await pg.waitForTimeout(1500);   // le temps qu'un message de trop arrive, s'il devait arriver
      const lu = await pg.evaluate(() => ({
        echec: document.getElementById('demarrage').classList.contains('erreur') ? document.getElementById('demarrage-etape').textContent : '',
        page: performance.getEntriesByType('mark').filter((m) => m.name.startsWith('stab:')).map((m) => [m.name.slice(5), m.startTime]),
        travailleur: globalThis.__sonde ? globalThis.__sonde.marques.map(([k, t]) => [k, t - performance.timeOrigin]) : null,
        details: globalThis.__sonde ? globalThis.__sonde.details : null,
        sonde: globalThis.__sonde !== undefined,
        types: [...new Set(window.__messages)].sort(), adresses: window.__travailleurs,
        donnees: document.getElementById('donnees').innerHTML, resultat: document.getElementById('resultat').innerHTML,
      }));
      await c.close();
      return { ...lu, err };
    };
    for (const cle of ['poutre', 'plancher']) {
      for (const fmt of ['PC', 'téléphone']) {
        const sans = await charger(cle, fmt, false);
        const avec = await charger(cle, fmt, true);
        const nom = `sonde · ${cle} · ${fmt}`;
        verif(`${nom} : sans « ?sonde=1 », le moteur démarre, aucune marque, aucun message de plus (étape, prêt, réponse), le travailleur à son adresse d'avant`,
          !sans.echec && !sans.page.length && !sans.sonde && JSON.stringify(sans.types) === '["etape","pret","reponse"]'
          && sans.adresses.length === 1 && !/sonde/.test(sans.adresses[0]) && !sans.err.length,
          JSON.stringify({ echec: sans.echec, page: sans.page, types: sans.types, adresses: sans.adresses, err: sans.err }).slice(0, 400));
        verif(`${nom} : avec ou sans la sonde, le même écran (#donnees, #resultat au caractère près)`,
          sans.donnees === avec.donnees && sans.resultat === avec.resultat && sans.donnees.length > 100,
          `${sans.donnees.length}/${avec.donnees.length} · ${sans.resultat.length}/${avec.resultat.length}`);
        // les marques, dans l'ordre du démarrage : la page, puis le travailleur, ramenés à la même origine
        const P = Object.fromEntries(avec.page), T = Object.fromEntries((avec.travailleur || []).filter(([k]) => !k.startsWith('wasm_')));
        const ordre = [['page', 'etabli_debut'], ['page', 'moteur_cree'], ['trav', 'travailleur_debut'], ['trav', 'manifeste_debut'], ['trav', 'loadPyodide_debut'],
          ['trav', 'loadPyodide_fin'], ['trav', 'empreinte_debut'], ['trav', 'deballage_fin'], ['trav', 'numpy_attendu'], ['trav', 'routeur_importe'],
          ['trav', 'pret'], ['page', 'pret_recu'], ['page', 'requete1_envoyee'], ['trav', 'module_import_debut'], ['trav', 'module_import_fin'],
          ['trav', 'requete1_debut'], ['trav', 'requete1_traitee'], ['trav', 'reponse1_postee'], ['page', 'reponse1_recue'],
          ['page', 'reponse1_appliquee'], ['page', 'premier_rendu']];
        const temps = ordre.map(([o, k]) => (o === 'page' ? P : T)[k]);
        const manque = ordre.filter((_, i) => temps[i] == null).map(([o, k]) => `${o}.${k}`);
        const desordre = ordre.filter((_, i) => i && temps[i] != null && temps[i - 1] != null && temps[i] < temps[i - 1] - 2).map(([o, k]) => `${o}.${k}`);
        const wasm = (avec.travailleur || []).some(([k]) => k.startsWith('wasm_'));
        const numpy = cle === 'plancher' ? T.numpy_debut != null && T.numpy_fin != null && T.numpy_import_fin != null : T.numpy_debut == null;
        verif(`${nom} : avec « ?sonde=1 », les marques de la page et du travailleur dans l'ordre du démarrage (wasm, numpy ${cle === 'plancher' ? 'chargé' : 'absent'})`,
          !avec.echec && !manque.length && !desordre.length && wasm && numpy && !avec.err.length,
          JSON.stringify({ echec: avec.echec, manque, desordre, wasm, numpy, err: avec.err }).slice(0, 400));
        const D = avec.details || {};
        verif(`${nom} : la sonde sépare l'import du module de son premier calcul, sans erreur, et mesure une requête ordinaire`,
          !D.import_erreur && !D.sonde_erreur && D.modules_apres_module > D.modules_apres_routeur && D.calculer_ms > 0 && D.cles_ms > 0
          && JSON.stringify(avec.types) === '["etape","pret","reponse","sonde"]', JSON.stringify({ ...D, types: avec.types }).slice(0, 400));
      }
    }
  }

  // ---------------------------------------------------------------- l'enregistrement, vu à l'écran
  // Un échec d'enregistrement se LIT à toutes les largeurs (1 440, 1 280, 820 et 390 px) : l'indicateur .et-enr — la
  // barre du haut au PC et à la tablette, la 2ᵉ ligne de l'en-tête au téléphone — passe au rouge, en entier, et y RESTE
  // (au-delà des 7 s d'un message éphémère) ; le message #et-alerte le dit une fois, avec « Exporter .stab », qui
  // exporte bien la note ; fermé, il ne revient pas à chaque saisie ; une écriture qui réussit efface tout. Défaut d'avant :
  // la charte cachait le texte (.saved .txt) sous 1 360 px — à 1 280 et 820 px, une coche grise, muette ; au téléphone,
  // rien tant que le menu restait fermé. Puis la mémoire des notes ABÎMÉE : jamais réécrite sans que son texte soit
  // copié à part ; la copie impossible, rien n'est écrit et l'écran le dit ; l'accueil le signale et rend le texte tel quel.
  //   filtre « enregistrement »
  if (!FILTRE || FILTRE === 'enregistrement') {
    const fs = require('fs');
    await p.goto(`${BASE}/`);
    const MODS = await p.evaluate(async () => (await import('./js/modules.js')).MODULES);
    const CLE = 'stab.notes.v1', PREFIXE = 'stab.notes.v1.illisible.';
    // la mémoire d'un bureau, abîmée : 30 notes, le JSON tronqué d'un octet (la mesure du jury)
    const trente = {};
    for (let i = 0; i < 30; i++) {
      const id = `abime${String(i).padStart(2, '0')}`;
      trente[id] = { id, module: 'poutre', nom: `Poutre P${i + 1}`, chantier: 'Réalex', dossier: 'Fondations', indice: 'A', cree: 1762500000000 + i,
        modifie: 1762500000000 + i, etat: null, resume: { taux: 40 + i, niveau: 'p' }, vignette: '' };
    }
    const ABIME = JSON.stringify(trente).slice(0, -1);
    const ouvrirContexte = async (w, h, tel, tactile, init) => {
      const c = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: tactile, acceptDownloads: true });
      if (init) await c.addInitScript(init);
      const pg = await c.newPage();
      pg.erreurs = [];
      pg.on('pageerror', (e) => pg.erreurs.push(e.message));
      pg.on('console', (m) => { if (m.type() === 'error' && !/enregistrement impossible|copie de la mémoire/.test(m.text())) pg.erreurs.push(m.text().slice(0, 300)); });
      return pg;
    };
    const demarre = async (pg) => {
      await pg.waitForFunction(() => { const d = document.getElementById('demarrage'); return d && (d.hidden || d.classList.contains('erreur')); }, null, { timeout: 180000 });
      return pg.evaluate(() => (document.getElementById('demarrage').classList.contains('erreur') ? document.getElementById('demarrage-etape').textContent : ''));
    };
    const calmePg = (pg) => pg.waitForFunction(() => !document.querySelector('.calcule.on'), null, { timeout: 60000 });
    // une saisie dans le premier champ numérique VISIBLE des données (une valeur qui change à chaque appel)
    let saisies = 0;
    const saisir = async (pg) => {
      const sel = await pg.evaluate(() => {
        const e = [...document.querySelectorAll('#donnees input[data-sorte="nombre"]:not([disabled])')].find((x) => x.checkVisibility());
        return e ? e.dataset.chemin : null;
      });
      if (!sel) return false;
      const champ = `#donnees input[data-chemin="${sel}"]`;
      const v = Number(String(await pg.inputValue(champ)).replace(/\s/g, '').replace(',', '.')) || 10;
      await pg.fill(champ, String(Math.round(v * (1.01 + 0.01 * (++saisies % 5)) * 100) / 100).replace('.', ','));
      await pg.press(champ, 'Enter');
      await calmePg(pg);
      await pg.waitForTimeout(150);
      return true;
    };
    // ce que l'œil voit de l'indicateur VISIBLE à ce format (il y en a un, et un seul) : son état, son texte entier (ni
    // caché, ni coupé par un parent, ni couvert), sa couleur comparée à celle de --nok ; et le message
    const lireEnr = (pg) => pg.evaluate(() => {
      const vu = (e) => !!e && e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
      const vis = [...document.querySelectorAll('.et-enr')].filter(vu);
      const e = vis[0];
      const t = e ? e.querySelector('.et-enr-txt') : null;
      const sonde = document.createElement('span'); sonde.style.color = 'var(--nok)'; document.body.appendChild(sonde);
      const rouge = getComputedStyle(sonde).color; sonde.remove();
      let entier = false, couvert = [], boite = null;
      if (t && vu(t)) {
        const rg = document.createRange(); rg.selectNodeContents(t);
        const r = rg.getBoundingClientRect();
        boite = { l: Math.round(r.width), h: Math.round(r.height), g: Math.round(r.left), d: Math.round(r.right), y: Math.round(r.top) };
        // le texte entier dans chaque parent qui rogne, et dans la fenêtre
        entier = r.width > 20 && r.height > 6 && r.left >= -0.5 && r.right <= innerWidth + 0.5 && r.top >= -0.5 && r.bottom <= innerHeight + 0.5
          && t.scrollWidth <= t.clientWidth + 1;
        for (let a = t.parentElement; a && entier; a = a.parentElement) {
          if (getComputedStyle(a).overflow === 'visible') continue;
          const q = a.getBoundingClientRect();
          if (r.left < q.left - 0.5 || r.right > q.right + 0.5 || r.top < q.top - 0.5 || r.bottom > q.bottom + 0.5) entier = false;
        }
        // rien ne le couvre : trois points du texte appartiennent à l'indicateur
        for (const x of [r.left + 3, (r.left + r.right) / 2, r.right - 3]) {
          const f = document.elementFromPoint(x, (r.top + r.bottom) / 2);
          if (!f || !f.closest('.et-enr')) couvert.push(f ? (f.id || f.className || f.tagName).toString().slice(0, 30) : 'rien');
        }
      }
      const al = document.getElementById('et-alerte');
      const bt = al && al.querySelector('.et-alerte-stab');
      const ra = al && vu(al) ? al.getBoundingClientRect() : null;
      const rb = bt && vu(bt) ? bt.getBoundingClientRect() : null;
      return { visibles: vis.length, etat: e ? e.dataset.enr : null, texte: t ? t.textContent : null, titre: e ? e.title : '', entier, couvert, boite,
        rouge: !!t && getComputedStyle(t).color === rouge, couleur: t ? getComputedStyle(t).color : null,
        alerte: ra ? { texte: al.querySelector('.et-alerte-txt').textContent, info: al.classList.contains('info'),
          dedans: ra.left >= -0.5 && ra.right <= innerWidth + 0.5 && ra.top >= -0.5 && ra.bottom <= innerHeight + 0.5,
          bouton: rb ? { texte: bt.textContent.trim(), h: Math.round(rb.height), l: Math.round(rb.width) } : null } : null,
        deborde: document.documentElement.scrollWidth > innerWidth };
    });
    const lireMemoire = (pg) => pg.evaluate((k) => {
      const out = { principal: localStorage.getItem(k), copies: {} };
      for (let i = 0; i < localStorage.length; i++) { const c = localStorage.key(i); if (c.startsWith(k + '.illisible.')) out.copies[c] = localStorage.getItem(c); }
      return out;
    }, CLE);
    const telechargement = async (pg, sel) => {
      const [dl] = await Promise.all([pg.waitForEvent('download', { timeout: 15000 }), pg.click(sel)]);
      return { nom: dl.suggestedFilename(), texte: fs.readFileSync(await dl.path(), 'utf8') };
    };

    // ------------------------------------------------ 1. le stockage plein, aux quatre largeurs
    for (const [fmt, w, h, tel, tactile, attendre] of [['PC 1440', 1440, 900, false, false, true], ['portable 1280', 1280, 800, false, false, false],
      ['tablette 820', 820, 1180, false, true, false], ['téléphone 390', 390, 844, true, true, true]]) {
      const nom = `enregistrement · ${fmt}`;
      const pg = await ouvrirContexte(w, h, tel, tactile);
      try {
        await pg.goto(`${BASE}/${MODS.poutre.page}?nom=${encodeURIComponent(`Essai d'enregistrement · ${fmt}`)}&chantier=R%C3%A9alex`);
        const ko0 = await demarre(pg);
        verif(`${nom} : le moteur démarre`, !ko0, ko0);
        let e = await lireEnr(pg);
        verif(`${nom} : l'indicateur d'enregistrement est visible, un seul, « enregistré » après le premier calcul`,
          e.visibles === 1 && e.etat === 'ok' && e.entier && /\d{2}:\d{2}/.test(e.texte) && !e.alerte, JSON.stringify(e));
        // le stockage devient plein : chaque écriture lève l'exception du navigateur
        await pg.evaluate(() => {
          window.__setItem = Storage.prototype.setItem;
          Storage.prototype.setItem = function () { throw new DOMException('Le quota est dépassé', 'QuotaExceededError'); };
        });
        verif(`${nom} : une saisie part au moteur`, await saisir(pg));
        e = await lireEnr(pg);
        verif(`${nom} : l'ÉCHEC se lit — « Non enregistré : stockage plein », visible, entier, non couvert, en rouge`,
          e.visibles === 1 && e.etat === 'echec' && e.texte === 'Non enregistré : stockage plein' && e.entier && !e.couvert.length && e.rouge && !e.deborde,
          JSON.stringify(e));
        verif(`${nom} : le message le dit, dans l'écran, avec « Exporter .stab »`,
          e.alerte && !e.alerte.info && e.alerte.dedans && /^Non enregistré : le stockage de ce navigateur est plein\./.test(e.alerte.texte)
          && e.alerte.bouton && e.alerte.bouton.texte === 'Exporter .stab' && e.alerte.bouton.h >= (tactile ? 44 : 24), JSON.stringify(e.alerte));
        await pg.screenshot({ path: path.join(OUT, `site_enregistrement_echec_${fmt.split(' ')[1]}.png`) });
        if (attendre) {
          // persistant : bien après les 7 s d'un message éphémère
          await pg.waitForTimeout(7600);
          const e2 = await lireEnr(pg);
          verif(`${nom} : 7,6 s plus tard, l'échec et le message sont toujours là`, e2.etat === 'echec' && e2.entier && e2.rouge && e2.alerte, JSON.stringify(e2));
        }
        // « Exporter .stab » : la note entière, rechargeable
        const f = await telechargement(pg, '#et-alerte .et-alerte-stab');
        let stab = null;
        try { stab = JSON.parse(f.texte); } catch (x) { /* dit plus bas */ }
        verif(`${nom} : « Exporter .stab » du message exporte la note (son état de calcul compris)`,
          /\.stab$/.test(f.nom) && stab && stab.format === 'stab' && stab.note && stab.note.module === 'poutre' && stab.note.etat && typeof stab.note.etat === 'object',
          `${f.nom} · ${f.texte.replace(/\s+/g, ' ').slice(0, 80)}`);
        // fermé, le message ne revient pas à la saisie suivante ; l'indicateur, lui, reste rouge
        await pg.click('#et-alerte [data-fermer-alerte]');
        await saisir(pg);
        e = await lireEnr(pg);
        verif(`${nom} : le message fermé ne revient pas à chaque saisie ; l'indicateur reste rouge`, !e.alerte && e.etat === 'echec' && e.entier && e.rouge, JSON.stringify(e));
        // le stockage revient : une écriture réussie efface l'échec
        await pg.evaluate(() => { Storage.prototype.setItem = window.__setItem; });
        await saisir(pg);
        e = await lireEnr(pg);
        verif(`${nom} : le stockage revenu, la saisie suivante s'enregistre et l'indicateur le dit`, e.etat === 'ok' && e.entier && !e.rouge && !e.alerte, JSON.stringify(e));
        verif(`${nom} : aucune erreur JavaScript`, !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
      } catch (x) { verif(`${nom} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      await pg.context().close();
    }

    // ------------------------------------------------ 2. la mémoire des notes abîmée, et sa copie impossible
    // (chaque copie « stab.notes.v1.illisible.* » est refusée par le navigateur ; les autres écritures passent)
    const refuserCopies = () => {
      const o = Storage.prototype.setItem;
      Storage.prototype.setItem = function (k, v) {
        if (String(k).startsWith('stab.notes.v1.illisible.')) throw new DOMException('Le quota est dépassé', 'QuotaExceededError');
        return o.call(this, k, v);
      };
    };
    for (const [fmt, w, h, tel] of [['PC', 1440, 900, false], ['téléphone', 390, 844, true]]) {
      const nom = `mémoire abîmée, copie impossible · ${fmt}`;
      const pg = await ouvrirContexte(w, h, tel, tel, refuserCopies);
      try {
        await pg.goto(`${BASE}/`);
        await pg.evaluate(([k, v]) => localStorage.setItem(k, v), [CLE, ABIME]);
        await pg.goto(`${BASE}/${MODS.poutre.page}`);
        const ko0 = await demarre(pg);
        verif(`${nom} : le moteur démarre`, !ko0, ko0);
        const e = await lireEnr(pg);
        const mem = await lireMemoire(pg);
        verif(`${nom} : RIEN n'est écrit — la mémoire abîmée reste à l'octet, aucune copie`, mem.principal === ABIME && !Object.keys(mem.copies).length,
          `${(mem.principal || '').length} caractères · ${Object.keys(mem.copies).length} copie(s)`);
        verif(`${nom} : l'écran le dit — « Non enregistré : mémoire illisible », en rouge, entier ; le message « Non enregistré : la mémoire des notes est illisible » avec « Exporter .stab »`,
          e.etat === 'echec' && e.texte === 'Non enregistré : mémoire illisible' && e.entier && !e.couvert.length && e.rouge
          && e.alerte && e.alerte.texte.startsWith('Non enregistré : la mémoire des notes est illisible.') && e.alerte.bouton && e.alerte.bouton.texte === 'Exporter .stab',
          JSON.stringify(e));
        await pg.screenshot({ path: path.join(OUT, `site_enregistrement_illisible_${tel ? 'tel' : 'pc'}.png`) });
        // l'accueil : il le signale et rend le texte brut, tel quel
        await pg.goto(`${BASE}/`);
        await pg.waitForTimeout(300);
        const a = await pg.evaluate(() => {
          const z = document.getElementById('et-memoire');
          const r = z.getBoundingClientRect();
          return { vu: z.checkVisibility({ visibilityProperty: true, opacityProperty: true }), texte: z.textContent, h: Math.round(r.height), haut: Math.round(r.top),
            date: document.getElementById('date').textContent, deborde: document.documentElement.scrollWidth > innerWidth };
        });
        verif(`${nom} : l'accueil signale la mémoire illisible (ses notes ne s'affichent pas, rien n'est effacé)`,
          a.vu && /mémoire des notes de ce navigateur est illisible/.test(a.texte) && /illisible/.test(a.date) && !a.deborde && a.haut < 400, JSON.stringify(a));
        const f = await telechargement(pg, '#et-memoire [data-texte-brut]');
        verif(`${nom} : l'accueil rend le texte de la mémoire abîmée, à l'octet`, f.texte === ABIME && /\.txt$/.test(f.nom), `${f.nom} · ${f.texte.length} caractères`);
        await pg.screenshot({ path: path.join(OUT, `site_accueil_memoire_illisible_${tel ? 'tel' : 'pc'}.png`) });
        verif(`${nom} : aucune erreur JavaScript`, !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
      } catch (x) { verif(`${nom} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      await pg.context().close();
    }

    // ------------------------------------------------ 3. la mémoire abîmée, copiée : la note s'écrit, le texte survit
    {
      const nom = 'mémoire abîmée, copiée';
      const pg = await ouvrirContexte(1440, 900, false, false);
      try {
        await pg.goto(`${BASE}/`);
        await pg.evaluate(([k, v]) => localStorage.setItem(k, v), [CLE, ABIME]);
        await pg.goto(`${BASE}/${MODS.poutre.page}`);
        const ko0 = await demarre(pg);
        verif(`${nom} : le moteur démarre`, !ko0, ko0);
        const e = await lireEnr(pg);
        const mem = await lireMemoire(pg);
        const copies = Object.entries(mem.copies);
        let principal = null;
        try { principal = JSON.parse(mem.principal); } catch (x) { /* dit plus bas */ }
        verif(`${nom} : le texte d'origine est copié à part, à l'octet, une fois ; la note s'écrit dans une mémoire neuve`,
          copies.length === 1 && copies[0][1] === ABIME && principal && Object.keys(principal).length === 1, `${copies.length} copie(s) · ${mem.principal && mem.principal.slice(0, 60)}`);
        verif(`${nom} : l'écran dit « enregistré », et une fois que la mémoire était illisible et que son texte est gardé`,
          e.etat === 'ok' && e.alerte && e.alerte.info && /était illisible/.test(e.alerte.texte) && !e.alerte.bouton, JSON.stringify(e));
        await saisir(pg);
        const mem2 = await lireMemoire(pg);
        verif(`${nom} : la saisie suivante ne copie rien de plus`, Object.keys(mem2.copies).length === 1, Object.keys(mem2.copies).join(', '));
        await pg.close();
        verif(`${nom} : aucune erreur JavaScript`, !pg.erreurs.length, pg.erreurs.join(' | ').slice(0, 300));
        // l'accueil, aux trois formats : la copie est signalée et se télécharge
        for (const [fmt, w, h, tel] of [['PC', 1440, 900, false], ['tablette', 820, 1180, false], ['téléphone', 390, 844, true]]) {
          const c2 = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: tel, hasTouch: fmt !== 'PC', acceptDownloads: true,
            storageState: await pg.context().storageState() });
          const g = await c2.newPage();
          await g.goto(`${BASE}/`);
          await g.waitForTimeout(300);
          const a = await g.evaluate(() => {
            const z = document.getElementById('et-memoire'), bt = z.querySelector('[data-copie]');
            const r = z.getBoundingClientRect(), rb = bt ? bt.getBoundingClientRect() : null;
            return { vu: z.checkVisibility({ visibilityProperty: true, opacityProperty: true }), texte: z.textContent.replace(/\s+/g, ' ').slice(0, 160),
              haut: Math.round(r.top), dedans: r.left >= 0 && r.right <= innerWidth, bouton: rb ? Math.round(rb.height) : 0, notes: document.querySelectorAll('#recentes .rcard').length,
              deborde: document.documentElement.scrollWidth > innerWidth };
          });
          verif(`${nom} · accueil ${fmt} : la copie gardée à part est signalée, dans l'écran, son bouton à la taille du ${tel ? 'doigt' : 'pointeur'}`,
            a.vu && /gardée à part/.test(a.texte) && a.dedans && !a.deborde && a.haut < 400 && a.bouton >= (tel ? 44 : 24) && a.notes === 1, JSON.stringify(a));
          const f = await telechargement(g, '#et-memoire [data-copie]');
          verif(`${nom} · accueil ${fmt} : la copie se télécharge, à l'octet`, f.texte === ABIME, `${f.nom} · ${f.texte.length} caractères`);
          await g.screenshot({ path: path.join(OUT, `site_accueil_memoire_copie_${fmt === 'téléphone' ? 'tel' : fmt.toLowerCase()}.png`) });
          await c2.close();
        }
      } catch (x) { verif(`${nom} : le parcours va au bout`, false, String(x.message).split('\n')[0]); }
      await pg.context().close();
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
  // (les formats, « télécharger », « météo », la sonde et l'enregistrement travaillent dans leurs propres navigateurs : leurs notes ne sont pas ici)
  if (!FILTRE.startsWith('formats') && !['telecharger', 'meteo', 'sonde', 'enregistrement'].includes(FILTRE)) verif('accueil : les notes créées sont à reprendre', n >= 1, `${n} note(s)`);
  // une vignette est un dessin : un tableau (vue HTML d'un module de consultation) y serait coupé et illisible
  const vignettes = await p.$$eval('#recentes .rcard', (cs) => cs.map((c) => ({ page: c.getAttribute('href').split('?')[0],
    v: c.querySelector('.thumb').firstElementChild ? c.querySelector('.thumb').firstElementChild.tagName.toLowerCase() : '' })));
  verif('accueil : chaque vignette est un dessin ou reste vide, jamais un tableau', vignettes.every((x) => x.v === '' || x.v === 'svg'), JSON.stringify(vignettes));
  verif('accueil : la note d\'un module qui a un dessin le montre en vignette', vignettes.every((x) => !avecDessin.has(x.page) || x.v === 'svg'), JSON.stringify(vignettes));
  if (!FILTRE || ['garde', 'poutre'].includes(FILTRE)) verif('accueil : le chantier « Réalex » est listé', (await p.textContent('#chantiers')).includes('Réalex'));
  m = await mesure();
  verif('accueil : rien ne déborde, aucun nom coupé', !m.deborde && !m.coupes.length, m.coupes.join(', '));
  await capture('accueil');
  verif('aucune erreur JavaScript', erreurs.length === 0, erreurs.join(' | ').slice(0, 400));
  await b.close();
  process.exit(ko ? 1 : 0);
})();
