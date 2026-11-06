// Captures et MESURE de l'écran « Garde-corps » (version 3.2, disposition sans onglets) aux largeurs PC 1440 / 1280, tablette 1024,
// téléphone 390 — même protocole que tools/captures_soudures.js : on mesure CE QUE L'ŒIL VOIT.
//
//   node tools/captures_garde_corps.js <dossier_sortie> [états] [largeurs] [port]
//   ex. node tools/captures_garde_corps.js scratchpad/gc defaut,lisses,murale,biblio,nok 1440,1024,390 8597
//
// L'écran : à gauche l'ÉLÉVATION, UN dessin de détail choisi par le sélecteur « gc_vue » (coupe A-A, plan, platine, fixation, section), la
// carte RÉSULTAT avec une ligne par élément (sa section à choisir sur place, le bouton « ★ » qui nomme l'optimal et ouvre en popover le
// tableau des profilés) ; à droite les cartes DÉFINITION, CHARGES, FIXATION ; en bas le détail replié.
// États (chaque état PART de la page vierge, le garde-corps par défaut) :
//   « defaut »  la page telle qu'elle s'ouvre : ELS, barreaudage vertical, droit 3,60 m à 0,90 m, RHS 50x30x3, RÉSULTAT 64 % ;
//   « lisses »  le type « Lisses horizontales » choisi dans le menu déroulant, forme en L : lisses intermédiaires, montant d'angle ;
//   « murale »  le type « Main courante murale sur consoles » : consoles C…, coupe au mur ;
//   « biblio »  le popover « ★ » du montant ouvert, CLIC sur la troisième ligne du tableau : la section cliquée est retenue — le sélecteur
//               de section doit AFFICHER cette valeur (liaison état → widget) ; la capture montre le tableau ouvert ;
//   « nok »     catégorie C5 (3 kN/m) : Non conforme ;
//   « laterale » fixation « platine latérale en nez de dalle » : les champs d_h, b_p, l_p, e_x, e_y, la vue « Platine latérale (vue de
//               face) » choisie au sélecteur, l'arrachement P dans le résumé de la fixation ;
//   « loggia »  les deux bouts « lisse ancrée dans le mur » et un entraxe maxi de 4000 : garde-corps SANS montant, la main courante porte
//               de mur à mur (ni platine ni fixation au sélecteur) ;
//   « rampant » pente 32° : élévation inclinée, « rampant, pente 32° » dans la phrase ;
//   « plan »    la vue en plan choisie au sélecteur.
// Mesures : aucune exception ; aucune erreur JavaScript ; aucun débordement horizontal ; élévation SVG visible (≥ 260 px de large,
// ≥ 120 px de haut) ; le dessin de détail visible (≥ 200 × 120 px) ; exactement deux dessins SVG visibles (le reste vit dans des popovers
// fermés) ; carte RÉSULTAT visible avec η ; les lignes de la carte ne se chevauchent pas (chaque élément de la carte finit avant que le
// suivant commence) ; aucun onglet ; états spécifiques. Fichiers : apres_garde_corps_<état>_<largeur>.png. Code de sortie 1 dès qu'un
// seuil n'est pas tenu.
const { chromium } = require('playwright');
const OUT = process.argv[2] || '.';
const ETATS = (process.argv[3] || 'defaut,lisses,murale,biblio,nok,laterale,loggia,rampant,plan').split(',');
const LARGEURS = (process.argv[4] || '1440,1280,1024,390').split(',').map(Number);
const PORT = process.argv[5] || '8597';
const BASE = `http://localhost:${PORT}/?page=Garde-corps`;
let KO = 0;
let biblio = { avant: null, apres: null, eta: '' };

async function settle(page, ms = 700) {
  await page.waitForTimeout(300);
  try { await page.waitForSelector('[data-testid="stStatusWidget"]', { state: 'detached', timeout: 60000 }); } catch (e) {}
  await page.waitForTimeout(ms);
}
async function cliquerSegment(page, cle, texte) {
  const b = page.locator(`.st-key-${cle} button:visible`, { hasText: texte }).first();   // Streamlit rend aussi un clone caché (mesure de largeur)
  await b.scrollIntoViewIfNeeded(); await b.click();
  await settle(page, 2500);
}
async function choisir(page, cle, texte) {
  // selectbox de Streamlit 1.66 : une combobox react-aria — on tape le texte, puis on prend l'option
  const inp = page.locator(`.st-key-${cle} input`).first();
  await inp.scrollIntoViewIfNeeded(); await inp.click(); await inp.fill(texte);
  await page.waitForTimeout(400);
  const opt = page.locator('[role="option"]', { hasText: texte }).first();
  if (await opt.count()) { await opt.click(); } else { await inp.press('Enter'); }
  await settle(page, 2500);
}
async function valeurSelect(page, cle) {
  return page.locator(`.st-key-${cle} input`).first().inputValue();
}
async function saisir(page, cle, valeur) {
  // number_input de Streamlit : on remplace le texte puis Entrée (le widget n'envoie sa valeur qu'à la validation)
  const inp = page.locator(`.st-key-${cle} input`).first();
  await inp.scrollIntoViewIfNeeded(); await inp.click({ clickCount: 3 }); await inp.fill(String(valeur)); await inp.press('Enter');
  await settle(page, 2500);
}
async function preparer(page, etat) {
  if (etat === 'lisses') {
    await choisir(page, 'gc_w_type', 'Lisses horizontales');
    await cliquerSegment(page, 'gc_w_geometrie_forme', 'en L');
  }
  if (etat === 'murale') await choisir(page, 'gc_w_type', 'murale');
  if (etat === 'biblio') {
    biblio.avant = await valeurSelect(page, 'gc_w_montant_section');
    // le bouton « ★ » du montant ouvre le popover ; le tableau des profilés est dedans
    const btn = page.locator('.st-key-gc_pop_montant [data-testid="stPopoverButton"]:visible').first();
    await btn.scrollIntoViewIfNeeded(); await btn.click(); await settle(page, 1500);
    const grid = page.locator('[data-testid="stPopoverBody"] [data-testid="stDataFrame"]').first();
    await grid.waitFor({ state: 'visible', timeout: 15000 });
    try { await grid.scrollIntoViewIfNeeded({ timeout: 5000 }); } catch (e) {}   // au téléphone le corps du popover défile lui-même
    const box = await grid.boundingBox();
    // la troisième ligne de données du tableau du montant : colonne des coches, en-tête 35 px + deux lignes de 35 px
    await page.mouse.click(box.x + 14, box.y + 35 + 2 * 35 + 17);
    await settle(page, 3000);
    biblio.apres = await valeurSelect(page, 'gc_w_montant_section');
    return;   // le popover reste ouvert sur la capture : on ne remonte pas la page
  }
  if (etat === 'nok') await choisir(page, 'gc_w_charges_categorie', 'C5');
  if (etat === 'laterale') { await cliquerSegment(page, 'gc_w_fixation_type', 'latérale'); await cliquerSegment(page, 'gc_vue', 'Platine'); }
  if (etat === 'loggia') {
    await choisir(page, 'gc_w_geometrie_debut', 'lisse ancrée dans le mur');
    await choisir(page, 'gc_w_geometrie_fin', 'lisse ancrée dans le mur');
    await saisir(page, 'gc_w_geometrie_troncons_0_s_max', 4000);
  }
  if (etat === 'rampant') await saisir(page, 'gc_w_geometrie_pente', 32);
  if (etat === 'plan') await cliquerSegment(page, 'gc_vue', 'Plan');
  await page.evaluate(() => { window.scrollTo(0, 0); for (const e of document.querySelectorAll('*')) { if (e.scrollTop) e.scrollTop = 0; } });
  await settle(page, 800);
}
function verdict(nom, ok, info) {
  if (!ok) KO++;
  console.log(`   ${ok ? 'OK ' : 'KO '} ${nom}${info ? '   [' + info + ']' : ''}`);
}
async function agrandir(page, w) {
  const h = await page.evaluate(() => {
    const m = document.querySelector('[data-testid="stMainBlockContainer"]') || document.body;
    return Math.ceil(m.getBoundingClientRect().height + 80);
  });
  await page.setViewportSize({ width: w, height: Math.min(Math.max(h, 800), 7000) });
  await settle(page, 800);
}

(async () => {
  const browser = await chromium.launch();
  for (const etat of ETATS) {
    for (const w of LARGEURS) {
      const mobile = w <= 640;
      const ctx = await browser.newContext({ viewport: { width: w, height: mobile ? 844 : 1000 },
        deviceScaleFactor: mobile ? 2 : 1, isMobile: mobile, hasTouch: mobile });
      const page = await ctx.newPage();
      const erreurs = [];
      page.on('pageerror', e => erreurs.push(e.message));
      console.log(`=== garde-corps (${etat}) @ ${w} px`);
      try {
        await page.goto(BASE, { waitUntil: 'networkidle' });
        await page.waitForSelector('text=Dimensionnement d\'un garde-corps', { timeout: 90000 }); await settle(page, 3000);
        await preparer(page, etat);
        const m = await page.evaluate(() => {
          const vis = e => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          // nos dessins portent un id « gc_… » (les icônes de Streamlit sont aussi des SVG : on ne les compte pas)
          const svgs = [...document.querySelectorAll('svg[id^="gc_"]')].filter(s => s.getAttribute('viewBox') && vis(s));
          const dessin = svgs.find(s => s.closest('.st-key-gc_dessin'));
          const detail = svgs.find(s => s.closest('.st-key-gc_dessin2'));
          const res = document.querySelector('.gc-res');
          const pill = document.querySelector('.gc-pill');
          const eta = document.querySelector('.gc-eta');
          const texte = (document.querySelector('[data-testid="stMainBlockContainer"]') || document.body).innerText;
          // les enfants directs de la carte RÉSULTAT, dans l'ordre : chacun doit finir avant que le suivant commence (pas de chevauchement)
          const carte = document.querySelector('.st-key-gc_k_res [data-testid="stVerticalBlock"]');
          let chevauche = 0;
          if (carte) {
            const boites = [...carte.children].filter(vis).map(c => c.getBoundingClientRect());
            for (let i = 1; i < boites.length; i++) { if (boites[i].top < boites[i - 1].bottom - 1) chevauche++; }
          }
          const vues = [...document.querySelectorAll('.st-key-gc_vue button')].map(b => b.innerText.trim()).filter(t => t);
          return { innerWidth: window.innerWidth, scrollW: document.documentElement.scrollWidth,
                   dessinW: dessin ? Math.round(dessin.getBoundingClientRect().width) : 0,
                   dessinH: dessin ? Math.round(dessin.getBoundingClientRect().height) : 0,
                   detailW: detail ? Math.round(detail.getBoundingClientRect().width) : 0,
                   detailH: detail ? Math.round(detail.getBoundingClientRect().height) : 0,
                   nSvg: svgs.length, nOnglets: document.querySelectorAll('[data-testid="stTab"]').length, vues, chevauche,
                   resH: res && vis(res) ? Math.round(res.getBoundingClientRect().height) : 0,
                   nLignes: [...document.querySelectorAll('.st-key-gc_k_res .gc-row')].filter(vis).length,
                   nSelecteurs: [...document.querySelectorAll('.st-key-gc_k_res [data-testid="stSelectbox"]')].filter(vis).length,
                   nEtoiles: [...document.querySelectorAll('.st-key-gc_k_res [data-testid="stPopoverButton"]')].filter(b => vis(b) && /★/.test(b.innerText)).length,
                   pill: pill ? pill.innerText : '', eta: eta ? eta.innerText : '',
                   angle: /d'angle/.test(texte), console: /Console C/.test(texte), enL: /en L/.test(texte),
                   lissesInter: /Lisses intermédiaires/.test(texte), murale: /murale sur consoles/i.test(texte),
                   laterale: /PLATINE LATÉRALE \(VUE DE FACE\)/.test(texte) && /arrachement P/.test(texte),
                   champsLat: ['d_h', 'b_p', 'l_p', 'e_x', 'e_y'].every(k => [...document.querySelectorAll('[data-testid="stNumberInput"] label')]
                                                                            .some(l => l.innerText.startsWith(k + ' —'))),
                   sansMontant: /Sans montant/.test(texte) && /porte de mur à mur/.test(texte) && !/PLATINE EN PLAN/.test(texte),
                   rampant: /rampant, pente 32°/.test(texte), plan: /VUE EN PLAN/.test(texte) && !/COUPE A-A/.test(texte),
                   popover: !!document.querySelector('[data-testid="stPopoverBody"] [data-testid="stDataFrame"]'),
                   exc: [...document.querySelectorAll('[data-testid="stException"]')].map(e => e.innerText.slice(0, 200)) };
        });
        verdict('aucune exception Streamlit', m.exc.length === 0, m.exc.join(' | '));
        verdict('aucune erreur JavaScript', erreurs.length === 0, erreurs.join(' | ').slice(0, 200));
        verdict('aucun débordement horizontal', m.scrollW <= m.innerWidth, `${m.scrollW}/${m.innerWidth}`);
        verdict('élévation visible (≥ 260 × 120 px)', m.dessinW >= 260 && m.dessinH >= 120, `${m.dessinW} × ${m.dessinH} px`);
        verdict('le dessin de détail visible (≥ 200 × 120 px)', m.detailW >= 200 && m.detailH >= 120, `${m.detailW} × ${m.detailH} px`);
        verdict('exactement deux dessins SVG visibles (élévation et détail ; le reste vit dans des popovers fermés)', m.nSvg === 2, `${m.nSvg}`);
        verdict('aucun onglet ; le sélecteur de vue offre ' + (etat === 'loggia' ? 'coupe, plan, section' : 'coupe, plan, platine, fixation, section'),
                m.nOnglets === 0 && m.vues.join('|') === (etat === 'loggia' ? 'Coupe A-A|Plan|Section' : 'Coupe A-A|Plan|Platine|Fixation|Section'), m.vues.join('|'));
        verdict('carte RÉSULTAT visible avec η', m.resH >= 80 && /η = \d+ %/.test(m.eta), `${m.resH} px · ${m.eta}`);
        verdict('la carte RÉSULTAT : une ligne par élément, un sélecteur de section et un bouton « ★ » par élément dimensionné, sans chevauchement',
                m.nLignes >= 2 && m.nSelecteurs >= 2 && m.nEtoiles === m.nSelecteurs && m.chevauche === 0,
                `${m.nLignes} lignes · ${m.nSelecteurs} sélecteurs · ${m.nEtoiles} ★ · ${m.chevauche} chevauchement(s)`);
        if (etat === 'laterale') verdict('latérale : les champs d_h, b_p, l_p, e_x, e_y ; « Platine latérale (vue de face) » au sélecteur, arrachement P',
                                         m.laterale && m.champsLat, `${m.eta}`);
        if (etat === 'loggia') verdict('loggia : « Sans montant », la main courante porte de mur à mur, pas de platine en plan', m.sansMontant, `${m.eta}`);
        if (etat === 'rampant') verdict('rampant : « rampant, pente 32° »', m.rampant, `${m.eta}`);
        if (etat === 'plan') verdict('plan : la vue en plan remplace la coupe A-A', m.plan, `${m.vues.join('|')}`);
        if (etat === 'defaut') verdict('défaut : Conforme, η = 64 %', /Conforme/.test(m.pill) && /64 %/.test(m.eta), `${m.pill} · ${m.eta}`);
        if (etat === 'lisses') verdict('lisses : lisses intermédiaires, en L, un montant d\'angle', m.lissesInter && m.enL && m.angle, `${m.eta}`);
        if (etat === 'murale') verdict('murale : des consoles C…', m.console && m.murale, `${m.eta}`);
        if (etat === 'biblio') verdict('biblio : le clic sur une ligne du tableau (dans le popover) retient une autre section, le sélecteur l\'AFFICHE (liaison état → widget), le popover reste ouvert',
                                       biblio.avant && biblio.apres && biblio.apres !== biblio.avant && m.popover, `avant ${biblio.avant} · après ${biblio.apres} · popover ${m.popover}`);
        if (etat === 'nok') verdict('nok : Non conforme (C5, 3 kN/m)', /Non conforme/.test(m.pill), `${m.pill} · ${m.eta}`);
        if (etat !== 'biblio') await agrandir(page, w);
        await page.screenshot({ path: `${OUT}/apres_garde_corps_${etat}_${w}.png`, fullPage: etat !== 'biblio' });
      } catch (e) {
        verdict('déroulement sans erreur', false, e.message.slice(0, 200));
        try { await page.screenshot({ path: `${OUT}/apres_garde_corps_${etat}_${w}_ERREUR.png`, fullPage: true }); } catch (e2) {}
      }
      await ctx.close();
    }
  }
  await browser.close();
  console.log(KO ? `\n${KO} contrôle(s) en défaut` : '\nTous les contrôles tenus');
  process.exit(KO ? 1 : 0);
})();
