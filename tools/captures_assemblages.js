// Captures « après » et MESURE des écrans d'assemblage aux quatre largeurs (charte ergonomique du
// 31/10/2026, docs/assemblages/ERGONOMIE.md §2) — issu de tel_carte.js + mesure_proto.js.
//
//   node tools/captures_assemblages.js <dossier_sortie> [modules] [largeurs] [port] [état]
//   ex. node tools/captures_assemblages.js scratchpad/ergo/apres plat_ame 1440,1280,1024,390 8597
//   [état] (consolidation du 01/11/2026) : « complet » (pied de poteau en vérification complète), « fixe »
//   (dessin interactif décoché : panneau Cotes), « simple » (Vue simple), « perso » (profil ou tube sans
//   dimensions), « avances » (Paramètres avancés ouverts) ; consolidation du 02/11/2026, famille
//   POUTRE – POTEAU : « ame » (cas d'appui « Âme du poteau »), « ameok » (le cas (b) RÉGLÉ de la platine
//   d'about : bp 190 et p3 110 par le panneau Cotes), « tying » (NEd = 250 kN en force de chaînage) —
//   combinables par « + » ; fichiers apres_<module>_<état>_<largeur>.png.
//
// Mesure CE QUE L'ŒIL VOIT (getBoundingClientRect + checkVisibility), jamais ce que le DOM garde :
//   - scrollWidth == innerWidth (aucun débordement horizontal) ;
//   - cases des lignes d'objet (<p>lg_*) ≥ 120 px ; 4 par rangée ≥ 1181 px, 2 en dessous ;
//   - titres de bloc (<p>bt_*) dans leur rangée ;
//   - barre d'outils (<p>barre) : 1 rangée ≥ 1024, ≤ 3 rangées et ≤ 130 px à 390 ;
//   - bandeau et premier dessin dans les 844 premiers px après le titre à 390 ;
//   - onglets (<p>ui_onglet) sans débordement ;
//   - consolidation du 01/11/2026 : texte visible NON TRONQUÉ dans toute case des lignes d'objet, des
//     continuations, des rangées de table et de l'en-tête (input.scrollWidth ≤ clientWidth : « Tube RHS / S »
//     passait inaperçu, le script ne mesurait que les lignes lg_) ; titres de bloc sans débordement (un mot de
//     11 lettres dans 76 px) ; sous 1351 px, un sous-titre « ↳ » plus proche de SA rangée que de la précédente ;
//     tables serrées (tab-serre) tenues dans leur colonne ; une case d'une ligne de ≤ 2 cases ≤ 187 px.
// La fenêtre est agrandie à la hauteur du contenu avant la capture (Streamlit défile dans son
// conteneur : fullPage ne suffit pas). Fichiers : apres_<module>_<largeur>.png. Code de sortie 1
// dès qu'un seuil n'est pas tenu.
const { chromium } = require('playwright');
const OUT = process.argv[2] || '.';
const MODULES = (process.argv[3] || 'plat_ame').split(',');
const LARGEURS = (process.argv[4] || '1440,1280,1024,390').split(',').map(Number);
const PORT = process.argv[5] || '8597';
const ETAT = (process.argv[6] || '').split('+').filter(Boolean);
const SUFFIXE = ETAT.length ? '_' + ETAT.join('_') : '';
const BASE = `http://localhost:${PORT}/?page=Assemblages%20m%C3%A9talliques`;
const PREFIXE = { doubles_cornieres: 'asm_', plat_ame: 'fpl_', platine_about: 'pab_', platine_beton: 'ppb_',
                  plat_ame_poteau: 'fpc_', doubles_cornieres_poteau: 'dcc_', platine_about_poteau: 'pac_' };
let KO = 0;

async function settle(page, ms = 700) {
  await page.waitForTimeout(300);
  try { await page.waitForSelector('[data-testid="stStatusWidget"]', { state: 'detached', timeout: 40000 }); } catch (e) {}
  await page.waitForTimeout(ms);
}
async function cliquerTexte(page, selecteur, texte) {
  await page.evaluate(([sel, t]) => {
    // la DERNIÈRE ligne du texte : un résumé d'expander porte d'abord le nom de son icône
    // (« keyboard_arrow_right »), et l'égalité stricte ne l'ouvrait pas (état « avances » capturé replié)
    const b = [...document.querySelectorAll(sel)].find(x => x.innerText.trim().split('\n').pop().trim() === t
                                                          && x.getBoundingClientRect().width > 0);
    if (b) b.click();
  }, [selecteur, texte]);
  await settle(page, 2500);
}
async function choisir(page, cle, option) {
  const ouvrir = page.locator(`.st-key-${cle} button[aria-label="Open"]`).first();
  await ouvrir.scrollIntoViewIfNeeded(); await ouvrir.click(); await page.waitForTimeout(500);
  await page.locator('[role="option"]', { hasText: option }).first().click();
  await settle(page, 2500);
}
async function preparer(page, mod, p) {
  for (const e of ETAT) {
    if (e === 'complet') await cliquerTexte(page, `.st-key-${p}mode button`, 'Vérif. complète');
    if (e === 'fixe') {
      await page.evaluate(k => { const c = document.querySelector(`.st-key-${k} input[type="checkbox"]`); if (c && c.checked) c.click(); }, `${p}ui_composant`);
      await settle(page, 2500);
    }
    if (e === 'simple') await cliquerTexte(page, `.st-key-${p}ui_niveau label`, 'Vue simple');
    if (e === 'avances') await cliquerTexte(page, 'summary', 'Paramètres avancés');
    if (e === 'perso') {
      if (mod === 'platine_beton') await choisir(page, `${p}forme`, 'Tube RHS / SHS');
      else await choisir(page, `${p}prof_P`, 'Personnalisé');
    }
    // ---- famille POUTRE – POTEAU (lot 5) : les deux cas d'appui et le tying
    // « ame » : le cas d'appui « Âme du poteau » (segmented_control de l'en-tête,
    // options affichées « Aile » / « Âme »).
    if (e === 'ame') {
      await cliquerTexte(page, `.st-key-${p}appui button`, 'Âme');
      // en cas (b), une L100 ne passe pas entre les congés d'un HEB 300
      // (4,3 + 100 = 104,3 > 104) : la L90x90x9 passe, et l'écran se mesure
      // sur une configuration VALIDE
      if (mod === 'doubles_cornieres_poteau') await choisir(page, `${p}corn_u`, 'L90x90x9');
    }
    // « ameok » : cas (b) RÉGLÉ pour la platine d'about — la platine 290 × 220 du
    // cas (a) ne tient pas entre les congés (198 mm). bp et p3 sont des cotes du
    // DESSIN (règle P6 de la charte) : on passe par le panneau « Cotes » (dessin
    // interactif décoché), puis on recoche.
    if (e === 'ameok') {
      await cliquerTexte(page, `.st-key-${p}appui button`, 'Âme');
      await page.evaluate(k => { const c = document.querySelector(`.st-key-${k} input[type="checkbox"]`); if (c && c.checked) c.click(); }, `${p}ui_composant`);
      await settle(page, 2500);
      for (const [cle, val] of [[`${p}cote_bp_u`, '190'], [`${p}cote_p3_u`, '110']]) {
        const inp = page.locator(`.st-key-${cle} input`).first();
        await inp.scrollIntoViewIfNeeded(); await inp.fill(val); await inp.press('Enter');
        await settle(page, 2500);
      }
      await page.evaluate(k => { const c = document.querySelector(`.st-key-${k} input[type="checkbox"]`); if (c && !c.checked) c.click(); }, `${p}ui_composant`);
      await settle(page, 2500);
    }
    // « tying » : NEd de traction en force de chaînage — la ligne EFFORTS ELU
    // passe à 4 cases (VEd · NEd · MEd · Nature de NEd) et γMu paraît aux avancés
    if (e === 'tying') {
      const inp = page.locator(`.st-key-${p}N_Ed input`).first();
      await inp.scrollIntoViewIfNeeded(); await inp.fill('250'); await inp.press('Enter');
      await settle(page, 2500);
      await choisir(page, `${p}nat_N`, 'Tying');
    }
  }
  // un état qui a défilé laisse la page au milieu : on REMONTE avant de mesurer.
  // Streamlit défile dans SON conteneur, pas dans la fenêtre : les deux.
  if (ETAT.length) {
    await page.evaluate(() => {
      window.scrollTo(0, 0);
      for (const e of document.querySelectorAll('*')) { if (e.scrollTop) e.scrollTop = 0; }
    });
    await settle(page, 800);
  }
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
      console.log(`=== ${mod}${SUFFIXE ? ' (' + ETAT.join(' + ') + ')' : ''} @ ${w} px`);
      try {
        await page.goto(BASE, { waitUntil: 'networkidle' });
        await page.waitForSelector('text=Premier assemblage du projet', { timeout: 90000 }); await settle(page, 800);
        await jsClick(page, `asm_btn_open_${mod}`);
        await page.waitForTimeout(3000); await settle(page, 2500);
        await preparer(page, mod, p);
        const m = await page.evaluate((p) => {
          const rect = e => { const r = e.getBoundingClientRect(); return { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height), b: Math.round(r.bottom) }; };
          const vis = e => e.checkVisibility({ visibilityProperty: true, opacityProperty: true });
          // rangées : les ordonnées de BAS groupées à 12 px près. `last = -99` rendait
          // « 0 rangée » pour une ligne passée AU-DESSUS du viewport après défilement
          // (ordonnées négatives : -200 - (-99) = -101, jamais > 12) — faux rouge
          // diagnostiqué deux fois. La PREMIÈRE ordonnée ouvre toujours une rangée,
          // quel que soit son signe.
          const rows = rs => { const bs = [...new Set(rs.map(r => r.b))].sort((a, b) => a - b); let n = 0, last = null; for (const b of bs) { if (last === null || b - last > 12) { n++; last = b; } } return n; };
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
          // texte tronqué : la valeur d'une case (combobox React-Aria ou nombre) plus large que sa zone visible
          const tronques = [];
          for (const zone of document.querySelectorAll(`[class*="st-key-${p}lg_"], [class*="st-key-${p}ls_"], [class*="st-key-${p}tr_"], .st-key-${p}en_tete`)) {
            if (!vis(zone)) continue;
            for (const inp of zone.querySelectorAll('input')) {
              if (!vis(inp) || inp.type === 'checkbox') continue;
              if (inp.scrollWidth > inp.clientWidth + 1) {
                const k = [...inp.closest('[class*="st-key-"]').classList].find(c => c.startsWith('st-key-'));
                tronques.push(`${k.replace('st-key-' + p, '')} « ${inp.value} » ${inp.scrollWidth}/${inp.clientWidth}`);
              }
            }
          }
          out.tronques = [...new Set(tronques)];
          // titres de bloc : aucun mot qui déborde de sa case (scrollWidth du texte ≤ sa largeur)
          out.titresDebordants = [...document.querySelectorAll(`[class*="st-key-${p}bt_"] [data-testid="stMarkdownContainer"] > div`)]
            .filter(e => vis(e) && e.scrollWidth > e.clientWidth + 1).map(e => `${e.innerText.trim()} ${e.scrollWidth}/${e.clientWidth}`);
          // sous-titres « ↳ » (sous 1351 px, au-dessus de leur rangée) : écart au-dessus et au-dessous
          // l'écart se mesure sur l'ENCRE : la boîte du texte (Range), pas celle du conteneur (marges, padding)
          const boiteTexte = el => { const r = document.createRange(); r.selectNodeContents(el); return r.getBoundingClientRect(); };
          out.sousTitres = [];
          for (const ls of document.querySelectorAll(`[class*="st-key-${p}ls_"]`)) {
            const t = ls.querySelector(`[class*="st-key-${p}bt_"] [data-testid="stMarkdownContainer"] > div`);
            const lab = [...ls.querySelectorAll('[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"], button')].find(vis);
            const enveloppe = ls.closest('[data-testid="stLayoutWrapper"]');
            let prev = enveloppe ? enveloppe.previousElementSibling : null;
            while (prev && !vis(prev)) prev = prev.previousElementSibling;
            if (!t || !vis(t) || !lab || !prev) continue;
            const rt = boiteTexte(t), rl = boiteTexte(lab);
            const cases = [...prev.querySelectorAll('input, button, [data-testid="stSelectbox"] [role="group"], p')].filter(vis);
            const basPrec = cases.length ? Math.max(...cases.map(c => c.getBoundingClientRect().bottom)) : prev.getBoundingClientRect().bottom;
            if (rl.top < rt.bottom - 1) continue;                     // sous-titre à gauche (≥ 1351 px) : sans objet
            out.sousTitres.push({ txt: t.innerText.trim().slice(0, 22), dessus: Math.round(rt.top - basPrec), dessous: Math.round(rl.top - rt.bottom) });
          }
          // tables serrées : tenues dans leur colonne, sans défilement
          out.tablesSerrees = [...document.querySelectorAll('.tab-serre')].filter(vis)
            .map(d => ({ table: Math.round(d.querySelector('table').getBoundingClientRect().width), boite: Math.round(d.clientWidth) }));
          const bar = document.querySelector(`.st-key-${p}barre`);
          if (bar) {
            const items = [...bar.querySelectorAll('button, [data-testid="stSelectbox"]')].filter(vis).map(rect);
            out.barre = { rangees: rows(items), hauteur: rect(bar).h, items: items.length };
          }
          const seg = document.querySelector(`.st-key-${p}ui_onglet`);
          if (seg) out.onglets = { right: Math.round(seg.getBoundingClientRect().right), rangees: rows([...seg.querySelectorAll('button')].filter(vis).map(rect)) };
          const h2 = document.querySelector('h2');
          const bandeau = [...document.querySelectorAll('[data-testid="stMarkdown"]')].find(e => /VÉRIFIÉ|Données incomplètes/.test(e.innerText) && vis(e));
          const dessin = document.querySelector(`.st-key-${p}col_dessin iframe, .st-key-${p}col_dessin svg`);
          const y0 = h2 ? rect(h2).y : 0;
          out.premierEcran = { bandeau: bandeau ? rect(bandeau).b - y0 : null, dessin: dessin ? rect(dessin).y - y0 : null };
          // débordement : ce qui SORT de l'écran. Un élément logé dans une boîte qui
          // DÉFILE horizontalement (tableau_html(defilant=), point 9 de la consolidation
          // du 01/11/2026 : « les tables serrées défilent au lieu de déborder ») n'en
          // sort pas — on le compte à part, il se lit en balayant. Sans cette distinction
          // la table « Poteau porteur » des cornières en tying rougissait à tort à 390 px.
          const scroller = e => {
            for (let p = e.parentElement; p; p = p.parentElement) {
              const ox = getComputedStyle(p).overflowX;
              if ((ox === 'auto' || ox === 'scroll') && p.scrollWidth > p.clientWidth + 1) return p;
            }
            return null;
          };
          const sortis = [...document.querySelectorAll('[data-testid="stMainBlockContainer"] *')]
            .filter(e => vis(e) && e.getBoundingClientRect().right > window.innerWidth + 1);
          out.debordants = sortis.filter(e => !scroller(e)).length;
          out.defilantes = [...new Set(sortis.filter(e => scroller(e)).map(e => {
            const p = scroller(e);
            return `${Math.round(p.scrollWidth)}/${Math.round(p.clientWidth)} px`;
          }))];
          return out;
        }, p);
        verdict('aucun débordement horizontal (scrollWidth == innerWidth)', m.scrollW === m.innerWidth && m.debordants === 0, `${m.scrollW} / ${m.innerWidth}, ${m.debordants} débordant(s)`);
        if (m.defilantes && m.defilantes.length) console.log('   boîtes qui DÉFILENT (contenu plus large que la boîte, lisible en balayant) :', m.defilantes.join(' · '));
        const petites = m.lignes.filter(l => l.largeurs.some(x => x < 120));
        verdict('cases de nombre ≥ 120 px', petites.length === 0, petites.map(l => l.key + ':' + l.largeurs).join(' '));
        const attendu = w >= 1181 ? 4 : 2;
        const mauvaises = m.lignes.filter(l => l.n === 4 && l.rangees !== 4 / attendu);
        verdict(`lignes de 4 : ${attendu} cases par rangée`, mauvaises.length === 0, mauvaises.map(l => l.key + ':' + l.rangees + ' rangées').join(' '));
        const hors = m.lignes.filter(l => l.titre && !l.titre.dansRangee);
        verdict('titres de bloc dans leur rangée', hors.length === 0, hors.map(l => l.key + ':' + l.titre.txt).join(' '));
        verdict('titres de bloc sans débordement (aucun mot coupé ni sorti de sa case)', m.titresDebordants.length === 0, m.titresDebordants.join(' · '));
        verdict('texte visible non tronqué (lignes, continuations, rangées de table, en-tête)', m.tronques.length === 0, m.tronques.join(' · '));
        const etires = m.lignes.filter(l => l.n > 0 && l.n <= 2 && w >= 1351 && l.largeurs.some(x => x > 188));
        verdict('ligne de ≤ 2 cases : case ≤ 187 px (≥ 1351 px)', etires.length === 0, etires.map(l => l.key + ':' + l.largeurs).join(' '));
        const loin = m.sousTitres.filter(x => x.dessous > x.dessus);
        verdict('sous-titre « ↳ » plus proche de sa rangée que de la précédente', loin.length === 0, JSON.stringify(loin));
        const debordees = m.tablesSerrees.filter(x => x.table > x.boite + 1);
        verdict('tables serrées tenues dans leur colonne', debordees.length === 0, JSON.stringify(debordees));
        if (m.sousTitres.length) console.log('   sous-titres ↳ (dessus / dessous, px) :', JSON.stringify(m.sousTitres));
        if (m.tablesSerrees.length) console.log('   tables serrées (table / boîte, px) :', JSON.stringify(m.tablesSerrees));
        if (m.barre) verdict(w >= 1024 ? 'barre d\'outils sur 1 rangée' : 'barre d\'outils ≤ 3 rangées et ≤ 130 px',
          w >= 1024 ? m.barre.rangees === 1 : (m.barre.rangees <= 3 && m.barre.hauteur <= 130), `${m.barre.rangees} rangée(s), ${m.barre.hauteur} px`);
        else verdict('barre d\'outils partagée présente', false, 'pas de .st-key-' + p + 'barre');
        if (m.onglets) verdict('onglets sans débordement', m.onglets.right <= m.innerWidth, `bord droit ${m.onglets.right}, ${m.onglets.rangees} rangée(s)`);
        if (mobile) verdict('bandeau et premier dessin dans les 844 px après le titre', m.premierEcran.bandeau !== null && m.premierEcran.bandeau <= 844 && m.premierEcran.dessin !== null && m.premierEcran.dessin <= 844, JSON.stringify(m.premierEcran));
        console.log('   lignes :', JSON.stringify(m.lignes.map(l => [l.key.replace('st-key-' + p, ''), l.largeurs, l.rangees])));
        const h = await page.evaluate(() => Math.max(...[...document.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main, .stMainBlockContainer')].map(e => e.scrollHeight), document.documentElement.scrollHeight));
        await page.setViewportSize({ width: w, height: Math.min(h + 100, 12000) });
        await page.waitForTimeout(2500);
        await page.screenshot({ path: `${OUT}/apres_${mod}${SUFFIXE}_${w}.png` });
        console.log(`   capture ${OUT}/apres_${mod}${SUFFIXE}_${w}.png (hauteur ${h})`);
      } catch (e) { KO++; console.log('   échec :', e.message.slice(0, 300)); }
      await ctx.close();
    }
  }
  await browser.close();
  console.log(KO ? `\n${KO} seuil(s) non tenu(s)` : '\nTous les seuils sont tenus');
  process.exit(KO ? 1 : 0);
})();
