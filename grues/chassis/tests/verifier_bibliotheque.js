/* Contrôles de la bibliothèque des châssis de grue.
   Usage : node tests/verifier_bibliotheque.js
   Sort avec le code 0 si tout est bon, 1 sinon. Aucune dépendance. */
const fs = require('fs'), path = require('path');
const CD = require('../src/chassis_dessin.js');
const base = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'donnees', 'chassis_grues.json'), 'utf8'));
const v = CD.valeur;
let ok = 0, ko = 0;
function test(nom, cond, detail) {
  if (cond) ok++; else { ko++; console.log('ÉCHEC  ' + nom + (detail ? '  → ' + detail : '')); }
}
function proche(a, b, tol) { return Math.abs(a - b) <= tol; }
function aire(poly) { let s = 0; for (let i = 0; i < poly.length; i++) { const p = poly[i], q = poly[(i + 1) % poly.length]; s += p[0] * q[1] - q[0] * p[1]; } return Math.abs(s) / 2; }

/* 1. Valeurs de référence relevées sur les sources (garde-fou contre une modification accidentelle de la base).
      [entraxe, forme, a ou Ø ou cote sur plats, b, rotation, hors-tout déclaré] */
const REF = {
  'liebherr-base-cruciforme-16ec160cb-0380': [3800, 'rectangle', 430, 380, 45, 4370],
  'liebherr-63lc-croix-3800-pyramides': [3800, 'octogone', 1100, null, 0, 4900],
  'liebherr-132hc-chassis-4600': [4600, 'rectangle', 580, 400, 45, 5293],
  'liebherr-200hc-chassis-6000': [6000, 'rectangle', 580, 400, 45, 6690],
  'liebherr-290hc-21hc290-chassis-6000': [6000, 'rectangle', 600, 560, 45, 6835],
  'liebherr-290hc-chassis-6000-pieds-a-vis': [6000, 'cercle', 500, null, 0, 6835],
  'liebherr-390hc-chassis-8000': [8000, 'rectangle', 700, 700, 0, 8700],
  'wolff-kr-6-40': [4000, 'rectangle', 320, 320, 45, 4460],
  'wolff-kr-7-32': [3200, 'rectangle', 330, 330, 45, 3700],
  'wolff-krv-7-32': [3200, 'rectangle', 320, 320, 45, 3700],
  'wolff-krv-7-32-46': [4600, 'rectangle', 320, 320, 45, 5100],
  'wolff-kr-8-46': [4600, 'rectangle', 400, 400, 45, 5100],
  'wolff-kr-10-46': [4600, 'rectangle', 450, 400, 45, 5100],
  'wolff-kr-10-46-60': [6000, 'rectangle', 450, 400, 45, 6500],
  'wolff-krv-10-60-a60': [6000, null, null, null, null, 6700],
  'wolff-krv-10-60-a50': [5000, null, null, null, null, 6700],
  'wolff-kr-12-60': [6000, 'rectangle', 550, 550, 45, 6800],
  'wolff-kr-12-60-80': [8000, 'rectangle', 550, 550, 45, 8600],
  'wolff-kr-1000-8': [8000, 'rectangle', 440, 440, 45, 8600],
  'wolff-kr-16-80': [8000, 'rectangle', 550, 550, 45, 8800],
  'wolff-kr-16-80-100': [10000, 'rectangle', 550, 550, 45, 10800]
};

test('21 châssis dans la base', base.chassis.length === 21, 'trouvé ' + base.chassis.length);
const ids = base.chassis.map(c => c.id);
test('identifiants uniques', new Set(ids).size === ids.length);
test('identifiants = table de référence', ids.slice().sort().join() === Object.keys(REF).sort().join());

for (const ch of base.chassis) {
  const id = ch.id, R = REF[id];
  if (!R) continue;
  const ex = v(ch.appuis.entraxe_x), ey = v(ch.appuis.entraxe_y), ap = ch.pied.appui;
  test(id + ' : champs obligatoires', ['constructeur', 'nom', 'type', 'source', 'appuis', 'pied', 'cotes_source', 'remarques', 'a_verifier'].every(k => k in ch));
  test(id + ' : entraxe', ex === R[0] && ey === R[0], ex + ' × ' + ey);
  test(id + ' : hors-tout déclaré', v(ch.appuis.hors_tout) === R[5], String(v(ch.appuis.hors_tout)));
  test(id + ' : source renseignée', !!ch.source.fichier && ch.source.pages.length > 0);
  /* positions des 4 appuis */
  const A = CD.appuis(ch);
  test(id + ' : 4 appuis symétriques', A.length === 4 && A.every(a => Math.abs(a.x) === ex / 2 && Math.abs(a.y) === ey / 2));
  test(id + ' : positions enregistrées', ch.appuis.positions.every((p, i) => p.x === A[i].x && p.y === A[i].y && p.n === i + 1));
  /* platine */
  if (R[1] === null) {
    test(id + ' : platine absente (non indiquée dans la source)', ap === null && CD.emprises(ch).every(e => e === null));
  } else {
    test(id + ' : forme de la platine', ap && ap.forme === R[1], ap && ap.forme);
    const dim = ap.forme === 'rectangle' ? v(ap.a) : ap.forme === 'cercle' ? v(ap.diametre) : v(ap.cote_sur_plats);
    test(id + ' : dimension a', dim === R[2], String(dim));
    if (ap.forme === 'rectangle') test(id + ' : dimension b', v(ap.b) === R[3], String(v(ap.b)));
    test(id + ' : rotation', ap.rotation_deg === R[4], String(ap.rotation_deg));
    /* aire : polygone de l'emprise = aire enregistrée */
    const E = CD.emprises(ch);
    const tolAire = ap.forme === 'cercle' ? 0.004 : 0.0005;
    test(id + ' : aire de l\'emprise', E.every(e => Math.abs(aire(e.sommets) / 1e6 - v(ap.aire_m2)) <= tolAire * v(ap.aire_m2) + 1e-4), aire(E[0].sommets) / 1e6 + ' / ' + v(ap.aire_m2));
    test(id + ' : côté du carré de même aire', proche(v(ap.cote_carre_meme_aire), Math.sqrt(v(ap.aire_m2)) * 1000, 1));
    /* hors-tout calculé à partir des emprises */
    const H = CD.horsToutEmprises(ch);
    test(id + ' : emprises symétriques', proche(H.x, H.y, 0.01));
    const nat = ch.appuis.hors_tout.nature || '';
    if (/platines|pyramides/.test(nat)) test(id + ' : hors-tout des emprises = cote de la source', proche(H.x, v(ch.appuis.hors_tout), 5), H.x.toFixed(1) + ' / ' + v(ch.appuis.hors_tout));
    if (ch.appuis.hors_tout_platines) test(id + ' : hors-tout des platines (calcul)', proche(H.x, v(ch.appuis.hors_tout_platines), 1), H.x.toFixed(1));
  }
  /* dessins */
  for (const [nom, f] of Object.entries({ plan: CD.plan, elevation: CD.elevation, piedPlan: CD.piedPlan, piedCoupe: CD.piedCoupe })) {
    let s = '';
    try { s = f(ch, { couleursFixes: true }); } catch (e) { s = 'ERREUR ' + e.message; }
    test(id + ' : dessin ' + nom, s.startsWith('<svg') && s.endsWith('</svg>') && !/NaN|undefined|Infinity|var\(/.test(s), s.slice(0, 80));
    const ouv = (s.match(/<(g|svg|clipPath)[ >]/g) || []).length, fer = (s.match(/<\/(g|svg|clipPath)>/g) || []).length;
    test(id + ' : dessin ' + nom + ' bien formé', ouv === fer, ouv + ' / ' + fer);
  }
}

/* 2. Recoupements entre cotes d'une même source */
const C = Object.fromEntries(base.chassis.map(c => [c.id, c]));
let c = C['liebherr-132hc-chassis-4600'];
test('132 HC : 3807 − 3402 = axe du pied de mât', v(c.elevation.hauteur_totale) - v(c.elevation.hauteur_axe_pied_de_mat_au_sommet) === v(c.elevation.axe_pied_de_mat));
test('132 HC : 3957 − 3807 = surélévation des massifs (150)', v(c.elevation.hauteur_depuis_niveau_bas) - v(c.elevation.hauteur_totale) === v(c.fondation.variante_plots.surelevation));
test('132 HC : 4600 + (580 + 400)/√2 = 5293', proche(4600 + 980 / Math.SQRT2, 5293, 1));
c = C['liebherr-200hc-chassis-6000'];
test('200 HC : 4112 − 3627 = axe du pied de mât', v(c.elevation.hauteur_totale) - v(c.elevation.hauteur_axe_pied_de_mat_au_sommet) === v(c.elevation.axe_pied_de_mat));
test('200 HC : 6000 + 2000 = 8000 (massifs dessinés)', v(c.appuis.entraxe_x) + v(c.fondation.variante_plots.cote_massif) === 8000);
c = C['liebherr-290hc-chassis-6000-pieds-a-vis'];
test('290 HC pieds à vis : sabots 600 × 560 → 6835 hors-tout', proche(6000 + 2 * (v(c.pied.sabot.axe_au_bord_exterieur) + v(c.pied.sabot.a) / 2) / Math.SQRT2, 6835, 3));
test('290 HC pieds à vis : longrine 7000 = 6000 + 2 × 500', v(c.appuis.entraxe_x) + 2 * v(c.fondation.fondation_dessinee.debord_exterieur) === v(c.fondation.fondation_dessinee.hors_tout));
test('290 HC pieds à vis : longrine 800 = 500 + 300', v(c.fondation.fondation_dessinee.debord_exterieur) + v(c.fondation.fondation_dessinee.debord_interieur) === v(c.fondation.fondation_dessinee.largeur));
c = C['liebherr-390hc-chassis-8000'];
test('390 HC : 8000 + 700 = 8700', v(c.appuis.entraxe_x) + v(c.pied.appui.a) === v(c.appuis.hors_tout));
c = C['liebherr-63lc-croix-3800-pyramides'];
test('63 LC : 3800 + 1100 = 4900', v(c.appuis.entraxe_x) + v(c.pied.appui.cote_sur_plats) === v(c.appuis.hors_tout));
test('63 LC : hauteur 1518 entre min. 1475 et max. 1560', c.pied.hauteur_totale.min <= v(c.pied.hauteur_totale) && v(c.pied.hauteur_totale) <= c.pied.hauteur_totale.max);

/* 3. Origine de chaque valeur déclarée */
const ORIG = ['cote', 'mesure', 'calcul', 'indicatif'];
let nval = 0, sansOrigine = 0;
(function parcourir(o) {
  if (Array.isArray(o)) o.forEach(parcourir);
  else if (o && typeof o === 'object') {
    if ('valeur' in o) { nval++; if (ORIG.indexOf(o.origine) < 0) sansOrigine++; }
    Object.values(o).forEach(parcourir);
  }
})(base.chassis);
test('chaque valeur porte son origine (cote, mesure, calcul, indicatif)', sansOrigine === 0, sansOrigine + ' sans origine sur ' + nval);

console.log(ok + ' contrôles réussis, ' + ko + ' échec(s) – ' + nval + ' valeurs dans la base.');
process.exit(ko ? 1 : 0);
