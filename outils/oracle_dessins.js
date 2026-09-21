// Oracle Node des DESSINS : exécute les deux premiers <script> du HTML de
// référence (moteur DC + dessins DCDraw) et écrit, pour chaque cas et chaque
// niveau de cotation, le SVG de l'élévation et de la vue en plan.
//
//   node oracle_dessins.js assemblage_double_corniere_EC3.html cas.json > dessins.json
//
// cas.json = [{ "id": "...", "inputs": {...}, "niveaux": [0,1,2], "hl": {...}|null, "interactive": true }]
const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
vm.runInThisContext(scripts[0]);
vm.runInThisContext(scripts[1]);
const DC = globalThis.DC, DR = globalThis.DCDraw;
const PDK = { corn_u: 1, LC_u: 1, boulon_u: 1, n1S_u: 1, n2S_u: 1, p1S_u: 1, e1S_u: 1, e2b_u: 1, n1P_u: 1, n2P_u: 1, p1P_u: 1, e1P_u: 1, gA_u: 1 };
const cas = process.argv[3] ? JSON.parse(fs.readFileSync(process.argv[3], 'utf8')) : [{ id: 'defaut', inputs: {} }];
const out = cas.map(c => {
  const R = DC.compute(c.inputs), niveaux = c.niveaux || [0, 1, 2], inter = c.interactive !== false;
  let hl = null;
  if (c.hl) { hl = { id: c.hl, fields: {}, dims: {}, elems: {} }; const a = R.alerts.find(x => x.id === c.hl); if (a) { a.fields.forEach(k => hl.fields[k] = 1); a.dims.forEach(k => hl.dims[k] = 1); a.elems.forEach(k => hl.elems[k] = 1); } }
  const vues = {};
  niveaux.forEach(l => { const opt = { lvl: l, interactive: inter, hl: hl, locked: R.pred ? PDK : null }; vues[l] = { elev: DR.elev(R, opt), plan: DR.plan(R, opt) }; });
  return { id: c.id, vues: vues };
});
process.stdout.write(JSON.stringify(out));
