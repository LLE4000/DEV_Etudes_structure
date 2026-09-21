const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
vm.runInThisContext(m[1]);
const DC = globalThis.DC;
const num = v => (typeof v === 'number' && !isFinite(v)) ? String(v) : v;
function snap(R) {
  const sc = {}; for (const k in R) if (['number', 'string', 'boolean'].includes(typeof R[k])) sc[k] = num(R[k]);
  const p = R.pd;
  return { scalaires: sc, statut: R.statut, verified: R.verified, eta_max: num(R.eta_max), gouvernante: R.gov ? R.gov.key : null,
    verifications: R.checks.map(c => ({ key: c.key, active: c.active, Ed: num(c.Ed), Rd: num(c.Rd), eta: num(c.eta), ok: c.ok, vals: c.vals })),
    alertes: R.alerts.map(a => ({ id: a.id, bloquant: a.block, message: a.msg, explication: a.why, champs: a.fields, cotes: a.dims, elements: a.elems })),
    tableau_3_3: R.dist.map(d => ({ libelle: d.lab, valeur: d.val, min: d.min, max: d.max, ok: d.ok })),
    efforts_boulons_S: R.tabS, efforts_boulons_P: R.tabP,
    predim: { boulon: p.boulon, n: p.n, e1: p.e1, p1: p.p1, e2: p.e2, LC: p.LC, eta: p.eta, treq: p.treq, tC: p.tC, gA: p.gA, legreq: p.legreq, leg: p.leg, rC: p.rC, found: p.found, hav: p.hav, msg: p.msg, reqs: p.reqs,
      lignes: p.rows.map(r => ({ boulon: r.b.n, n: r.n, e1: r.e1, p1: r.p1, e2: r.e2, Lc: r.Lc, geom: r.geom, eta: num(r.eta), etas: r.e, ok: r.ok, score: r.score, prop: r.prop || null })) } };
}
const cas = process.argv[3] ? JSON.parse(fs.readFileSync(process.argv[3], 'utf8')) : [{ id: 'defaut', inputs: {} }];
process.stdout.write(JSON.stringify(cas.map(c => ({ id: c.id, inputs: c.inputs, attendu: snap(DC.compute(c.inputs)) })), null, 1));
