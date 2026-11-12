// Les PLANS d'un ouvrage, faits par le moteur de RBAR (js/plans-travailleur.js) : la page lui passe le fichier
// « rbar.ouvrage » que le moteur Python a écrit (le même que « DWG pour AutoCAD »), il rend les feuilles en PDF. Un
// travailleur pour le document, créé à la première demande — ou plus tôt, quand le menu « Exporter » s'ouvre
// (prechaufferPlans : le moteur se charge en ≈ 2 s) ; tombé, il est refait à la demande suivante.
//
// Les deux plans (coffrage, armatures) sortent du MÊME calcul : le dernier résultat est gardé pour le même fichier — le
// second plan téléchargé ne recalcule rien.
let w = null;
let n = 0;
const attentes = new Map();
let dernier = { texte: null, promesse: null };

function travailleur() {
  if (w) return w;
  const t = w = new Worker(new URL("./plans-travailleur.js", import.meta.url), { type: "module" });
  t.onmessage = ({ data }) => {
    const a = attentes.get(data && data.id);
    if (!a) return;
    attentes.delete(data.id);
    if (data.erreur) a.ko(new Error(data.erreur)); else a.ok(data);
  };
  t.onerror = (e) => {
    if (w !== t) return;
    const raison = new Error(`le moteur de plans s'est arrêté (${(e && e.message) || "erreur inconnue"}) ; il redémarre à la demande suivante`);
    for (const a of attentes.values()) a.ko(raison);
    attentes.clear();
    try { t.terminate(); } catch (x) { /* déjà arrêté */ }
    w = null;
    dernier = { texte: null, promesse: null };
  };
  return t;
}

function demander(message) {
  const id = ++n;
  return new Promise((ok, ko) => {
    attentes.set(id, { ok, ko });
    travailleur().postMessage({ id, ...message });
  });
}

/** Le moteur se charge d'avance (le premier plan vient plus vite) ; rien n'est attendu, aucune erreur n'est rendue. */
export function prechaufferPlans() {
  if (w) return;
  demander({ fonction: "charger" }).catch(() => {});
}

/** Les plans PDF d'un fichier « rbar.ouvrage » (son texte JSON) : { rapport, pdfs: [{ numero, plan, nom, octets }],
 *  version, chargement_ms, calcul_ms } — ``rapport`` est le compte rendu de RBAR (``ok`` faux : ``refus`` dit pourquoi). */
export function plansPdf(texte, nom) {
  if (dernier.texte !== texte || !dernier.promesse) {
    const p = demander({ fonction: "plans", ouvrage: texte, nom })
      .then((d) => ({ ...d.resultat, version: d.version, chargement_ms: d.chargement_ms, calcul_ms: d.calcul_ms }));
    dernier = { texte, promesse: p };
    p.catch(() => { if (dernier.promesse === p) dernier = { texte: null, promesse: null }; });
  }
  return dernier.promesse;
}
