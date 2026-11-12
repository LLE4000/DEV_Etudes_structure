// RBAR — moteur de plans dans le navigateur (WebAssembly). Le MÊME moteur que la commande RBAROUVRAGE d'AutoCAD et que le programme
// rbar-plans : mêmes règles, mêmes textes, même dessin. Aucun réseau pendant le calcul : tous les fichiers sont à côté de celui-ci.
//
//   import { charger } from './rbar-plans.js';
//   const moteur = await charger();                       // une fois, tôt (de préférence dans un Web Worker : un calcul prend des secondes)
//   moteur.version()                                      // « 0.4.38 »
//   moteur.apercu(ouvrage, { standard, nom })             // { rapport, feuilles: [{ plan, numero, svg }] } — sans PDF
//   moteur.plans(ouvrage, { standard, nom, unSeulPdf })   // { rapport, pdfs: [{ numero, plan, nom, octets }] } — octets : Uint8Array
//
// ouvrage : le fichier « rbar.ouvrage » (son texte JSON, ou l'objet) ; standard : le CompanyStandard.json du bureau (texte ou objet ; absent :
// le standard livré) ; nom : le nom du fichier (compte rendu et noms des PDF, « nom-NUMERO.pdf » ; absent : « ouvrage.json ») ; unSeulPdf
// (0.4.40) : un seul PDF, les feuilles en pages dans leur ordre (« nom.pdf » ; une seule entrée dans pdfs).
// rapport : le compte rendu de rbar-plans (ok, refus, notes, messages, feuilles, duree_ms…) ; ok faux : refus dit pourquoi, à montrer tel quel.

import { dotnet } from './_framework/dotnet.js';

let chargement = null;

function texte(valeur) {
  if (valeur === undefined || valeur === null) return '';
  return typeof valeur === 'string' ? valeur : JSON.stringify(valeur);
}

function octets(base64) {
  const binaire = atob(base64);
  const resultat = new Uint8Array(binaire.length);
  for (let i = 0; i < binaire.length; i++) resultat[i] = binaire.charCodeAt(i);
  return resultat;
}

async function demarrer() {
  // culture fr-FR : les nombres et les textes du moteur sont en français, quelle que soit la langue du navigateur
  const { getAssemblyExports, getConfig } = await dotnet.withApplicationCulture('fr-FR').create();
  const exports = await getAssemblyExports(getConfig().mainAssemblyName);
  const m = exports.Rbar.Plans.Web.Moteur;
  m.Preparer(''); // le standard livré lu d'avance : le premier calcul vient plus vite
  return Object.freeze({
    version: () => m.Version(),
    apercu: (ouvrage, options = {}) => JSON.parse(m.Apercu(texte(ouvrage), texte(options.standard), texte(options.nom))),
    plans: (ouvrage, options = {}) => {
      const reponse = JSON.parse(m.Plans(texte(ouvrage), texte(options.standard), texte(options.nom), options.unSeulPdf === true));
      return {
        rapport: reponse.rapport,
        pdfs: reponse.pdfs.map((p) => ({ numero: p.numero, plan: p.plan, nom: p.nom, octets: octets(p.base64) })),
      };
    },
  });
}

/** Charge le moteur (une seule fois ; un échec de chargement peut être réessayé). */
export function charger() {
  if (!chargement) {
    chargement = demarrer().catch((erreur) => {
      chargement = null;
      throw erreur;
    });
  }
  return chargement;
}
