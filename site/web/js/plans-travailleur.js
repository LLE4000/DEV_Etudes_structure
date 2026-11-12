// Le moteur de PLANS de RBAR (WebAssembly : vendor/rbar-plans-web-<version>, déposé et vérifié par outils/vendre_rbar_plans.py)
// dans un travailleur à part : un calcul prend quelques secondes, l'écran ne se fige pas ; il ne partage rien avec le moteur
// Python (js/travailleur.js). C'est le MÊME moteur que la commande RBAROUVRAGE d'AutoCAD et que le programme rbar-plans :
// le site ne dessine aucun plan lui-même.
//
// Message reçu : { id, fonction: "charger" | "plans" | "apercu", ouvrage, nom } ; rendu : { id, version, resultat,
// chargement_ms, calcul_ms } ou { id, erreur }. Les PDF repartent sans copie (transfert de leurs octets).
// (le chemin du paquet est une COPIE de construire.RBAR_PLANS : tests/test_site_web.py les lie)
import { charger } from "../vendor/rbar-plans-web-0.4.38/rbar-plans.js";

self.onmessage = async ({ data }) => {
  const { id, fonction, ouvrage, nom } = data || {};
  try {
    const debut = performance.now();
    const moteur = await charger();
    const pret = performance.now();
    if (fonction === "charger") {
      self.postMessage({ id, version: moteur.version(), chargement_ms: Math.round(pret - debut), calcul_ms: 0 });
      return;
    }
    const resultat = fonction === "plans" ? moteur.plans(ouvrage, { nom }) : moteur.apercu(ouvrage, { nom });
    const fin = performance.now();
    const transferts = fonction === "plans" ? resultat.pdfs.map((p) => p.octets.buffer) : [];
    self.postMessage({ id, version: moteur.version(), resultat, chargement_ms: Math.round(pret - debut), calcul_ms: Math.round(fin - pret) },
      transferts);
  } catch (erreur) {
    self.postMessage({ id, erreur: String((erreur && erreur.message) || erreur) });
  }
};
