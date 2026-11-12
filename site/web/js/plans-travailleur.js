// Le moteur de PLANS de RBAR (WebAssembly : vendor/rbar-plans-web-<version>, déposé et vérifié par outils/vendre_rbar_plans.py)
// dans un travailleur à part : un calcul prend quelques secondes, l'écran ne se fige pas ; il ne partage rien avec le moteur
// Python (js/travailleur.js). C'est le MÊME moteur que la commande RBAROUVRAGE d'AutoCAD et que le programme rbar-plans :
// le site ne dessine aucun plan lui-même.
//
// Message reçu : { id, fonction: "charger" | "plans" | "apercu", ouvrage, nom } ; rendu : { id, version, resultat,
// chargement_ms, calcul_ms, pret_ms } ou { id, erreur, pret }. Les PDF repartent sans copie (transfert de leurs octets) :
// « plans » rend UN PDF, une page par feuille dans leur ordre (coffrage, puis armatures) — l'option « unSeulPdf » de RBAR
// (0.4.40), demandée par le site pour un geste, un fichier (le téléphone ne demande pas d'autoriser deux téléchargements).
// En chemin, l'ÉTAPE : { id, etape: "chargement", mo } (le moteur se télécharge et démarre ; mo : les mégaoctets déjà
// reçus du paquet, chaque seconde) puis { id, etape: "calcul" } (RBAR dessine les feuilles) — la page la dit à l'écran :
// sur une tablette, le premier plan peut prendre bien plus que les quelques secondes du poste de contrôle, et un écran
// muet ne dit pas si le moteur travaille ou s'il est bloqué (le propriétaire, 12/11/2026 : « 30 secondes, rien reçu »).
// Un incident hors d'une demande (une erreur que personne n'attrape) est dit aussi : { id: null, incident }.
//
// Le paquet est chargé ICI, à la première demande (import dynamique), et non en tête du fichier : un échec de chargement
// devient une erreur qui dit pourquoi, rendue à la demande — et non un travailleur mort sans un mot.
// (le chemin du paquet est une COPIE de construire.RBAR_PLANS : tests/test_site_web.py les lie)
const PAQUET = "../vendor/rbar-plans-web-0.4.40/rbar-plans.js";

let moteurPret = null; // le moteur, une fois chargé
let pretMs = null; // quand il l'a été (ms depuis la naissance du travailleur, c.-à-d. depuis l'ouverture du menu)

// (une erreur attrapée ici ne remonte pas à la page comme une panne du travailleur — elle le tuerait, alors qu'il vit
//  encore ; si elle l'a bloqué, le délai de js/plans.js le dira, avec ce dernier incident)
const texteErreur = (e) => String((e && e.message) || e || "erreur inconnue");
self.addEventListener("error", (ev) => {
  ev.preventDefault();
  self.postMessage({ id: null, incident: texteErreur(ev.error || ev.message) });
});
self.addEventListener("unhandledrejection", (ev) => self.postMessage({ id: null, incident: texteErreur(ev.reason) }));

// les mégaoctets du paquet déjà reçus (chronologie des ressources du travailleur) ; null si le navigateur ne la tient pas
function megaoctetsRecus() {
  try {
    const o = performance.getEntriesByType("resource").filter((e) => e.name.includes("/rbar-plans-web-"))
      .reduce((s, e) => s + (e.encodedBodySize || e.transferSize || 0), 0);
    return o > 0 ? Math.round(o / 1e5) / 10 : null;
  } catch (x) {
    return null;
  }
}

async function charger(id) {
  if (moteurPret) return moteurPret;
  const dire = () => self.postMessage({ id, etape: "chargement", mo: megaoctetsRecus() });
  dire();
  const veille = setInterval(dire, 1000);
  try {
    const paquet = await import(PAQUET);
    moteurPret = await paquet.charger();
    if (pretMs === null) pretMs = Math.round(performance.now());
    return moteurPret;
  } finally {
    clearInterval(veille);
  }
}

self.onmessage = async ({ data }) => {
  const { id, fonction, ouvrage, nom } = data || {};
  try {
    const debut = performance.now();
    const moteur = await charger(id);
    const pret = performance.now();
    if (fonction === "charger") {
      self.postMessage({ id, version: moteur.version(), chargement_ms: Math.round(pret - debut), calcul_ms: 0, pret_ms: pretMs });
      return;
    }
    self.postMessage({ id, etape: "calcul" });
    const resultat = fonction === "plans" ? moteur.plans(ouvrage, { nom, unSeulPdf: true }) : moteur.apercu(ouvrage, { nom });
    const fin = performance.now();
    const transferts = fonction === "plans" ? resultat.pdfs.map((p) => p.octets.buffer) : [];
    self.postMessage({ id, version: moteur.version(), resultat, chargement_ms: Math.round(pret - debut), calcul_ms: Math.round(fin - pret),
      pret_ms: pretMs }, transferts);
  } catch (erreur) {
    self.postMessage({ id, erreur: texteErreur(erreur), pret: !!moteurPret });
  }
};
