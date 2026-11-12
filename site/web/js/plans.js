// Les PLANS d'un ouvrage, faits par le moteur de RBAR (js/plans-travailleur.js) : la page lui passe le fichier
// « rbar.ouvrage » que le moteur Python a écrit (le même que « DWG pour AutoCAD »), il rend les feuilles en PDF. Un
// travailleur pour le document, créé à la première demande — ou plus tôt, quand le menu « Exporter » s'ouvre
// (prechaufferPlans : le moteur se charge en ≈ 2 s au poste de contrôle) ; tombé, il est refait à la demande suivante.
//
// Les plans sortent en UN PDF, une page par feuille (coffrage, puis armatures : RBAR 0.4.40, « unSeulPdf »). Le dernier
// résultat est gardé pour le même fichier : un second clic ne recalcule rien (``reutilise`` le dit).
//
// L'ATTENTE SE DIT, ET ELLE A UNE FIN. Chaque demande reçoit les étapes du travailleur (``suivre`` : { etape, mo }) ; une
// demande sans réponse après DELAI_PLANS_S secondes est abandonnée : le travailleur est arrêté (un moteur bloqué ne se
// débloque pas), l'erreur dit l'étape où il en était et le dernier incident qu'il a signalé ; la demande suivante en refait
// un neuf. Né du premier essai du propriétaire sur sa tablette (12/11/2026) : « 30 secondes, rien reçu », sans qu'aucun
// écran puisse dire si le moteur se téléchargeait, dessinait ou était bloqué.
export const DELAI_PLANS_S = 120;
// les ÉTAPES d'un plan, telles que l'écran les dit (js/etabli.js) et que le diagnostic d'un échec les nomme
export const ETAPES_PLAN = { fichier: "écriture du fichier de l'ouvrage", chargement: "chargement du moteur de RBAR", calcul: "dessin des feuilles par RBAR" };

let w = null;
let n = 0;
let incident = null; // le dernier incident hors demande dit par le travailleur en service
const attentes = new Map(); // id → { ok, ko, suivre, etape, minuteur }
let dernier = null; // { texte, promesse, suivis, etape } : les plans du dernier fichier demandé

// le travailleur est arrêté : chaque demande en cours reçoit la raison ; la suivante en refera un neuf
function arreter(raison) {
  const t = w;
  w = null;
  dernier = null;
  for (const a of attentes.values()) {
    clearTimeout(a.minuteur);
    a.ko(raison);
  }
  attentes.clear();
  if (t) {
    try { t.terminate(); } catch (x) { /* déjà arrêté */ }
  }
}

function travailleur() {
  if (w) return w;
  const t = w = new Worker(new URL("./plans-travailleur.js", import.meta.url), { type: "module" });
  incident = null;
  t.onmessage = ({ data }) => {
    if (w !== t || !data) return;
    if (data.id === null || data.id === undefined) {
      if (data.incident) incident = String(data.incident);
      return;
    }
    const a = attentes.get(data.id);
    if (!a) return;
    if (data.etape) {
      a.etape = data.etape;
      if (a.suivre) a.suivre({ etape: data.etape, mo: data.mo ?? null });
      return;
    }
    attentes.delete(data.id);
    clearTimeout(a.minuteur);
    if (!data.erreur) {
      a.ok(data);
      return;
    }
    a.ko(new Error(data.erreur));
    // un moteur qui n'a pas pu se charger ne se recharge pas dans le même travailleur : la demande suivante en refait un
    if (!data.pret) arreter(new Error(`le moteur de RBAR n'a pas pu se charger (${data.erreur})`));
  };
  // (le script du travailleur lui-même n'a pas pu démarrer : ses propres erreurs, il les attrape et les dit)
  t.onerror = (e) => {
    if (w !== t) return;
    arreter(new Error(`le moteur de plans s'est arrêté (${(e && e.message) || "erreur inconnue"}) ; il redémarre à la demande suivante`));
  };
  return t;
}

function demander(message, { suivre = null, delai_s = null } = {}) {
  const id = ++n;
  return new Promise((ok, ko) => {
    const a = { ok, ko, suivre, etape: null, minuteur: null };
    if (delai_s) {
      a.minuteur = setTimeout(() => {
        const s = String(delai_s).replace(".", ",");
        const raison = a.etape === "calcul" ? `RBAR n'a pas fini de dessiner les feuilles en ${s} s`
          : `le moteur de RBAR ne s'est pas chargé en ${s} s — réessaie, de préférence en Wi-Fi (il se télécharge la première fois)`;
        arreter(new Error(raison + (incident ? ` (dernier incident : ${incident})` : "")));
      }, delai_s * 1000);
    }
    attentes.set(id, a);
    travailleur().postMessage({ id, ...message });
  });
}

/** Le moteur se charge d'avance (le premier plan vient plus vite) ; rien n'est attendu, aucune erreur n'est rendue. */
export function prechaufferPlans() {
  lireAppareil();
  if (w) return;
  demander({ fonction: "charger" }).catch(() => {});
}

// ------------------------------------------------------------------ ce qu'un échec doit dire (RBAR, 12/11/2026, nuit)
// Pour comprendre un plan qui ne vient pas sur une tablette, RBAR demande : l'appareil (système, navigateur, modèle,
// mémoire, cœurs), l'étape atteinte et sa durée, les Mo reçus, si la page a été cachée pendant le calcul (Android ralentit
// puis GÈLE un onglet passé en arrière-plan, et son travailleur avec lui, sans rien lever), la dernière erreur du
// travailleur (le délai la donne déjà). Le message d'échec le dit — une capture d'écran suffit à le transmettre.
let appareilHaut = null; // le modèle et la version du système, quand le navigateur les donne (lus d'avance : c'est asynchrone)
function lireAppareil() {
  try {
    const d = globalThis.navigator && globalThis.navigator.userAgentData;
    if (!appareilHaut && d && typeof d.getHighEntropyValues === "function") {
      d.getHighEntropyValues(["model", "platformVersion"]).then((v) => { appareilHaut = v; }, () => {});
    }
  } catch (x) { /* rien à lire : le diagnostic s'en passe */ }
}

/** L'APPAREIL en quelques mots : système, navigateur et sa version, modèle (quand le navigateur le donne), mémoire, cœurs.
 *  ``nav`` : navigator (un autre objet pour les essais) ; ``haut`` : les valeurs lues par getHighEntropyValues, ou rien. */
export function appareil(nav = globalThis.navigator, haut = appareilHaut) {
  if (!nav) return "";
  const ua = String(nav.userAgent || "");
  const lu = (re) => (re.exec(ua) || [])[1] || "";
  const d = nav.userAgentData;
  const marques = d && Array.isArray(d.brands) ? d.brands.filter((m) => !/Not.?A.?Brand/i.test(m.brand)) : [];
  const marque = marques.find((m) => !/^Chromium$/i.test(m.brand)) || marques[0] || null;
  const pf = (haut && haut.platform) || (d && d.platform) || "";
  const systeme = pf ? `${pf}${haut && haut.platformVersion ? ` ${haut.platformVersion}` : ""}`
    : lu(/Android (\d+(?:\.\d+)?)/) ? `Android ${lu(/Android (\d+(?:\.\d+)?)/)}`
      : /iPad|iPhone/.test(ua) ? "iOS" : /Macintosh/.test(ua) ? "macOS" : /Windows/.test(ua) ? "Windows" : /Linux/.test(ua) ? "Linux" : "";
  const navigateur = marque ? `${marque.brand} ${marque.version}`
    : lu(/SamsungBrowser\/(\d+)/) ? `Samsung Internet ${lu(/SamsungBrowser\/(\d+)/)}`
      : lu(/Edg\/(\d+)/) ? `Edge ${lu(/Edg\/(\d+)/)}`
        : lu(/Firefox\/(\d+)/) ? `Firefox ${lu(/Firefox\/(\d+)/)}`
          : lu(/Chrome\/(\d+)/) ? `Chrome ${lu(/Chrome\/(\d+)/)}`
            : lu(/Version\/(\d+(?:\.\d+)?)[^)]*Safari/) ? `Safari ${lu(/Version\/(\d+(?:\.\d+)?)[^)]*Safari/)}` : "";
  return [systeme, navigateur, (haut && haut.model) || "",
    Number.isFinite(nav.deviceMemory) ? `${String(nav.deviceMemory).replace(".", ",")} Go de mémoire` : "",
    Number.isFinite(nav.hardwareConcurrency) ? `${nav.hardwareConcurrency} cœurs` : ""].filter(Boolean).join(" · ");
}

/** Le DIAGNOSTIC d'un plan qui n'est pas venu : l'étape atteinte et depuis quand, les Mo reçus, la page cachée, l'appareil. */
export function diagnosticPlans({ etape = null, depuis_ms = null, mo = null, cachee = false } = {}, nav = globalThis.navigator, haut = appareilHaut) {
  return [
    etape ? `étape : ${ETAPES_PLAN[etape] || etape}${Number.isFinite(depuis_ms) ? ` depuis ${Math.round(depuis_ms / 1000)} s` : ""}` : "",
    Number.isFinite(mo) ? `${String(mo).replace(".", ",")} Mo reçus` : "",
    cachee ? "page cachée pendant le calcul" : "",
    appareil(nav, haut),
  ].filter(Boolean).join(" · ");
}

/** L'ÉCRAN GARDÉ ALLUMÉ pendant un plan (Android gèle un onglet dont l'écran s'éteint) : la demande part tout de suite, sans
 *  être attendue ; refusée ou absente, le plan se fait quand même. Rend de quoi le relâcher. */
export function garderEcranAllume(nav = globalThis.navigator) {
  let lache = false;
  let verrou = null;
  try {
    if (nav && nav.wakeLock && typeof nav.wakeLock.request === "function") {
      nav.wakeLock.request("screen").then((v) => {
        if (lache) v.release().catch(() => {});
        else verrou = v;
      }, () => {});
    }
  } catch (x) { /* refusé ou absent */ }
  return () => {
    lache = true;
    if (verrou) {
      verrou.release().catch(() => {});
      verrou = null;
    }
  };
}

/** Les plans PDF d'un fichier « rbar.ouvrage » (son texte JSON) : { rapport, pdfs: [{ numero, plan, nom, octets }],
 *  version, chargement_ms, calcul_ms, pret_ms, reutilise } — ``rapport`` est le compte rendu de RBAR (``ok`` faux :
 *  ``refus`` dit pourquoi) ; ``pret_ms`` : quand le moteur a été prêt (ms depuis la naissance du travailleur) ;
 *  ``reutilise`` : le résultat d'un calcul déjà demandé pour ce fichier (rien n'a été recalculé pour cet appel).
 *  ``suivre({ etape, mo })`` reçoit les étapes : « chargement » (mo : mégaoctets du moteur déjà reçus), « calcul ». */
export function plansPdf(texte, nom, { suivre = null, delai_s = DELAI_PLANS_S } = {}) {
  lireAppareil();
  const reutilise = !!dernier && dernier.texte === texte;
  if (!reutilise) {
    const d = { texte, promesse: null, suivis: new Set(), etape: null };
    d.promesse = demander({ fonction: "plans", ouvrage: texte, nom }, {
      delai_s,
      suivre: (e) => {
        d.etape = e;
        for (const f of d.suivis) f(e);
      },
    }).then((r) => ({ ...r.resultat, version: r.version, chargement_ms: r.chargement_ms, calcul_ms: r.calcul_ms, pret_ms: r.pret_ms }));
    d.promesse.catch(() => { if (dernier === d) dernier = null; });
    dernier = d;
  }
  const d = dernier;
  if (suivre) {
    d.suivis.add(suivre);
    if (d.etape) suivre(d.etape);
  }
  return d.promesse.then((r) => ({ ...r, reutilise })).finally(() => { if (suivre) d.suivis.delete(suivre); });
}
