// La MÉTÉO DU LIEU (Open-Meteo) : une IMPORTATION de données à la demande, jamais du code.
//
// Le moteur (Python, beton/maturite/meteo.py) a fait le PLAN — le lieu à trouver, quelles dates vont à l'archive et
// lesquelles aux prévisions, quelles variables — et l'a posé sur le bouton (commun.bouton_meteo, data-meteo). Ce script
// fait les appels, et SEULEMENT vers les trois adresses ci-dessous : il construit lui-même chaque adresse (la base vient
// d'ici, jamais du plan ; chaque paramètre est filtré et encodé). Il rend ensuite au moteur les réponses BRUTES : c'est le
// moteur qui les lit, les contrôle et les garde dans la note — le calcul, lui, se fait hors ligne.
//
// Les messages de refus (pas de réseau, service en panne ou saturé, réponse illisible) sont écrits ici, parce qu'ici seul
// on le sait ; ceux qui portent sur les DONNÉES (lieu introuvable, jours manquants, dates hors de portée) viennent du moteur.
//
// Les trois adresses sont une COPIE nommée de beton/maturite/meteo.py (ADRESSES) et de la politique de sécurité du site
// (_headers, connect-src) : tests/test_site_web.py lie les trois.
export const ADRESSES = Object.freeze({
  geocodage: "https://geocoding-api.open-meteo.com/v1/search",
  archive: "https://archive-api.open-meteo.com/v1/archive",
  prevision: "https://api.open-meteo.com/v1/forecast",
});

// les paramètres permis à chaque adresse : aucun autre ne part
const PERMIS = Object.freeze({
  geocodage: ["name", "count", "language"],
  archive: ["latitude", "longitude", "start_date", "end_date", "timezone", "daily", "hourly"],
  prevision: ["latitude", "longitude", "start_date", "end_date", "timezone", "daily", "hourly"],
});

export const DELAI_MS = 20000;        // l'écran Streamlit attend 20 s (timeout=20)
export const ATTENTE_429_MS = 2000;   // … et réessaie une fois, 2 s après un « trop de demandes » (429)

const SECOURS = "saisissez une température constante ou un relevé (le calcul reste possible hors ligne)";
export const MESSAGES = Object.freeze({
  reseau: `Pas de réseau : la météo n'a pas pu être récupérée (Open-Meteo injoignable). En attendant, ${SECOURS}.`,
  delai: `Le service météo ne répond pas (${DELAI_MS / 1000} s) : réessayez plus tard, ou ${SECOURS}.`,
  sature: `Service météo saturé (Open-Meteo limite le nombre de demandes) : réessayez dans quelques minutes, ou ${SECOURS}.`,
  illisible: `Réponse illisible du service météo : réessayez plus tard, ou ${SECOURS}.`,
  plan: "Le plan de la récupération est illisible : rechargez la page, puis cliquez à nouveau.",
  panne: (code, raison) => `Service météo en panne (erreur ${code}${raison ? ` — ${raison}` : ""}) : réessayez plus tard, ou ${SECOURS}.`,
});

// L'adresse d'un appel : la base d'Open-Meteo de cette source, et les paramètres permis, encodés.
export function adresse(source, parametres) {
  if (!Object.prototype.hasOwnProperty.call(ADRESSES, source)) throw new Error(`adresse inconnue « ${String(source).slice(0, 40)} »`);
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(parametres || {})) {
    if (!PERMIS[source].includes(k)) throw new Error(`paramètre refusé « ${String(k).slice(0, 40)} »`);
    if (!((typeof v === "string" && v.length <= 200) || (typeof v === "number" && Number.isFinite(v)))) throw new Error(`valeur refusée pour « ${k} »`);
    q.append(k, String(v));
  }
  return `${ADRESSES[source]}?${q}`;
}

// Un appel : la réponse JSON, ou un refus en français. Un « trop de demandes » (429) est réessayé une fois, comme l'écran.
async function appeler(source, parametres, o) {
  let url;
  try { url = adresse(source, parametres); } catch (e) { return { ok: false, erreur: `${MESSAGES.plan} (${e.message})` }; }
  for (let essai = 0; essai < 2; essai++) {
    const ctl = new AbortController();
    const minuterie = setTimeout(() => ctl.abort(), o.delai);
    let rep, json = null, lu = false;
    try {
      rep = await o.fetch(url, { signal: ctl.signal, credentials: "omit", cache: "no-store", referrerPolicy: "no-referrer" });
      try { json = await rep.json(); lu = true; } catch (e) { if (e && e.name === "AbortError") throw e; }
    } catch (e) {
      return { ok: false, erreur: e && e.name === "AbortError" ? MESSAGES.delai : MESSAGES.reseau };
    } finally { clearTimeout(minuterie); }
    if (rep.status === 429) {
      if (essai === 0) { await o.attendre(ATTENTE_429_MS); continue; }
      return { ok: false, erreur: MESSAGES.sature };
    }
    if (!rep.ok) return { ok: false, erreur: MESSAGES.panne(rep.status, json && typeof json.reason === "string" ? json.reason.slice(0, 200) : "") };
    if (!lu || json === null || typeof json !== "object" || Array.isArray(json)) return { ok: false, erreur: MESSAGES.illisible };
    return { ok: true, json };
  }
  return { ok: false, erreur: MESSAGES.sature };
}

// La récupération d'un plan : le géocodage du lieu (sauf si le plan donne la position), puis chaque appel, dans l'ordre.
// Rend { ok: true, valeur: { lieu, plan, geocodage, reponses } } — à envoyer TEL QUEL au module — ou { ok: false, erreur }.
// Un lieu que le géocodage ne trouve pas part quand même au module (sans réponses) : c'est lui qui le dit.
export async function recupererMeteo(plan, options = {}) {
  const o = {
    fetch: options.fetch || ((u, init) => globalThis.fetch(u, init)),
    attendre: options.attendre || ((ms) => new Promise((ok) => setTimeout(ok, ms))),
    delai: options.delai || DELAI_MS,
  };
  try {
    if (!plan || typeof plan !== "object" || Array.isArray(plan)) return { ok: false, erreur: MESSAGES.plan };
    if (typeof plan.erreur === "string") return { ok: false, erreur: plan.erreur };
    if (!Array.isArray(plan.appels) || !plan.appels.every((a) => a && ["archive", "prevision"].includes(a.source) && a.parametres && typeof a.parametres === "object")) {
      return { ok: false, erreur: MESSAGES.plan };
    }
    let position = plan.position || null, geocodage = null;
    if (!position) {
      const g = await appeler("geocodage", plan.geocodage, o);
      if (!g.ok) return g;
      geocodage = g.json;
      const r0 = Array.isArray(geocodage.results) ? geocodage.results[0] : null;
      if (!r0 || !Number.isFinite(r0.latitude) || !Number.isFinite(r0.longitude)) return { ok: true, valeur: { lieu: plan.lieu ?? null, plan, geocodage, reponses: [] } };
      position = { latitude: r0.latitude, longitude: r0.longitude };
    }
    const reponses = [];
    for (const a of plan.appels) {
      const r = await appeler(a.source, { latitude: position.latitude, longitude: position.longitude, ...a.parametres }, o);
      if (!r.ok) return r;
      reponses.push(r.json);
    }
    return { ok: true, valeur: { lieu: plan.lieu ?? null, plan, geocodage, reponses } };
  } catch (e) {
    return { ok: false, erreur: `La météo n'a pas pu être récupérée : ${e && e.message ? e.message : e}` };
  }
}
