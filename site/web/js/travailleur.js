// Le travailleur du navigateur : Python (Pyodide) démarre ici, hors de la page, pour qu'elle reste fluide. Il charge le
// PAQUET DU MOTEUR (moteur/moteur.zip, dont l'empreinte est vérifiée contre moteur/manifeste.json), puis répond aux
// requêtes des écrans par api_web.routeur.traiter (JSON → JSON). Aucun calcul n'est écrit en JavaScript.
//
// UN travailleur pour TOUTE la session du document (lot S4, REFONTE_ETABLI.md, S4-D5 et C5) : il sert tous les écrans
// qui se succèdent (js/moteur.js, moteurPartage). Il est créé SANS module : la CLÉ d'une requête se lit DANS la requête
// (« module », sinon « biblio_<bibliothèque> » pour l'action « consulter », sinon rien) — une seule source.
//
// numpy pèse une à deux secondes : on ne le charge que pour les clés que le relevé du paquet a vues s'en servir
// (manifeste « avec_numpy », écrit par construire.py), AVANT la requête ou le préchauffage d'une telle clé. Si une
// telle intention arrive pendant le démarrage, numpy démarre juste après loadPyodide, en parallèle du paquet. Un import
// de numpy imprévu (un chemin rare, un module plus récent que le manifeste) est GUETTÉ en Python : la requête est alors
// rejouée, numpy chargé — même si le moteur a rattrapé l'erreur d'import en silence.
//
// TOUS les messages passent par UNE chaîne (fileT) : jamais deux traitements entrelacés (numpy, note, guet, import).
// Le PRÉCHAUFFAGE ({type:"prechauffer", cle}) est une tâche de FOND : il importe le module de cette clé, et lui seul ;
// il est sauté si un préchauffage plus récent ou une vraie requête est arrivé derrière lui (la requête importera
// d'elle-même ce qu'il lui faut) ; une erreur d'import est avalée (la vraie requête la dira).
//
// Une ERREUR FATALE de Pyodide (le moteur ne peut plus servir) est reconnue : le message « mort » part, et la page
// (js/moteur.js) rend toutes les requêtes en attente {ok:false, mort:true}, arrête ce travailleur et en recrée un.
//
// La SONDE (js/sonde.js, « sonde=1 » dans l'adresse du travailleur, posé par js/moteur.js quand la page porte
// « ?sonde=1 ») date chaque étape, sépare l'import du module de son premier calcul, PAR CLÉ (la première requête de
// chaque clé), et renvoie ses mesures à la page (message « sonde ») après cette réponse ; elle date aussi chaque
// préchauffage (« prechauffe ») et répond à « sonde_modules ». Sans elle, aucun de ces messages n'est émis : seuls
// etape, pret, erreur, reponse et mort partent.
import { loadPyodide } from "../vendor/pyodide-0.29.5/pyodide.mjs";
import { SONDE, marquer, finir, marquesAbsolues, effacerMarques } from "./sonde.js";

marquer("travailleur_debut");
if (SONDE) {
  // la compilation du WebAssembly : instantiateStreaming porte le téléchargement, la compilation et l'instanciation
  for (const f of ["instantiateStreaming", "instantiate", "compile", "compileStreaming"]) {
    const orig = WebAssembly[f];
    if (!orig) continue;
    WebAssembly[f] = async function (...a) {
      marquer(`wasm_${f}_debut`);
      const r = await orig.apply(this, a);
      marquer(`wasm_${f}_fin`);
      return r;
    };
  }
}

const base = new URL("../", import.meta.url);
const CLE = /^[a-z][a-z0-9_]*$/;
const details = {};   // SONDE : ce que Python compte (modules chargés, octets, coût d'une requête ordinaire)
const etape = (texte) => postMessage({ type: "etape", texte });
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");

// le guet : un chercheur d'import placé en tête, qui ne trouve rien lui-même mais note qu'on a demandé numpy — tant que
// numpy n'est pas chargé (« charge », posé par chargerNumpy) : après, un import de numpy n'a plus rien d'imprévu
const GUET = `
import sys
class GuetNumpy:
    """Note qu'un import de numpy a été tenté (le travailleur le charge alors et rejoue la requête)."""
    demande = False
    charge = False
    def find_spec(self, nom, chemin=None, cible=None):
        if not GuetNumpy.charge and (nom == "numpy" or nom.startswith("numpy.")):
            GuetNumpy.demande = True
        return None
sys.meta_path.insert(0, GuetNumpy())
GuetNumpy`;

// l'import d'UNE clé (préchauffage, sonde) : un module du registre par api_web.registre.module (les mêmes refus que le
// routeur), une bibliothèque par api_web.routeur.BIBLIOTHEQUES — et rien d'autre
const IMPORTER = `
def importer_cle(cle):
    import importlib
    from api_web import registre, routeur
    if cle.startswith("biblio_"):
        nom = routeur.BIBLIOTHEQUES.get(cle[len("biblio_"):])
        if nom:
            importlib.import_module(nom)
    else:
        registre.module(cle)
importer_cle`;

// SONDE : les modules api_web chargés et numpy (contrôle du préchauffage, tests/test_moteur_partage.py)
const MODULES = `
import sys, json
json.dumps({"modules": sorted(m for m in sys.modules if m.startswith("api_web.")), "numpy": "numpy" in sys.modules})`;

let py = null, traiter = null, guet = null, importerCle = null, numpy = null, numpyCharge = false, manifeste = null;
let demarrage = true;          // jusqu'au « pret » : une intention numpy démarre numpy aussitôt (en parallèle du paquet)
const intentions = new Set();  // les clés annoncées pendant le démarrage
let morte = false;             // une erreur fatale : plus rien n'est traité (la page arrête ce travailleur)

// un manifeste sans la liste (plus ancien) : numpy pour tous, comme avant
function numpyPour(cle) {
  const man = manifeste;
  return !!man && (!Array.isArray(man.avec_numpy) || man.avec_numpy.includes(cle));
}
function chargerNumpy() {
  if (!numpy) {
    marquer("numpy_debut");
    numpy = finir("numpy_fin", py.loadPackage("numpy", { messageCallback: () => {} })).then(() => {
      numpyCharge = true;
      if (guet) guet.charge = true;
    }, (e) => { numpy = null; throw e; });   // un échec (réseau) n'est pas retenu : la demande suivante réessaie
  }
  return numpy;
}
const compterModules = () => py.runPython("import sys\nlen(sys.modules)");

// une intention (requête ou préchauffage) pendant le démarrage : numpy part dès que Python est là, sans attendre le paquet
function noterIntention(cle) {
  if (!demarrage) return;
  intentions.add(cle);
  if (py && numpyPour(cle)) chargerNumpy();
}

const pret = (async () => {
  etape("Python démarre dans le navigateur…");
  // le manifeste et le paquet arrivent pendant que Python démarre
  marquer("manifeste_debut");
  const man = manifeste = await (await fetch(new URL("moteur/manifeste.json", base), { cache: "no-cache" })).json();
  marquer("manifeste_fin");
  const zipP = finir("paquet_recu", fetch(new URL(`moteur/moteur.zip?v=${man.sha256.slice(0, 16)}`, base)).then((r) => r.arrayBuffer()));
  marquer("loadPyodide_debut");
  py = await loadPyodide({ indexURL: new URL("vendor/pyodide-0.29.5/", base).href });
  marquer("loadPyodide_fin");
  // les clés déjà annoncées (la page qui s'ouvre, un préchauffage) : numpy en parallèle du paquet si l'une s'en sert
  if ([...intentions].some(numpyPour)) {
    etape("Chargement de numpy…");
    chargerNumpy();
  }
  etape("Chargement du moteur du bureau…");
  const zip = await zipP;
  marquer("empreinte_debut");
  const h = hex(await crypto.subtle.digest("SHA-256", zip));
  marquer("empreinte_fin");
  if (h !== man.sha256) throw new Error(`Le paquet du moteur ne correspond pas à son manifeste (${h.slice(0, 12)} ≠ ${man.sha256.slice(0, 12)})`);
  py.unpackArchive(zip, "zip", { extractDir: "/moteur" });
  marquer("deballage_fin");
  py.runPython("import sys\nsys.path.insert(0, '/moteur')");
  guet = py.runPython(GUET);
  if (numpy) await numpy;
  guet.charge = numpyCharge;
  marquer("numpy_attendu");
  if (SONDE) details.modules_avant_routeur = compterModules();
  traiter = py.pyimport("api_web.routeur").traiter;
  importerCle = py.runPython(IMPORTER);
  marquer("routeur_importe");
  if (SONDE) details.modules_apres_routeur = compterModules();
  etape("Premier calcul…");
  marquer("pret");
  demarrage = false;
  postMessage({ type: "pret" });
})();
pret.catch((e) => { demarrage = false; postMessage({ type: "erreur", texte: String(e && e.message || e) }); });

// ------------------------------------------------------------------ l'erreur fatale
// Pyodide « fatally failed » : l'erreur levée porte pyodide_fatal_error, et son API publique ne sert plus (ses
// accesseurs lèvent « Pyodide already fatally failed… »). Un appel à une fonction Python peut même revenir sans rien :
// on regarde donc l'état de Pyodide après chaque traitement, pas seulement l'erreur.
const pyodideTombe = () => { try { void py.runPython; return false; } catch (e) { return true; } };
const estFatale = (e) => !!(e && e.pyodide_fatal_error) || /fatal/i.test(String(e && e.message || e)) || (!!py && pyodideTombe());
function mourir(e) {
  morte = true;
  postMessage({ type: "mort", texte: String(e && e.message || e || "erreur fatale de Python") });
}

// ------------------------------------------------------------------ la clé d'une requête (dans la requête, seule source)
function cleDe(requete) {
  try {
    const q = JSON.parse(requete);
    if (q && typeof q.module === "string" && q.module) return q.module;
    if (q && q.action === "consulter" && typeof q.bibliotheque === "string" && q.bibliotheque) return "biblio_" + q.bibliotheque;
  } catch (e) { /* une requête illisible : le routeur dira son erreur */ }
  return "";
}

// ------------------------------------------------------------------ la SONDE (par clé)
const vues = new Set();   // SONDE : les clés déjà datées (première requête de chaque clé)
let session = true;       // SONDE : la toute première requête de la session (les 10 « calculer » de mesure)
const MARQUES_REQUETE = ["numpy_import_debut", "numpy_import_fin", "module_import_debut", "module_import_fin", "requete1_debut",
  "requete1_traitee", "requete1_rejouee", "reponse1_postee"];
const DETAILS_DEMARRAGE = ["modules_avant_routeur", "modules_apres_routeur"];

// l'import du module de cette clé (et celui de numpy s'il est chargé), daté À PART du premier calcul. C'est le même
// travail au même moment — le routeur l'aurait fait au début de cette requête — : la réponse n'attend pas plus. Une
// erreur d'import est notée, et la requête suit son chemin ordinaire (le guet de numpy compris).
function sonderImport(cle) {
  for (const k of Object.keys(details)) if (!DETAILS_DEMARRAGE.includes(k)) delete details[k];
  effacerMarques(MARQUES_REQUETE);
  if (!CLE.test(cle)) return;
  if (numpyCharge) { marquer("numpy_import_debut"); py.runPython("import numpy"); marquer("numpy_import_fin"); }
  marquer("module_import_debut");
  try { importerCle(cle); } catch (e) { details.import_erreur = String(e.message || e).slice(-300); }
  marquer("module_import_fin");
  details.modules_apres_module = compterModules();
}

// après la première réponse d'une clé (elle est déjà partie : rien de ceci ne la retarde), ce que Python a chargé ; à la
// toute première requête de la session seulement, le coût de la relecture du registre et celui d'une requête ordinaire
// (« calculer » sur l'état rendu, dix fois, la médiane) ; puis tout part à la page
function envoyerSonde(json, cle, premiere) {
  details.reponse1_octets = json.length;
  try {
    details.modules_apres_requete1 = compterModules();
    details.numpy_importe = py.runPython("import sys\n'numpy' in sys.modules");
    if (premiere) {
      details.cles_ms = py.runPython("import time\nfrom api_web import registre\n_t = time.perf_counter()\nfor _i in range(5): registre.cles()\n(time.perf_counter() - _t) / 5 * 1000");
      const etat = JSON.parse(json).etat;
      if (etat) {
        const req = JSON.stringify({ module: cle, action: "calculer", etat });
        const ts = [];
        for (let i = 0; i < 10; i++) { const t = performance.now(); traiter(req); ts.push(performance.now() - t); }
        ts.sort((a, b) => a - b);
        details.calculer_ms = +((ts[4] + ts[5]) / 2).toFixed(1);
        details.calculer_min_ms = +ts[0].toFixed(1);
      }
    }
  } catch (e) { details.sonde_erreur = String(e.message || e).slice(-300); }
  const T0 = performance.timeOrigin;
  postMessage({
    type: "sonde", cle, premiere, marques: marquesAbsolues(), details: { ...details },
    ressources: performance.getEntriesByType("resource").map((r) => ({ nom: r.name.replace(/^.*\//, "").replace(/\?.*$/, ""),
      debut: +(T0 + r.startTime).toFixed(1), reponse: +(T0 + r.responseStart).toFixed(1), fin: +(T0 + r.responseEnd).toFixed(1),
      transfert: r.transferSize, corps: r.encodedBodySize })),
  });
}

// ------------------------------------------------------------------ la NOTE PDF
// Son paquet (les générateurs des écrans Streamlit, reportlab, les polices — moteur/note.zip, quelques Mo) n'est chargé
// qu'à la PREMIÈRE demande de note (la page le dit : « note » dans le message, js/moteur.js), vérifié contre son
// empreinte, puis déballé à côté du moteur (le paquet du moteur porte déjà ce qu'il ne répète pas). Si le relevé a vu
// la note de CETTE clé charger numpy (manifeste « note.avec_numpy »), numpy est chargé AVANT la note — sans quoi le
// guet la ferait refaire une fois. Les notes suivantes sont immédiates.
let zipNote = null, noteDeballee = false;
async function chargerNote(cle) {
  const m = manifeste && manifeste.note;
  if (!m) throw new Error("le paquet de la note PDF manque à ce site (manifeste sans « note »)");
  if (!zipNote) {
    zipNote = (async () => {
      const zip = await (await fetch(new URL(`moteur/note.zip?v=${m.sha256.slice(0, 16)}`, base))).arrayBuffer();
      const h = hex(await crypto.subtle.digest("SHA-256", zip));
      if (h !== m.sha256) throw new Error(`Le paquet de la note ne correspond pas à son manifeste (${h.slice(0, 12)} ≠ ${m.sha256.slice(0, 12)})`);
      return zip;
    })();
    // un échec (réseau coupé) n'est pas retenu : la demande suivante réessaie
    zipNote.catch(() => { zipNote = null; });
  }
  if (Array.isArray(m.avec_numpy) && m.avec_numpy.includes(cle)) await chargerNumpy();
  const zip = await zipNote;
  if (!noteDeballee) {
    py.unpackArchive(zip, "zip", { extractDir: "/moteur" });
    py.runPython("import importlib\nimportlib.invalidate_caches()");
    noteDeballee = true;
  }
}

// ------------------------------------------------------------------ les traitements (un à la fois : la chaîne)
async function requete(data, cle) {
  if (morte) return;
  const premiere = SONDE && !vues.has(cle);
  const debutSession = premiere && session;
  if (premiere) { vues.add(cle); session = false; }
  let json;
  try {
    await pret;
    if (numpyPour(cle)) await chargerNumpy();
    if (premiere) { sonderImport(cle); marquer("requete1_debut"); }
    if (data.note) await chargerNote(cle);
    json = traiter(data.requete);
    if (premiere) marquer("requete1_traitee");
    if (!numpyCharge && guet.demande) {
      // numpy était demandé : on le charge, puis la même requête repart (elle ne dépend que de ce qu'elle porte)
      await chargerNumpy();
      guet.demande = false;
      json = traiter(data.requete);
      if (premiere) { marquer("requete1_rejouee"); details.rejouee = true; }
    }
    if (typeof json !== "string") throw new Error("le moteur n'a rien rendu");
    if (pyodideTombe()) throw new Error("Pyodide fatally failed");
  } catch (e) {
    if (estFatale(e)) { mourir(e); return; }
    json = JSON.stringify({ ok: false, erreur: String(e && e.message || e) });
  }
  if (premiere) marquer("reponse1_postee");
  postMessage({ type: "reponse", id: data.id, json });
  if (premiere) envoyerSonde(json, cle, debutSession);
}

let nRecus = 0, dernierPrechauffage = 0, derniereRequete = 0;
// l'import d'une clé, avec le guet (un import de numpy imprévu : numpy chargé, et l'import refait) ; une erreur d'import
// est avalée (la vraie requête la dira), une erreur fatale remonte
async function importerAvecGuet(cle) {
  if (numpyPour(cle)) await chargerNumpy();
  try { importerCle(cle); } catch (e) { if (estFatale(e)) throw e; /* avalée : la vraie requête dira l'erreur */ }
  if (!numpyCharge && guet.demande) {
    await chargerNumpy();
    guet.demande = false;
    try { importerCle(cle); } catch (e) { if (estFatale(e)) throw e; }
  }
  if (pyodideTombe()) throw new Error("Pyodide fatally failed");
}

async function prechauffer(cle, n) {
  if (morte) return;
  const t0 = performance.now();
  let saute = !CLE.test(cle);
  if (!saute) {
    try {
      await pret;
      // sauté : un préchauffage plus récent, ou une vraie requête arrivée derrière lui — regardé APRÈS le démarrage
      // (ce qui est arrivé pendant compte)
      saute = n !== dernierPrechauffage || derniereRequete > n;
      if (!saute) await importerAvecGuet(cle);
    } catch (e) {
      if (estFatale(e)) { mourir(e); return; }
      // le démarrage a échoué, ou numpy n'a pas pu se charger : la vraie requête dira l'erreur
    }
  }
  if (SONDE) postMessage({ type: "prechauffe", cle, ms: +(performance.now() - t0).toFixed(1), saute, numpy: numpyCharge });
}

function sondeModules() {
  if (morte) return;
  let lu;
  try {
    lu = JSON.parse(py.runPython(MODULES));
    lu.guet = !!(guet && guet.demande);
    lu.charge = numpyCharge;
  } catch (e) { lu = { erreur: String(e && e.message || e) }; }
  postMessage({ type: "sonde_modules", ...lu });
}

let fileT = Promise.resolve();
self.onmessage = ({ data }) => {
  const n = ++nRecus;
  let tache = null;
  if (data && data.type === "prechauffer") {
    const cle = String(data.cle || "");
    dernierPrechauffage = n;
    noterIntention(cle);
    tache = () => prechauffer(cle, n);
  } else if (data && data.type === "sonde_modules") {
    if (SONDE) tache = () => pret.then(sondeModules, sondeModules);
  } else if (data && data.id !== undefined) {
    derniereRequete = n;
    const cle = cleDe(data.requete);
    noterIntention(cle);
    tache = () => requete(data, cle);
  }
  if (tache) fileT = fileT.then(tache).catch(() => {});
};
