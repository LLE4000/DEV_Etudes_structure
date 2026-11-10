// Le travailleur du navigateur : Python (Pyodide) démarre ici, hors de la page, pour qu'elle reste fluide. Il charge le
// PAQUET DU MOTEUR (moteur/moteur.zip, dont l'empreinte est vérifiée contre moteur/manifeste.json), puis répond aux
// requêtes des écrans par api_web.routeur.traiter (JSON → JSON). Aucun calcul n'est écrit en JavaScript.
//
// UN travailleur pour TOUTE la session du document (lot S4, REFONTE_ETABLI.md, S4-D5 et C5) : il sert tous les écrans
// qui se succèdent (js/moteur.js, moteurPartage). Il est créé SANS module : la CLÉ d'une requête se lit DANS la requête
// (« module », sinon « biblio_<bibliothèque> » pour l'action « consulter », sinon rien) — une seule source.
//
// numpy pèse une à deux secondes : on ne le charge que pour les clés que le relevé du paquet a vues s'en servir
// (manifeste « avec_numpy », écrit par construire.py), AVANT la requête d'une telle clé. Une intention RÉELLE arrivée
// pendant le démarrage — l'écran qui se monte annonce sa clé ({type:"annoncer", cle}, js/moteur.js, annoncer), ou sa
// requête est déjà là — fait partir numpy juste après loadPyodide, en parallèle du paquet, et « pret » l'attend (comme
// l'ancien « ?module= » : cet écran ne calcule rien sans lui). Un PRÉCHAUFFAGE, lui, ne le fait jamais partir pendant
// le démarrage (C4 : il ne retarde jamais « pret »). Un import de numpy imprévu (un chemin rare, un module plus récent
// que le manifeste) est GUETTÉ en Python : la requête est alors rejouée, numpy chargé — même si le moteur a rattrapé
// l'erreur d'import en silence ; la demande du guet est remise à faux AVANT ce chargement : s'il échoue (hors ligne),
// la requête qui l'a déclenché garde son refus, et les suivantes, des autres modules, répondent.
//
// TOUS les messages passent par UNE chaîne (fileT) : jamais deux traitements entrelacés (note, guet, import, calcul).
// UNE exception, et elle seule (C5) : le CHARGEMENT de numpy d'une intention de fond — un préchauffage, ou une annonce
// après le démarrage — court HORS de la chaîne (loadPackage télécharge et installe sans retenir les requêtes qui n'en
// ont pas besoin). Pendant qu'il court, le guet REFUSE net tout import de numpy (rien d'un numpy à demi installé
// n'entre dans sys.modules) et la requête qui l'a tenté est rejouée quand il est là ; une requête d'une clé qui s'en sert
// attend la même promesse ; l'import du module préchauffé, lui, repasse par la chaîne.
// Le PRÉCHAUFFAGE ({type:"prechauffer", cle}) est une tâche de FOND : il importe le module de cette clé, et lui seul ;
// il est sauté si un préchauffage plus récent ou une vraie requête est arrivé derrière lui (la requête importera
// d'elle-même ce qu'il lui faut) — regardé après « pret », et de nouveau après numpy, avant l'import ; une erreur
// d'import est avalée (la vraie requête la dira).
//
// La VERSION du site : js/moteur.js passe celle du document dans l'adresse du travailleur (« v », la balise
// stab-version). Un manifeste d'une autre version — une mise en ligne entre-temps, ou un travailleur recréé après une
// mort sous des scripts anciens — n'est jamais servi : le travailleur n'annonce pas « pret » mais l'erreur nommée
// MIS_A_JOUR (« perime » : la page ne relance pas ce moteur, seul un rechargement répare). Une empreinte de paquet (le
// moteur, la note PDF) qui ne correspond plus : le manifeste est relu, et s'il a changé de version, la même erreur
// nommée remplace le message d'empreinte. Sans « v » (les essais dans node), rien n'est comparé.
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
// la version du site du document qui a créé ce travailleur (vide : rien n'est comparé)
const VERSION = new URL(self.location.href).searchParams.get("v") || "";
const MIS_A_JOUR = "Le site a été mis à jour : rechargez la page";
const perime = () => Object.assign(new Error(MIS_A_JOUR), { perime: true });
const details = {};   // SONDE : ce que Python compte (modules chargés, octets, coût d'une requête ordinaire)
const etape = (texte) => postMessage({ type: "etape", texte });
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");

// le guet : un chercheur d'import placé en tête, qui ne trouve rien lui-même mais note qu'on a demandé numpy — tant que
// numpy n'est pas chargé (« charge », posé par chargerNumpy) : après, un import de numpy n'a plus rien d'imprévu.
// Pendant un chargement (« en_cours », posé par chargerNumpy : il peut courir hors de la chaîne, C5), il refuse net
// l'import (jamais un numpy à demi installé) : la requête est rejouée quand numpy est là, comme sans lui.
const GUET = `
import sys
class GuetNumpy:
    """Note qu'un import de numpy a été tenté (le travailleur le charge alors et rejoue la requête) ; pendant que numpy
    se charge, refuse cet import (rien d'un numpy à demi installé n'entre dans sys.modules)."""
    demande = False
    charge = False
    en_cours = False
    def find_spec(self, nom, chemin=None, cible=None):
        if not GuetNumpy.charge and (nom == "numpy" or nom.startswith("numpy.")):
            GuetNumpy.demande = True
            if GuetNumpy.en_cours:
                raise ModuleNotFoundError(f"numpy se charge encore ({nom}) : la requête sera rejouée", name=nom)
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
let demarrage = true;          // jusqu'au « pret » : une intention RÉELLE démarre numpy aussitôt (en parallèle du paquet)
const intentions = new Set();  // les clés d'une intention RÉELLE (annonce, requête) arrivées pendant le démarrage
let morte = false;             // une erreur fatale : plus rien n'est traité (la page arrête ce travailleur)

// un manifeste sans la liste (plus ancien) : numpy pour tous, comme avant
function numpyPour(cle) {
  const man = manifeste;
  return !!man && (!Array.isArray(man.avec_numpy) || man.avec_numpy.includes(cle));
}
function chargerNumpy() {
  if (!numpy) {
    marquer("numpy_debut");
    if (guet) guet.en_cours = true;
    numpy = finir("numpy_fin", py.loadPackage("numpy", { messageCallback: () => {} })).then(() => {
      numpyCharge = true;
      if (guet) { guet.charge = true; guet.en_cours = false; }
    }, (e) => { numpy = null; if (guet) guet.en_cours = false; throw e; });   // un échec (réseau) n'est pas retenu : la demande suivante réessaie
  }
  return numpy;
}
// numpy en FOND (hors de la chaîne, C5) : un échec n'est dit que par la requête qui en a besoin
const numpyEnFond = () => { chargerNumpy().catch(() => {}); };
const compterModules = () => py.runPython("import sys\nlen(sys.modules)");

// une intention RÉELLE (l'écran qui se monte annonce sa clé, ou sa requête est arrivée) : pendant le démarrage, numpy
// part dès que Python est là, sans attendre le paquet, et « pret » l'attendra ; une ANNONCE après le démarrage le fait
// partir en fond (la requête de cette clé l'attendra d'elle-même). Jamais un import : aucun module n'est chargé ici.
function noterIntention(cle, annonce = false) {
  if (demarrage) {
    intentions.add(cle);
    if (py && numpyPour(cle)) numpyEnFond();
  } else if (annonce && numpyPour(cle) && !numpyCharge) numpyEnFond();
}

// une empreinte qui ne correspond plus : le manifeste relu (sans cache) dit si le site a été mis à jour entre-temps —
// alors l'erreur nommée, sinon le message d'empreinte
async function empreinteFausse(message) {
  let m = null;
  try { m = await (await fetch(new URL("moteur/manifeste.json", base), { cache: "no-cache" })).json(); } catch (e) { /* hors ligne : le message d'empreinte */ }
  return m && manifeste && m.version !== manifeste.version ? perime() : new Error(message);
}

const pret = (async () => {
  etape("Python démarre dans le navigateur…");
  // le manifeste et le paquet arrivent pendant que Python démarre
  marquer("manifeste_debut");
  const man = manifeste = await (await fetch(new URL("moteur/manifeste.json", base), { cache: "no-cache" })).json();
  marquer("manifeste_fin");
  // un autre site que celui du document (une mise en ligne entre-temps) : jamais servi, rien d'autre n'est chargé
  if (VERSION && man.version !== VERSION) throw perime();
  const zipP = finir("paquet_recu", fetch(new URL(`moteur/moteur.zip?v=${man.sha256.slice(0, 16)}`, base)).then((r) => r.arrayBuffer()));
  marquer("loadPyodide_debut");
  py = await loadPyodide({ indexURL: new URL("vendor/pyodide-0.29.5/", base).href });
  marquer("loadPyodide_fin");
  // les clés d'une intention réelle déjà là (l'écran qui s'ouvre) : numpy en parallèle du paquet si l'une s'en sert
  const numpyVoulu = () => [...intentions].some(numpyPour);
  if (numpyVoulu()) {
    etape("Chargement de numpy…");
    numpyEnFond();
  }
  etape("Chargement du moteur du bureau…");
  const zip = await zipP;
  marquer("empreinte_debut");
  const h = hex(await crypto.subtle.digest("SHA-256", zip));
  marquer("empreinte_fin");
  if (h !== man.sha256) throw await empreinteFausse(`Le paquet du moteur ne correspond pas à son manifeste (${h.slice(0, 12)} ≠ ${man.sha256.slice(0, 12)})`);
  py.unpackArchive(zip, "zip", { extractDir: "/moteur" });
  marquer("deballage_fin");
  py.runPython("import sys\nsys.path.insert(0, '/moteur')");
  guet = py.runPython(GUET);
  guet.en_cours = !!numpy && !numpyCharge;
  // « pret » n'attend numpy que pour une intention RÉELLE (l'écran qui s'ouvre ne calcule rien sans lui, comme avant
  // S4) ; jamais pour un préchauffage (C4). Un échec n'arrête pas le moteur : la requête de cette clé le dira.
  if (numpy && numpyVoulu()) await numpy.catch(() => {});
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
pret.catch((e) => { demarrage = false; postMessage({ type: "erreur", texte: String(e && e.message || e), ...(e && e.perime ? { perime: true } : {}) }); });

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
      // (une mise en ligne depuis le démarrage : le serveur sert les octets nouveaux — le manifeste relu le dit)
      if (h !== m.sha256) throw await empreinteFausse(`Le paquet de la note ne correspond pas à son manifeste (${h.slice(0, 12)} ≠ ${m.sha256.slice(0, 12)})`);
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
      // numpy était demandé : on le charge, puis la même requête repart (elle ne dépend que de ce qu'elle porte). La
      // demande est remise à faux AVANT : si le chargement échoue (hors ligne), CETTE requête garde son refus, et les
      // suivantes des autres modules répondent (une qui importe encore numpy relève le guet, et reste refusée)
      guet.demande = false;
      await chargerNumpy();
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
let fileT = Promise.resolve();
const enchainer = (tache) => { fileT = fileT.then(tache).catch(() => {}); };

// sauté : un préchauffage plus récent, ou une vraie requête arrivée derrière lui
const estSaute = (n) => n !== dernierPrechauffage || derniereRequete > n;

// Le préchauffage d'une clé, dans la chaîne, sans JAMAIS la retenir (C4) : si numpy manque — la clé s'en sert, ou le
// guet l'a vu demander —, il se charge EN FOND, hors de la chaîne (l'exception de C5), et le préchauffage reprend une
// place dans la chaîne quand il est là : « sauté » est regardé de nouveau, AVANT l'import. Une erreur d'import est
// avalée (la vraie requête la dira), une erreur fatale tue le travailleur.
async function prechauffer(cle, n, t0 = performance.now()) {
  if (morte) return;
  let saute = !CLE.test(cle), attend = false;
  if (!saute) {
    try {
      await pret;
      // regardé APRÈS le démarrage (ce qui est arrivé pendant compte), et après numpy à la reprise
      saute = estSaute(n);
      if (!saute) {
        if (numpyPour(cle) && !numpyCharge) attend = true;
        else {
          try { importerCle(cle); } catch (e) { if (estFatale(e)) throw e; /* avalée : la vraie requête dira l'erreur */ }
          if (pyodideTombe()) throw new Error("Pyodide fatally failed");
          if (!numpyCharge && guet.demande) { guet.demande = false; attend = true; }
        }
      }
    } catch (e) {
      if (estFatale(e)) { mourir(e); return; }
      // le démarrage a échoué : la vraie requête dira l'erreur
    }
  }
  const fin = (o) => { if (SONDE) postMessage({ type: "prechauffe", cle, ms: +(performance.now() - t0).toFixed(1), numpy: numpyCharge, ...o }); };
  if (!attend) return fin({ saute });
  // numpy en fond ; la suite repasse par la chaîne (un échec de numpy : rien d'importé, la vraie requête le dira)
  chargerNumpy().then(() => enchainer(() => prechauffer(cle, n, t0)), () => fin({ saute: false, erreur: "numpy n'a pas pu se charger" }));
}

function sondeModules() {
  if (morte) return;
  let lu;
  try {
    lu = JSON.parse(py.runPython(MODULES));
    lu.guet = !!(guet && guet.demande);
    lu.charge = numpyCharge;
    lu.en_cours = !!(guet && guet.en_cours);
  } catch (e) { lu = { erreur: String(e && e.message || e) }; }
  postMessage({ type: "sonde_modules", ...lu });
}

self.onmessage = ({ data }) => {
  const n = ++nRecus;
  let tache = null;
  if (data && data.type === "prechauffer") {
    // (pas une intention réelle : il ne fait partir numpy qu'après « pret », en fond — il ne retarde jamais « pret »)
    const cle = String(data.cle || "");
    dernierPrechauffage = n;
    tache = () => prechauffer(cle, n);
  } else if (data && data.type === "annoncer") {
    // l'écran qui se monte calculera cette clé : numpy (s'il s'en sert), rien d'autre, hors de la chaîne
    const cle = String(data.cle || "");
    if (CLE.test(cle)) noterIntention(cle, true);
  } else if (data && data.type === "sonde_modules") {
    if (SONDE) tache = () => pret.then(sondeModules, sondeModules);
  } else if (data && data.id !== undefined) {
    derniereRequete = n;
    const cle = cleDe(data.requete);
    noterIntention(cle);
    tache = () => requete(data, cle);
  }
  if (tache) enchainer(tache);
};
