// Le travailleur du navigateur : Python (Pyodide) démarre ici, hors de la page, pour qu'elle reste fluide. Il charge le
// PAQUET DU MOTEUR (moteur/moteur.zip, dont l'empreinte est vérifiée contre moteur/manifeste.json), numpy si le module
// de la page s'en sert, puis répond aux requêtes de l'écran par api_web.routeur.traiter (JSON → JSON). Aucun calcul
// n'est écrit en JavaScript.
//
// numpy pèse une à deux secondes au démarrage : on ne le charge d'emblée que pour les modules que le relevé du paquet
// a vus s'en servir (manifeste « avec_numpy », écrit par construire.py). Un import de numpy imprévu (un chemin rare,
// un module plus récent que le manifeste) est GUETTÉ en Python : la requête est alors rejouée, numpy chargé — même si
// le moteur a rattrapé l'erreur d'import en silence.
//
// La SONDE (js/sonde.js, « sonde=1 » dans l'adresse du travailleur, posé par js/moteur.js quand la page porte
// « ?sonde=1 ») date chaque étape, sépare l'import du module de son premier calcul et renvoie ses mesures à la page
// (message « sonde ») après la première réponse. Sans elle, ce fichier fait exactement ce qu'il faisait.
import { loadPyodide } from "../vendor/pyodide-0.29.5/pyodide.mjs";
import { SONDE, marquer, finir, marquesAbsolues } from "./sonde.js";

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
const MODULE = new URL(self.location.href).searchParams.get("module") || "";
const details = {};   // SONDE : ce que Python compte (modules chargés, octets, coût d'une requête ordinaire)
const etape = (texte) => postMessage({ type: "etape", texte });
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");

// le guet : un chercheur d'import placé en tête, qui ne trouve rien lui-même mais note qu'on a demandé numpy
const GUET = `
import sys
class GuetNumpy:
    """Note qu'un import de numpy a été tenté (le travailleur le charge alors et rejoue la requête)."""
    demande = False
    def find_spec(self, nom, chemin=None, cible=None):
        if nom == "numpy" or nom.startswith("numpy."):
            GuetNumpy.demande = True
        return None
sys.meta_path.insert(0, GuetNumpy())
GuetNumpy`;

let py = null, traiter = null, guet = null, numpy = null;
const chargerNumpy = () => (numpy = numpy || (marquer("numpy_debut"), finir("numpy_fin", py.loadPackage("numpy", { messageCallback: () => {} }))));
const compterModules = () => py.runPython("import sys\nlen(sys.modules)");

const pret = (async () => {
  etape("Python démarre dans le navigateur…");
  // le manifeste et le paquet arrivent pendant que Python démarre
  marquer("manifeste_debut");
  const man = await (await fetch(new URL("moteur/manifeste.json", base), { cache: "no-cache" })).json();
  marquer("manifeste_fin");
  const zipP = finir("paquet_recu", fetch(new URL(`moteur/moteur.zip?v=${man.sha256.slice(0, 16)}`, base)).then((r) => r.arrayBuffer()));
  marquer("loadPyodide_debut");
  py = await loadPyodide({ indexURL: new URL("vendor/pyodide-0.29.5/", base).href });
  marquer("loadPyodide_fin");
  // un manifeste sans la liste (plus ancien) : numpy pour tous, comme avant
  const avecNumpy = !Array.isArray(man.avec_numpy) || man.avec_numpy.includes(MODULE);
  if (avecNumpy) {
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
  marquer("numpy_attendu");
  if (SONDE) details.modules_avant_routeur = compterModules();
  traiter = py.pyimport("api_web.routeur").traiter;
  marquer("routeur_importe");
  if (SONDE) details.modules_apres_routeur = compterModules();
  etape("Premier calcul…");
  marquer("pret");
  postMessage({ type: "pret" });
})();
pret.catch((e) => postMessage({ type: "erreur", texte: String(e && e.message || e) }));

// SONDE : l'import du module de la page (et celui de numpy s'il est chargé), datés À PART du premier calcul. C'est le
// même travail au même moment — le routeur l'aurait fait au début de cette requête — : le premier rendu n'attend pas
// plus. Une erreur d'import est notée, et la requête suit son chemin ordinaire (le guet de numpy compris).
function sonderImport() {
  if (!/^[a-z][a-z0-9_]*$/.test(MODULE)) return;
  if (numpy) { marquer("numpy_import_debut"); py.runPython("import numpy"); marquer("numpy_import_fin"); }
  marquer("module_import_debut");
  try {
    py.runPython(`import importlib\nimportlib.import_module("api_web.${MODULE}")`);
  } catch (e) { details.import_erreur = String(e.message || e).slice(-300); }
  marquer("module_import_fin");
  details.modules_apres_module = compterModules();
}

// SONDE : après la première réponse (elle est déjà partie : rien de ceci ne retarde le premier rendu), le coût de la
// relecture du registre (registre.cles(), refaite à chaque requête) et celui d'une requête ordinaire (« calculer » sur
// l'état rendu, dix fois, la médiane), puis tout part à la page.
function envoyerSonde(json) {
  details.reponse1_octets = json.length;
  try {
    details.modules_apres_requete1 = compterModules();
    details.numpy_importe = py.runPython("import sys\n'numpy' in sys.modules");
    details.cles_ms = py.runPython("import time\nfrom api_web import registre\n_t = time.perf_counter()\nfor _i in range(5): registre.cles()\n(time.perf_counter() - _t) / 5 * 1000");
    const etat = JSON.parse(json).etat;
    const req = JSON.stringify({ module: MODULE, action: "calculer", etat });
    const ts = [];
    for (let i = 0; i < 10; i++) { const t = performance.now(); traiter(req); ts.push(performance.now() - t); }
    ts.sort((a, b) => a - b);
    details.calculer_ms = +((ts[4] + ts[5]) / 2).toFixed(1);
    details.calculer_min_ms = +ts[0].toFixed(1);
  } catch (e) { details.sonde_erreur = String(e.message || e).slice(-300); }
  const T0 = performance.timeOrigin;
  postMessage({
    type: "sonde", marques: marquesAbsolues(), details,
    ressources: performance.getEntriesByType("resource").map((r) => ({ nom: r.name.replace(/^.*\//, "").replace(/\?.*$/, ""),
      debut: +(T0 + r.startTime).toFixed(1), reponse: +(T0 + r.responseStart).toFixed(1), fin: +(T0 + r.responseEnd).toFixed(1),
      transfert: r.transferSize, corps: r.encodedBodySize })),
  });
}

let nRequetes = 0;   // SONDE seulement : la première requête est datée
onmessage = async ({ data }) => {
  let json;
  const premiere = SONDE && ++nRequetes === 1;
  try {
    await pret;
    if (premiere) sonderImport();
    if (premiere) marquer("requete1_debut");
    json = traiter(data.requete);
    if (premiere) marquer("requete1_traitee");
    if (!numpy && guet.demande) {
      // numpy était demandé : on le charge, puis la même requête repart (elle ne dépend que de ce qu'elle porte)
      await chargerNumpy();
      guet.demande = false;
      json = traiter(data.requete);
      if (premiere) marquer("requete1_rejouee");
    }
  } catch (e) {
    json = JSON.stringify({ ok: false, erreur: String(e && e.message || e) });
  }
  if (premiere) marquer("reponse1_postee");
  postMessage({ type: "reponse", id: data.id, json });
  if (premiere) envoyerSonde(json);
};
