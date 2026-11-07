// Le travailleur du navigateur : Python (Pyodide) démarre ici, hors de la page, pour qu'elle reste fluide. Il charge le
// PAQUET DU MOTEUR (moteur/moteur.zip, dont l'empreinte est vérifiée contre moteur/manifeste.json), numpy si le module
// de la page s'en sert, puis répond aux requêtes de l'écran par api_web.routeur.traiter (JSON → JSON). Aucun calcul
// n'est écrit en JavaScript.
//
// numpy pèse une à deux secondes au démarrage : on ne le charge d'emblée que pour les modules que le relevé du paquet
// a vus s'en servir (manifeste « avec_numpy », écrit par construire.py). Un import de numpy imprévu (un chemin rare,
// un module plus récent que le manifeste) est GUETTÉ en Python : la requête est alors rejouée, numpy chargé — même si
// le moteur a rattrapé l'erreur d'import en silence.
import { loadPyodide } from "../vendor/pyodide-0.29.5/pyodide.mjs";

const base = new URL("../", import.meta.url);
const MODULE = new URL(self.location.href).searchParams.get("module") || "";
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
const chargerNumpy = () => (numpy = numpy || py.loadPackage("numpy", { messageCallback: () => {} }));

const pret = (async () => {
  etape("Python démarre dans le navigateur…");
  // le manifeste et le paquet arrivent pendant que Python démarre
  const man = await (await fetch(new URL("moteur/manifeste.json", base), { cache: "no-cache" })).json();
  const zipP = fetch(new URL(`moteur/moteur.zip?v=${man.sha256.slice(0, 16)}`, base)).then((r) => r.arrayBuffer());
  py = await loadPyodide({ indexURL: new URL("vendor/pyodide-0.29.5/", base).href });
  // un manifeste sans la liste (plus ancien) : numpy pour tous, comme avant
  const avecNumpy = !Array.isArray(man.avec_numpy) || man.avec_numpy.includes(MODULE);
  if (avecNumpy) {
    etape("Chargement de numpy…");
    chargerNumpy();
  }
  etape("Chargement du moteur du bureau…");
  const zip = await zipP;
  const h = hex(await crypto.subtle.digest("SHA-256", zip));
  if (h !== man.sha256) throw new Error(`Le paquet du moteur ne correspond pas à son manifeste (${h.slice(0, 12)} ≠ ${man.sha256.slice(0, 12)})`);
  py.unpackArchive(zip, "zip", { extractDir: "/moteur" });
  py.runPython("import sys\nsys.path.insert(0, '/moteur')");
  guet = py.runPython(GUET);
  if (numpy) await numpy;
  traiter = py.pyimport("api_web.routeur").traiter;
  etape("Premier calcul…");
  postMessage({ type: "pret" });
})();
pret.catch((e) => postMessage({ type: "erreur", texte: String(e && e.message || e) }));

onmessage = async ({ data }) => {
  let json;
  try {
    await pret;
    json = traiter(data.requete);
    if (!numpy && guet.demande) {
      // numpy était demandé : on le charge, puis la même requête repart (elle ne dépend que de ce qu'elle porte)
      await chargerNumpy();
      guet.demande = false;
      json = traiter(data.requete);
    }
  } catch (e) {
    json = JSON.stringify({ ok: false, erreur: String(e && e.message || e) });
  }
  postMessage({ type: "reponse", id: data.id, json });
};
