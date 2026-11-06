// Le travailleur du navigateur : Python (Pyodide) démarre ici, hors de la page, pour qu'elle reste fluide. Il charge numpy
// et le PAQUET DU MOTEUR (moteur/moteur.zip, dont l'empreinte est vérifiée contre moteur/manifeste.json), puis répond
// aux requêtes de l'écran par api_web.routeur.traiter (JSON → JSON). Aucun calcul n'est écrit en JavaScript.
import { loadPyodide } from "../vendor/pyodide-0.29.5/pyodide.mjs";

const base = new URL("../", import.meta.url);
const etape = (texte) => postMessage({ type: "etape", texte });
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");

let traiter = null;
const pret = (async () => {
  etape("Python démarre dans le navigateur…");
  const py = await loadPyodide({ indexURL: new URL("vendor/pyodide-0.29.5/", base).href });
  etape("Chargement de numpy…");
  await py.loadPackage("numpy", { messageCallback: () => {} });
  etape("Chargement du moteur du bureau…");
  const man = await (await fetch(new URL("moteur/manifeste.json", base), { cache: "no-cache" })).json();
  const zip = await (await fetch(new URL(`moteur/moteur.zip?v=${man.sha256.slice(0, 16)}`, base))).arrayBuffer();
  const h = hex(await crypto.subtle.digest("SHA-256", zip));
  if (h !== man.sha256) throw new Error(`Le paquet du moteur ne correspond pas à son manifeste (${h.slice(0, 12)} ≠ ${man.sha256.slice(0, 12)})`);
  py.unpackArchive(zip, "zip", { extractDir: "/moteur" });
  py.runPython("import sys\nsys.path.insert(0, '/moteur')");
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
  } catch (e) {
    json = JSON.stringify({ ok: false, erreur: String(e && e.message || e) });
  }
  postMessage({ type: "reponse", id: data.id, json });
};
